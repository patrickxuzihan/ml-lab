"""Phase-5 deep model: a breathing waveform from face-only video (PyTorch).

The only module that imports torch (core modules stay torch-free; requirements-train.txt).

Input: model inputs of scripts/make_training_data.py, (T, 64, 64, 3) uint8 frames at 10 fps.
Each frame is normalized per video and channel, and paired with its difference to the next
frame, so the network can use both color and motion. A small 2D CNN, shared by all frames,
turns each frame into a feature vector; dilated temporal convolutions turn the sequence into
one waveform. The loss is 1 - Pearson correlation with the SCAMPS breathing label, the usual
choice for rPPG-style models: only the shape of the waveform matters, and its rate is
estimated afterwards by the evaluation protocol, like every other measurement.

Training (`train`) keeps a checkpoint of the last epoch and of the best one on validation, so
a run interrupted by a disconnect continues at the next epoch.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from breathing import model_input, pipeline

EPS = 1e-6


def model_input_tensor(frames: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W, 3) uint8 frames -> (B, T, 6, H, W) float: frames standardized per video
    and channel, and frame differences (to the next frame; zero for the last) standardized
    the same way."""
    x = frames.float().permute(0, 1, 4, 2, 3) / 255.0  # (B, T, 3, H, W)
    dims = (1, 3, 4)
    appearance = (x - x.mean(dims, keepdim=True)) / (x.std(dims, keepdim=True) + EPS)
    diff = torch.cat([x[:, 1:] - x[:, :-1], torch.zeros_like(x[:, :1])], dim=1)
    motion = diff / (diff.std(dims, keepdim=True) + EPS)
    return torch.cat([appearance, motion], dim=2)


def _conv2d(c_in: int, c_out: int) -> list[nn.Module]:
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU()]


class BreathingNet(nn.Module):
    """Per-frame 2D CNN (width, 2 width, 4 width channels; 64 -> 8 pixels, then averaged)
    followed by residual dilated 1D convolutions over time and a 1x1 output layer."""

    def __init__(self, width: int = 16, dilations: tuple[int, ...] = (1, 2, 4, 8, 16), kernel=5):
        super().__init__()
        w = width
        self.frame = nn.Sequential(
            *_conv2d(6, w), *_conv2d(w, w), nn.AvgPool2d(2),
            *_conv2d(w, 2 * w), nn.AvgPool2d(2),
            *_conv2d(2 * w, 4 * w), nn.AvgPool2d(2),
            *_conv2d(4 * w, 4 * w), nn.AdaptiveAvgPool2d(1),
        )  # fmt: skip
        c = 4 * w
        self.temporal = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(c, c, kernel, padding=d * (kernel - 1) // 2, dilation=d, bias=False),
                nn.BatchNorm1d(c),
                nn.ReLU(),
            )
            for d in dilations
        )
        self.head = nn.Conv1d(c, 1, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, 3) uint8 -> (B, T) waveform."""
        x = model_input_tensor(frames)
        b, t = x.shape[:2]
        f = self.frame(x.flatten(0, 1)).view(b, t, -1).transpose(1, 2)  # (B, C, T)
        for layer in self.temporal:
            f = f + layer(f)
        return self.head(f).squeeze(1)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of 1 - Pearson correlation along time (0: same shape)."""
    p = pred.float() - pred.float().mean(1, keepdim=True)
    t = target.float() - target.float().mean(1, keepdim=True)
    r = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1) + EPS)
    return (1 - r).mean()


# ---------------------------------------------------------------------------------------
# Data


class TrainingVersions(Dataset):
    """One version per training video and epoch (model_input.version_choice)."""

    def __init__(self, packed_dir: Path, policy: str, seed: int):
        self.frames, self.labels, _ = model_input.load_packed(packed_dir)
        self.policy, self.seed, self.epoch = policy, seed, 0

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        v = model_input.version_choice(self.policy, self.frames.shape[1], self.seed, self.epoch, i)
        return np.array(self.frames[i, v]), self.labels[i]  # a writable copy


class AllVersions(Dataset):
    """Every version of every video (validation and test)."""

    def __init__(self, packed_dir: Path):
        self.frames, self.labels, self.index = model_input.load_packed(packed_dir)
        self.n_versions = self.frames.shape[1]

    def __len__(self) -> int:
        return len(self.frames) * self.n_versions

    def __getitem__(self, k: int):
        i, v = divmod(k, self.n_versions)
        return np.array(self.frames[i, v]), self.labels[i]  # a writable copy


def _loader(dataset: Dataset, batch_size: int, workers: int, shuffle_seed: int | None):
    generator = None
    if shuffle_seed is not None:
        generator = torch.Generator().manual_seed(shuffle_seed)
    # fork: workers share the memory-mapped arrays instead of receiving pickled copies.
    context = multiprocessing.get_context("fork") if workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_seed is not None,
        generator=generator,
        num_workers=workers,
        multiprocessing_context=context,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------------------
# Training


def _autocast(device: torch.device):
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False), None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    return torch.autocast("cuda", dtype=dtype), scaler


def _save(obj, path: Path) -> None:
    """Written under a temporary name and renamed, so a checkpoint is never half written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def validate(
    model: nn.Module,
    dataset: AllVersions,
    refs: list[float | None],
    fs: float,
    protocol: dict,
    measure: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict:
    """Mean loss over every version of every video, and the protocol's success rate."""
    waves, losses = predict_dataset(model, dataset, device, batch_size, workers, with_loss=True)
    tolerance = protocol["tolerance_bpm"][protocol["measurements"][measure]["kind"]]
    hits, counted = 0, 0
    for k, wave in enumerate(waves):
        ref = refs[k // dataset.n_versions]
        window = (0.0, len(wave) / fs)
        (w,) = pipeline.measure_waveform(measure, wave, fs, [window], [ref], protocol)
        if w["excluded"] is None:
            counted += 1
            hits += w["error_bpm"] is not None and abs(w["error_bpm"]) <= tolerance
    return {"loss": float(np.mean(losses)), "success_rate": hits / counted if counted else None}


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    with_loss: bool = False,
):
    """Waveforms (N, T) float32 for every item, and per-item losses if asked."""
    model.eval()
    autocast, _ = _autocast(device)
    waves, losses = [], []
    for frames, labels in _loader(dataset, batch_size, workers, None):
        with autocast:
            pred = model(frames.to(device, non_blocking=True))
        pred = pred.float()
        waves.append(pred.cpu().numpy())
        if with_loss:
            y = labels.to(device)
            for p, t in zip(pred, y, strict=True):
                losses.append(float(pearson_loss(p[None], t[None])))
    out = np.concatenate(waves) if waves else np.zeros((0, 0), np.float32)
    return (out, losses) if with_loss else out


@torch.no_grad()
def predict(model: nn.Module, frames: np.ndarray, device: torch.device, batch_size: int = 8):
    """Waveforms (N, T) float32 of uint8 frames (N, T, H, W, 3)."""
    model.eval()
    autocast, _ = _autocast(device)
    out = []
    for start in range(0, len(frames), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(frames[start : start + batch_size]))
        with autocast:
            out.append(model(batch.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def build_model(job: dict, model_cfg: dict) -> BreathingNet:
    return BreathingNet(
        width=int(job["width"]),
        dilations=tuple(model_cfg["dilations"]),
        kernel=int(model_cfg["kernel"]),
    )


def load_model(checkpoint: Path, model_cfg: dict, device: torch.device) -> BreathingNet:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(state["job"], model_cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def train(
    job: dict,
    model_cfg: dict,
    train_cfg: dict,
    train_dir: Path,
    val_dir: Path,
    checkpoint_dir: Path,
    protocol: dict,
    measure: str,
    device: torch.device,
    workers: int = 0,
    log: Callable[[str], None] = print,
) -> dict:
    """Train one model (job: variant policy, width, lr, seed) and keep the epoch with the
    lowest validation loss. Resumes from checkpoint_dir/last.pt if it exists."""
    seed = int(job["seed"])
    torch.manual_seed(seed)
    model = build_model(job, model_cfg).to(device)
    epochs, batch = int(train_cfg["epochs"]), int(train_cfg["batch_size"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(job["lr"]), weight_decay=float(train_cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    autocast, scaler = _autocast(device)
    training = TrainingVersions(train_dir, job["versions"], seed)
    validation = AllVersions(val_dir)
    fs = float(train_cfg["fps"])
    refs = [
        pipeline.measure_waveform(measure, y, fs, [(0.0, len(y) / fs)], [None], protocol)[0][
            "est_bpm"
        ]
        for y in validation.labels
    ]  # the label through the same estimator gives each validation video's reference

    history, best, start = [], {"loss": float("inf"), "epoch": None}, 0
    last = checkpoint_dir / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        history, best, start = state["history"], state["best"], state["epoch"] + 1
        log(f"resuming at epoch {start}")
    for epoch in range(start, epochs):
        started = time.perf_counter()
        model.train()
        training.epoch = epoch
        losses = []
        for frames, labels in _loader(training, batch, workers, seed * 100_003 + epoch):
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                pred = model(frames.to(device, non_blocking=True))
            loss = pearson_loss(pred, labels.to(device))
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        val = validate(model, validation, refs, fs, protocol, measure, device, batch, workers)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val["loss"],
            "val_success_rate": val["success_rate"],
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        if val["loss"] < best["loss"]:
            best = {"loss": val["loss"], "epoch": epoch, "success_rate": val["success_rate"]}
            _save(
                {"job": job, "model": model.state_dict(), "epoch": epoch},
                checkpoint_dir / "best.pt",
            )
        _save(
            {
                "job": job,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch,
                "history": history,
                "best": best,
            },
            last,
        )
        log(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))
    return {"best": best, "history": history, "parameters": n_parameters(model)}
