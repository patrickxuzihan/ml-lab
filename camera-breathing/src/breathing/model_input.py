"""Inputs of the phase-5 deep model, and the compressed versions it is trained on.

A version's decoded frames are downscaled to size x size by area averaging and resampled to a
fixed frame rate: output frame k is the mean of the input frames whose time (index / fps)
falls in [k, k + 1) / fps_out. The label waveforms are resampled the same way, so frames and
labels share one time base whatever the version's frame rate. `pack_split` turns the
per-video files into arrays on local disk for training. Pure numpy; no torch.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from breathing.compression import Setting


def n_output(n_frames: int, fps: float, fps_out: float) -> int:
    """Output frames for a video of n_frames at fps: whole 1/fps_out intervals only."""
    return int(np.floor(n_frames / fps * fps_out + 1e-9))


def area_axis(values: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Resample one axis to n_out samples by area: each output sample is the mean of the input
    over its 1/n_out share of the axis, pixels being constant over their width. Computed from
    cumulative sums (no matrix products, so no BLAS threads competing with --jobs)."""
    x = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    n_in = x.shape[0]
    cum = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    edges = np.arange(n_out + 1) * (n_in / n_out)  # output edges in input pixels
    k = np.minimum(np.floor(edges).astype(int), n_in - 1)
    frac = (edges - k).reshape((-1,) + (1,) * (x.ndim - 1))
    integral = cum[k] + frac * x[k]  # of the input from 0 to each edge
    return np.moveaxis((integral[1:] - integral[:-1]) * (n_out / n_in), 0, axis)


def resize(frame: np.ndarray, size: int) -> np.ndarray:
    """(H, W, C) frame resampled to (size, size, C) by area (float64)."""
    return area_axis(area_axis(frame, size, 0), size, 1)


def _bin(i: int, fps: float, fps_out: float) -> int:
    return int(np.floor(i * fps_out / fps + 1e-9))


def _fill_empty(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Mean per bin; a bin without samples (input slower than the output) repeats the last
    bin that has some."""
    filled = np.flatnonzero(counts)
    if filled.size == 0:
        raise ValueError("no samples")
    last = np.searchsorted(filled, np.arange(len(counts)), side="right") - 1
    source = filled[np.clip(last, 0, None)]
    shape = (-1,) + (1,) * (sums.ndim - 1)
    return sums[source] / counts[source].reshape(shape)


def model_frames(
    frames: Iterable[np.ndarray], fps: float, n_out: int, fps_out: float, size: int
) -> np.ndarray:
    """(n_out, size, size, 3) uint8 model input from RGB frames at fps."""
    sums = np.zeros((n_out, size, size, 3))
    counts = np.zeros(n_out, dtype=int)
    for i, frame in enumerate(frames):
        k = _bin(i, fps, fps_out)
        if k >= n_out:
            continue  # past the last whole interval; still read so the decoder finishes
        sums[k] += resize(frame, size)
        counts[k] += 1
    return np.clip(np.rint(_fill_empty(sums, counts)), 0, 255).astype(np.uint8)


def model_signal(values: np.ndarray, fps: float, n_out: int, fps_out: float) -> np.ndarray:
    """A per-frame label waveform on the model's time base (float32)."""
    values = np.asarray(values, dtype=np.float64)
    bins = np.floor(np.arange(len(values)) * fps_out / fps + 1e-9).astype(int)
    keep = bins < n_out
    sums = np.bincount(bins[keep], weights=values[keep], minlength=n_out)
    counts = np.bincount(bins[keep], minlength=n_out)
    return _fill_empty(sums, counts).astype(np.float32)


# ---------------------------------------------------------------------------------------
# Compressed versions for training


def lossy(settings: Iterable[Setting]) -> list[Setting]:
    """Settings that change the picture through compression: no source, no uncompressed
    decimation or downscaling (rawvideo) and no lossless CRF 0."""
    return [s for s in settings if s.codec not in (None, "rawvideo") and s.crf != 0]


def augmentation_pools(settings: list[Setting], groups: list[dict]) -> list[list[Setting]]:
    """The settings each group draws from: the lossy settings of its ladders. Allocation
    cells at full frame rate and size are left out: they repeat bitrate rungs."""
    pools = []
    for group in groups:
        pool = [
            s
            for s in lossy(settings)
            if s.ladder in group["ladders"]
            and not (s.ladder == "allocation" and (s.decimate, s.scale) == (1, 1))
        ]
        if len(pool) < group["n"]:
            raise ValueError(
                f"group {group['ladders']} has {len(pool)} settings, needs {group['n']}"
            )
        pools.append(pool)
    return pools


def draw_settings(
    pools: list[list[Setting]], counts: list[int], seed: int, video_id: str
) -> list[Setting]:
    """counts[i] settings drawn without replacement from pools[i]. The draw depends only on
    the seed and the video, so a resumed or repeated run makes the same versions."""
    rng = np.random.default_rng([seed, zlib.crc32(video_id.encode())])
    drawn = []
    for pool, n in zip(pools, counts, strict=True):
        drawn += [pool[i] for i in sorted(rng.choice(len(pool), size=n, replace=False))]
    return drawn


# ---------------------------------------------------------------------------------------
# Training data on local disk


def pack_split(files: list[Path], out_dir: Path, label: str, jobs: int = 1) -> dict:
    """Unpack a split's .npz files (scripts/make_training_data.py) into uncompressed arrays in
    out_dir, so training reads one version of one video without decompressing whole files:

      frames.npy   (videos, versions, T, size, size, 3) uint8, opened as a memory map
      labels.npy   (videos, T) float32: the label waveform `label_<label>`
      index.json   per video: id, setting names, head_still; written last, so a complete
                   directory is reused (same files) and an interrupted one is redone

    Every video of a split must have the same number of versions and frames.
    """
    files = sorted(Path(f) for f in files)
    out_dir = Path(out_dir)
    index_path = out_dir / "index.json"
    names = [f.name for f in files]
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["files"] == names:
            return index
    if not files:
        raise ValueError(f"no .npz files for {out_dir.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path.unlink(missing_ok=True)
    with np.load(files[0]) as first:
        shape = first["frames"].shape
    frames = np.lib.format.open_memmap(
        out_dir / "frames.npy", mode="w+", dtype=np.uint8, shape=(len(files), *shape)
    )
    labels = np.zeros((len(files), shape[1]), dtype=np.float32)
    videos: list[dict] = [{}] * len(files)

    def one(i: int) -> None:
        with np.load(files[i]) as z:
            if z["frames"].shape != shape:
                raise ValueError(f"{files[i].name}: frames {z['frames'].shape}, expected {shape}")
            frames[i] = z["frames"]
            labels[i] = z[f"label_{label}"]
            meta = json.loads(str(z["meta"]))
            settings = [str(s) for s in z["settings"]]
        videos[i] = {"video": meta["video"], "settings": settings, "head_still": meta["head_still"]}

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        list(pool.map(one, range(len(files))))
    frames.flush()
    np.save(out_dir / "labels.npy", labels)
    index = {"files": names, "label": label, "shape": list(shape), "videos": videos}
    index_path.write_text(json.dumps(index))
    return index


def load_packed(out_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """(frames memory map, labels, index) of a packed split."""
    out_dir = Path(out_dir)
    index = json.loads((out_dir / "index.json").read_text())
    frames = np.load(out_dir / "frames.npy", mmap_mode="r")
    return frames, np.load(out_dir / "labels.npy"), index


def version_choice(policy: str, n_versions: int, seed: int, epoch: int, video: int) -> int:
    """The version of a training video shown in one epoch: always the source (0) for
    "source", one drawn at random among all versions for "all". The draw depends only on
    (seed, epoch, video), so a resumed run shows the same versions."""
    if policy == "source":
        return 0
    if policy == "all":
        return int(np.random.default_rng([seed, epoch, video]).integers(n_versions))
    raise ValueError(f"unknown version policy {policy!r}")
