"""Phase-5 deep model training (configs/deep_train.yaml), in two stages.

  tune:  every width and learning rate of `tuning`, seed 0, for both variants (A: uncompressed
         versions only; B: compression augmentation).
  final: each variant's best width and learning rate in the tune run (lowest validation
         loss), with every seed of `seeds`.

  python scripts/train_model.py --config configs/deep_train.yaml --stage tune \
      --run-id 20261128_scamps_deep-tune_s0 \
      --data-root /content/drive/MyDrive/camera-breathing/deep_data \
      --checkpoint-root /content/drive/MyDrive/camera-breathing/checkpoints \
      --results-root /content/drive/MyDrive/camera-breathing/results --cache-dir /content/deep_cache

  python scripts/train_model.py ... --stage final --tune-run 20261128_scamps_deep-tune_s0 \
      --run-id 20261128_scamps_deep-final_s0

The training and validation splits are unpacked once per session into --cache-dir (local
disk; model_input.pack_split). Checkpoints go to --checkpoint-root/<run_id>/<model>/ (Drive,
never git). One record per trained model in results/<run_id>/items.jsonl: a run resumes at
the next unfinished model, and a model at its next epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from breathing import deep, model_input, pipeline
from breathing.runinfo import RunDir, environment_snapshot

PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGES = ("tune", "final")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tune-run", help="run id of the tune stage (for --stage final)")
    parser.add_argument("--protocol", type=Path, default=pipeline.PROTOCOL_PATH)
    parser.add_argument("--data-root", type=Path, default=PROJECT_DIR / "data" / "deep")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / "work" / "deep_cache")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=PROJECT_DIR / "work" / "checkpoints"
    )
    parser.add_argument("--results-root", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--workers", type=int, default=4, help="data-loading processes")
    parser.add_argument(
        "--summarize-only", action="store_true", help="only rebuild the summary from items.jsonl"
    )
    return parser.parse_args(argv)


def job_id(job: dict) -> str:
    return f"{job['variant']}_w{job['width']}_lr{job['lr']:g}_s{job['seed']}"


def tune_jobs(config: dict) -> list[dict]:
    return [
        {"variant": v, "versions": policy, "width": int(w), "lr": float(lr), "seed": 0}
        for v, policy in config["variants"].items()
        for w in config["tuning"]["width"]
        for lr in config["tuning"]["lr"]
    ]


def best_settings(tune_records: list[dict], config: dict) -> dict[str, dict]:
    """Each variant's width and learning rate with the lowest validation loss."""
    expected = {job_id(j) for j in tune_jobs(config)}
    done = {r["id"]: r for r in tune_records if r["status"] == "ok"}
    missing = sorted(expected - set(done))
    if missing:
        raise SystemExit(f"the tune run is not finished; missing {missing}")
    best = {}
    for variant in config["variants"]:
        rows = [r for r in done.values() if r["variant"] == variant]
        top = min(rows, key=lambda r: r["best_val_loss"])
        best[variant] = {"width": top["width"], "lr": top["lr"], "from": top["id"]}
    return best


def final_jobs(config: dict, best: dict[str, dict]) -> list[dict]:
    return [
        {"variant": v, "versions": policy, "width": int(best[v]["width"]),
         "lr": float(best[v]["lr"]), "seed": int(seed)}
        for v, policy in config["variants"].items()
        for seed in config["seeds"]
    ]  # fmt: skip


def pack_splits(config: dict, data_root: Path, cache_dir: Path) -> dict[str, Path]:
    """Training and validation splits unpacked to local disk (reused if already there)."""
    out = {}
    for split in ("train", "validation"):
        files = sorted((data_root / config["splits"][split]).glob("*.npz"))
        started = time.perf_counter()
        index = model_input.pack_split(
            files, cache_dir / split, config["label"], jobs=os.cpu_count() or 1
        )
        with np.load(files[0]) as first:
            made = json.loads(str(first["meta"]))["model_input"]
        wanted = config["input"]
        if made["size"] != wanted["size"] or made["fps"] != wanted["fps"]:
            raise SystemExit(f"{split}: inputs are {made}, the config says {wanted}")
        print(
            f"{split}: {len(index['files'])} videos x {index['shape'][0]} versions ready in "
            f"{time.perf_counter() - started:.0f} s",
            flush=True,
        )
        out[split] = cache_dir / split
    return out


def compact(result: dict) -> dict:
    """The training history as lists: epoch, train loss, validation loss and success rate,
    seconds."""
    rows = [
        [h["epoch"], round(h["train_loss"], 5), round(h["val_loss"], 5),
         None if h["val_success_rate"] is None else round(h["val_success_rate"], 4),
         round(h["seconds"], 1)]
        for h in result["history"]
    ]  # fmt: skip
    best = result["best"]
    return {
        "best_epoch": best["epoch"],
        "best_val_loss": round(best["loss"], 5),
        "best_val_success_rate": best.get("success_rate"),
        "parameters": result["parameters"],
        "history": rows,
    }


def summarize(run: RunDir, config: dict, stage: str) -> str:
    records = [r for r in run.records() if r["status"] == "ok"]
    lines = [
        f"# Deep model training ({stage}): {run.run_id}",
        "",
        "Validation: every version of every validation video. Loss: 1 - Pearson correlation "
        f"with the label; success: estimate within the tolerance of {config['measure']}.",
        "",
        "| Model | Variant | Width | LR | Seed | Parameters | Best epoch | Val loss "
        "| Val success (%) | Minutes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        rate_text = (
            "–" if r["best_val_success_rate"] is None else f"{100 * r['best_val_success_rate']:.1f}"
        )
        minutes = sum(h[4] for h in r["history"]) / 60
        lines.append(
            f"| {r['id']} | {r['variant']} | {r['width']} | {r['lr']:g} | {r['seed']} "
            f"| {r['parameters']} | {r['best_epoch']} | {r['best_val_loss']:.4f} | {rate_text} "
            f"| {minutes:.0f} |"
        )
    summary: dict = {"models": len(records)}
    if stage == "tune" and records:
        try:
            summary["best"] = best_settings(records, config)
            lines += ["", "## Chosen for the final stage", ""]
            for v, b in summary["best"].items():
                lines.append(f"- {v}: width {b['width']}, learning rate {b['lr']:g} ({b['from']})")
        except SystemExit:
            pass
    if stage == "final" and records:
        lines += ["", "## Over seeds", "", "| Variant | Models | Val success (%) mean [min, max] |"]
        lines.append("|---|---|---|")
        for v in config["variants"]:
            rates = [100 * r["best_val_success_rate"] for r in records
                     if r["variant"] == v and r["best_val_success_rate"] is not None]  # fmt: skip
            if rates:
                spread = f"{np.mean(rates):.1f} [{min(rates):.1f}, {max(rates):.1f}]"
                lines.append(f"| {v} | {len(rates)} | {spread} |")
                summary.setdefault("val_success_rate", {})[v] = {
                    "mean": float(np.mean(rates)), "min": min(rates), "max": max(rates)
                }  # fmt: skip
    run.write_json("metrics.json", summary)
    text = "\n".join(lines) + "\n"
    (run.path / "summary.md").write_text(text)
    return text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    protocol = pipeline.load_protocol(args.protocol)
    run = RunDir(args.results_root, args.run_id)
    saved = run.saved_config()
    if args.summarize_only:
        if saved is None:
            raise SystemExit(f"no run {args.run_id}")
        print(summarize(run, saved["train"], saved["stage"]), flush=True)
        return 0
    if args.stage == "tune":
        jobs, best = tune_jobs(config), None
    else:
        if not args.tune_run:
            raise SystemExit("--stage final needs --tune-run")
        best = best_settings(RunDir(args.results_root, args.tune_run).records(), config)
        jobs = final_jobs(config, best)
    run.init({"train": config, "protocol": protocol, "stage": args.stage, "chosen": best,
              "jobs": [job_id(j) for j in jobs]})  # fmt: skip
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    extra = {
        "argv": sys.argv[1:],
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    run.add_session(environment_snapshot(PROJECT_DIR, extra))
    print(f"device: {extra['device']}", flush=True)
    dirs = pack_splits(config, args.data_root, args.cache_dir)
    done = {r["id"] for r in run.records() if r["status"] == "ok"}
    try:
        for job in jobs:
            name = job_id(job)
            if name in done:
                continue
            print(f"training {name}", flush=True)
            result = deep.train(
                job, config["model"], {**config["train"], "fps": config["input"]["fps"]},
                dirs["train"], dirs["validation"], args.checkpoint_root / args.run_id / name,
                protocol, config["measure"], device, args.workers,
            )  # fmt: skip
            checkpoint = f"{args.run_id}/{name}/best.pt"
            run.append(
                {"id": name, "status": "ok", **job, **compact(result), "checkpoint": checkpoint}
            )
    finally:
        summarize(run, config, args.stage)
    print(f"results in {run.path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
