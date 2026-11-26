"""Phase-5 model inputs from SCAMPS: face-only crops in compressed versions, downscaled.

The archive is streamed one video at a time (sources.stream_members). For each video of the
configured splits (test, validation, training; by member number), the face-only crop is
found once with pose on the uncompressed frames, by the rule of the face-only classic
experiment (sources.locate_face). Each version, the source and compressed copies, is encoded
at the crop's own size and frame rate, decoded, downscaled to size x size and resampled to a
fixed frame rate (model_input.model_frames); the label waveforms (d_br, d_ppg) are resampled
to the same time base. Test and validation videos get fixed versions, training videos the
source plus settings drawn at random (configs/deep_data_scamps.yaml).

One .npz per video, written to --data-root/<split>/<member>.npz:
  frames         (versions, T, size, size, 3) uint8
  settings       (versions,) setting names, the source first
  kbps           (versions,) measured bitrate of each version (NaN for the source)
  label_<name>   (T,) float32 label waveforms, e.g. label_breathing (d_br), label_ppg (d_ppg)
  meta           JSON text: crop, regions, frame rates and sizes of the versions, head pose

One record per video in results/<run_id>/items.jsonl, so a run resumes at the next video
(the stream is read again from the start, skipping finished videos without writing them).
SCAMPS-derived data stay on Drive, never in git (CLAUDE.md rule 2).

  python scripts/make_training_data.py --config configs/deep_data_scamps.yaml \
      --run-id 20261127_scamps_deep-data_s0 --jobs 8 \
      --data-root /content/drive/MyDrive/camera-breathing/deep_data \
      --results-root /content/drive/MyDrive/camera-breathing/results --work-dir /content/work
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import yaml

from breathing import compression, model_input, pipeline, sources
from breathing.runinfo import RunDir, environment_snapshot
from breathing.sources import make_pose

PROJECT_DIR = Path(__file__).resolve().parents[1]
MEMBER_NUMBER = re.compile(r"P(\d+)\.mat$")
VERSIONS = ("fixed", "random")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", type=Path, default=pipeline.PROTOCOL_PATH)
    parser.add_argument("--results-root", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--data-root", type=Path, default=PROJECT_DIR / "data" / "deep")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_DIR / "work")
    parser.add_argument("--source", help="override the dataset location in the config")
    parser.add_argument("--limit", type=int, help="stop after this many new videos")
    parser.add_argument("--jobs", type=int, default=1, help="versions encoded at once")
    parser.add_argument(
        "--summarize-only", action="store_true", help="only rebuild the summary from items.jsonl"
    )
    return parser.parse_args(argv)


def load_config(path: Path, source: str | None = None) -> dict:
    """The data config on top of the dataset config it names."""
    data = yaml.safe_load(path.read_text())
    config = yaml.safe_load((path.parent / data["dataset_config"]).read_text())
    config.update(data)
    if source:
        config["source"] = source
    if config["dataset"] != "scamps":
        raise SystemExit("model inputs are made from SCAMPS only (synthetic, with labels)")
    ranges = []
    for name, split in config["splits"].items():
        if split["versions"] not in VERSIONS:
            raise SystemExit(f"split {name}: versions must be one of {VERSIONS}")
        if not 1 <= split["first"] <= split["last"]:
            raise SystemExit(f"split {name}: needs 1 <= first <= last")
        ranges.append((split["first"], split["last"], name))
    ranges.sort()
    for (_, last, a), (first, _, b) in zip(ranges, ranges[1:], strict=False):
        if first <= last:
            raise SystemExit(f"splits {a} and {b} overlap")
    return config


def member_number(name: str) -> int | None:
    match = MEMBER_NUMBER.search(name)
    return int(match.group(1)) if match else None


def split_of(name: str, splits: dict) -> str | None:
    """The split whose range holds the archive member, if any."""
    number = member_number(name)
    if number is None:
        return None
    for split, cfg in splits.items():
        if cfg["first"] <= number <= cfg["last"]:
            return split
    return None


def complete(done: set[str], splits: dict) -> bool:
    """Every member number of every split is done."""
    numbers = {member_number(name) for name in done}
    return all(set(range(cfg["first"], cfg["last"] + 1)) <= numbers for cfg in splits.values())


class Skip:
    """Archive members to read past without writing them: finished ones and those outside
    every split. The stream consults it member by member, so additions apply at once."""

    def __init__(self, done: set[str], splits: dict):
        self.done, self.splits = done, splits

    def __contains__(self, name: str) -> bool:
        return name in self.done or split_of(name, self.splits) is None

    def add(self, name: str) -> None:
        self.done.add(name)


def video_settings(
    name: str, split: str, config: dict, all_settings: list[compression.Setting]
) -> list[compression.Setting]:
    """The versions of one video, the source first."""
    if config["splits"][split]["versions"] == "fixed":
        ladders = config["fixed_ladders"]
        return [s for s in all_settings if s.ladder == "source" or s.ladder in ladders]
    aug = config["augmentation"]
    pools = model_input.augmentation_pools(all_settings, aug["groups"])
    counts = [g["n"] for g in aug["groups"]]
    return [compression.SOURCE] + model_input.draw_settings(pools, counts, aug["seed"], name)


def write_npz(path: Path, **arrays) -> None:
    """Written under a temporary name and renamed, so a file is either whole or absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def process_video(
    name: str,
    local: Path,
    split: str,
    config: dict,
    protocol: dict,
    pose,
    data_root: Path,
    work_dir: Path,
    jobs: int,
) -> dict:
    """Model inputs of every version of one video, written to one .npz; returns the record."""
    started = time.perf_counter()
    screen, still = sources.scamps_screen(name, local, config)
    video = sources.scamps_video(name, local, config, protocol)
    crop, boxes, region_info, n_frames = sources.locate_face(video, pose, protocol)
    if crop is None:
        raise ValueError("no face found for the face-only crop")
    face = sources.cropped(video, crop)
    size, fps_out = int(config["model_input"]["size"]), float(config["model_input"]["fps"])
    n_out = model_input.n_output(n_frames, video.fps, fps_out)
    all_settings = compression.expand_ladders(protocol["compression"])
    settings = video_settings(name, split, config, all_settings)

    def one(setting: compression.Setting) -> tuple[np.ndarray, dict]:
        with pipeline.encoded_version(
            setting, face.frames, face.fps, face.size_hw, n_frames, protocol, work_dir
        ) as version:
            x = model_input.model_frames(version.frames(), version.fps, n_out, fps_out, size)
            info = {
                "setting": setting.name,
                "kbps": version.info.get("kbps"),
                "fps": version.fps,
                "height": version.size_hw[0],
                "width": version.size_hw[1],
            }
        return x, info

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(pool.map(one, settings))
    labels = {
        f"label_{k}": model_input.model_signal(v, video.fps, n_out, fps_out)
        for k, v in video.labels.items()
    }
    versions = [info for _, info in results]
    stem = PurePosixPath(name).stem
    meta = {
        "video": name,
        "split": split,
        "fps_source": video.fps,
        "n_frames_source": n_frames,
        "crop": crop,
        "regions": {"boxes": boxes, **region_info},
        "model_input": {"size": size, "fps": fps_out, "n_frames": n_out},
        "versions": versions,
        "head_pose_range": screen["head_pose_range"],
        "head_still": still,
        "action_units_max_range": screen["action_units_max_range"],
        "layout": video.info.get("layout"),
    }
    kbps = [np.nan if v["kbps"] is None else v["kbps"] for v in versions]
    relative = f"{split}/{stem}.npz"
    write_npz(
        data_root / relative,
        frames=np.stack([x for x, _ in results]),
        settings=np.array([v["setting"] for v in versions]),
        kbps=np.array(kbps, dtype=np.float64),
        meta=np.array(json.dumps(meta)),
        **labels,
    )
    # The record repeats the meta compactly (the versions' sizes and frame rates follow from
    # their settings and the crop) so that items.jsonl stays small for 2800 videos.
    record = {
        "id": name,
        "status": "ok",
        "subject": stem,
        **{k: v for k, v in meta.items() if k not in ("versions", "layout")},
        "settings": [v["setting"] for v in versions],
        "kbps": [v["kbps"] for v in versions],
        "file": relative,
        "bytes": (data_root / relative).stat().st_size,
        "seconds": time.perf_counter() - started,
    }
    return rounded(record)


def rounded(value, digits: int = 3):
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v, digits) for v in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def run_videos(config, protocol, run, pose, data_root, work_dir, jobs, limit) -> int:
    splits = config["splits"]
    done = {r["id"] for r in run.records() if r["status"] == "ok"}
    if complete(done, splits):
        return 0
    skip = Skip(done, splits)
    videos = 0
    members = sources.stream_members(config["source"], work_dir / "stage", skip)
    try:
        for name, local in members:
            if limit is not None and videos >= limit:
                break
            split = split_of(name, splits)
            try:
                record = process_video(
                    name, local, split, config, protocol, pose, data_root,
                    work_dir / "versions", jobs,
                )  # fmt: skip
            except Exception as err:  # recorded and retried on the next run
                traceback.print_exc()
                record = {"id": name, "status": "error", "split": split, "video": name}
                record["error"] = f"{type(err).__name__}: {err}"
            run.append(record)
            videos += 1
            took = f" in {record['seconds']:.0f} s" if "seconds" in record else ""
            print(f"{name} ({split}): {record['status']}{took}", flush=True)
            if record["status"] == "ok":
                skip.add(name)
                if complete(done, splits):
                    break  # the archive is in member order: nothing later is needed
    finally:
        members.close()
    return videos


# ---------------------------------------------------------------------------------------
# Summary


def summarize(run: RunDir, config: dict) -> str:
    records = run.records()
    ok = {r["id"]: r for r in records if r["status"] == "ok"}
    errors = {r["id"]: r for r in records if r["status"] == "error" and r["id"] not in ok}
    splits = {}
    for split, cfg in config["splits"].items():
        rows = [r for r in ok.values() if r["split"] == split]
        sides = [r["crop"][1] - r["crop"][0] for r in rows]
        splits[split] = {
            "expected": cfg["last"] - cfg["first"] + 1,
            "videos": len(rows),
            "head_still": sum(bool(r["head_still"]) for r in rows),
            "versions": sum(len(r["settings"]) for r in rows),
            "errors": sum(1 for r in errors.values() if r.get("split") == split),
            "bytes": sum(r["bytes"] for r in rows),
            "crop_side_px": {
                "min": min(sides) if sides else None,
                "median": float(np.median(sides)) if sides else None,
                "max": max(sides) if sides else None,
            },
        }
    drawn = Counter(
        name
        for r in ok.values()
        if config["splits"][r["split"]]["versions"] == "random"
        for name in r["settings"][1:]
    )
    seconds = [r["seconds"] for r in ok.values()]
    summary = {
        "splits": splits,
        "random_versions_drawn": dict(sorted(drawn.items())),
        "seconds_per_video": {
            "median": float(np.median(seconds)) if seconds else None,
            "total": float(np.sum(seconds)),
        },
        "errors": [{"id": k, "error": r.get("error")} for k, r in errors.items()],
    }
    run.write_json("metrics.json", summary)
    text = markdown(summary, run.run_id, config)
    (run.path / "summary.md").write_text(text)
    return text


def markdown(summary: dict, run_id: str, config: dict) -> str:
    fmt = pipeline._fmt
    mi = config["model_input"]
    lines = [
        f"# Model inputs: {run_id}",
        "",
        f"SCAMPS face-only crops, downscaled to {mi['size']}x{mi['size']} at {mi['fps']:g} fps. "
        "One .npz per video under the data root (not in git).",
        "",
        "| Split | Members | Videos | Head still | Versions | Errors | GB | Crop side px "
        "(min / median / max) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split, s in summary["splits"].items():
        cfg = config["splits"][split]
        side = s["crop_side_px"]
        lines.append(
            f"| {split} | P{cfg['first']:06d}-P{cfg['last']:06d} ({s['expected']}) "
            f"| {s['videos']} | {s['head_still']} | {s['versions']} | {s['errors']} "
            f"| {fmt(s['bytes'] / 1e9)} | {side['min']} / {fmt(side['median'], 0)} "
            f"/ {side['max']} |"
        )
    t = summary["seconds_per_video"]
    lines += [
        "",
        f"Seconds per video: median {fmt(t['median'], 1)}, total {fmt(t['total'], 0)}.",
        "",
        "## Random versions drawn (training)",
        "",
        "| Setting | Videos |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in summary["random_versions_drawn"].items()]
    if summary["errors"]:
        lines += ["", "## Errors", ""] + [f"- `{e['id']}`: {e['error']}" for e in summary["errors"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, args.source)
    protocol = pipeline.load_protocol(args.protocol)
    run = RunDir(args.results_root, args.run_id)
    saved = run.saved_config()
    if args.summarize_only and saved is not None:  # rebuild with the run's own config
        print(summarize(run, saved["data"]), flush=True)
        return 0
    run.init({"data": config, "protocol": protocol})
    if args.summarize_only:
        print(summarize(run, config), flush=True)
        return 0
    pose, pose_info = make_pose(config, args.work_dir)
    extra = {**pose_info, "argv": sys.argv[1:], "jobs": args.jobs}
    run.add_session(environment_snapshot(PROJECT_DIR, extra))
    try:
        videos = run_videos(
            config, protocol, run, pose, args.data_root, args.work_dir, args.jobs, args.limit
        )
    finally:
        if hasattr(pose, "close"):
            pose.close()
        summarize(run, config)
    print(f"worked on {videos} video(s); data in {args.data_root}, run in {run.path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
