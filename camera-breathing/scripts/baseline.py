"""Phase-3 baseline: the evaluation protocol on the uncompressed videos of a dataset.

One video at a time: stage it on local disk, find the regions with MediaPipe pose on the
source, compute the references, measure M1-M3 and delete the video. Results go to
<results-root>/<run-id>/; re-running the same command resumes (finished videos are skipped,
failed ones are retried). Only numbers are kept: no frames, landmarks or signals.

- ubfc:   every subject folder under `source` (the Drive shortcut), copied one at a time.
- scamps: the full archive is streamed; videos whose head pose varies are skipped until
          `selection.n_videos` head-still videos have been measured.

  python scripts/baseline.py --config configs/baseline_ubfc.yaml \
      --run-id 20261122_ubfc_baseline_s0 \
      --results-root /content/drive/MyDrive/camera-breathing/results --work-dir /content/work

`--summarize-only` rebuilds metrics.json, summary.md and windows.csv from items.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import yaml

from breathing import baseline, pipeline, sources
from breathing.compression import SOURCE
from breathing.datasets import ubfc
from breathing.runinfo import RunDir, environment_snapshot
from breathing.sources import make_pose

PROJECT_DIR = Path(__file__).resolve().parents[1]
WINDOW_COLUMNS = ["variant", "video", "subject", "measure", "window", "t0", "t1", "ref_bpm"]
WINDOW_COLUMNS += ["est_bpm", "error_bpm", "snr_db", "excluded"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", type=Path, default=pipeline.PROTOCOL_PATH)
    parser.add_argument("--results-root", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_DIR / "work")
    parser.add_argument("--source", help="override the dataset location in the config")
    parser.add_argument("--limit", type=int, help="stop after measuring this many new videos")
    parser.add_argument(
        "--summarize-only", action="store_true", help="only rebuild the summary from items.jsonl"
    )
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> dict:
    config = yaml.safe_load(args.config.read_text())
    if args.source:
        config["source"] = args.source
    if config["dataset"] not in ("ubfc", "scamps"):
        raise SystemExit(f"unsupported dataset {config['dataset']!r}")
    return config


def measure_video(
    video: sources.Video,
    pose,
    protocol: dict,
    work_dir: Path,
    wide_max_std_bpm: float | None = None,
) -> dict:
    """Regions from pose on the source, references, then M1-M3 (main and extra windows)."""
    boxes, region_info, n_frames = sources.locate(video, pose, protocol)
    duration = n_frames / video.fps
    windows = pipeline.analysis_windows(duration, protocol)
    refs, details = video.references(windows, n_frames)
    whole = baseline.whole_windows(duration, protocol)
    whole_refs, whole_details = video.references(whole, n_frames)
    variants = {baseline.WHOLE_VIDEO: (whole, whole_refs)}
    if wide_max_std_bpm is not None:
        wide = baseline.wide_references(refs, details, protocol, wide_max_std_bpm)
        variants[baseline.WIDE_REFERENCE] = (windows, wide)
    record = pipeline.process_version(
        SOURCE,
        video.frames,
        video.fps,
        video.size_hw,
        n_frames,
        boxes,
        windows,
        refs,
        protocol,
        work_dir,
        variants,
    )
    record["regions"] = {"boxes": boxes, **region_info}
    record["reference"] = details
    record["reference_whole_video"] = whole_details
    return record


def measure_ubfc(subject: ubfc.Subject, config: dict, protocol: dict, pose, work_dir: Path):
    started = time.perf_counter()
    with sources.ubfc_video(subject, protocol, work_dir / "stage") as video:
        wide = config["sensitivity"]["wide_reference_max_std_bpm"]
        record = measure_video(video, pose, protocol, work_dir / "versions", wide)
    record.update(
        video=subject.id,
        subject=subject.id,
        video_info=video.info["video_info"],
        flags=video.info["flags"],
        timing_s={"copy": video.info["timing_s"]["copy"], "total": time.perf_counter() - started},
    )
    return record


def measure_scamps(name: str, local: Path, config: dict, protocol: dict, pose, work_dir: Path):
    started = time.perf_counter()
    screen, still = sources.scamps_screen(name, local, config)
    if not still:
        largest = max(screen["head_pose_range"].values())
        print(f"  skipped: head pose varies by up to {largest:.4g}", flush=True)
        return {**screen, "status": "skipped", "reason": "head_motion"}
    video = sources.scamps_video(name, local, config, protocol)
    record = measure_video(video, pose, protocol, work_dir / "versions")
    record.update(
        screen, layout=video.info["layout"], timing_s={"total": time.perf_counter() - started}
    )
    return record


def process(item_id: str, measure: Callable[[], dict]) -> dict:
    print(f"{item_id} ...", flush=True)
    try:
        return {"id": item_id, "status": "ok", **measure()}
    except Exception as err:  # recorded and retried on the next run
        traceback.print_exc()
        return {"id": item_id, "status": "error", "error": f"{type(err).__name__}: {err}"}


def run_ubfc(config: dict, protocol: dict, run: RunDir, pose, work_dir: Path, limit) -> int:
    subjects, _ = ubfc.find_subjects(config["source"])
    if not subjects:
        raise SystemExit(
            f"no folder with {ubfc.VIDEO_NAME} and {ubfc.GT_NAME} under {config['source']}"
        )
    done = run.done_ids()
    measured = 0
    for subject in subjects:
        if subject.id in done:
            continue
        if limit is not None and measured >= limit:
            break
        record = process(
            subject.id, lambda s=subject: measure_ubfc(s, config, protocol, pose, work_dir)
        )
        run.append(record)
        measured += record["status"] == "ok"
    return measured


def run_scamps(config: dict, protocol: dict, run: RunDir, pose, work_dir: Path, limit) -> int:
    sel = config["selection"]
    records = run.records()
    seen = {r["id"] for r in records if r["status"] in ("ok", "skipped")}
    n_ok = sum(r["status"] == "ok" for r in records)
    n_skipped = sum(r["status"] == "skipped" for r in records)
    target = sel["n_videos"] if limit is None else min(sel["n_videos"], n_ok + limit)
    measured = 0
    if n_ok >= target:
        return measured
    members = sources.stream_members(config["source"], work_dir / "stage", seen)
    try:
        for name, local in members:
            record = process(
                name,
                lambda n=name, p=local: measure_scamps(n, p, config, protocol, pose, work_dir),
            )
            run.append(record)
            if record["status"] in ("ok", "skipped"):
                seen.add(name)
            n_ok += record["status"] == "ok"
            measured += record["status"] == "ok"
            n_skipped += record["status"] == "skipped"
            if n_ok == 0 and n_skipped >= sel["give_up_after"]:
                raise SystemExit(
                    f"none of the first {n_skipped} videos has a still head under "
                    f"max_head_pose_range={sel['max_head_pose_range']:g}; the measured "
                    "head-pose ranges are in items.jsonl. Push the results and ask for the "
                    "threshold to be checked."
                )
            if n_ok >= target:
                return measured
    finally:
        members.close()
    print(f"end of archive: {n_ok} head-still videos measured", flush=True)
    return measured


def summarize(run: RunDir, config: dict, protocol: dict) -> str:
    records = run.records()
    ok = [r for r in records if r.get("status") == "ok"]
    with open(run.path / "windows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WINDOW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(baseline.window_rows(ok))
    summary = baseline.summarize(records, config, protocol)
    run.write_json("metrics.json", summary)
    text = baseline.markdown(summary, config, protocol, run.run_id)
    (run.path / "summary.md").write_text(text)
    return text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args)
    protocol = pipeline.load_protocol(args.protocol)
    run = RunDir(args.results_root, args.run_id)
    saved = run.saved_config()
    if args.summarize_only and saved is not None:  # rebuild with the run's own config
        print(summarize(run, saved["experiment"], saved["protocol"]), flush=True)
        return 0
    run.init({"experiment": config, "protocol": protocol})
    if args.summarize_only:
        print(summarize(run, config, protocol), flush=True)
        return 0
    pose, pose_info = make_pose(config, args.work_dir)
    run.add_session(environment_snapshot(PROJECT_DIR, {**pose_info, "argv": sys.argv[1:]}))
    runner = {"ubfc": run_ubfc, "scamps": run_scamps}[config["dataset"]]
    try:
        measured = runner(config, protocol, run, pose, args.work_dir, args.limit)
    finally:
        if hasattr(pose, "close"):
            pose.close()
        summarize(run, config, protocol)
    print(f"measured {measured} new video(s); results in {run.path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
