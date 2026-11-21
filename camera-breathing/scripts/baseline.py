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
import shutil
import sys
import tarfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import yaml

from breathing import baseline, datacheck, pipeline, roi
from breathing.compression import SOURCE
from breathing.datasets import scamps, ubfc
from breathing.runinfo import RunDir, environment_snapshot
from breathing.video_io import iter_frames, probe

PROJECT_DIR = Path(__file__).resolve().parents[1]
STREAM_ATTEMPTS = 3
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


def make_pose(config: dict, work_dir: Path):
    model = roi.ensure_model(config["pose_model_url"], work_dir / "models")
    info = {"pose_model": {"file": model.name, "sha256": roi.file_sha256(model)}}
    return roi.PoseEstimator(model), info


def measure_video(
    frames: Callable,
    fps: float,
    size_hw: tuple[int, int],
    pose,
    protocol: dict,
    references: Callable,
    work_dir: Path,
    wide_max_std_bpm: float | None = None,
) -> dict:
    """Regions from pose on the source, references, then M1-M3 (main and extra windows).

    frames() yields the RGB frames (called twice: pose sampling, then the signals);
    references(windows, n_frames) returns (refs, details) for those windows.
    """
    reg = protocol["regions"]
    height, width = size_hw
    landmarks, n_frames, _ = datacheck.sample_pose(frames(), fps, reg["pose_sample_fps"], pose)
    boxes, region_info = roi.measurement_regions(
        landmarks, width, height, reg["visibility_threshold"], reg["min_shoulder_frac"]
    )
    duration = n_frames / fps
    windows = pipeline.analysis_windows(duration, protocol)
    refs, details = references(windows, n_frames)
    whole = baseline.whole_windows(duration, protocol)
    whole_refs, whole_details = references(whole, n_frames)
    variants = {baseline.WHOLE_VIDEO: (whole, whole_refs)}
    if wide_max_std_bpm is not None:
        wide = baseline.wide_references(refs, details, protocol, wide_max_std_bpm)
        variants[baseline.WIDE_REFERENCE] = (windows, wide)
    record = pipeline.process_version(
        SOURCE, frames, fps, size_hw, n_frames, boxes, windows, refs, protocol, work_dir, variants
    )
    record["regions"] = {"boxes": boxes, **region_info}
    record["reference"] = details
    record["reference_whole_video"] = whole_details
    return record


def measure_ubfc(subject: ubfc.Subject, config: dict, protocol: dict, pose, work_dir: Path):
    stage = work_dir / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    local = stage / ubfc.VIDEO_NAME
    started = time.perf_counter()
    shutil.copyfile(subject.video, local)  # from the Drive shortcut to local disk
    copied = time.perf_counter()
    try:
        info = probe(local)
        gt = ubfc.read_ground_truth(subject.ground_truth)

        def frames():
            return iter_frames(local, info.width, info.height)

        def references(windows, n_frames):
            # The protocol's time axis is frame index / mean frame rate, and the ground truth
            # has one sample per frame (checked in phase 1); refuse anything else.
            if len(gt["ppg"]) != n_frames:
                raise ValueError(f"{len(gt['ppg'])} ground-truth samples for {n_frames} frames")
            return baseline.ubfc_references(gt["ppg"], gt["hr"], info.fps, windows, protocol)

        record = measure_video(
            frames,
            info.fps,
            (info.height, info.width),
            pose,
            protocol,
            references,
            work_dir / "versions",
            config["sensitivity"]["wide_reference_max_std_bpm"],
        )
    finally:
        local.unlink(missing_ok=True)
    record.update(
        video=subject.id,
        subject=subject.id,
        video_info=info.to_dict(),
        flags=ubfc.timestamp_flags(gt["t"], protocol["reference"]["ubfc_gap_flag_s"]),
        timing_s={"copy": copied - started, "total": time.perf_counter() - started},
    )
    return record


def measure_scamps(name: str, local: Path, config: dict, protocol: dict, pose, work_dir: Path):
    started = time.perf_counter()
    sel = config["selection"]
    head_keys = sel["head_pose_keys"]
    waves = scamps.read_waveforms(local, head_keys + sel["action_unit_keys"])
    missing = [k for k in head_keys if k not in waves]
    if missing:
        raise ValueError(f"head-pose variables missing: {missing}")
    head = baseline.value_ranges({k: waves[k] for k in head_keys})
    units = baseline.value_ranges({k: v for k, v in waves.items() if k not in head_keys})
    base = {
        "video": name,
        "subject": PurePosixPath(name).stem,  # no identity labels: one cluster per video
        "head_pose_range": head,
        "action_units_max_range": max(units.values()) if units else None,
    }
    if not baseline.head_is_still(head, sel["max_head_pose_range"]):
        print(f"  skipped: head pose varies by up to {max(head.values()):.4g}", flush=True)
        return {**base, "status": "skipped", "reason": "head_motion"}

    video = scamps.read_video(local, config["frames_key"], config["waveform_keys"])
    fps = float(config["fps"])
    frames = video.frames
    known = {"breathing": video.waveforms["breathing"], "heart": video.waveforms["ppg"]}

    def references(windows, n_frames):
        if len(known["breathing"]) != n_frames:
            raise ValueError(f"{len(known['breathing'])} label samples for {n_frames} frames")
        return pipeline.waveform_references(known, fps, windows, protocol), {}

    record = measure_video(
        lambda: iter(frames),
        fps,
        frames.shape[1:3],
        pose,
        protocol,
        references,
        work_dir / "versions",
    )
    record.update(base, layout=video.layout, timing_s={"total": time.perf_counter() - started})
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
    for attempt in range(1, STREAM_ATTEMPTS + 1):
        if n_ok >= target:
            return measured
        try:
            members = scamps.iter_tar_members(config["source"], work_dir / "stage", skip=seen)
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
                    members.close()
                    raise SystemExit(
                        f"none of the first {n_skipped} videos has a still head under "
                        f"max_head_pose_range={sel['max_head_pose_range']:g}; the measured "
                        "head-pose ranges are in items.jsonl. Push the results and ask for the "
                        "threshold to be checked."
                    )
                if n_ok >= target:
                    members.close()
                    return measured
            print(f"end of archive: {n_ok} head-still videos measured", flush=True)
            return measured
        except (OSError, EOFError, tarfile.TarError) as err:
            # A dropped download: the next attempt streams again and skips finished members.
            print(f"stream attempt {attempt} failed: {err}", flush=True)
            if attempt == STREAM_ATTEMPTS:
                raise
            time.sleep(10 * attempt)
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
