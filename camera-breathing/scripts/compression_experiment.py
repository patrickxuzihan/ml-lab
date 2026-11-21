"""Phase-4 compression experiments: every configured compression setting of each source video.

For each source video the regions are found once, with pose on the uncompressed frames, and
the references once; then each setting is encoded, decoded, measured and deleted
(pipeline.process_version). There is one record per video and setting, so a run resumes at
the next unfinished setting. --jobs measures that many settings of a video at once; every
encoder is single-threaded, so the bitstreams do not depend on --jobs. Only numbers are kept.

An experiment config names a dataset config (configs/baseline_*.yaml, phase 3) and the
ladders of configs/protocol.yaml to run:

  python scripts/compression_experiment.py --config configs/ladders_ubfc.yaml \
      --run-id 20261123_ubfc_ladders_s0 --jobs 2 \
      --results-root /content/drive/MyDrive/camera-breathing/results --work-dir /content/work

`--summarize-only` rebuilds metrics.json and summary.md from items.jsonl.

To keep items.jsonl small, floats are rounded to 3 decimals, the windows of a record are
stored as lists in the order of WINDOW_FIELDS (`load_records` turns them back into dicts),
and a record keeps only the setting's name (its codec, CRF, bitrate, decimation and scale
follow from configs/protocol.yaml) and the measured bitrate `kbps` (not the file size).
"""

from __future__ import annotations

import argparse
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from breathing import baseline, compression, metrics, pipeline, sources
from breathing.datasets import ubfc
from breathing.runinfo import RunDir, environment_snapshot
from breathing.sources import make_pose

PROJECT_DIR = Path(__file__).resolve().parents[1]
LADDERS = ("crf", "bitrate", "allocation", "raw")
DIGITS = 3  # floats in items.jsonl are rounded to keep the files small
WINDOW_FIELDS = ("measure", "window", "t0", "t1", "ref_bpm", "est_bpm", "error_bpm", "snr_db")
WINDOW_FIELDS += ("excluded",)
# Implied by the setting's name, or by kbps (bytes): not stored.
DROPPED_FIELDS = ("name", "ladder", "codec", "crf", "decimate", "scale", "bytes")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", type=Path, default=pipeline.PROTOCOL_PATH)
    parser.add_argument("--results-root", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_DIR / "work")
    parser.add_argument("--source", help="override the dataset location in the config")
    parser.add_argument("--limit", type=int, help="stop after this many new source videos")
    parser.add_argument("--jobs", type=int, default=1, help="settings measured at once")
    parser.add_argument(
        "--summarize-only", action="store_true", help="only rebuild the summary from items.jsonl"
    )
    return parser.parse_args(argv)


def load_config(path: Path, source: str | None = None) -> dict:
    """The experiment config on top of the dataset config it names."""
    experiment = yaml.safe_load(path.read_text())
    config = yaml.safe_load((path.parent / experiment["dataset_config"]).read_text())
    config.update(experiment)
    if source:
        config["source"] = source
    if config["dataset"] not in ("ubfc", "scamps"):
        raise SystemExit(f"unsupported dataset {config['dataset']!r}")
    unknown = set(config["ladders"]) - set(LADDERS)
    if unknown:
        raise SystemExit(f"unknown ladders {sorted(unknown)}; choose from {LADDERS}")
    return config


def experiment_settings(config: dict, protocol: dict) -> list[compression.Setting]:
    """The source plus every setting of the chosen ladders, in protocol order."""
    settings = [
        s
        for s in compression.expand_ladders(protocol["compression"])
        if s.ladder == "source" or s.ladder in config["ladders"]
    ]
    names = {s.name for s in settings}
    missing = [n for n in config.get("relocate") or [] if n not in names]
    if missing:
        raise SystemExit(f"relocate names settings that are not run: {missing}")
    return settings


def rounded(value, digits: int = DIGITS):
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v, digits) for v in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def pack(record: dict) -> dict:
    """The record as stored: rounded floats, windows as lists, setting fields dropped."""
    out = {k: v for k, v in record.items() if k not in DROPPED_FIELDS}
    if "windows" in out:
        out["windows"] = [[w[f] for f in WINDOW_FIELDS] for w in out["windows"]]
    if "relocated" in out:
        moved = out["relocated"]["windows"]
        out["relocated"] = {
            **out["relocated"],
            "windows": [[w[f] for f in WINDOW_FIELDS] for w in moved],
        }
    return rounded(out)


def unpack(record: dict) -> dict:
    """A stored record with its windows as dicts again."""
    out = dict(record)
    if "windows" in out:
        out["windows"] = [dict(zip(WINDOW_FIELDS, w, strict=True)) for w in out["windows"]]
    if "relocated" in out:
        moved = out["relocated"]["windows"]
        out["relocated"] = {
            **out["relocated"],
            "windows": [dict(zip(WINDOW_FIELDS, w, strict=True)) for w in moved],
        }
    return out


def load_records(run: RunDir) -> list[dict]:
    return [unpack(r) for r in run.records()]


class LockedPose:
    """Pose estimator shared by worker threads (MediaPipe calls are serialized)."""

    def __init__(self, pose):
        self.pose, self.lock = pose, threading.Lock()

    def __call__(self, frame):
        with self.lock:
            return self.pose(frame)


def known_regions(by_id: dict, video_id: str) -> tuple | None:
    """Regions and frame count from the video's source record, if it was measured."""
    record = by_id.get(f"{video_id}/{compression.SOURCE.name}")
    if record is None or record.get("status") != "ok":
        return None
    regions = record["regions"]
    boxes = {k: None if v is None else tuple(v) for k, v in regions["boxes"].items()}
    info = {k: v for k, v in regions.items() if k != "boxes"}
    return boxes, info, record["n_frames"]


def todo_settings(by_id: dict, video_id: str, settings: list) -> list:
    return [s for s in settings if by_id.get(f"{video_id}/{s.name}", {}).get("status") != "ok"]


def measure_video(
    video: sources.Video,
    todo: list,
    known: tuple | None,
    config: dict,
    protocol: dict,
    run: RunDir,
    pose,
    work_dir: Path,
    jobs: int,
) -> int:
    """Measure the remaining settings of one source video; returns how many succeeded."""
    base = {"video": video.id, "subject": video.subject}
    try:
        boxes, region_info, n_frames = known or sources.locate(video, pose, protocol)
        windows = pipeline.analysis_windows(n_frames / video.fps, protocol)
        refs, details = video.references(windows, n_frames)
    except Exception as err:  # recorded and retried on the next run
        traceback.print_exc()
        item = f"{video.id}/{compression.SOURCE.name}"
        run.append(
            {
                "id": item,
                "status": "error",
                **base,
                "setting": compression.SOURCE.name,
                "error": f"{type(err).__name__}: {err}",
            }
        )
        return 0
    relocate_names = set(config.get("relocate") or [])

    def relocate(frames, fps, size_hw):
        found, info, _ = sources.locate_frames(frames, fps, size_hw, pose, protocol)
        return found, info

    def one(setting: compression.Setting) -> dict:
        item = f"{video.id}/{setting.name}"
        try:
            record = pipeline.process_version(
                setting,
                video.frames,
                video.fps,
                video.size_hw,
                n_frames,
                boxes,
                windows,
                refs,
                protocol,
                work_dir / "versions",
                relocate=relocate if setting.name in relocate_names else None,
            )
        except Exception as err:  # recorded and retried on the next run
            traceback.print_exc()
            return {
                "id": item,
                "status": "error",
                **base,
                "setting": setting.name,
                "error": f"{type(err).__name__}: {err}",
            }
        record = {"id": item, "status": "ok", **base, **record}
        if setting.ladder == "source":
            record.update(regions={"boxes": boxes, **region_info}, reference=details, **video.info)
        return record

    succeeded = 0
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for future in as_completed([pool.submit(one, s) for s in todo]):
            record = future.result()
            run.append(pack(record))
            succeeded += record["status"] == "ok"
            seconds = record.get("seconds")
            took = "" if seconds is None else f" in {seconds:.0f} s"
            print(f"{record['id']}: {record['status']}{took}", flush=True)
    return succeeded


def run_ubfc(config, protocol, run, settings, pose, work_dir, jobs, limit) -> int:
    subjects, _ = ubfc.find_subjects(config["source"])
    if not subjects:
        raise SystemExit(
            f"no folder with {ubfc.VIDEO_NAME} and {ubfc.GT_NAME} under {config['source']}"
        )
    subjects = subjects[: config.get("n_videos") or len(subjects)]
    by_id = {r["id"]: r for r in run.records()}
    videos = 0
    for subject in subjects:
        todo = todo_settings(by_id, subject.id, settings)
        if not todo:
            continue
        if limit is not None and videos >= limit:
            break
        print(f"{subject.id}: {len(todo)} setting(s) to measure", flush=True)
        try:
            with sources.ubfc_video(subject, protocol, work_dir / "stage") as video:
                known = known_regions(by_id, subject.id)
                measure_video(video, todo, known, config, protocol, run, pose, work_dir, jobs)
        except Exception as err:  # e.g. the copy from Drive failed; retried on the next run
            traceback.print_exc()
            run.append(
                {
                    "id": f"{subject.id}/{compression.SOURCE.name}",
                    "status": "error",
                    "video": subject.id,
                    "subject": subject.id,
                    "setting": compression.SOURCE.name,
                    "error": f"{type(err).__name__}: {err}",
                }
            )
        videos += 1
    return videos


def run_scamps(config, protocol, run, settings, pose, work_dir, jobs, limit) -> int:
    """The first n_videos head-still videos of the archive (the phase-3 selection rule)."""
    sel = config["selection"]
    target = config.get("n_videos") or sel["n_videos"]
    records = run.records()
    by_id = {r["id"]: r for r in records}
    skipped = {r["id"] for r in records if r["status"] == "skipped"}
    selected = {r["video"] for r in records if "setting" in r}
    complete = {v for v in selected if not todo_settings(by_id, v, settings)}
    if len(complete) >= target:
        return 0
    seen = skipped | complete
    videos = 0
    members = sources.stream_members(config["source"], work_dir / "stage", seen)
    try:
        for name, local in members:
            if name not in selected:
                if len(selected) >= target:
                    break  # selection follows archive order: every selected video came earlier
                screen, still = sources.scamps_screen(name, local, config)
                if not still:
                    run.append({"id": name, "status": "skipped", "reason": "head_motion", **screen})
                    skipped.add(name)
                    seen.add(name)
                    if not selected and len(skipped) >= sel["give_up_after"]:
                        raise SystemExit(
                            f"none of the first {len(skipped)} videos has a still head; "
                            "see the head-pose ranges in items.jsonl"
                        )
                    continue
                selected.add(name)
            if limit is not None and videos >= limit:
                break
            todo = todo_settings(by_id, name, settings)
            print(f"{name}: {len(todo)} setting(s) to measure", flush=True)
            video = sources.scamps_video(name, local, config, protocol)
            known = known_regions(by_id, name)
            measure_video(video, todo, known, config, protocol, run, pose, work_dir, jobs)
            videos += 1
            by_id = {r["id"]: r for r in run.records()}
            if not todo_settings(by_id, name, settings):
                complete.add(name)
                seen.add(name)
            if len(complete) >= target:
                break
    finally:
        members.close()
    return videos


# ---------------------------------------------------------------------------------------
# Summary


def _source_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r["setting"] == compression.SOURCE.name]


def relocated_comparison(records: list[dict], protocol: dict) -> dict:
    """Regions found on the compressed video minus regions from the source, per setting."""
    tolerance = {
        m: protocol["tolerance_bpm"][cfg["kind"]] for m, cfg in protocol["measurements"].items()
    }
    out: dict = {}
    for r in records:
        if "relocated" not in r:
            continue
        for fixed, moved in zip(r["windows"], r["relocated"]["windows"], strict=True):
            if fixed["excluded"] is None and moved["excluded"] is None:
                cell = out.setdefault(r["setting"], {}).setdefault(fixed["measure"], [])
                cell.append((r["subject"], fixed, moved))
    result: dict = {}
    for setting, cells in out.items():
        for m, rows in cells.items():
            a = pipeline._arrays([f for _, f, _ in rows])
            b = pipeline._arrays([g for _, _, g in rows])
            result.setdefault(setting, {})[m] = metrics.paired_difference(
                a, b, [s for s, _, _ in rows], tolerance[m], protocol["statistics"]
            )
    return result


def timing(records: list[dict]) -> dict:
    per_setting: dict[str, list[float]] = {}
    per_video: dict[str, float] = {}
    for r in records:
        per_setting.setdefault(r["setting"], []).append(r["seconds"])
        per_video[r["video"]] = per_video.get(r["video"], 0.0) + r["seconds"]
    return {
        "median_seconds_per_setting": {k: float(np.median(v)) for k, v in per_setting.items()},
        "seconds_per_video": [float(v) for v in per_video.values()],
    }


def summarize(run: RunDir, config: dict, protocol: dict, settings: list) -> str:
    print("summarizing (bootstrap CIs; this can take a few minutes) ...", flush=True)
    records = [r for r in load_records(run) if "setting" in r]
    ok = [r for r in records if r["status"] == "ok"]
    errors = [r for r in records if r["status"] != "ok"]
    agg = pipeline.aggregate(ok, settings, protocol)
    # Breakdown points exist only for the CRF and bitrate ladders, so the subsets need only
    # those settings.
    ladder_settings = [s for s in settings if s.ladder in ("source", "crf", "bitrate")]
    names = {s.name for s in ladder_settings}
    sensitivity = {}
    if len(ladder_settings) > 1:
        for name, keep in baseline.sensitivity_subsets(config["dataset"], config).items():
            kept = {r["video"] for r in _source_records(ok) if keep(r)}
            subset = [r for r in ok if r["video"] in kept and r["setting"] in names]
            removed = len({r["video"] for r in ok}) - len(kept)
            sensitivity[name] = {
                "n_removed": removed,
                "breakdown": pipeline.aggregate(subset, ladder_settings, protocol)["breakdown"],
            }
    encoded = [r for r in ok if r.get("keyframes") is not None]
    summary = {
        "counts": {
            "videos": len({r["video"] for r in ok}),
            "versions_measured": len(ok),
            "errors": len(errors),
            "settings": len(settings),
        },
        **agg,
        "relocated_minus_fixed": relocated_comparison(ok, protocol),
        "sensitivity_breakdown": sensitivity,
        "timing": timing(ok),
        "encoded_versions_with_more_than_one_keyframe": sum(r["keyframes"] != 1 for r in encoded),
        "errors": [{"id": r["id"], "error": r.get("error")} for r in errors],
    }
    run.write_json("metrics.json", summary)
    text = markdown(summary, ok, settings, run.run_id, config)
    (run.path / "summary.md").write_text(text)
    return text


def markdown(summary: dict, ok: list[dict], settings: list, run_id: str, config: dict) -> str:
    fmt = pipeline._fmt
    counts = summary["counts"]
    lines = [
        f"# Compression experiment: {run_id}",
        "",
        f"Dataset: {baseline.DATASET_NAMES.get(config['dataset'], config['dataset'])}. "
        f"Ladders: {', '.join(config['ladders'])}. Videos: {counts['videos']}; versions measured: "
        f"{counts['versions_measured']} ({counts['settings']} settings per video); errors: "
        f"{counts['errors']}. Encoded versions with more than one keyframe: "
        f"{summary['encoded_versions_with_more_than_one_keyframe']}.",
        "",
    ]
    text = "\n".join(lines) + pipeline.summary_markdown(summary, ok, settings)
    lines = []
    if summary["relocated_minus_fixed"]:
        lines += [
            "",
            "## Regions found on the compressed video (sensitivity)",
            "",
            "Relocated minus fixed regions, on the same windows. `*`: 95% CI excludes zero.",
            "",
            "| Setting | Measure | Windows | Δ success rate (pp) | Δ MAE (bpm) |",
            "|---|---|---|---|---|",
        ]
        for setting, cells in summary["relocated_minus_fixed"].items():
            for m, cell in cells.items():
                cols = []
                for key, scale, digits in (("success_rate", 100, 1), ("mae", 1, 2)):
                    d, (lo, hi) = cell[key]["delta"], cell[key]["ci"]
                    star = "*" if cell[key]["significant"] else ""
                    cols.append(
                        f"{fmt(d * scale, digits)} [{fmt(lo * scale, digits)}, "
                        f"{fmt(hi * scale, digits)}]{star}"
                    )
                lines.append(f"| {setting} | {m} | {cell['n_windows']} | {cols[0]} | {cols[1]} |")
    lines += [
        "",
        "## Breakdown points without some videos (sensitivity)",
        "",
        "| Analysis | Videos removed | Ladder | Measure | Breakdown |",
        "|---|---|---|---|---|",
    ]
    for name, cell in summary["sensitivity_breakdown"].items():
        label = baseline.SENSITIVITY_LABELS.get(name, name)
        for ladder, measures in cell["breakdown"].items():
            for m, b in measures.items():
                point = b["breakdown"] or "none"
                lines.append(f"| {label} | {cell['n_removed']} | {ladder} | {m} | {point} |")
    t = summary["timing"]
    per_video = t["seconds_per_video"]
    time_text = "No versions measured yet."
    if per_video:
        time_text = (
            "Seconds per video, summed over its versions (with --jobs N the wall-clock time is "
            f"shorter, down to about 1/N): median {fmt(float(np.median(per_video)), 0)}, total "
            f"{fmt(float(np.sum(per_video)), 0)}."
        )
    lines += [
        "",
        "## Time",
        "",
        time_text,
        "",
        "| Setting | Median seconds per version |",
        "|---|---|",
    ]
    for s in settings:
        if s.name in t["median_seconds_per_setting"]:
            lines.append(f"| {s.name} | {fmt(t['median_seconds_per_setting'][s.name], 1)} |")
    if summary["errors"]:
        lines += ["", "## Errors", ""] + [f"- `{e['id']}`: {e['error']}" for e in summary["errors"]]
    return text + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, args.source)
    protocol = pipeline.load_protocol(args.protocol)
    settings = experiment_settings(config, protocol)
    run = RunDir(args.results_root, args.run_id)
    run.init({"experiment": config, "protocol": protocol})
    if args.summarize_only:
        print(summarize(run, config, protocol, settings), flush=True)
        return 0
    pose, pose_info = make_pose(config, args.work_dir)
    shared = LockedPose(pose) if args.jobs > 1 else pose
    extra = {**pose_info, "argv": sys.argv[1:], "jobs": args.jobs}
    run.add_session(environment_snapshot(PROJECT_DIR, extra))
    runner = {"ubfc": run_ubfc, "scamps": run_scamps}[config["dataset"]]
    try:
        videos = runner(
            config, protocol, run, settings, shared, args.work_dir, args.jobs, args.limit
        )
    finally:
        if hasattr(pose, "close"):
            pose.close()
        summarize(run, config, protocol, settings)
    print(f"worked on {videos} source video(s); results in {run.path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
