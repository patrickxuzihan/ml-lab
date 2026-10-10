"""Phase-1 data check: download or stage one video at a time, record its specs, delete it.

Results go to <results-root>/<run-id>/ (see breathing.runinfo). Re-running the same command
resumes: finished videos are skipped and failed ones are retried.

  python scripts/inspect_dataset.py --config configs/inspect_scamps.yaml \
      --run-id 20261012_scamps_inspect \
      --results-root /content/drive/MyDrive/camera-breathing/results
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import time
import traceback
from pathlib import Path

import yaml

from breathing import datacheck, roi
from breathing.datasets import scamps, ubfc
from breathing.runinfo import RunDir, environment_snapshot

PROJECT_DIR = Path(__file__).resolve().parents[1]
SYNTHETIC_DATASETS = {"scamps"}  # the only datasets whose frames may be saved (CLAUDE.md rule 3)
STREAM_ATTEMPTS = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_DIR / "work")
    parser.add_argument("--source", help="override the dataset location in the config")
    parser.add_argument("--limit", type=int, help="stop after this many new items")
    parser.add_argument("--no-pose", action="store_true", help="skip pose estimation")
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> dict:
    config = yaml.safe_load(args.config.read_text())
    if args.source:
        config["source"] = args.source
    if args.no_pose:
        config["pose"]["enabled"] = False
    if config.get("save_example_figure") and config["dataset"] not in SYNTHETIC_DATASETS:
        raise SystemExit(f"refusing to save frames of a real-person dataset ({config['dataset']})")
    return config


def make_pose(config: dict, work_dir: Path) -> tuple[roi.PoseEstimator | None, dict]:
    if not config["pose"]["enabled"]:
        return None, {}
    model = roi.ensure_model(config["pose"]["model_url"], work_dir / "models")
    return roi.PoseEstimator(model), {
        "pose_model": {"file": model.name, "sha256": roi.file_sha256(model)}
    }


def run_scamps(config: dict, run: RunDir, work_dir: Path, pose, limit: int | None) -> int:
    if not (run.path / "downloads.json").exists():
        run.write_json("downloads.json", [scamps.http_head(url) for url in config["head_urls"]])
    if config.get("labels_csv") and not (run.path / "labels_csv.json").exists():
        print("summarizing labels CSV archive ...", flush=True)
        try:
            run.write_json("labels_csv.json", scamps.summarize_csv_tar(config["labels_csv"]))
        except (OSError, EOFError, tarfile.TarError) as err:  # retried on the next run
            print(f"labels CSV summary failed: {err}", flush=True)

    done = run.done_ids()
    figure = run.path / "example_frame_regions.png"
    processed = 0
    for attempt in range(1, STREAM_ATTEMPTS + 1):
        try:
            members = scamps.iter_tar_members(config["source"], work_dir / "stage", skip=done)
            for name, local in members:
                save = config.get("save_example_figure") and not figure.exists()

                def check(local=local, fig=figure if save else None):
                    return datacheck.check_scamps_video(local, config, pose, fig)

                process(run, name, check)
                done.add(name)
                processed += 1
                if limit is not None and processed >= limit:
                    members.close()
                    return processed
            return processed
        except (OSError, EOFError, tarfile.TarError) as err:
            # A dropped download: the next attempt streams again and skips finished members.
            print(f"stream attempt {attempt} failed: {err}", flush=True)
            if attempt == STREAM_ATTEMPTS:
                raise
            time.sleep(10 * attempt)
    return processed


def run_ubfc(config: dict, run: RunDir, work_dir: Path, pose, limit: int | None) -> int:
    subjects, skipped = ubfc.find_subjects(config["source"])
    run.write_json(
        "folders.json", {"subjects": [s.id for s in subjects], "skipped_no_ground_truth": skipped}
    )
    if not subjects:
        raise SystemExit(
            f"no folder with {ubfc.VIDEO_NAME} and {ubfc.GT_NAME} under {config['source']}"
        )
    done = run.done_ids()
    stage = work_dir / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    processed = 0
    for subject in subjects:
        if subject.id in done:
            continue
        if limit is not None and processed >= limit:
            break

        def check(subject=subject):
            local = stage / ubfc.VIDEO_NAME
            started = time.perf_counter()
            shutil.copyfile(subject.video, local)  # copy from the Drive shortcut to local disk
            try:
                record = datacheck.check_ubfc_subject(
                    local, ubfc.read_ground_truth(subject.ground_truth), config, pose
                )
            finally:
                local.unlink(missing_ok=True)
            record["timing_s"]["copy_and_check"] = time.perf_counter() - started
            return record

        process(run, subject.id, check)
        processed += 1
    return processed


def process(run: RunDir, item_id: str, check) -> None:
    print(f"checking {item_id} ...", flush=True)
    try:
        record = {"id": item_id, "status": "ok", **check()}
    except Exception as err:  # keep going; the error is recorded and retried next run
        traceback.print_exc()
        record = {"id": item_id, "status": "error", "error": f"{type(err).__name__}: {err}"}
    run.append(record)


def write_summary(run: RunDir, dataset: str) -> None:
    records = run.records()
    datacheck.write_items_csv(records, run.path / "items.csv")
    summary, markdown = datacheck.summarize(records, dataset)
    run.write_json("summary.json", summary)
    (run.path / "summary.md").write_text(markdown)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args)
    run = RunDir(args.results_root, args.run_id)
    run.init(config)
    pose, pose_info = make_pose(config, args.work_dir)
    run.add_session(environment_snapshot(PROJECT_DIR, {**pose_info, "argv": sys.argv[1:]}))
    runner = {"scamps": run_scamps, "ubfc": run_ubfc}[config["dataset"]]
    try:
        processed = runner(config, run, args.work_dir, pose, args.limit)
    finally:
        if pose is not None:
            pose.close()
        write_summary(run, config["dataset"])
    print(f"checked {processed} new item(s); results in {run.path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
