"""Confirm the SCAMPS frame rate from the time column of the labels CSV archive.

SCAMPS files do not store a frame rate; the protocol assumes 30 fps (notes/01-data.md 4.4).
The labels archive (about 60 MB) has one CSV per video with one row per frame and a time
column `t_ppg`. This script streams the archive and records, per file, the number of rows
and the sampling rate implied by the time column. Only numbers are kept.

  python scripts/check_scamps_timing.py --run-id 20261122_scamps_timing \
      --results-root /content/drive/MyDrive/camera-breathing/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from breathing.datacheck import timing_stats
from breathing.datasets import scamps
from breathing.runinfo import RunDir, environment_snapshot

PROJECT_DIR = Path(__file__).resolve().parents[1]
LABELS_CSV = (
    "https://facesyntheticspubwedata.z6.web.core.windows.net/neurips-2022/"
    "scamps_waveforms_csv.tar.gz"
)
ASSUMED_FPS = 30.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", default=LABELS_CSV, help="URL or local path of the archive")
    parser.add_argument("--column", default="t_ppg")
    parser.add_argument("--results-root", type=Path, default=PROJECT_DIR / "results")
    return parser.parse_args(argv)


def file_record(name: str, t: np.ndarray | None) -> dict:
    if t is None:
        return {"id": name, "status": "error", "error": "time column missing"}
    stats = timing_stats(t)
    mean_fs = (len(t) - 1) / stats["duration_s"] if stats.get("duration_s") else None
    return {
        "id": name,
        "status": "ok",
        **stats,
        "t_first_s": float(t[0]) if len(t) else None,
        "fs_mean_hz": mean_fs,
    }


def summary_markdown(records: list[dict]) -> tuple[dict, str]:
    ok = [r for r in records if r["status"] == "ok"]

    def spread(key: str) -> list[float | None]:
        values = np.array([r[key] for r in ok if r.get(key) is not None], dtype=np.float64)
        if not values.size:
            return [None, None, None]
        return [float(values.min()), float(np.median(values)), float(values.max())]

    off = [
        r["id"] for r in ok if r["fs_mean_hz"] is None or abs(r["fs_mean_hz"] - ASSUMED_FPS) > 0.01
    ]
    summary = {
        "files": len(records),
        "errors": len(records) - len(ok),
        "rows": spread("n"),
        "fs_median_dt_hz": spread("fs_hz"),
        "fs_mean_hz": spread("fs_mean_hz"),
        "first_time_s": spread("t_first_s"),
        "duration_s": spread("duration_s"),
        "all_monotonic": all(r.get("monotonic", False) for r in ok),
        "files_off_assumed_fps": off,
    }

    def f(values: list) -> str:
        return " / ".join("–" if v is None else f"{v:.6g}" for v in values)

    lines = [
        "# SCAMPS frame rate from the labels CSV time column",
        "",
        "Cells show min / median / max over files.",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Files | {summary['files']} |",
        f"| Errors | {summary['errors']} |",
        f"| Rows per file | {f(summary['rows'])} |",
        f"| Rate from median interval (Hz) | {f(summary['fs_median_dt_hz'])} |",
        f"| Rate from (rows - 1) / duration (Hz) | {f(summary['fs_mean_hz'])} |",
        f"| First time value (s) | {f(summary['first_time_s'])} |",
        f"| Duration, first to last row (s) | {f(summary['duration_s'])} |",
        f"| Time strictly increasing in every file | {summary['all_monotonic']} |",
        f"| Files more than 0.01 Hz from {ASSUMED_FPS:g} fps | {len(off)} |",
    ]
    return summary, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run = RunDir(args.results_root, args.run_id)
    run.init({"source": args.source, "column": args.column, "assumed_fps": ASSUMED_FPS})
    run.add_session(environment_snapshot(PROJECT_DIR, {"argv": sys.argv[1:]}))
    done = run.done_ids()
    for name, values in scamps.iter_csv_column(args.source, args.column):
        if name not in done:
            run.append(file_record(name, values))
    summary, markdown = summary_markdown(run.records())
    run.write_json("summary.json", summary)
    (run.path / "summary.md").write_text(markdown)
    print(markdown, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
