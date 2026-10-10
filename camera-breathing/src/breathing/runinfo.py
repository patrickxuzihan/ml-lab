"""Run directories: effective config, environment record and a resumable per-item log.

Layout of results/<run_id>/:
  config.yaml      effective config; a resumed run must use the same one
  manifest.json    one environment snapshot per session (git commit, versions, hardware)
  items.jsonl      one JSON record per processed item, appended and flushed immediately
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

from breathing.video_io import tool_version

RUN_ID_PATTERN = re.compile(r"^\d{8}_[a-z0-9]+_[a-z0-9-]+(_s\d+)?$")

PACKAGES = ("numpy", "scipy", "h5py", "PyYAML", "mediapipe", "opencv-contrib-python")


class ConfigMismatchError(RuntimeError):
    pass


def check_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.match(run_id):
        raise ValueError(f"run id {run_id!r} must look like 20261101_scamps_x264-ladder[_s0]")
    return run_id


def git_info(repo_dir: str | Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _gpus() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def environment_snapshot(repo_dir: str | Path, extra: dict | None = None) -> dict:
    snapshot = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(repo_dir),
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "ffmpeg": tool_version("ffmpeg"),
        "ffprobe": tool_version("ffprobe"),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "memory_bytes": _memory_bytes(),
            "gpus": _gpus(),
        },
    }
    if extra:
        snapshot.update(extra)
    return snapshot


class RunDir:
    def __init__(self, results_root: str | Path, run_id: str):
        self.run_id = check_run_id(run_id)
        self.path = Path(results_root) / run_id
        self.config_path = self.path / "config.yaml"
        self.manifest_path = self.path / "manifest.json"
        self.items_path = self.path / "items.jsonl"

    def init(self, config: dict) -> None:
        """Create the run, or check that a resumed run uses the same config."""
        self.path.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            saved = yaml.safe_load(self.config_path.read_text())
            if saved != config:
                raise ConfigMismatchError(
                    f"{self.config_path} differs from the current config; "
                    "use a new run id or restore the original config"
                )
        else:
            self.config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    def add_session(self, snapshot: dict) -> None:
        manifest = {"run_id": self.run_id, "sessions": []}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
        manifest["sessions"].append(snapshot)
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def records(self) -> list[dict]:
        """Latest record per item id, in first-seen order."""
        latest: dict[str, dict] = {}
        if self.items_path.exists():
            for line in self.items_path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    latest[record["id"]] = record
        return list(latest.values())

    def done_ids(self) -> set[str]:
        return {r["id"] for r in self.records() if r.get("status") == "ok"}

    def append(self, record: dict) -> None:
        with open(self.items_path, "a") as f:
            f.write(json.dumps(to_jsonable(record), allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_json(self, name: str, data) -> None:
        text = json.dumps(to_jsonable(data), indent=2, allow_nan=False)
        (self.path / name).write_text(text + "\n")


def to_jsonable(value):
    """Convert numpy types to plain Python and non-finite floats to None."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):  # numpy arrays and scalars
        return to_jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
