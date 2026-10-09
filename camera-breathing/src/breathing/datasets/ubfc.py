"""UBFC-rPPG reader (Bobbia et al., 2019), DATASET_2 layout.

Each subject folder holds `vid.avi` and `ground_truth.txt`. The text file has three lines:
the pulse-oximeter PPG waveform, the oximeter's heart rate, and the time of each sample in
seconds, reportedly one sample per video frame (the phase-1 data check verifies this).
Folders without `ground_truth.txt` (DATASET_1 uses a different format) are reported as
skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

VIDEO_NAME = "vid.avi"
GT_NAME = "ground_truth.txt"


@dataclass(frozen=True)
class Subject:
    id: str  # path relative to the dataset root, e.g. "DATASET_2/subject1"
    video: Path
    ground_truth: Path


def find_subjects(root: str | Path) -> tuple[list[Subject], list[str]]:
    """All subject folders under root, and folders that have a video but no ground truth."""
    root = Path(root)
    subjects, skipped = [], []
    for video in sorted(root.rglob(VIDEO_NAME)):
        folder = video.parent
        rel = folder.relative_to(root).as_posix()
        gt = folder / GT_NAME
        if gt.exists():
            subjects.append(Subject(id=rel, video=video, ground_truth=gt))
        else:
            skipped.append(rel)
    return subjects, skipped


def read_ground_truth(path: str | Path) -> dict[str, np.ndarray]:
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"{path}: expected 3 lines (PPG, heart rate, time), got {len(lines)}")
    ppg, hr, t = (np.array(line.split(), dtype=np.float64) for line in lines[:3])
    if not len(ppg) == len(hr) == len(t):
        raise ValueError(f"{path}: line lengths differ ({len(ppg)}, {len(hr)}, {len(t)})")
    return {"ppg": ppg, "hr": hr, "t": t}
