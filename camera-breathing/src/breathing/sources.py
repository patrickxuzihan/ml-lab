"""Source videos of each dataset for the experiment scripts, staged one at a time.

Thin adapters around the dataset readers. Each source video becomes a `Video`: a way to
re-read its frames, its frame rate and size, a function that computes its references for any
analysis windows, and what to record about it. `locate_face` and `cropped` turn a video into
its face-only version (phase 5).

- UBFC-rPPG: `ubfc_video` copies one subject's video from the dataset folder (the Drive
  shortcut in Colab) to local disk and deletes it afterwards.
- SCAMPS: `stream_members` streams the archive one member at a time; `scamps_screen` reads
  the head-pose and facial-action labels without loading the frames, and `scamps_video` loads
  a selected video.
"""

from __future__ import annotations

import shutil
import tarfile
import time
from collections.abc import Callable, Container, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

import numpy as np

from breathing import baseline, datacheck, pipeline, roi
from breathing.datasets import scamps, ubfc
from breathing.video_io import iter_frames, probe

STREAM_ATTEMPTS = 3


@dataclass
class Video:
    id: str  # item id prefix: UBFC subject folder or SCAMPS archive member
    subject: str  # bootstrap cluster
    fps: float
    size_hw: tuple[int, int]
    frames: Callable[[], Iterator[np.ndarray]]  # RGB frames; may be called several times
    references: Callable[[dict, int], tuple[dict, dict]]  # (windows, n_frames) -> (refs, details)
    info: dict = field(default_factory=dict)  # recorded with the source measurement
    labels: dict = field(default_factory=dict)  # per-frame label waveforms (SCAMPS d_br, d_ppg)


def make_pose(config: dict, work_dir: Path):
    """MediaPipe pose estimator (model cached in work_dir) and a record of the model file."""
    model = roi.ensure_model(config["pose_model_url"], Path(work_dir) / "models")
    info = {"pose_model": {"file": model.name, "sha256": roi.file_sha256(model)}}
    return roi.PoseEstimator(model), info


def locate_frames(
    frames: Iterable[np.ndarray], fps: float, size_hw: tuple[int, int], pose, protocol: dict
) -> tuple[dict, dict, int]:
    """Measurement regions from pose sampled over the frames: (boxes, info, frames seen)."""
    reg = protocol["regions"]
    height, width = size_hw
    landmarks, n_frames, _ = datacheck.sample_pose(frames, fps, reg["pose_sample_fps"], pose)
    boxes, info = roi.measurement_regions(
        landmarks, width, height, reg["visibility_threshold"], reg["min_shoulder_frac"]
    )
    return boxes, info, n_frames


def locate(video: Video, pose, protocol: dict) -> tuple[dict, dict, int]:
    """Measurement regions of a source video (found once, on the uncompressed frames)."""
    return locate_frames(video.frames(), video.fps, video.size_hw, pose, protocol)


def locate_face(video: Video, pose, protocol: dict) -> tuple[roi.Box | None, dict, dict, int]:
    """Face-only crop of a source video and the regions inside it, from pose on the
    uncompressed frames: (crop, boxes in crop coordinates, info, frames seen)."""
    reg = protocol["regions"]
    height, width = video.size_hw
    landmarks, n_frames, _ = datacheck.sample_pose(
        video.frames(), video.fps, reg["pose_sample_fps"], pose
    )
    crop, boxes, info = roi.face_only_regions(
        landmarks,
        width,
        height,
        reg["visibility_threshold"],
        reg["min_shoulder_frac"],
        reg["face_only"],
    )
    return crop, boxes, info, n_frames


def cropped(video: Video, crop: roi.Box) -> Video:
    """The video cut to the crop box: what a camera framing only the face would record."""
    r0, r1, c0, c1 = crop
    frames = video.frames
    return replace(
        video, size_hw=(r1 - r0, c1 - c0), frames=lambda: (f[r0:r1, c0:c1] for f in frames())
    )


@contextmanager
def ubfc_video(subject: ubfc.Subject, protocol: dict, stage_dir: Path) -> Iterator[Video]:
    """One UBFC-rPPG subject, copied to local disk for the duration of the block."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    local = stage_dir / ubfc.VIDEO_NAME
    started = time.perf_counter()
    shutil.copyfile(subject.video, local)
    copied = time.perf_counter() - started
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

        yield Video(
            id=subject.id,
            subject=subject.id,
            fps=info.fps,
            size_hw=(info.height, info.width),
            frames=frames,
            references=references,
            info={
                "video_info": info.to_dict(),
                "flags": ubfc.timestamp_flags(gt["t"], protocol["reference"]["ubfc_gap_flag_s"]),
                "timing_s": {"copy": copied},
            },
        )
    finally:
        local.unlink(missing_ok=True)


def scamps_screen(name: str, local: Path, config: dict) -> tuple[dict, bool]:
    """Head-pose ranges and facial-action activity of one SCAMPS video (frames not loaded),
    and whether it belongs to the head-still subset."""
    sel = config["selection"]
    head_keys = sel["head_pose_keys"]
    waves = scamps.read_waveforms(local, head_keys + sel["action_unit_keys"])
    missing = [k for k in head_keys if k not in waves]
    if missing:
        raise ValueError(f"head-pose variables missing: {missing}")
    head = baseline.value_ranges({k: waves[k] for k in head_keys})
    units = baseline.value_ranges({k: v for k, v in waves.items() if k not in head_keys})
    screen = {
        "video": name,
        "subject": PurePosixPath(name).stem,  # no identity labels: one cluster per video
        "head_pose_range": head,
        "action_units_max_range": max(units.values()) if units else None,
    }
    return screen, baseline.head_is_still(head, sel["max_head_pose_range"])


def scamps_video(name: str, local: Path, config: dict, protocol: dict) -> Video:
    """A SCAMPS video loaded into memory, with references from its d_br and d_ppg labels."""
    video = scamps.read_video(local, config["frames_key"], config["waveform_keys"])
    fps = float(config["fps"])
    frames = video.frames
    known = {"breathing": video.waveforms["breathing"], "heart": video.waveforms["ppg"]}

    def references(windows, n_frames):
        if len(known["breathing"]) != n_frames:
            raise ValueError(f"{len(known['breathing'])} label samples for {n_frames} frames")
        return pipeline.waveform_references(known, fps, windows, protocol), {}

    return Video(
        id=name,
        subject=PurePosixPath(name).stem,
        fps=fps,
        size_hw=tuple(frames.shape[1:3]),
        frames=lambda: iter(frames),
        references=references,
        info={"layout": video.layout},
        labels=dict(video.waveforms),
    )


def stream_members(
    source: str, stage_dir: Path, seen: Container[str], attempts: int = STREAM_ATTEMPTS
) -> Iterator[tuple[str, Path]]:
    """Archive members not in `seen`, written to stage_dir one at a time.

    A dropped download restarts the stream; the caller adds finished members to `seen`, so a
    restart skips them (they are still downloaded, since gzip can only be read from the start).
    """
    for attempt in range(1, attempts + 1):
        try:
            yield from scamps.iter_tar_members(source, stage_dir, skip=seen)
            return
        except (OSError, EOFError, tarfile.TarError) as err:
            print(f"stream attempt {attempt} failed: {err}", flush=True)
            if attempt == attempts:
                raise
            time.sleep(10 * attempt)
