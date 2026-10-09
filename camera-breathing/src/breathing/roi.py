"""Regions of interest from body-pose landmarks.

Landmarks are arrays of shape (n_frames, 33, 3) holding MediaPipe's normalized x, y and
visibility for each pose landmark; frames without a detected pose are all NaN. Everything
except `PoseEstimator` and `ensure_model` is pure numpy.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

N_LANDMARKS = 33
NOSE = 0
FACE = tuple(range(11))  # nose, eyes, ears, mouth
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

Box = tuple[int, int, int, int]  # (row0, row1, col0, col1), half-open


def visible(landmarks: np.ndarray, index: int, threshold: float) -> np.ndarray:
    """Per frame: was the landmark detected, confident and inside the image?"""
    x, y, vis = landmarks[:, index, 0], landmarks[:, index, 1], landmarks[:, index, 2]
    with np.errstate(invalid="ignore"):
        return (vis >= threshold) & (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)


def _median(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def _min(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.min(values)) if values.size else None


def visibility_summary(
    landmarks: np.ndarray, width: int, height: int, threshold: float = 0.5
) -> dict:
    """How much of the face, shoulders and chest is in view, as numbers only."""
    n = landmarks.shape[0]
    detected = np.isfinite(landmarks[:, NOSE, 0])
    nose = visible(landmarks, NOSE, threshold)
    shoulders = visible(landmarks, LEFT_SHOULDER, threshold) & visible(
        landmarks, RIGHT_SHOULDER, threshold
    )
    hips = visible(landmarks, LEFT_HIP, threshold) | visible(landmarks, RIGHT_HIP, threshold)

    # Geometry is measured only on frames where both shoulders are visible.
    sh = landmarks[shoulders]
    shoulder_y = (sh[:, LEFT_SHOULDER, 1] + sh[:, RIGHT_SHOULDER, 1]) / 2
    shoulder_width_px = np.abs(sh[:, LEFT_SHOULDER, 0] - sh[:, RIGHT_SHOULDER, 0]) * width
    rows_below_px = (1.0 - shoulder_y) * height
    with np.errstate(divide="ignore", invalid="ignore"):
        rows_below_ratio = rows_below_px / shoulder_width_px

    face = landmarks[detected][:, FACE, :2]
    face_width_px = (np.nanmax(face[..., 0], axis=1) - np.nanmin(face[..., 0], axis=1)) * width

    def frac(mask: np.ndarray) -> float | None:
        return float(mask.mean()) if n else None

    return {
        "n_sampled": int(n),
        "pose_detected_frac": frac(detected),
        "nose_visible_frac": frac(nose),
        "shoulders_visible_frac": frac(shoulders),
        "hip_visible_frac": frac(hips),
        "shoulder_y_frac_median": _median(shoulder_y),
        "shoulder_width_px_median": _median(shoulder_width_px),
        "rows_below_shoulders_px_median": _median(rows_below_px),
        "rows_below_shoulders_px_min": _min(rows_below_px),
        "rows_below_shoulders_per_shoulder_width_median": _median(rows_below_ratio),
        "face_width_px_median": _median(face_width_px),
    }


def _clip_box(r0: float, r1: float, c0: float, c1: float, height: int, width: int) -> Box | None:
    box = (
        int(np.clip(np.floor(r0), 0, height)),
        int(np.clip(np.ceil(r1), 0, height)),
        int(np.clip(np.floor(c0), 0, width)),
        int(np.clip(np.ceil(c1), 0, width)),
    )
    if box[1] - box[0] < 12 or box[3] - box[2] < 12:  # too small for the motion probe
        return None
    return box


def probe_regions(
    landmarks: np.ndarray, width: int, height: int, threshold: float = 0.5
) -> tuple[dict[str, Box | None], str]:
    """Fixed head, shoulder and chest boxes from the median pose over the sampled frames.

    The shoulder box straddles the shoulder line; the chest box lies below it. Box sizes
    scale with the shoulder width. Falls back to fixed fractions of the frame when the
    shoulders are not visible in at least half of the frames. Returns (boxes, source).
    """
    both = visible(landmarks, LEFT_SHOULDER, threshold) & visible(
        landmarks, RIGHT_SHOULDER, threshold
    )
    if landmarks.shape[0] == 0 or both.mean() < 0.5:
        h, w = height, width
        return {
            "head": _clip_box(h / 6, h / 2, w / 4, 3 * w / 4, h, w),
            "shoulder": _clip_box(h / 2, 2 * h / 3, 0, w, h, w),
            "chest": _clip_box(2 * h / 3, h, w / 6, 5 * w / 6, h, w),
        }, "fallback"

    med = np.full((N_LANDMARKS, 3), np.nan)
    used = [*FACE, LEFT_SHOULDER, RIGHT_SHOULDER]
    med[used] = np.median(landmarks[both][:, used], axis=0)
    lx, ly = med[LEFT_SHOULDER, 0] * width, med[LEFT_SHOULDER, 1] * height
    rx, ry = med[RIGHT_SHOULDER, 0] * width, med[RIGHT_SHOULDER, 1] * height
    sw = max(abs(lx - rx), 1.0)
    sy = (ly + ry) / 2
    cx = (lx + rx) / 2
    left, right = min(lx, rx), max(lx, rx)

    face = med[list(FACE), :2] * [width, height]
    fx0, fx1 = np.nanmin(face[:, 0]), np.nanmax(face[:, 0])
    fy = np.nanmean(face[:, 1])
    face_w = max(fx1 - fx0, 0.2 * sw)

    return {
        "head": _clip_box(
            fy - 0.8 * face_w,
            fy + 0.6 * face_w,
            fx0 - 0.2 * face_w,
            fx1 + 0.2 * face_w,
            height,
            width,
        ),
        "shoulder": _clip_box(
            sy - 0.2 * sw, sy + 0.2 * sw, left - 0.15 * sw, right + 0.15 * sw, height, width
        ),
        "chest": _clip_box(
            sy + 0.1 * sw, sy + 0.7 * sw, cx - 0.35 * sw, cx + 0.35 * sw, height, width
        ),
    }, "pose"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(url: str, cache_dir: str | Path) -> Path:
    """Download the model file once into cache_dir (never into git)."""
    path = Path(cache_dir) / Path(url).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


class PoseEstimator:
    """Thin adapter around MediaPipe's PoseLandmarker (Tasks API, image mode)."""

    def __init__(self, model_path: str | Path):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Landmarks (33, 3) of the first detected person, or all NaN."""
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        result = self._landmarker.detect(image)
        out = np.full((N_LANDMARKS, 3), np.nan)
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks[0][:N_LANDMARKS]):
                out[i] = (lm.x, lm.y, lm.visibility if lm.visibility is not None else np.nan)
        return out

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
