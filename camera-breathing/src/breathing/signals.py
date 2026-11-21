"""Signals extracted from video frames.

Motion: the mean motion of a region, estimated as one Lucas-Kanade flow vector for the whole
region between consecutive frames and summed over time. Static parts of the region
(background) do not pull the estimate to zero; they only scale it down. M1 uses the vertical
component in the shoulder band.

Color: the mean RGB of a skin region per frame, turned into a pulse signal with POS
(Wang et al., "Algorithmic Principles of Remote PPG", IEEE TBME 2017). M2 and M3 use it.

`extract_traces` streams frames and keeps only the per-frame numbers, so a long video never
has to be held in memory.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from breathing.roi import Box

# POS projection onto the plane orthogonal to the normalized skin tone (Wang et al. 2017).
POS_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


def to_gray(frames: np.ndarray) -> np.ndarray:
    """(…, H, W, 3) uint8 RGB -> (…, H, W) float32 luma (BT.601)."""
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return frames.astype(np.float32) @ weights


def global_flow(
    previous: np.ndarray, current: np.ndarray, sigma: float = 1.0
) -> tuple[float, float]:
    """Least-squares (dy, dx) motion from `previous` to `current`; positive dy = down.

    Valid for small motions (about a pixel or less), as between consecutive video frames.
    """
    a = np.asarray(previous, dtype=np.float64)
    b = np.asarray(current, dtype=np.float64)
    mean = (a + b) / 2
    # Derivative-of-Gaussian gradients: the same smoothing in space and time, and no
    # finite-difference bias for fine texture.
    gy = ndimage.gaussian_filter(mean, sigma, order=(1, 0))
    gx = ndimage.gaussian_filter(mean, sigma, order=(0, 1))
    gt = ndimage.gaussian_filter(b - a, sigma)
    inner = (slice(3, -3), slice(3, -3))  # drop borders affected by smoothing
    gy, gx, gt = gy[inner].ravel(), gx[inner].ravel(), gt[inner].ravel()
    normal = np.array([[gy @ gy, gy @ gx], [gx @ gy, gx @ gx]])
    rhs = -np.array([gy @ gt, gx @ gt])
    (dy, dx), *_ = np.linalg.lstsq(normal, rhs, rcond=None)
    return float(dy), float(dx)


def region_shift(gray: np.ndarray, box: Box) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (dy, dx) of a fixed region relative to its first frame. gray: (T, H, W)."""
    r0, r1, c0, c1 = box
    roi = gray[:, r0:r1, c0:c1]
    steps = np.array([global_flow(roi[i - 1], roi[i]) for i in range(1, len(roi))])
    steps = steps.reshape(-1, 2)
    zero = np.zeros(1)
    return np.concatenate([zero, np.cumsum(steps[:, 0])]), np.concatenate(
        [zero, np.cumsum(steps[:, 1])]
    )


def pos_projection(rgb: np.ndarray) -> np.ndarray:
    """POS on one window of mean RGB values (T, 3): zero-mean pulse signal (T,)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    mean = rgb.mean(axis=0)
    if rgb.shape[0] < 2 or np.any(mean <= 0):
        return np.zeros(rgb.shape[0])
    s1, s2 = POS_PROJECTION @ (rgb / mean).T
    std2 = s2.std()
    h = s1 + (s1.std() / std2) * s2 if std2 > 0 else s1
    return h - h.mean()


def pos(rgb: np.ndarray, fs: float, window_s: float | None) -> np.ndarray:
    """POS pulse signal from mean RGB values (T, 3).

    With `window_s`, projections over sliding windows of that length are overlap-added, as in
    the original algorithm; each window's temporal normalization acts as a high-pass filter
    with a cutoff around 1/window_s. With None, one projection covers the whole input.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    n = rgb.shape[0]
    if window_s is None:
        return pos_projection(rgb)
    length = max(2, int(np.ceil(window_s * fs)))
    if length >= n:
        return pos_projection(rgb)
    out = np.zeros(n)
    for start in range(n - length + 1):
        out[start : start + length] += pos_projection(rgb[start : start + length])
    return out


@dataclass
class Traces:
    """Per-frame signals of one video version."""

    fps: float
    n_frames: int
    motion: dict[str, np.ndarray | None]  # cumulative vertical displacement, px (+ = down)
    rgb: dict[str, np.ndarray | None]  # (T, 3) mean RGB of the region


def extract_traces(
    frames: Iterable[np.ndarray],
    fps: float,
    motion_boxes: dict[str, Box | None],
    color_boxes: dict[str, Box | None],
) -> Traces:
    """Stream RGB frames once; keep each region's vertical motion and mean color per frame."""
    previous: dict[str, np.ndarray] = {}
    steps: dict[str, list[float]] = {k: [] for k, b in motion_boxes.items() if b is not None}
    colors: dict[str, list[np.ndarray]] = {k: [] for k, b in color_boxes.items() if b is not None}
    n = 0
    for frame in frames:
        for name in steps:
            r0, r1, c0, c1 = motion_boxes[name]
            crop = to_gray(frame[r0:r1, c0:c1])
            if name in previous:
                steps[name].append(_vertical_step(previous[name], crop))
            previous[name] = crop
        for name in colors:
            r0, r1, c0, c1 = color_boxes[name]
            colors[name].append(frame[r0:r1, c0:c1].reshape(-1, 3).mean(axis=0, dtype=np.float64))
        n += 1
    motion: dict[str, np.ndarray | None] = {name: None for name in motion_boxes}
    for name, values in steps.items():
        motion[name] = np.concatenate([[0.0], np.cumsum(values)]) if n else np.zeros(0)
    rgb: dict[str, np.ndarray | None] = {name: None for name in color_boxes}
    for name, values in colors.items():
        rgb[name] = np.array(values).reshape(-1, 3)
    return Traces(fps=fps, n_frames=n, motion=motion, rgb=rgb)


def _vertical_step(previous: np.ndarray, current: np.ndarray) -> float:
    """Vertical flow between two crops; 0 when the crop is too small for the estimate."""
    if min(previous.shape) < 8:
        return 0.0
    return global_flow(previous, current)[0]
