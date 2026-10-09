"""Signals extracted from video frames.

Phase 1 only needs a motion probe: the mean vertical and horizontal motion of a region,
estimated as one Lucas-Kanade flow vector for the whole region between consecutive frames
and summed over time. Static parts of the region (background) do not pull the estimate to
zero; they only scale it down. Phase 2 defines the actual measurements (M1-M3).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from breathing.roi import Box


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
