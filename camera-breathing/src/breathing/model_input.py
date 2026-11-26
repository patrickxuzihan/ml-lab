"""Inputs of the phase-5 deep model, and the compressed versions it is trained on.

A version's decoded frames are downscaled to size x size by area averaging and resampled to a
fixed frame rate: output frame k is the mean of the input frames whose time (index / fps)
falls in [k, k + 1) / fps_out. The label waveforms are resampled the same way, so frames and
labels share one time base whatever the version's frame rate. Pure numpy; no torch.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable

import numpy as np

from breathing.compression import Setting


def n_output(n_frames: int, fps: float, fps_out: float) -> int:
    """Output frames for a video of n_frames at fps: whole 1/fps_out intervals only."""
    return int(np.floor(n_frames / fps * fps_out + 1e-9))


def area_axis(values: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Resample one axis to n_out samples by area: each output sample is the mean of the input
    over its 1/n_out share of the axis, pixels being constant over their width. Computed from
    cumulative sums (no matrix products, so no BLAS threads competing with --jobs)."""
    x = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    n_in = x.shape[0]
    cum = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    edges = np.arange(n_out + 1) * (n_in / n_out)  # output edges in input pixels
    k = np.minimum(np.floor(edges).astype(int), n_in - 1)
    frac = (edges - k).reshape((-1,) + (1,) * (x.ndim - 1))
    integral = cum[k] + frac * x[k]  # of the input from 0 to each edge
    return np.moveaxis((integral[1:] - integral[:-1]) * (n_out / n_in), 0, axis)


def resize(frame: np.ndarray, size: int) -> np.ndarray:
    """(H, W, C) frame resampled to (size, size, C) by area (float64)."""
    return area_axis(area_axis(frame, size, 0), size, 1)


def _bin(i: int, fps: float, fps_out: float) -> int:
    return int(np.floor(i * fps_out / fps + 1e-9))


def _fill_empty(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Mean per bin; a bin without samples (input slower than the output) repeats the last
    bin that has some."""
    filled = np.flatnonzero(counts)
    if filled.size == 0:
        raise ValueError("no samples")
    last = np.searchsorted(filled, np.arange(len(counts)), side="right") - 1
    source = filled[np.clip(last, 0, None)]
    shape = (-1,) + (1,) * (sums.ndim - 1)
    return sums[source] / counts[source].reshape(shape)


def model_frames(
    frames: Iterable[np.ndarray], fps: float, n_out: int, fps_out: float, size: int
) -> np.ndarray:
    """(n_out, size, size, 3) uint8 model input from RGB frames at fps."""
    sums = np.zeros((n_out, size, size, 3))
    counts = np.zeros(n_out, dtype=int)
    for i, frame in enumerate(frames):
        k = _bin(i, fps, fps_out)
        if k >= n_out:
            continue  # past the last whole interval; still read so the decoder finishes
        sums[k] += resize(frame, size)
        counts[k] += 1
    return np.clip(np.rint(_fill_empty(sums, counts)), 0, 255).astype(np.uint8)


def model_signal(values: np.ndarray, fps: float, n_out: int, fps_out: float) -> np.ndarray:
    """A per-frame label waveform on the model's time base (float32)."""
    values = np.asarray(values, dtype=np.float64)
    bins = np.floor(np.arange(len(values)) * fps_out / fps + 1e-9).astype(int)
    keep = bins < n_out
    sums = np.bincount(bins[keep], weights=values[keep], minlength=n_out)
    counts = np.bincount(bins[keep], minlength=n_out)
    return _fill_empty(sums, counts).astype(np.float32)


# ---------------------------------------------------------------------------------------
# Compressed versions for training


def lossy(settings: Iterable[Setting]) -> list[Setting]:
    """Settings that change the picture through compression: no source, no uncompressed
    decimation or downscaling (rawvideo) and no lossless CRF 0."""
    return [s for s in settings if s.codec not in (None, "rawvideo") and s.crf != 0]


def augmentation_pools(settings: list[Setting], groups: list[dict]) -> list[list[Setting]]:
    """The settings each group draws from: the lossy settings of its ladders. Allocation
    cells at full frame rate and size are left out: they repeat bitrate rungs."""
    pools = []
    for group in groups:
        pool = [
            s
            for s in lossy(settings)
            if s.ladder in group["ladders"]
            and not (s.ladder == "allocation" and (s.decimate, s.scale) == (1, 1))
        ]
        if len(pool) < group["n"]:
            raise ValueError(
                f"group {group['ladders']} has {len(pool)} settings, needs {group['n']}"
            )
        pools.append(pool)
    return pools


def draw_settings(
    pools: list[list[Setting]], counts: list[int], seed: int, video_id: str
) -> list[Setting]:
    """counts[i] settings drawn without replacement from pools[i]. The draw depends only on
    the seed and the video, so a resumed or repeated run makes the same versions."""
    rng = np.random.default_rng([seed, zlib.crc32(video_id.encode())])
    drawn = []
    for pool, n in zip(pools, counts, strict=True):
        drawn += [pool[i] for i in sorted(rng.choice(len(pool), size=n, replace=False))]
    return drawn
