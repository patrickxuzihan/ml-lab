"""Band-pass filtering and spectral peak frequency.

Minimal versions for the phase-1 data checks. Phase 2 fixes the evaluation protocol
(windows, interpolation, tolerances) and may change these.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def dominant_frequency(
    x: np.ndarray, fs: float, low_hz: float, high_hz: float, min_nfft: int = 4096
) -> float | None:
    """Frequency (Hz) of the highest periodogram peak inside [low_hz, high_hz].

    The signal is detrended and Hann-windowed, zero-padded to at least `min_nfft` points,
    and the peak is refined by parabolic interpolation. Returns None for a flat signal.
    """
    x = np.asarray(x, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(x)))) if x.size else 1.0
    x = signal.detrend(x)
    if x.size == 0 or np.ptp(x) <= 1e-9 * scale:
        return None
    nfft = max(min_nfft, 1 << int(np.ceil(np.log2(len(x)))))
    freqs, power = signal.periodogram(x, fs=fs, window="hann", nfft=nfft)
    band = np.flatnonzero((freqs >= low_hz) & (freqs <= high_hz))
    if band.size == 0:
        return None
    k = band[np.argmax(power[band])]
    if 0 < k < len(power) - 1:
        left, center, right = power[k - 1], power[k], power[k + 1]
        denom = left - 2 * center + right
        offset = 0.5 * (left - right) / denom if denom != 0 else 0.0
        return float(freqs[k] + offset * (freqs[1] - freqs[0]))
    return float(freqs[k])
