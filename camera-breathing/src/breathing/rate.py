"""Band-pass filtering, analysis windows, spectral peak frequency and spectral SNR.

The protocol's rate estimator (configs/protocol.yaml, `rate`): each window is linearly
detrended, band-passed with a zero-phase Butterworth filter, Hann-windowed and zero-padded,
and the highest periodogram peak in the band is refined by parabolic interpolation. A
single-segment Hann periodogram is Welch's method with one segment: it keeps the full
frequency resolution of the window.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def _scale(x: np.ndarray) -> float:
    return max(1.0, float(np.max(np.abs(x)))) if x.size else 1.0


def _is_flat(detrended: np.ndarray, scale: float) -> bool:
    """No variation beyond rounding error, relative to the raw signal's magnitude."""
    return detrended.size == 0 or np.ptp(detrended) <= 1e-9 * scale


def periodogram(x: np.ndarray, fs: float, min_nfft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed periodogram, zero-padded to at least `min_nfft` points."""
    nfft = max(min_nfft, 1 << int(np.ceil(np.log2(max(len(x), 1)))))
    return signal.periodogram(x, fs=fs, window="hann", nfft=nfft)


def peak_frequency(
    freqs: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float
) -> float | None:
    """Highest peak inside [low_hz, high_hz], refined by parabolic interpolation."""
    band = np.flatnonzero((freqs >= low_hz) & (freqs <= high_hz))
    if band.size == 0 or not np.any(power[band] > 0):
        return None
    k = band[np.argmax(power[band])]
    if 0 < k < len(power) - 1:
        left, center, right = power[k - 1], power[k], power[k + 1]
        denom = left - 2 * center + right
        offset = 0.5 * (left - right) / denom if denom != 0 else 0.0
        return float(freqs[k] + offset * (freqs[1] - freqs[0]))
    return float(freqs[k])


def dominant_frequency(
    x: np.ndarray, fs: float, low_hz: float, high_hz: float, min_nfft: int = 4096
) -> float | None:
    """Frequency (Hz) of the highest periodogram peak inside [low_hz, high_hz].

    The signal is detrended and Hann-windowed, zero-padded to at least `min_nfft` points,
    and the peak is refined by parabolic interpolation. Returns None for a flat signal.
    Used by the phase-1 data checks (no band-pass).
    """
    x = np.asarray(x, dtype=np.float64)
    scale = _scale(x)
    x = signal.detrend(x) if x.size else x
    if _is_flat(x, scale):
        return None
    return peak_frequency(*periodogram(x, fs, min_nfft), low_hz, high_hz)


def effective_band(
    band_hz: tuple[float, float], fs: float, max_nyquist_frac: float = 0.95
) -> tuple[float, float] | None:
    """The band, with its upper edge lowered below the Nyquist frequency if needed."""
    low, high = band_hz
    high = min(high, max_nyquist_frac * fs / 2)
    return (low, high) if high > low else None


def windows_s(duration_s: float, window_s: float) -> list[tuple[float, float]]:
    """Non-overlapping analysis windows in seconds; the whole signal if shorter than one.

    A remainder shorter than a window at the end is dropped.
    """
    if duration_s <= 0:
        return []
    if duration_s < window_s:
        return [(0.0, float(duration_s))]
    n = int(np.floor(duration_s / window_s + 1e-9))
    return [(k * window_s, (k + 1) * window_s) for k in range(n)]


def window_slice(t0: float, t1: float, fs: float, n: int) -> slice:
    """Samples of a signal at `fs` that fall in [t0, t1)."""
    return slice(min(n, int(round(t0 * fs))), min(n, int(round(t1 * fs))))


def analyze_window(
    x: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
    reference_hz: float | None = None,
    order: int = 2,
    min_nfft: int = 4096,
    max_nyquist_frac: float = 0.95,
    snr_half_width_bins: float = 2.0,
) -> tuple[float | None, float | None]:
    """(estimated frequency in Hz, SNR in dB) of one analysis window.

    The estimate is None when the window is flat or too short, or when the frame rate leaves
    no usable band. The SNR is None without a reference frequency.
    """
    x = np.asarray(x, dtype=np.float64)
    band = effective_band(band_hz, fs, max_nyquist_frac)
    if band is None or len(x) < 3 * (2 * order + 1) + 1:
        return None, None
    scale = _scale(x)
    x = signal.detrend(x)
    if _is_flat(x, scale):
        return None, None
    filtered = bandpass(x, fs, *band, order=order)
    freqs, power = periodogram(filtered, fs, min_nfft)
    estimate = peak_frequency(freqs, power, *band)
    snr = None
    if reference_hz is not None:
        half_width = snr_half_width_bins * fs / len(x)  # k bins of the unpadded window
        snr = spectral_snr_db(freqs, power, reference_hz, band, half_width)
    return estimate, snr


def spectral_snr_db(
    freqs: np.ndarray,
    power: np.ndarray,
    reference_hz: float,
    band_hz: tuple[float, float],
    half_width_hz: float,
) -> float | None:
    """Power near the reference frequency and its second harmonic over the rest of the band.

    The "signal" bins are within +-half_width_hz of f and 2f; both the signal and the rest
    are restricted to the band. Returns None if either part is empty or has no power.
    """
    in_band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    near = (np.abs(freqs - reference_hz) <= half_width_hz) | (
        np.abs(freqs - 2 * reference_hz) <= half_width_hz
    )
    signal_power = power[in_band & near].sum()
    noise_power = power[in_band & ~near].sum()
    if signal_power <= 0 or noise_power <= 0:
        return None
    return float(10 * np.log10(signal_power / noise_power))
