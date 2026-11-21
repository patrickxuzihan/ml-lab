"""Reference (ground-truth) rates for each analysis window.

- Waveform references (SCAMPS `d_br` and `d_ppg`, the UBFC contact PPG for heart rate) go
  through exactly the same per-window estimator as the camera signals (`rate.analyze_window`),
  so estimator bias cancels.
- Breathing reference from PPG (UBFC has no breathing belt): breathing modulates the PPG in
  three ways (Charlton et al., Physiol. Meas. 2016; IEEE RBME 2018): the baseline (BW: mean
  of each beat), the pulse amplitude (AM: peak minus preceding trough) and the beat interval
  (FM: sinus arrhythmia). Each beat-by-beat series is resampled to a uniform grid and its
  rate estimated per window; a window is usable only when the three rates agree.
- The pulse oximeter's own heart-rate channel is only a cross-check (notes/01-data.md 4.3).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from breathing.rate import analyze_window, bandpass, window_slice

Window = tuple[float, float]


def _bpm(hz: float | None) -> float | None:
    return None if hz is None else 60.0 * hz


def waveform_rates(
    x: np.ndarray,
    fs: float,
    windows: list[Window],
    band_hz: tuple[float, float],
    rate_cfg: dict,
) -> list[dict]:
    """Per window: rate (bpm) of a reference waveform and its SNR around its own peak (dB).

    The self-referenced SNR is a quality indicator for the reference itself.
    """
    x = np.asarray(x, dtype=np.float64)
    out = []
    for t0, t1 in windows:
        segment = x[window_slice(t0, t1, fs, len(x))]
        hz, _ = analyze_window(segment, fs, band_hz, **_rate_kwargs(rate_cfg))
        snr = None
        if hz is not None:
            _, snr = analyze_window(segment, fs, band_hz, hz, **_rate_kwargs(rate_cfg))
        out.append({"bpm": _bpm(hz), "snr_db": snr})
    return out


def _rate_kwargs(rate_cfg: dict) -> dict:
    return {
        "order": rate_cfg["filter_order"],
        "min_nfft": rate_cfg["min_nfft"],
        "max_nyquist_frac": rate_cfg["max_nyquist_frac"],
        "snr_half_width_bins": rate_cfg["snr_half_width_bins"],
    }


def detect_beats(
    ppg: np.ndarray,
    fs: float,
    band_hz: tuple[float, float] = (0.5, 5.0),
    max_heart_bpm: float = 180.0,
) -> dict[str, np.ndarray]:
    """Systolic peaks and the troughs between them.

    Peaks are found on a band-passed copy (drift and high-frequency noise removed), at least
    one shortest beat apart and with a prominence of at least 30% of the signal's 5-95%
    range; peak times are refined by parabolic interpolation. Returns sample indices of
    peaks and of the trough before each peak (first peak has none), and refined peak times.
    """
    x = signal.detrend(np.asarray(ppg, dtype=np.float64))
    high = min(band_hz[1], 0.45 * fs)
    filtered = bandpass(x, fs, band_hz[0], high)
    spread = np.percentile(filtered, 95) - np.percentile(filtered, 5)
    peaks, _ = signal.find_peaks(
        filtered,
        distance=max(1, int(fs * 60.0 / max_heart_bpm)),
        prominence=0.3 * spread if spread > 0 else None,
    )
    times = peaks.astype(np.float64)
    for i, k in enumerate(peaks):
        if 0 < k < len(filtered) - 1:
            left, center, right = filtered[k - 1], filtered[k], filtered[k + 1]
            denom = left - 2 * center + right
            if denom != 0:
                times[i] = k + 0.5 * (left - right) / denom
    troughs = np.array(
        [a + int(np.argmin(filtered[a:b])) for a, b in zip(peaks[:-1], peaks[1:], strict=True)],
        dtype=int,
    )
    return {"peaks": peaks, "troughs": troughs, "times_s": times / fs, "filtered": filtered}


def respiratory_series(ppg: np.ndarray, fs: float, beats: dict) -> dict[str, tuple]:
    """Beat-by-beat BW, AM and FM series as (times in s, values)."""
    raw = signal.detrend(np.asarray(ppg, dtype=np.float64))
    filtered, peaks, troughs, t = (
        beats["filtered"],
        beats["peaks"],
        beats["troughs"],
        beats["times_s"],
    )
    # Beat i (i >= 1) has a trough before it; BW needs the next trough as well.
    am_t, am_v = t[1:], filtered[peaks[1:]] - filtered[troughs]
    fm_t, fm_v = t[1:], np.diff(t)
    bw_t = t[1:-1]
    bw_v = np.array(
        [raw[a:b].mean() for a, b in zip(troughs[:-1], troughs[1:], strict=True)], dtype=float
    )
    return {"BW": (bw_t, bw_v), "AM": (am_t, am_v), "FM": (fm_t, fm_v)}


def resample(times: np.ndarray, values: np.ndarray, fs: float, duration_s: float) -> np.ndarray:
    """Linear interpolation onto a uniform grid starting at 0 s (edge values held)."""
    grid = np.arange(int(np.floor(duration_s * fs))) / fs
    if len(times) < 2:
        return np.zeros(len(grid))
    return np.interp(grid, times, values)


def respiration_from_ppg(
    ppg: np.ndarray,
    fs: float,
    windows: list[Window],
    band_hz: tuple[float, float],
    rate_cfg: dict,
    cfg: dict,
) -> list[dict]:
    """Per window: breathing rate (bpm) from BW, AM and FM, and the fused reference.

    The reference is the median of the three when all exist and span at most
    cfg["agreement_bpm"]; otherwise None (the window is not used).
    """
    duration = len(ppg) / fs
    beats = detect_beats(ppg, fs, tuple(cfg["beat_band_hz"]), cfg["max_heart_bpm"])
    fs_r = cfg["resample_hz"]
    grids = {
        name: resample(t, v, fs_r, duration)
        for name, (t, v) in respiratory_series(ppg, fs, beats).items()
    }
    out = []
    for t0, t1 in windows:
        rates = {}
        for name, grid in grids.items():
            hz, _ = analyze_window(
                grid[window_slice(t0, t1, fs_r, len(grid))], fs_r, band_hz, **_rate_kwargs(rate_cfg)
            )
            rates[name] = _bpm(hz)
        values = [v for v in rates.values() if v is not None]
        spread = max(values) - min(values) if len(values) == 3 else None
        usable = spread is not None and spread <= cfg["agreement_bpm"]
        out.append(
            {
                **{f"{k.lower()}_bpm": v for k, v in rates.items()},
                "spread_bpm": spread,
                "bpm": float(np.median(values)) if usable else None,
                "n_beats": int(np.sum((beats["times_s"] >= t0) & (beats["times_s"] < t1))),
            }
        )
    return out


def fuse_by_std(window: dict, max_std_bpm: float) -> float | None:
    """Wider agreement rule for the sensitivity analysis, after Karlen et al. (2013).

    window holds the per-method rates of `respiration_from_ppg` ("bw_bpm", "am_bpm",
    "fm_bpm"). Returns their mean when all three exist and their sample standard deviation
    (ddof=1) is at most max_std_bpm; otherwise None.
    """
    values = [window.get(key) for key in ("bw_bpm", "am_bpm", "fm_bpm")]
    if any(v is None for v in values):
        return None
    if float(np.std(values, ddof=1)) > max_std_bpm:
        return None
    return float(np.mean(values))


def oximeter_rates(
    hr: np.ndarray, fs: float, windows: list[Window], min_valid_bpm: float = 30.0
) -> list[dict]:
    """Per window: median of the oximeter heart-rate channel over valid values (>= min)."""
    hr = np.asarray(hr, dtype=np.float64)
    out = []
    for t0, t1 in windows:
        segment = hr[window_slice(t0, t1, fs, len(hr))]
        valid = segment[np.isfinite(segment) & (segment >= min_valid_bpm)]
        out.append(
            {
                "bpm": float(np.median(valid)) if valid.size else None,
                "valid_frac": float(valid.size / segment.size) if segment.size else 0.0,
            }
        )
    return out
