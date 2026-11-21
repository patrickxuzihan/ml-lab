"""Reference rates: waveforms, breathing derived from PPG, oximeter cross-check, UBFC timing."""

import numpy as np
import pytest

from breathing import pipeline, reference
from breathing.datasets.ubfc import timestamp_flags
from breathing.synth import synthetic_ppg

PROTOCOL = pipeline.load_protocol()
RATE = PROTOCOL["rate"]
PPG_CFG = PROTOCOL["reference"]["ppg_respiration"]
BREATHING_BAND = tuple(PROTOCOL["measurements"]["M1"]["band_hz"])
HEART_BAND = tuple(PROTOCOL["measurements"]["M3"]["band_hz"])
FS = 30.0
N = int(60 * FS)  # one minute, like UBFC-rPPG
WINDOWS = [(0.0, 30.0), (30.0, 60.0)]


def derived(ppg, cfg=PPG_CFG):
    return reference.respiration_from_ppg(ppg, FS, WINDOWS, BREATHING_BAND, RATE, cfg)


@pytest.mark.parametrize(
    "modulation, key",
    [
        ({"baseline": 0.2}, "bw_bpm"),
        ({"amplitude_depth": 0.2}, "am_bpm"),
        ({"interval_depth": 0.08}, "fm_bpm"),
    ],
)
def test_each_modulation_is_recovered_by_its_own_series(modulation, key):
    ppg = synthetic_ppg(N, FS, heart_hz=1.25, breathing_hz=0.22, noise_std=0.02, **modulation)
    for window in derived(ppg):
        assert window[key] == pytest.approx(60 * 0.22, abs=0.5)


def test_fused_reference_when_all_three_agree():
    ppg = synthetic_ppg(
        N, FS, 1.1, 0.3, baseline=0.1, amplitude_depth=0.1, interval_depth=0.05, noise_std=0.05
    )
    for window in derived(ppg):
        assert window["spread_bpm"] <= PPG_CFG["agreement_bpm"]
        assert window["bpm"] == pytest.approx(18.0, abs=0.5)
        assert window["n_beats"] == pytest.approx(30 * 1.1, abs=2)


def test_disagreeing_series_give_no_reference():
    t = np.arange(N) / FS
    # Baseline and beat intervals at 12 bpm, pulse amplitude at 27 bpm.
    ppg = synthetic_ppg(N, FS, 1.2, 0.2, baseline=0.3, interval_depth=0.06)
    ppg = ppg * (1 + 0.3 * np.sin(2 * np.pi * 0.45 * t))
    for window in derived(ppg):
        assert window["bw_bpm"] == pytest.approx(12.0, abs=0.7)
        assert window["am_bpm"] == pytest.approx(27.0, abs=0.7)
        assert window["spread_bpm"] > PPG_CFG["agreement_bpm"]
        assert window["bpm"] is None
    loose = {**PPG_CFG, "agreement_bpm": 20.0}
    for window in derived(ppg, loose):
        assert window["bpm"] == pytest.approx(12.0, abs=0.7)  # median of the three


def test_flat_ppg_gives_no_reference():
    for window in derived(np.zeros(N)):
        assert window["bpm"] is None and window["n_beats"] == 0


def test_heart_rate_from_ppg_waveform():
    ppg = synthetic_ppg(N, FS, 1.37, 0.25, baseline=0.2, amplitude_depth=0.1, noise_std=0.05)
    rates = reference.waveform_rates(ppg, FS, [(0, 10), (10, 20)], HEART_BAND, RATE)
    for r in rates:
        assert r["bpm"] == pytest.approx(60 * 1.37, abs=0.5)
        assert r["snr_db"] > 3


def test_oximeter_values_below_threshold_are_invalid():
    hr = np.concatenate([np.full(300, 72.0), np.full(300, 1.0)])
    out = reference.oximeter_rates(hr, FS, [(0, 10), (10, 20)], min_valid_bpm=30)
    assert out[0] == {"bpm": 72.0, "valid_frac": 1.0}
    assert out[1] == {"bpm": None, "valid_frac": 0.0}


def test_ubfc_timestamp_flags():
    t = np.arange(10) / 30.0
    assert timestamp_flags(t) == {
        "max_gap_s": pytest.approx(1 / 30),
        "n_duplicates": 0,
        "large_gap": False,
        "duplicates": False,
    }
    t = np.array([0.0, 0.03, 0.03, 0.06, 0.9, 0.93])
    flags = timestamp_flags(t, gap_s=0.5)
    assert flags["large_gap"] and flags["duplicates"] and flags["n_duplicates"] == 1
    assert flags["max_gap_s"] == pytest.approx(0.84)
