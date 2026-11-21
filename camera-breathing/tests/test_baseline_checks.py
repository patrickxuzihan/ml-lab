"""analysis/baseline_checks.py on hand-made records (no data needed)."""

import importlib.util

import pytest
from conftest import PROJECT_DIR

spec = importlib.util.spec_from_file_location(
    "baseline_checks", PROJECT_DIR / "analysis" / "baseline_checks.py"
)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def window(measure, k, ref, est, excluded=None):
    error = None if est is None or ref is None else est - ref
    return {"measure": measure, "window": k, "ref_bpm": ref, "est_bpm": est, "error_bpm": error,
            "excluded": excluded}  # fmt: skip


def test_heart_checks_count_harmonics_and_low_rates():
    records = [
        {
            "subject": "P000001",
            "windows": [window("M3", 0, 50.0, 100.0), window("M3", 1, 80.0, 81.0)],
        },
        {
            "subject": "P000002",
            "windows": [window("M3", 0, 100.0, 50.0), window("M3", 1, 90.0, None)],
        },
    ]
    h = checks.heart_checks(records, 5.0)
    assert (h["windows"], h["failed"]) == (4, 3)  # the missing estimate counts as a failure
    assert h["harmonics_among_failed"] == {"second harmonic (x2)": 1, "half (x0.5)": 1}
    assert h["by_reference_rate"]["below 60 bpm"] == {"windows": 1, "failed": 1}
    assert h["failed_by_video"] == {"P000002": 2, "P000001": 1}


def test_breathing_checks_compare_camera_and_each_ppg_rate():
    def record(m1, m2, ref, bw, am, fm):
        return {
            "subject": "s",
            "variants": {"wide_reference": [window("M1", 0, ref, m1), window("M2", 0, ref, m2)]},
            "reference": {"M1": [{"bw_bpm": bw, "am_bpm": am, "fm_bpm": fm}]},
        }

    records = [record(24.0, 8.0, 9.0, 8.5, 9.0, 9.5), record(21.0, 9.0, 20.0, 20.0, 19.0, 21.0)]
    b = checks.breathing_checks(records, 2.0, "wide_reference")
    assert b["windows"] == 2
    assert b["pairs"]["M1 vs M2"] == {
        "median_abs_diff_bpm": pytest.approx(14.0),
        "within_tolerance": 0,
        "n": 2,
    }
    assert b["pairs"]["M2 vs BW"]["within_tolerance"] == 1
    assert b["by_reference_range"]["0-12 bpm"] == {"windows": 1, "M1 within": 0, "M2 within": 1}
    assert b["by_reference_range"]["15-60 bpm"] == {"windows": 1, "M1 within": 1, "M2 within": 0}


def test_identical_videos():
    a = [{"subject": "P1", "windows": [window("M1", 0, 10.0, 10.1)]},
         {"subject": "P2", "windows": [window("M1", 0, 10.0, 10.1)]}]  # fmt: skip
    b = [{"subject": "P1", "windows": [window("M1", 0, 10.0, 10.1)]},
         {"subject": "P2", "windows": [window("M1", 0, 10.0, 10.2)]}]  # fmt: skip
    assert checks.identical_videos(a, b) == {"shared": 2, "identical": 1}
