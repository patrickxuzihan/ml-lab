"""analysis/compression_analysis.py on hand-made records (no data needed)."""

import importlib.util

import pytest
from conftest import PROJECT_DIR

from breathing import pipeline

spec = importlib.util.spec_from_file_location(
    "compression_analysis", PROJECT_DIR / "analysis" / "compression_analysis.py"
)
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)

PROTOCOL = pipeline.load_protocol()
PROTOCOL["statistics"]["bootstrap_samples"] = 50


def w(measure, k, est, ref=None, excluded=None):
    error = None if est is None or ref is None else est - ref
    return {"measure": measure, "window": k, "ref_bpm": ref, "est_bpm": est, "error_bpm": error,
            "snr_db": 1.0, "excluded": excluded}  # fmt: skip


def record(video, setting, windows, kbps=None):
    return {"video": video, "subject": video, "setting": setting, "windows": windows,
            "kbps": kbps, "fps": 30.0, "width": 320, "height": 240}  # fmt: skip


def test_agreement_with_the_uncompressed_estimate_needs_no_reference():
    records = [
        record("v1", "source", [w("M1", 0, 15.0, excluded="no_reference"), w("M3", 0, 80.0, 80.0),
                                w("M2", 0, None, excluded="no_region")]),
        record("v1", "x264_crf23", [w("M1", 0, 16.5), w("M3", 0, 90.0, 80.0), w("M2", 0, 15.0)]),
        record("v2", "source", [w("M1", 0, 12.0), w("M3", 0, 70.0, 70.0)]),
        record("v2", "x264_crf23", [w("M1", 0, None), w("M3", 0, 72.0, 70.0)]),
    ]  # fmt: skip
    con = ca.consistency(records, PROTOCOL)["x264_crf23"]
    # M1: v1 within 2 bpm although it has no reference; v2 lost its estimate.
    assert con["M1"]["value"] == 0.5 and con["M1"]["n"] == 2
    assert con["M3"]["value"] == 0.5  # 10 bpm off (v1), 2 bpm off (v2)
    assert "M2" not in con  # no region on the source: nothing to compare with


def test_paired_comparison_and_transitions():
    records = [
        record("v1", "source", [w("M3", 0, 80.0, 80.0), w("M3", 1, 100.0, 80.0)]),
        record("v1", "raw_d1_s2", [w("M3", 0, 80.5, 80.0), w("M3", 1, 81.0, 80.0)]),
    ]
    delta = ca.paired(records, "source", "raw_d1_s2", PROTOCOL)["M3"]
    assert delta["success_rate"]["delta"] == pytest.approx(0.5)
    assert ca.transitions(records, "raw_d1_s2", PROTOCOL)["M3"] == {
        "windows": 2,
        "fail_to_success": 1,
        "success_to_fail": 0,
    }


def test_m2_follows_m1_and_best_cell():
    records = [
        record("v1", "source", [w("M1", 0, 15.0), w("M2", 0, 15.5)]),
        record("v2", "source", [w("M1", 0, 15.0), w("M2", 0, 25.0)]),
    ]
    assert ca.m2_follows_m1(records, PROTOCOL)["source"]["value"] == 0.5
    grid = {"x264": {500: {"M3": {"d1_s1": {"value": 0.5, "ci": [0.4, 0.6]},
                                  "d2_s2": {"value": 0.7, "ci": [0.6, 0.8]}}}},
            "raw": {0: {"M3": {"d2_s2": {"value": 0.9, "ci": [0.8, 1.0]}}}}}  # fmt: skip
    best = ca.best_cells(grid)
    assert set(best) == {"x264"}
    assert best["x264"][500]["M3"]["cell"] == "d2_s2"
    assert best["x264"][500]["M3"]["full_rate"]["value"] == 0.5
