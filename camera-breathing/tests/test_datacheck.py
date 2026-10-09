import numpy as np
import pytest
import yaml
from conftest import PROJECT_DIR, fake_pose, write_scamps_mat, write_ubfc_subject

from breathing import datacheck
from breathing.datasets import ubfc
from breathing.signals import to_gray
from breathing.synth import breathing_displacement
from breathing.video_io import probe


def load_config(name):
    return yaml.safe_load((PROJECT_DIR / "configs" / name).read_text())


def test_timing_stats_finds_rate_and_gaps():
    t = np.arange(300) / 30.0
    t[150:] += 0.5  # a dropped half second
    stats = datacheck.timing_stats(t)
    assert stats["n"] == 300
    assert stats["fs_hz"] == pytest.approx(30.0)
    assert stats["dt_max_s"] == pytest.approx(0.5 + 1 / 30)
    assert stats["monotonic"] is True


def test_value_stats():
    stats = datacheck.value_stats(np.array([0.0, 60.0, 70.0, np.nan]))
    assert (stats["min"], stats["median"], stats["max"]) == (0.0, 60.0, 70.0)
    assert (stats["n_nonfinite"], stats["n_zero"]) == (1, 1)


def test_lagged_correlation_finds_delay():
    t = np.arange(600) / 30.0
    a = np.sin(2 * np.pi * 0.3 * t)
    b = np.roll(a, -9)  # a lags b by 9 samples
    r, lag = datacheck.lagged_correlation(a[20:-20], b[20:-20], max_lag=15)
    assert lag == 9 and r == pytest.approx(1.0, abs=1e-6)


def test_motion_probe_on_synthetic_video(spec, frames):
    row = int(spec.shoulder_y)
    boxes = {"shoulder": (row - 10, row + 10, 8, spec.width - 8), "missing": None}
    out = datacheck.motion_probe(
        to_gray(frames), boxes, spec.fps, breathing_displacement(spec), (0.1, 0.7), 2.0, 2.0
    )
    assert out["reference_peak_bpm"] == pytest.approx(60 * spec.breathing_hz, abs=0.5)
    dy, dx = out["shoulder"]["dy"], out["shoulder"]["dx"]
    assert dy["peak_match"] is True
    assert dy["corr_zero_lag"] > 0.95
    assert dy["std_px"] > 1.0 > 10 * dx["std_px"]
    assert out["missing"] is None


def test_check_scamps_video(tmp_path, spec, frames):
    path = tmp_path / "P000001.mat"
    write_scamps_mat(path, frames, spec)
    figure = tmp_path / "regions.png"
    record = datacheck.check_scamps_video(
        path, load_config("inspect_scamps.yaml"), fake_pose(spec), figure
    )
    assert record["video"]["n_frames"] == spec.n_frames
    assert record["video"]["layout"]["interpretation"] == "matlab_order"
    assert record["ground_truth"]["breathing"]["matches_frames"] is True
    assert record["ground_truth"]["breathing_peak_bpm"] == pytest.approx(18, abs=0.5)
    assert record["ground_truth"]["heart_peak_bpm"] == pytest.approx(90, abs=0.5)
    assert record["visibility"]["shoulders_visible_frac"] == 1.0
    assert record["motion_probe"]["region_source"] == "pose"
    for region in ("shoulder", "chest"):
        assert record["motion_probe"][region]["dy"]["peak_match"] is True
        assert abs(record["motion_probe"][region]["dy"]["corr_zero_lag"]) > 0.95
    assert probe(figure).width == spec.width


def test_check_scamps_label_file_without_frames(tmp_path):
    import h5py

    path = tmp_path / "labels.mat"
    with h5py.File(path, "w") as h5:
        h5["d_br"] = np.zeros((1, 10))
    record = datacheck.check_scamps_video(path, load_config("inspect_scamps.yaml"), None)
    assert "video" not in record and "no RawFrames" in record["note"]


def test_check_ubfc_subject(tmp_path, spec, frames):
    write_ubfc_subject(tmp_path / "subject1", frames, spec)
    subject = ubfc.find_subjects(tmp_path)[0][0]
    record = datacheck.check_ubfc_subject(
        subject.video,
        ubfc.read_ground_truth(subject.ground_truth),
        load_config("inspect_ubfc.yaml"),
        fake_pose(spec),
    )
    assert record["video"]["n_decoded_frames"] == spec.n_frames
    assert record["video"]["fps"] == pytest.approx(spec.fps)
    assert record["ground_truth"]["n_minus_frames"] == 0
    assert record["ground_truth"]["timing"]["fs_hz"] == pytest.approx(spec.fps)
    assert record["visibility"]["n_sampled"] == spec.n_frames // 15
    assert "motion_probe" not in record


def test_summary_tables():
    records = [
        {
            "id": f"s{i}",
            "status": "ok",
            "file": {"size_bytes": 2e9},
            "video": {"codec_name": "rawvideo", "fps": 30.0, "width": 640},
            "ground_truth": {"timing": {"fs_hz": 30.0 + i}},
        }
        for i in range(3)
    ] + [{"id": "bad", "status": "error", "error": "OSError: gone"}]
    summary, markdown = datacheck.summarize(records, "ubfc")
    rows = dict(summary["rows"])
    assert rows["Videos checked"] == "3" and rows["Errors"] == "1"
    assert rows["Total size of checked videos (GB)"] == "6"
    assert rows["Codec"] == "rawvideo (3)"
    assert rows["GT sampling rate (Hz)"] == "30 / 31 / 32"
    assert "`bad`: OSError: gone" in markdown


def test_flatten_and_csv(tmp_path):
    record = {"id": "x", "status": "ok", "a": {"b": 1, "c": [1, 2]}, "variables": [{"n": 1}]}
    assert datacheck.flatten(record)["a.b"] == 1
    datacheck.write_items_csv([record], tmp_path / "items.csv")
    header = (tmp_path / "items.csv").read_text().splitlines()[0]
    assert header == "id,status,a.b,a.c"
