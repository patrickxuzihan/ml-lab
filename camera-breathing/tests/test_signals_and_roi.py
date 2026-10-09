import numpy as np
import pytest
from scipy import ndimage

from breathing import roi
from breathing.signals import global_flow, region_shift, to_gray
from breathing.synth import breathing_displacement


def smooth_texture(shape, seed=0):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.normal(size=shape), 2.0) * 50 + 128


@pytest.mark.parametrize("dy, dx", [(0.0, 0.0), (0.2, -0.1), (-0.15, 0.25)])
def test_global_flow_recovers_subpixel_shifts(dy, dx):
    ref = smooth_texture((64, 96))
    moved = ndimage.shift(ref, (dy, dx), order=3, mode="nearest")
    est_dy, est_dx = global_flow(ref, moved)
    assert est_dy == pytest.approx(dy, abs=0.005)
    assert est_dx == pytest.approx(dx, abs=0.005)


def test_global_flow_is_approximate_for_larger_shifts():
    ref = smooth_texture((64, 96))
    est_dy, est_dx = global_flow(ref, ndimage.shift(ref, (-0.8, 0.8), order=3, mode="nearest"))
    assert est_dy == pytest.approx(-0.8, rel=0.05)
    assert est_dx == pytest.approx(0.8, rel=0.05)


def test_region_shift_follows_synthetic_breathing(spec, frames):
    gray = to_gray(frames)
    row = int(spec.shoulder_y)
    truth = breathing_displacement(spec) - breathing_displacement(spec)[0]

    # Inside the torso the whole region moves, so the displacement is recovered in pixels.
    dy, dx = region_shift(gray, (row + 4, spec.height, 8, spec.width - 8))
    assert np.corrcoef(dy, truth)[0, 1] > 0.99
    assert np.std(dy) / np.std(truth) == pytest.approx(1.0, abs=0.1)
    assert np.max(np.abs(dx)) < 0.1

    # Across the shoulder line part of the region is static background: same shape, smaller.
    dy, _ = region_shift(gray, (row - 10, row + 10, 8, spec.width - 8))
    assert np.corrcoef(dy, truth)[0, 1] > 0.98
    assert 0.2 < np.std(dy) / np.std(truth) <= 1.05


def test_to_gray():
    gray = to_gray(np.array([[[255, 255, 255], [255, 0, 0]]], dtype=np.uint8))
    assert gray[0, 0] == pytest.approx(255, abs=0.01)
    assert gray[0, 1] == pytest.approx(0.299 * 255, abs=0.01)


def make_landmarks(n, shoulder_y=0.6, detected=None):
    lm = np.full((n, roi.N_LANDMARKS, 3), np.nan)
    lm[:, :, 2] = 0.9
    lm[:, roi.FACE, 0] = np.linspace(0.45, 0.55, len(roi.FACE))
    lm[:, roi.FACE, 1] = 0.3
    lm[:, roi.LEFT_SHOULDER, :2] = 0.65, shoulder_y
    lm[:, roi.RIGHT_SHOULDER, :2] = 0.35, shoulder_y
    lm[:, roi.LEFT_HIP, :2] = 0.6, 1.2
    lm[:, roi.RIGHT_HIP, :2] = 0.4, 1.2
    if detected is not None:
        lm[~detected] = np.nan
    return lm


def test_visibility_summary_counts_and_geometry():
    detected = np.array([True, True, True, False])
    summary = roi.visibility_summary(make_landmarks(4, detected=detected), 200, 100)
    assert summary["n_sampled"] == 4
    assert summary["pose_detected_frac"] == 0.75
    assert summary["shoulders_visible_frac"] == 0.75
    assert summary["hip_visible_frac"] == 0.0  # below the frame
    assert summary["shoulder_y_frac_median"] == pytest.approx(0.6)
    assert summary["rows_below_shoulders_px_median"] == pytest.approx(40)
    assert summary["shoulder_width_px_median"] == pytest.approx(60)
    assert summary["rows_below_shoulders_per_shoulder_width_median"] == pytest.approx(40 / 60)
    assert summary["face_width_px_median"] == pytest.approx(20)


def test_low_confidence_or_out_of_frame_shoulders_are_not_visible():
    lm = make_landmarks(3)
    lm[0, roi.LEFT_SHOULDER, 2] = 0.1
    lm[1, roi.RIGHT_SHOULDER, 1] = 1.05
    summary = roi.visibility_summary(lm, 200, 100)
    assert summary["shoulders_visible_frac"] == pytest.approx(1 / 3)


def test_visibility_summary_without_any_pose():
    summary = roi.visibility_summary(np.full((5, roi.N_LANDMARKS, 3), np.nan), 200, 100)
    assert summary["pose_detected_frac"] == 0.0
    assert summary["shoulder_y_frac_median"] is None


def test_probe_regions_from_pose_are_ordered_and_inside_frame():
    boxes, source = roi.probe_regions(make_landmarks(10, shoulder_y=0.5), 200, 100)
    assert source == "pose"
    head, shoulder, chest = boxes["head"], boxes["shoulder"], boxes["chest"]
    assert head[0] < shoulder[0] < chest[0]
    assert shoulder[0] < 50 < shoulder[1]  # straddles the shoulder line
    assert chest[0] > 50
    for r0, r1, c0, c1 in boxes.values():
        assert 0 <= r0 < r1 <= 100 and 0 <= c0 < c1 <= 200


def test_probe_regions_fall_back_without_pose():
    boxes, source = roi.probe_regions(np.empty((0, roi.N_LANDMARKS, 3)), 320, 240)
    assert source == "fallback"
    assert all(box is not None for box in boxes.values())


def test_ensure_model_downloads_once(tmp_path):
    src = tmp_path / "model.task"
    src.write_bytes(b"weights")
    cache = tmp_path / "cache"
    path = roi.ensure_model(src.as_uri(), cache)
    assert path.read_bytes() == b"weights"
    src.write_bytes(b"changed")
    assert roi.ensure_model(src.as_uri(), cache).read_bytes() == b"weights"
    assert len(roi.file_sha256(path)) == 64
