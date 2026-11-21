"""M1-M3 end to end on synthetic videos with known rates, plus their building blocks."""

from dataclasses import replace

import numpy as np
import pytest

from breathing import pipeline, roi
from breathing.rate import analyze_window, effective_band, window_slice, windows_s
from breathing.signals import extract_traces, pos, pos_projection, region_shift, to_gray
from breathing.synth import (
    PULSE_RGB,
    SynthSpec,
    breathing_displacement,
    heart_modulation,
    make_video,
    synth_landmarks,
)

PROTOCOL = pipeline.load_protocol()

# Sub-pixel breathing motion, color changes of about one 8-bit level, sensor noise.
M_SPEC = SynthSpec(
    n_frames=600,
    fps=30.0,
    height=96,
    width=128,
    breathing_hz=0.27,
    heart_hz=1.35,
    breathing_amplitude_px=0.4,
    heart_amplitude=1.0,
    breathing_color_amplitude=0.8,
    noise_std=1.0,
    seed=3,
    face_center_frac=(0.25, 0.5),
    face_radius_frac=(0.15, 0.12),
    shoulder_half_width_frac=0.15,
)


def regions(spec: SynthSpec, n_sampled: int = 10):
    landmarks = np.repeat(synth_landmarks(spec)[None], n_sampled, axis=0)
    return roi.measurement_regions(landmarks, spec.width, spec.height)


def run_measures(spec: SynthSpec) -> list[dict]:
    frames = make_video(spec)
    boxes, _ = regions(spec)
    traces = extract_traces(iter(frames), spec.fps, {"shoulder": boxes["shoulder"]}, boxes)
    windows = pipeline.analysis_windows(spec.n_frames / spec.fps, PROTOCOL)
    waves = {"breathing": breathing_displacement(spec), "heart": heart_modulation(spec)}
    refs = pipeline.waveform_references(waves, spec.fps, windows, PROTOCOL)
    return pipeline.measure(traces, windows, refs, PROTOCOL)


@pytest.mark.parametrize(
    "changes",
    [
        {},  # 20 s at 30 fps: breathing uses the whole clip, heart rate two 10 s windows
        {"fps": 15.0, "breathing_hz": 0.18, "heart_hz": 1.05},  # 40 s: one 30 s window + rest
        {"fps": 24.0, "n_frames": 480, "height": 72, "width": 96, "heart_hz": 2.2},
    ],
)
def test_three_measurements_are_near_exact_on_synthetic_video(changes):
    spec = replace(M_SPEC, **changes)
    records = run_measures(spec)
    duration = spec.n_frames / spec.fps
    expected = {
        "M1": len(windows_s(duration, 30.0)),
        "M2": len(windows_s(duration, 30.0)),
        "M3": len(windows_s(duration, 10.0)),
    }
    for m, n in expected.items():
        rows = [r for r in records if r["measure"] == m]
        assert len(rows) == n
        truth = 60 * (spec.heart_hz if m == "M3" else spec.breathing_hz)
        for r in rows:
            assert r["excluded"] is None
            assert r["ref_bpm"] == pytest.approx(truth, abs=0.3)
            assert abs(r["error_bpm"]) < 0.5, (m, r)
            assert r["snr_db"] > 3


def test_measure_records_exclusions_and_missing_estimates():
    spec = replace(M_SPEC, n_frames=300)
    frames = make_video(spec)
    boxes, _ = regions(spec)
    traces = extract_traces(iter(frames), spec.fps, {"shoulder": None}, boxes)
    windows = pipeline.analysis_windows(10.0, PROTOCOL)
    refs = {"M1": [16.0], "M2": [None], "M3": [200.0]}  # M3 reference above the 180 bpm edge
    by_measure = {r["measure"]: r for r in pipeline.measure(traces, windows, refs, PROTOCOL)}
    assert by_measure["M1"]["excluded"] == "no_region"
    assert by_measure["M2"]["excluded"] == "no_reference"
    assert by_measure["M3"]["excluded"] == "reference_out_of_band"

    # A static region (the encoder dropped all motion) gives no estimate: a failure, not an
    # exclusion.
    still = np.repeat(frames[:1], 300, axis=0)
    traces = extract_traces(iter(still), spec.fps, {"shoulder": boxes["shoulder"]}, boxes)
    refs = {"M1": [16.0], "M2": [16.0], "M3": [81.0]}
    for r in pipeline.measure(traces, windows, refs, PROTOCOL):
        assert r["excluded"] is None and r["est_bpm"] is None


def test_regions_from_synthetic_landmarks():
    boxes, info = regions(M_SPEC)
    assert info["shoulder_source"] == "pose" and info["face_source"] == "pose"
    r0, r1, c0, c1 = boxes["shoulder"]
    assert r0 < M_SPEC.shoulder_y < r1  # straddles the shoulder line
    # Every pixel of the face box is skin (inside the face ellipse) ...
    f0, f1, g0, g1 = boxes["face"]
    rows, cols = np.mgrid[f0:f1, g0:g1]
    (fr, fc), (rr, rc) = M_SPEC.face_center, M_SPEC.face_radius
    assert np.all(((rows + 0.5 - fr) / rr) ** 2 + ((cols + 0.5 - fc) / rc) ** 2 <= 1.0)
    # ... and the face box does not overlap the shoulder band.
    assert f1 <= r0


def test_shoulder_band_falls_back_to_the_face_when_shoulders_are_hidden():
    lm = np.repeat(synth_landmarks(M_SPEC)[None], 10, axis=0)
    lm[:, roi.LEFT_SHOULDER, 2] = 0.1  # low confidence
    boxes, info = roi.measurement_regions(lm, M_SPEC.width, M_SPEC.height)
    assert info["shoulder_source"] == "face_anchored"
    r0, r1, _, _ = boxes["shoulder"]
    assert r0 > boxes["face"][1]  # below the mouth
    assert r1 == M_SPEC.height  # down to the bottom of the frame

    nothing = np.full_like(lm, np.nan)
    boxes, info = roi.measurement_regions(nothing, M_SPEC.width, M_SPEC.height)
    assert boxes == {"shoulder": None, "face": None}
    assert info["shoulder_source"] == "none"


def test_pose_band_matches_phase_one_definition():
    lm = np.repeat(synth_landmarks(M_SPEC)[None], 4, axis=0)
    probe, _ = roi.probe_regions(lm, M_SPEC.width, M_SPEC.height)
    boxes, _ = roi.measurement_regions(lm, M_SPEC.width, M_SPEC.height)
    assert boxes["shoulder"] == probe["shoulder"]


def test_scale_box_keeps_the_full_extent():
    assert roi.scale_box((10, 21, 5, 30), (100, 200), (50, 100)) == (5, 11, 2, 15)
    assert roi.scale_box((0, 100, 0, 200), (100, 200), (25, 50)) == (0, 25, 0, 50)
    assert roi.scale_box(None, (100, 200), (50, 100)) is None


def test_pos_recovers_a_blood_volume_pulse_and_cancels_intensity_changes():
    fs, t = 30.0, np.arange(300) / 30.0
    skin = np.array([180.0, 130.0, 100.0])
    pulse = np.sin(2 * np.pi * 1.2 * t)
    rgb = skin * (1 + 0.002 * pulse[:, None] * np.array(PULSE_RGB))
    assert abs(np.corrcoef(pos_projection(rgb), pulse)[0, 1]) > 0.999

    # The same pulse plus a 20x larger change of overall brightness: POS keeps the pulse.
    # (Overlap-add tapers the first and last window length, so compare the middle.)
    flicker = 0.04 * np.sin(2 * np.pi * 0.5 * t)
    out = pos(rgb * (1 + flicker[:, None]), fs, 1.6)
    middle = slice(48, -48)
    assert abs(np.corrcoef(out[middle], pulse[middle])[0, 1]) > 0.999


def test_short_pos_windows_suppress_slow_signals():
    """Why M2 uses one projection per window: 1.6 s windows high-pass the breathing band."""
    fs, t = 30.0, np.arange(900) / 30.0
    skin = np.array([180.0, 130.0, 100.0])

    def gain(f_hz: float) -> float:
        """Amplitude of 1.6 s overlap-added POS relative to one projection over 30 s."""
        x = np.sin(2 * np.pi * f_hz * t)
        rgb = skin * (1 + 0.002 * x[:, None] * np.array(PULSE_RGB))
        whole = pos(rgb, fs, None)
        assert abs(np.corrcoef(whole, x)[0, 1]) > 0.999
        return np.std(pos(rgb, fs, 1.6)) / np.std(whole)

    # Overlap-add also scales the output, so compare a breathing rate with a heart rate.
    assert gain(0.2) < 0.4 * gain(1.2)


def test_streamed_motion_matches_phase_one_region_shift(spec, frames):
    box = (int(spec.shoulder_y) - 10, int(spec.shoulder_y) + 10, 8, spec.width - 8)
    traces = extract_traces(iter(frames), spec.fps, {"shoulder": box}, {"face": None})
    dy, _ = region_shift(to_gray(frames), box)
    np.testing.assert_allclose(traces.motion["shoulder"], dy, atol=1e-9)
    assert traces.rgb["face"] is None and traces.n_frames == spec.n_frames


def test_windows_and_slices():
    assert windows_s(20.0, 30.0) == [(0.0, 20.0)]
    assert windows_s(68.2, 30.0) == [(0.0, 30.0), (30.0, 60.0)]
    assert windows_s(60.0, 30.0) == [(0.0, 30.0), (30.0, 60.0)]
    assert windows_s(0.0, 10.0) == []
    assert window_slice(10.0, 20.0, 29.7, 2000) == slice(297, 594)
    assert window_slice(0.0, 30.0, 30.0, 600) == slice(0, 600)


def test_band_is_clipped_below_nyquist():
    assert effective_band((0.7, 3.0), 30.0) == (0.7, 3.0)
    assert effective_band((0.7, 3.0), 5.0) == pytest.approx((0.7, 2.375))
    assert effective_band((0.7, 3.0), 1.4) is None


def test_analyze_window_snr_and_flat_signals():
    fs = 30.0
    t = np.arange(600) / fs
    tone = np.sin(2 * np.pi * 0.3 * t)
    hz, snr = analyze_window(tone, fs, (0.1, 0.7), 0.3)
    assert hz == pytest.approx(0.3, abs=0.002) and snr > 20
    noise = np.random.default_rng(0).normal(size=600)
    _, snr_noise = analyze_window(noise, fs, (0.1, 0.7), 0.3)
    # White noise does not give -inf: the "signal" bins cover a large part of this narrow
    # band. That floor depends on band and window, hence the protocol compares changes.
    assert snr_noise < snr - 15
    assert analyze_window(np.zeros(600), fs, (0.1, 0.7), 0.3) == (None, None)
    assert analyze_window(tone[:10], fs, (0.1, 0.7)) == (None, None)  # too short
