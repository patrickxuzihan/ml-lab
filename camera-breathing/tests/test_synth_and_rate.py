import numpy as np

from breathing.rate import bandpass, dominant_frequency
from breathing.synth import SynthSpec, make_video


def test_video_shape_and_determinism(spec, frames):
    assert frames.shape == (spec.n_frames, spec.height, spec.width, 3)
    assert frames.dtype == np.uint8
    noisy = SynthSpec(n_frames=5, noise_std=2.0, seed=1)
    np.testing.assert_array_equal(make_video(noisy), make_video(noisy))


def test_shoulder_band_moves_at_breathing_rate(spec, frames):
    row = int(spec.shoulder_y)
    band = frames[:, row - 3 : row + 4, :, :].mean(axis=(1, 2, 3))
    assert abs(dominant_frequency(band, spec.fps, 0.1, 0.7) - spec.breathing_hz) < 0.01


def test_face_color_changes_at_heart_rate(spec, frames):
    r, c = (int(v) for v in spec.face_center)
    face = frames[:, r - 3 : r + 4, c - 3 : c + 4, 1].mean(axis=(1, 2))
    assert abs(dominant_frequency(face, spec.fps, 0.7, 3.0) - spec.heart_hz) < 0.01


def test_dominant_frequency_is_sub_bin_accurate():
    fs, f = 30.0, 0.27
    t = np.arange(600) / fs
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * f * t) + 0.3 * rng.normal(size=t.size)
    assert abs(dominant_frequency(x, fs, 0.1, 0.7) - f) < 0.005


def test_dominant_frequency_of_flat_signal_is_none():
    assert dominant_frequency(np.ones(300), 30.0, 0.1, 0.7) is None


def test_bandpass_removes_out_of_band_component():
    fs = 30.0
    t = np.arange(900) / fs
    slow, fast = np.sin(2 * np.pi * 0.3 * t), np.sin(2 * np.pi * 5.0 * t)
    filtered = bandpass(slow + fast, fs, 0.1, 0.7)
    middle = slice(150, -150)  # ignore filter edges
    assert np.corrcoef(filtered[middle], slow[middle])[0, 1] > 0.99
    spectrum = np.abs(np.fft.rfft(filtered)) / np.abs(np.fft.rfft(slow + fast))
    assert spectrum[np.argmin(np.abs(np.fft.rfftfreq(t.size, 1 / fs) - 5.0))] < 0.01
