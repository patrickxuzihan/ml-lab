"""Synthetic test data with known breathing and heart rates.

The video is a textured torso whose top edge (the shoulder line) moves up and down at the
breathing rate, and a face ellipse whose skin color follows a blood-volume signal: a heart
component plus an optional breathing component, both along a green-dominated color direction
(blood absorbs green most). Optionally the face moves up and down with breathing too, as a
head does on real recordings (phase 5, face-only video). Everything is rendered analytically,
so sub-pixel motion is exact.

`synth_landmarks` gives the pose landmarks a detector would return for this scene, so tests
and synthetic experiments go through the same region code as real videos. `synthetic_ppg`
makes a contact-PPG-like waveform with the three breathing modulations (baseline, amplitude
and beat-interval) used to derive a breathing reference from PPG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from breathing import roi

# Relative change of R, G, B for a blood-volume pulse; only the direction matters.
PULSE_RGB = (0.33, 0.77, 0.53)


@dataclass(frozen=True)
class SynthSpec:
    n_frames: int = 300
    fps: float = 30.0
    height: int = 96
    width: int = 128
    breathing_hz: float = 0.25
    heart_hz: float = 1.2
    breathing_amplitude_px: float = 2.0
    heart_amplitude: float = 3.0  # peak change of the face's green channel, in 8-bit levels
    breathing_color_amplitude: float = 0.0  # same, for the breathing part of the face color
    head_breathing_amplitude_px: float = 0.0  # vertical motion of the face with breathing
    noise_std: float = 0.0
    seed: int = 0
    face_center_frac: tuple[float, float] = (0.3, 0.5)  # (row, column) / (height, width)
    face_radius_frac: tuple[float, float] = (0.18, 0.12)  # (row, column) / (height, width)
    shoulder_half_width_frac: float = 0.2  # half the shoulder width / width (landmarks only)

    @property
    def shoulder_y(self) -> float:
        """Resting row of the shoulder line."""
        return 0.55 * self.height

    @property
    def face_center(self) -> tuple[float, float]:
        """(row, column) of the face ellipse center."""
        return self.face_center_frac[0] * self.height, self.face_center_frac[1] * self.width

    @property
    def face_radius(self) -> tuple[float, float]:
        """(row, column) semi-axes of the face ellipse in pixels."""
        return self.face_radius_frac[0] * self.height, self.face_radius_frac[1] * self.width


def _time(spec: SynthSpec) -> np.ndarray:
    return np.arange(spec.n_frames) / spec.fps


def breathing_displacement(spec: SynthSpec) -> np.ndarray:
    """Downward displacement of the torso in pixels for each frame (negative = up)."""
    return -spec.breathing_amplitude_px * np.sin(2 * np.pi * spec.breathing_hz * _time(spec))


def head_displacement(spec: SynthSpec) -> np.ndarray:
    """Downward displacement of the face in pixels for each frame, in phase with the torso."""
    return -spec.head_breathing_amplitude_px * np.sin(2 * np.pi * spec.breathing_hz * _time(spec))


def heart_modulation(spec: SynthSpec) -> np.ndarray:
    """Heart part of the face's green-channel change in 8-bit levels for each frame."""
    return spec.heart_amplitude * np.sin(2 * np.pi * spec.heart_hz * _time(spec))


def breathing_color_modulation(spec: SynthSpec) -> np.ndarray:
    """Breathing part of the face's green-channel change in 8-bit levels for each frame."""
    return spec.breathing_color_amplitude * np.sin(2 * np.pi * spec.breathing_hz * _time(spec))


def make_video(spec: SynthSpec) -> np.ndarray:
    """Render the video as a (T, H, W, 3) uint8 RGB array."""
    rng = np.random.default_rng(spec.seed)
    rows = np.arange(spec.height, dtype=np.float64)[:, None]
    cols = np.arange(spec.width, dtype=np.float64)[None, :]

    background = 60.0 + 10.0 * np.sin(2 * np.pi * cols / 37.0) * np.cos(2 * np.pi * rows / 29.0)
    (face_r, face_c), (rad_r, rad_c) = spec.face_center, spec.face_radius
    face_mask = ((rows - face_r) / rad_r) ** 2 + ((cols - face_c) / rad_c) ** 2 <= 1.0
    skin = np.array([200.0, 150.0, 120.0])
    pulse_dir = np.array(PULSE_RGB) / PULSE_RGB[1]  # green change = 1 level per unit

    displacement = breathing_displacement(spec)
    head = head_displacement(spec)
    blood = heart_modulation(spec) + breathing_color_modulation(spec)
    frames = np.empty((spec.n_frames, spec.height, spec.width, 3), dtype=np.uint8)
    for i in range(spec.n_frames):
        d = displacement[i]
        # Torso texture moves rigidly with the shoulder line; anti-aliased top edge.
        torso = 140.0 + 30.0 * np.sin(2 * np.pi * cols / 23.0) * np.cos(
            2 * np.pi * (rows - d) / 17.0
        )
        coverage = np.clip(rows - (spec.shoulder_y + d) + 0.5, 0.0, 1.0)
        gray = coverage * torso + (1.0 - coverage) * background
        frame = np.repeat(gray[:, :, None], 3, axis=2)
        if spec.head_breathing_amplitude_px:  # moving face with an anti-aliased outline
            e = np.sqrt(((rows - face_r - head[i]) / rad_r) ** 2 + ((cols - face_c) / rad_c) ** 2)
            alpha = np.clip((1.0 - e) * min(rad_r, rad_c) + 0.5, 0.0, 1.0)[:, :, None]
            frame = (1.0 - alpha) * frame + alpha * (skin + blood[i] * pulse_dir)
        else:
            frame[face_mask] = skin + blood[i] * pulse_dir
        if spec.noise_std > 0:
            frame = frame + rng.normal(0.0, spec.noise_std, frame.shape)
        frames[i] = np.clip(np.round(frame), 0, 255).astype(np.uint8)
    return frames


def synth_landmarks(spec: SynthSpec) -> np.ndarray:
    """Pose landmarks (33, 3) of the synthetic scene: normalized x, y and visibility.

    Shoulders sit on the resting shoulder line; eyes, nose, mouth and ears are placed inside
    the face ellipse roughly where they are on a real face. Other landmarks are not visible.
    """
    lm = np.full((roi.N_LANDMARKS, 3), np.nan)
    (fr, fc), (rr, rc) = spec.face_center, spec.face_radius
    eye_y, mouth_y = fr - 0.25 * rr, fr + 0.5 * rr
    points = {
        roi.NOSE: (fc, fr + 0.1 * rr),
        1: (fc + 0.2 * rc, eye_y),  # left eye inner (the subject's left is image right)
        2: (fc + 0.4 * rc, eye_y),
        3: (fc + 0.6 * rc, eye_y),  # left eye outer
        4: (fc - 0.2 * rc, eye_y),
        5: (fc - 0.4 * rc, eye_y),
        6: (fc - 0.6 * rc, eye_y),  # right eye outer
        7: (fc + 1.0 * rc, eye_y),  # left ear
        8: (fc - 1.0 * rc, eye_y),
        9: (fc + 0.35 * rc, mouth_y),  # mouth left
        10: (fc - 0.35 * rc, mouth_y),
    }
    half = spec.shoulder_half_width_frac * spec.width
    points[roi.LEFT_SHOULDER] = (0.5 * spec.width + half, spec.shoulder_y)
    points[roi.RIGHT_SHOULDER] = (0.5 * spec.width - half, spec.shoulder_y)
    lm[:, 2] = 0.0
    for index, (x, y) in points.items():
        lm[index] = (x / spec.width, y / spec.height, 0.99)
    return lm


def synthetic_ppg(
    n: int,
    fs: float,
    heart_hz: float,
    breathing_hz: float,
    baseline: float = 0.0,
    amplitude_depth: float = 0.0,
    interval_depth: float = 0.0,
    noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Contact-PPG-like waveform with breathing modulations.

    baseline:        breathing-rate wander added to the waveform (units of pulse amplitude)
    amplitude_depth: relative pulse-amplitude modulation at the breathing rate
    interval_depth:  relative heart-rate modulation at the breathing rate (sinus arrhythmia)

    The pulse shape sin(p) + 0.25 sin(2p) has exactly one peak and one trough per beat.
    """
    t = np.arange(n) / fs
    breath = np.sin(2 * np.pi * breathing_hz * t)
    rate = heart_hz * (1.0 + interval_depth * breath)
    phase = 2 * np.pi * np.cumsum(rate) / fs
    pulse = np.sin(phase) + 0.25 * np.sin(2 * phase)
    ppg = (1.0 + amplitude_depth * breath) * pulse + baseline * breath
    if noise_std > 0:
        ppg = ppg + np.random.default_rng(seed).normal(0.0, noise_std, n)
    return ppg
