"""Synthetic test videos with known breathing and heart rates.

The scene is a textured torso whose top edge (the shoulder line) moves up and down at the
breathing rate, and a face ellipse whose skin color brightens and darkens at the heart rate.
Everything is rendered analytically, so sub-pixel motion is exact. Phase 2 extends this
generator; phase 1 only needs it as a test fixture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SynthSpec:
    n_frames: int = 300
    fps: float = 30.0
    height: int = 96
    width: int = 128
    breathing_hz: float = 0.25
    heart_hz: float = 1.2
    breathing_amplitude_px: float = 2.0
    heart_amplitude: float = 3.0  # peak change of the face color, in 8-bit levels
    noise_std: float = 0.0
    seed: int = 0

    @property
    def shoulder_y(self) -> float:
        """Resting row of the shoulder line."""
        return 0.55 * self.height

    @property
    def face_center(self) -> tuple[float, float]:
        """(row, column) of the face ellipse center."""
        return 0.3 * self.height, 0.5 * self.width


def breathing_displacement(spec: SynthSpec) -> np.ndarray:
    """Downward displacement of the torso in pixels for each frame (negative = up)."""
    t = np.arange(spec.n_frames) / spec.fps
    return -spec.breathing_amplitude_px * np.sin(2 * np.pi * spec.breathing_hz * t)


def heart_modulation(spec: SynthSpec) -> np.ndarray:
    """Change of the face color in 8-bit levels for each frame."""
    t = np.arange(spec.n_frames) / spec.fps
    return spec.heart_amplitude * np.sin(2 * np.pi * spec.heart_hz * t)


def make_video(spec: SynthSpec) -> np.ndarray:
    """Render the video as a (T, H, W, 3) uint8 RGB array."""
    rng = np.random.default_rng(spec.seed)
    rows = np.arange(spec.height, dtype=np.float64)[:, None]
    cols = np.arange(spec.width, dtype=np.float64)[None, :]

    background = 60.0 + 10.0 * np.sin(2 * np.pi * cols / 37.0) * np.cos(2 * np.pi * rows / 29.0)
    face_r, face_c = spec.face_center
    face_mask = ((rows - face_r) / (0.18 * spec.height)) ** 2 + (
        (cols - face_c) / (0.12 * spec.width)
    ) ** 2 <= 1.0
    skin = np.array([200.0, 150.0, 120.0])

    displacement = breathing_displacement(spec)
    modulation = heart_modulation(spec)
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
        frame[face_mask] = skin + modulation[i]
        if spec.noise_std > 0:
            frame = frame + rng.normal(0.0, spec.noise_std, frame.shape)
        frames[i] = np.clip(np.round(frame), 0, 255).astype(np.uint8)
    return frames
