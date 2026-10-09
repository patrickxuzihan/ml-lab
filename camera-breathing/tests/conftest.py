"""Shared fixtures. Everything is synthetic: no real recordings are used in tests."""

from __future__ import annotations

import importlib.util
import tarfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from breathing import roi
from breathing.synth import SynthSpec, breathing_displacement, heart_modulation, make_video
from breathing.video_io import write_video

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = SynthSpec(n_frames=600, fps=30.0, height=64, width=96, breathing_hz=0.3, heart_hz=1.5)


@pytest.fixture(scope="session")
def spec() -> SynthSpec:
    return SPEC


@pytest.fixture(scope="session")
def frames(spec) -> np.ndarray:
    return make_video(spec)


def write_ubfc_subject(folder: Path, frames: np.ndarray, spec: SynthSpec) -> None:
    """A UBFC-rPPG-style subject folder: uncompressed vid.avi and ground_truth.txt."""
    folder.mkdir(parents=True, exist_ok=True)
    spec = replace(spec, n_frames=len(frames))
    write_video(frames, folder / "vid.avi", spec.fps)
    t = np.arange(len(frames)) / spec.fps
    ppg = heart_modulation(spec)
    hr = np.full(len(frames), 60.0 * spec.heart_hz)
    lines = [" ".join(f"{v:.8e}" for v in row) for row in (ppg, hr, t)]
    (folder / "ground_truth.txt").write_text("\n".join(lines) + "\n")


def write_scamps_mat(path: Path, frames: np.ndarray, spec: SynthSpec, dtype=np.uint8) -> None:
    """A SCAMPS-style MATLAB v7.3 file: arrays stored with their axes reversed, as MATLAB does."""
    import h5py

    stored = frames.transpose(3, 2, 1, 0)  # (T, H, W, 3) in MATLAB -> (3, W, H, T) in HDF5
    if np.issubdtype(dtype, np.floating):
        stored = stored.astype(dtype) / 255.0
    with h5py.File(path, "w") as h5:
        h5["RawFrames"] = stored
        h5["d_br"] = breathing_displacement(spec)[None, :]  # MATLAB column vector -> (1, T)
        h5["d_ppg"] = heart_modulation(spec)[None, :]
        h5["hr_label"] = np.array([[60.0 * spec.heart_hz]])


def write_tar_gz(path: Path, files: dict[str, Path]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for arcname, src in files.items():
            tar.add(src, arcname=arcname)


def fake_pose(spec: SynthSpec):
    """Pose function that places shoulders on the synthetic shoulder line and a face above."""
    landmarks = np.full((roi.N_LANDMARKS, 3), np.nan)
    landmarks[:, 2] = 0.9
    face_r, face_c = spec.face_center
    for i in roi.FACE:
        landmarks[i, :2] = (face_c + (i - 5) * 1.0) / spec.width, face_r / spec.height
    y = spec.shoulder_y / spec.height
    landmarks[roi.LEFT_SHOULDER, :2] = 0.7, y
    landmarks[roi.RIGHT_SHOULDER, :2] = 0.3, y
    landmarks[roi.LEFT_HIP, :2] = 0.65, 1.3  # below the frame: not visible
    landmarks[roi.RIGHT_HIP, :2] = 0.35, 1.3
    landmarks[roi.LEFT_HIP + 2 :, :] = np.nan
    landmarks[roi.LEFT_HIP + 2 :, 2] = 0.0

    def pose(_frame: np.ndarray) -> np.ndarray:
        return landmarks.copy()

    return pose


def load_script(name: str):
    path = PROJECT_DIR / "scripts" / f"{name}.py"
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module
