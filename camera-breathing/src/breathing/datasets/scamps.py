"""SCAMPS reader (McDuff et al., NeurIPS 2022).

Each video is a MATLAB v7.3 .mat file, which is HDF5. MATLAB stores arrays column-major,
so h5py shows the axes in reverse order; reversing them gives MATLAB's order, which for
the frame arrays is expected to be (T, H, W, 3). Phase 1 records the raw layout so this
assumption can be checked on the real files.

The full dataset ships as one ~600 GB tar.gz, so videos are read by streaming the archive
and writing one member at a time to local disk.
"""

from __future__ import annotations

import csv
import io
import shutil
import tarfile
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

TIMEOUT_S = 60


@dataclass
class ScampsVideo:
    frames: np.ndarray  # (T, H, W, 3) uint8 RGB
    waveforms: dict[str, np.ndarray]  # 1-D, one sample per frame
    layout: dict  # how the frame array was interpreted


def list_variables(h5) -> list[dict]:
    """Name, shape (as stored by h5py) and dtype of every dataset in the file."""
    found = []

    def visit(name, obj):
        if name.startswith("#refs#"):
            return
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            found.append({"name": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})

    h5.visititems(visit)
    return found


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Floats in [0, 1] are scaled to 0-255; other values are rounded and clipped."""
    if array.dtype == np.uint8:
        return array
    values = np.asarray(array, dtype=np.float32)
    if np.issubdtype(array.dtype, np.floating) and np.nanmax(values) <= 1.0:
        values = values * 255.0
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def frames_to_thwc(stored: np.ndarray, n_frames: int) -> tuple[np.ndarray, dict]:
    """Interpret a stored frame array as (T, H, W, 3).

    First tries MATLAB order (all axes reversed). If that does not give time first and
    color last, falls back to locating those axes by size and flags it in the layout.
    """
    layout = {"stored_shape": list(stored.shape), "stored_dtype": str(stored.dtype)}
    if stored.ndim != 4:
        raise ValueError(f"expected a 4-D frame array, got shape {stored.shape}")
    matlab = stored.transpose(3, 2, 1, 0)
    if matlab.shape[0] == n_frames and matlab.shape[3] == 3:
        layout["interpretation"] = "matlab_order"
        frames = matlab
    else:
        axes = list(range(4))
        t_axis = next((a for a in axes if stored.shape[a] == n_frames), None)
        c_axis = next((a for a in axes if stored.shape[a] == 3 and a != t_axis), None)
        if t_axis is None or c_axis is None:
            raise ValueError(f"cannot find time ({n_frames}) and color axes in {stored.shape}")
        rest = [a for a in axes if a not in (t_axis, c_axis)]
        layout["interpretation"] = "by_size"
        frames = stored.transpose(t_axis, *rest, c_axis)
    layout["value_min"] = float(np.min(frames))
    layout["value_max"] = float(np.max(frames))
    frames = to_uint8(frames)
    layout["frame_shape"] = list(frames.shape[1:])
    return frames, layout


def read_video(path: str | Path, frames_key: str, waveform_keys: dict[str, str]) -> ScampsVideo:
    import h5py

    with h5py.File(path, "r") as h5:
        waveforms = {
            name: np.asarray(h5[key][()], dtype=np.float64).ravel()
            for name, key in waveform_keys.items()
        }
        lengths = {len(w) for w in waveforms.values()}
        if len(lengths) != 1:
            raise ValueError(f"waveforms have different lengths: {lengths}")
        frames, layout = frames_to_thwc(np.asarray(h5[frames_key][()]), lengths.pop())
    return ScampsVideo(frames=frames, waveforms=waveforms, layout=layout)


def small_numeric_values(h5, max_size: int = 4) -> dict[str, list[float]]:
    """Datasets with at most `max_size` numbers, e.g. per-video labels such as rates."""
    found = {}

    def visit(name, obj):
        if name.startswith("#refs#") or not hasattr(obj, "dtype"):
            return
        if obj.size <= max_size and np.issubdtype(obj.dtype, np.number):
            found[name] = np.asarray(obj[()], dtype=np.float64).ravel().tolist()

    h5.visititems(visit)
    return found


@contextmanager
def open_source(source: str):
    """Open a local path or a URL (http, https, file) as a binary stream."""
    if "://" in source:
        with urllib.request.urlopen(source, timeout=TIMEOUT_S) as response:
            yield response
    else:
        with open(source, "rb") as f:
            yield f


def iter_tar_members(
    source: str, work_dir: str | Path, suffix: str = ".mat", skip: set[str] | None = None
) -> Iterator[tuple[str, Path]]:
    """Stream a tar.gz and yield (member name, local path) one file at a time.

    Each file is written to work_dir and deleted once the caller moves on to the next one.
    Members in `skip` are read past without being written. Member paths are reduced to their
    file name, so an archive cannot write outside work_dir.
    """
    skip = skip or set()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    with open_source(source) as stream, tarfile.open(fileobj=stream, mode="r|*") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(suffix) or member.name in skip:
                continue
            local = work_dir / PurePosixPath(member.name).name
            with tar.extractfile(member) as src, open(local, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            try:
                yield member.name, local
            finally:
                local.unlink(missing_ok=True)


def summarize_csv_tar(source: str) -> list[dict]:
    """File name, size, columns and row count of every CSV in a tar.gz (no values kept)."""
    out = []
    with open_source(source) as stream, tarfile.open(fileobj=stream, mode="r|*") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            with tar.extractfile(member) as f:
                text = f.read().decode("utf-8", errors="replace")
            reader = csv.reader(io.StringIO(text))
            header = next(reader, [])
            n_rows = sum(1 for _ in reader)
            out.append(
                {
                    "name": member.name,
                    "size_bytes": member.size,
                    "columns": header,
                    "n_rows": n_rows,
                }
            )
    return out


def http_head(url: str) -> dict:
    """Size and range support of a download link, without downloading it."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            headers = response.headers
            length = headers.get("Content-Length")
            return {
                "url": url,
                "status": response.status,
                "size_bytes": int(length) if length else None,
                "accept_ranges": headers.get("Accept-Ranges"),
                "last_modified": headers.get("Last-Modified"),
            }
    except OSError as err:
        return {"url": url, "error": str(err)}
