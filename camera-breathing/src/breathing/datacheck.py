"""Phase-1 data checks: per-video specs, ground-truth timing, body visibility, and for
SCAMPS whether breathing shows up as body motion. Records hold numbers only; no frames,
landmark tracks or signals are kept.
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from breathing import roi
from breathing.datasets import scamps
from breathing.rate import bandpass, dominant_frequency
from breathing.signals import region_shift, to_gray
from breathing.video_io import iter_frames, probe, write_png

PoseFn = Callable[[np.ndarray], np.ndarray]  # RGB frame -> (33, 3) landmarks or NaN

HEART_BAND_HZ = (0.7, 3.0)


def timing_stats(t: np.ndarray) -> dict:
    """Sampling rate and gaps of a timestamp series in seconds."""
    t = np.asarray(t, dtype=np.float64)
    if len(t) < 2:
        return {"n": int(len(t)), "fs_hz": None, "duration_s": None}
    dt = np.diff(t)
    median_dt = float(np.median(dt))
    return {
        "n": int(len(t)),
        "fs_hz": 1.0 / median_dt if median_dt > 0 else None,
        "duration_s": float(t[-1] - t[0]),
        "dt_min_s": float(dt.min()),
        "dt_max_s": float(dt.max()),
        "monotonic": bool(np.all(dt > 0)),
    }


def value_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    return {
        "n": int(len(x)),
        "min": float(finite.min()) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "n_nonfinite": int(len(x) - finite.size),
        "n_zero": int(np.sum(finite == 0)),
    }


def lagged_correlation(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float | None, int]:
    """Pearson r at the lag (in samples) with the largest |r|; positive lag = a lags b."""
    best_r, best_lag = None, 0
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b[: len(b) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if best_r is None or abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    return best_r, best_lag


def _to_bpm(hz: float | None) -> float | None:
    return None if hz is None else 60.0 * hz


def motion_probe(
    gray: np.ndarray,
    boxes: dict[str, roi.Box | None],
    fps: float,
    reference: np.ndarray,
    band_hz: tuple[float, float],
    max_lag_s: float,
    match_tolerance_bpm: float,
) -> dict:
    """Does the vertical motion of each region follow the breathing reference?

    Horizontal motion is reported as a control: breathing should mostly move the body up
    and down.
    """
    low, high = band_hz
    ref = bandpass(reference, fps, low, high)
    ref_bpm = _to_bpm(dominant_frequency(ref, fps, low, high))
    max_lag = int(round(max_lag_s * fps))
    out = {"reference_peak_bpm": ref_bpm}
    for name, box in boxes.items():
        if box is None:
            out[name] = None
            continue
        dy, dx = region_shift(gray, box)
        result = {"box": list(box)}
        for axis, raw in (("dy", dy), ("dx", dx)):
            filtered = bandpass(raw, fps, low, high)
            peak = _to_bpm(dominant_frequency(filtered, fps, low, high))
            r, lag = lagged_correlation(filtered, ref, max_lag)
            r0, _ = lagged_correlation(filtered, ref, 0)
            error = None if peak is None or ref_bpm is None else peak - ref_bpm
            result[axis] = {
                "peak_bpm": peak,
                "peak_error_bpm": error,
                "peak_match": None if error is None else abs(error) <= match_tolerance_bpm,
                "corr_zero_lag": r0,
                "corr_best": r,
                "corr_best_lag_s": lag / fps,
                "std_px": float(np.std(filtered)),
            }
        out[name] = result
    return out


def sample_pose(
    frames: Iterable[np.ndarray], fps: float | None, sample_fps: float, pose: PoseFn | None
) -> tuple[np.ndarray, int, np.ndarray | None]:
    """Run pose on every k-th frame. Returns (landmarks, frames seen, first frame)."""
    step = max(1, int(round(fps / sample_fps))) if fps else 1
    landmarks, n, first = [], 0, None
    for i, frame in enumerate(frames):
        if i == 0:
            first = frame.copy()
        if pose is not None and i % step == 0:
            landmarks.append(pose(frame))
        n += 1
    stacked = np.array(landmarks).reshape(-1, roi.N_LANDMARKS, 3)
    return stacked, n, first


def draw_boxes(image: np.ndarray, boxes: dict[str, roi.Box | None]) -> np.ndarray:
    """Copy of the image with each box outlined (for synthetic frames only)."""
    colors = {"head": (255, 64, 64), "shoulder": (64, 255, 64), "chest": (64, 128, 255)}
    out = image.copy()
    for name, box in boxes.items():
        if box is None:
            continue
        r0, r1, c0, c1 = box
        color = colors.get(name, (255, 255, 0))
        out[r0, c0:c1] = out[r1 - 1, c0:c1] = color
        out[r0:r1, c0] = out[r0:r1, c1 - 1] = color
    return out


def check_scamps_video(
    path: str | Path, cfg: dict, pose: PoseFn | None, figure_path: str | Path | None = None
) -> dict:
    import h5py

    started = time.perf_counter()
    fps = float(cfg["fps"])
    with h5py.File(path, "r") as h5:
        variables = scamps.list_variables(h5)
        small_values = scamps.small_numeric_values(h5)
        has_frames = cfg["frames_key"] in h5
    record = {
        "file": {"size_bytes": Path(path).stat().st_size},
        "variables": variables,
        "small_numeric_values": small_values,
    }
    if not has_frames:
        record["note"] = f"no {cfg['frames_key']} variable; not a video file"
        return record

    video = scamps.read_video(path, cfg["frames_key"], cfg["waveform_keys"])
    read_s = time.perf_counter() - started
    n_frames, height, width = video.frames.shape[:3]
    breathing = video.waveforms["breathing"]
    low, high = cfg["probe"]["band_hz"]
    record["video"] = {
        "n_frames": int(n_frames),
        "height": int(height),
        "width": int(width),
        "fps_assumed": fps,
        "duration_s": n_frames / fps,
        "layout": video.layout,
    }
    record["ground_truth"] = {
        name: {"n": int(len(w)), "matches_frames": len(w) == n_frames}
        for name, w in video.waveforms.items()
    }
    record["ground_truth"]["breathing_peak_bpm"] = _to_bpm(
        dominant_frequency(breathing, fps, low, high)
    )
    if "ppg" in video.waveforms:
        record["ground_truth"]["heart_peak_bpm"] = _to_bpm(
            dominant_frequency(video.waveforms["ppg"], fps, *HEART_BAND_HZ)
        )

    pose_cfg = cfg["pose"]
    landmarks, _, first = sample_pose(video.frames, fps, pose_cfg["sample_fps"], pose)
    threshold = pose_cfg["visibility_threshold"]
    if pose is not None:
        record["visibility"] = roi.visibility_summary(landmarks, width, height, threshold)
    boxes, box_source = roi.probe_regions(landmarks, width, height, threshold)

    probe_cfg = cfg["probe"]
    record["motion_probe"] = {
        "region_source": box_source,
        **motion_probe(
            to_gray(video.frames),
            boxes,
            fps,
            breathing,
            tuple(probe_cfg["band_hz"]),
            probe_cfg["max_lag_s"],
            probe_cfg["match_tolerance_bpm"],
        ),
    }
    if figure_path is not None:
        write_png(draw_boxes(first, boxes), figure_path)
    record["timing_s"] = {"read": read_s, "total": time.perf_counter() - started}
    return record


def check_ubfc_subject(video_path: str | Path, gt: dict, cfg: dict, pose: PoseFn | None) -> dict:
    started = time.perf_counter()
    info = probe(video_path)
    pose_cfg = cfg["pose"]
    landmarks, n_decoded, _ = sample_pose(
        iter_frames(video_path, info.width, info.height), info.fps, pose_cfg["sample_fps"], pose
    )
    record = {
        "file": {"size_bytes": Path(video_path).stat().st_size},
        "video": {
            **info.to_dict(),
            "n_decoded_frames": n_decoded,
            "decoded_duration_s": n_decoded / info.fps if info.fps else None,
        },
        "ground_truth": {
            "timing": timing_stats(gt["t"]),
            "n_minus_frames": int(len(gt["t"]) - n_decoded),
            "ppg": value_stats(gt["ppg"]),
            "hr_bpm": value_stats(gt["hr"]),
        },
    }
    if pose is not None:
        record["visibility"] = roi.visibility_summary(
            landmarks, info.width, info.height, pose_cfg["visibility_threshold"]
        )
    record["timing_s"] = {"total": time.perf_counter() - started}
    return record


# ---------------------------------------------------------------------------------------
# Summaries


def get(record: dict, dotted: str):
    value = record
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def flatten(record: dict, prefix: str = "") -> dict:
    """Nested dict -> {"a.b.c": value}; lists are kept as text."""
    flat = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, name + "."))
        elif isinstance(value, list):
            flat[name] = repr(value) if len(repr(value)) <= 200 else f"<list of {len(value)}>"
        else:
            flat[name] = value
    return flat


def _fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _spread(values: list) -> str:
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not values:
        return "–"
    arr = np.array(values, dtype=np.float64)
    return f"{_fmt(arr.min())} / {_fmt(float(np.median(arr)))} / {_fmt(arr.max())}"


def _counts(values: list) -> str:
    counter = Counter(_fmt(v) for v in values)
    return ", ".join(f"{k} ({n})" for k, n in counter.most_common()) or "–"


def _count_true(values: list) -> str:
    known = [v for v in values if v is not None]
    return f"{sum(bool(v) for v in known)} of {len(known)}" if known else "–"


SPECS = {
    "ubfc": [
        ("File size (bytes)", "file.size_bytes", _spread),
        ("Container", "video.format_name", _counts),
        ("Codec", "video.codec_name", _counts),
        ("FourCC", "video.codec_tag", _counts),
        ("Pixel format", "video.pix_fmt", _counts),
        ("Width", "video.width", _counts),
        ("Height", "video.height", _counts),
        ("Frame rate (fps)", "video.fps", _spread),
        ("Frames (header)", "video.nb_frames", _spread),
        ("Frames (decoded)", "video.n_decoded_frames", _spread),
        ("Duration (s, decoded)", "video.decoded_duration_s", _spread),
        ("Bit rate (bit/s)", "video.bit_rate", _spread),
        ("GT samples", "ground_truth.timing.n", _spread),
        ("GT sampling rate (Hz)", "ground_truth.timing.fs_hz", _spread),
        ("GT duration (s)", "ground_truth.timing.duration_s", _spread),
        ("GT largest gap (s)", "ground_truth.timing.dt_max_s", _spread),
        ("GT timestamps monotonic", "ground_truth.timing.monotonic", _count_true),
        ("GT samples minus frames", "ground_truth.n_minus_frames", _spread),
        ("Oximeter HR median (bpm)", "ground_truth.hr_bpm.median", _spread),
        ("Oximeter HR zeros", "ground_truth.hr_bpm.n_zero", _spread),
        ("Copy from Drive + check per video (s)", "timing_s.copy_and_check", _spread),
    ],
    "scamps": [
        ("File size (bytes)", "file.size_bytes", _spread),
        ("Frame array as stored", "video.layout.stored_shape", _counts),
        ("Frame dtype as stored", "video.layout.stored_dtype", _counts),
        ("Interpretation", "video.layout.interpretation", _counts),
        ("Value range min", "video.layout.value_min", _spread),
        ("Value range max", "video.layout.value_max", _spread),
        ("Height", "video.height", _counts),
        ("Width", "video.width", _counts),
        ("Frames", "video.n_frames", _spread),
        ("Duration (s, at assumed fps)", "video.duration_s", _spread),
        ("Breathing waveform matches frames", "ground_truth.breathing.matches_frames", _count_true),
        ("PPG waveform matches frames", "ground_truth.ppg.matches_frames", _count_true),
        ("Breathing rate from waveform (bpm)", "ground_truth.breathing_peak_bpm", _spread),
        ("Heart rate from PPG (bpm)", "ground_truth.heart_peak_bpm", _spread),
        ("Probe regions from", "motion_probe.region_source", _counts),
    ],
}

VISIBILITY = [
    ("Pose detected (fraction of sampled frames)", "visibility.pose_detected_frac", _spread),
    ("Nose visible", "visibility.nose_visible_frac", _spread),
    ("Both shoulders visible", "visibility.shoulders_visible_frac", _spread),
    ("A hip visible", "visibility.hip_visible_frac", _spread),
    ("Shoulder line (fraction of height)", "visibility.shoulder_y_frac_median", _spread),
    ("Shoulder width (px)", "visibility.shoulder_width_px_median", _spread),
    ("Rows below shoulders (px, median)", "visibility.rows_below_shoulders_px_median", _spread),
    ("Rows below shoulders (px, min)", "visibility.rows_below_shoulders_px_min", _spread),
    (
        "Rows below shoulders / shoulder width",
        "visibility.rows_below_shoulders_per_shoulder_width_median",
        _spread,
    ),
    ("Face width (px)", "visibility.face_width_px_median", _spread),
    ("Processing time per video (s)", "timing_s.total", _spread),
]

PROBE_REGIONS = ("shoulder", "chest", "head")


def summarize(records: list[dict], dataset: str) -> tuple[dict, str]:
    """Dataset spec table from the per-item records. Cells show min / median / max."""
    ok = [r for r in records if r.get("status") == "ok"]
    videos = [r for r in ok if "video" in r]
    errors = [r for r in records if r.get("status") != "ok"]
    total_bytes = sum(get(r, "file.size_bytes") or 0 for r in videos)
    rows = [
        ("Items processed", str(len(records))),
        ("Videos checked", str(len(videos))),
        ("Errors", str(len(errors))),
        ("Total size of checked videos (GB)", _fmt(total_bytes / 1e9)),
    ]
    for label, path, fn in SPECS[dataset] + VISIBILITY:
        rows.append((label, fn([get(r, path) for r in videos])))

    lines = [
        f"# Data check: {dataset}",
        "",
        "Cells show min / median / max over videos, or value (count).",
        "",
    ]
    lines += ["| Quantity | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows]
    probe_rows = []
    if dataset == "scamps":
        for region in PROBE_REGIONS:
            for axis in ("dy", "dx"):
                base = f"motion_probe.{region}.{axis}"
                best_r = [get(r, f"{base}.corr_best") for r in videos]
                probe_rows.append(
                    (
                        region,
                        "vertical" if axis == "dy" else "horizontal",
                        _count_true([get(r, f"{base}.peak_match") for r in videos]),
                        _spread([get(r, f"{base}.corr_zero_lag") for r in videos]),
                        _spread([abs(v) for v in best_r if v is not None]),
                        _spread([get(r, f"{base}.std_px") for r in videos]),
                    )
                )
        lines += [
            "",
            "## Does breathing show up as body motion?",
            "",
            "Motion of each region (global optical flow), band-passed and compared with `d_br`.",
            "A peak match only means something when the motion std is clearly above zero.",
            "",
            "| Region | Direction | Peak matches reference | r at zero lag "
            "| best abs(r) within max lag | Motion std (px) |",
            "|---|---|---|---|---|---|",
        ]
        lines += [f"| {' | '.join(row)} |" for row in probe_rows]
    if errors:
        lines += ["", "## Errors", ""] + [f"- `{r['id']}`: {r.get('error')}" for r in errors]
    summary = {
        "dataset": dataset,
        "rows": [list(row) for row in rows],
        "probe_rows": [list(row) for row in probe_rows],
    }
    return summary, "\n".join(lines) + "\n"


def write_items_csv(records: list[dict], path: str | Path) -> None:
    flat = [flatten({k: v for k, v in r.items() if k != "variables"}) for r in records]
    columns = sorted({key for row in flat for key in row})
    columns = ["id", "status"] + [c for c in columns if c not in ("id", "status")]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
