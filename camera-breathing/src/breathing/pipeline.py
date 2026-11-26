"""The evaluation protocol applied to one video: regions -> traces -> windows -> rates.

Dataset scripts provide three things per source video: a function that yields its RGB
frames (called once per version, so long videos are streamed, not held in memory), the
regions found on the uncompressed source, and the reference rates per window. Every
compressed version is encoded to a temporary file, decoded, measured and deleted.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from breathing import compression, metrics, rate, reference
from breathing.compression import Setting
from breathing.roi import Box, scale_box
from breathing.signals import Traces, extract_traces, pos
from breathing.video_io import iter_frames, keyframe_count, write_video

PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "configs" / "protocol.yaml"
MEASURES = ("M1", "M2", "M3")  # the full-frame measurements; M4 is for face-only framing
RELOCATED = "_relocated"  # suffix of the regions found again on a compressed version

FrameSource = Callable[[], Iterator[np.ndarray]]


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def rate_kwargs(protocol: dict) -> dict:
    r = protocol["rate"]
    return {
        "order": r["filter_order"],
        "min_nfft": r["min_nfft"],
        "max_nyquist_frac": r["max_nyquist_frac"],
        "snr_half_width_bins": r["snr_half_width_bins"],
    }


def analysis_windows(
    duration_s: float, protocol: dict, measures: tuple[str, ...] = MEASURES
) -> dict[str, list[tuple[float, float]]]:
    """Analysis windows (seconds) of each measurement for a video of this length."""
    return {
        m: rate.windows_s(duration_s, protocol["measurements"][m]["window_s"]) for m in measures
    }


def waveform_references(
    waves: dict[str, np.ndarray],
    fs: float,
    windows: dict[str, list[tuple[float, float]]],
    protocol: dict,
) -> dict[str, list[float | None]]:
    """Reference rate (bpm) per measurement and window from known waveforms.

    waves maps a measurement kind ("breathing", "heart") to a waveform sampled at fs, e.g.
    SCAMPS `d_br` and `d_ppg`. The waveforms go through the same estimator as the camera
    signals.
    """
    out = {}
    for m, cfg in ((m, protocol["measurements"][m]) for m in windows):
        rates = reference.waveform_rates(
            waves[cfg["kind"]], fs, windows[m], tuple(cfg["band_hz"]), protocol["rate"]
        )
        out[m] = [r["bpm"] for r in rates]
    return out


def measurement_signal(
    measure: str, traces: Traces, window: tuple[float, float], protocol: dict, cache: dict
) -> np.ndarray | None:
    """The raw (unfiltered) signal of one measurement in one window, or None without region."""
    cfg = protocol["measurements"][measure]
    if cfg["signal"] == "motion":
        series = traces.motion.get(cfg["region"])
        if series is None:
            return None
        return series[rate.window_slice(*window, traces.fps, len(series))]
    rgb = traces.rgb.get(cfg["region"])
    if rgb is None:
        return None
    if cfg["pos_window_s"] is None:  # one POS projection per analysis window
        return pos(rgb[rate.window_slice(*window, traces.fps, len(rgb))], traces.fps, None)
    key = (measure, "pos")
    if key not in cache:  # overlap-added POS over the whole trace, then cut
        cache[key] = pos(rgb, traces.fps, cfg["pos_window_s"])
    full = cache[key]
    return full[rate.window_slice(*window, traces.fps, len(full))]


def measure(
    traces: Traces,
    windows: dict[str, list[tuple[float, float]]],
    references: dict[str, list[float | None]],
    protocol: dict,
    measures: tuple[str, ...] = MEASURES,
) -> list[dict]:
    """One record per measurement (of `measures`) and window: estimate, reference, error,
    SNR, exclusion.

    A window is excluded (not counted in any metric) when the video has no region for the
    measurement, the reference is missing, or the reference lies outside the nominal band.
    These do not depend on the compression setting. A missing estimate is not an exclusion:
    it counts as a failure. The estimate does not depend on the reference, so it is recorded
    for windows excluded for their reference too (e.g. to compare versions of a video); the
    error and SNR need a reference.
    """
    out, cache = [], {}
    kwargs = rate_kwargs(protocol)
    for m in measures:
        cfg = protocol["measurements"][m]
        low, high = cfg["band_hz"]
        for k, window in enumerate(windows[m]):
            ref = references[m][k]
            record = {
                "measure": m,
                "window": k,
                "t0": window[0],
                "t1": window[1],
                "ref_bpm": ref,
                "est_bpm": None,
                "error_bpm": None,
                "snr_db": None,
                "excluded": None,
            }
            out.append(record)
            segment = measurement_signal(m, traces, window, protocol, cache)
            if segment is None:
                record["excluded"] = "no_region"
                continue
            if ref is None:
                record["excluded"] = "no_reference"
            elif not low * 60 <= ref <= high * 60:
                record["excluded"] = "reference_out_of_band"
            ref_hz = None if ref is None else ref / 60
            hz, snr = rate.analyze_window(segment, traces.fps, (low, high), ref_hz, **kwargs)
            record["snr_db"] = snr
            if hz is not None:
                record["est_bpm"] = 60 * hz
                if ref is not None:
                    record["error_bpm"] = 60 * hz - ref
    return out


@dataclass
class Version:
    """One version of a video: its frames (decoded on each call), frame rate and size."""

    frames: FrameSource
    fps: float
    size_hw: tuple[int, int]
    encoded: bool = False  # False for the untouched source
    info: dict = field(default_factory=dict)  # bytes, kbps, keyframes of an encoded version


@contextmanager
def encoded_version(
    setting: Setting,
    frames: FrameSource,
    fps: float,
    size_hw: tuple[int, int],
    n_frames: int,
    protocol: dict,
    work_dir: str | Path,
) -> Iterator[Version]:
    """The version of a video for one setting, for the duration of the block.

    The source is passed through. Any other setting is encoded to a temporary file in
    work_dir, decoded on demand and deleted at the end of the block.
    """
    if setting.codec is None:
        yield Version(frames, fps, size_hw)
        return
    height, width = size_hw
    args, ext = compression.ffmpeg_args(setting, width, height, protocol["compression"])
    path = Path(work_dir) / f"version_{setting.name}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fps_out = fps / setting.decimate
        write_video(compression.decimate(frames(), setting.decimate), path, fps_out, args)
        out_w, out_h = compression.output_size(width, height, setting.scale)
        n_out = compression.n_decimated(n_frames, setting.decimate)
        size = path.stat().st_size
        info = {
            "bytes": size,
            "kbps": size * 8 / (n_out / fps_out) / 1000,
            "keyframes": keyframe_count(path) if ext == ".mp4" else None,
        }
        yield Version(lambda: iter_frames(path, out_w, out_h), fps_out, (out_h, out_w), True, info)
    finally:
        path.unlink(missing_ok=True)


def process_version(
    setting: Setting,
    frames: FrameSource,
    fps: float,
    size_hw: tuple[int, int],
    n_frames: int,
    boxes: dict[str, Box | None],
    windows: dict[str, list[tuple[float, float]]],
    references: dict[str, list[float | None]],
    protocol: dict,
    work_dir: str | Path,
    variants: dict[str, tuple[dict, dict]] | None = None,
    relocate: Callable[[Iterator[np.ndarray], float, tuple[int, int]], tuple[dict, dict]]
    | None = None,
    measures: tuple[str, ...] = MEASURES,
) -> dict:
    """Measure one version (the source or a compressed copy) of a video.

    measures: the measurements to make; the regions traced follow from them (boxes holds
    one box per region name, e.g. "shoulder", "face", "head").
    variants: optional extra measurements on the same signals, {name: (windows, references)},
    recorded under record["variants"][name]. They never change record["windows"].
    relocate: optional function (frames, fps, (height, width)) -> (boxes, info) that finds the
    regions again on a decoded compressed version (sensitivity analysis). The version is then
    also measured with those regions, recorded under record["relocated"]. Not used for the
    source, whose regions are the reference.
    """
    started = time.perf_counter()
    record = {"setting": setting.name, **setting.to_dict()}
    relocated = None
    regions = {"motion": [], "color": []}
    for m in measures:
        cfg = protocol["measurements"][m]
        if cfg["region"] not in regions[cfg["signal"]]:
            regions[cfg["signal"]].append(cfg["region"])
    with encoded_version(setting, frames, fps, size_hw, n_frames, protocol, work_dir) as version:
        record.update(version.info)
        fps_out, out_hw = version.fps, version.size_hw
        if relocate is not None and version.encoded:
            relocated = relocate(version.frames(), fps_out, out_hw)
        scaled = {name: scale_box(box, size_hw, out_hw) for name, box in boxes.items()}
        motion = {r: scaled.get(r) for r in regions["motion"]}
        color = {r: scaled.get(r) for r in regions["color"]}
        if relocated is not None:  # both sets of regions in one pass over the frames
            motion.update({r + RELOCATED: relocated[0].get(r) for r in regions["motion"]})
            color.update({r + RELOCATED: relocated[0].get(r) for r in regions["color"]})
        traces = extract_traces(version.frames(), fps_out, motion, color)
    record.update(
        fps=fps_out,
        height=out_hw[0],
        width=out_hw[1],
        n_frames=traces.n_frames,
        windows=measure(traces, windows, references, protocol, measures),
    )
    if variants:
        record["variants"] = {
            name: measure(traces, w, r, protocol, measures) for name, (w, r) in variants.items()
        }
    if relocated is not None:
        new_boxes, info = relocated
        moved = Traces(
            fps=traces.fps,
            n_frames=traces.n_frames,
            motion={r: traces.motion[r + RELOCATED] for r in regions["motion"]},
            rgb={r: traces.rgb[r + RELOCATED] for r in regions["color"]},
        )
        record["relocated"] = {
            "regions": {"boxes": new_boxes, **info},
            "windows": measure(moved, windows, references, protocol, measures),
        }
    record["seconds"] = time.perf_counter() - started
    return record


# ---------------------------------------------------------------------------------------
# Aggregation over videos


def window_rows(records: list[dict]) -> list[dict]:
    """Flatten per-version records ({"video", "subject", "setting", "windows"}) to windows."""
    rows = []
    for r in records:
        for w in r.get("windows", []):
            rows.append(
                {"video": r["video"], "subject": r["subject"], "setting": r["setting"], **w}
            )
    return rows


def _arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "est": np.array([metrics._f(r["est_bpm"]) for r in rows]),
        "ref": np.array([metrics._f(r["ref_bpm"]) for r in rows]),
        "snr": np.array([metrics._f(r["snr_db"]) for r in rows]),
    }


def aggregate(
    records: list[dict],
    settings: list[Setting],
    protocol: dict,
    measures: tuple[str, ...] = MEASURES,
) -> dict:
    """Metrics per setting and measurement, paired differences from the source, breakdown."""
    rows = [r for r in window_rows(records) if r["excluded"] is None]
    excluded = [r for r in window_rows(records) if r["excluded"] is not None]
    stats_cfg = protocol["statistics"]
    tolerance = {
        m: protocol["tolerance_bpm"][cfg["kind"]] for m, cfg in protocol["measurements"].items()
    }
    names = [s.name for s in settings]
    by_key: dict[tuple, dict] = {}
    for r in rows:
        by_key.setdefault((r["setting"], r["measure"]), {})[(r["video"], r["window"])] = r

    per_setting, paired = {}, {}
    for name in names:
        for m in measures:
            cell = by_key.get((name, m))
            if not cell:
                continue
            keys = sorted(cell)
            sel = [cell[k] for k in keys]
            arr = _arrays(sel)
            per_setting.setdefault(name, {})[m] = metrics.summarize(
                arr["est"],
                arr["ref"],
                arr["snr"],
                [r["subject"] for r in sel],
                tolerance[m],
                stats_cfg,
            )
            source = by_key.get(("source", m))
            if name == "source" or not source:
                continue
            common = [k for k in keys if k in source]
            if not common:
                continue
            a = _arrays([source[k] for k in common])
            b = _arrays([cell[k] for k in common])
            paired.setdefault(name, {})[m] = metrics.paired_difference(
                a, b, [cell[k]["subject"] for k in common], tolerance[m], stats_cfg
            )

    breakdown = {}
    for ladder, rung_names in compression.ladder_groups(settings).items():
        for m in measures:
            rungs = [
                {
                    "name": n,
                    "delta": paired[n][m]["success_rate"]["delta"],
                    "ci_high": paired[n][m]["success_rate"]["ci"][1],
                }
                for n in rung_names
                if n in paired and m in paired[n]
            ]
            if rungs:
                breakdown.setdefault(ladder, {})[m] = {
                    "rungs": [r["name"] for r in rungs],
                    "breakdown": metrics.breakdown_point(rungs, protocol["breakdown"]["drop_pp"]),
                }
    exclusions: dict[str, dict[str, int]] = {}
    for r in excluded:
        if r["setting"] == "source":
            cell = exclusions.setdefault(r["measure"], {})
            cell[r["excluded"]] = cell.get(r["excluded"], 0) + 1
    return {
        "per_setting": per_setting,
        "paired_vs_source": paired,
        "breakdown": breakdown,
        "excluded_windows_source": exclusions,
    }


def _fmt(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    return f"{value:.{digits}f}"


def _with_ci(cell: dict, key: str, scale: float = 1.0, digits: int = 2) -> str:
    value, (lo, hi) = cell[key]["value"], cell[key]["ci"]
    if value is None or not np.isfinite(value):
        return "–"
    return f"{_fmt(value * scale, digits)} [{_fmt(lo * scale, digits)}, {_fmt(hi * scale, digits)}]"


def summary_markdown(
    agg: dict,
    records: list[dict],
    settings: list[Setting],
    measures: tuple[str, ...] = MEASURES,
) -> str:
    """Tables of the aggregated metrics; every number comes from `agg` or `records`."""
    kbps: dict[str, list[float]] = {}
    for r in records:
        if r.get("kbps") is not None:
            kbps.setdefault(r["setting"], []).append(r["kbps"])
    lines = [
        "Point value [95% CI]; CIs from a cluster bootstrap over subjects.",
        "",
        "## Per setting",
        "",
        "| Setting | Median kbps | Measure | Windows | Coverage | MAE (bpm) | Success rate (%) "
        "| SNR (dB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in settings:
        cells = agg["per_setting"].get(s.name, {})
        rate_text = _fmt(float(np.median(kbps[s.name])), 0) if s.name in kbps else "–"
        for m in measures:
            if m not in cells:
                continue
            c = cells[m]
            lines.append(
                f"| {s.name} | {rate_text} | {m} | {c['n_windows']} | {_fmt(c['coverage'])} "
                f"| {_with_ci(c, 'mae')} | {_with_ci(c, 'success_rate', 100, 1)} "
                f"| {_with_ci(c, 'snr_db', 1, 1)} |"
            )
    lines += [
        "",
        "## Paired difference from the uncompressed source (setting minus source)",
        "",
        "`*`: 95% CI excludes zero.",
        "",
        "| Setting | Measure | Δ success rate (pp) | Δ MAE (bpm) | Δ SNR (dB) |",
        "|---|---|---|---|---|",
    ]

    def delta(cell: dict, key: str, scale: float = 1.0, digits: int = 2) -> str:
        d, (lo, hi) = cell[key]["delta"], cell[key]["ci"]
        if d is None or not np.isfinite(d):
            return "–"
        star = "*" if cell[key]["significant"] else ""
        return (
            f"{_fmt(d * scale, digits)} [{_fmt(lo * scale, digits)}, "
            f"{_fmt(hi * scale, digits)}]{star}"
        )

    for s in settings:
        for m in measures:
            cell = agg["paired_vs_source"].get(s.name, {}).get(m)
            if cell:
                lines.append(
                    f"| {s.name} | {m} | {delta(cell, 'success_rate', 100, 1)} "
                    f"| {delta(cell, 'mae')} | {delta(cell, 'snr_db', 1, 1)} |"
                )
    lines += ["", "## Breakdown points", "", "| Ladder | Measure | Rungs | Breakdown |"]
    lines.append("|---|---|---|---|")
    for ladder, cells in agg["breakdown"].items():
        for m, cell in cells.items():
            lines.append(
                f"| {ladder} | {m} | {', '.join(cell['rungs'])} | {cell['breakdown'] or 'none'} |"
            )
    lines += ["", "## Windows excluded on the source", ""]
    if agg["excluded_windows_source"]:
        for m, reasons in agg["excluded_windows_source"].items():
            lines.append(f"- {m}: " + ", ".join(f"{k} ({v})" for k, v in reasons.items()))
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"
