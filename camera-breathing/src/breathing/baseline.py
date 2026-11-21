"""Phase-3 baselines on uncompressed videos: references, extra measurements, sensitivity
subsets, reference reliability and the run summary.

The main result follows the evaluation protocol (configs/protocol.yaml) exactly. Two extra
measurements are made on the same signals and reported separately; neither replaces the
main result:

- whole_video: one window per measurement spanning the whole video, only for comparison
  with published results that estimate one rate per video (e.g. rPPG-Toolbox).
- wide_reference (UBFC-rPPG): the breathing reference from PPG under a wider agreement rule
  (sensitivity analysis of GUIDE.md section 4.4).

Sensitivity subsets re-aggregate the same per-window records without some videos.
Everything here is pure: records in, numbers and text out.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

import numpy as np

from breathing import metrics, pipeline, reference
from breathing.compression import SOURCE
from breathing.pipeline import _fmt, _with_ci

MEASURES = pipeline.MEASURES
METHODS = ("bw", "am", "fm")
WHOLE_VIDEO = "whole_video"
WIDE_REFERENCE = "wide_reference"
DATASET_NAMES = {"ubfc": "UBFC-rPPG", "scamps": "SCAMPS"}

Windows = dict[str, list[tuple[float, float]]]


# ---------------------------------------------------------------------------------------
# References and windows


def whole_windows(duration_s: float, protocol: dict) -> Windows:
    """One window per measurement covering the whole video."""
    window = [(0.0, float(duration_s))] if duration_s > 0 else []
    return {m: list(window) for m in protocol["measurements"]}


def ubfc_references(
    ppg: np.ndarray, hr: np.ndarray, fs: float, windows: Windows, protocol: dict
) -> tuple[dict[str, list[float | None]], dict[str, list[dict]]]:
    """Reference rate per measurement and window for UBFC-rPPG, and the details behind it.

    Heart rate: the contact PPG waveform through the protocol's estimator, with its SNR
    around its own peak (a quality indicator) and the oximeter's heart-rate channel as a
    cross-check. Breathing: derived from the PPG (BW, AM, FM); usable only when the three
    agree (reference.respiration_from_ppg). ppg and hr have one sample per video frame.
    Returns (refs, details): refs[m] lists the reference (bpm or None) of each window of m;
    details[m] holds one dict per window.
    """
    rate_cfg, ref_cfg = protocol["rate"], protocol["reference"]
    refs, details, cache = {}, {}, {}
    for m, cfg in protocol["measurements"].items():
        band = tuple(cfg["band_hz"])
        key = (cfg["kind"], band, tuple(windows[m]))
        if key not in cache:
            if cfg["kind"] == "heart":
                rates = reference.waveform_rates(ppg, fs, windows[m], band, rate_cfg)
                oximeter = reference.oximeter_rates(
                    hr, fs, windows[m], ref_cfg["oximeter_min_valid_bpm"]
                )
                cache[key] = [
                    {
                        "bpm": r["bpm"],
                        "ppg_snr_db": r["snr_db"],
                        "oximeter_bpm": o["bpm"],
                        "oximeter_valid_frac": o["valid_frac"],
                    }
                    for r, o in zip(rates, oximeter, strict=True)
                ]
            else:
                cache[key] = reference.respiration_from_ppg(
                    ppg, fs, windows[m], band, rate_cfg, ref_cfg["ppg_respiration"]
                )
        details[m] = cache[key]
        refs[m] = [w["bpm"] for w in details[m]]
    return refs, details


def wide_references(
    refs: dict[str, list[float | None]],
    details: dict[str, list[dict]],
    protocol: dict,
    max_std_bpm: float,
) -> dict[str, list[float | None]]:
    """The same references with breathing fused by the wide rule (reference.fuse_by_std)."""
    out = {}
    for m, cfg in protocol["measurements"].items():
        if cfg["kind"] == "breathing":
            out[m] = [reference.fuse_by_std(w, max_std_bpm) for w in details[m]]
        else:
            out[m] = list(refs[m])
    return out


def value_ranges(waves: dict[str, np.ndarray]) -> dict[str, float]:
    """Peak-to-peak range of each waveform (0 for an empty one)."""
    return {k: float(np.ptp(v)) if len(v) else 0.0 for k, v in waves.items()}


def head_is_still(ranges: dict[str, float], max_range: float) -> bool:
    """True when every head-pose channel stays within max_range over the whole video."""
    return all(v <= max_range for v in ranges.values())


# ---------------------------------------------------------------------------------------
# Aggregation


def variant_records(records: list[dict], name: str) -> list[dict]:
    """Records with `windows` replaced by the windows of the named extra measurement."""
    return [{**r, "windows": r["variants"][name]} for r in records if name in r.get("variants", {})]


def window_rows(records: list[dict]) -> list[dict]:
    """Per-window rows of the main result and of each extra measurement (`variant`)."""
    rows = [{"variant": "main", **row} for row in pipeline.window_rows(records)]
    for name in sorted({n for r in records for n in r.get("variants", {})}):
        variant = pipeline.window_rows(variant_records(records, name))
        rows += [{"variant": name, **row} for row in variant]
    return rows


def source_metrics(records: list[dict], protocol: dict) -> dict:
    """Metrics per measurement on the uncompressed source, with cluster-bootstrap CIs."""
    agg = pipeline.aggregate(records, [SOURCE], protocol)
    return {
        "n_videos": len(records),
        "per_measure": agg["per_setting"].get(SOURCE.name, {}),
        "excluded_windows": agg["excluded_windows_source"],
    }


def sensitivity_subsets(dataset: str, cfg: dict) -> dict[str, Callable[[dict], bool]]:
    """Filters (True = keep the video) for the sensitivity analyses of GUIDE.md 4.4."""
    subsets: dict[str, Callable[[dict], bool]] = {
        "without_face_anchored": lambda r: r["regions"].get("shoulder_source") != "face_anchored",
    }
    if dataset == "ubfc":
        min_fps = cfg["sensitivity"]["min_fps"]
        subsets["without_timestamp_flags"] = lambda r: (
            not (r["flags"]["large_gap"] or r["flags"]["duplicates"])
        )
        subsets["without_low_fps"] = lambda r: r["fps"] >= min_fps
    return subsets


def _quantiles(values) -> list[float] | None:
    """[min, 25%, median, 75%, max] of the values that are not None."""
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if not arr.size:
        return None
    return [float(q) for q in np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 1.0])]


def _fraction(flags: list[bool], clusters: list, stats_cfg: dict | None = None) -> dict:
    """Share of True flags, with a cluster-bootstrap CI when stats_cfg is given."""
    arr = np.asarray(flags, dtype=np.float64)
    out = {
        "count": int(arr.sum()),
        "n": int(arr.size),
        "value": float(arr.mean()) if arr.size else None,
    }
    if stats_cfg is not None:
        ci = metrics.bootstrap(
            lambda idx: {"value": float(arr[idx].mean())},
            metrics.cluster_groups(clusters),
            stats_cfg["bootstrap_samples"],
            stats_cfg["ci"],
            stats_cfg["seed"],
        )
        out["ci"] = list(ci.get("value", (np.nan, np.nan)))
    return out


def _measure_of_kind(protocol: dict, kind: str) -> str:
    return next(m for m, cfg in protocol["measurements"].items() if cfg["kind"] == kind)


def reference_reliability(records: list[dict], protocol: dict, wide_max_std_bpm: float) -> dict:
    """How usable and self-consistent the UBFC-rPPG references are.

    Breathing (windows of the first breathing measurement): how often BW, AM and FM each
    give a rate, the share of usable windows under the protocol rule and under the wide
    rule (cluster-bootstrap CIs over subjects), the spread of the three rates and their
    pairwise differences. Heart: the PPG's self-referenced SNR and its agreement with the
    oximeter's heart-rate channel.
    """
    stats_cfg = protocol["statistics"]
    agreement = protocol["reference"]["ppg_respiration"]["agreement_bpm"]
    b_measure, h_measure = (
        _measure_of_kind(protocol, "breathing"),
        _measure_of_kind(protocol, "heart"),
    )
    rows = [(r["subject"], w) for r in records for w in r["reference"].get(b_measure, [])]
    clusters, windows = [s for s, _ in rows], [w for _, w in rows]

    pairwise = {}
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1 :]:
            diffs = [
                abs(w[f"{a}_bpm"] - w[f"{b}_bpm"])
                for w in windows
                if w[f"{a}_bpm"] is not None and w[f"{b}_bpm"] is not None
            ]
            pairwise[f"{a}-{b}"] = {
                "abs_diff_bpm": _quantiles(diffs),
                "within_agreement": _fraction([d <= agreement for d in diffs], []),
            }
    breathing = {
        "measure": b_measure,
        "n_windows": len(windows),
        "n_subjects": len(set(clusters)),
        "estimate_available": {
            k: _fraction([w[f"{k}_bpm"] is not None for w in windows], []) for k in METHODS
        },
        "usable": _fraction([w["bpm"] is not None for w in windows], clusters, stats_cfg),
        "usable_wide": _fraction(
            [reference.fuse_by_std(w, wide_max_std_bpm) is not None for w in windows],
            clusters,
            stats_cfg,
        ),
        "spread_bpm": _quantiles(w["spread_bpm"] for w in windows),
        "pairwise": pairwise,
        "beats_per_window": _quantiles(w["n_beats"] for w in windows),
    }

    heart_windows = [w for r in records for w in r["reference"].get(h_measure, [])]
    both = [w for w in heart_windows if w["bpm"] is not None and w["oximeter_bpm"] is not None]
    tolerance = protocol["tolerance_bpm"]["heart"]
    diffs = [abs(w["bpm"] - w["oximeter_bpm"]) for w in both]
    heart = {
        "measure": h_measure,
        "n_windows": len(heart_windows),
        "reference_available": _fraction([w["bpm"] is not None for w in heart_windows], []),
        "ppg_snr_db": _quantiles(w["ppg_snr_db"] for w in heart_windows),
        "oximeter_valid_frac": _quantiles(w["oximeter_valid_frac"] for w in heart_windows),
        "oximeter_vs_ppg": {
            "n_windows": len(both),
            "abs_diff_bpm": _quantiles(diffs),
            "within_tolerance": _fraction([d <= tolerance for d in diffs], []),
        },
    }
    return {"breathing": breathing, "heart": heart}


def _natural_key(text: str) -> str:
    return re.sub(r"\d+", lambda m: f"{int(m.group()):08d}", text)


def ubfc_subject_rows(records: list[dict], protocol: dict) -> list[dict]:
    """One row per subject: frame rate, timestamp flags and reference quality."""
    b_measure, h_measure = (
        _measure_of_kind(protocol, "breathing"),
        _measure_of_kind(protocol, "heart"),
    )
    rows = []
    for r in sorted(records, key=lambda r: _natural_key(r["subject"])):
        heart = r["reference"].get(h_measure, [])
        breathing = r["reference"].get(b_measure, [])
        snr = [w["ppg_snr_db"] for w in heart if w["ppg_snr_db"] is not None]
        valid = [w["oximeter_valid_frac"] for w in heart]
        rows.append(
            {
                "subject": r["subject"],
                "fps": r["fps"],
                "n_frames": r["n_frames"],
                "large_gap": r["flags"]["large_gap"],
                "duplicates": r["flags"]["duplicates"],
                "shoulder_source": r["regions"].get("shoulder_source"),
                "ppg_snr_db_median": float(np.median(snr)) if snr else None,
                "oximeter_valid_frac_mean": float(np.mean(valid)) if valid else None,
                "breathing_usable": sum(w["bpm"] is not None for w in breathing),
                "breathing_windows": len(breathing),
            }
        )
    return rows


def scamps_selection(measured: list[dict], skipped: list[dict], cfg: dict) -> dict:
    """How the head-still subset of SCAMPS was drawn from the stream."""

    def largest_range(r: dict) -> float:
        return max(r["head_pose_range"].values())

    return {
        "rule": {
            "head_pose_keys": cfg["selection"]["head_pose_keys"],
            "max_head_pose_range": cfg["selection"]["max_head_pose_range"],
        },
        "streamed": len(measured) + len(skipped),
        "measured": len(measured),
        "skipped_head_motion": len(skipped),
        "head_pose_range_measured": _quantiles(largest_range(r) for r in measured),
        "head_pose_range_skipped": _quantiles(largest_range(r) for r in skipped),
        "action_units_active": sum((r.get("action_units_max_range") or 0.0) > 0 for r in measured),
    }


def summarize(records: list[dict], cfg: dict, protocol: dict) -> dict:
    """Every number of a baseline run, computed from its per-video records."""
    dataset = cfg["dataset"]
    ok = [r for r in records if r.get("status") == "ok"]
    skipped = [r for r in records if r.get("status") == "skipped"]
    errors = [r for r in records if r.get("status") not in ("ok", "skipped")]
    out = {
        "dataset": dataset,
        "counts": {
            "measured": len(ok),
            "subjects": len({r["subject"] for r in ok}),
            "skipped": len(skipped),
            "errors": len(errors),
        },
        "regions": {
            key: dict(Counter(str(r["regions"].get(key)) for r in ok))
            for key in ("shoulder_source", "face_source")
        },
        "main": source_metrics(ok, protocol),
        "variants": {WHOLE_VIDEO: source_metrics(variant_records(ok, WHOLE_VIDEO), protocol)},
        "sensitivity": {},
        "errors": [{"id": r["id"], "error": r.get("error")} for r in errors],
    }
    for name, keep in sensitivity_subsets(dataset, cfg).items():
        kept = [r for r in ok if keep(r)]
        out["sensitivity"][name] = {
            "n_removed": len(ok) - len(kept),
            **source_metrics(kept, protocol),
        }
    if dataset == "ubfc":
        wide = cfg["sensitivity"]["wide_reference_max_std_bpm"]
        out["variants"][WIDE_REFERENCE] = source_metrics(
            variant_records(ok, WIDE_REFERENCE), protocol
        )
        out["reference_reliability"] = reference_reliability(ok, protocol, wide)
        out["subjects"] = ubfc_subject_rows(ok, protocol)
    if dataset == "scamps":
        out["selection"] = scamps_selection(ok, skipped, cfg)
    return out


# ---------------------------------------------------------------------------------------
# Markdown


def _metrics_table(per_measure: dict) -> list[str]:
    lines = [
        "| Measure | Subjects | Windows | Coverage | MAE (bpm) | RMSE (bpm) | MAPE (%) "
        "| Pearson r | Success rate (%) | SNR (dB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in MEASURES:
        c = per_measure.get(m)
        if c is None:
            lines.append(f"| {m} | – | 0 | – | – | – | – | – | – | – |")
            continue
        lines.append(
            f"| {m} | {c['n_subjects']} | {c['n_windows']} | {_fmt(c['coverage'])} "
            f"| {_with_ci(c, 'mae')} | {_with_ci(c, 'rmse')} | {_with_ci(c, 'mape', 1, 1)} "
            f"| {_with_ci(c, 'pearson_r')} | {_with_ci(c, 'success_rate', 100, 1)} "
            f"| {_with_ci(c, 'snr_db', 1, 1)} |"
        )
    return lines


def _q(values: list[float] | None, digits: int = 2) -> str:
    return " / ".join(_fmt(v, digits) for v in values) if values else "–"


def _frac(cell: dict, ci: bool = False) -> str:
    if cell["value"] is None:
        return "–"
    text = f"{cell['count']} of {cell['n']} ({_fmt(100 * cell['value'], 1)}%)"
    if ci and "ci" in cell:
        lo, hi = cell["ci"]
        text += f" [{_fmt(100 * lo, 1)}, {_fmt(100 * hi, 1)}]"
    return text


def _reliability_lines(rel: dict, protocol: dict, wide_max_std: float) -> list[str]:
    b, h = rel["breathing"], rel["heart"]
    agreement = protocol["reference"]["ppg_respiration"]["agreement_bpm"]
    tolerance = protocol["tolerance_bpm"]["heart"]
    lines = [
        "## Breathing reference derived from PPG",
        "",
        f"Windows of {b['measure']}: {b['n_windows']} from {b['n_subjects']} subjects. "
        "Quantiles are min / 25% / median / 75% / max; CIs from a cluster bootstrap over "
        "subjects.",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Usable (BW, AM, FM all present, range ≤ {_fmt(agreement, 1)} bpm; protocol) "
        f"| {_frac(b['usable'], ci=True)} |",
        f"| Usable under the wide rule (sample SD ≤ {_fmt(wide_max_std, 1)} bpm) "
        f"| {_frac(b['usable_wide'], ci=True)} |",
    ]
    for k in METHODS:
        lines.append(f"| {k.upper()} gave a rate | {_frac(b['estimate_available'][k])} |")
    lines.append(f"| Range of BW, AM, FM (bpm) | {_q(b['spread_bpm'])} |")
    for pair, cell in b["pairwise"].items():
        name = pair.upper().replace("-", " vs ")
        lines.append(f"| {name}: abs difference (bpm) | {_q(cell['abs_diff_bpm'])} |")
        lines.append(
            f"| {name}: within {_fmt(agreement, 1)} bpm | {_frac(cell['within_agreement'])} |"
        )
    lines.append(f"| Beats per window | {_q(b['beats_per_window'], 0)} |")
    o = h["oximeter_vs_ppg"]
    lines += [
        "",
        "## Heart-rate reference",
        "",
        f"Windows of {h['measure']}: {h['n_windows']}. The reference is the contact PPG "
        "waveform; the oximeter's heart-rate channel is only a cross-check.",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| PPG gave a rate | {_frac(h['reference_available'])} |",
        f"| PPG SNR around its own peak (dB) | {_q(h['ppg_snr_db'], 1)} |",
        f"| Oximeter valid fraction per window | {_q(h['oximeter_valid_frac'])} |",
        f"| Windows with both rates | {o['n_windows']} |",
        f"| Oximeter vs PPG: abs difference (bpm) | {_q(o['abs_diff_bpm'])} |",
        f"| Oximeter vs PPG: within {_fmt(tolerance, 1)} bpm | {_frac(o['within_tolerance'])} |",
    ]
    return lines


def _subject_lines(rows: list[dict]) -> list[str]:
    lines = [
        "## Per subject",
        "",
        "| Subject | fps | Frames | Timestamp gap | Duplicate timestamps | Shoulder band "
        "| PPG SNR median (dB) | Oximeter valid (mean) | Usable breathing windows |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['subject']} | {_fmt(r['fps'])} | {r['n_frames']} | {r['large_gap']} "
            f"| {r['duplicates']} | {r['shoulder_source']} | {_fmt(r['ppg_snr_db_median'], 1)} "
            f"| {_fmt(r['oximeter_valid_frac_mean'])} "
            f"| {r['breathing_usable']} of {r['breathing_windows']} |"
        )
    return lines


SENSITIVITY_LABELS = {
    "without_face_anchored": "Without face-anchored shoulder bands",
    "without_timestamp_flags": "Without subjects with flagged timestamps",
    "without_low_fps": "Without low-frame-rate subjects",
}


def markdown(summary: dict, cfg: dict, protocol: dict, run_id: str) -> str:
    """The run summary; every number comes from `summary`."""
    dataset, counts = summary["dataset"], summary["counts"]
    tol = protocol["tolerance_bpm"]
    windows = {m: protocol["measurements"][m]["window_s"] for m in MEASURES}
    lines = [
        f"# Uncompressed baseline: {run_id}",
        "",
        f"Dataset: {DATASET_NAMES.get(dataset, dataset)}. Videos measured: {counts['measured']} "
        f"(subjects: {counts['subjects']}); errors: {counts['errors']}.",
    ]
    if dataset == "scamps":
        sel = summary["selection"]
        lines.append(
            f"Videos streamed: {sel['streamed']}; skipped for head motion: "
            f"{sel['skipped_head_motion']}."
        )
    lines += [
        "",
        "Evaluation protocol of configs/protocol.yaml on the uncompressed source: windows of "
        + ", ".join(f"{m} {_fmt(w, 0)} s" for m, w in windows.items())
        + f"; success means an error within {_fmt(tol['breathing'], 0)} bpm (breathing) or "
        f"{_fmt(tol['heart'], 0)} bpm (heart rate). Point value [95% CI]; CIs from a cluster "
        "bootstrap over subjects.",
        "",
        "## Main result",
        "",
        *_metrics_table(summary["main"]["per_measure"]),
        "",
        "## Windows excluded",
        "",
    ]
    excluded = summary["main"]["excluded_windows"]
    if excluded:
        for m, reasons in excluded.items():
            lines.append(f"- {m}: " + ", ".join(f"{k} ({v})" for k, v in reasons.items()))
    else:
        lines.append("None.")
    lines += ["", "## Regions", ""]
    for key, label in (("shoulder_source", "Shoulder band"), ("face_source", "Face box")):
        found = ", ".join(f"{k} ({v})" for k, v in summary["regions"][key].items())
        lines.append(f"- {label} from: {found or '–'}")
    if dataset == "scamps":
        sel = summary["selection"]
        lines += [
            "",
            "## Head-still subset",
            "",
            f"A video is measured when each of {', '.join(sel['rule']['head_pose_keys'])} "
            f"varies by at most {sel['rule']['max_head_pose_range']:g} over the video.",
            "",
            "| Quantity | Value |",
            "|---|---|",
            f"| Videos streamed | {sel['streamed']} |",
            f"| Measured | {sel['measured']} |",
            f"| Skipped for head motion | {sel['skipped_head_motion']} |",
            f"| Largest head-pose range, measured (min / 25% / median / 75% / max) "
            f"| {_q(sel['head_pose_range_measured'], 4)} |",
            f"| Largest head-pose range, skipped | {_q(sel['head_pose_range_skipped'], 4)} |",
            f"| Measured videos with facial action units | {sel['action_units_active']} |",
        ]
    if dataset == "ubfc":
        wide = cfg["sensitivity"]["wide_reference_max_std_bpm"]
        lines += ["", *_reliability_lines(summary["reference_reliability"], protocol, wide)]

    lines += [
        "",
        "## Sensitivity analyses",
        "",
        "The same windows without some videos (GUIDE.md section 4.4).",
        "",
        "| Analysis | Videos removed | Measure | Windows | MAE (bpm) | Success rate (%) |",
        "|---|---|---|---|---|---|",
    ]
    for name, cell in summary["sensitivity"].items():
        label = SENSITIVITY_LABELS.get(name, name)
        if name == "without_low_fps":
            label += f" (below {_fmt(cfg['sensitivity']['min_fps'], 0)} fps)"
        for m in MEASURES:
            c = cell["per_measure"].get(m)
            if c is None:
                continue
            lines.append(
                f"| {label} | {cell['n_removed']} | {m} | {c['n_windows']} "
                f"| {_with_ci(c, 'mae')} | {_with_ci(c, 'success_rate', 100, 1)} |"
            )
    if WIDE_REFERENCE in summary["variants"]:
        lines += [
            "",
            "## Sensitivity: wide breathing-reference rule",
            "",
            "Breathing windows use the mean of BW, AM and FM when their sample SD is at most "
            f"{_fmt(cfg['sensitivity']['wide_reference_max_std_bpm'], 1)} bpm (after Karlen et "
            "al. 2013); heart-rate windows are unchanged.",
            "",
            *_metrics_table(summary["variants"][WIDE_REFERENCE]["per_measure"]),
        ]
    lines += [
        "",
        "## For comparison with published results: one window per whole video",
        "",
        "Not the protocol: every measurement uses a single window spanning the whole video "
        "(the reference too), as in evaluations that report one rate per video.",
        "",
        *_metrics_table(summary["variants"][WHOLE_VIDEO]["per_measure"]),
    ]
    if dataset == "ubfc":
        lines += ["", *_subject_lines(summary["subjects"])]
    if summary["errors"]:
        lines += ["", "## Errors", ""]
        lines += [f"- `{e['id']}`: {e['error']}" for e in summary["errors"]]
    return "\n".join(lines) + "\n"
