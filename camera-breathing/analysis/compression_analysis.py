"""Phase-4 analysis of the compression ladders and the bitrate-allocation grid.

Reads the four phase-4 runs (items.jsonl and metrics.json; no data needed) and writes into an
output directory (default results/20261125_phase4_analysis/):

  analysis.json   every number used in notes/04-compression.md
  tables.md       the same numbers as tables
  figures/*.png   the figures
  manifest.json   input runs, git commit and package versions

Beyond the per-run metrics.json (accuracy against the reference, paired with the
uncompressed source, breakdown points), it computes:

- agreement with the uncompressed estimate: the share of windows whose estimate stays within
  the tolerance of the same window's estimate on the uncompressed source. It needs no
  reference, so it also covers the UBFC-rPPG breathing windows whose PPG-derived reference is
  unusable or doubtful (notes/03-baseline.md section 5);
- the compression effect in the allocation grid: each compressed setting minus the
  uncompressed version with the same frame decimation and downscaling (raw_d*_s*);
- checks of the cases where a compressed version scored better than the source.

  python analysis/compression_analysis.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from breathing import compression, metrics, pipeline
from breathing.runinfo import environment_snapshot, to_jsonable

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_DIR / "results"
RUNS = {
    ("scamps", "ladders"): "20261123_scamps_ladders_s0",
    ("ubfc", "ladders"): "20261123_ubfc_ladders_s0",
    ("scamps", "allocation"): "20261123_scamps_allocation_s0",
    ("ubfc", "allocation"): "20261123_ubfc_allocation_s0",
}
OUT_ID = "20261125_phase4_analysis"
MEASURES = pipeline.MEASURES
LADDERS = ("x264_crf", "x265_crf", "x264_bitrate", "x265_bitrate")


def _experiment_module():
    path = PROJECT_DIR / "scripts" / "compression_experiment.py"
    spec = importlib.util.spec_from_file_location("compression_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_run(run_dir: Path) -> tuple[dict, list[dict], dict]:
    """(effective config, ok setting records with windows as dicts, metrics.json)."""
    import yaml

    experiment = _experiment_module()
    config = yaml.safe_load((run_dir / "config.yaml").read_text())["experiment"]
    latest: dict[str, dict] = {}
    for line in (run_dir / "items.jsonl").read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["id"]] = record
    records = [
        experiment.unpack(r) for r in latest.values() if r["status"] == "ok" and "setting" in r
    ]
    return config, records, json.loads((run_dir / "metrics.json").read_text())


def tolerances(protocol: dict) -> dict[str, float]:
    return {m: protocol["tolerance_bpm"][c["kind"]] for m, c in protocol["measurements"].items()}


def median_kbps(records: list[dict]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for r in records:
        if r.get("kbps") is not None:
            values.setdefault(r["setting"], []).append(r["kbps"])
    return {k: float(np.median(v)) for k, v in values.items()}


def _fraction_ci(flags: list[bool], clusters: list, stats: dict) -> dict:
    arr = np.asarray(flags, dtype=np.float64)
    if not arr.size:
        return {"value": None, "ci": [None, None], "n": 0}
    ci = metrics.bootstrap(
        lambda idx: {"v": float(arr[idx].mean())},
        metrics.cluster_groups(clusters),
        stats["bootstrap_samples"],
        stats["ci"],
        stats["seed"],
    )["v"]
    return {"value": float(arr.mean()), "ci": [float(ci[0]), float(ci[1])], "n": int(arr.size)}


def consistency(records: list[dict], protocol: dict) -> dict:
    """Per setting and measure: share of windows whose estimate stays within the tolerance of
    the uncompressed estimate of the same window (a missing estimate does not agree).

    Uses every window that has a region and an estimate on the source, whether or not it has
    a usable reference; CIs from the cluster bootstrap over subjects.
    """
    tol = tolerances(protocol)
    source = {}
    for r in records:
        if r["setting"] == compression.SOURCE.name:
            for w in r["windows"]:
                if w["excluded"] != "no_region" and w["est_bpm"] is not None:
                    source[(r["video"], w["measure"], w["window"])] = w["est_bpm"]
    cells: dict[tuple, tuple[list, list]] = {}
    for r in records:
        if r["setting"] == compression.SOURCE.name:
            continue
        for w in r["windows"]:
            ref = source.get((r["video"], w["measure"], w["window"]))
            if ref is None:
                continue
            flags, clusters = cells.setdefault((r["setting"], w["measure"]), ([], []))
            flags.append(w["est_bpm"] is not None and abs(w["est_bpm"] - ref) <= tol[w["measure"]])
            clusters.append(r["subject"])
    out: dict = {}
    for (setting, m), (flags, clusters) in cells.items():
        out.setdefault(setting, {})[m] = _fraction_ci(flags, clusters, protocol["statistics"])
    return out


def paired(records: list[dict], base: str, other: str, protocol: dict) -> dict:
    """Metrics of `other` minus `base` on the windows both include, per measure."""
    tol = tolerances(protocol)
    by: dict[tuple, dict] = {}
    for r in records:
        if r["setting"] in (base, other):
            for w in r["windows"]:
                if w["excluded"] is None:
                    by.setdefault((r["video"], w["measure"], w["window"]), {})[r["setting"]] = (
                        r["subject"],
                        w,
                    )
    out = {}
    for m in MEASURES:
        rows = [v for k, v in sorted(by.items()) if k[1] == m and base in v and other in v]
        if not rows:
            continue
        a = pipeline._arrays([v[base][1] for v in rows])
        b = pipeline._arrays([v[other][1] for v in rows])
        out[m] = metrics.paired_difference(
            a, b, [v[base][0] for v in rows], tol[m], protocol["statistics"]
        )
    return out


def transitions(records: list[dict], setting: str, protocol: dict) -> dict:
    """Per measure: windows that failed on the source and succeeded on `setting`, and back."""
    tol = tolerances(protocol)
    by: dict[tuple, dict] = {}
    for r in records:
        if r["setting"] in (compression.SOURCE.name, setting):
            for w in r["windows"]:
                if w["excluded"] is None:
                    by.setdefault((r["video"], w["measure"], w["window"]), {})[r["setting"]] = w

    def ok(w):
        return w["est_bpm"] is not None and abs(w["error_bpm"]) <= tol[w["measure"]]

    out = {}
    for m in MEASURES:
        pairs = [v for k, v in by.items() if k[1] == m and len(v) == 2]
        src, other = compression.SOURCE.name, setting
        out[m] = {
            "windows": len(pairs),
            "fail_to_success": sum(not ok(v[src]) and ok(v[other]) for v in pairs),
            "success_to_fail": sum(ok(v[src]) and not ok(v[other]) for v in pairs),
        }
    return out


def m2_follows_m1(records: list[dict], protocol: dict) -> dict:
    """Per setting: share of breathing windows where the color estimate (M2) lies within the
    breathing tolerance of the motion estimate (M1), among windows that have both."""
    tol = protocol["tolerance_bpm"]["breathing"]
    out: dict = {}
    for r in records:
        est = {(w["measure"], w["window"]): w["est_bpm"] for w in r["windows"]}
        for (m, k), m1 in est.items():
            m2 = est.get(("M2", k))
            if m != "M1" or m1 is None or m2 is None:
                continue
            flags, clusters = out.setdefault(r["setting"], ([], []))
            flags.append(abs(m2 - m1) <= tol)
            clusters.append(r["subject"])
    return {s: _fraction_ci(f, c, protocol["statistics"]) for s, (f, c) in out.items()}


def ladder_rows(metric_json: dict, kbps: dict, settings: list) -> dict:
    """Per ladder and measure: the rungs with measured bitrate, accuracy and paired deltas."""
    groups = compression.ladder_groups(settings)
    out: dict = {}
    for ladder in LADDERS:
        for m in MEASURES:
            rows = []
            for name in groups.get(ladder, []):
                cell = metric_json["per_setting"].get(name, {}).get(m)
                delta = metric_json["paired_vs_source"].get(name, {}).get(m)
                if cell is None or delta is None:
                    continue
                rows.append(
                    {"setting": name, "kbps": kbps.get(name), "metrics": cell, "delta": delta}
                )
            source = metric_json["per_setting"].get(compression.SOURCE.name, {}).get(m)
            breakdown = metric_json["breakdown"].get(ladder, {}).get(m, {}).get("breakdown")
            out.setdefault(ladder, {})[m] = {"rows": rows, "source": source, "breakdown": breakdown}
    return out


def allocation_grid(metric_json: dict, settings: list, key: str = "success_rate") -> dict:
    """Success rate (or another metric) of the allocation grid and its uncompressed twin:
    grid[codec or "raw"][kbps][measure] -> {(decimate, scale): value}."""
    out: dict = {}
    for s in settings:
        if s.ladder not in ("allocation", "raw") and s.name != compression.SOURCE.name:
            continue
        cells = metric_json["per_setting"].get(s.name, {})
        if s.name == compression.SOURCE.name:
            group, kbps = "raw", 0
        elif s.ladder == "raw":
            group, kbps = "raw", 0
        else:
            group, kbps = compression.SHORT_CODEC[s.codec], int(s.kbps)
        for m, cell in cells.items():
            value = cell[key]["value"] if isinstance(cell.get(key), dict) else None
            ci = cell[key]["ci"] if isinstance(cell.get(key), dict) else None
            out.setdefault(group, {}).setdefault(kbps, {}).setdefault(m, {})[
                f"d{s.decimate}_s{s.scale}"
            ] = {"value": value, "ci": ci}
    return out


def best_cells(grid: dict) -> dict:
    """For each codec, bitrate and measure: the decimation/scale with the highest value."""
    out: dict = {}
    for codec, by_kbps in grid.items():
        if codec == "raw":
            continue
        for kbps, by_m in by_kbps.items():
            for m, cells in by_m.items():
                valid = {k: v for k, v in cells.items() if v["value"] is not None}
                if valid:
                    best = max(valid, key=lambda k: valid[k]["value"])
                    out.setdefault(codec, {}).setdefault(kbps, {})[m] = {
                        "cell": best,
                        **valid[best],
                        "full_rate": cells.get("d1_s1"),
                    }
    return out


CONTRASTS = {
    "half_resolution": "d1_s2",  # keeps the frame rate (WebRTC maintain-framerate)
    "half_frame_rate": "d2_s1",  # keeps the resolution (WebRTC maintain-resolution)
    "both_halved": "d2_s2",
}


def allocation_contrasts(records: list[dict], protocol: dict) -> dict:
    """At each codec and bitrate: the pre-chosen reductions minus full frame rate and
    resolution (d1_s1), paired on the same windows. Fixed in advance, so free of the
    selection bias of picking the best of the twelve cells."""
    out: dict = {}
    for codec in ("x264", "x265"):
        for kbps in (500, 250, 125):
            full = f"{codec}_br{kbps}_d1_s1"
            for name, cell in CONTRASTS.items():
                out.setdefault(codec, {}).setdefault(str(kbps), {})[name] = paired(
                    records, full, f"{codec}_br{kbps}_{cell}", protocol
                )
    return out


def bits_per_pixel(records: list[dict], con: dict) -> list[dict]:
    """Per compressed setting: median bits per pixel (measured bitrate over pixel rate) next to
    the agreement with the uncompressed estimate."""
    values: dict[str, list[float]] = {}
    for r in records:
        if r.get("kbps") is not None and not r["setting"].startswith("raw"):
            values.setdefault(r["setting"], []).append(
                r["kbps"] * 1000 / (r["fps"] * r["width"] * r["height"])
            )
    return [
        {
            "setting": s,
            "bpp": float(np.median(v)),
            "agreement": {m: c["value"] for m, c in con[s].items()},
        }
        for s, v in values.items()
        if s in con
    ]


def spearman(rows: list[dict]) -> dict:
    from scipy.stats import spearmanr

    out = {}
    for m in MEASURES:
        pairs = [(r["bpp"], r["agreement"][m]) for r in rows if r["agreement"].get(m) is not None]
        if len(pairs) > 2:
            x, y = zip(*pairs, strict=True)
            out[m] = {"rho": float(spearmanr(x, y).statistic), "settings": len(pairs)}
    return out


def analyse(protocol: dict) -> dict:
    experiment = _experiment_module()
    result: dict = {"runs": {f"{d}_{e}": run for (d, e), run in RUNS.items()}}
    for (dataset, kind), run_id in RUNS.items():
        config, records, metric_json = load_run(RESULTS / run_id)
        settings = experiment.experiment_settings(config, protocol)
        kbps = median_kbps(records)
        entry: dict = {
            "videos": len({r["video"] for r in records}),
            "median_kbps": kbps,
            "consistency": consistency(records, protocol),
        }
        if kind == "ladders":
            entry["ladders"] = ladder_rows(metric_json, kbps, settings)
            entry["m2_follows_m1"] = m2_follows_m1(records, protocol)
            entry["relocated_minus_fixed"] = metric_json["relocated_minus_fixed"]
            entry["sensitivity_breakdown"] = metric_json["sensitivity_breakdown"]
        else:
            grid = allocation_grid(metric_json, settings)
            entry["grid"] = grid
            entry["best"] = best_cells(grid)
            entry["contrasts"] = allocation_contrasts(records, protocol)
            entry["compression_effect"] = {
                s.name: paired(
                    records,
                    f"raw_d{s.decimate}_s{s.scale}"
                    if (s.decimate, s.scale) != (1, 1)
                    else "source",
                    s.name,
                    protocol,
                )
                for s in settings
                if s.ladder == "allocation"
            }
        entry["better_than_source"] = [
            {"setting": s, "measure": m, "metric": key, **cell[key]}
            for s, cells in metric_json["paired_vs_source"].items()
            for m, cell in cells.items()
            for key, sign in (("success_rate", 1), ("mae", -1))
            if cell[key]["significant"] and cell[key]["delta"] * sign > 0
        ]
        entry["better_than_source_transitions"] = {
            f["setting"]: transitions(records, f["setting"], protocol)
            for f in entry["better_than_source"]
        }
        entry["bits_per_pixel"] = bits_per_pixel(records, entry["consistency"])
        result[f"{dataset}_{kind}"] = entry
    for dataset in ("scamps", "ubfc"):
        rows = result[f"{dataset}_ladders"]["bits_per_pixel"]
        rows += result[f"{dataset}_allocation"]["bits_per_pixel"]
        result[f"{dataset}_bits_per_pixel_spearman"] = spearman(rows)
    return to_jsonable(result)


# ---------------------------------------------------------------------------------------
# Figures (matplotlib, imported only here) and tables

COLORS = {"M1": "#257de0", "M2": "#f47230", "M3": "#00b894"}  # categorical slots 1-3
MARKERS = {"M1": "o", "M2": "s", "M3": "^"}
NAMES = {"M1": "M1 breathing (motion)", "M2": "M2 breathing (color)", "M3": "M3 heart rate (color)"}
TEXT, TEXT_2, GRID = "#191919", "#666666", "#e5e5e5"
BLUES = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]
LADDER_TITLES = {
    "x264_crf": "H.264, quality (CRF 0-43)",
    "x265_crf": "H.265, quality (CRF 0-43)",
    "x264_bitrate": "H.264, video call (2000-125 kbps)",
    "x265_bitrate": "H.265, video call (2000-125 kbps)",
}


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbdbd")
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.set_xscale("log")


def _kbps_axis(ax, ladder: str) -> None:
    """Plain kbps tick labels on the log axis (no 2x10^3-style minor labels)."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    ticks = (
        [125, 250, 500, 1000, 2000]
        if "bitrate" in ladder
        else [10, 30, 100, 300, 1000, 3000, 10000, 30000]
    )
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([str(t) for t in ticks])


def _series(ax, x, y, lo, hi, m, label_end=True):
    order = np.argsort(x)
    x, y, lo, hi = (np.asarray(v, dtype=float)[order] for v in (x, y, lo, hi))
    ax.fill_between(x, lo, hi, color=COLORS[m], alpha=0.15, linewidth=0)
    ax.plot(
        x,
        y,
        color=COLORS[m],
        linewidth=1.6,
        marker=MARKERS[m],
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=NAMES[m],
    )
    if label_end:
        ax.annotate(
            m,
            (x[-1], y[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=TEXT,
        )


def plot_ladders(entry: dict, measures: tuple, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = (
        ("Success rate (%)", "success_rate", 100, False),
        ("MAE (bpm)", "mae", 1, False),
        ("ΔSNR vs uncompressed (dB)", "snr_db", 1, True),
    )
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.2), sharex="col")
    for col, ladder in enumerate(LADDERS):
        for row, (label, key, scale, is_delta) in enumerate(rows):
            ax = axes[row, col]
            _style(ax)
            for m in measures:
                cell = entry["ladders"][ladder][m]
                pts = [r for r in cell["rows"] if r["kbps"] is not None]
                x = [r["kbps"] for r in pts]
                if is_delta:
                    vals = [r["delta"][key] for r in pts]
                    y = [v["delta"] * scale for v in vals]
                else:
                    vals = [r["metrics"][key] for r in pts]
                    y = [v["value"] * scale for v in vals]
                lo = [v["ci"][0] * scale for v in vals]
                hi = [v["ci"][1] * scale for v in vals]
                _series(ax, x, y, lo, hi, m, label_end=row == 0)
                if not is_delta and cell["source"] is not None:
                    ax.axhline(
                        cell["source"][key]["value"] * scale,
                        color=COLORS[m],
                        linestyle=":",
                        linewidth=1.2,
                    )
            if is_delta:
                ax.axhline(0, color=TEXT_2, linewidth=0.8)
            _kbps_axis(ax, ladder)
            if row == 0:
                ax.set_title(LADDER_TITLES[ladder], fontsize=9.5, color=TEXT)
                ax.set_ylim(-3, 103)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, color=TEXT)
            if row == 2:
                ax.set_xlabel("Measured bitrate (kbps, log scale)", fontsize=8.5, color=TEXT_2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles.append(Line2D([], [], color=TEXT_2, linestyle=":", linewidth=1.2))
    labels.append("uncompressed (dotted, same color)")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.suptitle(title, fontsize=11, color=TEXT, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Shaded: 95% CI (cluster bootstrap over subjects).",
        ha="center",
        fontsize=8,
        color=TEXT_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_agreement(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.8), sharex="col", sharey=True)
    for row, dataset in enumerate(("scamps", "ubfc")):
        entry = result[f"{dataset}_ladders"]
        groups = {
            lad: [r["setting"] for r in entry["ladders"][lad]["M1"]["rows"]] for lad in LADDERS
        }
        for col, ladder in enumerate(LADDERS):
            ax = axes[row, col]
            _style(ax)
            for m in MEASURES:
                pts = [s for s in groups[ladder] if m in entry["consistency"].get(s, {})]
                x = [entry["median_kbps"][s] for s in pts]
                cells = [entry["consistency"][s][m] for s in pts]
                _series(
                    ax,
                    x,
                    [100 * c["value"] for c in cells],
                    [100 * c["ci"][0] for c in cells],
                    [100 * c["ci"][1] for c in cells],
                    m,
                    label_end=False,
                )
            ax.set_ylim(-3, 103)
            _kbps_axis(ax, ladder)
            if row == 0:
                ax.set_title(LADDER_TITLES[ladder], fontsize=9.5, color=TEXT)
            if col == 0:
                name = {"scamps": "SCAMPS", "ubfc": "UBFC-rPPG"}[dataset]
                ax.set_ylabel(f"{name}\nagreement with\nuncompressed (%)", fontsize=9, color=TEXT)
            if row == 1:
                ax.set_xlabel("Measured bitrate (kbps, log scale)", fontsize=8.5, color=TEXT_2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.suptitle(
        "Share of windows whose estimate stays within tolerance of the uncompressed "
        "estimate (no reference needed)",
        fontsize=11,
        color=TEXT,
        y=1.03,
    )
    fig.text(
        0.5,
        -0.01,
        "Tolerance: 2 bpm (breathing), 5 bpm (heart rate). Shaded: 95% CI.",
        ha="center",
        fontsize=8,
        color=TEXT_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_allocation(entry: dict, rows: list[tuple], path: Path, title: str) -> None:
    """rows: (measure, "success" or "agreement") per row of heatmaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    columns = [("raw", 0, "No compression")] + [
        (codec, kbps, f"{'H.264' if codec == 'x264' else 'H.265'} {kbps} kbps")
        for codec in ("x264", "x265")
        for kbps in (500, 250, 125)
    ]
    decimate, scale = (1, 2, 3, 6), (1, 2, 4)
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(15, 2.6 * len(rows) + 0.8))
    for i, (m, kind) in enumerate(rows):
        for j, (codec, kbps, label) in enumerate(columns):
            ax = axes[i, j]
            grid = np.full((len(decimate), len(scale)), np.nan)
            for a, d in enumerate(decimate):
                for b, s in enumerate(scale):
                    if kind == "success":
                        cell = (
                            entry["grid"]
                            .get(codec, {})
                            .get(str(kbps), {})
                            .get(m, {})
                            .get(f"d{d}_s{s}")
                        )
                        value = None if cell is None else cell["value"]
                    else:
                        name = f"raw_d{d}_s{s}" if codec == "raw" else f"{codec}_br{kbps}_d{d}_s{s}"
                        if codec == "raw" and (d, s) == (1, 1):
                            value = 1.0
                        else:
                            value = entry["consistency"].get(name, {}).get(m, {}).get("value")
                    if value is not None:
                        grid[a, b] = 100 * value
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")
            for a in range(len(decimate)):
                for b in range(len(scale)):
                    if np.isfinite(grid[a, b]):
                        dark = grid[a, b] > 55
                        ax.text(
                            b,
                            a,
                            f"{grid[a, b]:.0f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if dark else TEXT,
                        )
            ax.set_xticks(range(3), ["full", "1/2", "1/4"], fontsize=7.5, color=TEXT_2)
            ax.set_yticks(range(4), ["30", "15", "10", "5"], fontsize=7.5, color=TEXT_2)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(label, fontsize=9, color=TEXT)
            if j == 0:
                what = "success rate" if kind == "success" else "agreement w/ uncompressed"
                ax.set_ylabel(f"{NAMES[m]}\n{what} (%)\nframe rate (fps)", fontsize=8, color=TEXT)
            if i == len(rows) - 1:
                ax.set_xlabel("resolution", fontsize=8, color=TEXT_2)
    fig.suptitle(title, fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pct(cell: dict | None, key: str = "value") -> str:
    if cell is None or cell.get(key) is None:
        return "–"
    lo, hi = cell["ci"]
    if lo is None:
        return f"{100 * cell[key]:.1f}"
    return f"{100 * cell[key]:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def _num(cell: dict | None, key: str = "value", digits: int = 2) -> str:
    if cell is None or cell.get(key) is None:
        return "–"
    lo, hi = cell["ci"]
    star = "*" if cell.get("significant") else ""
    return f"{cell[key]:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]{star}"


def tables(result: dict) -> str:
    names = {"scamps": "SCAMPS", "ubfc": "UBFC-rPPG"}
    lines = [
        "# Phase-4 analysis tables",
        "",
        "Computed by analysis/compression_analysis.py from the runs "
        + ", ".join(f"`{r}`" for r in RUNS.values())
        + ". Point value [95% CI], cluster bootstrap over subjects; `*`: CI excludes zero. "
        "Success: error within 2 bpm (breathing) or 5 bpm (heart rate).",
        "",
        "## Breakdown points",
        "",
        "| Dataset | Ladder | Measure | Breakdown rung | Median kbps at that rung |",
        "|---|---|---|---|---|",
    ]
    for ds in ("scamps", "ubfc"):
        e = result[f"{ds}_ladders"]
        for ladder in LADDERS:
            for m in MEASURES:
                rung = e["ladders"][ladder][m]["breakdown"]
                kbps = e["median_kbps"].get(rung) if rung else None
                note = " (12 windows with a usable reference)" if ds == "ubfc" and m != "M3" else ""
                lines.append(
                    f"| {names[ds]} | {ladder} | {m}{note} | {rung or 'none'} "
                    f"| {'–' if kbps is None else f'{kbps:.0f}'} |"
                )
    for ds, measures in (("scamps", MEASURES), ("ubfc", ("M3",))):
        e = result[f"{ds}_ladders"]
        lines += [
            "",
            f"## Ladders, {names[ds]}: accuracy against the reference",
            "",
            "| Ladder | Setting | Median kbps | Measure | Success rate (%) | Δ success vs "
            "uncompressed (pp) | MAE (bpm) | ΔSNR (dB) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for ladder in LADDERS:
            for m in measures:
                cell = e["ladders"][ladder][m]
                src = cell["source"]
                lines.append(
                    f"| {ladder} | source | – | {m} | {_pct(src['success_rate'])} | – "
                    f"| {_num(src['mae'])} | – |"
                )
                for r in cell["rows"]:
                    d = r["delta"]
                    dsr = {
                        **d["success_rate"],
                        "delta": 100 * d["success_rate"]["delta"],
                        "ci": [100 * v for v in d["success_rate"]["ci"]],
                    }
                    lines.append(
                        f"| {ladder} | {r['setting']} | {r['kbps']:.0f} | {m} "
                        f"| {_pct(r['metrics']['success_rate'])} | {_num(dsr, 'delta', 1)} "
                        f"| {_num(r['metrics']['mae'])} | {_num(d['snr_db'], 'delta', 1)} |"
                    )
    lines += [
        "",
        "## Ladders: agreement with the uncompressed estimate (%)",
        "",
        "Share of windows whose estimate stays within the tolerance of the same window's "
        "estimate on the uncompressed source; every window with a region counts, with or "
        "without a usable reference.",
        "",
        "| Setting | SCAMPS kbps | SCAMPS M1 | SCAMPS M2 | SCAMPS M3 | UBFC kbps | UBFC M1 "
        "| UBFC M2 | UBFC M3 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    s_e, u_e = result["scamps_ladders"], result["ubfc_ladders"]
    for ladder in LADDERS:
        for r in s_e["ladders"][ladder]["M1"]["rows"]:
            s = r["setting"]
            sc, uc = s_e["consistency"][s], u_e["consistency"][s]
            lines.append(
                f"| {s} | {s_e['median_kbps'][s]:.0f} | {_pct(sc.get('M1'))} "
                f"| {_pct(sc.get('M2'))} | {_pct(sc.get('M3'))} "
                f"| {u_e['median_kbps'][s]:.0f} | {_pct(uc.get('M1'))} "
                f"| {_pct(uc.get('M2'))} | {_pct(uc.get('M3'))} |"
            )
    lines += [
        "",
        "## M2 estimate within 2 bpm of the M1 estimate (%)",
        "",
        "| Setting | SCAMPS | UBFC-rPPG |",
        "|---|---|---|",
    ]
    for s in ["source"] + [
        r["setting"] for lad in LADDERS for r in s_e["ladders"][lad]["M1"]["rows"]
    ]:
        lines.append(
            f"| {s} | {_pct(s_e['m2_follows_m1'].get(s))} | {_pct(u_e['m2_follows_m1'].get(s))} |"
        )
    for ds in ("scamps", "ubfc"):
        e = result[f"{ds}_allocation"]
        lines += [
            "",
            f"## Bitrate allocation, {names[ds]}: full frame rate and resolution vs the best cell",
            "",
            "Success rate (%) against the reference; for UBFC-rPPG M1 and M2 only 12 windows "
            "have a usable reference. Cells: dD_sS = every D-th frame, 1/S resolution "
            "(30 fps source: d2 = 15 fps, d3 = 10 fps, d6 = 5 fps).",
            "",
            "| Codec | kbps | Measure | d1_s1 | Best cell | Best | Compression effect at best "
            "(vs its uncompressed twin, pp) |",
            "|---|---|---|---|---|---|---|",
        ]
        for codec in ("x264", "x265"):
            for kbps in ("500", "250", "125"):
                for m in MEASURES:
                    b = e["best"][codec][kbps][m]
                    name = f"{codec}_br{kbps}_{b['cell']}"
                    eff = e["compression_effect"][name].get(m, {}).get("success_rate")
                    eff_txt = "–"
                    if eff:
                        eff_txt = _num(
                            {
                                **eff,
                                "delta": 100 * eff["delta"],
                                "ci": [100 * v for v in eff["ci"]],
                            },
                            "delta",
                            1,
                        )
                    lines.append(
                        f"| {codec} | {kbps} | {m} | {_pct(b['full_rate'])} | {b['cell']} "
                        f"| {_pct(b)} | {eff_txt} |"
                    )
    for ds in ("scamps", "ubfc"):
        e = result[f"{ds}_allocation"]
        lines += [
            "",
            f"## Bitrate allocation, {names[ds]}: pre-chosen reductions vs full frame rate and "
            "resolution (pp of success rate)",
            "",
            "Paired on the same windows, setting minus d1_s1 at the same codec and bitrate. "
            "half_resolution = d1_s2, half_frame_rate = d2_s1, both_halved = d2_s2.",
            "",
            "| Codec | kbps | Measure | half_resolution | half_frame_rate | both_halved |",
            "|---|---|---|---|---|---|",
        ]
        for codec in ("x264", "x265"):
            for kbps in ("500", "250", "125"):
                for m in MEASURES:
                    cols = []
                    for name in CONTRASTS:
                        c = e["contrasts"][codec][kbps][name].get(m, {}).get("success_rate")
                        cols.append(
                            "–"
                            if c is None
                            else _num(
                                {**c, "delta": 100 * c["delta"], "ci": [100 * v for v in c["ci"]]},
                                "delta",
                                1,
                            )
                        )
                    lines.append(f"| {codec} | {kbps} | {m} | " + " | ".join(cols) + " |")
    lines += [
        "",
        "## Bits per pixel and agreement with the uncompressed estimate",
        "",
        "Spearman rank correlation over all compressed settings of the ladder and allocation runs.",
        "",
        "| Dataset | Measure | Settings | rho |",
        "|---|---|---|---|",
    ]
    for ds in ("scamps", "ubfc"):
        for m, c in result[f"{ds}_bits_per_pixel_spearman"].items():
            lines.append(f"| {names[ds]} | {m} | {c['settings']} | {c['rho']:.2f} |")
    lines += [
        "",
        "## Compressed or reduced versions that scored better than the source",
        "",
        "| Dataset | Run | Setting | Measure | Metric | Δ | Windows | Failed → succeeded "
        "| Succeeded → failed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ds in ("scamps", "ubfc"):
        for kind in ("ladders", "allocation"):
            e = result[f"{ds}_{kind}"]
            for f in e["better_than_source"]:
                t = e["better_than_source_transitions"][f["setting"]][f["measure"]]
                lines.append(
                    f"| {names[ds]} | {kind} | {f['setting']} | {f['measure']} | {f['metric']} "
                    f"| {_num(f, 'delta', 3)} | {t['windows']} | {t['fail_to_success']} "
                    f"| {t['success_to_fail']} |"
                )
    lines += [
        "",
        "## Regions found on the compressed video (relocated minus fixed)",
        "",
        "| Dataset | Setting | Measure | Δ success (pp) | Δ MAE (bpm) |",
        "|---|---|---|---|---|",
    ]
    for ds in ("scamps", "ubfc"):
        for s, cells in result[f"{ds}_ladders"]["relocated_minus_fixed"].items():
            for m, c in cells.items():
                dsr = {
                    **c["success_rate"],
                    "delta": 100 * c["success_rate"]["delta"],
                    "ci": [100 * v for v in c["success_rate"]["ci"]],
                }
                lines.append(
                    f"| {names[ds]} | {s} | {m} | {_num(dsr, 'delta', 1)} "
                    f"| {_num(c['mae'], 'delta')} |"
                )
    return "\n".join(lines) + "\n"


def figures(result: dict, out: Path) -> list[str]:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_ladders(
        result["scamps_ladders"],
        MEASURES,
        fig_dir / "ladders_scamps.png",
        "SCAMPS (100 videos): accuracy against the exact reference",
    )
    plot_ladders(
        result["ubfc_ladders"],
        ("M3",),
        fig_dir / "ladders_ubfc_heart.png",
        "UBFC-rPPG (42 subjects): heart rate against the contact PPG",
    )
    plot_agreement(result, fig_dir / "ladders_agreement.png")
    plot_allocation(
        result["scamps_allocation"],
        [(m, "success") for m in MEASURES],
        fig_dir / "allocation_scamps.png",
        "SCAMPS: bitrate allocation, video-call settings",
    )
    plot_allocation(
        result["ubfc_allocation"],
        [("M1", "agreement"), ("M2", "agreement"), ("M3", "success")],
        fig_dir / "allocation_ubfc.png",
        "UBFC-rPPG: bitrate allocation, video-call settings",
    )
    return sorted(p.name for p in fig_dir.glob("*.png"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS / OUT_ID)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    protocol = pipeline.load_protocol()
    result = analyse(protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    (args.out / "tables.md").write_text(tables(result))
    if not args.no_figures:
        print("figures:", ", ".join(figures(result, args.out)), flush=True)
    snapshot = environment_snapshot(PROJECT_DIR, {"inputs": list(RUNS.values())})
    (args.out / "manifest.json").write_text(json.dumps(to_jsonable(snapshot), indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
