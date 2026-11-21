"""Extra checks on phase-3 baseline runs, computed from their items.jsonl (no data needed).

Writes checks.json and checks.md into each run directory:

- heart rate (M3): the failed windows, how many are harmonics of the reference, and the
  success rate below and above a heart-rate split;
- breathing (UBFC-rPPG): on the windows that have a reference under the wide rule, how the
  camera estimates M1 and M2 compare with each other and with each PPG-derived rate (BW, AM,
  FM), overall and by reference range;
- optionally, whether videos shared with another run gave identical per-window results.

  python analysis/baseline_checks.py results/20261122_ubfc_baseline_s0
  python analysis/baseline_checks.py results/20261122_scamps_baseline_s0 \
      --same-videos-as results/20261122_scamps_baseline-example_s0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

HARMONICS = {"second harmonic (x2)": (1.9, 2.1), "half (x0.5)": (0.45, 0.55)}
HEART_SPLIT_BPM = 60.0
REFERENCE_BINS = ((0.0, 12.0), (12.0, 15.0), (15.0, 60.0))


def load(run_dir: Path) -> list[dict]:
    latest = {}
    for line in (run_dir / "items.jsonl").read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["id"]] = record
    return [r for r in latest.values() if r.get("status") == "ok"]


def short(name: str) -> str:
    return PurePosixPath(name).name.removesuffix(".mat")


def heart_checks(records: list[dict], tolerance: float) -> dict:
    """Failed M3 windows (main result) and what they have in common."""
    windows = []
    for r in records:
        details = r.get("reference", {}).get("M3") or []
        main = [w for w in r["windows"] if w["measure"] == "M3" and w["excluded"] is None]
        for w in main:
            detail = details[w["window"]] if w["window"] < len(details) else {}
            windows.append(
                {
                    "video": short(r["subject"]),
                    "window": w["window"],
                    "ref_bpm": w["ref_bpm"],
                    "est_bpm": w["est_bpm"],
                    "failed": w["est_bpm"] is None or abs(w["error_bpm"]) > tolerance,
                    "ppg_snr_db": detail.get("ppg_snr_db"),
                    "oximeter_bpm": detail.get("oximeter_bpm"),
                }
            )
    failed = [w for w in windows if w["failed"]]
    ratios = [w["est_bpm"] / w["ref_bpm"] for w in failed if w["est_bpm"] is not None]
    harmonics = {name: sum(lo <= x <= hi for x in ratios) for name, (lo, hi) in HARMONICS.items()}
    split = {}
    for label, keep in (
        (f"below {HEART_SPLIT_BPM:g} bpm", lambda w: w["ref_bpm"] < HEART_SPLIT_BPM),
        (f"{HEART_SPLIT_BPM:g} bpm and above", lambda w: w["ref_bpm"] >= HEART_SPLIT_BPM),
    ):
        sel = [w for w in windows if keep(w)]
        split[label] = {"windows": len(sel), "failed": sum(w["failed"] for w in sel)}

    def snr(ws):
        values = [w["ppg_snr_db"] for w in ws if w["ppg_snr_db"] is not None]
        return float(np.median(values)) if values else None

    return {
        "windows": len(windows),
        "failed": len(failed),
        "failed_by_video": dict(Counter(w["video"] for w in failed).most_common()),
        "harmonics_among_failed": harmonics,
        "by_reference_rate": split,
        "ppg_snr_db_median": {
            "failed": snr(failed),
            "succeeded": snr([w for w in windows if not w["failed"]]),
        },
        "failed_windows": sorted(failed, key=lambda w: (w["video"], w["window"])),
    }


def _agreement(a: list[float], b: list[float], tolerance: float) -> dict:
    d = np.abs(np.asarray(a) - np.asarray(b))
    return {
        "median_abs_diff_bpm": float(np.median(d)) if d.size else None,
        "within_tolerance": int(np.sum(d <= tolerance)),
        "n": int(d.size),
    }


def breathing_checks(records: list[dict], tolerance: float, variant: str) -> dict:
    """Camera breathing estimates against each other and each PPG-derived rate."""
    rows = []
    for r in records:
        windows = {(w["measure"], w["window"]): w for w in r["variants"][variant]}
        for k, ref in enumerate(r["reference"]["M1"]):
            m1, m2 = windows[("M1", k)], windows[("M2", k)]
            if m1["excluded"] is None and m1["est_bpm"] is not None and m2["est_bpm"] is not None:
                rows.append(
                    {
                        "ref": m1["ref_bpm"],
                        "M1": m1["est_bpm"],
                        "M2": m2["est_bpm"],
                        "BW": ref["bw_bpm"],
                        "AM": ref["am_bpm"],
                        "FM": ref["fm_bpm"],
                    }
                )

    def col(key, sel=None):
        return [x[key] for x in (rows if sel is None else sel)]

    pairs = {"M1 vs M2": _agreement(col("M1"), col("M2"), tolerance)}
    for camera in ("M1", "M2"):
        for method in ("BW", "AM", "FM"):
            pairs[f"{camera} vs {method}"] = _agreement(col(camera), col(method), tolerance)
    by_range = {}
    for lo, hi in REFERENCE_BINS:
        sel = [x for x in rows if lo <= x["ref"] < hi]
        by_range[f"{lo:g}-{hi:g} bpm"] = {
            "windows": len(sel),
            "M1 within": int(sum(abs(x["M1"] - x["ref"]) <= tolerance for x in sel)),
            "M2 within": int(sum(abs(x["M2"] - x["ref"]) <= tolerance for x in sel)),
        }

    def quantiles(values):
        return [float(q) for q in np.quantile(values, [0, 0.25, 0.5, 0.75, 1])] if values else None

    return {
        "variant": variant,
        "windows": len(rows),
        "pairs": pairs,
        "by_reference_range": by_range,
        "estimate_quantiles_bpm": {k: quantiles(col(k)) for k in ("M1", "M2", "ref")},
    }


def identical_videos(records: list[dict], other: list[dict]) -> dict:
    """Videos present in both runs, and how many gave identical per-window results."""

    def key(r):
        return short(r["subject"])

    def windows(r):
        return [(w["measure"], w["window"], w["est_bpm"], w["ref_bpm"]) for w in r["windows"]]

    theirs = {key(r): r for r in other}
    shared = [r for r in records if key(r) in theirs]
    same = [key(r) for r in shared if windows(r) == windows(theirs[key(r)])]
    return {"shared": len(shared), "identical": len(same)}


def checks(records: list[dict], config: dict, other: list[dict] | None = None) -> dict:
    tolerance = config["protocol"]["tolerance_bpm"]
    out = {"heart": heart_checks(records, tolerance["heart"])}
    if config["experiment"]["dataset"] == "ubfc":
        out["breathing"] = breathing_checks(records, tolerance["breathing"], "wide_reference")
    ids = sorted(short(r["subject"]) for r in records)
    out["videos"] = {
        "n": len(ids),
        "first": ids[0] if ids else None,
        "last": ids[-1] if ids else None,
    }
    if other is not None:
        out["same_videos_as_other_run"] = identical_videos(records, other)
    return out


def _f(value, digits=1) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def markdown(result: dict, run_id: str) -> str:
    h = result["heart"]
    lines = [
        f"# Extra checks: {run_id}",
        "",
        "Computed from items.jsonl by analysis/baseline_checks.py.",
        "",
        f"Videos: {result['videos']['n']} ({result['videos']['first']} to "
        f"{result['videos']['last']}).",
        "",
        "## Heart rate (M3): failed windows",
        "",
        f"Failed: {h['failed']} of {h['windows']} windows.",
        "",
        "| Quantity | Value |",
        "|---|---|",
    ]
    for name, n in h["harmonics_among_failed"].items():
        lines.append(f"| Failures with estimate ≈ {name} of the reference | {n} |")
    for label, cell in h["by_reference_rate"].items():
        lines.append(
            f"| Reference {label}: failed / windows | {cell['failed']} / {cell['windows']} |"
        )
    snr = h["ppg_snr_db_median"]
    if snr["failed"] is not None or snr["succeeded"] is not None:
        lines.append(
            f"| PPG SNR median (dB): failed / succeeded | {_f(snr['failed'])} / "
            f"{_f(snr['succeeded'])} |"
        )
    by_video = ", ".join(f"{k} ({v})" for k, v in h["failed_by_video"].items())
    lines.append(f"| Failures by video | {by_video or '–'} |")
    lines += [
        "",
        "| Video | Window | Reference (bpm) | Estimate (bpm) | Estimate / reference "
        "| PPG SNR (dB) | Oximeter (bpm) |",
        "|---|---|---|---|---|---|---|",
    ]
    for w in h["failed_windows"]:
        ratio = None if w["est_bpm"] is None else w["est_bpm"] / w["ref_bpm"]
        lines.append(
            f"| {w['video']} | {w['window']} | {_f(w['ref_bpm'])} | {_f(w['est_bpm'])} "
            f"| {_f(ratio, 2)} | {_f(w['ppg_snr_db'])} | {_f(w['oximeter_bpm'], 0)} |"
        )
    if "breathing" in result:
        b = result["breathing"]
        q = b["estimate_quantiles_bpm"]
        lines += [
            "",
            "## Breathing: camera estimates against the PPG-derived rates",
            "",
            f"Windows with a reference under the {b['variant']} rule and both camera "
            f"estimates: {b['windows']}. Agreement means within the breathing tolerance.",
            "",
            "| Pair | Median abs difference (bpm) | Agree |",
            "|---|---|---|",
        ]
        for pair, cell in b["pairs"].items():
            lines.append(
                f"| {pair} | {_f(cell['median_abs_diff_bpm'])} "
                f"| {cell['within_tolerance']} of {cell['n']} |"
            )
        lines += [
            "",
            "| Reference range | Windows | M1 agrees | M2 agrees |",
            "|---|---|---|---|",
        ]
        for label, cell in b["by_reference_range"].items():
            lines.append(
                f"| {label} | {cell['windows']} | {cell['M1 within']} | {cell['M2 within']} |"
            )
        lines += ["", "Quantiles (min / 25% / median / 75% / max, bpm):", ""]
        for key in ("M1", "M2", "ref"):
            values = " / ".join(_f(v) for v in q[key]) if q[key] else "–"
            lines.append(f"- {'reference' if key == 'ref' else key}: {values}")
    if "same_videos_as_other_run" in result:
        s = result["same_videos_as_other_run"]
        lines += [
            "",
            "## Reproducibility",
            "",
            f"Videos also measured in the other run: {s['shared']}; identical per-window "
            f"results: {s['identical']}.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--same-videos-as", type=Path, help="another run of the same dataset")
    args = parser.parse_args(argv)
    import yaml

    config = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    other = load(args.same_videos_as) if args.same_videos_as else None
    result = checks(load(args.run_dir), config, other)
    (args.run_dir / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.run_dir / "checks.md").write_text(markdown(result, args.run_dir.name))
    print(markdown(result, args.run_dir.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
