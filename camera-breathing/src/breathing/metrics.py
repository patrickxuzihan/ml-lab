"""Metrics, subject-level bootstrap confidence intervals, paired comparisons and breakdown.

Inputs are per-window arrays: the estimate (NaN when the measurement produced none), the
reference, the spectral SNR, and a cluster id (the subject). A missing estimate counts as a
failure in the success rate; MAE, RMSE, MAPE and r use the windows with an estimate, and the
coverage (share of windows with an estimate) is reported next to them.

Windows from the same subject are not independent, so confidence intervals resample whole
subjects (cluster bootstrap, percentile intervals). Paired comparisons apply the same
resampled subjects to both settings and report the CI of the difference.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

METRICS = ("mae", "rmse", "mape", "pearson_r", "success_rate", "snr_db")


def _f(value) -> float:
    return np.nan if value is None else float(value)


def metric_values(est: np.ndarray, ref: np.ndarray, snr: np.ndarray, tolerance: float) -> dict:
    """Point values of every metric for one set of windows (arrays of equal length)."""
    est, ref, snr = (np.asarray(a, dtype=np.float64) for a in (est, ref, snr))
    have = np.isfinite(est)
    err = est[have] - ref[have]
    out = {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "pearson_r": np.nan}
    if err.size:
        out["mae"] = float(np.mean(np.abs(err)))
        out["rmse"] = float(np.sqrt(np.mean(err**2)))
        out["mape"] = float(100 * np.mean(np.abs(err) / ref[have]))
    if err.size >= 3 and np.std(est[have]) > 0 and np.std(ref[have]) > 0:
        out["pearson_r"] = float(np.corrcoef(est[have], ref[have])[0, 1])
    success = np.zeros(len(est), dtype=bool)
    success[have] = np.abs(err) <= tolerance
    out["success_rate"] = float(success.mean()) if len(est) else np.nan
    finite_snr = snr[np.isfinite(snr)]
    out["snr_db"] = float(finite_snr.mean()) if finite_snr.size else np.nan
    return out


def cluster_groups(clusters: list) -> list[np.ndarray]:
    """Window indices of each cluster, in first-seen order."""
    order: dict = {}
    for i, c in enumerate(clusters):
        order.setdefault(c, []).append(i)
    return [np.array(v, dtype=int) for v in order.values()]


def bootstrap(
    statistic: Callable[[np.ndarray], dict],
    groups: list[np.ndarray],
    n_boot: int,
    ci: float,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Percentile CI of each value returned by `statistic(indices)` under cluster resampling.

    Resamples whose value is NaN (e.g. r with too few windows) are left out for that value.
    """
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    g = len(groups)
    for _ in range(n_boot if g else 0):
        pick = rng.integers(0, g, g)
        values = statistic(np.concatenate([groups[k] for k in pick]))
        for key, value in values.items():
            samples.setdefault(key, []).append(value)
    alpha = (1 - ci) / 2
    out = {}
    for key, values in samples.items():
        arr = np.array(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        out[key] = (
            (float(np.quantile(arr, alpha)), float(np.quantile(arr, 1 - alpha)))
            if arr.size
            else (np.nan, np.nan)
        )
    return out


def summarize(
    est: np.ndarray,
    ref: np.ndarray,
    snr: np.ndarray,
    clusters: list,
    tolerance: float,
    stats_cfg: dict,
) -> dict:
    """Point values with bootstrap CIs, plus window and subject counts and coverage."""
    est, ref, snr = (np.asarray(a, dtype=np.float64) for a in (est, ref, snr))
    groups = cluster_groups(clusters)
    point = metric_values(est, ref, snr, tolerance)
    cis = bootstrap(
        lambda idx: metric_values(est[idx], ref[idx], snr[idx], tolerance),
        groups,
        stats_cfg["bootstrap_samples"],
        stats_cfg["ci"],
        stats_cfg["seed"],
    )
    out = {
        "n_windows": int(len(est)),
        "n_subjects": len(groups),
        "coverage": float(np.isfinite(est).mean()) if len(est) else np.nan,
    }
    for key in METRICS:
        out[key] = {"value": point[key], "ci": list(cis.get(key, (np.nan, np.nan)))}
    return out


def paired_difference(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    clusters: list,
    tolerance: float,
    stats_cfg: dict,
) -> dict:
    """Metric(b) - metric(a) on the same windows, with cluster-bootstrap CIs.

    a and b hold aligned arrays "est", "ref" and "snr" (window i of a is window i of b).
    A difference is called significant when its CI excludes zero.
    """

    def diff(idx: np.ndarray) -> dict:
        ma = metric_values(a["est"][idx], a["ref"][idx], a["snr"][idx], tolerance)
        mb = metric_values(b["est"][idx], b["ref"][idx], b["snr"][idx], tolerance)
        return {k: mb[k] - ma[k] for k in METRICS}

    groups = cluster_groups(clusters)
    point = diff(np.arange(len(clusters)))
    cis = bootstrap(
        diff, groups, stats_cfg["bootstrap_samples"], stats_cfg["ci"], stats_cfg["seed"]
    )
    out = {"n_windows": len(clusters), "n_subjects": len(groups)}
    for key in METRICS:
        lo, hi = cis.get(key, (np.nan, np.nan))
        out[key] = {
            "delta": point[key],
            "ci": [lo, hi],
            "significant": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
        }
    return out


def breakdown_point(rungs: list[dict], drop_pp: float) -> str | None:
    """First rung (best quality first) that has broken down, or None.

    Each rung is {"name", "delta", "ci_high"} with the success-rate difference from the
    uncompressed source as a fraction (negative = worse). A rung has broken down when the
    success rate fell by more than drop_pp percentage points with the CI's upper bound below
    zero, and every lower-quality rung also fell by more than drop_pp.
    """
    threshold = -drop_pp / 100.0

    def dropped(r: dict) -> bool:
        return np.isfinite(_f(r["delta"])) and _f(r["delta"]) < threshold

    for i, rung in enumerate(rungs):
        if dropped(rung) and _f(rung["ci_high"]) < 0 and all(dropped(r) for r in rungs[i:]):
            return rung["name"]
    return None
