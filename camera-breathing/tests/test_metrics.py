"""Metrics, cluster bootstrap, paired differences and the breakdown point."""

import numpy as np
import pytest

from breathing import metrics

STATS = {"bootstrap_samples": 500, "ci": 0.95, "seed": 0}
NAN = np.nan


def test_metric_values_by_hand():
    est = np.array([10.0, 12.0, NAN, 7.0])
    ref = np.array([10.0, 10.0, 10.0, 10.0])
    snr = np.array([3.0, NAN, 1.0, 2.0])
    m = metrics.metric_values(est, ref, snr, tolerance=2.0)
    assert m["mae"] == pytest.approx(5 / 3)  # errors 0, 2, -3; the missing one is left out
    assert m["rmse"] == pytest.approx(np.sqrt(13 / 3))
    assert m["mape"] == pytest.approx(100 * 5 / 30)
    assert m["success_rate"] == pytest.approx(2 / 4)  # missing counts as a failure
    assert m["snr_db"] == pytest.approx(2.0)
    assert np.isnan(m["pearson_r"])  # reference has no spread


def test_pearson_and_empty_input():
    m = metrics.metric_values([1.0, 2.0, 3.5], [1.0, 2.0, 3.0], [NAN] * 3, 1.0)
    assert m["pearson_r"] > 0.99
    empty = metrics.metric_values([], [], [], 1.0)
    assert all(np.isnan(v) for v in empty.values())


def test_cluster_groups_keep_first_seen_order():
    groups = metrics.cluster_groups(["b", "a", "b", "c", "a"])
    assert [g.tolist() for g in groups] == [[0, 2], [1, 4], [3]]


def test_bootstrap_resamples_whole_subjects():
    # Two subjects: one always right, one always wrong. Resampling windows would give a
    # narrow CI around 0.5; resampling subjects gives 0, 0.5 or 1.
    est = np.array([10.0, 10.0, 10.0, 20.0, 20.0, 20.0])
    ref = np.full(6, 10.0)
    clusters = ["a", "a", "a", "b", "b", "b"]
    out = metrics.summarize(est, ref, np.full(6, NAN), clusters, 2.0, STATS)
    assert out["n_subjects"] == 2 and out["n_windows"] == 6 and out["coverage"] == 1.0
    assert out["success_rate"]["value"] == 0.5
    assert out["success_rate"]["ci"] == [0.0, 1.0]


def test_bootstrap_of_identical_subjects_is_degenerate():
    est = np.tile([10.0, 11.0], 5)
    ref = np.full(10, 10.0)
    clusters = np.repeat(np.arange(5), 2).tolist()
    out = metrics.summarize(est, ref, np.full(10, NAN), clusters, 2.0, STATS)
    assert out["mae"]["ci"] == pytest.approx([0.5, 0.5])


def test_paired_difference_sign_and_significance():
    rng = np.random.default_rng(1)
    n_subjects, per = 10, 3
    clusters = np.repeat(np.arange(n_subjects), per).tolist()
    ref = rng.uniform(10, 20, n_subjects * per)
    a = {"est": ref + rng.normal(0, 0.3, ref.size), "ref": ref, "snr": np.full(ref.size, 10.0)}
    b = {"est": ref + 5.0, "ref": ref, "snr": np.full(ref.size, 4.0)}
    out = metrics.paired_difference(a, b, clusters, 2.0, STATS)
    assert out["success_rate"]["delta"] == pytest.approx(-1.0)
    assert out["success_rate"]["significant"]
    assert out["snr_db"]["delta"] == pytest.approx(-6.0)
    assert out["mae"]["delta"] > 4

    same = metrics.paired_difference(a, a, clusters, 2.0, STATS)
    assert same["mae"]["delta"] == 0 and not same["mae"]["significant"]


def rung(name, delta, ci_high):
    return {"name": name, "delta": delta, "ci_high": ci_high}


def test_breakdown_point():
    ok = [rung("q1", 0.0, 0.05), rung("q2", -0.05, 0.0)]
    assert metrics.breakdown_point(ok, 10.0) is None
    ladder = [rung("q1", 0.0, 0.05), rung("q2", -0.3, -0.1), rung("q3", -0.6, -0.4)]
    assert metrics.breakdown_point(ladder, 10.0) == "q2"
    # A one-rung blip followed by recovery is not a breakdown.
    blip = [rung("q1", -0.3, -0.1), rung("q2", 0.0, 0.05), rung("q3", -0.5, -0.2)]
    assert metrics.breakdown_point(blip, 10.0) == "q3"
    # A large but uncertain drop does not count; the next rung does.
    unsure = [rung("q1", -0.2, 0.05), rung("q2", -0.4, -0.2)]
    assert metrics.breakdown_point(unsure, 10.0) == "q2"
    assert metrics.breakdown_point([rung("q1", NAN, NAN)], 10.0) is None
