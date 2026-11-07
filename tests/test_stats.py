import numpy as np
import pytest

from graphswap import stats
from graphswap.stats import N_BOOT, paired


def rows(resolved, seed=0, prefix="t"):
    return [
        {"instance_id": f"{prefix}{i}", "seed": seed, "resolved": r}
        for i, r in enumerate(resolved)
    ]


def test_identical_runs_have_zero_diff_and_p_one():
    r = rows([True, False, True, False])
    out = stats.paired(r, list(r))
    assert out["n"] == 4
    assert out["diff"] == 0
    assert out["mcnemar_p"] == 1.0
    assert (out["wins"], out["losses"], out["ties"]) == (0, 0, 4)


def test_ten_wins_zero_losses_is_significant():
    out = stats.paired(rows([False] * 10), rows([True] * 10))
    assert (out["wins"], out["losses"]) == (10, 0)
    assert out["diff"] == 1.0
    assert out["mcnemar_p"] < 0.01


def test_mixed_discordant_counts_and_p():
    ctrl = rows([True, True, False, False, True])
    trt = rows([False, True, True, True, True])
    out = stats.paired(ctrl, trt)
    assert (out["wins"], out["losses"], out["ties"]) == (2, 1, 2)
    assert out["control_rate"] == pytest.approx(0.6)
    assert out["treatment_rate"] == pytest.approx(0.8)
    assert out["diff"] == pytest.approx(0.2)
    assert out["mcnemar_p"] == 1.0  # 2 * P(X<=1 | n=3) = 2 * 0.5, capped


def test_mismatched_ids_raise_with_ids_in_message():
    with pytest.raises(ValueError, match="t9"):
        stats.paired(rows([True, False]), rows([True, False]) + rows([True], prefix="t9"))
    with pytest.raises(ValueError, match="t1"):
        stats.paired(rows([True, False]), rows([True]))


def test_duplicate_keys_raise():
    r = rows([True, False])
    with pytest.raises(ValueError, match="t0"):
        stats.paired(r + r[:1], r)


def test_bootstrap_ci_contains_diff_and_is_deterministic():
    ctrl = rows([True] * 10 + [False] * 30)
    trt = rows([True] * 20 + [False] * 20)
    a = stats.paired(ctrl, trt)
    b = stats.paired(ctrl, trt)
    lo, hi = a["boot_ci"]
    assert lo <= a["diff"] <= hi
    assert lo > 0
    assert a["boot_ci"] == b["boot_ci"]


def test_noise_floor_pairs_on_instance_id_across_seeds():
    a = rows([True, True, False, False], seed=1)
    b = rows([True, False, True, False], seed=2)
    out = stats.noise_floor(a, b)
    assert out["n"] == 4
    assert out["discordant"] == 2
    assert out["discordant"] == out["wins"] + out["losses"]
    assert out["diff"] == 0


def test_string_resolved_values_match_bool_result():
    flags = [True, False, False, True]
    strs = ["True", "false", " 0 ", "1"]
    ctrl = rows(flags)
    trt = rows([True, True, False, False])
    expected = stats.paired(ctrl, trt)
    assert stats.paired(rows(strs), trt) == expected
    assert expected["diff"] != 0 or expected["wins"] > 0


@pytest.mark.parametrize("bad", [None, "", float("nan"), "maybe", 2])
def test_unparseable_resolved_raises(bad):
    r = rows([True, False])
    r[0]["resolved"] = bad
    with pytest.raises(ValueError, match="resolved"):
        stats.paired(r, rows([True, False]))


def test_paired_pooled_bootstrap_clusters_by_task():
    # 4 tasks x 2 seeds; treatment wins on task t1 at both seeds, nothing else differs
    control = [dict(instance_id=t, seed=s, resolved=False) for t in "t1 t2 t3 t4".split() for s in (1, 2)]
    treatment = [dict(instance_id=t, seed=s, resolved=(t == "t1")) for t in "t1 t2 t3 t4".split() for s in (1, 2)]
    out = paired(control, treatment)
    assert out["n"] == 8 and out["n_clusters"] == 4
    assert out["wins"] == 2 and out["losses"] == 0
    # with 4 clusters, the resampled mean delta can only take values k/4 for k in 0..4
    lo, hi = out["boot_ci"]
    assert lo == 0.0 and hi in (0.5, 0.75)


def test_paired_single_seed_ci_unchanged():
    # one seed: clusters == rows, so the CI equals the old row-level bootstrap
    control = [dict(instance_id=f"t{i}", seed=1, resolved=i % 3 == 0) for i in range(12)]
    treatment = [dict(instance_id=f"t{i}", seed=1, resolved=i % 2 == 0) for i in range(12)]
    out = paired(control, treatment)
    assert out["n"] == out["n_clusters"] == 12
    # _compare orders keys by str ("t0", "t1", "t10", ...), so build deltas in that order
    order = sorted(range(12), key=lambda i: f"t{i}")
    deltas = np.array([float(i % 2 == 0) - float(i % 3 == 0) for i in order])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 12, size=(N_BOOT, 12))
    lo, hi = np.percentile(deltas[idx].mean(axis=1), [2.5, 97.5])
    assert out["boot_ci"] == (float(lo), float(hi))


def test_cluster_order_is_natural_not_stringified():
    # the bootstrap indexes clusters by position, so the order is part of the result;
    # it must come from the ids themselves, not from str(id)
    assert stats._sorted_natural({2, 10, 1}) == [1, 2, 10]
    assert stats._sorted_natural({(1, 9), (1, 10), (2, 1)}) == [(1, 9), (1, 10), (2, 1)]
    assert stats._sorted_natural({"t10", "t2"}) == ["t10", "t2"]  # str ids: unchanged order
    assert stats._sorted_natural({1, "a"}) == sorted([1, "a"], key=str)  # mixed: str fallback


def test_string_cluster_ids_keep_their_previous_order():
    """Every committed number has str task ids, whose natural order is their str order."""
    ids = {f"astropy__astropy-{i}" for i in (7, 12907, 1234)} | {"t2", "t10"}
    assert stats._sorted_natural(ids) == sorted(ids, key=str)


def test_integer_cluster_ids_give_the_same_ci_as_their_sorted_order():
    control = [dict(instance_id=i, seed=1, resolved=False) for i in (1, 2, 10)]
    treatment = [dict(instance_id=i, seed=1, resolved=i != 2) for i in (1, 2, 10)]
    renamed = {1: "a1", 2: "a2", 10: "a3"}  # same natural order, different string order
    out = paired(control, treatment)
    same = paired([dict(r, instance_id=renamed[r["instance_id"]]) for r in control],
                  [dict(r, instance_id=renamed[r["instance_id"]]) for r in treatment])
    assert out["boot_ci"] == same["boot_ci"]
