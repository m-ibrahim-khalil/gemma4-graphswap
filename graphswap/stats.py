"""Paired comparison statistics over per-task resolution outcomes."""

from math import comb

import numpy as np

N_BOOT = 10_000


def _to_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1"):
            return True
        if text in ("false", "0"):
            return False
    raise ValueError(f"cannot interpret resolved value {value!r} as bool")


def _index(rows, key_fn, side):
    out = {}
    dupes = []
    for row in rows:
        k = key_fn(row)
        if k in out:
            dupes.append(k)
        out[k] = _to_bool(row["resolved"])
    if dupes:
        raise ValueError(f"duplicate keys in {side}: {sorted(set(dupes), key=str)}")
    return out


def _sorted_natural(values):
    """Sorted by the values' own order; only mixed types fall back to the string form.

    The bootstrap indexes clusters by position, so the order of the cluster ids is part of
    the result. Sorting ids of one type (str, int, tuple) by ``str`` would order 10 before
    2; sorting them naturally does not, and leaves str ids exactly where they were.
    """
    try:
        return sorted(values)
    except TypeError:  # e.g. int and str ids in one run
        return sorted(values, key=str)


def _compare(a_rows, b_rows, key_fn, cluster_fn=None):
    a = _index(a_rows, key_fn, "control")
    b = _index(b_rows, key_fn, "treatment")
    if a.keys() != b.keys():
        only_a = sorted(a.keys() - b.keys(), key=str)
        only_b = sorted(b.keys() - a.keys(), key=str)
        raise ValueError(
            f"unmatched rows: only in control {only_a}; only in treatment {only_b}"
        )
    keys = sorted(a, key=str)
    x = np.array([a[k] for k in keys], dtype=float)
    y = np.array([b[k] for k in keys], dtype=float)
    n = len(keys)
    if n == 0:
        raise ValueError("no rows to compare")
    deltas = y - x
    wins = int((deltas > 0).sum())
    losses = int((deltas < 0).sum())
    # Bootstrap over clusters (tasks), not rows: rows of one task at several
    # seeds are not independent. Each draw concatenates the drawn clusters'
    # deltas and takes the mean, so rows are weighted equally.
    cluster_ids = [cluster_fn(k) if cluster_fn else k for k in keys]
    uniq = _sorted_natural(set(cluster_ids))
    pos = {c: i for i, c in enumerate(uniq)}
    n_clusters = len(uniq)
    sums = np.zeros(n_clusters)
    counts = np.zeros(n_clusters)
    for c, d in zip(cluster_ids, deltas):
        sums[pos[c]] += d
        counts[pos[c]] += 1
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_clusters, size=(N_BOOT, n_clusters))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": n,
        "n_clusters": n_clusters,
        "control_rate": float(x.mean()),
        "treatment_rate": float(y.mean()),
        "diff": float(deltas.mean()),
        "wins": wins,
        "losses": losses,
        "ties": n - wins - losses,
        "mcnemar_p": _exact_two_sided(wins, losses),
        "boot_ci": (float(lo), float(hi)),
    }


def _exact_two_sided(wins, losses):
    m = wins + losses
    if m == 0:
        return 1.0

    def cdf(k):
        return sum(comb(m, i) for i in range(k + 1)) / 2**m

    return min(1.0, 2 * min(cdf(losses), cdf(wins)))


def paired(control, treatment):
    """Compare two runs, matching rows on (instance_id, seed).

    The bootstrap CI resamples tasks (instance_id), so pooled seeds are clustered.
    """
    return _compare(
        control,
        treatment,
        lambda r: (r["instance_id"], r["seed"]),
        cluster_fn=lambda key: key[0],
    )


def noise_floor(run_seed_a, run_seed_b):
    """Compare two same-condition runs that differ only in seed."""
    out = _compare(run_seed_a, run_seed_b, lambda r: r["instance_id"])
    out["discordant"] = out["wins"] + out["losses"]
    return out
