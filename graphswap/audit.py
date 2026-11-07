"""Audit statistics: embedding geometry and gold-symbol coverage of a graph."""
import re
from pathlib import Path

import numpy as np

from graphswap.builder import build_graph
from graphswap.tasks import Task

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


def embedding_geometry(vectors: dict[str, np.ndarray], n_pairs: int = 20000, seed: int = 0) -> dict:
    keys = list(vectors)
    n = len(keys)
    if n < 2:
        return dict.fromkeys(
            ("mean_random_cosine", "p05", "p95", "duplicate_rate", "effective_dim"), float("nan"))
    x = np.stack([np.asarray(vectors[k], dtype=np.float64) for k in keys])
    counts: dict[bytes, int] = {}
    for k in keys:
        b = np.asarray(vectors[k]).tobytes()
        counts[b] = counts.get(b, 0) + 1
    dups = sum(c for c in counts.values() if c > 1)  # vectors with a byte-equal twin
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.where(norms == 0, 1.0, norms)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = (i + rng.integers(1, n, n_pairs)) % n  # uniform over j != i
    cos = np.einsum("ij,ij->i", x[i], x[j])
    lam = np.clip(np.linalg.eigvalsh(np.atleast_2d(np.cov(x, rowvar=False))), 0, None)
    denom = float((lam ** 2).sum())
    return {
        "mean_random_cosine": float(cos.mean()),
        "p05": float(np.percentile(cos, 5)),
        "p95": float(np.percentile(cos, 95)),
        "duplicate_rate": dups / n,
        "effective_dim": float(lam.sum() ** 2 / denom) if denom > 0 else float("nan"),
    }


def _touched_lines(patch: str) -> dict[str, set[int]]:
    """Pre-image line numbers per source file: removed lines, plus the pre-image line
    nearest each pure addition. Files added by the patch (``--- /dev/null``) are omitted."""
    out: dict[str, set[int]] = {}
    path, old = None, 0
    lines = patch.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("--- ") and idx + 1 < len(lines) and lines[idx + 1].startswith("+++ "):
            src = line[4:].split("\t")[0]
            path = None if src == "/dev/null" else re.sub(r"^a/", "", src)
            old = 0
            continue
        m = _HUNK.match(line)
        if m:
            old = int(m.group(1))
            if m.group(2) == "0":  # pure insertion: start is the line *before* the insert point
                old += 1
            if path is not None:
                out.setdefault(path, set())
            continue
        if path is None or not old:
            continue
        if line.startswith("-"):
            out[path].add(old)
            old += 1
        elif line.startswith("+"):
            out[path].add(max(old - 1, 1))
        elif not line.startswith("\\"):
            old += 1
    return out


def gold_symbols(task: Task, worktree: Path, base_graph: dict | None = None) -> list[str] | None:
    """Ids of the innermost def/class enclosing each pre-image line the patch edits.

    ``None`` when the patch touches no existing .py file. ``base_graph`` is the builder graph
    of ``worktree`` if the caller already has it (avoids a rebuild)."""
    worktree = Path(worktree)
    touched = {f: ls for f, ls in _touched_lines(task.patch).items()
               if f.endswith(".py") and (worktree / f).is_file()}
    if not touched:
        return None
    graph = base_graph or build_graph(worktree, task.repo_short)
    spans: dict[str, list[dict]] = {}
    for n in graph["nodes"]:
        if n["kind"] != "module":
            spans.setdefault(n["file"], []).append(n)
    gold = set()
    for f, lines in touched.items():
        for ln in lines:
            hits = [n for n in spans.get(f, []) if n["lineno"] <= ln <= n["end_lineno"]]
            if hits:
                gold.add(max(hits, key=lambda n: n["lineno"])["id"])
    return sorted(gold)


_UNSET: list[str] | None = []  # sentinel compared by identity, never mutated


def gold_coverage(task: Task, worktree: Path, graph: dict, base_graph: dict | None = None,
                  *, gold: list[str] | None = _UNSET) -> dict:
    """``gold`` may be passed if the caller already computed ``gold_symbols`` (skips the call)."""
    if gold is _UNSET:
        gold = gold_symbols(task, worktree, base_graph)
    ids = {n["id"] for n in graph["nodes"]}
    covered = [g for g in gold or [] if g in ids]
    fraction = len(covered) / len(gold) if gold else None
    return {"gold": gold, "covered": covered, "fraction": fraction}
