"""Load the public task file."""
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    instance_id: str
    repo: str
    repo_short: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str


def load_tasks(path: Path) -> list[Task]:
    tasks = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            tasks.append(
                Task(
                    instance_id=d["instance_id"],
                    repo=d["repo"],
                    repo_short=d["repo"].split("/", 1)[1],
                    base_commit=d["base_commit"],
                    patch=d["patch"],
                    test_patch=d["test_patch"],
                    problem_statement=d["problem_statement"],
                )
            )
    return tasks


def commits(tasks: list[Task]) -> list[tuple[str, str]]:
    """Distinct (repo, base_commit) pairs, sorted."""
    return sorted({(t.repo, t.base_commit) for t in tasks})


def read_id_list(path: Path) -> list[str]:
    """One id per line; `#` comment and blank lines ignored."""
    lines = (l.strip() for l in Path(path).read_text().splitlines())
    return [l for l in lines if l and not l.startswith("#")]


def _quotas(sizes: dict[str, int], minimums: dict[str, int], n: int) -> dict[str, int]:
    """Proportional quotas summing to n (largest remainder); a repo never gets less than its minimum."""
    quota: dict[str, int] = {}
    while True:
        free = {r: s for r, s in sizes.items() if r not in quota}
        left, total = n - sum(quota.values()), sum(free.values())
        exact = {r: left * s / total for r, s in free.items()} if total else {}
        floor = {r: int(v) for r, v in exact.items()}
        extra = left - sum(floor.values())
        for r in sorted(free, key=lambda r: (-(exact[r] - floor[r]), r))[:extra]:
            floor[r] += 1
        over = {r: minimums[r] for r in free if r in minimums and floor[r] < minimums[r]}
        if not over:
            return {**quota, **floor}
        quota.update(over)


def draw_subset(tasks: list[Task], n: int, seed: int, include: list[str]) -> list[str]:
    """n ids stratified by repo_short, proportional to `tasks`, containing every id in `include`."""
    by_repo: dict[str, list[str]] = {}
    for t in tasks:
        by_repo.setdefault(t.repo_short, []).append(t.instance_id)
    known = {i: r for r, ids in by_repo.items() for i in ids}
    unknown = sorted(set(include) - set(known))
    if unknown:
        raise ValueError(f"include ids not in tasks: {unknown}")
    if not 0 < n <= len(known) or len(set(include)) > n:
        raise ValueError(f"cannot draw {n} ids with {len(set(include))} included")
    kept: dict[str, set[str]] = {r: set() for r in by_repo}
    for i in include:
        kept[known[i]].add(i)
    fixed = {r: len(s) for r, s in kept.items() if s}
    quota = _quotas({r: len(ids) for r, ids in by_repo.items()}, fixed, n)
    rng = random.Random(seed)
    chosen: list[str] = []
    for r in sorted(by_repo):
        rest = sorted(set(by_repo[r]) - kept[r])
        chosen += sorted(kept[r]) + rng.sample(rest, quota[r] - len(kept[r]))
    return sorted(chosen)
