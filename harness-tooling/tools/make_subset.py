#!/usr/bin/env python3
"""Draw a fixed, repo-stratified task subset from tasks.jsonl.

A subset is part of an experiment's control: every run you compare must use the
same one. Generate it once, commit the file, and leave it alone. Regenerating
with a different N or seed silently invalidates every earlier comparison.

The training set is heavily skewed (fastapi 67, rich 48, requests 13, httpx 1)
while the hidden test set comes from private repos, so the allocation guarantees
every repo at least one task rather than mirroring training proportions.

Usage:
    python3 tools/make_subset.py data/tasks.jsonl --n 24 --seed 0 > eval/subset.txt
    python3 tools/make_subset.py data/tasks.jsonl --smoke > eval/smoke.txt
"""

import argparse
import collections
import json
import random
import sys


def load(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def allocate(counts, n):
    """Largest-remainder allocation with a floor of one task per repo."""
    repos = sorted(counts)
    if n < len(repos):
        raise SystemExit(f"--n {n} is smaller than the {len(repos)} repos; need at least one each")
    alloc = {repo: 1 for repo in repos}
    remaining = n - len(repos)
    total = sum(counts.values())
    shares = {repo: remaining * counts[repo] / total for repo in repos}
    for repo in repos:
        alloc[repo] += min(int(shares[repo]), counts[repo] - 1)
    leftover = n - sum(alloc.values())
    by_remainder = sorted(repos, key=lambda r: (shares[r] - int(shares[r]), counts[r]), reverse=True)
    while leftover > 0:
        progressed = False
        for repo in by_remainder:
            if leftover and alloc[repo] < counts[repo]:
                alloc[repo] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            raise SystemExit(f"--n {n} exceeds the number of available tasks")
    return alloc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="two tasks with the smallest reference patches, from the two largest repos")
    args = parser.parse_args()

    tasks = load(args.tasks)
    by_repo = collections.defaultdict(list)
    for task in tasks:
        by_repo[task["repo"]].append(task)

    if args.smoke:
        largest = sorted(by_repo, key=lambda r: len(by_repo[r]), reverse=True)[:2]
        picked = [min(by_repo[r], key=lambda t: (len(t.get("patch") or ""), t["instance_id"])) for r in largest]
        header = "# smoke subset: smallest reference patch from each of the two largest repos"
    else:
        rng = random.Random(args.seed)
        alloc = allocate({r: len(v) for r, v in by_repo.items()}, args.n)
        picked = []
        for repo in sorted(alloc):
            pool = sorted(by_repo[repo], key=lambda t: t["instance_id"])
            picked.extend(rng.sample(pool, alloc[repo]))
        mix = ", ".join(f"{r.split('/')[-1]} {alloc[r]}" for r in sorted(alloc))
        header = f"# n={args.n} seed={args.seed} stratified by repo ({mix})"

    print(header)
    print("# Fixed control: do not regenerate. Every compared run must use this file.")
    for task in sorted(picked, key=lambda t: t["instance_id"]):
        print(task["instance_id"])


if __name__ == "__main__":
    sys.exit(main())
