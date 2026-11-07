#!/usr/bin/env python3
"""Compare two local eval runs task-by-task.

The aggregate resolution rate is the number the leaderboard shows you, and it is
the number least able to tell you what happened. A change that fixes two tasks
and breaks two others reports as "no effect". This prints the per-task flips so
that case is visible.

Usage:
    python3 tools/flips.py results/base results/candidate
    python3 tools/flips.py results/base/task_results.jsonl results/cand/task_results.jsonl
"""

import json
import pathlib
import sys

ID_FIELDS = ("instance_id", "task_id", "instance", "id")
RESOLVED_FIELDS = ("resolved", "is_resolved", "success", "passed", "solved")


def _pick(record, candidates, kind, path):
    for field in candidates:
        if field in record:
            return field
    raise SystemExit(
        f"{path}: could not find a {kind} field.\n"
        f"  looked for: {', '.join(candidates)}\n"
        f"  record has: {', '.join(sorted(record))}\n"
        f"Edit {kind.upper()}_FIELDS in tools/flips.py to match the harness output."
    )


def _truthy(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "resolved", "pass", "passed"}
    return False


def load(path):
    path = pathlib.Path(path)
    if path.is_dir():
        path = path / "task_results.jsonl"
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    results = {}
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: bad JSON ({exc})")
            id_field = _pick(record, ID_FIELDS, "id", path)
            resolved_field = _pick(record, RESOLVED_FIELDS, "resolved", path)
            results[record[id_field]] = _truthy(record[resolved_field])
    if not results:
        raise SystemExit(f"{path}: no task records found")
    return results, path


def manifest_for(arg):
    path = pathlib.Path(arg)
    path = path if path.is_dir() else path.parent
    candidate = path / "run_manifest.json"
    return json.loads(candidate.read_text()) if candidate.exists() else None


# What a condition (conditions/C*.yaml) resolved to on Kaggle. Under the same
# condition these must match; if one moved, the condition's inputs changed between runs.
CONDITION_INPUTS = ("graph_bundle_sha", "embeddings_bundle_sha", "analyzer_tools", "free_text_embed")


def _show(value):
    if value is None:
        return "(not recorded)"
    if isinstance(value, list):
        return "[" + ", ".join(map(str, value)) + "]"
    if isinstance(value, str) and len(value) == 64:
        return value[:12]          # a sha
    return str(value)


def coverage_text(m):
    """'covered/requested' from a manifest's graph_coverage, or None if not recorded."""
    cov = (m or {}).get("graph_coverage")
    if not isinstance(cov, dict) or "tasks" not in cov or "with_graph_tools" not in cov:
        return None
    return f"{cov['with_graph_tools']}/{cov['tasks']}"


def coverage_lines(a, b):
    """How many tasks each run actually offered the graph tools. Under one condition a
    change is a confound; across conditions that both have graph data it means part of
    the delta is coverage (tasks with vs without graph tools), not the data itself."""
    ca, cb = coverage_text(a), coverage_text(b)
    if ca == cb:
        return [], False
    lines = [f"graph_coverage: {ca or '(not recorded)'} != {cb or '(not recorded)'} tasks offered the graph tools"]
    same_condition = a.get("condition") == b.get("condition")
    no_graph = "none" in (a.get("graph_bundle_sha"), b.get("graph_bundle_sha"),
                          a.get("embeddings_bundle_sha"), b.get("embeddings_bundle_sha"))
    if not same_condition and not no_graph and ca and cb:
        lines.append("WARNING: graph coverage differs -- the delta mixes the graph data with which tasks got "
                     "graph tools at all; compare on the tasks covered in both runs too")
    return lines, same_condition


def compare_manifests(a, b):
    """Lines naming every way the two runs differ. A clean comparison differs in one thing.

    Manifests from before conditions existed have no condition fields; two of those
    compare as they always did, but one is never assumed to equal a C1 run.
    """
    if a is None or b is None:
        return ["(no run_manifest.json on one side -- cannot check the control)"]
    def norm(m, key):
        value = m.get(key)
        # analyzer_tools order is the declared tool order, not a set; keep it
        return sorted(value) if isinstance(value, list) and key != "analyzer_tools" else value
    def differs(key):
        return norm(a, key) != norm(b, key)
    lines = []
    for label, m in (("A", a), ("B", b)):
        cond = f"  condition {m['condition']}  seed {m.get('seed')}" if m.get("condition") is not None else ""
        lines.append(f"{label}  run {m.get('run')}  git {m.get('git_sha')}  "
                     f"submission {str(m.get('tree_hash'))[:12]}{cond}")
    for key in ("condition", "seed") + CONDITION_INPUTS:
        if differs(key):
            lines.append(f"{key}: {_show(a.get(key))} != {_show(b.get(key))}")
    cov_lines, cov_confound = coverage_lines(a, b)
    lines += cov_lines
    confounds = [k for k in ("task_ids_requested", "concurrency", "versions") if differs(k)]
    variables = [k for k in ("condition", "seed", "tree_hash", "model", "budgets") if differs(k)]
    if "condition" not in variables:
        confounds += [k for k in CONDITION_INPUTS if differs(k)]
        if cov_confound:
            confounds.append("graph_coverage")
    if not variables and not confounds:
        lines.append("same submission, same setup: this comparison measures the NOISE FLOOR")
    elif variables == ["seed"] and not confounds:
        lines.append("same submission and condition, different seed: this is a SEED PAIR (sampling variance)")
    else:
        if variables:
            lines.append(f"differs in: {', '.join(variables)}")
        for key in confounds:
            lines.append(f"WARNING: runs differ in {key} -- not a controlled comparison; any delta is confounded")
    if a.get("status") != "complete" or b.get("status") != "complete":
        lines.append("WARNING: at least one run did not complete")
    return lines


def control_check(a, b):
    print("\n".join(compare_manifests(a, b)) + "\n")


# Error strings the harness records when the AGENT runs out of budget or misbehaves
# (swegemma/harness/agent_runner.py, verification.py). The scorer gives these tasks
# 0 too, so they are legitimate results. Anything else means the run itself broke.
AGENT_OUTCOME_PREFIXES = (
    "Agent exceeded session timeout",
    "Agent exceeded tool call budget",
    "Agent exceeded turns budget",
    "Agent exceeded maximum allowed LLM turns",
    "Agent completed execution without calling submit_patch",
    "Failed to apply agent patch",
)


def is_infra_error(message):
    return bool(message) and not str(message).startswith(AGENT_OUTCOME_PREFIXES)


def errored_tasks(arg):
    """Tasks whose error means the run broke (not the agent) -- unsafe to compare."""
    path = pathlib.Path(arg)
    path = path / "task_results.jsonl" if path.is_dir() else path
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    return [r.get("instance_id") for r in rows if is_infra_error(r.get("error"))]


def rate(results):
    return sum(results.values()) / len(results)


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)

    before, before_path = load(argv[1])
    after, after_path = load(argv[2])

    control_check(manifest_for(argv[1]), manifest_for(argv[2]))
    for label, path in (("A", argv[1]), ("B", argv[2])):
        errored = errored_tasks(path)
        if errored:
            print(f"WARNING: {len(errored)} tasks hit INFRASTRUCTURE errors in {label} -- they score 0 for "
                  f"reasons unrelated to the agent, so flips on them are not evidence: {errored[:5]}")

    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))
    shared = sorted(set(before) & set(after))

    if not shared:
        raise SystemExit("the two runs share no task ids -- nothing to compare")

    fixed = [t for t in shared if not before[t] and after[t]]
    broke = [t for t in shared if before[t] and not after[t]]
    stable_pass = [t for t in shared if before[t] and after[t]]

    print(f"A  {before_path}  {rate(before):.4f}  ({sum(before.values())}/{len(before)})")
    print(f"B  {after_path}  {rate(after):.4f}  ({sum(after.values())}/{len(after)})")
    print()

    if only_before or only_after:
        print(f"!  task sets differ: {len(only_before)} only in A, {len(only_after)} only in B")
        print(f"!  comparing the {len(shared)} shared tasks only")
        print()

    net = len(fixed) - len(broke)
    granularity = 1 / len(shared)
    print(f"fixed  (A fail -> B pass): {len(fixed)}")
    print(f"broke  (A pass -> B fail): {len(broke)}")
    print(f"net                      : {net:+d}  ({net * granularity:+.4f})")
    print(f"churn  (total flips)     : {len(fixed) + len(broke)}")
    print(f"stable pass              : {len(stable_pass)}")
    print(f"one task is worth        : {granularity:.4f}")
    print()

    for label, tasks in (("fixed", fixed), ("broke", broke)):
        if tasks:
            print(f"{label}:")
            for task in tasks:
                print(f"  {task}")
            print()

    churn = len(fixed) + len(broke)
    if churn and abs(net) < churn:
        print(
            f"NOTE: {churn} tasks moved but the net is only {net:+d}. The aggregate is\n"
            f"      hiding offsetting changes -- read the task lists above, not the rate."
        )
    if churn == 0:
        print(
            "NOTE: zero tasks flipped. Before calling this a null result, confirm the\n"
            "      change was actually in effect (resolved config, loaded paths, counts)."
        )


if __name__ == "__main__":
    main(sys.argv)
