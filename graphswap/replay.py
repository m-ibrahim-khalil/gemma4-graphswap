"""Offline replay of search_similar_code calls under two embedding sets for the same graphs.

Each search_similar_code call in a harness trace is re-run through the harness's own retrieval
(swegemma 0.2.11 `graph.get_similar_nodes`, with the arguments tools/graph.py lines 237-259
pass), once with embedding set A and once with set B. A query that resolves to a graph node
(traces.resolve_query) is scored by cosine between that node's stored vector and every other
node's (retrieval_utils.py lines 872-896); for those calls the two top-k lists are compared.
Any other query is scored lexically (lines 824-863), which never reads the embeddings, so its
lists must agree; `unresolved_changed` counts the calls where they do not.

overlap = |topA & topB| / min(k, n_nodes - 1): the harness returns exactly that many nodes for
a resolved query, the query node itself excluded (lines 884-890); it is k for every real graph.

`observed` checks the replay against the trace: which set's top-k equals the node list the run
actually recorded for the call (the harness keeps one observation per step, so many parallel
calls have none). The summary splits that check by path: lexical lists match both sets by
construction, so only `observed_a_embed` / `observed_b_embed` test the embedding scoring.
Rows carry query and node names only, never node text or code.
"""
import json
from pathlib import Path

from graphswap.tasks import Task
from graphswap.traces import (SEARCH_TOOL, graph_node_names, harness_graph, resolution_tier,
                              resolve_query, search_calls)

DEFAULT_K = 10  # tools/graph.py lines 201 and 235; get_similar_nodes default, retrieval_utils.py line 514
ROW_COLUMNS = ["run", "instance_id", "step", "query", "k", "resolved_node", "tier", "overlap",
               "top1_same", "same_list", "observed"]
SUMMARY_COLUMNS = ["run", "unit", "n_calls", "n_resolved", "n_substring", "mean_overlap",
                   "frac_top1_same", "frac_same_list", "unresolved_changed", "n_observed_embed",
                   "observed_a_embed", "observed_b_embed", "n_observed_lexical", "observed_lexical"]
SUMMARY_COMMENT = ("# overlap = |topA & topB| / min(k, n_nodes - 1) over embedding-path calls "
                   "(n_resolved); unit=call counts every call, unit=query the first call per "
                   "(run, instance_id, query); observed_*_embed: embedding-path calls whose recorded "
                   "result list set A / B reproduces")
_OBSERVED = ("A", "B", "AB", "none")


def coerce_k(val, default: int = DEFAULT_K) -> int:
    """The tool's `k` coercion: tools/graph.py _coerce_int (lines 20-46), as called on line 235."""
    if val is None:
        return default
    if isinstance(val, bool):
        raise TypeError(f"k must be an integer, got boolean {val}")
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() in ("none", "null", "nil"):
            return default
        try:
            return int(s)
        except ValueError:
            try:
                f = float(s)
                if f.is_integer():
                    return int(f)
            except ValueError:
                pass
            raise ValueError(f"Invalid integer for k: {val!r}") from None
    raise ValueError(f"Invalid integer for k: {val!r}")


def harness_top_k(task: Task, query, k: int, graph_dir: Path, emb_dir: Path) -> list[str] | None:
    """Node names search_similar_code returns for `query`, or None where the tool returns an error."""
    sg = harness_graph()
    base_commit = task.base_commit
    if base_commit and task.repo.endswith(f"_{base_commit}"):  # tools/graph.py lines 237-241
        base_commit = None
    try:
        nodes = sg.get_similar_nodes(
            node=query, repo_name=task.repo, k=k,
            graphs_sstable_path=str(graph_dir), embeddings_sstable_path=str(emb_dir),
            use_local_storage=True, graph_dir=str(graph_dir), embeddings_dir=str(emb_dir),
            base_commit=base_commit,
        )
    except ValueError:  # k <= 0 or an empty name: the tool answers SimilaritySearchError (line 284)
        return None
    return [n["node_name"] for n in nodes]


def _recorded(trace: dict, calls: list[tuple[int, dict]]) -> list:
    """Per call: the node names its recorded result listed, "error" for a recorded error, or None.

    A step keeps one observation; an ok result is matched to the step's calls by its echoed
    `query` (several matches only when their arguments are identical), an error only when the
    step has a single search call.
    """
    out = [None] * len(calls)
    by_step: dict[int, list[int]] = {}
    for pos, (step, _) in enumerate(calls):
        by_step.setdefault(step, []).append(pos)
    for step, positions in by_step.items():
        obs = trace["steps"][step].get("observation") or {}
        if (obs.get("extra") or {}).get("tool_name") != SEARCH_TOOL:
            continue
        try:
            content = json.loads(obs.get("content"))
        except (TypeError, ValueError):
            continue
        if not isinstance(content, dict):
            continue
        if content.get("status") == "ok":
            args = [calls[p][1]["arguments"] for p in positions]
            matched = [p for p, a in zip(positions, args) if a.get("query") == content.get("query")]
            if matched and all(calls[p][1]["arguments"] == calls[matched[0]][1]["arguments"] for p in matched):
                names = [r.get("node_name") for r in content.get("results") or []]
                for p in matched:
                    out[p] = names
        elif len(positions) == 1:
            out[positions[0]] = "error"
    return out


def _match(recorded, a: list[str] | None, b: list[str] | None) -> str:
    if recorded is None or recorded == "error":
        return recorded or ""
    label = ("A" if recorded == a else "") + ("B" if recorded == b else "")
    return label or "none"


def replay_run(results_dir: Path, tasks: list[Task], graph_dir: Path, emb_a: Path, emb_b: Path,
               k: int = DEFAULT_K) -> list[dict]:
    """One row per search_similar_code call (in trace order) of the run's traces.

    `k` applies to calls that pass none; a call's own `k` is coerced as the tool does. Graphs and
    embeddings are `<repo_short>_<commit>.json` / `.npz` in the given directories; a missing file
    raises (the harness would silently score against zero vectors).
    """
    results_dir, graph_dir, emb_a, emb_b = map(Path, (results_dir, graph_dir, emb_a, emb_b))
    sg = harness_graph()
    run = json.loads((results_dir / "run_manifest.json").read_text()).get("run", results_dir.name)
    by_id = {t.instance_id: t for t in tasks}
    rows = []
    with open(results_dir / "task_results.jsonl") as f:
        ids = [json.loads(line)["instance_id"] for line in f if line.strip()]
    for iid in ids:
        trace_path = results_dir / "traces" / f"trace_{iid}.json"
        if not trace_path.exists():
            continue
        trace = json.loads(trace_path.read_text())
        calls = search_calls(trace)
        if not calls:
            continue
        task = by_id[iid]
        stem = f"{task.repo_short}_{task.base_commit}"
        for p in (graph_dir / f"{stem}.json", emb_a / f"{stem}.npz", emb_b / f"{stem}.npz"):
            if not p.exists():
                raise FileNotFoundError(f"{run}/{iid}: missing {p}")
        names = graph_node_names(graph_dir / f"{stem}.json")
        ks = []
        for _, call in calls:
            try:
                ks.append(coerce_k(call["arguments"].get("k"), k))
            except (TypeError, ValueError):  # the tool answers SimilaritySearchError
                ks.append(None)
        tops = {}
        for label, emb_dir in (("A", emb_a), ("B", emb_b)):
            sg.clear_caches()  # the harness memoizes per process; score each set from a cold start
            tops[label] = [None if kk is None else
                           harness_top_k(task, call["arguments"].get("query"), kk, graph_dir, emb_dir)
                           for (_, call), kk in zip(calls, ks)]
        sg.clear_caches()
        recorded = _recorded(trace, calls)
        for pos, ((step, call), kk) in enumerate(zip(calls, ks)):
            query = call["arguments"].get("query")
            node = resolve_query(query, names)
            a, b = tops["A"][pos], tops["B"][pos]
            row = {"run": run, "instance_id": iid, "step": step, "query": query,
                   "k": call["arguments"].get("k") if kk is None else kk,
                   "resolved_node": node or "", "tier": resolution_tier(query, node),
                   "overlap": "", "top1_same": "",
                   "same_list": "" if a is None or b is None else a == b,
                   "observed": _match(recorded[pos], a, b)}
            if node is not None and a and b:
                row["overlap"] = len(set(a) & set(b)) / min(kk, len(names) - 1)
                row["top1_same"] = a[0] == b[0]
            rows.append(row)
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    """The first row per (run, instance_id, query, k).

    ``k`` is part of the key because the same query at another k is a different retrieval:
    it asks for a different number of neighbours and gets a different overlap.
    """
    seen, out = set(), []
    for r in rows:
        key = (r["run"], r["instance_id"], str(r["query"]), str(r["k"]))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def summarize_replay(rows: list[dict], unit: str = "call") -> list[dict]:
    """One row per run (runs with no search calls have no rows and are absent). `unit` labels
    what a row of `rows` is: "call", or "query" for `dedupe(rows)`."""
    out = []
    for run in dict.fromkeys(r["run"] for r in rows):
        rs = [r for r in rows if r["run"] == run]
        scored = [r for r in rs if r["overlap"] != ""]
        embed = [r for r in rs if r["resolved_node"] != "" and r["observed"] in _OBSERVED]
        lexical = [r for r in rs if r["resolved_node"] == "" and r["observed"] in _OBSERVED]
        out.append({
            "run": run,
            "unit": unit,
            "n_calls": len(rs),
            "n_resolved": sum(r["resolved_node"] != "" for r in rs),
            "n_substring": sum(r["tier"] == "substring" for r in rs),
            "mean_overlap": sum(r["overlap"] for r in scored) / len(scored) if scored else "",
            "frac_top1_same": sum(r["top1_same"] for r in scored) / len(scored) if scored else "",
            "frac_same_list": sum(r["same_list"] is True for r in scored) / len(scored) if scored else "",
            "unresolved_changed": sum(r["resolved_node"] == "" and r["same_list"] is False for r in rs),
            "n_observed_embed": len(embed),
            "observed_a_embed": sum(r["observed"] in ("A", "AB") for r in embed),
            "observed_b_embed": sum(r["observed"] in ("B", "AB") for r in embed),
            "n_observed_lexical": len(lexical),
            "observed_lexical": sum(r["observed"] != "none" for r in lexical),
        })
    return out
