"""Per-task summaries of harness runs.

Field paths relied on (ATIF v1.7 trace, `traces/trace_<instance_id>.json`):
- `steps[].tool_calls[].function_name` / `.arguments` (agent steps; a step may carry
  several parallel calls). Path argument key is `filepath`.
- `steps[].observation.content`: JSON string of the tool result with `status`
  ("ok" or "error") and, on failure, `error_type` / `error_message` or `error`.
  `steps[].observation.extra.tool_name` names the tool. When a step issues parallel
  calls the harness keeps only ONE observation for the step, so graph_tool_errors
  is a lower bound for steps with parallel graph calls.
- `final_metrics.total_prompt_tokens` / `total_completion_tokens`; fallback is the
  sum of `steps[].metrics.prompt_tokens` / `completion_tokens`; else None.
Run-level: `run_manifest.json` (`run`, optional `condition`, `seed`) and
`task_results.jsonl` (`instance_id`, `resolved`, `duration_seconds`).

search_similar_code calls (`arguments.query`, issued by the code_analyzer sub-agent) are
classified against the graph of the task's commit: a query the harness resolves to a graph node
is scored with that node's stored embedding; any other query falls back to lexical token
scoring and never touches the embeddings (swegemma 0.2.11 retrieval_utils.get_similar_nodes).
"""
import csv
import functools
import importlib
import importlib.metadata
import json
import logging
import re
import sys
from pathlib import Path

from graphswap.tasks import Task

logger = logging.getLogger(__name__)

SEARCH_TOOL = "search_similar_code"
GRAPH_TOOLS = {SEARCH_TOOL, "get_code_neighbors", "get_code_subgraph"}
READ_TOOLS = {"read_file"}
EDIT_TOOLS = {"edit_file", "write_file"}
COLUMNS = [
    "instance_id", "run", "condition", "seed", "resolved", "patch_submitted",
    "tool_calls", "graph_tool_calls", "graph_tool_errors", "analyzer_calls",
    "gold_file_read", "gold_file_edited", "wall_s", "prompt_tokens", "completion_tokens",
    "search_calls", "search_resolved", "search_lexical", "search_substring",
]
SEARCH_COLUMNS = COLUMNS[-4:]
HARNESS_VERSION = "0.2.11"  # the harness the study ran; resolve_query mirrors its line numbers
VENDOR = Path(__file__).resolve().parent.parent / "vendor"


def _import_graph():
    try:
        return importlib.import_module("swegemma.graph")
    except ModuleNotFoundError as exc:
        if exc.name == "cachetools":
            raise ModuleNotFoundError(
                "swegemma.graph needs cachetools: install the dev group (uv sync)", name="cachetools") from exc
        raise


def _check_harness(pkg) -> None:
    """Raise unless `pkg` (the imported `swegemma`) is the vendored copy or an install of 0.2.11."""
    origin = Path(pkg.__file__).resolve().parent
    if origin.is_relative_to(VENDOR.resolve()):
        return  # `make vendor` extracts it from the pinned 0.2.11 wheel
    try:
        version = importlib.metadata.version("swegemma")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != HARNESS_VERSION:
        raise RuntimeError(
            f"swegemma {version or '(unknown version)'} at {origin} is imported ahead of {VENDOR}; "
            f"graphswap mirrors swegemma {HARNESS_VERSION}. Uninstall it or install {HARNESS_VERSION}.")


@functools.cache
def harness_graph():
    """`swegemma.graph`: an installed swegemma 0.2.11, else the vendored copy (git-ignored
    `vendor/`, filled by `make vendor`)."""
    try:
        graph = _import_graph()
    except ModuleNotFoundError as exc:
        if exc.name not in ("swegemma", "swegemma.graph"):
            raise
        if not (VENDOR / "swegemma" / "graph").is_dir():
            raise ModuleNotFoundError(
                f"swegemma.graph is not importable and {VENDOR} has no copy; run `make vendor`") from None
        sys.path.insert(0, str(VENDOR))
        graph = _import_graph()
    _check_harness(sys.modules["swegemma"])
    return graph


def graph_node_names(path: Path) -> list[str]:
    """Node names in the order the harness scores them.

    Loaded with the harness loader (graph_utils.load_graph_from_json); a node's name is its
    `name` attribute, defaulting to the node id (graph_utils.get_graph lines 302-304,
    retrieval_utils._extract_nodes_and_embeddings lines 181-184, 211).
    """
    graph = harness_graph().load_graph_from_json(str(path))
    names = []
    for n, data in graph.nodes(data=True):
        name = data.get("name", str(n))
        names.append(name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name))
    return names


def resolve_query(query, names: list[str]) -> str | None:
    """The graph node whose stored embedding search_similar_code(query) scores against, or None
    when the harness falls back to lexical matching.

    Mirrors retrieval_utils.get_similar_nodes for a string query with no node content (the tool
    passes `node=query`, tools/graph.py line 243): a `repo;` prefix is dropped (lines 569-574,
    602-605), graph_utils.resolve_node_name runs only for queries without whitespace (lines
    775-783), and the result must be a node name (line 785). An unresolved query still gets one
    lookup in the embeddings file (lines 806-822 -> embedding_utils.embed, lines 388-435). When
    the file's keys are exactly the node ids, as in the released and rebuilt bundles, that lookup
    can only hit a query equal to a node's full text; replay.py measures it (`unresolved_changed`).
    """
    node = _node_name(query)
    if not node:  # the harness raises ValueError -> SimilaritySearchError (lines 613-617)
        return None
    if node not in names and not any(c.isspace() for c in node):
        resolved = harness_graph().resolve_node_name(node, names)
        if resolved is not None and resolved in names:
            node = resolved
    return node if node in names else None


def _node_name(query) -> str | None:
    """The name get_similar_nodes looks up: a `repo;` prefix dropped twice (lines 569-574, 602-605)."""
    if isinstance(query, bytes):
        query = query.decode("utf-8", errors="replace")
    if not isinstance(query, str):
        return None
    node = query.split(";", 1)[1] if ";" in query else query
    return node.split(";", 1)[1] if ";" in node else node


def resolution_tier(query, resolved: str | None) -> str:
    """Which resolve_node_name tier produced `resolved`: exact, suffix, case-insensitive or
    substring ("" when unresolved). The tiers run in that order (graph_utils.py lines 378-431) and
    each returns on its first hit, so the hit's form identifies its tier. A substring hit can be
    any node containing the query (e.g. a docs example for a parameter name)."""
    if resolved is None:
        return ""
    node = _node_name(query)
    bare = node.split(";", 1)[1] if ";" in node else node  # resolve_node_name line 364
    if resolved in (node, bare):
        return "exact"
    if any(resolved.endswith(d + bare) for d in ".;/"):
        return "suffix"
    low, full = resolved.lower(), (node.lower(), bare.lower())
    if low in full or any(low.endswith(d + bare.lower()) for d in ".;/"):
        return "case-insensitive"
    return "substring"


def call_arguments(call: dict) -> dict:
    """A tool call's `arguments`, or {} when the trace records none: null or any non-object is a
    call with no arguments (it names no file and no query), not a reason to drop the task row."""
    arguments = call.get("arguments")
    return arguments if isinstance(arguments, dict) else {}


def search_calls(trace: dict) -> list[tuple[int, dict]]:
    """(step index, call) for every search_similar_code call, in trace order."""
    return [(i, c) for i, s in enumerate(trace["steps"])
            for c in s.get("tool_calls") or [] if c["function_name"] == SEARCH_TOOL]


def search_summary(trace: dict, names: list[str] | None) -> dict:
    """search_calls, plus resolved/lexical/substring counts when the graph's node names are given.
    `search_resolved` includes the substring hits counted again in `search_substring`."""
    queries = [call_arguments(c).get("query") for _, c in search_calls(trace)]
    if names is None:
        return {"search_calls": len(queries), "search_resolved": "", "search_lexical": "",
                "search_substring": ""}
    tiers = [resolution_tier(q, resolve_query(q, names)) for q in queries]
    resolved = sum(t != "" for t in tiers)
    return {"search_calls": len(queries), "search_resolved": resolved,
            "search_lexical": len(queries) - resolved, "search_substring": tiers.count("substring")}


def gold_files(patch: str) -> set[str]:
    """Paths modified by a unified diff (the b/ side; /dev/null deletions use the a/ side)."""
    paths = set()
    for m in re.finditer(r"^--- (?:a/(\S+)|/dev/null)\n\+\+\+ (?:b/(\S+)|/dev/null)$", patch, re.M):
        paths.add(m.group(2) or m.group(1))
    return paths


def _norm(path) -> str:
    path = str(path)
    return path[len("/workspace/"):] if path.startswith("/workspace/") else path


def _is_failure(content: str) -> bool:
    try:
        j = json.loads(content)
    except (TypeError, ValueError):
        return False
    return isinstance(j, dict) and (j.get("status", "ok") != "ok" or "error" in j)


def _tokens(trace: dict) -> tuple[int | None, int | None]:
    fm = trace.get("final_metrics") or {}
    if "total_prompt_tokens" in fm and "total_completion_tokens" in fm:
        return fm["total_prompt_tokens"], fm["total_completion_tokens"]
    ms = [s["metrics"] for s in trace["steps"] if s.get("metrics")]
    if not ms:
        return None, None
    return (sum(m.get("prompt_tokens", 0) for m in ms),
            sum(m.get("completion_tokens", 0) for m in ms))


def summarize_trace(trace: dict, gold: set[str]) -> dict:
    calls = [c for s in trace["steps"] for c in s.get("tool_calls") or []]
    names = [c["function_name"] for c in calls]

    def touched(tools):
        return any(
            c["function_name"] in tools and _norm(call_arguments(c).get("filepath", "")) in gold
            for c in calls
        )

    graph_errors = sum(
        1
        for s in trace["steps"]
        if s.get("observation")
        and (s["observation"].get("extra") or {}).get("tool_name") in GRAPH_TOOLS
        and _is_failure(s["observation"].get("content"))
    )
    prompt, completion = _tokens(trace)
    return {
        "patch_submitted": "submit_patch" in names,
        "tool_calls": len(names),
        "graph_tool_calls": sum(n in GRAPH_TOOLS for n in names),
        "graph_tool_errors": graph_errors,
        "analyzer_calls": names.count("code_analyzer"),
        "gold_file_read": touched(READ_TOOLS),
        "gold_file_edited": touched(EDIT_TOOLS),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
    }


def _names_for(task: Task, graphs_dir: Path, cache: dict) -> list[str] | None:
    path = graphs_dir / f"{task.repo_short}_{task.base_commit}.json"
    if path not in cache:
        if path.exists():
            cache[path] = graph_node_names(path)
        else:
            logger.warning("%s: no graph at %s; search resolution left blank", task.instance_id, path)
            cache[path] = None
    return cache[path]


def summarize_run(results_dir: Path, tasks: list[Task], *, graphs_dir: Path | None = None) -> list[dict]:
    """One row per task in `task_results.jsonl`. With `graphs_dir` (`<repo_short>_<commit>.json`
    files), search_similar_code calls are classified as resolved or lexical (and substring
    resolutions counted); without it the four search columns are empty strings."""
    results_dir = Path(results_dir)
    manifest = json.loads((results_dir / "run_manifest.json").read_text())
    gold = {t.instance_id: gold_files(t.patch) for t in tasks}
    by_id = {t.instance_id: t for t in tasks}
    names_cache: dict[Path, list[str] | None] = {}
    rows = []
    with open(results_dir / "task_results.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            res = json.loads(line)
            iid = res["instance_id"]
            trace_path = results_dir / "traces" / f"trace_{iid}.json"
            trace = json.loads(trace_path.read_text()) if trace_path.exists() else {"steps": []}
            if graphs_dir is None:
                search = dict.fromkeys(SEARCH_COLUMNS, "")
            else:  # a task without search calls needs no graph
                names = _names_for(by_id[iid], Path(graphs_dir), names_cache) if search_calls(trace) else []
                search = search_summary(trace, names)
            rows.append({
                "instance_id": iid,
                "run": manifest.get("run", ""),
                "condition": manifest.get("condition", ""),
                "seed": manifest.get("seed"),
                "resolved": bool(res["resolved"]),
                "wall_s": res.get("duration_seconds"),
                **summarize_trace(trace, gold[iid]),
                **search,
            })
    return [{k: r[k] for k in COLUMNS} for r in rows]


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
