import csv
import json
import os
from pathlib import Path

import pytest

from graphswap.tasks import Task
from graphswap.traces import resolution_tier, resolve_query, summarize_run, write_csv

FIXTURE = Path(__file__).parent / "fixtures" / "traces"
GRAPHS = Path(__file__).parent / "fixtures" / "graphs"  # r_abc.json: 3 fake nodes
BASELINE = os.environ.get("GRAPHSWAP_BASELINE_RESULTS")  # a local run directory, if you have one
TASKS_FILE = Path(__file__).parent.parent / "data" / "tasks.jsonl"


def _task(iid, gold_path):
    patch = f"diff --git a/{gold_path} b/{gold_path}\n--- a/{gold_path}\n+++ b/{gold_path}\n@@ -1 +1 @@\n-a\n+b\n"
    return Task(iid, "o/r", "r", "abc", patch, "", "")


def _rows():
    tasks = [_task("widgetlib_101", "src/widgetlib/frame.py"), _task("gearbox_202", "gearbox/painter.py")]
    return {r["instance_id"]: r for r in summarize_run(FIXTURE, tasks)}


def test_timeout_trace_not_resolved_not_submitted():
    r = _rows()["widgetlib_101"]
    assert r["resolved"] is False
    assert r["patch_submitted"] is False  # the run stops without submit_patch
    assert r["tool_calls"] == 18
    assert r["prompt_tokens"] == 104560 and r["completion_tokens"] == 8130
    assert r["wall_s"] == pytest.approx(274.5, abs=0.01)
    assert r["analyzer_calls"] == 1
    assert r["graph_tool_calls"] == 6  # 2 search + 3 neighbors + 1 subgraph
    assert r["graph_tool_errors"] == 1  # the one graph failure whose payload is short enough to parse
    assert r["gold_file_read"] is True  # only via the sandbox's /workspace/ prefix
    assert r["gold_file_edited"] is True  # via write_file: this run edits, it just never submits


def test_final_metrics_win_over_the_per_step_sums():
    """The fixture's final_metrics disagree with the step metrics (sub-agent contexts reset),
    so the row can only match final_metrics."""
    trace = json.loads((FIXTURE / "traces/trace_widgetlib_101.json").read_text())
    steps = [s["metrics"] for s in trace["steps"] if s.get("metrics")]
    assert sum(m["prompt_tokens"] for m in steps) != trace["final_metrics"]["total_prompt_tokens"]
    assert sum(m["completion_tokens"] for m in steps) != trace["final_metrics"]["total_completion_tokens"]
    r = _rows()["widgetlib_101"]
    assert (r["prompt_tokens"], r["completion_tokens"]) == (104560, 8130)


def test_workspace_prefix_is_stripped_before_the_gold_comparison():
    """The only read of the gold file spells it `/workspace/src/widgetlib/frame.py`."""
    trace = json.loads((FIXTURE / "traces/trace_widgetlib_101.json").read_text())
    paths = [c["arguments"].get("filepath") for s in trace["steps"]
             for c in s.get("tool_calls") or [] if c["function_name"] == "read_file"]
    assert "src/widgetlib/frame.py" not in paths
    assert "/workspace/src/widgetlib/frame.py" in paths
    assert _rows()["widgetlib_101"]["gold_file_read"] is True


def test_unparseable_observations_are_not_failures():
    """Both fixtures trim free text at 200 characters, so the longer tool payloads are cut
    mid-token. An observation that is not JSON must not be read as an error."""
    for name in ("trace_widgetlib_101.json", "trace_gearbox_202.json"):
        trace = json.loads((FIXTURE / "traces" / name).read_text())
        cut = 0
        for s in trace["steps"]:
            content = (s.get("observation") or {}).get("content")
            if content is None:
                continue
            assert len(content) <= 200
            try:
                json.loads(content)
            except ValueError:
                cut += 1
        assert cut > 0, f"{name} no longer exercises the unparseable-observation path"


def test_parallel_graph_calls_make_graph_tool_errors_a_lower_bound():
    """The harness keeps one observation per step however many calls the step issued, so a
    step whose two graph calls both failed is still counted once (traces.py says so)."""
    trace = json.loads((FIXTURE / "traces/trace_widgetlib_101.json").read_text())
    parallel = [s for s in trace["steps"] if len(s.get("tool_calls") or []) > 1]
    (step,) = parallel
    assert [c["function_name"] for c in step["tool_calls"]] == ["get_code_neighbors"] * 2
    assert json.loads(step["observation"]["content"])["status"] == "error"
    r = _rows()["widgetlib_101"]
    assert r["graph_tool_calls"] == 6  # both parallel calls are counted as calls
    assert r["graph_tool_errors"] == 1  # but the single observation is one failure


def test_write_file_is_an_edit_tool():
    """widgetlib_101 rewrites the gold file with write_file and only ever calls edit_file on
    a file that is not gold, so gold_file_edited depends on write_file being an edit tool."""
    trace = json.loads((FIXTURE / "traces/trace_widgetlib_101.json").read_text())
    edits = {(c["function_name"], c["arguments"].get("filepath"))
             for s in trace["steps"] for c in s.get("tool_calls") or []
             if c["function_name"] in ("edit_file", "write_file")}
    assert ("write_file", "src/widgetlib/frame.py") in edits  # the gold file
    assert ("edit_file", "src/widgetlib/theme.py") in edits   # not the gold file
    assert not any(tool == "edit_file" and path == "src/widgetlib/frame.py" for tool, path in edits)
    assert _rows()["widgetlib_101"]["gold_file_edited"] is True


def test_an_edit_without_a_submission_is_not_a_submission():
    """widgetlib_101 edits a file and never submits; gearbox_202 edits and submits. Only
    submit_patch separates them."""
    trace = json.loads((FIXTURE / "traces/trace_widgetlib_101.json").read_text())
    names = [c["function_name"] for s in trace["steps"] for c in s.get("tool_calls") or []]
    assert "edit_file" in names and "write_file" in names and "submit_patch" not in names
    assert _rows()["widgetlib_101"]["gold_file_edited"] is True  # it edited the gold file, too
    assert _rows()["widgetlib_101"]["patch_submitted"] is False


def test_resolved_trace_submits_and_edits_the_gold_file():
    r = _rows()["gearbox_202"]
    assert r["resolved"] is True
    assert r["patch_submitted"] is True
    assert r["graph_tool_calls"] == 1  # a single search_similar_code step
    assert r["gold_file_read"] is True and r["gold_file_edited"] is True


def test_failing_non_graph_calls_are_not_graph_errors():
    """The gearbox trace has two parseable run_command failures and no failing graph call."""
    trace = json.loads((FIXTURE / "traces/trace_gearbox_202.json").read_text())
    failed = [(s["observation"]["extra"]["tool_name"], json.loads(s["observation"]["content"]))
              for s in trace["steps"] if s.get("observation")
              and s["observation"]["content"].startswith('{"status": "error"')]
    assert [tool for tool, _ in failed] == ["run_command", "run_command"]
    assert _rows()["gearbox_202"]["graph_tool_errors"] == 0


@pytest.mark.parametrize("arguments", [None, "src/widgetlib/frame.py", ["filepath"]])
def test_a_call_whose_arguments_are_not_a_dict_has_no_arguments(tmp_path, arguments):
    """A trace step may record `arguments` as null (or as some other non-object); such a call is
    still a call, with no arguments, so it names no file and its row is written, not an
    AttributeError raised."""
    results, tasks = _search_run(tmp_path, [
        [{"function_name": "read_file", "arguments": arguments}],
        [{"function_name": "write_file", "arguments": arguments}],
        [{"function_name": "search_similar_code", "arguments": arguments}],
        [_call("read_file", filepath="toyshop/helpers.py")],  # the gold file, read properly
    ])
    (r,) = summarize_run(results, tasks)
    assert (r["tool_calls"], r["graph_tool_calls"]) == (4, 1)
    assert r["gold_file_read"] is True  # from the well-formed call only
    assert r["gold_file_edited"] is False  # the write names no file


def test_a_search_call_without_arguments_is_unresolved(tmp_path):
    pytest.importorskip("swegemma.graph.graph_utils", reason="run `make vendor`")
    results, tasks = _search_run(tmp_path, [[{"function_name": "search_similar_code", "arguments": None},
                                             _call("search_similar_code", query="Sprocket")]])
    (r,) = summarize_run(results, tasks, graphs_dir=GRAPHS)
    assert (r["search_calls"], r["search_resolved"], r["search_lexical"], r["search_substring"]) == (2, 1, 1, 0)


def test_gold_paths_must_match_exactly():
    tasks = [_task("widgetlib_101", "widgetlib/frame.py"), _task("gearbox_202", "gearbox/other.py")]
    rows = {r["instance_id"]: r for r in summarize_run(FIXTURE, tasks)}
    assert rows["gearbox_202"]["gold_file_read"] is False
    assert rows["gearbox_202"]["gold_file_edited"] is False
    assert rows["widgetlib_101"]["gold_file_read"] is True  # the read the sandbox rejected still counts


def test_rows_carry_manifest_fields():
    r = _rows()["gearbox_202"]
    assert (r["run"], r["condition"], r["seed"]) == ("fixture", "graph", 7)


def test_manifest_defaults_when_absent(tmp_path):
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces/trace_gearbox_202.json").write_text(
        (FIXTURE / "traces/trace_gearbox_202.json").read_text())
    one = [l for l in (FIXTURE / "task_results.jsonl").read_text().splitlines() if "gearbox_202" in l]
    (tmp_path / "task_results.jsonl").write_text("\n".join(one) + "\n")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"run": "baseline"}))
    (r,) = summarize_run(tmp_path, [_task("gearbox_202", "gearbox/painter.py")])
    assert (r["condition"], r["seed"]) == ("", None)


def test_trace_fixtures_are_what_the_generator_writes(tmp_path):
    """`tests/fixtures/make_traces.py` is the source of the fixtures; they are invented, and
    regenerating them must reproduce the committed bytes."""
    make_traces = _load_generator()
    make_traces.write_all(tmp_path)
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert written == ["run_manifest.json", "task_results.jsonl",
                       "traces/trace_gearbox_202.json", "traces/trace_widgetlib_101.json"]
    for rel in written:
        assert (tmp_path / rel).read_bytes() == (FIXTURE / rel).read_bytes(), rel


def _load_generator():
    import importlib.util

    path = Path(__file__).parent / "fixtures" / "make_traces.py"
    spec = importlib.util.spec_from_file_location("make_traces", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_csv_roundtrip(tmp_path):
    rows = list(_rows().values())
    out = tmp_path / "o.csv"
    write_csv(rows, out)
    got = list(csv.DictReader(out.open()))
    assert [g["instance_id"] for g in got] == [r["instance_id"] for r in rows]
    assert set(got[0]) == set(rows[0])


@pytest.mark.workspace
@pytest.mark.skipif(not (BASELINE and Path(BASELINE).exists() and TASKS_FILE.exists()),
                    reason="set GRAPHSWAP_BASELINE_RESULTS to a local baseline run directory")
def test_real_baseline_run():
    from graphswap.tasks import load_tasks

    rows = summarize_run(Path(BASELINE), load_tasks(TASKS_FILE))
    assert len(rows) == 24
    assert sum(r["resolved"] for r in rows) == 5


# --- search_similar_code classification (needs the vendored harness: `make vendor`) ---

GRAPH_NAMES = ["toyshop.gadgets.Sprocket", "toyshop.gadgets.Sprocket.whirl", "toyshop.helpers.tidy_up"]


def _call(name, **arguments):
    return {"function_name": name, "arguments": arguments}


def _search_run(tmp_path, queries_by_step):
    """A one-task run (instance toy_1 at repo o/r, commit abc) whose trace issues the given calls."""
    steps = [{"source": "agent", "tool_calls": calls} for calls in queries_by_step]
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces/trace_toy_1.json").write_text(json.dumps({"steps": steps}))
    (tmp_path / "task_results.jsonl").write_text(
        json.dumps({"instance_id": "toy_1", "resolved": False, "duration_seconds": 1.0}) + "\n")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"run": "toy", "condition": "C1", "seed": 1}))
    return tmp_path, [_task("toy_1", "toyshop/helpers.py")]


@pytest.fixture
def search_run(tmp_path):
    pytest.importorskip("swegemma.graph.graph_utils", reason="run `make vendor`")
    return _search_run(tmp_path, [
        [_call("search_similar_code", query="toyshop.helpers.tidy_up"),  # exact id
         _call("search_similar_code", query="Sprocket")],               # bare class name: suffix
        [_call("read_file", filepath="toyshop/helpers.py")],
        [_call("search_similar_code", query="nonexistent_thing")],      # lexical fallback
    ])


def test_search_resolved_and_lexical_counts(search_run):
    results, tasks = search_run
    (r,) = summarize_run(results, tasks, graphs_dir=GRAPHS)
    assert (r["search_calls"], r["search_resolved"], r["search_lexical"]) == (3, 2, 1)
    assert r["search_substring"] == 0
    assert r["graph_tool_calls"] == 3


def test_substring_resolutions_counted_within_resolved(tmp_path):
    pytest.importorskip("swegemma.graph.graph_utils", reason="run `make vendor`")
    results, tasks = _search_run(tmp_path, [[_call("search_similar_code", query="tidy"),  # substring hit
                                             _call("search_similar_code", query="Sprocket")]])
    (r,) = summarize_run(results, tasks, graphs_dir=GRAPHS)
    assert (r["search_calls"], r["search_resolved"], r["search_lexical"], r["search_substring"]) == (2, 2, 0, 1)


def test_search_columns_empty_without_graphs(search_run):
    results, tasks = search_run
    (r,) = summarize_run(results, tasks)
    assert (r["search_calls"], r["search_resolved"], r["search_lexical"], r["search_substring"]) == ("",) * 4


def test_search_columns_appended_last():
    from graphswap.traces import COLUMNS

    assert COLUMNS[-4:] == ["search_calls", "search_resolved", "search_lexical", "search_substring"]


def test_missing_graph_leaves_resolution_blank(search_run, tmp_path, caplog):
    results, tasks = search_run
    with caplog.at_level("WARNING"):
        (r,) = summarize_run(results, tasks, graphs_dir=tmp_path / "nographs")
    assert (r["search_calls"], r["search_resolved"], r["search_lexical"], r["search_substring"]) == (3, "", "", "")
    assert "r_abc.json" in caplog.text


@pytest.mark.parametrize("query, expected", [
    ("toyshop.helpers.tidy_up", "toyshop.helpers.tidy_up"),
    ("Sprocket", "toyshop.gadgets.Sprocket"),             # suffix on a '.' boundary
    ("Sprocket.whirl", "toyshop.gadgets.Sprocket.whirl"),
    ("sprocket", "toyshop.gadgets.Sprocket"),             # case-insensitive suffix
    ("tidy", "toyshop.helpers.tidy_up"),                  # substring, no whitespace
    ("o/r;Sprocket", "toyshop.gadgets.Sprocket"),         # 'repo;name' prefix is stripped
    ("nonexistent_thing", None),
    ("Sprocket whirl", None),                             # free text
    (" Sprocket", None),  # whitespace disables resolution in the harness, unlike resolve_node_name
    ("", None),
    (None, None),
])
def test_resolve_query_mirrors_harness(query, expected):
    gu = pytest.importorskip("swegemma.graph.graph_utils", reason="run `make vendor`")
    assert resolve_query(query, GRAPH_NAMES) == expected
    if query == " Sprocket":
        assert gu.resolve_node_name(query, GRAPH_NAMES) == "toyshop.gadgets.Sprocket"


def test_task_without_search_calls_needs_no_graph(tmp_path, caplog):
    results, tasks = _search_run(tmp_path, [[_call("read_file", filepath="a.py")]])
    with caplog.at_level("WARNING"):
        (r,) = summarize_run(results, tasks, graphs_dir=tmp_path / "nographs")
    assert (r["search_calls"], r["search_resolved"], r["search_lexical"], r["search_substring"]) == (0, 0, 0, 0)
    assert "nographs" not in caplog.text


@pytest.mark.parametrize("query, tier", [
    ("toyshop.helpers.tidy_up", "exact"),
    ("Sprocket", "suffix"),
    ("Sprocket.whirl", "suffix"),
    ("sprocket", "case-insensitive"),
    ("TOYSHOP.HELPERS.TIDY_UP", "case-insensitive"),
    ("tidy", "substring"),
    ("gadgets", "substring"),
    ("o/r;Sprocket", "suffix"),
    ("nonexistent_thing", ""),
])
def test_resolution_tier(query, tier):
    pytest.importorskip("swegemma.graph.graph_utils", reason="run `make vendor`")
    assert resolution_tier(query, resolve_query(query, GRAPH_NAMES)) == tier


def test_installed_harness_must_be_the_studied_version(tmp_path, monkeypatch):
    import types

    from graphswap import traces

    fake = types.ModuleType("swegemma")
    fake.__file__ = str(tmp_path / "site-packages" / "swegemma" / "__init__.py")
    monkeypatch.setattr(traces.importlib.metadata, "version", lambda name: "0.3.0")
    with pytest.raises(RuntimeError, match="swegemma 0.3.0 .* mirrors swegemma 0.2.11"):
        traces._check_harness(fake)
    monkeypatch.setattr(traces.importlib.metadata, "version", lambda name: "0.2.11")
    traces._check_harness(fake)
    vendored = types.ModuleType("swegemma")
    vendored.__file__ = str(traces.VENDOR / "swegemma" / "__init__.py")
    monkeypatch.setattr(traces.importlib.metadata, "version", lambda name: "9.9")
    traces._check_harness(vendored)  # the vendored copy is pinned by `make vendor`


def test_missing_cachetools_names_the_fix(monkeypatch):
    from graphswap import traces

    def boom(name):
        raise ModuleNotFoundError("No module named 'cachetools'", name="cachetools")

    monkeypatch.setattr(traces.importlib, "import_module", boom)
    with pytest.raises(ModuleNotFoundError, match=r"install the dev group \(uv sync\)"):
        traces._import_graph()
