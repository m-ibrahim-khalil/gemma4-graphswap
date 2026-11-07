import csv
import json

import numpy as np
import pytest

pytest.importorskip("swegemma.graph.retrieval_utils", reason="run `make vendor`")

from graphswap.cli import main  # noqa: E402
from graphswap.replay import (DEFAULT_K, ROW_COLUMNS, SUMMARY_COLUMNS, coerce_k,  # noqa: E402
                              dedupe, replay_run, summarize_replay)
from graphswap.tasks import Task  # noqa: E402

COMMIT = "c0ffee"
HUB = "toyshop.core.Hub"
OTHERS = [f"toyshop.core.part_{c}" for c in "abcdef"]  # n1..n6
TEXT = "def secret_body_marker(): pass"  # node text; must never reach an output file

# Cosine to the Hub node per set (vectors are unit, so cos(Hub, n_i) is exactly the listed value).
A = dict(zip(OTHERS, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]))           # order a b c d e f
B_HALF = dict(zip(OTHERS, [0.9, 0.7, 0.5, 0.4, 0.8, 0.6]))      # order a e b f c d: top4 shares a,b
B_SWAP = dict(zip(OTHERS, [0.8, 0.9, 0.7, 0.6, 0.5, 0.4]))      # order b a c d e f: same set, new top1


def _vectors(cos, dim):
    """Hub = e0; n_i = c e0 + sqrt(1-c^2) e_i, so cos(Hub, n_i) = c."""
    vecs = {HUB: np.eye(dim, dtype=np.float32)[0]}
    for i, name in enumerate(OTHERS, 1):
        v = np.zeros(dim, dtype=np.float32)
        v[0], v[i] = cos[name], np.sqrt(1 - cos[name] ** 2)
        vecs[name] = v
    return vecs


def _top(cos, k):
    return sorted(OTHERS, key=lambda n: -cos[n])[:k]


def _write_data(root, sets):
    graph = {"directed": True, "multigraph": True, "graph": {}, "edges": [],
             "nodes": [{"id": n, "name": n, "text": TEXT} for n in [HUB, *OTHERS]]}
    (root / "graphs").mkdir()
    (root / "graphs" / f"toy_{COMMIT}.json").write_text(json.dumps(graph))
    for label, (cos, dim) in sets.items():
        (root / label).mkdir()
        np.savez_compressed(root / label / f"toy_{COMMIT}.npz", **_vectors(cos, dim))


def _write_run(root, steps, name="R1"):
    run = root / name
    (run / "traces").mkdir(parents=True)
    (run / "traces" / "trace_toy_1.json").write_text(json.dumps({"steps": steps}))
    (run / "task_results.jsonl").write_text(json.dumps({"instance_id": "toy_1", "resolved": False}) + "\n")
    (run / "run_manifest.json").write_text(json.dumps({"run": name}))
    return run


def _search(query, **extra):
    return {"function_name": "search_similar_code", "arguments": {"query": query, **extra}}


def _observation(query, names):
    content = {"status": "ok", "query": query, "count": len(names), "is_truncated": False,
               "results": [{"node_name": n, "code": TEXT, "similarity": 0.5} for n in names]}
    return {"content": json.dumps(content), "extra": {"tool_name": "search_similar_code"}}


TASKS = [Task("toy_1", "acme/toy", "toy", COMMIT, "", "", "")]


@pytest.fixture
def data(tmp_path):
    _write_data(tmp_path, {"A": (A, 7), "B_HALF": (B_HALF, 9), "B_SWAP": (B_SWAP, 7)})
    steps = [
        {"tool_calls": [_search("Hub")], "observation": _observation("Hub", _top(A, 4))},
        {"tool_calls": [_search("part flow")],  # free text: lexical fallback, same list under A and B
         "observation": _observation("part flow", OTHERS[:4])},
        {"tool_calls": [{"function_name": "read_file", "arguments": {"filepath": "x.py"}},
                        _search("core.Hub", k="2")]},  # explicit k, coerced like the harness does
    ]
    return tmp_path, _write_run(tmp_path, steps)


def _replay(root, run, b, k=4):
    return replay_run(run, TASKS, root / "graphs", root / "A", root / b, k=k)


def test_default_k_is_harness_default():
    assert DEFAULT_K == 10


def test_overlap_is_known_fraction_and_top1_same(data):
    root, run = data
    rows = _replay(root, run, "B_HALF")
    assert [r["query"] for r in rows] == ["Hub", "part flow", "core.Hub"]
    hub, free, hub2 = rows
    assert (hub["resolved_node"], hub["k"], hub["overlap"], hub["top1_same"]) == (HUB, 4, 0.5, True)
    assert (free["resolved_node"], free["overlap"], free["top1_same"]) == ("", "", "")
    # k=2: A top2 = a,b; B top2 = a,e
    assert (hub2["resolved_node"], hub2["k"], hub2["overlap"], hub2["top1_same"]) == (HUB, 2, 0.5, True)
    assert [r["step"] for r in rows] == [0, 1, 2]


def test_same_set_new_top1(data):
    root, run = data
    hub = _replay(root, run, "B_SWAP")[0]
    assert (hub["overlap"], hub["top1_same"]) == (1.0, False)


def test_query_node_excluded_and_ties_by_name(tmp_path):
    # All of B's vectors equal: every cosine is 1, so the harness orders by node name, never the query.
    flat = dict.fromkeys(OTHERS, 1.0)
    _write_data(tmp_path, {"A": (A, 7), "B": (flat, 7)})
    run = _write_run(tmp_path, [{"tool_calls": [_search("Hub")]}])
    (hub,) = replay_run(run, TASKS, tmp_path / "graphs", tmp_path / "A", tmp_path / "B", k=6)
    assert hub["overlap"] == 1.0 and hub["top1_same"] is True  # all 6 others, a first either way


def test_observed_column_names_reproducing_set(data):
    root, run = data
    rows = _replay(root, run, "B_HALF")
    assert [r["observed"] for r in rows] == ["A", "AB", ""]
    same = _replay(root, run, "A")
    assert same[0]["observed"] == "AB" and same[0]["overlap"] == 1.0


def test_summary_per_run(data):
    root, run = data
    (s,) = summarize_replay(_replay(root, run, "B_HALF"))
    assert set(s) == set(SUMMARY_COLUMNS)
    assert (s["run"], s["unit"], s["n_calls"], s["n_resolved"], s["n_substring"]) == ("R1", "call", 3, 2, 0)
    assert s["mean_overlap"] == pytest.approx(0.5) and s["frac_top1_same"] == 1.0
    assert s["unresolved_changed"] == 0
    # the lexical call's recorded list matches both sets trivially; it is kept out of the *_embed counts
    assert (s["n_observed_embed"], s["observed_a_embed"], s["observed_b_embed"]) == (1, 1, 0)
    assert (s["n_observed_lexical"], s["observed_lexical"]) == (1, 1)


def test_missing_embeddings_raise(data):
    root, run = data
    with pytest.raises(FileNotFoundError, match=f"toy_{COMMIT}.npz"):
        replay_run(run, TASKS, root / "graphs", root / "A", root / "nowhere")


def test_coerce_k_mirrors_harness():
    assert coerce_k(None) == DEFAULT_K and coerce_k("") == DEFAULT_K and coerce_k("null") == DEFAULT_K
    assert coerce_k("7") == 7 and coerce_k(3.0) == 3 and coerce_k("5.0") == 5 and coerce_k(None, 4) == 4
    with pytest.raises(ValueError):
        coerce_k("many")
    with pytest.raises(TypeError):
        coerce_k(True)


def test_cli_writes_rows_and_summary_without_node_text(data, capsys):
    root, run = data
    out = root / "out" / "replay.csv"
    main(["replay-search", "--graph-dir", str(root / "graphs"), "--emb-a", str(root / "A"),
          "--emb-b", str(root / "B_HALF"), "--runs", str(run), "--out", str(out), "--k", "4",
          "--tasks", str(_tasks_file(root))])
    rows = list(csv.DictReader(open(out)))
    assert list(rows[0]) == ROW_COLUMNS and len(rows) == 3
    lines = (root / "out" / "replay.summary.csv").read_text().splitlines()
    assert lines[0].startswith("#") and "min(k, n_nodes - 1)" in lines[0]
    (s,) = csv.DictReader(lines[1:])
    assert (s["unit"], s["n_calls"], s["n_resolved"], s["mean_overlap"], s["frac_top1_same"]) == \
        ("call", "3", "2", "0.5", "1.0")
    for p in (out, root / "out" / "replay.summary.csv"):
        assert "secret_body_marker" not in p.read_text()
    assert "R1" in capsys.readouterr().out


def _tasks_file(root):
    p = root / "tasks.jsonl"
    p.write_text(json.dumps({"instance_id": "toy_1", "repo": "acme/toy", "base_commit": COMMIT, "patch": "",
                             "test_patch": "", "problem_statement": ""}) + "\n")
    return p


def test_recorded_error_marks_single_call_step(tmp_path):
    _write_data(tmp_path, {"A": (A, 7), "B": (B_HALF, 7)})
    error = {"content": json.dumps({"status": "error", "error_type": "BudgetExceeded", "error_message": "x"}),
             "extra": {"tool_name": "search_similar_code"}}
    run = _write_run(tmp_path, [
        {"tool_calls": [_search("Hub")], "observation": error},
        {"tool_calls": [_search("Hub"), _search("part_a")], "observation": error},  # ambiguous: unmarked
    ])
    rows = replay_run(run, TASKS, tmp_path / "graphs", tmp_path / "A", tmp_path / "B", k=4)
    assert [r["observed"] for r in rows] == ["error", "", ""]


def test_tier_column(tmp_path):
    _write_data(tmp_path, {"A": (A, 7), "B": (B_HALF, 7)})
    queries = ["toyshop.core.Hub", "Hub", "hub", "art_b", "part flow"]
    run = _write_run(tmp_path, [{"tool_calls": [_search(q) for q in queries]}])
    rows = replay_run(run, TASKS, tmp_path / "graphs", tmp_path / "A", tmp_path / "B", k=4)
    assert [r["tier"] for r in rows] == ["exact", "suffix", "case-insensitive", "substring", ""]
    assert rows[3]["resolved_node"] == "toyshop.core.part_b"  # any node containing the query
    (s,) = summarize_replay(rows)
    assert (s["n_resolved"], s["n_substring"]) == (4, 1)


def test_overlap_denominator_is_min_k_and_other_nodes(data):
    root, run = data
    hub = _replay(root, run, "B_HALF", k=10)[0]  # 7 nodes: the harness returns the 6 others
    assert (hub["k"], hub["overlap"]) == (10, 1.0)


def test_dedupe_keeps_first_row_per_query_and_k(tmp_path, capsys):
    _write_data(tmp_path, {"A": (A, 7), "B": (B_HALF, 7)})
    # the same query three times: twice at the run default k (a true duplicate) and once at k=2,
    # which is a different retrieval and must survive.
    run = _write_run(tmp_path, [{"tool_calls": [_search("Hub")]},
                                {"tool_calls": [_search("Hub", k=2), _search("Hub"),
                                                _search("part flow")]}])
    rows = replay_run(run, TASKS, tmp_path / "graphs", tmp_path / "A", tmp_path / "B", k=4)
    assert [(r["step"], r["query"], r["k"]) for r in rows] == [
        (0, "Hub", 4), (1, "Hub", 2), (1, "Hub", 4), (1, "part flow", 4)]
    assert [(r["step"], r["k"]) for r in dedupe(rows)] == [(0, 4), (1, 2), (1, 4)]
    out = tmp_path / "replay.csv"
    main(["replay-search", "--graph-dir", str(tmp_path / "graphs"), "--emb-a", str(tmp_path / "A"),
          "--emb-b", str(tmp_path / "B"), "--runs", str(run), "--out", str(out), "--k", "4", "--dedupe",
          "--tasks", str(_tasks_file(tmp_path))])
    assert len(list(csv.DictReader(open(out)))) == 3  # --out holds the deduplicated rows
    summary = list(csv.DictReader(out.with_suffix(".summary.csv").read_text().splitlines()[1:]))
    assert [(s["run"], s["unit"], s["n_calls"], s["n_resolved"]) for s in summary] == \
        [("R1", "call", "4", "3"), ("R1", "query", "3", "2")]
