import hashlib
import json
import os

import pytest

from graphswap.cli import main

COMMIT = "0b401c76b6e80a4eecf3c690085b2553f6e261ca"


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("GRAPHSWAP_NETWORK"), reason="set GRAPHSWAP_NETWORK=1")
def test_build_graphs_requests(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "instance_id": "psf__requests-1",
                "repo": "psf/requests",
                "base_commit": COMMIT,
                "patch": "",
                "test_patch": "",
                "problem_statement": "",
            }
        )
        + "\n"
    )
    out = tmp_path / "out"
    main(["build-graphs", "--tasks", str(tasks), "--repos", str(tmp_path / "repos"), "--out", str(out)])

    graphs = list(out.glob("requests_*.json"))
    assert [g.name for g in graphs] == [f"requests_{COMMIT}.json"]
    manifest = json.loads((out / "MANIFEST.json").read_text())
    entry = manifest["files"][graphs[0].name]
    assert entry["sha256"] == hashlib.sha256(graphs[0].read_bytes()).hexdigest()
    assert entry["node_count"] > 0 and entry["edge_count"] > 0
    assert manifest["builder_version"] == "graphswap-builder 0.1"
    assert manifest["built_at"]


def test_audit_fixture(tmp_path):
    import csv
    import shutil
    from pathlib import Path

    from graphswap.builder import build_graph, write_graph

    fix = Path(__file__).parent / "fixtures"
    commit = "c0ffee"
    wt = tmp_path / "repos" / "wt" / f"pkg_{commit}"
    shutil.copytree(fix / "repo", wt)
    g = build_graph(wt, "pkg")
    write_graph(g, tmp_path / "ours" / "graphs_rebuilt", "pkg", commit)
    rel = {**g, "nodes": [n for n in g["nodes"] if n["id"] != "pkg.core.Thing.run"]
           + [{**g["nodes"][0], "id": "pkg.bogus"}]}
    write_graph(rel, tmp_path / "rel" / "graphs", "pkg", commit)
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({
        "instance_id": "o__pkg-1", "repo": "o/pkg", "base_commit": commit,
        "patch": (fix / "patches" / "run.diff").read_text(), "test_patch": "",
        "problem_statement": ""}) + "\n")
    out = tmp_path / "results" / "audit.csv"
    main(["audit", "--tasks", str(tasks), "--released", str(tmp_path / "rel"),
          "--ours", str(tmp_path / "ours"), "--repos", str(tmp_path / "repos"), "--out", str(out)])
    (row,) = csv.DictReader(open(out))
    n = len(g["nodes"])
    assert int(row["rebuilt_nodes"]) == n and row["released_only"] == "1"
    assert float(row["id_parity"]) == (n - 1) / n
    assert row["rel_mean_cos"] == "" and row["ours_mean_cos"] == ""
    (trow,) = csv.DictReader(open(tmp_path / "results" / "audit_tasks.csv"))
    assert trow["n_gold"] == "1" and trow["released_fraction"] == "0.0"
    assert trow["rebuilt_fraction"] == "1.0"


def test_audit_geometry_columns(tmp_path):
    import csv
    import shutil
    from pathlib import Path

    import numpy as np

    from graphswap.builder import build_graph, write_graph

    fix = Path(__file__).parent / "fixtures"
    wt = tmp_path / "repos" / "wt" / "pkg_c1"
    shutil.copytree(fix / "repo", wt)
    g = build_graph(wt, "pkg")
    write_graph(g, tmp_path / "ours" / "graphs_rebuilt", "pkg", "c1")
    rng = np.random.default_rng(0)
    for d in ("rel", "ours"):
        (tmp_path / d / "embeddings").mkdir(parents=True)
        np.savez(tmp_path / d / "embeddings" / "pkg_c1.npz",
                 **{f"k{i}": rng.standard_normal(16).astype(np.float32) for i in range(50)})
    tasks = tmp_path / "t.jsonl"
    tasks.write_text(json.dumps({"instance_id": "x", "repo": "o/pkg", "base_commit": "c1",
                                 "patch": "", "test_patch": "", "problem_statement": ""}) + "\n")
    out = tmp_path / "audit.csv"
    main(["audit", "--tasks", str(tasks), "--released", str(tmp_path / "rel"),
          "--ours", str(tmp_path / "ours"), "--repos", str(tmp_path / "repos"), "--out", str(out)])
    (row,) = csv.DictReader(open(out))
    assert row["released_nodes"] == "" and row["id_parity"] == ""  # no released graph: skipped
    assert float(row["rel_eff_dim"]) > 1 and float(row["ours_dup_rate"]) == 0.0


def _write_tasks(path, counts):
    rows = [{"instance_id": f"{r}_{i}", "repo": f"o/{r}", "base_commit": "c", "patch": "",
             "test_patch": "", "problem_statement": ""}
            for r, n in counts.items() for i in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_summarize_fixture(tmp_path, capsys):
    import csv
    from pathlib import Path

    fix = Path(__file__).parent / "fixtures" / "traces"
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps({
        "instance_id": i, "repo": "o/x", "base_commit": "c", "patch": "", "test_patch": "",
        "problem_statement": ""}) for i in ("widgetlib_101", "gearbox_202")) + "\n")
    out = tmp_path / "s.csv"
    main(["summarize", "--results", str(fix), "--tasks", str(tasks), "--out", str(out)])
    rows = list(csv.DictReader(open(out)))
    assert [r["instance_id"] for r in rows] == ["widgetlib_101", "gearbox_202"]
    assert [r["resolved"] for r in rows] == ["False", "True"]
    assert rows[0]["run"] == "fixture" and rows[0]["seed"] == "7"
    printed = capsys.readouterr().out
    assert "resolved: 1/2" in printed and "graph_tool_calls" in printed


def test_compare_prints_stats(tmp_path, capsys):
    cols = "instance_id,seed,resolved\n"
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text(cols + "t1,1,False\nt2,1,False\nt3,1,True\n")
    b.write_text(cols + "t1,1,True\nt2,1,True\nt3,1,True\n")
    main(["compare", "--control", str(a), "--treatment", str(b)])
    out = capsys.readouterr().out
    assert "diff:" in out and "mcnemar_p:" in out and "boot_ci:" in out
    main(["compare", "--control", str(a), "--treatment", str(b), "--noise"])
    assert "discordant:" in capsys.readouterr().out


def test_draw_subset_cli(tmp_path):
    from collections import Counter

    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks, {"fastapi": 67, "rich": 48, "requests": 13, "httpx": 1})
    inc = tmp_path / "inc.txt"
    ids = ([f"fastapi_{i}" for i in range(11)] + ["httpx_0"]
           + [f"requests_{i}" for i in range(3)] + [f"rich_{i}" for i in range(9)])
    inc.write_text("# c\n" + "\n".join(ids) + "\n")
    outs = []
    for k in (1, 2):
        out = tmp_path / f"o{k}.txt"
        main(["draw-subset", "--tasks", str(tasks), "--n", "60", "--seed", "0",
              "--include", str(inc), "--out", str(out)])
        outs.append(out.read_text())
    assert outs[0] == outs[1]
    got = [l for l in outs[0].splitlines() if l and not l.startswith("#")]
    assert len(got) == 60 == len(set(got)) and set(ids) <= set(got)
    assert outs[0].startswith("#") and "Fixed: do not regenerate" in outs[0]
    counts = Counter(i.split("_")[0] for i in got)
    assert counts["fastapi"] in (31, 32) and counts["httpx"] == 1 and counts["requests"] in (6, 7)


def test_draw_subset_include_exceeds_quota():
    from graphswap.tasks import Task, draw_subset

    mk = lambda r, i: Task(f"{r}_{i}", f"o/{r}", r, "c", "", "", "")
    tasks = [mk("a", i) for i in range(50)] + [mk("b", i) for i in range(50)]
    inc = [f"a_{i}" for i in range(20)]
    got = draw_subset(tasks, 20, 0, inc)
    assert len(got) == 20 and set(inc) <= set(got)
    got = draw_subset(tasks, 30, 0, inc)
    assert len(got) == 30 and sum(g.startswith("b") for g in got) == 10


def test_summarize_with_graphs(tmp_path, capsys):
    import csv
    from pathlib import Path

    pytest.importorskip("swegemma.graph.graph_utils", reason="run `make vendor`")
    run = tmp_path / "run"
    (run / "traces").mkdir(parents=True)
    calls = [{"function_name": "search_similar_code", "arguments": {"query": q}}
             for q in ("Sprocket", "nonexistent_thing")]
    (run / "traces/trace_toy_1.json").write_text(json.dumps({"steps": [{"tool_calls": calls}]}))
    (run / "task_results.jsonl").write_text(json.dumps({"instance_id": "toy_1", "resolved": True}) + "\n")
    (run / "run_manifest.json").write_text(json.dumps({"run": "toy"}))
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"instance_id": "toy_1", "repo": "o/r", "base_commit": "abc", "patch": "",
                                 "test_patch": "", "problem_statement": ""}) + "\n")
    out = tmp_path / "s.csv"
    graphs = Path(__file__).parent / "fixtures" / "graphs"
    main(["summarize", "--results", str(run), "--tasks", str(tasks), "--out", str(out), "--graphs", str(graphs)])
    (row,) = csv.DictReader(open(out))
    assert (row["search_calls"], row["search_resolved"], row["search_lexical"], row["search_substring"]) == \
        ("2", "1", "1", "0")
    assert "search calls: 2, resolved 1 (50.0%) of which substring 0" in capsys.readouterr().out
