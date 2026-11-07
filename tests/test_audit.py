from pathlib import Path

import numpy as np

from graphswap.audit import _touched_lines, embedding_geometry, gold_coverage, gold_symbols
from graphswap.builder import build_graph
from graphswap.tasks import Task

FIX = Path(__file__).parent / "fixtures"
REPO = FIX / "repo"


def _task(patch_name: str) -> Task:
    return Task("i", "o/pkg", "pkg", "c", (FIX / "patches" / patch_name).read_text(), "", "")


def test_geometry_identical_vectors():
    v = {f"n{i}": np.ones(8, dtype=np.float32) for i in range(100)}
    g = embedding_geometry(v)
    assert g["duplicate_rate"] == 1.0
    assert abs(g["mean_random_cosine"] - 1.0) < 1e-6


def test_geometry_random_gaussian():
    rng = np.random.default_rng(1)
    v = {f"n{i}": rng.standard_normal(256).astype(np.float32) for i in range(2000)}
    g = embedding_geometry(v)
    assert g["effective_dim"] > 200
    assert g["duplicate_rate"] == 0.0
    assert abs(g["mean_random_cosine"]) < 0.02
    assert g["p05"] < g["p95"]


def test_geometry_tiny_is_nan():
    g = embedding_geometry({"a": np.ones(4)})
    assert np.isnan(g["mean_random_cosine"])


def test_gold_symbols_method():
    assert gold_symbols(_task("run.diff"), REPO) == ["pkg.core.Thing.run"]


def test_add_only_patch():
    t = _task("add_only.diff")
    assert gold_symbols(t, REPO) is None
    c = gold_coverage(t, REPO, build_graph(REPO, "pkg"))
    assert c["gold"] is None and c["fraction"] is None and c["covered"] == []


def test_gold_coverage_partial():
    g = build_graph(REPO, "pkg")
    g["nodes"] = [n for n in g["nodes"] if n["id"] != "pkg.core.Thing.run"]
    c = gold_coverage(_task("run.diff"), REPO, g)
    assert c["gold"] == ["pkg.core.Thing.run"] and c["covered"] == [] and c["fraction"] == 0.0


def test_pure_insertion_hunk_maps_to_enclosing_def():
    assert gold_symbols(_task("insert_after_def.diff"), REPO) == ["pkg.core.Thing.run"]


def test_touched_lines_pure_insertion():
    patch = Path(FIX / "patches" / "insert_after_def.diff").read_text()
    assert _touched_lines(patch) == {"pkg/core.py": {8}}


def test_gold_coverage_uses_supplied_gold(tmp_path):
    graph = {"nodes": [{"id": "a"}]}
    cov = gold_coverage(_task("run.diff"), tmp_path / "missing", graph, gold=["a", "b"])
    assert cov == {"gold": ["a", "b"], "covered": ["a"], "fraction": 0.5}


def test_write_csv_nan_is_empty(tmp_path):
    from graphswap.cli import _write_csv
    p = tmp_path / "o.csv"
    _write_csv(p, ["a", "b", "c"], [{"a": float("nan"), "b": None, "c": 1.5}])
    assert p.read_text().splitlines() == ["a,b,c", ",,1.5"]


def test_write_csv_numpy_nan_is_empty(tmp_path):
    """np.float32 is not a float subclass, so it needs its own check (np.float64 is one)."""
    from graphswap.cli import _write_csv
    p = tmp_path / "o.csv"
    _write_csv(p, ["a", "b", "c"], [{"a": np.float32("nan"), "b": np.float64("nan"),
                                     "c": np.float32(1.5)}])
    assert p.read_text().splitlines() == ["a,b,c", ",,1.5"]


def out_rows(path: Path) -> list[dict]:
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _audit_args(tmp_path, tasks_text=""):
    (tmp_path / "tasks.jsonl").write_text(tasks_text)
    for name in ("released", "ours", "repos"):
        (tmp_path / name).mkdir()
    return ["audit", "--tasks", str(tmp_path / "tasks.jsonl"), "--released", str(tmp_path / "released"),
            "--ours", str(tmp_path / "ours"), "--repos", str(tmp_path / "repos"),
            "--out", str(tmp_path / "out" / "audit.csv")]


def test_audit_with_no_tasks_writes_header_only_csvs(tmp_path):
    from graphswap.cli import COMMIT_COLS, TASK_COLS, main
    main(_audit_args(tmp_path))
    out = tmp_path / "out"
    assert out.joinpath("audit.csv").read_text().splitlines() == [",".join(COMMIT_COLS)]
    assert out.joinpath("audit_tasks.csv").read_text().splitlines() == [",".join(TASK_COLS)]


def test_audit_does_not_rewrite_the_csvs_after_the_loop(tmp_path):
    """The in-loop write already holds every row; the post-loop one is only for an empty run."""
    import json
    from unittest import mock
    from graphswap import cli
    task = {"instance_id": "i1", "repo": "o/pkg", "repo_short": "pkg", "base_commit": "c0",
            "patch": "", "problem_statement": "", "test_patch": ""}
    args = _audit_args(tmp_path, json.dumps(task) + "\n")
    # no rebuilt graph for the commit: the loop skips it, so only the post-loop write runs
    with mock.patch.object(cli, "_write_csv", wraps=cli._write_csv) as spy:
        cli.main(args)
    assert spy.call_count == 2
    (tmp_path / "ours" / "graphs_rebuilt").mkdir(parents=True)
    (tmp_path / "ours" / "graphs_rebuilt" / "pkg_c0.json").write_text(json.dumps({"nodes": []}))
    with mock.patch.object(cli, "_write_csv", wraps=cli._write_csv) as spy, \
            mock.patch.object(cli, "checkout", side_effect=OSError("no clone in tests")):
        cli.main(args)  # the checkout fails and is logged, but the commit row is written
    assert spy.call_count == 2  # once per CSV, in the loop; no duplicate pair afterwards
    assert len(out_rows(tmp_path / "out" / "audit.csv")) == 1


def test_geometry_cells_corrupt_npz_is_nan(tmp_path):
    import zlib
    from unittest import mock
    from graphswap import cli
    f = tmp_path / "x.npz"
    f.write_bytes(b"x")
    with mock.patch.object(cli.np, "load", side_effect=zlib.error("bad")):
        cells = cli._geometry_cells(f, "rel")
    assert set(cells) == {f"rel_{k}" for k in cli.GEOM_KEYS}
    assert all(v != v for v in cells.values())
