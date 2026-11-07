import json
import os
from pathlib import Path

import pytest

from graphswap.builder import build_graph, filter_edges, write_graph
from graphswap.repos import checkout

FIXTURE = Path(__file__).parent / "fixtures" / "repo"
COMMIT = "0b401c76b6e80a4eecf3c690085b2553f6e261ca"
RELEASED = os.environ.get("GRAPHSWAP_RELEASED_REQUESTS_GRAPH")


def test_fixture_nodes_and_kinds():
    g = build_graph(FIXTURE, "pkg")
    ids = {n["id"] for n in g["nodes"]}
    assert {"pkg.core.Thing", "pkg.core.Thing.run", "pkg.core.helper",
            "pkg.core.fetch_async", "pkg.core.inner", "pkg.util.util_fn"} <= ids
    kinds = {n["id"]: n["kind"] for n in g["nodes"]}
    assert kinds["pkg.core.fetch_async"] == "async_function"
    assert kinds["pkg.core.Thing"] == "class"
    assert kinds["pkg.core.Thing.run"] == "method"
    assert kinds["pkg.core"] == "module"
    by_id = {n["id"]: n for n in g["nodes"]}
    assert by_id["pkg.core.fetch_async"]["is_async"] is True
    assert by_id["pkg.core.helper"]["is_async"] is False
    assert by_id["pkg.core"]["text"] == "Core module."
    assert by_id["pkg.core.helper"]["text"].startswith("@functools.lru_cache")
    assert all(n["name"] == n["id"] for n in g["nodes"])


def test_edge_types():
    g = build_graph(FIXTURE, "pkg")
    types = {(e["source"], e["target"], e["type"]) for e in g["edges"]}
    assert ("pkg.core.Thing.run", "pkg.core.helper", "calls") in types
    assert ("pkg.core.Sub", "pkg.core.Thing", "inherits") in types
    assert ("pkg.core", "pkg.util", "imports") in types
    assert ("pkg.core.Thing", "pkg.core.Thing.run", "contains") in types
    assert ("pkg.core", "pkg.core.Thing", "contains") in types
    # imported name, self.method, inherited self.method, nested def
    assert ("pkg.core.Thing.other", "pkg.util.util_fn", "calls") in types
    assert ("pkg.core.Thing.run", "pkg.core.Thing.other", "calls") in types
    assert ("pkg.core.Sub.go", "pkg.core.Thing.run", "calls") in types
    assert ("pkg.core.helper", "pkg.core.inner", "calls") in types
    via = {(e["source"], e["target"]): e["via"] for e in g["edges"] if e["type"] == "calls"}
    assert via[("pkg.core.Thing.other", "pkg.util.util_fn")] == "import"
    assert via[("pkg.core.Sub.go", "pkg.core.Thing.run")] == "self"


def test_edge_keys_unique_per_pair():
    g = build_graph(FIXTURE, "pkg")
    seen = [(e["source"], e["target"], e["key"]) for e in g["edges"]]
    assert len(seen) == len(set(seen))
    assert all(isinstance(e["key"], int) for e in g["edges"])


def test_filter_edges():
    g = build_graph(FIXTURE, "pkg")
    f = filter_edges(g, {"calls"})
    assert f["edges"] and {e["type"] for e in f["edges"]} == {"calls"}
    assert f["nodes"] == g["nodes"]
    assert {e["type"] for e in g["edges"]} > {"calls"}  # input untouched


def test_default_discovery_indexes_all_py_strips_src_skips_hidden(tmp_path):
    for rel in ["src/lib/__init__.py", "src/lib/a.py", "tests/t.py", "docs/conf.py",
                ".hidden/h.py", "pkg/__pycache__/x.py", "setup.py"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("def f():\n    pass\n")
    ids = {n["id"] for n in build_graph(tmp_path, "lib")["nodes"] if n["kind"] == "function"}
    assert ids == {"lib.f", "lib.a.f", "tests.t.f", "docs.conf.f", "setup.f"}


def test_package_dirs_override(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "m.py").write_text("def f():\n    pass\n")
    (tmp_path / "b.py").write_text("def g():\n    pass\n")
    ids = {n["id"] for n in build_graph(tmp_path, "x", [tmp_path / "a"])["nodes"]}
    assert ids == {"a.m", "a.m.f"}


def test_duplicate_ids_keep_first_text_and_all_calls():
    g = build_graph(FIXTURE, "pkg")
    node = next(n for n in g["nodes"] if n["id"] == "pkg.core.Holder.value")
    assert node["text"].startswith("@property")
    types = {(e["source"], e["target"], e["type"]) for e in g["edges"]}
    assert ("pkg.core.Holder.value", "pkg.core.make_default", "calls") in types


def test_nested_def_ids_drop_function_scopes(tmp_path):
    (tmp_path / "m.py").write_text(
        "class C:\n    def meth(self):\n        def deep():\n            return 1\n"
        "        return deep()\n"
    )
    g = build_graph(tmp_path, "m")
    ids = {n["id"] for n in g["nodes"]}
    assert "m.C.deep" in ids and "m.C.meth.deep" not in ids
    types = {(e["source"], e["target"], e["type"]) for e in g["edges"]}
    assert ("m.C.meth", "m.C.deep", "contains") in types
    assert ("m.C.meth", "m.C.deep", "calls") in types


def test_class_nested_in_method_keeps_outer_class_prefix(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text(
        "class Outer:\n    def a(self):\n        class Inner:\n            def x(self):\n"
        "                pass\n        return Inner\n\n    def b(self):\n        pass\n"
    )
    g = build_graph(tmp_path, "pkg")
    ids = {n["id"] for n in g["nodes"] if n["kind"] != "module"}
    assert ids == {"pkg.m.Outer", "pkg.m.Outer.a", "pkg.m.Outer.Inner", "pkg.m.Outer.Inner.x",
                   "pkg.m.Outer.b"}
    types = {(e["source"], e["target"], e["type"]) for e in g["edges"]}
    assert ("pkg.m.Outer", "pkg.m.Outer.b", "contains") in types


def test_attribute_call_unique_fallback_only_targets_methods(tmp_path):
    (tmp_path / "m.py").write_text(
        "def send():\n    pass\n\n\nclass A:\n    def go(self):\n        self.session.send()\n"
        "        self.session.push()\n\n    def push(self):\n        pass\n"
    )
    types = {(e["source"], e["target"]) for e in build_graph(tmp_path, "m")["edges"] if e["type"] == "calls"}
    assert ("m.A.go", "m.send") not in types
    assert ("m.A.go", "m.A.push") in types


def test_unparseable_file_is_logged_and_skipped(tmp_path, caplog):
    (tmp_path / "bad.py").write_text("def (:\n")
    (tmp_path / "ok.py").write_text("def f():\n    pass\n")
    with caplog.at_level("WARNING"):
        ids = {n["id"] for n in build_graph(tmp_path, "m")["nodes"]}
    assert "ok.f" in ids and "bad" not in ids
    assert "bad.py" in caplog.text


def test_loads_in_harness_and_resolves_suffix(tmp_path):
    gu = pytest.importorskip(
        "swegemma.graph.graph_utils",
        reason="run `make vendor` to extract swegemma/graph into vendor/",
    )
    p = write_graph(build_graph(FIXTURE, "pkg"), tmp_path, "pkg", "deadbeef")
    g = gu.load_graph_from_json(str(p))
    names = [str(n) for n in g.nodes()]
    assert gu.resolve_node_name("core.fetch_async", names) == "pkg.core.fetch_async"
    assert g.number_of_edges() == len(build_graph(FIXTURE, "pkg")["edges"])


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("GRAPHSWAP_NETWORK"), reason="set GRAPHSWAP_NETWORK=1")
@pytest.mark.skipif(not RELEASED or not Path(RELEASED).exists(),
                    reason="set GRAPHSWAP_RELEASED_REQUESTS_GRAPH to the released requests graph")
def test_parity_with_released_requests_graph():
    ours = {n["id"] for n in build_graph(checkout("psf/requests", COMMIT, Path("repos")), "requests")["nodes"]}
    released = {n["id"] for n in json.load(open(RELEASED))["nodes"]}
    frac = len(released & ours) / len(released)
    print(f"parity: {len(released & ours)}/{len(released)} = {frac:.4f}")
    # Remaining misses are released ids misnamed by its nested-class scoping bug (a class nested
    # in a function drops the outer class from later methods); quantified later by the audit.
    assert frac >= 0.75
