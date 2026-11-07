import hashlib
from pathlib import Path

import numpy as np
import pytest

from graphswap.builder import build_graph
from graphswap.embedder import Cache, embed_graph, write_npz

FIXTURE = Path(__file__).parent / "fixtures" / "repo"


class CountingFakeEncoder:
    model_id = "fake/model"

    def __init__(self, dim):
        self.dim = dim
        self.calls = 0
        self.texts = []

    def encode(self, texts):
        self.calls += 1
        self.texts.extend(texts)
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little")
            out[i] = np.random.default_rng(seed).standard_normal(self.dim)
        return out


def test_npz_keys_match_nodes_and_cache_hits(tmp_path):
    g = build_graph(FIXTURE, "pkg")
    enc = CountingFakeEncoder(dim=8)
    vecs = embed_graph(g, enc, Cache(tmp_path / "c.sqlite"))
    assert set(vecs) == {n["id"] for n in g["nodes"]}
    assert enc.calls == 1
    again = embed_graph(g, enc, Cache(tmp_path / "c.sqlite"))
    assert enc.calls == 1  # second pass fully cached
    for k in vecs:
        assert np.array_equal(vecs[k], again[k])
        assert again[k].dtype == np.float32


def test_blank_text_embedded_as_node_id(tmp_path):
    g = {"nodes": [{"id": "a.b", "name": "a.b", "text": "   "}], "edges": []}
    enc = CountingFakeEncoder(8)
    vecs = embed_graph(g, enc, Cache(tmp_path / "c.sqlite"))
    assert np.isfinite(vecs["a.b"]).all()
    assert enc.texts == ["a.b"]


def test_non_finite_vector_raises_naming_node(tmp_path):
    class NanEncoder(CountingFakeEncoder):
        def encode(self, texts):
            v = super().encode(texts)
            v[0, 0] = np.nan
            return v

    g = {"nodes": [{"id": "a.b", "name": "a.b", "text": "x"}], "edges": []}
    with pytest.raises(ValueError, match="a.b"):
        embed_graph(g, NanEncoder(8), Cache(tmp_path / "c.sqlite"))


def test_cache_keyed_by_model(tmp_path):
    g = {"nodes": [{"id": "a", "name": "a", "text": "x"}], "edges": []}
    e1, e2 = CountingFakeEncoder(4), CountingFakeEncoder(4)
    e2.model_id = "other/model"
    c = Cache(tmp_path / "c.sqlite")
    embed_graph(g, e1, c)
    embed_graph(g, e2, c)
    assert e2.calls == 1


def test_npz_loads_in_harness(tmp_path):
    from swegemma.graph.embedding_utils import load_embeddings_from_npz

    p = write_npz({"a.b": np.ones(8, "float32")}, tmp_path, "pkg", "deadbeef")
    assert p.name == "pkg_deadbeef.npz"
    assert load_embeddings_from_npz(str(p))["a.b"].dtype == np.float32


def test_npz_roundtrip_with_kwarg_colliding_ids(tmp_path):
    from swegemma.graph.embedding_utils import load_embeddings_from_npz

    ids = ["file", "allow_pickle", "pkg.core.Thing"]
    vecs = {k: np.full(4, i, "float32") for i, k in enumerate(ids)}
    out = load_embeddings_from_npz(str(write_npz(vecs, tmp_path, "pkg", "abc")))
    assert set(out) == set(ids)
    for k in ids:
        assert out[k].dtype == np.float32 and np.array_equal(out[k], vecs[k])


def test_wrong_row_count_raises_value_error(tmp_path):
    class ShortEncoder(CountingFakeEncoder):
        def encode(self, texts):
            return super().encode(texts)[:-1]

    g = {"nodes": [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}], "edges": []}
    with pytest.raises(ValueError, match="shape"):
        embed_graph(g, ShortEncoder(4), Cache(tmp_path / "c.sqlite"))


def test_write_npz_atomic(tmp_path):
    import numpy as np
    from graphswap.embedder import write_npz

    path = write_npz({"a": np.ones(3, dtype=np.float32)}, tmp_path, "r", "abc")
    assert not list(tmp_path.glob("*.tmp"))
    with np.load(path) as d:
        assert d["a"].shape == (3,)
