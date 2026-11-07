import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from graphswap import cli
from graphswap.bundle import assemble


def _dirs(tmp_path):
    g, e = tmp_path / "g", tmp_path / "e"
    g.mkdir(), e.mkdir()
    (g / "a_1.json").write_text('{"nodes": []}')
    (e / "a_1.npz").write_bytes(b"npz-bytes")
    (e / "MANIFEST.json").write_text(json.dumps({"model_id": "m", "max_tokens": 8, "device": "cpu"}))
    return g, e


def test_assemble_layout_and_sha_changes(tmp_path):
    g, e = _dirs(tmp_path)
    b = assemble(g, e, tmp_path / "out", "full")
    assert (b / "graphs" / "a_1.json").exists() and (b / "embeddings" / "a_1.npz").exists()
    assert not (b / "embeddings" / "MANIFEST.json").exists()
    m = json.loads((b / "MANIFEST.json").read_text())
    assert set(m["files"]) == {"graphs/a_1.json", "embeddings/a_1.npz"}
    assert (m["model_id"], m["max_tokens"], m["device"]) == ("m", 8, "cpu")
    (e / "a_1.npz").write_bytes(b"changed")
    b2 = assemble(g, e, tmp_path / "out", "full")
    assert json.loads((b2 / "MANIFEST.json").read_text())["bundle_sha"] != m["bundle_sha"]


def test_assemble_single_kind(tmp_path):
    g, e = _dirs(tmp_path)
    b = assemble(g, None, tmp_path / "out", "x")
    assert (b / "graphs").is_dir() and not (b / "embeddings").exists()
    b = assemble(None, e, tmp_path / "out", "emb")
    assert (b / "embeddings").is_dir() and not (b / "graphs").exists()


class FakeEncoder:
    model_id = "fake"

    def __init__(self, *a, **k):
        pass

    def encode(self, texts):
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def test_embed_graphs_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "SentenceTransformerEncoder", FakeEncoder)
    gdir = tmp_path / "graphs"
    gdir.mkdir()
    for name in ("r_aaa", "r_bbb"):
        (gdir / f"{name}.json").write_text(json.dumps(
            {"nodes": [{"id": "n1", "text": "hello"}, {"id": "n2", "text": "x"}], "edges": []}))
    (gdir / "MANIFEST.json").write_text("{}")
    out = tmp_path / "emb"
    argv = ["embed-graphs", "--graphs", str(gdir), "--out", str(out), "--model", "fake",
            "--device", "cpu", "--max-tokens", "16", "--cache", str(tmp_path / "c.sqlite")]
    cli.main(argv)
    assert sorted(p.name for p in out.glob("*.npz")) == ["r_aaa.npz", "r_bbb.npz"]
    with np.load(out / "r_aaa.npz") as d:
        assert sorted(d.files) == ["n1", "n2"] and d["n1"].dtype == np.float32
    m = json.loads((out / "MANIFEST.json").read_text())
    assert (m["model_id"], m["max_tokens"], m["device"]) == ("fake", 16, "cpu")
    assert set(m["files"]) == {"r_aaa.npz", "r_bbb.npz"}
    mtime = (out / "r_aaa.npz").stat().st_mtime_ns
    cli.main(argv)  # skip existing
    assert (out / "r_aaa.npz").stat().st_mtime_ns == mtime


def test_push_bundles_requires_manifests(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["push-bundles", "--bundles", str(tmp_path), "--dataset", "u/s", "--private"])


def test_embed_graphs_refuses_different_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "SentenceTransformerEncoder", FakeEncoder)
    gdir = tmp_path / "graphs"
    gdir.mkdir()
    (gdir / "r_aaa.json").write_text(json.dumps({"nodes": [{"id": "n1", "text": "hi"}], "edges": []}))
    out = tmp_path / "emb"
    argv = ["embed-graphs", "--graphs", str(gdir), "--out", str(out), "--model", "fake",
            "--device", "cpu", "--max-tokens", "16", "--cache", str(tmp_path / "c.sqlite")]
    cli.main(argv)
    changed = [*argv[:-6], "--device", "cpu", "--max-tokens", "32", "--cache", str(tmp_path / "c.sqlite")]
    changed[changed.index("--max-tokens") + 1] = "32"
    with pytest.raises(SystemExit, match="max_tokens"):
        cli.main(changed)
    assert json.loads((out / "MANIFEST.json").read_text())["max_tokens"] == 16
    cli.main([*changed, "--force"])
    assert json.loads((out / "MANIFEST.json").read_text())["max_tokens"] == 32
    (out / "MANIFEST.json").unlink()
    with pytest.raises(SystemExit, match="provenance"):
        cli.main(changed)


def test_push_bundles_waits_for_new_version(tmp_path, monkeypatch):
    for name in cli.BUNDLE_NAMES:
        (tmp_path / name).mkdir()
        (tmp_path / name / "MANIFEST.json").write_text(json.dumps({"bundle_sha": name}))
    listing = {"v": "old"}
    calls = {"status": 0}

    def fake(*args, check=True):
        if args[:2] == ("datasets", "status"):
            calls["status"] += 1
            if calls["status"] == 1:  # existence probe
                return "ready"
            if calls["status"] > 4:
                listing["v"] = "new"  # new version registers after a few polls
            return "ready"  # the old version reports ready all along
        if args[:2] == ("datasets", "files"):
            return f"Next Page Token = \nname size date\n---- ---- ----\na {listing['v']} 1\n"
        return ""

    monkeypatch.setattr(cli, "_kaggle", fake)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    cli.push_bundles(argparse.Namespace(bundles=str(tmp_path), dataset="u/s", private=True))
    assert calls["status"] > 4 and listing["v"] == "new"


def test_status_word_is_exact():
    assert cli._status_word("ready") == "ready"
    assert cli._status_word("already exists") == "already"
    assert cli._status_word("403 Client Error") == "403"
