"""Assemble publishable bundles (graphs and/or embeddings) with a hashed manifest."""
import hashlib
import json
import os
import shutil
from pathlib import Path

EMBED_META_KEYS = ("model_id", "max_tokens", "device")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_sha(file_hashes: dict[str, str]) -> str:
    return hashlib.sha256("".join(sorted(file_hashes.values())).encode()).hexdigest()


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def assemble(graph_dir: Path | None, emb_dir: Path | None, out: Path, name: str) -> Path:
    """Create out/<name>/{graphs,embeddings}/ plus MANIFEST.json; returns the bundle dir."""
    root = Path(out) / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    files: dict[str, str] = {}
    meta: dict = {}
    for sub, src, pattern in (("graphs", graph_dir, "*.json"), ("embeddings", emb_dir, "*.npz")):
        if src is None:
            continue
        dest = root / sub
        dest.mkdir()
        for f in sorted(Path(src).glob(pattern)):
            if f.name == "MANIFEST.json":
                continue
            _link(f, dest / f.name)
            files[f"{sub}/{f.name}"] = sha256_file(dest / f.name)
    if emb_dir is not None and (Path(emb_dir) / "MANIFEST.json").exists():
        src_manifest = json.loads((Path(emb_dir) / "MANIFEST.json").read_text())
        meta = {k: src_manifest.get(k) for k in EMBED_META_KEYS}
    manifest = {"name": name, **meta, "files": files, "bundle_sha": bundle_sha(files)}
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root
