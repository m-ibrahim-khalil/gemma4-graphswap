"""Node embedding with a sqlite cache and harness-compatible npz output."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import zipfile
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "google/embeddinggemma-300m"

# Document-side prompt per model; unknown models get no prefix.
DOCUMENT_PROMPTS = {
    "google/embeddinggemma-300m": "title: none | text: ",
    "Qwen/Qwen3-Embedding-0.6B": "",
}


class Encoder(Protocol):
    model_id: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return float32 array of shape (len(texts), d)."""
        ...


class SentenceTransformerEncoder:
    def __init__(self, model_id: str, device: str, max_tokens: int = 1024):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.prefix = DOCUMENT_PROMPTS.get(model_id, "")
        self.model = SentenceTransformer(model_id, device=device)
        self.model.max_seq_length = max_tokens

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            [self.prefix + t for t in texts],
            batch_size=32,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


class Cache:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS emb "
            "(key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)"
        )
        self.db.commit()

    @staticmethod
    def key(model_id: str, text: str) -> str:
        return hashlib.sha256((model_id + "\0" + text).encode()).hexdigest()

    def get(self, key: str) -> np.ndarray | None:
        row = self.db.execute("SELECT dim, vec FROM emb WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[1], dtype=np.float32).reshape(row[0]).copy()

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?, ?)",
            [(k, int(v.shape[0]), v.astype(np.float32).tobytes()) for k, v in items.items()],
        )
        self.db.commit()


def embed_graph(graph: dict, encoder: Encoder, cache: Cache) -> dict[str, np.ndarray]:
    texts = {}
    for n in graph["nodes"]:
        text = n.get("text") or ""
        texts[n["id"]] = text if text.strip() else n["id"]

    keys = {nid: Cache.key(encoder.model_id, t) for nid, t in texts.items()}
    vectors: dict[str, np.ndarray] = {}
    misses: dict[str, str] = {}  # key -> text, deduplicated
    for nid, key in keys.items():
        vec = cache.get(key)
        if vec is None:
            misses[key] = texts[nid]
        else:
            vectors[nid] = vec

    if misses:
        miss_keys = list(misses)
        encoded = np.asarray(encoder.encode([misses[k] for k in miss_keys]), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != len(miss_keys):
            raise ValueError(
                f"encoder returned shape {encoded.shape} for {len(miss_keys)} texts"
            )
        by_key = dict(zip(miss_keys, encoded))
        for nid, key in keys.items():
            if nid not in vectors:
                if not np.isfinite(by_key[key]).all():
                    raise ValueError(f"non-finite embedding for node {nid}")
                vectors[nid] = by_key[key]
        cache.put_many(by_key)
    return {nid: vectors[nid] for nid in texts}


def write_npz(vectors: dict[str, np.ndarray], out_dir: Path, repo_short: str, commit: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{repo_short}_{commit}.npz"
    # Written entry by entry (the layout np.load reads) so node ids such as
    # "file" cannot collide with np.savez keyword parameters.
    tmp = path.with_name(path.name + ".tmp")  # atomic: interrupted runs leave no partial npz
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for key, vec in vectors.items():
            with zf.open(f"{key}.npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(
                    f, np.asarray(vec, dtype=np.float32), allow_pickle=False
                )
    os.replace(tmp, path)
    return path
