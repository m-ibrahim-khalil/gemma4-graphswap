"""Encode N node texts from a graph with a real model and print throughput."""
import argparse
import json
import time

from graphswap.embedder import SentenceTransformerEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("-n", type=int, default=100)
    a = ap.parse_args()
    nodes = json.load(open(a.graph))["nodes"]
    texts = [n["text"] if n["text"].strip() else n["id"] for n in nodes][: a.n]
    enc = SentenceTransformerEncoder(a.model, a.device)
    enc.encode(texts[:4])  # warm-up
    t = time.perf_counter()
    v = enc.encode(texts)
    dt = time.perf_counter() - t
    print(f"model={a.model} device={a.device} texts={len(texts)} dim={v.shape[1]} "
          f"dtype={v.dtype} finite={bool((abs(v) < 1e30).all())} "
          f"{len(texts) / dt:.1f} texts/s ({dt:.2f}s)")


if __name__ == "__main__":
    main()
