"""graphswap command line."""
import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from graphswap.audit import embedding_geometry, gold_coverage, gold_symbols
from graphswap.builder import build_graph, write_graph
from graphswap.bundle import assemble, sha256_file
from graphswap.embedder import (DEFAULT_MODEL, Cache, SentenceTransformerEncoder,
                                embed_graph, write_npz)
from graphswap.replay import (DEFAULT_K, ROW_COLUMNS, SUMMARY_COLUMNS, SUMMARY_COMMENT, dedupe,
                              replay_run, summarize_replay)
from graphswap.repos import checkout
from graphswap.stats import noise_floor, paired
from graphswap.tasks import commits, draw_subset, load_tasks, read_id_list
from graphswap.traces import summarize_run, write_csv

logger = logging.getLogger(__name__)

BUILDER_VERSION = "graphswap-builder 0.1"
DEFAULT_CACHE = "cache/embeddings.sqlite"
BUNDLE_NAMES = ("emb_released_nodes", "graphs_rebuilt", "rebuilt_full")

GEOM_KEYS = ("mean_cos", "p05", "p95", "dup_rate", "eff_dim")
COMMIT_COLS = ["repo_short", "commit", "released_nodes", "rebuilt_nodes", "id_parity",
               "released_only", *(f"rel_{k}" for k in GEOM_KEYS),
               *(f"ours_{k}" for k in GEOM_KEYS)]
TASK_COLS = ["instance_id", "repo_short", "commit", "n_gold", "released_covered",
             "released_fraction", "rebuilt_covered", "rebuilt_fraction"]


def _geometry_cells(npz: Path, prefix: str) -> dict:
    if not npz.exists():
        return {}
    try:
        with np.load(npz, allow_pickle=False) as data:  # same format as the harness loader
            geo = embedding_geometry({k: data[k].astype(np.float32) for k in data.files})
    except (EOFError, ValueError, OSError, zlib.error, zipfile.BadZipFile) as exc:  # e.g. a download still in progress
        logger.warning("%s: unreadable embeddings, skipped (%s)", npz.name, exc)
        return {f"{prefix}_{k}": float("nan") for k in GEOM_KEYS}
    return {f"{prefix}_{k}": geo[v] for k, v in zip(
        GEOM_KEYS, ("mean_random_cosine", "p05", "p95", "duplicate_rate", "effective_dim"))}


def _write_csv(path: Path, cols: list[str], rows: list[dict], comment: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        if comment:
            f.write(comment + "\n")
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        # np.float32 is not a float subclass (np.float64 is), so match numpy floats too.
        w.writerows({k: ("" if v is None or (isinstance(v, (float, np.floating)) and v != v) else v)
                     for k, v in r.items()} for r in rows)


def audit(args: argparse.Namespace) -> None:
    tasks = load_tasks(Path(args.tasks))
    released, ours, repos = Path(args.released), Path(args.ours), Path(args.repos)
    out = Path(args.out)
    by_commit: dict[tuple[str, str], list] = {}
    for t in tasks:
        by_commit.setdefault((t.repo, t.base_commit), []).append(t)
    commit_rows, task_rows, written = [], [], False
    for i, ((repo, commit), group) in enumerate(sorted(by_commit.items()), 1):
        short = repo.split("/", 1)[1]
        name = f"{short}_{commit}"
        rebuilt_path = ours / "graphs_rebuilt" / f"{name}.json"
        if not rebuilt_path.exists():
            logger.warning("%s: no rebuilt graph, skipping", name)
            continue
        rebuilt = json.loads(rebuilt_path.read_text(encoding="utf-8"))
        rebuilt_ids = {n["id"] for n in rebuilt["nodes"]}
        rel_path = released / "graphs" / f"{name}.json"
        rel = None
        if rel_path.exists():
            try:
                rel = json.loads(rel_path.read_text(encoding="utf-8"))
            except ValueError as exc:  # truncated download
                logger.warning("%s: unreadable released graph (%s)", name, exc)
        else:
            logger.warning("%s: no released graph", name)
        row = {"repo_short": short, "commit": commit, "rebuilt_nodes": len(rebuilt_ids)}
        if rel is not None:
            rel_ids = {n["id"] for n in rel["nodes"]}
            row.update(released_nodes=len(rel_ids),
                       id_parity=len(rel_ids & rebuilt_ids) / len(rel_ids) if rel_ids else None,
                       released_only=len(rel_ids - rebuilt_ids))
        row.update(_geometry_cells(released / "embeddings" / f"{name}.npz", "rel"))
        row.update(_geometry_cells(ours / "embeddings" / f"{name}.npz", "ours"))
        commit_rows.append(row)  # geometry/parity do not depend on the checkout
        try:
            wt = checkout(repo, commit, repos)
            base = build_graph(wt, short)  # once per commit; shared by every task at this commit
            rows = []
            for t in group:
                gold = gold_symbols(t, wt, base)  # once per task, shared by both coverages
                r_cov = gold_coverage(t, wt, rel, base, gold=gold) if rel is not None else None
                o_cov = gold_coverage(t, wt, rebuilt, base, gold=gold)
                rows.append({
                    "instance_id": t.instance_id, "repo_short": short, "commit": commit,
                    "n_gold": None if gold is None else len(gold),
                    "released_covered": len(r_cov["covered"]) if r_cov else None,
                    "released_fraction": r_cov["fraction"] if r_cov else None,
                    "rebuilt_covered": len(o_cov["covered"]),
                    "rebuilt_fraction": o_cov["fraction"],
                })
        except Exception as exc:  # one broken checkout must not abort the run
            logger.warning("%s: audit failed (%s)", name, exc)
            rows = []
        task_rows.extend(rows)
        _write_csv(out, COMMIT_COLS, commit_rows)  # incremental: partial results survive an interrupt
        _write_csv(out.with_name("audit_tasks.csv"), TASK_COLS, task_rows)
        written = True
        print(f"[{i}/{len(by_commit)}] {name}", flush=True)
    if not written:  # nothing to audit, or every commit skipped: still write the empty CSVs
        _write_csv(out, COMMIT_COLS, commit_rows)
        _write_csv(out.with_name("audit_tasks.csv"), TASK_COLS, task_rows)


def build_graphs(args: argparse.Namespace) -> None:
    pairs = commits(load_tasks(Path(args.tasks)))
    out_dir, repos = Path(args.out), Path(args.repos)
    files: dict[str, dict] = {}
    for i, (repo, commit) in enumerate(pairs, 1):
        short = repo.split("/", 1)[1]
        path = out_dir / f"{short}_{commit}.json"
        if path.exists() and not args.force:
            graph, status = json.loads(path.read_text(encoding="utf-8")), "cached"
        else:
            graph = build_graph(checkout(repo, commit, repos), short)
            path, status = write_graph(graph, out_dir, short, commit), "built"
        files[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
        }
        print(
            f"[{i}/{len(pairs)}] {path.name} {status} "
            f"nodes={len(graph['nodes'])} edges={len(graph['edges'])}",
            flush=True,
        )
    manifest = {
        "builder_version": BUILDER_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def embed_graphs(args: argparse.Namespace) -> None:
    graphs = sorted(p for p in Path(args.graphs).glob("*.json") if p.name != "MANIFEST.json")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model_id": args.model, "max_tokens": args.max_tokens, "device": args.device}
    prev_path = out_dir / "MANIFEST.json"
    if not prev_path.exists() and not args.force and any(out_dir.glob("*.npz")):
        raise SystemExit(f"{out_dir} has npz files but no MANIFEST.json (unknown provenance); "
                         "pass --force to re-embed")
    if prev_path.exists() and not args.force:
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        old = {k: prev.get(k) for k in meta}
        if old != meta:
            raise SystemExit(f"{prev_path} was made with {old}, now asked for {meta}; "
                             "pass --force to re-embed")
    todo = {g for g in graphs if args.force or not (out_dir / f"{g.stem}.npz").exists()}
    if todo:
        encoder = SentenceTransformerEncoder(args.model, args.device, args.max_tokens)
        cache = Cache(Path(args.cache))
    for i, g in enumerate(graphs, 1):
        if g in todo:
            repo_short, commit = g.stem.rsplit("_", 1)
            vectors = embed_graph(json.loads(g.read_text(encoding="utf-8")), encoder, cache)
            write_npz(vectors, out_dir, repo_short, commit)
            status = f"embedded nodes={len(vectors)}"
        else:
            status = "cached"
        print(f"[{i}/{len(graphs)}] {g.stem}.npz {status}", flush=True)
    files = {p.name: {"sha256": sha256_file(p)} for p in sorted(out_dir.glob("*.npz"))}
    manifest = {**meta, "built_at": datetime.now(timezone.utc).isoformat(), "files": files}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def bundle(args: argparse.Namespace) -> None:
    out = Path(args.out)
    specs = {
        "emb_released_nodes": (None, Path(args.emb_released)),
        "graphs_rebuilt": (Path(args.graphs_rebuilt), None),
        "rebuilt_full": (Path(args.graphs_rebuilt), Path(args.emb_rebuilt)),
    }
    for name, (g, e) in specs.items():
        root = assemble(g, e, out, name)
        m = json.loads((root / "MANIFEST.json").read_text())
        print(f"{name}: files={len(m['files'])} bundle_sha={m['bundle_sha']}", flush=True)


def _kaggle(*args: str, check: bool = True) -> str:
    exe = shutil.which("kaggle") or str(Path.home() / "Library/Python/3.11/bin/kaggle")
    proc = subprocess.run([exe, "-W", *args], capture_output=True, text=True)
    out = "\n".join(l for l in (proc.stdout + proc.stderr).splitlines() if not l.startswith("Warning:"))
    if check and proc.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out}")
    return out


def _status_word(text: str) -> str:
    m = re.match(r"\s*(\w+)", text)
    return m.group(1).lower() if m else ""


def _dataset_fingerprint(slug: str) -> str:
    """Hash of the (name, size, date) listing of the dataset's files, paging at 200."""
    rows, token = [], None
    while True:
        extra = ["--page-token", token] if token else []
        out = _kaggle("datasets", "files", slug, "--page-size", "200", *extra, check=False)
        lines = out.splitlines()
        token = next((l.split("=", 1)[1].strip() for l in lines if l.startswith("Next Page Token")), "")
        sep = next((i for i, l in enumerate(lines) if l.startswith("----")), None)
        if sep is None:
            break
        rows += [l for l in lines[sep + 1:] if l.strip()]
        if not token:
            break
    return hashlib.sha256("\n".join(rows).encode()).hexdigest() if rows else ""


def push_bundles(args: argparse.Namespace) -> None:
    root = Path(args.bundles)
    manifests = {}
    for name in BUNDLE_NAMES:
        mp = root / name / "MANIFEST.json"
        if not mp.exists():
            raise SystemExit(f"missing {mp}; run `graphswap bundle` first")
        manifests[name] = json.loads(mp.read_text())
    if not args.private:
        raise SystemExit("refusing to push a public dataset; pass --private")
    slug = args.dataset
    meta = {"title": slug.split("/", 1)[1], "id": slug, "licenses": [{"name": "CC0-1.0"}]}
    (root / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    note = "; ".join(f"{n}={m['bundle_sha'][:12]}" for n, m in manifests.items())
    probe = _kaggle("datasets", "status", slug, check=False)
    exists = "403" not in probe and "404" not in probe and "not found" not in probe.lower()
    before = _dataset_fingerprint(slug) if exists else ""
    if exists:
        print(f"versioning private dataset {slug}", flush=True)
        _kaggle("datasets", "version", "-p", str(root), "-m", note, "--dir-mode", "tar")
    else:
        print(f"creating private dataset {slug}", flush=True)
        _kaggle("datasets", "create", "-p", str(root), "--dir-mode", "tar")
    deadline = time.time() + 60 * 60
    while True:
        time.sleep(30)
        status = _status_word(_kaggle("datasets", "status", slug, check=False))
        # The old version also reports "ready", so require the file listing to change too.
        if status == "ready" and _dataset_fingerprint(slug) not in (before, ""):
            break
        if time.time() > deadline:
            raise SystemExit(f"dataset not ready with a new version after 60 min (status: {status!r})")
        print(f"  dataset status: {status or '?'}; waiting...", flush=True)
    print(f"dataset {slug} ready", flush=True)
    for name, m in manifests.items():
        print(f"{name}: bundle_sha={m['bundle_sha']}", flush=True)


def summarize(args: argparse.Namespace) -> None:
    graphs = Path(args.graphs) if args.graphs else None
    rows = summarize_run(Path(args.results), load_tasks(Path(args.tasks)), graphs_dir=graphs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, Path(args.out))
    print(f"resolved: {sum(r['resolved'] for r in rows)}/{len(rows)}")
    for k in ("graph_tool_calls", "graph_tool_errors", "gold_file_read"):
        print(f"mean {k}: {sum(float(r[k]) for r in rows) / max(len(rows), 1):.3f}")
    if graphs is not None:
        known = [r for r in rows if r["search_resolved"] != ""]
        calls = sum(r["search_calls"] for r in known)
        resolved = sum(r["search_resolved"] for r in known)
        substring = sum(r["search_substring"] for r in known)
        print(f"search calls: {calls}, resolved {resolved} ({100 * resolved / max(calls, 1):.1f}%) "
              f"of which substring {substring}"
              + (f"; {len(rows) - len(known)} task(s) without a graph" if len(known) < len(rows) else ""))


def replay_search(args: argparse.Namespace) -> None:
    tasks = load_tasks(Path(args.tasks))
    rows = []
    for run in args.runs:
        rows += replay_run(Path(run), tasks, Path(args.graph_dir), Path(args.emb_a), Path(args.emb_b), k=args.k)
    out = Path(args.out)
    summary = summarize_replay(rows, "call")
    if args.dedupe:
        rows = dedupe(rows)
        by_query = {s["run"]: s for s in summarize_replay(rows, "query")}
        summary = [x for s in summary for x in (s, by_query[s["run"]])]
    _write_csv(out, ROW_COLUMNS, rows)
    _write_csv(out.with_suffix(".summary.csv"), SUMMARY_COLUMNS, summary, comment=SUMMARY_COMMENT)
    for s in summary:
        print(", ".join(f"{k}={s[k]:.4f}" if isinstance(s[k], float) else f"{k}={s[k]}"
                        for k in SUMMARY_COLUMNS), flush=True)


def compare(args: argparse.Namespace) -> None:
    read = lambda p: list(csv.DictReader(open(p, newline="")))
    control, treatment = read(args.control), read(args.treatment)
    result = (noise_floor if args.noise else paired)(control, treatment)
    width = max(map(len, result))
    for k, v in result.items():
        if k == "boot_ci":
            v = f"({v[0]}, {v[1]})"
        print(f"{k + ':':<{width + 1}} {v}")


def draw_subset_cmd(args: argparse.Namespace) -> None:
    tasks = load_tasks(Path(args.tasks))
    ids = draw_subset(tasks, args.n, args.seed, read_id_list(Path(args.include)))
    repo = {t.instance_id: t.repo_short for t in tasks}
    counts = {r: sum(repo[i] == r for i in ids) for r in sorted(set(repo[i] for i in ids))}
    header = (f"# n={args.n} seed={args.seed} stratified by repo "
              f"({', '.join(f'{r} {c}' for r, c in counts.items())})\n"
              f"# Includes every id in {Path(args.include).name}. Fixed: do not regenerate.\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(header + "\n".join(ids) + "\n")
    print(f"wrote {len(ids)} ids to {args.out}: {counts}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="graphswap")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build-graphs", help="build an AST graph per (repo, base_commit)")
    p.add_argument("--tasks", required=True)
    p.add_argument("--repos", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true", help="rebuild graphs that already exist")
    p.set_defaults(func=build_graphs)
    p = sub.add_parser("audit", help="graph/embedding audit statistics per commit and task")
    p.add_argument("--tasks", required=True)
    p.add_argument("--released", required=True, help="dir with graphs/ and embeddings/")
    p.add_argument("--ours", required=True, help="dir with graphs_rebuilt/ (and embeddings/)")
    p.add_argument("--repos", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=audit)
    p = sub.add_parser("embed-graphs", help="embed every graph in a directory into npz files")
    p.add_argument("--graphs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="default is the offline-comparison model; the paper's C2/C3 bundles use "
                        "Qwen/Qwen3-Embedding-0.6B at --max-tokens 1024")
    p.add_argument("--device", default="mps")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--force", action="store_true", help="re-embed npz files that already exist")
    p.set_defaults(func=embed_graphs)
    p = sub.add_parser("bundle", help="assemble the three bundles")
    p.add_argument("--graphs-rebuilt", default="bundles/graphs_rebuilt")
    p.add_argument("--emb-rebuilt", default="bundles/emb_rebuilt")
    p.add_argument("--emb-released", default="bundles/emb_released_nodes_raw")
    p.add_argument("--out", default="bundles/out")
    p.set_defaults(func=bundle)
    p = sub.add_parser("push-bundles", help="create or version the Kaggle dataset")
    p.add_argument("--bundles", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--private", action="store_true")
    p.set_defaults(func=push_bundles)
    p = sub.add_parser("summarize", help="per-task CSV for one harness run")
    p.add_argument("--results", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--graphs", help="dir of <repo_short>_<commit>.json graphs: classify search calls")
    p.set_defaults(func=summarize)
    p = sub.add_parser("replay-search", help="replay search_similar_code calls under two embedding sets")
    p.add_argument("--graph-dir", required=True, help="dir of <repo_short>_<commit>.json graphs")
    p.add_argument("--emb-a", required=True, help="dir of <repo_short>_<commit>.npz (set A)")
    p.add_argument("--emb-b", required=True, help="dir of <repo_short>_<commit>.npz (set B)")
    p.add_argument("--runs", nargs="+", required=True, help="harness run directories")
    p.add_argument("--out", required=True,
                   help="CSV of the replayed rows: one per search call, or, with --dedupe, the "
                        "deduplicated rows only; the per-run summary (a leading '#' comment line, "
                        "then CSV) goes to <out stem>.summary.csv")
    p.add_argument("--dedupe", action="store_true",
                   help="keep one row per (run, instance_id, query, k), the first occurrence; "
                        "adds unit=query summary rows next to the unit=call ones")
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help="k for calls that pass none (harness default %(default)s)")
    p.add_argument("--tasks", default="data/tasks.jsonl")
    p.set_defaults(func=replay_search)
    p = sub.add_parser("compare", help="paired statistics between two summary CSVs")
    p.add_argument("--control", required=True)
    p.add_argument("--treatment", required=True)
    p.add_argument("--noise", action="store_true", help="same-condition, different-seed noise floor")
    p.set_defaults(func=compare)
    p = sub.add_parser("draw-subset", help="stratified task subset containing a fixed include list")
    p.add_argument("--tasks", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=draw_subset_cmd)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
