#!/usr/bin/env python3
"""Drive eval runs on Kaggle from this machine.

Reference copy, not runnable from this repo: it needs kaggle/eval_kernel.py and submission/
from the authors' private workspace, which are not published.

This Mac cannot serve the model (vLLM is Linux/CUDA only), so every eval runs as
a private Kaggle notebook on L4x4. This tool is the round trip:

    push    upload submission/ as a private dataset (skipped if unchanged)
    run     render kaggle/eval_kernel.py for a fixed task subset and push it
    status  show the eval kernel's state and any unpulled run
    wait    block until the kernel stops running
    pull    download results into results/<run>/ and verify them

Each step guards one way the loop can silently lie to you:

  * `kernels output` only fetches the LATEST run, so `run` refuses while an
    earlier run is still unpulled -- otherwise its results are lost.
  * A dataset version still processing would mount the previous config. The
    kernel checks the content hash of what it mounted before touching a GPU,
    and `pull` checks it again.
  * Unknown task ids are silently dropped by the harness's filter, giving a
    smaller run than you think, so `run` validates them up front.
  * A hung run should not drain the week's quota, so every push carries a hard
    session timeout sized from the subset.
  * A condition (conditions/C*.yaml) changes what the agent sees without changing
    submission/, so it is validated here, rendered into the kernel, recorded in the
    run manifest, and checked again by `pull`.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "submission"
BUILD = ROOT / "build"
STATE = BUILD / "state.json"
RESULTS = ROOT / "results"
TASKS = ROOT / "data" / "tasks.jsonl"
KERNEL_SRC = ROOT / "kaggle" / "eval_kernel.py"
CONDITIONS = ROOT / "conditions"

sys.path.insert(0, str(KERNEL_SRC.parent))
from eval_kernel import BUNDLES_SLUG, _skip, _stub_tools, overlaid_texts, tree_hash  # noqa: E402  (shared with the kernel)
from flips import coverage_text, is_infra_error  # noqa: E402  (shared with flips)

COMPETITION = "gemma-4-developer-agent"
# Own slugs for the conditions study, so its runs never share a kernel (and its
# `kernels output`) or a submission dataset with the main-track checkout.
DATASET_SLUG = "gemma4-cond-submission"
KERNEL_SLUG = "gemma4-cond-eval"
WHEELHOUSE = "metric/gemma-4-developer-agent-wheelhouse"
# Pinned to the official starter's image: the wheelhouse's +cu128 wheels are
# built for it, and Kaggle's "latest" image may move underneath them.
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")
MACHINE_SHAPE = "NvidiaL4"
BUNDLES_DATASET = f"ibrahimkhalil09/{BUNDLES_SLUG}"   # private; attached only when a condition names a bundle
# Free-text query encoder for conditions with free_text_embed (C4). Until both are
# filled in, such a run is refused here and by the kernel.
EMBED_MODEL_SOURCE = ""   # Kaggle model handle, owner/model/framework/variation/version
EMBED_MODEL_DIR = ""      # where that model mounts on Kaggle, under /kaggle/input
QUOTA_MULTIPLIER = 2   # L4x4 burns GPU quota at 2x (competition "Upgraded Accelerators" page)
BUSY = {"RUNNING", "QUEUED", "NEW_SCRIPT", "CANCEL_REQUESTED"}


# ---------------------------------------------------------------- helpers

def kaggle_bin():
    found = shutil.which("kaggle")
    if found:
        return found
    fallback = Path.home() / "Library/Python/3.11/bin/kaggle"
    if fallback.exists():
        return str(fallback)
    raise SystemExit("kaggle CLI not found on PATH")


def kaggle(*args, check=True):
    proc = subprocess.run([kaggle_bin(), "-W", *args], capture_output=True, text=True)
    out = "\n".join(l for l in (proc.stdout + proc.stderr).splitlines() if not l.startswith("Warning:"))
    if check and proc.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out}")
    return out


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state):
    BUILD.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True))


def git(*args):
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def git_identity():
    sha = git("rev-parse", "--short", "HEAD") or "nogit"
    dirty = bool(git("status", "--porcelain", "--", "submission"))
    return sha, dirty


def declared_model():
    models = set()
    for path in SUBMISSION.rglob("*.y*ml"):
        for match in re.finditer(r"^\s*model:\s*(\S+)", path.read_text(), re.M):
            models.add(re.sub(r"^(openai|google|hosted_vllm|custom)/", "", match.group(1)))
    if len(models) != 1:
        raise SystemExit(f"expected exactly one declared model in submission/, found {sorted(models)}")
    return models.pop()


def max_time_minutes():
    text = (SUBMISSION / "eval_config.yaml").read_text() if (SUBMISSION / "eval_config.yaml").exists() else ""
    match = re.search(r"^\s*max_time_minutes:\s*([\d.]+)", text, re.M)
    return float(match.group(1)) if match else 60.0


def read_subset(path):
    ids = [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    if not ids:
        raise SystemExit(f"{path}: no task ids")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate task ids")
    if TASKS.exists():
        known = {json.loads(l)["instance_id"] for l in TASKS.read_text().splitlines() if l.strip()}
        unknown = sorted(set(ids) - known)
        if unknown:
            raise SystemExit(f"{path}: unknown task ids (the harness would silently drop them): {unknown}")
    else:
        print(f"note: {TASKS} missing -- task ids not validated (run `make subset` to fetch it)")
    return ids


AUTH_HELP = ("Kaggle authentication has expired or is missing (OAuth sessions are short-lived).\n"
             "  Re-login:            ~/Library/Python/3.11/bin/kaggle auth login\n"
             "  or, for unattended runs, put a long-lived API token in ~/.kaggle/access_token")


def ensure_auth():
    """Fail loudly on expired auth. Unauthenticated, the CLI falls back to anonymous
    access, and a private kernel then reports 'permission denied' -- which looks
    exactly like a kernel that doesn't exist."""
    out = kaggle("competitions", "submission-limits", COMPETITION, check=False)
    if "Authentication required" in out:
        raise SystemExit(AUTH_HELP)


def kernel_status(user):
    out = kaggle("kernels", "status", f"{user}/{KERNEL_SLUG}", check=False)
    if "Authentication required" in out:
        raise SystemExit(AUTH_HELP)
    match = re.search(r'KernelWorkerStatus\.(\w+)', out)
    if match:
        return match.group(1)
    if "Cannot access kernel" in out or "404" in out or "denied" in out:
        return "ABSENT"
    raise SystemExit(f"could not parse kernel status:\n{out}")


def timeout_seconds(n_tasks, concurrency):
    """Hard session cap: model startup + waves of capped tasks + slack, max 12h."""
    per_task = max_time_minutes() + 7          # agent cap + setup + verification
    minutes = 30 + math.ceil(n_tasks / concurrency) * per_task + 15
    return int(min(minutes, 12 * 60) * 60)


# ---------------------------------------------------------------- conditions

CONDITION_KEYS = {"analyzer_tools", "graph_bundle", "embeddings_bundle", "free_text_embed"}
GRAPH_TOOLS = {"search_similar_code", "get_code_neighbors", "get_code_subgraph"}
SYMBOLIC_DIRS = ("released", "none")


def match_condition(cid, stems):
    """The one condition file stem that `cid` names, compared case-insensitively.

    APFS is case-insensitive, so conditions/c1.yaml opens C1.yaml; matching against
    the directory listing instead makes `c1` render and record CONDITION = "C1"."""
    hits = [s for s in stems if s.lower() == str(cid).lower()] if re.fullmatch(r"[A-Za-z0-9_-]+", str(cid)) else []
    if len(hits) != 1:
        known = ", ".join(sorted(stems))
        why = f"ambiguous ({', '.join(sorted(hits))})" if hits else "unknown"
        raise SystemExit(f"{why} condition {cid!r}; known: {known}")
    return hits[0]


def canonical_condition(cid, root=CONDITIONS):
    """The condition id exactly as its conditions/<ID>.yaml file is named."""
    return match_condition(cid, [p.stem for p in Path(root).glob("*.yaml")])


def load_condition(cid, root=CONDITIONS):
    """conditions/<cid>.yaml, refusing anything that would render a plausible wrong run."""
    path = Path(root) / f"{canonical_condition(cid, root)}.yaml"
    cond = yaml.safe_load(path.read_text()) or {}
    if set(cond) != CONDITION_KEYS:
        raise SystemExit(f"{path}: keys must be exactly {sorted(CONDITION_KEYS)}; "
                         f"missing {sorted(CONDITION_KEYS - set(cond))}, unexpected {sorted(set(cond) - CONDITION_KEYS)}")
    tools = cond["analyzer_tools"]
    if not isinstance(tools, list) or not tools or len(set(tools)) != len(tools):
        raise SystemExit(f"{path}: analyzer_tools must be a non-empty list without duplicates")
    unknown = sorted(set(tools) - set(_stub_tools()))
    if unknown:
        raise SystemExit(f"{path}: analyzer_tools {unknown} are not harness tools ({sorted(_stub_tools())})")
    for key in ("graph_bundle", "embeddings_bundle"):
        if not isinstance(cond[key], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cond[key]):
            raise SystemExit(f"{path}: {key} must be released, none, or a bundle name; got {cond[key]!r}")
    if not isinstance(cond["free_text_embed"], bool):
        raise SystemExit(f"{path}: free_text_embed must be true or false; got {cond['free_text_embed']!r}")
    if GRAPH_TOOLS & set(tools) and "none" in (cond["graph_bundle"], cond["embeddings_bundle"]):
        raise SystemExit(f"{path}: the analyzer has graph tools but a bundle is none -- the harness would "
                         f"withhold them silently")
    if cond["free_text_embed"] and ("search_similar_code" not in tools or cond["embeddings_bundle"] == "none"):
        raise SystemExit(f"{path}: free_text_embed needs search_similar_code and an embeddings bundle")
    return cond


def uses_bundles(cond):
    return any(cond[k] not in SYMBOLIC_DIRS for k in ("graph_bundle", "embeddings_bundle"))


def render_config(*, run, task_ids, concurrency, digest, git_sha, condition, seed):
    """The kernel's RENDERED CONFIG block for one run under one condition."""
    condition = canonical_condition(condition)
    cond = load_condition(condition)
    return "\n".join([
        "# ==== RENDERED CONFIG (rewritten by tools/kaggle_eval.py) ====",
        f"RUN_NAME = {json.dumps(run)}",
        f"TASK_IDS = {json.dumps(task_ids)}",
        f"CONCURRENCY = {int(concurrency)}",
        f"EXPECTED_TREE_HASH = {json.dumps(digest)}",
        f"GIT_SHA = {json.dumps(git_sha)}",
        f"DATASET_SLUG = {json.dumps(DATASET_SLUG)}",
        f"CONDITION = {json.dumps(condition)}",
        f"GRAPH_DIR = {json.dumps(cond['graph_bundle'])}",
        f"EMB_DIR = {json.dumps(cond['embeddings_bundle'])}",
        f"ANALYZER_TOOLS = {json.dumps(cond['analyzer_tools'])}",
        f"SEED = {int(seed)}",
        f"FREE_TEXT_EMBED = {cond['free_text_embed']!r}",
        f"EMBED_MODEL_DIR = {json.dumps(EMBED_MODEL_DIR)}",
        "# ==== END RENDERED CONFIG ====",
    ])


def render_kernel(config_block):
    """kaggle/eval_kernel.py with its RENDERED CONFIG block replaced."""
    source, n = re.subn(r"# ==== RENDERED CONFIG.*?# ==== END RENDERED CONFIG ====", lambda _: config_block,
                        KERNEL_SRC.read_text(), flags=re.S)
    if n != 1:
        raise SystemExit("could not find the RENDERED CONFIG block in kaggle/eval_kernel.py")
    return source


def kernel_metadata(user, model, cond):
    """kernel-metadata.json: the bundles dataset and the free-text encoder are inputs
    only for the conditions that use them, so C1 pushes exactly what it always did."""
    datasets = [WHEELHOUSE, f"{user}/{DATASET_SLUG}"] + ([BUNDLES_DATASET] if uses_bundles(cond) else [])
    models = [f"google/gemma-4/Other/{model}/2"]
    if cond["free_text_embed"] and EMBED_MODEL_SOURCE:
        models.append(EMBED_MODEL_SOURCE)
    return {
        "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG,
        "code_file": "eval_kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False,          # required for L4 sessions; the CLI defaults it to True
        "dataset_sources": datasets,
        "competition_sources": [COMPETITION], "kernel_sources": [],
        "model_sources": models,
        "docker_image": DOCKER_IMAGE, "docker_image_pinning_type": "original",
        "machine_shape": MACHINE_SHAPE,
    }


# ---------------------------------------------------------------- commands

def cmd_push(args, quiet=False):
    ensure_auth()
    state = load_state()
    digest = tree_hash(SUBMISSION)
    if state.get("pushed_tree_hash") == digest and not args.force:
        if not quiet:
            print(f"submission unchanged since last push ({digest[:12]}); nothing to upload")
        return digest

    stage = BUILD / "dataset"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    with zipfile.ZipFile(stage / "submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in SUBMISSION.rglob("*") if p.is_file()):
            rel = path.relative_to(SUBMISSION)
            if not _skip(rel.parts):
                archive.write(path, rel.as_posix())
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_SLUG, "id": f"{args.user}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

    sha, dirty = git_identity()
    note = f"{sha}{'-dirty' if dirty else ''} tree={digest[:12]}"
    exists = "403" not in kaggle("datasets", "status", f"{args.user}/{DATASET_SLUG}", check=False)
    if exists:
        print(f"versioning private dataset {args.user}/{DATASET_SLUG} ({note})")
        kaggle("datasets", "version", "-p", str(stage), "-m", note)
    else:
        print(f"creating private dataset {args.user}/{DATASET_SLUG} ({note})")
        kaggle("datasets", "create", "-p", str(stage))

    time.sleep(20)   # let the new version register before polling, so "ready" isn't the old one
    deadline = time.time() + 15 * 60
    while True:
        status = kaggle("datasets", "status", f"{args.user}/{DATASET_SLUG}", check=False).strip().lower()
        if "ready" in status:
            break
        if time.time() > deadline:
            raise SystemExit(f"dataset still not ready after 15 min (status: {status!r})")
        print(f"  dataset status: {status or '?'}; waiting...")
        time.sleep(15)

    state["pushed_tree_hash"] = digest
    state["pushed_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    save_state(state)
    print(f"dataset ready; submission tree {digest[:12]}")
    return digest


def cmd_run(args):
    args.condition = canonical_condition(args.condition)   # recorded and rendered as the file names it
    cond = load_condition(args.condition)
    if cond["free_text_embed"] and not (EMBED_MODEL_DIR and EMBED_MODEL_SOURCE) and not args.dry_run:
        raise SystemExit(f"condition {args.condition} embeds free-text queries, but EMBED_MODEL_DIR / "
                         f"EMBED_MODEL_SOURCE in tools/kaggle_eval.py are not set; the kernel would refuse it")
    # The kernel applies these overlays after verifying the pushed tree: fail here, not on a GPU.
    overlays = [rel for rel, text in overlaid_texts(SUBMISSION, args.seed, cond["analyzer_tools"]).items()
                if (SUBMISSION / rel).read_text(encoding="utf-8") != text]
    ensure_auth()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run):
        raise SystemExit(f"bad run name {args.run!r}: use letters, digits, . _ -")
    if (RESULTS / args.run).exists() and not args.force:
        raise SystemExit(f"results/{args.run} already exists; pick a new RUN name (or --force)")

    state = load_state()
    pending = state.get("pending_run")
    if pending and not args.force:
        raise SystemExit(f"run {pending!r} has not been pulled yet. `kernels output` only fetches the "
                         f"latest run, so starting another would lose it. `make pull RUN={pending}` first.")
    status = kernel_status(args.user)
    if status in BUSY:
        raise SystemExit(f"eval kernel is {status}; wait for it (make wait) before starting another run")

    ids = read_subset(args.subset)
    model = declared_model()
    digest = tree_hash(SUBMISSION) if args.dry_run else cmd_push(args, quiet=True)
    sha, dirty = git_identity()
    if dirty:
        print("warning: submission/ has uncommitted changes -- this run won't map to a commit")

    source = render_kernel(render_config(
        run=args.run, task_ids=ids, concurrency=args.concurrency, digest=digest,
        git_sha=sha + ('-dirty' if dirty else ''), condition=args.condition, seed=args.seed))

    kdir = BUILD / "kernel"
    if kdir.exists():
        shutil.rmtree(kdir)
    kdir.mkdir(parents=True)
    (kdir / "eval_kernel.py").write_text(source)
    (kdir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(args.user, model, cond), indent=2))

    timeout = timeout_seconds(len(ids), args.concurrency)
    bundles = f", bundles from {BUNDLES_DATASET}" if uses_bundles(cond) else ""
    print(f"pushing run {args.run!r}: condition {args.condition} (graphs {cond['graph_bundle']}, embeddings "
          f"{cond['embeddings_bundle']}{bundles}), seed {args.seed}")
    print(f"  overlays on Kaggle: {', '.join(overlays) or 'none -- submission/ runs exactly as committed'}")
    print(f"  {len(ids)} tasks, concurrency {args.concurrency}, model {model}, "
          f"submission {digest[:12]}, hard timeout {timeout / 3600:.1f} h "
          f"(<= {timeout / 3600 * QUOTA_MULTIPLIER:.1f} GPU-h quota)")
    if args.dry_run:
        print(f"dry run: rendered {kdir} -- nothing uploaded, nothing pushed")
        return
    print(kaggle("kernels", "push", "-p", str(kdir), "-t", str(timeout)))

    record = {
        "run": args.run, "tree_hash": digest, "git_sha": sha, "dirty": dirty,
        "subset_file": str(Path(args.subset)), "task_ids": ids,
        "subset_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "concurrency": int(args.concurrency), "model": model, "timeout_seconds": timeout,
        "condition": args.condition, "seed": args.seed,
        "pushed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (BUILD / "runs").mkdir(parents=True, exist_ok=True)
    (BUILD / "runs" / f"{args.run}.json").write_text(json.dumps(record, indent=2))
    state = load_state()
    state["pending_run"] = args.run
    save_state(state)
    print(f"queued. Next: make wait && make pull RUN={args.run}")


def cmd_status(args):
    ensure_auth()
    status = kernel_status(args.user)
    pending = load_state().get("pending_run")
    print(f"kernel {args.user}/{KERNEL_SLUG}: {status}")
    print(f"unpulled run: {pending or 'none'}")
    if pending:
        record = json.loads((BUILD / "runs" / f"{pending}.json").read_text())
        print(f"  pushed {record['pushed_at']}: {len(record['task_ids'])} tasks, "
              f"concurrency {record['concurrency']}, timeout {record['timeout_seconds'] / 3600:.1f} h")


def cmd_wait(args):
    ensure_auth()
    pending = load_state().get("pending_run")
    began = time.time()
    while True:
        status = kernel_status(args.user)
        if status == "ABSENT":
            if pending:
                ensure_auth()   # an expired session is the usual cause
                raise SystemExit(f"cannot see the eval kernel although run {pending!r} was pushed to it. "
                                 f"Not treating that as finished; check kaggle.com/code/{args.user}/{KERNEL_SLUG}")
            print("no eval kernel exists yet and no run is pending; nothing to wait for")
            return
        if status not in BUSY:
            print(f"kernel finished with status {status} after {(time.time() - began) / 60:.0f} min of waiting")
            return
        print(f"  {time.strftime('%H:%M')} {status} ({(time.time() - began) / 60:.0f} min)", flush=True)
        time.sleep(args.interval)


def cmd_pull(args):
    state = load_state()
    run = args.run or state.get("pending_run")
    if not run:
        raise SystemExit("no run to pull; pass RUN=<name>")
    record_path = BUILD / "runs" / f"{run}.json"
    if not record_path.exists():
        raise SystemExit(f"no local record for run {run!r} (expected {record_path})")
    record = json.loads(record_path.read_text())

    ensure_auth()
    status = kernel_status(args.user)
    if status in BUSY:
        raise SystemExit(f"kernel is still {status}; `make wait` first")
    if status == "ABSENT":
        raise SystemExit(f"cannot see the eval kernel {args.user}/{KERNEL_SLUG}; nothing to pull")

    dest = RESULTS / run
    download = dest / "_download"
    if download.exists():
        shutil.rmtree(download)
    download.mkdir(parents=True)
    kaggle("kernels", "output", f"{args.user}/{KERNEL_SLUG}", "-p", str(download), "-o", "-q")

    manifests = sorted(download.rglob("run_manifest.json"))
    if not manifests:
        raise SystemExit(f"no run_manifest.json in the kernel output (status {status}). The kernel may have "
                         f"died before writing anything; check its log on kaggle.com. Raw output: {download}")
    produced = manifests[0].parent
    for item in produced.iterdir():
        target = dest / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(item), str(target))
    log_files = [p for p in download.rglob("*") if p.is_file()]
    for p in log_files:   # keep the kernel's own log next to the results
        shutil.move(str(p), str(dest / p.name))
    shutil.rmtree(download, ignore_errors=True)

    manifest = json.loads((dest / "run_manifest.json").read_text())
    problems = []
    if manifest.get("run") != run:
        problems.append(f"output belongs to run {manifest.get('run')!r}, not {run!r} -- a later push "
                        f"overwrote it; this run's results are gone")
    if manifest.get("tree_hash") != record["tree_hash"]:
        problems.append(f"evaluated submission {str(manifest.get('tree_hash'))[:12]} but you pushed "
                        f"{record['tree_hash'][:12]}")
    for key in ("condition", "seed"):
        if key in record and manifest.get(key) != record[key]:
            problems.append(f"evaluated {key} {manifest.get(key)!r} but you pushed {record[key]!r}")
    if manifest.get("status") != "complete":
        problems.append(f"run status is {manifest.get('status')!r}: {manifest.get('error', 'no error recorded')}")
    jsonl = dest / "task_results.jsonl"
    rows = [json.loads(l) for l in jsonl.open() if l.strip()] if jsonl.exists() else []
    done = len(rows)
    if done != len(record["task_ids"]):
        problems.append(f"only {done}/{len(record['task_ids'])} tasks have results")
    # An infrastructure error scores 0 no matter how good the agent is; a run where
    # every task hit one looks exactly like an agent that solved nothing. Budget
    # exhaustion is different: the scorer applies the same caps, so it's a result.
    infra = [r for r in rows if is_infra_error(r.get("error"))]
    if infra:
        kinds = sorted({str(r["error"])[:200] for r in infra})
        problems.append(f"{len(infra)}/{done} tasks hit INFRASTRUCTURE errors -- not agent results. "
                        f"Distinct errors: " + " || ".join(kinds[:3]))
    silent = [r["instance_id"] for r in rows if not r.get("error") and r.get("total_llm_calls") == 0]
    if silent:
        problems.append(f"{len(silent)} tasks finished without a single model call: {silent[:5]}")
    outcomes = {}
    for r in rows:
        if r.get("error") and not is_infra_error(r["error"]):
            key = re.sub(r"\s*\(.*?\)|:.*$", "", str(r["error"])).strip()
            outcomes[key] = outcomes.get(key, 0) + 1
    call_cap = (manifest.get("budgets") or {}).get("max_tool_calls")
    at_call_cap = sum(1 for r in rows if call_cap and (r.get("tool_calls") or 0) >= call_cap)

    if state.get("pending_run") == run:
        state.pop("pending_run")
    save_state(state)

    wall = manifest.get("wall_seconds") or 0
    print(f"run {run}: {manifest.get('resolved', '?')} resolved, {done}/{len(record['task_ids'])} tasks, "
          f"concurrency {manifest.get('concurrency')}, model {manifest.get('model')}, "
          f"condition {manifest.get('condition')}, seed {manifest.get('seed')}")
    print(f"  vLLM startup {manifest.get('vllm_startup_seconds', '?')} s, wall {wall / 3600:.2f} h "
          f"(~{wall / 3600 * QUOTA_MULTIPLIER:.1f} GPU-h quota), submission {str(manifest.get('tree_hash'))[:12]}")
    if "condition" in manifest:
        print(f"  graph tools offered on {coverage_text(manifest) or '(not recorded)'} tasks; "
              f"graphs {str(manifest.get('graph_bundle'))} ({str(manifest.get('graph_bundle_sha'))[:12]}), "
              f"embeddings {str(manifest.get('embeddings_bundle'))} ({str(manifest.get('embeddings_bundle_sha'))[:12]})")
    pressure = [f"{n} {k.lower()}" for k, n in sorted(outcomes.items())]
    if call_cap:
        pressure.append(f"{at_call_cap} used all {call_cap} tool calls")
    if pressure:
        print(f"  budget pressure: {'; '.join(pressure)}  (agent outcomes -- the scorer applies the same caps)")
    print(f"  -> {dest}")
    if problems:
        print("\nPROBLEMS -- do not compare this run until these are understood:")
        for p in problems:
            print(f"  ! {p}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", default=os.environ.get("KAGGLE_USER", "ibrahimkhalil09"))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("push"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--run", required=True)
    p.add_argument("--subset", default=str(ROOT / "eval" / "subset.txt"))
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--condition", default="C1", help="conditions/<id>.yaml (default C1: the released data)")
    p.add_argument("--seed", type=int, default=1234, help="sampling seed written into configs/sampling.yaml")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="render the kernel locally; upload and push nothing")
    sub.add_parser("status")
    p = sub.add_parser("wait"); p.add_argument("--interval", type=int, default=60)
    p = sub.add_parser("pull"); p.add_argument("--run")

    args = parser.parse_args()
    {"push": cmd_push, "run": cmd_run, "status": cmd_status, "wait": cmd_wait, "pull": cmd_pull}[args.command](args)


if __name__ == "__main__":
    main()
