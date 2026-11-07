"""Analysis for the graph resolution study: every number and figure the writeup quotes.

Reads the per-run CSVs in ``<results>/runs/`` and the audit CSVs in ``<results>/``, and
writes ``<results>/summary/*.csv``, ``<results>/summary/tables.md`` and the figures.
Runs are discovered from the files; phase comes from the ``run`` prefix (``A_``, ``B_``),
condition and seed from their columns, so new conditions and phases need no code change.

    uv run python notebooks/analysis.py [--results DIR] [--figures DIR] [--strict]

A partial design (a phase where C1 lacks a seed, a run whose seed C1 lacks) is noted,
skipped and marked ``incomplete`` in the outputs; ``--strict`` raises instead and is
meant for the final paper regeneration.

Statistics come from the pre-registered module ``graphswap.stats``; the control
condition is C1 and the noise floor of a phase is C1 seed a vs C1 seed b. The headline
also carries each phase's minimum detectable effect under the pre-registered rule
(``mde_discordant``), computed from that phase's noise floor and pooled row count.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

from graphswap.stats import _exact_two_sided, noise_floor, paired

ROOT = Path(__file__).resolve().parent.parent
CONTROL = "C1"
ALPHA = 0.05
BOOL_COLS = ("resolved", "patch_submitted", "gold_file_read", "gold_file_edited")
NUM_COLS = (
    "tool_calls", "graph_tool_calls", "graph_tool_errors", "analyzer_calls",
    "wall_s", "prompt_tokens", "completion_tokens",
)
RUN_COLS = ("instance_id", "run", "condition", "seed") + BOOL_COLS + NUM_COLS
PHASE_RE = re.compile(r"^([A-Z])_")

STAT_COLS = [
    "n", "n_clusters", "control_rate", "treatment_rate", "diff",
    "wins", "losses", "ties", "mcnemar_p", "ci_lo", "ci_hi",
]
COMPARISON_COLS = ["phase", "condition", "scope"] + STAT_COLS + ["beyond_noise", "incomplete"]
NOISE_COLS = [
    "phase", "seed_a", "seed_b", "n", "rate_a", "rate_b", "diff",
    "discordant", "wins", "losses", "ties", "mcnemar_p", "ci_lo", "ci_hi",
]
PER_CONDITION_COLS = [
    "phase", "condition", "seed", "n", "resolved", "resolved_rate", "patch_submitted_rate",
    "mean_tool_calls", "mean_graph_tool_calls", "graph_tool_errors_total",
    "mean_analyzer_calls", "gold_file_read_rate", "gold_file_edited_rate",
    "mean_wall_s", "mean_prompt_tokens", "mean_completion_tokens", "incomplete",
]
GEOMETRY_COLS = ["group", "quantity", "n", "mean", "min", "max"]
# vector set -> (audit file, column prefix); released and qwen3 come from audit.csv.
VECTOR_SETS = {
    "released": ("audit.csv", "rel_"),
    "qwen3": ("audit.csv", "ours_"),
    "gemma": ("audit_gemma.csv", "gemma_"),
}
GEOMETRY_QUANTITIES = ("mean_cos", "dup_rate", "eff_dim")
FIGURES = ("fig_tool_profile", "fig_geometry")  # plus fig_solve_matrix_<phase>, one per phase


# ---------------------------------------------------------------- loading


def natural_key(text) -> tuple:
    """Sort key that orders C2 before C10 and seed 999 before 1234."""
    return tuple(
        (0, int(part), "") if part.isdigit() else (1, 0, part)
        for part in re.split(r"(\d+)", str(text))
        if part
    )


def load_runs(results: Path) -> list[dict]:
    """All rows of ``results/runs/*.csv`` as csv.DictReader dicts, plus a ``phase`` key.

    Runs whose name has no single-letter phase prefix (e.g. smoke runs) are skipped
    and named on stdout.
    """
    rows, skipped = [], set()
    paths = sorted(Path(results, "runs").glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no run CSVs in {Path(results, 'runs')}")
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            missing = set(RUN_COLS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
            for row in reader:
                m = PHASE_RE.match(row["run"])
                if not m:
                    skipped.add(row["run"])
                    continue
                row["phase"] = m.group(1)
                rows.append(row)
    if skipped:
        print(f"skipped runs without a phase prefix: {sorted(skipped)}")
    return rows


def group_runs(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    """Rows keyed by (phase, condition, seed), in phase/condition/seed order.

    Two runs for the same key (e.g. a crashed run and its rerun) raise: the analysis
    must not silently pick one.
    """
    groups: dict[tuple[str, str, str], list[dict]] = {}
    names: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        key = (row["phase"], row["condition"], row["seed"])
        groups.setdefault(key, []).append(row)
        names.setdefault(key, set()).add(row["run"])
    clashes = {k: sorted(v) for k, v in names.items() if len(v) > 1}
    if clashes:
        detail = "; ".join(f"{k}: {v}" for k, v in sorted(clashes.items()))
        raise ValueError(f"more than one run per (phase, condition, seed): {detail}")
    return {k: groups[k] for k in sorted(groups, key=lambda k: tuple(map(natural_key, k)))}


def _bool(value: str, col: str, row: dict) -> bool:
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(
        f"{col}: cannot read {value!r} as a boolean (run {row['run']}, task {row['instance_id']})"
    )


def typed_frame(rows: list[dict]) -> pd.DataFrame:
    """Rows as a DataFrame with boolean and numeric columns converted (strictly)."""
    records = []
    for row in rows:
        rec = {k: row[k] for k in ("phase", "run", "condition", "seed", "instance_id")}
        for col in BOOL_COLS:
            rec[col] = _bool(row[col], col, row)
        for col in NUM_COLS:
            try:
                rec[col] = float(row[col])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{col}: cannot read {row[col]!r} as a number "
                    f"(run {row['run']}, task {row['instance_id']})"
                ) from e
        records.append(rec)
    return pd.DataFrame.from_records(records, columns=["phase", "run", "condition", "seed", "instance_id", *BOOL_COLS, *NUM_COLS])


def _phases(keys) -> list[str]:
    return sorted({k[0] for k in keys}, key=natural_key)


# ---------------------------------------------------------------- statistics


def is_beyond_noise(scope: str, mcnemar_p: float, wins: int, losses: int, discordant) -> bool:
    """Pre-registered rule: pooled McNemar p < 0.05 AND |wins - losses| > noise-floor discordant.

    The rule is defined on pooled seeds, so per-seed rows are never beyond noise, and a
    phase without a noise floor (C1 not run at two seeds) supports no claim.
    """
    if scope != "pooled" or discordant is None:
        return False
    return bool(mcnemar_p < ALPHA and abs(wins - losses) > discordant)


def mde_discordant(discordant: int, alpha: float = ALPHA, max_m: int = 1000) -> int | None:
    """Smallest discordant count that the pre-registered rule could call beyond noise.

    The rule needs exact two-sided McNemar p < ``alpha`` and |wins - losses| > the phase's
    noise-floor ``discordant`` count, so the easiest qualifying pattern is ``m`` discordant
    pairs all in one direction: the smallest ``m`` with ``_exact_two_sided(m, 0) < alpha``
    and ``m > discordant``. ``None`` if no such ``m`` is at most ``max_m``.
    """
    for m in range(max(int(discordant) + 1, 1), max_m + 1):
        if _exact_two_sided(m, 0) < alpha:
            return m
    return None


def pooled_sizes(groups: dict) -> dict[str, int]:
    """Per phase, the control's pooled row count: its rows at both of its seeds.

    That is the pooled n of any condition which covers every control task, which every
    condition of a complete phase does; a condition covering fewer tasks has a smaller
    pooled n, and so a larger true minimum detectable effect than the one reported here.
    Phases where C1 lacks two seeds have no pooled row and are absent.
    """
    out = {}
    for phase in _phases(groups):
        seeds = sorted((k[2] for k in groups if k[0] == phase and k[1] == CONTROL), key=natural_key)
        if len(seeds) == 2:
            out[phase] = sum(len(groups[(phase, CONTROL, s)]) for s in seeds)
    return out


def _stat_row(result: dict) -> dict:
    lo, hi = result["boot_ci"]
    out = {k: result[k] for k in STAT_COLS if k in result}
    out.update(ci_lo=lo, ci_hi=hi)
    return out


def check_design(groups: dict, strict: bool = False) -> dict:
    """Find the gaps in the pre-registered design (two seeds, every condition paired with C1).

    Returns ``{"notes": [...], "phases": {...}, "conditions": {(phase, cond), ...},
    "skip": {(phase, cond, seed), ...}}``. A phase is incomplete when C1 has fewer than two
    seeds (no noise floor, no pooled rows). A condition is incomplete when one of its runs
    has a seed C1 lacks (that run is skipped) or when it does not match both C1 seeds (no
    pooled row). With ``strict`` any gap raises instead. More than two C1 seeds always raises.
    """
    notes, phases, conds, skip = [], set(), set(), set()
    for phase in _phases(groups):
        keys = [k for k in groups if k[0] == phase]
        control = sorted({k[2] for k in keys if k[1] == CONTROL}, key=natural_key)
        if len(control) > 2:
            raise ValueError(
                f"phase {phase}: {CONTROL} has {len(control)} seeds {control}; the pre-registration fixes two"
            )
        if not control:
            phases.add(phase)
            notes.append(f"phase {phase}: no {CONTROL} runs, so no noise floor and no comparisons against {CONTROL}")
            continue
        if len(control) < 2:
            phases.add(phase)
            notes.append(f"phase {phase}: {CONTROL} has only seed {control[0]}, so no noise floor and no pooled rows")
        for cond in sorted({k[1] for k in keys if k[1] != CONTROL}, key=natural_key):
            seeds = sorted({k[2] for k in keys if k[1] == cond}, key=natural_key)
            for seed in seeds:
                if seed not in control:
                    skip.add((phase, cond, seed))
                    conds.add((phase, cond))
                    run = groups[(phase, cond, seed)][0]["run"]
                    notes.append(f"phase {phase}: run {run} ({cond} seed {seed}) has no {CONTROL} run at "
                                 f"seed {seed}; skipped")
            matched = [x for x in seeds if x in control]
            if len(control) == 2 and matched != control:
                conds.add((phase, cond))
                notes.append(f"phase {phase}: {cond} is paired with {CONTROL} only at seed(s) "
                             f"{', '.join(matched) or 'none'}; no pooled row")
    if strict and notes:
        raise ValueError("incomplete design (--strict): " + "; ".join(notes))
    return {"notes": notes, "phases": phases, "conditions": conds, "skip": skip}


def is_incomplete(design: dict, phase: str, cond: str) -> bool:
    return phase in design["phases"] or (phase, cond) in design["conditions"]


def noise_floors(groups: dict, design: dict | None = None) -> dict[str, dict]:
    """Per phase: C1 seed a vs C1 seed b (seeds in ascending order).

    Phases where C1 lacks two seeds get no noise floor (see ``check_design``).
    """
    design = design if design is not None else check_design(groups)
    floors = {}
    for phase in _phases(groups):
        seeds = sorted((k[2] for k in groups if k[0] == phase and k[1] == CONTROL), key=natural_key)
        if len(seeds) != 2:
            continue
        a, b = seeds
        try:
            res = noise_floor(groups[(phase, CONTROL, a)], groups[(phase, CONTROL, b)])
        except ValueError as e:
            raise ValueError(f"phase {phase} noise floor ({CONTROL} {a} vs {b}): {e}") from e
        row = _stat_row(res)
        floors[phase] = {
            "phase": phase, "seed_a": a, "seed_b": b, "n": row["n"],
            "rate_a": row["control_rate"], "rate_b": row["treatment_rate"], "diff": row["diff"],
            "discordant": res["discordant"], "wins": row["wins"], "losses": row["losses"],
            "ties": row["ties"], "mcnemar_p": row["mcnemar_p"],
            "ci_lo": row["ci_lo"], "ci_hi": row["ci_hi"],
        }
    return floors


def comparisons(groups: dict, floors: dict, design: dict | None = None) -> list[dict]:
    """Each non-control condition vs C1 per phase: one row per seed C1 also ran, plus pooled.

    Runs at a seed C1 lacks are skipped; pooled needs both C1 seeds matched and no seed
    skipped (see ``check_design``). Rows of a flagged condition carry ``incomplete`` = True. Pooled matching is on
    (instance_id, seed) with the bootstrap clustered by task. Unmatched tasks raise
    (``graphswap.stats.paired``).
    """
    design = design if design is not None else check_design(groups)
    out = []
    for phase in _phases(groups):
        control = {k[2]: v for k, v in groups.items() if k[0] == phase and k[1] == CONTROL}
        if not control:
            continue
        discordant = floors.get(phase, {}).get("discordant")
        conds = sorted({k[1] for k in groups if k[0] == phase and k[1] != CONTROL}, key=natural_key)
        for cond in conds:
            treat = {k[2]: v for k, v in groups.items()
                     if k[0] == phase and k[1] == cond and k not in design["skip"]}
            matched = sorted(treat, key=natural_key)
            scopes = [(f"seed {s}", [s]) for s in matched]
            if not is_incomplete(design, phase, cond):  # both C1 seeds matched, none skipped
                scopes.append(("pooled", matched))
            for scope, seeds in scopes:
                ctrl_rows = [r for s in seeds for r in control[s]]
                trt_rows = [r for s in seeds for r in treat[s]]
                try:
                    res = paired(ctrl_rows, trt_rows)
                except ValueError as e:
                    raise ValueError(f"phase {phase} {cond} vs {CONTROL} ({scope}): {e}") from e
                row = {"phase": phase, "condition": cond, "scope": scope, **_stat_row(res)}
                row["beyond_noise"] = is_beyond_noise(
                    scope, row["mcnemar_p"], row["wins"], row["losses"], discordant
                )
                row["incomplete"] = is_incomplete(design, phase, cond)
                out.append(row)
    return out


# ---------------------------------------------------------------- descriptive tables


def _aggregate(df: pd.DataFrame) -> dict:
    return {
        "n": len(df),
        "resolved": int(df["resolved"].sum()),
        "resolved_rate": df["resolved"].mean(),
        "patch_submitted_rate": df["patch_submitted"].mean(),
        "mean_tool_calls": df["tool_calls"].mean(),
        "mean_graph_tool_calls": df["graph_tool_calls"].mean(),
        "graph_tool_errors_total": int(df["graph_tool_errors"].sum()),
        "mean_analyzer_calls": df["analyzer_calls"].mean(),
        "gold_file_read_rate": df["gold_file_read"].mean(),
        "gold_file_edited_rate": df["gold_file_edited"].mean(),
        "mean_wall_s": df["wall_s"].mean(),
        "mean_prompt_tokens": df["prompt_tokens"].mean(),
        "mean_completion_tokens": df["completion_tokens"].mean(),
    }


def per_condition(rows: list[dict], design: dict | None = None) -> pd.DataFrame:
    """Secondary metrics per (phase, condition, seed), then per (phase, condition) pooled.

    ``incomplete`` marks every row of a condition or phase that ``check_design`` flagged.
    """
    design = design if design is not None else check_design(group_runs(rows))
    df = typed_frame(rows)
    out = []
    keys = sorted({(p, c) for p, c in zip(df["phase"], df["condition"])}, key=lambda k: tuple(map(natural_key, k)))
    for phase, cond in keys:
        sub = df[(df["phase"] == phase) & (df["condition"] == cond)]
        flag = {"incomplete": is_incomplete(design, phase, cond)}
        for seed in sorted(sub["seed"].unique(), key=natural_key):
            out.append({"phase": phase, "condition": cond, "seed": seed,
                        **_aggregate(sub[sub["seed"] == seed]), **flag})
        out.append({"phase": phase, "condition": cond, "seed": "pooled", **_aggregate(sub), **flag})
    return pd.DataFrame(out, columns=PER_CONDITION_COLS)


def run_meta(rows: list[dict]) -> dict[str, tuple[str, str, str]]:
    """Run name -> (phase, condition, seed), ordered by phase, condition, seed."""
    keys = {(r["phase"], r["condition"], r["seed"], r["run"]) for r in rows}
    return {k[3]: k[:3] for k in sorted(keys, key=lambda k: tuple(map(natural_key, k)))}


def solve_matrix(rows: list[dict]) -> pd.DataFrame:
    """Task x run 0/1 grid; one block of rows per phase (A first), blank where a run did not cover a task."""
    runs = list(run_meta(rows))
    cells: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        task = cells.setdefault((row["phase"], row["instance_id"]), {})
        if row["run"] in task:
            raise ValueError(f"run {row['run']} has task {row['instance_id']} twice")
        task[row["run"]] = int(_bool(row["resolved"], "resolved", row))
    keys = sorted(cells, key=lambda k: tuple(map(natural_key, k)))
    df = pd.DataFrame(
        [{"phase": p, "instance_id": t, **cells[(p, t)]} for p, t in keys],
        columns=["phase", "instance_id", *runs],
    )
    for run in runs:
        df[run] = df[run].astype("Int64")
    return df


def read_summary_csv(path: Path) -> pd.DataFrame:
    """Read a results CSV, skipping a leading ``#`` header comment if the file carries one.

    Defensive: these files carry no comment line today, but ``graphswap.cli._write_csv`` can
    prefix one, and without ``comment="#"`` pandas would read it as the header. No column of
    these files ever contains a ``#``.
    """
    return pd.read_csv(path, comment="#")


def _describe(group: str, quantity: str, values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="raise").dropna()
    return {
        "group": group, "quantity": quantity, "n": int(len(v)),
        "mean": v.mean() if len(v) else math.nan,
        "min": v.min() if len(v) else math.nan,
        "max": v.max() if len(v) else math.nan,
    }


def geometry(results: Path, rows: list[dict] | None = None) -> pd.DataFrame:
    """Embedding geometry over commits per vector set, node-id parity, and gold coverage.

    Gold coverage is reported over every audited task and, when ``rows`` is given, over
    the tasks of each phase. Tasks without gold functions (blank fraction) are skipped.
    """
    results = Path(results)
    frames = {name: read_summary_csv(results / name) for name in {f for f, _ in VECTOR_SETS.values()}}
    out = []
    for vset, (name, prefix) in VECTOR_SETS.items():
        for q in GEOMETRY_QUANTITIES:
            out.append(_describe(vset, q, frames[name][prefix + q]))
    audit = frames["audit.csv"]
    out.append(_describe("node_ids", "id_parity", audit["id_parity"]))
    out.append(_describe("node_ids", "released_only", audit["released_only"]))
    tasks = read_summary_csv(results / "audit_tasks.csv")
    groups = [("gold_coverage", tasks)]
    if rows:
        for phase in _phases((r["phase"],) for r in rows):
            ids = {r["instance_id"] for r in rows if r["phase"] == phase}
            groups.append((f"gold_coverage_phase_{phase}", tasks[tasks["instance_id"].isin(ids)]))
    for group, sub in groups:
        for q in ("released_fraction", "rebuilt_fraction"):
            out.append(_describe(group, q, sub[q]))
    return pd.DataFrame(out, columns=GEOMETRY_COLS)


# ---------------------------------------------------------------- markdown


def _num(value: float, signed: bool = False) -> str:
    """3 decimals, ties rounded half-up (0.0625 -> 0.063, not the float default 0.062)."""
    text = str(Decimal(repr(float(value))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    if text == "-0.000":
        text = "0.000"
    return ("+" + text) if signed and not text.startswith("-") else text


def _fmt(value, col: str = "") -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer() and col in ("n", "wins", "losses", "ties", "discordant", "resolved"):
            return str(int(value))
        return _num(value, signed=(col == "diff"))
    if hasattr(value, "item"):  # numpy scalar
        return _fmt(value.item(), col)
    return str(value)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for rec in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, c) for v, c in zip(rec, cols)) + " |")
    return "\n".join(lines) + "\n"


def _ci(lo, hi) -> str:
    return f"[{_num(lo)}, {_num(hi)}]"


def headline(comp: pd.DataFrame, floors: pd.DataFrame, pooled_n: dict[str, int] | None = None) -> list[str]:
    pooled_n = pooled_n or {}
    lines = []
    for f in floors.itertuples(index=False):
        lines.append(
            f"- Phase {f.phase} noise floor ({CONTROL} seed {f.seed_a} vs {f.seed_b}): n {f.n}, "
            f"resolved {_num(f.rate_a)} vs {_num(f.rate_b)}, **discordant {f.discordant}** "
            f"(wins {f.wins}, losses {f.losses}), McNemar p {_num(f.mcnemar_p)}, "
            f"95% CI {_ci(f.ci_lo, f.ci_hi)}."
        )
        m, n = mde_discordant(f.discordant), pooled_n.get(f.phase)
        if m is not None and n:
            lines.append(
                f"- Phase {f.phase} minimum detectable effect: {m} discordant pairs in one "
                f"direction, pooled |diff| >= {_num(m / n)} (n {n} control rows, noise floor "
                f"{f.discordant})."
            )
    for c in comp[comp["scope"] == "pooled"].itertuples(index=False):
        lines.append(
            f"- Phase {c.phase} {c.condition} vs {CONTROL}, pooled seeds: n {c.n} "
            f"({c.n_clusters} tasks), resolved {_num(c.treatment_rate)} vs {_num(c.control_rate)} "
            f"({CONTROL}), diff {_num(c.diff, signed=True)}, wins {c.wins}, losses {c.losses}, "
            f"McNemar p {_num(c.mcnemar_p)}, "
            f"95% CI {_ci(c.ci_lo, c.ci_hi)}; beyond noise: {'yes' if c.beyond_noise else 'no'}."
        )
    return lines


def solve_counts(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per phase: tasks, runs, tasks solved by no run, tasks solved by every run that covered them."""
    out = []
    for phase in sorted(matrix["phase"].unique(), key=natural_key):
        block = matrix[matrix["phase"] == phase].drop(columns=["phase", "instance_id"])
        block = block.loc[:, block.notna().any()]
        solved = block.sum(axis=1)
        covered = block.notna().sum(axis=1)
        out.append({
            "phase": phase, "tasks": len(block), "runs": block.shape[1],
            "never_solved": int((solved == 0).sum()),
            "always_solved": int(((solved == covered) & (covered > 0)).sum()),
        })
    return pd.DataFrame(out, columns=["phase", "tasks", "runs", "never_solved", "always_solved"])


def write_tables(path: Path, comp, floors, per_cond, matrix, geo, notes=(), pooled_n=None) -> None:
    comp_md = comp.copy()
    if not comp_md.empty:
        comp_md.insert(comp_md.columns.get_loc("ci_lo"), "95% CI", [_ci(a, b) for a, b in zip(comp_md["ci_lo"], comp_md["ci_hi"])])
        comp_md = comp_md.drop(columns=["ci_lo", "ci_hi"])
    floors_md = floors.copy()
    if not floors_md.empty:
        floors_md["95% CI"] = [_ci(a, b) for a, b in zip(floors_md["ci_lo"], floors_md["ci_hi"])]
        floors_md = floors_md.drop(columns=["ci_lo", "ci_hi"])
    parts = [
        "# Summary tables",
        "",
        "Generated by `notebooks/analysis.py` from `results/runs/*.csv` and the audit CSVs; "
        "do not edit by hand. Rates are fractions of task-seed rows; 3 decimals.",
        "",
        "## Headline",
        "",
        *[f"INCOMPLETE: {n}" for n in notes],
        *([""] if notes else []),
        *(headline(comp, floors, pooled_n) or ["_(no noise floor or pooled comparison yet)_"]),
        "",
        f"## Noise floor ({CONTROL} seed a vs seed b, per phase)",
        "",
        md_table(floors_md),
        f"## Comparisons against {CONTROL}",
        "",
        "Paired per task at matched seed (`graphswap.stats.paired`); pooled rows match on "
        "(task, seed) with the 95% bootstrap CI clustered by task. `beyond_noise` is the "
        "pre-registered rule: pooled McNemar exact p < 0.05 and |wins - losses| > the phase's "
        "noise-floor discordant count; per-seed rows are never marked. `incomplete` marks rows "
        "of a phase or condition with a missing or skipped seed (see INCOMPLETE above).",
        "",
        md_table(comp_md),
        "## Per condition",
        "",
        "Means are per task-seed row; `graph_tool_errors_total` is a sum.",
        "",
        md_table(per_cond),
        "## Solve matrix",
        "",
        "1 = resolved, blank = task not in that run. `always_solved` counts tasks resolved by "
        "every run of the phase.",
        "",
        md_table(solve_counts(matrix)),
        md_table(matrix),
        "## Embedding geometry and gold coverage",
        "",
        "Over commits (vector sets, node ids) or tasks (gold coverage). released = released "
        "graph and embeddings; qwen3 = Qwen3-Embedding-0.6B and gemma = EmbeddingGemma-300m, "
        "both on the rebuilt graphs. Gold coverage skips tasks with no gold function.",
        "",
        md_table(geo),
    ]
    path.write_text("\n".join(parts))


# ---------------------------------------------------------------- figures
# Colours follow the dataviz reference palette (light theme), validated with its
# validate_palette.js: slot-1 blue for single-series marks; the first three categorical
# slots (all-pairs safe) for the three vector sets. Text always wears ink tokens.

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
ACCENT = "#2a78d6"
UNRESOLVED = "#e9e8e3"
SET_COLORS = {"released": "#2a78d6", "qwen3": "#eb6834", "gemma": "#1baf7a"}
SET_LABELS = {
    "released": "Released",
    "qwen3": "Qwen3-Emb.\n0.6B",
    "gemma": "Embedding-\nGemma 300m",
}
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],  # bundled with matplotlib: identical output on any machine
    "font.size": 8.5,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "axes.titlesize": 9,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "legend.frameon": False,
    "svg.hashsalt": "graphswap-analysis",  # stable SVG ids across regenerations
}


def _style():
    import matplotlib

    return matplotlib.rc_context(STYLE)


def _titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.995, title, ha="left", va="top", fontsize=11, weight="bold", color=INK)
    fig.text(0.012, 0.995 - 0.30 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=8, color=INK2)


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    fig.savefig(out_dir / f"{name}.svg", metadata={"Date": None})


def _cond_label(cond: str) -> str:
    return f"{cond} (control)" if cond == CONTROL else cond


def solve_legend_handles(any_blank: bool) -> list:
    """Legend patches for the solve matrix: filled, pale, and (when needed) an empty cell.

    A task a run never saw is drawn as no rectangle at all, so its swatch is an outline on
    the figure surface: it needs a dashed edge in a colour that reads against the surface,
    or the entry looks blank next to the two filled ones.
    """
    from matplotlib.patches import Patch

    handles = [Patch(color=ACCENT, label="resolved"), Patch(color=UNRESOLVED, label="not resolved")]
    if any_blank:
        handles.append(Patch(facecolor=SURFACE, edgecolor=MUTED, linewidth=1.0,
                             linestyle=(0, (2, 1.4)), label="task not in run"))
    return handles


def solve_blocks(matrix: pd.DataFrame, meta: dict[str, tuple[str, str, str]]) -> list[dict]:
    """Per phase (A first): that phase's runs and tasks, tasks sorted by solve fraction, then id.

    Each block is sorted on its own, so Phase B's tasks that also ran in Phase A are
    ranked by their Phase B runs only.
    """
    blocks = []
    for phase in sorted(matrix["phase"].unique(), key=natural_key):
        runs = [r for r in matrix.columns if r in meta and meta[r][0] == phase]
        data = matrix[matrix["phase"] == phase].set_index("instance_id")[runs]
        solved = data.sum(axis=1).astype(int)
        covered = data.notna().sum(axis=1).astype(int)
        frac = {t: (solved[t] / covered[t]) if covered[t] else 0.0 for t in data.index}
        order = sorted(data.index, key=lambda t: (-frac[t], natural_key(t)))
        blocks.append({"phase": phase, "runs": runs, "data": data.loc[order], "order": order,
                       "solved": solved, "covered": covered})
    return blocks


# solve-matrix layout, inches unless noted
SM_COL, SM_LEFT, SM_RIGHT = 0.42, 1.25, 0.95
SM_TITLE, SM_LEGEND, SM_HEADER = 0.8, 0.35, 0.7
SM_LEADING = 1.2  # row pitch / label font size when rows must shrink
SM_PITCH, SM_FONT, SM_MIN_FONT, SM_MAX_HEIGHT = 0.19, 7.5, 6.0, 11.0
SM_MIN_WIDTH = 5.2  # the legend row and the subtitle must fit however few runs a block has


def solve_layout(n_rows: int) -> tuple[float, float, float]:
    """(row pitch in, label font pt, figure height in) for one phase block of ``n_rows`` tasks.

    Rows keep the default pitch until the figure would pass SM_MAX_HEIGHT; then rows and
    font shrink together down to SM_MIN_FONT, and only after that does the figure grow.
    """
    fixed = SM_TITLE + SM_LEGEND + SM_HEADER
    pitch, font = SM_PITCH, SM_FONT
    if fixed + n_rows * pitch > SM_MAX_HEIGHT:
        pitch = (SM_MAX_HEIGHT - fixed) / n_rows
        font = min(SM_FONT, pitch * 72 / SM_LEADING)
        if font < SM_MIN_FONT:
            font, pitch = SM_MIN_FONT, SM_MIN_FONT * SM_LEADING / 72
    return pitch, font, fixed + n_rows * pitch


def plot_solve_matrix(matrix: pd.DataFrame, meta: dict[str, tuple[str, str, str]], out_dir: Path,
                      phase: str) -> dict:
    """Task x run grid for one phase: filled = resolved, pale = not resolved, blank = task not
    in that run. ``meta`` maps run name -> (phase, condition, seed), as from ``run_meta``.
    Saved as ``fig_solve_matrix_<phase>``.
    Returns the layout (figure height, row pitch, label font) for checks.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory, offset_copy

    found = [b for b in solve_blocks(matrix, meta) if b["phase"] == phase]
    if not found:
        raise ValueError(f"no runs for phase {phase!r}")
    (b,) = found
    xs, x, prev = {}, 0.0, None  # x positions: one unit per run, a gap between conditions
    for run in b["runs"]:
        if prev is not None and meta[run][1] != prev:
            x += 0.45
        xs[run], x, prev = x, x + 1.0, meta[run][1]
    b["xs"], b["width"] = xs, x - 1.0
    pitch, font, height = solve_layout(len(b["order"]))
    width = max(SM_LEFT + (b["width"] + 1.2) * SM_COL + SM_RIGHT, SM_MIN_WIDTH)
    any_blank = bool(b["data"].isna().any().any())

    with _style():
        fig = Figure(figsize=(width, height))
        n = len(b["order"])
        grid_top = SM_TITLE + SM_HEADER
        ax = fig.add_axes([SM_LEFT / width, 1 - (grid_top + n * pitch) / height,
                           (b["width"] + 1.2) * SM_COL / width, n * pitch / height])
        data = b["data"]
        for i, task in enumerate(b["order"]):
            for run in b["runs"]:
                v = data.at[task, run]
                if pd.isna(v):
                    continue
                ax.add_patch(Rectangle((b["xs"][run] - 0.42, i - 0.38), 0.84, 0.76, lw=0,
                                       color=ACCENT if v == 1 else UNRESOLVED))
            k, m = int(b["solved"][task]), int(b["covered"][task])
            note = "  never" if k == 0 else ("  always" if k == m else "")
            ax.text(b["width"] + 0.75, i, f"{k}/{m}{note}", va="center", ha="left", fontsize=font,
                    color=INK2 if note else MUTED)
        ax.set_xlim(-0.6, b["width"] + 0.6)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks(range(n), b["order"], fontsize=font)
        ax.set_xticks([b["xs"][r] for r in b["runs"]], [meta[r][2] for r in b["runs"]], fontsize=7.5)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=4)
        for side in ("left", "bottom", "top", "right"):
            ax.spines[side].set_visible(False)
        above = blended_transform_factory(ax.transData, ax.transAxes)
        groups: dict[str, list[float]] = {}
        for run in b["runs"]:
            groups.setdefault(meta[run][1], []).append(b["xs"][run])
        for cond, gx in groups.items():
            ax.text(sum(gx) / len(gx), 1, _cond_label(cond), ha="center", va="bottom", fontsize=8,
                    color=INK, transform=offset_copy(above, fig=fig, y=21, units="points"))
            ax.plot([min(gx) - 0.4, max(gx) + 0.4], [1, 1], color=AXIS, lw=0.8, clip_on=False,
                    transform=offset_copy(above, fig=fig, y=18, units="points"))
        ax.text(-0.6, 1, "seed", ha="right", va="bottom", fontsize=7.5, color=MUTED,
                transform=offset_copy(above, fig=fig, y=4, units="points"))
        ax.text(b["width"] + 0.75, 1, "resolved\nby", ha="left", va="bottom", fontsize=7.5,
                color=MUTED, transform=offset_copy(above, fig=fig, y=4, units="points"))
        never = int((b["solved"] == 0).sum())
        always = int(((b["solved"] == b["covered"]) & (b["covered"] > 0)).sum())
        fig.text(0.012, 1 - SM_TITLE / height,
                 f"Phase {b['phase']}: {n} tasks, {len(b['runs'])} runs "
                 f"({always} always resolved, {never} never)",
                 ha="left", va="top", fontsize=9, weight="bold", color=INK)
        handles = solve_legend_handles(any_blank)
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(SM_LEFT / width, 0.0),
                   ncol=len(handles), fontsize=7.5, handlelength=1.0, handleheight=1.0,
                   columnspacing=1.4, labelcolor=INK2)
        _titles(fig, f"Tasks resolved by each run, Phase {phase}",
                "Rows: tasks, most-resolved first.\nColumns: runs by condition and seed.")
        _save(fig, out_dir, f"fig_solve_matrix_{phase}")
    return {"height_in": height, "pitch_in": pitch, "font_pt": font}


TOOL_METRICS = [
    ("mean_graph_tool_calls", "Graph tool calls", "calls per task run", "{:.1f}"),
    ("mean_tool_calls", "All tool calls", "calls per task run", "{:.1f}"),
    ("gold_file_read_rate", "Gold file read", "share of task runs", "{:.2f}"),
    ("gold_file_edited_rate", "Gold file edited", "share of task runs", "{:.2f}"),
    ("patch_submitted_rate", "Patch submitted", "share of task runs", "{:.2f}"),
    ("resolved_rate", "Resolved", "share of task runs", "{:.2f}"),
]
PAGE_WIDTH = 7.5  # inches; the tool-profile figure keeps this width however many bars it has


def plot_tool_profile(per_cond: pd.DataFrame, out_dir: Path) -> None:
    """Small multiples, one panel per metric: a bar per condition (seeds pooled), a dot per seed.

    Phases sit side by side in each panel, split by a divider and labelled below the axis.
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    pooled = per_cond[per_cond["seed"] == "pooled"].reset_index(drop=True)
    seeds = per_cond[per_cond["seed"] != "pooled"]
    cats = list(zip(pooled["phase"], pooled["condition"]))
    phases = sorted(pooled["phase"].unique(), key=natural_key)
    multi_phase = len(phases) > 1
    xpos, x, prev = [], 0.0, None
    for phase, _ in cats:
        if prev is not None and phase != prev:
            x += 0.8
        xpos.append(x)
        x, prev = x + 1.0, phase
    span = (xpos[-1] + 0.75) - (-0.75)
    seed_names = sorted(seeds["seed"].unique(), key=natural_key)
    offsets = {s: (i - (len(seed_names) - 1) / 2) * 0.12 for i, s in enumerate(seed_names)}

    ncols = 3 if len(cats) <= 4 else 2
    nrows = len(TOOL_METRICS) // ncols
    left, right, wspace = 0.75 / PAGE_WIDTH, 0.99, 0.45
    panel_in = PAGE_WIDTH * (right - left) / (ncols + wspace * (ncols - 1))
    label_font = 7.5 if panel_in / span >= 0.4 else 6.5  # value labels must fit their slot
    row_in = 2.35 + (0.2 if multi_phase else 0.0)
    height = row_in * nrows + 0.5
    with _style():
        fig = Figure(figsize=(PAGE_WIDTH, height))
        axes = fig.subplots(nrows, ncols, gridspec_kw={
            "hspace": 0.62 + (0.12 if multi_phase else 0.0), "wspace": wspace,
            "top": 1 - 0.95 / height, "bottom": (0.65 + (0.2 if multi_phase else 0.0)) / height,
            "left": left, "right": right})
        for ax, (col, title, unit, fmt) in zip(axes.flat, TOOL_METRICS):
            vals = pooled[col].to_numpy(dtype=float)
            ax.bar(xpos, vals, width=0.4, color=ACCENT, zorder=2)
            tops = vals.copy()
            for j, (phase, cond) in enumerate(cats):
                sub = seeds[(seeds["phase"] == phase) & (seeds["condition"] == cond)]
                for s, v in zip(sub["seed"], sub[col].astype(float)):
                    ax.scatter(xpos[j] + offsets[s], v, s=22, color=INK, edgecolors=SURFACE,
                               linewidths=1.0, zorder=3)
                    tops[j] = max(tops[j], v)
            is_rate = col.endswith("_rate")
            ymax = 1.0 if is_rate else max(float(tops.max()), 1.0)
            for j, v in enumerate(vals):
                ax.text(xpos[j], tops[j] + ymax * 0.04, fmt.format(v), ha="center", va="bottom",
                        fontsize=label_font, color=INK)
            ax.set_ylim(0, ymax * 1.22)
            if is_rate:
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xlim(-0.75, xpos[-1] + 0.75)
            ax.set_xticks(xpos, [c for _, c in cats], fontsize=7.5)
            ax.tick_params(axis="x", length=0)
            ax.tick_params(axis="y", length=3, labelsize=7.5)
            ax.yaxis.grid(True, zorder=0)
            ax.set_axisbelow(True)
            ax.set_title(title, pad=6)
            ax.set_ylabel(unit, fontsize=7.5)
            if multi_phase:
                for phase in phases:
                    px = [xp for xp, (p, _) in zip(xpos, cats) if p == phase]
                    ax.annotate(f"Phase {phase}", xy=(sum(px) / len(px), 0), xycoords=("data", "axes fraction"),
                                xytext=(0, -20), textcoords="offset points", ha="center", va="top",
                                fontsize=7.5, color=MUTED)
                for j in range(1, len(cats)):
                    if cats[j][0] != cats[j - 1][0]:
                        ax.axvline((xpos[j] + xpos[j - 1]) / 2, color=AXIS, lw=0.8, zorder=1)
        handles = [Patch(color=ACCENT, label="both seeds pooled"),
                   Line2D([], [], ls="", marker="o", markersize=6.5, color=INK, markeredgecolor=SURFACE,
                          label="one seed")]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(left - 0.01, 0.0), ncol=2,
                   fontsize=7.5, labelcolor=INK2, handlelength=1.2)
        _titles(fig, "Tool use and outcomes by condition",
                f"Phase {', '.join(phases)}; {CONTROL} is the control. Bars: mean over task runs with "
                f"seeds pooled; dots: each seed ({', '.join(seed_names)}, left to right).")
        _save(fig, out_dir, "fig_tool_profile")


GEOMETRY_PANELS = [
    ("mean_cos", "Mean random-pair cosine", "cosine similarity", "{:.3f}"),
    ("eff_dim", "Effective dimension", "dimensions (participation ratio)", "{:.1f}"),
    ("dup_rate", "Duplicate vectors", "share of nodes with an identical twin", "{:.3f}"),
]


def plot_geometry(results: Path, out_dir: Path) -> None:
    """Distribution over commits of three geometry statistics, one box per vector set."""
    import numpy as np
    from matplotlib.figure import Figure

    results = Path(results)
    frames = {name: read_summary_csv(results / name) for name in {f for f, _ in VECTOR_SETS.values()}}
    sets = list(VECTOR_SETS)
    rng = np.random.default_rng(0)
    with _style():
        fig = Figure(figsize=(9.6, 3.9))
        axes = fig.subplots(1, 3, gridspec_kw={"wspace": 0.38, "top": 0.76, "bottom": 0.13,
                                               "left": 0.07, "right": 0.99})
        counts = set()
        for ax, (q, title, unit, fmt) in zip(axes, GEOMETRY_PANELS):
            data = []
            for vset in sets:
                name, prefix = VECTOR_SETS[vset]
                data.append(pd.to_numeric(frames[name][prefix + q], errors="raise").dropna().to_numpy())
            counts.update(len(d) for d in data)
            for i, (vset, d) in enumerate(zip(sets, data)):
                jitter = rng.uniform(-0.17, 0.17, len(d))
                ax.scatter(i + jitter, d, s=7, color=SET_COLORS[vset], alpha=0.55, linewidths=0, zorder=2)
            bp = ax.boxplot(data, positions=range(len(sets)), widths=0.5, showfliers=False,
                            patch_artist=True, zorder=3)
            for box, vset in zip(bp["boxes"], sets):
                box.set(facecolor=(*_rgb(SET_COLORS[vset]), 0.12), edgecolor=INK2, linewidth=0.9)
            for part in ("whiskers", "caps"):
                for line in bp[part]:
                    line.set(color=INK2, linewidth=0.9)
            for line in bp["medians"]:
                line.set(color=INK, linewidth=1.6)
            hi = max(float(d.max()) for d in data if len(d))
            for i, d in enumerate(data):
                if len(d):
                    ax.text(i, float(d.max()) + hi * 0.04, "mean " + fmt.format(d.mean()),
                            ha="center", va="bottom", fontsize=7.5, color=INK)
            ax.set_ylim(0, hi * 1.18)
            ax.set_xticks(range(len(sets)), [SET_LABELS.get(s, s) for s in sets], fontsize=7.5)
            ax.tick_params(axis="x", length=0)
            ax.tick_params(axis="y", length=3, labelsize=7.5)
            ax.yaxis.grid(True, zorder=0)
            ax.set_axisbelow(True)
            ax.set_title(title, pad=6)
            ax.set_ylabel(unit, fontsize=7.5)
        n_text = f"n = {counts.pop()}" if len(counts) == 1 else "n varies by set"
        _titles(fig, "Embedding geometry across commits",
                f"Each dot is one repository commit ({n_text} per vector set). Released vectors are on "
                "the released graphs, Qwen3 and EmbeddingGemma on the rebuilt graphs.\n"
                "Boxes: median and quartiles; whiskers: 1.5 IQR.")
        _save(fig, out_dir, "fig_geometry")


def _rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


# ---------------------------------------------------------------- compute


def compute(results: Path, strict: bool = False) -> dict:
    """Every table the writeup quotes, from the per-task CSVs alone. Writes nothing.

    Returns the five tables plus the rows they came from, the pooled row count per phase and
    the design notes (``incomplete``). ``run`` adds the summary files and the figures; the
    public notebook in ``kaggle/`` calls this and displays the tables, so both show the same
    numbers from the same code. Gaps in the design raise when ``strict`` is set.
    """
    results = Path(results)
    rows = load_runs(results)
    groups = group_runs(rows)
    design = check_design(groups, strict=strict)
    floors_by_phase = noise_floors(groups, design)
    floors = pd.DataFrame(list(floors_by_phase.values()), columns=NOISE_COLS)
    comp = pd.DataFrame(comparisons(groups, floors_by_phase, design), columns=COMPARISON_COLS)
    return {
        "comparisons": comp,
        "noise_floor": floors,
        "per_condition": per_condition(rows, design),
        "solve_matrix": solve_matrix(rows),
        "geometry": geometry(results, rows),
        "rows": rows,
        "pooled_n": pooled_sizes(groups),
        "incomplete": design["notes"],
    }


# ---------------------------------------------------------------- driver


def run(results: Path = ROOT / "results", figures: Path = ROOT / "paper" / "figures",
        strict: bool = False) -> dict:
    """Compute every table, write the summary CSVs, tables.md and figures; return the tables.

    Gaps in the design are printed as INCOMPLETE notes and marked in the outputs, or raise
    when ``strict`` is set.
    """
    results, figures = Path(results), Path(figures)
    tables = compute(results, strict=strict)
    for note in tables["incomplete"]:
        print(f"INCOMPLETE: {note}")
    comp, floors = tables["comparisons"], tables["noise_floor"]
    per_cond, matrix, geo = tables["per_condition"], tables["solve_matrix"], tables["geometry"]

    summary = results / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    comp.to_csv(summary / "comparisons.csv", index=False)
    floors.to_csv(summary / "noise_floor.csv", index=False)
    per_cond.to_csv(summary / "per_condition.csv", index=False)
    matrix.to_csv(summary / "solve_matrix.csv", index=False)
    geo.to_csv(summary / "geometry.csv", index=False)
    write_tables(summary / "tables.md", comp, floors, per_cond, matrix, geo,
                 tables["incomplete"], tables["pooled_n"])

    figures.mkdir(parents=True, exist_ok=True)
    meta = run_meta(tables["rows"])
    for phase in sorted({m[0] for m in meta.values()}, key=natural_key):
        plot_solve_matrix(matrix, meta, figures, phase=phase)
    plot_tool_profile(per_cond, figures)
    plot_geometry(results, figures)
    return {
        "comparisons": comp, "noise_floor": floors, "per_condition": per_cond,
        "solve_matrix": matrix, "geometry": geo, "summary_dir": summary, "figures_dir": figures,
        "incomplete": tables["incomplete"],
    }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=ROOT / "results", help="results directory (default: repo results/)")
    p.add_argument("--figures", type=Path, default=ROOT / "paper" / "figures", help="figure output directory")
    p.add_argument("--strict", action="store_true",
                   help="raise on any missing or unmatched seed instead of noting and skipping it "
                        "(use for the final paper regeneration)")
    args = p.parse_args(argv)
    out = run(args.results, args.figures, strict=args.strict)
    print(f"wrote {out['summary_dir']}/{{comparisons,noise_floor,per_condition,solve_matrix,geometry}}.csv, tables.md")
    print(f"wrote {out['figures_dir']}/{{{','.join(FIGURES)}}}.{{png,svg}} and fig_solve_matrix_<phase>.{{png,svg}}")
    return out


if __name__ == "__main__":
    main()
