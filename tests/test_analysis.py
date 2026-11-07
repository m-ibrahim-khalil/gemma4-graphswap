"""Tests for notebooks/analysis.py on a small synthetic results/ tree."""

import csv
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("analysis", ROOT / "notebooks" / "analysis.py")
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)

RUN_COLS = [
    "instance_id", "run", "condition", "seed", "resolved", "patch_submitted",
    "tool_calls", "graph_tool_calls", "graph_tool_errors", "analyzer_calls",
    "gold_file_read", "gold_file_edited", "wall_s", "prompt_tokens", "completion_tokens",
]
TASKS = ["t1", "t2", "t3", "t4"]


def _write(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_run(results, run, resolved, tasks=TASKS, graph_calls=3):
    phase, cond, seed = run.split("_")[:3]
    rows = []
    for i, (tid, r) in enumerate(zip(tasks, resolved)):
        rows.append({
            "instance_id": tid, "run": run, "condition": cond, "seed": seed,
            "resolved": str(bool(r)), "patch_submitted": "True",
            "tool_calls": 10 + i, "graph_tool_calls": graph_calls, "graph_tool_errors": 1,
            "analyzer_calls": 2, "gold_file_read": str(i % 2 == 0),
            "gold_file_edited": "False", "wall_s": 100.0 + i,
            "prompt_tokens": 1000, "completion_tokens": 50,
        })
    _write(results / "runs" / f"{run}.csv", RUN_COLS, rows)


def write_audits(results):
    _write(results / "audit.csv",
           ["repo_short", "commit", "released_nodes", "rebuilt_nodes", "id_parity", "released_only",
            "rel_mean_cos", "rel_p05", "rel_p95", "rel_dup_rate", "rel_eff_dim",
            "ours_mean_cos", "ours_p05", "ours_p95", "ours_dup_rate", "ours_eff_dim"],
           [
               {"repo_short": "r", "commit": "c1", "released_nodes": 10, "rebuilt_nodes": 12,
                "id_parity": 1.0, "released_only": 0, "rel_mean_cos": 0.8, "rel_p05": 0.6,
                "rel_p95": 0.9, "rel_dup_rate": 0.1, "rel_eff_dim": 3.0, "ours_mean_cos": 0.4,
                "ours_p05": 0.2, "ours_p95": 0.6, "ours_dup_rate": 0.05, "ours_eff_dim": 50.0},
               {"repo_short": "r", "commit": "c2", "released_nodes": 10, "rebuilt_nodes": 12,
                "id_parity": 0.5, "released_only": 2, "rel_mean_cos": 0.9, "rel_p05": 0.7,
                "rel_p95": 0.95, "rel_dup_rate": 0.2, "rel_eff_dim": 2.0, "ours_mean_cos": 0.3,
                "ours_p05": 0.1, "ours_p95": 0.5, "ours_dup_rate": 0.15, "ours_eff_dim": 60.0},
           ])
    _write(results / "audit_gemma.csv",
           ["repo_short", "commit", "gemma_mean_cos", "gemma_p05", "gemma_p95",
            "gemma_dup_rate", "gemma_eff_dim"],
           [
               {"repo_short": "r", "commit": "c1", "gemma_mean_cos": 0.35, "gemma_p05": 0.2,
                "gemma_p95": 0.5, "gemma_dup_rate": 0.05, "gemma_eff_dim": 40.0},
               {"repo_short": "r", "commit": "c2", "gemma_mean_cos": 0.33, "gemma_p05": 0.2,
                "gemma_p95": 0.5, "gemma_dup_rate": 0.15, "gemma_eff_dim": 44.0},
           ])
    _write(results / "audit_tasks.csv",
           ["instance_id", "repo_short", "commit", "n_gold", "released_covered",
            "released_fraction", "rebuilt_covered", "rebuilt_fraction"],
           [
               {"instance_id": "t1", "repo_short": "r", "commit": "c1", "n_gold": 2,
                "released_covered": 1, "released_fraction": 0.5, "rebuilt_covered": 2,
                "rebuilt_fraction": 1.0},
               {"instance_id": "t2", "repo_short": "r", "commit": "c1", "n_gold": 1,
                "released_covered": 1, "released_fraction": 1.0, "rebuilt_covered": 1,
                "rebuilt_fraction": 1.0},
               # no gold functions: fraction undefined, must be skipped, not read as 0
               {"instance_id": "t3", "repo_short": "r", "commit": "c2", "n_gold": 0,
                "released_covered": 0, "released_fraction": "", "rebuilt_covered": 0,
                "rebuilt_fraction": ""},
               {"instance_id": "x9", "repo_short": "r", "commit": "c2", "n_gold": 1,
                "released_covered": 0, "released_fraction": 0.0, "rebuilt_covered": 1,
                "rebuilt_fraction": 1.0},
           ])


def make_results(res):
    # control C1 is identical across seeds: noise floor discordant 0
    write_run(res, "A_C1_1234", [1, 0, 0, 0])
    write_run(res, "A_C1_4321", [1, 0, 0, 0])
    # treatment C2 wins one task at seed 1234 only: a 1-0 pooled win
    write_run(res, "A_C2_1234", [1, 1, 0, 0])
    write_run(res, "A_C2_4321", [1, 0, 0, 0])
    write_audits(res)
    return res


@pytest.fixture
def results(tmp_path):
    return make_results(tmp_path / "results")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    """One full run of the script on the synthetic tree, shared by the read-only output tests."""
    base = tmp_path_factory.mktemp("analysis")
    results = make_results(base / "results")
    figures = base / "figures"
    analysis.main(["--results", str(results), "--figures", str(figures)])
    return results / "summary", figures


def test_comparisons_have_both_seeds_and_pooled_for_treatment(outputs):
    summary, _ = outputs
    rows = read_csv(summary / "comparisons.csv")
    assert list(rows[0]) == [
        "phase", "condition", "scope", "n", "n_clusters", "control_rate", "treatment_rate",
        "diff", "wins", "losses", "ties", "mcnemar_p", "ci_lo", "ci_hi", "beyond_noise", "incomplete",
    ]
    c2 = [r for r in rows if r["condition"] == "C2"]
    assert [r["scope"] for r in c2] == ["seed 1234", "seed 4321", "pooled"]
    assert all(r["phase"] == "A" for r in c2)
    assert not [r for r in rows if r["condition"] == "C1"]  # control is never a treatment
    pooled = c2[-1]
    assert (pooled["n"], pooled["n_clusters"]) == ("8", "4")
    assert (pooled["wins"], pooled["losses"], pooled["ties"]) == ("1", "0", "7")
    assert float(pooled["diff"]) == pytest.approx(1 / 8)
    assert all(r["incomplete"] == "False" for r in rows)


def test_one_zero_win_is_not_beyond_noise(outputs):
    summary, _ = outputs
    rows = read_csv(summary / "comparisons.csv")
    assert all(r["beyond_noise"] == "False" for r in rows)


def test_noise_floor_one_row_per_phase(outputs):
    summary, _ = outputs
    rows = read_csv(summary / "noise_floor.csv")
    assert len(rows) == 1
    assert rows[0]["phase"] == "A"
    assert (rows[0]["seed_a"], rows[0]["seed_b"]) == ("1234", "4321")
    assert rows[0]["discordant"] == "0"
    for col in ("wins", "losses", "mcnemar_p", "ci_lo", "ci_hi"):
        assert col in rows[0]


def test_beyond_noise_rule():
    rule = analysis.is_beyond_noise
    assert rule("pooled", 0.03, 6, 0, discordant=2)
    assert not rule("pooled", 0.03, 6, 0, discordant=6)  # |w - l| must exceed the floor
    assert not rule("pooled", 0.05, 9, 0, discordant=2)  # p must be strictly below 0.05
    assert not rule("pooled", 0.03, 6, 0, discordant=None)  # no floor, no claim
    assert not rule("seed 1234", 0.01, 8, 0, discordant=2)  # pre-registered on pooled seeds only


def test_mde_is_the_smallest_pattern_the_rule_could_call_beyond_noise():
    mde = analysis.mde_discordant
    # 6 one-sided discordant pairs give exact p = 0.03125; 5 give 0.0625, so 6 is the floor-free
    # minimum and any noise floor below 6 leaves it unchanged.
    assert mde(0) == 6
    assert mde(2) == 6  # Phase A
    assert mde(5) == 6
    assert mde(6) == 7  # m must strictly exceed the floor
    assert mde(9) == 10
    assert all(analysis._exact_two_sided(mde(f), 0) < analysis.ALPHA for f in range(12))
    assert analysis._exact_two_sided(5, 0) > analysis.ALPHA  # 6 really is the smallest
    assert mde(0, max_m=4) is None  # no qualifying pattern within the search range


def test_mde_line_is_computed_per_phase(outputs):
    summary, _ = outputs
    text = (summary / "tables.md").read_text()
    # synthetic tree: noise floor 0, 4 tasks x 2 seeds pooled -> 6 pairs of 8 rows
    assert ("- Phase A minimum detectable effect: 6 discordant pairs in one direction, "
            "pooled |diff| >= 0.750 (n 8 control rows, noise floor 0)." in text)


def test_mde_line_absent_without_a_noise_floor(tmp_path):
    res = one_seed_control(tmp_path)
    analysis.run(res, tmp_path / "figures")
    assert "minimum detectable effect" not in (res / "summary" / "tables.md").read_text()


def test_pooled_sizes_count_control_rows_at_both_seeds(results):
    groups = analysis.group_runs(analysis.load_runs(results))
    assert analysis.pooled_sizes(groups) == {"A": 8}


def test_large_pooled_win_is_beyond_noise(tmp_path):
    res = tmp_path / "results"
    write_run(res, "A_C1_1234", [0, 0, 0, 1])
    write_run(res, "A_C1_4321", [0, 0, 0, 1])
    write_run(res, "A_C3_1234", [1, 1, 1, 1])
    write_run(res, "A_C3_4321", [1, 1, 1, 1])
    groups = analysis.group_runs(analysis.load_runs(res))
    floors = analysis.noise_floors(groups)
    rows = analysis.comparisons(groups, floors)
    pooled = [r for r in rows if r["scope"] == "pooled"][0]
    assert (pooled["wins"], pooled["losses"]) == (6, 0)
    assert pooled["mcnemar_p"] < 0.05
    assert pooled["beyond_noise"] is True


def test_solve_matrix_shape_and_order(outputs):
    summary, _ = outputs
    rows = read_csv(summary / "solve_matrix.csv")
    assert len(rows) == 4
    assert list(rows[0]) == ["phase", "instance_id", "A_C1_1234", "A_C1_4321", "A_C2_1234", "A_C2_4321"]
    by_id = {r["instance_id"]: r for r in rows}
    assert by_id["t2"]["A_C2_1234"] == "1"
    assert by_id["t2"]["A_C2_4321"] == "0"
    assert {v for r in rows for k, v in r.items() if k.startswith("A_")} == {"0", "1"}


def test_solve_matrix_puts_phase_a_block_before_b(results):
    write_run(results, "B_C0_1234", [0, 1, 0, 1, 1], tasks=TASKS + ["t5"])
    matrix = analysis.solve_matrix(analysis.load_runs(results))
    assert list(matrix["phase"]) == ["A"] * 4 + ["B"] * 5
    assert list(matrix.columns)[-1] == "B_C0_1234"
    a_block = matrix[matrix["phase"] == "A"]
    assert a_block["B_C0_1234"].isna().all()
    assert matrix[matrix["phase"] == "B"]["A_C1_1234"].isna().all()


def test_per_condition_seed_and_pooled_rows(outputs):
    summary, _ = outputs
    rows = read_csv(summary / "per_condition.csv")
    keys = [(r["phase"], r["condition"], r["seed"]) for r in rows]
    assert keys == [
        ("A", "C1", "1234"), ("A", "C1", "4321"), ("A", "C1", "pooled"),
        ("A", "C2", "1234"), ("A", "C2", "4321"), ("A", "C2", "pooled"),
    ]
    for col in ("resolved_rate", "patch_submitted_rate", "mean_tool_calls", "mean_graph_tool_calls",
                "graph_tool_errors_total", "mean_analyzer_calls", "gold_file_read_rate",
                "gold_file_edited_rate", "mean_wall_s", "mean_prompt_tokens", "mean_completion_tokens",
                "incomplete"):
        assert col in rows[0]
    assert all(r["incomplete"] == "False" for r in rows)
    c2_pooled = rows[5]
    assert float(c2_pooled["resolved_rate"]) == pytest.approx(3 / 8)
    assert int(c2_pooled["graph_tool_errors_total"]) == 8
    assert float(c2_pooled["mean_tool_calls"]) == pytest.approx(11.5)
    assert float(c2_pooled["gold_file_read_rate"]) == pytest.approx(0.5)


def test_geometry_summary(outputs):
    summary, _ = outputs
    rows = {(r["group"], r["quantity"]): r for r in read_csv(summary / "geometry.csv")}
    rel = rows[("released", "mean_cos")]
    assert float(rel["mean"]) == pytest.approx(0.85)
    assert float(rel["min"]) == pytest.approx(0.8)
    assert float(rel["max"]) == pytest.approx(0.9)
    assert rel["n"] == "2"
    assert float(rows[("qwen3", "eff_dim")]["mean"]) == pytest.approx(55.0)
    assert float(rows[("gemma", "dup_rate")]["max"]) == pytest.approx(0.15)
    assert float(rows[("node_ids", "id_parity")]["min"]) == pytest.approx(0.5)
    assert float(rows[("node_ids", "released_only")]["mean"]) == pytest.approx(1.0)
    cov = rows[("gold_coverage", "released_fraction")]
    assert cov["n"] == "3"  # the n_gold 0 task is skipped
    assert float(cov["mean"]) == pytest.approx(0.5)
    cov_a = rows[("gold_coverage_phase_A", "released_fraction")]
    assert cov_a["n"] == "2"  # t1, t2 (t3 has no gold, t4 not audited, x9 not in phase A)
    assert float(cov_a["mean"]) == pytest.approx(0.75)


def test_audit_csv_with_a_leading_comment_line_is_read(results, tmp_path):
    # graphswap.cli._write_csv can prefix a '#' line documenting the columns; the reader
    # must skip it instead of taking it for the header.
    for name in ("audit.csv", "audit_gemma.csv", "audit_tasks.csv"):
        path = results / name
        path.write_text("# columns documented here\n" + path.read_text())
    geo = analysis.geometry(results, analysis.load_runs(results))
    rows = {(r["group"], r["quantity"]): r for r in geo.to_dict("records")}
    assert rows[("released", "mean_cos")]["mean"] == pytest.approx(0.85)
    assert rows[("gold_coverage", "released_fraction")]["n"] == 3
    analysis.plot_geometry(results, tmp_path / "figures")  # same reader
    assert (tmp_path / "figures" / "fig_geometry.png").stat().st_size > 0


def test_tables_md_and_figures(outputs):
    summary, figures = outputs
    text = (summary / "tables.md").read_text()
    assert "| pooled |" in text
    assert "INCOMPLETE:" not in text
    assert "0.125" in text  # pooled diff to 3 decimals
    for name in ("fig_solve_matrix_A", "fig_tool_profile", "fig_geometry"):
        for ext in ("png", "svg"):
            path = figures / f"{name}.{ext}"
            assert path.exists() and path.stat().st_size > 0, path
    assert sorted(p.name for p in figures.iterdir()) == sorted(
        f"{n}.{e}" for n in ("fig_solve_matrix_A", "fig_tool_profile", "fig_geometry")
        for e in ("png", "svg"))  # no combined fig_solve_matrix


def test_markdown_numbers_round_half_up():
    assert analysis._num(0.0625) == "0.063"
    assert analysis._num(0.1875) == "0.188"
    assert analysis._num(-0.0416666) == "-0.042"
    assert analysis._num(0.0208333, signed=True) == "+0.021"
    assert analysis._num(-0.0001, signed=True) == "+0.000"


def test_phase_without_control_yields_no_comparisons(results, tmp_path):
    write_run(results, "B_C0_1234", [0, 1, 0, 1])
    write_run(results, "B_C3_1234", [1, 1, 0, 1])
    figures = tmp_path / "figures"
    analysis.main(["--results", str(results), "--figures", str(figures)])
    summary = results / "summary"
    assert {r["phase"] for r in read_csv(summary / "comparisons.csv")} == {"A"}
    assert {r["phase"] for r in read_csv(summary / "noise_floor.csv")} == {"A"}
    per = read_csv(summary / "per_condition.csv")
    assert ("B", "C3", "1234") in {(r["phase"], r["condition"], r["seed"]) for r in per}
    assert {r["incomplete"] for r in per if r["phase"] == "B"} == {"True"}
    assert {r["incomplete"] for r in per if r["phase"] == "A"} == {"False"}
    assert "INCOMPLETE: phase B: no C1 runs" in (summary / "tables.md").read_text()
    with pytest.raises(ValueError, match="phase B: no C1 runs"):
        analysis.main(["--results", str(results), "--figures", str(figures), "--strict"])


def test_treatment_at_one_seed_has_no_pooled_row(results):
    (results / "runs" / "A_C2_4321.csv").unlink()
    groups = analysis.group_runs(analysis.load_runs(results))
    rows = analysis.comparisons(groups, analysis.noise_floors(groups))
    assert [r["scope"] for r in rows] == ["seed 1234"]
    assert rows[0]["incomplete"] is True
    with pytest.raises(ValueError, match="C2 is paired with C1 only at seed"):
        analysis.check_design(groups, strict=True)


def test_treatment_seed_missing_from_control_is_skipped_and_marked(results, tmp_path):
    (results / "runs" / "A_C2_4321.csv").unlink()
    write_run(results, "A_C2_999", [1, 1, 1, 1])
    out = analysis.run(results, tmp_path / "figures")
    c2 = out["comparisons"][out["comparisons"]["condition"] == "C2"]
    assert list(c2["scope"]) == ["seed 1234"]  # 999 skipped, so no pooled row either
    assert c2["incomplete"].all()
    per = out["per_condition"]
    assert per[per["condition"] == "C2"]["incomplete"].all()
    assert not per[per["condition"] == "C1"]["incomplete"].any()
    text = (results / "summary" / "tables.md").read_text()
    headline = text.split("## Headline", 1)[1].lstrip()
    assert headline.startswith("INCOMPLETE: phase A: run A_C2_999")


def test_treatment_seed_missing_from_control_raises_when_strict(results, tmp_path):
    (results / "runs" / "A_C2_4321.csv").unlink()
    write_run(results, "A_C2_999", [1, 1, 1, 1])
    with pytest.raises(ValueError, match="A_C2_999"):
        analysis.run(results, tmp_path / "figures", strict=True)
    with pytest.raises(ValueError, match="A_C2_999"):
        analysis.main(["--results", str(results), "--figures", str(tmp_path / "figures"), "--strict"])


def one_seed_control(tmp_path):
    res = tmp_path / "results"
    write_run(res, "A_C1_1234", [1, 0, 0, 0])
    write_run(res, "A_C2_1234", [1, 1, 0, 0])
    write_run(res, "A_C2_4321", [1, 1, 0, 0])  # no C1 at 4321: skipped
    write_audits(res)
    return res


def test_one_seed_control_keeps_per_seed_rows_and_marks_phase(tmp_path):
    res = one_seed_control(tmp_path)
    out = analysis.run(res, tmp_path / "figures")
    assert out["noise_floor"].empty
    comp = out["comparisons"]
    assert list(comp["scope"]) == ["seed 1234"]
    assert comp["incomplete"].all() and not comp["beyond_noise"].any()
    assert out["per_condition"]["incomplete"].all()  # the whole phase, C1 included
    text = (res / "summary" / "tables.md").read_text()
    assert "INCOMPLETE: phase A: C1 has only seed 1234" in text
    assert "INCOMPLETE: phase A: run A_C2_4321" in text


def test_one_seed_control_raises_when_strict(tmp_path):
    res = one_seed_control(tmp_path)
    with pytest.raises(ValueError, match="C1 has only seed 1234"):
        analysis.run(res, tmp_path / "figures", strict=True)


def test_extra_unmatched_seed_also_drops_the_pooled_row(results):
    write_run(results, "A_C2_999", [1, 1, 1, 1])  # C2 matches both C1 seeds and has a stray one
    groups = analysis.group_runs(analysis.load_runs(results))
    rows = analysis.comparisons(groups, analysis.noise_floors(groups))
    assert [r["scope"] for r in rows] == ["seed 1234", "seed 4321"]
    assert all(r["incomplete"] for r in rows)


@pytest.mark.parametrize("strict", [False, True])
def test_three_control_seeds_raise_regardless(tmp_path, strict):
    res = tmp_path / "results"
    for seed in ("1234", "4321", "5555"):
        write_run(res, f"A_C1_{seed}", [1, 0, 0, 0])
    write_audits(res)
    with pytest.raises(ValueError, match="C1 has 3 seeds"):
        analysis.run(res, tmp_path / "figures", strict=strict)


def test_task_missing_from_treatment_run_raises(results, tmp_path):
    write_run(results, "A_C2_4321", [1, 0, 0], tasks=TASKS[:3])
    with pytest.raises(ValueError, match="t4"):
        analysis.run(results, tmp_path / "figures")


def test_solve_blocks_sort_within_each_phase(results):
    # t5 is B-only and solved by 1 of 2 B runs; t1 is solved by every A run but no B run
    write_run(results, "B_C0_1234", [0, 1, 0, 0, 1], tasks=TASKS + ["t5"])
    write_run(results, "B_C0_4321", [0, 1, 0, 0, 0], tasks=TASKS + ["t5"])
    rows = analysis.load_runs(results)
    blocks = analysis.solve_blocks(analysis.solve_matrix(rows), analysis.run_meta(rows))
    assert [b["phase"] for b in blocks] == ["A", "B"]
    assert blocks[0]["order"] == ["t1", "t2", "t3", "t4"]
    assert blocks[0]["runs"] == ["A_C1_1234", "A_C1_4321", "A_C2_1234", "A_C2_4321"]
    assert blocks[1]["order"] == ["t2", "t5", "t1", "t3", "t4"]
    assert blocks[1]["runs"] == ["B_C0_1234", "B_C0_4321"]


def test_solve_layout_keeps_sixty_by_six_legible():
    pitch, font, height = analysis.solve_layout(n_rows=24)
    assert (pitch, font) == (analysis.SM_PITCH, analysis.SM_FONT)
    fixed = analysis.SM_TITLE + analysis.SM_LEGEND + analysis.SM_HEADER
    assert height == pytest.approx(fixed + 24 * pitch)
    pitch, font, height = analysis.solve_layout(n_rows=60)
    assert font >= 6.0 and height <= 11.0
    assert pitch * 72 >= font  # rows never overlap their labels


def test_empty_cell_legend_swatch_is_visible():
    plain = analysis.solve_legend_handles(any_blank=False)
    assert [h.get_label() for h in plain] == ["resolved", "not resolved"]
    handles = analysis.solve_legend_handles(any_blank=True)
    empty = handles[-1]
    assert empty.get_label() == "task not in run"
    # an outline on the figure surface: the edge must differ from both the surface it sits on
    # and the fill, and be drawn (non-zero width, dashed), or the swatch reads as nothing
    assert empty.get_edgecolor() != empty.get_facecolor()
    assert analysis._rgb(analysis.MUTED) == empty.get_edgecolor()[:3]
    assert analysis._rgb(analysis.SURFACE) == empty.get_facecolor()[:3]
    assert empty.get_linewidth() >= 1.0
    assert empty.get_linestyle() != "solid"


def test_two_runs_for_one_condition_and_seed_raise(results):
    write_run(results, "A_C2_1234_rerun", [1, 1, 1, 1])
    with pytest.raises(ValueError, match="A_C2_1234_rerun"):
        analysis.group_runs(analysis.load_runs(results))


def test_bad_boolean_raises(results):
    path = results / "runs" / "A_C2_4321.csv"
    rows = read_csv(path)
    rows[0]["gold_file_read"] = "maybe"
    _write(path, RUN_COLS, rows)
    with pytest.raises(ValueError, match="gold_file_read"):
        analysis.per_condition(analysis.load_runs(results))


@pytest.mark.workspace
def test_real_phase_a_matches_ledger():
    """Today's C0 vs C1 numbers as recorded in docs/experiments.md."""
    res = ROOT / "results"
    needed = [res / "runs" / f"A_{c}_{s}.csv" for c in ("C0", "C1") for s in (1234, 4321)]
    if not all(p.exists() for p in needed):
        pytest.skip("Phase A C0/C1 run CSVs absent")
    groups = analysis.group_runs(analysis.load_runs(res))
    floors = analysis.noise_floors(groups)
    assert floors["A"]["discordant"] == 2
    rows = {(r["condition"], r["scope"]): r for r in analysis.comparisons(groups, floors) if r["phase"] == "A"}
    pooled = rows[("C0", "pooled")]
    assert pooled["diff"] == pytest.approx(0.0208, abs=1e-3)
    assert (pooled["wins"], pooled["losses"]) == (2, 1)
    assert pooled["mcnemar_p"] == 1.0
    assert (pooled["ci_lo"], pooled["ci_hi"]) == pytest.approx((0.0, 0.0625))
    assert pooled["beyond_noise"] is False
    assert rows[("C0", "seed 1234")]["diff"] == pytest.approx(-1 / 24)
    assert rows[("C0", "seed 4321")]["diff"] == pytest.approx(2 / 24)
    assert not math.isnan(pooled["control_rate"])


def test_solve_matrix_is_also_drawn_one_figure_per_phase(results, tmp_path):
    write_run(results, "B_C0_1234", [0, 1, 0, 0, 1], tasks=TASKS + ["t5"])
    write_run(results, "B_C0_4321", [0, 1, 0, 0, 0], tasks=TASKS + ["t5"])
    rows = analysis.load_runs(results)
    matrix, meta = analysis.solve_matrix(rows), analysis.run_meta(rows)
    for phase in ("A", "B"):
        layout = analysis.plot_solve_matrix(matrix, meta, tmp_path, phase=phase)
        assert (tmp_path / f"fig_solve_matrix_{phase}.png").stat().st_size > 0
        assert (tmp_path / f"fig_solve_matrix_{phase}.svg").stat().st_size > 0
    assert layout["height_in"] < 11.0
    with pytest.raises(TypeError):  # the stacked all-phases mode is gone: a phase is required
        analysis.plot_solve_matrix(matrix, meta, tmp_path)
    with pytest.raises(ValueError, match="phase 'C'"):
        analysis.plot_solve_matrix(matrix, meta, tmp_path, phase="C")
