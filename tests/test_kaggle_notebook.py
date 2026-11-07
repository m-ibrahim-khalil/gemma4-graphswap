"""The public Kaggle notebook: generated from this repo, and equal to analysis.py.

The notebook cannot import the repo, so it carries a copy of the statistics and of the
analysis code. These tests are what keeps the copy honest: it is byte-identical to what
``notebooks/build_kaggle_notebook.py`` generates from the two modules, and running it
produces exactly the tables ``notebooks/analysis.py`` produces.
"""

import hashlib
import importlib.util
import json
import re
import tempfile
from pathlib import Path

import nbclient  # dev group; a plain import, so a stripped environment fails loudly
import nbformat  # rather than skipping the one test that compares the notebook to analysis.py
import pytest
from hygiene import check_competition_paths

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "kaggle" / "analysis_notebook.ipynb"
TABLES = ("comparisons", "noise_floor", "per_condition", "solve_matrix", "geometry")


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


analysis = _module("analysis", ROOT / "notebooks" / "analysis.py")
build = _module("build_kaggle_notebook", ROOT / "notebooks" / "build_kaggle_notebook.py")


@pytest.fixture(scope="module")
def cells():
    nb = json.loads(NOTEBOOK.read_text())
    return [("".join(c["source"]), c) for c in nb["cells"]]


def _cell(cells, needle: str) -> str:
    hits = [src for src, _ in cells if needle in src]
    assert len(hits) == 1, f"expected exactly one cell containing {needle!r}, found {len(hits)}"
    return hits[0]


def test_committed_notebook_is_exactly_what_the_builder_generates():
    assert NOTEBOOK.read_text() == build.notebook_text(), (
        "kaggle/analysis_notebook.ipynb is stale: run "
        "`uv run python notebooks/build_kaggle_notebook.py`"
    )


def test_vendored_statistics_are_the_pre_registered_module_verbatim(cells):
    cell = _cell(cells, "def _exact_two_sided")
    module = (ROOT / "graphswap" / "stats.py").read_text()
    assert cell.endswith(module.rstrip("\n"))
    provenance = cell.splitlines()[0]  # path, digest of the copy, the pre-registration's hash
    assert "graphswap/stats.py" in provenance
    assert build.digest(module) in provenance
    assert f"docs/preregistration.md, sha256 {build.PREREGISTRATION_SHA256}" in provenance
    for name in ("N_BOOT", "_to_bool", "_index", "_compare", "_exact_two_sided", "paired",
                 "noise_floor"):
        assert name in cell


def test_vendored_analysis_is_the_module_without_its_driver(cells):
    cell = _cell(cells, "def comparisons")
    module = (ROOT / "notebooks" / "analysis.py").read_text()
    assert cell.endswith(build.analysis_source().rstrip("\n"))
    assert build.digest(module) in cell.splitlines()[0]
    # everything that computes a number is there; only the file-writing driver is not
    defs = [line for line in module.splitlines() if line.startswith("def ")]
    driver = module.partition(build.DRIVER_BANNER)[2]
    for line in defs:
        assert (line in cell) is (line not in driver), line
    for dropped in build.DROP_LINES:
        assert dropped.strip() not in cell


def test_notebook_imports_nothing_from_the_repo(cells):
    for src, cell in cells:
        if cell["cell_type"] != "code":
            continue
        for line in src.splitlines():
            code = line.split("#", 1)[0]
            assert "import graphswap" not in code and "from graphswap" not in code, line
            assert "import analysis" not in code, line


def check_kaggle_paths(text: str) -> None:
    """Every Kaggle path in the notebook is the one mount constant and the one slug constant.

    Scoped to occurrences, not lines: a line that mentions the placeholder must not thereby
    excuse a real competition path written beside it. This is the guard against publishing a
    path into competition data, so it is checked on the notebook, which is what gets published.
    """
    check_competition_paths(text)  # the rule shared with README.md and harness-tooling/
    mounts = re.findall(r"/kaggle/input[^\s`'\")]*", text)
    assert mounts.count("/kaggle/input") == 1, mounts  # the one constant
    assert all(m.startswith("/kaggle/input/<slug>")  # every other mention is the placeholder
               for m in mounts if m != "/kaggle/input"), mounts
    # and the constant is only ever joined with the slug constant, so a path built by joining
    # it with another literal (which carries no "/kaggle/input" of its own) is caught too
    assert re.findall(r"KAGGLE_INPUT\s*/\s*([^\s/]+)", text) == ["DATASET_SLUG"], text
    # the assignment, that one join, and the two searches below it (the datasets/<owner>/<slug>
    # glob and the last-resort walk); none of them spells a path
    assert text.count("KAGGLE_INPUT") == 4, text


def test_notebook_names_one_kaggle_mount_and_one_slug(cells):
    text = "".join(src for src, _ in cells)
    check_kaggle_paths(text)
    assert text.count('Path("/kaggle/input")') == 1  # one constant behind all three candidates
    assert text.count(build.DATASET_SLUG) == 2  # the constant, and the attach instruction
    assert "/Users/" not in text and str(ROOT) not in text  # no local path leaked in


SMUGGLED = [
    'Path("/kaggle/input/real-competition")  # like /kaggle/input/<slug>',  # placeholder beside
    "# like /kaggle/input/<slug>, also /kaggle/input/real-competition",  # ... and on one line
    'OTHER = KAGGLE_INPUT / "real-competition"',  # a join that spells no mount path at all
    'OTHER = Path("/kaggle/input/arc-prize-2026")',
    "KAGGLE_INPUT = Path('/kaggle/input')",  # a second mount constant
]


@pytest.mark.parametrize("smuggled", SMUGGLED)
def test_hygiene_rule_catches_a_competition_path_next_to_the_placeholder(cells, smuggled):
    text = "".join(src for src, _ in cells)
    check_kaggle_paths(text)  # the committed notebook passes
    with pytest.raises(AssertionError):
        check_kaggle_paths(text + "\n" + smuggled)


def test_notebook_states_what_it_reads_and_the_pre_registration(cells):
    intro = cells[0][0]
    # cited by file hash and timestamp proof: the public repository is one squashed commit, so
    # no development-history commit (the pre-registration's was faf0fb2) resolves for a reader
    assert f"`docs/preregistration.md` (sha256\n`{build.PREREGISTRATION_SHA256}`)" in intro
    assert f"OpenTimestamps proof (`{build.PREREGISTRATION_OTS}`)" in intro
    assert f"earliest Bitcoin attestation in {build.PREREGISTRATION_STAMP}" in intro
    assert "before the Phase B treatment runs, not before Phase A" in intro  # no overclaim
    text = "".join(src for src, _ in cells)
    assert "faf0fb2" not in text and "0bc2e87" not in text and "commit `" not in text
    assert "resolution rate" in intro
    for phrase in ("task text", "graph content", "query strings"):
        assert phrase in intro


def test_cited_pre_registration_hash_is_the_frozen_file():
    prereg = ROOT / "docs" / "preregistration.md"
    assert hashlib.sha256(prereg.read_bytes()).hexdigest() == build.PREREGISTRATION_SHA256
    listed = (ROOT / "docs" / "preregistration.sha256").read_text().splitlines()
    assert f"{build.PREREGISTRATION_SHA256}  docs/preregistration.md" in listed
    assert (ROOT / build.PREREGISTRATION_OTS).is_file()


def test_notebook_is_committed_with_cleared_outputs(cells):
    for _, cell in cells:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def _run_config(cells, monkeypatch, cwd: Path, kaggle_input: Path | None = None) -> dict:
    """Execute the input-resolution cell with the Kaggle mount point pointed at a temp dir.

    Only the one line that hard-codes /kaggle/input is replaced (asserted present), so the
    resolution order and the rest of the cell are the committed ones.
    """
    source = _cell(cells, "def find_results")
    mount = 'KAGGLE_INPUT = Path("/kaggle/input")\n'
    assert mount in source
    namespace = {"KAGGLE_INPUT": kaggle_input if kaggle_input is not None else cwd / "no-kaggle"}
    monkeypatch.setattr(tempfile, "tempdir", str(cwd))  # the cell's mkdtemp stays in tmp_path
    monkeypatch.chdir(cwd)
    exec(source.replace(mount, ""), namespace)
    return namespace


def _write_results(directory: Path) -> Path:
    directory.mkdir(parents=True)
    (directory / "runs").mkdir()
    for name in ("audit.csv", "audit_tasks.csv", "audit_gemma.csv"):
        (directory / name).write_text("")
    return directory


def test_attached_dataset_is_found_at_the_mount_point(tmp_path, monkeypatch, cells):
    kaggle = tmp_path / "input"
    wanted = _write_results(kaggle / build.DATASET_SLUG)
    assert _run_config(cells, monkeypatch, tmp_path, kaggle)["RESULTS"] == wanted


def test_attached_dataset_is_found_with_results_inside_it(tmp_path, monkeypatch, cells):
    # a dataset built by uploading the repo's results/ directory mounts one level deeper
    kaggle = tmp_path / "input"
    wanted = _write_results(kaggle / build.DATASET_SLUG / "results")
    assert _run_config(cells, monkeypatch, tmp_path, kaggle)["RESULTS"] == wanted


def test_attached_dataset_is_found_under_the_current_datasets_owner_layout(tmp_path, monkeypatch, cells):
    # Kaggle now mounts it at datasets/<owner>/<slug>/, files at the top, no old flat directory
    kaggle = tmp_path / "input"
    # a complete copy that sorts before ours: only the datasets/*/<slug> candidate, tried ahead
    # of the last-resort walk, picks the right one, so removing that candidate fails this test
    _write_results(kaggle / "aaa-decoy")
    wanted = _write_results(kaggle / "datasets" / "an-owner" / build.DATASET_SLUG)
    assert _run_config(cells, monkeypatch, tmp_path, kaggle)["RESULTS"] == wanted


def test_current_layout_with_results_inside_the_dataset(tmp_path, monkeypatch, cells):
    kaggle = tmp_path / "input"
    _write_results(kaggle / "aaa-decoy")  # see the test above
    wanted = _write_results(kaggle / "datasets" / "an-owner" / build.DATASET_SLUG / "results")
    assert _run_config(cells, monkeypatch, tmp_path, kaggle)["RESULTS"] == wanted


def test_a_renamed_dataset_is_found_by_its_contents(tmp_path, monkeypatch, cells):
    kaggle = tmp_path / "input"
    wanted = _write_results(kaggle / "datasets" / "an-owner" / "renamed-copy")
    _write_results(kaggle / "datasets" / "an-owner" / "z-later-copy")  # sorted: the first wins
    assert _run_config(cells, monkeypatch, tmp_path, kaggle)["RESULTS"] == wanted


def test_the_last_resort_walk_is_bounded(tmp_path, monkeypatch, cells):
    kaggle = tmp_path / "input"
    _write_results(kaggle / "a" / "b" / "c" / "d" / "too-deep")  # five levels down
    with pytest.raises(FileNotFoundError):
        _run_config(cells, monkeypatch, tmp_path, kaggle)


def test_a_named_candidate_wins_over_the_last_resort(tmp_path, monkeypatch, cells):
    kaggle = tmp_path / "input"
    _write_results(kaggle / "aaa-first-alphabetically")
    wanted = _write_results(kaggle / "datasets" / "an-owner" / build.DATASET_SLUG)
    assert _run_config(cells, monkeypatch, tmp_path, kaggle)["RESULTS"] == wanted


def test_repo_checkout_is_found_by_walking_up(tmp_path, monkeypatch, cells):
    (tmp_path / "pyproject.toml").write_text("")
    wanted = _write_results(tmp_path / "results")
    (tmp_path / "kaggle").mkdir()  # the notebook's own directory in a checkout
    assert _run_config(cells, monkeypatch, tmp_path / "kaggle")["RESULTS"] == wanted


def test_the_dataset_wins_over_a_checkout(tmp_path, monkeypatch, cells):
    kaggle = tmp_path / "input"
    wanted = _write_results(kaggle / build.DATASET_SLUG)
    (tmp_path / "pyproject.toml").write_text("")
    _write_results(tmp_path / "results")
    assert _run_config(cells, monkeypatch, tmp_path, kaggle)["RESULTS"] == wanted


def test_missing_inputs_raise_naming_every_candidate_and_what_it_lacks(tmp_path, monkeypatch, cells):
    kaggle = tmp_path / "input"
    partial = kaggle / build.DATASET_SLUG  # an incomplete upload: runs/ only
    (partial / "runs").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        _run_config(cells, monkeypatch, tmp_path, kaggle)
    message = str(excinfo.value)
    assert f"{partial} (missing audit.csv, audit_tasks.csv, audit_gemma.csv)" in message
    assert f"{partial / 'results'} (missing runs," in message
    assert f"{tmp_path / 'results'} (missing runs," in message
    assert build.DATASET_SLUG in message
    assert "0 match(es) of datasets/*/" in message and "within four levels" in message


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    """The committed notebook, run top to bottom, with a cell appended that exports its tables."""
    out = tmp_path_factory.mktemp("notebook-tables")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell(
        f"import pathlib\n_out = pathlib.Path({str(out)!r})\n"
        f"for _name in {TABLES!r}:\n"
        '    tables[_name].to_csv(_out / f"{_name}.csv", index=False)\n'
    ))
    nbclient.NotebookClient(
        nb, timeout=900, resources={"metadata": {"path": str(NOTEBOOK.parent)}}
    ).execute()
    return out


def test_notebook_reproduces_every_table_of_analysis_py(executed):
    expected = analysis.compute(ROOT / "results")
    for name in TABLES:
        assert (executed / f"{name}.csv").read_text() == expected[name].to_csv(index=False), name
