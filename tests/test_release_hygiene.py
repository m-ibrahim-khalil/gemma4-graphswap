"""What the published tree must not contain.

No real competition input path ships in the README or under harness-tooling/ (the rule is
tests/hygiene.py, shared with the notebook check in test_kaggle_notebook.py), and no published
file depends on the internal design-and-plan directory, which the release leaves out.
"""

import subprocess
from pathlib import Path

import pytest
from hygiene import check_competition_paths

ROOT = Path(__file__).resolve().parent.parent

# The design spec and plan live under docs/ in a directory the published tree omits, next to
# the private .superpowers/; spelled in parts so that this file does not cite it itself.
EXCLUDED_DIRS = ("/".join(("docs", "superpowers")), ".superpowers")
EXCLUDED_DOCS = EXCLUDED_DIRS[0]
# what .gitignore keeps out, plus what `uv sync` and pytest create: skipped when there is no
# git index to ask (an exported tree), so the walk sees only the files that ship
UNTRACKED = {".git", ".venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
             ".superpowers", "data", "repos", "bundles", "cache", "vendor"}


def _files() -> list[Path]:
    tooling = [p for p in (ROOT / "harness-tooling").rglob("*")
               if p.is_file() and "__pycache__" not in p.parts]
    return sorted(tooling + [ROOT / "README.md"])


def test_the_release_files_are_found():
    names = {p.relative_to(ROOT).as_posix() for p in _files()}
    assert {"README.md", "harness-tooling/README.md", "harness-tooling/Makefile",
            "harness-tooling/conditions/C2.yaml", "harness-tooling/tools/kaggle_eval.py"} <= names


@pytest.mark.parametrize("path", _files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_real_competition_input_path(path):
    check_competition_paths(path.read_text())


PASSES = [
    "see /kaggle/input/<slug>/tasks.jsonl (use your own slug)",
    "mounts under /kaggle/input",
    "# /kaggle/input/<slug>/a and /kaggle/input/<slug>/b",
    "`/kaggle/input/<slug>`, `/kaggle/input/<slug>/...`.",
    "kaggle/input/<slug>/x",
    "../kaggle/input/<slug>/x and ./kaggle/input/<slug>/y",
]
SMUGGLED = [
    "/kaggle/input/real-competition/tasks.jsonl",
    "/kaggle/input/<slug>/a, also /kaggle/input/real-competition",  # beside the placeholder
    'INPUT = Path("/kaggle/input") / "real-competition"',  # a join
    'os.path.join("/kaggle/input", "real-competition")',
    "/kaggle/input/<slug>-real-competition/tasks.jsonl",  # merely starts with the placeholder
    "/kaggle/input/<slugx>",  # not the placeholder
    'KAGGLE = "kaggle/input/real-competition/tasks.jsonl"',  # no leading slash
    'os.path.join("kaggle/input", "real-competition")',
    '"../kaggle/input/real-competition/x"',  # parent-relative
    '"./kaggle/input/real-competition/x"',
    '"../../kaggle/input/real-competition/x"',
]


@pytest.mark.parametrize("text", PASSES)
def test_placeholder_passes(text):
    check_competition_paths(text)


@pytest.mark.parametrize("text", SMUGGLED)
def test_real_path_fails(text):
    with pytest.raises(AssertionError):
        check_competition_paths(text)


def _git_files() -> list[str] | None:
    """Tracked files, when ROOT is the top of a git checkout; None in an exported tree."""
    try:
        top = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True).stdout.strip()
        if Path(top).resolve() != ROOT:
            return None  # an export placed inside some other repository
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [name for name in out.split("\0") if name]


def _published_files() -> list[str]:
    """Every file that ships, as a repo-relative posix path: outside the excluded directories."""
    names = _git_files()
    if names is None:
        names = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                 if p.is_file() and not UNTRACKED & set(p.relative_to(ROOT).parts)
                 and not p.relative_to(ROOT).as_posix().startswith("results/traces/")]
    return sorted(n for n in names if not any(n == d or n.startswith(d + "/") for d in EXCLUDED_DIRS))


def check_no_excluded_docs(data: bytes) -> None:
    assert EXCLUDED_DOCS.encode() not in data, f"cites {EXCLUDED_DOCS}/, which is not published"


def test_published_files_are_found():
    names = set(_published_files())
    assert {"README.md", "paper/sources.md", "paper/writeup.md", "docs/experiments.md",
            "kaggle/analysis_notebook.ipynb", "tests/test_release_hygiene.py"} <= names
    assert not any(n.startswith(EXCLUDED_DOCS + "/") for n in names)


def test_no_published_file_cites_the_excluded_design_docs():
    """The design spec and plan are internal working documents left out of the release, so a
    citation of them anywhere else would dangle for a reader of the published repository."""
    offenders = []
    for name in _published_files():
        try:
            check_no_excluded_docs((ROOT / name).read_bytes())
        except AssertionError:
            offenders.append(name)
    assert offenders == [], offenders


@pytest.mark.parametrize("text", [
    "spec: `docs/" + "superpowers/specs/design.md` line 63",
    "see docs/" + "superpowers/plans/plan.md",
])
def test_a_citation_of_the_excluded_docs_fails(text):
    with pytest.raises(AssertionError):
        check_no_excluded_docs(text.encode())


@pytest.mark.parametrize("text", [".superpowers/ in .gitignore", "superpowers (a skill library)"])
def test_other_mentions_pass(text):
    check_no_excluded_docs(text.encode())
