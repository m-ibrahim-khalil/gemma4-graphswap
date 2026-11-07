"""The one rule for Kaggle input paths, shared by every test that guards a published file.

The placeholder is ``/kaggle/input/<slug>`` everywhere: the published notebook, README.md
and harness-tooling/. Occurrence-scoped, not line-scoped: a line that mentions the
placeholder must not excuse a real path written beside it. A mount may appear bare (prose
such as "under /kaggle/input"), but anything below it must be the placeholder, and a mount
joined with a literal is a path too.
"""

import re

PLACEHOLDER = "/kaggle/input/<slug>"
# the leading slash is optional so that "kaggle/input/real" is caught as well; only a word
# character or hyphen before it excuses a match, so "./" and "../" prefixes are still caught
MOUNT = re.compile(r"(?<![\w-])/?kaggle/input(?P<rest>[^\s`'\")]*)")
# a mount constant joined with a literal spells no full path: Path("/kaggle/input") / "x",
# os.path.join("/kaggle/input", "x"), "/kaggle/input/" + "x"
JOINED = re.compile(r"""["']/?kaggle/input/?["']\s*\)?\s*(?:/|,|\+)\s*["']""")


def _is_placeholder(rest: str) -> bool:
    if rest.rstrip("/") == "":
        return True  # the bare mount
    tail = ("/" + rest.lstrip("/"))
    if not tail.startswith(PLACEHOLDER[len("/kaggle/input"):]):
        return False
    after = tail[len(PLACEHOLDER) - len("/kaggle/input"):]
    return after == "" or not re.match(r"[\w<-]", after)  # not a longer slug: /<slug>-real


def check_competition_paths(text: str) -> None:
    for m in MOUNT.finditer(text):
        assert _is_placeholder(m.group("rest")), f"real Kaggle input path: {m.group(0)!r}"
    joined = JOINED.search(text)
    assert not joined, f"Kaggle input mount joined with a literal: {joined.group(0)!r}"


# --- the rule for test fixtures: nothing under tests/fixtures/ comes from a real run ------
#
# The published repo ships no raw agent traces, so a fixture that names one of the four
# repositories the study ran on is a real trace (or a trimmed copy of one) that escaped the
# synthetic generator, tests/fixtures/make_traces.py. Enforced by tests/test_fixture_hygiene.py.

REAL_REPOS = ("fastapi", "httpx", "requests", "rich")
# word-bounded, so "enrich", "rich-text" and "requested" are not the repo, while the instance
# id form "rich_3278" is
REAL_REPO = re.compile(r"(?<![\w-])(?:" + "|".join(REAL_REPOS) + r")(?:_(?=\d)|(?![\w-]))", re.I)


def check_fixture_is_synthetic(text: str) -> None:
    m = REAL_REPO.search(text)
    assert not m, f"real repository name in a test fixture: {m.group(0)!r}"
    check_competition_paths(text)
