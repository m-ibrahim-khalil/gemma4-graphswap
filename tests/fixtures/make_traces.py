"""Generate the synthetic trace fixtures under `tests/fixtures/traces/`.

Everything here is invented: the repos (`acme/widgetlib`, `acme/gearbox`), the modules, the
file paths, the queries, the snippets and the numbers. No string is taken from a real agent
run, a real task or a real repository; see `tests/test_fixture_hygiene.py` for the rule.

The files imitate the shape `graphswap.traces` parses (an ATIF-v1.7 trace: `steps[]` with
`step_id`/`source`/`message`/`extra`, optional `tool_calls[]` and one `observation`, plus
`final_metrics`), and they deliberately keep the quirks the parser has to survive:

- free text is trimmed to 200 characters, as the stored fixtures were. No generated message
  is that long, but the longer observation payloads are, so they are cut mid-token and no
  longer parse: `_is_failure` must treat an unparseable observation as "not a failure";
- a failing graph call (`get_code_neighbors`) whose short error payload *does* parse, so
  `graph_tool_errors` is non-zero on one trace;
- failing `run_command` calls on the other trace, which must NOT reach `graph_tool_errors`;
- a step that issues no tool call, a system step that carries only an observation, and a
  sub-agent (`code_analyzer`) handoff;
- a step issuing two parallel graph calls behind a single observation, which is why
  `graph_tool_errors` is documented as a lower bound;
- an `edit_file` on a file that is not the gold file and a `write_file` on the gold file,
  so editing and submitting stay distinguishable (the run that edits is the run that never
  submits) and both edit tools are exercised against the gold path;
- a `read_file` path with the sandbox's `/workspace/` prefix, which `_norm` has to strip
  before the gold-file comparison, and an early read of a path the agent guessed wrong;
- `final_metrics` totals that differ from the sum of the per-step metrics, so the preference
  for `final_metrics` is observable;
- budget warnings on the last observations, and a run that stops without `submit_patch`.

Regenerate with `python tests/fixtures/make_traces.py`;
`test_trace_fixtures_are_what_the_generator_writes` fails if the committed files drift.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / "traces"

TRIM = 200  # the stored traces keep only the first 200 characters of free text
MODEL = "fixture-lm-8b"
SANDBOX = "sandbox 00000000-fix"
START = 1700000000.0  # fixed clock: the generator must be byte-for-byte reproducible

MANIFEST = {"run": "fixture", "condition": "graph", "seed": 7, "resolved": 1, "tasks_completed": 2}

RESULTS = [
    {"instance_id": "widgetlib_101", "repo": "acme/widgetlib", "resolved": False,
     "agent_patch_size": 0, "test_exit_code": -1, "duration_seconds": 274.5,
     "error": "stopped: session time limit reached (240 s)", "tool_calls": 18,
     "total_llm_calls": 20, "task_index": 3},
    {"instance_id": "gearbox_202", "repo": "acme/gearbox", "resolved": True,
     "agent_patch_size": 241, "test_exit_code": 0, "duration_seconds": 193.25,
     "error": None, "tool_calls": 10, "total_llm_calls": 12, "task_index": 1},
]

# --- invented payload bodies -------------------------------------------------------------

FRAME_SRC = (
    "class Frame:\n    def __init__(self, cells, border=\"|\"):\n        self.cells = cells\n"
    "        self.border = border\n\n    def render(self, width):\n"
    "        rows = pack_rows(self.cells, width)\n"
    "        return [self.border.join(row) for row in rows]\n"
)
PACK_ROWS_SRC = (
    "def pack_rows(cells, width):\n    row, rows = [], []\n    for cell in cells:\n"
    "        if sum(len(c) for c in row) + len(cell) > width:\n"
    "            rows.append(row)\n            row = []\n        row.append(cell)\n"
    "    return rows + [row]\n"
)
PAINTER_SRC = (
    "class Painter:\n    def fill_cell(self, cell, colour):\n"
    "        if not cell.strip():\n            return cell\n"
    "        return f\"<{colour}>{cell}</{colour}>\"\n"
)


def _ok(**fields):
    return {"status": "ok", **fields}


def _err(error_type, message):
    return {"status": "error", "error_type": error_type, "error_message": message}


# --- step builders -----------------------------------------------------------------------


class Builder:
    """Accumulates steps with monotonic ids and a deterministic clock."""

    def __init__(self, budget):
        self.steps = []
        self.calls = 0
        self.budget = budget  # the run's tool-call allowance, so the warnings stay coherent

    def _clock(self):
        return round(START + 12.5 * len(self.steps), 3)

    def _extra(self, author, event_type=None):
        extra = {"elapsed_s": round(1.5 + 12.5 * len(self.steps), 3), "timestamp": self._clock()}
        if event_type is not None:
            extra = {"event_type": event_type, **extra}
        extra["author"] = author
        return extra

    def _budget_warning(self):
        """The harness nudge, with the counters taken from the calls actually issued."""
        return (f"tool budget: {self.budget - self.calls} of {self.budget} calls left "
                f"({self.calls} used); wrap up the edit.")

    def _observation(self, tool_name, payload, author, warn=False):
        if warn:
            payload = {**payload, "budget_warning": self._budget_warning()}
        content = payload if isinstance(payload, str) else json.dumps(payload)
        return {"content": content[:TRIM],
                "extra": {"tool_name": tool_name, "elapsed_s": round(2.0 + 12.5 * len(self.steps), 3),
                          "timestamp": round(self._clock() + 0.5, 3), "author": author}}

    def harness(self, event_type):
        self.steps.append({
            "step_id": len(self.steps) + 1,
            "source": "system" if event_type != "task_prompt" else "user",
            "message": "[stripped]",
            "extra": self._extra("harness", event_type),
        })

    def subagent_result(self, content, author="coder"):
        """The system step that hands a `code_analyzer` answer back to the coder."""
        step = {
            "step_id": len(self.steps) + 1,
            "source": "system",
            "observation": self._observation("code_analyzer", content, author),
            "extra": {"elapsed_s": round(1.5 + 12.5 * len(self.steps), 3), "timestamp": self._clock()},
            "message": "[stripped]",
        }
        self.steps.append(step)

    def agent(self, message, *, author="coder", tool=None, arguments=None, calls=None,
              observation=None, observation_tool=None, warn=False, metrics=None,
              event_type="thinking"):
        """One agent step. `tool`/`arguments` issue a single call; `calls` is a list of
        (function_name, arguments) pairs for a step that issues several in parallel. The
        harness keeps ONE observation per step however many calls it carried, so
        `observation_tool` names the call the observation belongs to."""
        if calls is None:
            calls = [(tool, arguments or {})] if tool is not None else []
        step = {
            "step_id": len(self.steps) + 1,
            "source": "agent",
            "message": message[:TRIM],
            "model_name": MODEL,
        }
        if calls:
            step["tool_calls"] = []
            for name, args in calls:
                self.calls += 1
                step["tool_calls"].append({
                    "tool_call_id": f"call_{self.calls}",
                    "function_name": name,
                    "arguments": args,
                    "extra": {"elapsed_s": round(1.5 + 12.5 * len(self.steps), 3),
                              "timestamp": self._clock(), "author": author, "model_version": MODEL},
                })
        if observation is not None:
            step["observation"] = self._observation(
                observation_tool or calls[0][0], observation, author, warn)
        if metrics is not None:
            prompt, completion = metrics
            step["metrics"] = {"prompt_tokens": prompt, "completion_tokens": completion,
                               "cached_tokens": 0, "total_tokens": prompt + completion}
        step["llm_call_count"] = 1
        step["extra"] = self._extra(author, event_type)
        return self.steps.append(step)

    def trace(self, prompt_total, completion_total):
        return {
            "schema_version": "ATIF-v1.7",
            "agent": {"name": "fixture-agent", "version": "1.0"},
            "steps": self.steps,
            "final_metrics": {
                "total_prompt_tokens": prompt_total,
                "total_completion_tokens": completion_total,
                "total_cached_tokens": 0,
                "total_tokens": prompt_total + completion_total,
                "total_steps": len(self.steps),
            },
        }


# --- trace 1: widgetlib_101, a run that burns its budget and never submits -----------------


def widgetlib_trace() -> dict:
    b = Builder(budget=18)
    b.harness("system_instruction")
    b.harness("task_prompt")
    b.agent("The task is a wrong line width in Frame.render. I need the definition first.",
            tool="code_analyzer", arguments={"request": "where is Frame.render defined?"},
            metrics=(2400, 70))
    # A long "ok" payload: trimmed to 200 chars, so it no longer parses as JSON.
    b.agent("I will look the symbol up with search_similar_code.", author="code_analyzer",
            tool="search_similar_code", arguments={"query": "render_frame"},
            observation=_ok(query="render_frame", results=[
                {"node_name": "widgetlib.layout.pack_rows", "code": PACK_ROWS_SRC, "score": 0.43},
                {"node_name": "widgetlib.frame.Frame", "code": FRAME_SRC, "score": 0.39}]),
            metrics=(1150, 50))
    b.agent("No exact hit. I will walk the neighbours of the Frame class instead.",
            author="code_analyzer", tool="get_code_neighbors",
            arguments={"node": "widgetlib.frame.Frame"},
            observation=_ok(node="widgetlib.frame.Frame", neighbors=[
                "acme/widgetlib;widgetlib.frame.Frame.render",
                "acme/widgetlib;widgetlib.layout.pack_rows",
                "acme/widgetlib;widgetlib.frame.Frame.__init__",
                "acme/widgetlib;widgetlib.theme.Border"]),
            metrics=(2900, 330))
    # A guessed path that does not exist: short enough that the error payload parses.
    b.agent("Reading the module the neighbours point at.", author="code_analyzer",
            tool="read_file", arguments={"filepath": "widgetlib/frame.py"},
            observation=_err("FileReadError", f"no such file in {SANDBOX}: /workspace/widgetlib/frame.py"),
            metrics=(3380, 110))
    b.agent("Trying the bare class name as the query.", author="code_analyzer",
            tool="search_similar_code", arguments={"query": "Frame"},
            observation=_ok(query="Frame", results=[
                {"node_name": "widgetlib.frame.Frame", "code": FRAME_SRC, "score": 0.52},
                {"node_name": "widgetlib.theme.Border", "code": "class Border:\n    glyph = \"|\"\n",
                 "score": 0.31}]),
            metrics=(3550, 270))
    # Two graph calls in one step. The harness keeps a single observation for the step, and
    # it is a failure short enough to parse: graph_tool_errors counts the step once although
    # two graph calls were issued, which is the lower bound traces.py documents.
    b.agent("Now the neighbours of the method and of the helper, in one go.",
            author="code_analyzer",
            calls=[("get_code_neighbors", {"node": "widgetlib.frame.Frame.render"}),
                   ("get_code_neighbors", {"node": "widgetlib.layout.pack_rows"})],
            observation=_err("GraphNodeNotFound", "node 'widgetlib.frame.Frame.render' is not in the graph"),
            observation_tool="get_code_neighbors",
            metrics=(5700, 780))
    b.agent("Falling back to the package entry point.", author="code_analyzer",
            tool="read_file", arguments={"filepath": "widgetlib/__init__.py"},
            observation=_err("FileReadError", f"no such file in {SANDBOX}: /workspace/widgetlib/__init__.py"),
            metrics=(6600, 410))
    b.agent("Pulling the subgraph around the layout helper.", author="code_analyzer",
            tool="get_code_subgraph", arguments={"nodes": ["widgetlib.layout.pack_rows"]},
            observation=_ok(nodes=["widgetlib.layout.pack_rows"], edges=[], node_count=1,
                            edge_count=0, is_truncated=False),
            metrics=(7270, 220))
    # A step with no tool call at all, then the sub-agent's closing message with no metrics.
    b.agent("The symbol path says the method lives in the frame module; the sources are under "
            "a src/ directory, which is why the first read missed.", author="code_analyzer",
            metrics=(8530, 4400))
    b.agent("Frame.render is defined in src/widgetlib/frame.py and calls pack_rows.",
            author="code_analyzer", event_type="final")
    b.subagent_result(json.dumps({"raw": "Frame.render is defined in src/widgetlib/frame.py and "
                                         "calls widgetlib.layout.pack_rows for the row packing."}))
    b.harness("continuation_nudge")
    # The sandbox hands back an absolute path; _norm has to strip /workspace/ to match gold.
    b.agent("Reading the real file now.", tool="read_file",
            arguments={"filepath": "/workspace/src/widgetlib/frame.py"},
            observation=_ok(filepath="src/widgetlib/frame.py", content=FRAME_SRC),
            metrics=(2920, 120))
    b.agent("Locating the method by line number.", tool="run_command",
            arguments={"command": "grep -n \"def render\" src/widgetlib/frame.py"},
            observation=_ok(stdout="6:    def render(self, width):\n", stderr="", exit_code=0),
            metrics=(4240, 60))
    b.agent("Reading the packing helper around the suspect lines.", tool="read_file",
            arguments={"filepath": "/workspace/src/widgetlib/layout.py", "start_line": 1,
                       "end_line": 40},
            observation=_ok(filepath="src/widgetlib/layout.py", content=PACK_ROWS_SRC),
            metrics=(4370, 160))
    # An edit attempt outside the workspace: an EDIT_TOOLS call that is not on a gold file.
    b.agent("Writing a reproduction script.", tool="write_file",
            arguments={"filepath": "/tmp/repro_frame.py",
                       "content": "from widgetlib.frame import Frame\n\nprint(Frame([\"ab\", \"cd\"]).render(3))\n"},
            observation=_err("FileWriteError", "refused to write outside the sandbox root: '/tmp/repro_frame.py'"),
            metrics=(7950, 750))
    b.agent("Reproducing inline instead.", tool="run_command",
            arguments={"command": "python3 -c \"from widgetlib.frame import Frame; print(Frame(['ab','cd']).render(3))\""},
            observation=_ok(stdout="[['ab'], ['cd']]\n", stderr="", exit_code=0),
            metrics=(9680, 60))
    # An edit_file on a file that is NOT the gold file: this run edits and still never
    # submits, so patch_submitted cannot be inferred from the edit tools.
    b.agent("Widening the border glyph while I am here.", tool="edit_file",
            arguments={"filepath": "src/widgetlib/theme.py",
                       "old_string": "glyph = \"|\"", "new_string": "glyph = \"||\""},
            observation=_ok(filepath="src/widgetlib/theme.py", occurrences=1, strategy="unescaped",
                            diff="--- a/src/widgetlib/theme.py\n+++ b/src/widgetlib/theme.py\n"
                                 "@@ -1,2 +1,2 @@\n-    glyph = \"|\"\n+    glyph = \"||\"\n"),
            metrics=(9750, 210))
    # The gold file is rewritten with write_file, not edit_file: the summariser has to treat
    # both as edits. The run still never submits, so it stays unresolved.
    b.agent("Rewriting the render method with the width fix.", tool="write_file",
            arguments={"filepath": "src/widgetlib/frame.py",
                       "content": FRAME_SRC.replace("pack_rows(self.cells, width)",
                                                    "pack_rows(self.cells, width - 1)")},
            observation=_ok(filepath="src/widgetlib/frame.py", bytes_written=214),
            metrics=(9700, 540))
    b.agent("One more check before the real edit.", tool="run_command",
            arguments={"command": "python3 -c \"from widgetlib.layout import MAX_WIDTH; print(MAX_WIDTH)\""},
            observation=_ok(stdout="80\n", stderr="", exit_code=0), warn=True,
            metrics=(9840, 350))
    b.agent("Checking the border change took.", tool="run_command",
            arguments={"command": "python3 -c \"from widgetlib.theme import Border; print(Border.glyph)\""},
            observation=_ok(stdout="||\n", stderr="", exit_code=0), warn=True,
            metrics=(9900, 120))
    # final_metrics deliberately disagrees with the per-step sums (sub-agent contexts reset).
    return b.trace(104560, 8130)


# --- trace 2: gearbox_202, a run that edits the gold file and submits ----------------------


def gearbox_trace() -> dict:
    b = Builder(budget=10)
    b.harness("system_instruction")
    b.harness("task_prompt")
    b.agent("Blank cells come back wrapped in colour tags. Find the painter first.",
            tool="code_analyzer",
            arguments={"request": "Where is the cell painter implemented, and what fills a cell?"},
            metrics=(3800, 150))
    # A free-text query, as the agents often issue: it cannot resolve to a node name.
    b.agent("Searching for the painter.", author="code_analyzer",
            tool="search_similar_code", arguments={"query": "cell painter"},
            observation=_ok(query="cell painter", results=[
                {"node_name": "gearbox.painter.Painter.fill_cell", "code": PAINTER_SRC, "score": 0.47},
                {"node_name": "gearbox.palette.Palette.lookup",
                 "code": "def lookup(self, name):\n    return self.table[name]\n", "score": 0.22}]),
            metrics=(1170, 70))
    b.agent("Reading the painter module.", author="code_analyzer", tool="read_file",
            arguments={"filepath": "gearbox/painter.py"},
            observation=_ok(filepath="gearbox/painter.py", content=PAINTER_SRC),
            metrics=(3090, 180))
    b.agent("Reading the tail of the module.", author="code_analyzer", tool="read_file",
            arguments={"filepath": "gearbox/painter.py", "start_line": 120},
            observation=_ok(filepath="gearbox/painter.py",
                            content="    def fill_row(self, row, colour):\n"
                                    "        return [self.fill_cell(c, colour) for c in row]\n"),
            metrics=(4760, 190))
    b.agent("fill_cell returns the cell untouched when it is blank, but the caller still wraps "
            "it, so the guard belongs in fill_cell.", author="code_analyzer", metrics=(5820, 770))
    b.agent("The painter is gearbox/painter.py; fill_cell is the place to guard.",
            author="code_analyzer", event_type="final")
    b.subagent_result(json.dumps({"raw": "The cell painter is gearbox/painter.py. fill_cell wraps "
                                         "every cell, including the blank ones."}))
    b.harness("continuation_nudge")
    b.agent("Reading the file myself before editing.", tool="read_file",
            arguments={"filepath": "gearbox/painter.py", "start_line": 1},
            observation=_ok(filepath="gearbox/painter.py", content=PAINTER_SRC),
            metrics=(4410, 130))
    # Two failing non-graph calls: parseable errors that must not reach graph_tool_errors.
    b.agent("Running the painter tests.", tool="run_command",
            arguments={"command": "python3 -m pytest tests/test_painter.py -x"},
            observation=_err("CommandError", "exit 4: file or directory not found: tests/test_painter.py"),
            metrics=(6030, 240))
    b.agent("Looking for the right test module.", tool="run_command",
            arguments={"command": "python3 -m pytest tests/painter -x"},
            observation=_err("CommandError", "exit 4: file or directory not found: tests/painter"),
            metrics=(8350, 210))
    b.agent("Guarding the blank cell.", tool="edit_file",
            arguments={"filepath": "gearbox/painter.py",
                       "old_string": "        if not cell.strip():\n            return cell",
                       "new_string": "        if not cell or not cell.strip():\n            return cell"},
            observation=_ok(filepath="gearbox/painter.py", occurrences=1, strategy="unescaped",
                            diff="--- a/gearbox/painter.py\n+++ b/gearbox/painter.py\n@@ -2,4 +2,4 @@\n"
                                 "-        if not cell.strip():\n+        if not cell or not cell.strip():\n"),
            metrics=(7040, 670))
    b.agent("Checking the fix.", tool="run_command",
            arguments={"command": "python3 -c \"from gearbox.painter import Painter; print(Painter().fill_cell('', 'red'))\""},
            observation=_ok(stdout="\n", stderr="", exit_code=0), warn=True,
            metrics=(8720, 290))
    b.agent("The blank cell is left alone now. Submitting.", tool="submit_patch", arguments={},
            observation=_ok(patch_size=241, files_changed=1),
            metrics=(11360, 40))
    return b.trace(118740, 9480)


def write_all(dest: Path) -> None:
    traces = dest / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    (dest / "run_manifest.json").write_text(json.dumps(MANIFEST, indent=1) + "\n")
    (dest / "task_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in RESULTS))
    for iid, trace in (("widgetlib_101", widgetlib_trace()), ("gearbox_202", gearbox_trace())):
        (traces / f"trace_{iid}.json").write_text(json.dumps(trace, indent=1) + "\n")


if __name__ == "__main__":
    write_all(DEST)
    print(f"wrote {DEST}")
