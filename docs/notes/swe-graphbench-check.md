# swe-graphbench builder check

Time-boxed check (well under the one-hour budget) of `natemeton-hub/swe-graphbench`
(shallow clone in a scratch directory outside the repo).

## What was run

- `python -m swegraphbench.build_graph <repo_root> out.json --name pkg` (pure stdlib, no install).
- Run 1: pointed directly at `tests/fixtures/pkg`. Output ids are not rooted at the
  package name (`core.Thing`, root module id is the empty string), because it treats the
  given directory as the package root.
- Run 2: a scratch copy of the fixture placed one level down (`repo/pkg/...`). Ids are
  `pkg.core.fetch_async` etc., matching the harness convention.
- Run 2's output was loaded with the vendored harness `load_graph_from_json`
  (swegemma 0.2.11), and `resolve_node_name("core.fetch_async", names)` returned
  `pkg.core.fetch_async`.

## Result

- Passes the harness-load test and suffix resolution.
- Emits `is_async` (only `pkg.core.fetch_async` is true), plus `contains`, `imports`,
  `inherits`, `calls`; the call edges recorded on the fixture match ours.
- Differences from our builder: no `async_function` or `method` kinds (kind is
  `module | class | function`), call strategy attr is named `resolution` not `via`, and it
  has not been checked for `src/` layout handling (not tested).

## Decision

Viable as an alternative builder. Not wired in this task (the `--builder` flag belongs to
Task 6's CLI); our own `graphswap.builder` is required regardless and is the default.
