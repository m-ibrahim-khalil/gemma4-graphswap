# harness-tooling

The condition tooling the authors wrote around the official harness, copied from the
authors' separate eval workspace (branch `conditions`, commit `52fb02a`). It is here so the
conditions in `docs/preregistration.md` are inspectable and rerunnable. Only files the
authors wrote are included.

| file | what it is |
|---|---|
| `conditions/C0.yaml` | no code graph: `read_file` and `run_command` only |
| `conditions/C1.yaml` | control: the competition's released graphs and embeddings |
| `conditions/C2.yaml` | released graphs, node embeddings recomputed with Qwen3-Embedding-0.6B (1024-d) |
| `conditions/C3.yaml` | graphs rebuilt from source, embedded with Qwen3-Embedding-0.6B (1024-d) |
| `conditions/C4.yaml` | C3 plus free-text query embedding |
| `prompts/analyzer.md` | the analyzer sub-agent prompt |
| `Makefile` | `run`, `eval`, `push`, `pull`, `status`, `flips` targets for the eval loop |
| `tools/kaggle_eval.py` | pushes the eval kernel, waits, pulls, and checks the run manifest and graph-tool coverage |
| `tools/flips.py` | per-task diff of two runs, with the control check |
| `tools/make_subset.py`, `tools/presubmit.sh` | subset drawing and local submission validation |

EmbeddingGemma appears in the paper only as an offline embedding-geometry comparison. It
was never a run condition.

## Placeholders and what is not here

Kaggle competition input paths are placeholders of the form
`/kaggle/input/<slug>/...`; substitute your own competition slug. Running any of this
needs your own Kaggle account, access to the competition, and GPU quota.

Deliberately not included: the eval kernel template (`kaggle/eval_kernel.py`) and the
`submission/` tree that `kaggle_eval.py` and `presubmit.sh` read, any harness (swegemma)
source, competition task text, released graphs or embeddings, and run outputs. The two
pre-registered subsets are `eval/subset24.txt` and `eval/subset60.txt` at the repo root; the
Makefile's default `SUBSET` names the authors' original file, so pass `SUBSET=` explicitly.
These files are a record of the tooling and are not a self-contained runnable package.
The tooling's own tests (flips, kernel rendering, pull) stayed in the authors' workspace: the
published copies carry no tests here.
