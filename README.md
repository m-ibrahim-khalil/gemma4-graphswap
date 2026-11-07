# graphswap

An end-to-end ablation of code graphs and embeddings in the official Gemma 4 Developer
Agent harness: five pre-registered conditions (C0 to C4) that change what the analyzer
sub-agent sees, scored by resolution rate on pre-registered task subsets, with paired
statistics and a measured noise floor. C4 did not run: its pre-registered rule (run only if
C3 beats C1 beyond noise in Phase A) did not trigger.

- Repository: https://github.com/m-ibrahim-khalil/gemma4-graphswap (release `v1.0`)
- Public Kaggle notebook: https://www.kaggle.com/code/ibrahimkhalil09/gemma4-graphswap-analysis
- Bundles dataset (`rebuilt_full`): https://www.kaggle.com/datasets/ibrahimkhalil09/gemma4-graphswap-bundles

## Where things are

- Pre-registration: `docs/preregistration.md`
- Experiment ledger (runs, budget, decision rules): `docs/experiments.md`
- Writeup: `paper/writeup.md`
- Per-task results, summary tables: `results/`; figures: `paper/figures/`
- Search-path classification and embedding replay summaries: `results/summary/search_replay/`
- Public reproduction notebook (runs on Kaggle from the per-task CSVs): `kaggle/analysis_notebook.ipynb`
- Condition definitions and eval tooling: `harness-tooling/`

## Reproduce

```bash
# 1. Install (Python 3.11, uv)
uv sync

# 2. Regenerate every summary table and figure from the committed per-task CSVs
#    (no Kaggle access, no GPU)
uv run python notebooks/analysis.py --strict

# 3. Run the test suite. On a fresh clone 3 tests need local data and fail without it:
#    test_loads_127_distinct_commits needs data/tasks.jsonl, and the two
#    test_npz_* tests in test_embedder.py need `make vendor`. Without them: 3 failed,
#    203 passed, 32 skipped (the skips also need `make vendor`, the network, or local data).
uv run pytest

# 4. Rebuild the graph and embedding bundles (needs the competition tasks file, cloned repos,
#    and the harness graph code via `make vendor`)
uv run graphswap build-graphs --tasks data/tasks.jsonl --repos repos --out bundles/graphs_rebuilt
uv run graphswap embed-graphs --graphs bundles/graphs_rebuilt --out bundles/emb_rebuilt \
    --model Qwen/Qwen3-Embedding-0.6B --max-tokens 1024   # the C2/C3 encoder, not the default
#    --device defaults to mps (Apple Silicon); add `--device cpu` or `--device cuda` elsewhere
uv run graphswap bundle --out bundles/out   # also reads bundles/emb_released_nodes_raw
#    optional audit and upload: `graphswap audit`, `graphswap push-bundles`

# 5. Rerun an eval condition on Kaggle (your own account and competition access)
make -C harness-tooling run RUN=my_c1 CONDITION=C1 SEED=1234 SUBSET=../eval/subset24.txt
```

Steps 1 and 2 are self-contained. Step 3 passes in full only with the local data above. Steps 4 and 5 need the competition data and a Kaggle
account, and step 5 also needs files this repo does not ship; see `harness-tooling/README.md`.
Other subcommands: `summarize` (per-task CSV for one run), `replay-search`, `compare`,
`draw-subset`.

## Deliberately absent

No competition task text, no released graphs or embeddings, and no raw run traces. Only
derived per-task results are committed. Kaggle competition input paths in this repo are
placeholders (`/kaggle/input/<slug>/...`).

The pre-registration names the Kaggle dataset `gemma4-graph-bundles`: that is the authors'
private working dataset, which also holds `emb_released_nodes`, derived from the
competition's released node texts. The public release is the bundles dataset at
https://www.kaggle.com/datasets/ibrahimkhalil09/gemma4-graphswap-bundles, which holds `rebuilt_full` only.

## License

Apache License 2.0, see `LICENSE`.
