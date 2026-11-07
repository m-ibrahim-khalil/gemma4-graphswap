# Pre-registration: graph resolution study

## Fixed settings (every run)

| setting | value |
|---|---|
| model | gemma-4-31b-it-qat-w4a16-ct |
| budgets | 8 min / 28 tool calls / 240 s command timeout / 200 turns |
| concurrency | 4 |
| sampling | temperature 0.2, top_p 0.95, max_output_tokens 16384, thinking_budget 4096 |
| seeds | 1234 and 4321 |
| harness | swegemma 0.2.11 |

## Conditions

| id | analyzer tools | graph data | embeddings | role |
|---|---|---|---|---|
| C0 | `[read_file, run_command]` | none | none | no graph tooling |
| C1 | `[search_similar_code, get_code_neighbors, get_code_subgraph, read_file]` | released graphs | released embeddings | **control** |
| C2 | as C1 | released graphs | `emb_released_nodes` | embedding change only |
| C3 | as C1 | `rebuilt_full` graphs | `rebuilt_full` embeddings | graphs and embeddings rebuilt |
| C4 | C3 + live free-text query embedding | `rebuilt_full` | `rebuilt_full` | stretch; not scorer-legal |

C0 and C1 differ in two ways by design: C0 is the strongest non-graph analyzer we can give the agent (grep via run_command), not C1 minus the graph tools.

## Bundle provenance

Embeddings are produced by `Qwen/Qwen3-Embedding-0.6B` (EmbeddingGemma was gated at the time of writing), max_tokens 1024, 1024-d.
Kaggle dataset: `ibrahimkhalil09/gemma4-graph-bundles`.

| bundle | bundle_sha |
|---|---|
| emb_released_nodes | `1ab74f38a99b5f5341d06f9ffa7c5c47d2aeb610427cbadc27007e08768ace76` |
| graphs_rebuilt | `c94423f07688cd45aa5ddab8841279b3433a6268607664baf6a6682a08ec670e` |
| rebuilt_full | `c6e0913dabd920f737d92440f0cf4b7bac26e6562625a01c875d900311aa17b2` |

## Task sets

**Phase A (24 tasks, `eval/subset24.txt`):**
fastapi_13786 fastapi_14077 fastapi_14301 fastapi_14303 fastapi_14448 fastapi_14485 fastapi_14492 fastapi_14605 fastapi_14962 fastapi_14986 fastapi_9425 httpx_3672 requests_6629 requests_7427 requests_7502 rich_3006 rich_3469 rich_3472 rich_3521 rich_3535 rich_3675 rich_3777 rich_3782 rich_3882

**Phase B (60 tasks, `eval/subset60.txt`)**, drawn by
`uv run graphswap draw-subset --tasks data/tasks.jsonl --n 60 --seed 0 --include eval/subset24.txt --out eval/subset60.txt`
(fastapi 31, httpx 1, requests 6, rich 22; contains all 24 Phase A ids):
fastapi_12942 fastapi_13786 fastapi_13920 fastapi_14077 fastapi_14186 fastapi_14297 fastapi_14301 fastapi_14303 fastapi_14356 fastapi_14361 fastapi_14371 fastapi_14430 fastapi_14448 fastapi_14458 fastapi_14459 fastapi_14463 fastapi_14485 fastapi_14492 fastapi_14512 fastapi_14583 fastapi_14605 fastapi_14609 fastapi_14851 fastapi_14962 fastapi_14964 fastapi_14986 fastapi_15745 fastapi_15763 fastapi_15785 fastapi_9425 fastapi_9555 httpx_3672 requests_6629 requests_6644 requests_7205 requests_7427 requests_7433 requests_7502 rich_3006 rich_3061 rich_3064 rich_3469 rich_3470 rich_3472 rich_3480 rich_3506 rich_3518 rich_3521 rich_3535 rich_3675 rich_3676 rich_3718 rich_3777 rich_3782 rich_3882 rich_3934 rich_3942 rich_3944 rich_4070 rich_4079

## Metrics and analysis

- Primary metric: resolution rate, paired per task against C1 at the same seed.
- Test: McNemar exact two-sided on pooled seeds; bootstrap 95% CI over tasks (10,000 resamples, seed 0).
- Noise floor: the discordant count between C1/1234 and C1/4321.
- An effect is "beyond noise" only when pooled McNemar p < 0.05 AND |wins - losses| > the noise-floor discordant count.
- Secondary metrics: graph-tool calls, graph-tool error responses, analyzer calls, gold file read, gold file edited, wall seconds, prompt and completion tokens, patch_submitted.

## Phase rules

- Phase B runs the two conditions with the largest pooled |diff| vs C1 in Phase A. If none exceeds the noise floor, Phase B runs C0 and C3.
- C4 runs only if C3 beats C1 beyond noise in Phase A.
- No run is discarded. Crashed runs are rerun with the same seed and both manifests are kept.

Nothing in this document changes after the first Phase A run; later amendments go in `docs/experiments.md` with a dated note.
