# Number sources for paper/writeup.md

Every number in the writeup body, by section, with the repo file it comes from. "tables.md" is `results/summary/tables.md`; "ledger" is `docs/experiments.md` (including its "Facts recorded for the writeup" section); "prereg" is `docs/preregistration.md`; "replay" is `results/summary/search_replay/replay.summary.csv` (the `unit=call` rows; its leading `#` line defines the columns); "runs" is `results/runs/<run>.csv` (identical to `results/summary/search_replay/summ_<run>.csv`); "comparisons.csv", "per_condition.csv", "noise_floor.csv" and "solve_matrix.csv" are the files of those names in `results/summary/`. All six Phase B runs are in (B_C0_4321, ledger row 18, done 2026-11-07), so the design is complete and `notebooks/analysis.py --strict` regenerated every summary file and figure quoted below. External numbers cite the reference list in `paper/references.md`. Ratios marked "a/b" are the only numbers computed for the paper; both a and b appear in the named file.

## abstract

- 24 tasks; control seeds disagree on 2: tables.md, Noise floor.
- +0.021, McNemar p 1.000, inside the floor: tables.md, Comparisons against C1 (A, C0, pooled).
- C2 +0.042 (p 0.500); C3 +0.021 (p 1.000); no condition beyond the floor: tables.md, Headline and Comparisons against C1 (A, C2, pooled; A, C3, pooled; `beyond_noise` no in every pooled row).
- Regenerated embeddings change nearly every embedding-based search result: replay, `frac_same_list` 0.035 to 0.051 in the four C1/C2 runs (see Section 4.3 below).
- 31B: model id in prereg, Fixed settings.
- 60 confirmation tasks; control seeds disagree on 3: noise_floor.csv row B (n 60, discordant 3); tables.md, Noise floor (B).
- Rebuilt graphs -0.008, p 1.000, inside its noise floor: comparisons.csv row (B, C3, pooled): diff -0.00833, mcnemar_p 1.0, wins 2, losses 3 (|wins - losses| 1 <= floor 3), beyond_noise False; tables.md, Comparisons against C1 (B, C3, pooled); ledger, "Phase B C3 vs C1, pooled seeds" ("Not beyond noise").
- Grep analyzer -0.008, p 1.000 on the 60 tasks: comparisons.csv row (B, C0, pooled): diff -0.00833, mcnemar_p 1.0, wins 2, losses 3 (|wins - losses| 1 <= floor 3), beyond_noise False, incomplete False; tables.md, Headline ("Phase B C0 vs C1, pooled seeds") and Comparisons against C1 (B, C0, pooled); ledger, "Phase B C0 vs C1, pooled seeds" ("Not beyond noise"). As in Phase A, neither clears the floor: comparisons.csv `beyond_noise` False in rows (A, C0, pooled), (B, C0, pooled), (A, C3, pooled) and (B, C3, pooled); ledger, same line ("Agrees with Phase A").

## 1

- 8 minutes per task: prereg, Fixed settings.
- 31B model on four L4 GPUs: prereg (model id); ledger, Quota (L4x4) and Facts (tensor parallel over the 4 L4 GPUs).
- 8 of 30 tasks changed outcome: reference [4], PAPER.md Section 5 (external).

## 2

- swegemma 0.2.11: prereg, Fixed settings.
- Root coder tools; conditions change only the analyzer's tools: ledger, Facts ("Agent tools").
- Lexical fallback; four-tier name resolution: `graphswap/traces.py` (module docstring: a query that does not resolve to a node falls back to lexical token scoring, swegemma 0.2.11 `retrieval_utils.get_similar_nodes`; `resolution_tier`: exact, suffix, case-insensitive, substring, in that order, `graph_utils.py` lines 378-431); `graphswap/replay.py` (module docstring: lexical scoring never reads the embeddings).
- 5,000 characters per tool call: ledger, Facts ("Harness constants", `max_stdout_chars = 5000`).
- 127 released embedding sets, one per base commit: `results/audit.csv` (127 rows, one per `repo_short`, `commit`, each with released node counts and released-vector geometry).
- Mean random-pair cosine 0.830, mean effective dimension 3.138 (min 1.726, max 4.224), duplicate rate 0.162: tables.md, Embedding geometry (released rows).
- Definitions (participation ratio of the covariance eigenvalues; share of nodes whose vector has a byte-identical twin): `graphswap/audit.py` `embedding_geometry`; axis labels of `paper/figures/fig_geometry.png`.
- Gold coverage 0.931 over 121 tasks; 0.855 over 22 Phase A tasks; rebuilt 1.000: tables.md, Embedding geometry (gold_coverage and gold_coverage_phase_A rows). Gold definition: `graphswap/audit.py` `gold_symbols`.
- Fig. 1 caption, released effective dimension ranges from 1.726 to 4.224 (the min and max in the line above); both regenerated sets average 50.7: tables.md (qwen3 50.682, gemma 50.688); ledger, Amendments (EmbeddingGemma, "effective dimension 50.7 (Qwen3 50.7, released 3.14)").

## 3

- Conditions, analyzer tool lists, scorer-legal status: ledger, Conditions. Scorer applies the same harness and caps: ledger, Facts ("Resolved").
- C0/C1 asymmetry sentence: ledger, Conditions; prereg, Conditions.
- 0.878 of released node ids in rebuilt graphs: tables.md, Embedding geometry (node_ids id_parity).
- Released `calls` as a symmetric union; four rebuilt edge kinds; no-filter ruling before Phase A: ledger, Facts ("Edge types").
- Qwen3-Embedding-0.6B, 1024 tokens, 1024-d: prereg, Bundle provenance.
- EmbeddingGemma vectors for the rebuilt graphs, added by dated amendment, offline comparison only: ledger, Amendments (2026-10-24); ledger, Phase A outcome (EmbeddingGemma does not become a run condition).
- adk-submission 0.2.13, google-adk 1.36.1, vLLM 0.19.1, Kaggle model slug and version 2, `max_model_len` 32,768, tensor parallel over 4 L4: ledger, Facts ("Inference") and Fixed across every run. (The ledger's `gpu_memory_utilization` 0.90 and Gemma 4 tool-call and reasoning parsers are no longer quoted in the writeup.)
- Budgets 8 min, 28 harness tool calls per task shared by root coder and analyzer (`submit_patch` and `get_status` exempt), 240 s, 200 turns: prereg, Fixed settings; ledger, Facts ("Harness constants", "Agent tools").
- Temperature 0.2, top_p 0.95, 16,384 output tokens, 4,096 thinking budget, concurrency 4, seeds 1234 and 4321: prereg, Fixed settings.
- Tool-agnostic, byte-identical analyzer prompt: ledger, Fixed across every run.
- Mismatched manifests refused: `harness-tooling/tools/kaggle_eval.py` (module docstring: a condition is validated, rendered into the kernel, recorded in the run manifest and checked again by `pull`; the kernel checks the content hash of the dataset it mounted; `pull` exits non-zero when the run manifest's run, submission tree hash, condition or seed differs from what was pushed); `harness-tooling/tools/flips.py` `compare_manifests` (names each differing condition input, including both bundle shas, and warns that the comparison is confounded); ledger, run rows ("manifest ok", with bundle shas).
- Phase A: 24 tasks, seed 0, stratified by repo, fastapi 11, rich 9, requests 3, httpx 1: ledger, Facts ("Phase A task mix").
- Phase B: 60 tasks, seed 0, contains all 24, fastapi 31, rich 22, requests 6, httpx 1: prereg, Task sets.
- "Resolved" definition: ledger, Facts ("Resolved").
- McNemar exact two-sided; 10,000 resamples, seed 0; beyond-noise rule; Phase B selection rule; no run discarded: prereg, Metrics and analysis and Phase rules. Bootstrap clustered by task: ledger, Phase A comparisons ("bootstrap clustered by task per pre-registration").
- Context-window overflows counted as unresolved: ledger, runs 6 and 7.
- Pre-registration file and sha256 `83cda15eb084d9d6462096977e6a295c1a9c6d7d9970bf595a6604d65f24e92b`: `docs/preregistration.sha256` line 1 (recomputable with `shasum -a 256 docs/preregistration.md`). It lists both task sets: prereg, Task sets.
- Unchanged since a clarification made before the first Phase A run: ledger, Facts ("Pre-registration": the clarification came two minutes after the pre-registration, before any Phase A run); the prereg's own last line (nothing changes after the first Phase A run). Not "before the first run": ledger row 0 is a baseline C1 run pushed 2026-10-23, before the pre-registration, and not used (C1 was re-run).
- Manifest holds the sha256 of `eval/subset60.txt` (`d7daddbd57e8a8a598467c802e197d241a6d1e28c887892adf059eebdcddce90`): `docs/preregistration.sha256` line 2.
- OpenTimestamps proof `docs/preregistration.sha256.ots`, earliest Bitcoin attestation block 973351 (2026-10-27 05:26 UTC; manifest submitted 05:17 UTC): the `.ots` file (its Bitcoin attestations are at heights 973351, 973378, 973384, 973423; `ots info` prints them); block time from that block's header (`ots verify` against a Bitcoin node or explorer).
- Stamp before any Phase B treatment run, after Phase A and the Phase B controls: ledger, planned-runs rows 5-12 (Phase A pushed 2026-10-24), 13-14 (B_C1 pushed 2026-10-26), 15-17 (B_C3_1234, B_C3_4321, B_C0_1234 pushed 2026-11-01) and 18 (B_C0_4321 pushed 2026-11-07, after the stamp; the sentence names no run count).
- Phase B same-seed C1 controls: ledger, Amendments (2026-10-24, Phase B control).

## 4

### 4.1, 4.2

- C1 resolved 6 and 4 of 24: tables.md, Per condition (A, C1, seeds 1234 and 4321).
- Discordant 2 (fastapi_14492, fastapi_9425), McNemar p 0.500, 95% CI [-0.208, 0.000]: tables.md, Noise floor; task ids from ledger, Phase A comparisons.
- Minimum detectable effect, 6 discordant pairs in one direction, |diff| at least 0.125 on Phase A; Phase B floor 3 does not raise the 6-pair minimum because p < 0.05 binds first (the Phase B |diff|, 0.050, is quoted in 4.5): ledger, Facts ("Minimum detectable effect"); tables.md, Headline (Phase A and Phase B minimum detectable effect lines).
- Across all eight Phase A runs, 3 always resolved and 18 never resolved: tables.md, Solve matrix (runs 8, always_solved 3, never_solved 18). Only 3 tasks ever changed outcome (fastapi_14492, fastapi_9425, fastapi_14077): ledger, Phase A outcome; tables.md, Solve matrix rows.
- Fig. 2 caption, the three varying tasks: tables.md, Solve matrix (rows fastapi_9425, fastapi_14492, fastapi_14077 are the only rows with both 0 and 1); `paper/figures/fig_solve_matrix_A.png` (8/8, 7/8, 6/8 labels).
- C0 vs C1 pooled: n 48, 24 clusters, 0.229 vs 0.208, diff +0.021, wins 2, losses 1, p 1.000, CI [0.000, 0.063], not beyond noise: tables.md, Comparisons against C1. Per-seed task ids: ledger, Phase A comparisons.
- Superseded (moved out of the paper): an earlier row-level bootstrap interval (-0.042, 0.104) for the pooled C0 vs C1 comparison was replaced by the task-clustered interval: ledger, Phase A comparisons.

### 4.3 (C2 vs C1)

- Pooled C2 0.250 vs C1 0.208, +0.042, wins 2, losses 0, McNemar p 0.500, CI [0.000, 0.104], not beyond noise: tables.md, Comparisons against C1 (A, C2, pooled); ledger, "C2 vs C1, pooled seeds".
- Seed 1234 resolved exactly C1's tasks: tables.md, Comparisons against C1 (A, C2, seed 1234: wins 0, losses 0, ties 24); ledger, "C2 vs C1, seed 1234".
- Both wins are the noise-floor tasks at seed 4321 (fastapi_14492, fastapi_9425): ledger, "C2 vs C1, seed 4321" and "C2 vs C1, pooled seeds".
- Embedding-path share per run, n_resolved/n_calls: replay, A_C1_1234 75/122 = 0.615, A_C1_4321 85/137 = 0.620, A_C2_1234 69/107 = 0.645, A_C2_4321 59/96 = 0.615 (range 0.615 to 0.645). The same counts are the column sums of `search_resolved` and `search_calls` in runs.
- Substring-tier share of resolved calls, n_substring/n_resolved: replay, 14/75 = 0.187, 24/85 = 0.282, 13/69 = 0.188, 17/59 = 0.288 (range 0.187 to 0.288).
- Substring tier = a query without whitespace that is contained in node names resolves to the candidate with the fewest dotted parts, then the shortest, then alphabetical: vendored swegemma 0.2.11 `graph/graph_utils.py` lines 420-431 (sort key at line 430), in the main checkout's `vendor/swegemma` (gitignored, so not in this worktree); column definition in `results/summary/search_replay/README.md` ("arbitrary anchor"). Embedding path = query resolved to a node (includes the substring tier): same README.
- Mean top-k overlap 0.151, 0.126, 0.146, 0.158 (range 0.126 to 0.158); same top-1 0.080, 0.118, 0.087, 0.136 (range 0.080 to 0.136); identical list 0.040, 0.035, 0.043, 0.051 (range 0.035 to 0.051): replay, columns `mean_overlap`, `frac_top1_same`, `frac_same_list` for A_C1_1234, A_C1_4321, A_C2_1234, A_C2_4321. Overlap = |topA & topB| / min(k, n_nodes - 1) over embedding-path calls: replay header line. Set A = released embeddings, set B = released-node Qwen3 embeddings (`emb_released_nodes`): README.
- Reproduction of recorded embedding-path lists: replay, `observed_a_embed`/`n_observed_embed` A_C1_1234 60/61, A_C1_4321 71/74; `observed_b_embed`/`n_observed_embed` A_C2_1234 56/58, A_C2_4321 47/51. (Not quoted: set B reproduces 3/61 and 2/74 of the C1 lists, set A 1/58 and 1/51 of the C2 lists.)

### 4.4 (C3 vs C1, C3 vs C2, Phase A outcome)

- Pooled C3 0.229 vs C1 0.208, +0.021, wins 2, losses 1, p 1.000, CI [-0.042, 0.083], not beyond noise: tables.md, Comparisons against C1 (A, C3, pooled).
- Wins fastapi_14492 and fastapi_9425 at seed 4321; loss fastapi_14077 at seed 1234: ledger, "C3 vs C1, seed 1234" and "C3 vs C1, seed 4321".
- fastapi_14077 resolved by every other Phase A run: tables.md, Solve matrix (row fastapi_14077: 0 only in A_C3_1234). Patch submitted but not resolved: runs, A_C3_1234.csv row fastapi_14077 (`resolved` False, `patch_submitted` True).
- C3 vs C2, exploratory and not pre-registered: 0.229 vs 0.250, -0.021, wins 0, losses 1 (fastapi_14077), p 1.0, CI [-0.063, 0.000]: ledger, "Exploratory, not pre-registered: C3 vs C2, pooled seeds". Ledger (-0.0625, 0.0), rounded half-up to 3 decimals as `notebooks/analysis.py` does for tables.md (0.0625 -> 0.063).
- C2 and C3 share the embedding model (Qwen3-Embedding-0.6B) and differ in graphs: ledger, Conditions and Bundles; exploratory line ("embeddings rebuilt in both").
- Full gold coverage of the rebuilt graphs: refers back to Section 2 (no new number).
- Phase A outcome (no condition beyond noise; A_C1_4321 the only run that missed both fastapi_14492 and fastapi_9425, so each treatment's two seed-4321 wins are control seed variance; 4/192 context-window overruns kept as unresolved, none on a task any run resolved; Phase B runs C0 and C3 plus C1; C4 not run; EmbeddingGemma not a run condition): ledger, Phase A outcome paragraph; tables.md, Solve matrix (A_C1_4321 column is 0 for both tasks; every other column has at least one 1).
- C4 runs only if C3 beats C1 beyond noise: prereg, Phase rules. C4 not run: ledger, planned-runs row "A_C4_<seed>" (Section 4.7 of v0.3 folded into this paragraph). EmbeddingGemma becomes a run condition only if C2 beats C1 beyond noise, and does not: ledger, Amendments (2026-10-24, EmbeddingGemma) and Phase A outcome.
- The four overflow attempts (task ids no longer named in the writeup): ledger, Phase A outcome paragraph and planned-runs rows 6 (A_C1_4321, fastapi_13786), 7 (A_C0_1234, httpx_3672), 10 (A_C2_4321, rich_3535) and 12 (A_C3_4321, rich_3472). All four rows are 0 in every run: tables.md, Solve matrix. The 32,768-token window: ledger, row 6 and Facts ("Inference", `max_model_len=32768`).
- Phase B selection rule (two largest |diff|, else C0 and C3): ledger, Decision rules; prereg, Phase rules.

### 4.5 (Phase B)

- Phase B runs C1, C0 and C3 at both seeds: ledger, planned-runs rows 13-18 and Amendments (Phase B control); Phase A outcome (rule sends C0 and C3).
- 60 pre-registered tasks, `eval/subset60.txt`: prereg, Task sets (the 60 ids); ledger, Facts (2026-10-25 Phase B task mix: "re-drawn before any Phase B run and verified against the pre-registration id list"); rows 13-18 ("ids match subset60"). ("Frozen before any Phase B run" is no longer stated in 4.5; Section 3.4's "lists both task sets and is unchanged since a clarification made before the first Phase A run" implies it.) Its sha256 is in the stamped manifest (Section 3 above).
- Control 11 and 12 of 60 at seeds 1234 and 4321 (0.183, 0.200); discordant 3, p 1.000, CI [-0.033, 0.067]: noise_floor.csv row B (rate_a 0.18333, rate_b 0.2, discordant 3, mcnemar_p 1.0, ci -0.03333 to 0.06667; its wins 2 and losses 1 are not quoted, since a seed-vs-seed win has no direction); per_condition.csv rows (B, C1, 1234/4321: resolved 11, 12); ledger, "Phase B noise floor".
- Rule p < 0.05 and |wins - losses| > 3; smallest pattern 6-0, |diff| 0.050: ledger, "Phase B noise floor" (threshold) and Facts ("Minimum detectable effect"); tables.md, Headline ("Phase B minimum detectable effect: 6 discordant pairs in one direction, pooled |diff| >= 0.050").
- C3 vs C1 pooled: n 120, 60 clusters, 0.183 vs 0.192, diff -0.008, wins 2, losses 3, p 1.000, CI [-0.033, 0.017], not beyond noise: comparisons.csv row (B, C3, pooled): treatment_rate 0.18333, control_rate 0.19167, diff -0.00833, ci -0.03333 to 0.01667, beyond_noise False; ledger, "Phase B C3 vs C1, pooled seeds".
- Seeds 0-2 at 1234 and 2-1 at 4321: comparisons.csv rows (B, C3, seed 1234) and (B, C3, seed 4321); ledger, same line ("Per seed").
- fastapi_14851 loss at seed 1234 is a context-window overrun counted as unresolved under the rule stated in Section 3.4 (ledger runs 6 and 7; the pre-registration itself does not mention overflows, only the no-discard rule); both B_C1 runs resolved it: ledger, row 15 (B_C3_1234, ContextWindowExceededError on fastapi_14851) and "Phase B C3 vs C1, pooled seeds"; solve_matrix.csv row (B, fastapi_14851): B_C1_1234 1, B_C1_4321 1, B_C3_1234 0.
- Other discordant pairs on requests_6644 and rich_3676 (Phase B floor) and fastapi_9425 (Phase A floor), with the in-text pointer to Fig. 3: ledger, "Phase B C3 vs C1, pooled seeds" (wins fastapi_9425, requests_6644 at 4321; losses requests_6644 at 1234, rich_3676 at 4321) and "Phase B noise floor" (fastapi_9555, rich_3676, requests_6644); Phase A noise floor line (fastapi_9425).
- Phase B agrees with Phase A for C3 (no condition beyond noise in either): comparisons.csv `beyond_noise` False in rows (A, C3, pooled) and (B, C3, pooled).
- C0 vs C1 pooled ("the same way": n 120, 60 clusters): 0.183 vs 0.192, diff -0.008, wins 2, losses 3, p 1.000, CI [-0.042, 0.025], not beyond noise: comparisons.csv row (B, C0, pooled): n 120, n_clusters 60, treatment_rate 0.18333, control_rate 0.19167, diff -0.00833, wins 2, losses 3, ties 115, mcnemar_p 1.0, ci -0.04167 to 0.025, beyond_noise False, incomplete False; tables.md, Comparisons against C1 (B, C0, pooled); ledger, "Phase B C0 vs C1, pooled seeds". Agreeing with Phase A: comparisons.csv `beyond_noise` False in rows (A, C0, pooled) and (B, C0, pooled); ledger, same line ("Agrees with Phase A").
- Seeds 1-0 at 1234 and 1-3 at 4321: comparisons.csv rows (B, C0, seed 1234: wins 1, losses 0) and (B, C0, seed 4321: wins 1, losses 3); ledger, same line ("Per seed").
- C0's discordant tasks are C3's plus rich_3718 (as task sets: C0 fastapi_9425, requests_6644, fastapi_14851, rich_3676, rich_3718; C3 fastapi_9425, requests_6644, fastapi_14851, rich_3676): ledger, "Phase B C0 vs C1, pooled seeds" (wins fastapi_9425 at 1234, requests_6644 at 4321; losses fastapi_14851, rich_3676, rich_3718 at 4321) and "Phase B C3 vs C1, pooled seeds". rich_3718 resolved by every other Phase B run: solve_matrix.csv row (B, rich_3718): 1 in B_C0_1234, B_C1_1234, B_C1_4321, B_C3_1234, B_C3_4321, 0 in B_C0_4321; ledger, same line ("resolved in the other five Phase B runs"). None overran: the Phase B overrun tasks are, by ledger rows 13-14, fastapi_14361, rich_3061, rich_3934 (B_C1_1234) and fastapi_13786 (B_C1_4321), and by rows 17-18, fastapi_15745, rich_3944 (B_C0_1234) and requests_7427 (B_C0_4321); none is among C0's five discordant tasks. The run CSVs carry no overrun column, so the ledger rows are the source.
- Over all six Phase B runs, 46 never resolved, 8 always, 6 changed outcome: tables.md, Solve matrix (phase B: runs 6, never_solved 46, always_solved 8; 60 - 46 - 8 = 6); solve_matrix.csv (B rows: fastapi_14851, fastapi_9425, fastapi_9555, requests_6644, rich_3676, rich_3718 are the only rows with both 0 and 1); ledger, "Phase B solve matrix over all six runs".
- Overruns 10/360 Phase B vs 4/192 Phase A, Fisher exact p 0.78, pooled 14/552 = 2.5%: ledger, "2026-11-07 - Context-window overruns, all six Phase B runs" (B_C1 4/120, B_C3 3/120, B_C0 3/120; 2.8% vs 2.1%; the phases are not distinguishable, so one pooled rate is quoted); B_C0_4321 adds requests_7427, ledger row 18. Recomputed for this pass: scipy `fisher_exact([[10, 350], [4, 188]])` p 0.7796; 14/552 = 0.0254. (Superseded: 9/300, p 0.77, 13/492 = 2.6% over five runs, ledger 2026-11-02 line.)
- Fig. 3 (`paper/figures/fig_solve_matrix_B.png`), written by `notebooks/analysis.py --strict` (the Phase B design is complete): six columns, C0, C1, C3 at seeds 1234 and 4321 (figure subtitle "Phase B: 60 tasks, 6 runs (8 always resolved, 46 never)"); solve_matrix.csv B_C0_1234 to B_C3_4321 columns.

### 4.6 (mechanism)

- Graph-tool calls per task run 6.396 (C1), 4.979 (C2), 5.208 (C3): tables.md, Per condition (pooled rows).
- One graph-tool error response, C2 seed 1234: tables.md, Per condition (A, C2, 1234, `graph_tool_errors_total` 1; all other rows 0); ledger, row 9 (requests_7502).
- C3 search column sums (denominators below): A_C3_1234 110 calls, 70 resolved, 40 lexical, 7 substring; A_C3_4321 96, 58, 38, 7: runs. (Per-run search-call counts are no longer quoted in the writeup.)
- C3 embedding-path share 70/110 = 0.636 and 58/96 = 0.604; substring share of resolved 7/70 = 0.100 and 7/58 = 0.121: runs, A_C3_1234.csv and A_C3_4321.csv column sums (classified against the rebuilt graphs: README).
- Logged tool calls 25.708 (C1), 26.146 (C0), 26.125 (C3), 30.188 (C2): tables.md, Per condition (pooled rows). The quoted range 25.708 to 26.146 covers C0, C1 and C3.
- fastapi_14077 in A_C2_1234: 199 logged tool calls and 1,837,528 prompt tokens: runs, A_C2_1234.csv row fastapi_14077 (the next highest `tool_calls` in that run is 35). This row drives the C2 seed-1234 means (34.917 tool calls, 278,215.375 prompt tokens; tables.md, Per condition), the pooled C2 prompt-token mean 236,850 ("including that task") and the Fig. 4 "All tool calls" dot (next-highest prompt tokens in that run: 311,972).
- Logged `tool_calls` also counts exempt calls, `code_analyzer` delegations and calls rejected after the budget ran out, so it can exceed 28: ledger, Facts ("Agent tools").
- Gold file read at least 0.938 in every condition: tables.md, Per condition (pooled: C0 0.958, C1 0.938, C2 0.938, C3 0.938).
- Prompt tokens per task run 172,505.458 (C0), 200,759.917 (C1), 236,850.396 (C2), 203,847.708 (C3), rounded to whole tokens: tables.md, Per condition (pooled rows).
- Gold file edited 0.563 (C1) against 0.708 (C2, C3) and 0.750 (C0): tables.md, Per condition (pooled rows).
- Gold file = a file the reference patch edits: `graphswap/traces.py` `gold_files`.
- Fig. 4 (`paper/figures/fig_tool_profile.png`) shows Phase A (C0 to C3) and Phase B (C0, C1, C3): per_condition.csv phases; figure subtitle "Phase A, B". Both Phase B C0 seeds are shown: per_condition.csv rows (B, C0, 1234) and (B, C0, 4321).

## 5

- C1 5.217 graph-tool calls among 25.592 logged calls per task run; C3 4.175 among 25.517 (Phase B, pooled seeds): per_condition.csv rows (B, C1, pooled) and (B, C3, pooled), columns `mean_graph_tool_calls` and `mean_tool_calls`. "About one in five or six": 5.217/25.592 = 0.204 and 4.175/25.517 = 0.164.
- C0 read a gold file in 0.958 of Phase A task runs: per_condition.csv row (A, C0, pooled), `gold_file_read_rate` 0.95833.
- Phase A: 7 to 12 session timeouts per run against 0 to 2 tool-call-budget stops: ledger, planned-runs rows 5 to 12 (timeouts 9, 10, 10, 9, 8, 8, 12, 7; call-budget stops 1, 1, 0, 2, 2, 1, 0, 1).
- Phase B C1 and C3 runs: 23 to 28 timeouts against 1 or 2 budget stops; 20 to 30 tasks per run used all 28 calls: ledger, planned-runs rows 13-16 (timeouts 26, 27, 28, 23; budget stops 1, 2, 2, 1; "used all 28 calls" 30, 26, 20, 24). Rows 17-18 (B_C0_1234: 24, 3, 24; B_C0_4321: 21, 6, 21) are left out because the sentence names C1 and C3 only.
- 18 of 24 Phase A tasks never resolved: tables.md, Solve matrix (phase A never_solved 18).
- 47 of 60 Phase B tasks resolved by neither control seed: ledger, "Phase B noise floor" ("10 tasks resolved under both seeds; 47 under neither"); solve_matrix.csv (rows with B_C1_1234 = B_C1_4321 = 0: 47).

## 6

- Kit functions (build graphs, embed, write per-file and bundle SHA-256 hashes, produce audits, per-task CSVs, comparisons and task sets; the subcommands `build-graphs`, `embed-graphs`, `bundle`, `audit`, `summarize`, `compare`, `draw-subset`, no longer named in the writeup): `graphswap/cli.py`. Apache-2.0: `LICENSE`; `pyproject.toml` (`license`).
- Any sentence-transformers model in `embed-graphs`; node-link graph with `id`, `name`, `text` loads unchanged: `graphswap/cli.py` (`embed-graphs --model`), `graphswap/embedder.py` (SentenceTransformer; `embed_graph` reads only the nodes' `id` and `text`); `graphswap/traces.py` `graph_node_names` (the harness loader reads the node-link file; a node's name is its `name`, defaulting to the id).
- Bundle shas: ledger, Bundles; prereg, Bundle provenance.
- `rebuilt_full` in a public Kaggle dataset (https://www.kaggle.com/datasets/ibrahimkhalil09/gemma4-graphswap-bundles); repository (https://github.com/m-ibrahim-khalil/gemma4-graphswap, tag `v1.0`); notebook on Kaggle (https://www.kaggle.com/code/ibrahimkhalil09/gemma4-graphswap-analysis): publication URLs, filled 2026-11-08.
- `rebuilt_full` holds the rebuilt graphs, their embeddings and a `MANIFEST.json` of file hashes; check it against the table (its `bundle_sha` column): `graphswap/bundle.py` `assemble` ("out/<name>/{graphs,embeddings}/ plus MANIFEST.json"); the bundle as built locally (`bundles/out/rebuilt_full/{graphs,embeddings,MANIFEST.json}`, gitignored).
- Notebook ships as `kaggle/analysis_notebook.ipynb` and runs on Kaggle: the file itself; `notebooks/build_kaggle_notebook.py` (the notebook's own Inputs cell explains attaching the results dataset, so the writeup does not).
- `emb_released_nodes` not published (derived from the competition's released node texts) and regenerated by the kit from an entrant's copy: ledger, Facts ("Release scope", no released graphs or embeddings); README, "Deliberately absent" (no released graphs or embeddings); README step 4 (`graphswap bundle` reads `bundles/emb_released_nodes_raw`).
- `bundle_sha` hashes raw float32 bytes, so it can match only on the device class that built the bundle (Apple MPS: README step 4 and `graphswap/cli.py`, `embed-graphs --device` defaults to mps and the device is recorded in each embeddings MANIFEST; ledger, Facts ("Bundle hashes": "the MPS-built `emb_rebuilt` bundle") and Amendments (2026-10-24, EmbeddingGemma vectors built on mps); also `device: mps` in the gitignored local bundle MANIFESTs): ledger, Facts ("Bundle hashes": vectors regenerated on CUDA/CPU vs MPS will not match byte for byte); `graphswap/bundle.py` (`bundle_sha` over per-file hashes). Device tolerance rests on the rebuilt nodes only ("Off MPS we re-embedded only rebuilt nodes (same encoder and code path)"), re-embedded with the same encoder (Qwen3-Embedding-0.6B, --max-tokens 1024) and code path (`embed-graphs`): httpx at 4acf5c2 (the smallest rebuilt graph), 885 nodes, CPU bfloat16 vs the MPS-built bundle: per-node cosine min 0.999728, median 0.999851, none byte-identical: ledger, Facts ("Bundle hashes", measured 2026-11-01). No such measurement exists for `emb_released_nodes` itself, and the writeup claims none. (4acf5c2 is the upstream httpx base commit, not a commit of this repository.)
- Release scope (kit, per-task CSVs, summary tables, figures, notebook, `harness-tooling/`; excluded: swegemma source, task text, released graphs and embeddings, raw traces): ledger, Facts ("Release scope"); `harness-tooling/README.md`.
- Single squashed commit because the development history contained test fixtures derived from competition traces: visible in the public repository itself (one commit); the reason is the authors' release decision of 2026-10-27, recorded outside this repository. Per-task results, the pre-registration and its timestamp included: the files themselves (`results/runs/`, `docs/preregistration.md`, `docs/preregistration.sha256`, `docs/preregistration.sha256.ots`).

## 7

- 127 rebuilt graphs [1]; Hit@5 [2]; hit@5 [4]: references.md entries 1, 2, 4 (external).
- 2.00 to 2.66 points [5]; 92.7% [7]: references.md entries 5, 7 (external).
