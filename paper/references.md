# References and verification log

Every entry below was opened with WebFetch (or the Crossref / Europe PMC record for paywalled papers) on 2026-10-24. "Verified" states what the fetched page itself says; "Used for" names the claim in `paper/writeup.md` that relies on it. Numbering matches the writeup.

1. natemeton-hub. SWE-GraphBench: auditing and rebuilding code graphs for local software-engineering agents. GitHub. https://github.com/natemeton-hub/swe-graphbench (accessed 2026-10-24)
   Verified: built on the Gemma 4 Developer Agent public release; audit of released graphs and embeddings; stdlib builder; corrected graphs for all 127 commits; LLM-free localization baselines scored with Recall@k, Hit@k and MRR with bootstrap CIs; no end-to-end agent runs or resolution rates mentioned; Apache-2.0. The README gives no numeric audit findings.
   Used for: Sections 1 and 7 (audit, rebuild, offline localization metrics).


2. danielmantiglia. codegraph-loc. GitHub. https://github.com/danielmantiglia/codegraph-loc (accessed 2026-10-24)
   Verified: written for the Gemma 4 Developer Agent Paper Track; async-aware graph generator; "the released graphs contain no async definition at all"; released 256-d embeddings "effectively ~4-dimensional (4 principal components explain 90% of the variance)", random-pair cosine 0.79 raw; Experiment 5 is a pre-registered Gemma 4 graph-navigating localization agent scored by Hit@5; no patch or test-based resolution runs.
   Used for: Sections 1, 2 and 7 (no async definitions; collapsed vectors with a different effective-dimension definition; localization agent scored by Hit@5).


3. Benadict-Git. codegraph-loc. GitHub. https://github.com/Benadict-Git/codegraph-loc (accessed 2026-10-24)
   Verified: tree-sitter code graphs and function-level gold for graph-guided bug localization with small local LLM agents; SWE-bench Lite (300 instances); JSON tool-calling localization agent; audit tooling for the competition graphs and embeddings, but no audit findings on the page; built for the Gemma 4 Developer Agent Paper Track.
   Used for: Sections 1 and 7.


4. aghasalim. Does the Code Graph Find the Bug? Measuring Fault Localisation with the Released Gemma 4 Graphs and Embeddings (repository gemma4-code-graph-localization). GitHub. https://github.com/aghasalim/gemma4-code-graph-localization and https://raw.githubusercontent.com/aghasalim/gemma4-code-graph-localization/main/PAPER.md (accessed 2026-10-24)
   Verified: offline file-level hit@5 on 129 training tasks (embed 0.29, graph 0.30, grep 0.41, bm25 0.48); "The graph has no async functions at all"; "21.9% of node vectors are exact copies of another node's vector"; every edge is a `calls` edge; PAPER.md Section 5 reports end-to-end runs with the competition model and harness on 30 tasks, where "one agent run twice with the same settings solved 15 and then 11 tasks, and 8 of the 30 changed outcome"; Section 5 does not say whether graph tools were enabled and does not compare with and without graph tools or released and rebuilt graphs; no statistical test for the end-to-end runs.
   Used for: Sections 1, 2 and 7 (calls-only edges, duplicate vectors with a different definition, no async, 8 of 30 outcome changes, grep and BM25 beat graph and embedding tools).


5. Ouyang, S., Yu, W., Ma, K., et al. RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph. ICLR 2025. https://arxiv.org/abs/2410.14684 and https://arxiv.org/html/2410.14684v2 (accessed 2026-10-24)
   Verified: plug-in repository graph for four SWE-bench methods; Table 2 SWE-bench Lite resolve rates, e.g. Agentless (GPT-4o) 27.33 to 29.67 (+2.34), RAG (GPT-4) +2.66, AutoCodeRover (GPT-4) +2.33, SWE-agent (GPT-4o) +2.00; no runs, seeds, variance or significance tests reported.
   Used for: Section 7 (gains of 2.00 to 2.66 points without reported variance).


6. Liu, X., Lan, B., Hu, Z., et al. CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases. arXiv:2408.03910. https://arxiv.org/abs/2408.03910 (accessed 2026-10-24)
   Verified: LLM agents with graph database interfaces extracted from code repositories; evaluated on CrossCodeEval, SWE-bench and EvoCodeBench.
   Used for: Section 7.


7. Chen, Z., Tang, X., Deng, G., et al. LocAgent: Graph-Guided LLM Agents for Code Localization. arXiv:2503.09089. https://arxiv.org/abs/2503.09089 (accessed 2026-10-24)
   Verified: codebases parsed into directed heterogeneous graphs; "up to 92.7% accuracy on file-level localization" with a fine-tuned Qwen-2.5-Coder-Instruct-32B; downstream issue resolution improved 12% at Pass@10.
   Used for: Section 7 (92.7% file-level localization).


8. Jimenez, C. E., Yang, J., Wettig, A., et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024. https://arxiv.org/abs/2310.06770 (accessed 2026-10-24)
   Verified: 2,294 problems from real GitHub issues in 12 Python repositories; the model edits the codebase to resolve the issue.
   Used for: Section 3.4 (task format).


9. princeton-nlp. SWE-bench Verified (dataset card). Hugging Face. https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified (accessed 2026-10-24)
    Verified: 500-sample human-validated subset of the SWE-bench test set, evaluated with unit tests.
    Used for: Section 3.4 (task format).


10. Yang, J., Jimenez, C. E., Wettig, A., et al. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. arXiv:2405.15793. https://arxiv.org/abs/2405.15793 (accessed 2026-10-24)
    Verified: LM agent operating a computer through a custom agent-computer interface for editing, navigation and test execution.
    Used for: Section 3.3 (tool-using agent loop).


11. Xia, C. S., Deng, Y., Dunn, S., Zhang, L. Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489. https://arxiv.org/abs/2407.01489 (accessed 2026-10-24)
    Verified: fixed three-stage pipeline (localization, repair, patch validation) without the LLM deciding its own next steps.
    Used for: Section 3.3 (contrast with fixed pipelines).


12. McNemar, Q. Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages. Psychometrika 12(2):153-157, 1947. https://doi.org/10.1007/BF02295996 (Crossref record https://api.crossref.org/works/10.1007/BF02295996 accessed 2026-10-24; the Springer page redirects to a login)
    Verified: bibliographic record (title, author, journal, volume, issue, pages, year).
    Used for: Section 3.4 (exact paired test).


13. Fagerland, M. W., Lydersen, S., Laake, P. The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional. BMC Medical Research Methodology 13:91, 2013. https://doi.org/10.1186/1471-2288-13-91 (Europe PMC record accessed 2026-10-24; the publisher page redirects to a login)
    Verified: title and authors; results state "The exact conditional test and the asymptotic test with continuity correction did not perform well".
    Used for: Section 3.4 (exact test is conservative).


14. Field, C. A., Welsh, A. H. Bootstrapping Clustered Data. Journal of the Royal Statistical Society Series B 69(3):369-390, 2007. https://doi.org/10.1111/j.1467-9868.2007.00593.x (Crossref record accessed 2026-10-24)
    Verified: bibliographic record only.
    Used for: Section 3.4 (task-clustered bootstrap).


15. Miller, E. Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. arXiv:2411.00640, 2024. https://arxiv.org/abs/2411.00640 and https://arxiv.org/html/2411.00640v1 (accessed 2026-10-24)
    Verified: recommends clustered standard errors "When questions are drawn in related groups" and the paired standard error "wherever practicable" (Section 4.2).
    Used for: Section 3.4 (clustering by task, paired analysis).


16. Gemma Team. Gemma 4 Technical Report. arXiv:2607.02770, 2026. https://arxiv.org/abs/2607.02770 (accessed 2026-10-24)
    Verified: Gemma 4 family from 2.3B to 31B parameters, dense and MoE. The model card https://huggingface.co/google/gemma-4-31B-it (accessed 2026-10-24) links this report.
    Used for: Section 3.3 (model).


17. Qwen. Qwen3-Embedding-0.6B (model card). Hugging Face. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B (accessed 2026-10-24)
    Verified: 0.6B parameters, embedding dimension up to 1024, 32k context, Apache-2.0; paper arXiv:2506.05176 (not fetched).
    Used for: Section 3.2.


18. Google. EmbeddingGemma-300m (model card). Hugging Face. https://huggingface.co/google/embeddinggemma-300m (accessed 2026-10-24)
    Verified: about 300M parameters, 768-d default output with Matryoshka truncation, document prompt `title: none | text:`, 2048-token context, gated; paper arXiv:2509.20354 (not fetched).
    Used for: Section 3.2 (amendment).


19. Google, Kaggle. Google: The Gemma 4 Developer Agent Paper Track (competition page). https://www.kaggle.com/competitions/gemma-4-developer-agent-paper (accessed 2026-10-24)
    Verified: page title only (Kaggle renders client-side). The harness itself (swegemma 0.2.11) is distributed to entrants and has no public page; harness facts in Section 2 come from the repo design spec and the ledger's recorded facts, checked against the local harness source.
    Used for: Section 2 (harness attribution).

## Checked and not cited

- Aefinity-AI, gemma4-agent-reliability (https://github.com/Aefinity-AI/gemma4-agent-reliability, accessed 2026-10-24): verified as a pre-registered run-to-run noise study with no results yet; cited in draft v0 and dropped in review round 1 to save words.
- OpenAI, "Introducing SWE-bench Verified" (https://openai.com/index/introducing-swe-bench-verified/): HTTP 403; replaced by the Hugging Face dataset card [10].
- Kaggle notebook "SWE-GraphBench: reproduce the audit and benchmark" (https://www.kaggle.com/code/nateshingleton/swe-graphbench-reproduce-the-audit-and-benchmark): page title only; no Kaggle writeup URL found for SWE-GraphBench. The GitHub README [1] is cited instead.
- swegemma on PyPI (https://pypi.org/project/swegemma/): no package (PyPI JSON API returns 404). GitHub search finds only an unofficial `HashanEranga/swegemma`, not cited.
- Qwen3 Embedding paper (arXiv:2506.05176) and EmbeddingGemma paper (arXiv:2509.20354): linked from the model cards but not fetched; the cards are cited instead.
