# Experiment loop for the Gemma 4 Developer Agent competition.
#
# This Mac edits and orchestrates; Kaggle L4x4 notebooks evaluate (vLLM cannot
# run on Apple Silicon). The binding constraints are 1 submission/day and GPU
# quota burned at 2x on L4x4, so every target here either protects a submission
# or accounts for GPU-hours.

COMP        := gemma-4-developer-agent
KAGGLE      := $(shell command -v kaggle 2>/dev/null || echo $(HOME)/Library/Python/3.11/bin/kaggle)
EVAL        := python3 tools/kaggle_eval.py

SUBMISSION  := submission
ZIP         := submission.zip
TASKS       := data/tasks.jsonl
SUBSET      ?= eval/subset.txt
CONCURRENCY ?= 4
CONDITION   ?= C1
SEED        ?= 1234
RUN         ?=

.DEFAULT_GOAL := help
.PHONY: help check zip subset push dry-run run smoke status wait pull eval flips submit clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  Loop:  edit submission/ -> make check -> git commit -> make eval RUN=<name>"
	@echo "         -> make flips A=results/<base> B=results/<name> -> make submit MSG=..."

check: ## Validate submission/ against the harness contract (free, local)
	@bash tools/presubmit.sh $(SUBMISSION)

zip: check ## Build submission.zip (contents at archive root)
	@rm -f $(ZIP)
	@cd $(SUBMISSION) && zip -qr ../$(ZIP) . -x '.*' '*/.*' '*__pycache__*'
	@echo "built $(ZIP) ($$(du -h $(ZIP) | cut -f1))"

subset: ## Fetch tasks.jsonl and create the FIXED eval subsets if missing
	@test -f $(TASKS) || { mkdir -p data && $(KAGGLE) -W competitions download $(COMP) -f tasks.jsonl -p data -q; \
	  [ -f $(TASKS).zip ] && unzip -oq $(TASKS).zip -d data && rm $(TASKS).zip; true; }
	@test -f eval/subset.txt || python3 tools/make_subset.py $(TASKS) --n 24 --seed 0 > eval/subset.txt
	@test -f eval/smoke.txt || python3 tools/make_subset.py $(TASKS) --smoke > eval/smoke.txt
	@echo "subset: $$(grep -vc '^#' eval/subset.txt) tasks, smoke: $$(grep -vc '^#' eval/smoke.txt) tasks (fixed -- do not regenerate)"

push: check ## Upload submission/ as a private Kaggle dataset (skipped if unchanged)
	@$(EVAL) push

dry-run: check ## Render the eval kernel locally without uploading: make dry-run RUN=x [CONDITION=C1] [SEED=1234]
	@test -n "$(RUN)" || { echo "usage: make dry-run RUN=<name>"; exit 1; }
	@$(EVAL) run --run $(RUN) --subset $(SUBSET) --concurrency $(CONCURRENCY) --condition $(CONDITION) --seed $(SEED) --dry-run

run: check ## Start an eval on Kaggle: make run RUN=<name> [SUBSET=..] [CONCURRENCY=4] [CONDITION=C1] [SEED=1234]
	@test -n "$(RUN)" || { echo "usage: make run RUN=<name>   (name it after the experiment, e.g. baseline, noise_b)"; exit 1; }
	@$(EVAL) run --run $(RUN) --subset $(SUBSET) --concurrency $(CONCURRENCY) --condition $(CONDITION) --seed $(SEED)

smoke: check ## 2-task pipeline check on Kaggle (<= 2 GPU-h quota): make smoke [RUN=smoke_C1] [CONDITION=C1] [SEED=1234]
	@$(EVAL) run --run $(if $(RUN),$(RUN),smoke_$$(date +%Y%m%d_%H%M%S)) --subset eval/smoke.txt --concurrency 2 --condition $(CONDITION) --seed $(SEED)

status: ## Submissions left, GPU quota, eval kernel state, unpulled run
	@echo "== submissions =="; $(KAGGLE) -W competitions submission-limits $(COMP)
	@echo; echo "== accelerator quota (L4x4 burns it at 2x) =="; $(KAGGLE) -W quota
	@echo; echo "== eval kernel =="; $(EVAL) status

wait: ## Block until the eval kernel finishes
	@$(EVAL) wait

pull: ## Download + verify results: make pull [RUN=<name>] (defaults to the unpulled run)
	@$(EVAL) pull $(if $(RUN),--run $(RUN))

eval: ## run + wait + pull in one go (blocks for the whole run): make eval RUN=<name>
	@$(MAKE) --no-print-directory run RUN=$(RUN) SUBSET=$(SUBSET) CONCURRENCY=$(CONCURRENCY) CONDITION=$(CONDITION) SEED=$(SEED)
	@$(MAKE) --no-print-directory wait
	@$(MAKE) --no-print-directory pull RUN=$(RUN)

flips: ## Per-task diff of two runs + control check: make flips A=results/x B=results/y
	@test -n "$(A)" -a -n "$(B)" || { echo "usage: make flips A=results/x B=results/y"; exit 1; }
	@python3 tools/flips.py $(A) $(B)

submit: zip ## Guarded submit: make submit MSG="what changed and why"
	@test -n "$(MSG)" || { echo "refusing: MSG is required. Say what changed."; exit 1; }
	@remaining=$$($(KAGGLE) -W competitions submission-limits $(COMP) 2>/dev/null | grep -i 'Remaining today' | grep -oE '[0-9]+'); \
	if [ "$${remaining:-0}" -lt 1 ]; then echo "refusing: 0 submissions remaining today"; exit 1; fi; \
	echo "submissions remaining today: $$remaining"; \
	sha=$$(git rev-parse --short HEAD 2>/dev/null || echo nogit); \
	git diff --quiet -- $(SUBMISSION) 2>/dev/null || sha="$$sha-dirty"; \
	msg="$$sha | $(MSG)"; \
	echo "message: $$msg"; \
	printf 'spend one of your remaining lifetime submissions? [y/N] '; read ans; \
	case "$$ans" in [yY]*) $(KAGGLE) -W competitions submit $(COMP) -f $(ZIP) -m "$$msg";; *) echo aborted;; esac

clean: ## Remove build artifacts (keeps results/ and data/)
	rm -rf $(ZIP) build/dataset build/kernel
