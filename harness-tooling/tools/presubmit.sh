#!/usr/bin/env bash
# Validate submission/ against the harness contract before spending a submission.
# Checks what the scorer checks, locally and for free.
set -uo pipefail

DIR="${1:-submission}"
fail=0
note() { printf '  %-6s %s\n' "$1" "$2"; }
bad() { note "FAIL" "$2"; fail=1; }
ok()  { note "ok" "$2"; }

echo "validating ${DIR}/"

# 1. Exactly one root config.
roots=$(find "$DIR" -maxdepth 1 -type f \
  \( -name agent.yaml -o -name agent.yml -o -name root_agent.yaml -o -name root_agent.yml \) | wc -l | tr -d ' ')
case "$roots" in
  1) ok . "exactly one root config" ;;
  0) bad . "no root config (need agent.yaml | agent.yml | root_agent.yaml | root_agent.yml)" ;;
  *) bad . "$roots root configs found -- MultipleRootConfigsError" ;;
esac

# 2. Allowed extensions only.
disallowed=$(find "$DIR" -type f ! -name '*.yaml' ! -name '*.yml' ! -name '*.md' \
  ! -name '*.txt' ! -name '*.py' ! -name '*.json' ! -name '*.safetensors')
if [ -z "$disallowed" ]; then
  ok . "all file extensions allowed"
else
  bad . "disallowed files (allowed: yaml yml md txt py json safetensors):"
  echo "$disallowed" | sed 's/^/         /'
fi

# 3. No symlinks (PathTraversalError).
links=$(find "$DIR" -type l)
if [ -z "$links" ]; then ok . "no symlinks"; else bad . "symlinks present:"; echo "$links" | sed 's/^/         /'; fi

# 4. Total unpacked size < 3 GiB.
bytes=$(find "$DIR" -type f -exec cat {} + 2>/dev/null | wc -c | tr -d ' ')
limit=3221225472
human=$(awk -v b="$bytes" 'BEGIN{printf "%.1f MiB", b/1048576}')
if [ "$bytes" -lt "$limit" ]; then ok . "unpacked size ${human} (< 3 GiB)"
else bad . "unpacked size ${human} exceeds the 3 GiB cap"; fi

# 5. Single base model rule.
models=$(grep -rhoE '^[[:space:]]*model:[[:space:]]*\S+' "$DIR" --include='*.yaml' --include='*.yml' \
  | sed -E 's/.*model:[[:space:]]*//; s#^(openai|google|hosted_vllm|custom)/##' | sort -u)
count=$(printf '%s\n' "$models" | grep -c . || true)
if [ "$count" -eq 1 ]; then
  ok . "single base model: $models"
  case "$models" in
    gemma-4-31b-it-qat-w4a16-ct|gemma-4-12b-it-qat-w4a16-ct|gemma-4-e4b-it-qat-w4a16-ct|gemma-4-e2b-it-qat-w4a16-ct)
      ok . "model is competition-allowed" ;;
    *) bad . "model '$models' is NOT in ALLOWED_MODEL_NAMES -- scoring will reject it" ;;
  esac
else
  bad . "$count distinct base models declared (must be exactly 1):"
  printf '%s\n' "$models" | sed 's/^/         /'
fi

# 6. !include targets resolve -- RELATIVE TO THE INCLUDING FILE'S DIRECTORY, the
#    way adk_submission's yaml_loader does it, and must stay inside the root.
#    (Resolving from the root instead is how a broken sub_agents/ include once
#    passed this check and failed every task on Kaggle.)
include_report=$(python3 - "$DIR" <<'PY'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1]).resolve()
bad = 0
for yml in sorted(list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))):
    for n, line in enumerate(yml.read_text().splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        m = re.search(r"!include\s+([^\s#]+)", line)
        if not m:
            continue
        rel = m.group(1).strip("'\"")
        where = f"{yml.relative_to(root)}:{n}"
        target = (yml.parent / rel).resolve()
        if rel.startswith("/") or not target.is_relative_to(root):
            print(f"FAIL|!include escapes the submission root in {where}: {rel}"); bad = 1
        elif not target.is_file():
            print(f"FAIL|broken !include in {where}: {rel} -> {target.relative_to(root)} does not exist"); bad = 1
print(f"RC|{bad}")
PY
)
while IFS='|' read -r kind msg; do
  [ "$kind" = "FAIL" ] && note "FAIL" "$msg"
done <<< "$include_report"
if echo "$include_report" | grep -q '^RC|0'; then ok . "all !include targets resolve (relative to the including file)"; else fail=1; fi

echo
if [ "$fail" -eq 0 ]; then echo "PASS -- submission looks structurally valid"; else echo "FAILED -- fix the above before submitting"; fi
exit "$fail"
