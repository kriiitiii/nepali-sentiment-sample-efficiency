#!/usr/bin/env bash
# Learning-rate selection followed by the full learning curve for one transformer.
# Usage: bash scripts/run_transformer_pipeline.sh xlmr|nepberta
set -euo pipefail
MODEL="$1"
PY="${PYTHON:-python}"
LOG="outputs/metrics/log_${MODEL}.txt"

for lr in 1e-5 2e-5 3e-5; do
  "$PY" -X utf8 scripts/run_experiments.py --model "$MODEL" --sizes full --max-repeats 1 --lr "$lr" 2>&1 \
    | grep -E "^${MODEL}|Error|Traceback" >> "$LOG" || true
done

BEST=$("$PY" - "$MODEL" <<'EOF'
import json, sys
m = sys.argv[1]
runs = [json.loads(l) for l in open("outputs/metrics/runs.jsonl", encoding="utf-8")]
c = [r for r in runs if r["model"] == m and r["size"] == "full" and r["repeat"] == 0
     and r["split_mode"] == "grouped" and r["lr"] in (1e-5, 2e-5, 3e-5)]
print(max(c, key=lambda r: r["val_macro_f1"])["lr"])
EOF
)
echo "selected lr for $MODEL: $BEST" >> "$LOG"
"$PY" -X utf8 scripts/run_experiments.py --model "$MODEL" --lr "$BEST" 2>&1 \
  | grep -E "^${MODEL}|Error|Traceback" >> "$LOG" || true
echo "PIPELINE DONE $MODEL" >> "$LOG"
