#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo "Usage: bash run_variant.sh STARTER OUTPUT_DIR CHECKPOINT SPLIT RUN_NAME" >&2
  exit 2
fi
project="$(cd "$(dirname "$0")/.." && pwd)"
starter="$(cd "$1" && pwd)"
mkdir -p "$2"
results="$(cd "$2" && pwd)"
checkpoint="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
split="$4"
name="$5"
python_bin="${MP3_PYTHON:-python3}"
export PYTHONPATH="$project/submission:$starter"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MP3_MODEL_TRACE="$results/$name-calls.jsonl"
: > "$MP3_MODEL_TRACE"
cd "$starter"
"$python_bin" "$project/experiments/measure.py" "$results/$name-predict-resources.json" \
  "$python_bin" -m mp3_sql.predict --split "$split" --workers 1 --method "$project/submission/submission.py" \
  --checkpoint "$checkpoint" --output "$results/$name.json"
"$python_bin" "$project/experiments/measure.py" "$results/$name-score-resources.json" \
  "$python_bin" -m mp3_sql.score --split "$split" --workers 4 --predictions "$results/$name.json" \
  --output "$results/$name-score.json"
