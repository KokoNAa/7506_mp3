#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash compare_validation.sh OFFICIAL_STARTER_DIR OUTPUT_DIR" >&2
  exit 2
fi
project="$(cd "$(dirname "$0")/.." && pwd)"
starter="$(cd "$1" && pwd)"
mkdir -p "$2"
results="$(cd "$2" && pwd)"
python_bin="${MP3_PYTHON:-python3}"
export PYTHONPATH="$project/submission"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$starter"
for mode in rules qwen lora; do
  export MP3_MODEL_TRACE="$results/validation-$mode-calls.jsonl"
  : > "$MP3_MODEL_TRACE"
  "$python_bin" "$project/experiments/measure.py" "$results/validation-$mode-predict-resources.json" \
    "$python_bin" -m mp3_sql.predict --split validation --workers 1 \
      --method "$project/submission/submission.py" --checkpoint "$project/submission/$mode.json" \
      --output "$results/validation-$mode.json"
  "$python_bin" "$project/experiments/measure.py" "$results/validation-$mode-score-resources.json" \
    "$python_bin" -m mp3_sql.score --split validation --workers 4 \
      --predictions "$results/validation-$mode.json" --output "$results/validation-$mode-score.json"
done
"$python_bin" "$project/experiments/compare.py" --outputs "$results"
