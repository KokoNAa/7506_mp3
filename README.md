# DASE7506 Mini-project 3 · SQL Repair Agent

The frozen configuration in `submission/final.json` corrected **196 of 240
test tasks (81.67%)** on Linux CPU with the unchanged course scorer. A
correct query passed all three grading database instances. CUDA validation
scored **106/120 (88.33%)**. The CPU test averaged 6.55 charged tool calls
per task and took 962.7 seconds to predict, with a peak RSS of 4.86 GiB.

The controller checks the draft, uses tool feedback to generate and score
schema-aware candidates, and skips equivalent candidate checks. When the
checked candidates fail the public example, a Qwen3-0.6B rewriter receives
short paths over declared foreign keys and possible source tables for the
requested output fields. The same local LoRA also helps rank checked
candidates.

## Run the frozen configuration on Linux CPU

Download and extract the official starter package from the course project
page. Set up Python 3.12 and install the pinned runtime dependencies.
[Course project and starter package](https://xudongwu-0.github.io/courses/dase7506/mp3/).

```bash
git clone https://github.com/KokoNAa/7506_mp3.git
cd 7506_mp3
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r submission/requirements.txt -r experiments/requirements-model.txt

export PROJECT="$PWD"
export STARTER=/absolute/path/to/MP3_SQL_student_package
export MP3_BASE_MODEL="$PROJECT/model"
export MP3_ADAPTER_DIR="$PROJECT/adapter"
export MP3_PYTHON="$PROJECT/.venv/bin/python"
export MP3_MODEL_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=""

hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --include config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors \
  --local-dir "$MP3_BASE_MODEL"
```

Download the evaluated LoRA from its pinned Hugging Face revision.

```bash
hf download Fengqinfeng/7506_mp3 \
  --revision 5fc29cb0839b73a89a2ea29abba4e97ff6782f18 \
  --include adapter_config.json adapter_model.safetensors \
  --local-dir "$MP3_ADAPTER_DIR"
```

The evaluated adapter weights have this SHA-256 digest.

```text
1a09a5691f5cbee9375b781b296a2a7096f84c4d7701a014c364152de0e6b64c
```

Run the official test after provisioning the model and adapter.

```bash
bash experiments/run_variant.sh "$STARTER" "$PROJECT/outputs" \
  "$PROJECT/submission/final.json" test test
```

The runner uses four CPU threads and one prediction worker. Model loading
is offline. It writes predictions, the official score, model call logs and
resource records under `outputs/`. For validation, use `validation validation`
in place of `test test`. The rules-only configuration, `submission/rules.json`,
can run without model files.

## Training

`experiments/` contains the train-split data preparation, LoRA training,
and ranking comparison scripts. The evaluated rank-16 adapter uses
10,092,544 LoRA parameters and 606,142,464 parameters including the base.
The training examples derive from the 480 official training tasks.
Configuration choices were made on validation. Codex assisted with
implementation, experiment setup and analysis.
