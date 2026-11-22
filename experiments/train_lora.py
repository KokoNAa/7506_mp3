"""Train a Qwen3-0.6B ranking adapter from labeled train candidates."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model

from mp3_agent.ranker import BASE_REPOSITORY, BASE_REVISION, MAX_PROMPT_TOKENS, label_ids, load_base, prompt_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--adapter-out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7506)
    args = parser.parse_args()
    if min(args.rank, args.epochs, args.accumulate) < 1 or args.learning_rate <= 0:
        parser.error("Rank, epochs, accumulation and learning rate must be positive")
    if args.adapter_out.exists() and any(args.adapter_out.iterdir()):
        parser.error("Use a new adapter directory for each run")
    if not torch.cuda.is_available():
        parser.error("LoRA training requires a local CUDA GPU")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device("cuda")
    packed = args.train_data.read_bytes()
    records = [json.loads(line) for line in packed.splitlines() if line.strip()]
    if not records or any(not record["id"].startswith("train-") for record in records):
        raise ValueError("Expected labeled examples derived from the official train split")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Training task IDs must be unique")

    tokenizer, base = load_base(args.base_model, dtype, device)
    ids = torch.tensor(label_ids(tokenizer), device=device)
    kept = []
    for row in records:
        flags = row["correct"]
        if len(flags) != len(row["candidates"]) or not 2 <= len(flags) <= len(ids):
            raise ValueError("Invalid candidate label count")
        if not any(flags) or all(flags) or any(type(flag) is not bool for flag in flags):
            raise ValueError("Ranking rows need both correct and incorrect candidates")
        length = len(tokenizer.encode(prompt_text(tokenizer, row), add_special_tokens=False))
        if length <= MAX_PROMPT_TOKENS:
            kept.append(row)
    if not kept:
        raise ValueError("No training rows fit the prompt limit")

    adapter = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=2 * args.rank,
                         target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(base, adapter)
    tunable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not tunable or any(".lora_A." not in name and ".lora_B." not in name for name, _ in tunable):
        raise RuntimeError("Only LoRA A/B matrices may be trainable")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > 1_000_000_000:
        raise ValueError("Base plus adapter exceeds the course parameter limit")
    trainable = [param for _, param in tunable]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "event": "configuration", "model": BASE_REPOSITORY, "revision": BASE_REVISION,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "dtype": str(dtype),
        "trainable_parameters": sum(param.numel() for param in trainable),
        "total_parameters": parameter_count, "training_examples": len(kept),
        "too_long_examples": len(records) - len(kept), "train_data_sha256": sha256(packed).hexdigest(),
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
    }

    def log(entry):
        with args.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    log(metadata)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(kept)
        losses = []
        for offset in range(0, len(kept), args.accumulate):
            batch = kept[offset:offset + args.accumulate]
            optimizer.zero_grad(set_to_none=True)
            for record in batch:
                order = list(range(len(record["candidates"])))
                rng.shuffle(order)
                prompt = prompt_text(tokenizer, {
                    **record, "candidates": [record["candidates"][i] for i in order]
                })
                inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
                correct = torch.tensor([record["correct"][i] for i in order], device=device)
                logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1]
                eligible = logits[ids[:len(order)]].float()
                loss = eligible.logsumexp(dim=0) - eligible[correct].logsumexp(dim=0)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite ranking loss")
                (loss / len(batch)).backward()
                losses.append(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
        log({"event": "epoch", "epoch": epoch, "mean_loss": sum(losses) / len(losses),
             "wall_seconds": time.perf_counter() - start})
    model.save_pretrained(args.adapter_out, safe_serialization=True)
    # PEFT also emits a generated model card; keep the official two-file adapter export.
    card = args.adapter_out / "README.md"
    if card.is_file():
        card.unlink()
    expected = {"adapter_config.json", "adapter_model.safetensors"}
    if {path.name for path in args.adapter_out.iterdir()} != expected:
        raise RuntimeError("Unexpected files in the adapter export")
    log({"event": "saved", "adapter": str(args.adapter_out),
         "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
         "wall_seconds": time.perf_counter() - start})


if __name__ == "__main__":
    main()
