"""Fine-tune one Qwen LoRA for checked-candidate ranking and SQL repair."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from mp3_agent.ranker import BASE_REPOSITORY, BASE_REVISION, label_ids, load_base, prompt_text
from mp3_agent.rewrite import rewrite_prompt


def read_train(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or any(not row["id"].startswith("train-") for row in rows):
        raise ValueError(f"Expected only train task supervision in {path}")
    return rows, hashlib.sha256(raw).hexdigest()


def train(args):
    if args.adapter.exists():
        raise ValueError("Use a new adapter path")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    ranking, rank_hash = read_train(args.rank_data)
    generation, gen_hash = read_train(args.rewrite_data)
    tokenizer, base = load_base(args.base_model, torch.bfloat16, "cuda")
    if args.init_adapter:
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True,
                                         local_files_only=True, use_safetensors=True)
    else:
        model = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=2 * args.rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none"
        ))
    config = model.peft_config["default"]
    config.base_model_name_or_path = BASE_REPOSITORY
    config.revision = BASE_REVISION
    trainable = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if ".lora_A." not in name and ".lora_B." not in name:
                raise RuntimeError("Expected only A/B matrices to be trainable")
            trainable.append(parameter)
    params = sum(parameter.numel() for parameter in model.parameters())
    if not trainable or params > 1_000_000_000:
        raise ValueError("Invalid trainable or total parameter count")
    labels = torch.tensor(label_ids(tokenizer), device="cuda")
    prepared = []
    for record in generation:
        prompt = record.get("prompt") or rewrite_prompt(tokenizer, record)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        target = tokenizer.encode(record["sql"].strip(), add_special_tokens=False) + [tokenizer.eos_token_id]
        rejected = None
        if record.get("rejected_sql"):
            rejected = tokenizer.encode(record["rejected_sql"].strip(), add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prefix) <= 3072 and len(target) <= 192 and (rejected is None or len(rejected) <= 192):
            prepared.append((prefix, target, rejected))
    rank_usable = [
        record for record in ranking if len(tokenizer.encode(
            prompt_text(tokenizer, record, include_example=True), add_special_tokens=False
        )) <= 2048 and any(record["correct"]) and not all(record["correct"])
    ]
    if not prepared or not rank_usable:
        raise ValueError("Both training objectives need nonempty usable data")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    torch.cuda.reset_peak_memory_stats()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if args.log.exists():
        raise ValueError("Use a new training log")

    def log(record):
        with args.log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    log({"event": "configuration", "model": BASE_REPOSITORY, "revision": BASE_REVISION,
         "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
         "total_parameters": params, "trainable_parameters": sum(param.numel() for param in trainable),
         "ranking_examples": len(rank_usable), "rewriting_examples": len(prepared),
         "rewrite_preference_pairs": sum(rejected is not None for _, _, rejected in prepared),
         "rank_data_sha256": rank_hash, "rewrite_data_sha256": gen_hash,
         "args": {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}})

    model.train()

    def continuation_loss(prefix, target):
        inputs = torch.tensor([prefix + target[:-1]], device="cuda")
        targets = torch.tensor(target, device="cuda")
        logits = model(input_ids=inputs, logits_to_keep=len(target), use_cache=False).logits[0]
        return torch.nn.functional.cross_entropy(logits.float(), targets)

    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        samples = [("rewrite", record) for record in prepared]
        samples.extend(("rank", record) for _ in range(3) for record in rank_usable)
        rng.shuffle(samples)
        step_size = args.accumulate
        losses = {"rank": [], "rewrite": []}
        for start_index in range(0, len(samples), step_size):
            optimizer.zero_grad(set_to_none=True)
            batch = samples[start_index:start_index + step_size]
            for task, record in batch:
                if task == "rewrite":
                    prefix, target, rejected = record
                    loss = continuation_loss(prefix, target)
                    if rejected is not None and args.rewrite_preference_weight:
                        rejected_loss = continuation_loss(prefix, rejected)
                        loss = loss + args.rewrite_preference_weight * torch.nn.functional.softplus(
                            loss - rejected_loss + 0.25
                        )
                else:
                    order = list(range(len(record["candidates"])))
                    rng.shuffle(order)
                    prompt = prompt_text(tokenizer, {
                        **record, "candidates": [record["candidates"][i] for i in order]
                    }, include_example=True)
                    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
                    correct = torch.tensor([record["correct"][i] for i in order], device="cuda")
                    logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1]
                    eligible = logits[labels[:len(order)]].float()
                    loss = eligible.logsumexp(dim=0) - eligible[correct].logsumexp(dim=0)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite hybrid training loss")
                (loss / len(batch)).backward()
                losses[task].append(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
        log({"event": "epoch", "epoch": epoch,
             "mean_rank_loss": sum(losses["rank"]) / len(losses["rank"]),
             "mean_rewrite_loss": sum(losses["rewrite"]) / len(losses["rewrite"]),
             "seconds": time.monotonic() - start})
        model.save_pretrained(args.adapter / f"epoch-{epoch}", safe_serialization=True)
        card = args.adapter / f"epoch-{epoch}" / "README.md"
        if card.is_file():
            card.unlink()
    log({"event": "saved", "directory": str(args.adapter),
         "gpu_peak_bytes": torch.cuda.max_memory_allocated(), "seconds": time.monotonic() - start})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--rank-data", required=True, type=Path)
    parser.add_argument("--rewrite-data", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--accumulate", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--rewrite-preference-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=7506)
    args = parser.parse_args()
    if min(args.epochs, args.rank, args.accumulate) < 1 or args.learning_rate <= 0 or args.rewrite_preference_weight < 0:
        parser.error("Training hyperparameters must be positive")
    train(args)


if __name__ == "__main__":
    main()
