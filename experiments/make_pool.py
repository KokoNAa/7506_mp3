"""Label checked candidates using real rollouts and the unchanged scorer."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from mp3_agent.agent import Agent, checked_choice, visible_matches
from mp3_agent.ranker import selection_example
from mp3_agent.rewrite import rewrite_example
from mp3_sql.common import BUDGET, PROTOCOL, ROOT, digest, load_bank, read_jsonl, write_json, write_jsonl
from mp3_sql.environment import Tools, rollout
from mp3_sql.score import load_cases, score_document


def encode_predictions(bank, manifest, predictions):
    return {"protocol": PROTOCOL, "budget": BUDGET, "bank_sha256": digest(bank),
            "manifest_sha256": digest(manifest), "predictions": predictions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--guided", action="store_true")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--write-rewrites", action="store_true")
    args = parser.parse_args()
    if args.write_rewrites and args.split != "train":
        parser.error("Rewrite supervision is only supported for train")
    args.output.mkdir(parents=True, exist_ok=True)

    rows, manifest, folder = load_bank(args.split, ROOT / "data")
    cache = {}
    tools = Tools(folder, manifest, cache=cache)
    references, reference_root = load_cases(args.split, folder)
    bank_path = folder / f"{args.split}.jsonl"
    manifest_path = folder / "manifest.json"
    records = {}
    generation = {}

    def prepare(observation):
        record = selection_example(observation)
        record["id"] = observation["task"]["id"]
        records[record["id"]] = record
        if args.write_rewrites:
            generation[record["id"]] = rewrite_example(observation, omit_last=bool(len(record["candidates"]) > 1))
        return checked_choice(observation)

    def run(selector, selected_rows):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            return list(pool.map(lambda row: rollout(row, tools, Agent(
                selector, guided=args.guided, extended=args.extended, dedup=args.dedup
            )), selected_rows))

    initial = run(prepare, rows)
    by_id = {pred["id"]: pred for pred in initial}
    labels = {row["id"]: [None] * len(records[row["id"]]["candidates"]) for row in rows}
    max_choices = max(map(len, labels.values()), default=0)
    for slot in range(max_choices):
        def choose(observation):
            return visible_matches(observation)[slot].sql

        # score_document grades complete sets, so use each untouched initial
        # rollout for tasks without this slot and ignore their score here.
        eligible = [row for row in rows if slot < len(labels[row["id"]])]
        changed = {}
        for pred in run(choose, eligible):
            if pred["trace"][:-1] != by_id[pred["id"]]["trace"][:-1]:
                raise ValueError("Candidate tool trajectory changed with selector")
            changed[pred["id"]] = pred
        predictions = [changed.get(row["id"], by_id[row["id"]]) for row in rows]
        document = encode_predictions(bank_path, manifest_path, predictions)
        outcome = score_document(document, rows, manifest, folder, references, reference_root,
                                 workers=args.workers, cache=cache)
        for task in outcome["per_question"]:
            if task["id"] in changed:
                labels[task["id"]][slot] = task["passed"]
        print(f"{args.split}: scored candidate position {slot + 1}/{max_choices}", flush=True)

    training = []
    for row in rows:
        values = labels[row["id"]]
        if any(values) and not all(values):
            training.append({**records[row["id"]], "correct": values})
    summary = {
        "split": args.split, "bank_sha256": digest(bank_path),
        "candidate_counts": dict(sorted(Counter(map(len, labels.values())).items())),
        "n": len(rows), "oracle_correct": sum(map(any, labels.values())),
        "mixed_correct_incorrect": len(training),
        "mean_tools": sum(item["tool_calls"] for item in initial) / len(initial),
        "training_rows_written": len(training) if args.split == "train" else 0,
    }
    if args.split == "train":
        write_jsonl(args.output / "train-ranker.jsonl", training)
    if args.write_rewrites:
        gold = {item["id"]: item["gold_sql"] for item in read_jsonl(folder / "train_memory.jsonl")}
        rewrite_rows = [{**generation[row["id"]], "id": row["id"], "sql": gold[row["id"]]}
                        for row in rows]
        write_jsonl(args.output / "train-rewriter.jsonl", rewrite_rows)
    write_json(args.output / f"{args.split}-pool-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
