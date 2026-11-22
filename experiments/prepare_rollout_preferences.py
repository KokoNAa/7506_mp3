"""Label actual decoder mistakes with the official train-only scorer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from mp3_agent.rewrite import extract_query
from mp3_sql.common import ROOT, load_bank, write_json, write_jsonl
from mp3_sql.environment import Episode, Tools
from mp3_sql.score import load_cases, score_document

from make_pool import encode_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest, folder = load_bank("train", ROOT / "data")
    expected, case_root = load_cases("train", folder)
    labels = {item["id"]: item["sql"] for line in args.source.read_text().splitlines()
              if (item := json.loads(line))}
    ids = {row["id"] for row in rows}
    if ids != labels.keys():
        raise ValueError("Expected exactly the official train task IDs")
    raw = args.calls.read_bytes()
    calls = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if any(call["id"] not in ids for call in calls):
        raise ValueError("Every rollout must be from the official train split")
    grouped = {}
    for call in calls:
        if call.get("purpose") != "rewrite":
            continue
        sql = extract_query(call["generated_text"])
        if sql and sql != labels[call["id"]] and len(sql.encode()) <= 16000:
            group = grouped.setdefault(call["id"], {})
            group.setdefault(sql, call["prompt"])
    alternatives = {key: list(value)[:3] for key, value in grouped.items()}
    cache = {}
    tools = Tools(folder, manifest, cache=cache)

    def evaluate(targets):
        def predict(row):
            target = targets.get(row["id"], labels[row["id"]])
            state = Episode(row, tools)
            state.step({"tool": "check", "sql": target})
            state.step({"tool": "submit", "sql": target})
            return {"id":row["id"], "sql":target, "trace":state.trace, "tool_calls":1}

        with ThreadPoolExecutor(max_workers=4) as pool:
            predictions = list(pool.map(predict, rows))
        document = encode_predictions(folder/"train.jsonl", folder/"manifest.json", predictions)
        result = score_document(document, rows, manifest, folder, expected, case_root, workers=4, cache=cache)
        return {score["id"]:score["passed"] for score in result["per_question"]}

    reference = evaluate({})
    pairs = []
    for slot in range(max(map(len, alternatives.values()), default=0)):
        targets = {key: values[slot] for key, values in alternatives.items() if slot < len(values)}
        scored = evaluate(targets)
        for key, sql in targets.items():
            if reference[key] and not scored[key]:
                pairs.append({"id":key, "prompt":grouped[key][sql], "sql":labels[key], "rejected_sql":sql})
    if not pairs:
        raise ValueError("No valid preference pairs from training rollouts")
    report = {"split":"train", "train_examples":len(rows), "decoded_tasks":len(alternatives),
              "preference_pairs":len(pairs), "unique_pair_tasks":len({row["id"] for row in pairs}),
              "rollout_calls_sha256":hashlib.sha256(raw).hexdigest()}
    write_jsonl(args.output, pairs)
    write_json(args.summary, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
