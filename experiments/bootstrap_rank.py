"""Label actual model-written and rule-written candidates on train tasks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from mp3_agent.agent import checked_choice
from mp3_agent.ranker import selection_example
from mp3_sql.common import ROOT, load_bank, write_json, write_jsonl
from mp3_sql.environment import Episode, Tools, rollout
from mp3_sql.score import load_cases, score_document

from make_pool import encode_predictions
from submission import create_agent


class Draft:
    def act(self, observation):
        return {"tool": "submit", "sql": observation["issue_sql"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=240)
    args = parser.parse_args()
    rows, manifest, folder = load_bank("train", ROOT / "data")
    seed_order = sorted(rows, key=lambda row: sha256(("bootstrap:" + row["id"]).encode()).hexdigest())
    selected = {row["id"] for row in seed_order[:args.sample_size]}
    cases, case_root = load_cases("train", folder)
    cache = {}
    tools = Tools(folder, manifest, cache=cache)
    records = {}
    initial = []

    def save(observation):
        key = observation["task"]["id"]
        records[key] = {"id": key, **selection_example(observation)}
        return checked_choice(observation, guided=True)

    count = 0
    for row in rows:
        if row["id"] in selected:
            agent = create_agent(args.checkpoint)
            agent.selector = save
            initial.append(rollout(row, tools, agent))
            count += 1
            if count % 30 == 0:
                print(f"Collected {count}/{len(selected)} training trajectories", flush=True)
        else:
            initial.append(rollout(row, tools, Draft()))
    labels = {key: [None] * len(record["candidates"]) for key, record in records.items()}

    def select(row, prediction, index):
        key = row["id"]
        record = records.get(key)
        if record is None or index >= len(record["candidates"]):
            return prediction
        episode = Episode(row, tools)
        for event in prediction["trace"][:-1]:
            episode.step({k: v for k, v in event.items() if k != "observation"})
        if episode.trace != prediction["trace"][:-1]:
            raise ValueError("Training trajectory did not replay")
        scenario = record["candidates"][index]
        episode.step({"tool": "submit", "sql": scenario})
        return {**prediction, "sql": scenario, "trace": episode.trace}

    slots = max(map(len, labels.values()), default=0)
    for slot in range(slots):
        with ThreadPoolExecutor(max_workers=4) as pool:
            predictions = list(pool.map(lambda pair: select(*pair, slot), zip(rows, initial)))
        document = encode_predictions(folder / "train.jsonl", folder / "manifest.json", predictions)
        assessment = score_document(document, rows, manifest, folder, cases, case_root,
                                    workers=4, cache=cache)
        for result in assessment["per_question"]:
            key = result["id"]
            if key in labels and slot < len(labels[key]):
                labels[key][slot] = result["passed"]
        print(f"Scored candidate position {slot + 1}/{slots}", flush=True)
    prepared = [{**records[key], "correct": values} for key, values in labels.items()
                if any(values) and not all(values)]
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "train-bootstrapped-ranker.jsonl", prepared)
    write_json(args.output / "bootstrap-summary.json",
               {"evaluated": len(selected), "mixed_examples": len(prepared),
                "candidate_oracle": sum(any(values) for values in labels.values())})
    print(f"{len(prepared)} ranking examples from {len(selected)} official train tasks", flush=True)


if __name__ == "__main__":
    main()
