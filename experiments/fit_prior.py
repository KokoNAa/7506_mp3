"""Fit a small pairwise preference model using checked training candidates."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from mp3_agent.features import features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--attach-train-context", action="store_true")
    args = parser.parse_args()
    source = args.train.read_bytes()
    tasks = [json.loads(line) for line in source.splitlines() if line.strip()]
    if not tasks or any(not task["id"].startswith("train-") for task in tasks):
        raise ValueError("Fit the prior on labeled training tasks only")
    if args.attach_train_context:
        from concurrent.futures import ThreadPoolExecutor
        from mp3_sql.common import ROOT, load_bank
        from mp3_sql.environment import Episode, Tools

        rows, manifest, folder = load_bank("train", ROOT / "data")
        tools = Tools(folder, manifest)
        wanted = {task["id"] for task in tasks}

        def context(row):
            episode = Episode(row, tools)
            episode.step({"tool": "check", "sql": row["issue_sql"]})
            episode.step({"tool": "repair_hint"})
            return row["id"], {"draft_sql": row["issue_sql"], "hint_sqls": episode.trace[-1]["observation"]}

        with ThreadPoolExecutor(max_workers=4) as pool:
            data = dict(pool.map(context, [row for row in rows if row["id"] in wanted]))
        if data.keys() != wanted:
            raise ValueError("Ranking labels must belong to the official training bank")
        tasks = [{**task, **data[task["id"]]} for task in tasks]
    groups = []
    keys = set()
    for task in tasks:
        values = [features(task, sql) for sql in task["candidates"]]
        keys.update(key for value in values for key in value)
        groups.append([(positive, negative) for positive, good in zip(values, task["correct"])
                       for negative, bad in zip(values, task["correct"]) if good and not bad])
    keys = sorted(keys)
    differences = [
        ([positive.get(k, 0.0) - negative.get(k, 0.0) for k in keys], 1 / len(group) / len(groups))
        for group in groups for positive, negative in group
    ]
    weights = [0.0] * len(keys)
    for _ in range(150):
        gradient = [args.regularization * weight for weight in weights]
        for difference, scale in differences:
            margin = sum(a * b for a, b in zip(weights, difference))
            probability = 1 / (1 + math.exp(max(-40, min(40, margin))))
            for i, delta in enumerate(difference):
                gradient[i] -= scale * probability * delta
        weights = [weight - delta for weight, delta in zip(weights, gradient)]
    args.output.write_text(json.dumps({"data_sha256": sha256(source).hexdigest(), "n": len(tasks),
                                      "train_context": args.attach_train_context,
                                      "weights": dict(zip(keys, weights))}, indent=2) + "\n")


if __name__ == "__main__":
    main()
