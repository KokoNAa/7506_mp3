"""Select global ranking weights on validation, leaving the agent export independent."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
from pathlib import Path

from mp3_sql.common import ROOT, load_bank, write_json
from mp3_sql.environment import Episode, Tools
from mp3_sql.score import load_cases, score_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", action="store_true", help="Include per-task validation coverage")
    parser.add_argument("--learned-prior", type=Path, help="Recompute train-fitted prior from replayed candidates")
    args = parser.parse_args()
    saved = json.loads(args.predictions.read_text())
    if saved["split"] != "validation":
        raise ValueError("Only validation can select hyperparameters")
    logits = {record["id"]: record for line in args.calls.read_text().splitlines()
              if (record := json.loads(line)).get("purpose") == "rerank"}
    rows, manifest, folder = load_bank("validation", ROOT / "data")
    cases, case_root = load_cases("validation", folder)
    cache = {}
    tools = Tools(folder, manifest, cache=cache)
    grouped = defaultdict(dict)

    def trial(row, result, target):
        episode = Episode(row, tools)
        for event in result["trace"][:-1]:
            episode.step({key: value for key, value in event.items() if key != "observation"})
        if episode.trace != result["trace"][:-1]:
            raise ValueError("Original tool trajectory did not reproduce")
        episode.step({"tool": "submit", "sql": target})
        return {"id": row["id"], "sql": episode.submitted, "trace": episode.trace,
                "tool_calls": result["tool_calls"]}

    # All slots are genuine checked SQL submissions replayed from official actions.
    contenders = {}
    for result in saved["predictions"]:
        ident = result["id"]
        if ident in logits:
            contenders[ident] = [event["sql"] for event in result["trace"][:-1]
                                 if event["tool"] == "check" and event["observation"]["matches_example"]]
            if len(contenders[ident]) != len(logits[ident]["raw_logits"]):
                raise ValueError("Logits and candidate count differ")
        else:
            contenders[ident] = [result["sql"]]
    learned_weights = None
    if args.learned_prior:
        from mp3_agent.features import learned_score
        from mp3_agent.ranker import selection_example

        learned_weights = json.loads(args.learned_prior.read_text())["weights"]

        def recompute(pair):
            row, result = pair
            key = result["id"]
            if key not in logits:
                return key, None
            episode = Episode(row, tools)
            for call in result["trace"][:-1]:
                episode.step({k: v for k, v in call.items() if k != "observation"})
            if episode.trace != result["trace"][:-1]:
                raise ValueError("Original tool trajectory did not reproduce")
            record = selection_example(episode.observation())
            if record["candidates"] != contenders[key]:
                raise ValueError("Recomputed prior candidates differ from original ranking")
            return key, [learned_score(record, sql, learned_weights) for sql in contenders[key]]

        with ThreadPoolExecutor(max_workers=4) as pool:
            for key, values in pool.map(recompute, zip(rows, saved["predictions"])):
                if values is not None:
                    logits[key]["learned_prior"] = values
    slots = max(map(len, contenders.values()))
    for slot in range(slots):
        arguments = []
        for row, result in zip(rows, saved["predictions"]):
            targets = contenders[result["id"]]
            arguments.append((row, result, targets[slot] if slot < len(targets) else result["sql"]))
        with ThreadPoolExecutor(max_workers=4) as pool:
            predictions = list(pool.map(lambda args: trial(*args), arguments))
        document = {**saved, "predictions": predictions}
        scores = score_document(document, rows, manifest, folder, cases, case_root, workers=4, cache=cache)
        for task in scores["per_question"]:
            key = task["id"]
            if slot < len(contenders[key]):
                grouped[key][slot] = bool(task["passed"])

    grid = []
    generations = (0.0, 2.0, 5.0, 8.0) if any(any(rec.get("likelihood", [])) for rec in logits.values()) else (0.0,)
    for rw, lw, gw in product((0.0, 0.6, 1.4, 2.5), (0.0, 1.0, 2.5, 4.0, 7.0), generations):
        count = 0
        for key, values in grouped.items():
            slot = 0
            if key in logits:
                rec = logits[key]
                combined = [a + rw * b + lw * c + gw * d for a, b, c, d in zip(
                    rec["raw_logits"], rec["rule_prior"], rec["learned_prior"],
                    rec.get("likelihood", [0.0] * len(rec["raw_logits"]))
                )]
                slot = max(range(len(combined)), key=combined.__getitem__)
            count += values[slot]
        grid.append({"correct": count, "n": len(rows), "rule_weight": rw, "learned_strength": lw,
                     "likelihood_weight": gw})
    grid.sort(key=lambda record: (-record["correct"], record["likelihood_weight"], record["rule_weight"],
                                  record["learned_strength"]))
    checkpoint = json.loads(args.checkpoint.read_text())
    if learned_weights is not None:
        checkpoint["learned_weights"] = learned_weights
    best = {**checkpoint, **{field: grid[0][field] for field in (
        "rule_weight", "learned_strength", "likelihood_weight"
    )}}
    summary = {"pool_oracle": sum(any(v.values()) for v in grouped.values()),
               "tests": grid, "checkpoint": best}
    if args.details:
        summary["coverage"] = {key: [contenders[key][slot] for slot, passed in values.items() if passed]
                               for key, values in grouped.items()}
    write_json(args.output, summary)
    print(json.dumps({"best": grid[:5], "oracle": sum(any(v.values()) for v in grouped.values())}))


if __name__ == "__main__":
    main()
