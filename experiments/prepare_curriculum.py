"""Create additional repair and synthesis examples from train references only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

from sqlglot import exp

from mp3_agent.schema import identifier, parse, render
from mp3_sql.common import ROOT, load_bank, write_jsonl
from mp3_sql.environment import Episode, Tools


SWAPS = ((exp.GT, exp.GTE), (exp.LT, exp.LTE), (exp.EQ, exp.NEQ),
         (exp.And, exp.Or), (exp.Min, exp.Max))


def corruptions(tree):
    if not isinstance(tree, exp.Select):
        return
    nodes = list(tree.walk())
    for i, node in enumerate(nodes):
        for pair in SWAPS:
            if not isinstance(node, pair):
                continue
            other = tree.copy()
            target = list(other.walk())[i]
            opposite = pair[1] if isinstance(node, pair[0]) else pair[0]
            target.replace(opposite(**{key: value.copy() if isinstance(value, exp.Expression) else value
                                       for key, value in target.args.items()}))
            yield render(other)
            break
    if tree.expressions and len(tree.expressions) > 1:
        wrong = tree.copy()
        wrong.set("expressions", wrong.expressions[::-1])
        yield render(wrong)
    if tree.args.get("where"):
        wrong = tree.copy()
        wrong.set("where", None)
        yield render(wrong)
    if tree.args.get("group"):
        wrong = tree.copy()
        wrong.set("group", None)
        wrong.set("having", None)
        yield render(wrong)
    if tree.args.get("order"):
        wrong = tree.copy()
        first = wrong.args["order"].expressions[0]
        first.set("desc", not bool(first.args.get("desc")))
        yield render(wrong)
    wrong = tree.copy()
    wrong.set("distinct", None if wrong.args.get("distinct") else exp.Distinct())
    yield render(wrong)
    columns = list(tree.find_all(exp.Column))
    if columns:
        wrong = tree.copy()
        column = next(wrong.find_all(exp.Column))
        old = column.name
        if len(old) > 2:
            column.set("this", identifier(old[:1] + old[2:]))
            yield render(wrong)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {r["id"]: r for line in args.source.read_text().splitlines() if (r := json.loads(line))}
    rows, manifest, folder = load_bank("train", ROOT / "data")
    if set(records) != {row["id"] for row in rows}:
        raise ValueError("Expected the official train task IDs")
    tools = Tools(folder, manifest)

    def expand(row):
        item = dict(records[row["id"]])
        preview = row["example"]["rows"][:2]
        item["expected_preview"] = preview
        outputs = [{**item, "mode": "repair"}, {**item, "mode": "synthesize"}]
        tree = parse(item["sql"])
        if tree is None:
            return outputs
        options = list(dict.fromkeys(corruptions(tree)))
        rng = random.Random(row["id"])
        rng.shuffle(options)
        episode = Episode(row, tools)
        for draft in options[:3]:
            if len(draft.encode()) > 16000 or draft == render(tree):
                continue
            episode.step({"tool": "check", "sql": draft})
            outcome = episode.trace[-1]["observation"]
            if outcome["matches_example"]:
                continue
            attempt = {"sql": draft, "match": False, "error": (outcome["error"] or {}).get("message", ""),
                       "columns": outcome["columns"], "rows": outcome["row_count"], "preview": outcome["rows"][:2]}
            outputs.append({**item, "mode": "repair", "attempts": [attempt, *item["attempts"][:1]]})
            break
        return outputs

    with ThreadPoolExecutor(max_workers=4) as pool:
        expanded = [record for batch in pool.map(expand, rows) for record in batch]
    write_jsonl(args.output, expanded)
    print(f"{len(expanded)} examples from {len(rows)} train tasks")


if __name__ == "__main__":
    main()
