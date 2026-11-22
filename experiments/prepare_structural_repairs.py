"""Prepare targeted SQL repair supervision using only the official train split."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import traverse_scope

from mp3_agent.schema import Catalog, field, identifier, parse, render, scope_sources
from mp3_sql.common import ROOT, load_bank, write_json, write_jsonl
from mp3_sql.environment import Episode, Tools


COMPARISONS = {exp.EQ: exp.GTE, exp.GTE: exp.GT, exp.GT: exp.GTE,
               exp.LT: exp.LTE, exp.LTE: exp.LT}


def join_variants(tree, catalog):
    for index, join in enumerate(tree.find_all(exp.Join)):
        condition = join.args.get("on")
        equal = condition.find(exp.EQ) if condition is not None else None
        if equal is None or not all(isinstance(col, exp.Column) for col in (equal.this, equal.expression)):
            continue
        edited = tree.copy()
        target = list(edited.find_all(exp.Join))[index].args["on"].find(exp.EQ)
        scope = next((s for s in traverse_scope(edited)
                      if any(j is target.find_ancestor(exp.Join) for j in s.expression.args.get("joins", []))), None)
        if scope is None:
            continue
        _, aliases, _ = scope_sources(scope, catalog)
        rhs = target.expression
        cols = next((values for alias, values in aliases.items() if alias.casefold() == rhs.table.casefold()), {})
        alternatives = [name for name in cols.values() if name.casefold() != rhs.name.casefold()
                        and (name.casefold().endswith("id") or name.casefold().endswith("code"))]
        if alternatives:
            target.set("expression", field(sorted(alternatives)[0], rhs.table))
        else:
            rhs.set("this", identifier(rhs.name + "_id"))
        yield "join_key", edited


def grouping_variants(tree):
    for index, select in enumerate(tree.find_all(exp.Select)):
        having = select.args.get("having")
        grouping = select.args.get("group")
        if grouping is None and having is None:
            continue
        if having is not None:
            edited = tree.copy()
            where = list(edited.find_all(exp.Select))[index].args["having"]
            comparison = next((pred for pred in where.walk() if type(pred) in COMPARISONS), None)
            if comparison is not None:
                comparison.replace(COMPARISONS[type(comparison)](this=comparison.this.copy(),
                                                                 expression=comparison.expression.copy()))
                yield "having_boundary", edited
            removed = tree.copy()
            list(removed.find_all(exp.Select))[index].set("having", None)
            yield "missing_having", removed
        if grouping is not None and len(grouping.expressions) > 1:
            edited = tree.copy()
            changed = list(edited.find_all(exp.Select))[index].args["group"]
            changed.set("expressions", changed.expressions[:-1])
            yield "group_key", edited


def exclusion_variants(tree):
    if isinstance(tree, exp.Except):
        yield "exclusion", exp.Intersect(this=tree.this.copy(), expression=tree.expression.copy(), distinct=True)
    for index, negation in enumerate(tree.find_all(exp.Not)):
        if not isinstance(negation.this, (exp.Exists, exp.In)):
            continue
        edited = tree.copy()
        changed = list(edited.find_all(exp.Not))[index]
        changed.replace(changed.this.copy())
        yield "exclusion", edited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    source = {record["id"]: record for line in args.source.read_text().splitlines()
              if (record := json.loads(line))}
    rows, manifest, folder = load_bank("train", ROOT / "data")
    if set(source) != {row["id"] for row in rows}:
        raise ValueError("Expected the official training task IDs")
    tools = Tools(folder, manifest, cache={})

    def prepare(row):
        labels = source[row["id"]]
        reference = parse(labels["sql"])
        if reference is None:
            return []
        episode = Episode(row, tools)
        episode.step({"tool": "schema"})
        catalog = Catalog(episode.trace[-1]["observation"])
        try:
            edits = [*join_variants(reference, catalog), *grouping_variants(reference),
                     *exclusion_variants(reference)]
        except (SqlglotError, RecursionError):
            return []
        if not edits:
            return []
        rng = random.Random("structure:" + row["id"])
        rng.shuffle(edits)
        episode.step({"tool": "check", "sql": labels["sql"]})
        if not episode.trace[-1]["observation"]["matches_example"]:
            return []
        output = []
        kinds = set()
        seen = {labels["sql"], render(reference)}
        for name, amended in edits:
            if len(output) >= 3 or not episode.remaining or name in kinds:
                continue
            draft = render(amended)
            if draft in seen or len(draft.encode()) > 16000:
                continue
            seen.add(draft)
            episode.step({"tool": "check", "sql": draft})
            feedback = episode.trace[-1]["observation"]
            if feedback["matches_example"]:
                continue
            attempt = {"sql": draft, "match": False,
                       "error": (feedback["error"] or {}).get("message", ""), "columns": feedback["columns"],
                       "rows": feedback["row_count"], "preview": feedback["rows"][:2]}
            record = {**labels, "mode": "repair", "attempts": [attempt, *labels["attempts"][:1]],
                      "expected_preview": row["example"]["rows"][:2]}
            output.append((name, record))
            kinds.add(name)
        return output

    with ThreadPoolExecutor(max_workers=4) as pool:
        output = [entry for examples in pool.map(prepare, rows) for entry in examples]
    payload = [record for _, record in output]
    summary = {"split": "train", "source_tasks": len(rows), "unique_tasks": len({x["id"] for x in payload}),
               "examples": len(payload), "mutations": dict(sorted(Counter(kind for kind, _ in output).items()))}
    write_jsonl(args.output, payload)
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
