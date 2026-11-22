"""Generate new question/SQL pairs from official training schemas and execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import re
from pathlib import Path
import random

from sqlglot import exp

from mp3_agent.schema import Catalog, parse
from mp3_sql.common import ROOT, load_bank, write_jsonl
from mp3_sql.environment import Episode, Tools

from prepare_curriculum import corruptions


def quoted(value):
    return '"' + value.replace('"', '""') + '"'


def words(value):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", value).replace("_", " ")


def plans(schema):
    catalog = Catalog(schema)
    for key, table in catalog.tables.items():
        name = table["name"]
        source = quoted(name)
        cols = [column for column in table["columns"] if column["name"]]
        for col in cols:
            if re.search(r"int|real|double|float|numeric|decimal", col["type"] or "", re.I):
                field = col["name"]
                value = quoted(field)
                yield (f"What are the maximum and minimum {words(field)} values in {words(name)}?",
                       f"SELECT MAX({value}), MIN({value}) FROM {source}", False)
                yield (f"How many rows in {words(name)} have {words(field)} higher than the overall average {words(field)}?",
                       f"SELECT COUNT(*) FROM {source} WHERE {value} > (SELECT AVG({value}) FROM {source})", False)
                yield (f"List the {words(field)} values in {words(name)} below its overall average, in descending order.",
                       f"SELECT {value} FROM {source} WHERE {value} < (SELECT AVG({value}) FROM {source}) "
                       f"ORDER BY {value} DESC", True)
            if re.search(r"char|text|string", col["type"] or "", re.I):
                field = col["name"]
                dimension = quoted(field)
                yield (
                    f"Give each {words(field)} value in {words(name)} and its number of rows, keeping groups with at least two rows.",
                    f"SELECT {dimension}, COUNT(*) FROM {source} GROUP BY {dimension} HAVING COUNT(*) >= 2",
                    False,
                )
                yield (
                    f"Which three {words(field)} values occur most frequently in {words(name)}? "
                    f"Return each value and the count, sorting ties by {words(field)}.",
                    f"SELECT {dimension}, COUNT(*) FROM {source} GROUP BY {dimension} "
                    f"ORDER BY COUNT(*) DESC, {dimension} ASC LIMIT 3",
                    True,
                )
        for relation in catalog.edges[key]:
            if not relation.explicit:
                continue
            other_name = catalog.tables[relation.table]["name"]
            other = quoted(other_name)
            join = " AND ".join(f"lhs.{quoted(left)} = rhs.{quoted(right)}" for left, right in relation.pairs)
            yield (
                f"How many rows in {words(name)} have no matching row in {words(other_name)}?",
                f"SELECT COUNT(*) FROM {source} AS lhs WHERE NOT EXISTS "
                f"(SELECT 1 FROM {other} AS rhs WHERE {join})",
                False,
            )
            yield (
                f"How many rows of {words(name)} have at least one corresponding record in {words(other_name)}?",
                f"SELECT COUNT(*) FROM {source} AS lhs WHERE EXISTS (SELECT 1 FROM {other} AS rhs WHERE {join})",
                False,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest, folder = load_bank("train", ROOT / "data")
    tools = Tools(folder, manifest, cache={})

    def expand(row):
        episode = Episode(row, tools)
        episode.step({"tool": "schema"})
        schema = episode.trace[-1]["observation"]
        combos = list(dict.fromkeys(plans(schema)))
        rng = random.Random("new-train-pair:" + row["id"])
        rng.shuffle(combos)
        for question, sql, ordered in combos[:3]:
            tree = parse(sql)
            if tree is None or len(sql.encode()) > 16000 or any(event.get("sql") == sql for event in episode.trace):
                continue
            episode.step({"tool": "execute", "sql": sql})
            target = episode.trace[-1]["observation"]
            if not target["ok"]:
                continue
            drafts = list(dict.fromkeys(corruptions(tree)))
            rng.shuffle(drafts)
            if not drafts:
                continue
            draft = drafts[0]
            if any(event.get("sql") == draft for event in episode.trace):
                continue
            episode.step({"tool": "execute", "sql": draft})
            work = episode.trace[-1]["observation"]
            used = {table.name.casefold() for table in tree.find_all(exp.Table)}
            tables = sorted(schema, key=lambda t: t["name"].casefold() not in used)
            context = [f'{t["name"]}({", ".join(c["name"] for c in t["columns"])})' +
                       (f' FK {", ".join(f"{fk['from']}={fk['table']}.{fk['to']}" for fk in t["foreign_keys"])}'
                        if t.get("foreign_keys") else "") for t in tables[:22]]
            attempts = [{"sql": draft, "match": work["ok"] and work["rows"] == target["rows"] and
                         work["row_count"] == target["row_count"] and len(work["columns"]) == len(target["columns"]),
                         "preview": work["rows"][:2], "rows": work["row_count"], "columns": work["columns"],
                         "error": (work["error"] or {}).get("message", "")}]
            yield {"id": row["id"], "question": question, "schema": context, "ordered": ordered,
                   "sql": sql, "expected_columns": target["columns"], "expected_count": target["row_count"],
                   "expected_preview": target["rows"][:2], "attempts": attempts}

    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(lambda row: list(expand(row)), rows))
    unique = {}
    for row, group in zip(rows, groups):
        for record in group:
            unique.setdefault((row["db_id"], record["sql"]), record)
    samples = []
    for position, record in enumerate(unique.values()):
        samples.append({**record, "mode": "repair" if position % 3 else "synthesize"})
    write_jsonl(args.output, samples)
    print(f"Generated {len(samples)} executable examples across train schemas")


if __name__ == "__main__":
    main()
