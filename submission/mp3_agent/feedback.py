"""Small corrective edits selected by concrete output differences."""
from collections import Counter
import json
import re

from sqlglot import exp

from .guided import NUMBER, THRESHOLD, THRESHOLD_OP, signature
from .intent import DISTINCT, descending, projections
from .structural import conjunctions


def row_keys(rows):
    return [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]


def tied_order(tree, example, question):
    rows = example["rows"]
    count = len(example["columns"])
    if len(rows) < 2:
        return
    projected = projections(tree)
    if count != len(projected) or any(signature(node) is None for node in projected):
        return
    choices = []
    cue = descending(question)
    suffix = question.rsplit(" by ", 1)[-1] if " by " in question else ""
    for slot in range(count):
        vals = [row[slot] for row in rows]
        values = [value for value in vals if value is not None]
        if len(values) < 2 or any(type(x) is not type(values[0]) for x in values):
            continue
        no_tie = len(set(values)) > 1
        for reverse in (False, True):
            if not no_tie or cue is not None and cue != reverse:
                continue
            if values == sorted(values, reverse=reverse):
                name = example["columns"][slot].casefold().replace("_", " ")
                preference = float(name in suffix and bool(suffix))
                choices.append((preference, slot, reverse))
    for relevance, slot, reverse in sorted(choices, key=lambda item: -item[0])[:2]:
        repaired = tree.copy()
        repaired.set("order", exp.Order(expressions=[
            exp.Ordered(this=exp.Literal.number(slot + 1), desc=reverse, nulls_first=not reverse)
        ]))
        yield 5.1 + relevance, repaired


def obvious_grounding(predicate, question):
    for literal in predicate.find_all(exp.Literal):
        name = literal.name.casefold().strip("%")
        if name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question):
            return True
    return False


def contradicted_having(tree, question):
    match = THRESHOLD.search(question)
    clause = tree.args.get("having")
    if not match or clause is None:
        return
    target = THRESHOLD_OP.get(match[1], exp.EQ)
    number = NUMBER.get(match[2], match[2])
    for i, expr in enumerate(clause.find_all(exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE)):
        if not isinstance(expr.this, exp.Count) or not isinstance(expr.expression, exp.Literal):
            continue
        if isinstance(expr, target) and expr.expression.name == str(number):
            continue
        clone = tree.copy()
        other = list(clone.args["having"].find_all(exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE))[i]
        other.replace(target(this=other.this.copy(), expression=exp.Literal.number(number)))
        yield 4.6, clone


def feedback_repairs(tree, result, example, question, ordered):
    if not isinstance(tree, exp.Select) or not result["ok"] or result["matches_example"]:
        return
    if len(result["columns"]) != len(example["columns"]):
        return
    q = question.casefold()
    expected = Counter(row_keys(example["rows"]))
    actual = Counter(row_keys(result["rows"]))
    theirs = example["row_count"]
    ours = result["row_count"]
    if ordered and expected == actual and not result["truncated"] and not example["truncated"]:
        yield from tied_order(tree, example, q)
    if ours > theirs > 0 and actual and set(actual).issubset(expected):
        if any(value > expected[key] for key, value in actual.items()) and not tree.args.get("distinct"):
            amended = tree.copy()
            amended.set("distinct", exp.Distinct())
            yield 5.4, amended
    if 0 < ours < theirs and actual and set(actual).issubset(expected):
        if tree.args.get("distinct") and not DISTINCT.search(q):
            amended = tree.copy()
            amended.set("distinct", None)
            yield 5.1, amended
        where = tree.args.get("where")
        if where is not None and isinstance(where.this, exp.And):
            predicates = list(conjunctions(where.this))
            grounded = [obvious_grounding(term, q) for term in predicates]
            if any(grounded) and not all(grounded):
                amended = tree.copy()
                amended.set("where", exp.Where(this=exp.and_(*[
                    term.copy() for term, keep in zip(predicates, grounded) if keep
                ])))
                yield 4.7, amended
    yield from contradicted_having(tree, q)
