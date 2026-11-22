"""Schema and question features for a small train-only ranking prior."""
import re
from functools import lru_cache

from sqlglot import exp

from .guided import shape_agreement, table_relevance
from .intent import intent_score
from .schema import compact, parse, render


def bounded(count, cap=4):
    return min(count, cap) / cap


@lru_cache(maxsize=2048)
def sql_identity(sql):
    tree = parse(sql)
    return render(tree) if tree is not None else sql


def features(example, sql):
    tree = parse(sql)
    if tree is None:
        return {}
    q = example["question"].casefold()
    schema = {table["table"].casefold(): table for table in example["schema"]}
    tables = {table.alias_or_name.casefold(): table.name.casefold() for table in tree.find_all(exp.Table)}
    where = tree.args.get("where")
    predicates = [*where.find_all(exp.Predicate)] if where else []
    literals = []
    for pred in predicates:
        literals.extend(lit.name.casefold().strip("%") for lit in pred.find_all(exp.Literal))
    grounded = [lit for lit in literals if lit and re.search(r"(?<!\w)" + re.escape(lit) + r"(?!\w)", q)]
    explicit = re.search(r"\b(never|without|exclude|not|no|except)\b", q)
    sql_negative = any(tree.find(cls) is not None for cls in (exp.Not, exp.NEQ, exp.Except))
    fk_refs = set()
    for name, table in schema.items():
        for source, target, dest in table["foreign_keys"]:
            if source and target and dest:
                fk_refs.add(frozenset(((name, compact(source)), (target.casefold(), compact(dest)))))
    fk = 0
    joins = 0
    bad_fk = 0
    for join in tree.find_all(exp.Join):
        joins += 1
        on = join.args.get("on")
        if not on:
            bad_fk += 1
            continue
        for eq in on.find_all(exp.EQ):
            left, right = eq.this, eq.expression
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            a, b = tables.get(left.table.casefold()), tables.get(right.table.casefold())
            edge = frozenset(((a, compact(left.name)), (b, compact(right.name))))
            fk += edge in fk_refs
            if a and b and edge not in fk_refs and compact(left.name) != compact(right.name):
                bad_fk += 1
    easy_all = bool(re.search(r"\b(all|every)\b", q) and not re.search(
        r"\b(who|whose|where|having|that|which|with|without)\b", q
    ))
    aggregate = any(isinstance(node.this if isinstance(node, exp.Alias) else node, exp.AggFunc)
                    for node in tree.selects)
    values = {
        "shape": shape_agreement(tree, {"columns": example["expected_columns"]}) / 2.8,
        "question": intent_score(tree, q, example["ordered"]) / 4.0,
        "fk_join": bounded(fk),
        "non_fk_join": bounded(bad_fk),
        "joins": bounded(joins),
        "predicate_count": bounded(len(predicates)),
        "grounded_filter": bounded(len(grounded)),
        "ungrounded_filter": bounded(len(literals) - len(grounded)),
        "grounded_name": max((table_relevance(table, q) for table in tables.values()), default=0.0),
        "unexpected_filter": float(easy_all and bool(where)),
        "requested_negation": float(bool(explicit) and sql_negative),
        "unrequested_negation": float(not explicit and sql_negative),
        "group_count": float(bool(tree.args.get("group")) and aggregate),
        "unexpected_limit": float(bool(tree.args.get("limit")) and not re.search(
            r"\b(first|last|recent|lowest|highest|oldest|youngest|smallest|largest|most|least|top|bottom)\b", q
        )),
        "cast_count": bounded(sum(1 for _ in tree.find_all(exp.Cast))),
    }
    key = sql_identity(sql)
    draft = example.get("draft_sql")
    hints = example.get("hint_sqls", [])
    empty = float(example["expected_count"] == 0)
    draft_match = float(bool(draft) and key == sql_identity(draft))
    hint_votes = sum(key == sql_identity(hint) for hint in hints) / max(len(hints), 1)
    tautologies = sum(1 for join in tree.find_all(exp.Join) for eq in join.find_all(exp.EQ)
                      if eq.this == eq.expression)
    values.update(draft=draft_match, hint_votes=hint_votes, empty_draft=empty * draft_match,
                  empty_hint_votes=empty * hint_votes, empty_shape=empty * values["shape"],
                  empty_joins=empty * values["joins"], tautological_join=bounded(tautologies))
    return values


def learned_score(example, sql, weights):
    return sum(weights.get(key, 0.0) * value for key, value in features(example, sql).items())
