"""Generate SQLite candidates from public result shape and schema constraints."""
from collections import Counter
import re

from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from .schema import apply_path, compact, field, identifier, parse, render, scope_sources


AGGS = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
NUMBER = {word: i for i, word in enumerate(
    "zero one two three four five six seven eight nine ten".split())}
THRESHOLD = re.compile(
    r"\b(at least|at most|more than|less than|fewer than|exactly|with|built|have|having|has)"
    r"\s+(\d+|" + "|".join(NUMBER) + r")\b"
)
THRESHOLD_OP = {"at least": exp.GTE, "at most": exp.LTE, "more than": exp.GT,
                "less than": exp.LT, "fewer than": exp.LT}
PER_GROUP = re.compile(r"\b(each|per|grouped by|for every)\b")
FREQUENCY = re.compile(r"\b(common|frequent|frequently|popular)\b")
TOP_ONE = re.compile(r"\b(most|least|highest|lowest|largest|smallest|first|last)\b")


def signature(node):
    if isinstance(node, exp.Alias):
        return signature(node.this)
    if isinstance(node, exp.Column):
        return ("column", compact(node.name))
    if isinstance(node, exp.Star):
        return ("star",)
    if isinstance(node, exp.Distinct) and len(node.expressions) == 1:
        return ("distinct", signature(node.expressions[0]))
    if isinstance(node, AGGS) and node.this is not None:
        return (node.key, signature(node.this))
    return None


def header(label, catalog=None):
    if not isinstance(label, str) or not label.strip() or len(label) > 160:
        return None
    if catalog is not None and any(label.casefold() in cols for cols in catalog.columns.values()):
        return exp.Column(this=identifier(label))
    tree = parse("SELECT " + label)
    if tree is None or not isinstance(tree, exp.Select) or len(tree.expressions) != 1:
        return None
    if any(value for key, value in tree.args.items() if key not in ("expressions", "kind")):
        return None
    node = tree.expressions[0]
    key = signature(node)
    if not key or key[0] in ("star", "distinct") or isinstance(node, exp.Alias):
        return None
    return node


def shape_agreement(tree, example):
    if tree is None:
        return 0.0
    expected = [signature(h) for h in (header(label) for label in example.get("columns", []))]
    if not expected or not any(expected):
        return 0.0
    actual = [signature(node) for node in tree.selects]
    positional = sum(1 for x, y in zip(expected, actual) if x is not None and x == y)
    equal = positional / len(expected)
    return 2.8 * equal - 0.9 * abs(len(actual) - len(expected))


def projection(tree, catalog, desired, question):
    amended = tree.copy()
    scope = traverse_scope(amended)[-1]
    bindings, known, _ = scope_sources(scope, catalog)
    expressions = []
    original = {signature(node): node for node in amended.expressions}
    for goal in desired:
        existing = original.get(signature(goal))
        if existing is not None:
            expressions.append(existing.copy())
            continue
        candidate = goal.copy()
        columns = list(candidate.find_all(exp.Column))
        for col in columns:
            key = compact(col.name)
            matches = [(alias, name) for alias, cols in known.items() for name in cols.values()
                       if compact(name) == key]
            if not matches:
                paths = [(alias, path, name) for table, cols in catalog.columns.items()
                         if table not in bindings.values()
                         for name in cols.values() if compact(name) == key
                         for alias, path in catalog.paths(bindings, table)]
                if not paths:
                    return None
                alias, route, name = min(paths, key=lambda option: (
                    len(option[1]), sum(not edge.explicit for edge in option[1]),
                    -table_relevance(option[1][-1].table, question)
                ))
                alias = apply_path(amended, alias, route, catalog, bindings)
                for edge in route:
                    name_of_table = catalog.tables[edge.table]["name"]
                    bindings[name_of_table] = edge.table
                    known[name_of_table] = catalog.columns[edge.table]
                matches = [(alias, name)]
            alias, name = max(matches, key=lambda item: (
                item[0].casefold() == col.table.casefold(), table_relevance(bindings.get(item[0]) or item[0], question)
            ))
            replacement = field(name, alias)
            if col is candidate:
                candidate = replacement
            else:
                col.replace(replacement)
        expressions.append(candidate)
    amended.set("expressions", expressions)
    return amended


def table_relevance(name, question):
    terms = {word.rstrip("s") for word in re.findall(r"[a-z0-9]+", name.casefold())}
    qterms = {word.rstrip("s") for word in re.findall(r"[a-z0-9]+", question.casefold())}
    if terms & qterms:
        return len(terms & qterms) / max(1, len(terms))
    return 0.0


def relevant_filter(tree, question):
    where = tree.args.get("where")
    if not where:
        return False
    q = question.casefold()
    values = [literal.name.casefold().strip("%") for literal in where.find_all(exp.Literal)]
    return any(value and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", q) for value in values)


def fewer_joins(tree, catalog):
    """Drop tail inner joins referenced only through their own ON clause."""
    edited = tree.copy()
    exposed = edited.args.get("joins") or []
    if not exposed or any(not isinstance(join.this, exp.Table) or join.args.get("side") for join in exposed):
        return None
    scope = traverse_scope(edited)[-1]
    bindings, cols, _ = scope_sources(scope, catalog)
    used = set()
    for col in scope.columns:
        if col.find_ancestor(exp.Join) is not None:
            continue
        if col.table:
            used.add(col.table.casefold())
        else:
            aliases = [alias.casefold() for alias, names in cols.items() if col.name.casefold() in names]
            used.update(aliases)
    retained = []
    for join in reversed(exposed):
        alias = join.this.alias_or_name.casefold()
        if alias in used:
            retained.append(join)
            used.update(col.table.casefold() for col in join.find_all(exp.Column) if col.table)
    if len(retained) == len(exposed):
        return None
    edited.set("joins", list(reversed(retained)))
    return edited


def fk_variants(tree, catalog):
    scope = traverse_scope(tree)[-1]
    bindings, _, _ = scope_sources(scope, catalog)
    from_ = tree.args.get("from") or tree.args.get("from_")
    if not from_ or not isinstance(from_.this, exp.Table):
        return
    preceding = [from_.this.alias_or_name]
    for i, join in enumerate(tree.args.get("joins") or []):
        if not isinstance(join.this, exp.Table):
            continue
        right = join.this.alias_or_name
        target = bindings.get(right)
        on = join.args.get("on")
        if target and on and isinstance(on, (exp.EQ, exp.And)):
            matching = tuple((alias, edge) for alias in preceding
                             for edge in catalog.edges.get(bindings.get(alias, ""), [])
                             if edge.table == target and edge.explicit)
            for left, edge in matching[:2]:
                term = exp.and_(*(exp.EQ(this=field(a, left), expression=field(b, right))
                                  for a, b in edge.pairs))
                source_pairs = {
                    frozenset((compact(pair.this.name), compact(pair.expression.name)))
                    for pair in on.find_all(exp.EQ)
                    if isinstance(pair.this, exp.Column) and isinstance(pair.expression, exp.Column)
                }
                target_pairs = {frozenset((compact(a), compact(b))) for a, b in edge.pairs}
                if source_pairs == target_pairs:
                    continue
                alternative = tree.copy()
                alternative.args["joins"][i].set("on", term)
                yield alternative
        preceding.append(right)


def guided_edits(tree, catalog, question, example, ordered):
    if not isinstance(tree, exp.Select):
        return
    goals = [header(label, catalog) for label in example.get("columns", [])]
    q = question.casefold()
    all_goals = bool(goals) and all(goal is not None for goal in goals)
    current = tree
    if all_goals:
        variant = projection(tree, catalog, goals, question)
        if variant is not None and render(variant) != render(tree):
            current = variant
            if shape_agreement(tree, example) < shape_agreement(variant, example):
                yield 3.9, variant
    projections = [node.this if isinstance(node, exp.Alias) else node for node in current.expressions]
    dimensions = [node for node in projections if isinstance(node, exp.Column)]
    aggregates = [node for node in projections if isinstance(node, AGGS)]
    grouped = current.args.get("group")
    limit = current.args.get("limit")
    if grouped and aggregates and not dimensions and not PER_GROUP.search(q) and example["row_count"] <= 1:
        simpler = current.copy()
        for part in ("group", "having", "order", "limit"):
            simpler.set(part, None)
        yield 4.2, simpler
        trimmed = fewer_joins(simpler, catalog)
        if trimmed is not None:
            yield 3.5, trimmed
    if dimensions and aggregates and len(dimensions) + len(aggregates) == len(projections):
        if not grouped and PER_GROUP.search(q):
            variant = current.copy()
            variant.set("group", exp.Group(expressions=[field.copy() for field in dimensions]))
            yield 3.5, variant
        if grouped and not PER_GROUP.search(q) and example["row_count"] <= 1 and any(
            isinstance(node, (exp.Max, exp.Min)) for node in aggregates
        ):
            variant = current.copy()
            for part in ("group", "having", "order", "limit"):
                variant.set(part, None)
            yield 4.1, variant
    if len(dimensions) == len(projections) == 1 and not PER_GROUP.search(q):
        quantity = THRESHOLD.search(q)
        frequency = FREQUENCY.search(q) and TOP_ONE.search(q)
        if frequency or quantity and not any(isinstance(node, exp.Literal) for node in dimensions):
            variant = current.copy()
            variant.set("group", exp.Group(expressions=[dimensions[0].copy()]))
            if frequency:
                variant.set("order", exp.Order(expressions=[
                    exp.Ordered(this=exp.Count(this=exp.Star()), desc="least" not in q)]))
                variant.set("limit", exp.Limit(expression=exp.Literal.number(1)))
                if not relevant_filter(variant, q):
                    variant.set("where", None)
                yield 3.9, variant
            elif quantity:
                cls = THRESHOLD_OP.get(quantity[1], exp.EQ)
                n = NUMBER.get(quantity[2], quantity[2])
                variant.set("having", exp.Having(this=cls(this=exp.Count(this=exp.Star()),
                                                          expression=exp.Literal.number(n))))
                if not relevant_filter(variant, q):
                    variant.set("where", None)
                if not ordered and not limit:
                    variant.set("order", None)
                yield 3.35, variant
    if current.args.get("where") and re.search(r"\b(all|every)\b", q) and (
        not relevant_filter(current, q) and not re.search(r"\b(who|whose|with|without|where|having|that|which)\b", q)
    ):
        variant = current.copy()
        variant.set("where", None)
        yield 3.6, variant
    trimmed = fewer_joins(current, catalog)
    if trimmed is not None:
        yield 2.45, trimmed
    for amended in fk_variants(current, catalog):
        yield 3.05, amended


def simple_projections(catalog, question, example):
    goals = [header(label, catalog) for label in example.get("columns", [])]
    if not goals or any(goal is None for goal in goals):
        return
    wanted = Counter(compact(col.name) for goal in goals for col in goal.find_all(exp.Column))
    if not wanted:
        return
    q = question.casefold()
    if re.search(r"\b(who|whose|with|without|where|having|that|which|before|after|between|than)\b", q):
        return
    for key, cols in sorted(catalog.columns.items(), key=lambda pair: -table_relevance(pair[0], q)):
        if not set(wanted).issubset({compact(name) for name in cols.values()}):
            continue
        if not table_relevance(key, q) and len(catalog.tables) > 1:
            continue
        source = catalog.tables[key]["name"]
        tree = exp.select(*[goal.copy() for goal in goals]).from_(exp.Table(this=identifier(source)))
        for col in list(tree.find_all(exp.Column)):
            name = next(name for name in cols.values() if compact(name) == compact(col.name))
            col.replace(exp.Column(this=identifier(name)))
        yield 3.25, tree
