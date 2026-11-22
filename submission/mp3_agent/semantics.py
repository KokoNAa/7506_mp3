"""Scope-aware candidate edits for names, thresholds, and comparisons."""
import re

from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from .guided import projection, header, table_relevance
from .schema import compact, field, identifier, scope_sources
from .structural import name_edits


MEAN_COMPARISON = re.compile(
    r"\b(older|younger|higher|lower|greater|smaller|larger|more|less)\s+than\s+(?:the\s+)?(?:average|mean)\b"
)
TIME_QUANTIFIER = re.compile(r"\b(after|before)\s+(any|every|all)\b")
MINUTES = re.compile(r"\b(\d+)\s*[- ]?\s*(?:minute|minutes|min)\b")
COMPLEX_FILTER = re.compile(r"\b(and|or|but|including|excluding|except|between)\b")


def scope_name_edits(tree, catalog, question):
    for index, scope in enumerate(traverse_scope(tree)):
        if not isinstance(scope.expression, exp.Select):
            continue
        _, known, _ = scope_sources(scope, catalog)
        select = scope.expression.copy()
        where = select.args.get("where")
        if not where:
            continue
        for eq in where.find_all(exp.EQ):
            token = eq.expression
            if (isinstance(token, exp.Column) and not token.table and token.this.quoted
                    and not any(token.name.casefold() in cols for cols in known.values())):
                eq.set("expression", exp.Literal.string(token.name))
        for weight, repaired in name_edits(select, known, question):
            other = tree.copy()
            region = traverse_scope(other)[index].expression
            if region is other:
                yield weight + 0.4, repaired
            else:
                region.replace(repaired)
                yield weight + 0.4, other


def unit_variants(tree, question):
    quantity = MINUTES.search(question)
    if not quantity:
        return
    mins = int(quantity[1])
    if mins >= 60:
        return
    for index, node in enumerate(tree.find_all(exp.EQ)):
        col, lit = node.this, node.expression
        if not isinstance(col, exp.Column) or not any(name in compact(col.name)
                for name in ("duration", "length", "runtime", "minutes")):
            continue
        if not isinstance(lit, (exp.Literal, exp.Column)) or lit.name.strip("0") != str(mins).strip("0"):
            continue
        copy = tree.copy()
        target = list(copy.find_all(exp.EQ))[index]
        formatted = [str(mins), f"{mins}:00", f"{mins:02d}:00", f"00:{mins:02d}:00",
                     f"{mins} min", f"{mins} minutes"]
        variant = exp.In(this=exp.Lower(this=exp.Trim(this=target.this.copy())),
                         expressions=[exp.Literal.string(value) for value in dict.fromkeys(formatted)])
        target.replace(variant)
        yield 4.25, copy


def mean_variants(tree, catalog, question, example):
    cue = MEAN_COMPARISON.search(question)
    if not cue or COMPLEX_FILTER.search(question):
        return
    groups = tree.args.get("group")
    where = tree.args.get("where")
    if where or tree.find(exp.Avg) is not None:
        return
    scope = traverse_scope(tree)[-1]
    bindings, _, _ = scope_sources(scope, catalog)
    rows = []
    for alias, table in bindings.items():
        if table is None:
            continue
        for col in catalog.tables[table]["columns"]:
            name = col["name"]
            col_key = compact(name)
            if cue[1] in ("older", "younger"):
                score = float(col_key in ("age", "ageyears", "yearsold"))
            else:
                score = table_relevance(name, question)
            if score > 0 and (col_key == "age" or re.search(r"int|float|real|double|numeric|decimal",
                                                           col["type"], flags=re.I)):
                rows.append((score, alias, table, name))
    planned = sorted(rows, reverse=True)[:2]
    for weight, alias, table, name in planned:
        clone = tree.copy()
        field_name = catalog.tables[table]["name"]
        average = exp.select(exp.Avg(this=exp.Column(this=identifier(name)))).from_(
            exp.Table(this=identifier(field_name))
        )
        cls = exp.GT if cue[1] in ("older", "higher", "greater", "larger", "more") else exp.LT
        clone.set("where", exp.Where(this=cls(this=field(name, alias), expression=exp.Subquery(this=average))))
        if groups:
            goals = [header(label, catalog) for label in example["columns"]]
            if goals and all(goal is not None for goal in goals):
                projected = projection(clone, catalog, goals, question)
                if projected is not None:
                    clone = projected
        yield 4.75 + 0.25 * weight, clone


def time_variants(tree, question):
    cue = TIME_QUANTIFIER.search(question)
    if not cue:
        return
    after = cue[1] == "after"
    any_event = cue[2] == "any"
    wanted = exp.Min if after == any_event else exp.Max
    for index, query in enumerate(tree.find_all(exp.Subquery)):
        body = query.this
        if not isinstance(body, exp.Select) or len(body.expressions) != 1:
            continue
        function = body.expressions[0]
        if not isinstance(function, (exp.Min, exp.Max)) or isinstance(function, wanted):
            continue
        amended = tree.copy()
        holder = list(amended.find_all(exp.Subquery))[index].this
        original = holder.expressions[0]
        original.replace(wanted(this=original.this.copy()))
        yield 5.0, amended


def semantic_repairs(tree, catalog, question, example):
    if not isinstance(tree, exp.Select):
        return
    q = question.casefold()
    yield from scope_name_edits(tree, catalog, q)
    yield from unit_variants(tree, q)
    yield from mean_variants(tree, catalog, q, example)
    yield from time_variants(tree, q)
    yield from rebuild_joins(tree, catalog, q)


def rebuild_joins(tree, catalog, question):
    """Build a connected FK path for a simple SELECT's referenced fields."""
    from .schema import apply_path, render

    if not tree.args.get("joins") or len(traverse_scope(tree)) != 1 or tree.find(exp.Star):
        return
    if any(join.args.get("side") or not isinstance(join.this, exp.Table)
           for join in tree.args["joins"]):
        return
    for join in tree.args["joins"]:
        on = join.args.get("on")
        if not on or any(not isinstance(part, (exp.EQ, exp.And, exp.Column, exp.Identifier))
                         for part in on.walk()):
            return
    original_scope = traverse_scope(tree)[0]
    original, known, _ = scope_sources(original_scope, catalog)
    candidate = tree.copy()
    candidate.set("joins", [])
    projection_names = {node.alias.casefold() for node in tree.expressions if node.alias}
    anchor = tree.args.get("from")
    if not anchor or not isinstance(anchor.this, exp.Table):
        return
    required = {}
    bindings = []
    for node in list(candidate.find_all(exp.Column)):
        name = node.name.casefold()
        if node.table:
            alias = next((key for key in known if key.casefold() == node.table.casefold()), None)
            target = original.get(alias)
            if target and name in catalog.columns[target]:
                required[target] = alias
                bindings.append((node, target, catalog.columns[target][name]))
                continue
        else:
            if name in projection_names:
                continue
            containing = [(key, original.get(key)) for key, cols in known.items() if name in cols]
            if len(containing) == 1 and containing[0][1]:
                alias, target = containing[0]
                required[target] = alias
                bindings.append((node, target, catalog.columns[target][name]))
                continue
            if isinstance(node.this, exp.Identifier) and node.this.quoted and not containing:
                # An unresolved quoted token may be a SQLite string literal.
                continue
        targets = [target for target, cols in catalog.columns.items() if name in cols]
        if len(targets) != 1:
            return
        target = targets[0]
        required[target] = catalog.tables[target]["name"]
        bindings.append((node, target, catalog.columns[target][name]))
    if len(required) < 2 or len(required) > 4 or len(original) != len(set(original.values())):
        return
    head = anchor.this.name.casefold()
    if head not in required:
        head = max(required, key=lambda name: table_relevance(name, question))
    first_name = catalog.tables[head]["name"]
    head_alias = required[head]
    root = exp.Table(this=identifier(first_name))
    if first_name.casefold() != head_alias.casefold():
        root.set("alias", exp.TableAlias(this=identifier(head_alias)))
    candidate.set("from", exp.From(this=root))
    aliases = {head_alias: head}
    actual = {head: head_alias}
    while any(target not in actual for target in required):
        paths = [(alias, route) for target in required if target not in actual
                 for alias, route in catalog.paths(aliases, target)]
        if not paths:
            return
        start, best = min(paths, key=lambda item: (sum(not edge.explicit for edge in item[1]), len(item[1])))
        if any(not edge.explicit for edge in best) and len(best) > 1:
            return
        for edge in best:
            start = apply_path(candidate, start, [edge], catalog, aliases)
            aliases[start] = edge.table
            actual[edge.table] = start
    for node, table, name in bindings:
        node.replace(field(name, actual[table]))
    if render(candidate) != render(tree):
        yield 3.8, candidate
