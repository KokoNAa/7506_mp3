"""Conservative equivalence and structure-aware candidate construction."""
import re

from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import traverse_scope

from .guided import AGGS, NUMBER, header, projection, shape_agreement
from .linking import terms
from .schema import apply_path, compact, field, identifier, render, scope_sources
from .structural import NEGATION


TOP = re.compile(r"\b(top|bottom)\s+(\d+|" + "|".join(NUMBER) + r")\b")
FREQUENCY = re.compile(
    r"\b((largest|highest|greatest|smallest|lowest) number of|most (common|frequent)|least (common|frequent))\b"
)
GROUP_PHRASE = re.compile(r"\b(?:for each|per|grouped by|for every)\s+([a-z0-9_ -]{2,55})")
CONNECTIVE = re.compile(r"\b(?:and|or|with|which|that|whose|having|show|sorted|ordered)\b")
SPECIFIC_NEGATION = re.compile(r"\b(?:never|did not|does not|do not)\b")
ENTITY_BOUNDARY = re.compile(r"\b(?:who|whose|that|which|with|without|where|belonging|associated)\b")
PERSON_ROLES = {"employee", "worker", "staff", "member", "player", "student"}
PEOPLE = {"person", "people", "individual"}


def physical_scope(tree, catalog):
    """Resolve one flat SELECT over named tables, without rewriting nested scopes."""
    if not isinstance(tree, exp.Select) or list(tree.find_all(exp.Select)) != [tree]:
        return None
    scopes = traverse_scope(tree)
    if len(scopes) != 1:
        return None
    scope = scopes[0]
    if any(join.args.get("using") or join.args.get("method") == "NATURAL"
           for join in tree.args.get("joins", [])):
        return None
    bindings, known, tables = scope_sources(scope, catalog)
    if not bindings or not all(bindings.values()) or len(bindings) != len(tables):
        return None
    return bindings, known, tables


def owner(col, known):
    name = col.name.casefold()
    possible = [(alias, names[name]) for alias, names in known.items() if name in names
                and (not col.table or alias.casefold() == col.table.casefold())]
    return possible[0] if len(possible) == 1 else None


def equivalent_key(tree, catalog):
    """Normalize resolved identifiers, preserving all joins, predicates and ordering."""
    raw = render(tree)
    if tree.find(exp.Star) is not None and not all(isinstance(star.parent, exp.Count)
                                                  for star in tree.find_all(exp.Star)):
        return raw
    normalized = tree.copy()
    try:
        resolved = physical_scope(normalized, catalog)
    except (SqlglotError, RecursionError):
        return raw
    if not resolved:
        return raw
    bindings, known, tables = resolved
    if not all(x.isascii() for x in (*bindings, *bindings.values())):
        return raw
    aliases = {name: f"source_{idx}" for idx, name in enumerate(tables)}
    references = {col.name.casefold() for col in normalized.find_all(exp.Column)}
    output_aliases = {item.alias.casefold() for item in normalized.expressions if isinstance(item, exp.Alias)}
    if any(not col.table and col.name.casefold() in output_aliases for col in normalized.find_all(exp.Column)):
        return raw
    for item in list(normalized.expressions):
        if isinstance(item, exp.Alias) and item.alias.casefold() not in references:
            item.replace(item.this.copy())
    for column in list(normalized.find_all(exp.Column)):
        found = owner(column, known)
        if found is None or not found[1].isascii():
            # Includes SQL aliases used by ORDER BY and SQLite's double-quoted strings.
            return raw
        alias, name = found
        column.replace(field(name.casefold(), aliases[alias]))
    for alias, table in tables.items():
        table.set("this", identifier(bindings[alias]))
        table.set("alias", exp.TableAlias(this=identifier(aliases[alias])))
    return "resolved:" + render(normalized)


def focus_groups(tree, catalog, question):
    match = GROUP_PHRASE.search(question)
    if not match or not (tree.args.get("having") or tree.args.get("group")):
        return
    query_terms = terms(CONNECTIVE.split(match[1], maxsplit=1)[0])
    if not query_terms:
        return
    resolved = physical_scope(tree, catalog)
    if not resolved:
        return
    bindings, known, _ = resolved
    scored = []
    for alias, names in known.items():
        tterms = terms(bindings[alias])
        for name in names.values():
            cterms = terms(name)
            if cterms & query_terms:
                score = 2 * len(cterms & query_terms) - 0.25 * len(cterms - query_terms)
                if compact(name).endswith("id"):
                    score += 0.15
                if tterms & query_terms:
                    score += 0.1
                scored.append((score, alias, name))
    current = {(col.table.casefold(), compact(col.name))
               for col in tree.args.get("group", exp.Group()).find_all(exp.Column)}
    for _, alias, name in sorted(scored, key=lambda row: -row[0])[:2]:
        if (alias.casefold(), compact(name)) in current or ("", compact(name)) in current:
            continue
        amended = tree.copy()
        amended.set("group", exp.Group(expressions=[field(name, alias)]))
        yield 4.35, amended


def ranked_groups(tree, question, demanded):
    """Supply a count-based ordering for frequency questions."""
    cue, top = FREQUENCY.search(question), TOP.search(question)
    if not cue or len(demanded) != 1 or not isinstance(demanded[0], exp.Column):
        return
    choice = tree.expressions[0]
    column = choice.this if isinstance(choice, exp.Alias) else choice
    if not isinstance(column, exp.Column):
        return
    if top:
        limit = NUMBER[top[2]] if top[2] in NUMBER else int(top[2])
    else:
        limit = 1
    if not 1 <= limit <= 100:
        return
    down = not (top and top[1] == "bottom") and not any(x in cue[1] for x in ("smallest", "lowest", "least"))
    alternate = tree.copy()
    alternate.set("group", exp.Group(expressions=[column.copy()]))
    alternate.set("order", exp.Order(expressions=[
        exp.Ordered(this=exp.Count(this=exp.Star()), desc=down, nulls_first=not down)
    ]))
    alternate.set("limit", exp.Limit(expression=exp.Literal.number(limit)))
    alternate.set("distinct", None)
    yield 4.7, alternate


def same_relation_exclusion(tree, question):
    where = tree.args.get("where")
    if (not SPECIFIC_NEGATION.search(question) or not NEGATION.search(question)
            or tree.args.get("joins") or tree.args.get("group") or tree.args.get("order")
            or tree.args.get("limit") or where is None or where.find(exp.Subquery)
            or any(isinstance(expr, AGGS) for expr in tree.expressions)):
        return
    positive = where.this
    if not isinstance(positive, (exp.EQ, exp.In, exp.Like)):
        return
    left = tree.copy()
    left.set("where", None)
    yield 4.8, exp.Except(this=left, expression=tree.copy(), distinct=True)


def qualified_where(source, known, catalog, question):
    where = source.args.get("where")
    if where is None or where.find(exp.Subquery) is not None:
        return None
    if any(isinstance(node, (exp.Select, exp.Anonymous, exp.AggFunc)) for node in where.walk()):
        return None
    bound = where.copy()
    for col in list(bound.find_all(exp.Column)):
        if owner(col, known) is not None:
            continue
        if isinstance(col.this, exp.Identifier) and col.this.quoted and not col.table:
            # SQLite accepts strings in double quotes when the identifier is unresolved.
            if col.name.casefold() in question and not any(col.name.casefold() in values
                                                           for values in catalog.columns.values()):
                col.replace(exp.Literal.string(col.name))
                continue
        return None
    if any((lit.is_string and lit.name.strip("%") and lit.name.casefold().strip("%") in question)
           or (not lit.is_string and re.search(r"(?<![\w.])" + re.escape(lit.name) + r"(?![\w.])", question))
           for lit in bound.find_all(exp.Literal)):
        return bound
    return None


def transfer_filter(target, donor, catalog, question):
    """Attach a grounded predicate from another seed using declared FK paths."""
    left = physical_scope(target, catalog)
    right = physical_scope(donor, catalog)
    if not left or not right or target == donor:
        return None
    destination, _, _ = left
    bindings, known, _ = right
    if len(set(destination.values())) != len(destination) or len(set(bindings.values())) != len(bindings):
        return None
    where = qualified_where(donor, known, catalog, question)
    if where is None:
        return None
    output = target.copy()
    target_tables = {table: alias for alias, table in destination.items()}
    assembled = dict(destination)
    for col in list(where.find_all(exp.Column)):
        donor_alias, col_name = owner(col, known)
        table = bindings[donor_alias]
        if table not in target_tables:
            routes = [(anchor, route) for anchor, route in catalog.paths(assembled, table)
                      if all(x.explicit for x in route)]
            if not routes:
                return None
            anchor, route = routes[0]
            for edge in route:
                if edge.table in target_tables:
                    return None
                anchor = apply_path(output, anchor, [edge], catalog, assembled)
                target_tables[edge.table] = anchor
                assembled[anchor] = edge.table
        col.replace(field(col_name, target_tables[table]))
    output.set("where", where)
    if render(output) != render(target):
        return output
    return None


def entity_score(table, field_name, role):
    table_words = terms(table)
    column_words = terms(field_name)
    if role & PERSON_ROLES and table_words & PEOPLE:
        return 1.4
    return (1.6 * len(table_words & role) + 0.30 * len(column_words & role)
            - 0.08 * len(table_words - role))


def entity_outputs(tree, catalog, question):
    """Try the entity asked for when a shared output name has several owners."""
    prefix = ENTITY_BOUNDARY.split(question, maxsplit=1)
    if len(prefix) < 2:
        return
    role = terms(prefix[0])
    resolved = physical_scope(tree, catalog)
    if not role or not resolved:
        return
    bindings, known, _ = resolved
    if len(set(bindings.values())) != len(bindings):
        return
    for position, selected in enumerate(tree.expressions):
        column = selected.this if isinstance(selected, exp.Alias) else selected
        if not isinstance(column, exp.Column):
            continue
        found = owner(column, known)
        if not found:
            continue
        old_alias, old_name = found
        old_table = bindings[old_alias]
        previous = entity_score(old_table, old_name, role)
        choices = []
        for key, columns in catalog.columns.items():
            same = columns.get(column.name.casefold())
            if not same or key == old_table:
                continue
            evidence = entity_score(key, same, role)
            if evidence <= previous + 0.7:
                continue
            already = next((alias for alias, name in bindings.items() if name == key), None)
            if already:
                choices.append((evidence, already, (), same))
                continue
            for anchor, route in catalog.paths(bindings, key):
                if all(edge.explicit for edge in route):
                    choices.append((evidence - 0.15 * len(route), anchor, route, same))
        for _, anchor, route, name in sorted(choices, key=lambda item: -item[0])[:1]:
            alternate = tree.copy()
            alias = apply_path(alternate, anchor, route, catalog, bindings) if route else anchor
            selected = alternate.expressions[position]
            if isinstance(selected, exp.Alias):
                selected.set("this", field(name, alias))
            else:
                selected.replace(field(name, alias))
            yield 4.55, alternate


def relation_counts(tree, catalog, question):
    if (len(tree.expressions) != 1 or not isinstance(tree.expressions[0], exp.Count)
            or not isinstance(tree.expressions[0].this, exp.Star)
            or not tree.args.get("joins") or any(tree.args.get(x) for x in ("group", "having", "where", "limit"))):
        return
    if not re.search(r"\b(?:how many|count|number of)\b", question):
        return
    resolved = physical_scope(tree, catalog)
    if not resolved:
        return
    bindings, _, _ = resolved
    role = terms(ENTITY_BOUNDARY.split(question, maxsplit=1)[0])
    eligible = []
    for table in bindings.values():
        strength = entity_score(table, "", role)
        if strength > 0:
            eligible.append((strength, table))
    eligible.sort(key=lambda pair: -pair[0])
    for strength, table in eligible[:1]:
        # Only drop joins when every joined relation is a declared FK parent
        # of the counted row. Also retain the original join as another candidate.
        neighbors = {edge.table for edge in catalog.edges[table] if edge.explicit and all(
            right.casefold() in {col["name"].casefold() for col in catalog.tables[edge.table]["columns"]
                                 if col.get("pk")}
            for _, right in edge.pairs
        )}
        if set(bindings.values()) - {table} <= neighbors:
            yield 4.2, exp.select(exp.Count(this=exp.Star())).from_(
                exp.Table(this=identifier(catalog.tables[table]["name"]))
            )


def structural_search(tree, donors, catalog, question, example):
    """Compose projections, groups, and grounded filters into candidate SELECTs."""
    if not isinstance(tree, exp.Select):
        return
    q = question.casefold()
    goals = [header(label, catalog) for label in example["columns"]]
    if not goals or any(goal is None for goal in goals):
        return
    amended = projection(tree, catalog, goals, q)
    if amended is None or shape_agreement(amended, example) < 2:
        return
    yield from focus_groups(amended, catalog, q)
    yield from ranked_groups(amended, q, goals)
    yield from same_relation_exclusion(amended, q)
    yield from entity_outputs(amended, catalog, q)
    yield from relation_counts(amended, catalog, q)
    # Preserve a strong existing filter. Graft a separate checked or cached filter
    # when the current clause has no question-grounded literals.
    resolved = physical_scope(amended, catalog)
    if not resolved or qualified_where(amended, resolved[1], catalog, q) is not None:
        return
    for donor in donors[:3]:
        if donor is None:
            continue
        merged = transfer_filter(amended, donor, catalog, q)
        if merged is None:
            continue
        yield 4.45, merged
        yield from ranked_groups(merged, q, goals)
