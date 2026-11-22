"""Additional structural repairs triggered by language and schema evidence."""
import re

from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from .guided import relevant_filter, table_relevance
from .schema import apply_path, compact, field, identifier, scope_sources


NEGATION = re.compile(r"\b(?:never|without|do(?:es)? not have|don.t have|has no|have no|not living)\b")
FIRST_NAMES = {"firstname", "fname", "givenname", "forename"}
LAST_NAMES = {"lastname", "lname", "surname", "familyname"}


def conjunctions(predicate):
    if isinstance(predicate, exp.And):
        yield from conjunctions(predicate.this)
        yield from conjunctions(predicate.expression)
    else:
        yield predicate


def positive(predicate, question):
    if isinstance(predicate, exp.Not):
        return predicate.this.copy()
    if isinstance(predicate, exp.NEQ):
        return exp.EQ(this=predicate.this.copy(), expression=predicate.expression.copy())
    high = re.search(r"\b(?:higher|greater|more|longer|older|above)\b", question)
    low = re.search(r"\b(?:less|fewer|shorter|younger|lower|below)\b", question)
    if high and isinstance(predicate, (exp.LT, exp.LTE)) or low and isinstance(predicate, (exp.GT, exp.GTE)):
        cls = exp.GT if high else exp.LT
        return cls(this=predicate.this.copy(), expression=predicate.expression.copy())
    return predicate.copy()


def single_source(tree, alias, table_name):
    result = tree.copy()
    table = exp.Table(this=identifier(table_name))
    if alias.casefold() != table_name.casefold():
        table.set("alias", exp.TableAlias(this=identifier(alias)))
    result.set("from", exp.From(this=table))
    result.set("joins", [])
    return result


def local_columns(node, known):
    owners = set()
    for col in node.find_all(exp.Column):
        if col.table:
            owners.add(col.table.casefold())
        else:
            aliases = {alias.casefold() for alias, cols in known.items() if col.name.casefold() in cols}
            if len(aliases) == 1:
                owners.update(aliases)
            elif aliases:
                return set()
    return owners


def anti_edits(tree, catalog, question, scope, bindings, known):
    if tree.find(exp.Subquery) is not None:
        for slot, negation in enumerate(tree.find_all(exp.Not)):
            child = negation.this
            if not isinstance(child, exp.In):
                continue
            subquery = child.args.get("query")
            if subquery and isinstance(subquery.this, exp.Select) and subquery.this.args.get("where"):
                if not relevant_filter(subquery.this, question):
                    variant = tree.copy()
                    select = list(variant.find_all(exp.Not))[slot].this.args["query"].this
                    select.set("where", None)
                    yield 4.35, variant
        return
    where = tree.args.get("where")
    if where is None or tree.args.get("group") or tree.args.get("having"):
        return
    parents = set()
    for node in tree.expressions:
        parents.update(local_columns(node, known))
    if len(parents) == 1:
        aliases = [(alias, kind) for alias, kind in bindings.items() if alias.casefold() in parents and kind]
    elif not parents and len(tree.expressions) == 1 and tree.expressions[0].find(exp.Count):
        aliases = [(catalog.tables[name]["name"], name) for name in catalog.tables
                   if name not in bindings.values() and table_relevance(name, question) >= 0.8]
    else:
        return
    for alias, parent in aliases:
        for predicate in conjunctions(where.this):
            owners = local_columns(predicate, known)
            if len(owners) != 1:
                continue
            child_alias = next((item for item in bindings if item.casefold() in owners), None)
            child = bindings.get(child_alias)
            if not child or child == parent:
                continue
            apply_negative_pair = positive(predicate, question)
            for _, path in catalog.paths({alias: parent}, child)[:1]:
                if not all(edge.explicit for edge in path):
                    continue
                first = path[0]
                head = "anti_source"
                relation = catalog.tables[first.table]["name"]
                table = exp.Table(this=identifier(relation), alias=exp.TableAlias(this=identifier(head)))
                inner = exp.select(exp.Literal.number(1)).from_(table)
                equality = [exp.EQ(this=field(right, head), expression=field(left, alias))
                            for left, right in first.pairs]
                used = {head: first.table, alias: parent}
                target = apply_path(inner, head, path[1:], catalog, used) if len(path) > 1 else head
                term = apply_negative_pair.copy()
                for col in list(term.find_all(exp.Column)):
                    col.replace(field(col.name, target))
                inner.set("where", exp.Where(this=exp.and_(*equality, term)))
                outer = single_source(tree, alias, catalog.tables[parent]["name"])
                outer.set("where", exp.Where(this=exp.Not(this=exp.Exists(this=inner))))
                yield 4.45, outer


def name_edits(tree, known, question):
    q = question.casefold()
    filters = []
    where = tree.args.get("where")
    if not where:
        return
    for index, node in enumerate(where.find_all(exp.EQ)):
        col, lit = node.this, node.expression
        if not isinstance(col, exp.Column) or not isinstance(lit, exp.Literal) or not lit.is_string:
            continue
        key = compact(col.name)
        if key not in FIRST_NAMES | LAST_NAMES:
            continue
        for alias, columns in known.items():
            if col.table and col.table.casefold() != alias.casefold() or col.name.casefold() not in columns:
                continue
            first = next((name for name in columns.values() if compact(name) in FIRST_NAMES), None)
            last = next((name for name in columns.values() if compact(name) in LAST_NAMES), None)
            if first and last:
                filters.append((index, alias, first, last, key, lit.name))
    for index, alias, first, last, key, name in filters:
        parts = name.split()
        if len(parts) == 2 and name.casefold() in q:
            amended = tree.copy()
            original = list(amended.args["where"].find_all(exp.EQ))[index]
            given, family = parts
            original.replace(exp.and_(
                exp.EQ(this=field(first, alias), expression=exp.Literal.string(given)),
                exp.EQ(this=field(last, alias), expression=exp.Literal.string(family))
            ))
            yield 4.7, amended
    for i, alias, first, last, key, family in filters:
        if key not in LAST_NAMES:
            continue
        for j, other_alias, _, _, other_key, given in filters:
            if alias == other_alias and other_key in FIRST_NAMES and (family + " " + given).casefold() in q:
                amended = tree.copy()
                conditions = list(amended.args["where"].find_all(exp.EQ))
                conditions[i].set("expression", exp.Literal.string(given))
                conditions[j].set("expression", exp.Literal.string(family))
                yield 4.7, amended


def extra_edits(tree, catalog, question):
    if not isinstance(tree, exp.Select):
        return
    scope = traverse_scope(tree)[-1]
    bindings, known, _ = scope_sources(scope, catalog)
    yield from name_edits(tree, known, question)
    if NEGATION.search(question):
        yield from anti_edits(tree, catalog, question, scope, bindings, known)
    if not tree.args.get("joins") or tree.find(exp.Subquery) is not None:
        return
    without_joins = tree.copy()
    without_joins.set("joins", [])
    aliases = set()
    for col in without_joins.find_all(exp.Column):
        owners = local_columns(col, known)
        if not owners:
            return
        aliases.update(owners)
    if len(aliases) == 1:
        name = next(iter(aliases))
        for alias, table in bindings.items():
            if table and alias.casefold() == name:
                if not table_relevance(table, question) and len(bindings) > 2:
                    continue
                yield 3.5, single_source(tree, alias, catalog.tables[table]["name"])
