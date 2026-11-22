"""Generate bounded SQLite AST repairs using the schema returned by the tool."""
from collections import defaultdict, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, SqlglotError, TokenError
from sqlglot.optimizer.scope import Scope, traverse_scope


def parse(sql):
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (ParseError, TokenError, RecursionError):
        return None
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.SetOperation)):
        return None
    tree = statements[0]
    return tree if not any(isinstance(node, (exp.DDL, exp.DML, exp.Command, exp.Into))
                           for node in tree.walk()) else None


def render(tree):
    return tree.sql(dialect="sqlite", comments=False)


def compact(name):
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def similarity(a, b):
    a, b = compact(a), compact(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a.rstrip("s") == b.rstrip("s"):
        return 0.95
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def identifier(name):
    return exp.to_identifier(name, quoted=True)


def field(name, alias):
    return exp.Column(this=identifier(name), table=identifier(alias))


@dataclass(frozen=True)
class Edge:
    table: str
    pairs: tuple
    explicit: bool


class Catalog:
    def __init__(self, tables):
        self.tables = {table["name"].casefold(): table for table in tables}
        self.columns = {
            name: {column["name"].casefold(): column["name"] for column in table["columns"]}
            for name, table in self.tables.items()
        }
        self.edges = defaultdict(list)
        explicit = defaultdict(list)
        for table, record in self.tables.items():
            for fk in record.get("foreign_keys", []):
                target = fk["table"].casefold()
                source_col = self.columns[table].get((fk.get("from") or "").casefold())
                target_col = self.columns.get(target, {}).get((fk.get("to") or "").casefold())
                if source_col and target_col and table != target:
                    explicit[table, target].append((source_col, target_col))
        for (source, target), pairs in explicit.items():
            pairs = tuple(dict.fromkeys(pairs))
            target_keys = {col["name"] for col in self.tables[target]["columns"] if col.get("pk")}
            referenced = {right for _, right in pairs}
            if len(pairs) > 1 and len(pairs) == len(referenced) and referenced == target_keys:
                self.add_edge(source, target, pairs, True)
            else:
                for pair in pairs:
                    self.add_edge(source, target, (pair,), True)

        # A uniquely keyed, non-generic shared name is a fallback for schemas
        # that omit FK metadata. Avoid joins on shared "id" or "name" alone.
        primary = {}
        for table, record in self.tables.items():
            keys = [col["name"] for col in record["columns"] if col.get("pk")]
            if len(keys) == 1:
                primary[table] = keys[0]
        for target, key in primary.items():
            normalized = compact(key)
            if normalized in ("id", "name", "code", "key", "number", "no"):
                continue
            for source, cols in self.columns.items():
                if source == target or any(edge.table == target for edge in self.edges[source]):
                    continue
                matches = [name for name in cols.values() if compact(name) == normalized]
                if len(matches) == 1:
                    self.add_edge(source, target, ((matches[0], key),), False)

    def add_edge(self, source, target, pairs, explicit):
        self.edges[source].append(Edge(target, pairs, explicit))
        self.edges[target].append(Edge(source, tuple((right, left) for left, right in pairs), explicit))

    def paths(self, bindings, target):
        queue = deque((alias, table, (), {table}) for alias, table in bindings.items() if table)
        found = []
        while queue:
            alias, key, route, visited = queue.popleft()
            if key == target and route:
                found.append((alias, route))
            elif len(route) < 2:
                for edge in self.edges[key]:
                    if edge.table not in visited and (edge.table not in bindings.values() or edge.table == target):
                        queue.append((alias, edge.table, (*route, edge), visited | {edge.table}))
        return sorted(found, key=lambda item: (
            len(item[1]), sum(not edge.explicit for edge in item[1]), item[0],
            tuple(edge.table for edge in item[1])
        ))[:2]


def scope_sources(scope, catalog):
    bindings, known, tables = {}, {}, {}
    for alias, (node, source) in scope.selected_sources.items():
        if isinstance(source, exp.Table):
            table = source.name.casefold()
            bindings[alias] = table if table in catalog.tables else None
            known[alias] = catalog.columns.get(table, {})
            tables[alias] = node
        elif isinstance(source, Scope):
            bindings[alias] = None
            known[alias] = {name.casefold(): name for name in source.expression.named_selects}
    return bindings, known, tables


def apply_path(select, start, route, catalog, bindings):
    used = {alias.casefold() for alias in bindings}
    alias = start
    for edge in route:
        name = catalog.tables[edge.table]["name"]
        target = name
        while target.casefold() in used:
            target = f"repair_{len(used)}_{target}"
        table = exp.Table(this=identifier(name))
        if target != name:
            table.set("alias", exp.TableAlias(this=identifier(target)))
        predicates = [exp.EQ(this=field(left, alias), expression=field(right, target))
                      for left, right in edge.pairs]
        select.append("joins", exp.Join(this=table, on=exp.and_(*predicates)))
        alias = target
        used.add(alias.casefold())
    return alias


def issues(tree, catalog, error):
    """Find unresolved names in their SQL scope, preferring the engine's location."""
    message = error.casefold()
    ambiguous = "ambiguous column name:" in message
    missing = message.split(":", 1)[-1].strip()
    found = []
    for scope_index, scope in enumerate(traverse_scope(tree)):
        bindings, known, tables = scope_sources(scope, catalog)
        for alias, table in tables.items():
            if bindings[alias] is None:
                found.append((2, scope_index, "table", alias))
        for index, column in enumerate(scope.columns):
            if isinstance(column.this, exp.Star):
                continue
            name, qual = column.name.casefold(), column.table.casefold()
            qualified = next((cols for alias, cols in known.items() if alias.casefold() == qual), None)
            candidates = sum(name in cols for cols in known.values())
            hit = missing in {name, f"{qual}.{name}"} if qual else missing == name
            if isinstance(column.this, exp.Identifier) and column.this.quoted and not qual and not hit:
                # SQLite can interpret an unresolved double-quoted token as text.
                continue
            if any("*" in cols for cols in known.values()) and not hit:
                continue
            if (qual and (qualified is None or name not in qualified)
                    or not qual and (candidates == 0 or ambiguous and hit and candidates > 1)):
                found.append((0 if hit else 3, scope_index, "column", index))
    return sorted(found)


def repairs_once(tree, catalog, error):
    problems = issues(tree, catalog, error)
    if not problems:
        return []
    _, scope_index, kind, position = problems[0]
    scope = traverse_scope(tree)[scope_index]
    bindings, known, tables = scope_sources(scope, catalog)
    options = []
    if kind == "table":
        name = tables[position].name
        for table, record in catalog.tables.items():
            match = similarity(name, table)
            if match >= 0.70:
                options.append((match, "table", record["name"], None))
    else:
        column = scope.columns[position]
        name, qualifier = column.name, column.table
        for alias, cols in known.items():
            for field_name in cols.values():
                match = similarity(name, field_name)
                if match >= (0.69 if len(compact(name)) >= 4 else 0.84):
                    qualified = bool(qualifier) and qualifier.casefold() == alias.casefold()
                    # Exact spelling should usually beat a fuzzy spelling on another table.
                    options.append((match + (0.07 if qualified else 0.0), "column", alias, field_name))
        if not any(similarity(name, option[3]) >= 0.99 for option in options) and isinstance(scope.expression, exp.Select):
            for candidate, cols in catalog.columns.items():
                if candidate in bindings.values():
                    continue
                for field_name in cols.values():
                    match = similarity(name, field_name)
                    if match < 0.91:
                        continue
                    for alias, path in catalog.paths(bindings, candidate):
                        weight = match + (0.09 if qualifier.casefold() == candidate else 0)
                        weight += 0.06 if all(edge.explicit for edge in path) else -0.08
                        options.append((weight - 0.05 * len(path), "join", alias, (field_name, path)))

    variants = []
    for weight, action, alias, value in sorted(options, key=lambda item: -item[0])[:4]:
        alternative = tree.copy()
        target = traverse_scope(alternative)[scope_index]
        if action == "table":
            source = scope_sources(target, catalog)[2][position]
            if not source.alias and any(col.table.casefold() == position.casefold() for col in target.columns):
                source.set("alias", exp.TableAlias(this=identifier(position)))
            source.set("this", identifier(alias))
        else:
            old = target.columns[position]
            if action == "join":
                value, path = value
                alias = apply_path(target.expression, alias, path, catalog, bindings)
            old_name, old_table = old.name.casefold(), old.table.casefold()
            for col in target.columns:
                if col.name.casefold() == old_name and col.table.casefold() == old_table:
                    col.replace(field(value, alias))
        variants.append((weight, alternative))
    return variants


def schema_repairs(tree, catalog, error=""):
    if not catalog.tables:
        return []
    try:
        first = repairs_once(tree, catalog, error)
        output = []
        for weight, proposed in first:
            remaining = issues(proposed, catalog, "")
            if not remaining:
                output.append((weight, proposed))
                continue
            for quality, next_tree in repairs_once(proposed, catalog, "")[:2]:
                output.append((min(weight, quality) - 0.04, next_tree))
            output.append((weight - 0.18, proposed))
        return sorted(output, key=lambda item: -item[0])[:5]
    except (SqlglotError, RecursionError):
        return []


def candidate_key(tree, catalog):
    """Deduplicate equivalent spelling in simple single-table queries."""
    if not isinstance(tree, exp.Select) or tree.args.get("joins") or tree.args.get("with"):
        return render(tree)
    if tree.find(exp.Subquery) is not None or len(list(tree.find_all(exp.Select))) != 1:
        return render(tree)
    source = tree.args.get("from")
    if source is None or not isinstance(source.this, exp.Table):
        return render(tree)
    key = source.this.name.casefold()
    if key not in catalog.columns:
        return render(tree)
    alias = source.this.alias_or_name.casefold()
    other = tree.copy()
    other.set("from", exp.From(this=exp.Table(this=identifier(key))))
    columns = catalog.columns[key]
    referenced = {col.name.casefold() for col in other.find_all(exp.Column)}
    for slot, node in enumerate(other.expressions):
        if isinstance(node, exp.Alias) and node.alias.casefold() not in referenced:
            other.expressions[slot] = node.this
    for col in other.find_all(exp.Column):
        name = col.name.casefold()
        if name in columns and (not col.table or col.table.casefold() == alias):
            col.set("this", identifier(name))
            col.set("table", None)
        elif col.table:
            # Keep unresolved qualifiers distinct so a valid query is never
            # collapsed into an unresolved one.
            return render(tree)
    return "single:" + render(other)
