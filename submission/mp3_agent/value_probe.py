"""Use one bounded observation to resolve a question-grounded field spelling."""
from dataclasses import dataclass
import re

from sqlglot import exp
from sqlglot.errors import SqlglotError

from .schema import compact, identifier, parse, render
from .search import owner, physical_scope


MINUTES = re.compile(r"\b(\d{1,2})\s*[- ]?\s*(?:minutes?|min)\b")


@dataclass(frozen=True)
class Probe:
    query: str
    candidate: str
    predicate: int
    value: str
    minutes: int | None = None


def as_text(column):
    return exp.Cast(this=column.copy(), to=exp.DataType.build("TEXT"))


def lower_trim(column):
    return exp.Lower(this=exp.Trim(this=as_text(column)))


def predicate_literal(node, known):
    value = node.expression
    if isinstance(value, exp.Literal):
        return value.name, value.is_string
    if (isinstance(value, exp.Column) and isinstance(value.this, exp.Identifier)
            and value.this.quoted and not value.table and owner(value, known) is None):
        return value.name, True
    return None


def choose_probe(checked, example, catalog, question):
    q = question.casefold()
    candidates = []
    quantity = MINUTES.search(q)
    for index, candidate in enumerate(checked):
        if not candidate.feedback["ok"] or candidate.feedback["matches_example"]:
            continue
        tree = parse(candidate.sql)
        if tree is None:
            continue
        try:
            resolved = physical_scope(tree, catalog)
        except (SqlglotError, RecursionError):
            continue
        if not resolved or not tree.args.get("where"):
            continue
        bindings, known, _ = resolved
        for position, predicate in enumerate(tree.args["where"].find_all(exp.EQ)):
            col = predicate.this
            if not isinstance(col, exp.Column):
                continue
            found = owner(col, known)
            literal = predicate_literal(predicate, known)
            if not found or not literal:
                continue
            alias, name = found
            value, is_string = literal
            kind = catalog.tables[bindings[alias]]["name"]
            if quantity and compact(name) in {"duration", "length", "runningtime", "runtime", "minutes"}:
                minutes = int(quantity[1])
                if not 1 <= minutes <= 59:
                    continue
                column = exp.Column(this=identifier(name))
                formats = [exp.Like(this=as_text(column), expression=exp.Literal.string(s))
                           for s in (f"{minutes}:%", f"{minutes:02d}:%", f"00:{minutes:02d}:%")]
                condition = exp.or_(*formats)
                score = 6.0
            else:
                minutes = None
                if not is_string or len(value) < 3 or len(value) > 100 or not re.search(
                    r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)", q
                ):
                    continue
                # An empty answer does not establish an incorrect literal spelling.
                if example["row_count"] == 0:
                    continue
                column = exp.Column(this=identifier(name))
                condition = exp.EQ(this=lower_trim(column), expression=exp.Literal.string(value.strip().lower()))
                score = 2.0 if candidate.feedback["row_count"] == 0 else 1.0
            sql = exp.select(column).distinct().from_(exp.Table(this=identifier(kind))).where(condition).limit(8)
            candidates.append((score, -index, Probe(render(sql), candidate.sql, position, value, minutes)))
    return max(candidates, key=lambda item: item[:2])[-1] if candidates else None


def probe_repairs(probe, result):
    if not result.get("ok"):
        return
    original = parse(probe.candidate)
    if original is None:
        return
    values = [row[0] for row in result.get("rows", []) if len(row) == 1 and isinstance(row[0], str)
              and len(row[0]) <= 100]
    for value in dict.fromkeys(values):
        if probe.minutes is None and value != probe.value and value.strip().casefold() == probe.value.strip().casefold():
            clone = original.copy()
            eq = list(clone.args["where"].find_all(exp.EQ))[probe.predicate]
            eq.set("expression", exp.Literal.string(value))
            yield 0.15 if value.strip() == value else 0.0, clone
    if probe.minutes is None:
        return
    seen_prefixes = []
    for value in values:
        match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", value)
        if not match:
            continue
        minutes = int(match[1]) if not match[3] else int(match[1])*60 + int(match[2])
        seconds = int(match[3] or match[2])
        if minutes != probe.minutes or seconds >= 60:
            continue
        if seconds == 0:
            clone = original.copy()
            eq = list(clone.args["where"].find_all(exp.EQ))[probe.predicate]
            eq.set("expression", exp.Literal.string(value))
            yield 0.35, clone
        prefix = value.rsplit(":", 1)[0]
        if prefix not in seen_prefixes:
            seen_prefixes.append(prefix)
    if seen_prefixes:
        clone = original.copy()
        eq = list(clone.args["where"].find_all(exp.EQ))[probe.predicate]
        patterns = [exp.Like(this=as_text(eq.this), expression=exp.Literal.string(prefix + ":__"))
                    for prefix in seen_prefixes]
        eq.replace(exp.or_(*patterns))
        yield 0.2, clone
