"""Lexically link question words and output fields to schema tables."""
from collections import Counter
import math
import re

from sqlglot import exp

from .guided import header
from .schema import compact


COMMON = {"id", "name", "number", "type", "code", "the", "for", "of", "a", "in",
          "to", "on", "what", "which", "show", "list", "find", "count", "how", "many"}


def terms(text):
    def root(word):
        word = word.casefold()
        if len(word) >= 4 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) >= 4 and not word.endswith("ss"):
            return word.rstrip("s")
        return word

    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return {token for word in re.findall(r"[A-Za-z0-9]+", words) if (token := root(word)) not in COMMON}


def order_schema(schema, question, labels, referenced):
    if not schema:
        return []
    requested = set()
    for label in labels:
        tree = header(label)
        if tree is not None:
            requested.update(compact(col.name) for col in tree.find_all(exp.Column))
    q = terms(question)
    statistics = {}
    document_frequency = Counter()
    owners = Counter()
    for record in schema:
        name = record["name"].casefold()
        table_words = terms(record["name"])
        column_words = set().union(*(terms(column["name"]) for column in record["columns"]))
        exact_names = {compact(column["name"]) for column in record["columns"]}
        statistics[name] = table_words, column_words, exact_names
        document_frequency.update(table_words | column_words)
        owners.update(exact_names)
    scores = {}
    for record in schema:
        name = record["name"].casefold()
        table_words, column_words, exact_names = statistics[name]
        lexical = sum(math.log1p(len(schema) / document_frequency[word]) for word in sorted(q & table_words))
        columns = sum(math.log1p(len(schema) / document_frequency[word]) for word in sorted(q & column_words))
        expected = sum(1 / owners[key] for key in sorted(requested & exact_names))
        scores[name] = lexical + 0.32 * columns + expected + (0.7 if name in referenced else 0.0)
    neighbors = {key: 0.0 for key in scores}
    for record in schema:
        source = record["name"].casefold()
        for fk in record.get("foreign_keys", []):
            target = fk["table"].casefold()
            if target in scores:
                neighbors[source] = max(neighbors[source], 0.17 * scores[target])
                neighbors[target] = max(neighbors[target], 0.17 * scores[source])
    return sorted(schema, key=lambda record: -sum((scores[record["name"].casefold()],
                                                   neighbors[record["name"].casefold()])))
