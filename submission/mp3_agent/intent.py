"""Small question-conditioned SQL edits and deterministic ranking signals."""
import re

from sqlglot import exp


COUNT = re.compile(r"\b(how many|number of|count of|count the)\b")
DISTINCT = re.compile(r"\b(distinct|unique|different|without duplicates)\b")
DESC = re.compile(r"\b(descending|highest to lowest|largest to smallest|most to least|latest|newest|top \d+|highest|largest|greatest)\b")
ASC = re.compile(r"\b(ascending|lowest to highest|smallest to largest|least to most|earliest|oldest|bottom \d+|lowest|smallest)\b")
AGGREGATES = [(COUNT, exp.Count),
              (re.compile(r"\b(avg|average|mean)\b"), exp.Avg),
              (re.compile(r"\b(sum(?: of)?|total amount of)\b"), exp.Sum)]
COMPARATORS = [
    (r"(?:at least|no fewer than|no less than|minimum of)\s+", exp.GTE),
    (r"(?:at most|no more than|maximum of|up to)\s+", exp.LTE),
    (r"(?:greater than|more than|over|above)\s+", exp.GT),
    (r"(?:less than|fewer than|under|below)\s+", exp.LT),
]
INVERT = {exp.GTE: exp.LTE, exp.LTE: exp.GTE, exp.GT: exp.LT, exp.LT: exp.GT}


def comparison_intent(question):
    for pattern, comparison in COMPARATORS:
        for match in re.finditer(r"\b" + pattern + r"(-?\d+(?:\.\d+)?)\b", question):
            # Prevent 'more than' matching inside 'no more than'.
            if not question[:match.start()].endswith("no "):
                yield match[1], comparison


def descending(question):
    if DESC.search(question):
        return True
    if ASC.search(question):
        return False
    return None


def projections(tree):
    return [node.this if isinstance(node, exp.Alias) else node for node in tree.selects]


def is_count_distinct(node):
    return isinstance(node, exp.Count) and isinstance(node.this, exp.Distinct)


def intent_score(tree, question, ordered):
    if tree is None:
        return -20.0
    q = question.casefold()
    projected = projections(tree)
    score = 0.0
    groups = tree.args.get("group")
    for pattern, cls in AGGREGATES:
        if not pattern.search(q):
            continue
        if any(isinstance(node, cls) for node in projected):
            score += 1.8
        elif len(projected) == 1 and not groups:
            score -= 1.25
    if DISTINCT.search(q):
        if tree.args.get("distinct") or any(map(is_count_distinct, projected)):
            score += 1.4
        elif not groups:
            score -= 0.55
    order = tree.args.get("order")
    if ordered:
        score += 0.75 if order else -1.5
    direction = descending(q)
    if direction is not None and order and order.expressions:
        score += 0.65 if bool(order.expressions[0].args.get("desc")) == direction else -0.8
    explicit_limit = re.search(r"\b(?:top|bottom|first|last)\s+(\d+)\b", q)
    limit = tree.args.get("limit")
    if explicit_limit:
        score += 1.25 if limit and limit.expression.name == explicit_limit[1] else -0.7
    for value, cls in comparison_intent(q):
        nodes = [node for node in tree.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ)
                 if any(isinstance(arg, exp.Literal) and not arg.is_string and arg.name == value
                        for arg in (node.this, node.expression))]
        if nodes:
            correct = any(isinstance(node, cls if isinstance(node.expression, exp.Literal) else INVERT[cls])
                          for node in nodes)
            score += 1.3 if correct else -0.95
    return score


def semantic_edits(tree, question, example):
    if not isinstance(tree, exp.Select):
        return
    q = question.casefold()
    projected = projections(tree)
    if projected and not any(isinstance(node, exp.AggFunc) for node in projected):
        amended = tree.copy()
        amended.set("distinct", None if tree.args.get("distinct") else exp.Distinct())
        weight = 2.0 if DISTINCT.search(q) and not tree.args.get("distinct") else 0.4
        yield weight, amended
    if DISTINCT.search(q) and any(isinstance(node, exp.Count) and isinstance(node.this, exp.Column)
                                  for node in projected):
        amended = tree.copy()
        for node in projections(amended):
            if isinstance(node, exp.Count) and isinstance(node.this, exp.Column):
                node.set("this", exp.Distinct(expressions=[node.this.copy()]))
        yield 2.2, amended
    direction = descending(q)
    order = tree.args.get("order")
    if direction is not None and order and order.expressions:
        if isinstance(order.expressions[0], exp.Ordered) and bool(order.expressions[0].args.get("desc")) != direction:
            amended = tree.copy()
            first = amended.args["order"].expressions[0]
            first.set("desc", direction)
            first.set("nulls_first", not direction)
            yield 2.1, amended
    explicit_limit = re.search(r"\b(?:top|bottom|first|last)\s+(\d+)\b", q)
    limit = tree.args.get("limit")
    if explicit_limit and order and (not limit or limit.expression.name != explicit_limit[1]):
        amended = tree.copy()
        amended.set("limit", exp.Limit(expression=exp.Literal.number(explicit_limit[1])))
        yield 2.1, amended
    for value, expected in comparison_intent(q):
        amended = tree.copy()
        edited = False
        for node in list(amended.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ)):
            for side, target in ((node.expression, expected), (node.this, INVERT[expected])):
                if isinstance(side, exp.Literal) and not side.is_string and side.name == value:
                    if not isinstance(node, target):
                        node.replace(target(this=node.this.copy(), expression=node.expression.copy()))
                        edited = True
                    break
        if edited:
            yield 2.5, amended
    # A group heading like 'each country' usually needs GROUP BY. Leave
    # group-key synthesis to the hints, but try the common global-count repair.
    if COUNT.search(q) and len(projected) == 1 and len(example.get("columns", [])) == 1:
        if isinstance(projected[0], exp.Column) and not tree.args.get("group") and not re.search(r"\b(each|per)\b", q):
            amended = tree.copy()
            target = projections(amended)[0]
            count_arg = exp.Distinct(expressions=[target.copy()]) if DISTINCT.search(q) else exp.Star()
            target.replace(exp.Count(this=count_arg))
            yield 1.7, amended
