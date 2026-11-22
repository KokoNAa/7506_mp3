"""State-dependent harness shared by all candidate selectors."""
from dataclasses import dataclass
import math

from sqlglot.errors import SqlglotError

from .intent import intent_score, semantic_edits
from .schema import Catalog, parse, render, schema_repairs


@dataclass(frozen=True)
class Candidate:
    sql: str
    feedback: dict


def checked_candidates(observation):
    return [Candidate(event["sql"], event["observation"]) for event in observation["history"]
            if event["tool"] == "check"]


def visible_matches(observation):
    return [candidate for candidate in checked_candidates(observation)
            if candidate.feedback.get("matches_example")]


def rule_score(candidate, observation):
    feedback = candidate.feedback
    task = observation["task"]
    example = observation["example"]
    example_count = example.get("row_count", 0)
    score = 100.0 if feedback["matches_example"] else 0.0
    if feedback["ok"]:
        score += 10.0
        score -= 2.0 * abs(len(feedback["columns"]) - len(example.get("columns", [])))
        score -= abs(math.log1p(feedback["row_count"]) - math.log1p(example_count))
    if candidate.sql == observation["issue_sql"]:
        score += 0.9 if feedback["matches_example"] else 0.12
    score += intent_score(parse(candidate.sql), task["question"], observation["ordered"])
    return score


def checked_choice(observation, guided=False):
    """Submit a checked candidate, favoring public matches and question intent."""
    checked = checked_candidates(observation)
    if not checked:
        return observation["issue_sql"]
    if guided:
        from .guided import shape_agreement

        return max(checked, key=lambda candidate: rule_score(candidate, observation)
                   + shape_agreement(parse(candidate.sql), observation["example"])).sql
    return max(checked, key=lambda candidate: rule_score(candidate, observation)).sql


class Agent:
    def __init__(self, selector=None, use_schema=True, edits=True, guided=False, rewriter=None,
                 extended=False, dedup=False, rewrite_limit=2, proactive=False, deeper=False,
                 ambiguous_retry=False, compose=False, feedback=False, ambiguity_rows=0,
                 structural_search=False, equivalent_checks=False, value_probes=False,
                 preserve_candidates=False, retrieve_repairs=False):
        self.selector = selector or (lambda observation: checked_choice(observation, guided=guided))
        self.use_schema = use_schema
        self.edits = edits
        self.guided = guided
        self.extended = extended
        self.deeper = deeper
        self.rewriter = rewriter
        self.rewrites = 0
        self.rewrite_limit = rewrite_limit
        self.proactive = proactive
        self.ambiguous_retry = ambiguous_retry
        self.ambiguity_rows = ambiguity_rows
        self.compose = compose
        self.feedback = feedback
        self.dedup = dedup
        self.structural_search = structural_search
        self.equivalent_checks = equivalent_checks
        self.value_probes = value_probes
        self.preserve_candidates = preserve_candidates
        self.retrieve_repairs = retrieve_repairs
        self.catalog = None
        self.proposals = {}
        self.seeded = False
        self.handled_failures = set()
        self.handled_feedback = set()
        self.search_seeds = set()
        self.probe = None
        self.processed_probe = False
        self.probe_candidates = set()

    def key(self, tree):
        if self.equivalent_checks and self.catalog is not None:
            from .search import equivalent_key

            return equivalent_key(tree, self.catalog)
        if self.dedup and self.catalog is not None:
            from .schema import candidate_key

            return candidate_key(tree, self.catalog)
        return render(tree)

    def offer_guided(self, tree, observation, priority):
        from .guided import guided_edits, shape_agreement

        if self.catalog is None or tree is None:
            return
        try:
            question = observation["task"]["question"]
            projected = None
            best_shape = shape_agreement(tree, observation["example"]) if self.compose else 0
            for weight, alternative in guided_edits(tree, self.catalog, question,
                                                    observation["example"], observation["ordered"]):
                self.offer(render(alternative), alternative, priority + weight)
                if self.compose:
                    agreement = shape_agreement(alternative, observation["example"])
                    if agreement > best_shape:
                        projected = alternative
                        best_shape = agreement
            if self.extended:
                from .structural import extra_edits

                for weight, alternative in extra_edits(tree, self.catalog, question.casefold()):
                    self.offer(render(alternative), alternative, priority + weight)
            if self.deeper:
                from .semantics import semantic_repairs

                for weight, alternative in semantic_repairs(tree, self.catalog, question, observation["example"]):
                    self.offer(render(alternative), alternative, priority + weight)
            if projected is not None:
                from .semantics import semantic_repairs
                from .structural import extra_edits

                alternatives = [*semantic_repairs(projected, self.catalog, question, observation["example"]),
                                *extra_edits(projected, self.catalog, question.casefold())]
                for weight, alternative in sorted(alternatives, key=lambda pair: -pair[0])[:2]:
                    self.offer(render(alternative), alternative, priority + weight + 0.15)
        except (SqlglotError, RecursionError):
            # Malformed drafts can parse before failing scope analysis (e.g. duplicate aliases).
            return

    def offer(self, sql, tree, priority):
        if tree is None or len(sql.encode("utf-8")) > 16000:
            return
        key = self.key(tree)
        previous = self.proposals.get(key)
        if previous is None or priority > previous[0]:
            self.proposals[key] = (priority, sql)

    def offer_repair(self, tree, error, priority):
        if tree is None or self.catalog is None:
            return
        for weight, proposed in schema_repairs(tree, self.catalog, error):
            self.offer(render(proposed), proposed, priority + 3.0 * weight)

    def candidates(self, observation):
        history = observation["history"]
        question = observation["task"]["question"]
        draft = observation["issue_sql"]
        hints = next(event["observation"] for event in history if event["tool"] == "repair_hint")
        schema = next((event["observation"] for event in history if event["tool"] == "schema"), [])
        if not self.seeded:
            self.catalog = Catalog(schema) if self.use_schema else None
            draft_tree = parse(draft)
            hint_trees = [(sql, parse(sql)) for sql in hints]
            for index, (sql, tree) in enumerate(hint_trees):
                rank = intent_score(tree, question, observation["ordered"])
                self.offer(sql, tree, 10.0 + 0.24 * rank - 0.2 * index)
                self.offer_repair(tree, "", 6.2)
            if self.edits:
                for index, tree in enumerate([draft_tree, *(tree for _, tree in hint_trees)]):
                    for weight, alternative in semantic_edits(tree, question, observation["example"]):
                        self.offer(render(alternative), alternative, 6.0 + weight - 0.22 * index)
            if self.guided:
                from .guided import simple_projections

                for index, tree in enumerate([draft_tree, *(tree for _, tree in hint_trees)]):
                    self.offer_guided(tree, observation, 7.2 - 0.18 * index)
                if self.catalog is not None:
                    for weight, alternative in simple_projections(self.catalog, question, observation["example"]):
                        self.offer(render(alternative), alternative, 7.2 + weight)
            self.seeded = True
        tried = set()
        for candidate in checked_candidates(observation):
            tree = parse(candidate.sql)
            if tree is not None:
                tried.add(self.key(tree))
            error = candidate.feedback.get("error") or {}
            message = error.get("message", "")
            if (self.guided and candidate.feedback["ok"] and not candidate.feedback["matches_example"]
                    and candidate.sql not in self.handled_feedback):
                self.offer_guided(tree, observation, 6.45)
                if self.feedback:
                    from .feedback import feedback_repairs

                    for weight, alternative in feedback_repairs(
                        tree, candidate.feedback, observation["example"], question, observation["ordered"]
                    ):
                        self.offer(render(alternative), alternative, 7.25 + weight)
                self.handled_feedback.add(candidate.sql)
            if candidate.sql not in self.handled_failures and message.casefold().startswith(
                ("no such column:", "no such table:", "ambiguous column name:")
            ):
                self.offer_repair(tree, message, 8.5 if candidate.sql == draft else 6.8)
                self.handled_failures.add(candidate.sql)
        if self.structural_search and self.catalog is not None:
            self.expand_structure(observation, hints)
        if self.probe and not self.processed_probe:
            self.expand_probe(history)
        return sorted((proposal for key, proposal in self.proposals.items() if key not in tried),
                      key=lambda item: -item[0])

    def expand_structure(self, observation, hints):
        from .guided import shape_agreement
        from .search import structural_search

        checked = checked_candidates(observation)
        if len(checked) < 2 and len(self.proposals) > 1:
            return
        matches = visible_matches(observation)
        if self.preserve_candidates and (observation["remaining"] > 3 or matches):
            return
        example = observation["example"]
        rows = example.get("rows", [])
        weak = not rows or rows == [[0]] or rows == [[None]]
        if not weak and any(shape_agreement(parse(candidate.sql), example) >= 2.7 for candidate in matches):
            return
        question = observation["task"]["question"]
        sources = list(dict.fromkeys([*(candidate.sql for candidate in checked), *hints]))
        donors = [parse(sql) for sql in sources[:6]]
        base_priority = 5.3 if self.preserve_candidates else 7.45
        for position, sql in enumerate(sources):
            if sql in self.search_seeds:
                continue
            self.search_seeds.add(sql)
            tree = parse(sql)
            if tree is None:
                continue
            try:
                for weight, alternate in structural_search(tree, donors, self.catalog, question, example):
                    self.offer(render(alternate), alternate, base_priority + weight - 0.13 * position)
            except (SqlglotError, RecursionError):
                continue

    def expand_probe(self, history):
        from .value_probe import probe_repairs

        result = next((event["observation"] for event in history
                       if event["tool"] == "execute" and event["sql"] == self.probe.query), None)
        if result is None:
            return
        self.processed_probe = True
        for weight, alternative in probe_repairs(self.probe, result):
            sql = render(alternative)
            self.offer(sql, alternative, 15.0 + weight)
            self.probe_candidates.add(self.key(alternative))

    def generated_query(self, sql, observation, tried):
        tree = parse(sql)
        if tree is None:
            return None
        if self.structural_search and self.catalog is not None:
            from .guided import header, projection, shape_agreement
            from sqlglot import exp

            example = observation["example"]
            score = shape_agreement(tree, example)
            if score < 1.5 and isinstance(tree, exp.Select):
                columns = [header(name, self.catalog) for name in example.get("columns", [])]
                if columns and all(col is not None for col in columns):
                    try:
                        fix = projection(tree, self.catalog, columns, observation["task"]["question"])
                    except (SqlglotError, RecursionError):
                        fix = None
                    if (fix is not None and shape_agreement(fix, example) >= 2.7
                            and self.key(fix) not in tried and len(render(fix).encode()) <= 16000):
                        if self.preserve_candidates:
                            self.offer(render(fix), fix, 11.25)
                        else:
                            self.offer(sql, tree, 12.5)
                            return render(fix)
        return sql if self.key(tree) not in tried and len(sql.encode()) <= 16000 else None

    def act(self, observation):
        history = observation["history"]
        if not history:
            return {"tool": "check", "sql": observation["issue_sql"]}
        if not observation["remaining"]:
            return {"tool": "submit", "sql": self.selector(observation)}
        tools = {event["tool"] for event in history}
        if "repair_hint" not in tools:
            return {"tool": "repair_hint"}
        if "schema" not in tools:
            return {"tool": "schema"}
        candidates = self.candidates(observation)
        if candidates and self.probe_candidates:
            _, leading = candidates[0]
            if self.key(parse(leading)) in self.probe_candidates:
                return {"tool": "check", "sql": leading}
        if (self.retrieve_repairs and self.rewriter and self.rewrites < self.rewrite_limit
                and observation["remaining"] == 2 and "retrieve" not in tools
                and not visible_matches(observation)):
            from .rewrite import retrieval_query

            query = retrieval_query(observation)
            if query:
                return {"tool": "retrieve", "query": query}
        if (self.value_probes and self.catalog is not None and self.probe is None
                and observation["remaining"] == 2 and not visible_matches(observation)):
            from .value_probe import choose_probe

            self.probe = choose_probe(checked_candidates(observation), observation["example"],
                                      self.catalog, observation["task"]["question"])
            if self.probe:
                return {"tool": "execute", "sql": self.probe.query}
        if (self.rewriter and self.rewrites < self.rewrite_limit
                and (observation["remaining"] <= self.rewrite_limit or not candidates)):
            from .guided import shape_agreement

            example = observation["example"]
            matches = visible_matches(observation)
            uncertain = not matches or (self.rewrites == 0 and (
                example["row_count"] == 0 or max(
                    shape_agreement(parse(candidate.sql), example) for candidate in matches
                ) < 1.5
            ))
            if self.proactive and self.rewrites == 0:
                initial = history[0]["observation"]
                uncertain = uncertain or not initial["matches_example"] or len(matches) < 2
            if self.ambiguous_retry and self.rewrites == 1 and example["row_count"] <= 1:
                rows = example.get("rows", [])
                weak = not rows or rows == [[None]] or rows == [[0]]
                uncertain = uncertain or weak
            if self.ambiguity_rows and self.rewrites == 1 and example["row_count"] <= self.ambiguity_rows:
                uncertain = uncertain or not history[0]["observation"]["matches_example"]
            if uncertain:
                self.rewrites += 1
                generated = self.rewriter(observation)
                if generated:
                    tried = {self.key(tree) for candidate in checked_candidates(observation)
                             if (tree := parse(candidate.sql)) is not None}
                    prepared = self.generated_query(generated, observation, tried)
                    if prepared is not None:
                        return {"tool": "check", "sql": prepared}
        if candidates:
            return {"tool": "check", "sql": candidates[0][1]}
        return {"tool": "submit", "sql": self.selector(observation)}
