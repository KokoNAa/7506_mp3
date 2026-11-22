"""Course entry point for the shared SQL repair harness and its selectors."""
import json
from pathlib import Path

from mp3_agent.agent import Agent


def create_agent(checkpoint=None):
    config = json.loads(Path(checkpoint).read_text()) if checkpoint else {}
    ranker = config.get("ranker", "rules")
    guided = config.get("guided", False)
    beam_choices = config.get("beam_choices", False)
    if ranker == "rules":
        selector = None
    elif ranker in ("qwen", "lora"):
        from mp3_agent.ranker import QwenSelector

        selector = QwenSelector(use_adapter=ranker == "lora", include_example=config.get("model_examples", False),
                                rule_weight=config.get("rule_weight", 0.0), guided=guided,
                                learned_weights=config.get("learned_weights"),
                                learned_strength=config.get("learned_strength", 0.0),
                                rank_views=min(2, max(1, config.get("rank_views", 1))),
                                likelihood_weight=config.get("likelihood_weight", 0.0),
                                finalist_weight=config.get("finalist_weight", 0.0))
    else:
        raise ValueError(f"Unsupported ranker: {ranker!r}")
    rewriter = None
    if config.get("rewrite"):
        if ranker not in ("qwen", "lora"):
            raise ValueError("Rewrite and ranking must share one live model")
        from mp3_agent.rewrite import QwenRewriter

        rewriter = QwenRewriter(adapter=ranker == "lora", exploratory=config.get("exploratory", False),
                               linked=config.get("linked_schema", False), focused=config.get("focused_rewrite", False),
                               beam_choices=beam_choices, reasoned_retry=config.get("reasoned_retry", False),
                               show_paths=config.get("show_join_paths", False))
    return Agent(selector, use_schema=config.get("schema_repairs", True),
                 edits=config.get("semantic_edits", True), guided=guided, rewriter=rewriter,
                 extended=config.get("extended", False), dedup=config.get("dedup", False),
                 rewrite_limit=min(2 if beam_choices else 3, max(0, config.get("rewrite_limit", 2))),
                 proactive=config.get("proactive", False), deeper=config.get("deeper", False),
                 ambiguous_retry=config.get("ambiguous_retry", False), compose=config.get("compose", False),
                 feedback=config.get("feedback_repairs", False), ambiguity_rows=config.get("ambiguity_rows", 0),
                 structural_search=config.get("structural_search", False),
                 equivalent_checks=config.get("equivalent_checks", False),
                 value_probes=config.get("value_probes", False),
                 preserve_candidates=config.get("preserve_candidates", False),
                 retrieve_repairs=config.get("retrieve_repairs", False))
