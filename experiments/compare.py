"""Verify identical candidate tool traces, then summarize official scores."""
import argparse
from hashlib import sha256
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    fixed_trace = None
    for ranker in ("rules", "qwen", "lora"):
        prediction_path = args.outputs / f"validation-{ranker}.json"
        predictions = load(prediction_path)
        score = load(args.outputs / f"validation-{ranker}-score.json")
        if (score["split"] != "validation" or score["n"] != len(predictions["predictions"])
                or score["predictions_sha256"] != sha256(prediction_path.read_bytes()).hexdigest()):
            raise ValueError(f"Official score does not match validation predictions for {ranker}")
        trace = [(record["id"], record["trace"][:-1]) for record in predictions["predictions"]]
        if fixed_trace is None:
            fixed_trace = trace
        elif trace != fixed_trace:
            raise ValueError(f"Candidate pool or tool feedback changed for {ranker}")
        calls = args.outputs / f"validation-{ranker}-calls.jsonl"
        resources_path = args.outputs / f"validation-{ranker}-predict-resources.json"
        results[ranker] = {
            "correct": score["correct"], "total": score["n"], "accuracy": score["accuracy"],
            "mean_tool_calls": score["mean_tool_calls"],
            "mean_model_calls": len(calls.read_text().splitlines()) / score["n"] if calls.is_file() else 0.0,
            "prediction_seconds": predictions["prediction_seconds"],
            "scoring_seconds": score["scoring_seconds"],
        }
        if resources_path.is_file():
            resources = load(resources_path)
            results[ranker]["prediction_peak_rss_mib"] = resources["peak_rss_kib"] / 1024
    fingerprint = sha256(json.dumps(fixed_trace, sort_keys=True).encode()).hexdigest()
    report = {"candidate_trace_sha256": fingerprint, "results": results}
    output = json.dumps(report, indent=2)
    (args.outputs / "validation-comparison.json").write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
