"""
experiments/scripts/run_evaluation.py

Evaluate an LLM on a Logic Gym Worlds split.

Usage:
    python experiments/scripts/run_evaluation.py \
        --instances data/processed/test_operator_ood.json \
        --model claude-sonnet-4-6 \
        --style cot \
        --output experiments/results/claude_operator_ood_cot.json
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.llm_interface import get_llm, build_prompt, extract_answer
from src.evaluation.metrics import (
    compute_accuracy, compute_macro_f1, compute_per_label_accuracy,
    compute_contradiction_rate, compute_paraphrase_invariance,
    aggregate_results,
)


def load_instances(path: str):
    with open(path) as f:
        return json.load(f)


def build_negation_pairs(queries):
    """Extract (query_id, negation_id) pairs from the query list."""
    negation_of = {q["id"]: q["negation_of"] for q in queries if q.get("negation_of")}
    pairs = [(nid, qid) for nid, qid in negation_of.items()]
    return pairs


def build_paraphrase_groups(queries):
    """Build {group_id: [query_ids]} for paraphrase consistency."""
    groups = defaultdict(list)
    for q in queries:
        if q.get("paraphrase_group"):
            groups[q["paraphrase_group"]].append(q["id"])
            groups[q["paraphrase_group"]].append(q["paraphrase_group"])  # original
    # Deduplicate
    return {g: list(set(ids)) for g, ids in groups.items()}


class MockInstance:
    """Light wrapper around dict for prompt builder compatibility."""
    def __init__(self, d):
        self._d = d

    @property
    def rule_sheet(self):
        return self

    @property
    def rules(self):
        class Rule:
            def __init__(self, r):
                self.natural_language = r["natural_language"]
        return [Rule(r) for r in self._d["rules"]]

    @property
    def predicates(self):
        return self._d["predicates"]

    @property
    def facts(self):
        return self._d["facts"]


class MockQuery:
    def __init__(self, q):
        self.id = q["id"]
        self.natural_language = q["natural_language"]
        self.gold_label = q["gold_label"]
        self.query_type = q["query_type"]


def evaluate_instance(instance_dict, llm, style):
    """Run evaluation on one instance, return per-query predictions."""
    instance = MockInstance(instance_dict)
    results = {}

    for q_dict in instance_dict["queries"]:
        query = MockQuery(q_dict)
        prompt = build_prompt(instance, query, style=style)
        response = llm.query(prompt)
        pred = extract_answer(response)
        results[query.id] = {
            "prediction": pred,
            "gold": query.gold_label,
            "query_type": query.query_type,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True,  help="Path to split JSON")
    parser.add_argument("--model",     required=True,  help="Model name string")
    parser.add_argument("--style",     default="direct", choices=["direct", "cot", "few_shot"])
    parser.add_argument("--output",    required=True,  help="Output JSON path")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Limit number of instances (for quick tests)")
    args = parser.parse_args()

    print(f"Model:     {args.model}")
    print(f"Instances: {args.instances}")
    print(f"Style:     {args.style}")

    instances = load_instances(args.instances)
    if args.limit:
        instances = instances[:args.limit]

    llm = get_llm(args.model)

    all_predictions, all_gold = [], []
    bundle_results = []

    for inst in tqdm(instances, desc="Evaluating"):
        query_results = evaluate_instance(inst, llm, args.style)

        # Flatten for aggregate metrics
        preds_map = {qid: v["prediction"] for qid, v in query_results.items()}
        for qid, v in query_results.items():
            all_predictions.append(v["prediction"])
            all_gold.append(v["gold"])

        # Bundle-level
        queries = inst["queries"]
        neg_pairs  = build_negation_pairs(queries)
        para_groups = build_paraphrase_groups(queries)

        bundle_results.append({
            "instance_id":          inst["id"],
            "contradiction_rate":   compute_contradiction_rate(preds_map, neg_pairs),
            "paraphrase_invariance": compute_paraphrase_invariance(preds_map, para_groups),
            "joint_sat":            1,  # placeholder; real check needs live solver
            "query_results":        query_results,
        })

    # Aggregate
    report = aggregate_results(all_predictions, all_gold, bundle_results)
    report["model"]  = args.model
    report["style"]  = args.style
    report["split"]  = args.instances
    report["bundle_results"] = bundle_results

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Accuracy:          {report['accuracy']:.4f}")
    print(f"Macro-F1:          {report['macro_f1']:.4f}")
    print(f"Contradiction rate:{report['avg_contradiction_rate']:.4f}")
    print(f"Para invariance:   {report['avg_paraphrase_invariance']:.4f}")
    print(f"Results saved to:  {args.output}")


if __name__ == "__main__":
    main()
