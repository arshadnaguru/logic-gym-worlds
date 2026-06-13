"""
src/evaluation/metrics.py

Query-level and bundle-level evaluation metrics for Logic Gym Worlds.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict


# ── Query-level metrics ────────────────────────────────────────────────────

def compute_accuracy(predictions: List[str], gold_labels: List[str]) -> float:
    """Simple accuracy: fraction of exact matches."""
    if not predictions:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, gold_labels))
    return correct / len(predictions)


def compute_macro_f1(predictions: List[str], gold_labels: List[str]) -> float:
    """Macro-averaged F1 across all unique labels."""
    labels = list(set(gold_labels))
    f1_scores = []
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, gold_labels) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, gold_labels) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, gold_labels) if p != label and g == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def compute_per_label_accuracy(
    predictions: List[str], gold_labels: List[str]
) -> Dict[str, float]:
    """Accuracy broken down by gold label."""
    label_correct: Dict[str, int] = defaultdict(int)
    label_total:   Dict[str, int] = defaultdict(int)
    for p, g in zip(predictions, gold_labels):
        label_total[g] += 1
        if p == g:
            label_correct[g] += 1
    return {
        label: label_correct[label] / label_total[label]
        for label in label_total
    }


# ── Bundle-level metrics ───────────────────────────────────────────────────

def compute_contradiction_rate(
    predictions: Dict[str, str],
    negation_pairs: List[Tuple[str, str]],
) -> float:
    """
    Fraction of negation pairs where the model says both are 'entails'.
    A perfect reasoner should have contradiction_rate = 0.
    """
    if not negation_pairs:
        return 0.0
    contradictions = sum(
        1
        for q1_id, q2_id in negation_pairs
        if predictions.get(q1_id) == "entails"
        and predictions.get(q2_id) == "entails"
    )
    return contradictions / len(negation_pairs)


def compute_paraphrase_invariance(
    predictions: Dict[str, str],
    paraphrase_groups: Dict[str, List[str]],
) -> float:
    """
    For each paraphrase group, check if all queries get the same answer.
    Returns fraction of groups that are fully consistent.
    """
    if not paraphrase_groups:
        return 1.0
    consistent = 0
    for group_id, query_ids in paraphrase_groups.items():
        preds = [predictions.get(qid) for qid in query_ids if qid in predictions]
        if len(preds) > 1 and len(set(preds)) == 1:
            consistent += 1
    return consistent / len(paraphrase_groups)


def compute_joint_satisfiability_rate(bundle_results: List[Dict]) -> float:
    """Average joint satisfiability across all instances."""
    vals = [b.get("joint_sat", 0) for b in bundle_results]
    return float(np.mean(vals)) if vals else 0.0


# ── Aggregate report ────────────────────────────────────────────────────────

def aggregate_results(
    all_predictions: List[str],
    all_gold: List[str],
    bundle_results: List[Dict],
) -> Dict:
    """Return a flat results dict ready for JSON serialization."""
    return {
        "accuracy":            round(compute_accuracy(all_predictions, all_gold), 4),
        "macro_f1":            round(compute_macro_f1(all_predictions, all_gold), 4),
        "per_label_accuracy":  {
            k: round(v, 4)
            for k, v in compute_per_label_accuracy(all_predictions, all_gold).items()
        },
        "avg_contradiction_rate": round(
            float(np.mean([b.get("contradiction_rate", 0) for b in bundle_results]))
            if bundle_results else 0.0,
            4,
        ),
        "avg_joint_sat": round(
            compute_joint_satisfiability_rate(bundle_results), 4
        ),
        "avg_paraphrase_invariance": round(
            float(np.mean([b.get("paraphrase_invariance", 1) for b in bundle_results]))
            if bundle_results else 1.0,
            4,
        ),
        "num_instances": len(bundle_results),
        "num_queries":   len(all_predictions),
    }
