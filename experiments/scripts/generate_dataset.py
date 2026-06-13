"""
experiments/scripts/generate_dataset.py

Generates the full Logic Gym Worlds dataset across all OOD splits.

Usage (local or Colab):
    python experiments/scripts/generate_dataset.py \
        --track horn_fol \
        --num_instances 1000 \
        --output_dir data/processed

    # Quick smoke test:
    python experiments/scripts/generate_dataset.py --num_instances 20 --quick
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generators.instance_generator import InstanceGenerator
from src.templates.rule_templates import TRAIN_OPERATORS, OOD_OPERATORS


# ── OOD split definitions ──────────────────────────────────────────────────

SPLITS = {
    "train": {
        "allowed_operators": TRAIN_OPERATORS,
        "difficulty": 3,
        "description": "Training set — standard operators only",
    },
    "dev": {
        "allowed_operators": TRAIN_OPERATORS,
        "difficulty": 3,
        "description": "Dev set — same distribution as train",
    },
    "test_iid": {
        "allowed_operators": TRAIN_OPERATORS,
        "difficulty": 3,
        "description": "Test IID — new instances, same operators",
    },
    "test_lexical_ood": {
        "allowed_operators": TRAIN_OPERATORS,
        "difficulty": 4,
        "description": "Lexical OOD — new predicate names, same operators, harder",
    },
    "test_operator_ood": {
        "allowed_operators": TRAIN_OPERATORS | OOD_OPERATORS,
        "difficulty": 4,
        "description": "Operator OOD — novel counting/iff operators",
    },
    "test_compositional_ood": {
        "allowed_operators": TRAIN_OPERATORS | {"count", "geq"},
        "difficulty": 5,
        "description": "Compositional OOD — novel operator combinations at high difficulty",
    },
}


def instance_to_dict(instance) -> dict:
    """Serialize an Instance to a JSON-compatible dict."""
    return {
        "id": instance.id,
        "metadata": instance.metadata,
        "rules": [
            {
                "template_id": r.template_id,
                "logical_form": r.logical_form,
                "natural_language": r.natural_language,
                "operators": sorted(r.operators),
            }
            for r in instance.rule_sheet.rules
        ],
        "predicates": instance.rule_sheet.predicates,
        "entities": instance.entities,
        "facts": instance.facts,
        "queries": [
            {
                "id": q.id,
                "logical_form": q.logical_form,
                "natural_language": q.natural_language,
                "gold_label": q.gold_label,
                "query_type": q.query_type,
                "paraphrase_group": q.paraphrase_group,
                "negation_of": q.negation_of,
            }
            for q in instance.queries
        ],
    }


def generate_split(
    split_name: str,
    split_config: dict,
    num_instances: int,
    output_dir: Path,
    domain: str = "animals",
    seed: int = 42,
) -> dict:
    """Generate one split and write to disk."""
    print(f"\n{'='*60}")
    print(f"Split: {split_name}  ({split_config['description']})")
    print(f"Instances: {num_instances} | Difficulty: {split_config['difficulty']}")
    print(f"Operators: {sorted(split_config['allowed_operators'])}")
    print(f"{'='*60}")

    gen = InstanceGenerator(domain_name=domain, seed=seed)
    instances = []
    failed = 0

    for i in tqdm(range(num_instances), desc=split_name):
        inst = gen.generate_instance(
            num_rules=4,
            num_entities=3,
            num_queries=6,
            allowed_operators=split_config["allowed_operators"],
            difficulty=split_config["difficulty"],
        )
        if inst:
            instances.append(instance_to_dict(inst))
        else:
            failed += 1

    output_path = output_dir / f"{split_name}.json"
    with open(output_path, "w") as f:
        json.dump(instances, f, indent=2)

    summary = {
        "split": split_name,
        "generated": len(instances),
        "failed": failed,
        "output": str(output_path),
    }
    print(f"  ✓ Generated {len(instances)} | Failed: {failed} | Saved: {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate Logic Gym Worlds dataset")
    parser.add_argument("--track",         default="horn_fol", choices=["horn_fol"])
    parser.add_argument("--num_instances", type=int, default=500,
                        help="Instances per split (train gets 2x)")
    parser.add_argument("--domain",        default="animals",
                        choices=["animals", "company", "geography", "academic"])
    parser.add_argument("--output_dir",    default="data/processed")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--quick",         action="store_true",
                        help="Generate only 20 instances per split for smoke testing")
    args = parser.parse_args()

    if args.quick:
        args.num_instances = 20

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for split_name, split_config in SPLITS.items():
        n = args.num_instances * 2 if split_name == "train" else args.num_instances
        summary = generate_split(
            split_name=split_name,
            split_config=split_config,
            num_instances=n,
            output_dir=output_dir,
            domain=args.domain,
            seed=args.seed,
        )
        summaries.append(summary)

    # Save generation report
    report_path = output_dir / "generation_report.json"
    with open(report_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n✅ Done! Report saved to {report_path}")
    total = sum(s["generated"] for s in summaries)
    print(f"   Total instances generated: {total}")


if __name__ == "__main__":
    main()
