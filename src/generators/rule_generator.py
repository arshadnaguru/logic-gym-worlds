"""
src/generators/rule_generator.py

Instantiates rule templates with concrete predicates from a domain vocabulary.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

from src.templates.rule_templates import RuleTemplate, TEMPLATES
from src.templates.surface_forms import DOMAINS


@dataclass
class GeneratedRule:
    """A concrete rule instantiated from a template."""
    template_id: str
    logical_form: str       # e.g. "forall X: bird(X) -> can_fly(X)"
    natural_language: str   # e.g. "Every bird is able to fly."
    operators: Set[str]


@dataclass
class RuleSheet:
    """A complete set of rules for one benchmark instance."""
    rules: List[GeneratedRule]
    all_operators: Set[str]
    predicates: Dict[str, str]   # pred_id -> natural language name
    difficulty: int


class RuleGenerator:
    def __init__(
        self,
        templates: List[RuleTemplate] = None,
        domain_name: str = "animals",
        seed: Optional[int] = None,
    ):
        self.templates = templates or TEMPLATES
        self.domain = DOMAINS[domain_name]
        self.rng = random.Random(seed)

    def generate_rule_sheet(
        self,
        num_rules: int,
        allowed_operators: Set[str],
        target_difficulty: int,
    ) -> RuleSheet:
        """Generate a complete rule sheet from templates matching the constraints."""

        valid_templates = [
            t for t in self.templates
            if t.operators.issubset(allowed_operators)
            and t.difficulty <= target_difficulty
        ]

        if not valid_templates:
            raise ValueError(
                f"No templates match operators={allowed_operators}, "
                f"difficulty<={target_difficulty}"
            )

        rules: List[GeneratedRule] = []
        used_predicates: Dict[str, str] = {}
        all_operators: Set[str] = set()

        for _ in range(num_rules):
            template = self.rng.choice(valid_templates)
            rule, new_preds = self._instantiate_template(template, used_predicates)
            rules.append(rule)
            used_predicates.update(new_preds)
            all_operators.update(template.operators)

        return RuleSheet(
            rules=rules,
            all_operators=all_operators,
            predicates=used_predicates,
            difficulty=target_difficulty,
        )

    def _instantiate_template(
        self,
        template: RuleTemplate,
        existing_predicates: Dict[str, str],
    ) -> Tuple[GeneratedRule, Dict[str, str]]:
        """Fill a template with concrete predicates from the domain."""

        # Separate unary and binary pools
        unary_pool = [
            p for p in self.domain["unary_predicates"]
            if p[0] not in existing_predicates
        ]
        binary_pool = [
            p for p in self.domain.get("binary_predicates", [])
            if p[0] not in existing_predicates
        ]

        # Refill if exhausted
        if len(unary_pool) < template.predicate_types.count("unary"):
            unary_pool = list(self.domain["unary_predicates"])
        if len(binary_pool) < template.predicate_types.count("binary"):
            binary_pool = list(self.domain.get("binary_predicates", []))

        # Assign predicates by type
        pred_map: Dict[str, str] = {}  # placeholder -> pred_id
        name_map: Dict[str, str] = {}  # placeholder -> natural name
        new_preds: Dict[str, str] = {}

        placeholder_letters = list("PQRSTU")
        u_idx, b_idx = 0, 0

        for i, ptype in enumerate(template.predicate_types):
            ph = placeholder_letters[i]
            if ptype == "unary":
                pid, pname = unary_pool[u_idx]
                u_idx += 1
            else:
                pid, pname = binary_pool[b_idx]
                b_idx += 1
            pred_map[ph] = pid
            name_map[ph] = pname
            new_preds[pid] = pname

        # Build logical form substitution
        logical = template.logical_form
        for ph, pid in pred_map.items():
            # Replace P( with pid(  (single-letter placeholder)
            logical = logical.replace(f"{ph}(", f"{pid}(")

        # Build natural language substitution
        surface = self.rng.choice(template.surface_forms)
        for ph, pname in name_map.items():
            surface = surface.replace(f"{{{ph}}}", pname)
        # Replace entity placeholder
        entity_word = self.rng.choice(["something", "an entity", "it"])
        surface = surface.replace("{x}", entity_word)
        # Replace counting placeholder with a concrete n if present
        if "{n}" in surface:
            n = self.rng.choice([2, 3])
            surface = surface.replace("{n}", str(n))
            logical = logical.replace("N", str(n))

        return (
            GeneratedRule(
                template_id=template.id,
                logical_form=logical,
                natural_language=surface,
                operators=template.operators,
            ),
            new_preds,
        )
