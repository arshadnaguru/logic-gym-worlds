import re
"""
src/generators/instance_generator.py

Generates complete benchmark instances:
  RuleSheet + Facts + QueryBundle + GoldLabels (from Z3)
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple

from src.generators.rule_generator import RuleGenerator, RuleSheet, GeneratedRule
from src.solvers.z3_backend import Z3Solver, EntailmentResult
from src.templates.surface_forms import DOMAINS


@dataclass
class Query:
    """One query inside a bundle."""
    id: str
    logical_form: str          # e.g. "can_fly(Tweety)"
    natural_language: str      # e.g. "Is Tweety able to fly?"
    gold_label: str            # "entails" | "contradicts" | "unknown"
    query_type: str            # "entailment" | "negation_pair" | "paraphrase"
    paraphrase_group: Optional[str] = None   # links paraphrase variants
    negation_of: Optional[str] = None       # links Q and NOT-Q pairs


@dataclass
class Instance:
    """A complete benchmark instance."""
    id: str
    rule_sheet: RuleSheet
    facts: List[Dict]           # list of {predicate, args, positive}
    entities: List[str]
    queries: List[Query]
    metadata: Dict


class InstanceGenerator:
    """
    Generates benchmark instances by:
    1. Sampling a rule sheet
    2. Randomly assigning facts to entities
    3. Verifying satisfiability with Z3
    4. Generating a query bundle with gold labels
    5. Adding negation pairs and paraphrase variants
    """

    def __init__(
        self,
        domain_name: str = "animals",
        seed: Optional[int] = None,
    ):
        self.domain_name = domain_name
        self.domain = DOMAINS[domain_name]
        self.rng = random.Random(seed)
        self.rule_gen = RuleGenerator(domain_name=domain_name, seed=seed)
        self.solver = Z3Solver()

    def generate_instance(
        self,
        num_rules: int = 4,
        num_entities: int = 3,
        num_queries: int = 6,
        allowed_operators: Set[str] = None,
        difficulty: int = 2,
        max_retries: int = 50,
    ) -> Optional[Instance]:
        """
        Generate one valid instance. Returns None if max_retries exceeded.
        """
        if allowed_operators is None:
            allowed_operators = {"forall", "exists", "and", "or", "implies", "not"}

        for attempt in range(max_retries):
            # 1. Rule sheet
            try:
                rule_sheet = self.rule_gen.generate_rule_sheet(
                    num_rules=num_rules,
                    allowed_operators=allowed_operators,
                    target_difficulty=difficulty,
                )
            except ValueError:
                continue

            # 2. Sample entities
            entities = self.rng.sample(
                self.domain["entities"],
                min(num_entities, len(self.domain["entities"])),
            )

            # 3. Generate random facts
            facts = self._generate_facts(rule_sheet, entities)

            # 4. Load into solver and verify SAT
            self._load_into_solver(rule_sheet, facts, entities)
            if not self.solver.check_satisfiability():
                continue  # Unsatisfiable — retry

            # 5. Build query bundle
            queries = self._generate_query_bundle(rule_sheet, facts, entities, num_queries)

            # 6. Require label diversity
            labels = set(q.gold_label for q in queries)
            if len(labels) < 2:
                continue

            return Instance(
                id=str(uuid.uuid4())[:8],
                rule_sheet=rule_sheet,
                facts=facts,
                entities=entities,
                queries=queries,
                metadata={
                    "difficulty": difficulty,
                    "operators": sorted(rule_sheet.all_operators),
                    "num_rules": num_rules,
                    "num_entities": num_entities,
                    "domain": self.domain_name,
                },
            )

        return None  # Failed

    # ── Private helpers ─────────────────────────────────────────────────────

    def _generate_facts(
        self, rule_sheet: RuleSheet, entities: List[str]
    ) -> List[Dict]:
        """Randomly assign unary predicates to entities."""
        facts = []
        predicates = list(rule_sheet.predicates.keys())
        for entity in entities:
            num_assigned = self.rng.randint(1, max(1, len(predicates) // 2))
            assigned = self.rng.sample(predicates, min(num_assigned, len(predicates)))
            for pred in assigned:
                facts.append({"predicate": pred, "args": [entity], "positive": True})
        return facts

    def _load_into_solver(
        self, rule_sheet: RuleSheet, facts: List[Dict], entities: List[str]
    ):
        """Reset Z3 and load current rule sheet + facts."""
        self.solver.reset()

        # Declare all predicates
        for pred_id in rule_sheet.predicates:
            self.solver.declare_predicate(pred_id, arity=1)

        # Declare all entities
        for entity in entities:
            self.solver.declare_constant(entity)

        # Load rules as simple implications where possible
        for rule in rule_sheet.rules:
            self._add_rule_to_solver(rule)

        # Load facts
        for fact in facts:
            if fact["positive"]:
                self.solver.add_fact(fact["predicate"], *fact["args"])
            else:
                self.solver.add_negative_fact(fact["predicate"], *fact["args"])

    def _add_rule_to_solver(self, rule: GeneratedRule):
        """
        Parse simple logical forms and load into Z3.
        Handles: forall X: P(X) -> Q(X), P(X) -> not Q(X), (P(X) and Q(X)) -> R(X)
        Anything more complex is skipped (solver stays sound, just less constrained).
        """
        lf = rule.logical_form.strip()

        # Only handle "forall X: ..." for now
        if not lf.startswith("forall X:"):
            return

        body = lf[len("forall X:"):].strip()

        # Use word-boundary regex to detect structural " and " (not inside predicate names)
        has_structural_and = bool(re.search(r"\)\s+and\s+\w", body))

        # Pattern: P(X) -> Q(X)  (no structural 'and' in antecedent)
        if "->" in body and not has_structural_and:
            parts = body.split("->", 1)
            ant = parts[0].strip()
            con = parts[1].strip()
            ant_pred = self._extract_pred(ant)
            negated = con.startswith("not ")
            con_pred = self._extract_pred(con[4:].strip() if negated else con)
            if ant_pred and con_pred:
                if negated:
                    self.solver.add_negation_implication(ant_pred, con_pred)
                else:
                    self.solver.add_implication(ant_pred, con_pred)

        # Pattern: (P(X) and Q(X)) -> R(X)
        elif "->" in body and has_structural_and:
            parts = body.split("->", 1)
            ant = parts[0].strip().strip("()")
            con = parts[1].strip()
            ant_parts = re.split(r"\s+and\s+", ant, maxsplit=1)
            if len(ant_parts) == 2:
                ap = self._extract_pred(ant_parts[0].strip())
                bp = self._extract_pred(ant_parts[1].strip())
                neg_con = con.startswith("not ")
                cp = self._extract_pred(con[4:].strip() if neg_con else con)
                if ap and bp and cp:
                    if neg_con:
                        self.solver.add_negation_implication(ap, cp)
                        self.solver.add_negation_implication(bp, cp)
                    else:
                        self.solver.add_conjunction_implication(ap, bp, cp)

    @staticmethod
    def _extract_pred(atom: str) -> Optional[str]:
        """Extract predicate name from 'pred(X)' string."""
        if "(" in atom:
            return atom[:atom.index("(")].strip()
        return None

    def _generate_query_bundle(
        self,
        rule_sheet: RuleSheet,
        facts: List[Dict],
        entities: List[str],
        num_queries: int,
    ) -> List[Query]:
        """Build entailment queries + negation pairs + paraphrase variants."""
        queries: List[Query] = []
        predicates = list(rule_sheet.predicates.keys())
        seen_pairs: Set[Tuple[str, str]] = set()

        attempts = 0
        while len(queries) < num_queries and attempts < num_queries * 10:
            attempts += 1
            pred = self.rng.choice(predicates)
            entity = self.rng.choice(entities)
            key = (pred, entity)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            result = self.solver.check_entailment(pred, entity)
            pred_name = rule_sheet.predicates[pred]
            nl = f"Is {entity} {pred_name}?"

            qid = f"q_{len(queries)}"
            queries.append(Query(
                id=qid,
                logical_form=f"{pred}({entity})",
                natural_language=nl,
                gold_label=result.value,
                query_type="entailment",
            ))

        # Add negation pairs (opposite queries for bundle consistency check)
        neg_queries = []
        for q in queries[:4]:    # limit to first 4 to keep bundle manageable
            neg_id = f"{q.id}_neg"
            neg_nl = f"Is {q.logical_form.split('(')[1].rstrip(')')} NOT {rule_sheet.predicates.get(q.logical_form.split('(')[0], '?')}?"
            # Gold label for negation: flip entails/contradicts, unknown stays unknown
            if q.gold_label == "entails":
                neg_label = "contradicts"
            elif q.gold_label == "contradicts":
                neg_label = "entails"
            else:
                neg_label = "unknown"

            neg_queries.append(Query(
                id=neg_id,
                logical_form=f"not {q.logical_form}",
                natural_language=neg_nl,
                gold_label=neg_label,
                query_type="negation_pair",
                negation_of=q.id,
            ))

        # Add one paraphrase variant per original query (same label, different wording)
        para_queries = []
        paraphrase_templates = [
            "Does {entity} have the property of being {pred}?",
            "Can we conclude that {entity} is {pred}?",
            "Based on the rules, is {entity} {pred}?",
        ]
        for q in queries[:3]:
            pred_part = q.logical_form.split("(")[0]
            entity_part = q.logical_form.split("(")[1].rstrip(")")
            pred_name = rule_sheet.predicates.get(pred_part, pred_part)
            template = self.rng.choice(paraphrase_templates)
            para_nl = template.format(entity=entity_part, pred=pred_name)

            para_queries.append(Query(
                id=f"{q.id}_para",
                logical_form=q.logical_form,
                natural_language=para_nl,
                gold_label=q.gold_label,
                query_type="paraphrase",
                paraphrase_group=q.id,
            ))

        return queries + neg_queries + para_queries
