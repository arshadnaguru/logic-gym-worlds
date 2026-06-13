"""
src/templates/rule_templates.py

Rule templates for Logic Gym Worlds.
Each template defines a logical form, natural language surface forms,
the operators used, and metadata for OOD splitting.
"""

from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class RuleTemplate:
    """A template for generating logical rules."""

    # Unique identifier
    id: str

    # Logical form with placeholders: P, Q, R for predicates; X, Y for variables
    # e.g. "forall X: (P(X) and Q(X)) -> R(X)"
    logical_form: str

    # Multiple natural language paraphrases
    surface_forms: List[str]

    # Which logical operators this template uses (used for OOD splitting)
    operators: Set[str]

    # 1 (simple) to 5 (complex)
    difficulty: int

    # Number of distinct predicates needed
    num_predicates: int

    # Types of predicates: "unary" or "binary"
    predicate_types: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN OPERATORS: and, or, implies, not, forall, exists
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES: List[RuleTemplate] = [

    # ── Difficulty 1: Simple implication ─────────────────────────────────────
    RuleTemplate(
        id="implication_simple",
        logical_form="forall X: P(X) -> Q(X)",
        surface_forms=[
            "If {x} is {P}, then {x} is {Q}.",
            "All {P} things are {Q}.",
            "Being {P} implies being {Q}.",
            "Every {P} is also {Q}.",
            "Whenever something is {P}, it is {Q}.",
        ],
        operators={"forall", "implies"},
        difficulty=1,
        num_predicates=2,
        predicate_types=["unary", "unary"],
    ),

    RuleTemplate(
        id="negation_simple",
        logical_form="forall X: P(X) -> not Q(X)",
        surface_forms=[
            "If {x} is {P}, then {x} is not {Q}.",
            "Nothing that is {P} can be {Q}.",
            "{P} things are never {Q}.",
            "Being {P} rules out being {Q}.",
        ],
        operators={"forall", "implies", "not"},
        difficulty=1,
        num_predicates=2,
        predicate_types=["unary", "unary"],
    ),

    RuleTemplate(
        id="bidirectional_simple",
        logical_form="forall X: P(X) <-> Q(X)",
        surface_forms=[
            "{x} is {P} if and only if {x} is {Q}.",
            "Something is {P} exactly when it is {Q}.",
            "{P} and {Q} always go together.",
        ],
        operators={"forall", "iff"},
        difficulty=2,
        num_predicates=2,
        predicate_types=["unary", "unary"],
    ),

    # ── Difficulty 2: Conjunction / disjunction ───────────────────────────────
    RuleTemplate(
        id="conjunction_implication",
        logical_form="forall X: (P(X) and Q(X)) -> R(X)",
        surface_forms=[
            "If {x} is both {P} and {Q}, then {x} is {R}.",
            "Anything that is {P} and {Q} must be {R}.",
            "Being {P} and {Q} together implies being {R}.",
        ],
        operators={"forall", "and", "implies"},
        difficulty=2,
        num_predicates=3,
        predicate_types=["unary", "unary", "unary"],
    ),

    RuleTemplate(
        id="disjunction_implication",
        logical_form="forall X: (P(X) or Q(X)) -> R(X)",
        surface_forms=[
            "If {x} is {P} or {Q}, then {x} is {R}.",
            "Either being {P} or being {Q} is enough for {R}.",
            "Every {P} or {Q} thing is {R}.",
        ],
        operators={"forall", "or", "implies"},
        difficulty=2,
        num_predicates=3,
        predicate_types=["unary", "unary", "unary"],
    ),

    RuleTemplate(
        id="implication_disjunction",
        logical_form="forall X: P(X) -> (Q(X) or R(X))",
        surface_forms=[
            "Every {P} is either {Q} or {R}.",
            "If {x} is {P}, then {x} is {Q} or {R}.",
            "{P} things are always {Q} or {R}.",
        ],
        operators={"forall", "implies", "or"},
        difficulty=2,
        num_predicates=3,
        predicate_types=["unary", "unary", "unary"],
    ),

    RuleTemplate(
        id="existential_simple",
        logical_form="exists X: P(X) and Q(X)",
        surface_forms=[
            "There is something that is both {P} and {Q}.",
            "At least one thing is {P} and {Q}.",
            "Some {P} thing is also {Q}.",
        ],
        operators={"exists", "and"},
        difficulty=2,
        num_predicates=2,
        predicate_types=["unary", "unary"],
    ),

    RuleTemplate(
        id="negation_conjunction",
        logical_form="forall X: (P(X) and Q(X)) -> not R(X)",
        surface_forms=[
            "If {x} is both {P} and {Q}, then {x} is not {R}.",
            "Nothing that is {P} and {Q} can be {R}.",
            "{P} and {Q} things are never {R}.",
        ],
        operators={"forall", "and", "implies", "not"},
        difficulty=2,
        num_predicates=3,
        predicate_types=["unary", "unary", "unary"],
    ),

    # ── Difficulty 3: Chained / nested ───────────────────────────────────────
    RuleTemplate(
        id="chain_implication",
        logical_form="forall X: (P(X) -> Q(X)) and (Q(X) -> R(X))",
        surface_forms=[
            "Every {P} is {Q}, and every {Q} is {R}.",
            "{P} implies {Q}, and {Q} implies {R}.",
            "Being {P} leads to {Q}, which leads to {R}.",
        ],
        operators={"forall", "implies", "and"},
        difficulty=3,
        num_predicates=3,
        predicate_types=["unary", "unary", "unary"],
    ),

    RuleTemplate(
        id="negation_disjunction",
        logical_form="forall X: not (P(X) and Q(X))",
        surface_forms=[
            "Nothing can be both {P} and {Q}.",
            "It is impossible to be {P} and {Q} at the same time.",
            "{P} and {Q} are mutually exclusive.",
        ],
        operators={"forall", "not", "and"},
        difficulty=3,
        num_predicates=2,
        predicate_types=["unary", "unary"],
    ),

    RuleTemplate(
        id="existential_negation",
        logical_form="exists X: P(X) and not Q(X)",
        surface_forms=[
            "Some {P} thing is not {Q}.",
            "There exists something that is {P} but not {Q}.",
            "Not every {P} is {Q}.",
        ],
        operators={"exists", "and", "not"},
        difficulty=3,
        num_predicates=2,
        predicate_types=["unary", "unary"],
    ),

    RuleTemplate(
        id="conditional_disjunction_complex",
        logical_form="forall X: P(X) -> (Q(X) and not R(X))",
        surface_forms=[
            "Every {P} is {Q} but not {R}.",
            "If {x} is {P}, then {x} is {Q} and not {R}.",
            "{P} things are always {Q} and never {R}.",
        ],
        operators={"forall", "implies", "and", "not"},
        difficulty=3,
        num_predicates=3,
        predicate_types=["unary", "unary", "unary"],
    ),

    RuleTemplate(
        id="binary_relation_implication",
        logical_form="forall X Y: R(X, Y) -> P(X)",
        surface_forms=[
            "If {x} {R} {y}, then {x} is {P}.",
            "Anything that {R} something must be {P}.",
            "Having the property {R} toward someone makes {x} {P}.",
        ],
        operators={"forall", "implies"},
        difficulty=3,
        num_predicates=2,
        predicate_types=["binary", "unary"],
    ),

    RuleTemplate(
        id="binary_relation_symmetric",
        logical_form="forall X Y: R(X, Y) -> R(Y, X)",
        surface_forms=[
            "If {x} {R} {y}, then {y} {R} {x}.",
            "The relation {R} is symmetric.",
            "Whenever {x} {R} {y}, it follows that {y} {R} {x}.",
        ],
        operators={"forall", "implies"},
        difficulty=3,
        num_predicates=1,
        predicate_types=["binary"],
    ),

    # ── Difficulty 4: Multi-condition ─────────────────────────────────────────
    RuleTemplate(
        id="triple_conjunction_implication",
        logical_form="forall X: (P(X) and Q(X) and R(X)) -> S(X)",
        surface_forms=[
            "If {x} is {P}, {Q}, and {R}, then {x} is {S}.",
            "Being {P}, {Q}, and {R} together implies being {S}.",
            "The combination of {P}, {Q}, and {R} leads to {S}.",
        ],
        operators={"forall", "and", "implies"},
        difficulty=4,
        num_predicates=4,
        predicate_types=["unary", "unary", "unary", "unary"],
    ),

    RuleTemplate(
        id="disjunction_complex_conclusion",
        logical_form="forall X: P(X) -> (Q(X) or (R(X) and S(X)))",
        surface_forms=[
            "Every {P} is either {Q}, or both {R} and {S}.",
            "If {x} is {P}, then {x} is {Q} or ({R} and {S}).",
        ],
        operators={"forall", "implies", "or", "and"},
        difficulty=4,
        num_predicates=4,
        predicate_types=["unary", "unary", "unary", "unary"],
    ),

    RuleTemplate(
        id="universal_existential",
        logical_form="forall X: P(X) -> exists Y: R(X, Y) and Q(Y)",
        surface_forms=[
            "Every {P} has some {Q} that it {R}.",
            "For each {P}, there is a {Q} that it {R}.",
            "Every {P} thing {R} at least one {Q}.",
        ],
        operators={"forall", "exists", "implies", "and"},
        difficulty=4,
        num_predicates=3,
        predicate_types=["unary", "binary", "unary"],
    ),

    RuleTemplate(
        id="contrapositive_chain",
        logical_form="forall X: not P(X) -> (Q(X) or R(X))",
        surface_forms=[
            "Everything that is not {P} is either {Q} or {R}.",
            "If {x} is not {P}, then {x} must be {Q} or {R}.",
            "Non-{P} things are always {Q} or {R}.",
        ],
        operators={"forall", "not", "implies", "or"},
        difficulty=4,
        num_predicates=3,
        predicate_types=["unary", "unary", "unary"],
    ),

    # ── Difficulty 5: Complex ─────────────────────────────────────────────────
    RuleTemplate(
        id="biconditional_conjunction",
        logical_form="forall X: (P(X) and Q(X)) <-> (R(X) and S(X))",
        surface_forms=[
            "{x} is {P} and {Q} if and only if {x} is {R} and {S}.",
            "Being {P} and {Q} is equivalent to being {R} and {S}.",
        ],
        operators={"forall", "iff", "and"},
        difficulty=5,
        num_predicates=4,
        predicate_types=["unary", "unary", "unary", "unary"],
    ),

    RuleTemplate(
        id="nested_universal_negation",
        logical_form="forall X: (P(X) -> Q(X)) and (not P(X) -> R(X))",
        surface_forms=[
            "{P} things are {Q}, and non-{P} things are {R}.",
            "If {x} is {P} then {Q}; otherwise {R}.",
            "Everything is either ({P} and {Q}) or (not {P} and {R}).",
        ],
        operators={"forall", "implies", "and", "not"},
        difficulty=5,
        num_predicates=3,
        predicate_types=["unary", "unary", "unary"],
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # OOD OPERATORS: count, geq, leq, exactly, iff
    # These are HELD OUT — never seen during training
    # ─────────────────────────────────────────────────────────────────────────

    RuleTemplate(
        id="at_least_n",
        logical_form="count(X: P(X)) >= N",
        surface_forms=[
            "At least {n} things are {P}.",
            "There are {n} or more {P} things.",
            "The number of {P} things is at least {n}.",
            "No fewer than {n} entities are {P}.",
        ],
        operators={"count", "geq"},
        difficulty=3,
        num_predicates=1,
        predicate_types=["unary"],
    ),

    RuleTemplate(
        id="at_most_n",
        logical_form="count(X: P(X)) <= N",
        surface_forms=[
            "At most {n} things are {P}.",
            "There are no more than {n} {P} things.",
            "The number of {P} things is at most {n}.",
            "No more than {n} entities are {P}.",
        ],
        operators={"count", "leq"},
        difficulty=3,
        num_predicates=1,
        predicate_types=["unary"],
    ),

    RuleTemplate(
        id="exactly_n",
        logical_form="count(X: P(X)) == N",
        surface_forms=[
            "Exactly {n} things are {P}.",
            "There are precisely {n} {P} things.",
            "The number of {P} things is exactly {n}.",
        ],
        operators={"count", "eq"},
        difficulty=3,
        num_predicates=1,
        predicate_types=["unary"],
    ),

    RuleTemplate(
        id="count_conditional",
        logical_form="count(X: P(X) and Q(X)) >= N",
        surface_forms=[
            "At least {n} things are both {P} and {Q}.",
            "There are {n} or more entities that are {P} and {Q}.",
            "At least {n} {P} things are also {Q}.",
        ],
        operators={"count", "geq", "and"},
        difficulty=4,
        num_predicates=2,
        predicate_types=["unary", "unary"],
    ),

    RuleTemplate(
        id="exactly_one",
        logical_form="count(X: P(X)) == 1",
        surface_forms=[
            "Exactly one thing is {P}.",
            "There is precisely one {P}.",
            "One and only one entity is {P}.",
        ],
        operators={"count", "eq"},
        difficulty=2,
        num_predicates=1,
        predicate_types=["unary"],
    ),

    RuleTemplate(
        id="iff_count",
        logical_form="forall X: P(X) <-> count(Y: R(X, Y)) >= N",
        surface_forms=[
            "{x} is {P} if and only if it {R} at least {n} things.",
            "Being {P} is equivalent to {R} at least {n} others.",
        ],
        operators={"forall", "iff", "count", "geq"},
        difficulty=5,
        num_predicates=2,
        predicate_types=["unary", "binary"],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience lookups
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_BY_ID = {t.id: t for t in TEMPLATES}

TRAIN_OPERATORS = {"forall", "exists", "and", "or", "implies", "not"}

OOD_OPERATORS = {"count", "geq", "leq", "eq", "iff"}

TRAIN_TEMPLATES = [t for t in TEMPLATES if t.operators.issubset(TRAIN_OPERATORS)]
OOD_TEMPLATES   = [t for t in TEMPLATES if t.operators & OOD_OPERATORS]
