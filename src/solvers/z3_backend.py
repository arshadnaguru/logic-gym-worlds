"""
src/solvers/z3_backend.py

Z3-based solver for the Horn-FOL track.
Handles: declare sorts/predicates/constants, add rules/facts, check entailment.
"""

from z3 import (
    DeclareSort, Function, BoolSort, Const, ForAll, Exists,
    Implies, And, Or, Not, Solver, sat, unsat,
)
from enum import Enum
from typing import Dict, List, Optional, Tuple


class EntailmentResult(Enum):
    ENTAILS     = "entails"
    CONTRADICTS = "contradicts"
    UNKNOWN     = "unknown"


class Z3Solver:
    """Z3-based solver for Horn-FOL benchmark track."""

    def __init__(self):
        self._entity_sort = DeclareSort("Entity")
        self.solver      = Solver()
        self.predicates: Dict[str, object] = {}   # name -> Z3 Function
        self.constants:  Dict[str, object] = {}   # name -> Z3 Const

    # ── Setup ──────────────────────────────────────────────────────────────

    def reset(self):
        """Clear all solver state (keep sort, recreate everything else)."""
        self._entity_sort = DeclareSort("Entity")
        self.solver       = Solver()
        self.predicates   = {}
        self.constants    = {}

    def declare_predicate(self, name: str, arity: int = 1):
        """Register a predicate with the solver."""
        s = self._entity_sort
        if arity == 1:
            self.predicates[name] = Function(name, s, BoolSort())
        elif arity == 2:
            self.predicates[name] = Function(name, s, s, BoolSort())
        else:
            raise ValueError(f"Arity {arity} not supported (use 1 or 2).")

    def declare_constant(self, name: str):
        """Register a ground entity."""
        self.constants[name] = Const(name, self._entity_sort)

    # ── Assert facts ────────────────────────────────────────────────────────

    def add_fact(self, predicate: str, *args: str):
        """Assert pred(e1, ...) as True."""
        self.solver.add(self._ground(predicate, args))

    def add_negative_fact(self, predicate: str, *args: str):
        """Assert NOT pred(e1, ...) as True."""
        self.solver.add(Not(self._ground(predicate, args)))

    def _ground(self, predicate: str, args: Tuple[str, ...]):
        pred = self.predicates[predicate]
        entities = [self.constants[a] for a in args]
        return pred(*entities)

    # ── Assert rules ────────────────────────────────────────────────────────

    def add_implication(self, antecedent_pred: str, consequent_pred: str):
        """forall X: A(X) -> B(X)"""
        X = Const("X", self._entity_sort)
        A = self.predicates[antecedent_pred]
        B = self.predicates[consequent_pred]
        self.solver.add(ForAll([X], Implies(A(X), B(X))))

    def add_conjunction_implication(
        self, pred_a: str, pred_b: str, pred_c: str
    ):
        """forall X: (A(X) and B(X)) -> C(X)"""
        X = Const("X", self._entity_sort)
        A, B, C = (self.predicates[p] for p in (pred_a, pred_b, pred_c))
        self.solver.add(ForAll([X], Implies(And(A(X), B(X)), C(X))))

    def add_negation_implication(self, antecedent_pred: str, negated_pred: str):
        """forall X: A(X) -> NOT B(X)"""
        X = Const("X", self._entity_sort)
        A = self.predicates[antecedent_pred]
        B = self.predicates[negated_pred]
        self.solver.add(ForAll([X], Implies(A(X), Not(B(X)))))

    def add_raw_formula(self, formula):
        """Add a pre-built Z3 formula directly."""
        self.solver.add(formula)

    # ── Queries ─────────────────────────────────────────────────────────────

    def check_entailment(
        self, predicate: str, *args: str
    ) -> EntailmentResult:
        """
        3-way entailment check for a ground atom.

        ENTAILS     if Theory ∧ ¬φ is UNSAT
        CONTRADICTS if Theory ∧  φ is UNSAT
        UNKNOWN     otherwise
        """
        phi = self._ground(predicate, args)

        # Test ENTAILS
        self.solver.push()
        self.solver.add(Not(phi))
        result = self.solver.check()
        self.solver.pop()
        if result == unsat:
            return EntailmentResult.ENTAILS

        # Test CONTRADICTS
        self.solver.push()
        self.solver.add(phi)
        result = self.solver.check()
        self.solver.pop()
        if result == unsat:
            return EntailmentResult.CONTRADICTS

        return EntailmentResult.UNKNOWN

    def check_satisfiability(self) -> bool:
        """Return True if the current theory is satisfiable."""
        return self.solver.check() == sat

    def add_commitment(self, predicate: str, *args: str, positive: bool = True):
        """Add a model commitment (used for joint-satisfiability check)."""
        atom = self._ground(predicate, args)
        self.solver.add(atom if positive else Not(atom))
