from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from .naming_context import MoleculeContext
from .rule_dispatch import compile_rule_plan, rule_applies


@dataclass(frozen=True, slots=True)
class NameCandidate:
    """A deterministic name proposed by one rule family."""

    name: str
    family: str
    priority: int
    sort_key: tuple = ()


class MoleculeRule(Protocol):
    family: str
    priority: int

    def candidates(self, ctx: MoleculeContext) -> Iterable[NameCandidate]:
        ...


class NamingEngine:
    """OPSIN-inspired dispatcher over independent nomenclature rule families."""

    def __init__(self, Chem, rules: Iterable[MoleculeRule]):
        self._Chem = Chem
        self._rules = tuple(sorted(rules, key=lambda rule: rule.priority))
        self._plans = tuple(compile_rule_plan(rule) for rule in self._rules)

    def name(self, mol) -> str:
        ctx = MoleculeContext(mol, self._Chem)
        best: NameCandidate | None = None
        best_key: tuple | None = None
        for plan in self._plans:
            if not rule_applies(plan, ctx):
                continue
            rule = plan.rule
            for candidate in rule.candidates(ctx):
                if not candidate.name:
                    continue
                candidate_key = _candidate_key(candidate)
                if best_key is None or candidate_key < best_key:
                    best = candidate
                    best_key = candidate_key
            if best is not None and best.priority == rule.priority:
                return best.name
        return best.name if best is not None else ""


def _candidate_key(candidate: NameCandidate) -> tuple:
    return (candidate.priority, candidate.sort_key, candidate.name)
