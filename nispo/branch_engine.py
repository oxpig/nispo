from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


BranchRenderer = Callable[[object, int, int], str]
BranchPredicate = Callable[[object, int, int], bool]


@dataclass(frozen=True)
class BranchRule:
    """One substituent-prefix rule within a named OPSIN-inspired family."""

    family: str
    name: str
    renderer: BranchRenderer
    predicate: BranchPredicate | None = None

    def render(self, mol, idx: int, parent_idx: int) -> str:
        if self.predicate is not None and not self.predicate(mol, idx, parent_idx):
            return ""
        return self.renderer(mol, idx, parent_idx)


@dataclass(frozen=True)
class BranchRuleSet:
    """Ordered branch rules for one atom class or generic fallback."""

    family: str
    rules: tuple[BranchRule, ...]

    def render(self, mol, idx: int, parent_idx: int) -> str:
        for rule in self.rules:
            name = rule.render(mol, idx, parent_idx)
            if name:
                return name
        return ""


def first_branch_name(rule_sets: Iterable[BranchRuleSet], mol, idx: int, parent_idx: int) -> str:
    for rule_set in rule_sets:
        name = rule_set.render(mol, idx, parent_idx)
        if name:
            return name
    return ""
