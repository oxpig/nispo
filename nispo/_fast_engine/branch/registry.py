from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


BranchRenderer = Callable[[object, int, int], str] | str
BranchPredicate = Callable[[object, int, int], bool]


@dataclass(frozen=True)
class BranchRuleSpec:
    """One substituent-prefix rule in an ordered branch registry."""

    renderer: BranchRenderer
    predicate: BranchPredicate | None = None

    def render(self, mol, idx: int, parent_idx: int) -> str:
        if self.predicate is not None and not self.predicate(mol, idx, parent_idx):
            return ""
        if isinstance(self.renderer, str):
            return self.renderer
        return self.renderer(mol, idx, parent_idx)


BranchRuleSet = tuple[BranchRuleSpec, ...]


def first_branch_name(rule_sets: Iterable[BranchRuleSet], mol, idx: int, parent_idx: int) -> str:
    for rule_set in rule_sets:
        for rule in rule_set:
            name = rule.render(mol, idx, parent_idx)
            if name:
                return name
    return ""
