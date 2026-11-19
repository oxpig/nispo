from __future__ import annotations

import re
from dataclasses import dataclass

from ..template_access import template_record_locants, template_record_mol
from .rule_helpers import _has_unsaturated_template_bond


@dataclass(frozen=True, slots=True)
class PartialHydroCandidateFacts:
    template: dict
    match: tuple[int, ...]
    parent_name: str
    hydro_key: tuple
    structural_key: tuple


def partial_hydro_render_set(
    facts: tuple[PartialHydroCandidateFacts, ...],
    *,
    size: int,
) -> tuple[PartialHydroCandidateFacts, ...]:
    if len(facts) <= size:
        return facts
    return tuple(sorted(facts, key=lambda fact: fact.structural_key)[:size])


def retained_partial_hydro_template_saturation_key(template: dict, parent_name: str) -> tuple[int, int]:
    locants = partial_hydro_parent_locants(parent_name)
    if not locants:
        return (2, 0)
    locant_to_template_idx = {
        locant: template_idx
        for template_idx, locant in template_record_locants(template).items()
    }
    template_mol = template_record_mol(template)
    aromatic_count = 0
    unsaturated_count = 0
    for locant in locants:
        template_idx = locant_to_template_idx.get(locant)
        if template_idx is None:
            continue
        atom = template_mol.GetAtomWithIdx(template_idx)
        if atom.GetIsAromatic():
            aromatic_count += 1
        if _has_unsaturated_template_bond(template_mol, template_idx):
            unsaturated_count += 1
    return (aromatic_count, -unsaturated_count)


def partial_hydro_parent_locants(parent_name: str) -> tuple[str, ...]:
    match = re.match(r"^([0-9a-z,]+)-(?:di|tetra|hexa)hydro-", parent_name)
    if not match:
        return ()
    return tuple(locant for locant in match.group(1).split(",") if locant)
