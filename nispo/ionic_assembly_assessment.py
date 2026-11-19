"""Fail-closed facts for P-77 binary salt assembly.

The assessment is a small structural boundary, not an ion namer. Whole-ion
names must already be supplied by independently applicable component routes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal


@dataclass(frozen=True, slots=True)
class IonicComponentEvidence:
    name: str
    charge: int
    identity: str
    organic: bool
    source_atom_ids: frozenset[Hashable]
    named_atom_ids: frozenset[Hashable]


@dataclass(frozen=True, slots=True)
class BinaryAssemblyAssessment:
    status: Literal["admit", "pending", "decline"]
    reason: str
    species_counts: tuple[tuple[int, str, str, int], ...] = ()
    source_atom_ids: frozenset[Hashable] = frozenset()


def assess_binary_ionic_components(
    components: tuple[IonicComponentEvidence, ...] | list[IonicComponentEvidence],
    *,
    expected_source_atom_ids: frozenset[Hashable] | None = None,
) -> BinaryAssemblyAssessment:
    """Validate a binary cation/anion partition and preserve charge ratios.

    An admitted binary name has one independently named component identity on
    each charge side. Repeated copies remain a positive integer multiplicity;
    charge balance is checked over each copy. Multiple distinct ions on one
    side are returned as ``pending`` because P-77's binary construction does
    not supply their mixed-ion citation grammar.
    """
    rows = tuple(components)
    if len(rows) < 2:
        return BinaryAssemblyAssessment("decline", "binary assembly requires at least two charged components")
    seen_atoms: set[Hashable] = set()
    identity_facts: dict[tuple[int, str], tuple[str, bool, int]] = {}
    total_charge = 0
    for row in rows:
        if (not isinstance(row, IonicComponentEvidence)
                or type(row.charge) is not int or row.charge == 0
                or not row.identity or not row.name
                or not row.source_atom_ids
                or row.named_atom_ids != row.source_atom_ids):
            return BinaryAssemblyAssessment("decline", "component identity, charge, or complete ownership is unproved")
        if seen_atoms.intersection(row.source_atom_ids):
            return BinaryAssemblyAssessment("decline", "source atom ownership overlaps")
        seen_atoms.update(row.source_atom_ids)
        total_charge += row.charge
        key = (row.charge, row.identity)
        prior = identity_facts.get(key)
        if prior is None:
            identity_facts[key] = (row.name, row.organic, 1)
        elif prior[:2] != (row.name, row.organic):
            return BinaryAssemblyAssessment("decline", "same component identity has conflicting naming facts")
        else:
            identity_facts[key] = (prior[0], prior[1], prior[2] + 1)
    if expected_source_atom_ids is not None and seen_atoms != expected_source_atom_ids:
        return BinaryAssemblyAssessment("decline", "component atom sets do not partition the source graph")
    if total_charge != 0:
        return BinaryAssemblyAssessment("decline", "component charges are not balanced")
    positive = [key for key in identity_facts if key[0] > 0]
    negative = [key for key in identity_facts if key[0] < 0]
    if len(positive) != 1 or len(negative) != 1:
        return BinaryAssemblyAssessment(
            "pending", "mixed component identities exceed the documented binary P-77 assembly",
            source_atom_ids=frozenset(seen_atoms),
        )
    cation_key, anion_key = positive[0], negative[0]
    cation = identity_facts[cation_key]
    anion = identity_facts[anion_key]
    if cation_key[0] * cation[2] + anion_key[0] * anion[2] != 0:
        return BinaryAssemblyAssessment("decline", "species multiplicities do not balance their charges")
    species = (
        (cation_key[0], cation_key[1], cation[0], cation[2]),
        (anion_key[0], anion_key[1], anion[0], anion[2]),
    )
    return BinaryAssemblyAssessment("admit", "one cation identity and one anion identity; charge-balanced", species,
                                    frozenset(seen_atoms))
