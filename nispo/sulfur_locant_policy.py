"""Mapped source evidence for omission of sulfur suffix-parent locants."""
from __future__ import annotations

from dataclasses import dataclass


RULE = "P-14.3.4.2/P-82.6.1.1/P-82.6.1.2/P-82.6.1.3"


@dataclass(frozen=True, slots=True)
class ParentIsotopeLocantEvidence:
    """Whether the existing mapped descriptor writer cites a parent C locant.

    ``claims`` are the writer's original-atom claims, shaped as
    ``(atom_id, symbol, mass, source_locant, printed_locant)``. This keeps the
    policy tied to source graph facts and the actual component descriptor plan,
    without examining a rendered or parsed name string.
    """

    required: bool
    claims: tuple[tuple, ...]
    rule: str = RULE


def parent_isotope_locant_evidence(mol, Chem, component):
    """Return typed evidence when parent isotope locants can be decided.

    The shared ``_mapped_isotope_descriptor_plan`` is the source of truth for
    what the parent component actually writes. A partial carbon isotope label
    on a multi-carbon parent has a printed numerical locator; complete
    uniform-carbon labeling and a single-carbon parent do not. N-substituent
    branch atoms are embedded separately and therefore cannot force a parent
    suffix locant through this component map.
    """
    from .component_mapping import _mapped_isotope_descriptor_plan

    locants = dict(component.atom_locants)
    if set(locants) != set(component.atoms):
        return None
    plan = _mapped_isotope_descriptor_plan(mol, Chem, component, locants)
    if plan is None or len(plan) != 2:
        return None
    claims = tuple(plan[1])
    if any(not isinstance(row, tuple) or len(row) != 5
           or row[0] not in component.atoms
           or mol.GetAtomWithIdx(row[0]).GetSymbol() != row[1]
           or mol.GetAtomWithIdx(row[0]).GetIsotope() != row[2]
           or row[3] != locants[row[0]]
           for row in claims):
        return None
    carbon_claims = tuple(row for row in claims if row[1] == "C")
    return ParentIsotopeLocantEvidence(
        required=any(row[4] is not None for row in carbon_claims),
        claims=claims,
    )
