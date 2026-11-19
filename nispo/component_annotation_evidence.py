"""Typed isotope and stereochemistry evidence from mapped component writing.

The evidence API shares the descriptor and stereochemistry plans used by
``component_mapping.decorate_component``. It records only atoms within one
``MappedComponent`` scope and never claims that other original isotope atoms
have been covered by the molecule-level name.
"""

from __future__ import annotations

from dataclasses import dataclass

from .component_mapping import (
    MappedComponent,
    _mapped_isotope_descriptor_plan,
    _mapped_stereochemistry_plan,
    decorate_component,
    has_annotations,
)
from .isotope_component_coverage import (
    IsotopeComponentRecord,
    component_record_from_atom_map,
)


@dataclass(frozen=True, slots=True)
class MappedStereoAtomEvidence:
    atom_id: int
    component_locant: str
    descriptor: str


@dataclass(frozen=True, slots=True)
class MappedStereoBondEvidence:
    atom_ids: tuple[int, int]
    component_locant: str
    descriptor: str


@dataclass(frozen=True, slots=True)
class MappedComponentDecoration:
    """A mapped component rendering with evidence local to its own scope."""

    text: str
    isotope_record: IsotopeComponentRecord
    stereo_atoms: tuple[MappedStereoAtomEvidence, ...]
    stereo_bonds: tuple[MappedStereoBondEvidence, ...]
    evidence_scope: frozenset[int]
    evidence_status: str = "component-local"


def decorate_component_with_evidence(
    mol,
    Chem,
    component: MappedComponent,
    *,
    component_id: str,
) -> MappedComponentDecoration | None:
    """Render and bind evidence for ordinary mapped atoms in one component.

    The descriptor claims are emitted by the same internal grouping plan used
    by the string writer. Components with locally written isotope roles,
    anonymous descriptor parts, or labeled embedded children decline because
    those tokens need a separate original-atom evidence contract. The result
    is never a whole-molecule coverage assertion; callers combine disjoint
    component records and independently typed cross-word references.
    """
    if not isinstance(component, MappedComponent):
        return None
    try:
        if (not isinstance(component_id, str) or not component_id
                or not isinstance(component.atoms, frozenset)
                or not component.atoms
                or component.local_isotope_atoms
                or component.isotope_descriptor_parts):
            return None
        if has_annotations(mol, component.embedded_atoms):
            return None
        locants = dict(component.atom_locants)
        isotope_plan = _mapped_isotope_descriptor_plan(mol, Chem, component, locants)
        stereo_plan = _mapped_stereochemistry_plan(mol, Chem, component, locants)
        if isotope_plan is None or stereo_plan is None:
            return None
        text = decorate_component(mol, Chem, component)
        if not text:
            return None
        _parts, primary_claims = isotope_plan
        _stereo_rows, stereo_atom_rows, stereo_bond_rows = stereo_plan
        claim_ids = tuple(row[0] for row in primary_claims)
        # The only primary isotope atoms represented by the ordinary parent
        # writer are the atom-bound claims produced in its grouping plan.
        expected = {idx for idx in component.atoms
                    if mol.GetAtomWithIdx(idx).GetIsotope()}
        if len(claim_ids) != len(set(claim_ids)) or set(claim_ids) != expected:
            return None
        printed = {atom_id: printed_locant
                   for atom_id, _symbol, _mass, _locant, printed_locant
                   in primary_claims}
        record = component_record_from_atom_map(
            mol,
            component_id=component_id,
            component_text=text,
            atom_ids=component.atoms,
            atom_locants=tuple((idx, locant or None)
                               for idx, locant in component.atom_locants),
            descriptor_atom_ids=claim_ids,
            printed_locants=printed,
        )
        stereo_atoms = tuple(MappedStereoAtomEvidence(*row)
                             for row in stereo_atom_rows)
        stereo_bonds = tuple(MappedStereoBondEvidence((a, b), locant, descriptor)
                             for a, b, locant, descriptor in stereo_bond_rows)
        return MappedComponentDecoration(
            text=text,
            isotope_record=record,
            stereo_atoms=stereo_atoms,
            stereo_bonds=stereo_bonds,
            evidence_scope=component.atoms,
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None

