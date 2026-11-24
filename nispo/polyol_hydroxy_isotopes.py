"""Typed isotope evidence for a free hydroxy prefix on a mapped host.

The host atom and its parent locant are an attachment fact. The
hydroxy oxygen and any explicit O-bound isotope hydrogen form a separate
component scope with no invented parent locants for O or H.
"""

from __future__ import annotations

from dataclasses import dataclass

from .component_mapping import MappedComponent, _mapped_isotope_descriptor_plan, decorate_component
from .isotope_component_coverage import IsotopeComponentRecord, component_record_from_atom_map


RULE = "P-82.2.1/P-82.3.1/P-82.6.1.2/P-82.6.2"


@dataclass(frozen=True, slots=True)
class HydroxyPrefixDecoration:
    """One free hydroxy prefix and its original graph/evidence bindings.

    `host_binding` connects the independently locanted parent carbon to this
    prefix. `component_record` owns only the hydroxy O and explicit H atoms;
    those atoms receive `None` component locants because the named hydroxy
    unit has no internal numbered positions in this bounded operation.
    `isotope_host_fact` maps each isotope-bearing hydroxy group to the parent
    attachment locant for P-82.5 numbering comparisons without relabeling the
    O or H as parent atoms.
    """

    text: str
    host_binding: tuple[int, str]
    oxygen_atom_id: int
    hydrogen_atom_ids: tuple[int, ...]
    component_atom_ids: frozenset[int]
    isotope_atom_ids: frozenset[int]
    isotope_host_fact: tuple[int, str, tuple[tuple[int, int], ...]] | None
    component_record: IsotopeComponentRecord | None
    rule: str = RULE


def hydroxy_isotope_prefix(
    mol,
    Chem,
    *,
    host_atom_id: int,
    oxygen_atom_id: int,
    host_locant: str,
    component_id: str,
) -> HydroxyPrefixDecoration | None:
    """Render one isolated free-OH prefix and bind its O/H isotope atoms.

    The local isotope tokens are emitted by the existing mapped isotope plan.
    The host locant is returned separately for a typed parent-prefix caller.
    The helper accepts natural OH, O isotope substitution, and explicit
    protium, deuterium, or tritium on O, including their composition in the
    same hydroxy unit. Other explicit H isotope masses and ambiguous/nonterminal
    OH graphs fail closed.
    """
    try:
        if (not isinstance(host_atom_id, int) or isinstance(host_atom_id, bool)
                or not isinstance(oxygen_atom_id, int) or isinstance(oxygen_atom_id, bool)
                or not isinstance(host_locant, str) or not isinstance(component_id, str)
                or not component_id):
            return None
        host = mol.GetAtomWithIdx(host_atom_id)
        oxygen = mol.GetAtomWithIdx(oxygen_atom_id)
        bond = mol.GetBondBetweenAtoms(host_atom_id, oxygen_atom_id)
        if not ((host.GetAtomicNum() == 6 and host_locant.isdigit() and int(host_locant) >= 1)
                or (host.GetAtomicNum() == 7 and host_locant == "N")):
            return None
        if (oxygen.GetAtomicNum() != 8 or bond is None
                or bond.GetBondTypeAsDouble() != 1.0 or oxygen.GetFormalCharge()
                or oxygen.GetNumRadicalElectrons() or int(oxygen.GetChiralTag())
                or oxygen.GetTotalValence() != 2 or host.GetIsotope() < 0
                or oxygen.GetIsotope() < 0):
            return None
        heavy_neighbors = [neighbor.GetIdx() for neighbor in oxygen.GetNeighbors()
                           if neighbor.GetAtomicNum() > 1]
        if heavy_neighbors != [host_atom_id]:
            return None
        hydrogen_ids = tuple(sorted(neighbor.GetIdx() for neighbor in oxygen.GetNeighbors()
                                    if neighbor.GetAtomicNum() == 1))
        if oxygen.GetTotalNumHs(includeNeighbors=True) != 1:
            return None
        for idx in hydrogen_ids:
            hydrogen = mol.GetAtomWithIdx(idx)
            oh_bond = mol.GetBondBetweenAtoms(oxygen_atom_id, idx)
            if (hydrogen.GetFormalCharge() or hydrogen.GetNumRadicalElectrons()
                    or hydrogen.GetTotalValence() != 1 or oh_bond is None
                    or oh_bond.GetBondTypeAsDouble() != 1.0
                    or hydrogen.GetIsotope() not in {0, 1, 2, 3}):
                return None
        isotope_hydrogens = tuple(idx for idx in hydrogen_ids
                                  if mol.GetAtomWithIdx(idx).GetIsotope())
        isotope_ids = frozenset(
            ([oxygen_atom_id] if oxygen.GetIsotope() else []) + list(isotope_hydrogens)
        )
        local_atoms = frozenset((oxygen_atom_id, *hydrogen_ids))
        local_parts = tuple(
            ("H", int(mol.GetAtomWithIdx(idx).GetIsotope()), "", True)
            for idx in isotope_hydrogens
        )
        component = MappedComponent(
            parent="hydroxy",
            atom_locants=tuple((idx, "") for idx in sorted(local_atoms)),
            atoms=local_atoms,
            local_isotope_atoms=frozenset(isotope_hydrogens),
            isotope_descriptor_parts=local_parts,
        )
        text = decorate_component(mol, Chem, component)
        if not text:
            return None

        record = None
        host_fact = None
        if isotope_ids:
            locants = dict(component.atom_locants)
            plan = _mapped_isotope_descriptor_plan(mol, Chem, component, locants)
            if plan is None:
                return None
            _tokens, claims = plan
            # Ordinary atom-bound claims cover isotope oxygen. The H token is
            # a typed anonymous descriptor part, so map it back using the
            # explicit original O-H atom IDs and validate the exact set.
            claimed_ids = {row[0] for row in claims if row[0] is not None}
            expected_oxygen = {oxygen_atom_id} if oxygen.GetIsotope() else set()
            if claimed_ids != expected_oxygen:
                return None
            if len(isotope_hydrogens) != len(local_parts):
                return None
            record = component_record_from_atom_map(
                mol,
                component_id=component_id,
                component_text=text,
                atom_ids=local_atoms,
                atom_locants=tuple((idx, None) for idx in sorted(local_atoms)),
                descriptor_atom_ids=tuple(sorted(isotope_ids)),
                printed_locants={idx: None for idx in isotope_ids},
            )
            profile = tuple(sorted({
                (mol.GetAtomWithIdx(idx).GetAtomicNum(),
                 mol.GetAtomWithIdx(idx).GetIsotope())
                for idx in isotope_ids
            }))
            host_fact = (host_atom_id, host_locant, profile)
        return HydroxyPrefixDecoration(
            text=text,
            host_binding=(host_atom_id, host_locant),
            oxygen_atom_id=oxygen_atom_id,
            hydrogen_atom_ids=hydrogen_ids,
            component_atom_ids=local_atoms,
            isotope_atom_ids=isotope_ids,
            isotope_host_fact=host_fact,
            component_record=record,
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None


polyol_hydroxy_isotope_prefix = hydroxy_isotope_prefix
