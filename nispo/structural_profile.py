from __future__ import annotations

from dataclasses import dataclass

from .graph_model import graph_view


@dataclass(frozen=True, slots=True)
class StructuralProfile:
    """Cheap molecule-wide predicates used to gate expensive rule families."""

    ring_count: int
    has_charged_ring_atom: bool
    nonring_multiple_bond_count: int
    nonring_imino_like_count: int
    exocyclic_carbonyl_count: int
    halogenated_alkene_count: int
    nitrile_like_count: int
    sulfone_like_count: int
    nonring_positive_sulfur_count: int
    nonring_positive_nitrogen_count: int
    nonring_positive_phosphorus_count: int
    nonring_neutral_phosphorus_count: int
    nonring_neutral_pnictogen_count: int
    nonring_neutral_group14_count: int
    aromatic_atom_count: int

    @property
    def has_nonring_multiple_bond(self) -> bool:
        return self.nonring_multiple_bond_count > 0

    @property
    def has_imino_skeleton_signal(self) -> bool:
        return self.nonring_imino_like_count > 0

    @property
    def has_functional_skeleton_signal(self) -> bool:
        return (
            self.nitrile_like_count >= 2
            or self.sulfone_like_count > 0
            or self.exocyclic_carbonyl_count > 0
            or self.halogenated_alkene_count > 0
            or self.nonring_multiple_bond_count >= 2
        )

    @property
    def may_prefer_acyclic_skeleton(self) -> bool:
        return (
            self.nonring_multiple_bond_count >= 1
            or self.nitrile_like_count >= 1
            or self.sulfone_like_count >= 1
        )

    @property
    def has_nonring_positive_sulfur(self) -> bool:
        return self.nonring_positive_sulfur_count > 0

    @property
    def has_nonring_positive_nitrogen(self) -> bool:
        return self.nonring_positive_nitrogen_count > 0

    @property
    def has_nonring_positive_phosphorus(self) -> bool:
        return self.nonring_positive_phosphorus_count > 0

    @property
    def has_nonring_neutral_phosphorus(self) -> bool:
        return self.nonring_neutral_phosphorus_count > 0

    @property
    def has_nonring_neutral_pnictogen(self) -> bool:
        return self.nonring_neutral_pnictogen_count > 0

    @property
    def has_nonring_neutral_group14(self) -> bool:
        return self.nonring_neutral_group14_count > 0

    @property
    def has_aromatic_atom(self) -> bool:
        return self.aromatic_atom_count > 0

    @property
    def has_kekulized_polycycle_signal(self) -> bool:
        return self.has_aromatic_atom or self.ring_count >= 3



def structural_profile(mol, Chem=None) -> StructuralProfile:
    """Build a low-cost dispatch profile without substructure matching.

    This profile intentionally contains coarse structural signals, not naming
    decisions. Rule families can use it to avoid running expensive renderers on
    molecules they cannot plausibly name.
    """

    view = graph_view(mol, Chem)
    nonring_multiple = 0
    imino_like = 0
    exocyclic_carbonyl = 0
    halogenated_alkene = 0
    nitrile_like = 0
    sulfone_like = 0
    nonring_positive_sulfur = 0
    nonring_positive_nitrogen = 0
    nonring_positive_phosphorus = 0
    nonring_neutral_phosphorus = 0
    nonring_neutral_pnictogen = 0
    nonring_neutral_group14 = 0
    aromatic_atoms = 0
    for idx, symbol in enumerate(view.symbols):
        if view.aromatic_atoms[idx]:
            aromatic_atoms += 1
        if view.ring_atoms[idx]:
            continue
        if view.formal_charges[idx] > 0:
            if symbol == "S":
                nonring_positive_sulfur += 1
            elif symbol == "N":
                nonring_positive_nitrogen += 1
            elif symbol == "P":
                nonring_positive_phosphorus += 1
        elif symbol == "P" and view.formal_charges[idx] == 0:
            nonring_neutral_phosphorus += 1
            nonring_neutral_pnictogen += 1
        elif symbol in {"As", "Sb", "Bi"} and view.formal_charges[idx] == 0:
            nonring_neutral_pnictogen += 1
        elif symbol in {"Si", "Ge", "Sn", "Pb"} and view.formal_charges[idx] == 0:
            nonring_neutral_group14 += 1

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        begin_symbol = begin.GetSymbol()
        end_symbol = end.GetSymbol()
        order = bond.GetBondTypeAsDouble()
        if begin.IsInRing() and end.IsInRing():
            continue
        if order <= 1.0:
            continue
        nonring_multiple += 1
        symbols = {begin_symbol, end_symbol}
        if symbols == {"C", "O"} and order == 2.0 and (begin.IsInRing() or end.IsInRing()):
            exocyclic_carbonyl += 1
        if symbols == {"C"} and order == 2.0 and (
            _has_halogen_neighbor(begin, end.GetIdx()) or _has_halogen_neighbor(end, begin.GetIdx())
        ):
            halogenated_alkene += 1
        if "N" in symbols and order == 2.0:
            imino_like += 1
        if symbols == {"C", "N"} and order == 3.0:
            nitrile_like += 1
        if "S" in symbols and "O" in symbols:
            sulfone_like += 1

    return StructuralProfile(
        ring_count=view.ring_count,
        has_charged_ring_atom=any(
            in_ring and charge != 0
            for in_ring, charge in zip(view.ring_atoms, view.formal_charges, strict=True)
        ),
        nonring_multiple_bond_count=nonring_multiple,
        nonring_imino_like_count=imino_like,
        exocyclic_carbonyl_count=exocyclic_carbonyl,
        halogenated_alkene_count=halogenated_alkene,
        nitrile_like_count=nitrile_like,
        sulfone_like_count=sulfone_like,
        nonring_positive_sulfur_count=nonring_positive_sulfur,
        nonring_positive_nitrogen_count=nonring_positive_nitrogen,
        nonring_positive_phosphorus_count=nonring_positive_phosphorus,
        nonring_neutral_phosphorus_count=nonring_neutral_phosphorus,
        nonring_neutral_pnictogen_count=nonring_neutral_pnictogen,
        nonring_neutral_group14_count=nonring_neutral_group14,
        aromatic_atom_count=aromatic_atoms,
    )


def _has_halogen_neighbor(atom, excluded_idx: int) -> bool:
    return any(
        neighbor.GetIdx() != excluded_idx and neighbor.GetSymbol() in {"F", "Cl", "Br", "I"}
        for neighbor in atom.GetNeighbors()
    )
