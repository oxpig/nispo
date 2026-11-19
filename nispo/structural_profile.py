from __future__ import annotations

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .graph_model import graph_view


@dataclass(frozen=True, slots=True)
class StructuralProfile:
    """Cheap molecule-wide predicates used to gate expensive rule families."""

    ring_count: int
    formal_charge_atom_count: int
    has_charged_ring_atom: bool
    boron_atom_count: int
    iodine_atom_count: int
    nitrogen_atom_count: int
    oxygen_atom_count: int
    sulfur_atom_count: int
    nonring_multiple_bond_count: int
    nonring_imino_like_count: int
    exocyclic_carbonyl_count: int
    halogenated_alkene_count: int
    carbonyl_count: int
    amide_acyl_motif_count: int
    nonring_conjugated_diene_count: int
    nitrile_like_count: int
    sulfone_like_count: int
    nonring_positive_sulfur_count: int
    nonring_positive_nitrogen_count: int
    nonring_positive_phosphorus_count: int
    nonring_neutral_nitrogen_count: int
    nonring_neutral_phosphorus_count: int
    ring_neutral_phosphorus_count: int
    nonring_neutral_pnictogen_count: int
    nonring_neutral_group14_count: int
    sulfur_oxo_retained_parent_signal: bool
    terminal_n_oxide_count: int
    oxygen_oxygen_single_bond_count: int
    oxirane_dione_signal: bool
    aromatic_atom_count: int

    @property
    def has_formal_charge(self) -> bool:
        return self.formal_charge_atom_count > 0

    @property
    def has_nonring_multiple_bond(self) -> bool:
        return self.nonring_multiple_bond_count > 0

    @property
    def has_boron(self) -> bool:
        return self.boron_atom_count > 0

    @property
    def has_iodine(self) -> bool:
        return self.iodine_atom_count > 0

    @property
    def has_nitrogen(self) -> bool:
        return self.nitrogen_atom_count > 0

    @property
    def has_oxygen(self) -> bool:
        return self.oxygen_atom_count > 0

    @property
    def has_sulfur(self) -> bool:
        return self.sulfur_atom_count > 0

    @property
    def has_oxygen_oxygen_single_bond(self) -> bool:
        return self.oxygen_oxygen_single_bond_count > 0

    @property
    def has_oxirane_dione_signal(self) -> bool:
        return self.oxirane_dione_signal

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
            or self.nonring_multiple_bond_count >= 1
        )

    @property
    def has_nitrile_oxide_signal(self) -> bool:
        return self.has_formal_charge and self.nitrile_like_count > 0

    @property
    def has_oxime_sulfonate_nitrile_signal(self) -> bool:
        return self.nitrile_like_count > 0 and self.sulfone_like_count > 0 and self.nonring_imino_like_count > 0

    @property
    def has_terminal_carboxylate_signal(self) -> bool:
        return self.has_formal_charge and self.carbonyl_count > 0

    @property
    def has_amide_acyl_motif(self) -> bool:
        return self.amide_acyl_motif_count > 0

    @property
    def has_dithiocarbamate_signal(self) -> bool:
        return self.sulfur_atom_count >= 2 and self.nonring_neutral_nitrogen_count > 0 and self.nonring_multiple_bond_count > 0

    @property
    def has_small_inorganic_retained_signal(self) -> bool:
        return (
            self.has_formal_charge
            or self.boron_atom_count > 0
            or self.iodine_atom_count > 0
            or 0 < self.nitrogen_atom_count + self.oxygen_atom_count + self.sulfur_atom_count <= 4
            or (self.sulfur_atom_count > 0 and self.nonring_multiple_bond_count > 0)
        )

    @property
    def has_conjugated_carbonyl_diene_signal(self) -> bool:
        return self.carbonyl_count > 0 and self.nonring_conjugated_diene_count > 0

    @property
    def may_prefer_acyclic_skeleton(self) -> bool:
        return (
            self.nonring_multiple_bond_count >= 1
            or self.nitrile_like_count >= 1
            or self.sulfone_like_count >= 1
            or self.exocyclic_carbonyl_count >= 1
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
    def has_nonring_neutral_nitrogen(self) -> bool:
        return self.nonring_neutral_nitrogen_count > 0

    @property
    def has_nonring_neutral_phosphorus(self) -> bool:
        return self.nonring_neutral_phosphorus_count > 0

    @property
    def has_neutral_phosphorus(self) -> bool:
        return self.nonring_neutral_phosphorus_count + self.ring_neutral_phosphorus_count > 0

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

    @property
    def has_sulfur_oxo_retained_parent_signal(self) -> bool:
        return self.sulfur_oxo_retained_parent_signal

    @property
    def has_terminal_n_oxide(self) -> bool:
        return self.terminal_n_oxide_count > 0



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
    carbonyl = 0
    amide_acyl_motifs = 0
    nitrile_like = 0
    sulfone_like = 0
    nonring_positive_sulfur = 0
    nonring_positive_nitrogen = 0
    nonring_positive_phosphorus = 0
    nonring_neutral_nitrogen = 0
    nonring_neutral_phosphorus = 0
    ring_neutral_phosphorus = 0
    nonring_neutral_pnictogen = 0
    nonring_neutral_group14 = 0
    aromatic_atoms = 0
    terminal_n_oxide = 0
    formal_charge_atoms = 0
    boron_atoms = 0
    carbon_atoms = 0
    iodine_atoms = 0
    nitrogen_atoms = 0
    oxygen_atoms = 0
    sulfur_atoms = 0
    oxygen_oxygen_single = 0
    nonring_carbon_double_neighbors: dict[int, set[int]] = {}
    nonring_carbon_single_bonds: list[tuple[int, int]] = []
    amide_neighbor_counts: dict[int, list[int]] = {}
    for idx, symbol in enumerate(view.symbols):
        if symbol == "B":
            boron_atoms += 1
        elif symbol == "C":
            carbon_atoms += 1
        elif symbol == "I":
            iodine_atoms += 1
        elif symbol == "N":
            nitrogen_atoms += 1
        elif symbol == "O":
            oxygen_atoms += 1
        elif symbol == "S":
            sulfur_atoms += 1
        if view.formal_charges[idx] != 0:
            formal_charge_atoms += 1
        if view.aromatic_atoms[idx]:
            aromatic_atoms += 1
        if view.ring_atoms[idx]:
            if symbol == "P" and view.formal_charges[idx] == 0:
                ring_neutral_phosphorus += 1
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
        elif symbol == "N" and view.formal_charges[idx] == 0:
            nonring_neutral_nitrogen += 1
        elif symbol in {"As", "Sb", "Bi"} and view.formal_charges[idx] == 0:
            nonring_neutral_pnictogen += 1
        elif symbol in {"Si", "Ge", "Sn", "Pb"} and view.formal_charges[idx] == 0:
            nonring_neutral_group14 += 1

    for bond in iter_bonds(mol):
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        begin_idx = begin.GetIdx()
        end_idx = end.GetIdx()
        begin_symbol = begin.GetSymbol()
        end_symbol = end.GetSymbol()
        order = bond.GetBondTypeAsDouble()
        if begin_symbol == "C" and begin.GetFormalCharge() == 0:
            counts = amide_neighbor_counts.setdefault(begin_idx, [0, 0, 0, 0])
            _record_amide_acyl_neighbor(counts, end, order)
        if end_symbol == "C" and end.GetFormalCharge() == 0:
            counts = amide_neighbor_counts.setdefault(end_idx, [0, 0, 0, 0])
            _record_amide_acyl_neighbor(counts, begin, order)
        if begin.IsInRing() and end.IsInRing():
            continue
        if order <= 1.0:
            if order == 1.0 and begin_symbol == "O" and end_symbol == "O":
                oxygen_oxygen_single += 1
            if (
                order == 1.0
                and begin_symbol == "C"
                and end_symbol == "C"
                and not begin.IsInRing()
                and not end.IsInRing()
            ):
                nonring_carbon_single_bonds.append((begin.GetIdx(), end.GetIdx()))
            continue
        nonring_multiple += 1
        symbols = {begin_symbol, end_symbol}
        if symbols == {"C", "O"} and order == 2.0:
            carbonyl += 1
            if begin.IsInRing() or end.IsInRing():
                exocyclic_carbonyl += 1
        if (
            symbols == {"C"}
            and order == 2.0
            and not begin.IsInRing()
            and not end.IsInRing()
        ):
            nonring_carbon_double_neighbors.setdefault(begin.GetIdx(), set()).add(end.GetIdx())
            nonring_carbon_double_neighbors.setdefault(end.GetIdx(), set()).add(begin.GetIdx())
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
    for nitrogen in iter_atoms(mol):
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() <= 0:
            continue
        nidx = nitrogen.GetIdx()
        for oxygen in nitrogen.GetNeighbors():
            if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() >= 0:
                continue
            bond = mol.GetBondBetweenAtoms(nidx, oxygen.GetIdx())
            if bond is None or bond.GetBondTypeAsDouble() != 1.0:
                continue
            if sum(1 for nbr in oxygen.GetNeighbors() if nbr.GetAtomicNum() > 1) == 1:
                terminal_n_oxide += 1
    conjugated_diene = sum(
        1
        for begin_idx, end_idx in nonring_carbon_single_bonds
        if any(nidx != end_idx for nidx in nonring_carbon_double_neighbors.get(begin_idx, ()))
        and any(nidx != begin_idx for nidx in nonring_carbon_double_neighbors.get(end_idx, ()))
    )
    amide_acyl_motifs = sum(
        1
        for double_oxygen, single_neutral_nonring_nitrogen, single_carbon, other in amide_neighbor_counts.values()
        if double_oxygen == 1
        and single_neutral_nonring_nitrogen == 1
        and single_carbon == 1
        and other == 0
    )

    return StructuralProfile(
        ring_count=view.ring_count,
        formal_charge_atom_count=formal_charge_atoms,
        has_charged_ring_atom=any(
            in_ring and charge != 0
            for in_ring, charge in zip(view.ring_atoms, view.formal_charges, strict=True)
        ),
        boron_atom_count=boron_atoms,
        iodine_atom_count=iodine_atoms,
        nitrogen_atom_count=nitrogen_atoms,
        oxygen_atom_count=oxygen_atoms,
        sulfur_atom_count=sulfur_atoms,
        nonring_multiple_bond_count=nonring_multiple,
        nonring_imino_like_count=imino_like,
        exocyclic_carbonyl_count=exocyclic_carbonyl,
        halogenated_alkene_count=halogenated_alkene,
        carbonyl_count=carbonyl,
        amide_acyl_motif_count=amide_acyl_motifs,
        nonring_conjugated_diene_count=conjugated_diene,
        nitrile_like_count=nitrile_like,
        sulfone_like_count=sulfone_like,
        nonring_positive_sulfur_count=nonring_positive_sulfur,
        nonring_positive_nitrogen_count=nonring_positive_nitrogen,
        nonring_positive_phosphorus_count=nonring_positive_phosphorus,
        nonring_neutral_nitrogen_count=nonring_neutral_nitrogen,
        nonring_neutral_phosphorus_count=nonring_neutral_phosphorus,
        ring_neutral_phosphorus_count=ring_neutral_phosphorus,
        nonring_neutral_pnictogen_count=nonring_neutral_pnictogen,
        nonring_neutral_group14_count=nonring_neutral_group14,
        sulfur_oxo_retained_parent_signal=_has_sulfur_oxo_retained_parent_signal(mol),
        terminal_n_oxide_count=terminal_n_oxide,
        oxygen_oxygen_single_bond_count=oxygen_oxygen_single,
        oxirane_dione_signal=(
            view.atom_count == 5
            and view.ring_count == 1
            and carbon_atoms == 2
            and oxygen_atoms == 3
        ),
        aromatic_atom_count=aromatic_atoms,
    )


def _record_amide_acyl_neighbor(counts: list[int], neighbor, order: float) -> None:
    symbol = neighbor.GetSymbol()
    if symbol == "O" and order == 2.0:
        counts[0] += 1
    elif symbol == "N" and neighbor.GetFormalCharge() == 0 and not neighbor.IsInRing() and order == 1.0:
        counts[1] += 1
    elif symbol == "C" and order == 1.0:
        counts[2] += 1
    else:
        counts[3] += 1


def _has_sulfur_oxo_retained_parent_signal(mol) -> bool:
    for sulfur in iter_atoms(mol):
        if sulfur.GetSymbol() != "S" or sulfur.IsInRing():
            continue
        facts = _sulfur_oxo_neighbor_facts(mol, sulfur.GetIdx())
        charge = sulfur.GetFormalCharge()
        hydrogens = sulfur.GetTotalNumHs()
        if (
            charge == 1
            and hydrogens == 0
            and facts["double_o"] == 2
            and facts["single_c"] == 1
            and facts["other_heavy"] == 0
        ):
            return True
        if charge != 0:
            continue
        if (
            hydrogens == 1
            and facts["double_o"] == 2
            and facts["single_c"] == 1
            and facts["other_heavy"] == 0
        ):
            return True
        if (
            hydrogens == 0
            and facts["single_c"] == 1
            and facts["terminal_oxido"] == 1
            and facts["double_o"] == 0
            and facts["double_s"] in {0, 1}
            and facts["other_heavy"] == 0
        ):
            return True
        if (
            hydrogens == 1
            and facts["double_o"] == 1
            and facts["single_n"] == 1
            and facts["other_heavy"] == 0
        ):
            return True
        if (
            hydrogens == 2
            and facts["double_o"] == 1
            and facts["double_n"] == 1
            and facts["other_heavy"] == 0
        ):
            return True
        if facts["single_s"] != 1 or facts["other_heavy"] != 0:
            continue
        if hydrogens == 1 and facts["terminal_oxido"] == 0 and facts["double_o"] in {1, 2}:
            return True
        if hydrogens == 0 and facts["double_o"] == 1 and facts["terminal_oxido"] == 1:
            return True
    return False


def _sulfur_oxo_neighbor_facts(mol, sulfur_idx: int) -> dict[str, int]:
    counts = {
        "double_o": 0,
        "terminal_oxido": 0,
        "single_c": 0,
        "single_n": 0,
        "double_n": 0,
        "single_s": 0,
        "double_s": 0,
        "other_heavy": 0,
    }
    for nbr in mol.GetAtomWithIdx(sulfur_idx).GetNeighbors():
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nbr.GetIdx())
        if bond is None:
            continue
        order = bond.GetBondTypeAsDouble()
        symbol = nbr.GetSymbol()
        if symbol == "O" and nbr.GetFormalCharge() == 0 and order == 2.0:
            counts["double_o"] += 1
        elif (
            symbol == "O"
            and nbr.GetFormalCharge() == -1
            and len(nbr.GetNeighbors()) == 1
            and order == 1.0
        ):
            counts["terminal_oxido"] += 1
        elif symbol == "C" and order == 1.0:
            counts["single_c"] += 1
        elif symbol == "N" and nbr.GetFormalCharge() == 0 and order == 1.0:
            counts["single_n"] += 1
        elif symbol == "N" and nbr.GetFormalCharge() == 0 and order == 2.0:
            counts["double_n"] += 1
        elif symbol == "S" and nbr.GetFormalCharge() == 0:
            if order == 1.0:
                counts["single_s"] += 1
            elif order == 2.0:
                counts["double_s"] += 1
            else:
                counts["other_heavy"] += 1
        elif nbr.GetAtomicNum() > 1:
            counts["other_heavy"] += 1
    return counts


def _has_halogen_neighbor(atom, excluded_idx: int) -> bool:
    return any(
        neighbor.GetIdx() != excluded_idx and neighbor.GetSymbol() in {"F", "Cl", "Br", "I"}
        for neighbor in atom.GetNeighbors()
    )
