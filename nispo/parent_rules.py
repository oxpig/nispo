from __future__ import annotations

import re
from bisect import insort
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from functools import lru_cache

from .branch_rules import branch_prefix
from .branch.functional_groups import _is_nitrile_carbon
from .branch.hetero_atoms import _imine_carbon_ylidene
from .constants import HALO, MULT, ROOTS
from .cycle_utils import ordered_ring, ring_system_extends_core
from .formatting import format_prefixes as _format_prefixes
from .formatting import join_prefix_parent as _join_prefix_parent
from .formatting import loc_unsat
from .formatting import preferred_parent_stem as _preferred_parent_stem
from .formatting import unsaturated_parent as _unsaturated_parent
from .molecule_cache import mol_cache_get, mol_cache_set
from .fused_template_families import (
    candidate_generated_fused_templates as _candidate_generated_fused_templates,
    curated_fused_templates as _curated_fused_templates,
    generated_fusion_preference as _generated_fusion_preference,
    hydrocarbon_replacement_parent_templates_for_graph as _hydrocarbon_replacement_parent_templates_for_graph,
    retained_topology_replacement_parent_templates_for_graph as _retained_topology_replacement_parent_templates_for_graph,
)
from .scaffold_catalog import parent_scaffold_catalog
from .ring_resource_rules import resource_ring_parent_candidates as _resource_ring_parent_candidates
from .ring_rules import (
    adjust_ring_parent_for_charge as _adjust_ring_parent_for_charge,
    ring_parent_candidates as _ring_parent_candidates,
)
from .spiro_component_rules import component_spiro_parent_candidates
from .rule_dispatch import (
    RuleSpec,
    acyclic,
    acyclic_preference_signal,
    charged_ring_atom,
    conjugated_carbonyl_diene_signal,
    exact_ring_system,
    functional_skeleton_signal,
    generic_bicyclic_topology,
    generic_polycyclic_topology,
    imino_skeleton_signal,
    kekulized_polycycle_signal,
    min_ring_count,
    min_ring_system_ring_count,
    no_charged_ring_atom,
    no_exocyclic_charged_fused_parent,
    nonring_neutral_group14,
    nonring_neutral_pnictogen,
    nonring_neutral_phosphorus,
    nonring_positive_nitrogen,
    nonring_positive_phosphorus,
    nonring_positive_sulfur,
    spiro_topology,
)
from .rule_priority import parent_priority
from .rule_engine import NameCandidate
from .simple_prefixes import simple_external_prefix
from .stereochemistry import assigned_atom_stereochemistry
from .template_access import template_record_locants, template_record_mol, template_record_name
from .template_index import ring_system_with_graph_signatures


_MAX_RETAINED_RING_SCAFFOLD_CANDIDATES = 16
_RETAINED_RING_SCAFFOLD_TEMPLATE_FACT_SET_SIZE = 1
_RETAINED_RING_SCAFFOLD_RENDER_SET_SIZE = 16
_RETAINED_FUSED_RENDER_SET_SIZE = 12
_SINGLE_RING_PARENT_RING_FACT_SET_SIZE = 1
_GENERATED_PARTIAL_HYDRO_RENDER_SET_SIZE = 1
_GENERATED_FUSED_RENDER_SET_SIZE = 8
_MAX_GENERATED_FUSED_PARENT_CANDIDATES = 32
_SKELETAL_REPLACEMENT_FACT_SCAN_SIZE = 32
_SKELETAL_REPLACEMENT_RENDER_SET_SIZE = 12
_RETAINED_TOPOLOGY_REPLACEMENT_FACT_SCAN_SIZE = 32
_RETAINED_TOPOLOGY_REPLACEMENT_RENDER_SET_SIZE = 2
_MAX_RETAINED_TOPOLOGY_REPLACEMENT_CANDIDATES = 16


def _parenthesized_if_complex(name: str) -> str:
    if name.startswith("(") and name.endswith(")"):
        return name
    return f"({name})" if "-" in name or name.startswith("(") else name


def _simple_or_branch_prefix(mol, idx: int, parent_idx: int) -> str:
    return simple_external_prefix(mol, idx, parent_idx) or branch_prefix(mol, idx, parent_idx)


class AcyclicParentRule:
    family = "acyclic-parent"
    priority = parent_priority("acyclic-parent")
    acyclic_only = True

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() != 0:
            return
        name_record = ctx.acyclic_name_record()
        name = name_record.name
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=_heavy_atom_indices(mol),
                stereo_locants=_complete_acyclic_stereo_locants(ctx, name, name_record.stereo_locants),
                estimated_length=len(name),
            )


class SmallInorganicRetainedParentRule:
    family = "small-inorganic-retained-parent"
    priority = parent_priority("charged-inorganic-retained-parent")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        name = _nitryl_halide_name(mol, self._Chem)
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )
            return
        name = _sulfonyl_azanide_name(mol, self._Chem)
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )


def _nitryl_halide_name(mol, Chem) -> str:
    if mol.GetNumAtoms() != 4:
        return ""
    nitrogen_atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "N" and atom.GetFormalCharge() == 1 and not atom.IsInRing()
    ]
    if len(nitrogen_atoms) != 1:
        return ""
    nitrogen = nitrogen_atoms[0]
    nidx = nitrogen.GetIdx()
    double_oxo = 0
    oxido = 0
    halide = ""
    for nbr in nitrogen.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(nidx, nbr.GetIdx())
        if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxo += 1
        elif nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == -1 and bond.GetBondType() == Chem.BondType.SINGLE:
            oxido += 1
        elif nbr.GetSymbol() in MAGNESIUM_HALIDE_NAMES and bond.GetBondType() == Chem.BondType.SINGLE:
            halide = MAGNESIUM_HALIDE_NAMES[nbr.GetSymbol()]
        else:
            return ""
    if double_oxo != 1 or oxido != 1 or not halide:
        return ""
    return f"nitryl {halide}"


def _sulfonyl_azanide_name(mol, Chem) -> str:
    nitrogen_atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "N" and atom.GetFormalCharge() == -1 and not atom.IsInRing()
    ]
    if len(nitrogen_atoms) != 1:
        return ""
    nitrogen = nitrogen_atoms[0]
    nidx = nitrogen.GetIdx()
    sulfur_neighbors = []
    for nbr in nitrogen.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(nidx, nbr.GetIdx())
        if nbr.GetSymbol() != "S" or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        sulfur_neighbors.append(nbr.GetIdx())
    if len(sulfur_neighbors) not in {1, 2}:
        return ""
    if len(sulfur_neighbors) + nitrogen.GetTotalNumHs() != 2:
        return ""
    units = []
    covered = {nidx}
    for sulfur_idx in sulfur_neighbors:
        unit, unit_atoms = _sulfonyl_azanide_unit(mol, Chem, sulfur_idx, nidx)
        if not unit:
            return ""
        units.append(unit)
        covered.update(unit_atoms)
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    if len(units) == 1:
        return units[0] + "azanide"
    counts = Counter(units)
    if len(counts) == 1:
        unit = next(iter(counts))
        return f"bis({unit})azanide"
    return "".join(f"[{unit}]" for unit in sorted(units)) + "azanide"


def _sulfonyl_azanide_unit(mol, Chem, sulfur_idx: int, nitrogen_idx: int) -> tuple[str, set[int]]:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0 or sulfur.IsInRing():
        return "", set()
    double_oxygens = []
    substituent_idx = None
    covered = {sulfur_idx, nitrogen_idx}
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygens.append(nidx)
            covered.add(nidx)
        elif bond.GetBondType() == Chem.BondType.SINGLE and substituent_idx is None:
            substituent_idx = nidx
        else:
            return "", set()
    if len(double_oxygens) != 2 or substituent_idx is None:
        return "", set()
    substituent = mol.GetAtomWithIdx(substituent_idx)
    if substituent.GetSymbol() in HALO:
        covered.add(substituent_idx)
        return HALO[substituent.GetSymbol()] + "sulfonyl", covered
    if substituent.GetSymbol() != "C":
        return "", set()
    prefix = _simple_or_branch_prefix(mol, substituent_idx, sulfur_idx)
    if not prefix:
        return "", set()
    covered.update(_collect_component_atoms(mol, substituent_idx, sulfur_idx))
    return _strip_outer_parentheses(prefix) + "sulfonyl", covered


class SulfaniumParentRule:
    family = "sulfanium-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_positive_sulfur = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        sulfur_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "S" and atom.GetFormalCharge() > 0 and not atom.IsInRing()
        ]
        if len(sulfur_atoms) != 1:
            return
        sulfur = sulfur_atoms[0]
        sidx = sulfur.GetIdx()
        neighbors = [nbr.GetIdx() for nbr in sulfur.GetNeighbors()]
        implicit_hydrogens = sulfur.GetTotalNumHs()
        if len(neighbors) + implicit_hydrogens != 3 or len(neighbors) not in {2, 3}:
            return
        if any(mol.GetAtomWithIdx(nidx).GetSymbol() not in {"C", "S"} for nidx in neighbors):
            return
        if any(
            mol.GetBondBetweenAtoms(sidx, nidx).GetBondType() != self._Chem.BondType.SINGLE
            for nidx in neighbors
        ):
            return
        names = []
        for nidx in neighbors:
            prefix = _simple_or_branch_prefix(mol, nidx, sidx)
            if not prefix:
                return
            names.append(prefix)
        name = _sulfanium_name(names)
        if name:
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class PhosphaniumParentRule:
    family = "phosphanium-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_positive_phosphorus = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        phosphorus_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "P" and atom.GetFormalCharge() > 0 and not atom.IsInRing()
        ]
        if len(phosphorus_atoms) != 1:
            return
        phosphorus = phosphorus_atoms[0]
        pidx = phosphorus.GetIdx()
        neighbors = [nbr.GetIdx() for nbr in phosphorus.GetNeighbors()]
        implicit_hydrogens = phosphorus.GetTotalNumHs()
        if len(neighbors) + implicit_hydrogens != 4 or len(neighbors) not in {3, 4}:
            return
        if any(mol.GetAtomWithIdx(nidx).GetSymbol() == "P" for nidx in neighbors):
            return
        if any(
            mol.GetBondBetweenAtoms(pidx, nidx).GetBondType() != self._Chem.BondType.SINGLE
            for nidx in neighbors
        ):
            return
        names = []
        for nidx in neighbors:
            prefix = _simple_or_branch_prefix(mol, nidx, pidx)
            if not prefix:
                return
            names.append(prefix)
        name = _phosphanium_name(names)
        if name:
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class AzaniumParentRule:
    family = "azanium-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_positive_nitrogen = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        nitrogen_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "N" and atom.GetFormalCharge() > 0 and not atom.IsInRing()
        ]
        if len(nitrogen_atoms) != 1:
            return
        nitrogen = nitrogen_atoms[0]
        nidx = nitrogen.GetIdx()
        neighbors = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors()]
        implicit_hydrogens = nitrogen.GetTotalNumHs()
        if len(neighbors) + implicit_hydrogens != 4 or not 1 <= len(neighbors) <= 4:
            return
        if any(mol.GetAtomWithIdx(nbr_idx).GetSymbol() not in {"C", "Si"} for nbr_idx in neighbors):
            return
        if any(
            mol.GetBondBetweenAtoms(nidx, nbr_idx).GetBondType() != self._Chem.BondType.SINGLE
            for nbr_idx in neighbors
        ):
            return
        names = []
        for nbr_idx in neighbors:
            prefix = _simple_or_branch_prefix(mol, nbr_idx, nidx)
            if not prefix:
                return
            names.append(prefix)
        name = _azanium_name(names)
        if name:
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class NeutralAmineParentRule:
    family = "neutral-amine-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_neutral_pnictogen = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        nitrogen_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "N"
            and atom.GetFormalCharge() == 0
            and not atom.IsInRing()
        ]
        if len(nitrogen_atoms) != 1:
            return
        nitrogen = nitrogen_atoms[0]
        nidx = nitrogen.GetIdx()
        neighbors = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors()]
        if len(neighbors) != 3 or nitrogen.GetTotalNumHs() != 0:
            return
        if any(mol.GetAtomWithIdx(nbr_idx).GetSymbol() not in {"C", "Si"} for nbr_idx in neighbors):
            return
        if any(
            mol.GetBondBetweenAtoms(nidx, nbr_idx).GetBondType() != self._Chem.BondType.SINGLE
            for nbr_idx in neighbors
        ):
            return
        if any(_is_carbonyl_neighbor(mol, nbr_idx, nidx) for nbr_idx in neighbors):
            return
        names = []
        for nbr_idx in neighbors:
            prefix = _simple_or_branch_prefix(mol, nbr_idx, nidx)
            if not prefix:
                return
            names.append(prefix)
        name = _pnictogen_hydride_name(names, "amine")
        if name:
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


def _is_carbonyl_neighbor(mol, carbon_idx: int, nitrogen_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(carbon_idx)
    if atom.GetSymbol() != "C":
        return False
    return any(
        nbr.GetIdx() != nitrogen_idx
        and nbr.GetSymbol() == "O"
        and mol.GetBondBetweenAtoms(carbon_idx, nbr.GetIdx()).GetBondTypeAsDouble() == 2.0
        for nbr in atom.GetNeighbors()
    )


PNICTOGEN_HYDRIDE_PARENTS = {
    "P": "phosphane",
    "As": "arsane",
    "Sb": "stibane",
    "Bi": "bismuthane",
}


GROUP14_HYDRIDE_PARENTS = {
    "Si": "silane",
    "Ge": "germane",
    "Sn": "stannane",
    "Pb": "plumbane",
}


MAGNESIUM_HALIDE_NAMES = {
    "F": "fluoride",
    "Cl": "chloride",
    "Br": "bromide",
    "I": "iodide",
}


class Group14HydrideParentRule:
    family = "group14-hydride-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_neutral_group14 = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        centers = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() in GROUP14_HYDRIDE_PARENTS
            and atom.GetFormalCharge() == 0
            and not atom.IsInRing()
        ]
        if len(centers) != 1:
            return
        center = centers[0]
        cidx = center.GetIdx()
        neighbors = [nbr.GetIdx() for nbr in center.GetNeighbors()]
        implicit_hydrogens = center.GetTotalNumHs()
        if len(neighbors) + implicit_hydrogens != 4 or not 1 <= len(neighbors) <= 4:
            return
        if any(
            mol.GetBondBetweenAtoms(cidx, nidx).GetBondType() != self._Chem.BondType.SINGLE
            for nidx in neighbors
        ):
            return
        carbon_names = []
        carbon_neighbor_indices = []
        halo_names = []
        hydroxy_count = 0
        for nidx in neighbors:
            neighbor = mol.GetAtomWithIdx(nidx)
            symbol = neighbor.GetSymbol()
            if symbol == "C":
                prefix = _simple_or_branch_prefix(mol, nidx, cidx)
                if not prefix:
                    return
                carbon_names.append(prefix)
                carbon_neighbor_indices.append(nidx)
            elif symbol in HALO:
                halo_names.append(HALO[symbol])
            elif (
                symbol == "O"
                and center.GetSymbol() == "Si"
                and neighbor.GetFormalCharge() == 0
                and len(neighbor.GetNeighbors()) == 1
            ):
                hydroxy_count += 1
            else:
                return
        if hydroxy_count and halo_names:
            return
        if _has_organophosphorus_oxide_center(mol, self._Chem):
            return
        if any(_is_charged_ring_ligand(mol, nidx) for nidx in carbon_neighbor_indices):
            return
        if _should_defer_group14_hydride_to_polycycle(ctx, carbon_names):
            return
        parent = GROUP14_HYDRIDE_PARENTS[center.GetSymbol()]
        if hydroxy_count:
            name = _group14_hydroxyhydride_name(carbon_names, parent, hydroxy_count)
        elif halo_names:
            name = _group14_halohydride_name(carbon_names, halo_names, parent)
        else:
            name = _group14_hydride_name(carbon_names, parent)
        if name:
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


def _should_defer_group14_hydride_to_polycycle(ctx, carbon_names: Sequence[str]) -> bool:
    if ctx.profile.ring_count < 2 or not ctx.has_generic_polycyclic_topology:
        return False
    if not ctx.polycyclic_parent_records("kekulized"):
        return False
    return any(
        term in name
        for name in carbon_names
        for term in (
            "benzimidazol",
            "isoindolin",
            "bicyclo[",
            "tricyclo[",
            "diazabicyclo[",
            "azabicyclo[",
            "oxabicyclo[",
            "thiabicyclo[",
            "phosphabicyclo[",
        )
    )


def _has_organophosphorus_oxide_center(mol, Chem) -> bool:
    for phosphorus in mol.GetAtoms():
        if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0 or phosphorus.IsInRing():
            continue
        pidx = phosphorus.GetIdx()
        double_oxygen_count = 0
        carbon_neighbor_count = 0
        unsupported = False
        for nbr in phosphorus.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(pidx, nbr.GetIdx())
            if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                carbon_neighbor_count += 1
            elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygen_count += 1
            elif (
                nbr.GetSymbol() == "O"
                and bond.GetBondType() == Chem.BondType.SINGLE
                and nbr.GetFormalCharge() == 0
                and len(nbr.GetNeighbors()) == 1
            ):
                continue
            else:
                unsupported = True
                break
        if not unsupported and double_oxygen_count == 1 and carbon_neighbor_count:
            return True
    return False


def _is_charged_ring_ligand(mol, atom_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if not atom.IsInRing():
        return False
    for ring in mol.GetRingInfo().AtomRings():
        if atom_idx in ring and any(mol.GetAtomWithIdx(idx).GetFormalCharge() > 0 for idx in ring):
            return True
    return False


class PhosphorusAcidParentRule:
    family = "phosphorus-acid-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_neutral_phosphorus = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        phosphorus_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "P" and atom.GetFormalCharge() == 0 and not atom.IsInRing()
        ]
        if len(phosphorus_atoms) != 1:
            return
        phosphorus = phosphorus_atoms[0]
        pidx = phosphorus.GetIdx()
        carbon_neighbors = []
        double_oxygen_count = 0
        hydroxy_count = 0
        for nbr in phosphorus.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(pidx, nidx)
            if nbr.GetSymbol() == "C" and bond.GetBondType() == self._Chem.BondType.SINGLE:
                carbon_neighbors.append(nidx)
            elif nbr.GetSymbol() == "O" and bond.GetBondType() == self._Chem.BondType.DOUBLE:
                double_oxygen_count += 1
            elif (
                nbr.GetSymbol() == "O"
                and bond.GetBondType() == self._Chem.BondType.SINGLE
                and nbr.GetFormalCharge() == 0
                and len(nbr.GetNeighbors()) == 1
                and nbr.GetTotalNumHs() == 1
            ):
                hydroxy_count += 1
            else:
                return
        if len(carbon_neighbors) != 1 or double_oxygen_count != 1:
            return
        implicit_hydrogens = phosphorus.GetTotalNumHs()
        if hydroxy_count == 2 and implicit_hydrogens == 0:
            parent = "phosphonic acid"
        elif hydroxy_count == 1 and implicit_hydrogens == 1:
            parent = "phosphinic acid"
        else:
            return
        prefix = _simple_or_branch_prefix(mol, carbon_neighbors[0], pidx)
        if not prefix:
            return
        if _needs_substituent_parentheses(prefix):
            prefix = f"({prefix})"
        name = prefix + parent
        yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class PhosphiteEsterParentRule:
    family = "phosphite-ester-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_neutral_phosphorus = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        for name in _phosphite_ester_names(ctx.mol, self._Chem):
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class OrganophosphorusOxideParentRule:
    family = "organophosphorus-oxide-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_neutral_phosphorus = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        phosphorus_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "P" and atom.GetFormalCharge() == 0 and not atom.IsInRing()
        ]
        if len(phosphorus_atoms) != 1:
            return
        phosphorus = phosphorus_atoms[0]
        pidx = phosphorus.GetIdx()
        carbon_neighbors = []
        double_oxygen_count = 0
        hydroxy_count = 0
        for nbr in phosphorus.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(pidx, nidx)
            if nbr.GetSymbol() == "C" and bond.GetBondType() == self._Chem.BondType.SINGLE:
                carbon_neighbors.append(nidx)
            elif nbr.GetSymbol() == "O" and bond.GetBondType() == self._Chem.BondType.DOUBLE:
                double_oxygen_count += 1
            elif (
                nbr.GetSymbol() == "O"
                and bond.GetBondType() == self._Chem.BondType.SINGLE
                and nbr.GetFormalCharge() == 0
                and len(nbr.GetNeighbors()) == 1
                and nbr.GetTotalNumHs() == 1
            ):
                hydroxy_count += 1
            else:
                return
        if double_oxygen_count == 2 and hydroxy_count == 0 and len(carbon_neighbors) == 1:
            prefix = _simple_or_branch_prefix(mol, carbon_neighbors[0], pidx)
            if not prefix:
                return
            if _needs_substituent_parentheses(prefix):
                prefix = f"({prefix})"
            name = prefix + "phosphanedione"
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))
            return
        if double_oxygen_count != 1 or not carbon_neighbors:
            return
        names = []
        for nidx in carbon_neighbors:
            prefix = _simple_or_branch_prefix(mol, nidx, pidx)
            if not prefix:
                return
            names.append(prefix)
        if hydroxy_count == 0:
            parent = "phosphane oxide"
        elif hydroxy_count == 1 and len(carbon_neighbors) == 2:
            parent = "phosphinic acid"
        else:
            return
        prefix_text = _pnictogen_hydride_name(names, "")
        if prefix_text:
            name = prefix_text + parent
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class PhosphoraneParentRule:
    family = "phosphorane-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_neutral_phosphorus = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        phosphorus_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "P" and atom.GetFormalCharge() == 0 and not atom.IsInRing()
        ]
        if len(phosphorus_atoms) != 1:
            return
        phosphorus = phosphorus_atoms[0]
        pidx = phosphorus.GetIdx()
        carbon_names = []
        halo_names = []
        for nbr in phosphorus.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(pidx, nidx)
            if bond.GetBondType() != self._Chem.BondType.SINGLE:
                return
            if nbr.GetSymbol() == "C":
                prefix = _simple_or_branch_prefix(mol, nidx, pidx)
                if not prefix:
                    return
                carbon_names.append(prefix)
            elif nbr.GetSymbol() in HALO:
                halo_names.append(HALO[nbr.GetSymbol()])
            else:
                return
        if len(carbon_names) + len(halo_names) + phosphorus.GetTotalNumHs() != 5:
            return
        if phosphorus.GetTotalNumHs() != 0 or not carbon_names:
            return
        name = _phosphorane_parent_name(carbon_names, halo_names)
        if name:
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


def _phosphorane_parent_name(carbon_names: Sequence[str], halo_names: Sequence[str]) -> str:
    parts = []
    halo_counts = Counter(halo_names)
    for name in sorted(halo_counts):
        count = halo_counts[name]
        multiplier = MULT.get(count, "") if count > 1 else ""
        parts.append(multiplier + name)

    carbon_counts = Counter(carbon_names)
    for name in sorted(carbon_counts):
        count = carbon_counts[name]
        is_complex = _needs_substituent_parentheses(name)
        if count == 1:
            term = f"({name})" if is_complex or halo_names else name
        elif is_complex:
            multiplier = _PHOSPHORANE_COMPLEX_MULTIPLIERS.get(count, "")
            term = f"{multiplier}({name})" if multiplier else ""
        else:
            term = (MULT.get(count, "") if count > 1 else "") + name
        if not term:
            return ""
        parts.append(term)
    return "".join(parts) + "phosphorane" if parts else ""


_PHOSPHORANE_COMPLEX_MULTIPLIERS = {
    2: "bis",
    3: "tris",
    4: "tetrakis",
    5: "pentakis",
}


class OrganomagnesiumHalideParentRule:
    family = "organomagnesium-halide-parent"
    priority = parent_priority("senior-functional-exception")
    acyclic_only = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() != 0 or sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != 0:
            return
        magnesium_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "Mg" and atom.GetFormalCharge() == 0 and not atom.IsInRing()
        ]
        if len(magnesium_atoms) != 1:
            return
        magnesium = magnesium_atoms[0]
        midx = magnesium.GetIdx()
        carbon_neighbors = []
        halogen_names = []
        for nbr in magnesium.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(midx, nidx)
            if bond.GetBondType() != self._Chem.BondType.SINGLE:
                return
            if nbr.GetSymbol() == "C":
                carbon_neighbors.append(nidx)
            elif nbr.GetSymbol() in MAGNESIUM_HALIDE_NAMES:
                halogen_names.append(MAGNESIUM_HALIDE_NAMES[nbr.GetSymbol()])
            else:
                return
        if len(carbon_neighbors) != 1 or len(halogen_names) != 1:
            return
        prefix = _simple_or_branch_prefix(mol, carbon_neighbors[0], midx)
        if not prefix:
            return
        if _needs_substituent_parentheses(prefix):
            prefix = f"({prefix})"
        name = f"{prefix}magnesium {halogen_names[0]}"
        yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class OrganoboraneParentRule:
    family = "organoborane-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        for boron in mol.GetAtoms():
            if (
                boron.GetSymbol() != "B"
                or boron.GetFormalCharge() != 0
                or boron.IsInRing()
            ):
                continue
            boron_idx = boron.GetIdx()
            carbon_neighbors = []
            for nbr in boron.GetNeighbors():
                if nbr.GetAtomicNum() <= 1:
                    continue
                bond = mol.GetBondBetweenAtoms(boron_idx, nbr.GetIdx())
                if (
                    nbr.GetSymbol() != "C"
                    or bond is None
                    or bond.GetBondType() != self._Chem.BondType.SINGLE
                ):
                    break
                carbon_neighbors.append(nbr.GetIdx())
            else:
                if not 1 <= len(carbon_neighbors) <= 3:
                    continue
                if len(carbon_neighbors) + boron.GetTotalNumHs() != 3:
                    continue
                ligand_names = []
                covered = {boron_idx}
                for nidx in carbon_neighbors:
                    name = _simple_or_branch_prefix(mol, nidx, boron_idx)
                    if not name:
                        break
                    ligand_names.append(name)
                    covered.update(_collect_component_atoms(mol, nidx, boron_idx))
                else:
                    if covered != atom_set:
                        continue
                    name = _organoborane_parent_name(ligand_names)
                    if name:
                        yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


def _organoborane_parent_name(ligand_names: Sequence[str]) -> str:
    if not 1 <= len(ligand_names) <= 3:
        return ""
    counts = Counter(ligand_names)
    parts = []
    for ligand, count in sorted(counts.items(), key=lambda item: item[0]):
        unit = _organoborane_ligand_unit(ligand, count)
        if not unit:
            return ""
        parts.append(unit)
    return "".join(parts) + "borane"


def _organoborane_ligand_unit(ligand: str, count: int) -> str:
    ligand = _strip_outer_parentheses(ligand)
    if count == 1:
        return _parenthesized_if_complex(ligand)
    multiplier = MULT.get(count)
    if not multiplier:
        return ""
    if multiplier == "di":
        return f"bis({ligand})"
    if multiplier == "tri":
        return f"tris({ligand})"
    return f"{multiplier}({ligand})"


class BenziodoxoleRetainedParentRule:
    family = "benziodoxole-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem
        self._lactone = Chem.MolFromSmarts("O=C1O[I]([#6])c2ccccc21")
        self._geminal = Chem.MolFromSmarts("[#6]C1([#6])O[I]([#6])c2ccccc21")
        self._iodoxy_lactone = Chem.MolFromSmarts("O=C1O[I](=O)(O)c2ccccc21")
        self._hydride = Chem.MolFromSmarts("c1ccc2c(c1)CO[I]2")

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        for pattern, namer in (
            (self._iodoxy_lactone, _benziodoxole_iodoxy_lactone_name),
            (self._lactone, _benziodoxole_lactone_name),
            (self._geminal, _benziodoxole_geminal_name),
            (self._hydride, _benziodoxole_hydride_name),
        ):
            if pattern is None:
                continue
            for match in mol.GetSubstructMatches(pattern):
                name = namer(mol, self._Chem, match)
                if not name:
                    continue
                yield NameCandidate(
                    name,
                    self.family,
                    self.priority,
                    parent_atoms=set(range(atom_count)),
                    sort_key=(len(name), name),
                )
                return


def _benziodoxole_lactone_name(mol, Chem, match: Sequence[int]) -> str:
    iodine_idx = match[3]
    substituent_idx = match[4]
    covered = set(match)
    covered.update(_collect_component_atoms(mol, substituent_idx, iodine_idx))
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    prefix = _simple_or_branch_prefix(mol, substituent_idx, iodine_idx)
    if not prefix:
        return ""
    return f"1-{_parenthesized_if_complex(_strip_outer_parentheses(prefix))}-1,2-benziodoxol-3-one"


def _benziodoxole_geminal_name(mol, Chem, match: Sequence[int]) -> str:
    ring_carbon_idx = match[1]
    iodine_idx = match[4]
    iodine_substituent_idx = match[5]
    ring_substituent_indices = (match[0], match[2])
    covered = set(match)
    iodine_prefix = _simple_or_branch_prefix(mol, iodine_substituent_idx, iodine_idx)
    if not iodine_prefix:
        return ""
    covered.update(_collect_component_atoms(mol, iodine_substituent_idx, iodine_idx))
    ring_prefixes = []
    for substituent_idx in ring_substituent_indices:
        prefix = _simple_or_branch_prefix(mol, substituent_idx, ring_carbon_idx)
        if not prefix:
            return ""
        ring_prefixes.append(_strip_outer_parentheses(prefix))
        covered.update(_collect_component_atoms(mol, substituent_idx, ring_carbon_idx))
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    counts = Counter(ring_prefixes)
    if len(counts) != 1:
        return ""
    ring_prefix = next(iter(counts))
    if _needs_substituent_parentheses(ring_prefix):
        geminal_prefix = f"3,3-bis({ring_prefix})"
    else:
        geminal_prefix = f"3,3-di{ring_prefix}"
    iodine_prefix = _parenthesized_if_complex(_strip_outer_parentheses(iodine_prefix))
    return f"{geminal_prefix}-1-{iodine_prefix}-1,2-benziodoxole"


def _benziodoxole_iodoxy_lactone_name(mol, Chem, match: Sequence[int]) -> str:
    if set(match) != set(range(mol.GetNumAtoms())):
        return ""
    return "1-hydroxy-1-oxo-1,2-benziodoxol-3-one"


def _benziodoxole_hydride_name(mol, Chem, match: Sequence[int]) -> str:
    if set(match) != set(range(mol.GetNumAtoms())):
        return ""
    iodine_atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "I" and atom.GetFormalCharge() == 0 and atom.GetTotalNumHs() == 1
    ]
    if len(iodine_atoms) != 1:
        return ""
    return "1H-1,2-benziodoxole"


class IodaneParentRule:
    family = "iodane-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        for iodine in mol.GetAtoms():
            if iodine.GetSymbol() != "I" or iodine.GetFormalCharge() != 0 or iodine.IsInRing():
                continue
            iodine_idx = iodine.GetIdx()
            neighbors = [
                nbr.GetIdx()
                for nbr in iodine.GetNeighbors()
                if nbr.GetAtomicNum() > 1
            ]
            if len(neighbors) != 3:
                continue
            ligand_names = []
            covered = {iodine_idx}
            for nidx in neighbors:
                bond = mol.GetBondBetweenAtoms(iodine_idx, nidx)
                if bond is None or bond.GetBondType() != self._Chem.BondType.SINGLE:
                    break
                name = _simple_or_branch_prefix(mol, nidx, iodine_idx)
                if not name:
                    break
                ligand_names.append(name)
                covered.update(_collect_component_atoms(mol, nidx, iodine_idx))
            else:
                if covered != atom_set:
                    continue
                name = _iodane_parent_name(ligand_names)
                if name:
                    yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


def _iodane_parent_name(ligand_names: Sequence[str]) -> str:
    if len(ligand_names) != 3:
        return ""
    counts = Counter(ligand_names)
    parts = []
    for ligand, count in sorted(counts.items(), key=lambda item: item[0]):
        parts.append(_iodane_ligand_prefix(ligand, count))
    if not all(parts):
        return ""
    return "".join(parts) + "-lambda3-iodane"


def _iodane_ligand_prefix(ligand: str, count: int) -> str:
    ligand = _strip_outer_parentheses(ligand)
    if count == 1:
        return _parenthesized_if_complex(ligand)
    multiplier = MULT.get(count)
    if not multiplier:
        return ""
    if _simple_unparenthesized_n_substituent(ligand):
        return f"{multiplier}{ligand}"
    if multiplier == "di":
        return f"bis({ligand})"
    return f"{multiplier}({ligand})"


class IodoniumParentRule:
    family = "iodonium-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        for iodine in mol.GetAtoms():
            if (
                iodine.GetSymbol() != "I"
                or iodine.GetFormalCharge() != 1
                or iodine.IsInRing()
                or iodine.GetTotalNumHs() != 0
            ):
                continue
            iodine_idx = iodine.GetIdx()
            carbon_neighbors = []
            for nbr in iodine.GetNeighbors():
                if nbr.GetAtomicNum() <= 1:
                    continue
                bond = mol.GetBondBetweenAtoms(iodine_idx, nbr.GetIdx())
                if (
                    nbr.GetSymbol() != "C"
                    or bond is None
                    or bond.GetBondType() != self._Chem.BondType.SINGLE
                ):
                    break
                carbon_neighbors.append(nbr.GetIdx())
            else:
                if len(carbon_neighbors) != 2:
                    continue
                ligand_names = []
                covered = {iodine_idx}
                for nidx in carbon_neighbors:
                    name = _simple_or_branch_prefix(mol, nidx, iodine_idx)
                    if not name:
                        break
                    ligand_names.append(name)
                    covered.update(_collect_component_atoms(mol, nidx, iodine_idx))
                else:
                    if covered != atom_set:
                        continue
                    name = _iodonium_parent_name(ligand_names)
                    if name:
                        yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


def _iodonium_parent_name(ligand_names: Sequence[str]) -> str:
    if len(ligand_names) != 2:
        return ""
    counts = Counter(ligand_names)
    parts = []
    for ligand, count in sorted(counts.items(), key=lambda item: item[0]):
        unit = _iodonium_ligand_unit(ligand, count)
        if not unit:
            return ""
        parts.append(unit)
    return "".join(parts) + "iodonium"


def _iodonium_ligand_unit(ligand: str, count: int) -> str:
    ligand = _strip_outer_parentheses(ligand)
    if count == 1:
        return _parenthesized_if_complex(ligand)
    multiplier = MULT.get(count)
    if not multiplier:
        return ""
    return f"bis({ligand})" if multiplier == "di" else f"{multiplier}({ligand})"


class PnictogenHydrideParentRule:
    family = "pnictogen-hydride-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_neutral_pnictogen = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        pnictogen_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() in PNICTOGEN_HYDRIDE_PARENTS
            and atom.GetFormalCharge() == 0
            and not atom.IsInRing()
        ]
        if len(pnictogen_atoms) != 1:
            return
        pnictogen = pnictogen_atoms[0]
        pidx = pnictogen.GetIdx()
        neighbors = [nbr.GetIdx() for nbr in pnictogen.GetNeighbors()]
        implicit_hydrogens = pnictogen.GetTotalNumHs()
        if len(neighbors) + implicit_hydrogens != 3 or not 1 <= len(neighbors) <= 3:
            return
        if any(mol.GetAtomWithIdx(nidx).GetSymbol() not in {"C", "Si"} for nidx in neighbors):
            return
        if any(
            mol.GetBondBetweenAtoms(pidx, nidx).GetBondType() != self._Chem.BondType.SINGLE
            for nidx in neighbors
        ):
            return
        names = []
        for nidx in neighbors:
            prefix = _simple_or_branch_prefix(mol, nidx, pidx)
            if not prefix:
                return
            names.append(prefix)
        name = _pnictogen_hydride_name(names, PNICTOGEN_HYDRIDE_PARENTS[pnictogen.GetSymbol()])
        if name:
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class OxidoNitrosoAzaniumRule:
    family = "oxido-nitroso-azanium-parent"
    priority = parent_priority("senior-functional-exception")
    acyclic_only = True
    requires_nonring_positive_nitrogen = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() != 0:
            return
        for atom in mol.GetAtoms():
            nidx = atom.GetIdx()
            if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
                continue
            oxido = []
            nitroso = []
            organic = []
            ok = True
            for nbr in atom.GetNeighbors():
                nbidx = nbr.GetIdx()
                bond = mol.GetBondBetweenAtoms(nidx, nbidx)
                if (
                    nbr.GetSymbol() == "O"
                    and nbr.GetFormalCharge() < 0
                    and bond.GetBondType() == self._Chem.BondType.SINGLE
                ):
                    oxido.append(nbidx)
                elif nbr.GetSymbol() == "N" and bond.GetBondType() == self._Chem.BondType.SINGLE:
                    oxy = [
                        onbr.GetIdx()
                        for onbr in nbr.GetNeighbors()
                        if onbr.GetIdx() != nidx
                        and onbr.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(nbidx, onbr.GetIdx()).GetBondType() == self._Chem.BondType.DOUBLE
                    ]
                    if len(oxy) == 1:
                        nitroso.append(nbidx)
                    else:
                        ok = False
                        break
                else:
                    organic.append(nbidx)
            if not ok or len(oxido) != 1 or len(nitroso) != 1 or len(organic) != 1:
                continue
            prefix = _simple_or_branch_prefix(mol, organic[0], nidx)
            if not prefix:
                continue
            name = f"{prefix}(oxido)(nitroso)azanium"
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class OxoSulfaniumParentRule:
    family = "oxo-sulfanium-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_positive_sulfur = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        for atom in mol.GetAtoms():
            sidx = atom.GetIdx()
            if atom.GetSymbol() != "S" or atom.GetFormalCharge() <= 0 or atom.IsInRing():
                continue
            oxo = []
            heavy = []
            ok = True
            for nbr in atom.GetNeighbors():
                nidx = nbr.GetIdx()
                bond = mol.GetBondBetweenAtoms(sidx, nidx)
                if nbr.GetSymbol() == "O" and bond.GetBondType() == self._Chem.BondType.DOUBLE:
                    oxo.append(nidx)
                elif bond.GetBondType() == self._Chem.BondType.SINGLE:
                    heavy.append(nidx)
                else:
                    ok = False
                    break
            if not ok or len(oxo) != 1 or len(heavy) != 1:
                continue
            prefix = _simple_or_branch_prefix(mol, heavy[0], sidx)
            if not prefix:
                continue
            if _needs_substituent_parentheses(prefix):
                prefix = f"({prefix})"
            name = f"{prefix}-oxosulfanium"
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class NHaloImineParentRule:
    family = "n-halo-imine-parent"
    priority = parent_priority("senior-functional-exception")
    acyclic_only = True
    requires_imino_skeleton_signal = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() != 0:
            return
        for carbon in mol.GetAtoms():
            cidx = carbon.GetIdx()
            if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
                continue
            imine_n = []
            carbon_neighbors = []
            ok = True
            for nbr in carbon.GetNeighbors():
                nidx = nbr.GetIdx()
                bond = mol.GetBondBetweenAtoms(cidx, nidx)
                if nbr.GetSymbol() == "N" and bond.GetBondType() == self._Chem.BondType.DOUBLE:
                    imine_n.append(nidx)
                elif nbr.GetSymbol() == "C" and bond.GetBondType() == self._Chem.BondType.SINGLE:
                    carbon_neighbors.append(nidx)
                else:
                    ok = False
                    break
            if not ok or len(imine_n) != 1 or len(carbon_neighbors) != 2:
                continue
            halo_idx = _n_halo_atom(mol, imine_n[0], cidx)
            if halo_idx < 0 or not _n_halo_prefix(mol, imine_n[0], cidx, self._Chem):
                continue
            n_prefix = f"N-{HALO[mol.GetAtomWithIdx(halo_idx).GetSymbol()]}"
            names = []
            for path in ((carbon_neighbors[0], cidx, carbon_neighbors[1]), (carbon_neighbors[1], cidx, carbon_neighbors[0])):
                prefixes = []
                if not _append_terminal_halo_prefixes(mol, path[0], cidx, "1", prefixes, self._Chem):
                    continue
                if not _append_terminal_halo_prefixes(mol, path[2], cidx, "3", prefixes, self._Chem):
                    continue
                prefix_text = _format_prefixes(prefixes)
                pieces = [n_prefix]
                if prefix_text:
                    pieces.append(prefix_text)
                pieces.append("propan-2-imine")
                names.append("-".join(pieces))
            if names:
                name = min(names, key=lambda item: (len(item), item))
                yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class GuanidineRetainedParentRule:
    family = "guanidine-retained-parent"
    priority = parent_priority("senior-functional-exception")
    acyclic_only = True
    requires_imino_skeleton_signal = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() != 0:
            return
        for template in _guanidine_parent_templates(self._Chem):
            if template_record_mol(template).GetNumAtoms() > mol.GetNumAtoms():
                continue
            for match in ctx.template_matches(template):
                name = _name_acyclic_retained_core_match(mol, template, match)
                if not name:
                    continue
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=(-template_record_mol(template).GetNumAtoms(), len(name), name),
                )


class SulfonohydrazideRetainedParentRule:
    family = "sulfonohydrazide-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        for sulfur_idx, aryl_idx, distal_n_idx, imine_carbon_idx in _sulfonohydrazide_motifs(mol, self._Chem):
            base = _phenyl_sulfonohydrazide_base(mol, aryl_idx, sulfur_idx)
            ylidene = _hydrazone_ylidene_name(mol, imine_carbon_idx, distal_n_idx)
            if not base or not ylidene:
                continue
            ylidene_text = ylidene if ylidene.startswith("(") and ylidene.endswith(")") else f"({ylidene})"
            name = f"N'-{ylidene_text}-{base}sulfonohydrazide"
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


def _sulfonohydrazide_motifs(mol, Chem) -> list[tuple[int, int, int, int]]:
    motifs = []
    for sulfur in mol.GetAtoms():
        if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0:
            continue
        sulfur_idx = sulfur.GetIdx()
        double_oxygens = {
            nbr.GetIdx()
            for nbr in sulfur.GetNeighbors()
            if nbr.GetSymbol() == "O"
            and mol.GetBondBetweenAtoms(sulfur_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        }
        if len(double_oxygens) != 2:
            continue
        single_heavy = [nbr.GetIdx() for nbr in sulfur.GetNeighbors() if nbr.GetIdx() not in double_oxygens]
        aryl = [
            idx
            for idx in single_heavy
            if mol.GetAtomWithIdx(idx).GetSymbol() == "C" and mol.GetAtomWithIdx(idx).IsInRing()
        ]
        sulfonamide_n = [idx for idx in single_heavy if mol.GetAtomWithIdx(idx).GetSymbol() == "N"]
        if len(aryl) != 1 or len(sulfonamide_n) != 1:
            continue
        nidx = sulfonamide_n[0]
        n_heavy = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(nidx).GetNeighbors()
            if nbr.GetAtomicNum() > 1
        ]
        distal = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(nidx).GetNeighbors()
            if nbr.GetIdx() != sulfur_idx
            and nbr.GetSymbol() == "N"
            and mol.GetBondBetweenAtoms(nidx, nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
        ]
        if len(distal) != 1 or sorted(n_heavy) != sorted((sulfur_idx, distal[0])):
            continue
        distal_n_idx = distal[0]
        distal_heavy = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(distal_n_idx).GetNeighbors()
            if nbr.GetAtomicNum() > 1
        ]
        imine_carbons = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(distal_n_idx).GetNeighbors()
            if nbr.GetIdx() != nidx
            and nbr.GetSymbol() == "C"
            and mol.GetBondBetweenAtoms(distal_n_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        ]
        if len(imine_carbons) == 1 and sorted(distal_heavy) == sorted((nidx, imine_carbons[0])):
            motifs.append((sulfur_idx, aryl[0], distal_n_idx, imine_carbons[0]))
    return motifs


def _phenyl_sulfonohydrazide_base(mol, aryl_idx: int, sulfur_idx: int) -> str:
    aryl = _simple_or_branch_prefix(mol, aryl_idx, sulfur_idx)
    if aryl == "phenyl":
        return "benzene"
    if aryl.endswith("phenyl"):
        return aryl[: -len("phenyl")] + "benzene"
    return ""


def _hydrazone_ylidene_name(mol, carbon_idx: int, nitrogen_idx: int) -> str:
    existing = _imine_carbon_ylidene(mol, carbon_idx, nitrogen_idx)
    if existing:
        return existing
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.IsInRing():
        ring = _simple_or_branch_prefix(mol, carbon_idx, nitrogen_idx)
        if ring.endswith("yl"):
            return ring[: -len("yl")] + "ylidene"
        return ""
    substituents = [nbr.GetIdx() for nbr in carbon.GetNeighbors() if nbr.GetIdx() != nitrogen_idx]
    if len(substituents) == 1:
        name = _simple_or_branch_prefix(mol, substituents[0], carbon_idx)
        if name == "1-oxo-2-oxaethyl":
            return "carboxymethylidene"
        return name + "methylidene" if name and not name.endswith("ylidene") else name
    if len(substituents) == 2:
        names = [_simple_or_branch_prefix(mol, idx, carbon_idx) for idx in substituents]
        if any(not name for name in names):
            return ""
        if names.count("methyl") == 1:
            other = names[1] if names[0] == "methyl" else names[0]
            other_text = _parenthesized_if_complex(other)
            return f"1-{other_text}ethylidene"
    return ""


class CarbamateRetainedParentRule:
    family = "carbamate-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        for _carbonyl_idx, oxygen_idx, nitrogen_idx in _carbamate_motifs(mol, self._Chem):
            oxygen_substituent = _single_external_substituent_name(mol, oxygen_idx, _carbonyl_idx)
            if not oxygen_substituent:
                continue
            nitrogen_substituents = _external_substituent_names(mol, nitrogen_idx, _carbonyl_idx)
            if nitrogen_substituents is None or len(nitrogen_substituents) > 2:
                continue
            nitrogen_prefix = _n_substituent_prefix(nitrogen_substituents)
            if nitrogen_prefix is None:
                continue
            if _unstable_retained_functional_parent_substituent(oxygen_substituent) or any(
                _unstable_retained_functional_parent_substituent(name) for name in nitrogen_substituents
            ):
                continue
            name = f"{oxygen_substituent} {nitrogen_prefix}carbamat"
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class RetainedAcidEsterParentRule:
    family = "retained-acid-ester-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen = set()
        for name in _retained_acid_ester_names(mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class RetainedAmideParentRule:
    family = "retained-amide-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen = set()
        for name in _retained_amide_names(mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class BetaDicarbonylParentRule:
    family = "beta-dicarbonyl-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        seen = set()
        for name in _beta_dicarbonyl_parent_names(ctx.mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class BoronicPinacolEsterParentRule:
    family = "boronic-pinacol-ester-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen = set()
        for boron_idx, owner_idx in _boronic_pinacol_ester_motifs(mol, self._Chem):
            owner = _boronic_pinacol_owner_alias(_simple_or_branch_prefix(mol, owner_idx, boron_idx))
            if not _stable_boronic_pinacol_owner(owner):
                continue
            name = f"{owner}boronic acid pinacol ester"
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class SulfonamideRetainedParentRule:
    family = "sulfonamide-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        for sulfur_idx, substituent_idx, nitrogen_idx in _sulfonamide_motifs(mol, self._Chem):
            base = _sulfonamide_base_name(mol, sulfur_idx, substituent_idx)
            if not base:
                continue
            nitrogen_substituents = _external_substituent_names(mol, nitrogen_idx, sulfur_idx)
            if nitrogen_substituents is None or len(nitrogen_substituents) > 2:
                continue
            nitrogen_prefix = _n_substituent_prefix(nitrogen_substituents)
            if nitrogen_prefix is None:
                continue
            if any(_unstable_retained_functional_parent_substituent(name) for name in nitrogen_substituents):
                continue
            name = f"{nitrogen_prefix}-{base}" if _needs_explicit_n_parent_separator(nitrogen_prefix) else nitrogen_prefix + base
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class SulfonamideParentRule:
    family = "sulfonamide-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        for sulfur_idx, nitrogen_idx in _n_substituted_sulfonamide_motifs(ctx.mol, self._Chem):
            nitrogen_substituents = _external_substituent_names(ctx.mol, nitrogen_idx, sulfur_idx)
            if nitrogen_substituents is None or len(nitrogen_substituents) > 2:
                continue
            nitrogen_prefix = _n_substituent_prefix(nitrogen_substituents)
            if not nitrogen_prefix:
                continue
            name = f"{nitrogen_prefix}-sulfonamide"
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class UreaRetainedParentRule:
    family = "urea-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        for carbonyl_idx, nitrogen_a, nitrogen_b in _urea_motifs(mol, self._Chem):
            substituents_a = _external_substituent_names(mol, nitrogen_a, carbonyl_idx)
            substituents_b = _external_substituent_names(mol, nitrogen_b, carbonyl_idx)
            if substituents_a is None or substituents_b is None:
                continue
            prefix = _urea_n_substituent_prefix(substituents_a, substituents_b)
            if prefix is None:
                continue
            name = prefix + "urea"
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class GlutarimideRetainedParentRule:
    family = "glutarimide-retained-parent"
    priority = parent_priority("senior-functional-exception")
    min_ring_count = 1

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() == 0:
            return
        seen = set()
        for ring in mol.GetRingInfo().AtomRings():
            for name in _glutarimide_ring_parent_names(mol, self._Chem, ring):
                if name in seen:
                    continue
                seen.add(name)
                yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class AminoAcidRetainedParentRule:
    family = "amino-acid-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        for name in _amino_acid_parent_names(mol, self._Chem):
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


def _carbamate_motifs(mol, Chem) -> list[tuple[int, int, int]]:
    motifs = []
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
            continue
        cidx = carbon.GetIdx()
        double_o = []
        single_o = []
        single_n = []
        ok = True
        for nbr in carbon.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_o.append(nidx)
            elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
                single_o.append(nidx)
            elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
                single_n.append(nidx)
            else:
                ok = False
                break
        if ok and len(double_o) == 1 and len(single_o) == 1 and len(single_n) == 1:
            motifs.append((cidx, single_o[0], single_n[0]))
    return motifs


def _glutarimide_ring_parent_names(mol, Chem, ring: Sequence[int]) -> list[str]:
    if len(ring) != 6:
        return []
    ordered = ordered_ring(mol, ring)
    if not ordered:
        return []
    names = []
    for seq in _glutarimide_numberings(mol, Chem, ordered):
        name = _glutarimide_name_from_numbering(mol, Chem, seq)
        if name:
            names.append(name)
    return sorted(set(names), key=lambda item: (len(item), item))


def _glutarimide_numberings(mol, Chem, ordered: Sequence[int]) -> list[tuple[int, ...]]:
    candidates = []
    for n_idx in ordered:
        atom = mol.GetAtomWithIdx(n_idx)
        if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
            continue
        base = list(ordered)
        n_pos = base.index(n_idx)
        rotated = tuple(base[n_pos:] + base[:n_pos])
        for seq in (rotated, (rotated[0], *reversed(rotated[1:]))):
            if _is_glutarimide_numbering(mol, Chem, seq):
                candidates.append(seq)
    return candidates


def _is_glutarimide_numbering(mol, Chem, seq: Sequence[int]) -> bool:
    if len(seq) != 6 or mol.GetAtomWithIdx(seq[0]).GetSymbol() != "N":
        return False
    for left, right in zip(seq, (*seq[1:], seq[0])):
        bond = mol.GetBondBetweenAtoms(left, right)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return False
    for pos, idx in enumerate(seq, start=1):
        atom = mol.GetAtomWithIdx(idx)
        if pos == 1:
            if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
                return False
            continue
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
            return False
        oxo_count = _exocyclic_double_oxygen_count(mol, Chem, idx, set(seq))
        if pos in {2, 6}:
            if oxo_count != 1:
                return False
        elif oxo_count:
            return False
    return True


def _glutarimide_name_from_numbering(mol, Chem, seq: Sequence[int]) -> str:
    ring_set = set(seq)
    prefixes = []
    used_substituent_locs = set()
    for loc, idx in enumerate(seq, start=1):
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set:
                continue
            bond = mol.GetBondBetweenAtoms(idx, nidx)
            if loc in {2, 6} and _is_exocyclic_double_oxygen(mol, Chem, idx, nidx):
                continue
            if bond is not None and bond.GetBondType() != Chem.BondType.SINGLE:
                return ""
            prefix = _simple_or_branch_prefix(mol, nidx, idx)
            if not prefix:
                return ""
            locant = _glutarimide_substituent_locant(loc)
            if not locant or locant in used_substituent_locs:
                return ""
            used_substituent_locs.add(locant)
            prefixes.append((locant, prefix))
    prefix_text = _format_prefixes(prefixes)
    return _join_prefix_parent(prefix_text, "glutarimide") if prefix_text else "glutarimide"


def _glutarimide_substituent_locant(position: int) -> str:
    if position == 1:
        return "N"
    if position == 4:
        return "3"
    if position in {3, 5}:
        return "4"
    return ""


def _phosphite_ester_names(mol, Chem) -> list[str]:
    names = []
    for phosphorus in mol.GetAtoms():
        if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0 or phosphorus.IsInRing():
            continue
        pidx = phosphorus.GetIdx()
        terminal_anionic_oxygen = 0
        owner_names = []
        ok = True
        for nbr in phosphorus.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(pidx, nidx)
            if nbr.GetSymbol() != "O" or bond.GetBondType() != Chem.BondType.SINGLE:
                ok = False
                break
            heavy = [atom.GetIdx() for atom in nbr.GetNeighbors() if atom.GetIdx() != pidx and atom.GetAtomicNum() > 1]
            if not heavy and nbr.GetFormalCharge() < 0:
                terminal_anionic_oxygen += 1
                continue
            if len(heavy) != 1 or nbr.GetFormalCharge() != 0:
                ok = False
                break
            owner = _single_external_substituent_name(mol, nidx, pidx)
            if not owner or _unstable_retained_functional_parent_substituent(owner):
                ok = False
                break
            owner_names.append(owner)
        if ok and terminal_anionic_oxygen == 1 and len(owner_names) == 2:
            names.append(_phosphite_ester_name(owner_names))
    return names


def _phosphite_ester_name(owner_names: Sequence[str]) -> str:
    first, second = sorted(owner_names)
    if first == second:
        if _simple_unparenthesized_n_substituent(first):
            return f"di{first}phosphite"
        return f"bis({_strip_outer_parentheses(first)}) phosphite"
    return f"{_parenthesized_if_complex(first)} {_parenthesized_if_complex(second)} phosphite"


def _exocyclic_double_oxygen_count(mol, Chem, atom_idx: int, ring_set: set[int]) -> int:
    return sum(
        1
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if nbr.GetIdx() not in ring_set and _is_exocyclic_double_oxygen(mol, Chem, atom_idx, nbr.GetIdx())
    )


def _is_exocyclic_double_oxygen(mol, Chem, atom_idx: int, oxygen_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(oxygen_idx)
    if atom.GetSymbol() != "O":
        return False
    bond = mol.GetBondBetweenAtoms(atom_idx, oxygen_idx)
    return bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE


def _retained_acid_ester_names(mol, Chem) -> list[str]:
    names = []
    for carbonyl_idx, ester_oxygen_idx, acyl_idx in _ester_acyl_motifs(mol, Chem):
        owner = _stable_ester_owner_name(mol, ester_oxygen_idx, carbonyl_idx)
        if not owner:
            continue
        generic = _generic_acyl_ester_names(mol, Chem, carbonyl_idx, acyl_idx, owner)
        names.extend(generic)
        dicarboxylate = _dicarboxylate_ester_name(mol, Chem, carbonyl_idx, acyl_idx, owner)
        if dicarboxylate:
            names.append(dicarboxylate)
        acrylate = _acrylate_ester_name(mol, Chem, carbonyl_idx, acyl_idx, owner)
        if acrylate:
            names.append(acrylate)
    return names


def _retained_amide_names(mol, Chem) -> list[str]:
    names = []
    for carbonyl_idx, nitrogen_idx, acyl_idx in _amide_acyl_motifs(mol, Chem):
        n_substituents = _external_substituent_names(mol, nitrogen_idx, carbonyl_idx)
        if n_substituents is None or len(n_substituents) > 2:
            continue
        n_prefix = _n_substituent_prefix(n_substituents)
        if n_prefix is None:
            continue
        if any(_unstable_retained_functional_parent_substituent(name) for name in n_substituents):
            continue
        for path in _ester_acyl_parent_paths(mol, Chem, carbonyl_idx, acyl_idx):
            core = _generic_acyl_functional_parent_path_name(mol, Chem, carbonyl_idx, path, "amide")
            if not core:
                continue
            names.append(_join_n_prefix_functional_parent(n_prefix, core))
    return names


def _beta_dicarbonyl_parent_names(mol, Chem) -> list[str]:
    names = []
    seen_cores = set()
    for alpha in mol.GetAtoms():
        if alpha.GetSymbol() != "C" or alpha.GetFormalCharge() != 0 or alpha.IsInRing():
            continue
        alpha_idx = alpha.GetIdx()
        carbonyls = []
        alpha_substituents = []
        for nbr in alpha.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(alpha_idx, nidx)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if _is_ketone_carbonyl_for_beta_dicarbonyl(mol, Chem, nidx, alpha_idx):
                carbonyls.append(nidx)
            elif nbr.GetAtomicNum() > 1:
                alpha_substituents.append(nidx)
        if len(carbonyls) != 2 or len(alpha_substituents) > 2:
            continue
        carbonyl_ligands = []
        for carbonyl_idx in carbonyls:
            ligand_idx = _beta_dicarbonyl_terminal_ligand(mol, Chem, carbonyl_idx, alpha_idx)
            if ligand_idx is None:
                break
            ligand_name = _simple_or_branch_prefix(mol, ligand_idx, carbonyl_idx)
            if not ligand_name:
                break
            carbonyl_ligands.append((ligand_name, carbonyl_idx))
        else:
            alpha_prefixes = []
            for sub_idx in alpha_substituents:
                prefix = _simple_or_branch_prefix(mol, sub_idx, alpha_idx)
                if not prefix:
                    break
                alpha_prefixes.append(prefix)
            else:
                ordered = sorted(carbonyl_ligands, key=lambda item: (item[0], item[1]))
                core_key = (alpha_idx, tuple(sorted(carbonyls)))
                if core_key in seen_cores:
                    continue
                seen_cores.add(core_key)
                prefixes = [("1", ordered[0][0]), ("3", ordered[1][0])]
                prefixes.extend(("2", prefix) for prefix in alpha_prefixes)
                prefix_text = _format_prefixes(prefixes)
                parent = "propane-1,3-dione"
                names.append(_join_prefix_parent(prefix_text, parent) if prefix_text else parent)
    return names


def _is_ketone_carbonyl_for_beta_dicarbonyl(mol, Chem, carbonyl_idx: int, alpha_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbonyl_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
        return False
    double_oxygen = 0
    carbon_ligands = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen += 1
        elif nidx == alpha_idx and bond.GetBondType() == Chem.BondType.SINGLE:
            continue
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_ligands.append(nidx)
        else:
            return False
    return double_oxygen == 1 and len(carbon_ligands) == 1


def _beta_dicarbonyl_terminal_ligand(mol, Chem, carbonyl_idx: int, alpha_idx: int) -> int | None:
    for nbr in mol.GetAtomWithIdx(carbonyl_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == alpha_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "C" and bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            return nidx
    return None


def _join_n_prefix_functional_parent(n_prefix: str, core: str) -> str:
    if not n_prefix:
        return core
    if _needs_explicit_n_parent_separator(n_prefix) or re.match(r"^\(?\d", core):
        return f"{n_prefix}-{core}"
    return n_prefix + core


def _amide_acyl_motifs(mol, Chem) -> list[tuple[int, int, int]]:
    motifs = []
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
            continue
        cidx = carbon.GetIdx()
        double_oxygen = []
        single_nitrogen = []
        single_carbon = []
        ok = True
        for nbr in carbon.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygen.append(nidx)
            elif nbr.GetSymbol() == "N" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.SINGLE:
                single_nitrogen.append(nidx)
            elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                single_carbon.append(nidx)
            else:
                ok = False
                break
        if ok and len(double_oxygen) == 1 and len(single_nitrogen) == 1 and len(single_carbon) == 1:
            nitrogen = mol.GetAtomWithIdx(single_nitrogen[0])
            if nitrogen.GetFormalCharge() == 0 and not nitrogen.IsInRing():
                motifs.append((cidx, single_nitrogen[0], single_carbon[0]))
    return motifs


_AMINO_ACID_SIDECHAIN_NAMES = {
    "": "glycine",
    "methyl": "alanine",
    "1-methyl-ethyl": "valine",
    "isopropyl": "valine",
    "propyl": "norvaline",
    "butyl": "norleucine",
    "t-butyl": "t-leucine",
    "tert-butyl": "t-leucine",
    "isobutyl": "leucine",
    "1-methyl-propyl": "isoleucine",
    "sec-butyl": "isoleucine",
    "benzyl": "phenylalanine",
    "2-(phenyl)ethyl": "homophenylalanine",
    "4-hydroxybenzyl": "tyrosine",
    "3,4-dihydroxybenzyl": "dopa",
    "(1H-imidazol-4-yl)methyl": "histidine",
    "(indole-3-yl)methyl": "tryptophan",
    "hydroxymethyl": "serine",
    "1-hydroxyethyl": "threonine",
    "mercaptomethyl": "cysteine",
    "sulfanylmethyl": "cysteine",
    "2-methylsulfanyl-ethyl": "methionine",
    "carboxymethyl": "aspartic acid",
    "2-carboxyethyl": "glutamic acid",
    "4-azabutyl": "ornithine",
    "4-aminobutyl": "lysine",
    "5-amino-4,6-diaza5-hexen-1-yl": "arginine",
}


def _amino_acid_parent_names(mol, Chem) -> list[str]:
    names = []
    atom_count = mol.GetNumAtoms()
    for alpha in mol.GetAtoms():
        if alpha.GetSymbol() != "C" or alpha.GetFormalCharge() != 0 or alpha.IsInRing():
            continue
        alpha_idx = alpha.GetIdx()
        carboxyl = []
        amino = []
        sidechains = []
        for nbr in alpha.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(alpha_idx, nidx)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if _terminal_carboxyl_group_kind(mol, Chem, nidx, alpha_idx):
                carboxyl.append(nidx)
            elif nbr.GetSymbol() == "N" and nbr.GetFormalCharge() == 0 and not nbr.IsInRing():
                amino.append(nidx)
            elif nbr.GetAtomicNum() > 1:
                sidechains.append(nidx)
        if len(carboxyl) != 1 or len(amino) != 1 or len(sidechains) > 1:
            continue
        carboxyl_kind = _terminal_carboxyl_group_kind(mol, Chem, carboxyl[0], alpha_idx)
        sidechain_name = ""
        if sidechains:
            sidechain_name = _simple_or_branch_prefix(mol, sidechains[0], alpha_idx)
            if not sidechain_name:
                continue
        amino_acid = _AMINO_ACID_SIDECHAIN_NAMES.get(sidechain_name)
        if not amino_acid:
            continue
        covered = {alpha_idx}
        covered.update(_collect_component_atoms(mol, carboxyl[0], alpha_idx))
        if sidechains:
            covered.update(_collect_component_atoms(mol, sidechains[0], alpha_idx))
        nidx = amino[0]
        covered.add(nidx)
        n_substituents = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(nidx).GetNeighbors()
            if nbr.GetIdx() != alpha_idx and nbr.GetAtomicNum() > 1
        ]
        if len(n_substituents) > 2:
            continue
        n_names = []
        ok = True
        for sub_idx in n_substituents:
            if _amino_acid_n_substituent_is_opsin_fragile(mol, Chem, sub_idx, nidx):
                ok = False
                break
            name = _simple_or_branch_prefix(mol, sub_idx, nidx)
            if not name:
                ok = False
                break
            n_names.append(name)
            covered.update(_collect_component_atoms(mol, sub_idx, nidx))
        if not ok or len(covered) != atom_count:
            continue
        parent = _amino_acid_parent_for_carboxyl_kind(amino_acid, carboxyl_kind)
        if not parent:
            continue
        stereo = _amino_acid_alpha_stereo_prefix(mol, Chem, alpha_idx)
        prefix = _amino_acid_n_substituent_prefix(n_names)
        names.append(stereo + prefix + parent)
    return names


def _amino_acid_n_substituent_is_opsin_fragile(mol, Chem, sub_idx: int, nitrogen_idx: int) -> bool:
    component = _collect_component_atoms(mol, sub_idx, nitrogen_idx)
    if len(component) < 8:
        return False
    if not _is_n_acyl_or_carbamate_link(mol, Chem, sub_idx, nitrogen_idx):
        return False
    return _component_has_nonring_carbon_unsaturation(mol, Chem, component) or _component_has_fused_ring_system(
        mol,
        component,
    )


def _is_n_acyl_or_carbamate_link(mol, Chem, carbon_idx: int, nitrogen_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C":
        return False
    bond_to_n = mol.GetBondBetweenAtoms(carbon_idx, nitrogen_idx)
    if bond_to_n is None or bond_to_n.GetBondType() != Chem.BondType.SINGLE:
        return False
    return any(
        nbr.GetIdx() != nitrogen_idx
        and nbr.GetSymbol() == "O"
        and (bond := mol.GetBondBetweenAtoms(carbon_idx, nbr.GetIdx())) is not None
        and bond.GetBondType() == Chem.BondType.DOUBLE
        for nbr in carbon.GetNeighbors()
    )


def _component_has_nonring_carbon_unsaturation(mol, Chem, component: set[int]) -> bool:
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetIdx() not in component or end.GetIdx() not in component:
            continue
        if begin.IsInRing() or end.IsInRing():
            continue
        if begin.GetSymbol() != "C" or end.GetSymbol() != "C":
            continue
        if bond.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}:
            return True
    return False


def _component_has_fused_ring_system(mol, component: set[int]) -> bool:
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings() if set(ring) <= component]
    return any(
        len(left & right) >= 2
        for idx, left in enumerate(rings)
        for right in rings[idx + 1 :]
    )


def _amino_acid_parent_for_carboxyl_kind(name: str, kind: str) -> str:
    if kind == "acid":
        return name
    if kind != "carboxylate":
        return ""
    if name == "dopa":
        return "3,4-dihydroxyphenylalaninate"
    if name.endswith("ine"):
        return name[:-3] + "inate"
    if name.endswith("ic acid"):
        return name[:-7] + "ate"
    return name + "ate"


def _amino_acid_alpha_stereo_prefix(mol, Chem, alpha_idx: int) -> str:
    atom = mol.GetAtomWithIdx(alpha_idx)
    if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
        return ""
    try:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=False)
    except Exception:
        return ""
    code = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
    return f"({code})-" if code in {"R", "S"} else ""


def _amino_acid_n_substituent_prefix(names: Sequence[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return _amino_acid_n_piece("N", names[0])
    first, second = sorted(names)
    if first == second and _simple_unparenthesized_n_substituent(first):
        return f"N,N-di{first}"
    return _amino_acid_n_piece("N", first) + "-" + _amino_acid_n_piece("N", second)


def _amino_acid_n_piece(locant: str, name: str) -> str:
    if _simple_unparenthesized_n_substituent(name):
        return f"{locant}-{name}"
    return f"{locant}-({name})"


def _collect_component_atoms(mol, start_idx: int, blocked_idx: int) -> set[int]:
    seen = {blocked_idx}
    stack = [start_idx]
    component = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        component.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in seen and nbr.GetAtomicNum() > 1:
                stack.append(nidx)
    return component


def _boronic_pinacol_ester_motifs(mol, Chem) -> list[tuple[int, int]]:
    motifs = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "B" or atom.GetFormalCharge() != 0:
            continue
        bidx = atom.GetIdx()
        oxygen_neighbors = []
        carbon_neighbors = []
        ok = True
        for nbr in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(bidx, nbr.GetIdx())
            if bond.GetBondType() != Chem.BondType.SINGLE:
                ok = False
                break
            if nbr.GetSymbol() == "O":
                oxygen_neighbors.append(nbr.GetIdx())
            elif nbr.GetSymbol() == "C":
                carbon_neighbors.append(nbr.GetIdx())
            else:
                ok = False
                break
        if not ok or len(oxygen_neighbors) != 2 or len(carbon_neighbors) != 1:
            continue
        if _is_pinacol_boronate_oxygen_pair(mol, Chem, bidx, oxygen_neighbors):
            motifs.append((bidx, carbon_neighbors[0]))
    return motifs


def _is_pinacol_boronate_oxygen_pair(mol, Chem, boron_idx: int, oxygen_indices: Sequence[int]) -> bool:
    carbon_indices = []
    for oxygen_idx in oxygen_indices:
        oxygen = mol.GetAtomWithIdx(oxygen_idx)
        if oxygen.GetFormalCharge() != 0:
            return False
        non_boron = [
            nbr.GetIdx()
            for nbr in oxygen.GetNeighbors()
            if nbr.GetIdx() != boron_idx and nbr.GetAtomicNum() > 1
        ]
        if len(non_boron) != 1:
            return False
        carbon_idx = non_boron[0]
        carbon = mol.GetAtomWithIdx(carbon_idx)
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
            return False
        if mol.GetBondBetweenAtoms(oxygen_idx, carbon_idx).GetBondType() != Chem.BondType.SINGLE:
            return False
        carbon_indices.append(carbon_idx)
    if len(set(carbon_indices)) != 2:
        return False
    carbon_bond = mol.GetBondBetweenAtoms(carbon_indices[0], carbon_indices[1])
    if carbon_bond is None or carbon_bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    for carbon_idx in carbon_indices:
        methyl_leaves = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(carbon_idx).GetNeighbors()
            if nbr.GetAtomicNum() == 6
            and nbr.GetIdx() not in carbon_indices
            and mol.GetBondBetweenAtoms(carbon_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
            and nbr.GetDegree() == 1
        ]
        if len(methyl_leaves) != 2:
            return False
    return True


def _stable_boronic_pinacol_owner(name: str) -> bool:
    if not name or len(name) > 70 or any(fragment in name for fragment in ("[", "]", "spiro")):
        return False
    simple_tokens = {
        "methyl",
        "ethyl",
        "propyl",
        "isopropyl",
        "butyl",
        "isobutyl",
        "tert-butyl",
        "t-butyl",
        "vinyl",
        "ethynyl",
        "phenyl",
        "benzyl",
        "o-tolyl",
        "m-tolyl",
        "p-tolyl",
        "cyclopropyl",
        "cyclobutyl",
        "cyclopentyl",
        "cyclohexyl",
        "beta-allyl",
        "crotyl",
        "acetenyl",
    }
    if name in simple_tokens:
        return True
    phenyl_substituent = (
        r"(?:amino|hydroxy|fluoro|chloro|bromo|iodo|methyl|ethyl|methoxy|ethoxy|"
        r"cyano|nitro|trifluoromethyl|difluoromethyl|trifluoromethoxy|difluoromethoxy)"
    )
    phenyl_decorator = rf"\d+(?:,\d+)*-(?:di|tri|tetra)?{phenyl_substituent}"
    phenyl_decorators = rf"(?:{phenyl_decorator}-){{0,3}}{phenyl_decorator}"
    if re.fullmatch(rf"{phenyl_decorators}phenyl", name):
        return True
    if re.fullmatch(rf"{phenyl_decorators}phenylmethyl", name):
        return True
    if re.fullmatch(
        r"(?:[234]-pyridyl|[23456]-pyrimidyl|[234]-furyl|[234]-thienyl|then-[23]-yl|[234]-thiazolyl)",
        name,
    ):
        return True
    if re.fullmatch(r"(?:pyridin|pyrimidin|pyridazin|pyrazin)-\d+-yl", name):
        return True
    heteroaryl_stem = r"pyridin|pyrimidin|pyridazin|pyrazin|furan|thiophen|pyrazol|imidazol|triazol|thiazol|oxazol"
    if re.fullmatch(rf"{phenyl_decorators}(?:{heteroaryl_stem})-\d+-yl", name):
        return True
    simple_ring_substituent = (
        rf"(?:{phenyl_decorators})?phenyl|"
        rf"(?:{phenyl_decorators})?(?:{heteroaryl_stem})-\d+-yl|"
        r"naphthaline-\d+-yl|indol-\d+-yl|indazol-\d+-yl|quinolin-\d+-yl"
    )
    if re.fullmatch(rf"\d+-\({simple_ring_substituent}\)phenyl", name):
        return True
    if re.fullmatch(r"(?:\d+-(?:methyl|ethyl|isopropyl|phenyl|methoxy))?1H-(?:pyrazol|imidazol|triazol)-\d+-yl", name):
        return True
    if re.fullmatch(r"(?:1-(?:methyl|ethyl|isopropyl|phenyl))?(?:pyrazol|imidazol|triazol)-\d+-yl", name):
        return True
    if re.fullmatch(
        r"(?:indan|indol|indole|indazol|benzofuran|benzothiazol|benzoxazol|benzimidazol|carbazol|quinolin|quinazolin|quinoxalin|naphthyridin)-\d+-yl",
        name,
    ):
        return True
    if re.fullmatch(r"(?:prop|but|pent|hex|hept|oct)-\d+-en-\d+-yl", name):
        return True
    if re.fullmatch(r"(?:prop|but|pent|hex|hept|oct)-\d+-yn-\d+-yl", name):
        return True
    if re.fullmatch(r"(?:[12]-methyl)?prop-\d+-en-\d+-yl", name):
        return True
    return False


def _boronic_pinacol_owner_alias(name: str) -> str:
    for verbose, alias in (
        ("1,1,1-trifluoro-methyl", "trifluoromethyl"),
        ("1,1-difluoro-methyl", "difluoromethyl"),
        ("1-fluoro-methyl", "fluoromethyl"),
        ("1,1,1-trichloro-methyl", "trichloromethyl"),
        ("1,1-dichloro-methyl", "dichloromethyl"),
        ("1-chloro-methyl", "chloromethyl"),
        ("1,1,1-tribromo-methyl", "tribromomethyl"),
        ("1,1-dibromo-methyl", "dibromomethyl"),
        ("1-bromo-methyl", "bromomethyl"),
        ("1-methyl-ethyl", "isopropyl"),
        ("1,1-dimethyl-ethyl", "t-butyl"),
        ("2-methyl-propyl", "isobutyl"),
        ("1-phenyl-methyl", "benzyl"),
        ("eth-1-ene-1-yl", "vinyl"),
    ):
        name = name.replace(verbose, alias)
    for alias in (
        "trifluoromethyl",
        "difluoromethyl",
        "fluoromethyl",
        "trichloromethyl",
        "dichloromethyl",
        "chloromethyl",
        "dimethylamino",
        "isopropyl",
        "t-butyl",
        "isobutyl",
        "benzyl",
        "vinyl",
    ):
        name = name.replace(f"({alias})", alias)
    return name


def _ester_acyl_motifs(mol, Chem) -> list[tuple[int, int, int]]:
    motifs = []
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
            continue
        cidx = carbon.GetIdx()
        double_oxygen = []
        single_oxygen = []
        single_carbon = []
        ok = True
        for nbr in carbon.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygen.append(nidx)
            elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
                single_oxygen.append(nidx)
            elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                single_carbon.append(nidx)
            else:
                ok = False
                break
        if ok and len(double_oxygen) == 1 and len(single_oxygen) == 1 and len(single_carbon) == 1:
            motifs.append((cidx, single_oxygen[0], single_carbon[0]))
    return motifs


def _stable_ester_owner_name(mol, oxygen_idx: int, carbonyl_idx: int) -> str:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0:
        return ""
    heavy = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != carbonyl_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return ""
    if mol.GetAtomWithIdx(heavy[0]).GetSymbol() != "C":
        return ""
    name = _simple_or_branch_prefix(mol, heavy[0], oxygen_idx)
    if not name:
        return ""
    name = _retained_parent_substituent_alias(name)
    if len(name) > 40 or " " in name or "[" in name or "]" in name:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return ""
    if any(fragment in name for fragment in ("carbonyl", "carbox", "sulfon", "phosph", "urea")):
        return ""
    return name


def _generic_acyl_ester_names(mol, Chem, carbonyl_idx: int, acyl_idx: int, owner: str) -> list[str]:
    if _unstable_generic_ester_owner(owner):
        return []
    names = []
    for path in _ester_acyl_parent_paths(mol, Chem, carbonyl_idx, acyl_idx):
        name = _generic_acyl_ester_path_name(mol, Chem, carbonyl_idx, path, owner)
        if name:
            names.append(name)
    return names


def _unstable_generic_ester_owner(owner: str) -> bool:
    """Avoid ester-owner forms OPSIN does not tokenize reliably before the acid core."""

    if owner.startswith("("):
        return True
    if re.search(r"\([EZRS](?:,[EZRS])*\)-", owner):
        return True
    return False


def _ester_acyl_parent_paths(mol, Chem, carbonyl_idx: int, acyl_idx: int) -> list[tuple[int, ...]]:
    if mol.GetBondBetweenAtoms(carbonyl_idx, acyl_idx).GetBondType() != Chem.BondType.SINGLE:
        return []
    atoms = _collect_component_atoms(mol, acyl_idx, carbonyl_idx)
    allowed = {
        idx
        for idx in atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == "C"
        and mol.GetAtomWithIdx(idx).GetFormalCharge() == 0
        and not mol.GetAtomWithIdx(idx).IsInRing()
        and not _is_nitrile_carbon(mol, idx)
    }
    if acyl_idx not in allowed:
        return []
    leaves = [
        idx
        for idx in allowed
        if idx == acyl_idx
        or sum(
            1
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
            if nbr.GetIdx() in allowed
        )
        <= 1
    ]
    paths = set()
    best_len = 0
    for leaf in leaves:
        path = (acyl_idx,) if leaf == acyl_idx else tuple(Chem.rdmolops.GetShortestPath(mol, acyl_idx, leaf))
        if any(idx not in allowed for idx in path):
            continue
        if len(path) > best_len:
            best_len = len(path)
            paths.clear()
        if len(path) == best_len:
            paths.add(path)
    return sorted(paths)


def _generic_acyl_ester_path_name(
    mol,
    Chem,
    carbonyl_idx: int,
    path: Sequence[int],
    owner: str,
) -> str:
    core = _generic_acyl_functional_parent_path_name(mol, Chem, carbonyl_idx, path, "oate")
    return f"{owner} {core}" if core else ""


def _generic_acyl_functional_parent_path_name(
    mol,
    Chem,
    carbonyl_idx: int,
    path: Sequence[int],
    suffix: str,
) -> str:
    root = ROOTS.get(len(path) + 1)
    if not root:
        return ""
    path_set = set(path)
    prefixes: list[tuple[str, str]] = []
    for pos, idx in enumerate(path, start=2):
        allowed_neighbors = {carbonyl_idx if pos == 2 else path[pos - 3]}
        if pos - 1 < len(path):
            allowed_neighbors.add(path[pos - 1])
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in allowed_neighbors or nbr.GetAtomicNum() <= 1:
                continue
            if nidx in path_set:
                return ""
            prefix = _simple_or_branch_prefix(mol, nidx, idx)
            if not prefix:
                return ""
            prefixes.append((str(pos), prefix))
    parent = _generic_acyl_functional_parent_text(mol, Chem, root, path, suffix)
    if not parent:
        return ""
    prefix_text = _format_prefixes(prefixes)
    core = _join_prefix_parent(prefix_text, parent) if prefix_text else parent
    stereo = _generic_acyl_ester_stereo_prefix(mol, Chem, path)
    if stereo is None:
        return ""
    if stereo:
        core = f"({stereo})-{core}"
    return core


def _generic_acyl_functional_parent_text(mol, Chem, root: str, path: Sequence[int], suffix: str) -> str:
    ene_locs: list[int] = []
    yne_locs: list[int] = []
    for locant, (a, b) in enumerate(zip(path, path[1:]), start=2):
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            return ""
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            ene_locs.append(locant)
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            yne_locs.append(locant)
        elif bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
    if not ene_locs and not yne_locs:
        retained = _retained_simple_acyl_parent_text(root, suffix)
        if retained:
            return retained
    parent = _unsaturated_parent(root, ene_locs, yne_locs)
    if suffix == "amide":
        return parent[:-1] + "amide" if parent.endswith("e") else parent + "amide"
    return parent[:-1] + "oate" if parent.endswith("e") else parent + "oate"


def _retained_simple_acyl_parent_text(root: str, suffix: str) -> str:
    if suffix == "amide":
        return {
            "meth": "formamide",
            "eth": "acetamide",
        }.get(root, "")
    if suffix == "oate":
        return {
            "meth": "formate",
            "eth": "acetate",
        }.get(root, "")
    return ""


def _generic_acyl_ester_stereo_prefix(mol, Chem, path: Sequence[int]) -> str | None:
    locants = {idx: pos for pos, idx in enumerate(path, start=2)}
    assigned = assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
    atom_descriptors = []
    for idx, descriptor in assigned:
        if idx not in locants:
            return None
        atom_descriptors.append((locants[idx], f"{locants[idx]}{descriptor}"))
    bond_descriptors: list[tuple[int, str]] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        stereo = bond.GetStereo()
        if stereo == Chem.BondStereo.STEREONONE:
            continue
        if stereo in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}:
            descriptor = "E"
        elif stereo in {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}:
            descriptor = "Z"
        else:
            continue
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in locants or end not in locants:
            return None
        locant = min(locants[begin], locants[end])
        bond_descriptors.append((locant, f"{locant}{descriptor}"))
    descriptors = atom_descriptors + bond_descriptors
    return ",".join(text for _locant, text in sorted(descriptors))


_RETAINED_DICARBOXYLATE_ROOTS = {
    0: "oxalate",
    1: "malonate",
    2: "succinate",
    3: "glutarate",
    4: "adipate",
}


def _dicarboxylate_ester_name(mol, Chem, carbonyl_idx: int, acyl_idx: int, owner: str) -> str:
    path = _linear_saturated_dicarboxylate_path(mol, Chem, carbonyl_idx, acyl_idx)
    if path is None:
        return ""
    terminal_idx = path[-1]
    terminal_parent_idx = carbonyl_idx if len(path) == 1 else path[-2]
    terminal = _terminal_carboxyl_group_kind(mol, Chem, terminal_idx, terminal_parent_idx)
    if not terminal:
        return ""
    methylene_count = len(path) - 1
    root = _RETAINED_DICARBOXYLATE_ROOTS.get(methylene_count)
    if not root:
        return ""
    if terminal == "carboxylate":
        return f"{owner} {root}"
    if terminal == "acid":
        return f"{owner} hydrogen {root}"
    return ""


def _linear_saturated_dicarboxylate_path(mol, Chem, carbonyl_idx: int, acyl_idx: int) -> tuple[int, ...] | None:
    path = [acyl_idx]
    previous_idx = carbonyl_idx
    current_idx = acyl_idx
    for _depth in range(5):
        if _terminal_carboxyl_group_kind(mol, Chem, current_idx, previous_idx):
            return tuple(path)
        atom = mol.GetAtomWithIdx(current_idx)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            return None
        heavy = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetIdx() != previous_idx and nbr.GetAtomicNum() > 1
        ]
        if len(heavy) != 1:
            return None
        next_idx = heavy[0]
        bond = mol.GetBondBetweenAtoms(current_idx, next_idx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        previous_idx, current_idx = current_idx, next_idx
        path.append(current_idx)
    return None


def _terminal_carboxyl_group_kind(mol, Chem, carbonyl_idx: int, acyl_idx: int) -> str:
    atom = mol.GetAtomWithIdx(carbonyl_idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
        return ""
    double_oxygen = []
    single_oxygen = []
    single_carbon = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen.append(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            single_oxygen.append(nidx)
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            single_carbon.append(nidx)
    if len(double_oxygen) != 1 or len(single_oxygen) != 1 or single_carbon != [acyl_idx]:
        return ""
    oxygen = mol.GetAtomWithIdx(single_oxygen[0])
    external = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != carbonyl_idx and nbr.GetAtomicNum() > 1
    ]
    if external:
        return ""
    if oxygen.GetFormalCharge() == -1:
        return "carboxylate"
    if oxygen.GetFormalCharge() == 0:
        return "acid"
    return ""


def _acrylate_ester_name(mol, Chem, carbonyl_idx: int, alpha_idx: int, owner: str) -> str:
    alpha = mol.GetAtomWithIdx(alpha_idx)
    if alpha.GetSymbol() != "C" or alpha.GetFormalCharge() != 0 or alpha.IsInRing():
        return ""
    if mol.GetBondBetweenAtoms(carbonyl_idx, alpha_idx).GetBondType() != Chem.BondType.SINGLE:
        return ""
    beta_candidates = [
        nbr.GetIdx()
        for nbr in alpha.GetNeighbors()
        if nbr.GetIdx() != carbonyl_idx
        and nbr.GetAtomicNum() > 1
        and mol.GetBondBetweenAtoms(alpha_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]
    if len(beta_candidates) != 1:
        return ""
    beta_idx = beta_candidates[0]
    beta = mol.GetAtomWithIdx(beta_idx)
    if beta.GetSymbol() != "C" or beta.GetFormalCharge() != 0 or beta.IsInRing():
        return ""
    beta_external = [
        nbr.GetIdx()
        for nbr in beta.GetNeighbors()
        if nbr.GetIdx() != alpha_idx and nbr.GetAtomicNum() > 1
    ]
    if beta_external:
        return ""
    substituents = [
        nbr.GetIdx()
        for nbr in alpha.GetNeighbors()
        if nbr.GetIdx() not in {carbonyl_idx, beta_idx} and nbr.GetAtomicNum() > 1
    ]
    if not substituents:
        return f"{owner} acrylate"
    if len(substituents) != 1:
        return ""
    substituent_idx = substituents[0]
    if _is_nitrile_carbon_idx(mol, Chem, substituent_idx, alpha_idx):
        return f"{owner} 2-cyanoacrylate"
    if _is_terminal_methyl_substituent(mol, Chem, substituent_idx, alpha_idx):
        return f"{owner} methacrylate"
    symbol = mol.GetAtomWithIdx(substituent_idx).GetSymbol()
    if symbol in HALO:
        return f"{owner} 2-{HALO[symbol]}acrylate"
    return ""


def _is_nitrile_carbon_idx(mol, Chem, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return False
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]
    if parent_idx not in heavy or len(heavy) != 2:
        return False
    return any(
        nbr.GetIdx() != parent_idx
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        and nbr.GetSymbol() == "N"
        and nbr.GetFormalCharge() == 0
        for nbr in atom.GetNeighbors()
    )


def _is_terminal_methyl_substituent(mol, Chem, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return False
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    return not [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]


def _urea_motifs(mol, Chem) -> list[tuple[int, int, int]]:
    motifs = []
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
            continue
        cidx = carbon.GetIdx()
        double_o = []
        single_n = []
        ok = True
        for nbr in carbon.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_o.append(nidx)
            elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
                single_n.append(nidx)
            else:
                ok = False
                break
        if ok and len(double_o) == 1 and len(single_n) == 2:
            motifs.append((cidx, single_n[0], single_n[1]))
    return motifs


def _n_substituted_sulfonamide_motifs(mol, Chem) -> list[tuple[int, int]]:
    motifs = []
    for sulfur in mol.GetAtoms():
        if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0 or sulfur.IsInRing():
            continue
        sidx = sulfur.GetIdx()
        double_o = []
        single_n = []
        ok = True
        for nbr in sulfur.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(sidx, nidx)
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_o.append(nidx)
            elif nbr.GetSymbol() == "N" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.SINGLE:
                single_n.append(nidx)
            elif nbr.GetAtomicNum() > 1:
                ok = False
                break
        if ok and len(double_o) >= 1 and len(single_n) == 1:
            nitrogen = mol.GetAtomWithIdx(single_n[0])
            if any(nbr.GetIdx() != sidx and nbr.GetAtomicNum() > 1 for nbr in nitrogen.GetNeighbors()):
                motifs.append((sidx, single_n[0]))
    return motifs


def _sulfonamide_motifs(mol, Chem) -> list[tuple[int, int, int]]:
    motifs = []
    for sulfur in mol.GetAtoms():
        if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0:
            continue
        sidx = sulfur.GetIdx()
        double_o = []
        single_n = []
        single_substituent = []
        ok = True
        for nbr in sulfur.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(sidx, nidx)
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_o.append(nidx)
            elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
                single_n.append(nidx)
            elif bond.GetBondType() == Chem.BondType.SINGLE:
                single_substituent.append(nidx)
            else:
                ok = False
                break
        if ok and len(double_o) == 2 and len(single_n) == 1 and len(single_substituent) == 1:
            motifs.append((sidx, single_substituent[0], single_n[0]))
    return motifs


def _single_external_substituent_name(mol, atom_idx: int, blocked_idx: int) -> str:
    substituents = _external_substituent_names(mol, atom_idx, blocked_idx)
    if substituents is None or len(substituents) != 1:
        return ""
    return substituents[0]


def _external_substituent_names(mol, atom_idx: int, blocked_idx: int) -> list[str] | None:
    names = []
    for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == blocked_idx or nbr.GetAtomicNum() <= 1:
            continue
        name = _simple_or_branch_prefix(mol, nidx, atom_idx)
        if not name:
            return None
        names.append(_retained_parent_substituent_alias(name))
    return names


def _n_substituent_prefix(substituents: Sequence[str]) -> str | None:
    if not substituents:
        return ""
    counts = Counter(_retained_parent_substituent_alias(name) for name in substituents)
    pieces = []
    for name in sorted(counts):
        if not _allowed_n_retained_parent_substituent(name):
            return None
        count = counts[name]
        if count == 1:
            pieces.append(_n_substituent_piece("N", name))
        elif count == 2:
            pieces.append(_duplicate_n_substituent_piece("N,N", name))
        else:
            return None
    return "-".join(pieces)


def _urea_n_substituent_prefix(left: Sequence[str], right: Sequence[str]) -> str | None:
    left = tuple(_retained_parent_substituent_alias(name) for name in left)
    right = tuple(_retained_parent_substituent_alias(name) for name in right)
    if len(left) > 2 or len(right) > 2:
        return None
    if not left and not right:
        return ""
    if not all(_allowed_n_retained_parent_substituent(name) for name in (*left, *right)):
        return None
    if len(left) == 2 and not right:
        return _same_atom_urea_prefix("N", left)
    if len(right) == 2 and not left:
        return _same_atom_urea_prefix("N", right)
    if len(left) == 1 and not right:
        return _n_substituent_piece("N", left[0])
    if len(right) == 1 and not left:
        return _n_substituent_piece("N", right[0])
    if len(left) == 1 and len(right) == 1:
        if left[0] == right[0]:
            return _duplicate_n_substituent_piece("N,N'", left[0])
        first, second = sorted((left[0], right[0]))
        first_piece = _n_substituent_piece("N", first)
        second_piece = _n_substituent_piece("N'", second)
        return f"{first_piece}-{second_piece}"
    return None


def _same_atom_urea_prefix(locant: str, substituents: Sequence[str]) -> str | None:
    counts = Counter(substituents)
    if len(counts) == 1:
        name = next(iter(counts))
        return _duplicate_n_substituent_piece(f"{locant},{locant}", name)
    first, second = sorted(substituents)
    return f"{_n_substituent_piece(locant, first)}-{_n_substituent_piece(locant, second)}"


def _retained_parent_substituent_alias(name: str) -> str:
    aliases = {
        "phenylmethyl": "benzyl",
        "1-(1-oxo-methyl)-methyl": "2-oxoethyl",
        "1-oxo-methyl": "formyl",
    }
    name = aliases.get(name, name)
    return re.sub(r"(\d+)-\(1-oxo-methyl\)(?=[A-Za-z])", r"\1-formyl", name)


def _allowed_n_retained_parent_substituent(name: str) -> bool:
    if _simple_n_retained_parent_substituent(name):
        return True
    if len(name) > 48:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "carbam", "urea")):
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    return any(
        fragment in name
        for fragment in (
            "formyl",
            "oxo",
            "hydroxy",
            "cyclo",
            "phenyl",
            "benzyl",
            "pyrid",
            "pyrimid",
            "piperid",
            "piperaz",
            "morpholin",
            "pyrrolid",
            "azetidin",
            "oxetan",
        )
    )


def _n_substituent_piece(locant: str, name: str) -> str:
    if _simple_unparenthesized_n_substituent(name):
        return f"{locant}-{name}"
    return f"{locant}-({_strip_outer_parentheses(name)})"


def _duplicate_n_substituent_piece(locants: str, name: str) -> str:
    if _simple_unparenthesized_n_substituent(name):
        return f"{locants}-di{name}"
    return f"{locants}-bis({_strip_outer_parentheses(name)})"


def _simple_unparenthesized_n_substituent(name: str) -> bool:
    return name in _SIMPLE_N_RETAINED_PARENT_SUBSTITUENTS or bool(re.fullmatch(r"[A-Za-z0-9']+", name))


def _needs_explicit_n_parent_separator(prefix: str) -> bool:
    return bool(prefix and ("(" in prefix or ")" in prefix or "'" in prefix))


def _unstable_retained_functional_parent_substituent(name: str) -> bool:
    if any(fragment in name for fragment in ("cardenolide", "purine", "puryl")):
        return True
    return "fluoro-methylcarbonyl" in name and ("oxo" in name or "oxa" in name)


def _strip_outer_parentheses(name: str) -> str:
    if name.startswith("(") and name.endswith(")"):
        return name[1:-1]
    return name


_SIMPLE_N_RETAINED_PARENT_SUBSTITUENTS = {
    "methyl",
    "ethyl",
    "propyl",
    "isopropyl",
    "butyl",
    "isobutyl",
    "t-butyl",
    "tert-butyl",
    "cyclopropyl",
    "cyclobutyl",
    "cyclopentyl",
    "cyclohexyl",
    "phenyl",
    "benzyl",
    "o-tolyl",
    "m-tolyl",
    "p-tolyl",
}


def _simple_n_retained_parent_substituent(name: str) -> bool:
    if name in _SIMPLE_N_RETAINED_PARENT_SUBSTITUENTS:
        return True
    return bool(re.fullmatch(r"[234]-(?:fluoro|chloro|bromo|iodo|methyl|methoxy|trifluoromethyl)phenyl", name))


def _sulfonamide_base_name(mol, sulfur_idx: int, substituent_idx: int) -> str:
    substituent = _simple_or_branch_prefix(mol, substituent_idx, sulfur_idx)
    if not substituent:
        return ""
    simple_alkyl = {
        "methyl": "methanesulfonamid",
        "ethyl": "ethanesulfonamid",
        "propyl": "propanesulfonamid",
        "isopropyl": "propane-2-sulfonamid",
        "butyl": "butanesulfonamid",
        "t-butyl": "2-methylpropane-2-sulfonamid",
        "tert-butyl": "2-methylpropane-2-sulfonamid",
    }.get(substituent)
    if simple_alkyl:
        return simple_alkyl
    if substituent == "phenyl":
        return "benzenesulfonamid"
    if substituent in {"o-tolyl", "m-tolyl", "p-tolyl"}:
        return substituent[0] + "-toluenesulfonamid"
    if substituent.endswith("phenyl"):
        return substituent[: -len("phenyl")] + "benzenesulfonamid"
    return ""


class BiphenylParentRule:
    family = "biphenyl-parent"
    priority = parent_priority("whole-molecule-retained-parent")
    min_ring_count = 2

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        rings = tuple(tuple(ring) for ring in mol.GetRingInfo().AtomRings())
        if len(rings) != 2 or any(not _is_benzene_ring(mol, ring) for ring in rings):
            return
        left, right = rings
        if set(left) & set(right):
            return
        connector = _single_ring_connector(mol, left, right, self._Chem)
        if connector is None:
            return
        left_idx, right_idx = connector
        for left_seq in _ring_sequences_from_connector(mol, left, left_idx):
            for right_seq in _ring_sequences_from_connector(mol, right, right_idx):
                name = _biphenyl_name(mol, left_seq, right_seq)
            if name:
                yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class DiphenylbenzeneParentRule:
    family = "diphenylbenzene-parent"
    priority = parent_priority("whole-molecule-retained-parent")
    min_ring_count = 3

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        rings = tuple(tuple(ring) for ring in mol.GetRingInfo().AtomRings())
        if len(rings) != 3 or any(not _is_benzene_ring(mol, ring) for ring in rings):
            return
        if any(set(left) & set(right) for left, right in combinations(rings, 2)):
            return
        connectors = _benzene_ring_assembly_connectors(mol, rings, self._Chem)
        if len(connectors) != 2:
            return
        by_ring = {idx: [] for idx in range(len(rings))}
        for left_ring, left_idx, right_ring, right_idx in connectors:
            by_ring[left_ring].append((left_idx, right_ring))
            by_ring[right_ring].append((right_idx, left_ring))
        central = [ring_idx for ring_idx, edges in by_ring.items() if len(edges) == 2]
        terminals = [ring_idx for ring_idx, edges in by_ring.items() if len(edges) == 1]
        if len(central) != 1 or len(terminals) != 2:
            return
        central_idx = central[0]
        terminal_ring_indices = {ring_idx for _, ring_idx in by_ring[central_idx]}
        if terminal_ring_indices != set(terminals):
            return
        central_connectors = {ring_idx: atom_idx for atom_idx, ring_idx in by_ring[central_idx]}
        for central_seq in _ring_sequences_from_connector(mol, rings[central_idx], min(central_connectors.values())):
            name = _diphenylbenzene_name(mol, rings, central_idx, central_seq)
            if name:
                yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


def _is_benzene_ring(mol, ring: Sequence[int]) -> bool:
    if len(ring) != 6:
        return False
    return all(
        mol.GetAtomWithIdx(idx).GetSymbol() == "C" and mol.GetAtomWithIdx(idx).GetIsAromatic()
        for idx in ring
    )


def _single_ring_connector(mol, left: Sequence[int], right: Sequence[int], Chem):
    right_set = set(right)
    connectors = []
    for left_idx in left:
        for nbr in mol.GetAtomWithIdx(left_idx).GetNeighbors():
            right_idx = nbr.GetIdx()
            if right_idx not in right_set:
                continue
            bond = mol.GetBondBetweenAtoms(left_idx, right_idx)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            connectors.append((left_idx, right_idx))
    return connectors[0] if len(connectors) == 1 else None


def _benzene_ring_assembly_connectors(mol, rings: Sequence[Sequence[int]], Chem) -> list[tuple[int, int, int, int]]:
    connectors = []
    ring_sets = [set(ring) for ring in rings]
    for left_ring, right_ring in combinations(range(len(rings)), 2):
        found = []
        right_set = ring_sets[right_ring]
        for left_idx in rings[left_ring]:
            for nbr in mol.GetAtomWithIdx(left_idx).GetNeighbors():
                right_idx = nbr.GetIdx()
                if right_idx not in right_set:
                    continue
                bond = mol.GetBondBetweenAtoms(left_idx, right_idx)
                if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                    return []
                found.append((left_ring, left_idx, right_ring, right_idx))
        if len(found) > 1:
            return []
        connectors.extend(found)
    return connectors


def _ring_sequences_from_connector(mol, ring: Sequence[int], connector_idx: int) -> tuple[tuple[int, ...], ...]:
    ordered = ordered_ring(mol, ring)
    if not ordered or connector_idx not in ordered:
        return ()
    start = ordered.index(connector_idx)
    seq = ordered[start:] + ordered[:start]
    reverse_seq = (seq[0],) + tuple(reversed(seq[1:]))
    if reverse_seq == seq:
        return (seq,)
    return (seq, reverse_seq)


def _biphenyl_name(mol, left_seq: Sequence[int], right_seq: Sequence[int]) -> str:
    prefixes = []
    left_set = set(left_seq)
    right_set = set(right_seq)
    all_ring_atoms = left_set | right_set
    for loc, idx in enumerate(left_seq, start=1):
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in all_ring_atoms:
                continue
            prefix = _simple_or_branch_prefix(mol, nidx, idx)
            if not prefix:
                return ""
            prefixes.append((str(loc), prefix))
    for loc, idx in enumerate(right_seq, start=1):
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in all_ring_atoms:
                continue
            prefix = _simple_or_branch_prefix(mol, nidx, idx)
            if not prefix:
                return ""
            prefixes.append((f"{loc}'", prefix))
    prefix_text = _format_prefixes(prefixes)
    return _join_prefix_parent(prefix_text, "biphenyl") if prefix_text else "biphenyl"


def _diphenylbenzene_name(
    mol,
    rings: Sequence[Sequence[int]],
    central_idx: int,
    central_seq: Sequence[int],
) -> str:
    central_set = set(central_seq)
    all_ring_atoms = {atom_idx for ring in rings for atom_idx in ring}
    prefixes = []
    for loc, idx in enumerate(central_seq, start=1):
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in central_set:
                continue
            prefix = _simple_or_branch_prefix(mol, nidx, idx)
            if not prefix:
                return ""
            prefixes.append((str(loc), prefix))
    for ring_idx, ring in enumerate(rings):
        if ring_idx == central_idx:
            continue
        ring_set = set(ring)
        for idx in ring:
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx in all_ring_atoms:
                    continue
                return ""
    prefix_text = _format_prefixes(prefixes)
    return _join_prefix_parent(prefix_text, "benzene") if prefix_text else ""


_ACENE_PARENT_NAMES = {
    5: "pentacene",
    6: "hexacene",
    7: "heptacene",
    8: "octacene",
    9: "nonacene",
    10: "decacene",
    11: "undecacene",
    12: "dodecacene",
}


class AceneParentRule:
    family = "acene-retained-parent"
    priority = parent_priority("whole-molecule-retained-parent")
    min_ring_count = 5

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        name = _linear_acene_parent_name(ctx.mol)
        if not name:
            return
        yield NameCandidate(
            name,
            self.family,
            self.priority,
            sort_key=(len(name), name),
            parent_atoms=frozenset(range(ctx.mol.GetNumAtoms())),
            estimated_length=len(name),
        )


def _linear_acene_parent_name(mol) -> str:
    if _formal_charge(mol) != 0:
        return ""
    if any(atom.GetSymbol() != "C" or not atom.GetIsAromatic() for atom in mol.GetAtoms()):
        return ""
    rings = tuple(tuple(ring) for ring in mol.GetRingInfo().AtomRings())
    ring_count = len(rings)
    name = _ACENE_PARENT_NAMES.get(ring_count)
    if not name:
        return ""
    if any(not _is_benzene_ring(mol, ring) for ring in rings):
        return ""
    ring_sets = [set(ring) for ring in rings]
    if set().union(*ring_sets) != set(range(mol.GetNumAtoms())):
        return ""
    neighbors: dict[int, set[int]] = {idx: set() for idx in range(ring_count)}
    shared_edges: dict[int, list[frozenset[int]]] = {idx: [] for idx in range(ring_count)}
    for left, right in combinations(range(ring_count), 2):
        shared = ring_sets[left] & ring_sets[right]
        if not shared:
            continue
        if len(shared) != 2:
            return ""
        shared_edge = frozenset(shared)
        if not _ring_contains_edge(mol, rings[left], shared_edge) or not _ring_contains_edge(
            mol, rings[right], shared_edge
        ):
            return ""
        neighbors[left].add(right)
        neighbors[right].add(left)
        shared_edges[left].append(shared_edge)
        shared_edges[right].append(shared_edge)
    if sum(len(value) for value in neighbors.values()) // 2 != ring_count - 1:
        return ""
    degrees = sorted(len(value) for value in neighbors.values())
    if degrees != [1, 1] + [2] * (ring_count - 2):
        return ""
    if not _ring_graph_is_connected(neighbors):
        return ""
    for ring_index, edges in shared_edges.items():
        if len(edges) == 2 and not _shared_edges_are_opposite_in_ring(mol, rings[ring_index], edges):
            return ""
    return name


def _ring_contains_edge(mol, ring: Sequence[int], edge: frozenset[int]) -> bool:
    seq = ordered_ring(mol, ring)
    return any(
        edge == frozenset((seq[index], seq[(index + 1) % len(seq)]))
        for index in range(len(seq))
    )


def _shared_edges_are_opposite_in_ring(
    mol,
    ring: Sequence[int],
    edges: Sequence[frozenset[int]],
) -> bool:
    seq = ordered_ring(mol, ring)
    edge_positions = []
    for edge in edges:
        positions = [
            index
            for index in range(len(seq))
            if edge == frozenset((seq[index], seq[(index + 1) % len(seq)]))
        ]
        if len(positions) != 1:
            return False
        edge_positions.append(positions[0])
    distance = abs(edge_positions[0] - edge_positions[1])
    return min(distance, len(seq) - distance) == 3


def _ring_graph_is_connected(neighbors: dict[int, set[int]]) -> bool:
    if not neighbors:
        return False
    seen = set()
    stack = [next(iter(neighbors))]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(neighbors[current] - seen)
    return len(seen) == len(neighbors)


class AcyclicSkeletonWithRingSubstituentsRule:
    family = "acyclic-skeleton-with-ring-substituents"
    priority = parent_priority("acyclic-with-ring-substituents")
    min_ring_count = 1

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() == 0:
            return
        name_record = ctx.acyclic_name_record(allow_ring_atoms=False)
        name = name_record.name
        if (
            name
            and mol.GetRingInfo().NumRings() >= 2
            and _has_pterin_retained_fused_parent_substituent_name(name)
            and _should_defer_substituent_like_acyclic_name_to_ring_parent(ctx, name)
        ):
            return
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=_nonring_heavy_atom_indices(mol),
                stereo_locants=_complete_acyclic_stereo_locants(ctx, name, name_record.stereo_locants),
                estimated_length=len(name),
            )


class PreferredAcyclicSkeletonWithRingSubstituentsRule(AcyclicSkeletonWithRingSubstituentsRule):
    family = "preferred-acyclic-skeleton-with-ring-substituents"
    priority = parent_priority("polycyclic-functional-skeleton")
    requires_acyclic_preference_signal = True

    def candidates(self, ctx):
        mol = ctx.mol
        profile = ctx.profile
        if (
            profile.ring_count == 0
            or not profile.may_prefer_acyclic_skeleton
            or _nonring_heavy_atom_count(mol) < 8
        ):
            return
        if _should_defer_acyclic_skeleton_to_exocyclic_ring_parent(ctx):
            return
        name_record = ctx.acyclic_name_record(allow_ring_atoms=False)
        name = name_record.name
        if name and _should_defer_substituent_like_acyclic_name_to_ring_parent(ctx, name):
            return
        if name and _prefer_acyclic_skeleton_name(name):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=_nonring_heavy_atom_indices(mol),
                stereo_locants=_complete_acyclic_stereo_locants(ctx, name, name_record.stereo_locants),
                estimated_length=len(name),
            )


class ChainDominantAcyclicSkeletonWithRingSubstituentsRule(AcyclicSkeletonWithRingSubstituentsRule):
    family = "chain-dominant-acyclic-skeleton-with-ring-substituents"
    priority = parent_priority("spiro-functional-chain-substituent")

    def candidates(self, ctx):
        mol = ctx.mol
        profile = ctx.profile
        if profile.ring_count == 0:
            return
        if _nonring_heavy_atom_count(mol) < 2:
            return
        name_record = ctx.acyclic_name_record(allow_ring_atoms=False)
        name = name_record.name
        if name and _should_defer_substituent_like_acyclic_name_to_ring_parent(ctx, name):
            return
        if name and _chain_dominates_single_ring(mol, name):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=((750, 999, (), 999, name), len(name), name),
                parent_atoms=_nonring_heavy_atom_indices(mol),
                stereo_locants=_complete_acyclic_stereo_locants(ctx, name, name_record.stereo_locants),
                estimated_length=len(name),
            )


class SpiroFunctionalChainSubstituentRule(AcyclicSkeletonWithRingSubstituentsRule):
    family = "spiro-functional-chain-substituent"
    priority = parent_priority("senior-spiro-functional-chain-substituent")

    def candidates(self, ctx):
        mol = ctx.mol
        profile = ctx.profile
        if profile.ring_count == 0 or not ctx.has_spiro_topology:
            return
        ring_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if atom.IsInRing() and atom.GetAtomicNum() > 1}
        nonring_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if not atom.IsInRing() and atom.GetAtomicNum() > 1}
        if len(nonring_atoms) < 4 or len(nonring_atoms) < len(ring_atoms) // 2:
            return
        name_record = ctx.acyclic_name_record(allow_ring_atoms=False)
        name = name_record.name
        if (
            name
            and "spiro[" in name
            and _polycyclic_substituent_chain_signal(name)
            and not _fragile_senior_spiro_chain_name(name)
        ):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=_nonring_heavy_atom_indices(mol),
                stereo_locants=_complete_acyclic_stereo_locants(ctx, name, name_record.stereo_locants),
                estimated_length=len(name),
            )


class ChargedRingAcyclicSkeletonWithRingSubstituentsRule(AcyclicSkeletonWithRingSubstituentsRule):
    family = "charged-ring-acyclic-skeleton-with-ring-substituents"
    priority = parent_priority("polycyclic-functional-skeleton")
    requires_charged_ring_atom = True

    def candidates(self, ctx):
        profile = ctx.profile
        if profile.ring_count == 0 or not profile.has_charged_ring_atom:
            return
        name_record = ctx.acyclic_name_record(allow_ring_atoms=False)
        name = name_record.name
        if (
            name
            and "ium" in name
            and not any(term in name for term in ("bicyclo[", "tricyclo[", "spiro["))
        ):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=_nonring_heavy_atom_indices(ctx.mol),
                stereo_locants=_complete_acyclic_stereo_locants(ctx, name, name_record.stereo_locants),
                estimated_length=len(name),
            )


class FunctionalAcyclicSkeletonWithRingSubstituentsRule(AcyclicSkeletonWithRingSubstituentsRule):
    family = "functional-acyclic-skeleton-with-ring-substituents"
    priority = parent_priority("polycyclic-functional-skeleton")
    requires_functional_skeleton_signal = True
    requires_acyclic_preference_signal = True
    excludes_charged_ring_atom = True

    def candidates(self, ctx):
        mol = ctx.mol
        profile = ctx.profile
        if (
            profile.ring_count == 0
            or profile.has_charged_ring_atom
            or not profile.has_functional_skeleton_signal
            or not profile.may_prefer_acyclic_skeleton
            or _nonring_heavy_atom_count(mol) < 8
        ):
            return
        if _should_defer_acyclic_skeleton_to_exocyclic_ring_parent(ctx):
            return
        name_record = ctx.acyclic_name_record(allow_ring_atoms=False)
        name = name_record.name
        if name and _should_defer_substituent_like_acyclic_name_to_ring_parent(ctx, name):
            return
        if name and _prefer_acyclic_skeleton_name(name):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=_nonring_heavy_atom_indices(mol),
                stereo_locants=_complete_acyclic_stereo_locants(ctx, name, name_record.stereo_locants),
                estimated_length=len(name),
            )


class IminoFunctionalAcyclicSkeletonRule(FunctionalAcyclicSkeletonWithRingSubstituentsRule):
    family = "imino-functional-acyclic-skeleton-with-ring-substituents"
    priority = parent_priority("functional-acyclic-skeleton")
    defer_to_exocyclic_charged_fused_parent = True
    requires_imino_skeleton_signal = True

    def candidates(self, ctx):
        mol = ctx.mol
        profile = ctx.profile
        if (
            profile.ring_count == 0
            or not profile.has_imino_skeleton_signal
            or not profile.may_prefer_acyclic_skeleton
            or _nonring_heavy_atom_count(mol) < 2
        ):
            return
        if _should_defer_acyclic_skeleton_to_exocyclic_ring_parent(ctx):
            return
        name_record = ctx.acyclic_name_record(allow_ring_atoms=False)
        name = name_record.name
        if name and _should_defer_substituent_like_acyclic_name_to_ring_parent(ctx, name):
            return
        if name and "imino" in name and _prefer_acyclic_skeleton_name(name):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=_nonring_heavy_atom_indices(mol),
                stereo_locants=_complete_acyclic_stereo_locants(ctx, name, name_record.stereo_locants),
                estimated_length=len(name),
            )


class ConjugatedCarbonylAcyclicSkeletonRule(FunctionalAcyclicSkeletonWithRingSubstituentsRule):
    family = "conjugated-carbonyl-acyclic-skeleton-with-ring-substituents"
    priority = parent_priority("functional-acyclic-skeleton")

    def candidates(self, ctx):
        profile = ctx.profile
        if (
            profile.ring_count == 0
            or profile.has_charged_ring_atom
            or not profile.has_functional_skeleton_signal
            or not profile.may_prefer_acyclic_skeleton
            or not _has_carbonyl_group(ctx.mol, ctx.Chem)
            or _nonring_heavy_atom_count(ctx.mol) < 8
        ):
            return
        if _should_defer_acyclic_skeleton_to_exocyclic_ring_parent(ctx):
            return
        name_record = ctx.acyclic_name_record(allow_ring_atoms=False)
        name = name_record.name
        if name and _should_defer_substituent_like_acyclic_name_to_ring_parent(ctx, name):
            return
        if (
            name
            and "carbonyl" in name
            and "buta-1,3-diene" in name
            and _prefer_acyclic_skeleton_name(name)
        ):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=_nonring_heavy_atom_indices(ctx.mol),
                stereo_locants=_complete_acyclic_stereo_locants(ctx, name, name_record.stereo_locants),
                estimated_length=len(name),
            )


def _chain_dominates_single_ring(mol, name: str) -> bool:
    if "diazaeth" in name:
        return False
    has_polycyclic_substituent = any(term in name for term in ("bicyclo[", "tricyclo[", "spiro["))
    ring_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if atom.IsInRing() and atom.GetAtomicNum() > 1}
    nonring_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if not atom.IsInRing() and atom.GetAtomicNum() > 1}
    if not ring_atoms:
        return False
    cyclic_substituent = any(
        term in name
        for term in (
            "cyclo",
            "phenyl",
            "benz",
            "pyridin",
            "pyrimidin",
            "pyrazin",
            "pyrazol",
            "imidazol",
            "thiazol",
            "oxazol",
            "thiophen",
            "furan",
            "pyrrol",
            "pyrrolidin",
            "piperidin",
            "aziridin",
            "azetidin",
            "morpholin",
            "oxiran",
            "thietan",
            "oxetan",
            "oxolan",
            "oxan",
            "dioxan",
        )
    )
    if not cyclic_substituent:
        return False
    if has_polycyclic_substituent:
        if len(nonring_atoms) < 4:
            return False
        return _polycyclic_substituent_chain_signal(name)
    if len(nonring_atoms) >= len(ring_atoms):
        return True
    return any(
        term in name
        for term in (
            "bora",
            "azido",
            "bromo",
            "chloro",
            "fluoro",
            "iodo",
            "sila",
            "phospha",
            "sulfanyl",
            "sulfinyl",
            "carbonyl",
            "sulfonyl",
            "imino",
            "trithia",
            "disulfanyl",
            "dioxo",
            "hydroxy",
            "carboxy",
        )
    )


def _polycyclic_substituent_chain_signal(name: str) -> bool:
    """Return whether a functional chain should outrank a polycyclic substituent.

    Polycyclic ring fragments are usually good parents, so we only promote an
    acyclic skeleton when the generated name shows senior chain chemistry that
    OPSIN parses more faithfully as the parent: carbonyl/carboxy/amino/hydroxy
    chains, main-chain unsaturation, or heavily halogenated/heteroatom chains.
    """

    if _fragile_polycyclic_substituent_chain_name(name):
        return False
    if any(term in name for term in ("carbonyl", "carboxy", "dioxo", "trioxo", "tetraoxo", "oxo")):
        return True
    if any(term in name for term in ("amino", "hydroxy", "imino")) and any(
        term in name for term in ("aza", "oxa", "thia", "sulfonyl", "sulfinyl", "sulfanyl")
    ):
        return True
    if re.search(r"(?:prop|but|pent|hex|hept|oct|non|dec)-\d+(?:,\d+)*-(?:en|yn)", name):
        return True
    halo_count = sum(name.count(term) for term in ("fluoro", "chloro", "bromo", "iodo"))
    return halo_count >= 2 and any(term in name for term in ("bora", "sila", "phospha", "aza", "oxa", "thia"))


def _fragile_polycyclic_substituent_chain_name(name: str) -> bool:
    if any(marker in name for marker in ("(E)", "(Z)", ",E)", ",Z)")):
        return True
    if "methylidene" in name:
        return True
    if "2-oxaethyl" in name and "aminosulfinyl" in name:
        return True
    return "hydroxyspiro" in name and "aminosulfinyl" in name and "carbonyl" not in name


def _fragile_senior_spiro_chain_name(name: str) -> bool:
    if _fragile_polycyclic_substituent_chain_name(name):
        return True
    if "pyrrolidin" in name and "aminosulfinyl" in name:
        return True
    if "sulfonylamino" in name:
        return True
    return "imidazo[" in name


def _nonring_heavy_atom_count(mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1 and not atom.IsInRing())


def _heavy_atom_indices(mol) -> frozenset[int]:
    return frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def _nonring_heavy_atom_indices(mol) -> frozenset[int]:
    return frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1 and not atom.IsInRing())


def _complete_acyclic_stereo_locants(
    ctx,
    name: str,
    locants: tuple[tuple[int, str], ...],
) -> tuple[tuple[int, str], ...]:
    if not locants:
        return ()
    locanted_atoms = {atom_idx for atom_idx, _locant in locants}
    assigned_atoms = {
        atom_idx
        for atom_idx, _descriptor in assigned_atom_stereochemistry(
            ctx.mol,
            ctx.Chem,
        )
    }
    missing_atoms = assigned_atoms - locanted_atoms
    if missing_atoms and _has_retained_carbohydrate_substituent_name(name):
        return locants if assigned_atoms & locanted_atoms else ()
    if len(missing_atoms) > _internal_locanted_atom_stereo_count(name):
        return ()

    missing_bonds = [
        bond_atoms
        for bond_atoms in _assigned_bond_stereo_atom_sets(ctx)
        if not bond_atoms <= locanted_atoms
    ]
    if len(missing_bonds) > _internal_locanted_bond_stereo_count(name):
        return ()
    return locants


def _has_retained_carbohydrate_substituent_name(name: str) -> bool:
    return any(
        fragment in name
        for fragment in (
            "glucopyranosyl",
            "galactopyranosyl",
            "mannopyranosyl",
            "fucopyranosyl",
            "ribofuranosyl",
            "fructofuranosyl",
            "apiofuranosyl",
            "glucosaminyl",
            "galactosaminyl",
            "mannosaminyl",
            "muramyl",
            "neuraminyl",
        )
    )


def _assigned_bond_stereo_atom_sets(ctx) -> tuple[frozenset[int], ...]:
    records: list[frozenset[int]] = []
    Chem = ctx.Chem
    for bond in ctx.mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetStereo() in {
            Chem.BondStereo.STEREOE,
            Chem.BondStereo.STEREOTRANS,
            Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOCIS,
        }:
            records.append(frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
    return tuple(records)


def _internal_locanted_atom_stereo_count(name: str) -> int:
    return len(re.findall(r"(?<![A-Za-z])\d+'?[RS](?![a-z])", name))


def _internal_locanted_bond_stereo_count(name: str) -> int:
    return len(re.findall(r"(?<![A-Za-z])\d+'?[EZ](?![a-z])", name))


def _should_defer_acyclic_skeleton_to_exocyclic_ring_parent(ctx) -> bool:
    mol = ctx.mol
    if mol.GetRingInfo().NumRings() < 2:
        return False
    if not _nonring_atoms_are_terminal_exocyclic_multiples(mol, ctx.Chem):
        return False
    return bool(
        ctx.polycyclic_parent_records("bicyclic")
        or ctx.polycyclic_parent_records("spiro")
        or ctx.polycyclic_parent_records("kekulized")
    )


def _should_defer_substituent_like_acyclic_name_to_ring_parent(ctx, name: str) -> bool:
    """Prefer ring-system parents over tiny functional-chain decompositions.

    P-44 parent selection should not let a one-carbon carbonyl/heteroatom
    fragment become the senior parent when the ring system is the real skeleton.
    These string predicates describe renderer shape, while the ring-parent
    requirement comes from graph-derived polycyclic descriptors.
    """

    mol = ctx.mol
    ring_count = mol.GetRingInfo().NumRings()
    if ring_count < 1:
        return False
    if _has_unrendered_ring_isotope(mol, name):
        return True
    if _has_ringless_short_functional_parent_for_cyclic_molecule(name):
        return True
    if ring_count == 1:
        return _has_substituent_like_monocyclic_parent_name(name)
    has_polycyclic_parent_record = (
        ctx.polycyclic_parent_records("bicyclic")
        or ctx.polycyclic_parent_records("spiro")
        or ctx.polycyclic_parent_records("kekulized")
    )
    if not has_polycyclic_parent_record:
        return _has_substituent_like_monocyclic_parent_name(name)
    if any(term in name for term in ("bicyclo[", "tricyclo[", "spiro[")):
        return False
    if _has_ring_substituent_artifact_name(name):
        return True
    if (
        ctx.polycyclic_parent_records("kekulized")
        and _has_short_functional_acyclic_parent_suffix(name)
        and _has_functional_ring_substituent_chain_shape(name)
    ):
        return True
    if (
        ctx.polycyclic_parent_records("kekulized")
        and _has_pterin_retained_fused_parent_substituent_name(name)
        and _has_stemmed_short_acyclic_parent_suffix(name)
        and _has_functional_ring_substituent_chain_shape(name)
    ):
        return True
    if "diamyl" in name and any(term in name for term in ("cyclopropyl", "piperidin", "pyrrolidin")):
        return True
    if "diamyl" in name and "dioxolan" in name:
        return True
    if "azamethylcarbonyl" in name:
        return True
    if name.endswith("-methane") and any(
        term in name
        for term in (
            "carbonyl",
            "carboxy",
            "isocyano",
            "oxomethyl",
            "sulfonyl",
        )
    ):
        return True
    if _has_unsubstituted_sulfonimidoyl_methane_parent(name):
        return True
    if name.endswith("-propane") and "carbonyloxy" in name:
        return True
    if (
        _has_retained_fused_parent_substituent_name(name)
        and any(term in name for term in ("amino", "carbonyl", "carboxy", "cyano", "imino", "oxo", "aza", "oxa", "thia"))
        and _has_short_functional_acyclic_parent_suffix(name)
    ):
        return True
    return False


def _has_substituent_like_monocyclic_parent_name(name: str) -> bool:
    """Detect acyclic names that should leave a monocycle as the parent.

    This is the monocyclic counterpart of the polycyclic deferral above.  A
    short or hetero-interrupted chain bearing phenyl/cyclo/heterocycle text is
    often a renderer artifact: the cyclic component is the stable parent and
    the chain should be rendered as a substituent.
    """

    if not _has_monocyclic_substituent_stem(name):
        return False
    if _has_ring_substituent_artifact_name(name):
        return True
    if _has_short_functional_acyclic_parent_suffix(name) and _has_functional_ring_substituent_chain_shape(name):
        return True
    if name.endswith("-methane") and any(
        term in name
        for term in (
            "carbonyl",
            "carboxy",
            "isocyano",
            "oxomethyl",
            "sulfonyl",
            "thioxo",
        )
    ):
        return True
    if _has_unsubstituted_sulfonimidoyl_methane_parent(name):
        return True
    if re.search(r"(?:^|-)(?:di|tri)?(?:oxa|aza|thia)[a-z]*(?:eth|prop|but|pent|hex|hept|oct|non|dec)ane$", name):
        return True
    if (
        re.search(r"(?:eth|prop|but|pent|hex|hept|oct|non|dec)-\d+(?:,\d+)*-(?:en|yn)", name)
        and any(term in name for term in ("phenyl", "benz", "pyrid", "thiaz", "oxaz", "diaz"))
        and any(term in name for term in ("aza", "oxa", "thia", "carbonyl", "carboxy", "cyano", "imino"))
    ):
        return True
    return False


def _has_ringless_short_functional_parent_for_cyclic_molecule(name: str) -> bool:
    if _has_monocyclic_substituent_stem(name):
        return False
    if not _has_short_functional_acyclic_parent_suffix(name):
        return False
    return any(
        term in name
        for term in (
            "carbonyl",
            "carboxy",
            "carbonyloxy",
            "isocyano",
            "methanoyl",
            "oxamethyl",
            "sulfonyl",
            "thioxo",
        )
    )


def _has_unrendered_ring_isotope(mol, name: str) -> bool:
    for atom in mol.GetAtoms():
        isotope = atom.GetIsotope()
        if isotope and atom.IsInRing():
            symbol = atom.GetSymbol()
            if f"{isotope}{symbol}" not in name and f"[{isotope}{symbol}]" not in name:
                return True
    return False


def _has_unsubstituted_sulfonimidoyl_methane_parent(name: str) -> bool:
    if not name.endswith("-methane") or "sulfonimidoyl" not in name:
        return False
    parent_prefix = name.split("(", 1)[0]
    return parent_prefix in {"", "1-"}


def _has_monocyclic_substituent_stem(name: str) -> bool:
    return any(
        term in name
        for term in (
            "benzene",
            "phenyl",
            "cyclo",
            "furan",
            "imidazol",
            "isoxazol",
            "morpholin",
            "oxadiazol",
            "oxazol",
            "oxetan",
            "oxolan",
            "piperazin",
            "piperidin",
            "pyraz",
            "pyrid",
            "pyrimid",
            "pyrrol",
            "s-triazin",
            "thiadiazol",
            "thiazol",
            "thiophen",
            "triazol",
        )
    )


def _has_functional_ring_substituent_chain_shape(name: str) -> bool:
    if not any(
        term in name
        for term in (
            "amino",
            "carbonyl",
            "carbonyloxy",
            "carboxy",
            "cyano",
            "imino",
            "oxo",
            "oxy",
            "sulfonyl",
        )
    ):
        return False
    return any(
        term in name
        for term in (
            "cyclo",
            "phenyl",
            "pteridin",
            "pterin",
            "pyrid",
            "pyraz",
            "imidaz",
            "oxaz",
            "thiaz",
            "triaz",
        )
    )


def _has_ring_substituent_artifact_name(name: str) -> bool:
    """Detect acyclic renderings that flattened a cyclic parent to substituents."""

    if any(term in name for term in ("active-amyl", "dicrotyl", "dimethallyl", "methoxyl", "methylol")):
        return True
    if "diamyl" not in name:
        return False
    return any(
        term in name
        for term in (
            "azetidin",
            "cyclobut",
            "cyclopent",
            "cycloprop",
            "diazacyclopent",
            "diazaspiro",
            "dioxolan",
            "imidazolidin",
            "oxacyclopent",
            "oxan",
            "oxetan",
            "oxolan",
            "piperidin",
            "pyrrolidin",
        )
    )


def _has_short_functional_acyclic_parent_suffix(name: str) -> bool:
    return bool(
        re.search(r"(?:meth|eth|prop|but|pent|hex|hept|oct)ane$", name)
        or re.search(r"(?:eth|prop|but|pent|hex|hept|oct)-\d+(?:,\d+)*-(?:en|yn)e?$", name)
    )


def _has_stemmed_short_acyclic_parent_suffix(name: str) -> bool:
    return bool(re.search(r"(?:meth|eth|prop|but|pent|hex|hept|oct)an$", name))


def _has_retained_fused_parent_substituent_name(name: str) -> bool:
    if _has_fused_bracket_descriptor(name):
        return True
    if _has_pterin_retained_fused_parent_substituent_name(name):
        return True
    return any(
        term in name
        for term in (
            "anthranil",
            "benzimidazol",
            "benzofuran",
            "benzothiazol",
            "benzoxazol",
            "coumarin",
            "hypoxanthin",
            "imidazo[",
            "indazol",
            "indolin",
            "indol",
            "isoindolin",
            "isoquinolin",
            "naphth",
            "pyrazolo[",
            "pyrimido[",
            "pyrrolo[",
            "purin",
            "quinazolin",
            "quinolin",
            "triazolo[",
            "xanthen",
        )
    )


def _has_pterin_retained_fused_parent_substituent_name(name: str) -> bool:
    return "pterin" in name or "pteridin" in name


def _has_fused_bracket_descriptor(name: str) -> bool:
    return bool(re.search(r"[a-z]+o\[\d+(?:,\d+)*-[a-z]\][a-z]+", name))


def _nonring_atoms_are_terminal_exocyclic_multiples(mol, Chem) -> bool:
    nonring_atoms = [
        atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1 and not atom.IsInRing()
    ]
    if not nonring_atoms or len(nonring_atoms) > 3:
        return False
    for atom in nonring_atoms:
        if atom.GetSymbol() not in {"C", "N", "O", "S"}:
            return False
        heavy_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(heavy_neighbors) != 1:
            return False
        neighbor = heavy_neighbors[0]
        if not neighbor.IsInRing():
            return False
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            return False
    return True


def _has_carbonyl_group(mol, Chem) -> bool:
    return any(
        atom.GetSymbol() == "C"
        and any(
            nbr.GetSymbol() == "O"
            and mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            for nbr in atom.GetNeighbors()
        )
        for atom in mol.GetAtoms()
    )


class WholeMoleculeRetainedParentRule:
    family = "whole-molecule-retained-parent"
    priority = parent_priority("whole-molecule-retained-parent")
    min_ring_count = 0

    def __init__(self, Chem):
        self._Chem = Chem

    def _templates(self) -> tuple[dict, ...]:
        return _standalone_whole_molecule_retained_templates(self._Chem)

    def candidates(self, ctx):
        mol = ctx.mol
        if _too_many_rings_for_whole_molecule_retained_parent(mol):
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        formal_charge = _formal_charge(mol)
        has_stereo_evidence = _has_stereo_evidence(mol, self._Chem)
        for template in ctx.candidate_templates(self._templates(), exact_atom_count=True):
            name = _standalone_whole_molecule_retained_name(template)
            if not name:
                continue
            if _should_defer_natural_product_to_generic_polycycle(ctx, mol, self._Chem, template):
                continue
            if template_record_mol(template).GetNumAtoms() != atom_count:
                continue
            if _formal_charge(template_record_mol(template)) != formal_charge:
                continue
            if has_stereo_evidence and not template.get("stereochemical"):
                continue
            if template.get("stereochemical") and not has_stereo_evidence:
                continue
            for match in ctx.template_matches(template):
                if set(match) != atom_set or _match_has_extra_core_bonds(mol, template, match):
                    continue
                candidate_name = _template_isotope_prefixed_parent_name(mol, template, match, name)
                candidate_name = _opsin_parseable_retained_parent_name(candidate_name)
                stereo_prefix = ""
                if template.get("source_file") == "naturalProducts.xml" and template.get("stereochemical"):
                    stereo_prefix = _locanted_template_stereo_prefix(mol, self._Chem, template, match)
                yield NameCandidate(
                    name=candidate_name,
                    family=_whole_molecule_retained_candidate_family(name),
                    priority=self.priority,
                    sort_key=(len(candidate_name), template.get("source_family", ""), candidate_name),
                    stereo_prefix=stereo_prefix,
                    parent_atoms=frozenset(match),
                    template_id=id(template),
                    template_name=template_record_name(template),
                    source_family=str(template.get("source_family", "")),
                    source_file=str(template.get("source_file", "")),
                    estimated_length=len(candidate_name),
                )


def _standalone_whole_molecule_retained_name(template: dict) -> str:
    raw_name = template_record_name(template)
    name = _terminal_retained_parent_name(raw_name)
    source_file = template.get("source_file", "")
    if source_file == "aminoAcids.xml" and not _template_has_carbonyl_group(template):
        return ""
    if source_file == "carbohydrates.xml":
        carbohydrate_name = _standalone_carbohydrate_name(name)
        if carbohydrate_name:
            return carbohydrate_name
    if source_file == "naturalProducts.xml" and "'" not in raw_name and not raw_name.endswith("ylic"):
        return _opsin_parseable_retained_parent_name(name)
    if source_file in _EXACT_WHOLE_MOLECULE_RETAINED_SOURCE_FILES:
        acid_name = _standalone_retained_acid_name(raw_name)
        if acid_name:
            return acid_name
        return _opsin_parseable_retained_parent_name(name)
    if source_file == "opsin_retained_mono_fused_ring_smiles.csv":
        acid_name = _standalone_retained_acid_name(raw_name)
        if acid_name:
            return acid_name
        return _opsin_parseable_retained_parent_name(name)
    if name in {"benzyn", "tropylium"}:
        return _opsin_parseable_retained_parent_name(name)
    return ""


def _standalone_retained_acid_name(raw_name: str) -> str:
    if raw_name in _STANDALONE_RETAINED_ACID_STEM_EXCEPTIONS:
        return ""
    if raw_name.endswith(" acid"):
        return raw_name
    if raw_name.endswith("acid"):
        return f"{raw_name[:-4]} acid"
    if raw_name.endswith(("ic", "ylic")):
        return f"{raw_name} acid"
    return ""


def _template_has_carbonyl_group(template: dict) -> bool:
    mol = template_record_mol(template)
    return any(
        atom.GetSymbol() == "C"
        and any(
            nbr.GetSymbol() == "O"
            and mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondTypeAsDouble() == 2.0
            for nbr in atom.GetNeighbors()
        )
        for atom in mol.GetAtoms()
    )


def _formal_charge(mol) -> int:
    cached = mol_cache_get(mol, "formal_charge")
    if cached is not None:
        return cached
    return mol_cache_set(mol, "formal_charge", sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


_STANDALONE_RETAINED_ACID_STEM_EXCEPTIONS = {
    "psic",
}


_EXACT_WHOLE_MOLECULE_RETAINED_SOURCE_FILES = {
    "aminoAcids.xml",
    "arylGroups.xml",
    "carbohydrates.xml",
    "simpleCyclicGroups.xml",
    "simpleGroups.xml",
}


_CARBOHYDRATE_STANDALONE_NAMES = {
    "abequ": "abequose",
    "all": "allose",
    "altr": "altrose",
    "api": "apiose",
    "arabin": "arabinose",
    "erythr": "erythrose",
    "erythrul": "erythrulose",
    "fruct": "fructose",
    "fuc": "fucose",
    "galact": "galactose",
    "gluc": "glucose",
    "glycer": "D-glyceraldehyde",
    "gul": "gulose",
    "id": "idose",
    "lyx": "lyxose",
    "mann": "mannose",
    "psic": "psicose",
    "rhamn": "rhamnose",
    "rib": "ribose",
    "ribul": "ribulose",
    "sorb": "sorbose",
    "tagat": "tagatose",
    "tal": "talose",
    "tetr": "erythrose",
    "thre": "threose",
    "tri": "D-glyceraldehyde",
    "xyl": "xylose",
    "xylul": "xylulose",
}


def _standalone_carbohydrate_name(name: str) -> str:
    return _CARBOHYDRATE_STANDALONE_NAMES.get(name, "")


@lru_cache(maxsize=1)
def _standalone_whole_molecule_retained_templates(Chem) -> tuple[dict, ...]:
    return tuple(
        template
        for template in _curated_exact_parent_templates(Chem)
        if _standalone_whole_molecule_retained_name(template)
    )


def _whole_molecule_retained_candidate_family(name: str) -> str:
    if name in {"aporphin"}:
        return "whole-molecule-retained-parent-with-stereo"
    return WholeMoleculeRetainedParentRule.family


class FusedParentRule:
    family = "fused-retained-parent"
    priority = parent_priority("retained-fused-parent")
    min_ring_count = 2

    def __init__(self, Chem):
        self._Chem = Chem

    def _templates(self) -> tuple[dict, ...]:
        return _curated_fused_templates(self._Chem)

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.candidate_templates(self._templates())

    def _candidate_sort_key(self, mol, template: dict, match: Sequence[int], name: str) -> tuple:
        return (
            _benzimidazole_template_h_preference(mol, template, match, name),
            _benzotriazole_template_h_preference(mol, template, match, name),
            *_fused_candidate_preference(name),
            _template_source_rank(template),
            _hydrogenated_template_penalty(template_record_name(template)),
            -len(match),
            name,
        )

    def candidates(self, ctx):
        mol = ctx.mol
        if ctx.features.get("max_ring_system_ring_count", 0) < 2:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        if _high_rank_hetero_retained_fused_search_unpromising(mol):
            return
        facts = self._candidate_facts(ctx)
        if ctx.telemetry is not None:
            ctx.telemetry.candidate_facts_total[self.family] += len(facts)
        render_set = _retained_fused_render_set(facts)
        if ctx.telemetry is not None:
            ctx.telemetry.render_set_size[self.family] += len(render_set)
            ctx.telemetry.facts_suppressed_before_render[self.family] += max(0, len(facts) - len(render_set))

        yield from self._render_facts(ctx, render_set)

    def _candidate_facts(self, ctx) -> tuple["_RetainedFusedCandidateFacts", ...]:
        mol = ctx.mol
        facts: list[_RetainedFusedCandidateFacts] = []
        exact_facts = 0
        topology_facts = 0
        for template, match in _iter_fused_template_matches(
            ctx, self._template_iter(ctx), self._Chem, reject_extra_core_bonds=True
        ):
            facts.append(
                _RetainedFusedCandidateFacts(
                    template=template,
                    match=tuple(match),
                    topology_query=False,
                    structural_key=_retained_fused_fact_sort_key(template, match, topology_query=False),
                )
            )
            exact_facts += 1

        for template, match in _iter_fused_template_matches(
            ctx,
            self._template_iter(ctx),
            self._Chem,
            topology_query=True,
            require_one_ring_system=True,
            reject_extra_core_bonds=True,
        ):
            if not _retained_topology_fused_match_is_hydrogen_compatible(mol, template, match):
                continue
            facts.append(
                _RetainedFusedCandidateFacts(
                    template=template,
                    match=tuple(match),
                    topology_query=True,
                    structural_key=_retained_fused_fact_sort_key(template, match, topology_query=True),
                )
            )
            topology_facts += 1
        if ctx.telemetry is not None:
            ctx.telemetry.filter_stage_count[f"{self.family}:exact_facts"] += exact_facts
            ctx.telemetry.filter_stage_count[f"{self.family}:topology_facts"] += topology_facts
        return tuple(facts)

    def _render_facts(self, ctx, facts: tuple["_RetainedFusedCandidateFacts", ...]):
        mol = ctx.mol
        yielded_names: set[str] = set()
        for render_rank, fact in enumerate(facts, start=1):
            template = fact.template
            match = fact.match
            name = _name_fused_core_match(mol, template, match)
            if not name or name in yielded_names:
                continue
            if _unsafe_benz_a_anthracene_candidate(name):
                continue
            if _should_defer_natural_product_to_generic_polycycle(ctx, mol, self._Chem, template):
                continue
            if fact.topology_query and not _allow_retained_topology_fused_candidate(template_record_name(template), name):
                continue
            sort_key = self._candidate_sort_key(mol, template, match, name)
            if fact.topology_query:
                sort_key = (
                    _retained_topology_fused_candidate_rank(template_record_name(template), name),
                    sort_key,
                )
            yielded_names.add(name)
            mol_cache_set(mol, "has_retained_fused_parent_candidate", True)
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=self.priority,
                sort_key=sort_key,
                parent_atoms=frozenset(match),
                stereo_locants=_template_match_stereo_locants(template, match),
                template_id=id(template),
                template_name=template_record_name(template),
                source_family=str(template.get("source_family", "")),
                source_file=str(template.get("source_file", "")),
                estimated_length=len(name),
                render_rank=render_rank,
            )


@dataclass(frozen=True, slots=True)
class _RetainedFusedCandidateFacts:
    template: dict
    match: tuple[int, ...]
    topology_query: bool
    structural_key: tuple


def _retained_fused_fact_sort_key(
    template: dict,
    match: Sequence[int],
    *,
    topology_query: bool,
) -> tuple:
    template_name = template_record_name(template)
    return (
        1 if topology_query else 0,
        _template_source_rank(template),
        _hydrogenated_template_penalty(template_name),
        -len(match),
        len(template_name),
        template_name,
        tuple(match),
    )


def _retained_fused_render_set(
    facts: tuple[_RetainedFusedCandidateFacts, ...],
) -> tuple[_RetainedFusedCandidateFacts, ...]:
    if len(facts) <= _RETAINED_FUSED_RENDER_SET_SIZE:
        return facts
    return tuple(sorted(facts, key=lambda fact: fact.structural_key)[:_RETAINED_FUSED_RENDER_SET_SIZE])


class RetainedRingScaffoldParentRule:
    family = "fused-retained-parent"
    priority = parent_priority("retained-fused-parent")
    min_ring_count = 1

    def __init__(self, Chem):
        self._Chem = Chem

    def _templates(self) -> tuple[dict, ...]:
        return _general_retained_ring_scaffold_templates(self._Chem)

    def candidates(self, ctx):
        mol = ctx.mol
        if ctx.features.get("max_ring_system_ring_count", 0) < 2:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        facts = self._candidate_facts(ctx)
        if ctx.telemetry is not None:
            ctx.telemetry.candidate_facts_total["retained-ring-scaffold-parent"] += len(facts)
        finalists = _retained_ring_scaffold_render_set(facts)
        if ctx.telemetry is not None:
            ctx.telemetry.render_set_size["retained-ring-scaffold-parent"] += len(finalists)
            ctx.telemetry.facts_suppressed_before_render["retained-ring-scaffold-parent"] += max(
                0,
                len(facts) - len(finalists),
            )
        yielded_names: set[str] = set()
        yielded = 0
        for render_rank, fact in enumerate(finalists, start=1):
            name = _name_fused_core_match(mol, fact.template, fact.match)
            if not name or name in yielded_names:
                continue
            if _unsafe_retained_ring_scaffold_candidate(name):
                continue
            yielded_names.add(name)
            yielded += 1
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=self.priority,
                sort_key=_retained_ring_scaffold_parent_sort_key(fact.template, fact.match, name),
                parent_atoms=frozenset(fact.match),
                locants=fact.compiled.record.locants,
                stereo_locants=_template_match_stereo_locants(fact.compiled.runtime, fact.match),
                template_id=fact.compiled.record.id,
                template_name=fact.compiled.record.name,
                source_family=fact.compiled.record.source_family,
                source_file=fact.compiled.record.source_file,
                estimated_length=len(name),
                render_rank=render_rank,
            )
            if yielded >= _MAX_RETAINED_RING_SCAFFOLD_CANDIDATES:
                return

    def _candidate_facts(self, ctx) -> tuple["_RetainedRingScaffoldCandidateFacts", ...]:
        mol = ctx.mol
        facts: list[_RetainedRingScaffoldCandidateFacts] = []
        compiled_templates = self._candidate_template_facts(ctx)
        telemetry = ctx.telemetry
        family = "retained-ring-scaffold-parent"
        substructure_queries = 0
        match_count = 0
        after_extra_core_bonds = 0
        after_unlocanted_external = 0
        for template_fact in compiled_templates:
            compiled = template_fact.compiled
            template = compiled.runtime
            matches = ctx.template_matches(template)
            if telemetry is not None:
                substructure_queries += 1
                match_count += len(matches)
            for match in matches:
                if _match_has_extra_core_bonds(mol, template, match):
                    continue
                if telemetry is not None:
                    after_extra_core_bonds += 1
                if _match_has_unlocanted_external_substitution(mol, template, match):
                    continue
                if telemetry is not None:
                    after_unlocanted_external += 1
                facts.append(
                    _RetainedRingScaffoldCandidateFacts(
                        compiled=compiled,
                        template=template,
                        match=tuple(match),
                        structural_key=(*template_fact.structural_key, match),
                    )
                )
        if telemetry is not None:
            telemetry.filter_stage_count[f"{family}:substructure_queries"] += substructure_queries
            telemetry.filter_stage_count[f"{family}:matches"] += match_count
            telemetry.filter_stage_count[f"{family}:after_extra_core_bonds"] += after_extra_core_bonds
            telemetry.filter_stage_count[f"{family}:after_unlocanted_external"] += after_unlocanted_external
            telemetry.filter_stage_count[f"{family}:facts"] += len(facts)
        return tuple(facts)

    def _candidate_template_facts(self, ctx) -> tuple["_RetainedRingScaffoldTemplateFacts", ...]:
        mol = ctx.mol
        molecule_aromatic_atom_count = _aromatic_atom_count(mol)
        molecule_charged_atom_values = _charged_atom_values(mol)
        selected_facts: list[tuple[tuple, object]] = []
        compiled_templates = ctx.candidate_compiled_templates(self._templates())
        telemetry = ctx.telemetry
        family = "retained-ring-scaffold-parent"
        after_general_parent = 0
        after_atom_count = 0
        after_aromatic_count = 0
        after_charge_values = 0
        template_fact_count = 0
        trimmed = False
        if telemetry is not None:
            telemetry.filter_stage_count[f"{family}:candidate_templates"] += len(compiled_templates)
        for compiled in compiled_templates:
            if not _compiled_retained_ring_scaffold_template_can_be_general_parent(mol, compiled, self._Chem):
                continue
            if telemetry is not None:
                after_general_parent += 1
            if compiled.features.atom_count >= mol.GetNumAtoms():
                continue
            if telemetry is not None:
                after_atom_count += 1
            if compiled.features.aromatic_atom_count > molecule_aromatic_atom_count:
                continue
            if telemetry is not None:
                after_aromatic_count += 1
            if (
                compiled.features.charged_atom_values
                and not all(value in molecule_charged_atom_values for value in compiled.features.charged_atom_values)
            ):
                continue
            if telemetry is not None:
                after_charge_values += 1
            template_fact_count += 1
            structural_key = _retained_ring_scaffold_template_fact_sort_key(compiled)
            entry = (structural_key, compiled)
            if not trimmed:
                selected_facts.append(entry)
                if len(selected_facts) > _RETAINED_RING_SCAFFOLD_TEMPLATE_FACT_SET_SIZE:
                    selected_facts.sort(key=lambda item: item[0])
                    selected_facts.pop()
                    trimmed = True
            elif structural_key < selected_facts[-1][0]:
                selected_facts.pop()
                insort(selected_facts, entry)
        if telemetry is not None:
            telemetry.filter_stage_count[f"{family}:after_general_parent"] += after_general_parent
            telemetry.filter_stage_count[f"{family}:after_atom_count"] += after_atom_count
            telemetry.filter_stage_count[f"{family}:after_aromatic_count"] += after_aromatic_count
            telemetry.filter_stage_count[f"{family}:after_charge_values"] += after_charge_values
            telemetry.filter_stage_count[f"{family}:template_facts"] += template_fact_count
            telemetry.filter_stage_count[f"{family}:template_render_set"] += len(selected_facts)
            telemetry.filter_stage_count[f"{family}:template_facts_suppressed_before_match"] += max(
                0,
                template_fact_count - len(selected_facts),
            )
        return tuple(
            _RetainedRingScaffoldTemplateFacts(compiled=compiled, structural_key=structural_key)
            for structural_key, compiled in selected_facts
        )

    def _template_fact_render_set(
        self,
        ctx,
        facts: tuple["_RetainedRingScaffoldTemplateFacts", ...],
    ) -> tuple["_RetainedRingScaffoldTemplateFacts", ...]:
        family = "retained-ring-scaffold-parent"
        finalists = _retained_ring_scaffold_template_fact_render_set(facts)
        if ctx.telemetry is not None:
            ctx.telemetry.filter_stage_count[f"{family}:template_render_set"] += len(finalists)
            ctx.telemetry.filter_stage_count[f"{family}:template_facts_suppressed_before_match"] += max(
                0,
                len(facts) - len(finalists),
            )
        return finalists


@dataclass(frozen=True, slots=True)
class _RetainedRingScaffoldTemplateFacts:
    compiled: object
    structural_key: tuple


@dataclass(frozen=True, slots=True)
class _RetainedRingScaffoldCandidateFacts:
    compiled: object
    template: dict
    match: tuple[int, ...]
    structural_key: tuple


def _retained_ring_scaffold_template_fact_sort_key(compiled) -> tuple:
    template_name = compiled.record.name
    return (
        _template_source_rank(compiled.runtime),
        _hydrogenated_template_penalty(template_name),
        -compiled.features.atom_count,
        len(template_name),
        template_name,
        compiled.record.id,
    )


def _retained_ring_scaffold_render_set(
    facts: tuple[_RetainedRingScaffoldCandidateFacts, ...],
) -> tuple[_RetainedRingScaffoldCandidateFacts, ...]:
    if len(facts) <= _RETAINED_RING_SCAFFOLD_RENDER_SET_SIZE:
        return facts
    return tuple(sorted(facts, key=lambda fact: fact.structural_key)[:_RETAINED_RING_SCAFFOLD_RENDER_SET_SIZE])


def _retained_ring_scaffold_template_fact_render_set(
    facts: tuple[_RetainedRingScaffoldTemplateFacts, ...],
) -> tuple[_RetainedRingScaffoldTemplateFacts, ...]:
    if len(facts) <= _RETAINED_RING_SCAFFOLD_TEMPLATE_FACT_SET_SIZE:
        return facts
    return tuple(
        sorted(facts, key=lambda fact: fact.structural_key)[:_RETAINED_RING_SCAFFOLD_TEMPLATE_FACT_SET_SIZE]
    )


def _compiled_retained_ring_scaffold_template_can_be_general_parent(mol, compiled, Chem) -> bool:
    features = compiled.features
    if not compiled.record.name:
        return False
    if "general-parent-unsupported" in features.unsupported_flags:
        return False
    if features.unsupported_flags & {"radical", "organometallic"}:
        return False
    if features.has_stereochemistry:
        if "caroten" not in compiled.record.name:
            return False
        if not _has_stereo_evidence(mol, Chem):
            return False
    if features.ring_count > mol.GetRingInfo().NumRings():
        return False
    return True


def _aromatic_atom_count(mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())


def _charged_atom_values(mol) -> frozenset[int]:
    cached = mol_cache_get(mol, "charged_atom_values")
    if cached is not None:
        return cached
    return mol_cache_set(
        mol,
        "charged_atom_values",
        frozenset(atom.GetFormalCharge() for atom in mol.GetAtoms() if atom.GetFormalCharge()),
    )


@lru_cache(maxsize=1)
def _general_retained_ring_scaffold_templates(Chem) -> tuple[dict, ...]:
    templates = []
    for template in parent_scaffold_catalog(Chem).opsin_retained_ring_csv_parent:
        if not template.get("general_parent_safe", True):
            continue
        flags = set(template.get("flags", ()))
        if flags & {"radical", "organometallic", "general-parent-unsupported"}:
            continue
        if template.get("stereochemical") and "caroten" not in template_record_name(template):
            continue
        templates.append(template)
    return tuple(templates)


def _unsafe_retained_ring_scaffold_candidate(name: str) -> bool:
    if any(
        unsafe in name
        for unsafe in (
            "triazachroman",
            "azaisochinolin",
            "azaacridin",
            "azainden",
            "benzocyclo",
            "cyclopenta[b]pyridin",
            "azanaphthalin",
            "azanaphthyridin",
            "azatetralin",
            "azadecalin",
            "isobenzothiophen",
            "thianaphthalin",
        )
    ):
        return True
    if _unsafe_benz_a_anthracene_candidate(name):
        return True
    if re.search(
        r"(?:nicotinamide|picolinamide|isonicotinamide|nicotinic acid|picolinic acid|isonicotinic acid)-\d+[a-z]?\([^)]*H\)-one\b",
        name,
    ):
        return True
    if re.search(r"\b\d+-carbamyl-.*(?:imidazol|pyrazol)e?\b", name):
        return True
    if (
        "carbamyl" in name
        and re.search(r"(?:imidazol|pyrazol)e?\b", name)
        and not re.search(r"\dH-(?:imidazol|pyrazol)e?\b", name)
    ):
        return True
    if re.search(r"\b\d+-carboxy-.*(?:lutidin|pyridin).*-one\b", name):
        return True
    if re.search(r"\b1,4-benzodioxan-\d+-yl\b", name):
        return True
    if "aminosulfonamido" in name and "N-(" in name:
        return True
    return False


def _unsafe_benz_a_anthracene_candidate(name: str) -> bool:
    if _unsafe_benzoanthracene_candidate(name):
        return True
    if "benz[a]anthracene" not in name:
        return False
    marker = "benz[a]anthracene-"
    marker_index = name.find(marker)
    if marker_index >= 0 and "-yl" in name[marker_index:]:
        before = name[:marker_index]
        if before in {"", "-"} or before.endswith("("):
            return False
        return True
    return name != "benz[a]anthracene"


def _unsafe_benzoanthracene_candidate(name: str) -> bool:
    if "benzoanthracene" not in name:
        return False
    if name.startswith("("):
        return True
    if "6a-" in name or ",6a" in name:
        return True
    if "12b-" in name or ",12b" in name:
        return True
    if "acetyl-6-oxabenzoanthracene" in name:
        return True
    if "dioxo" in name and "triaza" in name:
        return True
    if "pentaaza" in name:
        return True
    return False


def _retained_ring_scaffold_parent_sort_key(template: dict, match: Sequence[int], name: str) -> tuple:
    return (
        _template_source_rank(template),
        _hydrogenated_template_penalty(template_record_name(template)),
        -len(match),
        len(name),
        name,
    )


_RETAINED_TOPOLOGY_FUSED_PARENT_STEMS = (
    "acridin",
    "acridine",
    "anthracen",
    "anthracene",
    "benzimidazole",
    "benzofuran",
    "benzothiazole",
    "benzothiophene",
    "benzotriazole",
    "benzoxazole",
    "carbazol",
    "carbazole",
    "chroman",
    "chromene",
    "chromone",
    "chrysen",
    "chrysene",
    "cinnoline",
    "coronen",
    "coronene",
    "coumarin",
    "fluoranthen",
    "fluoranthene",
    "fluoren",
    "fluorene",
    "guanin",
    "hypoxanthin",
    "indazole",
    "indole",
    "isochroman",
    "isochromene",
    "isoquinoline",
    "naphthalen",
    "naphthalene",
    "naphthyridine",
    "perylen",
    "perylene",
    "phthalazin",
    "phthalazine",
    "purine",
    "pteridine",
    "pyren",
    "pyrene",
    "quinazoline",
    "quinoline",
    "quinoxaline",
    "tetracen",
    "tetracene",
    "xanthen",
    "xanthene",
    "xanthin",
)

_RETAINED_TOPOLOGY_TAUTOMER_PARENT_NAMES = {
    "guanin",
    "hypoxanthin",
    "isoguanin",
    "mercaptopurine",
    "xanthin",
}

_HIGH_IMPACT_TOPOLOGY_FUSED_PARENT_STEMS = (
    "benzo[d][1,3]oxazin",
    "benzo[d]imidazo[5,1-b]thiazol",
    "benzo[e][1,2,4]thiadiazin",
    "benzo[4,5]imidazo[1,2-a]pyridin",
    "benzoboroxol",
    "cyclopenta[b]pyridin",
    "cyclopenta[e]pyrazolo[1,5-a]pyrimidin",
    "imidazo[1,2-a]pyridin",
    "isatoic anhydride",
    "pyrazino-pyridazin",
    "pyrido[1,2-a]pyrimidin",
    "pyrido[2,3-b][1,4]oxazin",
    "pyrido[2,3-d]pyrimidin",
    "pyrido[3,2-b]-1,4-oxazin",
    "pyrimido[1,2-a]pyrimidin",
    "pyrimido[1,2-b]pyridazin",
    "pyrrolo[2,3-c]pyridin",
    "pyrrolo[3,2-c]pyridin",
)


def _allow_retained_topology_fused_candidate(template_name: str, name: str) -> bool:
    """Allow retained fused parents across keto/hydro aromatic-state changes."""

    if _unsafe_benz_a_anthracene_candidate(name):
        return False
    if _retained_topology_name_has_isotope(template_name) or _retained_topology_name_has_isotope(name):
        return False
    if template_name in {"purine", "purin"} and re.search(r"(?:^|-)\d+(?:,\d+)+-dioxo-", name):
        return False
    if template_name in {"purine", "purin"} and _has_duplicate_retained_topology_prefix_locant(name):
        return False
    if "indole" in template_name and _has_retained_topology_functional_suffix_parent(name):
        return False
    if _has_high_impact_topology_parent_stem(template_name, name):
        return True
    if not any(stem in template_name for stem in _RETAINED_TOPOLOGY_FUSED_PARENT_STEMS):
        return False
    if _has_retained_topology_functional_suffix_parent(name):
        return True
    if _has_retained_topology_oxo_or_one_parent(name) or _has_hyphenless_retained_topology_oxo_parent(name):
        return True
    if _has_retained_topology_simple_functional_parent(name):
        return True
    return False


def _has_high_impact_topology_parent_stem(template_name: str, name: str) -> bool:
    if not any(stem in template_name for stem in _HIGH_IMPACT_TOPOLOGY_FUSED_PARENT_STEMS):
        return False
    if not any(stem in name for stem in _HIGH_IMPACT_TOPOLOGY_FUSED_PARENT_STEMS):
        return False
    return bool(
        re.search(
            r"(?:-one\b|dione\b|dioxide\b|anhydride\b|boroxole\b|pyridine\b|pyridazine\b|pyrimidine\b|oxazine\b|thiazole\b)",
            name,
        )
    )


def _has_duplicate_retained_topology_prefix_locant(name: str) -> bool:
    locants = [
        locant
        for locant_text in re.findall(r"(?:^|-)(\d+(?:,\d+)*[a-z]?)-(?=[a-z])", name)
        for locant in locant_text.split(",")
    ]
    return len(locants) != len(set(locants))


def _retained_topology_name_has_isotope(name: str) -> bool:
    return bool(
        re.search(r"\(\d+-\d+(?:C|N|O|S|P|F|Cl|Br|I)\)", name)
        or re.search(r"\[\d+(?:C|N|O|S|P|F|Cl|Br|I)\]", name)
    )


def _retained_topology_fused_candidate_rank(template_name: str, name: str) -> tuple:
    return (
        0 if _has_retained_topology_functional_suffix_parent(name) else 1,
        0 if _has_retained_topology_oxo_or_one_parent(name) else 1,
        _generated_fusion_preference(template_name),
        len(name),
        name,
    )


def _has_retained_topology_functional_suffix_parent(name: str) -> bool:
    return bool(
        re.search(
            _RETAINED_TOPOLOGY_PARENT_PATTERN
            +
            r".*-\d+[a-z]?-(?:carboxamide|carbonitrile|carboxylic acid|carboxylate)\b",
            name,
        )
    )


def _has_retained_topology_oxo_or_one_parent(name: str) -> bool:
    if "acenaphthen" in name:
        return False
    return bool(
        re.search(
            r"(?:^|-)\d+(?:,\d+)*[a-z]?-(?:(?:di|tri|tetra)?oxo|thioxo)-.*"
            + _RETAINED_TOPOLOGY_PARENT_PATTERN,
            name,
        )
        or re.search(
            _RETAINED_TOPOLOGY_PARENT_PATTERN
            +
            r".*-\d+[a-z]?\(\d+[a-z]?H\)-one\b",
            name,
        )
        or re.search(
            _RETAINED_TOPOLOGY_PARENT_PATTERN + r".*-\d+[a-z]?-one\b",
            name,
        )
    )


def _has_hyphenless_retained_topology_oxo_parent(name: str) -> bool:
    if "acenaphthen" in name:
        return False
    parts = name.split("-")
    for idx, part in enumerate(parts[:-1]):
        if not _is_numeric_locant_list(part):
            continue
        following = parts[idx + 1]
        if not following.startswith(("oxo", "dioxo", "trioxo", "tetraoxo", "thioxo")):
            continue
        if any(stem.rstrip("e") in following for stem in _RETAINED_TOPOLOGY_FUSED_PARENT_STEMS):
            return True
    return False


def _is_numeric_locant_list(text: str) -> bool:
    if not text:
        return False
    for locant in text.split(","):
        if not locant:
            return False
        suffix = locant[-1] if locant[-1].isalpha() else ""
        digits = locant[:-1] if suffix else locant
        if not digits.isdigit():
            return False
    return True


def _has_retained_topology_simple_functional_parent(name: str) -> bool:
    return bool(
        re.search(
            r"(?:^|-)\d+[a-z]?-(?:amino|hydroxy|mercapto)-.*"
            + _RETAINED_TOPOLOGY_PARENT_PATTERN,
            name,
        )
    )


_RETAINED_TOPOLOGY_PARENT_PATTERN = (
    r"(?:acridin|anthracen|benzimidazol|benzofuran|benzothiazol|"
    r"benzothiophen|benzotriazol|benzoxazol|carbazol|chrom|chrysen|coronen|"
    r"coumarin|fluoranthen|fluoren|guanin|hypoxanthin|indazol|indol|"
    r"isoquinolin|naphthalen|naphthyridin|perylen|phthalazin|purin|pteridin|pyren|"
    r"quinazolin|quinolin|quinoxalin|tetracen|xanthen|xanthin)"
)


def _retained_topology_fused_match_is_hydrogen_compatible(mol, template: dict, match: Sequence[int]) -> bool:
    if any(stem in template_record_name(template) for stem in _HIGH_IMPACT_TOPOLOGY_FUSED_PARENT_STEMS):
        return True
    if not _generated_topology_fused_match_is_hydrogen_compatible(mol, template, match):
        return False
    if template_record_name(template) in _RETAINED_TOPOLOGY_TAUTOMER_PARENT_NAMES:
        return not _topology_match_has_external_hetero_substitution(mol, template, match)
    return _topology_match_preserves_nonaromatic_template_bonds(mol, template, match)


def _topology_match_has_external_hetero_substitution(mol, template: dict, match: Sequence[int]) -> bool:
    template_mol = template_record_mol(template)
    core = set(match)
    for template_idx, mol_idx in enumerate(match):
        if template_mol.GetAtomWithIdx(template_idx).GetAtomicNum() not in {7, 8, 16}:
            continue
        mol_atom = mol.GetAtomWithIdx(mol_idx)
        if any(nbr.GetIdx() not in core and nbr.GetAtomicNum() > 1 for nbr in mol_atom.GetNeighbors()):
            return True
    return False


def _topology_match_preserves_nonaromatic_template_bonds(mol, template: dict, match: Sequence[int]) -> bool:
    mol_bonds = _topology_bond_facts(mol)
    for begin_idx, end_idx, template_order in _topology_nonaromatic_bonds(template_record_mol(template)):
        mol_bond = mol_bonds.get(_topology_bond_key(match[begin_idx], match[end_idx]))
        if mol_bond is None:
            return False
        mol_is_aromatic, mol_order = mol_bond
        if mol_is_aromatic:
            return False
        if mol_order != template_order:
            return False
    return True


def _topology_nonaromatic_bonds(mol) -> tuple[tuple[int, int, float], ...]:
    cached = mol_cache_get(mol, "topology_nonaromatic_bonds")
    if cached is not None:
        return cached
    bonds = tuple(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
        for bond in mol.GetBonds()
        if not bond.GetIsAromatic()
    )
    return mol_cache_set(mol, "topology_nonaromatic_bonds", bonds)


def _topology_bond_facts(mol) -> dict[tuple[int, int], tuple[bool, float]]:
    cached = mol_cache_get(mol, "topology_bond_facts")
    if cached is not None:
        return cached
    bonds = {
        _topology_bond_key(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()): (
            bond.GetIsAromatic(),
            bond.GetBondTypeAsDouble(),
        )
        for bond in mol.GetBonds()
    }
    return mol_cache_set(mol, "topology_bond_facts", bonds)


def _topology_bond_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _template_match_stereo_locants(template: dict, match: Sequence[int]) -> tuple[tuple[int, str], ...]:
    locants = template_record_locants(template)
    return tuple(
        (int(mol_idx), str(locant))
        for template_idx, mol_idx in enumerate(match)
        if (locant := locants.get(template_idx, ""))
    )


def _sequence_stereo_locants(seq: Sequence[int]) -> tuple[tuple[int, str], ...]:
    return tuple((int(atom_idx), str(position)) for position, atom_idx in enumerate(seq, start=1))


def _complete_single_ring_parent_stereo_locants(mol, seq: Sequence[int], Chem) -> tuple[tuple[int, str], ...]:
    parent_atoms = frozenset(seq)
    if not parent_atoms:
        return ()
    assigned_atoms = {idx for idx, _descriptor in assigned_atom_stereochemistry(mol, Chem)}
    if not assigned_atoms or not assigned_atoms.issubset(parent_atoms):
        return ()
    return _sequence_stereo_locants(seq)


class GeneratedFusedParentRule(FusedParentRule):
    """Fallback OPSIN-fusion scaffold family.

    These templates are generated in bulk from OPSIN fusion components. They
    broaden coverage, but they should not displace curated retained parents
    whose locant conventions are hand-checked more carefully.
    """

    family = "generated-fused-parent"
    priority = parent_priority("generated-parent")
    min_ring_system_ring_count = 2

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.generated_fused_templates

    def _candidate_sort_key(self, mol, template: dict, match: Sequence[int], name: str) -> tuple:
        return (_generated_fusion_preference(template_record_name(template)), *_fused_candidate_preference(name), -len(match), name)

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        if _high_rank_hetero_retained_fused_search_unpromising(mol):
            return

        facts = self._candidate_facts(ctx)
        if ctx.telemetry is not None:
            ctx.telemetry.candidate_facts_total[self.family] += len(facts)
        render_set = _generated_fused_render_set(facts)
        if ctx.telemetry is not None:
            ctx.telemetry.render_set_size[self.family] += len(render_set)
            ctx.telemetry.facts_suppressed_before_render[self.family] += max(0, len(facts) - len(render_set))

        yield from self._render_facts(ctx, render_set)

    def _candidate_facts(self, ctx) -> tuple["_GeneratedFusedCandidateFacts", ...]:
        mol = ctx.mol
        facts: list[_GeneratedFusedCandidateFacts] = []
        exact_facts = 0
        topology_facts = 0
        for template, match in _iter_fused_template_matches(
            ctx, self._template_iter(ctx), self._Chem, reject_extra_core_bonds=True
        ):
            facts.append(
                _GeneratedFusedCandidateFacts(
                    template=template,
                    match=tuple(match),
                    topology_query=False,
                    structural_key=_generated_fused_fact_sort_key(template, match, topology_query=False),
                )
            )
            exact_facts += 1

        topology_templates = tuple(
            template
            for template in self._template_iter(ctx)
            if _allow_generated_topology_template(template_record_name(template))
        )
        for template, match in _iter_fused_template_matches(
            ctx,
            topology_templates,
            self._Chem,
            topology_query=True,
            require_one_ring_system=True,
            reject_extra_core_bonds=True,
        ):
            if not _generated_topology_fused_match_is_hydrogen_compatible(mol, template, match):
                continue
            facts.append(
                _GeneratedFusedCandidateFacts(
                    template=template,
                    match=tuple(match),
                    topology_query=True,
                    structural_key=_generated_fused_fact_sort_key(template, match, topology_query=True),
                )
            )
            topology_facts += 1

        if ctx.telemetry is not None:
            ctx.telemetry.filter_stage_count[f"{self.family}:exact_facts"] += exact_facts
            ctx.telemetry.filter_stage_count[f"{self.family}:topology_facts"] += topology_facts
        return tuple(facts)

    def _render_facts(self, ctx, facts: tuple["_GeneratedFusedCandidateFacts", ...]):
        mol = ctx.mol
        yielded_names: set[str] = set()
        yielded = 0
        for render_rank, fact in enumerate(facts, start=1):
            template = fact.template
            match = fact.match
            name = _name_fused_core_match(mol, template, match)
            if not name or name in yielded_names:
                continue
            if _unsafe_benz_a_anthracene_candidate(name):
                continue
            template_name = template_record_name(template)
            if fact.topology_query and not _allow_generated_topology_fused_candidate(template_name, name):
                continue
            yielded_names.add(name)
            yielded += 1
            mol_cache_set(mol, "has_retained_fused_parent_candidate", True)
            sort_key = self._candidate_sort_key(mol, template, match, name)
            if fact.topology_query:
                sort_key = (_generated_topology_fused_candidate_rank(template_name, name), *sort_key)
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=self.priority,
                sort_key=sort_key,
                parent_atoms=frozenset(match),
                stereo_locants=_template_match_stereo_locants(template, match),
                estimated_length=len(name),
                render_rank=render_rank,
            )
            if yielded >= _MAX_GENERATED_FUSED_PARENT_CANDIDATES:
                return


@dataclass(frozen=True, slots=True)
class _GeneratedFusedCandidateFacts:
    template: dict
    match: tuple[int, ...]
    topology_query: bool
    structural_key: tuple


def _generated_fused_fact_sort_key(
    template: dict,
    match: Sequence[int],
    *,
    topology_query: bool,
) -> tuple:
    template_name = template_record_name(template)
    return (
        1 if topology_query else 0,
        _generated_fused_template_anchor_rank(template_name),
        _generated_fusion_preference(template_name),
        _template_source_rank(template),
        _hydrogenated_template_penalty(template_name),
        -len(match),
        len(template_name),
        template_name,
        tuple(match),
    )


def _generated_fused_template_anchor_rank(template_name: str) -> int:
    if re.match(r"^(?:benzo|benzeno)\[[^\]]+\](?!benzene$)", template_name):
        return 0
    if re.match(r"^(?:pyrimido|pyrido)\[[^\]]+\](?:pyrazole|indazole|pyrrole)$", template_name):
        return 0
    return 1


def _generated_fused_render_set(
    facts: tuple[_GeneratedFusedCandidateFacts, ...],
) -> tuple[_GeneratedFusedCandidateFacts, ...]:
    if len(facts) <= _GENERATED_FUSED_RENDER_SET_SIZE:
        return facts
    return tuple(sorted(facts, key=lambda fact: fact.structural_key)[:_GENERATED_FUSED_RENDER_SET_SIZE])


class GeneratedSpiroFusedParentRule(GeneratedFusedParentRule):
    """Promote OPSIN-derived component-spiro templates over retained spiro aliases."""

    family = "generated-spiro-fused-parent"
    priority = parent_priority("generated-spiro-component-parent")

    def _template_iter(self, ctx) -> Sequence[dict]:
        return tuple(
            template
            for template in ctx.generated_fused_templates
            if template_record_name(template).startswith("spiro[")
        )


def _allow_generated_topology_fused_candidate(template_name: str, name: str) -> bool:
    """Allow generated fused templates when exact aromatic matching is too strict.

    OPSIN-derived fused templates frequently have the right atom connectivity
    but a different aromatic/Kekule representation from RDKit input.  This path
    is limited to retained fused scaffold shapes where a suffix or indicated-H
    parent gives a more stable graph than generic bicyclo/tricyclo fallback.
    """

    if not _allow_generated_topology_template(template_name):
        return False
    if _has_generated_topology_fused_suffix_parent(name):
        return True
    return False


def _allow_generated_topology_template(template_name: str) -> bool:
    return template_name.startswith("indeno[")


def _generated_topology_fused_match_is_hydrogen_compatible(mol, template: dict, match: Sequence[int]) -> bool:
    template_mol = template_record_mol(template)
    template_charges, template_hydrogens, _template_heavy_neighbors = _topology_match_atom_facts(template_mol)
    mol_charges, mol_hydrogens, mol_heavy_neighbors = _topology_match_atom_facts(mol)
    expected_hydrogens = 0
    actual_hydrogens = 0
    core = set(match)
    for template_idx, mol_idx in enumerate(match):
        if template_charges[template_idx] != mol_charges[mol_idx]:
            return False
        external_neighbors = sum(
            1
            for nbr_idx in mol_heavy_neighbors[mol_idx]
            if nbr_idx not in core
        )
        expected_hydrogens += max(0, template_hydrogens[template_idx] - external_neighbors)
        actual_hydrogens += mol_hydrogens[mol_idx]
    return actual_hydrogens <= expected_hydrogens


def _topology_match_atom_facts(mol) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
    cached = mol_cache_get(mol, "topology_match_atom_facts")
    if cached is not None:
        return cached
    atoms = tuple(mol.GetAtoms())
    facts = (
        tuple(atom.GetFormalCharge() for atom in atoms),
        tuple(atom.GetTotalNumHs() for atom in atoms),
        tuple(
            tuple(nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1)
            for atom in atoms
        ),
    )
    return mol_cache_set(mol, "topology_match_atom_facts", facts)


def _generated_topology_fused_candidate_rank(template_name: str, name: str) -> tuple:
    return (
        0 if _has_generated_topology_fused_suffix_parent(name) else 1,
        len(name),
        name,
    )


def _has_generated_topology_fused_suffix_parent(name: str) -> bool:
    return bool(
        re.search(
            r"\bindeno\[[^\]]+\].*-\d+[a-z]?-(?:carboxamide|carbonitrile|carboxylic acid|carboxylate)\b",
            name,
        )
    )


class SingleRingParentRule:
    family = "single-ring-parent"
    priority = parent_priority("single-ring-parent")
    min_ring_count = 1

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        ring_count = mol.GetRingInfo().NumRings()
        if ring_count < 1:
            return
        parent_rings = tuple(mol.GetRingInfo().AtomRings())
        candidate_priority = self.priority
        if ring_count > 2:
            parent_rings = _unfused_assembly_monocycle_parent_rings(ctx)
            if not parent_rings:
                return
            if not _is_all_monocycle_ring_assembly(ctx):
                candidate_priority = parent_priority("acyclic-with-ring-substituents") + 1
        if ring_count > 2 and not parent_rings:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        ring_facts = self._ring_fact_render_set(ctx, self._candidate_ring_facts(ctx, parent_rings))
        for render_rank, ring_fact in enumerate(ring_facts, start=1):
            ring = ring_fact.ring
            for name, stereo_prefix, locant_seq in _named_ring_instance_names(mol, tuple(ring), self._Chem):
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=candidate_priority,
                    sort_key=_single_ring_candidate_sort_key(name),
                    stereo_prefix=stereo_prefix,
                    parent_atoms=frozenset(ring),
                    locants=_locants_from_sequence(locant_seq),
                    stereo_locants=_complete_single_ring_parent_stereo_locants(mol, locant_seq, self._Chem),
                    estimated_length=len(name),
                    render_rank=render_rank,
                    )

    def _candidate_ring_facts(self, ctx, parent_rings) -> tuple["_SingleRingParentFacts", ...]:
        facts = tuple(
            _SingleRingParentFacts(
                ring=tuple(ring),
                structural_key=_single_ring_parent_fact_sort_key(ctx.mol, ring),
            )
            for ring in parent_rings
        )
        if ctx.telemetry is not None:
            ctx.telemetry.filter_stage_count[f"{self.family}:ring_facts"] += len(facts)
        return facts

    def _ring_fact_render_set(
        self,
        ctx,
        facts: tuple["_SingleRingParentFacts", ...],
    ) -> tuple["_SingleRingParentFacts", ...]:
        if len(facts) <= _SINGLE_RING_PARENT_RING_FACT_SET_SIZE:
            finalists = facts
        else:
            finalists = tuple(
                sorted(facts, key=lambda fact: fact.structural_key)[:_SINGLE_RING_PARENT_RING_FACT_SET_SIZE]
            )
        if ctx.telemetry is not None:
            ctx.telemetry.filter_stage_count[f"{self.family}:ring_render_set"] += len(finalists)
            ctx.telemetry.filter_stage_count[f"{self.family}:ring_facts_suppressed_before_render"] += max(
                0,
                len(facts) - len(finalists),
            )
        return finalists


class PositiveSingleRingParentRule(SingleRingParentRule):
    family = "positive-single-ring-parent"
    priority = parent_priority("polycyclic-functional-skeleton")

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() != 1 or _too_large_for_exhaustive_ring_parent_search(mol):
            return
        for ring in mol.GetRingInfo().AtomRings():
            for name, stereo_prefix, locant_seq in _named_ring_instance_names(mol, tuple(ring), self._Chem):
                if not _has_positive_ring_parent_name(name):
                    continue
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=_single_ring_candidate_sort_key(name),
                    stereo_prefix=stereo_prefix,
                    parent_atoms=frozenset(ring),
                    locants=_locants_from_sequence(locant_seq),
                    stereo_locants=_complete_single_ring_parent_stereo_locants(mol, locant_seq, self._Chem),
                    estimated_length=len(name),
                    render_rank=1,
                )


def _has_positive_ring_parent_name(name: str) -> bool:
    return bool(re.search(r"(?:^|-)[^-]+-\d+-ium(?:$|-)", name))


@dataclass(frozen=True, slots=True)
class _SingleRingParentFacts:
    ring: tuple[int, ...]
    structural_key: tuple


def _locants_from_sequence(seq: Sequence[int]) -> tuple[tuple[int, str], ...]:
    return tuple((idx, str(locant)) for locant, idx in enumerate(seq, start=1))


def _single_ring_parent_fact_sort_key(mol, ring: Sequence[int]) -> tuple:
    ring_set = set(ring)
    atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
    size = len(ring)
    aromatic_atoms = sum(1 for atom in atoms if atom.GetIsAromatic())
    hetero_atoms = sum(1 for atom in atoms if atom.GetSymbol() != "C")
    external_bonds = sum(
        1
        for atom in atoms
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() not in ring_set
    )
    return (
        0 if aromatic_atoms == size else 1,
        abs(size - 6),
        external_bonds,
        hetero_atoms,
        tuple(sorted(ring)),
    )


def _unfused_assembly_monocycle_parent_rings(ctx) -> tuple[tuple[int, ...], ...]:
    """Return monocyclic parent components for P-44 ring assemblies.

    Independent monocycles joined through exocyclic bonds are ring assemblies
    rather than fused/spiro systems.  Letting the monocycle renderer propose
    parents for these systems gives the parent selector a senior ring candidate
    and renders the other ring systems as substituents.  For mixed assemblies,
    only standalone monocyclic components are exposed here; fused/spiro systems
    remain owned by their dedicated parent rules.
    """

    mol = ctx.mol
    ring_count = mol.GetRingInfo().NumRings()
    if ring_count < 3:
        return ()
    monocyclic_systems = {
        topology.atoms
        for topology in ctx.ring_topologies
        if topology.cycle_rank == 1
    }
    if not monocyclic_systems:
        return ()
    parent_rings = []
    for ring in mol.GetRingInfo().AtomRings():
        ring_set = frozenset(ring)
        if ring_set in monocyclic_systems:
            parent_rings.append(tuple(ring))
    return tuple(parent_rings)


def _is_all_monocycle_ring_assembly(ctx) -> bool:
    ring_count = ctx.mol.GetRingInfo().NumRings()
    return len(ctx.ring_topologies) == ring_count and all(
        topology.cycle_rank == 1 for topology in ctx.ring_topologies
    )


class HydroFusedParentRule:
    family = "hydro-fused-retained-parent"
    priority = parent_priority("hydro-retained-fused-parent")
    min_ring_count = 2
    min_ring_system_ring_count = 2

    def __init__(self, Chem):
        self._Chem = Chem

    def _templates(self) -> tuple[dict, ...]:
        return _curated_fused_templates(self._Chem)

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.candidate_templates(self._templates(), exact_atom_count=True)

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        if _high_rank_hetero_retained_fused_search_unpromising(mol):
            return
        for template, match in _iter_fused_template_matches(
            ctx,
            self._template_iter(ctx),
            self._Chem,
            topology_query=True,
            require_one_ring_system=True,
            require_fully_aromatic=True,
        ):
            parent_name = _hydro_parent_name(mol, template, match)
            if not parent_name:
                continue
            name = _name_fused_core_match(mol, template, match, parent_name=parent_name)
            if name:
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=(-len(match), name),
                    parent_atoms=frozenset(match),
                    stereo_locants=_template_match_stereo_locants(template, match),
                    template_id=id(template),
                    template_name=template_record_name(template),
                    source_family=str(template.get("source_family", "")),
                    source_file=str(template.get("source_file", "")),
                    estimated_length=len(name),
                )


class GeneratedHydroFusedParentRule(HydroFusedParentRule):
    family = "generated-hydro-fused-parent"
    priority = parent_priority("generated-parent")
    min_ring_system_ring_count = 2

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.generated_fused_templates


class GeneratedPartialHydroFusedParentRule(GeneratedHydroFusedParentRule):
    family = "generated-partial-hydro-fused-parent"
    priority = parent_priority("generated-partial-hydro-parent")

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        if _high_rank_hetero_retained_fused_search_unpromising(mol):
            return
        facts = []
        for template, match in _iter_fused_template_matches(
            ctx,
            ctx.generated_fused_templates,
            self._Chem,
            topology_query=True,
            require_one_ring_system=True,
        ):
            for parent_name, hydro_key in _partial_hydro_parent_names(mol, template, match):
                fusion_key = _partial_hydro_fusion_preference(template_record_name(template))
                facts.append(
                    _GeneratedPartialHydroCandidateFacts(
                        template=template,
                        match=tuple(match),
                        parent_name=parent_name,
                        hydro_key=hydro_key,
                        structural_key=(fusion_key, hydro_key, -len(match), parent_name, tuple(match)),
                    )
                )
        if ctx.telemetry is not None:
            ctx.telemetry.candidate_facts_total[self.family] += len(facts)
        render_set = _generated_partial_hydro_render_set(tuple(facts))
        if ctx.telemetry is not None:
            ctx.telemetry.render_set_size[self.family] += len(render_set)
            ctx.telemetry.facts_suppressed_before_render[self.family] += max(0, len(facts) - len(render_set))
        candidates = []
        for render_rank, fact in enumerate(render_set, start=1):
            template = fact.template
            match = fact.match
            parent_name = fact.parent_name
            hydro_key = fact.hydro_key
            name = _name_fused_core_match(mol, template, match, parent_name=parent_name)
            if not name:
                continue
            fusion_key = _partial_hydro_fusion_preference(template_record_name(template))
            sort_key = (fusion_key, hydro_key, -len(match), name)
            candidates.append(
                NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=sort_key,
                    parent_atoms=frozenset(match),
                    stereo_locants=_template_match_stereo_locants(template, match),
                    template_id=id(template),
                    template_name=template_record_name(template),
                    source_family=str(template.get("source_family", "")),
                    source_file=str(template.get("source_file", "")),
                    estimated_length=len(name),
                    render_rank=render_rank,
                )
            )
        seen = set()
        yielded = 0
        for candidate in sorted(candidates, key=lambda item: item.sort_key):
            if candidate.name in seen:
                continue
            seen.add(candidate.name)
            yield candidate
            yielded += 1
            if yielded >= 16:
                return


@dataclass(frozen=True, slots=True)
class _GeneratedPartialHydroCandidateFacts:
    template: dict
    match: tuple[int, ...]
    parent_name: str
    hydro_key: tuple
    structural_key: tuple


def _generated_partial_hydro_render_set(
    facts: tuple[_GeneratedPartialHydroCandidateFacts, ...],
) -> tuple[_GeneratedPartialHydroCandidateFacts, ...]:
    if len(facts) <= _GENERATED_PARTIAL_HYDRO_RENDER_SET_SIZE:
        return facts
    return tuple(sorted(facts, key=lambda fact: fact.structural_key)[:_GENERATED_PARTIAL_HYDRO_RENDER_SET_SIZE])


class GeneratedSpiroPartialHydroParentRule(GeneratedPartialHydroFusedParentRule):
    family = "generated-spiro-partial-hydro-parent"
    priority = parent_priority("generated-spiro-component-parent")

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        candidates = []
        templates = ctx.candidate_templates(
            parent_scaffold_catalog(self._Chem).generated_spiro_scaffolds,
            exact_atom_count=True,
        )
        for template, match in _iter_fused_template_matches(
            ctx,
            templates,
            self._Chem,
            topology_query=True,
            require_one_ring_system=True,
        ):
            for parent_name, hydro_key in _partial_hydro_parent_names(mol, template, match):
                name = _name_fused_core_match(mol, template, match, parent_name=parent_name)
                if not name:
                    continue
                sort_key = (
                    _partial_hydro_spiro_alias_penalty(parent_name),
                    hydro_key,
                    -len(match),
                    name,
                )
                candidates.append(
                    NameCandidate(
                        name=name,
                        family=self.family,
                        priority=self.priority,
                        sort_key=sort_key,
                        parent_atoms=frozenset(match),
                        stereo_locants=_template_match_stereo_locants(template, match),
                        template_id=id(template),
                        template_name=template_record_name(template),
                        source_family=str(template.get("source_family", "")),
                        source_file=str(template.get("source_file", "")),
                        estimated_length=len(name),
                    )
                )
        seen = set()
        yielded = 0
        for candidate in sorted(candidates, key=lambda item: item.sort_key):
            if candidate.name in seen:
                continue
            seen.add(candidate.name)
            yield candidate
            yielded += 1
            if yielded >= 16:
                return


class ComponentSpiroParentRule:
    family = "component-spiro-parent"
    priority = parent_priority("component-spiro-parent")
    min_ring_system_ring_count = 3

    def candidates(self, ctx):
        yield from component_spiro_parent_candidates(ctx, self.priority)


_NATURAL_PRODUCT_TOPOLOGY_STEMS = {
    "androst",
    "campest",
    "chol",
    "cholest",
    "ergost",
    "estr",
    "gon",
    "poriferast",
    "pregn",
    "stigmast",
}


_MAX_EXHAUSTIVE_RING_PARENT_HEAVY_ATOMS = 54
_MAX_WHOLE_MOLECULE_RETAINED_RINGS = 6


def _too_large_for_exhaustive_ring_parent_search(mol) -> bool:
    return mol.GetNumHeavyAtoms() > _MAX_EXHAUSTIVE_RING_PARENT_HEAVY_ATOMS


def _too_many_rings_for_whole_molecule_retained_parent(mol) -> bool:
    return mol.GetRingInfo().NumRings() > _MAX_WHOLE_MOLECULE_RETAINED_RINGS


def _high_rank_hetero_retained_fused_search_unpromising(mol) -> bool:
    # Retained fused templates carry the P-25 locant maps for large hetero
    # fused systems. Skipping them here forces generic bicyclo numbering that
    # often parses as the wrong fusion orientation.
    return False


def _has_positive_charged_ring_atom(mol) -> bool:
    return any(atom.IsInRing() and atom.GetFormalCharge() > 0 for atom in mol.GetAtoms())


class NaturalProductTopologyParentRule(FusedParentRule):
    family = "natural-product-topology-parent"
    priority = parent_priority("whole-molecule-retained-parent")
    min_ring_system_ring_count = 4

    def _templates(self) -> tuple[dict, ...]:
        return tuple(
            template
            for template in _curated_fused_templates(self._Chem)
            if template.get("source_file") == "naturalProducts.xml"
            and template_record_name(template) in _NATURAL_PRODUCT_TOPOLOGY_STEMS
        )

    def _candidate_sort_key(self, template: dict, match: Sequence[int], name: str) -> tuple:
        return (
            -template_record_mol(template).GetNumAtoms(),
            _natural_product_terminal_side_chain_key(name),
            _substituent_locant_tuple(name),
            len(name),
            name,
        )

    def candidates(self, ctx):
        mol = ctx.mol
        if ctx.features.get("max_ring_system_ring_count") != 4:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        if _high_rank_hetero_retained_fused_search_unpromising(mol):
            return
        if _has_generic_polycyclic_stereo_fallback(ctx, mol, self._Chem):
            return
        seen = set()
        for template, match in _iter_fused_template_matches(
            ctx,
            ctx.candidate_templates(self._templates()),
            self._Chem,
            topology_query=True,
            reject_extra_core_bonds=True,
        ):
            named = _name_natural_product_topology_match(mol, self._Chem, template, match)
            if not named:
                continue
            name, stereo_prefix = named
            key = (name, stereo_prefix)
            if key in seen:
                continue
            seen.add(key)
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=self.priority,
                sort_key=self._candidate_sort_key(template, match, name),
                stereo_prefix=stereo_prefix,
            )


def _natural_product_terminal_side_chain_key(name: str) -> tuple[int, int]:
    match = re.search(
        r"(?:^|-)(2[0-9])-(?:hydroxy|oxido)-\1-oxo-(?:cholest|cholesta|ergost|ergosta|stigmast|stigmasta)-25-ene\b",
        name,
    )
    if match is None:
        return (0, 0)
    return (-int(match.group(1)), 1)


def _should_defer_natural_product_to_generic_polycycle(ctx, mol, Chem, template: dict) -> bool:
    return (
        template.get("source_file") == "naturalProducts.xml"
        and template_record_name(template) in _NATURAL_PRODUCT_TOPOLOGY_STEMS
        and _has_generic_polycyclic_stereo_fallback(ctx, mol, Chem)
    )


def _has_generic_polycyclic_stereo_fallback(ctx, mol, Chem) -> bool:
    return (
        _has_explicit_stereochemistry(mol, Chem)
        and not _has_nonring_carbon_substituent_chain(mol, Chem)
        and bool(ctx.polycyclic_parent_records("kekulized"))
    )


def _has_explicit_stereochemistry(mol, Chem) -> bool:
    return any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()
    ) or any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        for bond in mol.GetBonds()
    )


def _has_nonring_carbon_substituent_chain(mol, Chem) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or atom.IsInRing():
            continue
        ring_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.IsInRing()]
        if len(ring_neighbors) != 1:
            return True
        ring_neighbor = ring_neighbors[0]
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), ring_neighbor.GetIdx())
        if bond.GetBondType() == Chem.BondType.DOUBLE and atom.GetDegree() == 1:
            continue
        return True
    return False


class SkeletalReplacementFusedParentRule(FusedParentRule):
    family = "skeletal-replacement-fused-parent"
    priority = parent_priority("skeletal-replacement-parent")
    min_ring_system_ring_count = 3

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        if _high_rank_hetero_retained_fused_search_unpromising(mol):
            return
        facts: list[_SkeletalReplacementCandidateFacts] = []
        stage_counts: Counter[str] = Counter()
        for atom_set, feature in zip(ctx.ring_systems, ctx.features["ring_systems"], strict=True):
            if feature["ring_count"] not in {3, 4} or set(feature["elements"]) <= {"C"}:
                continue
            stage_counts["ring_systems"] += 1
            feature = ring_system_with_graph_signatures(mol, feature, self._Chem)
            key = (feature["atom_count"], feature["ring_count"], feature.get("graph_signature", ""))
            templates = _hydrocarbon_replacement_parent_templates_for_graph(*key, self._Chem)
            stage_counts["graph_templates"] += len(templates)
            for template in templates:
                query = _skeletal_replacement_query(template_record_mol(template), self._Chem)
                if query.GetNumAtoms() != len(atom_set):
                    continue
                stage_counts["substructure_queries"] += 1
                for match in ctx.substruct_matches(query):
                    stage_counts["matches"] += 1
                    if set(match) != set(atom_set):
                        continue
                    stage_counts["after_atom_set"] += 1
                    parent_name = _skeletal_replacement_parent_name(mol, template, match)
                    if not parent_name:
                        continue
                    stage_counts["facts"] += 1
                    facts.append(
                        _SkeletalReplacementCandidateFacts(
                            template=template,
                            match=tuple(match),
                            parent_name=parent_name,
                            structural_key=(
                                _skeletal_replacement_sort_key(mol, match, parent_name),
                                _template_source_rank(template),
                                _hydrogenated_template_penalty(template_record_name(template)),
                                len(parent_name),
                                parent_name,
                                tuple(match),
                            ),
                        )
                    )
                    if len(facts) >= _SKELETAL_REPLACEMENT_FACT_SCAN_SIZE:
                        yield from self._render_facts(ctx, tuple(facts), stage_counts)
                        return

        yield from self._render_facts(ctx, tuple(facts), stage_counts)

    def _render_facts(
        self,
        ctx,
        facts: tuple["_SkeletalReplacementCandidateFacts", ...],
        stage_counts: Counter[str],
    ):
        if ctx.telemetry is not None:
            for stage, count in stage_counts.items():
                ctx.telemetry.filter_stage_count[f"{self.family}:{stage}"] += count
            ctx.telemetry.candidate_facts_total[self.family] += len(facts)
        render_set = _skeletal_replacement_render_set(facts)
        if ctx.telemetry is not None:
            ctx.telemetry.render_set_size[self.family] += len(render_set)
            ctx.telemetry.facts_suppressed_before_render[self.family] += max(0, len(facts) - len(render_set))

        mol = ctx.mol
        yielded_names: set[str] = set()
        yielded = 0
        for render_rank, fact in enumerate(render_set, start=1):
            name = _name_fused_core_match(mol, fact.template, fact.match, parent_name=fact.parent_name)
            name = _opsin_parseable_retained_parent_name(name)
            if not name or name in yielded_names:
                continue
            if _unsafe_benz_a_anthracene_candidate(name):
                continue
            yielded_names.add(name)
            yielded += 1
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=self.priority,
                sort_key=(
                    _skeletal_replacement_sort_key(mol, fact.match, fact.parent_name),
                    _substituent_locant_tuple(name),
                    len(name),
                    name,
                ),
                parent_atoms=frozenset(fact.match),
                stereo_locants=_template_match_stereo_locants(fact.template, fact.match),
                template_id=id(fact.template),
                template_name=template_record_name(fact.template),
                source_family=str(fact.template.get("source_family", "")),
                source_file=str(fact.template.get("source_file", "")),
                estimated_length=len(name),
                render_rank=render_rank,
            )
            if yielded >= _SKELETAL_REPLACEMENT_RENDER_SET_SIZE:
                return


@dataclass(frozen=True, slots=True)
class _SkeletalReplacementCandidateFacts:
    template: dict
    match: tuple[int, ...]
    parent_name: str
    structural_key: tuple


def _skeletal_replacement_render_set(
    facts: tuple[_SkeletalReplacementCandidateFacts, ...],
) -> tuple[_SkeletalReplacementCandidateFacts, ...]:
    if len(facts) <= _SKELETAL_REPLACEMENT_RENDER_SET_SIZE:
        return facts
    return tuple(sorted(facts, key=lambda fact: fact.structural_key)[:_SKELETAL_REPLACEMENT_RENDER_SET_SIZE])


class RetainedTopologyReplacementFusedParentRule(FusedParentRule):
    family = "retained-topology-replacement-fused-parent"
    priority = parent_priority("skeletal-replacement-parent")
    min_ring_system_ring_count = 2

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        facts: list[_RetainedTopologyReplacementCandidateFacts] = []
        stage_counts: Counter[str] = Counter()
        for atom_set, feature in zip(ctx.ring_systems, ctx.features["ring_systems"], strict=True):
            if feature["ring_count"] < 2 or feature["ring_count"] > 5:
                continue
            if set(feature["elements"]) <= {"C"}:
                continue
            stage_counts["ring_systems"] += 1
            feature = ring_system_with_graph_signatures(mol, feature, self._Chem)
            key = (feature["atom_count"], feature["ring_count"], feature.get("graph_signature", ""))
            templates = _retained_topology_replacement_parent_templates_for_graph(*key, self._Chem)
            stage_counts["graph_templates"] += len(templates)
            for template in templates:
                if _retained_topology_replacement_template_has_nh(template):
                    continue
                stage_counts["after_template_nh"] += 1
                if not _retained_topology_replacement_template_elements_possible(feature["elements"], template):
                    continue
                stage_counts["after_element_counts"] += 1
                query = _skeletal_replacement_query(template_record_mol(template), self._Chem)
                if query.GetNumAtoms() != len(atom_set):
                    continue
                stage_counts["substructure_queries"] += 1
                matches = ctx.substruct_matches(query)
                stage_counts["matches"] += len(matches)
                for match in matches:
                    if set(match) != set(atom_set):
                        continue
                    stage_counts["after_atom_set"] += 1
                    if not _retained_topology_fused_match_is_hydrogen_compatible(mol, template, match):
                        continue
                    stage_counts["after_hydrogen_compatibility"] += 1
                    parent_name = _retained_topology_replacement_parent_name(mol, template, match)
                    if not parent_name:
                        continue
                    stage_counts["facts"] += 1
                    facts.append(
                        _RetainedTopologyReplacementCandidateFacts(
                            template=template,
                            match=tuple(match),
                            parent_name=parent_name,
                            structural_key=(
                                _skeletal_replacement_sort_key(mol, match, parent_name),
                                _template_source_rank(template),
                                _hydrogenated_template_penalty(template_record_name(template)),
                                len(parent_name),
                                parent_name,
                                tuple(match),
                            ),
                        ),
                    )
                    if len(facts) >= _RETAINED_TOPOLOGY_REPLACEMENT_FACT_SCAN_SIZE:
                        yield from self._render_facts(ctx, tuple(facts), stage_counts)
                        return

        yield from self._render_facts(ctx, tuple(facts), stage_counts)

    def _render_facts(
        self,
        ctx,
        facts: tuple["_RetainedTopologyReplacementCandidateFacts", ...],
        stage_counts: Counter[str],
    ):
        if ctx.telemetry is not None:
            for stage, count in stage_counts.items():
                ctx.telemetry.filter_stage_count[f"{self.family}:{stage}"] += count
            ctx.telemetry.candidate_facts_total[self.family] += len(facts)
        render_set = _retained_topology_replacement_render_set(facts)
        if ctx.telemetry is not None:
            ctx.telemetry.render_set_size[self.family] += len(render_set)
            ctx.telemetry.facts_suppressed_before_render[self.family] += max(0, len(facts) - len(render_set))

        mol = ctx.mol
        yielded_names: set[str] = set()
        yielded = 0
        for render_rank, fact in enumerate(render_set, start=1):
            name = _name_fused_core_match(mol, fact.template, fact.match, parent_name=fact.parent_name)
            name = _opsin_parseable_retained_parent_name(name)
            if not name or name in yielded_names:
                continue
            if _unsafe_retained_ring_scaffold_candidate(name):
                continue
            if _unsafe_benz_a_anthracene_candidate(name):
                continue
            yielded_names.add(name)
            yielded += 1
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=self.priority,
                sort_key=(
                    _skeletal_replacement_sort_key(mol, fact.match, fact.parent_name),
                    _substituent_locant_tuple(name),
                    len(name),
                    name,
                ),
                parent_atoms=frozenset(fact.match),
                stereo_locants=_template_match_stereo_locants(fact.template, fact.match),
                template_id=id(fact.template),
                template_name=template_record_name(fact.template),
                source_family=str(fact.template.get("source_family", "")),
                source_file=str(fact.template.get("source_file", "")),
                estimated_length=len(name),
                render_rank=render_rank,
            )
            if yielded >= _MAX_RETAINED_TOPOLOGY_REPLACEMENT_CANDIDATES:
                return


@dataclass(frozen=True, slots=True)
class _RetainedTopologyReplacementCandidateFacts:
    template: dict
    match: tuple[int, ...]
    parent_name: str
    structural_key: tuple


def _retained_topology_replacement_template_elements_possible(
    molecule_elements: Counter[str],
    template: dict,
) -> bool:
    return all(
        count <= molecule_elements.get(symbol, 0)
        for symbol, count in _retained_topology_replacement_template_noncarbon_elements(
            template_record_mol(template)
        )
    )


@lru_cache(maxsize=None)
def _retained_topology_replacement_template_noncarbon_elements(template_mol) -> tuple[tuple[str, int], ...]:
    counts = Counter(
        atom.GetSymbol()
        for atom in template_mol.GetAtoms()
        if atom.GetSymbol() != "C"
    )
    return tuple(sorted(counts.items()))


def _retained_topology_replacement_template_has_nh(template: dict) -> bool:
    return _retained_topology_replacement_template_mol_has_nh(template_record_mol(template))


@lru_cache(maxsize=None)
def _retained_topology_replacement_template_mol_has_nh(template_mol) -> bool:
    return any(
        atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0
        for atom in template_mol.GetAtoms()
    )


def _retained_topology_replacement_render_set(
    facts: tuple[_RetainedTopologyReplacementCandidateFacts, ...],
) -> tuple[_RetainedTopologyReplacementCandidateFacts, ...]:
    if len(facts) <= _RETAINED_TOPOLOGY_REPLACEMENT_RENDER_SET_SIZE:
        return facts
    return tuple(
        sorted(facts, key=lambda fact: fact.structural_key)[
            :_RETAINED_TOPOLOGY_REPLACEMENT_RENDER_SET_SIZE
        ]
    )


class OxidoFusedParentRule:
    family = "oxido-fused-retained-parent"
    priority = parent_priority("charged-retained-fused-parent")
    min_ring_count = 2
    min_ring_system_ring_count = 2
    requires_charged_ring_atom = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        templates = ctx.candidate_templates(_curated_fused_templates(self._Chem))
        for template, match in _iter_fused_template_matches(
            ctx,
            templates,
            self._Chem,
            topology_query=True,
            require_one_ring_system=True,
        ):
            if not _has_n_oxide_on_match(mol, match):
                continue
            reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
            parent_name = _adjust_fused_parent_for_charge(mol, template, reverse)
            if parent_name == template_record_name(template):
                continue
            name = _name_fused_core_match(mol, template, match, parent_name=parent_name)
            if name:
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=(-len(match), name),
                    parent_atoms=frozenset(match),
                    stereo_locants=_template_match_stereo_locants(template, match),
                    template_id=id(template),
                    template_name=template_record_name(template),
                    source_family=str(template.get("source_family", "")),
                    source_file=str(template.get("source_file", "")),
                    estimated_length=len(name),
                )


class GeneratedOxidoFusedParentRule(OxidoFusedParentRule):
    family = "generated-oxido-fused-parent"
    priority = parent_priority("charged-retained-fused-parent")
    min_ring_system_ring_count = 2

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.generated_fused_templates

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        for template, match in _iter_fused_template_matches(
            ctx,
            self._template_iter(ctx),
            self._Chem,
            topology_query=True,
            require_one_ring_system=True,
        ):
            if not _has_n_oxide_on_match(mol, match):
                continue
            reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
            parent_name = _adjust_fused_parent_for_charge(mol, template, reverse)
            if parent_name == template_record_name(template):
                continue
            name = _name_fused_core_match(mol, template, match, parent_name=parent_name)
            if name:
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=(
                        _generated_fusion_preference(template_record_name(template)),
                        _substituent_locant_tuple(name),
                        -len(match),
                        name,
                    ),
                )


class _PolycyclicRecordParentRule:
    record_family = ""
    min_ring_count = 2

    def __init__(self, Chem):
        self._Chem = Chem

    def _candidate_priority(self, mol) -> int | None:
        return self.priority

    def _accept_record(self, name: str, seq: Sequence[int]) -> bool:
        return True

    def _defer_record(self, ctx, name: str, seq: Sequence[int]) -> bool:
        return False

    def _defer_all_records(self, ctx) -> bool:
        return False

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        priority = self._candidate_priority(mol)
        if priority is None:
            return
        if self._defer_all_records(ctx):
            return
        for record_order, (name, seq) in enumerate(ctx.polycyclic_parent_records(self.record_family)):
            if not self._accept_record(name, seq):
                continue
            if self._defer_record(ctx, name, seq):
                continue
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=priority,
                sort_key=(record_order, len(name), name),
                stereo_prefix=_polycyclic_stereo_prefix(mol, seq, self._Chem, name=name),
                parent_atoms=frozenset(seq),
                stereo_locants=_sequence_stereo_locants(seq),
                source_family=self.record_family,
                estimated_length=len(name),
                render_rank=record_order + 1,
            )


class GenericBicyclicParentRule(_PolycyclicRecordParentRule):
    family = "generic-bicyclic-parent"
    priority = parent_priority("generic-bicyclic-parent")
    record_family = "bicyclic"
    requires_generic_bicyclic_topology = True


_FUNCTIONAL_GENERIC_BICYCLIC_TERMS = (
    "carbonyl",
    "carboxy",
    "cyano",
    "amino",
    "hydroxy",
    "imino",
    "methylidene",
    "oxo",
    "oxy",
    "sulfonyl",
    "thioxo",
)


class FunctionalGenericBicyclicParentRule(GenericBicyclicParentRule):
    family = "functional-generic-bicyclic-parent"
    priority = parent_priority("polycyclic-functional-skeleton")

    def _accept_record(self, name: str, seq: Sequence[int]) -> bool:
        return any(term in name for term in _FUNCTIONAL_GENERIC_BICYCLIC_TERMS)

    def _defer_record(self, ctx, name: str, seq: Sequence[int]) -> bool:
        if not ctx.polycyclic_parent_records("kekulized"):
            return False
        return _has_retained_fused_parent_substituent_name(name)


class PreferredGenericBicyclicParentRule(GenericBicyclicParentRule):
    family = "preferred-generic-bicyclic-parent"
    priority = parent_priority("preferred-generic-bicyclic")
    min_ring_system_ring_count = 2

    def _accept_record(self, name: str, seq: Sequence[int]) -> bool:
        return "azabicyclo[3.1.0]hexane" in name


class GenericSpiroParentRule(_PolycyclicRecordParentRule):
    family = "generic-spiro-parent"
    priority = parent_priority("generic-spiro-parent")
    record_family = "spiro"
    min_ring_system_ring_count = 2
    requires_spiro_topology = True


class KekulizedGenericPolycyclicParentRule(_PolycyclicRecordParentRule):
    family = "kekulized-generic-polycyclic-parent"
    priority = parent_priority("polycyclic-functional-skeleton")
    record_family = "kekulized"
    defer_to_exocyclic_charged_fused_parent = True
    requires_kekulized_polycycle_signal = True
    requires_generic_polycyclic_topology = True

    def _candidate_priority(self, mol) -> int | None:
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return None
        ring_count = mol.GetRingInfo().NumRings()
        aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
        if not aromatic and ring_count < 3:
            return None
        return self.priority if aromatic else parent_priority("generic-polycyclic-parent")

    def _defer_record(self, ctx, name: str, seq: Sequence[int]) -> bool:
        return _has_retained_fused_parent_candidate(ctx, self._Chem)

    def _defer_all_records(self, ctx) -> bool:
        return _has_retained_fused_parent_candidate(ctx, self._Chem)


def _has_retained_fused_parent_candidate(ctx, Chem) -> bool:
    mol = ctx.mol
    cached = mol_cache_get(mol, "has_retained_fused_parent_candidate")
    if cached is not None:
        return cached
    template_sets = (
        ctx.candidate_templates(_curated_fused_templates(Chem)),
        ctx.generated_fused_templates,
    )
    for templates in template_sets:
        for template, match in _iter_fused_template_matches(
            ctx,
            templates,
            Chem,
            reject_extra_core_bonds=True,
        ):
            name = _name_fused_core_match(mol, template, match)
            if name and not _unsafe_benz_a_anthracene_candidate(name):
                return mol_cache_set(mol, "has_retained_fused_parent_candidate", True)
    return mol_cache_set(mol, "has_retained_fused_parent_candidate", False)


class ExactRetainedParentRule:
    family = "exact-retained-parent"
    priority = parent_priority("exact-retained-parent")
    min_ring_count = 1
    requires_exact_ring_system = True

    def __init__(self, Chem):
        self._Chem = Chem

    def _templates(self) -> tuple[dict, ...]:
        return _curated_exact_parent_templates(self._Chem)

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.candidate_templates(self._templates(), exact_atom_count=True)

    def _sort_key(self, template: dict) -> tuple:
        name = _preferred_parent_stem(template_record_name(template))
        return (len(name), template.get("source_family", ""), name)

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() == 0 or not ctx.features["all_atoms_in_one_ring_system"]:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        if _high_rank_hetero_retained_fused_search_unpromising(mol):
            return
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        for template in self._template_iter(ctx):
            if template.get("stereochemical") and not _has_stereo_evidence(mol, self._Chem):
                continue
            if template_record_mol(template).GetNumAtoms() != atom_count:
                continue
            if _formal_charge(template_record_mol(template)) != _formal_charge(mol):
                continue
            for match in ctx.template_matches(template):
                if set(match) == atom_set and not _match_has_extra_core_bonds(mol, template, match):
                    family = (
                        "exact-biochemical-retained-parent"
                        if template.get("source_family") == "static-exact-retained"
                        else self.family
                    )
                    yield NameCandidate(
                        name=_opsin_parseable_retained_parent_name(template_record_name(template)),
                        family=family,
                        priority=self.priority,
                        sort_key=self._sort_key(template),
                    )


class ExactFunctionalRetainedParentRule(ExactRetainedParentRule):
    family = "exact-functional-retained-parent"
    requires_exact_ring_system = False

    def _template_iter(self, ctx) -> Sequence[dict]:
        return tuple(
            template
            for template in ctx.candidate_templates(self._templates(), exact_atom_count=True)
            if template.get("source_family") in {"static-retained", "static-exact-retained"}
        )

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() == 0:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        for template in self._template_iter(ctx):
            if template.get("stereochemical") and not _has_stereo_evidence(mol, self._Chem):
                continue
            if template_record_mol(template).GetNumAtoms() != atom_count:
                continue
            if _formal_charge(template_record_mol(template)) != _formal_charge(mol):
                continue
            for match in ctx.template_matches(template):
                if set(match) == atom_set and not _match_has_extra_core_bonds(mol, template, match):
                    family = (
                        "exact-biochemical-retained-parent"
                        if template.get("source_family") == "static-exact-retained"
                        else self.family
                    )
                    yield NameCandidate(
                        name=_opsin_parseable_retained_parent_name(template_record_name(template)),
                        family=family,
                        priority=self.priority,
                        sort_key=self._sort_key(template),
                    )


class GeneratedExactParentRule(ExactRetainedParentRule):
    family = "generated-exact-parent"
    priority = parent_priority("generated-parent")

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.generated_fused_templates

    def _sort_key(self, template: dict) -> tuple:
        return _generated_exact_parent_sort_key(template)


class ExactBiochemicalRetainedParentRule(ExactFunctionalRetainedParentRule):
    family = "exact-biochemical-retained-parent"

    def _template_iter(self, ctx) -> Sequence[dict]:
        return tuple(
            template
            for template in ctx.candidate_templates(self._templates(), exact_atom_count=True)
            if template.get("source_family") == "static-exact-retained"
        )


def parent_rules(Chem) -> tuple:
    sulfanium = SulfaniumParentRule(Chem)
    small_inorganic = SmallInorganicRetainedParentRule(Chem)
    phosphanium = PhosphaniumParentRule(Chem)
    azanium = AzaniumParentRule(Chem)
    neutral_amine = NeutralAmineParentRule(Chem)
    group14_hydride = Group14HydrideParentRule(Chem)
    phosphorus_acid = PhosphorusAcidParentRule(Chem)
    phosphite_ester = PhosphiteEsterParentRule(Chem)
    organophosphorus_oxide = OrganophosphorusOxideParentRule(Chem)
    phosphorane = PhosphoraneParentRule(Chem)
    organomagnesium_halide = OrganomagnesiumHalideParentRule(Chem)
    organoborane = OrganoboraneParentRule(Chem)
    benziodoxole = BenziodoxoleRetainedParentRule(Chem)
    iodane = IodaneParentRule(Chem)
    iodonium = IodoniumParentRule(Chem)
    pnictogen_hydride = PnictogenHydrideParentRule(Chem)
    oxido_nitroso_azanium = OxidoNitrosoAzaniumRule(Chem)
    oxo_sulfanium = OxoSulfaniumParentRule(Chem)
    n_halo_imine = NHaloImineParentRule(Chem)
    guanidine = GuanidineRetainedParentRule(Chem)
    sulfonohydrazide = SulfonohydrazideRetainedParentRule(Chem)
    retained_acid_ester = RetainedAcidEsterParentRule(Chem)
    retained_amide = RetainedAmideParentRule(Chem)
    beta_dicarbonyl = BetaDicarbonylParentRule(Chem)
    boronic_pinacol_ester = BoronicPinacolEsterParentRule(Chem)
    carbamate = CarbamateRetainedParentRule(Chem)
    sulfonamide = SulfonamideRetainedParentRule(Chem)
    n_substituted_sulfonamide = SulfonamideParentRule(Chem)
    urea = UreaRetainedParentRule(Chem)
    glutarimide = GlutarimideRetainedParentRule(Chem)
    amino_acid = AminoAcidRetainedParentRule(Chem)
    biphenyl = BiphenylParentRule(Chem)
    diphenylbenzene = DiphenylbenzeneParentRule(Chem)
    acene = AceneParentRule(Chem)
    acyclic_parent = AcyclicParentRule()
    imino_functional_acyclic = IminoFunctionalAcyclicSkeletonRule()
    conjugated_carbonyl_acyclic = ConjugatedCarbonylAcyclicSkeletonRule()
    whole_molecule_retained = WholeMoleculeRetainedParentRule(Chem)
    exact_retained = ExactRetainedParentRule(Chem)
    retained_ring_scaffold = RetainedRingScaffoldParentRule(Chem)
    exact_functional_retained = ExactFunctionalRetainedParentRule(Chem)
    exact_biochemical_retained = ExactBiochemicalRetainedParentRule(Chem)
    preferred_bicyclic = PreferredGenericBicyclicParentRule(Chem)
    fused = FusedParentRule(Chem)
    hydro_fused = HydroFusedParentRule(Chem)
    oxido_fused = OxidoFusedParentRule(Chem)
    generated_oxido_fused = GeneratedOxidoFusedParentRule(Chem)
    generic_bicyclic = GenericBicyclicParentRule(Chem)
    functional_generic_bicyclic = FunctionalGenericBicyclicParentRule(Chem)
    generic_spiro = GenericSpiroParentRule(Chem)
    single_ring = SingleRingParentRule(Chem)
    positive_single_ring = PositiveSingleRingParentRule(Chem)
    generated_exact = GeneratedExactParentRule(Chem)
    generated_fused = GeneratedFusedParentRule(Chem)
    generated_spiro_fused = GeneratedSpiroFusedParentRule(Chem)
    generated_hydro_fused = GeneratedHydroFusedParentRule(Chem)
    generated_partial_hydro = GeneratedPartialHydroFusedParentRule(Chem)
    generated_spiro_partial_hydro = GeneratedSpiroPartialHydroParentRule(Chem)
    component_spiro = ComponentSpiroParentRule()
    natural_product_topology = NaturalProductTopologyParentRule(Chem)
    skeletal_replacement = SkeletalReplacementFusedParentRule(Chem)
    retained_topology_replacement = RetainedTopologyReplacementFusedParentRule(Chem)
    kekulized_polycyclic = KekulizedGenericPolycyclicParentRule(Chem)
    preferred_acyclic = PreferredAcyclicSkeletonWithRingSubstituentsRule()
    chain_dominant_acyclic = ChainDominantAcyclicSkeletonWithRingSubstituentsRule()
    spiro_functional_chain = SpiroFunctionalChainSubstituentRule()
    charged_ring_acyclic = ChargedRingAcyclicSkeletonWithRingSubstituentsRule()
    functional_acyclic = FunctionalAcyclicSkeletonWithRingSubstituentsRule()
    acyclic_with_ring_substituents = AcyclicSkeletonWithRingSubstituentsRule()

    return (
        RuleSpec(
            "small-inorganic-retained-parent",
            parent_priority("charged-inorganic-retained-parent"),
            (),
            small_inorganic.candidates,
        ),
        RuleSpec("sulfanium-parent", parent_priority("senior-functional-exception"), (nonring_positive_sulfur,), sulfanium.candidates),
        RuleSpec("phosphanium-parent", parent_priority("senior-functional-exception"), (nonring_positive_phosphorus,), phosphanium.candidates),
        RuleSpec("azanium-parent", parent_priority("senior-functional-exception"), (nonring_positive_nitrogen,), azanium.candidates),
        RuleSpec("neutral-amine-parent", parent_priority("senior-functional-exception"), (acyclic,), neutral_amine.candidates),
        RuleSpec("group14-hydride-parent", parent_priority("senior-functional-exception"), (nonring_neutral_group14,), group14_hydride.candidates),
        RuleSpec("phosphorus-acid-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), phosphorus_acid.candidates),
        RuleSpec("phosphite-ester-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), phosphite_ester.candidates),
        RuleSpec("organophosphorus-oxide-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), organophosphorus_oxide.candidates),
        RuleSpec("phosphorane-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), phosphorane.candidates),
        RuleSpec("organomagnesium-halide-parent", parent_priority("senior-functional-exception"), (acyclic,), organomagnesium_halide.candidates),
        RuleSpec("organoborane-parent", parent_priority("senior-functional-exception"), (), organoborane.candidates),
        RuleSpec("benziodoxole-retained-parent", parent_priority("senior-functional-exception"), (), benziodoxole.candidates),
        RuleSpec("iodane-parent", parent_priority("senior-functional-exception"), (), iodane.candidates),
        RuleSpec("iodonium-parent", parent_priority("senior-functional-exception"), (), iodonium.candidates),
        RuleSpec("pnictogen-hydride-parent", parent_priority("senior-functional-exception"), (nonring_neutral_pnictogen,), pnictogen_hydride.candidates),
        RuleSpec(
            "oxido-nitroso-azanium-parent",
            parent_priority("senior-functional-exception"),
            (acyclic, nonring_positive_nitrogen),
            oxido_nitroso_azanium.candidates,
        ),
        RuleSpec("oxo-sulfanium-parent", parent_priority("senior-functional-exception"), (nonring_positive_sulfur,), oxo_sulfanium.candidates),
        RuleSpec("n-halo-imine-parent", parent_priority("senior-functional-exception"), (acyclic, imino_skeleton_signal), n_halo_imine.candidates),
        RuleSpec("guanidine-retained-parent", parent_priority("senior-functional-exception"), (acyclic, imino_skeleton_signal), guanidine.candidates),
        RuleSpec("sulfonohydrazide-retained-parent", parent_priority("senior-functional-exception"), (imino_skeleton_signal,), sulfonohydrazide.candidates),
        RuleSpec("retained-acid-ester-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), retained_acid_ester.candidates),
        RuleSpec("retained-amide-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), retained_amide.candidates),
        RuleSpec("beta-dicarbonyl-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), beta_dicarbonyl.candidates),
        RuleSpec("boronic-pinacol-ester-parent", parent_priority("senior-functional-exception"), (min_ring_count(1),), boronic_pinacol_ester.candidates),
        RuleSpec("carbamate-retained-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), carbamate.candidates),
        RuleSpec("sulfonamide-retained-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), sulfonamide.candidates),
        RuleSpec("sulfonamide-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), n_substituted_sulfonamide.candidates),
        RuleSpec("urea-retained-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), urea.candidates),
        RuleSpec("glutarimide-retained-parent", parent_priority("senior-functional-exception"), (min_ring_count(1), functional_skeleton_signal), glutarimide.candidates),
        RuleSpec("amino-acid-retained-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), amino_acid.candidates),
        RuleSpec("biphenyl-parent", parent_priority("whole-molecule-retained-parent"), (min_ring_count(2),), biphenyl.candidates),
        RuleSpec("diphenylbenzene-parent", parent_priority("whole-molecule-retained-parent"), (min_ring_count(3),), diphenylbenzene.candidates),
        RuleSpec("acene-retained-parent", parent_priority("whole-molecule-retained-parent"), (min_ring_count(5),), acene.candidates),
        RuleSpec("acyclic-parent", parent_priority("acyclic-parent"), (acyclic,), acyclic_parent.candidates),
        RuleSpec(
            "imino-functional-acyclic-skeleton-with-ring-substituents",
            parent_priority("functional-acyclic-skeleton"),
            (imino_skeleton_signal, no_exocyclic_charged_fused_parent),
            imino_functional_acyclic.candidates,
        ),
        RuleSpec(
            "conjugated-carbonyl-acyclic-skeleton-with-ring-substituents",
            parent_priority("functional-acyclic-skeleton"),
            (
                functional_skeleton_signal,
                acyclic_preference_signal,
                conjugated_carbonyl_diene_signal,
                no_charged_ring_atom,
            ),
            conjugated_carbonyl_acyclic.candidates,
        ),
        RuleSpec(
            "functional-generic-bicyclic-parent",
            parent_priority("polycyclic-functional-skeleton"),
            (generic_bicyclic_topology,),
            functional_generic_bicyclic.candidates,
        ),
        RuleSpec(
            "preferred-acyclic-skeleton-with-ring-substituents",
            parent_priority("polycyclic-functional-skeleton"),
            (acyclic_preference_signal,),
            preferred_acyclic.candidates,
        ),
        RuleSpec(
            "positive-single-ring-parent",
            parent_priority("polycyclic-functional-skeleton"),
            (charged_ring_atom,),
            positive_single_ring.candidates,
        ),
        RuleSpec(
            "chain-dominant-acyclic-skeleton-with-ring-substituents",
            parent_priority("spiro-functional-chain-substituent"),
            (min_ring_count(1),),
            chain_dominant_acyclic.candidates,
        ),
        RuleSpec(
            "spiro-functional-chain-substituent",
            parent_priority("senior-spiro-functional-chain-substituent"),
            (min_ring_count(1), spiro_topology),
            spiro_functional_chain.candidates,
        ),
        RuleSpec(
            "charged-ring-acyclic-skeleton-with-ring-substituents",
            parent_priority("polycyclic-functional-skeleton"),
            (charged_ring_atom,),
            charged_ring_acyclic.candidates,
        ),
        RuleSpec(
            "whole-molecule-retained-parent",
            parent_priority("whole-molecule-retained-parent"),
            (),
            whole_molecule_retained.candidates,
        ),
        RuleSpec("exact-retained-parent", parent_priority("exact-retained-parent"), (min_ring_count(1), exact_ring_system), exact_retained.candidates),
        RuleSpec(
            "retained-ring-scaffold-parent",
            parent_priority("retained-fused-parent"),
            (min_ring_system_ring_count(2),),
            retained_ring_scaffold.candidates,
        ),
        RuleSpec(
            "exact-functional-retained-parent",
            parent_priority("exact-retained-parent"),
            (min_ring_count(1), functional_skeleton_signal),
            exact_functional_retained.candidates,
        ),
        RuleSpec(
            "exact-biochemical-retained-parent",
            parent_priority("exact-retained-parent"),
            (min_ring_count(1),),
            exact_biochemical_retained.candidates,
        ),
        RuleSpec(
            "preferred-generic-bicyclic-parent",
            parent_priority("preferred-generic-bicyclic"),
            (min_ring_system_ring_count(2), generic_bicyclic_topology),
            preferred_bicyclic.candidates,
        ),
        RuleSpec("fused-retained-parent", parent_priority("retained-fused-parent"), (min_ring_system_ring_count(2),), fused.candidates),
        RuleSpec("hydro-fused-retained-parent", parent_priority("hydro-retained-fused-parent"), (min_ring_count(2), min_ring_system_ring_count(2)), hydro_fused.candidates),
        RuleSpec(
            "generated-spiro-partial-hydro-parent",
            parent_priority("generated-spiro-component-parent"),
            (min_ring_system_ring_count(2), spiro_topology),
            generated_spiro_partial_hydro.candidates,
        ),
        RuleSpec(
            "component-spiro-parent",
            parent_priority("component-spiro-parent"),
            (min_ring_system_ring_count(3),),
            component_spiro.candidates,
        ),
        RuleSpec(
            "generated-spiro-fused-parent",
            parent_priority("generated-spiro-component-parent"),
            (min_ring_system_ring_count(2), spiro_topology),
            generated_spiro_fused.candidates,
        ),
        RuleSpec(
            "oxido-fused-retained-parent",
            parent_priority("charged-retained-fused-parent"),
            (min_ring_count(2), min_ring_system_ring_count(2), charged_ring_atom),
            oxido_fused.candidates,
        ),
        RuleSpec(
            "generated-oxido-fused-parent",
            parent_priority("charged-retained-fused-parent"),
            (min_ring_count(2), min_ring_system_ring_count(2), charged_ring_atom),
            generated_oxido_fused.candidates,
        ),
        RuleSpec("generic-bicyclic-parent", parent_priority("generic-bicyclic-parent"), (generic_bicyclic_topology,), generic_bicyclic.candidates),
        RuleSpec("generic-spiro-parent", parent_priority("generic-spiro-parent"), (min_ring_system_ring_count(2), spiro_topology), generic_spiro.candidates),
        RuleSpec("single-ring-parent", parent_priority("single-ring-parent"), (min_ring_count(1),), single_ring.candidates),
        RuleSpec("generated-exact-parent", parent_priority("generated-parent"), (min_ring_count(1), exact_ring_system), generated_exact.candidates),
        RuleSpec("generated-fused-parent", parent_priority("generated-parent"), (min_ring_system_ring_count(2),), generated_fused.candidates),
        RuleSpec("generated-hydro-fused-parent", parent_priority("generated-parent"), (min_ring_system_ring_count(2),), generated_hydro_fused.candidates),
        RuleSpec("generated-partial-hydro-fused-parent", parent_priority("generated-partial-hydro-parent"), (min_ring_system_ring_count(2),), generated_partial_hydro.candidates),
        RuleSpec("natural-product-topology-parent", parent_priority("whole-molecule-retained-parent"), (min_ring_system_ring_count(4),), natural_product_topology.candidates),
        RuleSpec("skeletal-replacement-fused-parent", parent_priority("skeletal-replacement-parent"), (min_ring_system_ring_count(3),), skeletal_replacement.candidates),
        RuleSpec(
            "retained-topology-replacement-fused-parent",
            parent_priority("skeletal-replacement-parent"),
            (min_ring_system_ring_count(2),),
            retained_topology_replacement.candidates,
        ),
        RuleSpec(
            "kekulized-generic-polycyclic-parent",
            parent_priority("polycyclic-functional-skeleton"),
            (kekulized_polycycle_signal, generic_polycyclic_topology, no_exocyclic_charged_fused_parent),
            kekulized_polycyclic.candidates,
        ),
        RuleSpec(
            "functional-acyclic-skeleton-with-ring-substituents",
            parent_priority("polycyclic-functional-skeleton"),
            (functional_skeleton_signal, acyclic_preference_signal, no_charged_ring_atom),
            functional_acyclic.candidates,
        ),
        RuleSpec(
            "acyclic-skeleton-with-ring-substituents",
            parent_priority("acyclic-with-ring-substituents"),
            (min_ring_count(1), acyclic_preference_signal),
            acyclic_with_ring_substituents.candidates,
        ),
    )

from .parent.rule_helpers import (
    _BICYCLIC_FUSION_BASES,
    _generated_exact_parent_sort_key,
    _single_ring_candidate_sort_key,
    _fusion_base_name,
    _partial_hydro_fusion_preference,
    _partial_hydro_spiro_alias_penalty,
    _curated_exact_parent_templates,
    _guanidine_parent_templates,
    _iter_fused_template_matches,
    _match_has_extra_core_bonds,
    _match_has_unlocanted_external_substitution,
    _name_acyclic_retained_core_match,
    _subtree_has_cumulated_diene,
    _skeletal_replacement_query,
    _skeletal_replacement_parent_name,
    _retained_topology_replacement_parent_name,
    _skeletal_replacement_sort_key,
    _spiro_norbornene_locant_penalty,
    _skeletal_replacement_saturation_penalty,
    _has_nonaromatic_core_imine,
    _locant_sort_key,
    _SKELETAL_REPLACEMENT_PREFIXES,
    _has_stereo_evidence,
    _locanted_template_stereo_prefix,
    _template_isotope_prefixed_parent_name,
    _topology_query,
    _template_has_one_ring_system,
    _template_is_fully_aromatic,
    _hydro_parent_name,
    _partial_hydro_parent_names,
    _hydro_locant_span,
    _partial_hydro_atom_score,
    _has_unsaturated_template_bond,
    _has_unsaturated_core_bond,
    _hydrogenated_template_penalty,
    _template_source_rank,
    _fused_candidate_preference,
    _substituent_locant_tuple,
    _benzimidazole_orientation_preference,
    _benzimidazole_template_h_preference,
    _benzotriazole_template_h_preference,
    _has_n_oxide_on_match,
    _name_fused_core_match,
    _name_natural_product_topology_match,
    _opsin_parseable_retained_parent_name,
    _fused_oxo_suffix_name,
    _adjust_fused_parent_for_charge,
    _drop_indicated_hydrogen_prefix,
    _drop_indicated_hydrogen_segments,
    _adjust_fused_parent_for_indicated_hydrogen,
    _named_ring_instance_names,
    _single_ring_stereo_prefix,
    _polycyclic_stereo_prefix,
    _single_ring_oxo_suffix_name,
    _oxo_adjusted_unsaturated_carbocycle_parent,
    _cycloalkapolyene_ring_size,
    _max_nonoxo_ring_double_bond_locs,
    _sulfanium_name,
    _phosphanium_name,
    _azanium_name,
    _pnictogen_hydride_name,
    _group14_hydride_name,
    _group14_halohydride_name,
    _group14_hydroxyhydride_name,
    _needs_substituent_parentheses,
    _n_halo_atom,
    _n_halo_prefix,
    _append_terminal_halo_prefixes,
    _prefer_acyclic_skeleton_name,
    _functional_term_score,
    _terminal_retained_parent_name,
)
