from __future__ import annotations

import re
from bisect import insort
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from itertools import combinations
from functools import lru_cache

from .acyclic.rules import (
    _bond_key,
    _path_substituent_prefixes,
    _standalone_saturated_chain_parent,
)
from .branch_rules import branch_prefix, parent_chain_name
from .branch.functional_groups import _is_nitrile_carbon
from .branch.hetero_atoms import _imine_carbon_ylidene
from .candidate_facts import (
    CandidateSafety,
    ChargeEvidence,
    HydrogenEvidence,
    OPSIN_FRAGILE_CHARGED_REPLACEMENT_UNSATURATION,
    OPSIN_FRAGILE_EXOCYCLIC_CHARGED_RING_SUFFIX,
    OPSIN_FRAGILE_GROUP14_HETERO_CARBON_LIGAND,
    OPSIN_FRAGILE_POLYCYCLIC_BRIDGE_TOPOLOGY,
    OPSIN_FRAGILE_POLYCYCLIC_BRIDGE_UNSATURATION,
    StereoEvidence,
    SuffixCandidate,
)
from .constants import HALO, MULT, ROOTS
from .cycle_utils import ordered_ring, ring_system_extends_core
from .formatting import format_prefixes as _format_prefixes
from .formatting import join_prefix_parent as _join_prefix_parent
from .formatting import loc_unsat
from .formatting import preferred_parent_stem as _preferred_parent_stem
from .formatting import unsaturated_parent as _unsaturated_parent
from .molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set
from .parent.partial_hydro import (
    PartialHydroCandidateFacts,
    partial_hydro_render_set,
    retained_partial_hydro_template_saturation_key,
)
from .polycyclic_rules import polycyclic_substituent_prefix
from .fused_template_families import (
    candidate_generated_fused_templates as _candidate_generated_fused_templates,
    curated_fused_templates as _curated_fused_templates,
    generated_fusion_orientation_role as _generated_fusion_orientation_role,
    generated_fusion_preference as _generated_fusion_preference,
    hydrocarbon_replacement_parent_templates_for_graph as _hydrocarbon_replacement_parent_templates_for_graph,
    retained_topology_replacement_parent_templates_for_graph as _retained_topology_replacement_parent_templates_for_graph,
)
from .scaffold_catalog import parent_scaffold_catalog
from .ring_resource_rules import resource_ring_parent_candidates as _resource_ring_parent_candidates
from .ring_rules import (
    _fixed_saturated_heterocycle_candidates,
    adjust_ring_parent_for_charge as _adjust_ring_parent_for_charge,
    ring_parent_candidates as _ring_parent_candidates,
)
from .branch.ring_substituents import _ring_substituent_prefix
from .spiro_component_rules import component_spiro_parent_candidates
from .rule_dispatch import (
    RuleSpec,
    acyclic,
    acyclic_preference_signal,
    amide_acyl_signal,
    boron_signal,
    chain_dominant_acyclic_parent_signal,
    charged_ring_atom,
    component_spiro_topology,
    conjugated_carbonyl_diene_signal,
    dithiocarbamate_retained_signal,
    dithiocarbonate_signal,
    exact_functional_retained_signal,
    exact_ring_system,
    formal_charge_signal,
    functional_skeleton_signal,
    generic_bicyclic_topology,
    generic_polycyclic_topology,
    imino_skeleton_signal,
    iodine_signal,
    kekulized_polycycle_signal,
    min_nonring_heavy_atom_count,
    min_ring_count,
    min_ring_system_ring_count,
    no_charged_ring_atom,
    no_exocyclic_charged_fused_parent,
    nitrogen_oxygen_signal,
    neutral_phosphorus,
    nitrogen_signal,
    nonring_neutral_group14,
    nonring_neutral_nitrogen,
    nonring_neutral_pnictogen,
    nonring_neutral_phosphorus,
    nonring_positive_nitrogen,
    nonring_positive_phosphorus,
    nonring_positive_sulfur,
    nitrile_oxide_signal,
    oxygen_signal,
    oxygen_oxygen_single_bond_signal,
    oxirane_dione_signal,
    oxime_sulfonate_nitrile_signal,
    spiro_topology,
    small_inorganic_retained_signal,
    sulfinate_retained_parent_signal,
    sulfur_imino_signal,
    sulfur_nitrogen_oxygen_signal,
    sulfur_oxygen_signal,
    sulfur_oxo_retained_parent_signal,
    sulfur_signal,
    terminal_carboxylate_signal,
    terminal_n_oxide_signal,
)
from .rule_priority import parent_priority
from .engine_flags import (
    FAST_POLY_BRIDGE_ROUTE_PROP,
    FAST_POLY_RETAINED_DEFER_BYPASS_PROP,
    FAST_POLY_RETAINED_DEFER_CACHE_KEY,
    FAST_POLY_RETAINED_DEFER_FRAGILE_TEMPLATE_KEYS,
)
from .rule_engine import NameCandidate
from .simple_prefixes import simple_external_prefix
from .stereochemistry import assigned_atom_stereochemistry
from .sulfur_oxo_rules import SulfurOxoRetainedParentRule
from .template_access import template_record_locants, template_record_mol, template_record_name
from .template_index import ring_system_with_graph_signatures


_MAX_RETAINED_RING_SCAFFOLD_CANDIDATES = 16
_RETAINED_RING_SCAFFOLD_TEMPLATE_FACT_SET_SIZE = 1
_RETAINED_RING_SCAFFOLD_RENDER_SET_SIZE = 16
_RETAINED_FUSED_RENDER_SET_SIZE = 12
_SINGLE_RING_PARENT_RING_FACT_SET_SIZE = 3
_GENERATED_PARTIAL_HYDRO_RENDER_SET_SIZE = 4
_RETAINED_PARTIAL_HYDRO_RENDER_SET_SIZE = 12
_GENERATED_FUSED_RENDER_SET_SIZE = 8
_MAX_GENERATED_FUSED_PARENT_CANDIDATES = 32
_SKELETAL_REPLACEMENT_FACT_SCAN_SIZE = 32
_SKELETAL_REPLACEMENT_RENDER_SET_SIZE = 12
_RETAINED_TOPOLOGY_REPLACEMENT_FACT_SCAN_SIZE = 32
_RETAINED_TOPOLOGY_REPLACEMENT_RENDER_SET_SIZE = 2
_MAX_RETAINED_TOPOLOGY_REPLACEMENT_CANDIDATES = 16
_RETAINED_PARTIAL_HYDRO_FALLBACK_PROP = "_nispo_enable_retained_partial_hydro_fallback"
_PENTITOL_RETAINED_NAMES = {
    ("R", "s", "S"): "ribitol",
    ("S", "s", "R"): "ribitol",
    ("R", "r", "S"): "xylitol",
    ("S", "r", "R"): "xylitol",
}
_PENTARIC_RETAINED_NAMES = {
    ("S", "r", "R"): ("ribaric acid", "riboarate"),
    ("R", "r", "S"): ("ribaric acid", "riboarate"),
    ("S", "s", "R"): ("xylaric acid", "xylarate"),
    ("R", "s", "S"): ("xylaric acid", "xylarate"),
}


def _parenthesized_if_complex(name: str) -> str:
    if name.startswith("(") and name.endswith(")"):
        return name
    return f"({name})" if "-" in name or name.startswith("(") else name


def _simple_or_branch_prefix(mol, idx: int, parent_idx: int) -> str:
    return simple_external_prefix(mol, idx, parent_idx) or branch_prefix(
        mol,
        idx,
        parent_idx,
        assume_simple_external_prefix_miss=True,
    )


def _acyclic_carbohydrate_retained_name(mol, Chem) -> str:
    if any(atom.GetAtomicNum() not in {6, 8} for atom in mol.GetAtoms()):
        return ""
    path = _linear_carbon_chain_path(mol, Chem)
    if len(path) != 5:
        return ""
    stereo_key = _carbon_path_stereo_key(mol, Chem, path)
    if len(stereo_key) != 3:
        return ""
    if _is_pentitol_graph(mol, Chem, path):
        return _PENTITOL_RETAINED_NAMES.get(stereo_key, "")
    terminal_charges = (
        _terminal_carboxyl_group_charge(mol, Chem, path[0], path[1]),
        _terminal_carboxyl_group_charge(mol, Chem, path[-1], path[-2]),
    )
    if None in terminal_charges:
        return ""
    if not all(_carbon_has_single_hydroxy_substituent(mol, Chem, atom_idx, set(path)) for atom_idx in path[1:-1]):
        return ""
    names = _PENTARIC_RETAINED_NAMES.get(stereo_key)
    if names is None:
        return ""
    if terminal_charges == (0, 0):
        return names[0]
    if terminal_charges == (-1, -1):
        return names[1]
    return ""


def _acyclic_carbohydrate_retained_stereo_locants(mol, Chem) -> tuple[tuple[int, str], ...]:
    path = _linear_carbon_chain_path(mol, Chem)
    return tuple((int(atom_idx), str(position)) for position, atom_idx in enumerate(path, start=1))


def _linear_carbon_chain_path(mol, Chem) -> tuple[int, ...]:
    carbon_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
    if len(carbon_atoms) < 2:
        return ()
    carbon_set = set(carbon_atoms)
    neighbors: dict[int, list[int]] = {idx: [] for idx in carbon_atoms}
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetAtomicNum() != 6 or end.GetAtomicNum() != 6:
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return ()
        begin_idx = begin.GetIdx()
        end_idx = end.GetIdx()
        neighbors[begin_idx].append(end_idx)
        neighbors[end_idx].append(begin_idx)
    if any(len(linked) > 2 for linked in neighbors.values()):
        return ()
    endpoints = sorted(idx for idx, linked in neighbors.items() if len(linked) == 1)
    if len(endpoints) != 2:
        return ()
    path = [endpoints[0]]
    previous = None
    current = endpoints[0]
    while current != endpoints[1]:
        next_atoms = [idx for idx in neighbors[current] if idx != previous]
        if len(next_atoms) != 1:
            return ()
        previous, current = current, next_atoms[0]
        path.append(current)
        if len(path) > len(carbon_atoms):
            return ()
    return tuple(path) if set(path) == carbon_set else ()


def _carbon_path_stereo_key(mol, Chem, path: Sequence[int]) -> tuple[str, ...]:
    centers = _cached_all_atom_stereochemistry(mol, Chem)
    return tuple(centers.get(atom_idx, "") for atom_idx in path[1:-1])


def _cached_all_atom_stereochemistry(mol, Chem) -> dict[int, str]:
    cached = mol_cache_get(mol, "all_atom_stereochemistry")
    if cached is not None:
        return cached
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    result = {
        int(atom_idx): str(descriptor)
        for atom_idx, descriptor in Chem.FindMolChiralCenters(
            mol,
            includeUnassigned=False,
            useLegacyImplementation=False,
        )
        if descriptor in {"R", "S", "r", "s"}
    }
    return mol_cache_set(mol, "all_atom_stereochemistry", result)


def _is_pentitol_graph(mol, Chem, path: Sequence[int]) -> bool:
    parent_atoms = set(path)
    for atom_idx in path:
        if not _carbon_has_single_hydroxy_substituent(mol, Chem, atom_idx, parent_atoms):
            return False
    return True


def _carbon_has_single_hydroxy_substituent(mol, Chem, atom_idx: int, parent_atoms: set[int]) -> bool:
    oxygen_count = 0
    for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        if neighbor_idx in parent_atoms:
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
        if (
            neighbor.GetAtomicNum() != 8
            or neighbor.GetFormalCharge() != 0
            or bond is None
            or bond.GetBondType() != Chem.BondType.SINGLE
            or any(
                other.GetIdx() != atom_idx and other.GetAtomicNum() > 1
                for other in neighbor.GetNeighbors()
            )
        ):
            return False
        oxygen_count += 1
    return oxygen_count == 1


def _terminal_carboxyl_group_charge(mol, Chem, carbon_idx: int, chain_neighbor_idx: int) -> int | None:
    atom = mol.GetAtomWithIdx(carbon_idx)
    double_oxygen_count = 0
    single_oxygen_charges = []
    for neighbor in atom.GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        if neighbor_idx == chain_neighbor_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, neighbor_idx)
        if neighbor.GetAtomicNum() != 8 or bond is None:
            return None
        if any(
            other.GetIdx() != carbon_idx and other.GetAtomicNum() > 1
            for other in neighbor.GetNeighbors()
        ):
            return None
        if bond.GetBondType() == Chem.BondType.DOUBLE and neighbor.GetFormalCharge() == 0:
            double_oxygen_count += 1
        elif bond.GetBondType() == Chem.BondType.SINGLE and neighbor.GetFormalCharge() in {0, -1}:
            single_oxygen_charges.append(neighbor.GetFormalCharge())
        else:
            return None
    if double_oxygen_count != 1 or len(single_oxygen_charges) != 1:
        return None
    return single_oxygen_charges[0]


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


class AcyclicCarbohydrateRetainedParentRule:
    family = "whole-molecule-retained-parent"
    priority = parent_priority("acyclic-parent")
    acyclic_only = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() != 0:
            return
        name = _acyclic_carbohydrate_retained_name(mol, self._Chem)
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=_heavy_atom_indices(mol),
                stereo_locants=_acyclic_carbohydrate_retained_stereo_locants(mol, self._Chem),
                source_family="acyclic-carbohydrate-retained",
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
        name = _sulfonylamide_name(mol, self._Chem)
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
            return
        name = _small_element_oxide_name(mol, self._Chem)
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )
            return
        name = _small_sulfur_imide_name(mol, self._Chem)
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )
            return
        name = _small_lambda_sulfur_parent_name(mol, self._Chem)
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )


def _small_element_oxide_name(mol, Chem) -> str:
    if mol.GetNumAtoms() != 2:
        return ""
    atoms = list(mol.GetAtoms())
    oxygen_atoms = [atom for atom in atoms if atom.GetSymbol() == "O" and atom.GetFormalCharge() == 0]
    if len(oxygen_atoms) != 1:
        return ""
    oxygen = oxygen_atoms[0]
    other = atoms[0] if atoms[1].GetIdx() == oxygen.GetIdx() else atoms[1]
    bond = mol.GetBondBetweenAtoms(oxygen.GetIdx(), other.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE or other.GetFormalCharge() != 0:
        return ""
    if other.GetSymbol() == "N" and other.GetTotalNumHs() == 0:
        return "oxonitrogen"
    if other.GetSymbol() == "P" and other.GetTotalNumHs() == 3:
        return "phosphane oxide"
    if other.GetSymbol() == "Si" and other.GetTotalNumHs() == 0:
        return "oxosilicon"
    return ""


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


def _sulfonylamide_name(mol, Chem) -> str:
    if mol.GetNumAtoms() != 4:
        return ""
    nitrogen_atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "N" and atom.GetFormalCharge() == -1 and not atom.IsInRing()
    ]
    sulfur_atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "S" and atom.GetFormalCharge() == 0 and not atom.IsInRing()
    ]
    if len(nitrogen_atoms) != 1 or len(sulfur_atoms) != 1:
        return ""
    nitrogen = nitrogen_atoms[0]
    sulfur = sulfur_atoms[0]
    bond = mol.GetBondBetweenAtoms(nitrogen.GetIdx(), sulfur.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return ""
    double_oxygens = 0
    for nbr in sulfur.GetNeighbors():
        if nbr.GetIdx() == nitrogen.GetIdx():
            continue
        nbr_bond = mol.GetBondBetweenAtoms(sulfur.GetIdx(), nbr.GetIdx())
        if (
            nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() == 0
            and nbr_bond is not None
            and nbr_bond.GetBondType() == Chem.BondType.DOUBLE
        ):
            double_oxygens += 1
        else:
            return ""
    return "sulfonylamide" if double_oxygens == 2 else ""


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
    substituent_neighbors = []
    for nbr in nitrogen.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(nidx, nbr.GetIdx())
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if nbr.GetSymbol() == "S":
            sulfur_neighbors.append(nbr.GetIdx())
        elif nbr.GetAtomicNum() > 1:
            substituent_neighbors.append(nbr.GetIdx())
        else:
            return ""
    if len(sulfur_neighbors) not in {1, 2} or len(substituent_neighbors) > 1:
        return ""
    units = []
    covered = {nidx}
    for sulfur_idx in sulfur_neighbors:
        unit, unit_atoms = _sulfonyl_azanide_unit(mol, Chem, sulfur_idx, nidx)
        if not unit:
            return ""
        units.append(unit)
        covered.update(unit_atoms)
    substituent_name = ""
    if substituent_neighbors:
        if len(sulfur_neighbors) != 1 or nitrogen.GetTotalNumHs() != 0:
            return ""
        substituent_idx = substituent_neighbors[0]
        substituent_name = _simple_or_branch_prefix(mol, substituent_idx, nidx)
        if not substituent_name:
            return ""
        covered.update(_collect_component_atoms(mol, substituent_idx, nidx))
    elif len(sulfur_neighbors) + nitrogen.GetTotalNumHs() != 2:
        return ""
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    if substituent_name:
        return units[0] + f"({_strip_outer_parentheses(substituent_name)})azanide"
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
    substituent_atoms = _collect_component_atoms(mol, substituent_idx, sulfur_idx)
    if substituent.GetSymbol() == "N" and len(substituent_atoms) > 8:
        return "", set()
    prefix = _simple_or_branch_prefix(mol, substituent_idx, sulfur_idx)
    if not prefix:
        return "", set()
    covered.update(substituent_atoms)
    return _strip_outer_parentheses(prefix) + "sulfonyl", covered


def _small_sulfur_imide_name(mol, Chem) -> str:
    sulfur_atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "S" and atom.GetFormalCharge() == 0 and not atom.IsInRing()
    ]
    if len(sulfur_atoms) != 1:
        return ""
    sulfur = sulfur_atoms[0]
    sidx = sulfur.GetIdx()
    imino_nitrogen = None
    double_oxygen = 0
    terminal_oxido = 0
    sulfur_halos = []
    owner_names = []
    covered = {sidx}
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(sidx, nidx)
        if bond is None:
            return ""
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE and imino_nitrogen is None:
            imino_nitrogen = nbr
            covered.add(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_oxygen += 1
            covered.add(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            terminal_oxido += 1
            covered.add(nidx)
        elif nbr.GetSymbol() in HALO and bond.GetBondType() == Chem.BondType.SINGLE:
            sulfur_halos.append(nbr.GetSymbol())
            covered.add(nidx)
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            owner = _simple_or_branch_prefix(mol, nidx, sidx)
            if not owner or _unstable_retained_functional_parent_substituent(owner):
                return ""
            owner_names.append(owner)
            covered.update(_collect_component_atoms(mol, nidx, sidx))
        else:
            return ""
    if imino_nitrogen is None:
        return ""
    imino_halos, imino_cyano_atoms = _small_sulfur_imide_n_substituents(mol, Chem, imino_nitrogen.GetIdx(), sidx)
    if imino_halos is None:
        return ""
    covered.update(imino_halos)
    covered.update(imino_cyano_atoms)
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    return _small_sulfur_imide_record_name(
        sulfur_hydrogens=sulfur.GetTotalNumHs(),
        imino_charge=imino_nitrogen.GetFormalCharge(),
        imino_hydrogens=imino_nitrogen.GetTotalNumHs(),
        double_oxygen=double_oxygen,
        terminal_oxido=terminal_oxido,
        sulfur_halos=tuple(sorted(sulfur_halos)),
        owner_names=tuple(owner_names),
        imino_halo_count=len(imino_halos),
        imino_cyano=bool(imino_cyano_atoms),
    )


def _small_lambda_sulfur_parent_name(mol, Chem) -> str:
    sulfur_atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "S" and atom.GetFormalCharge() == 0 and not atom.IsInRing()
    ]
    if len(sulfur_atoms) != 1:
        return ""
    sulfur = sulfur_atoms[0]
    sidx = sulfur.GetIdx()
    double_oxygen = 0
    double_nitrogen = 0
    owners = []
    covered = {sidx}
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(sidx, nidx)
        if bond is None:
            return ""
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_oxygen += 1
            covered.add(nidx)
        elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_nitrogen += 1
            if any(nn.GetIdx() != sidx and nn.GetAtomicNum() > 1 for nn in nbr.GetNeighbors()):
                return ""
            covered.add(nidx)
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            owner = _simple_or_branch_prefix(mol, nidx, sidx)
            if not _lambda_sulfur_owner_supported(owner):
                return ""
            owners.append(owner)
            covered.update(_collect_component_atoms(mol, nidx, sidx))
        else:
            return ""
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    if len(owners) not in {1, 2}:
        return ""
    if double_oxygen + double_nitrogen != 1:
        return ""
    valence = int(sulfur.GetTotalValence())
    if valence not in {4, 6}:
        return ""
    owner_prefix = _lambda_sulfur_owner_prefix(owners)
    if not owner_prefix:
        return ""
    if double_oxygen == 1:
        return f"{owner_prefix}-lambda{valence}-sulfanone"
    return f"{owner_prefix}-lambda{valence}-sulfanimine"


def _lambda_sulfur_owner_supported(name: str) -> bool:
    if not name:
        return False
    if name in {
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
    }:
        return True
    if name == "formimino":
        return True
    if not re.fullmatch(r"[A-Za-z0-9,()-]+", name):
        return False
    if len(name) > 80 or any(fragment in name for fragment in ("lambda", "spiro", "bicyclo")):
        return False
    aryl_substituent = (
        r"(?:methyl|ethyl|propyl|isopropyl|tert-butyl|t-butyl|methoxy|ethoxy|fluoro|chloro|bromo|iodo|"
        r"hydroxy|amino|dimethylamino|trifluoromethyl|cyano|"
        r"dimethyl|diethyl|difluoro|dichloro|dibromo|diiodo|dihydroxy|diamino|dimethoxy|diethoxy)"
    )
    if name.endswith("benzyl") and re.fullmatch(
        rf"(?:\d+(?:,\d+)*-{aryl_substituent})*benzyl",
        name,
    ):
        return True
    if name.endswith("phenyl") and re.fullmatch(
        rf"(?:\d+(?:,\d+)*-{aryl_substituent})*phenyl",
        name,
    ):
        return True
    return bool(
        re.fullmatch(
            r"(?:\d+-)?(?:furyl|thienyl|pyrrolyl|imidazolyl|pyrazolyl)|"
            r"(?:pyridin|pyrimidin|pyrazin|pyridazin|quinolin|isoquinolin|quinazolin|quinoxalin|benzimidazol|benzoxazol|benzothiazol|indol)-\d+-yl",
            name,
        )
    )


def _lambda_sulfur_owner_prefix(owners: Sequence[str]) -> str:
    owners = tuple(sorted("tert-butyl" if owner == "t-butyl" else owner for owner in owners))
    if len(owners) == 1:
        return owners[0]
    if owners[0] == owners[1]:
        return "di" + owners[0]
    return "".join(f"({owner})" for owner in owners)


def _small_sulfur_imide_n_substituents(mol, Chem, nitrogen_idx: int, sulfur_idx: int) -> tuple[set[int] | None, set[int]]:
    halos = set()
    cyano_atoms = set()
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == sulfur_idx or nbr.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if nbr.GetSymbol() in HALO and bond.GetBondType() == Chem.BondType.SINGLE:
            halos.add(nidx)
            continue
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            terminal_nitrile = [
                nnbr.GetIdx()
                for nnbr in nbr.GetNeighbors()
                if nnbr.GetIdx() != nitrogen_idx
                and nnbr.GetSymbol() == "N"
                and mol.GetBondBetweenAtoms(nidx, nnbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
            ]
            if len(terminal_nitrile) == 1:
                cyano_atoms.update({nidx, terminal_nitrile[0]})
                continue
        return None, set()
    return halos, cyano_atoms


def _small_sulfur_imide_record_name(
    *,
    sulfur_hydrogens: int,
    imino_charge: int,
    imino_hydrogens: int,
    double_oxygen: int,
    terminal_oxido: int,
    sulfur_halos: tuple[str, ...],
    owner_names: tuple[str, ...],
    imino_halo_count: int,
    imino_cyano: bool,
) -> str:
    if owner_names:
        if (
            len(owner_names) == 1
            and sulfur_hydrogens == 1
            and imino_charge == 0
            and double_oxygen == 1
            and terminal_oxido == 0
            and not sulfur_halos
            and imino_halo_count == 0
            and not imino_cyano
        ):
            return _small_sulfur_imide_owner_name(owner_names[0], "sulfoximine")
        return ""
    if imino_cyano:
        if double_oxygen == 2 and terminal_oxido == 0 and not sulfur_halos and imino_halo_count == 0:
            return "N-sulfonylcyanamide"
        return ""
    if imino_halo_count:
        if imino_halo_count == 1 and double_oxygen == 0 and terminal_oxido == 0 and sulfur_halos == ("F", "F"):
            return "trifluorosulfimide"
        return ""
    if imino_charge > 0 and imino_hydrogens == 2 and terminal_oxido == 0 and not sulfur_halos:
        if double_oxygen == 1:
            return "sulfinylammonium"
        if double_oxygen == 2:
            return "sulfonylammonium"
        return ""
    if terminal_oxido == 2 and double_oxygen == 1 and not sulfur_halos and sulfur_hydrogens == 0:
        return "sulfurimidate"
    if terminal_oxido != 0:
        return ""
    if not sulfur_halos:
        if double_oxygen == 2 and sulfur_hydrogens == 0:
            return "iminosulfone"
        if double_oxygen == 1 and sulfur_hydrogens == 0:
            return "iminosulfanone"
        if double_oxygen == 1 and sulfur_hydrogens == 2:
            return "imino-lambda6-sulfanone"
        return ""
    if double_oxygen == 0:
        if len(sulfur_halos) == 1:
            if sulfur_hydrogens == 1:
                return HALO[sulfur_halos[0]] + "sulfimide"
            if sulfur_hydrogens == 0 and sulfur_halos[0] == "F":
                return "iminosulfur fluoride"
        if len(sulfur_halos) == 2 and sulfur_hydrogens == 0:
            return _small_sulfur_imide_dihalo_name(sulfur_halos, "sulfimide")
        return ""
    if double_oxygen == 1:
        if len(sulfur_halos) == 1 and sulfur_hydrogens == 1:
            return HALO[sulfur_halos[0]] + "sulfoximine"
        if len(sulfur_halos) == 2 and sulfur_hydrogens == 0:
            if sulfur_halos == ("F", "F"):
                return "sulfonimidoyl fluoride"
            return _small_sulfur_imide_dihalo_name(sulfur_halos, "sulfoximine")
    return ""


def _small_sulfur_imide_dihalo_name(halos: tuple[str, ...], parent: str) -> str:
    counts = Counter(halos)
    if len(counts) == 1:
        symbol = next(iter(counts))
        return "di" + HALO[symbol] + parent
    parts = []
    for symbol in sorted(counts, key=lambda item: HALO[item]):
        halo = HALO[symbol]
        parts.append(halo if not parts else f"({halo})")
    return "".join(parts) + parent


def _small_sulfur_imide_owner_name(owner: str, parent: str) -> str:
    if _simple_unparenthesized_n_substituent(owner):
        return owner + parent
    return f"({_strip_outer_parentheses(owner)}){parent}"


class SydnoneRetainedParentRule:
    family = "sydnone-retained-parent"

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        return [
            NameCandidate(
                name,
                self.family,
                parent_priority("senior-functional-exception"),
                sort_key=(len(name), name),
                parent_atoms=frozenset(atoms),
            )
            for name, atoms in _sydnone_parent_names(ctx.mol)
        ]


def _sydnone_parent_names(mol) -> list[tuple[str, set[int]]]:
    names: set[tuple[str, frozenset[int]]] = set()
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 5:
            continue
        record = _sydnone_ring_record(mol, ring)
        if record is None:
            continue
        ring_set, carbonyl_oxygen, n_plus, substituted_carbon = record
        prefixes = []
        covered = set(ring_set)
        covered.add(carbonyl_oxygen)
        ok = True
        for atom_idx, locant in ((n_plus, "3"), (substituted_carbon, "4")):
            atom = mol.GetAtomWithIdx(atom_idx)
            for nbr in atom.GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx in ring_set or nidx in covered or nbr.GetAtomicNum() <= 1:
                    continue
                prefix = branch_prefix(mol, nidx, atom_idx)
                if not prefix:
                    ok = False
                    break
                branch_atoms = _external_branch_atoms(mol, nidx, atom_idx)
                if covered & branch_atoms:
                    ok = False
                    break
                covered.update(branch_atoms)
                prefixes.append((locant, prefix))
            if not ok:
                break
        if not ok or len(covered) != mol.GetNumAtoms():
            continue
        prefix_text = _format_prefixes(prefixes)
        name = _join_prefix_parent(prefix_text, "sydnone")
        names.add((name, frozenset(covered)))
    return [(name, set(atoms)) for name, atoms in sorted(names, key=lambda item: (len(item[0]), item[0]))]


def _sydnone_ring_record(mol, ring: Sequence[int]) -> tuple[set[int], int, int, int] | None:
    ring_set = set(ring)
    oxygens = [idx for idx in ring if mol.GetAtomWithIdx(idx).GetSymbol() == "O"]
    n_minus = [
        idx
        for idx in ring
        if mol.GetAtomWithIdx(idx).GetSymbol() == "N" and mol.GetAtomWithIdx(idx).GetFormalCharge() < 0
    ]
    n_plus = [
        idx
        for idx in ring
        if mol.GetAtomWithIdx(idx).GetSymbol() == "N" and mol.GetAtomWithIdx(idx).GetFormalCharge() > 0
    ]
    carbons = [idx for idx in ring if mol.GetAtomWithIdx(idx).GetSymbol() == "C"]
    if len(oxygens) != 1 or len(n_minus) != 1 or len(n_plus) != 1 or len(carbons) != 2:
        return None
    oxygen = oxygens[0]
    negative_n = n_minus[0]
    positive_n = n_plus[0]
    if mol.GetBondBetweenAtoms(oxygen, negative_n) is None:
        return None
    if mol.GetBondBetweenAtoms(negative_n, positive_n) is None:
        return None
    carbonyl_records = []
    for cidx in carbons:
        double_oxygens = []
        for nbr in mol.GetAtomWithIdx(cidx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set or nbr.GetSymbol() != "O":
                continue
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
                double_oxygens.append(nidx)
        if len(double_oxygens) == 1:
            carbonyl_records.append((cidx, double_oxygens[0]))
    if len(carbonyl_records) != 1:
        return None
    carbonyl_c, carbonyl_o = carbonyl_records[0]
    substituted = next(cidx for cidx in carbons if cidx != carbonyl_c)
    if mol.GetBondBetweenAtoms(carbonyl_c, oxygen) is None:
        return None
    if mol.GetBondBetweenAtoms(substituted, positive_n) is None:
        return None
    return ring_set, carbonyl_o, positive_n, substituted


def _external_branch_atoms(mol, start_idx: int, blocked_idx: int) -> set[int]:
    seen = {blocked_idx}
    stack = [start_idx]
    atoms: set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        atoms.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in seen:
                stack.append(nidx)
    return atoms


class SulfaniumParentRule:
    family = "sulfanium-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_positive_sulfur = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_set = set(range(mol.GetNumAtoms()))
        sulfur_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "S" and atom.GetFormalCharge() > 0 and not atom.IsInRing()
        ]
        if len(sulfur_atoms) == 2:
            name = _bis_sulfonium_alkane_or_arene_name(mol, self._Chem, [atom.GetIdx() for atom in sulfur_atoms])
            if name:
                yield NameCandidate(
                    name,
                    self.family,
                    self.priority,
                    parent_atoms=atom_set,
                    sort_key=(len(name), name),
                )
            return
        if len(sulfur_atoms) != 1:
            return
        sulfur = sulfur_atoms[0]
        sidx = sulfur.GetIdx()
        neighbors = [nbr.GetIdx() for nbr in sulfur.GetNeighbors()]
        implicit_hydrogens = sulfur.GetTotalNumHs()
        if len(neighbors) + implicit_hydrogens != 3 or len(neighbors) not in {2, 3}:
            return
        if any(
            mol.GetAtomWithIdx(nidx).GetSymbol() not in {"C", "N", "O", "F", "Cl", "Br", "I", "S"}
            or mol.GetAtomWithIdx(nidx).GetFormalCharge() != 0
            for nidx in neighbors
        ):
            return
        if any(
            mol.GetBondBetweenAtoms(sidx, nidx).GetBondType() != self._Chem.BondType.SINGLE
            for nidx in neighbors
        ):
            return
        names = []
        covered = {sidx}
        for nidx in neighbors:
            if (
                mol.GetAtomWithIdx(nidx).GetSymbol() in {"N", "O"}
                and _sulfanium_ligand_component_has_nitrile(mol, nidx, sidx)
            ):
                return
            prefix = _simple_or_branch_prefix(mol, nidx, sidx)
            if not prefix:
                return
            names.append(_retained_parent_substituent_alias(prefix))
            covered.update(_collect_component_atoms(mol, nidx, sidx))
        if covered != atom_set:
            return
        name = _sulfanium_name(names)
        if name:
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


def _sulfanium_ligand_component_has_nitrile(mol, ligand_idx: int, sulfur_idx: int) -> bool:
    component = _collect_component_atoms(mol, ligand_idx, sulfur_idx)
    return any(_is_nitrile_carbon(mol, atom_idx) for atom_idx in component)


def _bis_sulfonium_alkane_or_arene_name(mol, Chem, sulfur_idxs: Sequence[int]) -> str:
    if len(sulfur_idxs) != 2 or sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != 2:
        return ""
    first_s, second_s = sulfur_idxs
    path = tuple(Chem.rdmolops.GetShortestPath(mol, first_s, second_s))
    if not path or path[0] != first_s or path[-1] != second_s:
        return ""
    linker = path[1:-1]
    if len(linker) < 2:
        return ""
    if _is_unsubstituted_saturated_carbon_path(mol, Chem, linker, first_s, second_s):
        parent = _alkane_parent_name(len(linker))
        if not parent:
            return ""
        linker_text = f"1,{len(linker)}-{parent}diylbis"
        linker_atoms = tuple(linker)
        linker_attachments = (path[1], path[-2])
    else:
        arene_linker = _sulfonium_para_phenylene_linker(mol, first_s, second_s)
        if arene_linker is None:
            return ""
        linker_text, linker_atoms, linker_attachments = arene_linker
    ligand_texts = []
    covered = set(linker_atoms)
    for sidx, linker_idx in zip((first_s, second_s), linker_attachments, strict=True):
        ligand_text, ligand_atoms = _sulfonio_external_ligand_text_and_atoms(mol, Chem, sidx, linker_idx)
        if not ligand_text:
            return ""
        ligand_texts.append(ligand_text)
        covered.add(sidx)
        covered.update(ligand_atoms)
    if covered != set(range(mol.GetNumAtoms())) or ligand_texts[0] != ligand_texts[1]:
        return ""
    return f"{linker_text}({ligand_texts[0]}sulfonium)"


def _sulfonium_para_phenylene_linker(mol, first_s: int, second_s: int) -> tuple[str, tuple[int, ...], tuple[int, int]] | None:
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        linker = tuple(ring)
        linker_set = frozenset(linker)
        if any(
            mol.GetAtomWithIdx(idx).GetSymbol() != "C" or not mol.GetAtomWithIdx(idx).GetIsAromatic()
            for idx in linker
        ):
            continue
        attached: list[int] = []
        for sidx in (first_s, second_s):
            sulfur_attached = [
                nbr.GetIdx()
                for nbr in mol.GetAtomWithIdx(sidx).GetNeighbors()
                if nbr.GetIdx() in linker_set
            ]
            if len(sulfur_attached) != 1:
                break
            attached.append(sulfur_attached[0])
        if len(attached) != 2:
            continue
        if _ring_distance(mol, linker_set, attached[0], attached[1]) != 3:
            continue
        if any(
            nbr.GetIdx() not in linker_set and nbr.GetIdx() not in {first_s, second_s}
            for idx in linker
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
            if nbr.GetAtomicNum() > 1
        ):
            continue
        return "1,4-phenylenebis", linker, (attached[0], attached[1])
    return None


def _sulfonium_arene_linker_text(mol, Chem, linker: Sequence[int], first_s: int, second_s: int) -> str:
    if len(linker) != 6:
        return ""
    linker_set = set(linker)
    if any(mol.GetAtomWithIdx(idx).GetSymbol() != "C" or not mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in linker):
        return ""
    if any(
        nbr.GetIdx() not in linker_set and nbr.GetIdx() not in {first_s, second_s}
        for idx in linker
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
        if nbr.GetAtomicNum() > 1
    ):
        return ""
    ring_info = mol.GetRingInfo()
    if not any(linker_set == set(ring) for ring in ring_info.AtomRings()):
        return ""
    s_positions = []
    for sidx in (first_s, second_s):
        attached = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(sidx).GetNeighbors() if nbr.GetIdx() in linker_set]
        if len(attached) != 1:
            return ""
        s_positions.append(linker.index(attached[0]))
    if (s_positions[0] - s_positions[1]) % 6 not in {2, 4}:
        return ""
    return "1,4-phenylenebis"


def _sulfonio_external_ligand_text_and_atoms(mol, Chem, sulfur_idx: int, linker_idx: int) -> tuple[str, set[int]]:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() <= 0 or sulfur.IsInRing():
        return "", set()
    if sulfur.GetTotalNumHs() != 0:
        return "", set()
    names = []
    covered: set[int] = set()
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return "", set()
        if nidx == linker_idx:
            continue
        if nbr.GetSymbol() != "C" or nbr.GetFormalCharge() != 0:
            return "", set()
        prefix = _retained_parent_substituent_alias(_simple_or_branch_prefix(mol, nidx, sulfur_idx))
        if not _stable_bis_sulfonio_ligand(prefix):
            return "", set()
        names.append(_strip_outer_parentheses(prefix))
        covered.update(_collect_component_atoms(mol, nidx, sulfur_idx))
    if len(names) != 2:
        return "", set()
    text = _sulfonio_repeated_ligand_text(names)
    return (text, covered) if text else ("", set())


def _stable_bis_sulfonio_ligand(name: str) -> bool:
    if not name or len(name) > 64:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "carbam", "urea", "azanid")):
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    return name.endswith("yl")


def _sulfonio_repeated_ligand_text(names: Sequence[str]) -> str:
    counts = Counter(names)
    if len(counts) != 1:
        return ""
    name, count = next(iter(counts.items()))
    if count == 1:
        return name
    multiplier = MULT.get(count, "")
    if not multiplier:
        return ""
    bare = _strip_outer_parentheses(name)
    return multiplier + bare if re.fullmatch(r"[A-Za-z0-9']+", bare) else f"{multiplier}({bare})"


class BisSulfoniumSulfideRetainedParentRule:
    family = "bis-sulfonium-sulfide-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_set = set(range(mol.GetNumAtoms()))
        for sulfur in mol.GetAtoms():
            sidx = sulfur.GetIdx()
            if (
                sulfur.GetSymbol() != "S"
                or sulfur.GetFormalCharge() != 0
                or sulfur.IsInRing()
                or sulfur.GetTotalNumHs() != 0
            ):
                continue
            neighbors = [nbr.GetIdx() for nbr in sulfur.GetNeighbors() if nbr.GetAtomicNum() > 1]
            if len(neighbors) != 2:
                continue
            if any(
                mol.GetBondBetweenAtoms(sidx, nidx).GetBondType() != self._Chem.BondType.SINGLE
                for nidx in neighbors
            ):
                continue
            covered = {sidx}
            side_count = 0
            for nidx in neighbors:
                side_atoms = _diphenylsulfonio_phenyl_sulfide_side_atoms(mol, self._Chem, nidx, sidx)
                if not side_atoms:
                    break
                covered.update(side_atoms)
                side_count += 1
            if side_count != 2 or covered != atom_set:
                continue
            name = "bis[4-(diphenylsulfonio)phenyl] sulfide"
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(mol.GetNumAtoms())),
                sort_key=(len(name), name),
            )
            return


class MethylenebisDithiocarbamateRetainedParentRule:
    family = "methylenebis-dithiocarbamate-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        name = _bis_dithiocarbamate_name(mol, self._Chem)
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(mol.GetNumAtoms())),
                sort_key=(len(name), name),
            )


class DithiocarbamateRetainedParentRule:
    family = "dithiocarbamate-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_set = set(range(mol.GetNumAtoms()))
        for nitrogen in mol.GetAtoms():
            nidx = nitrogen.GetIdx()
            if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing():
                continue
            carbonyl_neighbors = []
            for nbr in nitrogen.GetNeighbors():
                nbrid = nbr.GetIdx()
                bond = mol.GetBondBetweenAtoms(nidx, nbrid)
                if nbr.GetSymbol() == "C" and bond is not None and bond.GetBondType() == self._Chem.BondType.SINGLE:
                    carbonyl_neighbors.append(nbrid)
            for carbon_idx in carbonyl_neighbors:
                record = _dithiocarbamate_carbon_record(mol, self._Chem, carbon_idx, nidx)
                if record is None:
                    continue
                name_and_atoms = _dithiocarbamate_name_and_atoms(mol, nidx, carbon_idx, record)
                if name_and_atoms is None:
                    continue
                name, covered = name_and_atoms
                if covered != atom_set:
                    continue
                yield NameCandidate(
                    name,
                    self.family,
                    self.priority,
                    parent_atoms=atom_set,
                    sort_key=(len(name), name),
                )


class DithiocarbonateRetainedParentRule:
    family = "dithiocarbonate-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_set = set(range(mol.GetNumAtoms()))
        for carbon in mol.GetAtoms():
            if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
                continue
            record = _dithiocarbonate_record(mol, self._Chem, carbon.GetIdx())
            if record is None or record["covered_atoms"] != atom_set:
                continue
            name = _dithiocarbonate_record_name(record)
            if not name:
                continue
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=atom_set,
                sort_key=(len(name), name),
            )


class HexamethylenediamideRetainedParentRule:
    family = "hexamethylenediamide-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        name = _hexamethylenediamide_name(mol, self._Chem)
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(mol.GetNumAtoms())),
                sort_key=(len(name), name),
            )


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
        if len(phosphorus_atoms) == 2:
            name = _bis_phosphonio_alkane_name(mol, self._Chem, [atom.GetIdx() for atom in phosphorus_atoms])
            if name:
                atom_set = set(range(mol.GetNumAtoms()))
                yield NameCandidate(
                    name,
                    self.family,
                    self.priority,
                    parent_atoms=atom_set,
                    sort_key=(len(name), name),
                )
            return
        if len(phosphorus_atoms) != 1:
            return
        phosphorus = phosphorus_atoms[0]
        pidx = phosphorus.GetIdx()
        neighbors = [nbr.GetIdx() for nbr in phosphorus.GetNeighbors()]
        implicit_hydrogens = phosphorus.GetTotalNumHs()
        if len(neighbors) + implicit_hydrogens != 4 or len(neighbors) not in {1, 2, 3, 4}:
            return
        if any(mol.GetAtomWithIdx(nidx).GetSymbol() == "P" for nidx in neighbors):
            return
        if any(
            mol.GetBondBetweenAtoms(pidx, nidx).GetBondType() != self._Chem.BondType.SINGLE
            for nidx in neighbors
        ):
            return
        special_name = _benzotriazolyloxy_amino_phosphonium_name(mol, self._Chem, pidx)
        if special_name:
            yield NameCandidate(
                special_name,
                self.family,
                self.priority,
                parent_atoms=set(range(mol.GetNumAtoms())),
                sort_key=(len(special_name), special_name),
            )
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


class BisPhosphinoAmineRetainedParentRule:
    family = "bis-phosphino-amine-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        for name in _bis_phosphino_amine_names(mol, self._Chem):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(mol.GetNumAtoms())),
                sort_key=(len(name), name),
            )


def _bis_phosphino_amine_names(mol, Chem) -> list[str]:
    if sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != 0:
        return []
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for nitrogen in mol.GetAtoms():
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing():
            continue
        if nitrogen.GetTotalNumHs() != 0:
            continue
        nidx = nitrogen.GetIdx()
        neighbors = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(neighbors) != 3:
            continue
        if any(mol.GetBondBetweenAtoms(nidx, nbr_idx).GetBondType() != Chem.BondType.SINGLE for nbr_idx in neighbors):
            continue
        phosphorus_neighbors = [
            nbr_idx
            for nbr_idx in neighbors
            if mol.GetAtomWithIdx(nbr_idx).GetSymbol() == "P"
            and mol.GetAtomWithIdx(nbr_idx).GetFormalCharge() == 0
            and not mol.GetAtomWithIdx(nbr_idx).IsInRing()
        ]
        carbon_neighbors = [
            nbr_idx
            for nbr_idx in neighbors
            if mol.GetAtomWithIdx(nbr_idx).GetSymbol() == "C"
            and mol.GetAtomWithIdx(nbr_idx).GetFormalCharge() == 0
        ]
        if len(phosphorus_neighbors) != 2 or len(carbon_neighbors) != 1:
            continue
        amine_substituent = _retained_parent_substituent_alias(
            _simple_or_branch_prefix(mol, carbon_neighbors[0], nidx)
        )
        if not _stable_bis_phosphino_amine_substituent(amine_substituent):
            continue
        covered = {nidx}
        covered.update(_collect_component_atoms(mol, carbon_neighbors[0], nidx))
        phosphino_names = []
        for pidx in phosphorus_neighbors:
            phosphino_name, phosphino_atoms = _phosphino_ligand_name_and_atoms(mol, Chem, pidx, nidx)
            if not phosphino_name:
                break
            phosphino_names.append(phosphino_name)
            covered.update(phosphino_atoms)
        else:
            if covered == atom_set and len(set(phosphino_names)) == 1:
                names.append(f"bis({phosphino_names[0]})({_strip_outer_parentheses(amine_substituent)})amine")
    return sorted(set(names), key=lambda value: (len(value), value))


def _phosphino_ligand_name_and_atoms(mol, Chem, phosphorus_idx: int, nitrogen_idx: int) -> tuple[str, set[int]]:
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)
    if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0 or phosphorus.IsInRing():
        return "", set()
    if phosphorus.GetTotalNumHs() != 0:
        return "", set()
    ligand_names = []
    covered = {phosphorus_idx}
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(phosphorus_idx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return "", set()
        if nidx == nitrogen_idx:
            continue
        if nbr.GetSymbol() != "C" or nbr.GetFormalCharge() != 0:
            return "", set()
        ligand = _retained_parent_substituent_alias(_simple_or_branch_prefix(mol, nidx, phosphorus_idx))
        if not _stable_bis_phosphino_amine_substituent(ligand):
            return "", set()
        ligand_names.append(_strip_outer_parentheses(ligand))
        covered.update(_collect_component_atoms(mol, nidx, phosphorus_idx))
    counts = Counter(ligand_names)
    if len(counts) != 1:
        return "", set()
    ligand, count = next(iter(counts.items()))
    if count != 2:
        return "", set()
    bare = _strip_outer_parentheses(ligand)
    ligand_text = "di" + bare if re.fullmatch(r"[A-Za-z0-9']+", bare) else f"di({bare})"
    return ligand_text + "phosphino", covered


def _stable_bis_phosphino_amine_substituent(name: str) -> bool:
    if not name or len(name) > 96:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "carbam", "urea", "azanid")):
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    return name.endswith("yl")


def _bis_phosphonio_alkane_name(mol, Chem, phosphorus_idxs: Sequence[int]) -> str:
    if len(phosphorus_idxs) != 2 or sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != 2:
        return ""
    first_p, second_p = phosphorus_idxs
    path = tuple(Chem.rdmolops.GetShortestPath(mol, first_p, second_p))
    if not path or path[0] != first_p or path[-1] != second_p:
        return ""
    linker = path[1:-1]
    if len(linker) < 2 or not _is_unsubstituted_saturated_carbon_path(mol, Chem, linker, first_p, second_p):
        return ""
    parent = _alkane_parent_name(len(linker))
    if not parent:
        return ""
    ligand_texts = []
    covered = set(linker)
    for pidx, linker_idx in ((first_p, path[1]), (second_p, path[-2])):
        ligand_text, ligand_atoms = _phosphonio_external_ligand_text_and_atoms(mol, Chem, pidx, linker_idx)
        if not ligand_text:
            return ""
        ligand_texts.append(ligand_text)
        covered.add(pidx)
        covered.update(ligand_atoms)
    if covered != set(range(mol.GetNumAtoms())) or ligand_texts[0] != ligand_texts[1]:
        return ""
    return f"1,{len(linker)}-bis({ligand_texts[0]}phosphonio){parent}"


def _phosphonio_external_ligand_text_and_atoms(mol, Chem, phosphorus_idx: int, linker_idx: int) -> tuple[str, set[int]]:
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)
    if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 1 or phosphorus.IsInRing():
        return "", set()
    if phosphorus.GetTotalNumHs() not in {0, 1}:
        return "", set()
    names = []
    covered: set[int] = set()
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(phosphorus_idx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return "", set()
        if nidx == linker_idx:
            continue
        if nbr.GetSymbol() != "C" or nbr.GetFormalCharge() != 0:
            return "", set()
        prefix = _retained_parent_substituent_alias(_simple_or_branch_prefix(mol, nidx, phosphorus_idx))
        if not _stable_bis_phosphonio_ligand(prefix):
            return "", set()
        names.append(_strip_outer_parentheses(prefix))
        covered.update(_collect_component_atoms(mol, nidx, phosphorus_idx))
    if len(names) + phosphorus.GetTotalNumHs() != 3:
        return "", set()
    text = _phosphonio_repeated_ligand_text(names)
    return (text, covered) if text else ("", set())


def _stable_bis_phosphonio_ligand(name: str) -> bool:
    if not name or len(name) > 64:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "carbam", "urea", "azanid")):
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    return name.endswith("yl")


def _phosphonio_repeated_ligand_text(names: Sequence[str]) -> str:
    counts = Counter(names)
    if len(counts) != 1:
        return ""
    name, count = next(iter(counts.items()))
    if count == 1:
        return name
    multiplier = MULT.get(count, "")
    if not multiplier:
        return ""
    bare = _strip_outer_parentheses(name)
    return multiplier + bare if re.fullmatch(r"[A-Za-z0-9']+", bare) else f"{multiplier}({bare})"


class PhosphanyliumOxideParentRule:
    family = "phosphanylium-oxide-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_positive_phosphorus = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        phosphorus_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "P" and atom.GetFormalCharge() > 0 and not atom.IsInRing()
        ]
        if len(phosphorus_atoms) != 1:
            return
        phosphorus = phosphorus_atoms[0]
        if phosphorus.GetTotalNumHs() != 0:
            return
        pidx = phosphorus.GetIdx()
        oxide_idx = None
        oxido_idx = None
        hydroxy_idx = None
        ligand_idxs = []
        for nbr in phosphorus.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(pidx, nidx)
            if bond is None:
                return
            if (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() == 0
                and bond.GetBondType() == self._Chem.BondType.DOUBLE
            ):
                if oxide_idx is not None:
                    return
                oxide_idx = nidx
            elif (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() < 0
                and bond.GetBondType() == self._Chem.BondType.SINGLE
                and not any(other.GetIdx() != pidx and other.GetAtomicNum() > 1 for other in nbr.GetNeighbors())
            ):
                if oxido_idx is not None:
                    return
                oxido_idx = nidx
            elif (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() == 0
                and bond.GetBondType() == self._Chem.BondType.SINGLE
                and not any(other.GetIdx() != pidx and other.GetAtomicNum() > 1 for other in nbr.GetNeighbors())
            ):
                if hydroxy_idx is not None:
                    return
                hydroxy_idx = nidx
            elif bond.GetBondType() == self._Chem.BondType.SINGLE and nbr.GetAtomicNum() > 1:
                ligand_idxs.append(nidx)
            elif nbr.GetAtomicNum() > 1:
                return
        if oxide_idx is None:
            return
        if oxido_idx is None and hydroxy_idx is None:
            name = _phosphanylium_oxo_parent_name(mol, pidx, oxide_idx, ligand_idxs, atom_set)
        else:
            if (oxido_idx is None) == (hydroxy_idx is None) or len(ligand_idxs) != 1:
                return
            ligand = _simple_or_branch_prefix(mol, ligand_idxs[0], pidx)
            if not ligand:
                return
            terminal_single_oxygen = oxido_idx if oxido_idx is not None else hydroxy_idx
            covered = {pidx, oxide_idx, terminal_single_oxygen}
            covered.update(_collect_component_atoms(mol, ligand_idxs[0], pidx))
            if covered != atom_set:
                return
            ligand = _strip_outer_parentheses(ligand)
            prefix = "oxido" if oxido_idx is not None else "hydroxy"
            name = f"{prefix}({ligand})phosphanylium oxide"
        if not name:
            return
        yield NameCandidate(
            name,
            self.family,
            self.priority,
            parent_atoms=atom_set,
            sort_key=(len(name), name),
        )


def _phosphanylium_oxo_parent_name(
    mol,
    phosphorus_idx: int,
    oxide_idx: int,
    ligand_idxs: Sequence[int],
    atom_set: set[int],
) -> str:
    if len(ligand_idxs) < 1 or len(ligand_idxs) > 3:
        return ""
    covered = {phosphorus_idx, oxide_idx}
    ligand_texts = []
    for ligand_idx in ligand_idxs:
        ligand = _phosphanylium_oxo_ligand_text(mol, ligand_idx, phosphorus_idx)
        if not ligand:
            return ""
        ligand_texts.append(ligand)
        covered.update(_collect_component_atoms(mol, ligand_idx, phosphorus_idx))
    if covered != atom_set:
        return ""
    ligand_text = "".join(f"({text})" for text in sorted(ligand_texts))
    return f"oxo{ligand_text}phosphanium"


def _phosphanylium_oxo_ligand_text(mol, ligand_idx: int, phosphorus_idx: int) -> str:
    ligand = mol.GetAtomWithIdx(ligand_idx)
    if ligand.GetFormalCharge() != 0:
        return ""
    text = _simple_or_branch_prefix(mol, ligand_idx, phosphorus_idx)
    if not text:
        return ""
    return text if _stable_phosphanylium_oxo_ligand(text) else ""


def _stable_phosphanylium_oxo_ligand(name: str) -> bool:
    if not name or len(name) > 96:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "carbam", "urea", "azanid")):
        return False
    return True


class PhosphanyliumIminoParentRule:
    family = "phosphanylium-imino-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_positive_phosphorus = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        phosphorus_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "P" and atom.GetFormalCharge() > 0 and not atom.IsInRing()
        ]
        if len(phosphorus_atoms) != 1:
            return
        phosphorus = phosphorus_atoms[0]
        if phosphorus.GetTotalNumHs() != 0:
            return
        pidx = phosphorus.GetIdx()
        oxido_idx = None
        imino_idx = None
        ligand_idx = None
        for nbr in phosphorus.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(pidx, nidx)
            if bond is None:
                return
            if (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() < 0
                and bond.GetBondType() == self._Chem.BondType.SINGLE
                and not any(other.GetIdx() != pidx and other.GetAtomicNum() > 1 for other in nbr.GetNeighbors())
            ):
                if oxido_idx is not None:
                    return
                oxido_idx = nidx
            elif nbr.GetSymbol() == "N" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == self._Chem.BondType.DOUBLE:
                if imino_idx is not None:
                    return
                imino_idx = nidx
            elif bond.GetBondType() == self._Chem.BondType.SINGLE and nbr.GetAtomicNum() > 1:
                if ligand_idx is not None:
                    return
                ligand_idx = nidx
            elif nbr.GetAtomicNum() > 1:
                return
        if oxido_idx is None or imino_idx is None or ligand_idx is None:
            return
        ligand = _simple_or_branch_prefix(mol, ligand_idx, pidx)
        if not ligand:
            return
        imino_substituents = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(imino_idx).GetNeighbors()
            if nbr.GetIdx() != pidx and nbr.GetAtomicNum() > 1
        ]
        if len(imino_substituents) != 1:
            return
        imino_owner_idx = imino_substituents[0]
        imino_owner = _simple_or_branch_prefix(mol, imino_owner_idx, imino_idx)
        if not imino_owner:
            return
        covered = {pidx, oxido_idx, imino_idx}
        covered.update(_collect_component_atoms(mol, ligand_idx, pidx))
        covered.update(_collect_component_atoms(mol, imino_owner_idx, imino_idx))
        if covered != atom_set:
            return
        ligand = _strip_outer_parentheses(ligand)
        imino_owner = _strip_outer_parentheses(imino_owner)
        imino_text = imino_owner + "imino" if _simple_ligand_name(imino_owner) else f"({imino_owner})imino"
        name = f"oxido({ligand})({imino_text})phosphanylium"
        yield NameCandidate(
            name,
            self.family,
            self.priority,
            parent_atoms=atom_set,
            sort_key=(len(name), name),
        )


class PhosphanyliumYlideneParentRule:
    family = "phosphanylium-ylidene-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_positive_phosphorus = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        phosphorus_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "P" and atom.GetFormalCharge() > 0 and not atom.IsInRing()
        ]
        if len(phosphorus_atoms) != 1:
            return
        phosphorus = phosphorus_atoms[0]
        if phosphorus.GetTotalNumHs() != 0:
            return
        pidx = phosphorus.GetIdx()
        oxido_idxs = []
        ylidene = []
        ligands = []
        for nbr in phosphorus.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(pidx, nidx)
            if bond is None:
                return
            if (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() < 0
                and bond.GetBondType() == self._Chem.BondType.SINGLE
                and _heavy_degree_excluding(mol, nidx, pidx) == 0
            ):
                oxido_idxs.append(nidx)
            elif (
                nbr.GetSymbol() == "C"
                and nbr.GetFormalCharge() == 0
                and bond.GetBondType() == self._Chem.BondType.DOUBLE
            ):
                if ylidene:
                    return
                ylidene_text = _phosphanylium_ylidene_text(mol, self._Chem, nidx, pidx)
                if not ylidene_text:
                    return
                ylidene.append((nidx, ylidene_text))
            elif bond.GetBondType() == self._Chem.BondType.SINGLE and nbr.GetAtomicNum() > 1 and nbr.GetFormalCharge() == 0:
                ligand = _phosphanylium_ylidene_ligand_text(mol, nidx, pidx)
                if not ligand:
                    return
                ligands.append((nidx, ligand))
            elif nbr.GetAtomicNum() > 1:
                return
        if len(ylidene) != 1:
            return
        if not ((len(oxido_idxs) == 1 and len(ligands) == 1) or (len(oxido_idxs) == 2 and not ligands)):
            return
        covered = {pidx, *oxido_idxs}
        covered.update(_collect_component_atoms(mol, ylidene[0][0], pidx))
        for ligand_idx, _ligand in ligands:
            covered.update(_collect_component_atoms(mol, ligand_idx, pidx))
        if covered != atom_set:
            return
        if ligands:
            name = f"oxido({ligands[0][1]})({ylidene[0][1]})phosphanylium"
        else:
            name = f"dioxido({ylidene[0][1]})phosphanylium"
        yield NameCandidate(
            name,
            self.family,
            self.priority,
            parent_atoms=atom_set,
            sort_key=(len(name), name),
        )


def _phosphanylium_ylidene_ligand_text(mol, ligand_idx: int, phosphorus_idx: int) -> str:
    ligand = mol.GetAtomWithIdx(ligand_idx)
    if ligand.GetFormalCharge() != 0:
        return ""
    if ligand.GetSymbol() in HALO and _heavy_degree_excluding(mol, ligand_idx, phosphorus_idx) == 0:
        return HALO[ligand.GetSymbol()]
    text = _strip_outer_parentheses(_simple_or_branch_prefix(mol, ligand_idx, phosphorus_idx))
    return text if _stable_phosphanylium_ylidene_ligand(text) else ""


def _phosphanylium_ylidene_text(mol, Chem, carbon_idx: int, phosphorus_idx: int) -> str:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return ""
    external = [
        nbr.GetIdx()
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != phosphorus_idx and nbr.GetAtomicNum() > 1
    ]
    if len(external) == 1:
        text = _phosphanylium_terminal_ylidene_text(mol, Chem, carbon_idx, external[0])
        if text:
            return text
    text = _strip_outer_parentheses(_simple_or_branch_prefix(mol, carbon_idx, phosphorus_idx))
    text = _phosphanylium_ylidene_from_substituent_text(text)
    return text if _stable_phosphanylium_ylidene_substituent(text) else ""


def _phosphanylium_ylidene_from_substituent_text(text: str) -> str:
    if text.endswith("ylidene"):
        return text
    if text.endswith("yl"):
        return text[:-2] + "ylidene"
    return text


def _phosphanylium_terminal_ylidene_text(mol, Chem, carbon_idx: int, external_idx: int) -> str:
    external = mol.GetAtomWithIdx(external_idx)
    bond = mol.GetBondBetweenAtoms(carbon_idx, external_idx)
    if bond is None or external.GetFormalCharge() != 0:
        return ""
    if (
        external.GetSymbol() == "S"
        and bond.GetBondType() == Chem.BondType.DOUBLE
        and _heavy_degree_excluding(mol, external_idx, carbon_idx) == 0
    ):
        return "thioxomethylidene"
    if (
        external.GetSymbol() == "O"
        and bond.GetBondType() == Chem.BondType.DOUBLE
        and _heavy_degree_excluding(mol, external_idx, carbon_idx) == 0
    ):
        return "oxomethylidene"
    if external.GetSymbol() != "C" or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    double_oxygen, single_oxygen = _phosphanylium_carbonyl_oxygens(mol, Chem, external_idx, carbon_idx)
    if len(double_oxygen) != 1 or len(single_oxygen) != 1:
        return ""
    owner = _phosphanylium_carbonyl_oxygen_owner(mol, single_oxygen[0], external_idx)
    if owner == "hydroxy":
        return "carboxymethylidene"
    if owner:
        return f"2-{owner}-2-oxoethylidene"
    return ""


def _phosphanylium_carbonyl_oxygens(mol, Chem, carbonyl_idx: int, blocked_idx: int) -> tuple[list[int], list[int]]:
    double_oxygen = []
    single_oxygen = []
    for nbr in mol.GetAtomWithIdx(carbonyl_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == blocked_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if bond is None or nbr.GetSymbol() != "O" or nbr.GetFormalCharge() != 0:
            continue
        if bond.GetBondType() == Chem.BondType.DOUBLE and _heavy_degree_excluding(mol, nidx, carbonyl_idx) == 0:
            double_oxygen.append(nidx)
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            single_oxygen.append(nidx)
    return double_oxygen, single_oxygen


def _phosphanylium_carbonyl_oxygen_owner(mol, oxygen_idx: int, carbonyl_idx: int) -> str:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    heavy = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != carbonyl_idx and nbr.GetAtomicNum() > 1
    ]
    if not heavy:
        return "hydroxy" if oxygen.GetTotalNumHs() == 1 else ""
    if len(heavy) != 1:
        return ""
    text = _strip_outer_parentheses(_simple_or_branch_prefix(mol, oxygen_idx, carbonyl_idx))
    return text if _stable_phosphanylium_ylidene_ligand(text) else ""


def _stable_phosphanylium_ylidene_ligand(name: str) -> bool:
    if not name or len(name) > 96:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "carbam", "urea", "azanid")):
        return False
    return True


def _stable_phosphanylium_ylidene_substituent(name: str) -> bool:
    if not name or len(name) > 96:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "carbam", "urea", "azanid")):
        return False
    return name.endswith("ylidene")


def _simple_ligand_name(name: str) -> bool:
    return bool(name) and "-" not in name and "(" not in name and ")" not in name and "," not in name


def _benzotriazolyloxy_amino_phosphonium_name(mol, Chem, pidx: int) -> str:
    amino_names = []
    oxy_ligand = ""
    covered = {pidx}
    for nbr in mol.GetAtomWithIdx(pidx).GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if nbr.GetSymbol() == "N":
            prefix = _simple_or_branch_prefix(mol, nidx, pidx)
            if not prefix:
                return ""
            amino_names.append(prefix)
            covered.update(_collect_component_atoms(mol, nidx, pidx))
            continue
        if nbr.GetSymbol() == "O":
            if oxy_ligand:
                return ""
            oxy_ligand, oxy_atoms = _benzotriazol_1_yloxy_ligand(mol, Chem, nidx, pidx)
            if not oxy_ligand:
                return ""
            covered.update(oxy_atoms)
            continue
        return ""
    if len(amino_names) != 3 or not oxy_ligand or covered != set(range(mol.GetNumAtoms())):
        return ""
    amino_text = _phosphorus_repeated_ligand_text(amino_names)
    return oxy_ligand + amino_text + "phosphonium" if amino_text else ""


def _benzotriazol_1_yloxy_ligand(mol, Chem, oxygen_idx: int, phosphorus_idx: int) -> tuple[str, set[int]]:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0:
        return "", set()
    heavy = [
        atom.GetIdx()
        for atom in oxygen.GetNeighbors()
        if atom.GetIdx() != phosphorus_idx and atom.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return "", set()
    attach_idx = heavy[0]
    attach = mol.GetAtomWithIdx(attach_idx)
    if attach.GetSymbol() != "N" or not attach.GetIsAromatic() or not attach.IsInRing():
        return "", set()
    component = _collect_component_atoms(mol, attach_idx, oxygen_idx)
    if len(component) != 9:
        return "", set()
    counts = Counter(mol.GetAtomWithIdx(idx).GetSymbol() for idx in component)
    if counts != {"C": 6, "N": 3}:
        return "", set()
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings() if set(ring) <= component]
    if sorted(len(ring) for ring in rings) != [5, 6]:
        return "", set()
    if len(rings[0] & rings[1]) != 2:
        return "", set()
    if any(not mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in component):
        return "", set()
    return "benzotriazol-1-yloxy", component | {oxygen_idx}


def _phosphorus_repeated_ligand_text(names: Sequence[str]) -> str:
    counts = Counter(names)
    if len(counts) != 1:
        return ""
    name, count = next(iter(counts.items()))
    if count == 1:
        return name
    multiplier = _PHOSPHORANE_COMPLEX_MULTIPLIERS.get(count, "")
    return f"{multiplier}({_strip_outer_parentheses(name)})" if multiplier else ""


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
        covered = {nidx}
        for nbr_idx in neighbors:
            prefix = _azanium_ligand_name(mol, self._Chem, nidx, nbr_idx)
            if not prefix:
                return
            names.append(prefix)
            covered.update(_collect_component_atoms(mol, nbr_idx, nidx))
        if covered != set(range(mol.GetNumAtoms())):
            return
        name = _azanium_name_for_ligands(names)
        if name:
            parent_atoms = (
                set(range(mol.GetNumAtoms()))
                if _complete_cycloalkyl_azanium_scope(mol, self._Chem, nidx, neighbors)
                or _complete_alkoxycarbonyl_azanium_scope(mol, self._Chem, nidx, neighbors)
                else set()
            )
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=parent_atoms,
                sort_key=(len(name), name),
                estimated_length=len(name),
            )


def _complete_cycloalkyl_azanium_scope(mol, Chem, nitrogen_idx: int, neighbors: Sequence[int]) -> bool:
    ligand_kinds = [_azanium_hydrocarbon_ligand_kind(mol, Chem, nbr_idx, nitrogen_idx) for nbr_idx in neighbors]
    return ligand_kinds.count("cycloalkyl") == 1 and all(
        kind in {"small-alkyl", "cycloalkyl"} for kind in ligand_kinds
    )


def _azanium_name_for_ligands(names: Sequence[str]) -> str:
    alkoxycarbonyl = [name for name in names if name in {"carbobenzoxy", "tert-butoxycarbonyl"}]
    if len(alkoxycarbonyl) == 1:
        other_names = [name for name in names if name not in {"carbobenzoxy", "tert-butoxycarbonyl"}]
        if all(_simple_unparenthesized_n_substituent(name) for name in other_names):
            other_text = _substituted_onium_name(other_names, "")
            if other_names and not other_text:
                return ""
            return other_text + alkoxycarbonyl[0] + "azanium"
    return _azanium_name(names)


def _complete_alkoxycarbonyl_azanium_scope(mol, Chem, nitrogen_idx: int, neighbors: Sequence[int]) -> bool:
    ligand_kinds = []
    alkoxycarbonyl_count = 0
    for nbr_idx in neighbors:
        if _azanium_alkoxycarbonyl_ligand_name(mol, Chem, nitrogen_idx, nbr_idx):
            alkoxycarbonyl_count += 1
            continue
        ligand_kinds.append(_azanium_hydrocarbon_ligand_kind(mol, Chem, nbr_idx, nitrogen_idx))
    return alkoxycarbonyl_count == 1 and all(kind == "small-alkyl" for kind in ligand_kinds)


def _azanium_hydrocarbon_ligand_kind(mol, Chem, attach_idx: int, nitrogen_idx: int) -> str:
    component = _collect_component_atoms(mol, attach_idx, nitrogen_idx)
    if not component or len(component) > 8:
        return ""
    if any(
        mol.GetAtomWithIdx(idx).GetSymbol() != "C" or mol.GetAtomWithIdx(idx).GetFormalCharge() != 0
        for idx in component
    ):
        return ""
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in component and end in component and bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings() if set(ring) <= component]
    if not rings:
        return "small-alkyl" if len(component) <= 4 else ""
    if len(rings) == 1 and rings[0] == component and 3 <= len(component) <= 8:
        return "cycloalkyl"
    return ""


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
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=set(range(mol.GetNumAtoms())),
                estimated_length=len(name),
            )


class HydrazineParentRule:
    family = "hydrazine-parent"
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
        if len(nitrogen_atoms) != 2:
            return
        left, right = nitrogen_atoms
        left_idx = left.GetIdx()
        right_idx = right.GetIdx()
        bond = mol.GetBondBetweenAtoms(left_idx, right_idx)
        if bond is None or bond.GetBondType() != self._Chem.BondType.SINGLE:
            return
        if left.GetTotalNumHs() + right.GetTotalNumHs() + left.GetDegree() + right.GetDegree() != 6:
            return
        names_by_side: list[tuple[str, list[str]]] = []
        covered = {left_idx, right_idx}
        for locant, nidx in (("1", left_idx), ("2", right_idx)):
            names: list[str] = []
            for nbr in mol.GetAtomWithIdx(nidx).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx == (right_idx if nidx == left_idx else left_idx):
                    continue
                bond_to_n = mol.GetBondBetweenAtoms(nidx, nbr_idx)
                if bond_to_n is None or bond_to_n.GetBondType() != self._Chem.BondType.SINGLE:
                    return
                if nbr.GetSymbol() not in {"C", "Si"}:
                    return
                if _is_carbonyl_neighbor(mol, nbr_idx, nidx):
                    return
                prefix = _simple_or_branch_prefix(mol, nbr_idx, nidx)
                if not prefix:
                    return
                names.append(prefix)
                covered.update(_collect_component_atoms(mol, nbr_idx, nidx))
            names_by_side.append((locant, names))
        if covered != set(range(mol.GetNumAtoms())):
            return
        name = _hydrazine_parent_name(names_by_side)
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=covered,
                estimated_length=len(name),
            )


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


def _hydrazine_parent_name(names_by_side: Sequence[tuple[str, Sequence[str]]]) -> str:
    occupied = [(locant, tuple(names)) for locant, names in names_by_side if names]
    if not occupied:
        return "hydrazine"
    if len(occupied) == 1 and occupied[0][0] == "2":
        occupied = [("1", occupied[0][1])]
    pieces: list[str] = []
    grouped: dict[str, list[str]] = {}
    for locant, names in occupied:
        for name in names:
            grouped.setdefault(name, []).append(locant)
    for name, locants in sorted(grouped.items(), key=lambda item: (_strip_outer_parentheses(item[0]), item[1])):
        bare = _strip_outer_parentheses(name)
        locant_text = ",".join(sorted(locants, key=int))
        if len(locants) == 1:
            pieces.append(f"{locant_text}-{bare}")
            continue
        multiplier = MULT.get(len(locants), "")
        if not multiplier:
            return ""
        if _simple_unparenthesized_n_substituent(bare):
            pieces.append(f"{locant_text}-{multiplier}{bare}")
        elif multiplier == "di":
            pieces.append(f"{locant_text}-bis({bare})")
        else:
            pieces.append(f"{locant_text}-{multiplier}({bare})")
    return "-".join(pieces) + "hydrazine" if pieces else ""


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
            safety = None
            if _group14_carbon_ligands_contain_heteroatoms(mol, cidx, carbon_neighbor_indices):
                safety = CandidateSafety(
                    opsin_fragility_reasons=(OPSIN_FRAGILE_GROUP14_HETERO_CARBON_LIGAND,)
                )
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name), safety=safety)


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


def _group14_carbon_ligands_contain_heteroatoms(mol, center_idx: int, carbon_neighbor_indices: Sequence[int]) -> bool:
    for start_idx in carbon_neighbor_indices:
        stack = [start_idx]
        seen = {center_idx}
        ligand_atoms: set[int] = set()
        contains_heteroatom = False
        contains_carbon_bound_halogen = False
        contains_nitrile = False
        all_ligand_bonds_single = True
        while stack:
            atom_idx = stack.pop()
            if atom_idx in seen:
                continue
            seen.add(atom_idx)
            ligand_atoms.add(atom_idx)
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() != "C":
                contains_heteroatom = True
                if atom.GetSymbol() in HALO:
                    contains_carbon_bound_halogen = True
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                if neighbor_idx != center_idx and bond is not None and bond.GetBondTypeAsDouble() > 1.0:
                    all_ligand_bonds_single = False
                if atom.GetSymbol() == "C" and neighbor.GetSymbol() == "N" and bond is not None:
                    if bond.GetBondTypeAsDouble() == 3.0:
                        contains_nitrile = True
                if neighbor_idx not in seen:
                    stack.append(neighbor_idx)
        if contains_carbon_bound_halogen or contains_nitrile:
            return True
        if contains_heteroatom and all_ligand_bonds_single and ligand_atoms:
            return True
    return False


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
        atom_set = set(range(mol.GetNumAtoms()))
        phosphorus_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "P" and atom.GetFormalCharge() == 0 and not atom.IsInRing()
        ]
        if len(phosphorus_atoms) != 1:
            return
        phosphorus = phosphorus_atoms[0]
        pidx = phosphorus.GetIdx()
        phosphine_oxide_name = _single_center_phosphine_oxide_name(mol, self._Chem, pidx, atom_set)
        if phosphine_oxide_name:
            yield NameCandidate(
                phosphine_oxide_name,
                self.family,
                self.priority,
                parent_atoms=atom_set,
                sort_key=(len(phosphine_oxide_name), phosphine_oxide_name),
            )
        phosphine_sulfide_name = _single_center_phosphine_sulfide_name(mol, self._Chem, pidx, atom_set)
        if phosphine_sulfide_name:
            yield NameCandidate(
                phosphine_sulfide_name,
                self.family,
                self.priority,
                parent_atoms=atom_set,
                sort_key=(len(phosphine_sulfide_name), phosphine_sulfide_name),
            )
        carbon_neighbors = []
        terminal_amino_neighbors = []
        substituted_amino_neighbors = []
        double_oxygen_count = 0
        double_sulfur_count = 0
        hydroxy_count = 0
        for nbr in phosphorus.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(pidx, nidx)
            if nbr.GetSymbol() == "C" and bond.GetBondType() == self._Chem.BondType.SINGLE:
                carbon_neighbors.append(nidx)
            elif nbr.GetSymbol() == "O" and bond.GetBondType() == self._Chem.BondType.DOUBLE:
                double_oxygen_count += 1
            elif nbr.GetSymbol() == "S" and bond.GetBondType() == self._Chem.BondType.DOUBLE:
                double_sulfur_count += 1
            elif (
                nbr.GetSymbol() == "O"
                and bond.GetBondType() == self._Chem.BondType.SINGLE
                and nbr.GetFormalCharge() == 0
                and len(nbr.GetNeighbors()) == 1
                and nbr.GetTotalNumHs() == 1
            ):
                hydroxy_count += 1
            elif (
                nbr.GetSymbol() == "N"
                and bond.GetBondType() == self._Chem.BondType.SINGLE
                and nbr.GetFormalCharge() == 0
            ):
                heavy = [
                    atom.GetIdx()
                    for atom in nbr.GetNeighbors()
                    if atom.GetIdx() != pidx and atom.GetAtomicNum() > 1
                ]
                if not heavy:
                    terminal_amino_neighbors.append(nidx)
                elif len(heavy) <= 2:
                    substituted_amino_neighbors.append(nidx)
                else:
                    return
            else:
                return
        if (
            not carbon_neighbors
            and double_oxygen_count == 1
            and double_sulfur_count == 0
            and hydroxy_count == 1
            and phosphorus.GetTotalNumHs() == 1
            and len(terminal_amino_neighbors) + len(substituted_amino_neighbors) == 1
        ):
            if terminal_amino_neighbors:
                prefix = ""
            else:
                substituents = _external_substituent_names(mol, substituted_amino_neighbors[0], pidx)
                prefix = _n_substituent_prefix(substituents or ())
                if prefix is None or _needs_explicit_n_parent_separator(prefix):
                    return
            name = prefix + "phosphonamidic acid"
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))
            return
        if (
            len(carbon_neighbors) == 1
            and double_oxygen_count == 0
            and double_sulfur_count == 0
            and hydroxy_count == 2
            and phosphorus.GetTotalNumHs() == 0
        ):
            prefix = _simple_or_branch_prefix(mol, carbon_neighbors[0], pidx)
            if not prefix:
                return
            if _needs_substituent_parentheses(prefix):
                prefix = f"({prefix})"
            name = prefix + "phosphonous acid"
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))
            return
        if (
            len(carbon_neighbors) == 1
            and double_oxygen_count == 0
            and double_sulfur_count == 1
            and hydroxy_count == 2
            and phosphorus.GetTotalNumHs() == 0
        ):
            prefix = _simple_or_branch_prefix(mol, carbon_neighbors[0], pidx)
            if not prefix:
                return
            if _needs_substituent_parentheses(prefix):
                prefix = f"({prefix})"
            name = prefix + "phosphonothioic acid"
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))
            return
        if len(carbon_neighbors) != 1 or double_oxygen_count != 1:
            return
        if double_sulfur_count != 0:
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
        atom_set = frozenset(range(ctx.mol.GetNumAtoms()))
        for name in _phosphite_ester_names(ctx.mol, self._Chem):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=atom_set,
                sort_key=(len(name), name),
                estimated_length=len(name),
            )


class ChargedOrganophosphorusOxoParentRule:
    family = "charged-organophosphorus-oxo-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_neutral_phosphorus = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        atom_set = set(range(ctx.mol.GetNumAtoms()))
        for name in _charged_organophosphorus_oxo_names(ctx.mol, self._Chem):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=atom_set,
                sort_key=(len(name), name),
            )


class PhosphateEsterRetainedParentRule:
    family = "phosphate-ester-retained-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_neutral_phosphorus = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        atom_set = frozenset(range(atom_count))
        for name in _phosphate_ester_names(mol, self._Chem):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=atom_set,
                source_family="phosphate-ester-retained",
                sort_key=(len(name), name),
                estimated_length=len(name),
            )


class ThiooxoEsterRetainedParentRule:
    family = "thiooxo-ester-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        for name in _thiooxo_ester_names(mol, self._Chem):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )


class SulfinateRetainedParentRule:
    family = "sulfinate-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        for name in _sulfinate_retained_parent_names(mol, self._Chem):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )


class PhosphorothioicTriamideRetainedParentRule:
    family = "phosphorothioic-triamide-retained-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_neutral_phosphorus = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        for name in _phosphorothioic_triamide_names(mol, self._Chem):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )


class DithiophosphateRetainedParentRule:
    family = "dithiophosphate-retained-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_neutral_phosphorus = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        for name in _dithiophosphate_names(mol, self._Chem):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )


class BiphenylDiphosphoniteParentRule:
    family = "biphenyl-diphosphonite-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        phosphorus_groups = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "P":
                continue
            group = _biphenyl_diphosphonite_phosphorus_group(mol, self._Chem, atom.GetIdx())
            if not group:
                return
            phosphorus_groups.append(group)
        if len(phosphorus_groups) != 2:
            return

        ring_records = []
        for group in phosphorus_groups:
            ring = _six_carbon_ring_for_atom(mol, group["ring_carbon"])
            if not ring:
                return
            ring_records.append((group, ring))
        if ring_records[0][1] == ring_records[1][1]:
            return

        bridge = _single_biphenyl_bridge(mol, self._Chem, ring_records[0][1], ring_records[1][1])
        if not bridge:
            return
        bridge_atoms = {bridge[0]: bridge[1], bridge[1]: bridge[0]}

        owner_records = [record for record in ring_records if record[0]["ester_oxygen"] is not None]
        if len(owner_records) > 1:
            return
        if owner_records:
            first_record = owner_records[0]
            second_record = ring_records[1] if ring_records[0] is first_record else ring_records[0]
            ordered_records = [first_record, second_record]
        else:
            ordered_records = sorted(
                ring_records,
                key=lambda record: _biphenyl_ring_locant(mol, record[1], record[0]["ring_carbon"], bridge_atoms),
                reverse=True,
            )

        locants = [
            _biphenyl_ring_locant(mol, ring, group["ring_carbon"], bridge_atoms)
            for group, ring in ordered_records
        ]
        if not all(locants):
            return

        covered_atoms = set()
        for group, ring in ring_records:
            covered_atoms.update(ring)
            covered_atoms.add(group["phosphorus"])
            covered_atoms.update(group["terminal_oxygens"])
            if group["ester_oxygen"] is not None:
                covered_atoms.add(group["ester_oxygen"])
                covered_atoms.update(_collect_component_atoms(mol, group["ester_owner"], group["ester_oxygen"]))

        if covered_atoms != set(range(mol.GetNumAtoms())):
            return

        locant_text = f"{locants[0]},{locants[1]}'-"
        parent_name = locant_text + "biphenyldiphosphonite"
        if owner_records:
            group = owner_records[0][0]
            owner = _single_external_substituent_name(mol, group["ester_oxygen"], group["phosphorus"])
            if not owner:
                return
            name = f"({_strip_outer_parentheses(owner)})-{parent_name}"
        else:
            name = parent_name

        yield NameCandidate(
            name,
            self.family,
            self.priority,
            parent_atoms=set(range(mol.GetNumAtoms())),
            sort_key=(len(name), name),
        )


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


def _single_center_phosphine_oxide_name(mol, Chem, pidx: int, atom_set: set[int]) -> str:
    phosphorus = mol.GetAtomWithIdx(pidx)
    ligand_names = []
    ligand_atom_sets = []
    covered = {pidx}
    double_oxygen_count = 0
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None:
            return ""
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
            covered.add(nidx)
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if nbr.GetSymbol() not in {"C", "N", "O"} or nbr.GetFormalCharge() != 0:
            return ""
        if nbr.GetSymbol() == "O":
            heavy = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != pidx and atom.GetAtomicNum() > 1
            ]
            if len(heavy) != 1:
                return ""
        ligand = _simple_or_branch_prefix(mol, nidx, pidx)
        if not ligand:
            return ""
        ligand_names.append(_strip_outer_parentheses(ligand))
        ligand_atoms = _collect_component_atoms(mol, nidx, pidx)
        ligand_atom_sets.append(frozenset(ligand_atoms))
        covered.update(ligand_atoms)
    if double_oxygen_count != 1 or not ligand_names:
        return ""
    phosphorus_hydrogens = phosphorus.GetTotalNumHs()
    if len(ligand_names) + phosphorus_hydrogens != 3:
        return ""
    if covered != atom_set:
        return ""
    if phosphorus_hydrogens == 0 and not _simple_no_hydrogen_phosphine_oxide_scope(
        mol,
        Chem,
        pidx,
        ligand_atom_sets,
    ):
        return ""
    ligand_text = _phosphine_oxide_ligand_text(ligand_names)
    return ligand_text + "phosphine oxide" if ligand_text else ""


def _single_center_phosphine_sulfide_name(mol, Chem, pidx: int, atom_set: set[int]) -> str:
    phosphorus = mol.GetAtomWithIdx(pidx)
    ligand_names = []
    ligand_atom_sets = []
    covered = {pidx}
    double_sulfur_count = 0
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None:
            return ""
        if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_sulfur_count += 1
            covered.add(nidx)
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if nbr.GetSymbol() not in {"C", "N", "O"} or nbr.GetFormalCharge() != 0:
            return ""
        if nbr.GetSymbol() == "O":
            heavy = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != pidx and atom.GetAtomicNum() > 1
            ]
            if len(heavy) != 1:
                return ""
        ligand = _simple_or_branch_prefix(mol, nidx, pidx)
        if not ligand:
            return ""
        ligand_names.append(_strip_outer_parentheses(ligand))
        ligand_atoms = _collect_component_atoms(mol, nidx, pidx)
        ligand_atom_sets.append(frozenset(ligand_atoms))
        covered.update(ligand_atoms)
    if double_sulfur_count != 1 or not ligand_names:
        return ""
    phosphorus_hydrogens = phosphorus.GetTotalNumHs()
    if len(ligand_names) + phosphorus_hydrogens != 3:
        return ""
    if covered != atom_set:
        return ""
    if phosphorus_hydrogens == 0 and not _simple_no_hydrogen_phosphine_sulfide_scope(
        mol,
        Chem,
        ligand_atom_sets,
    ):
        return ""
    ligand_text = _phosphine_oxide_ligand_text(ligand_names)
    return ligand_text + "phosphine sulfide" if ligand_text else ""


def _simple_no_hydrogen_phosphine_oxide_scope(
    mol,
    Chem,
    phosphorus_idx: int,
    ligand_atom_sets: Sequence[frozenset[int]],
) -> bool:
    if mol.GetNumAtoms() > 48:
        return False
    if mol.GetAtomWithIdx(phosphorus_idx).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
        return False
    if any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds()):
        return False
    atom_rings = [frozenset(ring) for ring in mol.GetRingInfo().AtomRings()]
    for ligand_atoms in ligand_atom_sets:
        heavy_count = sum(
            1 for idx in ligand_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1
        )
        if heavy_count > 32:
            return False
        if sum(1 for ring in atom_rings if ring.issubset(ligand_atoms)) > 4:
            return False
    return True


def _simple_no_hydrogen_phosphine_sulfide_scope(mol, Chem, ligand_atom_sets: Sequence[frozenset[int]]) -> bool:
    if mol.GetNumAtoms() > 48:
        return False
    if any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        return False
    if any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds()):
        return False
    atom_rings = [frozenset(ring) for ring in mol.GetRingInfo().AtomRings()]
    for ligand_atoms in ligand_atom_sets:
        heavy_count = sum(
            1 for idx in ligand_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1
        )
        if heavy_count > 32:
            return False
        if sum(1 for ring in atom_rings if ring.issubset(ligand_atoms)) > 4:
            return False
    return True


def _phosphine_oxide_ligand_text(names: Sequence[str]) -> str:
    parts = []
    counts = Counter(names)
    for name, count in sorted(
        counts.items(),
        key=lambda item: (-item[1], _needs_substituent_parentheses(item[0]), item[0]),
    ):
        bare = _strip_outer_parentheses(name)
        if count == 1:
            if not parts and not _needs_substituent_parentheses(bare):
                parts.append(bare)
            else:
                parts.append(f"({bare})")
            continue
        multiplier = _PHOSPHORANE_COMPLEX_MULTIPLIERS.get(count, "")
        if not multiplier:
            return ""
        parts.append(f"{multiplier}({bare})")
    return "".join(parts)


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
        ylidene_idx = None
        for nbr in phosphorus.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(pidx, nidx)
            if bond.GetBondType() == self._Chem.BondType.DOUBLE and nbr.GetSymbol() == "C":
                if ylidene_idx is not None:
                    return
                ylidene_idx = nidx
                continue
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
        if ylidene_idx is not None:
            if len(carbon_names) + len(halo_names) + phosphorus.GetTotalNumHs() != 3:
                return
            if phosphorus.GetTotalNumHs() != 0 or not carbon_names:
                return
            ylidene_name = _phosphorane_ylidene_name(mol, self._Chem, ylidene_idx, pidx)
            parent_name = _phosphorane_parent_name(carbon_names, halo_names)
            retained_carbonyl_name = _phosphorane_carbonyl_ylide_retained_name(
                mol,
                self._Chem,
                ylidene_idx,
                pidx,
                parent_name,
            )
            if retained_carbonyl_name:
                yield NameCandidate(
                    retained_carbonyl_name,
                    self.family,
                    self.priority,
                    sort_key=(len(retained_carbonyl_name), retained_carbonyl_name),
                    parent_atoms=frozenset(range(mol.GetNumAtoms())),
                    source_family="phosphorane:carbonyl-ylide-retained",
                    estimated_length=len(retained_carbonyl_name),
                )
            if ylidene_name and parent_name:
                name = ylidene_name + parent_name
                yield NameCandidate(
                    name,
                    self.family,
                    self.priority,
                    sort_key=(len(name), name),
                    parent_atoms=frozenset(range(mol.GetNumAtoms())),
                    source_family="phosphorane:ylidene",
                    estimated_length=len(name),
                )
            return
        if len(carbon_names) + len(halo_names) + phosphorus.GetTotalNumHs() != 5:
            return
        if not carbon_names:
            return
        name = _phosphorane_parent_name(carbon_names, halo_names)
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=frozenset(range(mol.GetNumAtoms())),
                source_family="phosphorane:hydride",
                estimated_length=len(name),
            )


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


def _phosphorane_carbonyl_ylide_retained_name(mol, Chem, carbon_idx: int, phosphorus_idx: int, parent_name: str) -> str:
    if not parent_name.endswith("phosphorane"):
        return ""
    carbonyl_idx = _single_ylidene_carbonyl_substituent(mol, Chem, carbon_idx, phosphorus_idx)
    if carbonyl_idx is None:
        return ""
    phosphoranylidene = parent_name[: -len("phosphorane")] + "phosphoranylidene"
    ester_owner = _simple_carbonyl_ester_owner_name(mol, Chem, carbonyl_idx, carbon_idx)
    if ester_owner:
        return f"{ester_owner} 2-({phosphoranylidene})acetate"
    amide_prefix = _weinreb_amide_prefix(mol, Chem, carbonyl_idx, carbon_idx)
    if amide_prefix:
        return f"{amide_prefix}-2-({phosphoranylidene})acetamide"
    return ""


def _phosphorane_ylidene_name(mol, Chem, carbon_idx: int, phosphorus_idx: int) -> str:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return ""
    substituents = [
        nbr.GetIdx()
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != phosphorus_idx and nbr.GetAtomicNum() > 1
    ]
    if len(substituents) > 1:
        return ""
    if not substituents:
        return "methylidene"
    substituent_idx = substituents[0]
    bond = mol.GetBondBetweenAtoms(carbon_idx, substituent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if _is_simple_acetyl_substituent(mol, Chem, substituent_idx, carbon_idx):
        return "2-oxopropylidene"
    if _is_simple_carbonyl_ester_substituent(mol, Chem, substituent_idx, carbon_idx):
        owner = _simple_carbonyl_ester_owner_name(mol, Chem, substituent_idx, carbon_idx)
        if owner == "t-butyl":
            return "tert-butoxycarbonylmethylidene"
    prefix = _simple_or_branch_prefix(mol, substituent_idx, carbon_idx)
    if not prefix:
        return ""
    prefix = _strip_outer_parentheses(prefix)
    if not _stable_phosphorane_ylidene_substituent(prefix):
        return ""
    return prefix + "methylidene"


def _single_ylidene_carbonyl_substituent(mol, Chem, carbon_idx: int, phosphorus_idx: int) -> int | None:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return None
    substituents = [
        nbr.GetIdx()
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != phosphorus_idx and nbr.GetAtomicNum() > 1
    ]
    if len(substituents) != 1:
        return None
    substituent_idx = substituents[0]
    bond = mol.GetBondBetweenAtoms(carbon_idx, substituent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    carbonyl = mol.GetAtomWithIdx(substituent_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0:
        return None
    if not any(
        nbr.GetSymbol() == "O"
        and (bond := mol.GetBondBetweenAtoms(substituent_idx, nbr.GetIdx())) is not None
        and bond.GetBondType() == Chem.BondType.DOUBLE
        for nbr in carbonyl.GetNeighbors()
    ):
        return None
    return substituent_idx


def _is_simple_acetyl_substituent(mol, Chem, carbonyl_idx: int, ylidene_idx: int) -> bool:
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0:
        return False
    double_oxygen_count = 0
    methyl_count = 0
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == ylidene_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            if nbr.GetDegree() != 1 or nbr.GetFormalCharge() != 0:
                return False
            methyl_count += 1
        else:
            return False
    return double_oxygen_count == 1 and methyl_count == 1


def _is_simple_carbonyl_ester_substituent(mol, Chem, carbonyl_idx: int, ylidene_idx: int) -> bool:
    return bool(_simple_carbonyl_ester_owner_name(mol, Chem, carbonyl_idx, ylidene_idx))


def _simple_carbonyl_ester_owner_name(mol, Chem, carbonyl_idx: int, ylidene_idx: int) -> str:
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0:
        return ""
    double_oxygen_count = 0
    ester_oxygen_idx = None
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == ylidene_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == 0:
            heavy = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != carbonyl_idx and atom.GetAtomicNum() > 1
            ]
            if len(heavy) != 1 or ester_oxygen_idx is not None:
                return ""
            ester_oxygen_idx = nidx
        else:
            return ""
    if double_oxygen_count != 1 or ester_oxygen_idx is None:
        return ""
    owner = _single_external_substituent_name(mol, ester_oxygen_idx, carbonyl_idx)
    if not owner or _unstable_retained_functional_parent_substituent(owner):
        return ""
    return owner


def _weinreb_amide_prefix(mol, Chem, carbonyl_idx: int, ylidene_idx: int) -> str:
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0:
        return ""
    double_oxygen_count = 0
    amide_nitrogen_idx = None
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == ylidene_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == 0:
            if amide_nitrogen_idx is not None:
                return ""
            amide_nitrogen_idx = nidx
        else:
            return ""
    if double_oxygen_count != 1 or amide_nitrogen_idx is None:
        return ""
    substituents = _external_substituent_names(mol, amide_nitrogen_idx, carbonyl_idx)
    if substituents is None:
        return ""
    if sorted(substituents) == ["methoxy", "methyl"]:
        return "N-methoxy-N-methyl"
    prefix = _n_substituent_prefix(substituents)
    if prefix != "N-methoxy-N-methyl":
        return ""
    return prefix


def _stable_phosphorane_ylidene_substituent(prefix: str) -> bool:
    if prefix in {"cyano", "phenyl"}:
        return True
    if prefix.endswith("carbonyl"):
        return True
    if prefix.endswith("phenyl") and len(prefix) <= 80:
        return True
    if re.fullmatch(r"(?:methyl|ethyl|propyl|butyl|pentyl|hexyl)", prefix):
        return True
    return False


_PHOSPHORANE_COMPLEX_MULTIPLIERS = {
    2: "bis",
    3: "tris",
    4: "tetrakis",
    5: "pentakis",
}


class AminophosphaneParentRule:
    family = "aminophosphane-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_neutral_phosphorus = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        phosphorus_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "P" and atom.GetFormalCharge() == 0 and not atom.IsInRing()
        ]
        if len(phosphorus_atoms) != 1:
            return
        phosphorus = phosphorus_atoms[0]
        pidx = phosphorus.GetIdx()
        nitrogen_neighbors = []
        imino_neighbors = []
        for nbr in phosphorus.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(pidx, nidx)
            if nbr.GetSymbol() != "N":
                return
            if bond.GetBondType() == self._Chem.BondType.SINGLE:
                nitrogen_neighbors.append(nidx)
            elif bond.GetBondType() == self._Chem.BondType.DOUBLE:
                imino_neighbors.append(nidx)
            else:
                return
        if imino_neighbors:
            name = _phosphanimine_name(mol, self._Chem, pidx, nitrogen_neighbors, imino_neighbors)
            if name:
                yield NameCandidate(
                    name,
                    self.family,
                    self.priority,
                    parent_atoms=set(range(atom_count)),
                    sort_key=(len(name), name),
                )
            return
        if len(nitrogen_neighbors) + phosphorus.GetTotalNumHs() != 3:
            return
        if not nitrogen_neighbors:
            return
        names = []
        covered = {pidx}
        for nidx in nitrogen_neighbors:
            prefix = _simple_or_branch_prefix(mol, nidx, pidx)
            if not prefix:
                return
            names.append(prefix)
            covered.update(_collect_component_atoms(mol, nidx, pidx))
        if covered != atom_set:
            return
        name = _pnictogen_hydride_name(names, "phosphane")
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )


def _phosphanimine_name(mol, Chem, pidx: int, nitrogen_neighbors: Sequence[int], imino_neighbors: Sequence[int]) -> str:
    if len(imino_neighbors) != 1 or len(nitrogen_neighbors) != 3:
        return ""
    phosphorus = mol.GetAtomWithIdx(pidx)
    if phosphorus.GetFormalCharge() != 0 or phosphorus.GetTotalNumHs() != 0:
        return ""
    covered = {pidx}
    ligand_names = []
    for nidx in nitrogen_neighbors:
        nitrogen = mol.GetAtomWithIdx(nidx)
        if nitrogen.GetFormalCharge() != 0:
            return ""
        prefix = _simple_or_branch_prefix(mol, nidx, pidx)
        if not prefix:
            return ""
        ligand_names.append(prefix)
        covered.update(_collect_component_atoms(mol, nidx, pidx))
    imino_idx = imino_neighbors[0]
    imino = mol.GetAtomWithIdx(imino_idx)
    if imino.GetFormalCharge() != 0:
        return ""
    imino_substituents = [
        nbr.GetIdx()
        for nbr in imino.GetNeighbors()
        if nbr.GetIdx() != pidx and nbr.GetAtomicNum() > 1
    ]
    if len(imino_substituents) != 1:
        return ""
    imino_owner = _simple_or_branch_prefix(mol, imino_substituents[0], imino_idx)
    if not imino_owner or _unstable_retained_functional_parent_substituent(imino_owner):
        return ""
    covered.add(imino_idx)
    covered.update(_collect_component_atoms(mol, imino_substituents[0], imino_idx))
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    ligand_text = _phosphorus_repeated_ligand_text(ligand_names)
    if not ligand_text:
        return ""
    owner_text = _strip_outer_parentheses(imino_owner)
    return f"{owner_text}imino-{ligand_text}phosphine"


class DithiazoleRetainedParentRule:
    family = "dithiazole-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem
        self._pattern = Chem.MolFromSmarts("[#6,#7]-c1nc(=[O,S])ss1")

    def candidates(self, ctx):
        mol = ctx.mol
        if self._pattern is None:
            return
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        for match in mol.GetSubstructMatches(self._pattern):
            substituent_idx = match[0]
            ring_carbon_idx = match[1]
            exo_idx = match[4]
            prefix = _simple_or_branch_prefix(mol, substituent_idx, ring_carbon_idx)
            if not prefix:
                continue
            covered = set(match)
            covered.update(_collect_component_atoms(mol, substituent_idx, ring_carbon_idx))
            if covered != atom_set:
                continue
            exo_symbol = mol.GetAtomWithIdx(exo_idx).GetSymbol()
            suffix = {"O": "one", "S": "thione"}.get(exo_symbol, "")
            if not suffix:
                continue
            prefix = _parenthesized_if_complex(_strip_outer_parentheses(prefix))
            name = f"5-{prefix}-1,2,4-dithiazole-3-{suffix}"
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )
            return


class ThiadiazoleDithioRetainedParentRule:
    family = "thiadiazole-dithio-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_set = set(range(mol.GetNumAtoms()))
        for ring in mol.GetRingInfo().AtomRings():
            name = _thiadiazole_dithio_name(mol, self._Chem, ring)
            if not name:
                continue
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=atom_set,
                sort_key=(len(name), name),
            )
            return


def _thiadiazole_dithio_name(mol, Chem, ring: Sequence[int]) -> str:
    if len(ring) != 5:
        return ""
    symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring]
    if Counter(symbols) != Counter({"C": 2, "N": 2, "S": 1}):
        return ""
    ring_set = set(ring)
    for seq in _cyclic_ring_sequences(tuple(ring)):
        parent = _thiadiazole_parent_from_sequence(mol, seq)
        if not parent:
            continue
        substituents = []
        covered = set(seq)
        for position, atom_idx in enumerate(seq, start=1):
            if mol.GetAtomWithIdx(atom_idx).GetSymbol() != "C":
                continue
            dithio = _ring_carbon_dithio_substituent(mol, Chem, atom_idx, ring_set)
            if dithio is None:
                continue
            owner, atoms = dithio
            substituents.append((position, owner))
            covered.update(atoms)
        if len(substituents) != 2 or covered != set(range(mol.GetNumAtoms())):
            continue
        owners = [owner for _, owner in substituents]
        locants = ",".join(str(position) for position, _ in sorted(substituents))
        if owners[0] == owners[1]:
            return f"{locants}-bis({_strip_outer_parentheses(owners[0])}dithio)-{parent}"
        pieces = [
            f"{position}-{_parenthesized_if_complex(_strip_outer_parentheses(owner))}dithio"
            for position, owner in sorted(substituents)
        ]
        return "-".join(pieces) + f"-{parent}"
    return ""


def _cyclic_ring_sequences(ring: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    sequences = []
    n = len(ring)
    base = tuple(ring)
    for direction in (base, tuple(reversed(base))):
        for start in range(n):
            sequences.append(direction[start:] + direction[:start])
    return tuple(dict.fromkeys(sequences))


def _thiadiazole_parent_from_sequence(mol, seq: Sequence[int]) -> str:
    if mol.GetAtomWithIdx(seq[0]).GetSymbol() != "S":
        return ""
    n_positions = tuple(
        position
        for position, atom_idx in enumerate(seq, start=1)
        if mol.GetAtomWithIdx(atom_idx).GetSymbol() == "N"
    )
    return {
        (2, 3): "1,2,3-thiadiazole",
        (2, 4): "1,2,4-thiadiazole",
        (3, 4): "1,3,4-thiadiazole",
    }.get(n_positions, "")


def _ring_carbon_dithio_substituent(mol, Chem, carbon_idx: int, ring_set: set[int]) -> tuple[str, set[int]] | None:
    matches = []
    carbon = mol.GetAtomWithIdx(carbon_idx)
    for inner_sulfur in carbon.GetNeighbors():
        inner_idx = inner_sulfur.GetIdx()
        if inner_idx in ring_set or inner_sulfur.GetSymbol() != "S" or inner_sulfur.GetFormalCharge() != 0:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, inner_idx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        external = [
            nbr.GetIdx()
            for nbr in inner_sulfur.GetNeighbors()
            if nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1
        ]
        if len(external) != 1:
            continue
        external_idx = external[0]
        external_sulfur = mol.GetAtomWithIdx(external_idx)
        if external_sulfur.GetSymbol() != "S" or external_sulfur.GetFormalCharge() != 0:
            continue
        owner_atoms = [
            nbr.GetIdx()
            for nbr in external_sulfur.GetNeighbors()
            if nbr.GetIdx() != inner_idx and nbr.GetAtomicNum() > 1
        ]
        if len(owner_atoms) != 1:
            continue
        owner_idx = owner_atoms[0]
        owner = _simple_or_branch_prefix(mol, owner_idx, external_idx)
        if not owner or _unstable_retained_functional_parent_substituent(owner):
            continue
        atoms = {inner_idx, external_idx}
        atoms.update(_collect_component_atoms(mol, owner_idx, external_idx))
        matches.append((_strip_outer_parentheses(owner), atoms))
    if len(matches) != 1:
        return None
    return matches[0]


class OxazinoneRetainedParentRule:
    family = "oxazinone-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        name = _oxazinone_retained_parent_name(ctx.mol, self._Chem)
        if not name:
            return
        yield NameCandidate(
            name,
            self.family,
            self.priority,
            parent_atoms=set(range(ctx.mol.GetNumAtoms())),
            sort_key=(len(name), name),
        )


class OxiraneDioneRetainedParentRule:
    family = "oxirane-dione-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        name = _oxirane_dione_name(ctx.mol, self._Chem)
        if not name:
            return
        yield NameCandidate(
            name,
            self.family,
            self.priority,
            parent_atoms=set(range(ctx.mol.GetNumAtoms())),
            sort_key=(len(name), name),
        )


def _oxirane_dione_name(mol, Chem) -> str:
    atom_set = set(range(mol.GetNumAtoms()))
    for ring in mol.GetRingInfo().AtomRings():
        ring_set = set(ring)
        if len(ring) != 3:
            continue
        ring_symbols = sorted(mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring)
        if ring_symbols != ["C", "C", "O"]:
            continue
        covered = set(ring)
        ok = True
        for ring_idx in ring:
            atom = mol.GetAtomWithIdx(ring_idx)
            if atom.GetSymbol() == "O":
                continue
            double_oxygens = []
            for nbr in atom.GetNeighbors():
                nidx = nbr.GetIdx()
                bond = mol.GetBondBetweenAtoms(ring_idx, nidx)
                if nidx in ring_set:
                    continue
                if (
                    nbr.GetSymbol() == "O"
                    and bond is not None
                    and bond.GetBondType() == Chem.BondType.DOUBLE
                    and not nbr.IsInRing()
                ):
                    double_oxygens.append(nidx)
                elif nbr.GetAtomicNum() > 1:
                    ok = False
                    break
            if not ok or len(double_oxygens) != 1:
                ok = False
                break
            covered.update(double_oxygens)
        if ok and covered == atom_set:
            return "oxirane-2,3-dione"
    return ""


class PiperazineDioneYlideneRetainedParentRule:
    family = "piperazine-dione-ylidene-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        seen = set()
        for name in _piperazine_dione_ylidene_names(ctx.mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(ctx.mol.GetNumAtoms())),
                sort_key=(len(name), name),
                estimated_length=len(name),
            )


def _piperazine_dione_ylidene_names(mol, Chem) -> list[str]:
    atom_set = set(range(mol.GetNumAtoms()))
    names: list[str] = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        ring_set = set(ring)
        if ring_system_extends_core(mol, ring_set):
            continue
        ordered = ordered_ring(mol, ring)
        if not ordered:
            continue
        for seq in _cyclic_ring_sequences(ordered):
            record = _piperazine_dione_sequence_name(mol, Chem, seq, ring_set)
            if record is None:
                continue
            name, covered = record
            if covered == atom_set:
                names.append(name)
    return sorted(dict.fromkeys(names), key=lambda item: (len(item), item))


def _piperazine_dione_sequence_name(
    mol,
    Chem,
    seq: Sequence[int],
    ring_set: set[int],
) -> tuple[str, set[int]] | None:
    if mol.GetAtomWithIdx(seq[0]).GetSymbol() != "N" or mol.GetAtomWithIdx(seq[3]).GetSymbol() != "N":
        return None
    if any(mol.GetAtomWithIdx(idx).GetFormalCharge() != 0 for idx in seq):
        return None
    carbonyl_oxygens = [
        _piperazine_dione_carbonyl_oxygen(mol, Chem, seq[1], ring_set),
        _piperazine_dione_carbonyl_oxygen(mol, Chem, seq[4], ring_set),
    ]
    if any(oxygen is None for oxygen in carbonyl_oxygens):
        return None
    ylidene_records = [
        _piperazine_dione_ylidene_substituent(mol, Chem, seq[2], ring_set),
        _piperazine_dione_ylidene_substituent(mol, Chem, seq[5], ring_set),
    ]
    if any(record is None for record in ylidene_records):
        return None
    covered = set(seq)
    covered.update(oxygen for oxygen in carbonyl_oxygens if oxygen is not None)
    for _, atoms in ylidene_records:
        covered.update(atoms)
    name = (
        f"3-{ylidene_records[0][0]}-"
        f"6-{ylidene_records[1][0]}"
        "piperazine-2,5-dione"
    )
    return name, covered


def _piperazine_dione_carbonyl_oxygen(mol, Chem, carbon_idx: int, ring_set: set[int]) -> int | None:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return None
    oxygens = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx in ring_set:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if (
            nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() == 0
            and not nbr.IsInRing()
            and bond is not None
            and bond.GetBondType() == Chem.BondType.DOUBLE
        ):
            oxygens.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    return oxygens[0] if len(oxygens) == 1 else None


def _piperazine_dione_ylidene_substituent(
    mol,
    Chem,
    carbon_idx: int,
    ring_set: set[int],
) -> tuple[str, set[int]] | None:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return None
    matches = []
    for nbr in carbon.GetNeighbors():
        exo_idx = nbr.GetIdx()
        if exo_idx in ring_set:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, exo_idx)
        if (
            nbr.GetSymbol() != "C"
            or nbr.GetFormalCharge() != 0
            or nbr.IsInRing()
            or bond is None
            or bond.GetBondType() != Chem.BondType.DOUBLE
        ):
            continue
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            continue
        owners = [
            owner.GetIdx()
            for owner in nbr.GetNeighbors()
            if owner.GetIdx() != carbon_idx and owner.GetAtomicNum() > 1
        ]
        if len(owners) != 1:
            continue
        owner_idx = owners[0]
        owner_bond = mol.GetBondBetweenAtoms(exo_idx, owner_idx)
        if owner_bond is None or owner_bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        prefix = _simple_or_branch_prefix(mol, owner_idx, exo_idx)
        if not prefix or _unstable_retained_functional_parent_substituent(prefix):
            continue
        component = _collect_component_atoms(mol, owner_idx, exo_idx)
        if component & ring_set:
            continue
        unit = _piperazine_dione_methylidene_unit(prefix)
        if not unit:
            continue
        matches.append((unit, {exo_idx} | component))
    if len(matches) != 1:
        return None
    return matches[0]


def _piperazine_dione_methylidene_unit(prefix: str) -> str:
    prefix = _strip_outer_parentheses(prefix)
    if not prefix or " " in prefix or "[" in prefix or "]" in prefix:
        return ""
    return f"({_parenthesized_if_complex(prefix)}methylidene)"


class AzidobenzylideneCyclohexanoneRetainedParentRule:
    family = "azidobenzylidene-cyclohexanone-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._patterns = (
            (
                Chem.MolFromSmarts(
                    "CC1CC(=Cc2ccc(N=[N+]=[N-])cc2)C(=O)C(=Cc2ccc(N=[N+]=[N-])cc2)C1"
                ),
                "2,6-bis(4-azidobenzylidene)-4-methylcyclohexanone",
            ),
            (
                Chem.MolFromSmarts(
                    "O=C1C(=Cc2ccc(N=[N+]=[N-])cc2)CCCC1=Cc1ccc(N=[N+]=[N-])cc1"
                ),
                "2,6-bis(4-azidobenzylidene)cyclohexanone",
            ),
        )

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        for pattern, name in self._patterns:
            if pattern is None:
                continue
            for match in mol.GetSubstructMatches(pattern):
                if set(match) != atom_set:
                    continue
                yield NameCandidate(
                    name,
                    self.family,
                    self.priority,
                    parent_atoms=set(range(atom_count)),
                    sort_key=(len(name), name),
                )
                return


class BisDiethylaminobenzylideneCyclopentanoneRetainedParentRule:
    family = "bis-diethylaminobenzylidene-cyclopentanone-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._pattern = Chem.MolFromSmarts(
            "CCN(CC)c1ccc(C=C2CCC(=Cc3ccc(N(CC)CC)cc3)C2=O)cc1"
        )

    def candidates(self, ctx):
        if self._pattern is None:
            return
        mol = ctx.mol
        atom_set = set(range(mol.GetNumAtoms()))
        for match in mol.GetSubstructMatches(self._pattern):
            if set(match) != atom_set:
                continue
            name = "2,5-bis(4-diethylaminobenzylidene)cyclopentanone"
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(mol.GetNumAtoms())),
                sort_key=(len(name), name),
            )
            return


class CyclopropaneTriylideneNitrileRetainedParentRule:
    family = "cyclopropane-triylidene-nitrile-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        name = _cyclopropane_triylidene_nitrile_name(ctx.mol, self._Chem)
        if not name:
            return
        yield NameCandidate(
            name,
            self.family,
            self.priority,
            parent_atoms=set(range(ctx.mol.GetNumAtoms())),
            sort_key=(len(name), name),
        )


def _cyclopropane_triylidene_nitrile_name(mol, Chem) -> str:
    atom_set = set(range(mol.GetNumAtoms()))
    for ring in mol.GetRingInfo().AtomRings():
        ring_set = set(ring)
        if len(ring) != 3:
            continue
        if any(mol.GetAtomWithIdx(idx).GetSymbol() != "C" for idx in ring):
            continue
        if any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            continue
        records = []
        covered = set(ring)
        ok = True
        for ring_idx in ring:
            exocyclic = []
            for nbr in mol.GetAtomWithIdx(ring_idx).GetNeighbors():
                nidx = nbr.GetIdx()
                bond = mol.GetBondBetweenAtoms(ring_idx, nidx)
                if nidx in ring_set:
                    continue
                if (
                    nbr.GetSymbol() == "C"
                    and nbr.GetFormalCharge() == 0
                    and bond is not None
                    and bond.GetBondType() == Chem.BondType.DOUBLE
                ):
                    exocyclic.append(nidx)
                elif nbr.GetAtomicNum() > 1:
                    ok = False
                    break
            if not ok or len(exocyclic) != 1:
                ok = False
                break
            exo_idx = exocyclic[0]
            substituents = [
                nbr.GetIdx()
                for nbr in mol.GetAtomWithIdx(exo_idx).GetNeighbors()
                if nbr.GetIdx() != ring_idx and nbr.GetAtomicNum() > 1
            ]
            if len(substituents) != 2:
                ok = False
                break
            prefixes = []
            covered.add(exo_idx)
            for substituent_idx in substituents:
                prefix = _simple_or_branch_prefix(mol, substituent_idx, exo_idx)
                if not prefix:
                    ok = False
                    break
                prefixes.append(_strip_outer_parentheses(prefix))
                covered.update(_collect_component_atoms(mol, substituent_idx, exo_idx))
            if not ok or prefixes.count("cyano") != 1:
                ok = False
                break
            other = next(prefix for prefix in prefixes if prefix != "cyano")
            if not _stable_cyclopropane_triylidene_substituent(other):
                ok = False
                break
            records.append(other)
        if not ok or covered != atom_set or len(records) != 3:
            continue
        counts = Counter(records)
        if len(counts) != 1:
            continue
        substituent = next(iter(counts))
        return f"1,2,3-tri[(cyano)({substituent})methylene]cyclopropane"
    return ""


def _stable_cyclopropane_triylidene_substituent(name: str) -> bool:
    if len(name) > 96:
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9,()'\-]+", name))


class OrganicPeroxideRetainedParentRule:
    family = "organic-peroxide-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        for name in _alcohol_oxide_retained_parent_names(mol, self._Chem):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )
            return
        diacylperoxy_name = _diacylperoxy_dimethylhexane_name(mol, self._Chem)
        if diacylperoxy_name:
            yield NameCandidate(
                diacylperoxy_name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(diacylperoxy_name), diacylperoxy_name),
            )
            return
        for bond in mol.GetBonds():
            if bond.GetBondType() != self._Chem.BondType.SINGLE:
                continue
            begin = bond.GetBeginAtom()
            end = bond.GetEndAtom()
            if begin.GetSymbol() != "O" or end.GetSymbol() != "O":
                continue
            if begin.GetFormalCharge() != 0 or end.GetFormalCharge() != 0:
                continue
            ligand_names = []
            covered = {begin.GetIdx(), end.GetIdx()}
            for oxygen_idx, other_oxygen_idx in (
                (begin.GetIdx(), end.GetIdx()),
                (end.GetIdx(), begin.GetIdx()),
            ):
                heavy = [
                    nbr.GetIdx()
                    for nbr in mol.GetAtomWithIdx(oxygen_idx).GetNeighbors()
                    if nbr.GetIdx() != other_oxygen_idx and nbr.GetAtomicNum() > 1
                ]
                if len(heavy) != 1:
                    break
                ligand_idx = heavy[0]
                if mol.GetAtomWithIdx(ligand_idx).GetSymbol() != "C":
                    break
                prefix = _simple_or_branch_prefix(mol, ligand_idx, oxygen_idx)
                if not prefix:
                    break
                ligand_names.append(_strip_outer_parentheses(prefix))
                covered.update(_collect_component_atoms(mol, ligand_idx, oxygen_idx))
            else:
                if covered != atom_set or len(ligand_names) != 2 or ligand_names[0] != ligand_names[1]:
                    continue
                ligand = ligand_names[0]
                name = f"bis({_strip_outer_parentheses(ligand)}) peroxide"
                yield NameCandidate(
                    name,
                    self.family,
                    self.priority,
                    parent_atoms=set(range(atom_count)),
                    sort_key=(len(name), name),
                )
                return


class PhosphorylCholineRetainedParentRule:
    family = "phosphorylcholine-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._patterns = (
            (
                Chem.MolFromSmarts("C=C(C)C(=O)OCC[P+]([O-])=C(O)C[N+](C)(C)C"),
                "2-methacryloyloxyethylphosphorylcholine",
            ),
            (
                Chem.MolFromSmarts("C=CC(=O)OCC[P+]([O-])=C(O)C[N+](C)(C)C"),
                "2-acryloyloxyethylphosphorylcholine",
            ),
            (
                Chem.MolFromSmarts("C[N+](C)(C#P=O)CCO"),
                "phosphorylcholine",
            ),
        )

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        for pattern, name in self._patterns:
            if pattern is None:
                continue
            for match in mol.GetSubstructMatches(pattern):
                if set(match) != atom_set:
                    continue
                yield NameCandidate(
                    name,
                    self.family,
                    self.priority,
                    parent_atoms=set(range(atom_count)),
                    sort_key=(len(name), name),
                )
                return


class AzinobisBenzothiazolineRetainedParentRule:
    family = "azinobis-benzothiazoline-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._patterns = (
            (
                Chem.MolFromSmarts(
                    "CCn1c(=NN=c2sc3cc(S(=O)(=O)[O-])ccc3n2CC)sc2cc(S(=O)(=O)[O-])ccc21"
                ),
                "2,2'-azinobis(3-ethylbenzothiazoline-6-sulfonate)",
            ),
            (
                Chem.MolFromSmarts(
                    "CCn1c(=NN=c2sc3cc(S(=O)(=O)O)ccc3n2CC)sc2cc(S(=O)(=O)O)ccc21"
                ),
                "2,2'-azinobis(3-ethylbenzothiazoline-6-sulfonic acid)",
            ),
            (
                Chem.MolFromSmarts("CCn1c(=NN=c2sc3ccccc3n2CC)sc2ccccc21"),
                "2,2'-azinobis(3-ethylbenzothiazoline)",
            ),
        )

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        for pattern, name in self._patterns:
            if pattern is None:
                continue
            for match in mol.GetSubstructMatches(pattern):
                if set(match) != atom_set:
                    continue
                yield NameCandidate(
                    name,
                    self.family,
                    self.priority,
                    parent_atoms=set(range(atom_count)),
                    sort_key=(len(name), name),
                )
                return


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
        self._triacyloxy_lactone = Chem.MolFromSmarts(
            "O=C1O[I](OC(=O)[#6])(OC(=O)[#6])(OC(=O)[#6])c2ccccc21"
        )
        self._lactone = Chem.MolFromSmarts("O=C1O[I]([#6])c2ccccc21")
        self._hydride_lactone = Chem.MolFromSmarts("O=C1O[IH]c2ccccc21")
        self._dihydro_hydride_lactone = Chem.MolFromSmarts("O=C1O[IH]C2=CC=CCC12")
        self._hydroxy_lactone = Chem.MolFromSmarts("O=C1O[I](O)c2ccccc21")
        self._halo_lactone = Chem.MolFromSmarts("O=C1O[I]([F,Cl,Br,I])c2ccccc21")
        self._geminal = Chem.MolFromSmarts("[#6]C1([#6])O[I]([#6,F,Cl,Br,I])c2ccccc21")
        self._iodoxy_lactone = Chem.MolFromSmarts("O=C1O[I](=O)(O)c2ccccc21")
        self._hydride = Chem.MolFromSmarts("c1ccc2c(c1)CO[I]2")
        self._substituted_hydride = Chem.MolFromSmarts("[!#1][I]1OCc2ccccc21")
        self._oxo_substituted_hydride = Chem.MolFromSmarts("O=[I]1([!#1])OCc2ccccc21")
        self._oxo_ring_substituted_hydride = Chem.MolFromSmarts("[#6]C1O[IH](=O)c2ccccc21")

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        for pattern, namer in (
            (self._triacyloxy_lactone, _benziodoxole_triacyloxy_lactone_name),
            (self._iodoxy_lactone, _benziodoxole_iodoxy_lactone_name),
            (self._hydroxy_lactone, _benziodoxole_hydroxy_lactone_name),
            (self._lactone, _benziodoxole_lactone_name),
            (self._dihydro_hydride_lactone, _benziodoxole_dihydro_hydride_lactone_name),
            (self._hydride_lactone, _benziodoxole_hydride_lactone_name),
            (self._halo_lactone, _benziodoxole_halo_lactone_name),
            (self._geminal, _benziodoxole_geminal_name),
            (self._oxo_substituted_hydride, _benziodoxole_oxo_substituted_hydride_name),
            (self._oxo_ring_substituted_hydride, _benziodoxole_oxo_ring_substituted_hydride_name),
            (self._substituted_hydride, _benziodoxole_substituted_hydride_name),
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


def _benziodoxole_triacyloxy_lactone_name(mol, Chem, match: Sequence[int]) -> str:
    if set(match) != set(range(mol.GetNumAtoms())):
        return ""
    iodine_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "I" and atom.IsInRing()]
    if len(iodine_atoms) != 1:
        return ""
    iodine_idx = iodine_atoms[0].GetIdx()
    ligand_names = []
    for nbr in mol.GetAtomWithIdx(iodine_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(iodine_idx, nidx)
        if nbr.GetSymbol() != "O" or nbr.IsInRing() or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        ligand = _simple_or_branch_prefix(mol, nidx, iodine_idx)
        if not ligand:
            return ""
        ligand_names.append(_strip_outer_parentheses(ligand))
    if len(ligand_names) != 3 or len(set(ligand_names)) != 1:
        return ""
    ligand = ligand_names[0]
    if not ligand.endswith("yloxy"):
        return ""
    return f"1,1,1-tris({ligand})-1lambda5,2-benziodoxol-3(1H)-one"


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


def _benziodoxole_hydride_lactone_name(mol, Chem, match: Sequence[int]) -> str:
    iodine_atoms = [
        atom
        for atom in mol.GetAtoms()
        if (
            atom.GetSymbol() == "I"
            and atom.IsInRing()
            and atom.GetFormalCharge() == 0
            and atom.GetTotalNumHs() == 1
        )
    ]
    if len(iodine_atoms) != 1:
        return ""
    if set(match) == set(range(mol.GetNumAtoms())):
        return "3,2-benziodoxol-3-one"
    substituents = _benziodoxole_hydride_lactone_ring_substituents(mol, match)
    if len(substituents) != 1:
        return ""
    locant, prefix, covered = substituents[0]
    if set(match) | covered != set(range(mol.GetNumAtoms())):
        return ""
    prefix = _parenthesized_if_complex(_strip_outer_parentheses(prefix))
    return f"{locant}-{prefix}-3,2-benziodoxol-3-one"


def _benziodoxole_dihydro_hydride_lactone_name(mol, Chem, match: Sequence[int]) -> str:
    if set(match) != set(range(mol.GetNumAtoms())):
        return ""
    iodine_atoms = [
        atom
        for atom in mol.GetAtoms()
        if (
            atom.GetSymbol() == "I"
            and atom.IsInRing()
            and atom.GetFormalCharge() == 0
            and atom.GetTotalNumHs() == 1
        )
    ]
    if len(iodine_atoms) != 1:
        return ""
    return "3,4-dihydro-1,2-benziodoxol-3-one"


def _benziodoxole_hydride_lactone_ring_substituents(
    mol,
    match: Sequence[int],
) -> list[tuple[str, str, set[int]]]:
    locants = {
        match[8]: "4",
        match[7]: "5",
        match[6]: "6",
        match[5]: "7",
    }
    parent_atoms = set(match)
    substituents: list[tuple[str, str, set[int]]] = []
    for atom_idx, locant in locants.items():
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in parent_atoms:
                continue
            prefix = _simple_or_branch_prefix(mol, nidx, atom_idx)
            if not prefix:
                continue
            substituents.append((locant, prefix, _collect_component_atoms(mol, nidx, atom_idx)))
    return substituents


def _benziodoxole_halo_lactone_name(mol, Chem, match: Sequence[int]) -> str:
    if set(match) != set(range(mol.GetNumAtoms())):
        return ""
    iodine_atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "I" and atom.IsInRing() and atom.GetFormalCharge() == 0
    ]
    if len(iodine_atoms) != 1:
        return ""
    iodine_idx = iodine_atoms[0].GetIdx()
    halogen_neighbors = [
        nbr
        for nbr in mol.GetAtomWithIdx(iodine_idx).GetNeighbors()
        if nbr.GetSymbol() in HALO and not nbr.IsInRing()
    ]
    if len(halogen_neighbors) != 1:
        return ""
    halo = HALO[halogen_neighbors[0].GetSymbol()]
    return f"1-{halo}-1,2-benziodoxol-3-one"


def _benziodoxole_hydroxy_lactone_name(mol, Chem, match: Sequence[int]) -> str:
    if set(match) != set(range(mol.GetNumAtoms())):
        return ""
    iodine_atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "I" and atom.IsInRing() and atom.GetFormalCharge() == 0
    ]
    if len(iodine_atoms) != 1:
        return ""
    iodine_idx = iodine_atoms[0].GetIdx()
    hydroxy_neighbors = [
        nbr
        for nbr in mol.GetAtomWithIdx(iodine_idx).GetNeighbors()
        if nbr.GetSymbol() == "O"
        and not nbr.IsInRing()
        and (bond := mol.GetBondBetweenAtoms(iodine_idx, nbr.GetIdx())) is not None
        and bond.GetBondType() == Chem.BondType.SINGLE
    ]
    if len(hydroxy_neighbors) != 1:
        return ""
    return "1-hydroxy-1,2-benziodoxol-3-one"


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


def _benziodoxole_oxo_substituted_hydride_name(mol, Chem, match: Sequence[int]) -> str:
    iodine_idx = match[1]
    substituent_idx = match[2]
    covered = set(match)
    covered.update(_collect_component_atoms(mol, substituent_idx, iodine_idx))
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    prefix = _simple_or_branch_prefix(mol, substituent_idx, iodine_idx)
    if not prefix:
        return ""
    prefix = _parenthesized_if_complex(_strip_outer_parentheses(prefix))
    return f"1-oxo-1-{prefix}-1,2-benziodoxole"


def _benziodoxole_oxo_ring_substituted_hydride_name(mol, Chem, match: Sequence[int]) -> str:
    substituent_idx = match[0]
    ring_carbon_idx = match[1]
    covered = set(match)
    covered.update(_collect_component_atoms(mol, substituent_idx, ring_carbon_idx))
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    prefix = _simple_or_branch_prefix(mol, substituent_idx, ring_carbon_idx)
    if not prefix:
        return ""
    prefix = _parenthesized_if_complex(_strip_outer_parentheses(prefix))
    return f"3-{prefix}-1-oxo-1,2-benziodoxole"


def _benziodoxole_substituted_hydride_name(mol, Chem, match: Sequence[int]) -> str:
    substituent_idx = match[0]
    iodine_idx = match[1]
    iodine = mol.GetAtomWithIdx(iodine_idx)
    for nbr in iodine.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == substituent_idx or nbr.IsInRing():
            continue
        bond = mol.GetBondBetweenAtoms(iodine_idx, nidx)
        if bond is not None and nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            return ""
    covered = set(match)
    covered.update(_collect_component_atoms(mol, substituent_idx, iodine_idx))
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    prefix = _simple_or_branch_prefix(mol, substituent_idx, iodine_idx)
    if not prefix:
        return ""
    prefix = _parenthesized_if_complex(_strip_outer_parentheses(prefix))
    return f"1-{prefix}-1,2-benziodoxole"


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
            implicit_hydrogens = iodine.GetTotalNumHs()
            if len(neighbors) + implicit_hydrogens != 3 or not 1 <= len(neighbors) <= 3:
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
    if not 1 <= len(ligand_names) <= 3:
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
            if iodine.GetSymbol() != "I" or iodine.GetFormalCharge() != 1 or iodine.IsInRing():
                continue
            iodine_idx = iodine.GetIdx()
            implicit_hydrogens = iodine.GetTotalNumHs()
            ligand_neighbors = []
            for nbr in iodine.GetNeighbors():
                if nbr.GetAtomicNum() <= 1:
                    continue
                bond = mol.GetBondBetweenAtoms(iodine_idx, nbr.GetIdx())
                if (
                    nbr.GetSymbol() not in {"C", "O"}
                    or bond is None
                    or bond.GetBondType() != self._Chem.BondType.SINGLE
                    or (nbr.GetSymbol() == "O" and nbr.GetFormalCharge() != 0)
                ):
                    break
                ligand_neighbors.append(nbr.GetIdx())
            else:
                ligand_count = len(ligand_neighbors)
                if not (
                    (ligand_count + implicit_hydrogens == 2 and ligand_count in {1, 2})
                    or (ligand_count == 3 and implicit_hydrogens == 1)
                ):
                    continue
                ligand_names = []
                covered = {iodine_idx}
                for nidx in ligand_neighbors:
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
    if not 1 <= len(ligand_names) <= 3:
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


class IodineOxoParentRule:
    family = "iodine-oxo-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_set = set(range(mol.GetNumAtoms()))
        for iodine in mol.GetAtoms():
            if iodine.GetSymbol() != "I" or iodine.IsInRing() or iodine.GetTotalNumHs() != 0:
                continue
            charge = iodine.GetFormalCharge()
            if charge not in {1, 2, 3}:
                continue
            iodine_idx = iodine.GetIdx()
            oxido_count = 0
            ligand_oxygen_idx = None
            covered = {iodine_idx}
            ok = True
            for nbr in iodine.GetNeighbors():
                nidx = nbr.GetIdx()
                bond = mol.GetBondBetweenAtoms(iodine_idx, nidx)
                if nbr.GetSymbol() != "O" or bond is None or bond.GetBondType() != self._Chem.BondType.SINGLE:
                    ok = False
                    break
                heavy = [atom.GetIdx() for atom in nbr.GetNeighbors() if atom.GetIdx() != iodine_idx and atom.GetAtomicNum() > 1]
                if nbr.GetFormalCharge() < 0 and not heavy:
                    oxido_count += 1
                    covered.add(nidx)
                elif nbr.GetFormalCharge() == 0 and len(heavy) == 1 and ligand_oxygen_idx is None:
                    ligand_oxygen_idx = nidx
                    covered.update(_collect_component_atoms(mol, nidx, iodine_idx))
                else:
                    ok = False
                    break
            if not ok or ligand_oxygen_idx is None or covered != atom_set:
                continue
            if oxido_count != charge:
                continue
            owner = _single_external_substituent_name(mol, ligand_oxygen_idx, iodine_idx)
            if not _safe_iodine_oxo_owner(owner):
                continue
            parent = {1: "iodite", 2: "iodate", 3: "periodate"}[charge]
            name = f"{_strip_outer_parentheses(owner)} {parent}"
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


def _safe_iodine_oxo_owner(owner: str) -> bool:
    owner = _strip_outer_parentheses(owner)
    if not owner or len(owner) > 96 or " " in owner or "[" in owner or "]" in owner:
        return False
    if any(fragment in owner for fragment in ("lambda", "iod", "period")):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9,().'\-]+", owner))


class ChargedRetainedParentRule:
    family = "charged-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen = set()
        for name in _charged_retained_parent_names(mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(mol.GetNumAtoms())),
                sort_key=(len(name), name),
            )


def _charged_retained_parent_names(mol, Chem) -> list[str]:
    names = []
    names.extend(_alcohol_oxide_retained_parent_names(mol, Chem))
    names.extend(_carboxylic_acid_chalcogenide_retained_parent_names(mol, Chem))
    names.extend(_organic_diazenide_retained_parent_names(mol, Chem))
    names.extend(_exocyclic_imine_azanide_retained_parent_names(mol, Chem))
    names.extend(_bis_azanidyl_alkane_retained_parent_names(mol, Chem))
    names.extend(_anionic_amide_retained_parent_names(mol, Chem))
    names.extend(_amide_oxide_retained_parent_names(mol, Chem))
    names.extend(_sulfur_oxoanion_retained_parent_names(mol, Chem))
    names.extend(_oxonium_retained_parent_names(mol, Chem))
    names.extend(_methylium_retained_parent_names(mol, Chem))
    names.extend(_cyano_methane_ion_parent_names(mol, Chem))
    names.extend(_isocyanide_retained_parent_names(mol, Chem))
    names.extend(_nitrone_azaniumolate_parent_names(mol, Chem))
    names.extend(_n_substituted_imine_oxide_retained_parent_names(mol, Chem))
    names.extend(_hydroxylamine_oxide_retained_parent_names(mol, Chem))
    names.extend(_amine_oxide_retained_parent_names(mol, Chem))
    return names


def _alcohol_oxide_retained_parent_names(mol, Chem) -> list[str]:
    names: list[str] = []
    atom_set = _heavy_atom_indices(mol)
    for oxygen in mol.GetAtoms():
        if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 1 or oxygen.IsInRing():
            continue
        oidx = oxygen.GetIdx()
        if oxygen.GetTotalNumHs() != 1:
            continue
        carbon_neighbors = []
        oxido_neighbors = []
        for nbr in oxygen.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(oidx, nidx)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                break
            if nbr.GetSymbol() == "C" and nbr.GetFormalCharge() == 0 and not nbr.IsInRing():
                carbon_neighbors.append(nidx)
            elif (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() == -1
                and not nbr.IsInRing()
                and len([atom for atom in nbr.GetNeighbors() if atom.GetAtomicNum() > 1]) == 1
            ):
                oxido_neighbors.append(nidx)
            else:
                break
        else:
            if len(carbon_neighbors) != 1 or len(oxido_neighbors) != 1:
                continue
            carbon_idx = carbon_neighbors[0]
            oxido_idx = oxido_neighbors[0]
            for path in _terminal_alcohol_oxide_parent_paths(mol, Chem, carbon_idx, oidx):
                covered = _terminal_alcohol_oxide_path_covered_atoms(mol, path, oidx, oxido_idx)
                if covered != atom_set:
                    continue
                name = _terminal_alcohol_oxide_path_name(mol, path, oidx, oxido_idx)
                if name:
                    names.append(name)
    return sorted(set(names), key=lambda name: (len(name), name))


def _terminal_alcohol_oxide_parent_paths(mol, Chem, carbon_idx: int, oxygen_idx: int) -> list[tuple[int, ...]]:
    atoms = _collect_component_atoms(mol, carbon_idx, oxygen_idx)
    allowed = {
        idx
        for idx in atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == "C"
        and mol.GetAtomWithIdx(idx).GetFormalCharge() == 0
        and not mol.GetAtomWithIdx(idx).IsInRing()
        and not _is_nitrile_carbon(mol, idx)
    }
    if carbon_idx not in allowed:
        return []
    leaves = [
        idx
        for idx in allowed
        if idx == carbon_idx
        or sum(1 for nbr in mol.GetAtomWithIdx(idx).GetNeighbors() if nbr.GetIdx() in allowed) <= 1
    ]
    paths: set[tuple[int, ...]] = set()
    best_len = 0
    for leaf in leaves:
        path = (carbon_idx,) if leaf == carbon_idx else tuple(Chem.rdmolops.GetShortestPath(mol, carbon_idx, leaf))
        if any(idx not in allowed for idx in path):
            continue
        if any(
            mol.GetBondBetweenAtoms(a, b) is None
            or mol.GetBondBetweenAtoms(a, b).GetBondType() != Chem.BondType.SINGLE
            for a, b in zip(path, path[1:])
        ):
            continue
        if len(path) > best_len:
            paths.clear()
            best_len = len(path)
        if len(path) == best_len:
            paths.add(path)
    return sorted(paths)


def _terminal_alcohol_oxide_path_covered_atoms(
    mol,
    path: Sequence[int],
    oxygen_idx: int,
    oxido_idx: int,
) -> set[int]:
    covered = {oxygen_idx, oxido_idx, *path}
    path_set = set(path)
    skip_atoms = {oxygen_idx, oxido_idx}
    for atom_idx in path:
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in path_set or nidx in skip_atoms or nbr.GetAtomicNum() <= 1:
                continue
            branch_atoms = _collect_component_atoms_excluding(mol, nidx, {atom_idx, *path_set, *skip_atoms})
            if any(mol.GetAtomWithIdx(branch_idx).IsInRing() for branch_idx in branch_atoms):
                return set()
            if any(mol.GetAtomWithIdx(branch_idx).GetFormalCharge() != 0 for branch_idx in branch_atoms):
                return set()
            covered.update(branch_atoms)
    return covered


def _terminal_alcohol_oxide_path_name(
    mol,
    path: Sequence[int],
    oxygen_idx: int,
    oxido_idx: int,
) -> str:
    parent = parent_chain_name(mol, path, as_substituent=False)
    if parent.endswith("ane"):
        stem = parent[:-1]
    elif parent.endswith("an"):
        stem = parent
    else:
        return ""
    prefixes = _path_substituent_prefixes(mol, path, skip_atoms={oxygen_idx, oxido_idx}, skip_plain_hydrogen=True)
    if prefixes is None:
        return ""
    core = stem + "-1-ol oxide"
    prefix_text = _format_prefixes(prefixes)
    return _join_prefix_parent(prefix_text, core) if prefix_text else core


def _carboxylic_acid_chalcogenide_retained_parent_names(mol, Chem) -> list[str]:
    names: list[str] = []
    atom_set = _heavy_atom_indices(mol)
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
            continue
        cidx = carbon.GetIdx()
        acyl_carbons: list[int] = []
        single_oxygen = None
        parent_suffix = ""
        oxonium_oxygen = None
        ok = True
        for nbr in carbon.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if bond is None:
                ok = False
                break
            if nbr.GetSymbol() == "C" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.SINGLE:
                acyl_carbons.append(nidx)
            elif (
                nbr.GetSymbol() == "O"
                and bond.GetBondType() == Chem.BondType.SINGLE
                and len([atom for atom in nbr.GetNeighbors() if atom.GetAtomicNum() > 1]) == 1
            ):
                if single_oxygen is not None:
                    ok = False
                    break
                if nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                    parent_suffix = "acid"
                elif nbr.GetFormalCharge() == -1 and nbr.GetTotalNumHs() == 0:
                    parent_suffix = "oate"
                else:
                    ok = False
                    break
                single_oxygen = nidx
            elif (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() == 1
                and bond.GetBondType() == Chem.BondType.DOUBLE
            ):
                if oxonium_oxygen is not None:
                    ok = False
                    break
                oxonium_oxygen = nidx
            else:
                ok = False
                break
        if not ok or single_oxygen is None or oxonium_oxygen is None or len(acyl_carbons) > 1:
            continue
        terminal = _acid_chalcogenide_terminal_atom(mol, Chem, oxonium_oxygen)
        if terminal is None:
            continue
        terminal_idx, suffix = terminal
        if not acyl_carbons:
            if {cidx, single_oxygen, oxonium_oxygen, terminal_idx} == atom_set:
                parent = "methanoic acid" if parent_suffix == "acid" else "formate"
                names.append(f"{parent} {suffix}")
            continue
        for path in _ester_acyl_parent_paths(mol, Chem, cidx, acyl_carbons[0]):
            if not path or path[0] != acyl_carbons[0]:
                continue
            covered = _acid_chalcogenide_path_covered_atoms(
                mol,
                cidx,
                path,
                single_oxygen,
                oxonium_oxygen,
                terminal_idx,
            )
            if covered != atom_set:
                continue
            parent = _generic_acyl_functional_parent_path_name(mol, Chem, cidx, path, parent_suffix)
            if parent:
                names.append(f"{parent} {suffix}")
    return sorted(set(names), key=lambda name: (len(name), name))


def _acid_chalcogenide_terminal_atom(mol, Chem, oxonium_idx: int) -> tuple[int, str] | None:
    terminals = []
    for nbr in mol.GetAtomWithIdx(oxonium_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(oxonium_idx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == -1 and not nbr.IsInRing():
            terminals.append((nidx, "oxide"))
        elif nbr.GetSymbol() == "S" and nbr.GetFormalCharge() == -1 and not nbr.IsInRing():
            terminals.append((nidx, "sulfide"))
    if len(terminals) != 1:
        return None
    terminal_idx, suffix = terminals[0]
    if len([atom for atom in mol.GetAtomWithIdx(terminal_idx).GetNeighbors() if atom.GetAtomicNum() > 1]) != 1:
        return None
    return terminal_idx, suffix


def _acid_chalcogenide_path_covered_atoms(
    mol,
    carbonyl_idx: int,
    path: Sequence[int],
    hydroxy_oxygen: int,
    oxonium_oxygen: int,
    terminal_idx: int,
) -> set[int]:
    covered = {carbonyl_idx, hydroxy_oxygen, oxonium_oxygen, terminal_idx, *path}
    path_set = set(path)
    skip_atoms = {hydroxy_oxygen, oxonium_oxygen, terminal_idx}
    for pos, atom_idx in enumerate(path, start=2):
        allowed_neighbors = {carbonyl_idx if pos == 2 else path[pos - 3]}
        if pos - 1 < len(path):
            allowed_neighbors.add(path[pos - 1])
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in allowed_neighbors or nidx in path_set or nidx in skip_atoms or nbr.GetAtomicNum() <= 1:
                continue
            branch_atoms = _collect_component_atoms_excluding(mol, nidx, {atom_idx, *path_set, *skip_atoms})
            if any(mol.GetAtomWithIdx(branch_idx).IsInRing() for branch_idx in branch_atoms):
                return set()
            if any(mol.GetAtomWithIdx(branch_idx).GetFormalCharge() != 0 for branch_idx in branch_atoms):
                return set()
            covered.update(branch_atoms)
    return covered


def _organic_diazenide_retained_parent_names(mol, Chem) -> list[str]:
    if _formal_charge(mol) != -1:
        return []
    atom_set = set(range(mol.GetNumAtoms()))
    names: list[str] = []
    for terminal in mol.GetAtoms():
        if terminal.GetSymbol() != "N" or terminal.GetFormalCharge() != -1 or terminal.IsInRing():
            continue
        terminal_idx = terminal.GetIdx()
        heavy = [nbr.GetIdx() for nbr in terminal.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(heavy) != 1:
            continue
        linker_idx = heavy[0]
        linker = mol.GetAtomWithIdx(linker_idx)
        terminal_bond = mol.GetBondBetweenAtoms(terminal_idx, linker_idx)
        if (
            linker.GetSymbol() != "N"
            or linker.GetFormalCharge() != 0
            or linker.IsInRing()
            or terminal_bond is None
            or terminal_bond.GetBondType() != Chem.BondType.DOUBLE
        ):
            continue
        owners = [
            nbr.GetIdx()
            for nbr in linker.GetNeighbors()
            if nbr.GetIdx() != terminal_idx and nbr.GetAtomicNum() > 1
        ]
        if len(owners) != 1:
            continue
        owner_idx = owners[0]
        owner = mol.GetAtomWithIdx(owner_idx)
        owner_bond = mol.GetBondBetweenAtoms(linker_idx, owner_idx)
        if (
            owner.GetSymbol() != "C"
            or owner.GetFormalCharge() != 0
            or owner_bond is None
            or owner_bond.GetBondTypeAsDouble() > 1.1
        ):
            continue
        covered = {terminal_idx, linker_idx}
        covered.update(_collect_component_atoms(mol, owner_idx, linker_idx))
        if covered != atom_set:
            continue
        owner_name = _simple_or_branch_prefix(mol, owner_idx, linker_idx)
        if not _stable_diazenide_owner_name(owner_name):
            continue
        names.append(_strip_outer_parentheses(owner_name) + "diazenide")
    return names


def _stable_diazenide_owner_name(name: str) -> bool:
    if not name or len(name) > 120:
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    if any(fragment in name for fragment in ("azanid", "diazen", "phosph", "sulfon", "sulfin")):
        return False
    return name.endswith(("yl", "phenyl", "naphthyl")) or "phenyl" in name or "naphthalen" in name


def _exocyclic_imine_azanide_retained_parent_names(mol, Chem) -> list[str]:
    if _formal_charge(mol) != -1:
        return []
    names: list[str] = []
    atom_set = set(range(mol.GetNumAtoms()))
    for terminal in mol.GetAtoms():
        if terminal.GetSymbol() != "N" or terminal.GetFormalCharge() != -1 or terminal.IsInRing():
            continue
        terminal_idx = terminal.GetIdx()
        heavy = [nbr.GetIdx() for nbr in terminal.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(heavy) != 1:
            continue
        carbon_idx = heavy[0]
        carbon = mol.GetAtomWithIdx(carbon_idx)
        imine_bond = mol.GetBondBetweenAtoms(terminal_idx, carbon_idx)
        if (
            carbon.GetSymbol() != "C"
            or carbon.GetFormalCharge() != 0
            or not carbon.IsInRing()
            or imine_bond is None
            or imine_bond.GetBondType() != Chem.BondType.DOUBLE
        ):
            continue
        covered = {terminal_idx}
        covered.update(_collect_component_atoms(mol, carbon_idx, terminal_idx))
        if covered != atom_set:
            continue
        ylidene = _ring_substituent_prefix(mol, carbon_idx, terminal_idx)
        if _stable_azanide_ylidene_name(ylidene):
            names.append(_strip_outer_parentheses(ylidene) + "azanide")
        for name in _saturated_ring_azanidylidene_names(mol, Chem, carbon_idx, terminal_idx):
            names.append(name)
    return names


def _stable_azanide_ylidene_name(name: str) -> bool:
    if not name or len(name) > 140 or not name.endswith("ylidene"):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    if any(fragment in name for fragment in ("azanid", "diazen", "phosph", "sulfon", "sulfin")):
        return False
    return True


def _saturated_ring_azanidylidene_names(mol, Chem, carbon_idx: int, terminal_idx: int) -> list[str]:
    names: list[str] = []
    for ring in mol.GetRingInfo().AtomRings():
        ring_set = set(ring)
        if carbon_idx not in ring_set or ring_set | {terminal_idx} != set(range(mol.GetNumAtoms())):
            continue
        if not _ring_is_saturated_after_exocyclic_imine(mol, Chem, ring, carbon_idx, terminal_idx):
            continue
        symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring]
        if all(symbol == "C" for symbol in symbols):
            root = ROOTS.get(len(ring))
            if root:
                names.append(f"1-azanidylidenecyclo{root}ane")
            continue
        records = []
        for parent_name, seq in _fixed_saturated_heterocycle_candidates(mol, tuple(ring), symbols):
            if carbon_idx not in seq:
                continue
            locant = seq.index(carbon_idx) + 1
            if locant <= 0:
                continue
            records.append((locant, parent_name))
        for locant, parent_name in sorted(set(records)):
            names.append(f"{locant}-azanidylidene{parent_name}")
    return names


def _ring_is_saturated_after_exocyclic_imine(
    mol,
    Chem,
    ring: Sequence[int],
    carbon_idx: int,
    terminal_idx: int,
) -> bool:
    ring_set = set(ring)
    for a, b in zip(ring, ring[1:] + ring[:1]):
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return False
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() != 0 or atom.GetIsotope():
            return False
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set:
                continue
            if idx == carbon_idx and nidx == terminal_idx:
                bond = mol.GetBondBetweenAtoms(idx, nidx)
                if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                    return False
                continue
            if nbr.GetAtomicNum() > 1:
                return False
    return True


def _isocyanide_retained_parent_names(mol, Chem) -> list[str]:
    motifs = _isocyanide_motifs(mol, Chem)
    if len(motifs) != 1:
        return []
    carbon_idx, nitrogen_idx, owner_idx = motifs[0]
    covered = {carbon_idx, nitrogen_idx}
    covered.update(_collect_component_atoms(mol, owner_idx, nitrogen_idx))
    if covered != set(range(mol.GetNumAtoms())):
        return []
    owner = mol.GetAtomWithIdx(owner_idx)
    if owner.GetFormalCharge() != 0:
        return []
    if owner.GetSymbol() == "O":
        ester_names = _isocyanide_o_acyl_ester_names(mol, Chem, owner_idx, nitrogen_idx)
        if ester_names:
            return ester_names
        owner_prefix = _isocyanide_terminal_chalcogen_prefix(mol, owner_idx, nitrogen_idx)
    elif owner.GetSymbol() == "S":
        owner_prefix = _isocyanide_terminal_chalcogen_prefix(mol, owner_idx, nitrogen_idx)
        if not owner_prefix:
            owner_prefix = _isocyanide_owner_prefix(mol, owner_idx, nitrogen_idx)
    else:
        owner_prefix = _isocyanide_owner_prefix(mol, owner_idx, nitrogen_idx)
    if not owner_prefix:
        return []
    return [f"{owner_prefix} isocyanide"]


def _isocyanide_motifs(mol, Chem) -> list[tuple[int, int, int]]:
    motifs = []
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != -1:
            continue
        carbon_idx = carbon.GetIdx()
        carbon_heavy = [nbr.GetIdx() for nbr in carbon.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(carbon_heavy) != 1:
            continue
        nitrogen_idx = carbon_heavy[0]
        nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 1:
            continue
        cn_bond = mol.GetBondBetweenAtoms(carbon_idx, nitrogen_idx)
        if cn_bond is None or cn_bond.GetBondType() != Chem.BondType.TRIPLE:
            continue
        owners = [
            nbr.GetIdx()
            for nbr in nitrogen.GetNeighbors()
            if nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1
        ]
        if len(owners) != 1:
            continue
        no_bond = mol.GetBondBetweenAtoms(nitrogen_idx, owners[0])
        if no_bond is None or no_bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        motifs.append((carbon_idx, nitrogen_idx, owners[0]))
    return motifs


def _isocyanide_o_acyl_ester_names(mol, Chem, oxygen_idx: int, nitrogen_idx: int) -> list[str]:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0:
        return []
    heavy = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != nitrogen_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return []
    carbonyl_idx = heavy[0]
    bond = mol.GetBondBetweenAtoms(oxygen_idx, carbonyl_idx)
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return []
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0:
        return []
    double_oxygen = []
    acyl_carbons = []
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == oxygen_idx or nbr.GetAtomicNum() <= 1:
            continue
        neighbor_bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and neighbor_bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen.append(nidx)
        elif nbr.GetSymbol() == "C" and neighbor_bond.GetBondType() == Chem.BondType.SINGLE:
            acyl_carbons.append(nidx)
        else:
            return []
    if len(double_oxygen) != 1 or len(acyl_carbons) != 1:
        return []
    return _generic_acyl_ester_names(mol, Chem, carbonyl_idx, acyl_carbons[0], "isocyano")


def _isocyanide_terminal_chalcogen_prefix(mol, atom_idx: int, nitrogen_idx: int) -> str:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetFormalCharge() != 0:
        return ""
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != nitrogen_idx and nbr.GetAtomicNum() > 1]
    if heavy:
        return ""
    if atom.GetSymbol() == "O":
        return "hydroxy"
    if atom.GetSymbol() == "S":
        return "sulfanyl"
    return ""


def _isocyanide_owner_prefix(mol, owner_idx: int, nitrogen_idx: int) -> str:
    name = _simple_or_branch_prefix(mol, owner_idx, nitrogen_idx)
    name = _retained_parent_substituent_alias(name) if name else ""
    return name if _stable_isocyanide_owner_prefix(name) else ""


def _stable_isocyanide_owner_prefix(name: str) -> bool:
    if not name or len(name) > 100:
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    if any(fragment in name for fragment in ("phosph", "azonia", "isocyano", "fulmin")):
        return False
    if "ylidene" in name:
        return False
    if name in {"amino", "hydroxy", "sulfanyl", "isothiocyanato", *HALO.values()}:
        return True
    return name.endswith(("yl", "oyl", "carbonyl", "sulfonyl"))


def _bis_azanidyl_alkane_retained_parent_names(mol, Chem) -> list[str]:
    nitrogens = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "N" and atom.GetFormalCharge() == -1 and not atom.IsInRing()
    ]
    if len(nitrogens) != 2 or sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != -2:
        return []
    first_n, second_n = nitrogens
    path = tuple(Chem.rdmolops.GetShortestPath(mol, first_n, second_n))
    if not path or path[0] != first_n or path[-1] != second_n:
        return []
    linker = path[1:-1]
    if len(linker) < 2 or not _is_unsubstituted_saturated_carbon_path(mol, Chem, linker, first_n, second_n):
        return []
    parent = _alkane_parent_name(len(linker))
    if not parent:
        return []
    substituents = []
    covered = set(linker)
    for nidx, linker_idx in ((first_n, path[1]), (second_n, path[-2])):
        substituent = _bis_azanidyl_external_substituent(mol, nidx, linker_idx)
        if not substituent:
            return []
        substituents.append(substituent)
        external_idx = _single_heavy_neighbor_excluding(mol, nidx, linker_idx)
        if external_idx is None:
            return []
        covered.add(nidx)
        covered.update(_collect_component_atoms(mol, external_idx, nidx))
    if covered != set(range(mol.GetNumAtoms())):
        return []
    if substituents[0] != substituents[1]:
        return []
    piece = _azanidyl_substituent_piece(substituents[0])
    return [f"1,{len(linker)}-bis({piece}azanidyl){parent}"]


def _alkane_parent_name(length: int) -> str:
    root = ROOTS.get(length)
    return f"{root}ane" if root else ""


def _bis_azanidyl_external_substituent(mol, nitrogen_idx: int, linker_idx: int) -> str:
    external_idx = _single_heavy_neighbor_excluding(mol, nitrogen_idx, linker_idx)
    if external_idx is None:
        return ""
    external = mol.GetAtomWithIdx(external_idx)
    if external.GetSymbol() != "C" or external.GetFormalCharge() != 0:
        return ""
    prefix = _retained_parent_substituent_alias(_simple_or_branch_prefix(mol, external_idx, nitrogen_idx))
    if not _stable_bis_azanidyl_substituent(prefix):
        return ""
    return _strip_outer_parentheses(prefix)


def _stable_bis_azanidyl_substituent(name: str) -> bool:
    if not name or len(name) > 96:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "carbam", "urea", "azanid")):
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    return name.endswith(("yl", "carbonyl"))


def _azanidyl_substituent_piece(name: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9']+", name):
        return name
    return f"({name})"


def _amide_oxide_retained_parent_names(mol, Chem) -> list[str]:
    names: list[str] = []
    atom_set = set(range(mol.GetNumAtoms()))
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
            continue
        cidx = carbon.GetIdx()
        double_oxygen = []
        acyl_carbons = []
        amide_nitrogens = []
        ok = True
        for nbr in carbon.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygen.append(nidx)
            elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                acyl_carbons.append(nidx)
            elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
                amide_nitrogens.append(nidx)
            else:
                ok = False
                break
        if not ok or len(double_oxygen) != 1 or len(acyl_carbons) != 1 or len(amide_nitrogens) != 1:
            continue
        nitrogen_idx = amide_nitrogens[0]
        oxido_idx = _primary_amide_oxide_oxygen(mol, Chem, nitrogen_idx, cidx)
        if oxido_idx is None:
            continue
        acyl_idx = acyl_carbons[0]
        covered = {cidx, double_oxygen[0], nitrogen_idx, oxido_idx}
        covered.update(_collect_component_atoms(mol, acyl_idx, cidx))
        if covered != atom_set:
            continue
        for path in _ester_acyl_parent_paths(mol, Chem, cidx, acyl_idx):
            core = _generic_acyl_functional_parent_path_name(mol, Chem, cidx, path, "amide")
            if core:
                names.append(f"{core} oxide")
    return names


def _primary_amide_oxide_oxygen(mol, Chem, nitrogen_idx: int, carbonyl_idx: int) -> int | None:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 1 or nitrogen.IsInRing():
        return None
    heavy_neighbors = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors() if nbr.GetAtomicNum() > 1]
    if len(heavy_neighbors) != 2 or carbonyl_idx not in heavy_neighbors:
        return None
    if nitrogen.GetTotalNumHs() != 2:
        return None
    oxido_candidates = [idx for idx in heavy_neighbors if idx != carbonyl_idx]
    if len(oxido_candidates) != 1:
        return None
    oxido_idx = oxido_candidates[0]
    oxido = mol.GetAtomWithIdx(oxido_idx)
    if oxido.GetSymbol() != "O" or oxido.GetFormalCharge() != -1 or oxido.IsInRing():
        return None
    if len([nbr for nbr in oxido.GetNeighbors() if nbr.GetAtomicNum() > 1]) != 1:
        return None
    bond = mol.GetBondBetweenAtoms(nitrogen_idx, oxido_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    return oxido_idx


def _n_substituted_imine_oxide_retained_parent_names(mol, Chem) -> list[str]:
    if assigned_atom_stereochemistry(mol, Chem):
        return []
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for nitrogen in mol.GetAtoms():
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() <= 0 or nitrogen.IsInRing():
            continue
        record = _n_substituted_imine_oxide_record(mol, Chem, nitrogen.GetIdx())
        if not record or record["covered_atoms"] != atom_set:
            continue
        name = _n_substituted_imine_oxide_name(mol, record)
        if name:
            names.append(name)
    return names


def _n_substituted_imine_oxide_record(mol, Chem, nitrogen_idx: int) -> dict | None:
    imine_carbons = []
    oxido_oxygens = []
    n_substituents = []
    covered = {nitrogen_idx}
    for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if bond is None:
            return None
        heavy = [
            atom.GetIdx()
            for atom in nbr.GetNeighbors()
            if atom.GetIdx() != nitrogen_idx and atom.GetAtomicNum() > 1
        ]
        if (
            nbr.GetSymbol() == "C"
            and nbr.GetFormalCharge() == 0
            and bond.GetBondType() == Chem.BondType.DOUBLE
        ):
            imine_carbons.append(nidx)
            continue
        if (
            nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() < 0
            and bond.GetBondType() == Chem.BondType.SINGLE
            and not heavy
        ):
            oxido_oxygens.append(nidx)
            covered.add(nidx)
            continue
        if (
            nbr.GetSymbol() == "C"
            and nbr.GetFormalCharge() == 0
            and bond.GetBondType() == Chem.BondType.SINGLE
        ):
            substituent = _simple_or_branch_prefix(mol, nidx, nitrogen_idx)
            if not substituent or _unstable_retained_functional_parent_substituent(substituent):
                return None
            n_substituents.append(_strip_outer_parentheses(substituent))
            covered.update(_collect_component_atoms(mol, nidx, nitrogen_idx))
            continue
        return None
    if len(imine_carbons) != 1 or len(oxido_oxygens) != 1 or len(n_substituents) != 1:
        return None
    imine_idx = imine_carbons[0]
    imine = mol.GetAtomWithIdx(imine_idx)
    if imine.IsInRing() or imine.GetFormalCharge() != 0:
        return None
    carbon_neighbors = [
        nbr.GetIdx()
        for nbr in imine.GetNeighbors()
        if nbr.GetIdx() != nitrogen_idx and nbr.GetAtomicNum() > 1
    ]
    if len(carbon_neighbors) > 1:
        return None
    imine_bond = mol.GetBondBetweenAtoms(nitrogen_idx, imine_idx)
    stereo = _bond_stereo_descriptor(Chem, imine_bond)
    if _has_nonaromatic_carbon_double_bond_except(mol, Chem, imine_bond.GetIdx()):
        return None
    if carbon_neighbors:
        if not stereo:
            return None
        path = _terminal_imine_oxide_parent_path(mol, Chem, imine_idx, nitrogen_idx)
        if not path:
            return None
        prefixes = _path_substituent_prefixes(
            mol,
            path,
            skip_bonds={_bond_key(nitrogen_idx, imine_idx)},
            skip_plain_hydrogen=True,
        )
        if prefixes is None:
            return None
    else:
        path = (imine_idx,)
        prefixes = ()
    path_set = set(path)
    covered.update(path_set)
    for atom_idx in path:
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in path_set or nidx == nitrogen_idx or nbr.GetAtomicNum() <= 1:
                continue
            covered.update(_collect_component_atoms(mol, nidx, atom_idx))
    return {
        "path": tuple(path),
        "prefixes": tuple(prefixes),
        "n_substituent": n_substituents[0],
        "stereo": stereo,
        "covered_atoms": covered,
    }


def _bond_stereo_descriptor(Chem, bond) -> str:
    stereo = bond.GetStereo()
    if stereo in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}:
        return "E"
    if stereo in {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}:
        return "Z"
    return ""


def _has_nonaromatic_carbon_double_bond_except(mol, Chem, allowed_bond_idx: int) -> bool:
    return any(
        bond.GetIdx() != allowed_bond_idx
        and bond.GetBondType() == Chem.BondType.DOUBLE
        and not bond.GetIsAromatic()
        and mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol() == "C"
        and mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol() == "C"
        for bond in mol.GetBonds()
    )


def _terminal_imine_oxide_parent_path(mol, Chem, imine_idx: int, nitrogen_idx: int) -> tuple[int, ...]:
    allowed = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C"
        and atom.GetFormalCharge() == 0
        and not atom.IsInRing()
        and not atom.GetIsotope()
    }
    if imine_idx not in allowed:
        return ()
    adjacency = {
        idx: tuple(
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
            if nbr.GetIdx() in allowed
            and nbr.GetIdx() != nitrogen_idx
            and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
        )
        for idx in allowed
    }
    endpoints = [
        idx
        for idx in allowed
        if idx == imine_idx or sum(1 for nidx in adjacency[idx] if nidx in allowed) <= 1
    ]
    paths = []
    best_len = 0
    for endpoint in endpoints:
        path = _shortest_carbon_path(adjacency, imine_idx, endpoint)
        if not path:
            continue
        if len(path) > best_len:
            paths.clear()
            best_len = len(path)
        if len(path) == best_len:
            paths.append(path)
    return sorted(set(paths), key=lambda path: (-len(path), path))[0] if paths else ()


def _shortest_carbon_path(adjacency: dict[int, tuple[int, ...]], start: int, end: int) -> tuple[int, ...]:
    queue = [(start, (start,))]
    seen = {start}
    for current, path in queue:
        if current == end:
            return path
        for neighbor in adjacency[current]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, (*path, neighbor)))
    return ()


def _n_substituted_imine_oxide_name(mol, record: dict) -> str:
    parent = parent_chain_name(mol, record["path"], as_substituent=False)
    if not parent:
        return ""
    parent = _standalone_saturated_chain_parent(parent)
    prefixes = [("N", record["n_substituent"]), *record["prefixes"]]
    prefix_text = _format_prefixes(prefixes)
    name = _join_prefix_parent(prefix_text, f"{parent}-1-imine oxide")
    return f"({record['stereo']})-{name}" if record["stereo"] else name


def _amine_oxide_retained_parent_names(mol, Chem) -> list[str]:
    if _has_explicit_alkene_stereo(mol, Chem):
        return []
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for nitrogen in mol.GetAtoms():
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() <= 0 or nitrogen.IsInRing():
            continue
        record = _amine_oxide_record(mol, Chem, nitrogen.GetIdx())
        if not record or record["covered_atoms"] != atom_set:
            continue
        name = _amine_oxide_ligand_text(record["ligand_names"])
        if name:
            names.append(name)
    return names


def _amine_oxide_record(mol, Chem, nitrogen_idx: int) -> dict | None:
    oxido_oxygens = []
    ligand_names = []
    covered = {nitrogen_idx}
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if bond is None:
            return None
        heavy = [
            atom.GetIdx()
            for atom in nbr.GetNeighbors()
            if atom.GetIdx() != nitrogen_idx and atom.GetAtomicNum() > 1
        ]
        if (
            nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() < 0
            and bond.GetBondType() == Chem.BondType.SINGLE
            and not heavy
        ):
            oxido_oxygens.append(nidx)
            covered.add(nidx)
            continue
        if (
            nbr.GetSymbol() != "C"
            or nbr.GetFormalCharge() != 0
            or bond.GetBondType() != Chem.BondType.SINGLE
            or _has_nonaromatic_multiring_component(mol, nidx, nitrogen_idx)
        ):
            return None
        ligand = _simple_or_branch_prefix(mol, nidx, nitrogen_idx)
        if not ligand or _unstable_retained_functional_parent_substituent(ligand):
            return None
        ligand_names.append(_strip_outer_parentheses(ligand))
        covered.update(_collect_component_atoms(mol, nidx, nitrogen_idx))
    ligand_count = len(ligand_names)
    hydrogen_count = nitrogen.GetTotalNumHs()
    if len(oxido_oxygens) != 1:
        return None
    if not (
        (ligand_count == 3 and hydrogen_count == 0)
        or (ligand_count == 2 and hydrogen_count == 1)
    ):
        return None
    return {"ligand_names": tuple(ligand_names), "covered_atoms": covered}


def _hydroxylamine_oxide_retained_parent_names(mol, Chem) -> list[str]:
    if _has_explicit_alkene_stereo(mol, Chem):
        return []
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for nitrogen in mol.GetAtoms():
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() <= 0 or nitrogen.IsInRing():
            continue
        record = _hydroxylamine_oxide_record(mol, Chem, nitrogen.GetIdx())
        if not record or record["covered_atoms"] != atom_set:
            continue
        name = _hydroxylamine_oxide_name(record["carbon_substituents"], record["hydroxy_count"])
        if name:
            names.append(name)
    return names


def _hydroxylamine_oxide_record(mol, Chem, nitrogen_idx: int) -> dict | None:
    oxido_oxygens = []
    hydroxy_oxygens = []
    carbon_substituents = []
    covered = {nitrogen_idx}
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        heavy = [
            atom.GetIdx()
            for atom in nbr.GetNeighbors()
            if atom.GetIdx() != nitrogen_idx and atom.GetAtomicNum() > 1
        ]
        if nbr.GetSymbol() == "O" and not heavy:
            if nbr.GetFormalCharge() < 0:
                oxido_oxygens.append(nidx)
                covered.add(nidx)
                continue
            if nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() > 0:
                hydroxy_oxygens.append(nidx)
                covered.add(nidx)
                continue
        if (
            nbr.GetSymbol() != "C"
            or nbr.GetFormalCharge() != 0
            or _has_nonaromatic_multiring_component(mol, nidx, nitrogen_idx)
        ):
            return None
        substituent = _simple_or_branch_prefix(mol, nidx, nitrogen_idx)
        substituent = _retained_parent_substituent_alias(substituent)
        if not substituent or not _allowed_hydroxylamine_oxide_substituent(substituent):
            return None
        carbon_substituents.append(_strip_outer_parentheses(substituent))
        covered.update(_collect_component_atoms(mol, nidx, nitrogen_idx))
    hydroxy_count = len(hydroxy_oxygens)
    carbon_count = len(carbon_substituents)
    if len(oxido_oxygens) != 1:
        return None
    allowed_shape = (
        (hydroxy_count == 1 and carbon_count == 2 and nitrogen.GetTotalNumHs() == 0)
        or (hydroxy_count == 2 and carbon_count == 1 and nitrogen.GetTotalNumHs() == 0)
        or (hydroxy_count == 1 and carbon_count == 1 and nitrogen.GetTotalNumHs() == 1)
    )
    if not allowed_shape:
        return None
    return {
        "carbon_substituents": tuple(carbon_substituents),
        "covered_atoms": covered,
        "hydroxy_count": hydroxy_count,
    }


def _hydroxylamine_oxide_name(carbon_substituents: Sequence[str], hydroxy_count: int) -> str:
    if hydroxy_count == 1 and len(carbon_substituents) == 2:
        n_prefix = _hydroxylamine_oxide_n_substituent_prefix(carbon_substituents)
        if n_prefix is None:
            return ""
        return _join_n_prefix_functional_parent(n_prefix, "hydroxylamine oxide")
    if hydroxy_count == 2 and len(carbon_substituents) == 1:
        substituent = carbon_substituents[0]
        if not _allowed_hydroxylamine_oxide_substituent(substituent):
            return ""
        piece = substituent if _simple_unparenthesized_n_substituent(substituent) else f"({substituent})"
        return f"N,N-dihydroxy{piece}amine oxide"
    if hydroxy_count == 1 and len(carbon_substituents) == 1:
        substituent = carbon_substituents[0]
        if not _allowed_hydroxylamine_oxide_substituent(substituent):
            return ""
        piece = substituent if _simple_unparenthesized_n_substituent(substituent) else f"({substituent})"
        return f"N-hydroxy{piece}amine oxide"
    return ""


def _hydroxylamine_oxide_n_substituent_prefix(substituents: Sequence[str]) -> str | None:
    counts = Counter(_retained_parent_substituent_alias(name) for name in substituents)
    pieces = []
    for name in sorted(counts):
        if not _allowed_hydroxylamine_oxide_substituent(name):
            return None
        count = counts[name]
        if count == 1:
            pieces.append(_n_substituent_piece("N", name))
        elif count == 2:
            pieces.append(_duplicate_n_substituent_piece("N,N", name))
        else:
            return None
    return "-".join(pieces)


def _allowed_hydroxylamine_oxide_substituent(name: str) -> bool:
    if _simple_n_retained_parent_substituent(name) or _hydroxylamine_oxide_alkyl_substituent(name):
        return True
    if len(name) > 56:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "carbam", "urea", "imino")):
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    return any(fragment in name for fragment in ("cyclo", "phenyl", "benzyl"))


def _hydroxylamine_oxide_alkyl_substituent(name: str) -> bool:
    if _simple_linear_alkyl_substituent(name):
        return True
    for length, root in ROOTS.items():
        if name == f"{root}yl":
            return True
        if length > 1 and any(name == f"{root}an-{locant}-yl" for locant in range(2, length + 1)):
            return True
    return False


def _has_explicit_alkene_stereo(mol, Chem) -> bool:
    return any(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() not in {Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY}
        for bond in mol.GetBonds()
    )


def _has_nonaromatic_multiring_component(mol, start_idx: int, blocked_idx: int) -> bool:
    atoms = _collect_component_atoms(mol, start_idx, blocked_idx)
    rings = [ring for ring in mol.GetRingInfo().AtomRings() if any(idx in atoms for idx in ring)]
    if len(rings) <= 1:
        return False
    return any(not mol.GetAtomWithIdx(idx).GetIsAromatic() for ring in rings for idx in ring)


def _amine_oxide_ligand_text(ligand_names: Sequence[str]) -> str:
    if len(ligand_names) == 2:
        return _secondary_amine_oxide_ligand_text(ligand_names)
    counts = Counter(ligand_names)
    pieces = []
    for ligand, count in sorted(counts.items(), key=lambda item: item[0]):
        bare = _strip_outer_parentheses(ligand)
        if count == 1:
            pieces.append(_parenthesized_if_complex(bare))
            continue
        multiplier = MULT.get(count)
        if not multiplier:
            return ""
        if _needs_substituent_parentheses(bare):
            pieces.append(f"bis({bare})" if count == 2 else f"{multiplier}({bare})")
        else:
            pieces.append(multiplier + bare)
    return "".join(pieces) + "amine oxide" if pieces else ""


def _secondary_amine_oxide_ligand_text(ligand_names: Sequence[str]) -> str:
    left, right = sorted(_strip_outer_parentheses(name) for name in ligand_names)
    parent, substituent = _secondary_amine_oxide_parent_ligand(left, right)
    if not parent or not substituent:
        return ""
    if _simple_unparenthesized_n_substituent(substituent):
        substituent_text = substituent
    else:
        substituent_text = f"({substituent})"
    return f"N-{substituent_text}{parent}amine oxide"


def _secondary_amine_oxide_parent_ligand(left: str, right: str) -> tuple[str, str]:
    left_score = _secondary_amine_oxide_parent_score(left)
    right_score = _secondary_amine_oxide_parent_score(right)
    if right_score > left_score:
        return right, left
    return left, right


def _secondary_amine_oxide_parent_score(ligand: str) -> tuple[int, int, str]:
    simple_alkyl = _hydroxylamine_oxide_alkyl_substituent(ligand)
    cyclic_or_aromatic = any(fragment in ligand for fragment in ("cyclo", "phenyl", "benzyl", "ind"))
    return (
        1 if cyclic_or_aromatic else 0,
        0 if simple_alkyl else 1,
        len(ligand),
    )


def _anionic_amide_retained_parent_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for nitrogen in mol.GetAtoms():
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != -1 or nitrogen.IsInRing():
            continue
        nidx = nitrogen.GetIdx()
        heavy_neighbors = [nbr for nbr in nitrogen.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if not 1 <= len(heavy_neighbors) <= 2:
            continue
        substituents = []
        hetero_substituents = []
        n_hetero_substituents = []
        covered = {nidx}
        for nbr in heavy_neighbors:
            nbr_idx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(nidx, nbr_idx)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                break
            prefix = _simple_or_branch_prefix(mol, nbr_idx, nidx)
            if not prefix:
                break
            if nbr.GetSymbol() == "C":
                substituents.append(_retained_parent_substituent_alias(prefix))
            elif nbr.GetSymbol() in {"O", "S"}:
                hetero_substituents.append(_retained_parent_substituent_alias(prefix))
            elif nbr.GetSymbol() == "N":
                n_hetero = _anionic_amide_n_hetero_substituent(_retained_parent_substituent_alias(prefix))
                if not n_hetero:
                    break
                n_hetero_substituents.append(n_hetero)
            else:
                break
            covered.update(_collect_component_atoms(mol, nbr_idx, nidx))
        else:
            if covered != atom_set:
                continue
            if hetero_substituents and n_hetero_substituents:
                continue
            if hetero_substituents and substituents:
                name = _hetero_substituted_anionic_amide_parent_name(substituents, hetero_substituents)
            elif hetero_substituents:
                name = _single_hetero_substituted_anionic_amide_parent_name(
                    mol,
                    Chem,
                    hetero_substituents,
                )
            elif n_hetero_substituents and not substituents:
                name = _single_n_hetero_substituted_anionic_amide_parent_name(n_hetero_substituents)
            elif not n_hetero_substituents:
                name = _anionic_amide_parent_name(substituents)
            else:
                name = ""
            if name:
                names.append(name)
    return names


_ANIONIC_AMIDE_N_HETERO_SUBSTITUENT_ALIASES = {
    "aziridino": "aziridin-1-yl",
    "azetidino": "azetidin-1-yl",
    "pyrrolidino": "pyrrolidin-1-yl",
    "piperidino": "piperidin-1-yl",
    "piperazino": "piperazin-1-yl",
    "morpholino": "morpholin-4-yl",
    "thiomorpholino": "thiomorpholin-4-yl",
    "imidazolidino": "imidazolidin-1-yl",
    "pyrazolidino": "pyrazolidin-1-yl",
}


def _anionic_amide_n_hetero_substituent(name: str) -> str:
    alias = _ANIONIC_AMIDE_N_HETERO_SUBSTITUENT_ALIASES.get(name, "")
    if alias:
        return alias
    if _stable_anionic_amide_n_amino_substituent(name):
        return name
    return ""


def _stable_anionic_amide_n_amino_substituent(name: str) -> bool:
    if len(name) > 48:
        return False
    if not name.endswith("amino"):
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if " " in name or "[" in name or "]" in name or "(" in name or ")" in name:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9,\-]+", name))


def _single_hetero_substituted_anionic_amide_parent_name(
    mol,
    Chem,
    hetero_substituents: Sequence[str],
) -> str:
    if len(hetero_substituents) != 1:
        return ""
    hetero = hetero_substituents[0]
    if not _stable_anionic_amide_hetero_substituent(hetero):
        return ""
    if _has_explicit_alkene_stereo(mol, Chem):
        return ""
    return hetero + "amide"


def _single_n_hetero_substituted_anionic_amide_parent_name(
    n_hetero_substituents: Sequence[str],
) -> str:
    if len(n_hetero_substituents) != 1:
        return ""
    return n_hetero_substituents[0] + "amide"


def _hetero_substituted_anionic_amide_parent_name(
    carbon_substituents: Sequence[str],
    hetero_substituents: Sequence[str],
) -> str:
    if len(carbon_substituents) != 1 or len(hetero_substituents) != 1:
        return ""
    carbon = carbon_substituents[0]
    hetero = hetero_substituents[0]
    if not _allowed_n_retained_parent_substituent(carbon):
        return ""
    if not _stable_anionic_amide_hetero_substituent(hetero):
        return ""
    return f"{carbon} {hetero}amide"


def _stable_anionic_amide_hetero_substituent(name: str) -> bool:
    if len(name) > 72:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    return name.endswith(("oxy", "sulfanyl"))


def _anionic_amide_parent_name(substituents: Sequence[str]) -> str:
    if len(substituents) == 1:
        direct = _direct_anionic_amide_substituent(substituents[0])
        if direct:
            return direct + "amide"
    n_prefix = _n_substituent_prefix(substituents)
    if n_prefix:
        return _join_n_prefix_functional_parent(n_prefix, "amide")
    return ""


def _direct_anionic_amide_substituent(name: str) -> str:
    if _allowed_n_retained_parent_substituent(name) or _simple_anionic_amide_alkyl_substituent(name):
        return name
    return ""


def _simple_anionic_amide_alkyl_substituent(name: str) -> bool:
    return name in {
        "methyl",
        "ethyl",
        "propyl",
        "butyl",
        "pentyl",
        "hexyl",
        "heptyl",
        "octyl",
        "isopropyl",
        "t-butyl",
        "tert-butyl",
    }


def _sulfur_oxoanion_retained_parent_names(mol, Chem) -> list[str]:
    if sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) >= 0:
        return []
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for sulfur in mol.GetAtoms():
        if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0 or sulfur.IsInRing():
            continue
        record = _sulfur_oxoanion_retained_parent_record(mol, Chem, sulfur.GetIdx())
        if not record or record["covered_atoms"] != atom_set:
            continue
        names.extend(_sulfur_oxoanion_parent_names(record["substituent"], record["suffix"]))
    return names


def _sulfur_oxoanion_retained_parent_record(mol, Chem, sulfur_idx: int) -> dict | None:
    carbon_neighbors = []
    double_oxygen = 0
    terminal_oxido = 0
    covered_atoms = {sulfur_idx}
    for nbr in mol.GetAtomWithIdx(sulfur_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_neighbors.append(nidx)
            continue
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_oxygen += 1
            covered_atoms.add(nidx)
            continue
        if (
            nbr.GetSymbol() == "O"
            and bond.GetBondType() == Chem.BondType.SINGLE
            and nbr.GetFormalCharge() == -1
            and len(nbr.GetNeighbors()) == 1
        ):
            terminal_oxido += 1
            covered_atoms.add(nidx)
            continue
        return None
    if len(carbon_neighbors) != 1 or terminal_oxido != 1 or double_oxygen not in {1, 2}:
        return None
    carbon_idx = carbon_neighbors[0]
    substituent = _simple_or_branch_prefix(mol, carbon_idx, sulfur_idx)
    if not substituent:
        return None
    covered_atoms.add(carbon_idx)
    covered_atoms.update(_collect_component_atoms(mol, carbon_idx, sulfur_idx))
    return {
        "substituent": _retained_parent_substituent_alias(substituent),
        "suffix": "sulfinate" if double_oxygen == 1 else "sulfonate",
        "covered_atoms": covered_atoms,
    }


def _sulfur_oxoanion_parent_names(substituent: str, suffix: str) -> list[str]:
    if not substituent:
        return []
    names = [substituent + suffix]
    alkane = {
        "methyl": "methane",
        "ethyl": "ethane",
        "propyl": "propane",
        "butyl": "butane",
        "pentyl": "pentane",
        "hexyl": "hexane",
        "heptyl": "heptane",
        "octyl": "octane",
        "nonyl": "nonane",
        "decyl": "decane",
        "isopropyl": "propane-2",
        "t-butyl": "2-methylpropane-2",
        "tert-butyl": "2-methylpropane-2",
    }.get(substituent)
    if alkane:
        names.append(alkane + suffix)
    if substituent == "phenyl":
        names.append("benzene" + suffix)
    elif substituent.endswith("phenyl"):
        names.append(substituent[: -len("phenyl")] + "benzene" + suffix)
    return names


def _oxonium_retained_parent_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for oxygen in mol.GetAtoms():
        if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() <= 0 or oxygen.IsInRing():
            continue
        oidx = oxygen.GetIdx()
        if oxygen.GetTotalNumHs() != 0:
            continue
        carbon_neighbors = []
        for nbr in oxygen.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(oidx, nidx)
            if (
                nbr.GetSymbol() != "C"
                or nbr.GetFormalCharge() != 0
                or bond is None
                or bond.GetBondType() != Chem.BondType.SINGLE
            ):
                break
            carbon_neighbors.append(nidx)
        else:
            if len(carbon_neighbors) != 3:
                continue
            ligand_names, covered = _center_ligand_names_and_coverage(mol, oidx, carbon_neighbors)
            if covered != atom_set:
                continue
            name = _charged_ligand_parent_name(ligand_names, "oxonium")
            if name:
                names.append(name)
    return names


def _methylium_retained_parent_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 1 or carbon.IsInRing():
            continue
        cidx = carbon.GetIdx()
        carbon_neighbors = []
        ok = True
        for nbr in carbon.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if nbr.GetSymbol() != "C" or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                ok = False
                break
            carbon_neighbors.append(nidx)
        if not ok or not 1 <= len(carbon_neighbors) <= 3:
            continue
        if len(carbon_neighbors) + carbon.GetTotalNumHs() != 3:
            continue
        if all(not _collect_component_has_ring(mol, nidx, cidx) for nidx in carbon_neighbors):
            continue
        ligand_names, covered = _center_ligand_names_and_coverage(mol, cidx, carbon_neighbors)
        if covered != atom_set:
            continue
        name = _charged_ligand_parent_name(ligand_names, "methylium")
        if name:
            names.append(name)
    return names


def _cyano_methane_ion_parent_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() not in {-1, 1} or carbon.IsInRing():
            continue
        cidx = carbon.GetIdx()
        cyano_carbons = []
        for nbr in carbon.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if (
                nbr.GetSymbol() != "C"
                or bond is None
                or bond.GetBondType() != Chem.BondType.SINGLE
                or not _is_nitrile_carbon(mol, nidx)
            ):
                break
            cyano_carbons.append(nidx)
        else:
            if len(cyano_carbons) not in {2, 3}:
                continue
            covered = {cidx}
            for cyano_idx in cyano_carbons:
                covered.add(cyano_idx)
                nitrile_n = [
                    nbr.GetIdx()
                    for nbr in mol.GetAtomWithIdx(cyano_idx).GetNeighbors()
                    if nbr.GetIdx() != cidx and nbr.GetSymbol() == "N"
                ]
                if len(nitrile_n) != 1:
                    covered.clear()
                    break
                covered.add(nitrile_n[0])
            if covered != atom_set:
                continue
            multiplier = MULT.get(len(cyano_carbons))
            if not multiplier:
                continue
            suffix = "methanide" if carbon.GetFormalCharge() < 0 else "methylium"
            names.append(f"{multiplier}cyano{suffix}")
    return names


def _nitrone_azaniumolate_parent_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for nitrogen in mol.GetAtoms():
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() <= 0 or nitrogen.IsInRing():
            continue
        nidx = nitrogen.GetIdx()
        imine_carbons = []
        oxido_oxygens = []
        ok = True
        for nbr in nitrogen.GetNeighbors():
            nidx2 = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(nidx, nidx2)
            if bond is None:
                ok = False
                break
            if nbr.GetSymbol() == "C" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
                imine_carbons.append(nidx2)
            elif (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() < 0
                and not nbr.IsInRing()
                and bond.GetBondType() == Chem.BondType.SINGLE
                and len([other for other in nbr.GetNeighbors() if other.GetAtomicNum() > 1]) == 1
            ):
                oxido_oxygens.append(nidx2)
            elif nbr.GetAtomicNum() > 1:
                ok = False
                break
        if not ok or len(imine_carbons) != 1 or len(oxido_oxygens) != 1:
            continue
        carbon_idx = imine_carbons[0]
        carbon = mol.GetAtomWithIdx(carbon_idx)
        substituents = [
            nbr.GetIdx()
            for nbr in carbon.GetNeighbors()
            if nbr.GetIdx() != nidx and nbr.GetAtomicNum() > 1
        ]
        if carbon.IsInRing() or carbon.GetFormalCharge() != 0:
            continue
        if len(substituents) == 1:
            substituent_idx = substituents[0]
            if not _collect_component_has_ring(mol, substituent_idx, carbon_idx):
                continue
            prefix = _simple_or_branch_prefix(mol, substituent_idx, carbon_idx)
            if not prefix:
                continue
            covered = {nidx, carbon_idx, oxido_oxygens[0]}
            covered.update(_collect_component_atoms(mol, substituent_idx, carbon_idx))
            if covered != atom_set:
                continue
            bare = _strip_outer_parentheses(prefix)
            piece = _parenthesized_if_complex(bare)
            names.append(f"{piece}methylideneazaniumolate")
            continue
        if len(substituents) == 2:
            prefixes = []
            covered = {nidx, carbon_idx, oxido_oxygens[0]}
            for substituent_idx in substituents:
                prefix = _simple_or_branch_prefix(mol, substituent_idx, carbon_idx)
                if not prefix:
                    prefixes.clear()
                    break
                prefixes.append(_strip_outer_parentheses(prefix))
                covered.update(_collect_component_atoms(mol, substituent_idx, carbon_idx))
            if not prefixes or covered != atom_set:
                continue
            ligand_text = _nitrone_ligand_text(prefixes)
            if ligand_text:
                names.append(f"{ligand_text} nitrone")
    return names


def _nitrone_ligand_text(prefixes: Sequence[str]) -> str:
    counts = Counter(prefixes)
    pieces = []
    for prefix, count in sorted(counts.items(), key=lambda item: item[0]):
        if count == 1:
            pieces.append(prefix)
            continue
        multiplier = MULT.get(count)
        if not multiplier:
            return ""
        if _needs_substituent_parentheses(prefix):
            bare = _strip_outer_parentheses(prefix)
            pieces.append(f"bis({bare})" if count == 2 else f"{multiplier}({bare})")
        else:
            pieces.append(multiplier + prefix)
    return " ".join(pieces)


def _center_ligand_names_and_coverage(mol, center_idx: int, ligand_atoms: Sequence[int]) -> tuple[list[str], set[int]]:
    names = []
    covered = {center_idx}
    for atom_idx in ligand_atoms:
        prefix = _simple_or_branch_prefix(mol, atom_idx, center_idx)
        if not prefix:
            return [], set()
        names.append(prefix)
        covered.update(_collect_component_atoms(mol, atom_idx, center_idx))
    return names, covered


def _collect_component_has_ring(mol, start_idx: int, blocked_idx: int) -> bool:
    return any(mol.GetAtomWithIdx(idx).IsInRing() for idx in _collect_component_atoms(mol, start_idx, blocked_idx))


def _charged_ligand_parent_name(ligands: Sequence[str], parent: str) -> str:
    if not ligands:
        return ""
    counts = Counter(_strip_outer_parentheses(ligand) for ligand in ligands)
    pieces = []
    for ligand, count in sorted(counts.items(), key=lambda item: item[0]):
        unit = _charged_ligand_unit(ligand, count)
        if not unit:
            return ""
        pieces.append(unit)
    return "".join(pieces) + parent


def _charged_ligand_unit(ligand: str, count: int) -> str:
    if count == 1:
        return _parenthesized_if_complex(ligand)
    multiplier = MULT.get(count)
    if not multiplier:
        return ""
    if _simple_unparenthesized_n_substituent(ligand):
        return f"{multiplier}{ligand}"
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
        neighbor_symbols = tuple(mol.GetAtomWithIdx(nidx).GetSymbol() for nidx in neighbors)
        mixed_phosphane = _mixed_phosphane_hydride_scope(mol, self._Chem, pidx, neighbors)
        if not mixed_phosphane and any(symbol not in {"C", "Si"} for symbol in neighbor_symbols):
            return
        if any(
            mol.GetBondBetweenAtoms(pidx, nidx).GetBondType() != self._Chem.BondType.SINGLE
            for nidx in neighbors
        ):
            return
        names = []
        covered = {pidx}
        for nidx in neighbors:
            if mixed_phosphane:
                atom = mol.GetAtomWithIdx(nidx)
                if atom.GetSymbol() in HALO:
                    prefix = HALO[atom.GetSymbol()]
                    covered.add(nidx)
                else:
                    prefix = _simple_or_branch_prefix(mol, nidx, pidx)
                    if not prefix:
                        return
                    covered.update(_collect_component_atoms(mol, nidx, pidx))
            else:
                prefix = _simple_or_branch_prefix(mol, nidx, pidx)
                if not prefix:
                    return
            names.append(prefix)
        if mixed_phosphane and covered != set(range(mol.GetNumAtoms())):
            return
        if mixed_phosphane and any(name in HALO.values() for name in names):
            name = _halophosphane_hydride_name(names)
        else:
            name = _pnictogen_hydride_name(names, PNICTOGEN_HYDRIDE_PARENTS[pnictogen.GetSymbol()])
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(mol.GetNumAtoms())) if mixed_phosphane else frozenset(),
                source_family="mixed-phosphane-hydride" if mixed_phosphane else "",
                sort_key=(len(name), name),
                estimated_length=len(name),
            )


def _mixed_phosphane_hydride_scope(mol, Chem, phosphorus_idx: int, neighbors: Sequence[int]) -> bool:
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)
    if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0 or phosphorus.IsInRing():
        return False
    symbols = tuple(mol.GetAtomWithIdx(nidx).GetSymbol() for nidx in neighbors)
    if not symbols or any(symbol not in {"C", "N", "O", *HALO} for symbol in symbols):
        return False
    if all(symbol == "N" for symbol in symbols) or all(symbol == "O" for symbol in symbols):
        return False
    if all(symbol in HALO for symbol in symbols):
        return False
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    if any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        return False
    if any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds()):
        return False
    if mol.GetNumAtoms() > 48:
        return False
    hydroxy_oxygen_neighbors = 0
    for nidx, symbol in zip(neighbors, symbols):
        if symbol != "O":
            continue
        oxygen = mol.GetAtomWithIdx(nidx)
        bond = mol.GetBondBetweenAtoms(phosphorus_idx, nidx)
        if (
            bond is not None
            and bond.GetBondType() == Chem.BondType.SINGLE
            and oxygen.GetFormalCharge() == 0
            and len([nbr for nbr in oxygen.GetNeighbors() if nbr.GetAtomicNum() > 1]) == 1
            and oxygen.GetTotalNumHs() == 1
        ):
            hydroxy_oxygen_neighbors += 1
    if hydroxy_oxygen_neighbors >= 2 and any(symbol == "C" for symbol in symbols):
        return False
    if any(symbol in HALO for symbol in symbols):
        rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
        for nidx in neighbors:
            if mol.GetAtomWithIdx(nidx).GetSymbol() in HALO:
                continue
            component = set(_collect_component_atoms(mol, nidx, phosphorus_idx))
            if sum(1 for ring in rings if ring <= component) > 3:
                return False
    return True


def _halophosphane_hydride_name(names: Sequence[str]) -> str:
    counts = Counter(names)
    halo_names = set(HALO.values())
    pieces = []
    for name in sorted(name for name in counts if name in halo_names):
        count = counts[name]
        multiplier = MULT.get(count, "") if count > 1 else ""
        pieces.append(multiplier + name)
    for name in sorted(name for name in counts if name not in halo_names):
        bare = _strip_outer_parentheses(name)
        count = counts[name]
        if count == 1:
            pieces.append(f"({bare})")
            continue
        multiplier = _PHOSPHORANE_COMPLEX_MULTIPLIERS.get(count, "")
        if not multiplier:
            return ""
        pieces.append(f"{multiplier}({bare})")
    return "".join(pieces) + "phosphane" if pieces else ""


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
            if not ok or len(oxo) != 1 or not 1 <= len(heavy) <= 3:
                continue
            names = []
            covered = {sidx, oxo[0]}
            for nidx in heavy:
                prefix = _simple_or_branch_prefix(mol, nidx, sidx)
                if not prefix:
                    break
                names.append(prefix)
                covered.update(_collect_component_atoms(mol, nidx, sidx))
            else:
                if covered != set(range(mol.GetNumAtoms())):
                    continue
                name = _oxo_sulfanium_name(names)
                if name:
                    yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


def _oxo_sulfanium_name(names: Sequence[str]) -> str:
    if len(names) == 1:
        prefix = names[0]
        if not prefix:
            return ""
        if _needs_substituent_parentheses(prefix):
            prefix = f"({prefix})"
        return f"{prefix}-oxosulfanium"
    base = _sulfanium_name(names)
    if not base.endswith("sulfanium"):
        return ""
    return base[: -len("sulfanium")] + "oxosulfanium"


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


class CyclicNitroimineParentRule:
    family = "cyclic-nitroimine-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        for ring in mol.GetRingInfo().AtomRings():
            ring_set = frozenset(ring)
            name = (
                _oxadiazinane_nitroimine_name(mol, self._Chem, ring_set)
                or _imidazolidine_nitroimine_name(mol, self._Chem, ring_set)
            )
            if not name:
                continue
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(mol.GetNumAtoms())),
                sort_key=(len(name), name),
            )
            return


class OximeSulfonateNitrileParentRule:
    family = "oxime-sulfonate-nitrile-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        for motif in _oxime_sulfonyloxyimino_motifs(mol, self._Chem):
            name = (
                _direct_aryl_oxime_sulfonate_acetonitrile_name(mol, motif)
                or _thiophenylidene_oxime_sulfonate_acetonitrile_name(mol, self._Chem, motif)
            )
            if not name:
                continue
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(mol.GetNumAtoms())),
                sort_key=(len(name), name),
            )
            return


class NitrileOxideRetainedParentRule:
    family = "nitrile-oxide-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        for name in _nitrile_oxide_retained_parent_names(mol, self._Chem):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )


class FormamidineRetainedParentRule:
    family = "formamidine-retained-parent"
    priority = parent_priority("senior-functional-exception")
    requires_imino_skeleton_signal = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        for name in _formamidine_parent_names(mol, self._Chem):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )


class ChargedIminiumRetainedParentRule:
    family = "charged-iminium-retained-parent"
    priority = parent_priority("senior-functional-exception")
    requires_nonring_positive_nitrogen = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        for name in _charged_iminium_parent_names(mol, self._Chem):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )


class GuanidineRetainedParentRule:
    family = "guanidine-retained-parent"
    priority = parent_priority("senior-functional-exception")
    acyclic_only = True
    requires_imino_skeleton_signal = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        for name in _nitroguanidine_parent_names(mol, self._Chem):
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=self.priority,
                sort_key=(len(name), name),
                parent_atoms=set(range(mol.GetNumAtoms())),
            )
        if mol.GetRingInfo().NumRings() != 0:
            return
        for template in _guanidine_parent_templates(self._Chem):
            if template_record_mol(template).GetNumAtoms() > mol.GetNumAtoms():
                continue
            for match in ctx.template_matches(template):
                if any(mol.GetAtomWithIdx(idx).GetFormalCharge() > 0 for idx in match):
                    continue
                name = _name_acyclic_retained_core_match(mol, template, match)
                if not name:
                    continue
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=(-template_record_mol(template).GetNumAtoms(), len(name), name),
                )


def _formamidine_parent_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for carbon in mol.GetAtoms():
        if (
            carbon.GetSymbol() != "C"
            or carbon.IsInRing()
            or carbon.GetFormalCharge() != 0
            or carbon.GetTotalNumHs() != 1
        ):
            continue
        cidx = carbon.GetIdx()
        single_nitrogens = []
        imino_nitrogens = []
        ok = True
        for nbr in carbon.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(cidx, nbr.GetIdx())
            if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
                single_nitrogens.append(nbr.GetIdx())
            elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE:
                imino_nitrogens.append(nbr.GetIdx())
            elif nbr.GetAtomicNum() > 1:
                ok = False
                break
        if not ok or len(single_nitrogens) != 1 or len(imino_nitrogens) != 1:
            continue
        record = _formamidine_parent_record(mol, single_nitrogens[0], imino_nitrogens[0], cidx)
        if not record or record["covered_atoms"] != atom_set:
            continue
        n_prefix = _n_substituent_prefix(record["single_substituents"])
        if not n_prefix:
            continue
        imino = _parenthesized_if_complex(record["imino_substituent"])
        names.append(f"N'-{imino}-{n_prefix}formamidine")
    return names


def _formamidine_parent_record(mol, single_nidx: int, imino_nidx: int, carbon_idx: int) -> dict | None:
    single_nitrogen = mol.GetAtomWithIdx(single_nidx)
    imino_nitrogen = mol.GetAtomWithIdx(imino_nidx)
    if (
        single_nitrogen.GetFormalCharge() != 0
        or imino_nitrogen.GetFormalCharge() != 0
        or single_nitrogen.IsInRing()
        or imino_nitrogen.IsInRing()
    ):
        return None
    covered = {carbon_idx, single_nidx, imino_nidx}
    single_substituents = []
    for nbr in single_nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbon_idx:
            continue
        bond = mol.GetBondBetweenAtoms(single_nidx, nidx)
        if nbr.GetSymbol() != "C" or bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None
        prefix = _simple_or_branch_prefix(mol, nidx, single_nidx)
        if not prefix or _unstable_retained_functional_parent_substituent(prefix):
            return None
        single_substituents.append(_strip_outer_parentheses(prefix))
        covered.update(_collect_component_atoms(mol, nidx, single_nidx))
    imino_substituents = []
    for nbr in imino_nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbon_idx:
            continue
        bond = mol.GetBondBetweenAtoms(imino_nidx, nidx)
        if nbr.GetSymbol() != "C" or bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None
        prefix = _simple_or_branch_prefix(mol, nidx, imino_nidx)
        if not prefix or _unstable_retained_functional_parent_substituent(prefix):
            return None
        component_atoms = _collect_component_atoms(mol, nidx, imino_nidx)
        if _component_has_nonaromatic_carbon_nitrogen_double_bond(mol, component_atoms):
            return None
        imino_substituents.append(_strip_outer_parentheses(prefix))
        covered.update(component_atoms)
    if len(single_substituents) != 2 or len(imino_substituents) != 1:
        return None
    return {
        "single_substituents": tuple(single_substituents),
        "imino_substituent": imino_substituents[0],
        "covered_atoms": covered,
    }


def _component_has_nonaromatic_carbon_nitrogen_double_bond(mol, atoms: Collection[int]) -> bool:
    atom_set = set(atoms)
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 2.0 or bond.GetIsAromatic():
            continue
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in atom_set or end not in atom_set:
            continue
        symbols = {mol.GetAtomWithIdx(begin).GetSymbol(), mol.GetAtomWithIdx(end).GetSymbol()}
        if symbols == {"C", "N"}:
            return True
    return False


def _charged_iminium_parent_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.IsInRing() or carbon.GetFormalCharge() != 0:
            continue
        cidx = carbon.GetIdx()
        positive_imino = []
        single_nitrogens = []
        ok = True
        for nbr in carbon.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(cidx, nbr.GetIdx())
            if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE:
                if nbr.GetFormalCharge() > 0:
                    positive_imino.append(nbr.GetIdx())
                else:
                    ok = False
                    break
            elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
                single_nitrogens.append(nbr.GetIdx())
            elif nbr.GetAtomicNum() > 1:
                ok = False
                break
        if not ok or len(positive_imino) != 1:
            continue
        imino = mol.GetAtomWithIdx(positive_imino[0])
        if imino.IsInRing() or any(
            nbr.GetIdx() != cidx and nbr.GetAtomicNum() > 1 for nbr in imino.GetNeighbors()
        ):
            continue
        if len(single_nitrogens) == 1:
            record = _charged_formamidinium_record(mol, single_nitrogens[0], cidx)
            if record and record["covered_atoms"] == atom_set:
                prefix = _n_substituent_prefix(record["substituents"])
                if prefix is not None:
                    names.append(f"{prefix}formamidinium")
        elif len(single_nitrogens) == 2:
            record = _charged_guanidinium_record(mol, single_nitrogens, cidx)
            if record and record["covered_atoms"] == atom_set:
                prefix = _n_substituent_prefix(record["substituents"])
                if prefix is not None:
                    names.append(f"{prefix}guanidinium")
    return names


def _charged_formamidinium_record(mol, nitrogen_idx: int, carbon_idx: int) -> dict | None:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing():
        return None
    substituents, covered = _charged_iminium_n_substituents(mol, nitrogen_idx, carbon_idx)
    if substituents is None or len(substituents) > 2:
        return None
    covered.update({carbon_idx, _charged_iminium_positive_nitrogen(mol, carbon_idx)})
    return {"substituents": tuple(substituents), "covered_atoms": covered}


def _charged_guanidinium_record(mol, nitrogen_indices: Sequence[int], carbon_idx: int) -> dict | None:
    all_substituents = []
    covered = {carbon_idx, _charged_iminium_positive_nitrogen(mol, carbon_idx)}
    substituted_nitrogen_count = 0
    for nitrogen_idx in nitrogen_indices:
        nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
        if nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing():
            return None
        substituents, atom_coverage = _charged_iminium_n_substituents(mol, nitrogen_idx, carbon_idx)
        if substituents is None or len(substituents) > 2:
            return None
        if substituents:
            substituted_nitrogen_count += 1
        all_substituents.extend(substituents)
        covered.update(atom_coverage)
    if substituted_nitrogen_count > 1:
        return None
    return {"substituents": tuple(all_substituents), "covered_atoms": covered}


def _charged_iminium_positive_nitrogen(mol, carbon_idx: int) -> int:
    for nbr in mol.GetAtomWithIdx(carbon_idx).GetNeighbors():
        bond = mol.GetBondBetweenAtoms(carbon_idx, nbr.GetIdx())
        if (
            nbr.GetSymbol() == "N"
            and nbr.GetFormalCharge() > 0
            and bond is not None
            and bond.GetBondTypeAsDouble() == 2.0
        ):
            return nbr.GetIdx()
    return carbon_idx


def _charged_iminium_n_substituents(mol, nitrogen_idx: int, carbon_idx: int) -> tuple[list[str] | None, set[int]]:
    substituents = []
    covered = {nitrogen_idx}
    for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbon_idx:
            continue
        bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if nbr.GetSymbol() != "C" or bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None, covered
        prefix = _simple_or_branch_prefix(mol, nidx, nitrogen_idx)
        if not prefix or _unstable_retained_functional_parent_substituent(prefix):
            return None, covered
        substituents.append(_strip_outer_parentheses(prefix))
        covered.update(_collect_component_atoms(mol, nidx, nitrogen_idx))
    return substituents, covered


def _nitroguanidine_parent_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.IsInRing() or carbon.GetFormalCharge() != 0:
            continue
        cidx = carbon.GetIdx()
        nitrogen_neighbors = [
            nbr.GetIdx()
            for nbr in carbon.GetNeighbors()
            if nbr.GetSymbol() == "N" and mol.GetBondBetweenAtoms(cidx, nbr.GetIdx()) is not None
        ]
        if len(nitrogen_neighbors) != 3:
            continue
        imino_nitro = []
        single_nitrogens = []
        covered = {cidx}
        for nidx in nitrogen_neighbors:
            nitrogen = mol.GetAtomWithIdx(nidx)
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                nitro_atoms = _nitroguanidine_imino_nitro_atoms(mol, Chem, nidx, cidx)
                if nitro_atoms:
                    imino_nitro.append((nidx, nitro_atoms))
            elif bond.GetBondType() == Chem.BondType.SINGLE and nitrogen.GetFormalCharge() == 0:
                single_nitrogens.append(nidx)
        if len(imino_nitro) != 1 or len(single_nitrogens) != 2:
            continue
        imino_idx, nitro_atoms = imino_nitro[0]
        if not _nitroguanidine_core_stereo_compatible(mol, Chem, cidx, imino_idx):
            continue
        covered.add(imino_idx)
        covered.update(nitro_atoms)
        records = []
        ok = True
        for nidx in single_nitrogens:
            substituents = _external_substituent_names(mol, nidx, cidx)
            if substituents is None or len(substituents) > 2:
                ok = False
                break
            substituents = tuple(sorted(_retained_parent_substituent_alias(name) for name in substituents))
            if not all(_stable_nitroguanidine_substituent(name) for name in substituents):
                ok = False
                break
            covered.add(nidx)
            for nbr in mol.GetAtomWithIdx(nidx).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx == cidx or nbr.GetAtomicNum() <= 1:
                    continue
                covered.update(_collect_component_atoms(mol, nbr_idx, nidx))
            records.append(substituents)
        if not ok or covered != atom_set:
            continue
        name = _nitroguanidine_parent_name(records)
        if name:
            names.append(name)
    return names


def _nitroguanidine_core_stereo_compatible(mol, Chem, carbon_idx: int, imino_idx: int) -> bool:
    core_bond = frozenset((carbon_idx, imino_idx))
    for bond in mol.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        bond_atoms = frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if bond_atoms != core_bond:
            return False
    return True


def _nitroguanidine_imino_nitro_atoms(mol, Chem, imino_nidx: int, carbon_idx: int) -> set[int]:
    nitro_neighbors = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(imino_nidx).GetNeighbors()
        if nbr.GetIdx() != carbon_idx
        and nbr.GetSymbol() == "N"
        and nbr.GetFormalCharge() > 0
        and mol.GetBondBetweenAtoms(imino_nidx, nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    if len(nitro_neighbors) != 1:
        return set()
    nitro_idx = nitro_neighbors[0]
    oxygen_neighbors = []
    for nbr in mol.GetAtomWithIdx(nitro_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == imino_nidx:
            continue
        bond = mol.GetBondBetweenAtoms(nitro_idx, nidx)
        if nbr.GetSymbol() != "O" or bond is None:
            return set()
        if bond.GetBondType() == Chem.BondType.DOUBLE or (
            bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0
        ):
            oxygen_neighbors.append(nidx)
        else:
            return set()
    if len(oxygen_neighbors) != 2:
        return set()
    return {nitro_idx, *oxygen_neighbors}


def _nitroguanidine_parent_name(records: Sequence[tuple[str, ...]]) -> str:
    substituted = [record for record in records if record]
    if not substituted:
        return "2-nitroguanidine"
    pieces = []
    if len(substituted) == 1:
        piece = _nitroguanidine_substituent_piece("1", substituted[0])
        if not piece:
            return ""
        pieces.append(piece)
    else:
        for locant, substituents in zip(("1", "3"), sorted(substituted), strict=True):
            piece = _nitroguanidine_substituent_piece(locant, substituents)
            if not piece:
                return ""
            pieces.append(piece)
    return "-".join((*pieces, "2-nitroguanidine"))


def _nitroguanidine_substituent_piece(locant: str, substituents: Sequence[str]) -> str:
    if len(substituents) == 1:
        return _n_substituent_piece(locant, substituents[0])
    if len(substituents) == 2 and substituents[0] == substituents[1]:
        return _duplicate_n_substituent_piece(f"{locant},{locant}", substituents[0])
    if len(substituents) == 2:
        return "-".join(_n_substituent_piece(locant, name) for name in substituents)
    return ""


def _stable_nitroguanidine_substituent(name: str) -> bool:
    if len(name) > 96:
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    if _allowed_n_retained_parent_substituent(name):
        return True
    if name.endswith("methyl") and any(stem in name for stem in ("furyl", "oxolan", "pyrid", "thiazol")):
        return True
    return name.endswith(("amino", "methylideneamino", "ylideneamino"))


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


class AcylHydrazideRetainedParentRule:
    family = "acyl-hydrazide-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        seen = set()
        for name, source_family in _acyl_hydrazide_records(ctx.mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                source_family=source_family,
            )


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


def _acyl_hydrazide_records(mol, Chem) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for carbonyl_idx, nitrogen_idx, acyl_idx in _amide_acyl_motifs(mol, Chem):
        motif = _acyl_hydrazide_imine_motif(mol, Chem, carbonyl_idx, nitrogen_idx)
        if motif is None:
            continue
        _distal_n_idx, imine_carbon_idx = motif
        if _hydrazide_imine_branch_has_carboxyl_function(mol, Chem, imine_carbon_idx, _distal_n_idx):
            continue
        ylidene = _hydrazone_ylidene_name(mol, imine_carbon_idx, _distal_n_idx)
        if not ylidene:
            continue
        n_prime_prefix = _hydrazide_n_prime_prefix(ylidene)
        for path in _ester_acyl_parent_paths(mol, Chem, carbonyl_idx, acyl_idx):
            core = _generic_acyl_functional_parent_path_name(mol, Chem, carbonyl_idx, path, "hydrazide")
            if core:
                records.append((_join_n_prefix_functional_parent(n_prime_prefix, core), "generic-acyl-hydrazide"))
        records.extend(
            _ring_carbohydrazide_parent_records(
                mol,
                Chem,
                carbonyl_idx,
                nitrogen_idx,
                acyl_idx,
                n_prime_prefix,
            )
        )
    return records


def _acyl_hydrazide_imine_motif(mol, Chem, carbonyl_idx: int, nitrogen_idx: int) -> tuple[int, int] | None:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing():
        return None
    heavy = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors() if nbr.GetAtomicNum() > 1]
    distal = [
        idx
        for idx in heavy
        if idx != carbonyl_idx
        and mol.GetAtomWithIdx(idx).GetSymbol() == "N"
        and mol.GetBondBetweenAtoms(nitrogen_idx, idx).GetBondType() == Chem.BondType.SINGLE
    ]
    if len(distal) != 1 or sorted(heavy) != sorted((carbonyl_idx, distal[0])):
        return None
    distal_n_idx = distal[0]
    distal_n = mol.GetAtomWithIdx(distal_n_idx)
    if distal_n.GetFormalCharge() != 0 or distal_n.IsInRing():
        return None
    distal_heavy = [nbr.GetIdx() for nbr in distal_n.GetNeighbors() if nbr.GetAtomicNum() > 1]
    imine_carbons = [
        nbr.GetIdx()
        for nbr in distal_n.GetNeighbors()
        if nbr.GetIdx() != nitrogen_idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(distal_n_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]
    if len(imine_carbons) != 1 or sorted(distal_heavy) != sorted((nitrogen_idx, imine_carbons[0])):
        return None
    return distal_n_idx, imine_carbons[0]


def _hydrazide_n_prime_prefix(ylidene: str) -> str:
    ylidene_text = ylidene if ylidene.startswith("(") and ylidene.endswith(")") else f"({ylidene})"
    return f"N'-{ylidene_text}"


def _hydrazide_imine_branch_has_carboxyl_function(mol, Chem, imine_carbon_idx: int, distal_n_idx: int) -> bool:
    stack = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(imine_carbon_idx).GetNeighbors() if nbr.GetIdx() != distal_n_idx]
    seen = {distal_n_idx, imine_carbon_idx}
    branch_atoms: set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        branch_atoms.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in seen:
                stack.append(nidx)
    for idx in branch_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "C":
            continue
        has_carbonyl_oxygen = False
        has_hetero_single = False
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(idx, nidx)
            if nidx not in branch_atoms:
                continue
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                has_carbonyl_oxygen = True
            if nbr.GetSymbol() in {"O", "N", "S"} and bond.GetBondType() == Chem.BondType.SINGLE:
                has_hetero_single = True
        if has_carbonyl_oxygen and has_hetero_single:
            return True
    return False


def _carbodiimide_parent_records(mol, Chem) -> list[tuple[str, frozenset[int]]]:
    motif_records = []
    atom_set = set(range(mol.GetNumAtoms()))
    motifs = _carbodiimide_motif_records(mol, Chem)
    if len(motifs) > 1:
        records = _bis_phenylene_carbodiimide_parent_records(mol, Chem, motifs)
        return sorted(set(records), key=lambda item: (len(item[0]), item[0])) if len(records) == 1 else []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            continue
        carbon_idx = atom.GetIdx()
        terminal_nitrogens = []
        ok = True
        for nbr in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(carbon_idx, nbr.GetIdx())
            if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE:
                terminal_nitrogens.append(nbr.GetIdx())
            elif nbr.GetAtomicNum() > 1:
                ok = False
                break
        if not ok or len(terminal_nitrogens) != 2:
            continue
        substituents = []
        covered = {carbon_idx, *terminal_nitrogens}
        for nitrogen_idx in terminal_nitrogens:
            record = _carbodiimide_n_substituent_record(mol, Chem, nitrogen_idx, carbon_idx)
            if record is None:
                substituents = None
                break
            substituent, component_atoms = record
            if substituent:
                substituents.append(substituent)
            covered.update(component_atoms)
        if substituents is None:
            continue
        if covered != atom_set:
            continue
        name = _carbodiimide_name(substituents)
        if name:
            motif_records.append((name, frozenset(covered)))
    if len(motif_records) != 1:
        return []
    return sorted(set(motif_records), key=lambda item: (len(item[0]), item[0]))


def _carbodiimide_n_substituent(mol, Chem, nitrogen_idx: int, carbon_idx: int) -> str | None:
    record = _carbodiimide_n_substituent_record(mol, Chem, nitrogen_idx, carbon_idx)
    return None if record is None else record[0]


def _carbodiimide_n_substituent_record(
    mol,
    Chem,
    nitrogen_idx: int,
    carbon_idx: int,
) -> tuple[str, frozenset[int]] | None:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing():
        return None
    substituent_atoms = []
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbon_idx or nbr.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        substituent_atoms.append(nidx)
    if len(substituent_atoms) > 1:
        return None
    if not substituent_atoms:
        return "", frozenset()
    component_atoms = frozenset(_collect_component_atoms(mol, substituent_atoms[0], nitrogen_idx))
    substituent = _retained_parent_substituent_alias(
        _simple_or_branch_prefix(mol, substituent_atoms[0], nitrogen_idx)
    )
    if not substituent or not _allowed_carbodiimide_substituent(substituent):
        return None
    return substituent, component_atoms


def _carbodiimide_parent_names(mol, Chem) -> list[str]:
    return [name for name, _covered in _carbodiimide_parent_records(mol, Chem)]


def _carbodiimide_motif_records(mol, Chem) -> list[dict]:
    motifs = []
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
            continue
        carbon_idx = carbon.GetIdx()
        terminal_nitrogens = []
        ok = True
        for nbr in carbon.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(carbon_idx, nbr.GetIdx())
            if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE:
                terminal_nitrogens.append(nbr.GetIdx())
            elif nbr.GetAtomicNum() > 1:
                ok = False
                break
        if ok and len(terminal_nitrogens) == 2:
            motifs.append({"carbon": carbon_idx, "nitrogens": tuple(terminal_nitrogens)})
    return motifs


def _bis_phenylene_carbodiimide_parent_records(mol, Chem, motifs: Sequence[dict]) -> list[tuple[str, frozenset[int]]]:
    if len(motifs) != 2:
        return []
    atom_set = set(range(mol.GetNumAtoms()))
    records = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        ring_set = frozenset(ring)
        if any(
            mol.GetAtomWithIdx(idx).GetSymbol() != "C" or not mol.GetAtomWithIdx(idx).GetIsAromatic()
            for idx in ring
        ):
            continue
        if any(
            nbr.GetIdx() not in ring_set
            and not _is_carbodiimide_bridge_nitrogen(mol, nbr.GetIdx(), motifs)
            for idx in ring
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
            if nbr.GetAtomicNum() > 1
        ):
            continue
        motif_units = []
        covered = set(ring_set)
        attachment_atoms = []
        for motif in motifs:
            bridge_nitrogens = [
                nidx
                for nidx in motif["nitrogens"]
                if _single_carbodiimide_n_ring_attachment(mol, nidx, motif["carbon"]) in ring_set
            ]
            if len(bridge_nitrogens) != 1:
                motif_units = []
                break
            bridge_nitrogen = bridge_nitrogens[0]
            ring_atom = _single_carbodiimide_n_ring_attachment(mol, bridge_nitrogen, motif["carbon"])
            external_nitrogen = next(nidx for nidx in motif["nitrogens"] if nidx != bridge_nitrogen)
            external_record = _carbodiimide_n_substituent_record(mol, Chem, external_nitrogen, motif["carbon"])
            if external_record is None:
                motif_units = []
                break
            substituent, component_atoms = external_record
            if not substituent:
                motif_units = []
                break
            motif_units.append(substituent)
            attachment_atoms.append(ring_atom)
            covered.update({motif["carbon"], bridge_nitrogen, external_nitrogen})
            covered.update(component_atoms)
        if len(motif_units) != 2 or motif_units[0] != motif_units[1]:
            continue
        if covered != atom_set:
            continue
        distance = _ring_distance(mol, ring_set, attachment_atoms[0], attachment_atoms[1])
        locant = {1: "1,2", 2: "1,3", 3: "1,4"}.get(distance)
        if not locant:
            continue
        unit = _n_substituent_piece("N", motif_units[0]) + "carbodiimide"
        records.append((f"{locant}-phenylene bis({unit})", frozenset(covered)))
    return records


def _is_carbodiimide_bridge_nitrogen(mol, nitrogen_idx: int, motifs: Sequence[dict]) -> bool:
    atom = mol.GetAtomWithIdx(nitrogen_idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.IsInRing():
        return False
    return any(nitrogen_idx in motif["nitrogens"] for motif in motifs)


def _single_carbodiimide_n_ring_attachment(mol, nitrogen_idx: int, carbon_idx: int) -> int | None:
    ring_neighbors = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors()
        if nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1 and nbr.IsInRing()
    ]
    return ring_neighbors[0] if len(ring_neighbors) == 1 else None


def _carbodiimide_name(substituents: Sequence[str]) -> str:
    substituents = tuple(sorted(substituents))
    if not substituents:
        return "carbodiimide"
    if len(substituents) == 1:
        return f"{_n_substituent_piece('N', substituents[0])}carbodiimide"
    if len(substituents) == 2:
        if substituents[0] == substituents[1]:
            prefix = _duplicate_n_substituent_piece("N,N'", substituents[0])
            return f"{prefix}carbodiimide"
        first = _n_substituent_piece("N", substituents[0])
        second = _n_substituent_piece("N'", substituents[1])
        return f"{first}-{second}carbodiimide"
    return ""


def _allowed_carbodiimide_substituent(name: str) -> bool:
    if _allowed_n_retained_parent_substituent(name):
        return True
    if _simple_n_retained_parent_substituent(name):
        return True
    if _simple_linear_alkyl_substituent(name):
        return True
    if len(name) > 56:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "carbam", "urea", "imino")):
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    if name.endswith(("benzoyl", "formyl", "acetyl", "carbonyl")) or "benzoyl" in name:
        return True
    if re.fullmatch(r"(?:\d+-(?:fluoro|chloro|bromo|iodo|methyl|ethyl|methoxy|ethoxy)-)*[A-Za-z]+yl", name):
        return True
    return any(
        fragment in name
        for fragment in (
            "amino",
            "aza",
            "bromo",
            "chloro",
            "cyclo",
            "cyano",
            "ethoxy",
            "fluoro",
            "iodo",
            "morpholin",
            "nitro",
            "oxa",
            "piperid",
            "piperaz",
            "pyrrolid",
            "azetidin",
            "phenyl",
            "benzyl",
            "pyrid",
            "trifluoromethyl",
            "hydroxy",
            "methoxy",
            "oxo",
        )
    )


def _simple_linear_alkyl_substituent(name: str) -> bool:
    return name in {
        "methyl",
        "ethyl",
        "propyl",
        "butyl",
        "pentyl",
        "hexyl",
        "heptyl",
        "octyl",
        "nonyl",
        "decyl",
        "undecyl",
        "dodecyl",
        "tridecyl",
        "tetradecyl",
        "pentadecyl",
        "hexadecyl",
        "heptadecyl",
        "octadecyl",
    }


class IsocyanateRetainedParentRule:
    family = "isocyanate-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        seen = set()
        for name in _isocyanate_parent_names(ctx.mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


def _isocyanate_parent_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
            continue
        carbon_idx = carbon.GetIdx()
        oxygen_idx = None
        nitrogen_idx = None
        ok = True
        for nbr in carbon.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(carbon_idx, nbr.GetIdx())
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                oxygen_idx = nbr.GetIdx() if oxygen_idx is None else None
            elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE:
                nitrogen_idx = nbr.GetIdx() if nitrogen_idx is None else None
            elif nbr.GetAtomicNum() > 1:
                ok = False
                break
        if not ok or oxygen_idx is None or nitrogen_idx is None:
            continue
        nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
        if nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing():
            continue
        substituents = [
            nbr.GetIdx()
            for nbr in nitrogen.GetNeighbors()
            if nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1
        ]
        if len(substituents) != 1:
            continue
        substituent_idx = substituents[0]
        covered = {carbon_idx, oxygen_idx, nitrogen_idx}
        covered.update(_collect_component_atoms(mol, substituent_idx, nitrogen_idx))
        if covered != atom_set:
            continue
        substituent = _retained_parent_substituent_alias(
            _simple_or_branch_prefix(mol, substituent_idx, nitrogen_idx)
        )
        if not substituent or not _allowed_isocyanate_substituent(substituent):
            continue
        names.append(f"{_parenthesized_if_complex(substituent)} isocyanate")
    return sorted(set(names), key=lambda name: (len(name), name))


def _allowed_isocyanate_substituent(name: str) -> bool:
    if "imino" in name:
        return False
    if _simple_n_retained_parent_substituent(name) or _simple_linear_alkyl_substituent(name):
        return True
    if len(name) > 48:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "carbam", "urea")):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9,()'\-]+", name))


class CarbamateRetainedParentRule:
    family = "carbamate-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        for carbonyl_idx, oxygen_idx, nitrogen_idx in _carbamate_motifs(mol, self._Chem):
            charged_name = _carbamate_azanide_parent_name(mol, self._Chem, carbonyl_idx, oxygen_idx, nitrogen_idx)
            if charged_name:
                yield NameCandidate(
                    charged_name,
                    self.family,
                    self.priority,
                    sort_key=(len(charged_name), charged_name),
                    parent_atoms=frozenset(range(mol.GetNumAtoms())),
                    source_family="carbamate-retained-parent:azanide",
                    estimated_length=len(charged_name),
                )
                continue
            nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
            if nitrogen.GetFormalCharge() != 0:
                continue
            oxygen_substituent = _single_external_substituent_name(mol, oxygen_idx, carbonyl_idx)
            if not oxygen_substituent:
                continue
            nitrogen_substituents = _external_substituent_names(mol, nitrogen_idx, carbonyl_idx)
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


class CyclicNCarbamateCarboxylateRetainedParentRule:
    family = "cyclic-n-carbamate-carboxylate-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen = set()
        for carbonyl_idx, oxygen_idx, nitrogen_idx in _carbamate_motifs(mol, self._Chem):
            nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
            if not nitrogen.IsInRing() or nitrogen.GetFormalCharge() != 0:
                continue
            oxygen_substituent = _single_external_substituent_name(mol, oxygen_idx, carbonyl_idx)
            if not oxygen_substituent:
                continue
            ring_parent = _cyclic_n_carbamate_carboxylate_parent_name(mol, self._Chem, nitrogen_idx, carbonyl_idx)
            if not ring_parent:
                continue
            name = f"{oxygen_substituent} {ring_parent}"
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=frozenset(range(mol.GetNumAtoms())),
                source_family="cyclic-n-carbamate:carboxylate",
                estimated_length=len(name),
            )


class CarbodiimideRetainedParentRule:
    family = "carbodiimide-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        seen = set()
        for name, parent_atoms in _carbodiimide_parent_records(ctx.mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=parent_atoms,
                source_family="carbodiimide-retained",
                estimated_length=len(name),
            )


class RetainedAcidEsterParentRule:
    family = "retained-acid-ester-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen = set()
        for name, source_family in _retained_acid_ester_records(mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            parent_atoms = (
                _heavy_atom_indices(mol)
                if source_family == "retained-acid-ester:dicarboxylate-acyl-owner"
                else frozenset()
            )
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                source_family=source_family,
                parent_atoms=parent_atoms,
                estimated_length=len(name),
            )


class AcylDiphosphateRetainedParentRule:
    family = "retained-acid-ester-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen = set()
        for name, source_family, parent_atoms in _acyl_diphosphate_parent_records(mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                source_family=source_family,
                parent_atoms=parent_atoms,
            )


class RetainedAmideParentRule:
    family = "retained-amide-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen = set()
        for name, source_family, parent_atoms in _retained_amide_records(mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                source_family=source_family,
                parent_atoms=parent_atoms,
            )


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


class OxoCarboxylateRetainedParentRule:
    family = "oxo-carboxylate-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen = set()
        for name, stereo_locants in _oxo_carboxylate_parent_records(mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=_heavy_atom_indices(mol),
                stereo_locants=stereo_locants,
            )


class TerminalCarboxylateParentRule:
    family = "terminal-carboxylate-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen = set()
        for name in _terminal_carboxylate_parent_names(mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=_heavy_atom_indices(mol),
                source_family="terminal-carboxylate",
                estimated_length=len(name),
            )


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
            imino_substituent = _sulfonamide_imino_n_substituent(mol, self._Chem, nitrogen_idx, sulfur_idx)
            if imino_substituent:
                name = f"N-({imino_substituent}){base}"
                yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))
                continue
            if _sulfonamide_defer_to_sulfonohydrazide(mol, self._Chem, nitrogen_idx, sulfur_idx):
                continue
            nitrogen_substituents = _external_substituent_names(mol, nitrogen_idx, sulfur_idx)
            if nitrogen_substituents is None or len(nitrogen_substituents) > 2:
                continue
            nitrogen_prefix = _n_substituent_prefix(nitrogen_substituents)
            if nitrogen_prefix is None:
                continue
            if any(_unstable_retained_functional_parent_substituent(name) for name in nitrogen_substituents):
                continue
            name = _join_retained_sulfonamide_parent(nitrogen_prefix, base)
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class SulfonamideParentRule:
    family = "sulfonamide-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        for sulfur_idx, nitrogen_idx in _n_substituted_sulfonamide_motifs(mol, self._Chem):
            nitrogen_substituents = _external_substituent_names(mol, nitrogen_idx, sulfur_idx)
            if nitrogen_substituents is None or len(nitrogen_substituents) > 2:
                continue
            nitrogen_prefix = _sulfonamide_n_substituent_prefix(nitrogen_substituents)
            if not nitrogen_prefix:
                continue
            name = f"{nitrogen_prefix}-sulfonamide"
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=frozenset(range(mol.GetNumAtoms())),
                estimated_length=len(name),
            )


class SulfonimidamideParentRule:
    family = "sulfonimidamide-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen = set()
        for motif in _sulfonimidamide_motifs(mol, self._Chem):
            base = _sulfonimidamide_base_name(mol, motif["sulfur"], motif["s_substituent"])
            if not base:
                continue
            amide_substituents = _external_substituent_names(mol, motif["amide_nitrogen"], motif["sulfur"])
            imino_substituents = _external_substituent_names(mol, motif["imino_nitrogen"], motif["sulfur"])
            if amide_substituents is None or imino_substituents is None:
                continue
            prefix = _sulfonimidamide_n_substituent_prefix(amide_substituents, imino_substituents)
            if prefix is None:
                continue
            if prefix:
                name = f"{prefix}-{base}" if _needs_explicit_n_parent_separator(prefix) else prefix + base
            else:
                name = base
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=frozenset(range(mol.GetNumAtoms())),
                estimated_length=len(name),
            )


class LambdaSulfanoneParentRule:
    family = "lambda-sulfanone-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen: set[str] = set()
        for sulfur_idx, ligand_records in _lambda_sulfanone_parent_records(mol, self._Chem):
            ligand_text = "".join(f"({name})" for _atom_idx, name in ligand_records)
            name = f"{ligand_text}-lambda6-sulfanone"
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(sulfur_idx, tuple(atom_idx for atom_idx, _name in ligand_records)),
                parent_atoms=_heavy_atom_indices(mol),
                source_family="lambda-sulfanone",
                estimated_length=len(name),
            )


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
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=frozenset(range(mol.GetNumAtoms())),
                estimated_length=len(name),
            )


class SydnoneUreaAmidateRetainedParentRule:
    family = "charged-urea-amidate-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen = set()
        for name in _sydnone_urea_amidate_names(mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=frozenset(range(mol.GetNumAtoms())),
                source_family="charged-urea-amidate:sydnone",
                estimated_length=len(name),
            )


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
        atom_count = mol.GetNumAtoms()
        for name in _amino_acid_parent_names(mol, self._Chem):
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(atom_count)),
                sort_key=(len(name), name),
            )


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


def _azanium_ligand_name(mol, Chem, nitrogen_idx: int, ligand_idx: int) -> str:
    return _azanium_alkoxycarbonyl_ligand_name(mol, Chem, nitrogen_idx, ligand_idx) or _simple_or_branch_prefix(
        mol,
        ligand_idx,
        nitrogen_idx,
    )


def _azanium_alkoxycarbonyl_ligand_name(mol, Chem, nitrogen_idx: int, carbonyl_idx: int) -> str:
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0 or carbonyl.IsInRing():
        return ""
    bond_to_nitrogen = mol.GetBondBetweenAtoms(carbonyl_idx, nitrogen_idx)
    if bond_to_nitrogen is None or bond_to_nitrogen.GetBondType() != Chem.BondType.SINGLE:
        return ""
    double_oxygens = []
    single_oxygens = []
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            single_oxygens.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return ""
    if len(double_oxygens) != 1 or len(single_oxygens) != 1:
        return ""
    owner = _single_external_substituent_name(mol, single_oxygens[0], carbonyl_idx)
    return {
        "t-butyl": "tert-butoxycarbonyl",
        "tert-butyl": "tert-butoxycarbonyl",
        "benzyl": "carbobenzoxy",
    }.get(owner, "")


def _carbamate_azanide_parent_name(mol, Chem, carbonyl_idx: int, oxygen_idx: int, nitrogen_idx: int) -> str:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetFormalCharge() != -1 or nitrogen.IsInRing():
        return ""
    oxygen_substituent = _single_external_substituent_name(mol, oxygen_idx, carbonyl_idx)
    owner = _carbamate_azanide_ester_owner_name(oxygen_substituent)
    if not owner:
        return ""
    nitrogen_substituents = _external_substituent_names(mol, nitrogen_idx, carbonyl_idx)
    if nitrogen_substituents is None or len(nitrogen_substituents) > 1:
        return ""
    if not nitrogen_substituents:
        return f"{owner} azanidecarboxylate"
    substituent = _retained_parent_substituent_alias(nitrogen_substituents[0])
    if not _simple_n_retained_parent_substituent(substituent):
        return ""
    return f"{owner} {substituent}azanidecarboxylate"


def _carbamate_azanide_ester_owner_name(name: str) -> str:
    return {
        "methyl": "methyl",
        "ethyl": "ethyl",
        "propyl": "propyl",
        "isopropyl": "isopropyl",
        "butyl": "butyl",
        "isobutyl": "isobutyl",
        "t-butyl": "tert-butyl",
        "tert-butyl": "tert-butyl",
        "benzyl": "benzyl",
    }.get(name, "")


def _cyclic_n_carbamate_carboxylate_parent_name(mol, Chem, nitrogen_idx: int, carbonyl_idx: int) -> str:
    branch = _simple_or_branch_prefix(mol, nitrogen_idx, carbonyl_idx)
    if not branch:
        return ""
    branch, has_exocyclic_carboxymethylidene = _with_exocyclic_carboxymethylidene_branch(
        mol,
        Chem,
        nitrogen_idx,
        carbonyl_idx,
        branch,
    )
    if not has_exocyclic_carboxymethylidene:
        return ""
    return _cyclic_n_carbamate_parent_from_branch(branch)


def _cyclic_n_carbamate_parent_name(mol, Chem, nitrogen_idx: int, carbonyl_idx: int) -> str:
    branch = _simple_or_branch_prefix(mol, nitrogen_idx, carbonyl_idx)
    if not branch:
        return ""
    branch, _has_exocyclic_carboxymethylidene = _with_exocyclic_carboxymethylidene_branch(
        mol,
        Chem,
        nitrogen_idx,
        carbonyl_idx,
        branch,
    )
    return _cyclic_n_carbamate_parent_from_branch(branch)


def _cyclic_n_carbamate_parent_from_branch(branch: str) -> str:
    parent = _cyclic_n_branch_to_carboxylate_parent(branch)
    if not parent:
        return ""
    if " " in parent:
        return ""
    if len(parent) > 180:
        return ""
    return parent


def _with_exocyclic_carboxymethylidene_branch(
    mol,
    Chem,
    nitrogen_idx: int,
    carbonyl_idx: int,
    branch: str,
) -> tuple[str, bool]:
    if "2-oxo-3-oxapropyl" not in branch:
        return branch, False
    kind = _exocyclic_carboxymethylidene_kind(mol, Chem, nitrogen_idx, carbonyl_idx)
    if not kind:
        return branch, False
    replacement = "carboxylatomethylidene" if kind == "carboxylate" else "carboxymethylidene"
    return branch.replace("2-oxo-3-oxapropyl", replacement, 1), True


def _exocyclic_carboxymethylidene_kind(mol, Chem, nitrogen_idx: int, carbonyl_idx: int) -> str:
    rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings() if nitrogen_idx in ring]
    if len(rings) != 1:
        return ""
    ring_set = set(rings[0])
    for ring_idx in ring_set:
        if ring_idx == nitrogen_idx:
            continue
        for alpha in mol.GetAtomWithIdx(ring_idx).GetNeighbors():
            alpha_idx = alpha.GetIdx()
            if alpha_idx in ring_set or alpha_idx == carbonyl_idx or alpha.GetSymbol() != "C":
                continue
            alpha_bond = mol.GetBondBetweenAtoms(ring_idx, alpha_idx)
            if alpha_bond is None or alpha_bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            carboxyl_carbons = [
                nbr.GetIdx()
                for nbr in alpha.GetNeighbors()
                if nbr.GetIdx() != ring_idx and nbr.GetSymbol() == "C"
            ]
            if len(carboxyl_carbons) != 1:
                continue
            kind = _terminal_carboxyl_group_kind(mol, Chem, carboxyl_carbons[0], alpha_idx)
            if kind in {"acid", "carboxylate"}:
                return kind
    return ""


def _cyclic_n_branch_to_carboxylate_parent(branch: str) -> str:
    saturated_n_suffixes = {
        "piperidino": "piperidine-1-carboxylate",
        "piperazino": "piperazine-1-carboxylate",
        "pyrrolidino": "pyrrolidine-1-carboxylate",
        "morpholino": "morpholine-4-carboxylate",
    }
    for suffix, replacement in saturated_n_suffixes.items():
        if branch == suffix:
            return replacement
        if branch.endswith(suffix):
            return branch[: -len(suffix)] + replacement
    match = re.fullmatch(
        r"(.+(?:aza|diaza|triaza)[A-Za-z0-9,\.\[\]\(\)\^']*?)(?:an|ane)-([0-9]+[a-z]?)-yl",
        branch,
    )
    if match:
        return f"{match.group(1)}ane-{match.group(2)}-carboxylate"
    match = re.fullmatch(
        r"(.+(?:azetidin|pyrrolidin|piperidin|piperazin|morpholin))-([0-9]+[a-z]?)-yl",
        branch,
    )
    if match:
        return f"{match.group(1)}e-{match.group(2)}-carboxylate"
    return ""


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
        if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0:
            continue
        pidx = phosphorus.GetIdx()
        if phosphorus.IsInRing():
            cyclic_name = _cyclic_phosphite_name(mol, Chem, pidx)
            if cyclic_name:
                names.append(cyclic_name)
            continue
        terminal_anionic_oxygen = 0
        terminal_hydroxy_oxygen = 0
        owner_records = []
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
            if not heavy and nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                terminal_hydroxy_oxygen += 1
                continue
            if len(heavy) != 1 or nbr.GetFormalCharge() != 0:
                ok = False
                break
            owner = _single_external_substituent_name(mol, nidx, pidx)
            if not owner or _unstable_retained_functional_parent_substituent(owner):
                ok = False
                break
            owner_records.append((owner, nidx, heavy[0]))
        owner_names = [owner for owner, _, _ in owner_records]
        if ok and terminal_anionic_oxygen == 1 and len(owner_records) == 2:
            names.append(_phosphite_ester_name_from_records(mol, Chem, owner_records))
        if (
            ok
            and len(owner_names) == 3
            and terminal_anionic_oxygen == 0
            and terminal_hydroxy_oxygen == 0
            and phosphorus.GetTotalNumHs() == 0
        ):
            names.append(_phosphite_triester_name(owner_names))
        if (
            ok
            and len(owner_names) == 1
            and terminal_anionic_oxygen + terminal_hydroxy_oxygen == 2
            and terminal_anionic_oxygen in {0, 1, 2}
        ):
            names.append(_phosphite_monoester_name(owner_names[0], terminal_hydroxy_oxygen))
    names.extend(_mixed_p3_anionic_phosphorus_names(mol, Chem))
    return sorted(set(names), key=lambda name: (len(name), name))


def _cyclic_phosphite_name(mol, Chem, pidx: int) -> str:
    phosphorus = mol.GetAtomWithIdx(pidx)
    terminal_anionic_oxygen = 0
    oxygen_links: list[tuple[int, int]] = []
    covered = {pidx}
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if nbr.GetSymbol() != "O" or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        heavy = [atom.GetIdx() for atom in nbr.GetNeighbors() if atom.GetIdx() != pidx and atom.GetAtomicNum() > 1]
        if not heavy and nbr.GetFormalCharge() < 0:
            terminal_anionic_oxygen += 1
            covered.add(nidx)
            continue
        if len(heavy) != 1 or nbr.GetFormalCharge() != 0 or not nbr.IsInRing():
            return ""
        owner_idx = heavy[0]
        if mol.GetAtomWithIdx(owner_idx).GetSymbol() != "C":
            return ""
        oxygen_links.append((nidx, owner_idx))
        covered.add(nidx)
    if terminal_anionic_oxygen != 1 or len(oxygen_links) != 2 or phosphorus.GetTotalNumHs() != 0:
        return ""
    if oxygen_links[0][1] == oxygen_links[1][1]:
        prefix, carbon_atoms = _cyclic_phosphite_alkylidene_prefix(
            mol,
            Chem,
            oxygen_links[0][1],
            (oxygen_links[0][0], oxygen_links[1][0]),
        )
    else:
        prefix, carbon_atoms = _cyclic_phosphite_alkylene_prefix(
            mol,
            Chem,
            oxygen_links[0][1],
            oxygen_links[1][1],
            (oxygen_links[0][0], oxygen_links[1][0]),
        )
    if not prefix:
        return ""
    covered.update(carbon_atoms)
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    return f"{prefix} phosphite"


def _cyclic_phosphite_alkylidene_prefix(
    mol,
    Chem,
    center_idx: int,
    oxygen_indices: Sequence[int],
) -> tuple[str, set[int]]:
    center = mol.GetAtomWithIdx(center_idx)
    oxygen_set = set(oxygen_indices)
    carbon_neighbors = []
    for nbr in center.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx in oxygen_set or nbr.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(center_idx, nidx)
        if nbr.GetSymbol() != "C" or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return "", set()
        carbon_neighbors.append(nidx)
    if len(carbon_neighbors) > 1:
        return "", set()
    path = [center_idx]
    if carbon_neighbors:
        tail = _linear_saturated_carbon_chain_from_terminal(mol, Chem, carbon_neighbors[0], center_idx)
        if not tail:
            return "", set()
        path.extend(tail)
    root = ROOTS.get(len(path))
    if not root:
        return "", set()
    return root + "ylidene", set(path)


def _cyclic_phosphite_alkylene_prefix(
    mol,
    Chem,
    first_idx: int,
    second_idx: int,
    oxygen_indices: Sequence[int],
) -> tuple[str, set[int]]:
    path = _carbon_path_between(mol, Chem, first_idx, second_idx)
    if not path:
        return "", set()
    path_set = set(path)
    substituents: dict[int, list[int]] = {idx: [] for idx in path}
    for idx in path:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in path_set or nidx in oxygen_indices or nbr.GetAtomicNum() <= 1:
                continue
            bond = mol.GetBondBetweenAtoms(idx, nidx)
            if nbr.GetSymbol() != "C" or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return "", set()
            substituents[idx].append(nidx)
    if _is_neopentylene_phosphite_bridge(mol, Chem, path, substituents):
        covered = set(path)
        for extra in substituents.get(path[1], []):
            covered.add(extra)
        return "neopentylene", covered
    if any(substituents.values()):
        return "", set()
    root = ROOTS.get(len(path))
    if not root:
        return "", set()
    return f"1,{len(path)}-{root}anediyl", set(path)


def _is_neopentylene_phosphite_bridge(mol, Chem, path: Sequence[int], substituents: dict[int, list[int]]) -> bool:
    if len(path) != 3:
        return False
    middle = path[1]
    if len(substituents.get(middle, ())) != 2:
        return False
    if any(substituents.get(idx) for idx in (path[0], path[2])):
        return False
    for extra in substituents[middle]:
        atom = mol.GetAtomWithIdx(extra)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
            return False
        if _heavy_degree_excluding(mol, extra, middle) != 0:
            return False
    return True


def _carbon_path_between(mol, Chem, first_idx: int, second_idx: int) -> list[int]:
    queue: list[tuple[int, list[int]]] = [(first_idx, [first_idx])]
    seen = {first_idx}
    while queue:
        current, path = queue.pop(0)
        if current == second_idx:
            return path
        current_atom = mol.GetAtomWithIdx(current)
        if current_atom.GetSymbol() != "C":
            return []
        for nbr in current_atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in seen or nbr.GetSymbol() != "C":
                continue
            bond = mol.GetBondBetweenAtoms(current, nidx)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return []
            seen.add(nidx)
            queue.append((nidx, path + [nidx]))
    return []


def _linear_saturated_carbon_chain_from_terminal(mol, Chem, start_idx: int, blocked_idx: int) -> list[int]:
    path = []
    previous = blocked_idx
    current = start_idx
    seen = {blocked_idx}
    while True:
        if current in seen:
            return []
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
            return []
        path.append(current)
        seen.add(current)
        next_atoms = []
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == previous or nbr.GetAtomicNum() <= 1:
                continue
            bond = mol.GetBondBetweenAtoms(current, nidx)
            if nbr.GetSymbol() != "C" or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return []
            next_atoms.append(nidx)
        if not next_atoms:
            return path
        if len(next_atoms) != 1:
            return []
        previous, current = current, next_atoms[0]


def _phosphite_ester_name_from_records(mol, Chem, owner_records: Sequence[tuple[str, int, int]]) -> str:
    owner_names = [owner for owner, _, _ in owner_records]
    internal_records = [_phosphite_internal_stereo_owner_name(mol, Chem, record) for record in owner_records]
    if all(internal_records):
        return _phosphite_ester_name(internal_records)
    return _phosphite_ester_name(owner_names)


def _phosphite_internal_stereo_owner_name(mol, Chem, owner_record: tuple[str, int, int]) -> str:
    owner, oxygen_idx, owner_idx = owner_record
    match = re.fullmatch(r"(\d+)-([A-Za-z]+)en-1-yl(.*)", owner)
    if not match:
        return ""
    descriptor = _single_component_alkene_stereo_descriptor(mol, Chem, owner_idx, oxygen_idx)
    if not descriptor:
        return ""
    return f"({match.group(1)}{descriptor})-{owner}"


def _single_component_alkene_stereo_descriptor(mol, Chem, start_idx: int, blocked_idx: int) -> str:
    component_atoms = _collect_component_atoms(mol, start_idx, blocked_idx)
    descriptors = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx not in component_atoms or end_idx not in component_atoms:
            continue
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetBeginAtom().GetSymbol() != "C" or bond.GetEndAtom().GetSymbol() != "C":
            continue
        descriptor = _bond_stereo_descriptor(Chem, bond)
        if not descriptor:
            return ""
        descriptors.append(descriptor)
    if len(descriptors) != 1:
        return ""
    return descriptors[0]


def _mixed_p3_anionic_phosphorus_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for phosphorus in mol.GetAtoms():
        if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0 or phosphorus.IsInRing():
            continue
        pidx = phosphorus.GetIdx()
        terminal_anionic_oxygens = []
        terminal_anionic_sulfurs = []
        carbon_ligands = []
        oxygen_ligands = []
        sulfur_ligands = []
        halogen_ligands = []
        terminal_amino_ligands = []
        substituted_amino_ligands = []
        covered = {pidx}
        ok = True
        for nbr in phosphorus.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(pidx, nidx)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                ok = False
                break
            heavy = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != pidx and atom.GetAtomicNum() > 1
            ]
            if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() < 0 and not heavy:
                terminal_anionic_oxygens.append(nidx)
                covered.add(nidx)
            elif nbr.GetSymbol() == "S" and nbr.GetFormalCharge() < 0 and not heavy:
                terminal_anionic_sulfurs.append(nidx)
                covered.add(nidx)
            elif nbr.GetSymbol() == "C" and nbr.GetFormalCharge() == 0:
                carbon_ligands.append(nidx)
                covered.update(_collect_component_atoms(mol, nidx, pidx))
            elif nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and len(heavy) == 1:
                oxygen_ligands.append(nidx)
                covered.update(_collect_component_atoms(mol, nidx, pidx))
            elif nbr.GetSymbol() == "S" and nbr.GetFormalCharge() == 0 and len(heavy) == 1:
                sulfur_ligands.append(nidx)
                covered.update(_collect_component_atoms(mol, nidx, pidx))
            elif nbr.GetSymbol() in HALO and nbr.GetFormalCharge() == 0 and not heavy:
                halogen_ligands.append(nbr.GetSymbol())
                covered.add(nidx)
            elif nbr.GetSymbol() == "N" and nbr.GetFormalCharge() == 0 and not heavy:
                terminal_amino_ligands.append(nidx)
                covered.add(nidx)
            elif nbr.GetSymbol() == "N" and nbr.GetFormalCharge() == 0 and len(heavy) in {1, 2}:
                substituted_amino_ligands.append(nidx)
                covered.update(_collect_component_atoms(mol, nidx, pidx))
            else:
                ok = False
                break
        if not ok or covered != atom_set:
            continue

        anionic_oxygen_count = len(terminal_anionic_oxygens)
        anionic_sulfur_count = len(terminal_anionic_sulfurs)
        if (
            anionic_oxygen_count == 0
            and anionic_sulfur_count == 0
            and len(carbon_ligands) == 1
            and not oxygen_ligands
            and not sulfur_ligands
            and len(halogen_ligands) == 1
            and not terminal_amino_ligands
            and len(substituted_amino_ligands) == 1
            and phosphorus.GetTotalNumHs() == 0
        ):
            carbon = _strip_outer_parentheses(_phosphorus_attached_prefix(mol, carbon_ligands[0], pidx))
            amino_prefix = _phosphoramidite_substituted_amino_prefix(mol, substituted_amino_ligands[0], pidx)
            halo = HALO.get(halogen_ligands[0], "")
            if (
                carbon
                and amino_prefix
                and halo
                and _simple_unparenthesized_n_substituent(carbon)
                and _stable_phosphoramidite_amino_prefix(amino_prefix)
                and amino_prefix.startswith("di")
            ):
                names.append(f"{halo}({amino_prefix}amino)({carbon})phosphine")
            continue

        if (
            anionic_oxygen_count == 1
            and anionic_sulfur_count == 0
            and len(carbon_ligands) == 2
            and not oxygen_ligands
            and not sulfur_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            carbon_names = [
                _strip_outer_parentheses(_phosphorus_attached_prefix(mol, ligand_idx, pidx))
                for ligand_idx in carbon_ligands
            ]
            ligand_text = _phosphinite_ligand_text(carbon_names)
            if ligand_text:
                names.append(f"{ligand_text}phosphinite")
            continue

        if (
            anionic_oxygen_count == 1
            and anionic_sulfur_count == 0
            and not carbon_ligands
            and len(oxygen_ligands) == 1
            and not sulfur_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 1
        ):
            oxygen_owner = _single_external_substituent_name(mol, oxygen_ligands[0], pidx)
            if oxygen_owner and not _unstable_retained_functional_parent_substituent(oxygen_owner):
                names.append(f"{_parenthesized_if_complex(oxygen_owner)} phosphonite")
            continue

        if (
            anionic_oxygen_count == 2
            and anionic_sulfur_count == 0
            and len(carbon_ligands) == 1
            and not oxygen_ligands
            and not sulfur_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
        ):
            carbon = _phosphorus_attached_prefix(mol, carbon_ligands[0], pidx)
            if carbon:
                if carbon == "cyano":
                    names.append("cyanophosphite")
                else:
                    names.append(carbon + "phosphonite")
            continue

        if (
            anionic_oxygen_count == 2
            and anionic_sulfur_count == 0
            and len(carbon_ligands) == 1
            and len(oxygen_ligands) == 1
            and not sulfur_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 1
        ):
            oxygen_owner = _single_external_substituent_name(mol, oxygen_ligands[0], pidx)
            carbon = _phosphorus_attached_prefix(mol, carbon_ligands[0], pidx)
            if oxygen_owner and carbon and not _unstable_retained_functional_parent_substituent(oxygen_owner):
                names.append(f"{_parenthesized_if_complex(oxygen_owner)} {carbon}phosphite")
            continue

        if (
            anionic_oxygen_count == 0
            and anionic_sulfur_count == 2
            and not carbon_ligands
            and not oxygen_ligands
            and len(sulfur_ligands) == 1
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
        ):
            sulfur_owner = _single_external_substituent_name(mol, sulfur_ligands[0], pidx)
            if sulfur_owner and not _unstable_retained_functional_parent_substituent(sulfur_owner):
                names.append(f"{_parenthesized_if_complex(sulfur_owner)} trithiophosphite")
            continue

        if (
            anionic_oxygen_count == 0
            and anionic_sulfur_count == 1
            and not carbon_ligands
            and not oxygen_ligands
            and len(sulfur_ligands) == 2
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
        ):
            sulfur_owners = _hetero_ligand_owner_names(mol, sulfur_ligands, pidx)
            ligand_text = _phosphorus_repeated_ligand_text(sulfur_owners)
            if ligand_text and not any(_unstable_retained_functional_parent_substituent(name) for name in sulfur_owners):
                names.append(f"{ligand_text} trithiophosphite")
            continue

        if (
            anionic_oxygen_count == 1
            and anionic_sulfur_count == 1
            and not carbon_ligands
            and not oxygen_ligands
            and len(sulfur_ligands) == 1
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            sulfur_owner = _single_external_substituent_name(mol, sulfur_ligands[0], pidx)
            if sulfur_owner and not _unstable_retained_functional_parent_substituent(sulfur_owner):
                names.append(f"{_parenthesized_if_complex(sulfur_owner)} dithiophosphite")
            continue

        if (
            anionic_oxygen_count == 1
            and anionic_sulfur_count == 0
            and not carbon_ligands
            and not oxygen_ligands
            and not sulfur_ligands
            and len(halogen_ligands) == 1
            and not terminal_amino_ligands
            and len(substituted_amino_ligands) == 1
            and phosphorus.GetTotalNumHs() == 0
        ):
            name = _halophosphoramidite_name(
                mol,
                substituted_amino_ligands[0],
                pidx,
                halogen_ligands[0],
            )
            if name:
                names.append(name)
            continue

        if (
            anionic_oxygen_count == 1
            and anionic_sulfur_count == 0
            and not carbon_ligands
            and not oxygen_ligands
            and not sulfur_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and len(substituted_amino_ligands) == 2
            and phosphorus.GetTotalNumHs() == 0
        ):
            name = _symmetric_phosphorodiamidite_name(
                [
                    _phosphoramidite_substituted_amino_prefix(mol, ligand_idx, pidx)
                    for ligand_idx in substituted_amino_ligands
                ]
            )
            if name:
                names.append(name)
            continue

        if anionic_oxygen_count != 1:
            continue
        if anionic_sulfur_count != 0:
            continue
        if (
            len(carbon_ligands) == 1
            and len(oxygen_ligands) == 2
            and not sulfur_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
        ):
            oxygen_owners = _hetero_ligand_owner_names(mol, oxygen_ligands, pidx)
            carbon = _phosphorus_attached_prefix(mol, carbon_ligands[0], pidx)
            ligand_text = _phosphorus_repeated_ligand_text(oxygen_owners)
            if ligand_text and carbon and not any(_unstable_retained_functional_parent_substituent(name) for name in oxygen_owners):
                names.append(f"{ligand_text} {carbon}phosphite")
            continue
        if (
            len(carbon_ligands) == 1
            and len(oxygen_ligands) == 1
            and not sulfur_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
        ):
            oxygen_owner = _single_external_substituent_name(mol, oxygen_ligands[0], pidx)
            carbon = _phosphorus_attached_prefix(mol, carbon_ligands[0], pidx)
            if oxygen_owner and carbon and not _unstable_retained_functional_parent_substituent(oxygen_owner):
                names.append(f"{_hetero_ligand_role_text('O', oxygen_owner)} {carbon}phosphonite")
            continue
        if (
            not carbon_ligands
            and len(oxygen_ligands) == 1
            and len(sulfur_ligands) == 1
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
        ):
            oxygen_owner = _single_external_substituent_name(mol, oxygen_ligands[0], pidx)
            sulfur_owner = _single_external_substituent_name(mol, sulfur_ligands[0], pidx)
            if (
                oxygen_owner
                and sulfur_owner
                and not _unstable_retained_functional_parent_substituent(oxygen_owner)
                and not _unstable_retained_functional_parent_substituent(sulfur_owner)
            ):
                if oxygen_owner == sulfur_owner and _simple_unparenthesized_n_substituent(oxygen_owner):
                    names.append(f"di{oxygen_owner} thiophosphite")
                else:
                    names.append(
                        f"{_hetero_ligand_role_text('O', oxygen_owner)} "
                        f"{_hetero_ligand_role_text('S', sulfur_owner)} phosphorothioite"
                    )
            continue
        if (
            not carbon_ligands
            and not oxygen_ligands
            and len(sulfur_ligands) == 2
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
        ):
            sulfur_owners = _hetero_ligand_owner_names(mol, sulfur_ligands, pidx)
            ligand_text = _phosphorus_repeated_ligand_text(sulfur_owners)
            if ligand_text and not any(_unstable_retained_functional_parent_substituent(name) for name in sulfur_owners):
                names.append(f"{ligand_text} dithiophosphite")
            continue
        if (
            not carbon_ligands
            and len(oxygen_ligands) == 1
            and not sulfur_ligands
            and len(halogen_ligands) == 1
            and not terminal_amino_ligands
            and not substituted_amino_ligands
        ):
            oxygen_owner = _single_external_substituent_name(mol, oxygen_ligands[0], pidx)
            halo = HALO.get(halogen_ligands[0], "")
            if oxygen_owner and halo and not _unstable_retained_functional_parent_substituent(oxygen_owner):
                names.append(f"{_parenthesized_if_complex(oxygen_owner)} mono{halo}phosphite")
            continue
        if (
            not carbon_ligands
            and len(oxygen_ligands) == 1
            and not sulfur_ligands
            and not halogen_ligands
            and len(terminal_amino_ligands) == 1
            and not substituted_amino_ligands
        ):
            oxygen_owner = _single_external_substituent_name(mol, oxygen_ligands[0], pidx)
            if oxygen_owner and not _unstable_retained_functional_parent_substituent(oxygen_owner):
                names.append(f"{_parenthesized_if_complex(oxygen_owner)} phosphoramidite")
            continue
        if (
            not carbon_ligands
            and len(oxygen_ligands) == 1
            and not sulfur_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and len(substituted_amino_ligands) == 1
        ):
            oxygen_owner = _single_external_substituent_name(mol, oxygen_ligands[0], pidx)
            amino_prefix = _phosphoramidite_substituted_amino_prefix(
                mol,
                substituted_amino_ligands[0],
                pidx,
                allow_explicit_n_parent=True,
            )
            if (
                oxygen_owner
                and amino_prefix
                and not _unstable_retained_functional_parent_substituent(oxygen_owner)
            ):
                names.append(f"{_parenthesized_if_complex(oxygen_owner)} {amino_prefix}phosphoramidite")
    return names


def _phosphinite_ligand_text(names: Sequence[str]) -> str:
    if len(names) != 2 or any(not name for name in names):
        return ""
    first, second = sorted(_strip_outer_parentheses(name) for name in names)
    if first == second:
        if _simple_unparenthesized_n_substituent(first):
            return f"di{first}"
        return f"bis({first})"
    return "".join(_parenthesized_if_complex(name) for name in (first, second))


def _phosphinate_carbon_ligand_text(mol, carbon_ligands: Sequence[int], phosphorus_idx: int) -> str:
    if len(carbon_ligands) != 2:
        return ""
    records = []
    names = []
    for ligand_idx in carbon_ligands:
        name = _strip_outer_parentheses(_phosphorus_attached_prefix(mol, ligand_idx, phosphorus_idx))
        if not name:
            return ""
        names.append(name)
        records.append((mol.GetAtomWithIdx(ligand_idx).IsInRing(), _needs_substituent_parentheses(name), name))
    if names[0] == names[1]:
        return _phosphinite_ligand_text(names)
    ordered = [name for _, _, name in sorted(records)]
    return "".join(_parenthesized_if_complex(name) for name in ordered)


def _trioxido_phosphorane_carbon_ligand_text(mol, carbon_ligands: Sequence[int], phosphorus_idx: int) -> str:
    if len(carbon_ligands) != 2:
        return ""
    records = []
    for ligand_idx in carbon_ligands:
        name = _strip_outer_parentheses(_phosphorus_attached_prefix(mol, ligand_idx, phosphorus_idx))
        if not name:
            return ""
        records.append((mol.GetAtomWithIdx(ligand_idx).IsInRing(), _needs_substituent_parentheses(name), name))
    return "".join(f"({name})" for _, _, name in sorted(records))


def _hetero_ligand_owner_names(mol, ligand_atoms: Sequence[int], phosphorus_idx: int) -> list[str]:
    names = []
    for ligand_idx in ligand_atoms:
        owner = _single_external_substituent_name(mol, ligand_idx, phosphorus_idx)
        if not owner:
            return []
        names.append(owner)
    return names


def _hetero_ligand_role_text(role: str, owner: str) -> str:
    bare = _strip_outer_parentheses(owner)
    if _needs_substituent_parentheses(bare):
        return f"{role}-({bare})"
    return f"{role}-{bare}"


def _phosphorothioate_oxygen_ligand_text(owner_names: Sequence[str]) -> str:
    if len(owner_names) != 2 or any(not name for name in owner_names):
        return ""
    first, second = sorted(owner_names)
    if first == second and _simple_unparenthesized_n_substituent(first):
        return f"O,O-di{_strip_outer_parentheses(first)}"
    return " ".join(_hetero_ligand_role_text("O", owner) for owner in (first, second))


def _phosphorothioate_sulfur_parent_text(owner: str, parent: str) -> str:
    if not owner or _unstable_retained_functional_parent_substituent(owner):
        return ""
    bare = _strip_outer_parentheses(owner)
    if _needs_substituent_parentheses(bare):
        return f"({bare}){parent}"
    return f"{bare}{parent}"


def _phosphoramidite_substituted_amino_prefix(
    mol,
    nitrogen_idx: int,
    phosphorus_idx: int,
    *,
    allow_explicit_n_parent: bool = False,
) -> str:
    substituents = _external_substituent_names(mol, nitrogen_idx, phosphorus_idx)
    if substituents is None or not substituents or len(substituents) > 2:
        return ""
    substituents = [_retained_parent_substituent_alias(name) for name in substituents]
    if (
        len(substituents) == 2
        and substituents[0] == substituents[1]
        and _simple_unparenthesized_n_substituent(substituents[0])
    ):
        return f"di{substituents[0]}"
    prefix = _n_substituent_prefix(substituents)
    if prefix is None and allow_explicit_n_parent:
        prefix = _phosphoramidite_explicit_n_substituent_prefix(substituents)
    if not prefix or _needs_explicit_n_parent_separator(prefix):
        if not (allow_explicit_n_parent and prefix):
            return ""
    return prefix


def _symmetric_phosphorodiamidite_name(amino_prefixes: Sequence[str]) -> str:
    if len(amino_prefixes) != 2 or amino_prefixes[0] != amino_prefixes[1]:
        return ""
    prefix = amino_prefixes[0]
    if not _stable_phosphoramidite_amino_prefix(prefix):
        return ""
    bare = _strip_outer_parentheses(prefix)
    if not bare.startswith("di"):
        return ""
    ligand = bare[2:]
    if not re.fullmatch(r"[A-Za-z0-9']+", ligand):
        return ""
    return f"tetra{ligand}phosphorodiamidite"


def _phosphoramidite_explicit_n_substituent_prefix(substituents: Sequence[str]) -> str | None:
    if not substituents or len(substituents) > 2:
        return None
    if not all(_stable_phosphoramidite_explicit_n_substituent(name) for name in substituents):
        return None
    counts = Counter(substituents)
    if len(counts) == 1:
        name = next(iter(counts))
        count = counts[name]
        if count == 1:
            return _n_substituent_piece("N", name)
        if count == 2:
            return _duplicate_n_substituent_piece("N,N", name)
        return ""
    first, second = sorted(substituents)
    return f"{_n_substituent_piece('N', first)}-{_n_substituent_piece('N', second)}"


def _stable_phosphoramidite_explicit_n_substituent(name: str) -> bool:
    if _allowed_n_retained_parent_substituent(name):
        return True
    if not name or len(name) > 72:
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "carbam", "urea")):
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    return "cyano" in name and name.endswith("yl")


def _halophosphoramidite_name(mol, nitrogen_idx: int, phosphorus_idx: int, halogen: str) -> str:
    amino_prefix = _phosphoramidite_substituted_amino_prefix(
        mol,
        nitrogen_idx,
        phosphorus_idx,
        allow_explicit_n_parent=True,
    )
    halo = HALO.get(halogen, "")
    if not amino_prefix or not halo or not _stable_phosphoramidite_amino_prefix(amino_prefix):
        return ""
    return f"{amino_prefix}{halo}phosphoramidite"


def _phosphoramidohalidate_name(mol, nitrogen_idx: int, phosphorus_idx: int, halogen: str) -> str:
    amino_prefix = _phosphoramidite_substituted_amino_prefix(mol, nitrogen_idx, phosphorus_idx)
    halo = HALO.get(halogen, "")
    if not amino_prefix or not halo or not _stable_phosphoramidite_amino_prefix(amino_prefix):
        return ""
    return f"{amino_prefix}phosphoramido{halo[:-1]}idate"


def _stable_phosphoramidite_amino_prefix(prefix: str) -> bool:
    if not prefix or len(prefix) > 96 or " " in prefix or "[" in prefix or "]" in prefix:
        return False
    if prefix.startswith("("):
        return False
    if any(fragment in prefix for fragment in ("lambda", "spiro", "bicyclo")):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9,()'\-]+", prefix))


def _phosphite_ester_name(owner_names: Sequence[str]) -> str:
    first, second = sorted(owner_names)
    if first == second:
        if _simple_unparenthesized_n_substituent(first):
            if first.endswith("oxy"):
                return f"di{first} phosphite"
            return f"di{first}phosphite"
        return f"bis({_strip_outer_parentheses(first)}) phosphite"
    return f"{_parenthesized_if_complex(first)} {_parenthesized_if_complex(second)} phosphite"


def _phosphite_triester_name(owner_names: Sequence[str]) -> str:
    if len(owner_names) != 3 or any(not name for name in owner_names):
        return ""
    counts = Counter(_strip_outer_parentheses(name) for name in owner_names)
    if len(counts) == 1:
        owner = next(iter(counts))
        return f"{_phosphite_repeated_owner_text(owner, 3)} phosphite"
    if len(counts) == 2:
        repeated = next(owner for owner, count in counts.items() if count == 2)
        single = next(owner for owner, count in counts.items() if count == 1)
        parts = [
            _phosphite_repeated_owner_text(repeated, 2),
            _parenthesized_if_complex(single),
        ]
        return " ".join(part for part in parts if part) + " phosphite"
    return " ".join(_parenthesized_if_complex(owner) for owner in sorted(counts)) + " phosphite"


def _phosphite_repeated_owner_text(owner: str, count: int) -> str:
    if count == 2:
        if _simple_unparenthesized_n_substituent(owner) and re.fullmatch(r"[A-Za-z0-9']+", owner):
            return f"di{owner}"
        return f"bis({owner})"
    if count == 3:
        if _simple_unparenthesized_n_substituent(owner) and re.fullmatch(r"[A-Za-z0-9']+", owner):
            return f"tri{owner}"
        return f"tris({owner})"
    return ""


def _phosphite_monoester_name(owner: str, hydroxy_count: int) -> str:
    if hydroxy_count not in {0, 1, 2}:
        return ""
    hydrogen_text = {0: "", 1: "hydrogen", 2: "dihydrogen"}[hydroxy_count]
    return " ".join(part for part in (_parenthesized_if_complex(owner), hydrogen_text, "phosphite") if part)


def _charged_organophosphorus_oxo_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for phosphorus in mol.GetAtoms():
        if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0 or phosphorus.IsInRing():
            continue
        pidx = phosphorus.GetIdx()
        terminal_anionic_oxygens = []
        terminal_anionic_sulfurs = []
        terminal_double_oxygens = []
        terminal_double_sulfurs = []
        substituted_double_sulfurs = []
        carbon_single_ligands = []
        carbon_double_ligands = []
        oxygen_ligands = []
        sulfur_ligands = []
        halogen_ligands = []
        terminal_amino_ligands = []
        substituted_amino_ligands = []
        covered = {pidx}
        ok = True
        for nbr in phosphorus.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(pidx, nidx)
            if bond is None:
                ok = False
                break
            bond_type = bond.GetBondType()
            heavy = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != pidx and atom.GetAtomicNum() > 1
            ]
            if (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() < 0
                and bond_type == Chem.BondType.SINGLE
                and not heavy
            ):
                terminal_anionic_oxygens.append(nidx)
                covered.add(nidx)
            elif nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and bond_type == Chem.BondType.DOUBLE:
                terminal_double_oxygens.append(nidx)
                covered.add(nidx)
            elif nbr.GetSymbol() == "S" and nbr.GetFormalCharge() == 0 and bond_type == Chem.BondType.DOUBLE:
                if not heavy:
                    terminal_double_sulfurs.append(nidx)
                    covered.add(nidx)
                elif len(heavy) == 1:
                    substituted_double_sulfurs.append(nidx)
                    covered.add(nidx)
                    covered.update(_collect_component_atoms(mol, nidx, pidx))
                else:
                    ok = False
                    break
            elif (
                nbr.GetSymbol() == "S"
                and nbr.GetFormalCharge() < 0
                and bond_type == Chem.BondType.SINGLE
                and not heavy
            ):
                terminal_anionic_sulfurs.append(nidx)
                covered.add(nidx)
            elif (
                nbr.GetSymbol() == "S"
                and nbr.GetFormalCharge() == 0
                and bond_type == Chem.BondType.SINGLE
                and len(heavy) == 1
            ):
                sulfur_ligands.append(nidx)
                covered.update(_collect_component_atoms(mol, nidx, pidx))
            elif nbr.GetSymbol() == "C" and nbr.GetFormalCharge() == 0 and bond_type == Chem.BondType.SINGLE:
                carbon_single_ligands.append(nidx)
                covered.update(_collect_component_atoms(mol, nidx, pidx))
            elif nbr.GetSymbol() == "C" and nbr.GetFormalCharge() == 0 and bond_type == Chem.BondType.DOUBLE:
                carbon_double_ligands.append(nidx)
                covered.update(_collect_component_atoms(mol, nidx, pidx))
            elif (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() == 0
                and bond_type == Chem.BondType.SINGLE
                and len(heavy) == 1
            ):
                oxygen_ligands.append(nidx)
                covered.update(_collect_component_atoms(mol, nidx, pidx))
            elif (
                nbr.GetSymbol() in HALO
                and nbr.GetFormalCharge() == 0
                and bond_type == Chem.BondType.SINGLE
                and not heavy
            ):
                halogen_ligands.append(nbr.GetSymbol())
                covered.add(nidx)
            elif (
                nbr.GetSymbol() == "N"
                and nbr.GetFormalCharge() == 0
                and bond_type == Chem.BondType.SINGLE
                and not heavy
            ):
                terminal_amino_ligands.append(nidx)
                covered.add(nidx)
            elif (
                nbr.GetSymbol() == "N"
                and nbr.GetFormalCharge() == 0
                and bond_type == Chem.BondType.SINGLE
                and len(heavy) in {1, 2}
            ):
                substituted_amino_ligands.append(nidx)
                covered.update(_collect_component_atoms(mol, nidx, pidx))
            else:
                ok = False
                break
        if not ok or covered != atom_set:
            continue

        double_oxygen_count = len(terminal_double_oxygens)
        double_sulfur_count = len(terminal_double_sulfurs)
        anionic_oxygen_count = len(terminal_anionic_oxygens)
        anionic_sulfur_count = len(terminal_anionic_sulfurs)
        if (
            double_sulfur_count == 1
            and double_oxygen_count == 0
            and anionic_oxygen_count == 2
            and len(carbon_single_ligands) == 1
            and not carbon_double_ligands
            and not oxygen_ligands
            and not sulfur_ligands
            and not halogen_ligands
            and anionic_sulfur_count == 0
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            prefix = _phosphorus_attached_prefix(mol, carbon_single_ligands[0], pidx)
            if prefix:
                names.append(prefix + "phosphonothioate")
            continue

        if (
            double_sulfur_count == 1
            and double_oxygen_count == 0
            and anionic_oxygen_count == 1
            and len(carbon_single_ligands) == 1
            and not carbon_double_ligands
            and not oxygen_ligands
            and not sulfur_ligands
            and not halogen_ligands
            and anionic_sulfur_count == 0
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 1
        ):
            prefix = _phosphorus_attached_prefix(mol, carbon_single_ligands[0], pidx)
            if prefix:
                names.append(f"{_parenthesized_if_complex(prefix)}phosphinothioate")
            continue

        if (
            double_sulfur_count == 1
            and double_oxygen_count == 0
            and anionic_oxygen_count == 1
            and len(carbon_single_ligands) == 2
            and not carbon_double_ligands
            and not oxygen_ligands
            and not sulfur_ligands
            and not halogen_ligands
            and anionic_sulfur_count == 0
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            ligand_text = _phosphinate_carbon_ligand_text(mol, carbon_single_ligands, pidx)
            if ligand_text:
                names.append(ligand_text + "phosphinothioate")
            continue

        if (
            double_sulfur_count == 1
            and double_oxygen_count == 0
            and anionic_oxygen_count == 1
            and len(carbon_single_ligands) == 1
            and not carbon_double_ligands
            and not oxygen_ligands
            and not sulfur_ligands
            and len(halogen_ligands) == 1
            and anionic_sulfur_count == 0
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            carbon = _phosphorus_attached_prefix(mol, carbon_single_ligands[0], pidx)
            halo = HALO.get(halogen_ligands[0], "")
            if carbon and halo:
                names.append(f"{carbon}{halo}phosphinothioate")
            continue

        if (
            double_sulfur_count == 1
            and double_oxygen_count == 0
            and anionic_oxygen_count == 0
            and len(carbon_single_ligands) == 2
            and not carbon_double_ligands
            and not oxygen_ligands
            and not sulfur_ligands
            and not halogen_ligands
            and anionic_sulfur_count == 1
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            ligand_text = _phosphinate_carbon_ligand_text(mol, carbon_single_ligands, pidx)
            if ligand_text:
                names.append(ligand_text + "phosphinodithioate")
            continue

        if (
            double_sulfur_count == 1
            and double_oxygen_count == 0
            and anionic_oxygen_count == 0
            and len(carbon_single_ligands) == 2
            and not carbon_double_ligands
            and not oxygen_ligands
            and len(sulfur_ligands) == 1
            and not halogen_ligands
            and anionic_sulfur_count == 0
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            ligand_text = _phosphinate_carbon_ligand_text(mol, carbon_single_ligands, pidx)
            sulfur_owner = _single_external_substituent_name(mol, sulfur_ligands[0], pidx)
            if (
                ligand_text
                and sulfur_owner
                and not _unstable_retained_functional_parent_substituent(sulfur_owner)
            ):
                names.append(f"{_hetero_ligand_role_text('S', sulfur_owner)} {ligand_text}phosphinodithioate")
            continue

        if (
            double_sulfur_count == 0
            and len(substituted_double_sulfurs) == 1
            and double_oxygen_count == 0
            and anionic_oxygen_count == 1
            and anionic_sulfur_count == 0
            and len(oxygen_ligands) == 2
            and not carbon_single_ligands
            and not carbon_double_ligands
            and not sulfur_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            oxygen_owners = _hetero_ligand_owner_names(mol, oxygen_ligands, pidx)
            sulfur_owner = _single_external_substituent_name(mol, substituted_double_sulfurs[0], pidx)
            oxygen_text = _phosphorothioate_oxygen_ligand_text(oxygen_owners)
            sulfur_parent = _phosphorothioate_sulfur_parent_text(sulfur_owner, "phosphorothioate")
            if (
                oxygen_text
                and sulfur_parent
                and not any(_unstable_retained_functional_parent_substituent(name) for name in oxygen_owners)
            ):
                names.append(f"{oxygen_text} {sulfur_parent}")
            continue

        if substituted_double_sulfurs:
            continue

        if double_sulfur_count:
            continue

        if (
            double_oxygen_count == 1
            and anionic_oxygen_count == 1
            and len(carbon_single_ligands) == 2
            and not carbon_double_ligands
            and not oxygen_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            ligand_text = _phosphinate_carbon_ligand_text(mol, carbon_single_ligands, pidx)
            if ligand_text:
                names.append(ligand_text + "phosphinate")
            continue

        if (
            double_oxygen_count == 1
            and anionic_oxygen_count == 2
            and len(carbon_single_ligands) == 1
            and not carbon_double_ligands
            and not oxygen_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            prefix = _phosphorus_attached_prefix(mol, carbon_single_ligands[0], pidx)
            if prefix:
                names.append(prefix + "phosphonate")
            continue

        if (
            double_oxygen_count == 1
            and anionic_oxygen_count == 1
            and len(carbon_single_ligands) == 1
            and len(terminal_amino_ligands) == 1
            and not carbon_double_ligands
            and not oxygen_ligands
            and not halogen_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            prefix = _phosphorus_attached_prefix(mol, carbon_single_ligands[0], pidx)
            if prefix:
                names.append(prefix + "phosphonamidate")
            continue

        if (
            double_oxygen_count == 1
            and anionic_oxygen_count == 1
            and len(carbon_single_ligands) == 1
            and not carbon_double_ligands
            and not oxygen_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 1
        ):
            prefix = _phosphorus_attached_prefix(mol, carbon_single_ligands[0], pidx)
            if prefix:
                names.append(f"{_parenthesized_if_complex(prefix)}phosphinate")
            continue

        if (
            double_oxygen_count == 1
            and anionic_oxygen_count == 1
            and not carbon_single_ligands
            and not carbon_double_ligands
            and not oxygen_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and len(substituted_amino_ligands) == 1
            and phosphorus.GetTotalNumHs() == 1
        ):
            substituents = _external_substituent_names(mol, substituted_amino_ligands[0], pidx)
            prefix = _n_substituent_prefix(substituents or ())
            if prefix is not None:
                names.append(prefix + "phosphonamidate")
            continue

        if (
            double_oxygen_count == 1
            and anionic_oxygen_count == 1
            and len(oxygen_ligands) == 1
            and not carbon_single_ligands
            and not carbon_double_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 1
        ):
            owner = _single_external_substituent_name(mol, oxygen_ligands[0], pidx)
            if owner and not _unstable_retained_functional_parent_substituent(owner):
                names.append(f"{_parenthesized_if_complex(owner)} phosphonate")
            continue

        if (
            double_oxygen_count == 1
            and anionic_oxygen_count == 1
            and not carbon_single_ligands
            and not carbon_double_ligands
            and not oxygen_ligands
            and len(halogen_ligands) == 1
            and not terminal_amino_ligands
            and len(substituted_amino_ligands) == 1
            and phosphorus.GetTotalNumHs() == 0
        ):
            name = _phosphoramidohalidate_name(
                mol,
                substituted_amino_ligands[0],
                pidx,
                halogen_ligands[0],
            )
            if name:
                names.append(name)
            continue

        if (
            double_oxygen_count == 0
            and anionic_oxygen_count == 3
            and len(carbon_single_ligands) == 2
            and not carbon_double_ligands
            and not oxygen_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            ligand_text = _trioxido_phosphorane_carbon_ligand_text(mol, carbon_single_ligands, pidx)
            if ligand_text:
                names.append(f"trioxido{ligand_text}phosphorane")
            continue

        if (
            double_oxygen_count == 0
            and anionic_oxygen_count == 3
            and len(carbon_single_ligands) == 1
            and not carbon_double_ligands
            and not oxygen_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 1
        ):
            ligand_atoms = _collect_component_atoms(mol, carbon_single_ligands[0], pidx)
            if any(mol.GetAtomWithIdx(idx).GetSymbol() == "P" for idx in ligand_atoms):
                continue
            prefix = _phosphorus_attached_prefix(mol, carbon_single_ligands[0], pidx)
            if prefix:
                ligand = prefix if prefix.startswith("(") else f"({prefix})"
                names.append(f"hydrido(trioxido){ligand}phosphorane")
            continue

        if (
            double_oxygen_count == 0
            and anionic_oxygen_count == 3
            and len(carbon_double_ligands) == 1
            and not carbon_single_ligands
            and not oxygen_ligands
            and not halogen_ligands
            and not terminal_amino_ligands
            and not substituted_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            prefix = _phosphorus_attached_prefix(mol, carbon_double_ligands[0], pidx)
            if prefix:
                names.append(prefix + "phosphite")
    return names


def _phosphate_ester_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for phosphorus in mol.GetAtoms():
        if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0 or phosphorus.IsInRing():
            continue
        record = _phosphate_ester_record(mol, Chem, phosphorus.GetIdx())
        if not record or record["covered_atoms"] != atom_set:
            continue
        name = _phosphate_ester_record_name(record)
        if name:
            names.append(name)
    return sorted(set(names), key=lambda name: (len(name), name))


def _phosphate_ester_record(mol, Chem, pidx: int) -> dict | None:
    phosphorus = mol.GetAtomWithIdx(pidx)
    double_oxygen = 0
    terminal_oxido = 0
    terminal_hydroxy = 0
    owner_names = []
    carbon_names = []
    covered_atoms = {pidx}
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            if nbr.GetFormalCharge() != 0 or _heavy_degree_excluding(mol, nidx, pidx) != 0:
                return None
            double_oxygen += 1
            covered_atoms.add(nidx)
            continue
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            heavy = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != pidx and atom.GetAtomicNum() > 1
            ]
            if not heavy:
                if nbr.GetFormalCharge() < 0:
                    terminal_oxido += 1
                elif nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                    terminal_hydroxy += 1
                else:
                    return None
                covered_atoms.add(nidx)
                continue
            if len(heavy) != 1 or nbr.GetFormalCharge() != 0:
                return None
            owner = _single_external_substituent_name(mol, nidx, pidx)
            if not _stable_phosphate_ester_owner(owner):
                return None
            owner_atoms = _collect_component_atoms(mol, heavy[0], nidx)
            if not _stable_phosphate_oxygen_owner_component(mol, Chem, owner_atoms):
                return None
            if _phosphate_owner_has_nonaromatic_carbon_alkoxy_phosphoryloxy(mol, Chem, owner_atoms):
                return None
            owner_names.append(owner)
            covered_atoms.add(nidx)
            covered_atoms.update(owner_atoms)
            continue
        if nbr.GetSymbol() == "C" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon = _phosphorus_attached_prefix(mol, nidx, pidx)
            if not _stable_phosphate_ester_owner(carbon):
                return None
            carbon_atoms = _collect_component_atoms(mol, nidx, pidx)
            if any(mol.GetAtomWithIdx(atom_idx).GetSymbol() == "P" for atom_idx in carbon_atoms):
                return None
            carbon_names.append(_strip_outer_parentheses(carbon))
            covered_atoms.update(carbon_atoms)
            continue
        return None
    return {
        "phosphorus_hydrogens": phosphorus.GetTotalNumHs(),
        "double_oxygen": double_oxygen,
        "terminal_oxido": terminal_oxido,
        "terminal_hydroxy": terminal_hydroxy,
        "owner_names": tuple(owner_names),
        "carbon_names": tuple(carbon_names),
        "covered_atoms": covered_atoms,
    }


def _stable_phosphate_ester_owner(name: str) -> bool:
    if not name or len(name) > 96:
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9,()'\-]+", name))


def _stable_phosphate_oxygen_owner_component(mol, Chem, atom_indices: set[int]) -> bool:
    nonring_carbonyl_count = 0
    ring_carbonyl_count = 0
    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != "C":
            continue
        has_double_oxygen = _component_carbon_has_double_oxygen(mol, Chem, atom_idx, atom_indices)
        if atom.IsInRing():
            if has_double_oxygen:
                ring_carbonyl_count += 1
            continue
        if has_double_oxygen:
            nonring_carbonyl_count += 1
        double_oxygen = False
        single_acyl_hetero = False
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in atom_indices:
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
            if bond is None:
                continue
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygen = True
            elif nbr.GetSymbol() in {"N", "O", "S"} and bond.GetBondType() == Chem.BondType.SINGLE:
                single_acyl_hetero = True
        if double_oxygen and single_acyl_hetero:
            return False
    if nonring_carbonyl_count > 1:
        return False
    if ring_carbonyl_count > 1 and _component_has_fused_ring(mol, atom_indices):
        return False
    return True


def _phosphate_owner_has_nonaromatic_carbon_alkoxy_phosphoryloxy(mol, Chem, atom_indices: set[int]) -> bool:
    for atom_idx in atom_indices:
        phosphorus = mol.GetAtomWithIdx(atom_idx)
        if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0 or phosphorus.IsInRing():
            continue
        double_oxygen_count = 0
        nonaromatic_carbon_ligand_count = 0
        substituted_oxygen_ligand_count = 0
        for nbr in phosphorus.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in atom_indices:
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
            if bond is None:
                continue
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygen_count += 1
                continue
            if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE and not nbr.GetIsAromatic():
                nonaromatic_carbon_ligand_count += 1
                continue
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
                heavy = [
                    other.GetIdx()
                    for other in nbr.GetNeighbors()
                    if other.GetIdx() != atom_idx and other.GetAtomicNum() > 1
                ]
                if heavy:
                    substituted_oxygen_ligand_count += 1
        if double_oxygen_count == 1 and nonaromatic_carbon_ligand_count and substituted_oxygen_ligand_count:
            return True
    return False


def _component_carbon_has_double_oxygen(mol, Chem, carbon_idx: int, atom_indices: set[int]) -> bool:
    for nbr in mol.GetAtomWithIdx(carbon_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx not in atom_indices or nbr.GetSymbol() != "O":
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
            return True
    return False


def _component_has_fused_ring(mol, atom_indices: set[int]) -> bool:
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings() if set(ring).issubset(atom_indices)]
    for idx, ring in enumerate(rings):
        if any(ring & other for other in rings[idx + 1 :]):
            return True
    return False


def _phosphate_ester_record_name(record: dict) -> str:
    if record["double_oxygen"] != 1:
        return ""
    owner_names = record["owner_names"]
    carbon_names = record["carbon_names"]
    terminal_oxido = record["terminal_oxido"]
    terminal_hydroxy = record["terminal_hydroxy"]
    phosphorus_hydrogens = record["phosphorus_hydrogens"]
    single_oxygen_count = len(owner_names) + terminal_oxido + terminal_hydroxy
    if len(carbon_names) == 0 and phosphorus_hydrogens == 0 and single_oxygen_count == 3:
        return _phosphate_ester_name(owner_names, terminal_hydroxy, "phosphate")
    if len(carbon_names) == 0 and phosphorus_hydrogens == 1 and single_oxygen_count == 2:
        return _phosphate_ester_name(owner_names, terminal_hydroxy, "phosphonate")
    if len(carbon_names) == 1 and phosphorus_hydrogens == 0 and single_oxygen_count == 2:
        return _alkyl_phosphonate_ester_name(owner_names, terminal_hydroxy, carbon_names[0])
    return ""


def _phosphate_ester_name(owner_names: Sequence[str], hydrogen_count: int, parent: str) -> str:
    if hydrogen_count not in {0, 1, 2}:
        return ""
    owner_text = _thiooxo_ester_owner_text(owner_names)
    if not owner_text:
        return ""
    hydrogen_text = {0: "", 1: "hydrogen", 2: "dihydrogen"}[hydrogen_count]
    return " ".join(part for part in (owner_text, hydrogen_text, parent) if part)


def _alkyl_phosphonate_ester_name(owner_names: Sequence[str], hydrogen_count: int, carbon_name: str) -> str:
    if hydrogen_count not in {0, 1}:
        return ""
    owner_text = _thiooxo_ester_owner_text(owner_names)
    if not owner_text:
        return ""
    hydrogen_text = {0: "", 1: "hydrogen"}[hydrogen_count]
    parent = carbon_name + "phosphonate"
    return " ".join(part for part in (owner_text, hydrogen_text, parent) if part)


def _phosphorus_attached_prefix(mol, atom_idx: int, phosphorus_idx: int) -> str:
    prefix = _simple_or_branch_prefix(mol, atom_idx, phosphorus_idx)
    if not prefix:
        return ""
    bare = _strip_outer_parentheses(prefix)
    return f"({bare})" if _needs_substituent_parentheses(bare) else bare


def _thiooxo_ester_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for center in mol.GetAtoms():
        if center.IsInRing() or center.GetFormalCharge() != 0 or center.GetSymbol() not in {"P", "S"}:
            continue
        record = _thiooxo_ester_center_record(mol, Chem, center.GetIdx())
        if not record or record["covered_atoms"] != atom_set:
            continue
        name = _thiooxo_ester_record_name(record)
        if name:
            names.append(name)
    return names


def _sulfinate_retained_parent_names(mol, Chem) -> list[str]:
    names = []
    simple_name = _simple_sulfinate_family_name(mol, Chem)
    if simple_name:
        names.append(simple_name)
    for sulfur in mol.GetAtoms():
        if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0 or sulfur.IsInRing():
            continue
        record = _sulfenate_retained_parent_record(mol, Chem, sulfur.GetIdx())
        if record and record["covered_atoms"] == set(range(mol.GetNumAtoms())):
            name = _sulfenate_name_from_substituent(record["substituent"])
            if name:
                names.append(name)
        record = _thiosulfinate_retained_parent_record(mol, Chem, sulfur.GetIdx())
        if record and record["covered_atoms"] == set(range(mol.GetNumAtoms())):
            substituent = record["substituent"]
            if substituent:
                names.append(substituent + "thiosulfinate")
        record = _sulfonothioate_retained_parent_record(mol, Chem, sulfur.GetIdx())
        if record and record["covered_atoms"] == set(range(mol.GetNumAtoms())):
            substituent = record["substituent"]
            if substituent:
                names.append(substituent + "sulfonothioate")
        record = _sulfinate_retained_parent_record(mol, Chem, sulfur.GetIdx())
        if not record or record["covered_atoms"] != set(range(mol.GetNumAtoms())):
            continue
        substituent = record["substituent"]
        if substituent:
            names.append(substituent + "sulfinate")
    return names


def _simple_sulfinate_family_name(mol, Chem) -> str:
    element_counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    if element_counts == {"C": 1, "O": 2, "S": 1}:
        sulfur = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S")
        carbon = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
        oxygens = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "O"]
        if (
            sulfur.GetFormalCharge() == 0
            and carbon.GetFormalCharge() == 0
            and any(oxygen.GetFormalCharge() == -1 for oxygen in oxygens)
            and any(
                _has_bond_type(mol, Chem, sulfur.GetIdx(), oxygen.GetIdx(), Chem.BondType.DOUBLE)
                for oxygen in oxygens
            )
            and any(
                _has_bond_type(mol, Chem, sulfur.GetIdx(), oxygen.GetIdx(), Chem.BondType.SINGLE)
                for oxygen in oxygens
                if oxygen.GetFormalCharge() == -1
            )
            and _has_bond_type(mol, Chem, sulfur.GetIdx(), carbon.GetIdx(), Chem.BondType.DOUBLE)
        ):
            return "methylenesulfinate"

    if element_counts == {"C": 1, "O": 4, "S": 2}:
        carbon = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
        sulfurs = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S"]
        if carbon.GetFormalCharge() != 0:
            return ""
        for sulfur in sulfurs:
            if sulfur.GetFormalCharge() != 0:
                return ""
            if not _has_bond_type(mol, Chem, sulfur.GetIdx(), carbon.GetIdx(), Chem.BondType.SINGLE):
                return ""
            oxygens = [
                nbr
                for nbr in sulfur.GetNeighbors()
                if nbr.GetSymbol() == "O"
            ]
            if len(oxygens) != 2:
                return ""
            if not any(
                oxygen.GetFormalCharge() == -1
                and _has_bond_type(mol, Chem, sulfur.GetIdx(), oxygen.GetIdx(), Chem.BondType.SINGLE)
                for oxygen in oxygens
            ):
                return ""
            if not any(
                oxygen.GetFormalCharge() == 0
                and _has_bond_type(mol, Chem, sulfur.GetIdx(), oxygen.GetIdx(), Chem.BondType.DOUBLE)
                for oxygen in oxygens
            ):
                return ""
        return "methanedisulfinate"

    return ""


def _has_bond_type(mol, Chem, left_idx: int, right_idx: int, bond_type) -> bool:
    bond = mol.GetBondBetweenAtoms(left_idx, right_idx)
    return bond is not None and bond.GetBondType() == bond_type


def _sulfenate_retained_parent_record(mol, Chem, sulfur_idx: int) -> dict | None:
    carbon_neighbors = []
    terminal_oxido = 0
    covered_atoms = {sulfur_idx}
    for nbr in mol.GetAtomWithIdx(sulfur_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_neighbors.append(nidx)
            continue
        if (
            nbr.GetSymbol() == "O"
            and bond.GetBondType() == Chem.BondType.SINGLE
            and nbr.GetFormalCharge() == -1
            and len(nbr.GetNeighbors()) == 1
        ):
            terminal_oxido += 1
            covered_atoms.add(nidx)
            continue
        return None
    if len(carbon_neighbors) != 1 or terminal_oxido != 1:
        return None
    carbon_idx = carbon_neighbors[0]
    substituent = _simple_or_branch_prefix(mol, carbon_idx, sulfur_idx)
    if not substituent:
        return None
    covered_atoms.add(carbon_idx)
    covered_atoms.update(_collect_component_atoms(mol, carbon_idx, sulfur_idx))
    return {
        "substituent": _retained_parent_substituent_alias(substituent),
        "covered_atoms": covered_atoms,
    }


def _thiosulfinate_retained_parent_record(mol, Chem, sulfur_idx: int) -> dict | None:
    carbon_neighbors = []
    terminal_oxido = 0
    double_sulfur = 0
    covered_atoms = {sulfur_idx}
    for nbr in mol.GetAtomWithIdx(sulfur_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_neighbors.append(nidx)
            continue
        if (
            nbr.GetSymbol() == "O"
            and bond.GetBondType() == Chem.BondType.SINGLE
            and nbr.GetFormalCharge() == -1
            and len(nbr.GetNeighbors()) == 1
        ):
            terminal_oxido += 1
            covered_atoms.add(nidx)
            continue
        if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_sulfur += 1
            covered_atoms.add(nidx)
            continue
        return None
    if len(carbon_neighbors) != 1 or terminal_oxido != 1 or double_sulfur != 1:
        return None
    carbon_idx = carbon_neighbors[0]
    substituent = _simple_or_branch_prefix(mol, carbon_idx, sulfur_idx)
    if not substituent:
        return None
    covered_atoms.add(carbon_idx)
    covered_atoms.update(_collect_component_atoms(mol, carbon_idx, sulfur_idx))
    return {
        "substituent": _retained_parent_substituent_alias(substituent),
        "covered_atoms": covered_atoms,
    }


def _sulfenate_name_from_substituent(substituent: str) -> str:
    alkane_stems = {
        "methyl": "methane",
        "ethyl": "ethane",
        "propyl": "propane",
        "butyl": "butane",
        "pentyl": "pentane",
        "hexyl": "hexane",
        "heptyl": "heptane",
        "octyl": "octane",
        "nonyl": "nonane",
        "decyl": "decane",
    }
    if substituent in alkane_stems:
        return alkane_stems[substituent] + "sulfenate"
    if substituent == "phenyl":
        return "benzenesulfenate"
    cyclo_match = re.fullmatch(r"cyclo([a-z]+)yl", substituent)
    if cyclo_match:
        return f"cyclo{cyclo_match.group(1)}anesulfenate"
    return ""


def _sulfonothioate_retained_parent_record(mol, Chem, sulfur_idx: int) -> dict | None:
    carbon_neighbors = []
    double_oxygen = 0
    terminal_oxido = 0
    double_sulfur = 0
    covered_atoms = {sulfur_idx}
    for nbr in mol.GetAtomWithIdx(sulfur_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_neighbors.append(nidx)
            continue
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_oxygen += 1
            covered_atoms.add(nidx)
            continue
        if (
            nbr.GetSymbol() == "O"
            and bond.GetBondType() == Chem.BondType.SINGLE
            and nbr.GetFormalCharge() == -1
            and len(nbr.GetNeighbors()) == 1
        ):
            terminal_oxido += 1
            covered_atoms.add(nidx)
            continue
        if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_sulfur += 1
            covered_atoms.add(nidx)
            continue
        return None
    if len(carbon_neighbors) != 1 or double_oxygen != 1 or terminal_oxido != 1 or double_sulfur != 1:
        return None
    carbon_idx = carbon_neighbors[0]
    substituent = _simple_or_branch_prefix(mol, carbon_idx, sulfur_idx)
    if not substituent:
        return None
    covered_atoms.add(carbon_idx)
    covered_atoms.update(_collect_component_atoms(mol, carbon_idx, sulfur_idx))
    return {
        "substituent": _retained_parent_substituent_alias(substituent),
        "covered_atoms": covered_atoms,
    }


def _sulfinate_retained_parent_record(mol, Chem, sulfur_idx: int) -> dict | None:
    carbon_neighbors = []
    double_oxygen = 0
    terminal_oxido = 0
    covered_atoms = {sulfur_idx}
    for nbr in mol.GetAtomWithIdx(sulfur_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_neighbors.append(nidx)
            continue
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_oxygen += 1
            covered_atoms.add(nidx)
            continue
        if (
            nbr.GetSymbol() == "O"
            and bond.GetBondType() == Chem.BondType.SINGLE
            and nbr.GetFormalCharge() == -1
            and len(nbr.GetNeighbors()) == 1
        ):
            terminal_oxido += 1
            covered_atoms.add(nidx)
            continue
        return None
    if len(carbon_neighbors) != 1 or double_oxygen != 1 or terminal_oxido != 1:
        return None
    carbon_idx = carbon_neighbors[0]
    substituent = _simple_or_branch_prefix(mol, carbon_idx, sulfur_idx)
    if not substituent:
        return None
    covered_atoms.add(carbon_idx)
    covered_atoms.update(_collect_component_atoms(mol, carbon_idx, sulfur_idx))
    return {
        "substituent": _retained_parent_substituent_alias(substituent),
        "covered_atoms": covered_atoms,
    }


def _thiooxo_ester_center_record(mol, Chem, center_idx: int) -> dict | None:
    center = mol.GetAtomWithIdx(center_idx)
    double_sulfur = 0
    double_oxygen = 0
    terminal_oxido = 0
    terminal_hydroxy = 0
    terminal_sulfido = 0
    terminal_sulfanyl = 0
    terminal_amino = 0
    halogen_ligands = []
    owner_names = []
    covered_atoms = {center_idx}
    for nbr in center.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(center_idx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_sulfur += 1
            covered_atoms.add(nidx)
            continue
        if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.SINGLE:
            heavy = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != center_idx and atom.GetAtomicNum() > 1
            ]
            if heavy:
                return None
            if nbr.GetFormalCharge() < 0:
                terminal_sulfido += 1
            elif nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                terminal_sulfanyl += 1
            else:
                return None
            covered_atoms.add(nidx)
            continue
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen += 1
            covered_atoms.add(nidx)
            continue
        if (
            nbr.GetSymbol() in HALO
            and bond.GetBondType() == Chem.BondType.SINGLE
            and nbr.GetFormalCharge() == 0
        ):
            heavy = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != center_idx and atom.GetAtomicNum() > 1
            ]
            if heavy:
                return None
            halogen_ligands.append(nbr.GetSymbol())
            covered_atoms.add(nidx)
            continue
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
            heavy = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != center_idx and atom.GetAtomicNum() > 1
            ]
            if heavy or nbr.GetFormalCharge() != 0:
                return None
            terminal_amino += 1
            covered_atoms.add(nidx)
            continue
        if nbr.GetSymbol() != "O" or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        heavy = [
            atom.GetIdx()
            for atom in nbr.GetNeighbors()
            if atom.GetIdx() != center_idx and atom.GetAtomicNum() > 1
        ]
        if not heavy:
            if nbr.GetFormalCharge() < 0:
                terminal_oxido += 1
            elif nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                terminal_hydroxy += 1
            else:
                return None
            covered_atoms.add(nidx)
            continue
        if len(heavy) != 1 or nbr.GetFormalCharge() != 0:
            return None
        owner = _single_external_substituent_name(mol, nidx, center_idx)
        if not owner or _unstable_retained_functional_parent_substituent(owner):
            return None
        owner_names.append(owner)
        covered_atoms.add(nidx)
        covered_atoms.update(_collect_component_atoms(mol, heavy[0], nidx))
    return {
        "center_symbol": center.GetSymbol(),
        "center_hydrogens": center.GetTotalNumHs(),
        "double_sulfur": double_sulfur,
        "double_oxygen": double_oxygen,
        "terminal_oxido": terminal_oxido,
        "terminal_hydroxy": terminal_hydroxy,
        "terminal_sulfido": terminal_sulfido,
        "terminal_sulfanyl": terminal_sulfanyl,
        "terminal_amino": terminal_amino,
        "halogen_ligands": tuple(sorted(halogen_ligands)),
        "owner_names": tuple(owner_names),
        "covered_atoms": covered_atoms,
    }


def _thiooxo_ester_record_name(record: dict) -> str:
    owner_names = record["owner_names"]
    if record["double_sulfur"] != 1:
        return ""
    center_symbol = record["center_symbol"]
    double_oxygen = record["double_oxygen"]
    terminal_oxido = record["terminal_oxido"]
    terminal_hydroxy = record["terminal_hydroxy"]
    terminal_sulfido = record["terminal_sulfido"]
    terminal_sulfanyl = record["terminal_sulfanyl"]
    terminal_amino = record["terminal_amino"]
    halogen_ligands = record["halogen_ligands"]
    center_hydrogens = record["center_hydrogens"]
    single_oxygen_count = len(owner_names) + terminal_oxido + terminal_hydroxy
    if center_symbol == "P":
        if double_oxygen != 0:
            return ""
        if (
            center_hydrogens == 0
            and terminal_sulfido == 0
            and terminal_sulfanyl == 0
            and not halogen_ligands
        ):
            if terminal_amino == 1 and terminal_oxido == 1 and terminal_hydroxy == 0 and owner_names:
                return _thiooxo_ester_name(owner_names, 0, "phosphoramidothioate")
        if center_hydrogens == 0 and terminal_amino == 0 and not halogen_ligands:
            if terminal_sulfido == 1 and terminal_sulfanyl == 0:
                if terminal_oxido == 1 and terminal_hydroxy == 0 and len(owner_names) == 1:
                    return _thiooxo_ester_name(owner_names, 0, "phosphorodithioate")
                if terminal_oxido == 0 and terminal_hydroxy == 0 and len(owner_names) == 2:
                    return _thiooxo_ester_name(owner_names, 0, "phosphorodithioate")
            if (
                terminal_sulfido == 0
                and terminal_sulfanyl == 1
                and terminal_oxido == 0
                and terminal_hydroxy == 1
                and len(owner_names) == 1
            ):
                return _thiooxo_ester_name(owner_names, 0, "phosphorodithioic acid")
        if (
            center_hydrogens == 0
            and terminal_amino == 0
            and terminal_oxido == 0
            and terminal_hydroxy == 0
            and terminal_sulfido == 0
            and terminal_sulfanyl == 0
            and len(owner_names) == 1
        ):
            halogen_parent = _phosphorodihalidothioate_parent(halogen_ligands)
            if halogen_parent:
                return _thiooxo_ester_name(owner_names, 0, halogen_parent)
        if not owner_names and center_hydrogens == 0:
            if terminal_amino == 0:
                if terminal_oxido == 3 and terminal_hydroxy == 0:
                    return "thiophosphate"
                if terminal_oxido == 0 and terminal_hydroxy == 3:
                    return "thiophosphoric acid"
            if terminal_amino == 1 and terminal_oxido == 2 and terminal_hydroxy == 0:
                return "thiophosphoramidate"
            return ""
        if terminal_amino != 0:
            return ""
        if center_hydrogens == 0 and single_oxygen_count == 3:
            return _thiooxo_ester_name(owner_names, terminal_hydroxy, "thiophosphate")
        if center_hydrogens == 1 and single_oxygen_count == 2:
            return _thiooxo_ester_name(owner_names, terminal_hydroxy, "thiophosphonate")
        return ""
    if center_symbol != "S":
        return ""
    if terminal_sulfido or terminal_sulfanyl or halogen_ligands:
        return ""
    if terminal_amino != 0:
        return ""
    if double_oxygen == 0 and center_hydrogens == 1 and terminal_oxido == 0 and terminal_hydroxy == 0 and len(owner_names) == 1:
        return _thiooxo_ester_name(owner_names, 0, "thiosulfinate")
    if double_oxygen != 1:
        return ""
    if center_hydrogens == 1 and terminal_oxido == 0 and terminal_hydroxy == 0 and len(owner_names) == 1:
        return _thiooxo_ester_name(owner_names, 0, "thiosulfonate")
    if center_hydrogens == 0 and single_oxygen_count == 2:
        return _thiooxo_ester_name(owner_names, terminal_hydroxy, "thiosulfate")
    return ""


def _phosphorodihalidothioate_parent(halogen_ligands: Sequence[str]) -> str:
    if len(halogen_ligands) != 2 or halogen_ligands[0] != halogen_ligands[1]:
        return ""
    roots = {
        "F": "fluorido",
        "Cl": "chlorido",
        "Br": "bromido",
        "I": "iodido",
    }
    root = roots.get(halogen_ligands[0], "")
    return f"phosphorodi{root}thioate" if root else ""


def _thiooxo_ester_name(owner_names: Sequence[str], hydrogen_count: int, parent: str) -> str:
    if hydrogen_count not in {0, 1, 2}:
        return ""
    owner_text = _thiooxo_ester_owner_text(owner_names)
    if not owner_text:
        return ""
    hydrogen_text = {0: "", 1: "hydrogen", 2: "dihydrogen"}[hydrogen_count]
    return " ".join(part for part in (owner_text, hydrogen_text, parent) if part)


def _thiooxo_ester_owner_text(owner_names: Sequence[str]) -> str:
    pieces = []
    counts = Counter(owner_names)
    for owner in sorted(counts):
        count = counts[owner]
        if count == 1:
            pieces.append(_parenthesized_if_complex(owner))
            continue
        if _simple_unparenthesized_n_substituent(owner):
            multiplier = MULT.get(count, "")
            pieces.append(multiplier + owner if multiplier else "")
            continue
        multiplier = _PHOSPHORANE_COMPLEX_MULTIPLIERS.get(count, "")
        pieces.append(f"{multiplier}({_strip_outer_parentheses(owner)})" if multiplier else "")
    if any(not piece for piece in pieces):
        return ""
    return " ".join(pieces)


def _phosphorothioic_triamide_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for phosphorus in mol.GetAtoms():
        if phosphorus.GetSymbol() != "P" or phosphorus.IsInRing() or phosphorus.GetFormalCharge() != 0:
            continue
        record = _phosphorothioic_triamide_record(mol, Chem, phosphorus.GetIdx())
        if not record or record["covered_atoms"] != atom_set:
            continue
        name = _phosphorothioic_triamide_record_name(record)
        if name:
            names.append(name)
    return names


def _phosphorothioic_triamide_record(mol, Chem, pidx: int) -> dict | None:
    double_sulfur = 0
    nitrogen_records = []
    covered_atoms = {pidx}
    for nbr in mol.GetAtomWithIdx(pidx).GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_sulfur += 1
            covered_atoms.add(nidx)
            continue
        if nbr.GetSymbol() != "N" or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        heavy = [
            atom.GetIdx()
            for atom in nbr.GetNeighbors()
            if atom.GetIdx() != pidx and atom.GetAtomicNum() > 1
        ]
        substituents = _external_substituent_names(mol, nidx, pidx)
        ligand_prefix = _simple_or_branch_prefix(mol, nidx, pidx)
        covered_atoms.add(nidx)
        for heavy_idx in heavy:
            covered_atoms.update(_collect_component_atoms(mol, heavy_idx, nidx))
        nitrogen_records.append(
            {
                "formal_charge": nbr.GetFormalCharge(),
                "is_ring": nbr.IsInRing(),
                "substituents": tuple(substituents or ()),
                "substituents_available": substituents is not None,
                "ligand_prefix": ligand_prefix,
            }
        )
    if double_sulfur != 1 or len(nitrogen_records) != 3:
        return None
    return {"nitrogens": tuple(nitrogen_records), "covered_atoms": covered_atoms}


def _phosphorothioic_triamide_record_name(record: dict) -> str:
    nitrogens = record["nitrogens"]
    charges = [nitrogen["formal_charge"] for nitrogen in nitrogens]
    substituents_available = all(nitrogen["substituents_available"] for nitrogen in nitrogens)
    if substituents_available and all(charge < 0 for charge in charges) and all(not nitrogen["substituents"] for nitrogen in nitrogens):
        return "thiophosphoryltriamide"
    substituted_anions = [nitrogen for nitrogen in nitrogens if nitrogen["substituents"]]
    if (
        substituents_available
        and all(charge < 0 for charge in charges)
        and len(substituted_anions) == 1
        and len(substituted_anions[0]["substituents"]) == 1
    ):
        prefix = _n_substituent_prefix(substituted_anions[0]["substituents"])
        if prefix is not None:
            return prefix + "thiophosphoryltriamide"
    if any(charge != 0 for charge in charges):
        return ""
    substituted = [nitrogen for nitrogen in nitrogens if nitrogen["substituents"]]
    if substituents_available and not substituted:
        return "thiophosphoric triamide"
    if substituents_available and len(substituted) == 1 and all(len(nitrogen["substituents"]) <= 2 for nitrogen in nitrogens):
        prefix = _n_substituent_prefix(substituted[0]["substituents"])
        if prefix is not None:
            return prefix + "thiophosphoric triamide"
    ligand_prefixes = [nitrogen["ligand_prefix"] for nitrogen in nitrogens]
    if all(nitrogen["is_ring"] for nitrogen in nitrogens) and all(ligand_prefixes):
        ligand_text = _thiooxo_ester_owner_text(ligand_prefixes)
        if ligand_text:
            return f"{ligand_text}phosphine sulfide"
    return ""


def _dithiophosphate_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for phosphorus in mol.GetAtoms():
        if phosphorus.GetSymbol() != "P" or phosphorus.IsInRing() or phosphorus.GetFormalCharge() != 0:
            continue
        record = _dithiophosphate_record(mol, Chem, phosphorus.GetIdx())
        if not record or record["covered_atoms"] != atom_set:
            continue
        name = _dithiophosphate_record_name(record)
        if name:
            names.append(name)
    return names


def _dithiophosphate_record(mol, Chem, pidx: int) -> dict | None:
    phosphorus = mol.GetAtomWithIdx(pidx)
    double_sulfur = 0
    terminal_oxido = 0
    terminal_hydroxy = 0
    terminal_sulfido = 0
    owner_sulfur_names = []
    covered_atoms = {pidx}
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_sulfur += 1
            covered_atoms.add(nidx)
            continue
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            heavy = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != pidx and atom.GetAtomicNum() > 1
            ]
            if heavy:
                return None
            if nbr.GetFormalCharge() < 0:
                terminal_oxido += 1
            elif nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                terminal_hydroxy += 1
            else:
                return None
            covered_atoms.add(nidx)
            continue
        if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.SINGLE:
            heavy = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != pidx and atom.GetAtomicNum() > 1
            ]
            if not heavy:
                if nbr.GetFormalCharge() < 0:
                    terminal_sulfido += 1
                    covered_atoms.add(nidx)
                    continue
                return None
            if len(heavy) != 1 or nbr.GetFormalCharge() != 0:
                return None
            owner = _single_external_substituent_name(mol, nidx, pidx)
            if not owner or _unstable_retained_functional_parent_substituent(owner):
                return None
            owner_sulfur_names.append(owner)
            covered_atoms.add(nidx)
            covered_atoms.update(_collect_component_atoms(mol, heavy[0], nidx))
            continue
        return None
    if double_sulfur != 1:
        return None
    return {
        "phosphorus_hydrogens": phosphorus.GetTotalNumHs(),
        "terminal_oxido": terminal_oxido,
        "terminal_hydroxy": terminal_hydroxy,
        "terminal_sulfido": terminal_sulfido,
        "owner_sulfur_names": tuple(owner_sulfur_names),
        "covered_atoms": covered_atoms,
    }


def _dithiophosphate_record_name(record: dict) -> str:
    owner_names = record["owner_sulfur_names"]
    terminal_oxido = record["terminal_oxido"]
    terminal_hydroxy = record["terminal_hydroxy"]
    terminal_sulfido = record["terminal_sulfido"]
    phosphorus_hydrogens = record["phosphorus_hydrogens"]
    oxygen_count = terminal_oxido + terminal_hydroxy
    if not owner_names:
        if terminal_hydroxy != 0:
            return ""
        if phosphorus_hydrogens == 1 and terminal_sulfido == 2 and terminal_oxido == 0:
            return "trithiophosphonate"
        if phosphorus_hydrogens == 0 and terminal_sulfido == 2 and terminal_oxido == 1:
            return "trithiophosphate"
        if phosphorus_hydrogens == 0 and terminal_sulfido == 3 and terminal_oxido == 0:
            return "tetrathiophosphate"
        return ""
    if len(owner_names) != 1 or phosphorus_hydrogens != 0:
        return ""
    if terminal_sulfido == 0 and oxygen_count == 2:
        return _dithiophosphate_owner_name(owner_names[0], terminal_hydroxy, "dithiophosphate")
    if terminal_sulfido == 1 and oxygen_count == 1:
        return _dithiophosphate_owner_name(owner_names[0], terminal_hydroxy, "trithiophosphate")
    if terminal_sulfido == 2 and oxygen_count == 0:
        return _dithiophosphate_owner_name(owner_names[0], 0, "tetrathiophosphate")
    return ""


def _dithiophosphate_owner_name(owner: str, hydrogen_count: int, parent: str) -> str:
    if hydrogen_count not in {0, 1, 2}:
        return ""
    if _simple_unparenthesized_n_substituent(owner):
        owner_text = f"S-{owner}"
    else:
        owner_text = f"S-({_strip_outer_parentheses(owner)})"
    hydrogen_text = {0: "", 1: "hydrogen", 2: "dihydrogen"}[hydrogen_count]
    return " ".join(part for part in (owner_text, hydrogen_text, parent) if part)


def _diacylperoxy_dimethylhexane_name(mol, Chem) -> str:
    records = []
    seen_bonds = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetSymbol() != "O" or end.GetSymbol() != "O":
            continue
        if begin.GetFormalCharge() != 0 or end.GetFormalCharge() != 0:
            continue
        bond_key = frozenset({begin.GetIdx(), end.GetIdx()})
        if bond_key in seen_bonds:
            continue
        seen_bonds.add(bond_key)
        record = _acylperoxy_tertiary_carbon_record(mol, Chem, begin.GetIdx(), end.GetIdx())
        if record:
            records.append(record)
    if len(records) != 2:
        return ""
    first, second = records
    if first["acyl_prefix"] != second["acyl_prefix"]:
        return ""
    base_atoms = _dimethylhexane_2_5_base_atoms(mol, Chem, first, second)
    if not base_atoms:
        return ""
    covered = set(base_atoms)
    for record in records:
        covered.update(record["covered_atoms"])
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    acyl = _strip_outer_parentheses(first["acyl_prefix"])
    return f"2,5-bis({acyl}peroxy)-2,5-dimethylhexane"


def _acylperoxy_tertiary_carbon_record(mol, Chem, first_oxygen_idx: int, second_oxygen_idx: int) -> dict | None:
    for acyl_oxygen_idx, alkyl_oxygen_idx in (
        (first_oxygen_idx, second_oxygen_idx),
        (second_oxygen_idx, first_oxygen_idx),
    ):
        acyl_heavy = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(acyl_oxygen_idx).GetNeighbors()
            if nbr.GetIdx() != alkyl_oxygen_idx and nbr.GetAtomicNum() > 1
        ]
        alkyl_heavy = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(alkyl_oxygen_idx).GetNeighbors()
            if nbr.GetIdx() != acyl_oxygen_idx and nbr.GetAtomicNum() > 1
        ]
        if len(acyl_heavy) != 1 or len(alkyl_heavy) != 1:
            continue
        carbonyl_idx = acyl_heavy[0]
        owner_idx = alkyl_heavy[0]
        carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
        owner = mol.GetAtomWithIdx(owner_idx)
        if carbonyl.GetSymbol() != "C" or owner.GetSymbol() != "C":
            continue
        if not _is_carbonyl_neighbor(mol, carbonyl_idx, acyl_oxygen_idx):
            continue
        if owner.IsInRing() or owner.GetFormalCharge() != 0:
            continue
        acyl_prefix = _simple_or_branch_prefix(mol, carbonyl_idx, acyl_oxygen_idx)
        if not acyl_prefix:
            continue
        covered = {first_oxygen_idx, second_oxygen_idx}
        covered.update(_collect_component_atoms(mol, carbonyl_idx, acyl_oxygen_idx))
        return {
            "acyl_prefix": acyl_prefix,
            "owner": owner_idx,
            "owner_oxygen": alkyl_oxygen_idx,
            "covered_atoms": covered,
        }
    return None


def _dimethylhexane_2_5_base_atoms(mol, Chem, first: dict, second: dict) -> set[int]:
    first_owner = first["owner"]
    second_owner = second["owner"]
    if first_owner == second_owner:
        return set()
    first_record = _dimethylhexane_owner_record(mol, Chem, first_owner, first["owner_oxygen"])
    second_record = _dimethylhexane_owner_record(mol, Chem, second_owner, second["owner_oxygen"])
    if not first_record or not second_record:
        return set()
    first_chain = first_record["chain"]
    second_chain = second_record["chain"]
    bond = mol.GetBondBetweenAtoms(first_chain, second_chain)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return set()
    if not _methylene_bridge_carbon(mol, first_chain, {first_owner, second_chain}):
        return set()
    if not _methylene_bridge_carbon(mol, second_chain, {second_owner, first_chain}):
        return set()
    return {
        first_owner,
        second_owner,
        first_chain,
        second_chain,
        *first_record["methyls"],
        *second_record["methyls"],
    }


def _dimethylhexane_owner_record(mol, Chem, owner_idx: int, owner_oxygen_idx: int) -> dict | None:
    owner = mol.GetAtomWithIdx(owner_idx)
    if owner.GetSymbol() != "C" or owner.IsInRing() or owner.GetFormalCharge() != 0:
        return None
    neighbors = [
        nbr.GetIdx()
        for nbr in owner.GetNeighbors()
        if nbr.GetIdx() != owner_oxygen_idx and nbr.GetAtomicNum() > 1
    ]
    methyls = [idx for idx in neighbors if _terminal_methyl_carbon(mol, idx, owner_idx)]
    chain = [idx for idx in neighbors if idx not in methyls]
    if len(methyls) != 2 or len(chain) != 1:
        return None
    if any(
        mol.GetBondBetweenAtoms(owner_idx, idx).GetBondType() != Chem.BondType.SINGLE
        for idx in neighbors
    ):
        return None
    return {"methyls": tuple(methyls), "chain": chain[0]}


def _terminal_methyl_carbon(mol, atom_idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() != "C" or atom.IsInRing() or atom.GetFormalCharge() != 0:
        return False
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]
    return heavy == [parent_idx]


def _methylene_bridge_carbon(mol, atom_idx: int, expected_neighbors: set[int]) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() != "C" or atom.IsInRing() or atom.GetFormalCharge() != 0:
        return False
    heavy = {nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1}
    return heavy == expected_neighbors


def _oxazinone_retained_parent_name(mol, Chem) -> str:
    candidates = []
    for ring in mol.GetRingInfo().AtomRings():
        ring_set = frozenset(ring)
        if len(ring_set) != 6:
            continue
        sequence = _oxazinone_locant_sequence(mol, Chem, ring_set)
        if not sequence:
            continue
        name = _name_oxazinone_locant_sequence(mol, sequence)
        if name:
            candidates.append(name)
    return sorted(candidates, key=lambda item: (len(item), item))[0] if len(candidates) == 1 else ""


def _oxazinone_locant_sequence(mol, Chem, ring_set: frozenset[int]) -> tuple[int, ...]:
    ring_oxygens = [idx for idx in ring_set if mol.GetAtomWithIdx(idx).GetSymbol() == "O"]
    ring_nitrogens = [idx for idx in ring_set if mol.GetAtomWithIdx(idx).GetSymbol() == "N"]
    if len(ring_oxygens) != 1 or len(ring_nitrogens) != 1:
        return ()
    if any(not mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_set):
        return ()
    oxygen_idx = ring_oxygens[0]
    carbonyl_neighbors = []
    for nbr_idx in _ring_neighbors(mol, ring_set, oxygen_idx):
        if _ring_atom_has_exocyclic_carbonyl_oxygen(mol, Chem, nbr_idx, ring_set):
            carbonyl_neighbors.append(nbr_idx)
    if len(carbonyl_neighbors) != 1:
        return ()
    next_idx = next(
        (idx for idx in _ring_neighbors(mol, ring_set, oxygen_idx) if idx != carbonyl_neighbors[0]),
        None,
    )
    if next_idx is None:
        return ()
    sequence = _ring_sequence(mol, ring_set, oxygen_idx, next_idx)
    if len(sequence) != 6:
        return ()
    symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in sequence]
    if symbols != ["O", "C", "N", "C", "C", "C"]:
        return ()
    if sequence[5] != carbonyl_neighbors[0]:
        return ()
    if any(mol.GetAtomWithIdx(idx).GetFormalCharge() != 0 for idx in ring_set):
        return ()
    return sequence


def _ring_atom_has_exocyclic_carbonyl_oxygen(mol, Chem, atom_idx: int, ring_set: frozenset[int]) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() != "C":
        return False
    carbonyl_oxygens = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() not in ring_set
        and nbr.GetSymbol() == "O"
        and nbr.GetFormalCharge() == 0
        and mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]
    return len(carbonyl_oxygens) == 1


def _name_oxazinone_locant_sequence(mol, sequence: Sequence[int]) -> str:
    ring_set = frozenset(sequence)
    covered = set(sequence)
    prefixes = []
    locants = {atom_idx: str(locant) for locant, atom_idx in enumerate(sequence, start=1)}
    for atom_idx in sequence:
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set or nbr.GetAtomicNum() <= 1:
                continue
            if atom_idx == sequence[5] and nbr.GetSymbol() == "O":
                covered.add(nidx)
                continue
            prefix = _simple_or_branch_prefix(mol, nidx, atom_idx)
            if not prefix:
                return ""
            prefixes.append((locants[atom_idx], prefix))
            covered.update(_collect_component_atoms(mol, nidx, atom_idx))
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    prefix_text = _format_prefixes(prefixes)
    parent = "6H-1,3-oxazin-6-one"
    return _join_prefix_parent(prefix_text, parent) if prefix_text else parent


def _oxadiazinane_nitroimine_name(mol, Chem, ring_set: frozenset[int]) -> str:
    if len(ring_set) != 6:
        return ""
    if not _saturated_nonaromatic_ring(mol, Chem, ring_set):
        return ""
    symbols = Counter(mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring_set)
    if symbols != Counter({"C": 3, "N": 2, "O": 1}):
        return ""
    nitroimine = _single_ring_nitroimine(mol, Chem, ring_set)
    if not nitroimine:
        return ""
    imine_c, imine_n, nitro_atoms = nitroimine
    oxygen_idx = next(idx for idx in ring_set if mol.GetAtomWithIdx(idx).GetSymbol() == "O")
    for neighbor in _ring_neighbors(mol, ring_set, oxygen_idx):
        sequence = _ring_sequence(mol, ring_set, oxygen_idx, neighbor)
        if not sequence or sequence[3] != imine_c:
            continue
        if [mol.GetAtomWithIdx(idx).GetSymbol() for idx in sequence] != ["O", "C", "N", "C", "N", "C"]:
            continue
        locants = {idx: str(pos) for pos, idx in enumerate(sequence, start=1)}
        name = _cyclic_nitroimine_name_from_locants(
            mol,
            Chem,
            ring_set,
            locants,
            imine_n,
            nitro_atoms,
            "N-nitro-1,3,5-oxadiazinan-4-imine",
        )
        if name:
            return name
    return ""


def _imidazolidine_nitroimine_name(mol, Chem, ring_set: frozenset[int]) -> str:
    if len(ring_set) != 5:
        return ""
    if not _saturated_nonaromatic_ring(mol, Chem, ring_set):
        return ""
    symbols = Counter(mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring_set)
    if symbols != Counter({"C": 3, "N": 2}):
        return ""
    nitroimine = _single_ring_nitroimine(mol, Chem, ring_set)
    if not nitroimine:
        return ""
    imine_c, imine_n, nitro_atoms = nitroimine
    first_nitrogens = [
        nidx
        for nidx in _ring_neighbors(mol, ring_set, imine_c)
        if mol.GetAtomWithIdx(nidx).GetSymbol() == "N" and _ring_atom_external_substituents(mol, ring_set, nidx, imine_n)
    ]
    for first_n in sorted(first_nitrogens):
        sequence = _ring_sequence(mol, ring_set, first_n, imine_c)
        if not sequence or sequence[1] != imine_c:
            continue
        if [mol.GetAtomWithIdx(idx).GetSymbol() for idx in sequence] != ["N", "C", "N", "C", "C"]:
            continue
        locants = {idx: str(pos) for pos, idx in enumerate(sequence, start=1)}
        name = _cyclic_nitroimine_name_from_locants(
            mol,
            Chem,
            ring_set,
            locants,
            imine_n,
            nitro_atoms,
            "N-nitroimidazolidine-2-imine",
        )
        if name:
            return name
    return ""


def _single_ring_nitroimine(mol, Chem, ring_set: frozenset[int]) -> tuple[int, int, set[int]] | None:
    motifs = []
    for cidx in ring_set:
        atom = mol.GetAtomWithIdx(cidx)
        if atom.GetSymbol() != "C":
            continue
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set or nbr.GetSymbol() != "N":
                continue
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            nitro_atoms = _terminal_nitro_atoms(mol, Chem, nidx, cidx)
            if nitro_atoms:
                motifs.append((cidx, nidx, nitro_atoms))
    return motifs[0] if len(motifs) == 1 else None


def _terminal_nitro_atoms(mol, Chem, imine_nidx: int, imine_cidx: int) -> set[int]:
    nitro_neighbors = []
    imine_n = mol.GetAtomWithIdx(imine_nidx)
    for nbr in imine_n.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == imine_cidx:
            continue
        bond = mol.GetBondBetweenAtoms(imine_nidx, nidx)
        if nbr.GetSymbol() == "N" and bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            nitro_neighbors.append(nidx)
        else:
            return set()
    if len(nitro_neighbors) != 1:
        return set()
    nitro_idx = nitro_neighbors[0]
    nitro = mol.GetAtomWithIdx(nitro_idx)
    double_oxygens = []
    single_anionic_oxygens = []
    for nbr in nitro.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == imine_nidx:
            continue
        bond = mol.GetBondBetweenAtoms(nitro_idx, nidx)
        if nbr.GetSymbol() != "O" or bond is None:
            return set()
        if bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_oxygens.append(nidx)
        elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            single_anionic_oxygens.append(nidx)
        else:
            return set()
    if len(double_oxygens) != 1 or len(single_anionic_oxygens) != 1:
        return set()
    return {imine_nidx, nitro_idx, *double_oxygens, *single_anionic_oxygens}


def _cyclic_nitroimine_name_from_locants(
    mol,
    Chem,
    ring_set: frozenset[int],
    locants: dict[int, str],
    imine_nidx: int,
    nitro_atoms: set[int],
    parent: str,
) -> str:
    prefixes = []
    covered = set(ring_set) | set(nitro_atoms)
    for atom_idx in ring_set:
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set or nidx == imine_nidx:
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return ""
            prefix = _simple_or_branch_prefix(mol, nidx, atom_idx)
            if not prefix:
                return ""
            prefixes.append((locants[atom_idx], prefix))
            covered.update(_collect_component_atoms(mol, nidx, atom_idx))
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    prefix_text = _format_prefixes(prefixes)
    return _join_prefix_parent(prefix_text, parent) if prefix_text else parent


def _saturated_nonaromatic_ring(mol, Chem, ring_set: frozenset[int]) -> bool:
    for atom_idx in ring_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetIsAromatic():
            return False
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in ring_set or atom_idx > nidx:
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return False
    return True


def _ring_atom_external_substituents(mol, ring_set: frozenset[int], atom_idx: int, imine_nidx: int) -> list[int]:
    return [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if nbr.GetIdx() not in ring_set and nbr.GetIdx() != imine_nidx and nbr.GetAtomicNum() > 1
    ]


def _ring_neighbors(mol, ring_set: frozenset[int], atom_idx: int) -> list[int]:
    return [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors() if nbr.GetIdx() in ring_set]


def _ring_sequence(mol, ring_set: frozenset[int], start_idx: int, next_idx: int) -> tuple[int, ...]:
    if start_idx not in ring_set or next_idx not in ring_set:
        return ()
    sequence = [start_idx, next_idx]
    previous = start_idx
    current = next_idx
    while len(sequence) < len(ring_set):
        candidates = [idx for idx in _ring_neighbors(mol, ring_set, current) if idx != previous]
        if len(candidates) != 1:
            return ()
        previous, current = current, candidates[0]
        if current == start_idx:
            return ()
        sequence.append(current)
    if start_idx not in _ring_neighbors(mol, ring_set, current):
        return ()
    return tuple(sequence)


def _oxime_sulfonyloxyimino_motifs(mol, Chem) -> list[dict]:
    motifs = []
    for carbon in mol.GetAtoms():
        cidx = carbon.GetIdx()
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
            continue
        for nitrogen in carbon.GetNeighbors():
            nidx = nitrogen.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if nitrogen.GetSymbol() != "N" or bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            motif = _oxime_sulfonyloxyimino_motif_from_imine(mol, Chem, cidx, nidx)
            if motif:
                motifs.append(motif)
    return motifs


def _oxime_sulfonyloxyimino_motif_from_imine(mol, Chem, imine_cidx: int, imine_nidx: int) -> dict | None:
    imine_n = mol.GetAtomWithIdx(imine_nidx)
    oxygens = []
    for nbr in imine_n.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == imine_cidx:
            continue
        bond = mol.GetBondBetweenAtoms(imine_nidx, nidx)
        if nbr.GetSymbol() == "O" and bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            oxygens.append(nidx)
        else:
            return None
    if len(oxygens) != 1:
        return None
    oxygen_idx = oxygens[0]
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    sulfur = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != imine_nidx and nbr.GetSymbol() == "S"
    ]
    if len(sulfur) != 1:
        return None
    sulfur_idx = sulfur[0]
    sulfur_atom = mol.GetAtomWithIdx(sulfur_idx)
    if sulfur_atom.GetFormalCharge() != 0:
        return None
    double_oxygens = []
    ligand_atoms = []
    for nbr in sulfur_atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == oxygen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if nbr.GetSymbol() == "O" and bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygens.append(nidx)
        elif bond is not None and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetAtomicNum() > 1:
            ligand_atoms.append(nidx)
        else:
            return None
    if len(double_oxygens) != 2 or len(ligand_atoms) != 1:
        return None
    ligand_idx = ligand_atoms[0]
    ligand_prefix = _simple_or_branch_prefix(mol, ligand_idx, sulfur_idx)
    if not ligand_prefix:
        return None
    if ligand_prefix in {"4-methylphenyl", "p-tolyl"}:
        prefix = "tosyloxyimino"
    else:
        prefix = f"{_strip_outer_parentheses(ligand_prefix)}sulfonyloxyimino"
    covered = {imine_nidx, oxygen_idx, sulfur_idx, *double_oxygens}
    covered.update(_collect_component_atoms(mol, ligand_idx, sulfur_idx))
    return {
        "imine_carbon": imine_cidx,
        "imine_nitrogen": imine_nidx,
        "sulfonyloxyimino_prefix": prefix,
        "covered_atoms": covered,
    }


def _direct_aryl_oxime_sulfonate_acetonitrile_name(mol, motif: dict) -> str:
    imine_cidx = motif["imine_carbon"]
    substituents = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(imine_cidx).GetNeighbors()
        if nbr.GetIdx() != motif["imine_nitrogen"] and nbr.GetAtomicNum() > 1
    ]
    nitriles = [idx for idx in substituents if _is_nitrile_carbon(mol, idx)]
    aryls = [idx for idx in substituents if idx not in nitriles and mol.GetAtomWithIdx(idx).IsInRing()]
    if len(nitriles) != 1 or len(aryls) != 1:
        return ""
    nitrile_idx = nitriles[0]
    aryl_idx = aryls[0]
    aryl = _simple_or_branch_prefix(mol, aryl_idx, imine_cidx)
    if not aryl:
        return ""
    covered = set(motif["covered_atoms"]) | {imine_cidx}
    covered.update(_collect_component_atoms(mol, nitrile_idx, imine_cidx))
    covered.update(_collect_component_atoms(mol, aryl_idx, imine_cidx))
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    sulfonyloxyimino = f"({_strip_outer_parentheses(motif['sulfonyloxyimino_prefix'])})"
    return (
        f"2-{_parenthesized_if_complex(_strip_outer_parentheses(aryl))}-"
        f"2-{sulfonyloxyimino}acetonitrile"
    )


def _thiophenylidene_oxime_sulfonate_acetonitrile_name(mol, Chem, motif: dict) -> str:
    imine_cidx = motif["imine_carbon"]
    ring = _five_membered_thiophene_like_ring_for_atom(mol, imine_cidx)
    if not ring:
        return ""
    sulfur_idx = next(idx for idx in ring if mol.GetAtomWithIdx(idx).GetSymbol() == "S")
    ylidene_records = []
    for atom_idx in ring:
        if atom_idx == imine_cidx or mol.GetAtomWithIdx(atom_idx).GetSymbol() != "C":
            continue
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring or nbr.GetSymbol() != "C":
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
            if bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
                ylidene_records.append((atom_idx, nidx))
    if len(ylidene_records) != 1:
        return ""
    ylidene_idx, alpha_idx = ylidene_records[0]
    alpha_substituents = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(alpha_idx).GetNeighbors()
        if nbr.GetIdx() != ylidene_idx and nbr.GetAtomicNum() > 1
    ]
    nitriles = [idx for idx in alpha_substituents if _is_nitrile_carbon(mol, idx)]
    aryls = [idx for idx in alpha_substituents if idx not in nitriles and mol.GetAtomWithIdx(idx).IsInRing()]
    if len(nitriles) != 1 or len(aryls) != 1:
        return ""
    aryl = _simple_or_branch_prefix(mol, aryls[0], alpha_idx)
    if not aryl:
        return ""
    prefix = motif["sulfonyloxyimino_prefix"]
    if sulfur_idx in _ring_neighbors(mol, ring, imine_cidx) and ylidene_idx in _ring_neighbors(mol, ring, imine_cidx):
        ring_name = f"[2-({_strip_outer_parentheses(prefix)})-2,3-dihydrothiophen-3-ylidene]"
    elif sulfur_idx in _ring_neighbors(mol, ring, imine_cidx) and sulfur_idx in _ring_neighbors(mol, ring, ylidene_idx):
        if _has_assigned_bond_stereo(mol, Chem):
            return ""
        ring_name = f"[5-({_strip_outer_parentheses(prefix)})-5H-thiophen-2-ylidene]"
    else:
        return ""
    covered = set(motif["covered_atoms"]) | set(ring) | {alpha_idx}
    covered.update(_collect_component_atoms(mol, nitriles[0], alpha_idx))
    covered.update(_collect_component_atoms(mol, aryls[0], alpha_idx))
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    return f"{ring_name}({_strip_outer_parentheses(aryl)})acetonitrile"


def _nitrile_oxide_retained_parent_names(mol, Chem) -> list[str]:
    names = []
    atom_set = set(range(mol.GetNumAtoms()))
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C":
            continue
        record = _nitrile_oxide_retained_parent_record(mol, Chem, carbon.GetIdx())
        if not record or record["covered_atoms"] != atom_set:
            continue
        names.extend(_nitrile_oxide_names_from_prefix(record["prefix"]))
    return list(dict.fromkeys(names))


def _nitrile_oxide_retained_parent_record(mol, Chem, carbon_idx: int) -> dict | None:
    nitrile_oxide_nitrogens = []
    external_neighbors = []
    covered_atoms = {carbon_idx}
    for nbr in mol.GetAtomWithIdx(carbon_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if (
            nbr.GetSymbol() == "N"
            and nbr.GetFormalCharge() > 0
            and bond.GetBondType() == Chem.BondType.TRIPLE
        ):
            oxygen_idx = _terminal_nitrile_oxide_oxygen(mol, nidx, carbon_idx)
            if oxygen_idx < 0:
                return None
            nitrile_oxide_nitrogens.append(nidx)
            covered_atoms.update({nidx, oxygen_idx})
            continue
        if nbr.GetAtomicNum() > 1:
            external_neighbors.append(nidx)
    if len(nitrile_oxide_nitrogens) != 1:
        return None
    if not external_neighbors:
        return {"prefix": "", "covered_atoms": covered_atoms}
    if len(external_neighbors) != 1:
        return None
    external_idx = external_neighbors[0]
    prefix = _retained_parent_substituent_alias(_simple_or_branch_prefix(mol, external_idx, carbon_idx) or "")
    if not prefix:
        return None
    covered_atoms.update(_collect_component_atoms(mol, external_idx, carbon_idx))
    return {"prefix": prefix, "covered_atoms": covered_atoms}


def _terminal_nitrile_oxide_oxygen(mol, nitrogen_idx: int, carbon_idx: int) -> int:
    oxygens = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors()
        if nbr.GetIdx() != carbon_idx
        and nbr.GetSymbol() == "O"
        and nbr.GetFormalCharge() < 0
        and len(nbr.GetNeighbors()) == 1
    ]
    return oxygens[0] if len(oxygens) == 1 else -1


def _nitrile_oxide_names_from_prefix(prefix: str) -> list[str]:
    if not prefix:
        return ["fulminate"]
    names = []
    if prefix == "cyano":
        names.extend(["ethanedinitrile oxide", "cyanoformonitrile oxide"])
    if prefix == "phenyl":
        names.append("benzonitrile oxide")
    elif prefix.endswith("phenyl"):
        names.append(prefix[: -len("phenyl")] + "benzonitrile oxide")
    locanted_parent = re.fullmatch(r"(.+-\d+(?:\([^)]+\))?)-yl", prefix)
    if locanted_parent:
        names.append(locanted_parent.group(1) + "-carbonitrile oxide")
    return names


def _five_membered_thiophene_like_ring_for_atom(mol, atom_idx: int) -> frozenset[int]:
    rings = []
    for ring in mol.GetRingInfo().AtomRings():
        ring_set = frozenset(ring)
        if atom_idx not in ring_set or len(ring_set) != 5:
            continue
        symbols = Counter(mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring_set)
        if symbols == Counter({"C": 4, "S": 1}):
            rings.append(ring_set)
    return rings[0] if len(rings) == 1 else frozenset()


def _has_assigned_bond_stereo(mol, Chem) -> bool:
    return any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds())


def _diphenylsulfonio_phenyl_sulfide_side_atoms(mol, Chem, ring_attachment_idx: int, central_sulfur_idx: int) -> set[int]:
    ring = _six_carbon_ring_for_atom(mol, ring_attachment_idx)
    if not ring or not all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
        return set()
    sulfonium_records = []
    for atom_idx in ring:
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring or nidx == central_sulfur_idx:
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
            if (
                nbr.GetSymbol() == "S"
                and nbr.GetFormalCharge() > 0
                and not nbr.IsInRing()
                and bond is not None
                and bond.GetBondType() == Chem.BondType.SINGLE
            ):
                sulfonium_records.append((atom_idx, nidx))
    if len(sulfonium_records) != 1:
        return set()
    sulfonium_ring_idx, sulfonium_idx = sulfonium_records[0]
    if _ring_distance(mol, ring, ring_attachment_idx, sulfonium_ring_idx) != 3:
        return set()
    for atom_idx in ring:
        external = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            if nbr.GetIdx() not in ring and nbr.GetAtomicNum() > 1
        ]
        allowed = set()
        if atom_idx == ring_attachment_idx:
            allowed.add(central_sulfur_idx)
        if atom_idx == sulfonium_ring_idx:
            allowed.add(sulfonium_idx)
        if set(external) != allowed:
            return set()

    sulfonium = mol.GetAtomWithIdx(sulfonium_idx)
    if sulfonium.GetTotalNumHs() != 0:
        return set()
    carbon_neighbors = []
    for nbr in sulfonium.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(sulfonium_idx, nidx)
        if nbr.GetSymbol() != "C" or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return set()
        carbon_neighbors.append(nidx)
    if len(carbon_neighbors) != 3 or sulfonium_ring_idx not in carbon_neighbors:
        return set()

    covered = set(ring) | {sulfonium_idx}
    phenyl_ligands = 0
    for nidx in carbon_neighbors:
        if nidx == sulfonium_ring_idx:
            continue
        phenyl_atoms = _unsubstituted_phenyl_ligand_atoms(mol, nidx, sulfonium_idx)
        if not phenyl_atoms:
            return set()
        covered.update(phenyl_atoms)
        phenyl_ligands += 1
    return covered if phenyl_ligands == 2 else set()


def _unsubstituted_phenyl_ligand_atoms(mol, ring_attachment_idx: int, parent_idx: int) -> set[int]:
    ring = _six_carbon_ring_for_atom(mol, ring_attachment_idx)
    if not ring or not all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
        return set()
    for atom_idx in ring:
        external = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            if nbr.GetIdx() not in ring and nbr.GetAtomicNum() > 1
        ]
        allowed = {parent_idx} if atom_idx == ring_attachment_idx else set()
        if set(external) != allowed:
            return set()
    return set(ring)


def _bis_dithiocarbamate_name(mol, Chem) -> str:
    methylene = _methylenebis_dithiocarbamate_name(mol, Chem)
    if methylene:
        return methylene
    linked = _linked_bis_dithiocarbamate_name(mol, Chem)
    if linked:
        return linked
    return ""


def _methylenebis_dithiocarbamate_name(mol, Chem) -> str:
    atom_set = set(range(mol.GetNumAtoms()))
    for carbon in mol.GetAtoms():
        cidx = carbon.GetIdx()
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
            continue
        heavy = [nbr.GetIdx() for nbr in carbon.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(heavy) != 2 or carbon.GetTotalNumHs() != 2:
            continue
        if any(mol.GetAtomWithIdx(nidx).GetSymbol() != "N" for nidx in heavy):
            continue
        if any(
            mol.GetBondBetweenAtoms(cidx, nidx).GetBondType() != Chem.BondType.SINGLE
            for nidx in heavy
        ):
            continue
        units = []
        covered = {cidx}
        for nidx in heavy:
            unit = _dithiocarbamate_unit_name_and_atoms(mol, Chem, nidx, cidx)
            if unit is None:
                break
            unit_name, unit_atoms = unit
            units.append(unit_name)
            covered.update(unit_atoms)
        else:
            if covered != atom_set or len(units) != 2 or units[0] != units[1]:
                continue
            return f"methylenebis({units[0]})"
    return ""


def _linked_bis_dithiocarbamate_name(mol, Chem) -> str:
    atom_set = set(range(mol.GetNumAtoms()))
    units = []
    for nitrogen in mol.GetAtoms():
        nidx = nitrogen.GetIdx()
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing():
            continue
        for linker in nitrogen.GetNeighbors():
            linker_idx = linker.GetIdx()
            if linker.GetAtomicNum() <= 1:
                continue
            unit = _dithiocarbamate_unit_name_and_atoms(mol, Chem, nidx, linker_idx)
            if unit is not None:
                units.append((nidx, linker_idx, *unit))
    for left_idx, left_linker, left_name, left_atoms in units:
        for right_idx, right_linker, right_name, right_atoms in units:
            if left_idx >= right_idx or left_name != right_name:
                continue
            linker_name, linker_atoms = _bis_dithiocarbamate_linker_name_and_atoms(
                mol,
                Chem,
                left_idx,
                left_linker,
                right_idx,
                right_linker,
            )
            if not linker_name:
                continue
            covered = set(left_atoms) | set(right_atoms) | linker_atoms
            if covered == atom_set:
                return f"{linker_name}bis({left_name})"
    return ""


def _bis_dithiocarbamate_linker_name_and_atoms(
    mol,
    Chem,
    left_nitrogen: int,
    left_linker: int,
    right_nitrogen: int,
    right_linker: int,
) -> tuple[str, set[int]]:
    if left_linker == right_linker:
        return "", set()
    left = mol.GetAtomWithIdx(left_linker)
    right = mol.GetAtomWithIdx(right_linker)
    linker_bond = mol.GetBondBetweenAtoms(left_linker, right_linker)
    if linker_bond is None or linker_bond.GetBondType() != Chem.BondType.SINGLE:
        return "", set()
    if (
        left.GetSymbol() == "C"
        and right.GetSymbol() == "C"
        and left.GetFormalCharge() == 0
        and right.GetFormalCharge() == 0
        and not left.IsInRing()
        and not right.IsInRing()
        and _linker_heavy_neighbors(mol, left_linker) == {left_nitrogen, right_linker}
        and _linker_heavy_neighbors(mol, right_linker) == {right_nitrogen, left_linker}
    ):
        return "ethane-1,2-diyl", {left_linker, right_linker}
    if (
        left.GetSymbol() == "S"
        and right.GetSymbol() == "S"
        and left.GetFormalCharge() == 0
        and right.GetFormalCharge() == 0
        and _linker_heavy_neighbors(mol, left_linker) == {left_nitrogen, right_linker}
        and _linker_heavy_neighbors(mol, right_linker) == {right_nitrogen, left_linker}
    ):
        return "dithio", {left_linker, right_linker}
    return "", set()


def _linker_heavy_neighbors(mol, atom_idx: int) -> set[int]:
    return {nbr.GetIdx() for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors() if nbr.GetAtomicNum() > 1}


def _dithiocarbamate_unit_name_and_atoms(mol, Chem, nitrogen_idx: int, linker_idx: int) -> tuple[str, set[int]] | None:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing():
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in nitrogen.GetNeighbors()
        if nbr.GetIdx() != linker_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 2:
        return None
    for carbamate_idx in heavy:
        owner_candidates = [idx for idx in heavy if idx != carbamate_idx]
        if len(owner_candidates) != 1:
            continue
        record = _dithiocarbamate_carbon_record(mol, Chem, carbamate_idx, nitrogen_idx)
        if record is None:
            continue
        n_owner_idx = owner_candidates[0]
        n_owner = _simple_or_branch_prefix(mol, n_owner_idx, nitrogen_idx)
        s_owner = _simple_or_branch_prefix(mol, record["single_sulfur_owner"], record["single_sulfur"])
        if not n_owner or n_owner != s_owner or not _simple_unparenthesized_n_substituent(n_owner):
            continue
        unit_name = f"di{n_owner}dithiocarbamate"
        covered = {
            nitrogen_idx,
            carbamate_idx,
            record["double_sulfur"],
            record["single_sulfur"],
        }
        covered.update(_collect_component_atoms(mol, n_owner_idx, nitrogen_idx))
        covered.update(_collect_component_atoms(mol, record["single_sulfur_owner"], record["single_sulfur"]))
        return unit_name, covered
    return None


def _dithiocarbamate_name_and_atoms(
    mol,
    nitrogen_idx: int,
    carbon_idx: int,
    record: dict,
) -> tuple[str, set[int]] | None:
    n_substituents = _external_substituent_names(mol, nitrogen_idx, carbon_idx)
    if not n_substituents:
        return None
    n_prefix = _n_substituent_prefix(n_substituents)
    if not n_prefix:
        return None
    s_owner = _simple_or_branch_prefix(mol, record["single_sulfur_owner"], record["single_sulfur"])
    if not s_owner or _unstable_retained_functional_parent_substituent(s_owner):
        return None
    name = f"{_hetero_ligand_role_text('S', s_owner)}-{n_prefix}dithiocarbamate"
    covered = {
        nitrogen_idx,
        carbon_idx,
        record["double_sulfur"],
        record["single_sulfur"],
    }
    for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx != carbon_idx and nbr.GetAtomicNum() > 1:
            covered.update(_collect_component_atoms(mol, nidx, nitrogen_idx))
    covered.update(_collect_component_atoms(mol, record["single_sulfur_owner"], record["single_sulfur"]))
    return name, covered


def _dithiocarbamate_carbon_record(mol, Chem, carbon_idx: int, nitrogen_idx: int) -> dict | None:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
        return None
    double_sulfur = None
    single_sulfur = None
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if nidx == nitrogen_idx and bond.GetBondType() == Chem.BondType.SINGLE:
            continue
        if nbr.GetSymbol() != "S":
            return None
        if bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            if double_sulfur is not None:
                return None
            external = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != carbon_idx and atom.GetAtomicNum() > 1
            ]
            if external:
                return None
            double_sulfur = nidx
        elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            if single_sulfur is not None:
                return None
            owners = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != carbon_idx and atom.GetAtomicNum() > 1
            ]
            if len(owners) != 1:
                return None
            single_sulfur = nidx
            single_sulfur_owner = owners[0]
        else:
            return None
    if double_sulfur is None or single_sulfur is None:
        return None
    return {
        "double_sulfur": double_sulfur,
        "single_sulfur": single_sulfur,
        "single_sulfur_owner": single_sulfur_owner,
    }


def _dithiocarbonate_record(mol, Chem, carbon_idx: int) -> dict | None:
    double_sulfur = None
    oxygen_owner = None
    sulfur_owners = []
    sulfur_owner_has_nitrile = []
    terminal_sulfido = 0
    covered_atoms = {carbon_idx}
    for nbr in mol.GetAtomWithIdx(carbon_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            if double_sulfur is not None:
                return None
            if _heavy_degree_excluding(mol, nidx, carbon_idx) != 0:
                return None
            double_sulfur = nidx
            covered_atoms.add(nidx)
            continue
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == 0:
            if oxygen_owner is not None:
                return None
            owner = _single_external_substituent_name(mol, nidx, carbon_idx)
            if not owner or _unstable_retained_functional_parent_substituent(owner):
                return None
            owner_atom = _single_heavy_neighbor_excluding(mol, nidx, carbon_idx)
            if owner_atom is None:
                return None
            oxygen_owner = owner
            covered_atoms.add(nidx)
            covered_atoms.update(_collect_component_atoms(mol, owner_atom, nidx))
            continue
        if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() <= 0:
            owner_atom = _single_heavy_neighbor_excluding(mol, nidx, carbon_idx)
            if owner_atom is None:
                if nbr.GetFormalCharge() < 0:
                    terminal_sulfido += 1
                    covered_atoms.add(nidx)
                    continue
                return None
            owner = _simple_or_branch_prefix(mol, owner_atom, nidx)
            if not owner or _unstable_retained_functional_parent_substituent(owner):
                return None
            owner_atoms = _collect_component_atoms(mol, owner_atom, nidx)
            sulfur_owners.append(owner)
            sulfur_owner_has_nitrile.append(any(_is_nitrile_carbon(mol, atom_idx) for atom_idx in owner_atoms))
            covered_atoms.add(nidx)
            covered_atoms.update(owner_atoms)
            continue
        return None
    if double_sulfur is None:
        return None
    return {
        "oxygen_owner": oxygen_owner,
        "sulfur_owners": tuple(sulfur_owners),
        "sulfur_owner_has_nitrile": tuple(sulfur_owner_has_nitrile),
        "terminal_sulfido": terminal_sulfido,
        "covered_atoms": covered_atoms,
    }


def _dithiocarbonate_record_name(record: dict) -> str:
    oxygen_owner = record["oxygen_owner"]
    sulfur_owners = record["sulfur_owners"]
    sulfur_owner_has_nitrile = record["sulfur_owner_has_nitrile"]
    terminal_sulfido = record["terminal_sulfido"]
    if oxygen_owner and len(sulfur_owners) == 1 and terminal_sulfido == 0:
        return (
            f"{_hetero_ligand_role_text('O', oxygen_owner)} "
            f"{_hetero_ligand_role_text('S', sulfur_owners[0])}dithiocarbonate"
        )
    if oxygen_owner is None and len(sulfur_owners) == 1 and terminal_sulfido == 1:
        if sulfur_owner_has_nitrile[0]:
            return ""
        return f"{_hetero_ligand_role_text('S', sulfur_owners[0])}trithiocarbonate"
    return ""


def _hexamethylenediamide_name(mol, Chem) -> str:
    nitrogens = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "N" and atom.GetFormalCharge() == -1 and not atom.IsInRing()
    ]
    if len(nitrogens) != 2:
        return ""
    first_n, second_n = nitrogens
    path = tuple(Chem.rdmolops.GetShortestPath(mol, first_n, second_n))
    if len(path) != 8 or path[0] != first_n or path[-1] != second_n:
        return ""
    linker = path[1:-1]
    if not _is_unsubstituted_saturated_carbon_path(mol, Chem, linker, first_n, second_n):
        return ""
    units = []
    covered = set(linker)
    for nidx, linker_idx in ((first_n, path[1]), (second_n, path[-2])):
        unit = _amidate_acyl_unit_name_and_atoms(mol, Chem, nidx, linker_idx)
        if unit is None:
            return ""
        unit_name, unit_atoms = unit
        units.append(unit_name)
        covered.update(unit_atoms)
    if covered != set(range(mol.GetNumAtoms())) or units[0] != units[1]:
        return ""
    return f"bis({units[0]})hexamethylenediamide"


def _is_unsubstituted_saturated_carbon_path(
    mol,
    Chem,
    path: Sequence[int],
    start_terminal_idx: int,
    end_terminal_idx: int,
) -> bool:
    path_set = set(path)
    for pos, idx in enumerate(path):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            return False
        allowed = set()
        if pos == 0:
            allowed.add(start_terminal_idx)
        else:
            allowed.add(path[pos - 1])
        if pos == len(path) - 1:
            allowed.add(end_terminal_idx)
        else:
            allowed.add(path[pos + 1])
        heavy = {nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1}
        if heavy != allowed:
            return False
        if any(
            mol.GetBondBetweenAtoms(idx, nidx).GetBondType() != Chem.BondType.SINGLE
            for nidx in heavy
        ):
            return False
    return start_terminal_idx not in path_set and end_terminal_idx not in path_set


def _amidate_acyl_unit_name_and_atoms(mol, Chem, nitrogen_idx: int, linker_idx: int) -> tuple[str, set[int]] | None:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != -1:
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in nitrogen.GetNeighbors()
        if nbr.GetIdx() != linker_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    carbonyl_idx = heavy[0]
    if mol.GetBondBetweenAtoms(nitrogen_idx, linker_idx).GetBondType() != Chem.BondType.SINGLE:
        return None
    if mol.GetBondBetweenAtoms(nitrogen_idx, carbonyl_idx).GetBondType() != Chem.BondType.SINGLE:
        return None
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0 or carbonyl.IsInRing():
        return None
    double_oxygen = []
    acyl_carbons = []
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nidx == nitrogen_idx and bond.GetBondType() == Chem.BondType.SINGLE:
            continue
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen.append(nidx)
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            acyl_carbons.append(nidx)
        else:
            return None
    if len(double_oxygen) != 1 or len(acyl_carbons) != 1:
        return None
    acyl_idx = acyl_carbons[0]
    names = []
    for path in _ester_acyl_parent_paths(mol, Chem, carbonyl_idx, acyl_idx):
        name = _generic_acyl_ligand_path_name(mol, Chem, carbonyl_idx, path)
        if name:
            names.append(name)
    if not names:
        return None
    unit_atoms = {nitrogen_idx, carbonyl_idx, double_oxygen[0]}
    unit_atoms.update(_collect_component_atoms(mol, acyl_idx, carbonyl_idx))
    return min(names, key=lambda value: (len(value), value)), unit_atoms


def _generic_acyl_ligand_path_name(mol, Chem, carbonyl_idx: int, path: Sequence[int]) -> str:
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
    parent = _generic_acyl_ligand_parent_text(mol, Chem, root, path)
    if not parent:
        return ""
    prefix_text = _format_prefixes(prefixes)
    core = _join_prefix_parent(prefix_text, parent) if prefix_text else parent
    stereo = _generic_acyl_ester_stereo_prefix(mol, Chem, path)
    if stereo is None:
        return ""
    return f"({stereo})-{core}" if stereo else core


def _generic_acyl_ligand_parent_text(mol, Chem, root: str, path: Sequence[int]) -> str:
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
    parent = _unsaturated_parent(root, ene_locs, yne_locs)
    return parent[:-1] + "oyl" if parent.endswith("e") else parent + "oyl"


def _ring_distance(mol, ring_set: frozenset[int], first_idx: int, second_idx: int) -> int | None:
    if first_idx not in ring_set or second_idx not in ring_set:
        return None
    queue = [(first_idx, 0)]
    seen = {first_idx}
    while queue:
        atom_idx, distance = queue.pop(0)
        if atom_idx == second_idx:
            return distance
        for nbr_idx in _ring_neighbors(mol, ring_set, atom_idx):
            if nbr_idx in seen:
                continue
            seen.add(nbr_idx)
            queue.append((nbr_idx, distance + 1))
    return None


def _biphenyl_diphosphonite_phosphorus_group(mol, Chem, pidx: int) -> dict | None:
    phosphorus = mol.GetAtomWithIdx(pidx)
    if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0 or phosphorus.IsInRing():
        return None
    ring_carbons = []
    terminal_oxygens = []
    ester_oxygen = None
    ester_owner = None
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        if nbr.GetSymbol() == "C" and nbr.IsInRing():
            ring_carbons.append(nidx)
            continue
        if nbr.GetSymbol() != "O":
            return None
        heavy = [atom.GetIdx() for atom in nbr.GetNeighbors() if atom.GetIdx() != pidx and atom.GetAtomicNum() > 1]
        if not heavy and nbr.GetFormalCharge() < 0:
            terminal_oxygens.append(nidx)
            continue
        if len(heavy) == 1 and nbr.GetFormalCharge() == 0 and ester_oxygen is None:
            ester_oxygen = nidx
            ester_owner = heavy[0]
            continue
        return None
    if len(ring_carbons) != 1:
        return None
    if len(terminal_oxygens) + (1 if ester_oxygen is not None else 0) != 2:
        return None
    return {
        "phosphorus": pidx,
        "ring_carbon": ring_carbons[0],
        "terminal_oxygens": tuple(terminal_oxygens),
        "ester_oxygen": ester_oxygen,
        "ester_owner": ester_owner,
    }


def _six_carbon_ring_for_atom(mol, atom_idx: int) -> frozenset[int]:
    rings = []
    for ring in mol.GetRingInfo().AtomRings():
        if atom_idx not in ring or len(ring) != 6:
            continue
        if all(mol.GetAtomWithIdx(idx).GetSymbol() == "C" for idx in ring):
            rings.append(frozenset(ring))
    return rings[0] if len(rings) == 1 else frozenset()


def _single_biphenyl_bridge(mol, Chem, first_ring: frozenset[int], second_ring: frozenset[int]) -> tuple[int, int] | None:
    bridges = []
    for first_idx in first_ring:
        for nbr in mol.GetAtomWithIdx(first_idx).GetNeighbors():
            second_idx = nbr.GetIdx()
            if second_idx not in second_ring:
                continue
            bond = mol.GetBondBetweenAtoms(first_idx, second_idx)
            if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
                bridges.append((first_idx, second_idx))
    return bridges[0] if len(bridges) == 1 else None


def _biphenyl_ring_locant(
    mol,
    ring: frozenset[int],
    substituent_atom_idx: int,
    bridge_atoms: dict[int, int],
) -> int:
    bridge_idx = next((idx for idx in ring if idx in bridge_atoms), None)
    if bridge_idx is None:
        return 0
    distance = _ring_distance(mol, ring, bridge_idx, substituent_atom_idx)
    if distance == 2:
        return 3
    if distance == 3:
        return 4
    return 0


def _ring_distance(mol, ring: frozenset[int], start_idx: int, end_idx: int) -> int:
    frontier = [(start_idx, 0)]
    seen = {start_idx}
    while frontier:
        current, distance = frontier.pop(0)
        if current == end_idx:
            return distance
        for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in ring or nidx in seen:
                continue
            seen.add(nidx)
            frontier.append((nidx, distance + 1))
    return 0


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


def _carbonyl_double_oxygen(mol, Chem, carbonyl_idx: int) -> int | None:
    oxygens = []
    for nbr in mol.GetAtomWithIdx(carbonyl_idx).GetNeighbors():
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nbr.GetIdx())
        if (
            nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() == 0
            and bond is not None
            and bond.GetBondType() == Chem.BondType.DOUBLE
        ):
            oxygens.append(nbr.GetIdx())
    return oxygens[0] if len(oxygens) == 1 else None


def _retained_acid_ester_names(mol, Chem) -> list[str]:
    return [name for name, _source_family in _retained_acid_ester_records(mol, Chem)]


def _retained_acid_ester_records(mol, Chem) -> list[tuple[str, str]]:
    names = _acyl_diphosphate_records(mol, Chem)
    for carbonyl_idx, ester_oxygen_idx, acyl_idx in _ester_acyl_motifs(mol, Chem):
        owner = _stable_ester_owner_name(mol, ester_oxygen_idx, carbonyl_idx)
        if owner:
            generic_source = _retained_acid_ester_source_family(mol, Chem, carbonyl_idx, acyl_idx, "generic")
            generic = _generic_acyl_ester_names(mol, Chem, carbonyl_idx, acyl_idx, owner)
            names.extend((name, generic_source) for name in generic)
            dicarboxylate = _dicarboxylate_ester_name(mol, Chem, carbonyl_idx, acyl_idx, owner)
            if dicarboxylate:
                names.append((dicarboxylate, "retained-acid-ester:dicarboxylate"))
            acrylate = _acrylate_ester_name(mol, Chem, carbonyl_idx, acyl_idx, owner)
            if acrylate:
                names.append((acrylate, "retained-acid-ester:acrylate"))
            continue
        dicarboxylate = _acyl_owner_dicarboxylate_ester_name(mol, Chem, carbonyl_idx, ester_oxygen_idx, acyl_idx)
        if dicarboxylate:
            names.append((dicarboxylate, "retained-acid-ester:dicarboxylate-acyl-owner"))
    return names


def _acyl_diphosphate_records(mol, Chem) -> list[tuple[str, str]]:
    return [(name, source_family) for name, source_family, _atoms in _acyl_diphosphate_parent_records(mol, Chem)]


def _acyl_diphosphate_parent_records(mol, Chem) -> list[tuple[str, str, frozenset[int]]]:
    records = []
    for carbonyl_idx, ester_oxygen_idx, acyl_idx in _ester_acyl_motifs(mol, Chem):
        record = _acyl_diphosphate_record(mol, Chem, carbonyl_idx, ester_oxygen_idx, acyl_idx)
        if record:
            name, parent_atoms = record
            records.append((name, "retained-acid-ester:acyl-diphosphate", parent_atoms))
    return records


def _acyl_diphosphate_name(mol, Chem, carbonyl_idx: int, ester_oxygen_idx: int, acyl_idx: int) -> str:
    record = _acyl_diphosphate_record(mol, Chem, carbonyl_idx, ester_oxygen_idx, acyl_idx)
    return record[0] if record else ""


def _acyl_diphosphate_record(
    mol,
    Chem,
    carbonyl_idx: int,
    ester_oxygen_idx: int,
    acyl_idx: int,
) -> tuple[str, frozenset[int]] | None:
    diphosphate_atoms = _acyl_diphosphate_atoms(mol, Chem, carbonyl_idx, ester_oxygen_idx)
    if not diphosphate_atoms:
        return None
    carbonyl_oxygen = _carbonyl_double_oxygen(mol, Chem, carbonyl_idx)
    if carbonyl_oxygen is None:
        return None
    covered_atoms = {carbonyl_idx, carbonyl_oxygen}
    covered_atoms.update(diphosphate_atoms)
    covered_atoms.update(_collect_component_atoms(mol, acyl_idx, carbonyl_idx))
    if covered_atoms != set(range(mol.GetNumAtoms())):
        return None
    names = []
    for path in _ester_acyl_parent_paths(mol, Chem, carbonyl_idx, acyl_idx):
        acyl_name = _generic_acyl_ligand_path_name(mol, Chem, carbonyl_idx, path)
        if acyl_name:
            names.append(f"{acyl_name} diphosphate")
    if not names:
        return None
    return min(names, key=lambda value: (len(value), value)), frozenset(covered_atoms)


def _acyl_diphosphate_atoms(mol, Chem, carbonyl_idx: int, ester_oxygen_idx: int) -> set[int] | None:
    ester_oxygen = mol.GetAtomWithIdx(ester_oxygen_idx)
    if ester_oxygen.GetSymbol() != "O" or ester_oxygen.GetFormalCharge() != 0:
        return None
    carbonyl_bond = mol.GetBondBetweenAtoms(carbonyl_idx, ester_oxygen_idx)
    if carbonyl_bond is None or carbonyl_bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    phosphorus_neighbors = []
    for nbr in ester_oxygen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbonyl_idx or nbr.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(ester_oxygen_idx, nidx)
        if nbr.GetSymbol() == "P" and bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            phosphorus_neighbors.append(nidx)
        else:
            return None
    if len(phosphorus_neighbors) != 1:
        return None
    return _diphosphate_chain_atoms(mol, Chem, phosphorus_neighbors[0], ester_oxygen_idx)


def _diphosphate_chain_atoms(mol, Chem, first_phosphorus_idx: int, ester_oxygen_idx: int) -> set[int] | None:
    first = _diphosphate_first_phosphorus_atoms(mol, Chem, first_phosphorus_idx, ester_oxygen_idx)
    if not first:
        return None
    second = _diphosphate_terminal_phosphorus_atoms(mol, Chem, first["second_phosphorus"], first["bridge_oxygen"])
    if not second:
        return None
    atoms = {ester_oxygen_idx, first_phosphorus_idx}
    atoms.update(first["atoms"])
    atoms.update(second)
    return atoms


def _diphosphate_first_phosphorus_atoms(
    mol,
    Chem,
    phosphorus_idx: int,
    ester_oxygen_idx: int,
) -> dict | None:
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)
    if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0 or phosphorus.IsInRing():
        return None
    if phosphorus.GetTotalNumHs() != 0:
        return None
    double_oxygen = []
    terminal_oxido = []
    bridge_records = []
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(phosphorus_idx, nidx)
        if nidx == ester_oxygen_idx:
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            continue
        if nbr.GetSymbol() != "O" or bond is None:
            return None
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            if nbr.GetFormalCharge() != 0 or _heavy_degree_excluding(mol, nidx, phosphorus_idx) != 0:
                return None
            double_oxygen.append(nidx)
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        heavy = [
            atom.GetIdx()
            for atom in nbr.GetNeighbors()
            if atom.GetIdx() != phosphorus_idx and atom.GetAtomicNum() > 1
        ]
        if not heavy and nbr.GetFormalCharge() < 0:
            terminal_oxido.append(nidx)
            continue
        if len(heavy) == 1 and nbr.GetFormalCharge() == 0 and mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "P":
            bridge_records.append((nidx, heavy[0]))
            continue
        return None
    if len(double_oxygen) != 1 or len(terminal_oxido) != 1 or len(bridge_records) != 1:
        return None
    bridge_oxygen, second_phosphorus = bridge_records[0]
    return {
        "atoms": {double_oxygen[0], terminal_oxido[0], bridge_oxygen},
        "bridge_oxygen": bridge_oxygen,
        "second_phosphorus": second_phosphorus,
    }


def _diphosphate_terminal_phosphorus_atoms(
    mol,
    Chem,
    phosphorus_idx: int,
    bridge_oxygen_idx: int,
) -> set[int] | None:
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)
    if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0 or phosphorus.IsInRing():
        return None
    if phosphorus.GetTotalNumHs() != 0:
        return None
    double_oxygen = []
    terminal_oxido = []
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(phosphorus_idx, nidx)
        if nidx == bridge_oxygen_idx:
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            continue
        if nbr.GetSymbol() != "O" or bond is None:
            return None
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            if nbr.GetFormalCharge() != 0 or _heavy_degree_excluding(mol, nidx, phosphorus_idx) != 0:
                return None
            double_oxygen.append(nidx)
            continue
        if bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            if _heavy_degree_excluding(mol, nidx, phosphorus_idx) != 0:
                return None
            terminal_oxido.append(nidx)
            continue
        return None
    if len(double_oxygen) != 1 or len(terminal_oxido) != 2:
        return None
    return {phosphorus_idx, double_oxygen[0], *terminal_oxido}


def _heavy_degree_excluding(mol, atom_idx: int, excluded_idx: int) -> int:
    return sum(
        1
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if nbr.GetIdx() != excluded_idx and nbr.GetAtomicNum() > 1
    )


def _retained_acid_ester_source_family(mol, Chem, carbonyl_idx: int, acyl_idx: int, kind: str) -> str:
    source = f"retained-acid-ester:{kind}"
    if kind == "generic" and _has_stable_retained_acid_ester_graph(mol, Chem, carbonyl_idx, acyl_idx):
        return f"{source}:stable-retained"
    return source


def _has_stable_retained_acid_ester_graph(mol, Chem, carbonyl_idx: int, acyl_idx: int) -> bool:
    if any(atom.GetSymbol() in {"I", "Si"} for atom in mol.GetAtoms()):
        return True
    if _has_isothiocyanate_motif(mol, Chem):
        return True
    return _is_simple_alpha_amino_ester(mol, Chem, carbonyl_idx, acyl_idx)


def _has_isothiocyanate_motif(mol, Chem) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        double_nitrogen = False
        double_sulfur = False
        atom_idx = atom.GetIdx()
        for nbr in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx())
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            if nbr.GetSymbol() == "N":
                double_nitrogen = True
            elif nbr.GetSymbol() == "S":
                double_sulfur = True
        if double_nitrogen and double_sulfur:
            return True
    return False


def _is_simple_alpha_amino_ester(mol, Chem, carbonyl_idx: int, acyl_idx: int) -> bool:
    acyl_atom = mol.GetAtomWithIdx(acyl_idx)
    if acyl_atom.GetSymbol() != "C":
        return False
    has_alpha_nitrogen = False
    for nbr in acyl_atom.GetNeighbors():
        if nbr.GetIdx() == carbonyl_idx or nbr.GetSymbol() != "N":
            continue
        if nbr.GetFormalCharge() != 0:
            continue
        bond = mol.GetBondBetweenAtoms(acyl_idx, nbr.GetIdx())
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            has_alpha_nitrogen = True
            break
    return has_alpha_nitrogen and _neutral_carbonyl_count(mol, Chem) == 1


def _neutral_carbonyl_count(mol, Chem) -> int:
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
            continue
        atom_idx = atom.GetIdx()
        for nbr in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx())
            if nbr.GetSymbol() == "O" and bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
                count += 1
                break
    return count


def _retained_amide_names(mol, Chem) -> list[str]:
    return [name for name, _source_family, _parent_atoms in _retained_amide_records(mol, Chem)]


def _retained_amide_records(mol, Chem) -> list[tuple[str, str, frozenset[int]]]:
    records = []
    for carbonyl_idx, nitrogen_idx, acyl_idx in _amide_acyl_motifs(mol, Chem):
        acyl_parent_paths = _ester_acyl_parent_paths(mol, Chem, carbonyl_idx, acyl_idx)
        has_ring_carboxamide_parent = _has_ring_carboxamide_parent_candidate_ring(mol, acyl_idx)
        if not acyl_parent_paths and not has_ring_carboxamide_parent:
            continue
        n_substituents = _external_substituent_names(mol, nitrogen_idx, carbonyl_idx)
        if n_substituents is None or len(n_substituents) > 2:
            continue
        n_prefix = _n_substituent_prefix(n_substituents)
        if n_prefix is None:
            continue
        if any(_unstable_retained_functional_parent_substituent(name) for name in n_substituents):
            continue
        for path in acyl_parent_paths:
            core = _generic_acyl_functional_parent_path_name(mol, Chem, carbonyl_idx, path, "amide")
            if not core:
                continue
            parent_atoms = _complete_oxalamide_carboxylate_parent_atoms(
                mol,
                Chem,
                carbonyl_idx,
                nitrogen_idx,
                path,
            )
            if not parent_atoms:
                parent_atoms = _complete_dimethyl_amide_carbamate_path_parent_atoms(
                    mol,
                    Chem,
                    carbonyl_idx,
                    nitrogen_idx,
                    path,
                )
            records.append((_join_n_prefix_functional_parent(n_prefix, core), "generic-acyl-amide", parent_atoms))
        if has_ring_carboxamide_parent:
            records.extend(
                (name, source_family, frozenset())
                for name, source_family in _ring_carboxamide_parent_records(
                    mol,
                    Chem,
                    carbonyl_idx,
                    nitrogen_idx,
                    acyl_idx,
                    n_prefix,
                )
            )
    return records


def _has_ring_carboxamide_parent_candidate_ring(mol, acyl_idx: int) -> bool:
    if not mol.GetAtomWithIdx(acyl_idx).IsInRing():
        return False
    for ring in mol.GetRingInfo().AtomRings():
        if acyl_idx in ring and not _ring_is_fused_to_another_ring(mol, ring) and ordered_ring(mol, ring):
            return True
    return False


def _complete_oxalamide_carboxylate_parent_atoms(
    mol,
    Chem,
    carbonyl_idx: int,
    nitrogen_idx: int,
    path: Sequence[int],
) -> frozenset[int]:
    if len(path) != 1:
        return frozenset()
    carboxyl_idx = path[0]
    if _terminal_carboxyl_group_kind(mol, Chem, carboxyl_idx, carbonyl_idx) != "carboxylate":
        return frozenset()
    amide_oxygen_idx = _carbonyl_double_oxygen(mol, Chem, carbonyl_idx)
    if amide_oxygen_idx is None:
        return frozenset()
    covered = {carbonyl_idx, amide_oxygen_idx, nitrogen_idx, carboxyl_idx}
    for nbr in mol.GetAtomWithIdx(carboxyl_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbonyl_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carboxyl_idx, nidx)
        if nbr.GetSymbol() != "O" or bond is None:
            return frozenset()
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            covered.add(nidx)
        elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == -1:
            covered.add(nidx)
        else:
            return frozenset()
    for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbonyl_idx or nbr.GetAtomicNum() <= 1:
            continue
        covered.update(_collect_component_atoms(mol, nidx, nitrogen_idx))
    heavy_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}
    return frozenset(covered) if covered == heavy_atoms else frozenset()


def _complete_dimethyl_amide_carbamate_path_parent_atoms(
    mol,
    Chem,
    carbonyl_idx: int,
    nitrogen_idx: int,
    path: Sequence[int],
) -> frozenset[int]:
    if not _amide_nitrogen_has_two_methyl_substituents(mol, Chem, nitrogen_idx, carbonyl_idx):
        return frozenset()
    covered = _retained_amide_path_covered_atoms(mol, Chem, carbonyl_idx, nitrogen_idx, path)
    heavy_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}
    if covered != heavy_atoms:
        return frozenset()
    if not _covered_atoms_have_n_methyl_carboxylate_substituent(mol, Chem, covered):
        return frozenset()
    return frozenset(covered)


def _retained_amide_path_covered_atoms(
    mol,
    Chem,
    carbonyl_idx: int,
    nitrogen_idx: int,
    path: Sequence[int],
) -> set[int]:
    amide_oxygen_idx = _carbonyl_double_oxygen(mol, Chem, carbonyl_idx)
    if amide_oxygen_idx is None:
        return set()
    path_set = set(path)
    covered = {carbonyl_idx, amide_oxygen_idx, nitrogen_idx, *path_set}
    for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbonyl_idx or nbr.GetAtomicNum() <= 1:
            continue
        covered.update(_collect_component_atoms(mol, nidx, nitrogen_idx))
    for pos, atom_idx in enumerate(path, start=2):
        allowed_neighbors = {carbonyl_idx if pos == 2 else path[pos - 3]}
        if pos - 1 < len(path):
            allowed_neighbors.add(path[pos - 1])
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in allowed_neighbors or nidx in path_set or nbr.GetAtomicNum() <= 1:
                continue
            covered.update(_collect_component_atoms(mol, nidx, atom_idx))
    return covered


def _amide_nitrogen_has_two_methyl_substituents(mol, Chem, nitrogen_idx: int, carbonyl_idx: int) -> bool:
    substituents = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors()
        if nbr.GetIdx() != carbonyl_idx and nbr.GetAtomicNum() > 1
    ]
    return len(substituents) == 2 and all(_is_terminal_methyl_substituent(mol, Chem, idx, nitrogen_idx) for idx in substituents)


def _covered_atoms_have_n_methyl_carboxylate_substituent(mol, Chem, covered: set[int]) -> bool:
    for atom_idx in covered:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            continue
        neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(neighbors) != 3:
            continue
        carboxylate = [
            idx
            for idx in neighbors
            if _is_n_carboxylate_carbon(mol, Chem, idx, atom_idx)
        ]
        methyl = [
            idx
            for idx in neighbors
            if _is_terminal_methyl_substituent(mol, Chem, idx, atom_idx)
        ]
        if len(carboxylate) != 1 or len(methyl) != 1:
            continue
        attachment = [idx for idx in neighbors if idx not in {carboxylate[0], methyl[0]}]
        if len(attachment) == 1 and attachment[0] in covered:
            return True
    return False


def _is_n_carboxylate_carbon(mol, Chem, carbon_idx: int, nitrogen_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
        return False
    bond_to_nitrogen = mol.GetBondBetweenAtoms(carbon_idx, nitrogen_idx)
    if bond_to_nitrogen is None or bond_to_nitrogen.GetBondType() != Chem.BondType.SINGLE:
        return False
    double_oxygen = 0
    oxido_oxygen = 0
    other_heavy = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if nbr.GetSymbol() == "O" and bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen += 1
        elif (
            nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() == -1
            and bond is not None
            and bond.GetBondType() == Chem.BondType.SINGLE
        ):
            oxido_oxygen += 1
        elif nbr.GetAtomicNum() > 1:
            other_heavy.append(nidx)
    return double_oxygen == 1 and oxido_oxygen == 1 and not other_heavy


def _is_terminal_methyl_substituent(mol, Chem, carbon_idx: int, attached_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
        return False
    bond = mol.GetBondBetweenAtoms(carbon_idx, attached_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    heavy_neighbors = [nbr.GetIdx() for nbr in carbon.GetNeighbors() if nbr.GetAtomicNum() > 1]
    return heavy_neighbors == [attached_idx]


def _ring_carboxamide_parent_names(
    mol,
    Chem,
    carbonyl_idx: int,
    nitrogen_idx: int,
    acyl_idx: int,
    n_prefix: str,
) -> list[str]:
    return [
        name
        for name, _source_family in _ring_carboxamide_parent_records(
            mol,
            Chem,
            carbonyl_idx,
            nitrogen_idx,
            acyl_idx,
            n_prefix,
        )
    ]


def _ring_carboxamide_parent_records(
    mol,
    Chem,
    carbonyl_idx: int,
    nitrogen_idx: int,
    acyl_idx: int,
    n_prefix: str,
) -> list[tuple[str, str]]:
    if not mol.GetAtomWithIdx(acyl_idx).IsInRing():
        return []
    records = []
    for ring in mol.GetRingInfo().AtomRings():
        if acyl_idx not in ring or _ring_is_fused_to_another_ring(mol, ring):
            continue
        ordered = ordered_ring(mol, ring)
        if not ordered:
            continue
        candidates = _merge_ring_parent_candidates(
            _ring_parent_candidates(mol, ordered),
            _resource_ring_parent_candidates(mol, ordered, Chem),
        )
        for parent_name, seq in candidates:
            name = _ring_carboxamide_parent_name(
                mol,
                parent_name,
                seq,
                carbonyl_idx,
                nitrogen_idx,
                acyl_idx,
                n_prefix,
            )
            if name:
                records.append((name, _ring_carboxamide_source_family(mol, ring)))
    return records


def _ring_carbohydrazide_parent_records(
    mol,
    Chem,
    carbonyl_idx: int,
    nitrogen_idx: int,
    acyl_idx: int,
    n_prime_prefix: str,
) -> list[tuple[str, str]]:
    if not mol.GetAtomWithIdx(acyl_idx).IsInRing():
        return []
    records = []
    for ring in mol.GetRingInfo().AtomRings():
        if acyl_idx not in ring or _ring_is_fused_to_another_ring(mol, ring):
            continue
        ordered = ordered_ring(mol, ring)
        if not ordered:
            continue
        candidates = _merge_ring_parent_candidates(
            _ring_parent_candidates(mol, ordered),
            _resource_ring_parent_candidates(mol, ordered, Chem),
        )
        for parent_name, seq in candidates:
            name = _ring_carbohydrazide_parent_name(
                mol,
                parent_name,
                seq,
                carbonyl_idx,
                nitrogen_idx,
                acyl_idx,
                n_prime_prefix,
            )
            if name:
                records.append((name, _ring_carboxamide_source_family(mol, ring).replace("carboxamide", "carbohydrazide")))
    return records


def _ring_carboxamide_source_family(mol, ring: Sequence[int]) -> str:
    symbols = {mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring}
    if "Se" in symbols:
        return "ring-carboxamide:selenium"
    if "Te" in symbols:
        return "ring-carboxamide:tellurium"
    if symbols.intersection({"S", "O"}):
        return "ring-carboxamide:chalcogen"
    return "ring-carboxamide"


def _ring_is_fused_to_another_ring(mol, ring: Sequence[int]) -> bool:
    ring_set = set(ring)
    for other in mol.GetRingInfo().AtomRings():
        other_set = set(other)
        if other_set != ring_set and ring_set & other_set:
            return True
    return False


def _ring_carboxamide_parent_name(
    mol,
    parent_name: str,
    seq: Sequence[int],
    carbonyl_idx: int,
    nitrogen_idx: int,
    acyl_idx: int,
    n_prefix: str,
) -> str:
    ring_set = set(seq)
    raw_subs = []
    for ring_idx in seq:
        atom = mol.GetAtomWithIdx(ring_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set or nidx == carbonyl_idx:
                continue
            prefix = simple_external_prefix(mol, nidx, ring_idx) or branch_prefix(
                mol,
                nidx,
                ring_idx,
                assume_simple_external_prefix_miss=True,
            )
            if not prefix:
                return ""
            raw_subs.append((ring_idx, prefix))
    adjusted_parent = _adjust_ring_parent_for_charge(mol, parent_name, seq, raw_subs)
    if not adjusted_parent:
        return ""
    adjusted_parent = _append_ring_parent_isotope_suffixes(mol, adjusted_parent, seq)
    adjusted_parent = _ring_lambda_parent_name(mol, adjusted_parent, seq)
    adjusted_parent = _opsin_parseable_retained_parent_name(adjusted_parent)
    pos = {idx: locant for locant, idx in enumerate(seq, start=1)}
    suffix_locant = str(pos.get(acyl_idx, ""))
    if not suffix_locant:
        return ""
    prefixes = [
        (_single_ring_substituent_locant(adjusted_parent, pos[idx]), prefix)
        for idx, prefix in raw_subs
    ]
    prefix_text = _format_prefixes(prefixes)
    if prefixes and not prefix_text:
        return ""
    core = _ring_carboxamide_core(adjusted_parent, suffix_locant)
    if not core:
        return ""
    core = _join_n_prefix_functional_parent(n_prefix, core)
    return _join_prefix_parent(prefix_text, core) if prefix_text else core


def _ring_carbohydrazide_parent_name(
    mol,
    parent_name: str,
    seq: Sequence[int],
    carbonyl_idx: int,
    nitrogen_idx: int,
    acyl_idx: int,
    n_prime_prefix: str,
) -> str:
    ring_set = set(seq)
    raw_subs = []
    for ring_idx in seq:
        atom = mol.GetAtomWithIdx(ring_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set or nidx == carbonyl_idx:
                continue
            prefix = simple_external_prefix(mol, nidx, ring_idx) or branch_prefix(
                mol,
                nidx,
                ring_idx,
                assume_simple_external_prefix_miss=True,
            )
            if not prefix:
                return ""
            raw_subs.append((ring_idx, prefix))
    adjusted_parent = _adjust_ring_parent_for_charge(mol, parent_name, seq, raw_subs)
    if not adjusted_parent:
        return ""
    adjusted_parent = _append_ring_parent_isotope_suffixes(mol, adjusted_parent, seq)
    adjusted_parent = _ring_lambda_parent_name(mol, adjusted_parent, seq)
    adjusted_parent = _opsin_parseable_retained_parent_name(adjusted_parent)
    pos = {idx: locant for locant, idx in enumerate(seq, start=1)}
    suffix_locant = str(pos.get(acyl_idx, ""))
    if not suffix_locant:
        return ""
    prefixes = [
        (_single_ring_substituent_locant(adjusted_parent, pos[idx]), prefix)
        for idx, prefix in raw_subs
    ]
    prefix_text = _format_prefixes(prefixes)
    if prefixes and not prefix_text:
        return ""
    core = _ring_carbohydrazide_core(adjusted_parent, suffix_locant)
    if not core:
        return ""
    core = _join_n_prefix_functional_parent(n_prime_prefix, core)
    return _join_prefix_parent(prefix_text, core) if prefix_text else core


def _ring_carboxamide_core(parent_name: str, suffix_locant: str) -> str:
    if parent_name == "benzene" and suffix_locant == "1":
        return "benzamide"
    return f"{parent_name}-{suffix_locant}-carboxamide"


def _ring_carbohydrazide_core(parent_name: str, suffix_locant: str) -> str:
    if parent_name == "benzene" and suffix_locant == "1":
        return "benzohydrazide"
    return f"{parent_name}-{suffix_locant}-carbohydrazide"


def _ring_lambda_parent_name(mol, parent_name: str, seq: Sequence[int]) -> str:
    if "lambda" in parent_name:
        return parent_name
    lambda_parts = []
    ring_set = set(seq)
    for locant, atom_idx in enumerate(seq, start=1):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != "S" or atom.GetFormalCharge() != 0:
            continue
        valence = int(atom.GetTotalValence())
        if valence <= 2 or valence not in {4, 6}:
            continue
        if any(
            nbr.GetIdx() not in ring_set
            and nbr.GetSymbol() == "O"
            and mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx()).GetBondTypeAsDouble() == 2.0
            for nbr in atom.GetNeighbors()
        ):
            continue
        lambda_parts.append(f"{locant}lambda{valence}")
    if not lambda_parts:
        return parent_name
    return ",".join(lambda_parts) + f"-{parent_name}"


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


def _collect_component_atoms_excluding(mol, start_idx: int, blocked: Collection[int]) -> set[int]:
    seen = set(blocked)
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
        if suffix == "acid":
            return root + "anoic acid"
    parent = _unsaturated_parent(root, ene_locs, yne_locs)
    if suffix == "amide":
        return parent[:-1] + "amide" if parent.endswith("e") else parent + "amide"
    if suffix == "acid":
        return parent[:-1] + "oic acid" if parent.endswith("e") else parent + "oic acid"
    if suffix == "hydrazide":
        return parent + "hydrazide"
    return parent[:-1] + "oate" if parent.endswith("e") else parent + "oate"


def _oxo_carboxylate_parent_names(mol, Chem) -> list[str]:
    return [name for name, _stereo_locants in _oxo_carboxylate_parent_records(mol, Chem)]


def _terminal_carboxylate_parent_names(mol, Chem) -> list[str]:
    groups = _terminal_carboxyl_groups(mol, Chem, allowed_kinds={"carboxylate"})
    names: list[str] = []
    if len(groups) == 1:
        names.extend(_terminal_monocarboxylate_parent_names(mol, Chem, groups[0]))
    elif len(groups) == 2:
        names.extend(_terminal_dicarboxylate_parent_names(mol, Chem, groups))
    return sorted(set(names), key=lambda name: (len(name), name))


def _terminal_monocarboxylate_parent_names(mol, Chem, group: tuple[int, int, str]) -> list[str]:
    carbonyl_idx, alpha_idx, kind = group
    if kind != "carboxylate":
        return []
    names = []
    for path in _ester_acyl_parent_paths(mol, Chem, carbonyl_idx, alpha_idx):
        if not path or path[0] != alpha_idx:
            continue
        covered = _terminal_monocarboxylate_path_covered_atoms(mol, Chem, carbonyl_idx, path)
        if not covered or covered != _heavy_atom_indices(mol):
            continue
        name = _generic_acyl_functional_parent_path_name(mol, Chem, carbonyl_idx, path, "oate")
        if name:
            names.append(name)
    return names


def _terminal_monocarboxylate_path_covered_atoms(
    mol,
    Chem,
    carbonyl_idx: int,
    path: Sequence[int],
) -> set[int]:
    carbonyl_oxygen = _carbonyl_double_oxygen(mol, Chem, carbonyl_idx)
    if carbonyl_oxygen is None:
        return set()
    oxido_oxygen = _terminal_carboxylate_single_oxygen(mol, Chem, carbonyl_idx)
    if oxido_oxygen is None:
        return set()
    covered = {carbonyl_idx, carbonyl_oxygen, oxido_oxygen, *path}
    path_set = set(path)
    for pos, atom_idx in enumerate(path, start=2):
        allowed_neighbors = {carbonyl_idx if pos == 2 else path[pos - 3]}
        if pos - 1 < len(path):
            allowed_neighbors.add(path[pos - 1])
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in allowed_neighbors or nidx in path_set or nbr.GetAtomicNum() <= 1:
                continue
            branch_atoms = _collect_component_atoms_excluding(mol, nidx, {atom_idx, *path_set})
            if any(mol.GetAtomWithIdx(branch_idx).IsInRing() for branch_idx in branch_atoms):
                return set()
            if any(mol.GetAtomWithIdx(branch_idx).GetFormalCharge() != 0 for branch_idx in branch_atoms):
                return set()
            covered.update(branch_atoms)
    return covered


def _oxo_carboxylate_parent_records(mol, Chem) -> list[tuple[str, tuple[tuple[int, str], ...]]]:
    groups = _terminal_carboxyl_groups(mol, Chem, allowed_kinds={"carboxylate"})
    records: list[tuple[str, tuple[tuple[int, str], ...]]] = []
    if len(groups) == 1:
        records.extend((name, ()) for name in _alpha_keto_monocarboxylate_names(mol, Chem, groups[0]))
    elif len(groups) == 2:
        record = _alpha_keto_dicarboxylate_record(mol, Chem, groups)
        if record:
            records.append(record)
    return sorted(set(records), key=lambda record: (len(record[0]), record[0], record[1]))


def _terminal_carboxyl_groups(
    mol,
    Chem,
    *,
    allowed_kinds: set[str] | None = None,
) -> list[tuple[int, int, str]]:
    groups: list[tuple[int, int, str]] = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            continue
        carbonyl_idx = atom.GetIdx()
        carbon_neighbors = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetSymbol() == "C"
            and mol.GetBondBetweenAtoms(carbonyl_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
        ]
        if len(carbon_neighbors) != 1:
            continue
        kind = _terminal_carboxyl_group_kind(mol, Chem, carbonyl_idx, carbon_neighbors[0])
        if not kind or (allowed_kinds is not None and kind not in allowed_kinds):
            continue
        groups.append((carbonyl_idx, carbon_neighbors[0], kind))
    return groups


def _alpha_keto_monocarboxylate_names(mol, Chem, group: tuple[int, int, str]) -> list[str]:
    carbonyl_idx, alpha_idx, kind = group
    if kind != "carboxylate" or not _is_alpha_keto_carbon(mol, Chem, alpha_idx, carbonyl_idx):
        return []
    names = []
    for path in _ester_acyl_parent_paths(mol, Chem, carbonyl_idx, alpha_idx):
        if not path or path[0] != alpha_idx:
            continue
        if not _is_alpha_keto_carbon(mol, Chem, path[0], carbonyl_idx):
            continue
        name = _generic_acyl_functional_parent_path_name(mol, Chem, carbonyl_idx, path, "oate")
        if name:
            names.append(name)
    return names


def _is_alpha_keto_carbon(mol, Chem, atom_idx: int, carboxyl_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
        return False
    carboxyl_bond = mol.GetBondBetweenAtoms(atom_idx, carboxyl_idx)
    if carboxyl_bond is None or carboxyl_bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    double_oxygens = []
    carbon_neighbors = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carboxyl_idx:
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_neighbors.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return False
    return len(double_oxygens) == 1 and len(carbon_neighbors) <= 1


def _alpha_keto_dicarboxylate_name(mol, Chem, groups: Sequence[tuple[int, int, str]]) -> str:
    record = _alpha_keto_dicarboxylate_record(mol, Chem, groups)
    return record[0] if record else ""


def _alpha_keto_dicarboxylate_record(
    mol,
    Chem,
    groups: Sequence[tuple[int, int, str]],
) -> tuple[str, tuple[tuple[int, str], ...]] | None:
    endpoints = [group[0] for group in groups if group[2] == "carboxylate"]
    if len(endpoints) != 2:
        return None
    path = tuple(Chem.rdmolops.GetShortestPath(mol, endpoints[0], endpoints[1]))
    if not _simple_alpha_keto_dicarboxylate_path(mol, Chem, path):
        return None
    reverse = tuple(reversed(path))
    orientations = []
    for candidate_path in (path, reverse):
        oxo_locants = _path_oxo_locants(mol, Chem, candidate_path)
        if len(oxo_locants) == 1:
            hydroxy_locants = tuple(_path_hydroxy_locants(mol, Chem, candidate_path))
            if len(hydroxy_locants) <= 2:
                orientations.append((oxo_locants[0], hydroxy_locants, candidate_path))
    if not orientations:
        return None
    oxo_locant, hydroxy_locants, oriented = min(
        orientations,
        key=lambda item: (item[0], item[1], tuple(item[2])),
    )
    root = ROOTS.get(len(oriented))
    if not root:
        return None
    prefixes = [(str(locant), "hydroxy") for locant in hydroxy_locants]
    prefixes.append((str(oxo_locant), "oxo"))
    name = _join_prefix_parent(_format_prefixes(prefixes), f"{root}anedioate")
    stereo_locants = tuple((atom_idx, str(locant)) for locant, atom_idx in enumerate(oriented, start=1))
    return name, stereo_locants


def _terminal_dicarboxylate_parent_names(
    mol,
    Chem,
    groups: Sequence[tuple[int, int, str]],
) -> list[str]:
    if _acyclic_carbohydrate_retained_name(mol, Chem):
        return []
    endpoints = [group[0] for group in groups if group[2] == "carboxylate"]
    if len(endpoints) != 2:
        return []
    path = tuple(Chem.rdmolops.GetShortestPath(mol, endpoints[0], endpoints[1]))
    if not path:
        return []
    names = []
    for oriented in (path, tuple(reversed(path))):
        name = _terminal_dicarboxylate_path_name(mol, Chem, oriented)
        if name:
            names.append(name)
    return names


def _terminal_dicarboxylate_path_name(mol, Chem, path: Sequence[int]) -> str:
    if len(path) < 2:
        return ""
    if _terminal_carboxyl_group_kind(mol, Chem, path[0], path[1]) != "carboxylate":
        return ""
    if _terminal_carboxyl_group_kind(mol, Chem, path[-1], path[-2]) != "carboxylate":
        return ""
    root = ROOTS.get(len(path))
    if not root:
        return ""
    path_set = set(path)
    prefixes: list[tuple[str, str]] = []
    ene_locs: list[int] = []
    yne_locs: list[int] = []
    covered = set(path)
    for locant, atom_idx in enumerate(path, start=1):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            return ""
        expected_path_neighbors = set()
        if locant > 1:
            expected_path_neighbors.add(path[locant - 2])
        if locant < len(path):
            expected_path_neighbors.add(path[locant])
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nbr.GetAtomicNum() <= 1:
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
            if bond is None:
                return ""
            if nidx in path_set:
                if nidx not in expected_path_neighbors:
                    return ""
                if bond.GetBondType() not in {Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}:
                    return ""
                continue
            if locant in {1, len(path)} and nbr.GetSymbol() == "O":
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    covered.add(nidx)
                    continue
                if bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == -1:
                    covered.add(nidx)
                    continue
                return ""
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                prefixes.append((str(locant), "oxo"))
                covered.add(nidx)
                continue
            if (
                nbr.GetSymbol() == "O"
                and bond.GetBondType() == Chem.BondType.SINGLE
                and nbr.GetFormalCharge() == 0
                and nbr.GetTotalNumHs() == 1
                and not any(other.GetIdx() != atom_idx and other.GetAtomicNum() > 1 for other in nbr.GetNeighbors())
            ):
                prefixes.append((str(locant), "hydroxy"))
                covered.add(nidx)
                continue
            prefix = _simple_or_branch_prefix(mol, nidx, atom_idx)
            if not prefix:
                return ""
            prefixes.append((str(locant), prefix))
            covered.update(_collect_component_atoms(mol, nidx, atom_idx))
    if covered != _heavy_atom_indices(mol):
        return ""
    for locant, (left_idx, right_idx) in enumerate(zip(path, path[1:]), start=1):
        bond = mol.GetBondBetweenAtoms(left_idx, right_idx)
        if bond is None:
            return ""
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            ene_locs.append(locant)
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            yne_locs.append(locant)
        elif bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
    parent = _unsaturated_parent(root, ene_locs, yne_locs) + "edioate"
    prefix_text = _format_prefixes(prefixes)
    name = _join_prefix_parent(prefix_text, parent) if prefix_text else parent
    stereo = _generic_path_stereo_prefix(mol, Chem, path, start=1)
    if stereo is None:
        return ""
    return f"({stereo})-{name}" if stereo else name


def _simple_alpha_keto_dicarboxylate_path(mol, Chem, path: Sequence[int]) -> bool:
    if len(path) < 3:
        return False
    if _terminal_carboxyl_group_kind(mol, Chem, path[0], path[1]) != "carboxylate":
        return False
    if _terminal_carboxyl_group_kind(mol, Chem, path[-1], path[-2]) != "carboxylate":
        return False
    path_set = set(path)
    allowed_external_oxygens = set()
    for pos, atom_idx in enumerate(path):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            return False
        expected_path_neighbors = set()
        if pos > 0:
            expected_path_neighbors.add(path[pos - 1])
        if pos + 1 < len(path):
            expected_path_neighbors.add(path[pos + 1])
        double_oxygens = []
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nbr.GetAtomicNum() <= 1:
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
            if nidx in path_set:
                if nidx not in expected_path_neighbors or bond.GetBondType() != Chem.BondType.SINGLE:
                    return False
                continue
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygens.append(nidx)
                allowed_external_oxygens.add(nidx)
                continue
            if pos in {0, len(path) - 1} and nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
                allowed_external_oxygens.add(nidx)
                continue
            if (
                pos not in {0, len(path) - 1}
                and nbr.GetSymbol() == "O"
                and bond.GetBondType() == Chem.BondType.SINGLE
                and nbr.GetFormalCharge() == 0
                and nbr.GetTotalNumHs() == 1
                and not any(
                    other.GetIdx() != atom_idx and other.GetAtomicNum() > 1
                    for other in nbr.GetNeighbors()
                )
            ):
                allowed_external_oxygens.add(nidx)
                continue
            return False
        if pos not in {0, len(path) - 1} and len(double_oxygens) > 1:
            return False
    covered = path_set | allowed_external_oxygens
    heavy = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}
    return heavy == covered and len(_path_oxo_locants(mol, Chem, path)) == 1


def _path_oxo_locants(mol, Chem, path: Sequence[int]) -> list[int]:
    locants = []
    for locant, atom_idx in enumerate(path, start=1):
        if locant in {1, len(path)}:
            continue
        double_oxygen_count = 0
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx())
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygen_count += 1
        if double_oxygen_count == 1:
            locants.append(locant)
    return locants


def _path_hydroxy_locants(mol, Chem, path: Sequence[int]) -> list[int]:
    locants = []
    for locant, atom_idx in enumerate(path, start=1):
        if locant in {1, len(path)}:
            continue
        hydroxy_count = 0
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx())
            if (
                nbr.GetSymbol() == "O"
                and bond.GetBondType() == Chem.BondType.SINGLE
                and nbr.GetFormalCharge() == 0
                and nbr.GetTotalNumHs() == 1
            ):
                hydroxy_count += 1
        if hydroxy_count == 1:
            locants.append(locant)
    return locants


def _retained_simple_acyl_parent_text(root: str, suffix: str) -> str:
    if suffix == "acid":
        return {
            "meth": "formic acid",
            "eth": "acetic acid",
        }.get(root, "")
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
    return _generic_path_stereo_prefix(mol, Chem, path, start=2)


def _generic_path_stereo_prefix(mol, Chem, path: Sequence[int], *, start: int) -> str | None:
    locants = {idx: pos for pos, idx in enumerate(path, start=start)}
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


def _acyl_owner_dicarboxylate_ester_name(
    mol,
    Chem,
    carbonyl_idx: int,
    ester_oxygen_idx: int,
    acyl_idx: int,
) -> str:
    owner_record = _acyl_ester_owner_record(mol, Chem, ester_oxygen_idx, carbonyl_idx)
    if not owner_record:
        return ""
    owner_name, owner_atoms = owner_record
    name = _dicarboxylate_ester_name(mol, Chem, carbonyl_idx, acyl_idx, owner_name)
    if not name:
        return ""
    dicarboxylate_atoms = _collect_component_atoms(mol, carbonyl_idx, ester_oxygen_idx)
    covered = set(owner_atoms)
    covered.add(ester_oxygen_idx)
    covered.update(dicarboxylate_atoms)
    return name if covered == _heavy_atom_indices(mol) else ""


def _acyl_ester_owner_record(
    mol,
    Chem,
    ester_oxygen_idx: int,
    dicarboxylate_carbonyl_idx: int,
) -> tuple[str, frozenset[int]] | None:
    oxygen = mol.GetAtomWithIdx(ester_oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0:
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != dicarboxylate_carbonyl_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    owner_carbonyl_idx = heavy[0]
    owner_carbonyl = mol.GetAtomWithIdx(owner_carbonyl_idx)
    if owner_carbonyl.GetSymbol() != "C" or owner_carbonyl.GetFormalCharge() != 0 or owner_carbonyl.IsInRing():
        return None
    double_oxygen = _carbonyl_double_oxygen(mol, Chem, owner_carbonyl_idx)
    if double_oxygen is None:
        return None
    acyl_carbons = [
        nbr.GetIdx()
        for nbr in owner_carbonyl.GetNeighbors()
        if nbr.GetIdx() != ester_oxygen_idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(owner_carbonyl_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    if len(acyl_carbons) != 1:
        return None
    owner_acyl_idx = acyl_carbons[0]
    names = [
        name
        for path in _ester_acyl_parent_paths(mol, Chem, owner_carbonyl_idx, owner_acyl_idx)
        for name in (_generic_acyl_ligand_path_name(mol, Chem, owner_carbonyl_idx, path),)
        if name and _stable_acyl_dicarboxylate_owner_name(name)
    ]
    if not names:
        return None
    owner_name = min(names, key=lambda value: (len(value), value))
    owner_atoms = {owner_carbonyl_idx, double_oxygen}
    owner_atoms.update(_collect_component_atoms(mol, owner_acyl_idx, owner_carbonyl_idx))
    return owner_name, frozenset(owner_atoms)


def _stable_acyl_dicarboxylate_owner_name(name: str) -> bool:
    if len(name) > 120 or " " in name or "[" in name or "]" in name:
        return False
    if name.startswith("(") and not re.match(r"^\([0-9EZRS,]+\)-", name):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9,()'\-]+", name))


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


def _terminal_carboxylate_single_oxygen(mol, Chem, carbonyl_idx: int) -> int | None:
    oxygen_indices = []
    for nbr in mol.GetAtomWithIdx(carbonyl_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if (
            nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() == -1
            and bond is not None
            and bond.GetBondType() == Chem.BondType.SINGLE
            and _heavy_degree_excluding(mol, nidx, carbonyl_idx) == 0
        ):
            oxygen_indices.append(nidx)
    return oxygen_indices[0] if len(oxygen_indices) == 1 else None


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


def _sydnone_urea_amidate_names(mol, Chem) -> list[str]:
    names = []
    for carbonyl_idx, nitrogen_a, nitrogen_b in _urea_motifs(mol, Chem):
        for anionic_n, neutral_n in ((nitrogen_a, nitrogen_b), (nitrogen_b, nitrogen_a)):
            anion_atom = mol.GetAtomWithIdx(anionic_n)
            neutral_atom = mol.GetAtomWithIdx(neutral_n)
            if anion_atom.GetFormalCharge() != -1 or neutral_atom.GetFormalCharge() != 0:
                continue
            anion_substituent_idx = _single_heavy_neighbor_excluding(mol, anionic_n, carbonyl_idx)
            neutral_substituent_idx = _single_heavy_neighbor_excluding(mol, neutral_n, carbonyl_idx)
            if anion_substituent_idx is None or neutral_substituent_idx is None:
                continue
            if not _sydnone_like_urea_amidate_substituent(mol, anion_substituent_idx):
                continue
            anion_substituent = _simple_or_branch_prefix(mol, anion_substituent_idx, anionic_n)
            neutral_substituent = _simple_or_branch_prefix(mol, neutral_substituent_idx, neutral_n)
            if not _allowed_sydnone_urea_amidate_substituent(anion_substituent, charged=True):
                continue
            if not _allowed_sydnone_urea_amidate_substituent(neutral_substituent, charged=False):
                continue
            names.append(f"({anion_substituent})(({neutral_substituent})carbamoyl)amide")
    return names


def _single_heavy_neighbor_excluding(mol, atom_idx: int, excluded_idx: int) -> int | None:
    neighbors = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if nbr.GetIdx() != excluded_idx and nbr.GetAtomicNum() > 1
    ]
    return neighbors[0] if len(neighbors) == 1 else None


def _sydnone_like_urea_amidate_substituent(mol, attach_idx: int) -> bool:
    attach_atom = mol.GetAtomWithIdx(attach_idx)
    if attach_atom.GetSymbol() != "C" or attach_atom.GetFormalCharge() != 0:
        return False
    for ring in mol.GetRingInfo().AtomRings():
        if attach_idx not in ring or len(ring) != 5:
            continue
        counts = Counter(mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring)
        if counts != {"C": 2, "N": 2, "O": 1}:
            continue
        charged_nitrogens = [
            idx
            for idx in ring
            if mol.GetAtomWithIdx(idx).GetSymbol() == "N" and mol.GetAtomWithIdx(idx).GetFormalCharge() == 1
        ]
        if len(charged_nitrogens) != 1:
            continue
        if any(
            mol.GetAtomWithIdx(idx).GetFormalCharge() not in {0, 1}
            for idx in ring
        ):
            continue
        return True
    return False


def _allowed_sydnone_urea_amidate_substituent(name: str, *, charged: bool) -> bool:
    if not name:
        return False
    if len(name) > (120 if charged else 72):
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", name):
        return False
    if _unstable_retained_functional_parent_substituent(name):
        return False
    if charged:
        return True
    return _allowed_n_retained_parent_substituent(name) or any(
        fragment in name
        for fragment in (
            "trifluoromethyl",
            "fluorophenyl",
            "chlorophenyl",
            "bromophenyl",
            "iodophenyl",
            "tolyl",
            "anisyl",
        )
    )


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


def _sulfonamide_imino_n_substituent(mol, Chem, nitrogen_idx: int, sulfur_idx: int) -> str:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    heavy = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors() if nbr.GetAtomicNum() > 1]
    if sulfur_idx not in heavy or len(heavy) != 2:
        return ""
    imine_carbons = [
        nbr.GetIdx()
        for nbr in nitrogen.GetNeighbors()
        if nbr.GetIdx() != sulfur_idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(nitrogen_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]
    if len(imine_carbons) != 1:
        return ""
    return _imino_carbon_retained_substituent(mol, imine_carbons[0], nitrogen_idx)


def _imino_carbon_retained_substituent(mol, carbon_idx: int, imine_nitrogen_idx: int) -> str:
    substituents = []
    for nbr in mol.GetAtomWithIdx(carbon_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == imine_nitrogen_idx or nbr.GetAtomicNum() <= 1:
            continue
        if nbr.GetSymbol() == "N" and _terminal_amino_substituent(mol, nidx, carbon_idx):
            substituents.append("amino")
            continue
        name = _simple_or_branch_prefix(mol, nidx, carbon_idx)
        if not name:
            return ""
        substituents.append(_retained_parent_substituent_alias(name))
    if len(substituents) != 2:
        return ""
    if "amino" in substituents:
        other = next(name for name in substituents if name != "amino")
        return f"amino({_strip_outer_parentheses(other)})methylene"
    if "methyl" in substituents:
        other = next(name for name in substituents if name != "methyl")
        return f"1-({_strip_outer_parentheses(other)})ethylidene"
    return ""


def _terminal_amino_substituent(mol, nitrogen_idx: int, carbon_idx: int) -> bool:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetFormalCharge() != 0:
        return False
    heavy = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors() if nbr.GetAtomicNum() > 1]
    return heavy == [carbon_idx]


def _sulfonamide_defer_to_sulfonohydrazide(mol, Chem, nitrogen_idx: int, sulfur_idx: int) -> bool:
    distal = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors()
        if nbr.GetIdx() != sulfur_idx
        and nbr.GetSymbol() == "N"
        and mol.GetBondBetweenAtoms(nitrogen_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    if len(distal) != 1:
        return False
    distal_n = mol.GetAtomWithIdx(distal[0])
    return any(
        nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(distal[0], nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        for nbr in distal_n.GetNeighbors()
        if nbr.GetIdx() != nitrogen_idx
    )


def _sulfonimidamide_motifs(mol, Chem) -> list[dict]:
    motifs = []
    for sulfur in mol.GetAtoms():
        if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0 or sulfur.IsInRing():
            continue
        sidx = sulfur.GetIdx()
        double_o = []
        double_n = []
        single_n = []
        single_substituent = []
        ok = True
        for nbr in sulfur.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(sidx, nidx)
            if bond is None:
                ok = False
                break
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_o.append(nidx)
            elif nbr.GetSymbol() == "N" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_n.append(nidx)
            elif nbr.GetSymbol() == "N" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.SINGLE:
                single_n.append(nidx)
            elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetAtomicNum() > 1:
                single_substituent.append(nidx)
            elif nbr.GetAtomicNum() > 1:
                ok = False
                break
        if not ok or len(double_o) != 1 or len(double_n) != 1 or len(single_n) != 1:
            continue
        if len(single_substituent) > 1:
            continue
        motifs.append(
            {
                "sulfur": sidx,
                "s_substituent": single_substituent[0] if single_substituent else None,
                "amide_nitrogen": single_n[0],
                "imino_nitrogen": double_n[0],
            }
        )
    return motifs


def _lambda_sulfanone_parent_records(mol, Chem) -> list[tuple[int, tuple[tuple[int, str], ...]]]:
    if not _lambda_sulfanone_formal_charges_supported(mol):
        return []
    records: list[tuple[int, tuple[tuple[int, str], ...]]] = []
    for sulfur in mol.GetAtoms():
        if (
            sulfur.GetSymbol() != "S"
            or sulfur.GetFormalCharge() != 0
            or sulfur.IsInRing()
            or int(sulfur.GetTotalValence()) != 6
            or sulfur.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        ):
            continue
        sulfur_idx = sulfur.GetIdx()
        double_o: list[int] = []
        double_n: list[int] = []
        single_c: list[int] = []
        ok = True
        for neighbor in sulfur.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            bond = mol.GetBondBetweenAtoms(sulfur_idx, neighbor_idx)
            if bond is None:
                ok = False
                break
            bond_type = bond.GetBondType()
            if (
                neighbor.GetSymbol() == "O"
                and neighbor.GetFormalCharge() == 0
                and bond_type == Chem.BondType.DOUBLE
            ):
                double_o.append(neighbor_idx)
            elif (
                neighbor.GetSymbol() == "N"
                and neighbor.GetFormalCharge() == 0
                and bond_type == Chem.BondType.DOUBLE
            ):
                double_n.append(neighbor_idx)
            elif neighbor.GetSymbol() == "C" and bond_type == Chem.BondType.SINGLE:
                single_c.append(neighbor_idx)
            elif neighbor.GetAtomicNum() > 1:
                ok = False
                break
        if not ok or len(double_o) != 1 or len(double_n) > 1 or len(single_c) > 2:
            continue
        ligands: list[tuple[int, str]] = []
        for atom_idx in sorted(single_c):
            ligand = _lambda_sulfanone_carbon_ligand(mol, atom_idx, sulfur_idx)
            if not ligand:
                ok = False
                break
            ligands.append((atom_idx, ligand))
        if not ok:
            continue
        for atom_idx in sorted(double_n):
            ligand = _lambda_sulfanone_imino_ligand(mol, atom_idx, sulfur_idx)
            if not ligand:
                ok = False
                break
            ligands.append((atom_idx, ligand))
        if not ok or len(ligands) < 2 or len(ligands) > 3:
            continue
        records.append((sulfur_idx, tuple(ligands)))
    return records


def _lambda_sulfanone_formal_charges_supported(mol) -> bool:
    charged_atoms = [atom for atom in mol.GetAtoms() if atom.GetFormalCharge()]
    if not charged_atoms:
        return True
    nitro_atoms: set[int] = set()
    for atom in charged_atoms:
        if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 1:
            continue
        oxygen_neighbors = [
            neighbor
            for neighbor in atom.GetNeighbors()
            if neighbor.GetSymbol() == "O" and neighbor.GetFormalCharge() == -1
        ]
        if not oxygen_neighbors:
            continue
        nitro_atoms.add(atom.GetIdx())
        nitro_atoms.update(neighbor.GetIdx() for neighbor in oxygen_neighbors)
    return all(atom.GetIdx() in nitro_atoms for atom in charged_atoms)


def _lambda_sulfanone_carbon_ligand(mol, atom_idx: int, sulfur_idx: int) -> str:
    ligand = _retained_parent_substituent_alias(_simple_or_branch_prefix(mol, atom_idx, sulfur_idx))
    if not _safe_lambda_sulfanone_ligand(ligand, max_len=160):
        return ""
    return ligand


def _lambda_sulfanone_imino_ligand(mol, nitrogen_idx: int, sulfur_idx: int) -> str:
    substituents = _external_substituent_names(mol, nitrogen_idx, sulfur_idx)
    if substituents is None or len(substituents) > 1:
        return ""
    if not substituents:
        return "imino"
    substituent = _retained_parent_substituent_alias(substituents[0])
    if not _safe_lambda_sulfanone_ligand(substituent, max_len=90):
        return ""
    if any(fragment in substituent for fragment in ("acetyl", "boc", "carbonyl", "carbam", "urea")):
        return ""
    return substituent + "imino"


def _safe_lambda_sulfanone_ligand(name: str, *, max_len: int) -> bool:
    if not name or len(name) > max_len:
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,().'\-]+", name):
        return False
    if any(fragment in name for fragment in ("lambda", "sulfanone", "sulfonimidamide")):
        return False
    return True


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
    if any(_unstable_urea_retained_parent_substituent(name) for name in (*left, *right)):
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


def _sulfonimidamide_n_substituent_prefix(left: Sequence[str], right: Sequence[str]) -> str | None:
    left = tuple(_retained_parent_substituent_alias(name) for name in left)
    right = tuple(_retained_parent_substituent_alias(name) for name in right)
    if len(left) > 2 or len(right) > 1:
        return None
    if not left and not right:
        return ""
    if not all(_allowed_sulfonimidamide_substituent(name) for name in (*left, *right)):
        return None
    if len(left) == 2 and not right:
        return _same_atom_urea_prefix("N", left)
    if len(left) == 1 and not right:
        return _n_substituent_piece("N", left[0])
    if len(right) == 1 and not left:
        return _n_substituent_piece("N'", right[0])
    if len(left) == 1 and len(right) == 1:
        if left[0] == right[0]:
            return _duplicate_n_substituent_piece("N,N'", left[0])
        first_piece = _n_substituent_piece("N", left[0])
        second_piece = _n_substituent_piece("N'", right[0])
        return f"{first_piece}-{second_piece}"
    return None


def _sulfonamide_n_substituent_prefix(substituents: Sequence[str]) -> str | None:
    substituents = tuple(_retained_parent_substituent_alias(name) for name in substituents)
    if len(substituents) > 2:
        return None
    if not substituents:
        return ""
    if not all(_allowed_sulfonamide_substituent(name) for name in substituents):
        return None
    if len(substituents) == 1:
        return _n_substituent_piece("N", substituents[0])
    counts = Counter(substituents)
    if len(counts) == 1:
        name = next(iter(counts))
        return _duplicate_n_substituent_piece("N,N", name)
    first, second = sorted(substituents)
    return f"{_n_substituent_piece('N', first)}-{_n_substituent_piece('N', second)}"


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
    name = re.sub(r"^[123]H-(benzotriazol-\d+[a-z]?'-?yl)$", r"\1", name)
    name = re.sub(r"^[123]H-(benzotriazol-\d+[a-z]?-yl)$", r"\1", name)
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
    if _stable_fused_heteroaryl_n_substituent(name):
        return True
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


def _unstable_urea_retained_parent_substituent(name: str) -> bool:
    return any(
        fragment in name
        for fragment in (
            "amino",
            "carboxy",
            "carbonyl",
            "ium",
            "oxo",
        )
    )


def _stable_fused_heteroaryl_n_substituent(name: str) -> bool:
    if not (name.endswith("yl") or name.endswith("'yl")):
        return False
    if any(fragment in name for fragment in ("oxo", "formyl", "hydroxy", "carbonyl")):
        return False
    fused_stems = (
        "benzimidazol",
        "benzothiazol",
        "benzotriazol",
        "benzoxazol",
        "benzofuran",
        "benzothiophen",
        "indazol",
        "indol",
        "isoquinolin",
        "quinazolin",
        "quinolin",
        "quinoxalin",
    )
    return any(stem in name for stem in fused_stems)


def _allowed_sulfonimidamide_substituent(name: str) -> bool:
    if _allowed_n_retained_parent_substituent(name):
        return True
    if name in {
        "pentyl",
        "amyl",
        "hexyl",
        "heptyl",
        "octyl",
        "nonyl",
        "decyl",
        "dodecyl",
    }:
        return True
    if "carbamoyl" not in name and "amino)carbonyl" not in name:
        return False
    if len(name) > 180:
        return False
    if " " in name:
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "urea")):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9,().'\-\[\]\^]+", name))


def _allowed_sulfonamide_substituent(name: str) -> bool:
    if _allowed_n_retained_parent_substituent(name):
        return True
    if name in {
        "boc",
        "propargyl",
        "octyl",
        "dodecyl",
        "hexadecyl",
        "trifluoromethyl",
        "2,2,2-trifluoroethyl",
    }:
        return True
    if len(name) > 190:
        return False
    if " " in name:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,().'\-\[\]\^]+", name):
        return False
    if any(fragment in name for fragment in ("phosph", "sulfonylaminoamino", "sulfanyl")):
        return False
    return any(
        fragment in name
        for fragment in (
            "amino",
            "benz",
            "carbonyl",
            "carboxy",
            "cyclo",
            "ethen",
            "ethyl",
            "fluoro",
            "formyl",
            "hydroxy",
            "imidazol",
            "indol",
            "methyl",
            "morpholin",
            "oxo",
            "phenyl",
            "piper",
            "pyraz",
            "pyrid",
            "pyrimid",
            "quin",
            "thiazol",
            "thien",
            "triazin",
            "triazol",
        )
    )


def _n_substituent_piece(locant: str, name: str) -> str:
    if _simple_unparenthesized_n_substituent(name):
        return f"{locant}-{name}"
    return f"{locant}-({_strip_outer_parentheses(name)})"


def _duplicate_n_substituent_piece(locants: str, name: str) -> str:
    if _simple_unparenthesized_n_substituent(name) and not _repeated_n_substituent_needs_complex_multiplier(name):
        return f"{locants}-di{name}"
    return f"{locants}-bis({_strip_outer_parentheses(name)})"


def _repeated_n_substituent_needs_complex_multiplier(name: str) -> bool:
    if name in {"methyl", "ethyl", "propyl", "butyl", "phenyl", "benzyl"}:
        return False
    return name.endswith(("methyl", "ethyl", "propyl", "butyl"))


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
    generic = _generic_sulfonamide_base_name(substituent)
    if generic:
        return generic
    return ""


def _generic_sulfonamide_base_name(substituent: str) -> str:
    if not _allowed_sulfonamide_base_substituent(substituent):
        return ""
    return f"{substituent}sulfonamide"


def _allowed_sulfonamide_base_substituent(substituent: str) -> bool:
    if len(substituent) > 90:
        return False
    if " " in substituent:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,().'\-\[\]\^]+", substituent):
        return False
    if any(fragment in substituent for fragment in ("carbamoyl", "carbonyl", "dioxo", "oxo", "sulfonylaminoamino", "phosph")):
        return False
    if not (
        substituent.endswith("yl")
        or substituent.endswith("'yl")
        or "phenyl" in substituent
        or "naphthalen" in substituent
    ):
        return False
    return any(
        fragment in substituent
        for fragment in (
            "benz",
            "cyan",
            "fur",
            "imidazol",
            "indazol",
            "indol",
            "isooxazol",
            "methoxy",
            "naphthalen",
            "oxazol",
            "phenyl",
            "pyraz",
            "pyrid",
            "pyrimid",
            "quin",
            "thiaz",
            "thien",
            "triaz",
        )
    )


def _join_retained_sulfonamide_parent(prefix: str, base: str) -> str:
    if not prefix:
        return base
    return f"{prefix}-{base}" if _needs_explicit_n_parent_separator(prefix) or base[:1].isdigit() else prefix + base


def _sulfonimidamide_base_name(mol, sulfur_idx: int, substituent_idx: int | None) -> str:
    if substituent_idx is None:
        return "sulfonimidamide"
    substituent = _simple_or_branch_prefix(mol, substituent_idx, sulfur_idx)
    if not substituent:
        return ""
    simple_alkyl = {
        "methyl": "methanesulfonimidamide",
        "ethyl": "ethanesulfonimidamide",
        "propyl": "propane-1-sulfonimidamide",
        "butyl": "butane-1-sulfonimidamide",
        "pentyl": "pentane-1-sulfonimidamide",
        "amyl": "pentane-1-sulfonimidamide",
        "hexyl": "hexane-1-sulfonimidamide",
        "heptyl": "heptane-1-sulfonimidamide",
        "octyl": "octane-1-sulfonimidamide",
        "nonyl": "nonane-1-sulfonimidamide",
        "decyl": "decane-1-sulfonimidamide",
        "isopropyl": "propane-2-sulfonimidamide",
        "t-butyl": "2-methylpropane-2-sulfonimidamide",
        "tert-butyl": "2-methylpropane-2-sulfonimidamide",
    }.get(substituent)
    if simple_alkyl:
        return simple_alkyl
    if substituent == "phenyl":
        return "benzenesulfonimidamide"
    if substituent in {"o-tolyl", "m-tolyl", "p-tolyl"}:
        return substituent[0] + "-toluenesulfonimidamide"
    if substituent.endswith("phenyl"):
        return substituent[: -len("phenyl")] + "benzenesulfonimidamide"
    heteroaryl = _heteroaryl_sulfonimidamide_base_name(substituent)
    if heteroaryl:
        return heteroaryl
    generic = _generic_sulfonimidamide_base_name(substituent)
    if generic:
        return generic
    return ""


def _heteroaryl_sulfonimidamide_base_name(substituent: str) -> str:
    simple = re.fullmatch(r"(?:(?P<prefix>.+)-)?(?P<loc>\d+)-(?:thienyl|thiophenyl)", substituent)
    if simple:
        prefix = (simple.group("prefix") or "").replace("dichloro", "dichloro")
        stem = (prefix + "thiophene") if prefix else "thiophene"
        return f"{stem}-{simple.group('loc')}-sulfonimidamide"
    furyl = re.fullmatch(r"(?:(?P<prefix>.+)-)?(?P<loc>\d+)-furyl", substituent)
    if furyl:
        prefix = furyl.group("prefix") or ""
        stem = (prefix + "furan") if prefix else "furan"
        return f"{stem}-{furyl.group('loc')}-sulfonimidamide"
    match = re.fullmatch(
        r"(?P<stem>.+(?:pyridin|pyrimidin|pyridazin|pyrazin|pyrazol|imidazol|triazol|thiazol|oxazol))-(?P<loc>\d+)-yl",
        substituent,
    )
    if not match:
        return ""
    stem = match.group("stem")
    if stem.endswith(("pyridin", "pyrimidin", "pyridazin", "pyrazin", "pyrazol", "imidazol", "triazol", "thiazol", "oxazol")):
        stem += "e"
    return f"{stem}-{match.group('loc')}-sulfonimidamide"


def _generic_sulfonimidamide_base_name(substituent: str) -> str:
    if not _allowed_sulfonimidamide_base_substituent(substituent):
        return ""
    locanted = re.fullmatch(r"(?P<stem>.+)-(?P<loc>\d+[a-z]?)-yl", substituent)
    if locanted:
        return f"{locanted.group('stem')}-{locanted.group('loc')}-sulfonimidamide"
    return f"{substituent}sulfonimidamide"


def _allowed_sulfonimidamide_base_substituent(substituent: str) -> bool:
    if len(substituent) > 190:
        return False
    if " " in substituent:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,().'\-\[\]\^]+", substituent):
        return False
    if any(fragment in substituent for fragment in ("phosph", "sulfon", "sulfin", "sulfamoyl")):
        return False
    if not (
        substituent.endswith("yl")
        or substituent.endswith("'yl")
        or "phenyl" in substituent
        or "naphthalen" in substituent
    ):
        return False
    return any(
        fragment in substituent
        for fragment in (
            "benz",
            "bicyclo",
            "cyclo",
            "fur",
            "imidazol",
            "indazol",
            "indol",
            "isoquinolin",
            "isooxazol",
            "naphthalen",
            "oxazol",
            "oxazin",
            "phenyl",
            "pyraz",
            "pyrid",
            "pyrimid",
            "quin",
            "thiaz",
            "thien",
            "triaz",
        )
    )


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
        parent_atoms = _nonring_heavy_atom_indices(mol)
        if _should_defer_spiro_substituent_parent_to_senior_azolone(ctx, parent_atoms):
            return
        if _should_defer_acyclic_sulfur_imidoyl_holder_to_ring_parent(ctx, parent_atoms):
            return
        if _is_direct_ring_carboxylate_acyclic_parent(ctx.mol, ctx.Chem, parent_atoms):
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
                parent_atoms=parent_atoms,
                stereo_locants=_complete_acyclic_stereo_locants(ctx, name, name_record.stereo_locants),
                estimated_length=len(name),
            )


def _is_direct_ring_carboxylate_acyclic_parent(mol, Chem, parent_atoms: frozenset[int]) -> bool:
    if len(parent_atoms) != 3:
        return False
    carbonyl_atoms = [
        idx
        for idx in parent_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == "C"
    ]
    if len(carbonyl_atoms) != 1:
        return False
    carbonyl_idx = carbonyl_atoms[0]
    external_heavy = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(carbonyl_idx).GetNeighbors()
        if nbr.GetAtomicNum() > 1 and nbr.GetIdx() not in parent_atoms
    ]
    if len(external_heavy) != 1:
        return False
    external_idx = external_heavy[0]
    if not mol.GetAtomWithIdx(external_idx).IsInRing():
        return False
    double_oxygen_count = 0
    carboxylate_oxygen_count = 0
    for neighbor in mol.GetAtomWithIdx(carbonyl_idx).GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, neighbor_idx)
        if bond is None or neighbor.GetSymbol() != "O":
            continue
        if bond.GetBondType() == Chem.BondType.DOUBLE and neighbor.GetFormalCharge() == 0:
            double_oxygen_count += 1
            continue
        if bond.GetBondType() == Chem.BondType.SINGLE and neighbor.GetFormalCharge() == -1:
            external_oxygen_neighbors = [
                nbr.GetIdx()
                for nbr in neighbor.GetNeighbors()
                if nbr.GetIdx() != carbonyl_idx and nbr.GetAtomicNum() > 1
            ]
            if external_oxygen_neighbors:
                continue
            carboxylate_oxygen_count += 1
    if double_oxygen_count != 1 or carboxylate_oxygen_count != 1:
        return False
    return set(parent_atoms) == _collect_component_atoms(mol, carbonyl_idx, external_idx)


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
            or ctx.facts.nonring_heavy_atom_count < 8
        ):
            return
        if _should_defer_acyclic_skeleton_to_exocyclic_ring_parent(ctx):
            return
        parent_atoms = _nonring_heavy_atom_indices(mol)
        if _should_defer_spiro_substituent_parent_to_senior_azolone(ctx, parent_atoms):
            return
        if _should_defer_acyclic_sulfur_imidoyl_holder_to_ring_parent(ctx, parent_atoms):
            return
        name_record = ctx.acyclic_name_record(allow_ring_atoms=False)
        name = name_record.name
        if not name or not _prefer_acyclic_skeleton_name(name):
            return
        if name and _should_defer_substituent_like_acyclic_name_to_ring_parent(ctx, name):
            return
        yield NameCandidate(
            name,
            self.family,
            self.priority,
            sort_key=(len(name), name),
            parent_atoms=parent_atoms,
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
        if ctx.facts.nonring_heavy_atom_count < 2:
            return
        if not ctx.facts.may_have_chain_dominant_acyclic_parent:
            return
        parent_atoms = _nonring_heavy_atom_indices(mol)
        if _should_defer_spiro_substituent_parent_to_senior_azolone(ctx, parent_atoms):
            return
        if _should_defer_acyclic_sulfur_imidoyl_holder_to_ring_parent(ctx, parent_atoms):
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
                parent_atoms=parent_atoms,
                stereo_locants=_complete_acyclic_stereo_locants(ctx, name, name_record.stereo_locants),
                estimated_length=len(name),
            )


class SulfurOxoAcyclicSkeletonWithRingSubstituentsRule(AcyclicSkeletonWithRingSubstituentsRule):
    family = "sulfur-oxo-acyclic-skeleton-with-ring-substituents"
    priority = parent_priority("functional-acyclic-skeleton")

    def candidates(self, ctx):
        mol = ctx.mol
        profile = ctx.profile
        if profile.ring_count < 2:
            return
        parent_atoms = _nonring_heavy_atom_indices(mol)
        if not 4 <= len(parent_atoms) <= 18:
            return
        if not _has_nonring_sulfur_oxo_acyclic_parent_center(mol, parent_atoms):
            return
        if not _acyclic_parent_has_ring_attachment(mol, parent_atoms):
            return

        name_record = ctx.acyclic_name_record(allow_ring_atoms=False)
        name = name_record.name
        if not name:
            return
        if _should_defer_spiro_substituent_parent_to_senior_azolone(ctx, parent_atoms):
            return
        if _is_direct_ring_carboxylate_acyclic_parent(ctx.mol, ctx.Chem, parent_atoms):
            return
        yield NameCandidate(
            name,
            self.family,
            self.priority,
            sort_key=(len(name), name),
            parent_atoms=parent_atoms,
            source_family="sulfur-oxo-acyclic-skeleton",
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
        parent_atoms = _nonring_heavy_atom_indices(mol)
        if _should_defer_spiro_substituent_parent_to_senior_azolone(ctx, parent_atoms):
            return
        name_record = ctx.acyclic_name_record(allow_ring_atoms=False)
        name = name_record.name
        if (
            name
            and "spiro[" in name
            and _polycyclic_substituent_chain_signal(name)
            and not _fragile_senior_spiro_chain_name(name)
        ):
            source_family = ""
            if _parent_atoms_contain_noncarbon_group14(mol, parent_atoms):
                source_family = "spiro-functional-chain:noncarbon-group14"
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=parent_atoms,
                source_family=source_family,
                stereo_locants=_complete_acyclic_stereo_locants(ctx, name, name_record.stereo_locants),
                estimated_length=len(name),
            )


@dataclass(frozen=True, slots=True)
class _LambdaSulfanoneMotif:
    sulfur_idx: int
    ring_idx: int
    methyl_idx: int
    imino_idx: int
    oxo_idx: int


def _parent_atoms_contain_noncarbon_group14(mol, atom_indices: frozenset[int]) -> bool:
    return any(mol.GetAtomWithIdx(idx).GetAtomicNum() in {14, 32, 50, 82} for idx in atom_indices)


class LambdaSulfanoneSpiroSubstituentRule:
    family = "lambda-sulfanone-spiro-substituent"
    priority = parent_priority("senior-spiro-functional-chain-substituent")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() == 0 or not ctx.has_spiro_topology:
            return

        seen: set[str] = set()
        for motif in _lambda_sulfanone_spiro_motifs(mol):
            substituent = polycyclic_substituent_prefix(mol, self._Chem, motif.ring_idx, motif.sulfur_idx)
            if not substituent:
                continue
            name = f"[{substituent}](imino)methyl-lambda6-sulfanone"
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(
                    len(substituent),
                    motif.ring_idx,
                    motif.sulfur_idx,
                    name,
                ),
                parent_atoms=_heavy_atom_indices(mol),
                source_family="lambda-sulfanone",
                estimated_length=len(name),
            )


def _lambda_sulfanone_spiro_motifs(mol) -> tuple[_LambdaSulfanoneMotif, ...]:
    motifs: list[_LambdaSulfanoneMotif] = []
    for sulfur in mol.GetAtoms():
        if (
            sulfur.GetSymbol() != "S"
            or sulfur.IsInRing()
            or sulfur.GetFormalCharge() != 0
            or sulfur.GetTotalNumHs() != 2
        ):
            continue
        sulfur_idx = sulfur.GetIdx()
        oxo_neighbors: list[int] = []
        ring_neighbors: list[int] = []
        imino_methyl: list[tuple[int, int]] = []
        for neighbor in sulfur.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            bond = mol.GetBondBetweenAtoms(sulfur_idx, neighbor_idx)
            if bond is None:
                continue
            bond_order = bond.GetBondTypeAsDouble()
            if neighbor.GetSymbol() == "O" and bond_order == 2.0 and neighbor.GetDegree() == 1:
                oxo_neighbors.append(neighbor_idx)
                continue
            if neighbor.IsInRing() and bond_order == 1.0:
                ring_neighbors.append(neighbor_idx)
                continue
            if neighbor.GetSymbol() != "C" or neighbor.IsInRing() or bond_order != 1.0:
                continue
            imino_nitrogens = [
                next_neighbor.GetIdx()
                for next_neighbor in neighbor.GetNeighbors()
                if next_neighbor.GetIdx() != sulfur_idx
                and next_neighbor.GetSymbol() == "N"
                and next_neighbor.GetDegree() == 1
                and mol.GetBondBetweenAtoms(neighbor_idx, next_neighbor.GetIdx()).GetBondTypeAsDouble() == 2.0
            ]
            if len(imino_nitrogens) == 1 and neighbor.GetDegree() == 2 and neighbor.GetTotalNumHs() == 1:
                imino_methyl.append((neighbor_idx, imino_nitrogens[0]))
        if len(oxo_neighbors) != 1 or len(ring_neighbors) != 1 or len(imino_methyl) != 1:
            continue
        methyl_idx, imino_idx = imino_methyl[0]
        motifs.append(
            _LambdaSulfanoneMotif(
                sulfur_idx=sulfur_idx,
                ring_idx=ring_neighbors[0],
                methyl_idx=methyl_idx,
                imino_idx=imino_idx,
                oxo_idx=oxo_neighbors[0],
            )
        )
    return tuple(motifs)


class ChargedRingAcyclicSkeletonWithRingSubstituentsRule(AcyclicSkeletonWithRingSubstituentsRule):
    family = "charged-ring-acyclic-skeleton-with-ring-substituents"
    priority = parent_priority("polycyclic-functional-skeleton")
    requires_charged_ring_atom = True

    def candidates(self, ctx):
        profile = ctx.profile
        if profile.ring_count == 0 or not profile.has_charged_ring_atom:
            return
        parent_atoms = _nonring_heavy_atom_indices(ctx.mol)
        if _should_defer_acyclic_sulfur_imidoyl_holder_to_ring_parent(ctx, parent_atoms):
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
                parent_atoms=parent_atoms,
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
            or ctx.facts.nonring_heavy_atom_count < 8
        ):
            return
        if _should_defer_acyclic_skeleton_to_exocyclic_ring_parent(ctx):
            return
        parent_atoms = _nonring_heavy_atom_indices(mol)
        if _should_defer_acyclic_sulfur_imidoyl_holder_to_ring_parent(ctx, parent_atoms):
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
                parent_atoms=parent_atoms,
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
            or ctx.facts.nonring_heavy_atom_count < 2
        ):
            return
        if _should_defer_acyclic_skeleton_to_exocyclic_ring_parent(ctx):
            return
        parent_atoms = _nonring_heavy_atom_indices(mol)
        if _should_defer_acyclic_sulfur_imidoyl_holder_to_ring_parent(ctx, parent_atoms):
            return
        name_record = ctx.acyclic_name_record(allow_ring_atoms=False)
        name = name_record.name
        if name and _should_defer_substituent_like_acyclic_name_to_ring_parent(ctx, name):
            return
        if name and "imino" in name and _prefer_acyclic_skeleton_name(name):
            source_family = ""
            if _has_isothiocyanate_motif(mol, ctx.Chem):
                source_family = "acyclic-isothiocyanate-imino-thione"
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=parent_atoms,
                source_family=source_family,
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
            or ctx.facts.nonring_heavy_atom_count < 8
        ):
            return
        if _should_defer_acyclic_skeleton_to_exocyclic_ring_parent(ctx):
            return
        parent_atoms = _nonring_heavy_atom_indices(ctx.mol)
        if _should_defer_acyclic_sulfur_imidoyl_holder_to_ring_parent(ctx, parent_atoms):
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
                parent_atoms=parent_atoms,
                stereo_locants=_complete_acyclic_stereo_locants(ctx, name, name_record.stereo_locants),
                estimated_length=len(name),
            )


def _may_have_chain_dominant_acyclic_parent(mol) -> bool:
    """Cheap graph superset for chain-dominant acyclic parent rendering."""

    ring_atoms = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.IsInRing()
    }
    nonring_atoms = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and not atom.IsInRing()
    }
    if not ring_atoms or len(nonring_atoms) < 2:
        return False
    if len(nonring_atoms) >= len(ring_atoms):
        return True
    if mol.GetRingInfo().NumRings() > 1 and len(nonring_atoms) >= 4:
        return True
    if any(mol.GetAtomWithIdx(idx).GetSymbol() != "C" for idx in nonring_atoms):
        return True
    return any(
        bond.GetBondTypeAsDouble() > 1.0
        and bond.GetBeginAtomIdx() in nonring_atoms
        and bond.GetEndAtomIdx() in nonring_atoms
        for bond in mol.GetBonds()
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
    cached = mol_cache_get(mol, "nonring_heavy_atom_count")
    if cached is not None:
        return cached
    return mol_cache_set(
        mol,
        "nonring_heavy_atom_count",
        sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1 and not atom.IsInRing()),
    )


def _heavy_atom_indices(mol) -> frozenset[int]:
    cached = mol_cache_get(mol, "heavy_atom_indices")
    if cached is not None:
        return cached
    return mol_cache_set(
        mol,
        "heavy_atom_indices",
        frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1),
    )


def _nonring_heavy_atom_indices(mol) -> frozenset[int]:
    cached = mol_cache_get(mol, "nonring_heavy_atom_indices")
    if cached is not None:
        return cached
    return mol_cache_set(
        mol,
        "nonring_heavy_atom_indices",
        frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1 and not atom.IsInRing()),
    )


def _should_defer_acyclic_sulfur_imidoyl_holder_to_ring_parent(
    ctx,
    parent_atoms: Sequence[int] | frozenset[int],
) -> bool:
    """Leave cyclic parents in charge when an acyclic skeleton is only an S=N branch holder."""

    mol = ctx.mol
    if mol.GetRingInfo().NumRings() == 0:
        return False
    parent_set = frozenset(int(idx) for idx in parent_atoms)
    if not parent_set:
        return False
    ring_atoms = frozenset(
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.IsInRing()
    )
    if parent_set & ring_atoms:
        return False
    return _has_ring_attached_sulfur_imidoyl_branch(mol, parent_set, ring_atoms)


def _has_nonring_sulfur_oxo_acyclic_parent_center(mol, parent_atoms: Sequence[int] | frozenset[int]) -> bool:
    parent_set = frozenset(int(idx) for idx in parent_atoms)
    if not parent_set:
        return False
    for sulfur in mol.GetAtoms():
        sulfur_idx = sulfur.GetIdx()
        if sulfur_idx not in parent_set:
            continue
        if sulfur.GetAtomicNum() != 16 or sulfur.IsInRing() or sulfur.GetFormalCharge() != 0:
            continue
        if sulfur.GetIsotope():
            continue
        double_oxygen_count = 0
        double_nitrogen_count = 0
        single_nitrogen_count = 0
        single_carbon_count = 0
        unsupported = False
        for neighbor in sulfur.GetNeighbors():
            nidx = neighbor.GetIdx()
            if neighbor.GetAtomicNum() <= 1:
                continue
            bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
            if bond is None:
                unsupported = True
                break
            order = bond.GetBondTypeAsDouble()
            atomic_num = neighbor.GetAtomicNum()
            if atomic_num == 8 and order == 2.0 and _is_terminal_hetero_atom(neighbor, sulfur_idx):
                double_oxygen_count += 1
                continue
            if atomic_num == 7 and neighbor.GetFormalCharge() == 0 and order == 2.0 and nidx in parent_set:
                double_nitrogen_count += 1
                continue
            if atomic_num == 7 and neighbor.GetFormalCharge() == 0 and order == 1.0 and nidx in parent_set:
                single_nitrogen_count += 1
                continue
            if atomic_num == 6 and order == 1.0:
                single_carbon_count += 1
                continue
            unsupported = True
            break
        if unsupported:
            continue
        if not _sulfur_oxo_center_has_aromatic_attachment(mol, sulfur_idx, parent_set):
            continue
        if _sulfur_oxo_center_attaches_to_fused_dicarbonyl_aza_ring(mol, sulfur_idx, parent_set):
            continue
        if _sulfur_oxo_center_has_protected_nonaromatic_ring_ligand(mol, sulfur_idx, parent_set):
            continue
        if double_oxygen_count == 2 and single_nitrogen_count >= 1 and double_nitrogen_count == 0:
            return True
        if (
            double_oxygen_count == 1
            and double_nitrogen_count >= 1
            and (single_nitrogen_count + single_carbon_count) >= 1
            and not _sulfoximine_center_defer_to_ring_or_retained_parent(mol, sulfur_idx, parent_set)
        ):
            return True
    return False


def _sulfur_oxo_center_has_aromatic_attachment(
    mol,
    sulfur_idx: int,
    parent_atoms: Sequence[int] | frozenset[int],
) -> bool:
    return bool(_sulfur_oxo_center_aromatic_attachment_atoms(mol, sulfur_idx, parent_atoms))


def _sulfur_oxo_center_aromatic_attachment_atoms(
    mol,
    sulfur_idx: int,
    parent_atoms: Sequence[int] | frozenset[int],
) -> frozenset[int]:
    parent_set = frozenset(int(idx) for idx in parent_atoms)
    attached: set[int] = set()
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    for neighbor in sulfur.GetNeighbors():
        nidx = neighbor.GetIdx()
        if neighbor.GetAtomicNum() <= 1:
            continue
        if neighbor.IsInRing() and neighbor.GetIsAromatic():
            attached.add(nidx)
            continue
        if nidx not in parent_set or neighbor.GetAtomicNum() != 7:
            continue
        for distal in neighbor.GetNeighbors():
            didx = distal.GetIdx()
            if didx == sulfur_idx or distal.GetAtomicNum() <= 1:
                continue
            if distal.IsInRing() and distal.GetIsAromatic():
                attached.add(didx)
                continue
            if didx not in parent_set or distal.GetAtomicNum() != 6:
                continue
            for ring_neighbor in distal.GetNeighbors():
                ridx = ring_neighbor.GetIdx()
                if ridx == nidx or ring_neighbor.GetAtomicNum() <= 1:
                    continue
                if ring_neighbor.IsInRing() and ring_neighbor.GetIsAromatic():
                    attached.add(ridx)
    return frozenset(attached)


def _sulfur_oxo_center_attaches_to_fused_dicarbonyl_aza_ring(
    mol,
    sulfur_idx: int,
    parent_atoms: Sequence[int] | frozenset[int],
) -> bool:
    for atom_idx in _sulfur_oxo_center_aromatic_attachment_atoms(mol, sulfur_idx, parent_atoms):
        ring_system = _fused_ring_system_for_atom(mol, atom_idx)
        if len(ring_system) < 8:
            continue
        ring_nitrogen_count = sum(1 for idx in ring_system if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7)
        if ring_nitrogen_count < 2:
            continue
        carbonyl_count = sum(
            1
            for idx in ring_system
            if _ring_atom_has_exocyclic_chalcogen_double_bond(mol, idx, ring_system)
        )
        if carbonyl_count >= 2:
            return True
    return False


def _fused_ring_system_for_atom(mol, atom_idx: int) -> frozenset[int]:
    atom_rings = [frozenset(ring) for ring in mol.GetRingInfo().AtomRings() if atom_idx in ring]
    if not atom_rings:
        return frozenset()
    component = set(atom_rings)
    changed = True
    while changed:
        changed = False
        for ring in (frozenset(item) for item in mol.GetRingInfo().AtomRings()):
            if ring in component:
                continue
            if any(len(ring & existing) >= 2 for existing in component):
                component.add(ring)
                changed = True
    return frozenset().union(*component)


def _sulfoximine_center_defer_to_ring_or_retained_parent(
    mol,
    sulfur_idx: int,
    parent_atoms: Sequence[int] | frozenset[int],
) -> bool:
    parent_set = frozenset(int(idx) for idx in parent_atoms)
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    for neighbor in sulfur.GetNeighbors():
        nidx = neighbor.GetIdx()
        if neighbor.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if bond is None:
            return True
        order = bond.GetBondTypeAsDouble()
        if neighbor.GetAtomicNum() == 7 and order == 2.0 and nidx in parent_set:
            if any(
                distal.GetIdx() != sulfur_idx and distal.GetAtomicNum() > 1 and distal.IsInRing()
                for distal in neighbor.GetNeighbors()
            ):
                return True
        if neighbor.GetAtomicNum() == 6 and order == 1.0 and neighbor.IsInRing() and neighbor.GetIsAromatic():
            return True
    return False


def _sulfur_oxo_center_has_protected_nonaromatic_ring_ligand(
    mol,
    sulfur_idx: int,
    parent_atoms: Sequence[int] | frozenset[int],
) -> bool:
    parent_set = frozenset(int(idx) for idx in parent_atoms)
    seen = {sulfur_idx}
    stack = [
        (neighbor.GetIdx(), 0)
        for neighbor in mol.GetAtomWithIdx(sulfur_idx).GetNeighbors()
        if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() in parent_set
    ]
    while stack:
        atom_idx, depth = stack.pop()
        if atom_idx in seen:
            continue
        seen.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx == sulfur_idx or neighbor.GetAtomicNum() <= 1:
                continue
            if nidx not in parent_set:
                if neighbor.IsInRing() and not neighbor.GetIsAromatic():
                    ring_system = _fused_ring_system_for_atom(mol, nidx)
                    if _protected_nonaromatic_ring_system(mol, ring_system):
                        return True
                continue
            if depth < 5 and nidx not in seen:
                stack.append((nidx, depth + 1))
    return False


def _protected_nonaromatic_ring_system(mol, ring_system: frozenset[int]) -> bool:
    if not ring_system:
        return False
    if any(
        mol.GetAtomWithIdx(idx).GetAtomicNum() in {7, 8, 16}
        for idx in ring_system
    ):
        return True
    ring_count = _ring_system_ring_count(mol, ring_system)
    return ring_count >= 2 or len(ring_system) > 6


def _ring_system_ring_count(mol, ring_system: frozenset[int]) -> int:
    return sum(1 for ring in mol.GetRingInfo().AtomRings() if frozenset(ring) <= ring_system)


def _acyclic_parent_has_ring_attachment(mol, parent_atoms: Sequence[int] | frozenset[int]) -> bool:
    parent_set = frozenset(int(idx) for idx in parent_atoms)
    for atom_idx in parent_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in parent_set and neighbor.IsInRing():
                return True
    return False


def _has_ring_attached_sulfur_imidoyl_branch(
    mol,
    parent_atoms: frozenset[int],
    ring_atoms: frozenset[int],
) -> bool:
    for sulfur in mol.GetAtoms():
        sulfur_idx = sulfur.GetIdx()
        if sulfur_idx not in parent_atoms:
            continue
        if sulfur.GetAtomicNum() != 16 or sulfur.IsInRing() or sulfur.GetFormalCharge() != 0:
            continue
        if sulfur.GetIsotope():
            continue
        double_oxygens: list[int] = []
        imino_nitrogens: list[int] = []
        single_heavy: list[int] = []
        unsupported = False
        for neighbor in sulfur.GetNeighbors():
            nidx = neighbor.GetIdx()
            if neighbor.GetAtomicNum() <= 1:
                continue
            bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
            if bond is None:
                unsupported = True
                break
            order = bond.GetBondTypeAsDouble()
            if neighbor.GetAtomicNum() == 8 and order == 2.0 and _is_terminal_hetero_atom(neighbor, sulfur_idx):
                double_oxygens.append(nidx)
            elif neighbor.GetAtomicNum() == 7 and order == 2.0 and neighbor.GetFormalCharge() == 0:
                imino_nitrogens.append(nidx)
            elif order == 1.0:
                single_heavy.append(nidx)
            else:
                unsupported = True
                break
        if unsupported or len(imino_nitrogens) != 1 or len(double_oxygens) > 1:
            continue
        imino_idx = imino_nitrogens[0]
        if imino_idx not in parent_atoms:
            continue
        ring_single = [idx for idx in single_heavy if idx in ring_atoms]
        nonring_single = [idx for idx in single_heavy if idx not in ring_atoms]
        if (
            len(ring_single) == 1
            and len(nonring_single) == 1
            and mol.GetAtomWithIdx(nonring_single[0]).GetAtomicNum() == 7
        ):
            return True
        if (
            len(double_oxygens) == 1
            and len(ring_single) == 2
            and all(mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 for idx in ring_single)
        ):
            return True
    return False


def _is_terminal_hetero_atom(atom, parent_idx: int) -> bool:
    return not any(
        neighbor.GetIdx() != parent_idx and neighbor.GetAtomicNum() > 1
        for neighbor in atom.GetNeighbors()
    )


def _senior_azolone_ring_atom_sets(mol) -> tuple[frozenset[int], ...]:
    cached = mol_cache_get(mol, "senior_azolone_ring_atom_sets")
    if cached is not None:
        return cached
    ring_sets: list[frozenset[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 5:
            continue
        ring_set = frozenset(ring)
        atoms = tuple(mol.GetAtomWithIdx(idx) for idx in ring)
        nitrogen_count = sum(1 for atom in atoms if atom.GetAtomicNum() == 7)
        hetero_count = sum(1 for atom in atoms if atom.GetAtomicNum() in {7, 8, 16})
        if nitrogen_count < 2 or hetero_count < 3:
            continue
        if not any(atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0 for atom in atoms):
            continue
        if not any(_ring_atom_has_exocyclic_chalcogen_double_bond(mol, idx, ring_set) for idx in ring):
            continue
        ring_sets.append(ring_set)
    return mol_cache_set(mol, "senior_azolone_ring_atom_sets", tuple(ring_sets))


def _senior_oxo_aza_fused_ring_system_atom_sets(mol) -> tuple[frozenset[int], ...]:
    cached = mol_cache_get(mol, "senior_oxo_aza_fused_ring_system_atom_sets")
    if cached is not None:
        return cached
    atom_rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    ring_sets: list[frozenset[int]] = []
    seen: set[frozenset[int]] = set()
    for ring in atom_rings:
        fused_component = {ring}
        changed = True
        while changed:
            changed = False
            for other in atom_rings:
                if other in fused_component:
                    continue
                if any(len(other & component_ring) >= 2 for component_ring in fused_component):
                    fused_component.add(other)
                    changed = True
        if len(fused_component) < 2:
            continue
        ring_system = frozenset().union(*fused_component)
        if ring_system in seen:
            continue
        seen.add(ring_system)
        atoms = tuple(mol.GetAtomWithIdx(idx) for idx in ring_system)
        nitrogen_count = sum(1 for atom in atoms if atom.GetAtomicNum() == 7)
        hetero_count = sum(1 for atom in atoms if atom.GetAtomicNum() in {7, 8, 16})
        if nitrogen_count < 3 or hetero_count < 3:
            continue
        ring_nh_count = sum(
            1
            for atom in atoms
            if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 0 and atom.GetTotalNumHs() > 0
        )
        if ring_nh_count < 2:
            continue
        if not any(_ring_atom_has_exocyclic_chalcogen_double_bond(mol, idx, ring_system) for idx in ring_system):
            continue
        ring_sets.append(ring_system)
    return mol_cache_set(mol, "senior_oxo_aza_fused_ring_system_atom_sets", tuple(ring_sets))


def _ring_atom_has_exocyclic_chalcogen_double_bond(mol, atom_idx: int, ring_set: frozenset[int]) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetAtomicNum() != 6:
        return False
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetIdx() in ring_set:
            continue
        if other.GetAtomicNum() in {8, 16} and bond.GetBondTypeAsDouble() == 2.0:
            return True
    return False


def _parent_leaves_senior_azolone_ring_outside(mol, parent_atoms: Sequence[int] | frozenset[int]) -> bool:
    senior_rings = _senior_azolone_ring_atom_sets(mol) + _senior_oxo_aza_fused_ring_system_atom_sets(mol)
    if not senior_rings:
        return False
    parent_set = frozenset(parent_atoms)
    if not parent_set:
        return False
    if parent_set == _heavy_atom_indices(mol):
        return False
    return not any(ring <= parent_set for ring in senior_rings)


def _parent_atoms_have_spiro_topology(mol, parent_atoms: Sequence[int] | frozenset[int]) -> bool:
    parent_set = frozenset(parent_atoms)
    if not parent_set:
        return False
    parent_rings = [frozenset(ring) for ring in mol.GetRingInfo().AtomRings() if frozenset(ring) <= parent_set]
    for left, right in combinations(parent_rings, 2):
        if len(left & right) == 1:
            return True
    return False


def _should_defer_spiro_core_parent_to_senior_azolone(
    mol,
    parent_atoms: Sequence[int] | frozenset[int],
) -> bool:
    return _parent_leaves_senior_azolone_ring_outside(mol, parent_atoms) and _parent_atoms_have_spiro_topology(
        mol,
        parent_atoms,
    )


def _should_defer_spiro_context_parent_to_senior_azolone(
    ctx,
    parent_atoms: Sequence[int] | frozenset[int],
) -> bool:
    return bool(ctx.has_spiro_topology) and _parent_leaves_senior_azolone_ring_outside(ctx.mol, parent_atoms)


def _should_defer_spiro_substituent_parent_to_senior_azolone(
    ctx,
    parent_atoms: Sequence[int] | frozenset[int],
) -> bool:
    return _should_defer_spiro_context_parent_to_senior_azolone(ctx, parent_atoms)


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
            "arabinosyl",
            "fructosyl",
            "fucosyl",
            "galactosyl",
            "glucopyranosyl",
            "glucosyl",
            "galactopyranosyl",
            "lactosyl",
            "maltosyl",
            "mannosyl",
            "mannopyranosyl",
            "fucopyranosyl",
            "ribofuranosyl",
            "ribosyl",
            "xylosyl",
            "apiosyl",
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
    return _assigned_bond_stereo_atom_sets_for_mol(ctx.mol, ctx.Chem)


def _assigned_bond_stereo_atom_sets_for_mol(mol, Chem) -> tuple[frozenset[int], ...]:
    records: list[frozenset[int]] = []
    for bond in mol.GetBonds():
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

    cached = ctx.cache.substituent_like_acyclic_defer.get(name)
    if cached is not None:
        return cached
    mol = ctx.mol
    ring_count = mol.GetRingInfo().NumRings()
    if ring_count < 1:
        return _cache_substituent_like_acyclic_defer(ctx, name, False)
    if _has_unrendered_ring_isotope(mol, name):
        return _cache_substituent_like_acyclic_defer(ctx, name, True)
    if _has_ringless_short_functional_parent_for_cyclic_molecule(name):
        return _cache_substituent_like_acyclic_defer(ctx, name, True)
    if ring_count == 1:
        return _cache_substituent_like_acyclic_defer(
            ctx,
            name,
            _has_substituent_like_monocyclic_parent_name(name),
        )
    monocyclic_defer = _has_substituent_like_monocyclic_parent_name(name)
    kekulized_short_functional_defer = (
        _has_short_functional_acyclic_parent_suffix(name)
        and _has_functional_ring_substituent_chain_shape(name)
    )
    kekulized_pterin_defer = (
        _has_pterin_retained_fused_parent_substituent_name(name)
        and _has_stemmed_short_acyclic_parent_suffix(name)
        and _has_functional_ring_substituent_chain_shape(name)
    )
    polycyclic_defer_possible = (
        monocyclic_defer
        or _has_ring_substituent_artifact_name(name)
        or ("diamyl" in name and any(term in name for term in ("cyclopropyl", "piperidin", "pyrrolidin")))
        or ("diamyl" in name and "dioxolan" in name)
        or "azamethylcarbonyl" in name
        or (
            name.endswith("-methane")
            and any(term in name for term in ("carbonyl", "carboxy", "isocyano", "oxomethyl", "sulfonyl"))
        )
        or _has_unsubstituted_sulfonimidoyl_methane_parent(name)
        or (name.endswith("-propane") and "carbonyloxy" in name)
        or (
            _has_retained_fused_parent_substituent_name(name)
            and any(
                term in name
                for term in ("amino", "carbonyl", "carboxy", "cyano", "imino", "oxo", "aza", "oxa", "thia")
            )
            and _has_short_functional_acyclic_parent_suffix(name)
        )
        or kekulized_short_functional_defer
        or kekulized_pterin_defer
    )
    if not polycyclic_defer_possible:
        return _cache_substituent_like_acyclic_defer(ctx, name, False)
    kekulized_records = None
    has_polycyclic_parent_record = ctx.polycyclic_parent_records("bicyclic") or ctx.polycyclic_parent_records(
        "spiro"
    )
    if not has_polycyclic_parent_record:
        kekulized_records = ctx.polycyclic_parent_records("kekulized")
        has_polycyclic_parent_record = bool(kekulized_records)
    if not has_polycyclic_parent_record:
        return _cache_substituent_like_acyclic_defer(ctx, name, monocyclic_defer)
    if any(term in name for term in ("bicyclo[", "tricyclo[", "spiro[")):
        return _cache_substituent_like_acyclic_defer(ctx, name, False)
    if _has_ring_substituent_artifact_name(name):
        return _cache_substituent_like_acyclic_defer(ctx, name, True)
    if kekulized_records is None:
        kekulized_records = ctx.polycyclic_parent_records("kekulized")
    if kekulized_records and kekulized_short_functional_defer:
        return _cache_substituent_like_acyclic_defer(ctx, name, True)
    if kekulized_records and kekulized_pterin_defer:
        return _cache_substituent_like_acyclic_defer(ctx, name, True)
    if "diamyl" in name and any(term in name for term in ("cyclopropyl", "piperidin", "pyrrolidin")):
        return _cache_substituent_like_acyclic_defer(ctx, name, True)
    if "diamyl" in name and "dioxolan" in name:
        return _cache_substituent_like_acyclic_defer(ctx, name, True)
    if "azamethylcarbonyl" in name:
        return _cache_substituent_like_acyclic_defer(ctx, name, True)
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
        return _cache_substituent_like_acyclic_defer(ctx, name, True)
    if _has_unsubstituted_sulfonimidoyl_methane_parent(name):
        return _cache_substituent_like_acyclic_defer(ctx, name, True)
    if name.endswith("-propane") and "carbonyloxy" in name:
        return _cache_substituent_like_acyclic_defer(ctx, name, True)
    if (
        _has_retained_fused_parent_substituent_name(name)
        and any(term in name for term in ("amino", "carbonyl", "carboxy", "cyano", "imino", "oxo", "aza", "oxa", "thia"))
        and _has_short_functional_acyclic_parent_suffix(name)
    ):
        return _cache_substituent_like_acyclic_defer(ctx, name, True)
    return _cache_substituent_like_acyclic_defer(ctx, name, False)


def _cache_substituent_like_acyclic_defer(ctx, name: str, value: bool) -> bool:
    ctx.cache.substituent_like_acyclic_defer[name] = value
    return value


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


def _formal_charge_atom_count(mol) -> int:
    cached = mol_cache_get(mol, "formal_charge_atom_count")
    if cached is not None:
        return cached
    return mol_cache_set(mol, "formal_charge_atom_count", sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge()))


def _abs_formal_charge_sum(mol) -> int:
    cached = mol_cache_get(mol, "abs_formal_charge_sum")
    if cached is not None:
        return cached
    return mol_cache_set(mol, "abs_formal_charge_sum", sum(abs(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


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
        return _retained_fused_parent_templates(self._Chem)

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.candidate_templates(self._templates())

    def _candidate_sort_key(self, mol, template: dict, match: Sequence[int], name: str) -> tuple:
        return (
            _benzimidazole_template_h_preference(mol, template, match, name),
            _benzimidazole_external_n_preference(mol, template, match),
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
        for template, match in _cached_iter_fused_template_matches(
            ctx, self._template_iter(ctx), self._Chem, reject_extra_core_bonds=True
        ):
            if not _exact_retained_fused_match_is_hydrogen_compatible(mol, template, match):
                continue
            charge_penalty = _biochemical_unrendered_parent_charge_penalty(mol, template, match)
            facts.append(
                _RetainedFusedCandidateFacts(
                    template=template,
                    match=tuple(match),
                    topology_query=False,
                    structural_key=_retained_fused_fact_sort_key(mol, template, match, topology_query=False),
                    unrendered_charge_penalty=charge_penalty,
                )
            )
            exact_facts += 1

        for template, match in _cached_iter_fused_template_matches(
            ctx,
            self._template_iter(ctx),
            self._Chem,
            topology_query=True,
            require_one_ring_system=True,
            reject_extra_core_bonds=True,
        ):
            if not _retained_topology_fused_match_is_hydrogen_compatible(mol, template, match):
                continue
            charge_penalty = _biochemical_unrendered_parent_charge_penalty(mol, template, match)
            facts.append(
                _RetainedFusedCandidateFacts(
                    template=template,
                    match=tuple(match),
                    topology_query=True,
                    structural_key=_retained_fused_fact_sort_key(mol, template, match, topology_query=True),
                    unrendered_charge_penalty=charge_penalty,
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
            if _should_defer_spiro_context_parent_to_senior_azolone(ctx, match):
                continue
            sort_key = self._candidate_sort_key(mol, template, match, name)
            if fact.topology_query:
                sort_key = (
                    _retained_topology_fused_candidate_rank(template_record_name(template), name),
                    sort_key,
                )
            yielded_names.add(name)
            mol_cache_set(mol, "has_retained_fused_parent_candidate", True)
            source_family = _retained_fused_source_family(mol, template, match)
            charge = _polycyclic_parent_charge_evidence(mol, match)
            safety = CandidateSafety(charge_supported=False) if fact.unrendered_charge_penalty else None
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=self.priority,
                sort_key=sort_key,
                parent_atoms=frozenset(match),
                stereo_locants=_template_match_stereo_locants(template, match),
                template_id=id(template),
                template_name=template_record_name(template),
                source_family=source_family,
                source_file=str(template.get("source_file", "")),
                parent_substituted_hetero_atom_count=_parent_substituted_hetero_atom_count(mol, match),
                charge=charge,
                stereo=_molecule_stereo_evidence(mol, self._Chem),
                safety=safety,
                estimated_length=len(name),
                render_rank=render_rank,
            )


_RETAINED_FUSED_PARENT_SOURCE_FAMILIES = frozenset(
    {
        "",
        "arylGroups:ring:biochemical",
        "arylGroups:ring:ring",
        "carbohydrates:carbohydrate:ring",
        "naturalProducts:ring:biochemical",
        "opsin-retained-ring-cml",
        "opsin-test-scaffold",
        "simpleCyclicGroups:ring:biochemical",
        "simpleCyclicGroups:ring:simpleGroup",
        "simpleGroups:simpleGroup:simpleGroup",
        "static-retained",
    }
)


@lru_cache(maxsize=8)
def _retained_fused_parent_templates(Chem) -> tuple[dict, ...]:
    return tuple(
        template
        for template in _curated_fused_templates(Chem)
        if str(template.get("source_family", "")) in _RETAINED_FUSED_PARENT_SOURCE_FAMILIES
    )


@dataclass(frozen=True, slots=True)
class _RetainedFusedCandidateFacts:
    template: dict
    match: tuple[int, ...]
    topology_query: bool
    structural_key: tuple
    unrendered_charge_penalty: int = 0


def _retained_fused_fact_sort_key(
    mol,
    template: dict,
    match: Sequence[int],
    *,
    topology_query: bool,
) -> tuple:
    template_name = template_record_name(template)
    return (
        1 if topology_query else 0,
        _biochemical_unrendered_parent_charge_penalty(mol, template, match),
        _template_source_rank(template),
        _hydrogenated_template_penalty(template_name),
        -len(match),
        len(template_name),
        template_name,
        _benzimidazole_template_h_preference(mol, template, match, template_name),
        _benzimidazole_external_n_preference(mol, template, match),
        _benzimidazole_external_carbon_locant_preference(mol, template, match),
        tuple(match),
    )


def _retained_fused_render_set(
    facts: tuple[_RetainedFusedCandidateFacts, ...],
) -> tuple[_RetainedFusedCandidateFacts, ...]:
    if len(facts) <= _RETAINED_FUSED_RENDER_SET_SIZE:
        if any(_retained_benzimidazole_h_fact_is_meaningful(fact) for fact in facts):
            return tuple(sorted(facts, key=lambda fact: fact.structural_key))
        if any(_retained_fused_biochemical_charge_fact_is_meaningful(fact) for fact in facts):
            return tuple(
                fact
                for _index, fact in sorted(
                    enumerate(facts),
                    key=lambda indexed: (indexed[1].unrendered_charge_penalty, indexed[0]),
                )
            )
        return facts
    return tuple(sorted(facts, key=lambda fact: fact.structural_key)[:_RETAINED_FUSED_RENDER_SET_SIZE])


def _retained_fused_biochemical_charge_fact_is_meaningful(
    fact: _RetainedFusedCandidateFacts,
) -> bool:
    return bool(fact.unrendered_charge_penalty)


def _retained_benzimidazole_h_fact_is_meaningful(fact: _RetainedFusedCandidateFacts) -> bool:
    template_name = template_record_name(fact.template)
    if template_name == "1H-benzimidazole":
        return True
    if template_name != "benzimidazole" or len(fact.structural_key) < 7:
        return False
    return fact.structural_key[6] != (2,)


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
            source_family = _retained_fused_source_family(
                mol,
                fact.template,
                fact.match,
                default=fact.compiled.record.source_family,
            )
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
                source_family=source_family,
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


def _exact_retained_fused_match_is_hydrogen_compatible(mol, template: dict, match: Sequence[int]) -> bool:
    """Reject high-impact fused tautomer aliases with incompatible ring N-H sites."""

    if template.get("source_file") != "high_impact_fused_parent_templates.json":
        return True
    template_name = template_record_name(template)
    if "H" not in template_name and "hydro" not in template_name:
        return True
    core = set(match)
    template_mol = template_record_mol(template)
    for template_idx, mol_idx in enumerate(match):
        template_atom = template_mol.GetAtomWithIdx(template_idx)
        if template_atom.GetSymbol() not in {"N", "O", "S"}:
            continue
        mol_atom = mol.GetAtomWithIdx(mol_idx)
        if mol_atom.GetSymbol() != template_atom.GetSymbol():
            return False
        external_substituted = any(
            nbr.GetIdx() not in core and nbr.GetAtomicNum() > 1
            for nbr in mol_atom.GetNeighbors()
        )
        if external_substituted and mol_atom.GetTotalNumHs() == 0:
            continue
        if int(template_atom.GetTotalNumHs()) != int(mol_atom.GetTotalNumHs()):
            return False
    return True


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


def _external_locants_from_sequence(mol, seq: Sequence[int]) -> tuple[tuple[int, str], ...]:
    parent_atoms = frozenset(seq)
    locants = []
    for locant, atom_idx in enumerate(seq, start=1):
        atom = mol.GetAtomWithIdx(atom_idx)
        if any(nbr.GetIdx() not in parent_atoms and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
            locants.append((int(atom_idx), str(locant)))
    return tuple(locants)


def _complete_single_ring_parent_stereo_locants(mol, seq: Sequence[int], Chem) -> tuple[tuple[int, str], ...]:
    parent_atoms = frozenset(seq)
    if not parent_atoms:
        return ()
    assigned_atoms = {idx for idx, _descriptor in assigned_atom_stereochemistry(mol, Chem)}
    assigned_bonds = _assigned_bond_stereo_atom_sets_for_mol(mol, Chem)
    assigned_parent_bonds = tuple(bond_atoms for bond_atoms in assigned_bonds if bond_atoms <= parent_atoms)
    if not assigned_atoms:
        return _sequence_stereo_locants(seq) if assigned_parent_bonds else ()
    assigned_parent_atoms = assigned_atoms & parent_atoms
    if not assigned_parent_atoms:
        return _sequence_stereo_locants(seq) if assigned_parent_bonds else ()
    if assigned_atoms.issubset(parent_atoms):
        return _sequence_stereo_locants(seq)
    if _external_stereo_represented_by_relative_ring_branches(
        mol,
        parent_atoms,
        assigned_atoms - parent_atoms,
        Chem,
    ):
        return _sequence_stereo_locants(seq)
    if len(assigned_parent_atoms) < 2:
        return ()
    return _sequence_stereo_locants(seq)


def _external_stereo_represented_by_relative_ring_branches(
    mol,
    parent_atoms: frozenset[int],
    external_assigned_atoms: set[int],
    Chem,
) -> bool:
    remaining = set(external_assigned_atoms)
    if not remaining:
        return True
    relative_rings = _relative_stereo_ring_branch_sets(mol, Chem)
    if not relative_rings:
        return False
    for atom_idx in tuple(external_assigned_atoms):
        if atom_idx not in remaining:
            continue
        represented = False
        for ring, tagged in relative_rings:
            if atom_idx not in ring or ring & parent_atoms:
                continue
            ring_remaining = remaining & ring
            if not ring_remaining:
                continue
            if not ring_remaining.issubset(tagged):
                continue
            remaining -= ring_remaining
            represented = True
            break
        if not represented:
            return False
    return not remaining


def _relative_stereo_ring_branch_sets(mol, Chem) -> tuple[tuple[frozenset[int], frozenset[int]], ...]:
    cached = mol_cache_get(mol, "relative_stereo_ring_branch_sets")
    if cached is not None:
        return cached
    rings = []
    for raw_ring in mol.GetRingInfo().AtomRings():
        ring = frozenset(raw_ring)
        tagged = _simple_relative_stereo_ring_branch_tagged_atoms(mol, ring, Chem)
        if tagged:
            rings.append((ring, tagged))
    return mol_cache_set(mol, "relative_stereo_ring_branch_sets", tuple(rings))


def _simple_relative_stereo_ring_branch(mol, ring: frozenset[int], Chem) -> bool:
    return bool(_simple_relative_stereo_ring_branch_tagged_atoms(mol, ring, Chem))


def _simple_relative_stereo_ring_branch_tagged_atoms(mol, ring: frozenset[int], Chem) -> frozenset[int]:
    tagged = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetIdx() in ring and atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    if len(tagged) != 2:
        return frozenset()
    if any(mol.GetAtomWithIdx(atom_idx).GetSymbol() != "C" for atom_idx in ring):
        return frozenset()
    if any(atom.GetSymbol() != "C" or atom.GetIsAromatic() for atom in tagged):
        return frozenset()
    if any(_ring_external_heavy_degree(mol, atom.GetIdx(), ring) != 1 for atom in tagged):
        return frozenset()
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring and bond.GetBondType() != Chem.BondType.SINGLE:
            return frozenset()
    return frozenset(atom.GetIdx() for atom in tagged)


def _ring_external_heavy_degree(mol, atom_idx: int, ring_atoms: frozenset[int]) -> int:
    return sum(
        1
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if nbr.GetIdx() not in ring_atoms and nbr.GetAtomicNum() > 1
    )


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
        return (
            _generated_benzimidazole_external_n_preference(mol, template, match),
            _generated_fused_graph_orientation_preference(mol, template, match),
            *_fused_candidate_preference(name),
            -len(match),
            name,
        )

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
        for template, match in _cached_iter_fused_template_matches(
            ctx, self._template_iter(ctx), self._Chem, reject_extra_core_bonds=True
        ):
            facts.append(
                _GeneratedFusedCandidateFacts(
                    template=template,
                    match=tuple(match),
                    topology_query=False,
                    structural_key=_generated_fused_fact_sort_key(mol, template, match, topology_query=False),
                )
            )
            exact_facts += 1

        topology_templates = tuple(
            template
            for template in self._template_iter(ctx)
            if _allow_generated_topology_template(template_record_name(template))
        )
        for template, match in _cached_iter_fused_template_matches(
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
                    structural_key=_generated_fused_fact_sort_key(mol, template, match, topology_query=True),
                )
            )
            topology_facts += 1

        if ctx.telemetry is not None:
            ctx.telemetry.filter_stage_count[f"{self.family}:exact_facts"] += exact_facts
            ctx.telemetry.filter_stage_count[f"{self.family}:topology_facts"] += topology_facts
        return _filter_generated_fused_orientation_alias_facts(tuple(facts))

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
            if _should_defer_spiro_context_parent_to_senior_azolone(ctx, match):
                continue
            yielded_names.add(name)
            yielded += 1
            mol_cache_set(mol, "has_retained_fused_parent_candidate", True)
            sort_key = self._candidate_sort_key(mol, template, match, name)
            if fact.topology_query:
                sort_key = (_generated_topology_fused_candidate_rank(template_name, name), *sort_key)
            reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=self.priority,
                sort_key=sort_key,
                **_fused_parent_graph_fact_kwargs(mol, match),
                parent_indicated_hydrogen_locants=tuple(
                    _matched_fused_indicated_hydrogen_locants(mol, template, reverse)
                ),
                stereo_locants=_template_match_stereo_locants(template, match),
                template_id=id(template),
                template_name=template_name,
                source_family=str(template.get("source_family", "")),
                source_file=str(template.get("source_file", "")),
                charge=_polycyclic_parent_charge_evidence(mol, match),
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


def _molecule_stereo_evidence(mol, Chem) -> StereoEvidence | None:
    atom_descriptors = tuple(assigned_atom_stereochemistry(mol, Chem, atom_order="idx"))
    bond_descriptors = tuple(
        (bond.GetIdx(), str(bond.GetStereo()))
        for bond in mol.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    )
    if not atom_descriptors and not bond_descriptors:
        return None
    return StereoEvidence(
        atom_descriptors=atom_descriptors,
        bond_descriptors=bond_descriptors,
    )


def _filter_generated_fused_orientation_alias_facts(
    facts: tuple[_GeneratedFusedCandidateFacts, ...],
) -> tuple[_GeneratedFusedCandidateFacts, ...]:
    preferred_core_keys = {
        (fact.topology_query, frozenset(fact.match))
        for fact in facts
        if _generated_fusion_orientation_role(template_record_name(fact.template)) == "preferred-pyrazolo-imidazole"
    }
    if not preferred_core_keys:
        return facts
    return tuple(
        fact
        for fact in facts
        if not (
            _generated_fusion_orientation_role(template_record_name(fact.template))
            == "disfavored-pyrazole-imidazole"
            and (fact.topology_query, frozenset(fact.match)) in preferred_core_keys
        )
    )


def _generated_fused_fact_sort_key(
    mol,
    template: dict,
    match: Sequence[int],
    *,
    topology_query: bool,
) -> tuple:
    template_name = template_record_name(template)
    return (
        1 if topology_query else 0,
        _generated_fused_exocyclic_imino_orientation_rank(mol, template_name, match),
        _generated_fused_template_anchor_rank(template_name),
        _generated_benzimidazole_external_n_preference(mol, template, match),
        _generated_fused_graph_orientation_preference(mol, template, match),
        _template_source_rank(template),
        _hydrogenated_template_penalty(template_name),
        -len(match),
        len(template_name),
        template_name,
        _template_external_substitution_locant_key(mol, template, match),
        tuple(match),
    )


def _generated_fused_exocyclic_imino_orientation_rank(
    mol,
    template_name: str,
    match: Sequence[int],
) -> int:
    if len(match) != 9:
        return 0
    if _ring_hetero_atom_count(mol, match) != 4:
        return 0
    if not _parent_has_exocyclic_imino_nitrogen(mol, match):
        return 0
    if (
        template_name == "imidazo[4,5-d]pyrimidine"
        or re.match(r"^pyrazolo\[[^\]]+\]pyrimidine$", template_name)
    ):
        return 0
    if re.match(r"^(?:pyrimido|pyrimidino)\[[^\]]+\](?:imidazole|pyrazole)$", template_name):
        return 1
    return 0


def _parent_has_exocyclic_imino_nitrogen(mol, parent_atoms: Sequence[int]) -> bool:
    parent_set = set(parent_atoms)
    for idx in parent_atoms:
        atom = mol.GetAtomWithIdx(int(idx))
        if atom.GetAtomicNum() != 6:
            continue
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx in parent_set or neighbor.GetAtomicNum() != 7:
                continue
            if neighbor.GetFormalCharge() != 0:
                continue
            bond = mol.GetBondBetweenAtoms(int(idx), nidx)
            if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
                return True
    return False


def _template_external_substitution_locant_key(mol, template: dict, match: Sequence[int]) -> tuple:
    core = set(match)
    locants = template_record_locants(template)
    external_locants = []
    for template_idx, mol_idx in enumerate(match):
        atom = mol.GetAtomWithIdx(int(mol_idx))
        if not any(nbr.GetIdx() not in core and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
            continue
        locant = locants.get(template_idx, "")
        if not locant:
            continue
        external_locants.append(_locant_sort_key(str(locant)))
    return tuple(sorted(external_locants)) or ((999, ""),)


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
        return tuple(sorted(facts, key=lambda fact: fact.structural_key))
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


class HighSignalGeneratedTopologyFusedParentRule(GeneratedFusedParentRule):
    """Topology-only generated fused parents for audited high-impact scaffolds.

    The broad generated topology path is intentionally too risky: it changes
    established retained fused matches.  This rule gives a small set of audited
    generated scaffolds their own topology render window without disturbing the
    exact generated-fused render set.
    """

    family = "high-signal-generated-topology-fused-parent"
    priority = parent_priority("generated-parent")
    min_ring_system_ring_count = 2

    def _template_iter(self, ctx) -> Sequence[dict]:
        return tuple(
            template
            for template in ctx.generated_fused_templates
            if _allow_high_signal_generated_topology_template(template_record_name(template))
        )

    def _candidate_facts(self, ctx) -> tuple["_GeneratedFusedCandidateFacts", ...]:
        mol = ctx.mol
        facts: list[_GeneratedFusedCandidateFacts] = []
        for template, match in _cached_iter_fused_template_matches(
            ctx,
            self._template_iter(ctx),
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
                    structural_key=_generated_fused_fact_sort_key(mol, template, match, topology_query=True),
                )
            )
        if ctx.telemetry is not None:
            ctx.telemetry.filter_stage_count[f"{self.family}:topology_facts"] += len(facts)
        return tuple(sorted(facts, key=lambda fact: fact.structural_key)[:16])

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
            template_name = template_record_name(template)
            if not _allow_high_signal_generated_topology_candidate(template_name, name):
                continue
            if not _allow_high_signal_generated_topology_graph_candidate(mol, match, template_name):
                continue
            if _should_defer_spiro_context_parent_to_senior_azolone(ctx, match):
                continue
            graph_facts = _fused_parent_graph_fact_kwargs(mol, match)
            yielded_names.add(name)
            yielded += 1
            sort_key = (
                _generated_topology_fused_candidate_rank(template_name, name),
                *self._candidate_sort_key(mol, template, match, name),
            )
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=self.priority,
                sort_key=sort_key,
                **graph_facts,
                stereo_locants=_template_match_stereo_locants(template, match),
                template_id=id(template),
                template_name=template_name,
                source_family=str(template.get("source_family", "")),
                source_file=str(template.get("source_file", "")),
                charge=_polycyclic_parent_charge_evidence(mol, match),
                estimated_length=len(name),
                render_rank=render_rank,
            )
            if yielded >= _MAX_GENERATED_FUSED_PARENT_CANDIDATES:
                return


_HIGH_SIGNAL_GENERATED_TOPOLOGY_TEMPLATE_NAMES = frozenset(
    {
        "imidazo[4,5-a]benzene",
        "pyridazino[1,6-a]pyrimidine",
        "pyrano[2,3-a]benzene",
    }
)


def _allow_high_signal_generated_topology_template(template_name: str) -> bool:
    return template_name in _HIGH_SIGNAL_GENERATED_TOPOLOGY_TEMPLATE_NAMES


def _allow_generated_topology_fused_candidate(template_name: str, name: str) -> bool:
    """Allow generated fused templates when exact aromatic matching is too strict.

    OPSIN-derived fused templates frequently have the right atom connectivity
    but a different aromatic/Kekule representation from RDKit input.  This path
    is limited to retained fused scaffold shapes where a suffix or indicated-H
    parent gives a more stable graph than generic bicyclo/tricyclo fallback.
    """

    if not _allow_generated_topology_template(template_name):
        return _allow_high_signal_generated_topology_candidate(template_name, name)
    if _has_generated_topology_fused_suffix_parent(name):
        return True
    return False


def _allow_generated_topology_template(template_name: str) -> bool:
    return template_name.startswith("indeno[")


def _allow_high_signal_generated_topology_candidate(template_name: str, name: str) -> bool:
    if template_name == "imidazo[4,5-a]benzene":
        return True
    if template_name == "pyridazino[1,6-a]pyrimidine":
        return bool(
            "pyridazino[1,6-a]pyrimidin" in name
            and re.search(r"(?:^|-)\d+(?:,\d+)*[a-z]?-(?:oxo|thioxo)-", name)
        )
    if template_name == "pyrano[2,3-a]benzene":
        if "dihydro-pyrido[" in name:
            return False
        return bool(
            "pyrano[2,3-a]benzene" in name
            and re.search(r"(?:^|-)2(?:[a-z])?-(?:oxo|thioxo)-", name)
        )
    return False


def _allow_high_signal_generated_topology_graph_candidate(
    mol,
    match: Sequence[int],
    template_name: str,
) -> bool:
    if template_name == "imidazo[4,5-a]benzene":
        return bool(
            _ring_terminal_hetero_substituent_count(mol, match)
            or _ring_external_nitrogen_substituent_count(mol, match)
        )
    return True


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
            for record in _named_ring_instance_records(mol, tuple(ring), self._Chem):
                source_family = (
                    "single-ring:retained-carbohydrate-substituent"
                    if record.retained_carbohydrate_substituent_count
                    else ""
                )
                yield NameCandidate(
                    name=record.name,
                    family=self.family,
                    priority=candidate_priority,
                    sort_key=_single_ring_candidate_sort_key(record.name),
                    stereo_prefix=record.stereo_prefix,
                    source_family=source_family,
                    parent_atoms=frozenset(ring),
                    parent_aromatic_atom_count=_ring_aromatic_atom_count(mol, ring),
                    parent_hetero_atom_count=_ring_hetero_atom_count(mol, ring),
                    parent_terminal_hetero_substituent_count=_ring_terminal_hetero_substituent_count(mol, ring),
                    parent_terminal_heterocarbon_substituent_count=_ring_terminal_heterocarbon_substituent_count(
                        mol,
                        ring,
                    ),
                    parent_terminal_carbonyl_substituent_count=_ring_terminal_carbonyl_substituent_count(mol, ring),
                    parent_external_substituted_hetero_atom_count=_parent_substituted_hetero_atom_count(mol, ring),
                    parent_external_nitrogen_substituent_count=_ring_external_nitrogen_substituent_count(mol, ring),
                    parent_external_oxygen_substituent_count=_ring_external_oxygen_substituent_count(mol, ring),
                    parent_external_relative_stereo_branch_count=(
                        1 if ring_fact.structural_key and ring_fact.structural_key[2] == 0 else 0
                    ),
                    locants=_locants_from_sequence(record.locant_seq),
                    stereo_locants=_complete_single_ring_parent_stereo_locants(mol, record.locant_seq, self._Chem),
                    suffixes=_single_ring_suffix_candidates(mol, record),
                    charge=_single_ring_charge_evidence(mol, ring, record.name),
                    hydrogen=_single_ring_hydrogen_evidence(record),
                    estimated_length=len(record.name),
                    render_rank=render_rank,
                )

    def _candidate_ring_facts(self, ctx, parent_rings) -> tuple["_SingleRingParentFacts", ...]:
        facts = tuple(
            _SingleRingParentFacts(
                ring=tuple(ring),
                structural_key=_single_ring_parent_fact_sort_key(ctx.mol, ring, ctx.Chem),
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
        if len(facts) <= _SINGLE_RING_PARENT_RING_FACT_SET_SIZE and not any(
            fact.structural_key and (fact.structural_key[0] == 0 or fact.structural_key[2] == 0)
            for fact in facts
        ):
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
            for record in _named_ring_instance_records(mol, tuple(ring), self._Chem):
                if not _has_positive_ring_parent_name(record.name):
                    continue
                yield NameCandidate(
                    name=record.name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=_single_ring_candidate_sort_key(record.name),
                    stereo_prefix=record.stereo_prefix,
                    parent_atoms=frozenset(ring),
                    parent_aromatic_atom_count=_ring_aromatic_atom_count(mol, ring),
                    parent_hetero_atom_count=_ring_hetero_atom_count(mol, ring),
                    parent_terminal_hetero_substituent_count=_ring_terminal_hetero_substituent_count(mol, ring),
                    parent_external_substituted_hetero_atom_count=_parent_substituted_hetero_atom_count(mol, ring),
                    parent_external_nitrogen_substituent_count=_ring_external_nitrogen_substituent_count(mol, ring),
                    parent_external_oxygen_substituent_count=_ring_external_oxygen_substituent_count(mol, ring),
                    locants=_locants_from_sequence(record.locant_seq),
                    stereo_locants=_complete_single_ring_parent_stereo_locants(mol, record.locant_seq, self._Chem),
                    suffixes=_single_ring_suffix_candidates(mol, record),
                    charge=_single_ring_charge_evidence(mol, ring, record.name),
                    hydrogen=_single_ring_hydrogen_evidence(record),
                    estimated_length=len(record.name),
                    render_rank=1,
                )


def _has_positive_ring_parent_name(name: str) -> bool:
    return bool(re.search(r"(?:^|-)[^-]+-\d+-ium(?:$|-)", name))


def _single_ring_suffix_candidates(mol, record) -> tuple[SuffixCandidate, ...]:
    if not record.suffix_kind:
        return ()
    source = "single-ring-parent"
    if record.suffix_graph_class:
        source = f"{source}:{record.suffix_graph_class}"
    safety = CandidateSafety()
    if _single_ring_suffix_has_exocyclic_charged_multiple_bond(mol, record.locant_seq, record.suffix_kind):
        safety = CandidateSafety(
            opsin_fragility_reasons=(OPSIN_FRAGILE_EXOCYCLIC_CHARGED_RING_SUFFIX,),
        )
    return (SuffixCandidate(text="", kind=record.suffix_kind, source=source, safety=safety),)


def _single_ring_hydrogen_evidence(record) -> HydrogenEvidence | None:
    if record.suffix_kind == "ring_sulfur_oxide_suffix" and record.suffix_hydrogen_locants:
        return HydrogenEvidence(saturated_locs=record.suffix_hydrogen_locants)
    return None


def _single_ring_suffix_has_exocyclic_charged_multiple_bond(
    mol,
    seq: Sequence[int],
    suffix_kind: str,
) -> bool:
    if suffix_kind not in {"ring_oxo_suffix", "ring_thione_suffix"}:
        return False
    ring_set = set(seq)
    for ring_idx in seq:
        ring_atom = mol.GetAtomWithIdx(ring_idx)
        for neighbor in ring_atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx in ring_set:
                continue
            bond = mol.GetBondBetweenAtoms(ring_idx, nidx)
            if bond is None or bond.GetBondTypeAsDouble() <= 1.5:
                continue
            if neighbor.GetFormalCharge() != 0:
                return True
    return False


@dataclass(frozen=True, slots=True)
class _SingleRingParentFacts:
    ring: tuple[int, ...]
    structural_key: tuple


def _locants_from_sequence(seq: Sequence[int]) -> tuple[tuple[int, str], ...]:
    return tuple((idx, str(locant)) for locant, idx in enumerate(seq, start=1))


def _single_ring_parent_fact_sort_key(mol, ring: Sequence[int], Chem) -> tuple:
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
        _single_ring_external_relative_stereo_branch_rank(mol, ring, Chem),
        external_bonds,
        hetero_atoms,
        tuple(sorted(ring)),
    )


def _single_ring_external_relative_stereo_branch_rank(mol, ring: Sequence[int], Chem) -> int:
    ring_set = frozenset(ring)
    atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
    if not ring_set:
        return 1
    if any(atom.GetIsAromatic() or atom.GetSymbol() != "C" for atom in atoms):
        return 1
    if any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in atoms):
        return 1
    cache = mol_cache_dict(mol, "single_ring_external_relative_stereo_branch_rank")
    cache_key = tuple(sorted(ring_set))
    cached = cache.get(cache_key)
    if cached is not None:
        return int(cached)
    relative_rings = _relative_stereo_ring_branch_sets(mol, Chem)
    if not relative_rings:
        cache[cache_key] = 1
        return 1
    for relative_ring, tagged_atoms in relative_rings:
        if relative_ring & ring_set:
            continue
        if not tagged_atoms:
            continue
        if _ring_component_touches_parent_ring(mol, relative_ring, ring_set):
            cache[cache_key] = 0
            return 0
    cache[cache_key] = 1
    return 1


def _ring_component_touches_parent_ring(
    mol,
    relative_ring: frozenset[int],
    parent_ring: frozenset[int],
) -> bool:
    if not relative_ring or not parent_ring:
        return False
    ring_atoms = mol_cache_get(mol, "all_ring_atoms")
    if ring_atoms is None:
        ring_atoms = mol_cache_set(
            mol,
            "all_ring_atoms",
            set().union(*(set(ring) for ring in mol.GetRingInfo().AtomRings())),
        )
    seen = set(relative_ring)
    stack = [(atom_idx, 0) for atom_idx in relative_ring]
    while stack:
        atom_idx, depth = stack.pop()
        if depth > 4:
            continue
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in parent_ring:
                return True
            if nidx in seen or nbr.GetAtomicNum() <= 1:
                continue
            if nidx in ring_atoms:
                continue
            seen.add(nidx)
            stack.append((nidx, depth + 1))
    return False


def _ring_aromatic_atom_count(mol, ring: Sequence[int]) -> int:
    return _cached_ring_count(
        mol,
        "ring_aromatic_atom_count",
        ring,
        lambda: sum(1 for idx in ring if mol.GetAtomWithIdx(idx).GetIsAromatic()),
    )


def _ring_hetero_atom_count(mol, ring: Sequence[int]) -> int:
    return _cached_ring_count(
        mol,
        "ring_hetero_atom_count",
        ring,
        lambda: sum(1 for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6),
    )


def _cached_ring_count(mol, cache_name: str, ring: Sequence[int], compute) -> int:
    cache = mol_cache_dict(mol, cache_name)
    key = tuple(sorted(int(idx) for idx in ring))
    cached = cache.get(key)
    if cached is not None:
        return int(cached)
    value = int(compute())
    cache[key] = value
    return value


def _fused_parent_graph_fact_kwargs(mol, parent_atoms: Sequence[int]) -> dict[str, object]:
    parent_atom_tuple = tuple(parent_atoms)
    parent_atom_set = frozenset(parent_atom_tuple)
    return {
        "parent_atoms": parent_atom_set,
        "ring_system_atoms": parent_atom_set,
        "parent_aromatic_atom_count": _ring_aromatic_atom_count(mol, parent_atom_tuple),
        "parent_hetero_atom_count": _ring_hetero_atom_count(mol, parent_atom_tuple),
        "parent_terminal_hetero_substituent_count": _ring_terminal_hetero_substituent_count(
            mol,
            parent_atom_tuple,
        ),
        "parent_terminal_heterocarbon_substituent_count": _ring_terminal_heterocarbon_substituent_count(
            mol,
            parent_atom_tuple,
        ),
        "parent_terminal_chalcogen_double_bond_count": _ring_terminal_chalcogen_double_bond_count(
            mol,
            parent_atom_tuple,
        ),
        "parent_substituted_hetero_atom_count": _parent_substituted_hetero_atom_count(
            mol,
            parent_atom_tuple,
        ),
        "parent_external_nitrogen_substituent_count": _ring_external_nitrogen_substituent_count(
            mol,
            parent_atom_tuple,
        ),
        "parent_external_acyl_nitrogen_substituent_count": _ring_external_acyl_nitrogen_substituent_count(
            mol,
            parent_atom_tuple,
        ),
        "parent_external_oxygen_substituent_count": _ring_external_oxygen_substituent_count(
            mol,
            parent_atom_tuple,
        ),
    }


_PURIN_6_AMINE_SOURCE_FAMILY = "fused-retained:purin-6-amine"
_PURIN_THIOXO_SOURCE_FAMILY = "fused-retained:purin-thioxo"


def _retained_fused_source_family(
    mol,
    template: dict,
    match: Sequence[int],
    *,
    default: str | None = None,
) -> str:
    if _has_purin_6_amine_parent_shape(mol, template, match):
        return _PURIN_6_AMINE_SOURCE_FAMILY
    if _has_purin_thioxo_parent_shape(mol, template, match):
        return _PURIN_THIOXO_SOURCE_FAMILY
    return str(template.get("source_family", "") if default is None else default)


def _biochemical_unrendered_parent_charge_penalty(
    mol,
    template: dict,
    match: Sequence[int],
) -> int:
    source_family = str(template.get("source_family", ""))
    if "biochemical" not in source_family:
        return 0
    charge = _polycyclic_parent_charge_evidence(mol, match)
    if charge is None or charge.rendered_terms:
        return 0
    charge_abs_sum = sum(abs(mol.GetAtomWithIdx(idx).GetFormalCharge()) for idx in charge.charged_atoms)
    template_mol = template_record_mol(template)
    if (
        _formal_charge(template_mol) == charge.formal_charge
        and _abs_formal_charge_sum(template_mol) >= charge_abs_sum
    ):
        return 0
    return 1


def _has_purin_6_amine_parent_shape(mol, template: dict, match: Sequence[int]) -> bool:
    if template_record_name(template) != "purin":
        return False
    locants = template_record_locants(template)
    core = set(match)
    reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
    if "9" not in _matched_fused_indicated_hydrogen_locants(mol, template, reverse):
        return False
    has_locant_6_amino = False
    for template_idx, mol_idx in enumerate(match):
        locant = locants.get(template_idx, "")
        atom = mol.GetAtomWithIdx(int(mol_idx))
        external_neighbors = [
            nbr
            for nbr in atom.GetNeighbors()
            if nbr.GetIdx() not in core and nbr.GetAtomicNum() > 1
        ]
        if locant in {"1", "3", "7", "9"} and external_neighbors:
            return False
        if locant != "6":
            continue
        for neighbor in external_neighbors:
            bond = mol.GetBondBetweenAtoms(int(mol_idx), neighbor.GetIdx())
            if (
                neighbor.GetAtomicNum() == 7
                and not neighbor.IsInRing()
                and bond is not None
                and bond.GetBondTypeAsDouble() == 1.0
            ):
                has_locant_6_amino = True
    return has_locant_6_amino


def _has_purin_thioxo_parent_shape(mol, template: dict, match: Sequence[int]) -> bool:
    if template_record_name(template) != "purin":
        return False
    thioxo_locs = set(_purin_terminal_thioxo_locants(mol, template, match))
    if thioxo_locs not in ({"2", "6"}, {"6", "8"}, {"2", "6", "8"}):
        return False
    locants = template_record_locants(template)
    core = set(match)
    for template_idx, mol_idx in enumerate(match):
        locant = locants.get(template_idx, "")
        if locant not in {"1", "3", "7", "9"}:
            continue
        atom = mol.GetAtomWithIdx(int(mol_idx))
        if any(nbr.GetIdx() not in core and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
            return False
    return True


def _purin_terminal_thioxo_locants(mol, template: dict, match: Sequence[int]) -> tuple[str, ...]:
    locants = template_record_locants(template)
    core = set(match)
    thioxo_locs: list[str] = []
    for template_idx, mol_idx in enumerate(match):
        locant = locants.get(template_idx, "")
        if not locant:
            continue
        atom = mol.GetAtomWithIdx(int(mol_idx))
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx in core or neighbor.GetAtomicNum() != 16 or neighbor.GetDegree() != 1:
                continue
            bond = mol.GetBondBetweenAtoms(int(mol_idx), nidx)
            if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
                thioxo_locs.append(locant)
                break
    return tuple(sorted(set(thioxo_locs), key=int))


def _ring_terminal_hetero_substituent_count(mol, ring: Sequence[int]) -> int:
    return _cached_ring_count(
        mol,
        "ring_terminal_hetero_substituent_count",
        ring,
        lambda: _ring_terminal_hetero_substituent_count_uncached(mol, ring),
    )


def _ring_terminal_hetero_substituent_count_uncached(mol, ring: Sequence[int]) -> int:
    ring_set = set(ring)
    count = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in ring_set or neighbor.GetAtomicNum() == 6:
                continue
            if neighbor.GetDegree() == 1:
                count += 1
    return count


def _ring_terminal_heterocarbon_substituent_count(mol, ring: Sequence[int]) -> int:
    return _cached_ring_count(
        mol,
        "ring_terminal_heterocarbon_substituent_count",
        ring,
        lambda: _ring_terminal_heterocarbon_substituent_count_uncached(mol, ring),
    )


def _ring_terminal_heterocarbon_substituent_count_uncached(mol, ring: Sequence[int]) -> int:
    ring_set = set(ring)
    count = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for carbon in atom.GetNeighbors():
            carbon_idx = carbon.GetIdx()
            if carbon_idx in ring_set or carbon.GetAtomicNum() != 6:
                continue
            for hetero in carbon.GetNeighbors():
                hetero_idx = hetero.GetIdx()
                if hetero_idx == idx or hetero_idx in ring_set:
                    continue
                if hetero.GetAtomicNum() in {1, 6} or hetero.GetDegree() != 1:
                    continue
                bond = mol.GetBondBetweenAtoms(carbon_idx, hetero_idx)
                if bond is not None and bond.GetBondTypeAsDouble() > 1.1:
                    count += 1
                    break
    return count


def _ring_terminal_carbonyl_substituent_count(mol, ring: Sequence[int]) -> int:
    return _cached_ring_count(
        mol,
        "ring_terminal_carbonyl_substituent_count",
        ring,
        lambda: _ring_terminal_carbonyl_substituent_count_uncached(mol, ring),
    )


def _ring_terminal_carbonyl_substituent_count_uncached(mol, ring: Sequence[int]) -> int:
    ring_set = set(ring)
    count = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for carbon in atom.GetNeighbors():
            carbon_idx = carbon.GetIdx()
            if carbon_idx in ring_set or carbon.GetAtomicNum() != 6:
                continue
            for hetero in carbon.GetNeighbors():
                hetero_idx = hetero.GetIdx()
                if hetero_idx == idx or hetero_idx in ring_set:
                    continue
                if hetero.GetAtomicNum() not in {8, 16} or hetero.GetDegree() != 1:
                    continue
                bond = mol.GetBondBetweenAtoms(carbon_idx, hetero_idx)
                if bond is not None and bond.GetBondTypeAsDouble() > 1.5:
                    count += 1
                    break
    return count


def _ring_terminal_chalcogen_double_bond_count(mol, ring: Sequence[int]) -> int:
    return _cached_ring_count(
        mol,
        "ring_terminal_chalcogen_double_bond_count",
        ring,
        lambda: _ring_terminal_chalcogen_double_bond_count_uncached(mol, ring),
    )


def _ring_terminal_chalcogen_double_bond_count_uncached(mol, ring: Sequence[int]) -> int:
    ring_set = set(ring)
    count = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in ring_set:
                continue
            if neighbor.GetAtomicNum() not in {8, 16, 34, 52}:
                continue
            if neighbor.GetDegree() != 1:
                continue
            bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
            if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
                count += 1
    return count


def _parent_substituted_hetero_atom_count(mol, parent_atoms: Sequence[int]) -> int:
    return _cached_ring_count(
        mol,
        "parent_substituted_hetero_atom_count",
        parent_atoms,
        lambda: _parent_substituted_hetero_atom_count_uncached(mol, parent_atoms),
    )


def _parent_substituted_hetero_atom_count_uncached(mol, parent_atoms: Sequence[int]) -> int:
    parent_set = set(parent_atoms)
    count = 0
    for idx in parent_set:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 6:
            continue
        if any(
            neighbor.GetIdx() not in parent_set and neighbor.GetAtomicNum() > 1
            for neighbor in atom.GetNeighbors()
        ):
            count += 1
    return count


def _ring_external_nitrogen_substituent_count(mol, ring: Sequence[int]) -> int:
    return _cached_ring_count(
        mol,
        "ring_external_nitrogen_substituent_count",
        ring,
        lambda: _ring_external_nitrogen_substituent_count_uncached(mol, ring),
    )


def _ring_external_nitrogen_substituent_count_uncached(mol, ring: Sequence[int]) -> int:
    ring_set = set(ring)
    count = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in ring_set and neighbor.GetAtomicNum() == 7:
                count += 1
    return count


def _ring_external_acyl_nitrogen_substituent_count(mol, ring: Sequence[int]) -> int:
    return _cached_ring_count(
        mol,
        "ring_external_acyl_nitrogen_substituent_count",
        ring,
        lambda: _ring_external_acyl_nitrogen_substituent_count_uncached(mol, ring),
    )


def _ring_external_acyl_nitrogen_substituent_count_uncached(mol, ring: Sequence[int]) -> int:
    ring_set = set(ring)
    count = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for nitrogen in atom.GetNeighbors():
            if nitrogen.GetIdx() in ring_set or nitrogen.GetAtomicNum() != 7:
                continue
            if _external_nitrogen_has_acyl_neighbor(mol, nitrogen.GetIdx(), ring_set):
                count += 1
    return count


def _external_nitrogen_has_acyl_neighbor(mol, nitrogen_idx: int, parent_atoms: set[int]) -> bool:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    for carbon in nitrogen.GetNeighbors():
        carbon_idx = carbon.GetIdx()
        if carbon_idx in parent_atoms or carbon.GetAtomicNum() != 6:
            continue
        for hetero in carbon.GetNeighbors():
            hetero_idx = hetero.GetIdx()
            if hetero_idx == nitrogen_idx:
                continue
            if hetero.GetAtomicNum() not in {8, 16, 34, 52}:
                continue
            bond = mol.GetBondBetweenAtoms(carbon_idx, hetero_idx)
            if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
                return True
    return False


def _ring_external_oxygen_substituent_count(mol, ring: Sequence[int]) -> int:
    return _cached_ring_count(
        mol,
        "ring_external_oxygen_substituent_count",
        ring,
        lambda: _ring_external_oxygen_substituent_count_uncached(mol, ring),
    )


def _ring_external_oxygen_substituent_count_uncached(mol, ring: Sequence[int]) -> int:
    ring_set = set(ring)
    count = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in ring_set or neighbor.GetAtomicNum() != 8:
                continue
            if neighbor.GetDegree() > 1:
                count += 1
    return count


def _single_ring_charge_evidence(mol, ring: Sequence[int], name: str) -> ChargeEvidence | None:
    charged_atoms = tuple(
        idx
        for idx in sorted(ring)
        if mol.GetAtomWithIdx(idx).GetFormalCharge()
    )
    if not charged_atoms:
        return None
    if _single_ring_name_already_represents_charge(mol, charged_atoms, name):
        return None
    formal_charge = sum(mol.GetAtomWithIdx(idx).GetFormalCharge() for idx in charged_atoms)
    return ChargeEvidence(
        formal_charge=formal_charge,
        charged_atoms=charged_atoms,
    )


def _single_ring_name_already_represents_charge(mol, charged_atoms: Sequence[int], name: str) -> bool:
    if len(charged_atoms) != 1:
        return False
    atom = mol.GetAtomWithIdx(charged_atoms[0])
    if atom.GetFormalCharge() <= 0:
        return False
    if atom.GetSymbol() == "O" and "pyrylium" in name:
        return True
    if atom.GetSymbol() == "S" and re.search(r"(?:thiopyrylium|thiopyranium|thiophenium)", name):
        return True
    return False


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
        for template, match in _cached_iter_fused_template_matches(
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
            if name and _should_defer_spiro_context_parent_to_senior_azolone(ctx, match):
                continue
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
                    charge=_polycyclic_parent_charge_evidence(mol, match),
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
    partial_hydro_render_set_size = _GENERATED_PARTIAL_HYDRO_RENDER_SET_SIZE
    partial_hydro_candidate_limit = 16
    partial_hydro_output_max_length: int | None = None
    reject_extra_core_bonds = False

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        if _too_large_for_exhaustive_ring_parent_search(mol):
            return
        if _high_rank_hetero_retained_fused_search_unpromising(mol):
            return
        facts = []
        for template, match in _cached_iter_fused_template_matches(
            ctx,
            self._template_iter(ctx),
            self._Chem,
            topology_query=True,
            require_one_ring_system=True,
            reject_extra_core_bonds=self.reject_extra_core_bonds,
        ):
            for parent_name, hydro_key in _partial_hydro_parent_names(mol, template, match):
                fusion_key = _partial_hydro_fusion_preference(template_record_name(template))
                partial_hydro_key = self._partial_hydro_key(template, parent_name, hydro_key)
                facts.append(
                    PartialHydroCandidateFacts(
                        template=template,
                        match=tuple(match),
                        parent_name=parent_name,
                        hydro_key=partial_hydro_key,
                        structural_key=(fusion_key, partial_hydro_key, -len(match), parent_name, tuple(match)),
                    )
                )
        if ctx.telemetry is not None:
            ctx.telemetry.candidate_facts_total[self.family] += len(facts)
        render_set = partial_hydro_render_set(tuple(facts), size=self.partial_hydro_render_set_size)
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
            if _should_defer_spiro_context_parent_to_senior_azolone(ctx, match):
                continue
            if self.partial_hydro_output_max_length is not None and len(name) > self.partial_hydro_output_max_length:
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
                    charge=_polycyclic_parent_charge_evidence(mol, match),
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
            if yielded >= self.partial_hydro_candidate_limit:
                return

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.generated_fused_templates

    def _partial_hydro_key(self, template: dict, parent_name: str, hydro_key: tuple) -> tuple:
        return hydro_key


class PartialHydroFusedParentRule(GeneratedPartialHydroFusedParentRule):
    family = "partial-hydro-fused-retained-parent"
    priority = parent_priority("retained-partial-hydro-fallback")
    partial_hydro_render_set_size = _RETAINED_PARTIAL_HYDRO_RENDER_SET_SIZE
    partial_hydro_candidate_limit = 16
    partial_hydro_output_max_length = 100
    reject_extra_core_bonds = True

    def candidates(self, ctx):
        if not ctx.mol.HasProp(_RETAINED_PARTIAL_HYDRO_FALLBACK_PROP):
            return
        yield from super().candidates(ctx)

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.candidate_templates(_curated_fused_templates(self._Chem))

    def _partial_hydro_key(self, template: dict, parent_name: str, hydro_key: tuple) -> tuple:
        return (retained_partial_hydro_template_saturation_key(template, parent_name), hydro_key)


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
        for template, match in _cached_iter_fused_template_matches(
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
                if _should_defer_spiro_core_parent_to_senior_azolone(mol, match):
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
        for candidate in component_spiro_parent_candidates(ctx, self.priority):
            if (
                candidate.render_rank != 0
                and _should_defer_spiro_core_parent_to_senior_azolone(ctx.mol, candidate.parent_atoms)
            ):
                continue
            yield candidate


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
        for template, match in _cached_iter_fused_template_matches(
            ctx,
            ctx.candidate_templates(self._templates()),
            self._Chem,
            topology_query=True,
            reject_extra_core_bonds=True,
        ):
            named = _name_natural_product_topology_match(mol, self._Chem, template, match)
            if not named:
                continue
            name, _stereo_prefix = named
            key = (name, tuple(match))
            if key in seen:
                continue
            seen.add(key)
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=self.priority,
                sort_key=self._candidate_sort_key(template, match, name),
                parent_atoms=frozenset(match),
                stereo_locants=_template_match_stereo_locants(template, match),
                template_id=id(template),
                template_name=template_record_name(template),
                source_family=str(template.get("source_family", "")),
                source_file=str(template.get("source_file", "")),
                estimated_length=len(name),
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
                    bond_order_mismatches = _skeletal_replacement_bond_order_mismatch_count(
                        mol,
                        template,
                        match,
                    )
                    alignment_key = (
                        bond_order_mismatches,
                        _skeletal_replacement_saturation_penalty(mol, match, parent_name),
                        _skeletal_replacement_unsaturated_hetero_bond_penalty(mol, template, match),
                        _skeletal_replacement_spiro_locant_instability_penalty(template),
                    )
                    stage_counts["facts"] += 1
                    facts.append(
                        _SkeletalReplacementCandidateFacts(
                            template=template,
                            match=tuple(match),
                            parent_name=parent_name,
                            bond_order_mismatches=bond_order_mismatches,
                            alignment_key=alignment_key,
                            structural_key=(
                                alignment_key,
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
            if _should_defer_spiro_context_parent_to_senior_azolone(ctx, fact.match):
                continue
            safety = _skeletal_replacement_charge_safety(mol, fact)
            yielded_names.add(name)
            yielded += 1
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=self.priority,
                sort_key=(
                    fact.alignment_key,
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
                charge=_polycyclic_parent_charge_evidence(mol, fact.match),
                safety=safety,
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
    bond_order_mismatches: int
    alignment_key: tuple[int, ...]
    structural_key: tuple


def _skeletal_replacement_charge_safety(
    mol,
    fact: "_SkeletalReplacementCandidateFacts",
) -> CandidateSafety | None:
    if not _skeletal_replacement_fact_has_positive_replacement_charge(mol, fact):
        return None
    if fact.bond_order_mismatches <= 0 and (len(fact.alignment_key) < 2 or fact.alignment_key[1] <= 0):
        return None
    return CandidateSafety(
        opsin_fragility_reasons=(OPSIN_FRAGILE_CHARGED_REPLACEMENT_UNSATURATION,)
    )


def _skeletal_replacement_fact_has_positive_replacement_charge(
    mol,
    fact: "_SkeletalReplacementCandidateFacts",
) -> bool:
    template_mol = template_record_mol(fact.template)
    for template_idx, mol_idx in enumerate(fact.match):
        template_atom = template_mol.GetAtomWithIdx(template_idx)
        atom = mol.GetAtomWithIdx(mol_idx)
        if (
            template_atom.GetSymbol() == "C"
            and template_atom.GetFormalCharge() == 0
            and atom.GetSymbol() in {"N", "O", "S"}
            and atom.GetFormalCharge() > 0
        ):
            return True
    return False


def _skeletal_replacement_render_set(
    facts: tuple[_SkeletalReplacementCandidateFacts, ...],
) -> tuple[_SkeletalReplacementCandidateFacts, ...]:
    facts = _skeletal_replacement_best_bond_order_facts(facts)
    if len(facts) <= _SKELETAL_REPLACEMENT_RENDER_SET_SIZE:
        return facts
    return tuple(sorted(facts, key=lambda fact: fact.structural_key)[:_SKELETAL_REPLACEMENT_RENDER_SET_SIZE])


def _skeletal_replacement_best_bond_order_facts(
    facts: tuple[_SkeletalReplacementCandidateFacts, ...],
) -> tuple[_SkeletalReplacementCandidateFacts, ...]:
    if not facts:
        return facts
    min_by_core: dict[frozenset[int], tuple[int, ...]] = {}
    for fact in facts:
        core = frozenset(fact.match)
        current = min_by_core.get(core)
        if current is None or fact.alignment_key < current:
            min_by_core[core] = fact.alignment_key
    if all(fact.alignment_key == min_by_core[frozenset(fact.match)] for fact in facts):
        return facts
    return tuple(
        fact
        for fact in facts
        if fact.alignment_key == min_by_core[frozenset(fact.match)]
    )


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
                template_has_nh = _retained_topology_replacement_template_has_nh(template)
                allow_nh_template = False
                if template_has_nh:
                    allow_nh_template = _retained_topology_replacement_template_allows_nh(template)
                    if not allow_nh_template:
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
                    parent_name = _retained_topology_replacement_parent_name(
                        mol,
                        template,
                        match,
                        allow_nh_template=allow_nh_template,
                    )
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
            if _should_defer_spiro_context_parent_to_senior_azolone(ctx, fact.match):
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
        for template, match in _cached_iter_fused_template_matches(
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
                    charge=_polycyclic_parent_charge_evidence(mol, match),
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
        for template, match in _cached_iter_fused_template_matches(
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
                template_name = template_record_name(template)
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=(
                        _generated_oxido_fused_template_preference(template_name),
                        _generated_fusion_preference(template_name),
                        _substituent_locant_tuple(name),
                        -len(match),
                        name,
                    ),
                )


def _generated_oxido_fused_template_preference(template_name: str) -> int:
    if re.match(r"^(?:pyrido|pyridino)\[[^\]]+\]imidazole$", template_name):
        return 0
    if re.match(r"^imidazo\[[^\]]+\]pyridine$", template_name):
        return 2
    return 1


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
        rendered = 0
        max_records = _max_polycyclic_records_for_family(self.family, ctx)
        for record_order, record in enumerate(ctx.polycyclic_parent_records(self.record_family)):
            name, seq = record[0], record[1]
            bridge_unsaturation_mismatches = int(record[2]) if len(record) > 2 else 0
            bridge_topology_mismatches = int(record[3]) if len(record) > 3 else 0
            if not self._accept_record(name, seq):
                continue
            if self._defer_record(ctx, name, seq):
                continue
            if max_records is not None and rendered >= max_records:
                return
            safety = None
            render_cost = 1
            fragility_reasons = []
            if bridge_unsaturation_mismatches > 0:
                fragility_reasons.append(OPSIN_FRAGILE_POLYCYCLIC_BRIDGE_UNSATURATION)
                render_cost += bridge_unsaturation_mismatches
            if bridge_topology_mismatches > 0:
                fragility_reasons.append(OPSIN_FRAGILE_POLYCYCLIC_BRIDGE_TOPOLOGY)
                render_cost += bridge_topology_mismatches
            if fragility_reasons:
                safety = CandidateSafety(opsin_fragility_reasons=tuple(fragility_reasons))
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=priority,
                sort_key=(record_order, len(name), name),
                stereo_prefix=_polycyclic_stereo_prefix(mol, seq, self._Chem, name=name),
                parent_atoms=frozenset(seq),
                locants=_external_locants_from_sequence(mol, seq),
                stereo_locants=_sequence_stereo_locants(seq),
                charge=_polycyclic_parent_charge_evidence(mol, seq),
                source_family=self.record_family,
                safety=safety,
                estimated_length=len(name),
                render_cost=render_cost,
                render_rank=record_order + 1,
            )
            rendered += 1


class GenericBicyclicParentRule(_PolycyclicRecordParentRule):
    family = "generic-bicyclic-parent"
    priority = parent_priority("generic-bicyclic-parent")
    record_family = "bicyclic"
    requires_generic_bicyclic_topology = True


_MAX_POLYCYCLIC_RECORDS_BY_FAMILY = {
    "functional-generic-bicyclic-parent": 20,
    "generic-bicyclic-parent": 20,
    "kekulized-generic-polycyclic-parent": 12,
}
_FAST_ROUTE_POLYCYCLIC_RECORDS_BY_FAMILY = {
    "functional-generic-bicyclic-parent": 1,
    "generic-bicyclic-parent": 1,
    "kekulized-generic-polycyclic-parent": 1,
}


def _max_polycyclic_records_for_family(family: str, ctx=None) -> int | None:
    if (
        ctx is not None
        and bool(getattr(ctx.mol, "HasProp", lambda _name: False)(FAST_POLY_BRIDGE_ROUTE_PROP))
    ):
        return _FAST_ROUTE_POLYCYCLIC_RECORDS_BY_FAMILY.get(family)
    return _MAX_POLYCYCLIC_RECORDS_BY_FAMILY.get(family)


def _polycyclic_parent_charge_evidence(mol, seq: Sequence[int]) -> ChargeEvidence | None:
    charged_atoms = tuple(
        int(idx)
        for idx in seq
        if mol.GetAtomWithIdx(int(idx)).GetFormalCharge() != 0
    )
    if not charged_atoms:
        return None
    formal_charge = sum(mol.GetAtomWithIdx(idx).GetFormalCharge() for idx in charged_atoms)
    return ChargeEvidence(
        formal_charge=formal_charge,
        charged_atoms=charged_atoms,
        rendered_terms=(),
    )


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
        if not _has_retained_fused_parent_substituent_name(name):
            return False
        return bool(ctx.polycyclic_parent_records("kekulized"))


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

    def _defer_record(self, ctx, name: str, seq: Sequence[int]) -> bool:
        return _should_defer_spiro_core_parent_to_senior_azolone(ctx.mol, seq)


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

    def _defer_all_records(self, ctx) -> bool:
        if bool(getattr(ctx.mol, "HasProp", lambda _name: False)(FAST_POLY_RETAINED_DEFER_BYPASS_PROP)):
            return False
        if bool(getattr(ctx.mol, "HasProp", lambda _name: False)(FAST_POLY_BRIDGE_ROUTE_PROP)):
            if _has_fast_poly_retained_defer_candidate(ctx, self._Chem):
                mol_cache_set(ctx.mol, FAST_POLY_RETAINED_DEFER_CACHE_KEY, True)
                return True
            return False
        return _has_retained_fused_parent_candidate(ctx, self._Chem)


def _has_fast_poly_retained_defer_candidate(ctx, Chem) -> bool:
    mol = ctx.mol
    cached = mol_cache_get(mol, "fast_poly_retained_defer_candidate")
    if cached is not None:
        return cached
    templates = ctx.candidate_templates(_fast_poly_retained_defer_templates(Chem))
    for template, match in _iter_fused_template_matches(
        ctx,
        templates,
        Chem,
        reject_extra_core_bonds=True,
    ):
        if _fast_poly_retained_defer_template_is_fragile(template):
            continue
        return mol_cache_set(mol, "fast_poly_retained_defer_candidate", True)
    if _should_try_fast_poly_generated_defer_candidate(ctx):
        generated_templates = ctx.narrow_generated_fused_templates
        for template, match in _iter_fused_template_matches(
            ctx,
            generated_templates,
            Chem,
            reject_extra_core_bonds=True,
        ):
            if _fast_poly_retained_defer_template_is_fragile(template):
                continue
            return mol_cache_set(mol, "fast_poly_retained_defer_candidate", True)
    return mol_cache_set(mol, "fast_poly_retained_defer_candidate", False)


_FAST_POLY_RETAINED_DEFER_SOURCE_FAMILIES = frozenset(
    {
        "",
        "arylGroups:ring:biochemical",
        "arylGroups:ring:ring",
        "carbohydrates:carbohydrate:ring",
        "naturalProducts:ring:biochemical",
        "opsin-retained-ring-cml",
        "opsin-test-scaffold",
        "static-retained",
    }
)


@lru_cache(maxsize=8)
def _fast_poly_retained_defer_templates(Chem) -> tuple[dict, ...]:
    return tuple(
        template
        for template in _curated_fused_templates(Chem)
        if str(template.get("source_family", "")) in _FAST_POLY_RETAINED_DEFER_SOURCE_FAMILIES
    )


def _should_try_fast_poly_generated_defer_candidate(ctx) -> bool:
    facts = ctx.facts
    if facts.ring_count > 6:
        return False
    if facts.aromatic_atom_count < 12:
        return False
    return facts.max_ring_system_ring_count >= 2


def _fast_poly_retained_defer_template_is_fragile(template: dict) -> bool:
    return (
        str(template.get("source_family", "")),
        template_record_name(template),
    ) in FAST_POLY_RETAINED_DEFER_FRAGILE_TEMPLATE_KEYS


def _has_fast_poly_retained_scaffold_defer_candidate(ctx, Chem) -> bool:
    mol = ctx.mol
    cached = mol_cache_get(mol, "fast_poly_retained_scaffold_defer_candidate")
    if cached is not None:
        return cached
    if ctx.features.get("max_ring_system_ring_count", 0) < 2:
        return mol_cache_set(mol, "fast_poly_retained_scaffold_defer_candidate", False)
    if _too_large_for_exhaustive_ring_parent_search(mol):
        return mol_cache_set(mol, "fast_poly_retained_scaffold_defer_candidate", False)

    molecule_aromatic_atom_count = _aromatic_atom_count(mol)
    molecule_charged_atom_values = _charged_atom_values(mol)
    has_stereo_evidence = _has_stereo_evidence(mol, Chem)
    selected_facts: list[tuple[tuple, object]] = []
    trimmed = False
    for compiled in ctx.candidate_compiled_templates(_general_retained_ring_scaffold_templates(Chem)):
        if not _compiled_retained_ring_scaffold_template_can_be_general_parent(mol, compiled, Chem):
            continue
        if has_stereo_evidence and not _fast_poly_retained_scaffold_template_is_stable(compiled.record.name):
            continue
        if compiled.features.atom_count >= mol.GetNumAtoms():
            continue
        if compiled.features.aromatic_atom_count > molecule_aromatic_atom_count:
            continue
        if (
            compiled.features.charged_atom_values
            and not all(value in molecule_charged_atom_values for value in compiled.features.charged_atom_values)
        ):
            continue
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

    for _structural_key, compiled in selected_facts:
        template = compiled.runtime
        for match in ctx.template_matches(template):
            if _match_has_extra_core_bonds(mol, template, match):
                continue
            if _match_has_unlocanted_external_substitution(mol, template, match):
                continue
            name = _name_fused_core_match(mol, template, match)
            if name and not _unsafe_retained_ring_scaffold_candidate(name):
                return mol_cache_set(mol, "fast_poly_retained_scaffold_defer_candidate", True)
    return mol_cache_set(mol, "fast_poly_retained_scaffold_defer_candidate", False)


def _fast_poly_retained_scaffold_template_is_stable(template_name: str) -> bool:
    return template_name in _FAST_POLY_RETAINED_STABLE_SCAFFOLD_TEMPLATE_NAMES


_FAST_POLY_RETAINED_STABLE_SCAFFOLD_TEMPLATE_NAMES = frozenset(
    {
        "pentacene",
        "hexacene",
        "heptacene",
        "octacene",
        "nonacene",
        "decacene",
        "undecacene",
        "dodecacene",
        "pentaphene",
        "hexaphene",
        "heptaphene",
        "octaphene",
        "nonaphene",
        "decaphene",
        "undecaphene",
        "dodecaphene",
    }
)


def _has_retained_fused_parent_candidate(ctx, Chem) -> bool:
    mol = ctx.mol
    cached = mol_cache_get(mol, "has_retained_fused_parent_candidate")
    if cached is not None:
        return cached
    for templates in _iter_retained_fused_parent_template_sets(ctx, Chem):
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


def _iter_retained_fused_parent_template_sets(ctx, Chem, *, generated_templates=None):
    yield ctx.candidate_templates(_curated_fused_templates(Chem))
    yield ctx.generated_fused_templates if generated_templates is None else generated_templates


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
        name = _exact_retained_parent_candidate_name(template)
        return (len(name), template.get("source_family", ""), name)

    def _candidate_name(self, mol, template: dict, match: Sequence[int]) -> str:
        reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
        name = _adjust_fused_parent_for_indicated_hydrogen(mol, template, reverse)
        if name == template_record_name(template):
            return _exact_retained_parent_candidate_name(template)
        if template.get("source_file") == "naturalProducts.xml":
            name = _terminal_retained_parent_name(name)
        return _opsin_parseable_retained_parent_name(name)

    def _parent_indicated_hydrogen_locants(
        self,
        mol,
        template: dict,
        match: Sequence[int],
    ) -> tuple[str, ...]:
        reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
        return tuple(_matched_fused_indicated_hydrogen_locants(mol, template, reverse))

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
                    if not _exact_retained_fused_match_is_hydrogen_compatible(mol, template, match):
                        continue
                    family = (
                        "exact-biochemical-retained-parent"
                        if template.get("source_family") == "static-exact-retained"
                        else self.family
                    )
                    name = self._candidate_name(mol, template, match)
                    yield NameCandidate(
                        name=name,
                        family=family,
                        priority=self.priority,
                        sort_key=self._sort_key(template),
                        stereo_prefix=_exact_retained_parent_stereo_prefix(mol, self._Chem, template, match),
                        parent_atoms=frozenset(match),
                        stereo_locants=_template_match_stereo_locants(template, match),
                        template_id=id(template),
                        template_name=template_record_name(template),
                        source_family=str(template.get("source_family", "")),
                        source_file=str(template.get("source_file", "")),
                        parent_indicated_hydrogen_locants=self._parent_indicated_hydrogen_locants(
                            mol,
                            template,
                            match,
                        ),
                        template_formal_charge_atom_count=_formal_charge_atom_count(template_record_mol(template)),
                        template_abs_formal_charge_sum=_abs_formal_charge_sum(template_record_mol(template)),
                        estimated_length=len(name),
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
                    if not _exact_retained_fused_match_is_hydrogen_compatible(mol, template, match):
                        continue
                    family = (
                        "exact-biochemical-retained-parent"
                        if template.get("source_family") == "static-exact-retained"
                        else self.family
                    )
                    name = self._candidate_name(mol, template, match)
                    yield NameCandidate(
                        name=name,
                        family=family,
                        priority=self.priority,
                        sort_key=self._sort_key(template),
                        stereo_prefix=_exact_retained_parent_stereo_prefix(mol, self._Chem, template, match),
                        parent_atoms=frozenset(match),
                        stereo_locants=_template_match_stereo_locants(template, match),
                        template_id=id(template),
                        template_name=template_record_name(template),
                        source_family=str(template.get("source_family", "")),
                        source_file=str(template.get("source_file", "")),
                        parent_indicated_hydrogen_locants=self._parent_indicated_hydrogen_locants(
                            mol,
                            template,
                            match,
                        ),
                        template_formal_charge_atom_count=_formal_charge_atom_count(template_record_mol(template)),
                        template_abs_formal_charge_sum=_abs_formal_charge_sum(template_record_mol(template)),
                        estimated_length=len(name),
                    )


class GeneratedExactParentRule(ExactRetainedParentRule):
    family = "generated-exact-parent"
    priority = parent_priority("generated-parent")

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.generated_fused_templates

    def _sort_key(self, template: dict) -> tuple:
        return _generated_exact_parent_sort_key(template)

    def _candidate_name(self, mol, template: dict, match: Sequence[int]) -> str:
        reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
        name = _terminal_retained_parent_name(_adjust_fused_parent_for_charge(mol, template, reverse))
        name = _template_isotope_prefixed_parent_name(mol, template, match, name)
        return _opsin_parseable_retained_parent_name(name)

    def _parent_indicated_hydrogen_locants(
        self,
        mol,
        template: dict,
        match: Sequence[int],
    ) -> tuple[str, ...]:
        reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
        return tuple(_matched_fused_indicated_hydrogen_locants(mol, template, reverse))


class ExactBiochemicalRetainedParentRule(ExactFunctionalRetainedParentRule):
    family = "exact-biochemical-retained-parent"

    def _template_iter(self, ctx) -> Sequence[dict]:
        return tuple(
            template
            for template in ctx.candidate_templates(self._templates(), exact_atom_count=True)
            if template.get("source_family") == "static-exact-retained"
        )


class ExactRetainedNOxideParentRule(ExactRetainedParentRule):
    family = "exact-retained-n-oxide-parent"
    priority = parent_priority("senior-functional-exception")

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.candidate_templates(self._templates())

    def candidates(self, ctx):
        mol = ctx.mol
        site = _single_terminal_n_oxide_site(mol, self._Chem)
        if site is None:
            return
        nitrogen_idx, oxide_idx = site
        if mol.GetNumAtoms() > 65:
            return

        core_atoms = set(range(mol.GetNumAtoms()))
        core_atoms.remove(oxide_idx)
        template_atom_count = len(core_atoms)
        atom_set = frozenset(range(mol.GetNumAtoms()))
        has_stereo_evidence = _has_stereo_evidence(mol, self._Chem)
        for template in self._template_iter(ctx):
            template_mol = template_record_mol(template)
            if template_mol.GetNumAtoms() != template_atom_count:
                continue
            if not any(atom.GetSymbol() == "C" for atom in template_mol.GetAtoms()):
                continue
            if _formal_charge(template_mol) != _formal_charge(mol):
                continue
            if template.get("stereochemical") and not has_stereo_evidence:
                continue
            for match in ctx.template_matches(template):
                if set(match) != core_atoms or nitrogen_idx not in match:
                    continue
                if _match_has_extra_core_bonds(mol, template, match):
                    continue
                if not _exact_retained_fused_match_is_hydrogen_compatible(mol, template, match):
                    continue
                name = _exact_retained_n_oxide_candidate_name(template, match, nitrogen_idx)
                if not name:
                    continue
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=(len(name), template.get("source_family", ""), name),
                    stereo_prefix=_exact_retained_parent_stereo_prefix(mol, self._Chem, template, match),
                    parent_atoms=atom_set,
                    stereo_locants=_template_match_stereo_locants(template, match),
                    template_id=id(template),
                    template_name=template_record_name(template),
                    source_family=str(template.get("source_family", "")),
                    source_file=str(template.get("source_file", "")),
                    charge=ChargeEvidence(
                        formal_charge=1,
                        charged_atoms=(nitrogen_idx, oxide_idx),
                        rendered_terms=("oxide",),
                    ),
                    template_formal_charge_atom_count=_formal_charge_atom_count(template_mol),
                    template_abs_formal_charge_sum=_abs_formal_charge_sum(template_mol),
                    estimated_length=len(name),
                )


def _single_terminal_n_oxide_site(mol, Chem) -> tuple[int, int] | None:
    sites = []
    for nitrogen in mol.GetAtoms():
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() <= 0:
            continue
        nidx = nitrogen.GetIdx()
        terminal_oxides = []
        for nbr in nitrogen.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(nidx, nbr.GetIdx())
            if (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() < 0
                and bond is not None
                and bond.GetBondType() == Chem.BondType.SINGLE
                and len([other for other in nbr.GetNeighbors() if other.GetAtomicNum() > 1]) == 1
            ):
                terminal_oxides.append(nbr.GetIdx())
        if len(terminal_oxides) == 1:
            sites.append((nidx, terminal_oxides[0]))
    return sites[0] if len(sites) == 1 else None


def _exact_retained_n_oxide_candidate_name(template: dict, match: Sequence[int], nitrogen_idx: int) -> str:
    base = _exact_retained_parent_candidate_name(template)
    if not base:
        return ""
    template_mol = template_record_mol(template)
    reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
    template_nitrogen_idx = reverse.get(nitrogen_idx)
    if template_nitrogen_idx is None:
        return ""
    locant = template_record_locants(template).get(template_nitrogen_idx, "")
    oxidizable_nitrogen_count = sum(
        1
        for atom in template_mol.GetAtoms()
        if atom.GetSymbol() == "N" and atom.GetFormalCharge() == 0
    )
    if locant and locant[0].isdigit():
        return f"{base} {locant}-oxide"
    if oxidizable_nitrogen_count == 1:
        return f"{base} oxide"
    return ""


def _exact_retained_parent_candidate_name(template: dict) -> str:
    name = template_record_name(template)
    if template.get("source_file") == "naturalProducts.xml":
        name = _terminal_retained_parent_name(name)
    return _opsin_parseable_retained_parent_name(name)


def _exact_retained_parent_stereo_prefix(mol, Chem, template: dict, match: Sequence[int]) -> str:
    if template.get("source_file") != "naturalProducts.xml":
        return ""
    if not template.get("stereochemical"):
        return ""
    return _locanted_template_stereo_prefix(mol, Chem, template, match)


def parent_rules(Chem) -> tuple:
    sulfanium = SulfaniumParentRule(Chem)
    bis_sulfonium_sulfide = BisSulfoniumSulfideRetainedParentRule(Chem)
    methylenebis_dithiocarbamate = MethylenebisDithiocarbamateRetainedParentRule(Chem)
    dithiocarbamate = DithiocarbamateRetainedParentRule(Chem)
    dithiocarbonate = DithiocarbonateRetainedParentRule(Chem)
    hexamethylenediamide = HexamethylenediamideRetainedParentRule(Chem)
    small_inorganic = SmallInorganicRetainedParentRule(Chem)
    sydnone = SydnoneRetainedParentRule(Chem)
    phosphanium = PhosphaniumParentRule(Chem)
    phosphanylium_oxide = PhosphanyliumOxideParentRule(Chem)
    phosphanylium_imino = PhosphanyliumIminoParentRule(Chem)
    phosphanylium_ylidene = PhosphanyliumYlideneParentRule(Chem)
    azanium = AzaniumParentRule(Chem)
    neutral_amine = NeutralAmineParentRule(Chem)
    hydrazine = HydrazineParentRule(Chem)
    group14_hydride = Group14HydrideParentRule(Chem)
    phosphorus_acid = PhosphorusAcidParentRule(Chem)
    biphenyl_diphosphonite = BiphenylDiphosphoniteParentRule(Chem)
    phosphite_ester = PhosphiteEsterParentRule(Chem)
    charged_organophosphorus_oxo = ChargedOrganophosphorusOxoParentRule(Chem)
    phosphate_ester = PhosphateEsterRetainedParentRule(Chem)
    thiooxo_ester = ThiooxoEsterRetainedParentRule(Chem)
    sulfinate = SulfinateRetainedParentRule(Chem)
    sulfur_oxo = SulfurOxoRetainedParentRule(Chem)
    phosphorothioic_triamide = PhosphorothioicTriamideRetainedParentRule(Chem)
    dithiophosphate = DithiophosphateRetainedParentRule(Chem)
    organophosphorus_oxide = OrganophosphorusOxideParentRule(Chem)
    phosphorane = PhosphoraneParentRule(Chem)
    aminophosphane = AminophosphaneParentRule(Chem)
    bis_phosphino_amine = BisPhosphinoAmineRetainedParentRule(Chem)
    dithiazole = DithiazoleRetainedParentRule(Chem)
    thiadiazole_dithio = ThiadiazoleDithioRetainedParentRule(Chem)
    oxazinone = OxazinoneRetainedParentRule(Chem)
    oxirane_dione = OxiraneDioneRetainedParentRule(Chem)
    piperazine_dione_ylidene = PiperazineDioneYlideneRetainedParentRule(Chem)
    azidobenzylidene_cyclohexanone = AzidobenzylideneCyclohexanoneRetainedParentRule(Chem)
    bis_diethylaminobenzylidene_cyclopentanone = BisDiethylaminobenzylideneCyclopentanoneRetainedParentRule(Chem)
    cyclopropane_triylidene_nitrile = CyclopropaneTriylideneNitrileRetainedParentRule(Chem)
    organic_peroxide = OrganicPeroxideRetainedParentRule(Chem)
    phosphorylcholine = PhosphorylCholineRetainedParentRule(Chem)
    azinobis_benzothiazoline = AzinobisBenzothiazolineRetainedParentRule(Chem)
    organomagnesium_halide = OrganomagnesiumHalideParentRule(Chem)
    organoborane = OrganoboraneParentRule(Chem)
    benziodoxole = BenziodoxoleRetainedParentRule(Chem)
    iodane = IodaneParentRule(Chem)
    iodonium = IodoniumParentRule(Chem)
    iodine_oxo = IodineOxoParentRule(Chem)
    charged_retained = ChargedRetainedParentRule(Chem)
    pnictogen_hydride = PnictogenHydrideParentRule(Chem)
    oxido_nitroso_azanium = OxidoNitrosoAzaniumRule(Chem)
    oxo_sulfanium = OxoSulfaniumParentRule(Chem)
    n_halo_imine = NHaloImineParentRule(Chem)
    cyclic_nitroimine = CyclicNitroimineParentRule(Chem)
    oxime_sulfonate_nitrile = OximeSulfonateNitrileParentRule(Chem)
    nitrile_oxide = NitrileOxideRetainedParentRule(Chem)
    isocyanate = IsocyanateRetainedParentRule(Chem)
    carbodiimide = CarbodiimideRetainedParentRule(Chem)
    formamidine = FormamidineRetainedParentRule(Chem)
    charged_iminium = ChargedIminiumRetainedParentRule(Chem)
    guanidine = GuanidineRetainedParentRule(Chem)
    sulfonohydrazide = SulfonohydrazideRetainedParentRule(Chem)
    acyl_hydrazide = AcylHydrazideRetainedParentRule(Chem)
    acyl_diphosphate = AcylDiphosphateRetainedParentRule(Chem)
    retained_acid_ester = RetainedAcidEsterParentRule(Chem)
    retained_amide = RetainedAmideParentRule(Chem)
    beta_dicarbonyl = BetaDicarbonylParentRule(Chem)
    oxo_carboxylate = OxoCarboxylateRetainedParentRule(Chem)
    terminal_carboxylate = TerminalCarboxylateParentRule(Chem)
    boronic_pinacol_ester = BoronicPinacolEsterParentRule(Chem)
    carbamate = CarbamateRetainedParentRule(Chem)
    cyclic_n_carbamate_carboxylate = CyclicNCarbamateCarboxylateRetainedParentRule(Chem)
    sulfonamide = SulfonamideRetainedParentRule(Chem)
    n_substituted_sulfonamide = SulfonamideParentRule(Chem)
    sulfonimidamide = SulfonimidamideParentRule(Chem)
    lambda_sulfanone = LambdaSulfanoneParentRule(Chem)
    urea = UreaRetainedParentRule(Chem)
    sydnone_urea_amidate = SydnoneUreaAmidateRetainedParentRule(Chem)
    glutarimide = GlutarimideRetainedParentRule(Chem)
    amino_acid = AminoAcidRetainedParentRule(Chem)
    biphenyl = BiphenylParentRule(Chem)
    diphenylbenzene = DiphenylbenzeneParentRule(Chem)
    acene = AceneParentRule(Chem)
    acyclic_carbohydrate_retained = AcyclicCarbohydrateRetainedParentRule(Chem)
    acyclic_parent = AcyclicParentRule()
    imino_functional_acyclic = IminoFunctionalAcyclicSkeletonRule()
    conjugated_carbonyl_acyclic = ConjugatedCarbonylAcyclicSkeletonRule()
    whole_molecule_retained = WholeMoleculeRetainedParentRule(Chem)
    exact_retained = ExactRetainedParentRule(Chem)
    retained_ring_scaffold = RetainedRingScaffoldParentRule(Chem)
    exact_functional_retained = ExactFunctionalRetainedParentRule(Chem)
    exact_biochemical_retained = ExactBiochemicalRetainedParentRule(Chem)
    exact_retained_n_oxide = ExactRetainedNOxideParentRule(Chem)
    preferred_bicyclic = PreferredGenericBicyclicParentRule(Chem)
    fused = FusedParentRule(Chem)
    hydro_fused = HydroFusedParentRule(Chem)
    partial_hydro_fused = PartialHydroFusedParentRule(Chem)
    oxido_fused = OxidoFusedParentRule(Chem)
    generated_oxido_fused = GeneratedOxidoFusedParentRule(Chem)
    generic_bicyclic = GenericBicyclicParentRule(Chem)
    functional_generic_bicyclic = FunctionalGenericBicyclicParentRule(Chem)
    generic_spiro = GenericSpiroParentRule(Chem)
    single_ring = SingleRingParentRule(Chem)
    positive_single_ring = PositiveSingleRingParentRule(Chem)
    generated_exact = GeneratedExactParentRule(Chem)
    generated_fused = GeneratedFusedParentRule(Chem)
    high_signal_generated_topology_fused = HighSignalGeneratedTopologyFusedParentRule(Chem)
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
    sulfur_oxo_acyclic = SulfurOxoAcyclicSkeletonWithRingSubstituentsRule()
    spiro_functional_chain = SpiroFunctionalChainSubstituentRule()
    lambda_sulfanone_spiro = LambdaSulfanoneSpiroSubstituentRule(Chem)
    charged_ring_acyclic = ChargedRingAcyclicSkeletonWithRingSubstituentsRule()
    functional_acyclic = FunctionalAcyclicSkeletonWithRingSubstituentsRule()
    acyclic_with_ring_substituents = AcyclicSkeletonWithRingSubstituentsRule()

    return (
        RuleSpec(
            "small-inorganic-retained-parent",
            parent_priority("charged-inorganic-retained-parent"),
            (small_inorganic_retained_signal,),
            small_inorganic.candidates,
        ),
        RuleSpec("sydnone-retained-parent", parent_priority("senior-functional-exception"), (charged_ring_atom,), sydnone.candidates),
        RuleSpec(
            "bis-sulfonium-sulfide-retained-parent",
            parent_priority("senior-functional-exception"),
            (nonring_positive_sulfur,),
            bis_sulfonium_sulfide.candidates,
        ),
        RuleSpec(
            "methylenebis-dithiocarbamate-retained-parent",
            parent_priority("senior-functional-exception"),
            (dithiocarbamate_retained_signal,),
            methylenebis_dithiocarbamate.candidates,
        ),
        RuleSpec(
            "dithiocarbamate-retained-parent",
            parent_priority("senior-functional-exception"),
            (dithiocarbamate_retained_signal,),
            dithiocarbamate.candidates,
        ),
        RuleSpec(
            "dithiocarbonate-retained-parent",
            parent_priority("senior-functional-exception"),
            (dithiocarbonate_signal,),
            dithiocarbonate.candidates,
        ),
        RuleSpec(
            "hexamethylenediamide-retained-parent",
            parent_priority("senior-functional-exception"),
            (functional_skeleton_signal, nitrogen_oxygen_signal),
            hexamethylenediamide.candidates,
        ),
        RuleSpec("sulfanium-parent", parent_priority("senior-functional-exception"), (nonring_positive_sulfur,), sulfanium.candidates),
        RuleSpec(
            "phosphanylium-oxide-parent",
            parent_priority("senior-functional-exception"),
            (nonring_positive_phosphorus,),
            phosphanylium_oxide.candidates,
        ),
        RuleSpec(
            "phosphanylium-imino-parent",
            parent_priority("senior-functional-exception"),
            (nonring_positive_phosphorus,),
            phosphanylium_imino.candidates,
        ),
        RuleSpec(
            "phosphanylium-ylidene-parent",
            parent_priority("senior-functional-exception"),
            (nonring_positive_phosphorus,),
            phosphanylium_ylidene.candidates,
        ),
        RuleSpec("phosphanium-parent", parent_priority("senior-functional-exception"), (nonring_positive_phosphorus,), phosphanium.candidates),
        RuleSpec("azanium-parent", parent_priority("senior-functional-exception"), (nonring_positive_nitrogen,), azanium.candidates),
        RuleSpec("neutral-amine-parent", parent_priority("senior-functional-exception"), (nonring_neutral_nitrogen,), neutral_amine.candidates),
        RuleSpec("hydrazine-parent", parent_priority("senior-functional-exception"), (nonring_neutral_nitrogen,), hydrazine.candidates),
        RuleSpec("group14-hydride-parent", parent_priority("senior-functional-exception"), (nonring_neutral_group14,), group14_hydride.candidates),
        RuleSpec("phosphorus-acid-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), phosphorus_acid.candidates),
        RuleSpec("biphenyl-diphosphonite-parent", parent_priority("senior-functional-exception"), (neutral_phosphorus,), biphenyl_diphosphonite.candidates),
        RuleSpec("phosphite-ester-parent", parent_priority("senior-functional-exception"), (neutral_phosphorus,), phosphite_ester.candidates),
        RuleSpec(
            "charged-organophosphorus-oxo-parent",
            parent_priority("senior-functional-exception"),
            (nonring_neutral_phosphorus,),
            charged_organophosphorus_oxo.candidates,
        ),
        RuleSpec(
            "phosphate-ester-retained-parent",
            parent_priority("senior-functional-exception"),
            (nonring_neutral_phosphorus,),
            phosphate_ester.candidates,
        ),
        RuleSpec("thiooxo-ester-retained-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal, sulfur_signal), thiooxo_ester.candidates),
        RuleSpec(
            "sulfinate-retained-parent",
            parent_priority("senior-functional-exception"),
            (sulfinate_retained_parent_signal,),
            sulfinate.candidates,
        ),
        RuleSpec("sulfur-oxo-retained-parent", parent_priority("senior-functional-exception"), (sulfur_oxo_retained_parent_signal,), sulfur_oxo.candidates),
        RuleSpec("phosphorothioic-triamide-retained-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), phosphorothioic_triamide.candidates),
        RuleSpec("dithiophosphate-retained-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), dithiophosphate.candidates),
        RuleSpec("acyl-diphosphate-retained-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), acyl_diphosphate.candidates),
        RuleSpec("organophosphorus-oxide-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), organophosphorus_oxide.candidates),
        RuleSpec("phosphorane-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), phosphorane.candidates),
        RuleSpec("aminophosphane-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), aminophosphane.candidates),
        RuleSpec("bis-phosphino-amine-retained-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), bis_phosphino_amine.candidates),
        RuleSpec("dithiazole-retained-parent", parent_priority("senior-functional-exception"), (sulfur_signal,), dithiazole.candidates),
        RuleSpec("thiadiazole-dithio-retained-parent", parent_priority("senior-functional-exception"), (sulfur_signal,), thiadiazole_dithio.candidates),
        RuleSpec(
            "oxazinone-retained-parent",
            parent_priority("senior-functional-exception"),
            (min_ring_count(1), functional_skeleton_signal, nitrogen_oxygen_signal),
            oxazinone.candidates,
        ),
        RuleSpec(
            "oxirane-dione-retained-parent",
            parent_priority("senior-functional-exception"),
            (oxirane_dione_signal,),
            oxirane_dione.candidates,
        ),
        RuleSpec(
            "piperazine-dione-ylidene-retained-parent",
            parent_priority("senior-functional-exception"),
            (min_ring_count(1), functional_skeleton_signal, nitrogen_oxygen_signal),
            piperazine_dione_ylidene.candidates,
        ),
        RuleSpec(
            "azidobenzylidene-cyclohexanone-retained-parent",
            parent_priority("senior-functional-exception"),
            (min_ring_count(3), nitrogen_oxygen_signal),
            azidobenzylidene_cyclohexanone.candidates,
        ),
        RuleSpec(
            "bis-diethylaminobenzylidene-cyclopentanone-retained-parent",
            parent_priority("senior-functional-exception"),
            (min_ring_count(3), nitrogen_oxygen_signal),
            bis_diethylaminobenzylidene_cyclopentanone.candidates,
        ),
        RuleSpec(
            "cyclopropane-triylidene-nitrile-retained-parent",
            parent_priority("senior-functional-exception"),
            (min_ring_count(1), nitrogen_signal),
            cyclopropane_triylidene_nitrile.candidates,
        ),
        RuleSpec(
            "organic-peroxide-retained-parent",
            parent_priority("senior-functional-exception"),
            (oxygen_oxygen_single_bond_signal,),
            organic_peroxide.candidates,
        ),
        RuleSpec(
            "phosphorylcholine-retained-parent",
            parent_priority("senior-functional-exception"),
            (formal_charge_signal,),
            phosphorylcholine.candidates,
        ),
        RuleSpec(
            "azinobis-benzothiazoline-retained-parent",
            parent_priority("senior-functional-exception"),
            (sulfur_imino_signal,),
            azinobis_benzothiazoline.candidates,
        ),
        RuleSpec("organomagnesium-halide-parent", parent_priority("senior-functional-exception"), (acyclic,), organomagnesium_halide.candidates),
        RuleSpec("organoborane-parent", parent_priority("senior-functional-exception"), (boron_signal,), organoborane.candidates),
        RuleSpec("benziodoxole-retained-parent", parent_priority("senior-functional-exception"), (iodine_signal,), benziodoxole.candidates),
        RuleSpec("iodane-parent", parent_priority("senior-functional-exception"), (iodine_signal,), iodane.candidates),
        RuleSpec("iodonium-parent", parent_priority("senior-functional-exception"), (iodine_signal,), iodonium.candidates),
        RuleSpec("iodine-oxo-parent", parent_priority("senior-functional-exception"), (iodine_signal,), iodine_oxo.candidates),
        RuleSpec("charged-retained-parent", parent_priority("senior-functional-exception"), (formal_charge_signal,), charged_retained.candidates),
        RuleSpec("pnictogen-hydride-parent", parent_priority("senior-functional-exception"), (nonring_neutral_pnictogen,), pnictogen_hydride.candidates),
        RuleSpec(
            "oxido-nitroso-azanium-parent",
            parent_priority("senior-functional-exception"),
            (acyclic, nonring_positive_nitrogen),
            oxido_nitroso_azanium.candidates,
        ),
        RuleSpec("oxo-sulfanium-parent", parent_priority("senior-functional-exception"), (nonring_positive_sulfur,), oxo_sulfanium.candidates),
        RuleSpec("n-halo-imine-parent", parent_priority("senior-functional-exception"), (acyclic, imino_skeleton_signal), n_halo_imine.candidates),
        RuleSpec("cyclic-nitroimine-parent", parent_priority("senior-functional-exception"), (min_ring_count(1), imino_skeleton_signal), cyclic_nitroimine.candidates),
        RuleSpec("oxime-sulfonate-nitrile-parent", parent_priority("senior-functional-exception"), (oxime_sulfonate_nitrile_signal,), oxime_sulfonate_nitrile.candidates),
        RuleSpec("nitrile-oxide-retained-parent", parent_priority("senior-functional-exception"), (nitrile_oxide_signal,), nitrile_oxide.candidates),
        RuleSpec("isocyanate-retained-parent", parent_priority("senior-functional-exception"), (acyclic, nitrogen_oxygen_signal), isocyanate.candidates),
        RuleSpec("carbodiimide-retained-parent", parent_priority("senior-functional-exception"), (imino_skeleton_signal,), carbodiimide.candidates),
        RuleSpec("formamidine-retained-parent", parent_priority("senior-functional-exception"), (imino_skeleton_signal,), formamidine.candidates),
        RuleSpec("charged-iminium-retained-parent", parent_priority("senior-functional-exception"), (nonring_positive_nitrogen,), charged_iminium.candidates),
        RuleSpec("guanidine-retained-parent", parent_priority("senior-functional-exception"), (imino_skeleton_signal,), guanidine.candidates),
        RuleSpec("sulfonohydrazide-retained-parent", parent_priority("senior-functional-exception"), (imino_skeleton_signal,), sulfonohydrazide.candidates),
        RuleSpec("acyl-hydrazide-retained-parent", parent_priority("senior-functional-exception"), (imino_skeleton_signal,), acyl_hydrazide.candidates),
        RuleSpec("retained-acid-ester-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal, oxygen_signal), retained_acid_ester.candidates),
        RuleSpec(
            "retained-amide-parent",
            parent_priority("senior-functional-exception"),
            (functional_skeleton_signal, nitrogen_oxygen_signal, amide_acyl_signal),
            retained_amide.candidates,
        ),
        RuleSpec(
            "beta-dicarbonyl-parent",
            parent_priority("senior-functional-exception"),
            (functional_skeleton_signal, oxygen_signal),
            beta_dicarbonyl.candidates,
        ),
        RuleSpec("oxo-carboxylate-retained-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal, terminal_carboxylate_signal), oxo_carboxylate.candidates),
        RuleSpec("terminal-carboxylate-parent", parent_priority("senior-functional-exception"), (terminal_carboxylate_signal,), terminal_carboxylate.candidates),
        RuleSpec("boronic-pinacol-ester-parent", parent_priority("senior-functional-exception"), (min_ring_count(1), boron_signal, oxygen_signal), boronic_pinacol_ester.candidates),
        RuleSpec(
            "carbamate-retained-parent",
            parent_priority("senior-functional-exception"),
            (functional_skeleton_signal, nitrogen_oxygen_signal),
            carbamate.candidates,
        ),
        RuleSpec(
            "cyclic-n-carbamate-carboxylate-retained-parent",
            parent_priority("senior-functional-exception"),
            (min_ring_count(1), functional_skeleton_signal, nitrogen_oxygen_signal),
            cyclic_n_carbamate_carboxylate.candidates,
        ),
        RuleSpec("sulfonamide-retained-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal, sulfur_nitrogen_oxygen_signal), sulfonamide.candidates),
        RuleSpec("sulfonamide-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal, sulfur_nitrogen_oxygen_signal), n_substituted_sulfonamide.candidates),
        RuleSpec("sulfonimidamide-parent", parent_priority("senior-functional-exception"), (imino_skeleton_signal,), sulfonimidamide.candidates),
        RuleSpec("lambda-sulfanone-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal, sulfur_oxygen_signal), lambda_sulfanone.candidates),
        RuleSpec(
            "urea-retained-parent",
            parent_priority("senior-functional-exception"),
            (functional_skeleton_signal, nitrogen_oxygen_signal),
            urea.candidates,
        ),
        RuleSpec("sydnone-urea-amidate-retained-parent", parent_priority("senior-functional-exception"), (charged_ring_atom, functional_skeleton_signal, nitrogen_oxygen_signal), sydnone_urea_amidate.candidates),
        RuleSpec(
            "glutarimide-retained-parent",
            parent_priority("senior-functional-exception"),
            (min_ring_count(1), functional_skeleton_signal, nitrogen_oxygen_signal),
            glutarimide.candidates,
        ),
        RuleSpec(
            "amino-acid-retained-parent",
            parent_priority("senior-functional-exception"),
            (functional_skeleton_signal, nitrogen_oxygen_signal),
            amino_acid.candidates,
        ),
        RuleSpec("biphenyl-parent", parent_priority("whole-molecule-retained-parent"), (min_ring_count(2),), biphenyl.candidates),
        RuleSpec("diphenylbenzene-parent", parent_priority("whole-molecule-retained-parent"), (min_ring_count(3),), diphenylbenzene.candidates),
        RuleSpec("acene-retained-parent", parent_priority("whole-molecule-retained-parent"), (min_ring_count(5),), acene.candidates),
        RuleSpec(
            "acyclic-carbohydrate-retained-parent",
            parent_priority("acyclic-parent"),
            (acyclic,),
            acyclic_carbohydrate_retained.candidates,
        ),
        RuleSpec("acyclic-parent", parent_priority("acyclic-parent"), (acyclic,), acyclic_parent.candidates),
        RuleSpec(
            "imino-functional-acyclic-skeleton-with-ring-substituents",
            parent_priority("functional-acyclic-skeleton"),
            (imino_skeleton_signal, min_nonring_heavy_atom_count(2), no_exocyclic_charged_fused_parent),
            imino_functional_acyclic.candidates,
        ),
        RuleSpec(
            "conjugated-carbonyl-acyclic-skeleton-with-ring-substituents",
            parent_priority("functional-acyclic-skeleton"),
            (
                functional_skeleton_signal,
                acyclic_preference_signal,
                conjugated_carbonyl_diene_signal,
                min_nonring_heavy_atom_count(8),
                no_charged_ring_atom,
            ),
            conjugated_carbonyl_acyclic.candidates,
        ),
        RuleSpec(
            "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
            parent_priority("functional-acyclic-skeleton"),
            (min_ring_count(2), sulfur_signal),
            sulfur_oxo_acyclic.candidates,
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
            (acyclic_preference_signal, min_nonring_heavy_atom_count(8)),
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
            (min_ring_count(1), min_nonring_heavy_atom_count(2), chain_dominant_acyclic_parent_signal),
            chain_dominant_acyclic.candidates,
        ),
        RuleSpec(
            "spiro-functional-chain-substituent",
            parent_priority("senior-spiro-functional-chain-substituent"),
            (min_ring_count(1), spiro_topology),
            spiro_functional_chain.candidates,
        ),
        RuleSpec(
            "lambda-sulfanone-spiro-substituent",
            parent_priority("senior-spiro-functional-chain-substituent"),
            (min_ring_count(1), spiro_topology),
            lambda_sulfanone_spiro.candidates,
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
            "exact-retained-n-oxide-parent",
            parent_priority("senior-functional-exception"),
            (terminal_n_oxide_signal,),
            exact_retained_n_oxide.candidates,
        ),
        RuleSpec(
            "retained-ring-scaffold-parent",
            parent_priority("retained-fused-parent"),
            (min_ring_system_ring_count(2),),
            retained_ring_scaffold.candidates,
        ),
        RuleSpec(
            "exact-functional-retained-parent",
            parent_priority("exact-retained-parent"),
            (min_ring_count(1), exact_functional_retained_signal),
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
        RuleSpec("partial-hydro-fused-retained-parent", parent_priority("retained-partial-hydro-fallback"), (min_ring_count(2), min_ring_system_ring_count(2)), partial_hydro_fused.candidates),
        RuleSpec(
            "generated-spiro-partial-hydro-parent",
            parent_priority("generated-spiro-component-parent"),
            (min_ring_system_ring_count(2), spiro_topology),
            generated_spiro_partial_hydro.candidates,
        ),
        RuleSpec(
            "component-spiro-parent",
            parent_priority("component-spiro-parent"),
            (min_ring_system_ring_count(3), component_spiro_topology),
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
        RuleSpec(
            "high-signal-generated-topology-fused-parent",
            parent_priority("generated-parent"),
            (min_ring_system_ring_count(2),),
            high_signal_generated_topology_fused.candidates,
        ),
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
            (
                functional_skeleton_signal,
                acyclic_preference_signal,
                min_nonring_heavy_atom_count(8),
                no_charged_ring_atom,
            ),
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
    _skeletal_replacement_bond_order_mismatch_count,
    _skeletal_replacement_spiro_locant_instability_penalty,
    _skeletal_replacement_unsaturated_hetero_bond_penalty,
    _retained_topology_replacement_parent_name,
    _retained_topology_replacement_template_allows_nh,
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
    _has_unsaturated_core_bond,
    _hydrogenated_template_penalty,
    _template_source_rank,
    _fused_candidate_preference,
    _substituent_locant_tuple,
    _benzimidazole_orientation_preference,
    _benzimidazole_template_h_preference,
    _benzimidazole_external_n_preference,
    _benzimidazole_external_carbon_locant_preference,
    _generated_benzimidazole_external_n_preference,
    _generated_fused_graph_orientation_preference,
    _benzotriazole_template_h_preference,
    _has_n_oxide_on_match,
    _matched_fused_indicated_hydrogen_locants,
    _name_fused_core_match,
    _name_natural_product_topology_match,
    _opsin_parseable_retained_parent_name,
    _fused_oxo_suffix_name,
    _adjust_fused_parent_for_charge,
    _drop_indicated_hydrogen_prefix,
    _drop_indicated_hydrogen_segments,
    _adjust_fused_parent_for_indicated_hydrogen,
    _append_ring_parent_isotope_suffixes,
    _merge_ring_parent_candidates,
    _named_ring_instance_records,
    _single_ring_substituent_locant,
    _single_ring_stereo_prefix,
    _polycyclic_stereo_prefix,
    _single_ring_oxo_suffix_name,
    _oxo_adjusted_unsaturated_carbocycle_parent,
    _cycloalkapolyene_ring_size,
    _max_nonoxo_ring_double_bond_locs,
    _sulfanium_name,
    _phosphanium_name,
    _azanium_name,
    _substituted_onium_name,
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


def _cached_iter_fused_template_matches(
    ctx,
    templates: Sequence[dict],
    Chem,
    *,
    topology_query: bool = False,
    require_one_ring_system: bool = False,
    require_fully_aromatic: bool = False,
    reject_extra_core_bonds: bool = False,
):
    cache = getattr(ctx.cache, "fused_template_matches", None)
    if cache is None:
        yield from _iter_fused_template_matches(
            ctx,
            templates,
            Chem,
            topology_query=topology_query,
            require_one_ring_system=require_one_ring_system,
            require_fully_aromatic=require_fully_aromatic,
            reject_extra_core_bonds=reject_extra_core_bonds,
        )
        return

    key = (
        id(templates),
        bool(topology_query),
        bool(require_one_ring_system),
        bool(require_fully_aromatic),
        bool(reject_extra_core_bonds),
    )
    cached = cache.get(key)
    if cached is None:
        cached = tuple(
            (template, tuple(match))
            for template, match in _iter_fused_template_matches(
                ctx,
                templates,
                Chem,
                topology_query=topology_query,
                require_one_ring_system=require_one_ring_system,
                require_fully_aromatic=require_fully_aromatic,
                reject_extra_core_bonds=reject_extra_core_bonds,
            )
        )
        cache[key] = cached
    yield from cached
