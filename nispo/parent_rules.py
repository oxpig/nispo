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
from .candidate_facts import (
    CandidateSafety,
    ChargeEvidence,
    OPSIN_FRAGILE_EXOCYCLIC_CHARGED_RING_SUFFIX,
    OPSIN_FRAGILE_GROUP14_HETERO_CARBON_LIGAND,
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
from .molecule_cache import mol_cache_get, mol_cache_set
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
    adjust_ring_parent_for_charge as _adjust_ring_parent_for_charge,
    ring_parent_candidates as _ring_parent_candidates,
)
from .spiro_component_rules import component_spiro_parent_candidates
from .rule_dispatch import (
    RuleSpec,
    acyclic,
    acyclic_preference_signal,
    charged_ring_atom,
    component_spiro_topology,
    conjugated_carbonyl_diene_signal,
    exact_functional_retained_signal,
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
        name = _methylenebis_dithiocarbamate_name(mol, self._Chem)
        if name:
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                parent_atoms=set(range(mol.GetNumAtoms())),
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
        ligand_idx = None
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
            elif bond.GetBondType() == self._Chem.BondType.SINGLE and nbr.GetAtomicNum() > 1:
                if ligand_idx is not None:
                    return
                ligand_idx = nidx
            elif nbr.GetAtomicNum() > 1:
                return
        if oxide_idx is None or oxido_idx is None or ligand_idx is None:
            return
        ligand = _simple_or_branch_prefix(mol, ligand_idx, pidx)
        if not ligand:
            return
        covered = {pidx, oxide_idx, oxido_idx}
        covered.update(_collect_component_atoms(mol, ligand_idx, pidx))
        if covered != atom_set:
            return
        ligand = _strip_outer_parentheses(ligand)
        name = f"oxido({ligand})phosphanylium oxide"
        yield NameCandidate(
            name,
            self.family,
            self.priority,
            parent_atoms=atom_set,
            sort_key=(len(name), name),
        )


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
        oxido_idx = None
        ylidene_idx = None
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
            elif (
                nbr.GetSymbol() == "C"
                and nbr.GetFormalCharge() == 0
                and not nbr.IsInRing()
                and bond.GetBondType() == self._Chem.BondType.DOUBLE
            ):
                if ylidene_idx is not None:
                    return
                ylidene_idx = nidx
            elif bond.GetBondType() == self._Chem.BondType.SINGLE and nbr.GetAtomicNum() > 1:
                if ligand_idx is not None:
                    return
                ligand_idx = nidx
            elif nbr.GetAtomicNum() > 1:
                return
        if oxido_idx is None or ylidene_idx is None or ligand_idx is None:
            return
        ligand = _simple_or_branch_prefix(mol, ligand_idx, pidx)
        ylidene = _simple_or_branch_prefix(mol, ylidene_idx, pidx)
        if not ligand or not ylidene or "methylidene" not in ylidene:
            return
        covered = {pidx, oxido_idx}
        covered.update(_collect_component_atoms(mol, ligand_idx, pidx))
        covered.update(_collect_component_atoms(mol, ylidene_idx, pidx))
        if covered != atom_set:
            return
        ligand = _strip_outer_parentheses(ligand)
        ylidene = _strip_outer_parentheses(ylidene)
        name = f"oxido({ligand})({ylidene})phosphanylium"
        yield NameCandidate(
            name,
            self.family,
            self.priority,
            parent_atoms=atom_set,
            sort_key=(len(name), name),
        )


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
        ligand_atom_sets,
    ):
        return ""
    ligand_text = _phosphine_oxide_ligand_text(ligand_names)
    return ligand_text + "phosphine oxide" if ligand_text else ""


def _simple_no_hydrogen_phosphine_oxide_scope(mol, Chem, ligand_atom_sets: Sequence[frozenset[int]]) -> bool:
    if mol.GetNumAtoms() > 18:
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
        if heavy_count > 12:
            return False
        if sum(1 for ring in atom_rings if ring.issubset(ligand_atoms)) > 1:
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


class OrganicPeroxideRetainedParentRule:
    family = "organic-peroxide-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
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
            if iodine.GetSymbol() != "I" or iodine.GetFormalCharge() != 1 or iodine.IsInRing():
                continue
            iodine_idx = iodine.GetIdx()
            implicit_hydrogens = iodine.GetTotalNumHs()
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
                if len(carbon_neighbors) + implicit_hydrogens != 2 or len(carbon_neighbors) not in {1, 2}:
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
    if not 1 <= len(ligand_names) <= 2:
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


def _carbodiimide_parent_names(mol, Chem) -> list[str]:
    motif_names = []
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
        for nitrogen_idx in terminal_nitrogens:
            substituent = _carbodiimide_n_substituent(mol, Chem, nitrogen_idx, carbon_idx)
            if substituent is None:
                substituents = None
                break
            if substituent:
                substituents.append(substituent)
        if substituents is None:
            continue
        name = _carbodiimide_name(substituents)
        if name:
            motif_names.append(name)
    if len(motif_names) != 1:
        return []
    return sorted(set(motif_names), key=lambda name: (len(name), name))


def _carbodiimide_n_substituent(mol, Chem, nitrogen_idx: int, carbon_idx: int) -> str | None:
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
        return ""
    substituent = _retained_parent_substituent_alias(
        _simple_or_branch_prefix(mol, substituent_atoms[0], nitrogen_idx)
    )
    if not substituent or not _allowed_carbodiimide_substituent(substituent):
        return None
    return substituent


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
    if _simple_n_retained_parent_substituent(name):
        return True
    if len(name) > 56:
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
            "amino",
            "cyclo",
            "morpholin",
            "piperid",
            "piperaz",
            "pyrrolid",
            "azetidin",
            "phenyl",
            "benzyl",
            "pyrid",
            "hydroxy",
            "methoxy",
            "oxo",
        )
    )


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
        for name in _carbodiimide_parent_names(ctx.mol, self._Chem):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


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
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                source_family=source_family,
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
        for name, source_family in _retained_amide_records(mol, self._Chem):
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
    acyclic_only = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() != 0:
            return
        seen = set()
        for name in _oxo_carboxylate_parent_names(mol, self._Chem):
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
        if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0 or phosphorus.IsInRing():
            continue
        pidx = phosphorus.GetIdx()
        terminal_anionic_oxygen = 0
        terminal_hydroxy_oxygen = 0
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
            owner_names.append(owner)
        if ok and terminal_anionic_oxygen == 1 and len(owner_names) == 2:
            names.append(_phosphite_ester_name(owner_names))
        if (
            ok
            and len(owner_names) == 1
            and terminal_anionic_oxygen + terminal_hydroxy_oxygen == 2
            and terminal_anionic_oxygen in {0, 1, 2}
        ):
            names.append(_phosphite_monoester_name(owner_names[0], terminal_hydroxy_oxygen))
    return names


def _phosphite_ester_name(owner_names: Sequence[str]) -> str:
    first, second = sorted(owner_names)
    if first == second:
        if _simple_unparenthesized_n_substituent(first):
            return f"di{first}phosphite"
        return f"bis({_strip_outer_parentheses(first)}) phosphite"
    return f"{_parenthesized_if_complex(first)} {_parenthesized_if_complex(second)} phosphite"


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
        terminal_double_oxygens = []
        carbon_single_ligands = []
        carbon_double_ligands = []
        oxygen_ligands = []
        terminal_amino_ligands = []
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
                nbr.GetSymbol() == "N"
                and nbr.GetFormalCharge() == 0
                and bond_type == Chem.BondType.SINGLE
                and not heavy
            ):
                terminal_amino_ligands.append(nidx)
                covered.add(nidx)
            else:
                ok = False
                break
        if not ok or covered != atom_set:
            continue

        double_oxygen_count = len(terminal_double_oxygens)
        anionic_oxygen_count = len(terminal_anionic_oxygens)
        if (
            double_oxygen_count == 1
            and anionic_oxygen_count == 2
            and len(carbon_single_ligands) == 1
            and not carbon_double_ligands
            and not oxygen_ligands
            and not terminal_amino_ligands
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
            and phosphorus.GetTotalNumHs() == 0
        ):
            prefix = _phosphorus_attached_prefix(mol, carbon_single_ligands[0], pidx)
            if prefix:
                names.append(prefix + "phosphonamidate")
            continue

        if (
            double_oxygen_count == 1
            and anionic_oxygen_count == 1
            and len(oxygen_ligands) == 1
            and not carbon_single_ligands
            and not carbon_double_ligands
            and not terminal_amino_ligands
            and phosphorus.GetTotalNumHs() == 1
        ):
            owner = _single_external_substituent_name(mol, oxygen_ligands[0], pidx)
            if owner and not _unstable_retained_functional_parent_substituent(owner):
                names.append(f"{_parenthesized_if_complex(owner)} phosphonate")
            continue

        if (
            double_oxygen_count == 0
            and anionic_oxygen_count == 3
            and len(carbon_double_ligands) == 1
            and not carbon_single_ligands
            and not oxygen_ligands
            and not terminal_amino_ligands
            and phosphorus.GetTotalNumHs() == 0
        ):
            prefix = _phosphorus_attached_prefix(mol, carbon_double_ligands[0], pidx)
            if prefix:
                names.append(prefix + "phosphite")
    return names


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
    for sulfur in mol.GetAtoms():
        if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0 or sulfur.IsInRing():
            continue
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
    terminal_amino = 0
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
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen += 1
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
        "terminal_amino": terminal_amino,
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
    terminal_amino = record["terminal_amino"]
    center_hydrogens = record["center_hydrogens"]
    single_oxygen_count = len(owner_names) + terminal_oxido + terminal_hydroxy
    if center_symbol == "P":
        if double_oxygen != 0:
            return ""
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


def _dithiocarbamate_unit_name_and_atoms(mol, Chem, nitrogen_idx: int, methylene_idx: int) -> tuple[str, set[int]] | None:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing():
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in nitrogen.GetNeighbors()
        if nbr.GetIdx() != methylene_idx and nbr.GetAtomicNum() > 1
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
        if not owner:
            continue
        generic_source = _retained_acid_ester_source_family(mol, Chem, carbonyl_idx, acyl_idx, "generic")
        generic = _generic_acyl_ester_names(mol, Chem, carbonyl_idx, acyl_idx, owner)
        names.extend((name, generic_source) for name in generic)
        dicarboxylate = _dicarboxylate_ester_name(mol, Chem, carbonyl_idx, acyl_idx, owner)
        if dicarboxylate:
            names.append((dicarboxylate, "retained-acid-ester:dicarboxylate"))
        acrylate = _acrylate_ester_name(mol, Chem, carbonyl_idx, acyl_idx, owner)
        if acrylate:
            names.append((acrylate, "retained-acid-ester:acrylate"))
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
    return [name for name, _source_family in _retained_amide_records(mol, Chem)]


def _retained_amide_records(mol, Chem) -> list[tuple[str, str]]:
    records = []
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
            records.append((_join_n_prefix_functional_parent(n_prefix, core), "generic-acyl-amide"))
        records.extend(
            _ring_carboxamide_parent_records(
                mol,
                Chem,
                carbonyl_idx,
                nitrogen_idx,
                acyl_idx,
                n_prefix,
            )
        )
    return records


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
            prefix = simple_external_prefix(mol, nidx, ring_idx) or branch_prefix(mol, nidx, ring_idx)
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
            prefix = simple_external_prefix(mol, nidx, ring_idx) or branch_prefix(mol, nidx, ring_idx)
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
    if suffix == "hydrazide":
        return parent + "hydrazide"
    return parent[:-1] + "oate" if parent.endswith("e") else parent + "oate"


def _oxo_carboxylate_parent_names(mol, Chem) -> list[str]:
    groups = _terminal_carboxyl_groups(mol, Chem, allowed_kinds={"carboxylate"})
    names: list[str] = []
    if len(groups) == 1:
        names.extend(_alpha_keto_monocarboxylate_names(mol, Chem, groups[0]))
    elif len(groups) == 2:
        name = _alpha_keto_dicarboxylate_name(mol, Chem, groups)
        if name:
            names.append(name)
    return sorted(set(names), key=lambda name: (len(name), name))


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
    endpoints = [group[0] for group in groups if group[2] == "carboxylate"]
    if len(endpoints) != 2:
        return ""
    path = tuple(Chem.rdmolops.GetShortestPath(mol, endpoints[0], endpoints[1]))
    if not _simple_alpha_keto_dicarboxylate_path(mol, Chem, path):
        return ""
    reverse = tuple(reversed(path))
    orientations = []
    for candidate_path in (path, reverse):
        oxo_locants = _path_oxo_locants(mol, Chem, candidate_path)
        if len(oxo_locants) == 1:
            orientations.append((oxo_locants[0], candidate_path))
    if not orientations:
        return ""
    oxo_locant, oriented = min(orientations, key=lambda item: (item[0], tuple(item[1])))
    root = ROOTS.get(len(oriented))
    if not root:
        return ""
    return f"{oxo_locant}-oxo{root}anedioate"


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
    if "carbamoyl" not in name:
        return False
    if len(name) > 180:
        return False
    if " " in name:
        return False
    if any(fragment in name for fragment in ("phosph", "sulfon", "sulfin", "urea")):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9,().'\-\[\]\^]+", name))


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
        parent_atoms = _nonring_heavy_atom_indices(mol)
        if name and _should_defer_spiro_substituent_parent_to_senior_azolone(ctx, parent_atoms):
            return
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
        parent_atoms = _nonring_heavy_atom_indices(mol)
        if name and _should_defer_spiro_substituent_parent_to_senior_azolone(ctx, parent_atoms):
            return
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
        parent_atoms = _nonring_heavy_atom_indices(mol)
        if name and _should_defer_spiro_substituent_parent_to_senior_azolone(ctx, parent_atoms):
            return
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
        parent_atoms = _nonring_heavy_atom_indices(mol)
        if name and _should_defer_spiro_substituent_parent_to_senior_azolone(ctx, parent_atoms):
            return
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
        for template, match in _iter_fused_template_matches(
            ctx, self._template_iter(ctx), self._Chem, reject_extra_core_bonds=True
        ):
            facts.append(
                _RetainedFusedCandidateFacts(
                    template=template,
                    match=tuple(match),
                    topology_query=False,
                    structural_key=_retained_fused_fact_sort_key(mol, template, match, topology_query=False),
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
                    structural_key=_retained_fused_fact_sort_key(mol, template, match, topology_query=True),
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
                parent_substituted_hetero_atom_count=_parent_substituted_hetero_atom_count(mol, match),
                charge=_polycyclic_parent_charge_evidence(mol, match),
                stereo=_molecule_stereo_evidence(mol, self._Chem),
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
    mol,
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
        _benzimidazole_template_h_preference(mol, template, match, template_name),
        _benzimidazole_external_n_preference(mol, template, match),
        _benzimidazole_external_carbon_locant_preference(mol, template, match),
        tuple(match),
    )


def _retained_fused_render_set(
    facts: tuple[_RetainedFusedCandidateFacts, ...],
) -> tuple[_RetainedFusedCandidateFacts, ...]:
    if len(facts) <= _RETAINED_FUSED_RENDER_SET_SIZE:
        if any(template_record_name(fact.template) == "1H-benzimidazole" for fact in facts):
            return tuple(sorted(facts, key=lambda fact: fact.structural_key))
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
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=self.priority,
                sort_key=sort_key,
                parent_atoms=frozenset(match),
                stereo_locants=_template_match_stereo_locants(template, match),
                template_id=id(template),
                template_name=template_name,
                source_family=str(template.get("source_family", "")),
                source_file=str(template.get("source_file", "")),
                parent_substituted_hetero_atom_count=_parent_substituted_hetero_atom_count(mol, match),
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
            for record in _named_ring_instance_records(mol, tuple(ring), self._Chem):
                yield NameCandidate(
                    name=record.name,
                    family=self.family,
                    priority=candidate_priority,
                    sort_key=_single_ring_candidate_sort_key(record.name),
                    stereo_prefix=record.stereo_prefix,
                    parent_atoms=frozenset(ring),
                    parent_aromatic_atom_count=_ring_aromatic_atom_count(mol, ring),
                    parent_hetero_atom_count=_ring_hetero_atom_count(mol, ring),
                    parent_terminal_hetero_substituent_count=_ring_terminal_hetero_substituent_count(mol, ring),
                    parent_external_nitrogen_substituent_count=_ring_external_nitrogen_substituent_count(mol, ring),
                    parent_external_oxygen_substituent_count=_ring_external_oxygen_substituent_count(mol, ring),
                    locants=_locants_from_sequence(record.locant_seq),
                    stereo_locants=_complete_single_ring_parent_stereo_locants(mol, record.locant_seq, self._Chem),
                    suffixes=_single_ring_suffix_candidates(mol, record),
                    charge=_single_ring_charge_evidence(mol, ring),
                    estimated_length=len(record.name),
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
        if len(facts) <= _SINGLE_RING_PARENT_RING_FACT_SET_SIZE and not any(
            fact.structural_key and fact.structural_key[0] == 0 for fact in facts
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
                    parent_external_nitrogen_substituent_count=_ring_external_nitrogen_substituent_count(mol, ring),
                    parent_external_oxygen_substituent_count=_ring_external_oxygen_substituent_count(mol, ring),
                    locants=_locants_from_sequence(record.locant_seq),
                    stereo_locants=_complete_single_ring_parent_stereo_locants(mol, record.locant_seq, self._Chem),
                    suffixes=_single_ring_suffix_candidates(mol, record),
                    charge=_single_ring_charge_evidence(mol, ring),
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


def _ring_aromatic_atom_count(mol, ring: Sequence[int]) -> int:
    return sum(1 for idx in ring if mol.GetAtomWithIdx(idx).GetIsAromatic())


def _ring_hetero_atom_count(mol, ring: Sequence[int]) -> int:
    return sum(1 for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6)


def _ring_terminal_hetero_substituent_count(mol, ring: Sequence[int]) -> int:
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


def _parent_substituted_hetero_atom_count(mol, parent_atoms: Sequence[int]) -> int:
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
    ring_set = set(ring)
    count = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in ring_set and neighbor.GetAtomicNum() == 7:
                count += 1
    return count


def _ring_external_oxygen_substituent_count(mol, ring: Sequence[int]) -> int:
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


def _single_ring_charge_evidence(mol, ring: Sequence[int]) -> ChargeEvidence | None:
    charged_atoms = tuple(
        idx
        for idx in sorted(ring)
        if mol.GetAtomWithIdx(idx).GetFormalCharge()
    )
    if not charged_atoms:
        return None
    formal_charge = sum(mol.GetAtomWithIdx(idx).GetFormalCharge() for idx in charged_atoms)
    return ChargeEvidence(
        formal_charge=formal_charge,
        charged_atoms=charged_atoms,
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
        for template, match in _iter_fused_template_matches(
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
            if _should_defer_spiro_core_parent_to_senior_azolone(ctx.mol, candidate.parent_atoms):
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
        for record_order, record in enumerate(ctx.polycyclic_parent_records(self.record_family)):
            name, seq = record[0], record[1]
            bridge_unsaturation_mismatches = int(record[2]) if len(record) > 2 else 0
            if not self._accept_record(name, seq):
                continue
            if self._defer_record(ctx, name, seq):
                continue
            safety = None
            render_cost = 1
            if bridge_unsaturation_mismatches > 0:
                safety = CandidateSafety(
                    opsin_fragility_reasons=(OPSIN_FRAGILE_POLYCYCLIC_BRIDGE_UNSATURATION,)
                )
                render_cost += bridge_unsaturation_mismatches
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


class GenericBicyclicParentRule(_PolycyclicRecordParentRule):
    family = "generic-bicyclic-parent"
    priority = parent_priority("generic-bicyclic-parent")
    record_family = "bicyclic"
    requires_generic_bicyclic_topology = True


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
        name = _exact_retained_parent_candidate_name(template)
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
                    name = _exact_retained_parent_candidate_name(template)
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
                    family = (
                        "exact-biochemical-retained-parent"
                        if template.get("source_family") == "static-exact-retained"
                        else self.family
                    )
                    name = _exact_retained_parent_candidate_name(template)
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
                        estimated_length=len(name),
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
    hexamethylenediamide = HexamethylenediamideRetainedParentRule(Chem)
    small_inorganic = SmallInorganicRetainedParentRule(Chem)
    sydnone = SydnoneRetainedParentRule(Chem)
    phosphanium = PhosphaniumParentRule(Chem)
    phosphanylium_oxide = PhosphanyliumOxideParentRule(Chem)
    phosphanylium_imino = PhosphanyliumIminoParentRule(Chem)
    phosphanylium_ylidene = PhosphanyliumYlideneParentRule(Chem)
    azanium = AzaniumParentRule(Chem)
    neutral_amine = NeutralAmineParentRule(Chem)
    group14_hydride = Group14HydrideParentRule(Chem)
    phosphorus_acid = PhosphorusAcidParentRule(Chem)
    biphenyl_diphosphonite = BiphenylDiphosphoniteParentRule(Chem)
    phosphite_ester = PhosphiteEsterParentRule(Chem)
    charged_organophosphorus_oxo = ChargedOrganophosphorusOxoParentRule(Chem)
    thiooxo_ester = ThiooxoEsterRetainedParentRule(Chem)
    sulfinate = SulfinateRetainedParentRule(Chem)
    phosphorothioic_triamide = PhosphorothioicTriamideRetainedParentRule(Chem)
    dithiophosphate = DithiophosphateRetainedParentRule(Chem)
    organophosphorus_oxide = OrganophosphorusOxideParentRule(Chem)
    phosphorane = PhosphoraneParentRule(Chem)
    aminophosphane = AminophosphaneParentRule(Chem)
    dithiazole = DithiazoleRetainedParentRule(Chem)
    oxazinone = OxazinoneRetainedParentRule(Chem)
    azidobenzylidene_cyclohexanone = AzidobenzylideneCyclohexanoneRetainedParentRule(Chem)
    bis_diethylaminobenzylidene_cyclopentanone = BisDiethylaminobenzylideneCyclopentanoneRetainedParentRule(Chem)
    organic_peroxide = OrganicPeroxideRetainedParentRule(Chem)
    phosphorylcholine = PhosphorylCholineRetainedParentRule(Chem)
    azinobis_benzothiazoline = AzinobisBenzothiazolineRetainedParentRule(Chem)
    organomagnesium_halide = OrganomagnesiumHalideParentRule(Chem)
    organoborane = OrganoboraneParentRule(Chem)
    benziodoxole = BenziodoxoleRetainedParentRule(Chem)
    iodane = IodaneParentRule(Chem)
    iodonium = IodoniumParentRule(Chem)
    pnictogen_hydride = PnictogenHydrideParentRule(Chem)
    oxido_nitroso_azanium = OxidoNitrosoAzaniumRule(Chem)
    oxo_sulfanium = OxoSulfaniumParentRule(Chem)
    n_halo_imine = NHaloImineParentRule(Chem)
    cyclic_nitroimine = CyclicNitroimineParentRule(Chem)
    oxime_sulfonate_nitrile = OximeSulfonateNitrileParentRule(Chem)
    nitrile_oxide = NitrileOxideRetainedParentRule(Chem)
    carbodiimide = CarbodiimideRetainedParentRule(Chem)
    guanidine = GuanidineRetainedParentRule(Chem)
    sulfonohydrazide = SulfonohydrazideRetainedParentRule(Chem)
    acyl_hydrazide = AcylHydrazideRetainedParentRule(Chem)
    acyl_diphosphate = AcylDiphosphateRetainedParentRule(Chem)
    retained_acid_ester = RetainedAcidEsterParentRule(Chem)
    retained_amide = RetainedAmideParentRule(Chem)
    beta_dicarbonyl = BetaDicarbonylParentRule(Chem)
    oxo_carboxylate = OxoCarboxylateRetainedParentRule(Chem)
    boronic_pinacol_ester = BoronicPinacolEsterParentRule(Chem)
    carbamate = CarbamateRetainedParentRule(Chem)
    cyclic_n_carbamate_carboxylate = CyclicNCarbamateCarboxylateRetainedParentRule(Chem)
    sulfonamide = SulfonamideRetainedParentRule(Chem)
    n_substituted_sulfonamide = SulfonamideParentRule(Chem)
    sulfonimidamide = SulfonimidamideParentRule(Chem)
    urea = UreaRetainedParentRule(Chem)
    sydnone_urea_amidate = SydnoneUreaAmidateRetainedParentRule(Chem)
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
    lambda_sulfanone_spiro = LambdaSulfanoneSpiroSubstituentRule(Chem)
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
        RuleSpec("sydnone-retained-parent", parent_priority("senior-functional-exception"), (charged_ring_atom,), sydnone.candidates),
        RuleSpec(
            "bis-sulfonium-sulfide-retained-parent",
            parent_priority("senior-functional-exception"),
            (),
            bis_sulfonium_sulfide.candidates,
        ),
        RuleSpec(
            "methylenebis-dithiocarbamate-retained-parent",
            parent_priority("senior-functional-exception"),
            (),
            methylenebis_dithiocarbamate.candidates,
        ),
        RuleSpec(
            "hexamethylenediamide-retained-parent",
            parent_priority("senior-functional-exception"),
            (),
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
        RuleSpec("neutral-amine-parent", parent_priority("senior-functional-exception"), (acyclic,), neutral_amine.candidates),
        RuleSpec("group14-hydride-parent", parent_priority("senior-functional-exception"), (nonring_neutral_group14,), group14_hydride.candidates),
        RuleSpec("phosphorus-acid-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), phosphorus_acid.candidates),
        RuleSpec("biphenyl-diphosphonite-parent", parent_priority("senior-functional-exception"), (), biphenyl_diphosphonite.candidates),
        RuleSpec("phosphite-ester-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), phosphite_ester.candidates),
        RuleSpec(
            "charged-organophosphorus-oxo-parent",
            parent_priority("senior-functional-exception"),
            (nonring_neutral_phosphorus,),
            charged_organophosphorus_oxo.candidates,
        ),
        RuleSpec("thiooxo-ester-retained-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), thiooxo_ester.candidates),
        RuleSpec("sulfinate-retained-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), sulfinate.candidates),
        RuleSpec("phosphorothioic-triamide-retained-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), phosphorothioic_triamide.candidates),
        RuleSpec("dithiophosphate-retained-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), dithiophosphate.candidates),
        RuleSpec("acyl-diphosphate-retained-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), acyl_diphosphate.candidates),
        RuleSpec("organophosphorus-oxide-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), organophosphorus_oxide.candidates),
        RuleSpec("phosphorane-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), phosphorane.candidates),
        RuleSpec("aminophosphane-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), aminophosphane.candidates),
        RuleSpec("dithiazole-retained-parent", parent_priority("senior-functional-exception"), (), dithiazole.candidates),
        RuleSpec("oxazinone-retained-parent", parent_priority("senior-functional-exception"), (), oxazinone.candidates),
        RuleSpec(
            "azidobenzylidene-cyclohexanone-retained-parent",
            parent_priority("senior-functional-exception"),
            (),
            azidobenzylidene_cyclohexanone.candidates,
        ),
        RuleSpec(
            "bis-diethylaminobenzylidene-cyclopentanone-retained-parent",
            parent_priority("senior-functional-exception"),
            (),
            bis_diethylaminobenzylidene_cyclopentanone.candidates,
        ),
        RuleSpec(
            "organic-peroxide-retained-parent",
            parent_priority("senior-functional-exception"),
            (),
            organic_peroxide.candidates,
        ),
        RuleSpec(
            "phosphorylcholine-retained-parent",
            parent_priority("senior-functional-exception"),
            (),
            phosphorylcholine.candidates,
        ),
        RuleSpec(
            "azinobis-benzothiazoline-retained-parent",
            parent_priority("senior-functional-exception"),
            (),
            azinobis_benzothiazoline.candidates,
        ),
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
        RuleSpec("cyclic-nitroimine-parent", parent_priority("senior-functional-exception"), (), cyclic_nitroimine.candidates),
        RuleSpec("oxime-sulfonate-nitrile-parent", parent_priority("senior-functional-exception"), (), oxime_sulfonate_nitrile.candidates),
        RuleSpec("nitrile-oxide-retained-parent", parent_priority("senior-functional-exception"), (), nitrile_oxide.candidates),
        RuleSpec("carbodiimide-retained-parent", parent_priority("senior-functional-exception"), (imino_skeleton_signal,), carbodiimide.candidates),
        RuleSpec("guanidine-retained-parent", parent_priority("senior-functional-exception"), (acyclic, imino_skeleton_signal), guanidine.candidates),
        RuleSpec("sulfonohydrazide-retained-parent", parent_priority("senior-functional-exception"), (imino_skeleton_signal,), sulfonohydrazide.candidates),
        RuleSpec("acyl-hydrazide-retained-parent", parent_priority("senior-functional-exception"), (imino_skeleton_signal,), acyl_hydrazide.candidates),
        RuleSpec("retained-acid-ester-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), retained_acid_ester.candidates),
        RuleSpec("retained-amide-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), retained_amide.candidates),
        RuleSpec("beta-dicarbonyl-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), beta_dicarbonyl.candidates),
        RuleSpec("oxo-carboxylate-retained-parent", parent_priority("senior-functional-exception"), (acyclic, functional_skeleton_signal), oxo_carboxylate.candidates),
        RuleSpec("boronic-pinacol-ester-parent", parent_priority("senior-functional-exception"), (min_ring_count(1),), boronic_pinacol_ester.candidates),
        RuleSpec("carbamate-retained-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), carbamate.candidates),
        RuleSpec(
            "cyclic-n-carbamate-carboxylate-retained-parent",
            parent_priority("senior-functional-exception"),
            (min_ring_count(1), functional_skeleton_signal),
            cyclic_n_carbamate_carboxylate.candidates,
        ),
        RuleSpec("sulfonamide-retained-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), sulfonamide.candidates),
        RuleSpec("sulfonamide-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), n_substituted_sulfonamide.candidates),
        RuleSpec("sulfonimidamide-parent", parent_priority("senior-functional-exception"), (imino_skeleton_signal,), sulfonimidamide.candidates),
        RuleSpec("urea-retained-parent", parent_priority("senior-functional-exception"), (functional_skeleton_signal,), urea.candidates),
        RuleSpec("sydnone-urea-amidate-retained-parent", parent_priority("senior-functional-exception"), (charged_ring_atom, functional_skeleton_signal), sydnone_urea_amidate.candidates),
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
