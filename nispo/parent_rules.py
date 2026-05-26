from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from itertools import combinations
from functools import lru_cache

from .branch_rules import branch_prefix
from .branch.hetero_atoms import _imine_carbon_ylidene
from .constants import HALO, MULT, ROOTS
from .cycle_utils import ordered_ring, ring_system_extends_core
from .formatting import format_prefixes as _format_prefixes
from .formatting import loc_unsat
from .molecule_cache import mol_cache_get, mol_cache_set
from .fused_template_families import (
    curated_fused_templates as _curated_fused_templates,
    generated_fusion_preference as _generated_fusion_preference,
    hydrocarbon_replacement_parent_templates_for_graph as _hydrocarbon_replacement_parent_templates_for_graph,
)
from .ring_resource_rules import resource_ring_parent_candidates as _resource_ring_parent_candidates
from .ring_rules import (
    adjust_ring_parent_for_charge as _adjust_ring_parent_for_charge,
    ring_parent_candidates as _ring_parent_candidates,
)
from .rule_dispatch import (
    RuleSpec,
    acyclic,
    acyclic_preference_signal,
    charged_ring_atom,
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
from .template_access import template_record_locants, template_record_mol, template_record_name


def _parenthesized_if_complex(name: str) -> str:
    if name.startswith("(") and name.endswith(")"):
        return name
    return f"({name})" if "-" in name or name.startswith("(") else name


class AcyclicParentRule:
    family = "acyclic-parent"
    priority = parent_priority("acyclic-parent")
    acyclic_only = True

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() != 0:
            return
        name = ctx.acyclic_name()
        if name:
            yield NameCandidate(name, self.family, self.priority)


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
            prefix = branch_prefix(mol, nidx, sidx)
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
            prefix = branch_prefix(mol, nidx, pidx)
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
            prefix = branch_prefix(mol, nbr_idx, nidx)
            if not prefix:
                return
            names.append(prefix)
        name = _azanium_name(names)
        if name:
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


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
        halo_names = []
        hydroxy_count = 0
        for nidx in neighbors:
            neighbor = mol.GetAtomWithIdx(nidx)
            symbol = neighbor.GetSymbol()
            if symbol == "C":
                prefix = branch_prefix(mol, nidx, cidx)
                if not prefix:
                    return
                carbon_names.append(prefix)
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
        parent = GROUP14_HYDRIDE_PARENTS[center.GetSymbol()]
        if hydroxy_count:
            name = _group14_hydroxyhydride_name(carbon_names, parent, hydroxy_count)
        elif halo_names:
            name = _group14_halohydride_name(carbon_names, halo_names, parent)
        else:
            name = _group14_hydride_name(carbon_names, parent)
        if name:
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


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
        prefix = branch_prefix(mol, carbon_neighbors[0], pidx)
        if not prefix:
            return
        if _needs_substituent_parentheses(prefix):
            prefix = f"({prefix})"
        name = prefix + parent
        yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


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
        prefix = branch_prefix(mol, carbon_neighbors[0], midx)
        if not prefix:
            return
        if _needs_substituent_parentheses(prefix):
            prefix = f"({prefix})"
        name = f"{prefix}magnesium {halogen_names[0]}"
        yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


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
            prefix = branch_prefix(mol, nidx, pidx)
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
            prefix = branch_prefix(mol, organic[0], nidx)
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
            prefix = branch_prefix(mol, heavy[0], sidx)
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
    aryl = branch_prefix(mol, aryl_idx, sulfur_idx)
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
        ring = branch_prefix(mol, carbon_idx, nitrogen_idx)
        if ring.endswith("yl"):
            return ring[: -len("yl")] + "ylidene"
        return ""
    substituents = [nbr.GetIdx() for nbr in carbon.GetNeighbors() if nbr.GetIdx() != nitrogen_idx]
    if len(substituents) == 1:
        name = branch_prefix(mol, substituents[0], carbon_idx)
        if name == "1-oxo-2-oxaethyl":
            return "carboxymethylidene"
        return name + "methylidene" if name and not name.endswith("ylidene") else name
    if len(substituents) == 2:
        names = [branch_prefix(mol, idx, carbon_idx) for idx in substituents]
        if any(not name for name in names):
            return ""
        if names.count("methyl") == 1:
            other = names[1] if names[0] == "methyl" else names[0]
            other_text = _parenthesized_if_complex(other)
            return f"1-{other_text}ethylidene"
    return ""


class AcyclicSkeletonWithRingSubstituentsRule:
    family = "acyclic-skeleton-with-ring-substituents"
    priority = parent_priority("acyclic-with-ring-substituents")
    min_ring_count = 1

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() == 0:
            return
        name = ctx.acyclic_name(allow_ring_atoms=False)
        if name:
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class PreferredAcyclicSkeletonWithRingSubstituentsRule(AcyclicSkeletonWithRingSubstituentsRule):
    family = "preferred-acyclic-skeleton-with-ring-substituents"
    priority = parent_priority("polycyclic-functional-skeleton")
    requires_acyclic_preference_signal = True

    def candidates(self, ctx):
        profile = ctx.profile
        if profile.ring_count == 0 or not profile.may_prefer_acyclic_skeleton:
            return
        name = ctx.acyclic_name(allow_ring_atoms=False)
        if name and _prefer_acyclic_skeleton_name(name):
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class ChargedRingAcyclicSkeletonWithRingSubstituentsRule(AcyclicSkeletonWithRingSubstituentsRule):
    family = "charged-ring-acyclic-skeleton-with-ring-substituents"
    priority = parent_priority("polycyclic-functional-skeleton")
    requires_charged_ring_atom = True

    def candidates(self, ctx):
        profile = ctx.profile
        if profile.ring_count == 0 or not profile.has_charged_ring_atom:
            return
        name = ctx.acyclic_name(allow_ring_atoms=False)
        if (
            name
            and "ium" in name
            and not any(term in name for term in ("bicyclo[", "tricyclo[", "spiro["))
        ):
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


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
        ):
            return
        name = ctx.acyclic_name(allow_ring_atoms=False)
        if name and _prefer_acyclic_skeleton_name(name):
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


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
        ):
            return
        name = ctx.acyclic_name(allow_ring_atoms=False)
        if name and "imino" in name and _prefer_acyclic_skeleton_name(name):
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


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
        ):
            return
        name = ctx.acyclic_name(allow_ring_atoms=False)
        if (
            name
            and "carbonyl" in name
            and "buta-1,3-diene" in name
            and _prefer_acyclic_skeleton_name(name)
        ):
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class WholeMoleculeRetainedParentRule:
    family = "whole-molecule-retained-parent"
    priority = parent_priority("whole-molecule-retained-parent")
    min_ring_count = 1

    def __init__(self, Chem):
        self._Chem = Chem

    def _templates(self) -> tuple[dict, ...]:
        return _curated_exact_parent_templates(self._Chem)

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() == 0:
            return
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        for template in ctx.candidate_templates(self._templates(), exact_atom_count=True):
            name = _standalone_whole_molecule_retained_name(template)
            if not name:
                continue
            if template_record_mol(template).GetNumAtoms() != atom_count:
                continue
            if template.get("stereochemical") and not _has_stereo_evidence(mol, self._Chem):
                continue
            for match in ctx.template_matches(template):
                if set(match) != atom_set or _match_has_extra_core_bonds(mol, template, match):
                    continue
                yield NameCandidate(
                    name=name,
                    family=_whole_molecule_retained_candidate_family(name),
                    priority=self.priority,
                    sort_key=(len(name), template.get("source_family", ""), name),
                )


def _standalone_whole_molecule_retained_name(template: dict) -> str:
    name = template_record_name(template)
    source_file = template.get("source_file", "")
    if source_file == "naturalProducts.xml" and "'" not in name and not name.endswith("ylic"):
        return name
    if name in {"benzyn", "tropylium"}:
        return name
    return ""


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

    def _candidate_sort_key(self, template: dict, match: Sequence[int], name: str) -> tuple:
        return (
            *_fused_candidate_preference(name),
            _template_source_rank(template),
            _hydrogenated_template_penalty(template_record_name(template)),
            -len(match),
            name,
        )

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        for template, match in _iter_fused_template_matches(
            ctx, self._template_iter(ctx), self._Chem, reject_extra_core_bonds=True
        ):
            name = _name_fused_core_match(mol, template, match)
            if name:
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=self._candidate_sort_key(template, match, name),
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

    def _candidate_sort_key(self, template: dict, match: Sequence[int], name: str) -> tuple:
        return (_generated_fusion_preference(template_record_name(template)), *_fused_candidate_preference(name), -len(match), name)


class SingleRingParentRule:
    family = "single-ring-parent"
    priority = parent_priority("single-ring-parent")
    min_ring_count = 1

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 1:
            return
        for ring in mol.GetRingInfo().AtomRings():
            for name, stereo_prefix in _named_ring_instance_names(mol, tuple(ring), self._Chem):
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=_single_ring_candidate_sort_key(name),
                    stereo_prefix=stereo_prefix,
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
        candidates = []
        for template, match in _iter_fused_template_matches(
            ctx,
            ctx.generated_fused_templates,
            self._Chem,
            topology_query=True,
            require_one_ring_system=True,
        ):
            for parent_name, hydro_key in _partial_hydro_parent_names(mol, template, match):
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


class NaturalProductTopologyParentRule(FusedParentRule):
    family = "natural-product-topology-parent"
    priority = parent_priority("exact-retained-parent")
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
            _substituent_locant_tuple(name),
            len(name),
            name,
        )

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 4:
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


class SkeletalReplacementFusedParentRule(FusedParentRule):
    family = "skeletal-replacement-fused-parent"
    priority = parent_priority("skeletal-replacement-parent")
    min_ring_system_ring_count = 3

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        yielded = 0
        for atom_set, feature in zip(ctx.ring_systems, ctx.features["ring_systems"], strict=True):
            if feature["ring_count"] != 3 or set(feature["elements"]) <= {"C"}:
                continue
            key = (feature["atom_count"], feature["ring_count"], feature.get("graph_signature", ""))
            for template in _hydrocarbon_replacement_parent_templates_for_graph(*key, self._Chem):
                query = _skeletal_replacement_query(template_record_mol(template), self._Chem)
                if query.GetNumAtoms() != len(atom_set):
                    continue
                for match in ctx.substruct_matches(query):
                    if set(match) != set(atom_set):
                        continue
                    parent_name = _skeletal_replacement_parent_name(mol, template, match)
                    if not parent_name:
                        continue
                    name = _name_fused_core_match(mol, template, match, parent_name=parent_name)
                    if not name:
                        continue
                    yielded += 1
                    yield NameCandidate(
                        name=name,
                        family=self.family,
                        priority=self.priority,
                        sort_key=(
                            _skeletal_replacement_sort_key(mol, match, parent_name),
                            _substituent_locant_tuple(name),
                            len(name),
                            name,
                        ),
                    )
                    if yielded >= 12:
                        return


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

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        priority = self._candidate_priority(mol)
        if priority is None:
            return
        for record_order, (name, seq) in enumerate(ctx.polycyclic_parent_records(self.record_family)):
            if not self._accept_record(name, seq):
                continue
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=priority,
                sort_key=(record_order, len(name), name),
                stereo_prefix=_polycyclic_stereo_prefix(mol, seq, self._Chem),
            )


class GenericBicyclicParentRule(_PolycyclicRecordParentRule):
    family = "generic-bicyclic-parent"
    priority = parent_priority("generic-bicyclic-parent")
    record_family = "bicyclic"
    requires_generic_bicyclic_topology = True


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
        ring_count = mol.GetRingInfo().NumRings()
        aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
        if not aromatic and ring_count < 3:
            return None
        return self.priority if aromatic else parent_priority("generic-polycyclic-parent")


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
        return (len(template_record_name(template)), template.get("source_family", ""), template_record_name(template))

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() == 0 or not ctx.features["all_atoms_in_one_ring_system"]:
            return
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        for template in self._template_iter(ctx):
            if template.get("stereochemical") and not _has_stereo_evidence(mol, self._Chem):
                continue
            if template_record_mol(template).GetNumAtoms() != atom_count:
                continue
            for match in ctx.template_matches(template):
                if set(match) == atom_set and not _match_has_extra_core_bonds(mol, template, match):
                    yield NameCandidate(
                        name=template_record_name(template),
                        family=self.family,
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


def parent_rules(Chem) -> tuple:
    sulfanium = SulfaniumParentRule(Chem)
    phosphanium = PhosphaniumParentRule(Chem)
    azanium = AzaniumParentRule(Chem)
    group14_hydride = Group14HydrideParentRule(Chem)
    phosphorus_acid = PhosphorusAcidParentRule(Chem)
    organomagnesium_halide = OrganomagnesiumHalideParentRule(Chem)
    pnictogen_hydride = PnictogenHydrideParentRule(Chem)
    oxido_nitroso_azanium = OxidoNitrosoAzaniumRule(Chem)
    oxo_sulfanium = OxoSulfaniumParentRule(Chem)
    n_halo_imine = NHaloImineParentRule(Chem)
    guanidine = GuanidineRetainedParentRule(Chem)
    sulfonohydrazide = SulfonohydrazideRetainedParentRule(Chem)
    acyclic_parent = AcyclicParentRule()
    imino_functional_acyclic = IminoFunctionalAcyclicSkeletonRule()
    conjugated_carbonyl_acyclic = ConjugatedCarbonylAcyclicSkeletonRule()
    whole_molecule_retained = WholeMoleculeRetainedParentRule(Chem)
    exact_retained = ExactRetainedParentRule(Chem)
    preferred_bicyclic = PreferredGenericBicyclicParentRule(Chem)
    fused = FusedParentRule(Chem)
    hydro_fused = HydroFusedParentRule(Chem)
    oxido_fused = OxidoFusedParentRule(Chem)
    generic_bicyclic = GenericBicyclicParentRule(Chem)
    generic_spiro = GenericSpiroParentRule(Chem)
    single_ring = SingleRingParentRule(Chem)
    generated_exact = GeneratedExactParentRule(Chem)
    generated_fused = GeneratedFusedParentRule(Chem)
    generated_hydro_fused = GeneratedHydroFusedParentRule(Chem)
    generated_partial_hydro = GeneratedPartialHydroFusedParentRule(Chem)
    natural_product_topology = NaturalProductTopologyParentRule(Chem)
    skeletal_replacement = SkeletalReplacementFusedParentRule(Chem)
    kekulized_polycyclic = KekulizedGenericPolycyclicParentRule(Chem)
    preferred_acyclic = PreferredAcyclicSkeletonWithRingSubstituentsRule()
    charged_ring_acyclic = ChargedRingAcyclicSkeletonWithRingSubstituentsRule()
    functional_acyclic = FunctionalAcyclicSkeletonWithRingSubstituentsRule()
    acyclic_with_ring_substituents = AcyclicSkeletonWithRingSubstituentsRule()

    return (
        RuleSpec("sulfanium-parent", parent_priority("senior-functional-exception"), (nonring_positive_sulfur,), sulfanium.candidates),
        RuleSpec("phosphanium-parent", parent_priority("senior-functional-exception"), (nonring_positive_phosphorus,), phosphanium.candidates),
        RuleSpec("azanium-parent", parent_priority("senior-functional-exception"), (nonring_positive_nitrogen,), azanium.candidates),
        RuleSpec("group14-hydride-parent", parent_priority("senior-functional-exception"), (nonring_neutral_group14,), group14_hydride.candidates),
        RuleSpec("phosphorus-acid-parent", parent_priority("senior-functional-exception"), (nonring_neutral_phosphorus,), phosphorus_acid.candidates),
        RuleSpec("organomagnesium-halide-parent", parent_priority("senior-functional-exception"), (acyclic,), organomagnesium_halide.candidates),
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
            (functional_skeleton_signal, acyclic_preference_signal, no_charged_ring_atom),
            conjugated_carbonyl_acyclic.candidates,
        ),
        RuleSpec(
            "preferred-acyclic-skeleton-with-ring-substituents",
            parent_priority("polycyclic-functional-skeleton"),
            (acyclic_preference_signal,),
            preferred_acyclic.candidates,
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
            (min_ring_count(1),),
            whole_molecule_retained.candidates,
        ),
        RuleSpec("exact-retained-parent", parent_priority("exact-retained-parent"), (min_ring_count(1), exact_ring_system), exact_retained.candidates),
        RuleSpec(
            "preferred-generic-bicyclic-parent",
            parent_priority("preferred-generic-bicyclic"),
            (min_ring_system_ring_count(2), generic_bicyclic_topology),
            preferred_bicyclic.candidates,
        ),
        RuleSpec("fused-retained-parent", parent_priority("retained-fused-parent"), (min_ring_count(2),), fused.candidates),
        RuleSpec("hydro-fused-retained-parent", parent_priority("hydro-retained-fused-parent"), (min_ring_count(2), min_ring_system_ring_count(2)), hydro_fused.candidates),
        RuleSpec(
            "oxido-fused-retained-parent",
            parent_priority("charged-retained-fused-parent"),
            (min_ring_count(2), min_ring_system_ring_count(2), charged_ring_atom),
            oxido_fused.candidates,
        ),
        RuleSpec("generic-bicyclic-parent", parent_priority("generic-bicyclic-parent"), (generic_bicyclic_topology,), generic_bicyclic.candidates),
        RuleSpec("generic-spiro-parent", parent_priority("generic-spiro-parent"), (min_ring_system_ring_count(2), spiro_topology), generic_spiro.candidates),
        RuleSpec("single-ring-parent", parent_priority("single-ring-parent"), (min_ring_count(1),), single_ring.candidates),
        RuleSpec("generated-exact-parent", parent_priority("generated-parent"), (min_ring_count(1), exact_ring_system), generated_exact.candidates),
        RuleSpec("generated-fused-parent", parent_priority("generated-parent"), (min_ring_system_ring_count(2),), generated_fused.candidates),
        RuleSpec("generated-hydro-fused-parent", parent_priority("generated-parent"), (min_ring_system_ring_count(2),), generated_hydro_fused.candidates),
        RuleSpec("generated-partial-hydro-fused-parent", parent_priority("generated-partial-hydro-parent"), (min_ring_system_ring_count(2),), generated_partial_hydro.candidates),
        RuleSpec("natural-product-topology-parent", parent_priority("exact-retained-parent"), (min_ring_system_ring_count(4),), natural_product_topology.candidates),
        RuleSpec("skeletal-replacement-fused-parent", parent_priority("skeletal-replacement-parent"), (min_ring_system_ring_count(3),), skeletal_replacement.candidates),
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
        RuleSpec("acyclic-skeleton-with-ring-substituents", parent_priority("acyclic-with-ring-substituents"), (min_ring_count(1),), acyclic_with_ring_substituents.candidates),
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
    _name_acyclic_retained_core_match,
    _subtree_has_cumulated_diene,
    _skeletal_replacement_query,
    _skeletal_replacement_parent_name,
    _skeletal_replacement_sort_key,
    _spiro_norbornene_locant_penalty,
    _skeletal_replacement_saturation_penalty,
    _has_nonaromatic_core_imine,
    _locant_sort_key,
    _SKELETAL_REPLACEMENT_PREFIXES,
    _has_stereo_evidence,
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
    _has_n_oxide_on_match,
    _name_fused_core_match,
    _name_natural_product_topology_match,
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
)
