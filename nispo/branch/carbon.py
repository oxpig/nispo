from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from functools import lru_cache

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from ..branch import BranchRuleSet, BranchRuleSpec
from ..constants import HALO, MULT, ROOTS
from ..formatting import format_prefixes as _format_prefixes
from ..molecule_cache import mol_cache_dict
from ..substituent_resource_rules import resource_substituent_prefix as _resource_substituent_prefix
from .functional_groups import (
    _acyl_halide_prefix,
    _carbonyl_substituent_prefix,
    _is_carbonyl_oxygen_idx,
    _is_carboxylate_carbon,
    _is_isocyano_carbon,
    _is_nitrile_carbon,
    _is_nitro_nitrogen,
    _is_tbdms_group,
    _is_terminal_carbonyl_oxygen,
    _is_trimethylsilyl_group,
    _group14_prefix,
    _metal_prefix,
    _phosphorus_prefix,
    _selenium_prefix,
    _sulfur_prefix,
    collect_subtree,
    is_carbonyl_carbon,
    rooted_paths,
    substituent_from_root,
    substituent_locants,
)
from .functional_suffixes import suffix_prefix_term
from .hetero_atoms import _nitrogen_prefix, _oxygen_prefix
from .ring_substituents import (
    _alkynyl_substituent_prefix,
    _aryloxy_methyl_prefix,
    _di_ring_methyl_prefix,
    _dicyanoethenyl_prefix,
    _difluoro_substituted_methyl_prefix,
    _iminomethyl_prefix,
    _ring_alkyl_prefix,
    _ring_alkenyl_prefix,
    _ring_substituent_prefix,
    _ring_ethynyl_prefix,
    _sulfonamidomethyl_prefix,
)


def branch_prefix(mol, idx: int, parent_idx: int) -> str:
    from .. import branch_rules

    return branch_rules.branch_prefix(mol, idx, parent_idx)


def _parenthesize_complex_prefix(name: str) -> str:
    if "-" in name or name[:1].isdigit() or name.startswith(("N-", "S-", "O-")):
        return f"({name})"
    return name


def _resource_prefix(mol, idx: int, parent_idx: int) -> str:
    return _resource_substituent_prefix(mol, idx, parent_idx, Chem)


def _carbon_prefix(mol, idx: int, parent_idx: int) -> str:
    if _is_carbonyl_oxygen_idx(mol, idx):
        return ""
    return substituent_from_root(mol, idx, parent_idx)


def _with_root_carbon_isotope(mol, idx: int, name: str) -> str:
    if not name:
        return name
    atom = mol.GetAtomWithIdx(idx)
    isotope = atom.GetIsotope()
    if atom.GetSymbol() == "C" and isotope:
        return f"{name}-{isotope}C"
    return name


def _halogen_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
        return ""
    symbol = atom.GetSymbol()
    name = HALO.get(symbol, "")
    if not name:
        return ""
    isotope = atom.GetIsotope()
    return f"[{isotope}{symbol}]{name}" if isotope else name


def _carbon_branch_prefix(mol, idx: int, parent_idx: int) -> str:
    """Dispatch carbon substituent rules from local structure before rendering.

    Carbon branches are the dominant branch-rendering hot path. The previous
    BranchRuleSet tried every carbon functional family in order for almost every
    carbon substituent. This keeps that order, but uses cheap local bond classes
    to select the only families that can possibly match.
    """

    terminal_methyl = _terminal_methyl_prefix(mol, idx, parent_idx)
    if terminal_methyl:
        return terminal_methyl
    linear_alkyl = _linear_alkyl_prefix(mol, idx, parent_idx)
    if linear_alkyl:
        return _with_root_carbon_isotope(mol, idx, linear_alkyl)
    if _is_nitrile_carbon(mol, idx):
        return _with_root_carbon_isotope(mol, idx, suffix_prefix_term("nitrile", "cyano"))
    deuterated_methyl = _deuterated_methyl_prefix(mol, idx, parent_idx)
    if deuterated_methyl:
        return deuterated_methyl
    fluorinated_alkyl = _branched_fluoroalkyl_prefix(mol, idx, parent_idx)
    if fluorinated_alkyl:
        return _with_root_carbon_isotope(mol, idx, fluorinated_alkyl)
    substituted_alkyl = _simple_substituted_alkyl_prefix(mol, idx, parent_idx)
    if substituted_alkyl:
        return _with_root_carbon_isotope(mol, idx, substituted_alkyl)

    atom = mol.GetAtomWithIdx(idx)
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    parent_order = parent_bond.GetBondType() if parent_bond is not None else None
    neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]

    if parent_order == Chem.BondType.SINGLE:
        name = _resource_prefix(mol, idx, parent_idx)
        if _prefer_resource_carbon_prefix(name):
            return _with_root_carbon_isotope(mol, idx, name)
        has_double_c = _has_neighbor_bond(mol, idx, neighbors, "C", Chem.BondType.DOUBLE)
        if has_double_c:
            for renderer in (
                _ketenyl_prefix,
                _enone_substituent_prefix,
                _enal_substituent_prefix,
                _haloethenyl_prefix,
            ):
                name = renderer(mol, idx, parent_idx)
                if name:
                    return _with_root_carbon_isotope(mol, idx, name)
        if _is_isocyano_carbon(mol, idx):
            return _with_root_carbon_isotope(mol, idx, "isocyano")
        if _is_carboxylate_carbon(mol, idx):
            return _with_root_carbon_isotope(mol, idx, suffix_prefix_term("oate", "carboxylato"))
        name = _protected_alkyl_prefix(mol, idx, parent_idx)
        if name:
            return _with_root_carbon_isotope(mol, idx, name)
        name = _boc_aminoalkyl_prefix(mol, idx, parent_idx)
        if name:
            return _with_root_carbon_isotope(mol, idx, name)
        if _has_neighbor_bond(mol, idx, neighbors, "O", Chem.BondType.DOUBLE):
            name = _acyl_halide_prefix(mol, idx)
            if name:
                return _with_root_carbon_isotope(mol, idx, name)
            name = _carbonyl_substituent_prefix(mol, idx, parent_idx)
            if name:
                return _with_root_carbon_isotope(mol, idx, name)
        if _has_neighbor_bond(mol, idx, neighbors, "N", Chem.BondType.DOUBLE):
            name = _iminomethyl_prefix(mol, idx, parent_idx)
            if name:
                return _with_root_carbon_isotope(mol, idx, name)
        if _has_neighbor_bond(mol, idx, neighbors, "N", Chem.BondType.SINGLE):
            name = _sulfonamidomethyl_prefix(mol, idx, parent_idx)
            if name:
                return _with_root_carbon_isotope(mol, idx, name)
        if _has_neighbor_bond(mol, idx, neighbors, "C", Chem.BondType.TRIPLE):
            name = _ring_ethynyl_prefix(mol, idx, parent_idx)
            if name:
                return _with_root_carbon_isotope(mol, idx, name)
            name = _alkynyl_substituent_prefix(mol, idx, parent_idx)
            if name:
                return _with_root_carbon_isotope(mol, idx, name)
        if _has_neighbor_bond(mol, idx, neighbors, "C", Chem.BondType.DOUBLE):
            name = _ring_alkenyl_prefix(mol, idx, parent_idx)
            if name:
                return _with_root_carbon_isotope(mol, idx, name)
            name = _dicyanoethenyl_prefix(mol, idx, parent_idx)
            if name:
                return _with_root_carbon_isotope(mol, idx, name)
        if _has_neighbor_bond(mol, idx, neighbors, "O", Chem.BondType.SINGLE):
            name = _mesyloxymethyl_prefix(mol, idx, parent_idx)
            if name:
                return _with_root_carbon_isotope(mol, idx, name)
            name = _aryloxy_methyl_prefix(mol, idx, parent_idx)
            if name:
                return _with_root_carbon_isotope(mol, idx, name)
        name = _di_ring_methyl_prefix(mol, idx, parent_idx)
        if name:
            return _with_root_carbon_isotope(mol, idx, name)
        name = _difluoro_substituted_methyl_prefix(mol, idx, parent_idx)
        if name:
            return _with_root_carbon_isotope(mol, idx, name)
        if mol.GetRingInfo().NumRings() and _subtree_has_ring(mol, idx, parent_idx):
            name = _ring_alkyl_prefix(mol, idx, parent_idx)
            if name:
                return _with_root_carbon_isotope(mol, idx, name)
    elif parent_order == Chem.BondType.DOUBLE:
        name = _alkylidene_prefix(mol, idx, parent_idx)
        if name:
            return _with_root_carbon_isotope(mol, idx, name)
    else:
        if _is_isocyano_carbon(mol, idx):
            return _with_root_carbon_isotope(mol, idx, "isocyano")
        if _is_carboxylate_carbon(mol, idx):
            return _with_root_carbon_isotope(mol, idx, suffix_prefix_term("oate", "carboxylato"))

    name = _resource_prefix(mol, idx, parent_idx)
    if name:
        return _with_root_carbon_isotope(mol, idx, name)
    return _with_root_carbon_isotope(mol, idx, _carbon_prefix(mol, idx, parent_idx))


def _terminal_methyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return ""
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if _path_has_isotopic_hydrogen(mol, (idx,)):
        return ""
    if any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
        return ""
    return "methyl"


def _linear_alkyl_prefix(mol, idx: int, parent_idx: int) -> str:
    chain = _linear_carbon_chain(mol, idx, parent_idx, max_len=max(ROOTS))
    if len(chain) < 2:
        return ""
    if _path_has_isotopic_hydrogen(mol, chain) or _path_has_assigned_stereo(mol, chain):
        return ""
    root = ROOTS.get(len(chain))
    return root + "yl" if root else ""


def _path_has_assigned_stereo(mol, atom_indices: Sequence[int]) -> bool:
    atom_set = set(atom_indices)
    for atom_idx in atom_indices:
        if mol.GetAtomWithIdx(atom_idx).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return True
    return any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        and bond.GetBeginAtomIdx() in atom_set
        and bond.GetEndAtomIdx() in atom_set
        for bond in mol.GetBonds()
    )


def _prefer_resource_carbon_prefix(name: str) -> bool:
    return _prefer_resource_prefix(name)


def _branched_fluoroalkyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atoms = collect_subtree(mol, idx, parent_idx)
    if len(atoms) > 18:
        return ""
    if any(mol.GetAtomWithIdx(atom_idx).GetSymbol() not in {"C", "F"} for atom_idx in atoms):
        return ""
    carbons = {atom_idx for atom_idx in atoms if mol.GetAtomWithIdx(atom_idx).GetSymbol() == "C"}
    if len(carbons) < 2:
        return ""
    for carbon_idx in carbons:
        atom = mol.GetAtomWithIdx(carbon_idx)
        if atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetTotalNumHs() != 0:
            return ""
    leaves = [
        carbon_idx
        for carbon_idx in carbons
        if carbon_idx == idx
        or sum(1 for nbr in mol.GetAtomWithIdx(carbon_idx).GetNeighbors() if nbr.GetIdx() in carbons) <= 1
    ]
    candidates = []
    for leaf in leaves:
        if leaf == idx:
            continue
        path = tuple(Chem.rdmolops.GetShortestPath(mol, idx, leaf))
        if not path or any(atom_idx not in carbons for atom_idx in path):
            continue
        name = _fluoroalkyl_path_name(mol, path, parent_idx, atoms)
        if name:
            candidates.append((len(path), -len(name), name))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][2]


def _fluoroalkyl_path_name(
    mol,
    path: Sequence[int],
    parent_idx: int,
    atoms: set[int],
) -> str:
    path_set = set(path)
    used = set(path)
    for left, right in zip(path, path[1:]):
        bond = mol.GetBondBetweenAtoms(left, right)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
    fluoro_locs: list[str] = []
    trifluoromethyl_locs: list[str] = []
    for pos, carbon_idx in enumerate(path, start=1):
        allowed_path_neighbors = set()
        if pos > 1:
            allowed_path_neighbors.add(path[pos - 2])
        if pos < len(path):
            allowed_path_neighbors.add(path[pos])
        if pos == 1:
            allowed_path_neighbors.add(parent_idx)
        for nbr in mol.GetAtomWithIdx(carbon_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in allowed_path_neighbors:
                continue
            bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return ""
            if nbr.GetSymbol() == "F" and nidx in atoms:
                fluoro_locs.append(str(pos))
                used.add(nidx)
                continue
            if nbr.GetSymbol() == "C" and nidx in atoms and nidx not in path_set:
                branch_atoms = _terminal_trifluoromethyl_atoms(mol, nidx, carbon_idx, atoms)
                if not branch_atoms:
                    return ""
                trifluoromethyl_locs.append(str(pos))
                used.update(branch_atoms)
                continue
            if nidx != parent_idx and nbr.GetAtomicNum() > 1:
                return ""
    if used != atoms:
        return ""
    root = ROOTS.get(len(path))
    if not root:
        return ""
    prefixes = []
    fluoro = _locanted_multiplier_prefix(fluoro_locs, "fluoro")
    if fluoro:
        prefixes.append(fluoro)
    trifluoromethyl = _locanted_multiplier_prefix(trifluoromethyl_locs, "trifluoromethyl", complex_term=True)
    if trifluoromethyl:
        prefixes.append(trifluoromethyl)
    if not prefixes:
        return ""
    return "-".join(prefixes + [root + "yl"])


def _terminal_trifluoromethyl_atoms(
    mol,
    carbon_idx: int,
    parent_idx: int,
    atoms: set[int],
) -> set[int]:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing() or carbon.GetTotalNumHs() != 0:
        return set()
    fluorines = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if nbr.GetSymbol() == "F" and nidx in atoms and bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            fluorines.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return set()
    if len(fluorines) != 3:
        return set()
    return {carbon_idx, *fluorines}


def _locanted_multiplier_prefix(locs: Sequence[str], term: str, *, complex_term: bool = False) -> str:
    if not locs:
        return ""
    ordered = sorted(locs, key=lambda value: int(value))
    locant_text = ",".join(ordered)
    if len(ordered) == 1:
        return f"{locant_text}-({term})" if complex_term else f"{locant_text}-{term}"
    multiplier = {2: "bis", 3: "tris", 4: "tetrakis"}.get(len(ordered), MULT.get(len(ordered), ""))
    if complex_term:
        return f"{locant_text}-{multiplier}({term})"
    return f"{locant_text}-{MULT.get(len(ordered), '')}{term}"


def _simple_substituted_alkyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if _path_has_isotopic_hydrogen(mol, (idx,)):
        return ""
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if not heavy:
        return ""
    branched_alkyl = _retained_branched_alkyl_prefix(mol, idx, parent_idx, heavy)
    if branched_alkyl:
        return branched_alkyl
    halo_methyl = _halo_methyl_prefix(mol, heavy)
    if halo_methyl:
        return halo_methyl
    if len(heavy) != 1:
        return ""
    attached_idx = heavy[0]
    attached = mol.GetAtomWithIdx(attached_idx)
    if _is_nitrile_carbon(mol, attached_idx):
        return "cyanomethyl"
    carboxymethyl = _terminal_carboxymethyl_prefix(mol, attached_idx, idx, parent_idx)
    if carboxymethyl:
        return carboxymethyl
    if attached.IsInRing():
        attached_bond = mol.GetBondBetweenAtoms(idx, attached_idx)
        if attached_bond is None:
            return ""
        if attached_bond.GetBondType() == Chem.BondType.DOUBLE:
            return _ring_alkenyl_prefix(mol, idx, parent_idx)
        if attached_bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        ring = _ring_substituent_prefix(mol, attached_idx, idx)
        if not ring:
            return ""
        if ring == "phenyl":
            return "benzyl"
        if ring.endswith("phenyl"):
            return ring[: -len("phenyl")] + "benzyl"
        if "-" in ring or ring[:1].isdigit() or ring.startswith("("):
            return f"({ring})methyl"
        return ring + "methyl"
    hetero_methyl = _heteroatom_methyl_prefix(mol, attached_idx, idx)
    if hetero_methyl:
        return hetero_methyl
    if attached.GetSymbol() == "C":
        functionalized = _terminal_functionalized_alkyl_prefix(mol, idx, parent_idx)
        if functionalized:
            return functionalized
        return _terminal_haloethyl_prefix(mol, attached_idx, idx)
    return ""


def _heteroatom_methyl_prefix(mol, hetero_idx: int, methylene_idx: int) -> str:
    atom = mol.GetAtomWithIdx(hetero_idx)
    if atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return ""
    bond = mol.GetBondBetweenAtoms(hetero_idx, methylene_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if atom.GetSymbol() == "O":
        oxy = _oxygen_prefix(mol, hetero_idx, methylene_idx)
        if not oxy or not oxy.endswith("oxy") and oxy != "hydroxy":
            return ""
        return oxy + "methyl" if oxy != "hydroxy" else "hydroxymethyl"
    if atom.GetSymbol() == "N":
        amino = _nitrogen_prefix(mol, hetero_idx, methylene_idx)
        if not amino or not amino.endswith("amino"):
            return ""
        return amino + "methyl"
    return ""


def _terminal_carboxymethyl_prefix(mol, carbonyl_idx: int, methylene_idx: int, parent_idx: int) -> str:
    parent = mol.GetAtomWithIdx(parent_idx)
    if parent.GetSymbol() == "N":
        return ""
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0 or carbonyl.IsInRing():
        return ""
    bond_to_methylene = mol.GetBondBetweenAtoms(carbonyl_idx, methylene_idx)
    if bond_to_methylene is None or bond_to_methylene.GetBondType() != Chem.BondType.SINGLE:
        return ""
    double_oxygen = []
    single_oxygen = []
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == methylene_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen.append(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            single_oxygen.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return ""
    if len(double_oxygen) != 1 or len(single_oxygen) != 1:
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
        return "carboxylatomethyl"
    if oxygen.GetFormalCharge() == 0:
        return "carboxymethyl"
    return ""


def _terminal_functionalized_alkyl_prefix(mol, idx: int, parent_idx: int) -> str:
    chain = []
    previous = parent_idx
    current = idx
    while len(chain) < 24:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
            return ""
        if _path_has_isotopic_hydrogen(mol, (current,)):
            return ""
        bond = mol.GetBondBetweenAtoms(previous, current)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        chain.append(current)
        next_carbons = []
        terminal_groups = []
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == previous or nbr.GetAtomicNum() <= 1:
                continue
            nbr_bond = mol.GetBondBetweenAtoms(current, nidx)
            if _terminal_functional_group_term(mol, nidx, current):
                terminal_groups.append(nidx)
            elif nbr.GetSymbol() == "C" and nbr_bond is not None and nbr_bond.GetBondType() == Chem.BondType.SINGLE:
                next_carbons.append(nidx)
            else:
                return ""
        if terminal_groups and not next_carbons:
            if len(terminal_groups) != 1:
                return ""
            term = _terminal_functional_group_term(mol, terminal_groups[0], current)
            root = ROOTS.get(len(chain))
            if not term or not root:
                return ""
            if "-" in term or "(" in term or ")" in term:
                term = f"({term})"
            return f"{len(chain)}-{term}{root}yl"
        if len(next_carbons) != 1:
            return ""
        previous, current = current, next_carbons[0]
    return ""


def _terminal_functional_group_term(mol, group_idx: int, chain_idx: int) -> str:
    atom = mol.GetAtomWithIdx(group_idx)
    if atom.GetSymbol() in {"N", "O", "S"}:
        return _terminal_hetero_functional_group_term(mol, group_idx, chain_idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return ""
    bond_to_chain = mol.GetBondBetweenAtoms(group_idx, chain_idx)
    if bond_to_chain is None or bond_to_chain.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if _is_nitrile_carbon(mol, group_idx):
        return "cyano"
    double_oxygen = []
    single_oxygen = []
    single_nitrogen = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == chain_idx or nbr.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(group_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen.append(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            single_oxygen.append(nidx)
        elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
            single_nitrogen.append(nidx)
        else:
            return ""
    if len(double_oxygen) != 1:
        return ""
    if len(single_oxygen) == 1 and not single_nitrogen:
        oxygen = mol.GetAtomWithIdx(single_oxygen[0])
        if oxygen.GetFormalCharge() == -1:
            return "carboxylato"
    if len(single_nitrogen) == 1 and not single_oxygen:
        nitrogen = mol.GetAtomWithIdx(single_nitrogen[0])
        extras = [
            nbr.GetIdx()
            for nbr in nitrogen.GetNeighbors()
            if nbr.GetIdx() != group_idx and nbr.GetAtomicNum() > 1
        ]
        if not extras and nitrogen.GetFormalCharge() == 0:
            return "carbamyl"
    return ""


def _terminal_hetero_functional_group_term(mol, group_idx: int, chain_idx: int) -> str:
    atom = mol.GetAtomWithIdx(group_idx)
    if atom.IsInRing() or atom.GetIsotope():
        return ""
    bond_to_chain = mol.GetBondBetweenAtoms(group_idx, chain_idx)
    if bond_to_chain is None or bond_to_chain.GetBondType() != Chem.BondType.SINGLE:
        return ""
    term = branch_prefix(mol, group_idx, chain_idx)
    if not term:
        return ""
    if term.endswith(("amino", "azaniumyl", "azanidyl", "diazo", "diazenyl", "hydrazono")):
        return term
    if term in {"azido", "nitro", "nitroso", "hydroxy", "oxido", "oxidaniumyl", "sulfanyl"}:
        return term
    if term.endswith(("oxy", "sulfanyl", "sulfinyl", "sulfonyl", "sulfonamido")):
        return term
    return ""


def _retained_branched_alkyl_prefix(mol, idx: int, parent_idx: int, heavy: Sequence[int]) -> str:
    if _branch_has_assigned_stereo(mol, idx, parent_idx):
        return ""
    if len(heavy) in {2, 3} and all(_is_terminal_methyl(mol, nbr_idx, idx) for nbr_idx in heavy):
        return "isopropyl" if len(heavy) == 2 else "t-butyl"
    if len(heavy) != 1:
        return ""
    attached_idx = heavy[0]
    attached = mol.GetAtomWithIdx(attached_idx)
    if attached.GetSymbol() != "C" or attached.GetFormalCharge() != 0 or attached.IsInRing():
        return ""
    if attached.GetIsotope() or _path_has_isotopic_hydrogen(mol, (attached_idx,)):
        return ""
    bond = mol.GetBondBetweenAtoms(idx, attached_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    attached_heavy = [
        nbr.GetIdx()
        for nbr in attached.GetNeighbors()
        if nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1
    ]
    if len(attached_heavy) == 2 and all(_is_terminal_methyl(mol, nbr_idx, attached_idx) for nbr_idx in attached_heavy):
        return "isobutyl"
    if len(attached_heavy) == 3 and all(_is_terminal_methyl(mol, nbr_idx, attached_idx) for nbr_idx in attached_heavy):
        return "neopentyl"
    return ""


def _branch_has_assigned_stereo(mol, idx: int, parent_idx: int) -> bool:
    atoms = collect_subtree(mol, idx, parent_idx)
    for atom_idx in atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return True
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in atoms and end in atoms and bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return True
    return False


def _halo_methyl_prefix(mol, heavy: Sequence[int]) -> str:
    symbols = []
    for nidx in heavy:
        atom = mol.GetAtomWithIdx(nidx)
        if atom.GetSymbol() not in HALO or atom.GetIsotope():
            return ""
        symbols.append(atom.GetSymbol())
    if not symbols or len(set(symbols)) != 1 or len(symbols) > 3:
        return ""
    return _halo_count_prefix(symbols[0], len(symbols)) + "methyl"


def _terminal_haloethyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
        return ""
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if _path_has_isotopic_hydrogen(mol, (idx,)):
        return ""
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    halo_methyl = _halo_methyl_prefix(mol, heavy)
    if not halo_methyl or not halo_methyl.endswith("methyl"):
        return ""
    halo = halo_methyl[: -len("methyl")]
    if len(heavy) == 3:
        return f"2,2,2-{halo}ethyl"
    if len(heavy) == 2:
        return f"2,2-{halo}ethyl"
    return f"2-{halo}ethyl"


def _halo_count_prefix(symbol: str, count: int) -> str:
    prefix = HALO.get(symbol, "")
    if not prefix:
        return ""
    if count == 1:
        return prefix
    return MULT.get(count, "") + prefix


def _protected_alkyl_prefix(mol, idx: int, parent_idx: int) -> str:
    for renderer in (
        _sem_methyl_prefix,
        _tbdmsoxymethyl_prefix,
        _tbdmsoxyethyl_prefix,
        _bocmethyl_prefix,
    ):
        name = renderer(mol, idx, parent_idx)
        if name:
            return name
    return ""


def _sem_methyl_prefix(mol, idx: int, parent_idx: int) -> str:
    oxygen_idx = _single_heavy_neighbor(mol, idx, parent_idx, "O")
    if oxygen_idx is None:
        return ""
    first_carbon = _single_heavy_neighbor(mol, oxygen_idx, idx, "C")
    if first_carbon is None:
        return ""
    second_carbon = _single_heavy_neighbor(mol, first_carbon, oxygen_idx, "C")
    if second_carbon is None:
        return ""
    silicon_idx = _single_heavy_neighbor(mol, second_carbon, first_carbon, "Si")
    if silicon_idx is None:
        return ""
    return "2-tms-ethoxymethyl" if _is_trimethylsilyl_group(mol, silicon_idx, second_carbon) else ""


def _tbdmsoxymethyl_prefix(mol, idx: int, parent_idx: int) -> str:
    oxygen_idx = _single_heavy_neighbor(mol, idx, parent_idx, "O")
    if oxygen_idx is None:
        return ""
    silicon_idx = _single_heavy_neighbor(mol, oxygen_idx, idx, "Si")
    if silicon_idx is None:
        return ""
    return "tbdmsoxymethyl" if _is_tbdms_group(mol, silicon_idx, oxygen_idx) else ""


def _tbdmsoxyethyl_prefix(mol, idx: int, parent_idx: int) -> str:
    second_carbon = _single_heavy_neighbor(mol, idx, parent_idx, "C")
    if second_carbon is None:
        return ""
    oxygen_idx = _single_heavy_neighbor(mol, second_carbon, idx, "O")
    if oxygen_idx is None:
        return ""
    silicon_idx = _single_heavy_neighbor(mol, oxygen_idx, second_carbon, "Si")
    if silicon_idx is None:
        return ""
    return "2-tbdmsoxyethyl" if _is_tbdms_group(mol, silicon_idx, oxygen_idx) else ""


def _bocmethyl_prefix(mol, idx: int, parent_idx: int) -> str:
    oxygen_idx = _single_heavy_neighbor(mol, idx, parent_idx, "O")
    if oxygen_idx is None:
        return ""
    carbonyl_idx = _single_heavy_neighbor(mol, oxygen_idx, idx, "C")
    if carbonyl_idx is None:
        return ""
    return "bocmethyl" if _is_boc_oxycarbonyl(mol, carbonyl_idx, oxygen_idx) else ""


def _is_boc_oxycarbonyl(mol, carbonyl_idx: int, oxygen_idx: int) -> bool:
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0:
        return False
    double_oxygen = []
    single_oxygen = []
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == oxygen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen.append(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            single_oxygen.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return False
    return len(double_oxygen) == 1 and len(single_oxygen) == 1 and _is_tert_butoxy_oxygen(mol, single_oxygen[0], carbonyl_idx)


def _single_heavy_neighbor(mol, atom_idx: int, blocked_idx: int, symbol: str) -> int | None:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetFormalCharge() != 0 or atom.IsInRing():
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != blocked_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    neighbor_idx = heavy[0]
    neighbor = mol.GetAtomWithIdx(neighbor_idx)
    bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
    if neighbor.GetSymbol() != symbol or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    return neighbor_idx


def _boc_aminoalkyl_prefix(mol, idx: int, parent_idx: int) -> str:
    path = _linear_carbon_chain_to_boc_nitrogen(mol, idx, parent_idx)
    if not path:
        return ""
    if _path_has_isotopic_hydrogen(mol, path[:-1]):
        return ""
    parent = mol.GetAtomWithIdx(parent_idx)
    if parent.IsInRing() and not parent.GetIsAromatic() and _has_assigned_stereo(mol):
        return ""
    nitrogen_idx = path[-1]
    if not _nitrogen_has_unsubstituted_boc_group(mol, nitrogen_idx, path[-2]):
        return ""
    return {
        1: "bocaminomethyl",
        2: "2-bocaminoethyl",
        3: "3-bocaminopropyl",
    }.get(len(path) - 1, "")


def _path_has_isotopic_hydrogen(mol, atom_indices: Sequence[int]) -> bool:
    return any(
        nbr.GetSymbol() == "H" and nbr.GetIsotope()
        for atom_idx in atom_indices
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
    )


def _has_assigned_stereo(mol) -> bool:
    if any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        return True
    return any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds())


def _linear_carbon_chain_to_boc_nitrogen(mol, idx: int, parent_idx: int) -> tuple[int, ...]:
    path = [idx]
    previous = parent_idx
    current = idx
    while len(path) <= 4:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            return ()
        if mol.GetBondBetweenAtoms(previous, current).GetBondType() != Chem.BondType.SINGLE:
            return ()
        heavy = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetIdx() != previous and nbr.GetAtomicNum() > 1
        ]
        if len(heavy) != 1:
            return ()
        next_idx = heavy[0]
        next_atom = mol.GetAtomWithIdx(next_idx)
        if next_atom.GetSymbol() == "N":
            return tuple([*path, next_idx]) if len(path) <= 3 else ()
        if next_atom.GetSymbol() != "C":
            return ()
        previous, current = current, next_idx
        path.append(current)
    return ()


def _nitrogen_has_unsubstituted_boc_group(mol, nitrogen_idx: int, chain_idx: int) -> bool:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing():
        return False
    carbonyl_neighbors = []
    extra_heavy = []
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == chain_idx:
            continue
        if nbr.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE and _is_boc_carbonyl(mol, nidx, nitrogen_idx):
            carbonyl_neighbors.append(nidx)
        else:
            extra_heavy.append(nidx)
    return len(carbonyl_neighbors) == 1 and not extra_heavy


def _is_boc_carbonyl(mol, carbonyl_idx: int, nitrogen_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbonyl_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return False
    double_oxygen = []
    single_oxygen = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen.append(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            single_oxygen.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return False
    return len(double_oxygen) == 1 and len(single_oxygen) == 1 and _is_tert_butoxy_oxygen(mol, single_oxygen[0], carbonyl_idx)


def _is_tert_butoxy_oxygen(mol, oxygen_idx: int, carbonyl_idx: int) -> bool:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0:
        return False
    heavy = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != carbonyl_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return False
    central_idx = heavy[0]
    central = mol.GetAtomWithIdx(central_idx)
    if central.GetSymbol() != "C" or central.GetFormalCharge() != 0:
        return False
    if mol.GetBondBetweenAtoms(oxygen_idx, central_idx).GetBondType() != Chem.BondType.SINGLE:
        return False
    methyl_neighbors = [
        nbr.GetIdx()
        for nbr in central.GetNeighbors()
        if nbr.GetIdx() != oxygen_idx and nbr.GetSymbol() == "C"
    ]
    if len(methyl_neighbors) != 3:
        return False
    return all(
        mol.GetBondBetweenAtoms(central_idx, methyl_idx).GetBondType() == Chem.BondType.SINGLE
        and mol.GetAtomWithIdx(methyl_idx).GetDegree() == 1
        for methyl_idx in methyl_neighbors
    )


def _mesyloxymethyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
        return ""
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return ""
    oxygen_idx = heavy[0]
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0:
        return ""
    if mol.GetBondBetweenAtoms(idx, oxygen_idx).GetBondType() != Chem.BondType.SINGLE:
        return ""
    sulfur_neighbors = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != idx and nbr.GetSymbol() == "S"
    ]
    if len(sulfur_neighbors) != 1:
        return ""
    return "mesyloxymethyl" if _is_methylsulfonyl_sulfur(mol, sulfur_neighbors[0], oxygen_idx) else ""


def _is_methylsulfonyl_sulfur(mol, sulfur_idx: int, oxygen_idx: int) -> bool:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0:
        return False
    double_oxygen_count = 0
    carbon_ligands = []
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == oxygen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_ligands.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return False
    if double_oxygen_count != 2 or len(carbon_ligands) != 1:
        return False
    methyl = mol.GetAtomWithIdx(carbon_ligands[0])
    return methyl.GetDegree() == 1 and methyl.GetFormalCharge() == 0


def _prefer_resource_prefix(name: str) -> bool:
    if not name:
        return False
    if "carboxy" in name:
        return False
    if name in {"methyl", "ethyl", "propyl", "butyl", "phenyl", "pyridine", "pyrimidine"}:
        return False
    if name.startswith(("active-amyl", "carbisopropoxy")):
        return False
    return True


def _deuterated_methyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return ""
    deuterium_count = 0
    tritium_count = 0
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        if nbr.GetSymbol() != "H":
            return ""
        if nbr.GetIsotope() == 2:
            deuterium_count += 1
        elif nbr.GetIsotope() == 3:
            tritium_count += 1
        else:
            return ""
    if deuterium_count and tritium_count:
        return ""
    if deuterium_count:
        return _isotopic_methyl_prefix(deuterium_count, "deuterio")
    if tritium_count:
        return _isotopic_methyl_prefix(tritium_count, "tritio")
    return ""


def _isotopic_methyl_prefix(count: int, isotope_prefix: str) -> str:
    if count == 1:
        return isotope_prefix + "methyl"
    multiplier = MULT.get(count)
    return f"{multiplier}{isotope_prefix}methyl" if multiplier else ""


def _has_neighbor_bond(mol, idx: int, neighbors: Sequence[int], symbol: str, bond_type) -> bool:
    return any(
        mol.GetAtomWithIdx(nidx).GetSymbol() == symbol
        and mol.GetBondBetweenAtoms(idx, nidx).GetBondType() == bond_type
        for nidx in neighbors
    )


def _subtree_has_ring(mol, root_idx: int, blocked_idx: int) -> bool:
    cache = mol_cache_dict(mol, "subtree_has_ring_cache")
    key = (root_idx, blocked_idx)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = any(mol.GetAtomWithIdx(atom_idx).IsInRing() for atom_idx in collect_subtree(mol, root_idx, blocked_idx))
    cache[key] = result
    return result


@lru_cache(maxsize=1)
def _branch_rule_sets() -> dict[str, tuple[BranchRuleSet, ...]]:
    return {
        "I": (
            (
                BranchRuleSpec(_iodanyl_acyloxy_prefix),
                BranchRuleSpec(_halogen_prefix),
            ),
        ),
        "F": ((BranchRuleSpec(_halogen_prefix),),),
        "Cl": ((BranchRuleSpec(_halogen_prefix),),),
        "Br": ((BranchRuleSpec(_halogen_prefix),),),
        "O": (
            (
                BranchRuleSpec(_oxygen_prefix),
                BranchRuleSpec(_resource_prefix),
            ),
        ),
        "N": (
            (
                BranchRuleSpec("nitro", _nitro_branch_predicate),
                BranchRuleSpec(_nitrogen_prefix),
                BranchRuleSpec(_resource_prefix),
            ),
        ),
        "S": (
            (
                BranchRuleSpec(_sulfur_prefix),
                BranchRuleSpec(_resource_prefix),
            ),
        ),
        "P": (
            (
                BranchRuleSpec(_phosphorus_prefix),
                BranchRuleSpec(_resource_prefix),
            ),
        ),
        "B": (
            (
                BranchRuleSpec(_boron_prefix),
                BranchRuleSpec(_resource_prefix),
            ),
        ),
        "Si": (
            (
                BranchRuleSpec(_group14_prefix),
                BranchRuleSpec(_resource_prefix),
            ),
        ),
        "Ge": (
            (
                BranchRuleSpec(_group14_prefix),
                BranchRuleSpec(_resource_prefix),
            ),
        ),
        "Sn": (
            (
                BranchRuleSpec(_group14_prefix),
                BranchRuleSpec(_resource_prefix),
            ),
        ),
        "Se": (
            (
                BranchRuleSpec(_selenium_prefix),
                BranchRuleSpec(_resource_prefix),
            ),
        ),
        "Li": (
            (
                BranchRuleSpec(_metal_prefix),
                BranchRuleSpec(_resource_prefix),
            ),
        ),
        "Na": (
            (
                BranchRuleSpec(_metal_prefix),
                BranchRuleSpec(_resource_prefix),
            ),
        ),
        "K": (
            (
                BranchRuleSpec(_metal_prefix),
                BranchRuleSpec(_resource_prefix),
            ),
        ),
        "Mg": (
            (
                BranchRuleSpec(_metal_prefix),
                BranchRuleSpec(_resource_prefix),
            ),
        ),
        "*": (
            (BranchRuleSpec(_resource_prefix),),
        ),
    }


def _boron_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "B" or atom.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1]
    pinacol = _pinacol_boronate_prefix(mol, idx, parent_idx, heavy)
    if pinacol:
        return pinacol
    if not heavy:
        return "boranyl"
    if len(heavy) == 2 and all(_is_terminal_hydroxy_oxygen(mol, nidx, idx) for nidx in heavy):
        return "borono"
    return ""


def _pinacol_boronate_prefix(mol, boron_idx: int, parent_idx: int, heavy: Sequence[int]) -> str:
    oxygen_indices = [idx for idx in heavy if mol.GetAtomWithIdx(idx).GetSymbol() == "O"]
    if len(oxygen_indices) != 2:
        return ""
    ring_carbons = []
    for oxygen_idx in oxygen_indices:
        carbon_idx = _single_heavy_neighbor_allow_ring(mol, oxygen_idx, boron_idx, "C")
        if carbon_idx is None:
            return ""
        ring_carbons.append(carbon_idx)
    if len(set(ring_carbons)) != 2:
        return ""
    if mol.GetBondBetweenAtoms(ring_carbons[0], ring_carbons[1]) is None:
        return ""
    if mol.GetBondBetweenAtoms(ring_carbons[0], ring_carbons[1]).GetBondType() != Chem.BondType.SINGLE:
        return ""
    methyl_counts = []
    ethyl_counts = []
    for carbon_idx in ring_carbons:
        methyl = 0
        ethyl = 0
        for nbr in mol.GetAtomWithIdx(carbon_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_carbons or nidx in oxygen_indices or nidx == parent_idx:
                continue
            if nbr.GetSymbol() != "C":
                return ""
            if _is_terminal_methyl(mol, nidx, carbon_idx):
                methyl += 1
            elif _is_terminal_ethyl(mol, nidx, carbon_idx):
                ethyl += 1
            else:
                return ""
        methyl_counts.append(methyl)
        ethyl_counts.append(ethyl)
    if ethyl_counts == [2, 2] and methyl_counts == [0, 0]:
        return "4,4,5,5-tetraethyl-1,3,2-dioxaborolan-2-yl"
    if sum(ethyl_counts):
        return ""
    methyl_counts = sorted(methyl_counts, reverse=True)
    if methyl_counts == [2, 2]:
        return "tetramethyl-1,3,2-dioxaborolan-2-yl"
    if methyl_counts == [2, 1]:
        return "4,4,5-trimethyl-1,3,2-dioxaborolan-2-yl"
    if methyl_counts == [2, 0]:
        return "4,4-dimethyl-1,3,2-dioxaborolan-2-yl"
    return ""


def _single_heavy_neighbor_allow_ring(mol, atom_idx: int, blocked_idx: int, symbol: str) -> int | None:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetFormalCharge() != 0:
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != blocked_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    neighbor_idx = heavy[0]
    neighbor = mol.GetAtomWithIdx(neighbor_idx)
    bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
    if neighbor.GetSymbol() != symbol or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    return neighbor_idx


def _is_terminal_methyl(mol, carbon_idx: int, parent_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
        return False
    if carbon.GetIsotope() or _path_has_isotopic_hydrogen(mol, (carbon_idx,)):
        return False
    bond = mol.GetBondBetweenAtoms(carbon_idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    heavy = [nbr.GetIdx() for nbr in carbon.GetNeighbors() if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1]
    return not heavy


def _is_terminal_ethyl(mol, carbon_idx: int, parent_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return False
    heavy = [nbr.GetIdx() for nbr in carbon.GetNeighbors() if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1]
    return len(heavy) == 1 and _is_terminal_methyl(mol, heavy[0], carbon_idx)


def _is_terminal_hydroxy_oxygen(mol, oxygen_idx: int, boron_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(oxygen_idx)
    if atom.GetSymbol() != "O" or atom.GetFormalCharge() != 0:
        return False
    bond = mol.GetBondBetweenAtoms(oxygen_idx, boron_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != boron_idx and nbr.GetAtomicNum() > 1]
    return not heavy


def _nitro_branch_predicate(mol, idx: int, parent_idx: int) -> bool:
    if not _is_nitro_nitrogen(mol, idx):
        return False
    atom = mol.GetAtomWithIdx(idx)
    non_oxygen = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() != "O"]
    if len(non_oxygen) == 1:
        return non_oxygen[0] == parent_idx
    if non_oxygen:
        return False
    return mol.GetAtomWithIdx(parent_idx).GetSymbol() == "O"


def _alkylidene_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.DOUBLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]
    if not heavy:
        return "methylidene"
    if len(heavy) == 1:
        heavy_bond = mol.GetBondBetweenAtoms(idx, heavy[0])
        heavy_atom = mol.GetAtomWithIdx(heavy[0])
        if (
            heavy_atom.GetSymbol() == "O"
            and heavy_atom.GetFormalCharge() == 0
            and heavy_bond is not None
            and heavy_bond.GetBondType() == Chem.BondType.DOUBLE
        ):
            return "carbonyl"
        if _is_nitrile_carbon(mol, heavy[0]):
            return "cyanomethylidene"
        if heavy_bond is not None and heavy_bond.GetBondType() == Chem.BondType.DOUBLE:
            terminal = [
                nbr.GetIdx()
                for nbr in heavy_atom.GetNeighbors()
                if nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1
            ]
            if not terminal and heavy_atom.GetFormalCharge() == 0:
                if heavy_atom.GetSymbol() == "C":
                    return "ethenylidene"
                if heavy_atom.GetSymbol() == "N":
                    return "iminomethylidene"
        hetero = _hetero_methylidene_prefix(mol, heavy[0], idx)
        if hetero:
            return hetero + "methylidene"
        if heavy_atom.IsInRing():
            ring = _ring_substituent_prefix(mol, heavy[0], idx)
            if not ring:
                return ""
            if ring == "phenyl":
                return "benzylidene"
            if ring.endswith("phenyl"):
                return ring[: -len("phenyl")] + "benzylidene"
            if "-" in ring or ring[:1].isdigit() or ring.startswith("("):
                return f"({ring})methylidene"
            return ring + "methylidene"
        chain = _linear_carbon_chain(mol, heavy[0], idx, max_len=9)
        if not chain:
            return ""
        root = ROOTS.get(len(chain) + 1)
        return root + "ylidene" if root else ""
    if len(heavy) == 2:
        halogens = [mol.GetAtomWithIdx(nidx).GetSymbol() for nidx in heavy]
        if all(symbol in HALO for symbol in halogens):
            parts = []
            counts = Counter(HALO[symbol] for symbol in halogens)
            for name in sorted(counts):
                count = counts[name]
                parts.append((MULT.get(count, "") if count > 1 else "") + name)
            return "".join(parts) + "methylidene"
        if all(_is_nitrile_carbon(mol, nidx) for nidx in heavy):
            return "dicyanomethylidene"
        chains = [_linear_carbon_chain(mol, nidx, idx, max_len=4) for nidx in heavy]
        if all(chains) and all(len(chain) == 1 for chain in chains):
            return "propan-2-ylidene"
        substituent_names = []
        for nidx in heavy:
            symbol = mol.GetAtomWithIdx(nidx).GetSymbol()
            if symbol in HALO:
                substituent_names.append(HALO[symbol])
                continue
            if _is_nitrile_carbon(mol, nidx):
                substituent_names.append("cyano")
                continue
            name = substituent_from_root(mol, nidx, idx)
            if not name:
                substituent_names = []
                break
            substituent_names.append(_parenthesize_complex_prefix(name))
        if substituent_names:
            return "".join(sorted(substituent_names)) + "methylidene"
        if any(not chain for chain in chains):
            return ""
    return ""


def _hetero_methylidene_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    symbol = atom.GetSymbol()
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if symbol in HALO:
        return HALO[symbol]
    if symbol == "O":
        return _oxygen_prefix(mol, idx, parent_idx)
    if symbol == "N":
        return _nitrogen_prefix(mol, idx, parent_idx)
    if symbol == "S":
        return _sulfur_prefix(mol, idx, parent_idx)
    return ""


def _enal_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    double_c = []
    single_c = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_c.append(nidx)
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            single_c.append(nidx)
        else:
            return ""
    if len(double_c) != 1 or len(single_c) > 1:
        return ""
    beta_idx = double_c[0]
    carbonyl = []
    for nbr in mol.GetAtomWithIdx(beta_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx:
            continue
        bond = mol.GetBondBetweenAtoms(beta_idx, nidx)
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE and _aldehyde_carbon(mol, nidx, beta_idx):
            carbonyl.append(nidx)
        else:
            return ""
    if len(carbonyl) != 1:
        return ""
    if not single_c:
        return "3-oxo-prop-1-en-1-yl"
    side = _linear_carbon_chain(mol, single_c[0], idx, max_len=1)
    if len(side) == 1:
        return "4-oxo-but-2-en-2-yl"
    return ""


def _enone_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    beta_atoms = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.DOUBLE:
            beta_atoms.append(nidx)
        else:
            return ""
    if len(beta_atoms) != 1:
        return ""
    beta_idx = beta_atoms[0]
    carbonyls = []
    for nbr in mol.GetAtomWithIdx(beta_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx:
            continue
        bond = mol.GetBondBetweenAtoms(beta_idx, nidx)
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE and is_carbonyl_carbon(mol, nidx):
            carbonyls.append(nidx)
        else:
            return ""
    if len(carbonyls) != 1:
        return ""
    carbonyl = _carbonyl_substituent_prefix(mol, carbonyls[0], beta_idx)
    if not carbonyl:
        return ""
    if "-" in carbonyl or carbonyl.startswith("("):
        carbonyl = f"({carbonyl})"
    return f"2-{carbonyl}eth-1-en-1-yl"


def _haloethenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    distal_carbons = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.DOUBLE:
            distal_carbons.append(nidx)
        else:
            return ""
    if len(distal_carbons) != 1:
        return ""
    distal_idx = distal_carbons[0]
    halo_names = []
    for nbr in mol.GetAtomWithIdx(distal_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx:
            continue
        bond = mol.GetBondBetweenAtoms(distal_idx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE or nbr.GetSymbol() not in HALO:
            return ""
        halo_names.append(HALO[nbr.GetSymbol()])
    if len(halo_names) != 2:
        return ""
    counts = Counter(halo_names)
    parts = []
    for name in sorted(counts):
        count = counts[name]
        parts.append((MULT.get(count, "") if count > 1 else "") + name)
    return "2,2-" + "".join(parts) + "ethenyl"


def _ketenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    double_c = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_c.append(nidx)
        else:
            return ""
    if len(double_c) != 1:
        return ""
    carbonyl = mol.GetAtomWithIdx(double_c[0])
    oxygens = []
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx:
            continue
        bond = mol.GetBondBetweenAtoms(double_c[0], nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            oxygens.append(nidx)
        else:
            return ""
    return "2-oxoethenyl" if len(oxygens) == 1 else ""


def _aldehyde_carbon(mol, idx: int, beta_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return False
    has_double_o = False
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == beta_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_double_o = True
        else:
            return False
    return has_double_o


def _iodanyl_acyloxy_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "I":
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    acyloxy = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx or nbr.GetSymbol() != "O":
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        name = _acyloxy_ligand_name(mol, nidx, idx)
        if name:
            acyloxy.append(name)
    if not acyloxy:
        return ""
    counts = Counter(acyloxy)
    parts = []
    for name in sorted(counts):
        count = counts[name]
        if count == 1:
            parts.append(name)
        elif count == 2:
            parts.append(f"bis({name})")
        else:
            parts.append(f"{MULT.get(count, '')}({name})")
    return "".join(parts) + "iodo"


def _acyloxy_ligand_name(mol, oxygen_idx: int, iodine_idx: int) -> str:
    carbonyls = []
    for nbr in mol.GetAtomWithIdx(oxygen_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == iodine_idx or nbr.GetSymbol() != "C":
            continue
        bond = mol.GetBondBetweenAtoms(oxygen_idx, nidx)
        if bond.GetBondType() == Chem.BondType.SINGLE and is_carbonyl_carbon(mol, nidx):
            carbonyls.append(nidx)
    if len(carbonyls) != 1:
        return ""
    acyl = _acyl_parent_name(mol, carbonyls[0], oxygen_idx)
    return acyl + "oxy" if acyl else ""


def _acyl_parent_name(mol, carbonyl_idx: int, blocked_idx: int) -> str:
    atoms = collect_subtree(mol, carbonyl_idx, blocked_idx)
    paths = rooted_paths(mol, carbonyl_idx, atoms)
    best = ""
    best_key = None
    for path in paths:
        if not path or path[0] != carbonyl_idx:
            continue
        if any(mol.GetAtomWithIdx(atom_idx).GetSymbol() != "C" for atom_idx in path):
            continue
        root = ROOTS.get(len(path))
        if not root:
            continue
        path_set = set(path)
        branch_prefixes = []
        ok = True
        for pos, atom_idx in enumerate(path, start=1):
            for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx == blocked_idx or nidx in path_set:
                    continue
                bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
                if pos == 1 and nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                    continue
                if nidx not in atoms:
                    continue
                prefix = branch_prefix(mol, nidx, atom_idx)
                if not prefix:
                    ok = False
                    break
                branch_prefixes.append((str(pos), prefix))
            if not ok:
                break
        if not ok:
            continue
        prefix_text = _format_prefixes(branch_prefixes)
        name = root + "anoyl"
        if prefix_text:
            name = prefix_text + name
        key = (-len(path), substituent_locants(mol, path, atoms, blocked_idx), name)
        if best_key is None or key < best_key:
            best_key = key
            best = name
    return best


def _linear_carbon_chain(mol, start_idx: int, blocked_idx: int, max_len: int) -> tuple[int, ...]:
    path = []
    previous = blocked_idx
    current = start_idx
    while len(path) < max_len:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            return ()
        bond = mol.GetBondBetweenAtoms(previous, current)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ()
        path.append(current)
        next_atoms = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetIdx() != previous and nbr.GetSymbol() == "C"
        ]
        other = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetIdx() != previous and nbr.GetSymbol() != "C"
        ]
        if other:
            return ()
        if not next_atoms:
            return tuple(path)
        if len(next_atoms) != 1:
            return ()
        previous, current = current, next_atoms[0]
    return ()
