from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from ..branch_rules import (
    branch_prefix,
    can_be_in_parent_skeleton,
    is_carbonyl_carbon,
    parent_chain_name,
    replacement_prefixes,
    rooted_paths,
    substituent_from_root,
    substituent_locants,
    collect_subtree,
)
from ..branch.functional_groups import _is_nitrile_carbon
from ..constants import MULT, ROOTS
from ..formatting import alkyl_to_alkoxy as _alkyl_to_alkoxy
from ..formatting import format_prefixes as _format_prefixes
from ..formatting import join_prefix_parent as _join_prefix_parent
from ..formatting import preferred_parent_stem as _preferred_parent_stem
from ..formatting import unsaturated_parent as _unsaturated_parent
from ..graph_model import graph_view
from ..molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set
from ..simple_prefixes import simple_external_prefix
from ..stereochemistry import assigned_atom_stereochemistry


MAX_ACYCLIC_PARENT_PATHS = 16
MAX_ACYCLIC_ORIENTATIONS_PER_DESCRIPTOR = 2
_SATURATED_CHAIN_PARENT_STEMS = frozenset(f"{root}an" for root in ROOTS.values())
_CHARGE_PRESERVING_PREFIX_TERMS = frozenset(
    (
        "carboxylato",
        "phosphonato",
        "phosphonatooxy",
        "sulfinato",
        "sulfonato",
    )
)
_CHARGE_PRESERVING_CENTER_SYMBOLS = frozenset(("C", "P", "S"))


@dataclass(frozen=True, slots=True)
class _AcyclicGraphView:
    symbols: tuple[str, ...]
    isotopes: tuple[int, ...]
    neighbors: tuple[tuple[int, ...], ...]
    bond_orders: dict[tuple[int, int], float]


@dataclass(frozen=True, slots=True)
class AcyclicNameRecord:
    name: str
    stereo_locants: tuple[tuple[int, str], ...] = ()


def name_acyclic(mol, allow_ring_atoms: bool = True) -> str:
    return name_acyclic_record(mol, allow_ring_atoms=allow_ring_atoms).name


def name_acyclic_record(mol, allow_ring_atoms: bool = True) -> AcyclicNameRecord:
    terminal_charge_name = _name_terminal_linear_hydrocarbon_charge(mol)
    if terminal_charge_name:
        return AcyclicNameRecord(terminal_charge_name)

    internal_charge_name = _name_internal_linear_hydrocarbon_charge(mol)
    if internal_charge_name:
        return AcyclicNameRecord(internal_charge_name)

    terminal_charge_name = _name_terminal_acyclic_carbon_charge(mol)
    if terminal_charge_name:
        return AcyclicNameRecord(terminal_charge_name)

    terminal_nitronate_name = _name_terminal_nitronate(mol)
    if terminal_nitronate_name:
        return AcyclicNameRecord(terminal_nitronate_name)

    terminal_imine_azanide_name = _name_terminal_imine_azanide(mol)
    if terminal_imine_azanide_name:
        return AcyclicNameRecord(terminal_imine_azanide_name)

    aminodiazenide_name = _name_aminodiazenide(mol)
    if aminodiazenide_name:
        return AcyclicNameRecord(aminodiazenide_name)

    dialkoxy_azanide_name = _name_dialkoxy_azanide(mol)
    if dialkoxy_azanide_name:
        return AcyclicNameRecord(dialkoxy_azanide_name)

    alkoxy_azanide_name = _name_alkoxy_azanide(mol)
    if alkoxy_azanide_name:
        return AcyclicNameRecord(alkoxy_azanide_name)

    diacyl_azanide_name = _name_terminal_diacyl_azanide(mol)
    if diacyl_azanide_name:
        return AcyclicNameRecord(diacyl_azanide_name)

    hydroxy_diphosphonic_name = _name_geminal_hydroxy_diphosphonic_acid(mol)
    if hydroxy_diphosphonic_name:
        return AcyclicNameRecord(hydroxy_diphosphonic_name)

    hydroxy_diyl_diphosphonic_name = _name_hydroxy_alkane_diyl_diphosphonic_acid(mol)
    if hydroxy_diyl_diphosphonic_name:
        return AcyclicNameRecord(hydroxy_diyl_diphosphonic_name)

    tris_phosphoryl_amine_name = _name_tris_hydroxy_alkoxy_phosphoryl_amine(mol)
    if tris_phosphoryl_amine_name:
        return AcyclicNameRecord(tris_phosphoryl_amine_name)

    terminal_nitrone_name = _name_terminal_nitrone(mol)
    if terminal_nitrone_name:
        return AcyclicNameRecord(terminal_nitrone_name)

    profile = _acyclic_functional_profile(mol)
    for signal, namer in (
        ("oxidanium", _name_oxidanium),
        ("ketene", _name_ketene),
        ("thioamide", _name_thioamide),
        ("ylidene_dioxosulfane", _name_ylidene_dioxosulfane),
        ("thione", _name_thione),
    ):
        if profile[signal]:
            name = namer(mol)
            if name:
                return AcyclicNameRecord(name)

    paths = _candidate_parent_paths(mol, allow_ring_atoms=allow_ring_atoms)
    if not paths:
        return AcyclicNameRecord("")

    return _best_acyclic_record_from_paths(mol, paths)


def _name_terminal_linear_hydrocarbon_charge(mol) -> str:
    if mol.GetRingInfo().NumRings():
        return ""
    atom_count = mol.GetNumAtoms()
    if atom_count == 0:
        return ""
    atoms = tuple(mol.GetAtoms())
    if any(atom.GetSymbol() != "C" for atom in atoms):
        return ""
    charged_atoms = [
        atom.GetIdx()
        for atom in atoms
        if atom.GetFormalCharge() in {-1, 1}
    ]
    if len(charged_atoms) != 1:
        return ""
    charge_idx = charged_atoms[0]
    charge = mol.GetAtomWithIdx(charge_idx).GetFormalCharge()
    if any(atom.GetFormalCharge() != 0 for atom in atoms if atom.GetIdx() != charge_idx):
        return ""
    view = _acyclic_graph_view(mol)
    if any(len(neighbors) > 2 for neighbors in view.neighbors):
        return ""
    if atom_count == 1:
        path = (charge_idx,)
    else:
        endpoints = [idx for idx, neighbors in enumerate(view.neighbors) if len(neighbors) <= 1]
        if len(endpoints) != 2 or charge_idx not in endpoints:
            return ""
        other_endpoint = endpoints[0] if endpoints[1] == charge_idx else endpoints[1]
        path = _shortest_path_in_allowed_graph(view, charge_idx, other_endpoint, set(range(atom_count)))
        if not path or len(path) != atom_count:
            return ""
    parent = parent_chain_name(mol, path, as_substituent=False)
    if not parent:
        return ""
    return f"{parent}{'ide' if charge < 0 else 'ylium'}"


def _name_internal_linear_hydrocarbon_charge(mol) -> str:
    if mol.GetRingInfo().NumRings():
        return ""
    atoms = tuple(mol.GetAtoms())
    if not atoms or any(atom.GetSymbol() != "C" for atom in atoms):
        return ""
    charged = [
        atom.GetIdx()
        for atom in atoms
        if atom.GetFormalCharge() in {-1, 1}
    ]
    if not charged:
        return ""
    charges = {mol.GetAtomWithIdx(idx).GetFormalCharge() for idx in charged}
    if len(charges) != 1:
        return ""
    if any(atom.GetFormalCharge() not in {0, next(iter(charges))} for atom in atoms):
        return ""
    view = _acyclic_graph_view(mol)
    if any(len(neighbors) > 2 for neighbors in view.neighbors):
        return ""
    endpoints = [idx for idx, neighbors in enumerate(view.neighbors) if len(neighbors) <= 1]
    if len(endpoints) != 2:
        return ""
    path = _shortest_path_in_allowed_graph(view, endpoints[0], endpoints[1], set(range(len(atoms))))
    if not path or len(path) != len(atoms):
        return ""
    path = _orient_linear_hydrocarbon_charge_path(mol, path, charged)
    parent = parent_chain_name(mol, path, as_substituent=False)
    if not parent:
        return ""
    charge = next(iter(charges))
    locants = [path.index(idx) + 1 for idx in charged]
    locants = sorted(locants)
    suffix = "ide" if charge < 0 else "ylium"
    if len(locants) == 1:
        stem = parent[:-1] if parent.endswith("e") else parent
        return f"{stem}-{locants[0]}-{suffix}"
    multiplier = MULT.get(len(locants))
    if charge > 0 or not multiplier:
        return ""
    return f"{parent}-{','.join(str(locant) for locant in locants)}-{multiplier}{suffix}"


def _orient_linear_hydrocarbon_charge_path(
    mol,
    path: Sequence[int],
    charged: Sequence[int],
) -> tuple[int, ...]:
    forward = tuple(path)
    reverse = tuple(reversed(path))
    return min(
        (forward, reverse),
        key=lambda oriented: _linear_hydrocarbon_charge_orientation_key(mol, oriented, charged),
    )


def _linear_hydrocarbon_charge_orientation_key(
    mol,
    path: Sequence[int],
    charged: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    ene_locs: list[int] = []
    yne_locs: list[int] = []
    for locant, (left, right) in enumerate(zip(path, path[1:]), start=1):
        bond = mol.GetBondBetweenAtoms(left, right)
        if bond is None:
            continue
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            ene_locs.append(locant)
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            yne_locs.append(locant)
    charge_locs = sorted(path.index(idx) + 1 for idx in charged)
    return (tuple(ene_locs), tuple(yne_locs), tuple(charge_locs))


def _name_terminal_acyclic_carbon_charge(mol) -> str:
    if mol.GetRingInfo().NumRings():
        return ""
    carbon_charges = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C" and atom.GetFormalCharge() in {-1, 1}
    ]
    if len(carbon_charges) != 1:
        return ""
    charge_idx = carbon_charges[0]
    charge_atom = mol.GetAtomWithIdx(charge_idx)
    carbon_neighbors = [
        nbr.GetIdx()
        for nbr in charge_atom.GetNeighbors()
        if nbr.GetAtomicNum() > 1 and nbr.GetSymbol() == "C"
    ]
    if len(carbon_neighbors) > 1:
        return ""

    view = _acyclic_graph_view(mol)
    allowed = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C"
        and not atom.IsInRing()
        and not atom.GetIsotope()
        and (atom.GetIdx() == charge_idx or not _is_nitrile_carbon(mol, atom.GetIdx()))
        and (atom.GetFormalCharge() == 0 or atom.GetIdx() == charge_idx)
    }
    if charge_idx not in allowed:
        return ""

    endpoints = [
        idx
        for idx in allowed
        if idx == charge_idx or sum(1 for nidx in view.neighbors[idx] if nidx in allowed) <= 1
    ]
    paths: set[tuple[int, ...]] = set()
    best_len = 0
    for endpoint in endpoints:
        if endpoint == charge_idx:
            continue
        path = _shortest_path_in_allowed_graph(view, charge_idx, endpoint, allowed)
        if not path:
            continue
        path_len = len(path)
        if path_len > best_len:
            best_len = path_len
            paths.clear()
        if path_len == best_len:
            paths.add(tuple(path))
    if not paths and len(allowed) == 1:
        paths.add((charge_idx,))

    names: list[str] = []
    suffix = "ide" if charge_atom.GetFormalCharge() < 0 else "ylium"
    for path in sorted(paths):
        name = _name_terminal_carbon_charge_path(mol, path, suffix)
        if name:
            names.append(name)
    return min(names, key=lambda name: (len(name), name)) if names else ""


def _name_terminal_carbon_charge_path(mol, path: Sequence[int], suffix: str) -> str:
    if not path:
        return ""
    parent = parent_chain_name(mol, path, as_substituent=False)
    if not parent:
        return ""
    prefixes = _path_substituent_prefixes(mol, path, skip_plain_hydrogen=True)
    if prefixes is None:
        return ""
    charged_parent = f"{parent}-1-{suffix}" if len(path) > 1 else f"{parent}{suffix}"
    prefix_text = _format_prefixes(prefixes)
    return _join_prefix_parent(prefix_text, charged_parent) if prefix_text else charged_parent


def _name_terminal_nitronate(mol) -> str:
    if mol.GetRingInfo().NumRings():
        return ""
    motifs: list[tuple[int, int, tuple[int, int]]] = []
    for nitrogen in mol.GetAtoms():
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 1 or nitrogen.IsInRing():
            continue
        nidx = nitrogen.GetIdx()
        carbon_idx = None
        oxido_oxygens = []
        ok = True
        for nbr in nitrogen.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(nidx, nbr.GetIdx())
            if bond is None:
                ok = False
                break
            if nbr.GetSymbol() == "C" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
                if carbon_idx is not None:
                    ok = False
                    break
                carbon_idx = nbr.GetIdx()
            elif (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() < 0
                and bond.GetBondType() == Chem.BondType.SINGLE
                and len([other for other in nbr.GetNeighbors() if other.GetAtomicNum() > 1]) == 1
            ):
                oxido_oxygens.append(nbr.GetIdx())
            else:
                ok = False
                break
        if ok and carbon_idx is not None and len(oxido_oxygens) == 2:
            motifs.append((nidx, carbon_idx, tuple(sorted(oxido_oxygens))))
    if len(motifs) != 1:
        return ""
    nitrogen_idx, carbon_idx, oxido_oxygens = motifs[0]
    covered = {nitrogen_idx, *oxido_oxygens}
    covered.update(collect_subtree(mol, carbon_idx, nitrogen_idx))
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    carbon_name = branch_prefix(mol, carbon_idx, nitrogen_idx)
    return carbon_name + "azaniumdiolate" if carbon_name else ""


def _name_terminal_imine_azanide(mol) -> str:
    if mol.GetRingInfo().NumRings():
        return ""
    motifs: list[tuple[int, int]] = []
    for nitrogen in mol.GetAtoms():
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != -1 or nitrogen.IsInRing():
            continue
        nidx = nitrogen.GetIdx()
        heavy = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(heavy) != 1:
            continue
        carbon = mol.GetAtomWithIdx(heavy[0])
        bond = mol.GetBondBetweenAtoms(nidx, carbon.GetIdx())
        if (
            carbon.GetSymbol() == "C"
            and carbon.GetFormalCharge() == 0
            and bond is not None
            and bond.GetBondType() == Chem.BondType.DOUBLE
        ):
            motifs.append((nidx, carbon.GetIdx()))
    if len(motifs) != 1:
        return ""
    nitrogen_idx, carbon_idx = motifs[0]
    covered = {nitrogen_idx}
    covered.update(collect_subtree(mol, carbon_idx, nitrogen_idx))
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    carbon_name = branch_prefix(mol, carbon_idx, nitrogen_idx)
    return carbon_name + "azanide" if carbon_name else ""


def _name_aminodiazenide(mol) -> str:
    if mol.GetRingInfo().NumRings():
        return ""
    if sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != -1:
        return ""
    motifs: list[tuple[int, int, int]] = []
    for terminal in mol.GetAtoms():
        if terminal.GetSymbol() != "N" or terminal.GetFormalCharge() != -1 or terminal.IsInRing():
            continue
        tidx = terminal.GetIdx()
        heavy = [nbr.GetIdx() for nbr in terminal.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(heavy) != 1:
            continue
        diazene_idx = heavy[0]
        diazene = mol.GetAtomWithIdx(diazene_idx)
        diazene_bond = mol.GetBondBetweenAtoms(tidx, diazene_idx)
        if (
            diazene.GetSymbol() != "N"
            or diazene.GetFormalCharge() != 0
            or diazene.IsInRing()
            or diazene_bond is None
            or diazene_bond.GetBondType() != Chem.BondType.DOUBLE
        ):
            continue
        owners = []
        for nbr in diazene.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == tidx or nbr.GetAtomicNum() <= 1:
                continue
            bond = mol.GetBondBetweenAtoms(diazene_idx, nidx)
            if (
                nbr.GetSymbol() == "N"
                and nbr.GetFormalCharge() == 0
                and not nbr.IsInRing()
                and bond is not None
                and bond.GetBondType() == Chem.BondType.SINGLE
            ):
                owners.append(nidx)
        if len(owners) == 1:
            motifs.append((tidx, diazene_idx, owners[0]))
    if len(motifs) != 1:
        return ""
    terminal_idx, diazene_idx, amino_idx = motifs[0]
    substituents = []
    covered = {terminal_idx, diazene_idx, amino_idx}
    for nbr in mol.GetAtomWithIdx(amino_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == diazene_idx or nbr.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(amino_idx, nidx)
        if nbr.GetSymbol() != "C" or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        prefix = branch_prefix(mol, nidx, amino_idx)
        if not prefix or prefix.startswith("(") or "[" in prefix or "]" in prefix or " " in prefix:
            return ""
        substituents.append(prefix)
        covered.update(collect_subtree(mol, nidx, amino_idx))
    if covered != set(range(mol.GetNumAtoms())) or len(substituents) != 2:
        return ""
    amino_prefix = _aminodiazenide_amino_prefix(substituents)
    return amino_prefix + "diazenide" if amino_prefix else ""


def _aminodiazenide_amino_prefix(substituents: Sequence[str]) -> str:
    counts = Counter(substituents)
    if len(counts) == 1:
        name, count = next(iter(counts.items()))
        if count != 2:
            return ""
        return f"di{name}amino" if name.isalpha() else f"bis({name})amino"
    if len(substituents) != 2:
        return ""
    first, second = sorted(substituents)
    if first.startswith("(") or second.startswith("("):
        return ""
    return f"{first}({second})amino"


def _name_dialkoxy_azanide(mol) -> str:
    if mol.GetRingInfo().NumRings():
        return ""
    motifs: list[tuple[int, tuple[int, int]]] = []
    for nitrogen in mol.GetAtoms():
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != -1 or nitrogen.IsInRing():
            continue
        nidx = nitrogen.GetIdx()
        heavy = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(heavy) != 2:
            continue
        oxygens = []
        for oxygen_idx in heavy:
            oxygen = mol.GetAtomWithIdx(oxygen_idx)
            bond = mol.GetBondBetweenAtoms(nidx, oxygen_idx)
            if (
                oxygen.GetSymbol() != "O"
                or oxygen.GetFormalCharge() != 0
                or oxygen.IsInRing()
                or bond is None
                or bond.GetBondType() != Chem.BondType.SINGLE
            ):
                break
            carbon_neighbors = [
                nbr.GetIdx()
                for nbr in oxygen.GetNeighbors()
                if nbr.GetIdx() != nidx and nbr.GetSymbol() == "C" and nbr.GetFormalCharge() == 0
            ]
            if len(carbon_neighbors) != 1:
                break
            oxygens.append((oxygen_idx, carbon_neighbors[0]))
        else:
            motifs.append((nidx, tuple(oxygens)))
    if len(motifs) != 1:
        return ""
    nitrogen_idx, oxygens = motifs[0]
    covered = {nitrogen_idx}
    oxy_names = []
    for oxygen_idx, carbon_idx in oxygens:
        covered.add(oxygen_idx)
        covered.update(collect_subtree(mol, carbon_idx, oxygen_idx))
        owner = branch_prefix(mol, carbon_idx, oxygen_idx)
        if not owner:
            return ""
        oxy_names.append(_alkyl_to_alkoxy(owner))
    if covered != set(range(mol.GetNumAtoms())) or not all(oxy_names):
        return ""
    counts = {name: oxy_names.count(name) for name in set(oxy_names)}
    if len(counts) == 1:
        name = next(iter(counts))
        if name.startswith("("):
            return f"bis({name})azanide"
        return f"di{name}azanide"
    first, second = sorted(oxy_names)
    return f"{first}({second})azanide"


def _name_alkoxy_azanide(mol) -> str:
    if mol.GetRingInfo().NumRings():
        return ""
    motifs: list[tuple[int, int, int]] = []
    for nitrogen in mol.GetAtoms():
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != -1 or nitrogen.IsInRing():
            continue
        nidx = nitrogen.GetIdx()
        heavy = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(heavy) != 1:
            continue
        oxygen = mol.GetAtomWithIdx(heavy[0])
        if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0 or oxygen.IsInRing():
            continue
        bond = mol.GetBondBetweenAtoms(nidx, oxygen.GetIdx())
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        carbon_neighbors = [
            nbr.GetIdx()
            for nbr in oxygen.GetNeighbors()
            if nbr.GetIdx() != nidx and nbr.GetSymbol() == "C" and nbr.GetFormalCharge() == 0
        ]
        if len(carbon_neighbors) == 1:
            motifs.append((nidx, oxygen.GetIdx(), carbon_neighbors[0]))
    if len(motifs) != 1:
        return ""
    nitrogen_idx, oxygen_idx, carbon_idx = motifs[0]
    covered = {nitrogen_idx, oxygen_idx}
    covered.update(collect_subtree(mol, carbon_idx, oxygen_idx))
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    owner = branch_prefix(mol, carbon_idx, oxygen_idx)
    return owner + "oxyazanide" if owner else ""


def _name_terminal_diacyl_azanide(mol) -> str:
    if mol.GetRingInfo().NumRings():
        return ""
    records: list[tuple[int, int, int]] = []
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
            continue
        cidx = carbon.GetIdx()
        oxo_idx = None
        azanide_idx = None
        ok = True
        for nbr in carbon.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(cidx, nbr.GetIdx())
            if bond is None:
                ok = False
                break
            if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
                if oxo_idx is not None:
                    ok = False
                    break
                oxo_idx = nbr.GetIdx()
            elif nbr.GetSymbol() == "N" and nbr.GetFormalCharge() == -1 and bond.GetBondType() == Chem.BondType.SINGLE:
                heavy = [atom.GetIdx() for atom in nbr.GetNeighbors() if atom.GetAtomicNum() > 1]
                if heavy != [cidx] or azanide_idx is not None:
                    ok = False
                    break
                azanide_idx = nbr.GetIdx()
            elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                continue
            else:
                ok = False
                break
        if ok and oxo_idx is not None and azanide_idx is not None:
            records.append((cidx, oxo_idx, azanide_idx))
    if len(records) != 2:
        return ""
    first, second = records
    carbon_allowed = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C" and atom.GetFormalCharge() == 0 and not atom.IsInRing()
    }
    path = _shortest_path_in_allowed_graph(_acyclic_graph_view(mol), first[0], second[0], carbon_allowed)
    if not path or set(path) != carbon_allowed:
        return ""
    covered = set(path) | {first[1], first[2], second[1], second[2]}
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    parent = parent_chain_name(mol, path, as_substituent=False)
    return parent + "edioylbis(azanide)" if parent else ""


def _name_geminal_hydroxy_diphosphonic_acid(mol) -> str:
    if mol.GetRingInfo().NumRings():
        return ""
    centers = []
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
            continue
        cidx = carbon.GetIdx()
        phosphonic_groups = []
        hydroxy_idx = None
        for nbr in carbon.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(cidx, nbr.GetIdx())
            if bond is None:
                continue
            if nbr.GetSymbol() == "P" and bond.GetBondType() == Chem.BondType.SINGLE:
                group_atoms = _neutral_phosphonic_acid_group_atoms(mol, nbr.GetIdx(), cidx)
                if group_atoms:
                    phosphonic_groups.append((nbr.GetIdx(), group_atoms))
            elif (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() == 0
                and bond.GetBondType() == Chem.BondType.SINGLE
                and nbr.GetTotalNumHs() == 1
                and len([atom for atom in nbr.GetNeighbors() if atom.GetAtomicNum() > 1]) == 1
            ):
                hydroxy_idx = nbr.GetIdx()
        if len(phosphonic_groups) == 2 and hydroxy_idx is not None:
            centers.append((cidx, tuple(phosphonic_groups), hydroxy_idx))
    if len(centers) != 1:
        return ""
    center_idx, phosphonic_groups, hydroxy_idx = centers[0]
    view = _acyclic_graph_view(mol)
    carbon_allowed = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C" and atom.GetFormalCharge() == 0 and not atom.IsInRing()
    }
    endpoints = [
        idx
        for idx in carbon_allowed
        if idx == center_idx or sum(1 for nidx in view.neighbors[idx] if nidx in carbon_allowed) <= 1
    ]
    paths = []
    best_len = 0
    for endpoint in endpoints:
        if endpoint == center_idx:
            continue
        path = _shortest_path_in_allowed_graph(view, center_idx, endpoint, carbon_allowed)
        if not path:
            continue
        if len(path) > best_len:
            best_len = len(path)
            paths.clear()
        if len(path) == best_len:
            paths.append(path)
    if not paths and carbon_allowed == {center_idx}:
        paths.append((center_idx,))

    skipped_atoms = {hydroxy_idx}
    for _pidx, group_atoms in phosphonic_groups:
        skipped_atoms.update(group_atoms)
    names = []
    for path in sorted(set(paths)):
        prefixes = _path_substituent_prefixes(
            mol,
            path,
            skip_atoms=skipped_atoms,
            skip_plain_hydrogen=True,
        )
        if prefixes is None:
            continue
        covered = set(path) | skipped_atoms
        path_set = set(path)
        for idx in path:
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx in path_set or nidx in skipped_atoms:
                    continue
                covered.update(collect_subtree(mol, nidx, idx))
        if covered != set(range(mol.GetNumAtoms())):
            continue
        parent = _standalone_saturated_chain_parent(parent_chain_name(mol, path, as_substituent=False))
        if not parent or not parent.endswith("e"):
            continue
        core = f"1-hydroxy-1,1-diphosphono{parent}"
        prefix_text = _format_prefixes(prefixes)
        names.append(_join_prefix_parent(prefix_text, core) if prefix_text else core)
    return min(names, key=lambda name: (len(name), name)) if names else ""


def _neutral_phosphonic_acid_group_atoms(mol, phosphorus_idx: int, carbon_idx: int) -> frozenset[int]:
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)
    if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0 or phosphorus.IsInRing():
        return frozenset()
    double_oxygen = 0
    hydroxy_oxygens = []
    atoms = {phosphorus_idx}
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(phosphorus_idx, nidx)
        if nidx == carbon_idx:
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return frozenset()
            continue
        if bond is None or nbr.GetSymbol() != "O" or nbr.GetFormalCharge() != 0:
            return frozenset()
        heavy = [atom.GetIdx() for atom in nbr.GetNeighbors() if atom.GetAtomicNum() > 1]
        if bond.GetBondType() == Chem.BondType.DOUBLE and len(heavy) == 1:
            double_oxygen += 1
            atoms.add(nidx)
        elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetTotalNumHs() == 1 and len(heavy) == 1:
            hydroxy_oxygens.append(nidx)
            atoms.add(nidx)
        else:
            return frozenset()
    if double_oxygen != 1 or len(hydroxy_oxygens) != 2:
        return frozenset()
    return frozenset(atoms)


def _name_hydroxy_alkane_diyl_diphosphonic_acid(mol) -> str:
    if mol.GetRingInfo().NumRings():
        return ""
    records = []
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing():
            continue
        cidx = carbon.GetIdx()
        phosphonic_group = None
        hydroxy_idx = None
        for nbr in carbon.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(cidx, nbr.GetIdx())
            if bond is None:
                continue
            if nbr.GetSymbol() == "P" and bond.GetBondType() == Chem.BondType.SINGLE:
                group_atoms = _neutral_phosphonic_acid_group_atoms(mol, nbr.GetIdx(), cidx)
                if not group_atoms or phosphonic_group is not None:
                    return ""
                phosphonic_group = (nbr.GetIdx(), group_atoms)
            elif (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() == 0
                and bond.GetBondType() == Chem.BondType.SINGLE
                and nbr.GetTotalNumHs() == 1
                and len([atom for atom in nbr.GetNeighbors() if atom.GetAtomicNum() > 1]) == 1
            ):
                if hydroxy_idx is not None:
                    return ""
                hydroxy_idx = nbr.GetIdx()
        if phosphonic_group is not None:
            records.append((cidx, phosphonic_group, hydroxy_idx))
    if len(records) != 2:
        return ""
    hydroxy_records = [record for record in records if record[2] is not None]
    if len(hydroxy_records) != 1:
        return ""
    view = _acyclic_graph_view(mol)
    carbon_allowed = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C" and atom.GetFormalCharge() == 0 and not atom.IsInRing()
    }
    first, second = records
    path = _shortest_path_in_allowed_graph(view, first[0], second[0], carbon_allowed)
    if not path or len(path) < 2 or set(path) != carbon_allowed:
        return ""
    hydroxy_carbon = hydroxy_records[0][0]
    if path[-1] == hydroxy_carbon:
        path = tuple(reversed(path))
    if path[0] != hydroxy_carbon:
        return ""
    skipped_atoms = {hydroxy_records[0][2]}
    for _cidx, (_pidx, group_atoms), _hydroxy_idx in records:
        skipped_atoms.update(group_atoms)
    if set(path) | skipped_atoms != set(range(mol.GetNumAtoms())):
        return ""
    parent = _standalone_saturated_chain_parent(parent_chain_name(mol, path, as_substituent=False))
    if not parent or not parent.endswith("e"):
        return ""
    return f"1-hydroxy{parent}-1,{len(path)}-diyldiphosphonic acid"


def _name_tris_hydroxy_alkoxy_phosphoryl_amine(mol) -> str:
    if mol.GetRingInfo().NumRings():
        return ""
    records = []
    for nitrogen in mol.GetAtoms():
        if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing():
            continue
        if nitrogen.GetTotalNumHs() != 0:
            continue
        heavy = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(heavy) != 3:
            continue
        ligand_records = []
        for pidx in heavy:
            bond = mol.GetBondBetweenAtoms(nitrogen.GetIdx(), pidx)
            phosphorus = mol.GetAtomWithIdx(pidx)
            if (
                phosphorus.GetSymbol() != "P"
                or phosphorus.GetFormalCharge() != 0
                or phosphorus.IsInRing()
                or bond is None
                or bond.GetBondType() != Chem.BondType.SINGLE
            ):
                break
            ligand = _hydroxy_alkoxy_phosphoryl_ligand(mol, pidx, nitrogen.GetIdx())
            if ligand is None:
                break
            ligand_records.append(ligand)
        else:
            records.append((nitrogen.GetIdx(), tuple(ligand_records)))
    if len(records) != 1:
        return ""
    nitrogen_idx, ligand_records = records[0]
    covered = {nitrogen_idx}
    ligand_names = []
    for ligand_name, ligand_atoms in ligand_records:
        ligand_names.append(ligand_name)
        covered.update(ligand_atoms)
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    counts = Counter(ligand_names)
    if len(counts) != 1:
        return ""
    ligand_name, count = next(iter(counts.items()))
    if count != 3 or not ligand_name:
        return ""
    return f"tris(hydroxy({ligand_name})phosphoryl)amine"


def _hydroxy_alkoxy_phosphoryl_ligand(mol, phosphorus_idx: int, nitrogen_idx: int) -> tuple[str, set[int]] | None:
    double_oxygen = None
    hydroxy_oxygen = None
    alkoxy_record = None
    atoms = {phosphorus_idx}
    for nbr in mol.GetAtomWithIdx(phosphorus_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(phosphorus_idx, nidx)
        if bond is None or nbr.GetSymbol() != "O" or nbr.GetFormalCharge() != 0 or nbr.IsInRing():
            return None
        heavy = [atom.GetIdx() for atom in nbr.GetNeighbors() if atom.GetAtomicNum() > 1]
        if bond.GetBondType() == Chem.BondType.DOUBLE and heavy == [phosphorus_idx]:
            if double_oxygen is not None:
                return None
            double_oxygen = nidx
            atoms.add(nidx)
        elif bond.GetBondType() == Chem.BondType.SINGLE and heavy == [phosphorus_idx] and nbr.GetTotalNumHs() == 1:
            if hydroxy_oxygen is not None:
                return None
            hydroxy_oxygen = nidx
            atoms.add(nidx)
        elif bond.GetBondType() == Chem.BondType.SINGLE and len(heavy) == 2 and nbr.GetTotalNumHs() == 0:
            if alkoxy_record is not None:
                return None
            carbon_neighbors = [
                idx
                for idx in heavy
                if idx != phosphorus_idx
                and mol.GetAtomWithIdx(idx).GetSymbol() == "C"
                and mol.GetAtomWithIdx(idx).GetFormalCharge() == 0
            ]
            if len(carbon_neighbors) != 1:
                return None
            carbon_idx = carbon_neighbors[0]
            owner_name = branch_prefix(mol, carbon_idx, nidx)
            alkoxy_name = _alkyl_to_alkoxy(owner_name) if owner_name else ""
            if not alkoxy_name:
                return None
            alkoxy_atoms = {nidx}
            alkoxy_atoms.update(collect_subtree(mol, carbon_idx, nidx))
            alkoxy_record = (alkoxy_name, alkoxy_atoms)
            atoms.update(alkoxy_atoms)
        else:
            return None
    if double_oxygen is None or hydroxy_oxygen is None or alkoxy_record is None:
        return None
    return alkoxy_record[0], atoms


def _name_terminal_nitrone(mol) -> str:
    motif = _terminal_nitrone_motif(mol)
    if motif is None:
        return ""
    carbon_idx, nitrogen_idx = motif
    paths = _terminal_nitrone_parent_paths(mol, carbon_idx)
    for path in paths:
        name = _name_terminal_nitrone_path(mol, path, carbon_idx, nitrogen_idx)
        if name:
            return name
    return ""


def _terminal_nitrone_motif(mol) -> tuple[int, int] | None:
    if mol.GetRingInfo().NumRings():
        return None
    motifs = []
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
        if ok and len(imine_carbons) == 1 and len(oxido_oxygens) == 1:
            motifs.append((imine_carbons[0], nidx))
    if len(motifs) != 1:
        return None
    carbon_idx, nitrogen_idx = motifs[0]
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.IsInRing() or carbon.GetFormalCharge() != 0:
        return None
    carbon_heavy = [
        nbr.GetIdx()
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != nitrogen_idx and nbr.GetAtomicNum() > 1
    ]
    if len(carbon_heavy) > 1:
        return None
    return carbon_idx, nitrogen_idx


def _terminal_nitrone_parent_paths(mol, carbon_idx: int) -> list[tuple[int, ...]]:
    view = _acyclic_graph_view(mol)
    allowed = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C"
        and not atom.IsInRing()
        and atom.GetFormalCharge() == 0
        and not atom.GetIsotope()
    }
    if carbon_idx not in allowed:
        return []
    endpoints = [
        idx
        for idx in allowed
        if idx == carbon_idx or sum(1 for nidx in view.neighbors[idx] if nidx in allowed) <= 1
    ]
    paths = []
    best_len = 0
    for endpoint in endpoints:
        path = _shortest_path_in_allowed_graph(view, carbon_idx, endpoint, allowed)
        if not path:
            continue
        path_len = len(path)
        if path_len > best_len:
            best_len = path_len
            paths.clear()
        if path_len == best_len:
            paths.append(tuple(path))
    return sorted(set(paths), key=lambda path: (-len(path), path))


def _name_terminal_nitrone_path(
    mol,
    path: Sequence[int],
    carbon_idx: int,
    nitrogen_idx: int,
) -> str:
    if not path or path[0] != carbon_idx:
        return ""
    root = ROOTS.get(len(path))
    if not root:
        return ""
    prefixes = _path_substituent_prefixes(
        mol,
        path,
        skip_bonds={_bond_key(carbon_idx, nitrogen_idx)},
        skip_plain_hydrogen=True,
    )
    if prefixes is None:
        return ""
    name = root + "ylideneazaniumolate"
    prefix_text = _format_prefixes(prefixes)
    return _join_prefix_parent(prefix_text, name) if prefix_text else name


def _acyclic_functional_profile(mol) -> dict[str, bool]:
    cached = mol_cache_get(mol, "acyclic_functional_profile")
    if cached is not None:
        return cached
    has_c_c_o_cumulene = False
    has_c_s_double = False
    has_c_s_n_thioamide = False
    has_c_sulfone_cumulene = False
    has_protonated_carbonyl = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        cidx = atom.GetIdx()
        double_sulfurs = []
        single_n = False
        double_c = []
        for nbr in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(cidx, nbr.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                if nbr.GetSymbol() == "S":
                    double_sulfurs.append(nbr.GetIdx())
                elif nbr.GetSymbol() == "O" and nbr.GetFormalCharge() > 0:
                    has_protonated_carbonyl = True
                elif nbr.GetSymbol() == "C":
                    double_c.append(nbr.GetIdx())
            elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetSymbol() == "N":
                single_n = True
        if double_sulfurs:
            has_c_s_double = True
            for sulfur_idx in double_sulfurs:
                if _is_dioxosulfane_sulfur(mol, sulfur_idx):
                    has_c_sulfone_cumulene = True
            if single_n:
                has_c_s_n_thioamide = True
        for carbon_idx in double_c:
            if any(
                onbr.GetSymbol() == "O"
                and mol.GetBondBetweenAtoms(carbon_idx, onbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                for onbr in mol.GetAtomWithIdx(carbon_idx).GetNeighbors()
                if onbr.GetIdx() != cidx
            ):
                has_c_c_o_cumulene = True
    profile = {
        "oxidanium": has_protonated_carbonyl,
        "ketene": has_c_c_o_cumulene,
        "thioamide": has_c_s_n_thioamide,
        "ylidene_dioxosulfane": has_c_sulfone_cumulene,
        "thione": has_c_s_double,
    }
    return mol_cache_set(mol, "acyclic_functional_profile", profile)


def _name_oxidanium(mol) -> str:
    motifs = _oxidanium_motifs(mol)
    if len(motifs) != 1:
        return ""
    carbon_idx, oxygen_idx = motifs[0]
    paths = _oxidanium_parent_paths(mol, carbon_idx)
    orientations = _oriented_paths_by_quality(mol, paths)
    index = 0
    while index < len(orientations):
        descriptor = orientations[index][0]
        names = []
        while index < len(orientations) and orientations[index][0] == descriptor:
            _descriptor, oriented = orientations[index]
            name = _name_oxidanium_path(mol, oriented, carbon_idx, oxygen_idx)
            if name:
                names.append(name)
            index += 1
        if names:
            return min(names)
    return ""


def _oxidanium_motifs(mol) -> list[tuple[int, int]]:
    motifs = []
    for atom in mol.GetAtoms():
        cidx = atom.GetIdx()
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() != "O" or nbr.GetFormalCharge() <= 0:
                continue
            bond = mol.GetBondBetweenAtoms(cidx, nbr.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                motifs.append((cidx, nbr.GetIdx()))
    return motifs


def _oxidanium_parent_paths(mol, carbon_idx: int) -> list[tuple[int, ...]]:
    allowed = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if not atom.IsInRing() and can_be_in_parent_skeleton(mol, atom.GetIdx())
    }
    if carbon_idx not in allowed:
        return []
    view = _acyclic_graph_view(mol)
    endpoints = [
        idx
        for idx in allowed
        if idx == carbon_idx
        or (
            view.symbols[idx] == "C"
            and not is_carbonyl_carbon(mol, idx)
            and sum(1 for nidx in view.neighbors[idx] if nidx in allowed) <= 1
        )
    ]
    if len(endpoints) < 2:
        endpoints = [idx for idx in allowed if idx == carbon_idx or view.symbols[idx] == "C"]
    paths: set[tuple[int, ...]] = set()
    best_len = 0
    for endpoint in endpoints:
        if endpoint == carbon_idx:
            continue
        path = _shortest_path_in_allowed_graph(view, carbon_idx, endpoint, allowed)
        if not path:
            continue
        path_len = len(path)
        if path_len > best_len:
            best_len = path_len
            paths.clear()
        if path_len == best_len:
            paths.add(tuple(path))
    if not paths:
        paths.add((carbon_idx,))
    return sorted(paths)


def _name_oxidanium_path(mol, path: Sequence[int], carbon_idx: int, oxygen_idx: int) -> str:
    if carbon_idx not in path:
        return ""
    prefixes = _path_substituent_prefixes(
        mol,
        path,
        skip_bonds={_bond_key(carbon_idx, oxygen_idx)},
    )
    if prefixes is None:
        return ""

    replacement_text = replacement_prefixes(mol, path)
    parent = _standalone_saturated_chain_parent(
        parent_chain_name(mol, path, as_substituent=False)
    )
    if not parent or not parent.endswith("e"):
        return ""
    locant = path.index(carbon_idx) + 1
    parent = _join_replacement_parent(replacement_text, parent) if replacement_text else parent
    name = f"{parent[:-1]}-{locant}-ylideneoxidanium"
    prefix_text = _format_prefixes(prefixes)
    return _join_prefix_parent(prefix_text, name) if prefix_text else _preferred_parent_stem(name)


def _best_acyclic_record_from_paths(mol, paths: Sequence[Sequence[int]]) -> AcyclicNameRecord:
    orientations = _oriented_paths_by_quality(mol, paths)
    _record_acyclic_filter_stage(mol, "path_facts", len(paths))
    _record_acyclic_filter_stage(mol, "orientation_facts", len(orientations))
    prefer_complete_stereo = _prefer_complete_acyclic_stereo_path(mol)
    first_records: list[AcyclicNameRecord] | None = None
    index = 0
    while index < len(orientations):
        descriptor = orientations[index][0]
        descriptor_orientations = []
        while index < len(orientations) and orientations[index][0] == descriptor:
            _descriptor, oriented = orientations[index]
            descriptor_orientations.append(oriented)
            index += 1
        render_set = _acyclic_orientation_render_set(mol, descriptor_orientations)
        _record_acyclic_filter_stage(mol, "orientation_render_set", len(render_set))
        _record_acyclic_filter_stage(
            mol,
            "orientations_suppressed_before_render",
            max(0, len(descriptor_orientations) - len(render_set)),
        )
        rendered_names: list[tuple[tuple[int, ...], str]] = []
        for oriented in render_set:
            name = _name_from_parent_path(mol, oriented, as_substituent=False)
            if name:
                rendered_names.append((oriented, name))
        records = [
            AcyclicNameRecord(name, _rendered_path_stereo_locants(mol, oriented))
            for oriented, name in rendered_names
        ]
        if records:
            if not prefer_complete_stereo:
                return min(records, key=lambda record: record.name)
            complete_stereo_names = [
                AcyclicNameRecord(name, _rendered_path_stereo_locants(mol, oriented))
                for oriented, name in rendered_names
                if _complete_path_stereo_prefix(mol, oriented)
            ]
            if complete_stereo_names:
                return min(complete_stereo_names, key=lambda record: record.name)
            if first_records is None:
                first_records = records
    if first_records:
        return min(first_records, key=lambda record: record.name)
    return AcyclicNameRecord("")


def _rendered_path_stereo_locants(mol, path: Sequence[int]) -> tuple[tuple[int, str], ...]:
    carboxy_oriented = _terminal_carboxy_stereo_path(mol, path)
    if carboxy_oriented is not None:
        return _path_stereo_locants(carboxy_oriented, start=2)
    return _path_stereo_locants(path)


def _terminal_carboxy_stereo_path(mol, path: Sequence[int]) -> tuple[int, ...] | None:
    prefixes = _path_substituent_prefixes(mol, path, skip_plain_hydrogen=True)
    if prefixes is None or replacement_prefixes(mol, path):
        return None
    n = len(path)
    terminal_groups = [
        (loc, prefix)
        for loc, prefix in prefixes
        if prefix in {"carboxy", "carboxylato", "carbamyl", "formyl"} and loc in {"1", str(n)}
    ]
    terminal = [
        item for item in terminal_groups if item[1] in {"carboxy", "carboxylato"}
    ] or [item for item in terminal_groups if item[1] == "carbamyl"] or [
        item for item in terminal_groups if item[1] == "formyl"
    ]
    if _diterminal_carboxy_suffix_parent_name(mol, path, prefixes, terminal):
        return tuple(path)
    if len(terminal) != 1:
        return None
    terminal_loc, _terminal_prefix = terminal[0]
    if not _terminal_carboxy_suffix_parent_name(mol, path, prefixes):
        return None
    return tuple(reversed(path)) if terminal_loc == str(n) else tuple(path)


def _path_stereo_locants(path: Sequence[int], *, start: int = 1) -> tuple[tuple[int, str], ...]:
    return tuple((int(atom_idx), str(position)) for position, atom_idx in enumerate(path, start=start))


def _prefer_complete_acyclic_stereo_path(mol) -> bool:
    return bool(assigned_atom_stereochemistry(mol, Chem, atom_order="idx") or _has_assigned_path_bond_stereo(mol))


def _has_assigned_path_bond_stereo(mol) -> bool:
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return True
    return False


def _acyclic_orientation_render_set(
    mol,
    orientations: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if len(orientations) <= MAX_ACYCLIC_ORIENTATIONS_PER_DESCRIPTOR:
        return tuple(tuple(path) for path in orientations)
    return tuple(
        sorted((tuple(path) for path in orientations), key=lambda path: _acyclic_orientation_fact_sort_key(mol, path))[
            :MAX_ACYCLIC_ORIENTATIONS_PER_DESCRIPTOR
        ]
    )


def _acyclic_orientation_fact_sort_key(mol, path: Sequence[int]) -> tuple:
    path_set = set(path)
    view = _acyclic_graph_view(mol)
    external_edges = 0
    hetero_atoms = 0
    unsaturation = 0
    for idx in path:
        if view.symbols[idx] != "C":
            hetero_atoms += 1
        for nidx in view.neighbors[idx]:
            if nidx not in path_set:
                external_edges += 1
    for a, b in zip(path, path[1:]):
        if view.bond_orders.get(_bond_key(a, b), 1.0) != 1.0:
            unsaturation += 1
    return (
        external_edges,
        hetero_atoms,
        -unsaturation,
        tuple(path),
    )


def _name_ketene(mol) -> str:
    motifs = _ketene_motifs(mol)
    if len(motifs) != 1:
        return ""
    alpha_idx, carbonyl_idx, oxygen_idx = motifs[0]
    paths = [path for path in _candidate_parent_paths(mol) if alpha_idx in path]
    orientations = _oriented_paths_by_quality(mol, paths)
    index = 0
    while index < len(orientations):
        descriptor = orientations[index][0]
        names = []
        while index < len(orientations) and orientations[index][0] == descriptor:
            _descriptor, oriented = orientations[index]
            if alpha_idx in oriented:
                name = _name_ketene_path(mol, oriented, alpha_idx, carbonyl_idx, oxygen_idx)
                if name:
                    names.append(name)
            index += 1
        if names:
            return min(names)
    return ""


def _ketene_motifs(mol) -> list[tuple[int, int, int]]:
    motifs = []
    for atom in mol.GetAtoms():
        alpha_idx = atom.GetIdx()
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
            continue
        for nbr in atom.GetNeighbors():
            carbonyl_idx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(alpha_idx, carbonyl_idx)
            if nbr.GetSymbol() != "C" or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            oxygens = [
                onbr.GetIdx()
                for onbr in nbr.GetNeighbors()
                if onbr.GetIdx() != alpha_idx
                and onbr.GetSymbol() == "O"
                and mol.GetBondBetweenAtoms(carbonyl_idx, onbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            ]
            if len(oxygens) == 1:
                motifs.append((alpha_idx, carbonyl_idx, oxygens[0]))
    return motifs


def _name_ketene_path(
    mol, path: Sequence[int], alpha_idx: int, carbonyl_idx: int, oxygen_idx: int
) -> str:
    prefixes = _path_substituent_prefixes(
        mol,
        path,
        skip_bonds={_bond_key(alpha_idx, carbonyl_idx)},
    )
    if prefixes is None:
        return ""

    replacement_text = replacement_prefixes(mol, path)
    parent = _standalone_saturated_chain_parent(
        parent_chain_name(mol, path, as_substituent=False)
    )
    if not parent:
        return ""
    if replacement_text:
        parent = _join_replacement_parent(replacement_text, parent)
    if not parent.endswith("e"):
        return ""
    locant = path.index(alpha_idx) + 1
    ketene_parent = f"{parent[:-1]}-{locant}-ylidenemethanone"
    prefix_text = _format_prefixes(prefixes)
    return _join_prefix_parent(prefix_text, ketene_parent) if prefix_text else ketene_parent


def _candidate_parent_paths(mol, allow_ring_atoms: bool = True) -> list[tuple[int, ...]]:
    view = _acyclic_graph_view(mol)
    allowed = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if (allow_ring_atoms or not atom.IsInRing())
        and can_be_in_parent_skeleton(mol, atom.GetIdx())
    ]
    allowed_set = set(allowed)
    if not allowed:
        return []

    endpoints = [
        idx
        for idx in allowed
        if view.symbols[idx] == "C"
        and not is_carbonyl_carbon(mol, idx)
    ]
    if len(endpoints) < 2:
        endpoints = [idx for idx in allowed if view.symbols[idx] == "C"]
    if not endpoints:
        endpoints = allowed

    endpoint_groups = _allowed_endpoint_groups(mol, allowed_set, endpoints, view)
    paths: set[tuple[int, ...]] = set()
    best_len = 0
    for endpoint_group in endpoint_groups:
        for a, b in combinations(endpoint_group, 2):
            path = _shortest_path_in_allowed_graph(view, a, b, allowed_set)
            if not path:
                continue
            path_len = len(path)
            if path_len > best_len:
                best_len = path_len
            paths.add(tuple(path))
    if not paths:
        for idx in endpoints:
            paths.add((idx,))
    min_len = max(1, best_len - 4)
    return sorted(
        (path for path in paths if len(path) >= min_len),
        key=lambda path: (-len(path), path),
    )[:MAX_ACYCLIC_PARENT_PATHS]


def _allowed_endpoint_groups(
    mol,
    allowed_set: set[int],
    endpoints: Sequence[int],
    view: _AcyclicGraphView | None = None,
) -> list[tuple[int, ...]]:
    """Group endpoints by the allowed skeleton graph before shortest-path probes."""

    view = view or _acyclic_graph_view(mol)
    endpoint_set = set(endpoints)
    remaining = set(allowed_set)
    groups: list[tuple[int, ...]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            idx = stack.pop()
            for nidx in view.neighbors[idx]:
                if nidx not in remaining:
                    continue
                remaining.remove(nidx)
                component.add(nidx)
                stack.append(nidx)
        group = tuple(sorted(component & endpoint_set))
        if len(group) >= 2:
            groups.append(group)
    return groups


def _shortest_path_in_allowed_graph(
    view: _AcyclicGraphView,
    start: int,
    target: int,
    allowed_set: set[int],
) -> tuple[int, ...]:
    if start == target:
        return (start,)
    if start not in allowed_set or target not in allowed_set:
        return ()
    queue = [start]
    previous: dict[int, int | None] = {start: None}
    for idx in queue:
        for nidx in view.neighbors[idx]:
            if nidx not in allowed_set or nidx in previous:
                continue
            previous[nidx] = idx
            if nidx == target:
                return _reconstruct_path(previous, target)
            queue.append(nidx)
    return ()


def _reconstruct_path(previous: dict[int, int | None], target: int) -> tuple[int, ...]:
    path = []
    current: int | None = target
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    return tuple(path)


def _oriented_paths_by_quality(mol, paths: Sequence[Sequence[int]]) -> list[tuple[tuple, tuple[int, ...]]]:
    orientations = []
    seen = set()
    for path in paths:
        forward = tuple(path)
        reverse = tuple(reversed(path))
        for oriented in (forward, reverse):
            if oriented in seen:
                continue
            seen.add(oriented)
            orientations.append((_path_quality_descriptor(mol, oriented), oriented))
    orientations.sort(key=lambda item: item[0])
    return orientations


def _path_quality_descriptor(mol, path: Sequence[int]) -> tuple:
    path = tuple(path)
    cache = mol_cache_dict(mol, "path_quality_cache")
    cached = cache.get(path)
    if cached is not None:
        return cached
    view = _acyclic_graph_view(mol)
    path_set = set(path)
    locants = []
    carbon_count = 0
    suffix_bonus = 0
    for pos, idx in enumerate(path, start=1):
        if view.symbols[idx] == "C":
            carbon_count += 1
        if view.symbols[idx] != "C":
            locants.append(pos)
        for nidx in view.neighbors[idx]:
            if nidx not in path_set:
                locants.append(pos)
    for bond_locant, (a, b) in enumerate(zip(path, path[1:]), start=1):
        if view.bond_orders.get(_bond_key(a, b), 1.0) != 1.0:
            locants.append(bond_locant)
            suffix_bonus += 1
    charge_prefix_count = _path_charge_preserving_prefix_count(mol, path)
    descriptor = (-charge_prefix_count, -len(path), -suffix_bonus, tuple(locants), -carbon_count)
    cache[path] = descriptor
    return descriptor


def _path_charge_preserving_prefix_count(mol, path: Sequence[int]) -> int:
    charged_atoms = _formal_charge_atom_indices(mol)
    if not charged_atoms:
        return 0
    path_set = set(path)
    if set(charged_atoms) <= path_set:
        return 0
    cache = mol_cache_dict(mol, "acyclic_path_charge_preserving_prefix_count")
    cache_key = tuple(sorted(path))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    if not _path_has_charge_preserving_prefix_candidate(mol, path, path_set):
        cache[cache_key] = 0
        return 0
    prefixes = _path_substituent_prefixes(mol, path, skip_plain_hydrogen=True)
    if prefixes is None:
        cache[cache_key] = 0
        return 0
    count = sum(1 for _locant, prefix in prefixes if _prefix_preserves_formal_charge(prefix))
    cache[cache_key] = count
    return count


def _has_formal_charge(mol) -> bool:
    cached = mol_cache_get(mol, "has_formal_charge")
    if cached is not None:
        return cached
    return mol_cache_set(mol, "has_formal_charge", any(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _formal_charge_atom_indices(mol) -> tuple[int, ...]:
    cached = mol_cache_get(mol, "formal_charge_atom_indices")
    if cached is not None:
        return cached
    return mol_cache_set(
        mol,
        "formal_charge_atom_indices",
        tuple(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge()),
    )


def _prefix_preserves_formal_charge(prefix: str) -> bool:
    return any(term in prefix for term in _CHARGE_PRESERVING_PREFIX_TERMS)


def _path_has_charge_preserving_prefix_candidate(mol, path: Sequence[int], path_set: set[int]) -> bool:
    roots = _charge_preserving_prefix_candidate_roots(mol)
    if not roots:
        return False
    view = _acyclic_graph_view(mol)
    for idx in path:
        for nidx in view.neighbors[idx]:
            if nidx not in path_set and (idx, nidx) in roots:
                return True
    return False


def _charge_preserving_prefix_candidate_roots(mol) -> frozenset[tuple[int, int]]:
    cached = mol_cache_get(mol, "acyclic_charge_preserving_prefix_candidate_roots")
    if cached is not None:
        return cached
    negative_oxygen_atoms = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 8 and atom.GetFormalCharge() < 0
    }
    if not negative_oxygen_atoms:
        return mol_cache_set(mol, "acyclic_charge_preserving_prefix_candidate_roots", frozenset())
    view = _acyclic_graph_view(mol)
    center_atoms = {
        idx
        for idx, symbol in enumerate(view.symbols)
        if symbol in _CHARGE_PRESERVING_CENTER_SYMBOLS
    }
    if not center_atoms:
        return mol_cache_set(mol, "acyclic_charge_preserving_prefix_candidate_roots", frozenset())
    roots: set[tuple[int, int]] = set()
    for parent_idx, neighbors in enumerate(view.neighbors):
        for root_idx in neighbors:
            if _component_has_charge_preserving_prefix_shape(
                view,
                root_idx,
                parent_idx,
                negative_oxygen_atoms,
                center_atoms,
            ):
                roots.add((parent_idx, root_idx))
    return mol_cache_set(mol, "acyclic_charge_preserving_prefix_candidate_roots", frozenset(roots))


def _component_has_charge_preserving_prefix_shape(
    view: _AcyclicGraphView,
    root_idx: int,
    parent_idx: int,
    negative_oxygen_atoms: set[int],
    center_atoms: set[int],
) -> bool:
    seen = {parent_idx}
    stack = [root_idx]
    has_negative_oxygen = False
    has_center = False
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        if idx in negative_oxygen_atoms:
            has_negative_oxygen = True
            if has_center:
                return True
        if idx in center_atoms:
            has_center = True
            if has_negative_oxygen:
                return True
        for nidx in view.neighbors[idx]:
            if nidx not in seen:
                stack.append(nidx)
    return False


def _acyclic_graph_view(mol) -> _AcyclicGraphView:
    cached = mol_cache_get(mol, "acyclic_graph_view")
    if cached is not None:
        return cached
    graph = graph_view(mol, Chem)
    view = _AcyclicGraphView(
        symbols=graph.symbols,
        isotopes=graph.isotopes,
        neighbors=graph.neighbors,
        bond_orders=graph.bond_orders,
    )
    return mol_cache_set(mol, "acyclic_graph_view", view)


def _record_acyclic_filter_stage(mol, stage: str, count: int) -> None:
    if count == 0:
        return
    telemetry = mol_cache_get(mol, "candidate_telemetry")
    if telemetry is None:
        return
    family = telemetry.active_rule_family or "acyclic"
    telemetry.filter_stage_count[f"{family}:acyclic_{stage}"] += count


def _bond_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _path_substituent_prefixes(
    mol,
    path: Sequence[int],
    *,
    skip_bonds: set[tuple[int, int]] | None = None,
    skip_atoms: set[int] | None = None,
    allowed_atoms: set[int] | None = None,
    skip_plain_hydrogen: bool = False,
) -> list[tuple[str, str]] | None:
    cache = mol_cache_dict(mol, "acyclic_path_substituent_prefixes")
    path_tuple = tuple(path)
    skip_bond_key = tuple(sorted(skip_bonds or ()))
    skip_atom_key = tuple(sorted(skip_atoms or ()))
    allowed_atom_key = tuple(sorted(allowed_atoms)) if allowed_atoms is not None else None
    allow_fragment_fallback = mol.HasProp("_nispo_allow_fragment_substituent_fallback")
    fragment_fallback_value = (
        mol.GetProp("_nispo_allow_fragment_substituent_fallback")
        if allow_fragment_fallback
        else ""
    )
    fragment_depth = mol.HasProp("_nispo_fragment_substituent_depth")
    fragment_depth_value = mol.GetProp("_nispo_fragment_substituent_depth") if fragment_depth else ""
    fragment_view = mol.HasProp("_nispo_fragment_substituent_view")
    cache_key = (
        path_tuple,
        skip_bond_key,
        skip_atom_key,
        allowed_atom_key,
        skip_plain_hydrogen,
        allow_fragment_fallback,
        fragment_fallback_value,
        fragment_depth,
        fragment_depth_value,
        fragment_view,
    )
    if cache_key in cache:
        cached = cache[cache_key]
        return None if cached is None else list(cached)
    reversed_path = tuple(reversed(path_tuple))
    if reversed_path != path_tuple:
        reverse_cache_key = (
            reversed_path,
            skip_bond_key,
            skip_atom_key,
            allowed_atom_key,
            skip_plain_hydrogen,
            allow_fragment_fallback,
            fragment_fallback_value,
            fragment_depth,
            fragment_depth_value,
            fragment_view,
        )
        if reverse_cache_key in cache:
            cached = cache[reverse_cache_key]
            if cached is None:
                cache[cache_key] = None
                return None
            mirrored = _mirror_path_substituent_prefixes(cached, len(path_tuple))
            cache[cache_key] = mirrored
            return list(mirrored)
    view = _acyclic_graph_view(mol)
    path_set = set(path_tuple)
    skip_bonds = skip_bonds or set()
    skip_atoms = skip_atoms or set()
    prefixes: list[tuple[str, str]] = []
    for pos, idx in enumerate(path_tuple, start=1):
        for nidx in view.neighbors[idx]:
            if nidx in path_set or nidx in skip_atoms:
                continue
            if allowed_atoms is not None and nidx not in allowed_atoms:
                continue
            if _bond_key(idx, nidx) in skip_bonds:
                continue
            if skip_plain_hydrogen and view.symbols[nidx] == "H" and view.isotopes[nidx] == 0:
                continue
            prefix = simple_external_prefix(mol, nidx, idx) or branch_prefix(
                mol,
                nidx,
                idx,
                assume_simple_external_prefix_miss=True,
            )
            if not prefix:
                cache[cache_key] = None
                return None
            prefixes.append((str(pos), prefix))
    cache[cache_key] = tuple(prefixes)
    return prefixes


def _mirror_path_substituent_prefixes(
    prefixes: Sequence[tuple[str, str]],
    path_len: int,
) -> tuple[tuple[str, str], ...]:
    by_reverse_locant: dict[int, list[str]] = {}
    for locant, prefix in prefixes:
        by_reverse_locant.setdefault(int(locant), []).append(prefix)
    mirrored: list[tuple[str, str]] = []
    for reverse_locant in range(path_len, 0, -1):
        current_locant = str(path_len + 1 - reverse_locant)
        for prefix in by_reverse_locant.get(reverse_locant, ()):
            mirrored.append((current_locant, prefix))
    return tuple(mirrored)


def _name_from_parent_path(mol, path: Sequence[int], as_substituent: bool) -> str:
    if not path:
        return ""
    prefixes = _path_substituent_prefixes(mol, path, skip_plain_hydrogen=True)
    if prefixes is None:
        return ""

    replacement_text = replacement_prefixes(mol, path)
    parent = parent_chain_name(mol, path, as_substituent=as_substituent)
    if not parent:
        return ""
    stereo_prefix = _complete_path_stereo_prefix(mol, path)
    if not as_substituent and not replacement_text:
        oxido_parent = _oxido_suffix_parent_name(mol, path, prefixes)
        if oxido_parent:
            return f"({stereo_prefix})-{oxido_parent}" if stereo_prefix else oxido_parent
        hydroxy_parent = _hydroxy_suffix_parent_name(mol, path, prefixes)
        if hydroxy_parent:
            return f"({stereo_prefix})-{hydroxy_parent}" if stereo_prefix else hydroxy_parent
        suffix_parent = _terminal_carboxy_suffix_parent_name(mol, path, prefixes)
        if suffix_parent:
            return suffix_parent
    prefix_text = _format_prefixes(prefixes)
    if not as_substituent:
        parent = _standalone_saturated_chain_parent(parent)
    if replacement_text:
        pieces = []
        if prefix_text:
            pieces.append(prefix_text)
        pieces.append(_join_replacement_parent(replacement_text, parent))
        name = "-".join(pieces)
    else:
        name = _join_prefix_parent(prefix_text, parent) if prefix_text else _preferred_parent_stem(parent)
    if stereo_prefix:
        name = f"({stereo_prefix})-{name}"
    if not as_substituent:
        name = _append_path_carbon_isotope_suffixes(
            mol,
            name,
            {atom_idx: str(locant) for locant, atom_idx in enumerate(path, start=1)},
            parent=parent,
        )
    return name


def _standalone_saturated_chain_parent(parent: str) -> str:
    if parent in _SATURATED_CHAIN_PARENT_STEMS:
        return parent + "e"
    return parent


def _oxido_suffix_parent_name(
    mol,
    path: Sequence[int],
    prefixes: Sequence[tuple[str, str]],
) -> str:
    oxido_locs = [loc for loc, prefix in prefixes if prefix == "oxido" and loc.isdigit()]
    if len(oxido_locs) != 1:
        return ""
    if any(mol.GetAtomWithIdx(idx).IsInRing() for idx in path):
        return ""
    remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "oxido"]
    if any(prefix in {"carboxy", "carboxylato", "carbamyl", "formyl", "oxo", "imino"} for _loc, prefix in remaining):
        return ""
    parent = parent_chain_name(mol, path, as_substituent=False)
    if not parent:
        return ""
    oxido_loc = oxido_locs[0]
    if parent.endswith(("en", "yn")) or "-" in parent:
        if oxido_loc != "1":
            return ""
    core = _mono_oxido_suffix_parent(parent, oxido_loc, bool(remaining))
    prefix_text = _format_prefixes(remaining)
    return _join_prefix_parent(prefix_text, core) if prefix_text else core


def _mono_oxido_suffix_parent(parent: str, locant: str, has_prefixes: bool) -> str:
    if locant == "1" and not has_prefixes and parent in {"methan", "ethan"}:
        return parent + "olate"
    return f"{parent}-{locant}-olate"


def _hydroxy_suffix_parent_name(
    mol,
    path: Sequence[int],
    prefixes: Sequence[tuple[str, str]],
) -> str:
    hydroxy_locs = [loc for loc, prefix in prefixes if prefix == "hydroxy" and loc.isdigit()]
    if not hydroxy_locs:
        return ""
    if len(hydroxy_locs) > 2:
        return ""
    if any(mol.GetAtomWithIdx(idx).IsInRing() for idx in path):
        return ""
    remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "hydroxy"]
    if any(prefix in {"carboxy", "carboxylato", "carbamyl", "formyl", "oxo", "imino"} for _loc, prefix in remaining):
        return ""
    parent = parent_chain_name(mol, path, as_substituent=False)
    if not parent:
        return ""
    hydroxy_locs = sorted(hydroxy_locs, key=lambda value: int(value))
    if (parent.endswith(("en", "yn")) or "-" in parent) and (hydroxy_locs != ["1"] or remaining):
        return ""
    if len(hydroxy_locs) == 1:
        core = _mono_hydroxy_suffix_parent(parent, hydroxy_locs[0])
    else:
        core = f"{parent}-{','.join(hydroxy_locs)}-diol"
    prefix_text = _format_prefixes(remaining)
    return _join_prefix_parent(prefix_text, core) if prefix_text else core


def _mono_hydroxy_suffix_parent(parent: str, locant: str) -> str:
    if locant == "1":
        return parent + "ol"
    return f"{locant}-{parent}ol"


def _join_replacement_parent(replacement_text: str, parent: str) -> str:
    if not replacement_text:
        return parent
    if parent and parent[0].isdigit():
        return f"{replacement_text}-{parent}"
    return replacement_text + parent


def _terminal_carboxy_suffix_parent_name(
    mol,
    path: Sequence[int],
    prefixes: Sequence[tuple[str, str]],
) -> str:
    n = len(path)
    root = ROOTS.get(n + 1)
    if n < 2 or not root:
        return ""
    terminal_groups = [
        (loc, prefix)
        for loc, prefix in prefixes
        if prefix in {"carboxy", "carboxylato", "carbamyl", "formyl"} and loc in {"1", str(n)}
    ]
    terminal = [
        item for item in terminal_groups if item[1] in {"carboxy", "carboxylato"}
    ] or [item for item in terminal_groups if item[1] == "carbamyl"] or [
        item for item in terminal_groups if item[1] == "formyl"
    ]
    diterminal = _diterminal_carboxy_suffix_parent_name(mol, path, prefixes, terminal)
    if diterminal:
        return diterminal
    if len(terminal) != 1:
        return ""
    terminal_loc, terminal_prefix = terminal[0]
    reverse = terminal_loc == str(n)
    remaining: list[tuple[str, str]] = []
    for loc, prefix in prefixes:
        if (loc, prefix) == (terminal_loc, terminal_prefix):
            continue
        if not loc.isdigit():
            return ""
        old_loc = int(loc)
        new_loc = n - old_loc + 2 if reverse else old_loc + 1
        remaining.append((str(new_loc), prefix))
    if terminal_prefix == "carbamyl":
        if any(prefix in {"carboxy", "carboxylato"} or "carbam" in prefix for _loc, prefix in remaining):
            return ""
    elif terminal_prefix == "formyl":
        if any(
            prefix in {"carboxy", "carboxylato", "carbamyl"} or "carbo" in prefix or "carbam" in prefix
            for _loc, prefix in remaining
        ):
            return ""
    elif any("carbo" in prefix or (prefix.startswith("carb") and prefix.endswith("oxy")) for _loc, prefix in remaining):
        return ""
    oriented_path = tuple(reversed(path)) if reverse else tuple(path)
    parent = _terminal_carboxy_parent_suffix_text(mol, root, oriented_path, terminal_prefix)
    if not parent:
        return ""
    prefix_text = _format_prefixes(remaining)
    name = _join_prefix_parent(prefix_text, parent) if prefix_text else parent
    stereo_prefix = _terminal_carboxy_stereo_prefix(mol, oriented_path)
    if stereo_prefix:
        name = f"({stereo_prefix})-{name}"
    locants_by_atom = {atom_idx: str(locant) for locant, atom_idx in enumerate(oriented_path, start=2)}
    terminal_root = _terminal_suffix_root_atom(mol, path, terminal_loc, terminal_prefix)
    if terminal_root is not None:
        locants_by_atom[terminal_root] = "1"
    return _append_path_carbon_isotope_suffixes(mol, name, locants_by_atom, parent=parent)


def _terminal_suffix_root_atom(
    mol,
    path: Sequence[int],
    terminal_loc: str,
    terminal_prefix: str,
) -> int | None:
    if not terminal_loc.isdigit():
        return None
    anchor_pos = int(terminal_loc) - 1
    if anchor_pos < 0 or anchor_pos >= len(path):
        return None
    anchor_idx = path[anchor_pos]
    path_set = set(path)
    for neighbor in mol.GetAtomWithIdx(anchor_idx).GetNeighbors():
        nidx = neighbor.GetIdx()
        if nidx in path_set or neighbor.GetAtomicNum() <= 1:
            continue
        if simple_external_prefix(mol, nidx, anchor_idx) == terminal_prefix:
            return nidx
    return None


def _append_path_carbon_isotope_suffixes(
    mol,
    name: str,
    locants_by_atom: dict[int, str],
    *,
    parent: str = "",
) -> str:
    if not name or any(f"{isotope}C" in name for isotope in (13, 14)):
        return name
    total_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    labeled: dict[int, list[str]] = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or not atom.GetIsotope():
            continue
        locant = locants_by_atom.get(atom.GetIdx())
        if not locant:
            return name
        labeled.setdefault(atom.GetIsotope(), []).append(locant)
    if not labeled:
        return name
    if len(labeled) == 1 and len(next(iter(labeled.values()))) == total_carbons:
        return name
    suffixes = []
    parent_prefixes = []
    for isotope, locants in sorted(labeled.items()):
        ordered = sorted(set(locants), key=lambda value: (int(value) if value.isdigit() else 9999, value))
        count_text = str(len(ordered)) if len(ordered) > 1 else ""
        if isotope in {13, 14}:
            suffixes.append(f"{','.join(ordered)}-{isotope}C{count_text}")
        elif parent:
            parent_prefixes.append(f"({','.join(ordered)}-{isotope}C{count_text})")
        else:
            return name
    if parent_prefixes:
        name = _insert_parent_isotope_prefixes(name, parent, parent_prefixes)
    return name + "-" + "-".join(suffixes) if suffixes else name


def _insert_parent_isotope_prefixes(name: str, parent: str, prefixes: Sequence[str]) -> str:
    if not parent or not prefixes or not name.endswith(parent):
        return name
    return name[: -len(parent)] + "".join(prefixes) + parent


def _diterminal_carboxy_suffix_parent_name(
    mol,
    path: Sequence[int],
    prefixes: Sequence[tuple[str, str]],
    terminal: Sequence[tuple[str, str]],
) -> str:
    n = len(path)
    root = ROOTS.get(n + 2)
    if n < 1 or not root or len(terminal) != 2:
        return ""
    terminal_by_loc = {loc: prefix for loc, prefix in terminal}
    if set(terminal_by_loc) != {"1", str(n)}:
        return ""
    terminal_types = set(terminal_by_loc.values())
    if len(terminal_types) != 1:
        return ""
    terminal_prefix = next(iter(terminal_types))
    if terminal_prefix not in {"carboxy", "carboxylato", "carbamyl"}:
        return ""
    remaining: list[tuple[str, str]] = []
    terminal_set = set(terminal)
    for loc, prefix in prefixes:
        if (loc, prefix) in terminal_set:
            continue
        if not loc.isdigit():
            return ""
        remaining.append((str(int(loc) + 1), prefix))
    if any("carbo" in prefix or (prefix.startswith("carb") and prefix.endswith("oxy")) for _loc, prefix in remaining):
        return ""
    parent = _diterminal_carboxy_parent_suffix_text(mol, root, path, terminal_prefix)
    if not parent:
        return ""
    prefix_text = _format_prefixes(remaining)
    name = _join_prefix_parent(prefix_text, parent) if prefix_text else parent
    stereo_prefix = _terminal_carboxy_stereo_prefix(mol, path)
    return f"({stereo_prefix})-{name}" if stereo_prefix else name


def _diterminal_carboxy_parent_suffix_text(
    mol,
    root: str,
    path: Sequence[int],
    terminal_prefix: str,
) -> str:
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
    if terminal_prefix == "carbamyl":
        return _diterminal_suffix(parent, "diamide")
    if terminal_prefix == "carboxylato":
        return _diterminal_suffix(parent, "dioate")
    return _diterminal_suffix(parent, "dioic acid")


def _diterminal_suffix(parent: str, suffix: str) -> str:
    if parent.endswith("ane"):
        return parent[:-1] + suffix
    if parent.endswith(("an", "en", "yn")):
        return parent + "e" + suffix
    if parent.endswith("e"):
        return parent[:-1] + suffix
    return parent + suffix


def _terminal_carboxy_parent_suffix_text(
    mol,
    root: str,
    oriented_path: Sequence[int],
    terminal_prefix: str,
) -> str:
    ene_locs: list[int] = []
    yne_locs: list[int] = []
    for locant, (a, b) in enumerate(zip(oriented_path, oriented_path[1:]), start=2):
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
    if terminal_prefix == "formyl":
        return parent[:-1] + "al" if parent.endswith("e") else parent + "al"
    if terminal_prefix == "carbamyl":
        return parent[:-1] + "amide" if parent.endswith("e") else parent + "amide"
    if terminal_prefix == "carboxylato":
        return parent[:-1] + "oate" if parent.endswith("e") else parent + "oate"
    return parent[:-1] + "oic acid" if parent.endswith("e") else parent + "oic acid"


def _terminal_carboxy_stereo_prefix(mol, oriented_path: Sequence[int]) -> str:
    locants = {idx: pos for pos, idx in enumerate(oriented_path, start=2)}
    assigned = assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
    if any(idx not in locants for idx, _descriptor in assigned):
        return ""
    atom_descriptors = [
        (locants[idx], f"{locants[idx]}{descriptor}")
        for idx, descriptor in assigned
    ]
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
            return ""
        begin_locant = locants[begin]
        end_locant = locants[end]
        if abs(begin_locant - end_locant) != 1:
            return ""
        locant = min(begin_locant, end_locant)
        bond_descriptors.append((locant, f"{locant}{descriptor}"))
    descriptors = atom_descriptors + bond_descriptors
    return ",".join(text for _locant, text in sorted(descriptors))


def _complete_path_stereo_prefix(mol, path: Sequence[int]) -> str:
    assigned = assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
    bond_descriptors = _path_bond_stereo_descriptors(mol, path)
    if bond_descriptors is None:
        return ""
    if not assigned and not bond_descriptors:
        return ""
    locants = {idx: pos for pos, idx in enumerate(path, start=1)}
    if any(idx not in locants for idx, _descriptor in assigned):
        return ""
    atom_descriptors = [
        (locants[idx], f"{locants[idx]}{descriptor}")
        for idx, descriptor in assigned
    ]
    descriptors = atom_descriptors + bond_descriptors
    return ",".join(text for _locant, text in sorted(descriptors))


def _path_bond_stereo_descriptors(mol, path: Sequence[int]) -> list[tuple[int, str]] | None:
    locants = {idx: pos for pos, idx in enumerate(path, start=1)}
    descriptors: list[tuple[int, str]] = []
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
        begin_locant = locants[begin]
        end_locant = locants[end]
        if abs(begin_locant - end_locant) != 1:
            return None
        locant = min(begin_locant, end_locant)
        descriptors.append((locant, f"{locant}{descriptor}"))
    return descriptors


def _name_thioamide(mol) -> str:
    candidates = []
    for atom in mol.GetAtoms():
        cidx = atom.GetIdx()
        if atom.GetSymbol() != "C":
            continue
        double_s = []
        single_n = []
        carbon_neighbors = []
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_s.append(nidx)
            elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
                single_n.append(nidx)
            elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                carbon_neighbors.append(nidx)
            else:
                carbon_neighbors = []
                break
        if len(double_s) != 1 or len(single_n) != 1 or len(carbon_neighbors) > 1:
            continue
        parent = _thioamide_parent_name(mol, cidx, double_s[0], single_n[0])
        if not parent:
            continue
        n_prefix = _thioamide_n_prefix(mol, single_n[0], cidx)
        if n_prefix is None:
            continue
        name = n_prefix + parent if n_prefix else parent
        candidates.append((len(name), name))
    return min(candidates)[1] if candidates else ""


def _name_thione(mol) -> str:
    motif = _single_exocyclic_sulfur_motif(mol)
    if motif is None:
        return ""
    thione_idx, sulfur_idx = motif
    paths = _thione_parent_paths(mol, thione_idx)
    orientations = _oriented_paths_by_quality(mol, paths)
    index = 0
    while index < len(orientations):
        descriptor = orientations[index][0]
        names = []
        while index < len(orientations) and orientations[index][0] == descriptor:
            _descriptor, oriented = orientations[index]
            if thione_idx in oriented:
                name = _name_thione_path(mol, oriented, thione_idx, sulfur_idx)
                if name:
                    names.append(name)
            index += 1
        if names:
            return min(names)
    return ""


def _name_ylidene_dioxosulfane(mol) -> str:
    motif = _single_exocyclic_sulfur_motif(mol, sulfur_predicate=_is_dioxosulfane_sulfur)
    if motif is None:
        return ""
    ylidene_idx, sulfur_idx = motif
    paths = _thione_parent_paths(mol, ylidene_idx)
    orientations = _oriented_paths_by_quality(mol, paths)
    index = 0
    while index < len(orientations):
        descriptor = orientations[index][0]
        names = []
        while index < len(orientations) and orientations[index][0] == descriptor:
            _descriptor, oriented = orientations[index]
            if ylidene_idx in oriented:
                name = _name_ylidene_dioxosulfane_path(mol, oriented, ylidene_idx, sulfur_idx)
                if name:
                    names.append(name)
            index += 1
        if names:
            return min(names)
    return ""


def _single_exocyclic_sulfur_motif(mol, sulfur_predicate=None) -> tuple[int, int] | None:
    motifs = []
    for atom in mol.GetAtoms():
        cidx = atom.GetIdx()
        if atom.GetSymbol() != "C":
            continue
        sulfur = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetSymbol() == "S"
            and mol.GetBondBetweenAtoms(cidx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            and (sulfur_predicate is None or sulfur_predicate(mol, nbr.GetIdx()))
        ]
        if sulfur:
            if len(sulfur) != 1:
                return None
            motifs.append((cidx, sulfur[0]))
    if len(motifs) != 1:
        return None
    return motifs[0]


def _is_dioxosulfane_sulfur(mol, sulfur_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(sulfur_idx)
    if atom.GetSymbol() != "S":
        return False
    double_oxygen_count = 0
    for nbr in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nbr.GetIdx())
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
    return double_oxygen_count == 2


def _thione_parent_paths(mol, thione_idx: int) -> list[tuple[int, ...]]:
    carbons = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "C" and not atom.IsInRing()}
    if thione_idx not in carbons:
        return []
    endpoints = [
        idx
        for idx in carbons
        if sum(1 for nbr in mol.GetAtomWithIdx(idx).GetNeighbors() if nbr.GetIdx() in carbons) <= 1
    ]
    if not endpoints:
        endpoints = list(carbons)
    paths = set()
    best_len = 0
    for a, b in combinations(endpoints, 2):
        path = Chem.rdmolops.GetShortestPath(mol, a, b)
        if thione_idx not in path or any(idx not in carbons for idx in path):
            continue
        if len(path) > best_len:
            best_len = len(path)
            paths.clear()
        if len(path) == best_len:
            paths.add(tuple(path))
    if not paths:
        paths.add((thione_idx,))
    return sorted(paths)


def _name_ylidene_dioxosulfane_path(
    mol,
    path: Sequence[int],
    ylidene_idx: int,
    sulfur_idx: int,
) -> str:
    return _name_exocyclic_sulfur_suffix_path(
        mol,
        path,
        center_idx=ylidene_idx,
        sulfur_idx=sulfur_idx,
        suffix="ylidene-dioxosulfane",
    )


def _name_thione_path(mol, path: Sequence[int], thione_idx: int, sulfur_idx: int) -> str:
    suffix = (
        "thione S-oxide"
        if _is_thione_s_oxide_sulfur(mol, sulfur_idx, thione_idx)
        else "thione"
    )
    return _name_exocyclic_sulfur_suffix_path(
        mol,
        path,
        center_idx=thione_idx,
        sulfur_idx=sulfur_idx,
        suffix=suffix,
    )


def _is_thione_s_oxide_sulfur(mol, sulfur_idx: int, thione_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(sulfur_idx)
    if atom.GetSymbol() != "S":
        return False
    double_oxygen_count = 0
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if nidx == thione_idx:
            if bond.GetBondType() != Chem.BondType.DOUBLE:
                return False
            continue
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
            continue
        if nbr.GetAtomicNum() > 1:
            return False
    return double_oxygen_count == 1


def _name_exocyclic_sulfur_suffix_path(
    mol,
    path: Sequence[int],
    *,
    center_idx: int,
    sulfur_idx: int,
    suffix: str,
) -> str:
    if any(mol.GetAtomWithIdx(idx).GetSymbol() != "C" for idx in path):
        return ""
    prefixes = _path_substituent_prefixes(
        mol,
        path,
        skip_bonds={_bond_key(center_idx, sulfur_idx)},
    )
    if prefixes is None:
        return ""
    parent = _standalone_saturated_chain_parent(
        parent_chain_name(mol, path, as_substituent=False)
    )
    if not parent or not parent.endswith("ane"):
        return ""
    locant = path.index(center_idx) + 1
    parent = parent[:-1] + f"-{locant}-{suffix}"
    prefix_text = _format_prefixes(prefixes)
    return _join_prefix_parent(prefix_text, parent) if prefix_text else parent


def _thioamide_parent_name(mol, carbonyl_idx: int, sulfur_idx: int, nitrogen_idx: int) -> str:
    atoms = collect_subtree(mol, carbonyl_idx, nitrogen_idx)
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
        prefixes = _path_substituent_prefixes(
            mol,
            path,
            skip_bonds={_bond_key(carbonyl_idx, sulfur_idx)},
            skip_atoms={nitrogen_idx},
            allowed_atoms=atoms,
        )
        if prefixes is None:
            continue
        prefix_text = _format_prefixes(prefixes)
        name = root + "anethioamide"
        if prefix_text:
            name = prefix_text + name
        key = (-len(path), substituent_locants(mol, path, atoms, nitrogen_idx), name)
        if best_key is None or key < best_key:
            best_key = key
            best = name
    return best


def _thioamide_n_prefix(mol, nitrogen_idx: int, carbonyl_idx: int) -> str | None:
    groups = []
    for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbonyl_idx:
            continue
        name = substituent_from_root(mol, nidx, nitrogen_idx)
        if not name:
            return None
        groups.append(name)
    if not groups:
        return ""
    counts = {}
    for name in groups:
        counts[name] = counts.get(name, 0) + 1
    parts = []
    for name in sorted(counts):
        count = counts[name]
        locants = ",".join("N" for _ in range(count))
        if count == 1:
            parts.append(f"{locants}-{name}")
        else:
            parts.append(f"{locants}-{MULT.get(count, '')}{name}")
    return "-".join(parts) + "-"
