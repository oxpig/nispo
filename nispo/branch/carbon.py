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
    _is_terminal_carbonyl_oxygen,
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

    if _is_nitrile_carbon(mol, idx):
        return _with_root_carbon_isotope(mol, idx, suffix_prefix_term("nitrile", "cyano"))
    deuterated_methyl = _deuterated_methyl_prefix(mol, idx, parent_idx)
    if deuterated_methyl:
        return deuterated_methyl

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
            name = _dicyanoethenyl_prefix(mol, idx, parent_idx)
            if name:
                return _with_root_carbon_isotope(mol, idx, name)
        if _has_neighbor_bond(mol, idx, neighbors, "O", Chem.BondType.SINGLE):
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


def _prefer_resource_carbon_prefix(name: str) -> bool:
    return _prefer_resource_prefix(name)


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
    if not heavy:
        return "boranyl"
    if len(heavy) == 2 and all(_is_terminal_hydroxy_oxygen(mol, nidx, idx) for nidx in heavy):
        return "borono"
    return ""


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
            return "oxomethylidene"
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
