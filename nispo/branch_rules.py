from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from functools import lru_cache

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from .constants import HALO, MULT, REPLACEMENT, ROOTS, SKELETON_ATOMS
from .branch_engine import BranchRule, BranchRuleSet, first_branch_name
from .cycle_utils import _ordered_ring, _ring_system_extends_core
from .fused_rules import FUSED_SUBSTITUENT_STEMS
from .fused_template_families import (
    candidate_generated_fused_templates as _candidate_generated_fused_templates,
    curated_fused_templates as _curated_fused_templates,
    generated_fusion_preference as _generated_fusion_preference,
)
from .formatting import (
    alkyl_to_alkoxy as _alkyl_to_alkoxy,
    format_prefixes as _format_prefixes,
    join_locants as _join_locants,
    substituent_parent_suffix as _substituent_parent_suffix,
    unsaturated_parent as _unsaturated_parent,
)
from .ring_rules import (
    ring_parent_candidates as _ring_parent_candidates,
    ring_substituent_suffix as _ring_substituent_suffix,
)
from .ring_resource_rules import resource_ring_parent_candidates as _resource_ring_parent_candidates
from .substituent_resource_rules import resource_substituent_prefix as _resource_substituent_prefix
from .template_index import candidate_templates as _candidate_templates


def _can_be_in_parent_skeleton(mol, idx: int) -> bool:
    cache = getattr(mol, "_nispo_parent_skeleton_atom_cache", None)
    if cache is None:
        cache = {}
        try:
            mol._nispo_parent_skeleton_atom_cache = cache
        except Exception:
            cache = None
    if cache is not None:
        cached = cache.get(idx)
        if cached is not None:
            return cached
    atom = mol.GetAtomWithIdx(idx)
    symbol = atom.GetSymbol()
    result = True
    if symbol not in SKELETON_ATOMS:
        result = False
    elif atom.GetFormalCharge() != 0 and not (symbol == "N" and atom.GetFormalCharge() > 0):
        result = False
    elif symbol == "O" and _is_terminal_carbonyl_oxygen(atom):
        result = False
    elif symbol == "C" and _is_nitrile_carbon(mol, idx):
        result = False
    elif symbol == "C" and _is_terminal_ketene_alpha_carbon(mol, idx):
        result = False
    elif symbol == "N" and _is_nitrile_nitrogen(mol, idx):
        result = False
    elif symbol == "N" and _is_nitro_nitrogen(mol, idx):
        result = False
    elif symbol == "N" and _is_diazo_nitrogen(mol, idx):
        result = False
    if cache is not None:
        cache[idx] = result
    return result


def _replacement_prefixes(mol, path: Sequence[int]) -> str:
    grouped: dict[str, list[int]] = defaultdict(list)
    for pos, idx in enumerate(path, start=1):
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol == "N" and atom.GetFormalCharge() > 0:
            grouped["azonia"].append(pos)
        elif symbol in REPLACEMENT:
            grouped[REPLACEMENT[symbol]].append(pos)
    parts = []
    for term in sorted(grouped):
        locants = grouped[term]
        if len(locants) == 1:
            parts.append(f"{locants[0]}-{term}")
        else:
            parts.append(f"{_join_locants(locants)}-{MULT.get(len(locants), '')}{term}")
    return "-".join(parts)


def _parent_chain_name(mol, path: Sequence[int], as_substituent: bool = False) -> str:
    n = len(path)
    root = ROOTS.get(n)
    if root is None:
        return ""
    ene_locs = []
    yne_locs = []
    for i, (a, b) in enumerate(zip(path, path[1:]), start=1):
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            ene_locs.append(i)
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            yne_locs.append(i)
        elif bond.GetBondType() != Chem.BondType.SINGLE:
            return ""

    base = _unsaturated_parent(root, ene_locs, yne_locs)
    if as_substituent:
        return _substituent_parent_suffix(base)
    return base


def _branch_prefix(mol, idx: int, parent_idx: int) -> str:
    cache = getattr(mol, "_nispo_branch_prefix_cache", None)
    if cache is None:
        cache = {}
        mol._nispo_branch_prefix_cache = cache
    key = (idx, parent_idx)
    cached = cache.get(key)
    if cached is not None:
        return cached
    in_progress = getattr(mol, "_nispo_branch_prefix_in_progress", None)
    if in_progress is None:
        in_progress = set()
        mol._nispo_branch_prefix_in_progress = in_progress
    if key in in_progress:
        return ""

    in_progress.add(key)
    try:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "H":
            prefix = _isotopic_hydrogen_prefix(atom)
        elif atom.IsInRing():
            prefix = _ring_substituent_prefix(mol, idx, parent_idx)
        elif atom.GetSymbol() == "C":
            prefix = _carbon_branch_prefix(mol, idx, parent_idx)
        else:
            rule_sets_by_symbol = _branch_rule_sets()
            rule_sets = rule_sets_by_symbol.get(atom.GetSymbol(), ()) + rule_sets_by_symbol.get("*", ())
            prefix = first_branch_name(rule_sets, mol, idx, parent_idx)
        cache[key] = prefix
        return prefix
    finally:
        in_progress.discard(key)


def _isotopic_hydrogen_prefix(atom) -> str:
    if atom.GetIsotope() == 2:
        return "deuterio"
    if atom.GetIsotope() == 3:
        return "tritio"
    return ""


def _constant_prefix(name: str):
    return lambda _mol, _idx, _parent_idx: name


def _parenthesize_complex_prefix(name: str) -> str:
    if "-" in name or name[:1].isdigit() or name.startswith(("N-", "S-", "O-")):
        return f"({name})"
    return name


def _resource_prefix(mol, idx: int, parent_idx: int) -> str:
    return _resource_substituent_prefix(mol, idx, parent_idx, Chem)


def _carbon_prefix(mol, idx: int, parent_idx: int) -> str:
    if _is_carbonyl_oxygen_idx(mol, idx):
        return ""
    return _substituent_from_root(mol, idx, parent_idx)


def _carbon_branch_prefix(mol, idx: int, parent_idx: int) -> str:
    """Dispatch carbon substituent rules from local structure before rendering.

    Carbon branches are the dominant branch-rendering hot path. The previous
    BranchRuleSet tried every carbon functional family in order for almost every
    carbon substituent. This keeps that order, but uses cheap local bond classes
    to select the only families that can possibly match.
    """

    if _nitrile_branch_predicate(mol, idx, parent_idx):
        return "cyano"
    deuterated_methyl = _deuterated_methyl_prefix(mol, idx, parent_idx)
    if deuterated_methyl:
        return deuterated_methyl

    atom = mol.GetAtomWithIdx(idx)
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    parent_order = parent_bond.GetBondType() if parent_bond is not None else None
    neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]

    if parent_order == Chem.BondType.SINGLE:
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
                    return name
        if _is_isocyano_carbon(mol, idx):
            return "isocyano"
        if _is_carboxylate_carbon(mol, idx):
            return "carboxylato"
        if _has_neighbor_bond(mol, idx, neighbors, "O", Chem.BondType.DOUBLE):
            name = _acyl_halide_rule(mol, idx, parent_idx)
            if name:
                return name
            name = _carbonyl_substituent_prefix(mol, idx, parent_idx)
            if name:
                return name
        if _has_neighbor_bond(mol, idx, neighbors, "N", Chem.BondType.DOUBLE):
            name = _iminomethyl_prefix(mol, idx, parent_idx)
            if name:
                return name
        if _has_neighbor_bond(mol, idx, neighbors, "N", Chem.BondType.SINGLE):
            name = _sulfonamidomethyl_prefix(mol, idx, parent_idx)
            if name:
                return name
        if _has_neighbor_bond(mol, idx, neighbors, "C", Chem.BondType.TRIPLE):
            name = _ring_ethynyl_prefix(mol, idx, parent_idx)
            if name:
                return name
            name = _alkynyl_substituent_prefix(mol, idx, parent_idx)
            if name:
                return name
        if _has_neighbor_bond(mol, idx, neighbors, "C", Chem.BondType.DOUBLE):
            name = _dicyanoethenyl_prefix(mol, idx, parent_idx)
            if name:
                return name
        if _has_neighbor_bond(mol, idx, neighbors, "O", Chem.BondType.SINGLE):
            name = _aryloxy_methyl_prefix(mol, idx, parent_idx)
            if name:
                return name
        name = _difluoro_substituted_methyl_prefix(mol, idx, parent_idx)
        if name:
            return name
        if mol.GetRingInfo().NumRings() and _subtree_has_ring(mol, idx, parent_idx):
            name = _ring_alkyl_prefix(mol, idx, parent_idx)
            if name:
                return name
    elif parent_order == Chem.BondType.DOUBLE:
        name = _alkylidene_prefix(mol, idx, parent_idx)
        if name:
            return name
    else:
        if _is_isocyano_carbon(mol, idx):
            return "isocyano"
        if _is_carboxylate_carbon(mol, idx):
            return "carboxylato"

    name = _carbon_prefix(mol, idx, parent_idx)
    if name:
        return name
    return _resource_prefix(mol, idx, parent_idx)


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
    cache = getattr(mol, "_nispo_subtree_has_ring_cache", None)
    if cache is None:
        cache = {}
        try:
            mol._nispo_subtree_has_ring_cache = cache
        except Exception:
            cache = None
    key = (root_idx, blocked_idx)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    result = any(mol.GetAtomWithIdx(atom_idx).IsInRing() for atom_idx in _collect_subtree(mol, root_idx, blocked_idx))
    if cache is not None:
        cache[key] = result
    return result


def _acyl_halide_rule(mol, idx: int, _parent_idx: int) -> str:
    return _acyl_halide_prefix(mol, idx)


@lru_cache(maxsize=1)
def _branch_rule_sets() -> dict[str, tuple[BranchRuleSet, ...]]:
    return {
        "I": (
            BranchRuleSet(
                "iodine-ligand",
                (
                    BranchRule("iodine-ligand", "iodanyl-acyloxy", _iodanyl_acyloxy_prefix),
                    BranchRule("halogen", "iodo", _constant_prefix(HALO["I"])),
                ),
            ),
        ),
        "F": (_halogen_rule_set("F"),),
        "Cl": (_halogen_rule_set("Cl"),),
        "Br": (_halogen_rule_set("Br"),),
        "O": (
            BranchRuleSet(
                "oxygen-prefix",
                (
                    BranchRule("oxygen-prefix", "hydroxy-oxo-alkoxy", _oxygen_prefix),
                    BranchRule("opsin-resource-substituent", "oxygen-resource", _resource_prefix),
                ),
            ),
        ),
        "N": (
            BranchRuleSet(
                "nitrogen-prefix",
                (
                    BranchRule("nitrogen-prefix", "nitro", _constant_prefix("nitro"), _nitro_branch_predicate),
                    BranchRule("nitrogen-prefix", "amino-azido-imino", _nitrogen_prefix),
                    BranchRule("opsin-resource-substituent", "nitrogen-resource", _resource_prefix),
                ),
            ),
        ),
        "S": (
            BranchRuleSet(
                "sulfur-prefix",
                (
                    BranchRule("sulfur-prefix", "sulfanyl-sulfinyl-sulfonyl", _sulfur_prefix),
                    BranchRule("opsin-resource-substituent", "sulfur-resource", _resource_prefix),
                ),
            ),
        ),
        "C": (
            BranchRuleSet(
                "carbon-functional-prefix",
                (
                    BranchRule("carbon-functional-prefix", "cyano", _constant_prefix("cyano"), _nitrile_branch_predicate),
                    BranchRule("carbon-functional-prefix", "ketenyl", _ketenyl_prefix),
                    BranchRule("carbon-functional-prefix", "enone", _enone_substituent_prefix),
                    BranchRule("carbon-functional-prefix", "enal", _enal_substituent_prefix),
                    BranchRule("carbon-functional-prefix", "haloethenyl", _haloethenyl_prefix),
                    BranchRule("carbon-functional-prefix", "alkylidene", _alkylidene_prefix),
                    BranchRule("carbon-functional-prefix", "isocyano", _constant_prefix("isocyano"), _isocyano_branch_predicate),
                    BranchRule("carbon-functional-prefix", "carboxylato", _constant_prefix("carboxylato"), _carboxylato_branch_predicate),
                    BranchRule("carbon-functional-prefix", "acyl-halide", _acyl_halide_rule),
                    BranchRule("carbon-functional-prefix", "carbonyl", _carbonyl_substituent_prefix),
                    BranchRule("carbon-functional-prefix", "iminomethyl", _iminomethyl_prefix),
                    BranchRule("carbon-functional-prefix", "ring-ethynyl", _ring_ethynyl_prefix),
                    BranchRule("carbon-functional-prefix", "alkynyl", _alkynyl_substituent_prefix),
                    BranchRule("carbon-chain-prefix", "aryloxymethyl", _aryloxy_methyl_prefix),
                    BranchRule("carbon-chain-prefix", "ring-alkyl", _ring_alkyl_prefix),
                    BranchRule("carbon-chain-prefix", "simple-carbon", _carbon_prefix),
                    BranchRule("opsin-resource-substituent", "carbon-resource", _resource_prefix),
                ),
            ),
        ),
        "*": (
            BranchRuleSet(
                "opsin-resource-substituent",
                (BranchRule("opsin-resource-substituent", "generic-resource", _resource_prefix),),
            ),
        ),
    }


def _halogen_rule_set(symbol: str) -> BranchRuleSet:
    return BranchRuleSet("halogen", (BranchRule("halogen", HALO[symbol], _constant_prefix(HALO[symbol])),))


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


def _nitrile_branch_predicate(mol, idx: int, _parent_idx: int) -> bool:
    return _is_nitrile_carbon(mol, idx)


def _isocyano_branch_predicate(mol, idx: int, _parent_idx: int) -> bool:
    return _is_isocyano_carbon(mol, idx)


def _carboxylato_branch_predicate(mol, idx: int, _parent_idx: int) -> bool:
    return _is_carboxylate_carbon(mol, idx)


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
            name = _substituent_from_root(mol, nidx, idx)
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
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE and _is_carbonyl_carbon(mol, nidx):
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
        if bond.GetBondType() == Chem.BondType.SINGLE and _is_carbonyl_carbon(mol, nidx):
            carbonyls.append(nidx)
    if len(carbonyls) != 1:
        return ""
    acyl = _acyl_parent_name(mol, carbonyls[0], oxygen_idx)
    return acyl + "oxy" if acyl else ""


def _acyl_parent_name(mol, carbonyl_idx: int, blocked_idx: int) -> str:
    atoms = _collect_subtree(mol, carbonyl_idx, blocked_idx)
    paths = _rooted_paths(mol, carbonyl_idx, atoms)
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
                prefix = _branch_prefix(mol, nidx, atom_idx)
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
        key = (-len(path), _substituent_locants(mol, path, atoms, blocked_idx), name)
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


def _ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    return _simple_ring_substituent_prefix(mol, attachment_idx, parent_idx) or _fused_ring_substituent_prefix(
        mol, attachment_idx, parent_idx
    ) or _generic_polycyclic_ring_substituent_prefix(
        mol, attachment_idx, parent_idx
    ) or _resource_substituent_prefix(mol, attachment_idx, parent_idx, Chem)


def _generic_polycyclic_ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    from .polycyclic_rules import polycyclic_substituent_prefix

    return polycyclic_substituent_prefix(mol, Chem, attachment_idx, parent_idx)


def _ring_alkyl_prefix(mol, idx: int, parent_idx: int) -> str:
    path = []
    previous = parent_idx
    current = idx
    ring_neighbor = None
    while len(path) < 8:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
            return ""
        bond = mol.GetBondBetweenAtoms(previous, current)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        path.append(current)
        ring_neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != previous and n.IsInRing()]
        carbon_neighbors = [
            n.GetIdx()
            for n in atom.GetNeighbors()
            if n.GetIdx() != previous and not n.IsInRing() and n.GetSymbol() == "C"
        ]
        other_neighbors = [
            n.GetIdx()
            for n in atom.GetNeighbors()
            if n.GetIdx() != previous and not n.IsInRing() and n.GetSymbol() != "C"
        ]
        if other_neighbors:
            return ""
        if ring_neighbors:
            if len(ring_neighbors) != 1 or carbon_neighbors:
                return ""
            ring_neighbor = ring_neighbors[0]
            break
        if len(carbon_neighbors) != 1:
            return ""
        previous, current = current, carbon_neighbors[0]
    if ring_neighbor is None:
        return ""
    ring = _ring_substituent_prefix(mol, ring_neighbor, path[-1])
    if not ring:
        return ""
    if len(path) != 1:
        root = ROOTS.get(len(path))
        if not root:
            return ""
        return f"{len(path)}-({ring}){root}yl"
    return ring + "methyl"


def _aryloxy_methyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    oxy = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            oxy.append(nidx)
        else:
            return ""
    if len(oxy) != 1:
        return ""
    oidx = oxy[0]
    ring_neighbors = [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(oidx).GetNeighbors()
        if n.GetIdx() != idx and n.IsInRing()
    ]
    if len(ring_neighbors) != 1:
        return ""
    ring = _ring_substituent_prefix(mol, ring_neighbors[0], oidx)
    if not ring:
        return ""
    if ring == "phenyl":
        return "phenoxymethyl"
    return f"({ring}oxy)methyl"


def _iminomethyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    double_n = []
    other = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_n.append(nidx)
        else:
            other.append(nidx)
    if len(double_n) != 1 or other:
        return ""
    nidx = double_n[0]
    distal = [n.GetIdx() for n in mol.GetAtomWithIdx(nidx).GetNeighbors() if n.GetIdx() != idx]
    if len(distal) != 1:
        return ""
    didx = distal[0]
    if mol.GetAtomWithIdx(didx).IsInRing():
        name = _ring_substituent_prefix(mol, didx, nidx)
    else:
        name = _substituent_from_root(mol, didx, nidx)
    if not name:
        return ""
    return name + "iminomethyl"


def _difluoro_substituted_methyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    fluorines = []
    heavy = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if nbr.GetSymbol() == "F":
            fluorines.append(nidx)
        else:
            heavy.append(nidx)
    if len(fluorines) != 2 or len(heavy) != 1:
        return ""
    attached = heavy[0]
    if mol.GetAtomWithIdx(attached).IsInRing():
        name = _ring_substituent_prefix(mol, attached, idx)
    else:
        name = _substituent_from_root(mol, attached, idx)
    if not name:
        return ""
    return f"({name})difluoromethyl"


def _sulfonamidomethyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    nitrogens = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
            nitrogens.append(nidx)
        else:
            return ""
    if len(nitrogens) != 1:
        return ""
    nidx = nitrogens[0]
    n_atom = mol.GetAtomWithIdx(nidx)
    sulfur = []
    for nbr in n_atom.GetNeighbors():
        sidx = nbr.GetIdx()
        if sidx == idx:
            continue
        bond = mol.GetBondBetweenAtoms(nidx, sidx)
        if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.SINGLE:
            sulfur.append(sidx)
        else:
            return ""
    if len(sulfur) != 1:
        return ""
    sulfonyl = _sulfur_prefix(mol, sulfur[0], nidx)
    if not sulfonyl or not sulfonyl.endswith("sulfonyl"):
        return ""
    return f"({sulfonyl}amino)methyl"


def _ring_ethynyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    triple_c = []
    for nbr in atom.GetNeighbors():
        if nbr.GetIdx() == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.TRIPLE:
            triple_c.append(nbr.GetIdx())
        else:
            return ""
    if len(triple_c) != 1:
        return ""
    distal = triple_c[0]
    ring_neighbors = [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(distal).GetNeighbors()
        if n.GetIdx() != idx and n.IsInRing()
    ]
    if len(ring_neighbors) != 1:
        return ""
    ring = _ring_substituent_prefix(mol, ring_neighbors[0], distal)
    if not ring:
        return ""
    return ring + "ethynyl"


def _dicyanoethenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    alkene = []
    other = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "C" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
            alkene.append(nidx)
        else:
            other.append(nidx)
    if len(alkene) != 1 or other:
        return ""
    distal = alkene[0]
    cyano = []
    for nbr in mol.GetAtomWithIdx(distal).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx:
            continue
        if _is_nitrile_carbon(mol, nidx):
            cyano.append(nidx)
        else:
            return ""
    if len(cyano) != 2:
        return ""
    return "2,2-dicyanoethenyl"


def _alkynyl_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    triple_neighbors = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
    ]
    if len(triple_neighbors) != 1:
        return ""
    paths = _linear_alkynyl_paths(mol, idx, parent_idx, triple_neighbors[0], max_len=8)
    if not paths:
        return ""
    best = ""
    best_key = None
    for path in paths:
        if not _simple_alkynyl_side_branches(mol, path, parent_idx):
            continue
        name = _name_subtree_path(mol, path, set(_collect_subtree(mol, idx, parent_idx)), parent_idx)
        if not name:
            continue
        key = (-len(path), _substituent_locants(mol, path, set(path), parent_idx), name)
        if best_key is None or key < best_key:
            best_key = key
            best = name
    return best


def _simple_alkynyl_side_branches(mol, path: Sequence[int], parent_idx: int) -> bool:
    path_set = set(path)
    for idx in path:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == parent_idx or nidx in path_set:
                continue
            if nbr.IsInRing():
                return False
            if nbr.GetSymbol() == "C" and not _is_nitrile_carbon(mol, nidx):
                return False
    return True


def _linear_alkynyl_paths(
    mol, root_idx: int, parent_idx: int, first_idx: int, max_len: int
) -> list[tuple[int, ...]]:
    paths: list[tuple[int, ...]] = []
    stack = [(first_idx, root_idx, (root_idx, first_idx))]
    while stack:
        current, previous, path = stack.pop()
        carbon_next = []
        ok = True
        for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == previous:
                continue
            bond = mol.GetBondBetweenAtoms(current, nidx)
            if nbr.GetSymbol() == "C" and not nbr.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE:
                carbon_next.append(nidx)
            elif nidx != parent_idx:
                continue
        if len(carbon_next) > 1:
            ok = False
        if ok:
            paths.append(path)
        if ok and len(path) < max_len:
            for nidx in carbon_next:
                if nidx not in path:
                    stack.append((nidx, current, path + (nidx,)))
    return paths


def _simple_ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    rings = [tuple(r) for r in mol.GetRingInfo().AtomRings() if attachment_idx in r and parent_idx not in r]
    if len(rings) != 1:
        return ""
    ring = rings[0]
    ordered = _ordered_ring(mol, ring)
    if not ordered:
        return ""
    candidates = _ring_parent_candidates(mol, ordered)
    if not candidates:
        candidates = _resource_ring_parent_candidates(mol, ordered, Chem)
    if not candidates:
        return ""
    ring_set = set(ordered)
    raw_subs = []
    for ring_idx in ordered:
        atom = mol.GetAtomWithIdx(ring_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == parent_idx or nidx in ring_set:
                continue
            prefix = _branch_prefix(mol, nidx, ring_idx)
            if not prefix:
                return ""
            raw_subs.append((ring_idx, prefix))

    best = ""
    best_key = None
    for parent_name, seq in candidates:
        if attachment_idx not in seq:
            continue
        loc = seq.index(attachment_idx) + 1
        charge_loc = _charged_ring_nitrogen_locant(mol, seq)
        suffix = _ring_substituent_suffix(parent_name, loc, charge_loc)
        if not suffix:
            continue
        suffix = _ring_substituent_unsaturated_suffix(mol, attachment_idx, parent_idx, suffix)
        if not suffix:
            continue
        pos = {idx: i + 1 for i, idx in enumerate(seq)}
        prefixes = [
            (str(pos[idx]), prefix)
            for idx, prefix in raw_subs
            if not (
                parent_name in {"1,3,4-dioxazol-2-one", "azet-2-one"} and pos[idx] == 2 and prefix == "oxo"
            )
        ]
        prefix_text = _format_prefixes(prefixes)
        if prefix_text:
            separator = "-" if suffix[:1].isdigit() else ""
            name = prefix_text + separator + suffix
        else:
            name = suffix
        locants = sorted(pos[idx] for idx, _ in raw_subs)
        key = (0 if loc == 1 else 1, locants, name)
        if best_key is None or key < best_key:
            best_key = key
            best = name
    return best


def _ring_substituent_unsaturated_suffix(
    mol, attachment_idx: int, parent_idx: int, suffix: str
) -> str:
    bond = mol.GetBondBetweenAtoms(attachment_idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return suffix
    atom = mol.GetAtomWithIdx(attachment_idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    if suffix.endswith("yl"):
        return suffix + "idene"
    return ""


def _charged_ring_nitrogen_locant(mol, seq: Sequence[int]) -> int | None:
    charged = [
        loc
        for loc, idx in enumerate(seq, start=1)
        if mol.GetAtomWithIdx(idx).GetSymbol() in {"N", "S"} and mol.GetAtomWithIdx(idx).GetFormalCharge() > 0
    ]
    return charged[0] if len(charged) == 1 else None


def _fused_ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    curated = _fused_ring_substituent_prefix_from_templates(
        mol, attachment_idx, parent_idx, _curated_fused_templates(Chem), generated=False
    )
    if curated:
        return curated
    generated_templates = _candidate_generated_fused_templates(mol, Chem)
    generated = _fused_ring_substituent_prefix_from_templates(
        mol, attachment_idx, parent_idx, generated_templates, generated=True
    ) or _hydro_fused_ring_substituent_prefix_from_templates(mol, attachment_idx, parent_idx, generated_templates)
    return generated


def _fused_ring_substituent_prefix_from_templates(
    mol, attachment_idx: int, parent_idx: int, templates: tuple[dict, ...], generated: bool
) -> str:
    best = ""
    best_key = None
    for template in _candidate_templates(mol, templates, Chem):
        core = template["mol"]
        for match in mol.GetSubstructMatches(core, uniquify=False):
            if attachment_idx not in match or parent_idx in match:
                continue
            core_set = set(match)
            if _ring_system_extends_core(mol, core_set):
                continue
            reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
            template_idx = reverse[attachment_idx]
            locant = template["locants"].get(template_idx)
            if not locant or locant.endswith("a"):
                continue
            suffix = _fused_substituent_suffix(template["name"], locant)
            if not suffix:
                continue
            prefixes = []
            ok = True
            for mol_idx in match:
                core_loc = template["locants"].get(reverse[mol_idx])
                for nbr in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
                    nidx = nbr.GetIdx()
                    if nidx == parent_idx or nidx in core_set:
                        continue
                    if not core_loc or core_loc.endswith("a"):
                        ok = False
                        break
                    prefix = _branch_prefix(mol, nidx, mol_idx)
                    if not prefix:
                        ok = False
                        break
                    prefixes.append((core_loc, prefix))
                if not ok:
                    break
            if not ok:
                continue
            prefix_text = _format_prefixes(prefixes)
            name = prefix_text + suffix if prefix_text else suffix
            key = _fused_substituent_candidate_key(template, match, locant, name, generated)
            if best_key is None or (key < best_key if generated else key > best_key):
                best_key = key
                best = name
    return best


def _fused_substituent_candidate_key(
    template: dict, match: Sequence[int], locant: str, name: str, generated: bool
) -> tuple:
    if generated:
        return (_generated_fusion_preference(template["name"]), -len(match), locant, name)
    return (len(match), locant, name)


def _hydro_fused_ring_substituent_prefix_from_templates(
    mol, attachment_idx: int, parent_idx: int, templates: tuple[dict, ...]
) -> str:
    best = ""
    best_key = None
    for template in _candidate_templates(mol, templates, Chem):
        if not _template_is_fully_aromatic(template["mol"]):
            continue
        query = _topology_query(template["mol"])
        for match in mol.GetSubstructMatches(query, uniquify=False):
            if attachment_idx not in match or parent_idx in match:
                continue
            core_set = set(match)
            if _ring_system_extends_core(mol, core_set):
                continue
            parent_name = _hydro_fused_parent_name(mol, template, match)
            if not parent_name:
                continue
            reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
            template_idx = reverse[attachment_idx]
            locant = template["locants"].get(template_idx)
            if not locant or locant.endswith("a"):
                continue
            suffix = _fused_substituent_suffix(parent_name, locant)
            prefixes = []
            ok = True
            for mol_idx in match:
                core_loc = template["locants"].get(reverse[mol_idx])
                for nbr in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
                    nidx = nbr.GetIdx()
                    if nidx == parent_idx or nidx in core_set:
                        continue
                    if not core_loc or core_loc.endswith("a"):
                        ok = False
                        break
                    prefix = _branch_prefix(mol, nidx, mol_idx)
                    if not prefix:
                        ok = False
                        break
                    prefixes.append((core_loc, prefix))
                if not ok:
                    break
            if not ok:
                continue
            prefix_text = _format_prefixes(prefixes)
            name = prefix_text + suffix if prefix_text else suffix
            key = (_generated_fusion_preference(template["name"]), -len(match), locant, name)
            if best_key is None or key < best_key:
                best_key = key
                best = name
    return best


@lru_cache(maxsize=None)
def _topology_query(template_mol):
    query = Chem.RWMol()
    for atom in template_mol.GetAtoms():
        query.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    for bond in template_mol.GetBonds():
        query.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.UNSPECIFIED)
    return query.GetMol()


@lru_cache(maxsize=None)
def _template_is_fully_aromatic(template_mol) -> bool:
    return all(atom.GetIsAromatic() for atom in template_mol.GetAtoms())


def _hydro_fused_parent_name(mol, template: dict, match: Sequence[int]) -> str:
    hydro_locs = []
    charged_locs = []
    charged_has_hydrogen = False
    for template_idx, mol_idx in enumerate(match):
        template_atom = template["mol"].GetAtomWithIdx(template_idx)
        atom = mol.GetAtomWithIdx(mol_idx)
        charge_matches = template_atom.GetFormalCharge() == atom.GetFormalCharge()
        charge_is_parent_ium = (
            template_atom.GetFormalCharge() == 0
            and atom.GetFormalCharge() > 0
            and atom.GetSymbol() in {"N", "S"}
        )
        if not charge_matches and not charge_is_parent_ium:
            return ""
        locant = template["locants"].get(template_idx, "")
        if charge_is_parent_ium:
            if not locant.isdigit():
                return ""
            charged_locs.append(int(locant))
            if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0:
                charged_has_hydrogen = True
        if not template_atom.GetIsAromatic() or atom.GetIsAromatic():
            continue
        if _has_unsaturated_core_bond(mol, mol_idx, match):
            continue
        if not locant.isdigit():
            return ""
        hydro_locs.append(int(locant))
    if not hydro_locs:
        return ""
    hydro_locs = sorted(set(hydro_locs))
    hydro = {
        2: "dihydro",
        4: "tetrahydro",
        6: "hexahydro",
        8: "octahydro",
        10: "decahydro",
        12: "dodecahydro",
    }.get(len(hydro_locs))
    if not hydro:
        return ""
    locants = ",".join(str(loc) for loc in hydro_locs)
    name = f"{locants}-{hydro}-{template['name']}"
    if len(charged_locs) == 1:
        stem = name[:-1] if name.endswith("e") else name
        return f"{stem}-{charged_locs[0]}-ium"
    return name


def _has_unsaturated_core_bond(mol, mol_idx: int, match: Sequence[int]) -> bool:
    core = set(match)
    atom = mol.GetAtomWithIdx(mol_idx)
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx not in core:
            continue
        bond = mol.GetBondBetweenAtoms(mol_idx, nidx)
        if bond is not None and bond.GetBondTypeAsDouble() > 1.0:
            return True
    return False


def _fused_substituent_suffix(parent_name: str, locant: str) -> str:
    stem = FUSED_SUBSTITUENT_STEMS.get(parent_name)
    if not stem:
        stem = parent_name
    return f"{stem}-{locant}-yl"


def _oxygen_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    parent = mol.GetAtomWithIdx(parent_idx)
    heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
    if atom.GetFormalCharge() < 0 and parent.GetSymbol() == "N" and parent.GetFormalCharge() > 0:
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            return "oxido"
    if atom.GetFormalCharge() < 0 and not heavy:
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            return "oxido"
    if atom.GetFormalCharge() > 0 and not heavy:
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            return "oxidaniumyl"
    if atom.GetFormalCharge() > 0 and len(heavy) == 1:
        ylidene_oxidanium = _ylidene_oxidaniumyl_prefix(mol, heavy[0], idx)
        if ylidene_oxidanium:
            return ylidene_oxidanium
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        return "oxo"
    if atom.GetFormalCharge() != 0:
        return ""
    if not heavy:
        return "hydroxy"
    if len(heavy) == 1:
        if _is_nitro_nitrogen(mol, heavy[0]):
            return "nitrooxy"
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() in HALO:
            return HALO[mol.GetAtomWithIdx(heavy[0]).GetSymbol()] + "oxy"
        carbonyl = _carbonyl_substituent_prefix(mol, heavy[0], idx)
        if carbonyl:
            return carbonyl + "oxy"
        if mol.GetAtomWithIdx(heavy[0]).IsInRing():
            ring = _ring_substituent_prefix(mol, heavy[0], idx)
            if not ring:
                return ""
            if ring == "phenyl":
                return "phenoxy"
            return f"({ring})oxy"
        alk = _substituent_from_root(mol, heavy[0], idx)
        return _alkyl_to_alkoxy(alk)
    return ""


def _ylidene_oxidaniumyl_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(carbon_idx, oxygen_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in carbon.GetNeighbors() if nbr.GetIdx() != oxygen_idx]
    if not heavy:
        return "methylideneoxidaniumyl"
    if len(heavy) != 1:
        return ""
    chain = _linear_carbon_chain(mol, heavy[0], carbon_idx, max_len=9)
    if not chain:
        return ""
    root = ROOTS.get(len(chain) + 1)
    return root + "ylideneoxidaniumyl" if root else ""


def _nitrogen_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if _is_diazo_attachment(mol, idx, parent_idx):
        return "diazo"
    if parent_bond is not None and parent_bond.GetBondType() == Chem.BondType.DOUBLE:
        return _imino_prefix(mol, idx, parent_idx)
    cumulene = _isocyanato_prefix(mol, idx, parent_idx)
    if cumulene:
        return cumulene
    if _is_isocyano_nitrogen(mol, idx):
        return "isocyano"
    if _is_diazonio_attachment(mol, idx, parent_idx):
        return "diazonio"
    azoxy = _azoxy_prefix(mol, idx, parent_idx)
    if azoxy:
        return azoxy
    diazenyl = _diazenyl_prefix(mol, idx, parent_idx)
    if diazenyl:
        return diazenyl
    hydrazone = _hydrazone_prefix(mol, idx, parent_idx)
    if hydrazone:
        return hydrazone
    imino_amino = _imino_amino_prefix(mol, idx, parent_idx)
    if imino_amino:
        return imino_amino
    amidino_amino = _amidino_amino_prefix(mol, idx, parent_idx)
    if amidino_amino:
        return amidino_amino
    if _is_azido_attachment(mol, idx, parent_idx):
        return "azido"
    if _nitro_branch_predicate(mol, idx, parent_idx):
        return "nitro"
    if _is_nitroso_attachment(mol, idx, parent_idx):
        return "nitroso"
    if atom.GetFormalCharge() > 0:
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
        if not heavy:
            return "azaniumyl"
        names = []
        for nidx in heavy:
            if mol.GetAtomWithIdx(nidx).IsInRing():
                name = _ring_substituent_prefix(mol, nidx, idx)
            else:
                name = _substituent_from_root(mol, nidx, idx)
            if not name:
                return ""
            names.append(name)
        counts = Counter(names)
        parts = []
        for name in sorted(counts):
            count = counts[name]
            if count > 1:
                parts.append(MULT.get(count, "") + name)
            else:
                parts.append(name)
        return "".join(parts) + "azaniumyl"
    if atom.GetFormalCharge() < 0:
        azanidyl = _azanidyl_prefix(mol, idx, parent_idx)
        if azanidyl:
            return azanidyl
    if atom.GetFormalCharge() != 0:
        return ""
    heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
    if heavy and all(mol.GetAtomWithIdx(nidx).GetSymbol() in HALO for nidx in heavy):
        names = [HALO[mol.GetAtomWithIdx(nidx).GetSymbol()] for nidx in heavy]
        counts = Counter(names)
        return "".join((MULT.get(count, "") if count > 1 else "") + name for name, count in sorted(counts.items())) + "amino"
    if not heavy:
        return "amino"
    names = []
    for nidx in heavy:
        symbol = mol.GetAtomWithIdx(nidx).GetSymbol()
        if symbol in HALO:
            name = HALO[symbol]
        elif mol.GetAtomWithIdx(nidx).IsInRing():
            name = _ring_substituent_prefix(mol, nidx, idx)
        elif symbol == "N":
            name = _nitrogen_prefix(mol, nidx, idx)
        elif symbol == "S":
            name = _sulfur_prefix(mol, nidx, idx)
        else:
            name = (
                "cyano"
                if _is_nitrile_carbon(mol, nidx)
                else _functional_carbon_substituent_name(mol, nidx, idx)
                or _carbonyl_substituent_prefix(mol, nidx, idx)
                or _substituent_from_root(mol, nidx, idx)
            )
        if not name:
            return ""
        names.append(name)
    if len(names) == 1:
        if "-" in names[0] or names[0].startswith("("):
            return f"({names[0]})amino"
        return names[0] + "amino"
    counts = Counter(names)
    if all(count == 1 for count in counts.values()) and all(
        "-" not in name and not name.startswith("(") and ")" not in name for name in counts
    ):
        ordered = sorted(counts)
        return ordered[0] + "".join(f"({name})" for name in ordered[1:]) + "amino"
    if all(count == 1 for count in counts.values()):
        parts = []
        for name in sorted(counts):
            if "-" in name or name.startswith("(") or ")" in name:
                parts.append(f"N-({name})")
            else:
                parts.append(f"N-{name}")
        return "-".join(parts) + "amino"
    parts = []
    for name in sorted(counts):
        count = counts[name]
        if count > 1 and name.endswith(("sulfonyl", "sulfinyl")):
            multi = {2: "bis", 3: "tris", 4: "tetrakis"}.get(count, MULT.get(count, "") + "kis")
            parts.append(f"{multi}({name})")
        elif count > 1 and ("-" in name or name.startswith("(")):
            multi = {2: "bis", 3: "tris", 4: "tetrakis"}.get(count, MULT.get(count, "") + "kis")
            parts.append(f"{multi}({name})")
        elif count == 1 and (name.endswith("carbonyl") or name.endswith("sulfonyl")) and len(names) > 1:
            parts.append(f"({name})")
        elif "-" in name or name.startswith("("):
            parts.append(f"({name})")
        else:
            parts.append((MULT.get(count, "") if count > 1 else "") + name)
    return "".join(parts) + "amino"


def _azanidyl_prefix(mol, idx: int, parent_idx: int) -> str:
    parent = mol.GetAtomWithIdx(parent_idx)
    if parent.GetFormalCharge() <= 0:
        return ""
    heavy = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() != parent_idx]
    if not heavy:
        return "azanidyl"
    if len(heavy) != 1:
        return ""
    substituent_idx = heavy[0]
    substituent = mol.GetAtomWithIdx(substituent_idx)
    if substituent.IsInRing():
        name = _ring_substituent_prefix(mol, substituent_idx, idx)
    elif substituent.GetSymbol() == "S":
        name = _sulfur_prefix(mol, substituent_idx, idx)
    elif substituent.GetSymbol() == "C":
        name = (
            "cyano"
            if _is_nitrile_carbon(mol, substituent_idx)
            else _functional_carbon_substituent_name(mol, substituent_idx, idx)
            or _carbonyl_substituent_prefix(mol, substituent_idx, idx)
            or _substituent_from_root(mol, substituent_idx, idx)
        )
    elif substituent.GetSymbol() == "N":
        name = _nitrogen_prefix(mol, substituent_idx, idx)
    else:
        return ""
    if not name:
        return ""
    if "-" in name or name.startswith("("):
        return f"({name})azanidyl"
    return name + "azanidyl"


def _functional_carbon_substituent_name(mol, idx: int, parent_idx: int) -> str:
    if mol.GetAtomWithIdx(idx).GetSymbol() != "C":
        return ""
    return (
        _ketenyl_prefix(mol, idx, parent_idx)
        or _enone_substituent_prefix(mol, idx, parent_idx)
        or _enal_substituent_prefix(mol, idx, parent_idx)
        or _haloethenyl_prefix(mol, idx, parent_idx)
        or _alkynyl_substituent_prefix(mol, idx, parent_idx)
    )


def _imino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
    if not heavy:
        return "imino"
    if len(heavy) != 1:
        return ""
    attached = heavy[0]
    if mol.GetAtomWithIdx(attached).IsInRing():
        name = _ring_substituent_prefix(mol, attached, idx)
    else:
        name = "cyano" if _is_nitrile_carbon(mol, attached) else _substituent_from_root(mol, attached, idx)
    if not name:
        return ""
    if "-" in name or name.startswith("("):
        return f"({name})imino"
    return name + "imino"


def _isocyanato_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    for nbr in atom.GetNeighbors():
        cidx = nbr.GetIdx()
        if cidx == parent_idx or nbr.GetSymbol() != "C":
            continue
        n_c_bond = mol.GetBondBetweenAtoms(idx, cidx)
        if n_c_bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        terminal = [
            other
            for other in nbr.GetNeighbors()
            if other.GetIdx() != idx
            and mol.GetBondBetweenAtoms(cidx, other.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        ]
        if len(terminal) != 1:
            continue
        symbol = terminal[0].GetSymbol()
        if symbol == "O":
            return "isocyanato"
        if symbol == "S":
            return "isothiocyanato"
    return ""


def _diazenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx or nbr.GetSymbol() != "N" or nbr.GetFormalCharge() != 0:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        distal = [n.GetIdx() for n in nbr.GetNeighbors() if n.GetIdx() != idx]
        if not distal:
            return "diazenyl"
        if len(distal) != 1:
            continue
        didx = distal[0]
        if mol.GetAtomWithIdx(didx).IsInRing():
            name = _ring_substituent_prefix(mol, didx, nidx)
        else:
            name = _substituent_from_root(mol, didx, nidx)
        if name:
            return name + "diazenyl"
    return ""


def _azoxy_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx or nbr.GetSymbol() != "N" or nbr.GetFormalCharge() <= 0:
            continue
        n_bond = mol.GetBondBetweenAtoms(idx, nidx)
        if n_bond is None or n_bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        oxido = []
        organic = []
        for distal in nbr.GetNeighbors():
            didx = distal.GetIdx()
            if didx == idx:
                continue
            bond = mol.GetBondBetweenAtoms(nidx, didx)
            if (
                distal.GetSymbol() == "O"
                and distal.GetFormalCharge() < 0
                and bond is not None
                and bond.GetBondType() == Chem.BondType.SINGLE
            ):
                oxido.append(didx)
            elif bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
                organic.append(didx)
        if len(oxido) != 1 or len(organic) != 1:
            continue
        didx = organic[0]
        organic_atom = mol.GetAtomWithIdx(didx)
        if organic_atom.IsInRing():
            name = _ring_substituent_prefix(mol, didx, nidx)
            if name and not (name.startswith("(") or "-" in name):
                return f"({name}azoxy)"
        else:
            name = _substituent_from_root(mol, didx, nidx)
        if not name:
            continue
        if "-" in name or name.startswith("("):
            return f"({name})azoxy"
        return name + "azoxy"
    return ""


def _hydrazone_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]
    if len(heavy) != 1:
        return ""
    distal_n_idx = heavy[0]
    distal_n = mol.GetAtomWithIdx(distal_n_idx)
    if distal_n.GetSymbol() != "N" or distal_n.GetFormalCharge() != 0:
        return ""
    n_bond = mol.GetBondBetweenAtoms(idx, distal_n_idx)
    if n_bond is None or n_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    imine_carbons = [
        nbr.GetIdx()
        for nbr in distal_n.GetNeighbors()
        if nbr.GetIdx() != idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(distal_n_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]
    if len(imine_carbons) != 1:
        return ""
    ylidene = _imine_carbon_ylidene(mol, imine_carbons[0], distal_n_idx)
    return ylidene + "hydrazinyl" if ylidene else ""


def _imino_amino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    imine_carbons = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]
    if len(imine_carbons) != 1:
        return ""
    ylidene = _imine_carbon_ylidene(mol, imine_carbons[0], idx)
    return ylidene + "amino" if ylidene else ""


def _amidino_amino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]
    if len(heavy) != 1:
        return ""
    carbon_idx = heavy[0]
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return ""
    n_c_bond = mol.GetBondBetweenAtoms(idx, carbon_idx)
    if n_c_bond is None or n_c_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    terminal_imino = []
    other_heavy = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if (
            nbr.GetSymbol() == "N"
            and nbr.GetFormalCharge() == 0
            and bond is not None
            and bond.GetBondType() == Chem.BondType.DOUBLE
            and not [nnbr for nnbr in nbr.GetNeighbors() if nnbr.GetIdx() != carbon_idx and nnbr.GetAtomicNum() > 1]
        ):
            terminal_imino.append(nidx)
        else:
            other_heavy.append(nidx)
    if len(terminal_imino) != 1 or other_heavy:
        return ""
    return "iminomethylamino"


def _imine_carbon_ylidene(mol, carbon_idx: int, nitrogen_idx: int) -> str:
    atom = mol.GetAtomWithIdx(carbon_idx)
    if atom.IsInRing():
        rings = [ring for ring in mol.GetRingInfo().AtomRings() if carbon_idx in ring]
        if len(rings) != 1:
            return ""
        ring = rings[0]
        if not all(mol.GetAtomWithIdx(idx).GetSymbol() == "C" for idx in ring):
            return ""
        ring_set = set(ring)
        for a, b in zip(ring, ring[1:] + ring[:1]):
            bond = mol.GetBondBetweenAtoms(a, b)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return ""
        for ring_idx in ring:
            for nbr in mol.GetAtomWithIdx(ring_idx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx not in ring_set and nidx != nitrogen_idx:
                    return ""
        root = ROOTS.get(len(ring))
        return f"cyclo{root}ylidene" if root else ""
    substituents = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != nitrogen_idx]
    if len(substituents) == 1:
        sub_idx = substituents[0]
        sub_atom = mol.GetAtomWithIdx(sub_idx)
        sub_bond = mol.GetBondBetweenAtoms(carbon_idx, sub_idx)
        if (
            sub_atom.GetSymbol() == "N"
            and sub_atom.GetFormalCharge() == 0
            and sub_bond is not None
            and sub_bond.GetBondType() == Chem.BondType.SINGLE
            and not [nbr for nbr in sub_atom.GetNeighbors() if nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1]
        ):
            return "aminomethylidene"
        if (
            sub_atom.GetSymbol() == "N"
            and sub_atom.GetFormalCharge() == 0
            and sub_bond is not None
            and sub_bond.GetBondType() == Chem.BondType.SINGLE
        ):
            amino = _nitrogen_prefix(mol, sub_idx, carbon_idx)
            if amino and amino.endswith("amino") and "-" not in amino and "(" not in amino and ")" not in amino:
                return amino + "methylidene"
        if (
            sub_atom.GetSymbol() == "N"
            and sub_atom.GetFormalCharge() == 0
            and sub_bond is not None
            and sub_bond.GetBondType() == Chem.BondType.DOUBLE
            and not [nbr for nbr in sub_atom.GetNeighbors() if nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1]
        ):
            return "iminomethylidene"
        if mol.GetAtomWithIdx(sub_idx).IsInRing():
            name = _ring_substituent_prefix(mol, sub_idx, carbon_idx)
            if name and name.endswith("yl"):
                return name + "methylidene"
            return ""
        name = _substituent_from_root(mol, sub_idx, carbon_idx)
        if name and "-" not in name and not name.startswith("("):
            return name + "methylidene"
        return ""
    if len(substituents) == 2:
        names = []
        for sub_idx in substituents:
            if mol.GetAtomWithIdx(sub_idx).IsInRing():
                name = _ring_substituent_prefix(mol, sub_idx, carbon_idx)
            else:
                name = _substituent_from_root(mol, sub_idx, carbon_idx)
            if not name or "-" in name or name.startswith("("):
                return ""
            names.append(name)
        counts = Counter(names)
        parts = []
        for name in sorted(counts):
            count = counts[name]
            parts.append((MULT.get(count, "") if count > 1 else "") + name)
        return "".join(parts) + "methylidene"
    return _alkylidene_prefix(mol, carbon_idx, nitrogen_idx)


def _sulfur_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if atom.GetFormalCharge() < 0:
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
        if not heavy and parent_bond is not None and parent_bond.GetBondType() == Chem.BondType.SINGLE:
            return "sulfido"
        return ""
    if atom.GetFormalCharge() > 0:
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
        if not heavy and parent_bond is not None and parent_bond.GetBondType() == Chem.BondType.SINGLE:
            return "sulfaniumyl"
        return ""
    if atom.GetFormalCharge() != 0:
        return ""
    if parent_bond is not None and parent_bond.GetBondType() == Chem.BondType.DOUBLE:
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
        if not heavy:
            return "thioxo"
    halogens = [
        n.GetSymbol()
        for n in atom.GetNeighbors()
        if n.GetIdx() != parent_idx and n.GetSymbol() in HALO
    ]
    if len(halogens) == 5 and all(symbol == "F" for symbol in halogens):
        return "pentafluorosulfanyl"
    non_parent = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
    if len(halogens) == 3 and all(symbol == "F" for symbol in halogens) and len(non_parent) == 3:
        return "trifluorosulfanyl"
    double_oxygens = {
        n.GetIdx()
        for n in atom.GetNeighbors()
        if n.GetSymbol() == "O"
        and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    }
    double_o = len(double_oxygens)
    if _is_sulfonate_sulfur(mol, idx):
        return "sulfonato"
    sulfonimidoyl = _sulfonimidoyl_prefix(mol, idx, parent_idx, double_o)
    if sulfonimidoyl:
        return sulfonimidoyl
    if double_o == 2 and len(halogens) == 1:
        return {"F": "fluorosulfonyl", "Cl": "chlorosulfonyl", "Br": "bromosulfonyl", "I": "iodosulfonyl"}.get(halogens[0], "")
    if double_o == 1 and len(halogens) == 1:
        return {"F": "fluorosulfinyl", "Cl": "chlorosulfinyl", "Br": "bromosulfinyl", "I": "iodosulfinyl"}.get(halogens[0], "")
    if double_o == 0 and len(non_parent) == 1 and halogens:
        return HALO[halogens[0]] + "sulfanyl"
    heavy = [
        n.GetIdx()
        for n in atom.GetNeighbors()
        if n.GetIdx() != parent_idx and n.GetIdx() not in double_oxygens and not _is_terminal_carbonyl_oxygen(n)
    ]
    if double_o == 2 and not heavy:
        return "sulfonyl"
    if not heavy:
        return "sulfanyl"
    if len(heavy) == 1:
        if mol.GetAtomWithIdx(heavy[0]).IsInRing():
            ring = _ring_substituent_prefix(mol, heavy[0], idx)
            if not ring:
                return ""
            if double_o == 2:
                return ring + "sulfonyl"
            if double_o == 1:
                return ring + "sulfinyl"
            return ring + "sulfanyl"
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "N" and double_o == 2:
            amino = _nitrogen_prefix(mol, heavy[0], idx)
            if not amino:
                return ""
            return amino + "sulfonyl"
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "O" and double_o == 2:
            oxy = _oxygen_prefix(mol, heavy[0], idx)
            if not oxy:
                return ""
            return oxy + "sulfonyl"
        if mol.GetAtomWithIdx(parent_idx).IsInRing() and mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "S" and double_o == 0:
            sulfur = _sulfur_prefix(mol, heavy[0], idx)
            if not sulfur:
                return ""
            if sulfur.endswith("sulfanyl"):
                return sulfur[: -len("sulfanyl")] + "disulfanyl"
            return sulfur + "sulfanyl"
        alk = _substituent_from_root(mol, heavy[0], idx)
        if not alk:
            return ""
        if double_o == 2:
            return alk + "sulfonyl"
        if double_o == 1:
            return alk + "sulfinyl"
        return alk + "sulfanyl"
    return ""

def _sulfonimidoyl_prefix(mol, idx: int, parent_idx: int, double_o: int) -> str:
    if double_o != 1:
        return ""
    imino = []
    single_groups = []
    for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx or nbr.GetSymbol() == "O":
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE:
            imino.append(nidx)
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            single_groups.append(nidx)
    if len(imino) != 1 or len(single_groups) != 1:
        return ""

    single_atom = mol.GetAtomWithIdx(single_groups[0])
    if single_atom.GetSymbol() in HALO:
        s_group = HALO[single_atom.GetSymbol()]
    else:
        s_group = _substituent_from_root(mol, single_groups[0], idx)
    if not s_group:
        return ""
    n_group = _imino_n_substituent_name(mol, imino[0], idx)
    if n_group is None:
        return ""
    if not n_group:
        return f"{s_group}sulfonimidoyl"
    return f"N-({n_group})-S-{s_group}sulfonimidoyl"


def _imino_n_substituent_name(mol, nidx: int, sulfur_idx: int) -> str | None:
    heavy = [n.GetIdx() for n in mol.GetAtomWithIdx(nidx).GetNeighbors() if n.GetIdx() != sulfur_idx]
    if not heavy:
        return ""
    if len(heavy) != 1:
        return None
    attached = heavy[0]
    carbamoyl = _carbamoyl_substituent_name(mol, attached, nidx)
    if carbamoyl:
        return carbamoyl
    name = _substituent_from_root(mol, attached, nidx)
    if not name:
        return None
    return name


def _carbamoyl_substituent_name(mol, cidx: int, nidx: int) -> str:
    atom = mol.GetAtomWithIdx(cidx)
    if atom.GetSymbol() != "C":
        return ""
    has_double_o = False
    amine = None
    for nbr in atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == nidx:
            continue
        bond = mol.GetBondBetweenAtoms(cidx, nbr_idx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_double_o = True
        elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
            amine = nbr_idx
        else:
            return ""
    if not has_double_o or amine is None:
        return ""
    groups = []
    for nbr in mol.GetAtomWithIdx(amine).GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == cidx:
            continue
        name = _substituent_from_root(mol, nbr_idx, amine)
        if not name:
            return ""
        groups.append(name)
    if not groups:
        return "carbamoyl"
    counts = Counter(groups)
    parts = []
    for name in sorted(counts):
        count = counts[name]
        if count > 1:
            if "-" in name or name.startswith("("):
                parts.append(f"{MULT.get(count, '')}({name})")
            else:
                parts.append(MULT.get(count, "") + name)
        else:
            parts.append(name)
    return "".join(parts) + "carbamoyl"


def _substituent_from_root(mol, root_idx: int, blocked_idx: int) -> str:
    cache = getattr(mol, "_nispo_substituent_from_root_cache", None)
    if cache is None:
        cache = {}
        try:
            mol._nispo_substituent_from_root_cache = cache
        except Exception:
            cache = None
    key = (root_idx, blocked_idx)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    atoms = _collect_subtree(mol, root_idx, blocked_idx)
    paths = _rooted_paths(mol, root_idx, atoms)
    if not paths:
        if cache is not None:
            cache[key] = ""
        return ""
    best = ""
    best_key = None
    for path in paths:
        name = _name_subtree_path(mol, path, atoms, blocked_idx)
        if not name:
            continue
        key = (-len(path), _substituent_locants(mol, path, atoms, blocked_idx), name)
        if best_key is None or key < best_key:
            best_key = key
            best = name
    if cache is not None:
        cache[(root_idx, blocked_idx)] = best
    return best


def _name_subtree_path(mol, path: Sequence[int], atoms: set[int], blocked_idx: int) -> str:
    path_set = set(path)
    prefixes = []
    for pos, idx in enumerate(path, start=1):
        atom = mol.GetAtomWithIdx(idx)
        if not _can_be_in_parent_skeleton(mol, idx):
            return ""
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == blocked_idx or nidx in path_set:
                continue
            if nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0:
                continue
            if nidx not in atoms:
                continue
            prefix = _branch_prefix(mol, nidx, idx)
            if not prefix:
                return ""
            prefixes.append((str(pos), prefix))
    parent = _parent_chain_name(mol, path, as_substituent=True)
    if not parent:
        return ""
    repl = _replacement_prefixes(mol, path)
    prefix_text = _format_prefixes(prefixes)
    pieces = []
    if prefix_text:
        pieces.append(prefix_text)
    if repl:
        pieces.append(repl + parent)
    else:
        pieces.append(parent)
    return "-".join(pieces)


def _substituent_locants(mol, path: Sequence[int], atoms: set[int], blocked_idx: int) -> list[int]:
    path_set = set(path)
    locs = []
    for pos, idx in enumerate(path, start=1):
        if mol.GetAtomWithIdx(idx).GetSymbol() != "C":
            locs.append(pos)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0:
                continue
            if nidx != blocked_idx and nidx in atoms and nidx not in path_set:
                locs.append(pos)
    return locs


def _rooted_paths(mol, root_idx: int, atoms: set[int]) -> list[tuple[int, ...]]:
    cache = getattr(mol, "_nispo_rooted_paths_cache", None)
    if cache is None:
        cache = {}
        try:
            mol._nispo_rooted_paths_cache = cache
        except Exception:
            cache = None
    cache_key = (root_idx, tuple(sorted(atoms)))
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return list(cached)
    if mol.GetAtomWithIdx(root_idx).IsInRing():
        path_atoms = set(atoms)
    else:
        path_atoms = {idx for idx in atoms if not mol.GetAtomWithIdx(idx).IsInRing()}
        path_atoms.add(root_idx)
    leaves = [
        idx
        for idx in path_atoms
        if idx != root_idx
        and sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() in path_atoms) <= 1
        and _can_be_in_parent_skeleton(mol, idx)
    ]
    if not leaves:
        leaves = [root_idx]
    paths = []
    best_len = 0
    for leaf in leaves:
        if leaf == root_idx:
            path = (root_idx,)
        else:
            path = Chem.rdmolops.GetShortestPath(mol, root_idx, leaf)
        if any(idx not in path_atoms or not _can_be_in_parent_skeleton(mol, idx) for idx in path):
            continue
        if len(path) > best_len:
            best_len = len(path)
            paths.clear()
        if len(path) == best_len:
            paths.append(tuple(path))
    if cache is not None:
        cache[cache_key] = tuple(paths)
    return paths


def _collect_subtree(mol, start_idx: int, blocked_idx: int) -> set[int]:
    cache = getattr(mol, "_nispo_collect_subtree_cache", None)
    if cache is None:
        cache = {}
        try:
            mol._nispo_collect_subtree_cache = cache
        except Exception:
            cache = None
    key = (start_idx, blocked_idx)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return set(cached)
    seen = {blocked_idx}
    stack = [start_idx]
    atoms = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        atoms.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0:
                continue
            if nidx not in seen:
                stack.append(nidx)
    if cache is not None:
        cache[key] = frozenset(atoms)
    return atoms


def _is_terminal_carbonyl_oxygen(atom) -> bool:
    if atom.GetSymbol() != "O":
        return False
    if atom.GetDegree() != 1:
        return False
    bond = atom.GetBonds()[0]
    return bond.GetBondType() == Chem.BondType.DOUBLE


def _is_carbonyl_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return False
    return any(
        nbr.GetSymbol() == "O"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        for nbr in atom.GetNeighbors()
    )


def _is_terminal_ketene_alpha_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return False
    single_neighbors = 0
    has_ketene_carbonyl = False
    for nbr in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if bond.GetBondType() == Chem.BondType.SINGLE:
            single_neighbors += 1
            continue
        if bond.GetBondType() != Chem.BondType.DOUBLE or nbr.GetSymbol() != "C":
            return False
        has_double_o = any(
            onbr.GetSymbol() == "O"
            and mol.GetBondBetweenAtoms(nbr.GetIdx(), onbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            for onbr in nbr.GetNeighbors()
            if onbr.GetIdx() != idx
        )
        if has_double_o:
            has_ketene_carbonyl = True
        else:
            return False
    return has_ketene_carbonyl and single_neighbors == 1


def _is_carboxylate_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return False
    has_double_o = False
    has_single_ominus = False
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() != "O":
            continue
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            has_double_o = True
        elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            has_single_ominus = True
    return has_double_o and has_single_ominus


def _acyl_halide_prefix(mol, idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return ""
    has_double_o = any(
        nbr.GetSymbol() == "O"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        for nbr in atom.GetNeighbors()
    )
    if not has_double_o:
        return ""
    halogens = [
        nbr.GetSymbol()
        for nbr in atom.GetNeighbors()
        if nbr.GetSymbol() in HALO
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    if len(halogens) != 1:
        return ""
    return {
        "F": "fluorocarbonyl",
        "Cl": "chlorocarbonyl",
        "Br": "bromocarbonyl",
        "I": "iodocarbonyl",
    }.get(halogens[0], "")


def _carbonyl_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    has_double_o = False
    heavy = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_double_o = True
            continue
        heavy.append(nidx)
    if not has_double_o or len(heavy) != 1:
        return ""
    distal = heavy[0]
    hydroxycarbamoyl = _hydroxycarbamoyl_prefix(mol, distal, idx)
    if hydroxycarbamoyl:
        return hydroxycarbamoyl
    alkoxycarbonyl = _alkoxycarbonyl_prefix(mol, distal, idx)
    if alkoxycarbonyl:
        return alkoxycarbonyl
    if mol.GetAtomWithIdx(distal).IsInRing():
        name = _ring_substituent_prefix(mol, distal, idx)
    else:
        name = _substituent_from_root(mol, distal, idx)
    if not name:
        return ""
    return name + "carbonyl"


def _hydroxycarbamoyl_prefix(mol, nitrogen_idx: int, carbonyl_idx: int) -> str:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(nitrogen_idx, carbonyl_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    hydroxy = []
    other = []
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbonyl_idx:
            continue
        n_bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and n_bond.GetBondType() == Chem.BondType.SINGLE:
            terminal = [nnbr for nnbr in nbr.GetNeighbors() if nnbr.GetIdx() != nitrogen_idx and nnbr.GetAtomicNum() > 1]
            if terminal:
                other.append(nidx)
            else:
                hydroxy.append(nidx)
        else:
            other.append(nidx)
    if len(hydroxy) == 1 and not other:
        return "(hydroxyamino)carbonyl"
    return ""


def _alkoxycarbonyl_prefix(mol, oxygen_idx: int, carbonyl_idx: int) -> str:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(oxygen_idx, carbonyl_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in oxygen.GetNeighbors() if nbr.GetIdx() != carbonyl_idx]
    if not heavy:
        return "carboxy"
    if len(heavy) != 1:
        return ""
    attached = heavy[0]
    if mol.GetAtomWithIdx(attached).IsInRing():
        ring = _ring_substituent_prefix(mol, attached, oxygen_idx)
        if not ring:
            return ""
        oxy = "phenoxy" if ring == "phenyl" else f"({ring})oxy"
    else:
        alk = _substituent_from_root(mol, attached, oxygen_idx)
        if not alk:
            return ""
        oxy = _alkyl_to_alkoxy(alk)
    return oxy + "carbonyl"


def _is_isocyano_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() >= 0:
        return False
    return any(
        nbr.GetSymbol() == "N"
        and nbr.GetFormalCharge() > 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_nitrile_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return False
    return any(
        nbr.GetSymbol() == "N"
        and nbr.GetFormalCharge() == 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_nitrile_nitrogen(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return False
    return any(
        nbr.GetSymbol() == "C"
        and nbr.GetFormalCharge() == 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_isocyano_nitrogen(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return False
    return any(
        nbr.GetSymbol() == "C"
        and nbr.GetFormalCharge() < 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_sulfonate_sulfur(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "S":
        return False
    double_o = 0
    single_ominus = 0
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() != "O":
            continue
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            double_o += 1
        elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            single_ominus += 1
    return double_o >= 2 and single_ominus >= 1


def _is_carbonyl_oxygen_idx(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    return _is_terminal_carbonyl_oxygen(atom)


def _is_nitro_nitrogen(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N":
        return False
    oxy = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O"]
    return atom.GetFormalCharge() > 0 and len(oxy) >= 2


def _is_azido_attachment(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N":
        return False
    for mid in atom.GetNeighbors():
        if mid.GetIdx() == parent_idx or mid.GetSymbol() != "N" or mid.GetFormalCharge() <= 0:
            continue
        for terminal in mid.GetNeighbors():
            if terminal.GetIdx() != idx and terminal.GetSymbol() == "N" and terminal.GetFormalCharge() < 0:
                return True
    return False


def _is_diazo_attachment(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return False
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.DOUBLE:
        return False
    return any(
        nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "N"
        and nbr.GetFormalCharge() < 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        for nbr in atom.GetNeighbors()
    )


def _is_diazo_nitrogen(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return False
    has_terminal_n = False
    has_carbon = False
    for nbr in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if nbr.GetSymbol() == "N" and nbr.GetFormalCharge() < 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_terminal_n = True
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_carbon = True
    return has_terminal_n and has_carbon


def _is_diazonio_attachment(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return False
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    return any(
        nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "N"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_nitroso_attachment(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N":
        return False
    return any(
        nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "O"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        for nbr in atom.GetNeighbors()
    )
