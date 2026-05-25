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
from .branch import BranchRuleSet, BranchRuleSpec, first_branch_name
from .cycle_utils import ordered_ring, ring_system_extends_core
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
from .molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set
from .ring_rules import (
    ring_parent_candidates as _ring_parent_candidates,
    ring_substituent_suffix as _ring_substituent_suffix,
)
from .ring_resource_rules import resource_ring_parent_candidates as _resource_ring_parent_candidates
from .stereochemistry import assigned_atom_stereochemistry
from .substituent_resource_rules import resource_substituent_prefix as _resource_substituent_prefix
from .template_access import template_record_locants, template_record_mol, template_record_name
from .template_store import template_store


_HALOSULFUR_OXO_PREFIXES = {
    2: {"F": "fluorosulfonyl", "Cl": "chlorosulfonyl", "Br": "bromosulfonyl", "I": "iodosulfonyl"},
    1: {"F": "fluorosulfinyl", "Cl": "chlorosulfinyl", "Br": "bromosulfinyl", "I": "iodosulfinyl"},
}

_SULFUR_OXO_SUBSTITUENT_SUFFIX = {2: "sulfonyl", 1: "sulfinyl", 0: "sulfanyl"}


def can_be_in_parent_skeleton(mol, idx: int) -> bool:
    cache = mol_cache_dict(mol, "parent_skeleton_atom_cache")
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
    cache[idx] = result
    return result


def replacement_prefixes(mol, path: Sequence[int]) -> str:
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


def parent_chain_name(mol, path: Sequence[int], as_substituent: bool = False) -> str:
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


def branch_prefix(mol, idx: int, parent_idx: int) -> str:
    cache = mol_cache_dict(mol, "branch_prefix_cache")
    key = (idx, parent_idx)
    cached = cache.get(key)
    if cached is not None:
        return cached
    in_progress = mol_cache_get(mol, "branch_prefix_in_progress")
    if in_progress is None:
        in_progress = mol_cache_set(mol, "branch_prefix_in_progress", set())
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


from .branch.carbon import (
    _parenthesize_complex_prefix,
    _resource_prefix,
    _carbon_prefix,
    _carbon_branch_prefix,
    _deuterated_methyl_prefix,
    _isotopic_methyl_prefix,
    _has_neighbor_bond,
    _subtree_has_ring,
    _branch_rule_sets,
    _nitro_branch_predicate,
    _alkylidene_prefix,
    _hetero_methylidene_prefix,
    _enal_substituent_prefix,
    _enone_substituent_prefix,
    _haloethenyl_prefix,
    _ketenyl_prefix,
    _aldehyde_carbon,
    _iodanyl_acyloxy_prefix,
    _acyloxy_ligand_name,
    _acyl_parent_name,
    _linear_carbon_chain,
)

from .branch.ring_substituents import (
    _ring_substituent_prefix,
    _generic_polycyclic_ring_substituent_prefix,
    _ring_alkyl_prefix,
    _aryloxy_methyl_prefix,
    _iminomethyl_prefix,
    _difluoro_substituted_methyl_prefix,
    _sulfonamidomethyl_prefix,
    _ring_ethynyl_prefix,
    _dicyanoethenyl_prefix,
    _alkynyl_substituent_prefix,
    _simple_alkynyl_side_branches,
    _linear_alkynyl_paths,
    _simple_ring_substituent_prefix,
    _simple_ring_substituent_stereo_prefix,
    _ring_substituent_unsaturated_suffix,
    _charged_ring_nitrogen_locant,
    _fused_ring_substituent_prefix,
    _fused_ring_substituent_prefix_from_templates,
    _fused_substituent_candidate_key,
    _hydro_fused_ring_substituent_prefix_from_templates,
    _fused_substituent_prefix_text,
    _topology_query,
    _template_is_fully_aromatic,
    _hydro_fused_parent_name,
    _has_unsaturated_core_bond,
    _fused_substituent_suffix,
)

from .branch.hetero_atoms import (
    _oxygen_prefix,
    _ylidene_oxidaniumyl_prefix,
    _nitrogen_prefix,
    _nitrogen_ligand_prefix,
    _amino_prefix_from_ligands,
    _azanidyl_prefix,
    _functional_carbon_substituent_name,
    _imino_prefix,
    _isocyanato_prefix,
    _diazenyl_prefix,
    _azoxy_prefix,
    _hydrazone_prefix,
    _imino_amino_prefix,
    _amidino_amino_prefix,
    _imine_carbon_ylidene,
)

from .branch.functional_groups import (
    _sulfur_prefix,
    _halosulfur_prefix,
    _sulfur_oxo_substituent_suffix,
    _sulfonimidoyl_prefix,
    _imino_n_substituent_name,
    _carbamoyl_substituent_name,
    substituent_from_root,
    _name_subtree_path,
    substituent_locants,
    rooted_paths,
    collect_subtree,
    _is_terminal_carbonyl_oxygen,
    is_carbonyl_carbon,
    _is_terminal_ketene_alpha_carbon,
    _is_carboxylate_carbon,
    _acyl_halide_prefix,
    _carbonyl_substituent_prefix,
    _hydroxycarbamoyl_prefix,
    _alkoxycarbonyl_prefix,
    _is_isocyano_carbon,
    _is_nitrile_carbon,
    _is_nitrile_nitrogen,
    _is_isocyano_nitrogen,
    _is_sulfonate_sulfur,
    _is_carbonyl_oxygen_idx,
    _is_nitro_nitrogen,
    _is_azido_attachment,
    _is_diazo_attachment,
    _is_diazo_nitrogen,
    _is_diazonio_attachment,
    _is_nitroso_attachment,
)
