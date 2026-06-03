from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from .branch import BranchRuleSpec, first_branch_name
from .branch_rendering import (
    branch_render_context,
    record_branch_legacy_prefix_call,
    render_branch_candidates,
    select_branch_prefix_text,
)
from .constants import HALO, MULT, REPLACEMENT, ROOTS, SKELETON_ATOMS
from .formatting import (
    join_locants as _join_locants,
    substituent_parent_suffix as _substituent_parent_suffix,
    unsaturated_parent as _unsaturated_parent,
)
from .molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set


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
    telemetry = mol_cache_get(mol, "candidate_telemetry")
    if telemetry is not None:
        telemetry.branch_prefix_calls += 1
        active_rule_family = telemetry.active_rule_family or "unknown"
        telemetry.branch_prefix_calls_by_rule[active_rule_family] += 1
    else:
        active_rule_family = ""
    cache = mol_cache_dict(mol, "branch_prefix_cache")
    key = (idx, parent_idx)
    cached = cache.get(key)
    if cached is not None:
        if telemetry is not None:
            telemetry.branch_prefix_cache_hits += 1
            telemetry.branch_prefix_cache_hits_by_rule[active_rule_family] += 1
        return cached
    if telemetry is not None:
        telemetry.branch_prefix_cache_misses += 1
        telemetry.branch_prefix_cache_misses_by_rule[active_rule_family] += 1
    in_progress = mol_cache_get(mol, "branch_prefix_in_progress")
    if in_progress is None:
        in_progress = mol_cache_set(mol, "branch_prefix_in_progress", set())
    if key in in_progress:
        if telemetry is not None:
            telemetry.branch_prefix_in_progress_rejections += 1
        return ""

    in_progress.add(key)
    try:
        render_ctx = branch_render_context(mol, Chem)
        prefix = select_branch_prefix_text(
            render_ctx,
            render_branch_candidates(render_ctx, idx, parent_idx),
        )
        if prefix:
            cache[key] = prefix
            if telemetry is not None:
                telemetry.branch_prefix_produced[_branch_prefix_telemetry_key(prefix)] += 1
            return prefix
        record_branch_legacy_prefix_call(render_ctx)
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "H":
            prefix = _isotopic_hydrogen_prefix(atom)
        elif atom.IsInRing():
            prefix = _ring_substituent_prefix(mol, idx, parent_idx)
        elif atom.GetSymbol() in HALO:
            prefix = _terminal_halogen_prefix(mol, idx, parent_idx)
            if not prefix:
                rule_sets_by_symbol = _branch_rule_sets()
                rule_sets = rule_sets_by_symbol.get(atom.GetSymbol(), ()) + rule_sets_by_symbol.get("*", ())
                prefix = first_branch_name(rule_sets, mol, idx, parent_idx)
        elif atom.GetSymbol() == "C":
            prefix = _carbon_branch_prefix(mol, idx, parent_idx)
        else:
            rule_sets_by_symbol = _branch_rule_sets()
            rule_sets = rule_sets_by_symbol.get(atom.GetSymbol(), ()) + rule_sets_by_symbol.get("*", ())
            prefix = first_branch_name(rule_sets, mol, idx, parent_idx)
        cache[key] = prefix
        if telemetry is not None:
            telemetry_key = _branch_prefix_telemetry_key(prefix)
            telemetry.branch_prefix_produced[telemetry_key] += 1
            telemetry.branch_legacy_prefix_produced[telemetry_key] += 1
            telemetry.branch_legacy_prefix_detail[_branch_prefix_detail_telemetry_key(prefix)] += 1
        return prefix
    finally:
        in_progress.discard(key)


def _branch_prefix_telemetry_key(prefix: str) -> str:
    if not prefix:
        return "empty"
    if len(prefix) <= 40 and "(" not in prefix:
        return prefix
    return "complex"


def _branch_prefix_detail_telemetry_key(prefix: str) -> str:
    if not prefix:
        return "empty"
    if len(prefix) <= 160:
        return prefix
    if prefix.startswith("("):
        return "long-parenthesized"
    if prefix[:1].isdigit():
        return "long-locanted"
    return "long-prefix"


def _isotopic_hydrogen_prefix(atom) -> str:
    if atom.GetIsotope() == 2:
        return "deuterio"
    if atom.GetIsotope() == 3:
        return "tritio"
    return ""


def _terminal_halogen_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
        return ""
    isotope = atom.GetIsotope()
    symbol = atom.GetSymbol()
    name = HALO.get(symbol, "")
    if not name:
        return ""
    return f"[{isotope}{symbol}]{name}" if isotope else name


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
