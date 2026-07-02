from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

from ..branch_rules import branch_prefix
from ..constants import HALO, MULT, ROOTS
from ..cycle_utils import ordered_ring, ring_system_extends_core
from ..formatting import format_prefixes as _format_prefixes
from ..formatting import join_prefix_parent as _join_prefix_parent
from ..formatting import preferred_parent_stem as _preferred_parent_stem
from ..formatting import loc_unsat
from ..formatting import _multiplied_prefix_unit_name
from ..benzimidazole_orientation import preferred_benzimidazole_n1_benzenoid_locant_map
from ..fused_template_families import (
    generated_fusion_preference as _generated_fusion_preference,
)
from ..molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set
from ..ring_resource_rules import resource_ring_parent_candidates as _resource_ring_parent_candidates
from ..ring_rules import (
    adjust_ring_parent_for_charge as _adjust_ring_parent_for_charge,
    ring_parent_candidates as _ring_parent_candidates,
)
from ..simple_prefixes import simple_external_prefix
from ..stereochemistry import _has_assigned_bond_stereo, assigned_atom_stereochemistry
from ..substituent_resource_rules import resource_substituent_match as _resource_substituent_match
from ..template_access import template_record_locants, template_record_mol, template_record_name

_OPSIN_PARSEABLE_RETAINED_PARENT_ALIASES = {
    "4-azachroman-3(4H)-one": "3,4-dihydro-2H-1,4-benzoxazin-3-one",
    "boronicacidpinacolester": "4,4,5,5-tetramethyl-1,3,2-dioxaborolane",
    "mequinol": "4-methoxyphenol",
    "paraben": "4-hydroxybenzoic acid",
    "xanthin": "purin-2,6(1H,3H,9H)-dione",
}

_ELEMENT_RADICAL_PARENT_WORDS = {
    "B": "boron",
    "C": "carbon",
    "N": "nitrogen",
    "O": "oxygen",
    "P": "phosphorus",
    "S": "sulfur",
    "Se": "selenium",
    "Si": "silicon",
}


def _opsin_parseable_retained_parent_name(name: str) -> str:
    preferred = _preferred_parent_stem(name)
    return _OPSIN_PARSEABLE_RETAINED_PARENT_ALIASES.get(preferred, preferred)

_BICYCLIC_FUSION_BASES = {
    "indole",
    "indazole",
    "benzofuran",
    "benzothiophene",
    "quinoline",
    "isoquinoline",
    "naphthyridine",
    "quinazoline",
    "quinoxaline",
    "cinnoline",
    "phthalazine",
    "purine",
    "pteridine",
}

_TERMINAL_RETAINED_PARENT_BY_STEM = {
    "aconit": "aconitane",
    "ajmal": "ajmalan",
    "androst": "androstane",
    "barbiturat": "barbituric acid",
    "barbiturate": "barbituric acid",
    "chol": "cholane",
    "cholest": "cholestane",
    "campest": "campestane",
    "cinchon": "cinchonan",
    "ergost": "ergostane",
    "estr": "estrane",
    "furost": "furostane",
    "morphin": "morphinan",
    "poriferast": "poriferastane",
    "prost": "prostane",
    "pterocarp": "pterocarpan",
    "roten": "rotenan",
    "stigmast": "stigmastane",
    "gon": "gonane",
    "gorgost": "gorgostane",
    "pregn": "pregnane",
    "spirost": "spirostane",
    "thrombox": "thromboxane",
    "trop": "tropane",
}


def _terminal_retained_parent_name(name: str) -> str:
    charged_name = _terminal_retained_charged_parent_name(name)
    if charged_name != name:
        return charged_name
    if name == "pyroglutam":
        return "pyroglutamic acid"
    if name.endswith(" acid"):
        return name
    if name.endswith("acid"):
        return f"{name[:-4]} acid"
    if name.endswith(("ic", "ylic")) and name not in _RETAINED_ACID_PARENT_STEM_EXCEPTIONS:
        return f"{name} acid"
    return _TERMINAL_RETAINED_PARENT_BY_STEM.get(name, name)


def _terminal_retained_charged_parent_name(name: str) -> str:
    match = re.fullmatch(r"(.+)-(\d+[a-z]?)-(ium|ide?|id)", name)
    if match is None:
        return name
    stem, locant, suffix = match.groups()
    parent = _TERMINAL_RETAINED_PARENT_BY_STEM.get(stem)
    if not parent:
        return name
    return f"{parent}-{locant}-{suffix}"


_RETAINED_ACID_PARENT_STEM_EXCEPTIONS = {
    "psic",
}


def _generated_exact_parent_sort_key(template: dict) -> tuple:
    name = template_record_name(template)
    base = _fusion_base_name(name)
    base_rank = 0 if base in _BICYCLIC_FUSION_BASES else 1
    if base == "imidazole":
        return (base_rank, len(name), template.get("source_family", ""), name)
    return (_generated_fusion_preference(name), base_rank, len(name), template.get("source_family", ""), name)


def _single_ring_candidate_sort_key(name: str) -> tuple:
    charged_parent_rank = 0 if re.search(r"(?:^|-)[^-]+-\d+-(?:ium|ide?|id)(?:$|-)", name) else 1
    suffix_carbonyl_parent_rank = 0 if _has_small_ring_carbonyl_suffix_parent(name) else 1
    ylidene_rank = 0 if "ylidene" in name else 1
    aromatic_parent_rank = 0 if _has_aromatic_single_ring_parent_stem(name) else 1
    functional_unsaturation_rank = 0 if _has_functional_unsaturated_substituent(name) else 1
    isotope_rank = 0 if _has_isotope_locant(name) else 1
    relative_stereo_rank = 0 if re.search(r"\((?:cis|trans)\)", name) else 1
    return (
        charged_parent_rank,
        suffix_carbonyl_parent_rank,
        ylidene_rank,
        functional_unsaturation_rank,
        isotope_rank,
        aromatic_parent_rank,
        relative_stereo_rank,
        len(name),
        name,
    )


def _has_small_ring_carbonyl_suffix_parent(name: str) -> bool:
    return "cyclobut-1-en-3,4-dione" in name or "cycloprop-1-en-3-one" in name


def _has_aromatic_single_ring_parent_stem(name: str) -> bool:
    bare_name = re.sub(r"^\([^)]+\)-", "", name)
    return bool(
        re.search(
            r"(?:^|-)(?:"
            r"benzene|pyridine|pyridin-\d+-one|pyrimidine|pyrazine|pyridazine|"
            r"pyrrole|furan|thiophene|imidazole|pyrazole|oxazole|isoxazole|"
            r"thiazole|isothiazole|triazole|tetrazole"
            r")$",
            bare_name,
        )
    )


def _has_functional_unsaturated_substituent(name: str) -> bool:
    return bool(
        re.search(
            r"(?:aza|diaza|oxa|thia)[a-z]*"
            r"(?:prop|but|pent|hex)-\d+(?:,\d+)*-(?:en|yn)e?",
            name,
        )
    )


def _has_isotope_locant(name: str) -> bool:
    return bool(
        re.search(r"(?:^|-)\d+(?:,\d+)*-\d+(?:C|N|O|S|P|F|Cl|Br|I)\d*(?:$|-)", name)
        or re.search(r"\[\d+(?:C|N|O|S|P|F|Cl|Br|I)\]", name)
    )


def _fusion_base_name(name: str) -> str:
    if "]" not in name:
        return name
    return name.rsplit("]", 1)[1]


def _partial_hydro_fusion_preference(name: str) -> tuple:
    key = _generated_fusion_preference(name)
    spiro_penalty = _partial_hydro_spiro_alias_penalty(name)
    if len(key) < 5:
        return (spiro_penalty, key)
    if len(key) == 5:
        base_rank, letter_rank, locants, component_rank, full_name = key
        return (spiro_penalty, base_rank, -letter_rank, locants, component_rank, full_name)
    orientation_rank, base_rank, letter_rank, locants, component_rank, full_name = key
    return (spiro_penalty, orientation_rank, base_rank, -letter_rank, locants, component_rank, full_name)


def _partial_hydro_spiro_alias_penalty(name: str) -> int:
    if not name.startswith("spiro["):
        return 0
    if re.search(r"-[0-9]+-en(?:e)?(?:\]|$)", name):
        return 30
    if re.search(r"(?:indan|tetralin|decalin|naphthalan)(?:\]|$)", name):
        return 20
    return 0


@lru_cache(maxsize=8)
def _curated_exact_parent_templates(Chem) -> tuple[dict, ...]:
    from ..scaffold_catalog import parent_scaffold_catalog

    return tuple(
        template
        for template in parent_scaffold_catalog(Chem).exact
        if template.get("source_family") != "generated-fused"
    )


@lru_cache(maxsize=8)
def _guanidine_parent_templates(Chem) -> tuple[dict, ...]:
    names = {
        "guanidin",
        "guanidinium",
        "biguanide",
        "triguanide",
        "tetraguanide",
        "pentaguanide",
    }
    return tuple(
        template for template in _curated_exact_parent_templates(Chem) if template_record_name(template) in names
    )


def _iter_fused_template_matches(
    ctx,
    templates: Sequence[dict],
    Chem,
    *,
    topology_query: bool = False,
    require_one_ring_system: bool = False,
    require_fully_aromatic: bool = False,
    reject_extra_core_bonds: bool = False,
):
    mol = ctx.mol
    telemetry = ctx.telemetry
    stage_counts: Counter[str] | None = Counter() if telemetry is not None else None
    try:
        for template in templates:
            if stage_counts is not None:
                stage_counts["templates"] += 1
            template_mol = template_record_mol(template)
            if template.get("stereochemical") and not _has_stereo_evidence(mol, Chem):
                continue
            if stage_counts is not None:
                stage_counts["after_stereo"] += 1
            if require_one_ring_system and not _template_has_one_ring_system(template_mol):
                continue
            if stage_counts is not None:
                stage_counts["after_one_ring_system"] += 1
            if require_fully_aromatic and not _template_is_fully_aromatic(template_mol):
                continue
            if stage_counts is not None:
                stage_counts["after_fully_aromatic"] += 1
            if topology_query:
                query = _topology_query(template_mol, Chem)
                if query.GetNumAtoms() > mol.GetNumAtoms():
                    continue
                if stage_counts is not None:
                    stage_counts["substructure_queries"] += 1
                matches = ctx.substruct_matches(query)
            else:
                if stage_counts is not None:
                    stage_counts["substructure_queries"] += 1
                matches = ctx.template_matches(template)
            if stage_counts is not None:
                stage_counts["matches"] += len(matches)
            for match in matches:
                if reject_extra_core_bonds and _match_has_extra_core_bonds(mol, template, match):
                    if stage_counts is not None:
                        stage_counts["rejected_extra_core_bonds"] += 1
                    continue
                if stage_counts is not None:
                    stage_counts["yielded_matches"] += 1
                yield template, match
    finally:
        if telemetry is not None and stage_counts:
            family = telemetry.active_rule_family or "unknown"
            for stage, count in stage_counts.items():
                telemetry.filter_stage_count[f"{family}:fused_iter:{stage}"] += count

def _match_has_extra_core_bonds(mol, template: dict, match: Sequence[int]) -> bool:
    cache = mol_cache_dict(mol, "extra_core_bonds_cache")
    cache_key = (id(template), tuple(match))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    core_atoms = set(match)
    neighbors = _atom_neighbor_indices(mol)
    core_bonds = sum(
        1
        for atom_idx in core_atoms
        for nbr_idx in neighbors[atom_idx]
        if nbr_idx in core_atoms
    ) // 2
    result = core_bonds > template_record_mol(template).GetNumBonds()
    cache[cache_key] = result
    return result


def _atom_neighbor_indices(mol) -> tuple[tuple[int, ...], ...]:
    cached = mol_cache_get(mol, "atom_neighbor_indices")
    if cached is not None:
        return cached
    neighbors = tuple(tuple(nbr.GetIdx() for nbr in atom.GetNeighbors()) for atom in mol.GetAtoms())
    return mol_cache_set(mol, "atom_neighbor_indices", neighbors)


def _name_acyclic_retained_core_match(mol, template: dict, match: Sequence[int]) -> str:
    core_set = set(match)
    prefixes = []
    for template_idx, mol_idx in enumerate(match):
        locant = template_record_locants(template).get(template_idx, "")
        for nbr in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core_set:
                continue
            if not locant:
                return ""
            if _subtree_has_cumulated_diene(mol, nidx, mol_idx):
                return ""
            prefix = simple_external_prefix(mol, nidx, mol_idx) or branch_prefix(mol, nidx, mol_idx)
            if not prefix:
                return ""
            prefixes.append((locant, prefix))
    parent_name = _terminal_retained_parent_name(template_record_name(template))
    prefix_text = _format_prefixes(prefixes)
    return _join_prefix_parent(prefix_text, parent_name)


def _subtree_has_cumulated_diene(mol, start_idx: int, blocked_idx: int) -> bool:
    seen = {blocked_idx}
    stack = [start_idx]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "C":
            double_carbon_bonds = 0
            for nbr in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if nbr.GetSymbol() == "C" and bond.GetBondTypeAsDouble() == 2.0:
                    double_carbon_bonds += 1
            if double_carbon_bonds >= 2:
                return True
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in seen:
                stack.append(nidx)
    return False


@lru_cache(maxsize=None)
def _skeletal_replacement_query(template_mol, Chem):
    query = Chem.RWMol()
    for _atom in template_mol.GetAtoms():
        query.AddAtom(Chem.AtomFromSmarts("[#5,#6,#7,#8,#14,#15,#16,#32,#33,#34,#50,#52]"))
    for bond in template_mol.GetBonds():
        query.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.UNSPECIFIED)
    return query.GetMol()


def _skeletal_replacement_bond_order_mismatch_count(mol, template: dict, match: Sequence[int]) -> int:
    template_mol = template_record_mol(template)
    mismatches = 0
    for template_bond in template_mol.GetBonds():
        begin = template_bond.GetBeginAtomIdx()
        end = template_bond.GetEndAtomIdx()
        mol_bond = mol.GetBondBetweenAtoms(match[begin], match[end])
        if mol_bond is None:
            mismatches += 4
            continue
        template_unsaturated = template_bond.GetIsAromatic() or template_bond.GetBondTypeAsDouble() > 1.0
        mol_unsaturated = mol_bond.GetIsAromatic() or mol_bond.GetBondTypeAsDouble() > 1.0
        if template_unsaturated != mol_unsaturated:
            mismatches += 1
            continue
        if template_unsaturated and template_bond.GetIsAromatic() != mol_bond.GetIsAromatic():
            mismatches += 1
    return mismatches


def _skeletal_replacement_unsaturated_hetero_bond_penalty(mol, template: dict, match: Sequence[int]) -> int:
    template_mol = template_record_mol(template)
    penalty = 0
    for template_bond in template_mol.GetBonds():
        if template_bond.GetIsAromatic() or template_bond.GetBondTypeAsDouble() <= 1.0:
            continue
        begin_atom = mol.GetAtomWithIdx(match[template_bond.GetBeginAtomIdx()])
        end_atom = mol.GetAtomWithIdx(match[template_bond.GetEndAtomIdx()])
        if begin_atom.GetSymbol() != "C" and end_atom.GetSymbol() != "C":
            penalty += 1
    return penalty


_SKELETAL_REPLACEMENT_UNSTABLE_SPIRO_LOCANT_TEMPLATES = frozenset(
    {
        "spiro[cyclohexane-1,5'-s-indacen]",
        "spiro[cyclohexane-1,5'-s-indacene]",
        "spiro[cyclopentane-1,8'-as-indacen]",
        "spiro[cyclopentane-1,8'-as-indacene]",
    }
)


def _skeletal_replacement_spiro_locant_instability_penalty(template: dict) -> int:
    """Generated spiro-fused aliases whose skeletal-replacement locants are unstable in OPSIN."""
    if template.get("source_family") != "generated-fused":
        return 0
    if template_record_name(template) in _SKELETAL_REPLACEMENT_UNSTABLE_SPIRO_LOCANT_TEMPLATES:
        return 1
    return 0


def _skeletal_replacement_parent_name(mol, template: dict, match: Sequence[int]) -> str:
    grouped: dict[str, list[str]] = {}
    charged_locs: list[str] = []
    for template_idx, mol_idx in enumerate(match):
        atom = mol.GetAtomWithIdx(mol_idx)
        symbol = atom.GetSymbol()
        if symbol == "C":
            continue
        replacement = _SKELETAL_REPLACEMENT_PREFIXES.get(symbol)
        locant = template_record_locants(template).get(template_idx, "")
        if not replacement or not locant:
            return ""
        if atom.GetFormalCharge() != 0:
            if atom.GetFormalCharge() > 0 and symbol in {"N", "O", "S"} and locant.isdigit():
                charged_locs.append(locant)
            else:
                return ""
        grouped.setdefault(replacement, []).append(locant)
    if not grouped:
        return ""
    parts = []
    for replacement in sorted(grouped):
        locants = sorted(grouped[replacement], key=_locant_sort_key)
        multiplier = MULT.get(len(locants), "")
        parts.append(f"{','.join(locants)}-{multiplier}{replacement}")
    parent_name = template_record_name(template)
    if parent_name == "gon":
        parent_name = "cyclopenta[a]phenanthren"
    return _charge_suffixed_replacement_parent_name("-".join(parts) + parent_name, charged_locs)


def _charge_suffixed_replacement_parent_name(parent_name: str, charged_locs: Sequence[str]) -> str:
    locs = sorted(set(charged_locs), key=_locant_sort_key)
    if not locs:
        return parent_name
    if len(locs) != 1:
        return ""
    stem = parent_name[:-1] if parent_name.endswith("e") else parent_name
    return f"{stem}-{locs[0]}-ium"


_RETAINED_TOPOLOGY_REPLACEMENT_NH_PARENT_NAMES = frozenset(
    {
        "corrin",
        "corrine",
        "porphin",
        "porphine",
        "porphyrin",
        "porphyrine",
    }
)


def _retained_topology_replacement_template_allows_nh(template: dict) -> bool:
    parent_name = _terminal_retained_parent_name(template_record_name(template))
    if parent_name not in _RETAINED_TOPOLOGY_REPLACEMENT_NH_PARENT_NAMES:
        return False
    template_mol = template_record_mol(template)
    if template_mol.GetNumAtoms() < 19 or len(template_mol.GetRingInfo().AtomRings()) < 4:
        return False
    return any(
        atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0
        for atom in template_mol.GetAtoms()
    )


def _retained_topology_replacement_parent_name(
    mol,
    template: dict,
    match: Sequence[int],
    *,
    allow_nh_template: bool = False,
) -> str:
    grouped: dict[str, list[str]] = {}
    charged_locs: list[str] = []
    template_mol = template_record_mol(template)
    if (
        not allow_nh_template
        and any(atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0 for atom in template_mol.GetAtoms())
    ):
        return ""
    locants = template_record_locants(template)
    for template_idx, mol_idx in enumerate(match):
        template_atom = template_mol.GetAtomWithIdx(template_idx)
        mol_atom = mol.GetAtomWithIdx(mol_idx)
        if template_atom.GetSymbol() == mol_atom.GetSymbol():
            if template_atom.GetFormalCharge() != mol_atom.GetFormalCharge():
                return ""
            continue
        if template_atom.GetSymbol() != "C" or template_atom.GetFormalCharge() != 0:
            return ""
        replacement = _SKELETAL_REPLACEMENT_PREFIXES.get(mol_atom.GetSymbol())
        locant = locants.get(template_idx, "")
        if not replacement or not locant:
            return ""
        if mol_atom.GetFormalCharge() != 0:
            if mol_atom.GetFormalCharge() > 0 and mol_atom.GetSymbol() in {"N", "O", "S"} and locant.isdigit():
                charged_locs.append(locant)
            else:
                return ""
        grouped.setdefault(replacement, []).append(locant)
    if not grouped:
        return ""
    parts = []
    for replacement in sorted(grouped):
        replacement_locants = sorted(grouped[replacement], key=_locant_sort_key)
        multiplier = MULT.get(len(replacement_locants), "")
        parts.append(f"{','.join(replacement_locants)}-{multiplier}{replacement}")
    parent_name = _terminal_retained_parent_name(template_record_name(template))
    if parent_name == "gon":
        parent_name = "cyclopenta[a]phenanthren"
    return _opsin_parseable_retained_parent_name(
        _charge_suffixed_replacement_parent_name("-".join(parts) + parent_name, charged_locs)
    )


def _skeletal_replacement_sort_key(mol, match: Sequence[int], parent_name: str) -> tuple:
    saturation_penalty = _skeletal_replacement_saturation_penalty(mol, match, parent_name)
    spiro_norbornene_penalty = _spiro_norbornene_locant_penalty(parent_name)
    replacement_count = sum(parent_name.count(term) for term in _SKELETAL_REPLACEMENT_PREFIXES.values())
    replacement_locants = _skeletal_replacement_locant_tuple(parent_name)
    replacement_prefix_locants = _skeletal_replacement_prefix_locant_key(parent_name)
    fusion_locant_penalty = parent_name.count("a-")
    return (
        saturation_penalty,
        spiro_norbornene_penalty,
        fusion_locant_penalty,
        replacement_count,
        replacement_prefix_locants,
        replacement_locants,
        len(parent_name),
        parent_name,
    )


def _skeletal_replacement_locant_tuple(parent_name: str) -> tuple[int, ...]:
    grouped = _skeletal_replacement_locants_by_prefix(parent_name)
    locants: list[int] = []
    for prefix_locants in grouped.values():
        locants.extend(prefix_locants)
    return tuple(sorted(locants)) or (999,)


def _skeletal_replacement_prefix_locant_key(parent_name: str) -> tuple[tuple[int, ...], ...]:
    grouped = _skeletal_replacement_locants_by_prefix(parent_name)
    prefix_order = ("aza", "oxa", "thia", "phospha", "selena", "tellura", "bora", "sila", "germa", "stanna", "arsa")
    return tuple(tuple(sorted(grouped.get(prefix, (999,)))) for prefix in prefix_order)


def _skeletal_replacement_locants_by_prefix(parent_name: str) -> dict[str, list[int]]:
    replacement_terms = tuple(sorted(_SKELETAL_REPLACEMENT_PREFIXES.values(), key=len, reverse=True))
    replacement_pattern = "|".join(re.escape(term) for term in replacement_terms)
    grouped: dict[str, list[int]] = {}
    for match in re.finditer(rf"(?<![\d^])(\d+(?:,\d+)*)-([a-z]*?(?:{replacement_pattern}))", parent_name):
        prefix_text = match.group(2)
        replacement = next((term for term in replacement_terms if prefix_text.endswith(term)), "")
        if not replacement:
            continue
        grouped.setdefault(replacement, []).extend(int(locant) for locant in match.group(1).split(","))
    return grouped


def _spiro_norbornene_locant_penalty(parent_name: str) -> int:
    if "spiro[" not in parent_name or "norbornen" not in parent_name:
        return 0
    return 1 if re.search(r",3'-norbornen", parent_name) else 0


def _skeletal_replacement_saturation_penalty(mol, match: Sequence[int], parent_name: str) -> int:
    core_atoms = set(match)
    has_unsaturation = False
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() not in core_atoms or bond.GetEndAtomIdx() not in core_atoms:
            continue
        if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1.0:
            has_unsaturation = True
            break
    if not has_unsaturation:
        return 0
    if "indan" in parent_name:
        return 2
    if "naphthalen" in parent_name and "dihydronaphthalen" not in parent_name:
        return 2
    saturated_terms = (
        "decalin",
        "norbornan",
        "bicyclo",
        "tricyclo",
    )
    unsaturated_terms = (
        "naphthalen",
        "naphthalene",
        "naphthaline",
        "tetralin",
        "inden",
        "indene",
        "fluoren",
        "fluorene",
        "norbornen",
        "norbornene",
    )
    if any(term in parent_name for term in unsaturated_terms):
        return 0
    if any(term in parent_name for term in saturated_terms):
        return 2
    return 0


def _has_nonaromatic_core_imine(mol, core_atoms: set[int]) -> bool:
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() not in core_atoms or bond.GetEndAtomIdx() not in core_atoms:
            continue
        if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() != 2.0:
            continue
        symbols = {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}
        if symbols == {"C", "N"}:
            return True
    return False


def _locant_sort_key(locant: str) -> tuple[int, str]:
    digits = "".join(ch for ch in locant if ch.isdigit())
    return (int(digits) if digits else 999, locant)


_SKELETAL_REPLACEMENT_PREFIXES = {
    "As": "arsa",
    "B": "bora",
    "Ge": "germa",
    "N": "aza",
    "O": "oxa",
    "P": "phospha",
    "S": "thia",
    "Se": "selena",
    "Si": "sila",
    "Sn": "stanna",
    "Te": "tellura",
}


def _has_stereo_evidence(mol, Chem) -> bool:
    cached = mol_cache_get(mol, "has_stereo_evidence")
    if cached is not None:
        return cached
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return mol_cache_set(mol, "has_stereo_evidence", True)
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return mol_cache_set(mol, "has_stereo_evidence", True)
    return mol_cache_set(mol, "has_stereo_evidence", False)


@lru_cache(maxsize=None)
def _topology_query(template_mol, Chem):
    query = Chem.RWMol()
    for atom in template_mol.GetAtoms():
        query.AddAtom(Chem.AtomFromSmarts(f"[#{atom.GetAtomicNum()}]"))
    for bond in template_mol.GetBonds():
        query.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.UNSPECIFIED)
    return query.GetMol()


@lru_cache(maxsize=None)
def _template_has_one_ring_system(template_mol) -> bool:
    rings = [set(ring) for ring in template_mol.GetRingInfo().AtomRings()]
    if len(rings) < 2:
        return False
    current = rings[0]
    remaining = rings[1:]
    changed = True
    while changed:
        changed = False
        rest = []
        for ring in remaining:
            if current & ring:
                current |= ring
                changed = True
            else:
                rest.append(ring)
        remaining = rest
    return not remaining


@lru_cache(maxsize=None)
def _template_is_fully_aromatic(template_mol) -> bool:
    return all(atom.GetIsAromatic() for atom in template_mol.GetAtoms())


def _hydro_parent_name(mol, template: dict, match: Sequence[int]) -> str:
    hydro_locs = []
    charged_locs = []
    charged_has_hydrogen = False
    for template_idx, mol_idx in enumerate(match):
        template_atom = template_record_mol(template).GetAtomWithIdx(template_idx)
        atom = mol.GetAtomWithIdx(mol_idx)
        charge_matches = template_atom.GetFormalCharge() == atom.GetFormalCharge()
        charge_is_parent_ium = (
            template_atom.GetFormalCharge() == 0
            and atom.GetFormalCharge() > 0
            and atom.GetSymbol() in {"N", "O", "S"}
        )
        if not charge_matches and not charge_is_parent_ium:
            return ""
        locant = template_record_locants(template).get(template_idx, "")
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
        if not _is_hydro_locant(locant):
            return ""
        hydro_locs.append(locant)
    if not hydro_locs:
        return ""
    hydro_locs = sorted(set(hydro_locs), key=_locant_sort_key)
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
    parent_name = _hydro_indicated_hydrogen_template_name(mol, template, match)
    name = f"{locants}-{hydro}-{parent_name}"
    if len(charged_locs) == 1:
        if not charged_has_hydrogen:
            name = _drop_indicated_hydrogen_prefix(name)
        stem = name[:-1] if name.endswith("e") else name
        return f"{stem}-{charged_locs[0]}-ium"
    return name


def _hydro_indicated_hydrogen_template_name(mol, template: dict, match: Sequence[int]) -> str:
    name = template_record_name(template)
    if not _supports_hydro_fused_indicated_hydrogen(name):
        return name
    if _match_has_terminal_chalcogen_double_bond(mol, match):
        return name
    reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
    h_locs = _matched_fused_indicated_hydrogen_locants(mol, template, reverse)
    if len(h_locs) != 1:
        return name
    return f"{h_locs[0]}H-{name}"


def _supports_hydro_fused_indicated_hydrogen(name: str) -> bool:
    if name.startswith("indeno["):
        return False
    if re.search(r"(?:benzene|pyrrole)$", name):
        return False
    return _supports_generated_fused_indicated_hydrogen(name)


def _match_has_terminal_chalcogen_double_bond(mol, match: Sequence[int]) -> bool:
    core = set(match)
    for idx in match:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx in core or neighbor.GetAtomicNum() not in {8, 16, 34, 52}:
                continue
            if neighbor.GetDegree() != 1:
                continue
            bond = mol.GetBondBetweenAtoms(idx, nidx)
            if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
                return True
    return False


def _is_hydro_locant(locant: str) -> bool:
    return bool(locant) and locant[0].isdigit() and not locant.endswith("a")


def _partial_hydro_parent_names(mol, template: dict, match: Sequence[int]):
    candidate_locs = []
    scored_hydro_atoms = 0
    scored_hydro_total = 0
    parent_name = _hydro_indicated_hydrogen_template_name(mol, template, match)
    for template_idx, mol_idx in enumerate(match):
        template_atom = template_record_mol(template).GetAtomWithIdx(template_idx)
        atom = mol.GetAtomWithIdx(mol_idx)
        if template_atom.GetFormalCharge() != atom.GetFormalCharge():
            return
        locant = template_record_locants(template).get(template_idx, "")
        score = _partial_hydro_atom_score(mol, template_record_mol(template), mol_idx, template_idx, match)
        if score <= 0:
            continue
        scored_hydro_atoms += 1
        scored_hydro_total += score
        if not locant or locant.endswith("a"):
            continue
        candidate_locs.append((locant, score))
    by_locant = {}
    for locant, score in candidate_locs:
        by_locant[locant] = max(score, by_locant.get(locant, 0))
    candidate_locs = sorted(by_locant.items(), key=lambda item: _locant_sort_key(item[0]))
    candidates = []
    if template_record_name(template).startswith("spiro["):
        for count, hydro in ((2, "dihydro"), (4, "tetrahydro"), (6, "hexahydro")):
            if scored_hydro_atoms >= count:
                candidates.append(((2, count, -scored_hydro_total, 0, ()), f"{hydro}{parent_name}"))
    if len(candidate_locs) < 2:
        for hydro_key, parent_name in sorted(candidates)[:12]:
            yield parent_name, hydro_key
        return
    for count, hydro in ((2, "dihydro"), (4, "tetrahydro"), (6, "hexahydro")):
        if len(candidate_locs) < count:
            continue
        combos = sorted(
            combinations(candidate_locs, count),
            key=lambda items: (
                -sum(score for _locant, score in items),
                -_hydro_locant_span([locant for locant, _score in items]),
                [_locant_sort_key(locant) for locant, _score in items],
            ),
        )
        for items in combos:
            locs = [locant for locant, _score in items]
            locant_text = ",".join(locs)
            full_even_set = count == len(candidate_locs) and count in {2, 4, 6, 8}
            candidates.append((
                (
                    0 if full_even_set else 1,
                    count,
                    -sum(score for _locant, score in items),
                    -_hydro_locant_span(locs),
                    [_locant_sort_key(locant) for locant in locs],
                ),
                f"{locant_text}-{hydro}-{parent_name}",
            ))
    for hydro_key, parent_name in sorted(candidates)[:12]:
        yield parent_name, hydro_key


def _hydro_locant_span(locants: Sequence[str]) -> int:
    values = []
    for locant in locants:
        digits = "".join(ch for ch in locant if ch.isdigit())
        if digits:
            values.append(int(digits))
    return max(values) - min(values) if values else 0


def _partial_hydro_atom_score(mol, template_mol, mol_idx: int, template_idx: int, match: Sequence[int]) -> int:
    atom = mol.GetAtomWithIdx(mol_idx)
    template_atom = template_mol.GetAtomWithIdx(template_idx)
    if atom.GetIsAromatic() or _has_unsaturated_core_bond(mol, mol_idx, match):
        return 0
    score = 0
    if atom.GetTotalNumHs() > template_atom.GetTotalNumHs():
        score += atom.GetTotalNumHs() - template_atom.GetTotalNumHs()
    if template_atom.GetIsAromatic() or _has_unsaturated_template_bond(template_mol, template_idx):
        score += 1
    return score


def _has_unsaturated_template_bond(template_mol, template_idx: int) -> bool:
    atom = template_mol.GetAtomWithIdx(template_idx)
    for nbr in atom.GetNeighbors():
        bond = template_mol.GetBondBetweenAtoms(template_idx, nbr.GetIdx())
        if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1.0:
            return True
    return False


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


def _hydrogenated_template_penalty(name: str) -> int:
    return 1 if "hydro" in name else 0


def _template_source_rank(template: dict) -> int:
    family = template.get("source_family", "")
    if not family:
        return 0
    if family.startswith("arylGroups") or family.startswith("simpleCyclicGroups"):
        return 1
    if family == "static-retained":
        return 2
    if family.startswith("fusionComponents"):
        return 3
    if family == "generated-fused":
        return 4
    return 5


def _fused_candidate_preference(name: str) -> tuple:
    oxo_locs = tuple(int(loc) for loc in re.findall(r"(?<!\d)(\d+)-oxo", name))
    return (
        1 if re.search(r"(?:^|-|,)\d+[a-z]-", name) else 0,
        1 if "ind-1-ene" in name else 0,
        _fused_parent_carbonyl_rank(name),
        _benzimidazole_orientation_preference(name),
        _greek_functional_locant_preference(name),
        _primed_functional_locant_preference(name),
        _substituent_locant_tuple(name),
        oxo_locs or (99,),
    )


def _fused_parent_carbonyl_rank(name: str) -> int:
    if _has_carboxyl_like_oxo_locant(name):
        return 1
    if name.endswith("one"):
        return 0
    if re.search(r"(?:^|-)\d+(?:,\d+)*-(?:di|tri|tetra)?oxo-", name):
        return 0
    return 1


def _has_carboxyl_like_oxo_locant(name: str) -> bool:
    oxo_locants = _functional_locants(name, "oxo")
    if not oxo_locants:
        return False
    hetero_locants = _functional_locants(name, "hydroxy") | _functional_locants(name, "amino")
    return bool(oxo_locants & hetero_locants)


def _functional_locants(name: str, term: str) -> set[int]:
    locants: set[int] = set()
    pattern = rf"(?<![\d^])(\d+(?:,\d+)*)-(?:di|tri|tetra)?{term}"
    for locant_text in re.findall(pattern, name):
        locants.update(int(locant) for locant in locant_text.split(",") if locant)
    return locants


def _greek_functional_locant_preference(name: str) -> tuple[int, ...]:
    locant_rank = {"alpha": 0, "alpha'": 1, "beta": 2, "beta'": 3}
    prefix_rank = {
        "amino": 0,
        "hydroxy": 1,
        "mercapto": 2,
        "oxo": 3,
        "thioxo": 4,
    }
    pairs = []
    for locant, prefix in re.findall(r"(alpha'?|beta'?)-(amino|hydroxy|mercapto|oxo|thioxo)", name):
        if locant in locant_rank:
            pairs.append((prefix_rank[prefix], locant_rank[locant]))
    if len(pairs) < 2:
        return ()
    return tuple(locant for _prefix, locant in sorted(pairs))


def _primed_functional_locant_preference(name: str) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    prefix_rank = {
        "hydroxy": 0,
        "amino": 1,
        "mercapto": 2,
        "oxo": 3,
        "thioxo": 4,
        "oxy": 5,
        "sulfanyl": 6,
    }
    grouped: dict[str, list[tuple[int, int]]] = {}
    for locant_text, multiplier, prefix in re.findall(
        r"((?:\d+'*)(?:,\d+'*)*)-(?:(di|tri|tetra|penta|hexa))?"
        r"(hydroxy|amino|mercapto|oxo|thioxo|oxy|sulfanyl)(?:-|$)",
        name,
    ):
        for locant in locant_text.split(","):
            grouped.setdefault(prefix, []).append(_primed_locant_rank(locant))
    if not grouped:
        return ()
    return tuple(
        (prefix_rank[prefix], tuple(sorted(locants)))
        for prefix, locants in sorted(grouped.items(), key=lambda item: prefix_rank[item[0]])
    )


def _primed_locant_rank(locant: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)('*?)", locant)
    if not match:
        return (999, 999)
    return (int(match.group(1)), len(match.group(2)))


def _substituent_locant_tuple(name: str) -> tuple[int, ...]:
    locants = [
        int(locant)
        for locant in re.findall(r"(?<![\d^])(\d+)(?=[a-z]?[-,])", name)
        if int(locant) < 100
    ]
    return tuple(sorted(locants)) or (99,)


def _benzimidazole_orientation_preference(name: str) -> tuple:
    if "benzimidazole" not in name or "oxo" in name:
        return ()
    if "carboxamide" in name and "carbonyl" in name and re.search(r"(?:^|-)N(?:,N)?-", name):
        locants = sorted(
            {int(loc) for loc in re.findall(r"(?<!\d)(\d+)-carboxamide", name) if int(loc) >= 4}
        )
        return tuple(locants)
    locants = sorted(
        {int(loc) for loc in re.findall(r"(?<!\d)(\d+)-", name) if int(loc) >= 4}
    )
    return tuple(locants)


def _benzimidazole_template_h_preference(mol, template: dict, match: Sequence[int], name: str) -> tuple:
    template_name = template_record_name(template)
    if (
        "benzimidazole" not in name
        and "benzimidazol" not in name
        and template_name not in {"benzimidazole", "1H-benzimidazole"}
    ):
        return ()
    reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
    h_locs = sorted(_matched_fused_indicated_hydrogen_locants(mol, template, reverse), key=_locant_sort_key)
    if template_name == "benzimidazole":
        if h_locs == ["3"]:
            return (0,)
        if h_locs == ["1"]:
            return (1,)
    if h_locs == ["1"]:
        return (0,)
    if h_locs:
        return (1, tuple(h_locs))
    return (2,)


def _benzimidazole_external_n_preference(mol, template: dict, match: Sequence[int]) -> tuple:
    if template_record_name(template) != "1H-benzimidazole":
        return ()
    template_mol = template_record_mol(template)
    locants = template_record_locants(template)
    core = set(match)
    external_n_locants = []
    for template_idx, mol_idx in enumerate(match):
        if template_mol.GetAtomWithIdx(template_idx).GetSymbol() != "N":
            continue
        atom = mol.GetAtomWithIdx(mol_idx)
        if not any(nbr.GetIdx() not in core and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
            continue
        locant = locants.get(template_idx, "")
        if locant:
            external_n_locants.append(locant)
    if not external_n_locants:
        return (1,)
    locants_key = tuple(sorted(external_n_locants, key=_locant_sort_key))
    if locants_key == ("1",):
        return (0,)
    if locants_key == ("3",):
        return (2,)
    return (1, tuple(_locant_sort_key(locant) for locant in locants_key))


def _generated_benzimidazole_external_n_preference(mol, template: dict, match: Sequence[int]) -> tuple:
    if not _is_generated_benzimidazole_template(template):
        return ()
    external_n_locants = _matched_external_n_locants(mol, template, match)
    if not external_n_locants:
        return (1,)
    locants_key = tuple(sorted(external_n_locants, key=_locant_sort_key))
    if locants_key == ("3",):
        return (0,)
    if locants_key == ("1",):
        return (2,)
    return (1, tuple(_locant_sort_key(locant) for locant in locants_key))


_GENERATED_FUSION_NAME_RE = re.compile(r"^(.+)\[([^]]+)-([a-z])\](.+)$")


def _generated_fused_graph_orientation_preference(mol, template: dict, match: Sequence[int]) -> tuple:
    name = template_record_name(template)
    base_key = _generated_fusion_preference(name)
    indazole_key = _generated_indazole_h_locant_fusion_preference(mol, template, match, base_key)
    if indazole_key != base_key:
        return indazole_key
    graph_orientation_rank = _generated_fused_graph_orientation_rank(mol, name, match)
    if graph_orientation_rank is None or len(base_key) != 6:
        return base_key
    _orientation_rank, base_rank, letter_rank, locants, component_rank, full_name = base_key
    return (graph_orientation_rank, base_rank, letter_rank, locants, component_rank, full_name)


def _generated_indazole_h_locant_fusion_preference(
    mol,
    template: dict,
    match: Sequence[int],
    base_key: tuple,
) -> tuple:
    name = template_record_name(template)
    if name not in {"benzo[1,2-c]pyrazole", "benzo[1,2-d]pyrazole"}:
        return base_key
    if len(base_key) != 6:
        return base_key
    reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
    h_locs = sorted(_matched_fused_indicated_hydrogen_locants(mol, template, reverse), key=_locant_sort_key)
    if len(h_locs) != 1:
        return base_key
    preferred_name = ""
    if h_locs[0] == "1":
        preferred_name = "benzo[1,2-d]pyrazole"
    elif h_locs[0] == "2":
        preferred_name = "benzo[1,2-c]pyrazole"
    if not preferred_name:
        return base_key
    orientation_rank, base_rank, _letter_rank, locants, component_rank, full_name = base_key
    h_letter_rank = 0 if name == preferred_name else 1
    return (orientation_rank, base_rank, h_letter_rank, locants, component_rank, full_name)


def _generated_fused_graph_orientation_rank(mol, name: str, match: Sequence[int]) -> int | None:
    parsed = _GENERATED_FUSION_NAME_RE.match(name)
    if parsed is None:
        return None
    component, _locants, _letter, base = parsed.groups()
    if _matched_formal_positive_atom_count(mol, match):
        if component.startswith(("pyridino", "pyrido")) and base == "imidazole":
            return 0
        if component.startswith("imidazo") and base == "pyridine":
            return 1
    return None


def _matched_formal_positive_atom_count(mol, match: Sequence[int]) -> int:
    return sum(1 for idx in match if mol.GetAtomWithIdx(int(idx)).GetFormalCharge() > 0)


def _is_generated_benzimidazole_template(template: dict) -> bool:
    if template.get("source_family") != "generated-fused":
        return False
    template_mol = template_record_mol(template)
    locants = template_record_locants(template)
    n_locants = {
        str(locants.get(atom.GetIdx(), ""))
        for atom in template_mol.GetAtoms()
        if atom.GetSymbol() == "N"
    }
    if n_locants != {"1", "3"}:
        return False
    atom_symbols = Counter(atom.GetSymbol() for atom in template_mol.GetAtoms())
    return atom_symbols["N"] == 2 and atom_symbols["C"] == 7


def _matched_external_n_locants(mol, template: dict, match: Sequence[int]) -> list[str]:
    template_mol = template_record_mol(template)
    locants = template_record_locants(template)
    core = set(match)
    external_n_locants = []
    for template_idx, mol_idx in enumerate(match):
        if template_mol.GetAtomWithIdx(template_idx).GetSymbol() != "N":
            continue
        atom = mol.GetAtomWithIdx(mol_idx)
        if not any(nbr.GetIdx() not in core and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
            continue
        locant = locants.get(template_idx, "")
        if locant:
            external_n_locants.append(str(locant))
    return external_n_locants


def _benzimidazole_external_carbon_locant_preference(mol, template: dict, match: Sequence[int]) -> tuple:
    if template_record_name(template) != "1H-benzimidazole":
        return ()
    template_mol = template_record_mol(template)
    locants = template_record_locants(template)
    core = set(match)
    external_locs: list[int] = []
    for template_idx, mol_idx in enumerate(match):
        if template_mol.GetAtomWithIdx(template_idx).GetSymbol() == "N":
            continue
        atom = mol.GetAtomWithIdx(mol_idx)
        if not any(nbr.GetIdx() not in core and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
            continue
        locant = locants.get(template_idx, "")
        if locant.isdigit() and int(locant) >= 4:
            external_locs.append(int(locant))
    return tuple(sorted(external_locs)) or (99,)


def _benzotriazole_template_h_preference(mol, template: dict, match: Sequence[int], name: str) -> tuple:
    if "benzotriazole" not in name and "benzotriazole" not in template_record_name(template):
        return ()
    reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
    h_locs = sorted(_matched_fused_indicated_hydrogen_locants(mol, template, reverse), key=_locant_sort_key)
    if h_locs == ["1"]:
        return (0,)
    if h_locs == ["2"]:
        return (1,)
    if h_locs:
        return (2, tuple(h_locs))
    return (3,)


def _has_n_oxide_on_match(mol, match: Sequence[int]) -> bool:
    core = set(match)
    for mol_idx in match:
        atom = mol.GetAtomWithIdx(mol_idx)
        if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
            continue
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core:
                continue
            bond = mol.GetBondBetweenAtoms(mol_idx, nidx)
            if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() < 0 and bond.GetBondTypeAsDouble() == 1.0:
                return True
    return False


def _name_fused_core_match(
    mol, template: dict, match: Sequence[int], parent_name: str | None = None
) -> str:
    cache = mol_cache_dict(mol, "fused_core_match_name_cache")
    cache_key = (
        id(template),
        tuple(match),
        parent_name,
        mol.HasProp("_nispo_allow_fragment_substituent_fallback"),
        mol.GetProp("_nispo_fragment_substituent_depth") if mol.HasProp("_nispo_fragment_substituent_depth") else "",
    )
    if cache_key in cache:
        return cache[cache_key]
    core_set = set(match)
    if ring_system_extends_core(mol, core_set):
        cache[cache_key] = ""
        return ""
    if _match_has_unlocanted_external_substitution(mol, template, match, core_set=core_set):
        cache[cache_key] = ""
        return ""
    reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
    parent_name = _terminal_retained_parent_name(parent_name or _adjust_fused_parent_for_charge(mol, template, reverse))
    parent_name = _template_isotope_prefixed_parent_name(mol, template, match, parent_name)
    parent_name = _opsin_parseable_retained_parent_name(parent_name)
    radical_parent_name = _fused_single_external_element_radical_name(mol, template, match, reverse, parent_name)
    if radical_parent_name:
        cache[cache_key] = radical_parent_name
        return radical_parent_name
    prefixes = []
    prefix_records = []
    template_mol = template_record_mol(template)
    for mol_idx in match:
        template_idx = reverse[mol_idx]
        locant = template_record_locants(template).get(template_idx)
        for nbr in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core_set:
                continue
            if not locant:
                cache[cache_key] = ""
                return ""
            prefix = simple_external_prefix(mol, nidx, mol_idx) or branch_prefix(mol, nidx, mol_idx)
            if not prefix:
                cache[cache_key] = ""
                return ""
            prefix_records.append((locant, prefix, template_mol.GetAtomWithIdx(template_idx).GetSymbol()))
    if prefix_records:
        locant_map = _benzimidazole_n1_benzenoid_prefix_locant_map(template, prefix_records)
        prefixes = [(locant_map.get(locant, locant), prefix) for locant, prefix, _symbol in prefix_records]
    if not prefixes:
        result = _join_prefix_parent("", parent_name)
        cache[cache_key] = result
        return result
    indicated_h_locs = _matched_fused_indicated_hydrogen_locants(mol, template, reverse)
    purin_amine_name = _fused_purin_6_amine_name(prefixes, parent_name)
    if purin_amine_name:
        cache[cache_key] = purin_amine_name
        return purin_amine_name
    purin_thioxo_name = _fused_purin_indicated_thioxo_name(prefixes, parent_name)
    if purin_thioxo_name:
        cache[cache_key] = purin_thioxo_name
        return purin_thioxo_name
    suffix_name = (
        _fused_functional_suffix_name(prefixes, parent_name, indicated_h_locs)
        if _may_have_fused_functional_suffix(prefixes)
        else ""
    )
    if suffix_name:
        cache[cache_key] = suffix_name
        return suffix_name
    parent_name = _drop_indicated_hydrogen_segments(parent_name, prefixes)
    suffix_name = (
        _fused_thione_suffix_name(prefixes, parent_name, indicated_h_locs)
        if _may_have_fused_thione_suffix(prefixes)
        else ""
    )
    if suffix_name:
        cache[cache_key] = suffix_name
        return suffix_name
    suffix_name = (
        _fused_oxo_suffix_name(prefixes, parent_name, indicated_h_locs)
        if _may_have_fused_oxo_suffix(prefixes)
        else ""
    )
    if suffix_name:
        cache[cache_key] = suffix_name
        return suffix_name
    suffix_name = (
        _fused_imine_suffix_name(prefixes, parent_name, indicated_h_locs)
        if _may_have_fused_imine_suffix(prefixes)
        else ""
    )
    if suffix_name:
        cache[cache_key] = suffix_name
        return suffix_name
    prefix_text = _format_prefixes(prefixes)
    if not prefix_text:
        cache[cache_key] = ""
        return ""
    result = _join_prefix_parent(prefix_text, parent_name)
    cache[cache_key] = result
    return result


def _benzimidazole_n1_benzenoid_prefix_locant_map(
    template: dict,
    prefix_records: Sequence[tuple[str, str, str]],
) -> dict[str, str]:
    if template_record_name(template) != "1H-benzimidazole":
        return {}
    if not any(locant == "1" and symbol == "N" for locant, _prefix, symbol in prefix_records):
        return {}
    carbon_locants = [locant for locant, _prefix, symbol in prefix_records if symbol != "N"]
    return preferred_benzimidazole_n1_benzenoid_locant_map(carbon_locants)


def _fused_single_external_element_radical_name(
    mol,
    template: dict,
    match: Sequence[int],
    reverse: dict[int, int],
    parent_name: str,
) -> str:
    core_set = set(match)
    external: list[tuple[str, int, int]] = []
    locants = template_record_locants(template)
    for mol_idx in match:
        template_idx = reverse[mol_idx]
        locant = locants.get(template_idx, "")
        for nbr in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core_set:
                continue
            external.append((locant, nidx, mol_idx))
    if len(external) != 1:
        return ""
    locant, radical_idx, parent_idx = external[0]
    if not locant:
        return ""
    radical = mol.GetAtomWithIdx(radical_idx)
    element = _ELEMENT_RADICAL_PARENT_WORDS.get(radical.GetSymbol())
    if not element:
        return ""
    if not _is_terminal_element_radical_substituent(mol, radical_idx, parent_idx):
        return ""
    return f"{parent_name}-{locant}-yl{element}"


def _is_terminal_element_radical_substituent(mol, atom_idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetFormalCharge() != 0 or atom.GetIsotope():
        return False
    if atom.GetNumRadicalElectrons() <= 0:
        return False
    if atom.GetTotalNumHs() != 0:
        return False
    if atom.IsInRing():
        return False
    bond = mol.GetBondBetweenAtoms(atom_idx, parent_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return False
    return not any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors())


def _fused_purin_6_amine_name(prefixes: Sequence[tuple[str, str]], parent_name: str) -> str:
    parent = _strip_leading_indicated_hydrogen(parent_name)
    if parent not in {"purin", "purine"}:
        return ""
    amine_prefixes = [(loc, prefix) for loc, prefix in prefixes if loc == "6" and prefix.endswith("amino")]
    if len(amine_prefixes) != 1:
        return ""
    amine_loc, amine_prefix = amine_prefixes[0]
    if amine_prefix == "amino":
        amine_core = "9H-purin-6-amine"
    else:
        group = amine_prefix.removesuffix("amino")
        if not group:
            return ""
        amine_core = f"{_format_amide_n_substituent(group)}-9H-purin-6-amine"
    remaining = [(loc, prefix) for loc, prefix in prefixes if (loc, prefix) != (amine_loc, amine_prefix)]
    if any(loc in {"1", "3", "7", "9"} for loc, _prefix in remaining):
        return ""
    prefix_text = _format_prefixes(remaining)
    return _join_prefix_parent(prefix_text, amine_core) if prefix_text else amine_core


def _fused_purin_indicated_thioxo_name(prefixes: Sequence[tuple[str, str]], parent_name: str) -> str:
    parent = _strip_leading_indicated_hydrogen(parent_name)
    if parent not in {"purin", "purine"}:
        return ""
    if any(prefix == "oxo" for _loc, prefix in prefixes):
        return ""
    thioxo_locs = tuple(
        sorted((loc for loc, prefix in prefixes if prefix == "thioxo" and loc.isdigit()), key=int)
    )
    h_locs = _purin_thioxo_indicated_h_locants(thioxo_locs)
    if not h_locs:
        return ""
    remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "thioxo"]
    if any(loc in {"1", "3", "7", "9"} for loc, _prefix in remaining):
        return ""
    prefix_text = _format_prefixes(remaining)
    locants = ",".join(thioxo_locs)
    multiplier = MULT.get(len(thioxo_locs), "")
    h_text = ",".join(f"{loc}H" for loc in h_locs)
    core = f"{locants}-{multiplier}thioxo-{h_text}-{parent}"
    return _join_prefix_parent(prefix_text, core) if prefix_text else core


def _purin_thioxo_indicated_h_locants(thioxo_locs: Sequence[str]) -> tuple[str, ...]:
    locant_set = set(thioxo_locs)
    if locant_set == {"2", "6"}:
        return ("3",)
    if locant_set == {"6", "8"}:
        return ("3", "9")
    if locant_set == {"2", "6", "8"}:
        return ("3", "7")
    return ()


def _match_has_unlocanted_external_substitution(
    mol,
    template: dict,
    match: Sequence[int],
    *,
    core_set: set[int] | None = None,
) -> bool:
    locants = template_record_locants(template)
    core = core_set if core_set is not None else set(match)
    for template_idx, mol_idx in enumerate(match):
        if locants.get(template_idx):
            continue
        atom = mol.GetAtomWithIdx(mol_idx)
        if any(nbr.GetIdx() not in core for nbr in atom.GetNeighbors()):
            return True
    return False


def _template_isotope_prefixed_parent_name(mol, template: dict, match: Sequence[int], parent_name: str) -> str:
    if not _molecule_has_isotope(mol):
        return parent_name
    grouped: dict[tuple[str, int], list[str]] = {}
    locants = template_record_locants(template)
    for template_idx, mol_idx in enumerate(match):
        atom = mol.GetAtomWithIdx(mol_idx)
        isotope = atom.GetIsotope()
        if not isotope:
            continue
        symbol = atom.GetSymbol()
        if f"{isotope}{symbol}" in parent_name or f"[{isotope}{symbol}]" in parent_name:
            continue
        locant = locants.get(template_idx, "")
        if not locant:
            return parent_name
        grouped.setdefault((symbol, isotope), []).append(locant)
    prefixes = []
    for (symbol, isotope), isotope_locants in sorted(grouped.items()):
        ordered = sorted(set(isotope_locants), key=_locant_sort_key)
        count_text = str(len(ordered)) if len(ordered) > 1 else ""
        prefixes.append(f"({','.join(ordered)}-{isotope}{symbol}{count_text})")
    return "".join(prefixes) + parent_name if prefixes else parent_name


def _molecule_has_isotope(mol) -> bool:
    cached = mol_cache_get(mol, "has_isotope")
    if cached is not None:
        return cached
    return mol_cache_set(mol, "has_isotope", any(atom.GetIsotope() for atom in mol.GetAtoms()))


def _name_natural_product_topology_match(mol, Chem, template: dict, match: Sequence[int]) -> tuple[str, str] | None:
    parent_name = _natural_product_topology_parent_name(mol, Chem, template, match)
    if not parent_name:
        return None
    name = _name_fused_core_match(mol, template, match, parent_name=parent_name)
    if not name:
        return None
    return name, _locanted_template_stereo_prefix(mol, Chem, template, match)


def _natural_product_topology_parent_name(mol, Chem, template: dict, match: Sequence[int]) -> str:
    parent = template_record_name(template)
    ene_locs = _natural_product_extra_double_bond_locs(mol, Chem, template, match)
    if ene_locs:
        return _format_natural_product_unsaturated_parent(parent, ene_locs)
    return parent + "ane"


def _format_natural_product_unsaturated_parent(parent: str, ene_locs: Sequence[str]) -> str:
    locs = sorted(set(ene_locs), key=_natural_product_unsaturation_sort_key)
    if len(locs) == 1:
        return f"{parent}-{locs[0]}-ene"
    return f"{parent}a-{','.join(locs)}-{MULT.get(len(locs), '')}ene"


def _natural_product_unsaturation_sort_key(locant: str) -> tuple[int, str]:
    digits = "".join(ch for ch in locant if ch.isdigit())
    return (int(digits) if digits else 0, locant)


def _natural_product_extra_double_bond_locs(mol, Chem, template: dict, match: Sequence[int]) -> list[str]:
    locants = template_record_locants(template)
    template_mol = template_record_mol(template)
    ene_locs, aromatic_bond_keys = _natural_product_aromatic_unsaturations(mol, template, match)
    for bond in template_mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        mol_begin = match[begin]
        mol_end = match[end]
        if frozenset((mol_begin, mol_end)) in aromatic_bond_keys:
            continue
        mol_bond = mol.GetBondBetweenAtoms(mol_begin, mol_end)
        if mol_bond is None:
            continue
        if mol_bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.DOUBLE:
            continue
        numeric = [
            int(locants[idx])
            for idx in (begin, end)
            if locants.get(idx, "").isdigit()
        ]
        if numeric:
            ene_locs.add(_natural_product_double_bond_locant(numeric))
    return sorted(ene_locs, key=_natural_product_unsaturation_sort_key)


def _natural_product_aromatic_unsaturations(
    mol, template: dict, match: Sequence[int]
) -> tuple[set[str], set[frozenset[int]]]:
    reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
    locants = template_record_locants(template)
    ene_locs: set[str] = set()
    aromatic_bond_keys: set[frozenset[int]] = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_set = set(ring)
        if not ring_set <= set(reverse):
            continue
        ring_bonds = [
            mol.GetBondBetweenAtoms(ring[idx], ring[(idx + 1) % len(ring)])
            for idx in range(len(ring))
        ]
        if not ring_bonds or not all(bond is not None and bond.GetIsAromatic() for bond in ring_bonds):
            continue
        ring_locants = {locants.get(reverse[idx], "") for idx in ring}
        if ring_locants == {"1", "2", "3", "4", "5", "10"}:
            ene_locs.update({"1", "3", "5(10)"})
            aromatic_bond_keys.update(frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) for bond in ring_bonds)
    return ene_locs, aromatic_bond_keys


def _natural_product_double_bond_locant(numeric_locants: Sequence[int]) -> str:
    pair = frozenset(numeric_locants)
    special = {
        frozenset((5, 10)): "5(10)",
        frozenset((8, 14)): "8(14)",
        frozenset((9, 11)): "9(11)",
        frozenset((13, 17)): "13(17)",
    }.get(pair)
    if special:
        return special
    return str(min(numeric_locants))


def _locanted_template_stereo_prefix(mol, Chem, template: dict, match: Sequence[int]) -> str:
    atom_descriptors = _fresh_atom_stereochemistry(mol, Chem)
    locants = template_record_locants(template)
    pieces = []
    seen = set()
    for template_idx, mol_idx in enumerate(match):
        descriptor = atom_descriptors.get(mol_idx)
        locant = locants.get(template_idx, "")
        if descriptor is None or not locant.isdigit():
            continue
        text = f"{locant}{descriptor}"
        if text in seen:
            continue
        seen.add(text)
        pieces.append(((int(locant), locant), text))
    for locant, descriptor in _locanted_template_bond_stereo(mol, Chem, template, match):
        text = f"{locant}{descriptor}"
        if text in seen:
            continue
        seen.add(text)
        pieces.append((_natural_product_unsaturation_sort_key(locant), text))
    return ",".join(text for _sort_key, text in sorted(pieces))


def _locanted_template_bond_stereo(mol, Chem, template: dict, match: Sequence[int]) -> list[tuple[str, str]]:
    locants = template_record_locants(template)
    descriptors = []
    for bond in template_record_mol(template).GetBonds():
        mol_bond = mol.GetBondBetweenAtoms(match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()])
        if mol_bond is None or mol_bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        stereo = mol_bond.GetStereo()
        if stereo in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}:
            descriptor = "E"
        elif stereo in {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}:
            descriptor = "Z"
        else:
            continue
        numeric = [
            int(locants[idx])
            for idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            if locants.get(idx, "").isdigit()
        ]
        if not numeric:
            continue
        locant = _natural_product_double_bond_locant(numeric)
        if "(" in locant:
            continue
        descriptors.append((locant, descriptor))
    return descriptors


def _fresh_atom_stereochemistry(mol, Chem) -> dict[int, str]:
    cached = mol_cache_get(mol, "fresh_atom_stereochemistry")
    if cached is not None:
        return cached
    result = {
        idx: descriptor
        for idx, descriptor in _fresh_all_atom_stereochemistry(mol, Chem).items()
        if descriptor in {"R", "S"}
    }
    return mol_cache_set(mol, "fresh_atom_stereochemistry", result)


def _fresh_all_atom_stereochemistry(mol, Chem) -> dict[int, str]:
    cached = mol_cache_get(mol, "fresh_all_atom_stereochemistry")
    if cached is not None:
        return cached
    if not any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        return mol_cache_set(mol, "fresh_all_atom_stereochemistry", {})
    work = Chem.Mol(mol)
    Chem.AssignStereochemistry(work, cleanIt=True, force=True)
    result = {
        int(idx): str(descriptor)
        for idx, descriptor in Chem.FindMolChiralCenters(
            work,
            includeUnassigned=False,
            useLegacyImplementation=False,
        )
        if descriptor in {"R", "S", "r", "s"}
    }
    return mol_cache_set(mol, "fresh_all_atom_stereochemistry", result)


def _fused_oxo_suffix_name(
    prefixes: Sequence[tuple[str, str]],
    parent_name: str,
    indicated_h_locs: Sequence[str] = (),
) -> str:
    oxo_locs = sorted((loc for loc, prefix in prefixes if prefix == "oxo" and loc.isdigit()), key=int)
    if parent_name == "pteridine" and oxo_locs == ["4", "7"]:
        remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "oxo"]
        prefix_text = _format_prefixes(remaining)
        name = "3H,8H-pteridine-4,7-dione"
        return _join_prefix_parent(prefix_text, name) if prefix_text else _preferred_parent_stem(name)
    if (
        len(oxo_locs) > 1
        and indicated_h_locs
        and not _has_fused_hydro_parent_segment(parent_name)
        and "acenaphthen" not in parent_name
        and "naphthyridine" not in parent_name
        and "isoquinoline" not in parent_name
        and "spiro[" not in parent_name
    ):
        remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "oxo"]
        prefix_text = _format_prefixes(remaining)
        parent = _strip_leading_indicated_hydrogen(parent_name)
        h_text = ",".join(f"{loc}H" for loc in sorted(set(indicated_h_locs), key=_locant_sort_key))
        locants = ",".join(oxo_locs)
        suffix = f"{MULT.get(len(oxo_locs), '')}one"
        name = f"{parent}-{locants}({h_text})-{suffix}"
        return _join_prefix_parent(prefix_text, name) if prefix_text else _preferred_parent_stem(name)
    if (
        len(oxo_locs) == 1
        and indicated_h_locs
        and "hydro-" not in parent_name
        and "acenaphthen" not in parent_name
        and "naphthyridine" not in parent_name
        and "isoquinoline" not in parent_name
        and "spiro[" not in parent_name
    ):
        remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "oxo"]
        prefix_text = _format_prefixes(remaining)
        parent = _strip_leading_indicated_hydrogen(parent_name)
        indicated_locs = tuple(sorted(set(indicated_h_locs), key=_locant_sort_key))
        if len(indicated_locs) >= 2 and not _fused_oxo_suffix_parent_uses_unindicated_lactam_grammar(parent):
            locant_text = ",".join(indicated_locs)
            hydro_prefix = f"{locant_text}-{MULT.get(len(indicated_locs), '')}hydro"
            name = f"{hydro_prefix}-{oxo_locs[0]}H-{parent}-{oxo_locs[0]}-one"
            return _join_prefix_parent(prefix_text, name) if prefix_text else _preferred_parent_stem(name)
        if _fused_oxo_suffix_parent_uses_unindicated_lactam_grammar(parent):
            name = f"{parent}-{oxo_locs[0]}-one"
        else:
            leading_h_locs, suffix_h_loc = _fused_oxo_suffix_indicated_h_locants(indicated_h_locs)
            leading = f"{','.join(f'{loc}H' for loc in leading_h_locs)}-" if leading_h_locs else ""
            name = f"{leading}{parent}-{oxo_locs[0]}({suffix_h_loc}H)-one"
        return _join_prefix_parent(prefix_text, name) if prefix_text else _preferred_parent_stem(name)
    return ""


def _fused_thione_suffix_name(
    prefixes: Sequence[tuple[str, str]],
    parent_name: str,
    indicated_h_locs: Sequence[str] = (),
) -> str:
    thioxo_locs = sorted((loc for loc, prefix in prefixes if prefix == "thioxo" and loc.isdigit()), key=int)
    if (
        len(thioxo_locs) != 1
        or any(prefix == "oxo" for _loc, prefix in prefixes)
        or not indicated_h_locs
        or _has_fused_hydro_parent_segment(parent_name)
        or "acenaphthen" in parent_name
        or "naphthyridine" in parent_name
        or "isoquinoline" in parent_name
        or "spiro[" in parent_name
    ):
        return ""
    remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "thioxo"]
    prefix_text = _format_prefixes(remaining)
    parent = _strip_leading_indicated_hydrogen(parent_name)
    leading_h_locs, suffix_h_loc = _fused_oxo_suffix_indicated_h_locants(indicated_h_locs)
    leading = f"{','.join(f'{loc}H' for loc in leading_h_locs)}-" if leading_h_locs else ""
    name = f"{leading}{parent}-{thioxo_locs[0]}({suffix_h_loc}H)-thione"
    return _join_prefix_parent(prefix_text, name) if prefix_text else _preferred_parent_stem(name)


def _fused_imine_suffix_name(
    prefixes: Sequence[tuple[str, str]],
    parent_name: str,
    indicated_h_locs: Sequence[str] = (),
) -> str:
    imino_locs = sorted(
        (loc for loc, prefix in prefixes if prefix == "imino" and re.fullmatch(r"\d+[a-z]?", loc)),
        key=_locant_sort_key,
    )
    if (
        len(imino_locs) != 1
        or any(prefix in {"oxo", "thioxo"} for _loc, prefix in prefixes)
        or not indicated_h_locs
        or _has_fused_hydro_parent_segment(parent_name)
        or not _fused_imine_suffix_parent_supported(parent_name)
    ):
        return ""
    remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "imino"]
    prefix_text = _format_prefixes(remaining)
    parent = _strip_leading_indicated_hydrogen(parent_name)
    leading_h_locs, suffix_h_loc = _fused_oxo_suffix_indicated_h_locants(indicated_h_locs)
    leading = f"{','.join(f'{loc}H' for loc in leading_h_locs)}-" if leading_h_locs else ""
    name = f"{leading}{parent}-{imino_locs[0]}({suffix_h_loc}H)-imine"
    return _join_prefix_parent(prefix_text, name) if prefix_text else name


def _fused_imine_suffix_parent_supported(parent_name: str) -> bool:
    parent = _strip_leading_indicated_hydrogen(parent_name)
    if any(blocked in parent for blocked in ("acenaphthen", "spiro[")):
        return False
    return bool(
        "purin" in parent
        or "pteridin" in parent
        or re.search(r"[a-z]+o\[[^\]]+\][a-z]+", parent)
    )


_FUSED_OXO_UNINDICATED_LACTAM_PARENT_STEMS = (
    "benzazepan",
    "morphinan",
)


def _fused_oxo_suffix_parent_uses_unindicated_lactam_grammar(parent_name: str) -> bool:
    return any(stem in parent_name for stem in _FUSED_OXO_UNINDICATED_LACTAM_PARENT_STEMS)


def _has_fused_hydro_parent_segment(parent_name: str) -> bool:
    return bool(re.search(r"(?:^|-)(?:di|tri|tetra|penta|hexa|hepta|octa)?hydro", parent_name))


def _fused_oxo_suffix_indicated_h_locants(indicated_h_locs: Sequence[str]) -> tuple[tuple[str, ...], str]:
    locs = tuple(sorted(set(indicated_h_locs), key=_locant_sort_key))
    if len(locs) <= 1:
        return (), locs[0]
    if "1" in locs:
        suffix_h_loc = next((loc for loc in locs if loc != "1"), locs[0])
        leading_h_locs = tuple(loc for loc in locs if loc != suffix_h_loc)
        return leading_h_locs, suffix_h_loc
    return (), locs[0]


def _adjust_fused_parent_for_charge(mol, template: dict, reverse: dict[int, int]) -> str:
    name = _adjust_fused_parent_for_indicated_hydrogen(mol, template, reverse)
    if name in {"quinuclidine", "quinuclidin"}:
        for mol_idx, template_idx in reverse.items():
            locant = template_record_locants(template).get(template_idx)
            atom = mol.GetAtomWithIdx(mol_idx)
            if locant == "1" and atom.GetSymbol() == "N" and atom.GetFormalCharge() > 0:
                return "quinuclidin-1-ium"
    cation_locs = []
    cation_suffix = ""
    charged_has_hydrogen = False
    anion_locs = []
    template_mol = template_record_mol(template)
    for mol_idx, template_idx in reverse.items():
        locant = template_record_locants(template).get(template_idx)
        atom = mol.GetAtomWithIdx(mol_idx)
        template_atom = template_mol.GetAtomWithIdx(template_idx)
        if (
            locant
            and locant.isdigit()
            and atom.GetFormalCharge() < 0
            and template_atom.GetFormalCharge() == 0
        ):
            anion_locs.append(int(locant))
        if locant and locant.isdigit() and atom.GetFormalCharge() > 0:
            if (
                atom.GetSymbol() == template_atom.GetSymbol()
                and atom.GetFormalCharge() == template_atom.GetFormalCharge()
            ):
                continue
            symbol = atom.GetSymbol()
            if symbol == "C":
                suffix = "ylium"
            elif symbol in {"N", "O", "S"}:
                suffix = "ium"
            else:
                continue
            cation_locs.append(int(locant))
            if cation_suffix and cation_suffix != suffix:
                return name
            cation_suffix = suffix
            if symbol in {"N", "O"} and atom.GetTotalNumHs() > 0:
                charged_has_hydrogen = True
    if len(anion_locs) == 1 and not cation_locs:
        stem = name[:-1] if name.endswith("e") else name
        return f"{stem}-{anion_locs[0]}-ide"
    if len(cation_locs) == 1:
        locant = cation_locs[0]
        fused_n_h_cation = (
            _fused_aromatic_n_h_cation_parent_name(name)
            if charged_has_hydrogen and cation_suffix == "ium"
            else ""
        )
        if fused_n_h_cation:
            return fused_n_h_cation
        if not charged_has_hydrogen:
            name = _drop_indicated_hydrogen_prefix(name)
        stem = name[:-1] if name.endswith("e") else name
        return f"{stem}-{locant}-{cation_suffix}"
    return name


def _fused_aromatic_n_h_cation_parent_name(name: str) -> str:
    base = _drop_indicated_hydrogen_prefix(name)
    if base == "benzotriazole":
        return "benzotriazolium"
    return ""


def _drop_indicated_hydrogen_prefix(name: str) -> str:
    if len(name) > 3 and name[0].isdigit() and name[1:3] == "H-":
        return name[3:]
    return name


def _drop_indicated_hydrogen_segments(name: str, prefixes: Sequence[tuple[str, str]]) -> str:
    if "benzimidazole" in name and any(prefix == "oxo" for _, prefix in prefixes):
        return name
    if _preserve_tautomer_sensitive_fused_indicated_hydrogen(name):
        return name
    if _preserve_generated_fused_indicated_hydrogen(name):
        return name

    def replacement(match: re.Match) -> str:
        locant = match.group(2)
        if locant == "2" and "indazole" in name:
            return match.group(0)
        if locant == "9" and "purine" in name:
            return match.group(0)
        return match.group(1)

    return re.sub(r"(^|-)(\d+)H-", replacement, name)


def _preserve_tautomer_sensitive_fused_indicated_hydrogen(name: str) -> bool:
    h_locs = re.findall(r"(?:^|-)(\d+)H-", name)
    if not any(loc != "1" for loc in h_locs):
        return False
    return "benzotriazol" in name


def _adjust_fused_parent_for_indicated_hydrogen(mol, template: dict, reverse: dict[int, int]) -> str:
    name = template_record_name(template)
    if name in {"indazole", "purine"}:
        h_locs = _matched_fused_indicated_hydrogen_locants(mol, template, reverse)
        if len(h_locs) != 1:
            return name
        return f"{h_locs[0]}H-{name}"
    if not name.startswith("1H-"):
        if not _supports_generated_fused_indicated_hydrogen(name):
            return name
        h_locs = _matched_fused_indicated_hydrogen_locants(mol, template, reverse)
        if len(h_locs) != 1:
            return name
        return f"{h_locs[0]}H-{name}"
    h_locs = _matched_fused_indicated_hydrogen_locants(mol, template, reverse)
    if len(h_locs) != 1 or h_locs[0] == "1":
        return name
    return f"{h_locs[0]}H-{name[3:]}"


def _matched_fused_indicated_hydrogen_locants(mol, template: dict, reverse: dict[int, int]) -> list[str]:
    h_locs = []
    for mol_idx, template_idx in reverse.items():
        atom = mol.GetAtomWithIdx(mol_idx)
        locant = template_record_locants(template).get(template_idx)
        if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0 and _is_indicated_h_locant(locant):
            h_locs.append(locant)
    return h_locs


def _is_indicated_h_locant(locant: str | None) -> bool:
    return bool(locant and re.match(r"\d+[a-z]?$", locant))


def _strip_leading_indicated_hydrogen(name: str) -> str:
    return re.sub(r"^(?:\d+[a-z]?H,)*\d+[a-z]?H-", "", name)


def _supports_generated_fused_indicated_hydrogen(name: str) -> bool:
    if "[" not in name or "]" not in name:
        return False
    if not re.match(r"^[a-z]+(?:o|ino)\[[^\]]+\][a-z]", name):
        return False
    return bool(
        re.search(
            r"(?:benzene|pyrrole|pyrazole|imidazol(?:e)?|triazol(?:e)?|"
            r"tetrazol(?:e)?|oxazol(?:e)?|thiazol(?:e)?|pyridin(?:e)?|"
            r"pyrimidin(?:e)?|pyridazin(?:e)?|pyrazin(?:e)?|indazole|"
            r"purin(?:e)?|quinolin(?:e)?|isoquinolin(?:e)?|"
            r"quinazolin(?:e)?|quinoxalin(?:e)?)$",
            name,
        )
    )


def _preserve_generated_fused_indicated_hydrogen(name: str) -> bool:
    match = re.search(r"(?:^|-)\d+[a-z]?H-(.+)", name)
    return bool(match and _supports_generated_fused_indicated_hydrogen(match.group(1)))


@dataclass(frozen=True, slots=True)
class RingNameRecord:
    name: str
    stereo_prefix: str
    locant_seq: tuple[int, ...]
    suffix_kind: str = ""
    suffix_graph_class: str = ""
    retained_carbohydrate_substituent_count: int = 0
    suffix_hydrogen_locants: tuple[str, ...] = ()


def _named_ring_instance_names(mol, ring: Sequence[int], Chem) -> list[tuple[str, str, tuple[int, ...]]]:
    return [
        (record.name, record.stereo_prefix, record.locant_seq)
        for record in _named_ring_instance_records(mol, ring, Chem)
    ]


def _named_ring_instance_records(mol, ring: Sequence[int], Chem) -> list[RingNameRecord]:
    cache = mol_cache_dict(mol, "named_ring_instance_records")
    cache_key = (
        frozenset(int(idx) for idx in ring),
        mol.HasProp("_nispo_allow_fragment_substituent_fallback"),
        mol.GetProp("_nispo_fragment_substituent_depth") if mol.HasProp("_nispo_fragment_substituent_depth") else "",
        mol.HasProp("_nispo_fragment_substituent_view"),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return list(cached)
    ordered = ordered_ring(mol, ring)
    if not ordered:
        return _store_named_ring_instance_records(cache, cache_key, [])
    candidates = _ring_parent_candidates(mol, ordered)
    candidates = _merge_ring_parent_candidates(
        candidates,
        _resource_ring_parent_candidates(mol, ordered, Chem),
    )
    if not candidates:
        return _store_named_ring_instance_records(cache, cache_key, [])
    ring_set = set(ordered)
    external_subs: list[tuple[int, int, str]] = []
    raw_subs = []
    retained_carbohydrate_substituent_count = 0
    for ring_idx in ordered:
        atom = mol.GetAtomWithIdx(ring_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set:
                continue
            prefix = simple_external_prefix(mol, nidx, ring_idx) or branch_prefix(mol, nidx, ring_idx)
            if not prefix:
                return _store_named_ring_instance_records(cache, cache_key, [])
            external_subs.append((ring_idx, nidx, prefix))
            raw_subs.append((ring_idx, prefix))
            if _is_retained_carbohydrate_substituent(mol, nidx, ring_idx, Chem):
                retained_carbohydrate_substituent_count += 1

    best_key = None
    best_name = ""
    best_seq: Sequence[int] = ()
    best_suffix_kind = ""
    for parent_name, seq in candidates:
        adjusted_parent = _adjust_ring_parent_for_charge(mol, parent_name, seq, raw_subs)
        if not adjusted_parent:
            continue
        adjusted_parent = _append_ring_parent_isotope_suffixes(mol, adjusted_parent, seq)
        adjusted_parent = _opsin_parseable_retained_parent_name(adjusted_parent)
        if not raw_subs:
            name = _join_prefix_parent("", adjusted_parent)
            key = ([], name)
            if best_key is None or key < best_key:
                best_key = key
                best_name = name
                best_seq = seq
                best_suffix_kind = ""
            continue
        pos = {idx: i + 1 for i, idx in enumerate(seq)}
        radical_name = _single_ring_external_element_radical_name(mol, adjusted_parent, pos, external_subs)
        if radical_name:
            locants = sorted(pos[idx] for idx, _, _ in external_subs if idx in pos)
            radical_key = (locants, -2, radical_name)
            if best_key is None or radical_key < best_key:
                best_key = radical_key
                best_name = radical_name
                best_seq = seq
                best_suffix_kind = ""
        prefixes = [
            (_single_ring_substituent_locant(adjusted_parent, pos[idx]), prefix)
            for idx, prefix in raw_subs
            if not (
                adjusted_parent in {"1,3,4-dioxazol-2-one", "azet-2-one"} and pos[idx] == 2 and prefix == "oxo"
            )
            and not (
                adjusted_parent in {"maleimide", "succinimide"} and pos[idx] in {2, 5} and prefix == "oxo"
            )
        ]
        indicated_h_locs = _single_ring_indicated_hydrogen_locants(mol, seq, pos)
        locants = sorted(pos[idx] for idx, _ in raw_subs)
        if not prefixes:
            name = _join_prefix_parent("", adjusted_parent)
            key = (locants, 0, name)
            if best_key is None or key < best_key:
                best_key = key
                best_name = name
                best_seq = seq
                best_suffix_kind = ""
            continue
        prefix_text = _format_prefixes(prefixes)
        if not prefix_text:
            continue
        if not _single_ring_parent_prefixes_supported(adjusted_parent, prefixes):
            continue
        name = _join_prefix_parent(prefix_text, adjusted_parent)
        key = (locants, 1, name)
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
            best_seq = seq
            best_suffix_kind = ""
        retained_benzene_name = _benzene_retained_parent_name(prefixes, adjusted_parent)
        if retained_benzene_name:
            retained_key = (locants, -1, len(retained_benzene_name), retained_benzene_name)
            if best_key is None or retained_key < best_key:
                best_key = retained_key
                best_name = retained_benzene_name
                best_seq = seq
                best_suffix_kind = ""
        suffix_name = (
            ""
            if adjusted_parent in {"1,3,4-dioxazol-2-one", "azet-2-one"}
            else _single_ring_oxo_suffix_name(prefixes, adjusted_parent, indicated_h_locs)
        )
        sulfur_oxide_name = _single_ring_sulfur_oxide_suffix_name(mol, raw_subs, pos, prefixes, adjusted_parent)
        if sulfur_oxide_name:
            sulfur_oxide_key = (locants, -1, sulfur_oxide_name)
            if best_key is None or sulfur_oxide_key < best_key:
                best_key = sulfur_oxide_key
                best_name = sulfur_oxide_name
                best_seq = seq
                best_suffix_kind = "ring_sulfur_oxide_suffix"
        thione_s_oxide_name = _single_ring_thione_s_oxide_suffix_name(
            mol,
            raw_subs,
            pos,
            prefixes,
            adjusted_parent,
            indicated_h_locs,
        )
        if thione_s_oxide_name:
            thione_s_oxide_key = (locants, 0, thione_s_oxide_name)
            if best_key is None or thione_s_oxide_key < best_key:
                best_key = thione_s_oxide_key
                best_name = thione_s_oxide_name
                best_seq = seq
                best_suffix_kind = "ring_thione_suffix"
        if suffix_name:
            suffix_key = (locants, 0, suffix_name)
            if best_key is None or suffix_key < best_key:
                best_key = suffix_key
                best_name = suffix_name
                best_seq = seq
                best_suffix_kind = "ring_oxo_suffix"
        thione_name = _single_ring_thione_suffix_name(prefixes, adjusted_parent, indicated_h_locs)
        if thione_name:
            thione_key = (locants, 0, thione_name)
            if best_key is None or thione_key < best_key:
                best_key = thione_key
                best_name = thione_name
                best_seq = seq
                best_suffix_kind = "ring_thione_suffix"
        imino_name = _single_ring_imino_suffix_name(prefixes, adjusted_parent)
        if imino_name:
            imino_key = (locants, 0, imino_name)
            if best_key is None or imino_key < best_key:
                best_key = imino_key
                best_name = imino_name
                best_seq = seq
                best_suffix_kind = "ring_imino_suffix"
        carbonyl_name = _single_ring_oxo_methylidene_suffix_name(prefixes, adjusted_parent)
        if carbonyl_name:
            carbonyl_key = (locants, 0, carbonyl_name)
            if best_key is None or carbonyl_key < best_key:
                best_key = carbonyl_key
                best_name = carbonyl_name
                best_seq = seq
                best_suffix_kind = "ring_carbonyl_suffix"
        functional_suffix_name = _single_ring_functional_suffix_name(prefixes, adjusted_parent)
        if functional_suffix_name:
            suffix_key = (locants, 0, functional_suffix_name)
            if best_key is None or suffix_key < best_key:
                best_key = suffix_key
                best_name = functional_suffix_name
                best_seq = seq
                best_suffix_kind = "ring_functional_suffix"
    if not best_name:
        return _store_named_ring_instance_records(cache, cache_key, [])
    if not (
        best_suffix_kind in {"ring_oxo_suffix", "ring_thione_suffix"}
        and _is_single_ring_indicated_h_chalcogen_suffix_full_form(best_name)
    ):
        best_name = _opsin_parseable_retained_parent_name(best_name)
    best_name = _drop_redundant_intrinsic_single_ring_charge_locant(mol, best_seq, best_name)
    suffix_hydrogen_locants = ()
    if best_suffix_kind == "ring_sulfur_oxide_suffix":
        pos = {idx: i + 1 for i, idx in enumerate(best_seq)}
        suffix_hydrogen_locants = _single_ring_sulfur_oxide_hydrogen_locants(mol, raw_subs, pos)
    records = [
        RingNameRecord(
            best_name,
            _single_ring_stereo_prefix(mol, best_seq, Chem, name=best_name),
            tuple(best_seq),
            best_suffix_kind,
            _single_ring_suffix_graph_class(mol, best_seq, best_suffix_kind),
            retained_carbohydrate_substituent_count,
            suffix_hydrogen_locants,
        )
    ]
    return _store_named_ring_instance_records(cache, cache_key, records)


def _store_named_ring_instance_records(cache: dict, cache_key: tuple, records: list[RingNameRecord]) -> list[RingNameRecord]:
    cache[cache_key] = tuple(records)
    return records


def _is_retained_carbohydrate_substituent(mol, substituent_idx: int, parent_idx: int, Chem) -> bool:
    template = _resource_substituent_match(mol, substituent_idx, parent_idx, Chem)
    if _is_retained_carbohydrate_substituent_template(template):
        return True
    substituent_atom = mol.GetAtomWithIdx(substituent_idx)
    if substituent_atom.GetAtomicNum() not in {7, 8, 16}:
        return False
    if substituent_atom.GetFormalCharge() != 0:
        return False
    for neighbor in substituent_atom.GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        if neighbor_idx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(substituent_idx, neighbor_idx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        template = _resource_substituent_match(mol, neighbor_idx, substituent_idx, Chem)
        if _is_retained_carbohydrate_substituent_template(template):
            return True
    return False


def _is_retained_carbohydrate_substituent_template(template: dict | None) -> bool:
    if not template or template.get("source_file") != "carbohydrates.xml":
        return False
    name = template_record_name(template)
    return name.endswith(("pyranosyl", "furanosyl", "osyl", "aminyl", "muramyl", "neuraminyl"))


def _drop_redundant_intrinsic_single_ring_charge_locant(mol, seq: Sequence[int], name: str) -> str:
    if not seq or "ium" not in name:
        return name
    locants = {idx: str(pos) for pos, idx in enumerate(seq, start=1)}
    for idx in seq:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() <= 0:
            continue
        locant = locants[idx]
        if atom.GetSymbol() == "O":
            name = name.replace(f"pyrylium-{locant}-ium", "pyrylium")
        elif atom.GetSymbol() == "S":
            for parent in ("thiopyrylium", "thiopyranium", "thiophenium"):
                name = name.replace(f"{parent}-{locant}-ium", parent)
    return name


def _single_ring_suffix_graph_class(mol, seq: Sequence[int], suffix_kind: str) -> str:
    if suffix_kind not in {"ring_oxo_suffix", "ring_thione_suffix"}:
        return ""
    if len(seq) > 5:
        return ""
    hetero_atoms = sum(
        1
        for idx in seq
        if mol.GetAtomWithIdx(idx).GetAtomicNum() not in {1, 6}
    )
    if hetero_atoms < 2:
        return ""
    return "compact_hetero_ring_chalcogen_suffix_parent"


def _single_ring_external_element_radical_name(
    mol,
    parent_name: str,
    pos: dict[int, int],
    external_subs: Sequence[tuple[int, int, str]],
) -> str:
    radical_subs = [
        (ring_idx, atom_idx, prefix)
        for ring_idx, atom_idx, prefix in external_subs
        if ring_idx in pos and _is_terminal_element_radical_substituent(mol, atom_idx, ring_idx)
    ]
    if len(radical_subs) != 1:
        return ""
    ring_idx, radical_idx, _ = radical_subs[0]
    radical = mol.GetAtomWithIdx(radical_idx)
    element = _ELEMENT_RADICAL_PARENT_WORDS.get(radical.GetSymbol())
    if not element:
        return ""
    radical_locant = _single_ring_substituent_locant(parent_name, pos[ring_idx])
    if not radical_locant.isdigit():
        return ""
    other_prefixes = [
        (_single_ring_substituent_locant(parent_name, pos[other_ring_idx]), prefix)
        for other_ring_idx, other_atom_idx, prefix in external_subs
        if other_atom_idx != radical_idx and other_ring_idx in pos
    ]
    if other_prefixes and not _single_ring_parent_prefixes_supported(parent_name, other_prefixes):
        return ""
    radical_parent = f"{parent_name}-{radical_locant}-yl{element}"
    prefix_text = _format_prefixes(other_prefixes)
    return _join_prefix_parent(prefix_text, radical_parent)


def _single_ring_substituent_locant(parent_name: str, loc: int) -> str:
    if parent_name in {"maleimide", "succinimide"} and loc == 1:
        return "N"
    return str(loc)


def _single_ring_indicated_hydrogen_locants(mol, seq: Sequence[int], pos: dict[int, int]) -> tuple[str, ...]:
    locants = []
    for atom_idx in seq:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0 and atom_idx in pos:
            locants.append(str(pos[atom_idx]))
    return tuple(sorted(set(locants), key=_locant_sort_key))


def _single_ring_parent_prefixes_supported(parent_name: str, prefixes: Sequence[tuple[str, str]]) -> bool:
    if parent_name not in {"maleimide", "succinimide"}:
        return True
    n_prefixes = [prefix for loc, prefix in prefixes if loc == "N"]
    if len(n_prefixes) > 1:
        return False
    if n_prefixes:
        n_prefix = n_prefixes[0]
        if n_prefix.startswith("(") or len(n_prefix) > 24:
            return False
        carbon_prefixes = [
            prefix
            for loc, prefix in prefixes
            if loc != "N" and not (loc in {"2", "5"} and prefix == "oxo")
        ]
        if len(carbon_prefixes) > 1:
            return False
    return True


def _merge_ring_parent_candidates(
    base_candidates: Sequence[tuple[str, tuple[int, ...]]],
    resource_candidates: Sequence[tuple[str, tuple[int, ...]]],
) -> list[tuple[str, tuple[int, ...]]]:
    merged: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    base_by_seq: dict[tuple[int, ...], set[str]] = {}
    for name, seq in base_candidates:
        seq_key = tuple(seq)
        base_by_seq.setdefault(seq_key, set()).add(name)
        base_by_seq[seq_key].add(_drop_indicated_hydrogen_prefix(name))
    for name, seq in tuple(resource_candidates) + tuple(base_candidates):
        key = (name, tuple(seq))
        if key in seen:
            continue
        if _resource_ring_alias_shadowed_by_base(name, tuple(seq), base_by_seq):
            continue
        seen.add(key)
        merged.append(key)
    return merged


def _resource_ring_alias_shadowed_by_base(
    name: str,
    seq: tuple[int, ...],
    base_by_seq: dict[tuple[int, ...], set[str]],
) -> bool:
    base_names = base_by_seq.get(seq, set())
    if not base_names:
        return False
    return name + "e" in base_names or _RESOURCE_RING_ALIAS_EQUIVALENTS.get(name, "") in base_names


_RESOURCE_RING_ALIAS_EQUIVALENTS = {
    "benzol": "benzene",
}


def _append_ring_parent_isotope_suffixes(mol, parent: str, seq: Sequence[int]) -> str:
    grouped: dict[tuple[str, int], list[int]] = {}
    for locant, idx in enumerate(seq, start=1):
        atom = mol.GetAtomWithIdx(idx)
        isotope = atom.GetIsotope()
        if not isotope:
            continue
        grouped.setdefault((atom.GetSymbol(), isotope), []).append(locant)
    if not grouped:
        return parent
    suffixes = []
    for (symbol, isotope), locants in sorted(grouped.items()):
        count_text = str(len(locants)) if len(locants) > 1 else ""
        suffixes.append(f"{','.join(str(locant) for locant in locants)}-{isotope}{symbol}{count_text}")
    return parent + "-" + "-".join(suffixes)


_BENZENE_RETAINED_PARENT_BY_PREFIX = {
    "1-azamethylcarbonyl": "benzamide",
    "aminosulfonyl": "benzenesulfonamide",
    "amino": "aniline",
    "carbamyl": "benzamide",
    "carbamoyl": "benzamide",
    "carboxy": "benzoic acid",
    "carboxylato": "benzoate",
    "cyano": "benzonitrile",
    "formyl": "benzaldehyde",
    "hydroxy": "phenol",
    "methoxy": "anisole",
    "methyl": "toluene",
}


_BENZENE_RETAINED_PARENT_RANK = {
    "carboxy": 0,
    "carboxylato": 1,
    "1-azamethylcarbonyl": 2,
    "carbamyl": 2,
    "carbamoyl": 2,
    "cyano": 3,
    "formyl": 4,
    "aminosulfonyl": 5,
    "hydroxy": 6,
    "amino": 7,
    "methoxy": 8,
    "methyl": 9,
}


def _benzene_retained_parent_name(prefixes: Sequence[tuple[str, str]], parent_name: str) -> str:
    if parent_name != "benzene":
        return ""
    best_key = None
    best_name = ""
    original_prefix_text = _format_prefixes(prefixes)
    original_name = original_prefix_text + "-benzene" if original_prefix_text else "benzene"
    for index, (loc, prefix) in enumerate(prefixes):
        retained_parent = _BENZENE_RETAINED_PARENT_BY_PREFIX.get(prefix)
        if not retained_parent:
            continue
        if loc != "1" and prefix not in _BENZENE_RETAINED_PARENT_REMAP_PREFIXES:
            continue
        for remapped_remaining in _benzene_remaining_prefix_orientations(prefixes, index, loc):
            prefix_text = _format_prefixes(remapped_remaining)
            name = prefix_text + retained_parent if prefix_text else retained_parent
            name = _benzene_retained_parent_alias_name(name)
            if len(name) >= len(original_name):
                continue
            rank = _BENZENE_RETAINED_PARENT_RANK[prefix]
            key = (rank, len(name), _substituent_locant_tuple(name), name)
            if best_key is None or key < best_key:
                best_key = key
                best_name = name
    return best_name


_BENZENE_RETAINED_PARENT_REMAP_PREFIXES = {
    "1-azamethylcarbonyl",
    "carbamyl",
    "carbamoyl",
    "carboxy",
    "carboxylato",
    "cyano",
    "formyl",
}


def _benzene_remaining_prefix_orientations(
    prefixes: Sequence[tuple[str, str]],
    retained_index: int,
    retained_loc: str,
) -> list[list[tuple[str, str]]]:
    if not retained_loc.isdigit():
        return []
    retained = int(retained_loc)
    if not 1 <= retained <= 6:
        return []
    remaining = list(prefixes[:retained_index]) + list(prefixes[retained_index + 1 :])
    if retained == 1:
        return [remaining]
    orientations = []
    for direction in (1, -1):
        remapped = []
        ok = True
        for loc, prefix in remaining:
            if not loc.isdigit():
                ok = False
                break
            value = int(loc)
            if not 1 <= value <= 6:
                ok = False
                break
            mapped = ((direction * (value - retained)) % 6) + 1
            remapped.append((str(mapped), prefix))
        if ok:
            orientations.append(remapped)
    return orientations


def _benzene_retained_parent_alias_name(name: str) -> str:
    replacements = (
        ("2-hydroxyphenol", "benzene-1,2-diol"),
        ("3-hydroxyphenol", "benzene-1,3-diol"),
        ("4-hydroxyphenol", "benzene-1,4-diol"),
        ("2-aminophenol", "2-aminophenol"),
        ("3-aminophenol", "3-aminophenol"),
        ("4-aminophenol", "4-aminophenol"),
        ("2-methylphenol", "o-cresol"),
        ("3-methylphenol", "m-cresol"),
        ("4-methylphenol", "p-cresol"),
        ("2-methoxyphenol", "guaiacol"),
        ("4-methoxyphenol", "mequinol"),
    )
    for verbose, alias in replacements:
        if alias in _PREFIX_SENSITIVE_BENZENE_RETAINED_ALIASES and name != verbose:
            continue
        if "formyl" in name and alias in {"guaiacol", "mequinol", "o-cresol", "m-cresol", "p-cresol"}:
            continue
        name = name.replace(verbose, alias)
    return name


_PREFIX_SENSITIVE_BENZENE_RETAINED_ALIASES = frozenset(
    {
        "guaiacol",
        "mequinol",
        "o-cresol",
        "m-cresol",
        "p-cresol",
    }
)


def _single_ring_stereo_prefix(mol, seq: Sequence[int], Chem, name: str = "") -> str:
    if not seq:
        return ""
    if all(mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 for idx in seq):
        absolute_prefix = _single_carbocycle_locanted_absolute_prefix(mol, seq, Chem, name=name)
        relative_prefix = _single_carbocycle_relative_cis_trans_prefix(mol, seq, Chem)
        if absolute_prefix and not (
            relative_prefix and _prefer_relative_carbocycle_prefix_for_complex_name(name, absolute_prefix)
        ):
            return absolute_prefix
        if relative_prefix:
            return relative_prefix
        return ""
    seq_set = set(seq)
    tagged_atoms = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    if len(tagged_atoms) < 2:
        return ""
    if _has_assigned_bond_stereo(mol, Chem):
        return ""
    centers = assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
    if len(centers) < 2:
        return ""
    locants = {idx: loc for loc, idx in enumerate(seq, start=1)}
    locanted_centers = [(idx, descriptor) for idx, descriptor in centers if idx in locants]
    if not locanted_centers:
        return ""
    outside_count = len(centers) - len(locanted_centers)
    if outside_count and len(_locanted_stereo_descriptors_in_text(name)) < outside_count:
        return ""
    prefix = ",".join(
        f"{locants[idx]}{descriptor}"
        for idx, descriptor in sorted(locanted_centers, key=lambda item: locants[item[0]])
    )
    return "" if _stereo_prefixed_name_exceeds_preserve_cap(name, prefix) else prefix


def _single_carbocycle_locanted_absolute_prefix(mol, seq: Sequence[int], Chem, name: str = "") -> str:
    if _has_assigned_bond_stereo(mol, Chem):
        return ""
    centers = assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
    if not centers:
        return ""
    locants = {ring_idx: loc for loc, ring_idx in enumerate(seq, start=1)}
    locanted_centers = [(idx, descriptor) for idx, descriptor in centers if idx in locants]
    if not locanted_centers:
        return ""
    outside_count = len(centers) - len(locanted_centers)
    if outside_count and len(_locanted_stereo_descriptors_in_text(name)) < outside_count:
        return ""
    if any(
        mol.GetAtomWithIdx(idx).GetSymbol() != "C" or mol.GetAtomWithIdx(idx).GetIsAromatic()
        for idx, _descriptor in locanted_centers
    ):
        return ""
    prefix = ",".join(
        f"{locants[idx]}{descriptor}"
        for idx, descriptor in sorted(locanted_centers, key=lambda item: locants[item[0]])
    )
    return "" if _stereo_prefixed_name_exceeds_preserve_cap(name, prefix) else prefix


def _stereo_prefixed_name_exceeds_preserve_cap(name: str, prefix: str) -> bool:
    if not name or not prefix:
        return False
    return len(name) + len(prefix) + 3 > _STEREO_PREFIX_PRESERVE_MAX_NAME_LENGTH


def _prefer_relative_carbocycle_prefix_for_complex_name(name: str, prefix: str) -> bool:
    if not name:
        return False
    if len(name) + len(prefix) + 3 > 96:
        return True
    return name.count("(") > 1


def _single_carbocycle_relative_cis_trans_prefix(mol, seq: Sequence[int], Chem) -> str:
    if _has_assigned_bond_stereo(mol, Chem):
        return ""
    seq_set = set(seq)
    assigned = assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
    if any(idx not in seq_set for idx, _descriptor in assigned):
        return ""
    tagged = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetIdx() in seq_set and atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    if len(tagged) != 2:
        return ""
    if any(atom.GetSymbol() != "C" or atom.GetIsAromatic() for atom in tagged):
        return ""
    substituted_ring_atoms = {
        idx
        for idx in seq
        if any(nbr.GetIdx() not in seq_set for nbr in mol.GetAtomWithIdx(idx).GetNeighbors())
    }
    if substituted_ring_atoms != {atom.GetIdx() for atom in tagged}:
        return ""
    external_prefixes = [
        _simple_relative_stereo_external_prefix(mol, atom.GetIdx(), seq_set) for atom in tagged
    ]
    if any(not prefix for prefix in external_prefixes):
        return ""
    ring_bonds = [
        bond
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in seq_set and bond.GetEndAtomIdx() in seq_set
    ]
    if any(bond.GetBondType() != Chem.BondType.SINGLE for bond in ring_bonds):
        return ""
    tags = [atom.GetChiralTag() for atom in tagged]
    return "cis" if tags[0] == tags[1] else "trans"


_SIMPLE_RELATIVE_STEREO_SUBSTITUENTS = {
    "amino",
    "acetamido",
    "bocamino",
    "carbethoxy",
    "carbmethoxy",
    "carboxy",
    "carboxymethyl",
    "carbamyl",
    "carbamoyl",
    "bromo",
    "chloro",
    "cyano",
    "dimethylamino",
    "dimethylcarbamyl",
    "dimethylcarbamoyl",
    "ethyl",
    "ethylamino",
    "ethoxy",
    "fluoro",
    "hydroxy",
    "iodo",
    "methoxy",
    "methyl",
    "methylamino",
    "methylcarbamoyl",
    "methylcarbonylamino",
    "methoxycarbonyl",
    "propyl",
    "propoxy",
    "tert-butoxycarbonylamino",
    "tert-butyloxycarbonylamino",
    "2-oxaethyl",
    "2-oxo-3-oxapropyl",
    "2-oxo-3-azapropyl",
    "sulfanyl",
}


def _simple_relative_stereo_external_prefix(mol, atom_idx: int, ring_atoms: set[int]) -> str:
    external = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if nbr.GetIdx() not in ring_atoms
    ]
    if len(external) != 1:
        return ""
    prefix = simple_external_prefix(mol, external[0], atom_idx) or branch_prefix(mol, external[0], atom_idx)
    if prefix in _SIMPLE_RELATIVE_STEREO_SUBSTITUENTS:
        return prefix
    if re.fullmatch(r"(?:[1-9]-)?(?:methyl|ethyl|propyl|butyl)oxy", prefix):
        return prefix
    if re.fullmatch(r"(?:di)?(?:methyl|ethyl|propyl|butyl)amino", prefix):
        return prefix
    if re.fullmatch(r"(?:[1-9]-)?(?:carboxy|carbamyl|carbamoyl|cyano)(?:methyl|ethyl|propyl|butyl)", prefix):
        return prefix
    if re.fullmatch(r"(?:[1-9]-)?(?:boc|acet|benz|propion)amido", prefix):
        return prefix
    if re.fullmatch(r"\(\(?1,1-dimethyl-ethyl\)?oxycarbonyl\)amino", prefix):
        return prefix
    if _safe_rendered_relative_stereo_external_prefix(prefix):
        return prefix
    return ""


def _safe_rendered_relative_stereo_external_prefix(prefix: str) -> bool:
    if not prefix or len(prefix) > 96:
        return False
    if prefix.count("(") != prefix.count(")"):
        return False
    if re.search(r"^\([^)]+\)-", prefix):
        return False
    if "methylmethyl" in prefix:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,()'\-]+", prefix):
        return False
    if any(term in prefix for term in ("ylidene", "imino", "hydrazono")):
        return False
    return bool(
        re.search(
            r"(?:"
            r"hydroxymethyl|isopropyl|t-butyl|tert-butyl|"
            r"(?:methyl|ethyl|propyl|butyl)$|"
            r"(?:carboxy|carbamyl|carbamoyl|cyano|oxo|hydroxy|amino)[a-z0-9,\-()']+|"
            r"[a-z0-9,\-()']+(?:azetidin|piperidin|piperazin|morpholin|pyrrolidin|"
            r"indazol|imidazol|pyrazol|pyrid|phenyl|cycloprop|cyclobut|cyclopent|cyclohex)[a-z0-9,\-()']*"
            r")",
            prefix,
        )
    )


def _polycyclic_stereo_prefix(mol, seq: Sequence[int], Chem, name: str = "") -> str:
    if not seq:
        return ""
    seq_set = set(seq)
    tagged_atoms = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    if not tagged_atoms:
        return ""
    if _has_assigned_bond_stereo(mol, Chem):
        return ""
    centers = assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
    if not centers:
        return _relative_bicyclic_stereo_prefix(mol, seq, Chem, name)
    locants = {idx: loc for loc, idx in enumerate(seq, start=1)}
    locanted_centers = [(idx, descriptor) for idx, descriptor in centers if idx in locants]
    if not locanted_centers:
        return ""
    outside_count = len(centers) - len(locanted_centers)
    if outside_count and len(_locanted_stereo_descriptors_in_text(name)) < outside_count:
        return ""
    return ",".join(
        f"{locants[idx]}{descriptor}"
        for idx, descriptor in sorted(locanted_centers, key=lambda item: locants[item[0]])
    )


def _locanted_stereo_descriptors_in_text(name: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])\d+'?[RS](?![a-z])", name)


def _relative_bicyclic_stereo_prefix(mol, seq: Sequence[int], Chem, name: str) -> str:
    """Use OPSIN's relative cis/trans prefix for pseudoasymmetric bicyclo parents."""

    if "bicyclo[" not in name or "spiro[" in name:
        return ""
    seq_set = set(seq)
    centers = [
        (idx, descriptor)
        for idx, descriptor in _fresh_all_atom_stereochemistry(mol, Chem).items()
        if descriptor in {"r", "s"} and idx in seq_set
    ]
    if len(centers) != 2:
        return ""
    tags = [mol.GetAtomWithIdx(idx).GetChiralTag() for idx, _descriptor in centers]
    return "cis" if tags[0] == tags[1] else "trans"


def _single_ring_oxo_suffix_name(
    prefixes: Sequence[tuple[str, str]],
    parent_name: str,
    indicated_h_locs: Sequence[str] = (),
) -> str:
    oxo_locs = [loc for loc, prefix in prefixes if prefix == "oxo" and loc.isdigit()]
    if not oxo_locs:
        return ""
    remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "oxo"]
    prefix_text = _format_prefixes(remaining)
    if parent_name == "1,2,3-triazoxole" and oxo_locs == ["5"]:
        name = "1,2,3-triazoxol-5(3H)-one"
        return _join_prefix_parent(prefix_text, name) if prefix_text else name
    indicated_h_name = _single_ring_indicated_h_chalcogen_suffix_name(
        parent_name,
        oxo_locs,
        indicated_h_locs,
        "one",
    )
    if indicated_h_name:
        return _join_prefix_parent(prefix_text, indicated_h_name) if prefix_text else indicated_h_name
    parent = _oxo_adjusted_unsaturated_carbocycle_parent(parent_name, oxo_locs)
    if not parent:
        parent = parent_name[:-1] if parent_name.endswith("e") else parent_name
    locants = ",".join(sorted(oxo_locs, key=int))
    if "lambda" in parent_name:
        suffix = "one" if len(oxo_locs) == 1 else f"{MULT.get(len(oxo_locs), '')}one"
    else:
        suffix = "on" if len(oxo_locs) == 1 else f"{MULT.get(len(oxo_locs), '')}on"
    name = f"{parent}-{locants}-{suffix}"
    return _join_prefix_parent(prefix_text, name) if prefix_text else _preferred_parent_stem(name)


def _single_ring_sulfur_oxide_suffix_name(
    mol,
    raw_subs: Sequence[tuple[int, str]],
    pos: dict[int, int],
    prefixes: Sequence[tuple[str, str]],
    parent_name: str,
) -> str:
    sulfur_oxo_locs = []
    sulfur_oxo_hydrogens = {}
    for ring_idx, prefix in raw_subs:
        if prefix != "oxo" or ring_idx not in pos:
            continue
        atom = mol.GetAtomWithIdx(ring_idx)
        if atom.GetSymbol() == "S":
            loc = str(pos[ring_idx])
            sulfur_oxo_locs.append(loc)
            sulfur_oxo_hydrogens[loc] = atom.GetTotalNumHs()
    if not sulfur_oxo_locs:
        return ""
    loc_counts = Counter(sulfur_oxo_locs)
    if any(count > 2 for count in loc_counts.values()):
        return ""
    sulfur_oxo_counter = Counter((loc, "oxo") for loc in sulfur_oxo_locs)
    remaining = []
    for item in prefixes:
        if item[1] == "oxo" and sulfur_oxo_counter[item] > 0:
            sulfur_oxo_counter[item] -= 1
            continue
        remaining.append(item)
    prefix_text = _format_prefixes(remaining)
    oxide_parts = []
    for loc in sorted(loc_counts, key=int):
        count = loc_counts[loc]
        locants = ",".join([loc] * count)
        suffix = "oxide" if count == 1 else "dioxide"
        oxide_parts.append(f"{locants}-{suffix}")
    parent = parent_name
    if len(loc_counts) == 1:
        loc, count = next(iter(loc_counts.items()))
        if count == 1 and sulfur_oxo_hydrogens.get(loc):
            parent = f"{loc}lambda6-" + parent
    name = parent + " " + " ".join(oxide_parts)
    return _join_prefix_parent(prefix_text, name) if prefix_text else name


def _single_ring_sulfur_oxide_hydrogen_locants(
    mol,
    raw_subs: Sequence[tuple[int, str]],
    pos: dict[int, int],
) -> tuple[str, ...]:
    locants = []
    for ring_idx, prefix in raw_subs:
        if prefix != "oxo" or ring_idx not in pos:
            continue
        atom = mol.GetAtomWithIdx(ring_idx)
        if atom.GetSymbol() == "S" and atom.GetTotalNumHs() > 0:
            locants.append(str(pos[ring_idx]))
    return tuple(sorted(set(locants), key=int))


def _single_ring_thione_s_oxide_suffix_name(
    mol,
    raw_subs: Sequence[tuple[int, str]],
    pos: dict[int, int],
    prefixes: Sequence[tuple[str, str]],
    parent_name: str,
    indicated_h_locs: Sequence[str] = (),
) -> str:
    thione_oxide_locs = []
    for ring_idx, prefix in raw_subs:
        if prefix != "sulfanyl" or ring_idx not in pos:
            continue
        if not _ring_external_thione_s_oxide_site(mol, ring_idx):
            continue
        thione_oxide_locs.append(str(pos[ring_idx]))
    if len(thione_oxide_locs) != 1:
        return ""
    loc = thione_oxide_locs[0]
    remaining = []
    removed = False
    for item in prefixes:
        if not removed and item == (loc, "sulfanyl"):
            removed = True
            continue
        remaining.append(item)
    if not removed:
        return ""
    prefix_text = _format_prefixes(remaining)
    indicated_h_name = _single_ring_indicated_h_chalcogen_suffix_name(
        parent_name,
        [loc],
        indicated_h_locs,
        "thione",
    )
    if indicated_h_name:
        name = indicated_h_name + " S-oxide"
        return _join_prefix_parent(prefix_text, name) if prefix_text else name
    parent = parent_name[:-1] if parent_name.endswith("e") else parent_name
    name = f"{parent}-{loc}-thione S-oxide"
    return _join_prefix_parent(prefix_text, name) if prefix_text else _preferred_parent_stem(name)


def _ring_external_thione_s_oxide_site(mol, ring_idx: int) -> bool:
    ring_atom = mol.GetAtomWithIdx(ring_idx)
    if ring_atom.GetSymbol() != "C":
        return False
    for sulfur in ring_atom.GetNeighbors():
        sulfur_idx = sulfur.GetIdx()
        bond = mol.GetBondBetweenAtoms(ring_idx, sulfur_idx)
        if (
            sulfur.GetSymbol() != "S"
            or sulfur.GetFormalCharge() != 0
            or bond is None
            or bond.GetBondTypeAsDouble() != 2.0
        ):
            continue
        double_oxygen_count = 0
        unsupported = False
        for nbr in sulfur.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == ring_idx:
                continue
            nbr_bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
            if (
                nbr.GetSymbol() == "O"
                and nbr.GetFormalCharge() == 0
                and nbr_bond is not None
                and nbr_bond.GetBondTypeAsDouble() == 2.0
                and len([other for other in nbr.GetNeighbors() if other.GetAtomicNum() > 1]) == 1
            ):
                double_oxygen_count += 1
            elif nbr.GetAtomicNum() > 1:
                unsupported = True
                break
        if not unsupported and double_oxygen_count == 1:
            return True
    return False


def _single_ring_thione_suffix_name(
    prefixes: Sequence[tuple[str, str]],
    parent_name: str,
    indicated_h_locs: Sequence[str] = (),
) -> str:
    thioxo_locs = [loc for loc, prefix in prefixes if prefix == "thioxo" and loc.isdigit()]
    if not thioxo_locs:
        return ""
    remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "thioxo"]
    prefix_text = _format_prefixes(remaining)
    indicated_h_name = _single_ring_indicated_h_chalcogen_suffix_name(
        parent_name,
        thioxo_locs,
        indicated_h_locs,
        "thione",
    )
    if indicated_h_name:
        return _join_prefix_parent(prefix_text, indicated_h_name) if prefix_text else indicated_h_name
    parent = parent_name[:-1] if parent_name.endswith("e") else parent_name
    locants = ",".join(sorted(thioxo_locs, key=int))
    suffix = "thione" if len(thioxo_locs) == 1 else f"{MULT.get(len(thioxo_locs), '')}thione"
    name = f"{parent}-{locants}-{suffix}"
    return _join_prefix_parent(prefix_text, name) if prefix_text else _preferred_parent_stem(name)


_SINGLE_RING_INDICATED_H_CHALCOGEN_SUFFIX_PARENTS = (
    "oxadiazol",
    "tetrazin",
    "thiadiazol",
    "triazin",
    "triazol",
    "tetrazol",
)

_SINGLE_RING_INDICATED_H_CHALCOGEN_SUFFIX_PARENT_FULL_FORMS = {
    "oxadiazol": "oxadiazole",
    "tetrazin": "tetrazine",
    "thiadiazol": "thiadiazole",
    "triazin": "triazine",
    "triazol": "triazole",
    "tetrazol": "tetrazole",
}


def _is_single_ring_indicated_h_chalcogen_suffix_full_form(name: str) -> bool:
    if "H)-" not in name or not name.endswith(("-one", "-thione")):
        return False
    return any(f"{full}-" in name for full in _SINGLE_RING_INDICATED_H_CHALCOGEN_SUFFIX_PARENT_FULL_FORMS.values())


def _single_ring_indicated_h_chalcogen_suffix_name(
    parent_name: str,
    suffix_locs: Sequence[str],
    indicated_h_locs: Sequence[str],
    suffix: str,
) -> str:
    if len(suffix_locs) != 1 or not indicated_h_locs:
        return ""
    parent = _strip_leading_indicated_hydrogen(parent_name)
    stem_parent = parent[:-1] if parent.endswith("e") else parent
    stem_parent = _preferred_parent_stem(stem_parent)
    if not any(stem in stem_parent for stem in _SINGLE_RING_INDICATED_H_CHALCOGEN_SUFFIX_PARENTS):
        return ""
    h_locs = [loc for loc in indicated_h_locs if loc != suffix_locs[0]]
    if not h_locs:
        return ""
    h_loc = sorted(h_locs, key=_locant_sort_key)[-1]
    for stem, full in _SINGLE_RING_INDICATED_H_CHALCOGEN_SUFFIX_PARENT_FULL_FORMS.items():
        if parent.endswith(stem):
            parent = f"{parent[:-len(stem)]}{full}"
            break
    return f"{parent}-{suffix_locs[0]}({h_loc}H)-{suffix}"


def _single_ring_imino_suffix_name(prefixes: Sequence[tuple[str, str]], parent_name: str) -> str:
    if parent_name != "1,2,3-triazoxole":
        return ""
    imino_locs = [loc for loc, prefix in prefixes if prefix == "imino" and loc.isdigit()]
    if imino_locs != ["5"]:
        return ""
    remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "imino"]
    prefix_text = _format_prefixes(remaining)
    name = "1,2,3-triazoxol-5(3H)-imine"
    return _join_prefix_parent(prefix_text, name) if prefix_text else name


def _single_ring_oxo_methylidene_suffix_name(prefixes: Sequence[tuple[str, str]], parent_name: str) -> str:
    if not re.fullmatch(r"\d+H-pyridine", parent_name):
        return ""
    h_loc = parent_name.split("H-", 1)[0]
    carbonyl_locs = [loc for loc, prefix in prefixes if prefix == "oxomethylidene" and loc == h_loc]
    if carbonyl_locs != [h_loc]:
        return ""
    remaining = [(loc, prefix) for loc, prefix in prefixes if not (loc == h_loc and prefix == "oxomethylidene")]
    prefix_text = _format_prefixes(remaining)
    parent = _preferred_parent_stem(parent_name)
    core = f"{h_loc}-carbonyl-{parent}"
    return _join_prefix_parent(prefix_text, core) if prefix_text else core


_SINGLE_RING_FUNCTIONAL_SUFFIXES = {
    "1-azamethylcarbonyl": "carboxamide",
    "carbamyl": "carboxamide",
    "carbamoyl": "carboxamide",
    "carboxy": "carboxylic acid",
    "carboxylato": "carboxylate",
    "cyano": "carbonitrile",
    "oxidosulfanyl": "sulfenate",
}
_STEREO_PREFIX_PRESERVE_MAX_NAME_LENGTH = 152

_FUSED_FUNCTIONAL_SUFFIXES = _SINGLE_RING_FUNCTIONAL_SUFFIXES


def _may_have_fused_functional_suffix(prefixes: Sequence[tuple[str, str]]) -> bool:
    for _loc, prefix in prefixes:
        if prefix in _FUSED_FUNCTIONAL_SUFFIXES:
            return True
        if "carbam" in prefix or ("aza" in prefix and "carbonyl" in prefix):
            return True
    return False


def _may_have_fused_oxo_suffix(prefixes: Sequence[tuple[str, str]]) -> bool:
    return any(prefix == "oxo" and loc.isdigit() for loc, prefix in prefixes)


def _may_have_fused_thione_suffix(prefixes: Sequence[tuple[str, str]]) -> bool:
    return any(prefix == "thioxo" and loc.isdigit() for loc, prefix in prefixes)


def _may_have_fused_imine_suffix(prefixes: Sequence[tuple[str, str]]) -> bool:
    return any(prefix == "imino" and re.fullmatch(r"\d+[a-z]?", loc) for loc, prefix in prefixes)


_SINGLE_RING_ESTER_SUFFIX_OWNERS = {
    "methoxycarbonyl": "methyl",
    "carbmethoxy": "methyl",
    "ethoxycarbonyl": "ethyl",
    "carbethoxy": "ethyl",
    "propoxycarbonyl": "propyl",
    "carbopropoxy": "propyl",
    "isopropoxycarbonyl": "isopropyl",
    "butoxycarbonyl": "butyl",
    "carbobutoxy": "butyl",
    "tert-butoxycarbonyl": "tert-butyl",
    "tert-butyloxycarbonyl": "tert-butyl",
    "boc": "tert-butyl",
    "benzyloxycarbonyl": "benzyl",
    "carbbenzoxy": "benzyl",
    "cbz": "benzyl",
    "phenoxycarbonyl": "phenyl",
    "carbphenoxy": "phenyl",
}


def _single_ring_functional_suffix_name(prefixes: Sequence[tuple[str, str]], parent_name: str) -> str:
    """Promote terminal ring functional groups to OPSIN-stable parent suffixes."""

    supports_parent = _supports_single_ring_functional_suffix_parent(parent_name)
    supports_ester_parent = supports_parent or parent_name == "benzene"
    supports_sulfenate_parent = supports_parent or _supports_sulfenate_suffix_parent(parent_name)
    if not supports_parent and not (
        supports_ester_parent
        and any(prefix in _SINGLE_RING_ESTER_SUFFIX_OWNERS for _loc, prefix in prefixes)
    ) and not (
        supports_sulfenate_parent
        and any(prefix == "oxidosulfanyl" for _loc, prefix in prefixes)
    ):
        return ""
    best_key = None
    best_name = ""
    for matched, suffix_base, n_substituent in _iter_single_ring_functional_suffix_terms(
        prefixes,
        supports_parent,
        supports_sulfenate_parent=supports_sulfenate_parent,
    ):
        suffix_locs = [loc for loc, _prefix in matched]
        matched_set = set(matched)
        remaining = [
            (other_loc, candidate_prefix)
            for other_loc, candidate_prefix in prefixes
            if (other_loc, candidate_prefix) not in matched_set
        ]
        suffix_locs = sorted(suffix_locs, key=_locant_sort_key)
        suffix = _single_ring_functional_suffix_text(suffix_base, len(suffix_locs))
        if not suffix:
            continue
        locants = ",".join(suffix_locs)
        retained_name = _retained_suffix_parent_alias_with_prefixes(
            parent_name,
            suffix_base,
            suffix_locs,
            n_substituent,
            remaining,
        )
        if retained_name:
            name = retained_name
        else:
            suffix_name = _retained_suffix_parent_alias(parent_name, suffix_base, suffix_locs, n_substituent)
            if not suffix_name:
                suffix_name = f"{parent_name}-{locants}-{suffix}"
                if n_substituent:
                    suffix_name = f"{n_substituent}-{suffix_name}"
            elif n_substituent:
                suffix_name = f"{n_substituent}-{suffix_name}"
            prefix_text = _format_prefixes(remaining)
            name = _join_prefix_parent(prefix_text, suffix_name) if prefix_text else _preferred_parent_stem(suffix_name)
        original_prefix_text = _format_prefixes(prefixes)
        original_name = _join_prefix_parent(original_prefix_text, parent_name) if original_prefix_text else parent_name
        original_name = _functional_suffix_prefix_baseline(original_name)
        if (
            (
                len(name) > len(original_name)
                or _tautomer_sensitive_single_ring_suffix_parent(parent_name, remaining)
            )
            and not _allow_simple_functional_suffix_parent(
                parent_name,
                suffix_base,
                suffix_locs,
                remaining,
                name,
                original_name,
            )
        ):
            continue
        key = (len(remaining), len(name), locants, name)
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
    if supports_ester_parent:
        ester_name = _single_ring_ester_suffix_name(prefixes, parent_name)
        if ester_name:
            ester_key = (0, len(ester_name), _substituent_locant_tuple(ester_name), ester_name)
            if best_key is None or ester_key < best_key:
                best_key = ester_key
                best_name = ester_name
    return best_name


def _iter_single_ring_functional_suffix_terms(
    prefixes: Sequence[tuple[str, str]],
    supports_parent: bool,
    *,
    supports_sulfenate_parent: bool = False,
) -> list[tuple[tuple[tuple[str, str], ...], str, str]]:
    if not supports_parent and not supports_sulfenate_parent:
        return []
    terms = []
    for prefix, suffix_base in _SINGLE_RING_FUNCTIONAL_SUFFIXES.items():
        if not supports_parent and suffix_base != "sulfenate":
            continue
        matched = tuple((loc, candidate_prefix) for loc, candidate_prefix in prefixes if candidate_prefix == prefix)
        if matched:
            terms.append((matched, suffix_base, ""))
    if not supports_parent:
        return terms
    for loc, prefix in prefixes:
        if prefix in _SINGLE_RING_FUNCTIONAL_SUFFIXES:
            continue
        n_substituent = _n_substituted_carboxamide_prefix(prefix)
        if n_substituent:
            terms.append((((loc, prefix),), "carboxamide", n_substituent))
    return terms


def _fused_functional_suffix_name(
    prefixes: Sequence[tuple[str, str]],
    parent_name: str,
    indicated_h_locs: Sequence[str] = (),
) -> str:
    """Promote fused-ring terminal functional groups to locanted suffix parents."""

    if "acenaphthen" in parent_name:
        return ""
    best_key = None
    best_name = ""
    for loc, prefix in prefixes:
        suffix_base = _FUSED_FUNCTIONAL_SUFFIXES.get(prefix)
        n_substituent = ""
        if suffix_base is None:
            n_substituent = _n_substituted_carboxamide_prefix(prefix)
            if not n_substituent:
                continue
            suffix_base = "carboxamide"
        if not _allow_fused_functional_suffix(parent_name, suffix_base, n_substituent, indicated_h_locs):
            continue
        if not _is_fused_functional_suffix_locant(loc):
            continue
        remaining = [(other_loc, other_prefix) for other_loc, other_prefix in prefixes if (other_loc, other_prefix) != (loc, prefix)]
        if any(other_prefix == "oxo" for _other_loc, other_prefix in remaining):
            continue
        parent = _fused_functional_suffix_parent(parent_name, suffix_base, indicated_h_locs)
        suffix = _single_ring_functional_suffix_text(suffix_base, 1)
        if not parent or not suffix:
            continue
        prefix_text = _format_prefixes(remaining)
        core = _retained_suffix_parent_alias(parent, suffix_base, (loc,), n_substituent)
        if not core:
            core = f"{parent}-{loc}-{suffix}"
            if n_substituent:
                core = f"{n_substituent}-{core}"
        elif n_substituent:
            core = f"{n_substituent}-{core}"
        name = _join_prefix_parent(prefix_text, core) if prefix_text else _preferred_parent_stem(core)
        original_prefix_text = _format_prefixes(prefixes)
        original_name = _join_prefix_parent(original_prefix_text, parent_name) if original_prefix_text else parent_name
        original_name = _functional_suffix_prefix_baseline(original_name)
        if len(name) > len(original_name) + _fused_functional_suffix_extra_budget(suffix_base, n_substituent):
            continue
        key = (len(remaining), len(name), loc, name)
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
    return best_name


def _is_fused_functional_suffix_locant(locant: str) -> bool:
    return bool(locant and re.match(r"\d+[a-z]?$", locant))


def _fused_functional_suffix_parent(parent_name: str, suffix_base: str, indicated_h_locs: Sequence[str]) -> str:
    if "hydro-" in parent_name:
        return _hydro_fused_functional_suffix_parent(parent_name, suffix_base, indicated_h_locs)
    if suffix_base == "carboxamide" and indicated_h_locs:
        return _drop_indicated_hydrogen_segments(parent_name, ())
    if indicated_h_locs and suffix_base in {"carbonitrile"}:
        return parent_name
    return _drop_indicated_hydrogen_segments(parent_name, ())


def _hydro_fused_functional_suffix_parent(
    parent_name: str,
    suffix_base: str,
    indicated_h_locs: Sequence[str],
) -> str:
    if suffix_base in {"carboxamide", "carboxylate", "carboxylic acid", "carbonitrile"} and "pyrazolo[" in parent_name:
        h_loc = _hydro_fused_nonhydro_indicated_h_loc(parent_name, indicated_h_locs)
        if h_loc and not re.search(r"(?:^|-)\d+[a-z]?H-", parent_name):
            return re.sub(
                r"^((?:\d+[a-z]?,)*\d+[a-z]?-[a-z]*hydro-)",
                lambda match: f"{match.group(1)}{h_loc}H-",
                parent_name,
                count=1,
            )
    return _drop_indicated_hydrogen_segments(parent_name, ())


def _hydro_fused_nonhydro_indicated_h_loc(parent_name: str, indicated_h_locs: Sequence[str]) -> str:
    hydro_locs: set[str] = set()
    match = re.search(r"((?:\d+[a-z]?,)*\d+[a-z]?)-[a-z]*hydro-", parent_name)
    if match:
        hydro_locs = set(match.group(1).split(","))
    candidates = [loc for loc in indicated_h_locs if loc not in hydro_locs]
    if not candidates:
        return ""
    return sorted(candidates, key=_locant_sort_key)[0]


def _fused_functional_suffix_extra_budget(suffix_base: str, n_substituent: str) -> int:
    if n_substituent:
        return 8
    if suffix_base == "carbonitrile":
        return 10
    if suffix_base in {"carboxylic acid", "carboxylate"}:
        return 18
    return 8


def _allow_fused_functional_suffix(
    parent_name: str,
    suffix_base: str,
    n_substituent: str,
    indicated_h_locs: Sequence[str],
) -> bool:
    if suffix_base not in {"carboxamide", "carboxylate", "carboxylic acid", "carbonitrile", "sulfenate"}:
        return False
    if suffix_base == "sulfenate":
        return _supports_fused_functional_suffix_parent(parent_name) or _supports_sulfenate_suffix_parent(parent_name)
    if suffix_base == "carbonitrile" and "benzimidazole" in parent_name:
        return False
    if suffix_base in {"carboxylate", "carboxylic acid"} and re.search(r"(?:^|-)(?:\d+[a-z]?\([^)]*H\)-)?one\b", parent_name):
        return False
    if n_substituent:
        return (
            suffix_base == "carboxamide"
            and _supports_fused_functional_suffix_parent(parent_name)
            and _supports_fused_n_substituted_carboxamide(parent_name, n_substituent)
        )
    if indicated_h_locs and suffix_base == "carboxylate":
        return False
    return _supports_fused_functional_suffix_parent(parent_name)


def _supports_fused_functional_suffix_parent(parent_name: str) -> bool:
    if "spiro[" in parent_name:
        return False
    return any(
        stem in parent_name
        for stem in (
            "benzimidazole",
            "benzothiazole",
            "benzoxazole",
            "benzotriazole",
            "cinnoline",
            "imidazo[",
            "indazole",
            "indeno[",
            "indole",
            "isoquinoline",
            "naphthyridine",
            "purine",
            "pteridine",
            "pyrazolo[",
            "pyrido[",
            "pyrimido[",
            "pyrrolo[",
            "quinazoline",
            "quinoline",
            "quinoxaline",
            "triazolo[",
        )
    )


def _supports_sulfenate_suffix_parent(parent_name: str) -> bool:
    if parent_name == "benzene":
        return True
    return any(
        stem in parent_name
        for stem in (
            "benzimidazol",
            "benzothiazol",
            "benzoxazol",
            "benzotriazol",
            "cinnolin",
            "furan",
            "imidazo[",
            "imidazol",
            "indazol",
            "indeno[",
            "indol",
            "isoquinolin",
            "isothiazol",
            "isoxazol",
            "naphthalen",
            "naphthyridin",
            "oxadiazol",
            "oxazol",
            "purin",
            "pteridin",
            "pyrazin",
            "pyrazol",
            "pyridazin",
            "pyridin",
            "pyrimido[",
            "pyrimidin",
            "pyrrolo[",
            "pyrrol",
            "quinazolin",
            "quinolin",
            "quinoxalin",
            "thiadiazol",
            "thiazol",
            "thiophen",
            "triazol",
            "triazolo[",
        )
    )


def _supports_fused_n_substituted_carboxamide(parent_name: str, n_substituent: str) -> bool:
    if not any(stem in parent_name for stem in ("imidazo[", "pyrazolo[", "pyrido[", "pyrimido[")):
        return True
    return bool(
        re.fullmatch(
            r"N-(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl|"
            r"isopropyl|t-butyl|tert-butyl)|N,N-di(?:methyl|ethyl|propyl|butyl)",
            n_substituent,
        )
    )


def _n_substituted_carboxamide_prefix(prefix: str) -> str:
    group = _azamethylcarbonyl_n_substituent(prefix)
    if group:
        return _format_amide_n_substituent(group)
    group = _azaalkylcarbonyl_n_substituent(prefix)
    if group:
        return _format_amide_n_substituent(group)
    group = _carbamoyl_n_substituent(prefix)
    if group:
        return _format_amide_n_substituent(group)
    return ""


def _azamethylcarbonyl_n_substituent(prefix: str) -> str:
    match = re.fullmatch(r"1-\((.+)\)-1-azamethylcarbonyl", prefix)
    return match.group(1) if match else ""


def _azaalkylcarbonyl_n_substituent(prefix: str) -> str:
    return {
        "1-azaethylcarbonyl": "methyl",
        "1-azapropylcarbonyl": "ethyl",
        "1-azabutylcarbonyl": "propyl",
        "1-azapentylcarbonyl": "butyl",
        "1-azahexylcarbonyl": "pentyl",
        "1-azaheptylcarbonyl": "hexyl",
        "1-azanonylcarbonyl": "octyl",
        "1-azadecylcarbonyl": "nonyl",
        "1-methyl-1-azaethylcarbonyl": "dimethyl",
    }.get(prefix, "")


def _carbamoyl_n_substituent(prefix: str) -> str:
    match = re.fullmatch(r"(.+?)carbamo?yl", prefix)
    return match.group(1) if match else ""


def _format_amide_n_substituent(group: str) -> str:
    if _already_locanted_amide_n_substituent(group):
        return group
    bis_match = re.fullmatch(r"bis\((.+)\)", group)
    if bis_match:
        return f"N,N-bis({bis_match.group(1)})"
    match = re.fullmatch(
        r"di(methyl|ethyl|propyl|isopropyl|butyl|isobutyl|tert-butyl|t-butyl|"
        r"phenyl|benzyl|cyclopropyl|cyclobutyl|cyclopentyl|cyclohexyl)",
        group,
    )
    if match:
        return f"N,N-di{match.group(1)}"
    if group.startswith("(") and group.endswith(")"):
        return f"N-{group}"
    if re.fullmatch(r"[A-Za-z0-9]+", group):
        return f"N-{group}"
    return f"N-({group})"


def _already_locanted_amide_n_substituent(group: str) -> bool:
    return group.startswith(("N-", "N,N-", "N,N'-"))


_PYRIDINE_RETAINED_SUFFIX_ALIASES = {
    "carboxamide": {
        "2": "picolinamide",
        "3": "nicotinamide",
        "4": "isonicotinamide",
    },
    "carboxylic acid": {
        "2": "picolinic acid",
        "3": "nicotinic acid",
        "4": "isonicotinic acid",
    },
    "carboxylate": {
        "2": "picolinate",
        "3": "nicotinate",
        "4": "isonicotinate",
    },
    "carbonitrile": {
        "2": "picolinonitrile",
        "3": "nicotinonitrile",
        "4": "isonicotinonitrile",
    },
}


def _retained_suffix_parent_alias(
    parent_name: str,
    suffix_base: str,
    suffix_locs: Sequence[str],
    n_substituent: str = "",
) -> str:
    """Return retained suffix-parent aliases selected by the parent locant map."""

    if n_substituent and suffix_base != "carboxamide":
        return ""
    if len(suffix_locs) != 1:
        return ""
    locant = suffix_locs[0]
    if parent_name in {"pyridine", "pyridin"}:
        return _PYRIDINE_RETAINED_SUFFIX_ALIASES.get(suffix_base, {}).get(locant, "")
    return ""


def _retained_suffix_parent_alias_with_prefixes(
    parent_name: str,
    suffix_base: str,
    suffix_locs: Sequence[str],
    n_substituent: str,
    remaining: Sequence[tuple[str, str]],
) -> str:
    if parent_name not in {"pyridine", "pyridin"} or len(suffix_locs) != 1:
        return ""
    if n_substituent and _simple_n_substituent_group(n_substituent) is None:
        return ""
    alias, locant_map = _pyridine_retained_suffix_alias_and_locants(suffix_base, suffix_locs[0], remaining)
    if not alias:
        return ""
    remapped = []
    for loc, prefix in remaining:
        mapped_loc = locant_map.get(loc)
        if not mapped_loc:
            return ""
        remapped.append((mapped_loc, prefix))
    combined_prefix = _combined_n_and_ring_prefix_text(n_substituent, remapped)
    if combined_prefix:
        return _join_prefix_parent(combined_prefix, alias)
    core = f"{n_substituent}-{alias}" if n_substituent else alias
    prefix_text = _format_prefixes(remapped)
    return _join_prefix_parent(prefix_text, core) if prefix_text else _preferred_parent_stem(core)


def _combined_n_and_ring_prefix_text(n_substituent: str, remapped: Sequence[tuple[str, str]]) -> str:
    if not n_substituent or not remapped:
        return ""
    parsed = _simple_n_substituent_group(n_substituent)
    if parsed is None:
        return ""
    term, n_locs = parsed
    if any(prefix != term for _loc, prefix in remapped):
        return ""
    ring_locs = sorted((loc for loc, _prefix in remapped), key=_locant_sort_key)
    all_locs = tuple(n_locs) + tuple(ring_locs)
    multiplier = MULT.get(len(all_locs), "")
    if not multiplier:
        return ""
    return f"{','.join(all_locs)}-{multiplier}{term}"


def _simple_n_substituent_group(n_substituent: str) -> tuple[str, tuple[str, ...]] | None:
    match = re.fullmatch(r"N-(methyl|ethyl|propyl|isopropyl|butyl|isobutyl|tert-butyl|t-butyl)", n_substituent)
    if match:
        return match.group(1), ("N",)
    match = re.fullmatch(r"N,N-di(methyl|ethyl|propyl|butyl)", n_substituent)
    if match:
        return match.group(1), ("N", "N")
    return None


def _pyridine_retained_suffix_alias_and_locants(
    suffix_base: str,
    locant: str,
    remaining: Sequence[tuple[str, str]],
) -> tuple[str, dict[str, str]]:
    aliases = _PYRIDINE_RETAINED_SUFFIX_ALIASES.get(suffix_base, {})
    if locant in {"2", "3"}:
        alias = aliases.get(locant, "")
        return alias, _PYRIDINE_IDENTITY_LOCANTS if alias else ("", {})
    if locant in {"5", "6"}:
        alias_locant = {"5": "3", "6": "2"}[locant]
        alias = aliases.get(alias_locant, "")
        return alias, _PYRIDINE_MIRROR_LOCANTS if alias else ("", {})
    if locant == "4":
        alias = aliases.get("4", "")
        if not alias:
            return "", {}
        identity_text = _format_prefixes([(loc, prefix) for loc, prefix in remaining])
        mirrored = []
        for loc, prefix in remaining:
            mapped = _PYRIDINE_MIRROR_LOCANTS.get(loc)
            if not mapped:
                return alias, _PYRIDINE_IDENTITY_LOCANTS
            mirrored.append((mapped, prefix))
        mirror_text = _format_prefixes(mirrored)
        return alias, _PYRIDINE_MIRROR_LOCANTS if mirror_text and mirror_text < identity_text else _PYRIDINE_IDENTITY_LOCANTS
    return "", {}


_PYRIDINE_IDENTITY_LOCANTS = {
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
}


_PYRIDINE_MIRROR_LOCANTS = {
    "1": "1",
    "2": "6",
    "3": "5",
    "4": "4",
    "5": "3",
    "6": "2",
}


def _single_ring_ester_suffix_name(prefixes: Sequence[tuple[str, str]], parent_name: str) -> str:
    best_key = None
    best_name = ""
    original_prefix_text = _format_prefixes(prefixes)
    original_name = _join_prefix_parent(original_prefix_text, parent_name) if original_prefix_text else parent_name
    original_name = _functional_suffix_prefix_baseline(original_name)
    ester_prefixes = {
        prefix for _loc, prefix in prefixes if prefix in _SINGLE_RING_ESTER_SUFFIX_OWNERS
    }
    for prefix in ester_prefixes:
        suffix_locs = [loc for loc, candidate_prefix in prefixes if candidate_prefix == prefix]
        if not suffix_locs or len(suffix_locs) > 2:
            continue
        owner = _single_ring_ester_owner_text(prefix, len(suffix_locs))
        if not owner:
            continue
        remaining = [(loc, candidate_prefix) for loc, candidate_prefix in prefixes if candidate_prefix != prefix]
        suffix_locs = sorted(suffix_locs, key=_locant_sort_key)
        locants = ",".join(suffix_locs)
        prefix_text = _format_prefixes(remaining)
        retained_core = _retained_suffix_parent_alias(parent_name, "carboxylate", suffix_locs)
        if retained_core and prefix_text:
            core = _join_prefix_parent(prefix_text, retained_core)
        elif retained_core:
            core = retained_core
        elif parent_name == "benzene" and len(suffix_locs) == 1 and suffix_locs[0] == "1":
            core = prefix_text + "benzoate" if prefix_text else "benzoate"
        else:
            suffix = "carboxylate" if len(suffix_locs) == 1 else "dicarboxylate"
            parent_core = f"{parent_name}-{locants}-{suffix}"
            core = _join_prefix_parent(prefix_text, parent_core) if prefix_text else parent_core
        name = f"{owner} {core}"
        if len(name) >= len(original_name):
            continue
        key = (len(remaining), len(name), locants, name)
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
    return best_name


def _single_ring_ester_owner_text(prefix: str, count: int) -> str:
    owner = _SINGLE_RING_ESTER_SUFFIX_OWNERS.get(prefix, "")
    if not owner:
        return ""
    if count == 1:
        return owner
    if count == 2:
        return "di" + owner
    return ""


def _functional_suffix_prefix_baseline(name: str) -> str:
    return (
        name.replace("1-azamethylcarbonyl", "carbamoyl")
        .replace("1-oxido-1-azamethylcarbonyl", "carbamoyl")
        .replace("(carbamoyl)", "carbamoyl")
        .replace("carboxylato", "carboxylato")
    )


def _tautomer_sensitive_single_ring_suffix_parent(
    parent_name: str,
    remaining: Sequence[tuple[str, str]],
) -> bool:
    return bool(
        parent_name in {"pyrazole", "imidazole"}
        or re.search(r"(?:^|-)\d+[a-z]?H-", parent_name)
        or re.search(r"(?:^|-)\d+[a-z]?-one\b", parent_name)
        or (
            parent_name in {"pyrazine", "pyridazine", "pyridine", "pyrimidine"}
            and any(prefix == "oxo" for _loc, prefix in remaining)
        )
    )


def _allow_simple_functional_suffix_parent(
    parent_name: str,
    suffix_base: str,
    suffix_locs: Sequence[str],
    remaining: Sequence[tuple[str, str]],
    name: str,
    original_name: str,
) -> bool:
    """Let high-value suffix parents win for simple monocyclic retained parents.

    The prefix forms are shorter, but OPSIN and SureChEMBL both strongly prefer
    locanted suffix parents for simple heteroaryl acids/amides/nitriles.  This
    deliberately avoids complex poly-substituted cases where the longer suffix
    parent can perturb the global parent-family winner.
    """

    if suffix_base not in {"carboxamide", "carboxylate", "carboxylic acid", "carbonitrile"}:
        return False
    if len(suffix_locs) != 1 or not suffix_locs[0].isdigit():
        return False
    max_extra = 8 if suffix_base in {"carboxylic acid", "carbonitrile"} else 5
    if suffix_base == "carboxamide" and re.search(r"(?:^|-)N(?:,N)?-", name):
        max_extra = 10
    if len(name) > len(original_name) + max_extra:
        return False
    if parent_name in {"pyrazole", "imidazole"}:
        return False
    if parent_name in {"pyrazine", "pyridazine", "pyridine", "pyrimidine"} and any(
        prefix == "oxo" for _loc, prefix in remaining
    ):
        return False
    if not _simple_functional_suffix_parent_name(parent_name):
        return False
    return all(_simple_functional_suffix_remaining_prefix(prefix) for _loc, prefix in remaining)


def _simple_functional_suffix_parent_name(parent_name: str) -> bool:
    return parent_name in {
        "furan",
        "imidazole",
        "isothiazole",
        "isoxazole",
        "oxazole",
        "pyran",
        "pyrazine",
        "pyrazole",
        "pyridazine",
        "pyridine",
        "pyrimidine",
        "pyrrole",
        "thiazole",
        "thiopyran",
        "thiophene",
        "1H-pyrrole",
        "1H-pyrazole",
        "1H-imidazole",
    }


def _simple_functional_suffix_remaining_prefix(prefix: str) -> bool:
    if prefix in {
        "amino",
        "bromo",
        "chloro",
        "cyano",
        "ethyl",
        "ethoxy",
        "fluoro",
        "formyl",
        "hydroxy",
        "iodo",
        "methoxy",
        "methyl",
        "nitro",
        "oxo",
        "phenyl",
        "trifluoromethyl",
    }:
        return True
    return bool(re.fullmatch(r"(?:[234]-)?(?:fluoro|chloro|bromo|iodo|methyl|methoxy|trifluoromethyl)phenyl", prefix))


def _supports_single_ring_functional_suffix_parent(parent_name: str) -> bool:
    if parent_name == "benzene":
        return False
    return any(
        stem in parent_name
        for stem in (
            "furan",
            "imidazole",
            "isothiazole",
            "isoxazole",
            "oxadiazole",
            "oxazole",
            "pyran",
            "selenopyran",
            "pyrazine",
            "pyrazole",
            "pyridazine",
            "pyridine",
            "pyrimidine",
            "pyrrole",
            "s-triazine",
            "tetrazole",
            "thiadiazole",
            "thiazole",
            "thiopyran",
            "thiophene",
            "telluropyran",
            "triazole",
            "v-triazine",
        )
    )


def _single_ring_functional_suffix_text(suffix_base: str, count: int) -> str:
    if count <= 0:
        return ""
    if count == 1:
        return suffix_base
    multiplier = MULT.get(count)
    if not multiplier:
        return ""
    if suffix_base == "carboxylic acid":
        return f"{multiplier}carboxylic acid"
    if suffix_base == "carbonitrile":
        return f"{multiplier}carbonitrile"
    return f"{multiplier}{suffix_base}"


def _oxo_adjusted_unsaturated_carbocycle_parent(parent_name: str, oxo_locs: Sequence[str]) -> str:
    ring_size = _cycloalkapolyene_ring_size(parent_name)
    if ring_size is None:
        return ""
    oxo_atoms = {int(loc) for loc in oxo_locs}
    ene_locs = _max_nonoxo_ring_double_bond_locs(ring_size, oxo_atoms)
    root = "cyclo" + ROOTS[ring_size]
    if not ene_locs:
        return root + "an"
    parent = loc_unsat(root, ene_locs, "ene")
    return parent[:-1] if parent.endswith("e") else parent


def _cycloalkapolyene_ring_size(parent_name: str) -> int | None:
    if not parent_name.startswith("cyclo") or "en" not in parent_name:
        return None
    if re.search(r"^cycloprop-\d(?:,\d+)*-en", parent_name):
        return 3
    if not re.search(r"^cyclo(?:but|pent|hept)a-1(?:,3)*(?:,5)?-", parent_name):
        return None
    for size, root in sorted(ROOTS.items(), key=lambda item: -len(item[1])):
        if size not in {4, 5, 7}:
            continue
        if parent_name.startswith("cyclo" + root):
            return size
    return None


def _max_nonoxo_ring_double_bond_locs(ring_size: int, oxo_atoms: set[int]) -> list[int]:
    edges = []
    for loc in range(1, ring_size + 1):
        a = loc
        b = loc + 1 if loc < ring_size else 1
        if a in oxo_atoms or b in oxo_atoms:
            continue
        edges.append((loc, a, b))
    best: tuple[int, tuple[int, ...]] | None = None

    def search(index: int, used_atoms: set[int], locs: list[int]) -> None:
        nonlocal best
        if index >= len(edges):
            key = (-len(locs), tuple(locs))
            if best is None or key < best:
                best = key
            return
        loc, a, b = edges[index]
        search(index + 1, used_atoms, locs)
        if a not in used_atoms and b not in used_atoms:
            locs.append(loc)
            search(index + 1, used_atoms | {a, b}, locs)
            locs.pop()

    search(0, set(), [])
    return list(best[1]) if best is not None else []


def _sulfanium_name(names: Sequence[str]) -> str:
    return _substituted_onium_name(names, "sulfanium")


def _phosphanium_name(names: Sequence[str]) -> str:
    return _substituted_onium_name(names, "phosphanium")


def _azanium_name(names: Sequence[str]) -> str:
    return _substituted_onium_name(names, "azanium")


def _pnictogen_hydride_name(names: Sequence[str], parent: str) -> str:
    return _substituted_onium_name(names, parent)


def _group14_hydride_name(names: Sequence[str], parent: str) -> str:
    return _substituted_onium_name(names, parent)


def _group14_halohydride_name(carbon_names: Sequence[str], halo_names: Sequence[str], parent: str) -> str:
    carbon_text = _substituted_onium_name(carbon_names, "") if carbon_names else ""
    halo_text = _substituted_onium_name(halo_names, "") if halo_names else ""
    return carbon_text + halo_text + parent if carbon_text or halo_text else ""


def _group14_hydroxyhydride_name(carbon_names: Sequence[str], parent: str, hydroxy_count: int) -> str:
    if hydroxy_count < 1:
        return ""
    carbon_text = _substituted_onium_name(carbon_names, "") if carbon_names else ""
    if hydroxy_count == 1:
        stem = parent[:-1] if parent.endswith("e") else parent
        return carbon_text + stem + "ol"
    multiplier = MULT.get(hydroxy_count, "")
    return carbon_text + parent + multiplier + "ol" if multiplier else ""


def _substituted_onium_name(names: Sequence[str], parent: str) -> str:
    counts = Counter(names)
    terms = []
    for name in sorted(counts):
        count = counts[name]
        if count > 1:
            name = _multiplied_prefix_unit_name(name)
        is_complex = _needs_substituent_parentheses(name)
        if count > 1 and (is_complex or _needs_repeated_onium_substituent_parentheses(name, count)):
            multiplier = _COMPLEX_MULTIPLIERS.get(count, "")
            term = f"{multiplier}({name})" if multiplier else ""
        else:
            term = (MULT.get(count, "") if count > 1 else "") + name
        if not term:
            return ""
        if count == 1 and is_complex:
            term = f"({term})"
        terms.append((is_complex, name, term))
    parts = [term for _is_complex, _name, term in sorted(terms)]
    return "".join(parts) + parent


def _needs_substituent_parentheses(name: str) -> bool:
    return (
        name[:1].isdigit()
        or "-" in name
        or "," in name
        or name.startswith("(")
        or (name.endswith("silyl") and name != "silyl")
        or _is_cyclic_onium_substituent(name)
        or _is_composite_onium_substituent(name)
    )


def _needs_repeated_onium_substituent_parentheses(name: str, count: int) -> bool:
    return (
        name.endswith(("ylethynyl", "ylmethyl"))
        or _is_cyclic_onium_substituent(name)
        or _repeated_onium_ligand_needs_complex_multiplier(name, count)
    )


def _repeated_onium_ligand_needs_complex_multiplier(name: str, count: int) -> bool:
    if name in {"methyl", "ethyl", "propyl", "butyl", "phenyl", "benzyl"}:
        return False
    return (count == 3 and name == "decyl") or name.endswith(("methyl", "ethyl", "propyl", "butyl"))


_COMPLEX_MULTIPLIERS = {
    2: "bis",
    3: "tris",
    4: "tetrakis",
    5: "pentakis",
    6: "hexakis",
}


def _is_composite_onium_substituent(name: str) -> bool:
    if name.endswith("diazenyl") and name != "diazenyl":
        return True
    return name.endswith("amino") and any(
        name.startswith(multiplier)
        for count, multiplier in MULT.items()
        if count > 1
    )


def _is_cyclic_onium_substituent(name: str) -> bool:
    bare = name[1:-1] if name.startswith("(") and name.endswith(")") else name
    if bare in {"phenyl", "benzyl", "naphthyl"}:
        return True
    if bare.startswith(("cyclo", "bicyclo[", "spiro[")) and bare.endswith("yl"):
        return True
    cyclic_suffixes = (
        "phenyl",
        "benzyl",
        "naphthyl",
        "pyridyl",
        "pyrimidyl",
        "pyridazinyl",
        "pyrazinyl",
        "furyl",
        "thienyl",
        "thenyl",
        "pyrrolyl",
        "pyrazolyl",
        "imidazolyl",
        "triazolyl",
        "tetrazolyl",
        "oxazolyl",
        "isoxazolyl",
        "thiazolyl",
        "isothiazolyl",
        "quinolyl",
        "isoquinolyl",
        "indolyl",
        "indazolyl",
        "benzimidazolyl",
        "benzotriazolyl",
    )
    if bare.endswith(cyclic_suffixes):
        return True
    return bool(
        re.fullmatch(
            r"(?:pyridin|pyrimidin|pyridazin|pyrazin|furan|thiophen|pyrrol|"
            r"pyrazol|imidazol|triazol|tetrazol|oxazol|thiazol|quinolin|"
            r"isoquinolin|indol|indazol|benzimidazol|benzotriazol)-\d+-yl",
            bare,
        )
    )


def _n_halo_atom(mol, nitrogen_idx: int, carbon_idx: int) -> int:
    halo_atoms = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors()
        if nbr.GetIdx() != carbon_idx and nbr.GetSymbol() in HALO
    ]
    return halo_atoms[0] if len(halo_atoms) == 1 else -1


def _n_halo_prefix(mol, nitrogen_idx: int, carbon_idx: int, Chem) -> str:
    atom = mol.GetAtomWithIdx(nitrogen_idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    halo_idx = _n_halo_atom(mol, nitrogen_idx, carbon_idx)
    if halo_idx < 0:
        return ""
    bond = mol.GetBondBetweenAtoms(nitrogen_idx, halo_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    return "N"


def _append_terminal_halo_prefixes(mol, atom_idx: int, parent_idx: int, locant: str, prefixes: list[tuple[str, str]], Chem) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return False
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
        if nbr.GetSymbol() not in HALO or bond.GetBondType() != Chem.BondType.SINGLE:
            return False
        prefixes.append((locant, HALO[nbr.GetSymbol()]))
    return True


def _prefer_acyclic_skeleton_name(name: str) -> bool:
    if (
        name.endswith("-methane")
        and "methylidene" in name
        and any(term in name for term in ("spiro[", "tricyclo[", "bicyclo["))
    ):
        return False
    if name.endswith("-methane") and "hydroxymethylidene" in name and "cyclo" in name:
        return False
    if (
        name.endswith("-methane")
        and "pyrazol-5-yl" in name
        and "azaethyl" in name
        and "oxo" in name
    ):
        return False
    if "phenylidene" in name:
        return False
    if "ylidene" in name:
        return True
    if "imino" in name and any(term in name for term in ("ene", "oxo", "carbonyl")):
        return True
    if "imino" in name and "nitro" in name and "azamethyl" in name:
        return True
    if "imino" in name and "cyano" in name:
        return True
    if "1-oxo-2-" in name and "2-azaethyl" in name:
        return True
    if "diazaprop" in name and "oxabut" in name and "ene" in name:
        return True
    if "diaza" in name and "ene" in name and any(term in name for term in ("carbonyl", "sulfonyl", "nitro")):
        return True
    if "hydroxy" in name and "dioxo" in name and any(term in name for term in ("prop-2-ene", "but-")):
        return True
    if (
        "tetrafluoro" in name
        and "thia" in name
        and "aza" in name
        and "oxo" in name
        and "ene" in name
    ):
        return True
    if "oxopyrazin" in name and "azapropane" in name:
        return True
    if "oxo" in name and "oxaoct-3-ene" in name:
        return True
    if name.count("cyano") >= 2:
        return True
    if "cyano" in name and re.search(r"(?:prop|but|pent|hex)-\d+(?:,\d+)*-ene", name):
        return True
    if any(term in name for term in ("fluoro", "chloro", "bromo", "iodo")) and re.search(
        r"(?:prop|but|pent|hex|hept)-\d+(?:,\d+)*-ene", name
    ):
        return True
    if (
        "cyclo" in name
        and re.search(r"(?:prop|but|pent|hex|hept)-\d+(?:,\d+)*-ene", name)
        and not any(term in name for term in ("bicyclo[", "tricyclo[", "spiro["))
    ):
        return True
    if (
        "amino" in name
        and "carbonyl" in name
        and any(term in name for term in ("carboxy", "hydroxy", "azamethylcarbonyl", "oxo"))
        and not any(term in name for term in ("bicyclo[", "tricyclo[", "spiro["))
        and not any(term in name for term in ("phospha", "bora", "sila"))
    ):
        return True
    if (
        not any(
            term in name
            for term in (
                "bicyclo[",
                "tricyclo[",
                "spiro[",
                "phospha",
                "bora",
                "sila",
                "norbornene",
                "indolizidin",
                "quinuclidine",
                "penam",
                "cephem",
                "azacycloprop",
            )
        )
        and (
            "carbonyl" in name
            or ("hydroxy" in name and "oxa" in name and "ium" not in name)
            or ("amino" in name and "carboxy" in name)
        )
    ):
        return True
    if (
        not any(
            term in name
            for term in (
                "bicyclo[",
                "tricyclo[",
                "spiro[",
                "phospha",
                "bora",
                "sila",
                "norbornene",
                "indolizidin",
                "quinuclidine",
                "penam",
                "cephem",
                "azacycloprop",
                "benzimidazol",
                "oxacyclobut",
                "pyrrolizidine",
                "decalin",
                "diazaeth",
            )
        )
        and "ium" not in name
        and (
            ("phenyl" in name and re.search(r"(?:prop|but|pent|hex|hept|oct|non)-\d+(?:,\d+)*-ene", name))
            or ("cyclo" in name and "oxa" in name)
            or ("cyclo" in name and "hydroxy" in name)
            or ("oxa" in name and "phenyl" in name)
            or ("hydroxy" in name and "phenyl" in name)
            or ("amino" in name and "phenyl" in name)
            or (
                "amino" in name
                and re.search(r"pyridin|pyrimidin|pyrazin|thiophen|furan|imidazol|oxazol|thiazol|indole|benz", name)
            )
            or (
                "cyclo" in name
                and not any(term in name for term in ("carbonyl", "hydroxy", "amino", "carboxy", "oxa", "thia", "phenyl"))
            )
        )
    ):
        return True
    if (
        not any(
            term in name
            for term in (
                "bicyclo[",
                "tricyclo[",
                "spiro[",
                "phospha",
                "bora",
                "sila",
                "norbornene",
                "indolizidin",
                "quinuclidine",
                "penam",
                "cephem",
                "azacycloprop",
                "benzimidazol",
                "oxacyclobut",
                "pyrrolizidine",
                "decalin",
                "diazaeth",
            )
        )
        and "ium" not in name
        and (
            ("cyclo" in name and "carboxy" in name)
            or ("cyclo" in name and "amino" in name)
            or (
                "amino" in name
                and not any(
                    term in name
                    for term in (
                        "phenyl",
                        "pyridin",
                        "pyrimidin",
                        "pyrazin",
                        "thiophen",
                        "furan",
                        "imidazol",
                        "oxazol",
                        "thiazol",
                        "indole",
                        "benz",
                        "carbonyl",
                        "carboxy",
                        "hydroxy",
                        "oxa",
                        "thia",
                    )
                )
            )
            or (
                "hydroxy" in name
                and re.search(r"pyridin|pyrimidin|pyrazin|thiophen|furan|imidazol|oxazol|thiazol|indole|benz", name)
            )
            or ("carboxy" in name and "oxa" in name)
        )
    ):
        return True
    if (
        "yne" in name
        and not any(term in name for term in ("bicyclo[", "tricyclo[", "spiro["))
        and any(
            term in name
            for term in (
                "amino",
                "carboxy",
                "carbonyl",
                "cyano",
                "fluoro",
                "chloro",
                "bromo",
                "iodo",
                "hydroxy",
                "imino",
                "phenyl",
            )
        )
    ):
        return True
    if "imino" in name and any(term in name for term in ("cyclo", "phenyl", "pyridin", "imidazol")):
        return True
    if _has_substituted_carboxylato_ethene(name):
        return True
    if _has_main_chain_trifluoro_oxo_alkane(name):
        return True
    if (
        "trifluoro" in name
        and "carbonyl" in name
        and re.search(r"(?:prop|but|pent|hex)-\d+(?:,\d+)*-ene", name)
    ):
        return True
    if any(term in name for term in ("diene", "triene", "but-3-ene")):
        return True
    if "eth-1-ene" in name and any(term in name for term in ("nitro", "difluoro", "cyano")):
        return True
    functional_terms = _functional_term_score(name)
    has_chain_unsaturation = bool(
        re.search(r"(?:^|-)\d+(?:,\d+)*-(?:en|yn)|[a-z]-\d+(?:,\d+)*-(?:en|yn)|(?:en|yn)e(?:$|-)", name)
    )
    return functional_terms >= 3 and has_chain_unsaturation


def _has_substituted_carboxylato_ethene(name: str) -> bool:
    """Prefer the chain parent for substituted carboxylato ethenes.

    Simple ethene-carboxylates and dicarboxylates are already handled better by
    ring/retained fallbacks; this gate is for cinnamate-like carboxylato alkenes
    where the substituted acyclic parent gives OPSIN a faithful chain.
    """

    if "carboxylato" not in name or "eth-1-ene" not in name:
        return False
    name = _strip_leading_descriptors(name)
    if re.search(r"(?:^|-)1-carboxylato-2-(?!2,)", name):
        return True
    if re.search(r"(?:^|-)2-carboxylato-1,1-", name):
        return True
    return bool(re.search(r"(?:^|-)1,2-dicarboxylato-(?:1|2)-", name))


def _strip_leading_descriptors(name: str) -> str:
    return re.sub(r"^(?:\([^)]*\)-)+", "", name)


def _has_main_chain_trifluoro_oxo_alkane(name: str) -> bool:
    if _leading_descriptor_has_alkene_geometry(name):
        return False
    name = _strip_leading_descriptors(name)
    if not name.startswith("1,1,1-trifluoro-") or "oxo" not in name:
        return False
    if re.search(r"(?:^|-)\d+-amino(?:-|$)", name):
        return False
    if "1-phenyl-methyl" in name:
        return False
    if "azabicyclo[3.3.0]" in name or "spiro[" in name:
        return False
    if re.search(r"methyl\d", name):
        return False
    return bool(re.search(r"(?:pent|hex|hept|oct)ane$", name))


def _leading_descriptor_has_alkene_geometry(name: str) -> bool:
    match = re.match(r"^\(([^)]*)\)-", name)
    if not match:
        return False
    return any(part in {"E", "Z"} for part in match.group(1).split(","))


def _functional_term_score(name: str) -> int:
    score = sum(name.count(term) for term in ("imino", "sulfonyl", "carboxy", "carbonyl"))
    multiplier_values = {
        "di": 2,
        "tri": 3,
        "tetra": 4,
        "penta": 5,
        "hexa": 6,
    }
    for match in re.finditer(r"(?:(di|tri|tetra|penta|hexa))?oxo", name):
        score += multiplier_values.get(match.group(1) or "", 1)
    return score
