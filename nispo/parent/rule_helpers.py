from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from itertools import combinations

from ..branch_rules import branch_prefix
from ..constants import HALO, MULT, ROOTS
from ..cycle_utils import ordered_ring, ring_system_extends_core
from ..formatting import format_prefixes as _format_prefixes
from ..formatting import loc_unsat
from ..fused_template_families import (
    generated_fusion_preference as _generated_fusion_preference,
)
from ..molecule_cache import mol_cache_get, mol_cache_set
from ..ring_resource_rules import resource_ring_parent_candidates as _resource_ring_parent_candidates
from ..ring_rules import (
    adjust_ring_parent_for_charge as _adjust_ring_parent_for_charge,
    ring_parent_candidates as _ring_parent_candidates,
)
from ..stereochemistry import _has_assigned_bond_stereo, assigned_atom_stereochemistry
from ..template_access import template_record_locants, template_record_mol, template_record_name

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
    "androst": "androstane",
    "barbiturat": "barbituric acid",
    "barbiturate": "barbituric acid",
    "chol": "cholane",
    "cholest": "cholestane",
    "campest": "campestane",
    "ergost": "ergostane",
    "estr": "estrane",
    "poriferast": "poriferastane",
    "stigmast": "stigmastane",
    "gon": "gonane",
    "pregn": "pregnane",
    "spirost": "spirostane",
}


def _terminal_retained_parent_name(name: str) -> str:
    if name == "pyroglutam":
        return "pyroglutamic acid"
    if name.endswith(" acid"):
        return name
    if name.endswith("acid"):
        return f"{name[:-4]} acid"
    if name.endswith(("ic", "ylic")) and name not in _RETAINED_ACID_PARENT_STEM_EXCEPTIONS:
        return f"{name} acid"
    return _TERMINAL_RETAINED_PARENT_BY_STEM.get(name, name)


_RETAINED_ACID_PARENT_STEM_EXCEPTIONS = {
    "psic",
}


def _generated_exact_parent_sort_key(template: dict) -> tuple:
    name = template_record_name(template)
    base = _fusion_base_name(name)
    base_rank = 0 if base in _BICYCLIC_FUSION_BASES else 1
    return (base_rank, len(name), template.get("source_family", ""), name)


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
    for template in templates:
        template_mol = template_record_mol(template)
        if template.get("stereochemical") and not _has_stereo_evidence(mol, Chem):
            continue
        if require_one_ring_system and not _template_has_one_ring_system(template_mol):
            continue
        if require_fully_aromatic and not _template_is_fully_aromatic(template_mol):
            continue
        if topology_query:
            query = _topology_query(template_mol, Chem)
            if query.GetNumAtoms() > mol.GetNumAtoms():
                continue
            matches = ctx.substruct_matches(query)
        else:
            matches = ctx.template_matches(template)
        for match in matches:
            if reject_extra_core_bonds and _match_has_extra_core_bonds(mol, template, match):
                continue
            yield template, match


def _match_has_extra_core_bonds(mol, template: dict, match: Sequence[int]) -> bool:
    core_atoms = set(match)
    core_bonds = sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in core_atoms and bond.GetEndAtomIdx() in core_atoms
    )
    return core_bonds > template_record_mol(template).GetNumBonds()


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
            prefix = branch_prefix(mol, nidx, mol_idx)
            if not prefix:
                return ""
            prefixes.append((locant, prefix))
    parent_name = _terminal_retained_parent_name(template_record_name(template))
    prefix_text = _format_prefixes(prefixes)
    return prefix_text + "-" + parent_name if prefix_text else parent_name


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


def _skeletal_replacement_parent_name(mol, template: dict, match: Sequence[int]) -> str:
    grouped: dict[str, list[str]] = {}
    for template_idx, mol_idx in enumerate(match):
        symbol = mol.GetAtomWithIdx(mol_idx).GetSymbol()
        if symbol == "C":
            continue
        replacement = _SKELETAL_REPLACEMENT_PREFIXES.get(symbol)
        locant = template_record_locants(template).get(template_idx, "")
        if not replacement or not locant:
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
    return "-".join(parts) + parent_name


def _skeletal_replacement_sort_key(mol, match: Sequence[int], parent_name: str) -> tuple:
    saturation_penalty = _skeletal_replacement_saturation_penalty(mol, match, parent_name)
    spiro_norbornene_penalty = _spiro_norbornene_locant_penalty(parent_name)
    replacement_count = sum(parent_name.count(term) for term in _SKELETAL_REPLACEMENT_PREFIXES.values())
    fusion_locant_penalty = parent_name.count("a-")
    return (
        saturation_penalty,
        spiro_norbornene_penalty,
        fusion_locant_penalty,
        replacement_count,
        len(parent_name),
        parent_name,
    )


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
            and atom.GetSymbol() in {"N", "S"}
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
    name = f"{locants}-{hydro}-{template_record_name(template)}"
    if len(charged_locs) == 1:
        if not charged_has_hydrogen:
            name = _drop_indicated_hydrogen_prefix(name)
        stem = name[:-1] if name.endswith("e") else name
        return f"{stem}-{charged_locs[0]}-ium"
    return name


def _is_hydro_locant(locant: str) -> bool:
    return bool(locant) and locant[0].isdigit() and not locant.endswith("a")


def _partial_hydro_parent_names(mol, template: dict, match: Sequence[int]):
    candidate_locs = []
    scored_hydro_atoms = 0
    scored_hydro_total = 0
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
                candidates.append(((2, count, -scored_hydro_total, 0, ()), f"{hydro}{template_record_name(template)}"))
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
                f"{locant_text}-{hydro}-{template_record_name(template)}",
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
    if "benzimidazole" not in name and template_record_name(template) != "1H-benzimidazole":
        return ()
    reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
    h_locs = sorted(_matched_fused_indicated_hydrogen_locants(mol, template, reverse), key=_locant_sort_key)
    if h_locs == ["1"]:
        return (0,)
    if h_locs:
        return (1, tuple(h_locs))
    return (2,)


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
    core_set = set(match)
    if ring_system_extends_core(mol, core_set):
        return ""
    reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
    parent_name = _terminal_retained_parent_name(parent_name or _adjust_fused_parent_for_charge(mol, template, reverse))
    parent_name = _template_isotope_prefixed_parent_name(mol, template, match, parent_name)
    prefixes = []
    for mol_idx in match:
        template_idx = reverse[mol_idx]
        locant = template_record_locants(template).get(template_idx)
        for nbr in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core_set:
                continue
            if not locant:
                return ""
            prefix = branch_prefix(mol, nidx, mol_idx)
            if not prefix:
                return ""
            prefixes.append((locant, prefix))
    if not prefixes:
        return parent_name
    indicated_h_locs = _matched_fused_indicated_hydrogen_locants(mol, template, reverse)
    suffix_name = _fused_functional_suffix_name(prefixes, parent_name, indicated_h_locs)
    if suffix_name:
        return suffix_name
    parent_name = _drop_indicated_hydrogen_segments(parent_name, prefixes)
    suffix_name = _fused_oxo_suffix_name(prefixes, parent_name, indicated_h_locs)
    if suffix_name:
        return suffix_name
    prefix_text = _format_prefixes(prefixes)
    if not prefix_text:
        return ""
    return prefix_text + "-" + parent_name


def _template_isotope_prefixed_parent_name(mol, template: dict, match: Sequence[int], parent_name: str) -> str:
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
    work = Chem.Mol(mol)
    Chem.AssignStereochemistry(work, cleanIt=True, force=True)
    result = {
        idx: descriptor
        for idx, descriptor in Chem.FindMolChiralCenters(
            work,
            includeUnassigned=False,
            useLegacyImplementation=False,
        )
        if descriptor in {"R", "S"}
    }
    return mol_cache_set(mol, "fresh_atom_stereochemistry", result)


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
        return prefix_text + "-" + name if prefix_text else name
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
        h_loc = sorted(set(indicated_h_locs), key=_locant_sort_key)[0]
        name = f"{parent}-{oxo_locs[0]}({h_loc}H)-one"
        return prefix_text + "-" + name if prefix_text else name
    return ""


def _adjust_fused_parent_for_charge(mol, template: dict, reverse: dict[int, int]) -> str:
    name = _adjust_fused_parent_for_indicated_hydrogen(mol, template, reverse)
    if name in {"quinuclidine", "quinuclidin"}:
        for mol_idx, template_idx in reverse.items():
            locant = template_record_locants(template).get(template_idx)
            atom = mol.GetAtomWithIdx(mol_idx)
            if locant == "1" and atom.GetSymbol() == "N" and atom.GetFormalCharge() > 0:
                return "quinuclidin-1-ium"
    charged_locs = []
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
        if locant and locant.isdigit() and atom.GetSymbol() in {"N", "S"} and atom.GetFormalCharge() > 0:
            charged_locs.append(int(locant))
            if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0:
                charged_has_hydrogen = True
    if len(anion_locs) == 1 and not charged_locs:
        stem = name[:-1] if name.endswith("e") else name
        return f"{stem}-{anion_locs[0]}-ide"
    if len(charged_locs) == 1:
        locant = charged_locs[0]
        if not charged_has_hydrogen:
            name = _drop_indicated_hydrogen_prefix(name)
        stem = name[:-1] if name.endswith("e") else name
        return f"{stem}-{locant}-ium"
    return name


def _drop_indicated_hydrogen_prefix(name: str) -> str:
    if len(name) > 3 and name[0].isdigit() and name[1:3] == "H-":
        return name[3:]
    return name


def _drop_indicated_hydrogen_segments(name: str, prefixes: Sequence[tuple[str, str]]) -> str:
    if "benzimidazole" in name and any(prefix == "oxo" for _, prefix in prefixes):
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
    return bool(re.match(r"(?:pyrazolo|pyrrolo|imidazo)\[[^\]]+\](?:pyridin|pyrimidin)", name))


def _preserve_generated_fused_indicated_hydrogen(name: str) -> bool:
    return bool(re.match(r"\d+H-(?:pyrazolo|pyrrolo|imidazo)\[[^\]]+\](?:pyridin|pyrimidin)", name))


def _named_ring_instance_names(mol, ring: Sequence[int], Chem) -> list[tuple[str, str]]:
    ordered = ordered_ring(mol, ring)
    if not ordered:
        return []
    candidates = _ring_parent_candidates(mol, ordered)
    candidates = _merge_ring_parent_candidates(
        candidates,
        _resource_ring_parent_candidates(mol, ordered, Chem),
    )
    if not candidates:
        return []
    ring_set = set(ordered)
    raw_subs = []
    for ring_idx in ordered:
        atom = mol.GetAtomWithIdx(ring_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set:
                continue
            prefix = branch_prefix(mol, nidx, ring_idx)
            if not prefix:
                return []
            raw_subs.append((ring_idx, prefix))

    best_key = None
    best_name = ""
    best_seq: Sequence[int] = ()
    for parent_name, seq in candidates:
        adjusted_parent = _adjust_ring_parent_for_charge(mol, parent_name, seq, raw_subs)
        if not adjusted_parent:
            continue
        adjusted_parent = _append_ring_parent_isotope_suffixes(mol, adjusted_parent, seq)
        if not raw_subs:
            name = adjusted_parent
            key = ([], name)
            if best_key is None or key < best_key:
                best_key = key
                best_name = name
                best_seq = seq
            continue
        pos = {idx: i + 1 for i, idx in enumerate(seq)}
        prefixes = [
            (str(pos[idx]), prefix)
            for idx, prefix in raw_subs
            if not (
                adjusted_parent in {"1,3,4-dioxazol-2-one", "azet-2-one"} and pos[idx] == 2 and prefix == "oxo"
            )
        ]
        locants = sorted(pos[idx] for idx, _ in raw_subs)
        prefix_text = _format_prefixes(prefixes)
        if not prefix_text:
            continue
        name = prefix_text + "-" + adjusted_parent
        key = (locants, 1, name)
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
            best_seq = seq
        retained_benzene_name = _benzene_retained_parent_name(prefixes, adjusted_parent)
        if retained_benzene_name:
            retained_key = (locants, -1, len(retained_benzene_name), retained_benzene_name)
            if best_key is None or retained_key < best_key:
                best_key = retained_key
                best_name = retained_benzene_name
                best_seq = seq
        suffix_name = (
            ""
            if adjusted_parent in {"1,3,4-dioxazol-2-one", "azet-2-one"}
            else _single_ring_oxo_suffix_name(prefixes, adjusted_parent)
        )
        if suffix_name:
            suffix_key = (locants, 0, suffix_name)
            if best_key is None or suffix_key < best_key:
                best_key = suffix_key
                best_name = suffix_name
                best_seq = seq
        functional_suffix_name = _single_ring_functional_suffix_name(prefixes, adjusted_parent)
        if functional_suffix_name:
            suffix_key = (locants, 0, functional_suffix_name)
            if best_key is None or suffix_key < best_key:
                best_key = suffix_key
                best_name = functional_suffix_name
                best_seq = seq
    if not best_name:
        return []
    return [(best_name, _single_ring_stereo_prefix(mol, best_seq, Chem, name=best_name))]


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
        if loc != "1":
            continue
        retained_parent = _BENZENE_RETAINED_PARENT_BY_PREFIX.get(prefix)
        if not retained_parent:
            continue
        remaining = list(prefixes[:index]) + list(prefixes[index + 1 :])
        prefix_text = _format_prefixes(remaining)
        name = prefix_text + retained_parent if prefix_text else retained_parent
        name = _benzene_retained_parent_alias_name(name)
        if len(name) >= len(original_name):
            continue
        rank = _BENZENE_RETAINED_PARENT_RANK[prefix]
        key = (rank, len(name), name)
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
    return best_name


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
        if "formyl" in name and alias in {"guaiacol", "mequinol", "o-cresol", "m-cresol", "p-cresol"}:
            continue
        name = name.replace(verbose, alias)
    return name


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
    "cyano",
    "dimethylamino",
    "ethyl",
    "ethylamino",
    "ethoxy",
    "fluoro",
    "hydroxy",
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
    prefix = branch_prefix(mol, external[0], atom_idx)
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
    return ""


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
    """Use OPSIN's relative cis prefix for pseudoasymmetric bicyclo parents."""

    if "bicyclo[" not in name or "spiro[" in name:
        return ""
    seq_set = set(seq)
    work = Chem.Mol(mol)
    Chem.AssignStereochemistry(work, cleanIt=True, force=True)
    centers = [
        (idx, descriptor)
        for idx, descriptor in Chem.FindMolChiralCenters(
            work,
            includeUnassigned=False,
            useLegacyImplementation=False,
        )
        if descriptor in {"r", "s"} and idx in seq_set
    ]
    if len(centers) != 2:
        return ""
    return "cis"


def _single_ring_oxo_suffix_name(prefixes: Sequence[tuple[str, str]], parent_name: str) -> str:
    oxo_locs = [loc for loc, prefix in prefixes if prefix == "oxo" and loc.isdigit()]
    if not oxo_locs:
        return ""
    remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "oxo"]
    prefix_text = _format_prefixes(remaining)
    parent = _oxo_adjusted_unsaturated_carbocycle_parent(parent_name, oxo_locs)
    if not parent:
        parent = parent_name[:-1] if parent_name.endswith("e") else parent_name
    locants = ",".join(sorted(oxo_locs, key=int))
    suffix = "one" if len(oxo_locs) == 1 else f"{MULT.get(len(oxo_locs), '')}one"
    name = f"{parent}-{locants}-{suffix}"
    return prefix_text + "-" + name if prefix_text else name


_SINGLE_RING_FUNCTIONAL_SUFFIXES = {
    "1-azamethylcarbonyl": "carboxamide",
    "carbamyl": "carboxamide",
    "carbamoyl": "carboxamide",
    "carboxy": "carboxylic acid",
    "carboxylato": "carboxylate",
    "cyano": "carbonitrile",
}
_STEREO_PREFIX_PRESERVE_MAX_NAME_LENGTH = 152

_FUSED_FUNCTIONAL_SUFFIXES = _SINGLE_RING_FUNCTIONAL_SUFFIXES

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
    if not supports_parent and not (
        supports_ester_parent
        and any(prefix in _SINGLE_RING_ESTER_SUFFIX_OWNERS for _loc, prefix in prefixes)
    ):
        return ""
    best_key = None
    best_name = ""
    for matched, suffix_base, n_substituent in _iter_single_ring_functional_suffix_terms(prefixes, supports_parent):
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
        suffix_name = f"{parent_name}-{locants}-{suffix}"
        if n_substituent:
            suffix_name = f"{n_substituent}-{suffix_name}"
        prefix_text = _format_prefixes(remaining)
        name = prefix_text + "-" + suffix_name if prefix_text else suffix_name
        original_prefix_text = _format_prefixes(prefixes)
        original_name = original_prefix_text + "-" + parent_name if original_prefix_text else parent_name
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
) -> list[tuple[tuple[tuple[str, str], ...], str, str]]:
    if not supports_parent:
        return []
    terms = []
    for prefix, suffix_base in _SINGLE_RING_FUNCTIONAL_SUFFIXES.items():
        matched = tuple((loc, candidate_prefix) for loc, candidate_prefix in prefixes if candidate_prefix == prefix)
        if matched:
            terms.append((matched, suffix_base, ""))
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
        core = f"{parent}-{loc}-{suffix}"
        if n_substituent:
            core = f"{n_substituent}-{core}"
        name = prefix_text + "-" + core if prefix_text else core
        original_prefix_text = _format_prefixes(prefixes)
        original_name = original_prefix_text + "-" + parent_name if original_prefix_text else parent_name
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
    if suffix_base not in {"carboxamide", "carboxylate", "carboxylic acid", "carbonitrile"}:
        return False
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


def _single_ring_ester_suffix_name(prefixes: Sequence[tuple[str, str]], parent_name: str) -> str:
    best_key = None
    best_name = ""
    original_prefix_text = _format_prefixes(prefixes)
    original_name = original_prefix_text + "-" + parent_name if original_prefix_text else parent_name
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
        if parent_name == "benzene" and len(suffix_locs) == 1 and suffix_locs[0] == "1":
            core = prefix_text + "benzoate" if prefix_text else "benzoate"
        else:
            suffix = "carboxylate" if len(suffix_locs) == 1 else "dicarboxylate"
            parent_core = f"{parent_name}-{locants}-{suffix}"
            core = prefix_text + "-" + parent_core if prefix_text else parent_core
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
    if not re.search(r"^cyclo(?:but|hept)a-1(?:,3)*(?:,5)?-", parent_name):
        return None
    for size, root in sorted(ROOTS.items(), key=lambda item: -len(item[1])):
        if size not in {4, 7}:
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
        is_complex = _needs_substituent_parentheses(name)
        if count > 1 and (is_complex or _needs_repeated_onium_substituent_parentheses(name)):
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
        or _is_composite_onium_substituent(name)
    )


def _needs_repeated_onium_substituent_parentheses(name: str) -> bool:
    return name.endswith(("ylethynyl", "ylmethyl"))


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
