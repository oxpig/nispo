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


def _generated_exact_parent_sort_key(template: dict) -> tuple:
    name = template_record_name(template)
    base = _fusion_base_name(name)
    base_rank = 0 if base in _BICYCLIC_FUSION_BASES else 1
    return (base_rank, len(name), template.get("source_family", ""), name)


def _single_ring_candidate_sort_key(name: str) -> tuple:
    charged_parent_rank = 0 if re.search(r"(?:^|-)[^-]+-\d+-ium(?:$|-)", name) else 1
    ylidene_rank = 0 if "ylidene" in name else 1
    aromatic_parent_rank = 0 if _has_aromatic_single_ring_parent_stem(name) else 1
    functional_unsaturation_rank = 0 if _has_functional_unsaturated_substituent(name) else 1
    return (charged_parent_rank, ylidene_rank, aromatic_parent_rank, functional_unsaturation_rank, len(name), name)


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


def _fusion_base_name(name: str) -> str:
    if "]" not in name:
        return name
    return name.rsplit("]", 1)[1]


def _partial_hydro_fusion_preference(name: str) -> tuple:
    key = _generated_fusion_preference(name)
    spiro_penalty = _partial_hydro_spiro_alias_penalty(name)
    if len(key) < 5:
        return (spiro_penalty, key)
    base_rank, letter_rank, locants, component_rank, full_name = key
    return (spiro_penalty, base_rank, -letter_rank, locants, component_rank, full_name)


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
    prefix_text = _format_prefixes(prefixes)
    return prefix_text + "-" + template_record_name(template) if prefix_text else template_record_name(template)


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
        query.AddAtom(Chem.AtomFromSmarts("[#6,#7,#8,#16]"))
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
    return "-".join(parts) + template_record_name(template)


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
    if "indan" in parent_name and _has_nonaromatic_core_imine(mol, core_atoms):
        return 2
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
    "N": "aza",
    "O": "oxa",
    "S": "thia",
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
    name = f"{locants}-{hydro}-{template_record_name(template)}"
    if len(charged_locs) == 1:
        if not charged_has_hydrogen:
            name = _drop_indicated_hydrogen_prefix(name)
        stem = name[:-1] if name.endswith("e") else name
        return f"{stem}-{charged_locs[0]}-ium"
    return name


def _partial_hydro_parent_names(mol, template: dict, match: Sequence[int]):
    candidate_locs = []
    for template_idx, mol_idx in enumerate(match):
        template_atom = template_record_mol(template).GetAtomWithIdx(template_idx)
        atom = mol.GetAtomWithIdx(mol_idx)
        if template_atom.GetFormalCharge() != atom.GetFormalCharge():
            return
        locant = template_record_locants(template).get(template_idx, "")
        if not locant or locant.endswith("a"):
            continue
        score = _partial_hydro_atom_score(mol, template_record_mol(template), mol_idx, template_idx, match)
        if score <= 0:
            continue
        candidate_locs.append((locant, score))
    by_locant = {}
    for locant, score in candidate_locs:
        by_locant[locant] = max(score, by_locant.get(locant, 0))
    candidate_locs = sorted(by_locant.items(), key=lambda item: _locant_sort_key(item[0]))
    if len(candidate_locs) < 2:
        return
    candidates = []
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
        _substituent_locant_tuple(name),
        oxo_locs or (99,),
        _benzimidazole_orientation_preference(name),
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
    locants = sorted(
        {int(loc) for loc in re.findall(r"(?<!\d)(\d+)-", name) if int(loc) >= 4},
        reverse=True,
    )
    return tuple(-locant for locant in locants)


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
    parent_name = parent_name or _adjust_fused_parent_for_charge(mol, template, reverse)
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
    parent_name = _drop_indicated_hydrogen_segments(parent_name, prefixes)
    suffix_name = _fused_oxo_suffix_name(prefixes, parent_name)
    if suffix_name:
        return suffix_name
    prefix_text = _format_prefixes(prefixes)
    if not prefix_text:
        return ""
    return prefix_text + "-" + parent_name


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
        return loc_unsat(parent, ene_locs, "ene")
    return parent + "ane"


def _natural_product_extra_double_bond_locs(mol, Chem, template: dict, match: Sequence[int]) -> list[int]:
    locants = template_record_locants(template)
    template_mol = template_record_mol(template)
    ene_locs = set()
    for bond in template_mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        mol_bond = mol.GetBondBetweenAtoms(match[begin], match[end])
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
            ene_locs.add(min(numeric))
    return sorted(ene_locs)


def _locanted_template_stereo_prefix(mol, Chem, template: dict, match: Sequence[int]) -> str:
    descriptors = _fresh_atom_stereochemistry(mol, Chem)
    if not descriptors:
        return ""
    locants = template_record_locants(template)
    pieces = []
    seen = set()
    for template_idx, mol_idx in enumerate(match):
        descriptor = descriptors.get(mol_idx)
        locant = locants.get(template_idx, "")
        if descriptor is None or not locant.isdigit():
            continue
        key = (int(locant), descriptor)
        if key in seen:
            continue
        seen.add(key)
        pieces.append(key)
    return ",".join(f"{locant}{descriptor}" for locant, descriptor in sorted(pieces))


def _fresh_atom_stereochemistry(mol, Chem) -> dict[int, str]:
    work = Chem.Mol(mol)
    Chem.AssignStereochemistry(work, cleanIt=True, force=True)
    return {
        idx: descriptor
        for idx, descriptor in Chem.FindMolChiralCenters(
            work,
            includeUnassigned=False,
            useLegacyImplementation=False,
        )
        if descriptor in {"R", "S"}
    }


def _fused_oxo_suffix_name(prefixes: Sequence[tuple[str, str]], parent_name: str) -> str:
    oxo_locs = sorted((loc for loc, prefix in prefixes if prefix == "oxo" and loc.isdigit()), key=int)
    if parent_name == "pteridine" and oxo_locs == ["4", "7"]:
        remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "oxo"]
        prefix_text = _format_prefixes(remaining)
        name = "3H,8H-pteridine-4,7-dione"
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
    for mol_idx, template_idx in reverse.items():
        locant = template_record_locants(template).get(template_idx)
        atom = mol.GetAtomWithIdx(mol_idx)
        if locant and locant.isdigit() and atom.GetSymbol() in {"N", "S"} and atom.GetFormalCharge() > 0:
            charged_locs.append(int(locant))
            if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0:
                charged_has_hydrogen = True
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

    def replacement(match: re.Match) -> str:
        locant = match.group(2)
        if locant == "2" and "indazole" in name:
            return match.group(0)
        return match.group(1)

    return re.sub(r"(^|-)(\d+)H-", replacement, name)


def _adjust_fused_parent_for_indicated_hydrogen(mol, template: dict, reverse: dict[int, int]) -> str:
    name = template_record_name(template)
    if not name.startswith("1H-"):
        return name
    h_locs = []
    for mol_idx, template_idx in reverse.items():
        atom = mol.GetAtomWithIdx(mol_idx)
        locant = template_record_locants(template).get(template_idx)
        if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0 and locant and locant.isdigit():
            h_locs.append(locant)
    if len(h_locs) != 1 or h_locs[0] == "1":
        return name
    return f"{h_locs[0]}H-{name[3:]}"


def _named_ring_instance_names(mol, ring: Sequence[int], Chem) -> list[tuple[str, str]]:
    ordered = ordered_ring(mol, ring)
    if not ordered:
        return []
    candidates = _ring_parent_candidates(mol, ordered)
    if not candidates:
        candidates = _resource_ring_parent_candidates(mol, ordered, Chem)
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
    if not best_name:
        return []
    return [(best_name, _single_ring_stereo_prefix(mol, best_seq, Chem))]


def _single_ring_stereo_prefix(mol, seq: Sequence[int], Chem) -> str:
    if not seq:
        return ""
    if all(mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 for idx in seq):
        return ""
    seq_set = set(seq)
    tagged_atoms = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    if len(tagged_atoms) < 2:
        return ""
    if any(idx not in seq_set for idx in tagged_atoms):
        return ""
    if _has_assigned_bond_stereo(mol, Chem):
        return ""
    centers = assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
    if len(centers) < 2:
        return ""
    locants = {idx: loc for loc, idx in enumerate(seq, start=1)}
    if any(idx not in locants for idx, _descriptor in centers):
        return ""
    return ",".join(
        f"{locants[idx]}{descriptor}"
        for idx, descriptor in sorted(centers, key=lambda item: locants[item[0]])
    )


def _polycyclic_stereo_prefix(mol, seq: Sequence[int], Chem) -> str:
    if not seq:
        return ""
    seq_set = set(seq)
    tagged_atoms = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    if len(tagged_atoms) < 2:
        return ""
    if any(idx not in seq_set for idx in tagged_atoms):
        return ""
    if _has_assigned_bond_stereo(mol, Chem):
        return ""
    centers = assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
    if len(centers) < 2:
        return ""
    locants = {idx: loc for loc, idx in enumerate(seq, start=1)}
    if any(idx not in locants for idx, _descriptor in centers):
        return ""
    return ",".join(
        f"{locants[idx]}{descriptor}"
        for idx, descriptor in sorted(centers, key=lambda item: locants[item[0]])
    )


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
    return name.endswith("ylmethyl")


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
