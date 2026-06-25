from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache

from .graph_model import graph_view
from .molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set


_TemplateIndex = tuple[
    dict[tuple, tuple[dict, ...]],
    dict[int, tuple[tuple[dict, dict], ...]],
    dict[tuple[int, tuple[tuple[str, int], ...]], tuple[tuple[dict, dict], ...]],
]
_TEMPLATE_INDEX_CACHE: dict[int, tuple[tuple[dict, ...], _TemplateIndex]] = {}
_TEMPLATE_SCAFFOLD_KEY_CACHE: dict[int, tuple[dict, str]] = {}
_TEMPLATE_FEATURE_CACHE: dict[int, tuple[dict, dict]] = {}
_MAX_TEMPLATE_ALIASES_PER_SCAFFOLD = 1


def candidate_templates(
    mol,
    templates: tuple[dict, ...],
    Chem,
    features: dict | None = None,
    *,
    exact_atom_count: bool = False,
) -> tuple[dict, ...]:
    cache = mol_cache_dict(mol, "candidate_template_cache")
    cache_key = (id(templates), exact_atom_count)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    features = features or molecule_features(mol, Chem)
    molecule_charge_counts = features.get("charged_atom_value_counts") or {}
    molecule_ring_system_graph_keys: set[tuple] = set()
    (
        pure_ring_templates,
        other_templates_by_atom_count,
        exact_other_templates_by_atom_elements,
    ) = _template_index(templates, Chem)
    candidates = []
    seen: set[int] = set()
    for system in features["ring_systems"]:
        if system.get("element_graph_signature"):
            key = _ring_system_index_key(system)
        else:
            system = ring_system_with_graph_signatures(mol, system, Chem)
            key = _ring_system_index_key(system)
        for template in pure_ring_templates.get(key, ()):
            marker = id(template)
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append(template)
    if exact_atom_count:
        atom_count = features["atom_count"]
        atom_count_items = (
            (
                atom_count,
                exact_other_templates_by_atom_elements.get(
                    (atom_count, _element_count_key(features["elements"])),
                    (),
                ),
            ),
        )
    else:
        atom_count_items = (
            (atom_count, items)
            for atom_count, items in other_templates_by_atom_count.items()
            if atom_count <= features["atom_count"]
        )
    for _atom_count, entries in atom_count_items:
        for template, template_features in entries:
            template_graph_keys = template_features.get("ring_system_graph_keys") or ()
            if template_graph_keys:
                if not molecule_ring_system_graph_keys:
                    molecule_ring_system_graph_keys = _molecule_ring_system_graph_keys(mol, features, Chem)
                if not any(key in molecule_ring_system_graph_keys for key in template_graph_keys):
                    continue
            if template_features["ring_count"] > features["ring_count"]:
                continue
            if template_features.get("max_ring_system_ring_count", 0) > features.get("max_ring_system_ring_count", 0):
                continue
            if template_features["max_ring_system_size"] > features["max_ring_system_size"]:
                continue
            template_charge_counts = template_features.get("charged_atom_value_counts") or {}
            if template_charge_counts and any(
                count > molecule_charge_counts.get(value, 0)
                for value, count in template_charge_counts.items()
            ):
                continue
            if any(
                count > features["elements"].get(symbol, 0)
                for symbol, count in template_features["elements"].items()
            ):
                continue
            candidates.append(template)
    candidates.sort(key=lambda item: _template_features(item, Chem)["atom_count"], reverse=True)
    result = tuple(candidates)
    cache[cache_key] = result
    return result


def _template_index(
    templates: tuple[dict, ...], Chem
) -> _TemplateIndex:
    cache_key = id(templates)
    cached = _TEMPLATE_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] is templates:
        return cached[1]
    pure_ring_templates: dict[tuple, list[dict]] = {}
    other_templates_by_atom_count: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for template in templates:
        template_features = _template_features(template, Chem)
        if template_features["all_atoms_in_one_ring_system"]:
            key = _ring_system_index_key(template_features)
            pure_ring_templates.setdefault(key, []).append(template)
        else:
            other_templates_by_atom_count[template_features["atom_count"]].append((template, template_features))
    preferred_other_templates_by_atom_count = {
        key: _preferred_template_entries_by_scaffold(value, Chem)
        for key, value in other_templates_by_atom_count.items()
    }
    indexed = (
        {key: _preferred_templates_by_scaffold(value, Chem) for key, value in pure_ring_templates.items()},
        preferred_other_templates_by_atom_count,
        _exact_other_template_entries_by_atom_elements(preferred_other_templates_by_atom_count),
    )
    _TEMPLATE_INDEX_CACHE[cache_key] = (templates, indexed)
    return indexed


def _exact_other_template_entries_by_atom_elements(
    entries_by_atom_count: dict[int, tuple[tuple[dict, dict], ...]],
) -> dict[tuple[int, tuple[tuple[str, int], ...]], tuple[tuple[dict, dict], ...]]:
    indexed: dict[tuple[int, tuple[tuple[str, int], ...]], tuple[tuple[dict, dict], ...]] = {}
    for atom_count, entries in entries_by_atom_count.items():
        grouped: dict[tuple[tuple[str, int], ...], list[tuple[dict, dict]]] = defaultdict(list)
        for template, template_features in entries:
            grouped[_element_count_key(template_features["elements"])].append((template, template_features))
        for element_key, group in grouped.items():
            indexed[(atom_count, element_key)] = tuple(group)
    return indexed


def _element_count_key(elements: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((symbol, count) for symbol, count in elements.items() if count))


def _preferred_templates_by_scaffold(templates: list[dict], Chem) -> tuple[dict, ...]:
    if not templates:
        return ()
    grouped: dict[str, list[dict]] = {}
    for template in templates:
        grouped.setdefault(_template_scaffold_key(template, Chem), []).append(template)
    selected = []
    for group in grouped.values():
        selected.extend(sorted(group, key=_template_alias_preference_key)[:_MAX_TEMPLATE_ALIASES_PER_SCAFFOLD])
    return tuple(selected)


def _preferred_template_entries_by_scaffold(entries: list[tuple[dict, dict]], Chem) -> tuple[tuple[dict, dict], ...]:
    if not entries:
        return ()
    grouped: dict[str, list[tuple[dict, dict]]] = {}
    for template, template_features in entries:
        grouped.setdefault(_template_scaffold_key(template, Chem), []).append((template, template_features))
    selected = []
    for group in grouped.values():
        selected.extend(
            sorted(group, key=lambda item: _template_alias_preference_key(item[0]))[
                :_MAX_TEMPLATE_ALIASES_PER_SCAFFOLD
            ]
        )
    return tuple(selected)


def _template_alias_preference_key(template: dict) -> tuple:
    name = str(template.get("name", ""))
    return (_template_alias_source_priority(template), _template_alias_priority(name), len(name), name)


def _template_alias_source_priority(template: dict) -> int:
    if (
        template.get("source_file") == "high_impact_fused_parent_templates.json"
        and _high_impact_template_alias_can_replace_legacy(template)
    ):
        return -1
    return 0


def _high_impact_template_alias_can_replace_legacy(template: dict) -> bool:
    name = str(template.get("name", "")).lower()
    if any(fragment in name for fragment in _HIGH_IMPACT_ALIAS_REPLACEMENT_RISK_FRAGMENTS):
        return False
    return True


_HIGH_IMPACT_ALIAS_REPLACEMENT_RISK_FRAGMENTS = (
    "purin",
    "purine",
    "pyrazolo",
    "pyrido",
    "pyrimido",
    "pyrimidin",
)


def _template_alias_priority(name: str) -> int:
    if name == "6,7,8,9-tetrahydro-5H-benzo[7]annulene":
        return -1
    return 0


def _template_scaffold_key(template: dict, Chem) -> str:
    cache_key = id(template)
    cached = _TEMPLATE_SCAFFOLD_KEY_CACHE.get(cache_key)
    if cached is not None and cached[0] is template:
        return cached[1]
    key = Chem.MolToSmiles(template["mol"], canonical=True, isomericSmiles=False)
    _TEMPLATE_SCAFFOLD_KEY_CACHE[cache_key] = (template, key)
    return key


def molecule_features(mol, Chem) -> dict:
    cached = mol_cache_get(mol, "molecule_features")
    if cached is not None:
        return cached
    view = graph_view(mol, Chem)
    ring_systems = view.ring_systems
    ring_system_features = []
    for system in ring_systems:
        system_rings = [set(ring) for ring in view.atom_rings if set(ring) <= system]
        ring_system_features.append(
            {
                "atom_indices": tuple(sorted(system)),
                "atom_count": len(system),
                "ring_count": len(system_rings),
                "elements": Counter(view.symbols[idx] for idx in system),
            }
        )
    max_system_size = max((len(system) for system in ring_systems), default=0)
    max_system_ring_count = max(
        (system["ring_count"] for system in ring_system_features),
        default=0,
    )
    features = {
        "atom_count": mol.GetNumAtoms(),
        "ring_count": view.ring_count,
        "max_ring_system_size": max_system_size,
        "max_ring_system_ring_count": max_system_ring_count,
        "elements": view.elements,
        "charged_atom_value_counts": Counter(charge for charge in view.formal_charges if charge),
        "ring_systems": tuple(ring_system_features),
        "all_atoms_in_one_ring_system": bool(ring_systems)
        and any(len(system) == mol.GetNumAtoms() for system in ring_systems),
    }
    return mol_cache_set(mol, "molecule_features", features)


def ring_system_feature_key(features: dict) -> tuple:
    return (
        features["atom_count"],
        features["ring_count"],
        tuple(sorted(features["elements"].items())),
    )


def _ring_system_graph_key(features: dict) -> tuple:
    return (
        *ring_system_feature_key(features),
        features.get("graph_signature", ""),
    )


def _ring_system_element_graph_key(features: dict) -> tuple:
    return (
        *ring_system_feature_key(features),
        features.get("element_graph_signature", ""),
    )


def _ring_system_index_key(features: dict) -> tuple:
    if features.get("element_graph_signature"):
        return _ring_system_element_graph_key(features)
    if features.get("graph_signature"):
        return _ring_system_graph_key(features)
    return ring_system_feature_key(features)


def _molecule_ring_system_graph_keys(mol, features: dict, Chem) -> set[tuple]:
    keys: set[tuple] = set()
    ring_systems = []
    changed = False
    for system in features.get("ring_systems", ()):
        if system.get("element_graph_signature"):
            enriched = system
        else:
            enriched = ring_system_with_graph_signatures(mol, system, Chem)
            changed = True
        ring_systems.append(enriched)
        if enriched.get("graph_signature"):
            keys.add(_ring_system_index_key(enriched))
    if changed:
        features["ring_systems"] = tuple(ring_systems)
    return keys


def ring_system_with_graph_signatures(mol, features: dict, Chem) -> dict:
    if "graph_signature" in features and "element_graph_signature" in features:
        return features
    atom_indices = features.get("atom_indices")
    if atom_indices is None:
        return features
    graph_signature, element_graph_signature = graph_signatures_for_atom_set(mol, atom_indices, Chem)
    features["graph_signature"] = graph_signature
    features["element_graph_signature"] = element_graph_signature
    return features


def ring_system_with_graph_signature(mol, features: dict, Chem) -> dict:
    if "graph_signature" in features:
        return features
    atom_indices = features.get("atom_indices")
    if atom_indices is None:
        return features
    features["graph_signature"] = graph_signature_for_atom_set(mol, atom_indices, Chem)
    return features


def graph_signature_for_atom_set(mol, atom_indices, Chem) -> str:
    """Canonical graph-shape key for a ring system, ignoring atom labels and bond order."""

    atom_indices = tuple(sorted(atom_indices))
    if not atom_indices:
        return ""
    cache = mol_cache_dict(mol, "graph_signature_for_atom_set")
    cached = cache.get(atom_indices)
    if cached is not None:
        return cached
    atom_set = set(atom_indices)
    graph_mol = Chem.RWMol()
    graph_index_map = {old_idx: graph_mol.AddAtom(Chem.Atom("C")) for old_idx in atom_indices}
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in atom_set and end in atom_set:
            graph_mol.AddBond(graph_index_map[begin], graph_index_map[end], Chem.BondType.SINGLE)
    signature = Chem.MolToSmiles(graph_mol.GetMol(), canonical=True, isomericSmiles=False)
    cache[atom_indices] = signature
    return signature


def element_graph_signature_for_atom_set(mol, atom_indices, Chem) -> str:
    """Canonical ring-system key preserving atom labels but ignoring bond order.

    Generated fused templates can share the same carbon-skeleton graph and atom
    counts while placing heteroatoms at different ring positions. OPSIN accepts
    the position-specific fused parent, so keeping this finer key lets NISPO
    dispatch directly to the relevant hetero-isomer family instead of probing
    hundreds of unrelated templates. Formal charge is intentionally ignored:
    charge naming is handled as a parent-name adjustment such as ``-ium``.
    """

    return graph_signatures_for_atom_set(mol, atom_indices, Chem)[1]


def graph_signatures_for_atom_set(mol, atom_indices, Chem) -> tuple[str, str]:
    """Canonical graph-shape keys for a ring system, ignoring bond order."""

    atom_indices = tuple(sorted(atom_indices))
    if not atom_indices:
        return "", ""
    cache = mol_cache_dict(mol, "graph_signatures_for_atom_set")
    cached = cache.get(atom_indices)
    if cached is not None:
        return cached
    atom_set = set(atom_indices)
    graph_mol = Chem.RWMol()
    element_mol = Chem.RWMol()
    graph_index_map = {old_idx: graph_mol.AddAtom(Chem.Atom("C")) for old_idx in atom_indices}
    element_index_map = {}
    for old_idx in atom_indices:
        old_atom = mol.GetAtomWithIdx(old_idx)
        atom = Chem.Atom(old_atom.GetAtomicNum())
        element_index_map[old_idx] = element_mol.AddAtom(atom)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in atom_set and end in atom_set:
            graph_mol.AddBond(graph_index_map[begin], graph_index_map[end], Chem.BondType.SINGLE)
            element_mol.AddBond(element_index_map[begin], element_index_map[end], Chem.BondType.SINGLE)
    signatures = (
        Chem.MolToSmiles(graph_mol.GetMol(), canonical=True, isomericSmiles=False),
        Chem.MolToSmiles(element_mol.GetMol(), canonical=True, isomericSmiles=False),
    )
    cache[atom_indices] = signatures
    mol_cache_dict(mol, "graph_signature_for_atom_set")[atom_indices] = signatures[0]
    return signatures


@lru_cache(maxsize=None)
def template_features_for_mol(template_mol, Chem) -> dict:
    return molecule_features(template_mol, Chem)


def _template_features(template: dict, Chem) -> dict:
    cache_key = id(template)
    cached = _TEMPLATE_FEATURE_CACHE.get(cache_key)
    if cached is not None and cached[0] is template:
        return cached[1]
    features = dict(template_features_for_mol(template["mol"], Chem))
    enriched_ring_systems = tuple(
        ring_system_with_graph_signatures(template["mol"], dict(system), Chem)
        for system in features.get("ring_systems", ())
    )
    features["ring_systems"] = enriched_ring_systems
    features["ring_system_graph_keys"] = tuple(
        _ring_system_index_key(system)
        for system in enriched_ring_systems
        if system.get("graph_signature")
    )
    if features.get("all_atoms_in_one_ring_system") and enriched_ring_systems:
        features["graph_signature"] = enriched_ring_systems[0].get("graph_signature", "")
        features["element_graph_signature"] = enriched_ring_systems[0].get("element_graph_signature", "")
    _TEMPLATE_FEATURE_CACHE[cache_key] = (template, features)
    return features


def max_ring_system_size(mol, Chem=None) -> int:
    return max((len(system) for system in ring_systems_for_mol(mol, Chem)), default=0)


def ring_systems_for_mol(mol, Chem=None) -> tuple[frozenset[int], ...]:
    return graph_view(mol, Chem).ring_systems
