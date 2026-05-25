from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache

from .graph_model import graph_view


_TemplateIndex = tuple[dict[tuple, tuple[dict, ...]], dict[int, tuple[tuple[dict, dict], ...]]]
_TEMPLATE_INDEX_CACHE: dict[int, tuple[tuple[dict, ...], _TemplateIndex]] = {}
_MAX_TEMPLATE_ALIASES_PER_SCAFFOLD = 5


def candidate_templates(
    mol,
    templates: tuple[dict, ...],
    Chem,
    features: dict | None = None,
    *,
    exact_atom_count: bool = False,
) -> tuple[dict, ...]:
    cache = getattr(mol, "_nispo_candidate_template_cache", None)
    if cache is None:
        cache = {}
        try:
            mol._nispo_candidate_template_cache = cache
        except Exception:
            cache = None
    cache_key = (id(templates), exact_atom_count)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    features = features or molecule_features(mol, Chem)
    pure_ring_templates, other_templates_by_atom_count = _template_index(templates, Chem)
    candidates = []
    seen: set[int] = set()
    for system in features["ring_systems"]:
        key = ring_system_feature_key(system)
        for template in pure_ring_templates.get(key, ()):
            marker = id(template)
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append(template)
    if exact_atom_count:
        atom_count_items = ((features["atom_count"], other_templates_by_atom_count.get(features["atom_count"], ())),)
    else:
        atom_count_items = (
            (atom_count, items)
            for atom_count, items in other_templates_by_atom_count.items()
            if atom_count <= features["atom_count"]
        )
    for _atom_count, entries in atom_count_items:
        for template, template_features in entries:
            if template_features["ring_count"] > features["ring_count"]:
                continue
            if template_features["max_ring_system_size"] > features["max_ring_system_size"]:
                continue
            if any(
                count > features["elements"].get(symbol, 0)
                for symbol, count in template_features["elements"].items()
            ):
                continue
            candidates.append(template)
    candidates.sort(key=lambda item: _template_features(item, Chem)["atom_count"], reverse=True)
    result = tuple(candidates)
    if cache is not None:
        cache[cache_key] = result
    return result


def _template_index(
    templates: tuple[dict, ...], Chem
) -> tuple[dict[tuple, tuple[dict, ...]], dict[int, tuple[tuple[dict, dict], ...]]]:
    cache_key = id(templates)
    cached = _TEMPLATE_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] is templates:
        return cached[1]
    pure_ring_templates: dict[tuple, list[dict]] = {}
    other_templates_by_atom_count: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for template in templates:
        template_features = _template_features(template, Chem)
        if template_features["all_atoms_in_one_ring_system"]:
            pure_ring_templates.setdefault(ring_system_feature_key(template_features), []).append(template)
        else:
            other_templates_by_atom_count[template_features["atom_count"]].append((template, template_features))
    indexed = (
        {key: _preferred_templates_by_scaffold(value, Chem) for key, value in pure_ring_templates.items()},
        {
            key: _preferred_template_entries_by_scaffold(value, Chem)
            for key, value in other_templates_by_atom_count.items()
        },
    )
    _TEMPLATE_INDEX_CACHE[cache_key] = (templates, indexed)
    return indexed


def _preferred_templates_by_scaffold(templates: list[dict], Chem) -> tuple[dict, ...]:
    if not templates:
        return ()
    grouped: dict[str, list[dict]] = {}
    for template in templates:
        grouped.setdefault(_template_scaffold_key(template, Chem), []).append(template)
    selected = []
    for group in grouped.values():
        selected.extend(group[:_MAX_TEMPLATE_ALIASES_PER_SCAFFOLD])
    return tuple(selected)


def _preferred_template_entries_by_scaffold(entries: list[tuple[dict, dict]], Chem) -> tuple[tuple[dict, dict], ...]:
    if not entries:
        return ()
    grouped: dict[str, list[tuple[dict, dict]]] = {}
    for template, template_features in entries:
        grouped.setdefault(_template_scaffold_key(template, Chem), []).append((template, template_features))
    selected = []
    for group in grouped.values():
        selected.extend(group[:_MAX_TEMPLATE_ALIASES_PER_SCAFFOLD])
    return tuple(selected)


def _template_scaffold_key(template: dict, Chem) -> str:
    cached = template.get("_nispo_scaffold_key")
    if cached is not None:
        return cached
    key = Chem.MolToSmiles(template["mol"], canonical=True, isomericSmiles=False)
    try:
        template["_nispo_scaffold_key"] = key
    except Exception:
        pass
    return key


def molecule_features(mol, Chem) -> dict:
    cached = getattr(mol, "_nispo_molecule_features", None)
    if cached is not None:
        return cached
    view = graph_view(mol, Chem)
    ring_systems = view.ring_systems
    ring_system_features = []
    for system in ring_systems:
        system_rings = [set(ring) for ring in view.atom_rings if set(ring) <= system]
        ring_system_features.append(
            {
                "atom_count": len(system),
                "ring_count": len(system_rings),
                "elements": Counter(view.symbols[idx] for idx in system),
                "graph_signature": graph_signature_for_atom_set(mol, system, Chem),
                "element_graph_signature": element_graph_signature_for_atom_set(mol, system, Chem),
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
        "ring_systems": tuple(ring_system_features),
        "all_atoms_in_one_ring_system": bool(ring_systems)
        and any(len(system) == mol.GetNumAtoms() for system in ring_systems),
    }
    try:
        mol._nispo_molecule_features = features
    except Exception:
        pass
    return features


def ring_system_feature_key(features: dict) -> tuple:
    return (
        features["atom_count"],
        features["ring_count"],
        tuple(sorted(features["elements"].items())),
    )


def graph_signature_for_atom_set(mol, atom_indices, Chem) -> str:
    """Canonical graph-shape key for a ring system, ignoring atom labels and bond order."""

    atom_indices = tuple(sorted(atom_indices))
    if not atom_indices:
        return ""
    atom_set = set(atom_indices)
    rw_mol = Chem.RWMol()
    index_map = {old_idx: rw_mol.AddAtom(Chem.Atom("C")) for old_idx in atom_indices}
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in atom_set and end in atom_set:
            rw_mol.AddBond(index_map[begin], index_map[end], Chem.BondType.SINGLE)
    return Chem.MolToSmiles(rw_mol.GetMol(), canonical=True, isomericSmiles=False)


def element_graph_signature_for_atom_set(mol, atom_indices, Chem) -> str:
    """Canonical ring-system key preserving atom labels but ignoring bond order.

    Generated fused templates can share the same carbon-skeleton graph and atom
    counts while placing heteroatoms at different ring positions. OPSIN accepts
    the position-specific fused parent, so keeping this finer key lets NISPO
    dispatch directly to the relevant hetero-isomer family instead of probing
    hundreds of unrelated templates. Formal charge is intentionally ignored:
    charge naming is handled as a parent-name adjustment such as ``-ium``.
    """

    atom_indices = tuple(sorted(atom_indices))
    if not atom_indices:
        return ""
    atom_set = set(atom_indices)
    rw_mol = Chem.RWMol()
    index_map = {}
    for old_idx in atom_indices:
        old_atom = mol.GetAtomWithIdx(old_idx)
        atom = Chem.Atom(old_atom.GetAtomicNum())
        index_map[old_idx] = rw_mol.AddAtom(atom)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in atom_set and end in atom_set:
            rw_mol.AddBond(index_map[begin], index_map[end], Chem.BondType.SINGLE)
    return Chem.MolToSmiles(rw_mol.GetMol(), canonical=True, isomericSmiles=False)


@lru_cache(maxsize=None)
def template_features_for_mol(template_mol, Chem) -> dict:
    return molecule_features(template_mol, Chem)


def _template_features(template: dict, Chem) -> dict:
    cached = template.get("_nispo_features")
    if cached is not None:
        return cached
    features = template_features_for_mol(template["mol"], Chem)
    try:
        template["_nispo_features"] = features
    except Exception:
        pass
    return features


def max_ring_system_size(mol, Chem=None) -> int:
    return max((len(system) for system in ring_systems_for_mol(mol, Chem)), default=0)


def ring_systems_for_mol(mol, Chem=None) -> tuple[frozenset[int], ...]:
    return graph_view(mol, Chem).ring_systems
