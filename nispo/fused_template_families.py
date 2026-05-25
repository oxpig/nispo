from __future__ import annotations

from functools import lru_cache
import json
from importlib import resources
import re

from .fused_rules import build_fused_templates
from .resource_templates import build_generated_fused_parent_templates_from_records
from .scaffold_catalog import generated_fused_parent_templates, parent_scaffold_catalog
from .template_index import ring_system_feature_key, molecule_features


_FUSION_NAME_RE = re.compile(r"^(.+)\[([^]]+)-([a-z])\](.+)$")
_MAX_GENERATED_RECORDS_PER_GRAPH = 256
_MAX_GENERATED_ALIASES_PER_SCAFFOLD = 2

_FUSION_BASE_PRIORITY = {
    name: index
    for index, name in enumerate(
        (
            "benzene",
            "pyridine",
            "pyrimidine",
            "pyrazine",
            "pyridazine",
            "s-triazine",
            "as-triazine",
            "v-triazine",
            "furan",
            "thiophene",
            "pyrrole",
            "pyrazole",
            "imidazole",
            "triazole",
            "tetrazole",
            "oxazole",
            "isoxazole",
            "thiazole",
            "isothiazole",
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
        )
    )
}


@lru_cache(maxsize=8)
def curated_fused_templates(Chem) -> tuple[dict, ...]:
    catalog = parent_scaffold_catalog(Chem)
    return (
        build_fused_templates(Chem)
        + catalog.static_retained
        + catalog.opsin_retained_fused
        + catalog.fusion_component_parents
    )


@lru_cache(maxsize=8)
def preferred_generated_fused_templates(Chem) -> tuple[dict, ...]:
    return _preferred_generated_templates(generated_fused_parent_templates(Chem), Chem, per_skeleton=3)


@lru_cache(maxsize=8)
def hydrocarbon_replacement_parent_templates(Chem) -> tuple[dict, ...]:
    templates = (
        generated_fused_parent_templates(Chem)
        + parent_scaffold_catalog(Chem).opsin_retained_fused
        + parent_scaffold_catalog(Chem).fusion_component_parents
        + parent_scaffold_catalog(Chem).static_retained
    )
    selected = []
    seen: set[tuple[str, str]] = set()
    for template in templates:
        features = molecule_features(template["mol"], Chem)
        if features["ring_count"] < 2 or set(features["elements"]) != {"C"}:
            continue
        key = (template["name"], Chem.MolToSmiles(template["mol"], canonical=True))
        if key in seen:
            continue
        seen.add(key)
        selected.append(template)
    return tuple(sorted(selected, key=lambda item: (item["mol"].GetNumAtoms(), item["name"])))


@lru_cache(maxsize=512)
def hydrocarbon_replacement_parent_templates_for_graph(
    atom_count: int, ring_count: int, graph_signature: str, Chem
) -> tuple[dict, ...]:
    feature_key = json.dumps((atom_count, ring_count, (("C", atom_count),)), separators=(",", ":"))
    records = _generated_fused_template_index().get(feature_key, ())
    templates = build_generated_fused_parent_templates_from_records(
        Chem,
        records,
        source_file="generated_fused_template_index.json",
    )
    retained = tuple(
        template
        for template in (
            parent_scaffold_catalog(Chem).opsin_retained_fused
            + parent_scaffold_catalog(Chem).fusion_component_parents
            + parent_scaffold_catalog(Chem).static_retained
        )
        if template["mol"].GetNumAtoms() == atom_count
    )
    selected = []
    seen: set[tuple[str, str]] = set()
    for template in templates + retained:
        features = molecule_features(template["mol"], Chem)
        if features["ring_count"] != ring_count or set(features["elements"]) != {"C"}:
            continue
        if not any(system["graph_signature"] == graph_signature for system in features["ring_systems"]):
            continue
        key = (template["name"], Chem.MolToSmiles(template["mol"], canonical=True))
        if key in seen:
            continue
        seen.add(key)
        selected.append(template)
    return tuple(sorted(selected, key=generated_template_key))


def candidate_generated_fused_templates(
    mol, Chem, features: dict | None = None, *, graph_filter: bool = True
) -> tuple[dict, ...]:
    cache_attr = "_nispo_generated_fused_templates" if graph_filter else "_nispo_generated_fused_templates_broad"
    cached = getattr(mol, cache_attr, None)
    if cached is not None:
        return cached
    features = features or molecule_features(mol, Chem)
    candidates = []
    seen: set[int] = set()
    for system in features["ring_systems"]:
        key = json.dumps(ring_system_feature_key(system), separators=(",", ":"))
        graph_key = system.get("graph_signature", "")
        element_graph_key = system.get("element_graph_signature", "") if graph_filter else ""
        for template in _generated_templates_for_key(
            key,
            graph_key if graph_filter else "",
            element_graph_key,
            graph_filter,
            Chem,
        ):
            marker = id(template)
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append(template)
    candidates.sort(key=generated_template_key)
    result = tuple(candidates)
    try:
        setattr(mol, cache_attr, result)
    except Exception:
        pass
    return result


@lru_cache(maxsize=None)
def generated_fusion_preference(name: str) -> tuple:
    match = _FUSION_NAME_RE.match(name)
    if match is None:
        return (999, 999, (), 999, name)
    component, locant_text, letter, base = match.groups()
    base_rank = _FUSION_BASE_PRIORITY.get(base, 500)
    locants = _fusion_descriptor_locants(locant_text)
    letter_rank = ord(letter) - ord("a")
    component_rank = _component_alias_penalty(component)
    return (base_rank, letter_rank, locants, component_rank, name)


def generated_template_key(template: dict) -> tuple:
    return (generated_fusion_preference(template["name"]), template["name"])


def _preferred_generated_templates(templates: tuple[dict, ...], Chem, per_skeleton: int) -> tuple[dict, ...]:
    grouped: dict[str, list[dict]] = {}
    for template in templates:
        key = Chem.MolToSmiles(template["mol"], canonical=True)
        grouped.setdefault(key, []).append(template)
    preferred = []
    for items in grouped.values():
        preferred.extend(sorted(items, key=generated_template_key)[:per_skeleton])
    return tuple(sorted(preferred, key=generated_template_key))


@lru_cache(maxsize=1)
def _generated_fused_template_index() -> dict[str, tuple[dict, ...]]:
    try:
        path = resources.files(__package__).joinpath("resources", "generated_fused_template_index.json")
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(key): tuple(value) for key, value in raw.items()}


@lru_cache(maxsize=1)
def _generated_fused_template_graph_index() -> dict[str, dict[str, tuple[dict, ...]]]:
    try:
        path = resources.files(__package__).joinpath("resources", "generated_fused_template_graph_index.json")
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        str(feature_key): {str(graph_key): tuple(records) for graph_key, records in graph_buckets.items()}
        for feature_key, graph_buckets in raw.items()
    }


@lru_cache(maxsize=1)
def _generated_fused_template_element_graph_index() -> dict[str, dict[str, dict[str, tuple[dict, ...]]]]:
    try:
        path = resources.files(__package__).joinpath(
            "resources", "generated_fused_template_element_graph_index.json"
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        str(feature_key): {
            str(graph_key): {
                str(element_key): tuple(records)
                for element_key, records in element_buckets.items()
            }
            for graph_key, element_buckets in graph_buckets.items()
        }
        for feature_key, graph_buckets in raw.items()
    }


@lru_cache(maxsize=1024)
def _generated_templates_for_key(
    key: str, graph_key: str, element_graph_key: str, graph_filter: bool, Chem
) -> tuple[dict, ...]:
    records = _generated_records_for_graph(key, graph_key, element_graph_key, graph_filter, Chem)
    return build_generated_fused_parent_templates_from_records(
        Chem,
        records,
        source_file="generated_fused_template_index.json",
    )


@lru_cache(maxsize=1024)
def _generated_records_for_graph(
    key: str, graph_key: str, element_graph_key: str, graph_filter: bool, Chem
) -> tuple[dict, ...]:
    if element_graph_key:
        records = (
            _generated_fused_template_element_graph_index()
            .get(key, {})
            .get(graph_key, {})
            .get(element_graph_key)
        ) or ()
        return _preferred_records_per_scaffold(records)
    if graph_filter:
        return ()
    graph_records = _generated_fused_template_graph_index().get(key, {}).get(graph_key)
    if graph_records is not None:
        return _preferred_records_per_scaffold(graph_records[:_MAX_GENERATED_RECORDS_PER_GRAPH])
    if graph_key:
        return ()
    return _preferred_records_per_scaffold(
        _generated_fused_template_index().get(key, ())[:_MAX_GENERATED_RECORDS_PER_GRAPH]
    )


def _preferred_records_per_scaffold(records: tuple[dict, ...]) -> tuple[dict, ...]:
    """Collapse OPSIN alias fan-out before materializing generated templates.

    OPSIN's fusion resources contain many accepted names for the same scaffold.
    That breadth is useful for parsing, but the inverse generator only needs a
    small deterministic set of preferred spellings per generated graph. Keeping
    this pruning at the record layer avoids creating RDKit query molecules and
    running substructure searches for hundreds of equivalent aliases.
    """

    if not records:
        return ()
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record.get("smiles", "")), []).append(record)
    selected = []
    for scaffold_records in grouped.values():
        selected.extend(
            sorted(
                scaffold_records,
                key=lambda record: generated_fusion_preference(str(record.get("name", ""))),
            )[:_MAX_GENERATED_ALIASES_PER_SCAFFOLD]
        )
    return tuple(
        sorted(
            selected,
            key=lambda record: generated_fusion_preference(str(record.get("name", ""))),
        )
    )


def _fusion_descriptor_locants(text: str) -> tuple:
    values = []
    for part in text.split(","):
        digits = "".join(ch for ch in part if ch.isdigit())
        suffix = "".join(ch for ch in part if ch.isalpha())
        if digits:
            values.append((int(digits), suffix))
        else:
            values.append((999, suffix or part))
    return tuple(values)


def _component_alias_penalty(component: str) -> tuple:
    expanded_penalty = 1 if component.endswith(("ano", "eno", "ino", "olo")) else 0
    return (expanded_penalty, len(component), component)
