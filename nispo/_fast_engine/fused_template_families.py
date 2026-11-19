from __future__ import annotations

from functools import lru_cache
import json
from importlib import resources
import re

from .fused_rules import build_fused_templates
from .molecule_cache import mol_cache_get, mol_cache_set
from .resource_io import read_json_resource
from .resource_templates import build_generated_fused_parent_templates_from_records
from .scaffold_catalog import generated_fused_parent_templates
from .template_store import template_store
from .template_access import template_record_mol, template_record_name
from .template_index import (
    intern_template_tuple,
    molecule_features,
    ring_system_feature_key,
    ring_system_with_graph_signatures,
)


_FUSION_NAME_RE = re.compile(r"^(.+)\[([^]]+)-([a-z])\](.+)$")
_MAX_GENERATED_RECORDS_PER_GRAPH = 256
_MAX_GENERATED_ALIASES_PER_SCAFFOLD = 1
_MAX_ORIENTATION_ALIASES_PER_SCAFFOLD = 1
_MAX_TARGETED_GENERATED_ALIASES_PER_SCAFFOLD = 3
_MAX_TARGETED_ORIENTATION_ALIASES_PER_SCAFFOLD = 3
_MAX_RETAINED_TOPOLOGY_REPLACEMENT_TEMPLATES_PER_GRAPH = 24

_TARGETED_GENERATED_FUSED_ALIAS_KEYS = frozenset(
    (
        (13, 3, (("C", 10), ("N", 3)), "C1CCC2C(C1)CCC1CCCC12", "C1CCC2C(C1)NCC1NCNC12"),
        (9, 2, (("C", 6), ("N", 3)), "C1CCC2CCCC2C1", "C1CNC2NCCC2N1"),
        (11, 3, (("C", 10), ("O", 1)), "C1CCC2C(C1)CCC21CC1", "C1CCC2C(C1)OCC21CC1"),
    )
)
_TARGETED_GENERATED_FUSED_ALIAS_JSON_KEYS = frozenset(
    (
        json.dumps((atom_count, ring_count, elements), separators=(",", ":")),
        graph_key,
        element_graph_key,
    )
    for atom_count, ring_count, elements, graph_key, element_graph_key in _TARGETED_GENERATED_FUSED_ALIAS_KEYS
)

_TARGET_FUSION_COMPONENT_PREFIXES = (
    "imidazo",
    "pyrazolo",
    "pyrrolo",
    "triazolo",
)
_TARGET_FUSION_BASES = {
    "pyridine",
    "pyrimidine",
    "pyrazine",
    "pyridazine",
    "quinoline",
    "quinazoline",
    "quinoxaline",
    "naphthyridine",
}

_IMIDAZOLE_DESCENDING_FUSION_COMPONENT_PREFIXES = (
    "isoquino",
    "naphtho",
    "pyridazino",
    "pyrido",
    "pyrimido",
    "quinazolino",
    "quino",
    "quinoxalino",
    "thieno",
)

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
    store = template_store(Chem)
    templates = (
        build_fused_templates(Chem)
        + store.static_retained()
        + store.opsin_test_scaffolds()
        + store.generated_spiro_scaffolds()
        + store.opsin_retained_fused()
        + store.opsin_retained_ring_csv_fused()
        + store.fusion_component_parents()
    )
    return tuple(template for template in templates if not _exact_retained_only(template))


@lru_cache(maxsize=8)
def preferred_generated_fused_templates(Chem) -> tuple[dict, ...]:
    return _preferred_generated_templates(generated_fused_parent_templates(Chem), Chem, per_skeleton=3)


@lru_cache(maxsize=8)
def hydrocarbon_replacement_parent_templates(Chem) -> tuple[dict, ...]:
    store = template_store(Chem)
    templates = (
        generated_fused_parent_templates(Chem)
        + store.opsin_test_scaffolds()
        + store.opsin_retained_fused()
        + store.opsin_retained_ring_csv_fused()
        + store.fusion_component_parents()
        + store.static_retained()
    )
    selected = []
    seen: set[tuple[str, str]] = set()
    for template in templates:
        features = molecule_features(template_record_mol(template), Chem)
        if features["ring_count"] < 2 or set(features["elements"]) != {"C"}:
            continue
        key = (template_record_name(template), Chem.MolToSmiles(template_record_mol(template), canonical=True))
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
    store = template_store(Chem)
    retained = tuple(
        template
        for template in (
            store.opsin_test_scaffolds()
            + store.opsin_retained_fused()
            + store.opsin_retained_ring_csv_fused()
            + store.fusion_component_parents()
            + store.static_retained()
        )
        if template_record_mol(template).GetNumAtoms() == atom_count
    )
    selected = []
    seen: set[tuple[str, str]] = set()
    for template in templates + retained:
        features = molecule_features(template_record_mol(template), Chem)
        if features["ring_count"] != ring_count or set(features["elements"]) != {"C"}:
            continue
        if not any(
            ring_system_with_graph_signatures(template_record_mol(template), system, Chem)["graph_signature"]
            == graph_signature
            for system in features["ring_systems"]
        ):
            continue
        key = (template_record_name(template), Chem.MolToSmiles(template_record_mol(template), canonical=True))
        if key in seen:
            continue
        seen.add(key)
        selected.append(template)
    return tuple(sorted(selected, key=generated_template_key))


@lru_cache(maxsize=512)
def retained_topology_replacement_parent_templates_for_graph(
    atom_count: int, ring_count: int, graph_signature: str, Chem
) -> tuple[dict, ...]:
    """Retained fused parents with the same ring topology for replacement naming.

    This is deliberately broader than the hydrocarbon-only replacement index:
    retained parents such as chroman or naphthyridine already encode useful
    heteroatom positions, and additional ring heteroatoms can be rendered as
    locanted replacement prefixes when the ring graph is identical.
    """

    store = template_store(Chem)
    templates = (
        store.opsin_retained_ring_csv_fused()
        + store.opsin_retained_fused()
        + store.opsin_test_scaffolds()
        + store.fusion_component_parents()
        + store.static_retained()
    )
    selected = []
    seen: set[tuple[str, str]] = set()
    for template in templates:
        if _exact_retained_only(template):
            continue
        if not template.get("general_parent_safe", True):
            continue
        if template.get("stereochemical"):
            continue
        if set(template.get("flags", ())) & {"radical", "organometallic", "general-parent-unsupported"}:
            continue
        features = molecule_features(template_record_mol(template), Chem)
        if features["ring_count"] != ring_count:
            continue
        if features["atom_count"] != atom_count:
            continue
        if not any(
            ring_system_with_graph_signatures(template_record_mol(template), system, Chem)["graph_signature"]
            == graph_signature
            for system in features["ring_systems"]
        ):
            continue
        key = (template_record_name(template), Chem.MolToSmiles(template_record_mol(template), canonical=True))
        if key in seen:
            continue
        seen.add(key)
        selected.append(template)
    return tuple(sorted(selected, key=generated_template_key))[
        :_MAX_RETAINED_TOPOLOGY_REPLACEMENT_TEMPLATES_PER_GRAPH
    ]


def _exact_retained_only(template: dict) -> bool:
    return "exact-retained-only" in set(template.get("flags", ()))


def candidate_generated_fused_templates(
    mol, Chem, features: dict | None = None, *, graph_filter: bool = True
) -> tuple[dict, ...]:
    cache_key = "generated_fused_templates" if graph_filter else "generated_fused_templates_broad"
    cached = mol_cache_get(mol, cache_key)
    if cached is not None:
        return cached
    features = features or molecule_features(mol, Chem)
    candidates = []
    seen: set[int] = set()
    for system in features["ring_systems"]:
        key = json.dumps(ring_system_feature_key(system), separators=(",", ":"))
        if graph_filter:
            system = ring_system_with_graph_signatures(mol, system, Chem)
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
    result = intern_template_tuple(candidates)
    return mol_cache_set(mol, cache_key, result)


@lru_cache(maxsize=None)
def generated_fusion_preference(name: str) -> tuple:
    match = _FUSION_NAME_RE.match(name)
    if match is None:
        return (999, 999, (), 999, name)
    component, locant_text, letter, base = match.groups()
    orientation_rank = _target_fusion_orientation_rank(component, base)
    base_rank = _FUSION_BASE_PRIORITY.get(base, 500)
    locants = _fusion_descriptor_locants(locant_text)
    locants = _fusion_descriptor_locant_preference(component, base, locants)
    letter_rank = _fusion_letter_rank(component, base, letter)
    component_rank = _component_alias_penalty(component)
    return (orientation_rank, base_rank, letter_rank, locants, component_rank, name)


def generated_template_key(template: dict) -> tuple:
    name = template_record_name(template)
    return (len(name), generated_fusion_preference(name), name)


def generated_record_key(record: dict) -> tuple:
    name = str(record.get("name", ""))
    return (len(name), generated_fusion_preference(name), name)


def is_targeted_generated_fused_alias_key(
    feature_key: tuple,
    graph_key: str,
    element_graph_key: str,
) -> bool:
    return _generated_fused_alias_key(feature_key, graph_key, element_graph_key) in _TARGETED_GENERATED_FUSED_ALIAS_KEYS


def _preferred_generated_templates(templates: tuple[dict, ...], Chem, per_skeleton: int) -> tuple[dict, ...]:
    grouped: dict[str, list[dict]] = {}
    for template in templates:
        key = Chem.MolToSmiles(template_record_mol(template), canonical=True)
        grouped.setdefault(key, []).append(template)
    preferred = []
    for items in grouped.values():
        preferred.extend(sorted(items, key=generated_template_key)[:per_skeleton])
    return tuple(sorted(preferred, key=generated_template_key))


@lru_cache(maxsize=1)
def _generated_fused_template_index() -> dict[str, tuple[dict, ...]]:
    return _load_generated_record_index("generated_fused_template_index.json", _generated_fused_record_table())


@lru_cache(maxsize=1)
def _generated_fused_template_graph_index() -> dict[str, dict[str, tuple[dict, ...]]]:
    return _load_generated_record_index("generated_fused_template_graph_index.json", _generated_fused_record_table())


@lru_cache(maxsize=1)
def _generated_fused_template_element_graph_index() -> dict[str, dict[str, dict[str, tuple[dict, ...]]]]:
    return _load_generated_record_index(
        "generated_fused_template_element_graph_index.json",
        _generated_fused_record_table(),
    )


@lru_cache(maxsize=1)
def _generated_fused_record_table() -> tuple[dict, ...]:
    try:
        raw = read_json_resource("resources", "generated_fused_template_index_records.json")
    except Exception:
        return ()
    return tuple(record for record in raw if isinstance(record, dict))


def _load_generated_record_index(filename: str, table: tuple[dict, ...]):
    try:
        path = resources.files(__package__).joinpath("resources", filename)
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _resolve_generated_record_refs(raw, table)


def _resolve_generated_record_refs(value, table: tuple[dict, ...]):
    if isinstance(value, dict):
        return {str(key): _resolve_generated_record_refs(item, table) for key, item in value.items()}
    if not isinstance(value, list):
        return ()
    records = []
    for item in value:
        if isinstance(item, int):
            if 0 <= item < len(table):
                records.append(table[item])
        elif isinstance(item, dict):
            records.append(item)
    return tuple(records)


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
        if _is_targeted_generated_fused_alias_json_key(key, graph_key, element_graph_key):
            return _preferred_records_per_scaffold(
                records,
                aliases_per_scaffold=_MAX_TARGETED_GENERATED_ALIASES_PER_SCAFFOLD,
                orientation_aliases_per_scaffold=_MAX_TARGETED_ORIENTATION_ALIASES_PER_SCAFFOLD,
            )
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


def _preferred_records_per_scaffold(
    records: tuple[dict, ...],
    *,
    aliases_per_scaffold: int | None = None,
    orientation_aliases_per_scaffold: int | None = None,
) -> tuple[dict, ...]:
    """Collapse OPSIN alias fan-out before materializing generated templates.

    OPSIN's fusion resources contain many accepted names for the same scaffold.
    That breadth is useful for parsing, but the inverse generator only needs a
    small deterministic set of preferred spellings per generated graph. Keeping
    this pruning at the record layer avoids creating RDKit query molecules and
    running substructure searches for hundreds of equivalent aliases.
    """

    if not records:
        return ()
    if aliases_per_scaffold is None:
        aliases_per_scaffold = _MAX_GENERATED_ALIASES_PER_SCAFFOLD
    if orientation_aliases_per_scaffold is None:
        orientation_aliases_per_scaffold = _MAX_ORIENTATION_ALIASES_PER_SCAFFOLD
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record.get("smiles", "")), []).append(record)
    selected = []
    for scaffold_records in grouped.values():
        selected.extend(
            _preferred_generated_records_for_scaffold(
                scaffold_records,
                aliases_per_scaffold=aliases_per_scaffold,
                orientation_aliases_per_scaffold=orientation_aliases_per_scaffold,
            )
        )
    return tuple(
        sorted(
            selected,
            key=generated_record_key,
        )
    )


def _preferred_generated_records_for_scaffold(
    records: list[dict],
    *,
    aliases_per_scaffold: int | None = None,
    orientation_aliases_per_scaffold: int | None = None,
) -> list[dict]:
    if aliases_per_scaffold is None:
        aliases_per_scaffold = _MAX_GENERATED_ALIASES_PER_SCAFFOLD
    if orientation_aliases_per_scaffold is None:
        orientation_aliases_per_scaffold = _MAX_ORIENTATION_ALIASES_PER_SCAFFOLD
    orientation_preferred = sorted(
        records,
        key=lambda record: (generated_fusion_preference(str(record.get("name", ""))), len(str(record.get("name", "")))),
    )[:orientation_aliases_per_scaffold]
    shortest = sorted(
        (
            record
            for record in records
            if not _skip_short_generated_fusion_alias(record, orientation_preferred)
        ),
        key=lambda record: (
            len(str(record.get("name", ""))),
            generated_fusion_preference(str(record.get("name", ""))),
        ),
    )[:aliases_per_scaffold]
    selected = []
    seen = set()
    for record in shortest + orientation_preferred:
        marker = id(record)
        if marker in seen:
            continue
        seen.add(marker)
        selected.append(record)
    return selected


def _skip_short_generated_fusion_alias(record: dict, orientation_preferred: list[dict]) -> bool:
    name = str(record.get("name", ""))
    if generated_fusion_orientation_role(name) != "disfavored-pyrazole-imidazole":
        return False
    return any(
        generated_fusion_orientation_role(str(preferred.get("name", ""))) == "preferred-pyrazolo-imidazole"
        for preferred in orientation_preferred
    )


def _is_targeted_generated_fused_alias_json_key(key: str, graph_key: str, element_graph_key: str) -> bool:
    return (key, graph_key, element_graph_key) in _TARGETED_GENERATED_FUSED_ALIAS_JSON_KEYS


def _generated_fused_alias_key(feature_key, graph_key: str, element_graph_key: str) -> tuple:
    return (
        int(feature_key[0]),
        int(feature_key[1]),
        tuple(tuple(item) for item in feature_key[2]),
        graph_key,
        element_graph_key,
    )


def _target_fusion_orientation_rank(component: str, base: str) -> int:
    if component.startswith("pyrazolo") and base == "imidazole":
        return 0
    if component.startswith("imidazo") and base == "pyrazole":
        return 3
    if component.startswith(_TARGET_FUSION_COMPONENT_PREFIXES) and base in _TARGET_FUSION_BASES:
        return 0
    if component.startswith(("pyrido", "pyrimido")) and base in {"pyrazole", "pyrrole", "imidazole", "triazole"}:
        return 2
    return 1


def _fusion_letter_rank(component: str, base: str, letter: str) -> int:
    if base.startswith("pyrazol") and not component.startswith(("imidazo", "pyrazolo")):
        return {"d": 0, "c": 1}.get(letter, ord(letter) - ord("a") + 2)
    return ord(letter) - ord("a")


def generated_fusion_orientation_role(name: str) -> str:
    match = _FUSION_NAME_RE.match(name)
    if match is None:
        return ""
    component, _locant_text, _letter, base = match.groups()
    if component.startswith("pyrazolo") and base == "imidazole":
        return "preferred-pyrazolo-imidazole"
    if component.startswith("imidazo") and base == "pyrazole":
        return "disfavored-pyrazole-imidazole"
    return ""


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


def _fusion_descriptor_locant_preference(component: str, base: str, locants: tuple) -> tuple:
    if base == "imidazole" and component.startswith(_IMIDAZOLE_DESCENDING_FUSION_COMPONENT_PREFIXES):
        return tuple((-number, suffix) for number, suffix in locants)
    return locants


def _component_alias_penalty(component: str) -> tuple:
    expanded_penalty = 1 if component.endswith(("ano", "eno", "ino", "olo")) else 0
    return (expanded_penalty, len(component), component)
