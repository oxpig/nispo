from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Mapping

from .molecule_iteration import iter_atoms, iter_bonds
from .resource_templates import (
    build_exact_biochemical_parent_templates,
    build_exact_unlocanted_resource_templates,
    build_fusion_component_parent_templates,
    build_generated_fused_parent_templates,
    build_generated_spiro_scaffold_templates,
    build_opsin_retained_ring_csv_templates,
    build_opsin_resource_templates,
    build_opsin_test_scaffold_templates,
    build_retained_parent_templates,
    build_static_parent_templates,
)
from .template_index import candidate_templates as _candidate_templates
from .template_index import molecule_features as _molecule_features
from .template_index import ring_system_feature_key
from .template_index import ring_system_with_graph_signature
from .template_index import _template_coarse_index
from .template_index import _template_canonical_smiles

_MAX_COMPILED_TEMPLATE_ALIASES_PER_SCAFFOLD = 1


@dataclass(frozen=True, slots=True)
class TemplateRecord:
    """Immutable metadata for one nomenclature template resource."""

    id: int
    name: str
    locants: tuple[tuple[int, str], ...]
    source_family: str
    source_file: str
    source_subtype: str
    stereochemical: bool
    flags: frozenset[str] = frozenset()
    preferred_iupac_parent: bool = False


@dataclass(frozen=True, slots=True)
class TemplateFeatures:
    """Cached structural features for a compiled template query."""

    atom_count: int
    ring_count: int
    heavy_atom_count: int
    formal_charge: int
    charged_atom_values: tuple[int, ...]
    charged_atom_value_counts: tuple[tuple[int, int], ...]
    aromatic_atom_count: int
    max_ring_system_size: int
    max_ring_system_ring_count: int
    elements: tuple[tuple[str, int], ...]
    ring_system_keys: tuple[tuple, ...]
    all_atoms_in_one_ring_system: bool
    ring_system_graph_keys: tuple[tuple, ...] = ()
    has_stereochemistry: bool = False
    has_isotope: bool = False
    unsupported_flags: frozenset[str] = frozenset()

    @classmethod
    def from_mapping(cls, features: Mapping) -> "TemplateFeatures":
        return cls(
            atom_count=int(features["atom_count"]),
            ring_count=int(features["ring_count"]),
            heavy_atom_count=int(features.get("heavy_atom_count", features["atom_count"])),
            formal_charge=int(features.get("formal_charge", 0)),
            charged_atom_values=tuple(sorted(int(value) for value in features.get("charged_atom_values", ()))),
            charged_atom_value_counts=tuple(
                sorted(
                    (int(value), int(count))
                    for value, count in (features.get("charged_atom_value_counts") or {}).items()
                )
            ),
            aromatic_atom_count=int(features.get("aromatic_atom_count", 0)),
            max_ring_system_size=int(features["max_ring_system_size"]),
            max_ring_system_ring_count=int(features.get("max_ring_system_ring_count", 0)),
            elements=tuple(sorted(features["elements"].items())),
            ring_system_keys=tuple(ring_system_feature_key(system) for system in features.get("ring_systems", ())),
            ring_system_graph_keys=tuple(
                _ring_system_graph_key(system)
                for system in features.get("ring_systems", ())
                if system.get("graph_signature")
            ),
            all_atoms_in_one_ring_system=bool(features["all_atoms_in_one_ring_system"]),
            has_stereochemistry=bool(features.get("has_stereochemistry", False)),
            has_isotope=bool(features.get("has_isotope", False)),
            unsupported_flags=frozenset(str(flag) for flag in features.get("unsupported_flags", ())),
        )

    @classmethod
    def from_mol(
        cls,
        mol,
        Chem,
        *,
        has_stereochemistry: bool = False,
        unsupported_flags: frozenset[str] = frozenset(),
    ) -> "TemplateFeatures":
        features = dict(_molecule_features(mol, Chem))
        features["heavy_atom_count"] = sum(1 for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1)
        features["formal_charge"] = sum(atom.GetFormalCharge() for atom in iter_atoms(mol))
        charged_atom_value_counts = features.get("charged_atom_value_counts") or {}
        features["charged_atom_values"] = tuple(sorted(charged_atom_value_counts))
        features["charged_atom_value_counts"] = charged_atom_value_counts
        features["aromatic_atom_count"] = sum(1 for atom in iter_atoms(mol) if atom.GetIsAromatic())
        features["ring_systems"] = tuple(
            ring_system_with_graph_signature(mol, dict(system), Chem)
            for system in features.get("ring_systems", ())
        )
        features["has_stereochemistry"] = has_stereochemistry or _mol_has_stereochemistry(mol, Chem)
        features["has_isotope"] = any(atom.GetIsotope() for atom in iter_atoms(mol))
        features["unsupported_flags"] = unsupported_flags
        return cls.from_mapping(features)


@dataclass(frozen=True, slots=True)
class CompiledTemplate:
    """Runtime template object with RDKit query and cached store-owned facts."""

    record: TemplateRecord
    mol: object
    features: TemplateFeatures
    scaffold_key: str
    runtime: dict


_CompiledTemplateIndex = tuple[
    dict[tuple, tuple[CompiledTemplate, ...]],
    dict[int, tuple[CompiledTemplate, ...]],
    frozenset[tuple],
]


@dataclass(frozen=True)
class ParentScaffoldCatalog:
    """Data-driven parent scaffold families used by naming rules."""

    static_retained: tuple[dict, ...]
    opsin_test_scaffolds: tuple[dict, ...]
    generated_spiro_scaffolds: tuple[dict, ...]
    opsin_retained_ring_csv: tuple[dict, ...]
    opsin_retained_ring_csv_parent: tuple[dict, ...]
    opsin_retained_ring_csv_single_ring: tuple[dict, ...]
    opsin_retained_ring_csv_fused: tuple[dict, ...]
    opsin_retained_fused: tuple[dict, ...]
    fusion_component_parents: tuple[dict, ...]
    fused: tuple[dict, ...]
    exact: tuple[dict, ...]
    single_ring: tuple[dict, ...]
    by_source_family: dict[str, tuple[dict, ...]]


class TemplateStore:
    """Single facade for loading, indexing, and selecting template resources."""

    def __init__(self, Chem):
        self.Chem = Chem
        self._compiled: dict[int, CompiledTemplate] = {}
        self._compiled_mol_features: dict[object, TemplateFeatures] = {}
        self._compiled_indexes: dict[int, tuple[tuple[dict, ...], _CompiledTemplateIndex]] = {}
        self._catalog: ParentScaffoldCatalog | None = None
        self._static_retained: tuple[dict, ...] | None = None
        self._opsin_test_scaffolds: tuple[dict, ...] | None = None
        self._generated_spiro_scaffolds: tuple[dict, ...] | None = None
        self._opsin_retained_ring_csv: tuple[dict, ...] | None = None
        self._opsin_retained_ring_csv_parent: tuple[dict, ...] | None = None
        self._opsin_retained_ring_csv_single_ring: tuple[dict, ...] | None = None
        self._opsin_retained_ring_csv_fused: tuple[dict, ...] | None = None
        self._opsin_retained_fused: tuple[dict, ...] | None = None
        self._fusion_component_parents: tuple[dict, ...] | None = None
        self._generated_fused_parent_templates: tuple[dict, ...] | None = None
        self._exact_templates: tuple[dict, ...] | None = None
        self._single_ring_templates: tuple[dict, ...] | None = None

    @property
    def catalog(self) -> ParentScaffoldCatalog:
        if self._catalog is not None:
            return self._catalog
        static_retained = self.static_retained()
        opsin_test_scaffolds = self.opsin_test_scaffolds()
        generated_spiro_scaffolds = self.generated_spiro_scaffolds()
        opsin_retained_ring_csv = self.opsin_retained_ring_csv()
        opsin_retained_ring_csv_parent = self.opsin_retained_ring_csv_parent()
        opsin_retained_ring_csv_single_ring = self.opsin_retained_ring_csv_single_ring()
        opsin_retained_ring_csv_fused = self.opsin_retained_ring_csv_fused()
        opsin_retained_fused = self.opsin_retained_fused()
        fusion_component_parents = self.fusion_component_parents()
        exact = self.exact_templates()
        fused = (
            static_retained
            + opsin_test_scaffolds
            + generated_spiro_scaffolds
            + opsin_retained_fused
            + opsin_retained_ring_csv_fused
            + fusion_component_parents
        )
        single_ring = self.single_ring_templates()
        self._catalog = ParentScaffoldCatalog(
            static_retained=static_retained,
            opsin_test_scaffolds=opsin_test_scaffolds,
            generated_spiro_scaffolds=generated_spiro_scaffolds,
            opsin_retained_ring_csv=opsin_retained_ring_csv,
            opsin_retained_ring_csv_parent=opsin_retained_ring_csv_parent,
            opsin_retained_ring_csv_single_ring=opsin_retained_ring_csv_single_ring,
            opsin_retained_ring_csv_fused=opsin_retained_ring_csv_fused,
            opsin_retained_fused=opsin_retained_fused,
            fusion_component_parents=fusion_component_parents,
            fused=fused,
            exact=exact,
            single_ring=single_ring,
            by_source_family=self._group_by_source_family(fused + exact + single_ring),
        )
        return self._catalog

    def static_retained(self) -> tuple[dict, ...]:
        if self._static_retained is None:
            self._static_retained = build_static_parent_templates(self.Chem)
        return self._static_retained

    def opsin_test_scaffolds(self) -> tuple[dict, ...]:
        if self._opsin_test_scaffolds is None:
            self._opsin_test_scaffolds = build_opsin_test_scaffold_templates(self.Chem)
        return self._opsin_test_scaffolds

    def generated_spiro_scaffolds(self) -> tuple[dict, ...]:
        if self._generated_spiro_scaffolds is None:
            self._generated_spiro_scaffolds = build_generated_spiro_scaffold_templates(self.Chem)
        return self._generated_spiro_scaffolds

    def opsin_retained_ring_csv(self) -> tuple[dict, ...]:
        if self._opsin_retained_ring_csv is None:
            self._opsin_retained_ring_csv = build_opsin_retained_ring_csv_templates(self.Chem)
        return self._opsin_retained_ring_csv

    def opsin_retained_ring_csv_parent(self) -> tuple[dict, ...]:
        if self._opsin_retained_ring_csv_parent is None:
            self._opsin_retained_ring_csv_parent = self._general_retained_ring_csv_parent_templates(
                self.opsin_retained_ring_csv()
            )
        return self._opsin_retained_ring_csv_parent

    def opsin_retained_ring_csv_single_ring(self) -> tuple[dict, ...]:
        if self._opsin_retained_ring_csv_single_ring is None:
            self._opsin_retained_ring_csv_single_ring = self._general_retained_mono_ring_csv_templates(
                self.opsin_retained_ring_csv()
            )
        return self._opsin_retained_ring_csv_single_ring

    def opsin_retained_ring_csv_fused(self) -> tuple[dict, ...]:
        if self._opsin_retained_ring_csv_fused is None:
            self._opsin_retained_ring_csv_fused = self._general_retained_ring_csv_templates(
                self.opsin_retained_ring_csv()
            )
        return self._opsin_retained_ring_csv_fused

    def _general_retained_ring_csv_parent_templates(self, templates: tuple[dict, ...]) -> tuple[dict, ...]:
        selected = []
        for template in templates:
            if not template.get("general_parent_safe", True):
                continue
            features = _molecule_features(template["mol"], self.Chem)
            if features["ring_count"] < 1:
                continue
            selected.append(template)
        return tuple(selected)

    def _general_retained_ring_csv_templates(self, templates: tuple[dict, ...]) -> tuple[dict, ...]:
        selected = []
        for template in templates:
            if not template.get("general_parent_safe", True):
                continue
            if template.get("stereochemical"):
                continue
            features = _molecule_features(template["mol"], self.Chem)
            if features["ring_count"] < 2 or not features["all_atoms_in_one_ring_system"]:
                continue
            selected.append(template)
        return tuple(selected)

    def _general_retained_mono_ring_csv_templates(self, templates: tuple[dict, ...]) -> tuple[dict, ...]:
        selected = []
        for template in templates:
            if not template.get("general_parent_safe", True):
                continue
            if template.get("stereochemical"):
                continue
            features = _molecule_features(template["mol"], self.Chem)
            if features["ring_count"] != 1 or not features["all_atoms_in_one_ring_system"]:
                continue
            selected.append(template)
        return tuple(selected)

    def opsin_retained_fused(self) -> tuple[dict, ...]:
        if self._opsin_retained_fused is None:
            self._opsin_retained_fused = build_opsin_resource_templates(self.Chem)
        return self._opsin_retained_fused

    def fusion_component_parents(self) -> tuple[dict, ...]:
        if self._fusion_component_parents is None:
            self._fusion_component_parents = build_fusion_component_parent_templates(self.Chem)
        return self._fusion_component_parents

    def generated_fused_parent_templates(self) -> tuple[dict, ...]:
        if self._generated_fused_parent_templates is None:
            self._generated_fused_parent_templates = build_generated_fused_parent_templates(self.Chem)
        return self._generated_fused_parent_templates

    def exact_templates(self) -> tuple[dict, ...]:
        if self._exact_templates is None:
            self._exact_templates = self._dedupe_templates(
                build_retained_parent_templates(self.Chem, min_rings=0, max_atoms=48)
                + self.static_retained()
                + build_exact_biochemical_parent_templates(self.Chem)
                + build_exact_unlocanted_resource_templates(self.Chem)
                + self.opsin_test_scaffolds()
                + self.generated_spiro_scaffolds()
                + self.opsin_retained_ring_csv()
                + self.fusion_component_parents()
            )
        return self._exact_templates

    def single_ring_templates(self) -> tuple[dict, ...]:
        if self._single_ring_templates is None:
            self._single_ring_templates = self._dedupe_templates(
                build_retained_parent_templates(self.Chem, min_rings=1, max_rings=1)
                + self.opsin_retained_ring_csv_single_ring()
            )
        return self._single_ring_templates

    def candidate_templates(
        self,
        mol,
        templates: tuple[dict, ...],
        features: dict | None = None,
        *,
        exact_atom_count: bool = False,
    ) -> tuple[dict, ...]:
        return _candidate_templates(
            mol,
            templates,
            self.Chem,
            features=features,
            exact_atom_count=exact_atom_count,
        )

    def candidate_compiled_templates(
        self,
        mol,
        templates: tuple[dict, ...],
        features: dict | None = None,
        *,
        exact_atom_count: bool = False,
    ) -> tuple[CompiledTemplate, ...]:
        features = features or _molecule_features(mol, self.Chem)
        pure_buckets, other_buckets = _template_coarse_index(templates, self.Chem)
        candidates: list[CompiledTemplate] = []
        seen: set[int] = set()
        molecule_charge_counts = features.get("charged_atom_value_counts") or {}
        molecule_ring_system_keys = {
            ring_system_feature_key(system)
            for system in features["ring_systems"]
        }
        molecule_ring_system_graph_keys: set[tuple] | None = None
        for system in features["ring_systems"]:
            coarse_key = ring_system_feature_key(system)
            bucket = pure_buckets.get(coarse_key)
            if bucket is None:
                continue
            pure_ring_templates, _, graph_index_coarse_keys = self._compiled_template_index(bucket)
            if coarse_key in graph_index_coarse_keys and not system.get("graph_signature"):
                system = ring_system_with_graph_signature(mol, system, self.Chem)
            if system.get("graph_signature") and (
                coarse_key in graph_index_coarse_keys
                or _ring_system_graph_key(system) in pure_ring_templates
            ):
                key = _ring_system_graph_key(system)
            else:
                key = coarse_key
            for compiled in pure_ring_templates.get(key, ()):
                marker = compiled.record.id
                if marker in seen:
                    continue
                seen.add(marker)
                candidates.append(compiled)
        if exact_atom_count:
            atom_count = features["atom_count"]
            bucket = other_buckets.get(atom_count, ())
            other = self._compiled_template_index(bucket)[1] if bucket else {}
            atom_count_items = (
                (atom_count, other.get(atom_count, ())),
            )
        else:
            atom_count_items = (
                (atom_count, self._compiled_template_index(bucket)[1].get(atom_count, ()))
                for atom_count, bucket in other_buckets.items()
                if atom_count <= features["atom_count"]
            )
        for _atom_count, entries in atom_count_items:
            for compiled in entries:
                if compiled.features.ring_system_keys and not any(
                    key in molecule_ring_system_keys
                    for key in compiled.features.ring_system_keys
                ):
                    continue
                if compiled.features.ring_system_graph_keys:
                    if molecule_ring_system_graph_keys is None:
                        molecule_ring_system_graph_keys = _molecule_ring_system_graph_keys(mol, features, self.Chem)
                    if not any(
                        key in molecule_ring_system_graph_keys
                        for key in compiled.features.ring_system_graph_keys
                    ):
                        continue
                if compiled.features.ring_count > features["ring_count"]:
                    continue
                if compiled.features.max_ring_system_ring_count > features.get("max_ring_system_ring_count", 0):
                    continue
                if compiled.features.max_ring_system_size > features["max_ring_system_size"]:
                    continue
                if compiled.features.charged_atom_value_counts and any(
                    count > molecule_charge_counts.get(value, 0)
                    for value, count in compiled.features.charged_atom_value_counts
                ):
                    continue
                if any(
                    count > features["elements"].get(symbol, 0)
                    for symbol, count in compiled.features.elements
                ):
                    continue
                candidates.append(compiled)
        candidates.sort(key=lambda item: item.features.atom_count, reverse=True)
        return tuple(candidates)

    def _compiled_template_index(self, templates: tuple[dict, ...]) -> _CompiledTemplateIndex:
        cache_key = id(templates)
        cached = self._compiled_indexes.get(cache_key)
        if cached is not None and cached[0] is templates:
            return cached[1]
        pure_ring_templates: dict[tuple, list[CompiledTemplate]] = defaultdict(list)
        other_templates_by_atom_count: dict[int, list[CompiledTemplate]] = defaultdict(list)
        for template in templates:
            compiled = self.compiled_template(template)
            features = compiled.features
            if features.all_atoms_in_one_ring_system:
                keys = features.ring_system_graph_keys or features.ring_system_keys
                for key in keys:
                    pure_ring_templates[key].append(compiled)
            else:
                other_templates_by_atom_count[features.atom_count].append(compiled)
        indexed = (
            {
                key: _preferred_compiled_templates_by_scaffold(value)
                for key, value in pure_ring_templates.items()
            },
            {
                key: _preferred_compiled_templates_by_scaffold(value)
                for key, value in other_templates_by_atom_count.items()
            },
            frozenset(key[:3] for key in pure_ring_templates if len(key) > 3),
        )
        self._compiled_indexes[cache_key] = (templates, indexed)
        return indexed

    def compiled_template(self, template: dict) -> CompiledTemplate:
        key = id(template)
        cached = self._compiled.get(key)
        if cached is not None:
            return cached
        mol = template["mol"]
        record = self._record_for_template(template, key)
        features = self._compiled_mol_features.get(mol)
        if features is None:
            features = TemplateFeatures.from_mol(mol, self.Chem)
            self._compiled_mol_features[mol] = features
        compiled = CompiledTemplate(
            record=record,
            mol=mol,
            features=replace(
                features,
                has_stereochemistry=record.stereochemical or features.has_stereochemistry,
                unsupported_flags=record.flags,
            ),
            scaffold_key=_template_canonical_smiles(mol, self.Chem, isomeric=False),
            runtime=template,
        )
        self._compiled[key] = compiled
        return compiled

    def template_features(self, template: dict) -> TemplateFeatures:
        return self.compiled_template(template).features

    def scaffold_key(self, template: dict) -> str:
        return self.compiled_template(template).scaffold_key

    def _record_for_template(self, template: dict, record_id: int) -> TemplateRecord:
        return TemplateRecord(
            id=record_id,
            name=str(template["name"]),
            locants=tuple(sorted((int(key), str(value)) for key, value in template.get("locants", {}).items())),
            source_family=str(template.get("source_family", "")),
            source_file=str(template.get("source_file", "")),
            source_subtype=str(template.get("source_subtype", "")),
            stereochemical=bool(template.get("stereochemical", False)),
            flags=frozenset(str(flag) for flag in template.get("flags", ())),
            preferred_iupac_parent=bool(template.get("preferred_iupac_parent", False)),
        )

    def _dedupe_templates(self, templates: tuple[dict, ...]) -> tuple[dict, ...]:
        unique = []
        seen: set[tuple[str, str]] = set()
        name_counts = Counter(template["name"] for template in templates)
        for template in templates:
            # A unique name cannot duplicate a (name, graph) pair. Preserve
            # catalog order and canonicalize only names needing comparison.
            if name_counts[template["name"]] == 1:
                unique.append(template)
                continue
            key = (template["name"], _template_canonical_smiles(template["mol"], self.Chem))
            if key in seen:
                continue
            seen.add(key)
            unique.append(template)
        return tuple(unique)

    @staticmethod
    def _group_by_source_family(templates: tuple[dict, ...]) -> dict[str, tuple[dict, ...]]:
        grouped: dict[str, list[dict]] = {}
        for template in templates:
            grouped.setdefault(template.get("source_family", "unknown"), []).append(template)
        return {family: tuple(items) for family, items in sorted(grouped.items())}

def _preferred_compiled_templates_by_scaffold(
    templates: list[CompiledTemplate],
) -> tuple[CompiledTemplate, ...]:
    if not templates:
        return ()
    grouped: dict[str, list[CompiledTemplate]] = {}
    for template in templates:
        grouped.setdefault(template.scaffold_key, []).append(template)
    selected: list[CompiledTemplate] = []
    for group in grouped.values():
        selected.extend(
            sorted(group, key=_compiled_template_alias_preference_key)[
                :_MAX_COMPILED_TEMPLATE_ALIASES_PER_SCAFFOLD
            ]
        )
    return tuple(selected)


def _compiled_template_alias_preference_key(template: CompiledTemplate) -> tuple:
    name = template.record.name
    return (not template.record.preferred_iupac_parent, _compiled_template_alias_priority(name), len(name), name)


def _compiled_template_alias_priority(name: str) -> int:
    if name == "6,7,8,9-tetrahydro-5H-benzo[7]annulene":
        return -1
    return 0


def _molecule_ring_system_graph_keys(mol, features: dict, Chem) -> set[tuple]:
    keys: set[tuple] = set()
    ring_systems = []
    changed = False
    for system in features.get("ring_systems", ()):
        if system.get("graph_signature"):
            enriched = system
        else:
            enriched = ring_system_with_graph_signature(mol, system, Chem)
            changed = True
        ring_systems.append(enriched)
        if enriched.get("graph_signature"):
            keys.add(_ring_system_graph_key(enriched))
    if changed:
        features["ring_systems"] = tuple(ring_systems)
    return keys


def _ring_system_graph_key(features: dict) -> tuple:
    return (
        *ring_system_feature_key(features),
        features.get("graph_signature", ""),
    )


def _mol_has_stereochemistry(mol, Chem) -> bool:
    for atom in iter_atoms(mol):
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return True
    for bond in iter_bonds(mol):
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return True
    return False


@lru_cache(maxsize=8)
def template_store(Chem) -> TemplateStore:
    return TemplateStore(Chem)
