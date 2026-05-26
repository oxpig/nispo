from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

from .resource_templates import (
    build_fusion_component_parent_templates,
    build_generated_fused_parent_templates,
    build_opsin_resource_templates,
    build_opsin_test_scaffold_templates,
    build_retained_parent_templates,
    build_static_parent_templates,
)
from .template_index import candidate_templates as _candidate_templates
from .template_index import molecule_features as _molecule_features


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


@dataclass(frozen=True, slots=True)
class TemplateFeatures:
    """Cached structural features for a compiled template query."""

    atom_count: int
    ring_count: int
    max_ring_system_size: int
    max_ring_system_ring_count: int
    elements: tuple[tuple[str, int], ...]
    all_atoms_in_one_ring_system: bool

    @classmethod
    def from_mapping(cls, features: Mapping) -> "TemplateFeatures":
        return cls(
            atom_count=int(features["atom_count"]),
            ring_count=int(features["ring_count"]),
            max_ring_system_size=int(features["max_ring_system_size"]),
            max_ring_system_ring_count=int(features.get("max_ring_system_ring_count", 0)),
            elements=tuple(sorted(features["elements"].items())),
            all_atoms_in_one_ring_system=bool(features["all_atoms_in_one_ring_system"]),
        )


@dataclass(frozen=True, slots=True)
class CompiledTemplate:
    """Runtime template object with RDKit query and cached store-owned facts."""

    record: TemplateRecord
    mol: object
    features: TemplateFeatures
    scaffold_key: str
    runtime: dict


@dataclass(frozen=True)
class ParentScaffoldCatalog:
    """Data-driven parent scaffold families used by naming rules."""

    static_retained: tuple[dict, ...]
    opsin_test_scaffolds: tuple[dict, ...]
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

    @property
    def catalog(self) -> ParentScaffoldCatalog:
        static_retained = self.static_retained()
        opsin_test_scaffolds = self.opsin_test_scaffolds()
        opsin_retained_fused = self.opsin_retained_fused()
        fusion_component_parents = self.fusion_component_parents()
        exact = self._dedupe_templates(
            build_retained_parent_templates(self.Chem, min_rings=0, max_atoms=48)
            + static_retained
            + opsin_test_scaffolds
            + fusion_component_parents
        )
        single_ring = build_retained_parent_templates(self.Chem, min_rings=1, max_rings=1)
        fused = static_retained + opsin_test_scaffolds + opsin_retained_fused + fusion_component_parents
        return ParentScaffoldCatalog(
            static_retained=static_retained,
            opsin_test_scaffolds=opsin_test_scaffolds,
            opsin_retained_fused=opsin_retained_fused,
            fusion_component_parents=fusion_component_parents,
            fused=fused,
            exact=exact,
            single_ring=single_ring,
            by_source_family=self._group_by_source_family(fused + exact + single_ring),
        )

    def static_retained(self) -> tuple[dict, ...]:
        return build_static_parent_templates(self.Chem)

    def opsin_test_scaffolds(self) -> tuple[dict, ...]:
        return build_opsin_test_scaffold_templates(self.Chem)

    def opsin_retained_fused(self) -> tuple[dict, ...]:
        return build_opsin_resource_templates(self.Chem)

    def fusion_component_parents(self) -> tuple[dict, ...]:
        return build_fusion_component_parent_templates(self.Chem)

    def generated_fused_parent_templates(self) -> tuple[dict, ...]:
        return build_generated_fused_parent_templates(self.Chem)

    def candidate_templates(
        self,
        mol,
        templates: tuple[dict, ...],
        features: dict | None = None,
        *,
        exact_atom_count: bool = False,
    ) -> tuple[dict, ...]:
        result = _candidate_templates(
            mol,
            templates,
            self.Chem,
            features=features,
            exact_atom_count=exact_atom_count,
        )
        for template in result:
            self.compiled_template(template)
        return result

    def compiled_template(self, template: dict) -> CompiledTemplate:
        key = id(template)
        cached = self._compiled.get(key)
        if cached is not None:
            return cached
        mol = template["mol"]
        features = _molecule_features(mol, self.Chem)
        compiled = CompiledTemplate(
            record=self._record_for_template(template, key),
            mol=mol,
            features=TemplateFeatures.from_mapping(features),
            scaffold_key=self.Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
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
        )

    def _dedupe_templates(self, templates: tuple[dict, ...]) -> tuple[dict, ...]:
        unique = []
        seen: set[tuple[str, str]] = set()
        for template in templates:
            key = (template["name"], self.Chem.MolToSmiles(template["mol"], canonical=True))
            if key in seen:
                continue
            seen.add(key)
            unique.append(template)
            self.compiled_template(template)
        return tuple(unique)

    @staticmethod
    def _group_by_source_family(templates: tuple[dict, ...]) -> dict[str, tuple[dict, ...]]:
        grouped: dict[str, list[dict]] = {}
        for template in templates:
            grouped.setdefault(template.get("source_family", "unknown"), []).append(template)
        return {family: tuple(items) for family, items in sorted(grouped.items())}


@lru_cache(maxsize=8)
def template_store(Chem) -> TemplateStore:
    return TemplateStore(Chem)
