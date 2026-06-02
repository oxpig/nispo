from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

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
    heavy_atom_count: int
    formal_charge: int
    aromatic_atom_count: int
    max_ring_system_size: int
    max_ring_system_ring_count: int
    elements: tuple[tuple[str, int], ...]
    ring_system_keys: tuple[tuple, ...]
    all_atoms_in_one_ring_system: bool
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
            aromatic_atom_count=int(features.get("aromatic_atom_count", 0)),
            max_ring_system_size=int(features["max_ring_system_size"]),
            max_ring_system_ring_count=int(features.get("max_ring_system_ring_count", 0)),
            elements=tuple(sorted(features["elements"].items())),
            ring_system_keys=tuple(ring_system_feature_key(system) for system in features.get("ring_systems", ())),
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
        features["heavy_atom_count"] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
        features["formal_charge"] = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        features["aromatic_atom_count"] = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        features["has_stereochemistry"] = has_stereochemistry or _mol_has_stereochemistry(mol, Chem)
        features["has_isotope"] = any(atom.GetIsotope() for atom in mol.GetAtoms())
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

    @property
    def catalog(self) -> ParentScaffoldCatalog:
        static_retained = self.static_retained()
        opsin_test_scaffolds = self.opsin_test_scaffolds()
        generated_spiro_scaffolds = self.generated_spiro_scaffolds()
        opsin_retained_ring_csv = self.opsin_retained_ring_csv()
        opsin_retained_ring_csv_parent = self._general_retained_ring_csv_parent_templates(
            opsin_retained_ring_csv
        )
        opsin_retained_ring_csv_single_ring = self._general_retained_mono_ring_csv_templates(
            opsin_retained_ring_csv
        )
        opsin_retained_ring_csv_fused = self._general_retained_ring_csv_templates(opsin_retained_ring_csv)
        opsin_retained_fused = self.opsin_retained_fused()
        fusion_component_parents = self.fusion_component_parents()
        exact_biochemical = build_exact_biochemical_parent_templates(self.Chem)
        exact_unlocanted = build_exact_unlocanted_resource_templates(self.Chem)
        exact = self._dedupe_templates(
            build_retained_parent_templates(self.Chem, min_rings=0, max_atoms=48)
            + static_retained
            + exact_biochemical
            + exact_unlocanted
            + opsin_test_scaffolds
            + generated_spiro_scaffolds
            + opsin_retained_ring_csv
            + fusion_component_parents
        )
        single_ring = build_retained_parent_templates(self.Chem, min_rings=1, max_rings=1)
        fused = (
            static_retained
            + opsin_test_scaffolds
            + generated_spiro_scaffolds
            + opsin_retained_fused
            + opsin_retained_ring_csv_fused
            + fusion_component_parents
        )
        single_ring = self._dedupe_templates(single_ring + opsin_retained_ring_csv_single_ring)
        return ParentScaffoldCatalog(
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

    def static_retained(self) -> tuple[dict, ...]:
        return build_static_parent_templates(self.Chem)

    def opsin_test_scaffolds(self) -> tuple[dict, ...]:
        return build_opsin_test_scaffold_templates(self.Chem)

    def generated_spiro_scaffolds(self) -> tuple[dict, ...]:
        return build_generated_spiro_scaffold_templates(self.Chem)

    def opsin_retained_ring_csv(self) -> tuple[dict, ...]:
        return build_opsin_retained_ring_csv_templates(self.Chem)

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
        return tuple(
            self.compiled_template(template)
            for template in self.candidate_templates(
                mol,
                templates,
                features=features,
                exact_atom_count=exact_atom_count,
            )
        )

    def compiled_template(self, template: dict) -> CompiledTemplate:
        key = id(template)
        cached = self._compiled.get(key)
        if cached is not None:
            return cached
        mol = template["mol"]
        record = self._record_for_template(template, key)
        compiled = CompiledTemplate(
            record=record,
            mol=mol,
            features=TemplateFeatures.from_mol(
                mol,
                self.Chem,
                has_stereochemistry=record.stereochemical,
                unsupported_flags=record.flags,
            ),
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
            flags=frozenset(str(flag) for flag in template.get("flags", ())),
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
        return tuple(unique)

    @staticmethod
    def _group_by_source_family(templates: tuple[dict, ...]) -> dict[str, tuple[dict, ...]]:
        grouped: dict[str, list[dict]] = {}
        for template in templates:
            grouped.setdefault(template.get("source_family", "unknown"), []).append(template)
        return {family: tuple(items) for family, items in sorted(grouped.items())}


def _mol_has_stereochemistry(mol, Chem) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return True
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return True
    return False


@lru_cache(maxsize=8)
def template_store(Chem) -> TemplateStore:
    return TemplateStore(Chem)
