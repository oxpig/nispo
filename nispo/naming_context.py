from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .acyclic_rules import name_acyclic
from .fused_template_families import candidate_generated_fused_templates
from .polycyclic_rules import (
    bicyclic_parent_candidate_records,
    kekulized_polycyclic_parent_candidate_records,
    spiro_parent_candidate_records,
)
from .structural_profile import StructuralProfile, structural_profile
from .template_access import template_record_mol
from .template_index import molecule_features
from .template_index import ring_systems_for_mol
from .template_store import TemplateStore, template_store
from .ring_topology import RingSystemTopology, ring_system_topologies
from .telemetry import CandidateTelemetry
from .molecule_cache import mol_cache_set


_TEMPLATE_COLLECTION_LABEL_CACHE: dict[int, tuple[tuple[dict, ...], str]] = {}


@dataclass(slots=True)
class ContextCache:
    template_candidates: dict[tuple[int, bool], tuple[dict, ...]] = field(default_factory=dict)
    compiled_template_candidates: dict[tuple[int, bool], tuple[object, ...]] = field(default_factory=dict)
    acyclic_names: dict[bool, str] = field(default_factory=dict)
    polycyclic_records: dict[str, tuple[tuple[str, tuple[int, ...]], ...]] = field(default_factory=dict)
    substruct_matches: dict[tuple[int, bool], tuple[tuple[int, ...], ...]] = field(default_factory=dict)
    topology_flags: dict[str, bool] = field(default_factory=dict)


@dataclass(slots=True)
class MoleculeContext:
    """Shared per-molecule views used by independent naming rule families."""

    mol: object
    Chem: object
    telemetry: CandidateTelemetry | None = None
    cache: ContextCache = field(default_factory=ContextCache)
    _features: dict | None = None
    _generated_fused_templates: tuple[dict, ...] | None = None
    _ring_systems: tuple[frozenset[int], ...] | None = None
    _ring_topologies: tuple[RingSystemTopology, ...] | None = None
    _profile: StructuralProfile | None = None
    _template_store: TemplateStore | None = None

    def __post_init__(self) -> None:
        if self.telemetry is not None:
            mol_cache_set(self.mol, "candidate_telemetry", self.telemetry)

    @property
    def templates(self) -> TemplateStore:
        if self._template_store is None:
            self._template_store = template_store(self.Chem)
        return self._template_store

    @property
    def features(self) -> dict:
        if self._features is None:
            self._features = molecule_features(self.mol, self.Chem)
        return self._features

    def candidate_templates(self, templates: tuple[dict, ...], *, exact_atom_count: bool = False) -> tuple[dict, ...]:
        key = (id(templates), exact_atom_count)
        cached = self.cache.template_candidates.get(key)
        if cached is None:
            cached = tuple(
                self.templates.candidate_templates(
                    self.mol,
                    templates,
                    features=self.features,
                    exact_atom_count=exact_atom_count,
                )
            )
            self.cache.template_candidates[key] = cached
            if self.telemetry is not None:
                self.telemetry.template_candidate_count[_template_collection_label(templates)] += len(cached)
        return cached

    def candidate_compiled_templates(
        self,
        templates: tuple[dict, ...],
        *,
        exact_atom_count: bool = False,
    ):
        key = (id(templates), exact_atom_count)
        cached = self.cache.compiled_template_candidates.get(key)
        if cached is None:
            cached = self.templates.candidate_compiled_templates(
                self.mol,
                templates,
                features=self.features,
                exact_atom_count=exact_atom_count,
            )
            self.cache.compiled_template_candidates[key] = cached
            if self.telemetry is not None:
                self.telemetry.template_candidate_count[_template_collection_label(templates)] += len(cached)
        return cached

    def acyclic_name(self, *, allow_ring_atoms: bool = True) -> str:
        """Return the best acyclic skeleton name, computed once per molecule.

        Several parent-family rules ask the same "can this molecule be named as
        an acyclic skeleton with substituents?" question before retained-ring
        rules run. Keeping that result on the per-molecule context avoids
        replaying the full parent-path search and branch rendering pipeline.
        """

        cached = self.cache.acyclic_names.get(allow_ring_atoms)
        if cached is None:
            cached = name_acyclic(self.mol, allow_ring_atoms=allow_ring_atoms)
            self.cache.acyclic_names[allow_ring_atoms] = cached
        return cached

    @property
    def profile(self) -> StructuralProfile:
        if self._profile is None:
            self._profile = structural_profile(self.mol, self.Chem)
        return self._profile

    @property
    def ring_systems(self) -> tuple[frozenset[int], ...]:
        if self._ring_systems is None:
            self._ring_systems = ring_systems_for_mol(self.mol, self.Chem)
        return self._ring_systems

    @property
    def ring_topologies(self) -> tuple[RingSystemTopology, ...]:
        if self._ring_topologies is None:
            self._ring_topologies = ring_system_topologies(self.mol, self.Chem)
        return self._ring_topologies

    @property
    def has_generic_bicyclic_topology(self) -> bool:
        cached = self.cache.topology_flags.get("generic_bicyclic")
        if cached is None:
            cached = any(topology.cycle_rank == 2 for topology in self.ring_topologies)
            self.cache.topology_flags["generic_bicyclic"] = cached
        return cached

    @property
    def has_generic_polycyclic_topology(self) -> bool:
        cached = self.cache.topology_flags.get("generic_polycyclic")
        if cached is None:
            cached = any(2 <= topology.cycle_rank <= 7 for topology in self.ring_topologies)
            self.cache.topology_flags["generic_polycyclic"] = cached
        return cached

    @property
    def has_spiro_topology(self) -> bool:
        cached = self.cache.topology_flags.get("spiro")
        if cached is not None:
            return cached
        rings = tuple(frozenset(ring) for ring in self.mol.GetRingInfo().AtomRings())
        for topology in self.ring_topologies:
            if topology.cycle_rank not in {2, 3}:
                continue
            system_rings = [ring for ring in rings if ring <= topology.atoms]
            if any(
                len(left & right) == 1
                for idx, left in enumerate(system_rings)
                for right in system_rings[idx + 1 :]
            ):
                self.cache.topology_flags["spiro"] = True
                return True
        self.cache.topology_flags["spiro"] = False
        return False

    def substruct_matches(self, query, *, use_chirality: bool = False) -> tuple[tuple[int, ...], ...]:
        key = (id(query), use_chirality)
        cached = self.cache.substruct_matches.get(key)
        if cached is not None:
            return cached
        if not use_chirality:
            matches = self.mol.GetSubstructMatches(query, uniquify=False)
        else:
            params = self.Chem.SubstructMatchParameters()
            params.uniquify = False
            params.useChirality = True
            matches = self.mol.GetSubstructMatches(query, params)
        if self.telemetry is not None:
            self.telemetry.substructure_match_calls += 1
            active_rule_family = self.telemetry.active_rule_family or "unknown"
            self.telemetry.substructure_match_calls_by_rule[active_rule_family] += 1
        result = tuple(tuple(match) for match in matches)
        self.cache.substruct_matches[key] = result
        return result

    def template_matches(self, template: dict) -> tuple[tuple[int, ...], ...]:
        return self.substruct_matches(
            template_record_mol(template),
            use_chirality=bool(template.get("stereochemical")),
        )

    def polycyclic_parent_records(self, family: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
        cached = self.cache.polycyclic_records.get(family)
        if cached is not None:
            return cached
        if family == "bicyclic":
            records = tuple(bicyclic_parent_candidate_records(self.mol, self.Chem))
        elif family == "spiro":
            records = tuple(spiro_parent_candidate_records(self.mol, self.Chem))
        elif family == "kekulized":
            records = tuple(kekulized_polycyclic_parent_candidate_records(self.mol, self.Chem))
        else:
            records = ()
        self.cache.polycyclic_records[family] = records
        return records

    @property
    def generated_fused_templates(self) -> tuple[dict, ...]:
        if self._generated_fused_templates is None:
            self._generated_fused_templates = candidate_generated_fused_templates(
                self.mol,
                self.Chem,
                features=self.features,
            )
            if not self._generated_fused_templates:
                self._generated_fused_templates = candidate_generated_fused_templates(
                    self.mol,
                    self.Chem,
                    features=self.features,
                    graph_filter=False,
                )
        return self._generated_fused_templates


def _template_collection_label(templates: tuple[dict, ...]) -> str:
    cache_key = id(templates)
    cached = _TEMPLATE_COLLECTION_LABEL_CACHE.get(cache_key)
    if cached is not None and cached[0] is templates:
        return cached[1]
    if not templates:
        label = "empty"
        _TEMPLATE_COLLECTION_LABEL_CACHE[cache_key] = (templates, label)
        return label
    families = Counter(str(template.get("source_family") or "unknown") for template in templates)
    if len(families) == 1:
        family = next(iter(families))
        label = f"{family}:n={len(templates)}"
        _TEMPLATE_COLLECTION_LABEL_CACHE[cache_key] = (templates, label)
        return label
    top_families = sorted(
        families.items(),
        key=lambda item: (-item[1], item[0]),
    )[:3]
    family_text = "+".join(
        f"{family}:{count}"
        for family, count in top_families
    )
    if len(families) > 3:
        top_count = sum(count for _, count in top_families)
        family_text += f"+other:{sum(families.values()) - top_count}"
    label = f"mixed:{family_text}:n={len(templates)}"
    _TEMPLATE_COLLECTION_LABEL_CACHE[cache_key] = (templates, label)
    return label
