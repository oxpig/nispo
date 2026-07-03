from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json

from .acyclic_rules import AcyclicNameRecord, name_acyclic_record
from .engine_flags import FAST_POLY_BRIDGE_ROUTE_PROP
from .fused_template_families import (
    candidate_generated_fused_templates,
    generated_record_key,
    generated_template_key,
    _generated_records_for_graph,
)
from .molecule_facts import MoleculeFacts, build_molecule_facts
from .polycyclic_rules import (
    bicyclic_parent_candidate_records,
    kekulized_polycyclic_parent_candidate_records,
    spiro_parent_candidate_records,
)
from .resource_templates import build_generated_fused_parent_templates_from_records
from .route_plan import RoutePlan, plan_routes
from .structural_profile import StructuralProfile, structural_profile
from .template_access import template_record_mol, template_record_name
from .template_index import molecule_features
from .template_index import (
    intern_template_tuple,
    ring_system_feature_key,
    ring_system_with_graph_signatures,
    ring_systems_for_mol,
)
from .template_store import TemplateStore, template_store
from .ring_topology import RingSystemTopology, ring_system_topologies
from .telemetry import CandidateTelemetry
from .molecule_cache import mol_cache_set


_TEMPLATE_COLLECTION_LABEL_CACHE: dict[int, tuple[tuple[dict, ...], str]] = {}
_GENERATED_FUSED_GRAPH_SUPPLEMENT_LIMIT = 16
_FAST_ROUTE_KEKULIZED_POLYCYCLIC_PARENT_RECORD_LIMIT = 1
_LARGE_UNCHARGED_KEKULIZED_POLYCYCLIC_PARENT_RECORD_LIMIT = 1
_KEKULIZED_POLYCYCLIC_PARENT_RECORD_LIMIT = 12
_KEKULIZED_POLYCYCLIC_PARENT_RECORD_FULL_LIMIT = 12
_HALOGEN_ATOMIC_NUMBERS = frozenset({9, 17, 35, 53, 85})


def _kekulized_polycyclic_parent_record_limit(mol) -> int:
    if bool(getattr(mol, "HasProp", lambda _name: False)(FAST_POLY_BRIDGE_ROUTE_PROP)):
        return _FAST_ROUTE_KEKULIZED_POLYCYCLIC_PARENT_RECORD_LIMIT
    if (
        mol.GetNumAtoms() >= 30
        and mol.GetRingInfo().NumRings() >= 4
        and not any(atom.GetFormalCharge() for atom in mol.GetAtoms())
    ):
        return _LARGE_UNCHARGED_KEKULIZED_POLYCYCLIC_PARENT_RECORD_LIMIT
    if not any(atom.GetAtomicNum() in _HALOGEN_ATOMIC_NUMBERS for atom in mol.GetAtoms()):
        return _KEKULIZED_POLYCYCLIC_PARENT_RECORD_FULL_LIMIT
    ring_count = mol.GetRingInfo().NumRings()
    chiral_count = _assigned_chiral_atom_count(mol)
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return _KEKULIZED_POLYCYCLIC_PARENT_RECORD_FULL_LIMIT
    if ring_count <= 4 and chiral_count >= 2:
        return _KEKULIZED_POLYCYCLIC_PARENT_RECORD_FULL_LIMIT
    if ring_count >= 10 and chiral_count >= 5:
        return _KEKULIZED_POLYCYCLIC_PARENT_RECORD_FULL_LIMIT
    if ring_count >= 8 and chiral_count >= 6:
        return _KEKULIZED_POLYCYCLIC_PARENT_RECORD_FULL_LIMIT
    if ring_count >= 7 and chiral_count >= 3 and 33 <= mol.GetNumAtoms() <= 36:
        return _KEKULIZED_POLYCYCLIC_PARENT_RECORD_FULL_LIMIT
    return _KEKULIZED_POLYCYCLIC_PARENT_RECORD_LIMIT


def _assigned_chiral_atom_count(mol) -> int:
    return sum(int(atom.GetChiralTag()) != 0 for atom in mol.GetAtoms())


def _polycyclic_parent_record_limit(mol, family: str) -> int | None:
    if family == "kekulized":
        return _kekulized_polycyclic_parent_record_limit(mol)
    return None


@dataclass(slots=True)
class ContextCache:
    template_candidates: dict[tuple[int, bool], tuple[dict, ...]] = field(default_factory=dict)
    compiled_template_candidates: dict[tuple[int, bool], tuple[object, ...]] = field(default_factory=dict)
    acyclic_names: dict[bool, AcyclicNameRecord] = field(default_factory=dict)
    compiled_rules: dict[object, tuple[tuple[object, ...], tuple[object, ...]]] = field(default_factory=dict)
    substituent_like_acyclic_defer: dict[str, bool] = field(default_factory=dict)
    polycyclic_records: dict[tuple[str, int | None], tuple[tuple[str, tuple[int, ...]], ...]] = field(
        default_factory=dict
    )
    substruct_matches: dict[tuple[int, bool], tuple[tuple[int, ...], ...]] = field(default_factory=dict)
    template_matches: dict[int, tuple[tuple[int, ...], ...]] = field(default_factory=dict)
    fused_template_matches: dict[
        tuple[int, bool, bool, bool, bool], tuple[tuple[dict, tuple[int, ...]], ...]
    ] = field(default_factory=dict)
    rule_candidates: dict[tuple[int, tuple[tuple[bool, str], ...]], tuple[object, ...]] = field(
        default_factory=dict
    )
    topology_flags: dict[str, bool] = field(default_factory=dict)

    def shared_for_fallback(self) -> "ContextCache":
        return ContextCache(
            template_candidates=self.template_candidates,
            compiled_template_candidates=self.compiled_template_candidates,
            acyclic_names=self.acyclic_names,
            compiled_rules=self.compiled_rules,
            substruct_matches=self.substruct_matches,
            template_matches=self.template_matches,
            fused_template_matches=self.fused_template_matches,
            rule_candidates=self.rule_candidates,
            topology_flags=self.topology_flags,
        )


@dataclass(slots=True)
class MoleculeContext:
    """Shared per-molecule views used by independent naming rule families."""

    mol: object
    Chem: object
    telemetry: CandidateTelemetry | None = None
    cache: ContextCache = field(default_factory=ContextCache)
    _features: dict | None = None
    _generated_fused_templates: tuple[dict, ...] | None = None
    _narrow_generated_fused_templates: tuple[dict, ...] | None = None
    _ring_systems: tuple[frozenset[int], ...] | None = None
    _ring_topologies: tuple[RingSystemTopology, ...] | None = None
    _profile: StructuralProfile | None = None
    _template_store: TemplateStore | None = None
    _facts: MoleculeFacts | None = None
    _route_plan: RoutePlan | None = None

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

    def acyclic_name_record(self, *, allow_ring_atoms: bool = True) -> AcyclicNameRecord:
        """Return the best acyclic skeleton name, computed once per molecule.

        Several parent-family rules ask the same "can this molecule be named as
        an acyclic skeleton with substituents?" question before retained-ring
        rules run. Keeping that result on the per-molecule context avoids
        replaying the full parent-path search and branch rendering pipeline.
        """

        key = (allow_ring_atoms, _fragment_fallback_state_key(self.mol))
        cached = self.cache.acyclic_names.get(key)
        if cached is None:
            cached = name_acyclic_record(self.mol, allow_ring_atoms=allow_ring_atoms)
            self.cache.acyclic_names[key] = cached
        return cached

    def acyclic_name(self, *, allow_ring_atoms: bool = True) -> str:
        return self.acyclic_name_record(allow_ring_atoms=allow_ring_atoms).name

    @property
    def profile(self) -> StructuralProfile:
        if self._profile is None:
            self._profile = structural_profile(self.mol, self.Chem)
        return self._profile

    @property
    def facts(self) -> MoleculeFacts:
        if self._facts is None:
            self._facts = build_molecule_facts(
                self.mol,
                features=self.features,
                ring_topologies=self.ring_topologies,
                profile=self.profile,
            )
        return self._facts

    @property
    def route_plan(self) -> RoutePlan:
        if self._route_plan is None:
            self._route_plan = plan_routes(self.facts)
        return self._route_plan

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
        return any(topology.cycle_rank == 2 for topology in self.ring_topologies)

    @property
    def has_generic_polycyclic_topology(self) -> bool:
        return any(2 <= topology.cycle_rank <= 9 for topology in self.ring_topologies)

    @property
    def has_spiro_topology(self) -> bool:
        return self._has_spiro_ring_pair(max_cycle_rank=3)

    @property
    def has_component_spiro_topology(self) -> bool:
        return self._has_spiro_ring_pair()

    def _has_spiro_ring_pair(self, *, max_cycle_rank: int | None = None) -> bool:
        cache_key = f"spiro_pair:{max_cycle_rank}"
        cached = self.cache.topology_flags.get(cache_key)
        if cached is not None:
            return cached
        rings = tuple(frozenset(ring) for ring in self.mol.GetRingInfo().AtomRings())
        for topology in self.ring_topologies:
            if topology.cycle_rank < 2:
                continue
            if max_cycle_rank is not None and topology.cycle_rank > max_cycle_rank:
                continue
            system_rings = [ring for ring in rings if ring <= topology.atoms]
            if any(
                len(left & right) == 1
                for idx, left in enumerate(system_rings)
                for right in system_rings[idx + 1 :]
            ):
                self.cache.topology_flags[cache_key] = True
                return True
        self.cache.topology_flags[cache_key] = False
        return False

    def substruct_matches(self, query, *, use_chirality: bool = False) -> tuple[tuple[int, ...], ...]:
        key = (id(query), use_chirality)
        cache = self.cache.substruct_matches
        cached = cache.get(key)
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
        result = matches if isinstance(matches, tuple) else tuple(tuple(match) for match in matches)
        cache[key] = result
        return result

    def template_matches(self, template: dict) -> tuple[tuple[int, ...], ...]:
        key = id(template)
        cached = self.cache.template_matches.get(key)
        if cached is not None:
            return cached
        cached = self.substruct_matches(
            template_record_mol(template),
            use_chirality=bool(template.get("stereochemical")),
        )
        self.cache.template_matches[key] = cached
        return cached

    def polycyclic_parent_records(self, family: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
        limit = _polycyclic_parent_record_limit(self.mol, family)
        cache_key = (family, limit)
        cached = self.cache.polycyclic_records.get(cache_key)
        if cached is not None:
            return cached
        if family == "bicyclic":
            records = tuple(bicyclic_parent_candidate_records(self.mol, self.Chem))
        elif family == "spiro":
            records = tuple(spiro_parent_candidate_records(self.mol, self.Chem))
        elif family == "kekulized":
            records = tuple(
                kekulized_polycyclic_parent_candidate_records(
                    self.mol,
                    self.Chem,
                    limit=limit,
                )
            )
        else:
            records = ()
        self.cache.polycyclic_records[cache_key] = records
        return records

    @property
    def generated_fused_templates(self) -> tuple[dict, ...]:
        if self._generated_fused_templates is None:
            self._generated_fused_templates = self.narrow_generated_fused_templates
            if not self._generated_fused_templates:
                self._generated_fused_templates = candidate_generated_fused_templates(
                    self.mol,
                    self.Chem,
                    features=self.features,
                    graph_filter=False,
                )
        return self._generated_fused_templates

    @property
    def narrow_generated_fused_templates(self) -> tuple[dict, ...]:
        if self._narrow_generated_fused_templates is None:
            narrow = candidate_generated_fused_templates(
                self.mol,
                self.Chem,
                features=self.features,
            )
            self._narrow_generated_fused_templates = _supplement_generated_fused_templates_by_ring_graph(
                self.mol,
                self.Chem,
                self.features,
                narrow,
            )
        return self._narrow_generated_fused_templates


def _supplement_generated_fused_templates_by_ring_graph(
    mol,
    Chem,
    features: dict,
    templates: tuple[dict, ...],
) -> tuple[dict, ...]:
    """Add same-topology generated fused aliases missing from element graph keys."""

    supplement_records = []
    candidate_records_found = False
    seen_record_ids: set[int] = set()
    seen_record_keys: set[tuple[str, str]] = set()
    record_limit = _GENERATED_FUSED_GRAPH_SUPPLEMENT_LIMIT
    for system in features["ring_systems"]:
        key = json.dumps(ring_system_feature_key(system), separators=(",", ":"))
        system = ring_system_with_graph_signatures(mol, system, Chem)
        graph_key = system.get("graph_signature", "")
        if not graph_key:
            continue
        records = _generated_records_for_graph(key, graph_key, "", False, Chem)
        if not records:
            continue
        candidate_records_found = True
        for record in records:
            name = str(record.get("name", ""))
            smiles = str(record.get("smiles", ""))
            if not name:
                continue
            record_id = id(record)
            record_key = (name, smiles)
            if record_id in seen_record_ids or record_key in seen_record_keys:
                continue
            seen_record_ids.add(record_id)
            seen_record_keys.add(record_key)
            supplement_records.append(record)
    if not candidate_records_found:
        return templates

    seen: set[tuple[str, str]] = {
        (
            template_record_name(template),
            Chem.MolToSmiles(template_record_mol(template), canonical=True),
        )
        for template in templates
    }
    supplements = []
    sorted_records = sorted(supplement_records, key=generated_record_key)
    for start in range(0, len(sorted_records), record_limit):
        record_batch = tuple(sorted_records[start : start + record_limit])
        for template in build_generated_fused_parent_templates_from_records(
            Chem,
            record_batch,
            source_file="generated_fused_template_index.json",
        ):
            marker = (
                template_record_name(template),
                Chem.MolToSmiles(template_record_mol(template), canonical=True),
            )
            if marker in seen:
                continue
            seen.add(marker)
            supplements.append(template)
            if len(supplements) >= record_limit:
                break
        if len(supplements) >= record_limit:
            break
    if supplements:
        supplements = sorted(supplements, key=generated_template_key)[:record_limit]
    return intern_template_tuple(sorted(tuple(templates) + tuple(supplements), key=generated_template_key))


def _fragment_fallback_state_key(mol) -> tuple[bool, str, bool, str, bool]:
    return (
        mol.HasProp("_nispo_allow_fragment_substituent_fallback"),
        mol.GetProp("_nispo_allow_fragment_substituent_fallback")
        if mol.HasProp("_nispo_allow_fragment_substituent_fallback")
        else "",
        mol.HasProp("_nispo_fragment_substituent_depth"),
        mol.GetProp("_nispo_fragment_substituent_depth") if mol.HasProp("_nispo_fragment_substituent_depth") else "",
        mol.HasProp("_nispo_fragment_substituent_view"),
    )


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
