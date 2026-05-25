from __future__ import annotations

from dataclasses import dataclass, field

from .acyclic_rules import _name_acyclic
from .fused_template_families import candidate_generated_fused_templates
from .polycyclic_rules import (
    bicyclic_parent_candidates,
    kekulized_polycyclic_parent_candidates,
    spiro_parent_candidates,
)
from .structural_profile import StructuralProfile, structural_profile
from .template_index import candidate_templates, molecule_features
from .template_index import ring_systems_for_mol
from .ring_topology import RingSystemTopology, ring_system_topologies


@dataclass(slots=True)
class MoleculeContext:
    """Shared per-molecule views used by independent naming rule families."""

    mol: object
    Chem: object
    _features: dict | None = None
    _template_candidates: dict[tuple[int, bool], tuple[dict, ...]] = field(default_factory=dict)
    _generated_fused_templates: tuple[dict, ...] | None = None
    _acyclic_names: dict[bool, str] = field(default_factory=dict)
    _polycyclic_names: dict[str, tuple[str, ...]] = field(default_factory=dict)
    _substruct_matches: dict[tuple[int, bool], tuple[tuple[int, ...], ...]] = field(default_factory=dict)
    _ring_systems: tuple[frozenset[int], ...] | None = None
    _ring_topologies: tuple[RingSystemTopology, ...] | None = None
    _profile: StructuralProfile | None = None

    @property
    def features(self) -> dict:
        if self._features is None:
            self._features = molecule_features(self.mol, self.Chem)
        return self._features

    def candidate_templates(self, templates: tuple[dict, ...], *, exact_atom_count: bool = False) -> tuple[dict, ...]:
        key = (id(templates), exact_atom_count)
        cached = self._template_candidates.get(key)
        if cached is None:
            cached = tuple(
                candidate_templates(
                    self.mol,
                    templates,
                    self.Chem,
                    features=self.features,
                    exact_atom_count=exact_atom_count,
                )
            )
            self._template_candidates[key] = cached
        return cached

    def acyclic_name(self, *, allow_ring_atoms: bool = True) -> str:
        """Return the best acyclic skeleton name, computed once per molecule.

        Several parent-family rules ask the same "can this molecule be named as
        an acyclic skeleton with substituents?" question before retained-ring
        rules run. Keeping that result on the per-molecule context avoids
        replaying the full parent-path search and branch rendering pipeline.
        """

        cached = self._acyclic_names.get(allow_ring_atoms)
        if cached is None:
            cached = _name_acyclic(self.mol, allow_ring_atoms=allow_ring_atoms)
            self._acyclic_names[allow_ring_atoms] = cached
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
        return any(topology.cycle_rank == 2 for topology in self.ring_topologies)

    @property
    def has_generic_polycyclic_topology(self) -> bool:
        return any(topology.cycle_rank in {2, 3} for topology in self.ring_topologies)

    @property
    def has_spiro_topology(self) -> bool:
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
                return True
        return False

    def substruct_matches(self, query, *, use_chirality: bool = False) -> tuple[tuple[int, ...], ...]:
        key = (id(query), use_chirality)
        cached = self._substruct_matches.get(key)
        if cached is not None:
            return cached
        if not use_chirality:
            matches = self.mol.GetSubstructMatches(query, uniquify=False)
        else:
            params = self.Chem.SubstructMatchParameters()
            params.uniquify = False
            params.useChirality = True
            matches = self.mol.GetSubstructMatches(query, params)
        result = tuple(tuple(match) for match in matches)
        self._substruct_matches[key] = result
        return result

    def template_matches(self, template: dict) -> tuple[tuple[int, ...], ...]:
        return self.substruct_matches(
            template["mol"],
            use_chirality=bool(template.get("stereochemical")),
        )

    def polycyclic_parent_names(self, family: str) -> tuple[str, ...]:
        cached = self._polycyclic_names.get(family)
        if cached is not None:
            return cached
        if family == "bicyclic":
            names = tuple(bicyclic_parent_candidates(self.mol, self.Chem))
        elif family == "spiro":
            names = tuple(spiro_parent_candidates(self.mol, self.Chem))
        elif family == "kekulized":
            names = tuple(kekulized_polycyclic_parent_candidates(self.mol, self.Chem))
        else:
            names = ()
        self._polycyclic_names[family] = names
        return names

    @property
    def generated_fused_templates(self) -> tuple[dict, ...]:
        if self._generated_fused_templates is None:
            self._generated_fused_templates = candidate_generated_fused_templates(
                self.mol,
                self.Chem,
                features=self.features,
            )
        return self._generated_fused_templates
