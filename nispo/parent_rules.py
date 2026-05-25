from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from itertools import combinations
from functools import lru_cache

from .branch_rules import branch_prefix
from .constants import HALO, MULT, ROOTS
from .cycle_utils import ordered_ring, ring_system_extends_core
from .formatting import format_prefixes as _format_prefixes
from .formatting import loc_unsat
from .molecule_cache import mol_cache_get, mol_cache_set
from .fused_template_families import (
    curated_fused_templates as _curated_fused_templates,
    generated_fusion_preference as _generated_fusion_preference,
    hydrocarbon_replacement_parent_templates_for_graph as _hydrocarbon_replacement_parent_templates_for_graph,
)
from .ring_resource_rules import resource_ring_parent_candidates as _resource_ring_parent_candidates
from .ring_rules import (
    adjust_ring_parent_for_charge as _adjust_ring_parent_for_charge,
    ring_parent_candidates as _ring_parent_candidates,
)
from .rule_dispatch import (
    RuleSpec,
    acyclic,
    acyclic_preference_signal,
    charged_ring_atom,
    exact_ring_system,
    functional_skeleton_signal,
    generic_bicyclic_topology,
    generic_polycyclic_topology,
    imino_skeleton_signal,
    kekulized_polycycle_signal,
    min_ring_count,
    min_ring_system_ring_count,
    no_charged_ring_atom,
    no_exocyclic_charged_fused_parent,
    nonring_positive_nitrogen,
    nonring_positive_sulfur,
    spiro_topology,
)
from .rule_engine import NameCandidate
from .template_access import template_record_locants, template_record_mol, template_record_name


class AcyclicParentRule:
    family = "acyclic-parent"
    priority = 10
    acyclic_only = True

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() != 0:
            return
        name = ctx.acyclic_name()
        if name:
            yield NameCandidate(name, self.family, self.priority)


class SulfaniumParentRule:
    family = "sulfanium-parent"
    priority = 9
    requires_nonring_positive_sulfur = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        sulfur_atoms = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "S" and atom.GetFormalCharge() > 0 and not atom.IsInRing()
        ]
        if len(sulfur_atoms) != 1:
            return
        sulfur = sulfur_atoms[0]
        sidx = sulfur.GetIdx()
        neighbors = [nbr.GetIdx() for nbr in sulfur.GetNeighbors()]
        if len(neighbors) != 3:
            return
        if any(mol.GetAtomWithIdx(nidx).GetSymbol() == "S" for nidx in neighbors):
            return
        if any(
            mol.GetBondBetweenAtoms(sidx, nidx).GetBondType() != self._Chem.BondType.SINGLE
            for nidx in neighbors
        ):
            return
        names = []
        for nidx in neighbors:
            prefix = branch_prefix(mol, nidx, sidx)
            if not prefix:
                return
            names.append(prefix)
        name = _sulfanium_name(names)
        if name:
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class OxidoNitrosoAzaniumRule:
    family = "oxido-nitroso-azanium-parent"
    priority = 9
    acyclic_only = True
    requires_nonring_positive_nitrogen = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() != 0:
            return
        for atom in mol.GetAtoms():
            nidx = atom.GetIdx()
            if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
                continue
            oxido = []
            nitroso = []
            organic = []
            ok = True
            for nbr in atom.GetNeighbors():
                nbidx = nbr.GetIdx()
                bond = mol.GetBondBetweenAtoms(nidx, nbidx)
                if (
                    nbr.GetSymbol() == "O"
                    and nbr.GetFormalCharge() < 0
                    and bond.GetBondType() == self._Chem.BondType.SINGLE
                ):
                    oxido.append(nbidx)
                elif nbr.GetSymbol() == "N" and bond.GetBondType() == self._Chem.BondType.SINGLE:
                    oxy = [
                        onbr.GetIdx()
                        for onbr in nbr.GetNeighbors()
                        if onbr.GetIdx() != nidx
                        and onbr.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(nbidx, onbr.GetIdx()).GetBondType() == self._Chem.BondType.DOUBLE
                    ]
                    if len(oxy) == 1:
                        nitroso.append(nbidx)
                    else:
                        ok = False
                        break
                else:
                    organic.append(nbidx)
            if not ok or len(oxido) != 1 or len(nitroso) != 1 or len(organic) != 1:
                continue
            prefix = branch_prefix(mol, organic[0], nidx)
            if not prefix:
                continue
            name = f"{prefix}(oxido)(nitroso)azanium"
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class OxoSulfaniumParentRule:
    family = "oxo-sulfanium-parent"
    priority = 9
    requires_nonring_positive_sulfur = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        for atom in mol.GetAtoms():
            sidx = atom.GetIdx()
            if atom.GetSymbol() != "S" or atom.GetFormalCharge() <= 0 or atom.IsInRing():
                continue
            oxo = []
            heavy = []
            ok = True
            for nbr in atom.GetNeighbors():
                nidx = nbr.GetIdx()
                bond = mol.GetBondBetweenAtoms(sidx, nidx)
                if nbr.GetSymbol() == "O" and bond.GetBondType() == self._Chem.BondType.DOUBLE:
                    oxo.append(nidx)
                elif bond.GetBondType() == self._Chem.BondType.SINGLE:
                    heavy.append(nidx)
                else:
                    ok = False
                    break
            if not ok or len(oxo) != 1 or len(heavy) != 1:
                continue
            prefix = branch_prefix(mol, heavy[0], sidx)
            if not prefix:
                continue
            if _needs_substituent_parentheses(prefix):
                prefix = f"({prefix})"
            name = f"{prefix}-oxosulfanium"
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class NHaloImineParentRule:
    family = "n-halo-imine-parent"
    priority = 9
    acyclic_only = True
    requires_imino_skeleton_signal = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() != 0:
            return
        for carbon in mol.GetAtoms():
            cidx = carbon.GetIdx()
            if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
                continue
            imine_n = []
            carbon_neighbors = []
            ok = True
            for nbr in carbon.GetNeighbors():
                nidx = nbr.GetIdx()
                bond = mol.GetBondBetweenAtoms(cidx, nidx)
                if nbr.GetSymbol() == "N" and bond.GetBondType() == self._Chem.BondType.DOUBLE:
                    imine_n.append(nidx)
                elif nbr.GetSymbol() == "C" and bond.GetBondType() == self._Chem.BondType.SINGLE:
                    carbon_neighbors.append(nidx)
                else:
                    ok = False
                    break
            if not ok or len(imine_n) != 1 or len(carbon_neighbors) != 2:
                continue
            halo_idx = _n_halo_atom(mol, imine_n[0], cidx)
            if halo_idx < 0 or not _n_halo_prefix(mol, imine_n[0], cidx, self._Chem):
                continue
            n_prefix = f"N-{HALO[mol.GetAtomWithIdx(halo_idx).GetSymbol()]}"
            names = []
            for path in ((carbon_neighbors[0], cidx, carbon_neighbors[1]), (carbon_neighbors[1], cidx, carbon_neighbors[0])):
                prefixes = []
                if not _append_terminal_halo_prefixes(mol, path[0], cidx, "1", prefixes, self._Chem):
                    continue
                if not _append_terminal_halo_prefixes(mol, path[2], cidx, "3", prefixes, self._Chem):
                    continue
                prefix_text = _format_prefixes(prefixes)
                pieces = [n_prefix]
                if prefix_text:
                    pieces.append(prefix_text)
                pieces.append("propan-2-imine")
                names.append("-".join(pieces))
            if names:
                name = min(names, key=lambda item: (len(item), item))
                yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class GuanidineRetainedParentRule:
    family = "guanidine-retained-parent"
    priority = 9
    acyclic_only = True
    requires_imino_skeleton_signal = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() != 0:
            return
        for template in _guanidine_parent_templates(self._Chem):
            if template_record_mol(template).GetNumAtoms() > mol.GetNumAtoms():
                continue
            for match in ctx.template_matches(template):
                name = _name_acyclic_retained_core_match(mol, template, match)
                if not name:
                    continue
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=(-template_record_mol(template).GetNumAtoms(), len(name), name),
                )


class AcyclicSkeletonWithRingSubstituentsRule:
    family = "acyclic-skeleton-with-ring-substituents"
    priority = 60
    min_ring_count = 1

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() == 0:
            return
        name = ctx.acyclic_name(allow_ring_atoms=False)
        if name:
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class FunctionalAcyclicSkeletonWithRingSubstituentsRule(AcyclicSkeletonWithRingSubstituentsRule):
    family = "functional-acyclic-skeleton-with-ring-substituents"
    priority = 14
    requires_functional_skeleton_signal = True
    requires_acyclic_preference_signal = True
    excludes_charged_ring_atom = True

    def candidates(self, ctx):
        mol = ctx.mol
        profile = ctx.profile
        if (
            profile.ring_count == 0
            or profile.has_charged_ring_atom
            or not profile.has_functional_skeleton_signal
            or not profile.may_prefer_acyclic_skeleton
        ):
            return
        name = ctx.acyclic_name(allow_ring_atoms=False)
        if name and _prefer_acyclic_skeleton_name(name):
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class IminoFunctionalAcyclicSkeletonRule(FunctionalAcyclicSkeletonWithRingSubstituentsRule):
    family = "imino-functional-acyclic-skeleton-with-ring-substituents"
    priority = 13
    defer_to_exocyclic_charged_fused_parent = True
    requires_imino_skeleton_signal = True

    def candidates(self, ctx):
        mol = ctx.mol
        profile = ctx.profile
        if (
            profile.ring_count == 0
            or not profile.has_imino_skeleton_signal
            or not profile.may_prefer_acyclic_skeleton
        ):
            return
        name = ctx.acyclic_name(allow_ring_atoms=False)
        if name and "imino" in name and _prefer_acyclic_skeleton_name(name):
            yield NameCandidate(name, self.family, self.priority, sort_key=(len(name), name))


class FusedParentRule:
    family = "fused-retained-parent"
    priority = 20
    min_ring_count = 2

    def __init__(self, Chem):
        self._Chem = Chem

    def _templates(self) -> tuple[dict, ...]:
        return _curated_fused_templates(self._Chem)

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.candidate_templates(self._templates())

    def _candidate_sort_key(self, template: dict, match: Sequence[int], name: str) -> tuple:
        return (
            *_fused_candidate_preference(name),
            _template_source_rank(template),
            _hydrogenated_template_penalty(template_record_name(template)),
            -len(match),
            name,
        )

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        for template, match in _iter_fused_template_matches(
            ctx, self._template_iter(ctx), self._Chem, reject_extra_core_bonds=True
        ):
            name = _name_fused_core_match(mol, template, match)
            if name:
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=self._candidate_sort_key(template, match, name),
                )


class GeneratedFusedParentRule(FusedParentRule):
    """Fallback OPSIN-fusion scaffold family.

    These templates are generated in bulk from OPSIN fusion components. They
    broaden coverage, but they should not displace curated retained parents
    whose locant conventions are hand-checked more carefully.
    """

    family = "generated-fused-parent"
    priority = 29
    min_ring_system_ring_count = 2

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.generated_fused_templates

    def _candidate_sort_key(self, template: dict, match: Sequence[int], name: str) -> tuple:
        return (_generated_fusion_preference(template_record_name(template)), *_fused_candidate_preference(name), -len(match), name)


class SingleRingParentRule:
    family = "single-ring-parent"
    priority = 30
    min_ring_count = 1

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 1:
            return
        for ring in mol.GetRingInfo().AtomRings():
            for name, stereo_prefix in _named_ring_instance_names(mol, tuple(ring), self._Chem):
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=_single_ring_candidate_sort_key(name),
                    stereo_prefix=stereo_prefix,
                    )


class HydroFusedParentRule:
    family = "hydro-fused-retained-parent"
    priority = 21
    min_ring_count = 2
    min_ring_system_ring_count = 2

    def __init__(self, Chem):
        self._Chem = Chem

    def _templates(self) -> tuple[dict, ...]:
        return _curated_fused_templates(self._Chem)

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.candidate_templates(self._templates(), exact_atom_count=True)

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        for template, match in _iter_fused_template_matches(
            ctx,
            self._template_iter(ctx),
            self._Chem,
            topology_query=True,
            require_one_ring_system=True,
            require_fully_aromatic=True,
        ):
            parent_name = _hydro_parent_name(mol, template, match)
            if not parent_name:
                continue
            name = _name_fused_core_match(mol, template, match, parent_name=parent_name)
            if name:
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=(-len(match), name),
                )


class GeneratedHydroFusedParentRule(HydroFusedParentRule):
    family = "generated-hydro-fused-parent"
    priority = 29
    min_ring_system_ring_count = 2

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.generated_fused_templates


class GeneratedPartialHydroFusedParentRule(GeneratedHydroFusedParentRule):
    family = "generated-partial-hydro-fused-parent"
    priority = 31

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        candidates = []
        for template, match in _iter_fused_template_matches(
            ctx,
            ctx.generated_fused_templates,
            self._Chem,
            topology_query=True,
            require_one_ring_system=True,
        ):
            for parent_name, hydro_key in _partial_hydro_parent_names(mol, template, match):
                name = _name_fused_core_match(mol, template, match, parent_name=parent_name)
                if not name:
                    continue
                fusion_key = _partial_hydro_fusion_preference(template_record_name(template))
                sort_key = (fusion_key, hydro_key, -len(match), name)
                candidates.append(
                    NameCandidate(
                        name=name,
                        family=self.family,
                        priority=self.priority,
                        sort_key=sort_key,
                    )
                )
        seen = set()
        yielded = 0
        for candidate in sorted(candidates, key=lambda item: item.sort_key):
            if candidate.name in seen:
                continue
            seen.add(candidate.name)
            yield candidate
            yielded += 1
            if yielded >= 16:
                return


class SkeletalReplacementFusedParentRule(FusedParentRule):
    family = "skeletal-replacement-fused-parent"
    priority = 32
    min_ring_system_ring_count = 3

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        yielded = 0
        for atom_set, feature in zip(ctx.ring_systems, ctx.features["ring_systems"], strict=True):
            if feature["ring_count"] != 3 or set(feature["elements"]) <= {"C"}:
                continue
            key = (feature["atom_count"], feature["ring_count"], feature.get("graph_signature", ""))
            for template in _hydrocarbon_replacement_parent_templates_for_graph(*key, self._Chem):
                query = _skeletal_replacement_query(template_record_mol(template), self._Chem)
                if query.GetNumAtoms() != len(atom_set):
                    continue
                for match in ctx.substruct_matches(query):
                    if set(match) != set(atom_set):
                        continue
                    parent_name = _skeletal_replacement_parent_name(mol, template, match)
                    if not parent_name:
                        continue
                    name = _name_fused_core_match(mol, template, match, parent_name=parent_name)
                    if not name:
                        continue
                    yielded += 1
                    yield NameCandidate(
                        name=name,
                        family=self.family,
                        priority=self.priority,
                        sort_key=(
                            _skeletal_replacement_sort_key(mol, match, parent_name),
                            _substituent_locant_tuple(name),
                            len(name),
                            name,
                        ),
                    )
                    if yielded >= 12:
                        return


class OxidoFusedParentRule:
    family = "oxido-fused-retained-parent"
    priority = 22
    min_ring_count = 2
    min_ring_system_ring_count = 2
    requires_charged_ring_atom = True

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        templates = ctx.candidate_templates(_curated_fused_templates(self._Chem))
        for template, match in _iter_fused_template_matches(
            ctx,
            templates,
            self._Chem,
            topology_query=True,
            require_one_ring_system=True,
        ):
            if not _has_n_oxide_on_match(mol, match):
                continue
            reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
            parent_name = _adjust_fused_parent_for_charge(mol, template, reverse)
            if parent_name == template_record_name(template):
                continue
            name = _name_fused_core_match(mol, template, match, parent_name=parent_name)
            if name:
                yield NameCandidate(
                    name=name,
                    family=self.family,
                    priority=self.priority,
                    sort_key=(-len(match), name),
                )


class _PolycyclicRecordParentRule:
    record_family = ""
    min_ring_count = 2

    def __init__(self, Chem):
        self._Chem = Chem

    def _candidate_priority(self, mol) -> int | None:
        return self.priority

    def _accept_record(self, name: str, seq: Sequence[int]) -> bool:
        return True

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() < 2:
            return
        priority = self._candidate_priority(mol)
        if priority is None:
            return
        for name, seq in ctx.polycyclic_parent_records(self.record_family):
            if not self._accept_record(name, seq):
                continue
            yield NameCandidate(
                name=name,
                family=self.family,
                priority=priority,
                sort_key=(len(name), name),
                stereo_prefix=_polycyclic_stereo_prefix(mol, seq, self._Chem),
            )


class GenericBicyclicParentRule(_PolycyclicRecordParentRule):
    family = "generic-bicyclic-parent"
    priority = 25
    record_family = "bicyclic"
    requires_generic_bicyclic_topology = True


class PreferredGenericBicyclicParentRule(GenericBicyclicParentRule):
    family = "preferred-generic-bicyclic-parent"
    priority = 19
    min_ring_system_ring_count = 2

    def _accept_record(self, name: str, seq: Sequence[int]) -> bool:
        return "azabicyclo[3.1.0]hexane" in name


class GenericSpiroParentRule(_PolycyclicRecordParentRule):
    family = "generic-spiro-parent"
    priority = 26
    record_family = "spiro"
    min_ring_system_ring_count = 2
    requires_spiro_topology = True


class KekulizedGenericPolycyclicParentRule(_PolycyclicRecordParentRule):
    family = "kekulized-generic-polycyclic-parent"
    priority = 14
    record_family = "kekulized"
    defer_to_exocyclic_charged_fused_parent = True
    requires_kekulized_polycycle_signal = True
    requires_generic_polycyclic_topology = True

    def _candidate_priority(self, mol) -> int | None:
        ring_count = mol.GetRingInfo().NumRings()
        aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
        if not aromatic and ring_count < 3:
            return None
        return self.priority if aromatic else 28


class ExactRetainedParentRule:
    family = "exact-retained-parent"
    priority = 45
    min_ring_count = 1
    requires_exact_ring_system = True

    def __init__(self, Chem):
        self._Chem = Chem

    def _templates(self) -> tuple[dict, ...]:
        return _curated_exact_parent_templates(self._Chem)

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.candidate_templates(self._templates(), exact_atom_count=True)

    def _sort_key(self, template: dict) -> tuple:
        return (len(template_record_name(template)), template.get("source_family", ""), template_record_name(template))

    def candidates(self, ctx):
        mol = ctx.mol
        if mol.GetRingInfo().NumRings() == 0 or not ctx.features["all_atoms_in_one_ring_system"]:
            return
        atom_count = mol.GetNumAtoms()
        atom_set = set(range(atom_count))
        for template in self._template_iter(ctx):
            if template.get("stereochemical") and not _has_stereo_evidence(mol, self._Chem):
                continue
            if template_record_mol(template).GetNumAtoms() != atom_count:
                continue
            for match in ctx.template_matches(template):
                if set(match) == atom_set and not _match_has_extra_core_bonds(mol, template, match):
                    yield NameCandidate(
                        name=template_record_name(template),
                        family=self.family,
                        priority=self.priority,
                        sort_key=self._sort_key(template),
                    )


class GeneratedExactParentRule(ExactRetainedParentRule):
    family = "generated-exact-parent"
    priority = 29

    def _template_iter(self, ctx) -> Sequence[dict]:
        return ctx.generated_fused_templates

    def _sort_key(self, template: dict) -> tuple:
        return _generated_exact_parent_sort_key(template)


def parent_rules(Chem) -> tuple:
    sulfanium = SulfaniumParentRule(Chem)
    oxido_nitroso_azanium = OxidoNitrosoAzaniumRule(Chem)
    oxo_sulfanium = OxoSulfaniumParentRule(Chem)
    n_halo_imine = NHaloImineParentRule(Chem)
    guanidine = GuanidineRetainedParentRule(Chem)
    acyclic_parent = AcyclicParentRule()
    imino_functional_acyclic = IminoFunctionalAcyclicSkeletonRule()
    exact_retained = ExactRetainedParentRule(Chem)
    preferred_bicyclic = PreferredGenericBicyclicParentRule(Chem)
    fused = FusedParentRule(Chem)
    hydro_fused = HydroFusedParentRule(Chem)
    oxido_fused = OxidoFusedParentRule(Chem)
    generic_bicyclic = GenericBicyclicParentRule(Chem)
    generic_spiro = GenericSpiroParentRule(Chem)
    single_ring = SingleRingParentRule(Chem)
    generated_exact = GeneratedExactParentRule(Chem)
    generated_fused = GeneratedFusedParentRule(Chem)
    generated_hydro_fused = GeneratedHydroFusedParentRule(Chem)
    generated_partial_hydro = GeneratedPartialHydroFusedParentRule(Chem)
    skeletal_replacement = SkeletalReplacementFusedParentRule(Chem)
    kekulized_polycyclic = KekulizedGenericPolycyclicParentRule(Chem)
    functional_acyclic = FunctionalAcyclicSkeletonWithRingSubstituentsRule()
    acyclic_with_ring_substituents = AcyclicSkeletonWithRingSubstituentsRule()

    return (
        RuleSpec("sulfanium-parent", 9, (nonring_positive_sulfur,), sulfanium.candidates),
        RuleSpec(
            "oxido-nitroso-azanium-parent",
            9,
            (acyclic, nonring_positive_nitrogen),
            oxido_nitroso_azanium.candidates,
        ),
        RuleSpec("oxo-sulfanium-parent", 9, (nonring_positive_sulfur,), oxo_sulfanium.candidates),
        RuleSpec("n-halo-imine-parent", 9, (acyclic, imino_skeleton_signal), n_halo_imine.candidates),
        RuleSpec("guanidine-retained-parent", 9, (acyclic, imino_skeleton_signal), guanidine.candidates),
        RuleSpec("acyclic-parent", 10, (acyclic,), acyclic_parent.candidates),
        RuleSpec(
            "imino-functional-acyclic-skeleton-with-ring-substituents",
            13,
            (imino_skeleton_signal, no_exocyclic_charged_fused_parent),
            imino_functional_acyclic.candidates,
        ),
        RuleSpec("exact-retained-parent", 45, (min_ring_count(1), exact_ring_system), exact_retained.candidates),
        RuleSpec(
            "preferred-generic-bicyclic-parent",
            19,
            (min_ring_system_ring_count(2), generic_bicyclic_topology),
            preferred_bicyclic.candidates,
        ),
        RuleSpec("fused-retained-parent", 20, (min_ring_count(2),), fused.candidates),
        RuleSpec("hydro-fused-retained-parent", 21, (min_ring_count(2), min_ring_system_ring_count(2)), hydro_fused.candidates),
        RuleSpec(
            "oxido-fused-retained-parent",
            22,
            (min_ring_count(2), min_ring_system_ring_count(2), charged_ring_atom),
            oxido_fused.candidates,
        ),
        RuleSpec("generic-bicyclic-parent", 25, (generic_bicyclic_topology,), generic_bicyclic.candidates),
        RuleSpec("generic-spiro-parent", 26, (min_ring_system_ring_count(2), spiro_topology), generic_spiro.candidates),
        RuleSpec("single-ring-parent", 30, (min_ring_count(1),), single_ring.candidates),
        RuleSpec("generated-exact-parent", 29, (min_ring_count(1), exact_ring_system), generated_exact.candidates),
        RuleSpec("generated-fused-parent", 29, (min_ring_system_ring_count(2),), generated_fused.candidates),
        RuleSpec("generated-hydro-fused-parent", 29, (min_ring_system_ring_count(2),), generated_hydro_fused.candidates),
        RuleSpec("generated-partial-hydro-fused-parent", 31, (min_ring_system_ring_count(2),), generated_partial_hydro.candidates),
        RuleSpec("skeletal-replacement-fused-parent", 32, (min_ring_system_ring_count(3),), skeletal_replacement.candidates),
        RuleSpec(
            "kekulized-generic-polycyclic-parent",
            14,
            (kekulized_polycycle_signal, generic_polycyclic_topology, no_exocyclic_charged_fused_parent),
            kekulized_polycyclic.candidates,
        ),
        RuleSpec(
            "functional-acyclic-skeleton-with-ring-substituents",
            14,
            (functional_skeleton_signal, acyclic_preference_signal, no_charged_ring_atom),
            functional_acyclic.candidates,
        ),
        RuleSpec("acyclic-skeleton-with-ring-substituents", 60, (min_ring_count(1),), acyclic_with_ring_substituents.candidates),
    )


_BICYCLIC_FUSION_BASES = {
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
}


def _generated_exact_parent_sort_key(template: dict) -> tuple:
    name = template_record_name(template)
    base = _fusion_base_name(name)
    base_rank = 0 if base in _BICYCLIC_FUSION_BASES else 1
    return (base_rank, len(name), template.get("source_family", ""), name)


def _single_ring_candidate_sort_key(name: str) -> tuple:
    charged_parent_rank = 0 if re.search(r"(?:^|-)[^-]+-\d+-ium(?:$|-)", name) else 1
    ylidene_rank = 0 if "ylidene" in name else 1
    return (charged_parent_rank, ylidene_rank, len(name), name)


def _fusion_base_name(name: str) -> str:
    if "]" not in name:
        return name
    return name.rsplit("]", 1)[1]


def _partial_hydro_fusion_preference(name: str) -> tuple:
    key = _generated_fusion_preference(name)
    spiro_penalty = _partial_hydro_spiro_alias_penalty(name)
    if len(key) < 5:
        return (spiro_penalty, key)
    base_rank, letter_rank, locants, component_rank, full_name = key
    return (spiro_penalty, base_rank, -letter_rank, locants, component_rank, full_name)


def _partial_hydro_spiro_alias_penalty(name: str) -> int:
    if not name.startswith("spiro["):
        return 0
    if re.search(r"-[0-9]+-en(?:e)?(?:\]|$)", name):
        return 30
    if re.search(r"(?:indan|tetralin|decalin|naphthalan)(?:\]|$)", name):
        return 20
    return 0


@lru_cache(maxsize=8)
def _curated_exact_parent_templates(Chem) -> tuple[dict, ...]:
    from .scaffold_catalog import parent_scaffold_catalog

    return tuple(
        template
        for template in parent_scaffold_catalog(Chem).exact
        if template.get("source_family") != "generated-fused"
    )


@lru_cache(maxsize=8)
def _guanidine_parent_templates(Chem) -> tuple[dict, ...]:
    names = {
        "guanidin",
        "guanidinium",
        "biguanide",
        "triguanide",
        "tetraguanide",
        "pentaguanide",
    }
    return tuple(
        template for template in _curated_exact_parent_templates(Chem) if template_record_name(template) in names
    )


def _iter_fused_template_matches(
    ctx,
    templates: Sequence[dict],
    Chem,
    *,
    topology_query: bool = False,
    require_one_ring_system: bool = False,
    require_fully_aromatic: bool = False,
    reject_extra_core_bonds: bool = False,
):
    mol = ctx.mol
    for template in templates:
        template_mol = template_record_mol(template)
        if template.get("stereochemical") and not _has_stereo_evidence(mol, Chem):
            continue
        if require_one_ring_system and not _template_has_one_ring_system(template_mol):
            continue
        if require_fully_aromatic and not _template_is_fully_aromatic(template_mol):
            continue
        if topology_query:
            query = _topology_query(template_mol, Chem)
            if query.GetNumAtoms() > mol.GetNumAtoms():
                continue
            matches = ctx.substruct_matches(query)
        else:
            matches = ctx.template_matches(template)
        for match in matches:
            if reject_extra_core_bonds and _match_has_extra_core_bonds(mol, template, match):
                continue
            yield template, match


def _match_has_extra_core_bonds(mol, template: dict, match: Sequence[int]) -> bool:
    core_atoms = set(match)
    core_bonds = sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in core_atoms and bond.GetEndAtomIdx() in core_atoms
    )
    return core_bonds > template_record_mol(template).GetNumBonds()


def _name_acyclic_retained_core_match(mol, template: dict, match: Sequence[int]) -> str:
    core_set = set(match)
    prefixes = []
    for template_idx, mol_idx in enumerate(match):
        locant = template_record_locants(template).get(template_idx, "")
        for nbr in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core_set:
                continue
            if not locant:
                return ""
            if _subtree_has_cumulated_diene(mol, nidx, mol_idx):
                return ""
            prefix = branch_prefix(mol, nidx, mol_idx)
            if not prefix:
                return ""
            prefixes.append((locant, prefix))
    prefix_text = _format_prefixes(prefixes)
    return prefix_text + "-" + template_record_name(template) if prefix_text else template_record_name(template)


def _subtree_has_cumulated_diene(mol, start_idx: int, blocked_idx: int) -> bool:
    seen = {blocked_idx}
    stack = [start_idx]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "C":
            double_carbon_bonds = 0
            for nbr in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if nbr.GetSymbol() == "C" and bond.GetBondTypeAsDouble() == 2.0:
                    double_carbon_bonds += 1
            if double_carbon_bonds >= 2:
                return True
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in seen:
                stack.append(nidx)
    return False


@lru_cache(maxsize=None)
def _skeletal_replacement_query(template_mol, Chem):
    query = Chem.RWMol()
    for _atom in template_mol.GetAtoms():
        query.AddAtom(Chem.AtomFromSmarts("[#6,#7,#8,#16]"))
    for bond in template_mol.GetBonds():
        query.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.UNSPECIFIED)
    return query.GetMol()


def _skeletal_replacement_parent_name(mol, template: dict, match: Sequence[int]) -> str:
    grouped: dict[str, list[str]] = {}
    for template_idx, mol_idx in enumerate(match):
        symbol = mol.GetAtomWithIdx(mol_idx).GetSymbol()
        if symbol == "C":
            continue
        replacement = _SKELETAL_REPLACEMENT_PREFIXES.get(symbol)
        locant = template_record_locants(template).get(template_idx, "")
        if not replacement or not locant:
            return ""
        grouped.setdefault(replacement, []).append(locant)
    if not grouped:
        return ""
    parts = []
    for replacement in sorted(grouped):
        locants = sorted(grouped[replacement], key=_locant_sort_key)
        multiplier = MULT.get(len(locants), "")
        parts.append(f"{','.join(locants)}-{multiplier}{replacement}")
    return "-".join(parts) + template_record_name(template)


def _skeletal_replacement_sort_key(mol, match: Sequence[int], parent_name: str) -> tuple:
    saturation_penalty = _skeletal_replacement_saturation_penalty(mol, match, parent_name)
    spiro_norbornene_penalty = _spiro_norbornene_locant_penalty(parent_name)
    replacement_count = sum(parent_name.count(term) for term in _SKELETAL_REPLACEMENT_PREFIXES.values())
    fusion_locant_penalty = parent_name.count("a-")
    return (
        saturation_penalty,
        spiro_norbornene_penalty,
        fusion_locant_penalty,
        replacement_count,
        len(parent_name),
        parent_name,
    )


def _spiro_norbornene_locant_penalty(parent_name: str) -> int:
    if "spiro[" not in parent_name or "norbornen" not in parent_name:
        return 0
    return 1 if re.search(r",3'-norbornen", parent_name) else 0


def _skeletal_replacement_saturation_penalty(mol, match: Sequence[int], parent_name: str) -> int:
    core_atoms = set(match)
    has_unsaturation = False
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() not in core_atoms or bond.GetEndAtomIdx() not in core_atoms:
            continue
        if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1.0:
            has_unsaturation = True
            break
    if not has_unsaturation:
        return 0
    saturated_terms = (
        "decalin",
        "norbornan",
        "bicyclo",
        "tricyclo",
    )
    unsaturated_terms = (
        "naphthalen",
        "naphthalene",
        "naphthaline",
        "tetralin",
        "inden",
        "indene",
        "fluoren",
        "fluorene",
        "norbornen",
        "norbornene",
    )
    if any(term in parent_name for term in unsaturated_terms):
        return 0
    if "indan" in parent_name and _has_nonaromatic_core_imine(mol, core_atoms):
        return 2
    if any(term in parent_name for term in saturated_terms):
        return 2
    return 0


def _has_nonaromatic_core_imine(mol, core_atoms: set[int]) -> bool:
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() not in core_atoms or bond.GetEndAtomIdx() not in core_atoms:
            continue
        if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() != 2.0:
            continue
        symbols = {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}
        if symbols == {"C", "N"}:
            return True
    return False


def _locant_sort_key(locant: str) -> tuple[int, str]:
    digits = "".join(ch for ch in locant if ch.isdigit())
    return (int(digits) if digits else 999, locant)


_SKELETAL_REPLACEMENT_PREFIXES = {
    "N": "aza",
    "O": "oxa",
    "S": "thia",
}


def _has_stereo_evidence(mol, Chem) -> bool:
    cached = mol_cache_get(mol, "has_stereo_evidence")
    if cached is not None:
        return cached
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return mol_cache_set(mol, "has_stereo_evidence", True)
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return mol_cache_set(mol, "has_stereo_evidence", True)
    return mol_cache_set(mol, "has_stereo_evidence", False)


@lru_cache(maxsize=None)
def _topology_query(template_mol, Chem):
    query = Chem.RWMol()
    for atom in template_mol.GetAtoms():
        query.AddAtom(Chem.AtomFromSmarts(f"[#{atom.GetAtomicNum()}]"))
    for bond in template_mol.GetBonds():
        query.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.UNSPECIFIED)
    return query.GetMol()


@lru_cache(maxsize=None)
def _template_has_one_ring_system(template_mol) -> bool:
    rings = [set(ring) for ring in template_mol.GetRingInfo().AtomRings()]
    if len(rings) < 2:
        return False
    current = rings[0]
    remaining = rings[1:]
    changed = True
    while changed:
        changed = False
        rest = []
        for ring in remaining:
            if current & ring:
                current |= ring
                changed = True
            else:
                rest.append(ring)
        remaining = rest
    return not remaining


@lru_cache(maxsize=None)
def _template_is_fully_aromatic(template_mol) -> bool:
    return all(atom.GetIsAromatic() for atom in template_mol.GetAtoms())


def _hydro_parent_name(mol, template: dict, match: Sequence[int]) -> str:
    hydro_locs = []
    charged_locs = []
    charged_has_hydrogen = False
    for template_idx, mol_idx in enumerate(match):
        template_atom = template_record_mol(template).GetAtomWithIdx(template_idx)
        atom = mol.GetAtomWithIdx(mol_idx)
        charge_matches = template_atom.GetFormalCharge() == atom.GetFormalCharge()
        charge_is_parent_ium = (
            template_atom.GetFormalCharge() == 0
            and atom.GetFormalCharge() > 0
            and atom.GetSymbol() in {"N", "S"}
        )
        if not charge_matches and not charge_is_parent_ium:
            return ""
        locant = template_record_locants(template).get(template_idx, "")
        if charge_is_parent_ium:
            if not locant.isdigit():
                return ""
            charged_locs.append(int(locant))
            if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0:
                charged_has_hydrogen = True
        if not template_atom.GetIsAromatic() or atom.GetIsAromatic():
            continue
        if _has_unsaturated_core_bond(mol, mol_idx, match):
            continue
        if not locant.isdigit():
            return ""
        hydro_locs.append(int(locant))
    if not hydro_locs:
        return ""
    hydro_locs = sorted(set(hydro_locs))
    hydro = {
        2: "dihydro",
        4: "tetrahydro",
        6: "hexahydro",
        8: "octahydro",
        10: "decahydro",
        12: "dodecahydro",
    }.get(len(hydro_locs))
    if not hydro:
        return ""
    locants = ",".join(str(loc) for loc in hydro_locs)
    name = f"{locants}-{hydro}-{template_record_name(template)}"
    if len(charged_locs) == 1:
        if not charged_has_hydrogen:
            name = _drop_indicated_hydrogen_prefix(name)
        stem = name[:-1] if name.endswith("e") else name
        return f"{stem}-{charged_locs[0]}-ium"
    return name


def _partial_hydro_parent_names(mol, template: dict, match: Sequence[int]):
    candidate_locs = []
    for template_idx, mol_idx in enumerate(match):
        template_atom = template_record_mol(template).GetAtomWithIdx(template_idx)
        atom = mol.GetAtomWithIdx(mol_idx)
        if template_atom.GetFormalCharge() != atom.GetFormalCharge():
            return
        locant = template_record_locants(template).get(template_idx, "")
        if not locant or locant.endswith("a"):
            continue
        score = _partial_hydro_atom_score(mol, template_record_mol(template), mol_idx, template_idx, match)
        if score <= 0:
            continue
        candidate_locs.append((locant, score))
    by_locant = {}
    for locant, score in candidate_locs:
        by_locant[locant] = max(score, by_locant.get(locant, 0))
    candidate_locs = sorted(by_locant.items(), key=lambda item: _locant_sort_key(item[0]))
    if len(candidate_locs) < 2:
        return
    candidates = []
    for count, hydro in ((2, "dihydro"), (4, "tetrahydro"), (6, "hexahydro")):
        if len(candidate_locs) < count:
            continue
        combos = sorted(
            combinations(candidate_locs, count),
            key=lambda items: (
                -sum(score for _locant, score in items),
                -_hydro_locant_span([locant for locant, _score in items]),
                [_locant_sort_key(locant) for locant, _score in items],
            ),
        )
        for items in combos:
            locs = [locant for locant, _score in items]
            locant_text = ",".join(locs)
            full_even_set = count == len(candidate_locs) and count in {2, 4, 6, 8}
            candidates.append((
                (
                    0 if full_even_set else 1,
                    count,
                    -sum(score for _locant, score in items),
                    -_hydro_locant_span(locs),
                    [_locant_sort_key(locant) for locant in locs],
                ),
                f"{locant_text}-{hydro}-{template_record_name(template)}",
            ))
    for hydro_key, parent_name in sorted(candidates)[:12]:
        yield parent_name, hydro_key


def _hydro_locant_span(locants: Sequence[str]) -> int:
    values = []
    for locant in locants:
        digits = "".join(ch for ch in locant if ch.isdigit())
        if digits:
            values.append(int(digits))
    return max(values) - min(values) if values else 0


def _partial_hydro_atom_score(mol, template_mol, mol_idx: int, template_idx: int, match: Sequence[int]) -> int:
    atom = mol.GetAtomWithIdx(mol_idx)
    template_atom = template_mol.GetAtomWithIdx(template_idx)
    if atom.GetIsAromatic() or _has_unsaturated_core_bond(mol, mol_idx, match):
        return 0
    score = 0
    if atom.GetTotalNumHs() > template_atom.GetTotalNumHs():
        score += atom.GetTotalNumHs() - template_atom.GetTotalNumHs()
    if template_atom.GetIsAromatic() or _has_unsaturated_template_bond(template_mol, template_idx):
        score += 1
    return score


def _has_unsaturated_template_bond(template_mol, template_idx: int) -> bool:
    atom = template_mol.GetAtomWithIdx(template_idx)
    for nbr in atom.GetNeighbors():
        bond = template_mol.GetBondBetweenAtoms(template_idx, nbr.GetIdx())
        if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1.0:
            return True
    return False


def _has_unsaturated_core_bond(mol, mol_idx: int, match: Sequence[int]) -> bool:
    core = set(match)
    atom = mol.GetAtomWithIdx(mol_idx)
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx not in core:
            continue
        bond = mol.GetBondBetweenAtoms(mol_idx, nidx)
        if bond is not None and bond.GetBondTypeAsDouble() > 1.0:
            return True
    return False


def _hydrogenated_template_penalty(name: str) -> int:
    return 1 if "hydro" in name else 0


def _template_source_rank(template: dict) -> int:
    family = template.get("source_family", "")
    if not family:
        return 0
    if family.startswith("arylGroups") or family.startswith("simpleCyclicGroups"):
        return 1
    if family == "static-retained":
        return 2
    if family.startswith("fusionComponents"):
        return 3
    if family == "generated-fused":
        return 4
    return 5


def _fused_candidate_preference(name: str) -> tuple:
    oxo_locs = tuple(int(loc) for loc in re.findall(r"(?<!\d)(\d+)-oxo", name))
    return (
        1 if re.search(r"(?:^|-|,)\d+[a-z]-", name) else 0,
        1 if "ind-1-ene" in name else 0,
        _substituent_locant_tuple(name),
        oxo_locs or (99,),
        _benzimidazole_orientation_preference(name),
    )


def _substituent_locant_tuple(name: str) -> tuple[int, ...]:
    locants = [
        int(locant)
        for locant in re.findall(r"(?<![\d^])(\d+)(?=[a-z]?[-,])", name)
        if int(locant) < 100
    ]
    return tuple(sorted(locants)) or (99,)


def _benzimidazole_orientation_preference(name: str) -> tuple:
    if "benzimidazole" not in name or "oxo" in name:
        return ()
    locants = sorted(
        {int(loc) for loc in re.findall(r"(?<!\d)(\d+)-", name) if int(loc) >= 4},
        reverse=True,
    )
    return tuple(-locant for locant in locants)


def _has_n_oxide_on_match(mol, match: Sequence[int]) -> bool:
    core = set(match)
    for mol_idx in match:
        atom = mol.GetAtomWithIdx(mol_idx)
        if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
            continue
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core:
                continue
            bond = mol.GetBondBetweenAtoms(mol_idx, nidx)
            if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() < 0 and bond.GetBondTypeAsDouble() == 1.0:
                return True
    return False


def _name_fused_core_match(
    mol, template: dict, match: Sequence[int], parent_name: str | None = None
) -> str:
    core_set = set(match)
    if ring_system_extends_core(mol, core_set):
        return ""
    reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
    parent_name = parent_name or _adjust_fused_parent_for_charge(mol, template, reverse)
    prefixes = []
    for mol_idx in match:
        template_idx = reverse[mol_idx]
        locant = template_record_locants(template).get(template_idx)
        for nbr in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core_set:
                continue
            if not locant:
                return ""
            prefix = branch_prefix(mol, nidx, mol_idx)
            if not prefix:
                return ""
            prefixes.append((locant, prefix))
    if not prefixes:
        return parent_name
    parent_name = _drop_indicated_hydrogen_segments(parent_name, prefixes)
    suffix_name = _fused_oxo_suffix_name(prefixes, parent_name)
    if suffix_name:
        return suffix_name
    prefix_text = _format_prefixes(prefixes)
    if not prefix_text:
        return ""
    return prefix_text + "-" + parent_name


def _fused_oxo_suffix_name(prefixes: Sequence[tuple[str, str]], parent_name: str) -> str:
    oxo_locs = sorted((loc for loc, prefix in prefixes if prefix == "oxo" and loc.isdigit()), key=int)
    if parent_name == "pteridine" and oxo_locs == ["4", "7"]:
        remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "oxo"]
        prefix_text = _format_prefixes(remaining)
        name = "3H,8H-pteridine-4,7-dione"
        return prefix_text + "-" + name if prefix_text else name
    return ""


def _adjust_fused_parent_for_charge(mol, template: dict, reverse: dict[int, int]) -> str:
    name = _adjust_fused_parent_for_indicated_hydrogen(mol, template, reverse)
    if name in {"quinuclidine", "quinuclidin"}:
        for mol_idx, template_idx in reverse.items():
            locant = template_record_locants(template).get(template_idx)
            atom = mol.GetAtomWithIdx(mol_idx)
            if locant == "1" and atom.GetSymbol() == "N" and atom.GetFormalCharge() > 0:
                return "quinuclidin-1-ium"
    charged_locs = []
    charged_has_hydrogen = False
    for mol_idx, template_idx in reverse.items():
        locant = template_record_locants(template).get(template_idx)
        atom = mol.GetAtomWithIdx(mol_idx)
        if locant and locant.isdigit() and atom.GetSymbol() in {"N", "S"} and atom.GetFormalCharge() > 0:
            charged_locs.append(int(locant))
            if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0:
                charged_has_hydrogen = True
    if len(charged_locs) == 1:
        locant = charged_locs[0]
        if not charged_has_hydrogen:
            name = _drop_indicated_hydrogen_prefix(name)
        stem = name[:-1] if name.endswith("e") else name
        return f"{stem}-{locant}-ium"
    return name


def _drop_indicated_hydrogen_prefix(name: str) -> str:
    if len(name) > 3 and name[0].isdigit() and name[1:3] == "H-":
        return name[3:]
    return name


def _drop_indicated_hydrogen_segments(name: str, prefixes: Sequence[tuple[str, str]]) -> str:
    if "benzimidazole" in name and any(prefix == "oxo" for _, prefix in prefixes):
        return name

    def replacement(match: re.Match) -> str:
        locant = match.group(2)
        if locant == "2" and "indazole" in name:
            return match.group(0)
        return match.group(1)

    return re.sub(r"(^|-)(\d+)H-", replacement, name)


def _adjust_fused_parent_for_indicated_hydrogen(mol, template: dict, reverse: dict[int, int]) -> str:
    name = template_record_name(template)
    if not name.startswith("1H-"):
        return name
    h_locs = []
    for mol_idx, template_idx in reverse.items():
        atom = mol.GetAtomWithIdx(mol_idx)
        locant = template_record_locants(template).get(template_idx)
        if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0 and locant and locant.isdigit():
            h_locs.append(locant)
    if len(h_locs) != 1 or h_locs[0] == "1":
        return name
    return f"{h_locs[0]}H-{name[3:]}"


def _named_ring_instance_names(mol, ring: Sequence[int], Chem) -> list[tuple[str, str]]:
    ordered = ordered_ring(mol, ring)
    if not ordered:
        return []
    candidates = _ring_parent_candidates(mol, ordered)
    if not candidates:
        candidates = _resource_ring_parent_candidates(mol, ordered, Chem)
    if not candidates:
        return []
    ring_set = set(ordered)
    raw_subs = []
    for ring_idx in ordered:
        atom = mol.GetAtomWithIdx(ring_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set:
                continue
            prefix = branch_prefix(mol, nidx, ring_idx)
            if not prefix:
                return []
            raw_subs.append((ring_idx, prefix))

    best_key = None
    best_name = ""
    best_seq: Sequence[int] = ()
    for parent_name, seq in candidates:
        adjusted_parent = _adjust_ring_parent_for_charge(mol, parent_name, seq, raw_subs)
        if not adjusted_parent:
            continue
        if not raw_subs:
            name = adjusted_parent
            key = ([], name)
            if best_key is None or key < best_key:
                best_key = key
                best_name = name
                best_seq = seq
            continue
        pos = {idx: i + 1 for i, idx in enumerate(seq)}
        prefixes = [
            (str(pos[idx]), prefix)
            for idx, prefix in raw_subs
            if not (
                adjusted_parent in {"1,3,4-dioxazol-2-one", "azet-2-one"} and pos[idx] == 2 and prefix == "oxo"
            )
        ]
        locants = sorted(pos[idx] for idx, _ in raw_subs)
        prefix_text = _format_prefixes(prefixes)
        if not prefix_text:
            continue
        name = prefix_text + "-" + adjusted_parent
        key = (locants, 1, name)
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
            best_seq = seq
        suffix_name = (
            ""
            if adjusted_parent in {"1,3,4-dioxazol-2-one", "azet-2-one"}
            else _single_ring_oxo_suffix_name(prefixes, adjusted_parent)
        )
        if suffix_name:
            suffix_key = (locants, 0, suffix_name)
            if best_key is None or suffix_key < best_key:
                best_key = suffix_key
                best_name = suffix_name
                best_seq = seq
    if not best_name:
        return []
    return [(best_name, _single_ring_stereo_prefix(mol, best_seq, Chem))]


def _single_ring_stereo_prefix(mol, seq: Sequence[int], Chem) -> str:
    if not seq:
        return ""
    if all(mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 for idx in seq):
        return ""
    seq_set = set(seq)
    tagged_atoms = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    if len(tagged_atoms) < 2:
        return ""
    if any(idx not in seq_set for idx in tagged_atoms):
        return ""
    if any(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() != Chem.BondStereo.STEREONONE
        for bond in mol.GetBonds()
    ):
        return ""
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    centers = [
        (idx, descriptor)
        for idx, descriptor in Chem.FindMolChiralCenters(
            mol,
            includeUnassigned=False,
            useLegacyImplementation=False,
        )
        if descriptor in {"R", "S"}
    ]
    if len(centers) < 2:
        return ""
    locants = {idx: loc for loc, idx in enumerate(seq, start=1)}
    if any(idx not in locants for idx, _descriptor in centers):
        return ""
    return ",".join(
        f"{locants[idx]}{descriptor}"
        for idx, descriptor in sorted(centers, key=lambda item: locants[item[0]])
    )


def _polycyclic_stereo_prefix(mol, seq: Sequence[int], Chem) -> str:
    if not seq:
        return ""
    seq_set = set(seq)
    tagged_atoms = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    if len(tagged_atoms) < 2:
        return ""
    if any(idx not in seq_set for idx in tagged_atoms):
        return ""
    if any(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() != Chem.BondStereo.STEREONONE
        for bond in mol.GetBonds()
    ):
        return ""
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    centers = [
        (idx, descriptor)
        for idx, descriptor in Chem.FindMolChiralCenters(
            mol,
            includeUnassigned=False,
            useLegacyImplementation=False,
        )
        if descriptor in {"R", "S"}
    ]
    if len(centers) < 2:
        return ""
    locants = {idx: loc for loc, idx in enumerate(seq, start=1)}
    if any(idx not in locants for idx, _descriptor in centers):
        return ""
    return ",".join(
        f"{locants[idx]}{descriptor}"
        for idx, descriptor in sorted(centers, key=lambda item: locants[item[0]])
    )


def _single_ring_oxo_suffix_name(prefixes: Sequence[tuple[str, str]], parent_name: str) -> str:
    oxo_locs = [loc for loc, prefix in prefixes if prefix == "oxo" and loc.isdigit()]
    if not oxo_locs:
        return ""
    remaining = [(loc, prefix) for loc, prefix in prefixes if prefix != "oxo"]
    prefix_text = _format_prefixes(remaining)
    parent = _oxo_adjusted_unsaturated_carbocycle_parent(parent_name, oxo_locs)
    if not parent:
        parent = parent_name[:-1] if parent_name.endswith("e") else parent_name
    locants = ",".join(sorted(oxo_locs, key=int))
    suffix = "one" if len(oxo_locs) == 1 else f"{MULT.get(len(oxo_locs), '')}one"
    name = f"{parent}-{locants}-{suffix}"
    return prefix_text + "-" + name if prefix_text else name


def _oxo_adjusted_unsaturated_carbocycle_parent(parent_name: str, oxo_locs: Sequence[str]) -> str:
    ring_size = _cycloalkapolyene_ring_size(parent_name)
    if ring_size is None:
        return ""
    oxo_atoms = {int(loc) for loc in oxo_locs}
    ene_locs = _max_nonoxo_ring_double_bond_locs(ring_size, oxo_atoms)
    root = "cyclo" + ROOTS[ring_size]
    if not ene_locs:
        return root + "an"
    parent = loc_unsat(root, ene_locs, "ene")
    return parent[:-1] if parent.endswith("e") else parent


def _cycloalkapolyene_ring_size(parent_name: str) -> int | None:
    if not parent_name.startswith("cyclo") or "en" not in parent_name:
        return None
    if not re.search(r"^cyclo(?:but|hept)a-1(?:,3)*(?:,5)?-", parent_name):
        return None
    for size, root in sorted(ROOTS.items(), key=lambda item: -len(item[1])):
        if size not in {4, 7}:
            continue
        if parent_name.startswith("cyclo" + root):
            return size
    return None


def _max_nonoxo_ring_double_bond_locs(ring_size: int, oxo_atoms: set[int]) -> list[int]:
    edges = []
    for loc in range(1, ring_size + 1):
        a = loc
        b = loc + 1 if loc < ring_size else 1
        if a in oxo_atoms or b in oxo_atoms:
            continue
        edges.append((loc, a, b))
    best: tuple[int, tuple[int, ...]] | None = None

    def search(index: int, used_atoms: set[int], locs: list[int]) -> None:
        nonlocal best
        if index >= len(edges):
            key = (-len(locs), tuple(locs))
            if best is None or key < best:
                best = key
            return
        loc, a, b = edges[index]
        search(index + 1, used_atoms, locs)
        if a not in used_atoms and b not in used_atoms:
            locs.append(loc)
            search(index + 1, used_atoms | {a, b}, locs)
            locs.pop()

    search(0, set(), [])
    return list(best[1]) if best is not None else []


def _sulfanium_name(names: Sequence[str]) -> str:
    counts = Counter(names)
    parts = []
    for name in sorted(counts):
        count = counts[name]
        term = (MULT.get(count, "") if count > 1 else "") + name
        if _needs_substituent_parentheses(term):
            term = f"({term})"
        parts.append(term)
    return "".join(parts) + "sulfanium"


def _needs_substituent_parentheses(name: str) -> bool:
    return name[:1].isdigit() or "-" in name or "," in name or name.startswith("(")


def _n_halo_atom(mol, nitrogen_idx: int, carbon_idx: int) -> int:
    halo_atoms = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors()
        if nbr.GetIdx() != carbon_idx and nbr.GetSymbol() in HALO
    ]
    return halo_atoms[0] if len(halo_atoms) == 1 else -1


def _n_halo_prefix(mol, nitrogen_idx: int, carbon_idx: int, Chem) -> str:
    atom = mol.GetAtomWithIdx(nitrogen_idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    halo_idx = _n_halo_atom(mol, nitrogen_idx, carbon_idx)
    if halo_idx < 0:
        return ""
    bond = mol.GetBondBetweenAtoms(nitrogen_idx, halo_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    return "N"


def _append_terminal_halo_prefixes(mol, atom_idx: int, parent_idx: int, locant: str, prefixes: list[tuple[str, str]], Chem) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return False
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
        if nbr.GetSymbol() not in HALO or bond.GetBondType() != Chem.BondType.SINGLE:
            return False
        prefixes.append((locant, HALO[nbr.GetSymbol()]))
    return True


def _prefer_acyclic_skeleton_name(name: str) -> bool:
    if (
        name.endswith("-methane")
        and "methylidene" in name
        and any(term in name for term in ("spiro[", "tricyclo[", "bicyclo["))
    ):
        return False
    if name.endswith("-methane") and "hydroxymethylidene" in name and "cyclo" in name:
        return False
    if (
        name.endswith("-methane")
        and "pyrazol-5-yl" in name
        and "azaethyl" in name
        and "oxo" in name
    ):
        return False
    if "phenylidene" in name:
        return False
    if "ylidene" in name:
        return True
    if "imino" in name and any(term in name for term in ("ene", "oxo", "carbonyl")):
        return True
    if "imino" in name and "nitro" in name and "azamethyl" in name:
        return True
    if "imino" in name and "cyano" in name:
        return True
    if "diaza" in name and "ene" in name and any(term in name for term in ("carbonyl", "sulfonyl", "nitro")):
        return True
    if "hydroxy" in name and "dioxo" in name and any(term in name for term in ("prop-2-ene", "but-")):
        return True
    if (
        "tetrafluoro" in name
        and "thia" in name
        and "aza" in name
        and "oxo" in name
        and "ene" in name
    ):
        return True
    if "oxopyrazin" in name and "azapropane" in name:
        return True
    if "oxo" in name and "oxaoct-3-ene" in name:
        return True
    if name.count("cyano") >= 2:
        return True
    if any(term in name for term in ("diene", "triene", "but-3-ene")):
        return True
    if "eth-1-ene" in name and any(term in name for term in ("nitro", "difluoro", "cyano")):
        return True
    functional_terms = _functional_term_score(name)
    has_chain_unsaturation = bool(
        re.search(r"(?:^|-)\d+(?:,\d+)*-(?:en|yn)|[a-z]-\d+(?:,\d+)*-(?:en|yn)|(?:en|yn)e(?:$|-)", name)
    )
    return functional_terms >= 3 and has_chain_unsaturation


def _functional_term_score(name: str) -> int:
    score = sum(name.count(term) for term in ("imino", "sulfonyl", "carboxy", "carbonyl"))
    multiplier_values = {
        "di": 2,
        "tri": 3,
        "tetra": 4,
        "penta": 5,
        "hexa": 6,
    }
    for match in re.finditer(r"(?:(di|tri|tetra|penta|hexa))?oxo", name):
        score += multiplier_values.get(match.group(1) or "", 1)
    return score
