"""Ordered principal-function selection, separate from final name emission.

Graph eligibility precedes candidate admission; parent coverage precedes suffix
numbering; existing rankers break remaining ties. All supported principal routes
live here, so new rules cannot silently acquire precedence through dispatcher
placement. Specialized compatibility and unsupported chemistry keep the legacy
fallback after this pipeline has exhausted its eligible stages.

Producers retain their bounded chemical and writing scopes. A preferred method
or parent that cannot be rendered stays unresolved; that does not invalidate
an independently rendered general name. Concrete construction failures still
exclude the affected candidates. This module does not claim complete coverage.
"""

from collections import Counter
from dataclasses import dataclass, replace
import re

from .molecule_iteration import iter_atoms, iter_bonds
from .molecule_cache import mol_cache_get, mol_cache_set
from .graph_model import graph_view, method_bond_counts
from .principal_carboxyl import principal_carboxyl_group, acyclic_acid_record
from .principal_carbonyl import (
    principal_carbonyl_sites,
    principal_ring_carbonyl_key,
    supported_ring_carbonyl_selection,
)
from .principal_functional import principal_organic_function
from .principal_groups import principal_ring_groups, sole_sulfonamide
from .principal_oxygen import principal_oxygen_record
from .retained_composition import sole_characteristic_amide
from .naming_result import SelectionLimitation


# Order is an explicit compatibility contract for the currently supported graph
# scopes. Composition and sulfur rules retain their conservative eligibility;
# these are not a universal seniority table for unsupported functional classes.
# P-65 supplies class-specific functional-class proofs before the generic
# P-51 search. A decline leaves multiplication/replacement eligibility intact.
PRINCIPAL_SELECTION_STAGES = (
    ("native-ocene-parent-method", "_native_ocene_parent_method"),
    ("preferred-functional-class-method", "_preferred_functional_class_method"),
    ("full-bluebook-frontier", "_full_bluebook_frontier"),
    ("source-general-pin-method", "_source_general_pin_method"),
    ("source-hydride-assembly-method", "_source_hydride_assembly_method"),
    ("source-fusion-functional-method", "_source_fusion_functional_method"),
    ("source-azoxy-spiro-method", "_source_azoxy_spiro_method"),
    ("source-functional-family-method", "_source_functional_family_method"),
    ("carbon-radical-method", "_carbon_radical_method"),
    ("amine-cation-component", "_amine_cation_component"),
    ("simple-amine-oxide-component", "_simple_amine_oxide_component"),
    ("carbon-cation-component", "_carbon_cation_component"),
    ("chalcogen-cation-component", "_chalcogen_cation_component"),
    ("carbon-anion-component", "_carbon_anion_component"),
    ("nitrogen-anion-component", "_nitrogen_anion_component"),
    ("retained-hydroxy-anion-component", "_retained_hydroxy_anion_component"),
    ("polyol-anion-component", "_polyol_anion_component"),
    ("amine-imine-oxide-method", "_amine_imine_oxide_method"),
    ("source-hetero-parent-method", "_source_hetero_parent_method"),
    ("source-ester-method", "_source_ester_method"),
    ("retained-ether-method", "_retained_ether_method"),
    ("principal-group-priority", "_principal_group_priority"),
    ("sulfinamide-method", "_sulfinamide_method"),
    ("carbon-acid-derivative-method", "_carbon_acid_derivative_method"),
    ("hidden-amide-pseudoketone-method", "_hidden_amide_pseudoketone_method"),
    ("chalcogen-chain-pseudoketone-method", "_chalcogen_chain_pseudoketone_method"),
    ("cyclic-sultam-heterone-method", "_cyclic_sultam_heterone_method"),
    ("functional-ring-assembly-method", "_functional_ring_assembly_method"),
    ("acid-oxygen-isotope-method", "_acid_oxygen_isotope_method"),
    ("isotope-characteristic-group-method", "_isotope_characteristic_group_method"),
    ("cyclic-dianhydride-method", "_cyclic_dianhydride_method"),
    ("sulfur-ester-method", "_sulfur_ester_method"),
    ("partial-polyol-ester-method", "_partial_polyol_ester_method"),
    ("acid-derivative-method", "_acid_derivative_method"),
    ("isocyanato-hydride-method", "_isocyanato_hydride_method"),
    ("fullerene-parent-method", "_fullerene_parent_method"),
    ("retained-hydrocarbon-parent-method", "_retained_hydrocarbon_parent_method"),
    ("retained-hydrocarbon-derivative-method", "_retained_hydrocarbon_derivative_method"),
    ("linear-phane-parent-method", "_linear_phane_parent_method"),
    ("ring-assembly-method", "_ring_assembly_method"),
    ("polyspiro-parent-method", "_polyspiro_parent_method"),
    ("dehydro-parent-method", "_dehydro_parent_method"),
    ("cumulene-parent-method", "_cumulene_parent_method"),
    ("symmetric-sulfone-ether-method", "_symmetric_sulfone_ether_method"),
    ("naming-method", "_naming_method"),
    ("mapped-diaryl-ketone-general", "_diaryl_ketone_general"),
    ("mapped-fused-acid", "_fused_acid"),
    ("mapped-amido-acid-general", "_amido_acid_general"),
    ("mapped-fused-imine-general", "_fused_imine_general"),
    ("cross-route-parent-suffix", "_cross_route"),
    ("mapped-monoacid-general", "_monoacid_chain_general"),
    ("mapped-ring-acid-general", "_ring_acid_general"),
    ("mapped-monoester-general", "_monoester_general"),
    ("mapped-n-sulfonyl-ring-amide", "_n_sulfonyl_ring_amide"),
    ("mapped-ring-chain-amide-frontier", "_ring_chain_amide"),
    ("mapped-linked-ring-amides", "_linked_ring_amides"),
    ("mapped-retained-fused-lactam", "_retained_fused_lactam"),
    ("mapped-ring-state", "_ring_state"),
    ("extended-functional-parent", "_extended_functional"),
    ("mapped-acyclic-amide-general", "_acyclic_amide_general"),
    ("ring-construction", "_ring_construction"),
    ("carboxylic-acid", "_acid"),
    ("mapped-functional-parent", "_mapped_functional"),
    ("multifunctional-parent", "_multifunctional"),
    ("retained-composition", "_composition"),
    ("sulfonamide", "_sulfonamide"),
    ("sulfuric-diamide", "_diamide"),
    ("organic-functional-class", "_functional"),
    ("mapped-principal-general", "_principal_general"),
    ("fused-hydrogen-state", "_fused_hydrogen_state"),
    ("ring-carbonyl", "_carbonyl"),
    ("ring-functional-parent", "_ring"),
    ("acyclic-oxygen", "_oxygen"),
    ("characteristic-amide", "_amide"),
    ("acyclic-carbon-parent-limit", "_acyclic_carbon_parent_limit"),
)

_ION_COMPONENT_STAGES = frozenset({
    'preferred-functional-class-method',
    'full-bluebook-frontier',
    'source-general-pin-method',
    'source-hydride-assembly-method',
    'source-fusion-functional-method',
    'source-azoxy-spiro-method',
    'carbon-cation-component', 'carbon-anion-component',
    'nitrogen-anion-component', 'chalcogen-cation-component',
    'retained-hydroxy-anion-component', 'amine-cation-component',
    'simple-amine-oxide-component',
})

_RING_FAMILIES = frozenset(
    {
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "single-ring-parent",
        "hydrogenated-fused-parent",
    }
)
_POLYCYCLIC_FAMILIES = frozenset(
    {
        "kekulized-generic-polycyclic-parent",
        "generic-bicyclic-parent",
        "generic-spiro-parent",
    }
)
_RULE_GROUPS = {
    "composition": {"retained-component-composition"},
    "sulfonamide": {"sulfonamide-retained-parent"},
    "ring": _RING_FAMILIES,
    "polycyclic-alcohol": _POLYCYCLIC_FAMILIES,
    "carbonyl": _RING_FAMILIES
    | _POLYCYCLIC_FAMILIES
    | {"natural-product-topology-parent"},
    "amide": {"retained-amide-parent"},
    "acid": _RING_FAMILIES | {"hydro-fused-retained-parent"},
}
_FUNCTIONAL_FAMILIES = frozenset(
    {
        "azanium-parent",
        "phosphorus-acid-parent",
        "charged-organophosphorus-oxo-parent",
        "phosphate-ester-retained-parent",
        "carbamate-retained-parent",
        "acyl-hydrazide-retained-parent",
        "terminal-carboxylate-parent",
        "retained-amide-parent",
        "retained-acid-ester-parent",
    }
)


@dataclass(frozen=True, slots=True)
class PrincipalRequirement:
    """Graph obligation that admitted the selected parent/suffix route."""

    kind: str
    atoms: tuple[int, ...] = ()
    suffix: str = ""


@dataclass(frozen=True, slots=True)
class PrincipalDecision:
    """One validated selection; the engine commits telemetry/output once."""

    stage: str
    requirement: PrincipalRequirement
    prepared: object = None
    output_name: str = ""
    direct_route: str = ""
    parent_selection: object = None
    atom_locants: tuple = ()
    comparison: object = None
    source_class_constraint: object = None
    deferred_parent_resolution: object = None
    # A preference comparison or writer limit is not a graph-invalidity proof.
    # Empty decisions opt in explicitly; chemistry-specific barriers remain
    # hard stops unless their producer distinguishes a usable alternative.
    preference_fallback: bool = False
    preference_issues: tuple[SelectionLimitation, ...] = ()
    justification: object = None


@dataclass(frozen=True, slots=True)
class DeferredParentResolution:
    """Evidence that a broader pool retained a deferred method's parents."""

    status: str
    witnesses: int
    resolved: int
    reason: str
    rule: str = 'P-44/P-45/P-51.1'
    evidence: tuple = ()


@dataclass(frozen=True, slots=True)
class PrincipalSiteCopyBound:
    """Necessary resource bound on repeats of one P-44-selected parent frame.

    This is not an eligibility proof for multiplicative nomenclature. A bound
    of two or more keeps P-45 multiplication live; a bound below two rules out
    two disjoint copies of this exact fully functionalized frame.
    """

    parent_atoms: tuple[int, ...]
    site_count: int
    parent_carbon_count: int
    total_source_sites: int
    total_carbon_atoms: int
    maximum_disjoint_copies: int
    rules: tuple[str, ...] = ('P-44.1.1', 'P-44.3', 'P-45.0', 'P-15.3.0')


@dataclass(frozen=True, slots=True)
class SourceClassConstraintBundle:
    """Retain independent class obligations when more than one is present."""

    obligations: tuple
    status: str


class PrincipalSelectionPipeline:
    """Compile rule pools once, then select lazily for each molecule."""

    def __init__(self, rules):
        self.groups = {
            key: tuple(r for r in rules if r.family in families)
            for key, families in _RULE_GROUPS.items()
        }
        self.groups.update(
            {
                family: tuple(r for r in rules if r.family == family)
                for family in _FUNCTIONAL_FAMILIES
            }
        )
        self.stages = tuple(
            (label, getattr(self, method))
            for label, method in PRINCIPAL_SELECTION_STAGES
        )

    def select(self, backend):
        # P-67.1.2.6.3 recognizes a nitric/nitrous amide class even when its
        # complete N-substitution grammar is outside the bounded writer.
        # Keep that class obligation while allowing a proved senior class
        # to reach its later mapped route; generic amine/aza fallbacks may
        # not discharge the obligation merely by returning some name.
        class_constraints = []
        mol = backend.mol
        # Initialized graphs can share the primitive facts already consumed
        # by the acid-derivative stage. Preserve native operation order for
        # unsanitized inputs: graph_view can otherwise initialize ring state
        # before a legacy producer would have rejected the graph.
        initialized = not mol.NeedsUpdatePropertyCache()
        if initialized:
            try:
                mol.GetRingInfo().NumRings()
            except RuntimeError:
                initialized = False
        bonds = method_bond_counts(mol, backend.Chem) if initialized else None
        if (any(a == c == 7 for a, c, _order in bonds) if bonds is not None else
                any(bond.GetBeginAtom().GetAtomicNum() == 7
                    and bond.GetEndAtom().GetAtomicNum() == 7
                    for bond in iter_bonds(mol))):
            from .nitrous_amide_methods import nitrous_amide_disposition
            disposition = nitrous_amide_disposition(mol, backend.Chem)
            if disposition.status in {'pending', 'higher-class-route'}:
                class_constraints.append(('nitrous-amide', disposition))
        if (any(a == 7 and c in {9, 17, 35, 53} for a, c, _order in bonds)
                if bonds is not None else
                any({bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()}
                    in ({7, 9}, {7, 17}, {7, 35}, {7, 53}) for bond in iter_bonds(mol))):
            from .halogen_acid_amide_methods import classify_halogen_acid_amide
            disposition = classify_halogen_acid_amide(mol, backend.Chem)
            if disposition.status in {'pending', 'preselected-parent', 'higher-class-route', 'cyclic-imide-route'}:
                class_constraints.append(('halogen-acid-amide', disposition))
        sulfinamide = self._sulfinamide_class(backend)
        if sulfinamide is not None:
            class_constraints.append(('sulfinamide', sulfinamide))
        backend.ctx.compatibility_eager = bool(class_constraints)
        required_centers = []
        for kind, constraint in class_constraints:
            if kind == 'nitrous-amide':
                higher = constraint.higher_class_centers
                senior_rank = min((rank for _kind, _center, rank in higher), default=None)
                centers = frozenset(center for _kind, center, rank in higher
                                    if rank == senior_rank)
            elif kind == 'sulfinamide':
                centers = (constraint.higher_centers
                           if constraint.status == 'higher-class-route'
                           else frozenset(site.center for site in constraint.sites))
            else:
                centers = (frozenset(constraint.centers)
                           if constraint.status in {'higher-class-route', 'cyclic-imide-route'}
                           else frozenset())
            required_centers.append((kind, constraint, centers))
        deferred_method = None
        unresolved = []
        last_pending = None
        for stage, select in self.stages:
            if backend.ctx.telemetry is not None:
                backend.ctx.telemetry.principal_stage_attempts[stage] += 1
            for kind, constraint, _centers in required_centers:
                if kind == 'nitrous-amide' and stage == 'acid-derivative-method' and constraint.status == 'pending':
                    return self._with_class_constraints(self._nitrous_amide_barrier(constraint), class_constraints)
                if kind == 'halogen-acid-amide' and stage == 'naming-method' and constraint.status in {'pending', 'preselected-parent'}:
                    return self._with_class_constraints(self._halogen_amide_barrier(constraint), class_constraints)
            decision = select(backend, stage)
            if decision is not None:
                if (stage == 'naming-method'
                        and (getattr(decision.comparison, 'deferred', False)
                             or decision.preference_fallback)):
                    # An unresolved substitutive parent pool may be completed
                    # by the later shared mapped pool. Retain its obligations.
                    deferred_method = decision
                    backend.ctx.unresolved_preference = True
                    continue
                if (decision.preference_fallback and not decision.output_name
                        and decision.prepared is None):
                    last_pending = decision
                    unresolved.extend(self._preference_issues(decision))
                    backend.ctx.unresolved_preference = True
                    continue
                allowed = True
                for kind, constraint, senior_centers in required_centers:
                    if kind == 'halogen-acid-amide' and constraint.status == 'cyclic-imide-route':
                        # A local imide core does not establish a complete
                        # name. Require the existing mapped writer to retain
                        # the N-X branch and both carbonyl sites; a legacy
                        # halo/amine fallback cannot close this handoff.
                        from .component_mapping import fully_covered
                        record = getattr(decision.comparison, 'selected', None)
                        component = getattr(record, 'construction', None)
                        if (stage != 'mapped-ring-state' or component is None
                                or constraint.nitrogen not in component.atoms
                                or not dict(component.atom_locants).get(constraint.nitrogen)
                                or constraint.halogen not in component.embedded_atoms
                                or component.atoms & component.embedded_atoms
                                or not fully_covered(mol, component.atoms, component.embedded_atoms)):
                            allowed = False
                            break
                    # Preserve the established earlier ion/radical/component
                    # routes for a deferred N-X writer. Its ordinary amine
                    # fallback is blocked at the naming-method boundary.
                    if kind == 'halogen-acid-amide' and constraint.status in {'pending', 'preselected-parent'}:
                        continue
                    if (decision.output_name or decision.prepared is not None):
                        # The projected group inventory identifies possible
                        # senior centers only. Require a route on the original
                        # graph that actually selects every senior center.
                        if not senior_centers <= set(decision.requirement.atoms) or not senior_centers:
                            allowed = False
                            break
                    elif (constraint.status == 'higher-class-route'
                          and not senior_centers <= set(decision.requirement.atoms)):
                        # A pending lower-class method is no more decisive
                        # than a rendered lower-class name. The original
                        # graph's later senior route must still get its try.
                        allowed = False
                        break
                if allowed:
                    for kind, constraint, _centers in required_centers:
                        if (kind == 'halogen-acid-amide'
                                and constraint.status == 'higher-class-route'
                                and not self._mapped_halogen_amide_suffix(backend, decision, constraint)):
                            # P-41/P-42 proves the senior class, not the
                            # preferred N-X prefix/method in a mixed graph.
                            # Only an actual mapped N-halo amide suffix join
                            # closes that local obligation. A valid general
                            # branch construction remains available below.
                            unresolved.append(SelectionLimitation(
                                stage, constraint.rule,
                                'Senior class selection is proved, but the preferred subordinate N-halogen construction has no complete mapped suffix-role proof.'))
                        if (kind == 'halogen-acid-amide'
                                and constraint.status == 'cyclic-imide-route'
                                and constraint.confidence == 'inference'):
                            unresolved.append(SelectionLimitation(
                                stage, constraint.rule,
                                'Mapped cyclic-imide name is available; preferred-method extension from the published N-bromo precedent to this halogen remains unresolved.'))
                    if deferred_method is not None:
                        resolution = self._resolve_deferred_parent(
                            backend, deferred_method.comparison, decision)
                        if ((decision.output_name or decision.prepared is not None)
                                and resolution.status != 'complete'):
                            # The later writer can be usable without closing
                            # the preferred-parent comparison. Preserve both.
                            unresolved.extend(self._preference_issues(deferred_method))
                        decision = replace(decision, deferred_parent_resolution=resolution)
                    if unresolved:
                        decision = replace(decision, preference_issues=tuple(dict.fromkeys(
                            (*unresolved, *decision.preference_issues))))
                    if decision.justification is None and (decision.output_name or decision.prepared is not None):
                        decision = self._join_preferred_evidence(backend, decision)
                    if (backend.ctx.telemetry is not None
                            and stage == 'cross-route-parent-suffix'
                            and getattr(decision.comparison, 'status', '') == 'pending'
                            and decision.prepared is None and decision.output_name):
                        # A local ring-state construction may be delivered
                        # while the whole-parent comparison remains pending.
                        # Record it only after class constraints accept the
                        # decision and only if this exact proposal is the
                        # output; do not count it as a globally selected PIN.
                        proposal = getattr(decision.comparison, 'general_proposal', None)
                        record = getattr(proposal, 'ring_state', None)
                        if (record is not None and proposal.name == decision.output_name
                                and record.name == decision.output_name):
                            telemetry = backend.ctx.telemetry
                            telemetry.ring_state_methods[record.method] += 1
                            from .ring_state_parent import _CRITERIA
                            from .heteropolycyclic_parents import RULES
                            # These are local ring-numbering/H-state criteria,
                            # not the unresolved P-44/P-45 parent comparison.
                            telemetry.ring_state_rules.update(RULES.get(record.method, ()) + _CRITERIA[1:])
                            telemetry.ring_state_hydrogenation[str(len(record.hydro_locants))] += 1
                            telemetry.ring_state_accommodation[
                                f'paired:{len(record.paired_suffix_locants)}/added:{len(record.added_h_locants)}'] += 1
                    return self._with_class_constraints(decision, class_constraints)
        if deferred_method is not None:
            last_pending = deferred_method
            unresolved.extend(self._preference_issues(deferred_method))
        if class_constraints:
            kind, constraint = class_constraints[0]
            barrier = (self._nitrous_amide_barrier(constraint) if kind == 'nitrous-amide'
                       else self._sulfinamide_barrier(constraint) if kind == 'sulfinamide'
                       else self._halogen_amide_barrier(constraint))
            if self._ordinary_halogen_higher_class_continuation(backend, class_constraints):
                # P-41 class recognition is useful even when the preferred
                # parent/suffix writer cannot finish. It does not disprove a
                # whole-graph general construction. Keep every deferred
                # comparison open while allowing the existing acyclic writer
                # to supply that construction on the original molecule.
                unresolved.append(SelectionLimitation(
                    barrier.stage, constraint.rule,
                    'A senior characteristic group is recognized, but its preferred method/parent/suffix construction is incomplete; a general name does not resolve that selection.'))
                barrier = replace(barrier, preference_fallback=True)
            if unresolved:
                barrier = replace(barrier, preference_issues=tuple(dict.fromkeys(
                    (*unresolved, *barrier.preference_issues))))
            return self._with_class_constraints(barrier, class_constraints)
        if last_pending is not None:
            return replace(last_pending, preference_fallback=True,
                           preference_issues=tuple(dict.fromkeys(unresolved)))
        return None

    @staticmethod
    def _mapped_halogen_amide_suffix(b, decision, constraint):
        """P-66.1.1.3.1/.3.5 proof for a directly N-halogenated amide.

        Class evidence alone cannot establish a prefix construction. Retain
        resolved direct amide substitution only when the selected original
        N and X have exactly the mapped N-locant/prefix roles being emitted.
        """
        from .component_mapping import MappedComponent, fully_covered, mapped_branch
        from .extended_inventory import selection_inventory

        inventory = selection_inventory(b.mol)
        nitrogen, halogen = constraint.nitrogen, constraint.halogen
        if not any(group.kind == 'amide' and group.ligand == nitrogen
                   and group.center in constraint.centers for group in inventory.principal):
            return False
        comparison = decision.comparison
        selected = getattr(comparison, 'selected', None)
        if selected is None:
            candidates = getattr(comparison, 'candidates', ())
            selected = candidates[0] if candidates else comparison
        component = (selected if isinstance(selected, MappedComponent)
                     else getattr(selected, 'construction', None))
        if component is None or component.atoms & component.embedded_atoms:
            return False
        locant = dict(component.atom_locants).get(nitrogen, '')
        if (not locant.startswith('N') or halogen not in component.embedded_atoms
                or not fully_covered(b.mol, component.atoms, component.embedded_atoms)):
            return False
        text, atoms = mapped_branch(b.mol, b.Chem, halogen, nitrogen)
        return atoms == frozenset({halogen}) and (locant, text) in component.prefixes

    @staticmethod
    def _ordinary_halogen_higher_class_continuation(b, constraints):
        """Separate a known senior class's writer limit from invalid chemistry.

        This continuation is for one ordinary neutral acyclic N-X unit only.
        Pending recognition, other class constraints and cyclic-imide handoffs
        retain their own admission rules. The general writer still has to
        render the complete original graph and preserve its annotations.
        """
        if len(constraints) != 1:
            return False
        kind, disposition = constraints[0]
        if (kind != 'halogen-acid-amide'
                or disposition.status != 'higher-class-route'
                or not disposition.centers or not disposition.higher_groups
                or disposition.centers != frozenset().union(
                    *(group.centers for group in disposition.higher_groups))):
            return False
        mol, Chem = b.mol, b.Chem
        if (len(Chem.GetMolFrags(mol)) != 1 or mol.GetRingInfo().NumRings()
                or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
                or mol.GetStereoGroups()
                or any(bond.GetBondDir() != Chem.BondDir.NONE for bond in iter_bonds(mol))
                or any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                       or atom.GetChiralTag() not in {
                           Chem.ChiralType.CHI_UNSPECIFIED,
                           Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                           Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
                       for atom in iter_atoms(mol))):
            return False
        nx = [bond for bond in iter_bonds(mol)
              if 7 in {bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()}
              and {bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()}
              & {9, 17, 35, 53}]
        if len(nx) != 1 or nx[0].GetBondType() != Chem.BondType.SINGLE:
            return False
        if {disposition.nitrogen, disposition.halogen} != {
                nx[0].GetBeginAtomIdx(), nx[0].GetEndAtomIdx()}:
            return False
        n, x = mol.GetAtomWithIdx(disposition.nitrogen), mol.GetAtomWithIdx(disposition.halogen)
        roots = [atom for atom in n.GetNeighbors() if atom.GetIdx() != x.GetIdx()]
        return (x.GetDegree() == 1 and x.GetTotalValence() == 1
                and n.GetTotalValence() == 3 and not n.GetIsAromatic()
                and len(roots) in {1, 2}
                and all(atom.GetAtomicNum() == 6 for atom in roots)
                and all(bond.GetBondType() == Chem.BondType.SINGLE for bond in n.GetBonds())
                and n.GetTotalNumHs(includeNeighbors=True) == 2 - len(roots))

    @staticmethod
    def _preference_issues(decision):
        if decision.preference_issues:
            return decision.preference_issues
        comparison = decision.comparison
        return (SelectionLimitation(
            decision.stage, getattr(comparison, 'rule', '') or decision.direct_route,
            getattr(comparison, 'reason', '') or decision.requirement.kind),)

    @staticmethod
    def _resolve_deferred_parent(b, outcome, decision):
        """Require original parent/numbering witnesses in a completed pool.

        A later name by itself is insufficient. A typed provider may replace a
        compatibility spelling only while preserving the original skeletal
        parent, suffix sites and numbering, with complete mapped construction.
        """
        from .cross_route_selection import CrossRouteSelection
        from .component_mapping import fully_covered
        from .parent_selection import compare_parents

        required_frontiers = tuple(row[1] for row in outcome.evidence
            if isinstance(row, tuple) and len(row) == 2
            and row[0] == 'required_parent_frontier')
        if required_frontiers:
            return PrincipalSelectionPipeline._resolve_required_ring_parent(
                b, required_frontiers, decision)

        site_frontiers = []
        for row in outcome.evidence:
            if isinstance(row, dict) and row.get('principal_site_frontier') is not None:
                site_frontiers.append(row['principal_site_frontier'])
            elif (isinstance(row, tuple) and len(row) == 2
                  and row[0] == 'principal_site_frontier'):
                site_frontiers.append(row[1])
        if site_frontiers:
            return PrincipalSelectionPipeline._resolve_deferred_site_parent(
                b, tuple(site_frontiers), decision,
            )

        frontiers = []
        for row in outcome.evidence:
            if isinstance(row, dict) and row.get('selection_frontier') is not None:
                frontiers.append(row['selection_frontier'])
            elif isinstance(row, tuple) and len(row) == 2 and row[0] == 'selection_frontier':
                frontiers.append(row[1])
        witnesses = []
        seen = set()
        for frontier in frontiers:
            for candidate in frontier.candidates:
                identity = (candidate.name, frozenset(candidate.parent_atoms),
                            candidate.principal_centers, candidate.atom_locants)
                if identity not in seen:
                    witnesses.append(candidate)
                    seen.add(identity)
        selection = decision.comparison
        if not witnesses or not isinstance(selection, CrossRouteSelection):
            return DeferredParentResolution('pending', len(witnesses), 0,
                'No completed shared proposal pool retained the deferred witnesses.')
        chosen = selection.selected
        resolved = 0
        typed_routes = {'mapped-chain-oxygen', 'mapped-chain-imine-family',
                        'mapped-chain-imine', 'mapped-chain-oxime',
                        'mapped-chain-peroxol', 'mapped-chain-hydroxylamine',
                        'mapped-hydrazide'}
        for witness in witnesses:
            comparison = compare_parents(chosen.parent_selection, witness.parent_selection)
            if comparison.preference == -1 or (
                    comparison.preference == 0 and chosen.name == witness.name):
                resolved += 1
                continue
            represented = False
            for proposal in selection.candidates:
                if (frozenset(proposal.parent_atoms) != frozenset(witness.parent_atoms)
                        or proposal.principal_centers != witness.principal_centers):
                    continue
                original_map, richer_map = dict(witness.atom_locants), dict(proposal.atom_locants)
                required = set(witness.parent_atoms) | set(witness.principal_centers)
                if any(original_map.get(idx) != richer_map.get(idx) for idx in required):
                    continue
                typed = (proposal.route.startswith('mapped-ring:')
                         or proposal.route in typed_routes)
                construction = proposal.construction
                if proposal.name == witness.name or (
                        typed and construction is not None
                        and fully_covered(b.mol, construction.atoms,
                                          construction.embedded_atoms)):
                    represented = True
                    break
            if represented:
                resolved += 1
        return DeferredParentResolution(
            'complete' if resolved == len(witnesses) else 'pending',
            len(witnesses), resolved,
            'Every deferred parent was proved junior or retained with its original numbering.'
            if resolved == len(witnesses) else 'A deferred parent obligation remains unproved.')

    @staticmethod
    def _resolve_required_ring_parent(b, frontiers, decision):
        """Join the graph-selected ring to an actual complete mapped writer.

        This closes only multiplication of the P-44-selected senior frame.
        It does not reinterpret an unsuccessful P-51 enumerator as chemical
        ineligibility, or allow a general name to stand in for that frame.
        """
        from .complete_functional_parent import (
            FunctionalParentFrontier, _source_senior_ring_parent,
        )
        from .cross_route_selection import CrossRouteSelection
        from .component_mapping import fully_covered
        from .extended_inventory import selection_inventory

        rule = 'P-44.1.1/P-44.1.2/P-45.0/P-45.1.1'
        pending = lambda reason: DeferredParentResolution('pending', len(frontiers), 0, reason, rule)
        inventory = selection_inventory(b.mol)
        groups = tuple(inventory.principal)
        proof = _source_senior_ring_parent(b.mol, b.Chem, inventory, groups)
        if proof is None:
            return pending('The original graph no longer proves the full-suffix senior ring.')
        core = proof.parent_atoms
        if any(not isinstance(frontier, FunctionalParentFrontier)
               or frontier.status != 'out_of_scope'
               or frozenset(frontier.required_parent_atoms) != core
               or frontier.required_parent_proof != proof
               or not {'P-44.1.1', 'P-44.1.2'} <= set(frontier.source_parent_rules)
               for frontier in frontiers):
            return pending('A deferred frontier lacks the same source-proved physical ring frame.')
        selection = decision.comparison
        if not isinstance(selection, CrossRouteSelection):
            return pending('No completed shared proposal pool supplied the required ring.')
        chosen = selection.selected
        centers = frozenset(group.center for group in groups)
        construction = chosen.construction
        if (not chosen.route.startswith('mapped-ring:')
                or frozenset(chosen.parent_atoms) != core
                or chosen.principal_centers != centers
                or chosen.parent_selection.suffix_count != len(groups)
                or construction is None
                or construction.atoms & construction.embedded_atoms
                or not fully_covered(b.mol, construction.atoms, construction.embedded_atoms)
                or not core <= construction.atoms
                or any(not set(group.atoms) <= construction.atoms | construction.embedded_atoms
                       for group in groups)):
            return pending('The selected proposal lacks the complete original ring/suffix construction.')
        mapping, built = dict(chosen.atom_locants), dict(construction.atom_locants)
        if any(not mapping.get(idx) or mapping[idx] != built.get(idx) for idx in core):
            return pending('The selected ring numbering was not retained by its mapped construction.')
        return DeferredParentResolution(
            'complete', len(frontiers), len(frontiers),
            'The mapped pool retained the full-suffix senior ring; pendant rings carry no second copy of its selected suffix set.',
            rule, (proof,))

    @staticmethod
    def _resolve_deferred_site_parent(b, frontiers, decision):
        """Require every P-44 site-parent maximum to survive in the mapped pool."""
        from .cross_route_selection import CrossRouteSelection
        from .component_mapping import fully_covered
        from .principal_site_parent_frontier import (
            PrincipalSiteFrontierStatus,
        )

        if not isinstance(decision.comparison, CrossRouteSelection):
            return DeferredParentResolution(
                'pending', 0, 0,
                'The later shared pool did not return a typed cross-route selection.',
                'P-41/P-44.1.1/P-44.3/P-45.0',
            )

        def key(site):
            return (site.kind, site.principal_center, site.host_atom,
                    tuple(sorted(site.group_atoms)))

        required = set()
        witnesses = []
        copy_bounds = []
        source_site_inventory = None
        for frontier in frontiers:
            if (frontier.status is not PrincipalSiteFrontierStatus.PROVED
                    or not frontier.candidate_universe_complete
                    or not frontier.candidates
                    or frontier.subordinate_groups
                    or any(site.kind != frontier.principal_class
                           for site in frontier.sites)):
                return DeferredParentResolution(
                    'pending', len(witnesses), 0,
                    'A deferred site frontier lacks a complete source-closed maximum.',
                    'P-41/P-44.1.1/P-44.3/P-45.0',
                )
            current_site_inventory = frozenset(key(site) for site in frontier.sites)
            if (source_site_inventory is not None
                    and current_site_inventory != source_site_inventory):
                return DeferredParentResolution(
                    'pending', len(witnesses), 0,
                    'Deferred site frontiers disagree on the complete source-site inventory.',
                    'P-41/P-44.1.1/P-44.3/P-45.0', tuple(copy_bounds),
                )
            source_site_inventory = current_site_inventory
            required.update(current_site_inventory)
            total_sites = len(frontier.sites)
            total_carbons = sum(
                atom.GetAtomicNum() == 6 for atom in iter_atoms(b.mol)
            )
            for candidate in frontier.candidates:
                selected_sites = len(candidate.selected_sites)
                parent_carbons = len(candidate.parent_atoms)
                if selected_sites <= 0 or parent_carbons <= 0:
                    return DeferredParentResolution(
                        'pending', len(witnesses), 0,
                        'A P-44 maximum lacks a positive source-site or carbon-frame count.',
                        'P-44.1.1/P-44.3/P-45.0', tuple(copy_bounds),
                    )
                maximum_copies = min(
                    total_sites // selected_sites,
                    total_carbons // parent_carbons,
                )
                copy_bounds.append(PrincipalSiteCopyBound(
                    tuple(candidate.parent_atoms), selected_sites,
                    parent_carbons, total_sites, total_carbons,
                    maximum_copies,
                ))
                witnesses.append((frontier, candidate))

        # The least restrictive equivalent P-44 maximum controls. If two
        # disjoint copies of any selected senior frame remain resource-
        # possible, P-45 multiplication is still an unresolved alternative.
        # The current mapped pool has no typed multiplication certificate, so
        # it cannot discharge this obligation by choosing the substitutive
        # proposal alone.
        if not copy_bounds or max(row.maximum_disjoint_copies
                                  for row in copy_bounds) >= 2:
            return DeferredParentResolution(
                'pending', len(witnesses), 0,
                'P-45 multiplication remains possible for at least one P-44 maximum frame; the pool has no typed repeated-parent join.',
                'P-44.1.1/P-44.3/P-45.0/P-15.3.0', tuple(copy_bounds),
            )

        def matches(candidate, proposal):
            proposal_keys = frozenset(getattr(proposal, 'principal_site_keys', ()))
            selected_evidence = tuple(getattr(proposal, 'selected_site_evidence', ()))
            omitted_evidence = tuple(getattr(proposal, 'omitted_site_evidence', ()))
            selected = frozenset(key(site) for site in candidate.selected_sites)
            omitted = frozenset(key(site) for site in candidate.omitted_sites)
            proposal_selected = frozenset(key(site) for site in selected_evidence)
            proposal_omitted = frozenset(key(site) for site in omitted_evidence)
            if (tuple(proposal.parent_atoms) != tuple(candidate.parent_atoms)
                    or proposal_keys != frozenset(required)
                    or selected & omitted
                    or selected | omitted != frozenset(required)
                    or proposal_selected != selected
                    or proposal_omitted != omitted
                    or proposal_selected & proposal_omitted
                    or proposal_selected | proposal_omitted != proposal_keys):
                return False
            construction = getattr(proposal, 'construction', None)
            if (construction is None
                    or not fully_covered(b.mol, construction.atoms,
                                         construction.embedded_atoms)):
                return False
            mapping = dict(proposal.atom_locants)
            if any(mapping.get(atom) != str(locant)
                   for locant, atom in enumerate(candidate.parent_atoms, 1)):
                return False
            expected_site_locants = {
                (center, host): str(locant)
                for center, host, locant in candidate.site_locants
            }
            if any(mapping.get(host) != locant
                   for (_center, host), locant in expected_site_locants.items()):
                return False
            return True

        resolved = 0
        for _frontier, candidate in witnesses:
            if any(matches(candidate, proposal)
                   for proposal in decision.comparison.candidates):
                resolved += 1
        selected_is_maximum = any(
            matches(candidate, decision.comparison.selected)
            for _frontier, candidate in witnesses
        )
        complete = (bool(witnesses) and resolved == len(witnesses)
                    and selected_is_maximum)
        return DeferredParentResolution(
            'complete' if complete else 'pending', len(witnesses), resolved,
            'Every maximum site-parent orientation is retained, and the selected proposal is one of those maxima with exact source-site roles and numbering.'
            if complete else 'A maximum site-parent orientation is missing, or the selected mapped proposal is outside the proved maximum.',
            'P-41/P-44.1.1/P-44.3/P-45.0',
            tuple(copy_bounds),
        )

    @staticmethod
    def _with_class_constraints(decision, constraints):
        if not constraints:
            return decision
        evidence = (constraints[0][1] if len(constraints) == 1 else
                    SourceClassConstraintBundle(tuple(constraints),
                        'pending' if any(c.status in {'pending', 'preselected-parent', 'deferred'}
                                         for _kind, c in constraints) else 'higher-class-route'))
        return replace(decision, source_class_constraint=evidence)

    @staticmethod
    def _halogen_amide_barrier(disposition):
        if disposition.status == 'cyclic-imide-route':
            disposition = replace(
                disposition, status='pending',
                reason='Cyclic-imide topology is recognized, but a complete mapped pseudoketone writer is unavailable.')
        centers = tuple(i for i in (disposition.nitrogen, disposition.halogen) if i is not None)
        return PrincipalDecision(
            'naming-method', PrincipalRequirement('unresolved-halogen-acid-amide', centers),
            direct_route='unsupported-halogen-acid-amide:' + disposition.rule,
            comparison=disposition, source_class_constraint=disposition)

    @staticmethod
    def _nitrous_amide_barrier(disposition):
        recognition = disposition.recognition
        centers = tuple(idx for idx in (
            recognition.acid_nitrogen, recognition.amide_nitrogen) if idx is not None)
        return PrincipalDecision(
            'acid-derivative-method', PrincipalRequirement('unresolved-nitrous-amide', centers),
            direct_route='unsupported-nitrous-amide:P-67.1.2.6.3/P-41/P-42',
            comparison=disposition, source_class_constraint=disposition)

    @staticmethod
    def _sulfinamide_class(b):
        """Retain the neutral C-hosted P-66 class before writer eligibility."""
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not all(view.elements[e] for e in ('C', 'N', 'O', 'S')):
            return None
        if any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
               for atom in iter_atoms(b.mol)):
            return None
        if not any(atom.GetAtomicNum() == 16
                   and sum(neighbor.GetAtomicNum() == 8
                           and b.mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                           for neighbor in atom.GetNeighbors()) == 1
                   and any(neighbor.GetAtomicNum() == 7
                           for neighbor in atom.GetNeighbors())
                   and any(neighbor.GetAtomicNum() == 6
                           for neighbor in atom.GetNeighbors())
                   for atom in iter_atoms(b.mol)):
            return None
        from .sulfinamide_components import recognize_sulfinamides

        recognition = recognize_sulfinamides(b.mol, b.Chem)
        return recognition if recognition.status != 'not-applicable' else None

    @staticmethod
    def _sulfinamide_barrier(recognition):
        if recognition.status == 'higher-class-route':
            # The unresolved operation is writing a subordinate sulfinamide
            # prefix. It does not promote that sulfur to the principal group.
            return PrincipalDecision(
                'extended-functional-parent', PrincipalRequirement(
                    'unresolved-principal-with-sulfinamide-prefix',
                    tuple(sorted(recognition.higher_centers))),
                direct_route='unsupported-sulfinamide-prefix:P-41/P-66.1.1.4.2',
                comparison=recognition, source_class_constraint=recognition)
        return PrincipalDecision(
            'sulfinamide-method', PrincipalRequirement(
                'unresolved-sulfinamide',
                tuple(sorted(site.center for site in recognition.sites))),
            direct_route='unsupported-sulfinamide:P-41/P-42.1/P-66.1.1.2',
            comparison=recognition, source_class_constraint=recognition)

    def _sulfinamide_method(self, b, stage):
        recognition = self._sulfinamide_class(b)
        if recognition is None or recognition.status == 'higher-class-route':
            return None
        from .sulfinamide_components import sulfinamide_decision

        decision = sulfinamide_decision(b.mol, b.Chem)
        if decision is None:
            pending = self._sulfinamide_barrier(recognition)
            from .molecule_cache import mol_cache_get
            alternative = mol_cache_get(b.mol, 'configured_mixed_oxyamide_alternative')
            if alternative is not None:
                # The suffix writer declined. Keep its outstanding class
                # requirement while returning the mapped general construction.
                return replace(pending, output_name=alternative.name,
                    direct_route='general:configured-oxyamide-prefixes', preference_fallback=True,
                    atom_locants=alternative.atom_locants, preference_issues=alternative.preference_issues)
            return pending
        heavy = frozenset(atom.GetIdx() for atom in iter_atoms(b.mol)
                          if atom.GetAtomicNum() > 1)
        roles = tuple(frozenset(atoms) for _role, atoms, _mapping
                      in decision.component_roles)
        centers = frozenset(site.center for site in recognition.sites)
        if (not decision.name or decision.atoms != heavy
                or decision.principal_centers != centers or not roles
                or sum(map(len, roles)) != len(heavy)
                or frozenset().union(*roles) != heavy):
            return self._sulfinamide_barrier(recognition)
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('sulfinamide', tuple(sorted(centers)),
                'sulfinamide'), output_name=decision.name,
            direct_route='sulfinamide:P-66.1.1.2',
            atom_locants=decision.atom_locants, comparison=decision,
            source_class_constraint=recognition, preference_issues=decision.preference_issues,
        )

    def select_ion_component(self, backend):
        """Try whole mapped ion rules before legacy common-ion shortcuts.

        These producers prove complete, mutually exclusive component scopes.
        A decline preserves the existing naming path for other ion classes.
        """
        for stage, select in self.stages:
            if stage not in _ION_COMPONENT_STAGES:
                continue
            if backend.ctx.telemetry is not None:
                backend.ctx.telemetry.principal_stage_attempts[stage] += 1
            decision = select(backend, stage)
            if decision is not None:
                return decision
        return None

    def _principal_group_priority(self, b, stage):
        """Keep an unresolved P-41/P-42 comparison ahead of name writers.

        The inventory's semantic consumer has explicit source-class scopes.
        Its ties describe chemical class priority only;
        isotope partition and parent/suffix selection remain later operations.
        An unknown chemical comparison cannot be settled by a legacy route's
        tuple subrank or by whichever writer happens to produce a name first.
        """
        from .extended_inventory import selection_inventory

        frontier = selection_inventory(b.mol).source_frontier
        if frontier.status != 'barrier':
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_evidence[
                'P-41/P-42:principal-group-selection:False'] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('unresolved-principal-group-priority',
                tuple(sorted(group.center for group in frontier.groups))),
            direct_route='unsupported-principal-group-priority:P-41/P-42',
            comparison=frontier, source_class_constraint=frontier,
            preference_fallback=True,
        )

    def _source_general_pin_method(self, b, stage):
        """Source-backed full-graph parent and functional-class methods."""
        view = graph_view(b.mol, b.Chem)
        counts = method_bond_counts(b.mol, b.Chem)
        decision = None
        if (view.ring_count >= 2 and 0 < view.elements['C'] < view.atom_count
                and not any(view.aromatic_atoms)):
            rings = view.atom_rings
            if sum(map(len, rings)) == view.atom_count + len(rings) - 1 and not any(view.isotopes):
                from .heteropolycyclic_parents import bare_spiro_name
                name = bare_spiro_name(b.mol, b.Chem, rings)
            elif (len(rings) >= 4 or
                  set(view.elements) - {'C', 'H', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I'}):
                from .heteropolycyclic_parents import substituted_spiro_name
                name = substituted_spiro_name(b.mol, b.Chem, rings)
            else:
                name = ''
            if name:
                return PrincipalDecision(stage, PrincipalRequirement('heterocyclic-spiro'),
                    output_name=name, direct_route='skeletal-replacement:P-24.2.4.1.2')
        if counts[6, 7, 2] and counts[7, 8, 1] and counts[1, 8, 1] and any(view.isotopes):
            from .oxime_components import isotopic_hydroxy_oxime_name
            name = isotopic_hydroxy_oxime_name(b.mol, b.Chem)
            if name:
                return PrincipalDecision(stage, PrincipalRequirement('isotopic-oxime'),
                    output_name=name, direct_route='substitutive:P-68.3.1.1.2/P-82.6.2')
        if view.ring_count and counts[6, 7, 2] and counts[7, 8, 1]:
            from .oxime_components import cyclic_oxime_name
            name = cyclic_oxime_name(b.mol, b.Chem)
            if name:
                return PrincipalDecision(stage, PrincipalRequirement('cyclic-oxime'),
                    output_name=name, direct_route='substitutive:P-68.3.1.1.2')
        if counts[6, 7, 2] and view.elements['N'] >= 2:
            if counts[7, 7, 1]:
                from .hydrazone_methods import hydrazone_parent_decision, hydrazone_query
                if b.mol.HasSubstructMatch(hydrazone_query(b.Chem)):
                    decision = hydrazone_parent_decision(b.mol, b.Chem)
            if decision is None and (counts[6, 8, 2] or counts[6, 16, 2] or counts[6, 34, 2] or counts[6, 52, 2]):
                from .urea_methods import urea_decision, ylidene_urea_query
                if b.mol.HasSubstructMatch(ylidene_urea_query(b.Chem)):
                    decision = urea_decision(b.mol, b.Chem)
        if (decision is None and view.elements['N'] >= 2
                and (counts[6, 16, 2] or counts[6, 34, 2] or counts[6, 52, 2]) and any(view.isotopes)):
            from .functional_class_queries import urea_query
            if b.mol.HasSubstructMatch(urea_query(b.Chem)):
                from .urea_methods import urea_decision
                decision = urea_decision(b.mol, b.Chem)
        if (decision is None and view.elements['N'] and counts[8, 16, 2] + counts[8, 34, 2]
                + counts[8, 52, 2] and (view.elements['Se'] or view.elements['Te']
                or counts[8, 16, 2] < 2 * view.elements['S'])):
            from .chalcogen_amide_parents import chalcogen_amide_decision
            decision = chalcogen_amide_decision(b.mol, b.Chem)
        if (decision is None and view.ring_count and view.elements['O']
                and (counts[6, 16, 2] or counts[6, 34, 2] or counts[6, 52, 2]
                     or view.ring_count >= 2 and counts[6, 8, 2] and counts[6, 8, 1] >= 2)):
            from .lactone_parents import lactone_decision
            decision = lactone_decision(b.mol, b.Chem)
        if decision is not None:
            return PrincipalDecision(stage, PrincipalRequirement('substitutive-parent',
                tuple(sorted(decision.principal_centers)), decision.method),
                output_name=decision.name, direct_route=decision.method + ':' + decision.rule,
                atom_locants=decision.atom_locants, comparison=decision, preference_issues=decision.preference_issues)
        if any(view.isotopes):
            from .bluebook_general_methods import (isotopic_benzene_suffix_decision, isotopic_formate_decision,
                isotopic_halomethane_decision, isotopic_methoxybenzene_decision, isotopic_phenylketone_decision)
            if view.ring_count == 1 and view.elements['C'] == 7:
                decision = isotopic_benzene_suffix_decision(b.mol, b.Chem)
            if view.elements['C'] == 1 and view.elements['O'] == 2:
                decision = isotopic_formate_decision(b.mol, b.Chem)
            if decision is None and view.elements['C'] == 1:
                decision = isotopic_halomethane_decision(b.mol, b.Chem)
            if decision is None and view.ring_count == 1 and view.elements['O'] == 1:
                decision = (isotopic_methoxybenzene_decision(b.mol, b.Chem)
                            or isotopic_phenylketone_decision(b.mol, b.Chem))
        if view.elements['C'] == 1 and view.elements['N'] == 1 and counts[6, 7, 3]:
            from .bluebook_general_methods import retained_cyanic_or_formonitrile
            decision = decision or retained_cyanic_or_formonitrile(b.mol, b.Chem)
        if (decision is None and counts[6, 7, 3]
                and any(charge > 0 for charge in view.formal_charges)):
            from .bluebook_general_methods import nitrile_chalcogenide_decision
            decision = nitrile_chalcogenide_decision(b.mol, b.Chem)
        if (decision is None and (view.elements['B'] or view.elements['Si'])
                and any(view.elements[e] for e in ('N', 'F', 'Cl', 'Br', 'I'))):
            from .bluebook_general_methods import boron_silicon_halide_decision
            decision = boron_silicon_halide_decision(b.mol, b.Chem)
        if decision is None and any(charge < 0 for charge in view.formal_charges):
            from .bluebook_general_methods import hydride_addition_anion_decision
            decision = hydride_addition_anion_decision(b.mol, b.Chem)
        if (decision is None and view.elements['B'] and view.elements['N']
                and any(charge < 0 for charge in view.formal_charges)):
            from .bluebook_general_methods import boron_cyanatido_anion_decision
            decision = boron_cyanatido_anion_decision(b.mol, b.Chem)
        if (decision is None and view.elements['C'] and view.elements['O']
                and any(charge < 0 for charge in view.formal_charges)):
            from .bluebook_general_methods import modified_carboxylate_decision
            decision = modified_carboxylate_decision(b.mol, b.Chem)
        if (decision is None and view.elements['C'] >= 3
                and any(view.elements[e] for e in ('S', 'Se', 'Te'))):
            from .bluebook_general_methods import modified_carbonic_diester_decision
            decision = modified_carbonic_diester_decision(b.mol, b.Chem)
        if (decision is None and view.elements['C'] and view.elements['N']
                and any(view.elements[e] for e in ('P', 'As', 'Sb', 'S'))):
            from .bluebook_general_methods import noncarbon_acid_pseudohalide_decision
            decision = noncarbon_acid_pseudohalide_decision(b.mol, b.Chem)
        if (decision is None and view.elements['N'] == 1
                and any(view.elements[e] for e in ('S', 'Se', 'Te'))):
            from .bluebook_general_methods import n_chalcogen_amine_decision
            decision = n_chalcogen_amine_decision(b.mol, b.Chem)
        if (decision is None and view.elements['N'] and counts[6, 7, 2]
                and any(view.elements[e] for e in ('O', 'S', 'Se', 'Te'))):
            from .bluebook_general_methods import carbonic_imido_hydrazono_decision
            decision = carbonic_imido_hydrazono_decision(b.mol, b.Chem)
        if (decision is None and view.elements['N'] == 3
                and sum(view.elements[e] for e in ('O', 'S', 'Se', 'Te')) == 1
                and counts[6, 7, 2] and counts[7, 7, 1]):
            from .bluebook_general_methods import semicarbazone_decision
            decision = semicarbazone_decision(b.mol, b.Chem)
        if (decision is None and view.elements['N'] and view.elements['O'] >= 2
                and any(view.elements[e] for e in ('F', 'Cl', 'Br', 'I'))):
            from .bluebook_general_methods import amide_competing_acid_halide_decision
            decision = amide_competing_acid_halide_decision(b.mol, b.Chem)
        if (decision is None and view.elements['O'] == 2
                and any(view.elements[e] for e in ('P', 'As', 'Sb'))):
            from .bluebook_general_methods import phosphane_carboxylic_acid_decision
            decision = phosphane_carboxylic_acid_decision(b.mol, b.Chem)
        if decision is None and view.ring_count == 1 and view.elements['N'] and view.elements['C']:
            from .bluebook_general_methods import mancude_azocine_decision
            decision = mancude_azocine_decision(b.mol, b.Chem)
        if decision is None and view.ring_count == 2 and view.elements['C'] >= 8:
            from .bluebook_general_methods import polyalene_parent_decision
            decision = polyalene_parent_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(stage, PrincipalRequirement('source-general-pin',
            tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, direct_route=decision.method + ':' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision)

    def _preferred_functional_class_method(self, b, stage):
        from .preferred_functional_classes import select
        decision = select(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(stage, PrincipalRequirement('preferred-functional-class',
            tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, direct_route=decision.method + ':' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision)

    def _full_bluebook_frontier(self, b, stage):
        """Source-directed, whole-graph constructions for the full engine."""
        from .full_bluebook_frontier import select
        decision = select(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(stage, PrincipalRequirement('source-functional-family',
            tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, direct_route=decision.method + ':' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision)

    def _source_hydride_assembly_method(self, b, stage):
        """Complete P-28, P-67 and P-73 source-scoped constructions."""
        view = graph_view(b.mol, b.Chem)
        counts = method_bond_counts(b.mol, b.Chem)
        decision = None
        if view.ring_count == 1 and any(charge > 0 for charge in view.formal_charges):
            from .bluebook_hydride_assembly_methods import parent_ring_cation_decision
            decision = parent_ring_cation_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] and view.elements['O'] >= 2:
            from .bluebook_hydride_assembly_methods import hydroxylamine_suffix_decision
            decision = hydroxylamine_suffix_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] == 3 and counts[7, 7, 1]:
            from .bluebook_hydride_assembly_methods import hydrazinyl_amine_decision
            decision = hydrazinyl_amine_decision(b.mol, b.Chem)
        if (decision is None and not view.ring_count and view.elements['N'] == 2
                and view.elements['C'] >= 4):
            from .bluebook_hydride_assembly_methods import methyl_diamine_numbering_decision
            decision = methyl_diamine_numbering_decision(b.mol, b.Chem)
        if (decision is None and (view.elements['P'] or view.elements['S']) and view.elements['O']
                and (view.elements['N'] or any(view.elements[e] for e in ('F', 'Cl', 'Br', 'I')))):
            from .bluebook_hydride_assembly_methods import noncarbon_replacement_acid_decision
            decision = noncarbon_replacement_acid_decision(b.mol, b.Chem)
        if (decision is None and any(view.elements[e] for e in ('P', 'As', 'Sb'))
                and any(view.elements[e] for e in ('F', 'Cl', 'Br', 'I'))):
            from .bluebook_hydride_assembly_methods import phosphorus_halide_decision
            decision = phosphorus_halide_decision(b.mol, b.Chem)
        if decision is None and view.elements['O'] >= 4 and counts[6, 8, 1] >= 4:
            from .bluebook_hydride_assembly_methods import linear_dianhydride_decision
            decision = linear_dianhydride_decision(b.mol, b.Chem)
        if decision is None and view.elements['O'] >= 2 and view.elements['C']:
            from .bluebook_hydride_assembly_methods import mixed_anhydride_decision
            decision = mixed_anhydride_decision(b.mol, b.Chem)
        if (decision is None and view.elements['N'] == 1 and view.elements['C'] >= 2
                and any(view.elements[e] for e in ('Si', 'Ge', 'Sn', 'Pb'))):
            from .bluebook_hydride_assembly_methods import numbered_group14_amine_decision
            decision = numbered_group14_amine_decision(b.mol, b.Chem)
        if (decision is None and view.ring_count == 1 and view.elements['Si'] >= 2
                and view.elements['C'] >= 6):
            from .bluebook_hydride_assembly_methods import benzene_silyl_multiplication
            decision = benzene_silyl_multiplication(b.mol, b.Chem)
        if (decision is None and view.ring_count >= 2 and view.elements['O'] == 1
                and (view.elements['C'] >= 14 or any(view.elements[e] for e in ('F', 'Cl', 'Br', 'I')))):
            from .bluebook_hydride_assembly_methods import substituted_diaryloxy_multiplication
            decision = substituted_diaryloxy_multiplication(b.mol, b.Chem)
        if (decision is None and view.ring_count >= 2 and any(
                not bond.IsInRing() and bond.GetBondTypeAsDouble() == 1
                and bond.GetBeginAtom().IsInRing() and bond.GetEndAtom().IsInRing()
                for bond in iter_bonds(b.mol))):
            from .bluebook_hydride_assembly_methods import indicated_h_ring_assembly_decision
            decision = indicated_h_ring_assembly_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(stage, PrincipalRequirement('source-hydride-assembly',
            tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, direct_route=decision.method + ':' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision)

    def _source_fusion_functional_method(self, b, stage):
        """Complete source-proved acid, ester and hydride constructions."""
        view = graph_view(b.mol, b.Chem)
        counts = method_bond_counts(b.mol, b.Chem)
        decision = None
        if (view.elements['N'] >= 2 and view.elements['O'] >= 2
                and counts[7, 7, 2] and any(view.formal_charges)):
            from .bluebook_fusion_functional_methods import diazo_ester_decision
            decision = diazo_ester_decision(b.mol, b.Chem)
        if (decision is None and view.elements['N'] == 2 and view.elements['O'] >= 2
                and counts[7, 7, 1]):
            from .bluebook_fusion_functional_methods import hydrazine_carboxylic_acid_decision
            decision = hydrazine_carboxylic_acid_decision(b.mol, b.Chem)
        if (decision is None and view.elements['C']
                and sum(view.elements[e] for e in ('S', 'Se', 'Te')) >= 2):
            from .bluebook_fusion_functional_methods import modified_chalcogen_acid_decision
            decision = modified_chalcogen_acid_decision(b.mol, b.Chem)
        if (decision is None and view.elements['N'] and view.elements['O']
                and (view.elements['P'] or view.elements['S'])):
            from .bluebook_fusion_functional_methods import amino_noncarbon_acid_decision
            decision = amino_noncarbon_acid_decision(b.mol, b.Chem)
        if (decision is None and view.elements['C'] and view.elements['O'] >= 3
                and (view.elements['P'] or view.elements['B'])):
            from .bluebook_fusion_functional_methods import acidic_mixed_anhydride_decision
            decision = acidic_mixed_anhydride_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] and counts[6, 7, 2]:
            from .bluebook_fusion_functional_methods import (
                aryl_isocyanato_sulfonyl_halide_decision, isocyanato_carboxylic_acid_decision,
                isocyanato_hydride_decision, isocyanato_noncarbon_acid_decision)
            decision = aryl_isocyanato_sulfonyl_halide_decision(b.mol, b.Chem)
            if decision is None:
                decision = isocyanato_carboxylic_acid_decision(b.mol, b.Chem)
            if decision is None:
                decision = isocyanato_noncarbon_acid_decision(b.mol, b.Chem)
            if decision is None:
                decision = isocyanato_hydride_decision(b.mol, b.Chem)
        if (decision is None and view.elements['C'] >= 2 and view.elements['O']
                and (view.elements['P'] or view.elements['Si'] or view.elements['N'])):
            from .bluebook_fusion_functional_methods import acyclic_pseudoketone_decision
            decision = acyclic_pseudoketone_decision(b.mol, b.Chem)
        if decision is None and view.ring_count == 1 and view.elements['P'] == 1 and view.elements['C'] >= 4:
            from .bluebook_fusion_functional_methods import phosphole_hydrogenation_decision
            decision = phosphole_hydrogenation_decision(b.mol, b.Chem)
        if decision is None and view.ring_count == 5 and view.elements['C'] >= 16:
            from .fusion_row_constructions import horizontal_row_fusion_decision
            decision = horizontal_row_fusion_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(stage, PrincipalRequirement('source-fusion-functional',
            tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, direct_route=decision.method + ':' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision)

    def _source_azoxy_spiro_method(self, b, stage):
        """Source-specific whole-graph methods with explicit class guards."""
        view = graph_view(b.mol, b.Chem)
        counts = method_bond_counts(b.mol, b.Chem)
        decision = None
        if (view.elements['N'] >= 2 and view.elements['O'] >= 1
                and counts[7, 7, 2] and any(view.formal_charges)):
            from .bluebook_azoxy_methods import azoxy_decision
            decision = azoxy_decision(b.mol, b.Chem)
        if (decision is None and view.ring_count == 1 and view.elements['N']
                and any(charge > 0 for charge in view.formal_charges)):
            from .bluebook_azoxy_methods import aromatic_ring_cation_decision
            decision = aromatic_ring_cation_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] and view.elements['C'] and counts[6, 7, 3]:
            from .bluebook_azoxy_methods import cyanic_ester_decision
            decision = cyanic_ester_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] and view.elements['O'] >= 2 and counts[6, 7, 2]:
            from .imidic_acid_methods import n_hydroxyimidic_acid_decision
            decision = n_hydroxyimidic_acid_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] and view.elements['S']:
            from .bluebook_azoxy_methods import thiohydroxylamine_decision
            decision = thiohydroxylamine_decision(b.mol, b.Chem)
        if (decision is None and view.elements['N'] >= 2 and view.elements['O']
                and any(view.elements[e] for e in ('S', 'Se', 'Te'))):
            from .bluebook_azoxy_methods import chalcogen_oxoacid_diamide_decision
            decision = chalcogen_oxoacid_diamide_decision(b.mol, b.Chem)
        if decision is None and view.ring_count == 2 and view.elements['N'] >= 2:
            from .bluebook_azoxy_methods import biphenyl_polyamine_decision
            decision = biphenyl_polyamine_decision(b.mol, b.Chem)
        if decision is None and view.ring_count >= 4 and view.elements['C']:
            from .bluebook_azoxy_methods import identical_polyspiro_decision
            decision = identical_polyspiro_decision(b.mol, b.Chem)
        if (decision is None and view.elements['C']
                and sum(view.elements[e] for e in ('O', 'S', 'Se', 'Te')) == 2
                and sum(bool(view.elements[e]) for e in ('O', 'S', 'Se', 'Te')) == 2):
            from .bluebook_azoxy_methods import mixed_peroxide_decision
            decision = mixed_peroxide_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(stage, PrincipalRequirement('source-azoxy-spiro',
            tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, direct_route=decision.method + ':' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision)

    def _source_functional_family_method(self, b, stage):
        """Source-specific acid, nitrile, azinic and heterohydride classes."""
        view = graph_view(b.mol, b.Chem)
        decision = None
        if any(number == 1 and isotope == 2 for number, isotope in zip(view.atomic_nums, view.isotopes)):
            from .bluebook_lambda_amine_methods import isotopic_alkene_decision
            decision = isotopic_alkene_decision(b.mol, b.Chem)
        if (decision is None and view.ring_count == 1
                and any(view.elements[e] for e in ('S', 'Se', 'Te', 'P', 'I'))
                and any(a.IsInRing() and a.GetAtomicNum() in {15, 16, 34, 52, 53}
                        and a.GetTotalValence() != {15: 3, 16: 2, 34: 2, 52: 2, 53: 1}[a.GetAtomicNum()]
                        for a in iter_atoms(b.mol))):
            from .bluebook_lambda_amine_methods import nonstandard_heteromonocycle_decision
            decision = nonstandard_heteromonocycle_decision(b.mol, b.Chem)
        if decision is None and view.elements['Si'] == 1 and view.elements['O'] >= 2 and view.elements['C']:
            from .bluebook_lambda_amine_methods import silyl_carboxylic_acid_decision, silyl_diol_decision
            decision = silyl_carboxylic_acid_decision(b.mol, b.Chem)
            if decision is None and view.elements['O'] == 2:
                decision = silyl_diol_decision(b.mol, b.Chem)
        if (decision is None and view.ring_count and view.elements['N'] >= 2
                and view.elements['C'] and method_bond_counts(b.mol, b.Chem)[6, 7, 2]):
            from .bluebook_lambda_amine_methods import ring_amidine_decision
            decision = ring_amidine_decision(b.mol, b.Chem)
        if (decision is None and view.elements['N'] >= 2 and view.elements['O']
                and method_bond_counts(b.mol, b.Chem)[7, 8, 1]
                and method_bond_counts(b.mol, b.Chem)[6, 7, 2]):
            from .amidine_methods import amidoxime_decision
            decision = amidoxime_decision(b.mol, b.Chem)
        if (decision is None and view.elements['C'] and view.elements['N']
                and view.elements['O'] and view.elements['S']
                and method_bond_counts(b.mol, b.Chem)[6, 16, 2]):
            from .bluebook_lambda_amine_methods import mixed_thioamide_decision
            decision = mixed_thioamide_decision(b.mol, b.Chem)
        if (decision is None and view.elements['C'] and view.elements['N'] and view.elements['S'] >= 2
                and method_bond_counts(b.mol, b.Chem)[6, 16, 2] >= 2):
            from .bluebook_lambda_amine_methods import symmetric_thioimide_decision
            decision = symmetric_thioimide_decision(b.mol, b.Chem)
        if (decision is None and view.elements['C'] and view.elements['N'] and view.elements['O'] >= 2
                and any(view.elements[e] for e in ('S', 'Se', 'Te'))):
            from .bluebook_lambda_amine_methods import sulfonyl_carboxamide_decision
            decision = sulfonyl_carboxamide_decision(b.mol, b.Chem)
        if (decision is None and view.elements['C']
                and any(view.elements[e] for e in ('F', 'Cl', 'Br', 'I'))
                and any(view.elements[e] for e in ('O', 'S', 'Se', 'Te'))
                and any(len(view.neighbors[i]) == 1
                        and view.atomic_nums[view.neighbors[i][0]] in {6, 16, 34, 52}
                        and any(view.atomic_nums[j] in {8, 16, 34, 52}
                                and view.bond_orders.get(tuple(sorted((view.neighbors[i][0], j)))) == 2
                                for j in view.neighbors[view.neighbors[i][0]])
                        for i, number in enumerate(view.atomic_nums) if number in {9, 17, 35, 53})):
            from .bluebook_lambda_amine_methods import acid_halide_decision
            decision = acid_halide_decision(b.mol, b.Chem)
        ion_atoms = (tuple((number, charge) for number, charge in
                           zip(view.atomic_nums, view.formal_charges) if charge)
                     if any(view.formal_charges) else ())
        possible_source_ion = (view.elements['C'] and (
            (view.elements['N'] >= 2 and any(number == 7 and charge > 0 for number, charge in ion_atoms))
            or (any(number in {6, 8, 15, 16} and charge > 0 for number, charge in ion_atoms)
                and any(view.elements[e] for e in ('O', 'S', 'Se', 'Te')))
            or any(number in {16, 34, 52} and charge < 0 for number, charge in ion_atoms)
            or (view.elements['O'] >= 2 and any(number == 8 and charge < 0 for number, charge in ion_atoms))))
        if decision is None and possible_source_ion:
            from .bluebook_ionic_acid_methods import ionic_source_decision
            decision = ionic_source_decision(b.mol, b.Chem)
        if decision is None and view.elements['P'] and view.elements['O'] >= 5 and view.elements['C']:
            from .bluebook_ionic_acid_methods import phosphono_carboxylic_acid_decision
            decision = phosphono_carboxylic_acid_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] == 1 and view.elements['O'] >= 2 and view.elements['C']:
            from .bluebook_ionic_acid_methods import nitrogen_oxoester_decision
            decision = nitrogen_oxoester_decision(b.mol, b.Chem)
        if (decision is None and view.elements['C']
                and (view.elements['F'] or view.elements['Cl'] or view.elements['Br'] or view.elements['I'])
                and sum(view.elements[e] for e in ('F', 'Cl', 'Br', 'I')) == 1):
            halo = next(i for i, number in enumerate(view.atomic_nums) if number in {9, 17, 35, 53})
            if len(view.neighbors[halo]) == 1:
                neighbor = view.atomic_nums[view.neighbors[halo][0]]
                if neighbor in {15, 33} and view.elements['O']:
                    from .bluebook_ionic_acid_methods import pnictogen_halogen_ester_decision
                    decision = pnictogen_halogen_ester_decision(b.mol, b.Chem)
                elif neighbor in {8, 16, 34, 52}:
                    from .bluebook_ionic_acid_methods import hypohalite_ester_decision
                    decision = hypohalite_ester_decision(b.mol, b.Chem)
        if (decision is None and (view.elements['P'] or view.elements['As']) and view.elements['O']
                and any(view.elements[e] for e in ('S', 'Se', 'Te'))):
            from .bluebook_ionic_acid_methods import pnictogen_chalcogen_ester_decision
            decision = pnictogen_chalcogen_ester_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] and view.elements['C'] and method_bond_counts(b.mol, b.Chem)[6, 7, 3]:
            if view.elements['O'] and any(view.elements[e] for e in ('S', 'Se', 'Te')):
                from .bluebook_ionic_acid_methods import sulfur_cyanide_decision
                decision = sulfur_cyanide_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] and view.elements['C'] and method_bond_counts(b.mol, b.Chem)[6, 7, 3]:
            from .carbonic_cyanide_methods import carbonic_cyanide_decision, cyanic_pseudohalide_decision
            decision = cyanic_pseudohalide_decision(b.mol, b.Chem)
            if decision is None:
                decision = carbonic_cyanide_decision(b.mol, b.Chem)
        if decision is None and view.elements['C'] and any(view.elements[e] for e in ('O', 'S', 'Se', 'Te')):
            from .functional_replacement_acids import (carbonic_replacement_acid_decision,
                                                       carboxylic_chalcogen_acid_decision)
            decision = carbonic_replacement_acid_decision(b.mol, b.Chem)
            if decision is None and any(view.elements[e] for e in ('S', 'Se', 'Te')):
                decision = carboxylic_chalcogen_acid_decision(b.mol, b.Chem)
        if decision is None and view.elements['S'] and view.elements['N'] and view.elements['C']:
            from .bluebook_ionic_acid_methods import imidothioic_acid_decision, thioamide_decision
            if method_bond_counts(b.mol, b.Chem)[6, 7, 2]:
                decision = imidothioic_acid_decision(b.mol, b.Chem)
            if decision is None and method_bond_counts(b.mol, b.Chem)[6, 16, 2]:
                decision = thioamide_decision(b.mol, b.Chem)
        if (decision is None and view.elements['O'] and view.elements['C'] >= 2
                and any(view.elements[e] >= 2 for e in ('S', 'Se', 'Te'))):
            from .bluebook_ionic_acid_methods import symmetric_modified_anhydride_decision
            decision = symmetric_modified_anhydride_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] and view.elements['O'] >= 2 and any(view.formal_charges):
            from .hetero_functional_methods import aci_nitro_acid_decision, aci_nitro_senior_acid_decision
            decision = aci_nitro_senior_acid_decision(b.mol, b.Chem)
            if decision is None:
                decision = aci_nitro_acid_decision(b.mol, b.Chem)
        heterohost = any(view.elements[e] for e in ('B', 'Si', 'P', 'Ge', 'As', 'Sn', 'Sb', 'Pb', 'Bi'))
        if decision is None and view.elements['N'] and heterohost:
            from .hetero_functional_methods import heterohydride_carbonitrile_decision
            decision = heterohydride_carbonitrile_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] and (heterohost or view.ring_count):
            from .hetero_functional_methods import heteroamine_decision
            decision = heteroamine_decision(b.mol, b.Chem)
        if decision is None and view.elements['O'] and heterohost:
            from .hetero_functional_methods import acyclic_heterone_decision
            decision = acyclic_heterone_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(stage, PrincipalRequirement('source-functional-family',
            tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, direct_route=decision.method + ':' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision)

    def _source_hetero_parent_method(self, b, stage):
        """Source-closed heterol, imidic acid, hydrazine and amide scopes."""
        view = graph_view(b.mol, b.Chem)
        decision = None
        if (view.elements['N'] and view.elements['C'] and any(view.formal_charges)
                and method_bond_counts(b.mol, b.Chem)[6, 7, 3]):
            from .isocyano_methods import isocyano_decision, isocyano_general_decision
            decision = isocyano_decision(b.mol, b.Chem)
            if decision is None:
                decision = isocyano_general_decision(b.mol, b.Chem)
        if decision is None and view.elements['C'] and any(view.elements[e] for e in ('S', 'Se', 'Te')):
            from .chalcogenol_methods import chalcogenol_decision
            decision = chalcogenol_decision(b.mol, b.Chem)
        if decision is None and view.elements['C'] >= 2 and view.elements['N'] and view.elements['O'] >= 3:
            from .oxamic_acid_methods import oxamic_acid_decision
            decision = oxamic_acid_decision(b.mol, b.Chem)
        if (decision is None and view.elements['O'] and any(view.elements[e] for e in
                ('B', 'Si', 'P', 'Ge', 'As', 'Sn', 'Sb', 'Pb', 'Bi'))):
            from .heterohydride_carbaldehyde_methods import heterohydride_carbaldehyde_decision
            decision = heterohydride_carbaldehyde_decision(b.mol, b.Chem)
        if decision is None and view.elements['S'] >= 6 and view.elements['C']:
            from .polysulfane_methods import symmetric_trisulfane_decision
            decision = symmetric_trisulfane_decision(b.mol, b.Chem)
        if decision is None and any(view.elements[e] for e in ('P', 'As', 'Sb')) and view.elements['O']:
            from .heterol_methods import pnictogen_ous_acid_decision
            decision = pnictogen_ous_acid_decision(b.mol, b.Chem)
        if decision is None and any(view.elements[e] for e in ('Si', 'Ge', 'Sn', 'Pb')) and view.elements['O']:
            from .heterol_methods import group14_cyclic_heterol_decision, group14_heterol_decision
            decision = group14_heterol_decision(b.mol, b.Chem)
            if decision is None and view.ring_count:
                decision = group14_cyclic_heterol_decision(b.mol, b.Chem)
        if decision is None and any(view.elements[e] for e in
                ('B', 'Si', 'P', 'Ge', 'As', 'Sn', 'Sb', 'Pb', 'Bi')) and view.elements['N']:
            from .heterohydride_carboxamide_methods import heterohydride_carboxamide_decision
            decision = heterohydride_carboxamide_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] and view.elements['O']:
            if method_bond_counts(b.mol, b.Chem)[6, 7, 2]:
                from .imidic_acid_methods import imidic_acid_decision
                decision = imidic_acid_decision(b.mol, b.Chem)
        if decision is None and view.ring_count and view.elements['N'] and view.elements['O']:
            from .heterol_methods import cyclic_heterol_decision
            decision = cyclic_heterol_decision(b.mol, b.Chem)
            if decision is None:
                from .hydroxylamine_methods import aromatic_amine_decision
                decision = aromatic_amine_decision(b.mol, b.Chem)
        if (decision is None and view.ring_count
                and (any(view.isotopes) or view.elements['N'] >= 3 or any(view.elements[e] for e in
                     ('P', 'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga', 'In', 'Tl')))):
            from .saturated_heterocycle_methods import saturated_heteromonocycle_decision
            decision = saturated_heteromonocycle_decision(b.mol, b.Chem)
        if (decision is None and view.elements['N'] == 2 and view.elements['C']
                and method_bond_counts(b.mol, b.Chem)[6, 7, 3]):
            from .cyanamide_methods import (cyano_amino_acid_decision,
                                            cyano_carboxamide_decision, cyanamide_decision)
            if view.elements['O']:
                decision = cyano_amino_acid_decision(b.mol, b.Chem)
                if decision is None:
                    decision = cyano_carboxamide_decision(b.mol, b.Chem)
            if decision is None:
                decision = cyanamide_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] == 2 and view.elements['C']:
            from .hydrazine_methods import hydrocarbyl_hydrazine_decision
            decision = hydrocarbyl_hydrazine_decision(b.mol, b.Chem)
        if decision is None and view.ring_count >= 2 and any(view.elements[e] for e in ('N', 'Se', 'Te')):
            from .retained_fused_closed_methods import retained_fused_closed_decision
            decision = retained_fused_closed_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(stage, PrincipalRequirement('source-hetero-parent',
            tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, direct_route='substitutive:' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision)

    def _source_ester_method(self, b, stage):
        """Apply complete, source-specific ester classes before generic search."""
        from .modified_ester_methods import (modified_carboxylic_ester_decision,
                                              modified_sulfur_ester_decision,
                                              sulfuric_ester_decision)

        view = graph_view(b.mol, b.Chem)
        decision = None
        if view.elements['Si'] and view.elements['O'] >= 4:
            from .silyl_peroxy_ester_methods import silyl_peroxy_ester_decision
            decision = silyl_peroxy_ester_decision(b.mol, b.Chem)
        if decision is None and view.elements['S'] and view.elements['O'] >= 3:
            decision = sulfuric_ester_decision(b.mol, b.Chem)
        if decision is None and view.elements['S']:
            decision = modified_sulfur_ester_decision(b.mol, b.Chem)
        if decision is None and any(view.elements[e] for e in ('S', 'Se', 'Te', 'N')):
            decision = modified_carboxylic_ester_decision(b.mol, b.Chem)
        if decision is None and view.elements['N'] and view.elements['O'] >= 2:
            from .amino_acid_esters import retained_amino_acid_ester_decision

            decision = retained_amino_acid_ester_decision(b.mol, b.Chem)
        if (decision is None and view.elements['P'] and view.elements['N'] >= 3
                and view.elements['O'] >= 5):
            from .fused_hydrogen_state import benzimidazole_imino_ester_decision

            decision = benzimidazole_imino_ester_decision(b.mol, b.Chem)
        if decision is None and view.elements['O'] and (view.elements['P'] or view.elements['S']):
            from .oxoacid_ester_methods import oxoacid_ester_decision

            decision = oxoacid_ester_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(stage, PrincipalRequirement('source-ester',
            tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, direct_route='functional_class:' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision,
            preference_issues=decision.preference_issues)

    def _retained_ether_method(self, b, stage):
        view = graph_view(b.mol, b.Chem)
        if view.elements['C'] != 7 or view.elements['O'] != 1 or not view.ring_count:
            return None
        from .retained_ether_methods import retained_ether_decision
        decision = retained_ether_decision(b.mol, b.Chem)
        if decision is None:
            return None
        return PrincipalDecision(stage, PrincipalRequirement('retained-ether'),
            output_name=decision.name, direct_route='retained_ether:' + decision.rule,
            comparison=decision)

    def _carbon_acid_derivative_method(self, b, stage):
        from .carbon_acid_derivative_methods import carbon_acid_derivative_frontier

        frontier = carbon_acid_derivative_frontier(b.mol, b.Chem)
        if frontier.status == 'outside':
            return None
        centers = (() if frontier.center is None else (frontier.center,))
        principal_kind = ('acid' if frontier.classification == 'carboxylic-acid'
                          else 'carbonic-acid')
        if frontier.status != 'complete' or frontier.decision is None:
            return PrincipalDecision(
                stage, PrincipalRequirement('unresolved-' + principal_kind, centers),
                direct_route='unsupported-carbon-acid-derivative:' + frontier.rule,
                comparison=frontier, source_class_constraint=frontier,
                preference_fallback=frontier.status == 'pending')
        decision = frontier.decision
        heavy = frozenset(atom.GetIdx() for atom in iter_atoms(b.mol)
                          if atom.GetAtomicNum() > 1)
        roles = tuple(frozenset(atoms) for _role, atoms, _mapping
                      in decision.component_roles)
        if (not decision.name or decision.atoms != heavy
                or decision.principal_centers != frozenset(centers)
                or not roles or sum(map(len, roles)) != len(heavy)
                or frozenset().union(*roles) != heavy):
            return PrincipalDecision(
                stage, PrincipalRequirement('unresolved-' + principal_kind, centers),
                direct_route='unsupported-carbon-acid-derivative:ownership',
                comparison=frontier, source_class_constraint=frontier)
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement(principal_kind, centers, decision.method),
            output_name=decision.name,
            direct_route='carbon_acid_derivative:' + frontier.rule,
            atom_locants=decision.atom_locants, comparison=frontier,
            source_class_constraint=frontier)

    def _choose(
        self, backend, stage, requirement, group, *, eligible=None, first=False
    ):
        candidates = backend.prepare(self.groups[group])
        if eligible is not None:
            candidates = [item for item in candidates if eligible(item)]
        item = (
            (candidates[0] if candidates else None)
            if first
            else backend.best(candidates)
        )
        return (
            PrincipalDecision(stage, requirement, prepared=item)
            if item is not None
            else None
        )

    def _carbon_radical_method(self, b, stage):
        centers=tuple(a.GetIdx() for a in iter_atoms(b.mol)
                      if a.GetNumRadicalElectrons())
        if not centers:return None
        from .radical_ions import carbon_radical_ion_decision
        decision=carbon_radical_ion_decision(b.mol,b.Chem) if any(
            a.GetFormalCharge() for a in iter_atoms(b.mol)) else None
        if decision is None and any(a.GetFormalCharge() for a in iter_atoms(b.mol)):
            from .cyclic_radical_ions import carbon_chain_or_cycle_radical_ion_decision
            decision=carbon_chain_or_cycle_radical_ion_decision(b.mol,b.Chem)
        if decision is None and any(a.GetFormalCharge() for a in iter_atoms(b.mol)):
            from .radical_ions import characteristic_radical_ion_decision
            decision=characteristic_radical_ion_decision(b.mol,b.Chem)
        if decision is None:
            from .carbon_radical_methods import carbon_radical_decision
            decision=carbon_radical_decision(b.mol,b.Chem)
        if decision is None:
            from .oxy_radical_methods import oxy_radical_decision
            decision=oxy_radical_decision(b.mol,b.Chem)
        if decision is None:
            from .hetero_radical_methods import hetero_radical_decision
            decision=hetero_radical_decision(b.mol,b.Chem)
        if decision is None:
            # P-70 gives radicals their own principal class. A closed-shell
            # fallback cannot represent an unsupported radical graph.
            return PrincipalDecision(stage,PrincipalRequirement(
                'unsupported-radical',centers),
                direct_route='unsupported-radical:P-70/P-71')
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule]+=1
        return PrincipalDecision(stage,PrincipalRequirement(
            'radical',tuple(sorted(decision.principal_centers)),decision.method),
            output_name=decision.name,direct_route='radical_method:'+decision.rule,
            atom_locants=decision.atom_locants,comparison=decision,
            preference_issues=getattr(decision,'preference_issues',()))

    def _amine_cation_component(self, b, stage):
        if not any(a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1 for a in iter_atoms(b.mol)):
            return None
        from .amine_cation_components import amine_cation_component
        component = amine_cation_component(b.mol, b.Chem)
        if component is None:
            from .bluebook_lambda_amine_methods import aryl_aminium_decision
            component = aryl_aminium_decision(b.mol, b.Chem)
        if component is None or sum(a.GetFormalCharge() for a in iter_atoms(b.mol)) != 1:
            return None
        heavy = frozenset(a.GetIdx() for a in iter_atoms(b.mol) if a.GetAtomicNum() > 1)
        if component.atoms != heavy:
            return None
        centers = tuple(a.GetIdx() for a in iter_atoms(b.mol) if a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1)
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[component.rule] += 1
        return PrincipalDecision(stage, PrincipalRequirement('amine-cation', centers, 'aminium'),
            output_name=component.name, direct_route='amine_cation:' + component.rule,
            atom_locants=component.atom_locants, comparison=component)

    def _simple_amine_oxide_component(self, b, stage):
        if not any(a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1 for a in iter_atoms(b.mol)):
            return None
        from .bluebook_lambda_amine_methods import simple_amine_oxide_decision
        decision = simple_amine_oxide_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(stage, PrincipalRequirement('amine-oxide',
            tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, direct_route='amine_oxide:' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision)

    def _carbon_cation_component(self, b, stage):
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not any(number == 6 and charge > 0
                for number, charge in zip(view.atomic_nums, view.formal_charges)):
            return None
        if not any(a.GetAtomicNum() == 6 and a.GetFormalCharge() > 0
                   for a in iter_atoms(b.mol)):
            return None
        from .carbon_cation_components import carbon_cation_component

        component = carbon_cation_component(b.mol, b.Chem)
        if component is None:
            return None
        heavy = frozenset(a.GetIdx() for a in iter_atoms(b.mol)
                          if a.GetAtomicNum() > 1)
        if (component.atoms != heavy or component.charge != 1
                or sum(a.GetFormalCharge() for a in iter_atoms(b.mol)) != 1):
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[component.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('carbon-cation',
                tuple(a.GetIdx() for a in iter_atoms(b.mol) if a.GetFormalCharge() > 0),
                'ylium'), output_name=component.name,
            direct_route='carbon_cation_component:' + component.rule,
            atom_locants=component.atom_locants, comparison=component)

    def _chalcogen_cation_component(self, b, stage):
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not any(number in {8, 16, 34, 52} and charge > 0
                for number, charge in zip(view.atomic_nums, view.formal_charges)):
            return None
        if not any(a.GetAtomicNum() in {8, 16, 34, 52} and a.GetFormalCharge() > 0
                   for a in iter_atoms(b.mol)):
            return None
        from .chalcogen_cation_components import chalcogen_cation_decision

        decision = chalcogen_cation_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('chalcogen-cation',
                tuple(sorted(decision.principal_centers)), 'ium'),
            output_name=decision.name,
            direct_route='chalcogen_cation_component:' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision)

    def _carbon_anion_component(self, b, stage):
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not any(number == 6 and charge < 0
                for number, charge in zip(view.atomic_nums, view.formal_charges)):
            return None
        if not any(a.GetAtomicNum() == 6 and a.GetFormalCharge() < 0
                   for a in iter_atoms(b.mol)):
            return None
        from .carbanion_components import carbanion_component

        component = carbanion_component(b.mol, b.Chem)
        if component is None:
            return None
        heavy = frozenset(a.GetIdx() for a in iter_atoms(b.mol)
                          if a.GetAtomicNum() > 1)
        centers = tuple(a.GetIdx() for a in iter_atoms(b.mol)
                        if a.GetFormalCharge() < 0)
        role_sets = [atoms for _, atoms, _ in component.component_roles]
        if (component.atoms != heavy or component.charge != -1
                or component.charge != sum(a.GetFormalCharge() for a in iter_atoms(b.mol))
                or not role_sets or frozenset().union(*role_sets) != heavy
                or sum(map(len, role_sets)) != len(heavy)):
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[component.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('carbon-anion', centers, 'ide'),
            output_name=component.name,
            direct_route='carbon_anion_component:' + component.rule,
            atom_locants=component.atom_locants, comparison=component)

    def _nitrogen_anion_component(self, b, stage):
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not any(number == 7 and charge < 0
                for number, charge in zip(view.atomic_nums, view.formal_charges)):
            return None
        if not any(a.GetAtomicNum() == 7 and a.GetFormalCharge() < 0
                   for a in iter_atoms(b.mol)):
            return None
        from .nitrogen_anion_components import (carbon_aminide_component,
            nitrogen_aminide_component, nitrogen_iminide_component,
            requires_nitrogen_anion_proof)

        component = nitrogen_aminide_component(b.mol, b.Chem)
        if component is None:
            component = carbon_aminide_component(b.mol, b.Chem)
        if component is None:
            component = nitrogen_iminide_component(b.mol, b.Chem)
        if component is None:
            if not requires_nitrogen_anion_proof(b.mol):
                return None
            return PrincipalDecision(
                stage, PrincipalRequirement('amine-derived-anion'),
                direct_route='unsupported:amine-derived-anion',
                preference_issues=(SelectionLimitation(stage, 'P-72.2.2.2.3',
                    'No complete parent/suffix and annotation proof for this multiply charged nitrogen; '
                    'monoanion and protonation-changing fallbacks are excluded.'),))
        heavy = frozenset(a.GetIdx() for a in iter_atoms(b.mol)
                          if a.GetAtomicNum() > 1)
        centers = component.anionic_centers or (component.anionic_center,)
        charged = {a.GetIdx() for a in iter_atoms(b.mol) if a.GetFormalCharge()}
        if (component.atoms != heavy or set(centers) != charged
                or component.charge != sum(a.GetFormalCharge() for a in iter_atoms(b.mol))
                or any(b.mol.GetAtomWithIdx(i).GetAtomicNum() != 7
                       or b.mol.GetAtomWithIdx(i).GetFormalCharge() >= 0 for i in centers)):
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[component.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('amine-derived-anion',
                centers, 'iminide' if component.name.endswith('iminide') else component.suffix),
            output_name=component.name,
            direct_route='nitrogen_anion_component:' + component.rule,
            atom_locants=component.atom_locants, comparison=component,
            preference_issues=component.preference_issues)

    def _acid_oxygen_isotope_method(self, b, stage):
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not any(view.isotopes):
            return None
        if not any(a.GetIsotope() for a in iter_atoms(b.mol)):
            return None
        from .acid_oxygen_isotope_frontier import acid_oxygen_isotope_frontier
        from .acid_oxygen_isotope_methods import recognize_linear_monoacid
        from .component_mapping import decorate_component

        frontier = acid_oxygen_isotope_frontier(b.mol, b.Chem)
        if frontier.disposition == 'outside-scope':
            return None
        names = ({decorate_component(b.mol, b.Chem, construction)
                  for construction in frontier.constructions}
                 if frontier.disposition == 'complete' else set())
        if len(names) != 1 or '' in names:
            return PrincipalDecision(
                stage, PrincipalRequirement('unresolved-acid-oxygen-isotope'),
                direct_route='unsupported-acid-oxygen-isotope:P-82.2/P-82.6.4',
                comparison=frontier)
        group = recognize_linear_monoacid(b.mol, b.Chem)
        rule = 'P-82.2.4/P-82.6.4/typed-acid-isotope-roles'
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('acid-oxygen-isotope', (group.center,), 'oic acid'),
            output_name=next(iter(names)), direct_route='acid_oxygen_isotope:' + rule,
            atom_locants=frontier.constructions[0].atom_locants,
            comparison=frontier)

    def _cyclic_dianhydride_method(self, b, stage):
        view = mol_cache_get(b.mol, 'graph_view')
        # Four carbonyl oxygens and two distinct bridging oxygens are
        # necessary for the existing graph witness. Missing cached facts
        # retain the original recognizer without earlier ring perception.
        if view is not None and view.elements['O'] < 6:
            return None
        from .cyclic_dianhydride_methods import cyclic_dianhydride_decision_frontier

        frontier = cyclic_dianhydride_decision_frontier(b.mol, b.Chem)
        if frontier.status == 'outside_scope':
            return None
        decision = frontier.decision
        if frontier.status != 'complete' or decision is None:
            # P-65.7.7.2 prefers the heterocyclic pseudoketone construction.
            # Incomplete P-25/P-58 evidence cannot establish another PIN.
            return PrincipalDecision(
                stage, PrincipalRequirement('unresolved-cyclic-dianhydride'),
                direct_route='unsupported-cyclic-dianhydride:P-65.7.7.2',
                comparison=frontier)
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('cyclic-dianhydride',
                tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name,
            direct_route='cyclic_dianhydride:' + decision.rule,
            atom_locants=decision.atom_locants, comparison=frontier)

    def _sulfur_ester_method(self, b, stage):
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not view.elements['S']:
            return None
        from .sulfur_ester_methods import (
            SulfurEsterFrontier, sulfur_ester_frontier, sulfur_polyol_ester_decision)

        frontier = sulfur_ester_frontier(b.mol, b.Chem)
        if not frontier.source_scope_proved:
            polyol = sulfur_polyol_ester_decision(b.mol, b.Chem)
            if polyol is None:
                return None
            # One polyhydroxylic component with repeated identical sulfur
            # anions has the same P-65 functional-class assembly requirement.
            # Component-specific tied maps remain on the typed decision;
            # they must not be flattened into a fictitious global numbering.
            frontier = SulfurEsterFrontier(
                'complete', True, polyol, (), polyol.rule,
                'complete mapped polyol with identical sulfur-acid anions')
        decision = frontier.decision
        if frontier.disposition != 'complete' or decision is None:
            # P-65.6.3.2 requires a class-specific ester construction, or
            # its prefix construction when a more senior group is present.
            # An unavailable construction cannot establish a lower method.
            return PrincipalDecision(
                stage, PrincipalRequirement('unresolved-sulfur-ester'),
                direct_route='unsupported-sulfur-ester:' + frontier.rule,
                comparison=frontier,
                # A proved neutral ester outside this preferred writer's arm
                # grammar may still receive a complete general construction.
                # Invalid valence/ownership and unsupported annotations do
                # not acquire this permission; preference remains unresolved.
                preference_fallback=frontier.general_construction_permitted)
        heavy = frozenset(a.GetIdx() for a in iter_atoms(b.mol)
                          if a.GetAtomicNum() > 1)
        roles = [frozenset(atoms) for _, atoms, _ in decision.component_roles]
        if (decision.atoms != heavy or not roles
                or frozenset().union(*roles) != heavy
                or sum(map(len, roles)) != len(heavy)):
            return PrincipalDecision(
                stage, PrincipalRequirement('unresolved-sulfur-ester'),
                direct_route='unsupported-sulfur-ester:ownership',
                comparison=frontier)
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('sulfur-acid-ester',
                tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name,
            direct_route='sulfur_ester:' + decision.rule,
            atom_locants=decision.atom_locants, comparison=frontier)

    def _partial_polyol_ester_method(self, b, stage):
        inventory = mol_cache_get(b.mol, 'selection_inventory')
        if inventory is not None and sum(group.kind == 'ester' for group in inventory.groups) < 2:
            return None
        from .naming_methods import NamingMethodStatus
        from .partial_polyol_ester_methods import partial_polyol_ester_frontier

        outcome = partial_polyol_ester_frontier(b.mol, b.Chem)
        if outcome.status == NamingMethodStatus.OUTSIDE_SCOPE:
            return None
        decision = outcome.decision
        if outcome.status != NamingMethodStatus.COMPLETE or decision is None:
            return PrincipalDecision(
                stage, PrincipalRequirement('unresolved-partial-polyol-ester'),
                direct_route='unsupported-partial-polyol-ester:' + outcome.rule,
                comparison=outcome)
        heavy = frozenset(a.GetIdx() for a in iter_atoms(b.mol)
                          if a.GetAtomicNum() > 1)
        roles = [frozenset(atoms) for _, atoms, _ in decision.component_roles]
        if (decision.atoms != heavy or not roles
                or frozenset().union(*roles) != heavy
                or sum(map(len, roles)) != len(heavy)):
            return PrincipalDecision(
                stage, PrincipalRequirement('unresolved-partial-polyol-ester'),
                direct_route='unsupported-partial-polyol-ester:ownership',
                comparison=outcome)
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('partial-polyol-ester',
                tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name,
            direct_route='partial_polyol_ester:' + decision.rule,
            atom_locants=decision.atom_locants, comparison=outcome,
            preference_issues=decision.preference_issues)

    def _native_ocene_parent_method(self,b,stage):
        from .coordination import native_ocene_decision
        decision=native_ocene_decision(b.mol,b.Chem)
        if decision is None:return None
        if b.ctx.telemetry is not None:b.ctx.telemetry.naming_method_selected_rule[decision.rule]+=1
        return PrincipalDecision(stage,PrincipalRequirement('retained-native-ocene'),output_name=decision.name,
            direct_route='native_ocene:'+decision.rule,atom_locants=decision.atom_locants,comparison=decision)

    def _fullerene_parent_method(self,b,stage):
        from .fullerenes import fullerene_parent_decision
        decision=fullerene_parent_decision(b.mol,b.Chem)
        if decision is None:return None
        if b.ctx.telemetry is not None:b.ctx.telemetry.naming_method_selected_rule[decision.rule]+=1
        return PrincipalDecision(stage,PrincipalRequirement('numbered-fullerene-parent'),
            output_name=decision.name,direct_route='fullerene_parent:'+decision.rule,
            atom_locants=decision.atom_locants,comparison=decision)

    def _retained_hydrocarbon_parent_method(self,b,stage):
        from .retained_hydrocarbon_parents import retained_hydrocarbon_parent_decision
        decision=retained_hydrocarbon_parent_decision(b.mol,b.Chem)
        if decision is None:return None
        if b.ctx.telemetry is not None:b.ctx.telemetry.naming_method_selected_rule[decision.rule]+=1
        return PrincipalDecision(stage,PrincipalRequirement('complete-retained-hydrocarbon-parent'),
            output_name=decision.name,direct_route='retained_hydrocarbon:'+decision.rule,comparison=decision)

    def _retained_hydrocarbon_derivative_method(self,b,stage):
        from .retained_hydrocarbon_derivatives import retained_hydrocarbon_derivative_decision
        decision=retained_hydrocarbon_derivative_decision(b.mol,b.Chem)
        if decision is None:return None
        if b.ctx.telemetry is not None:b.ctx.telemetry.naming_method_selected_rule[decision.rule]+=1
        return PrincipalDecision(stage,PrincipalRequirement('retained-hydrocarbon-derivative'),
            output_name=decision.name,direct_route='retained_hydrocarbon_derivative:'+decision.rule,
            atom_locants=decision.atom_locants,comparison=decision,preference_issues=decision.preference_issues)

    def _linear_phane_parent_method(self, b, stage):
        from .linear_phane_methods import linear_phane_frontier

        frontier = linear_phane_frontier(b.mol, b.Chem)
        if not frontier.mandatory_eligible:
            return None
        decision = frontier.decision
        if frontier.disposition != 'complete' or decision is None:
            # P-52.2.5.3 requires the proven phane method. A renderer limit
            # cannot make a substitutive or multiplicative name preferred.
            return PrincipalDecision(
                stage, PrincipalRequirement('unresolved-linear-phane'),
                direct_route='unsupported-linear-phane:' + frontier.source_rule,
                comparison=frontier, preference_fallback=True)
        heavy = frozenset(a.GetIdx() for a in iter_atoms(b.mol)
                          if a.GetAtomicNum() > 1)
        roles = [frozenset(atoms) for _, atoms, _ in decision.component_roles]
        if (decision.atoms != heavy or not roles
                or frozenset().union(*roles) != heavy
                or sum(map(len, roles)) != len(heavy)
                or any(frozenset(dict(mapping)) != heavy
                       for mapping in decision.equivalent_atom_locant_maps)):
            return PrincipalDecision(
                stage, PrincipalRequirement('unresolved-linear-phane'),
                direct_route='unsupported-linear-phane:ownership',
                comparison=frontier)
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('linear-phane-parent'),
            output_name=decision.name,
            direct_route='linear_phane_parent:' + decision.rule,
            atom_locants=decision.atom_locants, comparison=frontier)

    def _symmetric_sulfone_ether_method(self, b, stage):
        from .symmetric_sulfone_multiplication import symmetric_sulfone_ether_decision

        decision = symmetric_sulfone_ether_decision(b.mol, b.Chem)
        if decision is None:
            view = graph_view(b.mol, b.Chem)
            if view.ring_count >= 2 and (view.elements['S'] >= 2 or view.elements['N']):
                counts = method_bond_counts(b.mol, b.Chem)
                parents = []
                if counts[16, 16, 1]:
                    parents.append('disulfane')
                if counts[7, 7, 2]:
                    parents.append('diazene')
                if view.elements['N'] and counts[6, 7, 1] >= 2:
                    parents.append('amine')
                if parents:
                    from .equivalent_hydride_arms import equivalent_hydride_arms

                    decision = equivalent_hydride_arms(b.mol, b.Chem, parents=parents)
                    if decision is not None:
                        return PrincipalDecision(
                            stage, PrincipalRequirement('equivalent-hydride-arms',
                                tuple(sorted(decision.principal_centers))),
                            output_name=decision.name, direct_route='general:equivalent-hydride-arms',
                            comparison=decision, preference_issues=decision.preference_issues)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('multiplicative-sulfone-ether'),
            output_name=decision.name,
            direct_route='multiplicative_sulfone_ether:' + decision.rule,
            comparison=decision, preference_issues=decision.preference_issues)

    def _naming_method(self, b, stage):
        from .naming_methods import NamingMethodStatus, select_naming_method_outcome
        from .extended_inventory import selection_inventory

        evidence=[] if b.ctx.telemetry is not None else None
        outcome = select_naming_method_outcome(b.mol, b.Chem, trace=evidence)
        if evidence is not None:
            for row in evidence:
                b.ctx.telemetry.naming_method_evidence[
                    row['rule']+':'+row['method']+':'+str(row.get('eligible'))] += 1
        if outcome.status == NamingMethodStatus.PENDING:
            # P-51 eligibility and preference survive renderer failure. A
            # missing spelling cannot establish a lower-preference method.
            pending = PrincipalDecision(
                stage, PrincipalRequirement('unresolved-required-method', (), outcome.method),
                direct_route='unsupported-naming-method:' + outcome.rule,
                comparison=outcome, preference_fallback=True)
            if outcome.decision is not None and outcome.decision.name:
                usable = outcome.decision
                return replace(pending, output_name=usable.name,
                    requirement=PrincipalRequirement('unresolved-method-preference',
                        tuple(sorted(usable.principal_centers)), usable.method),
                    atom_locants=usable.atom_locants,
                    preference_issues=self._preference_issues(pending))
            return pending
        decision = outcome.decision
        if outcome.status == NamingMethodStatus.COMPLETE and decision is not None and decision.name:
            if b.ctx.telemetry is not None:
                b.ctx.telemetry.naming_method_selected_rule[decision.rule]+=1
            from .selection_evidence import functional_justification
            justification = decision.justification or functional_justification(
                b.mol, selection_inventory(b.mol).principal,
                decision.functional_parent_frontier,
                method_rule=decision.rule, method_proved=True)
            required = {group.center for group in selection_inventory(b.mol).principal}
            return PrincipalDecision(
                stage,
                PrincipalRequirement("maximum-principal-coverage" if required - decision.principal_centers
                                     else "complete-principal-coverage",
                                     tuple(sorted(decision.principal_centers)), decision.method),
                output_name=decision.name,
                direct_route="naming_method:" + decision.method,
                atom_locants=decision.atom_locants,comparison=outcome,
                justification=justification,
            )
        return None

    def _retained_hydroxy_anion_component(self, b, stage):
        from .retained_hydroxy_anion_components import retained_hydroxy_anion_component

        outcome = retained_hydroxy_anion_component(b.mol, b.Chem)
        component = outcome.component
        if outcome.status != 'complete' or component is None:
            # This is a retained-vocabulary decision. An isotope profile
            # outside that vocabulary may still have a systematic writer.
            return None
        heavy = frozenset(atom.GetIdx() for atom in iter_atoms(b.mol)
                          if atom.GetAtomicNum() > 1)
        if (not component.full_coverage or component.atoms != heavy
                or component.charge != -1
                or sum(atom.GetFormalCharge() for atom in iter_atoms(b.mol)) != -1
                or b.mol.GetAtomWithIdx(component.anionic_oxygen).GetFormalCharge() != -1):
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[component.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('hydroxy-anion',
                (component.anionic_oxygen,), 'retained hydroxy-anion'),
            output_name=component.name,
            direct_route='retained_hydroxy_anion:' + component.rule,
            atom_locants=component.atom_locants, comparison=component)

    def _polyol_anion_component(self, b, stage):
        # P-72/P-77 retain each alcoholate center as an anionic suffix;
        # remaining neutral hydroxy sites are subordinate prefixes.
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not any(number == 8 and charge == -1
                for number, charge in zip(view.atomic_nums, view.formal_charges)):
            return None
        if not any(a.GetAtomicNum() == 8 and a.GetFormalCharge() == -1
                   for a in iter_atoms(b.mol)):
            return None
        from .polyol_anion_components import polyol_anion_component
        component = polyol_anion_component(b.mol, b.Chem)
        if component is None:
            from .polyolate_anion_components import polyolate_anion_component
            component = polyolate_anion_component(b.mol, b.Chem)
        if component is None:return None
        anionic_oxygens = ((component.anionic_oxygen,)
            if hasattr(component, 'anionic_oxygen') else component.anionic_oxygens)
        heavy = frozenset(a.GetIdx() for a in iter_atoms(b.mol)
                          if a.GetAtomicNum() > 1)
        if (component.charge != -len(anionic_oxygens)
                or component.charge != sum(a.GetFormalCharge() for a in iter_atoms(b.mol))
                or component.atoms | component.embedded_atoms != heavy
                or component.atoms & component.embedded_atoms):
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[component.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('polyol-monoanion' if len(anionic_oxygens)==1
                else 'polyol-polyanion', tuple(anionic_oxygens), 'olate'),
            output_name=component.name,
            direct_route='polyol_anion_component:' + component.rule,
            atom_locants=component.atom_locants, comparison=component)

    def _isotope_characteristic_group_method(self, b, stage):
        # P-82.2.2.2 separates differently modified occurrences of the same
        # class before an ordinary multiplicative suffix can be constructed.
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not any(view.isotopes):
            return None
        if not any(a.GetIsotope() for a in iter_atoms(b.mol)):
            return None
        from .amide_isotopes import amide_isotope_outcome
        from .naming_methods import NamingMethodStatus
        amide = amide_isotope_outcome(b.mol, b.Chem)
        if amide.status != NamingMethodStatus.OUTSIDE_SCOPE:
            decision = amide.decision
            pending = amide.status == NamingMethodStatus.PENDING
            result = PrincipalDecision(
                stage, PrincipalRequirement('unresolved-isotope-amide' if pending else 'isotope-modified-amide',
                    tuple(sorted(decision.principal_centers)) if decision is not None else (), 'amide'),
                output_name=decision.name if decision is not None else '',
                direct_route='isotope_characteristic_group:' + amide.rule,
                atom_locants=decision.atom_locants if decision is not None else (), comparison=amide,
                preference_fallback=pending)
            if pending:
                return replace(result, preference_issues=self._preference_issues(result))
            if b.ctx.telemetry is not None:
                b.ctx.telemetry.naming_method_selected_rule[amide.rule] += 1
            return result
        from .isotope_diacid_methods import isotope_diacid_decision
        decision = isotope_diacid_decision(b.mol, b.Chem)
        kind = 'isotope-modified-acid'
        if decision is None:
            from .isotope_diol_methods import isotope_diol_outcome
            from .naming_methods import NamingMethodStatus
            outcome = isotope_diol_outcome(b.mol, b.Chem)
            if outcome.status == NamingMethodStatus.PENDING:
                return PrincipalDecision(
                    stage, PrincipalRequirement('unresolved-isotope-characteristic-group'),
                    direct_route='unsupported-isotope-characteristic-group:' + outcome.rule,
                    comparison=outcome)
            decision = outcome.decision
            if outcome.status != NamingMethodStatus.COMPLETE or decision is None:
                return None
            kind = 'isotope-modified-alcohol'
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement(kind,
                tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name,
            direct_route='isotope_characteristic_group:' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision)

    @staticmethod
    def _join_preferred_evidence(b, decision):
        """Join an earlier class/annotation writer to whole-parent evidence."""
        from .cross_route_selection import cross_route_selection_outcome
        from .selection_evidence import parent_justification
        from .extended_inventory import selection_inventory
        inventory = selection_inventory(b.mol)
        if (not inventory.supported or not inventory.principal
                or any(group.kind not in {'acid', 'amide', 'nitrile', 'ketone',
                       'alcohol', 'thiol', 'amine'} for group in inventory.principal)):
            return decision
        outcome = cross_route_selection_outcome(b.mol, b.Chem)
        selected = getattr(outcome.selection, 'selected', None)
        output = decision.output_name or (
            decision.prepared.output_name or decision.prepared.name)
        if selected is not None and selected.name == output:
            return replace(decision, justification=outcome.justification,
                           parent_selection=selected.parent_selection)
        general = outcome.general_proposal
        if general is not None and general.name == output:
            return replace(decision, justification=outcome.justification)
        return replace(decision, justification=parent_justification(
            None, (), None, method_rule='P-41/P-51/P-82.2',
            reason='The class or annotation construction has not been joined to a complete preferred parent/method/numbering universe.'))

    def _amine_imine_oxide_method(self, b, stage):
        # P-62.5 places these oxides with zwitterions in the class order.
        # Their original charged graph must establish eligibility before a
        # neutral characteristic-group inventory can select a lower class.
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not any(number == 7 and charge == 1
                for number, charge in zip(view.atomic_nums, view.formal_charges)):
            return None
        if not any(a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1
                   for a in iter_atoms(b.mol)):
            return None
        from .amine_oxide_methods import amine_oxide_decision
        from .functional_amine_oxide_methods import functional_amine_oxide_decision
        from .imine_oxide_methods import imine_oxide_decision

        for kind, producer in (("amine-oxide", amine_oxide_decision),
                               ("amine-oxide", functional_amine_oxide_decision),
                               ("imine-oxide", imine_oxide_decision)):
            decision = producer(b.mol, b.Chem)
            if decision is None:
                continue
            if b.ctx.telemetry is not None:
                b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
            return PrincipalDecision(
                stage, PrincipalRequirement(kind,
                    tuple(sorted(decision.principal_centers)), decision.method),
                output_name=decision.name,
                direct_route='oxide_method:' + decision.rule,
                atom_locants=decision.atom_locants, comparison=decision)
        return None

    def _hidden_amide_pseudoketone_method(self, b, stage):
        from .hidden_amide_pseudoketones import hidden_amide_pseudoketone_decision

        decision = hidden_amide_pseudoketone_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('pseudoketone',
                tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name,
            direct_route='hidden_amide_pseudoketone:' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision,
            preference_issues=decision.preference_issues)

    def _chalcogen_chain_pseudoketone_method(self, b, stage):
        # P-68.4.1.3/P-65.7.5.1 preserve an acyl-terminated homogeneous
        # chalcogen path as a pseudoketone. In particular an O-chain must not
        # first be fragmented by ordinary monoester construction.
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and max(view.elements[e] for e in ('O', 'S', 'Se', 'Te')) < 3:
            return None
        from .chalcogen_chain_pseudoketones import chalcogen_chain_pseudoketone_decision
        decision = chalcogen_chain_pseudoketone_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('pseudoketone',
                tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name,
            direct_route='pseudoketone_method:' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision)

    def _cyclic_sultam_heterone_method(self, b, stage):
        from .cyclic_sultam_heterones import cyclic_sultam_heterone_decision

        decision = cyclic_sultam_heterone_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('ketone',
                tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name,
            direct_route='cyclic_sultam_heterone:' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision,
            preference_issues=decision.preference_issues)

    def _acid_derivative_method(self,b,stage):
        # Necessary element conditions of the existing producers. These only
        # avoid impossible recognizers; their order and admission stay intact.
        view = graph_view(b.mol, b.Chem)
        elements = view.elements
        nitrogen, oxygen = elements['N'], elements['O']
        chalcogen = oxygen + elements['S'] + elements['Se'] + elements['Te']
        bonds = method_bond_counts(b.mol, b.Chem)
        carbonyl = bonds[6, 8, 2]
        carbon_chalcogen = sum(bonds[6, number, 2] for number in (8, 16, 34, 52))
        imino = bonds[6, 7, 2]
        nitrogen_pair = any(a == c == 7 for a, c, _order in bonds)
        nitrogen_oxygen = any((a, c) == (7, 8) for a, c, _order in bonds)
        decision = None
        inventory = mol_cache_get(b.mol, 'selection_inventory')
        kinds = (None if inventory is None else
                 Counter(group.kind for group in inventory.groups))
        derivative_possible = (inventory is None or any(
            group.kind in {'ester', 'acid-halide', 'anhydride'}
            for group in inventory.principal))
        if view.ring_count >= 3 and nitrogen >= 2 and oxygen >= 2 and carbonyl:
            from .three_face_heterolactam_fusion import three_face_lactam_decision

            decision = three_face_lactam_decision(b.mol, b.Chem)
        if elements['S'] and oxygen >= 2 and any(
                bonds[min(16, halogen), max(16, halogen), 1]
                for halogen in (9, 17, 35, 53)):
            from .bounded_sulfonyl_halides import benzene_sulfonyl_halide_decision, mapped_sulfonyl_halide_decision

            decision = (benzene_sulfonyl_halide_decision(b.mol, b.Chem)
                        or mapped_sulfonyl_halide_decision(b.mol, b.Chem))
        if (decision is None and view.ring_count >= 2 and oxygen >= 2 and carbonyl
                and (kinds is None or kinds['ester'] == 1)):
            from .bicyclic_carboxylate_methods import bicyclic_carboxylate_decision
            decision = bicyclic_carboxylate_decision(b.mol, b.Chem)
        if decision is None and chalcogen and derivative_possible:
            from .acid_derivative_methods import acid_derivative_decision
            decision=acid_derivative_decision(b.mol,b.Chem)
        if decision is None and (any(elements[e] for e in ('Si', 'Ge', 'Sn', 'Pb'))):
            # P-65.6.3.4.1 explicitly requires the ester method for the
            # admitted Group 14 pseudoesters before generic replacement.
            from .pseudoester_methods import pseudoester_decision
            decision=pseudoester_decision(b.mol,b.Chem)
        if decision is None and (any(elements[e] for e in ('Al', 'Ga', 'In', 'Tl'))):
            from .trivalent_pseudoester_methods import trivalent_pseudoester_decision
            decision=trivalent_pseudoester_decision(b.mol,b.Chem)
        if decision is None and (elements['B']):
            # The retained borinic acid parent establishes an anhydride
            # class under P-65.7.2, ahead of the P-65.6 pseudoester method.
            from .borinic_anhydride_methods import borinic_anhydride_decision
            decision=borinic_anhydride_decision(b.mol,b.Chem)
        if decision is None and (oxygen and (kinds is None or kinds['acid'] and kinds['ester'])):
            from .partial_acid_ester_methods import partial_acid_ester_decision
            decision=partial_acid_ester_decision(b.mol,b.Chem)
        if decision is None and (oxygen >= 3 and carbonyl):
            from .carbonate_ester_methods import carbonate_ester_decision
            decision=carbonate_ester_decision(b.mol,b.Chem)
        if decision is None and (nitrogen and oxygen >= 2 and carbonyl):
            from .carbamate_methods import carbamate_decision
            decision=carbamate_decision(b.mol,b.Chem,allow_haloalkyl=True)
        if decision is None and (nitrogen >= 2 and carbon_chalcogen):
            from .urea_methods import urea_decision
            decision=urea_decision(b.mol,b.Chem)
        if decision is None and (nitrogen >= 3 and nitrogen_pair and carbon_chalcogen):
            from .hydrazine_carboxamide_methods import hydrazine_carboxamide_decision
            decision=hydrazine_carboxamide_decision(b.mol,b.Chem)
        if decision is None and (nitrogen >= 2 and carbonyl and view.ring_count):
            # P-66.1.1.1.1.3 puts the carboxamide suffix on a heterocyclic
            # parent N. The acyclic urea producer cannot own that ring arm.
            from .annular_carboxamide_methods import annular_carboxamide_decision
            decision=annular_carboxamide_decision(b.mol,b.Chem)
        if decision is None and nitrogen >= 3 and carbonyl and view.ring_count:
            from .aromatic_n_carboxamide_methods import aromatic_n_carboxamide_decision

            decision = aromatic_n_carboxamide_decision(b.mol, b.Chem)
        if decision is None and (nitrogen >= 2 and nitrogen_pair):
            # P-67.1.2.6.3 selects the nitric/nitrous amide parent. Its exact
            # graph proof admits the internal N+/O- representation of a net
            # neutral nitramide; the generic neutral P-51 gate does not.
            from .nitrous_amide_methods import nitrous_amide_decision
            decision=nitrous_amide_decision(b.mol,b.Chem)
        if decision is None and (elements['B']):
            from .boron_acid_methods import boron_acid_decision
            decision=boron_acid_decision(b.mol,b.Chem)
        if decision is None and (elements['P']):
            from .phosphorus_acid_methods import phosphorus_acid_decision
            decision=phosphorus_acid_decision(b.mol,b.Chem)
        if decision is None and (elements['P']):
            # The specific P-67.1.3.2 partial-ester rule retains separate
            # organyl and hydrogen words before the phosphorus-acid anion.
            from .phosphorus_ester_methods import phosphorus_ester_decision
            decision=phosphorus_ester_decision(b.mol,b.Chem)
        if decision is None and elements['P'] and view.ring_count:
            from .ring_methyl_phosphate_methods import ring_methyl_phosphate_decision

            decision = ring_methyl_phosphate_decision(b.mol, b.Chem)
        if decision is None and elements['P'] and nitrogen:
            from .phosphorus_amide_methods import phosphorus_amide_decision
            decision=phosphorus_amide_decision(b.mol,b.Chem)
        if decision is None and bonds[7, 16, 2]:
            from .extended_inventory import selection_inventory
            from .nitrogen_acid_derivatives import sulfonimidamide_decision

            decision = sulfonimidamide_decision(b.mol, b.Chem, selection_inventory(b.mol))
        if decision is None and bonds[6, 16, 2] and bonds[7, 7, 1]:
            from .extended_inventory import selection_inventory
            from .hydrazide_components import thiosemicarbazone_decision

            decision = thiosemicarbazone_decision(b.mol, b.Chem, selection_inventory(b.mol))
        if decision is None and (elements['Se'] or elements['Te']):
            from .chalcogen_acid_methods import chalcogen_acid_decision
            decision=chalcogen_acid_decision(b.mol,b.Chem)
        if decision is None and (nitrogen >= 3 and imino):
            from .guanidine_methods import guanidine_decision
            decision=guanidine_decision(b.mol,b.Chem)
        if decision is None and (nitrogen >= 3 and imino):
            from .condensed_guanidine_methods import condensed_guanidine_decision
            decision=condensed_guanidine_decision(b.mol,b.Chem)
        if decision is None and (nitrogen >= 3 and imino):
            from .substituted_condensed_guanidine_methods import substituted_condensed_guanidine_decision
            decision=substituted_condensed_guanidine_decision(b.mol,b.Chem)
        if decision is None and (nitrogen >= 2 and imino):
            from .amidine_methods import amidine_decision
            decision=amidine_decision(b.mol,b.Chem)
        if decision is None and nitrogen >= 2 and (imino or bonds[6, 7, 3]):
            from .extended_inventory import selection_inventory
            from .nitrogen_acid_derivatives import nitrogen_acid_derivative_decision

            decision = nitrogen_acid_derivative_decision(
                b.mol, b.Chem, selection_inventory(b.mol))
        if decision is None and (oxygen and (kinds is None or kinds['ester'] == 4)):
            from .polyester_assembly_methods import polyester_assembly_decision
            decision=polyester_assembly_decision(b.mol,b.Chem)
        if decision is None and (oxygen and (kinds is None or kinds['ester'] >= 2)):
            from .polyacid_ester_methods import polyacid_ester_decision
            decision=polyacid_ester_decision(b.mol,b.Chem)
        if decision is None and (oxygen >= 6 and carbonyl >= 4):
            from .poly_anhydride_methods import poly_anhydride_decision
            decision=poly_anhydride_decision(b.mol,b.Chem)
        if decision is None and (oxygen and carbonyl >= 2):
            from .anhydride_components import anhydride_decision
            decision=anhydride_decision(b.mol,b.Chem)
        if decision is None and (nitrogen and carbonyl):
            from .acyl_pseudohalide_methods import acyl_pseudohalide_decision
            decision=acyl_pseudohalide_decision(b.mol,b.Chem)
        if decision is None and (oxygen >= 3 and carbonyl and bonds[8, 8, 1]):
            from .peroxyacid_components import peroxyacid_decision
            decision=peroxyacid_decision(b.mol,b.Chem)
        if decision is None and (nitrogen >= 2 and imino >= 2):
            from .carbodiimide_components import carbodiimide_decision
            decision=carbodiimide_decision(b.mol,b.Chem)
        if decision is None and (nitrogen >= 2 and imino and nitrogen_pair):
            from .hydrazone_methods import hydrazone_decision
            decision=hydrazone_decision(b.mol,b.Chem)
        if decision is None and (nitrogen and oxygen and nitrogen_oxygen):
            from .hydroxylamine_methods import hydroxylamine_decision
            decision=hydroxylamine_decision(b.mol,b.Chem)
        if decision is None and (nitrogen and nitrogen_oxygen and carbonyl):
            from .hydroxamic_amide_methods import hydroxamic_amide_decision
            decision=hydroxamic_amide_decision(b.mol,b.Chem)
        if decision is None and (chalcogen and carbon_chalcogen):
            from .chalcogen_aldehyde_methods import chalcogen_aldehyde_decision
            decision=chalcogen_aldehyde_decision(b.mol,b.Chem)
        if decision is None and (chalcogen and carbon_chalcogen):
            from .chalcogen_carbonyl_methods import chalcogen_carbonyl_decision
            decision=chalcogen_carbonyl_decision(b.mol,b.Chem)
        if decision is None and (nitrogen and carbonyl >= 2):
            from .imide_methods import imide_decision
            decision=imide_decision(b.mol,b.Chem)
        if decision is None and (elements['S']):
            from .heterohost_sulfur import heterohost_sulfur_decision
            decision=heterohost_sulfur_decision(b.mol,b.Chem)
        if decision is None:return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule]+=1
        return PrincipalDecision(stage,PrincipalRequirement('acid-derivative',
            tuple(sorted(decision.principal_centers)),decision.method),output_name=decision.name,
            direct_route=decision.method.replace('-','_')+':'+decision.rule,
            atom_locants=decision.atom_locants,comparison=decision,
            preference_issues=getattr(decision, 'preference_issues', ()))

    def _functional_ring_assembly_method(self,b,stage):
        if b.mol.GetRingInfo().NumRings()<2:
            return None
        from .functional_ring_assemblies import functional_assembly_outcome
        outcome=functional_assembly_outcome(b.mol,b.Chem)
        if outcome.status=='outside-scope':
            return None
        if outcome.status!='complete' or outcome.decision is None:
            # Retain a usable labelled name from an existing class-specific
            # writer without mistaking it for the preferred assembly parent.
            usable=None
            if any(atom.GetIsotope() for atom in iter_atoms(b.mol)):
                usable=self._acid_oxygen_isotope_method(b,stage)
                if usable is None:
                    usable=self._isotope_characteristic_group_method(b,stage)
            return PrincipalDecision(stage,PrincipalRequirement('unresolved-functional-ring-assembly',
                tuple(sorted(outcome.proof.principal_centers))),
                output_name=getattr(usable,'output_name',''),
                direct_route='unsupported-functional-ring-assembly:'+outcome.rule,
                comparison=outcome,preference_fallback=True)
        decision=outcome.decision
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule]+=1
        return PrincipalDecision(stage,PrincipalRequirement('functional-ring-assembly',
            tuple(sorted(decision.principal_centers)),decision.method),
            output_name=decision.name,direct_route='functional_ring_assembly:'+decision.rule,
            atom_locants=decision.atom_locants,comparison=outcome)

    def _ring_assembly_method(self,b,stage):
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not view.ring_count:
            return None
        from .ring_assembly_methods import ring_assembly_decision
        decision=ring_assembly_decision(b.mol,b.Chem)
        if decision is None:
            from .tree_ring_assemblies import tree_ring_assembly_outcome
            outcome=tree_ring_assembly_outcome(b.mol,b.Chem)
            if outcome.status=='pending':
                return PrincipalDecision(stage,PrincipalRequirement('unresolved-tree-ring-assembly',()),
                    direct_route='unsupported-tree-ring-assembly:'+outcome.rule,
                    comparison=outcome,preference_fallback=True)
            if outcome.status=='complete':decision=outcome.decision
        if decision is None:
            from .branched_ring_assembly_methods import branched_ring_assembly_decision
            decision=branched_ring_assembly_decision(b.mol,b.Chem)
        if decision is None:
            from .heteroring_assembly_methods import heteroring_assembly_decision
            decision=heteroring_assembly_decision(b.mol,b.Chem)
        if decision is None:return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule]+=1
        return PrincipalDecision(stage,PrincipalRequirement('parent-hydride',(),decision.method),
            output_name=decision.name,direct_route='ring_method:'+decision.rule,
            atom_locants=decision.atom_locants,comparison=decision)

    def _polyspiro_parent_method(self,b,stage):
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not view.ring_count:
            return None
        from .polyspiro_methods import polyspiro_decision
        decision=polyspiro_decision(b.mol,b.Chem)
        if decision is None:
            from .heteropolyspiro_methods import heteropolyspiro_decision
            decision=heteropolyspiro_decision(b.mol,b.Chem)
        if decision is None:return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule]+=1
        return PrincipalDecision(stage,PrincipalRequirement('parent-hydride',(),decision.method),
            output_name=decision.name,direct_route='ring_method:'+decision.rule,
            atom_locants=decision.atom_locants,comparison=decision)

    def _dehydro_parent_method(self,b,stage):
        from .dehydro_parent_methods import dehydro_parent_decision
        decision=dehydro_parent_decision(b.mol,b.Chem)
        if decision is None:return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule]+=1
        return PrincipalDecision(stage,PrincipalRequirement('parent-hydride',(),decision.method),
            output_name=decision.name,direct_route='ring_method:'+decision.rule,
            atom_locants=decision.atom_locants,comparison=decision)

    def _isocyanato_hydride_method(self, b, stage):
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None:
            if (view.ring_count or view.elements['N'] != 1 or view.elements['O'] != 1
                    or view.elements['C'] < 4):
                return None
            bonds = method_bond_counts(b.mol, b.Chem)
            if not bonds[6, 7, 2] or not bonds[6, 8, 2]:
                return None
        from .isocyanato_hydride_methods import isocyanato_hydride_decision
        decision = isocyanato_hydride_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(stage, PrincipalRequirement('parent-hydride', (), decision.method),
            output_name=decision.name, direct_route='substitutive:' + decision.rule,
            atom_locants=decision.atom_locants, comparison=decision,
            preference_issues=decision.preference_issues)

    def _cumulene_parent_method(self,b,stage):
        from .graph_model import method_bond_counts
        if (mol_cache_get(b.mol, 'graph_view') is not None
                and method_bond_counts(b.mol, b.Chem)[6, 6, 2] < 2):
            return None
        from .cumulene_chain_methods import cumulene_chain_decision
        proposal=cumulene_chain_decision(b.mol,b.Chem)
        if proposal is None:return None
        decision=proposal.decision
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule]+=1
        return PrincipalDecision(stage,PrincipalRequirement(
            'alcohol' if decision.principal_centers else 'parent-hydride',
            tuple(sorted(decision.principal_centers)),decision.method),
            output_name=decision.name,direct_route='substitutive:'+decision.rule,
            atom_locants=decision.atom_locants,comparison=proposal)

    def _ring_construction(self,b,stage):
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not view.ring_count:
            return None
        if (view is not None and view.ring_count == 3 and view.elements['S']
                and view.elements['N'] >= 2 and view.elements['O'] >= 4
                and method_bond_counts(b.mol, b.Chem)[8, 16, 2] >= 2):
            from .spiro_component_rules import sultam_spiro_dione

            sultam = sultam_spiro_dione(b.mol, b.Chem)
            if sultam is not None:
                return PrincipalDecision(stage, PrincipalRequirement(
                    'spiro-imide', tuple(sorted(sultam.principal_centers)), sultam.method),
                    output_name=sultam.name, direct_route='ring_method:' + sultam.rule,
                    atom_locants=sultam.atom_locants, comparison=sultam,
                    preference_issues=sultam.preference_issues)
        from .ring_naming_methods import ring_naming_decision
        decision=ring_naming_decision(b.mol,b.Chem)
        if decision is None:return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule]+=1
        return PrincipalDecision(stage,PrincipalRequirement('parent-hydride',(),decision.method),
            output_name=decision.name,direct_route='ring_method:'+decision.rule,
            atom_locants=decision.atom_locants,
            comparison=decision if decision.method == 'bridged-fused' else None,
            preference_issues=decision.preference_issues)

    def _retained_fused_lactam(self, b, stage):
        if b.mol.GetRingInfo().NumRings() < 2:
            return None
        from .mapped_fused_lactam_general import retained_fused_lactam_general
        decision = retained_fused_lactam_general(b.mol, b.Chem)
        if decision is None:
            return None
        return PrincipalDecision(stage, PrincipalRequirement(
            'lactam', tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, atom_locants=decision.atom_locants,
            direct_route='general-retained-fused-lactam:P-25/P-41/P-44',
            comparison=decision, preference_fallback=True,
            preference_issues=decision.preference_issues)

    def _ring_state(self, b, stage):
        view = mol_cache_get(b.mol, 'graph_view')
        if view is not None and not view.ring_count:
            return None
        from .ring_state_parent import select_ring_state
        selection = select_ring_state(b.ctx)
        if selection is None:
            return None
        record = selection.selected
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.ring_state_methods[record.method] += 1
            b.ctx.telemetry.ring_state_rules.update(selection.criteria)
            b.ctx.telemetry.ring_state_hydrogenation[str(len(record.hydro_locants))] += 1
            b.ctx.telemetry.ring_state_accommodation[
                f'paired:{len(record.paired_suffix_locants)}/added:{len(record.added_h_locants)}'] += 1
        return PrincipalDecision(stage, PrincipalRequirement('ring-state',
            tuple(sorted(record.principal_centers))), output_name=record.name,
            direct_route='ring_state:'+record.method, atom_locants=record.atom_locants,
            comparison=selection,preference_issues=record.preference_issues)

    def _diaryl_ketone_general(self, b, stage):
        """Deliver a completely mapped ketone while preference remains open."""
        from .acyclic_diaryl_ketone import diaryl_methanone_decision
        decision = diaryl_methanone_decision(b.mol, b.Chem)
        if decision is None:
            return None
        return PrincipalDecision(stage, PrincipalRequirement(
            'ketone', tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, direct_route='general-diaryl-ketone:P-64.2.2.1',
            atom_locants=decision.atom_locants, comparison=decision,
            preference_fallback=True, preference_issues=decision.preference_issues)

    def _fused_acid(self, b, stage):
        """Deliver a source-mapped fused acid with preference still open."""
        faces = b.mol.GetRingInfo().AtomRings()
        if (len(faces) < 3 or not {5, 6, 7} <= {len(face) for face in faces}
                or sum(atom.GetAtomicNum() == 8 for atom in iter_atoms(b.mol)) < 3
                or not any(atom.GetAtomicNum() == 6 and not atom.IsInRing()
                           and sum(neighbor.GetAtomicNum() == 8 for neighbor in atom.GetNeighbors()) == 2
                           for atom in iter_atoms(b.mol))):
            return None
        from .three_face_benzoxepine_acid import three_face_fused_acid_decision
        decision = three_face_fused_acid_decision(b.mol, b.Chem)
        if decision is None:
            return None
        return PrincipalDecision(
            stage, PrincipalRequirement('carboxylic-acid',
                tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, atom_locants=decision.atom_locants,
            direct_route='general-fused-acid:P-25.3/P-65.1', comparison=decision,
            preference_fallback=True, preference_issues=decision.preference_issues)

    def _amido_acid_general(self, b, stage):
        """Deliver a fully owned acid/amide construction with preference open."""
        from .mapped_amido_acid import has_scope_motifs, mapped_amido_acid_decision
        if not has_scope_motifs(b.mol):
            return None
        decision = mapped_amido_acid_decision(b.mol, b.Chem)
        if decision is None:
            return None
        return PrincipalDecision(
            stage, PrincipalRequirement('carboxylic-acid',
                tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, atom_locants=decision.atom_locants,
            direct_route='general-amido-acid:P-41/P-65.1', comparison=decision,
            preference_fallback=True, preference_issues=decision.preference_issues)

    def _fused_imine_general(self, b, stage):
        """Deliver a numbered fused imine with global preference still open."""
        if (sum(atom.GetAtomicNum() == 7 for atom in iter_atoms(b.mol)) < 4
                or len(b.mol.GetRingInfo().AtomRings()) < 3):
            return None
        from .fused_imidazo_quinoline import fused_imine_decision
        decision = fused_imine_decision(b.mol, b.Chem)
        if decision is None:
            return None
        return PrincipalDecision(
            stage, PrincipalRequirement('imine', tuple(sorted(decision.principal_centers)),
                                        decision.method),
            output_name=decision.name, atom_locants=decision.atom_locants,
            direct_route='general-fused-imine:P-25/P-62.3', comparison=decision,
            preference_fallback=True, preference_issues=decision.preference_issues)

    def _cross_route(self, b, stage):
        from .cross_route_selection import cross_route_scope, cross_route_selection_outcome

        inventory = cross_route_scope(b.mol)
        if inventory is None:
            return None
        nitrogen_oxo_higher_class = self._nitrogen_oxo_higher_class(b)
        from .complete_functional_parent import functional_parent_frontier
        prefix_atoms = frozenset(g.center for g in inventory.groups
                                 if g.kind == 'nitrile' and g not in inventory.principal)
        structural_frontier = functional_parent_frontier(
            b.mol, b.Chem, excluded_skeletal_atoms=prefix_atoms,
            mapped_ring_prefixes=True, principal=inventory.principal,
            nitrogen_oxo_higher_class=nitrogen_oxo_higher_class)
        if structural_frontier.status in {'blocked', 'conflict'}:
            from .unresolved_mapped_amides import equivalent_mapped_amide_general
            general = equivalent_mapped_amide_general(
                b.mol, b.Chem, inventory, structural_frontier)
            if general is not None:
                return PrincipalDecision(
                    stage, PrincipalRequirement('unresolved-functional-parent',
                        tuple(sorted(g.center for g in inventory.principal))),
                    output_name=general.name, atom_locants=general.atom_locants,
                    direct_route='general-mapped-amide:P-44.1.1/P-45.3',
                    comparison=structural_frontier, preference_fallback=True,
                    preference_issues=(SelectionLimitation(
                        stage, 'P-45.3/P-51',
                        'A mapped general construction is available; parent and method preference remain unresolved.'),))
            return PrincipalDecision(
                stage, PrincipalRequirement('unresolved-functional-parent',
                    tuple(sorted(g.center for g in inventory.principal))),
                direct_route='unsupported-functional-parent:P-44.1.1/P-44.3/P-45',
                comparison=structural_frontier, preference_fallback=True)
        # The shared selector's established-ring admission explicitly excludes
        # isotope-labelled inputs. Avoid rendering that ineligible pool, while
        # keeping the instrumented path's candidate telemetry unchanged.
        view = mol_cache_get(b.mol, 'graph_view')
        prepared = ()
        if b.ctx.telemetry is not None or view is None or not any(view.isotopes):
            def prepare_compatibility():
                return tuple(item for item in b.prepare(self.groups['ring'])
                             if not b.safety_reasons(item.candidate))

            from .proposal_plans import DeferredCompatibilityCandidates
            prepared = DeferredCompatibilityCandidates.for_context(
                b.ctx, self.groups['ring'], prepare_compatibility)
        # With no class barrier (the deferred pool enforces that guard), a
        # nonempty pending typed proposal is delivered immediately below.
        # Parent-obligation joins read its explicit outcome evidence; they do
        # not consume the excluded compatibility producers' scratch caches.
        outcome = cross_route_selection_outcome(
            b.mol, b.Chem, prepared,
            nitrogen_oxo_higher_class=nitrogen_oxo_higher_class,
            deliver_annotated_general=True, deliver_owned_general=True)
        if outcome.status == 'outside-scope':
            return None
        if outcome.status == 'pending':
            proposed = outcome.general_proposal
            continuation = self._principal_general(
                b, stage, minimum=len(proposed.principal_centers) if proposed else 0)
            if continuation is not None:
                return continuation
        if (outcome.status == 'pending' and len(inventory.principal) == 3
                and all(group.kind == 'amide' for group in inventory.principal)):
            # This is a complete general construction for one ring-suffix
            # candidate, not a proof that it wins the equal-rank frontier.
            from .nested_amide_general import nested_amide_general_decision
            general = nested_amide_general_decision(b.mol, b.Chem)
            if general is not None:
                return PrincipalDecision(
                    stage, PrincipalRequirement('unresolved-shared-parent',
                        tuple(sorted(group.center for group in inventory.principal))),
                    output_name=general.name, atom_locants=general.atom_locants,
                    direct_route='general-nested-amides:P-66.1.1.4.3',
                    comparison=general, preference_fallback=True,
                    justification=outcome.justification,
                    preference_issues=general.preference_issues)
        if outcome.status == 'pending':
            general = outcome.general_proposal
            preference_issues = ()
            typed_writing = None
            if general is not None and general.principal_kind=='acid':
                from .rules45_97_methods import same_parent_aryl_writing
                typed_writing=same_parent_aryl_writing(
                    b.mol,b.Chem,general.parent_atoms,general.principal_centers)
            if (general is not None and general.principal_kind == 'acid'
                    and general.name.startswith('2-') and general.name.endswith('acetic acid')):
                # A complete mapped chain proof can close the acetic-acid
                # locant omission even when the wider parent pool is pending.
                # Only replace the same physical parent and the same rendered
                # construction; unresolved whole-name preference stays open.
                from .mapped_monoacid_chain import mapped_monoacid_chain_decision
                proof = mapped_monoacid_chain_decision(b.mol, b.Chem)
                # Equivalent enclosing marks can differ before the public
                # formatter assigns nested bracket levels.  Compare their
                # structure, while requiring every other character and the
                # source-proved parent and suffix atoms to agree.
                enclosure = str.maketrans({'[': '(', '{': '(', ']': ')', '}': ')' })
                if (proof is not None and general.principal_centers == proof.principal_centers
                        and set(general.parent_atoms) == {
                            atom for atom, _locant in proof.atom_locants
                            if b.mol.GetAtomWithIdx(atom).GetAtomicNum() == 6}
                        and general.name[2:].translate(enclosure) == proof.name.translate(enclosure)):
                    return PrincipalDecision(
                        stage, PrincipalRequirement('carboxylic-acid',
                            tuple(sorted(proof.principal_centers)), proof.method),
                        output_name=proof.name, atom_locants=proof.atom_locants,
                        direct_route='general-mapped-monoacid:P-41/P-44/P-65',
                        comparison=proof, preference_fallback=True,
                        justification=outcome.justification,
                        preference_issues=proof.preference_issues)
            if (general is not None and general.principal_kind == 'acid'
                    and general.parent_selection is not None
                    and general.parent_selection.kind == 'chain'
                    and general.parent_atoms
                    and all(b.mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                            and not b.mol.GetAtomWithIdx(i).IsInRing()
                            for i in general.parent_atoms)
                    and any(n.GetAtomicNum() == 6 and not n.IsInRing()
                            and n.GetIdx() not in general.parent_atoms
                            for i in general.parent_atoms
                            for n in b.mol.GetAtomWithIdx(i).GetNeighbors())):
                # A source-owned monoacid chain can strictly extend a pending
                # shorter carbon-chain construction (e.g. an acid carbon
                # written as substituted formic acid).  P-44 chooses the
                # larger skeletal parent after equal suffix coverage.  Do
                # not infer a comparison against a ring or heteroatom chain.
                from .mapped_monoacid_chain import mapped_monoacid_chain_decision
                longer = mapped_monoacid_chain_decision(b.mol, b.Chem)
                if longer is None and len(general.parent_atoms) == 1:
                    # The general mapped frontier can own a substituted
                    # two-carbon acid even when its branch annotations fall
                    # outside the stricter monoacid continuation.
                    from .mapped_principal_general import mapped_principal_decision

                    longer = mapped_principal_decision(b.mol, b.Chem)
                if longer is not None and general.principal_centers == longer.principal_centers:
                    chain = {i for i, _locant in longer.atom_locants
                             if b.mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
                    if (set(general.parent_atoms) < chain and all(
                            not b.mol.GetAtomWithIdx(i).IsInRing() for i in chain)):
                        route = ('general-mapped-monoacid:P-41/P-44/P-65'
                                 if longer.method == 'mapped-monoacid-chain'
                                 else 'general-mapped-principal:' + longer.rule)
                        return PrincipalDecision(
                            stage, PrincipalRequirement('carboxylic-acid',
                                tuple(sorted(longer.principal_centers)), longer.method),
                            output_name=longer.name, atom_locants=longer.atom_locants,
                            direct_route=route,
                            comparison=longer, preference_fallback=True,
                            justification=outcome.justification,
                            preference_issues=longer.preference_issues)
            if (general is not None and general.principal_kind == 'acid'
                    and general.route == 'mapped-chain-partial-functional'
                    and not b.mol.GetRingInfo().NumRings()):
                from .acyclic_chalcogen_polyacid import (
                    acyclic_chalcogen_polyacid_scope, mapped_polyacid_proposal_complete,
                    polyacid_stereochemistry_issues,
                )

                scope = acyclic_chalcogen_polyacid_scope(b.mol, b.Chem, inventory)
                if (scope is not None and mapped_polyacid_proposal_complete(
                        b.mol, b.Chem, scope, general)):
                    preference_issues = polyacid_stereochemistry_issues(b.mol, b.Chem)
            return PrincipalDecision(stage, PrincipalRequirement(
                'unresolved-shared-parent', tuple(sorted(g.center for g in inventory.principal))),
                output_name=(typed_writing.name if typed_writing is not None else
                             general.name if general is not None else ''),
                atom_locants=general.atom_locants if general is not None else (),
                prepared=general.prepared if general is not None else None,
                direct_route='unsupported-shared-parent:P-44/P-45', comparison=outcome,
                preference_fallback=True, justification=outcome.justification,
                preference_issues=preference_issues)
        selection = outcome.selection
        chosen = selection.selected
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.cross_route_candidate_routes.update(p.route for p in selection.candidates)
            b.ctx.telemetry.cross_route_comparison_rules.update(
                c.rule+':'+str(c.preference) for c in selection.comparisons)
            b.ctx.telemetry.cross_route_comparison_rules.update(
                c.rule+':unrendered-parent-admission' for c in selection.admission_comparisons)
            b.ctx.telemetry.cross_route_selected_routes[chosen.route] += 1
            if chosen.principal_kind=='amine':
                from .amine_components import RULES as amine_rules
                b.ctx.telemetry.naming_method_selected_rule[amine_rules[0]]+=1
                b.ctx.telemetry.cross_route_comparison_rules.update(amine_rules)
            from .sulfur_selection import SULFUR_RULES
            if chosen.principal_kind in SULFUR_RULES:
                for rule in SULFUR_RULES[chosen.principal_kind]:
                    b.ctx.telemetry.naming_method_selected_rule[rule]+=1
            if chosen.ring_state is not None:
                record=chosen.ring_state
                b.ctx.telemetry.ring_state_methods[record.method]+=1
                from .ring_state_parent import _CRITERIA
                from .heteropolycyclic_parents import RULES
                b.ctx.telemetry.ring_state_rules.update(
                    RULES[record.method]+_CRITERIA[1:] if record.method in RULES else _CRITERIA)
                b.ctx.telemetry.ring_state_hydrogenation[str(len(record.hydro_locants))]+=1
                b.ctx.telemetry.ring_state_accommodation[f'paired:{len(record.paired_suffix_locants)}/added:{len(record.added_h_locants)}']+=1
        return PrincipalDecision(stage, PrincipalRequirement(chosen.principal_kind,
                                 tuple(sorted(chosen.principal_centers))),
                                 prepared=chosen.prepared, output_name=chosen.name,
                                 direct_route='cross_route:'+chosen.route,
                                 parent_selection=chosen.parent_selection,
                                 atom_locants=chosen.atom_locants, comparison=selection,
                                 justification=selection.justification,
                                 preference_issues=getattr(chosen.ring_state,'preference_issues',()))

    def _monoacid_chain_general(self, b, stage):
        """Deliver a source-owned chain acid while whole-name preference is open."""
        from .mapped_monoacid_chain import mapped_monoacid_chain_decision
        decision = mapped_monoacid_chain_decision(b.mol, b.Chem)
        route = 'general-mapped-monoacid:P-41/P-44/P-65'
        if decision is None:
            view = graph_view(b.mol, b.Chem)
            if not view.ring_count and view.elements['O'] >= 5:
                from .acyclic_chalcogen_polyacid import linear_ether_polyacid_decision
                decision = linear_ether_polyacid_decision(b.mol, b.Chem)
                route = 'general-mapped-replacement-chain-polyacid:P-15.4/P-44/P-65'
        if decision is None:
            return None
        if decision.method == 'mapped-diacid-chain':
            route = 'general-mapped-diacid:P-41/P-44/P-65'
        return PrincipalDecision(
            stage, PrincipalRequirement('carboxylic-acid',
                tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, atom_locants=decision.atom_locants,
            direct_route=route, comparison=decision,
            preference_fallback=True, preference_issues=decision.preference_issues)

    def _ring_acid_general(self, b, stage):
        """Deliver a mapped ring acid while complete parent comparison is open."""
        from .mapped_ring_acid_general import mapped_ring_acid_decision
        decision = mapped_ring_acid_decision(b.mol, b.Chem)
        if decision is None:
            return None
        return PrincipalDecision(
            stage, PrincipalRequirement('carboxylic-acid',
                tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, atom_locants=decision.atom_locants,
            direct_route='general-mapped-ring-acid:P-41/P-44/P-65', comparison=decision,
            preference_fallback=True, preference_issues=decision.preference_issues)

    def _monoester_general(self, b, stage):
        """Deliver a source-owned ester while whole-name preference is open."""
        from .mapped_monoester_general import mapped_monoester_decision
        decision = mapped_monoester_decision(b.mol, b.Chem)
        if decision is None:
            from .mapped_ester_general import mapped_ester_decision

            decision = mapped_ester_decision(b.mol, b.Chem)
            if decision is not None:
                return PrincipalDecision(
                    stage, PrincipalRequirement('ester',
                        tuple(sorted(decision.principal_centers)), decision.method),
                    output_name=decision.name, atom_locants=decision.atom_locants,
                    direct_route='general-mapped-ester:' + decision.rule, comparison=decision,
                    preference_fallback=True, preference_issues=decision.preference_issues)
        if decision is None:
            return None
        return PrincipalDecision(
            stage, PrincipalRequirement('carboxylic-ester',
                tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, atom_locants=decision.atom_locants,
            direct_route='general-mapped-monoester:P-41/P-44/P-65', comparison=decision,
            preference_fallback=True, preference_issues=decision.preference_issues)

    def _principal_general(self, b, stage, *, minimum=0):
        from .mapped_principal_general import mapped_principal_decision

        decision = mapped_principal_decision(b.mol, b.Chem, minimum=minimum)
        if decision is None:
            return None
        return PrincipalDecision(
            stage, PrincipalRequirement('mapped-principal-sites',
                tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, atom_locants=decision.atom_locants,
            direct_route='general-mapped-principal:' + decision.rule,
            comparison=decision, preference_fallback=True,
            preference_issues=decision.preference_issues)

    def _ring_chain_amide(self, b, stage):
        """Deliver the mapped ring candidate while method comparison is open."""
        if not b.mol.GetRingInfo().NumRings():
            return None
        carbonyls = 0
        for atom in iter_atoms(b.mol):
            if atom.GetAtomicNum() != 6 or atom.IsInRing():
                continue
            bonds = [(neighbor.GetAtomicNum(), b.mol.GetBondBetweenAtoms(
                atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble())
                for neighbor in atom.GetNeighbors()]
            if (sum(number == 8 and order == 2 for number, order in bonds) == 1
                    and sum(number == 7 and order == 1 for number, order in bonds) == 1):
                carbonyls += 1
        if carbonyls != 2:
            return None
        from .ring_chain_amide_frontier import ring_chain_amide_frontier
        frontier = ring_chain_amide_frontier(b.mol, b.Chem)
        if frontier is None:
            return None
        chosen = frontier.ring
        centers = frontier.ring.principal_centers | frontier.chain.principal_centers
        return PrincipalDecision(
            stage, PrincipalRequirement('unresolved-shared-parent', tuple(sorted(centers))),
            output_name=chosen.name, atom_locants=chosen.atom_locants,
            direct_route='general-ring-chain-amide:P-44.1.2.2', comparison=frontier,
            preference_fallback=True, preference_issues=chosen.preference_issues)

    def _linked_ring_amides(self, b, stage):
        """Deliver a source-mapped ring amide with its unresolved comparison."""
        if b.mol.GetRingInfo().NumRings() < 4:
            return None
        carbonyls = 0
        for atom in iter_atoms(b.mol):
            if atom.GetAtomicNum() != 6 or atom.IsInRing():
                continue
            bonds = [(neighbor.GetAtomicNum(), b.mol.GetBondBetweenAtoms(
                atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble())
                for neighbor in atom.GetNeighbors()]
            if (sum(number == 8 and order == 2 for number, order in bonds) == 1
                    and sum(number == 7 and order == 1 for number, order in bonds) == 1):
                carbonyls += 1
        if carbonyls != 2:
            return None
        from .mapped_linked_ring_amides import linked_ring_amide_decision
        decision = linked_ring_amide_decision(b.mol, b.Chem)
        if decision is None:
            return None
        from .extended_inventory import selection_inventory
        groups = selection_inventory(b.mol).source_frontier.groups
        centers = tuple(sorted(group.center for group in groups))
        if len(centers) != 2 or not decision.principal_centers <= set(centers):
            return None
        return PrincipalDecision(
            stage, PrincipalRequirement('unresolved-shared-parent', centers),
            output_name=decision.name, atom_locants=decision.atom_locants,
            direct_route='general-linked-ring-amides:P-44.2.1', comparison=decision,
            preference_fallback=True, preference_issues=decision.preference_issues)

    def _n_sulfonyl_ring_amide(self, b, stage):
        """Deliver a fully owned, source-senior general amide when available."""
        if not b.mol.GetRingInfo().NumRings():
            return None
        if not any(atom.GetAtomicNum() == 16
                and sum(neighbor.GetAtomicNum() == 8 and b.mol.GetBondBetweenAtoms(
                    atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                    for neighbor in atom.GetNeighbors()) == 2
                and any(neighbor.GetAtomicNum() == 7 for neighbor in atom.GetNeighbors())
                for atom in iter_atoms(b.mol)):
            return None
        from .n_sulfonyl_ring_amides import n_sulfonyl_ring_carboxamide_decision
        decision = n_sulfonyl_ring_carboxamide_decision(b.mol, b.Chem)
        if decision is None:
            return None
        if b.ctx.telemetry is not None:
            b.ctx.telemetry.naming_method_selected_rule[decision.rule] += 1
        return PrincipalDecision(
            stage, PrincipalRequirement('amide', tuple(sorted(decision.principal_centers))),
            output_name=decision.name, atom_locants=decision.atom_locants,
            direct_route='general-n-sulfonyl-ring-amide:P-66.1.2', comparison=decision,
            preference_fallback=True, preference_issues=decision.preference_issues)

    def _acyclic_amide_general(self, b, stage):
        """Deliver a mapped general amide while parent preference is open."""
        if not b.mol.GetRingInfo().NumRings():
            return None
        from .acyclic_amide_general import acyclic_amide_general
        decision = acyclic_amide_general(b.mol, b.Chem)
        if decision is None:
            return None
        return PrincipalDecision(stage, PrincipalRequirement(
            'amide', tuple(sorted(decision.principal_centers)), decision.method),
            output_name=decision.name, direct_route='general-mapped-acyclic-amide:P-66.1.1',
            atom_locants=decision.atom_locants, comparison=decision,
            preference_fallback=True, preference_issues=decision.preference_issues)

    def _extended_functional(self, b, stage):
        from .extended_inventory import selection_inventory
        from .complete_functional_parent import complete_functional_parent, functional_parent_frontier

        inventory = selection_inventory(b.mol)
        groups = inventory.principal
        if not groups:
            return None
        if groups[0].kind=='carboxylate':
            from .ionic_assembly import partial_carboxylate_name
            partial=partial_carboxylate_name(b.mol,b.Chem)
            if partial:
                return PrincipalDecision(stage,PrincipalRequirement('carboxylate',tuple(g.center for g in groups)),
                    output_name=partial,direct_route='acid_salt_component:P-77.3.2')
        requirement = PrincipalRequirement(
            groups[0].kind, tuple(g.center for g in groups)
        )
        nitrogen_oxo_higher_class = self._nitrogen_oxo_higher_class(b)
        from .functional_parent_admission import parent_admission_evidence
        admission = parent_admission_evidence(
            b.mol, b.Chem, inventory, groups,
            nitrogen_oxo_higher_class=nitrogen_oxo_higher_class)
        if (len(groups) == 1 and groups[0].kind == "acid"
                and not nitrogen_oxo_higher_class
                and admission.status == 'not-required'):
            established = self._acid(b, stage)
            if established is not None:
                return established
        frontier = functional_parent_frontier(
            b.mol, b.Chem, nitrogen_oxo_higher_class=nitrogen_oxo_higher_class)
        if frontier.status in {'blocked', 'conflict'}:
            return PrincipalDecision(
                stage, requirement,
                direct_route='unsupported-functional-parent:P-44.1.1/P-44.3/P-45',
                comparison=frontier, preference_fallback=True)
        if frontier.status in {'complete', 'tie'}:
            chosen = frontier.candidates[0]
            from .selection_evidence import functional_justification
            return PrincipalDecision(
                stage, replace(requirement, atoms=tuple(sorted(chosen.principal_centers))),
                output_name=chosen.name,
                direct_route='functional_parent_frontier:P-44.1.1/P-44.3/P-45',
                parent_selection=chosen.parent_selection,
                atom_locants=chosen.atom_locants, comparison=frontier,
                justification=functional_justification(b.mol, groups, frontier))
        # Compatibility availability is explicit. An out-of-scope source
        # frontier cannot be silently converted into a preferred first row.
        if (len(groups) == 1 and groups[0].kind == 'carboxylate'
                and b.mol.GetRingInfo().NumRings() >= 4):
            mapped = self._principal_general(b, stage)
            if mapped is not None:
                return mapped
        from .complete_functional_parent import functional_parent_candidates
        legacy = functional_parent_candidates(b.mol, b.Chem)
        name = legacy[0].name if legacy else ''
        if name:
            return PrincipalDecision(
                stage,
                requirement,
                output_name=name,
                direct_route="general-functional-parent",
                preference_issues=(SelectionLimitation(
                    stage, 'P-44/P-45/P-51',
                    'A general functional-parent name is available, but the eligible candidate universe is unproved.'),),
            )
        if len(groups) == 1:
            kind = groups[0].kind
            family = (
                "charged-organophosphorus-oxo-parent"
                if kind in {"phosphonate", "phosphinate", "phosphate"}
                else "phosphorus-acid-parent"
                if kind in {"phosphonic-acid", "phosphinic-acid"}
                else "phosphate-ester-retained-parent"
                if kind in {"phosphoric-acid", "phosphate-acid-ester"}
                else ""
            )
            if family:
                return self._choose(b, stage, requirement, family)
        return None

    @staticmethod
    def _nitrogen_oxo_higher_class(b):
        """Admit a mapped parent from a class obligation, before rendering it."""
        if not any(bond.GetBeginAtom().GetAtomicNum() == 7
                   and bond.GetEndAtom().GetAtomicNum() == 7
                   for bond in iter_bonds(b.mol)):
            return False
        from .nitrous_amide_methods import nitrous_amide_disposition
        return nitrous_amide_disposition(b.mol, b.Chem).status == 'higher-class-route'

    def _acid(self, b, stage):
        group = principal_carboxyl_group(b.mol)
        if not group:
            return None
        requirement = PrincipalRequirement("carboxylic-acid", (group[1],), "acid")
        record = acyclic_acid_record(b.mol, b.Chem, group)
        if record.preference_issues:
            mol_cache_set(b.mol, 'senior_carbon_parent_unavailable', True)
            return PrincipalDecision(stage, requirement,
                direct_route='unsupported:acyclic-acid-parent',
                preference_issues=record.preference_issues)
        if record.name:
            name = b.finalize_record(record)
            if name:
                return PrincipalDecision(
                    stage,
                    requirement,
                    output_name=name,
                    direct_route="principal_carboxylic_acid",
                )
        elif b.mol.GetAtomWithIdx(group[1]).IsInRing():
            return self._choose(
                b,
                stage,
                requirement,
                "acid",
                first=True,
                eligible=lambda item: (
                    group[1] in item.candidate.parent_atoms
                    and item.candidate.name.endswith(
                        ("carboxylic acid", "carbothioic acid", "benzoic acid")
                    )
                ),
            )
        return None

    def _mapped_functional(self, b, stage):
        from .mapped_functional_parent import mapped_functional_parent

        name = mapped_functional_parent(b.mol, b.Chem)
        if name and len(name) <= 260:
            return PrincipalDecision(
                stage,
                PrincipalRequirement("mapped-functional"),
                output_name=name,
                direct_route="mapped_functional_parent",
            )
        return None

    def _multifunctional(self, b, stage):
        from .multifunctional_parent import multifunctional_parent_record
        from .functional_inventory import functional_inventory

        record = multifunctional_parent_record(b.mol)
        if record.name:
            name = b.finalize_record(record)
            if name:
                groups = functional_inventory(b.mol).principal
                return PrincipalDecision(
                    stage,
                    PrincipalRequirement(
                        groups[0].kind, tuple(g.center for g in groups)
                    ),
                    output_name=name,
                    direct_route="multifunctional_parent",
                )
        from .principal_functional import extended_class_competition

        if extended_class_competition(functional_inventory(b.mol)):
            return self._functional(b, stage)
        return None

    def _composition(self, b, stage):
        return self._choose(
            b, stage, PrincipalRequirement("retained-composition"), "composition"
        )

    def _sulfonamide(self, b, stage):
        if sole_sulfonamide(b.mol):
            return self._choose(
                b, stage, PrincipalRequirement("sulfonamide"), "sulfonamide"
            )
        return None

    def _diamide(self, b, stage):
        from .sulfuric_diamide import sulfuric_diamide_name

        name = sulfuric_diamide_name(b.mol)
        if name:
            # This compatibility rule has always been terminal, including an
            # empty output after stereo/length validation. Do not reopen it.
            return PrincipalDecision(
                stage,
                PrincipalRequirement("sulfuric-diamide"),
                output_name=b.finalize_direct(name),
            )
        return None

    def _functional(self, b, stage):
        family = principal_organic_function(b.mol)
        charged = [a for a in iter_atoms(b.mol) if a.GetFormalCharge()]
        if (
            len(charged) == 1
            and charged[0].GetAtomicNum() == 8
            and charged[0].GetFormalCharge() == -1
            and not any(
                a.GetIsotope() or a.GetNumRadicalElectrons() for a in iter_atoms(b.mol)
            )
        ):
            family = "terminal-carboxylate-parent"
        if family:
            from .functional_inventory import functional_inventory

            groups = functional_inventory(b.mol).principal
            required = frozenset(g.center for g in groups)
            return self._choose(
                b,
                stage,
                PrincipalRequirement(family, tuple(sorted(required))),
                family,
                eligible=lambda item: (
                    required <= item.candidate.principal_group_centers
                ),
            )
        return None

    def _fused_hydrogen_state(self, b, stage):
        from .fused_hydrogen_state import fused_hydrogen_record
        from .acyclic.rules import AcyclicNameRecord

        record = fused_hydrogen_record(b.ctx, mapped=True)
        if not record.name:
            return None
        # Keep exact functional templates whose H/suffix coverage was already
        # established (including the preferred oxindole operation).
        if any(
            item.candidate.fused_lactam_seniority
            and item.candidate.fused_lactam_seniority[0] == len(record.carbonyl_atoms)
            for item in b.prepare(self.groups["carbonyl"])
        ):
            return None
        name = (
            record.name
            if record.mapped
            else b.finalize_record(AcyclicNameRecord(record.name, record.atom_locants))
        )
        if len(name) > 260:
            return None
        if name:
            return PrincipalDecision(
                stage,
                PrincipalRequirement("fused-carbonyl", record.carbonyl_atoms, "one"),
                output_name=name,
                direct_route="fused_hydrogen_state",
            )
        return None

    def _carbonyl(self, b, stage):
        from .bridged_fused_carbonyl import bridged_fused_ketone_decision

        bridged = bridged_fused_ketone_decision(b.mol, b.Chem)
        if bridged is not None:
            return PrincipalDecision(
                stage,
                PrincipalRequirement('ring-carbonyl',
                                     tuple(sorted(bridged.principal_centers)), 'one'),
                output_name=bridged.name,
                direct_route='bridged_fused_carbonyl:' + bridged.rule,
                atom_locants=bridged.atom_locants, comparison=bridged,
                preference_issues=bridged.preference_issues)
        suffix, sites = principal_carbonyl_sites(b.mol)
        if not supported_ring_carbonyl_selection(b.mol, suffix, sites):
            return None
        candidates = b.prepare(self.groups["carbonyl"])
        ranked = [
            (principal_ring_carbonyl_key(b.mol, item.candidate), item)
            for item in candidates
            if not b.safety_reasons(item.candidate)
        ]
        ranked = [(key, item) for key, item in ranked if key]
        if not ranked:
            return None
        key = min(key for key, _ in ranked)
        selected = b.best([item for other, item in ranked if other == key])
        if selected is not None:
            return PrincipalDecision(
                stage,
                PrincipalRequirement(
                    "ring-carbonyl", tuple(i for pair in sites for i in pair), suffix
                ),
                prepared=selected,
            )
        return None

    def _ring(self, b, stage):
        from .principal_oxygen import monocyclic_polyol_record
        polyol=monocyclic_polyol_record(b.mol,b.Chem)
        if polyol is not None:
            name,parent,mapping,centers=polyol
            return PrincipalDecision(stage,PrincipalRequirement('alcohol',centers,'ol'),
                output_name=name,direct_route='mapped_monocyclic_polyol:P-44.1.1/P-45',atom_locants=mapping)
        kind, atoms = principal_ring_groups(b.mol)
        if not kind or kind == "alcohol" and b.mol.GetRingInfo().NumRings() < 2:
            return None
        required = set(atoms)

        def admitted(item):
            candidate = item.candidate
            return (
                required <= candidate.parent_atoms
                and (
                    kind != "alcohol"
                    or candidate.name.endswith("ol")
                    and all(
                        b.mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                        for i in candidate.parent_atoms
                    )
                )
                and (
                    kind != "lactone"
                    or _principal_lactone_branches_supported(
                        b.mol, candidate.parent_atoms
                    )
                )
            )

        eligible = [item for item in b.prepare(self.groups["ring"]) if admitted(item)]
        if not eligible and kind == "alcohol":
            eligible = [
                item
                for item in b.prepare(self.groups["polycyclic-alcohol"])
                if admitted(item)
            ]
        if not eligible:
            return None
        if kind == "alcohol":
            # Fixed hydride topology is selected before optimizing suffix
            # numbering; do not let a suffix-only key rewrite the descriptor.
            if eligible[0].candidate.family in _POLYCYCLIC_FAMILIES:
                descriptor = re.search(
                    r"(?:\w*cyclo|\w*spiro)\[[^]]+\]", eligible[0].name
                )
                if descriptor:
                    eligible = [item for item in eligible if descriptor[0] in item.name]
            eligible.sort(key=lambda item: _principal_alcohol_suffix_key(item, atoms))
        return PrincipalDecision(
            stage,
            PrincipalRequirement(
                kind, tuple(atoms), "ol" if kind == "alcohol" else "one"
            ),
            prepared=eligible[0],
        )

    def _oxygen(self, b, stage):
        record = principal_oxygen_record(b.mol)
        if record.name:
            name = b.finalize_record(record)
            if name:
                return PrincipalDecision(
                    stage,
                    PrincipalRequirement("acyclic-oxygen"),
                    output_name=name,
                    direct_route="principal_oxygen",
                )
        return None

    def _amide(self, b, stage):
        if sole_characteristic_amide(b.ctx):
            return self._choose(b, stage, PrincipalRequirement("amide"), "amide")
        return None

    def _acyclic_carbon_parent_limit(self, b, stage):
        from .acyclic.rules import _carbon_hydride_scope, _unrenderable_carbon_parent

        if _carbon_hydride_scope(b.mol):
            record = b.ctx.acyclic_name_record()
            if not record.name and not record.preference_issues:
                from .acyclic.rules import _candidate_parent_paths
                paths = _candidate_parent_paths(b.mol)
                longest = max(map(len, paths), default=0)
                record = _unrenderable_carbon_parent(
                    [path for path in paths if len(path) == longest])
        else:
            from .constants import ROOTS
            carbon_count = sum(a.GetAtomicNum() == 6 for a in iter_atoms(b.mol))
            if all(n in ROOTS for n in range(1, carbon_count + 1)):
                return None
            from .principal_site_parent_frontier import (
                PrincipalSiteFrontierStatus, principal_site_parent_frontier,
            )
            frontier = principal_site_parent_frontier(
                b.mol, b.Chem, max_parent_carbons=carbon_count)
            if (frontier.status is not PrincipalSiteFrontierStatus.PROVED
                    or frontier.maximum_parent_length in ROOTS):
                return None
            # The frontier establishes source-class coverage BEFORE length,
            # then multiple-bond seniority. In particular, a shorter diol
            # parent is not rejected just because its alkyl arms are long.
            record = _unrenderable_carbon_parent(
                [row.parent_atoms for row in frontier.candidates])
        if not record.preference_issues:
            return None
        mol_cache_set(b.mol, 'senior_carbon_parent_unavailable', True)
        # This is a proved senior carbon-parent writer limit. Keep the
        # existing empty-string API and expose the reason in NamingResult.
        return PrincipalDecision(stage, PrincipalRequirement('carbon-parent'),
            direct_route='unsupported:acyclic-carbon-parent',
            preference_issues=record.preference_issues)


def _locant_number_key(locant):
    match = re.fullmatch(r"(\d+)(.*)", str(locant))
    return (int(match[1]), match[2]) if match else (999, str(locant))


def _principal_alcohol_suffix_key(item, group_atoms):
    # Untagged hydro components can rotate their suffix numbering. Read the
    # resulting suffix locants rather than the original component orientation.
    rendered = re.search(
        r"-([0-9a-z,]+)-(?:di|tri|tetr|pent|hex)?ol$", item.candidate.name
    )
    if rendered:
        locants = rendered[1].split(",")
    else:
        atom_locants = dict(item.candidate.stereo_locants)
        locants = [atom_locants.get(idx, "999") for idx in group_atoms]
    return tuple(sorted(_locant_number_key(loc) for loc in locants)), item.name


def _principal_lactone_branches_supported(mol, parent_atoms):
    """Keep extra fused/bridged substituents on their validated polycycle routes."""
    systems = []
    for ring in mol.GetRingInfo().AtomRings():
        atoms = set(ring)
        overlapping = [system for system in systems if system[0] & atoms]
        count = 1
        for system in overlapping:
            atoms.update(system[0])
            count += system[1]
            systems.remove(system)
        systems.append((atoms, count))
    for atoms, count in systems:
        if atoms <= parent_atoms or count == 1:
            continue
        local_rings = [set(r) for r in mol.GetRingInfo().AtomRings() if set(r) <= atoms]
        # A two-ring carbon fused component has a covered radical renderer.
        # Leave heterofused and bridged branch systems on their existing routes.
        if not (
            count == 2
            and len(local_rings) == 2
            and len(local_rings[0] & local_rings[1]) == 2
            and all(
                mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                and not int(mol.GetAtomWithIdx(i).GetChiralTag())
                for i in atoms
            )
        ):
            return False
    return True
