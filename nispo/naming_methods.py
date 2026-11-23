"""Graph-based P-44/P-51 selection for complete functional parent assemblies.

Class-specific rules precede the generic search. Eligible skeletal replacement
precedes substitutive parents; P-44 selects the senior parent before P-45
compares alternative expressions of that same parent. Names and their lengths
never establish eligibility or structural identity.

Supported multiplying units include functionalized carbon chains and mapped
rings. The connector accounts for identical arms and any independently
nameable substituents on its central group.
"""

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from itertools import groupby
from threading import RLock

from .molecule_iteration import iter_atoms, iter_bonds
from .molecule_cache import mol_cache_get
from .graph_model import graph_view, method_bond_counts
from .branch_rules import collect_subtree, parent_chain_name
from .complete_functional_parent import (
    FunctionalParentFrontier,
    functional_parent_candidates,
    functional_parent_frontier,
    _render,
    _render_functional_candidate as _DEFAULT_UNIT_PARENT_RENDER,
)
from .method_parent_frontier import MethodParentStatus, select_method_parent
from .acyl_replacement_methods import AcylReplacementProof
from .naming_result import SelectionLimitation
from .component_mapping import (
    MappedComponent, decorate_component, fully_covered, mapped_branch,
)
from .constants import HALO, MULT, ROOTS
from .extended_inventory import selection_inventory
from .formatting import (COMPLEX_MULT, format_prefixes, multiplied_suffix, join_prefix_parent,
                         alkyl_to_alkoxy, substituent_parent_suffix,attach_suffix,enclose_name_unit,
                         prefix_alphabetization_key)


_FUNCTIONS = frozenset({"acid", "amide", "ester", "nitrile"})
_LINK_ELEMENTS = frozenset({6, 7, 8, 16})


@dataclass(frozen=True, slots=True)
class MethodDecision:
    name: str
    method: str
    principal_centers: frozenset[int]
    atoms: frozenset[int]
    rule: str = ""
    atom_locants: tuple = ()
    heterounits: tuple = ()
    component_roles: tuple = ()
    # Retain every source-maximal mapped functional-parent candidate. A
    # selected display record alone is insufficient when equivalent maps tie.
    functional_parent_frontier: FunctionalParentFrontier | None = None
    preference_issues: tuple[SelectionLimitation, ...] = ()
    justification: object = None


@dataclass(frozen=True, slots=True)
class _MultiplicativeOutcome:
    status: str
    decision: MethodDecision | None = None
    reason: str = ""
    evidence: tuple = ()
    proposals: tuple[MethodDecision, ...] = ()


class NamingMethodStatus(str, Enum):
    """Whether a naming-method route completed, is required but blocked, or is inapplicable."""

    COMPLETE = "complete"
    PENDING = "pending"
    OUTSIDE_SCOPE = "outside-scope"


@dataclass(frozen=True, slots=True)
class NamingMethodOutcome:
    """Status-bearing result for callers that must not treat a required method as fallback."""

    status: NamingMethodStatus
    decision: MethodDecision | None = None
    rule: str = ""
    method: str = ""
    reason: str = ""
    evidence: tuple = ()
    # A later shared proposal pool may discharge a generic comparison barrier.
    deferred: bool = False



_DEFAULT_FUNCTIONAL_PARENT_CANDIDATES = functional_parent_candidates
_DEFAULT_FUNCTIONAL_PARENT_RENDER = _render

_UNIT_PARENT_CACHE = OrderedDict()
_UNIT_PARENT_CACHE_LOCK = RLock()

def select_naming_method_outcome(mol, Chem, *, trace=None):
    """Return complete/pending/outside-scope status for source-directed methods.

    ``select_naming_method`` remains the compatibility API and still returns a
    decision or ``None``. New principal-selection callers should use this
    method so an eligible mandatory replacement name whose renderer declines
    cannot be mistaken for permission to try a lower-preference route.
    """
    pending = []
    evidence = []
    decision = select_naming_method(mol, Chem, trace=evidence, _outcome_status=pending)
    if decision is not None and decision.preference_issues:
        issue = decision.preference_issues[0]
        _record_pending(pending, issue.rule, issue.reason, method=decision.method)
    if trace is not None:
        trace.extend(evidence)
    if decision is not None and not pending:
        return NamingMethodOutcome(NamingMethodStatus.COMPLETE, decision,
            rule=decision.rule, method=decision.method, reason="rendered",
            evidence=tuple(evidence))
    if pending:
        blocked = pending[-1]
        return NamingMethodOutcome(blocked.status, decision, rule=blocked.rule, method=blocked.method,
            reason=blocked.reason, evidence=tuple(evidence) + blocked.evidence,
            deferred=blocked.deferred)
    return NamingMethodOutcome(NamingMethodStatus.OUTSIDE_SCOPE,
                               reason="no-source-directed-method", evidence=tuple(evidence))


def _record_pending(status_sink, rule, reason, *, method="", deferred=False, **evidence):
    if status_sink is not None:
        status_sink.append(NamingMethodOutcome(
            NamingMethodStatus.PENDING, rule=rule, method=method, reason=reason,
            evidence=tuple(sorted(evidence.items())), deferred=deferred))


def _senior_parent_repetition_bound(mol, frontier, principal_group_count):
    """Return the least restrictive source-proved copy bound for P-44 frames.

    This is admitted only for the complete mapped all-carbon-tree functional
    frontier: graph rank and suffix coverage are present, every tied senior
    frame is retained, and every frame consists of original carbon atoms.
    Each disjoint copy needs ``K`` source groups and the frame's full carbon
    skeleton. The maximum bound across equivalent senior frames is the
    conservative, least restrictive result. Missing proof data yields None.
    """
    coverage = frontier.suffix_coverage
    if (not isinstance(coverage, int) or coverage <= 0
            or frontier.graph_parent_rank is None or not frontier.candidates):
        return None
    total_carbons = sum(atom.GetAtomicNum() == 6 for atom in iter_atoms(mol))
    frame_atoms = {
        (frozenset(record.parent_atoms), frozenset(record.principal_centers))
        for record in frontier.candidates
    }
    if not frame_atoms:
        return None
    frame_bounds = []
    for atoms, centers in frame_atoms:
        if len(centers) != coverage or not atoms:
            return None
        if any(mol.GetAtomWithIdx(index).GetAtomicNum() != 6 for index in atoms):
            return None
        frame_bounds.append({
            "suffix_copy_bound": principal_group_count // coverage,
            "carbon_skeleton_copy_bound": total_carbons // len(atoms),
            "parent_carbon_count": len(atoms),
            "parent_atoms": tuple(sorted(atoms)),
            "principal_centers": tuple(sorted(centers)),
        })
    maximum_copies = max(
        min(bound["suffix_copy_bound"], bound["carbon_skeleton_copy_bound"])
        for bound in frame_bounds
    )
    return {
        "maximum_copies": maximum_copies,
        "principal_group_count": principal_group_count,
        "total_parentable_carbons": total_carbons,
        "equivalent_senior_frames": tuple(frame_bounds),
    }


@dataclass(frozen=True, slots=True)
class _Unit:
    atoms: frozenset[int]
    root: int
    connector: int
    signature: str
    parent: str
    locant: str
    centers: frozenset[int]
    rank: tuple
    substituted: bool
    ester_owners: tuple[str, ...] = ()
    # P-44 evidence stays separate from rendered spelling and topology identity.
    parent_facts: object = None
    parent_atoms: frozenset[int] = frozenset()


def _principal_site_method_handoff(mol, Chem, *, trace=None, status_sink=None):
    """Apply mandatory replacement, then retain a maximum site parent for joining.

    A complete smaller-unit multiplication or legacy substitutive rendering
    cannot outrank the P-44.1.1/P-44.3 maximum parent unless its parent family
    is joined to that selected frame. The later shared proposal pool owns the
    complete mapped suffix/prefix spelling for this bounded tree universe.
    """
    # A saturated branched C/O/S graph can contain a qualifying replacement
    # path even when the residual whole-graph atom set is branched. Enumerate
    # this narrow, fully mapped path universe before the legacy whole-set
    # replacement test gets a chance to decline it.
    from .branched_replacement_methods import (
        BranchedReplacementStatus, branched_replacement_frontier,
    )
    replacement_frontier = branched_replacement_frontier(mol, Chem)
    if replacement_frontier.status is BranchedReplacementStatus.COMPLETE:
        selected = replacement_frontier.decision
        from .extended_inventory import selection_inventory
        principal_groups = tuple(selection_inventory(mol).principal)
        decision = MethodDecision(
            selected.name, "skeletal-replacement",
            frozenset(group.center for group in principal_groups),
            frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1),
            "P-51.4.1.1/P-44.1.1/P-44.3/P-44.4/P-15.4.3.2.1",
            selected.atom_locants, selected.heterounits,
            selected.ownership,
        )
        if trace is not None:
            trace.append(dict(method="skeletal-replacement", status="complete",
                eligible=True, rule=decision.rule,
                scope="complete mapped branched neutral C/N/O/S path frontier",
                replacement_frontier=replacement_frontier))
        return True, decision
    if replacement_frontier.status is BranchedReplacementStatus.PENDING:
        _record_pending(
            status_sink, "P-51.4.1.1/P-44.1.1/P-44.3/P-44.4",
            replacement_frontier.reason,
            method="skeletal-replacement",
            replacement_frontier=replacement_frontier,
        )
        if trace is not None:
            trace.append(dict(method="skeletal-replacement", status="pending",
                eligible=True, rule="P-51.4.1.1/P-44.1.1/P-44.3/P-44.4",
                replacement_frontier=replacement_frontier))
        return True, None

    from .principal_site_parent_frontier import (
        PrincipalSiteFrontierStatus,
        principal_site_parent_frontier,
    )

    frontier = principal_site_parent_frontier(mol, Chem)
    if frontier.status is not PrincipalSiteFrontierStatus.PROVED:
        return False, None
    source_groups = tuple(site.source_group for site in frontier.sites)
    site_atoms = frozenset(index for site in frontier.sites
                           for index in site.group_atoms)
    noncarbon = frozenset(atom.GetIdx() for atom in iter_atoms(mol)
                          if atom.GetAtomicNum() > 1
                          and atom.GetAtomicNum() != 6)
    if (frontier.subordinate_groups or noncarbon != site_atoms - frozenset(
            site.host_atom for site in frontier.sites
            if site.kind == "alcohol" or site.kind == "ketone")):
        # Keep this join bounded to complete ordinary principal-site graphs;
        # an unrecognized or subordinate heterogroup needs its own method
        # admission before P-51/P-44 can be settled here.
        return False, None

    replacement_diagnostic = {}
    replacement = _replacement_parent(
        mol, Chem, source_groups, evidence=True,
        diagnostic=replacement_diagnostic,
    )
    if replacement:
        if isinstance(replacement, AcylReplacementProof):
            decision = replacement.decision
        else:
            name, path, units = replacement
            centers = frozenset(site.principal_center for site in frontier.sites)
            atoms = frozenset(atom.GetIdx() for atom in iter_atoms(mol)
                              if atom.GetAtomicNum() > 1)
            decision = MethodDecision(
                name, "skeletal-replacement", centers, atoms,
                "P-51.4.1.1",
                tuple((idx, str(locant))
                      for locant, idx in enumerate(path, 1)),
                tuple(units),
            )
        if trace is not None:
            trace.append(dict(method="skeletal-replacement", status="complete",
                eligible=True, rule="P-51.4.1.1",
                scope="source-proved ordinary site frontier before mapped suffix selection",
                site_frontier=frontier))
        return True, decision
    if (replacement_diagnostic.get("eligibility_reason")
            == "eligible-acyclic-replacement"
            and replacement_diagnostic.get("rendering_reason") != "rendered"):
        _record_pending(
            status_sink, "P-51.4.1.1",
            "mandatory-replacement-renderer-unavailable",
            method="skeletal-replacement",
            principal_site_frontier=frontier,
            eligibility_reason=replacement_diagnostic["eligibility_reason"],
            heterounit_count=replacement_diagnostic.get("heterounit_count"),
            rendering_reason=replacement_diagnostic.get("rendering_reason"),
        )
        return True, None
    if trace is not None:
        trace.append(dict(
            method="principal-site-parent-frontier",
            status="proved",
            eligible=True,
            rule="P-41/P-44.1.1/P-44.3/P-45.0",
            scope="neutral all-carbon tree; source-site and parent-bond maxima with explicit annotation admission",
            maximum_site_count=frontier.maximum_site_count,
            maximum_parent_length=frontier.maximum_parent_length,
            candidates=frontier.candidates,
            annotation_proof=frontier.annotation_proof,
            reason=frontier.reason,
            deferred=True,
        ))
    _record_pending(
        status_sink,
        "P-41/P-44.1.1/P-44.3/P-45.0",
        "maximum-ordinary-principal-site-parent-requires-mapped-join",
        method="principal-site-parent",
        deferred=True,
        principal_site_frontier=frontier,
    )
    return True, None


def select_naming_method(mol, Chem, *, trace=None, _outcome_status=None):
    """Return a fully covered decision, or leave existing routes authoritative."""
    if mol.GetNumHeavyAtoms() > 128 or any(
        a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol)
    ) or mol.GetStereoGroups():
        return None
    view = mol_cache_get(mol, 'graph_view')
    initialized = False
    if not mol.NeedsUpdatePropertyCache():
        try:
            mol.GetRingInfo().NumRings()
        except RuntimeError:
            pass
        else:
            initialized = True
            if view is None:
                view = graph_view(mol, Chem)
    # Basic native reads are safe on an uninitialized graph, but constructing
    # graph_view would perceive its rings earlier than the original route.
    # Keep that route's operation order while sharing its harmless signals.
    numbers = (view.atomic_nums if view is not None else
               tuple(a.GetAtomicNum() for a in iter_atoms(mol)))
    elements = (view.elements if view is not None else
                Counter(a.GetSymbol() for a in iter_atoms(mol)))
    isotopes = (view.isotopes if view is not None else
                tuple(a.GetIsotope() for a in iter_atoms(mol)))
    bonds = (method_bond_counts(mol, Chem) if view is not None else
             Counter((min(b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()),
                      max(b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()),
                      b.GetBondTypeAsDouble()) for b in iter_bonds(mol)))
    # P-51.4 precedes substitutive mononuclear parents when the complete
    # original graph proves four heterounits. This mapped constructor also
    # covers standard skeletal elements and termini absent from the ordinary
    # C/N/O/S inventory; characteristic-group atoms never inflate its count.
    if initialized and sum(number not in {1, 6} for number in numbers) >= 4:
        from .standard_heterochain_methods import standard_heterochain_frontier
        heterochains = standard_heterochain_frontier(mol, Chem)
        if heterochains:
            decision = heterochains[0].decision
            if trace is not None:
                trace.append(dict(method=decision.method, eligible=True,
                    rule=decision.rule,
                    scope='complete standard-valence acyclic heterochain frontier',
                    equivalent_atom_locants=tuple(c.decision.atom_locants for c in heterochains)))
            return decision
    if elements['N'] == 2 and any(elements[element] for element in ('F','Cl','Br','I')):
        from .hydrazine_methods import halogenated_hydrazine_decision
        hydrazine = halogenated_hydrazine_decision(mol, Chem)
        if hydrazine is not None:
            return hydrazine
    # P-68.2 selects the mononuclear Group 14 parent and uses characteristic
    # suffixes where available. The producer proves its exact whole-graph
    # scope before the ordinary neutral C/N/O/S inventory is consulted.
    if any(elements[element] for element in ('Si', 'Ge', 'Sn', 'Pb')):
        from .group14_mononuclear_methods import (
            group14_mononuclear_proposal, recognize_group14_mononuclear)
        recognition = recognize_group14_mononuclear(mol)
        proposal = group14_mononuclear_proposal(mol, Chem)
        if trace is not None:
            trace.append(dict(method='substitutive', eligible=proposal is not None,
                rule='P-68.2.1/P-68.2.5/P-68.2.6',
                scope='one ordinary-valence mapped Group 14 parent',
                eligibility_reason=recognition.reason))
        if proposal is not None:
            return proposal.as_method_decision()
    # P-68.3 chooses the pnictogen hydride and its characteristic suffix
    # for this exact organyl scope. Retained-acid derivatives have already
    # reached their P-67 route and are excluded by this graph proof.
    if any(elements[element] for element in ('P', 'As', 'Sb', 'Bi')):
        from .group15_imine_methods import group15_imine_decision
        imine = group15_imine_decision(mol, Chem)
        if trace is not None:
            trace.append(dict(method='substitutive', eligible=imine is not None,
                rule='P-68.3.2.3.1/P-68.3.3/P-74.2.1.5',
                scope='one mapped pnictogen imine with typed N/element substitution'))
        if imine is not None:
            return imine
        from .group15_mononuclear_methods import group15_mononuclear_decision
        pnictogen = group15_mononuclear_decision(mol, Chem)
        if trace is not None:
            trace.append(dict(method='substitutive', eligible=pnictogen is not None,
                rule='P-21.1/P-68.3.2/P-68.3.3',
                scope='one mapped pnictogen hydride with an optional chalcogen suffix'))
        if pnictogen is not None:
            return pnictogen
    # The P-68.4.3 imide families use class-specific lambda parents and
    # suffixes. An exact ligand/whole-graph proof precedes generic P-51
    # chain replacement, which cannot establish their preferred method.
    if any(bonds[7, number, 2] for number in (16, 34, 52)):
        from .chalcogen_imide_methods import chalcogen_imide_decision
        chalcogen_imide = chalcogen_imide_decision(mol, Chem)
        if trace is not None:
            trace.append(dict(method='substitutive', eligible=chalcogen_imide is not None,
                rule='P-68.4.3.3-P-68.4.3.8',
                scope='one mapped neutral chalcogen imide with complete ligand ownership'))
        if chalcogen_imide is not None:
            return chalcogen_imide
    # P-68.3.1.3.2 prefers a substituted diazene parent for ordinary monoazo
    # compounds. Its producer rejects the characteristic-group contexts that
    # require a functional parent and a diazenyl/diazenediyl construction.
    if bonds[7, 7, 2]:
        from .diazene_methods import diazene_decision
        diazene = diazene_decision(mol, Chem)
        if trace is not None:
            trace.append(dict(method='substitutive', eligible=diazene is not None,
                rule='P-68.3.1.3.2.1/P-68.3.1.3.2.2',
                scope='one ordinary monoazo parent with complete mapped organyls'))
        if diazene is not None:
            return diazene
    # N--halogen bonds form amides of halogen acids under P-61.3.2.2 and
    # P-68.5.3. Ordinary halo-prefix/amine routes cannot override that class
    # merely because the bounded functional-parent renderer is unavailable.
    if any(a == 7 and c in {9, 17, 35, 53} for a, c, _order in bonds):
        from .halogen_acid_amide_methods import (
            recognize_hypohalous_amide,
            hypohalous_amide_decision, classify_halogen_acid_amide,
            halogen_acid_amide_proposal)
        disposition = classify_halogen_acid_amide(mol, Chem)
        recognition = recognize_hypohalous_amide(mol, Chem)
        halogen_amide = hypohalous_amide_decision(mol, Chem)
        if halogen_amide is None:
            proposal = halogen_acid_amide_proposal(mol, Chem)
            if proposal is not None:
                halogen_amide = proposal.as_method_decision()
        if trace is not None:
            trace.append(dict(method='functional-class',
                eligible=halogen_amide is not None,
                rule=halogen_amide.rule if halogen_amide is not None else disposition.rule,
                scope='one mapped organic halogen-acid amide',
                eligibility_reason=disposition.reason, recognition=recognition,
                class_disposition=disposition))
        if halogen_amide is not None:
            return halogen_amide
        if disposition.status not in {'higher-class-route', 'cyclic-imide-route'}:
            _record_pending(_outcome_status, disposition.rule, disposition.reason,
                            method='functional-class', recognition=disposition)
            return None
    site_handoff, site_decision = _principal_site_method_handoff(
        mol, Chem, trace=trace, status_sink=_outcome_status,
    )
    if site_handoff:
        return site_decision
    # Polyamine role assignment precedes generic annotation handling: the
    # original labelled graph must decide repeated-unit identity and CIP.
    from .component_mapping import chemical_inventory
    inventory=chemical_inventory(mol)
    if inventory.principal and inventory.principal[0].kind=='amine':
        from .polyamine_methods import select_polyamine_method
        decision=select_polyamine_method(mol,Chem,inventory)
        if trace is not None:
            trace.append(dict(method=decision.method if decision else 'complex-polyamine-method',eligible=decision is not None,
                rule=decision.rule if decision else 'P-62.2.4.1.3/P-51.3/P-51.4',scope='characteristic-N role partition and identical rooted senior amine units'))
        return decision
    # Class-specific graph proofs determine these methods before the generic
    # P-51 search. P-63.6 permits cyclic multiplication but excludes the
    # corresponding multiplication of two alkyl parents; P-68.4.3.2 chooses
    # a lambda parent for a chain of oxidized chalcogen atoms.
    # These are necessary signals from the existing producer scopes, not
    # eligibility proofs. Keep the native producer order on partial graphs,
    # where a recognizer's earlier native reads can still be significant.
    heavy_chalcogens = elements['S'] + elements['Se'] + elements['Te']
    oxo_chalcogen = any(bonds[8, element, 2] for element in (16, 34, 52))
    oxide = None
    if not initialized or heavy_chalcogens == 1 and oxo_chalcogen:
        from .chalcogen_oxide_methods import chalcogen_oxide_decision
        oxide = chalcogen_oxide_decision(mol, Chem)
    if oxide is None and (not initialized or 2 <= heavy_chalcogens <= 6 and oxo_chalcogen):
        from .polychalcogen_oxide_methods import polychalcogen_oxide_decision
        oxide = polychalcogen_oxide_decision(mol, Chem)
    if oxide is not None:
        if trace is not None:
            trace.append(dict(method=oxide.method, eligible=True,
                rule=oxide.rule,
                scope='complete class-specific chalcogen oxide construction'))
        return oxide
    # P-68.4 makes a homogeneous chain of three or more chalcogens the
    # parent hydride. Its exact graph proof precedes the generic P-51
    # replacement search; it does not fragment that chain into ether units.
    chalcogen_count = elements['O'] + heavy_chalcogens
    chalcogen_kinds = sum(bool(elements[element]) for element in ('O', 'S', 'Se', 'Te'))
    chalcogen_chain = None
    if not initialized or chalcogen_count >= 3 and chalcogen_kinds == 1:
        from .chalcogen_chain_methods import chalcogen_chain_decision
        chalcogen_chain = chalcogen_chain_decision(mol, Chem)
    if chalcogen_chain is not None:
        if trace is not None:
            trace.append(dict(method=chalcogen_chain.method, eligible=True,
                rule=chalcogen_chain.rule,
                scope='one homogeneous chalcogen chain with complete mapped terminal organyls'))
        return chalcogen_chain
    # P-21.2.3.1/P-68.4.2.1 gives the alternating a[ba]n subclass its
    # preselected hydride. Its complete graph proof is more specific than
    # the general heterogeneous-chain replacement search below.
    heterochalcogen_chain = None
    if (not initialized or chalcogen_count >= 3
            and chalcogen_count % 2 == 1 and chalcogen_kinds == 2):
        from .heterochalcogen_chain_methods import heterochalcogen_chain_decision
        heterochalcogen_chain = heterochalcogen_chain_decision(mol, Chem)
    if heterochalcogen_chain is not None:
        if trace is not None:
            trace.append(dict(method=heterochalcogen_chain.method, eligible=True,
                rule=heterochalcogen_chain.rule,
                scope='one alternating chalcogen parent with complete mapped terminal organyls'))
        return heterochalcogen_chain
    # P-68.4.2.2 retains identical chalcogen pairs/triples when a
    # heterogeneous terminal-H chain needs a hydride plus peroxol suffix.
    chain_suffix = None
    if not initialized or chalcogen_count >= 3 and chalcogen_kinds >= 2:
        from .chalcogen_chain_suffix_methods import chalcogen_chain_suffix_decision
        chain_suffix = chalcogen_chain_suffix_decision(mol, Chem)
    if chain_suffix is not None:
        if trace is not None:
            trace.append(dict(method=chain_suffix.method, eligible=True,
                rule=chain_suffix.rule,
                scope='intact terminal peroxol pair and homogeneous parent block'))
        return chain_suffix
    # P-63.3.1 also applies when the carbon hydride is branched or cyclic.
    # This producer proves the function-free single-bridge scope before it
    # can precede the general P-51 replacement/multiplication search.
    peroxide = None
    if (not initialized or elements['O'] + elements['S'] == 2
            and (bonds[8, 8, 1] or bonds[16, 16, 1])):
        from .peroxide_hydrides import peroxide_hydride_decision
        peroxide = peroxide_hydride_decision(mol, Chem)
    if peroxide is not None:
        if trace is not None:
            trace.append(dict(method=peroxide.method, eligible=True,
                rule=peroxide.rule,
                scope='one mapped peroxide/disulfide bridge below replacement threshold'))
        return peroxide
    if any(isotopes) or any(int(a.GetChiralTag()) for a in iter_atoms(mol)) or any(int(b.GetStereo()) for b in iter_bonds(mol)):
        return _annotated_method(mol,Chem,trace,_outcome_status)
    # P-51.4.1.1 excludes nitrogen that belongs to the principal group, not
    # every chain nitrogen. This producer proves a senior alcohol suffix and
    # counts only the remaining mapped skeletal N/O units.
    if elements['N']:
        from .nitrogen_replacement_methods import nitrogen_replacement_frontier
        nitrogen_numberings = nitrogen_replacement_frontier(mol, Chem)
        if nitrogen_numberings:
            replacement = nitrogen_numberings[0].decision
            if trace is not None:
                trace.append(dict(method=replacement.method, eligible=True,
                    rule=replacement.rule,
                    scope='mapped nonprincipal skeletal N/O units with an alcohol suffix',
                    equivalent_atom_locants=tuple(
                        candidate.decision.atom_locants
                        for candidate in nitrogen_numberings)))
            return replacement
    # P-51.4.1 fixes replacement numbering before unsaturation. This exact
    # function-free C/O/S proof precedes the narrower saturated inventory.
    replacement_numberings = None
    if (not initialized or elements['O'] + elements['S'] >= 4
            and (bonds[6, 6, 2] or bonds[6, 6, 3])):
        from .unsaturated_replacement_methods import unsaturated_replacement_frontier
        replacement_numberings = unsaturated_replacement_frontier(mol, Chem)
    if replacement_numberings:
        replacement = replacement_numberings[0].decision
        if trace is not None:
            trace.append(dict(method=replacement.method, eligible=True,
                rule=replacement.rule,
                scope='complete unsaturated C/O/S chain with at least four isolated heterounits',
                equivalent_atom_locants=tuple(
                    candidate.decision.atom_locants
                    for candidate in replacement_numberings)))
        return replacement
    inventory = selection_inventory(mol)
    ring_count = mol.GetRingInfo().NumRings()
    ordinary_heterogroups = (not ring_count
        and all(number in {6,8,16} and (number == 6 or
                (len(view.neighbors[idx]) if view is not None else mol.GetAtomWithIdx(idx).GetDegree()) == 2)
                for idx, number in enumerate(numbers))
        and all(order == 1 for _a, _b, order in bonds))
    if not inventory.supported and not ordinary_heterogroups:
        return None
    groups = inventory.principal if inventory.supported else ()
    if (len(groups) == 2 and all(g.kind == 'lactam' for g in groups) and ring_count >= 4
            and any(bond.GetBondTypeAsDouble() == 1 and bond.GetBeginAtom().GetIsAromatic()
                    and bond.GetEndAtom().GetIsAromatic() for bond in iter_bonds(mol))):
        atoms = frozenset(idx for idx, number in enumerate(numbers) if number > 1)
        return _multiplicative_parent_outcome(mol, Chem, groups, atoms).decision
    if groups and (groups[0].kind not in _FUNCTIONS | {'alcohol'}
                   or any(g.kind != groups[0].kind for g in groups)):
        return None
    centers = frozenset(g.center for g in groups)
    atoms = frozenset(idx for idx, number in enumerate(numbers) if number > 1)
    if not groups and not ring_count and elements['C'] == len(numbers):
        return None
    # P-51.1.3 makes eligible skeletal replacement precede substitutive
    # parents. Functional-class PIN exceptions retain their class route under
    # P-51.1.1/P-51.2.1, so esters reach that route before this producer.
    replacement_diagnostic = {}
    replacement = None
    if not groups or groups[0].kind != 'ester':
        replacement = _replacement_parent(mol, Chem, groups, evidence=True,
                                           diagnostic=replacement_diagnostic)
        if trace is not None:
            trace.append(dict(method='skeletal-replacement',eligible=bool(replacement),
                              rule='P-51.4.1.1',
                              scope='typed acyclic eligibility plus bounded C/N/O/S renderer',
                              eligibility_reason=replacement_diagnostic.get('eligibility_reason',
                                                                            'producer-scope-declined'),
                              heterounit_count=replacement_diagnostic.get('heterounit_count'),
                              rendering_reason=replacement_diagnostic.get('rendering_reason',
                                                                           'not-rendered'),
                              acyl_replacement_proof=replacement if
                                  isinstance(replacement, AcylReplacementProof) else None))
        if replacement:
            if isinstance(replacement, AcylReplacementProof):
                # The acyl producer already returns a fully mapped decision;
                # reconstructing it from the generic parent tuple would lose
                # typed suffix roles and fixed suffix-locant evidence.
                from .selection_evidence import acyl_replacement_justification
                return replace(replacement.decision, justification=acyl_replacement_justification(
                    mol, Chem, replacement, replacement.decision))
            name,path,units = replacement
            return MethodDecision(name, "skeletal-replacement", centers, atoms,'P-51.4.1.1',
                                  tuple((idx,str(j)) for j,idx in enumerate(path,1)),tuple(units))
        if (replacement_diagnostic.get('eligibility_reason') == 'eligible-acyclic-replacement'
                and replacement_diagnostic.get('rendering_reason') != 'rendered'):
            # The rule requires replacement for this proven graph. Do not let
            # a lower-priority substitutive candidate masquerade as the PIN
            # merely because this bounded renderer cannot spell it.
            _record_pending(
                _outcome_status, 'P-51.4.1.1', 'mandatory-replacement-renderer-unavailable',
                method='skeletal-replacement',
                eligibility_reason=replacement_diagnostic['eligibility_reason'],
                heterounit_count=replacement_diagnostic.get('heterounit_count'),
                rendering_reason=replacement_diagnostic.get('rendering_reason'))
            return None
    if groups and groups[0].kind=='alcohol' and len(groups)>1:
        carbon=[a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()==6]
        units=_alcohol_unit(mol,carbon[0]) if carbon else ()
        if units:
            name=min(units,key=lambda row:row[0])[1]
            carbon=set(carbon)
            ends=[i for i in carbon if sum(n.GetIdx() in carbon for n in mol.GetAtomWithIdx(i).GetNeighbors())<=1]
            path=_linear_order(mol,carbon,ends) if len(carbon)>1 else tuple(carbon)
            path=min((path,tuple(reversed(path))),key=lambda p:tuple(sorted(p.index(g.center) for g in groups)))
            if trace is not None:
                trace.append(dict(method='substitutive',eligible=True,rule='P-44.1.1',
                                  scope='all principal OH groups on one carbon parent'))
            return MethodDecision(name,'substitutive',centers,atoms,'P-44.1.1/P-51.1.1',
                                  tuple((i,str(j)) for j,i in enumerate(path,1)))
    if groups:
        # P-44.1.1 selects the complete applicable suffix set, then P-44.3
        # compares the eligible acyclic chain before P-45 prefix details.
        # The frontier ranks every structural maximum before rendering, so a
        # renderer failure cannot expose a shorter, lower-ranked parent.
        # This operation proves only the complete same-class carbon-chain
        # family exposed by complete_functional_parent; other classes retain
        # their existing bounded consumers below.
        frontier = functional_parent_frontier(mol, Chem)
        if frontier.required_parent_atoms:
            # P-44 has already selected a full-suffix ring over every
            # carbon-only chain in this graph-proved ring scope. P-45
            # multiplies that senior parent, not arbitrary smaller units.
            # Retain the physical frame until the shared mapped pool proves
            # its complete construction; an absent writer stays pending.
            _record_pending(
                _outcome_status, 'P-44.1.1/P-44.1.2/P-45.0/P-45.1.1',
                'source-proved-senior-ring-requires-complete-mapped-join',
                method='substitutive', deferred=True,
                required_parent_frontier=frontier,
            )
            return None
        if frontier.status in {"blocked", "conflict"}:
            _record_pending(
                _outcome_status, "P-44.1.1/P-44.3/P-45", 
                "complete-functional-parent-frontier-" + frontier.status,
                method="functional-class" if groups[0].kind == "ester" else "substitutive",
                frontier=frontier,
                maximal_skeletons=frontier.maximal_skeletons,
                frontier_reason=frontier.reason,
            )
            if trace is not None:
                trace.append(dict(method="functional-parent-frontier", status=frontier.status,
                                  rule="P-44.1.1/P-44.3/P-45", frontier=frontier))
            return None
        if frontier.status in {"complete", "tie"}:
            records = frontier.candidates
            valid = _valid_functional_parent_frontier(records, frontier, groups, mol)
            if not valid:
                _record_pending(
                    _outcome_status, "P-44.1.1/P-44.3/P-45",
                    "complete-functional-parent-frontier-incomplete-mapping",
                    method="functional-class" if groups[0].kind == "ester" else "substitutive",
                    frontier=frontier,
                    maximal_skeletons=frontier.maximal_skeletons,
                )
                if trace is not None:
                    trace.append(dict(method="functional-parent-frontier", status="blocked",
                                      rule="P-44.1.1/P-44.3/P-45", frontier=frontier))
                return None
            record = records[0]
            method = "functional-class" if groups[0].kind == "ester" else "substitutive"
            rule = "P-44.1.1/P-51.2.1" if method == "functional-class" else "P-44.1.1/P-51.1.1"
            if trace is not None:
                trace.append(dict(method="functional-parent-frontier", status=frontier.status,
                                  eligible=True, rule="P-44.1.1/P-44.3/P-45",
                                  frontier=frontier))
            # A full-coverage parent selected under P-44.1.1 remains senior
            # to splitting those same groups into multiplied parents. A
            # partial maximum may be compared with multiplication only when
            # the repeated unit is that same P-44-selected parent family.
            partial = frontier.suffix_coverage < len(centers)
            if partial and groups[0].kind in {"acid", "amide", "nitrile"}:
                # P-45.0 alternatives are based on the P-44-selected senior
                # parent. Disjoint copies need both their suffix groups and
                # their actual parent-skeleton atoms. The bounded graph
                # frontier supplies the complete equivalent senior frames.
                repetition_bound = _senior_parent_repetition_bound(
                    mol, frontier, len(centers),
                )
                if (repetition_bound is not None
                        and repetition_bound["maximum_copies"] < 2):
                    if trace is not None:
                        trace.append(dict(
                            method="multiplicative-method-comparison",
                            status="excluded-by-disjoint-parent-resource-bound",
                            eligible=False,
                            rule="P-44.1.1/P-44.3/P-45.0/P-45.1.1",
                            repetition_bound=repetition_bound,
                            evidence=("two identical selected parents require disjoint "
                                      "suffix groups and parent-skeleton atoms"),
                            frontier=frontier,
                        ))
                    return MethodDecision(record.name, method, record.principal_centers,
                                          atoms, rule, record.atom_locants,
                                          functional_parent_frontier=frontier)
                multiplicative = _multiplicative_parent_outcome(
                    mol, Chem, groups, atoms, senior_frontier=frontier,
                )
                if trace is not None:
                    trace.append(dict(
                        method="multiplicative-method-comparison",
                        status=multiplicative.status,
                        construction_complete=multiplicative.status == "complete",
                        rule="P-45.1.1/P-51.3.1/P-51.3.2.1",
                        reason=multiplicative.reason,
                        evidence=multiplicative.evidence,
                        graph_suffix_coverage=frontier.suffix_coverage,
                        coverage_authority="bounded graph maximum; not a full PIN certificate",
                        frontier=frontier,
                    ))
                if multiplicative.status == "complete":
                    return multiplicative.decision
                if multiplicative.status == "pending":
                    _record_pending(
                        _outcome_status, "P-45.1.1/P-51.3.1/P-51.3.2.1",
                        "partial-substitutive-frontier-multiplicative-comparison-unresolved",
                        method="multiplicative", deferred=True,
                        functional_parent_frontier=frontier,
                        multiplicative_outcome=multiplicative,
                        multiplicative_candidates=multiplicative.proposals,
                    )
                    return None
            # The mapped graph frontier may select a maximum-coverage parent
            # that carries only some same-class groups as suffixes. Omitted
            # groups remain typed mapped prefixes in the retained evidence.
            return MethodDecision(record.name, method, record.principal_centers, atoms, rule,
                                  record.atom_locants,
                                  functional_parent_frontier=frontier)
        if trace is not None:
            trace.append(dict(method="functional-parent-frontier", status="out_of_scope",
                              eligible=False, rule="P-44.1.1/P-44.3/P-45",
                              reason=frontier.reason))
        # P-44.1.1 precedes P-45.1: do not split a parent that already carries
        # all principal sites merely to create repeated smaller parents.
        if groups[0].kind in {"acid", "amide", "nitrile"}:
            complete = functional_parent_candidates(mol, Chem)
            proposals = tuple(
                candidate for candidate in complete
                if candidate.principal_centers == centers
            )
            selection = select_method_parent(
                proposals,
                lambda candidate: candidate.parent_selection,
                physical_parent_identity=lambda candidate: (
                    frozenset(candidate.parent_atoms) if candidate.parent_atoms else None
                ),
            )
            if trace is not None:
                trace.append(dict(
                    method="functional-parent-selection",
                    status=selection.status.value,
                    eligible=bool(proposals),
                    rule="P-44.1.1/P-44.3/P-45",
                    selection_frontier=selection,
                    deferred=selection.status == MethodParentStatus.UNRESOLVED,
                ))
            selected = selection.selected
            if selected is not None:
                method = "substitutive"
                return MethodDecision(selected.name, method, centers, atoms,
                    'P-44.1.1/P-51.1.1', selected.atom_locants)
            if selection.status == MethodParentStatus.UNRESOLVED:
                _record_pending(
                    _outcome_status, "P-44.1.1/P-44.3/P-45",
                    "generic-functional-parent-selection-unresolved",
                    method="substitutive", deferred=True,
                    selection_frontier=selection,
                )
                # The principal-selection layer owns the broader shared pool
                # that can discharge this bounded producer's unresolved facts.
                return None
    multiplicative = _multiplicative_parent_outcome(mol, Chem, groups, atoms)
    multiplied = multiplicative.decision
    if multiplicative.status == 'pending':
        _record_pending(_outcome_status, 'P-51.3.1/P-51.3.2.1',
                        multiplicative.reason, method='multiplicative',
                        multiplicative_outcome=multiplicative)
    if trace is not None:
        trace.append(dict(method='multiplicative',eligible=multiplied is not None,
            status=multiplicative.status,
            rule='P-51.3.1/P-51.3.2.1',scope='identical rooted units and symmetric noncentral connector blocks'))
    if multiplied is not None:return multiplied
    ordinary = _halogenated_linear_ether_substitutive(mol, groups, atoms)
    if ordinary is None:
        ordinary = _linear_substitutive(mol, Chem, groups, atoms)
    return ordinary


def _valid_functional_parent_frontier(records, frontier, groups, mol):
    """Check maximum suffix coverage and the complete original-atom map."""
    centers = frozenset(group.center for group in groups)
    maximum = frontier.suffix_coverage
    if (not isinstance(maximum, int) or isinstance(maximum, bool)
            or maximum < 1 or maximum > len(centers) or not records):
        return False
    if len({record.name for record in records}) != 1:
        return False
    group_atoms = {group.center: frozenset(group.atoms) for group in groups}
    for record in records:
        selected = record.principal_centers
        omitted = record.omitted_group_centers
        if (len(selected) != maximum or selected & omitted
                or selected | omitted != centers):
            return False
        if len(dict(record.atom_locants)) != len(record.atom_locants):
            return False
        construction = record.construction
        if (construction is None
                or not fully_covered(mol, construction.atoms,
                                     construction.embedded_atoms)):
            return False
        if not (selected | omitted) <= (
                construction.atoms | construction.embedded_atoms):
            return False

        owner_rows = tuple(record.omitted_prefix_atom_owners)
        owner_centers = tuple(center for center, _atoms in owner_rows)
        if (len(owner_centers) != len(set(owner_centers))
                or frozenset(owner_centers) != omitted):
            return False
        for center, owned_atoms in owner_rows:
            owned_atoms = frozenset(owned_atoms)
            if (center not in group_atoms or not group_atoms[center] <= owned_atoms
                    or not owned_atoms
                    or not owned_atoms <= construction.embedded_atoms
                    or owned_atoms & construction.atoms):
                return False
        if omitted and record.suffix_mode not in {"embedded", "external"}:
            return False
    return True


def _side(mol, root, blocked):
    return frozenset(collect_subtree(mol, root, blocked))


def _annotated_method(mol,Chem,trace,status_sink=None):
    """P-51 eligibility with independently mapped P-82/P-92 decoration.

    The constitutional copy proposes a method, never proves repeated-unit
    equivalence. Multiplicative identity is rechecked on the labelled graph.
    Original CIP labels are used only by the final mapped component writer.
    """
    if any(bond.GetStereo() in {Chem.BondStereo.STEREOATROPCW,Chem.BondStereo.STEREOATROPCCW}
           for bond in iter_bonds(mol)):
        _record_pending(status_sink,'P-93.5.7.1',
                        'assigned-axis-requires-mapped-axial-construction',method='axial-configuration')
        return None
    from .component_mapping import hydrogen_bindings,fully_covered
    base=Chem.Mol(mol);Chem.RemoveStereochemistry(base)
    for atom in iter_atoms(base):atom.SetIsotope(0)
    base_pending=[]
    base_trace=[]
    choice=select_naming_method(base,Chem,trace=base_trace,_outcome_status=base_pending)
    if trace is not None:
        trace.extend(base_trace)
    if choice is None:
        if status_sink is not None:
            status_sink.extend(base_pending)
        return None
    if any(role[0] == 'group14-parent' for role in choice.component_roles):
        # The constitutional P-68.2 parent is not an ordinary carbon ether.
        # Revalidate its own original-graph producer before transferring any
        # annotations. An unsupported branch must not be renumbered/decorated
        # by the unrelated generic substitutive operation below.
        from .group14_mononuclear_methods import group14_mononuclear_proposal
        original = group14_mononuclear_proposal(mol, Chem)
        return original.as_method_decision() if original is not None else None
    if status_sink is not None:
        # A usable constitutional method is not a completed preference proof.
        status_sink.extend(base_pending)
    inventory=selection_inventory(base);groups=inventory.principal if inventory.supported else ()
    centers=frozenset(g.center for g in groups)
    atoms=frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()>1)
    result=None
    if choice.method=='skeletal-replacement':
        replacement=_replacement_parent(base,Chem,groups,evidence=True)
        if isinstance(replacement, AcylReplacementProof):
            proof = replacement
            from .acyl_replacement_annotations import decorate_acyl_replacement_proof
            annotated = decorate_acyl_replacement_proof(mol, Chem, proof)
            annotated_decision = annotated.decision if annotated is not None else None
            if annotated_decision is None:
                from .acyl_suffix_isotopes223 import decorate_fixed_suffix_isotope_proof
                annotated_decision = decorate_fixed_suffix_isotope_proof(mol, Chem, proof)
                annotated = annotated_decision
            if annotated_decision is not None:
                if trace is not None:
                    trace.append(dict(
                        method='skeletal-replacement', status='complete',
                        eligible=True, rule=annotated_decision.rule,
                        scope='original mapped fixed parent; typed acyl suffix',
                        acyl_replacement_proof=proof,
                        annotation_evidence=annotated,
                    ))
                from .selection_evidence import acyl_replacement_justification
                return replace(annotated_decision, justification=acyl_replacement_justification(
                    mol, Chem, proof, annotated_decision))
            # A required fixed parent with an unproved suffix-annotation role
            # remains pending; another method cannot hide the missing proof.
            _record_pending(
                status_sink, proof.decision.rule + '/P-82',
                'annotated-acyl-replacement-transfer-unavailable',
                method='skeletal-replacement', acyl_replacement_proof=proof,
            )
            if trace is not None:
                trace.append(dict(method='skeletal-replacement', status='pending',
                                  rule=proof.decision.rule + '/P-82',
                                  reason='annotated-acyl-replacement-transfer-unavailable',
                                  acyl_replacement_proof=proof))
            return None
        if replacement:
            name,path,units=replacement;chemical=(_replacement_key(base,path),tuple(sorted(path.index(g.center) for g in groups)))
            candidates=[]
            for order in (path,tuple(reversed(path))):
                if (_replacement_key(base,order),tuple(sorted(order.index(g.center) for g in groups)))!=chemical:continue
                mapping=hydrogen_bindings(mol,{idx:str(j) for j,idx in enumerate(order,1)})
                embedded={a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()>1 or a.GetIsotope()}-mapping.keys()
                if any(mol.GetAtomWithIdx(i).GetIsotope() or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in embedded):continue
                value=decorate_component(mol,Chem,MappedComponent(name,tuple(mapping.items()),frozenset(mapping),embedded_atoms=frozenset(embedded),systematic_parent=True))
                # Preserve the ordinary P-44/P-45 numbering criteria before
                # applying the equivalent-numbering isotope key. Every
                # external atom belonging to the principal suffix is excluded
                # from ordinary prefixes; all other side components must have
                # a proved mapped-prefix rendering for this comparator.
                from .parent_selection import graph_parent_facts
                group_external={i for group in groups for i in group.atoms if i not in mapping}
                prefixes=[]
                prefixes_supported=True
                for idx in order:
                    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                        root=neighbor.GetIdx()
                        if root in mapping or neighbor.GetAtomicNum()==1 or root in group_external:
                            continue
                        text,owned=mapped_branch(mol,Chem,root,idx)
                        if not text or owned & (mapping.keys() | set(group_external)):
                            prefixes_supported=False
                            break
                        prefixes.append((mapping[idx],text))
                    if not prefixes_supported:break
                if not prefixes_supported:
                    continue
                # These are two numberings of the same fixed replacement
                # parent. graph_parent_facts compares all ordinary parent,
                # suffix, replacement, and prefix facts before P-82.5.2.
                facts=graph_parent_facts(
                    mol,order,mapping,
                    principal_class=groups[0].seniority if groups else (40,0),
                    kind='chain',
                    replacement=True,suffix_centers=centers,
                    suffix_locants=tuple(str(order.index(g.center)+1) for g in groups),
                    prefixes=tuple(prefixes),name=value,allow_annotations=True)
                if value and facts is not None:
                    candidates.append((facts,value,mapping))
            if candidates:
                from .parent_selection import parent_frontier
                frontier=parent_frontier(candidates,lambda row:row[0])
                if len(frontier)==1 or all(row[1]==candidates[0][1] for row in candidates):
                    _,name,mapping=frontier[0] if len(frontier)==1 else candidates[0]
                    result=MethodDecision(name,'skeletal-replacement',frozenset(g.center for g in groups),atoms,
                        'P-51.4.1.1',tuple(mapping.items()),tuple(units))
    elif choice.method == 'multiplicative':
        chiral = {a.GetIdx() for a in iter_atoms(mol) if a.GetChiralTag()}
        stereo_bonds = tuple(b for b in iter_bonds(mol) if b.GetStereo())
        azo_connector = frozenset()
        if stereo_bonds:
            bond = stereo_bonds[0]
            if (not groups or len(stereo_bonds) != 1 or chiral or bond.IsInRing()
                    or bond.GetBondTypeAsDouble() != 2
                    or any(atom.GetAtomicNum() != 7 for atom in (bond.GetBeginAtom(), bond.GetEndAtom()))
                    or any(a.GetIsotope() for a in iter_atoms(mol))):
                return None
            azo_connector = frozenset({bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()})
        elif chiral and (not groups or groups[0].kind != 'acid'
                         or not all(mol.GetAtomWithIdx(i).IsInRing() for i in chiral)):
            return None
        multiplicative = _multiplicative_parent_outcome(
            mol, Chem, groups, atoms, required_stereo_connector=chiral,
            required_azo_connector=azo_connector)
        result = multiplicative.decision
        if multiplicative.status == 'pending':
            _record_pending(status_sink, 'P-51.3.1/P-82', multiplicative.reason,
                            method='multiplicative', multiplicative_outcome=multiplicative)
    elif choice.method=='substitutive' and not groups:
        # Ordinary below-threshold linear ethers: use the common P-44/P-45
        # comparator for labelled alternatives to the same constitutional
        # method. An isotope may make a different carbon block senior.
        from .parent_selection import graph_parent_facts,compare_parents,parent_frontier
        blocks=[];remaining={i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum()==6}
        while remaining:
            block=set();stack=[next(iter(remaining))]
            while stack:
                i=stack.pop()
                if i not in remaining:continue
                remaining.remove(i);block.add(i)
                stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in remaining)
            blocks.append(block)
        candidates=[]
        for block in blocks:
            ends=[i for i in block if sum(n.GetIdx() in block for n in mol.GetAtomWithIdx(i).GetNeighbors())<=1]
            path=_linear_order(mol,block,ends) if len(block)>1 else tuple(block)
            if not path or len(path) not in ROOTS:continue
            for order in (path,tuple(reversed(path))):
                prefixes=[];embedded=set();mapping=hydrogen_bindings(mol,{i:str(j) for j,i in enumerate(order,1)})
                for i in order:
                    for n in mol.GetAtomWithIdx(i).GetNeighbors():
                        root=n.GetIdx()
                        if root in mapping or n.GetAtomicNum()==1:continue
                        text,owned=mapped_branch(mol,Chem,root,i)
                        if not text or owned&(mapping.keys()|embedded):break
                        prefixes.append((mapping[i],text));embedded.update(owned)
                    else:continue
                    break
                else:
                    if not fully_covered(mol,mapping,embedded):continue
                    parent=ROOTS[len(order)]+'ane'
                    value=decorate_component(mol,Chem,MappedComponent(
                        parent,tuple(mapping.items()),frozenset(mapping),tuple(prefixes),
                        frozenset(embedded),omit_methyl_prefix_locant=len(order)==1))
                    facts=graph_parent_facts(mol,order,mapping,principal_class=(40,0),prefixes=prefixes,name=value,allow_annotations=True)
                    if value and facts is not None:candidates.append((facts,value,mapping))
        for candidate in parent_frontier(candidates,lambda c:c[0]):
            if all(c[1]==candidate[1] or compare_parents(candidate[0],c[0]).preference in {-1,0} for c in candidates):
                result=MethodDecision(candidate[1],'substitutive',frozenset(),atoms,'P-51.1.1/P-51.4.1.1',tuple(candidate[2].items()));break
    if result is None and choice.method=='skeletal-replacement':
        _record_pending(status_sink, 'P-51.4.1.1/P-82',
                        'mandatory-replacement-annotation-renderer-unavailable',
                        method='skeletal-replacement',
                        base_method=choice.method)
    if trace is not None:trace.append(dict(method=choice.method,eligible=result is not None,
        rule=choice.rule,scope='original mapped annotations; labelled repeated-unit identity rechecked',
        base_method=choice.method,base_evidence=tuple(base_trace)))
    return result


def _fragment(mol, Chem, atoms):
    """Hydrogen-cap a bridge cut, retaining a separate input identity map."""
    order = sorted(atoms)
    editable = Chem.RWMol(mol)
    # Bracket atoms (including skeletal isotopes) have fixed explicit H
    # counts. Removing a bridge does not let RDKit replenish their valence.
    # Cap the cut on the proposal copy just as implicit-H atoms are capped;
    # the original graph remains the source of labels and atom ownership.
    for idx in order:
        atom = editable.GetAtomWithIdx(idx)
        lost = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()
                   if b.GetOtherAtomIdx(idx) not in atoms)
        if atom.GetAtomicNum() > 1 and atom.GetNoImplicit() and lost:
            if not float(lost).is_integer():
                return None, ()
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + int(lost))
    for idx in reversed(range(mol.GetNumAtoms())):
        if idx not in atoms:
            editable.RemoveAtom(idx)
    fragment = editable.GetMol()
    try:
        Chem.SanitizeMol(fragment)
    except Exception:
        return None, ()
    return fragment, tuple(order)


def _ring_units(fragment, Chem, root):
    rings = fragment.GetRingInfo().AtomRings()
    if len(rings) != 1 or root not in rings[0]:
        return ()
    ring = rings[0]
    if any(fragment.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring):
        return ()
    aromatic = all(fragment.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    if aromatic and len(ring) != 6 or len(ring) not in ROOTS:
        return ()
    if not aromatic and any(fragment.GetBondBetweenAtoms(
        ring[j], ring[(j + 1) % len(ring)]).GetBondTypeAsDouble() != 1
        for j in range(len(ring))):
        return ()
    parent = "benzene" if aromatic else "cyclo" + ROOTS[len(ring)] + "ane"
    results = []
    for direction in (ring, tuple(reversed(ring))):
        for start in range(len(ring)):
            path = direction[start:] + direction[:start]
            mapping = {i: str(j) for j, i in enumerate(path, 1)}
            embedded, prefixes = set(), []
            for i in path:
                for neighbor in fragment.GetAtomWithIdx(i).GetNeighbors():
                    n = neighbor.GetIdx()
                    if n in mapping or neighbor.GetAtomicNum() <= 1:
                        continue
                    name, owned = mapped_branch(fragment, Chem, n, i)
                    if not name or owned & (set(mapping) | embedded):
                        break
                    embedded.update(owned)
                    prefixes.append((mapping[i], name))
                else:
                    continue
                break
            else:
                if set(mapping) | embedded != set(range(fragment.GetNumAtoms())):
                    continue
                name = decorate_component(fragment, Chem, MappedComponent(
                    parent, tuple(mapping.items()), frozenset(mapping),
                    tuple(prefixes), frozenset(embedded)))
                prefix_locs = tuple(sorted([int(loc) for loc, _ in prefixes]
                                           + [int(mapping[root])]))
                from .parent_selection import graph_parent_facts
                facts = graph_parent_facts(
                    fragment, path, mapping, principal_class=(40, 0),
                    suffix_centers=(), suffix_locants=(), prefixes=prefixes,
                    kind="ring", name=name,
                )
                ring_atoms = frozenset(path)
                ring_bonds = tuple(sorted(
                    (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                     max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                     b.GetBondTypeAsDouble())
                    for b in iter_bonds(fragment)
                    if b.GetBeginAtomIdx() in ring_atoms
                    and b.GetEndAtomIdx() in ring_atoms
                ))
                frame_identity = ("mapped-carbon-monocycle", ring_atoms, ring_bonds)
                results.append((prefix_locs, int(mapping[root]), name,
                                mapping[root], bool(prefixes), facts,
                                frame_identity))
    return tuple(results)


def _same_frame_linker_numbering(rows, *, facts_index=5, locant_index=2,
                                  identity_index=6):
    """Choose local parent numbering, then the P-15.3.1.3 linker locant.

    This operates only inside an explicitly mapped physical parent frame. The
    linker position is never added to P-44 facts; unknown local numbering
    comparisons leave the construction unresolved.
    """
    identities = [row[identity_index] for row in rows]
    if not identities or identities[0] is None or any(
        identity != identities[0] for identity in identities
    ):
        return None, "unit-parent-frame-identity-unproved"
    from .parent_selection import compare_parents, locant_key

    edges = {index: set() for index in range(len(rows))}
    unknown = set()
    comparisons = {}
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            comparison = compare_parents(rows[left][facts_index],
                                         rows[right][facts_index])
            comparisons[(left, right)] = comparison.preference
            if comparison.preference == -1:
                edges[left].add(right)
            elif comparison.preference == 1:
                edges[right].add(left)
            elif comparison.preference is None:
                unknown.add((left, right))
    if any(_has_parent_preference_cycle(edges, index, set(), set())
           for index in edges):
        return None, "same-frame-parent-preference-cycle"
    dominated = {junior for seniors in edges.values() for junior in seniors}
    survivors = [index for index in range(len(rows)) if index not in dominated]
    if any(left in survivors and right in survivors for left, right in unknown):
        return None, "same-frame-parent-numbering-unknown"
    for index, left in enumerate(survivors):
        for right in survivors[index + 1:]:
            if comparisons[(left, right)] != 0:
                return None, "same-frame-parent-preferences-inconsistent"
    keyed = [(locant_key(rows[index][locant_index]), index)
             for index in survivors]
    if any(key is None for key, _index in keyed):
        return None, "same-frame-linker-locant-unmapped"
    minimum = min(key for key, _index in keyed)
    return [rows[index] for key, index in keyed if key == minimum], \
        "P-15.3.1.3-lowest-linker-locant"


def _rebase_fragment_parent_facts(facts, local_to_original):
    """Move fragment-local identity evidence into the source molecule namespace.

    ParentFacts contain locant evidence as well as physical atom references.
    Only the latter are rebased: two independently cut units can have identical
    local atom numbers while being different physical parents.
    """
    if facts is None:
        return None
    local_to_original = tuple(local_to_original)

    def atom(index):
        if not isinstance(index, int) or not 0 <= index < len(local_to_original):
            return None
        return local_to_original[index]

    scope = facts.numbering_scope
    if isinstance(scope, tuple) and len(scope) == 4 \
            and scope[0] == "unannotated-neutral-carbon-chain":
        _kind, atoms, centers, bonds = scope
        mapped_atoms = frozenset(atom(index) for index in atoms)
        mapped_centers = frozenset(atom(index) for index in centers)
        mapped_bonds = []
        for left, right, order in bonds:
            mapped_left, mapped_right = atom(left), atom(right)
            if mapped_left is None or mapped_right is None:
                mapped_bonds = None
                break
            mapped_bonds.append((min(mapped_left, mapped_right),
                                 max(mapped_left, mapped_right), order))
        if (None in mapped_atoms | mapped_centers or mapped_bonds is None):
            scope = None
        else:
            scope = (_kind, mapped_atoms, mapped_centers,
                     tuple(sorted(mapped_bonds)))
    elif isinstance(scope, tuple) and len(scope) == 2:
        atoms, centers = scope
        mapped_atoms = frozenset(atom(index) for index in atoms)
        mapped_centers = frozenset(atom(index) for index in centers)
        scope = None if None in mapped_atoms | mapped_centers else (
            mapped_atoms, mapped_centers,
        )

    identity = facts.stereo_identity
    if identity is not None:
        mapped_records = []
        for record in identity:
            if not isinstance(record, tuple) or not record:
                mapped_records = None
                break
            if record[0] == "atom" and len(record) == 4:
                idx = atom(record[1])
                if idx is None:
                    mapped_records = None
                    break
                mapped_records.append((record[0], idx, *record[2:]))
            elif record[0] == "bond" and len(record) == 4:
                # Bond indices are fragment-local too. The mapped endpoints
                # identify the physical bond; preserve endpoint locants and
                # assigned stereo so the identity remains source complete.
                ends = tuple(sorted((atom(index), locant)
                                    for index, locant in record[2]))
                if any(index is None for index, _locant in ends):
                    mapped_records = None
                    break
                mapped_records.append((record[0], ends, record[3]))
            else:
                mapped_records = None
                break
        identity = None if mapped_records is None else tuple(sorted(mapped_records))
    return replace(facts, numbering_scope=scope, stereo_identity=identity)


@dataclass(frozen=True, slots=True)
class _UnitParentComparison:
    selected: tuple | None = None
    reason: str = ""
    locant: str = ""
    facts: tuple = ()


@dataclass(frozen=True, slots=True)
class _CachedUnitParents:
    rows: tuple
    comparison: _UnitParentComparison | None
    immutable: bool


def _immutable_unit_parent_rows(value):
    """Only plain immutable values may suppress a later fact comparison."""
    value_type = type(value)
    if (value is None or value_type is bool or value_type is int
            or value_type is float or value_type is str or value_type is bytes):
        return True
    if value_type is tuple or value_type is frozenset:
        return all(_immutable_unit_parent_rows(item) for item in value)
    from .parent_selection import ParentFacts
    if value_type is ParentFacts:
        return all(_immutable_unit_parent_rows(getattr(value, name))
                   for name in ParentFacts.__dataclass_fields__)
    return False


def _compare_unit_parent_rows(possibilities):
    """Pure P-44/linker comparison in the unchanged fragment namespace."""
    if not possibilities:
        return _UnitParentComparison()
    from .parent_selection import parent_seniority_frontier

    comparisons = {}
    best = parent_seniority_frontier(
        possibilities, lambda row: row[5], comparisons=comparisons)
    best_comparisons = [
        comparisons[(id(left[5]), id(right[5]))].preference
        for index, left in enumerate(best)
        for right in best[index + 1:]
    ]
    best_edges = {index: set() for index in range(len(best))}
    pair_index = 0
    for left in range(len(best)):
        for right in range(left + 1, len(best)):
            preference = best_comparisons[pair_index]
            pair_index += 1
            if preference == -1:
                best_edges[left].add(right)
            elif preference == 1:
                best_edges[right].add(left)
    best_cycle = any(
        _has_parent_preference_cycle(best_edges, index, set(), set())
        for index in best_edges
    )
    if any(value is None for value in best_comparisons) or best_cycle:
        return _UnitParentComparison(
            reason="unit-parent-p44-facts-incomparable",
            facts=tuple(row[5] for row in best))
    renderings = {(row[1], row[2], row[3], row[4]) for row in best}
    if len(renderings) != 1:
        selected, local_reason = _same_frame_linker_numbering(best)
        if selected is not None:
            best = selected
            renderings = {(row[1], row[2], row[3], row[4]) for row in best}
        else:
            local_reason = "same-source-rank-unit-parent-renderings:" + local_reason
        if len(renderings) != 1:
            return _UnitParentComparison(
                reason=local_reason, locant=str(best[0][2]),
                facts=tuple(row[5] for row in best))
    return _UnitParentComparison(selected=best[0])


def _unit_parent_possibilities(fragment, Chem, root, kind):
    # Retain the raw-row contract for callers that do not request comparison.
    return _unit_parent_cache_entry(fragment, Chem, root, kind).rows


def _unit_parent_cache_entry(fragment, Chem, root, kind, *, compare=False, defer_render=False):
    # Serialize complete native properties as well as atom/bond order. Capped
    # fragments can inherit CIP assignments from different enclosing graphs;
    # equal connectivity alone would not make their renderings interchangeable.
    key = (Chem, fragment.ToBinary(Chem.PropertyPickleOptions.AllProps), root, kind)
    with _UNIT_PARENT_CACHE_LOCK:
        cached = _UNIT_PARENT_CACHE.get(key)
        if cached is not None:
            _UNIT_PARENT_CACHE.move_to_end(key)
    if cached is not None:
        if not compare or cached.comparison is not None:
            return cached
        comparison = _compare_unit_parent_rows(cached.rows)
        resolved = replace(cached, comparison=comparison)
        if cached.immutable:
            with _UNIT_PARENT_CACHE_LOCK:
                if _UNIT_PARENT_CACHE.get(key) is cached:
                    _UNIT_PARENT_CACHE[key] = resolved
        return resolved
    if kind == 'alcohol':
        possibilities = list(_alcohol_unit(fragment, root))
    elif kind is not None:
        from . import complete_functional_parent as functional
        from rdkit import Chem as native_Chem
        if (defer_render and Chem is native_Chem and not fragment.NeedsUpdatePropertyCache()
                and not any(atom.GetIsotope() or int(atom.GetChiralTag())
                            for atom in iter_atoms(fragment))
                and not any(int(bond.GetStereo()) or int(bond.GetBondDir())
                            for bond in iter_bonds(fragment))
                and not fragment.GetStereoGroups()
                and functional_parent_candidates is _DEFAULT_FUNCTIONAL_PARENT_CANDIDATES
                and functional._render is _DEFAULT_FUNCTIONAL_PARENT_RENDER
                and functional._render_functional_candidate is _DEFAULT_UNIT_PARENT_RENDER
                and mol_cache_get(fragment, 'candidate_telemetry') is None):
            # A repeated unit must contain the bridge's linker root in its
            # parent. Every plan fixes exactly the parent_atoms later returned
            # by its renderer. An entirely root-free pool therefore produces
            # no admissible unit, including when every rendering fails.
            # If any plan can contain the root, execute the entire original
            # pool in enumeration order: discarded renderings may initialize
            # native stereo facts consumed by a later eligible rendering.
            from .complete_functional_parent import FunctionalParentPlan
            plans = functional_parent_candidates(
                fragment, Chem, include_single=True, defer_render=True)
            records = []
            if any(root in record.parent_atoms for record in plans):
                for record in plans:
                    if isinstance(record, FunctionalParentPlan):
                        record = record.construct()
                        if record is None:
                            continue
                    records.append(record)
                # Deferred enumeration precedes the eager numbered-record
                # stable sort; retain the historical first representative.
                records.sort(key=lambda record: record.numbering_key)
        else:
            records = functional_parent_candidates(fragment, Chem, include_single=True)
        possibilities = []
        for record in records:
            loc = dict(record.atom_locants).get(root, "")
            if not loc.isdigit() or root not in record.parent_atoms:
                continue
            facts = record.parent_selection
            # The unit here is the parent structure. Its connection
            # to the central multiplicative group is not a P-44.4.1.9
            # point of attachment of a substituent group. Keep the
            # linker locant for rendering, but do not inject it into
            # parent-seniority facts.
            # Do not prefilter parent alternatives by a local total
            # rank before comparing their typed P-44 facts.
            identity = (("functional-parent-frame",
                         facts.numbering_scope)
                        if facts is not None and facts.numbering_scope is not None
                        else None)
            possibilities.append(((), record.component_name, loc,
                                  bool(record.prefixes), record.ester_owners,
                                  facts, identity, frozenset(record.parent_atoms)))
    else:
        possibilities = [
            ((), parent, loc, substituted, (), facts, identity)
            for (_prefix_locs, _link_rank, parent, loc, substituted,
                 facts, identity) in _ring_units(fragment, Chem, root)
        ]
    if kind is not None and fragment.GetAtomWithIdx(root).IsInRing():
        from .mapped_ring_parents import parent_facts, records

        groups = selection_inventory(fragment).principal
        if groups and all(group.kind == kind for group in groups):
            for row in records(fragment, Chem, principal=groups, general_branches=True):
                locant = dict(row.atom_locants).get(root)
                if (not locant or root not in row.parent_atoms
                        or row.principal_centers != frozenset(g.center for g in groups)):
                    continue
                facts = parent_facts(fragment, row, groups[0].seniority)
                identity = ('functional-parent-frame', facts.numbering_scope) if facts else None
                possibilities.append(((), row.name, locant, bool(row.prefixes), (), facts,
                                      identity, frozenset(row.parent_atoms)))
    rows = tuple(possibilities)
    # Only the already-cached fragment-local row values enter this pure
    # comparison. Native fragment validation and source-graph rebasing stay
    # outside the cache; opaque/custom values retain fresh comparisons.
    immutable = _immutable_unit_parent_rows(rows)
    comparison = _compare_unit_parent_rows(rows) if compare else None
    cached = _CachedUnitParents(rows, comparison if immutable else None, immutable)
    with _UNIT_PARENT_CACHE_LOCK:
        _UNIT_PARENT_CACHE[key] = cached
        _UNIT_PARENT_CACHE.move_to_end(key)
        if len(_UNIT_PARENT_CACHE) > 512:
            _UNIT_PARENT_CACHE.popitem(last=False)
    return cached if immutable else replace(cached, comparison=comparison)


def _parent_units(mol, Chem, groups, *, unresolved=None):
    centers = {g.center for g in groups}
    # The existing cut admission below requires a nonempty proper subset of
    # principal centers on either side. With one distinct center no cut can
    # qualify, regardless of chain length or the number of attached groups.
    if groups and len(centers) < 2:
        return
    seen = set()
    for bond in iter_bonds(mol):
        if bond.IsInRing() or bond.GetBondTypeAsDouble() != 1:
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for root, blocked in ((a, b), (b, a)):
            if mol.GetAtomWithIdx(root).GetAtomicNum() != 6:
                continue
            if not groups and not mol.GetAtomWithIdx(root).IsInRing():
                continue  # Acyclic hydrocarbons cannot be multiplied.
            atoms = _side(mol, root, blocked)
            if (root, atoms) in seen:
                continue
            seen.add((root, atoms))
            contained = centers & atoms
            if groups and (not contained or contained == centers):
                continue
            if any(bool(set(g.atoms) & atoms) != (g.center in atoms)
                   or g.center in atoms and not set(g.atoms) <= atoms for g in groups):
                continue
            fragment, original = _fragment(mol, Chem, atoms)
            if fragment is None:
                continue
            local_root = original.index(root)
            # Rooted isomorphism includes all substituents and their attachment
            # positions. Rendered spelling is not an identity test.
            for atom in iter_atoms(fragment):
                atom.SetAtomMapNum(0)
            fragment.GetAtomWithIdx(local_root).SetAtomMapNum(1)
            signature = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
            fragment.GetAtomWithIdx(local_root).SetAtomMapNum(0)
            pool = _unit_parent_cache_entry(
                fragment, Chem, local_root, groups[0].kind if groups else None,
                compare=True, defer_render=mol_cache_get(mol, 'candidate_telemetry') is None)
            comparison = pool.comparison
            if comparison.reason:
                if unresolved is not None:
                    unresolved.append(((signature, comparison.locant, len(contained)),
                                       comparison.reason, comparison.facts))
                continue
            if comparison.selected is not None:
                source_rank, parent, loc, substituted, owners, facts = comparison.selected[:6]
                parent_atoms = (frozenset(original[i] for i in comparison.selected[7])
                                if len(comparison.selected) > 7 else frozenset())
                facts = _rebase_fragment_parent_facts(facts, original)
                yield _Unit(atoms, root, blocked, signature, parent, loc,
                            frozenset(contained), source_rank, substituted, owners,
                            facts, parent_atoms)


def _linear_order(mol, atoms, endpoints):
    if not atoms or len(endpoints) != 2:
        return ()
    start, finish = endpoints
    path, previous = [], None
    current = start
    while current not in path:
        path.append(current)
        neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(current).GetNeighbors()
                     if a.GetIdx() in atoms and a.GetIdx() != previous]
        if not neighbors:
            return tuple(path) if current == finish and set(path) == set(atoms) else ()
        if len(neighbors) != 1:
            return ()
        previous, current = current, neighbors[0]
    return ()


def _isolated_heterounits(mol, path):
    """Count proved units, never individual atoms in an unknown heterogroup."""
    units=[]
    for carbon,items in groupby(path,key=lambda i:mol.GetAtomWithIdx(i).GetAtomicNum()==6):
        values=tuple(items)
        if carbon:continue
        elements=tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in values)
        if len(values)==1 or elements in {(8,8),(16,16),(16,8,16)}:
            units.append(values)
        else:
            return None
    return units


def _replacement_key(mol, path):
    return (
        tuple(j for j, i in enumerate(path, 1) if mol.GetAtomWithIdx(i).GetAtomicNum() != 6),
        tuple(tuple(j for j, i in enumerate(path, 1)
                    if mol.GetAtomWithIdx(i).GetAtomicNum() == element)
              for element in (8, 16, 7)),
    )


def _replacement_text(mol, path):
    """P-15.4 fixed heteroatom numbering; prefixes follow element seniority."""
    parts = []
    for element, term in ((8, "oxa"), (16, "thia"), (7, "aza")):
        locs = [str(j) for j, i in enumerate(path, 1)
                if mol.GetAtomWithIdx(i).GetAtomicNum() == element]
        if locs:
            parts.append(",".join(locs) + "-" + (MULT[len(locs)] if len(locs) > 1 else "") + term)
    return "-".join(parts) + ROOTS[len(path)] + "ane"


def _replacement_parent(mol, Chem, groups, *, evidence=False, diagnostic=None):
    if diagnostic is not None:
        diagnostic.update(eligibility_reason='producer-scope-declined',
                          rendering_reason='not-rendered')
    from .replacement_eligibility import acyclic_replacement_eligibility

    if mol.GetRingInfo().NumRings():
        return ""
    skipped, owners = set(), []
    centers = {g.center for g in groups}
    for g in groups:
        skipped.update(set(g.atoms) - {g.center})
        if g.kind in {"amide", "ester"}:
            for neighbor in mol.GetAtomWithIdx(g.ligand).GetNeighbors():
                if neighbor.GetIdx() == g.center or neighbor.GetAtomicNum() <= 1:
                    continue
                text, atoms = mapped_branch(mol, Chem, neighbor.GetIdx(), g.ligand)
                if not text or atoms & centers:
                    return ""
                skipped.update(atoms)
                if g.kind == "ester":
                    owners.append((text, atoms))
    atoms = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1} - skipped
    if (len(atoms) >= 7 and 1 <= len(groups) <= 2
            and groups[0].kind in {"acid", "amide", "nitrile"}
            and sum(mol.GetAtomWithIdx(i).GetAtomicNum() in {7, 8, 16} for i in atoms) >= 4):
        from .acyl_replacement_methods import replacement_leaf_prefix

        atoms = {idx for idx in atoms if idx in centers or not replacement_leaf_prefix(mol, idx)}
    principal_group_atoms = {
        i for group in groups for i in group.atoms
        if mol.GetAtomWithIdx(i).GetAtomicNum() != 6
    }
    proof = acyclic_replacement_eligibility(
        mol, atoms, principal_group_atoms=principal_group_atoms,
        has_principal_suffix=bool(groups)
    )
    if diagnostic is not None:
        diagnostic['eligibility_reason'] = proof.reason
        diagnostic['heterounit_count'] = proof.heterounit_count
    if not proof.eligible:
        return ""
    # Eligibility follows P-51.4.1.1–1.4 and can cover more elements and
    # named multivalent units than this renderer knows how to spell. Keep the
    # producer bounded: unsupported element/parent vocabulary simply declines.
    if len(atoms) not in ROOTS or any(
        mol.GetAtomWithIdx(i).GetAtomicNum() not in _LINK_ELEMENTS for i in atoms
    ):
        if diagnostic is not None:
            diagnostic['rendering_reason'] = 'parent-vocabulary-or-element-unsupported'
        return ""
    if len(groups) > 2 and groups[0].kind != 'alcohol':
        if diagnostic is not None:
            diagnostic['rendering_reason'] = 'multi-suffix-parent-renderer-unsupported'
        return ""
    path = proof.path
    if len(groups) in {1, 2} and groups[0].kind in {"acid", "amide", "nitrile"}:
        from .acyl_replacement_methods import acyl_replacement_decision
        acyl_proof = None
        for proposed_path in (path, tuple(reversed(path))):
            acyl_proof = acyl_replacement_decision(
                mol, Chem, groups, proposed_path, return_proof=True,
            )
            if acyl_proof is not None:
                break
        if acyl_proof is not None:
            if diagnostic is not None:
                diagnostic['rendering_reason'] = 'rendered-acyl-replacement-suffix'
            return acyl_proof if evidence else acyl_proof.decision.name
        if len(groups) == 2:
            # P-51.4.1.3 requires the full substitutive suffix-group count.
            # A failed mapped two-site construction cannot authorize the
            # generic renderer to omit the typed coverage/ownership proof.
            if diagnostic is not None:
                diagnostic['rendering_reason'] = 'multi-acyl-suffix-mapped-proof-unsupported'
            return ""
    if len(groups) == 1 and groups[0].kind in {
            "acid", "carboxylate", "amide", "ester", "nitrile"}:
        if diagnostic is not None:
            diagnostic['rendering_reason'] = 'single-acyl-suffix-parent-override-unsupported'
        return ""
    hetero = _isolated_heterounits(mol, path)
    ends = (path[0], path[-1]) if path else ()
    if (not path or hetero is None or len(hetero) < 4
            or not centers <= (set(path) if groups and groups[0].kind=='alcohol' else set(ends))):
        if diagnostic is not None:
            diagnostic['rendering_reason'] = 'heterounit-rendering-or-principal-site-unsupported'
        return ""
    if any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
           for a, b in zip(path, path[1:])):
        if diagnostic is not None:
            diagnostic['rendering_reason'] = 'unsaturated-parent-renderer-unsupported'
        return ""
    path = min((path, tuple(reversed(path))), key=lambda p: (_replacement_key(mol,p),
        tuple(sorted(p.index(g.center)+1 for g in groups))))
    parent = _replacement_text(mol, path)
    if diagnostic is not None:
        diagnostic['rendering_reason'] = 'rendered'
    if not groups:
        return (parent,path,_isolated_heterounits(mol,path)) if evidence else parent
    if groups[0].kind=='alcohol':
        name = _alcohol_suffix(parent,[str(path.index(g.center)+1) for g in groups])
        return (name,path,_isolated_heterounits(mol,path)) if evidence else name
    record = _render(mol, Chem, groups, groups[0].kind, path, False, False,
                     skipped, owners, parent_override=parent)
    if not record:
        if diagnostic is not None:
            diagnostic['rendering_reason'] = 'suffix-renderer-declined'
        return ""
    return (record[0],path,_isolated_heterounits(mol,path)) if evidence else record[0]


def _block_name(element, size, *, outward=False):
    if element == 6:
        if size == 1:
            return "methylene"
        if size not in ROOTS:
            return ""
        locs = f"{size},1" if outward else f"1,{size}"
        return ROOTS[size] + "ane-" + locs + "-diyl"
    return {(8, 1): "oxy", (8, 2): "peroxy", (16, 1): "sulfanediyl",
            (16, 2): "disulfanediyl", (7, 1): "azanediyl"}.get((element, size), "")


def _connector_name(mol, atoms, units, *, allow_azo=False):
    if not atoms or any(mol.GetAtomWithIdx(i).IsInRing() or mol.GetAtomWithIdx(i).GetIsotope()
                        or mol.GetAtomWithIdx(i).GetAtomicNum() not in _LINK_ELEMENTS
                        for i in atoms):
        return ""
    if len(atoms) == 1 and len(units) in {3, 4}:
        number = mol.GetAtomWithIdx(next(iter(atoms))).GetAtomicNum()
        return {(6, 3): "methanetriyl", (6, 4): "methanetetrayl",
                (7, 3): "nitrilo"}.get((number, len(units)), "")
    if len(units) in {3, 4}:
        # P-15.3.3.1 permits a multivalent central group with later groups
        # linked through identical bonds; P-29 supplies the multiradical
        # parent form. This writer is deliberately limited to a fully mapped,
        # saturated all-carbon path and does not impose central symmetry.
        if (len(atoms) not in ROOTS or any(
                mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                or mol.GetAtomWithIdx(i).GetIsotope()
                or mol.GetAtomWithIdx(i).IsInRing() for i in atoms)):
            return ""
        adjacency = {i: set() for i in atoms}
        for atom in atoms:
            for neighbor in mol.GetAtomWithIdx(atom).GetNeighbors():
                other = neighbor.GetIdx()
                if other in atoms:
                    if mol.GetBondBetweenAtoms(atom, other).GetBondTypeAsDouble() != 1:
                        return ""
                    adjacency[atom].add(other)
        endpoints = [atom for atom, neighbors in adjacency.items() if len(neighbors) == 1]
        if len(atoms) == 1:
            path = (next(iter(atoms)),)
        elif len(endpoints) == 2 and all(len(ns) <= 2 for ns in adjacency.values()):
            path = _linear_order(mol, atoms, endpoints)
        else:
            return ""
        if not path or any(unit.connector not in atoms for unit in units):
            return ""
        if any(mol.GetBondBetweenAtoms(unit.root, unit.connector).GetBondTypeAsDouble() != 1
               for unit in units):
            return ""
        orientations = (path, tuple(reversed(path)))
        path = min(orientations, key=lambda p: tuple(sorted(
            p.index(unit.connector) + 1 for unit in units)))
        locants = ",".join(str(value) for value in sorted(
            path.index(unit.connector) + 1 for unit in units))
        suffix = {2: "diyl", 3: "triyl", 4: "tetrayl"}[len(units)]
        return ROOTS[len(path)] + "ane-" + locants + "-" + suffix
    if len(units) != 2:
        return ""
    endpoints = [unit.connector for unit in units]
    if (allow_azo and len(atoms) == 2 and set(endpoints) == atoms
            and all(atom.GetAtomicNum() == 7 and atom.GetDegree() == 2
                    and not atom.GetFormalCharge() and not atom.GetIsotope()
                    for atom in (mol.GetAtomWithIdx(i) for i in endpoints))):
        bridge = mol.GetBondBetweenAtoms(*endpoints)
        if (bridge is not None and bridge.GetBondTypeAsDouble() == 2
                and all(mol.GetBondBetweenAtoms(unit.root, unit.connector).GetBondTypeAsDouble() == 1
                        for unit in units)):
            # P-68.3.1.3.2: this linker connects the two identical functional
            # parent units. Any E/Z descriptor belongs inside the linker.
            if not bridge.GetStereo():
                return "diazenediyl"
            from rdkit import Chem
            from .stereochemistry import assigned_bond_stereochemistry

            descriptor = next((value for a, b, value in assigned_bond_stereochemistry(mol, Chem)
                               if {a, b} == atoms), None)
            return f"({descriptor})-diazenediyl" if descriptor else ""
    if len(atoms) == 1:
        path = tuple(atoms)
    else:
        path = _linear_order(mol, atoms, endpoints)
    if not path or any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
                       for a, b in zip(path, path[1:])):
        return ""
    hetero = _isolated_heterounits(mol, path)
    if hetero is not None and len(hetero) >= 4 and len(path) in ROOTS:
        path = min((path, tuple(reversed(path))), key=lambda p: _replacement_key(mol, p))
        return _replacement_text(mol, path) + f"-1,{len(path)}-diyl"
    blocks = [(element, len(tuple(items))) for element, items in groupby(
        mol.GetAtomWithIdx(i).GetAtomicNum() for i in path)]
    if any(a != 6 and b != 6 and a != b for (a, _), (b, _) in zip(blocks, blocks[1:])):
        return ""  # Mixed contiguous heterounits need their own preferred grammar.
    if blocks != blocks[::-1] or len(blocks) % 2 != 1:
        return ""  # Noncentral units must be symmetrically substituted.
    middle = len(blocks) // 2
    central = _block_name(*blocks[middle])
    tail = "".join(_block_name(*b, outward=True) for b in blocks[middle + 1:])
    if not central or any(not _block_name(*b) for b in blocks):
        return ""
    return central + ("bis(" + tail + ")" if tail else "")


def _monocyclic_arylene_connector(mol, Chem, scope, units):
    """Number a complete symmetric divalent aromatic monocycle."""
    if len(scope) != 6 or len(units) != 2:
        return ''
    ports = {u.connector for u in units}
    if len(ports) != 2 or not ports <= scope or any(
            not mol.GetAtomWithIdx(i).GetIsAromatic()
            or mol.GetAtomWithIdx(i).GetFormalCharge() or mol.GetAtomWithIdx(i).GetIsotope()
            for i in scope):
        return ''
    from .mapped_ring_parents import skeletons

    cut = {(i, n.GetIdx()) for i in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
           if n.GetIdx() not in scope and n.GetAtomicNum() > 1}
    if cut != {(u.connector, u.root) for u in units}:
        return ''
    rows = [(row[0], tuple(int(loc) for loc in row[2]), row[1])
            for row in skeletons(mol, Chem, scope)
            if row[0] in {'benzene', 'pyridine', 'pyrimidine', 'pyridazine', 'pyrazine'}]
    options = defaultdict(set)
    for parent, locants, seq in rows:
        numbered = dict(zip(seq, locants, strict=True))
        pair = tuple(numbered[u.connector] for u in units)
        options[parent, tuple(sorted(pair))].add(pair)
    if len({parent for parent, _ in options}) != 1:
        return ''
    parent, locants = min(options, key=lambda row: row[1])
    # Both directions must be witnessed by an automorphism of the same
    # numbered parent. A pair of unlike attachment positions cannot multiply.
    if len(options[parent, locants]) != 2:
        return ''
    return attach_suffix(parent, 'diyl', locants)


def _central_connector(mol, Chem, scope, ports, arms):
    """Name a multivalent central group with mapped pendant substituents."""
    from .component_mapping import mapped_general_branch
    from .multifunctional_parent import _paths

    count = len(ports)
    suffix = {2: 'diyl', 3: 'triyl', 4: 'tetrayl'}.get(count)
    if suffix is None:
        return ''
    carbon = {i for i in scope if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
              and not mol.GetAtomWithIdx(i).IsInRing()}
    if set(ports) <= carbon:
        paths = [path for path in _paths(mol, carbon)
                 if len(path) in ROOTS and set(ports) <= set(path)]
        maximum = max(map(len, paths), default=0)
        paths = [order for path in paths if len(path) == maximum
                 for order in dict.fromkeys((path, path[::-1]))]
    else:
        paths = [ports[:1]] if len(set(ports)) == 1 else []

    choices = []
    for path in paths:
        if any(mol.GetAtomWithIdx(i).GetIsotope() or int(mol.GetAtomWithIdx(i).GetChiralTag())
               for i in path):
            continue
        mapping = {idx: j for j, idx in enumerate(path, 1)}
        positions = tuple(sorted(mapping[i] for i in ports))
        bond_orders = tuple(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
                            for a, b in zip(path, path[1:]))
        multiple = tuple(i for i, order in enumerate(bond_orders, 1) if order > 1)
        double = tuple(i for i, order in enumerate(bond_orders, 1) if order == 2)
        if any(mol.GetBondBetweenAtoms(a, b).GetStereo() for a, b in zip(path, path[1:])):
            continue
        remaining = scope - set(path)
        prefixes = []
        for idx in path:
            for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
                root = atom.GetIdx()
                if root not in scope or root in mapping or atom.GetAtomicNum() == 1:
                    continue
                if mol.GetBondBetweenAtoms(idx, root).GetBondTypeAsDouble() != 1:
                    break
                name, owned = mapped_general_branch(mol, Chem, root, idx)
                if not name or not owned <= remaining:
                    break
                remaining -= owned
                prefixes.append((str(mapping[idx]), name))
            else:
                continue
            break
        else:
            if remaining:
                continue
            if len(path) == 1:
                element = mol.GetAtomWithIdx(path[0]).GetAtomicNum()
                parent = _block_name(element, 1) if count == 2 else {
                    (6, 3): 'methanetriyl', (6, 4): 'methanetetrayl', (7, 3): 'nitrilo',
                }.get((element, count), '')
                if not parent or (prefixes and element not in {6, 7}):
                    continue
            else:
                parent = attach_suffix(parent_chain_name(mol, path), suffix, positions)
                if not parent:
                    continue
            prefix = format_prefixes(prefixes, omit_locants=len(path) == 1,
                                     hierarchical_enclosures=True)
            # The unlocanted branch and the atom form one connector word.
            # Keep its prefix attached before enclosing the whole connector.
            text = prefix + parent if len(path) == 1 else join_prefix_parent(prefix, parent)
            prefix_locants = tuple(sorted(int(loc) for loc, _ in prefixes))
            citation_locants = tuple(int(loc) for loc, _ in sorted(
                prefixes, key=lambda item: (prefix_alphabetization_key(item[1]), int(item[0]))))
            choices.append((-len(multiple), -len(double), positions, multiple, double,
                            -len(prefixes), prefix_locants, citation_locants, text))
    if not choices:
        return ''
    key = min(row[:-1] for row in choices)
    names = {row[-1] for row in choices if row[:-1] == key}
    return next(iter(names)) + arms if len(names) == 1 else ''


def _connector_arm_name(mol, path):
    blocks = [(element, len(tuple(items))) for element, items in groupby(
        mol.GetAtomWithIdx(i).GetAtomicNum() for i in path)]
    if any(a != 6 and b != 6 and a != b for (a, _), (b, _) in zip(blocks, blocks[1:])):
        return ''
    terms = [_block_name(*block, outward=True) for block in blocks]
    return ''.join(terms) if all(terms) else ''


def _mapped_ring_connector(mol, Chem, scope, units):
    """Render one numbered central ring at its original polyvalent ports."""
    from .component_mapping import MappedComponent, decorate_component, mapped_general_branch
    from .mapped_ring_parents import ring_systems, skeletons
    from .parent_selection import locant_key

    ports = {unit.connector for unit in units}
    cores = [core for core in ring_systems(mol) if ports <= core <= scope]
    if len(ports) != len(units) or len(cores) != 1:
        return ''
    core = cores[0]
    if (not 3 <= len(core) <= 16 or any(
            sum(n.GetIdx() in core for n in mol.GetAtomWithIdx(i).GetNeighbors()) != 2
            for i in core)):
        return ''
    cuts = {(u.connector, u.root) for u in units}
    if {(i, a.GetIdx()) for i in scope for a in mol.GetAtomWithIdx(i).GetNeighbors()
        if a.GetIdx() not in scope and a.GetAtomicNum() > 1} != cuts:
        return ''
    branches, owned = [], set(core)
    for i in core:
        for atom in mol.GetAtomWithIdx(i).GetNeighbors():
            root = atom.GetIdx()
            if root in core or (i, root) in cuts or atom.GetAtomicNum() == 1:
                continue
            text, atoms = mapped_general_branch(mol, Chem, root, i)
            if not text or atoms & owned or not atoms <= scope:
                return ''
            owned.update(atoms)
            branches.append((i, text))
    if owned != scope:
        return ''
    suffix = {2: 'diyl', 3: 'triyl', 4: 'tetrayl'}.get(len(ports))
    if suffix is None:
        return ''
    choices = []
    for parent, sequence, locants, _numbers, _edges, saturated, *_ in skeletons(mol, Chem, core):
        if not saturated:
            continue
        mapping = dict(zip(sequence, locants, strict=True))
        positions = tuple(sorted((mapping[i] for i in ports), key=locant_key))
        # Multivalent -diyl/-triyl/-tetrayl retains the hydride's final e.
        body = parent + '-' + ','.join(positions) + '-' + suffix
        prefixes = tuple((mapping[i], text) for i, text in branches)
        name = decorate_component(mol, Chem, MappedComponent(
            body, tuple(mapping.items()), core, prefixes, scope - core))
        if not name:
            continue
        key = (tuple(map(locant_key, positions)),
               tuple(sorted(locant_key(loc) for loc, _text in prefixes)),
               tuple(locant_key(loc) for loc, _text in sorted(
                   prefixes, key=lambda row: (prefix_alphabetization_key(row[1]), locant_key(row[0])))))
        choices.append((key, name))
    if not choices:
        return ''
    best = min(key for key, _name in choices)
    names = {name for key, name in choices if key == best}
    return next(iter(names)) if len(names) == 1 else ''


def _mapped_biphenyl_connector(mol, scope, units):
    """Number a biphenyl carrying identical units and matching halo patterns."""
    from .cycle_utils import ordered_ring

    if len(units) != 2 or not 12 <= len(scope) <= 20:
        return ''
    rings = [frozenset(ring) for ring in mol.GetRingInfo().AtomRings()
             if len(ring) == 6 and set(ring) <= scope and all(
                 mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and mol.GetAtomWithIdx(i).GetIsAromatic()
                 for i in ring)]
    if len(rings) != 2 or rings[0] & rings[1]:
        return ''
    core = rings[0] | rings[1]
    junctions = [(i, n.GetIdx()) for i in rings[0] for n in mol.GetAtomWithIdx(i).GetNeighbors()
                 if n.GetIdx() in rings[1]]
    if len(junctions) != 1:
        return ''
    junction = junctions[0]
    bond = mol.GetBondBetweenAtoms(*junction)
    if bond.GetBondTypeAsDouble() != 1 or bond.GetStereo():
        return ''
    cuts = {(u.connector, u.root) for u in units}
    if {(i, n.GetIdx()) for i in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
        if n.GetIdx() not in scope and n.GetAtomicNum() > 1} != cuts:
        return ''
    halo = {}
    for idx in scope:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumRadicalElectrons() or atom.GetChiralTag():
            return ''
        if idx in core:
            continue
        word = HALO.get(atom.GetSymbol())
        if not word or atom.GetDegree() != 1 or atom.GetBonds()[0].GetBondTypeAsDouble() != 1:
            return ''
        host = atom.GetNeighbors()[0].GetIdx()
        if host not in core or host in halo:
            return ''
        halo[host] = word
    patterns = []
    for ring, start in zip(rings, junction, strict=True):
        ports = {u.connector for u in units if u.connector in ring}
        order = ordered_ring(mol, ring)
        if len(ports) != 1 or not order:
            return ''
        port = ports.pop()
        pos = order.index(start)
        order = order[pos:] + order[:pos]
        choices = []
        for path in (order, (start, *reversed(order[1:]))):
            prefixes = tuple((loc, halo[idx]) for loc, idx in enumerate(path, 1) if idx in halo)
            choices.append((path.index(port) + 1, prefixes))
        patterns.append(min(choices, key=lambda row: (row[0], tuple(loc for loc, _ in row[1]),
            tuple(loc for loc, _ in sorted(row[1], key=lambda item: (item[1], item[0]))))))
    if patterns[0] != patterns[1]:
        return ''
    locant, prefixes = patterns[0]
    prefix = format_prefixes([(str(loc) + prime, name) for loc, name in prefixes for prime in ('', "'")])
    return join_prefix_parent(prefix, f"[1,1'-biphenyl]-{locant},{locant}'-diyl")


def _mapped_connector_name(mol, Chem, scope, units):
    """Peel identical outward arms, then render the entire central complement."""
    ports = tuple(unit.connector for unit in units)
    paths = tuple(Chem.GetShortestPath(mol, ports[0], i) for i in set(ports) - {ports[0]})
    spine = frozenset((ports[0],)).union(*paths)
    if not spine <= scope:
        return ''
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in spine):
        if all(mol.GetBondBetweenAtoms(u.root, u.connector).GetBondTypeAsDouble() == 1 for u in units):
            return (_mapped_ring_connector(mol, Chem, scope, units)
                    or _mapped_biphenyl_connector(mol, scope, units))
        return ''
    # Keep the existing source-preferred replacement form for long linear
    # heterochains, and leave branched replacement competition open.
    for left in set(ports):
        for right in set(ports) - {left}:
            units_on_path = _isolated_heterounits(mol, Chem.GetShortestPath(mol, left, right))
            if units_on_path is None or len(units_on_path) >= 4:
                return ''
    for unit in units:
        if mol.GetBondBetweenAtoms(unit.root, unit.connector).GetBondTypeAsDouble() != 1:
            return ''
    if any(b.GetBondTypeAsDouble() != 1
           and (b.GetBondTypeAsDouble() not in {2, 3} or b.GetStereo()
                or b.GetBeginAtom().GetAtomicNum() != 6 or b.GetEndAtom().GetAtomicNum() != 6)
           for i in spine for b in mol.GetAtomWithIdx(i).GetBonds() if b.GetOtherAtomIdx(i) in spine):
        return ''

    # A fork can transmit two equivalent attachments through one arm. Keep
    # the fan-out in the arm name and multiply the resulting identical arms.
    fanout = Counter(ports)
    if len(fanout) == 2 and set(fanout.values()) == {2}:
        ends = tuple(fanout)
        path = Chem.GetShortestPath(mol, *ends)
        if (len(path) >= 3 and len(path) % 2 and frozenset(path) == scope
                and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in ends)
                and all(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() == 1
                        for a, b in zip(path, path[1:]))
                and not any(mol.GetAtomWithIdx(i).GetIsotope() or int(mol.GetAtomWithIdx(i).GetChiralTag())
                            for i in path)):
            mid = len(path) // 2
            arms = (path[mid - 1::-1], path[mid + 1:])
            if tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in arms[0]) == tuple(
                    mol.GetAtomWithIdx(i).GetAtomicNum() for i in arms[1]):
                arm_body = _connector_arm_name(mol, arms[0][:-1])
                if arm_body or len(arms[0]) == 1:
                    return _central_connector(mol, Chem, frozenset({path[mid]}), (path[mid],) * 2,
                                              'bis' + enclose_name_unit(arm_body + 'nitrilo'))

    options = [(scope, ports, '')]
    arm = ()
    remaining = spine
    center = scope
    while len(set(ports)) == len(ports) and len(ports) < len(remaining):
        ends = set(ports)
        onward = [tuple(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                        if n.GetIdx() in center) for i in ports]
        elements = {mol.GetAtomWithIdx(i).GetAtomicNum() for i in ports}
        if (len(elements) != 1 or any(len(neighbors) != 1 or neighbors[0] not in remaining - ends
                                     for neighbors in onward)
                or any(mol.GetBondBetweenAtoms(i, neighbors[0]).GetBondTypeAsDouble() != 1
                       for i, neighbors in zip(ports, onward))
                or any(mol.GetAtomWithIdx(i).GetIsotope() or int(mol.GetAtomWithIdx(i).GetChiralTag())
                       for i in ports)):
            break
        arm = (ports[0], *arm)
        center -= ends
        remaining -= ends
        ports = tuple(neighbors[0] for neighbors in onward)
        text = _connector_arm_name(mol, arm)
        multiplier = COMPLEX_MULT.get(len(ports))
        if text and multiplier:
            options.append((center, ports, multiplier + enclose_name_unit(text)))
    for center, ports, arms in reversed(options):
        text = _central_connector(mol, Chem, center, ports, arms)
        if text:
            return text
    return ''


def _multiplicative_parent(mol, Chem, groups, all_atoms):
    """Compatibility wrapper for callers that only need a rendered decision."""
    return _multiplicative_parent_outcome(mol, Chem, groups, all_atoms).decision


def _joins_senior_parent(mol, Chem, groups, units, all_atoms, frontier):
    """Compare the repeated frame with the available whole-graph P-44 parents."""
    from .mapped_ring_parents import _site, parent_facts, records, ring_systems
    from .mapped_principal_general import _carbon_parent_plans
    from .multifunctional_parent import _paths
    from .parent_selection import compare_parent_seniority, graph_parent_facts
    from .proposal_plans import graph_proposal_bound

    selected = units[0].parent_facts
    if selected is None or not all(unit.parent_atoms for unit in units):
        return False
    if frontier is not None and frontier.candidates:
        return all(compare_parent_seniority(selected, row.parent_selection).preference == 0
                   for row in frontier.candidates)

    kind = groups[0].kind
    if kind not in {'acid', 'amide', 'nitrile', 'alcohol', 'lactam'}:
        return False
    # Equal intrinsic P-44 frames may differ in their prefixes. A cut gives
    # exact evidence for its own frame; other physical rings need their own
    # facts unless the graph-only bound already places them below the unit.
    mapped_frames = {unit.parent_atoms for unit in units}
    unresolved_rings = []
    for core in ring_systems(mol):
        sites = [g for g in groups if _site(mol, g, core) in core]
        if not sites or core in mapped_frames:
            continue
        bound = graph_proposal_bound(mol, core, sites, cyclic=True)
        comparison = compare_parent_seniority(selected, bound).preference
        if comparison == 1:
            return False
        if comparison != -1:
            unresolved_rings.append(core)
    if unresolved_rings:
        for core in unresolved_rings:
            rows = records(mol, Chem, scope=all_atoms, core_filter=core,
                           principal=tuple(groups), general_branches=True)
            if not rows or any(compare_parent_seniority(
                    selected, parent_facts(mol, row, groups[0].seniority)).preference
                    not in {-1, 0} for row in rows):
                return False

    if kind == 'alcohol':
        carbon = {i for i in all_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                  and not mol.GetAtomWithIdx(i).IsInRing()}
        paths = ((path, tuple(g for g in groups if g.center in path))
                 for path in _paths(mol, carbon))
    else:
        paths = ((path, sites) for path, sites, _external in
                 _carbon_parent_plans(mol, groups, selection_inventory(mol)))
    for path, sites in paths:
        if not sites or frozenset(path) in mapped_frames:
            continue
        bound = graph_proposal_bound(mol, path, sites)
        comparison = compare_parent_seniority(selected, bound).preference
        if comparison == -1:
            continue
        # On equal-length ordinary terminal chains, the selected P-44
        # suffix/unsaturation criteria still have to tie on the whole graph.
        if comparison == 1:
            return False
        mapping = {i: str(j) for j, i in enumerate(path, 1)}
        suffix_locants = [mapping[g.center if g.center in mapping else g.owner] for g in sites]
        other = graph_parent_facts(mol, path, mapping, principal_class=groups[0].seniority,
                                  suffix_centers=[g.center for g in sites],
                                  suffix_locants=suffix_locants, allow_annotations=True)
        if compare_parent_seniority(selected, other).preference not in {-1, 0}:
            return False
    return True


def _multiplicative_parent_outcome(mol, Chem, groups, all_atoms, *, senior_frontier=None,
                                   required_stereo_connector=(), required_azo_connector=()):
    """Classify bounded mapped P-51.3 constructions and unresolved competitors.

    A missing rendered decision is not enough to establish that multiplication
    is ineligible. If a full-center identical-unit proposal pool exists but
    topology or connector rendering is unresolved, keep the method comparison
    pending. Rendered unit-parent families are compared by their typed P-44
    facts before P-45 multiplicity; unknown and tied distinct families remain
    in the frontier. This helper reports construction eligibility only; its
    result cannot displace a different whole-molecule P-44 senior-parent
    family. The connector may own omitted same-class sites as mapped prefixes.
    """
    from .method_ineligibility import ordinary_multiplication_ineligibility
    excluded=ordinary_multiplication_ineligibility(mol,Chem,groups)
    if excluded is not None:
        return _MultiplicativeOutcome('ineligible',reason=excluded.reason,evidence=(excluded,))
    from .multiplicative_topology import multiplicative_topology

    grouped = defaultdict(list)
    unit_unresolved = []
    for unit in _parent_units(mol, Chem, groups, unresolved=unit_unresolved):
        grouped[(unit.signature, str(unit.locant), len(unit.centers))].append(unit)
    required = frozenset(g.center for g in groups)
    choices = []
    pending_reasons = list(unit_unresolved)
    full_center_pools = []
    # A rooted unit may be discovered by several cuts. Prove a maximum
    # disjoint packing rather than equating discovery count with multiplicity.
    required_atoms = frozenset(all_atoms) | frozenset(
        a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 1 and a.GetIsotope())
    for identity, candidates in grouped.items():
        parent_renderings = {unit.parent for unit in candidates}
        if len(parent_renderings) != 1:
            pending_reasons.append((identity, "same-rooted-unit-parent-wording-unresolved"))
            continue
        if len(candidates) < 2:
            continue
        packing_status = _full_center_unit_packing(candidates, required)
        if packing_status is True:
            full_center_pools.append((identity, tuple(candidates)))
        topology = multiplicative_topology(mol, candidates, required_atoms, required,
                                           allow_residual=bool(groups))
        if topology is None or topology.status not in {
                'maximal_identical_units', 'single_multiplicative_group', 'nearest_subset'}:
            if packing_status is True:
                pending_reasons.append((identity,
                    "full-center-identical-unit-pool-topology-unresolved",
                    (candidates[0].parent_facts,)))
            elif packing_status is None:
                pending_reasons.append((identity,
                    "full-center-unit-packing-search-unresolved",
                    (candidates[0].parent_facts,)))
            elif groups and any(not left.atoms & right.atoms for j, left in enumerate(candidates)
                                for right in candidates[j + 1:]):
                pending_reasons.append((identity, 'residual-unit-topology-unresolved',
                                        (candidates[0].parent_facts,)))
            continue
        units = [candidates[i] for i in topology.selected_indices]
        if len(units) not in MULT or len(units) < 2:
            continue
        covered = frozenset().union(*(u.atoms for u in units))
        if sum(len(u.atoms) for u in units) != len(covered):
            continue
        selected_centers = frozenset().union(*(u.centers for u in units))
        if not selected_centers <= required:
            continue
        connector = required_atoms - covered
        if required_azo_connector and connector != required_azo_connector:
            continue
        if required_stereo_connector and not (
                set(required_stereo_connector) <= topology.central_group_atoms <= connector):
            continue
        omitted = tuple(g for g in groups if g.center not in selected_centers)
        if any(not set(g.atoms) <= connector for g in omitted):
            continue
        # Each unit has exactly one bridge attachment; the connector owns all
        # remaining atoms. This prevents dropped groups and double ownership.
        if any(u.connector not in connector for u in units):
            continue
        if omitted or topology.status == 'nearest_subset':
            from .linear_phane_methods import linear_phane_frontier
            from .mapped_ring_parents import ring_systems

            # Four terminal or internal ring units can make the phane
            # comparison mandatory under P-51.3.2/P-52.2.5.1. The phane
            # producer runs first. Leave an unresolved phane domain open.
            if len(ring_systems(mol)) >= 4 and linear_phane_frontier(mol, Chem).mandatory_eligible:
                continue
        bridge = _connector_name(mol, connector, units, allow_azo=bool(selected_centers))
        mapped_bridge = False
        if not bridge and groups:
            bridge = _monocyclic_arylene_connector(mol, Chem, connector, units)
        extended = bool(omitted) or not bridge or senior_frontier is not None
        if required_stereo_connector:
            bridge = _mapped_ring_connector(mol, Chem, connector, units)
            mapped_bridge = bool(bridge)
        elif not bridge and groups:
            bridge = _mapped_connector_name(mol, Chem, connector, units)
            mapped_bridge = bool(bridge)
        if bridge and extended and not _joins_senior_parent(
                mol, Chem, groups, units, all_atoms, senior_frontier):
            pending_reasons.append((identity, 'repeated-unit-not-joined-to-senior-parent',
                                    (units[0].parent_facts,)))
            continue
        if not bridge:
            pending_reasons.append((identity,
                "mapped-complete-unit-set-connector-renderer-unavailable",
                (units[0].parent_facts,)))
            continue
        parent, loc = units[0].parent, units[0].locant
        locants = ",".join(loc + "'" * j for j in range(len(units)))
        substituted = units[0].substituted
        # A repeated substituted monocyclic carboxylic-acid unit requires
        # the acid-bearing ring locant in the complete enclosed parent word
        # (P-51.3.1); the isolated unit may have omitted that locant.
        simple_cyclo_acids = {'cyclo' + root + 'anecarboxylic acid'
                              for count, root in ROOTS.items() if count >= 3}
        if parent in simple_cyclo_acids:
            parent = parent[:-len('carboxylic acid')] + '-1-carboxylic acid'
        enclosed = substituted or any(c.isdigit() or c in "()[]" for c in parent)
        multiplier = (COMPLEX_MULT.get(len(units))
                      if substituted else MULT[len(units)])
        if not multiplier:
            pending_reasons.append((identity,
                "mapped-complete-unit-set-multiplier-renderer-unavailable",
                (units[0].parent_facts,)))
            continue
        multiplied = multiplier + (enclose_name_unit(parent) if enclosed else parent)
        enclosed_bridge = enclose_name_unit(bridge) if (
            mapped_bridge or omitted or any(c.isdigit() or c in "-()[]{}" for c in bridge)) else bridge
        if (loc == '1' and groups and groups[0].kind == 'alcohol'
                and all(len(unit.parent_atoms) == 1 for unit in units)):
            # Each methanol frame has just one possible attachment position.
            # P-15.3.1.2 does not need an independent repeated-parent locant.
            name = bridge + multiplied
        else:
            name = locants + "-" + enclosed_bridge + multiplied
        owners = [owner for unit in units for owner in unit.ester_owners]
        if owners:
            name = format_prefixes([("", owner) for owner in owners], omit_locants=True) + " " + name
        # P-45.1.2/P-51.3.2.1 prefer the greater multiplicity. Equal-count
        # decompositions need the remaining applicable parent criteria; this
        # bounded producer does not invent a text/rank tie-break.
        roles = tuple((f'repeated-parent:{j}', tuple(sorted(unit.atoms)),
                       ((unit.root, unit.locant),))
                      for j, unit in enumerate(units, 1))
        roles += (('connector', tuple(sorted(connector)), ()),)
        choices.append((len(units), identity, MethodDecision(name, "multiplicative", selected_centers, all_atoms,
            topology.rule,
            atom_locants=tuple((unit.root, unit.locant) for unit in units),
            component_roles=roles), units[0].parent_facts))
    if choices:
        from .parent_selection import compare_parent_seniority

        # P-45 multiplicity is considered only after comparing the repeated
        # parent structures under P-44. A known junior family cannot block a
        # rendered senior family; unknown comparisons remain explicit barriers.
        dominated = set()
        p44_junior_families = []
        comparisons = {}
        for left in range(len(choices)):
            for right in range(left + 1, len(choices)):
                comparison = compare_parent_seniority(choices[left][3], choices[right][3])
                comparisons[(left, right)] = comparison.preference
                if comparison.preference == -1:
                    dominated.add(right)
                    p44_junior_families.append((choices[right][1], comparison.rule))
                elif comparison.preference == 1:
                    dominated.add(left)
                    p44_junior_families.append((choices[left][1], comparison.rule))
        # A preference cycle invalidates the entire partial ordering.
        edges = {i: set() for i in range(len(choices))}
        for (left, right), preference in comparisons.items():
            if preference == -1:
                edges[left].add(right)
            elif preference == 1:
                edges[right].add(left)
        cycle = any(
            _has_parent_preference_cycle(edges, node, set(), set())
            for node in edges
        )
        p44_frontier = [row for i, row in enumerate(choices) if i not in dominated]
        frontier_indices = {i for i in range(len(choices)) if i not in dominated}
        surviving_unknown = any(
            value is None and left in frontier_indices and right in frontier_indices
            for (left, right), value in comparisons.items()
        )
        if surviving_unknown or cycle:
            return _MultiplicativeOutcome(
                "pending", reason="unit-parent-p44-comparison-unresolved-before-p45",
                evidence=(tuple((row[1], row[3]) for row in choices),
                          tuple(comparisons.items()), ("cycle", cycle)),
                proposals=tuple(row[2] for row in p44_frontier),
            )
        p44_facts = tuple(row[3] for row in p44_frontier)
        unresolved_blockers = []
        for item in pending_reasons:
            if len(item) >= 3 and isinstance(item[0], tuple) and len(item[0]) == 3:
                _identity, _reason, candidate_facts = item[:3]
                facts = (candidate_facts if isinstance(candidate_facts, tuple)
                         else (candidate_facts,))
                known = tuple(f for f in facts if f is not None)
                junior_rules = tuple(
                    compare_parent_seniority(senior, junior).rule
                    for junior in known for senior in p44_facts
                )
                if (known and len(known) == len(facts)
                        and len(junior_rules) == len(known) * len(p44_facts)
                        and all(compare_parent_seniority(senior, junior).preference == -1
                                for junior in known for senior in p44_facts)):
                    p44_junior_families.append((_identity, junior_rules))
                    continue
            unresolved_blockers.append(item)
        pending_reasons = unresolved_blockers
        choices = p44_frontier
        greatest = max(count for count, _, _, _ in choices)
        blocking_competitors = _multiplicative_competitors_at_or_above(
            pending_reasons, greatest,
        )
        if blocking_competitors:
            return _MultiplicativeOutcome(
                "pending",
                reason="unresolved-multiplicative-competitor-can-tie-or-exceed-rendered-count",
                evidence=(
                    ("best-rendered-count", greatest),
                    ("rendered-proposals", tuple(
                    (identity, count)
                    for count, identity, decision, _facts in choices
                    )),
                    *tuple(blocking_competitors),
                ),
                proposals=tuple(decision for _count, _identity, decision, _facts in choices),
            )
        preferred = [(identity, decision) for count, identity, decision, _facts in choices
                     if count == greatest]
        # A duplicate decomposition can arrive by several equivalent routes,
        # but different rooted unit identities with equal multiplicity are
        # unresolved until the remaining source criteria compare them.
        identities = {identity for identity, _ in preferred}
        names = {decision.name for _, decision in preferred}
        if len(identities) != 1 or len(names) != 1:
            return _MultiplicativeOutcome(
                "pending", reason="equal-multiplicity-parent-decompositions-unresolved",
                evidence=tuple((identity, "equal-multiplicity-needs-parent-comparison")
                               for identity in sorted(identities, key=repr)),
                proposals=tuple(decision for _identity, decision in preferred),
            )
        return _MultiplicativeOutcome(
            "complete", preferred[0][1],
            "mapped-multiplicative-parent-rendered",
            (tuple((identity, "complete-mapped-unit-set") for identity, _ in full_center_pools)
             + (("p44-junior-unit-families", tuple(p44_junior_families)),)),
            tuple(decision for _identity, decision in preferred),
        )
    if pending_reasons:
        return _MultiplicativeOutcome(
            "pending", reason="eligible-or-potential-complete-multiplicative-pool-unresolved",
            evidence=tuple(pending_reasons),
        )
    return _MultiplicativeOutcome(
        "pending",
        reason="complete-multiplicative-ineligibility-unproved-by-bounded-unit-enumeration",
        evidence=tuple((identity, tuple(sorted(frozenset().union(
            *(unit.centers for unit in candidates)))))
            for identity, candidates in grouped.items()),
    )


def _multiplicative_competitors_at_or_above(pending_reasons, selected_count):
    """Keep unresolved proposals that can still tie or beat a rendered count.

    A third tuple item is an explicit source-proved upper bound on multiplicity.
    Existing reason records have no such bound and therefore remain blockers.
    This is deliberately independent of candidate enumeration order.
    """
    blocking = []
    for item in pending_reasons:
        if len(item) >= 3 and isinstance(item[2], int):
            upper_bound = item[2]
            if upper_bound < selected_count:
                continue
        blocking.append(item)
    return tuple(blocking)


def _has_parent_preference_cycle(edges, node, visiting, visited):
    if node in visiting:
        return True
    if node in visited:
        return False
    visiting.add(node)
    if any(_has_parent_preference_cycle(edges, neighbor, visiting, visited)
           for neighbor in edges[node]):
        return True
    visiting.remove(node)
    visited.add(node)
    return False


def _full_center_unit_packing(candidates, required, *, state_limit=10000):
    """Whether disjoint identical candidates can partition all required sites."""
    rows = tuple(unit for unit in candidates if unit.centers and unit.centers <= required)
    by_center = defaultdict(list)
    for index, unit in enumerate(rows):
        for center in unit.centers:
            by_center[center].append((index, unit))
    if set(by_center) != set(required):
        return False
    states = 0

    def search(remaining, occupied_atoms, chosen):
        nonlocal states
        states += 1
        if states > state_limit:
            return None
        if not remaining:
            return len(chosen) >= 2
        center = min(remaining, key=lambda value: len(by_center[value]))
        unresolved = False
        for index, unit in by_center[center]:
            if index in chosen or unit.centers & (required - remaining) or unit.atoms & occupied_atoms:
                continue
            outcome = search(remaining - unit.centers, occupied_atoms | unit.atoms,
                             chosen | {index})
            if outcome is True:
                return True
            if outcome is None:
                unresolved = True
        return None if unresolved else False

    return search(frozenset(required), frozenset(), frozenset())


def _alcohol_suffix(parent, locants):
    suffix=multiplied_suffix(len(locants),'ol') if len(locants)>1 else 'ol'
    return attach_suffix(parent,suffix,sorted(locants,key=int))


def _alcohol_unit(fragment, local_root):
    """A capped straight carbon polyol with its original attachment map."""
    inventory=selection_inventory(fragment)
    groups=inventory.principal
    if not groups or any(g.kind!='alcohol' for g in groups):return ()
    oxygen={i for g in groups for i in g.atoms if i!=g.center}
    carbon=set(range(fragment.GetNumAtoms()))-oxygen
    if any(fragment.GetAtomWithIdx(i).GetAtomicNum()!=6 for i in carbon):return ()
    ends=[i for i in carbon if sum(n.GetIdx() in carbon for n in fragment.GetAtomWithIdx(i).GetNeighbors())<=1]
    path=_linear_order(fragment,carbon,ends) if len(carbon)>1 else tuple(carbon)
    if (not path or local_root not in carbon or len(path) not in ROOTS
            or any(b.GetBondTypeAsDouble()!=1 for b in iter_bonds(fragment))):
        return _arylmethanol_unit(fragment, local_root, groups, oxygen)
    choices=[]
    from .parent_selection import graph_parent_facts

    for p in (path,tuple(reversed(path))):
        locs=tuple(sorted(p.index(g.center)+1 for g in groups))
        loc=p.index(local_root)+1
        name=_alcohol_suffix(ROOTS[len(p)]+'ane',list(map(str,locs)))
        facts = graph_parent_facts(fragment, p, {i: str(j) for j, i in enumerate(p, 1)},
                                   principal_class=groups[0].seniority,
                                   suffix_centers=[g.center for g in groups],
                                   suffix_locants=tuple(map(str, locs)), name=name)
        choices.append(((-len(groups), -len(path), locs, loc), name,
                        str(loc), False, (), facts, ('carbon-polyol', frozenset(p)),
                        frozenset(p)))
    return tuple(choices)


def _arylmethanol_unit(mol, root, groups, oxygen):
    """Keep a complete aryl arm on a one-carbon repeated alcohol parent."""
    if len(groups) != 1 or groups[0].center != root or len(oxygen) != 1:
        return ()
    atom = mol.GetAtomWithIdx(root)
    neighbors = [a for a in atom.GetNeighbors() if a.GetIdx() not in oxygen]
    if (atom.IsInRing() or len(neighbors) != 1 or neighbors[0].GetAtomicNum() != 6
            or not neighbors[0].GetIsAromatic() or atom.GetTotalNumHs() != 2):
        return ()
    from rdkit import Chem
    from .component_mapping import has_annotations
    from .parent_selection import graph_parent_facts

    scope = frozenset(range(mol.GetNumAtoms()))
    if has_annotations(mol, scope):
        return ()
    text, owned = mapped_branch(mol, Chem, neighbors[0].GetIdx(), root)
    if not text or owned != scope - {root} - oxygen:
        return ()
    name = enclose_name_unit(text) + 'methanol'
    facts = graph_parent_facts(mol, (root,), {root: '1'}, principal_class=groups[0].seniority,
        suffix_centers=(root,), suffix_locants=('1',), prefixes=(('1', text),), name=name)
    frame = frozenset({root})
    return (((), name, '1', True, (), facts, ('carbon-polyol', frame), frame),)


def _halogenated_linear_ether_substitutive(mol, groups, all_atoms):
    """P-51.1.1: a halogenated C-O-C path with one ether heterounit.

    The complete heavy-atom graph must be a saturated, unbranched carbon and
    oxygen path with only terminal, unlabeled halogens off the path. Each
    alternative parent and orientation is compared using carbon-chain length,
    numbers and locants of prefixes, and their citation order (P-45.2).
    """
    if groups or mol.GetRingInfo().NumRings():
        return None
    atoms = tuple(iter_atoms(mol))
    if any(a.GetAtomicNum() == 1 or a.GetFormalCharge() or a.GetIsotope()
           or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) for a in atoms):
        return None
    if any(a.GetAtomicNum() not in {6, 8, 9, 17, 35, 53} for a in atoms):
        return None
    if any(b.GetBondTypeAsDouble() != 1.0 for b in iter_bonds(mol)):
        return None
    oxygens = [a.GetIdx() for a in atoms if a.GetAtomicNum() == 8]
    halogens = [a.GetIdx() for a in atoms if a.GetSymbol() in HALO]
    if len(oxygens) != 1 or not halogens:
        return None
    oxygen = mol.GetAtomWithIdx(oxygens[0])
    if (oxygen.GetDegree() != 2 or oxygen.GetTotalValence() != 2
            or oxygen.GetTotalNumHs(includeNeighbors=True)):
        return None
    halo_by_carbon = defaultdict(list)
    for halo_idx in halogens:
        halo = mol.GetAtomWithIdx(halo_idx)
        if halo.GetDegree() != 1:
            return None
        host = halo.GetNeighbors()[0]
        if host.GetAtomicNum() != 6:
            return None
        halo_by_carbon[host.GetIdx()].append(HALO[halo.GetSymbol()])
    counts = Counter(term for values in halo_by_carbon.values() for term in values)
    if any(count > 1 and count not in MULT for count in counts.values()):
        return None
    backbone = frozenset(all_atoms) - frozenset(halogens)
    ends = [i for i in backbone if sum(n.GetIdx() in backbone
            for n in mol.GetAtomWithIdx(i).GetNeighbors()) == 1]
    path = _linear_order(mol, backbone, ends)
    if not path or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ends):
        return None
    if _isolated_heterounits(mol, path) != [(oxygens[0],)]:
        return None
    blocks = [tuple(items) for carbon, items in groupby(
        path, key=lambda i: mol.GetAtomWithIdx(i).GetAtomicNum() == 6) if carbon]
    if len(blocks) != 2 or any(len(block) not in ROOTS for block in blocks):
        return None
    # A uniquely longer carbon chain avoids unresolved parent comparisons
    # between differently substituted equal-length chains. Methane parents
    # need separate grammar for several unlocanted independent substituents.
    if len(blocks[0]) == len(blocks[1]) or max(map(len, blocks)) < 2:
        return None

    def halo_rows(block):
        return [(str(locant), term)
                for locant, carbon in enumerate(block, 1)
                for term in halo_by_carbon[carbon]]

    choices = []
    for parent_block in blocks:
        distal = blocks[1] if parent_block == blocks[0] else blocks[0]
        if len(parent_block) < len(distal):
            continue
        # Number the distal alkyl from the oxygen, which joins the two blocks.
        distal_from_oxygen = (tuple(reversed(distal)) if distal == blocks[0] else distal)
        distal_rows = halo_rows(distal_from_oxygen)
        distal_prefix = format_prefixes(distal_rows, omit_locants=len(distal) == 1,
                                        hierarchical_enclosures=True)
        alkyl = distal_prefix + ROOTS[len(distal)] + "yl"
        # P-63.2.2.2 retains the C1-C4 alkoxy forms even when substituted.
        alkoxy = alkyl[:-2] + "oxy" if len(distal) <= 4 else alkyl + "oxy"
        for oriented in (parent_block, tuple(reversed(parent_block))):
            ether_host = next(i for i in oriented if mol.GetBondBetweenAtoms(i, oxygens[0]))
            rows = halo_rows(oriented) + [(str(oriented.index(ether_host) + 1), alkoxy)]
            locants = tuple(sorted(int(locant) for locant, _ in rows))
            by_name = defaultdict(list)
            for locant, text in rows:
                by_name[text].append(int(locant))
            citation_locants = tuple(tuple(sorted(by_name[text])) for text in
                                     sorted(by_name, key=prefix_alphabetization_key))
            key = (-len(oriented), -len(rows), locants, citation_locants)
            omit = len(oriented) == 1 or len(oriented) == 2 and len(rows) == 1
            prefix = format_prefixes(rows, omit_locants=omit,
                                     hierarchical_enclosures=True,
                                     separate_unlocanted=True)
            name = join_prefix_parent(prefix, ROOTS[len(oriented)] + "ane")
            choices.append((key, name, oriented))
    if not choices:
        return None
    best = min(row[0] for row in choices)
    maxima = [row for row in choices if row[0] == best]
    # A remaining tie with different renderings is an unproved comparison.
    if len({row[1] for row in maxima}) != 1:
        return None
    _, name, parent = maxima[0]
    return MethodDecision(name, "substitutive", frozenset(), all_atoms,
                          "P-51.1.1/P-51.4.1.1/P-63.2.2.2/P-63.2.4.1/P-45.2",
                          tuple((idx, str(j)) for j, idx in enumerate(parent, 1)),
                          ((oxygens[0],),))


def _linear_substitutive(mol, Chem, groups, all_atoms):
    """P-51.1.1 below the heterounit threshold: ordinary linear ethers/sulfides.

    Rooted branches retain the attachment carbon. Parent length precedes the
    number/locants of prefixes; no name-length score establishes a method.
    """
    if groups or mol.GetRingInfo().NumRings():return None
    if any(a.GetAtomicNum() not in {6,8,16} for a in iter_atoms(mol)):return None
    if any(a.GetAtomicNum() in {8, 16} and a.GetTotalValence() != 2
           for a in iter_atoms(mol)):return None
    ends=[i for i in all_atoms if mol.GetAtomWithIdx(i).GetDegree()==1]
    path=_linear_order(mol,all_atoms,ends)
    if not path or any(mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble()!=1 for a,b in zip(path,path[1:])):return None
    units=_isolated_heterounits(mol,path)
    if units is None or not 0<len(units)<4:return None
    if any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6 for i in ends):return None
    def prefix(p):
        if not p:return ''
        carbon=mol.GetAtomWithIdx(p[0]).GetAtomicNum()==6
        width=next((j for j,i in enumerate(p) if (mol.GetAtomWithIdx(i).GetAtomicNum()==6)!=carbon),len(p))
        block,remaining=p[:width],p[width:]
        if carbon:
            if width not in ROOTS:return ''
            parent=substituent_parent_suffix(ROOTS[width]+'ane')
            tail=prefix(remaining)
            if remaining and not tail:return ''
            # This is an explicitly typed substituent parent beginning with
            # letters. An internal locant in the distal prefix does not
            # create a hyphen before methyl/ethyl/... (P-16.2.4.1).
            return (format_prefixes([(str(width),tail)],omit_locants=width==1,
                                    hierarchical_enclosures=True)+parent) if tail else parent
        tail=prefix(remaining)
        if not tail:return ''
        elements=tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in block)
        if elements==(8,):return alkyl_to_alkoxy(tail)
        if elements==(8,8):
            # P-63.3.1: a named O-O heterounit is peroxy, including below
            # the P-51.4.1.1 replacement threshold. Complex distal groups
            # retain their component boundary inside this compound prefix.
            return (tail if tail.isalpha() else enclose_name_unit(tail))+'peroxy'
        if elements==(16,8,16):
            stem=tail if tail.isalpha() else enclose_name_unit(tail)
            return enclose_name_unit(enclose_name_unit(stem+'sulfanyl')+'oxy')+'sulfanyl'
        if elements in {(16,),(16,16)}:
            stem=tail if tail.isalpha() else '('+tail+')'
            return stem+('disulfanyl' if len(elements)==2 else 'sulfanyl')
        return ''
    choices=[]
    blocks=[tuple(items) for carbon,items in groupby(path,key=lambda i:mol.GetAtomWithIdx(i).GetAtomicNum()==6) if carbon]
    for block in blocks:
        for oriented in (block,tuple(reversed(block))):
            if len(oriented) not in ROOTS:continue
            prefixes=[]
            for endpoint,direction in ((oriented[0],-1 if path.index(oriented[0])<path.index(oriented[-1]) else 1),
                                        (oriented[-1],1 if path.index(oriented[0])<path.index(oriented[-1]) else -1)):
                j=path.index(endpoint)+direction
                if not 0<=j<len(path):continue
                side=tuple(path[j::-1]) if direction==-1 else path[j:]
                text=prefix(side)
                if not text:break
                prefixes.append((str(oriented.index(endpoint)+1),text))
            else:
                omit=len(oriented)==1 or len(oriented)==2 and len(prefixes)==1
                name=join_prefix_parent(format_prefixes(prefixes,omit_locants=omit,
                    hierarchical_enclosures=True),ROOTS[len(oriented)]+'ane')
                key=(-len(oriented),-len(prefixes),tuple(sorted(int(loc) for loc,_ in prefixes)),name)
                choices.append((key,name,oriented))
    if not choices:return None
    _,name,parent=min(choices)
    return MethodDecision(name,'substitutive',frozenset(),all_atoms,'P-51.1.1/P-51.4.1.1',
                          tuple((idx,str(j)) for j,idx in enumerate(parent,1)),tuple(units))
