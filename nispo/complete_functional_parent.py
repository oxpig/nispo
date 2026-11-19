"""Mapped complete suffixes on acyclic and monocyclic functional parents.

Coverage and chemical numbering are selected before presentation. Mixed ester
owners retain their attachment locants; polyamide N atoms have site locants.
Unsupported overlapping components and incomplete annotation maps decline.
"""

from dataclasses import dataclass, field, replace
from typing import Callable

from .molecule_iteration import iter_atoms, iter_bonds
from .molecule_cache import mol_cache_dict, naming_cache_state
from .graph_model import graph_view
from .component_mapping import (
    MappedComponent,
    decorate_component,
    fully_covered,
    hydrogen_bindings,
    mapped_branch,
    mapped_branch_ring_atoms,
)
from .extended_inventory import selection_inventory
from .functional_inventory import functional_inventory
from .multifunctional_parent import _paths, _suffix_parent
from .formatting import (
    complete_parent_name,
    format_prefixes,
    prefix_alphabetization_key,
)
from .branch_rules import parent_chain_name
from .constants import ROOTS, MULT
from .parent_selection import graph_parent_facts, parent_frontier, ParentFacts
from .functional_parent_admission import parent_admission_evidence
from .nonstandard_parent_evidence import build_substituent_nonstandard_proof
from .selection_evidence import CandidateCompleteness, proposal_identity

_CARBON = {"acid", "carboxylate", "ester", "amide", "nitrile", "ketone", "alcohol"}
_SULFUR = {"sulfonic-acid", "sulfinic-acid", "sulfonate", "sulfinate", "sulfonamide", "thiol"}


@dataclass(frozen=True, slots=True)
class FunctionalParent:
    """A rendered parent with explicit suffix coverage and input-atom locants."""

    name: str
    principal_centers: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    parent_atoms: tuple[int, ...]
    numbering_key: tuple
    prefix_locants: tuple[int, ...]
    prefixes: tuple[tuple[str, str], ...]
    component_name: str
    ester_owners: tuple[str, ...]
    parent_selection: ParentFacts | None = None
    construction: MappedComponent | None = None
    # Original source groups carried as suffixes versus mapped prefixes.
    # Existing callers may omit these additive fields.
    omitted_group_centers: frozenset[int] = frozenset()
    suffix_mode: str = ""
    omitted_prefix_atom_owners: tuple[tuple[int, frozenset[int]], ...] = ()
    parent_annotation_proof: object = None
    # Candidate-local prefix graph partition. This never certifies the global
    # candidate universe or P-45/P-51 method eligibility.
    prefix_graph_evidence: object = None


@dataclass(frozen=True, slots=True)
class FunctionalParentPlan:
    """Call-local legacy construction; no name or eligibility is implied."""

    parent_atoms: tuple[int, ...]
    principal_centers: frozenset[int]
    bound: ParentFacts | None
    construct: Callable[[], FunctionalParent | None] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class SeniorRingParentProof:
    """Original-graph full-suffix parent and composite-method exclusion.

    A distinct pendant ring does not carry a copy of this functionalized
    parent. Four or more cyclic systems and direct inter-ring bonds retain
    their separate phane/assembly obligations instead of using this proof.
    """

    parent_atoms: frozenset[int]
    principal_centers: frozenset[int]
    principal_kind: str
    ring_system_atoms: tuple[frozenset[int], ...]
    principal_group_atoms: tuple[frozenset[int], ...]
    maximum_disjoint_copies: int = 1
    rules: tuple[str, ...] = (
        "P-44.1.1", "P-44.1.2", "P-45.0", "P-45.1.1",
        "P-52.2.5.1", "P-52.2.7", "P-52.2.8",
    )


@dataclass(frozen=True, slots=True)
class FunctionalParentFrontier:
    """Status-bearing result for the pre-render, bounded chain comparison.

    ``complete`` means every structural maximum rendered and the remaining
    mapped comparison has a unique preferred name. ``tie`` retains equivalent
    preferred records; ``conflict`` reports distinct rendered maxima;
    ``blocked`` means at least one maximum could not be rendered. ``out_of_scope``
    means this operation did not prove the bounded all-carbon acyclic scope.
    ``structural_maxima`` retains graph-only maxima before isotope/stereo rank
    is applied in the annotated extension.
    """

    status: str
    candidates: tuple[FunctionalParent, ...] = ()
    maximal_skeletons: tuple[tuple[int, ...], ...] = ()
    reason: str = ""
    structural_maxima: tuple[tuple[int, ...], ...] = ()
    admission_evidence: object = None
    suffix_coverage: int | None = None
    graph_parent_rank: tuple[int, int, int, int] | None = None
    source_parent_rules: tuple[str, ...] = ()
    annotation_proofs: tuple = ()
    # All Job391 proposals considered on the live suffix-coverage maxima.
    prefix_graph_evidence: tuple = ()
    # A graph/source-proved parent outside this carbon-only writer still
    # requires a complete proposal from its owning route. Empty rendered
    # pools must retain this obligation rather than become out of scope.
    required_parent_atoms: tuple[int, ...] = ()
    required_parent_proof: SeniorRingParentProof | None = None
    candidate_completeness: CandidateCompleteness | None = None
    compared_candidates: tuple = ()


def complete_functional_parent(mol, Chem, *, nitrogen_oxo_higher_class=False):
    inventory = selection_inventory(mol)
    if inventory.principal and inventory.principal[0].kind == "ammonium":
        from .aminium_parent import aminium_parent

        return aminium_parent(mol, Chem, inventory)
    frontier = functional_parent_frontier(
        mol, Chem, nitrogen_oxo_higher_class=nitrogen_oxo_higher_class,
    )
    if frontier.status in {"blocked", "conflict"}:
        return ""
    # A legacy enumeration does not prove its own completeness. The principal
    # pipeline can deliver it explicitly as a general name with an obligation.
    candidates = frontier.candidates if frontier.status in {"complete", "tie"} else ()
    return candidates[0].name if candidates else ""


def functional_parent_frontier(mol, Chem, *, include_single=False,
                                excluded_skeletal_atoms=frozenset(),
                                mapped_ring_prefixes=False, principal=None,
                                renderer=None, nitrogen_oxo_higher_class=False):
    """Return the source-status frontier before any candidate is rendered.

    The operation is limited to complete-group, same-class, all-carbon
    acyclic candidates and one physical simple carbon monocycle. Its annotated
    extension admits only completely mapped isotope, ordinary tetrahedral R/S,
    and (for chains) E/Z facts. It is additive: existing
    callers of ``functional_parent_candidates`` retain their historical tuple
    contract.
    The optional renderer is a focused control seam for injecting a local
    rendering failure on a real graph.
    """
    result = functional_parent_candidates(
        mol, Chem, include_single=include_single,
        excluded_skeletal_atoms=excluded_skeletal_atoms,
        mapped_ring_prefixes=mapped_ring_prefixes, principal=principal,
        return_frontier=True, renderer=renderer,
        nitrogen_oxo_higher_class=nitrogen_oxo_higher_class,
    )
    if isinstance(result, FunctionalParentFrontier):
        return result
    return FunctionalParentFrontier(
        "out_of_scope", reason="Existing candidate eligibility did not establish the bounded frontier scope."
    )


def functional_parent_candidates(mol, Chem, *, include_single=False, excluded_skeletal_atoms=frozenset(),
                                 mapped_ring_prefixes=False, principal=None,
                                 return_frontier=False, renderer=None,
                                 nitrogen_oxo_higher_class=False, defer_render=False):
    """Expose chemical coverage before comparing nomenclature methods.

    ``include_single`` admits isolated parent components after a connector has
    been removed. It does not change the established whole-molecule scope.
    ``excluded_skeletal_atoms`` keeps indivisible subordinate prefix units
    outside the skeletal parent without removing them from branch ownership.
    """
    inventory = selection_inventory(mol)
    effective_principal = (tuple(inventory.principal)
                           if principal is None else tuple(principal))
    # P-66.5: a subordinate nitrile is the complete C#N cyano unit.
    # Enforce that role at the shared producer boundary, including method
    # selection and isolated acid components whose callers supply no explicit
    # exclusions. Absorbing its carbon into an alkane path would leave a
    # triple-bonded terminal nitrogen to be mistaken for an amino prefix.
    excluded_skeletal_atoms = frozenset(excluded_skeletal_atoms) | frozenset(
        group.center for group in inventory.groups
        if group.kind == 'nitrile' and group not in effective_principal)
    # A custom renderer is an explicit request to execute construction.
    if renderer is not None:
        return _functional_parent_candidates(
            mol, Chem, include_single=include_single,
            excluded_skeletal_atoms=excluded_skeletal_atoms,
            mapped_ring_prefixes=mapped_ring_prefixes, principal=principal,
            return_frontier=return_frontier, renderer=renderer,
            nitrogen_oxo_higher_class=nitrogen_oxo_higher_class,
            defer_render=defer_render)
    if principal is not None:
        principal = effective_principal
    try:
        hash(effective_principal)
        principal_key = effective_principal
    except TypeError:
        principal_key = tuple(id(group) for group in effective_principal)
    state = naming_cache_state(mol)
    key = (id(Chem), id(_render), include_single, frozenset(excluded_skeletal_atoms),
           mapped_ring_prefixes,
           principal_key,
           return_frontier, nitrogen_oxo_higher_class, state)
    cache = mol_cache_dict(mol, 'functional_parent_candidates')
    if defer_render and key not in cache:
        # Reuse an existing eager result when an earlier route already built
        # this exact pool. New callbacks remain call-local: storing them on
        # their owning molecule would retain it through a reference cycle.
        return _functional_parent_candidates(
            mol, Chem, include_single=include_single,
            excluded_skeletal_atoms=excluded_skeletal_atoms,
            mapped_ring_prefixes=mapped_ring_prefixes, principal=principal,
            return_frontier=return_frontier,
            nitrogen_oxo_higher_class=nitrogen_oxo_higher_class,
            defer_render=True,
        )
    if key not in cache:
        result = _functional_parent_candidates(
            mol, Chem, include_single=include_single,
            excluded_skeletal_atoms=excluded_skeletal_atoms,
            mapped_ring_prefixes=mapped_ring_prefixes, principal=principal,
            return_frontier=return_frontier,
            nitrogen_oxo_higher_class=nitrogen_oxo_higher_class)
        # Retain identity-keyed groups so their IDs cannot be recycled while
        # a result remains cached on this molecule.
        cache[key] = (effective_principal, result)
    return cache[key][1]


def _functional_parent_candidates(mol, Chem, *, include_single=False,
                                  excluded_skeletal_atoms=frozenset(),
                                  mapped_ring_prefixes=False, principal=None,
                                  return_frontier=False, renderer=None,
                                  nitrogen_oxo_higher_class=False, defer_render=False):
    inventory = selection_inventory(mol)
    if return_frontier and not inventory.supported:
        return FunctionalParentFrontier(
            "out_of_scope", reason="Principal-group inventory is not fully supported."
        )
    # Recognition/class seniority is distinct from the P-82.2.2.2 operation
    # that can select unlike isotope-modified groups for suffix citation.
    groups = inventory.principal if principal is None else tuple(principal)
    if any(group not in inventory.groups for group in groups):
        return ()
    if not groups:
        return ()
    admission = parent_admission_evidence(
        mol, Chem, inventory, groups,
        nitrogen_oxo_higher_class=nitrogen_oxo_higher_class,
    )
    if admission.status == "barrier":
        if return_frontier:
            return FunctionalParentFrontier(
                "blocked", reason=admission.reason,
                admission_evidence=admission,
            )
        return ()
    obligation = admission.status == "admitted"
    kind = groups[0].kind
    if any(g.kind != kind for g in groups):
        return ()
    charge_role_proofs = tuple(getattr(inventory, "charge_role_proofs", ()))
    charge_prefix_proofs = tuple(
        proof for proof in charge_role_proofs
        if getattr(proof, "kind", None) in {"nitro", "azido"}
    )
    charge_prefix_obligation = False
    if charge_prefix_proofs and tuple(groups) == tuple(inventory.principal):
        from .neutral_charge_components import (
            neutral_charge_prefix_facts, validate_neutral_charge_role_proofs,
        )

        prefix_facts = neutral_charge_prefix_facts(mol, charge_role_proofs)
        charge_prefix_obligation = bool(
            inventory.supported
            and prefix_facts is not None
            and len(charge_prefix_proofs) == len(charge_role_proofs)
            and len(prefix_facts) == len(charge_prefix_proofs)
            and validate_neutral_charge_role_proofs(mol, charge_role_proofs)
        )
    legacy = functional_inventory(mol)
    established_classes = (_CARBON - {"ketone", "alcohol"}) | _SULFUR
    if (kind not in established_classes
            and not (charge_prefix_obligation and kind in {"ketone", "alcohol"})):
        return ()
    ring_proof = (_source_senior_ring_parent(mol, Chem, inventory, groups)
                  if return_frontier else None)
    if ring_proof is not None:
        # This frontier constructs carbon parents. Its unavailable one-carbon
        # writer cannot block an intrinsically senior heterocycle carrying
        # every selected suffix. The shared pool still requires the mapped
        # ring writer and retains an unrendered ring witness if it fails.
        return FunctionalParentFrontier(
            "out_of_scope", reason="Source-proved full-suffix ring is senior to pendant rings and carbon-chain candidates; hand off to its complete mapped writer.",
            source_parent_rules=("P-41", *ring_proof.rules),
            admission_evidence=admission,
            required_parent_atoms=tuple(sorted(ring_proof.parent_atoms)),
            required_parent_proof=ring_proof)
    single_ring_suffix_handoff = bool(
        kind in {"amide", "nitrile"}
        and all(group.owner >= 0
                and mol.GetAtomWithIdx(group.owner).IsInRing()
                for group in groups)
        and len(mol.GetRingInfo().AtomRings()) == 1
    )
    mono_legacy_precedence = bool(
        len(groups) == 1
        and not include_single
        and legacy.supported
        and not any(a.GetIsotope() or int(a.GetChiralTag()) for a in iter_atoms(mol))
        and not any(int(b.GetStereo()) or int(b.GetBondDir()) for b in iter_bonds(mol))
        and not obligation
        and not charge_prefix_obligation
        and not single_ring_suffix_handoff
    )
    implicit_stereo_trigger = None
    if (mono_legacy_precedence and return_frontier
            and not any(int(atom.GetChiralTag()) for atom in iter_atoms(mol))
            and not any(int(bond.GetStereo()) or int(bond.GetBondDir())
                        for bond in iter_bonds(mol))):
        from .parent_annotation_proof import (
            PotentialStereoTriggerStatus, implicit_unknown_monocycle_trigger,
        )
        implicit_stereo_trigger = implicit_unknown_monocycle_trigger(
            mol, Chem, principal_groups=groups, principal_class=kind,
        )
        if implicit_stereo_trigger.status is PotentialStereoTriggerStatus.CANDIDATE:
            # The typed unknown-stereo proof takes precedence over the
            # historical mono-functional early return for this exact ring.
            mono_legacy_precedence = False
        elif implicit_stereo_trigger.status is PotentialStereoTriggerStatus.UNRESOLVED:
            return FunctionalParentFrontier(
                "blocked", reason=implicit_stereo_trigger.reason,
                admission_evidence=admission,
            )
    if mono_legacy_precedence and not return_frontier:
        return ()  # Preserve established tuple-API mono-functional precedence.
    if (any(mol.GetAtomWithIdx(g.center).IsInRing() for g in groups)
            and not (charge_prefix_obligation and all(
                mol.GetAtomWithIdx(g.center).IsInRing() for g in groups
            ))):
        return ()
    # Partially protonated polyacids require acid-salt grammar, not omission of
    # the residual acid group. Leave them to existing acid-salt renderers.
    if kind == "carboxylate" and any(g.kind == "acid" for g in inventory.groups):
        return ()
    skipped = set()
    owner_components = []
    for group in groups:
        skipped.update(i for i in group.atoms if i != group.center)
        if kind == "ester":
            oxygen = mol.GetAtomWithIdx(group.ligand)
            roots = [
                a.GetIdx() for a in oxygen.GetNeighbors() if a.GetIdx() != group.center
            ]
            if len(roots) != 1:
                return ()
            text, atoms = mapped_branch(mol, Chem, roots[0], group.ligand)
            if not text or atoms & {g.center for g in groups}:
                return ()
            skipped.update(atoms)
            owner_components.append((text, atoms))
        elif kind in {"amide", "sulfonamide"}:
            if kind == "sulfonamide":
                from .sulfur_selection import sulfonamide_n_sites

                n_sites = sulfonamide_n_sites(mol, group)
                if n_sites is None:
                    return ()
            else:
                nitrogen = mol.GetAtomWithIdx(group.ligand)
                n_sites = tuple(
                    n.GetIdx() for n in nitrogen.GetNeighbors()
                    if n.GetIdx() != group.center and n.GetAtomicNum() > 1
                )
            for n_idx in n_sites:
                text, atoms = mapped_branch(mol, Chem, n_idx, group.ligand)
                if not text or atoms & {g.center for g in groups}:
                    return ()
                skipped.update(atoms)
    centers = {g.center for g in groups}
    unavailable = skipped | set(excluded_skeletal_atoms)
    carbon = {
        a.GetIdx()
        for a in iter_atoms(mol)
        if a.GetAtomicNum() == 6 and a.GetIdx() not in unavailable
    }
    if len(carbon) > 40:
        return ()
    if return_frontier:
        if include_single or tuple(groups) != tuple(inventory.principal):
            return FunctionalParentFrontier(
                "out_of_scope", reason="The operation requires the complete principal-group set outside fragment mode."
            )
        if kind in {"acid", "amide", "nitrile"}:
            from .suffix_coverage_frontier import (
                SuffixCoverageBarrier, maximum_suffix_coverage,
            )

            # This graph-only frontier requires the complete carbon universe.
            # ``carbon`` above has already removed mapped N-/O-substituent
            # branches for writing, so passing it here incorrectly reported
            # legitimate functional-class branches as arbitrary omissions.
            # A heteroatom-separated carbon graph is outside this tree proof;
            # the existing mapped parent writer still owns those branches.
            coverage_carbon = carbon | {
                idx for idx in skipped if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
                and idx not in excluded_skeletal_atoms
            }
            coverage = maximum_suffix_coverage(
                mol, Chem, groups, coverage_carbon,
                excluded_skeletal_atoms=excluded_skeletal_atoms,
            )
            if coverage.barrier is SuffixCoverageBarrier.SOURCE_ROLE_COMPLETENESS:
                return FunctionalParentFrontier(
                    "blocked", reason=coverage.reason,
                    admission_evidence=admission,
                    suffix_coverage=None,
                )
            if coverage.status == "proved":
                # P-66.5.1.2.1 preserves the PIN parent vocabulary for the
                # shortest carbonitrile parents (including formonitrile,
                # acetonitrile, and oxalonitrile). Keep those constructions
                # on their established retained-parent route rather than
                # substituting a generic alkane+nitrile rendering here.
                if kind == "nitrile" and coverage.maximum_length <= 2:
                    return FunctionalParentFrontier(
                        "out_of_scope",
                        reason="Short carbonitrile parent uses the retained P-66.5.1.2.1 vocabulary route.",
                        admission_evidence=admission,
                    )
                # Mono-functional precedence is deferred only long enough to
                # recognize the narrow Job391 graph profile. Every other
                # single-site family keeps its historical route.
                if (mono_legacy_precedence
                        and not _job391_suffix_scope(mol, groups, coverage)):
                    return FunctionalParentFrontier(
                        "out_of_scope",
                        reason="Single-site legacy precedence applies outside the bounded prefix-evidence profile.",
                        admission_evidence=admission,
                    )
                # The graph frontier covers only an unannotated, acyclic
                # all-carbon tree. Within that source-closed family it must
                # compare every eligible attachment mode before rendering,
                # even when all groups fit on one path.
                return _functional_suffix_coverage_frontier(
                    mol, Chem, groups, kind, coverage, skipped,
                    owner_components, mapped_ring_prefixes=mapped_ring_prefixes,
                    renderer=renderer, charge_role_proofs=charge_role_proofs,
                    admission=admission,
                )
            if mono_legacy_precedence:
                return FunctionalParentFrontier(
                    "out_of_scope",
                    reason="Single-site legacy precedence applies because the bounded graph frontier was not proved.",
                    admission_evidence=admission,
                )
            if (coverage.status == "out_of_scope" and len(groups) > 1
                    and coverage.barrier is SuffixCoverageBarrier.ATTACHMENT_COMPLETENESS):
                return FunctionalParentFrontier(
                    "blocked", reason=coverage.reason,
                    admission_evidence=admission,
                )
        skeletons = _legacy_functional_skeletons(mol, groups, kind, carbon, centers)
        frontier = _pre_render_functional_frontier(
            mol, Chem, groups, kind, skeletons, skipped, owner_components,
            mapped_ring_prefixes=mapped_ring_prefixes, renderer=renderer,
            charge_role_proofs=charge_role_proofs,
        )
        return replace(
            frontier,
            admission_evidence=admission,
            suffix_coverage=(len(groups) if skeletons else None),
        )
    skeletons = _legacy_functional_skeletons(mol, groups, kind, carbon, centers)
    if not include_single:
        # Select on original skeleton facts before rendering. Otherwise a
        # renderer failure on the more senior chain could promote a shorter
        # or less unsaturated chain merely because that one can be written.
        skeletons = _acyclic_carbon_parent_seniority(mol, skeletons, centers)
    results = []
    for path, external, cyclic in skeletons:
        if defer_render and include_single:
            from .proposal_plans import graph_proposal_bound
            def construct(path=tuple(path), external=external, cyclic=cyclic):
                return _render_functional_candidate(
                    mol, Chem, groups, kind, path, external, cyclic, skipped, owner_components,
                    mapped_ring_prefixes=mapped_ring_prefixes,
                    charge_role_proofs=charge_role_proofs,
                )
            results.append(FunctionalParentPlan(
                tuple(path), frozenset(group.center for group in groups),
                graph_proposal_bound(mol, path, groups, cyclic=cyclic), construct,
            ))
            continue
        record = _render_functional_candidate(
            mol, Chem, groups, kind, path, external, cyclic, skipped, owner_components,
            mapped_ring_prefixes=mapped_ring_prefixes, charge_role_proofs=charge_role_proofs,
        )
        if record is not None:
            results.append(record)
    if defer_render and include_single:
        return tuple(results)
    # A capped component's linking prefix is not known yet. Its caller must
    # retain all orientations until that attachment joins the prefix locants.
    if not include_single:
        results = parent_frontier(results, lambda record: record.parent_selection)
    return tuple(sorted(results, key=lambda record: record.numbering_key))


def _acyclic_carbon_parent_seniority(mol, skeletons, centers):
    """Apply the source-ordered graph criteria to comparable legacy chains.

    The legacy parent-fact adapter can decline a candidate when a nested
    substituent's charge ownership is unavailable. Such a missing
    prefix proof cannot reverse P-44's earlier, graph-only comparison of two
    all-carbon acyclic chains carrying the same suffix centers. Keep rings,
    external suffixes and other unproved candidate kinds on their own routes.
    This is a pruning step, not a proof of complete candidate enumeration.
    """
    comparable = []
    for index, (path, external, cyclic) in enumerate(skeletons):
        if (not path or external or cyclic or not centers <= set(path)
                or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                       or mol.GetAtomWithIdx(i).IsInRing() for i in path)):
            continue
        bonds = tuple(mol.GetBondBetweenAtoms(a, b)
                      for a, b in zip(path, path[1:]))
        if any(bond is None or bond.GetBondTypeAsDouble() not in (1, 2, 3)
               for bond in bonds):
            continue
        orders = tuple(bond.GetBondTypeAsDouble() for bond in bonds)
        key = (-len(path), -sum(order in (2, 3) for order in orders),
               -sum(order == 2 for order in orders))
        comparable.append((index, key))
    if len(comparable) < 2:
        return skeletons
    best = min(key for _, key in comparable)
    dominated = {index for index, key in comparable if key != best}
    return [skeleton for index, skeleton in enumerate(skeletons)
            if index not in dominated]


def _legacy_functional_skeletons(mol, groups, kind, carbon, centers):
    """Build legacy paths only when the graph-only frontier did not return."""
    skeletons = []
    # Carbonyl carbons can enter a terminal acyclic suffix, or remain external
    # to a ring/chain carrying carboxamide/carboxylate/carbonitrile suffixes.
    for external in [False, True]:
        if not external and kind in _SULFUR:
            continue
        required = {g.owner if external else g.center for g in groups}
        allowed = {i for i in carbon if not mol.GetAtomWithIdx(i).IsInRing()} - (
            centers if external else set()
        )
        if required <= allowed and (not external or len(groups) > 2 or kind in _SULFUR):
            for path in _paths(mol, allowed):
                if not required <= set(path):
                    continue
                if (not external and kind not in {"ketone", "alcohol"} and any(
                    path.index(i) not in {0, len(path) - 1} for i in required
                )):
                    continue
                for oriented in [path, tuple(reversed(path))]:
                    skeletons.append((oriented, external, False))
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring) and len(ring) != 6:
            continue  # Other aromatic monocycles need their own hydride grammar.
        ring_sites = {
            g.center if mol.GetAtomWithIdx(g.center).IsInRing() else g.owner
            for g in groups
        }
        if not ring_sites <= set(ring) or not set(ring) <= carbon:
            continue
        if any(
            sum(i in other for other in mol.GetRingInfo().AtomRings()) != 1
            for i in ring
        ):
            continue
        if not (
            all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
            or all(
                mol.GetBondBetweenAtoms(
                    ring[j], ring[(j + 1) % len(ring)]
                ).GetBondTypeAsDouble()
                == 1
                for j in range(len(ring))
            )
        ):
            continue
        external_ring_suffix = any(
            not mol.GetAtomWithIdx(g.center).IsInRing() for g in groups
        )
        for direction in [ring, tuple(reversed(ring))]:
            for start in range(len(ring)):
                skeletons.append((direction[start:] + direction[:start], external_ring_suffix, True))
    return skeletons


def _render_functional_candidate(mol, Chem, groups, kind, path, external, cyclic,
                                  skipped, owner_components, *, mapped_ring_prefixes,
                                  charge_role_proofs, locanted_acid_fallback=False,
                                  general_branches=False, branch_resolver=None):
    record = _render(
        mol, Chem, groups, kind, path, external, cyclic, skipped, owner_components,
        mapped_ring_prefixes=mapped_ring_prefixes, return_component=True,
        charge_role_proofs=charge_role_proofs,
        locanted_acid_fallback=locanted_acid_fallback,
        general_branches=general_branches,
        branch_resolver=branch_resolver,
    )
    if record:
        record, construction = record
        name, locants, prefix_locs, mapping, prefixes, component_name, owners = record
        multiple = tuple(
            j
            for j, (a, b) in enumerate(zip(path, path[1:]), 1)
            if mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
        )
        key = (-len(groups), -len(path), locants, multiple, prefix_locs, name)
        facts = graph_parent_facts(
            mol, path, mapping, principal_class=groups[0].seniority,
            suffix_centers=[g.center for g in groups], suffix_locants=locants,
            prefixes=prefixes, kind="ring" if cyclic else "chain", name=name,
            allow_annotations=True,
            charge_role_proofs=charge_role_proofs,
            prefix_branch_ownership=construction.prefix_branch_ownership,
            substituent_nonstandard_proof=build_substituent_nonstandard_proof(
                mol, path, construction, prefixes, mapping, groups,
            ),
        )
        return FunctionalParent(
            name, frozenset(g.center for g in groups), tuple(mapping.items()),
            tuple(path), key, prefix_locs, prefixes, component_name, owners, facts, construction,
        )


def _functional_suffix_coverage_frontier(
    mol, Chem, groups, kind, coverage, skipped, owner_components, *,
    mapped_ring_prefixes=False, renderer=None, charge_role_proofs=(), admission=None,
):
    """Render every graph maximum from the bounded partial-coverage frontier."""
    from .method_parent_frontier import select_method_parent, MethodParentStatus

    coverage_maxima = tuple(coverage.candidates)
    # The graph helper has already applied P-44.1.1, P-44.3.2, and
    # P-44.4.1(a)/(b) to the complete parent universe. Every remaining record
    # is a structural maximum and survives renderer failure.
    maxima = coverage_maxima
    rendered = []
    failures = []
    use_prefix_graph_evidence = _job391_suffix_scope(mol, groups, coverage)
    prefix_evidence_rows = []
    prefix_validator = None
    if use_prefix_graph_evidence and renderer is None:
        from .functional_parent_prefix_evidence391 import _FunctionalPrefixEvidenceValidator

        prefix_validator = _FunctionalPrefixEvidenceValidator()
    for candidate in maxima:
        for path in (candidate.path, tuple(reversed(candidate.path))):
            prefix_evidence = None
            render_proofs = ()
            if use_prefix_graph_evidence:
                from .functional_parent_prefix_evidence391 import (
                    PrefixEvidenceStatus,
                    build_functional_parent_prefix_evidence391,
                    validate_functional_parent_prefix_evidence391,
                )

                prefix_evidence = build_functional_parent_prefix_evidence391(
                    mol, Chem, path, groups, suffix_mode="chain-included",
                    resolve_branch_tokens=True,
                )
                prefix_evidence_rows.append(prefix_evidence)
                validate_prefix_evidence = (
                    validate_functional_parent_prefix_evidence391
                    if prefix_validator is None else prefix_validator.validate
                )
                if (prefix_evidence.status is not PrefixEvidenceStatus.COMPLETE
                        or not validate_prefix_evidence(
                            mol, Chem, prefix_evidence, groups,
                            suffix_mode="chain-included",
                        )
                        or not _job391_candidate_agrees(
                            mol, candidate, path, groups, prefix_evidence,
                        )):
                    failures.append((tuple(path), candidate.mode, "prefix-graph"))
                    continue
                # The independently rebuilt partition has already proved
                # every ordinary carbon component and direct source role.
                # Only its functionalized carbon components can require a
                # writer-backed functionalized-branch proof. Do not repeat
                # whole-graph signatures and cut traversals merely to prove
                # that ordinary carbon components have no such obligation.
                if any(unit.kind == "functionalized-carbon-component"
                       for unit in (prefix_evidence.prefix_units or ())):
                    render_proofs = _functionalized_branch_proofs(
                        mol, Chem, path, groups,
                        reserved_direct_prefix_atoms=(
                            atom for unit in (prefix_evidence.prefix_units or ())
                            if unit.root is None
                            for atom in unit.atoms
                        ),
                    )
                if render_proofs is None:
                    failures.append((tuple(path), candidate.mode, "branch-writer-proof"))
                    continue
            render_fn = renderer or _render
            render_kwargs = {
                "mapped_ring_prefixes": mapped_ring_prefixes,
                "return_component": True,
            }
            if charge_role_proofs:
                render_kwargs["charge_role_proofs"] = charge_role_proofs
            if render_proofs:
                render_kwargs["functionalized_carbon_prefix_proofs"] = render_proofs
            record = render_fn(
                mol, Chem, candidate.selected_groups, kind, path,
                candidate.mode == "external", False, skipped, owner_components,
                **render_kwargs,
            )
            if not record:
                failures.append((tuple(path), candidate.mode))
                continue
            record, construction = record
            name, locants, prefix_locs, mapping, prefixes, component_name, owners = record
            if (prefix_evidence is not None
                    and not _job391_renderer_ownership_agrees(
                        mol, path, candidate, groups, prefix_evidence,
                        construction, prefixes, mapping,
                    )):
                failures.append((tuple(path), candidate.mode, "prefix-writer-ownership"))
                continue
            facts = graph_parent_facts(
                mol, path, mapping,
                principal_class=candidate.selected_groups[0].seniority,
                suffix_centers=[group.center for group in candidate.selected_groups],
                suffix_locants=locants, prefixes=prefixes, kind="chain", name=name,
                allow_annotations=False, charge_role_proofs=charge_role_proofs,
                substituent_nonstandard_proof=build_substituent_nonstandard_proof(
                    mol, path, construction, prefixes, mapping, candidate.selected_groups,
                ),
            )
            if facts is None:
                failures.append((tuple(path), candidate.mode))
                continue
            omitted_owners = _omitted_prefix_atom_ownership(
                mol, candidate.omitted_groups, construction,
            )
            if omitted_owners is None:
                failures.append((tuple(path), candidate.mode))
                continue
            multiple = tuple(
                index for index, (left, right) in enumerate(zip(path, path[1:]), 1)
                if mol.GetBondBetweenAtoms(left, right).GetBondTypeAsDouble() != 1
            )
            selected_centers = frozenset(group.center for group in candidate.selected_groups)
            omitted_centers = frozenset(group.center for group in candidate.omitted_groups)
            key = (-len(selected_centers), -len(path), locants, multiple, prefix_locs, name)
            rendered.append(FunctionalParent(
                name, selected_centers, tuple(mapping.items()), tuple(path), key,
                prefix_locs, prefixes, component_name, owners, facts, construction,
                omitted_centers, candidate.mode, omitted_owners,
                prefix_graph_evidence=prefix_evidence,
            ))
    structural_paths = tuple(tuple(candidate.path) for candidate in maxima)
    if failures:
        return FunctionalParentFrontier(
            "blocked", tuple(rendered), structural_paths,
            f"Renderer failed for {len(failures)} P-44.1.1-maximal suffix-coverage candidate(s).",
            structural_paths, admission, coverage.maximum,
            _coverage_graph_rank(coverage),
            prefix_graph_evidence=tuple(prefix_evidence_rows),
        )
    if not rendered:
        return FunctionalParentFrontier(
            "blocked", (), structural_paths,
            "No P-44.1.1-maximal suffix-coverage candidate rendered.",
            structural_paths, admission, coverage.maximum,
            _coverage_graph_rank(coverage),
            prefix_graph_evidence=tuple(prefix_evidence_rows),
        )
    completeness = CandidateCompleteness(
        'ordinary-carbon-tree-suffix-coverage',
        ('P-41/P-43', 'P-44.1.1', 'P-44.3', 'P-44.4'),
        structural_paths, tuple(tuple(record.parent_atoms) for record in rendered),
        'Every oriented graph maximum received a complete mapped suffix/prefix construction.',
        proposal_keys=tuple(proposal_identity(record) for record in rendered),
    )
    ranked = select_method_parent(
        rendered, lambda record: record.parent_selection,
        physical_parent_identity=lambda record: (
            frozenset(record.parent_atoms), record.principal_centers),
        completeness=completeness,
    )
    preferred = ranked.candidates
    if ranked.status == MethodParentStatus.UNRESOLVED:
        return FunctionalParentFrontier(
            "conflict", tuple(preferred),
            structural_paths,
            ranked.reason,
            structural_paths, admission, coverage.maximum,
            _coverage_graph_rank(coverage),
            prefix_graph_evidence=tuple(prefix_evidence_rows),
            candidate_completeness=completeness, compared_candidates=tuple(rendered),
        )
    status = "complete" if len(preferred) == 1 else "tie"
    return FunctionalParentFrontier(
        status, tuple(preferred), structural_paths,
        "All graph-maximal suffix-coverage candidates rendered and were compared.",
        structural_paths, admission, coverage.maximum,
        _coverage_graph_rank(coverage),
        prefix_graph_evidence=tuple(prefix_evidence_rows),
        candidate_completeness=completeness, compared_candidates=tuple(rendered),
    )


def _job391_suffix_scope(mol, groups, coverage):
    """Return whether this whole frontier enters the narrow Job391 join.

    A partial, external, unsaturated, annotated, or N-substituted amide
    frontier stays on its established route. Once admitted, unsupported
    pendant ownership becomes a barrier rather than an empty-prefix result.
    """
    if (not groups or coverage.maximum != len(groups) or not coverage.candidates
            or any(candidate.mode != "embedded"
                   or len(candidate.selected_groups) != len(groups)
                   or candidate.omitted_groups for candidate in coverage.candidates)):
        return False
    # Job391's parent and carbon-component grammar is saturated. The suffix
    # C=O and C#N bonds are not part of the carbon skeleton and remain allowed.
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if (left.GetAtomicNum() == right.GetAtomicNum() == 6
                and bond.GetBondTypeAsDouble() != 1):
            return False
    # N-substituted amides have existing N-locant/branch routes outside this
    # initial graph evidence contract.
    for group in groups:
        if group.kind != "amide":
            continue
        nitrogen = mol.GetAtomWithIdx(group.ligand)
        if any(neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() != group.center
               for neighbor in nitrogen.GetNeighbors()):
            return False
    # Preserve established routes for source roles outside the 391 inventory.
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if not inventory.supported:
        return False
    principal = tuple(inventory.principal)
    supported_roles = {"alcohol", "amine", "thiol", "nitrile"}
    if any(group not in principal and group.kind not in supported_roles
           for group in inventory.groups):
        return False
    return True


def _job391_candidate_agrees(mol, candidate, path, groups, evidence):
    from .functional_parent_prefix_evidence391 import PrefixEvidenceStatus

    if evidence.status is not PrefixEvidenceStatus.COMPLETE:
        return False
    identities = tuple(sorted(
        (group.kind, int(group.center), tuple(int(i) for i in group.atoms),
         int(group.owner), int(group.ligand)) for group in groups
    ))
    covered = tuple(sorted(
        (group.kind, int(group.center), tuple(int(i) for i in group.atoms),
         int(group.owner), int(group.ligand)) for group in candidate.selected_groups
    ))
    omitted = tuple(sorted(
        (group.kind, int(group.center), tuple(int(i) for i in group.atoms),
         int(group.owner), int(group.ligand)) for group in candidate.omitted_groups
    ))
    expected_suffix_atoms = set()
    for group in candidate.selected_groups:
        expected_suffix_atoms.update(set(group.atoms) - set(path))
    locants = tuple(sorted(path.index(group.center) + 1
                           for group in candidate.selected_groups))
    return (
        tuple(evidence.parent_atoms) == tuple(path)
        and tuple(evidence.selected_principal_identities) == identities
        and tuple(evidence.covered_principal_identities) == covered
        and tuple(evidence.omitted_principal_identities) == omitted
        and evidence.covered_principal_identities == identities
        and not evidence.omitted_principal_identities
        and evidence.p44_principal_site_count == len(candidate.selected_groups)
        and evidence.p44_skeletal_atom_count == len(path)
        and evidence.candidate_suffix_locants == locants
        and set(evidence.principal_suffix_atoms) == expected_suffix_atoms
        and evidence.parent_candidate_universe_proved is False
    )


def _job391_renderer_ownership_agrees(
    mol, path, candidate, groups, evidence, construction, prefixes, mapping,
):
    """Join each graph role to exact ownership returned by the real writer."""
    if (construction is None or evidence.prefix_units is None
            or construction.prefix_branch_ownership is None):
        return False
    path_set = set(path)
    suffix_atoms = set(evidence.principal_suffix_atoms)
    expected_owned = path_set | suffix_atoms
    actual_rows = tuple(construction.prefix_branch_ownership)
    if (set(construction.atoms) != set(mapping)
            or dict(construction.atom_locants) != dict(mapping)
            or tuple(construction.prefixes) != tuple(prefixes)):
        return False
    used_rows = set()
    for unit_index, unit in enumerate(evidence.prefix_units):
        unit_atoms = set(unit.atoms)
        if not unit_atoms or expected_owned & unit_atoms:
            return False
        expected_owned.update(unit_atoms)
        choices = []
        for row_index, row in enumerate(actual_rows):
            if not isinstance(row, tuple) or len(row) != 4:
                return False
            root, host, atoms, text = row
            if (host == unit.host and frozenset(atoms) == frozenset(unit.atoms)
                    and isinstance(text, str) and text):
                if unit.root is not None and root != unit.root:
                    continue
                if unit.root is None and unit.source_group_atoms:
                    source_atoms = set().union(*(set(atoms_) for atoms_ in unit.source_group_atoms))
                    if root not in source_atoms:
                        continue
                choices.append(row_index)
        if len(choices) != 1 or choices[0] in used_rows:
            return False
        row_index = choices[0]
        used_rows.add(row_index)
        _root, host, _atoms, text = actual_rows[row_index]
        locant = mapping.get(host)
        if locant is None or (str(locant), text) not in prefixes:
            return False
    if used_rows != set(range(len(actual_rows))):
        return False
    heavy_atoms = {atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1}
    if expected_owned != heavy_atoms:
        return False
    if set(construction.embedded_atoms) != {
            atom for row in actual_rows for atom in row[2]}:
        return False
    # The mapped ownership ledger and emitted prefix facts must describe the
    # same multiset. This catches extra, missing, or multiply rendered prefix
    # text even when every expected graph unit individually found a match.
    from collections import Counter

    expected_prefix_facts = Counter(
        (str(mapping[row[1]]), str(row[3])) for row in actual_rows
        if row[1] in mapping
    )
    return expected_prefix_facts == Counter(tuple(item) for item in prefixes)


def _coverage_graph_rank(coverage):
    return (
        coverage.maximum,
        coverage.maximum_length,
        coverage.maximum_multiple_bonds,
        coverage.maximum_double_bonds,
    )


def _omitted_prefix_atom_ownership(mol, omitted_groups, construction):
    """Bind each omitted source group to its fully mapped branch component."""
    embedded = set(construction.embedded_atoms)
    result = []
    for group in omitted_groups:
        center = int(group.center)
        if center not in embedded or not set(group.atoms) <= embedded:
            return None
        component, stack = set(), [center]
        while stack:
            idx = stack.pop()
            if idx in component or idx not in embedded:
                continue
            component.add(idx)
            stack.extend(
                neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                if neighbor.GetIdx() in embedded and neighbor.GetAtomicNum() > 1
            )
        if not set(group.atoms) <= component:
            return None
        result.append((center, frozenset(component)))
    return tuple(result)


def _valid_nitrogen_oxo_higher_class_obligation(mol, inventory, groups):
    """Compatibility projection for the former N-oxo boolean gate."""
    return parent_admission_evidence(
        mol, None, inventory, groups, nitrogen_oxo_higher_class=True,
    ).status == "admitted"


def _pre_render_functional_frontier(mol, Chem, groups, kind, skeletons, skipped,
                                    owner_components, *, mapped_ring_prefixes=False,
                                    renderer=None, charge_role_proofs=()):
    """Compare a bounded structural family before attempting its renderers.

    This scope covers same-class, complete-group, acyclic carbon-chain
    candidates or a single physical saturated-carbon or benzene monocycle.
    Mixed chain/ring alternatives, different ring atom sets, unsupported
    annotations, unsupported graph facts, or incomplete structural evidence
    return ``out_of_scope``.
    ``renderer`` exists to let focused controls inject a failure on a real
    candidate; normal callers should leave it unset.
    """
    charge_prefix_proofs = tuple(
        proof for proof in charge_role_proofs
        if getattr(proof, "kind", None) in {"nitro", "azido"}
    )
    if charge_prefix_proofs and skeletons:
        # P-44 compares parent skeleton size before subordinate-prefix
        # citation. Retain every global maximum path (including all equivalent
        # orientations) and remove only strict shorter candidates, including
        # a one-carbon acyclic parent when a larger ring parent is eligible.
        maximum = max(len(path) for path, _external, _cyclic in skeletons)
        skeletons = [
            row for row in skeletons
            if len(row[0]) == maximum
        ]
    if not skeletons:
        return FunctionalParentFrontier("out_of_scope", reason="No complete candidate family was proved.")
    cyclic_flags = {bool(cyclic) for _, _, cyclic in skeletons}
    source_parent_rules = ()
    if len(cyclic_flags) != 1:
        preferred_ring_candidates = _complete_source_class_ring_candidates(
            mol, Chem, groups, kind, skeletons,
        )
        if preferred_ring_candidates is None:
            return FunctionalParentFrontier(
                "out_of_scope", reason="Candidate set mixes chain and ring parents."
            )
        skeletons = preferred_ring_candidates
        cyclic_flags = {True}
        if kind == "acid":
            # Keep the established Job335 audit contract: its acid adapter
            # records the exact source clauses already independently closed.
            source_parent_rules = (
                "P-41", "P-42.1", "P-44.1.1", "P-44.1.2.2", "P-52.2.8",
            )
        else:
            source_parent_rules = (
                "P-41", "P-42", "P-43", "P-44.1.1", "P-44.1.2.2",
                "P-52.2.8", _ring_adapter_rule(kind),
            )
    cyclic = cyclic_flags.pop()
    if cyclic and not _is_bounded_monocycle_family(mol, skeletons):
        return FunctionalParentFrontier(
            "out_of_scope", reason="Ring candidates do not prove one bounded carbon monocycle."
        )
    suffix_modes = {external for _, external, _ in skeletons}
    if len(suffix_modes) != 1:
        return FunctionalParentFrontier(
            "out_of_scope", reason="Candidate set mixes internal and external suffix constructions."
        )
    if not cyclic and len({frozenset(path) for path, _external, _cyclic in skeletons}) > 1:
        # Earlier P-44 constitutional criteria precede annotation admission.
        # A strictly shorter chain can leave a label in an unsupported branch
        # even when the senior chain owns and renders that label correctly.
        # Retain unknown comparisons; never use writer availability to rank.
        from .parent_selection import graph_parent_facts, parent_seniority_frontier

        graph_facts = {}
        for path, external, _cyclic in skeletons:
            mapping = {idx: str(locant) for locant, idx in enumerate(path, 1)}
            sites = tuple(group.owner if external else group.center for group in groups)
            if not set(sites) <= mapping.keys():
                break
            facts = graph_parent_facts(
                mol, path, mapping, principal_class=groups[0].seniority,
                suffix_centers=[group.center for group in groups],
                suffix_locants=[mapping[idx] for idx in sites], kind="chain",
                allow_annotations=True,
                charge_role_proofs=charge_role_proofs,
            )
            if facts is None:
                break
            # Isotope/stereo criteria and equivalent numbering belong to the
            # later original-graph comparison, after complete mapped writers.
            graph_facts[(tuple(path), bool(external), False)] = replace(
                facts, isotope_key=None, stereo_key=None, stereo_identity=None,
            )
        else:
            skeletons = parent_seniority_frontier(
                skeletons, lambda row: graph_facts[(tuple(row[0]), bool(row[1]), False)],
            )
    annotated = any(atom.GetIsotope() or int(atom.GetChiralTag())
                    for atom in iter_atoms(mol)) or any(
        int(bond.GetStereo()) or int(bond.GetBondDir()) for bond in iter_bonds(mol)
    )
    implicit_stereo_trigger = None
    if cyclic and not annotated:
        from .parent_annotation_proof import (
            PotentialStereoTriggerStatus, implicit_unknown_monocycle_trigger,
        )
        implicit_stereo_trigger = implicit_unknown_monocycle_trigger(
            mol, Chem, principal_groups=groups, principal_class=groups[0].kind,
        )
        if implicit_stereo_trigger.status is PotentialStereoTriggerStatus.CANDIDATE:
            # The graph contains unspecified units that require the same
            # mapped proof and Xi serialization used by explicit annotations.
            annotated = True
        elif implicit_stereo_trigger.status is PotentialStereoTriggerStatus.UNRESOLVED:
            return FunctionalParentFrontier(
                "blocked", reason=implicit_stereo_trigger.reason,
                source_parent_rules=source_parent_rules,
            )
    if (any(atom.GetNumRadicalElectrons() for atom in iter_atoms(mol))
            or bool(mol.GetStereoGroups())):
        return FunctionalParentFrontier("out_of_scope", reason="Radical or enhanced-stereo graph is outside the bounded frontier scope.")
    from .neutral_charge_components import validate_neutral_charge_role_proofs
    charges_present = any(atom.GetFormalCharge() for atom in iter_atoms(mol))
    if charges_present and not validate_neutral_charge_role_proofs(
            mol, charge_role_proofs):
        return FunctionalParentFrontier(
            "out_of_scope",
            reason="Formal charges lack complete source-typed neutral-role evidence.",
        )
    annotation_proofs_by_skeleton = {}
    annotation_supported = True
    if annotated:
        if cyclic:
            annotation_supported, annotation_proofs_by_skeleton, annotation_status, annotation_reason = (
                _supports_annotated_monocycle_frontier(
                    mol, Chem, groups, skeletons, skipped,
                )
            )
            if not annotation_supported and annotation_status == "blocked":
                return FunctionalParentFrontier(
                    "blocked", reason=annotation_reason,
                    source_parent_rules=source_parent_rules,
                )
        else:
            annotation_supported = _supports_annotated_chain_frontier(
                mol, Chem, groups, skeletons,
            )
    if annotated and not annotation_supported:
        return FunctionalParentFrontier(
            "out_of_scope",
            reason="Isotope or stereochemistry is outside the independently mapped annotation scope.",
        )
    from .parent_selection import graph_parent_facts, parent_frontier

    facts_by_skeleton = {}
    functionalized_proofs_by_skeleton = {}
    for path, external, _ in skeletons:
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in path):
            return FunctionalParentFrontier("out_of_scope", reason="Candidate parent is not all-carbon.")
        mapping = {atom: str(index) for index, atom in enumerate(path, 1)}
        prefix_facts = None
        if cyclic or annotated or charge_prefix_proofs:
            # Ring comparisons and annotation ranking require fully proved
            # detachable-component facts before any candidate is rendered.
            # Preserve the established unannotated chain path: unknown prefix
            # facts remain a conservative comparison barrier until rendering
            # has produced complete mapped prefix components for every max.
            prefix_atoms = set(skipped) | {group.center for group in groups}
            # A direct characteristic group can itself be rooted at carbon
            # (notably –CN). Resolve its exact source-inventory identity,
            # parent host, atom set, and mapped writer ownership before asking
            # the detached-carbon adapter to interpret carbon-rooted cuts.
            # This is deliberately writer-backed ownership, not an element- or
            # inventory-center-based exemption: e.g. the aromatic carbon
            # bearing hydroxy remains in the selected parent path.
            nonprincipal_prefix_facts = _mapped_nonprincipal_group_prefix_facts(
                mol, Chem, groups, path,
            ) if cyclic or annotated or source_parent_rules else ()
            if nonprincipal_prefix_facts is None:
                return FunctionalParentFrontier(
                    "blocked" if cyclic or source_parent_rules else "out_of_scope",
                    reason="Subordinate characteristic-prefix facts are not completely mapped.",
                    source_parent_rules=source_parent_rules,
                )
            reserved_direct_atoms = {
                atom for _host, _text, atoms in nonprincipal_prefix_facts
                for atom in atoms
            }
            functionalized_proofs = _functionalized_branch_proofs(
                mol, Chem, path, groups,
                reserved_direct_prefix_atoms=reserved_direct_atoms,
            )
            if functionalized_proofs is None:
                return FunctionalParentFrontier(
                    # ``None`` is reserved for an identified source-role
                    # obligation whose component could not be proved (or
                    # whose proof scopes conflict). That obligation remains
                    # decisive for ring-only families too; absence of a
                    # chain-to-ring handoff does not make a recognized
                    # functionalized branch safe for the legacy renderer.
                    "blocked",
                    reason="A recognized functionalized carbon branch lacks a complete typed proof.",
                    source_parent_rules=source_parent_rules,
                )
            typed_prefix_facts = tuple(
                (str(proof.attachment_atom), proof.kind, tuple(proof.atoms))
                for proof in charge_prefix_proofs
            )
            oxygen_prefix_facts = _mapped_secondary_oxygen_prefix_facts(
                mol, Chem, groups, path,
            )
            if oxygen_prefix_facts is None:
                return FunctionalParentFrontier(
                    "blocked" if source_parent_rules else "out_of_scope",
                    reason="Subordinate oxygen prefix facts are not completely mapped.",
                    source_parent_rules=source_parent_rules,
                )
            typed_prefix_facts += oxygen_prefix_facts
            typed_prefix_facts += nonprincipal_prefix_facts
            typed_prefix_facts += tuple(
                (str(proof.parent_host), proof.text, tuple(sorted(proof.scope)))
                for proof in functionalized_proofs
            )
            prefix_facts = _mapped_linear_prefix_facts(
                mol, Chem, path, set(path), prefix_atoms,
                typed_prefix_facts=typed_prefix_facts,
                allow_noncarbon=bool(charge_prefix_proofs),
            )
            if prefix_facts is None:
                return FunctionalParentFrontier(
                    "blocked" if source_parent_rules else "out_of_scope",
                    reason="Detachable prefix facts are not completely mapped.",
                    source_parent_rules=source_parent_rules,
                )
        if annotated:
            # Explicit isotope hydrogens belong to the numbered parent atom.
            # Their ownership is derived from the original graph before any
            # candidate renderer is called.
            mapping = hydrogen_bindings(mol, mapping)
        suffix_locs = []
        for group in groups:
            site = group.owner if external else group.center
            if site not in mapping:
                return FunctionalParentFrontier(
                    "blocked" if source_parent_rules else "out_of_scope",
                    reason="Candidate does not map every selected suffix site.",
                    source_parent_rules=source_parent_rules,
                )
            suffix_locs.append(mapping[site])
        facts = graph_parent_facts(
            mol, path, mapping, principal_class=groups[0].seniority,
            suffix_centers=[group.center for group in groups],
            suffix_locants=suffix_locs,
            prefixes=prefix_facts,
            kind="ring" if cyclic else "chain",
            allow_annotations=True,
            charge_role_proofs=charge_role_proofs,
        )
        if facts is None:
            return FunctionalParentFrontier(
                "blocked" if source_parent_rules else "out_of_scope",
                reason="Direct graph facts or complete atom ownership are unproved.",
                source_parent_rules=source_parent_rules,
            )
        ring_unknown = annotation_proofs_by_skeleton.get((tuple(path), bool(external)))
        missing_annotation_fields = ("isotope_key", "equivalent_isotope_key")
        if not (cyclic and ring_unknown is not None
                and ring_unknown.unknown_stereo_locants):
            missing_annotation_fields += ("stereo_numbering_key",)
        if annotated and any(getattr(facts, field) is None for field in missing_annotation_fields):
            return FunctionalParentFrontier(
                "out_of_scope", reason="Mapped isotope or stereochemical comparison facts are incomplete.",
            )
        facts_by_skeleton[(tuple(path), bool(external))] = facts
        functionalized_proofs_by_skeleton[(tuple(path), bool(external))] = (
            functionalized_proofs if cyclic or annotated or charge_prefix_proofs else ()
        )

    wrappers = list(facts_by_skeleton)
    if annotated:
        # Preserve the P-44.1/44.3 and unannotated P-44.4 frontier first.
        # Isotope/stereo evidence is considered only among those structural
        # maxima, and same-parent numbering is resolved by P-14.4/P-82.5.
        structural = {
            item: replace(facts_by_skeleton[item], isotope_key=None,
                          equivalent_isotope_key=None, stereo_key=None,
                          stereo_identity=None, stereo_numbering_key=None,
                          numbering_scope=None)
            for item in wrappers
        }
        structural_maxima = parent_frontier(wrappers, lambda item: structural[item])
        structural_paths = tuple(tuple(item[0]) for item in structural_maxima)
        maxima = _annotation_frontier(
            structural_maxima, facts_by_skeleton,
            annotation_proofs_by_item=annotation_proofs_by_skeleton,
        )
        # Annotation ranking is source ordered and conservative: a live
        # isotope-equivalence or stereodescriptor comparison blocks this
        # bounded writer rather than falling through to enumeration order.
        for i, left in enumerate(structural_maxima):
            left_proof = annotation_proofs_by_skeleton.get(left)
            if not left_proof:
                continue
            for right in structural_maxima[i + 1:]:
                right_proof = annotation_proofs_by_skeleton.get(right)
                if not right_proof or set(left[0]) != set(right[0]):
                    continue
                if _compare_annotated_numberings_with_unknown(
                    facts_by_skeleton[left], facts_by_skeleton[right],
                    left_proof.unknown_stereo_locants,
                    right_proof.unknown_stereo_locants,
                    bool(left_proof.stereo_features or right_proof.stereo_features),
                    bool(left_proof.isotope_ownership or right_proof.isotope_ownership),
                ) is None:
                    return FunctionalParentFrontier(
                        "blocked", maximal_skeletons=tuple(tuple(row[0]) for row in structural_maxima),
                        structural_maxima=tuple(tuple(row[0]) for row in structural_maxima),
                        reason="Equivalent ring numberings leave a source-ordered isotope or stereodescriptor comparison unresolved.",
                        source_parent_rules=source_parent_rules,
                        annotation_proofs=tuple(annotation_proofs_by_skeleton.values()),
                    )
    else:
        maxima = parent_frontier(wrappers, lambda item: facts_by_skeleton[item])
        structural_paths = tuple(tuple(item[0]) for item in maxima)
    maximal_paths = tuple(tuple(item[0]) for item in maxima)
    rendered = []
    failures = []
    for path, external in maxima:
        render_fn = renderer or _render
        render_kwargs = dict(
            mapped_ring_prefixes=mapped_ring_prefixes,
            return_component=True,
        )
        annotation_proof = annotation_proofs_by_skeleton.get((tuple(path), bool(external)))
        if annotation_proof is not None:
            render_kwargs["parent_annotation_proof"] = annotation_proof
        if charge_role_proofs:
            render_kwargs["charge_role_proofs"] = charge_role_proofs
        render_proofs = functionalized_proofs_by_skeleton.get(
            (tuple(path), bool(external)), (),
        )
        if render_proofs:
            render_kwargs["functionalized_carbon_prefix_proofs"] = render_proofs
        record = render_fn(
            mol, Chem, groups, kind, path, external, cyclic, skipped, owner_components,
            **render_kwargs,
        )
        if not record:
            failures.append(tuple(path))
            continue
        record, construction = record
        name, locants, prefix_locs, mapping, prefixes, component_name, owners = record
        facts = graph_parent_facts(
            mol, path, mapping, principal_class=groups[0].seniority,
            suffix_centers=[group.center for group in groups], suffix_locants=locants,
            prefixes=prefixes, kind="ring" if cyclic else "chain", name=name,
            allow_annotations=True,
            charge_role_proofs=charge_role_proofs,
            substituent_nonstandard_proof=build_substituent_nonstandard_proof(
                mol, path, construction, prefixes, mapping, groups,
            ),
        )
        if facts is None:
            failures.append(tuple(path))
            continue
        edges = list(zip(path, path[1:]))
        if cyclic:
            edges.append((path[-1], path[0]))
        multiple = tuple(j for j, (a, b) in enumerate(edges, 1)
                         if mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1)
        key = (-len(groups), -len(path), locants, multiple, prefix_locs, name)
        rendered.append(FunctionalParent(
            name, frozenset(group.center for group in groups), tuple(mapping.items()),
            tuple(path), key, prefix_locs, prefixes, component_name, owners, facts,
            construction, frozenset(), "external" if external else "embedded",
            (), construction.parent_annotation_proof,
        ))
    if failures:
        return FunctionalParentFrontier(
            "blocked", tuple(rendered), maximal_paths,
            f"Renderer failed for {len(failures)} source-maximal candidate(s).",
            structural_paths, source_parent_rules=source_parent_rules,
            annotation_proofs=tuple(annotation_proofs_by_skeleton.values()),
        )
    if not rendered:
        return FunctionalParentFrontier("blocked", (), maximal_paths,
                                        "No structural maximum rendered.", structural_paths,
                                        source_parent_rules=source_parent_rules,
                                        annotation_proofs=tuple(annotation_proofs_by_skeleton.values()))
    from .method_parent_frontier import MethodParentStatus, select_method_parent

    completeness = CandidateCompleteness(
        'mapped-complete-functional-parent',
        source_parent_rules or ('P-41/P-43', 'P-44.1.1', 'P-44.3', 'P-44.4', 'P-82.5'),
        maximal_paths, tuple(tuple(record.parent_atoms) for record in rendered),
        'All source-maximal parent numberings rendered; earlier graph and annotation eliminations retain their proofs.',
        proposal_keys=tuple(proposal_identity(record) for record in rendered),
    )
    selection = select_method_parent(
        rendered, lambda record: record.parent_selection,
        physical_parent_identity=lambda record: (
            frozenset(record.parent_atoms), record.principal_centers),
        completeness=completeness,
    )
    preferred = selection.candidates
    if selection.status == MethodParentStatus.UNRESOLVED:
        return FunctionalParentFrontier(
            "conflict", tuple(preferred),
            maximal_paths, selection.reason,
            structural_paths, source_parent_rules=source_parent_rules,
            annotation_proofs=tuple(annotation_proofs_by_skeleton.values()),
            candidate_completeness=completeness, compared_candidates=tuple(rendered),
        )
    status = "complete" if len(preferred) == 1 else "tie"
    return FunctionalParentFrontier(status, tuple(preferred), maximal_paths,
                                    "All structural maxima rendered and were compared.",
                                    structural_paths, source_parent_rules=source_parent_rules,
                                    annotation_proofs=tuple(annotation_proofs_by_skeleton.values()),
                                    candidate_completeness=completeness,
                                    compared_candidates=tuple(rendered))


def _ring_adapter_rule(kind):
    return {
        "acid": "P-65.1.2",
        "amide": "P-66.1",
        "nitrile": "P-66.5",
    }.get(kind, "")


def _source_senior_ring_parent(mol, Chem, inventory, groups):
    """Graph/source proof only; a successful rendered name is never evidence.

    The complete principal set must be ordinary acid/amide/nitrile units on
    carbon hosts of one simple carbon or N/O/S ring. It reaches the absolute
    suffix-count maximum; pendant rings carry none of this selected set.
    P-44/P-52 retains the senior ring over eligible carbon-chain parents.
    Direct ring assemblies, possible linear phanes, fused/spiro parents,
    mixed classes and unproved source roles retain separate obligations.
    """
    if (not groups or not inventory.supported or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol))):
        return None
    kind = groups[0].kind
    if kind not in {"acid", "amide", "nitrile"} or any(g.kind != kind for g in groups):
        return None
    rings = mol.GetRingInfo().AtomRings()
    from .mapped_ring_parents import ring_systems
    systems = ring_systems(mol)
    # P-52.2.5.1(2): at least four cyclic systems are necessary for a linear
    # phane. An unsuccessful phane renderer is never an exclusion proof.
    if not systems or len(systems) >= 4:
        return None
    hosts = frozenset(group.owner for group in groups)
    candidates = tuple(system for system in systems if hosts <= system)
    if len(candidates) != 1:
        return None
    core = candidates[0]
    if sum(bool(set(ring) & core) for ring in rings) != 1:
        return None
    numbers = {mol.GetAtomWithIdx(i).GetAtomicNum() for i in core}
    if not numbers <= {6, 7, 8, 16}:
        return None
    # The existing all-carbon single-ring frontier already proves its own
    # complete candidate pool. Preserve that independent operation.
    if len(systems) == 1 and numbers == {6}:
        return None
    if any(g.owner not in core or g.center in core for g in groups):
        return None
    system_by_atom = {atom: number for number, system in enumerate(systems) for atom in system}
    if any(bond.GetBeginAtomIdx() in system_by_atom
           and bond.GetEndAtomIdx() in system_by_atom
           and system_by_atom[bond.GetBeginAtomIdx()] != system_by_atom[bond.GetEndAtomIdx()]
           for bond in iter_bonds(mol)):
        # P-28/P-52.2.7 owns directly connected assemblies. Do not discard a
        # larger composite parent because only its component writer exists.
        return None
    source = inventory.source_frontier
    from .amide_isotopes import source_principal_amides
    source_groups_proved = (source_principal_amides(mol, inventory, groups) if kind == 'amide'
                            else set(groups) == set(source.groups) == set(inventory.principal))
    if (source.status not in {"selected", "tie"} or not source_groups_proved
            or len(inventory.source_group_facts) != len(inventory.groups)):
        return None
    facts = dict(zip(inventory.groups, inventory.source_group_facts))
    if not all(_ring_site_source_class_proved(mol, Chem, g, facts.get(g), kind)
               for g in groups):
        return None
    return SeniorRingParentProof(
        core, frozenset(group.center for group in groups), kind,
        tuple(sorted(systems, key=lambda system: tuple(sorted(system)))),
        tuple(frozenset(group.atoms) for group in groups),
    )


def _complete_source_class_ring_candidates(mol, Chem, groups, kind, skeletons):
    """Apply ring-over-chain only after source class and full coverage are proved.

    This bounded handoff is available only for ordinary carboxylic acids,
    carbonyl amides, and nitriles. Their class-specific adapters establish
    original center/host/ligand roles before this shared P-44/P-52 operation.
    The ring must carry the complete selected source-principal set, so
    inventory cardinality proves the P-44.1.1 maximum before the ring rule.
    """
    if kind not in {"acid", "amide", "nitrile"} or not groups:
        return None
    if (any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
            for atom in iter_atoms(mol))
            or any(int(bond.GetStereo()) or int(bond.GetBondDir())
                   for bond in iter_bonds(mol))
            or mol.GetStereoGroups()):
        return None

    inventory = selection_inventory(mol)
    if not inventory.supported:
        return None

    source_frontier = inventory.source_frontier
    if source_frontier.status not in {"selected", "tie"}:
        return None

    def identity(group):
        return (group.kind, group.center, tuple(group.atoms), group.owner, group.ligand)

    principal_ids = {identity(group) for group in source_frontier.groups}
    from .amide_isotopes import source_principal_amides
    projected_amides = kind == 'amide' and source_principal_amides(mol, inventory, groups)
    if (not principal_ids
            or not projected_amides and {identity(group) for group in groups} != principal_ids
            or {identity(group) for group in inventory.principal} != principal_ids
            or any(group.kind != kind for group in source_frontier.groups)):
        return None
    source_by_id = {
        identity(group): fact
        for group, fact in zip(inventory.groups, inventory.source_group_facts)
    }
    if len(inventory.source_group_facts) != len(inventory.groups):
        return None
    for group in groups:
        fact = source_by_id.get(identity(group))
        if not _ring_site_source_class_proved(mol, Chem, group, fact, kind):
            return None

    ring_rows = tuple(row for row in skeletons if bool(row[2]))
    chain_rows = tuple(row for row in skeletons if not bool(row[2]))
    if not ring_rows or not chain_rows:
        return None
    if not _is_bounded_monocycle_family(mol, ring_rows):
        return None
    ring_atoms = frozenset(ring_rows[0][0])
    if any(frozenset(row[0]) != ring_atoms for row in ring_rows):
        return None
    physical_rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if len(physical_rings) != 1 or physical_rings[0] != ring_atoms:
        return None
    # All admitted sites are exocyclic; the original atom-role proof below
    # ensures that the proper carbon host, not an amide N substituent or a
    # coincidental neighboring atom, is on the ring.
    if any(group.center in ring_atoms or group.owner not in ring_atoms
           for group in groups):
        return None
    if any(not external for _path, external, _cyclic in ring_rows):
        return None
    if any(not all(group.owner in frozenset(path) for group in groups)
           for path, _external, _cyclic in ring_rows):
        return None
    # Every surviving ring candidate contains all source-selected groups.
    # No candidate can cover more than the complete principal inventory, so
    # these ring parents tie the P-44.1.1 maximum and satisfy the equal-count,
    # same-heteroatom-class condition for P-52.2.8. Keep every orientation.
    return ring_rows


def _ring_site_source_class_proved(mol, Chem, group, fact, kind):
    """Validate one local site adapter without inferring class from spelling."""
    rank = getattr(fact, "class_rank", None)
    class_name = getattr(fact, "class_name", None)
    if kind == "acid":
        acid = getattr(fact, "corresponding_acid", None)
        return bool(
            rank == 7 and class_name == "acid"
            and getattr(acid, "category", None) == "7a"
            and getattr(acid, "p42_1_family", None) == "carboxylic"
            and getattr(acid, "p42_1_form", None) == "ordinary"
            and _ordinary_external_acid_core(mol, group)
        )
    if kind == "amide":
        acid = getattr(fact, "corresponding_acid", None)
        return bool(
            rank == 11 and class_name == "amide"
            and getattr(acid, "category", None) == "7a"
            and getattr(acid, "p42_1_family", None) == "carboxylic"
            and getattr(acid, "p42_1_form", None) == "ordinary"
            and _ordinary_external_amide_core(mol, group)
        )
    if kind == "nitrile":
        return bool(rank == 14 and class_name == "nitrile"
                    and _ordinary_external_nitrile_core(mol, Chem, group))
    return False


def _ordinary_external_acid_core(mol, group):
    center = mol.GetAtomWithIdx(group.center)
    if center.GetAtomicNum() != 6 or center.GetFormalCharge():
        return False
    double_o = [n for n in center.GetNeighbors()
                if n.GetAtomicNum() == 8
                and mol.GetBondBetweenAtoms(group.center, n.GetIdx()).GetBondTypeAsDouble() == 2]
    hydroxyl_o = [n for n in center.GetNeighbors()
                  if n.GetAtomicNum() == 8
                  and mol.GetBondBetweenAtoms(group.center, n.GetIdx()).GetBondTypeAsDouble() == 1
                  and n.GetTotalNumHs(includeNeighbors=True) == 1]
    carbon_hosts = [n for n in center.GetNeighbors() if n.GetAtomicNum() == 6]
    expected_atoms = {group.center, *(n.GetIdx() for n in double_o + hydroxyl_o)}
    return bool(
        len(double_o) == len(hydroxyl_o) == len(carbon_hosts) == 1
        and len(center.GetNeighbors()) == 3
        and group.owner == carbon_hosts[0].GetIdx()
        and frozenset(group.atoms) == expected_atoms
    )


def _ordinary_external_amide_core(mol, group):
    center = mol.GetAtomWithIdx(group.center)
    if center.GetAtomicNum() != 6 or center.GetFormalCharge():
        return False
    double_o = [n for n in center.GetNeighbors()
                if n.GetAtomicNum() == 8
                and mol.GetBondBetweenAtoms(group.center, n.GetIdx()).GetBondTypeAsDouble() == 2]
    single_n = [n for n in center.GetNeighbors()
                if n.GetAtomicNum() == 7
                and mol.GetBondBetweenAtoms(group.center, n.GetIdx()).GetBondTypeAsDouble() == 1]
    carbon_hosts = [n for n in center.GetNeighbors() if n.GetAtomicNum() == 6]
    expected_atoms = {group.center, *(n.GetIdx() for n in double_o + single_n)}
    if not (
        len(double_o) == len(single_n) == len(carbon_hosts) == 1
        and len(center.GetNeighbors()) == 3
        and center.GetTotalValence() == 4
        and mol.GetBondBetweenAtoms(group.center, carbon_hosts[0].GetIdx()).GetBondTypeAsDouble() == 1
        and group.owner == carbon_hosts[0].GetIdx()
        and group.ligand == single_n[0].GetIdx()
        and frozenset(group.atoms) == expected_atoms
    ):
        return False
    nitrogen = single_n[0]
    return bool(
        nitrogen.GetAtomicNum() == 7 and not nitrogen.GetFormalCharge()
        and not nitrogen.IsInRing() and not nitrogen.GetIsAromatic()
        and nitrogen.GetTotalValence() == 3
        and mol.GetBondBetweenAtoms(group.center, nitrogen.GetIdx()).GetBondTypeAsDouble() == 1
    )


def _ordinary_external_nitrile_core(mol, Chem, group):
    center = mol.GetAtomWithIdx(group.center)
    if center.GetAtomicNum() != 6 or center.GetFormalCharge():
        return False
    terminal_triple_n = [n for n in center.GetNeighbors()
                         if n.GetAtomicNum() == 7
                         and mol.GetBondBetweenAtoms(group.center, n.GetIdx()).GetBondType() == Chem.BondType.TRIPLE]
    carbon_hosts = [n for n in center.GetNeighbors() if n.GetAtomicNum() == 6]
    expected_atoms = {group.center, *(n.GetIdx() for n in terminal_triple_n)}
    return bool(
        len(terminal_triple_n) == len(carbon_hosts) == 1
        and terminal_triple_n[0].GetDegree() == 1
        and not terminal_triple_n[0].GetFormalCharge()
        and center.GetTotalValence() == 4
        and terminal_triple_n[0].GetTotalValence() == 3
        and mol.GetBondBetweenAtoms(group.center, carbon_hosts[0].GetIdx()).GetBondTypeAsDouble() == 1
        and len(center.GetNeighbors()) == 2
        and group.owner == carbon_hosts[0].GetIdx()
        and frozenset(group.atoms) == expected_atoms
    )


def _is_bounded_monocycle_family(mol, skeletons):
    """Prove a single carbon cycloalkane or benzene parent family."""
    paths = [tuple(path) for path, _external, cyclic in skeletons if cyclic]
    if len(paths) != len(skeletons) or not paths:
        return False
    parent_sets = {frozenset(path) for path in paths}
    if len(parent_sets) != 1:
        return False
    atoms = next(iter(parent_sets))
    if len(atoms) < 3 or any(
        mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in atoms
    ):
        return False
    rings = [ring for ring in mol.GetRingInfo().AtomRings()
             if frozenset(ring) == atoms]
    if len(rings) != 1 or any(
        sum(idx in ring for ring in mol.GetRingInfo().AtomRings()) != 1
        for idx in atoms
    ):
        return False
    aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in atoms)
    if aromatic:
        return len(atoms) == 6 and all(
            mol.GetBondBetweenAtoms(a, b).GetIsAromatic()
            for path in paths for a, b in _closed_edges(path)
        )
    return all(
        not mol.GetBondBetweenAtoms(a, b).GetIsAromatic()
        and mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() == 1
        for path in paths for a, b in _closed_edges(path)
    )


def _closed_edges(path):
    return tuple(zip(path, path[1:] + path[:1]))


def _supports_annotated_monocycle_frontier(mol, Chem, groups, skeletons, skipped):
    """Build typed, atom-owned annotation evidence for each mapped ring path."""
    if not _is_bounded_monocycle_family(mol, skeletons):
        return False, {}, "out_of_scope", "Parent is outside the proved monocycle family."
    from .parent_annotation_proof import (
        ParentAnnotationStatus, prove_monocycle_parent_annotations,
    )

    proofs = {}
    suffix_atoms = set(skipped) | {group.center for group in groups}
    for path, _external, _cyclic in skeletons:
        if _mapped_linear_prefix_facts(
            mol, Chem, path, set(path), suffix_atoms,
        ) is None:
            return False, {}, "out_of_scope", "Subordinate ring branches lack complete mapped ownership."
        mapping = {atom: str(index) for index, atom in enumerate(path, 1)}
        proof = prove_monocycle_parent_annotations(
            mol, Chem, parent_atoms=path, parent_locants=mapping,
            principal_groups=groups, principal_class=groups[0].kind,
        )
        if proof.status is not ParentAnnotationStatus.PROVED:
            return (False, {}, proof.status.value, proof.reason)
        proofs[(tuple(path), bool(_external))] = proof
    return True, proofs, "proved", "All candidate ring paths have complete annotation ownership."


def _supports_annotated_chain_frontier(mol, Chem, groups, skeletons):
    """Gate the isotope/stereo extension to mapped chain and prefix facts.

    The bounded family is an all-carbon acyclic parent chain with mapped
    unlabelled side branches, including source-typed functionalized trees.
    Parent-chain single,
    double, and triple bonds are allowed. Labels may occur on parent carbons
    or explicit H atoms directly attached to them; ordinary assigned R/S
    centers and assigned E/Z parent double bonds are supported. Branch labels,
    heteroatom labels, unspecified stereogenic units, and off-parent features
    remain outside the operation.
    """
    suffix_atoms = set()
    for group in groups:
        suffix_atoms.update(group.atoms)
        suffix_atoms.add(group.center)
    parent_sets = [set(path) for path, _external, cyclic in skeletons if not cyclic]
    if not parent_sets or any(cyclic for _path, _external, cyclic in skeletons):
        return False
    parent_union = set().union(*parent_sets)
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
           for parent in parent_sets for i in parent):
        return False
    parent_bonds = set()
    for path, _external, _cyclic in skeletons:
        for left, right in zip(path, path[1:]):
            bond = mol.GetBondBetweenAtoms(left, right)
            if (bond is None or bond.GetIsAromatic()
                    or bond.GetBondType() not in {
                        Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                        Chem.BondType.TRIPLE,
                    }):
                return False
            parent_bonds.add(bond.GetIdx())
    # Use the same complete component facts consumed by the later writer.
    # The former linear-alkyl-only pre-gate rejected typed functionalized
    # branches before the actual parent/branch handoff could be attempted.
    for path, _external, _cyclic in skeletons:
        direct = _mapped_nonprincipal_group_prefix_facts(mol, Chem, groups, path)
        oxygen = _mapped_secondary_oxygen_prefix_facts(mol, Chem, groups, path)
        if direct is None or oxygen is None:
            return False
        proofs = _functionalized_branch_proofs(
            mol, Chem, path, groups,
            reserved_direct_prefix_atoms={idx for _host, _text, atoms in direct for idx in atoms},
        )
        if proofs is None:
            return False
        typed = direct + oxygen + tuple(
            (str(proof.parent_host), proof.text, tuple(sorted(proof.scope)))
            for proof in proofs
        )
        if _mapped_linear_prefix_facts(
                mol, Chem, path, set(path), suffix_atoms, typed_prefix_facts=typed) is None:
            return False
    for atom in iter_atoms(mol):
        idx = atom.GetIdx()
        if atom.GetIsotope():
            if atom.GetAtomicNum() == 6:
                if idx not in parent_union:
                    return False
            elif atom.GetAtomicNum() == 1:
                if not any(neighbor.GetIdx() in parent_union
                           for neighbor in atom.GetNeighbors()):
                    return False
            else:
                return False
        if int(atom.GetChiralTag()) and (idx not in parent_union
                                         or atom.GetAtomicNum() != 6):
            return False
    try:
        potential = Chem.FindPotentialStereo(mol)
        from .stereochemistry import assigned_atom_stereochemistry
        assigned = dict(assigned_atom_stereochemistry(
            mol, Chem, atom_order="idx", include_pseudo=True,
        ))
        from .stereochemistry import _indexed_bond_stereochemistry
        assigned_bonds = _indexed_bond_stereochemistry(mol, Chem)
    except Exception:
        return False
    assigned_bond_ids = {
        mol.GetBondBetweenAtoms(left, right).GetIdx()
        for left, right, descriptor in assigned_bonds
        if descriptor in {"E", "Z"}
        and mol.GetBondBetweenAtoms(left, right) is not None
    }
    # E/Z directions in SMILES are carried on the adjacent single bonds. They
    # are accepted only when they support a mapped parent C=C with an explicit
    # E/Z assignment; unrelated wedge/dash or directional information remains
    # outside this operation.
    for bond in iter_bonds(mol):
        direction = int(bond.GetBondDir())
        stereo = int(bond.GetStereo())
        if stereo and (bond.GetIdx() not in parent_bonds
                       or bond.GetBondType() != Chem.BondType.DOUBLE
                       or stereo not in {int(Chem.BondStereo.STEREOE),
                                         int(Chem.BondStereo.STEREOZ)}):
            return False
        if direction:
            if bond.GetIdx() in parent_bonds and bond.GetBondType() == Chem.BondType.DOUBLE:
                if bond.GetIdx() not in assigned_bond_ids:
                    return False
                continue
            ends = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
            adjacent_to_assigned_double = any(
                idx in parent_bonds
                and mol.GetBondWithIdx(idx).GetBondType() == Chem.BondType.DOUBLE
                and bool(ends & {mol.GetBondWithIdx(idx).GetBeginAtomIdx(),
                                 mol.GetBondWithIdx(idx).GetEndAtomIdx()})
                and idx in assigned_bond_ids
                for idx in parent_bonds
            )
            if not adjacent_to_assigned_double:
                return False
    potential_atoms = set()
    potential_bonds = set()
    for feature in potential:
        if feature.type == Chem.StereoType.Atom_Tetrahedral:
            idx = feature.centeredOn
            if idx not in parent_union or assigned.get(idx) not in {"R", "S"}:
                return False
            potential_atoms.add(idx)
        elif feature.type == Chem.StereoType.Bond_Double:
            bond_idx = int(feature.centeredOn)
            bond = mol.GetBondWithIdx(bond_idx)
            ends = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
            if (bond_idx not in parent_bonds or not ends <= parent_union
                    or bond_idx not in assigned_bond_ids):
                return False
            potential_bonds.add(bond_idx)
        else:
            return False
    if potential_bonds != assigned_bond_ids:
        return False
    if any(int(atom.GetIsotope()) for atom in iter_atoms(mol)):
        # P-82.5.1 treats isotope priority as last except when isotopic
        # substitution creates the stereogenicity itself. Compare the original
        # graph's mapped potential units with an isotope-cleared copy so such
        # isotope-created E/Z/R/S units cannot enter the ordinary ranking.
        original_units = _potential_stereo_units(mol, Chem)
        isotope_cleared_units = _potential_stereo_units(_isotope_cleared_copy(mol, Chem), Chem)
        if original_units is None or original_units != isotope_cleared_units:
            return False
    return all(not int(atom.GetChiralTag()) or atom.GetIdx() in potential_atoms
               for atom in iter_atoms(mol))


def _isotope_cleared_copy(mol, Chem):
    """Copy the graph with nuclide masses removed for the P-82.5.1 gate."""
    probe = Chem.Mol(mol)
    for atom in iter_atoms(probe):
        atom.SetIsotope(0)
    Chem.SanitizeMol(probe)
    return probe


def _potential_stereo_units(mol, Chem):
    """Return graph-identity-independent unit positions, or None if unknown."""
    try:
        return frozenset((str(feature.type), int(feature.centeredOn))
                         for feature in Chem.FindPotentialStereo(mol))
    except Exception:
        return None


def _mapped_linear_prefix_facts(mol, Chem, path, parent_atoms, suffix_atoms,
                                *, typed_prefix_facts=(), allow_noncarbon=False):
    """Return graph-proved singly attached prefixes, or None on uncertainty."""
    parent_atoms = set(parent_atoms)
    remaining = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
    remaining -= parent_atoms | set(suffix_atoms)
    mapping = {idx: str(number) for number, idx in enumerate(path, 1)}
    prefixes = []
    typed_atoms = set()
    for host, text, atoms in typed_prefix_facts:
        host = int(host)
        atoms = set(atoms)
        if (host not in mapping or not atoms or atoms & typed_atoms
                or atoms & parent_atoms or not atoms <= remaining):
            return None
        typed_atoms.update(atoms)
        prefixes.append((mapping[host], text))
    remaining -= typed_atoms
    while remaining:
        start = next(iter(remaining))
        component, stack, attachments = set(), [start], []
        while stack:
            idx = stack.pop()
            if idx not in remaining:
                continue
            remaining.remove(idx)
            component.add(idx)
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                other = neighbor.GetIdx()
                if other in parent_atoms:
                    attachments.append(other)
                elif other in remaining:
                    stack.append(other)
        if (not attachments or len(set(attachments)) != 1
                or (not allow_noncarbon and any(
                    mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in component
                ))
                or any(mol.GetAtomWithIdx(i).GetIsotope()
                       or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in component)):
            return None
        attachment = attachments[0]
        internal_degrees = {
            i: sum(n.GetIdx() in component for n in mol.GetAtomWithIdx(i).GetNeighbors())
            for i in component
        }
        expected_ends = 0 if len(component) == 1 else 2
        if (any(degree > 2 for degree in internal_degrees.values())
                or sum(degree == 1 for degree in internal_degrees.values()) != expected_ends
                or len(component) > 1 and any(degree == 0 for degree in internal_degrees.values())
                or any(mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() != 1
                       for i in component for j in component if i < j
                       and mol.GetBondBetweenAtoms(i, j) is not None)):
            return None
        roots = [i for i in component
                 if mol.GetBondBetweenAtoms(i, attachment) is not None]
        if (len(roots) != 1
                or mol.GetBondBetweenAtoms(roots[0], attachment).GetBondTypeAsDouble() != 1):
            return None
        text, owned = mapped_branch(mol, Chem, roots[0], attachment)
        if not text or set(owned) != component:
            return None
        prefixes.append((mapping[attachment], text))
    return tuple(prefixes)


def _mapped_secondary_oxygen_prefix_facts(mol, Chem, principal_groups, path):
    """Map exact subordinate ketone/alcohol O atoms as oxo/hydroxy prefixes."""
    from .functional_inventory import functional_inventory

    principal_ids = {group.center for group in principal_groups}
    facts = []
    for group in functional_inventory(mol).groups:
        if group.center in principal_ids or group.kind not in {"ketone", "alcohol"}:
            continue
        if group.center not in path:
            continue
        roots = tuple(idx for idx in group.atoms if idx != group.center)
        if len(roots) != 1:
            return None
        text, atoms = mapped_branch(mol, Chem, roots[0], group.center)
        expected = "oxo" if group.kind == "ketone" else "hydroxy"
        if text != expected or set(atoms) != set(roots):
            return None
        facts.append((str(group.center), text, tuple(sorted(atoms))))
    return tuple(facts)


def _mapped_nonprincipal_group_prefix_facts(mol, Chem, principal_groups, path):
    """Map exact, source-inventoried subordinate groups as detachable prefixes.

    The generic residual-component walk below intentionally accepts only
    simple carbon trees. Characteristic groups such as sulfo/sulfino have
    branched heteroatom topology, so they must enter through an existing
    mapped branch construction whose ownership exactly equals the inventoried
    group. This does not admit arbitrary text-shaped components or partial
    group ownership.
    """
    # P-61.3.1 halogen prefixes are not characteristic-group inventory rows.
    # Admit only an ordinary, neutral, unlabelled monovalent halogen attached
    # directly to this parent by a single bond, and require the mapped branch
    # writer to claim exactly that original atom. This feeds a typed prefix
    # fact into the same pre-render comparison as other mapped substituents.
    path_atoms = set(path)
    facts = []
    claimed = set()
    halogen_tokens = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}
    for atom in iter_atoms(mol):
        number = atom.GetAtomicNum()
        if number not in halogen_tokens:
            continue
        if atom.GetIdx() in path_atoms:
            return None
        neighbors = tuple(atom.GetNeighbors())
        # A halogen on an N substituent or another detached component is not
        # this parent prefix. Its owning component must prove it separately.
        if not any(neighbor.GetIdx() in path_atoms for neighbor in neighbors):
            continue
        if (len(neighbors) != 1
                or atom.GetFormalCharge() != 0 or atom.GetIsotope() != 0
                or atom.GetNumRadicalElectrons() != 0 or atom.GetChiralTag()
                or atom.GetTotalValence() != 1
                or mol.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx()).GetBondTypeAsDouble() != 1):
            return None
        host = neighbors[0].GetIdx()
        text, owned = mapped_branch(mol, Chem, atom.GetIdx(), host)
        if text != halogen_tokens[number] or set(owned) != {atom.GetIdx()}:
            return None
        if atom.GetIdx() in claimed:
            return None
        facts.append((str(host), text, (atom.GetIdx(),)))
        claimed.add(atom.GetIdx())

    inventory = selection_inventory(mol)
    if not inventory.supported or not inventory.groups:
        return tuple(facts)
    principal_ids = {
        (group.kind, group.center, tuple(group.atoms), group.owner, group.ligand)
        for group in principal_groups
    }
    for group in inventory.groups:
        identity = (group.kind, group.center, tuple(group.atoms), group.owner, group.ligand)
        if identity in principal_ids:
            continue
        host = group.owner
        if group.kind == "amine":
            # The amine inventory's center is N and its owner is unset.
            # P-62.2.3 expresses an ordinary subordinate NH2 as amino; prove
            # its direct carbon attachment without borrowing another class's
            # center/owner convention. N-substituted prefixes need more roles.
            nitrogen = mol.GetAtomWithIdx(group.center)
            neighbors = tuple(nitrogen.GetNeighbors())
            if (set(group.atoms) != {group.center}
                    or nitrogen.GetAtomicNum() != 7 or nitrogen.GetFormalCharge()
                    or nitrogen.GetNumRadicalElectrons() or nitrogen.GetIsotope()
                    or nitrogen.GetTotalValence() != 3
                    or nitrogen.GetTotalNumHs(includeNeighbors=True) != 2
                    or len(neighbors) != 1 or neighbors[0].GetAtomicNum() != 6
                    or mol.GetBondBetweenAtoms(group.center, neighbors[0].GetIdx())
                        .GetBondTypeAsDouble() != 1):
                continue
            host = neighbors[0].GetIdx()
        if host not in path_atoms:
            continue
        atoms = set(group.atoms)
        if group.kind == 'amide':
            from .carboxamide_prefixes import preferred_carbamoyl_prefix
            text, owned = mapped_branch(mol, Chem, group.center, host)
            if (preferred_carbamoyl_prefix(mol, group.center, host, text, owned) is not None
                    and not set(owned) & (path_atoms | claimed)):
                facts.append((str(host), text, tuple(sorted(owned))))
                claimed.update(owned)
                continue
        if not atoms or atoms & path_atoms or atoms & claimed:
            continue
        text, owned = mapped_branch(mol, Chem, group.center, host)
        if (not text or set(owned) != atoms
                or group.kind == "amine" and text != "amino"):
            continue
        facts.append((str(host), text, tuple(sorted(atoms))))
        claimed.update(atoms)
    return tuple(facts)


def _functionalized_branch_proofs(mol, Chem, path, principal_groups,
                                  *, reserved_direct_prefix_atoms=()):
    """Pre-prove direct carbon-rooted functionalized branches on a candidate.

    Ordinary carbon-only and existing noncarbon routes remain with their old
    producers. A recognized pendant source group inside a carbon branch makes
    this adapter an obligation; an incomplete proof is a blocker.
    """
    from .functionalized_carbon_prefixes import functionalized_carbon_prefix

    parent_atoms = set(path)
    reserved_direct_prefix_atoms = set(reserved_direct_prefix_atoms)
    from .extended_inventory import selection_inventory
    carbamoyl_roots = {group.center for group in selection_inventory(mol).groups
                       if group.kind == 'amide'}
    selected_atoms = {idx for group in principal_groups for idx in group.atoms}
    selected_atoms.update(group.owner for group in principal_groups if group.owner >= 0)
    proofs = []
    seen_roots = set()
    for host in path:
        for neighbor in mol.GetAtomWithIdx(host).GetNeighbors():
            root = neighbor.GetIdx()
            if (root in parent_atoms or root in selected_atoms or root in seen_roots
                    or root in reserved_direct_prefix_atoms
                    or neighbor.GetAtomicNum() != 6):
                continue
            seen_roots.add(root)
            # A complete source-shaped carbamoyl unit has its own mapped
            # original C/O/N(+N-methyl) ownership. Do not decompose its amide
            # oxygen/N into ketone and amino roles of a carbon-chain prefix.
            if root in carbamoyl_roots:
                from .carboxamide_prefixes import preferred_carbamoyl_prefix
                text, owned = mapped_branch(mol, Chem, root, host)
                if preferred_carbamoyl_prefix(mol, root, host, text, owned) is not None:
                    continue
            proof, status = functionalized_carbon_prefix(
                mol, Chem, root, host, principal_groups,
            )
            if status == "blocked":
                return None
            if status == "complete":
                proofs.append(proof)
    scopes = [set(proof.scope) for proof in proofs]
    if any(scopes[i] & scopes[j] for i in range(len(scopes)) for j in range(i)):
        return None
    return tuple(proofs)


def _compare_annotated_numberings(left, right):
    """P-14.4 order for equivalent orientations of one saturated chain."""
    for field in ("suffix_locants", "indicated_h_locants",
                  "unsaturation_locants", "double_locants",
                  "prefix_locants", "prefix_citation_locants", "nonstandard_key",
                  "equivalent_isotope_key", "stereo_numbering_key"):
        a, b = getattr(left, field), getattr(right, field)
        if a is None or b is None:
            return None
        if a != b:
            return -1 if a < b else 1
    return 0


def _compare_annotated_numberings_with_unknown(left, right, left_unknown, right_unknown,
                                               has_stereo_features=False,
                                               has_parent_isotopes=False):
    """Compare P-14.4(i) before (j), conservatively gating isotope ties.

    If isotope locants would decide a pair that has parent stereo features,
    unmodified-numbering equivalence is not established by this bounded proof.
    Earlier criteria can still resolve the orientation without consulting the
    isotope or Xi locants.
    """
    fields = ("suffix_locants", "indicated_h_locants",
              "unsaturation_locants", "double_locants",
              "prefix_locants", "prefix_citation_locants", "nonstandard_key")
    for field in fields:
        a, b = getattr(left, field), getattr(right, field)
        if a is None or b is None:
            return None
        if a != b:
            return -1 if a < b else 1
    if has_parent_isotopes and has_stereo_features:
        return None
    isotope_left, isotope_right = left.equivalent_isotope_key, right.equivalent_isotope_key
    if isotope_left is None or isotope_right is None:
        return None
    if isotope_left != isotope_right:
        if has_stereo_features:
            return None
        return -1 if isotope_left < isotope_right else 1
    # Xi positions are not assigned a preferred direction here, but differing
    # unresolved-unit positions keep the late comparison live. Do not allow a
    # known-only descriptor key to decide while omitting that unknown unit.
    if tuple(left_unknown) != tuple(right_unknown):
        return None
    a, b = left.stereo_numbering_key, right.stereo_numbering_key
    if a is None or b is None:
        return None
    if a != b:
        return -1 if a < b else 1
    return 0


def _annotation_frontier(items, facts_by_item, annotation_proofs_by_item=None):
    """Apply P-44 annotation criteria and same-parent P-14 numbering rules."""
    from .parent_selection import _criteria

    edges = [set() for _ in items]
    annotation_proofs_by_item = annotation_proofs_by_item or {}
    for i, left in enumerate(items):
        for j in range(i + 1, len(items)):
            right = items[j]
            left_facts = facts_by_item[left]
            right_facts = facts_by_item[right]
            if set(left[0]) == set(right[0]):
                left_proof = annotation_proofs_by_item.get(left)
                right_proof = annotation_proofs_by_item.get(right)
                if left_proof is not None and right_proof is not None:
                    preference = _compare_annotated_numberings_with_unknown(
                        left_facts, right_facts,
                        left_proof.unknown_stereo_locants,
                        right_proof.unknown_stereo_locants,
                        bool(left_proof.stereo_features or right_proof.stereo_features),
                        bool(left_proof.isotope_ownership or right_proof.isotope_ownership),
                    )
                else:
                    preference = _compare_annotated_numberings(left_facts, right_facts)
            else:
                preference = 0
                for rule, a, b in _criteria(left_facts, right_facts):
                    if rule.startswith("P-45"):
                        # P-45 applies after a senior parent has been chosen.
                        break
                    if a is None or b is None:
                        preference = None
                        break
                    if a != b:
                        preference = -1 if a < b else 1
                        break
            if preference == -1:
                edges[i].add(j)
            elif preference == 1:
                edges[j].add(i)
    # A cycle signals contradictory evidence; retain all rather than
    # resolving it by atom-ID or candidate enumeration order.
    visiting, visited = set(), set()

    def has_cycle(index):
        if index in visiting:
            return True
        if index in visited:
            return False
        visiting.add(index)
        if any(has_cycle(other) for other in edges[index]):
            return True
        visiting.remove(index)
        visited.add(index)
        return False

    if any(has_cycle(index) for index in range(len(items))):
        return list(items)
    dominated = set().union(*edges) if edges else set()
    return [item for index, item in enumerate(items) if index not in dominated]


def _external_suffix_locations(mol, path, mapping, prefixes, locants, count, cyclic):
    """P-14.3.4.2(c) display omission; original numbering remains mapped.

    A saturated homogeneous monocycle with one characteristic attachment has
    equivalent unsubstituted positions. N-substitution belongs to the external
    amide unit and does not add an attachment to the ring. Other skeletal
    prefixes, annulations or required isotope locants retain the ring locant.
    """
    locations = "-" + ",".join(map(str, sorted(locants))) + "-"
    if not cyclic or count != 1 or tuple(locants) != (1,):
        return locations
    ring = frozenset(path)
    if (any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in path)
            or any(loc != "N" for loc, _ in prefixes)
            or any(set(other) & ring and frozenset(other) != ring
                   for other in mol.GetRingInfo().AtomRings())
            or any(mol.GetBondBetweenAtoms(path[i], path[(i + 1) % len(path)])
                   .GetBondTypeAsDouble() != 1 for i in range(len(path)))):
        return locations
    # P-14.3.3/P-82.6.1.1 require the other locants in an isotope-numbered
    # unit. Uniform-label omission is not inferred for an external acyl group.
    if any(mol.GetAtomWithIdx(i).GetIsotope()
           or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in mapping):
        return locations
    return ""


def _render(mol, Chem, groups, kind, path, external, cyclic, skipped, owner_components,
            *, parent_override=None, mapped_ring_prefixes=False, return_component=False,
            charge_role_proofs=(), functionalized_carbon_prefix_proofs=(),
            parent_annotation_proof=None, branch_resolver=None, locanted_acid_fallback=False,
            general_branches=False):
    proved_heteroring_atoms=set()

    # This barrier concerns labelled acid-suffix oxygen/hydrogen only. Carbon
    # labels and unlabelled inputs cannot trigger it; avoid reconstructing the
    # same whole-chain recognition for every alternative parent path.
    view = graph_view(mol, Chem)
    if kind == "acid" and any(mass and number in {1, 8}
                              for number, mass in zip(view.atomic_nums, view.isotopes)):
        from .acid_oxygen_isotope_methods import recognize_linear_monoacid
        from .oxygen_characteristic_suffixes import oxygen_group

        linear_acid = recognize_linear_monoacid(mol, Chem)
        if linear_acid is not None:
            acid_roles = oxygen_group(mol, linear_acid)
            role_atoms = (() if acid_roles is None else
                          acid_roles.oxygen_atoms + acid_roles.attached_hydrogens)
            if (role_atoms and any(mol.GetAtomWithIdx(i).GetIsotope()
                                   for i in role_atoms)
                    and set(path) != {atom.GetIdx() for atom in iter_atoms(mol)
                                      if atom.GetAtomicNum() == 6}):
                # The exact neutral monoacid scope proves one unbranched
                # carbon parent. Do not let a shorter formic/acetic acid
                # candidate survive merely because its local isotope grammar
                # is renderable after the maximal parent candidate declines.
                return None

    prefix_branch_ownership = []

    def branch(root,blocked):
        proof = next((item for item in functionalized_carbon_prefix_proofs
                      if item.root == root and item.parent_host == blocked), None)
        if proof is not None:
            if proof.scope & embedded:
                return "", frozenset()
            used_functionalized_proofs.append(proof)
            text, atoms = proof.text, proof.scope
            prefix_branch_ownership.append((
                int(root), int(blocked), frozenset(atoms), str(text),
            ))
            return text, atoms
        resolved = branch_resolver(mol, Chem, root, blocked) if branch_resolver else None
        if resolved is not None:
            text, atoms = resolved
        elif general_branches:
            from .component_mapping import mapped_general_branch

            text, atoms = mapped_general_branch(mol, Chem, root, blocked)
        else:
            text, atoms = mapped_branch(mol, Chem, root, blocked)
        if text and atoms:
            prefix_branch_ownership.append((
                int(root), int(blocked), frozenset(atoms), str(text),
            ))
            if general_branches:
                # This candidate carries separately scoped general prefixes.
                # It cannot resolve the preferred ring-prefix comparison.
                return text, atoms
            if mapped_ring_prefixes:
                # Nested ring prefixes can be reached through an amide N or
                # carbon chain. Only the actual mapped writer's propagated
                # ring ownership may discharge their construction barrier.
                proved_heteroring_atoms.update(mapped_branch_ring_atoms(
                    mol, Chem, root, blocked, text, atoms))
            elif mol.GetAtomWithIdx(root).IsInRing():
                # The P-28/P-29 tree writer supplies complete cyclic-unit
                # ownership even to legacy parent consumers. Rejoin its
                # actual rooted record; text alone cannot authorize this.
                from .tree_ring_assemblies import (
                    ring_assembly_prefix_outcome, _DEFAULT_RING_ASSEMBLY_PREFIX_OUTCOME,
                )
                from .mapped_ring_parents import ring_systems
                # The tree proof requires two complete physical ring systems.
                # Single systems cannot add assembly ownership; preserve any
                # explicitly replaced producer's call instead of bypassing it.
                if (ring_assembly_prefix_outcome is not _DEFAULT_RING_ASSEMBLY_PREFIX_OUTCOME
                        or sum(core <= atoms for core in ring_systems(mol)) >= 2):
                    assembly=ring_assembly_prefix_outcome(mol,Chem,root,blocked,atoms)
                    if (assembly.status=='complete' and assembly.decision.name==text
                            and assembly.decision.atoms==atoms):
                        proved_heteroring_atoms.update(frozenset().union(*assembly.decision.tree.systems))
        if mapped_ring_prefixes and text and mol.GetAtomWithIdx(root).IsInRing():
            from .mapped_ring_parents import rooted_prefix
            mapped_text,owned=rooted_prefix(mol,Chem,root,blocked,atoms)
            if mapped_text==text and owned==atoms:
                proved_heteroring_atoms.update(i for i in atoms
                    if mol.GetAtomWithIdx(i).IsInRing() and mol.GetAtomWithIdx(i).GetAtomicNum()!=6)
        return text,atoms
    if external:
        # Geminal suffix sites need primed element locants to distinguish labelled
        # groups or separately substituted amide N atoms. Keep unproved cases out.
        from collections import Counter

        owner_counts = Counter(g.owner for g in groups)
        duplicate_owners = {i for i, count in owner_counts.items() if count > 1}
        for owner in duplicate_owners:
            attached = [g for g in groups if g.owner == owner]
            if any(
                mol.GetAtomWithIdx(i).GetIsotope() for g in attached for i in g.atoms
            ):
                from .amide_isotopes import amide_isotope_partition
                partition = amide_isotope_partition(mol, attached)
                if (partition is None or partition.selection.barrier
                        or len(partition.selection.multiplicity_groups) != 1):
                    return ''
    mapping = {idx: str(pos) for pos, idx in enumerate(path, 1)}
    amide_n_labels = {}
    if kind == 'amide' and external:
        from .amide_site_locants import external_amide_nitrogen_locants
        assignment = external_amide_nitrogen_locants(mol, Chem, groups, mapping)
        if assignment is None:
            return None
        amide_n_labels = dict(assignment.nitrogen_locants)
    embedded = set()
    used_functionalized_proofs = []
    prefixes = []
    characteristic_prefix_roles = []
    typed_charge_prefixes = {}
    for proof in charge_role_proofs:
        if proof.kind not in {"nitro", "azido", "nitrooxy", "isocyano"}:
            continue
        typed_charge_prefixes.setdefault(proof.attachment_atom, []).append(
            (proof.atoms[0], proof.kind, set(proof.atoms))
        )
    locants = []
    n_labels = {}
    for number, g in enumerate(groups):
        loc = mapping[g.owner if external else g.center]
        locants.append(int(loc))
        if external:
            mapping[g.center] = (loc if kind == 'amide' and mol.GetAtomWithIdx(g.center).GetIsotope()
                                 else "C" + loc if kind in _CARBON else loc)
        if kind in {"amide", "sulfonamide"}:
            if kind == "sulfonamide":
                from .sulfur_selection import sulfonamide_n_sites

                n_sites = sulfonamide_n_sites(mol, g)
                if n_sites is None:
                    return None
                # P-66 has N locants for distinct amide sites; geminal
                # sulfonamide Ns need an explicit priming rule not supplied by
                # this bounded renderer.
                if any(other is not g and other.owner == g.owner for other in groups):
                    return None
            else:
                nitrogen = mol.GetAtomWithIdx(g.ligand)
                n_sites = tuple(
                    n.GetIdx() for n in nitrogen.GetNeighbors()
                    if n.GetIdx() != g.center and n.GetAtomicNum() > 1
                )
            nloc = amide_n_labels.get(g.ligand) or (
                (
                    ("N" if loc == "1" else "N'")
                    if kind == "amide" and not external and len(path) == 2
                    else "N" + loc
                )
                if len(groups) > 1
                else "N"
            )
            n_labels[g.ligand] = nloc
            for n_idx in n_sites:
                text, atoms = branch(n_idx, g.ligand)
                if not text or embedded & atoms:
                    return None
                prefixes.append((nloc, text))
                embedded.update(atoms)
        for idx in g.atoms:
            if idx == g.center:
                continue
            atom = mol.GetAtomWithIdx(idx)
            order = mol.GetBondBetweenAtoms(g.center, idx).GetBondTypeAsDouble()
            if atom.GetAtomicNum() == 7:
                mapping[idx] = n_labels.get(idx, "N" + loc)
            elif atom.GetAtomicNum() == 8:
                mapping[idx] = (
                    "O'"
                    if order == 2 and kind in {"acid", "carboxylate", "ester"}
                    else "O"
                ) + (
                    loc
                    if len(groups) > 1 or kind in {"acid", "carboxylate", "ester"}
                    else ""
                )
            else:
                return None
        if kind == "ester":
            embedded.update(owner_components[number][1])
    owned = set(mapping)
    for idx in path:
        for other in mol.GetAtomWithIdx(idx).GetNeighbors():
            n = other.GetIdx()
            if n in owned or n in skipped or other.GetAtomicNum() <= 1:
                continue
            typed_charge_prefix = next(
                (fact for fact in typed_charge_prefixes.get(idx, ()) if fact[0] == n),
                None,
            )
            if typed_charge_prefix is not None:
                _root, text, atoms = typed_charge_prefix
                atoms = set(atoms)
            else:
                text, atoms = branch(n, idx)
                typed_charge_prefix = None
            if (
                not text
                or atoms & (owned | embedded)
                or any(char.isspace() for char in text)
            ):
                return None
            typed_nitramide_branch = False
            if mol.GetAtomWithIdx(n).GetAtomicNum() == 7:
                from .nitrogen_oxo_amide_prefixes import nitrogen_oxo_amide_prefix
                typed_nitramide_branch = nitrogen_oxo_amide_prefix(
                    mol, n, idx, atoms,
                ) is not None
            if len(atoms) > 1 and not typed_nitramide_branch and typed_charge_prefix is None:
                charged = {i for i in atoms if mol.GetAtomWithIdx(i).GetFormalCharge()}
                if charged:
                    from .neutral_charge_components import validate_neutral_charge_role_proofs

                    # A nitro/azido resonance charge wholly inside a mapped
                    # branch is part of a neutral prefix, not a charge on the
                    # prefix itself. Leave that writer's text intact so the
                    # normal enclosure rules and exact branch ownership apply.
                    neutral_atoms = set().union(*(
                        set(proof.atoms) for proof in charge_role_proofs
                        if proof.kind in {"nitro", "azido", "nitrooxy", "isocyano"}
                        and set(proof.atoms) | {proof.attachment_atom} <= set(atoms)
                    ))
                    if (not charged <= neutral_atoms
                            or not validate_neutral_charge_role_proofs(mol, charge_role_proofs)):
                        text = "(" + text + ")"
            if typed_charge_prefix is None:
                from .nonstandard_parent_evidence import characteristic_prefix_role
                from .extended_inventory import selection_inventory

                inventory = selection_inventory(mol)
                if inventory.supported:
                    role = characteristic_prefix_role(
                        mol, idx, text, atoms, inventory.groups,
                    )
                    if role is not None:
                        characteristic_prefix_roles.append(role)
            prefixes.append((mapping[idx], text))
            embedded.update(atoms)
    mapping = hydrogen_bindings(mol, mapping)
    if not fully_covered(mol, mapping, embedded):
        return None
    if parent_override is not None:
        parent = parent_override
    elif cyclic:
        parent = (
            "benzene"
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in path)
            and len(path) == 6
            else ("cyclo" + ROOTS[len(path)] + "ane" if len(path) in ROOTS else "")
        )
    else:
        parent = complete_parent_name(parent_chain_name(mol, path))
    if not general_branches and any(
        mol.GetAtomWithIdx(i).IsInRing() and mol.GetAtomWithIdx(i).GetAtomicNum() != 6
        for i in embedded-proved_heteroring_atoms
    ):
        return None
    if not parent:
        return None
    count = len(groups)
    if kind in _SULFUR:
        from .sulfur_selection import sulfur_suffix, sulfur_suffix_locants

        suffix = sulfur_suffix(kind)
        if not suffix:
            return None
        numeric_prefix_count = sum(loc.isdigit() for loc, _ in prefixes)
        parent_carbons = [i for i in path if mol.GetAtomWithIdx(i).GetAtomicNum() == 6]
        isotope_masses = [mol.GetAtomWithIdx(i).GetIsotope() for i in parent_carbons]
        uniform_full_carbon_label = (
            bool(isotope_masses)
            and all(isotope_masses)
            and len(set(isotope_masses)) == 1
        )
        isotope_needs_parent_locants = (
            len(parent_carbons) > 1
            and any(isotope_masses)
            and not uniform_full_carbon_label
        )
        locations = sulfur_suffix_locants(
            count=count,
            locants=tuple(sorted(locants)),
            cyclic=cyclic,
            path_length=len(path),
            prefix_count=numeric_prefix_count,
            force_locants=isotope_needs_parent_locants,
        )
        ending = (
            parent + locations + (MULT.get(count, "") if count > 1 else "") + suffix
        )
    elif external:
        suffix = {
            "acid": "carboxylic acid",
            "carboxylate": "carboxylate",
            "ester": "carboxylate",
            "amide": "carboxamide",
            "nitrile": "carbonitrile",
        }[kind]
        ending = (
            parent
            + _external_suffix_locations(mol, path, mapping, prefixes,
                                         locants, count, cyclic)
            + (MULT.get(count, "") if count > 1 else "")
            + suffix
        )
    elif count == 1 and kind in {"acid", "carboxylate", "amide", "ester"}:
        from .parent_rules import _generic_acyl_functional_parent_text

        ending = _generic_acyl_functional_parent_text(
            mol,
            Chem,
            ROOTS[len(path)],
            path[1:],
            kind if kind in {"acid", "amide"} else "oate",
        )
    elif kind in {"ketone", "alcohol"}:
        from .formatting import attach_suffix, multiplied_suffix

        suffix = "one" if kind == "ketone" else "ol"
        ending = attach_suffix(parent, multiplied_suffix(count, suffix), sorted(locants))
    else:
        ending = _suffix_parent(
            parent,
            "ester" if kind == "carboxylate" else kind,
            count,
            locants,
            False,
            len(path),
        )
    if cyclic and parent == "benzene" and count == 1:
        retained = {
            "acid": "benzoic acid",
            "carboxylate": "benzoate",
            "ester": "benzoate",
            "amide": "benzamide",
            "nitrile": "benzonitrile",
        }.get(kind)
        if retained:
            ending = retained
            mapping[groups[0].center] = "7"
        elif kind == "alcohol":
            ending = "phenol"
    if not ending:
        return None
    local_isotopes = frozenset()
    amide_evidence = None
    isotope_parts = ()
    isotope_role_evidence = None
    isotope_role_evidence_mode = ""
    if kind == 'amide':
        from .amide_isotopes import amide_suffix_isotopes
        location = (_external_suffix_locations(mol, path, mapping, prefixes,
                    locants, count, cyclic) if external else '')
        amide = amide_suffix_isotopes(mol, groups, ending, external=external,
            hydride=parent, location_text=location)
        if amide is None:
            return None
        ending, local_isotopes, amide_evidence = amide
    if kind == "acid":
        from .oxygen_characteristic_suffixes import oxygen_group

        oxygen_units = tuple(oxygen_group(mol, group) for group in groups)
        modified = any(
            unit is not None and any(mol.GetAtomWithIdx(i).GetIsotope()
                                    for i in unit.oxygen_atoms + unit.attached_hydrogens)
            for unit in oxygen_units
        )
        if modified:
            if count != 1:
                return None
            from .acid_oxygen_isotope_methods import (
                acid_oxygen_isotope_component, recognize_linear_monoacid,
            )

            if recognize_linear_monoacid(mol, Chem) is not None:
                typed_acid = acid_oxygen_isotope_component(mol, Chem, ending)
                if (typed_acid is None
                        or typed_acid.record.status == "clause-derived-count"):
                    return None
                isotope_role_evidence = typed_acid.record
                local_isotopes = typed_acid.record.owned_atom_ids
                if ending in {"formic acid", "acetic acid"}:
                    isotope_parts = typed_acid.record.parts
                    isotope_role_evidence_mode = "component-parts"
                elif ending.endswith("oic acid"):
                    # The cited systematic hydroxyl-H descriptor is inserted
                    # by the acid suffix writer, not by the parent isotope
                    # descriptor builder. Retain its atom evidence on the
                    # construction while leaving it out of parent-level parts.
                    if (len(typed_acid.record.parts) != 1
                            or typed_acid.record.parts[0][0] != "H"):
                        return None
                    ending = typed_acid.component_text
                    isotope_role_evidence_mode = "typed-suffix"
                else:
                    return None
            else:
                # Outside the exact neutral linear monoacid scope, preserve
                # legacy behavior. The separate status frontier owns blocking
                # for recognized in-scope but unproved O/H constructions.
                from .oxygen_characteristic_suffixes import (
                    retained_acid_isotope_parts, systematic_acid_suffix,
                )
                if ending in {"formic acid", "acetic acid", "benzoic acid"}:
                    isotope_parts, local_isotopes = retained_acid_isotope_parts(mol, groups[0])
                    if not isotope_parts:
                        return None
                else:
                    ending, local_isotopes = systematic_acid_suffix(mol, ending, groups[0])
                    if not ending:
                        return None
    retained_acid_evidence = None
    if (ending in {"acetic acid", "acetate"}
            and kind in {"acid", "carboxylate", "ester"}):
        from .retained_acid_rules import retained_acid_locant_evidence

        retained_acid_evidence = retained_acid_locant_evidence(
            mol, Chem, parent_class="acetic acid", principal_kind=kind,
            parent_atoms=frozenset(mapping), atom_locants=tuple(mapping.items()),
            prefixes=tuple(prefixes),
            isotope_forces_locants=any(atom.GetIsotope() for atom in iter_atoms(mol)),
        )
    construction = MappedComponent(
            ending,
            tuple(mapping.items()),
            frozenset(mapping),
            tuple(prefixes),
            frozenset(embedded),
            systematic_parent=kind in _SULFUR,
            nitrogen_site_locants=kind in {"amide", "sulfonamide"},
            local_isotope_atoms=local_isotopes,
            amide_isotope_evidence=amide_evidence,
            isotope_descriptor_parts=isotope_parts,
            isotope_role_evidence=isotope_role_evidence,
            isotope_role_evidence_mode=isotope_role_evidence_mode,
            retained_acid_locant_evidence=retained_acid_evidence,
            omit_methyl_prefix_locant=(general_branches and kind == 'sulfonamide'
                and len(path) == count == len(prefixes) == 1 and prefixes[0][0] == '1'
                and not any(mol.GetAtomWithIdx(i).GetIsotope() for i in mapping)),
            characteristic_prefix_roles=tuple(characteristic_prefix_roles),
            functionalized_carbon_prefix_proofs=tuple(used_functionalized_proofs),
            prefix_branch_ownership=tuple(prefix_branch_ownership),
            parent_annotation_proof=parent_annotation_proof,
        )
    name = decorate_component(mol, Chem, construction)
    if not name and locanted_acid_fallback and retained_acid_evidence is not None:
        # A valid general form can still be emitted when the more specific
        # retained-parent omission sequence is not proved by this writer.
        from dataclasses import replace
        construction = replace(construction, retained_acid_locant_evidence=None)
        name = decorate_component(mol, Chem, construction)
    if not name:
        return None
    component_name = name
    owners = []
    if kind == "ester":
        owners = [x[0] for x in owner_components]
        if len(set(owners)) == 1:
            prefix = format_prefixes(
                [("", owner) for owner in owners], omit_locants=True
            )
        else:
            # Explicit distributive locants attach each unlike owner to its own
            # carboxyl group. Their order follows the owner names, not input IDs.
            prefix = " ".join(
                format_prefixes([(str(loc), owner)])
                for owner, loc in sorted(
                    zip(owners, locants),
                    key=lambda p: (prefix_alphabetization_key(p[0]), p[1]),
                )
            )
        name = prefix + " " + name
    rendered = (
        name,
        tuple(sorted(locants)),
        tuple(sorted(int(loc) for loc, _ in prefixes if loc.isdigit())),
        mapping,
        tuple(prefixes),
        component_name,
        tuple(owners),
    )
    return (rendered, construction) if return_component else rendered
