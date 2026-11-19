"""Call-local parent construction plans and conservative graph-only pruning.

Plans are not eligible named candidates. Their bounds describe the best a
construction could achieve on the supplied original parent and suffix sites.
A missing bound remains live. Every deferred plan must be discharged against
the final validated selection, or constructed before a decision is returned.
No callbacks, scratch molecules or suspended generators enter molecule caches.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from .graph_model import graph_view
from .parent_selection import ParentFacts, compare_parents


class DeferredCompatibilityCandidates:
    """Call-local preparation with explicit admission and ownership proofs.

    Annotated inputs cannot admit any established-ring candidate into the
    shared selector. Their preparation is restored when subsequent fallback
    stages may still need its evidence; immediate typed general-name delivery
    can retain an explicit unresolved preference without loading this pool.

    Ordinary ring facts require a connected all-ring skeleton, with each
    external heavy component attached once. Within any connected ring system,
    a proper skeletal subset leaves an external annular component attached at
    least twice. A connected all-ring parent cannot span separate systems.
    Thus each admissible compatibility candidate must use one complete original
    ring system. All principal-bearing systems must be owned before preparation
    can be omitted. No template, rendering or priority cutoff is changed;
    a miss or unresolved unnamed result restores the entire eager pool.
    """

    def __init__(self, parent_atoms, load, *, annotation_excluded=False, ring_systems=None):
        self.annotation_excluded = annotation_excluded
        self.ownership_excluded = False
        self.parent_atoms = parent_atoms
        self.ring_systems = ((parent_atoms,) if ring_systems is None else tuple(ring_systems))
        self._load = load
        self._prepared = None

    @classmethod
    def for_context(cls, ctx, rules, load):
        from .molecule_cache import mol_cache_get, naming_cache_state
        from .molecule_iteration import iter_atoms, iter_bonds

        if (ctx.telemetry is not None or ctx.compatibility_eager
                or not rules or not all(rule.ring_parent_scope for rule in rules)
                or naming_cache_state(ctx.mol)):
            return load()
        if ctx.unresolved_preference:
            # The preceding parent handoff can leave a preference unresolved
            # without invalidating this same-ring ownership proof. The loader
            # retains that flag and its candidate safety filter when restored;
            # cross-route construction never changes it. Preserve the original
            # eager native operation order for partially initialized inputs.
            view = mol_cache_get(ctx.mol, 'graph_view')
            if view is None or ctx.mol.NeedsUpdatePropertyCache():
                return load()
            try:
                ctx.mol.GetRingInfo().NumRings()
            except RuntimeError:
                return load()
        else:
            view = graph_view(ctx.mol, ctx.Chem)
        if (any(view.isotopes)
                or any(int(atom.GetChiralTag()) for atom in iter_atoms(ctx.mol))
                or any(int(bond.GetStereo()) for bond in iter_bonds(ctx.mol))):
            if ctx.mol.NeedsUpdatePropertyCache():
                return load()
            # _finish_cross_route_selection excludes every established-ring
            # candidate for these annotations, independently of its parent.
            # Keep the loader for a pending/outside outcome: its evidence and
            # caches may still be required by the subsequent fallback stages.
            return cls(frozenset(), load, annotation_excluded=True)
        if not view.ring_systems:
            return load()
        return cls(frozenset().union(*view.ring_systems), load,
                   ring_systems=view.ring_systems)

    @property
    def loaded(self):
        return self._prepared is not None

    def restore(self):
        if self._prepared is None:
            self._prepared = tuple(self._load())
        return self._prepared

    def resolve(self, owned, principal):
        # This proof applies only to the most recent actual typed pool.
        self.ownership_excluded = False
        if self.loaded:
            return self._prepared
        if self.annotation_excluded:
            return ()
        for parent_atoms in self.ring_systems:
            centers = frozenset(group.center for group in principal
                                if group.center in parent_atoms
                                or group.owner in parent_atoms)
            if centers and (parent_atoms, centers) not in owned:
                return self.restore()
        self.ownership_excluded = True
        return ()


@dataclass(frozen=True, slots=True)
class ParentProposalPlan:
    route: str
    parent_atoms: tuple[int, ...]
    principal_centers: frozenset[int]
    bound: ParentFacts | None
    construct: Callable = field(repr=False, compare=False)
    key: tuple | None = None
    ring_route: bool = False
    # Opt in only when the callback itself wraps at most one proposal. This
    # bounds cycle vertices, never eligibility or the eventual comparison.
    single_proposal: bool = False


@dataclass(frozen=True, slots=True)
class DeferredProposalEvidence:
    route: str
    parent_atoms: tuple[int, ...]
    principal_centers: frozenset[int]
    bound: ParentFacts | None
    disposition: str
    comparison: object = None


def physical_parent(proposal):
    return frozenset(proposal.parent_atoms), proposal.principal_centers


def graph_proposal_bound(mol, atoms, groups, *, cyclic=False):
    """An optimistic suffix count and intrinsic skeletal criteria only.

    The producer fixes the parent atom set and supplies every possible suffix
    site for that plan. An eventual valid rendering can cover fewer sites,
    never more. Locants, vocabulary, annotations and eligibility remain open.
    """
    atoms, groups = frozenset(atoms), tuple(groups)
    if not atoms or not groups:
        return None
    view = graph_view(mol, None)
    edges = sum(neighbor in atoms for idx in atoms for neighbor in view.neighbors[idx]) // 2
    return ParentFacts(
        principal_class=groups[0].seniority, suffix_count=len(groups),
        kind="ring" if cyclic else "chain",
        elements=tuple(sorted(Counter(view.atomic_nums[idx] for idx in atoms).items())),
        ring_count=edges - len(atoms) + 1,
    )


def strict_graph_dominance(facts, bound):
    if bound is None:
        return None
    comparison = compare_parents(facts, bound)
    # Only these intrinsic criteria survive a change of renderer/numbering
    # for the same parent. Never use a later tie-breaker to prune a plan.
    if comparison.preference == -1 and (
        comparison.rule in {"P-41/P-43", "P-44.1.1", "P-44.1.2", "P-44.1.2.2",
                            "P-44.3.1", "P-44.3.2", "P-44.3.3"}
        or comparison.rule.startswith("P-44.2.1(")
    ):
        return comparison
    return None


class RouteProposalPool:
    """Schedule construction independently of final candidate enumeration."""

    def __init__(self, proposals):
        self.proposals = tuple(proposals)
        self.rendered = {index: (proposal,) for index, proposal in enumerate(self.proposals)
                         if not isinstance(proposal, ParentProposalPlan)}
        self.skipped = {}
        self.memo = {}
        self._ordinary_ready = [proposal for proposal in self.proposals
                                if not isinstance(proposal, ParentProposalPlan)]
        self._rings_ready = []

    def construct(self, index):
        plan = self.proposals[index]
        if index not in self.rendered:
            if plan.key is not None and plan.key in self.memo:
                rows = self.memo[plan.key]
            else:
                rows = tuple(plan.construct())
                if plan.key is not None:
                    self.memo[plan.key] = rows
            self.rendered[index] = rows
            (self._rings_ready if plan.ring_route else self._ordinary_ready).extend(rows)
        self.skipped.pop(index, None)

    def realize(self, *, lazy=True):
        plans = [index for index in range(len(self.proposals)) if index not in self.rendered]

        def schedule(index):
            plan = self.proposals[index]
            count = plan.bound.suffix_count if plan.bound is not None else None
            # Scheduling is only a performance choice. Unknown comparisons
            # still demand construction and the original partial comparator.
            return (-(count if count is not None else 0), not plan.ring_route,
                    -len(plan.parent_atoms), index)

        for index in sorted(plans, key=schedule) if lazy else plans:
            plan = self.proposals[index]
            ordinary, rings = self._ordinary_ready, self._rings_ready
            if lazy:
                identity = physical_parent(plan)
                if not plan.ring_route and plan.parent_atoms and any(
                    physical_parent(ring) == identity for ring in rings
                ):
                    self.skipped[index] = DeferredProposalEvidence(
                        plan.route, plan.parent_atoms, plan.principal_centers,
                        plan.bound, "typed-ring-owned",
                    )
                    continue
                decisive = next((comparison for proposal in (*ordinary, *rings)
                                 if (comparison := strict_graph_dominance(
                                     proposal.parent_selection, plan.bound)) is not None), None)
                if decisive is not None:
                    self.skipped[index] = DeferredProposalEvidence(
                        plan.route, plan.parent_atoms, plan.principal_centers,
                        plan.bound, "strictly-junior", decisive,
                    )
                    continue
            self.construct(index)

    def materialized(self):
        ordinary, rings = [], []
        for index, proposal in enumerate(self.proposals):
            rows = self.rendered.get(index, ())
            if isinstance(proposal, ParentProposalPlan) and proposal.ring_route:
                rings.extend(rows)
            else:
                ordinary.extend(rows)
        legacy = [proposal for proposal in ordinary if proposal.enumeration_key is not None]
        if legacy:
            ordinary = (sorted(legacy, key=lambda proposal: proposal.enumeration_key)
                        + [proposal for proposal in ordinary if proposal.enumeration_key is None])
        return ordinary, rings

    def ensure_cycle_safety(self):
        """Do not hide a source-comparison cycle among deferred losers.

        A plan is one potential numbered proposal. Unknown comparisons are
        overapproximated by edges in both directions; proved strict bounds
        permit only the senior-to-junior edge. A real preference cycle needs
        at least three vertices. Construct deferred vertices in any larger
        strongly connected component before accepting the optimized pool.
        This preserves the eager selector's global cycle barrier without
        assuming transitivity of its pair-dependent late criteria.
        """
        single_keys = None
        while True:
            ring_owners = {physical_parent(row) for row in self._rings_ready}
            for index, evidence in tuple(self.skipped.items()):
                if (not self.proposals[index].ring_route
                        and physical_parent(evidence) in ring_owners):
                    self.skipped[index] = DeferredProposalEvidence(
                        evidence.route, evidence.parent_atoms, evidence.principal_centers,
                        evidence.bound, "typed-ring-owned",
                    )
            pending = [index for index, evidence in self.skipped.items()
                       if evidence.disposition == "strictly-junior"]
            if not pending:
                return
            # Actual typed ring ownership removes compatibility spellings
            # before the eager comparison as well. Annotation-proof peers
            # remain present for the original ownership validator.
            ready = [row for row in self._ordinary_ready
                     if physical_parent(row) not in ring_owners
                     or getattr(row.construction, "parent_annotation_proof", None) is not None]
            ready.extend(self._rings_ready)
            nodes = [(None, row.parent_selection) for row in ready]
            nodes.extend((index, self.proposals[index].bound) for index in pending)
            edges, reverse = [0] * len(nodes), [0] * len(nodes)

            def edge(left, right):
                edges[left] |= 1 << right
                reverse[right] |= 1 << left

            for left, (left_plan, left_facts) in enumerate(nodes):
                for right in range(left + 1, len(nodes)):
                    right_plan, right_facts = nodes[right]
                    if left_plan is None and right_plan is None:
                        preference = compare_parents(left_facts, right_facts).preference
                        if preference == -1:
                            edge(left, right)
                        elif preference == 1:
                            edge(right, left)
                    elif left_plan is None and strict_graph_dominance(left_facts, right_facts) is not None:
                        edge(left, right)
                    elif right_plan is None and strict_graph_dominance(right_facts, left_facts) is not None:
                        edge(right, left)
                    else:
                        edge(left, right)
                        edge(right, left)
            uncertain = []
            for component in _strong_components(edges, reverse):
                if len(component) >= 3:
                    if single_keys is None:
                        single_keys = self._single_proposal_keys()
                    if single_keys:
                        aliases, count = set(), 0
                        for node in component:
                            index = nodes[node][0]
                            key = self.proposals[index].key if index is not None else None
                            if key in single_keys:
                                if key in aliases:
                                    continue
                                aliases.add(key)
                            count += 1
                        if count < 3:
                            # These aliases already share exactly one memo
                            # output tuple. A zero-or-one callback contributes
                            # at most one vertex after the final duplicate
                            # collapse, so two such vertices cannot cycle.
                            continue
                    uncertain.extend(nodes[node][0] for node in component if nodes[node][0] is not None)
            if not uncertain:
                return
            # A valid ring constructor may discharge many compatibility
            # plans at once. If none is available, resolve the entire possible
            # cycle rather than suppressing an unresolved competitor.
            rings = [index for index in uncertain if self.proposals[index].ring_route]
            for index in rings or uncertain:
                self.construct(index)

    def _single_proposal_keys(self):
        """Keep opaque/custom memo keys on the original conservative path."""
        if not any(isinstance(plan, ParentProposalPlan) and plan.single_proposal is True
                   for plan in self.proposals):
            return set()

        def plain(value):
            if type(value) in {str, int, bool, type(None)}:
                return True
            return type(value) in {tuple, frozenset} and all(plain(item) for item in value)

        single, other = set(), set()
        for plan in self.proposals:
            if not isinstance(plan, ParentProposalPlan) or plan.key is None:
                continue
            if not plain(plan.key):
                return set()
            (single if plan.single_proposal is True else other).add(plan.key)
        # A shared memo slot can contain multiple rows if any producer of
        # that key lacks the cardinality proof. Do not merge those vertices.
        return single - other

    @property
    def evidence(self):
        return tuple(self.skipped[index] for index in sorted(self.skipped))

    def discharged(self, selection):
        if selection is None:
            return not self.skipped
        selected = selection.selected
        for evidence in self.evidence:
            if evidence.disposition == "strictly-junior":
                if strict_graph_dominance(selected.parent_selection, evidence.bound) is None:
                    return False
            else:
                owners = [proposal for proposal in selection.candidates
                          if physical_parent(proposal) == physical_parent(evidence)]
                if not any(proposal.route.startswith("mapped-ring:")
                           or getattr(proposal.construction, "parent_annotation_proof", None) is not None
                           for proposal in owners):
                    return False
        return True

    def realize_remaining(self):
        for index in tuple(self.skipped):
            self.construct(index)


def _strong_components(edges, reverse):
    """Iterative Kosaraju traversal over compact adjacency bitsets."""
    visited, order = 0, []
    for start in range(len(edges)):
        if visited & (1 << start):
            continue
        visited |= 1 << start
        stack = [(start, edges[start])]
        while stack:
            node, neighbors = stack[-1]
            neighbors &= ~visited
            if neighbors:
                bit = neighbors & -neighbors
                other = bit.bit_length() - 1
                stack[-1] = node, neighbors ^ bit
                visited |= bit
                stack.append((other, edges[other]))
            else:
                order.append(node)
                stack.pop()
    remaining = (1 << len(edges)) - 1
    for start in reversed(order):
        if not remaining & (1 << start):
            continue
        component, stack = [], [start]
        remaining ^= 1 << start
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = reverse[node] & remaining
            remaining ^= neighbors
            while neighbors:
                bit = neighbors & -neighbors
                stack.append(bit.bit_length() - 1)
                neighbors ^= bit
        yield component
