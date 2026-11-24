"""Bounded complete numbering frontiers for simple saturated tricyclic cores.

The domain is a connected, biconnected, all-carbon graph of cycle rank three
whose edges can be partitioned into a longest simple main ring, a simple main
bridge, and one independent secondary bridge. This is a numbering helper only:
retained/fused/spiro parent eligibility and all non-ring naming remain with the
caller. Enumeration caps produce an explicit ``incomplete`` result with no
candidate maps, so a partial frontier can never select an atom-order favorite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class TricyclicNumbering:
    """One map from original graph identifiers to a source-valid numbering."""

    sequence: tuple[Hashable, ...]
    locants: tuple[tuple[Hashable, int], ...]
    main_ring: tuple[Hashable, ...]
    main_ring_segments: tuple[int, int]
    main_bridge: tuple[Hashable, ...]
    secondary_bridge: tuple[Hashable, ...]
    secondary_attachment_locants: tuple[int, int]
    descriptor_lengths: tuple[int, int, int, int]

    @property
    def bridge_descriptor(self):
        a, b, main, secondary = self.descriptor_lengths
        left, right = self.secondary_attachment_locants
        return f'tricyclo[{a}.{b}.{main}.{secondary}^{left},{right}]'


@dataclass(frozen=True, slots=True)
class TricyclicFrontier:
    status: str  # complete, unsupported, incomplete
    candidates: tuple[TricyclicNumbering, ...] = ()
    reason: str = ""
    explored_states: int = 0
    decompositions: int = 0


def _order_key(value):
    # Used only to make traversal and returned frontier order reproducible.
    # It is never included in a P-23 preference key.
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _normalize(adjacency: Mapping[Hashable, Iterable[Hashable]]):
    if not adjacency:
        return None, "empty graph"
    vertices = tuple(adjacency)
    graph = {}
    for vertex in vertices:
        try:
            neighbors = tuple(adjacency[vertex])
            unique = set(neighbors)
        except (TypeError, KeyError):
            return None, "adjacency identifiers must be hashable and available"
        if vertex in unique or len(unique) != len(neighbors):
            return None, "self-loops and duplicate edges are unsupported"
        if not unique <= set(vertices):
            return None, "all edge endpoints must be declared"
        graph[vertex] = frozenset(unique)
    if any(vertex not in graph[neighbor] for vertex, neighbors in graph.items()
           for neighbor in neighbors):
        return None, "adjacency must be undirected"
    return graph, ""


def _edges(graph):
    return frozenset(frozenset((left, right)) for left, neighbors in graph.items()
                     for right in neighbors)


def _components(graph, removed=frozenset()):
    vertices = set(graph) - set(removed)
    groups = []
    while vertices:
        start = min(vertices, key=_order_key)
        seen, pending = {start}, [start]
        while pending:
            current = pending.pop()
            for neighbor in graph[current] & vertices:
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        groups.append(frozenset(seen))
        vertices -= seen
    return tuple(groups)


def _is_biconnected(graph):
    if len(graph) < 3 or len(_components(graph)) != 1:
        return False
    return all(len(_components(graph, {vertex})) == 1 for vertex in graph)


def _simple_cycles(graph, *, budget):
    """Enumerate all undirected simple cycles, stopping only by explicit cap."""
    vertices = sorted(graph, key=_order_key)
    seen_edge_sets = set()
    cycles = []
    for start in vertices:
        start_key = _order_key(start)

        def visit(current, path, visited):
            budget[0] += 1
            if budget[0] > budget[1]:
                budget[2] = "state cap reached during cycle enumeration"
                return
            for neighbor in sorted(graph[current], key=_order_key):
                if neighbor == start:
                    if len(path) >= 3:
                        cycle_edges = frozenset(frozenset((a, b))
                                                for a, b in zip(path, path[1:] + [start]))
                        if cycle_edges not in seen_edge_sets:
                            seen_edge_sets.add(cycle_edges)
                            cycles.append(tuple(path))
                    continue
                if neighbor in visited or _order_key(neighbor) < start_key:
                    continue
                visit(neighbor, path + [neighbor], visited | {neighbor})
                if budget[2]:
                    return

        visit(start, [start], {start})
        if budget[2]:
            return (), False
    return tuple(cycles), True


def _path_from_edges(edge_set, *, allowed_internal, allowed_endpoints):
    """Recover one path iff all listed edges form that path."""
    if not edge_set:
        return None
    adjacency = {}
    for edge in edge_set:
        left, right = tuple(edge)
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    if len(_components(adjacency)) != 1:
        return None
    ends = [vertex for vertex, neighbors in adjacency.items() if len(neighbors) == 1]
    if len(ends) != 2 or any(len(neighbors) not in {1, 2}
                             for neighbors in adjacency.values()):
        return None
    if not set(ends) <= set(allowed_endpoints):
        return None
    internal = set(adjacency) - set(ends)
    if not internal <= set(allowed_internal):
        return None
    first = min(ends, key=_order_key)
    path = [first]
    previous = None
    current = first
    while True:
        onward = adjacency[current] - ({previous} if previous is not None else set())
        if not onward:
            break
        nxt = min(onward, key=_order_key)
        path.append(nxt)
        previous, current = current, nxt
        if current in ends and current != first:
            break
    if len(path) != len(adjacency) or len(path) - 1 != len(edge_set):
        return None
    return tuple(path)


def _outside_paths(graph, ring, ring_edges, *, budget):
    """Enumerate all simple paths between ring atoms using only nonring edges."""
    found = {}
    ring_set = set(ring)

    def visit(start, current, path, visited):
        budget[0] += 1
        if budget[0] > budget[1]:
            budget[2] = "state cap reached during bridge-path enumeration"
            return
        for neighbor in sorted(graph[current], key=_order_key):
            edge = frozenset((current, neighbor))
            if edge in ring_edges:
                continue
            if neighbor == start or neighbor in visited:
                continue
            if neighbor in ring_set:
                candidate = path + [neighbor]
                reverse = tuple(reversed(candidate))
                path_key = min(tuple(map(_order_key, candidate)), tuple(map(_order_key, reverse)))
                found[path_key] = tuple(candidate if tuple(map(_order_key, candidate)) == path_key
                                         else reverse)
                continue
            visit(start, neighbor, path + [neighbor], visited | {neighbor})
            if budget[2]:
                return

    for start in sorted(ring_set, key=_order_key):
        visit(start, start, [start], {start})
        if budget[2]:
            return (), False
    return tuple(found.values()), True


def _path_edges(path):
    return frozenset(frozenset((left, right)) for left, right in zip(path, path[1:]))


def _numberings(cycle, bridge, secondary):
    """Apply P-23.2.3/.5 numbering to one tricyclic decomposition."""
    cycle_edges = _path_edges(cycle + (cycle[0],))
    bridge_left, bridge_right = bridge[0], bridge[-1]
    # The two arcs have the same bridgeheads. Recover both simple arcs directly
    # from the cycle's adjacency, independent of its arbitrary traversal start.
    adj = {atom: set() for atom in cycle}
    for edge in cycle_edges:
        a, b = tuple(edge)
        adj[a].add(b); adj[b].add(a)
    # Walk each direction around the cycle from the first bridgehead.
    neighbors = sorted(adj[bridge_left], key=_order_key)
    arcs = []
    for first_step in neighbors:
        path = [bridge_left, first_step]
        previous, current = bridge_left, first_step
        while current != bridge_right:
            nxts = adj[current] - {previous}
            if len(nxts) != 1:
                break
            nxt = next(iter(nxts)); path.append(nxt)
            previous, current = current, nxt
        if current == bridge_right:
            arcs.append(tuple(path))
    if len(arcs) != 2:
        return ()
    arcs.sort(key=lambda path: (-(len(path) - 2), tuple(map(_order_key, path))))
    long_arc, short_arc = arcs
    if len(long_arc) == len(short_arc):
        # Both main-ring arc orders are source-valid; orientation below will
        # enumerate both.  The selected long/short labels must not decide a tie.
        arc_orders = ((long_arc, short_arc), (short_arc, long_arc))
    else:
        arc_orders = ((long_arc, short_arc),)
    rows = []
    for first_head, second_head in ((bridge_left, bridge_right), (bridge_right, bridge_left)):
        oriented_bridge = bridge if bridge[0] == first_head else tuple(reversed(bridge))
        for first_arc, second_arc in arc_orders:
            # Orient both arcs from first to second bridgehead.
            first_arc = first_arc if first_arc[0] == first_head else tuple(reversed(first_arc))
            second_arc = second_arc if second_arc[0] == first_head else tuple(reversed(second_arc))
            base = (first_head,) + first_arc[1:-1] + (second_head,) + tuple(reversed(second_arc[1:-1])) + oriented_bridge[1:-1]
            if len(base) != len(set(base)):
                continue
            locant = {atom: i for i, atom in enumerate(base, 1)}
            sec_left, sec_right = secondary[0], secondary[-1]
            # P-23.2.5.2: the secondary bridge is numbered from the higher-
            # numbered bridgehead of the already numbered bicyclic system.
            sec_path = secondary if locant[sec_left] > locant[sec_right] else tuple(reversed(secondary))
            sequence = base + sec_path[1:-1]
            if len(sequence) != len(set(sequence)):
                continue
            locant = {atom: i for i, atom in enumerate(sequence, 1)}
            endpoints = tuple(sorted((locant[sec_left], locant[sec_right])))
            rows.append((sequence, endpoints))
    unique = {}
    for sequence, endpoints in rows:
        unique[sequence] = endpoints
    return tuple((seq, endpoints) for seq, endpoints in unique.items())


def tricyclic_von_baeyer_frontier(
    adjacency: Mapping[Hashable, Iterable[Hashable]],
    *, max_atoms: int = 18, max_states: int = 100_000,
    max_decompositions: int = 20_000,
) -> TricyclicFrontier:
    """Return all P-23-tied maps or an explicit unsupported/incomplete state.

    ``adjacency`` is the carbon topology before skeletal replacement or
    localized unsaturation. The caller validates valences and applies those
    operations to the tied maps. The accepted topology is a simple biconnected graph
    with cycle rank three, decomposable exactly into a main simple cycle and two
    internally disjoint bridge paths (either may be a direct edge, i.e. a
    zero-length bridge). This closure includes all such decompositions in the
    finite graph; it does not claim a parent name or resolve caller-level
    suffix, substituent, isotope, stereo, fused, or retained-vocabulary rules.
    """
    graph, error = _normalize(adjacency)
    if error:
        return TricyclicFrontier("unsupported", reason=error)
    if len(graph) > max_atoms:
        return TricyclicFrontier("incomplete", reason="atom cap reached before enumeration")
    if max_states <= 0 or max_decompositions <= 0:
        return TricyclicFrontier("incomplete", reason="nonpositive enumeration cap")
    if not _is_biconnected(graph):
        return TricyclicFrontier("unsupported", reason="graph is not biconnected")
    if any(len(neighbors) < 2 for neighbors in graph.values()):
        return TricyclicFrontier("unsupported", reason="all ring atoms must have degree at least two")
    edge_set = _edges(graph)
    if len(edge_set) - len(graph) + 1 != 3:
        return TricyclicFrontier("unsupported", reason="cycle rank is not three")

    budget = [0, max_states, ""]
    cycles, complete = _simple_cycles(graph, budget=budget)
    if not complete:
        return TricyclicFrontier("incomplete", reason=budget[2], explored_states=budget[0])
    candidates = []
    decompositions = 0
    for cycle in cycles:
        ring_edges = _path_edges(cycle + (cycle[0],))
        ring_set = set(cycle)
        bridges, complete = _outside_paths(graph, cycle, ring_edges, budget=budget)
        if not complete:
            return TricyclicFrontier("incomplete", reason=budget[2], explored_states=budget[0],
                                     decompositions=decompositions)
        for bridge in bridges:
            bridge_edges = _path_edges(bridge)
            if bridge_edges & ring_edges:
                continue
            core_vertices = ring_set | set(bridge[1:-1])
            core_edges = ring_edges | bridge_edges
            secondary_edges = edge_set - core_edges
            if not secondary_edges:
                continue
            secondary = _path_from_edges(
                secondary_edges,
                allowed_internal=set(graph) - core_vertices,
                allowed_endpoints=core_vertices,
            )
            if secondary is None:
                continue
            decompositions += 1
            if decompositions > max_decompositions:
                return TricyclicFrontier("incomplete", reason="decomposition cap reached before complete ranking",
                                         explored_states=budget[0], decompositions=decompositions)
            arcs_count = []
            ring_adjacency = {v: set() for v in cycle}
            for edge in ring_edges:
                left, right = tuple(edge)
                ring_adjacency[left].add(right); ring_adjacency[right].add(left)
            start, end = bridge[0], bridge[-1]
            for next_atom in ring_adjacency[start]:
                path = [start, next_atom]
                previous, current = start, next_atom
                while current != end:
                    onward = ring_adjacency[current] - {previous}
                    if len(onward) != 1:
                        break
                    nxt = next(iter(onward)); path.append(nxt)
                    previous, current = current, nxt
                if current == end:
                    arcs_count.append(len(path) - 2)
            if len(arcs_count) != 2:
                continue
            main_segments = tuple(sorted(arcs_count, reverse=True))
            descriptor = (main_segments[0], main_segments[1], len(bridge) - 2,
                          len(secondary) - 2)
            for sequence, endpoint_locants in _numberings(cycle, bridge, secondary):
                locants = tuple((atom, index) for index, atom in enumerate(sequence, 1))
                candidates.append(TricyclicNumbering(
                    sequence=sequence,
                    locants=locants,
                    main_ring=cycle,
                    main_ring_segments=main_segments,
                    main_bridge=bridge,
                    secondary_bridge=secondary,
                    secondary_attachment_locants=endpoint_locants,
                    descriptor_lengths=descriptor,
                ))
                if len(candidates) > max_decompositions:
                    return TricyclicFrontier("incomplete", reason="candidate cap reached before complete ranking",
                                             explored_states=budget[0], decompositions=decompositions)
    if not candidates:
        return TricyclicFrontier("unsupported", reason="no complete main-ring/main-bridge/secondary-bridge decomposition",
                                 explored_states=budget[0], decompositions=decompositions)

    def p23_key(row):
        # P-23.2.1 / .4 / .6.2.1 / .6.2.2 / .6.2.4-.5, in rule order:
        # largest ring, most external atoms in the main bridge, most symmetric
        # ring split, longest independent secondary bridge, lowest ordered
        # secondary endpoints. Atom identifiers never enter this key.
        return (
            -len(row.main_ring),
            -(len(row.main_bridge) - 2),
            abs(row.main_ring_segments[0] - row.main_ring_segments[1]),
            -(len(row.secondary_bridge) - 2),
            row.secondary_attachment_locants,
        )

    best = min(map(p23_key, candidates))
    tied = {row.sequence: row for row in candidates if p23_key(row) == best}
    rows = tuple(tied[sequence] for sequence in sorted(tied, key=lambda seq: tuple(map(_order_key, seq))))
    return TricyclicFrontier("complete", rows, explored_states=budget[0],
                             decompositions=decompositions)
