"""Complete bounded P-23 frontiers for theta-shaped bicyclic parents.

This standalone graph helper deliberately covers one narrow construction:
connected, saturated, simple bicyclic theta graphs with exactly two degree-3
bridgeheads, three internally disjoint paths, and degree-2 internal atoms.
It emits every numbering tied under the hydrocarbon descriptor and, where
allowed degree-2 heteroatoms are supplied, P-23.3.2 replacement locants.
External suffix, isotope, stereochemical, and substituent facts remain for the
caller to compare across the returned mapped sequences.

The helper does not import or call the legacy polycyclic candidate generator;
the bounded topology has a direct completeness proof from its degree pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Hashable, Iterable, Mapping


_ELEMENT_PRIORITY = {8: 0, 16: 1, 34: 2, 52: 3, 7: 4, 15: 5, 33: 6,
                     51: 7, 83: 8, 14: 9, 32: 10, 50: 11, 82: 12,
                     5: 13, 13: 14, 31: 15, 49: 16, 81: 17}
_ALLOWED_HETERO = frozenset(_ELEMENT_PRIORITY)


@dataclass(frozen=True, slots=True)
class Numbering:
    """One source-valid numbering, preserving the caller's vertex identifiers."""

    sequence: tuple[Hashable, ...]
    locants: tuple[tuple[Hashable, int], ...]
    bridge_lengths: tuple[int, int, int]
    hetero_locants: tuple[int, ...]
    hetero_element_locants: tuple[tuple[int, tuple[int, ...]], ...]


@dataclass(frozen=True, slots=True)
class Frontier:
    """A complete tied numbering frontier or an explicit non-result."""

    status: str  # "complete", "unsupported", or "incomplete"
    candidates: tuple[Numbering, ...] = ()
    reason: str = ""
    explored_states: int = 0


def _result(status: str, reason: str, explored: int = 0) -> Frontier:
    return Frontier(status=status, reason=reason, explored_states=explored)


def _normalize_graph(adjacency: Mapping[Hashable, Iterable[Hashable]]):
    if not adjacency:
        return None, "empty graph"
    vertices = tuple(adjacency)
    if len(set(vertices)) != len(vertices):
        return None, "duplicate vertex identifiers"
    vertex_set = set(vertices)
    graph = {}
    for vertex in vertices:
        try:
            neighbors = tuple(adjacency[vertex])
            neighbor_set = set(neighbors)
        except (TypeError, KeyError):
            return None, "unhashable or unavailable vertex adjacency"
        if vertex in neighbor_set or len(neighbors) != len(neighbor_set):
            return None, "self-loop or duplicate edge"
        if not neighbor_set <= vertex_set:
            return None, "edge endpoint is absent from graph"
        graph[vertex] = frozenset(neighbor_set)
    for vertex, neighbors in graph.items():
        if any(vertex not in graph[neighbor] for neighbor in neighbors):
            return None, "adjacency must be undirected"
    return graph, ""


def _connected(graph):
    start = next(iter(graph))
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in graph[current] - seen:
            seen.add(neighbor)
            stack.append(neighbor)
    return seen == set(graph)


def _three_theta_paths(graph, bridgeheads, *, max_states):
    """Recover the three paths by deterministic degree-2 walks.

    With exactly two degree-3 vertices, all other vertices degree 2, and
    cycle rank 2, a connected admitted graph is a theta graph. Each neighbor
    of one bridgehead starts exactly one path. State counting is explicit so
    the helper never reports a partial traversal as complete.
    """
    first, second = bridgeheads
    paths = []
    states = 0
    for first_step in graph[first]:
        path = [first, first_step]
        previous, current = first, first_step
        visited = {first, first_step}
        while current != second:
            states += 1
            if states > max_states:
                return None, states, "walk state cap reached"
            if current in bridgeheads or len(graph[current]) != 2:
                return None, states, "internal vertex is not degree 2"
            onward = graph[current] - {previous}
            if len(onward) != 1:
                return None, states, "path continuation is ambiguous"
            next_vertex = next(iter(onward))
            if next_vertex in visited:
                return None, states, "path loops before reaching other bridgehead"
            path.append(next_vertex)
            visited.add(next_vertex)
            previous, current = current, next_vertex
        paths.append(tuple(path))
    paths.sort(key=lambda path: (len(path), tuple(map(repr, path))))
    internals = [set(path[1:-1]) for path in paths]
    if (len(paths) != 3 or any(internals[i] & internals[j]
                               for i in range(3) for j in range(i + 1, 3))
            or any(len(path) < 2 for path in paths)
            or set().union(*internals, set(bridgeheads)) != set(graph)):
        return None, states, "three internally disjoint paths do not cover the graph"
    path_edges = {frozenset((left, right)) for path in paths
                  for left, right in zip(path, path[1:])}
    graph_edges = {frozenset((left, right)) for left, nbrs in graph.items()
                   for right in nbrs}
    if path_edges != graph_edges:
        return None, states, "edges are not exactly the three path edges"
    return tuple(paths), states, ""


def _candidate_sequences(paths):
    """Enumerate all P-23.2.3 orientations and equal-length bridge orders."""
    sequences = set()
    first_head, second_head = paths[0][0], paths[0][-1]
    for start, end in ((first_head, second_head), (second_head, first_head)):
        oriented = tuple(path if path[0] == start else tuple(reversed(path))
                         for path in paths)
        for ordered in permutations(oriented):
            lengths = tuple(len(path) - 2 for path in ordered)
            if lengths != tuple(sorted(lengths, reverse=True)):
                continue
            main_ring_long, main_ring_return, main_bridge = ordered
            sequence = ((start,) + main_ring_long[1:-1] + (end,)
                        + tuple(reversed(main_ring_return[1:-1]))
                        + main_bridge[1:-1])
            if len(sequence) == sum(lengths) + 2 and len(set(sequence)) == len(sequence):
                sequences.add((sequence, lengths))
    return sequences


def bicyclic_theta_frontier(
    adjacency: Mapping[Hashable, Iterable[Hashable]],
    *,
    elements: Mapping[Hashable, int] | None = None,
    max_atoms: int = 64,
    max_states: int = 4096,
    max_candidates: int = 12,
) -> Frontier:
    """Return all tied P-23.2 numberings for a bounded theta bicyclic graph.

    ``adjacency`` is the ring-skeleton graph after removal/capping of external
    components. Vertices are original atom identifiers; output maps those
    identifiers directly to locants. All-carbon input represents the
    saturated hydrocarbon case. ``elements`` may instead label degree-2
    replacement atoms with ordinary single-bond elements supported by
    P-23.3. A direct bridgehead edge is the zero-length bridge of a fused
    bicyclic topology. Degree-3 heteroatoms, charges, bond-order states,
    spiro graphs and other topologies remain outside this helper's contract.

    All equivalent bridgehead orientations and equal-length path orders are
    retained after comparing the ordered P-23.2 descriptor. For replacement
    atoms, the P-23.3.2 heteroatom set locants and then element seniority are
    applied. External substituents/suffixes can break remaining ties in a
    caller, so every such tied mapping is returned.
    """
    graph, error = _normalize_graph(adjacency)
    if error:
        return _result("unsupported", error)
    atom_count = len(graph)
    if atom_count > max_atoms:
        return _result("incomplete", "atom cap reached before topology enumeration")
    if atom_count < 4:
        return _result("unsupported", "simple theta bicyclic cores need at least four vertices")
    if max_states <= 0 or max_candidates <= 0:
        return _result("incomplete", "nonpositive enumeration cap")
    if not _connected(graph):
        return _result("unsupported", "graph is disconnected")
    edge_count = sum(map(len, graph.values())) // 2
    if edge_count - atom_count + 1 != 2:
        return _result("unsupported", "cycle rank is not two")
    degree_three = tuple(vertex for vertex, neighbors in graph.items()
                         if len(neighbors) == 3)
    if (len(degree_three) != 2
            or any(len(neighbors) not in {2, 3} for neighbors in graph.values())):
        return _result("unsupported", "expected two degree-3 bridgeheads and degree-2 internals")

    if elements is None:
        elements = {vertex: 6 for vertex in graph}
    if set(elements) != set(graph):
        return _result("unsupported", "element map must cover every ring vertex exactly")
    if any(number != 6 and (number not in _ALLOWED_HETERO
                             or len(graph[vertex]) != 2)
           for vertex, number in elements.items()):
        return _result("unsupported", "only carbon or ordinary degree-2 replacement atoms are admitted")

    paths, explored, reason = _three_theta_paths(
        graph, degree_three, max_states=max_states
    )
    if reason:
        status = "incomplete" if "cap" in reason else "unsupported"
        return _result(status, reason, explored)
    raw = _candidate_sequences(paths)
    if not raw:
        return _result("unsupported", "no P-23.2.3 numbering was generated", explored)
    if len(raw) > max_candidates:
        return _result("incomplete", "candidate cap reached before tied frontier was complete", explored)

    rows = []
    for sequence, lengths in raw:
        locants = {vertex: number for number, vertex in enumerate(sequence, 1)}
        by_element = {}
        for vertex, element in elements.items():
            if element != 6:
                by_element.setdefault(element, []).append(locants[vertex])
        element_locants = tuple(
            (element, tuple(sorted(values)))
            for element, values in sorted(by_element.items(),
                                          key=lambda item: _ELEMENT_PRIORITY[item[0]])
        )
        rows.append(Numbering(
            sequence=sequence,
            locants=tuple((vertex, locants[vertex]) for vertex in sequence),
            bridge_lengths=lengths,
            hetero_locants=tuple(sorted(loc for values in by_element.values()
                                        for loc in values)),
            hetero_element_locants=element_locants,
        ))

    if any(row.bridge_lengths != min(row.bridge_lengths for row in rows) for row in rows):
        best_descriptor = min(row.bridge_lengths for row in rows)
        rows = [row for row in rows if row.bridge_lengths == best_descriptor]
    if any(row.hetero_locants for row in rows):
        best_set = min(row.hetero_locants for row in rows)
        rows = [row for row in rows if row.hetero_locants == best_set]
        best_elements = min(
            tuple(locs for element, locs in row.hetero_element_locants)
            for row in rows
        )
        rows = [row for row in rows if tuple(locs for _, locs in row.hetero_element_locants)
                == best_elements]
    rows.sort(key=lambda row: tuple(map(repr, row.sequence)))
    if len(rows) > max_candidates:
        return _result("incomplete", "tied frontier exceeds candidate cap", explored)
    return Frontier("complete", tuple(rows), explored_states=explored)

