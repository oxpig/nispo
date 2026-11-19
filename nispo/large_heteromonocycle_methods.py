"""Mapped P-22.2.3 records for saturated large heteromonocycles.

The API accepts only a ring-skeleton graph already isolated and valence-checked
by its caller. It enumerates every rotation and direction, applies the P-22.2.3
heteroatom locant criteria, and maps locants back to the caller's original atom
identifiers. Suffixes, branches, isotopes, and stereochemistry remain on the
original graph for downstream mapped rendering and P-14 comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

from .constants import MULT, ROOTS


_TERMS = {8: "oxa", 16: "thia", 34: "selena", 52: "tellura", 7: "aza",
          15: "phospha", 33: "arsa", 51: "stiba", 83: "bismutha",
          14: "sila", 32: "germa", 50: "stanna", 82: "plumba", 5: "bora"}
_SENIORITY = {element: rank for rank, element in enumerate(_TERMS)}  # P-22.2.3.1


def _numerical_term(number):
    """IUPAC NT-1 simple numerical terms through 9999."""
    if not 1 <= number <= 9999:
        return ''
    if number == 1:
        return 'mono'
    if number in MULT:
        return MULT[number]
    units = {1: 'hen', 2: 'do', 3: 'tri', 4: 'tetra', 5: 'penta',
             6: 'hexa', 7: 'hepta', 8: 'octa', 9: 'nona'}
    tens = {6: 'hexaconta', 7: 'heptaconta', 8: 'octaconta', 9: 'nonaconta'}
    hundreds = {1: 'hecta', 2: 'dicta', 3: 'tricta', 4: 'tetracta', 5: 'pentacta',
                6: 'hexacta', 7: 'heptacta', 8: 'octacta', 9: 'nonacta'}
    thousands = {1: 'kilia', 2: 'dilia', 3: 'trilia', 4: 'tetralia', 5: 'pentalia',
                 6: 'hexalia', 7: 'heptalia', 8: 'octalia', 9: 'nonalia'}
    def below_hundred(n):
        if not n:
            return ''
        if n < 10:
            return units[n]
        if n in MULT:
            return MULT[n]
        return units.get(n % 10, '') + tens[n // 10]
    remainder = number % 1000
    lower = below_hundred(remainder % 100)
    if remainder >= 100:
        lower += hundreds[remainder // 100]
    if number >= 1000:
        lower += thousands[number // 1000]
    return lower


def _cycloalkane_root(number):
    if number in ROOTS:
        return ROOTS[number]
    term = _numerical_term(number)
    return term[:-1] if term.endswith('a') else ''


@dataclass(frozen=True, slots=True)
class ReplacementTerm:
    atomic_number: int
    term: str
    locants: tuple[int, ...]
    omit_locant: bool


@dataclass(frozen=True, slots=True)
class LargeHeteromonocycleNumbering:
    sequence: tuple[Hashable, ...]
    atom_locants: tuple[tuple[Hashable, str], ...]
    ring_size: int
    base_hydride: str
    hetero_locants: tuple[int, ...]
    element_locants: tuple[tuple[int, tuple[int, ...]], ...]
    replacement_terms: tuple[ReplacementTerm, ...]


@dataclass(frozen=True, slots=True)
class LargeHeteromonocycleFrontier:
    status: str  # complete / unsupported / incomplete
    candidates: tuple[LargeHeteromonocycleNumbering, ...] = ()
    reason: str = ""
    explored_states: int = 0


def _failure(status: str, reason: str, explored: int = 0):
    return LargeHeteromonocycleFrontier(status, reason=reason, explored_states=explored)


def _normalize(adjacency: Mapping[Hashable, Iterable[Hashable]]):
    if not adjacency:
        return None, "empty graph"
    vertices = tuple(adjacency)
    try:
        vertex_set = set(vertices)
    except TypeError:
        return None, "vertex identifiers must be hashable"
    graph = {}
    for vertex in vertices:
        try:
            values = tuple(adjacency[vertex])
            neighbors = frozenset(values)
        except (TypeError, KeyError):
            return None, "adjacency contains an invalid vertex or neighbor"
        if vertex in neighbors or len(neighbors) != len(values):
            return None, "self-loop or duplicate edge"
        if not neighbors <= vertex_set:
            return None, "edge endpoint is absent from graph"
        graph[vertex] = neighbors
    if any(vertex not in graph[neighbor]
           for vertex, neighbors in graph.items() for neighbor in neighbors):
        return None, "adjacency must be undirected"
    return graph, ""


def _cycle_sequence(graph, start, first_neighbor, *, max_states):
    sequence = [start, first_neighbor]
    previous, current = start, first_neighbor
    states = 0
    while current != start:
        states += 1
        if states > max_states:
            return None, states, "walk-state cap reached"
        if len(graph[current]) != 2:
            return None, states, "ring vertices must have exactly two skeletal neighbors"
        onward = graph[current] - {previous}
        if len(onward) != 1:
            return None, states, "cycle continuation is ambiguous"
        following = next(iter(onward))
        if following in sequence and following != start:
            return None, states, "walk closes before visiting all vertices"
        if following != start:
            sequence.append(following)
        previous, current = current, following
    if len(sequence) != len(graph) or len(set(sequence)) != len(graph):
        return None, states, "cycle does not cover every supplied vertex"
    return tuple(sequence), states, ""


def large_heteromonocycle_frontier(
    adjacency: Mapping[Hashable, Iterable[Hashable]],
    elements: Mapping[Hashable, int],
    *,
    max_atoms: int = 60,
    max_states: int = 10000,
    max_candidates: int = 128,
) -> LargeHeteromonocycleFrontier:
    """Return every numbering tied under the P-22.2.3 heteroatom criteria.

    Admitted graphs are single connected cycles of at least 11 atoms,
    containing carbon and at least one supported heteroatom. The caller
    must ensure the core is saturated, all ring bonds are single, and all
    formal charges/bonding numbers are ordinary. Branch atoms are excluded
    from ``adjacency``; retaining original vertex IDs lets the caller map
    them, suffix sites, isotope labels, and CIP data against the full molecule.

    The default atom budget is 60; a caller may raise it up to the source
    numerical-term limit of 9999 and provide matching state/candidate budgets.
    The maximum unfiltered numbering frontier is ``2 * ring_size``. Hitting a
    configured cap returns an empty ``incomplete`` result so an arbitrary
    start atom or direction cannot become the selected numbering.
    """
    graph, error = _normalize(adjacency)
    if error:
        return _failure("unsupported", error)
    size = len(graph)
    if size < 11:
        return _failure("unsupported", "P-22.2.3 scope begins at eleven ring members")
    if size > max_atoms:
        return _failure("incomplete", "atom cap reached before numbering enumeration")
    root = _cycloalkane_root(size)
    if not root:
        return _failure("unsupported", "no IUPAC numerical term for this ring size")
    if max_states <= 0 or max_candidates <= 0:
        return _failure("incomplete", "nonpositive enumeration cap")
    if set(elements) != set(graph):
        return _failure("unsupported", "element map must cover the ring core exactly")
    if any(number not in {6, *_TERMS} for number in elements.values()):
        return _failure("unsupported", "unsupported skeletal element")
    if not any(number != 6 for number in elements.values()) or 6 not in elements.values():
        return _failure("unsupported", "requires carbon and at least one heteroatom")
    if any(len(neighbors) != 2 for neighbors in graph.values()):
        return _failure("unsupported", "each supplied skeletal atom must have ring degree two")
    if sum(map(len, graph.values())) // 2 != size:
        return _failure("unsupported", "cycle edge count is inconsistent")
    start = next(iter(graph))
    seen, stack = {start}, [start]
    while stack:
        current = stack.pop()
        for neighbor in graph[current] - seen:
            seen.add(neighbor)
            stack.append(neighbor)
    if seen != set(graph):
        return _failure("unsupported", "ring graph is disconnected")
    if max_candidates < 2 * size:
        return _failure("incomplete", "candidate cap is below the complete rotation frontier")
    if max_states < 2 * size:
        return _failure("incomplete", "state cap is below the complete rotation frontier")

    highest = min(_SENIORITY[e] for e in elements.values() if e != 6)
    highest_element = next(e for e, rank in _SENIORITY.items() if rank == highest)
    raw = []
    explored = 0
    for vertex in graph:
        if elements[vertex] != highest_element:
            continue
        for neighbor in graph[vertex]:
            sequence, states, reason = _cycle_sequence(
                graph, vertex, neighbor, max_states=max_states - explored
            )
            explored += states
            if reason:
                return _failure("incomplete" if "cap" in reason else "unsupported",
                                reason, explored)
            raw.append(sequence)
    expected = 2 * sum(element == highest_element for element in elements.values())
    if len(raw) != expected or len(set(raw)) != expected:
        return _failure("unsupported", "did not enumerate each senior-heteroatom start and direction", explored)
    rows = []
    for sequence in raw:
        if elements[sequence[0]] != highest_element:
            continue
        locants = {atom: i for i, atom in enumerate(sequence, 1)}
        by_element = {}
        for atom, element in elements.items():
            if element != 6:
                by_element.setdefault(element, []).append(locants[atom])
        hetero_set = tuple(sorted(loc for values in by_element.values() for loc in values))
        element_sets = tuple((element, tuple(sorted(by_element[element])))
                             for element in sorted(by_element, key=_SENIORITY.__getitem__))
        rows.append((sequence, hetero_set, element_sets))
    if not rows:
        return _failure("unsupported", "no numbering assigns locant 1 to the senior heteroatom", explored)
    best_set = min(row[1] for row in rows)
    rows = [row for row in rows if row[1] == best_set]
    best_elements = min(tuple(locants for _, locants in row[2]) for row in rows)
    rows = [row for row in rows
            if tuple(locants for _, locants in row[2]) == best_elements]

    results = []
    for sequence, hetero_set, element_sets in rows:
        atom_locants = tuple((atom, str(i)) for i, atom in enumerate(sequence, 1))
        terms = []
        for element, locants in element_sets:
            multiplier = _numerical_term(len(locants)) if len(locants) > 1 else ""
            omit_single_hetero_locant = len(hetero_set) == 1 and locants == (1,)
            terms.append(ReplacementTerm(
                atomic_number=element,
                term=multiplier + _TERMS[element],
                locants=() if omit_single_hetero_locant else locants,
                omit_locant=omit_single_hetero_locant,
            ))
        results.append(LargeHeteromonocycleNumbering(
            sequence=sequence,
            atom_locants=atom_locants,
            ring_size=size,
            base_hydride="cyclo" + root + "ane",
            hetero_locants=hetero_set,
            element_locants=element_sets,
            replacement_terms=tuple(terms),
        ))
    results.sort(key=lambda row: tuple(map(repr, row.sequence)))
    return LargeHeteromonocycleFrontier("complete", tuple(results), explored_states=explored)

