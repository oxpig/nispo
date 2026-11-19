"""Mapped frontier for ordinary unsaturated monocyclic hydrides.

This is an enumeration layer, not a name selector. It retains every cyclic
numbering tied by skeletal heteroatom locants, leaving P-14.4 suffix and
unsaturation ordering to the consumer operating on the original graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping


from .constants import MULT, ROOTS

_ELEMENT = {8:'oxa',16:'thia',7:'aza'}
_SENIORITY = {8:0,16:1,7:2}
_VALENCE = {6:4,7:3,8:2,16:2}


@dataclass(frozen=True, slots=True)
class UnsaturationState:
    """Typed P-31 locants in one numbered cyclic hydride.

    ``multiple_bond_locants`` and ``double_bond_locants`` are the independent
    comparison sets required by P-31.1.1.1. ``edge_locants`` lists only multiple-bond edges and retains compound
    locant endpoints (including ring-closure ``n(1)`` bonds).
    """
    edge_locants: tuple[tuple[int, int, int], ...]  # (lower, higher, bond order)
    multiple_bond_locants: tuple[int, ...]
    double_bond_locants: tuple[int, ...]
    triple_bond_locants: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class UnsaturatedMonocycleNumbering:
    sequence: tuple[Hashable, ...]
    atom_locants: tuple[tuple[Hashable, int], ...]
    ring_size: int
    parent_component: str
    base_hydride: str
    hetero_locants: tuple[int, ...]
    element_locants: tuple[tuple[int, tuple[int, ...]], ...]
    replacement_terms: tuple[tuple[int, str, tuple[int, ...], bool], ...]
    reference_hydrogens: tuple[int, ...]
    unsaturation: UnsaturationState


@dataclass(frozen=True, slots=True)
class UnsaturatedMonocycleFrontier:
    status: str  # complete / unsupported / incomplete
    candidates: tuple[UnsaturatedMonocycleNumbering, ...] = ()
    reason: str = ""
    explored_states: int = 0


def _decline(reason: str, status: str = 'unsupported', explored: int = 0):
    return UnsaturatedMonocycleFrontier(status, reason=reason, explored_states=explored)


def _walk(graph, start, first):
    seq = [start]
    previous, current = start, first
    while current != start:
        if current in seq or len(graph[current]) != 2:
            return None
        seq.append(current)
        onward = graph[current] - {previous}
        if len(onward) != 1:
            return None
        previous, current = current, next(iter(onward))
    return tuple(seq) if len(seq) == len(graph) else None


def unsaturated_monocycle_frontier(
    adjacency: Mapping[Hashable, Iterable[Hashable]],
    elements: Mapping[Hashable, int],
    bond_orders: Mapping[frozenset, int],
    *, max_atoms: int = 60, max_candidates: int = 128,
) -> UnsaturatedMonocycleFrontier:
    """Enumerate a complete mapped numbering frontier for a single ring.

    Inputs contain only the cyclic skeletal core. The caller must separately
    validate formal charges, aromatic annotations, isotope/stereo decorations,
    and valence against the full original molecule. Only ordinary integer
    single/double/triple C-C ring bonds are admitted; cumulative systems and
    unsaturation involving a heteroatom are explicitly outside this operation.
    Heterocycles are admitted only at 11–60 atoms under skeletal-replacement
    nomenclature (P-22.2.3); smaller Hantzsch–Widman rings use another route.
    """
    if not adjacency or set(adjacency) != set(elements):
        return _decline('element map must cover a nonempty core exactly')
    vertices = tuple(adjacency)
    try:
        graph = {v: frozenset(adjacency[v]) for v in vertices}
    except (TypeError, KeyError):
        return _decline('invalid adjacency mapping')
    if any(v in ns or len(ns) != len(tuple(adjacency[v])) for v, ns in graph.items()):
        return _decline('self-loop or duplicate edge')
    if any(not ns <= set(vertices) for ns in graph.values()):
        return _decline('edge endpoint missing from core')
    if any(v not in graph[u] for u, ns in graph.items() for v in ns):
        return _decline('adjacency must be undirected')
    n = len(graph)
    if n > max_atoms or n > 60:
        return _decline('ring atom limit is 60', 'incomplete')
    if n < 3 or n not in ROOTS:
        return _decline('ring size is outside the supported root inventory')
    if any(len(ns) != 2 for ns in graph.values()) or sum(map(len, graph.values())) != 2*n:
        return _decline('core must be a simple cycle')
    seen, todo = {vertices[0]}, [vertices[0]]
    while todo:
        u = todo.pop()
        for v in graph[u] - seen:
            seen.add(v); todo.append(v)
    if len(seen) != n:
        return _decline('core must be one connected cycle')
    if any(elements[v] not in _VALENCE for v in vertices):
        return _decline('only C/N/O/S skeletal atoms are supported')
    hetero = {v: elements[v] for v in vertices if elements[v] != 6}
    if hetero and (n < 11 or 6 not in elements.values()):
        return _decline('hetero skeletal replacement scope is C/N/O/S rings of 11–60 members')
    normalized = {}
    for key, order in bond_orders.items():
        if not isinstance(key, frozenset) or len(key) != 2 or not key <= set(vertices):
            return _decline('bond-order map must use two-atom core edge keys')
        if not isinstance(order, int) or order not in (1, 2, 3):
            return _decline('only integer single, double, and triple bonds are supported')
        normalized[key] = order
    edges = {frozenset((u, v)) for u in graph for v in graph[u]}
    if len(normalized) != n or set(normalized) != edges:
        return _decline('bond-order map must cover each ring edge exactly once')
    if not any(order > 1 for order in normalized.values()):
        return _decline('this frontier is only for unsaturated cycles')
    if any(order > 1 and (elements[u] != 6 or elements[v] != 6)
           for edge, order in normalized.items() if order > 1 for u, v in (tuple(edge),)):
        return _decline('multiple bonds involving heteroatoms are not covered')
    incident = {v: 0 for v in vertices}
    for edge, order in normalized.items():
        if order > 1:
            for v in edge: incident[v] += 1
    if any(count > 1 for count in incident.values()):
        return _decline('cumulative or adjacent multiple-bond systems are not covered')
    if max_candidates < 2*n:
        return _decline('candidate cap below complete cyclic orientation frontier', 'incomplete')

    raw = set()
    for start in vertices:
        for first in graph[start]:
            seq = _walk(graph, start, first)
            if seq is None:
                return _decline('failed to enumerate a full cyclic traversal')
            raw.add(seq)
    if len(raw) != 2*n:
        return _decline('cyclic walk enumeration was not complete')
    if hetero:
        senior_element = min(hetero.values(), key=lambda e: _SENIORITY[e])
        raw = {seq for seq in raw if elements[seq[0]] == senior_element}
        rows = []
        for seq in raw:
            loc = {v: i for i, v in enumerate(seq, 1)}
            by_element = {e: tuple(sorted(loc[v] for v in hetero if hetero[v] == e))
                          for e in _SENIORITY if e in hetero.values()}
            rows.append((seq, tuple(sorted(x for vals in by_element.values() for x in vals)), by_element))
        if not rows:
            return _decline('no traversal begins at senior heteroatom')
        low_all = min(row[1] for row in rows)
        rows = [row for row in rows if row[1] == low_all]
        low_by_element = min(tuple(row[2][e] for e in _SENIORITY if e in row[2]) for row in rows)
        raw = {row[0] for row in rows if tuple(row[2][e] for e in _SENIORITY if e in row[2]) == low_by_element}

    results = []
    for seq in raw:
        loc = {v: i for i, v in enumerate(seq, 1)}
        by_element = {e: tuple(sorted(loc[v] for v in hetero if hetero[v] == e))
                      for e in _SENIORITY if e in hetero.values()}
        hlocs = tuple(sorted(x for vals in by_element.values() for x in vals))
        terms = []
        typed_terms = []
        for element, positions in by_element.items():
            count = len(positions)
            mult = '' if count == 1 else MULT.get(count)
            if mult is None:
                return _decline('heteroatom multiplicity exceeds independent prefix inventory')
            # This frontier always represents unsaturation, so preserve the
            # locator where the source unsaturated example requires it. A
            # downstream context-aware renderer may omit only when allowed.
            omit = False
            term = mult+_ELEMENT[element]
            terms.append((positions, term, omit))
            typed_terms.append((element, term, positions, omit))
        prefix = '-'.join(','.join(map(str, positions))+'-'+term
                          for positions, term, omit in terms)
        hydride = 'cyclo'+ROOTS[n]+'ane'
        parent = prefix+hydride
        local_edges = []
        for edge, order in normalized.items():
            u, v = tuple(edge)
            a, b = sorted((loc[u], loc[v]))
            local_edges.append((a, b, order))
        local_edges.sort()
        multiple = tuple(sorted(a for a, b, order in local_edges if order > 1))
        doubles = tuple(sorted(a for a, b, order in local_edges if order == 2))
        triples = tuple(sorted(a for a, b, order in local_edges if order == 3))
        # Hydrogens referenced to the saturated corresponding monocycle,
        # computed from the actual weighted bonds, not a saturated flag.
        reference = tuple(_VALENCE[elements[v]]-
                          sum(normalized[frozenset((v, w))] for w in graph[v])
                          for v in seq)
        if any(h < 0 for h in reference):
            return _decline('weighted ring bonds exceed ordinary reference valence')
        results.append(UnsaturatedMonocycleNumbering(
            sequence=seq, atom_locants=tuple((v, loc[v]) for v in seq), ring_size=n,
            parent_component=parent, base_hydride=hydride, hetero_locants=hlocs,
            element_locants=tuple((e, by_element[e]) for e in _SENIORITY if e in by_element),
            replacement_terms=tuple(typed_terms),
            reference_hydrogens=reference,
            unsaturation=UnsaturationState(tuple(row for row in local_edges if row[2] > 1), multiple, doubles, triples)))
    results.sort(key=lambda row: tuple(map(repr, row.sequence)))
    return UnsaturatedMonocycleFrontier('complete', tuple(results), explored_states=2*n)


def _locant_text(a: int, b: int) -> str:
    """Format a ring-bond locant under P-31.1.1.1."""
    return str(a) if b-a == 1 else f"{a}({b})"


def render_unsaturated_parent(
    parent_component: str,
    state: UnsaturationState,
    *, omit_single_locant: bool = False,
) -> str:
    """Render the P-31 ene/yne component for one numbered cyclic parent.

    ``parent_component`` is the saturated hydride component ending in ``ane``
    (including any explicit replacement prefixes). The caller decides whether
    a sole locant 1 may be omitted; this routine accepts that decision only
    for a single ordinary bond at locant 1. It does not select a numbering.
    """
    if not isinstance(parent_component, str) or not parent_component.endswith('ane'):
        raise ValueError('parent component must end in ane')
    edges = tuple(state.edge_locants)
    if not edges or any(len(edge) != 3 for edge in edges):
        raise ValueError('unsaturation descriptor must contain numbered multiple-bond edges')
    if any(not isinstance(a, int) or not isinstance(b, int) or not isinstance(order, int)
           or a < 1 or b <= a or order not in (2, 3) for a, b, order in edges):
        raise ValueError('invalid numbered double/triple bond edge')
    if len({(a, b) for a, b, _ in edges}) != len(edges):
        raise ValueError('duplicate multiple-bond edge')
    if len({atom for a, b, _ in edges for atom in (a, b)}) != 2*len(edges):
        raise ValueError('cumulative or adjacent multiple bonds are outside this descriptor scope')
    doubles = tuple(sorted(a for a, b, order in edges if order == 2))
    triples = tuple(sorted(a for a, b, order in edges if order == 3))
    multiple = tuple(sorted(a for a, b, _ in edges))
    if (multiple != tuple(state.multiple_bond_locants)
            or doubles != tuple(state.double_bond_locants)
            or triples != tuple(state.triple_bond_locants)):
        raise ValueError('typed locant sets do not agree with the mapped bond edges')
    if not doubles and not triples:
        raise ValueError('at least one double or triple bond is required')

    sole = len(edges) == 1
    if omit_single_locant and not (sole and multiple == (1,) and edges[0][1]-edges[0][0] == 1):
        raise ValueError('a locant may be omitted only for one ordinary bond at locant 1')

    # P-31.1.1.2 inserts a connecting ``a`` before locants when a multiplying
    # prefix for either unsaturation ending is present.
    stem = parent_component[:-3]
    if len(doubles) > 1 or len(triples) > 1:
        stem += 'a'

    def locants(values):
        lookup = {a: b for a, b, order in edges}
        return ','.join(_locant_text(a, lookup[a]) for a in values)

    if sole and omit_single_locant:
        return stem + ('ene' if doubles else 'yne')
    if doubles and triples:
        dmult = '' if len(doubles) == 1 else MULT.get(len(doubles))
        ymult = '' if len(triples) == 1 else MULT.get(len(triples))
        if dmult is None or ymult is None:
            raise ValueError('unsaturation multiplicity is outside the supported prefix table')
        return (stem + '-' + locants(doubles) + '-' + dmult + 'en-'
                + locants(triples) + '-' + ymult + 'yne')
    if doubles:
        mult = '' if len(doubles) == 1 else MULT.get(len(doubles))
        if mult is None:
            raise ValueError('double-bond multiplicity is outside the supported prefix table')
        return stem + '-' + locants(doubles) + '-' + mult + 'ene'
    mult = '' if len(triples) == 1 else MULT.get(len(triples))
    if mult is None:
        raise ValueError('triple-bond multiplicity is outside the supported prefix table')
    return stem + '-' + locants(triples) + '-' + mult + 'yne'
