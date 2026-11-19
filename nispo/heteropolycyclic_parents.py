"""Proved heterobicyclic constructions (P-23.3/P-24.2.4).

Number the carbon topology first. Replacement positions define a new fixed
hydride before suffix, isotope and stereo operations. No legacy renderer,
SSSR count, name-length preference or structural parser selects a numbering.
"""
from itertools import permutations, product

from .constants import MULT, ROOTS

_ELEMENTS = ((8, 'oxa'), (16, 'thia'), (7, 'aza'))
RULES = {
    'skeletal-replacement-von-Baeyer': ('P-51.4.2.2', 'P-23.2.1/P-23.2.3',
                                      'P-23.3.2.1', 'P-23.3.2.2'),
    'skeletal-replacement-spiro': ('P-51.4.2.3', 'P-24.2.1',
                                 'P-24.2.4.1.2(a)', 'P-24.2.4.1.2(b)'),
}


def _theta_orders(graph):
    from .von_baeyer_frontier import bicyclic_theta_frontier

    # P-23 numbers the hydrocarbon topology before the existing replacement
    # comparison below. The caller already proves neutral single-bond valence.
    result = bicyclic_theta_frontier(graph)
    if result.status == "complete":
        for numbering in result.candidates:
            yield numbering.sequence, numbering.bridge_lengths


def _spiro_orders(graph):
    junctions = [i for i in graph if len(graph[i]) == 4]
    if len(junctions) != 1 or any(len(v) not in {2, 4} for v in graph.values()):
        return
    junction = junctions[0]
    paths, consumed = [], set()
    for first in graph[junction]:
        if first in consumed:
            continue
        previous, current, path = junction, first, []
        while current != junction:
            if current in path or current in consumed or len(graph[current]) != 2:
                return
            path.append(current)
            previous, current = current, next(i for i in graph[current] if i != previous)
        if len(path) < 2:
            return
        paths.append(tuple(path)); consumed.update(path)
    if len(paths) != 2 or consumed | {junction} != set(graph):
        return
    for first, second in permutations(paths):
        if len(first) > len(second):
            continue
        for a, b in product((first, first[::-1]), (second, second[::-1])):
            yield (*a, junction, *b), (len(a), len(b))


def skeletons(graph):
    """Yield fixed numbered hydrides of a capped, unannotated complete core."""
    size = graph.GetNumAtoms()
    if size not in ROOTS or any(a.GetAtomicNum() not in {6, 7, 8, 16}
            or a.GetFormalCharge() or a.GetNumRadicalElectrons()
            or a.GetTotalValence() != {6: 4, 7: 3, 8: 2, 16: 2}[a.GetAtomicNum()]
            for a in graph.GetAtoms()):
        return
    numbers = tuple(a.GetAtomicNum() for a in graph.GetAtoms())
    if 6 not in numbers or not any(n != 6 for n in numbers):
        return
    multiple = tuple(b for b in graph.GetBonds() if b.GetBondTypeAsDouble() != 1)
    if multiple and (len(multiple) != 1 or multiple[0].GetBondTypeAsDouble() != 2
                     or any(a.GetAtomicNum() != 6
                            for a in (multiple[0].GetBeginAtom(), multiple[0].GetEndAtom()))):
        return
    adjacency = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()} for a in graph.GetAtoms()}
    if graph.GetNumBonds() - size + 1 != 2:
        return
    options = []
    for method, orders in (('von-Baeyer', () if multiple else _theta_orders(adjacency)),
                           ('spiro', _spiro_orders(adjacency))):
        for seq, descriptor in orders:
            if len(seq) != size or len(set(seq)) != size:
                continue
            by_element = tuple(tuple(j for j, i in enumerate(seq, 1) if numbers[i] == n)
                               for n, _ in _ELEMENTS)
            key = (tuple(sorted(j for locations in by_element for j in locations)), by_element)
            parts = []
            for (_, term), locations in zip(_ELEMENTS, by_element):
                if not locations:
                    continue
                if len(locations) > 1 and len(locations) not in MULT:
                    break
                parts.append(','.join(map(str, locations)) + '-' +
                             (MULT[len(locations)] if len(locations) > 1 else '') + term)
            else:
                parent = '-'.join(parts) + ('bicyclo' if method == 'von-Baeyer' else 'spiro')
                parent += '[' + '.'.join(map(str, descriptor)) + ']' + ROOTS[size] + 'ane'
                inverse = {i: j for j, i in enumerate(seq)}
                edges = tuple((inverse[b.GetBeginAtomIdx()], inverse[b.GetEndAtomIdx()])
                              for b in graph.GetBonds())
                row = (parent, tuple(seq), tuple(map(str, range(1, size + 1))),
                       tuple(numbers[i] for i in seq), edges, not multiple, key,
                       'skeletal-replacement-' + method)
                if multiple:
                    from .localized_bicyclic_states import LocalizedBicyclicState
                    from .unsaturated_ring_parents import UnsaturationState

                    left, right = sorted(inverse[i] + 1 for i in (
                        multiple[0].GetBeginAtomIdx(), multiple[0].GetEndAtomIdx()))
                    state = LocalizedBicyclicState(key[0], tuple(
                        graph.GetAtomWithIdx(i).GetTotalNumHs() for i in seq),
                        UnsaturationState(((left, right, 2),), (left,), (left,), ()),
                        von_baeyer=False)
                    row += (state,)
                options.append(row)
    if options:
        best = min(row[6] for row in options)
        seen = set()
        for row in options:
            if row[6] == best and row[1] not in seen:
                seen.add(row[1]); yield row
