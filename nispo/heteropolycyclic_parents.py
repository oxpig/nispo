"""Proved heterobicyclic constructions (P-23.3/P-24.2.4).

Number the carbon topology first. Replacement positions define a new fixed
hydride before suffix, isotope and stereo operations. No legacy renderer,
SSSR count, name-length preference or structural parser selects a numbering.
"""
from itertools import permutations, product

from .constants import MULT, ROOTS, STANDARD_REPLACEMENTS, STANDARD_REPLACEMENT_VALENCE

RULES = {
    'skeletal-replacement-von-Baeyer': ('P-51.4.2.2', 'P-23.2.1/P-23.2.3',
                                      'P-23.3.2.1', 'P-23.3.2.2'),
    'skeletal-replacement-spiro': ('P-51.4.2.3', 'P-24.2.1',
                                 'P-24.2.4.1.2(a)', 'P-24.2.4.1.2(b)'),
}


def replacement_locants(sequence, numbers):
    """P-24.2.4.1.2 first compares all replacement locants, then each element."""
    positions = {}
    for loc, atom in enumerate(sequence, 1):
        z = numbers[atom]
        if z != 6:
            positions.setdefault(z, []).append(loc)
    ordered = tuple(tuple(positions[z]) for z, _, _ in STANDARD_REPLACEMENTS if z in positions)
    return tuple(sorted(n for sites in ordered for n in sites)), ordered


def replacement_prefix(numbers, ordered):
    elements = set(numbers)
    terms = (term for z, term, _ in STANDARD_REPLACEMENTS if z in elements)
    parts = []
    for term, locants in zip(terms, ordered, strict=True):
        multiplier = '' if len(locants) == 1 else MULT.get(len(locants))
        if multiplier is None:
            return None
        parts.append(','.join(map(str, locants)) + '-' + multiplier + term)
    return '-'.join(parts)


def free_spiro_core(rings):
    """Return the atoms of a connected tree of cycles sharing single, distinct atoms."""
    if len(rings) < 2:
        return frozenset()
    union, junctions = set(), set()
    vertices = [set(r) for r in rings]
    for i, ring in enumerate(vertices):
        for other in vertices[:i]:
            shared = ring & other
            if len(shared) > 1 or shared & junctions:
                return frozenset()
            junctions.update(shared)
        union.update(ring)
    if len(junctions) != len(rings) - 1:
        return frozenset()
    pending = [vertices[0]]
    unseen = vertices[1:]
    while pending:
        block = pending.pop()
        adjacent = [v for v in unseen if v & block]
        pending.extend(adjacent)
        unseen = [v for v in unseen if not v & block]
    return frozenset() if unseen else frozenset(union)


def bare_spiro_name(mol, Chem, rings):
    """Select an unannotated saturated spiro parent with no external branches."""
    if (sum(map(len, rings)) != mol.GetNumAtoms() + len(rings) - 1
            or len(free_spiro_core(rings)) != mol.GetNumAtoms()
            or mol.GetStereoGroups()
            or any(a.GetIsotope() or a.GetChiralTag() for a in mol.GetAtoms())
            or any(b.GetBondTypeAsDouble() != 1 or b.GetStereo() for b in mol.GetBonds())):
        return ''
    if len(rings) == 2:
        row = next(skeletons(mol), None)
        return row[0] if row else ''
    if all(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        from .polyspiro_methods import polyspiro_decision
        decision = polyspiro_decision(mol, Chem)
    else:
        from .heteropolyspiro_methods import heteropolyspiro_decision
        decision = heteropolyspiro_decision(mol, Chem)
    return decision.name if decision else ''


def substituted_spiro_name(mol, Chem, rings):
    """Decorate a proved spiro core after checking every principal site on the original graph."""
    from .mapped_ring_parents import records
    from .ring_state_parent import _inventory

    core = free_spiro_core(rings)
    if not core:
        return ''
    rows = records(mol, Chem, core_filter=core, general_branches=True)
    if len(rows) != 1 or rows[0].principal_centers != {g.center for g in _inventory(mol, Chem).principal}:
        return ''
    return rows[0].name


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
    if size not in ROOTS or any(a.GetAtomicNum() not in STANDARD_REPLACEMENT_VALENCE
            or a.GetFormalCharge() or a.GetNumRadicalElectrons()
            or a.GetTotalValence() != STANDARD_REPLACEMENT_VALENCE[a.GetAtomicNum()]
            for a in graph.GetAtoms()):
        return
    numbers = tuple(a.GetAtomicNum() for a in graph.GetAtoms())
    if 6 not in numbers or not any(n != 6 for n in numbers):
        return
    rank = graph.GetNumBonds() - size + 1
    if rank not in {2, 3}:
        return
    multiple = tuple(b for b in graph.GetBonds() if b.GetBondTypeAsDouble() != 1)
    if (rank == 2 and multiple and (len(multiple) != 1 or multiple[0].GetBondTypeAsDouble() != 2)
            or any(b.GetBondTypeAsDouble() not in (2, 3) or b.GetBeginAtom().GetAtomicNum() != 6
                   or b.GetEndAtom().GetAtomicNum() != 6 for b in multiple)):
        return
    adjacency = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()} for a in graph.GetAtoms()}
    if rank == 3:
        from .tricyclic_von_baeyer_frontier import tricyclic_von_baeyer_frontier
        frontier = tricyclic_von_baeyer_frontier(adjacency)
        if frontier.status != 'complete':
            return
        frames = ((n.sequence, n.bridge_descriptor, 'von-Baeyer') for n in frontier.candidates)
    else:
        theta = not multiple and max(map(len, adjacency.values())) == 3 and set(numbers) <= {6, 7, 8, 16}
        orders = _theta_orders(adjacency) if theta else ()
        frames = ((seq, stem + '[' + '.'.join(map(str, lengths)) + ']', method)
                  for method, stem, variants in (('von-Baeyer', 'bicyclo', orders), ('spiro', 'spiro', _spiro_orders(adjacency)))
                  for seq, lengths in variants)
    options = []
    for seq, descriptor, method in frames:
        if len(seq) != size or len(set(seq)) != size:
            continue
        key = replacement_locants(seq, numbers)
        prefix = replacement_prefix(numbers, key[1])
        if prefix is None:
            continue
        parent = prefix + descriptor + ROOTS[size] + 'ane'
        inverse = {i: j for j, i in enumerate(seq)}
        edges = tuple((inverse[b.GetBeginAtomIdx()], inverse[b.GetEndAtomIdx()])
                      for b in graph.GetBonds())
        row = (parent, tuple(seq), tuple(map(str, range(1, size + 1))),
               tuple(numbers[i] for i in seq), edges, not multiple, key,
               'skeletal-replacement-' + method)
        if multiple:
            from .localized_bicyclic_states import LocalizedBicyclicState
            from .unsaturated_ring_parents import UnsaturationState

            bonds = tuple(sorted((*sorted((inverse[b.GetBeginAtomIdx()] + 1, inverse[b.GetEndAtomIdx()] + 1)),
                                  int(b.GetBondTypeAsDouble())) for b in multiple))
            state = LocalizedBicyclicState(key[0], tuple(
                graph.GetAtomWithIdx(i).GetTotalNumHs() for i in seq),
                UnsaturationState(bonds, tuple(a for a, _, _ in bonds),
                                  tuple(a for a, _, order in bonds if order == 2),
                                  tuple(a for a, _, order in bonds if order == 3)),
                von_baeyer=method == 'von-Baeyer')
            row += (state,)
        options.append(row)
    if options:
        best = min(row[6] for row in options)
        seen = set()
        for row in options:
            if row[6] == best and row[1] not in seen:
                seen.add(row[1]); yield row
