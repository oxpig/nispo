"""Construct a bounded graph-checked von Baeyer candidate for fused arenes.

This is a general-name fallback, not a proof of preferred fused nomenclature.
It does not call OPSIN. It operates only on one unannotated all-carbon aromatic
ring system, and declines if its bounded structural search is incomplete.
"""
from __future__ import annotations

# The cycle-space enumeration is exponential in cyclomatic rank. Both bounds
# are on work, not molecular identities, and exhausted searches simply decline.
_MAX_RANK = 14
_MAX_STATES = 200_000


class _BudgetExceeded(Exception):
    pass


class _Budget:
    def __init__(self):
        self.count = 0

    def step(self):
        self.count += 1
        if self.count > _MAX_STATES:
            raise _BudgetExceeded


def _graph(mol):
    edges = tuple(sorted(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                         for b in mol.GetBonds()))
    adjacency = [[] for _ in mol.GetAtoms()]
    for index, (a, b) in enumerate(edges):
        adjacency[a].append((b, index))
        adjacency[b].append((a, index))
    return edges, tuple(tuple(sorted(neighbors)) for neighbors in adjacency)


def _basis(edges, adjacency):
    parents = {0: (None, None)}
    queue = [0]
    for vertex in queue:
        for neighbor, edge in adjacency[vertex]:
            if neighbor not in parents:
                parents[neighbor] = (vertex, edge)
                queue.append(neighbor)
    if len(parents) != len(adjacency):
        return None
    tree = {edge for _, edge in parents.values() if edge is not None}
    result = []
    for index, (a, b) in enumerate(edges):
        if index in tree:
            continue
        ancestors, left = {}, 0
        while a is not None:
            ancestors[a] = left
            a, edge = parents[a]
            if edge is not None:
                left ^= 1 << edge
        right = 0
        while b not in ancestors:
            b, edge = parents[b]
            if b is None:
                return None
            right ^= 1 << edge
        result.append(ancestors[b] ^ right ^ (1 << index))
    return tuple(result)


def _longest_cycles(edges, adjacency, budget):
    basis = _basis(edges, adjacency)
    if basis is None or not 4 <= len(basis) <= _MAX_RANK:
        return ()
    best_size, cycles, mask = 0, [], 0
    for index in range(1, 1 << len(basis)):
        budget.step()
        mask ^= basis[(index & -index).bit_length() - 1]
        size = mask.bit_count()
        if size < max(best_size, 3):
            continue
        neighbors = [[] for _ in adjacency]
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            a, b = edges[bit.bit_length() - 1]
            neighbors[a].append(b)
            neighbors[b].append(a)
        if any(len(row) not in (0, 2) for row in neighbors):
            continue
        start = next(i for i, row in enumerate(neighbors) if row)
        path = [start]
        previous, current = start, min(neighbors[start])
        while current != start and len(path) <= size:
            path.append(current)
            a, b = neighbors[current]
            previous, current = current, a if b == previous else b
        if current != start or len(path) != size:
            continue  # Several disjoint cycles, not one main ring.
        if size > best_size:
            best_size, cycles = size, []
        cycles.append(tuple(path))
    return tuple(sorted(cycles))


def _open_paths(adjacency, built, budget, *, include_edges):
    """All paths between built vertices with unbuilt interior vertices."""
    paths = set()
    for start in sorted(built):
        for first, _ in adjacency[start]:
            if first in built:
                if include_edges and start < first:
                    paths.add((start, first))
                continue
            stack = [(first, (start, first), frozenset((start, first)))]
            while stack:
                budget.step()
                node, path, seen = stack.pop()
                for neighbor, _ in adjacency[node]:
                    if neighbor in seen:
                        continue
                    if neighbor in built:
                        if start < neighbor:
                            paths.add((*path, neighbor))
                    else:
                        stack.append((neighbor, (*path, neighbor), seen | {neighbor}))
    return tuple(sorted(paths))


def _main_paths(cycles, adjacency, budget):
    """Select maximum-length main bridges and most balanced main rings."""
    best, choices = None, []
    for cycle in cycles:
        built = frozenset(cycle)
        cycle_edges = {frozenset((a, b)) for a, b in zip(cycle, (*cycle[1:], cycle[0]))}
        for path in _open_paths(adjacency, built, budget, include_edges=True):
            if len(path) == 2 and frozenset(path) in cycle_edges:
                continue
            start, end = cycle.index(path[0]), cycle.index(path[-1])
            distance = abs(start - end)
            balance = min(distance, len(cycle) - distance)
            score = (len(path) - 2, balance)
            if best is None or score > best:
                best, choices = score, []
            if score != best:
                continue
            rotated = cycle[start:] + cycle[:start]
            end_pos = rotated.index(path[-1])
            main = (rotated[:end_pos+1],
                    (rotated[0], *reversed(rotated[end_pos+1:]), rotated[end_pos]),
                    path)
            choices.append(main)
    return tuple(choices)


def _decompositions(adjacency, base_paths, budget):
    initial = frozenset(v for path in base_paths for v in path)
    all_nodes = frozenset(range(len(adjacency)))
    pending = [(initial, ())]
    seen = set()
    while pending:
        budget.step()
        built, paths = pending.pop()
        marker = (built, tuple(sorted(paths)))
        if marker in seen:
            continue
        seen.add(marker)
        if built == all_nodes:
            yield paths
            continue
        for path in _open_paths(adjacency, built, budget, include_edges=False):
            pending.append((built | frozenset(path), (*paths, path)))


def _numbered_candidate(mol, edges, base_paths, base_seq, base_bracket, ears):
    """Build locants, compare all edges, then render one structural name."""
    from ._fast_engine.constants import MULT, ROOTS
    from ._fast_engine.polycyclic_rules import _locant_graph_edges, _opsin_oriented_polybridge_edges

    used = {tuple(sorted((a, b))) for path in (*base_paths, *ears)
            for a, b in zip(path, path[1:])}
    seq = list(base_seq)
    positions = {node: index + 1 for index, node in enumerate(seq)}
    main_atoms = frozenset(seq)
    pending = list(ears)
    while pending:
        available = [path for path in pending if path[0] in positions and path[-1] in positions]
        if not available:
            return None
        available.sort(key=lambda path: (max(positions[path[0]], positions[path[-1]]),
                                         min(positions[path[0]], positions[path[-1]]), len(path)),
                       reverse=True)
        for path in available:
            internal = path[1:-1] if positions[path[0]] > positions[path[-1]] else tuple(reversed(path[1:-1]))
            for atom in internal:
                if atom in positions:
                    return None
                seq.append(atom)
                positions[atom] = len(seq)
            pending.remove(path)
    if len(seq) != mol.GetNumAtoms():
        return None
    independent, dependent = [], []
    for path in (*ears, *(edge for edge in edges if edge not in used)):
        low, high = sorted((positions[path[0]], positions[path[-1]]))
        row = (len(path) - 2, low, high)
        (independent if path[0] in main_atoms and path[-1] in main_atoms else dependent).append(row)
    independent.sort(key=lambda row: (-row[0], row[1], row[2]))
    dependent.sort(key=lambda row: (-row[0], row[1], row[2]))
    descriptors = (*independent, *dependent)
    bracket = (*base_bracket, *(f'{length}^{low},{high}' for length, low, high in descriptors))
    if len(bracket) != len(edges) - len(seq) + 2:
        return None
    reconstructed = _opsin_oriented_polybridge_edges(bracket)
    if not reconstructed or reconstructed != _locant_graph_edges(mol, seq):
        return None
    ring_multiplier = MULT.get(len(bracket) - 1)
    root = ROOTS.get(len(seq))
    unsaturation = []
    for bond in mol.GetBonds():
        order = bond.GetBondTypeAsDouble()
        if order == 2:
            unsaturation.append(tuple(sorted((positions[bond.GetBeginAtomIdx()], positions[bond.GetEndAtomIdx()]))))
        elif order != 1:
            return None
    unsaturation.sort()
    unsaturation_multiplier = MULT.get(len(unsaturation)) if len(unsaturation) > 1 else ''
    if not root or not ring_multiplier or not unsaturation or (len(unsaturation) > 1 and not unsaturation_multiplier):
        return None
    unsaturation_text = ','.join(str(a) if b == a + 1 else f'{a}({b})' for a, b in unsaturation)
    stem = f"{ring_multiplier}cyclo[{'.'.join(map(str, bracket))}]{root}"
    name = (f'{stem}-{unsaturation_text}-ene' if len(unsaturation) == 1
            else f'{stem}a-{unsaturation_text}-{unsaturation_multiplier}ene')
    # P-23.2.6.2: compare independent bridge sizes, dependent count, then
    # superscript locants. Fused-parent and full bond-locant seniority remain
    # unresolved even when this local structural comparison completes.
    independent_lengths = sorted((row[0] for row in independent), reverse=True)
    independent_lengths += [0] * (len(descriptors) - len(independent_lengths))
    locants = tuple(sorted(value for _, low, high in descriptors for value in (low, high)))
    cited_locants = tuple(value for _, low, high in descriptors for value in (low, high))
    score = (tuple(-value for value in independent_lengths), len(dependent), locants,
             cited_locants, tuple(unsaturation), name)
    return score, name


def candidate_name(mol, Chem):
    """Return an unverified general candidate, or an empty string if outside scope."""
    if (mol is None or not 1 <= mol.GetNumAtoms() <= 60 or mol.GetStereoGroups()
            or mol.HasProp('_CXSMILES_Data') or Chem.GetMolSubstanceGroups(mol)
            or len(Chem.GetMolFrags(mol)) != 1
            or any(atom.GetAtomicNum() != 6 or not atom.GetIsAromatic() or atom.GetFormalCharge()
                   or atom.GetIsotope() or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag())
                   or atom.HasQuery() for atom in mol.GetAtoms())
            or any(not bond.GetIsAromatic() or not bond.IsInRing() or bond.HasQuery()
                   or int(bond.GetStereo()) or int(bond.GetBondDir()) for bond in mol.GetBonds())):
        return ''
    try:
        work = Chem.Mol(mol)
        Chem.Kekulize(work, clearAromaticFlags=True)
        work.SetBoolProp('_nispo_kekulized_aromatic_view', True)
        edges, adjacency = _graph(work)
        budget = _Budget()
        cycles = _longest_cycles(edges, adjacency, budget)
        if not cycles:
            return ''
        main_choices = _main_paths(cycles, adjacency, budget)
        from ._fast_engine.polycyclic_rules import _polybridge_numbering_sequences
        candidates = []
        for paths in main_choices:
            numberings = _polybridge_numbering_sequences(paths, alternate_remaining_paths=True)
            for ears in _decompositions(adjacency, paths, budget):
                for seq, bracket in numberings:
                    budget.step()
                    candidate = _numbered_candidate(work, edges, paths, seq, bracket, ears)
                    if candidate:
                        candidates.append(candidate)
        return min(candidates)[1] if candidates else ''
    except (_BudgetExceeded, ValueError, RuntimeError, RecursionError):
        return ''
