"""Bounded P-24.2 frontier for linear polyspiro carbon ring chains.

The complete graph must be a chain of three to five saturated monocyclic
carbon blocks, joined at distinct free spiro atoms. Annulated, bridged,
branched, decorated, and annotated systems are outside this producer.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import MULT, ROOTS

MAX_SKELETAL_ATOMS = 60
MIN_RINGS = 3
MAX_RINGS = 5
RULE = "P-24.1/P-24.2.0/P-24.2.2/P-24.2.2.1-.2"


@dataclass(frozen=True, slots=True)
class PolyspiroTerm:
    linking_atoms: int
    repeated_spiro_locant: int | None = None


@dataclass(frozen=True, slots=True)
class PolyspiroNumbering:
    sequence: tuple[int, ...]
    atom_locants: tuple[tuple[int, str], ...]
    descriptor: tuple[PolyspiroTerm, ...]
    spiro_atoms: tuple[int, ...]
    spiro_locants: tuple[int, ...]
    ring_sizes: tuple[int, ...]
    initial_number_set: tuple[int, ...]
    criteria_key: tuple


@dataclass(frozen=True, slots=True)
class PolyspiroFrontier:
    status: str
    reason: str
    candidates: tuple[PolyspiroNumbering, ...] = ()


def _decline(reason: str) -> PolyspiroFrontier:
    return PolyspiroFrontier("unsupported", reason, ())


def _biconnected_edge_blocks(graph: dict[int, set[int]]):
    discovery = {}
    low = {}
    stack = []
    blocks = []
    tick = 0

    def visit(node, parent):
        nonlocal tick
        tick += 1
        discovery[node] = low[node] = tick
        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            edge = tuple(sorted((node, neighbor)))
            if neighbor not in discovery:
                stack.append(edge)
                visit(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] >= discovery[node]:
                    block = set()
                    while stack:
                        item = stack.pop()
                        block.add(item)
                        if item == edge:
                            break
                    blocks.append(frozenset(block))
            elif discovery[neighbor] < discovery[node]:
                stack.append(edge)
                low[node] = min(low[node], discovery[neighbor])

    for node in graph:
        if node not in discovery:
            visit(node, None)
            if stack:
                blocks.append(frozenset(stack))
                stack.clear()
    return tuple(blocks)


def _block_vertices(edges):
    return frozenset(atom for edge in edges for atom in edge)


def _cycle_order(edges, start, first_neighbor):
    adjacency = {atom: set() for atom in _block_vertices(edges)}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    order = [start]
    previous, current = start, first_neighbor
    while current != start:
        if current in order or len(order) > len(adjacency):
            return ()
        order.append(current)
        next_atoms = adjacency[current] - {previous}
        if len(next_atoms) != 1:
            return ()
        previous, current = current, next(iter(next_atoms))
    if len(order) != len(adjacency):
        return ()
    return tuple(order)


def _cycle_paths(edges, left, right):
    adjacency = {atom: set() for atom in _block_vertices(edges)}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    result = []
    for first in sorted(adjacency[left]):
        path = [left]
        previous, current = left, first
        while current != right:
            if current in path or len(path) > len(adjacency):
                break
            path.append(current)
            choices = adjacency[current] - {previous}
            if len(choices) != 1:
                break
            previous, current = current, next(iter(choices))
        if current == right:
            result.append(tuple(path + [right]))
    return tuple(result)


def _connected(graph):
    seen = set()
    todo = [next(iter(graph))]
    while todo:
        node = todo.pop()
        if node in seen:
            continue
        seen.add(node)
        todo.extend(graph[node] - seen)
    return len(seen) == len(graph)


def _linear_cycle_chain(graph):
    blocks = _biconnected_edge_blocks(graph)
    if not MIN_RINGS <= len(blocks) <= MAX_RINGS:
        return None, "requires three to five monocyclic ring blocks"
    vertices = [_block_vertices(block) for block in blocks]
    if any(len(block) != len(nodes) or len(nodes) < 3
           for block, nodes in zip(blocks, vertices)):
        return None, "each biconnected block must be a simple cycle"
    shared_membership = {}
    for index, nodes in enumerate(vertices):
        for atom in nodes:
            shared_membership.setdefault(atom, []).append(index)
    shared = {atom: tuple(indices) for atom, indices in shared_membership.items()
              if len(indices) > 1}
    if len(shared) != len(blocks) - 1 or any(len(indices) != 2 for indices in shared.values()):
        return None, "rings must share distinct pairwise free spiro atoms"
    block_neighbors = {i: set() for i in range(len(blocks))}
    edge_junction = {}
    for atom, (left, right) in shared.items():
        if len(vertices[left] & vertices[right]) != 1:
            return None, "a ring pair shares more than one atom"
        block_neighbors[left].add(right)
        block_neighbors[right].add(left)
        edge_junction[frozenset((left, right))] = atom
    if any(len(neighbors) > 2 for neighbors in block_neighbors.values()):
        return None, "the ring-block graph branches"
    ends = [index for index, neighbors in block_neighbors.items() if len(neighbors) == 1]
    if len(ends) != 2 or not all(len(n) in {1, 2} for n in block_neighbors.values()):
        return None, "the ring-block graph is not a simple path"

    def path_from(start):
        order = [start]
        previous = None
        current = start
        while True:
            choices = block_neighbors[current] - ({previous} if previous is not None else set())
            if not choices:
                break
            if len(choices) != 1:
                return ()
            previous, current = current, next(iter(choices))
            if current in order:
                return ()
            order.append(current)
        return tuple(order)

    block_path = path_from(ends[0])
    if len(block_path) != len(blocks):
        return None, "the block path does not cover every ring"
    junctions = tuple(edge_junction[frozenset((a, b))]
                      for a, b in zip(block_path, block_path[1:]))
    if len(set(junctions)) != len(junctions):
        return None, "each spiro union must use a distinct atom"
    cycles = {}
    for block_index, junction in zip((block_path[0], block_path[-1]),
                                     (junctions[0], junctions[-1])):
        edges = blocks[block_index]
        cycle_neighbors = sorted(n for n in graph[junction] if n in vertices[block_index])
        cycles[block_index] = tuple(_cycle_order(edges, junction, n)
                                    for n in cycle_neighbors)
    for position, block_index in enumerate(block_path[1:-1], 1):
        left, right = junctions[position - 1], junctions[position]
        paths = _cycle_paths(blocks[block_index], left, right)
        if len(paths) != 2 or not all(paths):
            return None, "an internal ring does not have two junction-to-junction paths"
        cycles[block_index] = paths
    return (blocks, vertices, block_path, junctions, cycles), ""


def _descriptor_text(spiro_count, terms):
    prefix = MULT.get(spiro_count, "") + "spiro"
    if not prefix or prefix == "spiro":
        return ""
    parts = [str(term.linking_atoms) +
             (f"^{term.repeated_spiro_locant}"
              if term.repeated_spiro_locant is not None else "")
             for term in terms]
    return f"{prefix}[{'.'.join(parts)}]"


def polyspiro_parent(mol, Chem=None) -> PolyspiroFrontier:
    """Return every P-24.2-tied whole-parent record for a linear polyspiro core."""
    if mol.GetNumAtoms() > MAX_SKELETAL_ATOMS or mol.GetNumAtoms() < 7:
        return _decline("skeletal size is outside the bounded root vocabulary")
    if mol.GetNumAtoms() != mol.GetNumHeavyAtoms():
        return _decline("explicit hydrogens are outside this complete-core scope")
    if any(atom.GetAtomicNum() != 6 or atom.GetFormalCharge()
           or atom.GetNumRadicalElectrons() or atom.GetIsotope()
           or int(atom.GetChiralTag()) or atom.GetIsAromatic()
           or atom.GetTotalValence() != 4 for atom in iter_atoms(mol)):
        return _decline("only neutral, unannotated saturated carbon atoms are supported")
    if any(bond.GetBondTypeAsDouble() != 1 or bond.GetStereo()
           or int(bond.GetBondDir()) != 0
           for bond in iter_bonds(mol)):
        return _decline("unsaturation or bond stereo is outside P-24.2.2 scope")
    graph = {atom.GetIdx(): {neighbor.GetIdx() for neighbor in atom.GetNeighbors()}
             for atom in iter_atoms(mol)}
    if not graph or not _connected(graph):
        return _decline("the complete spiro skeleton must be connected")
    if sum(map(len, graph.values())) // 2 - len(graph) + 1 != sum(
            len(neighbors) == 4 for neighbors in graph.values()) + 1:
        # A chain of r simple cycles has cycle rank r and r-1 spiro atoms.
        return _decline("the cycle rank does not equal the distinct spiro-union count")
    decomposition, reason = _linear_cycle_chain(graph)
    if decomposition is None:
        return _decline(reason)
    _blocks, vertices, block_path, junctions, cycles = decomposition
    ring_count = len(block_path)
    if len(graph) not in ROOTS:
        return _decline("the total atom count has no supported hydrocarbon root")
    endpoint_sizes = (len(vertices[block_path[0]]), len(vertices[block_path[-1]]))
    starts = (block_path[0], block_path[-1])
    if endpoint_sizes[0] != endpoint_sizes[1]:
        starts = (block_path[0] if endpoint_sizes[0] < endpoint_sizes[1]
                  else block_path[-1],)

    candidates = []
    for start in starts:
        oriented_blocks = block_path if start == block_path[0] else tuple(reversed(block_path))
        oriented_junctions = junctions if start == block_path[0] else tuple(reversed(junctions))
        start_end = oriented_junctions[0]
        end_start = oriented_junctions[-1]
        start_cycles = cycles[oriented_blocks[0]]
        end_cycles = cycles[oriented_blocks[-1]]
        internal_path_options = []
        for pos, block_index in enumerate(oriented_blocks[1:-1], 1):
            paths = _cycle_paths(_blocks[block_index],
                                 oriented_junctions[pos - 1],
                                 oriented_junctions[pos])
            shortest = min(len(path) for path in paths)
            internal_path_options.append(tuple(path for path in paths
                                               if len(path) == shortest))
        for first_cycle, internal_paths, last_cycle in product(
                start_cycles, product(*internal_path_options), end_cycles):
            # A cycle order begins and ends at its spiro atom only once in the
            # resulting sequence: internal ring paths carry both endpoints.
            first_walk = tuple(first_cycle[1:])
            # Reverse terminal walk is a distinct mapped orientation.
            for first_atoms in (first_walk, tuple(reversed(first_walk))):
                for last_atoms in (tuple(last_cycle[1:]), tuple(reversed(last_cycle[1:]))):
                    sequence = list(first_atoms) + [start_end]
                    short_paths = []
                    for path, next_junction in zip(internal_paths, oriented_junctions[1:]):
                        oriented = path if path[0] == sequence[-1] else tuple(reversed(path))
                        short_paths.append(oriented)
                        sequence.extend(oriented[1:])
                    sequence.extend(last_atoms)
                    initial_count = [len(first_atoms)] + [len(path) - 2 for path in short_paths] + [len(last_atoms)]
                    locant_map = {atom: str(number) for number, atom in enumerate(sequence, 1)}
                    terms = [PolyspiroTerm(len(first_atoms))]
                    terms.extend(PolyspiroTerm(len(path) - 2) for path in short_paths)
                    terms.append(PolyspiroTerm(len(last_atoms), int(locant_map[end_start])))
                    # The return traversal adds the longer arc inside every
                    # internal ring, from the last junction back to the first.
                    for path, block_index, left_junction in zip(
                            reversed(short_paths), reversed(oriented_blocks[1:-1]),
                            reversed(oriented_junctions[:-1])):
                        original_paths = _cycle_paths(_blocks[block_index],
                            left_junction, oriented_junctions[oriented_blocks.index(block_index)])
                        long_options = [candidate for candidate in original_paths
                                        if candidate != path and candidate != tuple(reversed(path))]
                        if len(long_options) != 1:
                            sequence = []
                            break
                        long_path = long_options[0]
                        if long_path[0] != path[-1]:
                            long_path = tuple(reversed(long_path))
                        sequence.extend(long_path[1:-1])
                        terms.append(PolyspiroTerm(len(long_path) - 2,
                                                   int(locant_map[left_junction])))
                    if not sequence or len(sequence) != len(graph) or len(set(sequence)) != len(graph):
                        continue
                    locant_map = {atom: str(number) for number, atom in enumerate(sequence, 1)}
                    spiro_locs = tuple(sorted(int(locant_map[atom]) for atom in oriented_junctions))
                    # P-24.2.2.1 minimizes the initial path-number set and
                    # spiro locant set; P-24.2.2.2 compares descriptor numbers
                    # in citation order. Keep every complete tied atom map.
                    # P-24.2.2.1 first compares the initial linking-number
                    # set: for a chain of r rings this is the first r-1
                    # descriptor numbers (the last terminal ring is not in
                    # that set). P-24.2.2.2 then compares the full sequence.
                    initial_set = tuple(sorted(initial_count[:-1]))
                    key = (initial_set, spiro_locs,
                           tuple(term.linking_atoms for term in terms))
                    candidates.append(PolyspiroNumbering(
                        sequence=tuple(sequence),
                        atom_locants=tuple((atom, locant_map[atom]) for atom in sequence),
                        descriptor=tuple(terms), spiro_atoms=oriented_junctions,
                        spiro_locants=spiro_locs,
                        ring_sizes=tuple(len(vertices[index]) for index in oriented_blocks),
                        initial_number_set=initial_set,
                        criteria_key=key))
    if not candidates:
        return _decline("no complete P-24.2 numbering was constructed")
    best = min(row.criteria_key for row in candidates)
    by_sequence = {row.sequence: row for row in candidates if row.criteria_key == best}
    result = []
    for sequence in sorted(by_sequence):
        row = by_sequence[sequence]
        descriptor = _descriptor_text(ring_count - 1, row.descriptor)
        if not descriptor:
            return _decline("the spiro multiplicative prefix is unavailable")
        name = descriptor + ROOTS[len(graph)] + "ane"
        result.append((name, row))
    return PolyspiroFrontier("complete", "source-ranked linear polyspiro frontier",
                             tuple(row for _name, row in result))


def polyspiro_parent_records(mol, Chem=None):
    """Return `(whole parent name, mapped sequence, numbering)` records."""
    frontier = polyspiro_parent(mol, Chem)
    if frontier.status != "complete":
        return frontier
    # Ring count is junction count plus one; descriptor term count is 2r-2.
    ring_count = len(frontier.candidates[0].spiro_atoms) + 1
    records = []
    for candidate in frontier.candidates:
        descriptor = _descriptor_text(ring_count - 1, candidate.descriptor)
        records.append((descriptor + ROOTS[len(candidate.sequence)] + "ane",
                        candidate.sequence, candidate))
    return records
