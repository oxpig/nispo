"""Number connected trees of saturated monocyclic spiro components under P-24.2."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import MULT, ROOTS

MAX_SKELETAL_ATOMS = 60
MIN_RINGS = 3
# Each additional free monocycle contributes at least two new atoms.
MAX_RINGS = (MAX_SKELETAL_ATOMS - 1) // 2
MAX_ORIENTATIONS = 2048
RULE = "P-24.1/P-24.2.0/P-24.2.2/P-24.2.2.1-.2/P-24.2.3"


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


def _cycle_tree(graph):
    """Order each simple cyclic block once and identify its neighboring blocks."""
    blocks = _biconnected_edge_blocks(graph)
    if not MIN_RINGS <= len(blocks) <= MAX_RINGS:
        return None
    cycles, memberships = [], {}
    for index, edges in enumerate(blocks):
        ring = _block_vertices(edges)
        if len(edges) != len(ring) or len(ring) < 3:
            return None
        neighbors = {i: graph[i] & ring for i in ring}
        if any(len(pair) != 2 for pair in neighbors.values()):
            return None
        start = next(iter(ring))
        order = [start]
        previous, current = start, next(iter(neighbors[start]))
        while current != start:
            order.append(current)
            previous, current = current, next(iter(neighbors[current] - {previous}))
        cycles.append(tuple(order))
        for atom in ring:
            memberships.setdefault(atom, []).append(index)
    ports = [{} for _ in cycles]
    for atom, members in memberships.items():
        if len(members) == 1:
            continue
        if len(members) != 2:
            return None
        left, right = members
        ports[left][atom], ports[right][atom] = right, left
    # The edge blocks of a connected graph form a tree at their articulations.
    if sum(map(len, ports)) != 2 * (len(cycles) - 1):
        return None
    return cycles, ports


def _cycle_walk_choices(cycles, ports, root, junction):
    entries = {root: junction}
    pending = [root]
    while pending:
        ring = pending.pop()
        for atom, child in ports[ring].items():
            if child not in entries:
                entries[child] = atom
                pending.append(child)
    choices = []
    for ring, cycle in enumerate(cycles):
        entry = entries[ring]
        i = cycle.index(entry)
        walk = cycle[i+1:] + cycle[:i]
        directions = (walk, walk[::-1])
        if len(ports[ring]) > 1:
            # When first entering a ring, take the shortest path to
            # the next spiro atom. Equal paths remain distinct maps.
            distances = [next(j for j, atom in enumerate(w) if atom in ports[ring])
                         for w in directions]
            directions = tuple(w for w, d in zip(directions, distances)
                               if d == min(distances))
        choices.append(directions)
    return choices


def _number_cycle_tree(cycles, ports):
    ends = [i for i, neighbors in enumerate(ports) if len(neighbors) == 1]
    minimum = min(len(cycles[i]) for i in ends)
    linear = len(ends) == 2
    plans, count = [], 0
    for root in ends:
        if len(cycles[root]) != minimum:
            continue
        junction = next(iter(ports[root]))
        choices = _cycle_walk_choices(cycles, ports, root, junction)
        count += 1 << sum(len(walks) == 2 for walks in choices)
        if count > MAX_ORIENTATIONS:
            return ()  # Never choose among a truncated set of numbering maps.
        plans.append((root, junction, choices))

    best, winners = None, []
    for root, junction, choices in plans:
        for walks in product(*choices):
            sequence = list(walks[root]) + [junction]
            locants = {junction: len(sequence)}
            terms = [PolyspiroTerm(len(walks[root]))]
            sizes = [len(cycles[root])]

            def visit(ring, entry):
                sizes.append(len(cycles[ring]))
                linking = 0
                for atom in walks[ring]:
                    sequence.append(atom)
                    if atom in ports[ring]:
                        locants[atom] = len(sequence)
                        terms.append(PolyspiroTerm(linking))
                        visit(ports[ring][atom], atom)
                        linking = 0
                    else:
                        linking += 1
                terms.append(PolyspiroTerm(linking, locants[entry]))

            visit(ports[root][junction], junction)
            counts = tuple(term.linking_atoms for term in terms)
            spiro_locs = tuple(locants.values())
            # P-24.2.2 compares the initial outward numbers of a linear chain.
            # P-24.2.3 compares junction locants first on a branched skeleton.
            initial = tuple(sorted(counts[:len(cycles)-1])) if linear else ()
            key = (initial, spiro_locs, counts)
            if best is None or key < best:
                best = key
                winners.clear()
            if key == best:
                winners.append(PolyspiroNumbering(
                    tuple(sequence), tuple((a, str(i)) for i, a in enumerate(sequence, 1)),
                    tuple(terms), tuple(locants), spiro_locs, tuple(sizes), initial, key))
    return tuple(sorted(winners, key=lambda row: row.sequence))


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
    """Return every P-24.2-tied whole-parent record for a saturated polyspiro core."""
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
    if (len(graph) not in ROOTS or any(len(v) not in {2, 4} for v in graph.values())
            or not MIN_RINGS <= sum(len(v) == 4 for v in graph.values()) + 1 <= MAX_RINGS):
        return _decline("requires a bounded core of free spiro cycles")
    tree = _cycle_tree(graph)
    if tree is None:
        return _decline("the ring blocks do not form one tree of simple cycles")
    cycles, ports = tree
    if len(cycles) - 1 not in MULT:
        return _decline("the spiro multiplicative prefix is unavailable")
    candidates = _number_cycle_tree(cycles, ports)
    if not candidates:
        return _decline("the complete numbering exceeds the orientation budget")
    return PolyspiroFrontier("complete", "source-ranked polyspiro frontier", candidates)


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
