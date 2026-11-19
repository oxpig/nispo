"""Original-atom numbering frontier for linear saturated carbon dispiro cores.

This is a graph-only P-24.2.2 producer. It does not select from rendered
names, legacy candidate order, or name length.
"""

from dataclasses import dataclass

from .constants import ROOTS

RULE = "P-24.1/P-24.2.0/P-24.2.2/P-24.2.2.1-.2/P-14/P-44"
MAX_SKELETAL_ATOMS = 60


@dataclass(frozen=True, slots=True)
class DescriptorTerm:
    linking_atoms: int
    repeated_spiro_locant: int | None = None


@dataclass(frozen=True, slots=True)
class DispiroSkeleton:
    sequence: tuple[int, ...]
    locants: tuple[tuple[int, str], ...]
    descriptor: tuple[DescriptorTerm, ...]
    spiro_atoms: tuple[int, int]
    spiro_locants: tuple[int, int]
    ring_sizes: tuple[int, int, int]
    hydrocarbon_parent: str
    edges: tuple[tuple[int, int], ...]
    criteria_key: tuple
    rule: str = RULE


@dataclass(frozen=True, slots=True)
class DispiroFrontier:
    status: str
    reason: str
    candidates: tuple[DispiroSkeleton, ...] = ()


def _decline(reason):
    return DispiroFrontier("unsupported", reason, ())


def _simple_paths(adjacency, start, end, *, limit=4):
    paths = []
    stack = [(start, (start,))]
    while stack:
        current, path = stack.pop()
        if current == end:
            paths.append(path)
            if len(paths) > limit:
                return ()
            continue
        for neighbor in sorted(adjacency[current] - set(path), reverse=True):
            stack.append((neighbor, path + (neighbor,)))
    return tuple(paths)


def _terminal_path(adjacency, junction, first, last, *, junctions):
    path = [first]
    previous, current = junction, first
    while current != last:
        if current in junctions or len(path) > len(adjacency):
            return ()
        choices = adjacency[current] - {previous}
        if len(choices) != 1:
            return ()
        next_atom = next(iter(choices))
        if next_atom in path or (next_atom in junctions and next_atom != junction):
            return ()
        path.append(next_atom)
        previous, current = current, next_atom
    if junction not in adjacency[last] or set(adjacency[last]) - {previous, junction}:
        return ()
    return tuple(path)


def _decompose(adjacency, junctions):
    first, second = junctions
    central = _simple_paths(adjacency, first, second)
    if len(central) != 2:
        return None
    central_edges = [{frozenset((a, b)) for a, b in zip(path, path[1:])}
                     for path in central]
    central_neighbors = {
        first: {path[1] for path in central},
        second: {path[-2] for path in central},
    }
    terminal = {}
    for junction in junctions:
        arms = sorted(adjacency[junction] - central_neighbors[junction])
        if len(arms) != 2:
            return None
        path = _terminal_path(adjacency, junction, arms[0], arms[1], junctions=set(junctions))
        if not path:
            return None
        terminal[junction] = path
    parts = [set(terminal[first]), set(terminal[second]),
             *[set(path[1:-1]) for path in central], set(junctions)]
    # A central path may be the direct spiro--spiro bond.  That is the
    # zero-linking-atom segment; its edge and the other central path still
    # define one saturated monocyclic component.
    if any(not part for part in parts[:2]):
        return None
    if sum(len(part) for part in parts) != len(adjacency):
        return None
    if set().union(*parts) != set(adjacency):
        return None
    expected_edges = set()
    for junction, path in terminal.items():
        walk = (junction,) + path + (junction,)
        expected_edges.update(frozenset((a, b)) for a, b in zip(walk, walk[1:]))
    for path in central:
        expected_edges.update(frozenset((a, b)) for a, b in zip(path, path[1:]))
    graph_edges = {frozenset((a, b)) for a, ns in adjacency.items()
                   for b in ns if a < b}
    if expected_edges != graph_edges:
        return None
    return terminal, central


def dispiro_frontier(mol, Chem=None):
    """Return every source-tied mapped numbering for a bounded linear dispiro.

    The complete input must be an unannotated neutral saturated carbon graph
    with two degree-four spiro junctions, all other atoms degree two, and the
    exact three-cycle linear path partition. ``Chem`` is accepted for caller
    symmetry but not needed by this graph-only producer.
    """
    if mol.GetNumAtoms() > MAX_SKELETAL_ATOMS or mol.GetNumAtoms() < 7:
        return _decline("skeletal size is outside the bounded root vocabulary")
    if mol.GetNumAtoms() != mol.GetNumHeavyAtoms():
        return _decline("explicit hydrogen atoms are outside this complete-core scope")
    atoms = tuple(mol.GetAtoms())
    if any(a.GetAtomicNum() != 6 or a.GetFormalCharge() or a.GetNumRadicalElectrons()
           or a.GetIsotope() or int(a.GetChiralTag()) or a.GetIsAromatic()
           or a.GetTotalValence() != 4 for a in atoms):
        return _decline("only neutral, unannotated saturated carbon atoms are supported")
    if any(b.GetBondTypeAsDouble() != 1 or b.GetStereo() for b in mol.GetBonds()):
        return _decline("unsaturation or bond stereochemistry is outside P-24.2.2 scope")
    adjacency = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()} for a in atoms}
    degrees = {idx: len(neighbors) for idx, neighbors in adjacency.items()}
    junctions = tuple(sorted(idx for idx, degree in degrees.items() if degree == 4))
    if len(junctions) != 2 or any(degree not in {2, 4} for degree in degrees.values()):
        return _decline("requires exactly two degree-four spiro junctions and degree-two remainder")
    if any(degrees[idx] != 2 for idx in degrees if idx not in junctions):
        return _decline("a branched or non-linear ring partition is outside this scope")
    visited = set()
    stack = [next(iter(adjacency))]
    while stack:
        idx = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        stack.extend(adjacency[idx] - visited)
    if len(visited) != len(adjacency):
        return _decline("the spiro skeleton is disconnected")
    edges = {tuple(sorted((a, b))) for a, neighbors in adjacency.items()
             for b in neighbors if a < b}
    if len(edges) - len(adjacency) + 1 != 3:
        return _decline("the complete carbon graph does not have cycle rank three")
    decomposition = _decompose(adjacency, junctions)
    if decomposition is None:
        return _decline("the graph lacks the proven linear three-monocycle path partition")
    terminal, central = decomposition
    first_junction, second_junction = junctions
    terminal_counts = {idx: len(path) for idx, path in terminal.items()}
    central_counts = [len(path) - 2 for path in central]
    if (any(terminal_counts[idx] < 2 for idx in junctions)
            or any(n < 0 for n in central_counts) or sum(central_counts) < 1):
        return _decline("each constituent saturated monocycle must contain at least three atoms")
    first_options = list(junctions)
    smaller = min(terminal_counts.values())
    if terminal_counts[first_junction] != terminal_counts[second_junction]:
        first_options = [idx for idx in junctions if terminal_counts[idx] == smaller]
    total = len(atoms)
    root = ROOTS.get(total)
    if root is None:
        return _decline("the total atom count has no supported acyclic hydrocarbon root")

    raw = []
    for start_junction in first_options:
        end_junction = second_junction if start_junction == first_junction else first_junction
        first_terminal = terminal[start_junction]
        second_terminal = terminal[end_junction]
        oriented_central = []
        for path in central:
            if path[0] == start_junction and path[-1] == end_junction:
                oriented_central.append(path)
            elif path[0] == end_junction and path[-1] == start_junction:
                oriented_central.append(tuple(reversed(path)))
        if len(oriented_central) != 2:
            return _decline("central paths cannot be oriented from the selected terminal ring")
        short_count = min(len(path) - 2 for path in oriented_central)
        short_paths = [path for path in oriented_central if len(path) - 2 == short_count]
        for short_path in short_paths:
            long_path = next(path for path in oriented_central if path != short_path)
            for first_walk in (first_terminal, tuple(reversed(first_terminal))):
                for second_walk in (second_terminal, tuple(reversed(second_terminal))):
                    locant_start = len(first_walk) + 1
                    locant_end = locant_start + short_count + 1
                    sequence = (first_walk + (start_junction,) + short_path[1:-1]
                                + (end_junction,) + second_walk + tuple(reversed(long_path[1:-1])))
                    if len(sequence) != total or len(set(sequence)) != total:
                        continue
                    descriptor = (
                        DescriptorTerm(len(first_walk)),
                        DescriptorTerm(short_count),
                        DescriptorTerm(len(second_walk), locant_end),
                        DescriptorTerm(len(long_path) - 2, locant_start),
                    )
                    locant_map = {idx: str(number) for number, idx in enumerate(sequence, 1)}
                    spiro_locs = (int(locant_map[start_junction]), int(locant_map[end_junction]))
                    # P-24.2.2.1: first minimize the initial linking-number set,
                    # then the spiro-locant set; P-24.2.2.2 then compares every
                    # descriptor number in order of citation.
                    number_set = tuple(sorted((descriptor[0].linking_atoms,
                                               descriptor[1].linking_atoms)))
                    criterion_key = (number_set, tuple(sorted(spiro_locs)),
                                     tuple(term.linking_atoms for term in descriptor))
                    raw.append(DispiroSkeleton(
                        sequence=tuple(sequence),
                        locants=tuple((idx, locant_map[idx]) for idx in sequence),
                        descriptor=descriptor,
                        spiro_atoms=(start_junction, end_junction),
                        spiro_locants=spiro_locs,
                        ring_sizes=tuple(sorted((len(first_walk) + 1,
                                                 len(second_walk) + 1,
                                                 len(oriented_central[0]) + len(oriented_central[1]) - 2))),
                        hydrocarbon_parent=root + "ane",
                        edges=tuple(sorted(edges)),
                        criteria_key=criterion_key,
                    ))
    if not raw:
        return _decline("no complete P-24 numbering was constructed")
    best = min(candidate.criteria_key for candidate in raw)
    frontier = {}
    for candidate in raw:
        if candidate.criteria_key == best:
            frontier.setdefault(candidate.sequence, candidate)
    return DispiroFrontier("complete", "source-ranked dispiro numbering frontier",
                           tuple(frontier[key] for key in sorted(frontier)))
