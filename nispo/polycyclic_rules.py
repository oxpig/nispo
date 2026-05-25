from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations, permutations

from .branch_rules import _branch_prefix
from .constants import MULT, REPLACEMENT, ROOTS, SKELETON_ATOMS
from .formatting import format_prefixes as _format_prefixes, join_locants as _join_locants
from .ring_topology import ring_system_graph as _cached_ring_system_graph
from .ring_topology import ring_system_topologies as _cached_ring_system_topologies


def bicyclic_parent_candidate_records(mol, Chem) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    for ring_system in _ring_systems(mol, Chem):
        candidates.extend(_bicyclic_ring_system_candidate_records(mol, Chem, ring_system))
    return sorted(candidates, key=lambda item: (len(item[0]), item[0]))


def spiro_parent_candidate_records(mol, Chem) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    for ring_system in _ring_systems(mol, Chem):
        candidates.extend(_dispiro_ring_system_candidate_records(mol, Chem, ring_system))
        candidates.extend(_spiro_ring_system_candidate_records(mol, Chem, ring_system))
    return sorted(candidates, key=lambda item: (len(item[0]), item[0]))


def kekulized_polycyclic_parent_candidate_records(mol, Chem) -> list[tuple[str, tuple[int, ...]]]:
    """Late generic fallback for fused systems not covered by retained parents.

    OPSIN accepts many replacement-style bicyclo/tricyclo names with explicit
    unsaturation.  Kekulizing lets the generic bridge rules see aromatic systems
    as deterministic single/double-bond graphs while leaving curated retained
    names to higher-priority rules.
    """

    kekulized = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekulized, clearAromaticFlags=True)
    except Exception:
        kekulized = mol
    candidates = []
    for ring_system in _ring_systems(kekulized, Chem):
        graph = _ring_system_graph(kekulized, Chem, ring_system)
        if not graph:
            continue
        rank = _graph_cycle_rank(graph)
        if rank == 2:
            record_sets = (
                _bicyclic_ring_system_candidate_records(kekulized, Chem, ring_system),
                _spiro_ring_system_candidate_records(kekulized, Chem, ring_system),
            )
        elif rank == 3:
            record_sets = (
                _tricyclic_ring_system_candidate_records(kekulized, Chem, ring_system),
                _dispiro_ring_system_candidate_records(kekulized, Chem, ring_system),
            )
        else:
            continue
        for records in record_sets:
            candidates.extend(records)
    seen = set()
    unique = []
    for name, seq in sorted(candidates, key=lambda item: (len(item[0]), item[0], item[1])):
        key = (name, seq)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, seq))
    return unique


def polycyclic_substituent_prefix(mol, Chem, attachment_idx: int, parent_idx: int) -> str:
    candidates = _polycyclic_substituent_candidates(mol, Chem, attachment_idx, parent_idx)
    if candidates:
        return min(candidates)[-1]

    kekulized = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekulized, clearAromaticFlags=True)
    except Exception:
        return ""
    candidates = _polycyclic_substituent_candidates(kekulized, Chem, attachment_idx, parent_idx)
    return min(candidates)[-1] if candidates else ""


def _polycyclic_substituent_candidates(mol, Chem, attachment_idx: int, parent_idx: int) -> list[tuple]:
    candidates = []
    for atoms in _ring_systems(mol, Chem):
        if attachment_idx not in atoms or parent_idx in atoms:
            continue
        for parent, seq in _polycyclic_parent_sequence_candidates(mol, Chem, atoms):
            if attachment_idx not in seq:
                continue
            name = _name_polycyclic_substituent(mol, seq, atoms, parent, attachment_idx, parent_idx)
            if name:
                loc = seq.index(attachment_idx) + 1
                candidates.append(
                    (
                        0 if loc == 1 else 1,
                        _hetero_locants(mol, seq),
                        _substituent_locants(mol, seq, atoms),
                        name,
                    )
                )
    return candidates


def _bicyclic_ring_system_candidate_records(mol, Chem, atoms: set[int]) -> list[tuple[str, tuple[int, ...]]]:
    graph = _ring_system_graph(mol, Chem, atoms)
    if not graph:
        return []
    edge_count = sum(len(nbrs) for nbrs in graph.values()) // 2
    if edge_count - len(atoms) + 1 != 2:
        return []
    if any(_unsuitable_ring_atom(mol, idx) for idx in atoms):
        return []

    return _decorated_polycyclic_records(
        mol,
        atoms,
        _bicyclic_parent_sequence_candidates(mol, atoms, graph),
    )


def _polycyclic_parent_sequence_candidates(mol, Chem, atoms: set[int]) -> list[tuple[str, tuple[int, ...]]]:
    graph = _ring_system_graph(mol, Chem, atoms)
    if not graph:
        return []
    rank = _graph_cycle_rank(graph)
    if any(_unsuitable_ring_atom(mol, idx) for idx in atoms):
        return []
    candidates: list[tuple[str, tuple[int, ...]]] = []
    if rank == 2:
        candidates.extend(_bicyclic_parent_sequence_candidates(mol, atoms, graph))
        candidates.extend(_spiro_parent_sequence_candidates(mol, atoms, graph))
    elif rank == 3:
        candidates.extend(_secondary_zero_bridge_tricyclic_sequence_candidates(mol, atoms, graph))
    seen = set()
    unique = []
    for parent, seq in candidates:
        key = (parent, seq)
        if key in seen:
            continue
        seen.add(key)
        unique.append((parent, seq))
    return unique


def _bicyclic_parent_sequence_candidates(
    mol, atoms: set[int], graph: dict[int, set[int]]
) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    for a, b in _bridgehead_pair_candidates(graph, atoms, 3):
        paths = _bridge_paths(graph, atoms, a, b)
        if not paths:
            continue
        for seq, bracket in _numbering_sequences(paths):
            parent = _bicyclic_parent(mol, seq, bracket)
            if parent:
                candidates.append((parent, seq))
    return candidates


def _spiro_parent_sequence_candidates(
    mol, atoms: set[int], graph: dict[int, set[int]]
) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    for spiro_idx in sorted(atoms):
        components = _components_without_atom(graph, spiro_idx)
        if len(components) != 2:
            continue
        paths = []
        ok = True
        for component in components:
            endpoints = [idx for idx in component if len(graph[idx] & component) == 1]
            if len(endpoints) != 2:
                ok = False
                break
            path = _path_within_component(graph, endpoints[0], endpoints[1], component)
            if not path:
                ok = False
                break
            paths.append(path)
        if not ok:
            continue
        for seq, bracket in _spiro_numbering_sequences(paths, spiro_idx):
            parent = _spiro_parent(mol, seq, bracket)
            if parent:
                candidates.append((parent, seq))
    return candidates


def _tricyclic_ring_system_candidate_records(mol, Chem, atoms: set[int]) -> list[tuple[str, tuple[int, ...]]]:
    graph = _ring_system_graph(mol, Chem, atoms)
    if not graph:
        return []
    edge_count = sum(len(nbrs) for nbrs in graph.values()) // 2
    if edge_count - len(atoms) + 1 != 3:
        return []
    if any(_unsuitable_ring_atom(mol, idx) for idx in atoms):
        return []
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings() if set(ring) <= atoms]
    if any(len(left & right) == 1 for left, right in combinations(rings, 2)):
        return _secondary_zero_bridge_tricyclic_candidate_records(mol, atoms, graph)
    parent_sequences = []
    for a, b in _bridgehead_pair_candidates(graph, atoms, 4):
        paths = _bridge_paths_with_count(graph, atoms, a, b, 4)
        if not paths:
            continue
        for seq, bracket in _polybridge_numbering_sequences(paths, annotate_zero_bridges=True):
            parent = _tricyclic_parent(mol, seq, bracket)
            if parent:
                parent_sequences.append((parent, seq))
    candidates = _decorated_polycyclic_records(mol, atoms, parent_sequences)
    if not candidates:
        return _secondary_zero_bridge_tricyclic_candidate_records(mol, atoms, graph)
    return candidates


def _secondary_zero_bridge_tricyclic_candidate_records(
    mol, atoms: set[int], graph: dict[int, set[int]]
) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    for parent, seq, bracket in _secondary_zero_bridge_tricyclic_sequence_candidates(
        mol, atoms, graph, include_bracket=True
    ):
        name = _decorate_polycyclic_parent(mol, seq, atoms, parent)
        if not name:
            continue
        key = (
            _bridge_sort_key(bracket),
            _hetero_locants(mol, seq),
            _substituent_locants(mol, seq, atoms),
            name,
        )
        candidates.append((key, name, tuple(seq)))
    candidates.sort(key=lambda item: item[0])
    return [(name, seq) for _key, name, seq in candidates]


def _secondary_zero_bridge_tricyclic_sequence_candidates(
    mol, atoms: set[int], graph: dict[int, set[int]], *, include_bracket: bool = False
):
    candidates = []
    for left, right in _graph_edges(graph):
        reduced = _graph_without_edge(graph, left, right)
        if not _graph_is_connected(reduced):
            continue
        if _graph_cycle_rank(reduced) != 2:
            continue
        for a, b in _bridgehead_pair_candidates(reduced, atoms, 3):
            paths = _bridge_paths(reduced, atoms, a, b)
            if not paths:
                continue
            for seq, bracket in _numbering_sequences(paths):
                pos = {idx: loc for loc, idx in enumerate(seq, start=1)}
                if left not in pos or right not in pos:
                    continue
                zero_bridge = _zero_bridge_descriptor(pos[left], pos[right])
                full_bracket = tuple(bracket) + (zero_bridge,)
                parent = _tricyclic_parent(mol, seq, full_bracket)
                if not parent:
                    continue
                if include_bracket:
                    candidates.append((parent, seq, full_bracket))
                else:
                    candidates.append((parent, seq))
    return candidates


def _spiro_ring_system_candidate_records(mol, Chem, atoms: set[int]) -> list[tuple[str, tuple[int, ...]]]:
    graph = _ring_system_graph(mol, Chem, atoms)
    if not graph:
        return []
    edge_count = sum(len(nbrs) for nbrs in graph.values()) // 2
    if edge_count - len(atoms) + 1 != 2:
        return []
    if any(_unsuitable_ring_atom(mol, idx) for idx in atoms):
        return []

    return _decorated_polycyclic_records(
        mol,
        atoms,
        _spiro_parent_sequence_candidates(mol, atoms, graph),
    )


def _decorated_polycyclic_records(
    mol, atoms: set[int], parent_sequences: Sequence[tuple[str, tuple[int, ...]]]
) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    for parent, seq in parent_sequences:
        name = _decorate_polycyclic_parent(mol, seq, atoms, parent)
        if not name:
            continue
        key = (_hetero_locants(mol, seq), _substituent_locants(mol, seq, atoms), name)
        candidates.append((key, name, tuple(seq)))
    candidates.sort(key=lambda item: item[0])
    return [(name, seq) for _key, name, seq in candidates]


def _dispiro_ring_system_candidate_records(mol, Chem, atoms: set[int]) -> list[tuple[str, tuple[int, ...]]]:
    graph = _ring_system_graph(mol, Chem, atoms)
    if not graph:
        return []
    edge_count = sum(len(nbrs) for nbrs in graph.values()) // 2
    if edge_count - len(atoms) + 1 != 3:
        return []
    if any(_unsuitable_ring_atom(mol, idx) for idx in atoms):
        return []

    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings() if set(ring) <= atoms]
    if len(rings) != 3:
        return []
    spiro_atoms = sorted(idx for idx in atoms if sum(idx in ring for ring in rings) == 2)
    if len(spiro_atoms) != 2:
        return []
    if any(sum((left & right) == {idx} for left, right in combinations(rings, 2)) == 0 for idx in spiro_atoms):
        return []
    if any(len(left & right) > 1 for left, right in combinations(rings, 2)):
        return []

    terminal_rings = [ring for ring in rings if len(ring & set(spiro_atoms)) == 1]
    central_rings = [ring for ring in rings if set(spiro_atoms) <= ring]
    if len(terminal_rings) != 2 or len(central_rings) != 1:
        return []

    terminal_paths = {}
    for ring in terminal_rings:
        spiro_idx = next(iter(ring & set(spiro_atoms)))
        path = _terminal_ring_path(graph, ring, spiro_idx)
        if not path:
            return []
        terminal_paths[spiro_idx] = path

    central_paths = _ring_paths_between(graph, central_rings[0], spiro_atoms[0], spiro_atoms[1])
    if len(central_paths) != 2:
        return []

    candidates = []
    for first_spiro, second_spiro in (tuple(spiro_atoms), tuple(reversed(spiro_atoms))):
        first_terminal = terminal_paths[first_spiro]
        second_terminal = terminal_paths[second_spiro]
        central_oriented = [
            _orient_path(path, first_spiro, second_spiro)
            for path in central_paths
            if path[0] in {first_spiro, second_spiro} and path[-1] in {first_spiro, second_spiro}
        ]
        if len(central_oriented) != 2:
            continue
        central_orderings = (
            tuple(sorted(central_oriented, key=lambda path: (len(path), path))),
            tuple(sorted(central_oriented, key=lambda path: (len(path), path), reverse=True)),
        )
        for first_path in (first_terminal, tuple(reversed(first_terminal))):
            if first_path[-1] != first_spiro:
                continue
            for second_path in (second_terminal, tuple(reversed(second_terminal))):
                if second_path[0] != second_spiro:
                    continue
                for central_pair in central_orderings:
                    short_central, long_central = central_pair
                    seq = (
                        first_path[:-1]
                        + (first_spiro,)
                        + short_central[1:-1]
                        + (second_spiro,)
                        + second_path[1:]
                        + tuple(reversed(long_central[1:-1]))
                    )
                    if len(seq) != len(atoms) or len(seq) != len(set(seq)):
                        continue
                    bracket = (
                        len(first_path) - 1,
                        len(short_central) - 2,
                        len(second_path) - 1,
                        len(long_central) - 2,
                    )
                    parent = _dispiro_parent(mol, seq, bracket)
                    if not parent:
                        continue
                    name = _decorate_polycyclic_parent(mol, seq, atoms, parent)
                    if not name:
                        continue
                    key = (
                        sorted(bracket),
                        _hetero_locants(mol, seq),
                        _substituent_locants(mol, seq, atoms),
                        name,
                    )
                    candidates.append((key, name, tuple(seq)))
    candidates.sort(key=lambda item: item[0])
    return [(name, seq) for _key, name, seq in candidates]


def _ring_systems(mol, Chem=None) -> list[set[int]]:
    if Chem is not None:
        return [set(topology.atoms) for topology in _cached_ring_system_topologies(mol, Chem)]

    cached = getattr(mol, "_nispo_polycyclic_ring_systems", None)
    if cached is not None:
        return [set(system) for system in cached]
    systems: list[set[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        current = set(ring)
        merged = []
        for system in systems:
            if current & system:
                current |= system
            else:
                merged.append(system)
        merged.append(current)
        systems = merged

    changed = True
    while changed:
        changed = False
        merged = []
        while systems:
            current = systems.pop()
            overlaps = [system for system in systems if current & system]
            if overlaps:
                changed = True
                for system in overlaps:
                    current |= system
                    systems.remove(system)
            merged.append(current)
        systems = merged
    result = tuple(frozenset(system) for system in systems)
    try:
        mol._nispo_polycyclic_ring_systems = result
    except Exception:
        pass
    return [set(system) for system in result]


def _ring_system_graph(mol, Chem, atoms: set[int]) -> dict[int, set[int]]:
    return {idx: set(nbrs) for idx, nbrs in _cached_ring_system_graph(mol, Chem, atoms).items()}


def _graph_edges(graph: dict[int, set[int]]) -> list[tuple[int, int]]:
    return [(left, right) for left in sorted(graph) for right in sorted(graph[left]) if left < right]


def _graph_without_edge(graph: dict[int, set[int]], left: int, right: int) -> dict[int, set[int]]:
    reduced = {idx: set(nbrs) for idx, nbrs in graph.items()}
    reduced[left].discard(right)
    reduced[right].discard(left)
    return reduced


def _graph_is_connected(graph: dict[int, set[int]]) -> bool:
    if not graph:
        return False
    start = next(iter(graph))
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for nbr in graph[current]:
            if nbr in seen:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return len(seen) == len(graph)


def _graph_cycle_rank(graph: dict[int, set[int]]) -> int:
    edge_count = sum(len(nbrs) for nbrs in graph.values()) // 2
    return edge_count - len(graph) + 1


def _unsuitable_ring_atom(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    charge_allowed = atom.GetFormalCharge() == 0 or (
        atom.GetFormalCharge() > 0 and atom.GetSymbol() in {"N", "S"}
    )
    return (
        atom.GetSymbol() not in SKELETON_ATOMS
        or atom.GetIsAromatic()
        or not charge_allowed
        or atom.GetSymbol() not in {"C", "N", "O", "S"}
    )


def _bridge_paths(graph: dict[int, set[int]], atoms: set[int], a: int, b: int) -> list[tuple[int, ...]]:
    return _bridge_paths_with_count(graph, atoms, a, b, 3)


def _bridgehead_pair_candidates(
    graph: dict[int, set[int]], atoms: set[int], path_count: int
) -> tuple[tuple[int, int], ...]:
    """Return plausible bridgehead pairs for von Baeyer path searches.

    A bridged bicyclo/tricyclo parent is a theta-style ring graph: the shared
    bridgehead atoms are the vertices where three or more ring paths meet.  The
    previous generic search tried every atom pair and then rejected almost all
    of them after enumerating simple paths.  Restricting the expensive path
    search to branch vertices keeps the rule deterministic while matching the
    graph-theoretic construction OPSIN parses for bicyclo/tricyclo names.
    """

    branch_atoms = sorted(idx for idx in atoms if len(graph.get(idx, ())) >= 3)
    if len(branch_atoms) >= 2:
        return tuple(combinations(branch_atoms, 2))
    if path_count >= 4:
        return ()
    return tuple(combinations(sorted(atoms), 2))


def _bridge_paths_with_count(
    graph: dict[int, set[int]], atoms: set[int], a: int, b: int, path_count: int
) -> list[tuple[int, ...]]:
    paths = _simple_paths(graph, a, b, max_paths=24)
    valid = []
    for selected in combinations(paths, path_count):
        internals = [set(path[1:-1]) for path in selected]
        if any(left & right for i, left in enumerate(internals) for right in internals[i + 1 :]):
            continue
        if set().union(*internals, {a, b}) == atoms:
            valid.append(selected)
    if not valid:
        return []
    valid.sort(key=lambda item: sorted((len(path) - 2 for path in item), reverse=True), reverse=True)
    return list(valid[0])


def _simple_paths(graph: dict[int, set[int]], start: int, end: int, max_paths: int) -> list[tuple[int, ...]]:
    paths = []
    stack = [(start, (start,))]
    while stack and len(paths) < max_paths:
        current, path = stack.pop()
        for nbr in sorted(graph[current]):
            if nbr in path:
                continue
            next_path = path + (nbr,)
            if nbr == end:
                paths.append(next_path)
            else:
                stack.append((nbr, next_path))
    return paths


def _components_without_atom(graph: dict[int, set[int]], removed: int) -> list[set[int]]:
    remaining = set(graph) - {removed}
    components = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for nbr in graph[current]:
                if nbr == removed or nbr in component:
                    continue
                component.add(nbr)
                remaining.discard(nbr)
                stack.append(nbr)
        components.append(component)
    return components


def _path_within_component(
    graph: dict[int, set[int]], start: int, end: int, component: set[int]
) -> tuple[int, ...]:
    stack = [(start, (start,))]
    while stack:
        current, path = stack.pop()
        if current == end and set(path) == component:
            return path
        for nbr in sorted(graph[current] & component):
            if nbr not in path:
                stack.append((nbr, path + (nbr,)))
    return ()


def _terminal_ring_path(graph: dict[int, set[int]], ring: set[int], spiro_idx: int) -> tuple[int, ...]:
    neighbours = sorted(graph[spiro_idx] & ring)
    if len(neighbours) != 2:
        return ()
    path = _path_within_component(graph, neighbours[0], neighbours[1], ring - {spiro_idx})
    if not path:
        return ()
    return path + (spiro_idx,)


def _ring_paths_between(
    graph: dict[int, set[int]], ring: set[int], first: int, second: int
) -> tuple[tuple[int, ...], ...]:
    ring_graph = {idx: graph[idx] & ring for idx in ring}
    paths = [
        path
        for path in _simple_paths(ring_graph, first, second, max_paths=8)
        if set(path) <= ring
    ]
    return tuple(sorted(paths, key=lambda path: (len(path), path))[:2])


def _orient_path(path: tuple[int, ...], first: int, second: int) -> tuple[int, ...]:
    if path[0] == first and path[-1] == second:
        return path
    if path[0] == second and path[-1] == first:
        return tuple(reversed(path))
    return ()


def _numbering_sequences(paths: Sequence[tuple[int, ...]]) -> list[tuple[tuple[int, ...], tuple[int, int, int]]]:
    return _polybridge_numbering_sequences(paths)


def _polybridge_numbering_sequences(
    paths: Sequence[tuple[int, ...]], annotate_zero_bridges: bool = False
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    sequences = []
    for oriented_paths in (paths, tuple(tuple(reversed(path)) for path in paths)):
        for ordered in _bridge_path_orderings(oriented_paths):
            first, *remaining = ordered
            seq = first[:1] + first[1:-1] + first[-1:]
            for path in remaining:
                seq += tuple(reversed(path[1:-1]))
            if len(seq) == len(set(seq)):
                pos = {idx: loc for loc, idx in enumerate(seq, start=1)}
                bracket = tuple(_bridge_descriptor(path, pos, annotate_zero_bridges) for path in ordered)
                sequences.append((seq, bracket))
    return list(dict.fromkeys(sequences))


def _bridge_path_orderings(paths: Sequence[tuple[int, ...]]) -> list[tuple[tuple[int, ...], ...]]:
    """Return von Baeyer bridge orders, including equal-length alternatives.

    Bridge lengths are ordered descending.  When several bridges have the same
    length, OPSIN accepts the order that gives preferred hetero/substituent
    locants.  NISPO therefore needs to expose all tied orders to the existing
    candidate sorter rather than locking in RDKit's atom-index order.
    """

    groups: list[list[tuple[int, ...]]] = []
    for path in sorted(paths, key=lambda item: -(len(item) - 2)):
        if not groups or len(groups[-1][0]) != len(path):
            groups.append([path])
        else:
            groups[-1].append(path)
    orderings: list[tuple[tuple[int, ...], ...]] = [()]
    for group in groups:
        next_orderings = []
        for prefix in orderings:
            for permuted in permutations(group):
                next_orderings.append(prefix + tuple(permuted))
        orderings = next_orderings
    return orderings


def _bridge_descriptor(path: Sequence[int], pos: dict[int, int], annotate_zero: bool):
    length = len(path) - 2
    if length == 0 and annotate_zero:
        return _zero_bridge_descriptor(pos[path[0]], pos[path[-1]])
    return length


def _zero_bridge_descriptor(left: int, right: int) -> str:
    low, high = sorted((left, right))
    return f"0^{low},{high}"


def _bridge_sort_key(bracket: Sequence[int | str]) -> tuple:
    values = []
    for item in bracket:
        if isinstance(item, int):
            values.append((item, 0, 0))
            continue
        length, locants = item.split("^", 1)
        left, right = locants.split(",", 1)
        values.append((int(length), int(left), int(right)))
    return tuple(values)


def _spiro_numbering_sequences(
    paths: Sequence[tuple[int, ...]], spiro_idx: int
) -> list[tuple[tuple[int, ...], tuple[int, int]]]:
    ordered_paths = sorted(paths, key=lambda path: (len(path), path))
    sequences = []
    for first in (ordered_paths[0], tuple(reversed(ordered_paths[0]))):
        for second in (ordered_paths[1], tuple(reversed(ordered_paths[1]))):
            seq = first + (spiro_idx,) + second
            if len(seq) == len(set(seq)):
                sequences.append((seq, (len(first), len(second))))
    return sequences


def _bicyclic_parent(mol, seq: Sequence[int], bracket: tuple[int, int, int]) -> str:
    root = ROOTS.get(len(seq))
    if not root:
        return ""
    parent = _poly_parent_with_unsaturation(
        mol,
        seq,
        f"bicyclo[{'.'.join(str(value) for value in bracket)}]{root}",
        allow_unsaturation=True,
    )
    if not parent:
        return ""
    grouped: dict[str, list[int]] = defaultdict(list)
    for loc, idx in enumerate(seq, start=1):
        replacement = _polycyclic_replacement_term(mol, idx)
        if not replacement:
            continue
        grouped[replacement].append(loc)
    parts = []
    for replacement in sorted(grouped):
        locants = grouped[replacement]
        if len(locants) == 1:
            parts.append(f"{locants[0]}-{replacement}")
        else:
            parts.append(f"{_join_locants(locants)}-{MULT.get(len(locants), '')}{replacement}")
    return "-".join(parts) + parent if parts else parent


def _tricyclic_parent(mol, seq: Sequence[int], bracket: tuple[int, ...]) -> str:
    root = ROOTS.get(len(seq))
    if not root or len(bracket) != 4:
        return ""
    parent = _poly_parent_with_unsaturation(
        mol,
        seq,
        f"tricyclo[{'.'.join(str(value) for value in bracket)}]{root}",
        allow_unsaturation=True,
    )
    if not parent:
        return ""
    grouped: dict[str, list[int]] = defaultdict(list)
    for loc, idx in enumerate(seq, start=1):
        replacement = _polycyclic_replacement_term(mol, idx)
        if not replacement:
            continue
        grouped[replacement].append(loc)
    parts = []
    for replacement in sorted(grouped):
        locants = grouped[replacement]
        if len(locants) == 1:
            parts.append(f"{locants[0]}-{replacement}")
        else:
            parts.append(f"{_join_locants(locants)}-{MULT.get(len(locants), '')}{replacement}")
    return "-".join(parts) + parent if parts else parent


def _spiro_parent(mol, seq: Sequence[int], bracket: tuple[int, int]) -> str:
    root = ROOTS.get(len(seq))
    if not root:
        return ""
    parent = _poly_parent_with_unsaturation(
        mol,
        seq,
        f"spiro[{'.'.join(str(value) for value in bracket)}]{root}",
        allow_unsaturation=True,
    )
    if not parent:
        return ""
    grouped: dict[str, list[int]] = defaultdict(list)
    for loc, idx in enumerate(seq, start=1):
        replacement = _polycyclic_replacement_term(mol, idx)
        if not replacement:
            continue
        grouped[replacement].append(loc)
    parts = []
    for replacement in sorted(grouped):
        locants = grouped[replacement]
        if len(locants) == 1:
            parts.append(f"{locants[0]}-{replacement}")
        else:
            parts.append(f"{_join_locants(locants)}-{MULT.get(len(locants), '')}{replacement}")
    return "-".join(parts) + parent if parts else parent


def _dispiro_parent(mol, seq: Sequence[int], bracket: tuple[int, int, int, int]) -> str:
    root = ROOTS.get(len(seq))
    if not root:
        return ""
    parent = _poly_parent_with_unsaturation(
        mol,
        seq,
        f"dispiro[{'.'.join(str(value) for value in bracket)}]{root}",
        allow_unsaturation=True,
    )
    if not parent:
        return ""
    grouped: dict[str, list[int]] = defaultdict(list)
    for loc, idx in enumerate(seq, start=1):
        replacement = _polycyclic_replacement_term(mol, idx)
        if not replacement:
            continue
        grouped[replacement].append(loc)
    parts = []
    for replacement in sorted(grouped):
        locants = grouped[replacement]
        if len(locants) == 1:
            parts.append(f"{locants[0]}-{replacement}")
        else:
            parts.append(f"{_join_locants(locants)}-{MULT.get(len(locants), '')}{replacement}")
    return "-".join(parts) + parent if parts else parent


def _polycyclic_replacement_term(mol, idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    symbol = atom.GetSymbol()
    if symbol == "C":
        return ""
    if atom.GetFormalCharge() > 0 and symbol == "N":
        return "azonia"
    if atom.GetFormalCharge() > 0 and symbol == "S":
        return "thionia"
    return REPLACEMENT.get(symbol, "")


def _poly_parent_with_unsaturation(
    mol, seq: Sequence[int], base: str, allow_unsaturation: bool
) -> str:
    pos = {idx: loc for loc, idx in enumerate(seq, start=1)}
    ene_locs = []
    yne_locs = []
    seen = set()
    for idx in seq:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in pos:
                continue
            key = tuple(sorted((idx, nidx)))
            if key in seen:
                continue
            seen.add(key)
            bond = mol.GetBondBetweenAtoms(idx, nidx)
            order = bond.GetBondTypeAsDouble()
            if order == 1.0:
                continue
            loc = _bond_locant(pos[idx], pos[nidx], len(seq))
            if order == 2.0:
                ene_locs.append(loc)
            elif order == 3.0:
                yne_locs.append(loc)
            else:
                return ""
    if not ene_locs and not yne_locs:
        return base + "ane"
    if not allow_unsaturation:
        return ""
    if ene_locs and not yne_locs:
        return _poly_loc_unsat(base, sorted(ene_locs, key=_unsat_sort_key), "ene")
    if yne_locs and not ene_locs:
        return _poly_loc_unsat(base, sorted(yne_locs, key=_unsat_sort_key), "yne")
    ene = _poly_loc_unsat(base, sorted(ene_locs, key=_unsat_sort_key), "en", terminal_e=False)
    yne = _poly_loc_unsat("", sorted(yne_locs, key=_unsat_sort_key), "yne", terminal_e=True)
    return ene + "-" + yne


def _bond_locant(left: int, right: int, atom_count: int) -> str:
    low, high = sorted((left, right))
    return f"{low}({high})"


def _poly_loc_unsat(root: str, locs: list[str], suffix: str, terminal_e: bool = True) -> str:
    if len(locs) == 1:
        if not root:
            return f"{locs[0]}-{suffix}"
        return f"{root}-{locs[0]}-{suffix}"
    mult = MULT.get(len(locs), "")
    joiner = "a-" if root and suffix in {"ene", "yne"} else "-"
    ending = f"{mult}{suffix}"
    if not terminal_e and ending.endswith("e"):
        ending = ending[:-1]
    return f"{root}{joiner}{','.join(locs)}-{ending}"


def _unsat_sort_key(locant: str) -> tuple[int, int]:
    if "(" not in locant:
        value = int(locant)
        return (value, value)
    left, right = locant.rstrip(")").split("(", 1)
    return (int(left), int(right))


def _decorate_polycyclic_parent(mol, seq: Sequence[int], core_atoms: set[int], parent: str) -> str:
    pos = {idx: str(loc) for loc, idx in enumerate(seq, start=1)}
    prefixes = []
    for idx in seq:
        locant = pos[idx]
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core_atoms:
                continue
            prefix = _branch_prefix(mol, nidx, idx)
            if not prefix:
                return ""
            prefixes.append((locant, prefix))
    prefix_text = _format_prefixes(prefixes)
    return prefix_text + "-" + parent if prefix_text else parent


def _name_polycyclic_substituent(
    mol,
    seq: Sequence[int],
    core_atoms: set[int],
    parent: str,
    attachment_idx: int,
    blocked_idx: int,
) -> str:
    pos = {idx: loc for loc, idx in enumerate(seq, start=1)}
    attachment_loc = pos.get(attachment_idx)
    if attachment_loc is None:
        return ""
    suffix = _polycyclic_substituent_suffix(parent, attachment_loc)
    prefixes = []
    for idx in seq:
        locant = str(pos[idx])
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == blocked_idx or nidx in core_atoms:
                continue
            prefix = _branch_prefix(mol, nidx, idx)
            if not prefix:
                return ""
            prefixes.append((locant, prefix))
    prefix_text = _format_prefixes(prefixes)
    return prefix_text + "-" + suffix if prefix_text else suffix


def _polycyclic_substituent_suffix(parent: str, locant: int) -> str:
    stem = parent[:-1] if parent.endswith("e") else parent
    return f"{stem}-{locant}-yl"


def _hetero_locants(mol, seq: Sequence[int]) -> list[int]:
    return [loc for loc, idx in enumerate(seq, start=1) if mol.GetAtomWithIdx(idx).GetSymbol() != "C"]


def _substituent_locants(mol, seq: Sequence[int], core_atoms: set[int]) -> list[int]:
    locants = []
    for loc, idx in enumerate(seq, start=1):
        if any(nbr.GetIdx() not in core_atoms for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()):
            locants.append(loc)
    return locants
