from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations, permutations

from .branch_rules import branch_prefix
from .constants import MULT, REPLACEMENT, ROOTS, SKELETON_ATOMS
from .formatting import format_prefixes as _format_prefixes
from .formatting import join_locants as _join_locants
from .formatting import join_prefix_parent as _join_prefix_parent
from .ring_topology import ring_system_graph as _cached_ring_system_graph
from .ring_topology import ring_system_topologies as _cached_ring_system_topologies
from .simple_prefixes import simple_external_prefix
from .stereochemistry import _has_assigned_bond_stereo, assigned_atom_stereochemistry


def bicyclic_parent_candidate_records(mol, Chem) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    for ring_system in _ring_systems(mol, Chem):
        candidates.extend(_bicyclic_ring_system_candidate_records(mol, Chem, ring_system))
    return candidates


def _simple_or_branch_prefix(mol, idx: int, parent_idx: int) -> str:
    return simple_external_prefix(mol, idx, parent_idx) or branch_prefix(mol, idx, parent_idx)


def spiro_parent_candidate_records(mol, Chem) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    for ring_system in _ring_systems(mol, Chem):
        candidates.extend(_dispiro_ring_system_candidate_records(mol, Chem, ring_system))
        candidates.extend(_spiro_ring_system_candidate_records(mol, Chem, ring_system))
    return candidates


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
        kekulized.SetBoolProp("_nispo_kekulized_aromatic_view", True)
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
        elif 4 <= rank <= 9:
            record_sets = (
                _secondary_bridge_polycyclic_candidate_records(kekulized, Chem, ring_system),
            )
        else:
            continue
        for records in record_sets:
            candidates.extend(records)
    seen = set()
    unique = []
    for name, seq in candidates:
        key = (name, seq)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, seq))
    return unique


def polycyclic_substituent_prefix(mol, Chem, attachment_idx: int, parent_idx: int) -> str:
    if not _has_polycyclic_substituent_system(mol, Chem, attachment_idx, parent_idx):
        return ""
    candidates = _polycyclic_substituent_candidates(mol, Chem, attachment_idx, parent_idx)
    if candidates:
        return min(candidates)[-1]

    kekulized = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekulized, clearAromaticFlags=True)
        kekulized.SetBoolProp("_nispo_kekulized_aromatic_view", True)
    except Exception:
        return ""
    candidates = _polycyclic_substituent_candidates(kekulized, Chem, attachment_idx, parent_idx)
    return min(candidates)[-1] if candidates else ""


def _has_polycyclic_substituent_system(mol, Chem, attachment_idx: int, parent_idx: int) -> bool:
    for topology in _cached_ring_system_topologies(mol, Chem):
        if attachment_idx in topology.atoms and parent_idx not in topology.atoms and topology.cycle_rank in {2, 3}:
            return True
    return False


def _polycyclic_substituent_candidates(mol, Chem, attachment_idx: int, parent_idx: int) -> list[tuple]:
    candidates = []
    for atoms in _ring_systems(mol, Chem):
        if attachment_idx not in atoms or parent_idx in atoms:
            continue
        for parent, seq in _polycyclic_parent_sequence_candidates(mol, Chem, atoms):
            if attachment_idx not in seq:
                continue
            name = _name_polycyclic_substituent(mol, Chem, seq, atoms, parent, attachment_idx, parent_idx)
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
        candidates = _secondary_bridge_tricyclic_candidate_records(mol, atoms, graph)
    if not candidates:
        return _secondary_zero_bridge_tricyclic_candidate_records(mol, atoms, graph)
    return candidates


def _secondary_bridge_tricyclic_candidate_records(
    mol, atoms: set[int], graph: dict[int, set[int]]
) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    branch_atoms = sorted(idx for idx in atoms if len(graph.get(idx, ())) >= 3)
    multiple_bond_count = _ring_multiple_bond_count(mol, atoms)
    allow_relaxed_base_paths = multiple_bond_count <= 8
    use_saturated_bridge_selection = multiple_bond_count == 0
    for left, right in combinations(branch_atoms, 2):
        for base_paths in _secondary_bridge_base_path_sets(
            graph,
            atoms,
            left,
            right,
            allow_relaxed=allow_relaxed_base_paths,
        ):
            for base_seq, base_bracket in _polybridge_numbering_sequences(base_paths):
                completed = _secondary_bridge_completion(graph, atoms, base_paths, base_seq, base_bracket, 3)
                if completed is None:
                    continue
                seq, bracket = completed
                if len(bracket) != 4:
                    continue
                parent = _tricyclic_parent(mol, seq, bracket)
                if not parent:
                    continue
                name = _decorate_polycyclic_parent(mol, seq, atoms, parent)
                if not name:
                    continue
                key = (
                    _secondary_bridge_preference_key(
                        bracket,
                        saturated_alicyclic=use_saturated_bridge_selection,
                    ),
                    *_polycyclic_numbering_preference_key(mol, seq, atoms),
                    name,
                )
                candidates.append((key, name, tuple(seq)))
    candidates.sort(key=lambda item: item[0])
    return [(name, seq) for _key, name, seq in candidates[:24]]


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
            *_polycyclic_numbering_preference_key(mol, seq, atoms),
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
        key = (*_polycyclic_numbering_preference_key(mol, seq, atoms), name)
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
                        *_polycyclic_numbering_preference_key(mol, seq, atoms),
                        name,
                    )
                    candidates.append((key, name, tuple(seq)))
    candidates.sort(key=lambda item: item[0])
    return [(name, seq) for _key, name, seq in candidates]


def _secondary_bridge_polycyclic_candidate_records(mol, Chem, atoms: set[int]) -> list[tuple[str, tuple[int, ...]]]:
    graph = _ring_system_graph(mol, Chem, atoms)
    if not graph:
        return []
    rank = _graph_cycle_rank(graph)
    if not (4 <= rank <= 9):
        return []
    if any(_unsuitable_ring_atom(mol, idx) for idx in atoms):
        return []

    candidates = []
    branch_atoms = sorted(idx for idx in atoms if len(graph.get(idx, ())) >= 3)
    multiple_bond_count = _ring_multiple_bond_count(mol, atoms)
    relaxed_aromatic_fallback = (
        multiple_bond_count > 8
        and bool(getattr(mol, "HasProp", lambda _name: False)("_nispo_kekulized_aromatic_view"))
    )
    allow_relaxed_base_paths = multiple_bond_count <= 8 or relaxed_aromatic_fallback
    use_saturated_bridge_selection = multiple_bond_count == 0 and mol.GetRingInfo().NumRings() >= 7
    use_kekulized_blue_book_bridge_selection = _use_kekulized_blue_book_bridge_selection(
        mol,
        Chem,
        atoms,
        rank,
        multiple_bond_count,
    )
    use_stereo_hetero_macrocycle_bridge_order = (
        _use_stereo_hetero_kekulized_macrocycle_bridge_order(
            mol,
            Chem,
            atoms,
            rank,
            multiple_bond_count,
        )
    )
    candidate_limit = 8 if relaxed_aromatic_fallback else _secondary_bridge_candidate_limit(rank)
    use_large_hetero_bridge_order = bool(
        getattr(mol, "HasProp", lambda _name: False)("_nispo_kekulized_aromatic_view")
    ) and _use_large_hetero_kekulized_bridge_order(
        mol,
        atoms,
    )
    for left, right in combinations(branch_atoms, 2):
        for base_paths in _secondary_bridge_base_path_sets(
            graph, atoms, left, right, allow_relaxed=allow_relaxed_base_paths
        ):
            for base_seq, base_bracket in _polybridge_numbering_sequences(base_paths):
                completed = _secondary_bridge_completion(graph, atoms, base_paths, base_seq, base_bracket, rank)
                if completed is None:
                    continue
                seq, bracket = completed
                parent = _generic_polycyclic_parent(mol, seq, bracket)
                if not parent:
                    continue
                name = _decorate_polycyclic_parent(mol, seq, atoms, parent)
                if not name:
                    continue
                if use_large_hetero_bridge_order:
                    key = (
                        _large_hetero_kekulized_bridge_preference_key(mol, seq, atoms, bracket),
                        name,
                    )
                elif use_stereo_hetero_macrocycle_bridge_order:
                    key = (
                        _stereo_hetero_kekulized_macrocycle_bridge_preference_key(
                            mol,
                            seq,
                            atoms,
                            bracket,
                        ),
                        name,
                    )
                else:
                    key = (
                        _secondary_bridge_preference_key(
                            bracket,
                            saturated_alicyclic=(
                                use_saturated_bridge_selection
                                or use_kekulized_blue_book_bridge_selection
                            ),
                        ),
                        *_polycyclic_numbering_preference_key(mol, seq, atoms),
                        name,
                    )
                candidates.append((key, name, tuple(seq)))
                if len(candidates) >= candidate_limit:
                    candidates.sort(key=lambda item: item[0])
                    return [(name, seq) for _key, name, seq in candidates[:candidate_limit]]
    candidates.sort(key=lambda item: item[0])
    return [(name, seq) for _key, name, seq in candidates[:candidate_limit]]


def _has_stereochemical_evidence(mol, Chem) -> bool:
    if any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        return True
    return any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds())


def _secondary_bridge_candidate_limit(rank: int) -> int:
    if rank <= 6:
        return 48
    if rank == 7:
        return 36
    return 24


def _use_large_hetero_kekulized_bridge_order(mol, atoms: set[int]) -> bool:
    if len(atoms) < 20:
        return False
    if _ring_multiple_bond_count(mol, atoms) < 9:
        return False
    has_oxygen_or_sulfur = False
    external_attachment_atoms = 0
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() in {"O", "S"}:
            has_oxygen_or_sulfur = True
        if any(nbr.GetIdx() not in atoms for nbr in atom.GetNeighbors()):
            external_attachment_atoms += 1
    return has_oxygen_or_sulfur and external_attachment_atoms >= 3


def _use_kekulized_blue_book_bridge_selection(
    mol,
    Chem,
    atoms: set[int],
    rank: int,
    multiple_bond_count: int,
) -> bool:
    if not bool(getattr(mol, "HasProp", lambda _name: False)("_nispo_kekulized_aromatic_view")):
        return False
    if rank < 4 or multiple_bond_count < 5:
        return False
    if not _has_stereochemical_evidence(mol, Chem):
        return False

    hetero_atoms = 0
    external_attachment_atoms = 0
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "C":
            hetero_atoms += 1
        if any(nbr.GetIdx() not in atoms for nbr in atom.GetNeighbors()):
            external_attachment_atoms += 1

    if len(atoms) >= 30 and external_attachment_atoms == 0 and hetero_atoms >= 2:
        return True
    return 14 <= len(atoms) <= 18 and external_attachment_atoms >= 4 and hetero_atoms >= 3


def _use_stereo_hetero_kekulized_macrocycle_bridge_order(
    mol,
    Chem,
    atoms: set[int],
    rank: int,
    multiple_bond_count: int,
) -> bool:
    if not bool(getattr(mol, "HasProp", lambda _name: False)("_nispo_kekulized_aromatic_view")):
        return False
    if rank not in {4, 5}:
        return False
    if not 22 <= len(atoms) <= 24:
        return False
    if multiple_bond_count < 5:
        return False
    if not _has_stereochemical_evidence(mol, Chem):
        return False
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False

    hetero_atoms = 0
    external_attachment_atoms = 0
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol not in {"C", "N", "O"}:
            return False
        if symbol != "C":
            hetero_atoms += 1
        if any(nbr.GetIdx() not in atoms for nbr in atom.GetNeighbors()):
            external_attachment_atoms += 1

    return hetero_atoms >= 5 and 1 <= external_attachment_atoms <= 3


def _stereo_hetero_kekulized_macrocycle_bridge_preference_key(
    mol,
    seq: Sequence[int],
    core_atoms: set[int],
    bracket: Sequence[int | str],
) -> tuple:
    parsed = _bridge_sort_key(bracket)
    first_main = parsed[0][0] if len(parsed) > 0 else 99
    second_main = parsed[1][0] if len(parsed) > 1 else 0
    third_main = parsed[2][0] if len(parsed) > 2 else 99
    return (
        third_main != 1,
        -second_main,
        *_polycyclic_numbering_preference_key(mol, seq, core_atoms),
        first_main,
        _secondary_bridge_preference_key(bracket),
    )


def _large_hetero_kekulized_bridge_preference_key(
    mol,
    seq: Sequence[int],
    core_atoms: set[int],
    bracket: Sequence[int | str],
) -> tuple:
    parsed = _bridge_sort_key(bracket)
    max_secondary_start = max((left for _length, left, _right in parsed[3:]), default=0)
    return (
        -max_secondary_start,
        _substituent_locants(mol, seq, core_atoms),
        _secondary_bridge_preference_key(bracket),
        _hetero_locants(mol, seq),
        _unsaturation_contiguity_key(mol, seq),
    )


def _secondary_bridge_completion(
    graph: dict[int, set[int]],
    atoms: set[int],
    base_paths: Sequence[tuple[int, ...]],
    base_seq: Sequence[int],
    base_bracket: Sequence[int],
    rank: int,
) -> tuple[tuple[int, ...], tuple[int | str, ...]] | None:
    base_edges = set()
    for path in base_paths:
        for left, right in zip(path, path[1:]):
            base_edges.add(tuple(sorted((left, right))))
    extra_edges = [edge for edge in _graph_edges(graph) if edge not in base_edges]
    base_positions = {idx: loc for loc, idx in enumerate(base_seq, start=1)}
    used_edges: set[tuple[int, int]] = set()
    bridges = []
    for left, right in extra_edges:
        edge = (left, right)
        if edge in used_edges:
            continue
        if left in base_positions and right in base_positions:
            low, high = sorted((base_positions[left], base_positions[right]))
            bridges.append((0, low, high, ()))
            used_edges.add(edge)
            continue
        paths = _secondary_bridge_paths(graph, base_positions.keys(), extra_edges, used_edges, left, right)
        if len(paths) != 1:
            return None
        path = paths[0]
        start, end = path[0], path[-1]
        if start not in base_positions or end not in base_positions:
            return None
        bridge_edges = {tuple(sorted((a, b))) for a, b in zip(path, path[1:])}
        used_edges.update(bridge_edges)
        low, high = sorted((base_positions[start], base_positions[end]))
        internals = tuple(path[1:-1])
        bridges.append((len(internals), low, high, internals))

    if used_edges != set(extra_edges):
        return None
    seq = tuple(base_seq) + tuple(idx for bridge in bridges for idx in bridge[3])
    if len(seq) != len(atoms) or set(seq) != atoms:
        return None
    bracket = list(base_bracket)
    for length, low, high, _internals in sorted(bridges, key=lambda item: (-item[0], item[1], item[2])):
        bracket.append(f"{length}^{low},{high}")
    if len(bracket) != rank + 1:
        return None
    return seq, tuple(bracket)


def _secondary_bridge_base_path_sets(
    graph: dict[int, set[int]], atoms: set[int], a: int, b: int, *, allow_relaxed: bool
) -> list[tuple[tuple[int, ...], ...]]:
    exact = _bridge_paths(graph, atoms, a, b)
    if exact:
        return [tuple(exact)]
    if not allow_relaxed:
        return []

    paths = _simple_paths(graph, a, b, max_paths=36)
    candidates = []
    for selected in combinations(paths, 3):
        internals = [set(path[1:-1]) for path in selected]
        if any(left & right for i, left in enumerate(internals) for right in internals[i + 1 :]):
            continue
        covered = set().union(*internals, {a, b})
        if not covered <= atoms:
            continue
        candidates.append(selected)
    candidates.sort(
        key=lambda item: (
            -len(set().union(*(set(path[1:-1]) for path in item), {a, b})),
            sorted((len(path) - 2 for path in item), reverse=True),
            item,
        )
    )
    return list(dict.fromkeys(candidates))[:12]


def _ring_multiple_bond_count(mol, atoms: set[int]) -> int:
    count = 0
    for bond in mol.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        if left in atoms and right in atoms and bond.GetBondTypeAsDouble() > 1.0:
            count += 1
    return count


def _secondary_bridge_paths(
    graph: dict[int, set[int]],
    base_atoms,
    extra_edges: Sequence[tuple[int, int]],
    used_edges: set[tuple[int, int]],
    left: int,
    right: int,
) -> list[tuple[int, ...]]:
    base_atoms = set(base_atoms)
    available = set(extra_edges) - used_edges
    if left in base_atoms and right not in base_atoms:
        starts = [(left, right)]
    elif right in base_atoms and left not in base_atoms:
        starts = [(right, left)]
    else:
        return []
    paths = []
    for start, first in starts:
        stack = [(first, (start, first))]
        while stack:
            current, path = stack.pop()
            if current in base_atoms and current != start:
                paths.append(path)
                continue
            for nbr in sorted(graph[current]):
                edge = tuple(sorted((current, nbr)))
                if edge not in available or nbr in path:
                    continue
                stack.append((nbr, path + (nbr,)))
    return paths


def _ring_systems(mol, Chem) -> list[set[int]]:
    return [set(topology.atoms) for topology in _cached_ring_system_topologies(mol, Chem)]


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
    ) or (
        atom.GetSymbol() == "C" and atom.GetFormalCharge() in {-1, 1}
    ) or (
        atom.GetSymbol() == "B" and atom.GetFormalCharge() == -1
    )
    return (
        atom.GetSymbol() not in SKELETON_ATOMS
        or atom.GetIsAromatic()
        or not charge_allowed
        or atom.GetSymbol() not in {"C", "N", "O", "S", "Se", "Te", "B", "Si", "P"}
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
    paths = _simple_paths(graph, a, b, max_paths=12)
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


def _dominant_bridge_sort_key(bracket: Sequence[int | str]) -> tuple:
    return tuple((-length, left, right) for length, left, right in _bridge_sort_key(bracket))


def _secondary_bridge_preference_key(
    bracket: Sequence[int | str], *, saturated_alicyclic: bool = False
) -> tuple:
    if saturated_alicyclic:
        return (0, _blue_book_secondary_bridge_selection_key(bracket))

    nonzero_locant_one = 0
    for length, left, right in _bridge_sort_key(bracket):
        if length > 0 and left == 1:
            nonzero_locant_one += 1
    return (1, nonzero_locant_one, _dominant_bridge_sort_key(bracket))


def _blue_book_secondary_bridge_selection_key(bracket: Sequence[int | str]) -> tuple:
    """Approximate P-23.2.6.2 selection for saturated von Baeyer systems."""

    parsed = _bridge_sort_key(bracket)
    if len(parsed) < 4:
        return (_dominant_bridge_sort_key(bracket),)

    first, second, main_bridge = (item[0] for item in parsed[:3])
    base_atom_count = first + second + main_bridge + 2
    secondary = parsed[3:]
    independent = [
        (length, left, right)
        for length, left, right in secondary
        if left <= base_atom_count and right <= base_atom_count
    ]
    dependent = [item for item in secondary if item not in independent]
    independent_lengths = tuple(-length for length, _left, _right in independent)
    dependent_lengths = tuple(-length for length, _left, _right in dependent)
    all_secondary_locants = tuple(sorted(locant for _length, left, right in secondary for locant in (left, right)))
    citation_locants = tuple(locant for _length, left, right in secondary for locant in (left, right))
    return (
        -(first + second + 2),
        independent_lengths,
        len(dependent),
        dependent_lengths,
        abs(first - second),
        main_bridge == 0,
        all_secondary_locants,
        citation_locants,
        _dominant_bridge_sort_key(bracket),
    )


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
    return _polycyclic_parent_with_prefixes(mol, seq, parent)


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
    return _polycyclic_parent_with_prefixes(mol, seq, parent)


def _generic_polycyclic_parent(mol, seq: Sequence[int], bracket: Sequence[int | str]) -> str:
    root = ROOTS.get(len(seq))
    prefix = _generic_polycyclic_prefix(len(bracket) - 1)
    if not root or not prefix:
        return ""
    parent = _poly_parent_with_unsaturation(
        mol,
        seq,
        f"{prefix}[{'.'.join(str(value) for value in bracket)}]{root}",
        allow_unsaturation=True,
    )
    if not parent:
        return ""
    return _polycyclic_parent_with_prefixes(mol, seq, parent)


def _generic_polycyclic_prefix(cycle_count: int) -> str:
    multiplier = MULT.get(cycle_count)
    return f"{multiplier}cyclo" if multiplier and cycle_count >= 4 else ""


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
    return _polycyclic_parent_with_prefixes(mol, seq, parent)


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
    return _polycyclic_parent_with_prefixes(mol, seq, parent)


def _polycyclic_parent_with_prefixes(mol, seq: Sequence[int], parent: str) -> str:
    parent = _polycyclic_parent_with_charge_suffix(mol, seq, parent)
    if not parent:
        return ""
    replacement_locs: dict[str, list[str]] = defaultdict(list)
    lambda_locs: list[str] = []
    for loc, idx in enumerate(seq, start=1):
        lambda_loc = _polycyclic_lambda_locant(mol, idx, loc)
        replacement = _polycyclic_replacement_term(mol, idx)
        if replacement:
            replacement_locs[replacement].append(lambda_loc or str(loc))
        elif lambda_loc:
            lambda_locs.append(lambda_loc)

    parts = sorted(lambda_locs, key=_polycyclic_locant_sort_key)
    for replacement in sorted(replacement_locs):
        locants = sorted(replacement_locs[replacement], key=_polycyclic_locant_sort_key)
        if len(locants) == 1:
            parts.append(f"{locants[0]}-{replacement}")
        else:
            parts.append(f"{_join_polycyclic_locants(locants)}-{MULT.get(len(locants), '')}{replacement}")
    return "-".join(parts) + parent if parts else parent


def _polycyclic_parent_with_charge_suffix(mol, seq: Sequence[int], parent: str) -> str:
    charged_suffix_atoms = []
    for loc, idx in enumerate(seq, start=1):
        atom = mol.GetAtomWithIdx(idx)
        charge = atom.GetFormalCharge()
        if charge == 0:
            continue
        if atom.GetSymbol() == "C" and charge in {-1, 1}:
            charged_suffix_atoms.append((loc, charge))
            continue
        if atom.GetSymbol() == "B" and charge == -1:
            charged_suffix_atoms.append((loc, charge))
    if not charged_suffix_atoms:
        return parent
    if len(charged_suffix_atoms) != 1:
        return ""
    loc, charge = charged_suffix_atoms[0]
    stem = parent[:-1] if parent.endswith("e") else parent
    return f"{stem}-{loc}-{'ylium' if charge > 0 else 'ide'}"


def _polycyclic_lambda_locant(mol, idx: int, loc: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    symbol = atom.GetSymbol()
    if atom.GetFormalCharge() != 0:
        return ""
    if symbol == "N" and atom.GetTotalNumHs() == 0 and atom.GetNumRadicalElectrons():
        return f"{loc}lambda2"
    if symbol == "S" and atom.GetTotalNumHs() > 0:
        valence = int(atom.GetTotalValence())
        if valence in {4, 6}:
            return f"{loc}lambda{valence}"
    return ""


def _join_polycyclic_locants(locants: Sequence[str]) -> str:
    return ",".join(sorted(locants, key=_polycyclic_locant_sort_key))


def _polycyclic_locant_sort_key(locant: str) -> tuple[int, str]:
    digits = []
    for char in locant:
        if not char.isdigit():
            break
        digits.append(char)
    value = int("".join(digits)) if digits else 0
    return (value, locant)


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


def _unsaturation_contiguity_key(mol, seq: Sequence[int]) -> tuple[int, tuple[tuple[int, int], ...]]:
    pos = {idx: loc for loc, idx in enumerate(seq, start=1)}
    locants = []
    noncontiguous = 0
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
            if bond.GetBondTypeAsDouble() <= 1.0:
                continue
            low, high = sorted((pos[idx], pos[nidx]))
            locants.append((low, high))
            if high != low + 1:
                noncontiguous += 1
    return noncontiguous, tuple(sorted(locants))


def _noncontiguous_unsaturation_locants(mol, seq: Sequence[int]) -> tuple[tuple[int, int], ...]:
    pos = {idx: loc for loc, idx in enumerate(seq, start=1)}
    locants = []
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
            if bond.GetBondTypeAsDouble() <= 1.0:
                continue
            low, high = sorted((pos[idx], pos[nidx]))
            if high != low + 1:
                locants.append((low, high))
    return tuple(sorted(locants))


def _bridge_unsaturation_preference_key(
    mol, seq: Sequence[int], core_atoms: set[int]
) -> tuple[int, tuple[tuple[int, int], ...]]:
    if not mol.HasProp("_nispo_kekulized_aromatic_view"):
        return (1, ())
    if len(core_atoms) < 20:
        return (1, ())
    if _ring_multiple_bond_count(mol, set(core_atoms)) < 5:
        return (1, ())
    locants = _noncontiguous_unsaturation_locants(mol, seq)
    if len(locants) < 2:
        return (1, ())
    return (0, locants)


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
    parent = _append_polycyclic_parent_isotope_suffixes(mol, seq, parent)
    pos = {idx: str(loc) for loc, idx in enumerate(seq, start=1)}
    prefixes = []
    for idx in seq:
        locant = pos[idx]
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core_atoms:
                continue
            prefix = _simple_or_branch_prefix(mol, nidx, idx)
            if not prefix:
                return ""
            prefixes.append((locant, prefix))
    prefix_text = _format_prefixes(prefixes)
    return _join_prefix_parent(prefix_text, parent) if prefix_text else parent


def _append_polycyclic_parent_isotope_suffixes(mol, seq: Sequence[int], parent: str) -> str:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for locant, idx in enumerate(seq, start=1):
        atom = mol.GetAtomWithIdx(idx)
        isotope = atom.GetIsotope()
        if isotope:
            grouped[(atom.GetSymbol(), isotope)].append(locant)
    if not grouped:
        return parent
    suffixes = []
    for (symbol, isotope), locants in sorted(grouped.items()):
        count_text = str(len(locants)) if len(locants) > 1 else ""
        suffixes.append(f"{_join_locants(locants)}-{isotope}{symbol}{count_text}")
    return parent + "-" + "-".join(suffixes)


def _name_polycyclic_substituent(
    mol,
    Chem,
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
            prefix = _simple_or_branch_prefix(mol, nidx, idx)
            if not prefix:
                return ""
            prefixes.append((locant, prefix))
    prefix_text = _format_prefixes(prefixes)
    name = _join_prefix_parent(prefix_text, suffix) if prefix_text else suffix
    stereo_prefix = _polycyclic_substituent_stereo_prefix(mol, seq, Chem)
    return f"({stereo_prefix})-{name}" if stereo_prefix else name


def _polycyclic_substituent_stereo_prefix(mol, seq: Sequence[int], Chem) -> str:
    if not seq or _has_assigned_bond_stereo(mol, Chem):
        return ""
    locants = {idx: loc for loc, idx in enumerate(seq, start=1)}
    centers = [
        (idx, descriptor)
        for idx, descriptor in assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
        if idx in locants and descriptor in {"R", "S"}
    ]
    if not centers:
        return ""
    return ",".join(
        f"{locants[idx]}{descriptor}"
        for idx, descriptor in sorted(centers, key=lambda item: locants[item[0]])
    )


def _polycyclic_substituent_suffix(parent: str, locant: int) -> str:
    stem = parent[:-1] if parent.endswith("e") else parent
    return f"{stem}-{locant}-yl"


def _hetero_locants(mol, seq: Sequence[int]) -> list[int]:
    return [loc for loc, idx in enumerate(seq, start=1) if mol.GetAtomWithIdx(idx).GetSymbol() != "C"]


def _polycyclic_numbering_preference_key(mol, seq: Sequence[int], core_atoms: set[int]) -> tuple:
    hetero_locants = _hetero_locants(mol, seq)
    return (
        hetero_locants,
        _bridge_unsaturation_preference_key(mol, seq, core_atoms),
        _unsaturation_contiguity_key(mol, seq),
        _substituent_locants(mol, seq, core_atoms),
    )


def _substituent_locants(mol, seq: Sequence[int], core_atoms: set[int]) -> list[int]:
    locants = []
    for loc, idx in enumerate(seq, start=1):
        if any(nbr.GetIdx() not in core_atoms for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()):
            locants.append(loc)
    return locants
