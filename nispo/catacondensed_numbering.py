"""Graph validation and source-seeded peripheral numbering for catacondensed cores.

The graph determines fused faces, fusion edges, and the outer perimeter. It
does not determine the P-25.3.2.3 preferred drawing orientation. A caller must
therefore supply any numbering orientation as an explicit, sourced seed.
"""

from __future__ import annotations

from dataclasses import dataclass


_MAX_CORE_ATOMS = 60
_MAX_CYCLE_SEARCH_STEPS = 250_000
_HETERO_ORDER = (9, 17, 35, 53, 8, 16, 34, 52, 7, 15, 33, 51, 83,
                 14, 32, 50, 82, 5, 13, 31, 49, 81)


@dataclass(frozen=True, slots=True)
class PeripheralOrientationSeed:
    """Caller-proved clockwise boundary order from a P-25.3.2.3 drawing."""

    clockwise_order: tuple[int, ...]
    evidence: str


@dataclass(frozen=True, slots=True)
class PeripheralNumbering:
    seed: PeripheralOrientationSeed
    atom_locants: tuple[tuple[int, str], ...]
    heteroatom_locants: tuple[str, ...]
    fusion_carbon_locants: tuple[str, ...]
    criterion_key: tuple
    bond_order_facts: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True, slots=True)
class CatacondensedNumberingFrontier:
    admitted: bool
    reason: str | None
    core_atoms: frozenset[int] = frozenset()
    ring_faces: tuple[frozenset[int], ...] = ()
    fusion_edges: frozenset[frozenset[int]] = frozenset()
    perimeter_edges: frozenset[frozenset[int]] = frozenset()
    perimeter_cycle: tuple[int, ...] = ()
    candidates: tuple[PeripheralNumbering, ...] = ()


def peripheral_numbering_frontier(mol, *, core_atoms=None, orientation_seeds=(),
                                  allow_three_face_n_junction=False,
                                  allow_multi_face_n_junction=False,
                                  allow_larger_system=False,
                                  allow_shared_carbon_junctions=False,
                                  allow_peri_fused=False):
    """Return validated catacondensed graph facts and source-seeded numbering.

    `core_atoms` defaults to atoms reported as ring members by RDKit's graph
    cycle analysis; callers may provide an explicit subset when exocyclic
    groups are present. Each seed is an explicit clockwise outer-cycle order
    supported by the caller's P-25.3.2.3 geometry proof. With no seeds, the
    graph frontier is returned without a selected locant mapping.

    The default scope is a neutral, connected, edge-fused graph whose
    chordless ring faces form a tree and whose face-boundary edges give one
    cycle through every core atom. The opt-in shared-carbon path also proves
    a degree-four carbon shared by three faces when it is peripheral.
    The graph search does not infer a preferred orientation from atom
    indices, SMILES order, bond insertion order, or a canonical traversal.
    """
    if mol is None:
        return CatacondensedNumberingFrontier(False, "missing molecule")
    from rdkit import Chem

    if core_atoms is None:
        core = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
    else:
        try:
            supplied_core = tuple(core_atoms)
        except Exception:
            return CatacondensedNumberingFrontier(False, "invalid core atom set")
        if any(not isinstance(i, int) or isinstance(i, bool) for i in supplied_core):
            return CatacondensedNumberingFrontier(False, "core atom IDs must be integers")
        core = frozenset(supplied_core)
    if (len(core) < 3 or (not allow_larger_system and len(core) > _MAX_CORE_ATOMS)
            or any(i < 0 or i >= mol.GetNumAtoms() for i in core)):
        return CatacondensedNumberingFrontier(False, "core atom bound or index invalid")
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 1 for i in core):
        return CatacondensedNumberingFrontier(False, "hydrogen cannot be a skeletal core atom")
    if any(mol.GetAtomWithIdx(i).GetFormalCharge()
           or mol.GetAtomWithIdx(i).GetIsotope()
           or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
           or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in core):
        return CatacondensedNumberingFrontier(False, "charged, isotopic, radical, or stereo core unsupported")
    atoms = sorted(core)
    adjacency = {i: set() for i in atoms}
    bond_orders = {}
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left not in core or right not in core:
            continue
        if (int(bond.GetStereo()) or bond.GetBondDir() != Chem.BondDir.NONE
                or bond.GetBondTypeAsDouble() not in (1.0, 1.5, 2.0)):
            return CatacondensedNumberingFrontier(False, "unsupported core bond state")
        adjacency[left].add(right)
        adjacency[right].add(left)
        bond_orders[frozenset((left, right))] = float(bond.GetBondTypeAsDouble())
    if any(len(adjacency[i]) not in (2, 3)
           and not (allow_shared_carbon_junctions and len(adjacency[i]) == 4
                    and mol.GetAtomWithIdx(i).GetAtomicNum() == 6) for i in atoms):
        return CatacondensedNumberingFrontier(False, "core degree outside nonfusion/fusion scope")
    if not _connected(adjacency):
        return CatacondensedNumberingFrontier(False, "disconnected core")

    cycle_rank = len(bond_orders) - len(core) + 1
    if cycle_rank < 2 or cycle_rank > len(core) - 2:
        return CatacondensedNumberingFrontier(False, "cycle rank outside fused-core scope")
    faces = _chordless_cycles(adjacency)
    if faces is None:
        return CatacondensedNumberingFrontier(False, "bounded cycle-face search exceeded")
    if len(faces) != cycle_rank and allow_peri_fused:
        faces = _peri_face_partition(adjacency, faces, cycle_rank)
    if faces is None or len(faces) != cycle_rank:
        return CatacondensedNumberingFrontier(False, "chordless faces do not uniquely span cycle space")
    edge_owners = {}
    atom_faces = {i: set() for i in atoms}
    for face_index, face in enumerate(faces):
        for atom in face:
            atom_faces[atom].add(face_index)
        for a, b in zip(face, face[1:] + face[:1]):
            edge_owners.setdefault(frozenset((a, b)), []).append(face_index)
    if any(not memberships or len(memberships) > 2
           and not (allow_shared_carbon_junctions and len(memberships) == 3
                    and len(adjacency[i]) == 4 and mol.GetAtomWithIdx(i).GetAtomicNum() == 6)
           and not (allow_peri_fused and len(memberships) == 3
                    and len(adjacency[i]) == 3 and mol.GetAtomWithIdx(i).GetAtomicNum() in {6, 7})
           for i, memberships in atom_faces.items()):
        return CatacondensedNumberingFrontier(False, "ring faces share an unsupported atom pattern")
    fusion_edges = frozenset(edge for edge, owners in edge_owners.items() if len(owners) == 2)
    if any(len(owners) != 1 and len(owners) != 2 for owners in edge_owners.values()):
        return CatacondensedNumberingFrontier(False, "edge belongs to too many ring faces")
    if not allow_peri_fused and len(fusion_edges) != cycle_rank - 1:
        return CatacondensedNumberingFrontier(False, "face junction count is not catacondensed")
    face_graph = [set() for _ in faces]
    for edge in fusion_edges:
        left, right = (edge_owners[edge])
        face_graph[left].add(right)
        face_graph[right].add(left)
    if not _connected({i: face_graph[i] for i in range(len(faces))}):
        return CatacondensedNumberingFrontier(False, "ring-face graph is disconnected")
    def peri_interior(i):
        return (allow_peri_fused and len(atom_faces[i]) == len(adjacency[i]) == 3
                and sum(i in edge for edge in fusion_edges) == 3)
    if any(sum(i in edge for edge in fusion_edges) != max(0, len(atom_faces[i])-1)
           and not peri_interior(i) for i in atoms):
        return CatacondensedNumberingFrontier(False, "fusion-atom incidence is not catacondensed")
    if any(len(adjacency[i]) != len(atom_faces[i])+1 and not peri_interior(i) for i in atoms):
        return CatacondensedNumberingFrontier(False, "degree and ring-face fusion facts disagree")
    if any(len(edge) != 2 or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6,7} for i in edge)
           for edge in fusion_edges):
        return CatacondensedNumberingFrontier(False, "only carbon/ordinary nitrogen fusion edges are supported")
    fusion_n = {i for edge in fusion_edges for i in edge
                if mol.GetAtomWithIdx(i).GetAtomicNum()==7}
    # The original-graph callers may separately prove a bounded three-face
    # component construction. P-25.3.3.1.1 includes peripheral fusion
    # heteroatoms in ordinary numbering; the same two-face incidence and
    # neutral trivalent N checks still apply at every such junction.
    n_face_counts = {2, 3} if allow_three_face_n_junction else {2}
    if fusion_n and ((not allow_multi_face_n_junction and cycle_rank not in n_face_counts)
                     or any(mol.GetAtomWithIdx(i).GetTotalValence()!=3
                                         for i in fusion_n)):
        return CatacondensedNumberingFrontier(False, "neutral N junction proof unavailable")

    perimeter_edges = frozenset(edge for edge, owners in edge_owners.items() if len(owners) == 1)
    perimeter_graph = {i: set() for i in atoms}
    for edge in perimeter_edges:
        a, b = tuple(edge)
        perimeter_graph[a].add(b)
        perimeter_graph[b].add(a)
    interior = {i for i, neighbors in perimeter_graph.items() if not neighbors}
    if any(not peri_interior(i) and not (
            allow_shared_carbon_junctions and mol.GetAtomWithIdx(i).GetAtomicNum() == 6
            and len(adjacency[i]) == 4 and len(atom_faces[i]) == 3) for i in interior):
        return CatacondensedNumberingFrontier(False, "unsupported interior junction")
    perimeter_graph = {i: neighbors for i, neighbors in perimeter_graph.items() if neighbors}
    if (any(len(neighbors) != 2 for neighbors in perimeter_graph.values())
            or not _connected(perimeter_graph)):
        return CatacondensedNumberingFrontier(False, "ring faces do not define one all-atom perimeter")
    perimeter_cycle = _walk_cycle(perimeter_graph)
    if perimeter_cycle is None or len(perimeter_cycle) != len(core)-len(interior):
        return CatacondensedNumberingFrontier(False, "perimeter cycle is incomplete")

    candidates = []
    try:
        seeds = tuple(orientation_seeds)
    except Exception:
        return CatacondensedNumberingFrontier(False, "orientation seeds are not iterable")
    for seed in seeds:
        if interior and (not allow_peri_fused
                         or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7} for i in interior)):
            return CatacondensedNumberingFrontier(False, "interior-atom numbering requires a separate proof")
        if (not isinstance(seed, PeripheralOrientationSeed)
                or not isinstance(seed.evidence, str)
                or not seed.evidence.strip()):
            return CatacondensedNumberingFrontier(False, "orientation seed lacks source evidence")
        if (not isinstance(seed.clockwise_order, tuple)
                or any(not isinstance(i, int) or isinstance(i, bool)
                       for i in seed.clockwise_order)):
            return CatacondensedNumberingFrontier(False, "orientation seed atom IDs must be integer tuple")
        mapped = _number_seed(mol, core, atom_faces, perimeter_edges, seed, interior=interior)
        if not mapped:
            return CatacondensedNumberingFrontier(False, "orientation seed is not a valid source-oriented perimeter",
                                                 core, faces, fusion_edges, perimeter_edges,
                                                 perimeter_cycle)
        candidates.extend(mapped)
    if candidates:
        best = min(candidate.criterion_key for candidate in candidates)
        candidates = [candidate for candidate in candidates if candidate.criterion_key == best]
        candidates = list({candidate.atom_locants: candidate for candidate in candidates}.values())
    return CatacondensedNumberingFrontier(
        True,
        "preferred orientation geometry required" if not seeds else None,
        core,
        faces,
        fusion_edges,
        perimeter_edges,
        perimeter_cycle,
        tuple(candidates),
    )


def _connected(graph):
    if not graph:
        return False
    start = next(iter(graph))
    seen, stack = set(), [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph[node] - seen)
    return len(seen) == len(graph)


def _canonical_cycle(path):
    rotations = []
    for seq in (tuple(path), tuple(reversed(path))):
        for offset in range(len(seq)):
            rotations.append(seq[offset:] + seq[:offset])
    return min(rotations)


def _chordless_cycles(adjacency):
    found = set()
    steps = 0
    for start in sorted(adjacency):
        stack = [(start, (start,), frozenset((start,)))]
        while stack:
            current, path, used = stack.pop()
            steps += 1
            if steps > _MAX_CYCLE_SEARCH_STEPS:
                return None
            if len(path) >= 3 and start in adjacency[current]:
                # The closing edge would be a chord in any extension.
                found.add(_canonical_cycle(path))
                continue
            earlier = used - {start, current}
            for nxt in adjacency[current]:
                # An edge to an earlier noninitial vertex can only be a chord.
                if nxt > start and nxt not in used and earlier.isdisjoint(adjacency[nxt]):
                    stack.append((nxt, path + (nxt,), used | {nxt}))
    return tuple(sorted(found))


def _peri_face_partition(adjacency, cycles, rank):
    """Select a unique simple-perimeter face partition, if graph-proved."""
    from math import comb

    if not cycles or len(cycles) < rank or comb(len(cycles), rank) > _MAX_CYCLE_SEARCH_STEPS:
        return None
    # Precompute the edge set of each face. During a partial selection an
    # edge seen twice cannot be used again, and a missing edge that no
    # remaining face contains makes that branch impossible. These are the
    # same ownership requirements as the original complete-combination test.
    edges = tuple(sorted({tuple(sorted((a, b))) for a, neighbors in adjacency.items() for b in neighbors}))
    edge_numbers = {edge: i for i, edge in enumerate(edges)}
    masks = []
    for cycle in cycles:
        mask = 0
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            index = edge_numbers.get(tuple(sorted((a, b))))
            if index is None or mask & (1 << index):
                mask = None
                break
            mask |= 1 << index
        masks.append(mask)
    all_edges = (1 << len(edges)) - 1
    suffix = [0] * (len(masks) + 1)
    for i in range(len(masks) - 1, -1, -1):
        suffix[i] = suffix[i + 1] | (masks[i] or 0)
    valid = []

    def visit(start, needed, covered, doubled, chosen):
        if len(valid) > 1 or len(masks) - start < needed or covered | suffix[start] != all_edges:
            return
        if needed:
            for index in range(start, len(masks) - needed + 1):
                mask = masks[index]
                if mask is None or mask & doubled:
                    continue
                visit(index + 1, needed - 1, covered | mask, doubled | (covered & mask), chosen + (index,))
                if len(valid) > 1:
                    return
            return
        if covered != all_edges:
            return
        selected = tuple(cycles[index] for index in chosen)
        boundary = {i: set() for i in adjacency}
        singles = all_edges ^ doubled
        while singles:
            bit = singles & -singles
            singles ^= bit
            a, b = edges[bit.bit_length() - 1]
            boundary[a].add(b)
            boundary[b].add(a)
        interior = {i for i, neighbors in boundary.items() if not neighbors}
        boundary = {i: neighbors for i, neighbors in boundary.items() if neighbors}
        if (not boundary or any(len(row) != 2 for row in boundary.values())
                or not _connected(boundary)):
            return
        if any(len(adjacency[i]) != 3 or sum(i in cycle for cycle in selected) != 3
               for i in interior):
            return
        if any(len(adjacency[i]) != sum(i in cycle for cycle in selected)+1 for i in boundary):
            return
        valid.append(selected)

    visit(0, rank, 0, 0, ())
    return valid[0] if len(valid) == 1 else None


def _walk_cycle(graph):
    start = min(graph)
    first = min(graph[start])
    path = [start, first]
    while len(path) < len(graph):
        choices = graph[path[-1]] - {path[-2]}
        if len(choices) != 1:
            return None
        nxt = next(iter(choices))
        if nxt in path:
            return None
        path.append(nxt)
    return tuple(path) if start in graph[path[-1]] else None


def _locant_key(text):
    import re
    match = re.fullmatch(r'(\d+)([a-z]*)(?:\^(\d+))?', str(text))
    return (int(match[1]), match[2], int(match[3] or 0)) if match else None


def _number_seed(mol, core, atom_faces, perimeter_edges, seed, *, interior=frozenset()):
    order = tuple(seed.clockwise_order)
    if (len(order) != len(core)-len(interior) or set(order) != set(core)-set(interior)
            or any(frozenset((a, b)) not in perimeter_edges
                   for a, b in zip(order, order[1:] + order[:1]))):
        return ()
    fusion_c = {i for i in core-interior if len(atom_faces[i]) >= 2
                and mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
    if order[0] in fusion_c:
        return ()
    labels = {}
    number = 0
    previous_number = None
    letter_index = 0
    for atom_idx in order:
        if atom_idx in fusion_c:
            if previous_number is None:
                return ()
            label = previous_number + chr(ord("a") + letter_index)
            letter_index += 1
            labels[atom_idx] = label
        else:
            number += 1
            previous_number = str(number)
            letter_index = 0
            labels[atom_idx] = previous_number
    if number < 1:
        return ()
    interior_hetero = {i for i in interior if mol.GetAtomWithIdx(i).GetAtomicNum() != 6}
    interior_carbon = set(interior) - interior_hetero
    orders = _interior_hetero_orders(mol, core, interior_hetero, labels) if interior_hetero else ((),)
    carbon_maps = _interior_carbon_mappings(mol, core, interior_carbon, labels) if interior_carbon else ({},)
    if not orders or not carbon_maps:
        return ()
    proximity = (_interior_proximity_key(mol, core, interior_hetero, labels, atom_faces)
                 if interior_hetero else ())
    if proximity is None:
        return ()
    bond_facts = tuple(sorted((min(a, b), max(a, b), float(bond.GetBondTypeAsDouble()))
                              for bond in mol.GetBonds()
                              if bond.GetBeginAtomIdx() in core and bond.GetEndAtomIdx() in core
                              for a, b in [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())]))
    result = []
    from itertools import product
    for order, carbon_map in product(orders, carbon_maps):
        numbered = dict(labels)
        for offset, atom_idx in enumerate(order, 1):
            numbered[atom_idx] = str(number+offset)
        numbered.update(carbon_map)
        hetero = [(mol.GetAtomWithIdx(i).GetAtomicNum(), numbered[i]) for i in core
                  if mol.GetAtomWithIdx(i).GetAtomicNum() != 6]
        hetero_set = tuple(sorted((loc for _, loc in hetero), key=_locant_key))
        priority = []
        for element in _HETERO_ORDER:
            priority.append(tuple(sorted((loc for atomic_num, loc in hetero if atomic_num == element),
                                         key=_locant_key)))
        fusion_locs = tuple(sorted((numbered[i] for i in fusion_c), key=_locant_key))
        fusion_hetero = tuple(tuple(sorted((numbered[i] for i in core
            if len(atom_faces[i]) >= 2 and mol.GetAtomWithIdx(i).GetAtomicNum()==element),
            key=_locant_key)) for element in _HETERO_ORDER)
        # P-25.3.3.1.2(a-e): after peripheral hetero/fusion locants,
        # compare proximity of interior heteroatoms to peripheral junctions.
        criteria_key = (tuple(_locant_key(loc) for loc in hetero_set),
                        tuple(tuple(_locant_key(loc) for loc in group) for group in priority),
                        tuple(_locant_key(loc) for loc in fusion_locs),
                        tuple(tuple(_locant_key(loc) for loc in group) for group in fusion_hetero),
                        proximity, tuple(sorted(map(_locant_key, carbon_map.values()))))
        result.append(PeripheralNumbering(seed, tuple(sorted(numbered.items())), hetero_set,
                                         fusion_locs, criteria_key, bond_facts))
    return tuple(result)


def _interior_proximity_key(mol, core, interior, labels, atom_faces):
    """P-25.3.3.1.2(e): distance from ordered peripheral junctions."""
    from collections import deque

    distance = {atom: 0 for atom in interior}
    queue = deque(interior)
    while queue:
        atom = queue.popleft()
        for neighbor in mol.GetAtomWithIdx(atom).GetNeighbors():
            nxt = neighbor.GetIdx()
            if nxt in core and nxt not in distance:
                distance[nxt] = distance[atom]+1
                queue.append(nxt)
    junctions = sorted((atom for atom in labels if len(atom_faces[atom]) >= 2),
                       key=lambda atom: _locant_key(labels[atom]))
    if not junctions or any(atom not in distance for atom in junctions):
        return None
    return tuple(distance[atom] for atom in junctions)


def _interior_hetero_orders(mol, core, interior, labels):
    """P-25.3.3.2: rank each interior heteroatom by its nearest periphery."""
    from collections import defaultdict, deque
    from itertools import permutations, product
    from math import factorial, prod

    groups = defaultdict(list)
    for atom in interior:
        queue, distance, nearest, boundary = deque([atom]), {atom: 0}, None, []
        while queue:
            current = queue.popleft()
            if nearest is not None and distance[current] > nearest:
                break
            if current in labels:
                nearest = distance[current]
                boundary.append(labels[current])
                continue
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                nxt = neighbor.GetIdx()
                if nxt in core and nxt not in distance:
                    distance[nxt] = distance[current]+1
                    queue.append(nxt)
        if not boundary:
            return ()
        element = mol.GetAtomWithIdx(atom).GetAtomicNum()
        if element not in _HETERO_ORDER:
            return ()
        groups[(min(map(_locant_key, boundary)), _HETERO_ORDER.index(element))].append(atom)
    ordered_groups = [groups[key] for key in sorted(groups)]
    if prod(factorial(len(group)) for group in ordered_groups) > _MAX_CYCLE_SEARCH_STEPS:
        return ()
    return tuple(tuple(atom for group in choice for atom in group)
                 for choice in product(*(tuple(permutations(group)) for group in ordered_groups)))


def _interior_carbon_mappings(mol, core, interior, labels):
    """P-25.3.3.3 shortest-path superscript locants for interior carbons."""
    from collections import deque

    choices = {}
    for atom in interior:
        queue, distance, nearest, boundary = deque([atom]), {atom: 0}, None, []
        while queue:
            current = queue.popleft()
            if nearest is not None and distance[current] > nearest:
                break
            if current in labels:
                nearest = distance[current]
                boundary.append(labels[current])
                continue
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                nxt = neighbor.GetIdx()
                if nxt in core and nxt not in distance:
                    distance[nxt] = distance[current]+1
                    queue.append(nxt)
        if nearest is None or not boundary:
            return ()
        choices[atom] = tuple(sorted({label+'^'+str(nearest) for label in boundary}, key=_locant_key))
    atoms = tuple(sorted(interior))
    work, candidates = [0], []

    def assign(position, mapping, used):
        work[0] += 1
        if work[0] > _MAX_CYCLE_SEARCH_STEPS:
            raise OverflowError
        if position == len(atoms):
            key = tuple(sorted(map(_locant_key, mapping.values())))
            candidates.append((key, dict(mapping)))
            return
        atom = atoms[position]
        for label in choices[atom]:
            if label not in used:
                mapping[atom] = label
                assign(position+1, mapping, used | {label})
                del mapping[atom]

    try:
        assign(0, {}, set())
    except OverflowError:
        return ()
    if not candidates:
        return ()
    best = min(key for key, _mapping in candidates)
    return tuple(mapping for key, mapping in candidates if key == best)

