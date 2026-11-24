"""P-25.1.2.5 polynaphthylenes, including the two-unit fusion exception."""
from functools import lru_cache

from .number_terms import MAX_CARBON_CHAIN_LENGTH, basic_number_term


@lru_cache(maxsize=16)
def _locants(units):
    """One peripheral numbering (P-25.3.3) and its two fusion edges per arm.

    In each ten-atom repeat the perimeter passes four outer CH positions,
    a junction, one CH, two junctions, one CH and a junction. Joining the
    two pairs of junctions across each repeat boundary closes the two
    hexagons. The unshared edges of the central 2n-cycle alternate with
    the edges fused to the 2,3-positions of the naphthalene units.
    """
    labels = []
    number = 0
    size = 10 * units
    for i in range(size):
        junction = i % 10 in (4, 6, 7, 9)
        if junction:
            labels.append(f'{number}b' if i % 10 == 7 else f'{number}a')
        else:
            number += 1
            labels.append(str(number))
    return tuple(labels)


def _series_name(units):
    term = basic_number_term(units)
    return term + 'naphthylene' if units >= 3 and term else ''


def _ordinary(mol):
    return (not mol.GetStereoGroups() and not any(
        a.GetFormalCharge() or a.HasQuery() or a.GetNumRadicalElectrons()
        or a.GetChiralTag() for a in mol.GetAtoms())
        and not any(b.GetStereo() or b.HasQuery() for b in mol.GetBonds()))


def _mancude_core(mol, Chem, core):
    # Bond orders may represent another Kekulé form of the same mancude
    # hydride. Every core carbon must contribute exactly one internal pi bond.
    graph = Chem.Mol(mol)
    try:
        Chem.Kekulize(graph, clearAromaticFlags=True)
    except (ValueError, RuntimeError):
        return False
    for i in core:
        bonds = [b.GetBondType() for b in graph.GetAtomWithIdx(i).GetBonds() if b.GetOtherAtomIdx(i) in core]
        if bonds.count(Chem.BondType.DOUBLE) != 1 or any(
                order not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE) for order in bonds):
            return False
    return True


def _core_perimeter(rings, units):
    """Find the numbered outer circuit and prove both fusion bonds of every arm."""
    counts = {}
    for ring in rings:
        for a, b in zip(ring, ring[1:] + ring[:1]):
            edge = (min(a, b), max(a, b))
            counts[edge] = counts.get(edge, 0) + 1
    boundary, chords = {}, set()
    for (a, b), count in counts.items():
        if count == 1:
            boundary.setdefault(a, []).append(b)
            boundary.setdefault(b, []).append(a)
        elif count == 2:
            chords.add((a, b))
        else:
            return ()
    size = 10 * units
    if len(boundary) != size or len(chords) != 2 * units or any(len(ns) != 2 for ns in boundary.values()):
        return ()
    start = next(iter(boundary))
    path = [start]
    previous, current = start, boundary[start][0]
    while current != start:
        path.append(current)
        left, right = boundary[current]
        previous, current = current, right if left == previous else left
    if len(path) != size:
        return ()
    # The frame repeats every ten perimeter atoms. Testing one repeat in
    # either direction suffices even for the largest numerical prefixes.
    for order in (path, path[::-1]):
        for offset in range(10):
            seq = order[offset:] + order[:offset]
            def chord(i, j):
                a, b = seq[i], seq[j]
                return (min(a, b), max(a, b)) in chords
            if all(chord(i + 4, i - 1) and chord(i + 6, i - 3) for i in range(0, size, 10)):
                return tuple(seq)
    return ()


def _numberings(perimeter):
    # Reflection sends position i to 3-i, exchanging the two ends of the
    # four consecutive outer CH positions. All other equivalent numberings
    # differ by an integral number of ten-atom arms.
    reflected = perimeter[3::-1] + perimeter[:3:-1]
    for order in (perimeter, reflected):
        for start in range(0, len(order), 10):
            yield order[start:] + order[:start]


def _parent_frame(mol, Chem, rings):
    if len(rings) < 5 or len(rings) % 2 == 0:
        return None
    units = (len(rings) - 1) // 2
    if any(len(r) not in (6, 2 * units) for r in rings):
        return None
    core = frozenset(i for ring in rings for i in ring)
    if len(core) != 10 * units or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
            or mol.GetAtomWithIdx(i).GetTotalValence() != 4 for i in core):
        return None
    name = 'dibenzo[b,h]biphenylene' if units == 2 else _series_name(units)
    perimeter = _core_perimeter(rings, units)
    if not name or not perimeter or not _mancude_core(mol, Chem, core):
        return None
    return name, core, perimeter


def fusion_component(name):
    """Numbered P-25 component for the fusion parent and attached-prefix forms."""
    from rdkit import Chem
    from .fusion_components import component_from_graph

    if name.endswith('naphthyleno'):
        name = name[:-1] + 'e'
    units = next((n for n in range(3, MAX_CARBON_CHAIN_LENGTH + 1) if _series_name(n) == name), None)
    if units is None:
        return None
    labels = _locants(units)
    size = len(labels)
    graph = Chem.RWMol()
    for _ in labels:
        graph.AddAtom(Chem.Atom(6))
    # Alternating peripheral bonds give a perfect matching for every size.
    for i in range(size):
        graph.AddBond(i, (i + 1) % size, Chem.BondType.DOUBLE if i % 2 == 0 else Chem.BondType.SINGLE)
    for i in range(0, size, 10):
        graph.AddBond(i + 4, (i - 1) % size, Chem.BondType.SINGLE)
        graph.AddBond(i + 6, (i - 3) % size, Chem.BondType.SINGLE)
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return component_from_graph(mol, enumerate(labels))


def _benzannulated_parent(mol, Chem, rings):
    """Recognize first-order benzo fusions and number all attachments together."""
    system = frozenset(i for ring in rings for i in ring)
    size, faces = len(system), len(rings)
    # A central 2n-cycle with n naphthalene arms has 10n atoms and 2n+1
    # faces. Each terminal benzo fusion adds four atoms and one face.
    twice_units = size - 4 * (faces - 1)
    units = twice_units // 2
    extra = faces - 1 - 2 * units
    if twice_units % 2 or units < 3 or extra < 1 or not _series_name(units):
        return ''
    if (any(len(r) not in (6, 2 * units) for r in rings)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   or mol.GetAtomWithIdx(i).GetTotalValence() != 4 for i in system)):
        return ''
    by_edge, adjacent = {}, [set() for _ in rings]
    for k, ring in enumerate(rings):
        for a, b in zip(ring, ring[1:] + ring[:1]):
            edge = (min(a, b), max(a, b))
            if edge in by_edge:
                other = by_edge[edge]
                adjacent[k].add(other)
                adjacent[other].add(k)
            else:
                by_edge[edge] = k
    all_faces = set(range(faces))
    positions = []
    for center, inner in enumerate(adjacent):
        if len(rings[center]) != 2 * units or len(inner) != units:
            continue
        if any(len(adjacent[i]) != 2 for i in inner):
            continue
        outer = set.union(*(adjacent[i] for i in inner)) - {center}
        core_faces = {center} | inner | outer
        leaves = all_faces - core_faces
        if (len(outer) != units or len(leaves) != extra
                or any(len(adjacent[i]) != 1 or not adjacent[i] <= outer for i in leaves)):
            continue
        perimeter = _core_perimeter([rings[k] for k in core_faces], units)
        if not perimeter:
            continue
        core = frozenset(perimeter)
        ports = [frozenset(rings[k]) & core for k in leaves]
        if any(len(pair) != 2 for pair in ports):
            continue
        edges = {frozenset((a, b)): j for j, (a, b) in
                 enumerate(zip(perimeter, perimeter[1:] + perimeter[:1]))}
        if not all(pair in edges for pair in ports):
            continue
        sites = [edges[pair] for pair in ports]
        for offset in range(0, len(perimeter), 10):
            for oriented in (sites, [2 - j for j in sites]):
                positions.append(tuple(sorted((j - offset) % len(perimeter) for j in oriented)))
    if not positions or not _mancude_core(mol, Chem, system):
        return ''
    # FR-4.1 numbers a second circuit of letters a1, b1, ... after z.
    sites = min(positions)
    labels = [chr(ord('a') + j % 26) + (str(j // 26) if j >= 26 else '') for j in sites]
    multiplier = basic_number_term(extra) if extra > 1 else ''
    if extra > 1 and not multiplier:
        return ''
    name = multiplier + 'benzo[' + ','.join(labels) + ']' + _series_name(units)
    # The full annulated hydride receives a new peripheral numbering.
    can_number = units in (3, 4, 5)
    return name, system, () if can_number else None


@lru_cache(maxsize=64)
def _annulated_numberings(smiles):
    """Preferred peripheral numberings in the canonical bare hydride's atom order."""
    from rdkit import Chem
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fused_orientation import fused_orientation_frontier

    skeleton = Chem.MolFromSmiles(smiles)
    sizes = {len(r) for r in skeleton.GetRingInfo().AtomRings()}
    if not sizes <= {6, 8, 10}:
        return ()
    orientation = fused_orientation_frontier(skeleton, allow_larger_system=True,
                                             polynaphthylene_core=bool(sizes & {8, 10}))
    if not orientation.orientation_seeds:
        return ()
    numbered = peripheral_numbering_frontier(skeleton,
        orientation_seeds=orientation.orientation_seeds, allow_larger_system=True)
    return tuple(candidate.atom_locants for candidate in numbered.candidates)


def _substitution_maps(mol, Chem, core, perimeter):
    if perimeter is None:
        return ()
    if perimeter:
        locants = _locants(len(perimeter) // 10)
        return (dict(zip(match, locants)) for match in _numberings(perimeter))
    # Annulation changes the numbering of the whole system. Compute geometry
    # once per bare hydride, retaining every tied numbering. The SMILES output
    # permutation maps each cached skeletal atom back to this input graph.
    from .functional_parent_extensions import capped

    skeleton, original_ids = capped(mol, Chem, core)
    if skeleton is None:
        return ()
    for a in skeleton.GetAtoms():
        if a.GetAtomMapNum():
            a.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(skeleton, isomericSmiles=False)
    order = skeleton.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']
    original = tuple(original_ids[i] for i in order)
    return ({original[i]: loc for i, loc in pairs} for pairs in _annulated_numberings(smiles))


def polynaphthylene_decision(mol, Chem, rings):
    if not _ordinary(mol):
        return None
    from .naming_methods import MethodDecision
    from .retained_hydrocarbon_derivatives import ring_pendants, ring_substitution

    scope = frozenset(range(mol.GetNumAtoms()))
    labels = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetIsotope())
    rule = 'P-25.1.2.5/P-25.3.3'
    parent = _parent_frame(mol, Chem, rings) or _benzannulated_parent(mol, Chem, rings)
    if not parent:
        # A separate carbocycle can be a substituent on the senior fused
        # ring system. Only admit one polycyclic parent and monocyclic sides.
        from .mapped_ring_parents import ring_systems
        systems = ring_systems(mol)
        fused = [s for s in systems if sum(r[0] in s for r in rings) > 1]
        if len(systems) < 2 or len(fused) != 1:
            return None
        rings = tuple(r for r in rings if r[0] in fused[0])
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for s in systems if s != fused[0] for i in s):
            return None
        parent = _parent_frame(mol, Chem, rings) or _benzannulated_parent(mol, Chem, rings)
        if not parent:
            return None
    name, core, perimeter = parent
    if scope == core and not labels:
        return MethodDecision(name, 'fused-parent', frozenset(), scope, rule,
                              component_roles=(('fused-parent', core, ()),))
    branch_labels = labels - core
    parts = ring_pendants(mol, Chem, core, branch_isotopes=branch_labels)
    if parts is None:
        return None
    rows = []
    for mapping in _substitution_maps(mol, Chem, core, perimeter):
        row = ring_substitution(name, mapping, parts, annotated_graph=mol if labels else None,
                                branch_isotopes=bool(branch_labels))
        if row is not None:
            rows.append((*row, tuple(mapping.items())))
    if not rows:
        return None
    best = min(row[0] for row in rows)
    winners = [row for row in rows if row[0] == best]
    if len({row[1] for row in winners}) != 1:
        return None
    _, text, centers, mapping = winners[0]
    roles = (('fused-parent', core, mapping),) + tuple(
        ('pendant', p.atoms, ((p.root, p.host),)) for p in parts)
    return MethodDecision(text, 'fused-parent', centers, scope, rule, mapping, component_roles=roles)
