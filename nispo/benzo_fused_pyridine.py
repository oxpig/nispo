"""P-25 fusion frontier for benzochalcogen–azine ring systems.

This covers a neutral, fully aromatic 5/6/6 face tree with one furan or
thiophene face, one benzene face and one nitrogen-containing six-membered
face, with disjoint six-membered faces. The benzochalcogen unit is one
component under P-25.3.5. Carbon-only fusion edges and distinct six-faces
exclude a senior retained two-face component in this construction.
Original substituent, annotation and H-state proofs remain with the caller.
"""
from functools import lru_cache

METHOD = "component-fused-benzofuran-pyridine"


def source_supported(mol):
    """Keep unproved source annotations and characteristic groups out."""
    if mol.GetStereoGroups() or any(
        atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumRadicalElectrons()
        or int(atom.GetChiralTag()) or atom.HasQuery() or atom.GetAtomMapNum()
        for atom in mol.GetAtoms()
    ) or any(int(bond.GetStereo()) or int(bond.GetBondDir()) or bond.HasQuery()
             for bond in mol.GetBonds()):
        return False
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    return bool(inventory.supported and not inventory.groups)


@lru_cache(maxsize=128)
def _networks(chalcogen=8, azine='pyridine'):
    """Enumerate both component-parent choices and their edge fusions."""
    from .fusion_components import resolve_component
    from .fusion_networks import _lettered_path, _perimeter, _prefix, parse_network

    if chalcogen not in {8, 16}:
        return ()
    benzo_word = "[1]benzofuran" if chalcogen == 8 else "[1]benzothiophene"
    benzo_prefix = "[1]benzofuro" if chalcogen == 8 else "[1]benzothiopheno"
    benzofuran = resolve_component(benzo_word)
    pyridine = resolve_component(azine)
    azine_prefix = _prefix(azine)
    if (benzofuran is None or pyridine is None or azine_prefix is None
            or pyridine.ring_count != 1 or pyridine.atom_count != 6
            or not any(number == 7 for number, _count in pyridine.elements)):
        return ()
    candidates = []
    for prefix, attached, parent_word, parent in (
        (benzo_prefix, benzofuran, azine, pyridine),
        (azine_prefix, pyridine, benzo_word, benzofuran),
    ):
        perimeter = _perimeter(parent)
        if perimeter is None:
            return ()
        attached_elements = dict(attached.label_elements)
        # The supported original face tree has distinct junctions on the
        # outer six-rings, so its shared edge has two nonfusion C locants.
        edges = sorted(tuple(sorted(edge, key=int)) for edge in attached.bonds
                       if len(edge) == 2 and all(label.isdigit()
                       and attached_elements[label] == 6 for label in edge))
        for first, second in edges:
            for left, right in ((first, second), (second, first)):
                for position in range(len(perimeter)):
                    letter = chr(ord("a") + position)
                    path = _lettered_path(parent, letter)
                    if path is None or any(dict(parent.label_elements)[label] != 6
                                           for label in path[0]):
                        continue
                    text = f"{prefix}[{left},{right}-{letter}]{parent_word}"
                    network = parse_network(text, resolve_component)
                    if network is not None and network.citation is True:
                        candidates.append((text, network))
    return tuple(candidates)


def skeletons(mol, Chem):
    """Yield only unique, graph-matched component and numbering frontiers."""
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if (len(rings) != 3 or sorted(map(len, rings)) != [5, 6, 6]
            or mol.GetNumAtoms() != 13 or not all(a.GetIsAromatic() for a in mol.GetAtoms())
            or any(a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) for a in mol.GetAtoms())):
        return ()
    five = next(ring for ring in rings if len(ring) == 5)
    six = [ring for ring in rings if len(ring) == 6]
    five_elements = sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in five)
    benzene = [ring for ring in six if all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ring)]
    if (six[0] & six[1] or five_elements not in ([6, 6, 6, 6, 8], [6, 6, 6, 6, 16])
            or len(benzene) != 1):
        return ()
    aza = next(ring for ring in six if ring != benzene[0])
    if (any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7} for i in aza)
            or not any(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in aza)):
        return ()
    for face in six:
        shared = five & face
        if (len(shared) != 2 or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in shared)
                or mol.GetBondBetweenAtoms(*tuple(shared)) is None):
            return ()
    from .three_face_aza_fusion import _component_rows
    azines = {name for name, _sequence, _component in _component_rows(mol, tuple(aza))}
    if len(azines) != 1:
        return ()
    azine = azines.pop()
    from .fusion_networks import matches_skeleton
    from .fusion_state_skeletons import _two_component_preference
    candidates = [(name, network) for name, network in _networks(five_elements[-1], azine)
                  if matches_skeleton(network, mol, range(mol.GetNumAtoms()))]
    if not candidates:
        return ()
    winners = []
    for name, network in candidates:
        relations = [_two_component_preference(network, other)
                     for _, other in candidates if other is not network]
        if all(value in (-1, 0) for value in relations):
            winners.append((name, network))
    if len({name for name, _ in winners}) != 1:
        return ()  # A missing component or numbering comparison remains open.
    from .fused_orientation import fused_orientation_frontier
    from .catacondensed_numbering import peripheral_numbering_frontier
    drawing = fused_orientation_frontier(mol)
    if not drawing.admitted:
        return ()
    numbering = peripheral_numbering_frontier(mol, orientation_seeds=drawing.orientation_seeds)
    if not numbering.admitted or not numbering.candidates:
        return ()
    seq = tuple(range(mol.GetNumAtoms()))
    numbers = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in seq)
    edges = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    return tuple((winners[0][0], seq, tuple(dict(row.atom_locants)[i] for i in seq),
                  numbers, edges, False, (), METHOD) for row in numbering.candidates)


def perimeter(parent, edges, size):
    """Revalidate the numbered source topology for H-state construction."""
    if size != 13 or not parent.startswith(("[1]benzofuro[", "[1]benzothiopheno[")):
        return None
    from rdkit import Chem
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fusion_networks import skeleton_key

    chalcogen = 16 if parent.startswith("[1]benzothiopheno[") else 8
    from .fusion_components import resolve_component
    from .fusion_networks import parse_network
    network = parse_network(parent, resolve_component)
    if network is None or len(network.parents) != 1:
        return None
    azine = network.names[network.parents[0]]
    network = next((candidate for name, candidate in _networks(chalcogen, azine) if name == parent), None)
    if network is None or network.atom_count != size:
        return None
    query = Chem.MolFromSmiles(network.skeleton)
    if query is None:
        return None
    source_edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in query.GetBonds()]
    if skeleton_key([6] * size, edges) != skeleton_key([6] * size, source_edges):
        return None
    graph = Chem.RWMol()
    for _ in range(size):
        graph.AddAtom(Chem.Atom(6))
    for left, right in edges:
        graph.AddBond(left, right, Chem.BondType.SINGLE)
    topology = graph.GetMol()
    try:
        Chem.SanitizeMol(topology)
    except (ValueError, RuntimeError):
        return None
    frontier = peripheral_numbering_frontier(topology)
    return frozenset(frontier.perimeter_cycle) if frontier.admitted else None


def requires_full(mol, Chem):
    """Route only an actually rendered, completely mapped source graph.

    The compatibility shortcut cannot compare the bounded fusion parent to
    its alternative ring parents. This predicate checks the new full writer's
    component graph and original-atom partition before requesting a handoff;
    it does not assign preferred-name status.
    """
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if (len(rings) < 3 or not any(len(ring) == 5 for ring in rings)
            or sum(len(ring) == 6 for ring in rings) < 2):
        return False
    from .mapped_ring_parents import ring_systems
    if not any(len(core) == 13 and sorted(
            mol.GetAtomWithIdx(i).GetAtomicNum() for i in core) == [6] * 11 + [7, 8]
            for core in ring_systems(mol)) or not source_supported(mol):
        return False
    from .mapped_ring_parents import heteropolycyclic_hydride
    from .fusion_networks import matches_skeleton
    record = heteropolycyclic_hydride(mol, Chem)
    if record is None or record.method != METHOD or record.construction is None:
        return False
    construction, network = record.construction, record.fusion_construction
    mapped = frozenset(dict(record.atom_locants))
    required = frozenset(atom.GetIdx() for atom in mol.GetAtoms()
                         if atom.GetAtomicNum() > 1 or atom.GetIsotope())
    return bool(network is not None and matches_skeleton(network, mol, record.parent_atoms)
                and mapped == construction.atoms and not mapped & construction.embedded_atoms
                and mapped | construction.embedded_atoms == required)
