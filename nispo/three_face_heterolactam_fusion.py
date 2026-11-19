"""Source-mapped P-25/P-31 lactams on a bounded three-face fusion tree.

The neutral face tree contains a five-membered aza ring, a six-membered
oxaza ring and a six-membered aza ring. The five-membered face shares N-C
with the middle face; the other shared edge is C-C. The capped *topology*
supplies component and drawing candidates only. Hydrogen, suffix and stereo
evidence are calculated and owned on the unchanged source molecule.
"""
from collections import Counter
from functools import lru_cache

from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


@lru_cache(maxsize=256)
def _component_frontier(smiles):
    """Enumerate the primitive components and require a proved unique winner."""
    from rdkit import Chem
    from .fusion_components import compare_constructions
    from .three_face_aza_fusion import _candidate_networks

    graph = Chem.MolFromSmiles(smiles)
    if graph is None:
        return None
    candidates = _candidate_networks(graph, graph.GetRingInfo().AtomRings())
    winners = [(name, network) for name, network in candidates
               if all(compare_constructions(network, other).preference in (-1, 0)
                      for _, other in candidates)]
    return winners[0] if len(winners) == 1 else None


@lru_cache(maxsize=256)
def perimeter(parent, edges, size):
    """Prove this component network's actual outside cycle from face edges.

    The generic degree-three heuristic removes a nonfusion edge when two
    different junctions happen to be adjacent. Face incidence, rather than
    junction degree, identifies fusion edges for this three-face family.
    """
    if size != 13:
        return None
    from rdkit import Chem
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fusion_components import fusion_construction
    from .fusion_networks import FusionNetwork, skeleton_key

    network = fusion_construction(parent)
    if (not isinstance(network, FusionNetwork) or network.citation is not True
            or network.atom_count != 13 or network.ring_count != 3
            or network.elements != ((6, 10), (7, 2), (8, 1))
            or len(network.components) != 3
            or sorted(component.atom_count for component in network.components) != [5, 6, 6]):
        return None
    query = Chem.MolFromSmiles(network.skeleton)
    if query is None:
        return None
    query_edges = tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in query.GetBonds())
    if skeleton_key([6] * size, edges) != skeleton_key([6] * size, query_edges):
        return None
    graph = Chem.RWMol()
    for _ in range(size):
        graph.AddAtom(Chem.Atom(6))
    for a, b in edges:
        graph.AddBond(a, b, Chem.BondType.SINGLE)
    topology = graph.GetMol()
    try:
        Chem.SanitizeMol(topology)
    except (RuntimeError, ValueError):
        return None
    frontier = peripheral_numbering_frontier(topology)
    if not frontier.admitted or len(frontier.ring_faces) != 3:
        return None
    return frozenset(frontier.perimeter_cycle)


def _core_topology(mol, Chem, core):
    """Copy only element-labelled bonds, with an explicit source map."""
    source = tuple(sorted(core))
    inverse = {original: local for local, original in enumerate(source)}
    graph = Chem.RWMol()
    for original in source:
        graph.AddAtom(Chem.Atom(mol.GetAtomWithIdx(original).GetAtomicNum()))
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in inverse and b in inverse:
            graph.AddBond(inverse[a], inverse[b], Chem.BondType.SINGLE)
    graph = graph.GetMol()
    try:
        Chem.SanitizeMol(graph)
    except (RuntimeError, ValueError):
        return None
    # The constructed graph must retain the induced original atom/bond graph.
    original_edges = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                      for b in mol.GetBonds()
                      if b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core}
    copied_edges = {frozenset((source[b.GetBeginAtomIdx()], source[b.GetEndAtomIdx()]))
                    for b in graph.GetBonds()}
    if original_edges != copied_edges:
        return None
    return graph, source


def _supported_faces(graph):
    """Prove a three-face tree and exclude retained/polyparent alternatives."""
    faces = tuple(map(frozenset, graph.GetRingInfo().AtomRings()))
    if len(faces) != 3 or sorted(map(len, faces)) != [5, 6, 6]:
        return False
    five = next(face for face in faces if len(face) == 5)
    sixes = tuple(face for face in faces if len(face) == 6)
    def elements(face):
        return Counter(graph.GetAtomWithIdx(i).GetAtomicNum() for i in face)

    if elements(five) != {6: 4, 7: 1}:
        return False
    middle = next((face for face in sixes if elements(face) == {6: 4, 7: 1, 8: 1}), None)
    outer = next((face for face in sixes if elements(face) == {6: 5, 7: 1}), None)
    if middle is None or outer is None or five & outer:
        return False
    # The currently validated fusion/hydrogen renderer binds O adjacent to
    # the C-C junction with the outer aza face. An O next to the C-N junction
    # can induce another component orientation and needs a separate mapping
    # proof; graph isomorphism of the unnumbered union is insufficient.
    oxygen = next(i for i in middle if graph.GetAtomWithIdx(i).GetAtomicNum() == 8)
    oxygen_neighbors = {a.GetIdx() for a in graph.GetAtomWithIdx(oxygen).GetNeighbors()}
    if not oxygen_neighbors & (middle & outer) or oxygen_neighbors & (middle & five):
        return False
    for first, second, numbers in ((five, middle, [6, 7]), (middle, outer, [6, 6])):
        shared = first & second
        if (len(shared) != 2 or sorted(graph.GetAtomWithIdx(i).GetAtomicNum() for i in shared) != numbers
                or graph.GetBondBetweenAtoms(*tuple(shared)) is None):
            return False
    return True


def three_face_lactam_decision(mol, Chem):
    """Return a mapped general fusion construction or decline outside scope."""
    from .branch_rules import collect_subtree
    from .component_mapping import _mapped_stereochemistry_plan, mapped_branch
    from .extended_inventory import selection_inventory
    from .fusion_networks import matches_skeleton
    from .input_stereochemistry import unsupported_relative_stereochemistry
    from .mapped_ring_parents import ring_systems
    from .naming_context import MoleculeContext
    from .ring_state_parent import _inventory, _reference_states, _render
    from .fused_orientation import fused_orientation_frontier
    from .catacondensed_numbering import peripheral_numbering_frontier

    if (mol.GetNumHeavyAtoms() > 80 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups() or unsupported_relative_stereochemistry(mol, Chem)
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 17, 35, 53} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons() or a.HasQuery() or a.GetAtomMapNum()
                   for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    cores = [core for core in ring_systems(mol) if len(core) == 13
             and Counter(mol.GetAtomWithIdx(i).GetAtomicNum() for i in core) == {6: 10, 7: 2, 8: 1}]
    if len(cores) != 1:
        return None
    core = cores[0]
    faces = [face for face in mol.GetRingInfo().AtomRings() if set(face) <= core]
    if len(faces) != 3 or sorted(map(len, faces)) != [5, 6, 6]:
        return None
    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.principal) != 1
            or inventory.principal[0].kind != 'lactam'
            or sum(g.kind == 'lactam' for g in inventory.groups) != 1
            or any(g.kind not in {'lactam', 'amine'} for g in inventory.groups)):
        return None
    group = inventory.principal[0]
    oxygen = [i for i in group.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 8 and i not in core]
    if (group.center not in core or group.ligand not in core or len(oxygen) != 1
            or {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8}
               != {oxygen[0]} | {i for i in core if mol.GetAtomWithIdx(i).GetAtomicNum() == 8}):
        return None
    # No outside N-N, N=, isotope or out-of-core stereo can hide a competing
    # characteristic class or require a separate annotation ownership proof.
    for atom in mol.GetAtoms():
        if atom.GetIdx() in core or atom.GetAtomicNum() != 7:
            continue
        if atom.IsInRing():
            if not atom.GetIsAromatic():
                return None
        elif (atom.GetIsAromatic() or atom.GetTotalValence() != 3
              or any(n.GetAtomicNum() != 6 for n in atom.GetNeighbors())
              or any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds())):
            return None
    potential = tuple(Chem.FindPotentialStereo(mol))
    if (any(item.type != Chem.StereoType.Atom_Tetrahedral or item.centeredOn not in core
            for item in potential)
            or any(int(a.GetChiralTag()) and (a.GetIdx() not in core or a.GetChiralTag() not in
                       (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW))
                   for a in mol.GetAtoms())):
        return None
    copied = _core_topology(mol, Chem, core)
    if copied is None:
        return None
    topology, sequence = copied
    if not _supported_faces(topology):
        return None
    faces = tuple(map(frozenset, topology.GetRingInfo().AtomRings()))
    outer = next(face for face in faces if len(face) == 6
                 and Counter(topology.GetAtomWithIdx(i).GetAtomicNum() for i in face) == {6: 5, 7: 1})
    middle = next(face for face in faces if len(face) == 6 and face != outer)
    if any(not mol.GetAtomWithIdx(sequence[i]).GetIsAromatic() for i in outer):
        return None
    for bond in topology.GetBonds():
        endpoints = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
        if endpoints <= middle and not endpoints <= outer:
            source_bond = mol.GetBondBetweenAtoms(*(sequence[i] for i in endpoints))
            if source_bond.GetIsAromatic() or source_bond.GetBondTypeAsDouble() != 1:
                return None
    # The lactam's N and carbonyl carbon must both belong to the five face,
    # with its N also shared with the middle face.
    five = next(set(face) for face in topology.GetRingInfo().AtomRings() if len(face) == 5)
    face_original = {sequence[i] for i in five}
    if (not {group.center, group.ligand} <= face_original
            or mol.GetBondBetweenAtoms(group.center, group.ligand) is None
            or mol.GetBondBetweenAtoms(group.center, oxygen[0]).GetBondTypeAsDouble() != 2):
        return None
    winner = _component_frontier(Chem.MolToSmiles(topology, canonical=True))
    if winner is None or not matches_skeleton(winner[1], mol, core):
        return None
    drawing = fused_orientation_frontier(topology, allow_three_face_n_junction=True)
    if not drawing.admitted:
        return None
    numbering = peripheral_numbering_frontier(topology, orientation_seeds=drawing.orientation_seeds,
                                               allow_three_face_n_junction=True)
    if not numbering.admitted or not numbering.candidates:
        return None
    edges = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in topology.GetBonds())
    numbers = tuple(a.GetAtomicNum() for a in topology.GetAtoms())
    scope = frozenset(a.GetIdx() for a in mol.GetAtoms())
    context, inventory_for_render = MoleculeContext(mol, Chem), _inventory(mol, Chem)
    if not inventory_for_render.supported or inventory_for_render.principal != inventory.principal:
        return None
    rows = []
    for candidate in numbering.candidates:
        local = dict(candidate.atom_locants)
        if set(local) != set(range(len(sequence))):
            return None
        skeleton = (winner[0], sequence, tuple(local[i] for i in range(len(sequence))),
                    numbers, edges, False, (), 'component-fused')
        for reference in _reference_states(numbers, edges):
            rows.extend(_render(context, skeleton, reference, inventory_for_render, scope=scope))
    if not rows:
        return None
    best = min(row.numbering_key for row in rows)
    finalists = [row for row in rows if row.numbering_key == best]
    if len({row.name for row in finalists}) != 1:
        return None
    record = finalists[0]
    construction = record.construction
    if (construction is None or record.fusion_construction != winner[1]
            or not record.hydrogen_budget_proved or not record.annotation_coverage
            or record.principal_centers != {group.center} or frozenset(record.parent_atoms) != core
            or construction.atoms != core or construction.atoms & construction.embedded_atoms
            or construction.atoms | construction.embedded_atoms != scope
            or oxygen[0] not in construction.embedded_atoms):
        return None
    branches, roles = set(), []
    for root, anchor, owned, text in construction.prefix_branch_ownership:
        if (anchor not in core or root not in owned or owned & (core | {oxygen[0]} | branches)
                or frozenset(collect_subtree(mol, root, anchor)) != owned):
            return None
        cuts = {(i, n.GetIdx()) for i in owned for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in owned}
        if cuts != {(root, anchor)} or mapped_branch(mol, Chem, root, anchor) != (text, owned):
            return None
        branches.update(owned)
        roles.append(('ring-substituent', owned, ()))
    if branches | {oxygen[0]} != set(construction.embedded_atoms):
        return None
    plan = _mapped_stereochemistry_plan(mol, Chem, construction, dict(construction.atom_locants))
    tagged = {i for i in core if int(mol.GetAtomWithIdx(i).GetChiralTag())}
    if plan is None or {item[0] for item in plan[1]} != tagged or plan[2]:
        return None
    issues = [SelectionLimitation(
        'fused-lactam-parent', 'P-25/P-44/P-45/P-51',
        'The bounded component frontier, original hydrogen state, lactam and stereo are mapped; '
        'the complete preferred-parent and method comparison remains unresolved.')]
    if any(item.specified != Chem.StereoSpecified.Specified for item in potential):
        issues.append(SelectionLimitation(
            'stereochemistry', 'P-91.2.1.2.1/P-91.2.2',
            'One or more potential stereogenic units have no source configuration.'))
    return MethodDecision(record.name, 'fused-lactam', frozenset({group.center}), scope,
                          'P-25.3/P-31.2.3.3.2/P-58.2/P-64.3/P-91',
                          record.atom_locants,
                          component_roles=(('fused-parent', core, construction.atom_locants),
                                           ('lactam-oxygen', frozenset(oxygen), ()), *roles),
                          preference_issues=tuple(issues))
