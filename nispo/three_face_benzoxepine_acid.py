"""Mapped carboxylic acids on a bounded 5/7/6 benzoheterocycle.

The P-25 benzoheterocycle component and its cyclopenta attachment are derived
from source atom/edge sets. The original molecule alone supplies suffix,
hydrogenation and stereo evidence. This is a general construction, not a
complete preferred-parent/method frontier.
"""
from collections import Counter
from functools import lru_cache

from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


@lru_cache(maxsize=256)
def perimeter(parent, edges, size):
    """Use ring-face incidence for the graph-proved 5/7/6 component union."""
    if size != 14:
        return None
    from rdkit import Chem
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fusion_components import resolve_component
    from .fusion_networks import FusionNetwork, parse_network, skeleton_key

    network = parse_network(parent, resolve_component)
    if (not isinstance(network, FusionNetwork) or network.citation is not True
            or network.atom_count != 14 or network.ring_count != 3
            or network.elements != ((6, 13), (8, 1))
            or len(network.components) != 2
            or sorted((c.atom_count, c.ring_count) for c in network.components) != [(5, 1), (11, 2)]):
        return None
    query = Chem.MolFromSmiles(network.skeleton)
    if query is None:
        return None
    query_edges = tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in query.GetBonds())
    if skeleton_key([6] * size, edges) != skeleton_key([6] * size, query_edges):
        return None
    rw = Chem.RWMol()
    for _ in range(size):
        rw.AddAtom(Chem.Atom(6))
    for a, b in edges:
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    graph = rw.GetMol()
    try:
        Chem.SanitizeMol(graph)
    except (ValueError, RuntimeError):
        return None
    frontier = peripheral_numbering_frontier(graph)
    if (not frontier.admitted or len(frontier.ring_faces) != 3
            or sorted(map(len, frontier.ring_faces)) != [5, 6, 7]):
        return None
    return frozenset(frontier.perimeter_cycle)


def _face_path(graph, face, start, end):
    current = [n.GetIdx() for n in graph.GetAtomWithIdx(start).GetNeighbors()
               if n.GetIdx() in face and n.GetIdx() != end]
    if len(current) != 1:
        return ()
    path, current = [start], current[0]
    while current != end:
        if current in path:
            return ()
        path.append(current)
        following = [n.GetIdx() for n in graph.GetAtomWithIdx(current).GetNeighbors()
                     if n.GetIdx() in face and n.GetIdx() != path[-2]]
        if len(following) != 1:
            return ()
        current = following[0]
    return (*path, end)


def _parent_component(graph, five, six, seven):
    """Map the P-25.2.2.4 component and its attachment to original vertices."""
    from .fusion_components import resolve_component
    from .fusion_networks import parse_network
    from .parent_selection import locant_key

    shared = six & seven
    if (len(shared) != 2 or graph.GetBondBetweenAtoms(*tuple(shared)) is None
            or five & six):
        return None
    attached = five & seven
    if len(attached) != 2 or graph.GetBondBetweenAtoms(*tuple(attached)) is None:
        return None
    oxygen = next((i for i in seven if graph.GetAtomWithIdx(i).GetAtomicNum() == 8), None)
    if oxygen is None:
        return None
    neighbors = {n.GetIdx() for n in graph.GetAtomWithIdx(oxygen).GetNeighbors()}
    if len(neighbors & shared) != 1 or len(neighbors & attached) != 1:
        return None
    rows = []
    for a, b in (tuple(shared), tuple(reversed(tuple(shared)))):
        first = _face_path(graph, seven, a, b)
        second = _face_path(graph, six, b, a)
        if not first or not second:
            return None
        ordered = (*first[1:], *second[1:])
        count, labels = 0, []
        for atom in ordered:
            if atom in shared:
                labels.append(str(count) + 'a')
            else:
                count += 1
                labels.append(str(count))
        mapping = dict(zip(ordered, labels))
        if set(mapping) != (six | seven) or oxygen not in mapping:
            return None
        rows.append((int(mapping[oxygen]), mapping))
    best = min(row[0] for row in rows)
    finalists = [row for row in rows if row[0] == best]
    if len(finalists) != 1 or best != 1:
        return None
    mapping = finalists[0][1]
    component = resolve_component(f'[{best}]benzoxepine')
    if component is None:
        return None
    # Component-native element/edge labels must equal the mapped source
    # subgraph, rather than merely matching its unnumbered topology.
    if (dict(component.label_elements) != {mapping[i]: graph.GetAtomWithIdx(i).GetAtomicNum()
                                           for i in six | seven}
            or component.bonds != frozenset(
                frozenset((mapping[b.GetBeginAtomIdx()], mapping[b.GetEndAtomIdx()]))
                for b in graph.GetBonds()
                if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= (six | seven))):
        return None
    labels = tuple(sorted(mapping.values(), key=locant_key))
    component_edges = component.bonds
    letters = [chr(ord('a') + i) for i, (a, b) in enumerate(zip(labels, labels[1:] + labels[:1]))
               if frozenset((a, b)) in component_edges and {a, b} == {mapping[i] for i in attached}]
    if len(letters) != 1:
        return None
    name = f'cyclopenta[{letters[0]}][{best}]benzoxepine'
    network = parse_network(name, resolve_component)
    if network is None or network.citation is not True or network.parent != component:
        return None
    return name, network, mapping


def three_face_fused_acid_decision(mol, Chem):
    """Return a complete source-mapped fused-acid construction, or decline."""
    from .branch_rules import collect_subtree
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .component_mapping import _mapped_stereochemistry_plan, has_annotations, mapped_branch
    from .extended_inventory import selection_inventory
    from .functionalized_carbon_prefixes import (functionalized_carbon_prefix,
                                                 validate_functionalized_carbon_prefix_proof)
    from .fusion_components import fusion_construction
    from .fused_orientation import fused_orientation_frontier
    from .fusion_networks import matches_skeleton
    from .input_stereochemistry import unsupported_relative_stereochemistry
    from .mapped_ring_parents import ring_systems
    from .naming_context import MoleculeContext
    from .ring_state_parent import _inventory, _reference_states, _render
    from .stereochemistry import _indexed_bond_stereochemistry, assigned_atom_stereochemistry
    from .three_face_heterolactam_fusion import _core_topology

    if (mol.GetNumHeavyAtoms() > 80 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups() or unsupported_relative_stereochemistry(mol, Chem)
            or any(a.GetAtomicNum() not in {6, 8, 9, 17, 35, 53} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons() or a.HasQuery() or a.GetAtomMapNum()
                   or int(a.GetChiralTag()) and a.GetChiralTag() not in
                       (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                   for a in mol.GetAtoms())
            or any(b.HasQuery() for b in mol.GetBonds())):
        return None
    cores = ring_systems(mol)
    if (len(cores) != 1 or len(cores[0]) != 14
            or Counter(mol.GetAtomWithIdx(i).GetAtomicNum() for i in cores[0]) != {6: 13, 8: 1}):
        return None
    core = cores[0]
    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status != 'selected' or frontier.unknown_pairs or frontier.cycle
            or len(frontier.groups) != 1 or frontier.groups[0].kind != 'acid'
            or tuple(frontier.groups) != tuple(inventory.principal)
            or any(g.kind not in {'acid', 'alcohol'} for g in inventory.groups)):
        return None
    group = frontier.groups[0]
    if group.owner not in core or group.center in core:
        return None
    acid_atoms = frozenset(group.atoms)
    if (len(acid_atoms) != 3 or group.center not in acid_atoms
            or sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in acid_atoms) != 2):
        return None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 8 or atom.GetIdx() in core or atom.GetIdx() in acid_atoms:
            continue
        if (atom.IsInRing() or atom.GetDegree() != 1 or atom.GetTotalNumHs() != 1
                or atom.GetNeighbors()[0].GetAtomicNum() != 6
                or atom.GetBonds()[0].GetBondTypeAsDouble() != 1):
            return None
    copied = _core_topology(mol, Chem, core)
    if copied is None:
        return None
    graph, sequence = copied
    faces = tuple(map(frozenset, graph.GetRingInfo().AtomRings()))
    if len(faces) != 3 or sorted(map(len, faces)) != [5, 6, 7]:
        return None
    five, six, seven = (next(face for face in faces if len(face) == size) for size in (5, 6, 7))
    if (any(graph.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in five | six)
            or sum(graph.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in seven) != 1
            or any(not mol.GetAtomWithIdx(sequence[i]).GetIsAromatic() for i in six)):
        return None
    for bond in graph.GetBonds():
        indices = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
        if not indices <= six:
            actual = mol.GetBondBetweenAtoms(*(sequence[i] for i in indices))
            if actual.GetIsAromatic() or actual.GetBondTypeAsDouble() != 1:
                return None
    parent = _parent_component(graph, five, six, seven)
    if parent is None or not matches_skeleton(parent[1], mol, core):
        return None
    drawing = fused_orientation_frontier(graph)
    if not drawing.admitted:
        return None
    numbering = peripheral_numbering_frontier(graph, orientation_seeds=drawing.orientation_seeds)
    if not numbering.admitted or not numbering.candidates:
        return None
    edges = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in graph.GetBonds())
    numbers = tuple(a.GetAtomicNum() for a in graph.GetAtoms())
    scope = frozenset(a.GetIdx() for a in mol.GetAtoms())
    context, inventory_for_render = MoleculeContext(mol, Chem), _inventory(mol, Chem)
    if not inventory_for_render.supported or tuple(inventory_for_render.principal) != (group,):
        return None
    rows = []
    for candidate in numbering.candidates:
        locants = dict(candidate.atom_locants)
        if set(locants) != set(range(len(sequence))):
            return None
        skeleton = (parent[0], sequence, tuple(locants[i] for i in range(len(sequence))),
                    numbers, edges, False, (), 'component-fused')
        for reference in _reference_states(numbers, edges):
            rows.extend(_render(context, skeleton, reference, inventory_for_render, scope=scope))
    if not rows:
        return None
    best = min(row.numbering_key for row in rows)
    finalists = [row for row in rows if row.numbering_key == best]
    if len({row.name for row in finalists}) != 1:
        return None
    record, construction = finalists[0], finalists[0].construction
    if (construction is None or record.fusion_construction != fusion_construction(parent[0])
            or not record.hydrogen_budget_proved or not record.annotation_coverage
            or record.principal_centers != {group.center} or frozenset(record.parent_atoms) != core
            or construction.atoms != core or construction.atoms & construction.embedded_atoms
            or construction.atoms | construction.embedded_atoms != scope
            or not acid_atoms <= construction.embedded_atoms):
        return None
    branches, roles, branch_paths = set(), [], []
    for root, anchor, owned, text in construction.prefix_branch_ownership:
        if (anchor not in core or root not in owned or owned & (core | acid_atoms | branches)
                or frozenset(collect_subtree(mol, root, anchor)) != owned):
            return None
        cuts = {(i, n.GetIdx()) for i in owned for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in owned}
        if cuts != {(root, anchor)} or mapped_branch(mol, Chem, root, anchor) != (text, owned):
            return None
        if mol.GetAtomWithIdx(root).GetAtomicNum() == 6:
            proof, status = functionalized_carbon_prefix(mol, Chem, root, anchor, (group,))
            carbon = {i for i in owned if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
            if status == 'complete':
                if (proof.scope != owned or proof.text != text or set(proof.carbon_path) != carbon
                        or not validate_functionalized_carbon_prefix_proof(mol, Chem, proof)):
                    return None
                branch_paths.append(frozenset(proof.carbon_path))
            else:
                # A simple unannotated saturated carbon-only arm does not
                # contain functional roles and has its own bounded writer.
                from .substituent_selection import mapped_saturated_carbon_prefix

                if (carbon != set(owned) or has_annotations(mol, owned)
                        or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds()
                               if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= owned)
                        or mapped_saturated_carbon_prefix(mol, Chem, root, anchor, owned) != (text, owned)):
                    return None
                branch_paths.append(frozenset(carbon))
        elif mol.GetAtomWithIdx(root).GetAtomicNum() not in {9, 17, 35, 53} or has_annotations(mol, owned):
            return None
        branches.update(owned)
        roles.append(('ring-substituent', owned, ()))
    if branches | acid_atoms != set(construction.embedded_atoms):
        return None
    assigned = dict(assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True))
    tagged = {a.GetIdx() for a in mol.GetAtoms() if int(a.GetChiralTag())}
    if set(assigned) != tagged or not tagged <= (core | frozenset().union(*branch_paths)):
        return None
    stereo_plan = _mapped_stereochemistry_plan(mol, Chem, construction, dict(construction.atom_locants))
    if (stereo_plan is None or {item[0] for item in stereo_plan[1]} != tagged & core
            or stereo_plan[2]):
        return None
    stereo_bonds = _indexed_bond_stereochemistry(mol, Chem)
    if any(not any({a, b} <= path for path in branch_paths) for a, b, _ in stereo_bonds):
        return None
    potential = tuple(Chem.FindPotentialStereo(mol))
    if any(item.type not in {Chem.StereoType.Atom_Tetrahedral, Chem.StereoType.Bond_Double}
           for item in potential):
        return None
    issues = [SelectionLimitation(
        'fused-acid-parent', 'P-25/P-44/P-45/P-51',
        'The benzoheterocycle component, fused-ring drawing, acid and original stereo are mapped; '
        'the complete preferred-parent and method comparison remains unresolved.')]
    if any(item.specified != Chem.StereoSpecified.Specified for item in potential):
        issues.append(SelectionLimitation(
            'stereochemistry', 'P-91.2.1.2.1/P-91.2.2',
            'One or more potential stereogenic units have no source configuration.'))
    return MethodDecision(record.name, 'fused-acid', frozenset({group.center}), scope,
                          'P-25.2.2.4/P-25.3/P-31.2.3.3.2/P-41/P-65.1.2/P-91',
                          record.atom_locants,
                          component_roles=(('fused-parent', core, construction.atom_locants),
                                           ('carboxylic-acid', acid_atoms, ()), *roles),
                          preference_issues=tuple(issues))
