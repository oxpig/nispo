"""Productive two-face fusion frames for original-graph H-state naming.

Hantzsch-Widman components and source-constrained whole-system drawings supply
numbered constructions. General fusion resources provide additional frames.
Neither supplies whole-molecule answers. This adapter bypasses historical
alias-length budgets; unproved ties remain barriers.
"""
from functools import lru_cache
from itertools import combinations
import json
import re

from .template_index import (
    molecule_features, ring_system_feature_key, ring_system_with_graph_signatures,
)


@lru_cache(maxsize=4)
def dibenzo_heterocycle_templates(Chem):
    """Fixed retained frames and a numbered P-25 oxazepine fusion frame."""
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fusion_components import resolve_component
    from .fusion_networks import matches_skeleton, parse_network
    from .preferred_fused import special_retained_fused_locants

    rows = []
    locants = dict(enumerate('1 2 3 4 4a 5 5a 6 7 8 9 9a 10 10a'.split()))
    for name, element in (('phenoxazine', 'O'), ('phenothiazine', 'S')):
        graph = Chem.MolFromSmiles(f'C1=CC=CC=2{element}C3=CC=CC=C3NC12')
        numbered = special_retained_fused_locants(name, graph, locants)
        if numbered is not None:
            rows.append(dict(name=name, mol=graph, locants=numbered, method='retained-fused'))
    name = 'dibenzo[b,f][1,4]oxazepine'
    graph = Chem.MolFromSmiles('C1=CC=CC2=C1C=NC1=C(O2)C=CC=C1')
    locants = dict(enumerate('1 2 3 4 4a 11a 11 10 9a 5a 5 6 7 8 9'.split()))
    topology = peripheral_numbering_frontier(graph)
    network = parse_network(name, resolve_component)
    if (topology.admitted and sorted(map(len, topology.ring_faces)) == [6, 6, 7]
            and network is not None and network.citation is True
            and matches_skeleton(network, graph, range(graph.GetNumAtoms()))):
        rows.append(dict(name=name, mol=graph, locants=locants, method='component-fused'))
    return tuple(rows)


@lru_cache(maxsize=16)
def dibenzo_heterocycle_perimeter(parent, edges, numbers):
    from rdkit import Chem
    from .catacondensed_numbering import peripheral_numbering_frontier

    template = next((row for row in dibenzo_heterocycle_templates(Chem)
                     if row['name'] == parent), None)
    if template is None:
        return None
    graph = template['mol']
    if (tuple(a.GetAtomicNum() for a in graph.GetAtoms()) != numbers
            or {frozenset(edge) for edge in edges} !=
            {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in graph.GetBonds()}):
        return None
    topology = peripheral_numbering_frontier(graph)
    return frozenset(topology.perimeter_cycle) if topology.admitted else None


@lru_cache(maxsize=4)
def benzo_carbazole_templates(Chem):
    """The three distinct benzenoid edge fusions of the carbazole parent.

    The P-25 component network verifies each topology. The locants are an
    independent numbered-component witness from OPSIN's extended SMILES.
    """
    from .fusion_components import resolve_component
    from .fusion_networks import matches_skeleton, parse_network

    frames = (
        ('a', 'C1=CC=CC=2C1=C1NC3=CC=CC=C3C1=CC2',
         '1 2 3 4 4a 11b 11a 11 10a 10 9 8 7 6b 6a 6 5'),
        ('b', 'C1=C2C=3C=C4C(=CC3NC2=CC=C1)C=CC=C4',
         '1 11b 11a 11 10a 6a 6 5a 5 4a 4 3 2 7 8 9 10'),
        ('c', 'C1=CC=CC=2C=CC=3NC=4C=CC=CC4C3C21',
         '1 2 3 4 4a 5 6 6a 7 7a 8 9 10 11 11a 11b 11c'),
    )
    templates = []
    for edge, smiles, labels in frames:
        name = f'benzo[{edge}]carbazole'
        mol = Chem.MolFromSmiles(smiles)
        network = parse_network(name, resolve_component)
        locants = tuple(labels.split())
        if (mol is not None and network is not None and network.citation is True
                and len(set(locants)) == mol.GetNumAtoms() == 17
                and matches_skeleton(network, mol, range(17))):
            templates.append({'name': name, 'mol': mol, 'locants': dict(enumerate(locants)),
                              'ring_state_method': 'component-fused-benzocarbazole'})
    return tuple(templates)


@lru_cache(maxsize=12)
def benzo_carbazole_perimeter(parent, edges, numbers):
    """Prove the source numbered perimeter and the complete fused topology."""
    if not parent.startswith('benzo[') or not parent.endswith(']carbazole') or len(numbers) != 17:
        return None
    from rdkit import Chem
    from .catacondensed_numbering import peripheral_numbering_frontier

    template = next((row for row in benzo_carbazole_templates(Chem) if row['name'] == parent), None)
    if template is None:
        return None
    graph = template['mol']
    if (tuple(a.GetAtomicNum() for a in graph.GetAtoms()) != numbers
            or {frozenset(edge) for edge in edges} !=
            {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in graph.GetBonds()}):
        return None
    proof = peripheral_numbering_frontier(graph, allow_peri_fused=True)
    return frozenset(proof.perimeter_cycle) if proof.admitted else None


@lru_cache(maxsize=4)
def benzo_imidazothiazole_templates(Chem):
    """Build benzo edge fusions on the numbered imidazole–thiazole frame."""
    from .fusion_components import resolve_component
    from .fusion_networks import matches_skeleton, parse_network
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fused_orientation import fused_orientation_frontier

    result = []
    for letter in 'cde':
        name = f'benzo[{letter}]imidazo[2,1-b][1,3]thiazole'
        network = parse_network(name, resolve_component)
        if network is None or not network.citation or network.atom_count != 12:
            continue
        graph = Chem.MolFromSmiles(network.skeleton)
        if graph is None or not matches_skeleton(network, graph, range(12)):
            continue
        drawing = fused_orientation_frontier(graph, allow_three_face_n_junction=True)
        if not drawing.admitted:
            continue
        proof = peripheral_numbering_frontier(graph, orientation_seeds=drawing.orientation_seeds,
                                             allow_three_face_n_junction=True)
        for row in proof.candidates:
            result.append(dict(name=name, mol=graph, locants=dict(row.atom_locants),
                               perimeter=frozenset(proof.perimeter_cycle)))
    return tuple(result)


@lru_cache(maxsize=16)
def benzo_imidazothiazole_perimeter(parent, edges, numbers):
    from rdkit import Chem

    for template in benzo_imidazothiazole_templates(Chem):
        if template['name'] != parent:
            continue
        graph = template['mol']
        if (tuple(a.GetAtomicNum() for a in graph.GetAtoms()) == numbers
                and {frozenset(edge) for edge in edges} ==
                {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in graph.GetBonds()}):
            return template['perimeter']
    return None


def peri_benzo_ketone_skeletons(mol, Chem):
    """Prove an all-carbon peri-fused ring frame with two benzo components."""
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fused_orientation import fused_orientation_frontier
    from .general_fusion_components import candidate_networks
    from .fusion_networks import matches_skeleton, parse_network
    from .fusion_components import resolve_component
    from .primitive_three_face_fusion import _ordered_face

    count = mol.GetNumAtoms()
    faces = mol.GetRingInfo().AtomRings()
    if (not 18 <= count <= 24 or not 5 <= len(faces) <= 6
            or any(len(ring) != 6 for ring in faces)
            or any(a.GetAtomicNum() != 6 or a.GetDegree() not in {2, 3}
                   or a.GetTotalNumHs() > 2 for a in mol.GetAtoms())
            or sum(a.GetTotalNumHs() == 2 for a in mol.GetAtoms()) != 2):
        return ()
    graph = peripheral_numbering_frontier(mol, allow_peri_fused=True, allow_larger_system=True)
    if not graph.admitted or len(graph.perimeter_cycle) == count:
        return ()
    cycles = tuple(_ordered_face(mol, face) for face in graph.ring_faces)
    if any(not face for face in cycles):
        return ()
    winners = candidate_networks(mol, cycles, Chem, allow_peri_fused=True)
    if len(winners) != 1:
        return ()
    name, network = winners[0]
    if (len(network.parents) != 1 or len(network.links) != 2
            or any(link.target != network.parents[0] or network.names[link.child] != 'benzene'
                   or len(link.letters) != 2 for link in network.links)):
        return ()
    descriptors = sorted(''.join(letter for letter, _ in link.letters) for link in network.links)
    parent = 'dibenzo[' + ','.join(descriptors) + ']' + network.names[network.parents[0]]
    compact = parse_network(parent, resolve_component)
    if (compact is None or compact.skeleton != network.skeleton
            or not matches_skeleton(compact, mol, range(count))):
        return ()
    drawing = fused_orientation_frontier(mol, allow_peri_fused=True, allow_larger_system=True)
    if not drawing.admitted:
        return ()
    numbering = peripheral_numbering_frontier(
        mol, orientation_seeds=drawing.orientation_seeds, allow_peri_fused=True, allow_larger_system=True)
    edges = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    return tuple((parent, tuple(range(count)), tuple(dict(row.atom_locants)[i] for i in range(count)),
                  (6,) * count, edges, False, (), 'component-fused-peri-ketone')
                 for row in numbering.candidates)


@lru_cache(maxsize=16)
def peri_benzo_perimeter(parent, edges, size):
    from rdkit import Chem
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fusion_components import resolve_component
    from .fusion_networks import parse_network, skeleton_key

    network = parse_network(parent, resolve_component)
    if network is None or network.atom_count != size or network.skeleton != skeleton_key([6] * size, edges):
        return None
    graph = Chem.RWMol()
    for _ in range(size):
        graph.AddAtom(Chem.Atom(6))
    for left, right in edges:
        graph.AddBond(left, right, Chem.BondType.SINGLE)
    graph = graph.GetMol()
    Chem.SanitizeMol(graph)
    proof = peripheral_numbering_frontier(graph, allow_peri_fused=True, allow_larger_system=True)
    return frozenset(proof.perimeter_cycle) if proof.admitted else None


def benzo_heterocycle_skeletons(mol):
    """P-25.2.2.4 numbering and vocabulary from two actual annular faces."""
    from .ring_state_parent import _hw_name, _monocyclic_skeletons
    rings = tuple(map(frozenset, mol.GetRingInfo().AtomRings()))
    if len(rings) != 2:
        return
    shared = rings[0] & rings[1]
    if (len(shared) != 2 or mol.GetBondBetweenAtoms(*sorted(shared)) is None
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in shared)):
        return
    benzene = next((ring for ring in rings if len(ring) == 6
        and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ring)), None)
    if benzene is None:
        return
    hetero = next(ring for ring in rings if ring != benzene)
    numbers = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in hetero]
    if (len(hetero) not in {5, 6, 7} or any(n not in {6,7,8,16} for n in numbers)
            or not 1 <= sum(n != 6 for n in numbers) <= 4):
        return
    def outer_path(face, start, end):
        path = [start]
        current = next(n.GetIdx() for n in mol.GetAtomWithIdx(start).GetNeighbors()
                       if n.GetIdx() in face and n.GetIdx() != end)
        while current != end:
            if current in path:
                return ()
            path.append(current)
            candidates = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
                          if n.GetIdx() in face and n.GetIdx() != path[-2]]
            if len(candidates) != 1:
                return ()
            current = candidates[0]
        return (*path, end)
    rows = []
    for start, end in (tuple(sorted(shared)), tuple(reversed(sorted(shared)))):
        first, second = outer_path(hetero,start,end), outer_path(benzene,end,start)
        if not first or not second:
            return
        # Start at the heterocycle atom adjacent to a junction; the original
        # whole-system numbering continues round the one outer perimeter.
        seq = (*first[1:], *second[1:])
        locants = [];number=0
        for idx in seq:
            if idx in shared:
                locants.append(str(number)+'a')
            else:
                number += 1;locants.append(str(number))
        element_numbers = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in seq)
        hetero_set = tuple(int(loc) for loc,n in zip(locants,element_numbers) if n != 6)
        ordered = tuple(tuple(int(loc) for loc,n in zip(locants,element_numbers) if n == e)
                        for e in (8,16,7))
        # Only the HW ending/counts are needed: the compulsory locants belong
        # to this bicyclic graph, not the isolated monocycle numbering.
        component_numbers = next(_monocyclic_skeletons(mol, first))[3]
        component = _hw_name(component_numbers,False)
        stem = re.sub(r'^[0-9,]+-', '', component)
        prefix = 'benz' if stem[0] in 'aeiou' else 'benzo'
        locations = tuple(loc for group in ordered for loc in group)
        parent = ','.join(map(str,locations))+'-'+prefix+stem
        inverse = {idx:j for j,idx in enumerate(seq)}
        edges = tuple((inverse[b.GetBeginAtomIdx()],inverse[b.GetEndAtomIdx()])
                      for b in mol.GetBonds() if b.GetBeginAtomIdx() in inverse
                      and b.GetEndAtomIdx() in inverse)
        rows.append(((hetero_set,ordered),(parent,seq,tuple(locants),element_numbers,
                    edges,False,(), 'benzo-heterocycle')))
    best = min(key for key,_ in rows)
    yield from (row for key,row in rows if key == best)


def _preferred_component_vocabulary(name):
    from .fusion_components import _FUSION, _HYDROCARBON_FUSION, _HW_NAMES
    match = _FUSION.fullmatch(name)
    if match is not None:
        prefix, numbers, letter, base = match.groups()
    else:
        match = _HYDROCARBON_FUSION.fullmatch(name)
        if match is None:
            return name
        prefix, letter, base = match.groups();numbers = ''
    for alias, hw in _HW_NAMES.items():
        if prefix == alias.removesuffix('e')+'o':
            prefix = hw.removesuffix('e')+'o'
    # P-25.3.2.2.3 retained component contractions, independent of the
    # assembled molecule. Older resource aliases include uncontracted forms.
    prefix = {'furano':'furo','thiopheno':'thieno','imidazolo':'imidazo',
              'pyridino':'pyrido','pyrimidino':'pyrimido',
              'isoquino':'isoquinolino','isochinolino':'isoquinolino',
              'quino':'quinolino','chinolino':'quinolino',
              'isooxazolo':'[1,2]oxazolo','isothiazolo':'[1,2]thiazolo'}.get(prefix,prefix)
    base = {'isochinoline': 'isoquinoline', 'chinoline': 'quinoline'}.get(base, base)
    base = _HW_NAMES.get(base, base)
    # The terminal monocyclic hydrocarbon's numeric descriptor is omitted
    # under P-25.3.8. Ordinary heterocomponents retain both numbers.
    if prefix in {'benzo','cyclopenta','cyclohepta'}:
        numbers = ''
    return prefix+'['+(numbers+'-' if numbers else '')+letter+']'+base


def _two_component_preference(left, right):
    """Compare proved two-face constructions in P-25.3.1.3 order.

    The location rules in P-25.3.4 concern polycomponent constructions.
    With one parent and one attached component, the earlier parent letter
    precedes the attached-component numbers. Do not use the polycomponent
    network's first-order location criteria to reverse that source order.
    """
    from .fusion_components import compare_components
    from .parent_selection import locant_key

    if any(len(network.parents) != 1 or len(network.components) != 2
           or len(network.links) != 1
           or network.links[0].target != network.parents[0]
           or len(network.links[0].letters) != 1
           or len(network.links[0].child_locants) != 2
           for network in (left, right)):
        return None
    for a, b in ((left.parent, right.parent),
                 (left.components[left.links[0].child],
                  right.components[right.links[0].child])):
        comparison = compare_components(a, b).preference
        if comparison != 0:
            return comparison
    def descriptor(network):
        link = network.links[0]
        numbers = tuple(locant_key(value) for value in link.child_locants)
        if any(value is None for value in numbers):
            return None
        return (link.letters[0], tuple(sorted(numbers)), numbers)
    a, b = descriptor(left), descriptor(right)
    return None if a is None or b is None else (a > b) - (a < b)


def two_face_fusion_templates(mol, Chem):
    rings = tuple(map(frozenset, mol.GetRingInfo().AtomRings()))
    if len(rings) != 2 or any(len(ring) not in {5, 6, 7} for ring in rings):
        return ()
    shared = rings[0] & rings[1]
    if (len(shared) != 2 or mol.GetBondBetweenAtoms(*sorted(shared)) is None
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6,7} for i in shared)
            or sum(a.GetAtomicNum() != 6 for a in mol.GetAtoms() if a.IsInRing()) > 4):
        return ()
    features = molecule_features(mol, Chem)
    if len(features['ring_systems']) != 1:
        return ()
    system = ring_system_with_graph_signatures(mol, features['ring_systems'][0], Chem)
    graph = Chem.RWMol()
    original = tuple(system['atom_indices']);inverse={idx:j for j,idx in enumerate(original)}
    for idx in original:
        graph.AddAtom(Chem.Atom(mol.GetAtomWithIdx(idx).GetAtomicNum()))
    for bond in mol.GetBonds():
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if a in inverse and b in inverse:
            graph.AddBond(inverse[a],inverse[b],Chem.BondType.SINGLE)
    topology=graph.GetMol();Chem.SanitizeMol(topology)
    productive = _productive_templates(Chem.MolToSmiles(topology,canonical=True),Chem)
    if productive:
        return productive
    return _templates(
        json.dumps(ring_system_feature_key(system), separators=(',', ':')),
        system['graph_signature'], system['element_graph_signature'], Chem,
    )


@lru_cache(maxsize=1)
def _dibenzopyran_networks():
    """Construct the bounded dibenzo-pyran component family from P-25.

    Two benzene rings are attached to distinct pyran sides. The component
    parser constructs and verifies their labelled unions; no substituted
    molecules or whole-molecule answer names are stored here. Only a proved
    three-face catacondensed graph, with both benzene faces distinct from one
    another, is admitted. P-25.3.5.4 explicitly illustrates this component
    choice for dibenzo[b,d]pyran; P-25.3.6.1 gives the multiplied construction.
    """
    from rdkit import Chem
    from .fusion_components import resolve_component
    from .fusion_networks import parse_network, skeleton_key
    from .catacondensed_numbering import peripheral_numbering_frontier

    # P-25.2.1/Table 2.8 and P-25.3.3 give xanthene its own retained
    # parent and special numbering. Its frame can also be decomposed into
    # benzene-pyran components but must not be renamed by that route.
    xanthene = resolve_component('xanthene')
    if xanthene is None:
        return ()
    elements = dict(xanthene.label_elements)
    labels = tuple(elements)
    position = {label: i for i, label in enumerate(labels)}
    xanthene_graph = skeleton_key(
        [elements[label] for label in labels],
        [tuple(position[label] for label in edge) for edge in xanthene.bonds])
    results = []
    for left, right in combinations('abcdef', 2):
        name = f'dibenzo[{left},{right}]pyran'
        network = parse_network(name, resolve_component)
        if (network is None or network.citation is not True
                or network.skeleton == xanthene_graph):
            continue
        mol = Chem.MolFromSmiles(network.skeleton)
        if mol is None:
            continue
        graph = peripheral_numbering_frontier(mol)
        if not graph.admitted or len(graph.ring_faces) != 3:
            continue
        faces = [tuple(sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in face))
                 for face in graph.ring_faces]
        if (faces.count((6, 6, 6, 6, 6, 6)) != 2
                or faces.count((6, 6, 6, 6, 6, 8)) != 1):
            continue
        carbon_faces = [face for face in graph.ring_faces
                        if all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in face)]
        if set(carbon_faces[0]) & set(carbon_faces[1]):
            continue  # A retained naphthalene component would need separate selection.
        results.append((name, network))
    return tuple(results)


def dibenzopyran_fusion_templates(mol, Chem):
    """Generate numbered fusion templates only after component/graph proof."""
    from .fusion_components import compare_constructions
    from .fusion_networks import matches_skeleton
    from .fused_orientation import fused_orientation_frontier
    from .catacondensed_numbering import peripheral_numbering_frontier

    features = molecule_features(mol, Chem)
    if len(features['ring_systems']) != 1:
        return ()
    core = features['ring_systems'][0]['atom_indices']
    candidates = [(name, network) for name, network in _dibenzopyran_networks()
                  if matches_skeleton(network, mol, core)]
    frontier = [(name, network) for name, network in candidates
                if not any(compare_constructions(other, network).preference == -1
                           for _, other in candidates)]
    if len(frontier) != 1:
        return ()  # Unknown/tied component or fusion-locant choices delegate.
    name, network = frontier[0]
    if any(compare_constructions(network, other).preference not in (-1, 0)
           for _, other in candidates):
        return ()
    query = Chem.MolFromSmiles(network.skeleton)
    orientation = fused_orientation_frontier(query)
    if not orientation.admitted:
        return ()
    numbered = peripheral_numbering_frontier(query, orientation_seeds=orientation.orientation_seeds)
    if not numbered.admitted or not numbered.candidates:
        return ()
    return tuple(dict(name=name, mol=query, locants=dict(row.atom_locants),
                      source_family='constructed-dibenzopyran',
                      source_file='Blue Book P-25.3.5.4/P-25.3.6.1')
                 for row in numbered.candidates)


def dibenzopyran_fusion_perimeter(name, edges, size):
    """Validate the matched component topology before returning its perimeter."""
    from rdkit import Chem
    from .fusion_networks import skeleton_key
    from .catacondensed_numbering import peripheral_numbering_frontier

    network = next((network for candidate, network in _dibenzopyran_networks()
                    if candidate == name), None)
    if network is None or network.atom_count != size:
        return None
    query = Chem.MolFromSmiles(network.skeleton)
    source_edges = tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                         for bond in query.GetBonds())
    if skeleton_key([6] * size, edges) != skeleton_key([6] * size, source_edges):
        return None
    topology = Chem.RWMol()
    for _ in range(size):
        topology.AddAtom(Chem.Atom(6))
    for left, right in edges:
        topology.AddBond(left, right, Chem.BondType.SINGLE)
    topology = topology.GetMol()
    try:
        Chem.SanitizeMol(topology)
    except (ValueError, RuntimeError):
        return None
    graph = peripheral_numbering_frontier(topology)
    return frozenset(graph.perimeter_cycle) if graph.admitted else None


@lru_cache(maxsize=1)
def _dibenzo_five_member_networks():
    """Build benzo/benzo fusions of five-member O/S components from P-25."""
    from rdkit import Chem
    from .fusion_components import resolve_component
    from .fusion_networks import parse_network
    from .catacondensed_numbering import peripheral_numbering_frontier

    networks = {}
    for parent, hetero in (('furan', 8), ('thiophene', 16)):
        if resolve_component(parent) is None:
            continue
        for left, right in combinations('abcde', 2):
            name = f'dibenzo[{left},{right}]{parent}'
            network = parse_network(name, resolve_component)
            if (network is None or network.citation is not True
                    or len(network.components) != 3 or len(network.links) != 2
                    or network.ring_count != 3):
                continue
            graph = Chem.MolFromSmiles(network.skeleton)
            if graph is None:
                continue
            proof = peripheral_numbering_frontier(graph)
            if not proof.admitted or len(proof.ring_faces) != 3:
                continue
            face_elements = [tuple(sorted(graph.GetAtomWithIdx(i).GetAtomicNum() for i in face))
                             for face in proof.ring_faces]
            if (face_elements.count((6, 6, 6, 6, 6, 6)) != 2
                    or face_elements.count(tuple(sorted((6, 6, 6, 6, hetero)))) != 1):
                continue
            benzene_faces = [face for face in proof.ring_faces
                             if all(graph.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in face)]
            if set(benzene_faces[0]) & set(benzene_faces[1]):
                continue
            networks[name] = network
    return tuple(sorted(networks.items()))


def dibenzo_five_member_fusion_templates(mol, Chem):
    """Number a matching component fusion on its proved whole-core graph."""
    from .fusion_components import compare_constructions
    from .fusion_networks import matches_skeleton
    from .fused_orientation import fused_orientation_frontier
    from .catacondensed_numbering import peripheral_numbering_frontier

    features = molecule_features(mol, Chem)
    if len(features['ring_systems']) != 1:
        return ()
    core = features['ring_systems'][0]['atom_indices']
    candidates = [(name, network) for name, network in _dibenzo_five_member_networks()
                  if matches_skeleton(network, mol, core)]
    frontier = [(name, network) for name, network in candidates
                if not any(compare_constructions(other, network).preference == -1
                           for _, other in candidates)]
    if len(frontier) != 1:
        return ()
    name, network = frontier[0]
    if any(compare_constructions(network, other).preference not in (-1, 0)
           for _, other in candidates):
        return ()
    query = Chem.MolFromSmiles(network.skeleton)
    orientation = fused_orientation_frontier(query)
    if not orientation.admitted:
        return ()
    numbering = peripheral_numbering_frontier(query, orientation_seeds=orientation.orientation_seeds)
    if not numbering.admitted or not numbering.candidates:
        return ()
    return tuple(dict(name=name, mol=query, locants=dict(row.atom_locants),
                      source_family='constructed-dibenzo-five-member',
                      source_file='Blue Book P-25.3/P-66.1.1.4.3')
                 for row in numbering.candidates)


def dibenzo_five_member_fusion_perimeter(name, edges, size, numbers):
    """Validate an element-labelled component network before using its perimeter."""
    from rdkit import Chem
    from .fusion_networks import skeleton_key
    from .catacondensed_numbering import peripheral_numbering_frontier

    network = next((network for candidate, network in _dibenzo_five_member_networks()
                    if candidate == name), None)
    if network is None or network.atom_count != size or len(numbers) != size:
        return None
    query = Chem.MolFromSmiles(network.skeleton)
    source_edges = tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in query.GetBonds())
    if skeleton_key(numbers, edges) != skeleton_key(
            [atom.GetAtomicNum() for atom in query.GetAtoms()], source_edges):
        return None
    topology = Chem.RWMol()
    for _ in range(size):
        topology.AddAtom(Chem.Atom(6))
    for left, right in edges:
        topology.AddBond(left, right, Chem.BondType.SINGLE)
    topology = topology.GetMol()
    try:
        Chem.SanitizeMol(topology)
    except (ValueError, RuntimeError):
        return None
    proof = peripheral_numbering_frontier(topology)
    return frozenset(proof.perimeter_cycle) if proof.admitted else None


@lru_cache(maxsize=512)
def _productive_templates(smiles, Chem):
    """Join two independently numbered HW units and number the whole graph."""
    from .ring_state_parent import _hw_name, _monocyclic_skeletons
    from .fusion_components import resolve_component
    from .fusion_networks import parse_network, matches_skeleton
    from .fused_orientation import fused_orientation_frontier
    from .catacondensed_numbering import peripheral_numbering_frontier

    mol=Chem.MolFromSmiles(smiles)
    rings=tuple(map(tuple,mol.GetRingInfo().AtomRings()))
    shared=frozenset(rings[0]) & frozenset(rings[1])
    faces=[]
    for ring in rings:
        if not any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6 for i in ring):
            # Homogeneous five- and seven-member carbon components are
            # attached cyclopenta/cyclohepta units. Benzene uses the separate
            # P-25.2.2.4 constructor.
            if len(ring) not in {5, 7}:return ()
            carbon_component = {5: 'cyclopenta', 7: 'cyclohepta'}[len(ring)]
            choices=[]
            for direction in (ring,tuple(reversed(ring))):
                for start in range(len(ring)):
                    seq=direction[start:]+direction[:start]
                    choices.append((carbon_component,seq))
        else:
            choices=[(_hw_name(skeleton[3],False),skeleton[1])
                     for skeleton in _monocyclic_skeletons(mol,ring)]
        component_rows=[]
        for name,seq in choices:
            name=re.sub(r'^([0-9,]+)-',r'[\1]',name)
            component=resolve_component(name)
            if component is not None:
                component_rows.append((name,seq,component))
        if not component_rows:return ()
        faces.append(component_rows)
    candidates={}
    for parent_side in (0,1):
        for base,parent_seq,parent in faces[parent_side]:
            if base in {'cyclopenta', 'cyclohepta'}:continue
            parent_positions={idx:j+1 for j,idx in enumerate(parent_seq)}
            endpoint_positions=sorted(parent_positions[idx] for idx in shared)
            side=(endpoint_positions[0] if endpoint_positions[1]-endpoint_positions[0]==1
                  else len(parent_seq) if endpoint_positions==[1,len(parent_seq)] else None)
            if side is None:continue
            letter=chr(ord('a')+side-1)
            for attached_name,attached_seq,attached in faces[1-parent_side]:
                positions={idx:j+1 for j,idx in enumerate(attached_seq)}
                numbers=tuple(positions[idx] for idx in shared)
                prefix=(attached_name.removesuffix('e')+'o'
                        if attached_name not in {'cyclopenta','cyclohepta'} else attached_name)
                prefix={'furano':'furo','thiopheno':'thieno','imidazolo':'imidazo',
                        'pyridino':'pyrido','pyrimidino':'pyrimido'}.get(prefix,prefix)
                for ordered in (numbers,tuple(reversed(numbers))):
                    text=prefix+'['+','.join(map(str,ordered))+'-'+letter+']'+base
                    text=_preferred_component_vocabulary(text)
                    network=parse_network(text,resolve_component)
                    if network is not None and matches_skeleton(network,mol,range(mol.GetNumAtoms())):
                        candidates[text]=network
    frontier=[]
    for text,network in candidates.items():
        if not any(_two_component_preference(network,other)==1
                   for other in candidates.values() if other is not network):
            frontier.append(text)
    if len(frontier)!=1:return ()
    orientation=fused_orientation_frontier(mol)
    if not orientation.admitted:return ()
    numbered=peripheral_numbering_frontier(mol,orientation_seeds=orientation.orientation_seeds)
    if not numbered.admitted or not numbered.candidates:return ()
    # All source-equivalent numberings survive until suffix/H/isotope/stereo
    # criteria operate on the original molecule in the shared ring writer.
    return tuple(dict(name=frontier[0],mol=mol,locants=dict(row.atom_locants),
                      source_family='productive-component-fused',
                      source_file='Blue Book P-22/P-25 graph construction')
                 for row in numbered.candidates)


@lru_cache(maxsize=512)
def _templates(key, graph_key, element_key, Chem):
    from .fused_template_families import _generated_fused_template_element_graph_index
    from .fusion_components import FusionConstruction, fusion_construction, compare_constructions, _INDICATED
    from .resource_templates import build_generated_fused_parent_templates_from_records
    from .ring_state_parent import _peripheral_junction_locants

    records = (_generated_fused_template_element_graph_index().get(key, {})
               .get(graph_key, {}).get(element_key, ()))
    candidates = []
    for record in records:
        name = _preferred_component_vocabulary(_INDICATED.sub('', str(record.get('name', ''))))
        construction = fusion_construction(name)
        if (not isinstance(construction, FusionConstruction)
                or construction.ring_count != 2
                or any(c.ring_count != 1 or c.atom_count not in {5, 6}
                       for c in (construction.parent, construction.attached))):
            continue
        templates = build_generated_fused_parent_templates_from_records(
            Chem, (dict(record, name=name),), 'generated_fused_template_index.json')
        for template in templates:
            query = template['mol']
            if (query.GetNumAtoms() != construction.atom_count
                    or any(a.GetAtomicNum() not in {6, 7, 8, 16}
                           or a.GetFormalCharge() or not a.IsInRing()
                           for a in query.GetAtoms())):
                continue
            labels = _peripheral_junction_locants(query, template['locants'])
            if labels is not None:
                candidates.append((dict(template, locants=labels), construction))
    # This is a complete bounded source frontier, without alias truncation.
    # All physical frames share the indexed element-labelled topology; each
    # input mapping is still proved by the caller's exact topology query.
    def comparison(left, right):
        preference = compare_constructions(left, right).preference
        if preference != 0:
            return preference
        # P-25.3.4.2.4(a-d), specialized to one first-order component:
        # parent letter, attached locant set, then citation order. Do not
        # compare these later criteria when component seniority is unknown.
        a = (left.letter, tuple(sorted(left.numbers)), left.numbers)
        b = (right.letter, tuple(sorted(right.numbers)), right.numbers)
        return (a > b) - (a < b)
    dominated = set()
    for i, (_, left) in enumerate(candidates):
        for j, (_, right) in enumerate(candidates):
            if i != j and comparison(left, right) == 1:
                dominated.add(i)
                break
    frontier = [template for i, (template, _) in enumerate(candidates) if i not in dominated]
    if len({template['name'] for template in frontier}) != 1:
        return ()
    return tuple(frontier)
