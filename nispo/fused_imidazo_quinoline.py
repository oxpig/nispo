"""Bounded P-25 construction of imidazole-fused quinoline imines.

Two independently constructed components are fused at source graph edges.
Their numbering is additionally checked against a finite, independently
numbered component-graph witness before any source substituent is rendered.
The complete whole-name parent/method competition remains unresolved.
"""
from functools import lru_cache
import hashlib
import json

from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def network_identity(network):
    """Identity of a complete two-component, single-link construction."""
    if (network is None or len(network.parents) != 1 or len(network.components) != 2
            or len(network.links) != 1 or len(network.names) != 2):
        return None
    payload = {
        'components': [(component.identity, name)
                       for component, name in zip(network.components, network.names, strict=True)],
        'parents': network.parents,
        'links': [(link.child, link.target, link.child_locants, link.target_locants, link.letters)
                  for link in network.links],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def networks():
    """Enumerate both parent orientations for quinoline/isoquinoline fusions."""
    from .fusion_components import resolve_component
    from .fusion_networks import _lettered_path, _perimeter, parse_network

    candidates = {}
    for prefix, child_name, parent_name in (
            ('imidazo', 'imidazole', 'quinoline'),
            ('imidazo', 'imidazole', 'isoquinoline'),
            ('quino', 'quinoline', 'imidazole'),
            ('isoquino', 'isoquinoline', 'imidazole')):
        child, parent = resolve_component(child_name), resolve_component(parent_name)
        if child is None or parent is None:
            return ()
        perimeter = _perimeter(parent)
        if perimeter is None:
            return ()
        child_elements, parent_elements = dict(child.label_elements), dict(parent.label_elements)
        edges = sorted(tuple(sorted(edge, key=int)) for edge in child.bonds
                       if len(edge) == 2 and all(label.isdigit() and child_elements[label] == 6
                                                  for label in edge))
        for first, second in edges:
            for left, right in ((first, second), (second, first)):
                for position in range(len(perimeter)):
                    letter = chr(ord('a') + position)
                    target = _lettered_path(parent, letter)
                    if target is None or any(parent_elements[label] != 6 for label in target[0]):
                        continue
                    name = f'{prefix}[{left},{right}-{letter}]{parent_name}'
                    network = parse_network(name, resolve_component)
                    if network is not None and network.citation is True:
                        candidates[name] = network
    return tuple(sorted(candidates.items()))


@lru_cache(maxsize=1)
def _witnesses():
    from .resource_io import read_json_resource
    try:
        table = read_json_resource('resources', 'imidazo_quinoline_numbered_witness.json')
        if table.get('format') != 'source-numbered-two-component-graphs-v1':
            return {}
        rows = table['graphs']
        index = {row['network_id']: row for row in rows}
        return index if len(index) == len(rows) else {}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def numbered_mappings(mol, seq, network):
    """Map every independently numbered component graph onto original atoms.

    This parser-derived numbering is a conservative general-name witness; it
    does not itself prove the preferred orientation under P-25.
    """
    from rdkit import Chem
    entry = _witnesses().get(network_identity(network))
    seq = tuple(seq)
    if entry is None or len(seq) != 13 or len(set(seq)) != 13:
        return ()
    try:
        elements, labels, edges = tuple(entry['elements']), tuple(entry['locants']), tuple(entry['edges'])
        if (len(elements) != 13 or len(labels) != 13 or len(edges) != 15
                or len(set(labels)) != 13
                or any(number not in {6, 7} for number in elements)):
            return ()
        def graph(numbers, pairs):
            result = Chem.RWMol()
            for number in numbers:
                atom = Chem.Atom(number)
                atom.SetNoImplicit(True)
                result.AddAtom(atom)
            for left, right in pairs:
                result.AddBond(left, right, Chem.BondType.SINGLE)
            return result.GetMol()
        reference = graph(elements, edges)
        positions = {idx: index for index, idx in enumerate(seq)}
        source_edges = [(positions[b.GetBeginAtomIdx()], positions[b.GetEndAtomIdx()])
                        for b in mol.GetBonds()
                        if b.GetBeginAtomIdx() in positions and b.GetEndAtomIdx() in positions]
        if len(source_edges) != 15:
            return ()
        target = graph((mol.GetAtomWithIdx(i).GetAtomicNum() for i in seq), source_edges)
        matches = target.GetSubstructMatches(reference, uniquify=False, maxMatches=4096)
        if len(matches) == 4096:
            return ()
        maps = set()
        for match in matches:
            local = [''] * len(seq)
            for reference_idx, target_idx in enumerate(match):
                local[target_idx] = labels[reference_idx]
            if all(local):
                maps.add(tuple(local))
        return tuple(sorted(maps))
    except (KeyError, TypeError, ValueError, RuntimeError, IndexError):
        return ()


@lru_cache(maxsize=256)
def perimeter(parent, edges, size):
    """Prove the perimeter of this particular three-face component union."""
    if size != 13:
        return None
    from rdkit import Chem
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fusion_components import resolve_component
    from .fusion_networks import parse_network, skeleton_key
    network = parse_network(parent, resolve_component)
    if (network is None or network.citation is not True or network.atom_count != 13
            or network.ring_count != 3 or network.elements != ((6, 10), (7, 3))
            or len(network.components) != 2):
        return None
    query = Chem.MolFromSmiles(network.skeleton)
    if query is None:
        return None
    query_edges = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in query.GetBonds())
    if skeleton_key([6] * size, edges) != skeleton_key([6] * size, query_edges):
        return None
    graph = Chem.RWMol()
    for _ in range(size):
        graph.AddAtom(Chem.Atom(6))
    for left, right in edges:
        graph.AddBond(left, right, Chem.BondType.SINGLE)
    graph = graph.GetMol()
    try:
        Chem.SanitizeMol(graph)
    except (RuntimeError, ValueError):
        return None
    frontier = peripheral_numbering_frontier(graph)
    return (frozenset(frontier.perimeter_cycle) if frontier.admitted
            and len(frontier.ring_faces) == 3 else None)


def _supported_faces(graph):
    """Limit this family to an imidazole fused to the N-containing six-face."""
    faces = tuple(map(frozenset, graph.GetRingInfo().AtomRings()))
    if len(faces) != 3 or sorted(map(len, faces)) != [5, 6, 6]:
        return False
    five = next(face for face in faces if len(face) == 5)
    sixes = tuple(face for face in faces if len(face) == 6)

    def count_n(face):
        return sum(graph.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in face)

    if count_n(five) != 2 or sorted(map(count_n, sixes)) != [0, 1]:
        return False
    aza = next(face for face in sixes if count_n(face) == 1)
    benzene = next(face for face in sixes if count_n(face) == 0)
    if five & benzene:
        return False
    for first, second in ((five, aza), (aza, benzene)):
        shared = first & second
        if (len(shared) != 2 or any(graph.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in shared)
                or graph.GetBondBetweenAtoms(*tuple(shared)) is None):
            return False
    return True


def fused_imine_decision(mol, Chem):
    """Render a source-mapped fused imine only after all local proofs pass."""
    if (mol is None or mol.GetNumHeavyAtoms() > 80 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups() or any(
                a.GetAtomicNum() not in {6, 7, 9, 17, 35, 53}
                or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                or int(a.GetChiralTag()) or a.HasQuery() or a.GetAtomMapNum()
                for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status not in {'selected', 'outside-scope'} or frontier.unknown_pairs
            or frontier.cycle or len(frontier.groups) != 1
            or frontier.groups[0].kind != 'imine' or len(inventory.groups) != 1):
        return None
    imine = frontier.groups[0]
    from .imine_components import ordinary_imine
    nitrogen = mol.GetAtomWithIdx(imine.ligand)
    if (not ordinary_imine(mol, imine.center) or nitrogen.GetDegree() != 1
            or nitrogen.GetTotalNumHs(includeNeighbors=True) != 1):
        return None
    from .mapped_ring_parents import ring_systems
    cores = [core for core in ring_systems(mol) if imine.center in core]
    if len(cores) != 1 or len(cores[0]) != 13:
        return None
    core = cores[0]
    if (sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in core) != 3
            or any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in core)):
        return None
    from .three_face_heterolactam_fusion import _core_topology
    copied = _core_topology(mol, Chem, core)
    if copied is None:
        return None
    graph, sequence = copied
    if not _supported_faces(graph):
        return None
    faces = tuple(map(frozenset, graph.GetRingInfo().AtomRings()))
    five = next(face for face in faces if len(face) == 5)
    local_center = sequence.index(imine.center)
    if (local_center not in five or sum(graph.GetAtomWithIdx(n.GetIdx()).GetAtomicNum() == 7
            for n in graph.GetAtomWithIdx(local_center).GetNeighbors() if n.GetIdx() in five) != 2):
        return None
    from .fusion_components import compare_constructions, fusion_construction
    from .fusion_networks import matches_skeleton
    candidates = [(name, network) for name, network in networks()
                  if matches_skeleton(network, graph, range(graph.GetNumAtoms()))]
    winners = [(name, network) for name, network in candidates
               if all(compare_constructions(network, other).preference in (-1, 0)
                      for _, other in candidates)]
    if len(winners) != 1:
        return None
    parent, network = winners[0]
    mappings = numbered_mappings(mol, sequence, network)
    if not mappings:
        return None
    from .naming_context import MoleculeContext
    from .ring_state_parent import _inventory, _reference_states, _render
    context, inv = MoleculeContext(mol, Chem), _inventory(mol, Chem)
    if not inv.supported or tuple(inv.principal) != (imine,):
        return None
    edges = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in graph.GetBonds())
    numbers = tuple(a.GetAtomicNum() for a in graph.GetAtoms())
    scope = frozenset(a.GetIdx() for a in mol.GetAtoms())
    rows = []
    for locants in mappings:
        skeleton = (parent, sequence, locants, numbers, edges, False, (), 'component-fused')
        for reference in _reference_states(numbers, edges):
            rows.extend(_render(context, skeleton, reference, inv, scope=scope))
    if not rows:
        return None
    best = min(row.numbering_key for row in rows)
    finalists = [row for row in rows if row.numbering_key == best]
    if len({row.name for row in finalists}) != 1:
        return None
    row, construction = finalists[0], finalists[0].construction
    if (construction is None or row.fusion_construction != fusion_construction(parent)
            or not row.hydrogen_budget_proved or not row.annotation_coverage
            or row.principal_centers != {imine.center} or frozenset(row.parent_atoms) != core
            or construction.atoms & construction.embedded_atoms
            or construction.atoms | construction.embedded_atoms != scope):
        return None
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch
    claimed = set()
    for root, host, owned, text in construction.prefix_branch_ownership:
        if (host not in core or root not in owned or claimed & owned
                or frozenset(collect_subtree(mol, root, host)) != owned
                or mapped_branch(mol, Chem, root, host) != (text, owned)):
            return None
        cuts = {(idx, nbr.GetIdx()) for idx in owned for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
                if nbr.GetAtomicNum() > 1 and nbr.GetIdx() not in owned}
        if cuts != {(root, host)}:
            return None
        claimed.update(owned)
    issue = SelectionLimitation(
        'fused-imine-parent', 'P-25/P-44/P-45/P-51',
        'The fused components, independently numbered atom graph and imine are source-mapped; '
        'P-25 orientation and the complete preferred-parent/method comparison remain unresolved.')
    return MethodDecision(row.name, 'fused-imine', frozenset({imine.center}), scope,
                          'P-25.3/P-31/P-41/P-62.3', row.atom_locants,
                          preference_issues=(*row.preference_issues, issue))


def requires_full(mol, Chem):
    if (sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()) < 4
            or len(mol.GetRingInfo().AtomRings()) < 3):
        return False
    proof = fused_imine_decision(mol, Chem)
    if proof is None:
        return False
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision
    actual = _actual_principal_decision(mol, Chem)
    return bool(actual is not None and actual.prepared is None
                and actual.output_name == proof.name and actual.comparison == proof
                and set(proof.preference_issues) <= set(actual.preference_issues))
