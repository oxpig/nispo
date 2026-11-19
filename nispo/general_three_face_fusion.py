"""Generic source-mapped, two-component, three-face fusion proposals.

The static input is an abstract component vocabulary, not molecule answers.
Every proposed name is joined to an independent OPSIN-numbered graph and the
original source topology. Whole-name preference and P-25 orientation remain
explicitly unresolved for this producer.
"""
from functools import lru_cache
import json


@lru_cache(maxsize=1)
def _witnesses():
    from .resource_io import read_json_resource
    try:
        table = read_json_resource('resources', 'general_three_face_numbered_witness_v2.json')
        if table.get('format') != 'source-numbered-two-component-graphs-v1':
            return {}
        rows = table['graphs']
        index = {row['network_id']: row for row in rows}
        return index if len(index) == len(rows) else {}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def numbered_mappings(mol, seq, network):
    """Map every witnessed component numbering to the original core atoms."""
    from rdkit import Chem
    from .fused_imidazo_quinoline import network_identity

    entry = _witnesses().get(network_identity(network))
    seq = tuple(seq)
    if entry is None or len(seq) != network.atom_count or len(set(seq)) != len(seq):
        return ()
    try:
        elements = tuple(entry['elements'])
        labels = tuple(entry['locants'])
        edges = tuple(entry['edges'])
        size = len(seq)
        if (len(elements) != size or len(labels) != size or len(edges) != size + 2
                or len(set(labels)) != size or any(n not in {6, 7, 8, 16} for n in elements)):
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
        if len(source_edges) != size + 2:
            return ()
        target = graph((mol.GetAtomWithIdx(i).GetAtomicNum() for i in seq), source_edges)
        matches = target.GetSubstructMatches(reference, uniquify=False, maxMatches=4096)
        if len(matches) == 4096:
            return ()
        mappings = set()
        for match in matches:
            locants = [''] * size
            for ref_idx, target_idx in enumerate(match):
                locants[target_idx] = labels[ref_idx]
            if all(locants):
                mappings.add(tuple(locants))
        return tuple(sorted(mappings))
    except (KeyError, TypeError, ValueError, RuntimeError, IndexError):
        return ()


def skeletons(mol, Chem):
    """Enumerate an indexed two-component frontier for a three-face core."""
    if (mol.GetNumAtoms() > 18 or mol.GetNumBonds() - mol.GetNumAtoms() + 1 != 3
            or any(a.GetAtomicNum() not in {6, 7, 8, 16} for a in mol.GetAtoms())
            or any(len(face) not in {5, 6} for face in mol.GetRingInfo().AtomRings())):
        return ()
    from .catacondensed_numbering import peripheral_numbering_frontier
    graph_proof = peripheral_numbering_frontier(mol, allow_three_face_n_junction=True)
    if not graph_proof.admitted or len(graph_proof.ring_faces) != 3:
        return ()
    from .template_index import molecule_features, ring_system_feature_key, ring_system_with_graph_signatures
    from .fused_template_families import _generated_fused_template_element_graph_index
    from .fusion_components import resolve_component
    from .fusion_networks import parse_network, matches_skeleton, _render_rerooted
    from .fusion_state_skeletons import _preferred_component_vocabulary, _two_component_preference
    from .fused_imidazo_quinoline import network_identity
    import re

    features = molecule_features(mol, Chem)
    if len(features['ring_systems']) != 1:
        return ()
    system = ring_system_with_graph_signatures(mol, features['ring_systems'][0], Chem)
    key = json.dumps(ring_system_feature_key(system), separators=(',', ':'))
    rows = (_generated_fused_template_element_graph_index().get(key, {})
            .get(system['graph_signature'], {}).get(system['element_graph_signature'], ()))
    witnesses = _witnesses()
    candidates = {}
    for row in rows:
        name = _preferred_component_vocabulary(re.sub(
            r'^(?:\d+[a-z]?H,)*\d+[a-z]?H-', '', str(row.get('name', ''))))
        network = parse_network(name, resolve_component)
        if (network is None or network.citation is not True or len(network.components) != 2
                or len(network.links) != 1 or network.ring_count != 3
                or not matches_skeleton(network, mol, range(mol.GetNumAtoms()))):
            continue
        candidates[name] = network
        # The index may contain only one parent orientation. Re-root the
        # same physical component graph before judging component seniority;
        # an unrenderable better alternative remains a real competitor.
        for root in range(len(network.components)):
            alternate = _render_rerooted(network, (root,))
            if not alternate:
                continue
            alternate = _preferred_component_vocabulary(alternate)
            rerooted = parse_network(alternate, resolve_component)
            if (rerooted is not None and rerooted.citation is True
                    and rerooted.skeleton == network.skeleton
                    and len(rerooted.components) == 2 and len(rerooted.links) == 1
                    and matches_skeleton(rerooted, mol, range(mol.GetNumAtoms()))):
                candidates[alternate] = rerooted
    # A candidate survives only if its comparison with every indexed
    # alternative is known and noninferior. Missing component criteria are
    # not ties, and the index itself is not a complete P-25 universe.
    winners = [(name, network) for name, network in candidates.items()
               if all((preference := _two_component_preference(network, other)) is not None
                      and preference <= 0 for other in candidates.values() if other is not network)]
    if len(winners) != 1:
        return ()
    parent, network = winners[0]
    if network_identity(network) not in witnesses:
        return ()
    seq = tuple(range(mol.GetNumAtoms()))
    numbers = tuple(atom.GetAtomicNum() for atom in mol.GetAtoms())
    edges = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    return tuple((parent, seq, locants, numbers, edges, False, (), 'component-fused-general')
                 for locants in numbered_mappings(mol, seq, network))


def requires_full(mol, Chem):
    """Use the full writer when a witnessed three-face core is available."""
    if len(mol.GetRingInfo().AtomRings()) < 3:
        return False
    from .mapped_ring_parents import _core_graph, ring_systems

    for core in ring_systems(mol):
        if not 8 <= len(core) <= 18:
            continue
        prepared = _core_graph(mol, Chem, core)
        if prepared is not None and skeletons(prepared[0], Chem):
            return True
    return False


@lru_cache(maxsize=512)
def perimeter(parent, edges, size, numbers):
    """Validate the source-element-labelled perimeter for a witnessed name."""
    from rdkit import Chem
    from .fusion_components import resolve_component
    from .fusion_networks import parse_network, skeleton_key
    from .fused_imidazo_quinoline import network_identity
    from .catacondensed_numbering import peripheral_numbering_frontier

    network = parse_network(parent, resolve_component)
    if (network is None or network.ring_count != 3 or network.atom_count != size
            or len(numbers) != size
            or network_identity(network) not in _witnesses()):
        return None
    query = Chem.MolFromSmiles(network.skeleton)
    if query is None:
        return None
    query_edges = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in query.GetBonds())
    if skeleton_key(numbers, edges) != skeleton_key(
            [a.GetAtomicNum() for a in query.GetAtoms()], query_edges):
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
    result = peripheral_numbering_frontier(graph)
    return frozenset(result.perimeter_cycle) if result.admitted else None
