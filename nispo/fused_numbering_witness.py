"""Independent numbered-component graph witness for bounded aza fusions.

The source-derived table is a complete enumeration of the 54 neutral 5/6/6
component graphs, not a molecule list. P-25 constructs and numbers each
candidate separately; this module only declines a construction if the full
numbered atom graph differs from the independent parser's numbered graph.
No name, molecular prediction, Java process or OPSIN call is used at runtime.
"""
from functools import lru_cache
import hashlib
import json


def network_identity(network):
    """Bind evidence to a complete labelled component-link construction."""
    if (network is None or len(network.parents) != 1 or len(network.components) != 3
            or len(network.links) != 2 or len(network.names) != 3):
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
def _witnesses():
    from .resource_io import read_json_resource
    try:
        table = read_json_resource('resources', 'three_face_aza_numbered_witness.json')
        if table.get('format') != 'source-numbered-component-graphs-v1':
            return {}
        rows = table.get('graphs', ())
        index = {row['network_id']: row for row in rows}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}
    return index if len(index) == len(rows) == 54 else {}


def numbering_matches(mol, seq, locants, parent):
    """Match every original atom/locant, considering all graph isomorphisms."""
    from rdkit import Chem
    from .fusion_components import resolve_component
    from .fusion_networks import parse_network

    seq, locants = tuple(seq), tuple(map(str, locants))
    if len(seq) != 13 or len(locants) != 13 or len(set(seq)) != 13 or len(set(locants)) != 13:
        return False
    network = parse_network(parent, resolve_component)
    identity = network_identity(network)
    entry = _witnesses().get(identity)
    if entry is None:
        return False
    try:
        elements = tuple(entry['elements'])
        numbered = tuple(entry['locants'])
        reference_edges = tuple(tuple(edge) for edge in entry['edges'])
    except (KeyError, TypeError):
        return False
    if (len(elements) != 13 or len(numbered) != 13
            or any(not isinstance(locant, str) or not locant for locant in numbered)
            or len(set(numbered)) != 13
            or any(number not in {6, 7} for number in elements)
            or len(reference_edges) != 15
            or any(len(edge) != 2 or any(not isinstance(i, int) or i < 0 or i >= 13 for i in edge)
                   or edge[0] == edge[1] for edge in reference_edges)
            or len({frozenset(edge) for edge in reference_edges}) != 15
            or any(not isinstance(i, int) or i < 0 or i >= mol.GetNumAtoms() for i in seq)):
        return False
    def topology(numbers, edges):
        graph = Chem.RWMol()
        for number in numbers:
            atom = Chem.Atom(number)
            atom.SetNoImplicit(True)
            graph.AddAtom(atom)
        for left, right in edges:
            graph.AddBond(left, right, Chem.BondType.SINGLE)
        return graph.GetMol()
    reference = topology(elements, reference_edges)
    positions = {idx: i for i, idx in enumerate(seq)}
    edges = [(positions[b.GetBeginAtomIdx()], positions[b.GetEndAtomIdx()]) for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in positions and b.GetEndAtomIdx() in positions]
    if len(edges) != len(reference_edges):
        return False
    target = topology((mol.GetAtomWithIdx(i).GetAtomicNum() for i in seq), edges)
    # Enumerate all isomorphisms, not just the first symmetry choice. If the
    # engineering bound is reached, the unproved remainder blocks admission.
    matches = target.GetSubstructMatches(reference, uniquify=False, maxMatches=4096)
    return bool(len(matches) < 4096 and any(
        all(locants[target_index] == numbered[reference_index]
            for reference_index, target_index in enumerate(match))
        for match in matches))
