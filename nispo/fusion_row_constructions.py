"""Two-component P-25 constructions with a proved horizontal-row choice.

Only complete, neutral, aromatic five-hexagon hydrides are admitted. Component
graphs and native locants are independent parent vocabularies; no complete
fused-name answer table is used.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product

from .fusion_components import compare_components, component_from_graph, resolve_component
from .fusion_networks import _prefix, skeleton_key
from .parent_selection import locant_key, locant_set


def _single_graph(elements, edges, Chem):
    graph = Chem.RWMol()
    for element in elements:
        atom = Chem.Atom(element)
        atom.SetNoImplicit(True)
        graph.AddAtom(atom)
    for a, b in edges:
        graph.AddBond(a, b, Chem.BondType.SINGLE)
    mol = graph.GetMol()
    mol.UpdatePropertyCache(False)
    Chem.GetSymmSSSR(mol)
    return mol


@lru_cache(maxsize=1024)
def _component_graph(component):
    from rdkit import Chem

    labels = tuple(label for label, _element in component.label_elements)
    positions = {label: i for i, label in enumerate(labels)}
    edges = [tuple(positions[label] for label in edge) for edge in component.bonds]
    mol = _single_graph([element for _label, element in component.label_elements], edges, Chem)
    rings = tuple(tuple(r) for r in Chem.GetSymmSSSR(mol))
    if len(rings) != component.ring_count:
        return None
    counts = Counter(frozenset((ring[i], ring[(i + 1) % len(ring)]))
                     for ring in rings for i in range(len(ring)))
    if any(count > 2 for count in counts.values()):
        return None
    boundary = defaultdict(set)
    for edge, count in counts.items():
        if count == 1:
            a, b = tuple(edge)
            boundary[labels[a]].add(labels[b])
            boundary[labels[b]].add(labels[a])
    if len(boundary.get('1', ())) != 2 or '2' not in boundary['1']:
        return None
    perimeter = ['1', '2']
    while True:
        next_labels = boundary[perimeter[-1]] - {perimeter[-2]}
        if len(next_labels) != 1:
            return None
        nxt = next(iter(next_labels))
        if nxt == '1':
            break
        if nxt in perimeter:
            return None
        perimeter.append(nxt)
    if len(perimeter) != len(boundary):
        return None
    return mol, labels, tuple(perimeter)


@lru_cache(maxsize=1)
def _component_inventory():
    from rdkit import Chem
    from .preferred_fused import RETAINED_FUSED_HYDRIDES
    from .resource_io import read_json_resource
    from .resource_templates import OPSIN_PARENT_TEMPLATES

    entries = {}

    def add(name, component):
        if component is not None and component.ring_count <= 5:
            entries.setdefault(component.identity, (name, component))

    for name in sorted(RETAINED_FUSED_HYDRIDES | {'benzene'}):
        add(name, resolve_component(name))
    hydrocarbon = read_json_resource('resources', 'bluebook_retained_hydrocarbon_parents.json')
    for record in hydrocarbon['parents']:
        name = record['name']
        if record['independent_cycle_rank'] > 5 or 'H-' in name:
            continue
        mol = Chem.MolFromSmiles(record['original_frame_smiles'])
        labels = {int(i): label for i, label in record['original_locants'].items()}
        add(name, component_from_graph(mol, labels) if mol is not None else None)
    for name, smiles, labels in OPSIN_PARENT_TEMPLATES:
        if name not in {'tetracene', 'pentacene'}:
            continue
        mol = Chem.MolFromSmiles(smiles)
        add(name, component_from_graph(mol, dict(enumerate(labels))) if mol is not None else None)
    # Tetraphene is a P-25 parent component, with the native numbered graph
    # independently obtained by parsing its standalone component name.
    mol = Chem.MolFromSmiles('C1=CC=CC2=CC=C3C=C4C=CC=CC4=CC3=C12')
    labels = ('1', '2', '3', '4', '4a', '5', '6', '6a', '7', '7a',
              '8', '9', '10', '11', '11a', '12', '12a', '12b')
    add('tetraphene', component_from_graph(mol, dict(enumerate(labels))))
    return tuple(entries.values())


def _edge_set(component, mapping):
    return frozenset(frozenset(mapping[label] for label in edge) for edge in component.bonds)


def horizontal_row_fusion_decision(mol, Chem):
    """Choose a two-component complete hydride only when (g) is decisive."""
    if (len(Chem.GetMolFrags(mol)) != 1 or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or mol.GetStereoGroups() or mol.GetNumAtoms() > 24
            or any(a.GetAtomicNum() not in {6, 7} or not a.GetIsAromatic()
                   or not a.IsInRing() or a.GetFormalCharge() or a.GetIsotope()
                   or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
                   or a.GetAtomicNum() == 7 and a.GetTotalNumHs(includeNeighbors=True)
                   for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or b.GetBondDir() != Chem.BondDir.NONE for b in mol.GetBonds())):
        return None
    rings = tuple(tuple(r) for r in Chem.GetSymmSSSR(mol))
    rank = mol.GetNumBonds() - mol.GetNumAtoms() + 1
    if rank != 5 or len(rings) != 5 or any(len(r) != 6 for r in rings):
        return None
    edges = frozenset(frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds())
    elements = tuple(a.GetAtomicNum() for a in mol.GetAtoms())
    target = _single_graph(elements, (tuple(edge) for edge in edges), Chem)
    target_key = skeleton_key(elements, (tuple(edge) for edge in edges))
    choices = []
    for name, component in _component_inventory():
        info = _component_graph(component)
        if info is None or any(e not in {6, 7} for e, _count in component.elements):
            continue
        graph, labels, perimeter = info
        component_edges = [tuple(edge) for edge in component.bonds]
        if component.ring_count == rank and component.atom_count == mol.GetNumAtoms():
            if skeleton_key([element for _label, element in component.label_elements],
                            (tuple(labels.index(label) for label in edge) for edge in component_edges)) == target_key:
                return None  # A whole retained or preselected parent takes priority.
        if component.ring_count not in {1, 2, 3, 4} or component.atom_count >= mol.GetNumAtoms():
            continue
        matches = target.GetSubstructMatches(graph, uniquify=False, maxMatches=2001)
        if len(matches) == 2001:
            return None
        if matches:
            choices.append((name, component, labels, perimeter, matches))
    all_atoms = frozenset(range(mol.GetNumAtoms()))
    candidates = []
    for parent_entry, child_entry in product(choices, repeat=2):
        parent_name, parent, parent_labels, perimeter, parent_matches = parent_entry
        child_name, child, child_labels, child_perimeter, child_matches = child_entry
        if (parent.ring_count + child.ring_count != rank
                or (parent.ring_count, child.ring_count) not in {(4, 1), (3, 2)}
                or child.ring_count == 1 and child_name != 'benzene'):
            continue
        # A bicyclic attached unit is accepted only as a retained hydride;
        # it is not split into a second-order benzene/heterocycle name.
        if child.ring_count == 2:
            from .preferred_fused import RETAINED_FUSED_HYDRIDES
            if child_name not in RETAINED_FUSED_HYDRIDES:
                continue
        if compare_components(parent, child).preference not in {-1, 0}:
            continue
        for pm in parent_matches:
            parent_map = dict(zip(parent_labels, pm, strict=True))
            parent_atoms = frozenset(pm)
            parent_edges = _edge_set(parent, parent_map)
            for cm in child_matches:
                child_atoms = frozenset(cm)
                common = parent_atoms & child_atoms
                if (parent_atoms | child_atoms != all_atoms or not 2 <= len(common) <= 4
                        or child.ring_count == 2 and len(common) != 2):
                    continue
                child_map = dict(zip(child_labels, cm, strict=True))
                child_edges = _edge_set(child, child_map)
                if parent_edges | child_edges != edges or len(parent_edges & child_edges) != len(common) - 1:
                    continue
                inverse_child = {child_map[label]: label for label in child_perimeter}
                if not common <= inverse_child.keys():
                    continue
                for start in range(len(perimeter)):
                    path = tuple(perimeter[(start + i) % len(perimeter)] for i in range(len(common)))
                    if {parent_map[label] for label in path} != common:
                        continue
                    cpath = tuple(inverse_child[parent_map[label]] for label in path)
                    if not any(all(cpath[i + 1] == child_perimeter[(child_perimeter.index(cpath[i]) + step)
                                                                  % len(child_perimeter)]
                                   for i in range(len(cpath) - 1)) for step in (-1, 1)):
                        continue
                    letters = ''.join(chr(ord('a') + (start + i) % len(perimeter))
                                      for i in range(len(common) - 1))
                    candidates.append((parent_name, parent, child_name, child, letters, cpath,
                                       parent_atoms, child_atoms - parent_atoms))
    if not candidates:
        return None
    parents = {candidate[1].identity: candidate[1] for candidate in candidates}
    best = []
    for identity, parent in parents.items():
        comparisons = [compare_components(parent, other) for other_id, other in parents.items()
                       if other_id != identity]
        if all(row.preference in {-1, 0} for row in comparisons):
            if any(row.preference == -1 and row.rule == 'P-25.3.2.4(g)' for row in comparisons):
                best.append(identity)
    if len(best) != 1:
        return None
    selected = [candidate for candidate in candidates if candidate[1].identity == best[0]]
    if len({candidate[3].identity for candidate in selected}) != 1:
        return None
    rendered = []
    for parent_name, _parent, child_name, _child, letters, cpath, parent_atoms, child_atoms in selected:
        if child_name == 'benzene':
            name = f'benzo[{letters}]{parent_name}'
            numbers = ()
        else:
            prefix = _prefix(child_name)
            if not prefix:
                return None
            numbers = cpath
            name = f"{prefix}[{','.join(numbers)}-{letters}]{parent_name}"
        key = (tuple(sorted(letters)), letters, locant_set(numbers), tuple(map(locant_key, numbers)))
        rendered.append((key, name, parent_atoms, child_atoms))
    minimum = min(row[0] for row in rendered)
    winners = [row for row in rendered if row[0] == minimum]
    if len({row[1] for row in winners}) != 1:
        return None
    _key, name, parent_atoms, child_atoms = winners[0]
    from .bluebook_azoxy_methods import _decision
    return _decision(mol, name, (), 'P-25.3.2.4(g)/P-25.3.4.2.4/P-25.3.8.1',
                     (("parent-component", parent_atoms, ()), ("attached-component", child_atoms, ())))
