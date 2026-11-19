"""A bounded P-25 component frontier for neutral aromatic three-face azines.

The admissible face tree consists of a diazole, a pyridine and a diazine.
The diazole and pyridine share one trivalent nitrogen and one carbon; the
two six-membered faces share a carbon-carbon bond and no diazole atom. Every
component is numbered independently before its shared bonds are assembled.

Within this family the two-face unions cannot be any retained component in
P-25.2.1 Table 2.8: the six/six union has three nitrogen atoms, whereas the
five/six union has two nitrogen atoms, one of which is a fusion atom. There
is no carbocyclic face and hence no P-25.3.5 benzoheterocycle. The three
different components preclude a multiparent or multiplied construction.
The outer caller still treats global parent and method choice as unresolved.
"""
from itertools import product
import re

METHOD = "component-fused-three-face-azine"


def source_supported(mol, core=None):
    """Admit only original aromatic, function-free and unannotated graphs."""
    if mol.GetStereoGroups() or any(
        atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumRadicalElectrons()
        or int(atom.GetChiralTag()) or atom.HasQuery() or atom.GetAtomMapNum()
        for atom in mol.GetAtoms()
    ) or any(int(bond.GetStereo()) or int(bond.GetBondDir()) or bond.HasQuery()
             for bond in mol.GetBonds()):
        return False
    if core is not None and any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in core):
        return False
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    return bool(inventory.supported and not inventory.groups)


def _base_link(child, parent):
    """Bind the child positions to the parent's directed lettered edge."""
    common = set(child) & set(parent)
    if len(common) != 2:
        return None
    for index, first in enumerate(parent):
        second = parent[(index + 1) % len(parent)]
        if {first, second} == common:
            return ((child.index(first) + 1, child.index(second) + 1),
                    chr(ord('a') + index))
    return None


def _higher_links(child, parent):
    common = set(child) & set(parent)
    if len(common) != 2:
        return ()
    first, second = sorted(common)
    return tuple((tuple(child.index(i) + 1 for i in pair),
                  tuple(parent.index(i) + 1 for i in pair))
                 for pair in ((first, second), (second, first)))


def _component_rows(mol, face):
    from .fusion_components import resolve_component
    from .ring_state_parent import _hw_name, _monocyclic_skeletons

    rows = []
    if all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in face):
        # Monocyclic hydrocarbon fusion prefixes are generated from their
        # ring sizes (P-25.3.2.2.1); every orientation is a possible binding.
        from .constants import ROOTS
        root = ROOTS.get(len(face))
        # The six-membered mancude hydrocarbon component is benzene, whose
        # preferred fusion prefix is benzo (P-25.3.2.2.3).
        name = 'benzene' if len(face) == 6 else 'cyclo' + root + 'a' if root else ''
        component = resolve_component(name) if name else None
        if component is None:
            return ()
        cycle = tuple(face)
        for direction in (cycle, tuple(reversed(cycle))):
            for start in range(len(cycle)):
                rows.append((name, direction[start:] + direction[:start], component))
        return tuple(rows)
    for skeleton in _monocyclic_skeletons(mol, tuple(face)):
        name = _hw_name(skeleton[3], False)
        name = re.sub(r'^([0-9,]+)-', r'[\1]', name)
        component = resolve_component(name)
        if component is not None:
            rows.append((name, skeleton[1], component))
    return tuple(rows)


def _candidate_networks(mol, faces):
    """Construct every primitive-component orientation for this face tree."""
    from .fusion_components import resolve_component
    from .fusion_networks import _prefix, matches_skeleton, parse_network

    choices = tuple(_component_rows(mol, face) for face in faces)
    sets = tuple(frozenset(face) for face in faces)
    if any(not rows for rows in choices):
        return ()
    candidates = {}
    seen = set()

    def add_candidate(text):
        if text in seen:
            return
        seen.add(text)
        network = parse_network(text, resolve_component)
        if (network is not None and network.citation is True
                and matches_skeleton(network, mol, range(mol.GetNumAtoms()))):
            candidates[text] = network

    for root in range(3):
        others = tuple(index for index in range(3) if index != root)
        neighbors = tuple(index for index in others if len(sets[index] & sets[root]) == 2)
        for selected in product(*choices):
            base, base_seq, _ = selected[root]
            if len(neighbors) == 1:
                first = neighbors[0]
                second = next(index for index in others if index != first)
                first_name, first_seq, _ = selected[first]
                second_name, second_seq, _ = selected[second]
                lower = _base_link(first_seq, base_seq)
                if lower is None:
                    continue
                for high_numbers, low_numbers in _higher_links(second_seq, first_seq):
                    text = (f"{_prefix(second_name)}[{high_numbers[0]}',{high_numbers[1]}':"
                            f"{low_numbers[0]},{low_numbers[1]}]"
                            f"{_prefix(first_name)}[{lower[0][0]},{lower[0][1]}-{lower[1]}]"
                            f"{base}")
                    add_candidate(text)
            elif len(neighbors) == 2:
                parts = []
                for child in neighbors:
                    name, seq, _ = selected[child]
                    link = _base_link(seq, base_seq)
                    if link is None:
                        break
                    prefix = _prefix(name)
                    parts.append((prefix, f'{prefix}[{link[0][0]},{link[0][1]}-{link[1]}]'))
                if len(parts) != 2:
                    continue
                text = ''.join(part for _, part in sorted(parts)) + base
                add_candidate(text)
    return tuple(candidates.items())


def skeletons(mol, Chem):
    """Return only a uniquely ranked, numbered source-supported core."""
    rings = tuple(tuple(face) for face in mol.GetRingInfo().AtomRings())
    faces = tuple(frozenset(face) for face in rings)
    if (len(faces) != 3 or sorted(map(len, faces)) != [5, 6, 6]
            or mol.GetNumAtoms() != 13
            or any(a.GetAtomicNum() not in {6, 7} or not a.GetIsAromatic()
                   or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) for a in mol.GetAtoms())):
        return ()
    five = next(face for face in faces if len(face) == 5)
    sixes = tuple(face for face in faces if len(face) == 6)
    if sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in five) != 2:
        return ()
    middle = next((face for face in sixes
                   if sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in face) == 1), None)
    outer = next((face for face in sixes
                  if sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in face) == 2), None)
    if middle is None or outer is None or five & outer:
        return ()
    for left, right, elements in ((five, middle, [6, 7]), (middle, outer, [6, 6])):
        shared = left & right
        if (len(shared) != 2 or sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in shared) != elements
                or mol.GetBondBetweenAtoms(*tuple(shared)) is None):
            return ()
    from .fusion_components import compare_constructions
    candidates = _candidate_networks(mol, rings)
    winners = [(name, network) for name, network in candidates
               if all(compare_constructions(network, other).preference in (-1, 0)
                      for _, other in candidates)]
    if len(winners) != 1:
        return ()  # Unknown or tied component comparisons are not guessed.
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fused_orientation import fused_orientation_frontier
    drawing = fused_orientation_frontier(mol, allow_three_face_n_junction=True)
    if not drawing.admitted:
        return ()
    numbering = peripheral_numbering_frontier(
        mol, orientation_seeds=drawing.orientation_seeds, allow_three_face_n_junction=True)
    if not numbering.admitted or not numbering.candidates:
        return ()
    seq = tuple(range(mol.GetNumAtoms()))
    numbers = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in seq)
    edges = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    return tuple((winners[0][0], seq, tuple(dict(row.atom_locants)[i] for i in seq),
                  numbers, edges, False, (), METHOD) for row in numbering.candidates)


def perimeter(parent, edges, size):
    """Revalidate the component and all-carbon topology for H-state naming."""
    if size != 13 or "'" not in parent:
        return None
    from rdkit import Chem
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fusion_components import resolve_component
    from .fusion_networks import parse_network, skeleton_key

    network = parse_network(parent, resolve_component)
    if network is None or network.atom_count != size or network.ring_count != 3:
        return None
    query = Chem.MolFromSmiles(network.skeleton)
    if query is None:
        return None
    source_edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in query.GetBonds()]
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
    """Hand off only a rendered fusion with complete original-atom ownership."""
    rings = mol.GetRingInfo().AtomRings()
    if (mol.GetNumAtoms() < 13 or len(rings) < 3
            or not any(len(ring) == 5 for ring in rings)
            or sum(len(ring) == 6 for ring in rings) < 2
            or sum(atom.GetAtomicNum() == 7 and atom.GetIsAromatic()
                   for atom in mol.GetAtoms()) < 4):
        return False
    from .mapped_ring_parents import ring_systems
    cores = tuple(core for core in ring_systems(mol)
                  if len(core) == 13 and sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in core)
                  == [6] * 9 + [7] * 4)
    if len(cores) != 1:
        return False
    if not source_supported(mol, cores[0]):
        # A nested N-aryl sulfonamide may join a mapped fused prefix and its
        # selection limitations to a fully owned whole-molecule decision.
        # Other characteristic groups and source annotations remain outside.
        from .nested_aza_sulfonamide import nested_aza_sulfonamide_decision
        decision = nested_aza_sulfonamide_decision(mol, Chem)
        if decision is None or not decision.name or not decision.preference_issues:
            return False
        roles = {name: frozenset(atoms) for name, atoms, _ in decision.component_roles}
        required = frozenset(atom.GetIdx() for atom in mol.GetAtoms())
        values = tuple(roles.values())
        return bool(len(roles) == len(decision.component_roles)
                    and roles.get('fused-parent') == cores[0]
                    and decision.atoms == required and frozenset().union(*values) == required
                    and all(not values[i] & values[j]
                            for i in range(len(values)) for j in range(i)))
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
                and mapped | construction.embedded_atoms == required
                and record.hydrogen_budget_proved and record.annotation_coverage
                and record.preference_issues)
