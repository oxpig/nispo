"""Bounded P-28 assemblies of identical simple heteromonocycles.

The operation handles neutral aromatic pyridine, furan, and thiophene units
joined in an unbranched chain by single C-C bonds. It does not extend to
annulated rings, indicated-hydrogen heterocycles, or decorated assemblies.
"""

from dataclasses import dataclass
from itertools import product

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision


_MULTIPLIERS = {2: "bi", 3: "ter", 4: "quater", 5: "quinque", 6: "sexi"}
_UNIT = {
    "pyridine": (6, 7, "pyridine"),
    "furan": (5, 8, "furan"),
    "thiophene": (5, 16, "thiophene"),
}
_RULE = "P-28.1/P-28.2.1/P-28.3.1/P-28.4.1/P-52.2.7.1/P-52.2.7.2"


@dataclass(frozen=True, slots=True)
class HeteroringAssemblyProof:
    components: tuple[frozenset[int], ...]
    path: tuple[int, ...]
    unit: str


def heteroring_assembly_decision(mol, Chem):
    """Return a fully mapped P-28 decision for an admitted heteroring chain."""
    proof = _prove_assembly(mol, Chem)
    if proof is None:
        return None
    component_numberings = tuple(
        _numberings(mol, ring) for ring in proof.components
    )
    if any(not options for options in component_numberings):
        return None

    candidates = []
    paths = (proof.path, tuple(reversed(proof.path)))
    for path in paths:
        for sequences in product(*(component_numberings[i] for i in path)):
            atom_locs = tuple({atom: loc for loc, atom in enumerate(seq, 1)}
                              for seq in sequences)
            connectors = []
            cited = []
            for i in range(len(path) - 1):
                a, b = _connector(mol, proof.components[path[i]], proof.components[path[i + 1]])
                if a is None or a not in atom_locs[i] or b not in atom_locs[i + 1]:
                    return None
                pair = (atom_locs[i][a], atom_locs[i + 1][b])
                connectors.append(pair)
                cited.extend(((i + 1, pair[0]), (i + 2, pair[1])))
            candidates.append((tuple(sorted(cited)), tuple(connectors), path,
                               tuple(sequences)))
    if not candidates:
        return None
    locant_set, connectors, path, sequences = min(
        candidates, key=lambda candidate: (candidate[0], candidate[1])
    )

    count = len(path)
    unit_name = _UNIT[proof.unit][2]
    parent = _MULTIPLIERS[count] + unit_name
    if count == 2:
        # P-28.2.1 gives the general parent-hydride construction. The examples
        # bipyridine and bifuran confirm the form; the same operation applies
        # to the supported thiophene parent hydride.
        first, second = connectors[0]
        locants = f"{first},{second}′"
        name = f"{locants}-{parent}"
    else:
        locants = ":".join(
            f"{i + 1}^{{{left}}},{i + 2}^{{{right}}}"
            for i, (left, right) in enumerate(connectors)
        )
        name = f"{locants}-{parent}"

    role_items = []
    atom_locants = []
    for ring_number, (component_index, sequence) in enumerate(zip(path, sequences), 1):
        local = tuple((atom, str(loc)) for loc, atom in enumerate(sequence, 1))
        role_items.append((f"ring-{ring_number}", proof.components[component_index], local))
        if count == 2:
            prime = "" if ring_number == 1 else "′"
            atom_locants.extend((atom, f"{loc}{prime}")
                                for loc, atom in enumerate(sequence, 1))
        else:
            atom_locants.extend((atom, f"{ring_number}^{loc}")
                                for loc, atom in enumerate(sequence, 1))
    atoms = frozenset().union(*proof.components)
    return MethodDecision(name, "ring-assembly", frozenset(), atoms, _RULE,
                          tuple(atom_locants), component_roles=tuple(role_items))


def prove_heteroring_assembly(mol, Chem):
    """Expose the bounded topology proof for focused callers."""
    return _prove_assembly(mol, Chem)


def _prove_assembly(mol, Chem):
    if mol.GetNumHeavyAtoms() > 60 or mol.GetStereoGroups():
        return None
    for atom in iter_atoms(mol):
        if (atom.GetAtomicNum() not in {6, 7, 8, 16} or atom.GetFormalCharge()
                or atom.GetIsotope() or atom.GetNumRadicalElectrons()
                or int(atom.GetChiralTag())):
            return None
    if any(int(bond.GetStereo()) for bond in iter_bonds(mol)):
        return None
    if any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in iter_atoms(mol)):
        return None
    heavy = {atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1}
    rings = tuple(frozenset(r) for r in mol.GetRingInfo().AtomRings())
    if not 2 <= len(rings) <= 6 or len(set().union(*rings)) != len(heavy):
        return None
    if any(a & b for i, a in enumerate(rings) for b in rings[i + 1:]):
        return None

    unit_names = []
    for ring in rings:
        size = len(ring)
        elements = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring]
        hetero = [z for z in elements if z != 6]
        unit = next((name for name, (expected_size, z, _) in _UNIT.items()
                     if size == expected_size and hetero == [z]), None)
        if unit is None or elements.count(6) != size - 1:
            return None
        internal = [bond for bond in iter_bonds(mol)
                    if bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring]
        if len(internal) != size or any(b.GetBondType() != Chem.BondType.AROMATIC
                                        for b in internal):
            return None
        if any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            return None
        unit_names.append(unit)
    if len(set(unit_names)) != 1:
        return None

    ring_for_atom = {atom: n for n, ring in enumerate(rings) for atom in ring}
    edges = []
    connectors = []
    for bond in iter_bonds(mol):
        left_atom, right_atom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left, right = ring_for_atom.get(left_atom), ring_for_atom.get(right_atom)
        if left == right:
            continue
        if left is None or right is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        if (mol.GetAtomWithIdx(left_atom).GetAtomicNum() != 6
                or mol.GetAtomWithIdx(right_atom).GetAtomicNum() != 6):
            return None
        edges.append(tuple(sorted((left, right))))
        connectors.append((left_atom, right_atom))
    if len(edges) != len(rings) - 1 or len(set(edges)) != len(edges):
        return None
    adjacency = {i: [] for i in range(len(rings))}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    if any(len(neighbors) not in {1, 2} for neighbors in adjacency.values()):
        return None
    ends = [i for i, neighbors in adjacency.items() if len(neighbors) == 1]
    if len(ends) != 2:
        return None
    path = _walk(adjacency, ends[0])
    if path is None or len(path) != len(rings):
        return None
    # An aromatic carbon may have at most one assembly bond in this scope.
    if any(sum(n.GetIdx() not in rings[ring_for_atom[atom.GetIdx()]]
               for n in atom.GetNeighbors()) > 1 for atom in iter_atoms(mol)
           if atom.GetAtomicNum() == 6):
        return None
    return HeteroringAssemblyProof(rings, path, unit_names[0])


def _walk(adjacency, start):
    path, previous, current = [], None, start
    while True:
        if current in path:
            return None
        path.append(current)
        following = [x for x in adjacency[current] if x != previous]
        if not following:
            return tuple(path)
        if len(following) != 1:
            return None
        previous, current = current, following[0]


def _connector(mol, left, right):
    for atom in left:
        for neighbor in mol.GetAtomWithIdx(atom).GetNeighbors():
            if neighbor.GetIdx() in right:
                return atom, neighbor.GetIdx()
    return None, None


def _numberings(mol, ring):
    hetero = next((i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() != 6), None)
    if hetero is None:
        return ()
    neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(hetero).GetNeighbors()
                 if n.GetIdx() in ring]
    if len(neighbors) != 2:
        return ()
    sequences = []
    for next_atom in neighbors:
        sequence = [hetero, next_atom]
        previous, current = hetero, next_atom
        while len(sequence) < len(ring):
            following = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
                         if n.GetIdx() in ring and n.GetIdx() != previous]
            if len(following) != 1:
                return ()
            previous, current = current, following[0]
            if current == hetero or current in sequence:
                return ()
            sequence.append(current)
        if mol.GetBondBetweenAtoms(sequence[-1], hetero) is None:
            return ()
        sequences.append(tuple(sequence))
    return tuple(sequences)


heteroring_assembly_proposal = heteroring_assembly_decision
