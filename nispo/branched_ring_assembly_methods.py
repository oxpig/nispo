"""Bounded mapped P-28.6 construction for simple carbon ring assemblies."""

from dataclasses import dataclass
from itertools import product

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import ROOTS
from .naming_methods import MethodDecision


_LATIN = {3: "ter", 4: "quater", 5: "quinque", 6: "sexi"}


@dataclass(frozen=True, slots=True)
class BranchedAssemblyProof:
    ring_atoms: tuple[frozenset[int], ...]
    adjacency: tuple[tuple[int, ...], ...]
    connectors: tuple[tuple[int, int, int, int], ...]
    ring_size: int
    kind: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    key: tuple
    path: tuple[int, ...]
    sequences: tuple[tuple[int, ...], ...]
    branches: tuple[tuple[int, int, int], ...]
    name: str


def branched_ring_assembly_decision(mol, Chem):
    """Return a fully mapped bounded P-28.6 construction, or decline."""
    proof = prove_branched_ring_assembly(mol, Chem)
    if proof is None:
        return None
    candidates = _enumerate_candidates(proof, mol)
    if not candidates:
        return None
    minimum = min(candidate.key for candidate in candidates)
    winners = [candidate for candidate in candidates if candidate.key == minimum]
    # The source key must determine one spelling; if a tie survives and would
    # change the cited name, this bounded producer cannot select safely.
    if len({candidate.name for candidate in winners}) != 1:
        return None
    selected = min(winners, key=lambda candidate: (candidate.path, candidate.sequences))
    path_set = set(selected.path)
    component_roles = []
    atom_locants = []
    for component, sequence in enumerate(selected.sequences):
        atoms = proof.ring_atoms[component]
        composite = component in path_set
        assembly_number = selected.path.index(component) + 1 if composite else None
        locants = tuple((atom, f"{assembly_number}^{loc}" if composite else str(loc))
                        for loc, atom in enumerate(sequence, 1))
        role_kind = "path-ring" if composite else "pendant-ring"
        component_roles.append((f"{role_kind}:{assembly_number if composite else component + 1}", atoms, locants))
        if composite:
            atom_locants.extend(locants)
    heavy = frozenset(atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1)
    role_sets = [set(atoms) for _, atoms, _ in component_roles]
    if (any(role_sets[i] & role_sets[j] for i in range(len(role_sets))
            for j in range(i + 1, len(role_sets)))
            or frozenset().union(*(frozenset(group) for group in role_sets)) != heavy):
        return None
    return MethodDecision(
        name=selected.name,
        method="ring-assembly",
        principal_centers=frozenset(),
        atoms=heavy,
        rule="P-14.3.1/P-14.3.5/P-28.1/P-28.3/P-28.6/P-29.3.5/P-52.2.7",
        atom_locants=tuple(atom_locants),
        component_roles=tuple(component_roles),
    )


def prove_branched_ring_assembly(mol, Chem):
    """Prove an identical carbon monocycle unit-tree with a P-28.6 branch."""
    if mol is None or mol.GetNumHeavyAtoms() > 60 or mol.GetStereoGroups():
        return None
    for atom in iter_atoms(mol):
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() or atom.GetIsotope()
                or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag())):
            return None
    if any(int(bond.GetStereo()) for bond in iter_bonds(mol)):
        return None
    if any(atom.GetAtomicNum() == 1 and atom.GetIsotope() for atom in iter_atoms(mol)):
        return None
    heavy = {atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1}
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if not 4 <= len(rings) <= 6 or len(set().union(*rings)) != len(heavy):
        return None
    if set().union(*rings) != heavy or any(left & right for i, left in enumerate(rings)
                                          for right in rings[i + 1:]):
        return None
    sizes, kinds = set(), set()
    for ring in rings:
        sizes.add(len(ring))
        internal = [bond for bond in iter_bonds(mol)
                    if bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring]
        if (len(internal) != len(ring)
                or any(sum(n.GetIdx() in ring for n in mol.GetAtomWithIdx(idx).GetNeighbors()) != 2
                       for idx in ring)):
            return None
        bond_kinds = {bond.GetBondType() for bond in internal}
        if bond_kinds == {Chem.BondType.SINGLE} and 3 <= len(ring) <= 10:
            kinds.add("saturated")
        elif len(ring) == 6 and bond_kinds == {Chem.BondType.AROMATIC}:
            kinds.add("benzene")
        else:
            return None
    if len(sizes) != 1 or len(kinds) != 1:
        return None
    size, kind = next(iter(sizes)), next(iter(kinds))
    if kind == "saturated" and size not in ROOTS:
        return None

    owner = {atom: unit for unit, ring in enumerate(rings) for atom in ring}
    edges = []
    for bond in iter_bonds(mol):
        left_atom, right_atom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left, right = owner.get(left_atom), owner.get(right_atom)
        if left == right:
            continue
        # Inter-ring single bonds from aromatic ring atoms are valid; their
        # aromatic atom flags do not change the bond's single-bond status.
        if left is None or right is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        edges.append((left, right, left_atom, right_atom))
    if len(edges) != len(rings) - 1:
        return None
    adjacency = [[] for _ in rings]
    pairs = set()
    for left, right, _, _ in edges:
        pair = tuple(sorted((left, right)))
        if pair in pairs:
            return None
        pairs.add(pair)
        adjacency[left].append(right)
        adjacency[right].append(left)
    if any(not neighbors or len(neighbors) > 3 for neighbors in adjacency):
        return None
    if not _connected(adjacency):
        return None
    paths = _longest_paths(adjacency)
    if not paths or len(paths[0]) < 3 or len(paths[0]) == len(rings):
        return None
    # A source-selected longest path whose off-path component has multiple
    # rings would need a compound ring-assembly substituent grammar. Decline
    # the graph instead of choosing a potentially inferior simpler path.
    for path in paths:
        path_set = set(path)
        off = set(range(len(rings))) - path_set
        if any(len(adjacency[unit]) != 1 for unit in off):
            return None
        if any(sum(neighbor in path_set for neighbor in adjacency[unit]) != 1 for unit in off):
            return None
    return BranchedAssemblyProof(
        rings, tuple(tuple(sorted(values)) for values in adjacency), tuple(edges), size, kind
    )


def _connected(adjacency):
    seen, stack = set(), [0]
    while stack:
        unit = stack.pop()
        if unit in seen:
            continue
        seen.add(unit)
        stack.extend(adjacency[unit])
    return len(seen) == len(adjacency)


def _longest_paths(adjacency):
    paths = []
    for start in range(len(adjacency)):
        for end in range(start + 1, len(adjacency)):
            path = _unique_tree_path(adjacency, start, end)
            if path:
                paths.append(path)
    if not paths:
        return ()
    maximum = max(map(len, paths))
    longest = {path for path in paths if len(path) == maximum}
    longest.update(tuple(reversed(path)) for path in tuple(longest))
    return tuple(sorted(longest))


def _unique_tree_path(adjacency, start, end):
    parent = {start: None}
    stack = [start]
    while stack:
        node = stack.pop()
        if node == end:
            break
        for neighbor in adjacency[node]:
            if neighbor not in parent:
                parent[neighbor] = node
                stack.append(neighbor)
    if end not in parent:
        return None
    path, current = [], end
    while current is not None:
        path.append(current)
        current = parent[current]
    return tuple(reversed(path))


def _ring_sequences(mol, ring, attached):
    if not attached:
        return ()
    choices = set()
    for start in attached:
        internal = sorted(neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(start).GetNeighbors()
                          if neighbor.GetIdx() in ring)
        if len(internal) != 2:
            return ()
        for next_atom in internal:
            sequence, previous, current = [start, next_atom], start, next_atom
            while len(sequence) < len(ring):
                next_values = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(current).GetNeighbors()
                               if neighbor.GetIdx() in ring and neighbor.GetIdx() != previous]
                if len(next_values) != 1:
                    return ()
                previous, current = current, next_values[0]
                if current in sequence:
                    return ()
                sequence.append(current)
            if mol.GetBondBetweenAtoms(sequence[-1], start) is None:
                return ()
            choices.add(tuple(sequence))
    return tuple(sorted(choices))


def _format_composite(unit, locant):
    return f"{unit}^{{{locant}}}"


def _enumerate_candidates(proof, mol):
    mol_ring_count = len(proof.ring_atoms)
    candidates = []
    for path in proof_paths(proof.adjacency):
        path_set = set(path)
        branches = []
        for unit in set(range(mol_ring_count)) - path_set:
            neighbor = next(item for item in proof.adjacency[unit] if item in path_set)
            edge = next(e for e in proof.connectors if frozenset(e[:2]) == frozenset((unit, neighbor)))
            path_atom = edge[3] if edge[0] == unit else edge[2]
            branches.append((neighbor, path_atom, unit))
        attachments = [set() for _ in proof.ring_atoms]
        for left, right, left_atom, right_atom in proof.connectors:
            attachments[left].add(left_atom)
            attachments[right].add(right_atom)
        # Ring numbering choices depend only on all direct attachment sites.
        sequences_by_unit = []
        for unit, ring in enumerate(proof.ring_atoms):
            sequences_by_unit.append(_ring_sequences(mol, ring, attachments[unit]))
        if any(not choices for choices in sequences_by_unit):
            return ()
        for sequences in product(*sequences_by_unit):
            loc_maps = [{atom: loc for loc, atom in enumerate(seq, 1)}
                        for seq in sequences]
            junctions = []
            for left, right in zip(path, path[1:]):
                raw = next(e for e in proof.connectors if frozenset(e[:2]) == frozenset((left, right)))
                left_atom = raw[2] if raw[0] == left else raw[3]
                right_atom = raw[3] if raw[0] == left else raw[2]
                junctions.append((left, loc_maps[left][left_atom], right, loc_maps[right][right_atom]))
            junction_key = tuple(value for left, lloc, right, rloc in junctions
                                 for value in (path.index(left) + 1, lloc,
                                               path.index(right) + 1, rloc))
            branch_specs = tuple(sorted((path.index(unit) + 1, loc_maps[unit][atom], pendant)
                                        for unit, atom, pendant in branches))
            substituent_key = tuple(value for unit, loc, _ in branch_specs for value in (unit, loc))
            name = _candidate_name(proof, path, junctions, branch_specs)
            key = (junction_key, -len(branch_specs), substituent_key,
                   tuple(_substituent_name(proof) for _ in branch_specs))
            candidates.append(_Candidate(key, path, tuple(sequences), branch_specs, name))
    return tuple(candidates)


def proof_paths(adjacency):
    return _longest_paths(adjacency)


def _substituent_name(proof):
    return "phenyl" if proof.kind == "benzene" else "cyclo" + ROOTS[proof.ring_size] + "yl"


def _candidate_name(proof, path, junctions, branches):
    unit = "phenyl" if proof.kind == "benzene" else "cyclo" + ROOTS[proof.ring_size] + "ane"
    if proof.kind == "benzene":
        parent = _LATIN[len(path)] + "phenyl"
        substituent = "phenyl"
    else:
        parent = _LATIN[len(path)] + unit
        substituent = "cyclo" + ROOTS[proof.ring_size] + "yl"
    branch_prefix = ""
    if branches:
        locants = ",".join(_format_composite(ring_no, loc) for ring_no, loc, _ in branches)
        multiplier = {1: "", 2: "di", 3: "tri"}.get(len(branches))
        if multiplier is None:
            return ""
        branch_prefix = f"{locants}-{multiplier}{substituent}-"
    junction_text = ":".join(
        f"{_format_composite(path.index(left) + 1, lloc)},{_format_composite(path.index(right) + 1, rloc)}"
        for left, lloc, right, rloc in junctions
    )
    return f"{branch_prefix}{junction_text}-{parent}"
