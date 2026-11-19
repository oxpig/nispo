"""Mapped constructions for bounded ring assemblies and equivalent fused units."""

from dataclasses import dataclass
from itertools import product

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import ROOTS
from .naming_methods import MethodDecision


_ASSEMBLY_MULTIPLIERS = {3: "ter", 4: "quater", 5: "quinque", 6: "sexi"}
_PHENYL_MULTIPLIERS = {3: "ter", 4: "quater", 5: "quinque", 6: "sexi"}


def rooted_biphenyl_from_prefix(mol, Chem, root, blocked, prefix):
    """Reuse the mapped assembly when the simple-ring branch uses two phenyls."""
    from .branch_rules import collect_subtree
    from .mapped_ring_parents import ring_systems

    scope = frozenset(collect_subtree(mol, root, blocked))
    return rooted_biphenyl(mol, Chem, root, blocked, scope, ring_systems(mol)) or prefix


def rooted_biphenyl(mol, Chem, root, blocked, scope, systems):
    """Number a singly rooted biphenyl and its independently mapped branches."""
    from .parent_rules import _ring_sequences_from_connector, _single_ring_connector

    cut = mol.GetBondBetweenAtoms(root, blocked)
    if (blocked in scope or cut is None or cut.GetBondTypeAsDouble() != 1 or cut.GetStereo()
            or {(i, n.GetIdx()) for i in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in scope} != {(root, blocked)}):
        return ''
    proximal = next((ring for ring in systems if root in ring), frozenset())
    phenyls = tuple(ring for ring in systems if len(ring) == 6 and ring <= scope
                    and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                            and mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
    if len(phenyls) != 2 or proximal not in phenyls:
        return ''
    distal = next(ring for ring in phenyls if ring != proximal)
    join = _single_ring_connector(mol, proximal, distal, Chem)
    if join is None or join[0] == root:
        return ''
    from .component_mapping import _record_branch_ring_atoms, mapped_general_branch, mapped_branch_ring_atoms
    from .formatting import format_prefixes, prefix_alphabetization_key
    from .parent_selection import locant_key

    left, right = join
    core = proximal | distal
    owned = set(core)
    ring_atoms = set(core)
    branches = []
    if any(a.GetIsotope() or a.GetFormalCharge() or a.GetChiralTag() or a.GetNumRadicalElectrons()
           for a in (mol.GetAtomWithIdx(i) for i in core)):
        return ''
    for host in core:
        for atom in mol.GetAtomWithIdx(host).GetNeighbors():
            i = atom.GetIdx()
            if i in core or i == blocked:
                continue
            name, atoms = mapped_general_branch(mol, Chem, i, host)
            if not name or not atoms or owned & atoms:
                return ''
            owned.update(atoms)
            ring_atoms.update(mapped_branch_ring_atoms(mol, Chem, i, host, name, atoms))
            branches.append((host, name))
    if owned != scope:
        return ''
    names = []
    for first in _ring_sequences_from_connector(mol, tuple(proximal), left):
        for second in _ring_sequences_from_connector(mol, tuple(distal), right):
            locants = {i: str(n) for n, i in enumerate(first, 1)}
            locants.update({i: f"{n}'" for n, i in enumerate(second, 1)})
            prefixes = [(locants[i], name) for i, name in branches]
            alphabetic = sorted(prefixes, key=lambda row: (prefix_alphabetization_key(row[1]), locant_key(row[0])))
            key = (locant_key(locants[root]), tuple(sorted(locant_key(loc) for loc, _ in prefixes)),
                   tuple(locant_key(loc) for loc, _ in alphabetic))
            head = format_prefixes(prefixes, hierarchical_enclosures=True)
            name = (head + '-' if head else '') + f"[1,1'-biphenyl]-{locants[root]}-yl"
            names.append((key, name))
    if not names:
        return ''
    name = min(names)[1]
    _record_branch_ring_atoms(mol, Chem, root, blocked, name, scope, ring_atoms)
    return name


def identical_cyclic_imide_arms(mol, Chem):
    """Compose equivalent imide-bearing branches on one saturated carbon chain."""
    from .branch_rules import collect_subtree, parent_chain_name
    from .component_mapping import _complete_rooted_ring
    from .extended_inventory import selection_inventory
    from .formatting import complete_parent_name, format_prefixes, join_prefix_parent
    from .mapped_ring_parents import capped_subtree_fragment, ring_systems
    from .multifunctional_parent import _paths
    from .naming_result import SelectionLimitation

    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups() or any(
            a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetFormalCharge()
            or a.GetNumRadicalElectrons() or a.GetChiralTag() or a.HasQuery()
            for a in iter_atoms(mol)) or any(
            b.GetStereo() or b.GetBondDir() or b.HasQuery() for b in iter_bonds(mol))):
        return None
    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.principal) != 4
            or any(g.kind != 'lactam' for g in inventory.principal)):
        return None
    centers = frozenset(g.center for g in inventory.principal)
    imides = []
    for ring in ring_systems(mol):
        sites = centers & ring
        if not sites:
            continue
        if len(sites) != 2:
            return None
        nitrogens = [i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
                     and all(mol.GetBondBetweenAtoms(i, j) is not None for j in sites)]
        if len(nitrogens) != 1:
            return None
        imides.append(sites | set(nitrogens))
    if len(imides) != 2:
        return None
    all_atoms = frozenset(range(mol.GetNumAtoms()))
    options = []
    for atom in iter_atoms(mol):
        if atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetDegree() < 2:
            continue
        host = atom.GetIdx()
        roots = [a.GetIdx() for a in atom.GetNeighbors() if a.IsInRing() and a.GetAtomicNum() == 6]
        if len(roots) != 2 or any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds()):
            continue
        scopes = tuple(frozenset(collect_subtree(mol, i, host)) for i in roots)
        if (host in scopes[0] | scopes[1] or scopes[0] & scopes[1]
                or not all(any(unit <= scope for unit in imides) for scope in scopes)):
            continue
        carbon = all_atoms - scopes[0] - scopes[1]
        if not all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                   and not mol.GetAtomWithIdx(i).IsInRing() for i in carbon):
            continue
        paths = [path for path in _paths(mol, carbon) if set(path) == carbon]
        if len(paths) != 1:
            continue
        path = paths[0]
        if any(mol.GetBondBetweenAtoms(a, b).GetBondType() != Chem.BondType.SINGLE
               for a, b in zip(path, path[1:])):
            continue
        sides = []
        for root, scope in zip(roots, scopes, strict=True):
            capped = capped_subtree_fragment(mol, Chem, scope)
            text = _complete_rooted_ring(mol, Chem, root, scope)
            if not text or capped is None:
                break
            fragment, mapping = capped
            fragment.GetAtomWithIdx(mapping[root]).SetAtomMapNum(1)
            sides.append((text, Chem.MolToSmiles(fragment, isomericSmiles=True)))
        if len(sides) != 2 or sides[0] != sides[1]:
            continue
        if path.index(host) > (len(path) - 1) / 2:
            path = path[::-1]
        locant = str(path.index(host) + 1)
        parent = complete_parent_name(parent_chain_name(mol, path))
        prefixes = format_prefixes([(locant, sides[0][0])] * 2, hierarchical_enclosures=True)
        if parent.endswith('ane') and prefixes:
            name = join_prefix_parent(prefixes, parent)
            options.append((name, path, scopes))
    if len(options) != 1:
        return None
    name, path, scopes = options[0]
    mapping = tuple((i, str(locant)) for locant, i in enumerate(path, 1))
    return MethodDecision(name, 'substitutive', frozenset(), all_atoms, 'P-15.1/P-16.3/P-29',
                          mapping, component_roles=(
                              ('carbon-chain', frozenset(path), mapping),
                              *(('equivalent-imide-arm', scope, ()) for scope in scopes)),
                          preference_issues=(SelectionLimitation(
                              'cyclic-imide-assembly', 'P-44/P-51.3',
                              'Equivalent rooted imide arms are mapped. Multiplicative '
                              'and preferred-parent comparison remain unresolved.'),))


def identical_fused_ring_units(mol, Chem):
    """Compose two identical N-substituted mancude units with proved joins."""
    from .extended_inventory import selection_inventory
    from .mapped_ring_parents import assembly_unit_skeletons, capped_subtree_fragment, ring_systems
    from .naming_result import SelectionLimitation

    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups() or any(
            a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetChiralTag()
            or a.HasQuery() for a in iter_atoms(mol)) or any(
            b.GetStereo() or b.GetBondDir() or b.HasQuery() for b in iter_bonds(mol))):
        return None
    inventory = selection_inventory(mol)
    if not inventory.supported or inventory.groups:
        return None
    systems = ring_systems(mol)
    units = [core for core in systems if len(core) == 13 and
             sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in core) == [6] * 12 + [7]
             and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in core)]
    if len(units) != 2:
        return None
    unit_atoms = units[0] | units[1]
    from .component_mapping import mapped_general_branch
    from .formatting import enclose_name_unit
    from .parent_selection import locant_key

    def describe(core, connection):
        n_root, child, owned, prefix = None, None, None, ''
        for i in core:
            for atom in mol.GetAtomWithIdx(i).GetNeighbors():
                j = atom.GetIdx()
                if j in core or (i, j) == connection:
                    continue
                if n_root is not None or mol.GetAtomWithIdx(i).GetAtomicNum() != 7:
                    return None
                prefix, owned = mapped_general_branch(mol, Chem, j, i)
                if not prefix or not owned or owned & unit_atoms:
                    return None
                n_root, child = i, j
        if n_root is None:
            return None
        # This composition needs a numbered N-H hydride as its unmodified
        # component. The common carbazole numbering already supplies N9.
        locants = set()
        for row in assembly_unit_skeletons(mol, Chem, core):
            if row[0] != 'carbazole':
                continue
            mapping = dict(zip(row[1], row[2], strict=True))
            attach = mapping[connection[0]]
            if mapping[n_root] == '9' and attach.isdigit() and attach != '9':
                locants.add(attach)
        capped = capped_subtree_fragment(mol, Chem, owned)
        if not locants or capped is None:
            return None
        fragment, mapping = capped
        fragment.GetAtomWithIdx(mapping[child]).SetAtomMapNum(1)
        identity = Chem.MolToSmiles(fragment, isomericSmiles=True)
        return min(locants, key=locant_key), prefix, identity, owned

    contacts = [tuple((i, n.GetIdx()) for i in core
                      for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() not in core)
                for core in units]
    connections = [(a, b) for a in contacts[0] for b in contacts[1] if
                   (a == b[::-1] or a[1] != b[1] and any(a[1] in system and b[1] in system for system in systems
                                      if not system & unit_atoms))]
    if len(connections) != 1:
        return None
    left, right = connections[0]
    if any(mol.GetBondBetweenAtoms(*edge).GetBondTypeAsDouble() != 1 for edge in (left, right)):
        return None
    first, second = (describe(core, edge) for core, edge in zip(units, (left, right), strict=True))
    if first is None or second is None or first[:3] != second[:3] or first[3] & second[3]:
        return None
    attach, prefix, _, _ = first
    scope = frozenset(range(mol.GetNumAtoms()))
    rest = scope - unit_atoms - first[3] - second[3]
    pair = attach + ',' + attach + "'"
    if left == right[::-1]:
        if rest:
            return None
        ligands = 'di' + prefix if prefix == 'phenyl' else 'bis' + enclose_name_unit(prefix)
        name = f"9,9'-{ligands}-9H,9'H-{pair}-bicarbazole"
        method, rule = 'ring-assembly', 'P-28.2.1/P-28.2.3/P-16.3'
    else:
        connector = _fused_unit_linker(mol, Chem, rest, (left, right), systems)
        if not connector:
            return None
        ligand = prefix if prefix == 'phenyl' else enclose_name_unit(prefix)
        name = pair + '-' + enclose_name_unit(connector) + 'bis' + enclose_name_unit(
            f'9-{ligand}-9H-carbazole')
        method, rule = 'multiplicative', 'P-15.3/P-51.3'
    return MethodDecision(name, method, frozenset(), scope, rule,
        preference_issues=(SelectionLimitation('ring-parent-method', rule,
            'Identical rooted ring units and their original junctions are accounted for.'),))


def _fused_unit_linker(mol, Chem, scope, edges, systems):
    """Number a substituted aromatic monocycle with two equivalent ports."""
    from .component_mapping import mapped_general_branch
    from .formatting import attach_suffix, format_prefixes, join_prefix_parent
    from .mapped_ring_parents import skeletons

    ports = {edge[1] for edge in edges}
    core = next((system for system in systems if ports <= system <= scope), None)
    if (core is None or len(core) != 6 or len(ports) != 2
            or any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in core)):
        return ''
    covered = set(core)
    branches = []
    boundary = {(host, ring) for ring, host in edges}
    for host in core:
        for a in mol.GetAtomWithIdx(host).GetNeighbors():
            root = a.GetIdx()
            if root in core or (host, root) in boundary:
                continue
            word, owned = mapped_general_branch(mol, Chem, root, host)
            if not word or not owned or not owned <= scope or covered & owned:
                return ''
            covered.update(owned)
            branches.append((host, word))
    if covered != scope:
        return ''
    options = {}
    for row in skeletons(mol, Chem, core):
        parent = row[0]
        mapping = dict(zip(row[1], row[2], strict=True))
        if not all(mapping[i].isdigit() for i in core):
            continue
        ordered = tuple(int(mapping[edge[1]]) for edge in edges)
        locants = tuple(sorted(ordered))
        cited = tuple(sorted((int(mapping[i]), word) for i, word in branches))
        options.setdefault((parent, locants, cited), set()).add(ordered)
    if len({parent for parent, _, _ in options}) != 1:
        return ''
    parent, locants, cited = min(options, key=lambda row: row[1:])
    if len(options[parent, locants, cited]) != 2:
        return ''
    return join_prefix_parent(format_prefixes([(str(i), word) for i, word in cited]),
                              attach_suffix(parent, 'diyl', locants))


@dataclass(frozen=True, slots=True)
class RingAssemblyProof:
    """Graph evidence used by the P-28 naming decision."""

    component_atoms: tuple[frozenset[int], ...]
    path: tuple[int, ...]
    connectors: tuple[tuple[int, int], ...]
    ring_size: int
    unit_kind: str


def ring_assembly_decision(mol, Chem):
    """Name a narrowly proved three-to-six-unit P-28 assembly, else decline.

    All locant alternatives for each admitted monocycle are enumerated. The
    name is selected from the composite attachment-locant set, not atom order,
    name length, or a legacy retained-name candidate.
    """
    proof = _prove_path_assembly(mol, Chem)
    if proof is None:
        return None
    sequences_by_component = tuple(
        _local_ring_numberings(mol, atoms, proof.connectors, proof.component_atoms)
        for atoms in proof.component_atoms
    )
    if any(not candidates for candidates in sequences_by_component):
        return None

    candidates = []
    for path in (proof.path, tuple(reversed(proof.path))):
        ordered_sets = [proof.component_atoms[i] for i in path]
        local_choices = [sequences_by_component[i] for i in path]
        for sequences in product(*local_choices):
            atom_to_loc = [
                {atom: loc for loc, atom in enumerate(seq, 1)}
                for seq in sequences
            ]
            connectors = []
            complete_locants = []
            for edge_index in range(len(path) - 1):
                left_id, right_id = path[edge_index], path[edge_index + 1]
                left_atom, right_atom = _connector_for_pair(
                    mol, proof.component_atoms[left_id], proof.component_atoms[right_id]
                )
                if left_atom is None:
                    return None
                left_loc = atom_to_loc[edge_index][left_atom]
                right_loc = atom_to_loc[edge_index + 1][right_atom]
                connectors.append((left_loc, right_loc))
                complete_locants.extend(((edge_index + 1, left_loc), (edge_index + 2, right_loc)))
            # P-14.3.5: compare locant sets in increasing order, term by term.
            locant_set = tuple(sorted(complete_locants))
            citation = tuple(connectors)
            candidates.append((locant_set, citation, path, tuple(sequences), tuple(connectors)))

    if not candidates:
        return None
    locant_set, citation, path, sequences, connectors = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    if proof.unit_kind == "benzene":
        parent = _PHENYL_MULTIPLIERS[len(path)] + "phenyl"
    else:
        root = ROOTS.get(proof.ring_size)
        if not root:
            return None
        parent = _ASSEMBLY_MULTIPLIERS[len(path)] + "cyclo" + root + "ane"

    # Each junction cites the two component/local locants in assembly order;
    # successive junctions are separated by colons (P-28.3.1/.2).
    locant_text = ":".join(
        f"{i + 1}^{{{left}}},{i + 2}^{{{right}}}"
        for i, (left, right) in enumerate(connectors)
    )
    name = f"{locant_text}-{parent}"
    component_roles = []
    atom_locants = []
    for component_number, (component_id, seq) in enumerate(zip(path, sequences), 1):
        role_atoms = proof.component_atoms[component_id]
        component_locants = tuple((atom, str(loc)) for loc, atom in enumerate(seq, 1))
        component_roles.append((f"ring-{component_number}", role_atoms, component_locants))
        atom_locants.extend((atom, f"{component_number}^{loc}") for loc, atom in enumerate(seq, 1))
    connectors_original = tuple(
        _connector_for_pair(mol, proof.component_atoms[path[i]], proof.component_atoms[path[i + 1]])
        for i in range(len(path) - 1)
    )
    atoms = frozenset().union(*proof.component_atoms)
    return MethodDecision(
        name=name,
        method="ring-assembly",
        principal_centers=frozenset(),
        atoms=atoms,
        rule=("P-28.1/P-28.3.1/P-52.2.7.2" if proof.unit_kind != "benzene"
              else "P-28.1/P-28.3.2/P-52.2.7.2"),
        atom_locants=tuple(atom_locants),
        component_roles=tuple(component_roles),
    )


def prove_ring_assembly(mol, Chem):
    """Expose bounded topology evidence for callers that need proof metadata."""
    return _prove_path_assembly(mol, Chem)


def _prove_path_assembly(mol, Chem):
    if mol.GetNumHeavyAtoms() > 60 or mol.GetStereoGroups():
        return None
    for atom in iter_atoms(mol):
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() or atom.GetIsotope()
                or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag())):
            return None
    if any(int(bond.GetStereo()) for bond in iter_bonds(mol)):
        return None
    if any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in iter_atoms(mol)):
        return None
    heavy = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
    raw_rings = tuple(tuple(r) for r in mol.GetRingInfo().AtomRings())
    if not 3 <= len(raw_rings) <= 6:
        return None
    components = tuple(frozenset(ring) for ring in raw_rings)
    if len(set().union(*components)) != len(heavy) or set().union(*components) != heavy:
        return None
    if any(a & b for i, a in enumerate(components) for b in components[i + 1:]):
        return None

    signatures = []
    ring_sizes = set()
    for component in components:
        ring_sizes.add(len(component))
        internal = [b for b in iter_bonds(mol)
                    if b.GetBeginAtomIdx() in component and b.GetEndAtomIdx() in component]
        if len(internal) != len(component):
            return None
        if any(sum(n.GetIdx() in component for n in mol.GetAtomWithIdx(i).GetNeighbors()) != 2
               for i in component):
            return None
        kinds = {b.GetBondType() for b in internal}
        if kinds == {Chem.BondType.SINGLE}:
            signatures.append("saturated")
        elif len(component) == 6 and kinds == {Chem.BondType.AROMATIC}:
            signatures.append("benzene")
        else:
            return None
    if len(ring_sizes) != 1 or len(set(signatures)) != 1:
        return None

    index_by_atom = {atom: i for i, comp in enumerate(components) for atom in comp}
    quotient_edges = []
    external_bonds = []
    for bond in iter_bonds(mol):
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left, right = index_by_atom.get(begin), index_by_atom.get(end)
        if left == right:
            continue
        if left is None or right is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        pair = tuple(sorted((left, right)))
        quotient_edges.append(pair)
        external_bonds.append((begin, end))
    if len(quotient_edges) != len(components) - 1 or len(set(quotient_edges)) != len(quotient_edges):
        return None
    adjacency = {i: [] for i in range(len(components))}
    for left, right in quotient_edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    if any(len(neighbors) not in {1, 2} for neighbors in adjacency.values()):
        return None
    endpoints = [i for i, neighbors in adjacency.items() if len(neighbors) == 1]
    if len(endpoints) != 2:
        return None
    path = _walk_path(adjacency, endpoints[0])
    if path is None or len(path) != len(components):
        return None
    return RingAssemblyProof(components, path, tuple(external_bonds), next(iter(ring_sizes)), signatures[0])


def _walk_path(adjacency, start):
    path, previous, current = [], None, start
    while True:
        if current in path:
            return None
        path.append(current)
        nxt = [item for item in adjacency[current] if item != previous]
        if not nxt:
            return tuple(path)
        if len(nxt) != 1:
            return None
        previous, current = current, nxt[0]


def _connector_for_pair(mol, left_atoms, right_atoms):
    for atom in left_atoms:
        for neighbor in mol.GetAtomWithIdx(atom).GetNeighbors():
            other = neighbor.GetIdx()
            if other in right_atoms:
                return atom, other
    return None, None


def _local_ring_numberings(mol, component, connector_bonds, components):
    # Number each unit from one attachment vertex and enumerate both usual
    # cyclic directions. Interior units are started at either adjacent junction.
    attachments = set()
    for begin, end in connector_bonds:
        if begin in component:
            attachments.add(begin)
        if end in component:
            attachments.add(end)
    if not attachments:
        return ()
    sequences = set()
    for start in attachments:
        neighbors = sorted(
            n.GetIdx() for n in mol.GetAtomWithIdx(start).GetNeighbors()
            if n.GetIdx() in component
        )
        if len(neighbors) != 2:
            return ()
        for nxt in neighbors:
            sequence = [start, nxt]
            previous, current = start, nxt
            while len(sequence) < len(component):
                next_atoms = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
                              if n.GetIdx() in component and n.GetIdx() != previous]
                if len(next_atoms) != 1:
                    return ()
                previous, current = current, next_atoms[0]
                if current == start or current in sequence:
                    return ()
                sequence.append(current)
            if mol.GetBondBetweenAtoms(sequence[-1], start) is None:
                return ()
            sequences.add(tuple(sequence))
    return tuple(sequences)
