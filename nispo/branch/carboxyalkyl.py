"""One-cut, source-owned P-65 carboxyalkyl substituent construction."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CarboxyalkylProof:
    name: str
    root: int
    host: int
    atoms: frozenset[int]
    carbon_path: tuple[int, ...]
    acid_atoms: frozenset[int]
    amino_atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    source_acid_identities: tuple
    original_graph_signature: tuple


@dataclass(frozen=True, slots=True)
class CarboxyalkoxyProof:
    """A source-owned substituted retained alkoxy unit at its parent cut."""
    text: str
    prefix_nature: str
    root: int
    host: int
    atoms: frozenset[int]
    child: CarboxyalkylProof


def onium_carboxyalkyl_prefix(mol, Chem, root, host):
    """Cite a short ordinary carboxyalkyl arm at a senior onium site."""
    parent = mol.GetAtomWithIdx(host)
    cut = mol.GetBondBetweenAtoms(root, host)
    if (parent.GetAtomicNum() != 7 or parent.GetFormalCharge() != 1
            or cut is None or cut.GetBondTypeAsDouble() != 1):
        return None
    from ..extended_inventory import selection_inventory
    from ..functionalized_carbon_prefixes import _scope
    from ..group_priority import compare_functional_group_priority

    inventory = selection_inventory(mol)
    if not inventory.supported or len(inventory.principal) != 1:
        return None
    senior = inventory.principal[0]
    if senior.kind != 'ammonium' or senior.center != host:
        return None
    atoms = _scope(mol, root, host)
    if atoms is None or not 4 <= len(atoms) <= 8:
        return None
    acids = [g for g in inventory.groups if g.kind == 'acid' and set(g.atoms) <= atoms]
    if len(acids) != 1 or acids[0].owner not in atoms:
        return None
    acid = acids[0]
    facts = dict(zip(inventory.groups, inventory.source_group_facts, strict=True))
    if compare_functional_group_priority(facts[senior], facts[acid]).status != 'senior':
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetFormalCharge()
           or mol.GetAtomWithIdx(i).IsInRing() or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
           or any(n.GetAtomicNum() == 1 for n in mol.GetAtomWithIdx(i).GetNeighbors()) for i in atoms):
        return None
    chain = atoms - set(acid.atoms)
    if root not in chain or not 1 <= len(chain) <= 5:
        return None
    adjacency = {i: {a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors()
                     if a.GetIdx() in chain} for i in chain}
    if (any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 or len(neighbors) > 2
            or any(mol.GetBondBetweenAtoms(i, n).GetBondTypeAsDouble() != 1 for n in neighbors)
            for i, neighbors in adjacency.items())
            or sum(map(len, adjacency.values())) != 2 * (len(chain) - 1)):
        return None
    from ..component_mapping import _internally_rooted_carbon_prefix

    text, owned = _internally_rooted_carbon_prefix(mol, Chem, root, host, atoms, chain)
    return (text, owned) if text and owned == atoms else None


def carboxyalkyl_prefix_proof(mol, Chem, root, host, atoms):
    """Prove the entire linear carbon branch and its carboxy/amino roles.

    The acid must be a source-classified principal acid on this branch, with
    a second principal acid outside it. Thus a selected outside suffix can
    cite this acid as a carboxy prefix. This is a branch construction only;
    the enclosing parent's method and preference are still compared upstream.
    """
    from ..functionalized_carbon_prefixes import _graph_signature, _scope
    from ..extended_inventory import selection_inventory

    if (mol is None or mol.GetStereoGroups() or root == host
            or not 0 <= root < mol.GetNumAtoms() or not 0 <= host < mol.GetNumAtoms()
            or mol.GetAtomWithIdx(root).GetAtomicNum() != 6
            or mol.GetAtomWithIdx(host).GetAtomicNum() not in {8, 16}
            or mol.GetBondBetweenAtoms(root, host) is None
            or mol.GetBondBetweenAtoms(root, host).GetBondTypeAsDouble() != 1
            or any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or a.HasQuery() or a.GetAtomMapNum() or a.HasProp('_UnknownStereo')
                   for a in mol.GetAtoms())):
        return None
    branch = _scope(mol, root, host)
    if (branch is None or branch != frozenset(atoms) or len(branch) > 24
            or any(mol.GetAtomWithIdx(i).IsInRing() for i in branch)):
        return None
    if any(int(b.GetStereo()) or b.HasQuery() or b.GetIsAromatic()
           for b in mol.GetBonds() if b.GetBeginAtomIdx() in branch
           or b.GetEndAtomIdx() in branch):
        return None
    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    principal = tuple(inventory.principal)
    if (not inventory.supported or frontier.status != 'tie'
            or frontier.unknown_pairs or frontier.cycle
            or set(frontier.groups) != set(principal) or len(principal) < 2
            or any(group.kind != 'acid' for group in principal)):
        return None
    local_acids = [group for group in principal if set(group.atoms) <= branch]
    if len(local_acids) != 1 or not any(set(group.atoms).isdisjoint(branch)
                                        for group in principal):
        return None
    acid = local_acids[0]
    acid_atoms = frozenset(acid.atoms)
    carbonyl = mol.GetAtomWithIdx(acid.center)
    oxygen = [mol.GetAtomWithIdx(i) for i in acid_atoms if i != acid.center]
    if (len(acid_atoms) != 3 or carbonyl.GetAtomicNum() != 6 or len(oxygen) != 2
            or any(a.GetAtomicNum() != 8 or a.GetDegree() != 1 for a in oxygen)
            or acid.owner not in branch or acid.owner == acid.center
            or set(n.GetIdx() for n in carbonyl.GetNeighbors() if n.GetAtomicNum() > 1)
                != {acid.owner, *(a.GetIdx() for a in oxygen)}
            or sorted((mol.GetBondBetweenAtoms(acid.center, a.GetIdx()).GetBondTypeAsDouble(),
                       a.GetTotalNumHs(includeNeighbors=True)) for a in oxygen)
                != [(1.0, 1), (2.0, 0)]):
        return None
    carbon = {i for i in branch if mol.GetAtomWithIdx(i).GetAtomicNum() == 6} - {acid.center}
    nitrogen = {i for i in branch if mol.GetAtomWithIdx(i).GetAtomicNum() == 7}
    if (root not in carbon or acid.owner not in carbon or len(nitrogen) > 1
            or branch != frozenset(carbon | set(acid_atoms) | nitrogen)):
        return None
    amino_atoms = frozenset(nitrogen)
    amino_host = None
    if nitrogen:
        idx = next(iter(nitrogen))
        n = mol.GetAtomWithIdx(idx)
        neighbors = [a.GetIdx() for a in n.GetNeighbors() if a.GetAtomicNum() > 1]
        if (len(neighbors) != 1 or neighbors[0] not in carbon
                or n.GetTotalNumHs(includeNeighbors=True) != 2
                or mol.GetBondBetweenAtoms(idx, neighbors[0]).GetBondTypeAsDouble() != 1
                or not any(group.kind == 'amine' and group.center == idx for group in inventory.groups)):
            return None
        amino_host = neighbors[0]
    for group in inventory.groups:
        if not set(group.atoms) & branch:
            continue
        if group != acid and not (group.kind == 'amine' and group.center in nitrogen):
            return None
    # P-46: after removing the typed carboxy and amino roles, the rooted
    # carbon skeleton must be an unbranched chain. Its maximal path is unique.
    adjacency = {i: {n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                     if n.GetIdx() in carbon} for i in carbon}
    if (not carbon or len(carbon) > 8 or len(adjacency[root]) > 1
            or any(len(neighbors) > 2 for neighbors in adjacency.values())):
        return None
    # A one-carbon acid arm has one substitutable position. Oxygen may use
    # fully substitutable retained methoxy (P-63.2.2.2); a separate graph
    # proof below carries its compound nature to the parent citation site.
    if (len(carbon) == 1 and (nitrogen
                              or mol.GetAtomWithIdx(root).GetTotalNumHs(includeNeighbors=True) != 2)):
        return None
    path, previous, current = [], None, root
    while current is not None:
        path.append(current)
        neighbors = adjacency[current] - ({previous} if previous is not None else set())
        if len(neighbors) > 1 or any(idx in path for idx in neighbors):
            return None
        previous, current = current, next(iter(neighbors), None)
    if set(path) != carbon:
        return None
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in carbon and right in carbon and bond.GetBondTypeAsDouble() != 1:
            return None
    tagged = {i for i in branch if int(mol.GetAtomWithIdx(i).GetChiralTag())}
    if not tagged <= carbon:
        return None
    if tagged:
        from ..stereochemistry import assigned_atom_stereochemistry
        try:
            descriptors = dict(assigned_atom_stereochemistry(mol, Chem, include_pseudo=True))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        if any(descriptors.get(i) not in {'R', 'S'} for i in tagged):
            return None
    from ..component_mapping import MappedComponent, decorate_component, hydrogen_bindings
    from ..constants import ROOTS

    if len(path) not in ROOTS:
        return None
    mapping = hydrogen_bindings(mol, {idx: str(i) for i, idx in enumerate(path, 1)})
    rows = [(mapping[acid.owner] if len(path) > 1 else '', 'carboxy')]
    if amino_host is not None:
        rows.append((mapping[amino_host], 'amino'))
    component = MappedComponent(ROOTS[len(path)]+'yl', tuple(mapping.items()),
                                frozenset(mapping), tuple(rows), branch-frozenset(mapping),
                                systematic_parent=True,
                                prefix_mode='mononuclear' if len(path) == 1 else 'locanted')
    name = decorate_component(mol, Chem, component)
    if not name:
        return None
    identities = tuple(sorted((group.kind, group.center, tuple(group.atoms), group.owner,
                               group.ligand) for group in principal))
    return CarboxyalkylProof(name, root, host, branch, tuple(path), acid_atoms,
                             amino_atoms, tuple(mapping.items()), identities,
                             _graph_signature(mol))


def carboxyalkoxy_prefix_proof(mol, Chem, root, host, atoms, parent_atoms):
    """Bind substituted alkoxy nature to original branch and acid suffix.

    The retained methoxy/ethoxy/propoxy/butoxy unit is simple before
    substitution (P-63.2.2.2). The original acid group is a carboxy
    substitution on that unit (P-65.1.2.2.3), making the complete prefix
    compound (P-29.1.2/P-16.5.1.1). At least one separate principal acid must
    be completely owned by the parent component, not just somewhere outside
    the branch. This proof does not select the overall parent or method.
    """
    from ..functionalized_carbon_prefixes import _scope
    from ..extended_inventory import selection_inventory
    from ..formatting import alkyl_to_alkoxy

    if (mol is None or not isinstance(root, int) or not isinstance(host, int)
            or not 0 <= root < mol.GetNumAtoms() or not 0 <= host < mol.GetNumAtoms()):
        return None
    linker = mol.GetAtomWithIdx(root)
    bond = mol.GetBondBetweenAtoms(root, host)
    if (linker.GetAtomicNum() != 8 or linker.IsInRing() or linker.GetFormalCharge()
            or linker.GetIsotope() or linker.GetNumRadicalElectrons()
            or linker.GetTotalNumHs(includeNeighbors=True) or bond is None
            or bond.GetBondTypeAsDouble() != 1 or parent_atoms is None):
        return None
    atoms, parent_atoms = frozenset(atoms), frozenset(parent_atoms)
    if (root not in atoms or host not in parent_atoms or atoms & parent_atoms
            or _scope(mol, root, host) != atoms):
        return None
    children = [n.GetIdx() for n in linker.GetNeighbors()
                if n.GetIdx() != host and n.GetAtomicNum() > 1]
    if (len(children) != 1 or mol.GetAtomWithIdx(children[0]).GetAtomicNum() != 6
            or mol.GetBondBetweenAtoms(root, children[0]).GetBondTypeAsDouble() != 1):
        return None
    child = carboxyalkyl_prefix_proof(mol, Chem, children[0], root, atoms - {root})
    if child is None or len(child.carbon_path) > 4:
        return None
    inventory = selection_inventory(mol)
    if (not inventory.supported or not any(
            group.kind == 'acid' and set(group.atoms) <= parent_atoms
            and group.owner in parent_atoms for group in inventory.principal)):
        return None
    text = alkyl_to_alkoxy(child.name)
    if not text:
        return None
    return CarboxyalkoxyProof(text, 'compound', root, host, atoms, child)


def validate_carboxyalkyl_proof(mol, Chem, proof):
    """Rebuild all roles and compare with a fresh same-order source graph."""
    if not isinstance(proof, CarboxyalkylProof):
        return False
    probe = Chem.Mol(mol)
    return carboxyalkyl_prefix_proof(probe, Chem, proof.root, proof.host, proof.atoms) == proof


def validate_carboxyalkoxy_proof(mol, Chem, proof, parent_atoms):
    """Rebuild the source roles and parent partition on a fresh graph."""
    if not isinstance(proof, CarboxyalkoxyProof):
        return False
    probe = Chem.Mol(mol)
    return carboxyalkoxy_prefix_proof(
        probe, Chem, proof.root, proof.host, proof.atoms, parent_atoms) == proof
