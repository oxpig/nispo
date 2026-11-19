"""Fast-mode handoff only for an actually selected mapped nested amide."""


def bounded_nested_amide_requires_full(mol, Chem):
    """Check the original class frontier and the full writer's atom roles."""
    # Every admitted site has one acyclic C(=O)–N group, one of which is
    # attached directly to a ring carbon. This inexpensive prefilter cannot
    # itself establish a name or principal class.
    sites = []
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6 or carbon.IsInRing():
            continue
        neighbors = [(atom, mol.GetBondBetweenAtoms(carbon.GetIdx(), atom.GetIdx())
                      .GetBondTypeAsDouble()) for atom in carbon.GetNeighbors()]
        if (sum(atom.GetAtomicNum() == 8 and order == 2 for atom, order in neighbors) == 1
                and sum(atom.GetAtomicNum() == 7 and order == 1 for atom, order in neighbors) == 1
                and sum(atom.GetAtomicNum() == 6 and order == 1 for atom, order in neighbors) == 1):
            sites.append((carbon.GetIdx(), any(atom.IsInRing() and atom.GetAtomicNum() == 6
                                               and order == 1 for atom, order in neighbors)))
    if len(sites) != 3 or sum(is_ring for _, is_ring in sites) != 1:
        return False

    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status != 'tie'
            or frontier.unknown_pairs or frontier.cycle or len(frontier.groups) != 3
            or any(group.kind != 'amide' for group in frontier.groups)
            or {group.center for group in frontier.groups} != {center for center, _ in sites}):
        return False

    from .nested_amide_general import nested_amide_general_decision
    proof = nested_amide_general_decision(mol, Chem)
    if proof is None or len(proof.principal_centers) != 1:
        return False
    roles = {name: frozenset(atoms) for name, atoms, _ in proof.component_roles}
    heavy = frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    if (set(roles) != {'ring-carboxamide-parent', 'mapped-parent-substituents'}
            or len(proof.component_roles) != len(roles) or proof.atoms != heavy
            or sum(map(len, roles.values())) != len(heavy)
            or frozenset().union(*roles.values()) != heavy):
        return False
    selected = [group for group in frontier.groups if group.center in proof.principal_centers]
    if len(selected) != 1 or not set(selected[0].atoms) <= roles['ring-carboxamide-parent']:
        return False

    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision
    actual = _actual_principal_decision(mol, Chem)
    return bool(actual is not None and actual.prepared is None
                and actual.comparison == proof and actual.output_name == proof.name
                and set(actual.requirement.atoms) == {group.center for group in frontier.groups})
