"""Fast handoff for actually selected mapped N-sulfonyl ring amides."""


def bounded_n_sulfonyl_ring_amide_requires_full(mol, Chem):
    """Join the original class frontier to the full writer's atom roles."""
    if not mol.GetRingInfo().NumRings():
        return False
    sulfur = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16
              and sum(neighbor.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(
                  atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                  for neighbor in atom.GetNeighbors()) == 2
              and any(neighbor.GetAtomicNum() == 7 for neighbor in atom.GetNeighbors())]
    if not sulfur:
        return False
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status != 'selected'
            or frontier.unknown_pairs or frontier.cycle or len(frontier.groups) != 1
            or frontier.groups[0].kind != 'amide'):
        return False
    from .n_sulfonyl_ring_amides import n_sulfonyl_ring_carboxamide_decision
    proof = n_sulfonyl_ring_carboxamide_decision(mol, Chem)
    if proof is None or proof.principal_centers != {frontier.groups[0].center}:
        return False
    roles = {name: frozenset(atoms) for name, atoms, _ in proof.component_roles}
    heavy = frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    if (set(roles) != {'ring-carboxamide', 'N-sulfonyl'}
            or len(proof.component_roles) != len(roles) or proof.atoms != heavy
            or sum(map(len, roles.values())) != len(heavy)
            or frozenset().union(*roles.values()) != heavy
            or not set(frontier.groups[0].atoms) <= roles['ring-carboxamide']
            or not set(sulfur) & roles['N-sulfonyl']):
        return False
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision
    actual = _actual_principal_decision(mol, Chem)
    return bool(actual is not None and actual.prepared is None
                and actual.comparison == proof and actual.output_name == proof.name
                and actual.requirement.atoms == (frontier.groups[0].center,))
