"""Fast handoff for an actually selected mapped linked-ring amide."""


def bounded_linked_ring_amide_requires_full(mol, Chem):
    if mol.GetRingInfo().NumRings() < 4:
        return False
    centers = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.IsInRing():
            continue
        bonds = [(neighbor.GetAtomicNum(), mol.GetBondBetweenAtoms(
            atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble()) for neighbor in atom.GetNeighbors()]
        if (sum(number == 8 and order == 2 for number, order in bonds) == 1
                and sum(number == 7 and order == 1 for number, order in bonds) == 1):
            centers.append(atom.GetIdx())
    if len(centers) != 2:
        return False
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status != 'tie' or frontier.unknown_pairs or frontier.cycle
            or len(frontier.groups) != 2 or any(group.kind != 'amide' for group in frontier.groups)
            or {group.center for group in frontier.groups} != set(centers)):
        return False
    from .mapped_linked_ring_amides import linked_ring_amide_decision
    proof = linked_ring_amide_decision(mol, Chem)
    if proof is None or len(proof.principal_centers) != 1:
        return False
    roles = {name: frozenset(atoms) for name, atoms, _ in proof.component_roles}
    expected = {'selected-ring-amide', 'central-ring', 'linked-ring',
                'subordinate-ring-amide', 'mapped-substituents'}
    heavy = frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    if (set(roles) != expected or len(roles) != len(proof.component_roles) or proof.atoms != heavy
            or sum(map(len, roles.values())) != len(heavy)
            or frozenset().union(*roles.values()) != heavy):
        return False
    selected = [group for group in frontier.groups if group.center in proof.principal_centers]
    subordinate = [group for group in frontier.groups if group.center not in proof.principal_centers]
    if (len(selected) != 1 or len(subordinate) != 1
            or not set(selected[0].atoms) <= roles['selected-ring-amide']
            or not set(subordinate[0].atoms) <= roles['subordinate-ring-amide']):
        return False
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision
    actual = _actual_principal_decision(mol, Chem)
    return bool(actual is not None and actual.prepared is None and actual.comparison == proof
                and actual.output_name == proof.name and set(actual.requirement.atoms) == set(centers))
