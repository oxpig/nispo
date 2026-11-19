"""Handoff for actually selected, fully mapped branched sulfinamides."""


def bounded_branched_sulfinamide_requires_full(mol, Chem):
    """Require the source-selected full producer and exact atom roles."""
    sulfur = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 16 or atom.IsInRing() or atom.GetDegree() != 3:
            continue
        groups = [(n.GetAtomicNum(), mol.GetBondBetweenAtoms(
            atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble()) for n in atom.GetNeighbors()]
        if sorted(groups) == [(6, 1.0), (7, 1.0), (8, 2.0)]:
            sulfur.append(atom.GetIdx())
    if len(sulfur) != 1:
        return False
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status != 'selected'
            or frontier.unknown_pairs or frontier.cycle or len(frontier.groups) != 1
            or frontier.groups[0].kind != 'sulfinamide'
            or frontier.groups[0].center != sulfur[0]):
        return False
    from .branched_sulfinamide_methods import branched_sulfinamide_decision

    proof = branched_sulfinamide_decision(mol, Chem)
    if proof is None or proof.principal_centers != {sulfur[0]}:
        return False
    roles = {name: frozenset(atoms) for name, atoms, _ in proof.component_roles}
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if (set(roles) != {'sulfinamide-parent', 'sulfinamide-suffix-unit',
                       'parent-substituents', 'N-substituents'}
            or len(proof.component_roles) != len(roles) or proof.atoms != heavy
            or sum(map(len, roles.values())) != len(heavy)
            or frozenset().union(*roles.values()) != heavy
            or not set(frontier.groups[0].atoms) <= roles['sulfinamide-suffix-unit']):
        return False
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision

    actual = _actual_principal_decision(mol, Chem)
    return bool(actual is not None and actual.prepared is None
                and actual.comparison == proof and actual.output_name == proof.name
                and actual.requirement.atoms == (sulfur[0],))
