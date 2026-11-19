"""Fast handoff for an actually selected, mapped ring/chain amide frontier."""


def bounded_ring_chain_amide_requires_full(mol, Chem):
    if not mol.GetRingInfo().NumRings():
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
    source = inventory.source_frontier
    if (not inventory.supported or source.status != 'tie' or source.unknown_pairs or source.cycle
            or len(source.groups) != 2 or any(g.kind != 'amide' for g in source.groups)
            or {g.center for g in source.groups} != set(centers)):
        return False
    from .ring_chain_amide_frontier import ring_chain_amide_frontier
    proof = ring_chain_amide_frontier(mol, Chem)
    if proof is None or proof.ring.principal_centers | proof.chain.principal_centers != set(centers):
        return False
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    for candidate, expected_roles in ((proof.ring, {'ring-amide-parent', 'N-aryl-amido'}),
                                      (proof.chain, {'chain-amide-parent', 'chain-substituents'})):
        roles = {name: frozenset(atoms) for name, atoms, _ in candidate.component_roles}
        if (set(roles) != expected_roles or len(roles) != len(candidate.component_roles)
                or candidate.atoms != heavy or sum(map(len, roles.values())) != len(heavy)
                or frozenset().union(*roles.values()) != heavy):
            return False
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision
    actual = _actual_principal_decision(mol, Chem)
    return bool(actual is not None and actual.prepared is None and actual.comparison == proof
                and actual.output_name == proof.ring.name and set(actual.requirement.atoms) == set(centers))
