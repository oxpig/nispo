"""Fast handoff only for the selected mapped O/S-linked polyacid parent."""


def bounded_chalcogen_polyacid_requires_full(mol, Chem):
    """Join the source graph to the actual comprehensive selection."""
    if (mol is None or mol.GetRingInfo().NumRings()
            or sum(atom.GetAtomicNum() == 8 for atom in mol.GetAtoms()) < 6):
        return False
    possible_acids = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        oxygen = [(neighbor, mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()))
                  for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 8]
        if (any(bond.GetBondTypeAsDouble() == 2 for _neighbor, bond in oxygen)
                and any(bond.GetBondTypeAsDouble() == 1 and neighbor.GetDegree() == 1
                        for neighbor, bond in oxygen)):
            possible_acids += 1
    if possible_acids < 3:
        return False
    from .acyclic_chalcogen_polyacid import (
        acyclic_chalcogen_polyacid_scope, mapped_polyacid_proposal_complete,
    )
    from .bounded_mixed_acid_ester_fast_guard import _actual_pending_proposal
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    scope = acyclic_chalcogen_polyacid_scope(mol, Chem, inventory)
    if scope is None:
        return False
    selected = _actual_pending_proposal(mol, Chem)
    if selected is None:
        return False
    decision, proposal = selected
    acid_centers = frozenset(group.center for group in scope.acid_groups)
    return (frozenset(decision.requirement.atoms) == acid_centers
            and mapped_polyacid_proposal_complete(mol, Chem, scope, proposal))
