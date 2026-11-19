"""Fast handoff for the actually selected mapped phosphate ester."""


def bounded_ring_phosphate_requires_full(mol, Chem):
    """Require complete producer roles and the identical full selection."""
    if not mol.GetRingInfo().NumRings():
        return False
    phosphorus = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 15]
    if len(phosphorus) != 1:
        return False
    p = phosphorus[0]
    if (p.GetDegree() != 4 or sum(n.GetAtomicNum() == 8 for n in p.GetNeighbors()) != 4
            or not any(n.GetAtomicNum() == 8 and n.GetDegree() == 2
                       for n in p.GetNeighbors())):
        return False
    from .ring_methyl_phosphate_methods import ring_methyl_phosphate_decision

    proof = ring_methyl_phosphate_decision(mol, Chem)
    if proof is None or proof.principal_centers != {p.GetIdx()}:
        return False
    roles = tuple(frozenset(atoms) for _role, atoms, _locants in proof.component_roles)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if (proof.atoms != heavy or len(roles) != 2
            or sum(map(len, roles)) != len(heavy) or frozenset().union(*roles) != heavy):
        return False
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision

    actual = _actual_principal_decision(mol, Chem)
    return bool(actual is not None and actual.prepared is None
                and actual.comparison == proof and actual.output_name == proof.name
                and actual.requirement.atoms == (p.GetIdx(),))
