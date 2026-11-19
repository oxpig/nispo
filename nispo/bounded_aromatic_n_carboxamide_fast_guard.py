"""Fast handoff for source-proved aromatic N-carboxamides."""


def bounded_aromatic_n_carboxamide_requires_full(mol, Chem):
    """Require the actual full decision and its complete source partition."""
    if mol.GetRingInfo().NumRings() != 1:
        return False
    possible = []
    for center in mol.GetAtoms():
        if center.GetAtomicNum() != 6 or center.IsInRing() or center.GetDegree() != 3:
            continue
        groups = [(n, mol.GetBondBetweenAtoms(center.GetIdx(), n.GetIdx()).GetBondTypeAsDouble())
                  for n in center.GetNeighbors()]
        if (sum(n.GetAtomicNum() == 8 and order == 2 for n, order in groups) == 1
                and sum(n.GetAtomicNum() == 7 and n.IsInRing() and n.GetIsAromatic()
                        and order == 1 for n, order in groups) == 1
                and sum(n.GetAtomicNum() == 7 and not n.IsInRing()
                        and order == 1 for n, order in groups) == 1):
            possible.append(center.GetIdx())
    if len(possible) != 1:
        return False
    from .aromatic_n_carboxamide_methods import aromatic_n_carboxamide_decision

    proof = aromatic_n_carboxamide_decision(mol, Chem)
    if proof is None or proof.principal_centers != {possible[0]} or not proof.preference_issues:
        return False
    roles = proof.component_roles
    sets = [frozenset(atoms) for _role, atoms, _locants in roles]
    role_names = [role for role, _atoms, _locants in roles]
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if (role_names.count('parent-hydride') != 1 or role_names.count('carboxamide') != 1
            or any(role not in {'parent-hydride', 'carboxamide', 'amide-N-substituent',
                                'ring-substituent'} for role in role_names)
            or proof.atoms != heavy or sum(map(len, sets)) != len(heavy)
            or frozenset().union(*sets) != heavy
            or possible[0] not in sets[role_names.index('carboxamide')]):
        return False
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision

    actual = _actual_principal_decision(mol, Chem)
    return bool(actual is not None and actual.prepared is None
                and actual.comparison == proof and actual.output_name == proof.name
                and actual.requirement.atoms == (possible[0],)
                and set(proof.preference_issues) <= set(actual.preference_issues))
