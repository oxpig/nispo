"""Fast handoff for the actual mapped three-face lactam decision."""


def bounded_fused_lactam_requires_full(mol, Chem):
    faces = mol.GetRingInfo().AtomRings()
    if (len(faces) < 3 or not any(len(face) == 5 for face in faces)
            or sum(len(face) == 6 for face in faces) < 2
            or sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()) != 2
            or sum(a.GetAtomicNum() == 7 for a in mol.GetAtoms()) < 2
            or not any(a.GetAtomicNum() == 6 and a.IsInRing()
                       and any(n.GetAtomicNum() == 8 and not n.IsInRing()
                               and mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
                               for n in a.GetNeighbors()) for a in mol.GetAtoms())):
        return False
    from .three_face_heterolactam_fusion import three_face_lactam_decision

    proof = three_face_lactam_decision(mol, Chem)
    if proof is None or len(proof.principal_centers) != 1 or not proof.preference_issues:
        return False
    roles = proof.component_roles
    role_names = [role for role, _atoms, _locants in roles]
    sets = [frozenset(atoms) for _role, atoms, _locants in roles]
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if (role_names.count('fused-parent') != 1 or role_names.count('lactam-oxygen') != 1
            or any(role not in {'fused-parent', 'lactam-oxygen', 'ring-substituent'}
                   for role in role_names)
            or proof.atoms != heavy or sum(map(len, sets)) != len(heavy)
            or frozenset().union(*sets) != heavy
            or not proof.principal_centers <= sets[role_names.index('fused-parent')]):
        return False
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision

    actual = _actual_principal_decision(mol, Chem)
    return bool(actual is not None and actual.prepared is None
                and actual.comparison == proof and actual.output_name == proof.name
                and actual.requirement.atoms == tuple(sorted(proof.principal_centers))
                and set(proof.preference_issues) <= set(actual.preference_issues))
