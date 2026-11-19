"""Fast handoff for an actually selected source-mapped 5/7/6 fused acid."""


def bounded_fused_acid_requires_full(mol, Chem):
    faces = mol.GetRingInfo().AtomRings()
    if (len(faces) < 3 or not all(any(len(face) == size for face in faces) for size in (5, 6, 7))
            or sum(atom.GetAtomicNum() == 8 for atom in mol.GetAtoms()) < 3
            or not any(atom.GetAtomicNum() == 6 and not atom.IsInRing()
                       and sum(neighbor.GetAtomicNum() == 8 for neighbor in atom.GetNeighbors()) == 2
                       for atom in mol.GetAtoms())):
        return False
    from .extended_inventory import selection_inventory
    from .three_face_benzoxepine_acid import three_face_fused_acid_decision

    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status != 'selected' or frontier.unknown_pairs or frontier.cycle
            or len(frontier.groups) != 1 or frontier.groups[0].kind != 'acid'):
        return False
    acid = frontier.groups[0]
    proof = three_face_fused_acid_decision(mol, Chem)
    if proof is None or proof.principal_centers != {acid.center} or not proof.preference_issues:
        return False
    roles = proof.component_roles
    role_names = [role for role, _atoms, _locants in roles]
    sets = [frozenset(atoms) for _role, atoms, _locants in roles]
    heavy = frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    if (role_names.count('fused-parent') != 1 or role_names.count('carboxylic-acid') != 1
            or any(role not in {'fused-parent', 'carboxylic-acid', 'ring-substituent'}
                   for role in role_names)
            or proof.atoms != heavy or sum(map(len, sets)) != len(heavy)
            or frozenset().union(*sets) != heavy
            or acid.owner not in sets[role_names.index('fused-parent')]
            or frozenset(acid.atoms) != sets[role_names.index('carboxylic-acid')]):
        return False
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision

    actual = _actual_principal_decision(mol, Chem)
    return bool(actual is not None and actual.prepared is None
                and actual.comparison == proof and actual.output_name == proof.name
                and actual.requirement.atoms == (acid.center,)
                and set(proof.preference_issues) <= set(actual.preference_issues))
