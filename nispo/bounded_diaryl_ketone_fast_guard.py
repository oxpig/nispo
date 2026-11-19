"""Fast handoff only for an actually selected, fully owned ketone result."""


def bounded_diaryl_ketone_requires_full(mol, Chem):
    from .acyclic_diaryl_ketone import diaryl_ketone_site
    if diaryl_ketone_site(mol) is None:
        return False
    from .acyclic_diaryl_ketone import diaryl_methanone_decision
    proof = diaryl_methanone_decision(mol, Chem)
    if proof is None:
        return False
    roles = {name: frozenset(atoms) for name, atoms, _ in proof.component_roles}
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    carbon, oxygen, roots = diaryl_ketone_site(mol)
    if (set(roles) != {'methanone', 'organyl-a', 'organyl-b'}
            or len(proof.component_roles) != len(roles) or proof.atoms != heavy
            or sum(map(len, roles.values())) != len(heavy)
            or frozenset().union(*roles.values()) != heavy
            or roles['methanone'] != {carbon, oxygen}
            or roots[0] not in roles['organyl-a'] or roots[1] not in roles['organyl-b']
            or proof.principal_centers != {carbon}):
        return False
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision
    actual = _actual_principal_decision(mol, Chem)
    return bool(actual is not None and actual.prepared is None
                and actual.comparison == proof and actual.output_name == proof.name
                and actual.requirement.atoms == (carbon,)
                and actual.preference_issues)
