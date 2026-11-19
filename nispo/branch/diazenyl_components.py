"""Source-owned geometrical diazenyl prefixes under a ketone suffix.

P-68.3.1.3 describes diazenyl prefix nomenclature when a principal
characteristic group is present. This deliberately limited construction
requires the original graph's sole recognized principal group to be an
acyclic ketone and accepts one specified azo stereobond only.
"""


def mapped_diazenyl_prefix(mol, Chem, root, blocked, atoms):
    from ..branch_rules import collect_subtree
    from ..component_mapping import mapped_branch, mapped_branch_ring_atoms, _record_branch_ring_atoms
    from ..formatting import enclose_name_unit
    from ..molecule_cache import mol_cache_dict
    from ..stereochemistry import assigned_bond_stereochemistry

    near = mol.GetAtomWithIdx(root)
    cut = mol.GetBondBetweenAtoms(root, blocked)
    if (near.GetAtomicNum() != 7 or near.IsInRing() or near.GetFormalCharge()
            or near.GetNumRadicalElectrons() or near.GetTotalNumHs(includeNeighbors=True)
            or near.GetDegree() != 2 or cut is None or cut.GetBondTypeAsDouble() != 1
            or mol.GetAtomWithIdx(blocked).GetAtomicNum() != 6):
        return None
    remote = [a for a in near.GetNeighbors() if a.GetIdx() != blocked]
    if len(remote) != 1:
        return None
    far = remote[0]
    pair = mol.GetBondBetweenAtoms(root, far.GetIdx())
    if (far.GetAtomicNum() != 7 or far.IsInRing() or far.GetFormalCharge()
            or far.GetNumRadicalElectrons() or far.GetTotalNumHs(includeNeighbors=True)
            or far.GetDegree() != 2 or pair.GetBondTypeAsDouble() != 2
            or int(pair.GetStereo()) not in {2, 3}):
        return None
    distal = [a for a in far.GetNeighbors() if a.GetIdx() != root]
    if (len(distal) != 1 or distal[0].GetAtomicNum() != 6
            or mol.GetBondBetweenAtoms(far.GetIdx(), distal[0].GetIdx()).GetBondTypeAsDouble() != 1):
        return None
    from ..acyclic_diaryl_ketone import diaryl_methanone_scope
    source_scope = diaryl_methanone_scope(mol, Chem)
    if (source_scope is None or len(source_scope[3]) != 1
            or source_scope[3][0].GetIdx() != pair.GetIdx()):
        return None
    labels = [(a, b, label) for a, b, label in assigned_bond_stereochemistry(mol, Chem)
              if {a, b} == {root, far.GetIdx()}]
    if len(labels) != 1 or labels[0][2] not in {'E', 'Z'}:
        return None
    scope = frozenset(atoms)
    distal_root = distal[0].GetIdx()
    distal_scope = frozenset(collect_subtree(mol, distal_root, far.GetIdx()))
    cuts = {(i, n.GetIdx()) for i in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() not in scope}
    if (scope != frozenset(collect_subtree(mol, root, blocked))
            or cuts != {(root, blocked)} or root in distal_scope or far.GetIdx() in distal_scope
            or scope != distal_scope | {root, far.GetIdx()}):
        return None
    text, owned = mapped_branch(mol, Chem, distal_root, far.GetIdx())
    rings = {i for i in distal_scope if mol.GetAtomWithIdx(i).IsInRing()}
    if (not text or owned != distal_scope
            or not rings <= mapped_branch_ring_atoms(mol, Chem, distal_root, far.GetIdx(), text, owned)):
        return None
    ligand = text if text.isalpha() else enclose_name_unit(text)
    prefix = f'({labels[0][2]})-{ligand}diazenyl'
    mol_cache_dict(mol, 'rendered_mapped_branch_bond_stereo')[
        (root, far.GetIdx(), prefix, scope)] = labels[0][2]
    _record_branch_ring_atoms(mol, Chem, root, blocked, prefix, scope, rings)
    return prefix, scope
