"""Route only source-proved acid-class shortcuts to the full writer."""


def bounded_acid_class_requires_full(mol, Chem):
    """Check cheap source motifs before invoking a mapped class proof.

    A complete compatibility shortcut can otherwise cite the principal acid
    or acyl halide only as a substituent. A positive result is returned only
    after one of the bounded full-writer producers proves its actual source
    graph and complete component ownership; this does not inspect name text.
    """
    atoms = tuple(mol.GetAtoms())
    if any(a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
           or int(a.GetChiralTag()) for a in atoms):
        return False
    if any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds()):
        return False
    if any(a.GetAtomicNum() == 7 and any(n.GetAtomicNum() == 7
           for n in a.GetNeighbors()) for a in atoms):
        from .bounded_hydrazide_acid_fast_guard import bounded_hydrazide_acid_requires_full

        if bounded_hydrazide_acid_requires_full(mol, Chem):
            return True
    halogens = {9, 17, 35, 53}
    for atom in atoms:
        if atom.GetAtomicNum() != 16:
            continue
        if (sum(n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(
                atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
                for n in atom.GetNeighbors()) == 2
                and any(n.GetAtomicNum() in halogens for n in atom.GetNeighbors())):
            from .bounded_sulfonyl_halides import benzene_sulfonyl_halide_decision

            return benzene_sulfonyl_halide_decision(mol, Chem) is not None
    for carbonyl in atoms:
        if carbonyl.GetAtomicNum() != 6 or carbonyl.IsInRing():
            continue
        neighbors = tuple(carbonyl.GetNeighbors())
        if not any(n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(
                carbonyl.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2 for n in neighbors):
            continue
        leaving = [n for n in neighbors if n.GetAtomicNum() in halogens | {8}
                   and mol.GetBondBetweenAtoms(carbonyl.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
        owners = [n for n in neighbors if n.GetAtomicNum() == 6
                  and mol.GetBondBetweenAtoms(carbonyl.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(leaving) != 1 or len(owners) != 1:
            continue
        owner = owners[0]
        aromatic_benzoic = owner.IsInRing() and owner.GetIsAromatic()
        alpha_branch = (not owner.IsInRing() and owner.GetDegree() == 2
                        and any(n.GetIdx() != carbonyl.GetIdx()
                                and ((n.GetAtomicNum() == 6 and n.IsInRing()
                                      and n.GetIsAromatic()) or n.GetAtomicNum() == 7)
                                for n in owner.GetNeighbors()))
        if not aromatic_benzoic and not alpha_branch:
            continue
        if leaving[0].GetAtomicNum() == 8 and leaving[0].GetDegree() == 1:
            from .bounded_retained_carboxylic_parents import (
                alkoxycarbonylamino_acetic_acid_parent, amino_acetic_acid_parent,
                aromatic_acetic_acid_parent,
            )

            return (alkoxycarbonylamino_acetic_acid_parent(mol, Chem) is not None
                    or amino_acetic_acid_parent(mol, Chem) is not None
                    or aromatic_acetic_acid_parent(mol, Chem) is not None)
        from .acid_derivative_methods import acid_derivative_decision

        return acid_derivative_decision(mol, Chem) is not None
    return False
