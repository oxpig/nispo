from __future__ import annotations


def apply_stereochemical_prefix(mol, Chem, name: str) -> str:
    """Prepend OPSIN bracket stereochemistry for assigned input stereo.

    NISPO's structural rules usually produce the right connectivity, while
    OPSIN defaults to unspecified stereochemistry unless an `(R)`, `(S)`,
    `(E)`, or `(Z)` bracket appears before the name.  Unlocanted descriptors
    are intentionally used here: OPSIN applies them in generated atom order,
    which matches many of NISPO's deterministic parent names and avoids
    inventing locants for every rule family.
    """

    if not name or name.startswith("("):
        return name
    if not _has_assigned_stereo(mol, Chem):
        return name

    bond_descriptors = _bond_stereochemistry(mol, Chem)
    descriptors = _atom_stereochemistry(mol, Chem)
    descriptors.extend(bond_descriptors)
    if not descriptors:
        return name
    return f"({','.join(descriptors)})-{name}"


def _has_assigned_stereo(mol, Chem) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return True
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return True
    return False


def _atom_stereochemistry(mol, Chem) -> list[str]:
    has_chiral_tag = any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    )
    if not has_chiral_tag:
        return []
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
    assigned = [
        (idx, descriptor)
        for idx, descriptor in centers
        if descriptor in {"R", "S"}
    ]
    return [descriptor for _idx, descriptor in sorted(assigned, reverse=True)]


def _bond_stereochemistry(mol, Chem) -> list[str]:
    descriptors: list[str] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        stereo = bond.GetStereo()
        if stereo in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}:
            descriptors.append("E")
        elif stereo in {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}:
            descriptors.append("Z")
    return descriptors
