from __future__ import annotations

import re

from .molecule_cache import mol_cache_get, mol_cache_set


def apply_stereochemical_prefix(mol, Chem, name: str, *, atom_order: str = "idx_desc") -> str:
    """Prepend OPSIN bracket stereochemistry for assigned input stereo.

    NISPO's structural rules usually produce the right connectivity, while
    OPSIN defaults to unspecified stereochemistry unless an `(R)`, `(S)`,
    `(E)`, or `(Z)` bracket appears before the name.  Unlocanted descriptors
    are intentionally used here: OPSIN applies them in generated atom order,
    which matches many of NISPO's deterministic parent names and avoids
    inventing locants for every rule family.
    """

    if not name or _has_leading_stereochemical_prefix(name):
        return name
    if not _has_assigned_stereo(mol, Chem):
        return name

    bond_descriptors = _bond_stereochemistry(mol, Chem)
    descriptors = []
    locanted_atom_descriptors = _locanted_atom_stereo_descriptors_in_name(name)
    if not _name_has_complete_locanted_atom_stereo(mol, Chem, locanted_atom_descriptors):
        descriptors = _atom_stereochemistry(mol, Chem, atom_order=atom_order)
        if locanted_atom_descriptors:
            descriptors = _remove_locanted_atom_descriptors(descriptors, locanted_atom_descriptors)
    descriptors.extend(bond_descriptors)
    if not descriptors:
        return name
    return f"({','.join(descriptors)})-{name}"


def _has_assigned_stereo(mol, Chem) -> bool:
    cached = mol_cache_get(mol, "has_assigned_stereo")
    if cached is not None:
        return cached
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return mol_cache_set(mol, "has_assigned_stereo", True)
    result = _has_assigned_bond_stereo(mol, Chem)
    return mol_cache_set(mol, "has_assigned_stereo", result)


def _has_assigned_bond_stereo(mol, Chem) -> bool:
    cached = mol_cache_get(mol, "has_assigned_bond_stereo")
    if cached is not None:
        return cached
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return mol_cache_set(mol, "has_assigned_bond_stereo", True)
    return mol_cache_set(mol, "has_assigned_bond_stereo", False)


def _has_leading_stereochemical_prefix(name: str) -> bool:
    return bool(re.match(r"^\((?:\d*[RSZE](?:,\d*[RSZE])*)\)-", name))


def _locanted_atom_stereo_descriptors_in_name(name: str) -> list[str]:
    return [
        descriptor[-1]
        for descriptor in re.findall(r"(?<![A-Za-z])\d+[RS](?![a-z])", name)
    ]


def _name_has_complete_locanted_atom_stereo(mol, Chem, locanted_descriptors: list[str]) -> bool:
    if not locanted_descriptors:
        return False
    chiral_count = len(assigned_atom_stereochemistry(mol, Chem))
    return chiral_count > 0 and len(locanted_descriptors) >= chiral_count


def _remove_locanted_atom_descriptors(descriptors: list[str], locanted_descriptors: list[str]) -> list[str]:
    remaining = list(descriptors)
    for locanted in locanted_descriptors:
        try:
            remaining.remove(locanted)
        except ValueError:
            continue
    return remaining


def _atom_stereochemistry(mol, Chem, *, atom_order: str) -> list[str]:
    return [
        descriptor
        for _idx, descriptor in assigned_atom_stereochemistry(
            mol,
            Chem,
            atom_order=atom_order,
        )
    ]


def assigned_atom_stereochemistry(mol, Chem, *, atom_order: str = "idx_desc") -> list[tuple[int, str]]:
    cache_key = f"assigned_atom_stereochemistry_{atom_order}"
    cached = mol_cache_get(mol, cache_key)
    if cached is not None:
        return cached
    has_chiral_tag = any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    )
    if not has_chiral_tag:
        return mol_cache_set(mol, cache_key, [])
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
    assigned = [
        (idx, descriptor)
        for idx, descriptor in centers
        if descriptor in {"R", "S"}
    ]
    if not assigned:
        assigned = _atom_cip_stereochemistry_from_props(mol)
    reverse = atom_order != "idx"
    return mol_cache_set(mol, cache_key, sorted(assigned, reverse=reverse))


def _atom_cip_stereochemistry_from_props(mol) -> list[tuple[int, str]]:
    assigned = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("_CIPCode"):
            continue
        descriptor = atom.GetProp("_CIPCode")
        if descriptor in {"R", "S"}:
            assigned.append((atom.GetIdx(), descriptor))
    return assigned


def _bond_stereochemistry(mol, Chem) -> list[str]:
    cached = mol_cache_get(mol, "bond_stereochemistry")
    if cached is not None:
        return cached
    descriptors: list[str] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        stereo = bond.GetStereo()
        if stereo in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}:
            descriptors.append("E")
        elif stereo in {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}:
            descriptors.append("Z")
    return mol_cache_set(mol, "bond_stereochemistry", descriptors)
