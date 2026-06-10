from __future__ import annotations

import re

from .molecule_cache import mol_cache_get, mol_cache_set


_STEREO_ATOM_LOCANT_RE = re.compile(r"^\d+'?$")
_STEREO_MULTIPLIERS = {
    "bis": 2,
    "tris": 3,
    "tetrakis": 4,
    "pentakis": 5,
    "hexakis": 6,
}


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
    locanted_bond_descriptors = _locanted_bond_stereo_descriptors_in_name(name)
    if locanted_bond_descriptors:
        bond_descriptors = _remove_locanted_atom_descriptors(bond_descriptors, locanted_bond_descriptors)
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


def apply_locanted_stereochemical_prefix(
    mol,
    Chem,
    name: str,
    locants: tuple[tuple[int, str], ...],
    *,
    atom_order: str = "idx_desc",
    include_unlocanted: bool = True,
) -> str:
    """Prepend a locanted stereo prefix when a candidate provides atom locants.

    Mapped parent stereocenters are locanted. Unmapped centers/bonds keep the
    existing unlocanted fallback descriptors, after removing descriptors already
    represented inside the rendered name.
    """

    if not name or _has_leading_stereochemical_prefix(name):
        return name
    if not locants or not _has_assigned_stereo(mol, Chem):
        return name
    locant_by_atom = _stereo_locant_map(locants)
    if not locant_by_atom:
        return name

    atom_pieces, unlocanted_atom_descriptors = _locanted_atom_stereo_pieces(
        mol,
        Chem,
        locant_by_atom,
        name,
        atom_order=atom_order,
    )
    bond_pieces, unlocanted_bond_descriptors = _locanted_bond_stereo_pieces(
        _indexed_bond_stereochemistry(mol, Chem),
        locant_by_atom,
        name,
    )
    pieces = atom_pieces + bond_pieces
    if not pieces:
        return name
    locanted = [piece for _sort_key, piece in sorted(pieces)]
    descriptors = locanted
    if include_unlocanted:
        descriptors += unlocanted_atom_descriptors + unlocanted_bond_descriptors
    prefix = ",".join(descriptors)
    return f"({prefix})-{name}"


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
    descriptors = [
        descriptor[-1]
        for descriptor in re.findall(r"(?<![A-Za-z])\d+'?[RS](?![a-z])", name)
    ]
    descriptors.extend(_extra_multiplied_locanted_stereo_descriptors(name, "RS"))
    return descriptors


def _locanted_bond_stereo_descriptors_in_name(name: str) -> list[str]:
    descriptors = [
        descriptor[-1]
        for descriptor in re.findall(r"(?<![A-Za-z])\d+'?[EZ](?![a-z])", name)
    ]
    descriptors.extend(_extra_multiplied_locanted_stereo_descriptors(name, "EZ"))
    return descriptors


def _extra_multiplied_locanted_stereo_descriptors(name: str, kinds: str) -> list[str]:
    extras: list[str] = []
    pattern = re.compile(
        r"\b(?P<mult>bis|tris|tetrakis|pentakis|hexakis)\({2,}(?P<locants>\d+'?[RSZE](?:,\d+'?[RSZE])*)\)-"
    )
    for match in pattern.finditer(name):
        count = _STEREO_MULTIPLIERS.get(match.group("mult"), 1)
        if count <= 1:
            continue
        pieces = [
            descriptor[-1]
            for descriptor in re.findall(r"\d+'?[RSZE]", match.group("locants"))
            if descriptor[-1] in kinds
        ]
        for _ in range(count - 1):
            extras.extend(pieces)
    return extras


def _locanted_atom_stereo_texts_in_name(name: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])\d+'?[RS](?![a-z])", name))


def _locanted_bond_stereo_texts_in_name(name: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])\d+'?[EZ](?![a-z])", name))


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


def _stereo_locant_map(locants: tuple[tuple[int, str], ...]) -> dict[int, str]:
    mapped: dict[int, str] = {}
    for item in locants:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        atom_idx, locant = item
        if not isinstance(atom_idx, int):
            continue
        text = str(locant)
        if not text or not _STEREO_ATOM_LOCANT_RE.match(text):
            continue
        mapped[atom_idx] = text
    return mapped


def _locanted_atom_stereo_pieces(
    mol,
    Chem,
    locant_by_atom: dict[int, str],
    name: str,
    *,
    atom_order: str,
) -> tuple[list[tuple[tuple, str]], list[str]]:
    existing = _locanted_atom_stereo_texts_in_name(name)
    pieces = []
    unlocanted = []
    for atom_idx, descriptor in assigned_atom_stereochemistry(mol, Chem, atom_order=atom_order):
        locant = locant_by_atom.get(atom_idx)
        if not locant:
            unlocanted.append(descriptor)
            continue
        text = f"{locant}{descriptor}"
        pieces.append((_stereo_locant_sort_key(locant), text))
    if len(pieces) <= 1 and existing:
        pieces = [piece for piece in pieces if piece[1] not in existing]
    if unlocanted:
        unlocanted = _remove_locanted_atom_descriptors(unlocanted, _locanted_atom_stereo_descriptors_in_name(name))
    return pieces, unlocanted


def _indexed_bond_stereochemistry(mol, Chem) -> list[tuple[int, int, str]]:
    cached = mol_cache_get(mol, "indexed_bond_stereochemistry")
    if cached is not None:
        return cached
    records: list[tuple[int, int, str]] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        stereo = bond.GetStereo()
        if stereo in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}:
            descriptor = "E"
        elif stereo in {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}:
            descriptor = "Z"
        else:
            continue
        records.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), descriptor))
    return mol_cache_set(mol, "indexed_bond_stereochemistry", records)


def _locanted_bond_stereo_pieces(
    records: list[tuple[int, int, str]],
    locant_by_atom: dict[int, str],
    name: str,
) -> tuple[list[tuple[tuple, str]], list[str]]:
    existing = _locanted_bond_stereo_texts_in_name(name)
    pieces = []
    unlocanted = []
    for begin_idx, end_idx, descriptor in records:
        begin_locant = locant_by_atom.get(begin_idx)
        end_locant = locant_by_atom.get(end_idx)
        if not begin_locant or not end_locant:
            unlocanted.append(descriptor)
            continue
        locant = min(begin_locant, end_locant, key=_stereo_locant_sort_key)
        text = f"{locant}{descriptor}"
        if text in existing:
            continue
        pieces.append((_stereo_locant_sort_key(locant), text))
    if unlocanted:
        unlocanted = _remove_locanted_atom_descriptors(unlocanted, _locanted_bond_stereo_descriptors_in_name(name))
    return pieces, unlocanted


def _stereo_locant_sort_key(locant: str) -> tuple[int, int, str]:
    base = locant.rstrip("'")
    return (int(base), locant.count("'"), locant)


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
