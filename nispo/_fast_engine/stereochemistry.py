from __future__ import annotations

import re
from math import prod

from ..local_cip import assigned_chiral_centers
from ..molecule_cache import mol_cache_get as _shared_mol_cache_get
from ..stereochemistry import _rendered_mapped_branch_bonds, terminal_haloalkene_descriptor
from .molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set


_STEREO_LOCANT_TEXT = r"\d+(?:[a-z])?'?"
_STEREO_ATOM_LOCANT_RE = re.compile(rf"^{_STEREO_LOCANT_TEXT}$")
_STEREO_MULTIPLIERS = {
    "bis": 2,
    "tris": 3,
    "tetrakis": 4,
    "pentakis": 5,
    "hexakis": 6,
}
_STEREO_SCOPE_DELIMITERS = re.compile(
    r"(?:(?<![A-Za-z])(?P<mult>bis|tris|tetrakis|pentakis|hexakis))?"
    r"(?P<open>[\[({])|[])}]"
)


def apply_stereochemical_prefix(
    mol,
    Chem,
    name: str,
    *,
    atom_order: str = "idx_desc",
    invert_unlocanted_bond_stereo: bool = False,
) -> str:
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

    remaining_bonds, bond_name = _remaining_bond_stereochemistry(mol, Chem, name)
    bond_descriptors = [d for _, _, d in remaining_bonds]
    if invert_unlocanted_bond_stereo:
        bond_descriptors = _invert_bond_stereo_descriptors(bond_descriptors)
    if bond_descriptors:
        locanted_bond_descriptors = _locanted_bond_stereo_descriptors_in_name(bond_name)
        if locanted_bond_descriptors:
            bond_descriptors = _remove_represented_atom_descriptors(bond_descriptors, locanted_bond_descriptors)
    locanted_atom_descriptors = _locanted_atom_stereo_descriptors_in_name(name)
    relative_descriptor_count = len(re.findall(r"\((?:cis|trans)\)-", name))
    relative_descriptor_groups = _internal_relative_stereo_descriptor_groups(mol, Chem, relative_descriptor_count)
    relative_atom_descriptors = [
        descriptor for group in relative_descriptor_groups for descriptor in group
    ]
    internal_relative_descriptor_count = _internal_relative_stereo_descriptor_count(
        relative_descriptor_count,
        represented_relative_group_count=len(relative_descriptor_groups),
        represented_relative_descriptors=relative_atom_descriptors,
    )
    represented_atom_descriptors = (
        locanted_atom_descriptors
        + _internal_atom_stereo_descriptors_in_name(name)
        + relative_atom_descriptors
    )
    descriptors = []
    if not _name_has_complete_atom_stereo(
        mol,
        Chem,
        represented_atom_descriptors,
        internal_relative_descriptor_count=internal_relative_descriptor_count,
    ):
        aminoacyl = mol_cache_get(mol, "rendered_aminoacyl_stereo", {})
        owned = {idx for (_, text), idx in aminoacyl.items() if text in name}
        descriptors = ([d for i, d in assigned_atom_stereochemistry(mol, Chem, atom_order=atom_order)
                        if i not in owned] if owned else
                       _atom_stereochemistry(mol, Chem, atom_order=atom_order))
        if represented_atom_descriptors:
            descriptors = _remove_represented_atom_descriptors(descriptors, represented_atom_descriptors)
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
    invert_unlocanted_bond_stereo: bool = False,
    invert_locanted_bond_stereo: bool = False,
    reverse_descriptor_sequence: bool = False,
    reverse_atom_descriptor_sequence: bool = False,
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
    remaining_bonds, bond_name = _remaining_bond_stereochemistry(mol, Chem, name, locant_by_atom)
    bond_pieces, unlocanted_bond_descriptors = _locanted_bond_stereo_pieces(
        remaining_bonds, locant_by_atom, bond_name,
    )
    if invert_unlocanted_bond_stereo:
        unlocanted_bond_descriptors = _invert_bond_stereo_descriptors(unlocanted_bond_descriptors)
    if not atom_pieces and not bond_pieces:
        return name
    locanted_atoms = [piece for _sort_key, piece in sorted(atom_pieces)]
    locanted_bonds = [piece for _sort_key, piece in sorted(bond_pieces)]
    if invert_locanted_bond_stereo:
        locanted_bonds = _invert_locanted_bond_stereo_texts(locanted_bonds)
    if reverse_atom_descriptor_sequence:
        locanted_atoms = list(reversed(locanted_atoms))
    descriptors = locanted_atoms + locanted_bonds
    if include_unlocanted:
        descriptors += unlocanted_atom_descriptors + unlocanted_bond_descriptors
    if reverse_descriptor_sequence:
        descriptors = list(reversed(descriptors))
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
    descriptor = rf"(?:{_STEREO_LOCANT_TEXT})?[RSZE]"
    return bool(re.match(rf"^\((?:{descriptor}(?:,{descriptor})*)\)-", name))


def _locanted_atom_stereo_descriptors_in_name(name: str) -> list[str]:
    descriptors = [
        descriptor[-1]
        for descriptor in re.findall(rf"(?<![A-Za-z]){_STEREO_LOCANT_TEXT}[RS](?![a-z])", name)
    ]
    descriptors.extend(_extra_multiplied_locanted_stereo_descriptors(name, "RS"))
    return descriptors


def _internal_atom_stereo_descriptors_in_name(name: str) -> list[str]:
    return [
        match.group("descriptor")
        for match in re.finditer(
            r"(?<![A-Za-z])\((?P<descriptor>[RS])\)-[A-Za-z0-9'\[\]\(\),.-]*"
            r"(?:sulfinyl|sulfinimidoyl|sulfonimidoyl|sulfimidoyl)",
            name,
        )
    ]


def _internal_relative_stereo_descriptor_count(
    descriptor_count: int,
    *,
    represented_relative_group_count: int,
    represented_relative_descriptors: list[str],
) -> int:
    if descriptor_count <= 0:
        return 0
    if not represented_relative_descriptors:
        return 2 * descriptor_count
    return 2 * max(0, descriptor_count - represented_relative_group_count)


def _internal_relative_stereo_descriptor_groups(mol, Chem, descriptor_count: int) -> list[list[str]]:
    if descriptor_count <= 0:
        return []
    assigned = dict(assigned_atom_stereochemistry(mol, Chem))
    descriptor_groups = []
    for raw_ring in mol.GetRingInfo().AtomRings():
        tagged = _simple_relative_ring_branch_tagged_atoms(mol, Chem, frozenset(raw_ring))
        if tagged:
            descriptors = [assigned[atom_idx] for atom_idx in sorted(tagged) if atom_idx in assigned]
            if descriptors:
                descriptor_groups.append(descriptors)
    descriptor_groups.sort(key=lambda values: (-len(values), values))
    return descriptor_groups[:descriptor_count]


def _simple_relative_ring_branch_tagged_atoms(mol, Chem, ring: frozenset[int]) -> frozenset[int]:
    tagged = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetIdx() in ring and atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    if len(tagged) != 2:
        return frozenset()
    if any(mol.GetAtomWithIdx(atom_idx).GetSymbol() != "C" for atom_idx in ring):
        return frozenset()
    if any(atom.GetSymbol() != "C" or atom.GetIsAromatic() for atom in tagged):
        return frozenset()
    if any(_ring_external_heavy_degree(mol, atom.GetIdx(), ring) != 1 for atom in tagged):
        return frozenset()
    for bond in mol.GetBonds():
        if (
            bond.GetBeginAtomIdx() in ring
            and bond.GetEndAtomIdx() in ring
            and bond.GetBondType() != Chem.BondType.SINGLE
        ):
            return frozenset()
    return frozenset(atom.GetIdx() for atom in tagged)


def _ring_external_heavy_degree(mol, atom_idx: int, ring_atoms: frozenset[int]) -> int:
    return sum(
        1
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if nbr.GetIdx() not in ring_atoms and nbr.GetAtomicNum() > 1
    )


def _locanted_bond_stereo_descriptors_in_name(name: str) -> list[str]:
    descriptors = [
        descriptor[-1]
        for descriptor in re.findall(rf"(?<![A-Za-z]){_STEREO_LOCANT_TEXT}[EZ](?![a-z])", name)
    ]
    descriptors.extend(_extra_multiplied_locanted_stereo_descriptors(name, "EZ"))
    return descriptors


def _extra_multiplied_locanted_stereo_descriptors(name: str, kinds: str) -> list[str]:
    extras: list[str] = []
    pattern = re.compile(
        rf"\b(?P<mult>bis|tris|tetrakis|pentakis|hexakis)\({{2,}}(?P<locants>{_STEREO_LOCANT_TEXT}[RSZE](?:,{_STEREO_LOCANT_TEXT}[RSZE])*)\)-"
    )
    for match in pattern.finditer(name):
        count = _STEREO_MULTIPLIERS.get(match.group("mult"), 1)
        if count <= 1:
            continue
        pieces = [
            descriptor[-1]
            for descriptor in re.findall(rf"{_STEREO_LOCANT_TEXT}[RSZE]", match.group("locants"))
            if descriptor[-1] in kinds
        ]
        for _ in range(count - 1):
            extras.extend(pieces)
    return extras


def _locanted_atom_stereo_texts_in_name(name: str) -> set[str]:
    return set(re.findall(rf"(?<![A-Za-z]){_STEREO_LOCANT_TEXT}[RS](?![a-z])", name))


def _locanted_bond_stereo_texts_in_name(name: str) -> set[str]:
    return set(re.findall(rf"(?<![A-Za-z]){_STEREO_LOCANT_TEXT}[EZ](?![a-z])", name))


def _name_has_complete_atom_stereo(
    mol,
    Chem,
    represented_descriptors: list[str],
    *,
    internal_relative_descriptor_count: int = 0,
) -> bool:
    represented_count = len(represented_descriptors) + internal_relative_descriptor_count
    if represented_count <= 0:
        return False
    chiral_count = len(assigned_atom_stereochemistry(mol, Chem))
    return chiral_count > 0 and represented_count >= chiral_count


def _remove_represented_atom_descriptors(descriptors: list[str], represented_descriptors: list[str]) -> list[str]:
    remaining = list(descriptors)
    for represented in represented_descriptors:
        try:
            remaining.remove(represented)
        except ValueError:
            continue
    return remaining


def _invert_bond_stereo_descriptors(descriptors: list[str]) -> list[str]:
    return [{"E": "Z", "Z": "E"}.get(descriptor, descriptor) for descriptor in descriptors]


def _invert_locanted_bond_stereo_texts(descriptors: list[str]) -> list[str]:
    return [descriptor[:-1] + {"E": "Z", "Z": "E"}.get(descriptor[-1:], descriptor[-1:]) for descriptor in descriptors]


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
    raw_cached = mol_cache_get(mol, "assigned_atom_stereochemistry_raw")
    if raw_cached is None:
        raw_cached = _assigned_atom_stereochemistry_raw(mol, Chem)
        mol_cache_set(mol, "assigned_atom_stereochemistry_raw", raw_cached)
    reverse = atom_order != "idx"
    return mol_cache_set(mol, cache_key, sorted(raw_cached, reverse=reverse))


def _assigned_atom_stereochemistry_raw(mol, Chem) -> list[tuple[int, str]]:
    has_chiral_tag = any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    )
    if not has_chiral_tag:
        return []
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centers = assigned_chiral_centers(mol, Chem)
    assigned = [
        (idx, descriptor)
        for idx, descriptor in centers
        if descriptor in {"R", "S"}
    ]
    if not assigned:
        # The relative-ring writer can use the lower-case pseudo descriptors
        # from this same stereo census without asking RDKit to repeat it.
        if centers:
            mol_cache_set(mol, "fast_pseudo_stereo_centers", tuple(centers))
        assigned = _atom_cip_stereochemistry_from_props(mol)
    return assigned


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
    sulfur = mol_cache_get(mol, "rendered_sulfur_stereo", {})
    owned_sulfur = {idx for idx, text in sulfur if text in name}
    pieces = []
    unlocanted = []
    for atom_idx, descriptor in assigned_atom_stereochemistry(mol, Chem, atom_order=atom_order):
        if atom_idx in owned_sulfur:
            continue
        locant = locant_by_atom.get(atom_idx)
        if not locant:
            unlocanted.append(descriptor)
            continue
        text = f"{locant}{descriptor}"
        pieces.append((_stereo_locant_sort_key(locant), text))
    if len(pieces) <= 1 and existing:
        pieces = [piece for piece in pieces if piece[1] not in existing]
    if unlocanted:
        represented = _locanted_atom_stereo_descriptors_in_name(name) + _internal_atom_stereo_descriptors_in_name(name)
        if owned_sulfur:
            represented = _remove_represented_atom_descriptors(
                represented, [d for (idx, text), d in sulfur.items()
                              if idx in owned_sulfur and text in name])
        unlocanted = _remove_represented_atom_descriptors(unlocanted, represented)
    return pieces, unlocanted


def _record_rendered_bond_stereo(mol, rendered, bonds, unlabelled=None):
    units = mol_cache_dict(mol, "rendered_bond_stereo").setdefault(rendered, {})
    units[frozenset(bonds)] = unlabelled


def _remaining_bond_stereochemistry(mol, Chem, name, parent_locants=()):
    records = _indexed_bond_stereochemistry(mol, Chem)
    if not records:
        return records, name
    rendered = mol_cache_get(mol, "rendered_bond_stereo")
    kekulized = mol_cache_get(mol, "kekulized_polycyclic_view")
    # Kekulization retains atom and bond indices. Branches rendered on the
    # copy may already have described a bond from the input molecule.
    view_rendered = mol_cache_get(kekulized, "rendered_bond_stereo") if kekulized else None
    shared = _shared_unlocanted_bonds(mol, name)
    if kekulized:
        shared.update(_shared_unlocanted_bonds(kekulized, name))
    if not (rendered or view_rendered or shared):
        return records, name
    owned = set()
    candidates = {}
    for claims in (rendered, view_rendered):
        if not claims:
            continue
        for prefix, units in claims.items():
            if prefix in name:
                candidates.setdefault(prefix, {}).update(units)
    remaining_name = name
    # An enclosing stereo header can contain another registered branch.
    # Resolve the enclosing claim first so its bond cannot be claimed again
    # in a shorter, repeated branch.
    for prefix in sorted(candidates, key=len, reverse=True):
        units = candidates[prefix]
        # Candidate exploration can render an identical branch from several
        # parts of the graph. The selected parent owns its mapped bonds.
        if parent_locants:
            units = {bonds: unlabelled for bonds, unlabelled in units.items()
                     if not any(a in parent_locants and b in parent_locants for a, b in bonds)}
        if owned:
            units = {bonds: unlabelled for bonds, unlabelled in units.items() if not bonds <= owned}
        if len(units) > 1 and _rendered_branch_count(name, prefix) < len(units):
            continue
        for bonds, unlabelled in units.items():
            owned.update(bonds)
            if unlabelled is not None:
                remaining_name = remaining_name.replace(prefix, unlabelled)
    owned.update((a, b) for a, b in shared
                 if a not in parent_locants or b not in parent_locants)
    return [(a, b, label) for a, b, label in records if (a, b) not in owned and (b, a) not in owned], remaining_name


def _rendered_branch_count(name, prefix):
    total = 0
    for match in re.finditer(re.escape(prefix), name):
        scopes = []
        for token in _STEREO_SCOPE_DELIMITERS.finditer(name, 0, match.start()):
            if token['open']:
                scopes.append(_STEREO_MULTIPLIERS.get(token['mult'], 1))
            elif scopes:
                scopes.pop()
        total += prod(scopes)
    return total


def _shared_unlocanted_bonds(mol, name):
    claims = _shared_mol_cache_get(mol, "rendered_mapped_branch_bond_stereo")
    if not claims:
        return set()
    # Shared writers such as carbamimidoyl emit an unlocanted descriptor
    # inside the branch. Existing legacy matching handles locanted pieces.
    unlocanted = {(a, b) for a, b, prefix, _ in claims if prefix.startswith(("(E)-", "(Z)-"))}
    if not unlocanted:
        return unlocanted
    represented = _rendered_mapped_branch_bonds(mol, name)
    return {(a, b) for a, b in unlocanted if frozenset((a, b)) in represented}


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
        descriptor = terminal_haloalkene_descriptor(mol, Chem, bond, descriptor)
        records.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), descriptor))
    return mol_cache_set(mol, "indexed_bond_stereochemistry", records)


def _locanted_bond_stereo_pieces(
    records: list[tuple[int, int, str]],
    locant_by_atom: dict[int, str],
    name: str,
) -> tuple[list[tuple[tuple, str]], list[str]]:
    if not records:
        return [], []
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
        unlocanted = _remove_represented_atom_descriptors(unlocanted, _locanted_bond_stereo_descriptors_in_name(name))
    return pieces, unlocanted


def _stereo_locant_sort_key(locant: str) -> tuple[int, int, str]:
    match = re.match(r"^(\d+)([a-z]?)(\'?)$", locant)
    if not match:
        return (9999, 9999, locant.count("'"), locant)
    number, letter, prime = match.groups()
    letter_rank = ord(letter) - ord("a") + 1 if letter else 0
    return (int(number), letter_rank, len(prime), locant)


