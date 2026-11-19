"""Typed oxygen-isotope operations for characteristic-group suffixes.

The recognizer keeps original atom indices and role names separate from
rendering.  The renderer owns only isotopically modified oxygen atoms in the
selected suffix groups and isotopic hydrogens attached directly to those
oxygens.  Parent selection and skeletal isotope naming remain with callers.
"""

from dataclasses import dataclass

from .constants import MULT
from .formatting import attach_suffix, enclose_name_unit, multiplied_suffix

RULES = (
    "P-82.2.1", "P-82.2.2.2", "P-82.2.3", "P-82.6.1",
    "P-82.6.2", "P-82.6.3.1", "P-82.6.4", "P-63.1.2",
    "P-64.2.2", "P-65.1.2", "P-16.3.3", "P-16.5", "P-16.7",
)


@dataclass(frozen=True, slots=True)
class OxygenSuffixGroup:
    """One characteristic group with oxygen role-to-atom bindings."""

    kind: str
    center: int
    roles: tuple[tuple[str, int], ...]
    attached_hydrogens: tuple[int, ...] = ()

    @property
    def oxygen_atoms(self):
        return tuple(idx for role, idx in self.roles if role.endswith("oxygen"))


@dataclass(frozen=True, slots=True)
class OxygenPriority:
    """P-82.2.2.2 disposition for unlike isotope-modified groups."""

    suffix_groups: tuple[OxygenSuffixGroup, ...]
    prefix_groups: tuple[OxygenSuffixGroup, ...]
    tied: bool
    evidence: str = "P-82.2.2.2: modification count, then nuclide Z and mass"


def oxygen_group(mol, group):
    """Return an original-graph typed view, or ``None`` outside the grammar."""
    kind, center = getattr(group, "kind", ""), getattr(group, "center", -1)
    if center < 0 or center >= mol.GetNumAtoms():
        return None
    atom = mol.GetAtomWithIdx(center)
    roles = []
    if kind in {"alcohol", "phenol"}:
        candidates = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8
                      and mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 1
                      and n.GetTotalNumHs(includeNeighbors=True) == 1]
        if len(candidates) != 1:
            return None
        roles.append(("hydroxyl_oxygen", candidates[0].GetIdx()))
    elif kind in {"aldehyde", "ketone", "amide", "acid"}:
        oxygens = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8]
        double = [n for n in oxygens if mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 2]
        if len(double) != 1:
            return None
        roles.append(("carbonyl_oxygen", double[0].GetIdx()))
        if kind == "acid":
            single = [n for n in oxygens if mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 1]
            hydroxyl = [n for n in single if n.GetTotalNumHs(includeNeighbors=True) == 1]
            if len(hydroxyl) != 1:
                return None
            roles.append(("acid_hydroxyl_oxygen", hydroxyl[0].GetIdx()))
        elif kind == "amide":
            # P-82 gives no direct amide carbonyl-oxygen suffix example.
            # Keep the role visible for future integration but do not render it.
            pass
    else:
        return None
    oxygen_ids = tuple(i for _, i in roles)
    isotope_h = tuple(sorted(n.GetIdx() for i in oxygen_ids
                             for n in mol.GetAtomWithIdx(i).GetNeighbors()
                             if n.GetAtomicNum() == 1 and n.GetIsotope()))
    return OxygenSuffixGroup(kind, center, tuple(roles), isotope_h)


def _isotope_signature(mol, item):
    masses = []
    for role, idx in item.roles:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetIsotope():
            masses.append((role, atom.GetAtomicNum(), atom.GetIsotope()))
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope():
                masses.append((role + "_hydrogen", 1, neighbor.GetIsotope()))
    return tuple(sorted(masses, reverse=True))


def prioritize_oxygen_groups(mol, groups):
    """Choose suffix-eligible group types by P-82.2.2.2 evidence.

    Identically modified groups remain together for multiplication.  Among
    unlike groups, groups with more modified atoms win; ties use the highest
    nuclide atomic number and then mass.  Unmodified groups lose to modified
    groups.  Atom order never participates in the comparison.
    """
    typed = tuple(oxygen_group(mol, g) for g in groups)
    if not typed or any(x is None for x in typed):
        return OxygenPriority((), tuple(x for x in typed if x), False, "unproved group role")
    from .characteristic_isotope_selection import (
        CharacteristicGroupMap, select_characteristic_isotope_frontier,
    )
    role_maps = []
    for item in typed:
        roles = tuple((role, (idx,)) for role, idx in item.roles)
        roles += (("attached_isotope_hydrogen", tuple(item.attached_hydrogens)),)
        atoms = frozenset(idx for _role, ids in roles for idx in ids)
        role_maps.append(CharacteristicGroupMap(
            f"{item.kind}:{item.center}", atoms, roles,
        ))
    selection = select_characteristic_isotope_frontier(
        mol, tuple(role_maps), frozenset().union(*(x.atoms for x in role_maps)),
    )
    if selection is None:
        return OxygenPriority((), typed, False, "unproved isotope-role frontier")
    if selection.barrier or len(selection.principal_frontier) != 1:
        return OxygenPriority((), typed, False, selection.criterion)
    cluster = selection.principal_frontier[0]
    winning_indices = selection.multiplicity_groups[cluster]
    winning_set = frozenset(winning_indices)
    winners = tuple(item for index, item in enumerate(typed) if index in winning_set)
    losers = tuple(item for index, item in enumerate(typed) if index not in winning_set)
    return OxygenPriority(winners, losers, len(winners) > 1, selection.criterion)


def _descriptor(mol, item):
    """Descriptor text local to a suffix unit; ``None`` means declined."""
    if item.kind == "amide":
        return None
    if item.kind == "acid" and item.attached_hydrogens:
        # Acid-hydroxyl deuterium uses an oxygen locant (P-82.2.4). Its
        # placement within systematic suffix forms is left to a typed prefix
        # integration until that exact boundary is implemented.
        return None
    entries = []
    for idx in item.attached_hydrogens:
        atom = mol.GetAtomWithIdx(idx)
        entries.append((1, atom.GetIsotope(), "H", 1))
    oxygen = [(role, idx) for role, idx in item.roles]
    isotope_oxygens = [(role, mol.GetAtomWithIdx(idx).GetIsotope()) for role, idx in oxygen
                       if mol.GetAtomWithIdx(idx).GetIsotope()]
    if item.kind == "acid":
        # P-82.6.4 demonstrates 18O1 in an ester anion. That does not prove
        # the same count for every systematic acid suffix; retained-acid
        # forms are handled by retained_acid_isotope_parts with explicit roles.
        if isotope_oxygens:
            return None
    else:
        entries.extend((8, mass, "O", 1) for _, mass in isotope_oxygens)
    if not entries:
        return ""
    # For repeated same-element nuclides at one named position, count them.
    merged = {}
    for z, mass, symbol, count in entries:
        merged[(z, mass, symbol)] = merged.get((z, mass, symbol), 0) + count
    parts = []
    for (z, mass, symbol), count in sorted(merged.items(), key=lambda x: (x[0][2], x[0][1])):
        parts.append(f"{mass}{symbol}" + (str(count) if count > 1 or item.kind == "acid" else ""))
    return "(" + ",".join(parts) + ")"


def oxygen_ending(mol, Chem, parent, groups, locants=(), *, suffix=None):
    """Render a proven oxygen-modified suffix and return its owned atom IDs.

    Supported suffix bases are ``ol``, ``one``, ``al``, ``oic acid`` and
    ``carboxylic acid``. For isotope-unequal groups this function declines;
    callers first use :func:`prioritize_oxygen_groups`, send its suffix group
    selection here, and name its prefix groups separately.
    """
    del Chem  # Kept in the project helper signature for call-site symmetry.
    groups = tuple(groups)
    if not groups or not parent:
        return "", frozenset()
    base = suffix or {"alcohol": "ol", "phenol": "ol", "aldehyde": "al",
                      "ketone": "one", "acid": "oic acid"}.get(groups[0].kind)
    if not base or any(getattr(g, "kind", None) != groups[0].kind for g in groups):
        return "", frozenset()
    if base not in {"ol", "one", "al", "oic acid", "carboxylic acid"}:
        return "", frozenset()
    typed = tuple(oxygen_group(mol, g) for g in groups)
    if any(item is None or item.kind == "amide" for item in typed):
        return "", frozenset()
    descriptors = tuple(_descriptor(mol, item) for item in typed)
    if any(x is None for x in descriptors) or len(set(descriptors)) > 1:
        return "", frozenset()
    decorated_multiple = len(groups) > 1 and bool(descriptors[0])
    if decorated_multiple:
        # P-82.2.2 allows identical isotope-modified units to combine, and
        # P-82.2.2.2 demonstrates the enclosing form with
        # ``di[(14C)carboxylic acid]``. Apply that component pattern to the
        # one-oxygen ``ol``, ``one``, and ``al`` units only. Acid O-only
        # descriptors remain declined above because their count grammar is
        # not established by the ester example.
        if base not in {"ol", "one", "al"}:
            return "", frozenset()
        if len(set(_isotope_signature(mol, item) for item in typed)) != 1:
            return "", frozenset()
        multiplier = MULT.get(len(groups))
        if not multiplier:
            return "", frozenset()
        unit = enclose_name_unit(descriptors[0] + base)
        suffix_name = multiplier + unit
    else:
        suffix_name = multiplied_suffix(len(groups), base)
    name = attach_suffix(parent, suffix_name, locants)
    descriptor = descriptors[0]
    if descriptor and not decorated_multiple:
        # The descriptor precedes the suffix word; locant hyphens stay before
        # the parenthesized isotope descriptor (cyclopentan-1-(18O)ol).
        token = base
        pos = name.rfind(token)
        if pos < 0:
            return "", frozenset()
        name = name[:pos] + descriptor + name[pos:]
    owned = frozenset(idx for item in typed for idx in item.oxygen_atoms + item.attached_hydrogens
                      if mol.GetAtomWithIdx(idx).GetIsotope())
    return name, owned


def _retained_acid_parts_result(mol, group):
    from .acid_oxygen_isotope_methods import retained_acid_oxygen_record

    record = retained_acid_oxygen_record(mol, group)
    if record is None:
        return (), frozenset(), False
    return record.parts, record.owned_atom_ids, True


def retained_acid_isotope_parts(mol, group):
    """Return typed entries for directly shown retained-acid descriptor parts.

    Entries have ``(symbol, mass, component_locant, omit_single_count)``
    shape, compatible with ``MappedComponent.isotope_descriptor_parts``.
    Examples return ``(("H", 2, "O", True), ("O", 18, "", False))`` and
    ``(("H", 2, "18O", True), ("O", 18, "", False))``. The original atom
    IDs are returned separately. This is boundary metadata: no completed name
    is parsed or edited.
    """
    parts, owned, valid = _retained_acid_parts_result(mol, group)
    return (parts, owned) if valid else ((), frozenset())


def retained_acid_base(mol, base_name, group):
    """Decorate a completed retained-acid base component from original roles.

    ``base_name`` is the selected acid component (for example,
    ``"acetic acid"``), not a whole molecule name. It returns that component
    with its retained-acid isotope descriptor plus the isotope atom IDs owned
    there. Parent-skeleton labels remain separately mapped.
    """
    if not base_name or getattr(group, "kind", None) != "acid":
        return "", frozenset()
    parts, owned, valid = _retained_acid_parts_result(mol, group)
    if not valid:
        return "", frozenset()
    from .acid_oxygen_isotope_methods import (
        retained_acid_component_text, retained_acid_oxygen_record,
    )

    record = retained_acid_oxygen_record(mol, group)
    return retained_acid_component_text(base_name, record), owned


def systematic_acid_suffix(mol, base_name, group):
    """Place an isotopic acid hydroxyl hydrogen in a systematic acid base.

    P-82.2.4 shows ``pentan(3H)oic acid``. The nuclide mass is generic under
    P-82.3; carboxyl-oxygen isotope suffix forms remain outside this grammar.
    """
    if not base_name.endswith("oic acid") or getattr(group, "kind", None) != "acid":
        return "", frozenset()
    from .acid_oxygen_isotope_methods import systematic_acid_hydrogen_record

    record = systematic_acid_hydrogen_record(mol, group)
    if record is None:
        return "", frozenset()
    mass = record.parts[0][1]
    token = f"({mass}H)"
    boundary = "oic acid"
    pos = base_name.rfind(boundary)
    if pos < 0:
        return "", frozenset()
    return base_name[:pos] + token + base_name[pos:], record.owned_atom_ids

