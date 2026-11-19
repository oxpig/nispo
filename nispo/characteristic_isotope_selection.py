"""Typed P-82.2.2.2 selection for caller-mapped characteristic groups.

This module does not identify a functional-group class, choose a parent, or
render a name. The caller supplies one already-selected constitutional class
as disjoint original-atom group maps with complete site-role ownership.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CharacteristicGroupMap:
    """Original graph atom ownership for one characteristic group.

    ``atoms`` is the caller-declared complete atom set for the group, including
    explicit isotope-bearing hydrogens when those belong to the group.
    ``roles`` partitions that set into named structural sites, for example
    ``(("carbonyl_carbon", (i,)), ("carbonyl_oxygen", (j,)))``. Role names
    are labels supplied by a trusted graph recognizer; this module never
    infers a class from them.
    """

    group_id: str
    atoms: frozenset[int]
    roles: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True, slots=True)
class CharacteristicIsotopeSelection:
    """Multiplicative profile partition and source-supported suffix frontier.

    ``principal_frontier`` indexes ``multiplicity_groups`` rather than atom
    indices. A multi-cluster frontier is an explicit barrier: the caller must
    not choose one by text, atom order, or another unsupported tiebreak.
    This result is a characteristic-group comparison only, not a naming
    decision.
    """

    multiplicity_groups: tuple[tuple[int, ...], ...]
    principal_frontier: tuple[int, ...]
    modified_counts: tuple[int, ...]
    isotope_role_profiles: tuple[tuple[tuple[str, tuple[tuple[int, int], ...]], ...], ...]
    criterion: str
    barrier: bool


def select_characteristic_isotope_frontier(
    mol,
    groups,
    scope_atoms,
) -> CharacteristicIsotopeSelection | None:
    """Compare isotope profiles for a preselected, single characteristic class.

    The P-82.2.2.2 comparison is limited to the directly supported steps:

    * identical role-specific isotope profiles form one multiplicative unit;
    * a greater number of modified atoms wins;
    * for equal-count profiles whose modifications are all the same nuclide
      identity within each competing unit, higher atomic number and then
      higher mass number wins.

    Equal nuclide ranks with different role profiles remain a frontier.
    Equal-count profiles containing multiple nuclide identities also remain a
    frontier because this consumer has not proved a comparison between
    multi-nuclide profiles. Invalid or incomplete original-atom maps return
    ``None``.

    ``scope_atoms`` must enumerate exactly the original atoms assigned to the
    supplied characteristic groups. It should be assembled by the caller's
    selected-class recognizer, not from a name string. Explicit isotope H
    contributes to a count only when it is included in the group map and is
    bonded to another atom owned by that group.
    """
    try:
        groups = tuple(groups)
    except TypeError:
        return None
    if not groups:
        return None
    try:
        scope = frozenset(scope_atoms)
    except TypeError:
        return None
    if not scope or any(type(idx) is not int or idx < 0 or idx >= mol.GetNumAtoms() for idx in scope):
        return None

    seen_ids = set()
    seen_atoms: set[int] = set()
    role_schema: tuple[str, ...] | None = None
    role_heavy_atom_shape: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] | None = None
    signatures = []
    for group in groups:
        if not isinstance(group, CharacteristicGroupMap):
            return None
        if not isinstance(group.group_id, str) or not group.group_id or group.group_id in seen_ids:
            return None
        seen_ids.add(group.group_id)
        try:
            atoms = frozenset(group.atoms)
        except TypeError:
            return None
        if not atoms or any(type(idx) is not int or idx < 0 or idx >= mol.GetNumAtoms() for idx in atoms):
            return None
        if seen_atoms & atoms:
            return None
        seen_atoms.update(atoms)

        role_map = {}
        assigned: set[int] = set()
        try:
            role_entries = tuple(group.roles)
        except TypeError:
            return None
        for entry in role_entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                return None
            role, atom_indices = entry
            if not isinstance(role, str) or not role or role in role_map:
                return None
            try:
                indices = tuple(atom_indices)
            except TypeError:
                return None
            if (any(type(idx) is not int or idx < 0 or idx >= mol.GetNumAtoms()
                                   for idx in indices)
                    or len(set(indices)) != len(indices)):
                return None
            role_atoms = set(indices)
            if assigned & role_atoms:
                return None
            assigned.update(role_atoms)
            role_map[role] = tuple(indices)
        if assigned != atoms:
            return None
        schema = tuple(sorted(role_map))
        if role_schema is None:
            role_schema = schema
        elif schema != role_schema:
            return None

        # Role names alone are not sufficient evidence that two mappings
        # describe comparable sites. Require the same heavy-element/count
        # shape for each role while permitting isotope-bearing H atoms to be
        # explicit in one group and implicit/natural in another.
        heavy_shape = tuple(
            (role, tuple(sorted(
                (int(mol.GetAtomWithIdx(idx).GetAtomicNum()), 1)
                for idx in role_map[role]
                if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1
            )))
            for role in schema
        )
        if role_heavy_atom_shape is None:
            role_heavy_atom_shape = heavy_shape
        elif heavy_shape != role_heavy_atom_shape:
            return None

        role_profile = []
        for role in schema:
            labels = []
            for idx in role_map[role]:
                atom = mol.GetAtomWithIdx(idx)
                mass = int(atom.GetIsotope())
                if mass:
                    if atom.GetAtomicNum() <= 0:
                        return None
                    labels.append((int(atom.GetAtomicNum()), mass))
            role_profile.append((role, tuple(sorted(labels))))

        # Explicit isotope-bearing hydrogen atoms must be directly connected
        # to an atom owned by this group; a caller cannot count a remote H.
        for idx in atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 1 and atom.GetIsotope():
                neighbors = tuple(atom.GetNeighbors())
                if len(neighbors) != 1 or neighbors[0].GetIdx() not in atoms:
                    return None
            # A deuterium/tritium directly attached to a group-owned heavy
            # atom cannot be silently omitted from both the group and scope.
            if atom.GetAtomicNum() > 1:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope():
                        if neighbor.GetIdx() not in atoms:
                            return None
        signatures.append(tuple(role_profile))

    if seen_atoms != scope:
        return None

    multiplicity_groups = []
    signature_indices = {}
    for group_index, signature in enumerate(signatures):
        if signature not in signature_indices:
            signature_indices[signature] = len(multiplicity_groups)
            multiplicity_groups.append([])
        multiplicity_groups[signature_indices[signature]].append(group_index)
    multiplicity_groups = tuple(tuple(indices) for indices in multiplicity_groups)
    profiles = tuple(signatures[indices[0]] for indices in multiplicity_groups)
    counts = tuple(sum(len(labels) for _role, labels in profile) for profile in profiles)

    if len(multiplicity_groups) == 1:
        return CharacteristicIsotopeSelection(
            multiplicity_groups,
            (0,),
            counts,
            profiles,
            "single-profile",
            False,
        )

    greatest_count = max(counts)
    candidate_clusters = tuple(i for i, count in enumerate(counts) if count == greatest_count)
    if len(candidate_clusters) == 1:
        return CharacteristicIsotopeSelection(
            multiplicity_groups,
            candidate_clusters,
            counts,
            profiles,
            "modification-count",
            False,
        )

    # P-82.2.2.2 directly ranks the nuclide by atomic number and then mass.
    # Apply that rank when every modified atom in each candidate unit is the
    # same nuclide identity (including repeated atoms of that identity). A
    # mixed-nuclide vector remains an unproved comparison in this consumer.
    ranks = {}
    for cluster_index in candidate_clusters:
        labels = [label for _role, role_labels in profiles[cluster_index]
                  for label in role_labels]
        identities = set(labels)
        if not labels or len(identities) != 1:
            return CharacteristicIsotopeSelection(
                multiplicity_groups,
                candidate_clusters,
                counts,
                profiles,
                "unresolved-isotope-profile",
                True,
            )
        ranks[cluster_index] = next(iter(identities))
    greatest_rank = max(ranks.values())
    winners = tuple(i for i in candidate_clusters if ranks[i] == greatest_rank)
    criterion = "atomic-number" if len({ranks[i][0] for i in candidate_clusters}) > 1 else "mass-number"
    if len(winners) > 1:
        criterion = "equal-nuclide-rank"
    return CharacteristicIsotopeSelection(
        multiplicity_groups,
        winners,
        counts,
        profiles,
        criterion,
        len(winners) > 1,
    )
