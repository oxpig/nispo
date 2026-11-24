"""Bounded annotated P-46 selection for saturated acyclic carbon prefixes.

This producer starts with the chemical P-46.1 frontier on a constitutional
copy, then applies only the isotope and tetrahedral-R comparisons explicitly
proved here.  Atom IDs, isotope labels, and original CIP assignments remain
from the input molecule; the constitutional copy is used only to enumerate
paths.  Unsupported annotations or unresolved comparisons decline.

Sources: IUPAC Blue Book P-46.2.1/.2 and P-46.3.2; P-82.2.1, P-82.5.1/.2,
P-82.6.3.2; P-14.4(i)/(j); P-29.2, P-29.3.2.2, and P-29.6.1.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnnotatedCandidate:
    method: int
    path_atom_ids: tuple[int, ...]
    locants: tuple[tuple[int, str], ...]
    chain_modified_atoms: int
    nuclide_profile: tuple[tuple[int, int], ...]
    modified_side_group_count: int
    side_nuclide_profile: tuple[tuple[int, int], ...]
    r_centers: int
    isotope_locants: tuple[int, ...]
    isotope_seniority: tuple
    stereo_numbering: tuple


def _scope_is_admitted(mol, root, attachment, atoms):
    from rdkit import Chem

    atoms = frozenset(atoms)
    if (root not in atoms or attachment in atoms
            or mol.GetBondBetweenAtoms(root, attachment) is None
            or mol.GetBondBetweenAtoms(root, attachment).GetBondTypeAsDouble() != 1):
        return None
    carbon_scope = frozenset(i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 6)
    if root not in carbon_scope:
        return None
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 1:
            # collect_subtree deliberately keeps isotopic hydrogen. Keep it
            # in ownership, but use the carbon skeleton for P-46 path work.
            if atom.GetIsotope() not in {2, 3} or atom.GetDegree() != 1:
                return None
            parent = atom.GetNeighbors()[0]
            if parent.GetAtomicNum() != 6 or parent.GetIdx() not in carbon_scope:
                return None
            continue
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != 0
                or atom.GetNumRadicalElectrons() or atom.IsInRing()):
            return None
        if atom.GetTotalValence() != 4:
            return None
        if any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds()):
            return None
        for neighbor in atom.GetNeighbors():
            n = neighbor.GetIdx()
            if neighbor.GetAtomicNum() == 1:
                if neighbor.GetIsotope() not in {0, 2, 3}:
                    return None
            elif n in carbon_scope:
                continue
            elif idx == root and n == attachment:
                continue
            else:
                return None
    edges = sum(1 for idx in carbon_scope for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                if n.GetIdx() in carbon_scope and idx < n.GetIdx())
    if edges != len(carbon_scope) - 1:
        return None
    # Only specified tetrahedral carbon configurations are in scope. In
    # particular, do not turn an unspecified potentially stereogenic center
    # into an assumed S/non-R descriptor.
    for feature in Chem.FindPotentialStereo(mol):
        if feature.type == Chem.StereoType.Atom_Tetrahedral and feature.centeredOn in carbon_scope:
            atom = mol.GetAtomWithIdx(feature.centeredOn)
            if atom.GetAtomicNum() != 6 or atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                return None
        elif feature.type == Chem.StereoType.Bond_Double:
            bond = mol.GetBondWithIdx(feature.centeredOn)
            if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & carbon_scope:
                return None
        elif feature.type not in {Chem.StereoType.Atom_Tetrahedral, Chem.StereoType.Bond_Double}:
            centers = set(getattr(feature, "controllingAtoms", ()))
            if feature.centeredOn in carbon_scope or centers & carbon_scope:
                return None
    # Explicit isotope labels in this grammar are limited to C, D, and T.
    if any(mol.GetAtomWithIdx(i).GetIsotope()
           and mol.GetAtomWithIdx(i).GetAtomicNum() not in {1, 6} for i in atoms):
        return None
    return atoms


def _constitutional_copy(mol):
    from rdkit import Chem
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetIsotope(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in copy.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    Chem.AssignStereochemistry(copy, cleanIt=True, force=True)
    return copy


def _isotopically_modified_atom_ids(mol, atoms):
    labels = set()
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetIsotope():
            labels.add(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope():
                labels.add(neighbor.GetIdx())
    return labels


def _candidate(mol, path, scope, carbon_scope, method, assigned):
    path = tuple(path)
    path_set = frozenset(path)
    modified = _isotopically_modified_atom_ids(mol, path_set)
    side_carbons = carbon_scope - path_set
    side_modified = _isotopically_modified_atom_ids(mol, side_carbons)
    path_profile = Counter(mol.GetAtomWithIdx(idx).GetIsotope() for idx in modified)
    side_profile = Counter(mol.GetAtomWithIdx(idx).GetIsotope() for idx in side_modified)
    # Distinct isotope-labelled side components are groups under P-46.2.2.
    remaining = set(side_carbons)
    modified_side_groups = 0
    while remaining:
        seed = next(iter(remaining))
        group, stack = set(), [seed]
        while stack:
            idx = stack.pop()
            if idx not in remaining:
                continue
            remaining.remove(idx)
            group.add(idx)
            stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                         if n.GetIdx() in remaining and n.GetAtomicNum() == 6)
        group_labels = _isotopically_modified_atom_ids(mol, group)
        modified_side_groups += bool(group_labels)
    descriptors = {idx: text for idx, text in assigned.items() if idx in path_set}
    # P-46.3.2 names R centers specifically. Pseudoasymmetric r/s are not
    # silently counted as R; their numbering interactions are not implemented.
    locants = tuple((idx, str(pos)) for pos, idx in enumerate(path, 1))
    isotope_group_locants = set()
    isotope_labels = []
    for idx in modified:
        host = idx if idx in path_set else next((n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                                                 if n.GetIdx() in path_set), None)
        if host is not None:
            loc = path.index(host) + 1
            isotope_group_locants.add(loc)
            atom = mol.GetAtomWithIdx(idx)
            isotope_labels.append((loc, atom.GetAtomicNum(), atom.GetIsotope()))
    isotope_locants = tuple(sorted(isotope_group_locants))
    isotope_seniority = tuple(sorted(
        (loc, -z, -mass) for loc, z, mass in isotope_labels
    ))
    descriptor_rank = {"R": 0, "r": 1, "S": 2, "s": 3}
    if any(value not in descriptor_rank for value in descriptors.values()):
        return None
    stereo_numbering = tuple(sorted(
        (descriptor_rank[d], path.index(i) + 1)
        for i, d in descriptors.items()
    ))
    return AnnotatedCandidate(
        method, path, locants, len(modified), tuple(sorted(path_profile.items(), reverse=True)),
        modified_side_groups, tuple(sorted(side_profile.items(), reverse=True)),
        sum(d == "R" for d in descriptors.values()),
        isotope_locants, tuple(sorted(isotope_labels)), stereo_numbering)


def _rank(candidates):
    """Apply P-46.2/.3 to an already established P-46.1 frontier.

    P-46.2.1 compares modified atoms on the principal chain. P-46.2.2 then
    compares heavier nuclides and modified side groups. The source does not
    provide a complete tie algorithm for arbitrary mixed isotope groups, so
    those ties are retained as an explicit decline. P-46.3.2 counts R centers.
    """
    candidates = tuple(candidates)
    if not candidates:
        return ()
    if len(candidates) == 1:
        return candidates
    # Each criterion is applied only after the previous one ties.
    best_count = max(c.chain_modified_atoms for c in candidates)
    candidates = tuple(c for c in candidates if c.chain_modified_atoms == best_count)
    if len(candidates) == 1:
        return candidates
    profiles = {c.nuclide_profile for c in candidates}
    side_counts = {c.modified_side_group_count for c in candidates}
    if len(profiles) > 1 and len(side_counts) > 1:
        # P-46.2.2 supplies both isotope-mass and modified-substituent
        # evidence, but this producer does not generalize their tradeoff.
        max_profile = max(profiles)
        max_groups = max(side_counts)
        isotope_winner = {c for c in candidates
                          if c.nuclide_profile == max_profile}
        group_winner = {c for c in candidates
                        if c.modified_side_group_count == max_groups}
        if isotope_winner != group_winner:
            return ()
        candidates = tuple(isotope_winner)
    elif len(profiles) > 1:
        best = max(profiles)
        candidates = tuple(c for c in candidates if c.nuclide_profile == best)
    elif len(side_counts) > 1:
        best = max(side_counts)
        candidates = tuple(c for c in candidates if c.modified_side_group_count == best)
    if len(candidates) == 1:
        return candidates
    # P-46.3.2 counts R descriptors only, after the isotope criterion.
    best_r = max(c.r_centers for c in candidates)
    candidates = tuple(c for c in candidates if c.r_centers == best_r)
    if len(candidates) == 1:
        return candidates
    # P-82.5.2: use the lowest combined locant set, then give lower locants
    # to nuclides of higher atomic number.
    best_locants = min(c.isotope_locants for c in candidates)
    candidates = tuple(c for c in candidates if c.isotope_locants == best_locants)
    if len(candidates) == 1:
        return candidates
    best_atomic_numbers = min(
        tuple((loc, neg_z) for loc, neg_z, _mass in c.isotope_seniority)
        for c in candidates
    )
    candidates = tuple(c for c in candidates
                       if tuple((loc, neg_z) for loc, neg_z, _mass in c.isotope_seniority)
                       == best_atomic_numbers)
    if len(candidates) == 1:
        return candidates
    # P-14.4(i): for different nuclides of the same element, prefer the
    # higher mass number at the lower locant. This remains ahead of (j).
    best_masses = min(
        tuple((loc, neg_mass) for loc, _neg_z, neg_mass in c.isotope_seniority)
        for c in candidates
    )
    candidates = tuple(c for c in candidates
                       if tuple((loc, neg_mass) for loc, _neg_z, neg_mass in c.isotope_seniority)
                       == best_masses)
    if len(candidates) == 1:
        return candidates
    # P-14.4(j) then prefers lower locants for R over r over S over s.
    best_stereo = min(c.stereo_numbering for c in candidates)
    candidates = tuple(c for c in candidates if c.stereo_numbering == best_stereo)
    # Symmetry can leave multiple atom paths tied on every source criterion.
    # Preserve that equivalence frontier; the mapped writer below accepts it
    # only when all paths produce the same rendered prefix.
    return candidates


def _annotation_frontier(mol, Chem, root, attachment, atoms, method):
    from .stereochemistry import assigned_atom_stereochemistry
    from .substituent_selection import (
        select_rooted_substituent_chain, select_alkanyl_substituent_path,
    )
    scope = _scope_is_admitted(mol, root, attachment, atoms)
    if scope is None or len(scope) > 64:
        return None
    carbon_scope = frozenset(i for i in scope if mol.GetAtomWithIdx(i).GetAtomicNum() == 6)
    clean = _constitutional_copy(mol)
    if method == 1:
        base = select_rooted_substituent_chain(clean, root, attachment)
    else:
        base = select_alkanyl_substituent_path(clean, root, attachment)
    if not base:
        return None
    assigned = dict(assigned_atom_stereochemistry(mol, Chem, atom_order="idx", include_pseudo=True))
    tagged = {i for i in carbon_scope if int(mol.GetAtomWithIdx(i).GetChiralTag())}
    if tagged - assigned.keys():
        # Some isotope-induced or otherwise complex centers are not assigned
        # by the current whole-graph CIP backend. Do not cap/relabel them or
        # emit an isotope name that drops their original configuration.
        return None
    candidates = []
    for item in base:
        candidate = _candidate(mol, item.path_atom_ids, scope, carbon_scope, method, assigned)
        if candidate is None:
            return None
        candidates.append(candidate)
    selected = _rank(candidates)
    return selected or None


def annotated_chain_frontier(mol, Chem, root, attachment, atoms):
    """Return selected source-ordered chain candidates for independent review.

    This function does not render or rank complete PINs. ``None`` is an
    explicit scope/criterion decline. Returned atom IDs refer to ``mol``.
    """
    scope = _scope_is_admitted(mol, root, attachment, atoms)
    if scope is None:
        return None
    carbon_scope = frozenset(i for i in scope if mol.GetAtomWithIdx(i).GetAtomicNum() == 6)
    degree = sum(n.GetIdx() in carbon_scope for n in mol.GetAtomWithIdx(root).GetNeighbors())
    method = 1 if degree <= 1 else 2
    return _annotation_frontier(mol, Chem, root, attachment, scope, method)


def select_annotated_saturated_prefix(mol, Chem, root, attachment, atoms):
    """Return ``(prefix, owned_atoms)`` or ``None``/empty-name decline.

    ``None`` means the graph is outside this producer's grammar. ``('', frozenset())``
    means it fits the shape but contains an unresolved selection or spelling
    fact. Output atom ownership is exactly the original heavy-atom branch set.
    """
    from .component_mapping import (MappedComponent, decorate_component,
                                    hydrogen_bindings, mapped_branch)
    from .constants import ROOTS

    scope = _scope_is_admitted(mol, root, attachment, atoms)
    if scope is None:
        return None
    carbon_scope = frozenset(i for i in scope if mol.GetAtomWithIdx(i).GetAtomicNum() == 6)
    root_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(root).GetNeighbors()
                      if n.GetIdx() in carbon_scope]
    method = 1 if len(root_neighbors) <= 1 else 2
    frontier = _annotation_frontier(mol, Chem, root, attachment, scope, method)
    if frontier is None:
        return "", frozenset()
    names = set()
    for candidate in frontier:
        path = candidate.path_atom_ids
        if len(path) not in ROOTS:
            return "", frozenset()
        mapping = {idx: str(pos) for pos, idx in enumerate(path, 1)}
        path_set = frozenset(path)
        prefixes, embedded = [], set()
        remaining = set(carbon_scope - path_set)
        while remaining:
            seed = min(remaining)
            group, stack = set(), [seed]
            boundary = set()
            while stack:
                idx = stack.pop()
                if idx not in remaining:
                    continue
                remaining.remove(idx)
                group.add(idx)
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                    other = neighbor.GetIdx()
                    if other in path_set:
                        boundary.add(other)
                    elif other in remaining:
                        stack.append(other)
            if len(boundary) != 1:
                return "", frozenset()
            host = next(iter(boundary))
            member = next((i for i in group if any(n.GetIdx() == host
                                                   for n in mol.GetAtomWithIdx(i).GetNeighbors())), None)
            if member is None:
                return "", frozenset()
            text, owned = mapped_branch(mol, Chem, member, host)
            group_scope = hydrogen_bindings(mol, dict.fromkeys(group, ''))
            if not text or set(owned) != set(group_scope) or embedded & set(owned):
                return "", frozenset()
            embedded.update(owned)
            prefixes.append((mapping[host], text))
        bindings = hydrogen_bindings(mol, mapping)
        if set(bindings) | embedded != set(scope):
            return "", frozenset()
        parent = (ROOTS[len(path)] + "yl" if method == 1 else
                  ROOTS[len(path)] + "an-" + mapping[root] + "-yl")
        text = decorate_component(mol, Chem, MappedComponent(
            parent, tuple(bindings.items()), frozenset(bindings), tuple(prefixes),
            frozenset(embedded), systematic_parent=True))
        if not text:
            return "", frozenset()
        names.add(text)
    return (names.pop(), frozenset(scope)) if len(names) == 1 else ("", frozenset())
