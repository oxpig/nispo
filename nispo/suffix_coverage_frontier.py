"""Graph-only suffix-coverage frontier for ordinary carbon trees.

The frontier separates P-44.1.1 suffix coverage from rendering. It is limited
to complete same-class acid, amide, and nitrile inventories on neutral,
annotation-free carbon trees. Acid and amide trees may contain ordinary
single, double, and triple C-C bonds. Nitrile external parents remain limited
to saturated chains under their source rule. Candidate rows retain original
group objects so callers can map selected and omitted sites without reducing
them to counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from .molecule_iteration import iter_atoms, iter_bonds


_USE_TREE_PATH_DP = True


class SuffixCoverageBarrier(str, Enum):
    """Typed reasons an otherwise established graph obligation is incomplete."""

    ATTACHMENT_COMPLETENESS = "attachment_completeness"
    SOURCE_ROLE_COMPLETENESS = "source_role_completeness"


@dataclass(frozen=True, slots=True)
class ExcludedJuniorNitrileRole:
    """A source-inventoried subordinate cyano unit outside the parent tree."""

    center: int
    nitrogen: int
    host: int
    group_identity: tuple
    source_rule: str = "P-41/P-43/P-66.5.1.1.4"


@dataclass(frozen=True, slots=True)
class SuffixCoverageCandidate:
    path: tuple[int, ...]
    mode: str
    selected_groups: tuple[object, ...]
    omitted_groups: tuple[object, ...]
    multiple_bond_count: int = 0
    double_bond_count: int = 0
    source_rule: str = ""


@dataclass(frozen=True, slots=True)
class SuffixCoverageFrontier:
    status: str
    candidates: tuple[SuffixCoverageCandidate, ...] = ()
    maximum: int | None = None
    reason: str = ""
    rule: str = "P-44.1.1"
    maximum_length: int | None = None
    maximum_multiple_bonds: int | None = None
    maximum_double_bonds: int | None = None
    barrier: SuffixCoverageBarrier | None = None
    excluded_prefix_roles: tuple[ExcludedJuniorNitrileRole, ...] = ()


def maximum_suffix_coverage(mol, Chem, groups, carbon_atoms, *,
                            excluded_skeletal_atoms=()) -> SuffixCoverageFrontier:
    """Return every graph maximum across embedded and external suffix modes.

    The proof is intentionally narrow. Rendering and prefix ownership remain
    the caller's responsibility; an in-scope graph maximum must therefore be
    retained even if its later writer fails.
    """
    groups = tuple(groups)
    supplied_carbon = frozenset(int(i) for i in carbon_atoms)
    all_carbon = frozenset(atom.GetIdx() for atom in iter_atoms(mol)
                           if atom.GetAtomicNum() == 6)
    if not groups or groups[0].kind not in {"acid", "amide", "nitrile"}:
        return _outside("Only acid, amide, and nitrile suffix classes are covered.")
    kind = groups[0].kind
    if any(group.kind != kind for group in groups):
        return _outside("The principal sites do not belong to one suffix class.")
    if any(atom.GetFormalCharge() or atom.GetIsotope() or int(atom.GetChiralTag())
           or atom.GetNumRadicalElectrons() for atom in iter_atoms(mol)):
        return _outside("Charged, isotopically modified, radical, or stereochemical input is outside this frontier.")
    if any(int(bond.GetStereo()) or int(bond.GetBondDir()) for bond in iter_bonds(mol)):
        return _outside("Bond stereochemistry is outside this frontier.")
    if any(atom.IsInRing() for atom in iter_atoms(mol)):
        return _outside("Cyclic input is outside the acyclic carbon-tree frontier.")
    if not supplied_carbon or any(
            mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in supplied_carbon):
        return _outside("The proposed parent atom set is not all carbon.")
    if not supplied_carbon <= all_carbon:
        return _outside("The proposed parent set includes a non-carbon source atom.")
    if len(all_carbon) > 40:
        return _outside("The carbon-tree size limit was exceeded.")

    # A directly attached ordinary subordinate nitrile is a cyano prefix
    # under P-66.5.1.1.4, so its carbon/nitrogen pair is a source-owned prefix
    # unit rather than a continuation of the carbon parent. Derive this only
    # from the complete source inventory and its P-41 frontier; callers cannot
    # nominate arbitrary carbon exclusions.
    excluded_roles, exclusion_error = _junior_nitrile_exclusions(
        mol, Chem, groups, all_carbon,
    )
    if exclusion_error:
        return _outside(
            exclusion_error,
            barrier=SuffixCoverageBarrier.SOURCE_ROLE_COMPLETENESS,
        )
    proved_exclusions = {role.center for role in excluded_roles}
    requested_exclusions = frozenset(int(i) for i in excluded_skeletal_atoms)
    if requested_exclusions and requested_exclusions != proved_exclusions:
        return _outside(
            "Caller-supplied skeletal exclusions do not exactly match source-proved junior nitrile centers.",
            barrier=SuffixCoverageBarrier.SOURCE_ROLE_COMPLETENESS,
        )
    expected_parentable = all_carbon - proved_exclusions
    supplied_omissions = all_carbon - supplied_carbon
    # Accept either the historical full-carbon call or the exact normalized
    # set passed by cross-route producers. No other carbon may disappear.
    if supplied_omissions not in (frozenset(), frozenset(proved_exclusions)):
        return _outside(
            "Supplied parent carbon set omits atoms beyond the complete source-proved nitrile roles.",
            barrier=SuffixCoverageBarrier.SOURCE_ROLE_COMPLETENESS,
        )
    if requested_exclusions and supplied_omissions != frozenset(proved_exclusions):
        return _outside(
            "A source-proved caller exclusion was not reflected in its parent carbon set.",
            barrier=SuffixCoverageBarrier.SOURCE_ROLE_COMPLETENESS,
        )
    carbon = expected_parentable
    if not carbon:
        return _outside("No parentable carbon remains after source-role normalization.")

    adjacency = {idx: set() for idx in carbon}
    unsaturated = False
    for idx in carbon:
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            other = neighbor.GetIdx()
            if other not in carbon:
                continue
            bond = mol.GetBondBetweenAtoms(idx, other)
            if bond is None or bond.GetIsAromatic():
                return _outside("The carbon skeleton has unsupported bond grammar.")
            if bond.GetBondType() != Chem.BondType.SINGLE:
                if (kind not in {"acid", "amide"}
                        or bond.GetBondType() not in {
                            Chem.BondType.DOUBLE, Chem.BondType.TRIPLE,
                        }):
                    return _outside(
                        "This suffix class has no source-closed unsaturated-chain frontier."
                    )
                unsaturated = True
            adjacency[idx].add(other)
    if not _is_tree(adjacency):
        return _outside("The carbon skeleton is disconnected or cyclic.")
    # Potential but unassigned stereogenic features are not used to rank this
    # structural frontier. If they distinguish surviving parent records,
    # graph_parent_facts/method selection must retain its existing unresolved
    # status rather than claiming a complete PIN.

    centers = {group.center for group in groups}
    if not centers <= carbon:
        return _outside("Every selected suffix center must be in the carbon skeleton.")
    if kind == "acid" and any(
        group.owner >= 0 and adjacency[group.center] != {group.owner}
        for group in groups
    ):
        return _outside(
            "The acid carbon centers are not proved terminal atoms of the carbon tree."
        )
    if kind == "amide" and any(group.ligand < 0 for group in groups):
        return _outside("An amide site lacks its recognized nitrogen ligand.")
    for group in groups:
        if group.owner < 0:
            # A carbonyl/nitrile carbon with no carbon neighbor cannot be
            # hosted by an external carbon parent (e.g. a formyl group).
            # When a carbon neighbor exists, however, its owner is unknown;
            # omitting that external mode would leave the maximum unproved.
            outside_group = [neighbor for neighbor in mol.GetAtomWithIdx(group.center).GetNeighbors()
                             if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in group.atoms]
            carbon_hosts = [neighbor for neighbor in outside_group
                            if neighbor.GetAtomicNum() == 6]
            if carbon_hosts or outside_group:
                return _outside(
                    "The complete class inventory lacks an external attachment fact.",
                    barrier=SuffixCoverageBarrier.ATTACHMENT_COMPLETENESS,
                )
        elif group.owner not in carbon:
            return _outside(
                "An external attachment fact lies outside the proved carbon-tree parent set.",
                barrier=SuffixCoverageBarrier.ATTACHMENT_COMPLETENESS,
            )

    # With at most four leaves there are at most six endpoint pairs; direct
    # reconstruction is cheaper than setting up the dynamic program. This
    # work estimate changes only the algorithm, never the candidate frontier.
    if _USE_TREE_PATH_DP and sum(len(neighbors) <= 1 for neighbors in adjacency.values()) > 4:
        return _maximum_tree_coverage(mol, Chem, groups, kind, adjacency, centers, excluded_roles)
    embedded_paths = _all_leaf_paths(adjacency)
    rows = []
    for path in embedded_paths:
        path_atoms = set(path)
        selected = tuple(group for group in groups if group.center in path_atoms)
        if selected:
            multiple, double = _path_bond_counts(mol, path, Chem)
            rows.append(SuffixCoverageCandidate(
                path, "embedded", selected,
                tuple(group for group in groups if group.center not in path_atoms),
                multiple, double, _embedded_rule(kind),
            ))

    external_carbon = carbon - centers
    external_adjacency = {
        idx: neighbors & external_carbon
        for idx, neighbors in adjacency.items() if idx in external_carbon
    }
    for path in _all_leaf_paths(external_adjacency):
        path_atoms = set(path)
        selected = tuple(
            group for group in groups
            if group.owner in path_atoms and group.center not in path_atoms
        )
        if not selected:
            continue
        # Any same-class center on the parent is an embedded suffix site in
        # this construction and cannot be detached as a prefix.
        if centers & path_atoms:
            continue
        # For ordinary mono-/diacids in this connected carbon tree, the
        # acid-carbon leaves can be included together as chain termini; use
        # P-65.1.2.1 rather than inventing an external small-group mode.
        # The explicit external-chain rule is P-65.1.2.2.1 for >2 groups.
        # P-66 gives analogous >2-only external rules for amides and nitriles.
        if kind == "acid" and len(selected) <= 2:
            selected_centers = {group.center for group in selected}
            if not any(selected_centers <= set(embedded) for embedded in embedded_paths):
                return _outside(
                    "A small external acid mode lacks an embedded terminal-group parent proof."
                )
            continue
        if kind in {"amide", "nitrile"} and len(selected) <= 2:
            continue
        source_rule = _external_rule(kind, selected, path)
        if source_rule is None:
            continue
        multiple, double = _path_bond_counts(mol, path, Chem)
        rows.append(SuffixCoverageCandidate(
            path, "external", selected,
            tuple(group for group in groups if group not in selected),
            multiple, double, source_rule,
        ))

    if not rows:
        return _outside("No parent candidate carries a recognized suffix site.")
    # P-44.1.1, then P-44.3.2, then P-44.4.1(a) and (b) are all
    # graph-comparable before a renderer is called.
    rank = lambda row: (
        len(row.selected_groups), len(row.path),
        row.multiple_bond_count, row.double_bond_count,
    )
    maximum_rank = max(rank(candidate) for candidate in rows)
    maxima = tuple(row for row in rows if rank(row) == maximum_rank)
    return SuffixCoverageFrontier(
        "proved", maxima, maximum_rank[0],
        "Every source-eligible embedded and external carbon-tree parent was ranked by P-44.1.1, P-44.3.2, and P-44.4.1(a)/(b).",
        maximum_length=maximum_rank[1],
        maximum_multiple_bonds=maximum_rank[2],
        maximum_double_bonds=maximum_rank[3],
        excluded_prefix_roles=excluded_roles,
    )


def _maximum_tree_coverage(mol, Chem, groups, kind, adjacency, centers, excluded_roles):
    """Use the same source-closed modes and additive rank, retaining all ties.

    All eligibility and original-role proofs above have already run. Acid
    centres are terminal leaves in the connected tree, so every one- or
    two-centre external acid selection is contained in an embedded leaf path.
    This proves the exhaustive small-external-mode check without listing it.
    """
    from collections import Counter
    from .tree_path_frontier import maximum_weight_leaf_paths

    edge_weights = {}
    for left, neighbors in adjacency.items():
        for right in neighbors:
            if left < right:
                order = mol.GetBondBetweenAtoms(left, right).GetBondType()
                edge_weights[left, right] = (0, 0, int(order != Chem.BondType.SINGLE),
                                             int(order == Chem.BondType.DOUBLE))
    external_rank, external_paths = None, ()
    if len(groups) > 2:
        external = {i: neighbors - centers for i, neighbors in adjacency.items() if i not in centers}
        external_counts = Counter(group.owner for group in groups)
        external_rank, external_paths = maximum_weight_leaf_paths(external,
            {i: (external_counts[i], 1, 0, 0) for i in external}, edge_weights,
            minimum_rank=(3, 0, 0, 0))
    embedded_counts = Counter(group.center for group in groups)
    embedded_rank, embedded_paths = maximum_weight_leaf_paths(adjacency,
        {i: (embedded_counts[i], 1, 0, 0) for i in adjacency}, edge_weights,
        minimum_rank=external_rank if external_paths else None)
    modes = []
    if embedded_rank is not None and embedded_rank[0]:
        modes.append((embedded_rank, embedded_paths, 'embedded'))
    if external_rank is not None and external_rank[0] > 2:
        modes.append((external_rank, external_paths, 'external'))
    if not modes:
        return _outside("No parent candidate carries a recognized suffix site.")
    maximum_rank = max(rank for rank, _, _ in modes)
    rows = []
    for rank, paths, mode in modes:
        if rank != maximum_rank:
            continue
        for path in paths:
            members = frozenset(path)
            selected = tuple(group for group in groups
                             if (group.center if mode == 'embedded' else group.owner) in members)
            rows.append(SuffixCoverageCandidate(path, mode, selected,
                tuple(group for group in groups if group not in selected), rank[2], rank[3],
                _embedded_rule(kind) if mode == 'embedded' else _external_rule(kind, selected, path)))
    return SuffixCoverageFrontier(
        "proved", tuple(rows), maximum_rank[0],
        "Every source-eligible embedded and external carbon-tree parent was ranked by P-44.1.1, P-44.3.2, and P-44.4.1(a)/(b).",
        maximum_length=maximum_rank[1], maximum_multiple_bonds=maximum_rank[2],
        maximum_double_bonds=maximum_rank[3], excluded_prefix_roles=excluded_roles,
    )


def _group_identity(group):
    return (group.kind, int(group.center), tuple(int(i) for i in group.atoms),
            int(group.owner), int(group.ligand))


def _junior_nitrile_exclusions(mol, Chem, groups, all_carbon):
    """Validate every C#N parent exclusion from complete source evidence.

    The inventory's principal frontier establishes that the nitrile is junior;
    this helper additionally binds its exact C/N atoms and single carbon host.
    Any observed nitrile-like motif that the source inventory cannot account
    for leaves the parent-carbon universe unresolved.
    """
    observed = []
    for atom in iter_atoms(mol):
        if atom.GetAtomicNum() != 6:
            continue
        triple_n = tuple(
            neighbor.GetIdx() for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 7
            and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
            == Chem.BondType.TRIPLE
        )
        if triple_n:
            observed.append((atom.GetIdx(), triple_n))
    if not observed:
        return (), ""

    from .extended_inventory import selection_inventory

    inventory = selection_inventory(Chem.Mol(mol))
    frontier = getattr(inventory, "source_frontier", None)
    requested_ids = {_group_identity(group) for group in groups}
    if (not inventory.supported or frontier is None
            or frontier.status not in {"selected", "tie"}
            or tuple(getattr(frontier, "unknown_pairs", ()))
            or bool(getattr(frontier, "cycle", False))
            or {_group_identity(group) for group in frontier.groups} != requested_ids
            or {_group_identity(group) for group in inventory.principal} != requested_ids
            or len(inventory.source_group_facts) != len(inventory.groups)):
        return (), "A nitrile-like carbon exclusion lacks a complete P-41 source frontier."

    fact_by_identity = {
        _group_identity(group): fact
        for group, fact in zip(inventory.groups, inventory.source_group_facts)
    }
    principal_facts = [fact_by_identity.get(_group_identity(group)) for group in groups]
    if (not principal_facts or any(fact is None or fact.class_rank is None
                                  for fact in principal_facts)):
        return (), "The selected principal class rank is not source-proved."
    principal_rank = min(fact.class_rank for fact in principal_facts)

    nitrile_by_pair = {}
    for group in inventory.groups:
        if group.kind == "nitrile":
            nitrile_by_pair[(int(group.center), tuple(sorted(
                i for i in group.atoms if i != group.center
            )))] = group

    excluded = []
    for center, nitrogens in observed:
        if len(nitrogens) != 1:
            return (), "A nitrile-like carbon has an ambiguous nitrogen role."
        nitrogen = nitrogens[0]
        identity_key = (center, (nitrogen,))
        group = nitrile_by_pair.get(identity_key)
        if group is None:
            return (), "A nitrile-like carbon is absent from the complete source inventory."
        identity = _group_identity(group)
        if identity in requested_ids:
            # A principal nitrile carbon remains part of the parentable carbon
            # universe and is handled as the suffix center.
            continue
        fact = fact_by_identity.get(identity)
        if fact is None or fact.class_rank is None:
            return (), "A subordinate nitrile lacks its source class fact."
        if fact.class_rank <= principal_rank:
            # It is not proved junior and cannot be removed from the parent.
            continue
        host = int(group.owner)
        role = ExcludedJuniorNitrileRole(center, nitrogen, host, identity)
        if not _validate_excluded_role(mol, Chem, role, group, all_carbon):
            return (), "A junior nitrile is not a directly attached, exact C#N prefix role."
        excluded.append(role)

    if len(excluded) != sum(
            1 for center, _nitrogens in observed
            if (group := nitrile_by_pair.get((center, tuple(
                n for n in _nitrogens
            )))) is not None
            and _group_identity(group) not in requested_ids
            and fact_by_identity.get(_group_identity(group)) is not None
            and fact_by_identity[_group_identity(group)].class_rank > principal_rank):
        return (), "Not every source-junior nitrile was bound to one exact prefix role."
    return tuple(sorted(excluded, key=lambda role: (role.host, role.center, role.nitrogen))), ""


def _validate_excluded_role(mol, Chem, role, group, all_carbon):
    """Revalidate a derived cyano exclusion against the current source graph."""
    center, nitrogen, host = role.center, role.nitrogen, role.host
    if (role.group_identity != _group_identity(group)
            or group.kind != "nitrile"
            or center != group.center
            or nitrogen not in group.atoms
            or host != group.owner
            or center not in all_carbon or host not in all_carbon
            or set(group.atoms) != {center, nitrogen}
            or not 0 <= center < mol.GetNumAtoms()
            or not 0 <= nitrogen < mol.GetNumAtoms()
            or not 0 <= host < mol.GetNumAtoms()):
        return False
    center_atom = mol.GetAtomWithIdx(center)
    nitrogen_atom = mol.GetAtomWithIdx(nitrogen)
    host_bond = mol.GetBondBetweenAtoms(center, host)
    nitrile_bond = mol.GetBondBetweenAtoms(center, nitrogen)
    if (center_atom.GetAtomicNum() != 6 or nitrogen_atom.GetAtomicNum() != 7
            or center_atom.GetDegree() != 2 or nitrogen_atom.GetDegree() != 1
            or host_bond is None or host_bond.GetBondType() != Chem.BondType.SINGLE
            or nitrile_bond is None or nitrile_bond.GetBondType() != Chem.BondType.TRIPLE):
        return False
    return all(
        neighbor.GetIdx() in {host, nitrogen}
        for neighbor in center_atom.GetNeighbors()
    )


def _path_bond_counts(mol, path, Chem):
    multiple = double = 0
    for left, right in zip(path, path[1:]):
        bond = mol.GetBondBetweenAtoms(left, right)
        if bond.GetBondType() != Chem.BondType.SINGLE:
            multiple += 1
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            double += 1
    return multiple, double


def _embedded_rule(kind):
    return {
        "acid": "P-65.1.2.1",
        "amide": "P-66.1.1.1.1.1",
        "nitrile": "P-66.5.1.1.1",
    }[kind]


def _external_rule(kind, selected, path):
    if kind == "amide":
        return "P-66.1.1.1.1.2" if len(selected) > 2 else None
    if kind == "nitrile":
        # P-66.5.1.1.2 is limited to terminal cyano groups linked to an
        # unbranched alkane; the caller has already excluded unsaturated C-C.
        return "P-66.5.1.1.2" if len(selected) > 2 else None
    if kind == "acid":
        if len(selected) > 2:
            return "P-65.1.2.2.1"
        return None
    return None


def _outside(reason: str, *, barrier: SuffixCoverageBarrier | None = None) -> SuffixCoverageFrontier:
    return SuffixCoverageFrontier("out_of_scope", reason=reason, barrier=barrier)


def _is_tree(adjacency):
    if not adjacency:
        return False
    seen = set()
    stack = [next(iter(adjacency))]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(adjacency[idx] - seen)
    edge_count = sum(map(len, adjacency.values())) // 2
    return len(seen) == len(adjacency) and edge_count == len(adjacency) - 1


def _all_leaf_paths(adjacency):
    """Enumerate all endpoint paths in a tree, including singleton parents."""
    if not adjacency:
        return ()
    leaves = sorted(idx for idx, neighbors in adjacency.items() if len(neighbors) <= 1)
    paths = set()
    if len(adjacency) == 1:
        return ((next(iter(adjacency)),),)
    # Every endpoint pair in a tree has one path. Build the predecessor tree
    # once per left endpoint, then recover all of its later endpoint paths.
    # The external-parent graph may be a forest; unreachable pairs remain
    # absent just as they do with a separate traversal for each pair.
    for position, left in enumerate(leaves[:-1]):
        parent = {left: None}
        queue = [left]
        for idx in queue:
            for neighbor in sorted(adjacency[idx]):
                if neighbor not in parent:
                    parent[neighbor] = idx
                    queue.append(neighbor)
        for right in leaves[position + 1:]:
            if right not in parent:
                continue
            path = []
            idx = right
            while idx is not None:
                path.append(idx)
                idx = parent[idx]
            paths.add(tuple(reversed(path)))
    return tuple(sorted(paths))
