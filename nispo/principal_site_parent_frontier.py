"""Source-bounded graph frontier for ordinary principal suffix sites.

This module performs source recognition and parent-path comparison only. It
does not render names. Its bounded candidate universe is neutral, connected
all-carbon trees with ordinary single, double, or triple C-C bonds and
complete ketone, alcohol, thiol, or primary-amine group recognition. The
unannotated universe includes C-C unsaturation. Complete suffix-coverage
maxima share the existing mapped chain writer's parent isotope/R/S/E/Z census;
partial coverage retains the narrow Job334 annotation adapter. Unsupported
profiles remain outside this helper so existing naming routes retain them.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from .molecule_iteration import iter_atoms, iter_bonds


class PrincipalSiteFrontierStatus(str, Enum):
    PROVED = "proved"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class PrincipalSite:
    """One recognized source group with separate center and carbon host."""

    kind: str
    principal_center: int
    host_atom: int
    group_atoms: tuple[int, ...]
    source_rule: str
    class_rank: int
    source_group: object


@dataclass(frozen=True, slots=True)
class PrincipalSiteParentCandidate:
    """An oriented all-carbon chain with mapped selected and omitted sites."""

    parent_atoms: tuple[int, ...]
    mode: str
    selected_sites: tuple[PrincipalSite, ...]
    omitted_sites: tuple[PrincipalSite, ...]
    site_locants: tuple[tuple[int, int, int], ...]
    rank: tuple[int, int]
    parent_multiple_bond_count: int = 0
    parent_double_bond_count: int = 0
    structural_rank: tuple[int, int, int, int] = ()


@dataclass(frozen=True, slots=True)
class PrincipalSiteParentFrontier:
    """Status and proof metadata for this helper's bounded candidate universe.

    A true ``candidate_universe_complete`` covers only the declared ordinary
    all-carbon acyclic-chain universe; it does not claim completeness over
    other nomenclature methods or parent classes.
    """

    status: PrincipalSiteFrontierStatus
    candidates: tuple[PrincipalSiteParentCandidate, ...] = ()
    sites: tuple[PrincipalSite, ...] = ()
    subordinate_groups: tuple[object, ...] = ()
    principal_class: str = ""
    class_rank: int | None = None
    maximum_site_count: int | None = None
    maximum_parent_length: int | None = None
    candidate_universe_complete: bool = False
    selection_rules: tuple[str, ...] = ()
    reason: str = ""
    maximum_parent_multiple_bond_count: int | None = None
    maximum_parent_double_bond_count: int | None = None
    structural_rank: tuple[int, int, int, int] | None = None
    annotation_proof: object = None


def principal_site_parent_frontier(mol, Chem, *, principal_groups=None,
                                   max_parent_carbons=40) -> PrincipalSiteParentFrontier:
    """Prove maximum suffix coverage on an ordinary carbon tree.

    P-41 selects the recognized senior class before graph comparison. The
    class-17 selection is delegated to the aligned source frontier (including
    Job291's P-63.1.5 O/S order) before site counting.
    A tree path that ends before a leaf can be extended until it reaches a leaf;
    this cannot remove a site and either adds coverage or improves chain
    length. Thus a global P-44.1.1/P-44.3.2 maximum has leaf endpoints. The
    helper enumerates every such endpoint pair and both orientations.

    P-44.1.1 ranks the maximum number of principal sites, followed by P-44.3.2
    chain length, P-44.4.1.1 multiple-bond count, and P-44.4.1.2 double-bond
    count. All tied graph maxima are retained for downstream numbering and
    rendering.

    The normal proposal pool keeps its established 40-carbon resource bound.
    A caller checking root availability may raise that bound for graph facts
    alone; this does not expand any renderer's or preferred-method scope.
    """
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if not inventory.supported or not inventory.groups:
        return _outside("A complete recognized functional-group inventory is unavailable.")
    if any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
           for atom in iter_atoms(mol)):
        return _outside("Charged or radical input is outside this frontier.")
    if any(atom.IsInRing() for atom in iter_atoms(mol)):
        return _outside("Cyclic input requires a separate P-44.1.2.2 and P-44.2 comparison.")

    source_facts = tuple(getattr(inventory, "source_group_facts", ()))
    if len(source_facts) != len(inventory.groups):
        return _outside("P-41 source facts are missing or not aligned with the complete group inventory.")
    ranks = []
    for group, fact in zip(inventory.groups, source_facts):
        class_rank = getattr(fact, "class_rank", None)
        class_name = getattr(fact, "class_name", None)
        if (not isinstance(class_rank, int) or class_rank < 1
                or class_name != group.kind):
            return _outside("P-41 source facts are incomplete or misaligned with recognized groups.")
        ranks.append(class_rank)
    ranks = tuple(ranks)
    senior_rank = min(ranks)
    class_principal = tuple(group for group, rank in zip(inventory.groups, ranks)
                            if rank == senior_rank)
    if senior_rank == 17:
        source_frontier = inventory.source_frontier
        if source_frontier.status not in {"selected", "tie"}:
            return _outside("The applicable class-17 source-priority frontier is not selected or tied.")
        source_principal = tuple(source_frontier.groups)
    else:
        # P-41 alone closes the selected class for the homogeneous class-16
        # and class-19 site families admitted here. Their within-class
        # suffix-group identities are exactly the same recognized type.
        source_principal = class_principal

    def group_key(group):
        return (getattr(group, "kind", None), getattr(group, "center", None),
                tuple(getattr(group, "atoms", ())), getattr(group, "owner", None))

    if principal_groups is None:
        principal_groups = source_principal
    else:
        principal_groups = tuple(principal_groups)
        if {group_key(group) for group in principal_groups} != {
                group_key(group) for group in source_principal}:
            return _outside("Supplied principal groups do not match the source-qualified frontier.")
    if not principal_groups or not {group_key(group) for group in principal_groups} <= {
            group_key(group) for group in inventory.groups}:
        return _outside("A source-qualified principal group is absent from the complete inventory.")
    principal_ranks = {rank for group, rank in zip(inventory.groups, ranks)
                       if group_key(group) in {group_key(item) for item in principal_groups}}
    if len(principal_ranks) != 1:
        return _outside("Principal groups do not share one aligned P-41 class rank.")
    senior_rank = next(iter(principal_ranks))
    principal_groups = tuple(principal_groups)
    subordinate = tuple(
        group for group in inventory.groups if group_key(group) not in
        {group_key(item) for item in principal_groups}
    )
    if senior_rank not in {16, 17, 19}:
        return _outside("The principal source class is outside the bounded ketone/hydroxy/thiol/amine set.")
    kinds = {group.kind for group in principal_groups}
    if len(kinds) != 1:
        return _outside("The principal source class contains mixed group types outside the bounded comparison.")
    kind = next(iter(kinds))
    if (senior_rank, kind) not in {
        (16, "ketone"), (17, "alcohol"), (17, "thiol"), (19, "amine"),
    }:
        return _outside("The principal suffix is not an admitted ordinary group type.")

    carbon = frozenset(
        atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() == 6
    )
    if not carbon or len(carbon) > max_parent_carbons:
        return _outside("The complete carbon parent set is empty or exceeds the bounded size.")
    adjacency = {idx: set() for idx in carbon}
    for idx in carbon:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetTotalValence() != 4:
            return _outside("Only neutral carbon atoms with ordinary valence are admitted.")
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            if other not in carbon:
                continue
            bond = mol.GetBondBetweenAtoms(idx, other)
            if (bond.GetIsAromatic() or bond.GetBondType() not in {
                    Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}):
                return _outside("Only ordinary nonaromatic single, double, or triple C-C bonds are admitted.")
            adjacency[idx].add(other)
    if not _is_tree(adjacency):
        return _outside("The complete carbon parent set is not one acyclic connected tree.")

    sites = []
    for group in principal_groups:
        site = _recognize_site(mol, group, kind, senior_rank, Chem)
        if site is None:
            return _outside(
                "At least one principal group lacks the bounded source-qualified local valence or host mapping."
            )
        if site.host_atom not in carbon:
            return _outside("A principal site has no mapped carbon parent host.")
        sites.append(site)
    if len({(site.kind, site.principal_center, site.group_atoms) for site in sites}) != len(sites):
        return _outside("Distinct principal sites do not have distinct original group identities.")
    sites = tuple(sorted(sites, key=_site_key))

    paths = _endpoint_paths(adjacency)
    rows = []
    maximum = None
    for path in paths:
        selected = tuple(site for site in sites if site.host_atom in path)
        if not selected:
            continue
        # Coverage, size and bond counts are unchanged by orientation. Rank
        # the graph once, retaining the original order of every tied maximum,
        # before building its two numbering records.
        parent_bonds = tuple(
            mol.GetBondBetweenAtoms(left, right)
            for left, right in zip(path, path[1:])
        )
        multiple_count = sum(bond.GetBondType() in {
            Chem.BondType.DOUBLE, Chem.BondType.TRIPLE,
        } for bond in parent_bonds)
        double_count = sum(bond.GetBondType() == Chem.BondType.DOUBLE
                           for bond in parent_bonds)
        structural_rank = (len(selected), len(path), multiple_count, double_count)
        if maximum is not None and structural_rank < maximum:
            continue
        if maximum is None or structural_rank > maximum:
            maximum = structural_rank
            rows.clear()
        omitted = tuple(site for site in sites if site not in selected)
        for oriented in (path, tuple(reversed(path))):
            locant_by_atom = {atom: locant for locant, atom in enumerate(oriented, 1)}
            site_locants = tuple(
                (site.principal_center, site.host_atom, locant_by_atom[site.host_atom])
                for site in selected
            )
            rows.append(PrincipalSiteParentCandidate(
                tuple(oriented), "acyclic_carbon_chain", selected, omitted,
                site_locants, (len(selected), len(oriented)), multiple_count,
                double_count, structural_rank,
            ))
    if not rows:
        return _outside("No eligible acyclic parent path carries a recognized principal site.")

    maxima = tuple(rows)
    annotation_proof = None
    if (any(atom.GetIsotope() or int(atom.GetChiralTag()) for atom in iter_atoms(mol))
            or any(int(bond.GetStereo()) or int(bond.GetBondDir()) for bond in iter_bonds(mol))):
        from .principal_site_annotation_proof import (
            PrincipalSiteAnnotationStatus,
            prove_parent_annotations,
        )
        annotation_proof = prove_parent_annotations(mol, Chem, maxima)
        if annotation_proof.status is not PrincipalSiteAnnotationStatus.PROVED:
            return _outside(annotation_proof.reason)

    return PrincipalSiteParentFrontier(
        PrincipalSiteFrontierStatus.PROVED,
        maxima,
        sites,
        subordinate,
        kind,
        senior_rank,
        maximum[0],
        maximum[1],
        True,
        ("P-41", "P-44.1.1", "P-44.3.2", "P-44.4.1.1", "P-44.4.1.2"),
        "Every source-qualified principal site and every eligible endpoint path in the ordinary carbon tree was enumerated.",
        maximum[2],
        maximum[3],
        maximum,
        annotation_proof,
    )


def ordinary_source_principal_sites(mol, Chem, inventory):
    """Close the ordinary P-41 site class independently of parent topology.

    Class 17 retains its P-63.1.5 source comparison. Homogeneous ordinary
    ketones and primary amines use their P-41 class, followed by exact local
    valence/host recognition. This does not admit pseudoketones, mixed
    within-class chemistry, unknown source facts or legacy rank guesses.
    """
    if (not inventory.supported or not inventory.groups
            or len(inventory.source_group_facts) != len(inventory.groups)):
        return None
    ranks = []
    for group, fact in zip(inventory.groups, inventory.source_group_facts):
        rank = getattr(fact, 'class_rank', None)
        if not isinstance(rank, int) or rank < 1 or fact.class_name != group.kind:
            return None
        ranks.append(rank)
    rank = min(ranks)
    if rank == 17:
        frontier = inventory.source_frontier
        if frontier.status not in {'selected', 'tie'} or frontier.unknown_pairs or frontier.cycle:
            return None
        groups = tuple(frontier.groups)
    elif rank in {16, 19}:
        groups = tuple(group for group, value in zip(inventory.groups, ranks) if value == rank)
    else:
        return None
    if not groups or set(groups) != set(inventory.principal):
        return None
    kind = groups[0].kind
    if (rank, kind) not in {(16, 'ketone'), (17, 'alcohol'), (17, 'thiol'), (19, 'amine')}:
        return None
    if any(group.kind != kind for group in groups):
        return None
    sites = tuple(_recognize_site(mol, group, kind, rank, Chem) for group in groups)
    if any(site is None for site in sites):
        return None
    if len({_site_key(site) for site in sites}) != len(sites):
        return None
    return sites


def _recognize_site(mol, group, kind, class_rank, Chem):
    """Validate each source group locally without overloading ``owner``."""
    center = int(group.center)
    listed_atoms = tuple(sorted(int(atom) for atom in group.atoms))
    atom = mol.GetAtomWithIdx(center)
    if kind == "alcohol":
        oxygens = [idx for idx in listed_atoms
                   if mol.GetAtomWithIdx(idx).GetAtomicNum() == 8]
        if len(oxygens) != 1 or atom.GetAtomicNum() != 6:
            return None
        oxygen = mol.GetAtomWithIdx(oxygens[0])
        bond = mol.GetBondBetweenAtoms(center, oxygen.GetIdx())
        if (bond is None or bond.GetBondType() != Chem.BondType.SINGLE
                or oxygen.GetFormalCharge() or oxygen.GetDegree() != 1
                or oxygen.GetTotalNumHs(includeNeighbors=True) != 1):
            return None
        if set(listed_atoms) != {center, oxygen.GetIdx()}:
            return None
        source_rule = "P-63.1.2"
        host = center
    elif kind == "thiol":
        if atom.GetAtomicNum() != 16 or set(listed_atoms) != {center}:
            return None
        carbon_neighbors = [neighbor for neighbor in atom.GetNeighbors()
                            if neighbor.GetAtomicNum() == 6]
        if (len(carbon_neighbors) != 1 or atom.GetFormalCharge()
                or atom.GetDegree() != 1 or atom.GetTotalNumHs(includeNeighbors=True) != 1
                or any(bond.GetBondType() != Chem.BondType.SINGLE for bond in atom.GetBonds())):
            return None
        host = carbon_neighbors[0].GetIdx()
        if group.owner >= 0 and int(group.owner) != host:
            return None
        source_rule = "P-63.1.5"
    elif kind == "amine":
        if atom.GetAtomicNum() != 7 or set(listed_atoms) != {center}:
            return None
        heavy_neighbors = [neighbor for neighbor in atom.GetNeighbors()
                           if neighbor.GetAtomicNum() > 1]
        carbon_neighbors = [neighbor for neighbor in heavy_neighbors
                            if neighbor.GetAtomicNum() == 6]
        if (len(heavy_neighbors) != 1 or len(carbon_neighbors) != 1
                or atom.GetFormalCharge() or atom.GetDegree() != 1
                or atom.GetTotalValence() != 3
                or atom.GetTotalNumHs(includeNeighbors=True) != 2
                or any(bond.GetBondType() != Chem.BondType.SINGLE for bond in atom.GetBonds())):
            return None
        host = carbon_neighbors[0].GetIdx()
        source_rule = "P-62.1; P-62.2.1.2"
    elif kind == "ketone":
        if atom.GetAtomicNum() != 6:
            return None
        oxygens = [neighbor for neighbor in atom.GetNeighbors()
                   if neighbor.GetAtomicNum() == 8
                   and mol.GetBondBetweenAtoms(center, neighbor.GetIdx()).GetBondType()
                   == Chem.BondType.DOUBLE]
        carbons = [neighbor for neighbor in atom.GetNeighbors()
                   if neighbor.GetAtomicNum() == 6]
        if (len(oxygens) != 1 or len(carbons) != 2
                or set(listed_atoms) != {center, oxygens[0].GetIdx()}
                or oxygens[0].GetDegree() != 1 or oxygens[0].GetFormalCharge()
                or len(atom.GetNeighbors()) != 3 or atom.GetTotalValence() != 4
                or any(mol.GetBondBetweenAtoms(center, carbon.GetIdx()).GetBondType()
                       != Chem.BondType.SINGLE for carbon in carbons)):
            return None
        host = center
        source_rule = "P-64.1.1; P-64.2.2.1"
    else:
        return None
    return PrincipalSite(
        kind, center, host, listed_atoms, source_rule, class_rank, group,
    )


def _site_key(site):
    return (site.host_atom, site.principal_center, site.kind, site.group_atoms)


def _outside(reason):
    return PrincipalSiteParentFrontier(
        PrincipalSiteFrontierStatus.OUT_OF_SCOPE, reason=reason,
    )


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
    edge_count = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    return len(seen) == len(adjacency) and edge_count == len(adjacency) - 1


def _endpoint_paths(adjacency):
    """Return every unique leaf-to-leaf path in a connected carbon tree."""
    leaves = sorted(idx for idx, neighbors in adjacency.items() if len(neighbors) <= 1)
    if len(adjacency) == 1:
        return ((next(iter(adjacency)),),)
    paths = set()
    for position, left in enumerate(leaves[:-1]):
        predecessor = {left: None}
        queue = [left]
        for current in queue:
            for neighbor in sorted(adjacency[current]):
                if neighbor not in predecessor:
                    predecessor[neighbor] = current
                    queue.append(neighbor)
        for right in leaves[position + 1:]:
            if right not in predecessor:
                continue
            path = []
            current = right
            while current is not None:
                path.append(current)
                current = predecessor[current]
            paths.add(tuple(reversed(path)))
    return tuple(sorted(paths))
