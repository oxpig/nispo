"""Mapped P-51.4 frontier for branched ordinary C/N/O/S trees.

This module recognizes a complete ordinary ketone, alcohol, thiol or primary
amine site inventory,
then enumerates carbon-leaf paths sufficient to prove every maximum that owns
all sites and isolated skeletal N/O/S units. It does not claim names for off-path non-carbon groups or
for paths outside the admitted finite graph grammar.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import ROOTS
from .component_mapping import MappedComponent, decorate_component, fully_covered, mapped_branch


_HETERO_ORDER = (8, 16, 7)
_TERMS = {8: "oxa", 16: "thia", 7: "aza"}


class BranchedReplacementStatus(str, Enum):
    OUTSIDE_SCOPE = "outside_scope"
    PENDING = "pending"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class BranchedReplacementCandidate:
    path: tuple[int, ...]
    name: str
    heterounits: tuple[tuple[int, str], ...]
    atom_locants: tuple[tuple[int, str], ...]
    branch_roles: tuple
    rank: tuple
    prefix_facts: tuple = ()
    suffix_roles: tuple = ()
    ownership: tuple = ()


@dataclass(frozen=True, slots=True)
class BranchedReplacementOutcome:
    status: BranchedReplacementStatus
    decision: BranchedReplacementCandidate | None = None
    candidates: tuple[BranchedReplacementCandidate, ...] = ()
    reason: str = ""
    evidence: tuple = ()
    universe_complete: bool = False
    path_comparisons: tuple = ()


def branched_replacement_frontier(mol, Chem):
    """Resolve an ordinary suffix-bearing tree with isolated N/O/S linkers."""
    # A cheap upper bound avoids walking ordinary carbon graphs that cannot
    # meet P-51.4.1.1 even if every O/S atom were a skeletal unit.
    if sum(a.GetAtomicNum() in _TERMS for a in iter_atoms(mol)) < 4:
        return _outside("fewer than four total N/O/S atoms make the replacement threshold impossible")
    from .extended_inventory import selection_inventory

    if (mol.GetRingInfo().NumRings()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or a.GetAtomicNum() == 1
                   or (a.GetIsotope() and a.GetAtomicNum() != 6)
                   or (int(a.GetChiralTag()) and
                       a.GetChiralTag() not in {Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                               Chem.ChiralType.CHI_TETRAHEDRAL_CCW})
                   for a in iter_atoms(mol))
            or mol.GetStereoGroups()
            or any(b.GetBondType() != Chem.BondType.SINGLE and not (
                   b.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
                   and b.GetBeginAtom().GetAtomicNum() == b.GetEndAtom().GetAtomicNum() == 6
                   or b.GetBondType() == Chem.BondType.DOUBLE
                   and {b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()} == {6, 8})
                   for b in iter_bonds(mol))):
        return _outside("unsupported annotation, ring, charge, radical, or heteroatom multiple bond")
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if (not heavy or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7, 8, 16}
                         for i in heavy)):
        return _outside("the complete heavy-atom inventory is not C/N/O/S")
    from .stereochemistry import _potential_stereochemistry
    if any(feature.type not in {Chem.StereoType.Atom_Tetrahedral,
                                Chem.StereoType.Bond_Double}
           for feature in _potential_stereochemistry(mol, Chem)):
        return _outside("the graph has a stereogenic unit outside ordinary tetrahedral/EZ scope")
    adjacency = {i: frozenset(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                               if n.GetAtomicNum() > 1) for i in heavy}
    if not _is_tree(adjacency):
        return _outside("the complete heavy-atom graph is not a connected tree")

    inventory = selection_inventory(mol)
    groups = tuple(inventory.principal) if inventory.supported else ()
    if (not inventory.supported or not groups
            or groups[0].kind not in {"ketone", "alcohol", "thiol", "amine"}
            or any(g.kind != groups[0].kind for g in groups)):
        return _outside("complete homogeneous ordinary principal-site inventory is unavailable")
    if groups[0].kind in {'alcohol', 'thiol'}:
        source_frontier = inventory.source_frontier
        if (source_frontier.status not in {'selected', 'tie'}
                or source_frontier.unknown_pairs or source_frontier.cycle):
            return _pending("the homogeneous OH/SH inventory lacks a closed P-41/P-43 frontier",
                            universe_complete=False, evidence=(source_frontier,))
    # Reuse the same original-atom site/host and valence proof as the carbon
    # parent frontier. A carbonyl O is a suffix atom, never a fifth skeletal
    # heterounit; primary-amine N is likewise excluded from the parent chain.
    from .principal_site_parent_frontier import ordinary_source_principal_sites
    sites = ordinary_source_principal_sites(mol, Chem, inventory)
    if sites is None:
        return _outside("principal sites lack a closed ordinary source class and valence/host proof")
    suffix_atoms = set()
    hosts = set()
    for site in sites:
        suffix_atoms.update(set(site.group_atoms) - {site.host_atom})
        hosts.add(site.host_atom)

    skeletal = heavy - suffix_atoms
    for idx in skeletal:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() in _TERMS:
            nitrogen = atom.GetAtomicNum() == 7
            if (atom.GetDegree() != 2 or atom.GetFormalCharge()
                    or atom.GetTotalNumHs(includeNeighbors=True) != int(nitrogen)
                    or atom.GetTotalValence() != (3 if nitrogen else 2)
                    or any(mol.GetBondBetweenAtoms(idx, n).GetBondType() != Chem.BondType.SINGLE
                           for n in adjacency[idx])):
                return _outside("only neutral divalent O/S or secondary NH linkers are admitted")
        elif atom.GetAtomicNum() == 6 and atom.GetTotalValence() != 4:
            return _outside("only ordinary neutral carbon valence is admitted")
    skeleton_adjacency = {i: frozenset(n for n in adjacency[i] if n in skeletal)
                          for i in skeletal}
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() in _TERMS
           and any(mol.GetAtomWithIdx(j).GetAtomicNum() in _TERMS
                   for j in skeleton_adjacency[i]) for i in skeletal):
        return _outside("adjacent skeletal heteroatoms need separate heterounit evidence")
    if not _is_tree(skeleton_adjacency):
        return _outside("removing typed suffix atoms does not leave one tree")

    paths = _carbon_ended_paths(mol, skeleton_adjacency)
    if not paths:
        return _outside("no carbon-ended path exists in the admitted skeleton")
    all_hetero = frozenset(i for i in skeletal
                           if mol.GetAtomWithIdx(i).GetAtomicNum() in _TERMS)
    complete_paths = []
    site_paths = []
    threshold_seen = False
    for path in paths:
        path_set = frozenset(path)
        unit_ids = tuple(i for i in path
                         if mol.GetAtomWithIdx(i).GetAtomicNum() in _TERMS)
        if len(unit_ids) >= 4:
            threshold_seen = True
        if hosts <= path_set:
            site_paths.append((path, unit_ids))
            if all_hetero <= path_set and len(unit_ids) >= 4:
                complete_paths.append((path, unit_ids))
    if not site_paths:
        if threshold_seen:
            return _pending("a threshold-eligible chain exists, but full site/heterounit path coverage is not proved",
                            universe_complete=True,
                            evidence=("P-51.4.1.1 threshold detected on enumerated paths",
                                      "complete inventory is not jointly owned by one path"))
        return _outside("all maximum-relevant source-site paths were enumerated and none reaches four isolated heterounits")
    if not complete_paths:
        if threshold_seen:
            return _pending("a threshold-eligible path exists, but no complete maximum-site path covers every skeletal heterounit",
                            universe_complete=True,
                            evidence=("P-44.1.1 site paths enumerated",
                                      "leaf-endpoint dominance preserves sites and heterounits while increasing parent size",
                                      "heterounit upper-bound path is not source-complete"))
        return _outside("all maximum-relevant source-site paths were enumerated and none reaches four isolated heterounits")

    # In this tree scope every skeletal N/O/S has degree two and each leaf is C.
    # Extending an internal-ended path to a carbon leaf cannot remove a site or
    # heterounit and increases parent size. Thus leaf-pair enumeration is
    # complete for threshold detection and all P-44 maxima.
    # P-44.1.1 is fixed at full site coverage in this admitted graph scope.
    # Compare every site-covering chain at P-44.3(a) before length: a longer
    # branch-origin chain with fewer O/S units cannot displace the complete
    # all-heterounit path. That upper bound closes the unit-count maximum.
    maximum_hetero_count = max(len(units) for _, units in site_paths)
    if maximum_hetero_count != len(all_hetero) or maximum_hetero_count < 4:
        return _pending("the maximum-site path frontier does not prove a four-unit replacement parent",
                        universe_complete=True,
                        evidence=("P-44.1.1/P-44.3(a) comparison incomplete",
                                  "leaf-endpoint dominance proof applies only within admitted tree scope"))
    p44_hetero_paths = [(path, units) for path, units in site_paths
                        if len(units) == maximum_hetero_count]
    longest = max(len(path) for path, _ in p44_hetero_paths)
    maxima = [(path, units) for path, units in p44_hetero_paths if len(path) == longest]
    # P-44.4.1(a)-(b) compares the number of multiple/double bonds before
    # the fixed-numbering locants. A same-length saturated arm must not win
    # merely because its heteroatom locants are lower.
    bond_rank = min(_multiple_count_rank(mol, path) for path, _ in maxima)
    maxima = [(path, units) for path, units in maxima
              if _multiple_count_rank(mol, path) == bond_rank]
    path_comparisons = tuple((tuple(path), len(units), len(path))
                             for path, units in site_paths)
    numbered = []
    for path, units in maxima:
        for oriented in (path, tuple(reversed(path))):
            rank = _numbered_parent_rank(mol, oriented, groups)
            numbered.append((rank, oriented, units))
    best_rank = min(rank for rank, _path, _units in numbered)
    rendered = []
    for rank, path, units in numbered:
        if rank != best_rank:
            continue
        candidate = _render_candidate(mol, Chem, path, units, groups, hosts,
                                      heavy, suffix_atoms)
        if candidate is None or candidate.rank != rank:
            return _pending("a source-maximal chain has no complete mapped branch rendering",
                            universe_complete=True,
                            evidence=("maximum-path-renderer-unavailable", tuple(path)))
        rendered.append(candidate)
    winners = tuple(rendered)
    # Fixed P-15 numbering and the P-44 locant sequence normally collapse the
    # two orientations. Distinct equally ranked physical parent frames remain
    # unresolved rather than being selected by traversal order or spelling.
    parent_structures = {_parent_structure_key(mol, candidate, groups)
                         for candidate in winners}
    if len(parent_structures) != 1:
        return BranchedReplacementOutcome(
            BranchedReplacementStatus.PENDING, candidates=winners,
            reason="equally ranked distinct maximum parent frames remain",
            evidence=("P-44.3(a)-(b)/P-44.4/P-45 comparison evidence",),
            universe_complete=True, path_comparisons=path_comparisons)
    # Compare annotations on the selected parent independently of prefix
    # spelling. Unsupported stereochemical comparisons remain pending.
    from .parent_annotations import isotope_seniority, stereo_seniority, stereo_identity
    from .stereochemistry import _has_assigned_stereo
    has_stereo = _has_assigned_stereo(mol, Chem)
    identities = tuple(stereo_identity(mol, {
        idx: str(offset) for offset, idx in enumerate(candidate.path, 1)})
        for candidate in winners) if has_stereo else ()
    same_stereo_identity = (bool(identities) and identities[0] is not None
                            and all(item == identities[0] for item in identities))
    annotated = []
    for candidate in winners:
        mapping = {idx: str(offset) for offset, idx in enumerate(candidate.path, 1)}
        iso = isotope_seniority(mol, mapping)
        stereo = stereo_seniority(mol, mapping) if has_stereo and not same_stereo_identity else ()
        if iso is None or stereo is None:
            return _pending("parent annotation seniority is not source-complete",
                            universe_complete=True, evidence=("P-44.4.1.11/.12",))
        annotated.append(((iso, stereo), candidate))
    if len({rank[1][0] if rank[1] else () for rank, _ in annotated}) > 1:
        return _pending("different stereogenic-unit inventories need a broader P-44 comparison",
                        universe_complete=True, evidence=("P-44.4.1.12",))
    annotation_rank = min(rank for rank, _ in annotated)
    winners = tuple(candidate for rank, candidate in annotated if rank == annotation_rank)
    # P-45.0/.2 applies here because the mapped candidates now share one
    # numbered senior parent structure. Compare typed prefix count and locants;
    # only its final alphanumeric criterion may use the rendered prefix text.
    prefix_rank = min(_p45_prefix_rank(candidate) for candidate in winners)
    p45_winners = tuple(candidate for candidate in winners
                        if _p45_prefix_rank(candidate) == prefix_rank)
    from .parent_annotations import isotope_numbering_key, stereo_numbering
    numbered_annotations = []
    for candidate in p45_winners:
        mapping = {idx: str(offset) for offset, idx in enumerate(candidate.path, 1)}
        iso = isotope_numbering_key(mol, mapping)
        stereo = stereo_numbering(mol, mapping)
        if iso is None or stereo is None:
            return _pending("same-parent annotation numbering is incomplete",
                            universe_complete=True, evidence=("P-14.4(i)-(j)",))
        numbered_annotations.append(((iso, stereo), candidate))
    numbering_rank = min(rank for rank, _ in numbered_annotations)
    p45_winners = tuple(candidate for rank, candidate in numbered_annotations
                        if rank == numbering_rank)
    alpha_rank = min(_p45_alphabetical_key(candidate) for candidate in p45_winners)
    selected_set = tuple(candidate for candidate in p45_winners
                          if _p45_alphabetical_key(candidate) == alpha_rank)
    if len({candidate.name for candidate in selected_set}) != 1:
        return BranchedReplacementOutcome(
            BranchedReplacementStatus.PENDING, candidates=selected_set,
            reason="same-parent P-45 prefix criteria leave distinct renderings",
            evidence=("P-45.0/P-45.2/P-45.5 comparison unresolved",),
            universe_complete=True, path_comparisons=path_comparisons)
    selected = selected_set[0]
    # P-63.1.2/P-63.1.5 multiply distinct same-kind suffix groups, including
    # groups sharing one carbon host. P-14.3.5 repeats that fixed locant;
    # only the exclusive suffix atoms, rather than the hosts, are disjoint.
    return BranchedReplacementOutcome(
        BranchedReplacementStatus.COMPLETE, selected, winners,
        "complete mapped maximum replacement chain", selected.branch_roles,
        universe_complete=True, path_comparisons=path_comparisons)


def _numbered_parent_rank(mol, path, groups):
    """Fixed P-15 numbering and P-44 locants, independent of any renderer."""
    position = {atom: offset + 1 for offset, atom in enumerate(path)}
    elements = {atom: mol.GetAtomWithIdx(atom).GetAtomicNum() for atom in path}
    hetero = tuple(position[atom] for atom in path if elements[atom] in _TERMS)
    by_element = tuple(tuple(position[atom] for atom in path if elements[atom] == number)
                       for number in _HETERO_ORDER)
    hosts = (group.center if group.kind in {"alcohol", "ketone"} else
             next(n.GetIdx() for n in mol.GetAtomWithIdx(group.center).GetNeighbors()
                  if n.GetAtomicNum() == 6) for group in groups)
    suffix = tuple(sorted(position[host] for host in hosts))
    ene, yne = _unsaturation_locants(mol, path)
    return hetero, *by_element, suffix, tuple(sorted(ene + yne)), ene, yne


def _unsaturation_locants(mol, path):
    ene, yne = [], []
    for position, (left, right) in enumerate(zip(path, path[1:]), 1):
        order = mol.GetBondBetweenAtoms(left, right).GetBondTypeAsDouble()
        if order == 2:
            ene.append(position)
        elif order == 3:
            yne.append(position)
    return tuple(ene), tuple(yne)


def _multiple_count_rank(mol, path):
    ene, yne = _unsaturation_locants(mol, path)
    return -len(ene + yne), -len(ene)


def _render_candidate(mol, Chem, path, units, groups, hosts, all_atoms, suffix_atoms):
    if len(path) not in ROOTS or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                                     for i in (path[0], path[-1])):
        return None
    path_set = frozenset(path)
    position = {atom: offset + 1 for offset, atom in enumerate(path)}
    branches = []
    roles = []
    owned = set(path) | set(suffix_atoms)
    for atom in path:
        for neighbor in sorted(
                (n for n in mol.GetAtomWithIdx(atom).GetNeighbors()
                 if n.GetAtomicNum() > 1 and n.GetIdx() not in path_set
                 and n.GetIdx() not in suffix_atoms),
                key=lambda n: n.GetIdx()):
            root = neighbor.GetIdx()
            text, atoms = mapped_branch(mol, Chem, root, atom)
            if not text or not atoms or atoms & owned:
                return None
            if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   or mol.GetAtomWithIdx(i).GetIsotope()
                   or mol.GetAtomWithIdx(i).IsInRing()
                   or mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetTotalValence() != 4
                   for i in atoms):
                return None
            if any(bond.GetBondType() != Chem.BondType.SINGLE
                   for i in atoms for bond in mol.GetAtomWithIdx(i).GetBonds()
                   if bond.GetOtherAtomIdx(i) in atoms):
                return None
            branches.append((str(position[atom]), text))
            roles.append(("carbon-branch", text, root, atom, tuple(sorted(atoms)),
                          position[atom]))
            owned.update(atoms)
    if not fully_covered(mol, owned, frozenset()):
        return None

    # P-15.4.3.2.1 fixes the lower all-hetero locant set, then element order.
    # P-44.4.1.5/.6 and .8 use the corresponding parent and suffix locants.
    rank = _numbered_parent_rank(mol, path, groups)
    prefix = ""
    for element in _HETERO_ORDER:
        term = _TERMS[element]
        locants = [str(position[i]) for i in path
                   if mol.GetAtomWithIdx(i).GetAtomicNum() == element]
        if locants:
            from .constants import MULT
            prefix = (prefix + "-" if prefix else "") + ",".join(locants) + "-" + (
                MULT[len(locants)] if len(locants) > 1 else "") + term
    from .formatting import complete_parent_name, unsaturated_parent
    ene, yne = _unsaturation_locants(mol, path)
    parent = prefix + complete_parent_name(unsaturated_parent(ROOTS[len(path)], list(ene), list(yne)))
    suffix_locants = tuple(sorted(
        position[g.center] if g.kind in {"alcohol", "ketone"} else
        position[next(n.GetIdx() for n in mol.GetAtomWithIdx(g.center).GetNeighbors()
                      if n.GetAtomicNum() == 6)]
        for g in groups
    ))
    from .formatting import attach_suffix, multiplied_suffix
    suffix = {"ketone": "one", "alcohol": "ol", "thiol": "thiol", "amine": "amine"}[groups[0].kind]
    if len(groups) > 1:
        suffix = multiplied_suffix(len(groups), suffix)
    parent = attach_suffix(parent, suffix, [str(i) for i in suffix_locants])
    prefix_facts = tuple(branches)
    parent = decorate_component(mol, Chem, MappedComponent(
        parent, tuple((idx, str(position[idx])) for idx in path), frozenset(path),
        tuple(branches), frozenset(owned) - frozenset(path)))
    if not parent:
        return None
    locants = tuple((idx, str(position[idx])) for idx in path)
    locants += tuple((g.center, str(position[g.center]) if g.kind in {"alcohol", "ketone"} else
                      str(position[next(n.GetIdx() for n in mol.GetAtomWithIdx(g.center).GetNeighbors()
                                        if n.GetAtomicNum() == 6)])) for g in groups)
    suffix_roles = tuple((group.kind, int(group.center),
                          group.center if group.kind in {"alcohol", "ketone"} else
                          next(n.GetIdx() for n in mol.GetAtomWithIdx(group.center).GetNeighbors()
                               if n.GetAtomicNum() == 6),
                          tuple(sorted(int(i) for i in group.atoms)),
                          {"ketone": "P-64.2.2.1", "alcohol": "P-63.1.2",
                           "thiol": "P-63.1.5", "amine": "P-62.2.1.2"}[group.kind])
                         for group in groups)
    exclusive_suffix_atoms = tuple(
        tuple(atom for atom in atom_tuple if atom != host)
        for _, _, host, atom_tuple, _ in suffix_roles)
    if len({atom for atom_tuple in exclusive_suffix_atoms for atom in atom_tuple}) != sum(
            len(atom_tuple) for atom_tuple in exclusive_suffix_atoms):
        return None
    ownership = (("replacement-parent", tuple(path)),) + tuple(
        ("principal-suffix", kind, center, host, atoms, source)
        for kind, center, host, atoms, source in suffix_roles) + tuple(roles)
    return BranchedReplacementCandidate(
        tuple(path), parent, tuple((i, _TERMS[mol.GetAtomWithIdx(i).GetAtomicNum()])
                                    for i in units), locants, tuple(roles), rank,
        prefix_facts, suffix_roles, ownership)


def _parent_structure_key(mol, candidate, groups):
    """Identity of a fixed-numbered parent hydride plus suffix-bearing sites."""
    sequence = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in candidate.path)
    suffix_rows = candidate.atom_locants[len(candidate.path):]
    suffix_identity = tuple(sorted((group.kind, int(row[1]))
                                   for group, row in zip(groups, suffix_rows)))
    bond_orders = tuple(mol.GetBondBetweenAtoms(left, right).GetBondTypeAsDouble()
                        for left, right in zip(candidate.path, candidate.path[1:]))
    return sequence, suffix_identity, bond_orders


def _p45_prefix_rank(candidate):
    from .formatting import prefix_alphabetization_key
    ordered = tuple(sorted(candidate.prefix_facts,
                           key=lambda row: (prefix_alphabetization_key(row[1]), int(row[0]))))
    locants = tuple(int(locant) for locant, _ in candidate.prefix_facts)
    citation_locants = tuple(int(locant) for locant, _ in ordered)
    return -len(locants), tuple(sorted(locants)), citation_locants


def _p45_alphabetical_key(candidate):
    from .formatting import prefix_alphabetization_key
    # P-45.5 compares the letters of the complete name, including the outer
    # multipliers actually written, then italic terms and numerical locants.
    # This is applied only after parent identity and all earlier facts tie.
    return prefix_alphabetization_key(candidate.name)[:3]


def _carbon_ended_paths(mol, adjacency):
    """Enumerate leaf-ended paths, complete for maxima in the admitted tree.

    Every skeletal O/S has degree two and all leaves are carbon. An internal
    endpoint can therefore be extended to a carbon leaf without dropping any
    mapped suffix host or heterounit, while strictly increasing parent size.
    Leaf-pair enumeration covers the threshold and every P-44 maximum; it is
    not a claim to enumerate every shorter carbon-ended subpath.
    """
    leaves = tuple(sorted(i for i, neighbors in adjacency.items() if len(neighbors) <= 1))
    paths = set()
    for left, right in combinations(leaves, 2):
        predecessor = {left: None}
        queue = [left]
        for current in queue:
            if current == right:
                break
            for neighbor in sorted(adjacency[current]):
                if neighbor not in predecessor:
                    predecessor[neighbor] = current
                    queue.append(neighbor)
        if right not in predecessor:
            continue
        path = []
        current = right
        while current is not None:
            path.append(current)
            current = predecessor[current]
        path = tuple(reversed(path))
        if all(mol.GetAtomWithIdx(i).GetAtomicNum() in {6, 8, 16} for i in (path[0], path[-1])) and all(
                mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in (path[0], path[-1])):
            paths.add(path)
    if len(adjacency) == 1:
        only = next(iter(adjacency))
        if mol.GetAtomWithIdx(only).GetAtomicNum() == 6:
            paths.add((only,))
    return tuple(sorted(paths))


def _is_tree(adjacency):
    if not adjacency:
        return False
    seen = set()
    stack = [next(iter(adjacency))]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adjacency[current] - seen)
    edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    return len(seen) == len(adjacency) and edges == len(adjacency) - 1


def _outside(reason):
    return BranchedReplacementOutcome(BranchedReplacementStatus.OUTSIDE_SCOPE,
                                      reason=reason)


def _pending(reason, *, universe_complete, evidence=()):
    return BranchedReplacementOutcome(BranchedReplacementStatus.PENDING,
                                      reason=reason, evidence=tuple(evidence),
                                      universe_complete=universe_complete)
