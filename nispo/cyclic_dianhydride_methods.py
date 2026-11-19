"""Mapped pseudoketone decisions for bounded P-65.7.7.2 cyclic dianhydrides."""

from dataclasses import dataclass
from functools import lru_cache

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision


@dataclass(frozen=True, slots=True)
class CyclicDianhydrideGraph:
    parent_ring_atoms: frozenset[int]
    parent_cycle: tuple[int, ...]
    carbonyl_carbons: frozenset[int]
    carbonyl_oxygens: frozenset[int]
    bridge_oxygens: frozenset[int]
    acid_attachment_hosts: frozenset[int]
    bridge_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class CyclicDianhydrideParentConstruction:
    """A candidate fused hydride parent and graph map, before name selection."""
    fused_parent_name: str
    hydrogenation_prefix: str
    fusion_network: object
    parent_atom_map: tuple[int, ...]
    core_original_atoms: tuple[int, ...]
    rule: str = "P-25.3.7.1/P-25.3.3.1/P-65.7.7.2"
    hydrogen_state: str = "complete"
    hydrogen_state_reason: str | None = None
    mancude_carbon_matchings: tuple = ()
    residual_carbon_matchings: tuple = ()
    suffix_accommodation: str = "complete"


@dataclass(frozen=True, slots=True)
class CyclicDianhydrideNameCandidate:
    """One complete name/map supported by a parent and numbering candidate."""
    name: str
    parent: CyclicDianhydrideParentConstruction
    atom_locants: tuple[tuple[int, str], ...]
    suffix_locants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CyclicDianhydrideDecisionFrontier:
    """Rendered candidates and a decision only when every survivor agrees."""
    status: str
    reason: str | None
    candidates: tuple[CyclicDianhydrideNameCandidate, ...] = ()
    decision: object | None = None


@dataclass(frozen=True, slots=True)
class CyclicDianhydrideMethodDecision(MethodDecision):
    """Decision retaining symmetry-equivalent original-atom locant maps."""
    equivalent_atom_locants: tuple[tuple[tuple[int, str], ...], ...] = ()


def cyclic_dianhydride_graph(mol):
    """Return a role-complete graph witness, or ``None`` outside the scope.

    The recognized operation is two O-bridges joining four carbonyl carbons;
    each carbonyl carbon is attached to a different atom of one saturated,
    unsubstituted all-carbon monocycle (4--8 atoms). Every heavy atom must be
    accounted for by that construction. No name, fused-system locant, or PIN
    assertion is made here.
    """
    if (mol.GetNumAtoms() > 24 or mol.GetNumHeavyAtoms() != mol.GetNumAtoms()
            or mol.GetStereoGroups()):
        return None
    atoms = tuple(iter_atoms(mol))
    if any(a.GetAtomicNum() not in {6, 8} or a.GetFormalCharge()
           or a.GetIsotope() or a.GetNumRadicalElectrons()
           or int(a.GetChiralTag()) for a in atoms):
        return None
    if any(int(b.GetStereo()) or int(b.GetBondDir())
           or b.GetBondTypeAsDouble() not in {1.0, 2.0}
           for b in iter_bonds(mol)):
        return None

    carbonyls = set()
    carbonyl_oxygens = set()
    for atom in atoms:
        if atom.GetAtomicNum() != 6:
            continue
        doubles = [n.GetIdx() for n in atom.GetNeighbors()
                   if n.GetAtomicNum() == 8
                   and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2.0]
        if doubles:
            if len(doubles) != 1:
                return None
            carbonyls.add(atom.GetIdx())
            carbonyl_oxygens.update(doubles)
    if len(carbonyls) != 4 or len(carbonyl_oxygens) != 4:
        return None

    bridges = []
    for atom in atoms:
        if atom.GetAtomicNum() != 8 or atom.GetIdx() in carbonyl_oxygens:
            continue
        if atom.GetDegree() != 2:
            return None
        neighbors = tuple(n.GetIdx() for n in atom.GetNeighbors())
        if (len(neighbors) != 2 or not set(neighbors) <= carbonyls
                or any(mol.GetBondBetweenAtoms(atom.GetIdx(), n).GetBondTypeAsDouble() != 1.0
                       for n in neighbors)):
            return None
        bridges.append((atom.GetIdx(), tuple(sorted(neighbors))))
    if len(bridges) != 2 or len({c for _, pair in bridges for c in pair}) != 4:
        return None

    hosts = set()
    for center in carbonyls:
        carbon_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(center).GetNeighbors()
                            if n.GetAtomicNum() == 6 and n.GetIdx() not in carbonyls]
        oxygen_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(center).GetNeighbors()
                            if n.GetAtomicNum() == 8]
        if len(carbon_neighbors) != 1 or len(oxygen_neighbors) != 2:
            return None
        hosts.add(carbon_neighbors[0])
    if len(hosts) != 4:
        return None

    parent_atoms = {a.GetIdx() for a in atoms if a.GetAtomicNum() == 6} - carbonyls
    if not 4 <= len(parent_atoms) <= 8 or hosts - parent_atoms:
        return None
    # The parent is a simple saturated carbon cycle with no other heavy-atom
    # substituents. Its order is graph-local and is not a nomenclature locant.
    adjacency = {idx: set() for idx in parent_atoms}
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in parent_atoms and right in parent_atoms:
            if bond.GetBondTypeAsDouble() != 1.0:
                return None
            adjacency[left].add(right)
            adjacency[right].add(left)
        elif (left in parent_atoms and right not in carbonyls
              and mol.GetAtomWithIdx(right).GetAtomicNum() != 1
              or right in parent_atoms and left not in carbonyls
              and mol.GetAtomWithIdx(left).GetAtomicNum() != 1):
            return None
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return None
    seen, todo = set(), [min(parent_atoms)]
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        todo.extend(adjacency[current] - seen)
    if seen != parent_atoms:
        return None
    if any(sum(host in parent_atoms for host in [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors()]) != 1
           for c in carbonyls):
        return None
    expected_heavy = parent_atoms | carbonyls | carbonyl_oxygens | {b for b, _ in bridges}
    if expected_heavy != {a.GetIdx() for a in atoms}:
        return None

    # Stable traversal is only a convenient structural representation. It is
    # explicitly not the P-44/P-25 fused-system numbering.
    start = min(parent_atoms)
    first = min(adjacency[start])
    cycle = [start, first]
    while len(cycle) < len(parent_atoms):
        nxt = (adjacency[cycle[-1]] - {cycle[-2]}).pop()
        if nxt in cycle:
            return None
        cycle.append(nxt)
    if start not in adjacency[cycle[-1]]:
        return None
    return CyclicDianhydrideGraph(
        frozenset(parent_atoms), tuple(cycle), frozenset(carbonyls),
        frozenset(carbonyl_oxygens), frozenset(b for b, _ in bridges),
        frozenset(hosts), tuple(sorted(pair for _, pair in bridges)))


_CENTRAL_COMPONENTS = {
    4: "cyclobuta", 5: "cyclopenta", 6: "benzo",
    7: "cyclohepta", 8: "cycloocta",
}
_HYDRO_MULTIPLIERS = {
    2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa",
    7: "hepta", 8: "octa", 9: "nona", 10: "deca", 11: "undeca",
    12: "dodeca", 13: "trideca", 14: "tetradeca", 15: "pentadeca",
    16: "hexadeca", 17: "heptadeca", 18: "octadeca", 19: "nonadeca",
    20: "icosa",
}


def _maximum_carbon_matching_frontier(mol, *, excluded=frozenset()):
    """Return every maximum C–C matching on a selected parent-carbon graph."""
    graph = {atom.GetIdx(): set() for atom in iter_atoms(mol)
             if atom.GetAtomicNum() == 6 and atom.GetIdx() not in excluded}
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in graph and right in graph:
            graph[left].add(right)
            graph[right].add(left)

    @lru_cache(maxsize=None)
    def solve(remaining):
        if not remaining:
            return 0, (frozenset(),)
        atom = min(remaining)
        count, matchings = solve(remaining - {atom})
        options = [(count, matching) for matching in matchings]
        for neighbor in sorted(graph[atom] & remaining):
            count, matchings = solve(remaining - {atom, neighbor})
            options.extend((count + 1, matching | {tuple(sorted((atom, neighbor)))})
                           for matching in matchings)
        best = max(count for count, _ in options)
        frontier = tuple(sorted({matching for count, matching in options if count == best},
                                key=lambda matching: tuple(sorted(matching))))
        return best, frontier

    return solve(frozenset(graph))


def _cycle_numberings(cycle):
    size = len(cycle)
    for direction in (tuple(cycle), tuple(reversed(cycle))):
        for start in range(size):
            yield direction[start:] + direction[:start]


def _core_without_carbonyl_oxygens(mol, witness, Chem):
    keep = tuple(idx for idx in range(mol.GetNumAtoms())
                 if idx not in witness.carbonyl_oxygens)
    local = {original: index for index, original in enumerate(keep)}
    graph = Chem.RWMol()
    for idx in keep:
        source = mol.GetAtomWithIdx(idx)
        atom = Chem.Atom(source)
        atom.SetIsotope(0)
        graph.AddAtom(atom)
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in local and right in local:
            graph.AddBond(local[left], local[right], bond.GetBondType())
    result = graph.GetMol()
    try:
        Chem.SanitizeMol(result)
    except (ValueError, RuntimeError):
        return None
    return result, keep


def cyclic_dianhydride_parent_constructions(mol, Chem):
    """Enumerate graph-derived P-25 multiparent constructions for edge-fused cases.

    This constructs only the fused hydride parent witness and its map to the
    original graph. It does not select among P-25 numbering candidates or
    assign the four tetrone locants; callers must use a separately proved
    P-25 numbering frontier before rendering a PIN.
    """
    witness = cyclic_dianhydride_graph(mol)
    if witness is None:
        return ()
    size = len(witness.parent_ring_atoms)
    component = _CENTRAL_COMPONENTS.get(size)
    if component is None:
        return ()

    # P-65.7.7.2 nonadjacent pairings require a bridged-fused parent route;
    # this function is restricted to two disjoint edge-fusions.
    host_for_carbonyl = {}
    for center in witness.carbonyl_carbons:
        host_for_carbonyl[center] = next(
            n.GetIdx() for n in mol.GetAtomWithIdx(center).GetNeighbors()
            if n.GetAtomicNum() == 6 and n.GetIdx() not in witness.carbonyl_carbons)
    parent_adjacency = {idx: set() for idx in witness.parent_ring_atoms}
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in parent_adjacency and right in parent_adjacency:
            parent_adjacency[left].add(right)
            parent_adjacency[right].add(left)
    edge_pairs = []
    for left, right in witness.bridge_pairs:
        hosts = (host_for_carbonyl[left], host_for_carbonyl[right])
        host_left, host_right = hosts
        if (host_right not in parent_adjacency[host_left]
                or host_left not in parent_adjacency[host_right]):
            return ()
        edge_pairs.append(hosts)
    if len({host for pair in edge_pairs for host in pair}) != 4:
        return ()

    from .fusion_components import fusion_construction
    from .fusion_networks import FusionNetwork
    core = _core_without_carbonyl_oxygens(mol, witness, Chem)
    if core is None:
        return ()
    core_mol, core_original_atoms = core
    target_hydrogens = sum(atom.GetTotalNumHs(includeNeighbors=True)
                           for atom in iter_atoms(mol))
    found = {}
    for numbered_cycle in _cycle_numberings(witness.parent_cycle):
        locants = {atom: str(index + 1) for index, atom in enumerate(numbered_cycle)}
        descriptor_pairs = []
        for left, right in edge_pairs:
            a, b = locants[left], locants[right]
            if abs(int(a) - int(b)) != 1:
                descriptor_pairs = []
                break
            descriptor_pairs.append((a, b))
        if len(descriptor_pairs) != 2:
            continue
        # The identical furan parent components are primed by occurrence;
        # enumerate both citation orders and both orientations of each edge.
        variants = []
        for first_pair in (descriptor_pairs, tuple(reversed(descriptor_pairs))):
            for reverse_a in (False, True):
                for reverse_b in (False, True):
                    a, b = first_pair[0]
                    c, d = first_pair[1]
                    if reverse_a:
                        a, b = b, a
                    if reverse_b:
                        c, d = d, c
                    variants.append((a, b, c, d))
        for a, b, c, d in variants:
            name = f"{component}[{a},{b}-c:{c},{d}-c']difuran"
            network = fusion_construction(name)
            if not isinstance(network, FusionNetwork):
                continue
            parent_graph = Chem.MolFromSmiles(network.skeleton)
            if parent_graph is None or parent_graph.GetNumAtoms() != core_mol.GetNumAtoms():
                continue
            mappings = core_mol.GetSubstructMatches(
                parent_graph, uniquify=False, maxMatches=10000)
            # RDKit silently truncates at maxMatches. A truncated embedding
            # frontier cannot establish that every eligible parent agrees.
            if len(mappings) >= 10000:
                return ()
            if not mappings:
                continue
            matching_count, matchings = _maximum_carbon_matching_frontier(parent_graph)
            carbon_atoms = frozenset(a.GetIdx() for a in parent_graph.GetAtoms()
                                     if a.GetAtomicNum() == 6)
            perfect_mancude_frontier = 2 * matching_count == len(carbon_atoms)
            parent_hydrogens = sum(atom.GetTotalNumHs() for atom in parent_graph.GetAtoms())
            for mapping in mappings:
                original_map = tuple(core_original_atoms[idx] for idx in mapping)
                query_by_local = {local: query for query, local in enumerate(mapping)}
                local_by_original = {original: local
                                     for local, original in enumerate(core_original_atoms)}
                suffix_sites = frozenset(query_by_local[local_by_original[center]]
                                         for center in witness.carbonyl_carbons)
                if len(suffix_sites) != 4:
                    return ()
                residual_matching_count, residual_matchings = (
                    _maximum_carbon_matching_frontier(parent_graph, excluded=suffix_sites))
                residual_count = len(carbon_atoms - suffix_sites)
                perfect_parent_matching = perfect_mancude_frontier
                perfect_residual_matching = (2 * residual_matching_count == residual_count)
                suffix_sites_covered = any(
                    all(any(site in edge for edge in matching) for site in suffix_sites)
                    for matching in matchings)
                suffix_bond_loss = matching_count - residual_matching_count
                suffix_accommodation = (
                    "complete" if (perfect_parent_matching and perfect_residual_matching
                                   and suffix_sites_covered and suffix_bond_loss == 2)
                    else "unresolved")
                hydrogen_state = "unresolved"
                reason = ("mancude or residual suffix matching leaves indicated-H "
                          "positions unresolved")
                hydro_prefix = ""
                if suffix_accommodation == "complete":
                    # Full matching proves no unmatched parent carbon needs
                    # indicated H. Perfect matching of the graph after all four
                    # suffix carbons are excluded proves that two C=C bonds can
                    # be selected wholly outside those sites. The full-minus-
                    # residual maximum-matching count of exactly two then
                    # accounts for the remaining two pairs at the four suffix
                    # carbons, allowing P-58.2.2.3's rearrangement operation.
                    # Thus the four ketones remove four parent CH hydrogens.
                    hydrogenation_count = (target_hydrogens
                        - (parent_hydrogens - 2 * matching_count)
                        + len(witness.carbonyl_carbons))
                    if hydrogenation_count >= 0 and hydrogenation_count % 2 == 0:
                        if hydrogenation_count == 0:
                            hydrogen_state, reason = "complete", None
                        else:
                            multiplier = _HYDRO_MULTIPLIERS.get(hydrogenation_count)
                            if multiplier is not None:
                                hydro_prefix = multiplier + "hydro"
                                hydrogen_state, reason = "complete", None
                            else:
                                reason = "hydro prefix multiplier is outside the proved vocabulary"
                    else:
                        reason = "hydrogen balance is incompatible with pairwise hydro addition"
                signature = (name, hydro_prefix, original_map, hydrogen_state)
                found[signature] = CyclicDianhydrideParentConstruction(
                    name, hydro_prefix, network, original_map, core_original_atoms,
                    hydrogen_state=hydrogen_state,
                    hydrogen_state_reason=reason,
                    mancude_carbon_matchings=tuple(
                        tuple(sorted(matching)) for matching in matchings),
                    residual_carbon_matchings=tuple(
                        tuple(sorted(matching)) for matching in residual_matchings),
                    suffix_accommodation=suffix_accommodation)
    return tuple(found[key] for key in sorted(found))


def cyclic_dianhydride_parent_frontier(mol, Chem):
    """Return the P-25 construction frontier without choosing a tied name.

    The result is ``(status, records)`` where status is ``complete``,
    ``incomplete``, or ``unavailable``. Comparison is solely over the
    independent fusion-network structures generated from the input graph.
    A complete parent frontier still needs the P-25 peripheral-numbering
    orientation before suffix atoms can be assigned name locants.
    """
    records = cyclic_dianhydride_parent_constructions(mol, Chem)
    if not records:
        return "unavailable", ()
    from .fusion_networks import network_criteria

    def compare(left, right):
        for _, a, b in network_criteria(left.fusion_network, right.fusion_network):
            if a is None or b is None:
                return None
            if a != b:
                return -1 if a < b else 1
        return 0

    unique = {}
    for record in records:
        unique.setdefault(record.fused_parent_name, record)
    names = tuple(sorted(unique))
    comparisons = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            result = compare(unique[left], unique[right])
            comparisons[left, right] = result
    if any(value is None for value in comparisons.values()):
        return "incomplete", records
    winners = []
    for name in names:
        dominated = False
        for other in names:
            if name == other:
                continue
            result = comparisons.get((name, other))
            if result is None:
                inverse = comparisons.get((other, name))
                result = None if inverse is None else -inverse
            if result is None:
                return "incomplete", records
            if result > 0:
                dominated = True
                break
        if not dominated:
            winners.extend(record for record in records if record.fused_parent_name == name)
    if not winners:
        return "incomplete", records
    return "complete", tuple(winners)


def cyclic_dianhydride_decision_frontier(mol, Chem):
    """Build mapped tetrone decisions from all source-admissible P-25 maps.

    The source geometry frontier supplies orientation seeds; the numbering
    frontier applies P-25.3.3.1. A complete decision is returned only if every
    surviving parent and numbering candidate renders the same name. Symmetry
    equivalent original-atom maps are retained as a sidecar; one representative
    map is serialized only after name consensus and is not claimed unique.
    Any eligible but unrenderable candidate blocks the decision.
    """
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fused_orientation import fused_orientation_frontier

    witness = cyclic_dianhydride_graph(mol)
    if witness is None:
        return CyclicDianhydrideDecisionFrontier("outside_scope", "graph operation not admitted")
    parent_status, parents = cyclic_dianhydride_parent_frontier(mol, Chem)
    if parent_status != "complete" or not parents:
        return CyclicDianhydrideDecisionFrontier(
            "blocked", "edge-fused parent frontier is incomplete or unavailable")
    if any(parent.hydrogen_state != "complete" for parent in parents):
        return CyclicDianhydrideDecisionFrontier(
            "blocked", "mancude matching leaves indicated-H/added-H locants unresolved")

    expected_core = frozenset(range(mol.GetNumAtoms())) - witness.carbonyl_oxygens
    if any(frozenset(parent.core_original_atoms) != expected_core for parent in parents):
        return CyclicDianhydrideDecisionFrontier("blocked", "parent construction map is incomplete")
    orientation = fused_orientation_frontier(mol, core_atoms=expected_core)
    if not orientation.admitted or not orientation.orientation_seeds:
        return CyclicDianhydrideDecisionFrontier(
            "blocked", "eligible fused parent has no source-admitted orientation frontier")
    numbering = peripheral_numbering_frontier(
        mol, core_atoms=expected_core, orientation_seeds=orientation.orientation_seeds)
    if not numbering.admitted or not numbering.candidates:
        return CyclicDianhydrideDecisionFrontier(
            "blocked", "source orientation has no complete P-25 peripheral numbering")

    required_heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if frozenset(dict(numbering.candidates[0].atom_locants)) != expected_core:
        return CyclicDianhydrideDecisionFrontier("blocked", "numbering map does not cover the parent core")
    if not witness.carbonyl_carbons <= expected_core:
        return CyclicDianhydrideDecisionFrontier("blocked", "a carbonyl center is outside the numbered core")

    candidates = []
    for parent in parents:
        for mapped in numbering.candidates:
            locants = dict(mapped.atom_locants)
            suffix_locants = tuple(sorted(
                (locants[center] for center in witness.carbonyl_carbons),
                key=_numbered_locant_key))
            if len(suffix_locants) != 4 or len(set(suffix_locants)) != 4:
                return CyclicDianhydrideDecisionFrontier(
                    "blocked", "tetrone centers do not have four distinct locants")
            parent_text = parent.fused_parent_name.replace("'", "′")
            suffix = ",".join(suffix_locants) + "-tetrone"
            complete_name = parent.hydrogenation_prefix + parent_text + "-" + suffix
            core_mapping = tuple(sorted(locants.items(), key=lambda row: (row[1], row[0])))
            candidates.append(CyclicDianhydrideNameCandidate(
                complete_name, parent, core_mapping, suffix_locants))

    if not candidates:
        return CyclicDianhydrideDecisionFrontier("blocked", "no complete name candidates")
    names = {candidate.name for candidate in candidates}
    if len(names) != 1:
        return CyclicDianhydrideDecisionFrontier(
            "ambiguous", "source-admissible complete candidates disagree",
            tuple(candidates))
    # The original carbonyl oxygens are local to their mapped ketone suffixes;
    # the parent owns all other heavy atoms. Keep the two role sets disjoint.
    core_atoms = expected_core
    suffix_oxygens = witness.carbonyl_oxygens
    if (core_atoms & suffix_oxygens or core_atoms | suffix_oxygens != required_heavy):
        return CyclicDianhydrideDecisionFrontier(
            "blocked", "parent and tetrone-oxygen roles do not cover the heavy graph",
            tuple(candidates))

    complete_maps = []
    for candidate in candidates:
        suffix_locant_by_center = {
            center: dict(candidate.atom_locants)[center]
            for center in witness.carbonyl_carbons
        }
        suffix_map = tuple(sorted(
            ((oxygen.GetIdx(), suffix_locant_by_center[center])
             for center in witness.carbonyl_carbons
             for oxygen in mol.GetAtomWithIdx(center).GetNeighbors()
             if oxygen.GetIdx() in suffix_oxygens),
            key=lambda row: (row[1], row[0])))
        if len(suffix_map) != 4:
            return CyclicDianhydrideDecisionFrontier(
                "blocked", "each tetrone oxygen must map to one suffix center", tuple(candidates))
        complete_maps.append(tuple(sorted(candidate.atom_locants + suffix_map,
                                          key=lambda row: (row[1], row[0]))))
    # All candidates already agree on the preferred text. The representative
    # map below is only a serialization choice; every symmetry-equivalent map
    # remains attached to the decision and is not asserted chemically unique.
    equivalent_maps = tuple(dict.fromkeys(complete_maps))
    representative = min(equivalent_maps)
    representative_core = tuple(row for row in representative if row[0] in core_atoms)
    representative_suffix = tuple(row for row in representative if row[0] in suffix_oxygens)
    roles = (
        ("fused-pseudoketone-parent", core_atoms, representative_core),
        ("tetrone-suffix-oxygens", suffix_oxygens, representative_suffix),
    )
    decision = CyclicDianhydrideMethodDecision(
        name=candidates[0].name, method="pseudoketone",
        principal_centers=witness.carbonyl_carbons,
        atoms=required_heavy,
        rule="P-65.7.7.2/P-64.3.1/P-25.3.2.3/P-25.3.3.1/P-14.4",
        atom_locants=representative,
        component_roles=roles,
        equivalent_atom_locants=equivalent_maps,
    )
    return CyclicDianhydrideDecisionFrontier("complete", None, tuple(candidates), decision)


def _numbered_locant_key(locant):
    import re

    match = re.fullmatch(r"(\d+)([a-z]*)", locant)
    return (int(match[1]), match[2]) if match else (10**9, locant)

