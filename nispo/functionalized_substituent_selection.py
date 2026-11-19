"""Source-ordered P-29/P-46 rooted path facts for functionalized C trees.

This module selects graph paths and positional method facts only. It does not
render a substituent or join an enclosing parent. Exact pendant-role
admission is delegated to the shared immutable role partition; every returned
path retains original atom indices and writer obligations for off-path arms.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations


class FunctionalizedSubstituentStatus(str, Enum):
    OUTSIDE_SCOPE = "outside_scope"
    PENDING = "pending"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class CompoundSubstituentUnit:
    """One typed pendant role or off-path carbon component on a candidate."""

    kind: str
    host: int
    root: int | None
    atoms: tuple[int, ...]
    token: str | None
    locant: int
    source_rule: str
    writer_required: bool = False
    role_atoms: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class P46CriterionFact:
    rule: str
    value: object
    applicable: bool
    proof: str


@dataclass(frozen=True, slots=True)
class FunctionalizedSubstituentPath:
    path_atom_ids: tuple[int, ...]
    root: int
    root_locant: int
    method: int
    units: tuple[CompoundSubstituentUnit, ...]
    criteria: tuple[P46CriterionFact, ...]
    branch_writer_obligations: tuple[CompoundSubstituentUnit, ...]
    original_graph_signature: tuple
    role_partition: object


@dataclass(frozen=True, slots=True)
class PathComparison:
    preference: int | None
    rule: str
    reason: str


@dataclass(frozen=True, slots=True)
class FunctionalizedSubstituentFrontier:
    status: FunctionalizedSubstituentStatus
    candidates: tuple[FunctionalizedSubstituentPath, ...] = ()
    reason: str = ""
    evidence: tuple = ()
    comparisons: tuple[tuple[int, int, PathComparison], ...] = ()
    universe_complete: bool = False


@dataclass(frozen=True, slots=True)
class BranchWriterEvidence:
    root: int
    host: int
    atoms: tuple[int, ...]
    text: str


@dataclass(frozen=True, slots=True)
class BranchWriterClosure:
    complete: bool
    reason: str
    evidence: tuple = ()


_P46_RULES = (
    "P-46.1(a)", "P-46.1(b)", "P-46.1(c)", "P-46.1(d)",
    "P-46.1(e)", "P-46.1(f)", "P-46.1(g)", "P-46.1(h)",
    "P-46.1(i)", "P-46.1(j)", "P-46.1(k)", "P-46.1(l)",
    "P-46.1(m)",
)


def functionalized_substituent_frontier(mol, Chem, root, parent_host, *, role_partition=None,
                                        resolve_branch_tokens=False):
    """Return the P-29/P-46 rooted path frontier, without rendering names.

    The shared role-admission helper validates source inventory, pendant role
    atoms and the unique attachment cut. This function then independently
    revalidates that record against the current graph and proves an acyclic
    carbon-tree skeleton before enumerating paths that contain ``root``.
    """
    if type(root) is not int or type(parent_host) is not int:
        return _outside("root and parent host must be original integer atom indices")
    from .molecule_cache import exact_graph_computation
    from .functionalized_carbon_prefixes import (
        functionalized_carbon_role_partition,
        validate_functionalized_carbon_role_partition,
    )
    return exact_graph_computation(
        mol, Chem, 'functionalized_substituent_frontier',
        (Chem, functionalized_carbon_role_partition,
         validate_functionalized_carbon_role_partition,
         root, parent_host, role_partition, resolve_branch_tokens),
        lambda: _functionalized_substituent_frontier(
            mol, Chem, root, parent_host, role_partition=role_partition,
            resolve_branch_tokens=resolve_branch_tokens),
    )


def _functionalized_substituent_frontier(mol, Chem, root, parent_host, *, role_partition=None,
                                         resolve_branch_tokens=False):
    from .extended_inventory import selection_inventory
    from .functionalized_carbon_prefixes import (
        functionalized_carbon_role_partition,
        validate_functionalized_carbon_role_partition,
    )

    inventory = selection_inventory(mol)
    if role_partition is None:
        partition, admission = functionalized_carbon_role_partition(
            mol, Chem, root, parent_host, tuple(inventory.principal),
        )
    else:
        partition, admission = role_partition, "complete"
        if (partition.root != root or partition.parent_host != parent_host
                or not validate_functionalized_carbon_role_partition(
                    mol, Chem, partition, tuple(inventory.principal))):
            return _pending("supplied shared role partition failed original-graph validation")
    if admission == "outside":
        return _outside("component does not meet the shared functionalized-role scope")
    if admission != "complete" or partition is None:
        return _pending("typed pendant-role/source inventory is incomplete",
                        evidence=(admission,))
    if not validate_functionalized_carbon_role_partition(
            mol, Chem, partition, tuple(inventory.principal)):
        return _pending("shared role partition failed original-graph revalidation")
    if (partition.root != root or partition.parent_host != parent_host
            or partition.boundary_cut != (root, parent_host)
            or partition.scope != partition.carbon_atoms | frozenset(
                atom for role in partition.roles for atom in role.atoms)
            or partition.carbon_atoms & frozenset(
                atom for role in partition.roles for atom in role.atoms)):
        return _pending("shared role partition does not form an exact disjoint atom map")
    carbon = frozenset(partition.carbon_atoms)
    if root not in carbon:
        return _pending("free-valence root is not part of the carbon skeleton")

    adjacency = {idx: set() for idx in carbon}
    for idx in carbon:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons()
                or atom.GetIsAromatic() or atom.IsInRing()
                or atom.GetTotalValence() != 4):
            return _outside("skeleton is not a neutral acyclic carbon graph")
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            if neighbor.GetAtomicNum() == 1:
                if neighbor.GetIsotope():
                    return _outside("isotopic hydrogen is outside the saturated carbon-tree scope")
                continue
            bond = mol.GetBondBetweenAtoms(idx, other)
            if other in carbon:
                if bond.GetBondTypeAsDouble() not in {1,2,3}:
                    return _outside("carbon skeleton contains an unsupported bond")
                adjacency[idx].add(other)
            elif other in partition.scope:
                carbonyl_role = any(role.host == idx and other in role.atoms
                                    and role.kind in {"ketone", "aldehyde"}
                                    for role in partition.roles)
                if (bond.GetBondType() != Chem.BondType.SINGLE
                        and not (carbonyl_role and bond.GetBondTypeAsDouble() == 2)):
                    return _pending("pendant-role attachment has unsupported bond order")
            elif (idx, other) != (root, parent_host) and (other, idx) != (root, parent_host):
                return _pending("unowned heavy-atom attachment touches the carbon skeleton")
    if not _is_tree(adjacency):
        return _outside("the admitted carbon skeleton is not one connected acyclic tree")
    if any(sum(mol.GetBondBetweenAtoms(idx,j).GetBondTypeAsDouble()>1
               for j in neighbors)>1 for idx,neighbors in adjacency.items()):
        return _outside("cumulated multiple bonds require a separate component grammar")
    if any(role.host not in carbon for role in partition.roles):
        return _pending("a recognized pendant role has no carbon-skeleton host")

    paths = _root_containing_leaf_paths(root, adjacency)
    if not paths:
        return _pending("the complete rooted path universe could not be enumerated")
    maximum_rooted_length = max(map(len, paths))
    # P-29.2/P-46.0 locate the free valence on the chosen principal
    # substituent chain in both methods. In a tree, any root-containing
    # maximum can be extended at both ends to leaves without losing the
    # root. A longer unrooted diameter that omits the root therefore cannot
    # be a competitor to these completely enumerated source candidates.
    # Scope this extension to the saturated, unannotated family for which
    # subsequent P-46 facts and original-atom rendering are source-closed.
    # Annotated or unsaturated root-excluded cases retain their existing
    # barrier until their cross-path P-46.2/P-46.3 and pi-state joins are
    # assessed independently; a rooted-topology proof alone cannot close them.
    advanced = any(mol.GetAtomWithIdx(idx).GetIsotope()
                   or int(mol.GetAtomWithIdx(idx).GetChiralTag())
                   for idx in partition.scope)
    if not advanced:
        advanced = any(mol.GetBondBetweenAtoms(idx, other).GetBondTypeAsDouble() != 1
                       or int(mol.GetBondBetweenAtoms(idx, other).GetStereo())
                       for idx, neighbors in adjacency.items() for other in neighbors
                       if idx < other)
    if advanced:
        global_diameter = _tree_diameter_length(adjacency)
        if global_diameter > maximum_rooted_length:
            return _pending(
                "an annotated or unsaturated root-excluded diameter needs a separate P-46 proof",
                evidence=(("rooted_maximum_atoms", maximum_rooted_length),
                          ("unrooted_diameter_atoms", global_diameter)),
                universe_complete=False,
            )

    maxima = tuple(path for path in paths if len(path) == maximum_rooted_length)
    oriented = []
    for path in maxima:
        if path[0] == root or path[-1] == root:
            orientations = (path if path[0] == root else tuple(reversed(path)),)
        else:
            orientations = (path, tuple(reversed(path)))
        for ordered in orientations:
            candidate = _make_candidate(mol, partition, ordered, adjacency)
            if candidate is None:
                return _pending("maximum rooted path lacks complete P-46 role facts",
                                evidence=(tuple(path),))
            oriented.append(candidate)
    if resolve_branch_tokens:
        # P-46.1(a-l) are graph facts. Discard strictly junior paths before
        # asking any arm writer for the names needed by P-46.1(m). A missing
        # writer on a junior path must not suppress a complete maximum.
        early, _ = _frontier(oriented, include_citation=False)
        resolved = []
        from .functionalized_carbon_prefixes import functionalized_arm_prefix
        for candidate in early:
            units = []
            for unit in candidate.units:
                if unit.writer_required:
                    text, owned = functionalized_arm_prefix(mol, Chem, unit)
                    if not text or frozenset(owned) != frozenset(unit.atoms):
                        return _pending("a graph-maximum arm lacks a complete mapped writer",
                                        universe_complete=True)
                    unit = replace(unit, token=text)
                units.append(unit)
            units = tuple(units)
            criteria = tuple(
                replace(fact, value=_alphabetical_locants(units))
                if fact.rule == "P-46.1(m)" else fact
                for fact in candidate.criteria
            )
            resolved.append(replace(
                candidate, units=units, criteria=criteria,
                branch_writer_obligations=tuple(unit for unit in units if unit.writer_required),
            ))
        oriented = resolved
    survivors, comparisons = _frontier(oriented)
    if survivors is None:
        return FunctionalizedSubstituentFrontier(
            FunctionalizedSubstituentStatus.PENDING, tuple(oriented),
            "unknown P-46 comparison survives the proven earlier criteria",
            evidence=("P-46.1 order preserved; no path chosen by rendering or atom order",),
            comparisons=comparisons, universe_complete=True,
        )
    return FunctionalizedSubstituentFrontier(
        FunctionalizedSubstituentStatus.COMPLETE, survivors,
        "complete rooted maximum path frontier; enclosing name not rendered",
        evidence=("P-29 method classification is positional",
                  "P-46.1 criteria applied in source order"),
        comparisons=comparisons, universe_complete=True,
    )


def validate_branch_writer_closure(mol, Chem, candidate, writer_evidence):
    """Validate post-selection writer ownership for every candidate arm.

    This is deliberately separate from path selection. A consumer may call a
    real mapped writer after graph maxima are known, then provide its exact
    root/host/atom result here. Missing or extra results keep the selected
    maximum pending; this function never promotes a junior path.
    """
    from .functionalized_carbon_prefixes import (
        FunctionalizedCarbonRolePartition,
        _graph_signature,
    )
    partition = candidate.role_partition
    if (not isinstance(partition, FunctionalizedCarbonRolePartition)
            or candidate.original_graph_signature != _graph_signature(mol)):
        return BranchWriterClosure(False, "candidate graph or source-role proof is stale")
    rebuilt_frontier = functionalized_substituent_frontier(
        mol, Chem, partition.root, partition.parent_host, role_partition=partition,
        resolve_branch_tokens=True,
    )
    if (rebuilt_frontier.status is not FunctionalizedSubstituentStatus.COMPLETE
            or candidate not in rebuilt_frontier.candidates):
        return BranchWriterClosure(False, "candidate is not a rebuilt member of the live graph maximum frontier")
    obligations = candidate.branch_writer_obligations
    evidence = tuple(writer_evidence)
    expected = {(item.root, item.host, item.atoms): item for item in obligations}
    observed = {}
    for row in evidence:
        if not isinstance(row, BranchWriterEvidence) or not row.text:
            return BranchWriterClosure(False, "maximum branch writer returned no complete typed result")
        key = (row.root, row.host, tuple(row.atoms))
        if key in observed:
            return BranchWriterClosure(False, "duplicate branch writer ownership result")
        observed[key] = row
    if set(observed) != set(expected):
        return BranchWriterClosure(False, "maximum branch writer ownership is incomplete or overbroad")
    for key, obligation in expected.items():
        proof = observed[key]
        atoms = set(proof.atoms)
        if (obligation.token is None
                or not atoms or proof.root not in atoms or proof.host in atoms
                or obligation.token is not None and proof.text != obligation.token):
            return BranchWriterClosure(False, "branch writer lacks a closed source token or exact atom ownership")
        carbon_atoms = atoms - set(obligation.role_atoms)
        if (not carbon_atoms
                or any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in carbon_atoms)
                or not _single_attachment_component(mol, atoms, proof.host,
                                                    role_atoms=obligation.role_atoms)):
            return BranchWriterClosure(False, "branch writer atoms are not one exact singly attached carbon component")
    return BranchWriterClosure(True, "all graph-maximum branch writers match exact atom ownership",
                               evidence=tuple((r.root, r.host, r.atoms, r.text) for r in evidence))


def _make_candidate(mol, partition, path, adjacency):
    path = tuple(path)
    path_set = set(path)
    root = partition.root
    root_locant = path.index(root) + 1
    unsaturated = any(mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble()>1
                      for a,b in zip(path,path[1:]))
    method = 1 if root_locant == 1 and not unsaturated else 2
    locants = {atom: index + 1 for index, atom in enumerate(path)}
    units = []
    for role in partition.roles:
        if role.host in path_set:
            units.append(CompoundSubstituentUnit(
                "pendant-role", role.host, role.atoms[0] if role.atoms else None,
                tuple(role.atoms), role.token, locants[role.host], role.rule, False,
                tuple(role.atoms),
            ))
    for atoms, host, root_atom in _off_path_components(path_set, adjacency):
        if host not in path_set or root_atom not in atoms:
            return None
        roles_inside = tuple(role for role in partition.roles if role.host in atoms)
        root_degree = sum(neighbor in atoms for neighbor in adjacency[root_atom])
        token = (_linear_alkyl_token(len(atoms))
                 if not roles_inside and root_degree <= 1
                 and _is_linear_component(atoms, adjacency)
                 and all(mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble()==1
                         for a in atoms for b in adjacency[a] if b in atoms) else None)
        role_atoms = tuple(sorted(atom for role in roles_inside for atom in role.atoms))
        units.append(CompoundSubstituentUnit(
            "carbon-arm" if not roles_inside else "functionalized-carbon-arm",
            host, root_atom, tuple(sorted(set(atoms) | set(role_atoms))), token,
            locants[host], "P-29.4.1/P-46.1", True, role_atoms,
        ))
    # Roles hosted off path are represented inside their complete carbon arm,
    # never counted a second time as a direct substituent on this candidate.
    direct_host_roles = sum(role.host in path_set for role in partition.roles)
    direct_arms = len(units) - direct_host_roles
    units = tuple(sorted(units, key=lambda unit: (unit.locant, unit.kind,
                                                   unit.root if unit.root is not None else -1,
                                                   unit.atoms)))
    alpha_locants = _alphabetical_locants(units)
    orders = tuple(mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble()
                   for a,b in zip(path,path[1:]))
    multiple = tuple(i for i,order in enumerate(orders,1) if order>1)
    double = tuple(i for i,order in enumerate(orders,1) if order==2)
    values = {
        "P-46.1(a)": 0,
        "P-46.1(b)": -len(path),
        "P-46.1(c)": (),
        "P-46.1(d)": (-len(multiple), -len(double)),
        "P-46.1(e)": 0,
        "P-46.1(f)": (),
        "P-46.1(g)": (),
        "P-46.1(h)": root_locant,
        "P-46.1(i)": (multiple, double),
        "P-46.1(j)": (),
        "P-46.1(k)": -(direct_host_roles + direct_arms),
        "P-46.1(l)": tuple(sorted(unit.locant for unit in units)),
        "P-46.1(m)": alpha_locants,
    }
    facts = tuple(P46CriterionFact(
        rule, values[rule],
        rule not in {"P-46.1(a)", "P-46.1(f)", "P-46.1(g)"} or method == 2,
        _criterion_proof(rule, method),
    ) for rule in _P46_RULES)
    signature = partition.original_graph_signature
    obligations = tuple(unit for unit in units if unit.writer_required)
    return FunctionalizedSubstituentPath(
        path, root, root_locant, method, units, facts, obligations,
        signature, partition,
    )


def _criterion_proof(rule, method):
    if rule == "P-46.1(a)":
        return "P-46.1(a) counts skeletal heteroatoms only in method (2); this all-carbon path has zero"
    if rule == "P-46.1(b)":
        return "P-46.1(b) compares the number of skeletal atoms in the candidate chain"
    if rule == "P-46.1(c)":
        return "P-46.1(c) heteroatom seniority ties because both paths contain no heteroatoms"
    if rule == "P-46.1(d)":
        return "P-46.1(d) compares mapped multiple-bond counts, then double-bond counts"
    if rule == "P-46.1(e)":
        return "P-46.1(e) count of atoms with nonstandard bonding numbers; none occur"
    if rule == "P-46.1(f)":
        return "P-46.1(f) method-(2)-only lowest heteroatom locants; this path has none"
    if rule == "P-46.1(g)":
        return "P-46.1(g) method-(2)-only lowest locants for heteroatoms by seniority; this path has none"
    if rule == "P-46.1(h)":
        return "P-46.1(h) lowest locants for original free valences"
    if rule == "P-46.1(i)":
        return "P-46.1(i) lowest mapped multiple-bond locants, then double-bond locants"
    if rule == "P-46.1(j)":
        return "P-46.1(j) lowest locants for atoms with nonstandard bonding numbers; none occur"
    if rule == "P-46.1(k)":
        return "count of distinct direct pendant groups and singly attached components"
    if rule == "P-46.1(l)":
        return "complete sorted locant multiset of direct substituent units"
    if rule == "P-46.1(m)":
        return "alphabetically ordered substituent locants; unknown arm names remain unresolved"
    return ""


def _frontier(candidates, *, include_citation=True):
    beaten = set()
    unknown = set()
    comparisons = []
    for left_index, right_index in combinations(range(len(candidates)), 2):
        comparison = compare_functionalized_paths(candidates[left_index], candidates[right_index],
                                                  include_citation=include_citation)
        comparisons.append((left_index, right_index, comparison))
        if comparison.preference == -1:
            beaten.add(right_index)
        elif comparison.preference == 1:
            beaten.add(left_index)
        elif comparison.preference is None:
            unknown.add((left_index, right_index))
    survivors = tuple(index for index in range(len(candidates)) if index not in beaten)
    if any(left in survivors and right in survivors for left, right in unknown):
        return None, tuple(comparisons)
    return tuple(candidates[index] for index in survivors), tuple(comparisons)


def compare_functionalized_paths(left, right, *, include_citation=True):
    """Compare candidate paths at the first proved P-46 criterion."""
    if (left.original_graph_signature != right.original_graph_signature
            or left.role_partition != right.role_partition):
        return PathComparison(None, "P-46", "Candidates do not share one validated graph/role partition")
    left_facts = {fact.rule: fact for fact in left.criteria}
    right_facts = {fact.rule: fact for fact in right.criteria}
    for rule in _P46_RULES:
        if rule == "P-46.1(m)" and not include_citation:
            break
        a, b = left_facts[rule], right_facts[rule]
        if not a.applicable and not b.applicable:
            continue
        if a.value is None or b.value is None:
            return PathComparison(None, rule, "A required ordered criterion is unknown")
        if a.value != b.value:
            return PathComparison(-1 if a.value < b.value else 1, rule,
                                  "First decisive proved P-46 criterion")
    return PathComparison(0, "tie", "All recorded applicable P-46 criteria tie")


def _alphabetical_locants(units):
    if any(unit.token is None for unit in units):
        return None
    from .formatting import prefix_alphabetization_key
    ordered = sorted(units, key=lambda unit: (prefix_alphabetization_key(unit.token)[:3],
                                               unit.locant))
    return tuple(unit.locant for unit in ordered)


def _off_path_components(path, adjacency):
    remaining = set(adjacency) - path
    while remaining:
        seed = min(remaining)
        atoms, stack, boundary = set(), [seed], set()
        while stack:
            current = stack.pop()
            if current not in remaining:
                continue
            remaining.remove(current)
            atoms.add(current)
            for neighbor in adjacency[current]:
                if neighbor in path:
                    boundary.add(neighbor)
                elif neighbor in remaining:
                    stack.append(neighbor)
        if len(boundary) != 1:
            yield frozenset(atoms), -1, -1
            return
        host = next(iter(boundary))
        roots = sorted(idx for idx in atoms if host in adjacency[idx])
        if len(roots) != 1:
            yield frozenset(atoms), host, -1
            return
        yield frozenset(atoms), host, roots[0]


def _linear_alkyl_token(atom_count):
    from .constants import ROOTS
    root = ROOTS.get(atom_count)
    return root + "yl" if root else None


def _is_linear_component(atoms, adjacency):
    degrees = [sum(neighbor in atoms for neighbor in adjacency[idx]) for idx in atoms]
    return (len(atoms) == 1 or
            all(degree <= 2 for degree in degrees) and sum(degree == 1 for degree in degrees) == 2)


def _single_attachment_component(mol, atoms, host, *, role_atoms=()):
    atoms = set(atoms)
    boundary = {(idx, n.GetIdx()) for idx in atoms for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms}
    if len(boundary) != 1 or next(iter(boundary))[1] != host:
        return False
    # Pendant atoms belong to this same singly attached component, but do
    # not become skeletal carbons when checking the off-path carbon tree.
    atoms -= set(role_atoms)
    if any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in atoms):
        return False
    edges = sum(sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(idx).GetNeighbors())
                for idx in atoms) // 2
    if edges != len(atoms) - 1:
        return False
    seen, stack = set(), [next(iter(atoms))]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                     if n.GetIdx() in atoms)
    return seen == atoms


def _root_containing_leaf_paths(root, adjacency):
    leaves = tuple(sorted(idx for idx, neighbors in adjacency.items() if len(neighbors) <= 1))
    paths = set()
    if len(adjacency) == 1:
        return ((root,),)
    for left, right in combinations(leaves, 2):
        path = _tree_path(left, right, adjacency)
        if root in path:
            paths.add(path)
    return tuple(sorted(paths))


def _tree_diameter_length(adjacency):
    leaves = tuple(sorted(idx for idx, neighbors in adjacency.items() if len(neighbors) <= 1))
    if len(leaves) <= 1:
        return 1
    return max(len(_tree_path(a, b, adjacency)) for a, b in combinations(leaves, 2))


def _tree_path(start, end, adjacency):
    predecessor = {start: None}
    stack = [start]
    for current in stack:
        if current == end:
            break
        for neighbor in sorted(adjacency[current]):
            if neighbor not in predecessor:
                predecessor[neighbor] = current
                stack.append(neighbor)
    if end not in predecessor:
        return ()
    path, current = [], end
    while current is not None:
        path.append(current)
        current = predecessor[current]
    return tuple(reversed(path))


def _is_tree(adjacency):
    if not adjacency:
        return False
    edges = sum(len(row) for row in adjacency.values()) // 2
    if edges != len(adjacency) - 1:
        return False
    seen, stack = set(), [next(iter(adjacency))]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adjacency[current] - seen)
    return len(seen) == len(adjacency)


def _outside(reason):
    return FunctionalizedSubstituentFrontier(
        FunctionalizedSubstituentStatus.OUTSIDE_SCOPE, reason=reason,
    )


def _pending(reason, evidence=(), universe_complete=False):
    return FunctionalizedSubstituentFrontier(
        FunctionalizedSubstituentStatus.PENDING, reason=reason,
        evidence=tuple(evidence), universe_complete=universe_complete,
    )
