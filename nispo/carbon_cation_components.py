"""Mapped saturated carbon-tree cations formed by formal hydride loss.

This bounded component producer implements P-73.2.2.1 for one carbon
ylium center in an acyclic saturated carbon parent. Hydron-added cations,
heteroatom cations, rings, and annotated structures are outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


RULE = "P-73.2.2.1.1/P-73.2.2.1.2/P-73.7/P-44.3/P-44.4/P-14.3.4.4/P-14.4/P-15.1.7.1/P-45.2"


@dataclass(frozen=True, slots=True)
class CarbonCationComponent:
    """One fully owned mapped carbon cation component."""

    name: str
    atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    charge: int
    rule: str
    proof: tuple
    component_roles: tuple
    parent_atoms: frozenset[int]
    equivalent_parent_mappings: tuple[tuple[int, ...], ...] = ()


def _graph_proof(mol, Chem):
    if (mol is None or mol.GetNumHeavyAtoms() == 0 or mol.GetNumHeavyAtoms() > 64
            or mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1):
        return None
    atoms = tuple(mol.GetAtoms())
    if (any(atom.GetAtomicNum() != 6 for atom in atoms)
            or any(atom.GetIsotope() or int(atom.GetChiralTag())
                   or atom.GetNumRadicalElectrons() for atom in atoms)
            or any(bond.GetBondType() != Chem.BondType.SINGLE
                   or int(bond.GetStereo()) or bond.GetBondDir() != Chem.BondDir.NONE
                   for bond in mol.GetBonds())):
        return None
    charged = [atom for atom in atoms if atom.GetFormalCharge()]
    if (len(charged) != 1 or charged[0].GetFormalCharge() != 1
            or sum(atom.GetFormalCharge() for atom in atoms) != 1):
        return None
    center = charged[0]
    vertices = frozenset(atom.GetIdx() for atom in atoms)
    adjacency = {idx: set() for idx in vertices}
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[left].add(right)
        adjacency[right].add(left)
    if sum(map(len, adjacency.values())) != 2 * (len(vertices) - 1):
        return None
    seen, stack = set(), [min(vertices)]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(adjacency[idx] - seen)
    if seen != set(vertices):
        return None
    # A tetravalent neutral carbon parent becomes a trivalent cation when one
    # hydride is removed. Validate that local composition from the original
    # graph rather than infer the operation from charge alone.
    for atom in atoms:
        degree = len(adjacency[atom.GetIdx()])
        expected_h = (3 if atom.GetIdx() == center.GetIdx() else 4) - degree
        expected_valence = 3 if atom.GetIdx() == center.GetIdx() else 4
        if (expected_h < 0 or atom.GetTotalNumHs(includeNeighbors=True) != expected_h
                or atom.GetTotalValence() != expected_valence):
            return None
    return vertices, adjacency, center.GetIdx()


def _tree_path(adjacency, start, end):
    stack = [(start, (start,))]
    while stack:
        current, path = stack.pop()
        if current == end:
            return path
        for neighbor in sorted(adjacency[current] - set(path), reverse=True):
            stack.append((neighbor, path + (neighbor,)))
    return ()


def _maximal_parent_paths(adjacency, center):
    leaves = sorted(idx for idx, neighbors in adjacency.items() if len(neighbors) <= 1)
    candidates = {(center,)} if len(adjacency) == 1 else set()
    for left, right in combinations(leaves, 2):
        path = _tree_path(adjacency, left, right)
        if path and center in path:
            candidates.add(path)
    if not candidates:
        return ()
    length = max(map(len, candidates))
    return tuple(sorted(path for path in candidates if len(path) == length))


def _linear_branch_path(mol, adjacency, branch_atoms, root, host):
    """Name only unbranched simple alkyl branches from their attachment end."""
    branch_atoms = set(branch_atoms)
    if not branch_atoms or any(len(adjacency[idx] & branch_atoms) > 2
                               for idx in branch_atoms):
        return None
    if root not in branch_atoms or host not in adjacency[root]:
        return None
    ends = [idx for idx in branch_atoms
            if len(adjacency[idx] & set(branch_atoms)) <= 1]
    if len(ends) != 2 and len(branch_atoms) > 1:
        return None
    terminal = next((idx for idx in ends if idx != root), root)
    path = _tree_path({idx: adjacency[idx] & branch_atoms for idx in branch_atoms},
                      root, terminal)
    if len(path) != len(branch_atoms):
        return None
    return tuple(path)


def _branch_rows(mol, Chem, adjacency, parent_path):
    parent = set(parent_path)
    remaining = set(adjacency) - parent
    rows = []
    from .branch_rules import parent_chain_name
    from .constants import ROOTS
    while remaining:
        root = min(remaining)
        component, stack, hosts = set(), [root], set()
        while stack:
            idx = stack.pop()
            if idx in component:
                continue
            component.add(idx)
            remaining.discard(idx)
            for neighbor in adjacency[idx]:
                if neighbor in parent:
                    hosts.add(neighbor)
                elif neighbor not in component:
                    stack.append(neighbor)
        if len(hosts) != 1:
            return None
        host = next(iter(hosts))
        roots = [idx for idx in component if host in adjacency[idx]]
        if len(roots) != 1:
            return None
        branch_root = roots[0]
        branch_path = _linear_branch_path(mol, adjacency, component, branch_root, host)
        if branch_path is None or len(branch_path) not in ROOTS:
            return None
        # The source branch is an ordinary, unsubstituted acyclic alkyl group.
        # The maximal parent operation retains the actual branch atoms.
        text = parent_chain_name(mol, branch_path, as_substituent=True)
        if not text:
            return None
        rows.append((parent_path.index(host) + 1, text, frozenset(component)))
    return tuple(rows)


def _name_for(mol, Chem, path, center, branch_rows):
    from .branch_rules import parent_chain_name
    from .constants import ROOTS
    from .formatting import complete_parent_name, format_prefixes, join_prefix_parent
    from .parent_selection import graph_parent_facts

    if len(path) not in ROOTS:
        return None
    stem = complete_parent_name(parent_chain_name(mol, path))
    if not stem:
        return None
    cation_locant = path.index(center) + 1
    specific_terminal = (
        not branch_rows and cation_locant in {1, len(path)}
        and all(len(mol.GetAtomWithIdx(idx).GetNeighbors()) <= 2 for idx in path)
    )
    if specific_terminal:
        # P-73.2.2.1.1 replaces the parent-hydride ending “ane” by “ylium”.
        suffix_parent = stem[:-3] + "ylium" if stem.endswith("ane") else None
        if not suffix_parent:
            return None
    else:
        from .formatting import attach_suffix
        suffix_parent = attach_suffix(stem, "ylium", (cation_locant,))
    prefix_rows = [(str(locant), text) for locant, text, _ in branch_rows]
    prefix = format_prefixes(prefix_rows)
    name = join_prefix_parent(prefix, suffix_parent) if prefix else suffix_parent
    if not name:
        return None
    # P-45.2 counts/cites prefix groups, not repeated copies of the same
    # group name. The shared graph facts also apply P-45.5 to the complete
    # candidate name (including multiplicative prefixes) after those earlier
    # locant criteria.
    neutral = Chem.Mol(mol)
    cation = neutral.GetAtomWithIdx(center)
    cation.SetFormalCharge(0)
    cation.SetNumExplicitHs(0)
    cation.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(neutral)
    except Exception:
        return None
    if neutral.GetAtomWithIdx(center).GetTotalNumHs(includeNeighbors=True) != (
        mol.GetAtomWithIdx(center).GetTotalNumHs(includeNeighbors=True) + 1
    ):
        return None
    parent_locants = {idx: str(position) for position, idx in enumerate(path, 1)}
    facts = graph_parent_facts(
        neutral, path, parent_locants, principal_class=(6, 0),
        suffix_centers=(center,), suffix_locants=(str(cation_locant),),
        prefixes=tuple(prefix_rows), kind="chain", name=name,
    )
    return (name, facts) if facts is not None else None


def carbon_cation_component(mol, Chem):
    """Return a complete mapped C+ylium component, or ``None`` on decline."""
    proof = _graph_proof(mol, Chem)
    if proof is None:
        return None
    vertices, adjacency, center = proof
    parent_paths = _maximal_parent_paths(adjacency, center)
    if not parent_paths:
        return None
    candidates, failed_max = [], False
    for raw_path in parent_paths:
        for path in (raw_path, tuple(reversed(raw_path))):
            branches = _branch_rows(mol, Chem, adjacency, path)
            if branches is None:
                failed_max = True
                continue
            named = _name_for(mol, Chem, path, center, branches)
            if named is None:
                failed_max = True
                continue
            name, facts = named
            candidates.append((name, tuple(path), branches, facts))
    # Never fall through to a lower-quality parent or orientation when any
    # maximum eligible candidate cannot be named and compared.
    if failed_max or not candidates:
        return None
    from .parent_selection import parent_frontier
    frontier = tuple(parent_frontier(candidates, lambda row: row[3]))
    if not frontier or len({row[0] for row in frontier}) != 1:
        # A still-distinct result needs more preferred-name evidence than this
        # bounded source combination supplies.
        return None
    selected_name = frontier[0][0]
    selected_paths = tuple(sorted({row[1] for row in frontier}))
    chosen = min(frontier, key=lambda row: row[1])
    path, branches = chosen[1], chosen[2]
    mapping = tuple((idx, str(position)) for position, idx in enumerate(path, 1))
    heavy = frozenset(vertices)
    roles = [("parent-chain", frozenset(path), mapping)]
    for locant in sorted({loc for loc, _, _ in branches}):
        owned = frozenset().union(*(atoms for loc, _, atoms in branches if loc == locant))
        if not owned:
            return None
        roles.append((f"alkyl-substituents-at-{locant}", owned, ()))
    role_atoms = [idx for _, atoms, _ in roles for idx in atoms]
    if len(role_atoms) != len(set(role_atoms)) or set(role_atoms) != set(heavy):
        return None
    if len({idx for idx, _ in mapping}) != len(mapping):
        return None
    return CarbonCationComponent(
        selected_name, heavy, mapping, 1, RULE,
        (("operation", "formal hydride loss from saturated carbon parent"),
         ("selected_parent_path", tuple(path)),
         ("maximal_parent_mappings", selected_paths)),
        tuple(roles), frozenset(path), selected_paths,
    )
