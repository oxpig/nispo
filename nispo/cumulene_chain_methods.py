"""Bounded mapped names for acyclic carbon chains with contiguous cumulenes.

This opt-in method constructs source-typed names and atom mappings. It accepts
only saturated acyclic carbon branches attached to a chain containing every
C=C edge, with all double bonds in one consecutive cumulene segment. One
ordinary alcohol suffix is admitted. Stereochemical or isotopic information is
outside the grammar and causes a decline.
"""
from collections import defaultdict, deque
from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import MULT, ROOTS
from .formatting import (attach_suffix, format_prefixes, join_prefix_parent,
                         prefix_alphabetization_key)
from .naming_methods import MethodDecision


_RULE = "P-31.1.1/P-31.1.2.2.1/P-33.2.1/P-44.1.1/P-44.3.2/P-44.4.1.1/P-44.4.1.8/P-44.4.1.10/P-45.2"
_ALKYL = {
    1: "methyl", 2: "ethyl", 3: "propyl", 4: "butyl", 5: "pentyl",
    6: "hexyl", 7: "heptyl", 8: "octyl", 9: "nonyl", 10: "decyl",
    11: "undecyl", 12: "dodecyl",
}


@dataclass(frozen=True, slots=True)
class CumuleneChainProposal:
    """A MethodDecision plus every source-tied original-atom numbering."""

    decision: MethodDecision
    numbering_alternatives: tuple[tuple[tuple[int, str], ...], ...]
    paths: tuple[tuple[int, ...], ...]

    @property
    def name(self):
        return self.decision.name

    @property
    def atoms(self):
        return self.decision.atoms

    @property
    def atom_locants(self):
        return self.decision.atom_locants

    @property
    def component_roles(self):
        return self.decision.component_roles


def cumulene_chain_decision(mol, Chem):
    """Return a bounded complete-chain proposal, or ``None`` if unproved.

    Carbon-chain candidates follow P-44 chain length first, then include all
    double bonds and the optional principal alcohol group. P-44 suffix locants
    precede unsaturation locants; P-45 prefix count and locants resolve the
    supported branch choices. Exact source ties retain every original atom map
    in ``numbering_alternatives`` rather than using atom-index order.
    """
    proof = _prove_graph(mol, Chem)
    if proof is None:
        return None
    carbons, double_edges, hydroxyl, adjacency = proof
    if not 2 <= len(double_edges) <= 12:
        return None
    if not _one_contiguous_double_bond_segment(double_edges):
        return None

    all_paths = _all_leaf_paths(adjacency)
    if not all_paths:
        return None
    # P-44.1.1 first maximizes parent occurrences of the principal group.
    # Only then does P-44.3.2 prefer the longer parent chain.
    suffix_paths = tuple(path for path in all_paths
                         if hydroxyl is None or hydroxyl in path)
    if not suffix_paths:
        return None
    longest = max(map(len, suffix_paths))
    paths = tuple(path for path in suffix_paths if len(path) == longest)
    # P-44.4.1.1 next prefers the chain with more multiple bonds. The method
    # only renders a complete cumulene parent, so if the senior chain cannot
    # include every C=C edge it declines rather than selecting a shorter one.
    maximum_double_count = max(len(set(double_edges) & _path_edges(path))
                              for path in paths)
    paths = tuple(path for path in paths
                  if len(set(double_edges) & _path_edges(path)) == maximum_double_count
                  and set(double_edges) <= _path_edges(path))
    if not paths:
        return None
    # An unrenderable longest candidate is a barrier: silently skipping it
    # could make a shorter or otherwise inferior renderable path appear best.
    branch_facts = {}
    for path in paths:
        branches = _branches_for_path(adjacency, double_edges, path)
        if branches is None:
            return None
        branch_facts[path] = branches
    candidates = []
    for path in paths:
        for oriented in (path, tuple(reversed(path))):
            locants = {atom: str(i) for i, atom in enumerate(oriented, 1)}
            double_locs = tuple(sorted(
                min(oriented.index(a), oriented.index(b)) + 1
                for a, b in double_edges
            ))
            suffix_locants = (oriented.index(hydroxyl) + 1,) if hydroxyl is not None else ()
            prefix_items = []
            branch_roles = []
            branches = branch_facts[path]
            for branch_root, host, branch_atoms, text, branch_path in branches:
                host_locant = str(oriented.index(host) + 1)
                prefix_items.append((host_locant, text))
                branch_roles.append((branch_root, host_locant, branch_atoms,
                                     text, branch_path))
            alpha = sorted({name for _loc, name in prefix_items},
                           key=prefix_alphabetization_key)
            prefix_citation_locants = tuple(
                tuple(sorted(int(loc) for loc, text in prefix_items if text == name))
                for name in alpha
            )
            key = (
                -len(oriented),
                -len(double_locs),
                suffix_locants,
                double_locs,
                -len(prefix_items),
                tuple(sorted(int(loc) for loc, _ in prefix_items)),
                prefix_citation_locants,
            )
            candidates.append((key, oriented, double_locs, suffix_locants,
                               tuple(prefix_items), tuple(branch_roles), locants))
    if not candidates:
        return None
    best_key = min(candidate[0] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate[0] == best_key]

    rendered = []
    for item in best:
        _key, oriented, double_locs, suffix_locants, prefix_items, role_branches, locants = item
        text = _render_name(len(oriented), double_locs, suffix_locants, prefix_items)
        if text is None:
            return None
        atom_locants = tuple((atom, locants[atom]) for atom in oriented)
        rendered.append((text, oriented, atom_locants, role_branches))
    names = {item[0] for item in rendered}
    if len(names) != 1:
        return None
    name = next(iter(names))
    alternatives = tuple(sorted({item[2] for item in rendered}))
    selected = alternatives[0] if len(alternatives) == 1 else ()
    component_roles = []
    # Roles describe one atom-disjoint representative component ownership
    # partition. Every source-tied numbering remains available independently
    # in ``numbering_alternatives``.
    representative = next(item for item in rendered if item[2] == alternatives[0])
    component_roles.append(("cumulene-parent", frozenset(representative[1]),
                            representative[2]))
    for _branch_root, host_locant, branch_atoms, text, branch_path in representative[3]:
        component_roles.append((f"alkyl-branch-at-{host_locant}:{text}",
                                branch_atoms,
                                tuple((atom, str(i))
                                      for i, atom in enumerate(branch_path, 1))))
    if hydroxyl is not None:
        oxygen = next(idx for idx in mol.GetAtomWithIdx(hydroxyl).GetNeighbors()
                      if idx.GetAtomicNum() == 8).GetIdx()
        component_roles.append(("hydroxy-suffix", frozenset({oxygen}),
                                ((oxygen, str(best[0][3][0])),)))
    heavy = frozenset(atom.GetIdx() for atom in iter_atoms(mol)
                      if atom.GetAtomicNum() > 1)
    if set().union(*(set(role[1]) for role in component_roles)) != set(heavy):
        return None
    decision = MethodDecision(
        name=name, method="substitutive",
        principal_centers=frozenset({hydroxyl}) if hydroxyl is not None else frozenset(),
        atoms=heavy, rule=_RULE, atom_locants=selected,
        component_roles=tuple(component_roles),
    )
    return CumuleneChainProposal(decision, alternatives,
                                 tuple(sorted({item[1] for item in rendered})))


def _prove_graph(mol, Chem):
    if mol.GetNumHeavyAtoms() > 32 or mol.GetStereoGroups():
        return None
    if any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons() or atom.GetIsotope()
           or int(atom.GetChiralTag()) for atom in iter_atoms(mol)):
        return None
    if any(atom.GetAtomicNum() == 1 and atom.GetIsotope() for atom in iter_atoms(mol)):
        return None
    if any(int(bond.GetStereo()) or int(bond.GetBondDir()) for bond in iter_bonds(mol)):
        return None
    heavy = {atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1}
    if not heavy:
        return None
    carbons = {atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() == 6}
    oxygens = {atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() == 8}
    if not carbons or len(oxygens) > 1 or heavy != carbons | oxygens:
        return None
    if any(atom.GetAtomicNum() not in {6, 8} for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1):
        return None
    if any((atom.GetAtomicNum() == 6 and atom.GetTotalValence() != 4)
           or (atom.GetAtomicNum() == 8 and atom.GetTotalValence() != 2)
           for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1):
        return None

    hydroxyl = None
    if oxygens:
        oxygen = mol.GetAtomWithIdx(next(iter(oxygens)))
        if (oxygen.GetDegree() != 1 or oxygen.GetTotalNumHs(includeNeighbors=True) != 1
                or oxygen.GetNeighbors()[0].GetAtomicNum() != 6):
            return None
        hydroxyl = oxygen.GetNeighbors()[0].GetIdx()
        bond = mol.GetBondBetweenAtoms(oxygen.GetIdx(), hydroxyl)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None

    adjacency = {idx: set() for idx in carbons}
    double_edges = set()
    for bond in iter_bonds(mol):
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in carbons and b in carbons:
            order = bond.GetBondType()
            if order not in {Chem.BondType.SINGLE, Chem.BondType.DOUBLE}:
                return None
            adjacency[a].add(b)
            adjacency[b].add(a)
            if order == Chem.BondType.DOUBLE:
                double_edges.add(frozenset({a, b}))
        elif (a in oxygens and b == hydroxyl) or (b in oxygens and a == hydroxyl):
            continue
        elif a in heavy and b in heavy:
            return None
    if any(len(neighbors) > 3 for neighbors in adjacency.values()):
        return None
    if not _is_tree(adjacency):
        return None

    return carbons, double_edges, hydroxyl, adjacency


def _branches_for_path(adjacency, double_edges, path):
    branches = []
    parent_atoms = set(path)
    seen = set()
    for host in path:
        for root in adjacency[host]:
            if root in parent_atoms or root in seen:
                continue
            component = _component_away_from(adjacency, root, host)
            seen.update(component)
            if any(frozenset({a, b}) in double_edges
                   for a in component for b in adjacency[a] if b in component):
                return None
            if len(component) > 12:
                return None
            if len(component) == 1:
                branch_path = (root,)
            else:
                endpoints = [idx for idx in component if idx != root
                             and sum(1 for n in adjacency[idx] if n in component) <= 1]
                if len(endpoints) != 1:
                    return None
                branch_path = _path_between(adjacency, root, endpoints[0], component)
            if (not branch_path or set(branch_path) != component
                    or any(sum(1 for n in adjacency[idx] if n in component) > 2
                           for idx in component)):
                return None
            text = _ALKYL.get(len(component))
            if text is None:
                return None
            branches.append((root, host, frozenset(component), text, branch_path))
    return tuple(branches)


def _component_away_from(adjacency, root, blocked):
    seen, stack = set(), [root]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(neighbor for neighbor in adjacency[idx]
                     if neighbor != blocked and neighbor not in seen)
    return seen


def _all_leaf_paths(adjacency):
    leaves = [idx for idx, neighbors in adjacency.items() if len(neighbors) <= 1]
    paths = set()
    for i, left in enumerate(leaves):
        for right in leaves[i + 1:]:
            path = _path_between(adjacency, left, right, set(adjacency))
            if path:
                paths.add(path)
    return tuple(sorted(paths))


def _path_between(adjacency, start, end, allowed):
    parent = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == end:
            break
        for neighbor in adjacency[current]:
            if neighbor in allowed and neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)
    if end not in parent:
        return ()
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent[current]
    return tuple(reversed(path))


def _path_edges(path):
    return {frozenset({a, b}) for a, b in zip(path, path[1:])}


def _one_contiguous_double_bond_segment(edges):
    if len(edges) < 2:
        return False
    degree = defaultdict(int)
    for edge in edges:
        a, b = tuple(edge)
        degree[a] += 1
        degree[b] += 1
    return max(degree.values(), default=0) <= 2 and sum(v == 1 for v in degree.values()) == 2


def _is_tree(adjacency):
    if not adjacency:
        return False
    seen, stack = set(), [next(iter(adjacency))]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(adjacency[idx] - seen)
    edge_count = sum(map(len, adjacency.values())) // 2
    return seen == set(adjacency) and edge_count == len(adjacency) - 1


def _render_name(carbon_count, double_locs, suffix_locants, prefixes):
    root = ROOTS.get(carbon_count)
    multiplier = MULT.get(len(double_locs))
    if root is None or not multiplier or len(double_locs) < 2:
        return None
    suffix = multiplier + "ene"
    parent = f"{root}a-{','.join(map(str, double_locs))}-{suffix}"
    if suffix_locants:
        parent = attach_suffix(parent, "ol", suffix_locants)
    prefix_text = format_prefixes(list(prefixes))
    return join_prefix_parent(prefix_text, parent)
