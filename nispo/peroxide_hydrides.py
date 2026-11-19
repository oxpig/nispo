"""Mapped substitutive parents for simple neutral P-63.3.1 peroxides.

This bounded route fills the function-free branched carbon-parent gap below
P-51.4's four-heterounit replacement threshold. It delegates the additive
peroxy/disulfanyl fragment name to the established mapped peroxide helper and
uses shared parent facts for P-44/P-45 choice.
"""
from dataclasses import dataclass
from collections import deque

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision

RULES = ("P-44", "P-45", "P-51.3", "P-51.4.1.1", "P-52.2.8", "P-63.3.1")


@dataclass(frozen=True, slots=True)
class _Candidate:
    name: str
    kind: str
    parent_atoms: frozenset[int]
    atom_locants: tuple
    branch_roles: tuple
    peroxide_role: frozenset[int]
    facts: object


def _single(mol, left, right):
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondTypeAsDouble() == 1


def _components(mol, carbon_atoms):
    unseen = set(carbon_atoms)
    result = []
    while unseen:
        start = min(unseen)
        seen, queue = {start}, deque([start])
        unseen.remove(start)
        while queue:
            current = queue.popleft()
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                idx = neighbor.GetIdx()
                if idx in unseen and idx in carbon_atoms:
                    unseen.remove(idx)
                    seen.add(idx)
                    queue.append(idx)
        result.append(frozenset(seen))
    return tuple(result)


def _longest_paths(mol, atoms, root):
    """Return every maximal terminal-to-terminal path in a carbon tree."""
    graph = {i: sorted(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                       if n.GetIdx() in atoms) for i in atoms}
    if len(atoms) == 1:
        return ((root,),) if root in atoms else ()
    if sum(map(len, graph.values())) // 2 != len(atoms) - 1:
        return ()
    leaves = sorted(i for i, row in graph.items() if len(row) <= 1)
    paths = set()
    for left_i, left in enumerate(leaves):
        for right in leaves[left_i + 1:]:
            stack = [(left, None, (left,))]
            while stack:
                current, previous, path = stack.pop()
                if current == right:
                    paths.add(path)
                    break
                for nxt in graph[current]:
                    if nxt != previous:
                        stack.append((nxt, current, path + (nxt,)))
    if not paths:
        return ()
    longest = max(map(len, paths))
    return tuple(sorted(path for path in paths if len(path) == longest))


def _parent_isotope_requires_all_prefix_locants(mol, path):
    """P-82.6.1.1: partial isotope modification forces parent prefix locants."""
    carbon = [mol.GetAtomWithIdx(idx) for idx in path
              if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6]
    labels = [atom.GetIsotope() for atom in carbon]
    return any(labels) and len(set(labels)) > 1


def _ring_orders(mol, atoms, root):
    if root not in atoms or len(atoms) < 3 or len(atoms) > 20:
        return ()
    graph = {i: sorted(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                       if n.GetIdx() in atoms) for i in atoms}
    if any(len(row) != 2 for row in graph.values()):
        return ()
    orders = []
    for first in graph[root]:
        path, previous, current = [root], root, first
        while current != root and len(path) <= len(atoms):
            path.append(current)
            nxt = [i for i in graph[current] if i != previous]
            if len(nxt) != 1:
                return ()
            previous, current = current, nxt[0]
        if current == root and len(path) == len(atoms):
            orders.append(tuple(path))
    return tuple(dict.fromkeys(orders))


def _render_component(mol, Chem, parent_name, path, kind, branch_prefixes,
                      bridge_locant, bridge_name, branch_roles, peroxide_role,
                      all_heavy, ring=False):
    from .component_mapping import MappedComponent, decorate_component
    from .parent_selection import graph_parent_facts

    atom_locants = tuple((idx, str(pos)) for pos, idx in enumerate(path, 1))
    prefixes = list(branch_prefixes)
    prefixes.append((str(bridge_locant), bridge_name))
    # The unique substituent locant is normally implicit on a one-carbon
    # parent, on ethane, and on a monocycle. P-82.6.1.1 requires all locants
    # when a partial isotope modification makes a parent locant necessary.
    render_prefixes = prefixes
    omit_locants = (len(prefixes) == 1 and
                    (len(path) == 1 or len(path) == 2 or ring)
                    and not _parent_isotope_requires_all_prefix_locants(mol, path))
    if omit_locants:
        render_prefixes = [("", text) for _, text in prefixes]
    mapped = MappedComponent(
        parent_name, atom_locants, frozenset(path), tuple(render_prefixes),
        frozenset(all_heavy - set(path)), systematic_parent=True,
        omit_methyl_prefix_locant=omit_locants,
    )
    name = decorate_component(mol, Chem, mapped)
    if not name:
        return None
    facts = graph_parent_facts(
        mol, path, atom_locants, principal_class=(), prefixes=tuple(prefixes),
        kind=kind, name=parent_name, allow_annotations=True,
        attachment_locants=(str(bridge_locant),),
    )
    if facts is None:
        return None
    return _Candidate(name, kind, frozenset(path), atom_locants,
                      tuple(branch_roles), frozenset(peroxide_role), facts)


def _chain_candidates(mol, Chem, atoms, root, bridge_atom, bridge_name,
                      peroxide_role, all_heavy):
    from .branch_rules import parent_chain_name
    from .component_mapping import mapped_branch
    from .formatting import complete_parent_name
    from .parent_selection import parent_frontier
    from .constants import ROOTS

    paths = _longest_paths(mol, atoms, root)
    # Do not restrict the parent to a shorter chain just to keep the bridge
    # host in that chain. Naming a bridge on a branch requires additional
    # mapped-substituent grammar, so this helper declines the off-parent case.
    if not paths or any(root not in path for path in paths):
        return ()
    candidates = []
    for raw_path in paths:
        for path in (raw_path, tuple(reversed(raw_path))):
            path_set = set(path)
            locants = {idx: str(pos) for pos, idx in enumerate(path, 1)}
            branch_prefixes, branch_roles = [], []
            remaining = set(atoms) - path_set
            for component in _components(mol, remaining):
                attachments = [(parent, idx) for idx in component
                               for parent in (n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors())
                               if parent in path_set]
                if len(attachments) != 1:
                    branch_prefixes = []
                    break
                parent_idx, branch_root = attachments[0]
                text, owned = mapped_branch(mol, Chem, branch_root, parent_idx)
                if not text or set(owned) != set(component):
                    branch_prefixes = []
                    break
                branch_prefixes.append((locants[parent_idx], text))
                branch_roles.append((frozenset(component), ()))
            if len(remaining) and not branch_prefixes:
                continue
            chain_stem = parent_chain_name(mol, path, as_substituent=False)
            chain_name = complete_parent_name(chain_stem)
            if not chain_name or len(path) not in ROOTS:
                continue
            bridge_locant = int(locants[root])
            candidate = _render_component(
                mol, Chem, chain_name, path, "chain", branch_prefixes,
                bridge_locant, bridge_name, branch_roles, peroxide_role,
                all_heavy,
            )
            if candidate:
                candidates.append(candidate)
    frontier = parent_frontier(candidates, lambda item: item.facts) if candidates else []
    if not frontier:
        return ()
    names = {item.name for item in frontier}
    if len(names) != 1:
        return ()
    return (sorted(frontier, key=lambda item: item.atom_locants)[0],)


def _ring_candidates(mol, Chem, atoms, root, bridge_name, peroxide_role,
                     all_heavy):
    from .constants import ROOTS
    orders = _ring_orders(mol, atoms, root)
    if not orders:
        return ()
    aromatic = len(atoms) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)
    if aromatic:
        parent_name = "benzene"
    else:
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
            return ()
        if any(not _single(mol, a, b) for order in orders
               for a, b in zip(order, order[1:] + order[:1])):
            return ()
        root_name = ROOTS.get(len(atoms))
        if not root_name:
            return ()
        parent_name = "cyclo" + root_name + "ane"
    candidates = []
    for order in orders:
        candidate = _render_component(
            mol, Chem, parent_name, order, "ring", (), 1, bridge_name,
            (), peroxide_role, all_heavy, ring=True,
        )
        if candidate:
            candidates.append(candidate)
    from .parent_selection import parent_frontier
    frontier = parent_frontier(candidates, lambda item: item.facts) if candidates else []
    if not frontier:
        return ()
    if len({item.name for item in frontier}) != 1:
        return ()
    return (sorted(frontier, key=lambda item: item.atom_locants)[0],)


def _candidate_for_side(mol, Chem, carbon_atoms, root, bridge_atom,
                        bridge_name, peroxide_role, all_heavy):
    carbon_atoms = frozenset(carbon_atoms)
    edge_count = sum(1 for bond in iter_bonds(mol)
                     if bond.GetBeginAtomIdx() in carbon_atoms
                     and bond.GetEndAtomIdx() in carbon_atoms)
    ring_count = edge_count - len(carbon_atoms) + 1
    if ring_count:
        if ring_count != 1 or any(not mol.GetAtomWithIdx(i).IsInRing() for i in carbon_atoms):
            return ()
        return _ring_candidates(mol, Chem, carbon_atoms, root, bridge_name,
                                peroxide_role, all_heavy)
    if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in carbon_atoms):
        return ()
    if any(not _single(mol, a, b)
           for a in carbon_atoms
           for b in (n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors())
           if b in carbon_atoms):
        return ()
    return _chain_candidates(mol, Chem, carbon_atoms, root, bridge_atom,
                             bridge_name, peroxide_role, all_heavy)


def peroxide_hydride_decision(mol, Chem):
    """Name a bounded neutral branched-alkyl/monoaryl peroxide or disulfide.

    Only one ordinary C–O–O–C or C–S–S–C bridge and two carbon components
    are admitted. Components are acyclic saturated carbon trees or one plain
    saturated/aromatic carbon monocycle. The bridge is named as one additive
    P-63.3.1 prefix, while the parent side is selected with shared P-44/P-45
    facts. Unsupported functions, annotations on O/S/H, mixed bridges,
    multiple bridges/rings, and other heterounits decline.
    """
    if (mol.GetStereoGroups() or mol.GetNumAtoms() > 128
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in iter_atoms(mol))):
        return None
    from .peroxide_components import peroxide_bridge, mapped_peroxide_prefix
    from .branch_rules import collect_subtree
    from .parent_selection import compare_parents

    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if any(a.GetAtomicNum() not in {6, 8, 16} for a in iter_atoms(mol)):
        return None
    if any(a.GetAtomicNum() in {8, 16} and
           (a.GetIsotope() or int(a.GetChiralTag()) or a.GetDegree() != 2 or a.IsInRing())
           for a in iter_atoms(mol)):
        return None
    if any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in iter_atoms(mol)):
        return None
    hetero = [a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() in {8, 16}]
    if len(hetero) != 2:
        return None
    pairs = []
    for idx in hetero:
        atom = mol.GetAtomWithIdx(idx)
        carbons = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 6]
        linkers = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() in {8, 16}]
        if len(carbons) == len(linkers) == 1:
            pairs.append((idx, carbons[0], linkers[0]))
    if len(pairs) != 2 or pairs[0][2] != pairs[1][0] and pairs[1][2] != pairs[0][0]:
        return None
    if pairs[0][2] != pairs[1][0] or pairs[1][2] != pairs[0][0]:
        return None
    x0, c0, x1 = pairs[0]
    x2, c1, x3 = pairs[1]
    if x1 != x2 or mol.GetAtomWithIdx(x0).GetAtomicNum() != mol.GetAtomWithIdx(x1).GetAtomicNum():
        return None
    if not _single(mol, c0, x0) or not _single(mol, x0, x1) or not _single(mol, x1, c1):
        return None
    if c0 == c1:
        return None
    # This route only admits two ordinary carbon components plus the typed
    # bridge. Other O/S atoms, annular heteroatoms, and mixed blocks are not
    # silently interpreted as substituents.
    carbon_atoms = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6)
    components = _components(mol, carbon_atoms)
    if len(components) != 2:
        return None
    side0 = next((comp for comp in components if c0 in comp), None)
    side1 = next((comp for comp in components if c1 in comp), None)
    if side0 is None or side1 is None or side0 == side1:
        return None
    # The implementation covers one skeletal monocycle total. Two cyclic
    # sides require a separate P-45/P-51 multiplicative comparison.
    total_cycle_rank = sum(
        sum(1 for bond in iter_bonds(mol)
            if bond.GetBeginAtomIdx() in component
            and bond.GetEndAtomIdx() in component) - len(component) + 1
        for component in components
    )
    if total_cycle_rank > 1:
        return None
    if any(a.GetIsotope() for a in iter_atoms(mol) if a.GetAtomicNum() != 6):
        return None
    prefix0_atoms = collect_subtree(mol, x0, c0)
    prefix1_atoms = collect_subtree(mol, x1, c1)
    prefix0, owned0 = mapped_peroxide_prefix(mol, Chem, x0, c0, prefix0_atoms)
    prefix1, owned1 = mapped_peroxide_prefix(mol, Chem, x1, c1, prefix1_atoms)
    if not prefix0 or not prefix1 or set(owned0) != set(prefix0_atoms) or set(owned1) != set(prefix1_atoms):
        return None
    candidates0 = _candidate_for_side(mol, Chem, side0, c0, x0, prefix0,
                                      owned0, heavy)
    candidates1 = _candidate_for_side(mol, Chem, side1, c1, x1, prefix1,
                                      owned1, heavy)
    if len(candidates0) != 1 or len(candidates1) != 1:
        return None
    left, right = candidates0[0], candidates1[0]
    comparison = compare_parents(left.facts, right.facts)
    if comparison.preference == -1:
        chosen = left
    elif comparison.preference == 1:
        chosen = right
    elif comparison.preference in {0, None} and left.name == right.name:
        chosen = left
    else:
        return None
    roles = [("parent_hydride", chosen.parent_atoms, chosen.atom_locants)]
    roles.extend(("carbon_substituent", atoms, locants)
                 for atoms, locants in chosen.branch_roles)
    roles.append(("peroxide_substituent", chosen.peroxide_role, ()))
    role_atoms = [set(atoms) for _, atoms, _ in roles]
    if (any(role_atoms[i] & role_atoms[j]
            for i in range(len(role_atoms)) for j in range(i + 1, len(role_atoms)))
            or set().union(*role_atoms) != set(heavy)
            or any({idx for idx, _ in locants} - set(atoms)
                   for _, atoms, locants in roles)):
        return None
    return MethodDecision(
        chosen.name, "substitutive", frozenset(), heavy,
        "P-63.3.1/P-44/P-45/P-51.4.1.1/P-52.2.8",
        chosen.atom_locants, component_roles=tuple(roles),
    )
