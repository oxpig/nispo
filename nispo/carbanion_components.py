"""Mapped preferred names for simple saturated carbon-anion components.

The producer is limited to an ordinary all-carbon tree with one hydron-loss
center. Parent paths and orientations are compared before a name is rendered.
"""

from dataclasses import dataclass
from itertools import combinations


_RULE = "P-72.2.2.1/P-72.7/P-44/P-45/P-14.3.4.4"


@dataclass(frozen=True, slots=True)
class CarbanionComponent:
    """A complete mapped organic anion component."""

    name: str
    atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    charge: int
    rule: str
    proof: tuple
    component_roles: tuple[tuple[str, frozenset[int], tuple[tuple[int, str], ...]], ...]
    parent_atoms: frozenset[int]
    equivalent_parent_mappings: tuple[tuple[int, ...], ...] = ()


def carbanion_component(mol, Chem):
    """Return a typed component for a bounded saturated C-tree carbanion."""
    if (mol is None or mol.GetNumHeavyAtoms() > 64
            or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1):
        return None
    atoms = tuple(mol.GetAtoms())
    if (not atoms or any(atom.GetAtomicNum() != 6 for atom in atoms)
            or any(atom.GetIsotope() or int(atom.GetChiralTag())
                   or atom.GetNumRadicalElectrons() for atom in atoms)
            or any(int(bond.GetStereo()) or bond.GetBondDir() != Chem.BondDir.NONE
                   or bond.GetBondType() != Chem.BondType.SINGLE for bond in mol.GetBonds())):
        return None
    charged = [atom for atom in atoms if atom.GetFormalCharge()]
    if len(charged) != 1 or charged[0].GetFormalCharge() != -1:
        return None
    anion = charged[0]
    carbon_atoms = frozenset(atom.GetIdx() for atom in atoms)
    adjacency = {idx: set() for idx in carbon_atoms}
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[left].add(right)
        adjacency[right].add(left)
    if sum(map(len, adjacency.values())) != 2 * (len(carbon_atoms) - 1):
        return None
    if any(mol.GetAtomWithIdx(idx).IsInRing() for idx in carbon_atoms):
        return None
    seen, stack = set(), [next(iter(carbon_atoms))]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(adjacency[idx] - seen)
    if seen != carbon_atoms:
        return None

    # Verify hydron-loss valences from the input representation. The unique
    # charged carbon has one fewer hydrogen than a neutral saturated carbon;
    # all other atoms retain their ordinary tetravalent hydrogen count.
    for atom in atoms:
        degree = len(adjacency[atom.GetIdx()])
        expected_h = (3 if atom.GetIdx() == anion.GetIdx() else 4) - degree
        if expected_h < 0 or atom.GetTotalNumHs(includeNeighbors=True) != expected_h:
            return None
        if atom.GetTotalValence() != degree + expected_h:
            return None

    leaves = sorted(idx for idx in carbon_atoms if len(adjacency[idx]) <= 1)
    paths = {tuple((idx,)) for idx in carbon_atoms} if len(carbon_atoms) == 1 else set()
    for left, right in combinations(leaves, 2):
        path = _tree_path(adjacency, left, right)
        if path and anion.GetIdx() in path:
            paths.add(path)
    if not paths:
        return None
    max_length = max(map(len, paths))
    paths = {path for path in paths if len(path) == max_length}

    # Parent facts are computed on a neutral saturated copy only. The actual
    # component name and charge ownership remain tied to the original graph.
    facts_mol = Chem.Mol(mol)
    neutral_center = facts_mol.GetAtomWithIdx(anion.GetIdx())
    neutral_center.SetFormalCharge(0)
    neutral_center.SetNoImplicit(False)
    neutral_center.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(facts_mol)
    except (ValueError, RuntimeError):
        return None

    from .acyclic.rules import _path_substituent_prefixes
    from .anion_parent_selection import (
        AnionicCenterFact, AnionicParentCandidate, anion_parent_frontier,
    )
    from .constants import ROOTS
    from .formatting import format_prefixes, join_prefix_parent
    from .parent_selection import graph_parent_facts

    candidates = []
    unrenderable_maximal_candidate = False
    for path in paths:
        for oriented in (path, tuple(reversed(path))):
            locants = {idx: str(position)
                       for position, idx in enumerate(oriented, start=1)}
            prefixes = _path_substituent_prefixes(
                mol, oriented, skip_plain_hydrogen=True)
            if prefixes is None:
                unrenderable_maximal_candidate = True
                continue
            root = ROOTS.get(len(oriented))
            if root is None:
                unrenderable_maximal_candidate = True
                continue
            if len(oriented) == 1:
                suffix_name = "methanide"
            elif len(oriented) == 2 and not prefixes:
                # The two ethane positions are equivalent (P-14.3.4.4).
                suffix_name = "ethanide"
            else:
                suffix_name = f"{root}an-{locants[anion.GetIdx()]}-ide"
            prefix_text = format_prefixes(prefixes)
            name = join_prefix_parent(prefix_text, suffix_name) if prefix_text else suffix_name
            facts = graph_parent_facts(
                facts_mol, oriented, locants,
                principal_class=(18,), suffix_centers=(anion.GetIdx(),),
                suffix_locants=(locants[anion.GetIdx()],), prefixes=prefixes,
                kind="chain", name=name,
            )
            if facts is None:
                unrenderable_maximal_candidate = True
                continue
            center = AnionicCenterFact(
                atom_ids=frozenset((anion.GetIdx(),)),
                charge_atom_ids=frozenset((anion.GetIdx(),)), class_symbol="C",
                operation="ide", evidence_kind="skeletal", source_rule="P-72.2.2.1",
                proof="one saturated skeletal carbon with formal charge −1 and hydron-loss valence",
            )
            candidate = AnionicParentCandidate(
                identity="saturated-carbon-ide-parent",
                parent_atom_ids=frozenset(oriented),
                considered_charge_atoms=frozenset((anion.GetIdx(),)),
                parent_centers=(center,), suffix_key=(0,), parent_facts=facts,
                proof="complete longest eligible saturated carbon chain through the anionic center",
            )
            candidates.append({"candidate": candidate, "name": name,
                               "path": oriented, "locants": locants,
                               "prefixes": tuple(prefixes)})
    # Never let a successfully rendered but inferior chain win after an
    # eligible maximum chain failed at any naming/fact-construction step.
    if unrenderable_maximal_candidate or not candidates:
        return None

    frontier = anion_parent_frontier(mol, [row["candidate"] for row in candidates])
    frontier_ids = {id(candidate) for candidate in frontier}
    selected = [row for row in candidates if id(row["candidate"]) in frontier_ids]
    if not selected or len({row["name"] for row in selected}) != 1:
        return None
    # Symmetric tied mappings can differ by an automorphism; they share one
    # preferred name. Use a deterministic row only for the convenience map.
    selected_mappings = tuple(sorted({tuple(row["path"]) for row in selected}))
    chosen = min(selected, key=lambda row: tuple(row["path"]))
    path = chosen["path"]
    parent_atoms = frozenset(path)
    branch_atoms = carbon_atoms - parent_atoms
    heavy = frozenset(carbon_atoms)
    roles = [("principal-anionic-chain", parent_atoms,
              tuple((idx, chosen["locants"][idx]) for idx in path))]
    for locant in sorted({row[0] for row in chosen["prefixes"]}, key=int):
        # Prefix-bearing branch atoms are recovered by the parent attachment
        # locant. They remain fully owned without pretending to be host atoms.
        host_pos = int(locant)
        host = path[host_pos - 1]
        component = _branch_atoms(mol, adjacency, parent_atoms, host)
        if not component or component & (set().union(*(set(row[1]) for row in roles))):
            return None
        roles.append((f"branches-at-{locant}", frozenset(component), ()))
    if set().union(*(set(row[1]) for row in roles)) != set(heavy):
        return None
    atom_locants = tuple((idx, chosen["locants"][idx]) for idx in path)
    return CarbanionComponent(
        chosen["name"], heavy, atom_locants, -1, _RULE,
        ("all longest eligible chains and orientations enumerated",
         "P-72.7 frontier compared maximum anionic centers/class then P-44/P-45 facts",
         ("frontier_names", tuple(sorted({row["name"] for row in selected}))),
         ("candidate_count", len(candidates)),
         ("equivalent_parent_mappings", selected_mappings),
         ("selected_parent_atoms", tuple(path)),
         ("anionic_center", anion.GetIdx()),),
        tuple(roles), parent_atoms, selected_mappings,
    )


def _tree_path(adjacency, start, end):
    stack = [(start, (start,))]
    while stack:
        current, path = stack.pop()
        if current == end:
            return path
        for neighbor in sorted(adjacency[current] - set(path), reverse=True):
            stack.append((neighbor, path + (neighbor,)))
    return ()


def _branch_atoms(mol, adjacency, parent_atoms, host):
    branches = set()
    stack = [neighbor for neighbor in adjacency[host] if neighbor not in parent_atoms]
    while stack:
        idx = stack.pop()
        if idx in branches or idx in parent_atoms:
            continue
        branches.add(idx)
        stack.extend(neighbor for neighbor in adjacency[idx]
                     if neighbor not in parent_atoms and neighbor not in branches)
    return branches
