"""Source-ordered P-46 path frontiers for bounded carbon unsaturated groups.

The caller chooses P-29 method (1) or method (2). This module preserves
original atom indices and returns structural comparison facts; it does not
render names or make a whole-molecule PIN claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .formatting import prefix_alphabetization_key


RULES = ("P-29.2", "P-29.3.2", "P-46.0", "P-46.1(b)", "P-46.1(d)",
         "P-46.1(h)", "P-46.1(i)", "P-46.1(k)", "P-46.1(l)",
         "P-46.1(m)")
_PROVED = ("P-46.1(b)", "P-46.1(d)", "P-46.1(e)", "P-46.1(h)",
           "P-46.1(i)", "P-46.1(j)", "P-46.1(k)", "P-46.1(l)")
MAX_ATOMS = 32
MAX_PATHS = 2048
MAX_COMPARISONS = 100_000


@dataclass(frozen=True, slots=True)
class PrefixComponent:
    atom_ids: tuple[int, ...]
    host_atom_id: int
    locant: int
    proved_name: str | None


@dataclass(frozen=True, slots=True)
class UnsaturatedPathCandidate:
    method: int
    path_atom_ids: tuple[int, ...]
    atom_locants: tuple[tuple[int, int], ...]
    free_valence_locants: tuple[int, ...]
    multiple_bond_locants: tuple[int, ...]
    double_bond_locants: tuple[int, ...]
    prefixes: tuple[PrefixComponent, ...]
    proved_criteria: tuple[str, ...] = _PROVED

    @property
    def atom_to_locant(self) -> dict[int, int]:
        return dict(self.atom_locants)


@dataclass(frozen=True, slots=True)
class CriterionComparison:
    preference: int | None
    criterion: str
    reason: str


@dataclass(frozen=True, slots=True)
class UnsaturatedSelection:
    method: int
    scope_atom_ids: tuple[int, ...]
    frontier: tuple[UnsaturatedPathCandidate, ...]
    status: str
    comparisons: tuple[CriterionComparison, ...]
    rules: tuple[str, ...] = RULES


def _neighbors(mol, atom_id, scope):
    return tuple(n.GetIdx() for n in mol.GetAtomWithIdx(atom_id).GetNeighbors()
                 if n.GetIdx() in scope)


def _graph_scope(mol, root_id, attachment_id, atoms):
    if (not isinstance(atoms, (tuple, list, set, frozenset))
            or isinstance(root_id, bool) or not isinstance(root_id, int)
            or isinstance(attachment_id, bool) or not isinstance(attachment_id, int)):
        return None
    scope = frozenset(atoms)
    if (not scope or any(isinstance(i, bool) or not isinstance(i, int) for i in scope)
            or root_id not in scope or attachment_id in scope or root_id == attachment_id
            or root_id < 0 or root_id >= mol.GetNumAtoms()
            or any(not isinstance(i, int) or i < 0 or i >= mol.GetNumAtoms() for i in scope)
            or not isinstance(attachment_id, int) or not (0 <= attachment_id < mol.GetNumAtoms())):
        return None
    bond = mol.GetBondBetweenAtoms(root_id, attachment_id)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    # This selector covers carbon-tree frontiers, saturated or unsaturated.
    # Saturated trees remain in scope because P-46 ranks applicable parent
    # chains; the module does not itself choose a naming method.
    for atom_id in scope:
        atom = mol.GetAtomWithIdx(atom_id)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != 0
                or atom.GetNumRadicalElectrons() or atom.GetIsotope()
                or int(atom.GetChiralTag()) or atom.IsInRing()
                or atom.GetTotalValence() != 4):
            return None
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope():
                return None
            other = neighbor.GetIdx()
            if neighbor.GetAtomicNum() > 1 and other not in scope:
                if not (atom_id == root_id and other == attachment_id):
                    return None

    edges = []
    multiple_degree = {i: 0 for i in scope}
    for atom_id in scope:
        atom = mol.GetAtomWithIdx(atom_id)
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            if other not in scope or atom_id >= other:
                continue
            bond = mol.GetBondBetweenAtoms(atom_id, other)
            order = bond.GetBondTypeAsDouble()
            if order not in {1.0, 2.0, 3.0} or int(bond.GetStereo()) != 0:
                return None
            if order > 1:
                multiple_degree[atom_id] += 1
                multiple_degree[other] += 1
            edges.append((atom_id, other))
    if len(edges) != len(scope) - 1 or any(v > 1 for v in multiple_degree.values()):
        # A connected graph with n-1 edges is a tree; adjacent multiple bonds
        # are cumulenes and lie outside this implementation's proof scope.
        return None
    seen = set()
    stack = [root_id]
    while stack:
        atom_id = stack.pop()
        if atom_id in seen:
            continue
        seen.add(atom_id)
        stack.extend(n for n in _neighbors(mol, atom_id, scope) if n not in seen)
    if seen != set(scope):
        return None
    return scope


def _all_paths(mol, scope, root_id, method, max_paths):
    leaves = [i for i in scope if len(_neighbors(mol, i, scope)) <= 1]
    paths = set()
    if len(scope) == 1:
        return ((root_id,),)
    if method == 1:
        # Method (1) fixes the free valence at chain locant 1. The root may
        # have additional branches in the complete substituent; those become
        # side prefixes rather than disqualifying it as a chain endpoint.
        stack = [(root_id, (root_id,), frozenset({root_id}))]
        while stack:
            atom_id, path, visited = stack.pop()
            next_atoms = [n for n in _neighbors(mol, atom_id, scope) if n not in visited]
            if atom_id != root_id and atom_id in leaves:
                paths.add(path)
                if len(paths) > max_paths:
                    return None
                continue
            for other in next_atoms:
                stack.append((other, path + (other,), visited | {other}))
        return tuple(sorted(paths))
    # The admitted graph is a tree, so each endpoint pair has one path.
    for start in leaves:
        stack = [(start, (start,), frozenset({start}))]
        while stack:
            atom_id, path, visited = stack.pop()
            if atom_id in leaves and len(path) >= 2 and root_id in visited:
                reverse = tuple(reversed(path))
                paths.add(min(path, reverse))
                if len(paths) > max_paths:
                    return None
                continue
            for other in _neighbors(mol, atom_id, scope):
                if other not in visited:
                    stack.append((other, path + (other,), visited | {other}))
    return tuple(sorted(paths))


def _component_prefixes(mol, scope, path, names):
    path_set = frozenset(path)
    remaining = set(scope - path_set)
    locants = {atom_id: i for i, atom_id in enumerate(path, 1)}
    result = []
    while remaining:
        seed = min(remaining)
        stack, component, boundary = [seed], set(), set()
        while stack:
            atom_id = stack.pop()
            if atom_id not in remaining:
                continue
            remaining.remove(atom_id)
            component.add(atom_id)
            for neighbor in _neighbors(mol, atom_id, scope):
                if neighbor in path_set:
                    boundary.add(neighbor)
                elif neighbor in remaining:
                    stack.append(neighbor)
        if len(boundary) != 1:
            return None
        host = next(iter(boundary))
        key = tuple(sorted(component))
        result.append(PrefixComponent(key, host, locants[host], names.get(key)))
    return tuple(sorted(result, key=lambda p: (p.locant, p.atom_ids)))


def _candidate(mol, scope, path, root_id, method, names):
    locants = {atom_id: i for i, atom_id in enumerate(path, 1)}
    free_locants = (locants[root_id],)
    multi, doubles = [], []
    for i, (left, right) in enumerate(zip(path, path[1:]), 1):
        order = mol.GetBondBetweenAtoms(left, right).GetBondTypeAsDouble()
        if order > 1:
            multi.append(i)
        if order == 2:
            doubles.append(i)
    prefixes = _component_prefixes(mol, scope, path, names)
    if prefixes is None:
        return None
    return UnsaturatedPathCandidate(
        method, tuple(path), tuple((atom_id, locants[atom_id]) for atom_id in path),
        free_locants, tuple(multi), tuple(doubles), prefixes,
    )


def compare_unsaturated_candidates(left, right):
    """Return the first decisive applicable P-46 criterion or an explicit barrier."""
    if left.method not in {1, 2} or right.method != left.method:
        return CriterionComparison(None, "P-29.2", "Methods differ or are unsupported.")
    if left.proved_criteria != _PROVED or right.proved_criteria != _PROVED:
        return CriterionComparison(None, "P-46.1", "A required comparison fact is unproved.")
    criteria = (
        ("P-46.1(b)", -len(left.path_atom_ids), -len(right.path_atom_ids),
         "longer skeletal chain"),
        ("P-46.1(d).1", -len(left.multiple_bond_locants),
         -len(right.multiple_bond_locants), "greater multiple-bond count"),
        ("P-46.1(d).2", -len(left.double_bond_locants),
         -len(right.double_bond_locants), "greater double-bond count"),
        ("P-46.1(h)", left.free_valence_locants, right.free_valence_locants,
         "lower free-valence locants"),
        ("P-46.1(i).1", left.multiple_bond_locants, right.multiple_bond_locants,
         "lower multiple-bond locants"),
        ("P-46.1(i).2", left.double_bond_locants, right.double_bond_locants,
         "lower double-bond locants"),
        ("P-46.1(k)", -len(left.prefixes), -len(right.prefixes),
         "greater number of substituent components"),
        ("P-46.1(l)", tuple(sorted(p.locant for p in left.prefixes)),
         tuple(sorted(p.locant for p in right.prefixes)), "lower substituent locants"),
    )
    for rule, a, b, reason in criteria:
        if a != b:
            return CriterionComparison(-1 if a < b else 1, rule, reason)
    if not left.prefixes:
        return CriterionComparison(0, "tie", "No prefixes remain for criterion (m).")
    if any(p.proved_name is None for p in left.prefixes + right.prefixes):
        return CriterionComparison(None, "P-46.1(m)",
                                   "Caller-proved names are needed for prefix alphabetization.")
    def alpha_locs(candidate):
        ordered = sorted(candidate.prefixes,
                         key=lambda p: (prefix_alphabetization_key(p.proved_name), p.locant))
        return tuple(p.locant for p in ordered)
    a, b = alpha_locs(left), alpha_locs(right)
    if a != b:
        return CriterionComparison(-1 if a < b else 1, "P-46.1(m)",
                                   "Lower locants for prefixes cited earlier alphanumerically.")
    return CriterionComparison(0, "tie", "All applicable P-46 criteria tie.")


def _incomplete(method, scope, reason):
    return UnsaturatedSelection(method, tuple(sorted(scope)), (), "incomplete",
                                (CriterionComparison(None, "resource cap", reason),))


def unsaturated_substituent_frontier(mol, root_id: int, attachment_id: int,
                                     atoms, *, method: int,
                                     prefix_names: Mapping[tuple[int, ...], str] | None = None,
                                     max_atoms: int = MAX_ATOMS,
                                     max_paths: int = MAX_PATHS,
                                     max_comparisons: int = MAX_COMPARISONS):
    """Return complete P-46 frontier facts, or ``None`` outside the proof scope.

    ``root_id`` is the original free-valence carbon and ``attachment_id`` is
    the host-side carbon connected to it. ``atoms`` must be the complete
    heavy-atom substituent scope. ``method`` is explicitly 1 (free-valence
    carbon at path locant 1) or 2 (free-valence locant can be any position).
    Optional ``prefix_names`` maps each off-path component's sorted original
    atom-ID tuple to a caller-proved complete prefix name for criterion (m).
    The configurable limits are engineering budgets, not chemical criteria;
    their hard ceilings are ``MAX_ATOMS``, ``MAX_PATHS``, and
    ``MAX_COMPARISONS``. A reached budget returns an ``incomplete`` selection
    with an empty frontier, never a selection from partial candidates. A
    returned selection may have status ``unresolved``; callers must retain
    the reported frontier instead of selecting by atom order.
    """
    if isinstance(method, bool) or method not in {1, 2}:
        return None
    for value, hard_limit in ((max_atoms, MAX_ATOMS), (max_paths, MAX_PATHS),
                              (max_comparisons, MAX_COMPARISONS)):
        if (isinstance(value, bool) or not isinstance(value, int)
                or value < 1 or value > hard_limit):
            return None
    if not isinstance(atoms, (tuple, list, set, frozenset)):
        return None
    requested_scope = frozenset(atoms)
    if (not requested_scope or any(isinstance(i, bool) or not isinstance(i, int)
                                   for i in requested_scope)
            or isinstance(root_id, bool) or not isinstance(root_id, int)
            or isinstance(attachment_id, bool) or not isinstance(attachment_id, int)
            or attachment_id in requested_scope):
        return None
    if len(requested_scope) > max_atoms:
        return _incomplete(method, requested_scope,
                           f"atom limit exceeded ({len(requested_scope)} > {max_atoms})")
    scope = _graph_scope(mol, root_id, attachment_id, atoms)
    if scope is None:
        return None
    names = dict(prefix_names or {})
    if any(not isinstance(key, tuple) or not isinstance(value, str) or not value
           for key, value in names.items()):
        return None
    paths = _all_paths(mol, scope, root_id, method, max_paths)
    if paths is None:
        return _incomplete(method, scope,
                           f"path limit exceeded (more than {max_paths} paths)")
    candidates = []
    for undirected in paths:
        if method == 1:
            orientations = (undirected,)
        else:
            orientations = (undirected, tuple(reversed(undirected)))
        for path in orientations:
            if root_id not in path:
                continue
            candidates.append(_candidate(mol, scope, path, root_id, method, names))
            if len(candidates) > max_paths:
                return _incomplete(method, scope,
                                   f"oriented candidate limit exceeded ({max_paths})")
    if not candidates:
        return None
    required_comparisons = len(candidates) * (len(candidates) - 1) // 2
    if required_comparisons > max_comparisons:
        return _incomplete(method, scope,
                           f"comparison limit exceeded ({required_comparisons} > {max_comparisons})")
    beaten, comparisons = set(), []
    unresolved_pairs = []
    for i, left in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            right = candidates[j]
            result = compare_unsaturated_candidates(left, right)
            comparisons.append(result)
            if result.preference == -1:
                beaten.add(j)
            elif result.preference == 1:
                beaten.add(i)
            elif result.preference is None:
                unresolved_pairs.append((i, j))
    survivors = tuple(i for i in range(len(candidates)) if i not in beaten)
    frontier = tuple(candidates[i] for i in survivors)
    unresolved = any(i in survivors and j in survivors for i, j in unresolved_pairs)
    status = "unresolved" if unresolved else ("resolved" if len(frontier) == 1 else "tie")
    return UnsaturatedSelection(method, tuple(sorted(scope)), frontier, status,
                                tuple(comparisons))
