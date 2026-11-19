"""Bounded rooted P-46.1 method-(1) path selection for saturated C trees.

This module deliberately stops before name rendering. It returns all graph
paths that survive a source-ordered comparison, preserving input atom IDs.
Only neutral, isotope-free, acyclic, saturated carbon substituent trees are
admitted. Side components must be singly attached carbon trees; simple linear
alkyl prefixes have known alphabetical terms, while complex branch terms stay
unknown at P-46.1(m). Other graphs return an explicit decline (``None``), so
an unsupported path cannot disappear from comparison.

For the admitted scope, P-46.1(a) is inapplicable (method (1)); (b) maximizes
skeletal atoms. Criteria (c)–(j) tie because the entire graph is an ordinary
saturated carbon tree with method-(1) free valence at position 1. Criteria
(k)–(m) then compare substituent count, all substituent locants, and locants
of substituents in alphabetical order. See the official P-46 source at
https://iupac.qmul.ac.uk/BlueBook/P4.html#46.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .molecule_iteration import iter_atoms


_ALKYL_PREFIXES = {
    1: "methyl", 2: "ethyl", 3: "propyl", 4: "butyl", 5: "pentyl",
    6: "hexyl", 7: "heptyl", 8: "octyl", 9: "nonyl", 10: "decyl",
    11: "undecyl", 12: "dodecyl",
}


@dataclass(frozen=True, slots=True)
class PrefixUnit:
    """One alkyl substituent on the candidate path.

    ``term`` is available for simple linear prefixes; complex branch names are
    left unknown and become a barrier only if P-46.1(m) is reached.
    """

    locant: int
    atom_ids: tuple[int, ...]
    term: str | None


@dataclass(frozen=True, slots=True)
class RootedChainCandidate:
    """A root-ended chain and its fully proved method-(1) P-46 facts."""

    path_atom_ids: tuple[int, ...]
    locants: tuple[tuple[int, int], ...]
    prefixes: tuple[PrefixUnit, ...]
    criteria_key: tuple
    proved_criteria: tuple[str, ...]

    @property
    def atom_to_locant(self) -> dict[int, int]:
        return dict(self.locants)


@dataclass(frozen=True, slots=True)
class Comparison:
    preference: int | None
    rule: str
    reason: str


_PROVED = (
    "P-46.1(b)", "P-46.1(c)", "P-46.1(d)", "P-46.1(e)",
    "P-46.1(f)", "P-46.1(g)", "P-46.1(h)", "P-46.1(i)",
    "P-46.1(j)", "P-46.1(k)", "P-46.1(l)", "P-46.1(m)",
)


def _heavy_neighbors(mol, atom_id, scope):
    atom = mol.GetAtomWithIdx(atom_id)
    return tuple(n.GetIdx() for n in atom.GetNeighbors()
                 if n.GetAtomicNum() > 1 and n.GetIdx() in scope)


def _collect_rooted_scope(mol, root_id, attachment_id):
    if root_id == attachment_id or not (0 <= root_id < mol.GetNumAtoms()) or not (
        0 <= attachment_id < mol.GetNumAtoms()
    ):
        return None
    bond = mol.GetBondBetweenAtoms(root_id, attachment_id)
    if bond is None or bond.GetBondTypeAsDouble() != 1:
        return None

    seen = set()
    stack = [root_id]
    while stack:
        atom_id = stack.pop()
        if atom_id == attachment_id or atom_id in seen:
            continue
        seen.add(atom_id)
        stack.extend(_heavy_neighbors(mol, atom_id, set(range(mol.GetNumAtoms())))
                     if atom_id != root_id else
                     (n for n in _heavy_neighbors(mol, atom_id, set(range(mol.GetNumAtoms())))
                      if n != attachment_id))

    if not seen:
        return None
    # A complete carbon-only, single-bond tree is necessary for all the
    # claimed P-46.1 ties and prefix-unit decompositions below.
    for atom_id in seen:
        atom = mol.GetAtomWithIdx(atom_id)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != 0
                or atom.GetNumRadicalElectrons() or atom.GetIsotope()
                or atom.GetTotalValence() != 4
                or int(atom.GetChiralTag()) or atom.IsInRing()):
            return None
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope():
                return None
    edges = []
    for atom_id in seen:
        for neighbor in mol.GetAtomWithIdx(atom_id).GetNeighbors():
            other = neighbor.GetIdx()
            if other in seen and atom_id < other:
                bond = mol.GetBondBetweenAtoms(atom_id, other)
                if bond.GetBondTypeAsDouble() != 1:
                    return None
                edges.append((atom_id, other))
            elif other not in seen and not (atom_id == root_id and other == attachment_id):
                if neighbor.GetAtomicNum() > 1:
                    return None
    if len(edges) != len(seen) - 1:
        return None
    # Exactly one heavy-atom cut at the root is the rooted-substituent scope.
    external = [(a.GetIdx(), n.GetIdx()) for a in iter_atoms(mol) if a.GetIdx() in seen
                for n in a.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in seen]
    if external != [(root_id, attachment_id)]:
        if len(external) != 1 or external[0] != (root_id, attachment_id):
            return None
    return frozenset(seen)


def _rooted_leaf_paths(mol, root_id, scope):
    """Enumerate root-to-leaf paths of the acyclic heavy-atom tree."""
    paths = []
    stack = [(root_id, (root_id,), frozenset({root_id}))]
    while stack:
        atom_id, path, seen = stack.pop()
        next_atoms = [n for n in _heavy_neighbors(mol, atom_id, scope) if n not in seen]
        if not next_atoms:
            paths.append(path)
            continue
        for other in next_atoms:
            stack.append((other, path + (other,), seen | {other}))
    return tuple(paths)


def _prefix_unit(mol, attachment_on_path, component):
    """Prove an off-path carbon tree is a singly attached alkyl component."""
    component = frozenset(component)
    if not component:
        return None
    attachment_atoms = [i for i in component if any(
        n.GetIdx() == attachment_on_path
        for n in mol.GetAtomWithIdx(i).GetNeighbors()
    )]
    if len(attachment_atoms) != 1:
        return None
    attachment = attachment_atoms[0]
    degree = sum(n.GetIdx() in component for n in mol.GetAtomWithIdx(attachment).GetNeighbors())
    linear = degree <= 1
    for atom_id in component:
        degree = sum(n.GetIdx() in component for n in mol.GetAtomWithIdx(atom_id).GetNeighbors())
        if degree > 2 or (atom_id != attachment and degree < 1):
            linear = False
    # P-46.1(k)/(l) facts do not require a spelling for a complex branch.
    # P-46.1(m) can use a term only for the simple linear prefix vocabulary
    # independently supported by this scope.
    term = _ALKYL_PREFIXES.get(len(component)) if linear else None
    return attachment, term


def _prefix_unit_order(unit):
    """Return the P-14.5 ordering key for a proved complete prefix name."""
    if unit.term is None:
        return None
    from .formatting import prefix_alphabetization_key
    return prefix_alphabetization_key(unit.term)


def _prefix_citation_locants(prefixes):
    if any(_prefix_unit_order(unit) is None for unit in prefixes):
        return None
    return tuple(unit.locant for unit in sorted(
        prefixes, key=lambda unit: (_prefix_unit_order(unit), unit.locant)))


def _candidate_for_path(mol, path, scope):
    path_set = frozenset(path)
    remaining = set(scope - path_set)
    locant_by_atom = {atom_id: pos for pos, atom_id in enumerate(path, 1)}
    prefixes = []
    while remaining:
        seed = min(remaining)
        component, stack = set(), [seed]
        boundary = set()
        while stack:
            atom_id = stack.pop()
            if atom_id not in remaining:
                continue
            remaining.remove(atom_id)
            component.add(atom_id)
            for neighbor in mol.GetAtomWithIdx(atom_id).GetNeighbors():
                other = neighbor.GetIdx()
                if other in path_set:
                    boundary.add(other)
                elif other in remaining:
                    stack.append(other)
        if len(boundary) != 1:
            return None
        path_attachment = next(iter(boundary))
        prefix = _prefix_unit(mol, path_attachment, component)
        if prefix is None:
            return None
        _, term = prefix
        prefixes.append(PrefixUnit(locant_by_atom[path_attachment],
                                   tuple(sorted(component)), term))

    all_prefix_locs = tuple(sorted(prefix.locant for prefix in prefixes))
    citation_locs = _prefix_citation_locants(prefixes)
    # Criteria (b)–(j) are either maximized/ordered here or are proved equal
    # by the strict input scope. Active keys begin at (k).
    key = (-len(path), -len(prefixes), all_prefix_locs, citation_locs)
    return RootedChainCandidate(tuple(path), tuple(locant_by_atom.items()),
                                tuple(prefixes), key, _PROVED)


def compare_rooted_chains(left: RootedChainCandidate,
                          right: RootedChainCandidate) -> Comparison:
    """Compare complete candidates in the P-46.1 order for this scope."""
    if left.proved_criteria != _PROVED or right.proved_criteria != _PROVED:
        return Comparison(None, "P-46.1", "A required criterion is not proved.")
    fields = (
        ("P-46.1(b)", -len(left.path_atom_ids), -len(right.path_atom_ids)),
        ("P-46.1(k)", -len(left.prefixes), -len(right.prefixes)),
        ("P-46.1(l)", tuple(sorted(p.locant for p in left.prefixes)),
         tuple(sorted(p.locant for p in right.prefixes))),
    )
    for rule, a, b in fields:
        if a != b:
            return Comparison(-1 if a < b else 1, rule,
                              "First decisive P-46.1 criterion for a rooted method-(1) carbon tree.")
    if any(p.term is None for p in left.prefixes + right.prefixes):
        left_locs = tuple(sorted(p.locant for p in left.prefixes))
        right_locs = tuple(sorted(p.locant for p in right.prefixes))
        if left_locs == right_locs and len(set(left_locs)) <= 1:
            return Comparison(0, "tie", "Alphabetical order cannot change a single shared locant.")
        return Comparison(None, "P-46.1(m)",
                          "A complex prefix name is needed to prove alphabetical order.")
    left_alpha = _prefix_citation_locants(left.prefixes)
    right_alpha = _prefix_citation_locants(right.prefixes)
    if left_alpha != right_alpha:
        return Comparison(-1 if left_alpha < right_alpha else 1, "P-46.1(m)",
                          "First decisive P-46.1 criterion for a rooted method-(1) carbon tree.")
    return Comparison(0, "tie", "All applicable P-46.1 criteria tie in the admitted scope.")


def rooted_chain_frontier(candidates):
    """Return all nondominated paths; do not use atom IDs to break a tie."""
    candidates = tuple(candidates)
    if not candidates:
        return ()
    beaten = set()
    unresolved = set()
    for i, left in enumerate(candidates):
        for j, right in enumerate(candidates):
            if i >= j:
                continue
            comparison = compare_rooted_chains(left, right)
            if comparison.preference == -1:
                beaten.add(j)
            elif comparison.preference == 1:
                beaten.add(i)
            if comparison.preference is None:
                unresolved.add((i, j))
    survivors = tuple(i for i in range(len(candidates)) if i not in beaten)
    if any(i in survivors and j in survivors for i, j in unresolved):
        return ()
    return tuple(candidates[i] for i in survivors)


def select_rooted_substituent_chain(mol, root_id, attachment_id):
    """Return a P-46.1 method-(1) frontier, or ``None`` when scope is unknown.

    The returned candidate(s) preserve original atom IDs. The caller owns
    integration into branch rendering and must retain all ties until later
    source rules or graph equivalence resolve them.
    """
    scope = _collect_rooted_scope(mol, root_id, attachment_id)
    if scope is None or root_id not in scope:
        return None
    return _rooted_substituent_chain_in_scope(mol, root_id, scope)


def _rooted_substituent_chain_in_scope(mol, root_id, scope):
    """Select only after this call has proved the original strict scope."""
    paths = _rooted_leaf_paths(mol, root_id, scope)
    if not paths:
        return None
    max_size = max(map(len, paths))
    maximal = [path for path in paths if len(path) == max_size]
    candidates = [_candidate_for_path(mol, path, scope) for path in maximal]
    if any(candidate is None for candidate in candidates):
        return None
    frontier = rooted_chain_frontier(candidates)
    return frontier or None


@dataclass(frozen=True, slots=True)
class AlkanylPathCandidate:
    """One P-46 method-(2) path candidate for an interior carbon root.

    ``method2_eligible`` is deliberately a local graph fact, not a choice
    between method (1) and method (2); P-29 permissions and PIN preference
    remain caller responsibilities. ``root_locant`` is the locant of the
    single free valence on the substituent's attachment carbon.
    """

    method: int
    path_atom_ids: tuple[int, ...]
    locants: tuple[tuple[int, int], ...]
    root_locant: int
    prefixes: tuple[PrefixUnit, ...]
    criteria_key: tuple
    proved_criteria: tuple[str, ...]

    @property
    def atom_to_locant(self) -> dict[int, int]:
        return dict(self.locants)


_METHOD2_PROVED = (
    "P-46.1(a)", "P-46.1(b)", "P-46.1(c)", "P-46.1(d)",
    "P-46.1(e)", "P-46.1(f)", "P-46.1(g)", "P-46.1(h)",
    "P-46.1(i)", "P-46.1(j)", "P-46.1(k)", "P-46.1(l)",
    "P-46.1(m)",
)


def _all_simple_paths_containing_root(mol, scope, root_id):
    """Enumerate oriented simple paths through a root in a tree."""
    leaves = [i for i in scope if len(_heavy_neighbors(mol, i, scope)) <= 1]
    paths = set()
    for start in leaves:
        stack = [(start, (start,), frozenset({start}))]
        while stack:
            atom_id, path, seen = stack.pop()
            if atom_id in leaves and root_id in seen and len(path) >= 3:
                paths.add(path)
            for other in _heavy_neighbors(mol, atom_id, scope):
                if other not in seen:
                    stack.append((other, path + (other,), seen | {other}))
    return tuple(paths)


def _method2_candidate(mol, path, root_id, scope):
    if root_id not in path or path[0] == root_id or path[-1] == root_id:
        return None
    base = _candidate_for_path(mol, path, scope)
    if base is None:
        return None
    root_locant = path.index(root_id) + 1
    key = (-len(path), root_locant, -len(base.prefixes),
           tuple(sorted(p.locant for p in base.prefixes)),
           _prefix_citation_locants(base.prefixes))
    return AlkanylPathCandidate(2, tuple(path), base.locants, root_locant,
                                base.prefixes, key, _METHOD2_PROVED)


def compare_alkanyl_paths(left: AlkanylPathCandidate,
                          right: AlkanylPathCandidate) -> Comparison:
    """Compare method-(2) candidates under the proved saturated-C scope."""
    if (left.method != 2 or right.method != 2
            or left.proved_criteria != _METHOD2_PROVED
            or right.proved_criteria != _METHOD2_PROVED):
        return Comparison(None, "P-46.1", "Method-(2) comparison facts are incomplete.")
    fields = (
        ("P-46.1(b)", -len(left.path_atom_ids), -len(right.path_atom_ids)),
        ("P-46.1(h)", left.root_locant, right.root_locant),
        ("P-46.1(k)", -len(left.prefixes), -len(right.prefixes)),
        ("P-46.1(l)", tuple(sorted(p.locant for p in left.prefixes)),
         tuple(sorted(p.locant for p in right.prefixes))),
    )
    for rule, a, b in fields:
        if a != b:
            return Comparison(-1 if a < b else 1, rule,
                              "First decisive criterion for method-(2) saturated carbon paths.")
    if any(p.term is None for p in left.prefixes + right.prefixes):
        return Comparison(None, "P-46.1(m)",
                          "Complex prefix alphabetization is not proved.")
    alpha_left = _prefix_citation_locants(left.prefixes)
    alpha_right = _prefix_citation_locants(right.prefixes)
    if alpha_left != alpha_right:
        return Comparison(-1 if alpha_left < alpha_right else 1, "P-46.1(m)",
                          "First decisive criterion for method-(2) saturated carbon paths.")
    return Comparison(0, "tie", "All applicable criteria tie in the admitted scope.")


def select_alkanyl_substituent_path(mol, root_id, attachment_id):
    """Return the method-(2) path frontier for a saturated carbon tree.

    P-29.2(2) confines this route to a free valence at a locant other than 1;
    this function therefore requires an interior root on the path. It does not
    assert that method (2) is preferred or permitted for the enclosing use.
    It returns all unresolved ties, or ``None`` when the graph or a decisive
    criterion falls outside the proved scope.
    """
    scope = _collect_rooted_scope(mol, root_id, attachment_id)
    if scope is None or root_id not in scope:
        return None
    return _alkanyl_substituent_path_in_scope(mol, root_id, scope)


def _alkanyl_substituent_path_in_scope(mol, root_id, scope):
    """Continue method (2) from its freshly admitted immutable scope."""
    paths = _all_simple_paths_containing_root(mol, scope, root_id)
    candidates = [_method2_candidate(mol, path, root_id, scope) for path in paths]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        return None
    longest = max(len(candidate.path_atom_ids) for candidate in candidates)
    candidates = [c for c in candidates if len(c.path_atom_ids) == longest]
    beaten, unresolved = set(), set()
    for i, left in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            result = compare_alkanyl_paths(left, candidates[j])
            if result.preference == -1:
                beaten.add(j)
            elif result.preference == 1:
                beaten.add(i)
            elif result.preference is None:
                unresolved.add((i, j))
    survivors = tuple(i for i in range(len(candidates)) if i not in beaten)
    if any(i in survivors and j in survivors for i, j in unresolved):
        return None
    # Equivalent atom-order permutations describe identical paths; preserve
    # distinct ties, but remove duplicate oriented tuples only.
    return tuple(candidates[i] for i in survivors) or None


_SOURCE_SCOPE_COLLECTOR = _collect_rooted_scope
_SOURCE_HEAVY_NEIGHBORS = _heavy_neighbors
_SOURCE_ROOTED_CHAIN_SELECTOR = select_rooted_substituent_chain
_SOURCE_ALKANYL_PATH_SELECTOR = select_alkanyl_substituent_path


def mapped_saturated_carbon_prefix(mol, Chem, root_id, attachment_id, atoms):
    """Render a proved P-29/P-46 carbon-tree prefix, or decline its scope.

    ``None`` leaves an unsupported graph to other mapped producers. An empty
    name means the tree is in scope but an applicable comparison or spelling
    is unresolved. All tied paths must produce the same prefix; neither atom
    order nor the shortest spelling selects one of them.
    """
    scope = _collect_rooted_scope(mol, root_id, attachment_id)
    if scope is None or set(scope) != set(atoms) or len(scope) > 64:
        return None
    # API-owned native molecules and immutable integer scopes cannot run user
    # code between this admission and selection. Reuse that exact proof only
    # with the original observers; custom consumers/selectors keep their fresh
    # public validation path, including its native mutation observations.
    reuse_scope = (
        type(mol) is getattr(Chem, "Mol", None)
        and type(atoms) is frozenset and type(scope) is frozenset
        and all(type(i) is int for i in atoms)
        and "GetAtomWithIdx" not in vars(mol)
        and _collect_rooted_scope is _SOURCE_SCOPE_COLLECTOR
        and _heavy_neighbors is _SOURCE_HEAVY_NEIGHBORS
    )
    # P-29.6.1 retains this unsubstituted prefix without a numbering scheme.
    # The scope above has already excluded isotopes and stereo annotations.
    root_neighbors = _heavy_neighbors(mol, root_id, scope)
    if (len(scope) == 4 and len(root_neighbors) == 3
            and all(len(_heavy_neighbors(mol, i, scope)) == 1 for i in root_neighbors)):
        return "tert-butyl", frozenset(scope)
    if len(root_neighbors) <= 1:
        method = 1
        frontier = (_rooted_substituent_chain_in_scope(mol, root_id, scope)
                    if reuse_scope and select_rooted_substituent_chain is _SOURCE_ROOTED_CHAIN_SELECTOR
                    else select_rooted_substituent_chain(mol, root_id, attachment_id))
    else:
        method = 2
        frontier = (_alkanyl_substituent_path_in_scope(mol, root_id, scope)
                    if reuse_scope and select_alkanyl_substituent_path is _SOURCE_ALKANYL_PATH_SELECTOR
                    else select_alkanyl_substituent_path(mol, root_id, attachment_id))
    if not frontier:
        return "", frozenset()
    from .component_mapping import MappedComponent, decorate_component, mapped_branch
    from .constants import ROOTS

    names = set()
    for candidate in frontier:
        path = candidate.path_atom_ids
        if len(path) not in ROOTS:
            return "", frozenset()
        mapping = {i: str(j) for j, i in enumerate(path, 1)}
        prefixes, embedded = [], set()
        for unit in candidate.prefixes:
            member_atoms = set(unit.atom_ids)
            bonds = [(i, n.GetIdx()) for i in path
                     for n in mol.GetAtomWithIdx(i).GetNeighbors()
                     if n.GetIdx() in member_atoms]
            if len(bonds) != 1:
                return "", frozenset()
            host, root = bonds[0]
            text, owned = mapped_branch(mol, Chem, root, host)
            if not text or set(owned) != member_atoms or embedded & set(owned):
                return "", frozenset()
            embedded.update(owned)
            prefixes.append((mapping[host], text))
        if set(mapping) | embedded != set(scope):
            return "", frozenset()
        parent = (ROOTS[len(path)] + "yl" if method == 1 else
                  ROOTS[len(path)] + "an-" + mapping[root_id] + "-yl")
        text = decorate_component(mol, Chem, MappedComponent(
            parent, tuple(mapping.items()), frozenset(mapping),
            tuple(prefixes), frozenset(embedded), systematic_parent=True))
        if not text:
            return "", frozenset()
        names.add(text)
    return (names.pop(), frozenset(scope)) if len(names) == 1 else ("", frozenset())


_HALO_PREFIXES = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}
_TERMINAL_PREFIXES = {**_HALO_PREFIXES, 8: "hydroxy"}


def _collect_terminal_halo_carbon_scope(mol, root_id, attachment_id):
    """Admit a saturated carbon tree with halo and optional hydroxy groups."""
    if (root_id == attachment_id or not 0 <= root_id < mol.GetNumAtoms()
            or not 0 <= attachment_id < mol.GetNumAtoms()):
        return None
    cut = mol.GetBondBetweenAtoms(root_id, attachment_id)
    if cut is None or cut.GetBondTypeAsDouble() != 1:
        return None
    heavy = set()
    stack = [root_id]
    while stack:
        atom_id = stack.pop()
        if atom_id == attachment_id or atom_id in heavy:
            continue
        heavy.add(atom_id)
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(atom_id).GetNeighbors()
                     if n.GetAtomicNum() > 1 and n.GetIdx() != attachment_id)
    if not heavy:
        return None
    carbons = {i for i in heavy if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
    terminals = heavy - carbons
    if not carbons or not any(mol.GetAtomWithIdx(i).GetAtomicNum() in _HALO_PREFIXES
                              for i in terminals):
        return None
    for atom_id in heavy:
        atom = mol.GetAtomWithIdx(atom_id)
        if (atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                or atom.GetIsotope() or int(atom.GetChiralTag()) or atom.IsInRing()):
            return None
        if atom.GetAtomicNum() == 6:
            if atom.GetTotalValence() != 4:
                return None
            if any(n.GetAtomicNum() == 1 and n.GetIsotope()
                   for n in atom.GetNeighbors()):
                return None
        elif atom.GetAtomicNum() in _TERMINAL_PREFIXES:
            hydroxy = atom.GetAtomicNum() == 8
            if (atom.GetTotalValence() != 1 + hydroxy
                    or atom.GetTotalNumHs(includeNeighbors=True) != hydroxy
                    or any(n.GetIsotope() for n in atom.GetNeighbors())
                    or len(_heavy_neighbors(mol, atom_id, heavy)) != 1
                    or mol.GetAtomWithIdx(next(iter(_heavy_neighbors(mol, atom_id, heavy)))).GetAtomicNum() != 6):
                return None
        else:
            return None
        for bond in atom.GetBonds():
            if (bond.GetBondTypeAsDouble() != 1 or int(bond.GetStereo())
                    or int(bond.GetBondDir())):
                return None
    carbon_edges = sum(
        1 for i in carbons for n in _heavy_neighbors(mol, i, carbons) if i < n
    )
    if carbon_edges != len(carbons) - 1:
        return None
    external = [(i, n.GetIdx()) for i in heavy for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in heavy]
    if external != [(root_id, attachment_id)]:
        return None
    for terminal in terminals:
        parent = _heavy_neighbors(mol, terminal, heavy)
        if len(parent) != 1 or parent[0] not in carbons:
            return None
    return frozenset(carbons), frozenset(terminals), frozenset(heavy)


def _enrich_halo_candidate(mol, Chem, candidate, carbon_scope, terminals):
    """Add mapped terminal and compound-carbon prefixes to a carbon path."""
    from .component_mapping import mapped_branch

    path = candidate.path_atom_ids
    path_set = set(path)
    locants = {atom_id: pos for pos, atom_id in enumerate(path, 1)}
    prefixes = []
    owned_prefixes = set()
    for unit in candidate.prefixes:
        member_carbons = set(unit.atom_ids)
        boundaries = [(n.GetIdx(), i) for i in member_carbons
                      for n in mol.GetAtomWithIdx(i).GetNeighbors()
                      if n.GetIdx() in path_set]
        if len(boundaries) != 1:
            return None
        host, branch_root = boundaries[0]
        text, owned = mapped_branch(mol, Chem, branch_root, host)
        owned = frozenset(owned)
        if (not text or not member_carbons <= set(owned)
                or owned & owned_prefixes or set(owned) & path_set):
            return None
        owned_prefixes.update(owned)
        prefixes.append(PrefixUnit(locants[host], tuple(sorted(owned)), text))
    for terminal in terminals:
        neighbors = _heavy_neighbors(mol, terminal, carbon_scope)
        if len(neighbors) != 1:
            return None
        host = neighbors[0]
        if host in path_set:
            text = _TERMINAL_PREFIXES[mol.GetAtomWithIdx(terminal).GetAtomicNum()]
            if terminal in owned_prefixes:
                return None
            owned_prefixes.add(terminal)
            prefixes.append(PrefixUnit(locants[host], (terminal,), text))
        else:
            # The complete nested carbon branch owns its attached halo. It is
            # accounted for by the recursively mapped branch above.
            if terminal not in owned_prefixes:
                return None
    prefixes = tuple(prefixes)
    citation = _prefix_citation_locants(prefixes)
    if isinstance(candidate, RootedChainCandidate):
        key = (-len(candidate.path_atom_ids), -len(prefixes),
               tuple(sorted(unit.locant for unit in prefixes)), citation)
    else:
        key = (-len(candidate.path_atom_ids), candidate.root_locant,
               -len(prefixes), tuple(sorted(unit.locant for unit in prefixes)),
               citation)
    return replace(candidate, prefixes=prefixes, criteria_key=key)


def mapped_saturated_halo_carbon_prefix(mol, Chem, root_id, attachment_id, atoms):
    """Map a saturated carbon tree bearing ordinary halo and hydroxy groups.

    This route handles only neutral, nonisotopic acyclic carbon trees with
    terminal F/Cl/Br/I or OH and no assigned atom or bond stereo. A ``None`` result
    leaves out-of-scope structures to legacy producers; an empty name signals
    an admitted graph whose P-46 comparison or rendering remains unresolved.
    """
    admitted = _collect_terminal_halo_carbon_scope(mol, root_id, attachment_id)
    if admitted is None:
        return None
    carbon_scope, terminals, heavy_scope = admitted
    if set(heavy_scope) != set(atoms) or len(heavy_scope) > 64:
        return None
    from .component_mapping import MappedComponent, decorate_component
    from .constants import ROOTS

    root_neighbors = _heavy_neighbors(mol, root_id, carbon_scope)
    method = 1 if len(root_neighbors) <= 1 else 2
    if method == 1:
        paths = _rooted_leaf_paths(mol, root_id, carbon_scope)
        if not paths:
            return "", frozenset()
        length = max(map(len, paths))
        raw = [_candidate_for_path(mol, path, carbon_scope)
               for path in paths if len(path) == length]
        if any(item is None for item in raw):
            return "", frozenset()
        candidates = [_enrich_halo_candidate(mol, Chem, item, carbon_scope, terminals)
                      for item in raw]
        if any(item is None for item in candidates):
            return "", frozenset()
        frontier = rooted_chain_frontier(candidates)
    else:
        paths = _all_simple_paths_containing_root(mol, carbon_scope, root_id)
        raw = [_method2_candidate(mol, path, root_id, carbon_scope) for path in paths]
        raw = [item for item in raw if item is not None]
        if not raw:
            return "", frozenset()
        length = max(len(item.path_atom_ids) for item in raw)
        raw = [item for item in raw if len(item.path_atom_ids) == length]
        candidates = [_enrich_halo_candidate(mol, Chem, item, carbon_scope, terminals)
                      for item in raw]
        if any(item is None for item in candidates):
            return "", frozenset()
        beaten, unresolved = set(), set()
        for i, left in enumerate(candidates):
            for j in range(i + 1, len(candidates)):
                result = compare_alkanyl_paths(left, candidates[j])
                if result.preference == -1:
                    beaten.add(j)
                elif result.preference == 1:
                    beaten.add(i)
                elif result.preference is None:
                    unresolved.add((i, j))
        survivors = tuple(i for i in range(len(candidates)) if i not in beaten)
        if any(i in survivors and j in survivors for i, j in unresolved):
            return "", frozenset()
        frontier = tuple(candidates[i] for i in survivors)
    if not frontier:
        return "", frozenset()

    names = set()
    for candidate in frontier:
        path = candidate.path_atom_ids
        if len(path) not in ROOTS:
            return "", frozenset()
        mapping = {i: str(j) for j, i in enumerate(path, 1)}
        prefixes, embedded = [], set()
        for unit in candidate.prefixes:
            member_atoms = set(unit.atom_ids)
            bonds = [(i, n.GetIdx()) for i in path
                     for n in mol.GetAtomWithIdx(i).GetNeighbors()
                     if n.GetIdx() in member_atoms]
            if len(bonds) != 1:
                return "", frozenset()
            host, branch_root = bonds[0]
            from .component_mapping import mapped_branch
            text, owned = mapped_branch(mol, Chem, branch_root, host)
            if not text or set(owned) != member_atoms or embedded & set(owned):
                return "", frozenset()
            embedded.update(owned)
            locant = "" if len(path) == 1 else mapping[host]
            prefixes.append((locant, text))
        if set(mapping) | embedded != set(heavy_scope):
            return "", frozenset()
        parent = (ROOTS[len(path)] + "yl" if method == 1 else
                  ROOTS[len(path)] + "an-" + mapping[root_id] + "-yl")
        text = decorate_component(mol, Chem, MappedComponent(
            parent, tuple(mapping.items()), frozenset(mapping),
            tuple(prefixes), frozenset(embedded), systematic_parent=True,
            prefix_mode="unlocanted" if len(path) == 1 else "locanted"))
        if not text:
            return "", frozenset()
        names.add(text)
    if len(names) != 1:
        return '', frozenset()
    name = names.pop()
    if len(carbon_scope) == 1 and len({mol.GetAtomWithIdx(i).GetSymbol() for i in terminals}) > 1:
        # Mixed halo modifiers are one carbon-rooted component, not separate
        # substituents on the next parent. Preserve that boundary explicitly.
        name = '(' + name + ')'
    return name, frozenset(heavy_scope)
