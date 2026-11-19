"""Mapped P-31.1.4 pi matching for a bounded bicyclic substituent."""
from __future__ import annotations

from collections import defaultdict


def mapped_benzo_cyclobutane_substituent(mol, Chem, root, blocked, *, simple_only=False):
    """Construct a ring prefix only from a complete original-graph proof.

    This first scope is an all-carbon 6/4 edge-fused bicyclic ring system:
    the six-membered face is aromatic, the other face is saturated, and the
    prefix has one single cut. The theta numberings and both aromatic
    matchings are enumerated. Each external substituent must have its own
    complete, disjoint mapped branch. Fast mode admits terminal groups.
    Annotated local scopes decline; stereochemistry elsewhere is untouched.
    """
    from .cycle_utils import ordered_ring
    from .formatting import format_prefixes, join_prefix_parent, prefix_alphabetization_key
    from .unsaturated_ring_parents import UnsaturationState, render_unsaturated_parent
    from .von_baeyer_frontier import bicyclic_theta_frontier

    if mol.GetStereoGroups():
        return None
    cut = mol.GetBondBetweenAtoms(root, blocked)
    if cut is None or cut.GetBondTypeAsDouble() != 1:
        return None
    rings = tuple(map(frozenset, mol.GetRingInfo().AtomRings()))
    faces = [(six, four) for four in rings if len(four) == 4
             for six in rings if len(six) == 6 and root in six | four
             and len(four & six) == 2
             and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in six)]
    if len(faces) != 1:
        return None
    aromatic, short = faces[0]
    core = aromatic | short
    if (blocked in core or any(ring & core for ring in rings if ring not in (aromatic, short))
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in core)
            or any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in short - aromatic)):
        return None
    for a in core:
        for bond in mol.GetAtomWithIdx(a).GetBonds():
            b = bond.GetOtherAtomIdx(a)
            if b not in core or b < a:
                continue
            if int(bond.GetStereo()) or int(bond.GetBondDir()):
                return None
            if a in aromatic and b in aromatic:
                if not bond.GetIsAromatic():
                    return None
            elif bond.GetBondTypeAsDouble() != 1:
                return None

    branches = _mapped_branches(mol, Chem, root, blocked, core, simple_only)
    if branches is None:
        return None
    cycle = ordered_ring(mol, tuple(aromatic))
    if len(cycle) != 6:
        return None
    # The proved aromatic six-cycle has exactly two perfect matchings. All
    # remaining annular bonds are single, so both preserve the aromatic graph.
    matching_edges = tuple(tuple((cycle[i], cycle[(i + 1) % 6])
                                 for i in range(start, 6, 2)) for start in (0, 1))
    adjacency = {i: {n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                     if n.GetIdx() in core} for i in core}
    frontier = bicyclic_theta_frontier(adjacency)
    if (frontier.status != "complete" or not frontier.candidates
            or any(row.bridge_lengths != (4, 2, 0) for row in frontier.candidates)):
        return None

    choices = []
    for numbering in frontier.candidates:
        mapping = dict(numbering.locants)
        rows = [(str(mapping[host]), text) for host, text in branches]
        citation = defaultdict(list)
        for locant, text in rows:
            citation[text].append(int(locant))
        citation_locants = tuple(tuple(sorted(citation[text])) for text in
                                 sorted(citation, key=prefix_alphabetization_key))
        for edges in matching_edges:
            projected = tuple(sorted((min(mapping[a], mapping[b]), max(mapping[a], mapping[b]), 2)
                                     for a, b in edges))
            lows = tuple(a for a, _, _ in projected)
            compound_count = sum(b - a != 1 for a, b, _ in projected)
            all_locants = tuple(sorted(loc for a, b, _ in projected
                                       for loc in ((a,) if b - a == 1 else (a, b))))
            key = (compound_count, lows, all_locants, mapping[root],
                   tuple(sorted(int(loc) for loc, _ in rows)), citation_locants)
            state = UnsaturationState(projected, lows, lows, ())
            parent = render_unsaturated_parent("bicyclo[4.2.0]octane", state)
            substituent = parent[:-1] + f"-{mapping[root]}-yl"
            prefix = format_prefixes(rows, hierarchical_enclosures=True)
            choices.append((key, join_prefix_parent(prefix, substituent)))
    if not choices:
        return None
    best = min(key for key, _ in choices)
    names = {name for key, name in choices if key == best}
    return next(iter(names)) if len(names) == 1 else None


def _mapped_branches(mol, Chem, root, blocked, core, simple_only):
    if simple_only:
        from ._fast_engine.simple_prefixes import simple_external_prefix
        from .constants import HALO

        terminal_elements = {*HALO, 'C'}
    else:
        from .branch_rules import collect_subtree
        from .component_mapping import mapped_branch, mapped_branch_ring_atoms

    def one_cut(atoms, start, host):
        return {(i, n.GetIdx()) for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms} == {(start, host)}

    scope = None if simple_only else frozenset(collect_subtree(mol, root, blocked))
    if scope is not None and (not core <= scope or not one_cut(scope, root, blocked)):
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetFormalCharge()
           or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
           or int(mol.GetAtomWithIdx(i).GetChiralTag())
           or any(n.GetAtomicNum() == 1 for n in mol.GetAtomWithIdx(i).GetNeighbors())
           for i in (core if simple_only else scope)):
        return None
    branches, embedded = [], set()
    for host in core:
        for neighbor in mol.GetAtomWithIdx(host).GetNeighbors():
            child = neighbor.GetIdx()
            if child in core or (host, child) == (root, blocked):
                continue
            if mol.GetBondBetweenAtoms(host, child).GetBondTypeAsDouble() != 1:
                return None
            if simple_only:
                if (neighbor.GetDegree() != 1 or neighbor.GetSymbol() not in terminal_elements
                        or neighbor.GetIsotope() or neighbor.GetFormalCharge()
                        or neighbor.GetNumRadicalElectrons()
                        or neighbor.GetAtomicNum() == 6 and neighbor.GetTotalNumHs() != 3):
                    return None
                atoms = frozenset({child})
                text = simple_external_prefix(mol, child, host)
            else:
                atoms = frozenset(collect_subtree(mol, child, host))
                if not atoms or atoms & (core | embedded) or not atoms <= scope or not one_cut(atoms, child, host):
                    return None
                text, owned = mapped_branch(mol, Chem, child, host)
                rings = frozenset(i for i in atoms if mol.GetAtomWithIdx(i).IsInRing())
                if owned != atoms or rings and mapped_branch_ring_atoms(mol, Chem, child, host, text, owned) != rings:
                    return None
            if not text or any(c.isspace() for c in text):
                return None
            branches.append((host, text))
            embedded.update(atoms)
    return branches if scope is None or core | embedded == scope else None
