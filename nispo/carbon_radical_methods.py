"""Mapped, source-bounded names for simple monovalent carbon radicals.

This opt-in producer handles one neutral carbon-centered H-loss radical on a
saturated acyclic carbon tree.  It deliberately declines rings, annotations,
heteroatoms, unsaturation, and more than one free valence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .naming_methods import MethodDecision

_RULE = "P-71.2.1.1/P-71.2.1.2/P-71.2.4/P-71.7/P-46.1"


@dataclass(frozen=True, slots=True)
class CarbonRadicalProposal:
    name: str
    parent_atoms: tuple[int, ...]
    atom_locants: tuple[tuple[int, str], ...]
    radical_center: int
    branches: tuple[tuple[str, int, tuple[int, ...]], ...]
    criterion_key: tuple


def _carbon_tree(mol):
    """Return carbon indices and the sole radical center, or None."""
    if (mol is None or mol.GetNumAtoms() == 0 or mol.GetNumHeavyAtoms() > 64
            or mol.GetRingInfo().NumRings() or mol.GetStereoGroups()):
        return None
    atoms = tuple(mol.GetAtoms())
    if any(a.GetAtomicNum() != 6 for a in atoms):
        return None
    if any(a.GetFormalCharge() or a.GetIsotope() or int(a.GetChiralTag())
           or any(n.GetAtomicNum() == 1 for n in a.GetNeighbors()) for a in atoms):
        return None
    bonds = tuple(mol.GetBonds())
    if any(b.GetBondTypeAsDouble() != 1.0 or int(b.GetStereo())
           or int(b.GetBondDir()) for b in bonds):
        return None
    if len(bonds) != len(atoms) - 1:
        return None
    centers = [a for a in atoms if a.GetNumRadicalElectrons()]
    if (len(centers) != 1 or centers[0].GetNumRadicalElectrons() != 1
            or centers[0].GetTotalValence() != 3):
        return None
    if any(a is not centers[0] and
           (a.GetNumRadicalElectrons() or a.GetTotalValence() != 4)
           for a in atoms):
        return None
    # Connectedness is not implied by the edge count alone.
    seen = set()
    stack = [atoms[0].GetIdx()]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors())
    if len(seen) != len(atoms):
        return None
    return frozenset(seen), centers[0].GetIdx()


def _path_between(graph, start, end):
    stack = [(start, -1, (start,))]
    while stack:
        current, previous, path = stack.pop()
        if current == end:
            return path
        for nxt in graph[current]:
            if nxt != previous:
                stack.append((nxt, current, path + (nxt,)))
    return ()


def _candidate_paths(graph, radical):
    # P-71.7 requires the radical center to be included in the chosen parent;
    # P-46.1 then selects among these carbon-only paths.
    vertices = sorted(graph)
    out = set()
    for i, left in enumerate(vertices):
        for right in vertices[i:]:
            path = _path_between(graph, left, right)
            if path and radical in path:
                out.add(path)
    return tuple(sorted(out))


def _off_parent_branches(mol, path):
    path_set = set(path)
    locants = {idx: str(pos) for pos, idx in enumerate(path, 1)}
    remaining = {a.GetIdx() for a in mol.GetAtoms()} - path_set
    rows = []
    while remaining:
        root = min(remaining)
        stack, component, attachments = [root], set(), []
        while stack:
            idx = stack.pop()
            if idx in component:
                continue
            component.add(idx)
            remaining.discard(idx)
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                other = neighbor.GetIdx()
                if other in path_set:
                    attachments.append((other, idx))
                elif other not in component:
                    stack.append(other)
        if len(attachments) != 1:
            return None
        host, branch_root = attachments[0]
        # Keep branch spelling inside the simple acyclic alkyl vocabulary.
        # A branched off-parent component needs its own P-46 choice and is
        # intentionally deferred here.
        subgraph = {idx: tuple(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                                if n.GetIdx() in component) for idx in component}
        if any(len(neighbors) > 2 for neighbors in subgraph.values()):
            return None
        termini = [idx for idx, neighbors in subgraph.items() if len(neighbors) <= 1]
        if len(termini) != 2 and len(component) > 1:
            return None
        endpoint = next((idx for idx in termini if idx != branch_root), branch_root)
        branch_path = _path_between(subgraph, branch_root, endpoint)
        if len(branch_path) != len(component):
            return None
        from .branch_rules import parent_chain_name
        text = parent_chain_name(mol, branch_path, as_substituent=True)
        if not text:
            return None
        rows.append((text, host, tuple(sorted(component))))
    return tuple(sorted(rows, key=lambda row: (
        path.index(row[1]) + 1, row[0], row[2])))


def _score(path, radical, branches):
    """Applicable P-46.1 sequence for this saturated carbon-only scope."""
    loc = path.index(radical) + 1
    branch_locs = sorted(path.index(host) + 1 for _, host, _ in branches)
    # P-46.1(b), (h), (k), and (l) are the numeric criteria. P-46.1(m) is
    # resolved separately only when candidates have the same prefix-name set.
    return (len(path), -loc, len(branches), tuple(-x for x in branch_locs))


def _citation_locant_score(path, branches):
    by_name = {}
    for text, host, _ in branches:
        by_name.setdefault(text, []).append(path.index(host) + 1)
    return tuple(-loc for text in sorted(by_name)
                 for loc in sorted(by_name[text]))


def _render(mol, path, radical, branches):
    from .branch_rules import parent_chain_name
    from .formatting import attach_suffix, format_prefixes, substituent_parent_suffix

    stem = parent_chain_name(mol, path)
    if not stem:
        return None
    unbranched = (len(branches) == 0 and all(
        mol.GetAtomWithIdx(idx).GetDegree() <= 2 for idx in path))
    radical_pos = path.index(radical) + 1
    if len(path) == 1:
        name = "methyl"
    elif unbranched and radical_pos in {1, len(path)}:
        # P-71.2.1.1: terminal radical from an unbranched acyclic parent.
        name = substituent_parent_suffix(stem)
    else:
        # P-71.2.1.2 / P-71.2.4: the locanted H-loss radical is preferred.
        name = attach_suffix(stem, "yl", (radical_pos,))
    prefix = format_prefixes([(str(path.index(host) + 1), text)
                              for text, host, _ in branches])
    return prefix + name


def carbon_radical_proposals(mol, Chem=None):
    """Return locally mapped parent candidates, or an empty tuple on decline."""
    proof = _carbon_tree(mol)
    if proof is None:
        return ()
    carbon, radical = proof
    graph = {idx: tuple(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors())
             for idx in carbon}
    paths = _candidate_paths(graph, radical)
    if not paths:
        return ()
    max_length = max(len(path) for path in paths)
    if max_length > 60:
        return ()
    maximal_paths = tuple(path for path in paths if len(path) == max_length)
    rows = []
    # Do not let a lower-ranked, renderable parent replace any maximum-length
    # candidate whose applicable branch facts cannot be represented.
    for raw in maximal_paths:
        for path in (raw, tuple(reversed(raw))):
            branches = _off_parent_branches(mol, path)
            if branches is None:
                return ()
            name = _render(mol, path, radical, branches)
            if not name:
                return ()
            mapping = tuple((idx, str(pos)) for pos, idx in enumerate(path, 1))
            rows.append(CarbonRadicalProposal(
                name, tuple(path), mapping, radical, branches,
                _score(path, radical, branches)))
    if not rows:
        return ()
    best = max(row.criterion_key for row in rows)
    contenders = tuple(row for row in rows if row.criterion_key == best)
    prefix_sets = {tuple(sorted(text for text, _, _ in row.branches))
                   for row in contenders}
    # If the P-46.1(m) criterion compares unlike sets of branch names, this
    # bounded writer does not infer how that tie is resolved.
    if len(prefix_sets) != 1:
        return ()
    best_citation = max(_citation_locant_score(row.parent_atoms, row.branches)
                        for row in contenders)
    winners = tuple(row for row in contenders
                    if _citation_locant_score(row.parent_atoms, row.branches)
                    == best_citation)
    # The last deterministic tie-break is used only where every rendered
    # locant criterion above ties; equal names are equivalent P-46 numberings.
    names = {row.name for row in winners}
    if len(names) != 1:
        return ()
    return tuple(sorted(winners, key=lambda row: (row.name, row.parent_atoms)))


def carbon_radical_decision(mol, Chem):
    """Return a fully owned opt-in MethodDecision for the bounded radical."""
    proposals = carbon_radical_proposals(mol, Chem)
    if not proposals or len({p.name for p in proposals}) != 1:
        from .hydrogen_loss_components import mapped_carbon_hydrogen_loss
        return mapped_carbon_hydrogen_loss(mol, Chem)
    proposal = proposals[0]
    all_atoms = frozenset(a.GetIdx() for a in mol.GetAtoms())
    all_heavy = frozenset(a.GetIdx() for a in mol.GetAtoms()
                          if a.GetAtomicNum() > 1)
    roles = [("parent_hydride", frozenset(proposal.parent_atoms),
              proposal.atom_locants)]
    for text, host, component in proposal.branches:
        attachment = proposal.parent_atoms.index(host) + 1
        roles.append((f"alkyl_prefix_at_{attachment}", frozenset(component), ()))
    owned = [idx for _, role_atoms, _ in roles for idx in role_atoms]
    if (len(owned) != len(set(owned)) or set(owned) != set(all_atoms)
            or set(owned) != set(all_heavy)):
        return None
    return MethodDecision(
        name=proposal.name, method="substitutive",
        principal_centers=frozenset({proposal.radical_center}), atoms=all_heavy,
        rule=_RULE, atom_locants=proposal.atom_locants,
        component_roles=tuple(roles),
    )
