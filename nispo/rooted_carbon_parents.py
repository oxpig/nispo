"""Mapped carbon-chain parents containing one required attachment atom."""

from .branch_rules import parent_chain_name
from .component_mapping import MappedComponent, mapped_general_branch, nitrogen_locant
from .formatting import prefix_alphabetization_key
from .multifunctional_parent import _paths
from .parent_annotations import isotope_numbering_key, stereo_numbering
from .parent_selection import locant_set


def grouped_site_orders(sites, positions, *, factor=1):
    """Enumerate prime assignments only within groups on the same parent atom."""
    from itertools import permutations, product
    from math import factorial, prod

    groups = {}
    for site, loc in zip(sites, positions, strict=True):
        groups.setdefault(loc, []).append(site)
    if factor * prod(factorial(len(group)) for group in groups.values()) <= 256:
        for order in product(*(permutations(group) for group in groups.values())):
            yield tuple(zip(groups, order))


def carbon_paths(mol, required, scope):
    """Return terminal-to-terminal paths through all required acyclic carbons."""
    root = next(iter(required))
    if mol.GetAtomWithIdx(root).IsInRing():
        return ()
    core = {root}
    stack = [root]
    while stack:
        idx = stack.pop()
        for a in mol.GetAtomWithIdx(idx).GetNeighbors():
            if (a.GetAtomicNum() == 6 and not a.IsInRing() and a.GetIdx() in scope
                    and a.GetIdx() not in core):
                core.add(a.GetIdx())
                stack.append(a.GetIdx())
    if len(core) > 32 or not required <= core:
        return ()
    return tuple(path for path in _paths(mol, core) if required.issubset(path))


def chain_orientations(mol, Chem, carbon, blocked, scope):
    """Yield fully accounted longest chains through the attachment and their numbering keys."""
    paths = carbon_paths(mol, {carbon}, scope)
    if not paths:
        return
    longest = max(map(len, paths))
    for path in paths:
        if len(path) != longest:
            continue
        for seq in dict.fromkeys((path, path[::-1])):
            mapping = {idx: str(i) for i, idx in enumerate(seq, 1)}
            framework = set(seq) | {blocked}
            occupied, prefixes = set(seq), []
            ports = ((idx, a.GetIdx()) for idx in seq for a in mol.GetAtomWithIdx(idx).GetNeighbors()
                     if a.GetIdx() not in framework and a.GetAtomicNum() > 1)
            complete = True
            for idx, other in ports:
                text, owned = mapped_general_branch(mol, Chem, other, idx)
                if not text or owned & occupied or not owned <= scope:
                    complete = False
                    break
                occupied.update(owned)
                prefixes.append((mapping[idx], text))
            if not complete or occupied != scope:
                continue
            parent = parent_chain_name(mol, seq)
            if not parent:
                continue
            orders = [mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() for a, b in zip(seq, seq[1:])]
            key = (-sum(order > 1 for order in orders), -orders.count(2), int(mapping[carbon]),
                   tuple(i for i, order in enumerate(orders, 1) if order > 1),
                   locant_set(loc for loc, _ in prefixes),
                   tuple(int(loc) for loc, _ in sorted(
                       prefixes, key=lambda item: (prefix_alphabetization_key(item[1]), int(item[0])))))
            yield key, MappedComponent(parent, tuple(mapping.items()), frozenset(seq), tuple(prefixes),
                                       scope - set(seq), systematic_parent=True)


def hydride_numberings(mol, Chem, host, skeleton, *, extended=False):
    """Parent orientations before suffix, substituent and N-site numbering."""
    from .formatting import complete_parent_name
    from .mapped_ring_parents import ring_systems, skeletons

    if mol.GetAtomWithIdx(host).IsInRing():
        ring = next((r for r in ring_systems(mol) if host in r), None)
        if not ring or not ring <= skeleton:
            return
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() and not (
            mol.GetAtomWithIdx(i).GetAtomicNum() != 6 and mol.GetAtomWithIdx(i).GetTotalNumHs())
            for i in ring)
        for hydride, seq, locants, _, _, saturated, key, method, *details in skeletons(mol, Chem, ring):
            mapping = dict(zip(seq, locants))
            if saturated or aromatic:
                yield hydride, mapping, (key,), ()
            elif extended and method == 'unsaturated-monocycle' and details:
                from .unsaturated_ring_parents import render_unsaturated_parent
                state = details[0].unsaturation
                name = render_unsaturated_parent(hydride, state)
                if name:
                    yield name, mapping, (key,), (state.multiple_bond_locants, state.double_bond_locants)
            elif extended and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                hs = [mapping[i] for i in seq if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
                      and mol.GetAtomWithIdx(i).GetTotalNumHs()]
                if len(hs) == 1:
                    yield hs[0] + 'H-' + hydride, mapping, (key, locant_set(hs)), ()
        return
    if extended and mol.GetAtomWithIdx(host).GetAtomicNum() in {14, 32, 50, 82}:
        from .rules45_97_chains_locants import _hydride, _path
        atoms = {i for i in skeleton if mol.GetAtomWithIdx(i).GetAtomicNum() in {14, 32, 50, 82}}
        path = _path(mol, atoms)
        if (not path or set(path) != atoms or any(mol.GetAtomWithIdx(i).IsInRing()
                or mol.GetAtomWithIdx(i).GetTotalValence() != 4 for i in path)
                or any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1 for a, b in zip(path, path[1:]))):
            return
        for seq in dict.fromkeys((path, path[::-1])):
            name = _hydride(tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in seq))
            if name:
                yield name, {i: str(n) for n, i in enumerate(seq, 1)}, (), ()
        return
    for path in carbon_paths(mol, {host}, skeleton):
        for seq in dict.fromkeys((path, path[::-1])):
            orders = [mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() for a, b in zip(seq, seq[1:])]
            hydride = parent_chain_name(mol, seq)
            if hydride:
                yield (complete_parent_name(hydride), {idx: str(n) for n, idx in enumerate(seq, 1)},
                       (-len(seq), -sum(o > 1 for o in orders), -orders.count(2)),
                       (tuple(n for n, o in enumerate(orders, 1) if o > 1),
                        tuple(n for n, o in enumerate(orders, 1) if o == 2)))


def skeleton_prefixes(mol, Chem, skeleton, scope):
    occupied = set(skeleton)
    prefixes = []
    for idx, loc in skeleton.items():
        for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
            root = atom.GetIdx()
            if root in occupied or root not in scope:
                continue
            word, atoms = mapped_general_branch(mol, Chem, root, idx)
            if not word or atoms & occupied or not atoms <= scope:
                return None
            occupied.update(atoms)
            prefixes.append((loc, word))
    return prefixes if occupied == scope else None


def substituent_numbering_key(mol, mapping, prefixes):
    from .locant_order import writing_locant_key
    # Internal N1 and N'1 distinguish the N prime from the site superscript.
    def key(loc):
        if loc.startswith(('C', 'O')):
            position = loc[1:].lstrip("'")
            if position.isdecimal():
                return (0, loc[0], writing_locant_key(position), len(loc) - len(position) - 1)
        return writing_locant_key(nitrogen_locant(loc))
    def series(locs):
        return tuple(sorted(key(loc) for loc in locs))
    citation = sorted(prefixes, key=lambda row: (prefix_alphabetization_key(row[1]), key(row[0])))
    stereo = stereo_numbering(mol, mapping)
    if stereo is None:
        return None
    return (series(loc for loc, _ in prefixes), tuple(key(loc) for loc, _ in citation),
            isotope_numbering_key(mol, mapping, locant_series=series), stereo)
