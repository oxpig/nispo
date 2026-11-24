"""Mapped carbon-chain parents containing one required attachment atom."""

from .branch_rules import parent_chain_name
from .component_mapping import MappedComponent, mapped_general_branch
from .formatting import prefix_alphabetization_key
from .multifunctional_parent import _paths
from .parent_selection import locant_set


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
