"""Fast fallback for a graph-mapped heteroaryl-attached ordinary amine."""


def bounded_heteroaryl_amine_requires_full(mol, Chem):
    """Retain an ordinary amine or a senior acid/alcohol suffix.

    The local mapped ring must be an isolated aromatic five- or six-member
    system with exactly one N/O/S atom, reached from an ordinary amine through
    at most two nonring carbon atoms. All ring atoms must be witnessed by the
    actual mapped branch writer. The full writer keeps global comparisons
    unresolved where its own evidence is incomplete.
    """
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if not rings:
        return False
    eligible = []
    for ring in rings:
        if (len(ring) not in {5, 6} or any(ring & other for other in rings if other != ring)
                or any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)):
            continue
        nums = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring]
        if (nums.count(6) != len(ring) - 1
                or sum(value in {7, 8, 16} for value in nums) != 1):
            continue
        eligible.append(ring)
    if not eligible:
        return False
    from .amine_components import ordinary_amine

    amines = [a.GetIdx() for a in mol.GetAtoms()
              if a.GetAtomicNum() == 7 and ordinary_amine(mol, a.GetIdx())]
    if len(amines) != 1:
        return False
    nitrogen = amines[0]
    pairs = []
    for first in mol.GetAtomWithIdx(nitrogen).GetNeighbors():
        if first.GetAtomicNum() != 6 or first.IsInRing():
            continue
        pending = [(first.GetIdx(), nitrogen, 1)]
        while pending:
            current, previous, depth = pending.pop()
            for other in mol.GetAtomWithIdx(current).GetNeighbors():
                idx = other.GetIdx()
                if idx == previous or other.GetAtomicNum() != 6:
                    continue
                bond = mol.GetBondBetweenAtoms(current, idx)
                if bond.GetBondTypeAsDouble() != 1:
                    continue
                ring = next((ring for ring in eligible if idx in ring), None)
                if ring is not None:
                    pairs.append((idx, current, ring))
                elif depth < 2 and not other.IsInRing() and not other.GetIsAromatic():
                    pending.append((idx, current, depth + 1))
    if not pairs:
        return False
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.principal) != 1
            or inventory.principal[0].kind not in {'amine', 'alcohol', 'acid'}):
        return False
    from .component_mapping import mapped_branch, mapped_branch_ring_atoms

    for root, blocked, ring in pairs:
        text, owned = mapped_branch(mol, Chem, root, blocked)
        if text and owned and ring <= mapped_branch_ring_atoms(
                mol, Chem, root, blocked, text, owned):
            return True
    return False
