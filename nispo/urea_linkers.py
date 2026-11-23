"""Atom ownership for P-66.1.6.1.1.3 multiplicative urea linkers."""


def urea_link_atoms(mol, left, right):
    """Identify a complete –NH–CO–NH– group through its two nitrogen ports."""
    if left == right:
        return frozenset()
    ends = [mol.GetAtomWithIdx(i) for i in (left, right)]
    if any(a.GetAtomicNum() != 7 or a.GetDegree() != 2 or a.IsInRing()
           or a.GetTotalNumHs(includeNeighbors=True) != 1 or a.GetFormalCharge() or a.GetIsotope()
           or any(b.GetBondTypeAsDouble() != 1 for b in a.GetBonds()) for a in ends):
        return frozenset()
    common = {a.GetIdx() for a in ends[0].GetNeighbors()} & {a.GetIdx() for a in ends[1].GetNeighbors()}
    if len(common) != 1:
        return frozenset()
    center = mol.GetAtomWithIdx(next(iter(common)))
    if (center.GetAtomicNum() != 6 or center.GetDegree() != 3 or center.GetFormalCharge()
            or center.GetIsotope() or center.IsInRing()):
        return frozenset()
    oxygen = next(a for a in center.GetNeighbors() if a.GetIdx() not in {left, right})
    if (oxygen.GetAtomicNum() != 8 or oxygen.GetDegree() != 1 or oxygen.GetFormalCharge()
            or oxygen.GetIsotope() or mol.GetBondBetweenAtoms(center.GetIdx(), oxygen.GetIdx()).GetBondTypeAsDouble() != 2):
        return frozenset()
    return frozenset((left, center.GetIdx(), oxygen.GetIdx(), right))


def extended_urea_link(mol, Chem, scope, units):
    """Cite two equivalent divalent aromatic rings outside the urea group."""
    from .mapped_ring_parents import ring_systems, skeletons
    from .parent_selection import locant_key
    from .preferred_fused import _PREFERRED_BENZO_HETEROCYCLE_HYDRIDES

    if len(units) != 2 or len(scope) < 14:
        return ''
    ports = tuple(u.connector for u in units)
    if not all(mol.GetAtomWithIdx(i).IsInRing() for i in ports):
        return ''
    path = Chem.GetShortestPath(mol, *ports)
    center = len(path) // 2
    if len(path) < 5 or len(path) % 2 == 0:
        return ''
    nitrogens = (path[center - 1], path[center + 1])
    mid = urea_link_atoms(mol, *nitrogens)
    if not mid or not mid <= scope or path[center] not in mid:
        return ''
    cores = [core for core in ring_systems(mol) if core <= scope - mid]
    if len(cores) != 2 or frozenset.union(*cores) != scope - mid:
        return ''
    words = []
    for port, nitrogen, unit in zip(ports, nitrogens, units, strict=True):
        core = next((core for core in cores if port in core), ())
        neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(nitrogen).GetNeighbors()
                     if a.GetIdx() in core]
        if len(neighbors) != 1 or neighbors[0] == port or any(
                not mol.GetAtomWithIdx(i).GetIsAromatic() or mol.GetAtomWithIdx(i).GetIsotope()
                or mol.GetAtomWithIdx(i).GetFormalCharge() for i in core):
            return ''
        inner = neighbors[0]
        if {(i, a.GetIdx()) for i in core for a in mol.GetAtomWithIdx(i).GetNeighbors()
                if a.GetAtomicNum() > 1 and a.GetIdx() not in core} != {(inner, nitrogen), (port, unit.root)}:
            return ''
        labels = set()
        for parent, sequence, locants, *_ in skeletons(mol, Chem, core):
            mapping = dict(zip(sequence, locants, strict=True))
            a, b = mapping[inner], mapping[port]
            if a.isdigit() and b.isdigit():
                labels.add((parent, a, b))
        if not labels or len({parent for parent, _, _ in labels}) != 1:
            return ''
        parent, a, b = min(labels, key=lambda row: (locant_key(row[1]), locant_key(row[2])))
        parent = _PREFERRED_BENZO_HETEROCYCLE_HYDRIDES.get(parent, parent)
        words.append(f'{parent}-{a},{b}-diyl')
    return f'carbonylbis[azanediyl({words[0]})]' if words[0] == words[1] else ''
