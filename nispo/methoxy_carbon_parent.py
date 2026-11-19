"""An uninterrupted carbon parent with only methoxy substituents."""


def methoxy_carbon_parent(mol):
    """Prove the complete ordinary C/O tree and return its carbon path."""
    if mol.GetRingInfo().NumRings() or not 4 <= mol.GetNumHeavyAtoms() <= 32:
        return ()
    carbon, oxygen, methyl = set(), [], set()
    for atom in mol.GetAtoms():
        if (atom.GetIsotope() or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                or atom.HasQuery()):
            return ()
        if atom.GetAtomicNum() == 6:
            carbon.add(atom.GetIdx())
        elif atom.GetAtomicNum() == 8 and atom.GetDegree() == atom.GetTotalValence() == 2:
            oxygen.append(atom)
        else:
            return ()
    if not oxygen or any(b.GetBondTypeAsDouble() != 1 or b.HasQuery() for b in mol.GetBonds()):
        return ()
    hosts = set()
    for atom in oxygen:
        left, right = atom.GetNeighbors()
        if left.GetIdx() not in carbon or right.GetIdx() not in carbon:
            return ()
        if right.GetDegree() == 1 and left.GetDegree() > 1:
            left, right = right, left
        if left.GetDegree() != 1 or left.GetTotalNumHs() != 3 or right.GetDegree() == 1:
            return ()
        methyl.add(left.GetIdx())
        hosts.add(right.GetIdx())
    carbon.difference_update(methyl)
    if len(carbon) < 2 or not hosts <= carbon:
        return ()
    neighbors = {i: tuple(a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors()
                          if a.GetIdx() in carbon) for i in carbon}
    ends = [i for i, adjacent in neighbors.items() if len(adjacent) == 1]
    if len(ends) != 2 or any(not 1 <= len(adjacent) <= 2 for adjacent in neighbors.values()):
        return ()
    path = [ends[0]]
    previous, current = None, ends[0]
    while current != ends[1]:
        following = next(i for i in neighbors[current] if i != previous)
        path.append(following)
        previous, current = current, following
    # Each ether oxygen has a terminal methyl on its far side. Even a long
    # path through the graph can contain at most two heterounits (P-51.4).
    return tuple(path) if len(path) == len(carbon) else ()
