"""Multiplicative names for two symmetrical trisulfane parent hydrides."""

from .naming_methods import MethodDecision


def symmetric_trisulfane_decision(mol, Chem):
    """Keep two complete trisulfane hydrides out of the heterounit count.

    Admits an unbranched carbon bridge between two identical, terminally
    alkylated trisulfanes. The closed graph has no principal characteristic
    group, branching, annotations, or other heteroatoms.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetRingInfo().NumRings()
            or any(a.GetAtomicNum() not in {6, 16} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(b.GetBondTypeAsDouble() != 1 or int(b.GetStereo())
                   or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    carbon = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
    sulfur = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 16}
    if len(sulfur) != 6 or any(mol.GetAtomWithIdx(i).GetDegree() != 2
                               or mol.GetAtomWithIdx(i).GetTotalValence() != 2 for i in sulfur):
        return None
    def components(indices):
        unseen, result = set(indices), []
        while unseen:
            stack, found = [next(iter(unseen))], set()
            while stack:
                idx = stack.pop()
                if idx in found:
                    continue
                found.add(idx)
                stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                             if n.GetIdx() in unseen and n.GetIdx() not in found)
            unseen -= found
            result.append(frozenset(found))
        return result
    cparts, sparts = components(carbon), components(sulfur)
    if len(cparts) != 3 or len(sparts) != 2 or any(len(part) != 3 for part in sparts):
        return None
    def contacts(part):
        return [(i, n.GetIdx()) for i in part for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetIdx() in sulfur]
    middle = [part for part in cparts if len(contacts(part)) == 2]
    outer = [part for part in cparts if len(contacts(part)) == 1]
    if len(middle) != 1 or len(outer) != 2:
        return None
    bridge = middle[0]
    if not 2 <= len(bridge) <= 8 or any(len(part) > 4 for part in outer):
        return None
    for part in cparts:
        if any(sum(n.GetIdx() in part for n in mol.GetAtomWithIdx(i).GetNeighbors()) > 2 for i in part):
            return None
        endpoints = {i for i in part if sum(n.GetIdx() in part
                       for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 1}
        if any(i not in endpoints for i, _s in contacts(part)):
            return None
    middle_contacts = contacts(bridge)
    if middle_contacts[0][0] == middle_contacts[1][0]:
        return None
    if {s for _c, s in middle_contacts} & {s for part in outer for _c, s in contacts(part)}:
        return None
    if any(sum(s in part for _c, s in middle_contacts) != 1 for part in sparts):
        return None
    if any(sum(contacts(part)[0][1] in sulfur_part for part in outer) != 1 for sulfur_part in sparts):
        return None
    from .component_mapping import mapped_branch
    from .constants import ROOTS
    names = []
    for part in outer:
        root, blocked = contacts(part)[0]
        text, atoms = mapped_branch(mol, Chem, root, blocked)
        if not text or frozenset(atoms) != part or text not in {'methyl', 'ethyl', 'propyl', 'butyl'}:
            return None
        names.append(text)
    if len(set(names)) != 1:
        return None
    root = ROOTS.get(len(bridge))
    if not root:
        return None
    name = f"1,1′-({root}ane-1,{len(bridge)}-diyl)bis(3-{names[0]}trisulfane)"
    roles = [('carbon-bridge', bridge, ())]
    roles.extend(('trisulfane-parent', part, ()) for part in sparts)
    roles.extend(('terminal-substituent', part, ()) for part in outer)
    return MethodDecision(name, 'multiplicative', frozenset(), frozenset(carbon | sulfur),
                          'P-51.3.1/P-51.4.1.1/P-21.2.2', (), component_roles=tuple(roles))
