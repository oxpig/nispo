"""P-61.9 substitutive isocyano names with a proved parent mapping."""

from .naming_methods import MethodDecision


def isocyano_decision(mol, Chem):
    """Name one carbon-bound isocyano group on a closed simple parent.

    The parent is an unbranched chain or a simple monocycle with at most one
    methyl substituent. A private fluoro analogue delegates parent/suffix
    choice and numbering. The admitted co-substituents cannot change their
    relative citation order when fluoro is replaced by isocyano.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {6, 7, 8} or a.GetIsotope()
                   or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) or a.HasQuery()
                   for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    from .parent_rules import _isocyanide_motifs

    motifs = _isocyanide_motifs(mol, Chem)
    if len(motifs) != 1:
        return None
    terminal, nitrogen, owner = motifs[0]
    if mol.GetAtomWithIdx(owner).GetAtomicNum() != 6:
        return None
    if any(a.GetFormalCharge() != (-1 if a.GetIdx() == terminal else 1 if a.GetIdx() == nitrogen else 0)
           for a in mol.GetAtoms()):
        return None
    other_nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetIdx() != nitrogen]
    if len(other_nitrogens) > 1 or any(
            not a.IsInRing() and not (a.GetDegree() == 1
                and a.GetBonds()[0].GetBondTypeAsDouble() == 3
                and a.GetNeighbors()[0].GetAtomicNum() == 6) for a in other_nitrogens):
        return None
    carbon = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIdx() != terminal}
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if owner not in carbon:
        return None
    acid = None
    if len(oxygens) == 1:
        oxygen = oxygens[0]
        if (oxygen.GetDegree() != 1 or oxygen.GetTotalNumHs(includeNeighbors=True) != 1
                or oxygen.GetNeighbors()[0].GetAtomicNum() != 6
                or oxygen.GetBonds()[0].GetBondTypeAsDouble() != 1):
            return None
    elif oxygens:
        if len(oxygens) != 2:
            return None
        acids = []
        for idx in carbon:
            atom = mol.GetAtomWithIdx(idx)
            double = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8 and n.GetDegree() == 1
                      and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() == 2]
            hydroxyl = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8 and n.GetDegree() == 1
                        and n.GetTotalNumHs(includeNeighbors=True) == 1
                        and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() == 1]
            if len(double) == len(hydroxyl) == 1:
                acids.append(idx)
        if len(acids) != 1:
            return None
        acid = acids[0]
    # An isocyanide directly on an acyl carbon follows acid-derivative rules.
    if any(n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(owner, n.GetIdx()).GetBondTypeAsDouble() == 2
           for n in mol.GetAtomWithIdx(owner).GetNeighbors()):
        return None
    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        if any(sum(n.GetIdx() in carbon for n in mol.GetAtomWithIdx(i).GetNeighbors()) > 2 for i in carbon):
            return None
    elif len(rings) == 1:
        ring = frozenset(rings[0])
        if (not 3 <= len(ring) <= 10 or owner not in ring
                or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7} for i in ring)):
            return None
        extras = carbon - ring - ({acid} if acid is not None else set())
        if len(extras) > 1 or any(mol.GetAtomWithIdx(i).GetDegree() != 1
                                  or mol.GetAtomWithIdx(i).GetTotalNumHs() != 3 for i in extras):
            return None
    else:
        return None
    marker = Chem.RWMol(mol)
    replacement = marker.GetAtomWithIdx(nitrogen)
    replacement.SetAtomicNum(9)
    replacement.SetFormalCharge(0)
    replacement.SetNumExplicitHs(0)
    replacement.SetNoImplicit(True)
    marker.RemoveAtom(terminal)
    marker = marker.GetMol()
    try:
        Chem.SanitizeMol(marker)
    except (RuntimeError, ValueError):
        return None
    from .api import mol_to_iupac
    name = mol_to_iupac(marker)
    if name.count('fluoro') != 1 or any(word in name for word in ('hydroxy', 'formyl', 'imino')):
        return None
    name = name.replace('fluoro', 'isocyano', 1)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    core = heavy - {terminal, nitrogen}
    return MethodDecision(name, 'substitutive', frozenset({acid if acid is not None else owner}),
                          heavy, 'P-61.9/P-41/P-44/P-45', (),
                          component_roles=(('parent', core, ()),
                                           ('isocyano', frozenset((terminal, nitrogen)), ())))


def isocyano_general_decision(mol, Chem):
    """Prove a broader parent choice using two order-bracketing markers.

    Fluoro alphabetizes before isocyano; methoxy alphabetizes after it. If
    both private analogues produce the same name after marker replacement,
    their parent, characteristic suffix, locants and relative prefix order
    are invariant across the isocyano citation position. A simple
    iodine-only co-substitution has its own direct alphabetic proof. Acyl
    isocyanides, charged or annotated structures, and ambiguous names decline.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() == 9 or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    from .parent_rules import _isocyanide_motifs

    motifs = _isocyanide_motifs(mol, Chem)
    if len(motifs) != 1:
        return None
    terminal, nitrogen, owner = motifs[0]
    if (mol.GetAtomWithIdx(owner).GetAtomicNum() != 6
            or any(a.GetFormalCharge() != (-1 if a.GetIdx() == terminal else 1 if a.GetIdx() == nitrogen else 0)
                   for a in mol.GetAtoms())
            or any(n.GetAtomicNum() in {8, 16, 34, 52}
                   and mol.GetBondBetweenAtoms(owner, n.GetIdx()).GetBondTypeAsDouble() == 2
                   for n in mol.GetAtomWithIdx(owner).GetNeighbors())):
        return None
    fluoro = Chem.RWMol(mol)
    atom = fluoro.GetAtomWithIdx(nitrogen)
    atom.SetAtomicNum(9)
    atom.SetFormalCharge(0)
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(True)
    fluoro.RemoveAtom(terminal)
    fluoro = fluoro.GetMol()
    methoxy = Chem.RWMol(mol)
    oxygen, carbon = methoxy.GetAtomWithIdx(nitrogen), methoxy.GetAtomWithIdx(terminal)
    oxygen.SetAtomicNum(8)
    oxygen.SetFormalCharge(0)
    oxygen.SetNumExplicitHs(0)
    oxygen.SetNoImplicit(True)
    carbon.SetFormalCharge(0)
    carbon.SetNumExplicitHs(0)
    carbon.SetNoImplicit(False)
    methoxy.GetBondBetweenAtoms(nitrogen, terminal).SetBondType(Chem.BondType.SINGLE)
    methoxy = methoxy.GetMol()
    try:
        Chem.SanitizeMol(fluoro)
        Chem.SanitizeMol(methoxy)
    except (RuntimeError, ValueError):
        return None
    from .api import mol_to_iupac

    before = mol_to_iupac(fluoro)
    after = mol_to_iupac(methoxy)
    if before.count('fluoro') != 1 or after.count('methoxy') != 1:
        return None
    first = before.replace('fluoro', 'isocyano', 1)
    second = after.replace('methoxy', 'isocyano', 1)
    if first != second:
        # With exactly one iodine on a simple carbon chain or carbocycle,
        # iodo precedes both isocyano and methoxy. The methoxy witness then
        # resolves their alphabetic tie directly, including orientation.
        iodines = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 53]
        carbons = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIdx() != terminal}
        if (len(iodines) != 1 or iodines[0].GetDegree() != 1
                or iodines[0].GetBonds()[0].GetBondTypeAsDouble() != 1
                or iodines[0].GetNeighbors()[0].GetIdx() not in carbons
                or sum(a.GetAtomicNum() == 7 for a in mol.GetAtoms()) != 1
                or any(a.GetAtomicNum() not in {6, 7, 53} for a in mol.GetAtoms())
                or after.count('iodo') != 1):
            return None
        neighbors = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in carbons]
                     for i in carbons}
        rings = mol.GetRingInfo().AtomRings()
        if ((not rings and any(len(row) > 2 for row in neighbors.values()))
                or (rings and (len(rings) != 1 or frozenset(rings[0]) != frozenset(carbons)
                               or any(len(row) != 2 for row in neighbors.values())))):
            return None
        first = second
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    return MethodDecision(first, 'substitutive', frozenset((owner,)), heavy,
                          'P-61.9/P-41/P-44/P-45', (),
                          component_roles=(('parent', heavy - {terminal, nitrogen}, ()),
                                           ('isocyano', frozenset((terminal, nitrogen)), ())))
