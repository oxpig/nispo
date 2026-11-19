"""Bounded mapped P-61.8 isocyanato substitutive construction."""
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def isocyanato_hydride_decision(mol, Chem):
    """Reuse the acyclic writer after proving a complete simple hydride graph."""
    atoms = list(mol.GetAtoms())
    if (not atoms or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
        or any(a.GetAtomicNum() == 1 or a.GetFormalCharge() or a.GetIsotope()
               or a.GetNumRadicalElectrons() or a.IsInRing() or int(a.GetChiralTag()) for a in atoms)
        or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds())):
        return None
    nitrogens = [a for a in atoms if a.GetAtomicNum() == 7]
    if len(nitrogens) != 1:
        return None
    n = nitrogens[0]
    nidx = n.GetIdx()
    if n.GetTotalNumHs() or n.GetDegree() != 2:
        return None
    singles = [a for a in n.GetNeighbors() if mol.GetBondBetweenAtoms(nidx, a.GetIdx()).GetBondTypeAsDouble() == 1]
    doubles = [a for a in n.GetNeighbors() if mol.GetBondBetweenAtoms(nidx, a.GetIdx()).GetBondTypeAsDouble() == 2]
    if len(singles) != 1 or len(doubles) != 1 or any(a.GetAtomicNum() != 6 for a in singles + doubles):
        return None
    host, center = singles[0], doubles[0]
    cidx = center.GetIdx()
    ends = [a for a in center.GetNeighbors() if a.GetIdx() != nidx]
    if (center.GetDegree() != 2 or center.GetTotalNumHs() or len(ends) != 1
        or ends[0].GetAtomicNum() != 8 or ends[0].GetDegree() != 1
        or mol.GetBondBetweenAtoms(cidx, ends[0].GetIdx()).GetBondTypeAsDouble() != 2):
        return None
    group = frozenset((nidx, cidx, ends[0].GetIdx()))
    carbons = frozenset(a.GetIdx() for a in atoms if a.GetAtomicNum() == 6 and a.GetIdx() != cidx)
    halos = frozenset(a.GetIdx() for a in atoms if a.GetAtomicNum() in (9, 17, 35, 53))
    if len(carbons) < 3 or group | carbons | halos != frozenset(range(len(atoms))):
        return None
    if any(a.GetDegree() != 1 or a.GetNeighbors()[0].GetIdx() not in carbons
           for a in atoms if a.GetIdx() in halos):
        return None
    for i in carbons:
        a = mol.GetAtomWithIdx(i)
        if (sum(other.GetIdx() in carbons for other in a.GetNeighbors()) > 2
            or any(mol.GetBondBetweenAtoms(i, other.GetIdx()).GetBondTypeAsDouble() != 1 for other in a.GetNeighbors())):
            return None
    if host.GetIdx() not in carbons:
        return None
    from .acyclic.rules import name_acyclic_record
    record = name_acyclic_record(mol, allow_ring_atoms=False)
    locants = tuple(record.stereo_locants)
    if not record.name or set(i for i, _ in locants) != carbons or record.name.count('isocyanato') != 1:
        return None
    return MethodDecision(record.name, 'substitutive', frozenset(), frozenset(range(len(atoms))),
        'P-61.8', locants, component_roles=(('parent-hydride', carbons, locants),
        ('isocyanato', group, ()), ('halogen-substituents', halos, ())),
        preference_issues=(SelectionLimitation('method-selection', 'P-41/P-44/P-61.8',
            'Complete comparison of naming methods remains unresolved.'),))
