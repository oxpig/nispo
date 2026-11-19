"""P-65.1.3.1.1 imidic acids, built from a mapped carboxylic acid."""

from .naming_methods import MethodDecision


def _imidic_ending(acid, count):
    """Apply functional replacement to a selected systematic acid name."""
    if count == 1:
        if acid == 'formic acid':
            return 'methanimidic acid'
        if acid == 'acetic acid':
            return 'ethanimidic acid'
        if acid.endswith('benzoic acid'):
            return acid[:-len('benzoic acid')] + 'benzenecarboximidic acid'
        if acid.endswith('carboxylic acid'):
            return acid[:-len('carboxylic acid')] + 'carboximidic acid'
        if acid.endswith('oic acid'):
            return acid[:-len('oic acid')] + 'imidic acid'
    if count == 2:
        if acid == 'oxalic acid':
            return 'ethanediimidic acid'
        if acid.endswith('dicarboxylic acid'):
            return acid[:-len('dicarboxylic acid')] + 'dicarboximidic acid'
        if acid.endswith('dioic acid'):
            return acid[:-len('dioic acid')] + 'diimidic acid'
    return ''


def imidic_acid_decision(mol, Chem):
    """Construct neutral mono/diimidic acids on an unannotated carbon skeleton.

    Replace each original terminal =NH with =O on a private graph, select the
    corresponding acid through the existing naming pipeline, and retain the
    original acid atoms in the returned roles. Other functional groups and
    annotations require separate P-41 and decoration comparisons.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 17, 35, 53} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    sites = []
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6 or carbon.IsInRing():
            continue
        cidx = carbon.GetIdx()
        nitrogens = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 7
                     and mol.GetBondBetweenAtoms(cidx, a.GetIdx()).GetBondTypeAsDouble() == 2
                     and a.GetDegree() == 1 and a.GetTotalNumHs(includeNeighbors=True) == 1]
        oxygens = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 8
                   and mol.GetBondBetweenAtoms(cidx, a.GetIdx()).GetBondTypeAsDouble() == 1
                   and a.GetDegree() == 1 and a.GetTotalNumHs(includeNeighbors=True) == 1]
        if len(nitrogens) == len(oxygens) == 1:
            others = [a for a in carbon.GetNeighbors()
                      if a.GetIdx() not in {nitrogens[0].GetIdx(), oxygens[0].GetIdx()}]
            if len(others) <= 1 and all(a.GetAtomicNum() == 6 for a in others):
                sites.append((cidx, nitrogens[0].GetIdx(), oxygens[0].GetIdx()))
    if len(sites) not in {1, 2}:
        return None
    suffix = frozenset(i for site in sites for i in site)
    if any(a.GetAtomicNum() != 6 and a.GetIdx() not in suffix
           and not (a.GetAtomicNum() in {9, 17, 35, 53} and a.GetDegree() == 1
                    and a.GetNeighbors()[0].GetAtomicNum() == 6
                    and mol.GetBondBetweenAtoms(a.GetIdx(), a.GetNeighbors()[0].GetIdx())
                    .GetBondTypeAsDouble() == 1)
           for a in mol.GetAtoms()):
        return None
    # Remaining atoms belong to the carbon skeleton or are halogen prefixes;
    # no competing characteristic group is silently ignored by the acid copy.
    copy = Chem.RWMol(mol)
    for _carbon, nitrogen, _oxygen in sites:
        atom = copy.GetAtomWithIdx(nitrogen)
        atom.SetAtomicNum(8)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    acid = copy.GetMol()
    try:
        Chem.SanitizeMol(acid)
    except (ValueError, RuntimeError):
        return None
    from .api import mol_to_iupac
    name = _imidic_ending(mol_to_iupac(acid), len(sites))
    if not name:
        return None
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    framework = heavy - suffix
    roles = (("imidic-acid-groups", suffix, ()), ("carbon-framework", framework, ()))
    return MethodDecision(name, 'substitutive', frozenset(site[0] for site in sites),
                          heavy, 'P-41/P-65.1.3.1.1', (), component_roles=roles)


def n_hydroxyimidic_acid_decision(mol, Chem):
    """P-65.1.3.3.1 for a single, otherwise unfunctionalized acid site.

    Select the systematic carboxylic-acid parent on a private graph.  The
    removed hydroxyl oxygen and the imine nitrogen still belong to the source
    group; no atom mapping is inferred from the generated acid name.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 17, 35, 53}
                   or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    sites = []
    for c in mol.GetAtoms():
        if c.GetAtomicNum() != 6 or c.IsInRing():
            continue
        ns = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 7
              and mol.GetBondBetweenAtoms(c.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
        os = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8
              and a.GetDegree() == 1 and a.GetTotalNumHs(includeNeighbors=True) == 1
              and mol.GetBondBetweenAtoms(c.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(ns) != 1 or len(os) != 1:
            continue
        n = ns[0]
        no = [a for a in n.GetNeighbors() if a.GetIdx() != c.GetIdx()]
        others = [a for a in c.GetNeighbors() if a.GetIdx() not in {n.GetIdx(), os[0].GetIdx()}]
        if (n.GetDegree() != 2 or len(no) != 1 or no[0].GetAtomicNum() != 8
                or no[0].GetDegree() != 1 or no[0].GetTotalNumHs(includeNeighbors=True) != 1
                or mol.GetBondBetweenAtoms(n.GetIdx(), no[0].GetIdx()).GetBondTypeAsDouble() != 1
                or len(others) > 1 or any(a.GetAtomicNum() != 6 for a in others)):
            continue
        sites.append((c.GetIdx(), n.GetIdx(), os[0].GetIdx(), no[0].GetIdx()))
    if len(sites) != 1:
        return None
    site = sites[0]
    group = frozenset(site)
    if any(a.GetAtomicNum() != 6 and a.GetIdx() not in group
           and not (a.GetAtomicNum() in {9, 17, 35, 53} and a.GetDegree() == 1
                    and a.GetNeighbors()[0].GetAtomicNum() == 6)
           for a in mol.GetAtoms()):
        return None
    copy = Chem.RWMol(mol)
    n = copy.GetAtomWithIdx(site[1])
    n.SetAtomicNum(8)
    n.SetNumExplicitHs(0)
    n.SetNoImplicit(True)
    copy.RemoveAtom(site[3])
    acid = copy.GetMol()
    try:
        Chem.SanitizeMol(acid)
    except (ValueError, RuntimeError):
        return None
    from .api import mol_to_iupac
    base = _imidic_ending(mol_to_iupac(acid), 1)
    if not base:
        return None
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    name = 'N-hydroxy' + ('-' if base[0].isdigit() else '') + base
    return MethodDecision(name, 'substitutive',
        frozenset({site[0]}), heavy, 'P-41/P-65.1.3.3.1', (),
        component_roles=(("N-hydroxyimidic-acid", group, ()),
                         ("carbon-framework", heavy - group, ())))
