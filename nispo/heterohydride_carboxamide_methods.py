"""P-66.1.1.1.1.3 carboxamide suffixes on heteroatom hydride parents."""

from .naming_methods import MethodDecision


def heterohydride_carboxamide_decision(mol, Chem):
    """Prove one primary carboxamide on a neutral mononuclear heterohydride.

    Remaining ligands must be disjoint carbon-only organyls; N substitution,
    higher characteristic groups, isotopes, and stereo need other writers.
    """
    parents = {5: ('borane', 3), 14: ('silane', 4), 15: ('phosphane', 3),
               32: ('germane', 4), 33: ('arsane', 3), 50: ('stannane', 4),
               51: ('stibane', 3), 82: ('plumbane', 4), 83: ('bismuthane', 3)}
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {6, 7, 8, *parents} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    hosts = [a for a in mol.GetAtoms() if a.GetAtomicNum() in parents]
    if (len(hosts) != 1 or hosts[0].IsInRing()
            or hosts[0].GetTotalValence() != parents[hosts[0].GetAtomicNum()][1]):
        return None
    host = hosts[0]
    sites = []
    for atom in host.GetNeighbors():
        if atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetDegree() != 3:
            continue
        cidx = atom.GetIdx()
        if mol.GetBondBetweenAtoms(host.GetIdx(), cidx).GetBondTypeAsDouble() != 1:
            continue
        oxygen = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8 and n.GetDegree() == 1
                  and mol.GetBondBetweenAtoms(cidx, n.GetIdx()).GetBondTypeAsDouble() == 2]
        nitrogen = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 7 and n.GetDegree() == 1
                    and n.GetTotalNumHs(includeNeighbors=True) == 2
                    and mol.GetBondBetweenAtoms(cidx, n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(oxygen) == len(nitrogen) == 1:
            sites.append((cidx, oxygen[0].GetIdx(), nitrogen[0].GetIdx()))
    if len(sites) != 1:
        return None
    carbon, oxygen, nitrogen = sites[0]
    core = frozenset({host.GetIdx(), carbon, oxygen, nitrogen})
    from .phosphorus_ester_methods import _carbon_component
    from .component_mapping import mapped_branch
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence

    branches, units, seen = [], [], set(core)
    for atom in host.GetNeighbors():
        root = atom.GetIdx()
        if root == carbon:
            continue
        if atom.GetAtomicNum() != 6 or mol.GetBondBetweenAtoms(root, host.GetIdx()).GetBondTypeAsDouble() != 1:
            return None
        branch = _carbon_component(mol, root, host.GetIdx())
        if branch is None or branch & seen:
            return None
        text, owned = mapped_branch(mol, Chem, root, host.GetIdx())
        if not text or frozenset(owned) != branch:
            return None
        path = all(sum(n.GetIdx() in branch for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2
                   for i in branch)
        nature = 'simple' if text in {'phenyl', 'tert-butyl'} or path else 'compound'
        units.append(PrefixUnit(text, nature))
        branches.append(branch)
        seen.update(branch)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    if frozenset(seen) != heavy:
        return None
    prefix = mononuclear_prefix_sequence(units) if units else None
    if units and prefix is None:
        return None
    name = (prefix.text if prefix else '') + parents[host.GetAtomicNum()][0] + 'carboxamide'
    roles = [('heterohydride-parent-and-amide', core, ())]
    roles.extend(('organyl-substituent', branch, ()) for branch in branches)
    return MethodDecision(name, 'substitutive', frozenset({carbon}), heavy,
                          'P-66.1.1.1.1.3/P-41', (), component_roles=tuple(roles))
