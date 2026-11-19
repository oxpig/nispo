"""P-65.1.1.1 and P-65.1.6.1 retained oxamic acid parent."""

from .naming_methods import MethodDecision


def oxamic_acid_decision(mol, Chem):
    """Name oxamic acid and carbon-only substitution on its amide nitrogen.

    Whole-graph ownership excludes competing functions and modifications of
    the acid. The rule explicitly omits N locants for these substituents.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {6, 7, 8} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(oxygens) != 3 or len(nitrogens) != 1:
        return None
    n = nitrogens[0]
    if n.IsInRing() or n.GetTotalValence() != 3:
        return None
    candidates = []
    for amide in n.GetNeighbors():
        if (amide.GetAtomicNum() != 6 or amide.GetDegree() != 3 or amide.IsInRing()
                or mol.GetBondBetweenAtoms(n.GetIdx(), amide.GetIdx()).GetBondTypeAsDouble() != 1):
            continue
        double_o = [a for a in amide.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                    and mol.GetBondBetweenAtoms(a.GetIdx(), amide.GetIdx()).GetBondTypeAsDouble() == 2]
        acids = [a for a in amide.GetNeighbors() if a.GetAtomicNum() == 6 and a.GetDegree() == 3
                 and not a.IsInRing()
                 and mol.GetBondBetweenAtoms(a.GetIdx(), amide.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(double_o) != 1 or len(acids) != 1:
            continue
        acid = acids[0]
        carbonyl = [a for a in acid.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                    and mol.GetBondBetweenAtoms(a.GetIdx(), acid.GetIdx()).GetBondTypeAsDouble() == 2]
        hydroxyl = [a for a in acid.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                    and a.GetTotalNumHs(includeNeighbors=True) == 1
                    and mol.GetBondBetweenAtoms(a.GetIdx(), acid.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(carbonyl) == len(hydroxyl) == 1:
            candidates.append((acid.GetIdx(), amide.GetIdx(), double_o[0].GetIdx(),
                               carbonyl[0].GetIdx(), hydroxyl[0].GetIdx(), n.GetIdx()))
    if len(candidates) != 1:
        return None
    core = frozenset(candidates[0])
    acid_idx, amide_idx = candidates[0][:2]
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence

    seen, units, roles, identities = set(core), [], [], {}
    for atom in n.GetNeighbors():
        root = atom.GetIdx()
        if root == amide_idx:
            continue
        if atom.GetAtomicNum() != 6 or mol.GetBondBetweenAtoms(n.GetIdx(), root).GetBondTypeAsDouble() != 1:
            return None
        branch = frozenset(collect_subtree(mol, root, n.GetIdx()))
        if not branch or branch & seen or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in branch):
            return None
        text, owned = mapped_branch(mol, Chem, root, n.GetIdx())
        if not text or frozenset(owned) != branch:
            return None
        identity = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(branch), rootedAtAtom=root, canonical=True)
        if text in identities and identities[text] != identity:
            return None
        identities[text] = identity
        path = all(sum(neighbor.GetIdx() in branch for neighbor in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2
                   for i in branch)
        units.append(PrefixUnit(text, 'simple' if text in {'phenyl', 'tert-butyl'} or path else 'compound'))
        roles.append(('amide-nitrogen-substituent', branch, ()))
        seen.update(branch)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    if frozenset(seen) != heavy or len(units) > 2:
        return None
    prefixes = mononuclear_prefix_sequence(units) if units else None
    if units and prefixes is None:
        return None
    name = (prefixes.text if prefixes else '') + 'oxamic acid'
    return MethodDecision(name, 'retained-parent', frozenset({acid_idx}), heavy,
                          'P-65.1.1.1/P-65.1.6.1', (),
                          component_roles=(('oxamic-acid-parent', core, ()), *roles))
