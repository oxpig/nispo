"""Retained cyanamide parents with graph-owned amino-nitrogen substituents."""

from .naming_methods import MethodDecision


def cyanamide_decision(mol, Chem):
    """Use P-66.1.6.2 for an isolated, neutral cyanic-acid amide.

    Substituents are carbon-only organyls, halogens, or hydroxyl/alkoxy
    groups. A competing senior characteristic group, ring-closing amide
    nitrogen, isotope, or stereochemical annotation uses another producer.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 17, 35, 53} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(nitrogens) != 2:
        return None
    sites = []
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6 or carbon.GetDegree() != 2:
            continue
        neighbors = carbon.GetNeighbors()
        triple = [n for n in neighbors if n.GetAtomicNum() == 7 and n.GetDegree() == 1
                  and mol.GetBondBetweenAtoms(carbon.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 3]
        amino = [n for n in neighbors if n.GetAtomicNum() == 7 and not n.IsInRing()
                 and n.GetTotalValence() == 3
                 and mol.GetBondBetweenAtoms(carbon.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(triple) == len(amino) == 1:
            sites.append((carbon.GetIdx(), triple[0].GetIdx(), amino[0].GetIdx()))
    if len(sites) != 1:
        return None
    carbon, cyano_n, amino_n = sites[0]
    core = frozenset((carbon, cyano_n, amino_n))
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence

    seen, units, roles, identities = set(core), [], [], {}
    halogens = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    for atom in mol.GetAtomWithIdx(amino_n).GetNeighbors():
        root = atom.GetIdx()
        if root == carbon:
            continue
        if mol.GetBondBetweenAtoms(amino_n, root).GetBondTypeAsDouble() != 1:
            return None
        if atom.GetAtomicNum() in halogens and atom.GetDegree() == 1:
            units.append(PrefixUnit(halogens[atom.GetAtomicNum()], 'simple'))
            roles.append(('N-substituent', frozenset((root,)), ()))
            seen.add(root)
            continue
        if atom.GetAtomicNum() not in {6, 8}:
            return None
        atoms = frozenset(collect_subtree(mol, root, amino_n))
        if not atoms or atoms & seen:
            return None
        if atom.GetAtomicNum() == 8:
            if (atom.GetTotalValence() != 2 or atom.GetDegree() not in {1, 2}
                    or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms if i != root)):
                return None
        elif any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
            return None
        text, owned = mapped_branch(mol, Chem, root, amino_n)
        if not text or frozenset(owned) != atoms:
            return None
        identity = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), rootedAtAtom=root, canonical=True)
        if text in identities and identities[text] != identity:
            return None
        identities[text] = identity
        path = all(sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2
                   for i in atoms)
        nature = 'simple' if text in {'phenyl', 'tert-butyl', 'hydroxy'} or path else 'compound'
        units.append(PrefixUnit(text, nature))
        roles.append(('N-substituent', atoms, ()))
        seen.update(atoms)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    if frozenset(seen) != heavy or len(units) > 2:
        return None
    prefixes = mononuclear_prefix_sequence(units) if units else None
    if units and prefixes is None:
        return None
    name = (prefixes.text if prefixes else '') + 'cyanamide'
    return MethodDecision(name, 'retained-parent', frozenset({carbon}), heavy,
                          'P-66.1.6.2/P-41/P-14.3.4.6', (),
                          component_roles=(('cyanamide-parent', core, ()), *roles))


def cyano_carboxamide_decision(mol, Chem):
    """Express N-cyano substitution on a senior ordinary carboxamide.

    This acyclic single-amide proof excludes every other functional group.
    Naming a private graph with the N substituents removed delegates the
    retained/systematic amide-parent choice to the existing amide machinery.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetRingInfo().NumRings()
            or any(a.GetAtomicNum() not in {6, 7, 8} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(nitrogens) != 2 or len(oxygens) != 1:
        return None
    sites = []
    for atom in nitrogens:
        if atom.GetTotalValence() != 3:
            continue
        roots = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 6
                 and mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
        cyano = [a for a in roots if a.GetDegree() == 2 and any(
            n.GetAtomicNum() == 7 and n.GetDegree() == 1
            and mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 3
            for n in a.GetNeighbors())]
        acyl = [a for a in roots if any(n.GetAtomicNum() == 8 and n.GetDegree() == 1
            and mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
            for n in a.GetNeighbors())]
        if len(cyano) == len(acyl) == 1 and cyano[0].GetIdx() != acyl[0].GetIdx():
            sites.append((atom.GetIdx(), cyano[0].GetIdx(), acyl[0].GetIdx()))
    if len(sites) != 1:
        return None
    nitrogen, cyano, acyl = sites[0]
    nitrile = next(a.GetIdx() for a in mol.GetAtomWithIdx(cyano).GetNeighbors()
                   if a.GetAtomicNum() == 7
                   and mol.GetBondBetweenAtoms(cyano, a.GetIdx()).GetBondTypeAsDouble() == 3)
    if any(n.GetAtomicNum() not in {6, 7, 8} for n in mol.GetAtomWithIdx(acyl).GetNeighbors()):
        return None
    extras = [a.GetIdx() for a in mol.GetAtomWithIdx(nitrogen).GetNeighbors()
              if a.GetIdx() not in {cyano, acyl}]
    if len(extras) > 1:
        return None
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch
    from .formatting import format_prefixes, join_prefix_parent

    prefixes = [('N', 'cyano')]
    extra_atoms = frozenset()
    if extras:
        root = extras[0]
        extra_atoms = frozenset(collect_subtree(mol, root, nitrogen))
        if (not extra_atoms or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in extra_atoms)
                or extra_atoms & {cyano, nitrile, acyl, nitrogen}):
            return None
        text, owned = mapped_branch(mol, Chem, root, nitrogen)
        if not text or frozenset(owned) != extra_atoms:
            return None
        prefixes.append(('N', text))
    removed = set(extra_atoms) | {cyano, nitrile}
    copy = Chem.RWMol(mol)
    for idx in sorted(removed, reverse=True):
        copy.RemoveAtom(idx)
    parent = copy.GetMol()
    try:
        Chem.SanitizeMol(parent)
    except (RuntimeError, ValueError):
        return None
    from .api import mol_to_iupac
    parent_name = mol_to_iupac(parent)
    if not parent_name.endswith('amide'):
        return None
    name = join_prefix_parent(format_prefixes(prefixes), parent_name)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    roles = [('carboxamide-parent', heavy - removed, ()),
             ('cyano-prefix', frozenset((cyano, nitrile)), ())]
    if extra_atoms:
        roles.append(('N-substituent', extra_atoms, ()))
    return MethodDecision(name, 'substitutive', frozenset({acyl}), heavy,
                          'P-41/P-66.1.1.3.1/P-66.1.6.2', (), component_roles=tuple(roles))


def cyano_amino_acid_decision(mol, Chem):
    """Keep an ordinary carboxylic acid senior to an N-cyano amino group.

    The copy retains the acid and carbon skeleton while removing only the
    cyano group, so the acid writer selects the correct parent and locants.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetRingInfo().NumRings()
            or any(a.GetAtomicNum() not in {6, 7, 8} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    nitrogen = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    oxygen = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(nitrogen) != 2 or len(oxygen) != 2:
        return None
    # The textual prefix is reattached only when the acid parent has an
    # unbranched, saturated carbon skeleton and hence no competing carbon
    # prefix whose alphabetical position could change.
    if (any(sum(n.GetAtomicNum() == 6 for n in atom.GetNeighbors()) > 2
            for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
            or any(b.GetBeginAtom().GetAtomicNum() == b.GetEndAtom().GetAtomicNum() == 6
                   and b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds())):
        return None
    acids = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        oxo = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8 and n.GetDegree() == 1
               and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2]
        hydroxyl = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 8 and n.GetDegree() == 1
                    and n.GetTotalNumHs(includeNeighbors=True) == 1
                    and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(oxo) == len(hydroxyl) == 1:
            acids.append(atom.GetIdx())
    if len(acids) != 1:
        return None
    sites = []
    for atom in nitrogen:
        if atom.GetTotalValence() != 3 or atom.GetTotalNumHs(includeNeighbors=True) != 1:
            continue
        carbons = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 6
                   and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(carbons) != 2:
            continue
        cyano = [c for c in carbons if c.GetDegree() == 2 and any(
            n.GetAtomicNum() == 7 and n.GetDegree() == 1
            and mol.GetBondBetweenAtoms(c.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 3
            for n in c.GetNeighbors())]
        if len(cyano) != 1:
            continue
        root = next(c.GetIdx() for c in carbons if c.GetIdx() != cyano[0].GetIdx())
        if root == acids[0]:
            continue
        triple_n = next(n.GetIdx() for n in cyano[0].GetNeighbors() if n.GetAtomicNum() == 7
                        and mol.GetBondBetweenAtoms(cyano[0].GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 3)
        sites.append((atom.GetIdx(), cyano[0].GetIdx(), triple_n))
    if len(sites) != 1:
        return None
    amino_n, cyano, triple_n = sites[0]
    copy = Chem.RWMol(mol)
    for idx in sorted((cyano, triple_n), reverse=True):
        copy.RemoveAtom(idx)
    parent = copy.GetMol()
    try:
        Chem.SanitizeMol(parent)
    except (RuntimeError, ValueError):
        return None
    from .api import mol_to_iupac
    parent_name = mol_to_iupac(parent)
    if not parent_name.endswith(' acid') or parent_name.count('amino') != 1:
        return None
    name = parent_name.replace('amino', '(cyanoamino)', 1)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    return MethodDecision(name, 'substitutive', frozenset(acids), heavy,
                          'P-41/P-65.1/P-66.1.6.2', (),
                          component_roles=(('acid-parent-and-amino', heavy - {cyano, triple_n}, ()),
                                           ('cyano-substituent', frozenset((cyano, triple_n)), ())))
