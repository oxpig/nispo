"""Functional-class names for ordinary silyl esters bearing a peroxy group."""

from .naming_methods import MethodDecision


def silyl_peroxy_ester_decision(mol, Chem):
    """Prove a single R-C(=O)-O-Si(R)2-O-O-R ester and own all atoms.

    Only carbon-only acyl and organyl components are admitted. The principal
    ester atoms are therefore not miscounted as replacement heterounits.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetRingInfo().NumRings()
            or any(a.GetAtomicNum() not in {6, 8, 14} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    silicon = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 14]
    if len(silicon) != 1 or silicon[0].GetTotalValence() != 4:
        return None
    si = silicon[0].GetIdx()
    oxygens = [n.GetIdx() for n in silicon[0].GetNeighbors() if n.GetAtomicNum() == 8]
    if (len(oxygens) != 2 or any(mol.GetBondBetweenAtoms(si, i).GetBondTypeAsDouble() != 1
                                  or mol.GetAtomWithIdx(i).GetDegree() != 2 for i in oxygens)):
        return None
    ester, peroxy = [], []
    for oidx in oxygens:
        other = next(n for n in mol.GetAtomWithIdx(oidx).GetNeighbors() if n.GetIdx() != si)
        if other.GetAtomicNum() == 6:
            ester.append((oidx, other.GetIdx()))
        elif other.GetAtomicNum() == 8:
            peroxy.append((oidx, other.GetIdx()))
    if len(ester) != 1 or len(peroxy) != 1:
        return None
    linker, acyl = ester[0]
    po, distal_o = peroxy[0]
    acyl_atom, distal = mol.GetAtomWithIdx(acyl), mol.GetAtomWithIdx(distal_o)
    if (mol.GetBondBetweenAtoms(linker, acyl).GetBondTypeAsDouble() != 1
            or acyl_atom.GetDegree() not in {2, 3} or distal.GetDegree() != 2
            or mol.GetBondBetweenAtoms(po, distal_o).GetBondTypeAsDouble() != 1):
        return None
    oxo = [a.GetIdx() for a in acyl_atom.GetNeighbors() if a.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(acyl, a.GetIdx()).GetBondTypeAsDouble() == 2
           and a.GetDegree() == 1]
    acid_roots = [a.GetIdx() for a in acyl_atom.GetNeighbors() if a.GetIdx() not in {linker, *oxo}]
    distal_roots = [a.GetIdx() for a in distal.GetNeighbors() if a.GetIdx() != po]
    si_roots = [a.GetIdx() for a in silicon[0].GetNeighbors() if a.GetIdx() not in oxygens]
    if (len(oxo) != 1 or len(acid_roots) > 1 or len(distal_roots) != 1
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in acid_roots + distal_roots + si_roots)):
        return None
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch
    from .formatting import format_prefixes

    occupied = {si, linker, acyl, oxo[0], po, distal_o}
    roles = [('ester-core', frozenset(occupied), ())]
    def branch(root, blocked):
        atoms = frozenset(collect_subtree(mol, root, blocked))
        if not atoms or atoms & occupied or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
            return None
        occupied.update(atoms)
        return atoms
    acid_atoms = branch(acid_roots[0], acyl) if acid_roots else frozenset()
    if acid_atoms is None:
        return None
    if acid_atoms:
        roles.append(('acyl-carbon-framework', acid_atoms, ()))
    distal_atoms = branch(distal_roots[0], distal_o)
    if distal_atoms is None:
        return None
    distal_text, owned = mapped_branch(mol, Chem, distal_roots[0], distal_o)
    if distal_text not in {'methyl', 'ethyl', 'propyl', 'butyl', 'tert-butyl'} or frozenset(owned) != distal_atoms:
        return None
    roles.append(('peroxy-organyl', distal_atoms, ()))
    prefixes = [('', distal_text + 'peroxy')]
    for root in si_roots:
        atoms = branch(root, si)
        if atoms is None:
            return None
        text, owned = mapped_branch(mol, Chem, root, si)
        if not text or frozenset(owned) != atoms:
            return None
        prefixes.append(('', text))
        roles.append(('silicon-organyl', atoms, ()))
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    if frozenset(occupied) != heavy:
        return None
    # The acid copy owns only the acyl and hydroxy atoms. The existing acid
    # writer chooses retained or systematic acid names and numbering.
    acid_ids = set(acid_atoms) | {acyl, oxo[0], linker}
    copy = Chem.RWMol(mol)
    for idx in sorted(set(range(mol.GetNumAtoms())) - acid_ids, reverse=True):
        copy.RemoveAtom(idx)
    acid = copy.GetMol()
    try:
        Chem.SanitizeMol(acid)
    except (RuntimeError, ValueError):
        return None
    from .api import mol_to_iupac
    acid_name = mol_to_iupac(acid)
    retained = {'formic acid': 'formate', 'acetic acid': 'acetate'}
    if acid_name in retained:
        anion = retained[acid_name]
    elif acid_name.endswith('oic acid'):
        anion = acid_name[:-len('oic acid')] + 'oate'
    else:
        return None
    prefix = format_prefixes(prefixes, omit_locants=True)
    if not prefix:
        return None
    name = prefix + 'silyl ' + anion
    return MethodDecision(name, 'functional-class', frozenset({acyl}), heavy,
                          'P-51.4.1.1/P-65.6.3/P-63.3.1', (), component_roles=tuple(roles))
