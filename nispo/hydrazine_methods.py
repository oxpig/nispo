"""Ordinary terminal-halogen substitution on the retained hydrazine parent."""

from .component_mapping import MappedComponent, decorate_component


def halogenated_hydrazine_decision(mol, Chem):
    """P-14.3.4.2(b): distinguish this N--N hydride from halogen acid amides."""
    from .naming_methods import MethodDecision

    names = {9:'fluoro',17:'chloro',35:'bromo',53:'iodo'}
    atoms = tuple(mol.GetAtoms())
    if (mol.GetStereoGroups() or any(a.GetAtomicNum() == 1 or a.GetIsotope()
            or a.GetFormalCharge() or a.GetNumRadicalElectrons()
            or int(a.GetChiralTag()) or a.HasQuery() for a in atoms)
            or any(int(b.GetStereo()) for b in mol.GetBonds())):
        return None
    nitrogens = tuple(a.GetIdx() for a in atoms if a.GetAtomicNum() == 7)
    halogens = tuple(a.GetIdx() for a in atoms if a.GetAtomicNum() in names)
    if len(nitrogens) != 2 or not halogens or len(nitrogens)+len(halogens) != len(atoms):
        return None
    bond = mol.GetBondBetweenAtoms(*nitrogens)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    if any(mol.GetAtomWithIdx(idx).GetTotalValence() != 3 for idx in nitrogens):
        return None
    if any(mol.GetAtomWithIdx(idx).GetDegree() != 1
           or mol.GetAtomWithIdx(idx).GetNeighbors()[0].GetIdx() not in nitrogens
           or mol.GetAtomWithIdx(idx).GetBonds()[0].GetBondType() != Chem.BondType.SINGLE
           for idx in halogens):
        return None
    rows = []
    for sequence in (nitrogens, nitrogens[::-1]):
        mapping = dict(zip(sequence, ('1','2'), strict=True))
        prefixes = tuple((mapping[mol.GetAtomWithIdx(idx).GetNeighbors()[0].GetIdx()],
                          names[mol.GetAtomWithIdx(idx).GetAtomicNum()]) for idx in halogens)
        component = MappedComponent('hydrazine',tuple(mapping.items()),frozenset(nitrogens),
            prefixes,frozenset(halogens),systematic_parent=True)
        name = decorate_component(mol,Chem,component)
        if name:
            key = (tuple(sorted(int(loc) for loc,_ in prefixes)),
                   tuple((text,int(loc)) for loc,text in sorted(prefixes,key=lambda row:(row[1],int(row[0])))))
            rows.append((key,name,component))
    if not rows:
        return None
    _,name,component = min(rows,key=lambda row:row[0])
    return MethodDecision(name,'substitutive',frozenset(nitrogens),frozenset(range(len(atoms))),
        'P-21.1/P-14.3.4.2(b)/P-14.3.4.4/P-14.3.4.5',component.atom_locants,
        component_roles=(('hydrazine-parent',component.atoms,component.atom_locants),
                         ('halogen-prefixes',frozenset(halogens),())))


def hydrocarbyl_hydrazine_decision(mol, Chem):
    """Construct the preselected hydrazine parent with mapped C/H ligands.

    Exactly two neutral hydrazine nitrogens and carbon-only branches prove
    that no more senior characteristic group is hidden in this operation.
    The two orientations are compared by numerical locants and then by
    the locants of prefixes in alphabetical order.
    """
    from .naming_methods import MethodDecision
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {6, 7} or a.GetFormalCharge()
                   or (a.GetIsotope() and a.GetAtomicNum() != 6) or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(nitrogens) != 2 or any(a.IsInRing() or a.GetTotalValence() != 3 for a in nitrogens):
        return None
    first, second = (a.GetIdx() for a in nitrogens)
    bond = mol.GetBondBetweenAtoms(first, second)
    if bond is None or bond.GetBondTypeAsDouble() != 1:
        return None
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch
    from .formatting import format_prefixes, prefix_alphabetization_key

    core = frozenset((first, second))
    seen, branches, identity_by_text = set(core), [], {}
    for nidx, other in ((first, second), (second, first)):
        for neighbor in mol.GetAtomWithIdx(nidx).GetNeighbors():
            root = neighbor.GetIdx()
            if root == other:
                continue
            if (neighbor.GetAtomicNum() != 6
                    or mol.GetBondBetweenAtoms(nidx, root).GetBondTypeAsDouble() != 1):
                return None
            atoms = frozenset(collect_subtree(mol, root, nidx))
            if not atoms or atoms & seen or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
                return None
            text, owned = mapped_branch(mol, Chem, root, nidx)
            if not text or frozenset(owned) != atoms:
                return None
            identity = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms),
                                                rootedAtAtom=root, canonical=True)
            if text in identity_by_text and identity_by_text[text] != identity:
                return None
            identity_by_text[text] = identity
            seen.update(atoms)
            branches.append((nidx, root, atoms, text))
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    if frozenset(seen) != heavy or not branches:
        return None
    names = []
    for one, two in ((first, second), (second, first)):
        mapping = {one: '1', two: '2'}
        prefix_rows = [(mapping[nidx], text) for nidx, _root, _atoms, text in branches]
        if len(prefix_rows) == 1:
            prefix = format_prefixes([('', prefix_rows[0][1])], omit_locants=True)
        else:
            prefix = format_prefixes(prefix_rows)
        if not prefix:
            continue
        text_order = sorted({text for _loc, text in prefix_rows}, key=prefix_alphabetization_key)
        score = (tuple(sorted(int(loc) for loc, _text in prefix_rows)),
                 tuple(tuple(sorted(int(loc) for loc, value in prefix_rows if value == text))
                       for text in text_order))
        names.append((score, prefix + 'hydrazine', mapping))
    if not names:
        return None
    best_score = min(row[0] for row in names)
    best = [row for row in names if row[0] == best_score]
    if len({row[1] for row in best}) != 1:
        return None
    _score, name, mapping = best[0]
    roles = [('hydrazine-parent', core, tuple(mapping.items()))]
    roles.extend(('hydrocarbyl-substituent', atoms, ()) for _n, _root, atoms, _text in branches)
    return MethodDecision(name, 'substitutive', core, heavy, 'P-68.3.1.2.1/P-14.4',
                          tuple(mapping.items()), component_roles=tuple(roles))
