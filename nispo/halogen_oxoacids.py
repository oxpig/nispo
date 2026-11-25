"""Bond and charge accounting for organic derivatives of monobasic halogen acids."""
from functools import lru_cache

ADJECTIVES = {
    9: ('hypofluorous',),
    17: ('hypochlorous', 'chlorous', 'chloric', 'perchloric'),
    35: ('hypobromous', 'bromous', 'bromic', 'perbromic'),
    53: ('hypoiodous', 'iodous', 'iodic', 'periodic'),
}


@lru_cache(maxsize=2)
def derivative_query(Chem):
    return Chem.MolFromSmarts('[#9,#17,#35,#53]-[#7,$([#8;D2]),#15,#16,#33,#34,#51,#52]')


def oxo_atoms(halogen, ligand):
    """Return all acid oxos, counting X=O and X(+)-O(-) equivalently."""
    z = halogen.GetAtomicNum()
    if (z not in ADJECTIVES or halogen.GetNumRadicalElectrons() or halogen.GetChiralTag()
            or halogen.GetTotalNumHs() or halogen.IsInRing()):
        return None
    oxygen = []
    anions = 0
    for bond in halogen.GetBonds():
        atom = bond.GetOtherAtom(halogen)
        order = bond.GetBondTypeAsDouble()
        if atom.GetIdx() == ligand:
            if order != 1:
                return None
            continue
        if (atom.GetAtomicNum() != 8 or atom.GetDegree() != 1 or atom.GetTotalNumHs()
                or atom.GetNumRadicalElectrons()):
            return None
        if order == 1 and atom.GetFormalCharge() == -1:
            anions += 1
        elif order != 2 or atom.GetFormalCharge():
            return None
        oxygen.append(atom.GetIdx())
    if len(oxygen) >= len(ADJECTIVES[z]) or halogen.GetFormalCharge() != anions:
        return None
    return frozenset(oxygen)


def read_smiles(Chem, smiles):
    """Retry a hypervalent Lewis drawing with equivalent charge-separated terminal oxos."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None or not isinstance(smiles, (str, bytes)):
        return mol
    source = smiles.decode() if isinstance(smiles, bytes) else smiles
    if '=' not in source or not any(x in source for x in ('Cl', 'Br', 'I')):
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    changed = False
    for x in mol.GetAtoms():
        if x.GetAtomicNum() not in (17, 35, 53):
            continue
        ligands = [a for a in x.GetNeighbors() if a.GetAtomicNum() in (7, 8, 16, 34, 52)
                   and (a.GetAtomicNum() != 8 or a.GetDegree() == 2)]
        if len(ligands) != 1:
            continue
        for idx in oxo_atoms(x, ligands[0].GetIdx()) or ():
            b = mol.GetBondBetweenAtoms(x.GetIdx(), idx)
            if b.GetBondType() == Chem.BondType.DOUBLE:
                b.SetBondType(Chem.BondType.SINGLE)
                mol.GetAtomWithIdx(idx).SetFormalCharge(-1)
                x.SetFormalCharge(x.GetFormalCharge() + 1)
                changed = True
    if not changed or Chem.SanitizeMol(mol, catchErrors=True):
        return None
    # Apply normal SMILES hydrogen removal and stereo perception after sanitizing.
    return Chem.MolFromSmiles(Chem.MolToCXSmiles(mol))


def carbon_branch(mol, Chem, root, host):
    """Require a complete rooted carbon ligand including its assigned configuration."""
    from .component_mapping import mapped_general_branch, _general_branch_stereo_is_cited
    if mol.GetAtomWithIdx(root).GetAtomicNum() != 6:
        return '', frozenset()
    name, owned = mapped_general_branch(mol, Chem, root, host)
    if not name or any(mol.GetAtomWithIdx(i).GetFormalCharge()
                       or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in owned):
        return '', frozenset()
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in owned):
        from .constants import ROOTS
        from .isotope_mapping import locanted_isotope_prefix
        if len(owned) not in ROOTS:
            return '', frozenset()
        path, prev, idx = [], host, root
        while idx in owned:
            a = mol.GetAtomWithIdx(idx)
            if (a.GetAtomicNum() != 6 or a.IsInRing() or a.GetChiralTag()
                    or any(b.GetBondTypeAsDouble() != 1 for b in a.GetBonds())):
                return '', frozenset()
            path.append(idx)
            children = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != prev]
            if len(children) != 1:
                break
            prev, idx = idx, children[0]
        if len(path) != len(owned):
            return '', frozenset()
        word = ROOTS[len(path)] + 'yl'
        masses = {mol.GetAtomWithIdx(i).GetIsotope() for i in path}
        if len(masses) == 1:
            count = str(len(path)) if len(path) > 1 else ''
            descriptor = f'({masses.pop()}C{count})'
        else:
            descriptor = locanted_isotope_prefix(mol, ((i, j) for j, i in enumerate(path, 1)))
        return descriptor + word, owned
    if any(mol.GetAtomWithIdx(i).GetChiralTag() or any(b.GetStereo() for b in
           mol.GetAtomWithIdx(i).GetBonds()) for i in owned):
        if not _general_branch_stereo_is_cited(mol, Chem, owned, name):
            return '', frozenset()
    return name, owned


def derivative_decision(mol, Chem):
    """Return the shared whole-graph decision and whether a lossy fallback must stop."""
    sites = mol.GetSubstructMatches(derivative_query(Chem))
    if not sites:
        return None, False
    elements = {mol.GetAtomWithIdx(y).GetAtomicNum() for _, y in sites}
    if elements <= {15, 33, 51}:
        from .bluebook_hydride_assembly_methods import phosphorus_halide_decision
        return phosphorus_halide_decision(mol, Chem), False
    def has_terminal_oxo(x):
        atom = mol.GetAtomWithIdx(x)
        for bond in atom.GetBonds():
            oxygen = bond.GetOtherAtom(atom)
            if (oxygen.GetAtomicNum() == 8 and oxygen.GetDegree() == 1 and
                    (oxygen.GetFormalCharge() == -1 or bond.GetBondTypeAsDouble() == 2)):
                return True
        return False
    oxo = any(has_terminal_oxo(x) for x, _ in sites)
    if oxo and len(sites) == 1:
        x, y = sites[0]
        oxos = oxo_atoms(mol.GetAtomWithIdx(x), y)
        if oxos:
            senior = _senior_oxo_derivative(mol, Chem, x, y, oxos)
            if senior:
                return senior, True
    if elements <= {8, 16, 34, 52}:
        from .bluebook_ionic_acid_methods import hypohalite_ester_decision
        return hypohalite_ester_decision(mol, Chem, sites=sites), oxo
    if elements == {7}:
        from .halogen_acid_amide_methods import halogen_acid_amide_proposal, dihaloazane_decision
        if len(sites) == 1:
            proposal = halogen_acid_amide_proposal(mol, Chem)
            return (proposal.as_method_decision() if proposal else None), oxo
        if len({y for _, y in sites}) == 1:
            host = mol.GetAtomWithIdx(sites[0][1])
            if any(a.GetAtomicNum() == 6 for a in host.GetNeighbors()):
                return dihaloazane_decision(mol, Chem, sites), True
    return None, oxo


def senior_organyl_group(mol, Chem, owned, rank):
    """Check the organic arm even when a heavy inorganic center blocks its inventory."""
    if not any(mol.GetAtomWithIdx(i).GetAtomicNum() in (7, 8, 15, 16, 34, 52) for i in owned):
        return False
    from .functional_parent_extensions import capped
    from .extended_inventory import selection_inventory
    part, _ = capped(mol, Chem, owned)
    if part is None:
        return True
    inventory = selection_inventory(part)
    return not inventory.supported or any(g.seniority[0] <= rank for g in inventory.groups)


def _senior_oxo_derivative(mol, Chem, x, y, oxos):
    """Retain a senior suffix and cite the N/O/chalcogen-bound oxohalogen by Table 5.1."""
    from .constants import HALO
    from .extended_inventory import selection_inventory
    from .mapped_principal_general import mapped_principal_decision
    from .naming_methods import MethodDecision
    from .naming_result import SelectionLimitation

    # A single distinguishable monovalent placeholder keeps the original
    # ligand and its attachment on the senior parent. It cannot share a word
    # with any other atom. Configuration/label changes need a local mapping.
    if (mol.GetStereoGroups() or any(a.GetIsotope() or a.GetChiralTag() or a.GetNumRadicalElectrons()
                                     or a.HasQuery() for a in mol.GetAtoms()) or
            any(b.GetStereo() or b.HasQuery() for b in mol.GetBonds())):
        return None
    kept = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in oxos]
    used = {mol.GetAtomWithIdx(i).GetAtomicNum() for i in kept if i != x}
    available = next((z for z in ADJECTIVES if z not in used), None)
    if available is None:
        return None
    shadow = Chem.RWMol(mol)
    marker = shadow.GetAtomWithIdx(x)
    marker.SetAtomicNum(available)
    marker.SetFormalCharge(0)
    word = HALO[marker.GetSymbol()]
    for i in sorted(oxos, reverse=True):
        shadow.RemoveAtom(i)
    if Chem.SanitizeMol(shadow, catchErrors=True):
        return None
    inventory = selection_inventory(shadow)
    rank = 11 if mol.GetAtomWithIdx(y).GetAtomicNum() == 7 else 9
    if not inventory.principal or any(g.seniority[0] > rank for g in inventory.principal):
        return None
    method = mapped_principal_decision(shadow, Chem)
    if not method or method.name.count(word) != 1 or len(method.atoms) != len(kept):
        return None
    stems = {17: 'chlor', 35: 'brom', 53: 'iod'}
    stem = stems[mol.GetAtomWithIdx(x).GetAtomicNum()]
    oxo_prefix = (stem + 'osyl', stem + 'yl', 'per' + stem + 'yl')[len(oxos)-1]
    issue = SelectionLimitation('halogen-acid-derivative', 'P-41/P-59.1.9',
                                'The substitutive construction preserves the senior group; final preference remains open.')
    return MethodDecision(method.name.replace(word, oxo_prefix), 'substitutive',
                          frozenset(kept[i] for i in method.principal_centers),
                          frozenset(range(mol.GetNumAtoms())), 'P-41/P-59.1.9/P-61.3.2.2/P-68.5.3',
                          preference_issues=(issue,))
