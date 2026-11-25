"""Cached recognition patterns shared by independent and comprehensive naming."""
from functools import lru_cache


@lru_cache(maxsize=1)
def amide_query(Chem):
    return Chem.MolFromSmarts('[O;D1]=[S,Se,Te;!R;+0](-[#6,$([N;R])])-[N;!R;+0]')


@lru_cache(maxsize=1)
def urea_query(Chem):
    return Chem.MolFromSmarts('[N;!R]-[C;!R](=[O,S,Se,Te;D1;+0])-[N;!R]')


@lru_cache(maxsize=1)
def ylidene_urea_query(Chem):
    return Chem.MolFromSmarts('[N;!R]-[C;!R](=[O,S,Se,Te;D1;+0])-[N;!R]=[#6]')


@lru_cache(maxsize=1)
def hydrazone_query(Chem):
    return Chem.MolFromSmarts('[#6]=[N;!R;+0]-[N;!R;+0]')


@lru_cache(maxsize=1)
def amidine_query(Chem):
    return Chem.MolFromSmarts('[N;!R;+0]=[C,S,Se,Te;!R;+0]-[N;!R;+0]')


@lru_cache(maxsize=1)
def acylhydrazide_query(Chem):
    return Chem.MolFromSmarts('[O;D1]=[C,S,Se,Te;!R;+0]-[N;!R;+0]-[N;!R;+0]')


@lru_cache(maxsize=1)
def lactone_query(Chem):
    return Chem.MolFromSmarts('[#8;R;+0]@[#6;R;+0](=[O,S,Se,Te;D1;+0])@[#6;R]')


@lru_cache(maxsize=1)
def _repeated_suffix_queries(Chem):
    query = Chem.MolFromSmarts
    oxygen_acyl = query('[C;!R](=[O;D1])-[O;X2;+0]')
    acid = query('[C](=[O;D1])-[O;D1;H1]')
    amides = (query('[C;!R](=[O;D1])-[N;X3;!R]'), query('[S;!R](=O)(=O)-[N;X3;!R]'))
    nitrile = query('[C]#[N;D1]')
    lower = (query('[CH;D2](=[O;D1])-[#6]'), query('[C;D3](=[O;D1])(-[#6])-[#6]'),
             query('[#6]-[O;D1;H1]'), query('[N;X3;!R;+0;!$([N]~[!#6]);!$([N]-[#6]=[!#6])]'))
    return oxygen_acyl, acid, amides, nitrile, lower


def repeated_suffix_centers(mol, Chem):
    """Screen in P-41 order; the complete source inventory verifies each proposed class."""
    oxygen_acyl, acid, amides, nitrile, lower = _repeated_suffix_queries(Chem)
    # An acyclic acid or ester outranks all the remaining families. A lone
    # such center cannot be multiplied. Ring carbonyls can instead be
    # pseudoketones and must leave senior aldehydes eligible.
    oxo = mol.GetSubstructMatches(oxygen_acyl, maxMatches=2)
    if oxo:
        if len(oxo) == 2:
            matches = mol.GetSubstructMatches(acid, maxMatches=5)
            if 2 <= len(matches) <= 4:
                yield tuple(dict.fromkeys(m[0] for m in matches))
        return

    # An acylated N can belong to a hydrazide. Allow the inventory to
    # compare it with any sulfonamides before ruling out the lower classes.
    has_amide = False
    for query in amides:
        matches = mol.GetSubstructMatches(query, maxMatches=5)
        if matches:
            has_amide = True
            if 2 <= len(matches) <= 4:
                yield tuple(dict.fromkeys(m[0] for m in matches))
    if has_amide:
        return

    # C#N can also belong to a larger characteristic group. The inventory
    # decides whether that site is an ordinary nitrile.
    matches = mol.GetSubstructMatches(nitrile, maxMatches=5)
    if 2 <= len(matches) <= 4:
        yield tuple(m[0] for m in matches)
    for query in lower:
        matches = mol.GetSubstructMatches(query)
        if matches:
            if len(matches) >= 2 and (len(matches) <= 4 or mol.GetRingInfo().NumRings() >= 2):
                yield tuple(dict.fromkeys(m[0] for m in matches))
            return


def only_lower_substituents(mol, core):
    """Bound an unclassified residue to organyl, halo, hydroxy, ether and amine groups."""
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if atom.GetIdx() in core or z in {1, 6, 9, 17, 35, 53}:
            continue
        if z in {14, 32, 50, 82} and atom.GetTotalValence() == 4 and all(b.GetBondTypeAsDouble() == 1 for b in atom.GetBonds()):
            continue
        if atom.GetIsAromatic() and z in {7, 8, 16}:
            continue
        if z in {34, 52} and not atom.GetFormalCharge() and atom.GetDegree() == 2 and all(
                b.GetBondTypeAsDouble() == 1 and b.GetOtherAtom(atom).GetAtomicNum() == 6 for b in atom.GetBonds()):
            continue
        if z in {7, 8, 16} and all(b.GetBondTypeAsDouble() == 1 and
                b.GetOtherAtom(atom).GetAtomicNum() in {1, 6} for b in atom.GetBonds()):
            continue
        return False
    return True
