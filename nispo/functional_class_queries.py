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


@lru_cache(maxsize=2)
def suffix_query(Chem):
    # One match per potential suffix center, including alcohol and acid OH only once.
    return Chem.MolFromSmarts('[$([#6]-[O;D1;H1]),$([C](=O)-[N;X3]),'
                              '$([C]#[N;D1]),$([S](=O)(=O)-[N;X3])]')


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
