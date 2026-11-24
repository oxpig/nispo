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
def lactone_query(Chem):
    return Chem.MolFromSmarts('[#8;R;+0]@[#6;R;+0](=[O,S,Se,Te;D1;+0])@[#6;R]')
