"""Functional-class names for mixed O,S-diorganyl dithiophosphates."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _query(Chem):
    return Chem.MolFromSmarts('[#6]-[O;+0;X2]-[P;+0;X4](=[S;+0;D1])([O-;D1])-[S;+0;X2]-[#6]')


def anionic_dithiophosphate_name(mol, Chem):
    """Cite both ester attachments and the original anionic phosphorus unit."""
    match = mol.GetSubstructMatch(_query(Chem))
    if not match or mol.GetStereoGroups():
        return ''
    o_root, oxygen, phosphorus, thioxo, oxido, sulfur, s_root = match
    if any(a.GetFormalCharge() != (-1 if a.GetIdx() == oxido else 0)
           or a.GetIsotope() or a.GetNumRadicalElectrons() or a.HasQuery() for a in mol.GetAtoms()):
        return ''
    from .component_mapping import mapped_general_branch
    from .formatting import enclose_name_unit

    o_word, o_atoms = mapped_general_branch(mol, Chem, o_root, oxygen)
    s_word, s_atoms = mapped_general_branch(mol, Chem, s_root, sulfur)
    center = {oxygen, phosphorus, thioxo, oxido, sulfur}
    if (not o_word or not s_word or o_atoms & s_atoms or (o_atoms | s_atoms) & center
            or o_atoms | s_atoms | center != set(range(mol.GetNumAtoms()))):
        return ''
    o_word = o_word if o_word.isalpha() else enclose_name_unit(o_word)
    s_word = s_word if s_word.isalpha() else enclose_name_unit(s_word)
    return f'O-{o_word} S-{s_word} dithiophosphate'
