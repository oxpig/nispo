"""Indexed RDKit iteration for naming loops that keep molecule size fixed.

Bind the native getter and size once rather than traversing RDKit's Python
sequence wrapper for every atom or bond. Do not use these iterators in loops
that add or remove atoms/bonds from the molecule being traversed.
"""


def iter_atoms(mol):
    return map(mol.GetAtomWithIdx, range(mol.GetNumAtoms()))


def iter_bonds(mol):
    return map(mol.GetBondWithIdx, range(mol.GetNumBonds()))
