"""Mapped oxygen-isotope alkoxy prefixes for fused-parent constructions."""
import re


def mapped_fused_branch(mol, Chem, root, host):
    """Retain an alkoxy oxygen isotope when the general branch writer declines."""
    from .component_mapping import mapped_branch
    word, atoms = mapped_branch(mol, Chem, root, host)
    atom = mol.GetAtomWithIdx(root)
    if word or atom.GetAtomicNum() != 8 or not atom.GetIsotope():
        return word, atoms
    probe = Chem.Mol(mol)
    probe.GetAtomWithIdx(root).SetIsotope(0)
    word, atoms = mapped_branch(probe, Chem, root, host)
    if not word or root not in atoms or any(
            mol.GetAtomWithIdx(i).GetIsotope() and mol.GetAtomWithIdx(i).GetAtomicNum() != 6
            for i in atoms if i != root):
        return '', frozenset()
    from .formatting import _ISOTOPE_DESCRIPTOR
    descriptor = _ISOTOPE_DESCRIPTOR.match(word)
    body = word[descriptor.end():] if descriptor else word
    if not re.fullmatch(r'[a-z]+oxy', body):
        return '', frozenset()
    existing = descriptor.group()[1:-1] + ',' if descriptor else ''
    return f'({existing}{atom.GetIsotope()}O){body}', atoms
