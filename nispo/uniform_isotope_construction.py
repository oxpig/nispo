"""P-82 front descriptors with a source-proved absence of positional choices."""

from collections import Counter
import re


def uniform_one_word_isotope_name(mol, Chem, construct):
    """Apply uniform nuclides to a complete current one-word construction.

    The unmodified graph has identical atoms, bonds, formal charges and
    hydrogen populations. Every atom of each specified element must carry
    the same isotope; otherwise a positional/characteristic-group decision
    is required and this operation declines. Assigned configuration needs
    the ordinary mapped annotation writers, since isotope removal can alter
    stereogenicity or CIP priorities.
    """
    if (mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or any(int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return ''
    labelled = [a for a in mol.GetAtoms() if a.GetIsotope()]
    if not labelled or any(a.GetAtomicNum() == 1 for a in labelled):
        return ''
    periodic = Chem.GetPeriodicTable()
    for atom in labelled:
        number, mass = atom.GetAtomicNum(), atom.GetIsotope()
        if (periodic.GetMassForIsotope(number, mass) <= 0
                or any(a.GetIsotope() != mass for a in mol.GetAtoms()
                       if a.GetAtomicNum() == number)):
            return ''
    # A global descriptor is not a proof of a descriptor boundary inside a
    # composite prefix. Keep this construction on carbon populations,
    # terminal carbonyl/chalcocarbonyl atoms and an unsubstituted urea core;
    # other roles need their actual local mapped component writer.
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    urea_core = (len(heavy) == 4 and
        sorted(a.GetAtomicNum() for a in heavy).count(7) == 2 and
        any(a.GetAtomicNum() == 6 and a.GetDegree() == 3
            and sum(n.GetAtomicNum() == 7 for n in a.GetNeighbors()) == 2
            and any(n.GetAtomicNum() in {8, 16, 34, 52}
                    and mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
                    for n in a.GetNeighbors()) for a in heavy))
    if not urea_core and any(a.GetAtomicNum() != 6 and not (
            a.GetAtomicNum() in {8, 16, 34, 52} and a.GetDegree() == 1
            and a.GetNeighbors()[0].GetAtomicNum() == 6
            and a.GetBonds()[0].GetBondTypeAsDouble() == 2) for a in labelled):
        return ''
    unmodified = Chem.Mol(mol)
    for atom in unmodified.GetAtoms():
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
    name = construct(unmodified)
    if (not name or any(c.isspace() or c in '()[]{}' for c in name)
            or re.search(r'(?:^|-)[NOSP](?:\^|\d|[′\x27,]|-)', name)):
        return ''
    populations = Counter((a.GetSymbol(), a.GetIsotope()) for a in labelled)
    parts = [f'{mass}{symbol}' + (str(count) if count > 1 else '')
             for (symbol, mass), count in sorted(populations.items())]
    return '(' + ','.join(parts) + ')' + name
