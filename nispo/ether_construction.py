"""P-63.2.4 general functional-class delivery for mapped cyclic ethers.

Each organyl word owns one complete side of the original C-O-C bridge.
This does not resolve preferred parent/method selection and is attempted only
after the normal writers have returned no name.
"""
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation

RULE = 'P-15.2/P-63.2.4.2/P-16.3/P-16.5/P-9'


def cyclic_ether_construction(mol, Chem):
    if (mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
        return None
    from .component_mapping import mapped_branch
    from .formatting import enclose_name_unit, prefix_alphabetization_key
    atoms = frozenset(a.GetIdx() for a in mol.GetAtoms())
    candidates = []
    for oxygen in mol.GetAtoms():
        if (oxygen.GetAtomicNum() != 8 or oxygen.IsInRing() or oxygen.GetIsotope()
                or oxygen.GetDegree() != 2 or oxygen.GetTotalNumHs(includeNeighbors=True)
                or oxygen.GetTotalValence() != 2):
            continue
        roots = tuple(oxygen.GetNeighbors())
        if (any(a.GetAtomicNum() != 6 or not a.IsInRing() for a in roots)
                or any(b.GetBondTypeAsDouble() != 1 for b in oxygen.GetBonds())
                or any(b.GetBondTypeAsDouble() > 1 and not b.GetIsAromatic()
                       and b.GetOtherAtom(root).GetAtomicNum() not in {1, 6}
                       for root in roots for b in root.GetBonds())):
            continue
        center = oxygen.GetIdx()
        words = []
        for root in roots:
            text, owned = mapped_branch(mol, Chem, root.GetIdx(), center)
            if not text or any(c.isspace() for c in text):
                break
            words.append((text, frozenset(owned)))
        if (len(words) != 2 or words[0][1] & words[1][1]
                or words[0][1] | words[1][1] | {center} != atoms):
            continue
        words.sort(key=lambda row: (prefix_alphabetization_key(row[0]), row[0]))
        if words[0][0] == words[1][0]:
            text = 'bis' + enclose_name_unit(words[0][0]) + ' ether'
        else:
            text = ' '.join(enclose_name_unit(word) for word, _owned in words) + ' ether'
        roles = (('ether-oxygen', frozenset({center}), ((center, ''),)),
                 ('first-organyl', words[0][1], ()), ('second-organyl', words[1][1], ()))
        candidates.append((text, center, roles))
    if not candidates:
        return None
    text, center, roles = min(candidates, key=lambda row: row[0])
    return MethodDecision(text, 'functional-class', frozenset({center}), atoms, RULE,
                          ((center, ''),), component_roles=roles,
                          preference_issues=(SelectionLimitation(
                              'cyclic-ether-functional-class', 'P-44/P-45/P-51/P-63.2.4.2',
                              'Both complete original organyl components are joined to the ether oxygen; preferred substitutive or multiplicative selection remains unresolved.'),))
