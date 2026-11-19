"""Mapped P-71.2.1.2 ordinary mononuclear hydride H-loss radicals."""

from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings, mapped_branch
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


_PARENTS = {5: (3, 'boranyl'), 7: (3, 'azanyl'), 8: (2, 'oxidanyl'),
            14: (4, 'silyl'), 15: (3, 'phosphanyl'), 16: (2, 'sulfanyl'),
            32: (4, 'germyl'), 33: (3, 'arsanyl'), 34: (2, 'selanyl'),
            50: (4, 'stannyl'), 51: (3, 'stibanyl'), 52: (2, 'tellanyl'),
            13: (3, 'alumanyl'), 31: (3, 'gallanyl'), 49: (3, 'indiganyl'),
            81: (3, 'thallanyl'), 82: (4, 'plumbyl'), 83: (3, 'bismuthanyl'),
            84: (2, 'polanyl'), 9: (1, 'fluoranyl'), 17: (1, 'chloranyl'),
            35: (1, 'bromanyl'), 53: (1, 'iodanyl'), 85: (1, 'astatanyl')}

_NONSTANDARD = {15: {5}, 33: {5}, 51: {5}, 83: {5},
                16: {4, 6}, 34: {4, 6}, 52: {4, 6}, 84: {4, 6},
                17: {3, 5, 7}, 35: {3, 5, 7}, 53: {3, 5, 7}, 85: {3, 5, 7}}


def hetero_hydride_radical(mol, Chem, *, allow_nonstandard=False):
    if mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if any(a.GetFormalCharge() for a in mol.GetAtoms()):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetNumRadicalElectrons()]
    if len(centers) != 1:
        return None
    center = centers[0]
    spec = _PARENTS.get(center.GetAtomicNum())
    order = center.GetNumRadicalElectrons()
    bonding_number = center.GetTotalValence() + order
    nonstandard = (spec is not None and bonding_number != spec[0]
                   and bonding_number in _NONSTANDARD.get(center.GetAtomicNum(), ()))
    if (spec is None or order not in {1, 2, 3}
            or bonding_number != spec[0] and not (allow_nonstandard and nonstandard)
            or center.IsInRing() or int(center.GetChiralTag())
            or any(b.GetBondTypeAsDouble() not in ({1, 2, 3} if allow_nonstandard else {1})
                   for b in center.GetBonds())):
        return None
    root = center.GetIdx()
    bindings = hydrogen_bindings(mol, {root: ''})
    embedded, prefixes, roles = set(), [], []
    for neighbor in center.GetNeighbors():
        if neighbor.GetAtomicNum() == 1:
            continue
        text, owned = mapped_branch(mol, Chem, neighbor.GetIdx(), root)
        if not text or owned & (bindings.keys() | embedded):
            return None
        cut_order = mol.GetBondBetweenAtoms(root, neighbor.GetIdx()).GetBondTypeAsDouble()
        if cut_order > 1 and not (text.endswith('ylidene' if cut_order == 2 else 'ylidyne')
                or cut_order == 2 and (text in {'oxo', 'thioxo'} or text.endswith('imino'))):
            return None
        prefixes.append(('', text))
        embedded.update(owned)
        roles.append(('radical-substituent', owned, ()))
    needed = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1 or a.GetIsotope()}
    if bindings.keys() | embedded != needed:
        return None
    parent = spec[1][:-2] + {1: 'yl', 2: 'ylidene', 3: 'ylidyne'}[order]
    if nonstandard:
        parent = f'lambda{bonding_number}-' + parent
    component = MappedComponent(parent, tuple(bindings.items()), frozenset(bindings),
        tuple(prefixes), frozenset(embedded), prefix_mode='mononuclear', systematic_parent=not nonstandard)
    name = decorate_component(mol, Chem, component)
    if not name:
        return None
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return MethodDecision(name=name, method='substitutive', principal_centers=frozenset({root}),
        atoms=heavy, rule='P-71.2.1.2/P-71.2.2/P-71.3.4/P-29/P-82', atom_locants=tuple(bindings.items()),
        component_roles=(('parent-radical', frozenset({root}), ((root, ''),)), *roles),
        preference_issues=(SelectionLimitation('radical-parent', 'P-71.3/P-71.7',
            'The mapped ordinary hydride radical is complete; preferred functional radical/method comparisons remain unresolved.'),))
