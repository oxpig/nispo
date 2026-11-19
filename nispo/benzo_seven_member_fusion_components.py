"""Numbered benzo-oxepine/thiepine components from P-25.2.2.4.

The names describe only an isolated two-face component. Locants 1–5 are the
heterocycle's nonfusion atoms, and 5a/9a its C-C fusion atoms. Oxygen or
sulfur occupies the lowest equivalent locant, so only 1, 2, and 3 occur.
No larger fused molecule or preferred total parent is selected here.
"""
import re


def benzo_seven_member_component(name):
    from rdkit import Chem
    from .fusion_components import component_from_graph
    from .ring_state_parent import _reference_states

    match = re.fullmatch(r'(?:([123])-|\[([123])\])(benzoxepin[eo]|benzothiepin[eo])', name)
    if match is None:
        return None
    locant = int(match[1] or match[2])
    element = 8 if match[3].startswith('benzox') else 16
    labels = ('1', '2', '3', '4', '5', '5a', '6', '7', '8', '9', '9a')
    numbers = tuple(element if index == locant - 1 else 6 for index in range(11))
    edges = tuple((i, (i + 1) % 11) for i in range(11)) + ((5, 10),)
    references = _reference_states(numbers, edges)
    if len(references) != 1:
        return None
    hydrogens, indicated, bonds = references[0]
    if (indicated or len(bonds) != len(edges)
            or hydrogens[locant - 1] != 0
            or any(hydrogens[i] != (0 if i in {5, 10} else 1)
                   for i in range(11) if i != locant - 1)):
        return None
    graph = Chem.RWMol()
    for number, count in zip(numbers, hydrogens):
        atom = Chem.Atom(number)
        atom.SetNumExplicitHs(count)
        atom.SetNoImplicit(True)
        graph.AddAtom(atom)
    for a, b, aromatic, order in bonds:
        if not aromatic and order not in (1, 2):
            return None
        graph.AddBond(a, b, Chem.BondType.AROMATIC if aromatic else
                      Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    molecule = graph.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except (ValueError, RuntimeError):
        return None
    if (molecule.GetNumAtoms() != 11 or molecule.GetRingInfo().NumRings() != 2
            or sorted(map(len, molecule.GetRingInfo().AtomRings())) != [6, 7]
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in molecule.GetAtoms())):
        return None
    return component_from_graph(molecule, dict(enumerate(labels)))
