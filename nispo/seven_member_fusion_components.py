"""Mancude seven-member O/S fusion components from P-22/P-25.

Only the isolated component graph and its native numbering are supplied;
no larger fused parent, hydrogenated molecule, or preferred-name comparison
is inferred from the component vocabulary.
"""


def seven_member_chalcogen_component(name):
    from rdkit import Chem
    from .fusion_components import component_from_graph
    from .ring_state_parent import _reference_states

    elements = {'oxepine': 8, 'oxepino': 8, 'thiepine': 16, 'thiepino': 16}
    if name not in elements:
        return None
    numbers = (elements[name], 6, 6, 6, 6, 6, 6)
    edges = tuple((i, (i + 1) % 7) for i in range(7))
    references = _reference_states(numbers, edges)
    if len(references) != 1:
        return None
    hydrogens, indicated, bonds = references[0]
    if (hydrogens != (0, 1, 1, 1, 1, 1, 1) or indicated
            or len(bonds) != 7 or sum(order == 2 for _a, _b, _aromatic, order in bonds) != 3):
        return None
    graph = Chem.RWMol()
    for number, count in zip(numbers, hydrogens):
        atom = Chem.Atom(number)
        atom.SetNumExplicitHs(count)
        atom.SetNoImplicit(True)
        graph.AddAtom(atom)
    for a, b, aromatic, order in bonds:
        if aromatic or order not in (1, 2):
            return None
        graph.AddBond(a, b, Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
    molecule = graph.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except (ValueError, RuntimeError):
        return None
    if (molecule.GetRingInfo().NumRings() != 1 or molecule.GetNumAtoms() != 7
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in molecule.GetAtoms())):
        return None
    return component_from_graph(molecule, {i: str(i + 1) for i in range(7)})
