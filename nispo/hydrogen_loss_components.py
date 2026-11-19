"""Mapped P-71/P-73 hydrogen-loss operations on current component writers."""

from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def mapped_carbon_hydrogen_loss(mol, Chem, *, cation=False, allow_acyl=False):
    """Construct one carbon free-valence component with exact source ownership.

    A temporary cap defines the same rooted component operation used by the
    current P-29 writer. The cap is excluded from its atom cover. No closed-shell
    whole-molecule name is substituted for a radical or a charged structure.
    """
    if len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups():
        return None
    if (any(int(atom.GetChiralTag()) for atom in mol.GetAtoms())
            or any(int(bond.GetStereo()) or int(bond.GetBondDir()) for bond in mol.GetBonds())):
        return None
    centers = ([atom for atom in mol.GetAtoms() if atom.GetFormalCharge()] if cation else
               [atom for atom in mol.GetAtoms() if atom.GetNumRadicalElectrons()])
    if len(centers) != 1:
        return None
    center = centers[0]
    if center.GetAtomicNum() != 6:
        return None
    if cation:
        if center.GetFormalCharge() != 1 or any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
            return None
        order = 1
    else:
        if any(atom.GetFormalCharge() for atom in mol.GetAtoms()):
            return None
        order = center.GetNumRadicalElectrons()
        if order not in {1, 2, 3}:
            return None
    if center.GetTotalValence() != 4 - order:
        return None
    multiple_hetero = [b for b in center.GetBonds()
                      if b.GetBondTypeAsDouble() > 1 and b.GetOtherAtom(center).GetAtomicNum() not in {1, 6}]
    if multiple_hetero:
        acyl = (allow_acyl and not cation and order == 1 and len(multiple_hetero) == 1
                and multiple_hetero[0].GetBondTypeAsDouble() == 2
                and multiple_hetero[0].GetOtherAtom(center).GetAtomicNum() in {8, 16, 34, 52}
                and all(b.GetOtherAtom(center).GetAtomicNum() in {1, 6}
                        for b in center.GetBonds() if b.GetIdx() != multiple_hetero[0].GetIdx()))
        if not acyl:
            return None
    graph = Chem.RWMol(mol)
    root = graph.GetAtomWithIdx(center.GetIdx())
    root.SetNumRadicalElectrons(0)
    root.SetFormalCharge(0)
    cap = graph.AddAtom(Chem.Atom(6))
    graph.AddBond(center.GetIdx(), cap, {1: Chem.BondType.SINGLE,
                  2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}[order])
    capped = graph.GetMol()
    try:
        Chem.SanitizeMol(capped)
    except (ValueError, RuntimeError):
        return None
    if any(atom.GetNumRadicalElectrons() for atom in capped.GetAtoms()):
        return None
    from .component_mapping import mapped_branch
    name, owned = mapped_branch(capped, Chem, center.GetIdx(), cap)
    needed = frozenset(atom.GetIdx() for atom in mol.GetAtoms()
                       if atom.GetAtomicNum() > 1 or atom.GetIsotope())
    if not name or cap in owned or not needed <= owned or not owned <= set(range(mol.GetNumAtoms())):
        return None
    ending = {1: 'yl', 2: 'ylidene', 3: 'ylidyne'}[order]
    if not name.endswith(ending):
        return None
    if cation:
        name = name[:-2] + 'ylium'
    from .isotope_mapping import preserves_isotope_information
    if not preserves_isotope_information(mol, name):
        return None
    heavy = frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    rule = 'P-73.2.2.1/P-29' if cation else 'P-71.2.1/P-71.2.2/P-29'
    if multiple_hetero:
        rule = 'P-71.3.1/P-29'
    return MethodDecision(name=name, method='substitutive',
        principal_centers=frozenset((center.GetIdx(),)), atoms=heavy, rule=rule,
        component_roles=(('hydrogen-loss-component', heavy, ()),),
        preference_issues=(SelectionLimitation('hydrogen-loss-parent', rule,
            'The complete rooted component is constructed; the global preferred radical/ion parent comparison remains unresolved.'),))
