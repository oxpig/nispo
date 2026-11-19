"""Complete mapped ionic phosphorus centers with explicit ligand roles.

A retained thiophosphate name owns terminal chalcogens only.  This general
substitutive construction retains a nonterminal =S/=Se/=Te ligand, its
ordinary carbon organyls, and its original nonstandard bonding number.
Preferred anion-parent and method comparison remains unresolved.
"""

from .component_mapping import MappedComponent, decorate_component, fully_covered, mapped_branch
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


_CHALCOGEN = {16: 'sulfanylidene', 34: 'selanylidene', 52: 'tellanylidene'}
_ANION_PREFIX = {8: 'oxidanidyl', 16: 'sulfanidyl',
                 34: 'selanidyl', 52: 'tellanidyl'}
_RULE = 'P-21.1.2/P-29.3.1/P-68.4.3/P-72/P-16.5.1.3.1'


def phosphorus_chalcogen_anion_decision(mol, Chem):
    """Construct one completely owned P(=X(R)R')(Y-)3 graph, else decline.

    The doubly attached chalcogen has bonding number four or six and one or
    two carbon organyls. Hydrogen counts are checked on its original atom;
    ligand annotations and competing charges are not inferred from text.
    """
    if (mol is None or not 6 <= mol.GetNumHeavyAtoms() <= 128
            or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1):
        return None
    phosphorus = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 15]
    if len(phosphorus) != 1:
        return None
    center = phosphorus[0]
    pidx = center.GetIdx()
    if (center.IsInRing() or center.GetFormalCharge() or center.GetIsotope()
            or center.GetNumRadicalElectrons() or int(center.GetChiralTag())
            or center.GetTotalValence() != 5
            or center.GetTotalNumHs(includeNeighbors=True)):
        return None
    anions, chalcogens = [], []
    for neighbor in center.GetNeighbors():
        idx = neighbor.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, idx)
        if int(bond.GetStereo()):
            return None
        order = bond.GetBondTypeAsDouble()
        if (neighbor.GetAtomicNum() in _ANION_PREFIX and order == 1
                and neighbor.GetFormalCharge() == -1
                and neighbor.GetDegree() == 1
                and not neighbor.GetTotalNumHs(includeNeighbors=True)
                and not neighbor.GetIsotope() and not neighbor.GetNumRadicalElectrons()):
            anions.append(idx)
        elif (neighbor.GetAtomicNum() in _CHALCOGEN and order == 2
              and not neighbor.IsInRing() and not neighbor.GetFormalCharge()
              and not neighbor.GetIsotope() and not neighbor.GetNumRadicalElectrons()
              and not int(neighbor.GetChiralTag())):
            chalcogens.append(idx)
        else:
            return None
    if len(anions) != 3 or len(chalcogens) != 1:
        return None
    cidx = chalcogens[0]
    chalcogen = mol.GetAtomWithIdx(cidx)
    bonding_number = chalcogen.GetTotalValence()
    if bonding_number not in {4, 6}:
        return None
    roots = [a.GetIdx() for a in chalcogen.GetNeighbors()
             if a.GetIdx() != pidx and a.GetAtomicNum() > 1]
    if (not 1 <= len(roots) <= 2
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   or mol.GetBondBetweenAtoms(cidx, i).GetBondTypeAsDouble() != 1
                   for i in roots)
            or chalcogen.GetTotalNumHs(includeNeighbors=True) != bonding_number - 2 - len(roots)):
        return None
    core_atoms = frozenset({pidx, *anions})
    owned = {cidx}
    branches, roles = [], []
    for root in roots:
        text, atoms = mapped_branch(mol, Chem, root, cidx)
        if (not text or not atoms or atoms & (core_atoms | owned)
                or any(mol.GetAtomWithIdx(i).GetFormalCharge()
                       or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                       for i in atoms)):
            return None
        # A complete carbon hydride or azaarene organyl is independently
        # bounded here. Other characteristic groups need their own source
        # partition instead of trusting a generic branch ownership claim.
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {1, 6}
               and not (mol.GetAtomWithIdx(i).GetAtomicNum() == 7
                        and mol.GetAtomWithIdx(i).GetIsAromatic()
                        and mol.GetAtomWithIdx(i).GetDegree() == 2)
               for i in atoms):
            return None
        branches.append(('', text))
        owned.update(atoms)
        roles.append(('chalcogen-organyl', atoms, ()))
    if not fully_covered(mol, core_atoms, owned):
        return None
    ligand = decorate_component(mol, Chem, MappedComponent(
        f'lambda{bonding_number}-' + _CHALCOGEN[chalcogen.GetAtomicNum()],
        ((cidx, ''),), frozenset({cidx}), tuple(branches),
        frozenset(owned - {cidx}), prefix_mode='mononuclear'))
    if not ligand:
        return None
    prefixes = tuple(('', _ANION_PREFIX[mol.GetAtomWithIdx(i).GetAtomicNum()])
                     for i in anions) + (('', ligand),)
    name = decorate_component(mol, Chem, MappedComponent(
        'lambda5-phosphane', ((pidx, ''),), frozenset({pidx}), prefixes,
        frozenset(set(anions) | owned), prefix_mode='mononuclear'))
    if not name:
        return None
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return MethodDecision(
        name, 'substitutive', frozenset({pidx}), heavy, _RULE,
        ((pidx, ''),), component_roles=(
            ('phosphorus-center', frozenset({pidx}), ((pidx, ''),)),
            ('terminal-anions', frozenset(anions), ()),
            ('chalcogen-center', frozenset({cidx}), ((cidx, ''),)), *roles),
        preference_issues=(SelectionLimitation(
            'phosphorus-chalcogen-anion-parent', 'P-70.4/P-72/P-44/P-45',
            'The original phosphorus, charged chalcogen groups and substituted '
            'chalcogenylidene ligand are complete; preferred anion-parent and '
            'method comparison remains unresolved.'),))


def mapped_imino_phosphanylium(mol, Chem):
    """Retain the original P+/O- pair and both fully mapped organic arms."""
    atoms = tuple(mol.GetAtoms())
    positive = [a for a in atoms if a.GetAtomicNum() == 15 and a.GetFormalCharge() == 1]
    if len(positive) != 1 or mol.GetStereoGroups():
        return None
    p, = positive
    center = p.GetIdx()
    if (p.IsInRing() or p.GetDegree() != 3 or p.GetTotalValence() != 4
            or p.GetTotalNumHs(includeNeighbors=True) or p.GetChiralTag()
            or any(a.GetIsotope() or a.GetNumRadicalElectrons() or a.HasQuery() for a in atoms)):
        return None
    ligands = {(b.GetOtherAtom(p).GetAtomicNum(), b.GetOtherAtom(p).GetFormalCharge(),
                b.GetBondTypeAsDouble()): b.GetOtherAtom(p) for b in p.GetBonds()}
    o = ligands.get((8, -1, 1))
    n = ligands.get((7, 0, 2))
    ligand = ligands.get((8, 0, 1)) or ligands.get((6, 0, 1))
    if (o is None or n is None or ligand is None or o.GetDegree() != 1
            or o.GetTotalNumHs(includeNeighbors=True) or n.GetDegree() != 2
            or n.GetTotalNumHs(includeNeighbors=True) or n.GetChiralTag()
            or mol.GetBondBetweenAtoms(center, n.GetIdx()).GetStereo()
            or {a.GetIdx() for a in atoms if a.GetFormalCharge()} != {center, o.GetIdx()}):
        return None
    child = next(a for a in n.GetNeighbors() if a.GetIdx() != center)
    if (child.GetAtomicNum() != 6
            or mol.GetBondBetweenAtoms(n.GetIdx(), child.GetIdx()).GetBondTypeAsDouble() != 1):
        return None
    from .component_mapping import _general_branch_stereo_is_cited, mapped_general_branch
    from .formatting import enclose_name_unit

    other, other_atoms = mapped_general_branch(mol, Chem, ligand.GetIdx(), center)
    imino, imino_atoms = mapped_general_branch(mol, Chem, child.GetIdx(), n.GetIdx())
    core = frozenset({center, n.GetIdx(), o.GetIdx()})
    if (not other or not imino or other_atoms & (imino_atoms | core) or imino_atoms & core
            or not fully_covered(mol, core, other_atoms | imino_atoms)
            or not _general_branch_stereo_is_cited(mol, Chem, other_atoms, other)
            or not _general_branch_stereo_is_cited(mol, Chem, imino_atoms, imino)):
        return None
    name = ('oxido' + enclose_name_unit(other)
            + enclose_name_unit(enclose_name_unit(imino) + 'imino') + 'phosphanylium')
    rule = 'P-68/P-70/P-72'
    return MethodDecision(
        name, 'additive', frozenset({center}), core | other_atoms | imino_atoms, rule,
        component_roles=(('phosphorus-core', core, ()), ('organyl', other_atoms, ()),
                         ('imino-organyl', imino_atoms, ())),
        preference_issues=(SelectionLimitation(
            'phosphorus-parent', rule,
            'The original charge-separated center and both ligands are mapped; preferred method comparison remains unresolved.'),))
