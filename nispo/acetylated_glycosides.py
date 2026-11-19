"""Original-graph assignment of O-acetylated aldohexopyranosides (P-102)."""

from .carbohydrates import _parent_word, _projection_tag, _reordered_tag
from .component_mapping import mapped_general_branch
from .formatting import enclose_name_unit
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def acetylated_glycoside_decision(mol, Chem, groups, atoms, centers, parts):
    if (mol.GetStereoGroups() or any(a.GetAtomicNum() == 1 or a.GetIsotope()
            or a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            or any(b.GetStereo() or b.GetBondDir() == Chem.BondDir.UNKNOWN for b in mol.GetBonds())):
        return None
    acyl_atoms = frozenset().union(*(part.atoms for part in parts))
    ester_oxygens = frozenset(group.ligand for group in groups)
    if (sum(len(part.atoms) for part in parts) != len(acyl_atoms)
            or len(ester_oxygens) != 4 or ester_oxygens & acyl_atoms):
        return None
    choices = []
    for face in mol.GetRingInfo().AtomRings():
        if len(face) != 6 or sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in face) != [6]*5 + [8]:
            continue
        if any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
               for a, b in zip(face, face[1:] + face[:1])):
            continue
        ring_o = next(i for i in face if mol.GetAtomWithIdx(i).GetAtomicNum() == 8)
        if mol.GetAtomWithIdx(ring_o).GetDegree() != 2:
            continue
        offset = face.index(ring_o)
        sequence = face[offset+1:] + face[:offset]
        for order in (sequence, sequence[::-1]):
            candidate = _pyranoside(mol, Chem, order, ring_o, acyl_atoms, ester_oxygens, atoms)
            if candidate is not None:
                choices.append(candidate)
    if len(choices) != 1:
        return None
    name, mapping, sugar, aglycone = choices[0]
    rule = 'P-102.5.6.2/P-102.5.6.1.1'
    roles = [('glycosyl', sugar, mapping), ('aglycone', aglycone, ())]
    roles.extend(('acetyl', part.atoms, part.atom_locants) for part in parts)
    return MethodDecision(name, 'functional-class', centers, atoms, rule, mapping,
                          component_roles=tuple(roles), preference_issues=(SelectionLimitation(
                              'carbohydrate-method', rule,
                              'The specialized carbohydrate name retains the source anomer and all acetyl sites.'),))


def _pyranoside(mol, Chem, sequence, ring_o, acyl_atoms, ester_oxygens, atoms):
    core = frozenset((*sequence, ring_o))
    if core & (ester_oxygens | acyl_atoms):
        return None
    carbons = tuple(mol.GetAtomWithIdx(i) for i in sequence)
    if any(a.GetDegree() != 3 or a.GetTotalNumHs() != 1
           or sum(n.GetIdx() in core for n in a.GetNeighbors()) != 2 or a.GetChiralTag() not in {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
           for a in carbons):
        return None
    outer = tuple(next(n for n in a.GetNeighbors() if n.GetIdx() not in core) for a in carbons)
    bridge, *secondary, primary = outer
    if (bridge.GetAtomicNum() != 8 or bridge.GetDegree() != 2 or bridge.IsInRing()
            or bridge.GetIdx() in ester_oxygens or primary.GetAtomicNum() != 6
            or primary.GetDegree() != 2 or primary.GetTotalNumHs() != 2
            or primary.IsInRing()
            or any(b.GetBondTypeAsDouble() != 1 for atom in (*carbons, primary, bridge)
                   for b in atom.GetBonds())):
        return None
    primary_o = next(n.GetIdx() for n in primary.GetNeighbors() if n.GetIdx() != sequence[-1])
    if ester_oxygens != {primary_o, *(o.GetIdx() for o in secondary)}:
        return None
    sugar = core | ester_oxygens | {bridge.GetIdx(), primary.GetIdx()}
    ligand = next(a.GetIdx() for a in bridge.GetNeighbors() if a.GetIdx() != sequence[0])
    text, aglycone = mapped_general_branch(mol, Chem, ligand, bridge.GetIdx())
    if not text or aglycone & (sugar | acyl_atoms) or aglycone | sugar | acyl_atoms != atoms:
        return None
    chain = (*sequence, primary.GetIdx())
    sides = []
    for j, side_oxygen in enumerate((*secondary, mol.GetAtomWithIdx(ring_o)), 1):
        atom = carbons[j]
        right_fischer = {chain[j-1]: 'top', chain[j+1]: 'bottom',
                         side_oxygen.GetIdx(): 'right', None: 'left'}
        order = tuple(a.GetIdx() for a in atom.GetNeighbors()) + (None,)
        sides.append('R' if atom.GetChiralTag() == _projection_tag(
            tuple(right_fischer[i] for i in order)) else 'L')
    side = 'right' if sides[-1] == 'R' else 'left'
    opposite = 'left' if side == 'right' else 'right'
    alpha_tag = _projection_tag((side, 'top', 'bottom', opposite))
    order = tuple(a.GetIdx() for a in carbons[0].GetNeighbors()) + (None,)
    alpha_tag = _reordered_tag(alpha_tag, (chain[1], bridge.GetIdx(), ring_o, None), order, Chem)
    anomer = 'α' if carbons[0].GetChiralTag() == alpha_tag else 'β'
    parent = _parent_word(6, 1, ''.join(sides), 'pyranose').removesuffix('e') + 'ide'
    if any(c in text for c in '()[]{}'):
        text = enclose_name_unit(text)
    mapping = tuple((i, str(j)) for j, i in enumerate(chain, 1))
    mapping += tuple((i, 'O'+str(j)) for j, i in enumerate(
        (bridge.GetIdx(), *(o.GetIdx() for o in secondary), ring_o, primary_o), 1))
    return f'{text} 2,3,4,6-tetra-O-acetyl-{anomer}-{parent}', mapping, sugar, aglycone
