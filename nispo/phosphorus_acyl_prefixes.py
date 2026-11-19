"""Typed P-67.1.4.1.1.5 P(V) acyl prefixes with chalcogen ligands.

This is a subordinate-group writer, not a whole-molecule principal selector.
P-C/P-H and non-single attachment operations retain their distinct acid parents.
"""

from dataclasses import dataclass

from .component_mapping import mapped_branch
from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence


_ENDING = {8: 'phosphoryl', 16: 'phosphorothioyl',
           34: 'phosphoroselenoyl', 52: 'phosphorotelluroyl'}
_TERMINAL = {8: 'hydroxy', 16: 'sulfanyl', 34: 'selanyl', 52: 'tellanyl'}
_RETAINED_OXY = frozenset({'methoxy','ethoxy','propoxy','butoxy','phenoxy','tert-butoxy'})


@dataclass(frozen=True, slots=True)
class PhosphorusAcylPrefix:
    name: str
    phosphorus: int
    terminal_chalcogen: int
    ligand_atoms: tuple[frozenset[int], ...]
    atoms: frozenset[int]
    rule: str = 'P-67.1.4.1.1.5/P-16.3.6(a)/P-16.5.1.3.1'


def phosphorus_acyl_prefix(mol, Chem, idx, blocked):
    """Prove P(=X)(YR)(ZR)– with disjoint mapped ligand components."""
    atom = mol.GetAtomWithIdx(idx)
    bond = mol.GetBondBetweenAtoms(idx, blocked)
    if (atom.GetAtomicNum() != 15 or atom.IsInRing() or atom.GetFormalCharge()
            or atom.GetNumRadicalElectrons() or atom.GetIsotope() or int(atom.GetChiralTag())
            or atom.GetTotalValence() != 5 or atom.GetTotalNumHs(includeNeighbors=True)
            or bond is None or bond.GetBondType() != Chem.BondType.SINGLE):
        return None
    neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() != blocked]
    oxo = [n for n in neighbors if n.GetAtomicNum() in _ENDING
           and mol.GetBondBetweenAtoms(idx,n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
           and sum(a.GetAtomicNum() > 1 for a in n.GetNeighbors()) == 1
           and n.GetTotalValence() == 2 and not n.GetTotalNumHs(includeNeighbors=True)]
    if len(oxo) != 1 or len(neighbors) != 3:
        return None
    terminal = oxo[0]
    if terminal.GetIsotope() or terminal.GetFormalCharge() or terminal.GetNumRadicalElectrons():
        return None
    units, roles, owned = [], [], {idx,terminal.GetIdx()}
    for ligand in neighbors:
        root = ligand.GetIdx()
        if root == terminal.GetIdx():
            continue
        number = ligand.GetAtomicNum()
        if (number not in _TERMINAL or ligand.IsInRing() or ligand.GetFormalCharge()
                or ligand.GetIsotope() or ligand.GetNumRadicalElectrons()
                or int(ligand.GetChiralTag()) or ligand.GetTotalValence() != 2
                or any(n.GetAtomicNum() == 1 and n.GetIsotope() for n in ligand.GetNeighbors())
                or mol.GetBondBetweenAtoms(idx,root).GetBondType() != Chem.BondType.SINGLE):
            return None
        distal = [n.GetIdx() for n in ligand.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() != idx]
        if not distal:
            if ligand.GetTotalNumHs(includeNeighbors=True) != 1:
                return None
            text, atoms, nature = _TERMINAL[number], frozenset({root}), 'simple'
        else:
            if len(distal) != 1 or ligand.GetTotalNumHs(includeNeighbors=True):
                return None
            from .group14_mononuclear_methods import _carbon_organyl
            carbon_atoms = _carbon_organyl(mol,distal[0],root)
            if carbon_atoms is None or any(mol.GetAtomWithIdx(i).GetIsotope()
                    or int(mol.GetAtomWithIdx(i).GetChiralTag())
                    or mol.GetAtomWithIdx(i).GetFormalCharge()
                    or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in carbon_atoms):
                return None
            text, atoms = mapped_branch(mol,Chem,root,idx)
            if not text or atoms != carbon_atoms | {root}:
                return None
            nature = 'simple' if number == 8 and text in _RETAINED_OXY else 'compound'
        if atoms & owned or blocked in atoms:
            return None
        units.append(PrefixUnit(text,nature))
        roles.append(frozenset(atoms))
        owned.update(atoms)
    if len(units) != 2:
        return None
    if terminal.GetAtomicNum() == 8 and all(u.component == 'hydroxy' for u in units):
        name = 'phosphono'
    else:
        sequence = mononuclear_prefix_sequence(units)
        if sequence is None:
            return None
        name = sequence.text + _ENDING[terminal.GetAtomicNum()]
    return PhosphorusAcylPrefix(name,idx,terminal.GetIdx(),tuple(roles),frozenset(owned))
