"""Source retained-parent permissions, kept separate from derivatives."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _vocabulary():
    from .resource_io import read_json_resource
    records=read_json_resource('resources','bluebook_retained_hydrocarbon_parents.json')
    rows=records['parents']
    if len(rows)!=19 or len({row['smiles'] for row in rows})!=19:
        raise ValueError('The complete source retained-hydride vocabulary is unavailable.')
    return {row['smiles']:row for row in rows}


def retained_hydrocarbon_parent_decision(mol,Chem):
    acetylene = _unsubstituted_acetylene(mol, Chem)
    if acetylene is not None:
        return acetylene
    from .retained_von_baeyer_parents import parent_decision as cage_parent_decision
    cage=cage_parent_decision(mol,Chem)
    if cage is not None:return cage
    if (mol is None or not 9<=mol.GetNumAtoms()<=32 or mol.GetNumAtoms()!=mol.GetNumHeavyAtoms()
            or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol))!=1):return None
    if any(a.GetAtomicNum()!=6 or not a.IsInRing() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
           or a.GetIsotope() or int(a.GetChiralTag()) for a in mol.GetAtoms()):return None
    if any(b.GetBondType() not in {Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC}
           or int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds()):return None
    copy=Chem.Mol(mol)
    for atom in copy.GetAtoms():atom.SetAtomMapNum(0)
    Chem.SanitizeMol(copy)
    key=Chem.MolToSmiles(copy,canonical=True,isomericSmiles=True);record=_vocabulary().get(key)
    if record is None:return None
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    if (CalcMolFormula(copy)!=record['formula']
            or copy.GetNumBonds()-copy.GetNumAtoms()+1!=record['independent_cycle_rank']):return None
    from .naming_methods import MethodDecision
    atoms=frozenset(range(mol.GetNumAtoms()))
    # No arbitrary symmetry frame is exported for this unannotated whole
    # parent. Typed joins/derivatives still require their own original maps.
    return MethodDecision(record['name'],'retained-hydrocarbon-parent',frozenset(),atoms,
        'P-25.1.1/Table-2.7/P-25.7',component_roles=(('complete-source-retained-hydride',atoms,()),))


def _unsubstituted_acetylene(mol, Chem):
    """P-31.1.2.1/P-52.2.1.1: retained parent, never a substituted alkyne.

    Isotopic modification uses the original atom map under P-82; it does not
    authorize chemical substitution or replace the preferred ethynyl prefix.
    """
    if mol is None or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1:
        return None
    carbons = tuple(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    if (len(carbons) != 2 or any(a.GetAtomicNum() not in {1, 6}
            or a.GetFormalCharge() or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
            for a in mol.GetAtoms())):
        return None
    bond = mol.GetBondBetweenAtoms(*carbons)
    if bond is None or bond.GetBondType() != Chem.BondType.TRIPLE:
        return None
    if any(mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True) != 1
           or mol.GetAtomWithIdx(i).GetTotalValence() != 4 for i in carbons):
        return None
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings, fully_covered
    from .parent_annotations import isotope_numbering_key
    from .naming_methods import MethodDecision
    candidates = []
    for order in (carbons, carbons[::-1]):
        mapping = {idx: str(loc) for loc, idx in enumerate(order, 1)}
        bindings = hydrogen_bindings(mol, mapping)
        component = MappedComponent('acetylene', tuple(bindings.items()), frozenset(bindings),
                                    systematic_parent=True)
        name = decorate_component(mol, Chem, component)
        rank = isotope_numbering_key(mol, bindings)
        if not name or rank is None or not fully_covered(mol, bindings, ()):
            return None
        candidates.append((rank, name, component))
    best = min(rank for rank, _name, _component in candidates)
    winners = [(name, component) for rank, name, component in candidates if rank == best]
    if len({name for name, _component in winners}) != 1:
        return None
    name, component = winners[0]
    return MethodDecision(name, 'retained-hydrocarbon-parent', frozenset(), component.atoms,
        'P-31.1.2.1/P-52.2.1.1/P-82', component.atom_locants,
        component_roles=(('unsubstituted-retained-parent', component.atoms, component.atom_locants),))
