"""P-93.5.1.1 whole-graph classification of redundant native stereo flags."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True,slots=True)
class CyclicPolyolNonstereogenicEvidence:
    parent_atoms: frozenset[int]
    nonstereogenic_atoms: frozenset[int]
    assigned: tuple[tuple[int,str],...]
    graph_signature: tuple
    rule: str = 'P-93.5.1.1/P-92'


def _signature(mol):
    return (tuple((a.GetAtomicNum(),a.GetIsotope(),a.GetFormalCharge(),a.GetNumRadicalElectrons(),
        int(a.GetChiralTag()),a.GetNumExplicitHs(),a.GetNoImplicit(),tuple(n.GetIdx() for n in a.GetNeighbors())) for a in mol.GetAtoms()),
        tuple((b.GetBeginAtomIdx(),b.GetEndAtomIdx(),str(b.GetBondType()),int(b.GetStereo()),tuple(b.GetStereoAtoms())) for b in mol.GetBonds()))


def cyclic_polyol_nonstereogenic_evidence(mol,Chem):
    """A successful complete modern label operation can classify native tags.

    Native tetrahedral flags encode relative ring arrangements even on atoms
    that are nonstereogenic in this particular fully specified isomer. P-93
    omits descriptors on those atoms. No failed/missing-label exception is
    interpreted as such a proof, and partial or advanced stereo stays outside.
    """
    from .extended_inventory import selection_inventory
    from .method_ineligibility import ordinary_multiplication_ineligibility
    from rdkit.Chem import rdCIPLabeler
    inventory=selection_inventory(mol)
    proof=ordinary_multiplication_ineligibility(mol,Chem,inventory.principal)
    if proof is None or not proof.reason.startswith('a-single-carbon-cycle-with-only-terminal-OH') or mol.GetStereoGroups():return None
    if any(a.GetChiralTag() not in {Chem.ChiralType.CHI_UNSPECIFIED,Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW} for a in mol.GetAtoms()) or any(int(b.GetStereo()) for b in mol.GetBonds()):return None
    potential=Chem.FindPotentialStereo(Chem.Mol(mol))
    if any(s.type!=Chem.StereoType.Atom_Tetrahedral or s.specified!=Chem.StereoSpecified.Specified for s in potential):return None
    graph=Chem.Mol(mol)
    for obj in list(graph.GetAtoms())+list(graph.GetBonds()):
        for prop in ('_CIPCode','_CIPRank'):
            if obj.HasProp(prop):obj.ClearProp(prop)
    try:rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
    except (ValueError,RuntimeError,OverflowError):return None
    assigned=tuple((a.GetIdx(),a.GetProp('_CIPCode')) for a in graph.GetAtoms()
        if a.HasProp('_CIPCode') and a.GetProp('_CIPCode') in {'R','S','r','s'})
    if not assigned:return None
    identified={i for i,_code in assigned}
    redundant=frozenset(a.GetIdx() for a in graph.GetAtoms() if int(a.GetChiralTag()) and a.GetIdx() not in identified)
    if not redundant or not redundant<=proof.cyclic_parent_atoms:return None
    return CyclicPolyolNonstereogenicEvidence(proof.cyclic_parent_atoms,redundant,assigned,_signature(mol))


def validate_cyclic_polyol_nonstereogenic_evidence(mol,evidence,parent_atoms,assigned,missing):
    return (isinstance(evidence,CyclicPolyolNonstereogenicEvidence)
        and evidence.rule=='P-93.5.1.1/P-92' and evidence.graph_signature==_signature(mol)
        and evidence.parent_atoms==parent_atoms and evidence.nonstereogenic_atoms==frozenset(missing)
        and dict(evidence.assigned)==assigned and bool(evidence.nonstereogenic_atoms))
