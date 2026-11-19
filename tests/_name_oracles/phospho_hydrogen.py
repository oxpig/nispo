"""Name-only proof of the neutral diphosphooxy prefix in a monocation.

OPSIN 2.9 can deprotonate this acid prefix when balancing the explicitly
named cation. The prefix denotes -O-P(=O)(OH)-O-P(=O)(OH)2. This supplement
owns only that unique original parser subgraph and its three OH hydrogens;
it neither receives an input structure nor changes the cation or any stereo.
"""
from collections import Counter
import re
from rdkit import Chem


def recover_neutral_diphosphooxy(name, parse_raw):
    if (not isinstance(name,str) or name.count('diphosphooxy')!=1
            or name.count('ium')!=1
            or 'phosph' in name.replace('diphosphooxy','')
            or re.search(r'ide|ate|\([0-9]+[-−]\)',name)):
        return None
    parsed=parse_raw(name)
    if not parsed:return None
    mol=Chem.MolFromSmiles(parsed) if isinstance(parsed,str) else Chem.Mol(parsed)
    if (mol is None or len(Chem.GetMolFrags(mol))!=1 or mol.GetStereoGroups()
            or any(a.GetIsotope() or a.GetNumRadicalElectrons() or a.HasQuery()
                   for a in mol.GetAtoms())):return None
    phosphorus=[a for a in mol.GetAtoms() if a.GetAtomicNum()==15]
    if len(phosphorus)!=2:return None
    ps={a.GetIdx() for a in phosphorus};terminal=[];bridges=set();links=set()
    for p in phosphorus:
        if p.GetFormalCharge() or p.GetDegree()!=4 or p.GetTotalValence()!=5:return None
        oxo=0
        for bond in p.GetBonds():
            o=bond.GetOtherAtom(p);idx=o.GetIdx()
            if o.GetAtomicNum()!=8:return None
            if bond.GetBondTypeAsDouble()==2:
                if o.GetDegree()!=1 or o.GetFormalCharge():return None
                oxo+=1;continue
            if bond.GetBondTypeAsDouble()!=1:return None
            if o.GetDegree()==1:
                if (o.GetFormalCharge(),o.GetTotalNumHs()) not in {(0,1),(-1,0)}:return None
                terminal.append(idx)
            elif o.GetDegree()==2 and not o.GetFormalCharge() and not o.GetTotalNumHs():
                neighbors={a.GetIdx() for a in o.GetNeighbors()}
                if neighbors==ps:bridges.add(idx)
                elif sum(a.GetAtomicNum()==6 for a in o.GetNeighbors())==1:links.add(idx)
                else:return None
            else:return None
        if oxo!=1:return None
    positive=[a for a in mol.GetAtoms() if a.GetFormalCharge()>0]
    negative=[a for a in mol.GetAtoms() if a.GetFormalCharge()<0]
    if (len(bridges)!=1 or len(links)!=1 or len(terminal)!=3
            or len(positive)!=1 or positive[0].GetFormalCharge()!=1
            or positive[0].GetAtomicNum() not in {7,8}
            or len(negative)!=1 or negative[0].GetIdx() not in terminal):return None
    before=Counter(a.GetAtomicNum() for a in mol.GetAtoms())
    h_before=sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    atom=negative[0];atom.SetFormalCharge(0);atom.SetNumExplicitHs(1);atom.SetNoImplicit(True)
    try:Chem.SanitizeMol(mol)
    except (ValueError,RuntimeError):return None
    if (Counter(a.GetAtomicNum() for a in mol.GetAtoms())!=before
            or sum(a.GetTotalNumHs() for a in mol.GetAtoms())!=h_before+1
            or sum(a.GetFormalCharge() for a in mol.GetAtoms())!=1):return None
    return Chem.MolToSmiles(mol,isomericSmiles=True)
