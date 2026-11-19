"""Mapped mixed relative sets and explicitly declared P-93.1.4 epimers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .naming_result import SelectionLimitation
from .stereo_ensembles import StereoEnsembleNamingResult,_leading_configuration


@dataclass(frozen=True,slots=True)
class StereoSubgroup:
    representative_smiles: str
    interpretation: Literal['relative-group','ambo']
    atom_maps: tuple[int,...]


@dataclass(frozen=True,slots=True)
class StereoSubgroupNamingResult(StereoEnsembleNamingResult):
    group_locants: tuple[str,...] = ()
    representative_atom_maps: tuple[int,...] = ()
    proportions: str = 'unspecified'


def _failure(status,reason):
    return StereoSubgroupNamingResult('',status,'not-assessed' if status=='invalid-input' else 'unresolved',
        (SelectionLimitation('stereo-subgroup','P-93.1.2.2/P-93.1.4',reason),),'explicit-stereo-subgroup')


def stereo_subgroup_to_iupac_result(declaration:StereoSubgroup) -> StereoSubgroupNamingResult:
    """Name one relative subgroup or one explicitly declared epimer mixture.

    The selected molecular route must provide the original atom map for one
    complete leading descriptor group. Unknown/embedded mappings, coupled CIP
    changes and pseudoasymmetric centres remain unavailable. Inverting the
    declared relative set describes either member; ambo declares both with
    no imposed proportion. No mixture is inferred from a lone input SMILES.
    """
    from .api import Chem,_engine
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    if (not isinstance(declaration,StereoSubgroup)
            or declaration.interpretation not in {'relative-group','ambo'}
            or not isinstance(declaration.representative_smiles,str)
            or not isinstance(declaration.atom_maps,tuple)
            or not declaration.atom_maps
            or any(type(i) is not int or i<=0 for i in declaration.atom_maps)
            or len(set(declaration.atom_maps))!=len(declaration.atom_maps)):
        return _failure('invalid-input','A declared interpretation and distinct positive mapped stereocentres are required.')
    try:mol=Chem.MolFromSmiles(declaration.representative_smiles)
    except (ValueError,TypeError):mol=None
    if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol))!=1:
        return _failure('invalid-input','A connected representative is required.')
    if mol.GetStereoGroups():return _failure('unavailable','Existing enhanced stereo groups require a combined group operation.')
    from .input_stereochemistry import _source_has_relative_flag
    if _source_has_relative_flag(mol):
        return _failure('unavailable',
            'A whole-molecule relative flag does not establish the absolute centres outside the declared subgroup.')
    mapped={};duplicates=set()
    for atom in mol.GetAtoms():
        number=atom.GetAtomMapNum()
        if number:
            if number in mapped:duplicates.add(number)
            mapped[number]=atom.GetIdx()
    if duplicates or not set(declaration.atom_maps)<=mapped.keys():
        return _failure('invalid-input','Group maps must identify unique existing atoms.')
    tetra={Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    for feature in Chem.FindPotentialStereo(mol):
        if str(feature.type) not in {'Atom_Tetrahedral','Bond_Double'}:
            return _failure('unavailable','A nonordinary stereogenic unit needs another operation.')
        if str(feature.specified)!='Specified':return _failure('invalid-input','Every potential stereogenic unit must be assigned.')
    if any(a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED and a.GetChiralTag() not in tetra for a in mol.GetAtoms()):
        return _failure('unavailable','Only ordinary tetrahedral group inversion is established.')
    group=frozenset(mapped[number] for number in declaration.atom_maps)
    from .stereochemistry import assigned_atom_stereochemistry,_indexed_bond_stereochemistry
    for atom in mol.GetAtoms():atom.SetAtomMapNum(0)
    first=dict(assigned_atom_stereochemistry(mol,Chem,include_pseudo=True))
    if not group<first.keys():return _failure('invalid-input','The group must contain assigned centres and leave an absolute centre outside it.')
    if any(code not in {'R','S'} for code in first.values()):
        return _failure('unavailable','Pseudoasymmetric group interpretation needs a broader sequence operation.')
    if declaration.interpretation=='ambo' and len(group)!=1:
        return _failure('unavailable','This operation declares one epimeric centre at a time.')
    if declaration.interpretation=='relative-group' and len(group)<2:
        return _failure('invalid-input','A relative set needs at least two related centres.')
    other=Chem.Mol(mol)
    for idx in group:other.GetAtomWithIdx(idx).InvertChirality()
    Chem.AssignStereochemistry(other,cleanIt=True,force=True)
    second=dict(assigned_atom_stereochemistry(other,Chem,include_pseudo=True))
    reverse={'R':'S','S':'R'}
    if set(first)!=set(second) or any(second[idx]!=(reverse[code] if idx in group else code) for idx,code in first.items()):
        return _failure('unavailable','Group inversion changes coupled or fixed original-graph sequence assignments.')
    if _indexed_bond_stereochemistry(mol,Chem)!=_indexed_bond_stereochemistry(other,Chem):
        return _failure('unavailable','Group inversion changes a fixed double-bond assignment.')
    identities=tuple(Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True) for graph in (mol,other))
    if identities[0]==identities[1]:return _failure('invalid-input','The declaration does not create two distinct configurations.')
    engine=_engine();named=tuple(engine.name_mapped_result(Chem.Mol(graph)) for graph in (mol,other))
    configurations=[];maps=[]
    for (result,raw_mapping),assigned in zip(named,(first,second),strict=True):
        if not result.name:return _failure('unavailable','A member has no complete molecular naming operation.')
        configuration=_leading_configuration(result.name,len(assigned))
        mapping=dict(raw_mapping)
        if configuration is None or any(idx not in mapping for idx in assigned):
            return _failure('unavailable','A complete leading descriptor-to-original-atom map is not supplied by the selected route.')
        descriptors={loc:code for loc,code in configuration[1] if code in {'R','S','r','s'}}
        if (len({mapping[idx] for idx in assigned})!=len(assigned)
                or any(descriptors.get(mapping[idx])!=code for idx,code in assigned.items())):
            return _failure('unavailable','Committed atom locants do not prove every delivered stereodescriptor.')
        configurations.append(configuration);maps.append(mapping)
    if (configurations[0][0]!=configurations[1][0]
            or any(maps[0][idx]!=maps[1][idx] for idx in first)
            or tuple(loc for loc,_code in configurations[0][1])!=tuple(loc for loc,_code in configurations[1][1])):
        return _failure('unavailable','The members do not share a proved constitutional name and original-atom numbering.')
    from .parent_selection import locant_key
    locants=tuple(sorted((maps[0][idx] for idx in group),key=locant_key))
    low=next(idx for idx in group if maps[0][idx]==locants[0])
    chosen=0 if first[low]=='R' else 1;alternate=1-chosen
    base,pieces,_representative=configurations[chosen]
    if declaration.interpretation=='relative-group':
        pieces=tuple((loc,code+'*' if loc in locants else code) for loc,code in pieces)
        prefix='';membership='or-relative-diastereomer'
    else:
        prefix=locants[0]+'-ambo-';membership='and-epimer'
    name=prefix+'('+','.join(loc+code for loc,code in pieces)+')-'+base
    issues=[]
    for result,_mapping in named:
        for issue in result.unresolved:
            if issue not in issues:issues.append(issue)
    status=('unresolved' if issues else 'resolved' if all(result.preferred_status=='resolved' for result,_mapping in named) else 'not-assessed')
    return StereoSubgroupNamingResult(
        name=name, generation_status='available', preferred_status=status,
        unresolved=tuple(issues), route='explicit-stereo-subgroup',
        interpretation=declaration.interpretation,
        member_smiles=(identities[chosen],identities[alternate]), mole_fractions=(),
        membership=membership, group_locants=locants,
        representative_atom_maps=tuple(declaration.atom_maps), proportions='unspecified')


def stereo_subgroup_to_iupac(declaration:StereoSubgroup) -> str:
    return stereo_subgroup_to_iupac_result(declaration).name
