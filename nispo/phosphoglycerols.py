"""Explicit sn-glycerol phosphate and neutral polar-head condensation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .naming_result import SelectionLimitation
from .lipids import (
    GlycerolAcyl,GlycerolEther,GlycerolSpec,GlycerolNamingResult,LipidAcylBinding,LipidEtherBinding,_OWNER,
    _LipidInput,_LipidScope,_build,_sn_tag,_acyl_prefixes,_original_acyl_parts,_original_ether_parts,
)
from .nucleotides import _phosphate_chain
from .nucleosides import _NucleosideInput,_NucleosideScope


@dataclass(frozen=True,slots=True)
class PhosphoglycerolSpec:
    acyl_groups: tuple[GlycerolAcyl,...] = ()
    phosphate_position: int = 3
    head_group: str = 'none'
    configuration: str = 'sn'
    ether_groups: tuple[GlycerolEther,...] = ()


@dataclass(frozen=True,slots=True)
class LipidPhosphateBinding:
    position: int
    glycerol_oxygen_map: int
    phosphorus_map: int
    acidic_atom_maps: tuple[int,...]
    head_oxygen_map: int|None
    atom_maps: tuple[int,...]
    head_atom_maps: tuple[int,...]
    phosphorus_configuration: str = 'unspecified'


@dataclass(frozen=True,slots=True)
class PhosphoglycerolNamingResult(GlycerolNamingResult):
    phosphate_binding: LipidPhosphateBinding|None = None
    head_group: str = ''
    acid_hydrogen_count: int = 0


def _failure(status,reason):
    return PhosphoglycerolNamingResult('',status,'not-assessed',
        (SelectionLimitation('phosphoglycerol-construction','P-107.3',reason),),'explicit-phosphoglycerol')


def _head(smiles,Chem):
    graph=Chem.MolFromSmiles(smiles)
    if graph is None:raise _LipidScope('The source neutral head graph is unavailable.')
    sites=[a.GetIdx() for a in graph.GetAtoms() if a.GetAtomicNum()==8 and a.GetDegree()==1
           and a.GetTotalNumHs()==1 and a.GetNeighbors()[0].GetHybridization()==Chem.HybridizationType.SP3]
    if len(sites)!=1:raise _LipidScope('The head has no unique alcoholic OH.')
    return graph,sites[0]


def _head_prefix(parts):
    from .formatting import prefix_alphabetization_key,enclose_name_unit
    grouped=defaultdict(list)
    for position,_c,_o,word,_atoms in parts:grouped[word].append(position)
    words=[]
    for word,positions in sorted(grouped.items(),key=lambda item:prefix_alphabetization_key(item[0])):
        locants=','.join(str(p) for p in sorted(positions))
        enclosed=enclose_name_unit(word) if '-' in word or word[:1]=='(' or len(positions)>1 else word
        word=('di' if len(positions)==2 else '')+enclosed
        words.append(locants+'-'+word)
    return '-'.join(words)


def phosphoglycerol_to_iupac_result(request:PhosphoglycerolSpec) -> PhosphoglycerolNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        if (not isinstance(request,PhosphoglycerolSpec) or type(request.phosphate_position) is not int
                or request.phosphate_position not in {1,2,3} or not isinstance(request.head_group,str)
                or request.head_group not in {'none','2-aminoethanol','L-serine'}
                or request.configuration!='sn' or not isinstance(request.acyl_groups,tuple)
                or not isinstance(request.ether_groups,tuple) or len(request.acyl_groups)+len(request.ether_groups)>2):
            raise _LipidInput('Declare one sn-glycerol phosphate site, up to two O-acyl/O-alkyl groups, and a neutral source head.')
        if any(isinstance(group,(GlycerolAcyl,GlycerolEther)) and group.position==request.phosphate_position for group in request.acyl_groups+request.ether_groups):
            raise _LipidInput('The phosphate site is already occupied by an O-substituent.')
        # The final phosphate/head graph determines configuration and acyl
        # decoration; the initial acyl scaffold supplies atom ownership only.
        base=GlycerolSpec(request.acyl_groups,'unspecified',request.ether_groups)
        graph,carbon,oxygen,parts=_build(base,Chem,allow_unacylated=True)
        editable=Chem.RWMol(graph);anchor=oxygen[request.phosphate_position]
        backbone,acidic=_phosphate_chain(editable,Chem,anchor,1,'oxygen','phosphate')
        phosphorus=backbone[0];head_oxygen=None;head_maps=()
        # The phosphate helper uses its own owner property. Give these atoms
        # the common lipid owner before a possible head-group condensation.
        for atom in editable.GetAtoms():
            if not atom.HasProp(_OWNER):atom.SetProp(_OWNER,'phosphate')
        if request.head_group!='none':
            piece,site=_head('NCCO' if request.head_group=='2-aminoethanol' else 'N[C@@H](CO)C(=O)O',Chem)
            if acidic[-1]!=editable.GetNumAtoms()-1:raise _LipidScope('The phosphate leaving oxygen is not safely mapped.')
            editable.RemoveAtom(acidic[-1]);acidic=acidic[:-1];offset=editable.GetNumAtoms()
            for atom in piece.GetAtoms():atom.SetProp(_OWNER,'head')
            editable.InsertMol(piece);head_oxygen=offset+site
            atom=editable.GetAtomWithIdx(head_oxygen)
            if atom.GetNumExplicitHs():atom.SetNumExplicitHs(atom.GetNumExplicitHs()-1)
            editable.AddBond(phosphorus,head_oxygen,Chem.BondType.SINGLE);editable.UpdatePropertyCache(strict=False)
            head_maps=tuple(i+1 for i in range(offset,editable.GetNumAtoms()))
        graph=editable.GetMol();graph.GetAtomWithIdx(2).SetChiralTag(_sn_tag())
        Chem.SanitizeMol(graph);Chem.AssignStereochemistry(graph,cleanIt=True,force=True)
        from rdkit.Chem import rdCIPLabeler
        rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
        if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in graph.GetAtoms()):
            raise _LipidScope('The explicitly conventional-neutral head/phosphate form was not retained.')
        acid_h=sum(graph.GetAtomWithIdx(i).GetTotalNumHs() for i in acidic)
        if acid_h!=(2 if request.head_group=='none' else 1):raise _LipidScope('The phosphate acid H count is not proved.')
        if request.head_group=='L-serine':
            centers=[a for a in graph.GetAtoms() if a.GetProp(_OWNER)=='head' and a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED]
            if len(centers)!=1 or centers[0].GetProp('_CIPCode')!='S':
                raise _LipidScope('The L-serine head configuration was not retained.')
        parts=_original_acyl_parts(graph,Chem,[(p,c,o) for p,c,o,_word,_atoms in parts])
        ethers=_original_ether_parts(graph,Chem) if request.ether_groups else ()
        if request.head_group=='none':
            prefix,_key=_acyl_prefixes(parts+ethers,{1:1,2:2,3:3})
            name=(prefix+'-' if prefix else '')+'sn-glycerol '+str(request.phosphate_position)+'-phosphate'
        else:
            prefix=_head_prefix(parts+ethers)
            name=(prefix+'-' if prefix else '')+'sn-glycero-'+str(request.phosphate_position)+'-phospho'+('ethanolamine' if request.head_group=='2-aminoethanol' else '-L-serine')
        identity=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
        phosphate_maps=tuple(a.GetIdx()+1 for a in graph.GetAtoms() if a.GetProp(_OWNER)=='phosphate')
        for atom in graph.GetAtoms():atom.SetAtomMapNum(atom.GetIdx()+1)
        binding=LipidPhosphateBinding(request.phosphate_position,anchor+1,phosphorus+1,tuple(i+1 for i in acidic),
            None if head_oxygen is None else head_oxygen+1,phosphate_maps,head_maps)
        acyls=tuple(LipidAcylBinding(p,c+1,o+1,tuple(i+1 for i in sorted(atoms)),word) for p,c,o,word,atoms in parts)
        ether_bindings=tuple(LipidEtherBinding(p,c+1,o+1,tuple(i+1 for i in sorted(atoms)),word) for p,c,o,word,atoms in ethers)
        roles=tuple(sorted([(f'C{p}',i+1) for p,i in carbon.items()]+[(f'O{p}',i+1) for p,i in oxygen.items()]))
        return PhosphoglycerolNamingResult(name=name,generation_status='available',preferred_status='not-assessed',
            route='explicit-phosphoglycerol',canonical_smiles=identity,mapped_smiles=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True),
            glycerol_atom_maps=roles,acyl_bindings=acyls,ether_bindings=ether_bindings,configuration='sn',phosphate_binding=binding,
            head_group=request.head_group,acid_hydrogen_count=acid_h)
    except (_LipidInput,_NucleosideInput) as error:return _failure('invalid-input',str(error))
    except (_LipidScope,_NucleosideScope,ValueError,RuntimeError,OverflowError) as error:return _failure('unavailable',str(error))


def phosphoglycerol_to_iupac(request:PhosphoglycerolSpec) -> str:return phosphoglycerol_to_iupac_result(request).name
