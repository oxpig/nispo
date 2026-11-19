"""Source P-107.4 glycoglycerol and C18 glycosphingoid construction."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass,field

from .naming_result import NamingResult,SelectionLimitation
from .carbohydrates import CarbohydrateSpec,_SugarInput,_SugarScope
from .carbohydrate_modifications import _ROLE,_OWNER as _SUGAR_OWNER,SugarPhosphateBinding
from .glycosyl_construction import sugar_component,join_alcohol_oxygen
from .glycosides import GlycosidicBinding
from .lipids import (
    GlycerolSpec,LipidAcylBinding,LipidEtherBinding,_LipidInput,_LipidScope,_OWNER as _LIPID_OWNER,
    _build as _glycerol_graph,_sn_tag,_original_acyl_parts,_original_ether_parts,_acyl_prefixes,
)
from .sphingoids import SphingoidSpec,SphingoidAcylBinding,sphingoid_to_iupac_result,_word as _sphingoid_word


_SOURCE_INDEX='_nispo_glycolipid_carrier_index'


@dataclass(frozen=True,slots=True)
class GlycoglycerolSpec:
    carbohydrate: CarbohydrateSpec
    glycerol: GlycerolSpec


@dataclass(frozen=True,slots=True)
class GlycosphingoidSpec:
    carbohydrate: CarbohydrateSpec
    sphingoid: SphingoidSpec


@dataclass(frozen=True,slots=True)
class GlycolipidNamingResult(NamingResult):
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    sugar_atom_maps: tuple[tuple[str,int],...] = ()
    lipid_atom_maps: tuple[tuple[str,int],...] = ()
    component_atom_maps: tuple[tuple[str,tuple[int,...]],...] = ()
    glycosidic_binding: GlycosidicBinding|None = None
    acyl_bindings: tuple[LipidAcylBinding|SphingoidAcylBinding,...] = ()
    ether_bindings: tuple[LipidEtherBinding,...] = field(default=(),kw_only=True)
    phosphate_bindings: tuple[SugarPhosphateBinding,...] = ()
    lipid_kind: str = ''
    lipid_configuration: str = ''
    anomer: str = ''
    declared_anomer: str = ''
    member_smiles: tuple[str,...] = ()
    member_anomers: tuple[str,...] = ()
    proportions_unspecified: bool = False
    substance_form: str = 'conventional-neutral'
    nomenclature: str = 'specialized-P-107.4; PIN-not-assessed-under-P-100'


def _failure(status,reason):
    return GlycolipidNamingResult('',status,'not-assessed',
        (SelectionLimitation('glycolipid-construction','P-107.4.2/P-107.4.3',reason),),'explicit-glycolipid')


def _carrier(request,Chem):
    if isinstance(request,GlycoglycerolSpec):
        spec=request.glycerol
        if (not isinstance(spec,GlycerolSpec) or spec.configuration!='sn' or not isinstance(spec.acyl_groups,tuple)
                or not isinstance(spec.ether_groups,tuple) or not 1<=len(spec.acyl_groups)+len(spec.ether_groups)<=2):
            raise _SugarInput('A glycoglycerol requires one or two O-acyl/O-alkyl groups on an explicitly sn glycerol frame.')
        graph,carbons,oxygens,parts=_glycerol_graph(spec,Chem)
        if any(p==3 for p,*_rest in parts) or any(group.position==3 for group in spec.ether_groups):raise _SugarInput('The source glycerol 3-OH is occupied.')
        roles={**{'C'+str(p):idx for p,idx in carbons.items()},**{'O'+str(p):idx for p,idx in oxygens.items()}}
        refs=tuple((p,c,o) for p,c,o,_word,_owned in parts)
        return graph,oxygens[3],roles,refs,'glycerol'
    if isinstance(request,GlycosphingoidSpec):
        if not isinstance(request.sphingoid,SphingoidSpec):raise _SugarInput('A source C18 sphingoid declaration is required.')
        named=sphingoid_to_iupac_result(request.sphingoid)
        if not named.name:raise _SugarScope('The source sphingoid/acyl construction is unavailable.')
        if any(binding.site=='O1' for binding in named.acyl_bindings):raise _SugarInput('The source sphingoid 1-OH is occupied.')
        graph=Chem.MolFromSmiles(named.mapped_smiles)
        if graph is None:raise _SugarScope('The source mapped sphingoid cannot be recovered.')
        atoms={a.GetAtomMapNum():a for a in graph.GetAtoms()}
        roles={role:atoms[number].GetIdx() for role,number in named.parent_atom_maps}
        for atom in graph.GetAtoms():atom.SetProp(_LIPID_OWNER,'sphingoid')
        refs=[]
        for b in named.acyl_bindings:
            for number in b.atom_maps:atoms[number].SetProp(_LIPID_OWNER,'acyl:'+b.site)
            refs.append((b.site,atoms[b.carbonyl_carbon_map].GetIdx(),atoms[b.sphingoid_heteroatom_map].GetIdx()))
        for atom in graph.GetAtoms():atom.SetAtomMapNum(0)
        return graph,roles['O1'],roles,tuple(refs),'sphingoid'
    raise _SugarInput('Declare a glycoglycerol or glycosphingoid source operation.')


def _lipid_name(request,kind,parts,word,citation_name):
    from .formatting import prefix_alphabetization_key,enclose_name_unit
    key=prefix_alphabetization_key(citation_name)
    if kind=='sphingoid':
        extra=((key,'1-O-'+enclose_name_unit(word)),)
        return _sphingoid_word(parts,request.sphingoid.double_bonds,extra_prefixes=extra)
    # P-107.4.2 prints a bare unmodified glycosyl unit after 3-O. Detachable
    # chemical prefixes on that unit retain an explicit enclosing scope.
    glycosyl=enclose_name_unit(word) if request.carbohydrate.modifications else word
    prefixes=[(key,'3-O-'+glycosyl)];groups=defaultdict(list)
    for part in parts:groups[part[3]].append(part)
    for acyl,rows in groups.items():
        rendered,_comparison=_acyl_prefixes(rows,{1:1,2:2,3:3})
        prefixes.append((prefix_alphabetization_key(acyl),rendered))
    return '-'.join(text for _key,text in sorted(prefixes,key=lambda row:row[0]))+'-sn-glycerol'


def glycolipid_to_iupac_result(request:GlycoglycerolSpec|GlycosphingoidSpec) -> GlycolipidNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        if (not isinstance(request,(GlycoglycerolSpec,GlycosphingoidSpec)) or not isinstance(request.carbohydrate,CarbohydrateSpec)
                or not isinstance(request.carbohydrate.anomer,str)
                or request.carbohydrate.anomer not in {'alpha','beta','unspecified','alpha,beta'}):
            raise _SugarInput('Declare an explicit cyclic glycone/anomer and source lipid parent.')
        carrier,oxygen,lipid_roles,acyl_refs,kind=_carrier(request,Chem)
        for atom in carrier.GetAtoms():atom.SetIntProp(_SOURCE_INDEX,atom.GetIdx())
        interpretations=('alpha','beta') if request.carbohydrate.anomer=='alpha,beta' else (request.carbohydrate.anomer,)
        constructions=[]
        for interpretation in interpretations:
            sugar,word,facts=sugar_component(request.carbohydrate,interpretation,Chem,context='glycosyl')
            graph,indices=join_alcohol_oxygen(sugar,facts['carbonyl'],carrier,oxygen,Chem)
            translated={a.GetIntProp(_SOURCE_INDEX):a.GetIdx() for a in graph.GetAtoms() if a.HasProp(_SOURCE_INDEX)}
            roles={role:translated[idx] for role,idx in lipid_roles.items()}
            for atom in graph.GetAtoms():
                if not atom.HasProp(_LIPID_OWNER):atom.SetProp(_LIPID_OWNER,'glycone')
            if kind=='glycerol':
                atom=graph.GetAtomWithIdx(roles['C2'])
                expected=(roles['C1'],roles['O2'],roles['C3'])
                if tuple(a.GetIdx() for a in atom.GetNeighbors())!=expected:
                    raise _SugarScope('The sn Fischer source port order is not retained.')
                atom.SetChiralTag(_sn_tag())
            for atom in graph.GetAtoms():
                if atom.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED and atom.GetNumImplicitHs():
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs()+atom.GetNumImplicitHs());atom.SetNoImplicit(True)
            Chem.AssignStereochemistry(graph,cleanIt=True,force=True)
            from rdkit.Chem import rdCIPLabeler
            rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
            if kind=='sphingoid' and tuple(graph.GetAtomWithIdx(roles[str(i)]).GetProp('_CIPCode') for i in (2,3))!=('S','R'):
                raise _SugarScope('The source sphinganine physical configuration is not proved.')
            if len(interpretations)>1 and any(info.type in {Chem.StereoType.Atom_Tetrahedral,Chem.StereoType.Bond_Double}
                    and info.specified==Chem.StereoSpecified.Unspecified for info in Chem.FindPotentialStereo(graph)):
                raise _SugarScope('The declared anomer-pair substance has additional unassigned lipid configuration.')
            refs=tuple((p,translated[c],translated[o]) for p,c,o in acyl_refs)
            parts=_original_acyl_parts(graph,Chem,refs)
            ethers=_original_ether_parts(graph,Chem) if kind=='glycerol' and request.glycerol.ether_groups else ()
            name=_lipid_name(request,kind,parts+ethers,word,facts['citation_name'])
            constructions.append((facts['anomer'],graph,name,indices,facts,roles,parts,ethers))
        constructions.sort(key=lambda row:row[0]);anomer,graph,name,indices,facts,roles,parts,ethers=constructions[0]
        members=tuple(Chem.MolToSmiles(row[1],canonical=True,isomericSmiles=True) for row in constructions) if len(constructions)>1 else ()
        if members:
            if len(set(members))!=2:raise _SugarScope('The glycolipid anomer substance has no two distinct members.')
            _sugar,word,_facts=sugar_component(request.carbohydrate,'alpha',Chem,context='glycosyl',substance_anomer='alpha,beta')
            name=_lipid_name(request,kind,parts+ethers,word,_facts['citation_name'])
        identity='' if members else Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
        sugar_roles=[];groups=defaultdict(list)
        for atom in graph.GetAtoms():
            number=atom.GetIdx()+1;atom.SetAtomMapNum(number)
            if atom.HasProp(_SOURCE_INDEX):owner='lipid:'+atom.GetProp(_LIPID_OWNER)
            else:
                owner=atom.GetProp(_SUGAR_OWNER)
                if atom.HasProp(_ROLE):sugar_roles.append((atom.GetProp(_ROLE),number))
            groups[owner].append(number)
        mapped=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True);check=Chem.MolFromSmiles(mapped)
        if check is None:raise _SugarScope('The mapped glycolipid cannot be recovered.')
        for atom in check.GetAtoms():atom.SetAtomMapNum(0)
        if Chem.MolToSmiles(check,canonical=True,isomericSmiles=True)!=(members[0] if members else identity):
            raise _SugarScope('The mapped glycolipid does not conserve physical configuration.')
        sugar_roles=tuple(sorted(sugar_roles));sugar_map=dict(sugar_roles);p_bindings=[]
        for p in facts['phosphate_sites']:
            numbers=tuple(groups[f'sugar:mod:{p}:phosphate']);atoms=[graph.GetAtomWithIdx(i-1) for i in numbers]
            phosphorus=next(a for a in atoms if a.GetAtomicNum()==15)
            acidic=tuple(a.GetIdx()+1 for a in atoms if a.GetAtomicNum()==8 and a.GetDegree()==1 and a.GetTotalNumHs()==1)
            if len(acidic)!=2:raise _SugarScope('The glycosyl phosphate acid H count is not proved.')
            p_bindings.append(SugarPhosphateBinding(p,sugar_map['O'+str(p)],phosphorus.GetIdx()+1,acidic,numbers))
        acyls=tuple((LipidAcylBinding(p,c+1,o+1,tuple(i+1 for i in sorted(owned)),word) if kind=='glycerol' else
                     SphingoidAcylBinding(p,c+1,o+1,tuple(i+1 for i in sorted(owned)),word)) for p,c,o,word,owned in parts)
        ether_bindings=tuple(LipidEtherBinding(p,c+1,o+1,tuple(i+1 for i in sorted(owned)),word) for p,c,o,word,owned in ethers)
        return GlycolipidNamingResult(name=name,generation_status='available',preferred_status='not-assessed',route='explicit-glycolipid',
            canonical_smiles=identity,mapped_smiles=mapped,sugar_atom_maps=sugar_roles,
            lipid_atom_maps=tuple(sorted((role,idx+1) for role,idx in roles.items())),
            component_atom_maps=tuple((key,tuple(value)) for key,value in sorted(groups.items())),
            glycosidic_binding=GlycosidicBinding(*(i+1 for i in indices)),acyl_bindings=acyls,ether_bindings=ether_bindings,phosphate_bindings=tuple(p_bindings),
            lipid_kind=kind,lipid_configuration='sn' if kind=='glycerol' else 'source-C18-(2S,3R)',
            anomer='alpha,beta' if members else anomer,declared_anomer=request.carbohydrate.anomer,member_smiles=members,
            member_anomers=tuple(row[0] for row in constructions) if members else (),proportions_unspecified=bool(members),
            substance_form='anomer-mixture' if members else 'anomer-unspecified' if anomer=='unspecified' else 'conventional-neutral')
    except (_SugarInput,_LipidInput) as error:return _failure('invalid-input',str(error))
    except (_SugarScope,_LipidScope,ValueError,RuntimeError,OverflowError,KeyError) as error:return _failure('unavailable',str(error))


def glycolipid_to_iupac(request:GlycoglycerolSpec|GlycosphingoidSpec) -> str:return glycolipid_to_iupac_result(request).name
