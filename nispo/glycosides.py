"""Explicit source-functional-class O-glycosides with mapped oxygen joins."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .naming_result import NamingResult,SelectionLimitation
from .carbohydrates import CarbohydrateSpec,_SugarInput,_SugarScope
from .carbohydrate_modifications import _ROLE,_OWNER,SugarPhosphateBinding
from .glycosyl_construction import sugar_component,join_alcohol_oxygen


@dataclass(frozen=True,slots=True)
class GlycosideSpec:
    carbohydrate: CarbohydrateSpec
    alcohol_smiles: str


@dataclass(frozen=True,slots=True)
class GlycosidicBinding:
    anomeric_carbon_map: int
    bridge_oxygen_map: int
    aglycone_carbon_map: int
    bridge_oxygen_origin: str = 'alcohol'


@dataclass(frozen=True,slots=True)
class GlycosideNamingResult(NamingResult):
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    sugar_atom_maps: tuple[tuple[str,int],...] = ()
    component_atom_maps: tuple[tuple[str,tuple[int,...]],...] = ()
    glycosidic_binding: GlycosidicBinding|None = None
    phosphate_bindings: tuple[SugarPhosphateBinding,...] = ()
    anomer: str = ''
    declared_anomer: str = ''
    member_smiles: tuple[str,...] = ()
    member_anomers: tuple[str,...] = ()
    proportions_unspecified: bool = False
    substance_form: str = 'conventional-neutral'
    nomenclature: str = 'specialized-P-102; PIN-not-assessed-under-P-100'


def _failure(status,reason):
    return GlycosideNamingResult('',status,'not-assessed',
        (SelectionLimitation('glycoside-construction','P-102.5.6.2/P-102.6.1',reason),),'explicit-glycoside')


def _alcohol(smiles,Chem):
    if not isinstance(smiles,str) or not smiles:raise _SugarInput('An explicit alcohol graph is required.')
    graph=Chem.MolFromSmiles(smiles)
    if graph is None or len(Chem.GetMolFrags(graph))!=1:raise _SugarInput('The aglycone must be one valid connected alcohol.')
    if (graph.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in graph.GetAtoms())
            or any(a.GetAtomicNum() not in {1,6,8,9,17,35,53} for a in graph.GetAtoms())
            or any(a.GetIsotope() and a.GetAtomicNum()!=6 for a in graph.GetAtoms())
            or any(b.GetStereo() in {Chem.BondStereo.STEREOATROPCW,Chem.BondStereo.STEREOATROPCCW} for b in graph.GetBonds())):
        raise _SugarScope('The aglycone needs a separate charge, isotope-role, enhanced/axial or characteristic-class construction.')
    oxygens=[a for a in graph.GetAtoms() if a.GetAtomicNum()==8]
    if len(oxygens)!=1 or oxygens[0].GetDegree()!=1 or oxygens[0].GetTotalNumHs()!=1 or oxygens[0].GetNeighbors()[0].GetAtomicNum()!=6:
        raise _SugarInput('This organyl writer requires one OH and an otherwise carbon/halogen aglycone.')
    for atom in graph.GetAtoms():atom.SetAtomMapNum(0)
    return graph,oxygens[0].GetIdx()


def glycoside_to_iupac_result(spec:GlycosideSpec) -> GlycosideNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        if (not isinstance(spec,GlycosideSpec) or not isinstance(spec.carbohydrate,CarbohydrateSpec)
                or not isinstance(spec.carbohydrate.anomer,str)
                or spec.carbohydrate.anomer not in {'alpha','beta','unspecified','alpha,beta'}):
            raise _SugarInput('Declare a cyclic sugar/anomer and its explicit alcohol component.')
        alcohol,oxygen=_alcohol(spec.alcohol_smiles,Chem)
        if spec.carbohydrate.anomer=='alpha,beta' and any(info.type in {Chem.StereoType.Atom_Tetrahedral,Chem.StereoType.Bond_Double}
                                                        and info.specified==Chem.StereoSpecified.Unspecified
                                                        for info in Chem.FindPotentialStereo(alcohol)):
            raise _SugarScope('The anomer-pair substance requires assigned aglycone stereocentres.')
        interpretations=('alpha','beta') if spec.carbohydrate.anomer=='alpha,beta' else (spec.carbohydrate.anomer,)
        constructions=[]
        for interpretation in interpretations:
            sugar,word,facts=sugar_component(spec.carbohydrate,interpretation,Chem,context='glycoside')
            graph,indices=join_alcohol_oxygen(sugar,facts['carbonyl'],alcohol,oxygen,Chem)
            from .component_mapping import mapped_branch
            organyl,owned=mapped_branch(graph,Chem,indices[2],indices[1])
            expected=frozenset(a.GetIdx() for a in graph.GetAtoms() if a.GetProp(_OWNER)=='aglycone' and a.GetIdx()!=indices[1])
            if not organyl or owned!=expected:
                raise _SugarScope('The complete original-graph organyl has no proved mapped component construction.')
            constructions.append((facts['anomer'],graph,organyl+' '+word,indices,facts))
        constructions.sort(key=lambda row:row[0]);anomer,graph,name,indices,facts=constructions[0]
        members=tuple(Chem.MolToSmiles(item[1],canonical=True,isomericSmiles=True) for item in constructions) if len(constructions)>1 else ()
        if members:
            if len(set(members))!=2:raise _SugarScope('The glycoside mixture has no two distinct anomers.')
            # Re-render the typed component in its mixture interpretation.
            _sugar,mixture_word,_facts=sugar_component(spec.carbohydrate,'alpha',Chem,context='glycoside',substance_anomer='alpha,beta')
            organyl,_owned=mapped_branch(graph,Chem,indices[2],indices[1]);name=organyl+' '+mixture_word
        identity='' if members else Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
        roles=[];groups=defaultdict(list)
        for atom in graph.GetAtoms():
            number=atom.GetIdx()+1;atom.SetAtomMapNum(number);owner=atom.GetProp(_OWNER);groups[owner].append(number)
            if owner.startswith('sugar:') and atom.HasProp(_ROLE):roles.append((atom.GetProp(_ROLE),number))
        mapped=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True);check=Chem.MolFromSmiles(mapped)
        if check is None:raise _SugarScope('The mapped glycoside graph cannot be recovered.')
        for atom in check.GetAtoms():atom.SetAtomMapNum(0)
        if Chem.MolToSmiles(check,canonical=True,isomericSmiles=True)!=(members[0] if members else identity):
            raise _SugarScope('Mapped glycoside serialization does not conserve configuration.')
        roles=tuple(sorted(roles));role_map=dict(roles);p_bindings=[]
        for p in facts['phosphate_sites']:
            numbers=tuple(groups[f'sugar:mod:{p}:phosphate']);atoms=[graph.GetAtomWithIdx(i-1) for i in numbers]
            phosphorus=next(a for a in atoms if a.GetAtomicNum()==15)
            acidic=tuple(a.GetIdx()+1 for a in atoms if a.GetAtomicNum()==8 and a.GetDegree()==1 and a.GetTotalNumHs()==1)
            if len(acidic)!=2:raise _SugarScope('The glycosyl phosphate acid hydrogen count is unproved.')
            p_bindings.append(SugarPhosphateBinding(p,role_map['O'+str(p)],phosphorus.GetIdx()+1,acidic,numbers))
        return GlycosideNamingResult(name=name,generation_status='available',preferred_status='not-assessed',route='explicit-glycoside',
            canonical_smiles=identity,mapped_smiles=mapped,sugar_atom_maps=roles,
            component_atom_maps=tuple((key,tuple(value)) for key,value in sorted(groups.items())),
            glycosidic_binding=GlycosidicBinding(*(i+1 for i in indices)),phosphate_bindings=tuple(p_bindings),
            anomer='alpha,beta' if members else anomer,declared_anomer=spec.carbohydrate.anomer,member_smiles=members,
            member_anomers=tuple(item[0] for item in constructions) if members else (),proportions_unspecified=bool(members),
            substance_form='anomer-mixture' if members else 'anomer-unspecified' if anomer=='unspecified' else 'conventional-neutral')
    except _SugarInput as error:return _failure('invalid-input',str(error))
    except (_SugarScope,ValueError,RuntimeError,OverflowError,KeyError) as error:return _failure('unavailable',str(error))


def glycoside_to_iupac(spec:GlycosideSpec) -> str:return glycoside_to_iupac_result(spec).name
