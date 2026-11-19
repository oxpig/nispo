"""Productive P-102 Fischer parents and retained-family cyclic anomers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

from .naming_result import NamingResult,SelectionLimitation


# Configurational vocabulary from Tables 10.2/10.3, not cyclic graph templates.
_ALDOSE={
    'R':'glyceraldehyde','RR':'erythrose','LR':'threose',
    'RRR':'ribose','LRR':'arabinose','RLR':'xylose','LLR':'lyxose',
    'RRRR':'allose','LRRR':'altrose','RLRR':'glucose','LLRR':'mannose',
    'RRLR':'gulose','LRLR':'idose','RLLR':'galactose','LLLR':'talose',
}
_KETOSE={'R':'erythrulose','RR':'ribulose','LR':'xylulose',
    'RRR':'psicose','LRR':'fructose','RLR':'sorbose','LLR':'tagatose'}
_CONFIG_PREFIX={'R':'glycero','RR':'erythro','LR':'threo',
    'RRR':'ribo','LRR':'arabino','RLR':'xylo','LLR':'lyxo',
    'RRRR':'allo','LRRR':'altro','RLRR':'gluco','LLRR':'manno',
    'RRLR':'gulo','LRLR':'ido','RLLR':'galacto','LLLR':'talo'}
_RETAINED={name:(len(sides)+2,1,sides) for sides,name in _ALDOSE.items()}
_RETAINED.update({name:(len(sides)+3,2,sides) for sides,name in _KETOSE.items()})
_OPPOSITE={'R':'L','L':'R'}


@dataclass(frozen=True,slots=True)
class FischerSugar:
    carbon_count: int
    hydroxy_sides: str
    carbonyl_position: int = 1


@dataclass(frozen=True,slots=True)
class CarbohydrateModification:
    position: int
    kind: str


@dataclass(frozen=True,slots=True)
class CarbohydrateSpec:
    parent: str|FischerSugar
    ring: str = 'pyranose'
    anomer: str = 'unspecified'
    modifications: tuple[CarbohydrateModification,...] = ()


@dataclass(frozen=True,slots=True)
class CarbohydrateNamingResult(NamingResult):
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    parent_atom_maps: tuple[tuple[str,int],...] = ()
    carbon_count: int = 0
    carbonyl_position: int = 0
    fischer_hydroxy_sides: str = ''
    ring: str = ''
    ring_closing_position: int|None = None
    ring_oxygen_map: int|None = None
    anomer: str = ''
    member_smiles: tuple[str,...] = ()
    member_anomers: tuple[str,...] = ()
    proportions_unspecified: bool = False
    substance_form: str = 'conventional-neutral'
    nomenclature: str = 'specialized-P-102; PIN-not-assessed-under-P-100'


class _SugarInput(ValueError):pass
class _SugarScope(ValueError):pass


def _failure(status,reason):
    return CarbohydrateNamingResult('',status,'not-assessed',
        (SelectionLimitation('carbohydrate-construction','P-102.3/P-102.5',reason),),'explicit-carbohydrate')


def _mirror(sides):return ''.join(_OPPOSITE[side] for side in sides)


def _carbohydrate_root(count):
    from .constants import ROOTS
    return {3:'tri',4:'tetr'}.get(count,ROOTS[count])


def _parent_facts(parent):
    if isinstance(parent,str):
        match=re.fullmatch(r'([dDlL])-([a-zA-Z]+)',parent)
        if match is None or match[2].lower() not in _RETAINED:
            raise _SugarInput('A retained parent must explicitly give its D/L series, for example D-glucose.')
        count,carbonyl,sides=_RETAINED[match[2].lower()]
        if match[1].upper()=='L':sides=_mirror(sides)
        return count,carbonyl,sides
    if (not isinstance(parent,FischerSugar) or type(parent.carbon_count) is not int
            or type(parent.carbonyl_position) is not int or parent.carbonyl_position not in {1,2}
            or not (3 if parent.carbonyl_position==1 else 4)<=parent.carbon_count<=16
            or not isinstance(parent.hydroxy_sides,str)
            or len(parent.hydroxy_sides)!=parent.carbon_count-(2 if parent.carbonyl_position==1 else 3)
            or re.fullmatch('[RL]+',parent.hydroxy_sides) is None):
        raise _SugarInput('Declare a 3..16-C aldose or 4..16-C 2-ketose and every Fischer OH side in carbonyl-to-terminal order.')
    return parent.carbon_count,parent.carbonyl_position,parent.hydroxy_sides


def _parent_word(count,carbonyl,sides,ring):
    normalized=sides if sides[-1]=='R' else _mirror(sides);series='D' if sides[-1]=='R' else 'L'
    retained=(_ALDOSE if carbonyl==1 else _KETOSE).get(normalized)
    if retained:
        if ring=='acyclic':return series+'-'+retained
        return series+'-'+retained[:-3]+'o'+ring
    if ring!='acyclic':
        raise _SugarScope('The systematic multi-prefix cyclic reference-centre convention is not yet proved.')
    groups=[sides[i:i+4] for i in range(0,len(sides),4)];prefixes=[]
    for group in reversed(groups):
        normalized=group if group[-1]=='R' else _mirror(group)
        prefixes.append(('D' if group[-1]=='R' else 'L')+'-'+_CONFIG_PREFIX[normalized])
    noun=_carbohydrate_root(count)+('ose' if carbonyl==1 else '-2-ulose')
    return '-'.join(prefixes)+'-'+noun


@lru_cache(maxsize=24)
def _projection_tag(ports):
    """Native ordered-neighbour parity of the source Fischer projection."""
    from rdkit import Chem
    points={'top':(0,1,-1),'bottom':(0,-1,-1),'right':(1,0,1),'left':(-1,0,1)}
    probe=Chem.MolFromSmiles('C(F)(Cl)(Br)I');conf=Chem.Conformer(5);conf.Set3D(True);conf.SetAtomPosition(0,(0,0,0))
    for idx,key in enumerate(ports,1):conf.SetAtomPosition(idx,points[key])
    probe.AddConformer(conf);Chem.AssignAtomChiralTagsFromStructure(probe,replaceExistingTags=True)
    tag=probe.GetAtomWithIdx(0).GetChiralTag()
    if tag==Chem.ChiralType.CHI_UNSPECIFIED:raise _SugarScope('The source Fischer port parity is unavailable.')
    return tag


def _reordered_tag(tag,reference,actual,Chem):
    if len(reference)!=4 or set(reference)!=set(actual) or len(set(reference))!=4:
        raise _SugarScope('The anomeric port correspondence is incomplete.')
    positions=[reference.index(atom) for atom in actual]
    odd=sum(a>b for i,a in enumerate(positions) for b in positions[i+1:])%2
    if odd:return Chem.ChiralType.CHI_TETRAHEDRAL_CCW if tag==Chem.ChiralType.CHI_TETRAHEDRAL_CW else Chem.ChiralType.CHI_TETRAHEDRAL_CW
    return tag


def _build_sugar(count,carbonyl,sides,ring,anomer,Chem):
    editable=Chem.RWMol()
    for _ in range(count):editable.AddAtom(Chem.Atom(6))
    for i in range(count-1):editable.AddBond(i,i+1,Chem.BondType.SINGLE)
    for i in range(count):
        oxygen=editable.AddAtom(Chem.Atom(8))
        editable.AddBond(i,oxygen,Chem.BondType.DOUBLE if i==carbonyl-1 else Chem.BondType.SINGLE)
    graph=editable.GetMol();Chem.SanitizeMol(graph)
    chiral_positions=[i for i in range(2,count) if i!=carbonyl]
    for position,side in zip(chiral_positions,sides,strict=True):
        atom=graph.GetAtomWithIdx(position-1);other='left' if side=='R' else 'right';side='right' if side=='R' else 'left'
        reference={position-2:'top',position:'bottom',count+position-1:side,None:other}
        actual=tuple(n.GetIdx() for n in atom.GetNeighbors())+(None,)
        atom.SetChiralTag(_projection_tag(tuple(reference[idx] for idx in actual)))
    closure=None
    if ring!='acyclic':
        closure=carbonyl+(3 if ring=='furanose' else 4)
        if closure>count:raise _SugarInput('The carbon chain cannot form the declared ring size.')
        editable=Chem.RWMol(graph);editable.GetBondBetweenAtoms(carbonyl-1,count+carbonyl-1).SetBondType(Chem.BondType.SINGLE)
        editable.AddBond(carbonyl-1,count+closure-1,Chem.BondType.SINGLE);graph=editable.GetMol();Chem.SanitizeMol(graph)
        if anomer!='unspecified':
            # Equal ordered-port parity is source alpha; beta reverses it.
            # Reference: O / lower-C / higher-C / H. Anomer: ring-C /
            # exocyclic-O / ring-O / acyclic-C (ketose) or H (aldose).
            side='right' if sides[-1]=='R' else 'left';other='left' if side=='right' else 'right'
            tag=_projection_tag((side,'top','bottom',other))
            reference=(carbonyl,count+carbonyl-1,count+closure-1,carbonyl-2 if carbonyl==2 else None)
            atom=graph.GetAtomWithIdx(carbonyl-1);actual=tuple(n.GetIdx() for n in atom.GetNeighbors())
            if carbonyl==1:actual+=(None,)
            atom.SetChiralTag(_reordered_tag(tag,reference,actual,Chem))
            if anomer=='beta':atom.InvertChirality()
    # RDKit's mapped SMILES writer can reverse a newly assigned cyclic atom
    # with an implicit H when that atom becomes the traversal root. Retain
    # that same H explicitly in the atom record before either serialization.
    for atom in graph.GetAtoms():
        if atom.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED and atom.GetNumImplicitHs():
            atom.SetNumExplicitHs(atom.GetNumExplicitHs()+atom.GetNumImplicitHs());atom.SetNoImplicit(True)
    Chem.AssignStereochemistry(graph,cleanIt=True,force=True)
    from rdkit.Chem import rdCIPLabeler
    rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
    expected_centres=len(sides)+(ring!='acyclic' and anomer!='unspecified')
    if sum(a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED for a in graph.GetAtoms())!=expected_centres:
        raise _SugarScope('A declared parent/anomer centre was not retained.')
    return graph,closure


def carbohydrate_to_iupac_result(spec:CarbohydrateSpec) -> CarbohydrateNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        if (not isinstance(spec,CarbohydrateSpec) or not isinstance(spec.ring,str) or spec.ring not in {'acyclic','furanose','pyranose'}
                or not isinstance(spec.anomer,str) or spec.anomer not in {'alpha','beta','unspecified','alpha,beta'}
                or spec.ring=='acyclic' and spec.anomer!='unspecified' or not isinstance(spec.modifications,tuple)):
            raise _SugarInput('Declare an acyclic/furanose/pyranose parent; only a cyclic form has an anomer.')
        if spec.modifications:
            from .carbohydrate_modifications import modified_carbohydrate_result
            return modified_carbohydrate_result(spec,Chem)
        count,carbonyl,sides=_parent_facts(spec.parent);word=_parent_word(count,carbonyl,sides,spec.ring)
        chosen='alpha' if spec.anomer=='alpha,beta' else spec.anomer
        graph,closure=_build_sugar(count,carbonyl,sides,spec.ring,chosen,Chem)
        identity=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True);members=();labels=()
        if spec.anomer=='alpha,beta':
            other,_=_build_sugar(count,carbonyl,sides,spec.ring,'beta',Chem)
            members=(identity,Chem.MolToSmiles(other,canonical=True,isomericSmiles=True));labels=('alpha','beta')
            if len(set(members))!=2:raise _SugarScope('The two declared anomers are not distinct members.')
            identity=''
        prefix={'alpha':'α-','beta':'β-','unspecified':'','alpha,beta':'α,β-'}[spec.anomer]
        for atom in graph.GetAtoms():atom.SetAtomMapNum(atom.GetIdx()+1)
        mapped=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
        recovered=Chem.MolFromSmiles(mapped)
        if recovered is None:raise _SugarScope('The mapped parent graph cannot be recovered.')
        for atom in recovered.GetAtoms():atom.SetAtomMapNum(0)
        if Chem.MolToSmiles(recovered,canonical=True,isomericSmiles=True)!=(members[0] if members else identity):
            raise _SugarScope('The mapped parent serialization does not conserve its physical configuration.')
        roles=tuple([(str(i+1),i+1) for i in range(count)]+[('O'+str(i+1),count+i+1) for i in range(count)])
        return CarbohydrateNamingResult(name=prefix+word,generation_status='available',preferred_status='not-assessed',
            route='explicit-carbohydrate',canonical_smiles=identity,mapped_smiles=mapped,
            parent_atom_maps=roles,carbon_count=count,carbonyl_position=carbonyl,fischer_hydroxy_sides=sides,ring=spec.ring,
            ring_closing_position=closure,ring_oxygen_map=None if closure is None else count+closure,anomer=spec.anomer,
            member_smiles=members,member_anomers=labels,proportions_unspecified=bool(members),
            substance_form='anomer-mixture' if members else 'anomer-unspecified' if spec.ring!='acyclic' and spec.anomer=='unspecified' else 'conventional-neutral')
    except _SugarInput as error:return _failure('invalid-input',str(error))
    except (_SugarScope,ValueError,RuntimeError,OverflowError,KeyError) as error:return _failure('unavailable',str(error))


def carbohydrate_to_iupac(spec:CarbohydrateSpec) -> str:return carbohydrate_to_iupac_result(spec).name
