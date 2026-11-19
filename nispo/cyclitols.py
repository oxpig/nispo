"""Productive P-104 cyclitol rings, retained inositols and original OH joins."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import cos,sin,pi

from .carbohydrates import CarbohydrateModification,_SugarInput,_SugarScope
from .carbohydrate_modifications import (
    _ROLE,_OWNER,_modify,_validate,_name as _hydroxy_name,SugarPhosphateBinding,
)
from .naming_result import NamingResult,SelectionLimitation
from .cyclitol_substitution import CyclitolSubstituent,CyclitolSubstituentBinding


@dataclass(frozen=True,slots=True)
class CyclitolSpec:
    ring_size: int
    above: tuple[int,...]
    below: tuple[int,...]
    numbering: str = 'clockwise'


@dataclass(frozen=True,slots=True)
class InositolSpec:
    parent: str = 'myo'
    series: str = '1L'
    modifications: tuple[CarbohydrateModification,...] = ()
    substituents: tuple[CyclitolSubstituent,...] = ()


@dataclass(frozen=True,slots=True)
class CyclitolNamingResult(NamingResult):
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    parent_atom_maps: tuple[tuple[str,int],...] = ()
    component_atom_maps: tuple[tuple[str,tuple[int,...]],...] = ()
    ring_size: int = 0
    above: tuple[int,...] = ()
    below: tuple[int,...] = ()
    numbering: str = ''
    inositol_parent: str = ''
    declared_series: str = ''
    configuration_descriptor_required: bool = False
    formula: str = ''
    phosphate_bindings: tuple[SugarPhosphateBinding,...] = ()
    systematic_result: NamingResult|None = None
    construction_limitations: tuple[SelectionLimitation,...] = ()
    nomenclature: str = 'specialized-P-104; PIN-not-assessed-under-P-100'
    numbered_parent_atom_maps: tuple[tuple[str,int],...] = ()
    declared_to_numbered_locants: tuple[tuple[str,str],...] = ()
    numbered_series: str = ''
    retained_numbering_complete: bool = False
    substituent_bindings: tuple[CyclitolSubstituentBinding,...] = ()


_FACES={'cis':(1,2,3,4,5,6),'epi':(1,2,3,4,5),'allo':(1,2,3,4),
        'myo':(1,2,3,5),'muco':(1,2,4,5),'neo':(1,2,3),'scyllo':(1,3,5),'chiro':(1,2,4)}
_CHALCOGEN_WORDS={'thio':'sulfanyl','seleno':'selanyl','telluro':'tellanyl'}


def _serialization_copy(graph,Chem):
    # Modern CIP reporting and legacy native SMILES traversal use different
    # computed rank caches. Carry physical tags and clear both caches on a
    # private serialization copy, especially for coupled symmetric rings.
    copy=Chem.Mol(graph)
    for obj in list(copy.GetAtoms())+list(copy.GetBonds()):
        for prop in ('_CIPCode','_CIPRank'):
            if obj.HasProp(prop):obj.ClearProp(prop)
    if copy.HasProp('_StereochemDone'):copy.ClearProp('_StereochemDone')
    return copy


def _serialize(graph,Chem):return Chem.MolToSmiles(_serialization_copy(graph,Chem),canonical=True,isomericSmiles=True)


def _same_isomer(left,right,Chem):
    if left.GetNumAtoms()!=right.GetNumAtoms() or left.GetNumBonds()!=right.GetNumBonds():return False
    a=_serialization_copy(left,Chem);b=_serialization_copy(right,Chem)
    return a.HasSubstructMatch(b,useChirality=True) and b.HasSubstructMatch(a,useChirality=True)


def _failure(status,reason):
    return CyclitolNamingResult('',status,'not-assessed',
        (SelectionLimitation('cyclitol-construction','P-104',reason),),'explicit-cyclitol')


def _validate_ring(spec):
    if (not isinstance(spec,CyclitolSpec) or type(spec.ring_size) is not int or not 3<=spec.ring_size<=8
            or not isinstance(spec.above,tuple) or not isinstance(spec.below,tuple)
            or not isinstance(spec.numbering,str) or spec.numbering not in {'clockwise','anticlockwise'}):
        raise _SugarInput('Declare a three-to-eight-carbon ring and explicit above/below OH positions and numbering direction.')
    positions=spec.above+spec.below
    if (len(positions)<3 or len(set(positions))!=len(positions)
            or any(type(i) is not int or not 1<=i<=spec.ring_size for i in positions)):
        raise _SugarInput('At least three distinct original ring positions must each carry one declared OH.')


def _ring(spec,Chem):
    _validate_ring(spec);n=spec.ring_size;graph=Chem.RWMol()
    for i in range(n):
        atom=Chem.Atom(6);atom.SetProp(_ROLE,str(i+1));atom.SetProp(_OWNER,'parent');graph.AddAtom(atom)
    for i in range(n):graph.AddBond(i,(i+1)%n,Chem.BondType.SINGLE)
    oxygen={}
    for position in sorted(spec.above+spec.below):
        atom=Chem.Atom(8);atom.SetProp(_ROLE,'O'+str(position));atom.SetProp(_OWNER,'parent');idx=graph.AddAtom(atom)
        graph.AddBond(position-1,idx,Chem.BondType.SINGLE);oxygen[position]=idx
    graph=graph.GetMol();Chem.SanitizeMol(graph)
    direction=-1 if spec.numbering=='clockwise' else 1
    xy=tuple((cos(direction*2*pi*i/n),sin(direction*2*pi*i/n)) for i in range(n))
    for position,idx in oxygen.items():
        center=graph.GetAtomWithIdx(position-1);x,y=xy[position-1];face=1 if position in spec.above else -1
        probe=Chem.MolFromSmiles('C(F)(Cl)(Br)I');conformer=Chem.Conformer(5);conformer.Set3D(True)
        for i,neighbor in enumerate(center.GetNeighbors(),1):
            target=neighbor.GetIdx()
            conformer.SetAtomPosition(i,(0,0,face) if target==idx else (xy[target][0]-x,xy[target][1]-y,0))
        conformer.SetAtomPosition(4,(0,0,-face));probe.AddConformer(conformer);Chem.AssignAtomChiralTagsFromStructure(probe)
        center.SetChiralTag(probe.GetAtomWithIdx(0).GetChiralTag());center.SetNumExplicitHs(1);center.SetNoImplicit(True)
    graph.UpdatePropertyCache(strict=True);Chem.AssignStereochemistry(graph,cleanIt=True,force=True)
    from rdkit.Chem import rdCIPLabeler
    rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
    return graph


def _inositol_ring(parent,series,Chem):
    above=_FACES[parent];below=tuple(i for i in range(1,7) if i not in above)
    return _ring(CyclitolSpec(6,above,below,'clockwise' if series=='1L' else 'anticlockwise'),Chem)


@lru_cache(maxsize=1)
def _retained_graphs():
    from .api import Chem
    rows={}
    for parent in _FACES:
        for series in (('1L','1D') if parent=='chiro' else ('1L',)):
            graph=_inositol_ring(parent,series,Chem);identity=_serialize(graph,Chem)
            if identity in rows:raise _SugarScope('Two published inositol face patterns lost their distinct original configuration.')
            rows[identity]=(parent,series,graph)
    if len(rows)!=9:raise _SugarScope('All nine source inositol configurations are not retained.')
    return rows


@lru_cache(maxsize=128)
def _retained_frames(above,clockwise,parent):
    """Every proper view of one numbered ring with source-canonical faces.

    A change of viewed face reverses both the OH signs and the winding. This
    is a rigid proper rotation of the drawing, never an enantiomer choice.
    Fixed retained face sets implement P-104.2.3(a-c).
    """
    face=set(above);target=set(_FACES[parent]);rows=[]
    for start in range(1,7):
        for step in (1,-1):
            path=tuple((start-1+step*j)%6+1 for j in range(6))
            flip=1 if start in face else -1
            candidate={j+1 for j,old in enumerate(path) if (old in face)==(flip==1)}
            if candidate!=target:continue
            winding=(-1 if clockwise else 1)*step*flip
            series='1L' if winding==-1 else '1D'
            old_to_new=tuple((old,path.index(old)+1) for old in range(1,7))
            rows.append((old_to_new,series))
    return tuple(rows)


def _preferred_retained_frame(above,clockwise,parent,mods,terms,chiral):
    # A deleted source OH erases a defining parent port. Mixed class-word
    # esters need an additional citation-priority proof. Do not call a search
    # over surviving original frames a complete comparison in either case.
    kinds={m.kind for m in mods}
    if 'deoxy' in kinds or kinds&{'O-acetyl','phosphate'} and len(kinds)>1:return None
    groups=defaultdict(list)
    for position,word,marker in terms:groups[word,marker].append(position)
    rows=[]
    for pairs,series in _retained_frames(tuple(sorted(above)),clockwise,parent):
        mapping=dict(pairs)
        sites=tuple(sorted(mapping[m.position] for m in mods))
        from .formatting import prefix_alphabetization_key
        cited=tuple(tuple(sorted(mapping[p] for p in groups[group])) for group in sorted(groups,key=lambda g:(prefix_alphabetization_key(g[0]),g[1])))
        key=(sites,cited,(0 if series=='1L' else 1) if not chiral else 0)
        rows.append((key,pairs,series))
    if not rows:raise _SugarScope('The complete physical retained-parent frame family is unavailable.')
    best_key=min(r[0] for r in rows);best=[r for r in rows if r[0]==best_key]
    # A chiral tie must not acquire L preference from the meso-only rule.
    if len({(series,tuple(sorted((dict(pairs)[m.position],m.kind) for m in mods)))
            for _key,pairs,series in best})!=1:return None
    _key,pairs,series=min(best,key=lambda row:row[1]);return dict(pairs),series


def _pack(graph,spec,name,Chem,*,parent='',series='',descriptor_required=False,phosphates=(),systematic=None,issues=(),numbered=None,substituent_bindings=()):
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    identity=_serialize(graph,Chem);roles=[];components=defaultdict(list)
    for atom in graph.GetAtoms():
        number=atom.GetIdx()+1;atom.SetAtomMapNum(number);components[atom.GetProp(_OWNER)].append(number)
        if atom.HasProp(_ROLE):roles.append((atom.GetProp(_ROLE),number))
    mapped=_serialize(graph,Chem);restored=Chem.MolFromSmiles(mapped)
    if restored is None:raise _SugarScope('The complete cyclitol graph cannot be recovered from its mapped serialization.')
    for atom in restored.GetAtoms():atom.SetAtomMapNum(0)
    if not _same_isomer(graph,restored,Chem):raise _SugarScope('Cyclitol mapping changes the original configuration.')
    bindings=[];rolemap=dict(roles)
    for position,p,acidic in phosphates:
        owner=f'mod:{position}:phosphate'
        bindings.append(SugarPhosphateBinding(position,rolemap['O'+str(position)],p+1,tuple(i+1 for i in acidic),tuple(components[owner])))
    numbered_roles=();numbered_locants=();numbered_series=''
    if numbered is not None:
        mapping,numbered_series=numbered
        numbered_locants=tuple((str(i),str(mapping[i])) for i in range(1,7))
        numbered_roles=tuple((('O' if role.startswith('O') else '')+str(mapping[int(role.lstrip('O'))]),number) for role,number in roles)
    return CyclitolNamingResult(name,'available' if name else 'unavailable',
        systematic.preferred_status if systematic is not None else 'not-assessed',systematic.unresolved if systematic is not None else (),
        'explicit-cyclitol',canonical_smiles=identity,mapped_smiles=mapped,parent_atom_maps=tuple(roles),
        component_atom_maps=tuple((owner,tuple(values)) for owner,values in sorted(components.items())),ring_size=spec.ring_size,
        above=tuple(sorted(spec.above)),below=tuple(sorted(spec.below)),numbering=spec.numbering,inositol_parent=parent,declared_series=series,
        configuration_descriptor_required=descriptor_required,formula=CalcMolFormula(graph),phosphate_bindings=tuple(bindings),systematic_result=systematic,
        construction_limitations=tuple(issues),nomenclature='systematic-P-1-P-9' if systematic is not None else 'specialized-P-104; PIN-not-assessed-under-P-100',
        numbered_parent_atom_maps=numbered_roles,declared_to_numbered_locants=numbered_locants,
        numbered_series=numbered_series,retained_numbering_complete=numbered is not None,substituent_bindings=tuple(substituent_bindings))


def inositol_to_iupac_result(spec:InositolSpec) -> CyclitolNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        if (not isinstance(spec,InositolSpec) or not isinstance(spec.parent,str) or spec.parent not in _FACES
                or not isinstance(spec.series,str) or spec.series not in {'1L','1D'} or not isinstance(spec.modifications,tuple)
                or not isinstance(spec.substituents,tuple)):
            raise _SugarInput('Declare a named inositol, its 1L/1D numbered drawing and immutable original-site modifications.')
        _validate(spec.modifications,6)
        base=_inositol_ring(spec.parent,spec.series,Chem)
        graph,terms,esters,phosphates=_modify(base,6,0,spec.modifications,Chem)
        bindings=()
        if spec.substituents:
            from .cyclitol_substitution import substitute
            graph,substituent_terms,bindings=substitute(graph,spec.substituents,Chem)
            terms=list(terms)+substituent_terms
        # P-104.3.1 uses deoxy/sulfanyl, rather than the carbohydrate thio
        # functional-replacement spelling, for hydroxyl-to-SH derivatives.
        converted=[]
        for position,word,marker in terms:
            if word in _CHALCOGEN_WORDS:converted.extend(((position,_CHALCOGEN_WORDS[word],''),(position,'deoxy','')))
            else:converted.append((position,word,marker))
        mirror=Chem.Mol(graph)
        for atom in mirror.GetAtoms():
            if atom.GetChiralTag() in {Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW}:atom.InvertChirality()
        chiral=not _same_isomer(graph,mirror,Chem)
        required=spec.parent=='chiro' or chiral
        above=_FACES[spec.parent]
        numbering_mods=spec.modifications+tuple(CarbohydrateModification(b.position,b.attachment+'-group:'+b.organyl_name) for b in bindings)
        numbered=(None if any(b.has_annotations or b.unassigned_stereo_atom_maps for b in bindings) else
            _preferred_retained_frame(above,spec.series=='1L',spec.parent,numbering_mods,converted,chiral))
        mapping,selected_series=numbered if numbered is not None else ({i:i for i in range(1,7)},spec.series)
        terms=[(mapping[p],word,marker) for p,word,marker in converted]
        named_mods=tuple(CarbohydrateModification(mapping[m.position],m.kind) for m in spec.modifications)
        named_phosphates=tuple(sorted((mapping[p],idx,acid) for p,idx,acid in phosphates))
        if bindings:
            from .cyclitol_substitution import prefix_terms
            body=_hydroxy_name(spec.parent+'-inositol','unspecified',[],[mapping[p] for p in esters],named_phosphates,named_mods,base,6,0,Chem)
            prefix=prefix_terms(terms);name=(prefix+'-' if prefix else '')+body
        else:name=_hydroxy_name(spec.parent+'-inositol','unspecified',terms,[mapping[p] for p in esters],named_phosphates,named_mods,base,6,0,Chem)
        if required:name=selected_series+'-'+name
        issues=()
        if numbered is None:issues=(SelectionLimitation('inositol-numbering','P-104.2.3/P-104.3.1',
            'Original numbered construction is retained; global preferred-numbering comparisons for this derivative remain unproved.'),)
        if any(b.unassigned_stereo_atom_maps for b in bindings):issues+=(SelectionLimitation('inositol-substituent-configuration','P-93',
            'Potential organyl stereocentres remain unassigned exactly as declared by the source graph and new attachment.'),)
        ring=CyclitolSpec(6,above,tuple(i for i in range(1,7) if i not in above),'clockwise' if spec.series=='1L' else 'anticlockwise')
        return _pack(graph,ring,name,Chem,parent=spec.parent,series=spec.series,descriptor_required=required,phosphates=phosphates,issues=issues,
            numbered=numbered,substituent_bindings=bindings)
    except _SugarInput as error:return _failure('invalid-input',str(error))
    except (_SugarScope,ValueError,RuntimeError,OverflowError,TypeError,KeyError) as error:return _failure('unavailable',str(error))


def inositol_to_iupac(spec:InositolSpec) -> str:return inositol_to_iupac_result(spec).name


def cyclitol_to_iupac_result(spec:CyclitolSpec) -> CyclitolNamingResult:
    from .api import Chem,mol_to_iupac_result
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        graph=_ring(spec,Chem)
        if spec.ring_size==6 and len(spec.above+spec.below)==6:
            registry=_retained_graphs();match=registry.get(_serialize(graph,Chem))
            if match is None:
                matches=[row for row in registry.values() if _same_isomer(graph,row[2],Chem)]
                if len(matches)==1:match=matches[0]
            if match is None:raise _SugarScope('The complete six-OH configuration has no proved retained inositol correspondence.')
            parent,series,_template=match;name=(series+'-' if parent=='chiro' else '')+parent+'-inositol'
            numbered=_preferred_retained_frame(spec.above,spec.numbering=='clockwise',parent,(),(),parent=='chiro')
            if numbered is None:raise _SugarScope('The complete retained-parent correspondence has no proved physical numbering.')
            if parent=='chiro':name=numbered[1]+'-'+parent+'-inositol'
            return _pack(graph,spec,name,Chem,parent=parent,series=series,descriptor_required=parent=='chiro',numbered=numbered)
        systematic=mol_to_iupac_result(graph)
        return _pack(graph,spec,systematic.name,Chem,systematic=systematic)
    except _SugarInput as error:return _failure('invalid-input',str(error))
    except (_SugarScope,ValueError,RuntimeError,OverflowError,TypeError,KeyError) as error:return _failure('unavailable',str(error))


def cyclitol_to_iupac(spec:CyclitolSpec) -> str:return cyclitol_to_iupac_result(spec).name
