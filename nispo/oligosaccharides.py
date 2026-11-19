"""P-102.7 O-glycosidic trees/cycles with explicit original-site ownership."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .carbohydrates import CarbohydrateSpec,_SugarInput,_SugarScope,_reordered_tag
from .carbohydrate_modifications import _ROLE,_OWNER,SugarPhosphateBinding
from .glycosyl_construction import sugar_component
from .naming_result import NamingResult,SelectionLimitation


_UID='_nispo_oligosaccharide_atom'
_RESIDUE='_nispo_oligosaccharide_residue'


@dataclass(frozen=True,slots=True)
class SugarLink:
    """Zero-based residues; donor position is its declared carbonyl position."""
    donor_residue: int
    acceptor_residue: int
    acceptor_position: int


@dataclass(frozen=True,slots=True)
class OligosaccharideSpec:
    residues: tuple[CarbohydrateSpec,...]
    links: tuple[SugarLink,...]
    glycoside_residue: int|None = None


@dataclass(frozen=True,slots=True)
class CyclicOligosaccharideSpec:
    residue: CarbohydrateSpec
    repeat_count: int
    acceptor_position: int = 4
    naming_method: str = 'auto'


@dataclass(frozen=True,slots=True)
class SugarLinkBinding:
    donor_residue: int
    acceptor_residue: int
    donor_position: int
    acceptor_position: int
    donor_carbon_map: int
    bridge_oxygen_map: int
    acceptor_carbon_map: int
    bridge_oxygen_origin: str = 'acceptor'


@dataclass(frozen=True,slots=True)
class OligosaccharidePhosphateBinding:
    residue: int
    phosphate: SugarPhosphateBinding


@dataclass(frozen=True,slots=True)
class OligosaccharideNamingResult(NamingResult):
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    residue_atom_maps: tuple[tuple[tuple[str,int],...],...] = ()
    component_atom_maps: tuple[tuple[str,tuple[int,...]],...] = ()
    glycosidic_bindings: tuple[SugarLinkBinding,...] = ()
    phosphate_bindings: tuple[OligosaccharidePhosphateBinding,...] = ()
    reducing_residue: int|None = None
    glycoside_residue: int|None = None
    parent_selection: str = ''
    residue_anomers: tuple[str,...] = ()
    member_smiles: tuple[str,...] = ()
    member_anomers: tuple[str,...] = ()
    proportions_unspecified: bool = False
    substance_form: str = 'conventional-neutral'
    nomenclature: str = 'specialized-P-102.7; PIN-not-assessed-under-P-100'
    is_cyclic: bool = False
    cyclic_repeat_count: int = 0
    cyclic_acceptor_position: int|None = None


def _failure(status,reason):
    return OligosaccharideNamingResult('',status,'not-assessed',
        (SelectionLimitation('oligosaccharide-construction','P-102.7',reason),),'explicit-oligosaccharide')


def _atom(graph,residue,role):
    atoms=[a for a in graph.GetAtoms() if a.GetIntProp(_RESIDUE)==residue
           and a.HasProp(_ROLE) and a.GetProp(_ROLE)==role]
    if len(atoms)!=1:raise _SugarInput('An original sugar role is unavailable for a glycosidic link.')
    return atoms[0]


def _validate(spec):
    if (not isinstance(spec,OligosaccharideSpec) or not isinstance(spec.residues,tuple)
            or not 2<=len(spec.residues)<=16 or not isinstance(spec.links,tuple)
            or len(spec.links)!=len(spec.residues)-1
            or any(not isinstance(s,CarbohydrateSpec) or not isinstance(s.anomer,str)
                   or s.anomer not in {'alpha','beta','unspecified','alpha,beta'} for s in spec.residues)):
        raise _SugarInput('Declare two to sixteen cyclic sugar residues and a tree of explicit O-glycosidic links.')
    n=len(spec.residues)
    if spec.glycoside_residue is not None and (type(spec.glycoside_residue) is not int or not 0<=spec.glycoside_residue<n):
        raise _SugarInput('The optional glycoside parent must identify one declared residue.')
    adjacent=[set() for _ in range(n)];donors=set();sites=set()
    for link in spec.links:
        if (not isinstance(link,SugarLink) or type(link.donor_residue) is not int or type(link.acceptor_residue) is not int
                or type(link.acceptor_position) is not int or not 0<=link.donor_residue<n
                or not 0<=link.acceptor_residue<n or link.donor_residue==link.acceptor_residue
                or link.acceptor_position<1):
            raise _SugarInput('Every link requires distinct zero-based donor/acceptor residues and an original acceptor position.')
        if link.donor_residue in donors or (link.acceptor_residue,link.acceptor_position) in sites:
            raise _SugarInput('A donor anomeric OH or acceptor OH cannot be consumed twice.')
        donors.add(link.donor_residue);sites.add((link.acceptor_residue,link.acceptor_position))
        adjacent[link.donor_residue].add(link.acceptor_residue);adjacent[link.acceptor_residue].add(link.donor_residue)
    seen={0};todo=[0]
    while todo:
        for other in adjacent[todo.pop()]-seen:seen.add(other);todo.append(other)
    if len(seen)!=n or sum(map(len,adjacent))!=2*(n-1):
        raise _SugarInput('The residue graph must be a connected tree with no duplicate or cyclic links.')
    return donors


def _topology(spec,facts,donors):
    core=[];occupied=set();children=defaultdict(list)
    for link in spec.links:
        d,a,p=link.donor_residue,link.acceptor_residue,link.acceptor_position
        ports=((d,facts[d]['carbonyl']),(a,p))
        if any(port in occupied for port in ports):raise _SugarInput('Two glycosidic links consume the same original sugar OH.')
        occupied.update(ports)
        if p==facts[a]['carbonyl']:core.append(link)
        else:children[a].append(link)
    if len(core)>1:raise _SugarInput('A carbohydrate tree has at most one anomer-to-anomer core.')
    root=next(i for i in range(len(facts)) if i not in donors)
    if core and core[0].acceptor_residue!=root:raise _SugarInput('A bound anomeric acceptor cannot donate a second glycosidic link.')
    fixed=donors|({root} if core else set())
    if any(spec.residues[i].anomer not in {'alpha','beta'} for i in fixed):
        raise _SugarScope('Every bonded anomer requires explicit alpha or beta configuration; only a free end may be unspecified or a mixture.')
    if not core and spec.glycoside_residue is not None:
        raise _SugarInput('A tree with a free hemiacetal ends in the reducing residue, rather than a chosen glycoside.')
    if core and spec.glycoside_residue is not None and spec.glycoside_residue not in {core[0].donor_residue,root}:
        raise _SugarInput('The glycoside portion must be one of the two anomer-to-anomer core residues.')
    return children,root,core[0] if core else None


def _parent_key(facts):
    # P-102.4(a)-(c) for the unmodified aldose/2-ketose source families.
    word=facts['parent_name'];series=0 if word.startswith('D-') else 1
    return (facts['carbonyl']!=1,-facts['count'],word[2:],series,facts['anomer']!='alpha')


def _render(spec,facts,children,root,core):
    def depth(i):return 1+max((depth(e.donor_residue) for e in children[i]),default=0)
    def component(i,context):
        word=facts[i]['context_names'][context]
        return '('+word+')' if context=='glycosyl' and word[0].isdigit() else word
    def tree(i,context):
        word=component(i,context);edges=children[i]
        if not edges:return word
        ordered=sorted(edges,key=lambda e:(-depth(e.donor_residue),e.acceptor_position))
        main=ordered[0]
        def link_text(e):return tree(e.donor_residue,'glycosyl')+f"-({facts[e.donor_residue]['carbonyl']}→{e.acceptor_position})"
        branches=''.join('['+link_text(e)+']-' for e in sorted(ordered[1:],key=lambda e:e.acceptor_position))
        return link_text(main)+'-'+branches+word
    if core is None:return tree(root,'sugar'),None,'free-hemiacetal',()
    ends=(core.donor_residue,core.acceptor_residue)
    issues=[]
    if spec.glycoside_residue is not None:
        parent=spec.glycoside_residue;selection='declared-glycoside-parent'
        issues.append(SelectionLimitation('oligosaccharide-parent','P-102.4/P-102.7.2.1',
            'The declared glycoside parent is an explicit construction choice; preferred parent comparison is not certified.'))
    else:
        keys={i:_parent_key(facts[i]) for i in ends}
        # Equal source parents can still have different substituted subtrees.
        # Do not use a rendered-name string as a chemical ranking tiebreak.
        if keys[ends[0]]==keys[ends[1]] and any(children[i] for i in ends):
            raise _SugarScope('Equivalent core parents with different residue branches require an explicit glycoside parent.')
        parent=min(ends,key=lambda i:keys[i]);selection='source-simple-P-102.4'
        if any(spec.residues[i].modifications for i in ends):
            selection='modified-parent-comparison-pending'
            issues.append(SelectionLimitation('oligosaccharide-parent','P-102.4',
                'Simple source-parent priorities choose this construction; full modified-parent class/prefix comparison remains unresolved.'))
    other=next(i for i in ends if i!=parent)
    return tree(other,'glycosyl')+' '+tree(parent,'glycoside'),parent,selection,tuple(issues)


def _join(spec,pieces,facts,Chem):
    graph=Chem.RWMol()
    for i,piece in enumerate(pieces):
        copy=Chem.Mol(piece)
        for atom in copy.GetAtoms():
            atom.SetIntProp(_RESIDUE,i);atom.SetProp(_UID,f'{i}:{atom.GetIdx()}')
            atom.SetProp(_OWNER,f'residue:{i}:'+atom.GetProp(_OWNER));atom.SetAtomMapNum(0)
        graph.InsertMol(copy)
    for e in sorted(spec.links,key=lambda e:(e.donor_residue,e.acceptor_residue,e.acceptor_position)):
        d,a,p=e.donor_residue,e.acceptor_residue,e.acceptor_position
        donor=_atom(graph,d,str(facts[d]['carbonyl']));leaving=_atom(graph,d,'O'+str(facts[d]['carbonyl']))
        acceptor=_atom(graph,a,'O'+str(p))
        for oxygen in (leaving,acceptor):
            if oxygen.GetAtomicNum()!=8 or oxygen.GetDegree()!=1 or oxygen.GetTotalNumHs()!=1:
                raise _SugarInput('Every condensation requires a free original oxygen OH on both residues.')
        if acceptor.GetNeighbors()[0].GetAtomicNum()!=6:raise _SugarInput('The acceptor OH must be attached to its original carbon.')
        tag=donor.GetChiralTag();donor_uid=donor.GetProp(_UID);bridge_uid=acceptor.GetProp(_UID)
        reference=tuple(bridge_uid if n.GetIdx()==leaving.GetIdx() else n.GetProp(_UID) for n in donor.GetNeighbors())
        if acceptor.GetNumExplicitHs():acceptor.SetNumExplicitHs(acceptor.GetNumExplicitHs()-1)
        graph.RemoveAtom(leaving.GetIdx())
        by_uid={atom.GetProp(_UID):atom for atom in graph.GetAtoms()};donor=by_uid[donor_uid];acceptor=by_uid[bridge_uid]
        graph.AddBond(donor.GetIdx(),acceptor.GetIdx(),Chem.BondType.SINGLE);graph.UpdatePropertyCache(strict=False)
        actual=tuple(n.GetProp(_UID) for n in donor.GetNeighbors())
        if len(reference)==3:reference+=(None,);actual+=(None,)
        donor.SetChiralTag(_reordered_tag(tag,reference,actual,Chem) if tag!=Chem.ChiralType.CHI_UNSPECIFIED else tag)
    result=graph.GetMol();Chem.SanitizeMol(result);Chem.AssignStereochemistry(result,cleanIt=True,force=True)
    from rdkit.Chem import rdCIPLabeler
    rdCIPLabeler.AssignCIPLabels(result,maxRecursiveIterations=1000000)
    return result


def _serialization_copy(graph,Chem):
    copy=Chem.Mol(graph)
    for atom in copy.GetAtoms():
        for prop in ('_CIPCode','_CIPRank'):
            if atom.HasProp(prop):atom.ClearProp(prop)
    if copy.HasProp('_StereochemDone'):copy.ClearProp('_StereochemDone')
    return copy


def _smiles(graph,Chem):return Chem.MolToSmiles(_serialization_copy(graph,Chem),canonical=True,isomericSmiles=True)


def _mapped_parts(spec,graph,facts,Chem):
    roles=[[] for _ in spec.residues];owners=defaultdict(list)
    for atom in graph.GetAtoms():
        number=atom.GetIdx()+1;atom.SetAtomMapNum(number);owners[atom.GetProp(_OWNER)].append(number)
        if atom.HasProp(_ROLE):roles[atom.GetIntProp(_RESIDUE)].append((atom.GetProp(_ROLE),number))
    mapped=_smiles(graph,Chem);restored=Chem.MolFromSmiles(mapped)
    if restored is None:raise _SugarScope('The complete mapped carbohydrate graph cannot be recovered.')
    for atom in restored.GetAtoms():atom.SetAtomMapNum(0)
    original=_serialization_copy(graph,Chem)
    for atom in original.GetAtoms():atom.SetAtomMapNum(0)
    if (not original.HasSubstructMatch(restored,useChirality=True) or not restored.HasSubstructMatch(original,useChirality=True)
            or original.GetNumAtoms()!=restored.GetNumAtoms()):
        raise _SugarScope('Mapped carbohydrate serialization changes physical configuration.')
    bindings=tuple(SugarLinkBinding(e.donor_residue,e.acceptor_residue,facts[e.donor_residue]['carbonyl'],e.acceptor_position,
        _atom(graph,e.donor_residue,str(facts[e.donor_residue]['carbonyl'])).GetIdx()+1,
        _atom(graph,e.acceptor_residue,'O'+str(e.acceptor_position)).GetIdx()+1,
        _atom(graph,e.acceptor_residue,str(e.acceptor_position)).GetIdx()+1)
        for e in sorted(spec.links,key=lambda e:(e.donor_residue,e.acceptor_residue,e.acceptor_position)))
    phosphates=[]
    for i,f in enumerate(facts):
        for position in f['phosphate_sites']:
            maps=tuple(owners[f'residue:{i}:mod:{position}:phosphate']);atoms=[graph.GetAtomWithIdx(n-1) for n in maps]
            p=next(a for a in atoms if a.GetAtomicNum()==15);acid=tuple(a.GetIdx()+1 for a in atoms if a.GetAtomicNum()==8 and a.GetDegree()==1 and a.GetTotalNumHs()==1)
            if len(acid)!=2:raise _SugarScope('A residue phosphate does not retain its two acidic hydrogen atoms.')
            phosphates.append(OligosaccharidePhosphateBinding(i,SugarPhosphateBinding(position,
                _atom(graph,i,'O'+str(position)).GetIdx()+1,p.GetIdx()+1,acid,maps)))
    return mapped,tuple(tuple(sorted(row)) for row in roles),tuple((owner,tuple(maps)) for owner,maps in sorted(owners.items())),bindings,tuple(phosphates)


def oligosaccharide_to_iupac_result(spec:OligosaccharideSpec) -> OligosaccharideNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        donors=_validate(spec);preliminary=[]
        for residue in spec.residues:
            interpretation='alpha' if residue.anomer=='alpha,beta' else residue.anomer
            preliminary.append(sugar_component(residue,interpretation,Chem,context='glycosyl',additional_contexts=('sugar','glycoside')))
        children,root,core=_topology(spec,[item[2] for item in preliminary],donors)
        if core is None and {'O-acetyl','phosphate'}<={m.kind for m in spec.residues[root].modifications}:
            raise _SugarScope('Joint acetate/acid-phosphate class-word citation on the free reducing end requires a separate construction.')
        interpretations=('alpha','beta') if spec.residues[root].anomer=='alpha,beta' else (spec.residues[root].anomer,)
        constructions=[]
        for interpretation in interpretations:
            components=list(preliminary)
            if len(interpretations)>1:
                components[root]=sugar_component(spec.residues[root],interpretation,Chem,context='glycosyl',additional_contexts=('sugar','glycoside'))
            facts=[item[2] for item in components]
            name,parent,selection,issues=_render(spec,facts,children,root,core)
            graph=_join(spec,[item[0] for item in components],facts,Chem)
            constructions.append((facts[root]['anomer'],graph,name,facts,parent,selection,issues))
        constructions.sort(key=lambda row:row[0]);anomer,graph,name,facts,parent,selection,issues=constructions[0]
        members=tuple(_smiles(row[1],Chem) for row in constructions) if len(constructions)>1 else ()
        if members:
            if len(set(members))!=2:raise _SugarScope('The reducing-end anomer mixture does not have two distinct members.')
            mixed=list(facts)
            mixed[root]=sugar_component(spec.residues[root],'alpha',Chem,context='glycosyl',substance_anomer='alpha,beta',additional_contexts=('sugar','glycoside'))[2]
            name,parent,selection,issues=_render(spec,mixed,children,root,core)
        identity='' if members else _smiles(graph,Chem)
        mapped,roles,owners,bindings,phosphates=_mapped_parts(spec,graph,facts,Chem)
        anomers=tuple('alpha,beta' if i==root and members else f['anomer'] for i,f in enumerate(facts))
        return OligosaccharideNamingResult(name=name,generation_status='available',preferred_status='not-assessed',unresolved=issues,
            route='explicit-oligosaccharide',canonical_smiles=identity,mapped_smiles=mapped,
            residue_atom_maps=roles,component_atom_maps=owners,
            glycosidic_bindings=bindings,phosphate_bindings=phosphates,reducing_residue=None if core else root,
            glycoside_residue=parent,parent_selection=selection,residue_anomers=anomers,member_smiles=members,
            member_anomers=tuple(row[0] for row in constructions) if members else (),proportions_unspecified=bool(members),
            substance_form='anomer-mixture' if members else 'anomer-unspecified' if core is None and anomer=='unspecified' else 'conventional-neutral')
    except _SugarInput as error:return _failure('invalid-input',str(error))
    except (_SugarScope,ValueError,RuntimeError,OverflowError,KeyError,IndexError) as error:return _failure('unavailable',str(error))


def oligosaccharide_to_iupac(spec:OligosaccharideSpec) -> str:return oligosaccharide_to_iupac_result(spec).name


def cyclic_oligosaccharide_to_iupac_result(spec:CyclicOligosaccharideSpec) -> OligosaccharideNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        if (not isinstance(spec,CyclicOligosaccharideSpec) or not isinstance(spec.residue,CarbohydrateSpec)
                or type(spec.repeat_count) is not int or not 3<=spec.repeat_count<=16
                or type(spec.acceptor_position) is not int or spec.acceptor_position<1
                or not isinstance(spec.naming_method,str) or spec.naming_method not in {'auto','systematic'}
                or not isinstance(spec.residue.anomer,str) or spec.residue.anomer not in {'alpha','beta'}):
            raise _SugarInput('Declare one alpha/beta cyclic residue, three to sixteen repeats, an original acceptor OH position and auto/systematic notation.')
        unit,word,facts=sugar_component(spec.residue,spec.residue.anomer,Chem,context='glycosyl')
        if spec.acceptor_position==facts['carbonyl']:
            raise _SugarInput('A homogeneous cyclic glycosyl chain requires a nonanomeric acceptor OH.')
        n=spec.repeat_count;position=spec.acceptor_position
        construction=OligosaccharideSpec((spec.residue,)*n,tuple(SugarLink(i,(i+1)%n,position) for i in range(n)))
        graph=_join(construction,[unit]*n,[facts]*n,Chem)
        from .constants import MULT
        from .formatting import enclose_name_unit
        if (spec.naming_method=='auto' and not spec.residue.modifications and facts['parent_name']=='D-glucopyranose'
                and facts['anomer']=='alpha' and position==4):
            name='cyclomalto'+MULT[n]+'ose'
        else:
            multiplier='tris' if n==3 else MULT[n]+'kis'
            rendered=enclose_name_unit(word) if word[0].isdigit() else word
            name=f"cyclo{multiplier}-({facts['carbonyl']}→{position})-"+rendered
        identity=_smiles(graph,Chem);mapped,roles,owners,bindings,phosphates=_mapped_parts(construction,graph,[facts]*n,Chem)
        return OligosaccharideNamingResult(name=name,generation_status='available',preferred_status='not-assessed',
            route='explicit-cyclic-oligosaccharide',canonical_smiles=identity,mapped_smiles=mapped,residue_atom_maps=roles,
            component_atom_maps=owners,glycosidic_bindings=bindings,phosphate_bindings=phosphates,
            parent_selection='homogeneous-cycle-1996-37.4',residue_anomers=(facts['anomer'],)*n,
            is_cyclic=True,cyclic_repeat_count=n,cyclic_acceptor_position=position)
    except _SugarInput as error:return _failure('invalid-input',str(error))
    except (_SugarScope,ValueError,RuntimeError,OverflowError,KeyError,IndexError) as error:return _failure('unavailable',str(error))


def cyclic_oligosaccharide_to_iupac(spec:CyclicOligosaccharideSpec) -> str:return cyclic_oligosaccharide_to_iupac_result(spec).name
