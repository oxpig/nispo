"""Source-typed P-102 deoxy, amino, halo, chalcogen and O operations."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .carbohydrates import (
    CarbohydrateModification,CarbohydrateNamingResult,_SugarInput,_SugarScope,
    _parent_facts,_parent_word,_build_sugar,_CONFIG_PREFIX,_mirror,_carbohydrate_root,
)


_ROLE='_nispo_sugar_source_role'
_OWNER='_nispo_sugar_owner'
_HALOGENS={'fluoro':9,'chloro':17,'bromo':35,'iodo':53}
_CHALCOGENS={'thio':16,'seleno':34,'telluro':52}
_KINDS={'deoxy','amino','acetamido',*_HALOGENS,*_CHALCOGENS,'O-methyl','O-ethyl','O-acetyl','phosphate'}


@dataclass(frozen=True,slots=True)
class SugarPhosphateBinding:
    position: int
    sugar_oxygen_map: int
    phosphorus_map: int
    acidic_atom_maps: tuple[int,...]
    atom_maps: tuple[int,...]
    acid_hydrogen_count: int = 2
    phosphorus_configuration: str = 'unspecified'


@dataclass(frozen=True,slots=True)
class ModifiedCarbohydrateNamingResult(CarbohydrateNamingResult):
    source_fischer_hydroxy_sides: str = ''
    declared_anomer: str = ''
    modifications: tuple[CarbohydrateModification,...] = ()
    component_atom_maps: tuple[tuple[str,tuple[int,...]],...] = ()
    ring_heteroatom_map: int|None = None
    phosphate_bindings: tuple[SugarPhosphateBinding,...] = ()


def _atom(graph,role):
    found=[a for a in graph.GetAtoms() if a.HasProp(_ROLE) and a.GetProp(_ROLE)==role]
    if len(found)!=1:raise _SugarInput('The requested parent role is unavailable: '+role)
    return found[0]


def _validate(modifications,count,*,prefix_context=False):
    sites=[]
    for mod in modifications:
        if (not isinstance(mod,CarbohydrateModification) or type(mod.position) is not int
                or not 1<=mod.position<=count or not isinstance(mod.kind,str) or mod.kind not in _KINDS):
            raise _SugarInput('Every operation needs an original sugar position and one supported explicit modification.')
        sites.append(mod.position)
    if len(sites)!=len(set(sites)):raise _SugarInput('Two operations cannot consume the same parent site in this construction.')
    if not prefix_context and any(m.kind=='O-acetyl' for m in modifications) and any(m.kind=='phosphate' for m in modifications):
        raise _SugarScope('The joint acetate/acid-phosphate citation requires a separately proved ester-class construction.')


def _append(graph,Chem,number,owner):
    atom=Chem.Atom(number);atom.SetProp(_OWNER,owner);return graph.AddAtom(atom)


def _consume(atom):
    if atom.GetTotalNumHs()<1:raise _SugarInput('The site has no available hydrogen for this operation.')
    if atom.GetNumExplicitHs():atom.SetNumExplicitHs(atom.GetNumExplicitHs()-1)


def _modify(base,count,carbonyl,mods,Chem):
    graph=Chem.RWMol(base);terms=[];esters=[];phosphates=[]
    for atom in graph.GetAtoms():
        idx=atom.GetIdx();atom.SetProp(_ROLE,str(idx+1) if idx<count else 'O'+str(idx-count+1));atom.SetProp(_OWNER,'parent')
    for mod in sorted(mods,key=lambda m:m.position):
        position=mod.position;oxygen=_atom(graph,'O'+str(position));idx=oxygen.GetIdx();owner=f'mod:{position}:{mod.kind}'
        free_oh=oxygen.GetAtomicNum()==8 and oxygen.GetDegree()==1 and oxygen.GetTotalNumHs()==1
        if mod.kind in _CHALCOGENS:
            if oxygen.GetAtomicNum()!=8:raise _SugarInput('Chalcogen replacement needs the source oxygen atom.')
            oxygen.SetAtomicNum(_CHALCOGENS[mod.kind]);oxygen.SetNumExplicitHs(0);oxygen.SetNoImplicit(False)
            terms.append((position,mod.kind,''))
        else:
            if not free_oh:raise _SugarInput('This operation requires an available free sugar OH.')
            if position==carbonyl:
                if mod.kind=='O-methyl' or mod.kind=='O-ethyl':
                    raise _SugarScope('Anomeric O-alkylation requires the glycoside functional-class construction.')
                if mod.kind not in {'O-acetyl','phosphate'}:
                    raise _SugarScope('This anomeric substitution requires a separate glycosyl/characteristic-class construction.')
            if mod.kind=='deoxy':
                host=oxygen.GetNeighbors()[0];host.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                if host.GetNoImplicit():host.SetNumExplicitHs(host.GetNumExplicitHs()+1)
                graph.RemoveAtom(idx);terms.append((position,'deoxy',''))
            elif mod.kind in {'amino','acetamido',*_HALOGENS}:
                oxygen.SetAtomicNum(7 if mod.kind in {'amino','acetamido'} else _HALOGENS[mod.kind])
                oxygen.SetNumExplicitHs(0);oxygen.SetNoImplicit(mod.kind in _HALOGENS)
                if mod.kind=='acetamido':
                    carbon=_append(graph,Chem,6,owner);oxo=_append(graph,Chem,8,owner);methyl=_append(graph,Chem,6,owner)
                    graph.AddBond(idx,carbon,Chem.BondType.SINGLE);graph.AddBond(carbon,oxo,Chem.BondType.DOUBLE);graph.AddBond(carbon,methyl,Chem.BondType.SINGLE)
                terms.extend(((position,mod.kind,''),(position,'deoxy','')))
            elif mod.kind in {'O-methyl','O-ethyl'}:
                _consume(oxygen);previous=idx
                for _ in range(1 if mod.kind=='O-methyl' else 2):
                    carbon=_append(graph,Chem,6,owner);graph.AddBond(previous,carbon,Chem.BondType.SINGLE);previous=carbon
                terms.append((position,mod.kind[2:],'O'))
            elif mod.kind=='O-acetyl':
                _consume(oxygen);carbon=_append(graph,Chem,6,owner);oxo=_append(graph,Chem,8,owner);methyl=_append(graph,Chem,6,owner)
                graph.AddBond(idx,carbon,Chem.BondType.SINGLE);graph.AddBond(carbon,oxo,Chem.BondType.DOUBLE);graph.AddBond(carbon,methyl,Chem.BondType.SINGLE)
                esters.append(position)
            elif mod.kind=='phosphate':
                from .nucleotides import _phosphate_chain
                p,acidic=_phosphate_chain(graph,Chem,idx,1,'oxygen',owner)
                for atom in graph.GetAtoms():
                    if not atom.HasProp(_OWNER):atom.SetProp(_OWNER,owner)
                phosphates.append((position,p[0],acidic))
        graph.UpdatePropertyCache(strict=False)
    result=graph.GetMol();Chem.SanitizeMol(result);Chem.AssignStereochemistry(result,cleanIt=True,force=True)
    from rdkit.Chem import rdCIPLabeler
    rdCIPLabeler.AssignCIPLabels(result,maxRecursiveIterations=1000000)
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in result.GetAtoms()):
        raise _SugarScope('A conventional-neutral closed-valence sugar derivative was not retained.')
    # Atom indices may have shifted during deoxy deletion. Phosphate new-atom
    # references are recovered from their stable owner before serialization.
    phosphate_sites=[]
    for position,_old_p,_old_acid in phosphates:
        owner=f'mod:{position}:phosphate';atoms=[a for a in result.GetAtoms() if a.GetProp(_OWNER)==owner]
        p=[a for a in atoms if a.GetAtomicNum()==15]
        acids=[a.GetIdx() for a in atoms if a.GetAtomicNum()==8 and a.GetDegree()==1 and a.GetTotalNumHs()==1]
        if len(p)!=1 or len(acids)!=2:raise _SugarScope('The phosphate acid H count is not proved.')
        phosphate_sites.append((position,p[0].GetIdx(),tuple(acids)))
    return result,terms,esters,tuple(phosphate_sites)


def _derived_parent(count,carbonyl,sides,ring,anomer,mods,terms):
    original_positions=[i for i in range(2,count) if i!=carbonyl]
    plain_deoxy={m.position for m in mods if m.kind=='deoxy'}
    remaining=''.join(side for pos,side in zip(original_positions,sides,strict=True) if pos not in plain_deoxy)
    derived_anomer=anomer
    if not plain_deoxy:return _parent_word(count,carbonyl,sides,ring),remaining,derived_anomer,terms
    normalized=sides if sides[-1]=='R' else _mirror(sides);series='D' if sides[-1]=='R' else 'L'
    trivial=({'RLRR':'quinovose','LLRR':'rhamnose','RLLR':'fucose'}.get(normalized)
             if count==6 and carbonyl==1 and plain_deoxy=={6} else None)
    if trivial:
        word=series+'-'+(trivial if ring=='acyclic' else trivial[:-3]+'o'+ring)
        terms=[term for term in terms if term!=(6,'deoxy','')]
        return word,remaining,derived_anomer,terms
    if ring!='acyclic' and (not remaining or len(remaining)>4):
        raise _SugarScope('The modified cyclic reference/configurational-prefix convention is not proved.')
    if remaining and sides[-1]!=remaining[-1] and anomer in {'alpha','beta'}:
        derived_anomer='beta' if anomer=='alpha' else 'alpha'
    chunks=[remaining[i:i+4] for i in range(0,len(remaining),4)];prefixes=[]
    for chunk in reversed(chunks):
        canonical=chunk if chunk[-1]=='R' else _mirror(chunk)
        prefixes.append(('D' if chunk[-1]=='R' else 'L')+'-'+_CONFIG_PREFIX[canonical])
    noun=(_carbohydrate_root(count)+('ose' if ring=='acyclic' else 'o'+ring) if carbonyl==1 else
          _carbohydrate_root(count)+'-2-ulo'+('se' if ring=='acyclic' else ring))
    return ('-'.join(prefixes)+'-' if prefixes else '')+noun,remaining,derived_anomer,terms


def _name(parent,anomer,terms,esters,phosphates,mods,base,count,carbonyl,Chem,*,context='sugar'):
    from .constants import MULT
    if context not in {'sugar','glycoside','glycosyl'}:raise _SugarScope('The sugar rendering context is unproved.')
    if context!='sugar':
        if not parent.endswith('ose'):raise _SugarScope('The cyclic glycosyl parent ending is unavailable.')
        parent=parent[:-1]+('ide' if context=='glycoside' else 'yl')
        terms=list(terms)+[(p,'acetyl','O') for p in esters]+[(p,'phosphono','O') for p,_idx,_acid in phosphates]
        esters=[];phosphates=[]
    groups=defaultdict(list);markers=defaultdict(set)
    for position,word,marker in terms:groups[word,marker].append(position);markers[word].add(marker)
    if any(len(m)>1 for m in markers.values()):raise _SugarScope('Equal-word attachment classes need a proved joint citation.')
    words=[]
    for (word,marker),sites in sorted(groups.items()):
        n=len(sites);mult=MULT[n] if n>1 else ''
        unit=mult+('-O-' if mult else 'O-')+word if marker=='O' else mult+word
        words.append(','.join(str(i) for i in sorted(sites))+'-'+unit)
    prefix='-'.join(words)
    stereo={'alpha':'α-','beta':'β-','unspecified':'','alpha,beta':'α,β-'}[anomer]
    result=(prefix+'-' if prefix else '')+stereo+parent
    if esters:
        # P-102.5.6.1.1: an unchanged ose ending uses the ester class word.
        all_oh={i for i in range(1,count+1) if _atom(base,'O'+str(i)).GetTotalNumHs()==1}
        omit=all(m.kind=='O-acetyl' for m in mods) and set(esters)==all_oh
        locants='' if omit else ','.join(str(i) for i in sorted(esters))+'-'
        result+=' '+locants+(MULT[len(esters)] if len(esters)>1 else '')+'acetate'
    if phosphates:
        sites=','.join(str(p) for p,_idx,_acid in phosphates);n=len(phosphates)
        multiplier='' if n==1 else 'bis' if n==2 else 'tris' if n==3 else MULT[n]+'kis'
        result+=' '+sites+'-'+multiplier+'(dihydrogen phosphate)'
    return result


def modified_carbohydrate_result(spec,Chem):
    count,carbonyl,sides=_parent_facts(spec.parent);_parent_word(count,carbonyl,sides,spec.ring);_validate(spec.modifications,count)
    # Build both declared members separately, always from the original source
    # configuration. No cached or unmodified member stands in for a derivative.
    interpretations=('alpha','beta') if spec.anomer=='alpha,beta' else (spec.anomer,)
    constructions=[]
    for interpretation in interpretations:
        base,closure=_build_sugar(count,carbonyl,sides,spec.ring,interpretation,Chem)
        for atom in base.GetAtoms():
            idx=atom.GetIdx();atom.SetProp(_ROLE,str(idx+1) if idx<count else 'O'+str(idx-count+1))
        graph,terms,esters,phosphates=_modify(base,count,carbonyl,spec.modifications,Chem)
        parent,remaining,anomer,terms=_derived_parent(count,carbonyl,sides,spec.ring,interpretation,spec.modifications,terms)
        name=_name(parent,anomer,terms,esters,phosphates,spec.modifications,base,count,carbonyl,Chem)
        constructions.append((anomer,graph,name,remaining,closure,phosphates))
    if len(constructions)==2:constructions.sort(key=lambda row:row[0])
    anomer,graph,name,remaining,closure,phosphates=constructions[0]
    members=tuple(Chem.MolToSmiles(row[1],canonical=True,isomericSmiles=True) for row in constructions) if len(constructions)>1 else ()
    if members:
        if len(set(members))!=2:raise _SugarScope('The modified anomer mixture has no two distinct members.')
        # Derive the mixture spelling from the source anomer marker, not a
        # replacement of arbitrary alpha text in a chemical prefix.
        parent,_remaining,_anomer,terms=_derived_parent(count,carbonyl,sides,spec.ring,'alpha,beta',spec.modifications,terms)
        name=_name(parent,'alpha,beta',terms,esters,phosphates,spec.modifications,base,count,carbonyl,Chem)
    identity='' if members else Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
    roles=[];groups=defaultdict(list)
    for atom in graph.GetAtoms():
        number=atom.GetIdx()+1;atom.SetAtomMapNum(number);groups[atom.GetProp(_OWNER)].append(number)
        if atom.HasProp(_ROLE):roles.append((atom.GetProp(_ROLE),number))
    roles=tuple(sorted(roles));mapped=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True);check=Chem.MolFromSmiles(mapped)
    if check is None:raise _SugarScope('The mapped derivative cannot be recovered.')
    for atom in check.GetAtoms():atom.SetAtomMapNum(0)
    if Chem.MolToSmiles(check,canonical=True,isomericSmiles=True)!=(members[0] if members else identity):
        raise _SugarScope('Mapped derivative serialization does not conserve its physical configuration.')
    ring_atom=None if closure is None else _atom(graph,'O'+str(closure))
    bindings=tuple(SugarPhosphateBinding(p,_atom(graph,'O'+str(p)).GetIdx()+1,idx+1,tuple(i+1 for i in acids),
        tuple(groups[f'mod:{p}:phosphate'])) for p,idx,acids in phosphates)
    return ModifiedCarbohydrateNamingResult(name=name,generation_status='available',preferred_status='not-assessed',
        route='explicit-modified-carbohydrate',canonical_smiles=identity,mapped_smiles=mapped,parent_atom_maps=roles,
        carbon_count=count,carbonyl_position=carbonyl,fischer_hydroxy_sides=remaining,source_fischer_hydroxy_sides=sides,
        ring=spec.ring,ring_closing_position=closure,ring_oxygen_map=ring_atom.GetIdx()+1 if ring_atom is not None and ring_atom.GetAtomicNum()==8 else None,
        ring_heteroatom_map=None if ring_atom is None else ring_atom.GetIdx()+1,anomer='alpha,beta' if members else anomer,
        declared_anomer=spec.anomer,member_smiles=members,member_anomers=tuple(row[0] for row in constructions) if members else (),
        proportions_unspecified=bool(members),substance_form='anomer-mixture' if members else 'anomer-unspecified' if spec.ring!='acyclic' and anomer=='unspecified' else 'conventional-neutral',
        modifications=tuple(sorted(spec.modifications,key=lambda m:m.position)),component_atom_maps=tuple((key,tuple(value)) for key,value in sorted(groups.items())),
        phosphate_bindings=bindings)
