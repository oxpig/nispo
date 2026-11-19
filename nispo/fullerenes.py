"""Numbered C60 fundamental parent and P-27 cage modification operations."""
from __future__ import annotations

from dataclasses import dataclass,replace
from functools import lru_cache
import json
from pathlib import Path
import re

from .naming_result import NamingResult,SelectionLimitation


@dataclass(frozen=True,slots=True)
class FullereneHomo:
    bond: tuple[int,int]
    count: int = 1


@dataclass(frozen=True,slots=True)
class FullereneReplacement:
    site: int|str
    element: str


@dataclass(frozen=True,slots=True)
class FullereneSpec:
    parent: str = 'C60-Ih'
    saturation: str = 'fullerene'
    nor: tuple[int,...] = ()
    homo: tuple[FullereneHomo,...] = ()
    seco: tuple[tuple[int|str,int|str],...] = ()
    cyclo: tuple[tuple[int|str,int|str],...] = ()
    replacements: tuple[FullereneReplacement,...] = ()
    indicated_hydrogen: int|str|None = None


@dataclass(frozen=True,slots=True)
class FullereneNamingResult(NamingResult):
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    parent_atom_maps: tuple[tuple[str,int],...] = ()
    declared_to_numbered_locants: tuple[tuple[str,str],...] = ()
    deleted_locants: tuple[str,...] = ()
    homo_locants: tuple[str,...] = ()
    indicated_hydrogen_locants: tuple[str,...] = ()
    double_bond_count: int = 0
    hydrogen_count: int = 0
    graph_atom_count: int = 0
    construction_rule: str = 'P-27.2-P-27.5/Fu-3-Fu-5'


class _Input(ValueError):pass
class _Scope(ValueError):pass


_ELEMENTS={'O':(8,2,'oxa'),'S':(16,2,'thia'),'N':(7,3,'aza'),'Si':(14,4,'sila'),
           'Ge':(32,4,'germa'),'B':(5,3,'bora')}
_ORDER=tuple(_ELEMENTS)
_PATTERN=re.compile(r'([1-9]\d*)\(([1-9]\d*)\)([a-z])')


def _site(value):
    if isinstance(value,int) and not isinstance(value,bool) and 1<=value<=60:return value
    if isinstance(value,str):
        match=_PATTERN.fullmatch(value)
        if match:
            low,high=map(int,match.group(1,2));k=ord(match.group(3))-97
            if 1<=low<high<=60:return low,high,k
    raise _Input('A site must be an original C60 integer or an unambiguous compound homo locant, e.g. 1(9)a.')


def _key(site):return (site,0,0,0) if isinstance(site,int) else (site[0],1,site[1],site[2])
def _text(site):return str(site) if isinstance(site,int) else f'{site[0]}({site[1]})'+chr(97+site[2])
def _pair(value):
    if not isinstance(value,tuple) or len(value)!=2:raise _Input('Every bond must be one explicit pair of source sites.')
    result=tuple(sorted((_site(v) for v in value),key=_key))
    if result[0]==result[1]:raise _Input('A fullerene bond needs two distinct atoms.')
    return result


@lru_cache(maxsize=1)
def _resource():
    data=json.loads((Path(__file__).with_name('resources')/'numbered_c60_parent.json').read_text())
    return tuple(tuple(edge) for edge in data['edges'])


@lru_cache(maxsize=2)
def _reference(Chem):
    graph=Chem.RWMol()
    for _ in range(60):graph.AddAtom(Chem.Atom(6))
    for a,b in _resource():graph.AddBond(a-1,b-1,Chem.BondType.SINGLE)
    mol=graph.GetMol();Chem.SanitizeMol(mol)
    return mol.ToBinary()


@lru_cache(maxsize=2)
def _frames(Chem):
    mol=Chem.Mol(_reference(Chem))
    frames=mol.GetSubstructMatches(mol,uniquify=False,maxMatches=121)
    if len(frames)!=120:raise _Scope('The complete C60 systematic-numbering frame family is not proved.')
    return tuple(tuple(i+1 for i in frame) for frame in frames)


def _validate(spec):
    if not isinstance(spec,FullereneSpec):raise _Input('Declare a FullereneSpec construction.')
    if spec.parent!='C60-Ih':raise _Scope('Only the source-verified C60-Ih numbered fundamental parent is available.')
    if not isinstance(spec.saturation,str) or spec.saturation not in {'fullerene','fullerane'}:raise _Input('Declare fullerene or fully saturated fullerane.')
    if any(not isinstance(value,tuple) for value in (spec.nor,spec.homo,spec.seco,spec.cyclo,spec.replacements)):
        raise _Input('Modification collections must be tuples.')
    deleted=tuple(_site(i) for i in spec.nor)
    if any(not isinstance(i,int) for i in deleted) or len(set(deleted))!=len(deleted):raise _Input('Nor deletes distinct original vertices.')
    if len(deleted)>30:raise _Scope('This bounded fullerene construction retains at least half the original C60 vertices.')
    edges=set(_resource());homo=[]
    for h in spec.homo:
        if not isinstance(h,FullereneHomo) or isinstance(h.count,bool) or not isinstance(h.count,int) or not 1<=h.count<=26:
            raise _Input('A homo insertion has a distinct original edge and 1..26 methylene atoms.')
        bond=_pair(h.bond)
        if any(not isinstance(i,int) for i in bond) or bond not in edges:raise _Input('Homo replaces an original cage edge.')
        if any(i in deleted for i in bond):raise _Input('A homo edge has been removed by nor.')
        homo.append((bond,h.count))
    if len({b for b,_n in homo})!=len(homo):raise _Input('Declare one count for each distinct homo edge.')
    inserted={(*bond,k) for bond,n in homo for k in range(n)}
    available=set(range(1,61))-set(deleted)|inserted
    if len(available)>120:raise _Scope('The modified cage exceeds the 120-atom construction limit.')
    seco=tuple(_pair(b) for b in spec.seco);cyclo=tuple(_pair(b) for b in spec.cyclo)
    if len(set(seco))!=len(seco) or len(set(cyclo))!=len(cyclo) or set(seco)&set(cyclo):
        raise _Input('Repeated or cancelling seco/cyclo operations are not a minimal construction.')
    replacements=[]
    for r in spec.replacements:
        if not isinstance(r,FullereneReplacement) or not isinstance(r.element,str) or r.element not in _ELEMENTS:raise _Input('Declare a supported ordinary-valence replacement element.')
        replacements.append((_site(r.site),r.element))
    if len({s for s,_e in replacements})!=len(replacements):raise _Input('A vertex has more than one replacement declaration.')
    indicated=None if spec.indicated_hydrogen is None else _site(spec.indicated_hydrogen)
    mentioned={s for pair in seco+cyclo for s in pair}|{s for s,_e in replacements}|({indicated} if indicated is not None else set())
    if not mentioned<=available:raise _Input('An operation names a deleted or undeclared inserted atom.')
    return deleted,tuple(homo),seco,cyclo,tuple(replacements),indicated


def _numbered(spec,Chem):
    deleted,homo,seco,cyclo,replacements,indicated=_validate(spec);counts=dict(homo);rows=[]
    for frame in _frames(Chem):
        def translate(site):
            if isinstance(site,int):return frame[site-1]
            a,b,k=site;first,last=frame[a-1],frame[b-1]
            return (first,last,k) if first<last else (last,first,counts[a,b]-1-k)
        def pair(b):return tuple(sorted(map(translate,b),key=_key))
        d=tuple(sorted(translate(s) for s in deleted));h=tuple(sorted((pair(b),n) for b,n in homo))
        s=tuple(sorted((pair(b) for b in seco),key=lambda p:tuple(map(_key,p))))
        c=tuple(sorted((pair(b) for b in cyclo),key=lambda p:tuple(map(_key,p))))
        r=tuple(sorted(((translate(s),e) for s,e in replacements),key=lambda v:(_ORDER.index(v[1]),_key(v[0]))))
        ih=None if indicated is None else translate(indicated)
        # P-27.4.5/P-27.5.2: structural operations receive low locants before
        # replacement; indicated H is a separate final source-state choice.
        rank=(d,tuple((b,n) for b,n in h),tuple(tuple(map(_key,p)) for p in s),tuple(tuple(map(_key,p)) for p in c),
              tuple(sorted(_key(site) for site,_element in r)),
              tuple(tuple(_key(site) for site,element in r if element==e) for e in _ORDER),() if ih is None else _key(ih))
        rows.append((rank,d,h,s,c,r,ih,frame))
    return min(rows,key=lambda row:row[0])


def _perfect_matching(vertices,edges,*,state_limit=100000):
    vertices=frozenset(vertices);adj={i:set() for i in vertices}
    for a,b in edges:
        if a in vertices and b in vertices:adj[a].add(b);adj[b].add(a)
    failed=set();states=0
    def search(remaining):
        nonlocal states
        states+=1
        if states>state_limit:raise _Scope('The exact noncumulative pi-matching search exceeded its resource bound.')
        if not remaining:return ()
        if remaining in failed:return None
        first=min(remaining,key=lambda i:(len(adj[i]&remaining),_key(i)))
        for second in sorted(adj[first]&remaining,key=_key):
            found=search(remaining-{first,second})
            if found is not None:return ((first,second),)+found
        failed.add(remaining);return None
    return search(vertices)


def _graph(spec,Chem):
    _rank,deleted,homo,seco,cyclo,replacements,indicated,frame=_numbered(spec,Chem)
    atoms=set(range(1,61))-set(deleted);edges={e for e in _resource() if not set(e)&set(deleted)};inserted=[]
    for (a,b),count in homo:
        edges.remove((a,b));added=tuple((a,b,k) for k in range(count));inserted.extend(added);atoms.update(added)
        chain=(a,)+added+(b,)
        edges.update(tuple(sorted(pair,key=_key)) for pair in zip(chain,chain[1:]))
    for bond in seco:
        if bond not in edges:raise _Input('A seco operation does not remove a currently present bond.')
        edges.remove(bond)
    for bond in cyclo:
        if bond in edges:raise _Input('A cyclo operation must create a new bond.')
        edges.add(bond)
    adjacency={i:set() for i in atoms}
    for a,b in edges:adjacency[a].add(b);adjacency[b].add(a)
    if not atoms:raise _Input('The construction has no cage vertices.')
    seen=set();todo=[next(iter(atoms))]
    while todo:
        i=todo.pop()
        if i in seen:continue
        seen.add(i);todo.extend(adjacency[i]-seen)
    if seen!=atoms:raise _Scope('A source fullerene construction must retain one connected parent.')
    element=dict(replacements);valences={i:_ELEMENTS[element[i]][1] if i in element else 4 for i in atoms}
    if any(not 1<=len(adjacency[i])<=valences[i] for i in atoms):
        raise _Scope('A modified/replaced source vertex has unsupported degree or ordinary valence.')
    matching=();extra=None
    if spec.saturation=='fullerene':
        eligible={i for i in atoms if i not in inserted and len(adjacency[i])<valences[i]}
        if indicated is not None and (indicated not in eligible or len(eligible)%2!=1):
            raise _Scope('The indicated-H declaration is not the required odd unsaturated population site.')
        candidates=(indicated,) if indicated is not None else tuple(sorted(eligible,key=_key)) if len(eligible)%2 else (None,)
        for extra in candidates:
            matching=_perfect_matching(eligible-({extra} if extra is not None else set()),edges)
            if matching is not None:break
        else:raise _Scope('A maximum source noncumulative pi population is not proved.')
    elif indicated is not None:raise _Scope('An additional indicated-H declaration is not used for the fully saturated parent.')
    matching={frozenset(pair) for pair in matching};order=tuple(sorted(atoms,key=_key));indices={i:j for j,i in enumerate(order)};graph=Chem.RWMol()
    hydrogen={i:valences[i]-len(adjacency[i])-sum(i in pair for pair in matching) for i in atoms}
    for site in order:
        atom=Chem.Atom(_ELEMENTS[element[site]][0] if site in element else 6)
        atom.SetNumExplicitHs(hydrogen[site]);atom.SetNoImplicit(True);graph.AddAtom(atom)
    for a,b in sorted(edges,key=lambda pair:tuple(map(_key,pair))):
        graph.AddBond(indices[a],indices[b],Chem.BondType.DOUBLE if frozenset((a,b)) in matching else Chem.BondType.SINGLE)
    graph=graph.GetMol();Chem.SanitizeMol(graph)
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in graph.GetAtoms()):raise _Scope('A neutral closed-valence modified fullerene is not proved.')
    homo_h={i for i in inserted if hydrogen[i] or element.get(i) not in {'O','S'}}
    indicated_sites=tuple(sorted((homo_h|({extra} if extra is not None else set())) if spec.saturation=='fullerene' else (),key=_key))
    if any(hydrogen[i]<1 for i in indicated_sites):raise _Scope('The indicated-H homo site has no actual hydrogen.')
    return graph,order,deleted,homo,seco,cyclo,replacements,indicated_sites,frame,len(matching),sum(hydrogen.values())


def _word(spec,deleted,homo,seco,cyclo,replacements,indicated):
    from .constants import MULT
    def mult(n):
        if n==1:return ''
        if n not in MULT:raise _Scope('The source numerical term is outside validated vocabulary.')
        return MULT[n]
    terms=[]
    if indicated:terms.append(','.join(_text(site)+'H' for site in indicated)+'-')
    for element in _ORDER:
        sites=tuple(sorted((s for s,e in replacements if e==element),key=_key))
        if sites:
            omit=(len(replacements)==1 and not (deleted or homo or seco or cyclo or indicated))
            terms.append(('' if omit else ','.join(map(_text,sites))+'-')+mult(len(sites))+_ELEMENTS[element][2]+'-')
    for bonds,term in ((cyclo,'cyclo'),(seco,'seco')):
        if bonds:terms.append(':'.join(','.join(map(_text,pair)) for pair in bonds)+'-'+mult(len(bonds))+term+'-')
    inserted=tuple((*bond,k) for bond,n in homo for k in range(n))
    if inserted:terms.append(','.join(map(_text,inserted))+'-'+mult(len(inserted))+'homo-')
    if deleted:terms.append(','.join(map(str,deleted))+'-'+mult(len(deleted))+'nor-')
    # Nondetachable terminal prefix joins directly to the parent identifier.
    if terms:terms[-1]=terms[-1].removesuffix('-')
    return ''.join(terms)+'(C60-Ih)[5,6]'+spec.saturation


def fullerene_to_iupac_result(spec:FullereneSpec=FullereneSpec()) -> FullereneNamingResult:
    from .api import Chem
    if Chem is None:return FullereneNamingResult('','unavailable','not-assessed',route='explicit-fullerene')
    try:
        graph,order,deleted,homo,seco,cyclo,replacements,indicated,frame,doubles,hydrogens=_graph(spec,Chem)
        rendered_spec=replace(spec,saturation='fullerane') if doubles==0 else spec
        name=_word(rendered_spec,deleted,homo,seco,cyclo,replacements,indicated)
        identity=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
        for atom in graph.GetAtoms():atom.SetAtomMapNum(atom.GetIdx()+1)
        maps=tuple((_text(site),j+1) for j,site in enumerate(order));declared=tuple((str(j+1),str(i)) for j,i in enumerate(frame))
        # A productive operation sequence does not by itself prove P-52
        # minimality and comparison against every competing skeleton method.
        changed=bool(deleted or homo or seco or cyclo or replacements)
        issues=(SelectionLimitation('fullerene-parent-selection','P-52.2.6',
            'The construction is mapped; minimal combined operations and complete competing-parent preference remain unproved.'),) if changed else ()
        return FullereneNamingResult(name=name,generation_status='available',preferred_status='unresolved' if issues else 'resolved',
            unresolved=issues,route='explicit-fullerene',canonical_smiles=identity,mapped_smiles=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True),
            parent_atom_maps=maps,declared_to_numbered_locants=declared,deleted_locants=tuple(map(str,deleted)),
            homo_locants=tuple(_text((*b,k)) for b,n in homo for k in range(n)),indicated_hydrogen_locants=tuple(map(_text,indicated)),
            double_bond_count=doubles,hydrogen_count=hydrogens,graph_atom_count=graph.GetNumAtoms())
    except _Input as error:return FullereneNamingResult('','invalid-input','not-assessed',
        (SelectionLimitation('fullerene-input','P-27',str(error)),),'explicit-fullerene')
    except (_Scope,ValueError,RuntimeError,OverflowError) as error:return FullereneNamingResult('','unavailable','not-assessed',
        (SelectionLimitation('fullerene-construction','P-27',str(error)),),'explicit-fullerene')


def fullerene_to_iupac(spec:FullereneSpec=FullereneSpec()) -> str:return fullerene_to_iupac_result(spec).name


def fullerene_parent_decision(mol,Chem):
    """Recognize the two unmodified C60 hydrides on ordinary public dispatch."""
    if mol.GetNumAtoms()!=60 or mol.GetNumBonds()!=90:return None
    if mol.GetStereoGroups() or any(a.GetAtomicNum()!=6 or a.GetDegree()!=3 or a.GetIsotope() or a.GetFormalCharge()
            or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) for a in mol.GetAtoms()):return None
    hydrogens=tuple(a.GetTotalNumHs() for a in mol.GetAtoms())
    if set(hydrogens) not in ({0},{1}):return None
    # A necessary distance invariant cheaply rejects e.g. a 60-vertex prism
    # before a symmetric whole-graph substructure search. C60-Ih is vertex
    # transitive; this does not replace the exact topology match below.
    expected=(1,3,6,8,10,10,10,8,3,1)
    neighbors=tuple(tuple(n.GetIdx() for n in a.GetNeighbors()) for a in mol.GetAtoms())
    for start in range(60):
        seen={start};front={start};shells=[]
        while front:
            shells.append(len(front));front={n for i in front for n in neighbors[i]}-seen;seen.update(front)
        if tuple(shells)!=expected:return None
    copy=Chem.RWMol(mol)
    for atom in copy.GetAtoms():
        atom.SetIsAromatic(False);atom.SetNumExplicitHs(1);atom.SetNoImplicit(True);atom.SetAtomMapNum(0)
    for bond in copy.GetBonds():bond.SetIsAromatic(False);bond.SetBondType(Chem.BondType.SINGLE)
    graph=copy.GetMol();Chem.SanitizeMol(graph);reference=Chem.Mol(_reference(Chem))
    match=graph.GetSubstructMatch(reference)
    if len(match)!=60:return None
    from .naming_methods import MethodDecision
    name='(C60-Ih)[5,6]'+('fullerene' if hydrogens[0]==0 else 'fullerane')
    mapping=tuple((original,str(j+1)) for j,original in enumerate(match))
    return MethodDecision(name,'fullerene',frozenset(),frozenset(range(60)),
        'P-27.1.1/P-27.2.3/P-27.3/P-52.2.6',atom_locants=mapping,
        component_roles=(('numbered-C60-parent',tuple(range(60)),mapping),))
