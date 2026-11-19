"""Mapped noncarbon esters, acid ions and uronium functional parents.

Complete original-graph partitions prevent a senior characteristic group in
an organic arm from being hidden by an inorganic functional parent.
"""
from collections import Counter, defaultdict
from itertools import permutations

from .molecule_iteration import iter_atoms, iter_bonds
from .bluebook_lambda_amine_methods import _plain, _out
from .component_mapping import mapped_branch
from .constants import MULT
from .formatting import (format_prefixes, enclose_name_unit,
                         multiply_component, prefix_alphabetization_key)


def _order(mol,a,b):
    bond=mol.GetBondBetweenAtoms(a,b)
    return bond.GetBondTypeAsDouble() if bond is not None else 0


def _side(mol,start,blocked):
    seen=set();todo=[start]
    while todo:
        i=todo.pop()
        if i in seen or i==blocked:continue
        seen.add(i);todo.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors())
    return frozenset(seen)


def _branch(mol,Chem,root,host,allowed=frozenset({6})):
    atoms=_side(mol,root,host)
    if not atoms or host in atoms or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in allowed
            or mol.GetAtomWithIdx(i).GetFormalCharge() or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
            for i in atoms):return None
    if 8 in allowed and any(mol.GetAtomWithIdx(i).GetAtomicNum()==8 and (
            mol.GetAtomWithIdx(i).GetDegree()!=1 or mol.GetAtomWithIdx(i).GetTotalNumHs()!=1
            or any(b.GetBondTypeAsDouble()!=1 for b in mol.GetAtomWithIdx(i).GetBonds())) for i in atoms):return None
    if _order(mol,root,host)!=1:return None
    text,owned=mapped_branch(mol,Chem,root,host)
    if not text or frozenset(owned)!=atoms:return None
    signature=Chem.MolFragmentToSmiles(mol,atomsToUse=sorted(atoms),rootedAtAtom=root,
                                     canonical=True,isomericSmiles=True)
    return dict(text=text,atoms=atoms,root=root,host=host,identity=signature)


def _simple(mol,row):
    atoms=row['atoms']
    return all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 for i in atoms) and (
        row['text'] in {'phenyl','tert-butyl'} or all(
        sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors())<=2 for i in atoms))


def _prefixes(mol,rows):
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    if not rows:return ''
    sequence=mononuclear_prefix_sequence([PrefixUnit(r['text'],'simple' if _simple(mol,r) else 'compound') for r in rows])
    return sequence.text if sequence is not None else ''


def _words(mol,rows,*,locanted=False,numbered=False):
    groups=defaultdict(list)
    for row in rows:groups[(row['identity'],row['text'])].append(row)
    words=[]
    for rs in sorted(groups.values(),key=lambda rs:prefix_alphabetization_key(rs[0]['text'])):
        count=len(rs);text=rs[0]['text']
        word=multiply_component(text,count,nature='simple' if _simple(mol,rs[0]) else 'compound') if count>1 else text
        if not word:return ''
        if locanted:
            locs=[r['element']+(('^'+str(r['site'])) if numbered else '') for r in rs]
            word=','.join(sorted(locs))+'-'+word
        words.append(word)
    return ' '.join(words)


def boron_silicon_esters(mol,Chem):
    if not _plain(mol,Chem,{5,6,8,14,16,34,52}):return None
    centers=[a for a in iter_atoms(mol) if a.GetAtomicNum() in {5,14}]
    if len(centers)!=1:return None
    center=centers[0];element=center.GetAtomicNum();c=center.GetIdx()
    if center.IsInRing() or center.GetTotalValence()!={5:3,14:4}[element]:return None
    direct,esters,acid_sites=[],[],[];used={c}
    for a in center.GetNeighbors():
        i=a.GetIdx()
        if _order(mol,c,i)!=1:return None
        if a.GetAtomicNum()==6:
            row=_branch(mol,Chem,i,c,frozenset({6,8}))
            if row is None or row['atoms']&used:return None
            direct.append(row);used.update(row['atoms'])
        elif a.GetAtomicNum() in {8,16,34,52}:
            others=[n for n in a.GetNeighbors() if n.GetIdx()!=c]
            if a.GetDegree()==1 and a.GetTotalNumHs()==1:
                acid_sites.append(a);used.add(i)
            elif a.GetDegree()==2 and len(others)==1 and others[0].GetAtomicNum()==6:
                row=_branch(mol,Chem,others[0].GetIdx(),i,frozenset({6,8}))
                if row is None or row['atoms']&used:return None
                row.update(element=a.GetSymbol(),bridge=i)
                esters.append(row);used.update(row['atoms']);used.add(i)
            else:return None
        else:return None
    if not esters or used!=set(range(mol.GetNumAtoms())):return None
    fixed=len(direct)+center.GetTotalNumHs()
    modified=Counter(a.GetAtomicNum() for a in acid_sites if a.GetAtomicNum()!=8)
    modified.update(mol.GetAtomWithIdx(r['bridge']).GetAtomicNum() for r in esters if r['element']!='O')
    replacement=''.join((MULT[n] if n>1 else '')+{16:'thio',34:'seleno',52:'telluro'}[e] for e,n in sorted(modified.items()))
    if element==14:
        if fixed or len(esters)+len(acid_sites)!=4:return None
        anion=replacement+'silicate'
    else:
        if fixed not in {0,1,2} or len(esters)+len(acid_sites)!=3-fixed:return None
        anion=('borate','boronate','borinate')[fixed] if not replacement else ('boro','borono','borino')[fixed]+replacement+'ate'
        prefix=_prefixes(mol,direct)
        if direct and not prefix:return None
        anion=prefix+anion
    locanted=len({r['element'] for r in esters}|{a.GetSymbol() for a in acid_sites})>1
    words=_words(mol,esters,locanted=locanted)
    if not words:return None
    hydrogen=('' if not acid_sites else ('hydrogen' if len(acid_sites)==1 else MULT[len(acid_sites)]+'hydrogen'))
    name=' '.join(x for x in (words,hydrogen,anion) if x)
    return _out(mol,name,(c,),'P-67.1.3.2/P-68.1.4.1/P-68.2.4',method='functional-class')


def arsinic_acid_or_anion(mol,Chem):
    if not _plain(mol,Chem,{6,8,33},charges=(0,-1)):return None
    centers=[a for a in iter_atoms(mol) if a.GetAtomicNum()==33]
    if len(centers)!=1:return None
    center=centers[0];c=center.GetIdx()
    if center.IsInRing() or center.GetFormalCharge() or center.GetTotalValence()!=5:return None
    oxo,acid,rows=[],[],[];used={c}
    for a in center.GetNeighbors():
        i=a.GetIdx();bond=_order(mol,c,i)
        if a.GetAtomicNum()==6 and bond==1:
            row=_branch(mol,Chem,i,c)
            if row is None or row['atoms']&used:return None
            rows.append(row);used.update(row['atoms'])
        elif a.GetAtomicNum()==8 and a.GetDegree()==1 and bond==2 and not a.GetFormalCharge():oxo.append(a);used.add(i)
        elif a.GetAtomicNum()==8 and a.GetDegree()==1 and bond==1 and (
            a.GetFormalCharge()==-1 and a.GetTotalNumHs()==0 or a.GetFormalCharge()==0 and a.GetTotalNumHs()==1):acid.append(a);used.add(i)
        else:return None
    if (len(oxo)!=1 or len(acid)!=1 or not rows or len(rows)+center.GetTotalNumHs()!=2
            or used!=set(range(mol.GetNumAtoms()))):return None
    prefix=_prefixes(mol,rows)
    if not prefix:return None
    return _out(mol,prefix+('arsinate' if acid[0].GetFormalCharge() else 'arsinic acid'),(c,),
                'P-67.1.3.1/P-67.1.4.1.1.3',method='functional-class')


def boron_acid_anion(mol,Chem):
    if not _plain(mol,Chem,{5,6,8,16,34,52},charges=(0,-1)):return None
    bs=[a for a in iter_atoms(mol) if a.GetAtomicNum()==5]
    if len(bs)!=1:return None
    b=bs[0];center=b.GetIdx();rows=[];acid=[];used={center}
    if b.IsInRing() or b.GetFormalCharge() or b.GetTotalValence()!=3:return None
    for a in b.GetNeighbors():
        i=a.GetIdx()
        if _order(mol,center,i)!=1:return None
        if a.GetAtomicNum()==6:
            row=_branch(mol,Chem,i,center)
            if row is None or row['atoms']&used:return None
            rows.append(row);used.update(row['atoms'])
        elif a.GetAtomicNum() in {8,16,34,52} and a.GetDegree()==1 and (
            a.GetFormalCharge()==-1 and not a.GetTotalNumHs() or not a.GetFormalCharge() and a.GetTotalNumHs()==1):
            acid.append(a);used.add(i)
        else:return None
    fixed=len(rows)+b.GetTotalNumHs()
    if not rows or fixed not in {1,2} or len(acid)!=3-fixed or not any(a.GetFormalCharge()==-1 for a in acid):return None
    if used!=set(range(mol.GetNumAtoms())):return None
    prefix=_prefixes(mol,rows)
    if not prefix:return None
    modified=Counter(a.GetAtomicNum() for a in acid if a.GetAtomicNum()!=8)
    infix=''.join((MULT[n] if n>1 else '')+{16:'thio',34:'seleno',52:'telluro'}[e] for e,n in sorted(modified.items()))
    parent=('boronate' if fixed==1 else 'borinate') if not infix else ('borono' if fixed==1 else 'borino')+infix+'ate'
    hydrogens=sum(a.GetTotalNumHs() for a in acid)
    hydrogen=('hydrogen ' if hydrogens==1 else MULT[hydrogens]+'hydrogen ' if hydrogens else '')
    return _out(mol,hydrogen+prefix+parent,(center,),'P-68.1.4.1/P-67.1.3.1',method='functional-class')


def acid_salt(mol,Chem):
    fragments=Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True)
    if len(fragments)<2 or sum(a.GetFormalCharge() for a in iter_atoms(mol))!=0:return None
    anions=[];cations=[]
    from .fragment_rules import single_atom_cation_name, common_polyatomic_fragment_name
    for fragment in fragments:
        charge=sum(a.GetFormalCharge() for a in iter_atoms(fragment))
        if charge<0:
            decision=boron_acid_anion(fragment,Chem) or arsinic_acid_or_anion(fragment,Chem)
            if decision is None:return None
            anions.append((decision.name,charge))
        elif charge>0:
            name=single_atom_cation_name(fragment,Chem) or common_polyatomic_fragment_name(fragment,Chem)
            if not name:return None
            cations.append((name,charge))
        else:return None
    if len(anions)!=1 or not cations:return None
    words=[]
    for name,count in sorted(Counter(n for n,q in cations).items(),key=lambda r:prefix_alphabetization_key(r[0])):
        text=multiply_component(name,count) if count>1 else name
        if not text:return None
        words.append(text)
    return _out(mol,' '.join(words+[anions[0][0]]),(), 'P-68.1.4.1/P-67.1.3.1',method='functional-class')


def _silylmethyl(mol,Chem,root,host):
    """Rooted Si-substituted methyl groups, without a generic tms alias."""
    carbon=mol.GetAtomWithIdx(root)
    if carbon.GetAtomicNum()!=6 or carbon.IsInRing():return None
    arms=[n for n in carbon.GetNeighbors() if n.GetIdx()!=host]
    if not arms or any(a.GetAtomicNum()!=14 or _order(mol,root,a.GetIdx())!=1 for a in arms):return None
    used={root};groups=[]
    for si in arms:
        if si.GetTotalValence()!=4 or si.IsInRing():return None
        rows=[];used.add(si.GetIdx())
        for c in si.GetNeighbors():
            if c.GetIdx()==root:continue
            row=_branch(mol,Chem,c.GetIdx(),si.GetIdx())
            if row is None or row['atoms']&used:return None
            rows.append(row);used.update(row['atoms'])
        prefix=_prefixes(mol,rows)
        if rows and not prefix:return None
        owned=_side(mol,si.GetIdx(),root)
        groups.append(dict(text=prefix+'silyl',atoms=owned,
                           identity=Chem.MolFragmentToSmiles(mol,atomsToUse=sorted(owned),rootedAtAtom=si.GetIdx())))
    if len({g['text'] for g in groups})!=1 or len({g['identity'] for g in groups})!=1:return None
    if len(groups)>1 and groups[0]['text']=='silyl':
        text=MULT[len(groups)]+'silyl'
    else:
        text=multiply_component(groups[0]['text'],len(groups),nature='compound') if len(groups)>1 else groups[0]['text']
    if not text:return None
    atoms=_side(mol,root,host)
    if atoms!=used:return None
    return dict(text=text+'methyl',atoms=atoms,root=root,host=host,
                identity=Chem.MolFragmentToSmiles(mol,atomsToUse=sorted(atoms),rootedAtAtom=root))


def stannanol(mol,Chem):
    if not _plain(mol,Chem,{6,8,14,50}):return None
    tins=[a for a in iter_atoms(mol) if a.GetAtomicNum()==50]
    if len(tins)!=1:return None
    tin=tins[0];center=tin.GetIdx();used={center};rows=[];oh=[]
    if tin.IsInRing() or tin.GetTotalValence()!=4:return None
    for a in tin.GetNeighbors():
        i=a.GetIdx()
        if _order(mol,center,i)!=1:return None
        if a.GetAtomicNum()==8 and a.GetDegree()==1 and a.GetTotalNumHs()==1:oh.append(a);used.add(i)
        elif a.GetAtomicNum()==6:
            row=_silylmethyl(mol,Chem,i,center) or _branch(mol,Chem,i,center,frozenset({6,14}))
            if row is None or row['atoms']&used:return None
            if 'tms' in row['text']:return None
            rows.append(row);used.update(row['atoms'])
        else:return None
    if len(oh)!=1 or used!=set(range(mol.GetNumAtoms())):return None
    prefix=_prefixes(mol,rows)
    if rows and not prefix:return None
    return _out(mol,prefix+'stannanol',(center,),'P-68.2.5')


def pentaalkoxyphosphane(mol,Chem):
    if not _plain(mol,Chem,{6,8,15}):return None
    ps=[a for a in iter_atoms(mol) if a.GetAtomicNum()==15]
    if len(ps)!=1:return None
    p=ps[0];center=p.GetIdx()
    if p.IsInRing() or p.GetDegree()!=5 or p.GetTotalValence()!=5:return None
    used={center};prefixes=[]
    for a in p.GetNeighbors():
        if a.GetAtomicNum()!=8 or a.GetDegree()!=2 or _order(mol,center,a.GetIdx())!=1:return None
        roots=[n for n in a.GetNeighbors() if n.GetIdx()!=center]
        if len(roots)!=1 or roots[0].GetAtomicNum()!=6:return None
        carbon=_branch(mol,Chem,roots[0].GetIdx(),a.GetIdx())
        text,atoms=mapped_branch(mol,Chem,a.GetIdx(),center)
        if carbon is None or not text or frozenset(atoms)!=carbon['atoms']|{a.GetIdx()} or used&atoms:return None
        used.update(atoms);prefixes.append(dict(text=text,atoms=carbon['atoms'],identity=carbon['identity']))
    if used!=set(range(mol.GetNumAtoms())):return None
    prefix=_prefixes(mol,prefixes)
    if not prefix:return None
    return _out(mol,prefix+'-λ5-phosphane',(center,),'P-68.3.2.3.2.1')


def uronium(mol,Chem):
    if not _plain(mol,Chem,{6,7,8,16,34,52},charges=(0,1)):return None
    if sum(a.GetFormalCharge() for a in iter_atoms(mol))!=1:return None
    sites=[]
    for c in iter_atoms(mol):
        if c.GetAtomicNum()!=6 or c.IsInRing() or c.GetDegree()!=3:continue
        ns=[a for a in c.GetNeighbors() if a.GetAtomicNum()==7]
        xs=[a for a in c.GetNeighbors() if a.GetAtomicNum() in {8,16,34,52}]
        if len(ns)!=2 or len(xs)!=1:continue
        n_bonds=sorted(_order(mol,c.GetIdx(),n.GetIdx()) for n in ns)
        x_bond=_order(mol,c.GetIdx(),xs[0].GetIdx())
        if (n_bonds==[1,2] and x_bond==1 or n_bonds==[1,1] and x_bond==2):
            sites.append((c,ns,xs[0]))
    if len(sites)!=1:return None
    c,ns,x=sites[0];center=c.GetIdx();core={center,x.GetIdx(),*(n.GetIdx() for n in ns)}
    imino=[n for n in ns if _order(mol,center,n.GetIdx())==2]
    if imino:
        if imino[0].GetFormalCharge()!=1 or any(n.GetFormalCharge() for n in ns if n.GetIdx()!=imino[0].GetIdx()) or x.GetFormalCharge():return None
    elif sum(a.GetFormalCharge() for a in [*ns,x])!=1:
        return None
    arms={};used=set(core)
    for host in [*ns,x]:
        rows=[]
        for a in host.GetNeighbors():
            if a.GetIdx()==center:continue
            if a.GetAtomicNum()!=6 or _order(mol,host.GetIdx(),a.GetIdx())!=1:return None
            row=_branch(mol,Chem,a.GetIdx(),host.GetIdx())
            if row is None or row['atoms']&used:return None
            rows.append(row);used.update(row['atoms'])
        arms[host.GetIdx()]=rows
    if used!=set(range(mol.GetNumAtoms())):return None
    choices=[]
    for arrangement in permutations(ns):
        nlocs={n.GetIdx():'N'+("'"*j) for j,n in enumerate(arrangement)}
        rows=[(nlocs[n.GetIdx()],r['text']) for n in arrangement for r in arms[n.GetIdx()]]
        rows.extend((x.GetSymbol(),r['text']) for r in arms[x.GetIdx()])
        alpha=sorted({r['text'] for rs in arms.values() for r in rs},key=prefix_alphabetization_key)
        key=tuple(tuple(sorted(loc for loc,text in rows if text==prefix)) for prefix in alpha)
        name=format_prefixes(rows,hierarchical_enclosures=True).lstrip('-')+{8:'uronium',16:'thiouronium',34:'selenouronium',52:'tellurouronium'}[x.GetAtomicNum()]
        choices.append((key,name))
    return _out(mol,min(choices)[1],(center,),'P-73.1.2.2')


def identical_polyol_ester(mol,Chem):
    from .extended_inventory import selection_inventory
    inventory=selection_inventory(mol)
    groups=inventory.principal
    if (not inventory.supported or len(groups)<2 or any(g.kind!='ester' for g in groups)
            or mol.GetStereoGroups()):return None
    from .acid_derivative_methods import _acid_component
    allow_mapped_chain = len(groups) == 2 and not mol.GetRingInfo().NumRings()
    parts=[_acid_component(mol,Chem,g.center,g.ligand,allow_mapped_chain=allow_mapped_chain)
           for g in groups]
    if any(p is None for p in parts):return None
    if len({(p.identity,p.name) for p in parts})!=1:return None
    from .polyol_ester_methods import polyol_ester_decision
    return polyol_ester_decision(mol,Chem,groups,frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()>1),
                                frozenset(g.center for g in groups),allow_multiple_stereocenters=True)


def noncarbon_polyacid_ester(mol,Chem):
    if not _plain(mol,Chem,{6,8,15,16}):return None
    centers=[a for a in iter_atoms(mol) if a.GetAtomicNum()==15 and a.GetDegree()>=2
             or a.GetAtomicNum()==16 and a.GetDegree()>=3]
    if len(centers)<2 or len({a.GetAtomicNum() for a in centers})!=1:return None
    central={a.GetIdx() for a in centers};element=centers[0].GetAtomicNum()
    bridges=set();oxo={i:[] for i in central};esters=[];used=set(central)
    direct_edges=set()
    for a in centers:
        i=a.GetIdx()
        for n in a.GetNeighbors():
            j=n.GetIdx();bond=_order(mol,i,j)
            if j in central:
                if element!=15 or bond!=1:return None
                direct_edges.add(frozenset({i,j}));continue
            if n.GetAtomicNum() not in {8,16}:return None
            if n.GetDegree()==1 and bond==2 and n.GetAtomicNum()==8:
                oxo[i].append(j);used.add(j);continue
            if bond!=1 or n.GetDegree()!=2:return None
            others=[k for k in n.GetNeighbors() if k.GetIdx()!=i]
            if len(others)!=1:return None
            other=others[0]
            if other.GetIdx() in central:
                if _order(mol,j,other.GetIdx())!=1:return None
                bridges.add(j);used.add(j)
            elif other.GetAtomicNum()==6:
                row=_branch(mol,Chem,other.GetIdx(),j)
                if row is None:return None
                row.update(element=n.GetSymbol(),bridge=j,center=i)
                esters.append(row)
            else:return None
    # Each terminal ester arm is discovered only from its original centre.
    for row in esters:
        if row['atoms']&used:return None
        used.update(row['atoms']);used.add(row['bridge'])
    if not esters or used!=set(range(mol.GetNumAtoms())):return None
    adjacency={i:set() for i in central}
    for edge in direct_edges:
        i,j=tuple(edge);adjacency[i].add(j);adjacency[j].add(i)
    for b in bridges:
        ends=[n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors()]
        if len(ends)!=2 or any(i not in central for i in ends):return None
        adjacency[ends[0]].add(ends[1]);adjacency[ends[1]].add(ends[0])
    ends=[i for i,ns in adjacency.items() if len(ns)==1]
    if len(ends)!=2 or any(len(ns)>2 for ns in adjacency.values()):return None
    path=[ends[0]]
    while len(path)<len(central):
        nxt=adjacency[path[-1]]-set(path)
        if len(nxt)!=1:return None
        path.append(nxt.pop())
    n=len(path)
    if n not in MULT:return None
    if element==16:
        if direct_edges or any(len(oxo[i])!=2 for i in central) or len(esters)!=2:return None
        parent=MULT[n]+'sulfate'
    else:
        if direct_edges:
            if len(direct_edges)!=1 or n!=2 or bridges:return None
            if all(len(oxo[i])==1 for i in central) and len(esters)==4:parent='hypodiphosphate'
            elif all(not oxo[i] for i in central) and len(esters)==2:parent='hypodiphosphonite'
            else:return None
        else:
            if any(len(oxo[i])!=1 for i in central) or len(esters)!=n+2:return None
            parent=MULT[n]+'phosphate'
    sulfur=sum(mol.GetAtomWithIdx(i).GetAtomicNum()==16 for i in bridges)
    sulfur+=sum(r['element']=='S' for r in esters)
    if sulfur:parent=(MULT[sulfur] if sulfur>1 else '')+'thio'+parent
    # In a hypodiphosphonite there is no oxo oxygen and only the two
    # terminal ester sites can be replaced. Identical organyl groups then
    # occupy every possible site, so no element locants distinguish an
    # alternative. Oxo/bridge replacements still require specificity.
    locanted=bool(sulfur) and not (element==15 and direct_edges and not any(oxo.values())
                                 and len({r['identity'] for r in esters})==1)
    numbered=locanted and any(mol.GetAtomWithIdx(i).GetAtomicNum()==16 for i in bridges)
    if numbered:
        # Native polyacid skeletal centres lie at 1,3,... on the alternating
        # chalcogen-bridged chain. Element-locant numbering follows citation.
        def key(seq):
            mapping={i:2*j+1 for j,i in enumerate(seq)}
            return tuple(tuple(sorted(mapping[r['center']] for r in esters if r['text']==text))
                         for text in sorted({r['text'] for r in esters},key=prefix_alphabetization_key))
        path=min((path,path[::-1]),key=key)
        mapping={i:2*j+1 for j,i in enumerate(path)}
        for row in esters:row['site']=mapping[row['center']]
    words=_words(mol,esters,locanted=locanted,numbered=numbered)
    if not words:return None
    return _out(mol,words+' '+parent,central,'P-67.2.5.2',method='functional-class')


def select(mol,Chem,numbers):
    if numbers[5] or numbers[33]:
        if len(Chem.GetMolFrags(mol))>1:
            result=acid_salt(mol,Chem)
            if result:return result
        if numbers[5]:
            result=boron_acid_anion(mol,Chem) or boron_silicon_esters(mol,Chem)
            if result:return result
        if numbers[33]:
            result=arsinic_acid_or_anion(mol,Chem)
            if result:return result
    if numbers[14] and numbers[8]+numbers[16]+numbers[34]+numbers[52]>=4:
        result=boron_silicon_esters(mol,Chem)
        if result:return result
    if numbers[50] and numbers[8]:
        result=stannanol(mol,Chem)
        if result:return result
    if numbers[15]==1 and numbers[8]>=5:
        result=pentaalkoxyphosphane(mol,Chem)
        if result:return result
    if numbers[15]>=2 or numbers[16]>=2:
        result=noncarbon_polyacid_ester(mol,Chem)
        if result:return result
    if numbers[7]==2 and any(a.GetFormalCharge()==1 for a in iter_atoms(mol)):
        result=uronium(mol,Chem)
        if result:return result
    if numbers[8]>=4 and numbers[6]>=4:
        return identical_polyol_ester(mol,Chem)
    return None
