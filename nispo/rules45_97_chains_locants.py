"""Mapped full-mode parent hydrides and intrinsic-locant constructions.

The chain search compares typed P-44 facts before spelling a parent. Cuts and
neutral acid copies supply vocabulary only; every emitted construction owns
the original graph and preserves the original attachment direction.
"""
from collections import Counter
from dataclasses import replace
from itertools import combinations

from .bluebook_lambda_amine_methods import _plain, _out, _sanitize
from .component_mapping import MappedComponent, decorate_component, mapped_branch
from .constants import MULT, ROOTS
from .formatting import (attach_suffix, enclose_name_unit, format_prefixes,
                         multiply_component, prefix_alphabetization_key)
from .molecule_iteration import iter_atoms, iter_bonds


_HYDRIDES = {7:'azane',8:'oxidane',14:'silane',15:'phosphane',16:'sulfane',
             32:'germane',33:'arsane',34:'selane',50:'stannane',51:'stibane',
             52:'tellane',82:'plumbane',83:'bismuthane'}
_VALENCE = {6:4,7:3,8:2,14:4,15:3,16:2,32:4,33:3,34:2,
            50:4,51:3,52:2,82:4,83:3}
_ORDER = (8,16,34,52,7,15,33,51,83,14,32,50,82)
_TERMS = {8:'oxa',16:'thia',34:'selena',52:'tellura',7:'aza',15:'phospha',
          33:'arsa',51:'stiba',83:'bisma',14:'sila',32:'germa',50:'stanna',82:'plumba'}


def _plain_graph(mol,Chem,elements):
    # Retain the shared annotation/charge/connectedness contract without an
    # unrelated total-size cutoff on otherwise proved parent constructions.
    return _plain(mol,Chem,elements,limit=mol.GetNumHeavyAtoms())


def _order(mol,a,b):
    bond=mol.GetBondBetweenAtoms(a,b)
    return bond.GetBondTypeAsDouble() if bond is not None else 0


def _path(mol,atoms):
    atoms=frozenset(atoms)
    if len(atoms)==1:
        return tuple(atoms)
    adjacency={i:tuple(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                       if n.GetIdx() in atoms) for i in atoms}
    ends=[i for i,ns in adjacency.items() if len(ns)==1]
    if len(ends)!=2 or any(len(ns)>2 for ns in adjacency.values()):
        return ()
    path=[]; previous=None; current=ends[0]
    while current not in path:
        path.append(current)
        following=[i for i in adjacency[current] if i!=previous]
        if not following:
            return tuple(path) if set(path)==set(atoms) else ()
        if len(following)!=1:
            return ()
        previous,current=current,following[0]
    return ()


def _side(mol,root,blocked):
    seen=set();todo=[root]
    while todo:
        i=todo.pop()
        if i==blocked or i in seen:
            continue
        seen.add(i)
        todo.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors())
    return frozenset(seen)


def _carbon_branch(mol,Chem,root,host):
    from .phosphorus_ester_methods import _carbon_component
    atoms=_carbon_component(mol,root,host)
    text,owned=mapped_branch(mol,Chem,root,host)
    return (text,atoms) if atoms and text and atoms==owned else None


def _hydride(sequence):
    if not sequence:
        return None
    if len(set(sequence))==1:
        element=sequence[0]
        if len(sequence)==2 and element==7:
            return 'hydrazine'
        prefix='' if len(sequence)==1 else MULT.get(len(sequence))
        return prefix+_HYDRIDES[element] if prefix is not None and element in _HYDRIDES else None
    # P-21.2.3.1 a[ba]n chains: only the more junior terminal element is
    # multiplied; the interior count is intrinsic to the alternating chain.
    if (len(sequence)%2==1 and sequence==sequence[::-1] and len(set(sequence))==2
            and all(z==sequence[i%2] for i,z in enumerate(sequence))
            and 7 not in sequence and _ORDER.index(sequence[0])>_ORDER.index(sequence[1])):
        multiplier=MULT.get((len(sequence)+1)//2)
        if multiplier is None:
            return None
        first,second=_TERMS[sequence[0]],_TERMS[sequence[1]]
        if second[0] in 'aeiou':
            first=first[:-1]
        return multiplier+first+second[:-1]+'ane'
    return None


def homogeneous_parent(mol,Chem):
    """P-21.2.2: count the complete homogeneous skeletal path."""
    if not _plain_graph(mol,Chem,{6,*_HYDRIDES}):
        return None
    hetero={a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()!=6}
    if len(hetero)<2 or len({mol.GetAtomWithIdx(i).GetAtomicNum() for i in hetero})!=1:
        return None
    element=mol.GetAtomWithIdx(next(iter(hetero))).GetAtomicNum()
    # Dinuclear O/S links have peroxide/ether functional-class rules. Longer
    # homogeneous chains and the other parent-hydride families use P-21.2.2.
    if element in {8,16,34,52} and len(hetero)<3:
        return None
    path=_path(mol,hetero)
    if (not path or any(mol.GetAtomWithIdx(i).IsInRing() or
            mol.GetAtomWithIdx(i).GetTotalValence()!=_VALENCE[element] for i in path)
            or any(_order(mol,a,b)!=1 for a,b in zip(path,path[1:]))):
        return None
    parent=_hydride(tuple(element for _i in path))
    if not parent:
        return None
    choices=[]
    for seq in (path,path[::-1]):
        used=set(hetero); prefixes=[]
        for j,i in enumerate(seq,1):
            for a in mol.GetAtomWithIdx(i).GetNeighbors():
                if a.GetIdx() in hetero:
                    continue
                row=_carbon_branch(mol,Chem,a.GetIdx(),i)
                if row is None or used&row[1] or _order(mol,i,a.GetIdx())!=1:
                    break
                prefixes.append((str(j),row[0]));used.update(row[1])
            else:
                continue
            break
        else:
            if used!=set(range(mol.GetNumAtoms())):
                continue
            # P-14.3.4.2(b) omits the single locant on a dinuclear homologue.
            omit=(len(path)==2 and len(prefixes)==1) or element in {8,16,34,52}
            key=(tuple(sorted(int(loc) for loc,_text in prefixes)),
                 tuple((prefix_alphabetization_key(text),int(loc)) for loc,text in
                       sorted(prefixes,key=lambda r:(prefix_alphabetization_key(r[1]),int(r[0])))))
            if omit and element in {8,16,34,52} and prefixes:
                from .organyl_prefix_components import organyl_prefix_nature
                from .prefix_sequences import PrefixUnit,mononuclear_prefix_sequence
                typed=[]
                for loc,text in prefixes:
                    host=seq[int(loc)-1]
                    branch=next(a for a in mol.GetAtomWithIdx(host).GetNeighbors() if a.GetIdx() not in hetero)
                    scope=_side(mol,branch.GetIdx(),host)
                    nature=organyl_prefix_nature(mol,branch.GetIdx(),scope,text)
                    if nature is None:
                        return None
                    typed.append(PrefixUnit(text,nature))
                sequence=mononuclear_prefix_sequence(typed)
                if sequence is None:
                    return None
                text=sequence.text
            else:
                text=format_prefixes(prefixes,omit_locants=omit,hierarchical_enclosures=True)
            choices.append((key,text+parent,tuple((i,str(j)) for j,i in enumerate(seq,1))))
    if not choices:
        return None
    best=min(row[0] for row in choices); winners=[row for row in choices if row[0]==best]
    if len({row[1] for row in winners})!=1:
        return None
    decision=_out(mol,winners[0][1],(),'P-21.2.2/P-44.3/P-14.3.4.2')
    return replace(decision,atom_locants=winners[0][2])


def named_chalcogen_parent(mol,Chem):
    """A contiguous >=3 chalcogen parent is not a replacement heterounit.

    P-68.4.1.1/P-51.4.1.1 keep the homogeneous hydride even when its two
    carbon-rooted organyl substituents themselves contain ethers/sulfides.
    Principal OH, carbonyl, nitrogen or nonchalcogen chemistry delegates.
    """
    if (not _plain_graph(mol,Chem,{6,8,16,34,52}) or any(a.IsInRing() and a.GetAtomicNum()!=6 for a in iter_atoms(mol))
            or any(b.GetBondTypeAsDouble()!=1 and not (b.GetBeginAtom().GetAtomicNum()==b.GetEndAtom().GetAtomicNum()==6
                and (b.GetIsAromatic() or b.GetBondTypeAsDouble() in {2,3})) for b in iter_bonds(mol))):
        return None
    hetero={a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()!=6}
    visited=set();runs=[]
    for seed in hetero:
        if seed in visited:
            continue
        z=mol.GetAtomWithIdx(seed).GetAtomicNum();found=set();todo=[seed]
        while todo:
            i=todo.pop()
            if i in found:
                continue
            found.add(i)
            todo.extend(a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetAtomicNum()==z)
        visited.update(found)
        if len(found)>=3:
            runs.append(found)
    if len(runs)!=1:
        return None
    core=runs[0];path=_path(mol,core)
    if not path or any(mol.GetAtomWithIdx(i).IsInRing() or mol.GetAtomWithIdx(i).GetTotalValence()!=2 for i in core):
        return None
    if any(a.GetIdx() not in core and a.GetAtomicNum()!=6 and a.GetTotalNumHs() for a in iter_atoms(mol)):
        return None
    element=mol.GetAtomWithIdx(path[0]).GetAtomicNum()
    parent=_hydride(tuple(element for _i in path))
    if not parent:
        return None
    used=set(core);units=[]
    from .organyl_prefix_components import organyl_prefix_nature
    from .prefix_sequences import PrefixUnit,mononuclear_prefix_sequence
    for i in path:
        for a in mol.GetAtomWithIdx(i).GetNeighbors():
            if a.GetIdx() in core:
                continue
            if a.GetAtomicNum()!=6 or i not in {path[0],path[-1]} or _order(mol,i,a.GetIdx())!=1:
                return None
            scope=_side(mol,a.GetIdx(),i);text,owned=mapped_branch(mol,Chem,a.GetIdx(),i)
            if not text or owned!=scope or scope&used:
                return None
            nature=organyl_prefix_nature(mol,a.GetIdx(),scope,text)
            units.append(PrefixUnit(text,nature or 'compound'));used.update(scope)
    if used!=set(range(mol.GetNumAtoms())) or not units:
        return None
    sequence=mononuclear_prefix_sequence(units)
    if sequence is None:
        return None
    return _out(mol,sequence.text+parent,(),'P-68.4.1.1/P-21.2.2/P-51.4.1.1/P-14.3.4.4')


def silicon_imine(mol,Chem):
    """Selected silicon hydride carrying principal Si=N groups."""
    if not _plain_graph(mol,Chem,{6,7,14}):
        return None
    silicon={a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()==14}
    ns=[a for a in iter_atoms(mol) if a.GetAtomicNum()==7]
    if not silicon or not ns:
        return None
    path=_path(mol,silicon)
    if (not path or any(mol.GetAtomWithIdx(i).IsInRing() or mol.GetAtomWithIdx(i).GetTotalValence()!=4 for i in path)
            or any(_order(mol,a,b)!=1 for a,b in zip(path,path[1:]))):
        return None
    sites=[]
    for n in ns:
        attached=[a.GetIdx() for a in n.GetNeighbors() if a.GetIdx() in silicon and _order(mol,n.GetIdx(),a.GetIdx())==2]
        if len(attached)!=1 or n.IsInRing() or n.GetTotalValence()!=3:
            return None
        sites.append((attached[0],n.GetIdx()))
    if len({host for host,_n in sites})!=len(sites):
        return None
    imine_multiplier='' if len(sites)==1 else MULT.get(len(sites))
    if imine_multiplier is None:
        return None
    parent=_hydride(tuple(14 for _i in path))
    if parent is None:
        return None
    choices=[]
    for seq in (path,path[::-1]):
        position={i:j for j,i in enumerate(seq,1)};used=set(silicon)|{n for _host,n in sites};prefixes=[]
        for host in sorted(used):
            for a in mol.GetAtomWithIdx(host).GetNeighbors():
                if a.GetIdx() in used:
                    continue
                row=_carbon_branch(mol,Chem,a.GetIdx(),host)
                if row is None or used&row[1] or _order(mol,host,a.GetIdx())!=1:
                    break
                if host in silicon:
                    loc=str(position[host])
                else:
                    si=next(h for h,n in sites if n==host)
                    loc='N' if len(sites)==1 else 'N^'+str(position[si])
                prefixes.append((loc,row[0]));used.update(row[1])
            else:
                continue
            break
        else:
            if used!=set(range(mol.GetNumAtoms())):
                continue
            imine_locs=tuple(sorted(position[host] for host,_n in sites))
            ending=parent[:-1]+'imine' if len(seq)==1 else attach_suffix(parent,imine_multiplier+'imine',[str(i) for i in imine_locs])
            mapping={i:str(position[i]) for i in seq}
            mapping.update({n:'N' if len(sites)==1 else 'N^'+str(position[host]) for host,n in sites})
            from .parent_selection import locant_key
            def site_key(loc):
                # Display superscripts are the typed N-site index understood
                # by the shared locant comparator in its canonical N1 form.
                return locant_key(loc.replace('N^','N'))
            key=(imine_locs,tuple(sorted(site_key(loc) for loc,_text in prefixes)),
                 tuple((prefix_alphabetization_key(text),site_key(loc)) for loc,text in sorted(prefixes,key=lambda r:(prefix_alphabetization_key(r[1]),site_key(r[0])))))
            choices.append((key,format_prefixes(prefixes,hierarchical_enclosures=True)+ending,tuple(mapping.items())))
    if not choices:
        return None
    best=min(c[0] for c in choices); winners=[c for c in choices if c[0]==best]
    if len({row[1] for row in winners})!=1:
        return None
    decision=_out(mol,winners[0][1],(host for host,_n in sites),'P-62.3.1.3/P-44')
    return replace(decision,atom_locants=winners[0][2])


def intrinsic_carbonyl_locants(mol,Chem):
    """P-14.3.4.4/.5: one/two fully occupied carbonyl positions."""
    if not _plain_graph(mol,Chem,{6,7,8,9,17,35,53}):
        return None
    oxygens=[a for a in iter_atoms(mol) if a.GetAtomicNum()==8 and a.GetDegree()==1
             and a.GetNeighbors()[0].GetAtomicNum()==6 and _order(mol,a.GetIdx(),a.GetNeighbors()[0].GetIdx())==2]
    if len(oxygens) not in {1,2}:
        return None
    centers=[]
    for oxygen in oxygens:
        if oxygen.GetDegree()!=1:
            return None
        c=oxygen.GetNeighbors()[0]
        if c.GetAtomicNum()!=6 or c.IsInRing() or c.GetDegree()!=3 or _order(mol,c.GetIdx(),oxygen.GetIdx())!=2:
            return None
        centers.append(c.GetIdx())
    if len(set(centers))!=len(centers) or len(centers)==2 and _order(mol,*centers)!=1:
        return None
    from .extended_inventory import selection_inventory
    from .group_priority import functional_group_priority_facts
    inventory=selection_inventory(mol)
    if not inventory.supported:
        return None
    # A completely owned cyanophenyl arm is still subordinate to a nitrile
    # suffix. Inspect original characteristic sites rather than mistaking
    # complete branch rendering for proof of ketone seniority. The exact
    # diazenyl pseudo-ketone core is interpreted below, so its inventory
    # amide projection does not veto the independently proved construction.
    for group in inventory.groups:
        if group.center in centers:
            continue
        facts=functional_group_priority_facts(group,mol)
        if facts is None or facts.class_rank is None or facts.class_rank<16:
            return None
    core=set(centers)|{a.GetIdx() for a in oxygens};used=set(core);arms=[]
    for center in centers:
        for a in mol.GetAtomWithIdx(center).GetNeighbors():
            root=a.GetIdx()
            if root in core:
                continue
            if _order(mol,center,root)!=1:
                return None
            if a.GetAtomicNum()==6:
                if not a.IsInRing():
                    return None  # An alkyl arm lengthens the ketone parent.
            elif a.GetAtomicNum()==7:
                distal=[n for n in a.GetNeighbors() if n.GetIdx()!=center]
                if (len(distal)!=1 or distal[0].GetAtomicNum()!=7 or _order(mol,root,distal[0].GetIdx())!=2
                        or distal[0].GetDegree()!=2):
                    return None  # Amides and ureas have their own senior class.
            else:
                return None
            scope=_side(mol,root,center)
            text,owned=mapped_branch(mol,Chem,root,center)
            if not text or owned!=scope or scope&used:
                return None
            from .naming_methods import _fragment
            fragment,original=_fragment(mol,Chem,scope)
            if fragment is None:
                return None
            fragment.GetAtomWithIdx(original.index(root)).SetAtomMapNum(1)
            identity=Chem.MolToSmiles(fragment,canonical=True,isomericSmiles=True)
            from .organyl_prefix_components import organyl_prefix_nature
            # Ring atoms alone are insufficient: separately linked aryl
            # rings and substituted heteroaryl units are compound prefixes.
            simple=organyl_prefix_nature(mol,root,scope,text)=='simple'
            arms.append((text,scope,identity,simple));used.update(scope)
    if len(arms)!=2 or used!=set(range(mol.GetNumAtoms())):
        return None
    if arms[0][2]==arms[1][2]:
        if arms[0][0]!=arms[1][0]:
            return None
        name=arms[0][0]
        simple=arms[0][3] and arms[1][3]
        prefix=multiply_component(name,2,nature='simple' if simple else 'compound')
        if not prefix and simple:
            prefix='di'+enclose_name_unit(name)
        if not prefix:
            return None
    else:
        prefix=format_prefixes([('',a[0]) for a in arms],omit_locants=True,
                               hierarchical_enclosures=True,separate_unlocanted=True)
    suffix='methanone' if len(centers)==1 else 'ethanedione'
    return _out(mol,prefix+suffix,centers,'P-14.3.4.4/P-14.3.4.5/P-16.3.4/P-64.2.2')


def simple_aryl_branch(mol,root,blocked,atoms):
    """Original-graph proof of an unmodified aromatic parent-hydride prefix.

    Methyl, halo, amino, hydro, isotope and charge modifications do not pass
    this proof. A fused or Hantzsch–Widman aryl hydride need not be all-carbon.
    Prefix spelling supplies no eligibility or identity evidence.
    """
    atoms=frozenset(atoms)
    if not atoms or root not in atoms or blocked in atoms:
        return False
    if any(not mol.GetAtomWithIdx(i).IsInRing() or not mol.GetAtomWithIdx(i).GetIsAromatic()
           or mol.GetAtomWithIdx(i).GetFormalCharge() or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
           or mol.GetAtomWithIdx(i).GetIsotope() or int(mol.GetAtomWithIdx(i).GetChiralTag())
           or mol.GetAtomWithIdx(i).HasQuery() for i in atoms):
        return False
    boundary=[(i,n.GetIdx()) for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()
              if n.GetAtomicNum()>1 and n.GetIdx() not in atoms]
    if boundary!=[(root,blocked)] or _order(mol,root,blocked)!=1:
        return False
    if any(not bond.GetIsAromatic() for bond in iter_bonds(mol)
           if bond.GetBeginAtomIdx() in atoms and bond.GetEndAtomIdx() in atoms):
        return False  # A separately linked ring assembly is a compound unit.
    seen=set();todo=[root]
    while todo:
        i=todo.pop()
        if i in seen:
            continue
        seen.add(i)
        todo.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in atoms)
    return seen==set(atoms)


def branch_prefix(mol,Chem,root,blocked):
    """P-67.1.4.1.3: original X–P order in phosphono-X prefixes."""
    atom=mol.GetAtomWithIdx(root)
    if (atom.GetAtomicNum() not in {7,8,16,34,52} or atom.IsInRing()
            or atom.GetFormalCharge() or atom.GetNumRadicalElectrons() or atom.GetIsotope()
            or int(atom.GetChiralTag()) or atom.GetDegree()!=2 or _order(mol,root,blocked)!=1):
        return None
    others=[a for a in atom.GetNeighbors() if a.GetIdx()!=blocked]
    if len(others)!=1 or others[0].GetAtomicNum()!=15 or _order(mol,root,others[0].GetIdx())!=1:
        return None
    if atom.GetTotalValence()!={7:3,8:2,16:2,34:2,52:2}[atom.GetAtomicNum()]:
        return None
    from .phosphorus_acyl_prefixes import phosphorus_acyl_prefix
    proof=phosphorus_acyl_prefix(mol,Chem,others[0].GetIdx(),root)
    if proof is None or root in proof.atoms or blocked in proof.atoms:
        return None
    tail={7:'amino',8:'oxy',16:'sulfanyl',34:'selanyl',52:'tellanyl'}[atom.GetAtomicNum()]
    text=proof.name+tail if proof.name=='phosphono' else enclose_name_unit(proof.name)+tail
    return text,proof.atoms|{root}


def phosphono_linked_acid(mol,Chem):
    """Keep an ordinary carboxylic acid senior to a linked phosphorus acid."""
    if not _plain_graph(mol,Chem,{6,7,8,15,16,34,52}):
        return None
    p=[a for a in iter_atoms(mol) if a.GetAtomicNum()==15]
    if len(p)!=1:
        return None
    proposals=[]
    for root in p[0].GetNeighbors():
        for host in root.GetNeighbors():
            if host.GetIdx()==p[0].GetIdx() or host.GetAtomicNum()!=6:
                continue
            proof=branch_prefix(mol,Chem,root.GetIdx(),host.GetIdx())
            if proof is not None:
                proposals.append((root.GetIdx(),proof))
    if len(proposals)!=1:
        return None
    root,(prefix,owned)=proposals[0]
    removed=set(owned)-{root};kept=tuple(i for i in range(mol.GetNumAtoms()) if i not in removed)
    rw=Chem.RWMol(mol)
    marker=rw.GetAtomWithIdx(root);marker.SetAtomicNum(9);marker.SetNumExplicitHs(0);marker.SetNoImplicit(True)
    for idx in sorted(removed,reverse=True):
        rw.RemoveAtom(idx)
    proxy=_sanitize(rw,Chem)
    if proxy is None:
        return None
    from .extended_inventory import selection_inventory
    inv=selection_inventory(proxy)
    if not inv.supported or not inv.principal or any(g.kind!='acid' for g in inv.principal):
        return None
    from .complete_functional_parent import functional_parent_candidates
    records=functional_parent_candidates(proxy,Chem,include_single=True,mapped_ring_prefixes=True)
    from .parent_selection import parent_seniority_frontier,parent_frontier,locant_key
    records=parent_seniority_frontier(list(records),lambda r:r.parent_selection)
    names=[]
    for record in records:
        component=record.construction
        if component is None or len(component.prefix_branch_ownership)!=len(component.prefixes):
            return None
        parent_atoms=frozenset(kept[i] for i in component.atoms)
        embedding=frozenset(range(mol.GetNumAtoms()))-parent_atoms
        original_locants=tuple((kept[i],loc) for i,loc in component.atom_locants)
        labels=dict(original_locants);prefixes=[];source_branches=[];claimed=set();compound=[]
        for old_root,old_host,_old_scope,_old_text in component.prefix_branch_ownership:
            branch_root,host=kept[old_root],kept[old_host]
            text,scope=mapped_branch(mol,Chem,branch_root,host)
            if not text or not scope or scope&parent_atoms or scope&claimed or host not in labels:
                return None
            prefixes.append((labels[host],text));source_branches.append((branch_root,host,scope,text));claimed.update(scope)
            if scope&set(owned):
                if not set(owned)<=scope:
                    return None
                compound.append(text)
        if claimed!=set(embedding) or len(compound)!=1:
            return None
        prefixes=tuple(prefixes)
        changed=replace(component,atom_locants=original_locants,
            atoms=parent_atoms,embedded_atoms=embedding,prefixes=prefixes,
            prefix_branch_ownership=tuple(source_branches),functionalized_carbon_prefix_proofs=(),
            retained_acid_locant_evidence=None,systematic_parent=True,
            compound_prefix_names=tuple(compound))
        name=''
        if component.parent=='acetic acid':
            from .retained_acid_rules import retained_acid_locant_evidence
            evidence=retained_acid_locant_evidence(mol,Chem,parent_class='acetic acid',principal_kind='acid',
                parent_atoms=parent_atoms,atom_locants=original_locants,prefixes=prefixes)
            if evidence is not None:
                from .prefix_sequences import PrefixUnit,mononuclear_prefix_sequence
                typed=[]
                for loc,text in prefixes:
                    if text in compound:
                        typed.append(PrefixUnit(text,'compound'))
                        continue
                    # This omission path only admits separately proved
                    # terminal amino/hydroxy or ordinary carbon branches.
                    host=next(i for i,label in original_locants if label==loc and mol.GetAtomWithIdx(i).GetAtomicNum()==6)
                    matched=False
                    for neighbor in mol.GetAtomWithIdx(host).GetNeighbors():
                        if neighbor.GetIdx() in parent_atoms:
                            continue
                        rendered,scope=mapped_branch(mol,Chem,neighbor.GetIdx(),host)
                        if rendered!=text:
                            continue
                        simple=(len(scope)==1 and neighbor.GetAtomicNum() in {7,8}
                                or all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 for i in scope))
                        typed.append(PrefixUnit(text,'simple' if simple else 'compound'));matched=True;break
                    if not matched:
                        break
                else:
                    sequence=mononuclear_prefix_sequence(typed)
                    if sequence is not None:
                        name=sequence.text+'acetic acid'
        if not name:
            name=decorate_component(mol,Chem,changed)
        if name:
            facts=record.parent_selection
            if facts is None:
                continue
            prefix_locs=tuple(sorted(locant_key(loc) for loc,_text in prefixes))
            citation=tuple(locant_key(loc) for loc,text in sorted(prefixes,key=lambda r:(prefix_alphabetization_key(r[1]),locant_key(r[0]))))
            facts=replace(facts,prefix_count=len(prefixes),prefix_locants=prefix_locs,prefix_citation_locants=citation,
                          alphanumerical_key=(prefix_alphabetization_key(name),(),()))
            names.append((name,tuple(kept[i] for i in record.principal_centers),facts))
    names=parent_frontier(names,lambda r:r[2])
    if not names or len({row[0] for row in names})!=1:
        return None
    return _out(mol,names[0][0],names[0][1],'P-67.1.4.1.3/P-42/P-44')


def heterohydride_carboxylic_acid(mol,Chem):
    if not _plain_graph(mol,Chem,{6,*_HYDRIDES}):
        return None
    acids=[]
    for c in iter_atoms(mol):
        if c.GetAtomicNum()!=6 or c.IsInRing() or c.GetDegree()!=3:
            continue
        oxo=[a for a in c.GetNeighbors() if a.GetAtomicNum()==8 and a.GetDegree()==1 and _order(mol,c.GetIdx(),a.GetIdx())==2]
        hydroxy=[a for a in c.GetNeighbors() if a.GetAtomicNum()==8 and a.GetDegree()==1 and a.GetTotalNumHs()==1 and _order(mol,c.GetIdx(),a.GetIdx())==1]
        roots=[a for a in c.GetNeighbors() if a.GetAtomicNum() not in {6,8,16,34,52} and _order(mol,c.GetIdx(),a.GetIdx())==1]
        if len(oxo)==len(hydroxy)==len(roots)==1:
            acids.append((roots[0].GetIdx(),c.GetIdx(),oxo[0].GetIdx(),hydroxy[0].GetIdx()))
    if not acids or len(acids)>1 and len(acids) not in MULT:
        return None
    acid_atoms={i for _root,c,o,h in acids for i in (c,o,h)}
    hetero={a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()!=6 and a.GetIdx() not in acid_atoms}
    path=_path(mol,hetero)
    if (not path or any(mol.GetAtomWithIdx(i).IsInRing() or
        mol.GetAtomWithIdx(i).GetTotalValence()!=_VALENCE.get(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in path)
        or any(_order(mol,a,b)!=1 for a,b in zip(path,path[1:]))):
        return None
    parent=_hydride(tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in path))
    if not parent:
        return None
    if len(path)==1 and mol.GetAtomWithIdx(path[0]).GetAtomicNum()==7 and len(acids)==1:
        return None  # P-65.2.1.1: carbamic acid, including N substitution.
    choices=[]
    for seq in (path,path[::-1]):
        mapping={i:j for j,i in enumerate(seq,1)};used=set(hetero)|acid_atoms;prefixes=[];prefix_units=[]
        for i in seq:
            for a in mol.GetAtomWithIdx(i).GetNeighbors():
                if a.GetIdx() in used:
                    continue
                row=_carbon_branch(mol,Chem,a.GetIdx(),i)
                if row is None or row[1]&used or _order(mol,i,a.GetIdx())!=1:
                    break
                prefixes.append((str(mapping[i]),row[0]));used.update(row[1])
                if len(seq)==1:
                    from .organyl_prefix_components import organyl_prefix_nature
                    from .prefix_sequences import PrefixUnit
                    nature=organyl_prefix_nature(mol,a.GetIdx(),row[1],row[0])
                    if nature is None:
                        break
                    prefix_units.append(PrefixUnit(row[0],nature))
            else:
                continue
            break
        else:
            if used!=set(range(mol.GetNumAtoms())):
                continue
            locants=tuple(sorted(mapping[root] for root,_c,_o,_h in acids))
            # P-14.3.4.2(a): every suffix and substituent attachment is at
            # the sole skeletal atom of a mononuclear parent. This proof
            # does not depend on acid or organyl multiplicity, or on element.
            mononuclear=len(seq)==1
            omit=mononuclear
            if not mononuclear and len(acids)==1 and not prefixes:
                from .naming_methods import _fragment
                fragment,original=_fragment(mol,Chem,hetero)
                if fragment is not None:
                    classes=tuple(Chem.CanonicalRankAtoms(fragment,breakTies=False))
                    substitutable=[original.index(i) for i in hetero if _VALENCE[mol.GetAtomWithIdx(i).GetAtomicNum()]>
                                   sum(_order(mol,i,n.GetIdx()) for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in hetero)]
                    omit=bool(substitutable) and len({classes[i] for i in substitutable})==1
            suffix=parent+('' if omit else '-'+','.join(map(str,locants))+'-')+(MULT[len(acids)] if len(acids)>1 else '')+'carboxylic acid'
            key=(locants,tuple(sorted(int(loc) for loc,_text in prefixes)),
                 tuple((prefix_alphabetization_key(text),int(loc)) for loc,text in sorted(prefixes,key=lambda r:(prefix_alphabetization_key(r[1]),int(r[0])))))
            if mononuclear and prefix_units:
                from .prefix_sequences import mononuclear_prefix_sequence
                sequence=mononuclear_prefix_sequence(prefix_units)
                if sequence is None:
                    continue
                prefix_text=sequence.text
            else:
                prefix_text=format_prefixes(prefixes,hierarchical_enclosures=True)
            choices.append((key,prefix_text+suffix))
    if not choices:
        return None
    best=min(key for key,_name in choices);names={name for key,name in choices if key==best}
    return _out(mol,names.pop(),(c for _root,c,_o,_h in acids),'P-65.1.2.2.2/P-14.3.4.2/P-14.3.4.3/P-44') if len(names)==1 else None


def ring_polycarboxylate(mol,Chem):
    """Acid-derived ring suffix and a separately balanced cation citation."""
    if mol.GetStereoGroups() or any(a.GetIsotope() or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) or a.HasQuery()
            for a in iter_atoms(mol)) or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in iter_bonds(mol)):
        return None
    fragments=Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True)
    anions=[f for f in fragments if sum(a.GetFormalCharge() for a in f.GetAtoms())<0]
    if len(anions)!=1:
        return None
    anion=anions[0]
    if not anion.GetRingInfo().NumRings() or any(a.GetAtomicNum() not in {6,7,8,9,16,17,35,53} for a in anion.GetAtoms()):
        return None
    sites=[];used_charges=set()
    for c in anion.GetAtoms():
        if c.GetAtomicNum()!=6 or c.IsInRing() or c.GetDegree()!=3:
            continue
        oxo=[a for a in c.GetNeighbors() if a.GetAtomicNum()==8 and a.GetDegree()==1 and _order(anion,c.GetIdx(),a.GetIdx())==2 and a.GetFormalCharge()==0]
        minus=[a for a in c.GetNeighbors() if a.GetAtomicNum()==8 and a.GetDegree()==1 and _order(anion,c.GetIdx(),a.GetIdx())==1 and a.GetFormalCharge()==-1]
        root=[a for a in c.GetNeighbors() if a.IsInRing() and _order(anion,c.GetIdx(),a.GetIdx())==1]
        if len(oxo)==len(minus)==len(root)==1:
            sites.append((c.GetIdx(),minus[0].GetIdx()));used_charges.add(minus[0].GetIdx())
    if len(sites)<2 or used_charges!={a.GetIdx() for a in anion.GetAtoms() if a.GetFormalCharge()}:
        return None
    rw=Chem.RWMol(anion)
    for _c,i in sites:
        atom=rw.GetAtomWithIdx(i);atom.SetFormalCharge(0);atom.SetNumExplicitHs(1);atom.SetNoImplicit(True)
    acid=_sanitize(rw,Chem)
    if acid is None:
        return None
    from .extended_inventory import selection_inventory
    acid_inventory=selection_inventory(acid)
    if (not acid_inventory.supported or not acid_inventory.principal
            or any(group.kind!='acid' for group in acid_inventory.principal)
            or {group.center for group in acid_inventory.principal}!={c for c,_o in sites}):
        return None  # A pre-existing COOH is not another carboxylate charge.
    from .api import mol_to_iupac
    name=mol_to_iupac(acid)
    if not name.endswith('carboxylic acid'):
        return None
    name=name[:-len('carboxylic acid')]+'carboxylate'
    cations=[];charge=-len(sites)
    simple={3:'lithium',11:'sodium',19:'potassium',12:'magnesium',20:'calcium',30:'zinc'}
    for fragment in fragments:
        if fragment is anion:
            continue
        if fragment.GetNumAtoms()!=1:
            return None
        atom=fragment.GetAtomWithIdx(0);z=atom.GetAtomicNum();q=atom.GetFormalCharge()
        if z in simple and q==({12:2,20:2,30:2}.get(z,1)):
            cations.append(simple[z]);charge+=q
        elif z==7 and q==1 and atom.GetTotalNumHs()==4:
            cations.append('ammonium');charge+=1
        else:
            return None
    if cations:
        if charge:
            return None
        words=[]
        for text,count in sorted(Counter(cations).items()):
            prefix=MULT.get(count) if count>1 else ''
            if prefix is None:
                return None
            words.append(prefix+text)
        name=' '.join(words)+' '+name
    return _out(mol,name,(),'P-72.2.2.2.1.1/P-65.6.2.1/P-77',method='functional-class')


def _tree_paths(mol,atoms):
    adjacency={i:tuple(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in atoms) for i in atoms}
    if sum(map(len,adjacency.values()))!=2*(len(atoms)-1):
        return ()
    ends=[i for i,ns in adjacency.items() if len(ns)==1]
    paths=[]
    for start,end in combinations(ends,2):
        stack=[(start,(start,))]
        while stack:
            current,path=stack.pop()
            if current==end:
                paths.append(path);break
            stack.extend((n,path+(n,)) for n in adjacency[current] if n not in path)
    return tuple(paths)


def heterochain_parent(mol,Chem):
    """P-44 facts first; P-51 counts named heterounits rather than atoms."""
    if (not _plain_graph(mol,Chem,{6,8,14,15,16,32,33,34,50,51,52,82,83})
            or mol.GetRingInfo().NumRings() or any(b.GetBondTypeAsDouble()!=1 and not (
                b.GetBondTypeAsDouble() in {2,3} and b.GetBeginAtom().GetAtomicNum()==b.GetEndAtom().GetAtomicNum()==6)
                for b in iter_bonds(mol))):
        return None
    atoms=frozenset(range(mol.GetNumAtoms()))
    if not any(a.GetAtomicNum()==6 for a in iter_atoms(mol)) or sum(a.GetAtomicNum()!=6 for a in iter_atoms(mol))<4:
        return None
    if any(a.GetTotalValence()!=_VALENCE[a.GetAtomicNum()] for a in iter_atoms(mol)):
        return None
    # Terminal chalcogen H is a senior hydroxy/thiol suffix in a mixed chain.
    if any(a.GetAtomicNum() in {8,16,34,52} and a.GetTotalNumHs() for a in iter_atoms(mol)):
        return None
    paths=_tree_paths(mol,atoms)
    if not paths:
        return None
    from .parent_selection import ParentFacts,parent_seniority_frontier,parent_frontier,locant_key
    from .replacement_eligibility import acyclic_replacement_eligibility
    candidates=[]
    for path in paths:
        for seq in (path,path[::-1]):
            if mol.GetAtomWithIdx(seq[0]).GetAtomicNum() not in {6,14,15,32,33,50,51,82,83} or mol.GetAtomWithIdx(seq[-1]).GetAtomicNum() not in {6,14,15,32,33,50,51,82,83}:
                continue
            mapping={i:str(j) for j,i in enumerate(seq,1)}
            ene=tuple(j for j,(a,b) in enumerate(zip(seq,seq[1:]),1) if _order(mol,a,b)==2)
            yne=tuple(j for j,(a,b) in enumerate(zip(seq,seq[1:]),1) if _order(mol,a,b)==3)
            locants=tuple(locant_key(mapping[i]) for i in seq if mol.GetAtomWithIdx(i).GetAtomicNum()!=6)
            by_element=tuple(tuple(locant_key(mapping[i]) for i in seq if mol.GetAtomWithIdx(i).GetAtomicNum()==z) for z in _ORDER)
            facts=ParentFacts(principal_class=(40,0),suffix_count=0,kind='chain',
                elements=tuple(sorted(Counter(mol.GetAtomWithIdx(i).GetAtomicNum() for i in seq).items())),
                ring_count=0,multiple_count=len(ene)+len(yne),double_count=len(ene),nonstandard_key=(),indicated_h_locants=(),
                replacement_locants=locants,replacement_element_locants=by_element,fusion_locants=(),suffix_locants=(),
                attachment_locants=(),unsaturation_locants=tuple(sorted(locant_key(str(i)) for i in ene+yne)),
                double_locants=tuple(locant_key(str(i)) for i in ene),isotope_key=(),equivalent_isotope_key=(),
                stereo_key=(),multiplied_parents=1,substituent_nonstandard_key=(),substituent_isotope_key=())
            candidates.append((seq,facts))
    maxima=parent_seniority_frontier(candidates,lambda row:row[1])
    if not maxima:
        return None
    rendered=[]
    for seq,facts in maxima:
        proof=acyclic_replacement_eligibility(mol,seq)
        if not proof.eligible or len(seq) not in ROOTS:
            return None
        mapping={i:str(j) for j,i in enumerate(seq,1)};used=set(seq);prefixes=[];branch_ownership=[]
        for i in seq:
            for a in mol.GetAtomWithIdx(i).GetNeighbors():
                if a.GetIdx() in seq:
                    continue
                text,owned=mapped_branch(mol,Chem,a.GetIdx(),i)
                if not text or not owned or owned&used:
                    return None
                prefixes.append((mapping[i],text));used.update(owned)
                branch_ownership.append((a.GetIdx(),i,owned,text))
            else:
                continue
            break
        else:
            if used!=set(atoms):
                return None
            chunks=[]
            for z in _ORDER:
                locs=[mapping[i] for i in seq if mol.GetAtomWithIdx(i).GetAtomicNum()==z]
                if locs:
                    multiplier=MULT.get(len(locs)) if len(locs)>1 else ''
                    if multiplier is None:
                        return None
                    chunks.append(','.join(locs)+'-'+multiplier+_TERMS[z])
            from .formatting import complete_parent_name,unsaturated_parent
            ene=[j for j,(a,b) in enumerate(zip(seq,seq[1:]),1) if _order(mol,a,b)==2]
            yne=[j for j,(a,b) in enumerate(zip(seq,seq[1:]),1) if _order(mol,a,b)==3]
            parent='-'.join(chunks)+complete_parent_name(unsaturated_parent(ROOTS[len(seq)],ene,yne))
            component=MappedComponent(parent,tuple(mapping.items()),frozenset(seq),tuple(prefixes),
                frozenset(atoms-set(seq)),prefix_branch_ownership=tuple(branch_ownership))
            name=decorate_component(mol,Chem,component)
            if not name:
                return None
            prefix_locs=tuple(sorted(locant_key(loc) for loc,_text in prefixes))
            citation=tuple(locant_key(loc) for loc,text in sorted(prefixes,key=lambda r:(prefix_alphabetization_key(r[1]),locant_key(r[0]))))
            full=replace(facts,prefix_count=len(prefixes),prefix_locants=prefix_locs,prefix_citation_locants=citation,
                         alphanumerical_key=(prefix_alphabetization_key(name),(),()))
            rendered.append((name,component,full,proof))
    best=parent_frontier(rendered,lambda row:row[2])
    if not best or len({row[0] for row in best})!=1:
        return None
    name,component,_facts,proof=best[0]
    decision=_out(mol,name,(),'P-44.3/P-44.4/P-45/P-51.4.1.1/P-15.4',method='skeletal-replacement')
    return replace(decision,atom_locants=component.atom_locants,
                   heterounits=tuple(unit.atoms for unit in proof.heterounits))


def select(mol,Chem,numbers):
    if numbers[8] and any(a.GetFormalCharge()<0 for a in iter_atoms(mol)) and mol.GetRingInfo().NumRings():
        result=ring_polycarboxylate(mol,Chem)
        if result:
            return result
    if numbers[14] and numbers[7]:
        result=silicon_imine(mol,Chem)
        if result:
            return result
    if numbers[15] and numbers[8]>=5 and numbers[6]:
        result=phosphono_linked_acid(mol,Chem)
        if result:
            return result
    if numbers[8]>=2 and any(numbers[z] for z in _HYDRIDES if z!=8):
        result=heterohydride_carboxylic_acid(mol,Chem)
        if result:
            return result
    if numbers[8] and (mol.GetRingInfo().NumRings()>=2 or numbers[7]>=4):
        result=intrinsic_carbonyl_locants(mol,Chem)
        if result:
            return result
    hetero_count=sum(numbers[z] for z in _HYDRIDES)
    if hetero_count>=2 and sum(bool(numbers[z]) for z in _HYDRIDES)==1:
        result=homogeneous_parent(mol,Chem)
        if result:
            return result
    if any(numbers[z]>=3 for z in (8,16,34,52)) and numbers[6]:
        result=named_chalcogen_parent(mol,Chem)
        if result:
            return result
    if hetero_count>=4 and numbers[6] and not mol.GetRingInfo().NumRings():
        return heterochain_parent(mol,Chem)
    return None
