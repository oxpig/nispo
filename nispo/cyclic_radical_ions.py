"""P-75.2 localized carbon radical ions on chains and single ring parents."""
from __future__ import annotations


def _cycle_sequences(adjacency):
    for start in adjacency:
        for first in adjacency[start]:
            sequence=[start];previous,current=start,first
            while current!=start:
                if current in sequence or len(adjacency[current])!=2:return
                sequence.append(current);onward=adjacency[current]-{previous}
                if len(onward)!=1:return
                previous,current=current,next(iter(onward))
            if len(sequence)==len(adjacency):yield tuple(sequence)


def carbon_chain_or_cycle_radical_ion_decision(mol,Chem):
    """Complete carbon skeleton/H-loss facts, never inferred electron location."""
    if (mol is None or not 2<=mol.GetNumHeavyAtoms()<=64 or mol.GetNumAtoms()!=mol.GetNumHeavyAtoms()
            or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol))!=1):return None
    atoms=tuple(mol.GetAtoms());bonds=tuple(mol.GetBonds())
    if any(a.GetAtomicNum()!=6 or a.GetIsotope() or int(a.GetChiralTag()) or a.GetIsAromatic()
           or a.GetFormalCharge() not in {-1,0,1} or a.GetNumRadicalElectrons() not in {0,1} for a in atoms):return None
    if any(b.GetBondType() not in {Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE}
           or int(b.GetStereo()) or int(b.GetBondDir()) for b in bonds):return None
    radicals=frozenset(a.GetIdx() for a in atoms if a.GetNumRadicalElectrons())
    negative=frozenset(a.GetIdx() for a in atoms if a.GetFormalCharge()<0)
    positive=frozenset(a.GetIdx() for a in atoms if a.GetFormalCharge()>0)
    if not radicals or not negative|positive:return None
    for atom in atoms:
        expected=4-abs(atom.GetFormalCharge())-atom.GetNumRadicalElectrons()
        weighted=sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
        if (atom.GetTotalValence()!=expected or atom.GetTotalNumHs(includeNeighbors=True)!=expected-weighted
                or sum(b.GetBondTypeAsDouble()>1 for b in atom.GetBonds())>1):return None
    graph={a.GetIdx():frozenset(n.GetIdx() for n in a.GetNeighbors()) for a in atoms}
    cycle_rank=len(bonds)-len(atoms)+1;ring=cycle_rank==1
    if cycle_rank not in {0,1}:return None
    from .constants import ROOTS
    states={}
    if ring:
        core=frozenset(a.GetIdx() for a in atoms if a.IsInRing())
        adjacency={i:graph[i]&core for i in core}
        if not (radicals|negative|positive)<=core or not core or any(len(ns)!=2 for ns in adjacency.values()):return None
        multiple={frozenset((b.GetBeginAtomIdx(),b.GetEndAtomIdx())):int(b.GetBondTypeAsDouble())
                  for b in bonds if b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core}
        if any(order>1 for order in multiple.values()):
            from .unsaturated_ring_parents import unsaturated_monocycle_frontier,render_unsaturated_parent
            frontier=unsaturated_monocycle_frontier(adjacency,{i:6 for i in core},multiple)
            if frontier.status!='complete':return None
            for item in frontier.candidates:
                cited=lambda order:tuple(sorted((a,) if b-a==1 else (a,b)
                    for a,b,degree in item.unsaturation.edge_locants if order is None or degree==order))
                states[item.sequence]=(render_unsaturated_parent(item.parent_component,item.unsaturation),
                    cited(None),cited(2))
        else:
            if len(core) not in ROOTS:return None
            for sequence in _cycle_sequences(adjacency):states[sequence]=('cyclo'+ROOTS[len(core)]+'ane',(),())
        if len(states)!=2*len(core):return None
    else:
        # The entire parent is one unbranched chain, so radical and ionic
        # coverage cannot force selection of a shorter competing path.
        if any(len(ns)>2 for ns in graph.values()) or not any(b.GetBondTypeAsDouble()>1 for b in bonds):return None
        ends=[i for i,ns in graph.items() if len(ns)==1]
        if len(ends)!=2:return None
        sequence=[];previous,current=None,ends[0]
        while current is not None:
            sequence.append(current);onward=graph[current]-{previous}
            previous,current=current,next(iter(onward)) if onward else None
        if len(sequence)!=len(atoms) or len(sequence) not in ROOTS:return None
        from .unsaturated_ring_parents import UnsaturationState,render_unsaturated_parent
        for path in (tuple(sequence),tuple(reversed(sequence))):
            mult=tuple(i for i,(a,b) in enumerate(zip(path,path[1:]),1) if mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble()>1)
            double=tuple(i for i,(a,b) in enumerate(zip(path,path[1:]),1) if mol.GetBondBetweenAtoms(a,b).GetBondType()==Chem.BondType.DOUBLE)
            triple=tuple(i for i in mult if i not in double)
            edges=tuple((i,i+1,int(mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble()))
                        for i,(a,b) in enumerate(zip(path,path[1:]),1) if i in mult)
            state=UnsaturationState(edges,mult,double,triple)
            states[path]=(render_unsaturated_parent(ROOTS[len(path)]+'ane',state,omit_single_locant=len(path)==2),
                tuple((i,) for i in mult),tuple((i,) for i in double))
    from .formatting import attach_suffix,multiplied_suffix,format_prefixes,join_prefix_parent,prefix_alphabetization_key
    from .component_mapping import mapped_branch,fully_covered
    from .naming_methods import MethodDecision
    candidates=[];centres=radicals|negative|positive
    for path,(parent,multiple,double) in states.items():
        if not parent:return None
        mapping={idx:str(position) for position,idx in enumerate(path,1)}
        rloc=tuple(sorted(int(mapping[i]) for i in radicals));nloc=tuple(sorted(int(mapping[i]) for i in negative));ploc=tuple(sorted(int(mapping[i]) for i in positive))
        prefixes=[];roles=[('radical-ion-parent',frozenset(path),tuple(mapping.items()))];owned=set(path)
        for host in path:
            for root in graph[host]-set(path):
                if mol.GetBondBetweenAtoms(host,root).GetBondType()!=Chem.BondType.SINGLE:return None
                text,branch=mapped_branch(mol,Chem,root,host)
                if not text or not branch or branch&owned or branch&centres:return None
                prefixes.append((mapping[host],text));roles.append(('neutral-carbon-prefix',frozenset(branch),()));owned.update(branch)
        if not fully_covered(mol,owned,frozenset()):return None
        for locants,term in ((ploc,'ylium'),(nloc,'ide'),(rloc,'yl')):
            if locants:
                suffix=multiplied_suffix(len(locants),term)
                if not suffix:return None
                if term=='yl' and parent.endswith('e'):parent=parent[:-1]
                parent=attach_suffix(parent,suffix,locants)
        name=join_prefix_parent(format_prefixes(prefixes),parent) if prefixes else parent
        citation=tuple(int(loc) for loc,text in sorted(prefixes,key=lambda row:(prefix_alphabetization_key(row[1]),int(row[0]))))
        rank=(rloc,nloc,ploc,multiple,double,-len(prefixes),tuple(sorted(int(loc) for loc,_word in prefixes)),citation)
        candidates.append((rank,name,path,tuple(roles)))
    if not candidates:return None
    best=min(row[0] for row in candidates);winners=[row for row in candidates if row[0]==best]
    if len({row[1] for row in winners})!=1:return None
    _rank,name,path,roles=winners[0]
    return MethodDecision(name,'radical-ion',radicals,frozenset(graph),'P-75.2/P-71.7/P-72.2/P-73.2/P-31',
        tuple((idx,str(position)) for position,idx in enumerate(path,1)),component_roles=roles)
