"""Mapped imine, repeated-ring and hydroxylamine parent assemblies."""
from itertools import product
from types import SimpleNamespace

from .functional_parent_extensions import side, capped, order
from .bluebook_lambda_amine_methods import _plain, _out
from .constants import ROOTS
from .formatting import enclose_name_unit, format_prefixes


def unit(mol, Chem, atoms, root):
    """Choose a fully capped unit independently of its central linker."""
    fragment, original = capped(mol, Chem, atoms)
    if fragment is None:
        return None
    local = original.index(root)
    fragment.GetAtomWithIdx(local).SetAtomMapNum(1)
    identity = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
    fragment.GetAtomWithIdx(local).SetAtomMapNum(0)
    from .extended_inventory import selection_inventory
    inv = selection_inventory(fragment)
    if not inv.supported:
        return None
    from .naming_methods import _unit_parent_cache_entry
    if inv.principal:
        kinds = {g.kind for g in inv.principal}
        if len(kinds) != 1 or not kinds <= {'acid','alcohol'}:
            return None
        kind = next(iter(kinds))
        if kind == 'alcohol':
            rings=fragment.GetRingInfo().AtomRings()
            if len(rings)==1 and len(rings[0])==6 and local in rings[0]:
                ring=rings[0]
                hydroxyls=[g for g in inv.principal if g.center in ring]
                if (len(hydroxyls)==1 and len(original)==7 and all(
                        fragment.GetAtomWithIdx(i).GetAtomicNum()==6 and fragment.GetAtomWithIdx(i).GetIsAromatic() for i in ring)):
                    h=hydroxyls[0].center
                    distance=min((ring.index(local)-ring.index(h))%6,(ring.index(h)-ring.index(local))%6)
                    return identity,'phenol',str(distance+1)
        pool = _unit_parent_cache_entry(fragment, Chem, local, kind, compare=kind != 'alcohol')
        if kind != 'alcohol':
            if pool.comparison.reason or pool.comparison.selected is None:
                return None
            row = pool.comparison.selected
            return identity, row[1], str(row[2])
        # Alcohol units have one parent family here. Numbering is fixed by
        # the suffix before minimizing the attachment locant.
        rows = pool.rows
        if not rows or len({r[1] for r in rows}) != 1:
            return None
        row = min(rows, key=lambda r: (r[0],int(r[2])))
        return identity, row[1], str(row[2])
    rings = frozenset(i for ring in fragment.GetRingInfo().AtomRings() for i in ring)
    if rings != frozenset(range(fragment.GetNumAtoms())) or local not in rings:
        return None
    from .mapped_ring_parents import assembly_unit_skeletons
    rows = list(assembly_unit_skeletons(fragment, Chem, rings))
    if not rows or len({r[0] for r in rows}) != 1:
        return None
    locants = [dict(zip(r[1],r[2]))[local] for r in rows]
    if not all(s.isdigit() for s in locants):
        return None
    return identity, rows[0][0], min(locants,key=int)


def _assembly_name(parent, locant, bridge):
    # bis avoids implying a continuous hydride, and encloses parents with
    # internal punctuation (P-15.3.2.3/P-16.3).
    complex_parent = parent.startswith(('di','tri')) or parent[0].isdigit()
    if parent.endswith(('silane','phosphane','azane','sulfane','oxidane','hydroxylamine')):
        complex_parent = True
    suffix = 'bis('+parent+')' if complex_parent else 'di'+(enclose_name_unit(parent) if '[' in parent else parent)
    connector = bridge if bridge in {'oxy','sulfanediyl','azanediyl'} else enclose_name_unit(bridge)
    return locant+','+locant+"'-"+connector+suffix


def repeated_ring_parents(mol, Chem):
    if not _plain(mol, Chem, {6,7,8,16}) or mol.GetRingInfo().NumRings() < 2:
        return None
    from .mapped_ring_parents import ring_systems
    systems = ring_systems(mol)
    if len(systems) != 2:
        return None
    # Recover exactly one unit per terminal ring system. A branch crossing
    # back into the other ring is a connector, never a unit substituent.
    choices = []
    for core in systems:
        options = []
        for root in core:
            for a in mol.GetAtomWithIdx(root).GetNeighbors():
                if a.GetIdx() in core or order(mol,root,a.GetIdx()) != 1:
                    continue
                atoms = side(mol,root,a.GetIdx())
                if not core <= atoms or any(other <= atoms for other in systems if other != core):
                    continue
                row = unit(mol,Chem,atoms,root)
                if row:
                    options.append((atoms,root,a.GetIdx(),row))
        choices.append(options)
    names = set()
    for left,right in product(*choices):
        if left[0] & right[0] or left[3] != right[3]:
            continue
        connector = set(range(mol.GetNumAtoms())) - left[0] - right[0]
        if not connector or left[2] not in connector or right[2] not in connector:
            continue
        from .naming_methods import _connector_name, _linear_order
        units = [SimpleNamespace(root=r[1],connector=r[2]) for r in (left,right)]
        bridge = _connector_name(mol,connector,units)
        path = _linear_order(mol,connector,[left[2],right[2]])
        if path and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in path):
            if len(path) not in ROOTS:
                continue
            doubles = [j for j,(a,b) in enumerate(zip(path,path[1:]),1) if order(mol,a,b) == 2]
            if doubles and all(order(mol,a,b) in {1,2} for a,b in zip(path,path[1:])):
                # A fully owned carbon path; no branches or lost H sites.
                from .formatting import unsaturated_parent, complete_parent_name
                base = complete_parent_name(unsaturated_parent(ROOTS[len(path)],doubles,[]))
                bridge = base+'-1,'+str(len(path))+'-diyl'
        if bridge:
            # O is named oxy in P-15.3.2.1 repeated-parent constructions.
            if len(connector) == 1 and mol.GetAtomWithIdx(next(iter(connector))).GetAtomicNum() == 8:
                bridge = 'oxy'
            names.add(_assembly_name(left[3][1],left[3][2],bridge))
    if len(names) != 1:
        return None
    return _out(mol,names.pop(),(),'P-15.3.2.1/P-15.3.2.3',method='multiplicative')


def imine_assembly(mol, Chem):
    if not _plain(mol,Chem,{6,7,8}) or sum(a.GetAtomicNum()==7 for a in mol.GetAtoms()) != 2:
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum()==7]
    ends = []
    for n in ns:
        imines = [a for a in n.GetNeighbors() if a.GetAtomicNum()==6 and order(mol,n.GetIdx(),a.GetIdx())==2]
        carbon = [a for a in n.GetNeighbors() if a.GetAtomicNum()==6 and order(mol,n.GetIdx(),a.GetIdx())==1]
        if len(imines)!=1 or len(carbon)!=1 or n.GetTotalNumHs():
            return None
        c=imines[0]; ring_roots=[a for a in c.GetNeighbors() if a.IsInRing() and order(mol,c.GetIdx(),a.GetIdx())==1]
        if len(ring_roots)==1 and c.GetTotalNumHs()==1:
            root=ring_roots[0].GetIdx(); blocked=c.GetIdx(); extra=True
        else:
            root=c.GetIdx(); blocked=n.GetIdx(); extra=False
        atoms=side(mol,root,blocked)
        row=unit(mol,Chem,atoms,root)
        if row is None or not row[1].endswith(('acid','phenol','ol')):
            return None
        ends.append((atoms,root,blocked,row,carbon[0].GetIdx(),extra))
    left,right=ends
    if left[0]&right[0] or left[3]!=right[3] or left[5]!=right[5]:
        return None
    nids={n.GetIdx() for n in ns}
    outer={r[2] for r in ends} if left[5] else set()
    center=set(range(mol.GetNumAtoms()))-left[0]-right[0]-nids-outer
    from .naming_methods import _connector_name
    units=[SimpleNamespace(root=n.GetIdx(),connector=r[4]) for n,r in zip(ns,ends)]
    bridge=_connector_name(mol,center,units)
    if not bridge:
        return None
    group='azanylylidenemethanylylidene' if left[5] else 'azanylylidene'
    bridge=bridge+'bis('+group+')'
    return _out(mol,_assembly_name(left[3][1],left[3][2],bridge),(),
                'P-62.3.1.2/P-15.3',method='multiplicative')


def hydroxylamine_assembly(mol,Chem):
    if not _plain(mol,Chem,{6,7,8}):
        return None
    ns=[a for a in mol.GetAtoms() if a.GetAtomicNum()==7]
    if len(ns)!=2 or any(a.GetDegree()!=1 or a.GetTotalNumHs()!=2 for a in ns):
        return None  # N-organyl hydroxylamines use amine parents.
    os=[a.GetNeighbors()[0] for a in ns]
    if any(a.GetAtomicNum()!=8 or a.GetDegree()!=2 for a in os) or os[0].GetIdx()==os[1].GetIdx():
        return None
    core=set(range(mol.GetNumAtoms()))-{a.GetIdx() for a in ns+os}
    if not core or any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6 for i in core):
        return None
    roots=[[n.GetIdx() for n in o.GetNeighbors() if n.GetAtomicNum()==6] for o in os]
    if any(len(r)!=1 for r in roots):
        return None
    from .naming_methods import _connector_name
    units=[SimpleNamespace(root=o.GetIdx(),connector=r[0]) for o,r in zip(os,roots)]
    bridge=_connector_name(mol,core,units)
    if not bridge:
        rings=mol.GetRingInfo().AtomRings()
        if len(rings)==1 and frozenset(rings[0])==core and len(core)==6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in core):
            ring=rings[0]; distance=min((ring.index(roots[1][0])-ring.index(roots[0][0]))%6,
                                     (ring.index(roots[0][0])-ring.index(roots[1][0]))%6)
            bridge='benzene-1,'+str(distance+1)+'-diyl'
    if not bridge:
        return None
    return _out(mol,"O,O'-"+enclose_name_unit(bridge)+'bis(hydroxylamine)',
                (a.GetIdx() for a in ns),'P-68.3.1.1.1.2',method='multiplicative')


def replacement_spirobi(mol,Chem):
    """P-24.3.4 compares identical carbon skeletons before replacement."""
    if (not _plain(mol,Chem,{6,7,8,14,15,16,34,52}) or mol.GetRingInfo().NumRings()<4
            or any(b.GetBondTypeAsDouble()!=1 for b in mol.GetBonds())
            or all(a.GetAtomicNum()==6 for a in mol.GetAtoms())):
        return None
    atoms=set(range(mol.GetNumAtoms()))
    if set(i for r in mol.GetRingInfo().AtomRings() for i in r)!=atoms:
        return None
    from .ring_topology import ring_system_graph
    from .spiro_component_rules import _two_component_splits_by_articulation, _submol_for_atoms, _cycle_rank_for_atoms
    graph=ring_system_graph(mol,Chem,atoms)
    splits=_two_component_splits_by_articulation(graph,atoms)
    if len(splits)!=1:
        return None
    spiro,parts=next(iter(splits.items()))
    if len(parts)!=2 or mol.GetAtomWithIdx(spiro).GetAtomicNum()!=6:
        return None
    components=[]
    from .polycyclic_rules import bicyclic_parent_candidate_records
    for part in parts:
        subset=set(part)|{spiro}
        if _cycle_rank_for_atoms(graph,subset)!=2:
            return None
        row=_submol_for_atoms(mol,Chem,subset)
        if row is None:
            return None
        sub,back,forward=row
        abstract=Chem.RWMol(sub)
        for a in abstract.GetAtoms():
            a.SetAtomicNum(6);a.SetNumExplicitHs(0);a.SetNoImplicit(False)
        abstract=abstract.GetMol()
        try:Chem.SanitizeMol(abstract)
        except (ValueError,RuntimeError):return None
        mark=Chem.Mol(abstract);mark.GetAtomWithIdx(forward[spiro]).SetIsotope(13)
        identity=Chem.MolToSmiles(mark)
        candidates=[]
        for record in bicyclic_parent_candidate_records(abstract,Chem):
            name,seq=record[:2]
            if len(seq)!=len(subset) or not name.startswith('bicyclo'):
                continue
            mapping={back[i]:str(j) for j,i in enumerate(seq,1)}
            candidates.append((name,mapping))
        if not candidates:
            return None
        components.append((identity,candidates))
    if components[0][0]!=components[1][0]:
        return None
    replacement={8:'oxa',16:'thia',34:'selena',52:'tellura',7:'aza',15:'phospha',14:'sila'}
    hierarchy=(8,16,34,52,7,15,14)
    choices=[]
    for left,right in product(components[0][1],components[1][1]):
        if left[0]!=right[0] or left[1][spiro]!=right[1][spiro]:
            continue
        for a,b in ((left,right),(right,left)):
            mapping=dict(a[1]);mapping.update({i:loc+"'" for i,loc in b[1].items() if i!=spiro})
            prefixes=[(loc,replacement[mol.GetAtomWithIdx(i).GetAtomicNum()]) for i,loc in mapping.items()
                      if mol.GetAtomWithIdx(i).GetAtomicNum()!=6]
            loc_key=lambda loc:(int(loc.rstrip("'")),loc.endswith("'"))
            key=(int(a[1][spiro]),tuple(sorted(loc_key(loc) for loc,_text in prefixes)),
                 tuple(tuple(sorted(loc_key(loc) for i,loc in mapping.items()
                     if mol.GetAtomWithIdx(i).GetAtomicNum()==e)) for e in hierarchy))
            # Skeletal replacement prefixes follow element seniority, not
            # alphabetical order; one prefix owns both primed components.
            chunks=[]
            for e in hierarchy:
                locs=sorted((loc for i,loc in mapping.items() if mol.GetAtomWithIdx(i).GetAtomicNum()==e),key=loc_key)
                if locs:
                    from .constants import MULT
                    chunks.append(','.join(locs)+'-'+(MULT[len(locs)] if len(locs)>1 else '')+replacement[e])
            loc=a[1][spiro]
            text='-'.join(chunks)+'-'+loc+','+loc+"'-spirobi["+a[0]+']'
            choices.append((key,text))
    if not choices:
        return None
    best=min(c[0] for c in choices)
    names={text for key,text in choices if key==best}
    return _out(mol,names.pop(),(spiro,),'P-24.3.4') if len(names)==1 else None
