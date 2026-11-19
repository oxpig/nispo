"""Bounded P-62.2.4.1.3/P-51 method proofs for ordinary polyamines.

Terminal characteristic Ns are not replacement heterounits. Repeated units
are senior amine parents attached through a mapped N, not convenient monoamine
fragments cut from a more senior polyamine. All rendering uses original atoms.
"""
from collections import defaultdict
from dataclasses import dataclass

from .molecule_cache import mol_cache_dict


def _heavy(mol):
    return frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()>1)


def _linear_skeleton(mol,groups):
    from .naming_methods import _linear_order,_isolated_heterounits
    heavy=_heavy(mol)
    if mol.GetRingInfo().NumRings() or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6,7} for i in heavy):return None
    if any(b.GetBondTypeAsDouble()!=1 for b in mol.GetBonds()):return None
    terminal={g.center for g in groups if sum(n.GetAtomicNum()>1 for n in mol.GetAtomWithIdx(g.center).GetNeighbors())==1}
    if not 1<=len(terminal)<=2:return None
    atoms=heavy-terminal
    ends=[i for i in atoms if sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors())==1]
    path=_linear_order(mol,atoms,ends)
    if not path or any(not any(n.GetIdx() in terminal for n in mol.GetAtomWithIdx(i).GetNeighbors())
                       for i in ends if mol.GetAtomWithIdx(i).GetAtomicNum()!=6):return None
    if any(next(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum()>1) not in ends for i in terminal):return None
    units=_isolated_heterounits(mol,path)
    return (path,terminal,units) if units is not None else None


def _replacement(mol,Chem,groups):
    from .naming_methods import MethodDecision,_replacement_key,_replacement_text
    from .amine_components import amine_ending,suffix_site_maps
    from .component_mapping import MappedComponent,decorate_component,hydrogen_bindings
    from .parent_annotations import isotope_seniority,stereo_numbering
    from .constants import ROOTS
    skeleton=_linear_skeleton(mol,groups)
    if skeleton is None or len(skeleton[2])<4:return None
    path,terminal,units=skeleton
    if len(path) not in ROOTS:return None
    suffix_groups=tuple(g for g in groups if g.center in terminal)
    options=[]
    for seq in (path,tuple(reversed(path))):
        mapping={idx:str(j) for j,idx in enumerate(seq,1)}
        sites,nmap=next(suffix_site_maps(mol,suffix_groups,seq,mapping))
        locs=tuple(sorted((mapping[i] for i in sites),key=int))
        parent=_replacement_text(mol,seq)
        name,owned=amine_ending(mol,Chem,parent,suffix_groups,locs)
        mapping=hydrogen_bindings(mol,{**mapping,**nmap})
        if not name:continue
        name=decorate_component(mol,Chem,MappedComponent(name,tuple(mapping.items()),frozenset(mapping),
            systematic_parent=True,nitrogen_site_locants=True,local_isotope_atoms=owned))
        iso,stereo=isotope_seniority(mol,mapping),stereo_numbering(mol,mapping)
        if not name or iso is None or stereo is None:continue
        key=(_replacement_key(mol,seq),tuple(map(int,locs)),iso,stereo)
        options.append((key,name,mapping,seq))
    if not options:return None
    key=min(r[0] for r in options);best=[r for r in options if r[0]==key]
    if len({r[1] for r in best})!=1:return None
    _,name,mapping,path=best[0]
    return MethodDecision(name,'skeletal-replacement',frozenset(terminal),frozenset(mapping),
        'P-62.2.4.1.3/P-51.4.1.1/P-51.4.1.2',tuple(mapping.items()),tuple(units),
        (('replacement-skeleton',tuple(path)),('characteristic-N',tuple(sorted(terminal)))))


@dataclass(frozen=True)
class _Unit:
    atoms:frozenset
    root:int
    connector:int
    signature:str
    name:str
    locant:str
    centers:frozenset
    mapping:tuple
    facts:object
    substituted:bool


def _units(mol,Chem,inventory):
    from .naming_methods import _side,_fragment
    from .amine_components import chain_proposals,carbon_hosts
    from .parent_selection import parent_frontier,locant_key
    from dataclasses import replace
    groups=inventory.principal
    # P-44.1.1/P-62.2.4.1.3: a multiplied monoamine cannot displace a
    # diamine parent. This producer proves complete, single-carbon-block
    # polyamine units; more complex units remain barriers in the scope check.
    frameworks=[];remaining={a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==6}
    while remaining:
        block=set();stack=[next(iter(remaining))]
        while stack:
            i=stack.pop()
            if i not in remaining:continue
            remaining.remove(i);block.add(i)
            stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in remaining)
        frameworks.append(block)
    senior_count=max((sum(bool(carbon_hosts(mol,g,b)) for g in groups) for b in frameworks),default=0)
    if senior_count<2:return
    def block_key(block):
        # These complete all-carbon blocks are paths or simple monocycles.
        # For them P-44.2/P-44.3 size and bond criteria are fully established.
        if any(sum(n.GetIdx() in block for n in mol.GetAtomWithIdx(i).GetNeighbors())>2 for i in block):return None
        rings=sum(set(r)<=block for r in mol.GetRingInfo().AtomRings())
        if rings>1 or any(mol.GetAtomWithIdx(i).IsInRing() for i in block) and rings!=1:return None
        bonds=[b.GetBondTypeAsDouble() for b in mol.GetBonds() if {b.GetBeginAtomIdx(),b.GetEndAtomIdx()}<=block]
        return (-sum(bool(carbon_hosts(mol,g,block)) for g in groups),-bool(rings),-len(block),
                -sum(o>1 for o in bonds),-sum(o==2 for o in bonds))
    keys=[block_key(b) for b in frameworks]
    if any(k is None for k in keys):return
    senior_key=min(keys)
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondTypeAsDouble()!=1:continue
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        for root,blocked in ((a,b),(b,a)):
            if mol.GetAtomWithIdx(root).GetAtomicNum()!=7:continue
            atoms=_side(mol,root,blocked)
            contained=tuple(g for g in groups if g.center in atoms)
            if len(contained)!=senior_count or len(contained)==len(groups):continue
            carbon={i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum()==6}
            if not any(carbon==block for block in frameworks):continue
            if block_key(carbon)!=senior_key:continue
            if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {1,6} and i not in {g.center for g in contained} for i in atoms):continue
            if any(not carbon_hosts(mol,g,carbon) for g in contained):continue
            hosts=[carbon_hosts(mol,g,carbon) for g in contained]
            if any(len(h)!=1 for h in hosts) or len({h[0] for h in hosts})!=len(hosts):continue
            candidates=[p for p in chain_proposals(mol,Chem,inventory,scope=atoms,principal=contained)
                        if root in p.principal_centers and len(p.principal_centers)==senior_count]
            from .mapped_ring_parents import records,parent_facts
            from .cross_route_selection import ParentProposal
            for record in records(mol,Chem,scope=atoms,principal=contained,nitrogen_attachment=root):
                facts=parent_facts(mol,record,contained[0].seniority)
                if facts is not None and root in record.principal_centers and len(record.principal_centers)==senior_count:
                    candidates.append(ParentProposal(record.name,'polyamine-ring-unit','amine',record.principal_centers,
                        record.parent_atoms,record.atom_locants,facts,True))
            # The connector is a chemical prefix when numbering a repeated
            # parent. Its N-site locant precedes isotope/stereo tie breakers;
            # it cannot be added after choosing the capped parent's numbering.
            numbered=[]
            for p in candidates:
                f=p.parent_selection;loc=locant_key(dict(p.atom_locants)[root])
                if f.prefix_count:continue  # Connector citation among other units is unproved here.
                f=replace(f,prefix_count=1,prefix_locants=(loc,),prefix_citation_locants=(loc,),
                    numbering_scope=(frozenset(p.parent_atoms),p.principal_centers))
                numbered.append(replace(p,parent_selection=f))
            candidates=numbered
            frontier=parent_frontier(candidates,lambda p:p.parent_selection)
            if not frontier:continue
            best=min(locant_key(dict(p.atom_locants)[root]) for p in frontier)
            frontier=[p for p in frontier if locant_key(dict(p.atom_locants)[root])==best]
            if len({p.name for p in frontier})!=1:continue
            p=frontier[0]
            fragment,original=_fragment(mol,Chem,atoms)
            if fragment is None:continue
            for atom in fragment.GetAtoms():atom.SetAtomMapNum(0)
            fragment.GetAtomWithIdx(original.index(root)).SetAtomMapNum(1)
            signature=Chem.MolToSmiles(fragment,canonical=True,isomericSmiles=True)
            yield _Unit(atoms,root,blocked,signature,p.name,dict(p.atom_locants)[root],
                p.principal_centers,p.atom_locants,p.parent_selection,False)


def _multiplicative(mol,Chem,inventory):
    from .naming_methods import MethodDecision,_connector_name
    from .component_mapping import has_annotations,nitrogen_locant
    from .formatting import enclose_name_unit
    from .constants import MULT
    grouped=defaultdict(list)
    for unit in _units(mol,Chem,inventory):
        grouped[(unit.signature,unit.name,unit.locant)].append(unit)
    needed={a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()>1 or a.GetIsotope()}
    choices=[]
    for units in grouped.values():
        if not 2<=len(units)<=4:continue
        covered=frozenset().union(*(u.atoms for u in units))
        if sum(len(u.atoms) for u in units)!=len(covered):continue
        connector=frozenset(needed-covered)
        if any(u.connector not in connector for u in units):continue
        if has_annotations(mol,connector):
            mol_cache_dict(mol,'polyamine_methods')['unproved-connector']=True
            continue
        bridge=_connector_name(mol,connector,units)
        if not bridge:
            # A ring connector needs P-28/P-62 construction, not the linear
            # connector grammar. A failed symmetry test on a known linear
            # connector, by contrast, proves multiplication ineligible.
            if any(mol.GetAtomWithIdx(i).IsInRing() for i in connector):
                mol_cache_dict(mol,'polyamine_methods')['unproved-connector']=True
            continue
        # P-62.2.4.1.3 uses di(alkane-x,1-diyl) for the two simple arms.
        if bridge.startswith('azanediylbis('):bridge=bridge.replace('azanediylbis(','azanediyldi(',1)
        locs=','.join(nitrogen_locant(units[0].locant+"'"*j) for j in range(len(units)))
        multiplier=({2:'bis',3:'tris',4:'tetrakis'}[len(units)] if units[0].substituted else MULT[len(units)])
        name=locs+'-'+(enclose_name_unit(bridge) if any(c.isdigit() or c in '()' for c in bridge) else bridge)
        name+=multiplier+enclose_name_unit(units[0].name)
        mapping=[]
        for j,unit in enumerate(units):
            mapping.extend((idx,loc+"'"*j) for idx,loc in unit.mapping)
        roles=tuple(('repeated-parent',tuple(sorted(u.atoms))) for u in units)+(('connector',tuple(sorted(connector))),)
        centers=frozenset().union(*(u.centers for u in units))
        choices.append((-len(units),name,MethodDecision(name,'multiplicative',centers,frozenset(needed),
            'P-62.2.4.1.3/P-62.2.5.2/P-45.1/P-51.3.1',tuple(mapping),(),roles)))
    if not choices:return None
    best=min(r[0] for r in choices);final=[r for r in choices if r[0]==best]
    return final[0][2] if len({r[1] for r in final})==1 else None


def select_polyamine_method(mol,Chem,inventory):
    cache=mol_cache_dict(mol,'polyamine_methods')
    if 'decision' not in cache:
        from .amine_components import ordinary_amine,common_framework
        groups=inventory.principal
        if not inventory.supported or not groups or any(not ordinary_amine(mol,g.center) for g in groups) or common_framework(mol,groups):
            cache['decision']=None
        else:
            cache['decision']=_replacement(mol,Chem,groups) or _multiplicative(mol,Chem,inventory)
    return cache['decision']


def _mononitrogen_repeat_boundary(mol,groups):
    """P-62.2.5.2(1): a one-carbon geminal parent can compete with
    repeated longer monoamine parents. Do not resolve that class-specific
    conflict using a generic per-parent suffix count.
    """
    from .naming_methods import _side,_fragment
    from rdkit import Chem
    centers={g.center for g in groups}
    if len(centers)<2:return False
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum()!=6 or any(n.GetAtomicNum()==6 for n in carbon.GetNeighbors()):continue
        if not centers<={n.GetIdx() for n in carbon.GetNeighbors()}:continue
        signatures=[]
        for n in centers:
            atoms=_side(mol,n,carbon.GetIdx())
            if atoms&centers!={n} or sum(mol.GetAtomWithIdx(i).GetAtomicNum()==6 for i in atoms)<=1:break
            fragment,original=_fragment(mol,Chem,atoms)
            if fragment is None:break
            for atom in fragment.GetAtoms():atom.SetAtomMapNum(0)
            fragment.GetAtomWithIdx(original.index(n)).SetAtomMapNum(1)
            signatures.append(Chem.MolToSmiles(fragment,canonical=True,isomericSmiles=True))
        if len(signatures)==len(centers) and len(set(signatures))==1:return True
    return False


def substitutive_scope(mol,groups):
    from .amine_components import common_framework
    if common_framework(mol,groups):return not _mononitrogen_repeat_boundary(mol,groups)
    if any(a.GetAtomicNum()==6 and
           sum(n.GetAtomicNum()==6 for n in a.GetNeighbors())>2 for a in mol.GetAtoms()):return False
    # Four or more rings invoke phane/replacement questions outside this
    # producer. Four internal amine units require a proved role partition.
    if mol.GetRingInfo().NumRings()>=4:return False
    skeleton=_linear_skeleton(mol,groups)
    if skeleton is not None and len(skeleton[2])>=4:return False
    internal=sum(sum(n.GetAtomicNum()>1 for n in mol.GetAtomWithIdx(g.center).GetNeighbors())>=2 for g in groups)
    if internal>=4:return False
    from rdkit import Chem
    from .component_mapping import chemical_inventory
    if select_polyamine_method(mol,Chem,chemical_inventory(mol)) is not None:return False
    if mol_cache_dict(mol,'polyamine_methods').get('unproved-connector'):return False
    return True
