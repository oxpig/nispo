"""Source class and partial actual-parent comparisons for organic adducts."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True,slots=True)
class ComponentPriority:
    class_rank: int|None = None
    functional_facts: object = None
    parent_facts: object = None


def _source_class(mol,inventory,Chem):
    charge=sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):return (2 if charge<0 else 3 if charge>0 else 1),None,()
    if charge:return (4 if charge<0 else 6),None,()
    from .neutral_charge_components import validate_neutral_charge_role_proofs
    if any(a.GetFormalCharge() for a in mol.GetAtoms()) and not validate_neutral_charge_role_proofs(mol,inventory.charge_role_proofs):
        # Opposed recognized ionic groups establish a zwitterion. A net-zero
        # charge sum alone also describes covalent Lewis representations.
        ionic=[g for g,f in zip(inventory.groups,inventory.source_group_facts) if f is not None and f.class_rank in {4,6}]
        covered={i for g in ionic for i in g.atoms}
        charged={a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()}
        ranks={f.class_rank for f in inventory.source_group_facts if f is not None}
        return (5 if inventory.supported and charged<=covered and {4,6}<=ranks and len(Chem.GetMolFrags(mol))==1 else None),None,()
    if not inventory.supported:return None,None,()
    if inventory.groups:
        if (len(inventory.groups)!=len(inventory.source_group_facts)
                or any(f is None or f.class_rank is None for f in inventory.source_group_facts)):return None,None,()
        facts={id(g):f for g,f in zip(inventory.groups,inventory.source_group_facts)}
        rank=min(f.class_rank for f in facts.values())
        from .group_priority import priority_frontier_analysis,compare_functional_group_priority
        frontier=priority_frontier_analysis(inventory.groups,lambda g:facts[id(g)])
        if frontier.status not in {'selected','tie'} or frontier.unknown_pairs or frontier.cycle:return rank,None,()
        principal=frontier.items
        if not principal:return rank,None,()
        representative=facts[id(principal[0])]
        if representative is None or any(compare_functional_group_priority(representative,facts[id(g)]).status!='tie' for g in principal):
            return rank,None,()
        return representative.class_rank,representative,principal
    motifs={i for proof in inventory.charge_role_proofs for i in proof.atoms}
    heavy=[a for a in mol.GetAtoms() if a.GetAtomicNum()>1 and a.GetIdx() not in motifs]
    if (not heavy or any(a.GetAtomicNum() not in {6,7,8,16,9,17,35,53} for a in heavy)
            or any(a.GetAtomicNum() in {7,8,16} and not a.IsInRing() for a in heavy)):
        return None,None,()
    elements={a.GetAtomicNum() for a in heavy}
    rank=21 if 7 in elements else 35 if 8 in elements else 36 if 16 in elements else 40
    return rank,None,()


def _leading_parent(mol,inventory,rank,principal,Chem):
    if rank is None or rank<=6:return None
    # These families have the hydrocarbon-parent suffix permissions used
    # below. A known class alone does not make, e.g., a nitramide or peroxol
    # functional parent into the molecule's entire carbon skeleton.
    permitted_suffixes={'acid','amide','nitrile','aldehyde','ketone','alcohol','thiol','amine'}
    if any(g.kind not in permitted_suffixes for g in principal):return None
    from .neutral_charge_components import validate_neutral_charge_role_proofs
    if not inventory.supported or not validate_neutral_charge_role_proofs(mol,inventory.charge_role_proofs):return None
    carbon=frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==6)
    if not carbon:return None
    from .mapped_ring_parents import ring_systems
    systems=ring_systems(mol)
    if systems:
        if len(systems)!=1:return None
        parent=frozenset(systems[0]);kind='ring'
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6,7,8,16} for i in parent):return None
        permitted={'acid','amide','nitrile','aldehyde','amine','sulfonic-acid','sulfinic-acid','sulfonamide'}
        hosts=[]
        for g in principal:
            if g.center in parent:hosts.append(g.center);continue
            attached={n.GetIdx() for n in mol.GetAtomWithIdx(g.center).GetNeighbors() if n.GetIdx() in parent}
            if g.kind not in permitted or len(attached)!=1:return None
            hosts.append(next(iter(attached)))
        # A higher-class acyclic skeleton outside the ring is not an alkyl
        # prefix. Complete supported function/prefix ownership must cover it.
        accounted=set(parent)|{i for g in inventory.groups for i in g.atoms}|{i for p in inventory.charge_role_proofs for i in p.atoms}
        if any(a.GetIdx() not in accounted and a.GetAtomicNum() not in {1,6,9,17,35,53} for a in mol.GetAtoms()):return None
        adjacency={i:frozenset(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in parent) for i in parent}
        edges=sum(map(len,adjacency.values()))//2;rings=edges-len(parent)+1
        if rings<1:return None
        sequences=[]
        if rings==1 and all(len(ns)==2 for ns in adjacency.values()):
            for start in parent:
                for first in adjacency[start]:
                    seq=[start];previous,current=start,first
                    while current!=start:
                        if current in seq:return None
                        seq.append(current);previous,current=current,next(iter(adjacency[current]-{previous}))
                    if len(seq)!=len(parent):return None
                    sequences.append(tuple(seq))
    else:
        parent=carbon;kind='chain';rings=0;hosts=[]
        adjacency={i:frozenset(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in parent) for i in parent}
        if any(len(ns)>2 for ns in adjacency.values()) or sum(map(len,adjacency.values()))!=2*(len(parent)-1):return None
        for g in principal:
            if g.center in parent:hosts.append(g.center);continue
            attached={n.GetIdx() for n in mol.GetAtomWithIdx(g.center).GetNeighbors() if n.GetIdx() in parent}
            if len(attached)!=1:return None
            hosts.append(next(iter(attached)))
        accounted=set(parent)|{i for g in inventory.groups for i in g.atoms}|{i for p in inventory.charge_role_proofs for i in p.atoms}
        if any(a.GetIdx() not in accounted and a.GetAtomicNum() not in {1,9,17,35,53} for a in mol.GetAtoms()):return None
        ends=[i for i,ns in adjacency.items() if len(ns)<=1]
        if not ends:return None
        seq=[];previous,current=None,ends[0]
        while current is not None:
            if current in seq:return None
            seq.append(current);onward=adjacency[current]-{previous};previous,current=current,next(iter(onward)) if onward else None
        if len(seq)!=len(parent):return None
        sequences=[tuple(seq),tuple(reversed(seq))]
    if any(mol.GetAtomWithIdx(i).GetFormalCharge() or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in parent):return None
    bonds=[b for b in mol.GetBonds() if b.GetBeginAtomIdx() in parent and b.GetEndAtomIdx() in parent]
    if any(b.GetBondType() not in {Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC} for b in bonds):return None
    if any(b.GetIsAromatic() for b in bonds):
        from .ring_parent_selection import aromatic_bond_counts
        multiple,double=aromatic_bond_counts(mol,parent)
    else:
        multiple=sum(b.GetBondTypeAsDouble()>1 for b in bonds);double=sum(b.GetBondType()==Chem.BondType.DOUBLE for b in bonds)
    # Only leading intrinsic criteria are supplied. Missing later facts stop
    # comparison; a true parent tie does not order distinct whole compounds.
    suffix=None
    if sequences:
        suffix=min(tuple(sorted(seq.index(host)+1 for host in hosts)) for seq in sequences)
    normal={6:4,7:3,8:2,16:2}
    ordinary=all(mol.GetAtomWithIdx(i).GetTotalValence()==normal[mol.GetAtomWithIdx(i).GetAtomicNum()] for i in parent)
    from .parent_selection import ParentFacts,locant_set
    return ParentFacts(principal_class=(rank,),suffix_count=len(principal),kind=kind,
        elements=tuple(sorted(Counter(mol.GetAtomWithIdx(i).GetAtomicNum() for i in parent).items())),
        ring_count=rings,multiple_count=multiple,double_count=double,nonstandard_key=() if ordinary else None,
        indicated_h_locants=() if kind=='chain' or rings==1 and all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 for i in parent) else None,
        replacement_locants=() if all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 for i in parent) else None,
        replacement_element_locants=() if all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 for i in parent) else None,
        fusion_locants=() if rings<=1 else None,suffix_locants=locant_set(suffix) if suffix is not None else None,
        attachment_locants=())


def component_priority(mol,Chem):
    from .extended_inventory import selection_inventory
    inventory=selection_inventory(mol)
    rank,facts,principal=_source_class(mol,inventory,Chem)
    parent=None if rank is not None and rank<21 and facts is None else _leading_parent(mol,inventory,rank,principal,Chem)
    return ComponentPriority(rank,facts,parent)
