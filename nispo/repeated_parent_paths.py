"""Original-graph P-51.3.2.2 nearest parents and mandatory phane competition."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from .naming_result import NamingResult,SelectionLimitation
from .linear_phane_methods import (
    _prove_path,_ordered_path,_edge_atom_for_node_pair,_ring_sequences,
    _best_ring_numberings,linear_phane_frontier,
)
from .formatting import enclose_name_unit,format_prefixes,join_prefix_parent


@dataclass(frozen=True,slots=True)
class RepeatedParentCore:
    atom_maps: tuple[int,...]
    disposition: str
    owner: str


@dataclass(frozen=True,slots=True)
class RepeatedParentNamingResult(NamingResult):
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    multiplicative_name: str = ''
    preferred_method: str = ''
    core_bindings: tuple[RepeatedParentCore,...] = ()
    component_atom_maps: tuple[tuple[str,tuple[int,...]],...] = ()
    repeated_parent_atom_locants: tuple[tuple[int,str],...] = ()
    simplified_node_count: int = 0
    repeated_core_count: int = 0
    multiplied_parent_count: int = 0
    construction_rule: str = 'P-51.3.1/P-51.3.2.2/P-15.3.1.2'


class _Scope(ValueError):pass


def _unit(mol,Chem,proof,node,neighbor):
    from .naming_methods import _side,_fragment
    from .component_mapping import mapped_branch
    root,blocked=_edge_atom_for_node_pair(proof,node,neighbor)
    scope=_side(mol,root,blocked);core=proof.nodes[node][1]
    if not core<=scope:raise _Scope('A nearest parent core crosses its unit boundary.')
    fragment,original=_fragment(mol,Chem,scope)
    if fragment is None:raise _Scope('A source repeated unit cannot be capped.')
    for atom in fragment.GetAtoms():atom.SetAtomMapNum(0)
    fragment.GetAtomWithIdx(original.index(root)).SetAtomMapNum(1)
    signature=Chem.MolToSmiles(fragment,canonical=True,isomericSmiles=True)
    rows=[]
    for sequence in _ring_sequences(mol,core):
        mapping={idx:str(p) for p,idx in enumerate(sequence,1)};prefixes=[];owned=set(core);attachments=[int(mapping[root])]
        for idx in sequence:
            for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
                target=atom.GetIdx()
                if target in core or target==blocked or atom.GetAtomicNum()<=1:continue
                text,branch=mapped_branch(mol,Chem,target,idx)
                if not text or branch&owned or not branch<=scope:raise _Scope('A distal original-graph branch has no complete mapped prefix.')
                prefixes.append((mapping[idx],text));owned.update(branch);attachments.append(int(mapping[idx]))
        if owned!=scope:raise _Scope('A repeated unit has unowned original atoms.')
        # P-15.3.1.3 preserves intrinsic ring numbering and gives the linking
        # multiplicative attachment the lowest locant within that full set.
        key=(tuple(sorted(attachments)),int(mapping[root]))
        name=join_prefix_parent(format_prefixes(prefixes),proof.nodes[node][0])
        rows.append((key,name,mapping[root],tuple(mapping.items())))
    if not rows:raise _Scope('The source parent numbering is unavailable.')
    key=min(row[0] for row in rows);winners=[row for row in rows if row[0]==key]
    if len({(r[1],r[2]) for r in winners})!=1:raise _Scope('Nearest-parent numbering leaves distinct source renderings.')
    _key,name,locant,mapping=winners[0]
    return scope,root,blocked,signature,name,locant,mapping


def _connector(mol,Chem,proof,path,units):
    from .naming_methods import _connector_name,_block_name
    atoms=frozenset().union(*(proof.nodes[n][1] for n in path))
    rings=[n for n in path if n in proof.ring_nodes]
    if not rings:
        from types import SimpleNamespace
        text=_connector_name(mol,atoms,[SimpleNamespace(root=u[1],connector=u[2]) for u in units])
        if not text:raise _Scope('The mapped central path has no source-symmetrical concatenated writer.')
        return text,atoms
    if len(rings)!=1:raise _Scope('More than one central amplificant is outside this nearest-path writer.')
    center=rings[0];position=path.index(center)
    arms=(tuple(reversed(path[:position])),tuple(path[position+1:]))
    if not arms[0] or len(arms[0])!=len(arms[1]):raise _Scope('The central ring needs two identical nonempty outward arms.')
    elements=[tuple(mol.GetAtomWithIdx(next(iter(proof.nodes[n][1]))).GetAtomicNum() for n in arm) for arm in arms]
    if elements[0]!=elements[1]:raise _Scope('The noncentral multiplicative components are asymmetrical.')
    blocks=[(element,len(tuple(items))) for element,items in groupby(elements[0])]
    if any(a!=6 and b!=6 and a!=b for (a,_),(b,_) in zip(blocks,blocks[1:])):
        raise _Scope('Mixed adjacent heterogroups need their own source vocabulary.')
    terms=[_block_name(*block,outward=True) for block in blocks]
    if not all(terms):raise _Scope('A central arm component has no validated multivalent term.')
    ports=tuple(_edge_atom_for_node_pair(proof,center,arm[0])[0] for arm in arms)
    sequences=_best_ring_numberings(mol,proof.nodes[center][1],ports)
    if not sequences:raise _Scope('The central ring attachment numbering is unproved.')
    locs=sorted(sequences[0].index(idx)+1 for idx in ports);listed=','.join(map(str,locs));parent=proof.nodes[center][0]
    central=listed+'-phenylene' if parent=='benzene' else parent+'-'+listed+'-diyl'
    return central+'bis'+enclose_name_unit(''.join(terms)),atoms


def repeated_parent_to_iupac_result(smiles:str) -> RepeatedParentNamingResult:
    from .api import Chem
    if not isinstance(smiles,str) or not smiles.strip() or Chem is None:
        return RepeatedParentNamingResult('','invalid-input','not-assessed',
            (SelectionLimitation('nearest-repeated-parent','P-51.3.2.2','A valid connected SMILES input is required.'),),'nearest-repeated-parent')
    try:
        mol=Chem.MolFromSmiles(smiles)
        if mol is None or len(Chem.GetMolFrags(mol))!=1:raise _Scope('A connected source molecule is required.')
        for atom in mol.GetAtoms():atom.SetAtomMapNum(0)
        proof=_prove_path(mol,Chem)
        if proof is None:raise _Scope('A neutral identical-aromatic-core path with original C/N/O/S links is not proved.')
        path=_ordered_path(proof)
        if path is None:raise _Scope('The complete simplified skeleton is not one path.')
        core_positions=[i for i,n in enumerate(path) if n in proof.ring_nodes];count=len(core_positions)
        if count<4 or any(b-a<2 for a,b in zip(core_positions,core_positions[1:])):
            raise _Scope('At least four ring copies separated by nonempty linking arms are required.')
        # Distal copies are enclosed in each complete rooted parent unit;
        # they are not discarded by contracting just terminal ring cores.
        left=(count-1)//2 if count%2==0 else count//2-1
        right=left+1 if count%2==0 else left+2
        a,b=core_positions[left],core_positions[right]
        units=(_unit(mol,Chem,proof,path[a],path[a+1]),_unit(mol,Chem,proof,path[b],path[b-1]))
        if units[0][3]!=units[1][3] or units[0][4:6]!=units[1][4:6]:
            raise _Scope('The two nearest complete parent units are not identically substituted and attached.')
        connector,central=_connector(mol,Chem,proof,path[a+1:b],units)
        heavy=frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum()>1)
        components=(units[0][0],units[1][0],central)
        if sum(map(len,components))!=len(heavy) or frozenset().union(*components)!=heavy:
            raise _Scope('The repeated units and central group do not partition the original molecule.')
        parent,locant=units[0][4:6]
        alternative=locant+','+locant+"'-"+enclose_name_unit(connector)+'bis'+enclose_name_unit(parent)
        phane=linear_phane_frontier(mol,Chem)
        if not phane.mandatory_eligible:raise _Scope('Preferred-method competition is outside the admitted phane gate.')
        preferred=phane.decision.name if phane.disposition=='complete' and phane.decision else ''
        justification=phane.justification if preferred else None
        preferred_status=('resolved' if justification is not None and justification.complete
                          else 'not-assessed' if preferred else 'unresolved')
        issues=() if preferred else (SelectionLimitation('repeated-parent-method',phane.source_rule,phane.reason),)
        identity=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
        owners={idx:'parent:'+str(j) for j,u in enumerate(units,1) for idx in u[0]}
        owners.update({idx:'central-group' for idx in central})
        selected={path[a],path[b]};bindings=[]
        for n in path:
            if n not in proof.ring_nodes:continue
            maps=tuple(sorted(idx+1 for idx in proof.nodes[n][1]));owner=owners[next(iter(proof.nodes[n][1]))]
            bindings.append(RepeatedParentCore(maps,'selected' if n in selected else 'central' if owner=='central-group' else 'distal-substituent',owner))
        groups=tuple((owner,tuple(idx+1 for idx in sorted(atoms))) for owner,atoms in zip(('parent:1','parent:2','central-group'),components))
        mapping=tuple((idx+1,loc+"'"*j) for j,u in enumerate(units) for idx,loc in u[6])
        for atom in mol.GetAtoms():atom.SetAtomMapNum(atom.GetIdx()+1)
        return RepeatedParentNamingResult(name=preferred or alternative,generation_status='available',preferred_status=preferred_status,
            unresolved=issues,route='nearest-repeated-parent',justification=justification,canonical_smiles=identity,mapped_smiles=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True),
            multiplicative_name=alternative,preferred_method='phane',core_bindings=tuple(bindings),component_atom_maps=groups,
            repeated_parent_atom_locants=mapping,simplified_node_count=len(path),repeated_core_count=count,multiplied_parent_count=2)
    except (_Scope,ValueError,RuntimeError,OverflowError) as error:
        return RepeatedParentNamingResult('','unavailable','not-assessed',
            (SelectionLimitation('nearest-repeated-parent','P-51.3.2.2/P-52.2.5.1(2)',str(error)),),'nearest-repeated-parent')


def repeated_parent_to_iupac(smiles:str) -> str:return repeated_parent_to_iupac_result(smiles).name
