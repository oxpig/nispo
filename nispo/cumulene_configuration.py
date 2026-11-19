"""Explicit 3D axial configuration joined to mapped even-cumulene parents."""
from __future__ import annotations

from dataclasses import dataclass
import re

from .axial_stereochemistry import AxialNamingResult,AxialUnit
from .naming_result import SelectionLimitation


@dataclass(frozen=True,slots=True)
class CumuleneNamingResult(AxialNamingResult):
    conformer_id: int = -1
    coordinates: tuple[tuple[float,float,float],...] = ()
    torsion_degrees: float | None = None


def _failure(status,reason):
    return CumuleneNamingResult(name='', generation_status=status,
        preferred_status='not-assessed' if status=='invalid-input' else 'unresolved',
        unresolved=(SelectionLimitation('cumulene-configuration','P-93.4.2.2/P-92.1.2.2',reason),),
        route='explicit-3D-cumulene')


def _axis_path(mol,Chem):
    adjacency={}
    for bond in mol.GetBonds():
        if bond.GetBondType()!=Chem.BondType.DOUBLE:continue
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(a).GetAtomicNum()!=6 or mol.GetAtomWithIdx(b).GetAtomicNum()!=6:return ()
        adjacency.setdefault(a,set()).add(b);adjacency.setdefault(b,set()).add(a)
    ends=[idx for idx,neighbors in adjacency.items() if len(neighbors)==1]
    if len(ends)!=2 or any(len(neighbors)>2 for neighbors in adjacency.values()):return ()
    sequence=[];previous=None;current=ends[0]
    while current is not None:
        if current in sequence:return ()
        sequence.append(current);onward=adjacency[current]-{previous}
        previous,current=current,next(iter(onward)) if onward else None
    return tuple(sequence) if len(sequence)==len(adjacency) else ()


def _constitutional_proposal(mol,Chem):
    """Use the established source parent frontier on a private H-capped copy."""
    from .cumulene_chain_methods import cumulene_chain_decision
    if (mol.GetRingInfo().NumRings() or any(a.GetAtomicNum() not in {1,6,8,9,17,35,53}
            or a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
        return None
    halogens=[a for a in mol.GetAtoms() if a.GetAtomicNum() in {9,17,35,53}]
    carbons=[a for a in mol.GetAtoms() if a.GetAtomicNum()==6]
    unbranched=all(sum(n.GetAtomicNum()==6 for n in a.GetNeighbors())<=2 for a in carbons)
    if (halogens or any(a.GetIsotope() for a in mol.GetAtoms())) and not unbranched:
        return None
    if any(a.GetIsotope() and (a.GetAtomicNum() not in {1,6}
            or Chem.GetPeriodicTable().GetMassForIsotope(a.GetAtomicNum(),a.GetIsotope())<=0) for a in mol.GetAtoms()):
        return None
    if any(a.GetDegree()!=1 or a.GetNeighbors()[0].GetAtomicNum()!=6
            or mol.GetBondBetweenAtoms(a.GetIdx(),a.GetNeighbors()[0].GetIdx()).GetBondType()!=Chem.BondType.SINGLE for a in halogens):
        return None
    # Retain carbon/OH scaffold IDs separately. Deletion and H completion only
    # propose the constitutional parent; every final atom/annotation comes
    # from the untouched original graph.
    original_ids=tuple(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in {6,8})
    editable=Chem.RWMol(mol)
    for idx in reversed(range(mol.GetNumAtoms())):
        if idx not in original_ids:editable.RemoveAtom(idx)
    base=editable.GetMol();Chem.RemoveStereochemistry(base)
    for atom in base.GetAtoms():
        atom.SetIsotope(0);atom.SetAtomMapNum(0);atom.SetNumExplicitHs(0);atom.SetNoImplicit(False)
    try:Chem.SanitizeMol(base)
    except (ValueError,RuntimeError):return None
    proposal=cumulene_chain_decision(base,Chem)
    return (proposal,original_ids) if proposal is not None else None


def _independent_ordinary_configuration(mol,Chem):
    """Prove ordinary centres without unrepresented axial auxiliaries.

    Retaining each centre alone tests whether its sequence is already decided
    by constitution/isotopes. Coupled and pseudoasymmetric assignments need a
    joint digraph carrying the cumulene axis, which this operation lacks.
    """
    from .stereochemistry import assigned_atom_stereochemistry
    from rdkit.Chem import rdCIPLabeler
    assigned=dict(assigned_atom_stereochemistry(mol,Chem,include_pseudo=True))
    for atom in mol.GetAtoms():
        if atom.GetChiralTag()==Chem.ChiralType.CHI_UNSPECIFIED:continue
        idx=atom.GetIdx();code=assigned.get(idx)
        if code not in {'R','S'}:return False
        probe=Chem.Mol(mol);Chem.RemoveStereochemistry(probe)
        for obj in list(probe.GetAtoms())+list(probe.GetBonds()):
            if obj.HasProp('_CIPCode'):obj.ClearProp('_CIPCode')
        probe.GetAtomWithIdx(idx).SetChiralTag(atom.GetChiralTag())
        try:rdCIPLabeler.AssignCIPLabels(probe,maxRecursiveIterations=1000000)
        except (ValueError,RuntimeError,OverflowError):return False
        center=probe.GetAtomWithIdx(idx)
        if not center.HasProp('_CIPCode') or center.GetProp('_CIPCode')!=code:return False
    return True


def _render_original(mol,Chem,proposal,original_ids,axis,descriptor):
    from .cumulene_chain_methods import _render_name
    from .component_mapping import MappedComponent,mapped_branch,fully_covered,decorate_component,hydrogen_bindings
    from .formatting import prefix_alphabetization_key
    from .parent_annotations import isotope_numbering_key,stereo_numbering
    from .parent_selection import locant_key
    hydroxyl=[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==8]
    candidates=[]
    for alternative in proposal.numbering_alternatives:
        mapping={original_ids[idx]:loc for idx,loc in alternative}
        path=tuple(sorted(mapping,key=lambda idx:int(mapping[idx])))
        if not set(axis)<=mapping.keys():continue
        numbers=sorted(int(mapping[idx]) for idx in axis)
        if any(b-a!=1 for a,b in zip(numbers,numbers[1:])):continue
        locant=str(numbers[0])
        double_locs=tuple(sorted(min(int(mapping[b.GetBeginAtomIdx()]),int(mapping[b.GetEndAtomIdx()]))
            for b in mol.GetBonds() if b.GetBondType()==Chem.BondType.DOUBLE))
        suffix_locs=tuple(int(mapping[mol.GetAtomWithIdx(o).GetNeighbors()[0].GetIdx()]) for o in hydroxyl)
        parent=_render_name(len(path),double_locs,suffix_locs,())
        if not parent:continue
        mapping=hydrogen_bindings(mol,mapping);embedded=set(hydroxyl);prefixes=[];valid=True
        for idx in path:
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                root=neighbor.GetIdx()
                if root in mapping or root in embedded or neighbor.GetAtomicNum()==1 and not neighbor.GetIsotope():continue
                text,owned=mapped_branch(mol,Chem,root,idx)
                if not text or owned&(mapping.keys()|embedded):valid=False;break
                prefixes.append((mapping[idx],text));embedded.update(owned)
            if not valid:break
        if not valid or not fully_covered(mol,mapping,embedded):continue
        isotope=isotope_numbering_key(mol,mapping,include_substituent_groups=True);stereo=stereo_numbering(mol,mapping)
        if isotope is None or stereo is None:continue
        component=MappedComponent(parent,tuple(mapping.items()),frozenset(mapping),tuple(prefixes),frozenset(embedded),systematic_parent=True)
        name=decorate_component(mol,Chem,component)
        if not name:continue
        # New halo prefixes are admitted only on an unbranched carbon frame,
        # where they cannot alter the pre-proved parent or prefix count.
        prefix_locs=tuple(sorted(locant_key(loc) for loc,_word in prefixes))
        citation=tuple(locant_key(loc) for loc,_word in sorted(prefixes,
            key=lambda row:(prefix_alphabetization_key(row[1])[:-1],locant_key(row[0]))))
        leading=re.match(r'^\(([^()]*)\)-',name)
        pieces=[];body=name
        if leading and all(re.fullmatch(r"\d+[a-z]?'*[RSrsEZ]",part) for part in leading[1].split(',')):
            pieces=leading[1].split(',');body=name[leading.end():]
        pieces.append(locant+descriptor)
        pieces.sort(key=lambda part:locant_key(part[:-1]))
        name='('+','.join(pieces)+')-'+body
        candidates.append(((prefix_locs,citation,isotope,stereo),name,tuple(mapping.items()),locant))
    if not candidates:return None
    rank=min(row[0] for row in candidates);best=[row for row in candidates if row[0]==rank]
    if len({(name,loc) for _rank,name,_mapping,loc in best})!=1:return None
    return best[0][1:]


def cumulene_to_iupac_result(mol,*,axis_path=None,conformer_id=-1,
                             distance_tolerance=1e-3,angular_tolerance=1.0) -> CumuleneNamingResult:
    """Name one declared 3D even-cumulene axis with retained geometric input.

    The supplied conformer, not a constitution-only SMILES, provides the axial
    configuration. Initial parent scope is the established acyclic C/OH chain
    grammar; unbranched chains also admit halo pendants and C/H isotopes.
    Axis atoms must be collinear with orthogonal end-ligand projections at the
    stated tolerances. General cyclic/coupled/pseudoasymmetric cases decline.
    """
    from .api import Chem
    from .conformations import describe_conformer,_ligands,_precedence
    from .conformer_geometry import signed_torsion
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    described=describe_conformer(mol,conformer_id=conformer_id,axes=(),rings=(),
        distance_tolerance=distance_tolerance,angular_tolerance=angular_tolerance)
    if described.status!='available':
        return _failure(described.status,described.limitations[0].reason if described.limitations else 'The supplied conformer is unavailable.')
    graph=Chem.Mol(mol)
    for atom in graph.GetAtoms():atom.SetAtomMapNum(0)
    Chem.AssignStereochemistryFrom3D(graph,confId=described.conformer_id,replaceExistingTags=True)
    Chem.AssignStereochemistry(graph,cleanIt=True,force=True)
    if not _independent_ordinary_configuration(graph,Chem):
        return _failure('unavailable','Coupled or pseudoasymmetric centres need a joint sequence digraph carrying the cumulene axis.')
    proposed=_constitutional_proposal(graph,Chem)
    if proposed is None:return _failure('unavailable','This annotated cumulene has no completed source parent/branch frontier.')
    proposal,ids=proposed
    parent_sets={frozenset(ids[idx] for idx,_loc in alternative) for alternative in proposal.numbering_alternatives}
    if len(parent_sets)>1 and any(a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED for a in graph.GetAtoms()):
        return _failure('unavailable','Distinct annotated parent frames need a completed configurational parent comparison.')
    path=_axis_path(graph,Chem)
    if not path or (len(path)-1)%2 or not 2<=len(path)-1<=12:
        return _failure('unavailable','One maximal even-carbon cumulene axis is required; odd cumulenes need E/Z.')
    if axis_path is not None:
        if (not isinstance(axis_path,tuple) or any(type(i) is not int for i in axis_path)
                or axis_path not in {path,tuple(reversed(path))}):
            return _failure('invalid-input','The explicit axis must be the complete ordered maximal cumulene path.')
        path=axis_path
    ends=((path[0],path[1]),(path[-1],path[-2]));selected=[]
    for end,next_atom in ends:
        groups=_ligands(graph,Chem,end,next_atom,allow_axis_double=True)
        if groups is None or len(groups)!=2:return _failure('unavailable','Each terminal atom needs two complete distinguishable ligands.')
        precedence=_precedence(graph,Chem,groups[0],groups[1])
        if precedence is None or precedence==0:return _failure('unavailable','Terminal ligand sequence precedence is unproved or equivalent.')
        ligand=groups[0] if precedence==1 else groups[1]
        if ligand[0] is None:return _failure('unavailable','A selected implicit reference has no supplied coordinate.')
        selected.append(ligand[0])
    refs=(selected[0],path[0],path[-1],selected[1]);points=described.coordinates
    angle=signed_torsion([points[idx] for idx in refs],[points[idx] for idx in path],distance_tolerance)
    if angle is None or abs(abs(angle)-90)>max(angular_tolerance,1e-9):
        return _failure('unavailable','The coordinates do not prove a collinear axis with orthogonal ligand planes at the supplied tolerances.')
    descriptor='P' if angle>0 else 'M'
    rendered=_render_original(graph,Chem,proposal,ids,path,descriptor)
    if rendered is None:return _failure('unavailable','Original annotations or tied parent maps lack one completed axial naming join.')
    name,mapping,locant=rendered
    unit=AxialUnit(path,descriptor,locant,refs)
    return CumuleneNamingResult(name=name, generation_status='available', preferred_status='resolved',
        route='mapped-3D-cumulene', isomeric_cxsmiles=Chem.MolToCXSmiles(graph),
        axial_units=(unit,), atom_locants=mapping, conformer_id=described.conformer_id,
        coordinates=points, torsion_degrees=angle)


def cumulene_to_iupac(mol,**kwargs) -> str:
    return cumulene_to_iupac_result(mol,**kwargs).name
