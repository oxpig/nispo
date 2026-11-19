"""Original-graph axial CIP and a bounded mapped biaryl naming operation."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .input_stereochemistry import unsupported_relative_stereochemistry
from .naming_result import NamingResult,SelectionLimitation


@dataclass(frozen=True,slots=True)
class AxialUnit:
    atoms: tuple[int,...]
    descriptor: str
    locant: str = ''
    reference_atoms: tuple[int,...] = ()


@dataclass(frozen=True,slots=True)
class AxialNamingResult(NamingResult):
    isomeric_cxsmiles: str = ''
    axial_units: tuple[AxialUnit,...] = ()
    atom_locants: tuple[tuple[int,str],...] = ()


def _failure(status,reason,identity='',units=()):
    return AxialNamingResult(name='', generation_status=status,
        preferred_status='not-assessed' if status=='invalid-input' else 'unresolved',
        unresolved=(SelectionLimitation('axial-configuration','P-92.1.2.2/P-93.5.7.1',reason),),
        route='explicit-axial-configuration', isomeric_cxsmiles=identity,
        axial_units=tuple(units))


def _biphenyl_construction(mol,Chem,axis):
    """All original atoms belong to two benzene units or mapped pendants."""
    from .component_mapping import MappedComponent,mapped_branch,fully_covered,decorate_component,hydrogen_bindings
    from .parent_selection import locant_key
    from .parent_annotations import isotope_numbering_key,stereo_numbering
    from .stereochemistry import assigned_atom_stereochemistry,_indexed_bond_stereochemistry
    from .formatting import prefix_alphabetization_key
    from .parent_rules import _ring_sequences_from_connector
    rings=tuple(frozenset(r) for r in mol.GetRingInfo().AtomRings())
    if (len(rings)!=2 or rings[0]&rings[1] or any(len(ring)!=6 for ring in rings)
            or any(not mol.GetAtomWithIdx(idx).GetIsAromatic() or mol.GetAtomWithIdx(idx).GetAtomicNum()!=6 for ring in rings for idx in ring)
            or any(a.GetAtomicNum() not in {1,6,9,17,35,53} or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
        return None
    a,b=axis
    if a in rings[0] and b in rings[1]:left,right=rings
    elif b in rings[0] and a in rings[1]:right,left=rings
    else:return None
    between=[bond for bond in mol.GetBonds() if
        (bond.GetBeginAtomIdx() in left and bond.GetEndAtomIdx() in right)
        or (bond.GetBeginAtomIdx() in right and bond.GetEndAtomIdx() in left)]
    if len(between)!=1 or between[0].GetBondType()!=Chem.BondType.SINGLE:return None
    candidates=[]
    assigned=dict(assigned_atom_stereochemistry(mol,Chem,include_pseudo=True))
    fixed_bonds=_indexed_bond_stereochemistry(mol,Chem)
    for first,second,x,y in ((left,right,a,b),(right,left,b,a)):
        first_seqs=_ring_sequences_from_connector(mol,tuple(first),x)
        second_seqs=_ring_sequences_from_connector(mol,tuple(second),y)
        for front,rear in product(first_seqs,second_seqs):
            mapping={idx:str(loc) for loc,idx in enumerate(front,1)}
            mapping.update({idx:str(loc)+"'" for loc,idx in enumerate(rear,1)})
            mapping=hydrogen_bindings(mol,mapping);embedded=set();prefixes=[];branch_units=[];valid=True
            for idx in front+rear:
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                    root=neighbor.GetIdx()
                    if root in mapping or neighbor.GetAtomicNum()==1 and not neighbor.GetIsotope():continue
                    text,owned=mapped_branch(mol,Chem,root,idx)
                    if not text or owned&(mapping.keys()|embedded):valid=False;break
                    prefixes.append((mapping[idx],text));embedded.update(owned)
                    branch_units.append((mapping[idx],root,owned))
                if not valid:break
            if not valid or not fully_covered(mol,mapping,embedded):continue
            isotope=isotope_numbering_key(mol,mapping,include_substituent_groups=True)
            stereo=stereo_numbering(mol,mapping)
            if isotope is None or stereo is None:continue
            component=MappedComponent("1,1'-biphenyl",tuple(mapping.items()),frozenset(mapping),
                tuple(prefixes),frozenset(embedded),systematic_parent=True)
            name=decorate_component(mol,Chem,component)
            if not name:continue
            locants=tuple(sorted((locant_key(loc) for loc,_word in prefixes)))
            cited=tuple(locant_key(loc) for loc,word in sorted(prefixes,
                key=lambda item:(prefix_alphabetization_key(item[1])[:-1],locant_key(item[0]))))
            # R/S is a late configurational comparison, not part of the
            # Roman-letter prefix citation order. A singly stereogenic
            # pendant rooted at that centre has a complete local sequence.
            # More elaborate branch comparisons leave differently spelled
            # tied maps unresolved instead of using raw descriptor text.
            branch_stereo=[];proved=True
            for loc,root,owned in branch_units:
                centres=owned & assigned.keys()
                if centres and centres!={root} or any({x,y}<=owned for x,y,_d in fixed_bonds):
                    proved=False;break
                if centres:branch_stereo.append((locant_key(loc),0 if assigned[root] in {'R','r'} else 1))
            branch_rank=tuple(sorted(branch_stereo)) if proved else ()
            candidates.append(((locants,cited,isotope,stereo,branch_rank),name,tuple(mapping.items())))
    if not candidates:return None
    rank=min(row[0] for row in candidates)
    best=[row for row in candidates if row[0]==rank]
    if len({name for _rank,name,_mapping in best})!=1:return None
    # All tied original-atom maps use the same lowest axis locant. Retain one
    # map for this delivered spelling; no text tiebreak chooses a parent.
    return best[0][1:]


def _atropisomer_result(mol,Chem):
    if unsupported_relative_stereochemistry(mol, Chem):
        return _failure('unavailable',
            'Relative or mixed axial configuration needs an explicitly declared ensemble operation.')
    if mol.GetStereoGroups():return _failure('unavailable','Enhanced axial stereo groups need a combined substance operation.')
    native={Chem.BondStereo.STEREOATROPCW,Chem.BondStereo.STEREOATROPCCW}
    bonds=[bond for bond in mol.GetBonds() if bond.GetStereo() in native]
    if not bonds:return _failure('invalid-input','An explicitly assigned native atropisomeric bond is required.')
    if len(Chem.GetMolFrags(mol))!=1:return _failure('invalid-input','One connected axial molecular structure is required.')
    if len(bonds)>1:return _failure('unavailable','Multiple axes need a completed joint numbering operation.')
    if any(bond.GetBondType()!=Chem.BondType.SINGLE for bond in bonds):
        return _failure('invalid-input','An atropisomeric axis must be a single bond.')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp('_CIPCode'):atom.ClearProp('_CIPCode')
    for bond in mol.GetBonds():
        if bond.HasProp('_CIPCode'):bond.ClearProp('_CIPCode')
    from rdkit.Chem import rdCIPLabeler
    try:rdCIPLabeler.AssignCIPLabels(mol,maxRecursiveIterations=1000000)
    except (ValueError,RuntimeError,OverflowError):return _failure('unavailable','Complete original-graph axial CIP assignment did not finish.')
    identity=Chem.MolToCXSmiles(mol)
    units=[]
    for bond in bonds:
        code=bond.GetProp('_CIPCode') if bond.HasProp('_CIPCode') else ''
        if code not in {'M','P','m','p'}:
            return _failure('invalid-input','The declared axis has no proved distinct ligand configuration.',identity)
        anchors=tuple(bond.GetPropsAsDict().get('_CIPNeighborOrder',()))
        units.append(AxialUnit((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()),code,'',anchors))
    construction=_biphenyl_construction(mol,Chem,units[0].atoms)
    if construction is None:
        return _failure('unavailable','Axial CIP is retained but this parent has no completed mapped axial-name join.',identity,units)
    name,mapping=construction
    from dataclasses import replace
    from .parent_selection import locant_key
    locants=dict(mapping);locant=min((locants[i] for i in units[0].atoms),key=locant_key)
    unit=replace(units[0],locant=locant)
    name='('+locant+unit.descriptor+')-'+name
    return AxialNamingResult(name=name, generation_status='available', preferred_status='resolved',
        route='mapped-biphenyl-axial', isomeric_cxsmiles=identity,
        axial_units=(unit,), atom_locants=mapping)


def atropisomer_to_iupac_result(mol) -> AxialNamingResult:
    """Name an explicitly declared atropisomer; retain its native CX identity.

    The input declares restricted rotation. This operation does not estimate
    a kinetic rotational barrier from constitution. Complete original-graph
    CIP is separate from the mapped parent scope of the initial writer.
    Equivalent CX wedge encodings need not be identical strings in RDKit.
    """
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        graph=Chem.Mol(mol)
        if not graph.GetNumAtoms():return _failure('invalid-input','A nonempty molecular graph is required.')
        Chem.SanitizeMol(graph)
    except (ValueError,RuntimeError,TypeError,AttributeError):return _failure('invalid-input','A valid molecular graph is required.')
    try:return _atropisomer_result(graph,Chem)
    except (ValueError,RuntimeError,OverflowError):
        return _failure('unavailable','The declared axial graph cannot complete this construction.')


def atropisomer_to_iupac(mol) -> str:
    return atropisomer_to_iupac_result(mol).name
