"""Explicit conventional-neutral P-106 nucleotide phosphate constructions."""
from __future__ import annotations

from dataclasses import dataclass

from .naming_result import SelectionLimitation
from .nucleosides import (
    NucleosideSpec,NucleosideNamingResult,_PARENTS,_ROLE,_OWNER,
    _NucleosideInput,_NucleosideScope,_build_nucleoside,_prefixes,
    _role_index,_consume_hydrogen,_new_atom,_validate_graph,_serialized_graph,
)


@dataclass(frozen=True,slots=True)
class NucleotideSpec:
    nucleoside: NucleosideSpec
    position: int = 5
    phosphate_count: int = 1
    bridge_replacement: str = 'oxygen'
    phosphorothioate: bool = False


@dataclass(frozen=True,slots=True)
class NucleotideNamingResult(NucleosideNamingResult):
    phosphate_position: int = 0
    phosphate_count: int = 0
    phosphate_backbone_maps: tuple[int,...] = ()
    acidic_atom_maps: tuple[int,...] = ()
    acid_hydrogen_count: int = 0
    bridge_replacement: str = 'oxygen'
    phosphorus_configuration: str = 'unspecified'


def _failure(status,reason):
    return NucleotideNamingResult('',status,'not-assessed',
        (SelectionLimitation('nucleotide-construction','P-106',reason),),'explicit-nucleotide')


def _validate_spec(spec):
    if (not isinstance(spec,NucleotideSpec) or not isinstance(spec.nucleoside,NucleosideSpec)
            or type(spec.position) is not int or spec.position not in {2,3,5}
            or type(spec.phosphate_count) is not int or spec.phosphate_count not in {1,2,3}
            or not isinstance(spec.bridge_replacement,str)
            or spec.bridge_replacement not in {'oxygen','sulfur','methylene'}
            or type(spec.phosphorothioate) is not bool):
        raise _NucleosideInput('A nucleoside, one 2/3/5-prime site and a one-to-three-P chain with explicit operations are required.')
    if spec.phosphate_count==1 and spec.bridge_replacement!='oxygen':
        raise _NucleosideInput('A monophosphate has no P-O-P bridge to replace.')
    if spec.phosphorothioate and spec.phosphate_count!=1:
        raise _NucleosideScope('The nonbridging thio position in a multi-P chain is not specified by this input model.')


def _phosphate_chain(editable,Chem,anchor,count,mode,owner,*,thio=False):
    """Append R-O-P(=O)(OH)-X-...-P(=O)(OH)2 to one available OH.

    Returns original indices in parent-acid order P1, X2, P3, X4, P5 and
    the nonbridging acid OH/SH indices. No phosphorus configuration is assigned.
    """
    atom=editable.GetAtomWithIdx(anchor)
    if atom.GetAtomicNum()!=8 or atom.GetDegree()!=1 or atom.GetTotalNumHs()!=1:
        raise _NucleosideInput('Phosphate attachment requires one available sugar OH.')
    _consume_hydrogen(atom);previous=anchor;backbone=[];acidic=[]
    for order in range(count):
        phosphorus=_new_atom(editable,Chem,15,owner)
        oxygen=_new_atom(editable,Chem,8,owner)
        editable.AddBond(previous,phosphorus,Chem.BondType.SINGLE)
        editable.AddBond(phosphorus,oxygen,Chem.BondType.DOUBLE)
        backbone.append(phosphorus)
        acid=_new_atom(editable,Chem,16 if thio else 8,owner)
        editable.AddBond(phosphorus,acid,Chem.BondType.SINGLE);acidic.append(acid)
        if order==count-1:
            acid=_new_atom(editable,Chem,8,owner)
            editable.AddBond(phosphorus,acid,Chem.BondType.SINGLE);acidic.append(acid)
        else:
            bridge=_new_atom(editable,Chem,{'oxygen':8,'sulfur':16,'methylene':6}[mode],owner)
            editable.AddBond(phosphorus,bridge,Chem.BondType.SINGLE)
            backbone.append(bridge);previous=bridge
    editable.UpdatePropertyCache(strict=False)
    return tuple(backbone),tuple(acidic)


def _chain_word(count,mode):
    word={2:'diphosphate',3:'triphosphate'}[count]
    if mode!='oxygen':
        locants=','.join(str(2*n) for n in range(1,count))
        replacement=('di' if count==3 else '')+('thio' if mode=='sulfur' else 'carba')
        word=locants+'-'+replacement+word
    return ('trihydrogen' if count==2 else 'tetrahydrogen')+' '+word


def nucleotide_to_iupac_result(spec:NucleotideSpec) -> NucleotideNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        _validate_spec(spec)
        graph,parent,terms=_build_nucleoside(spec.nucleoside,Chem)
        editable=Chem.RWMol(graph);anchor=_role_index(editable,f"O{spec.position}'")
        backbone,acidic=_phosphate_chain(editable,Chem,anchor,spec.phosphate_count,
            spec.bridge_replacement,'phosphate',thio=spec.phosphorothioate)
        graph=editable.GetMol();_validate_graph(graph,Chem)
        acid_h=sum(graph.GetAtomWithIdx(i).GetTotalNumHs() for i in acidic)
        if acid_h!=spec.phosphate_count+1 or any(graph.GetAtomWithIdx(i).GetTotalNumHs()!=1 for i in acidic):
            raise _NucleosideScope('The neutral phosphate chain has not retained its source acid-hydrogen count.')
        position=str(spec.position)+"'"
        if spec.phosphate_count==1:
            if spec.phosphorothioate:terms+=(("P",'thio','P'),)
            name=_prefixes(terms,position+'-'+_PARENTS[parent][2])
        else:
            name=_prefixes(terms,parent)+' '+position+'-('+_chain_word(spec.phosphate_count,spec.bridge_replacement)+')'
        return NucleotideNamingResult(name=name,generation_status='available',preferred_status='not-assessed',
            route='explicit-nucleotide',parent=parent,phosphate_position=spec.position,
            phosphate_count=spec.phosphate_count,phosphate_backbone_maps=tuple(i+1 for i in backbone),
            acidic_atom_maps=tuple(i+1 for i in acidic),acid_hydrogen_count=acid_h,
            bridge_replacement=spec.bridge_replacement,**_serialized_graph(graph,Chem))
    except _NucleosideInput as error:return _failure('invalid-input',str(error))
    except (_NucleosideScope,ValueError,RuntimeError,OverflowError) as error:return _failure('unavailable',str(error))


def nucleotide_to_iupac(spec:NucleotideSpec) -> str:return nucleotide_to_iupac_result(spec).name
