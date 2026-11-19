"""Explicit P-106.3.4/.5 linear 3-prime-to-5-prime oligonucleotides."""
from __future__ import annotations

from dataclasses import dataclass

from .naming_result import NamingResult,SelectionLimitation
from .nucleosides import (
    NucleosideSpec,_PARENTS,_ROLE,_OWNER,_NucleosideInput,_NucleosideScope,
    _build_nucleoside,_prefixes,_role_index,_consume_hydrogen,_validate_graph,_serialized_graph,
)
from .nucleotides import _phosphate_chain


@dataclass(frozen=True,slots=True)
class OligonucleotideSequence:
    residues: tuple[NucleosideSpec|str,...]
    phosphorothioate_links: tuple[int,...] = ()


@dataclass(frozen=True,slots=True)
class NucleosideResidueBinding:
    position: int
    parent: str
    atom_maps: tuple[int,...]
    atom_roles: tuple[tuple[str,int],...]


@dataclass(frozen=True,slots=True)
class PhosphodiesterBinding:
    position: int
    first_sugar_oxygen_map: int
    phosphorus_map: int
    second_sugar_oxygen_map: int
    acidic_atom_map: int
    atom_maps: tuple[int,...]
    phosphorothioate: bool
    phosphorus_configuration: str = 'unspecified'


@dataclass(frozen=True,slots=True)
class OligonucleotideNamingResult(NamingResult):
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    parent_atom_maps: tuple[tuple[str,int],...] = ()
    component_atom_maps: tuple[tuple[str,tuple[int,...]],...] = ()
    residue_bindings: tuple[NucleosideResidueBinding,...] = ()
    phosphodiester_bindings: tuple[PhosphodiesterBinding,...] = ()
    linkage: str = '3-prime-to-5-prime'
    terminal_groups: str = 'declared-nucleoside-termini; no-extra-phosphate'
    substance_form: str = 'conventional-neutral'
    nomenclature: str = 'specialized-P-106; PIN-not-assessed-under-P-100'


def _failure(status,reason):
    return OligonucleotideNamingResult('',status,'not-assessed',
        (SelectionLimitation('oligonucleotide-construction','P-106.3.4/P-106.3.5',reason),),'explicit-oligonucleotide')


def _validate_sequence(sequence):
    if (not isinstance(sequence,OligonucleotideSequence) or not isinstance(sequence.residues,tuple)
            or not 2<=len(sequence.residues)<=1000
            or not isinstance(sequence.phosphorothioate_links,tuple)
            or any(type(i) is not int or not 1<=i<len(sequence.residues) for i in sequence.phosphorothioate_links)
            or len(set(sequence.phosphorothioate_links))!=len(sequence.phosphorothioate_links)):
        raise _NucleosideInput('An immutable ordered sequence of 2..1000 nucleosides and unique internal link positions is required.')
    specs=[]
    for residue in sequence.residues:
        if isinstance(residue,str):residue=NucleosideSpec(residue)
        if not isinstance(residue,NucleosideSpec):raise _NucleosideInput('Every residue must declare a retained nucleoside and its explicit operations.')
        specs.append(residue)
    return tuple(specs)


def _owned_piece(graph,position,Chem):
    piece=Chem.Mol(graph)
    for atom in piece.GetAtoms():
        if atom.HasProp(_ROLE):atom.SetProp(_ROLE,str(position)+':'+atom.GetProp(_ROLE))
        atom.SetProp(_OWNER,f'nucleoside:{position}:'+atom.GetProp(_OWNER))
    return piece


def oligonucleotide_to_iupac_result(sequence:OligonucleotideSequence) -> OligonucleotideNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        specs=_validate_sequence(sequence);pieces=[];parents=[];words=[]
        for i,spec in enumerate(specs,1):
            graph,parent,terms=_build_nucleoside(spec,Chem)
            if i in sequence.phosphorothioate_links:terms+=(("P",'thio','P'),)
            words.append(_prefixes(terms,_PARENTS[parent][3] if i<len(specs) else parent))
            pieces.append(_owned_piece(graph,i,Chem));parents.append(parent)
        editable=Chem.RWMol(pieces[0]);links=[]
        for i,piece in enumerate(pieces[1:],1):
            first=_role_index(editable,f"{i}:O3'")
            thio=i in sequence.phosphorothioate_links;owner='link:'+str(i)
            backbone,acidic=_phosphate_chain(editable,Chem,first,1,'oxygen',owner,thio=thio)
            phosphorus=backbone[0]
            # Retain the original OH oxygen from each sugar. The final newly
            # appended P-OH is the leaving oxygen in the condensation.
            if acidic[-1]!=editable.GetNumAtoms()-1:raise _NucleosideScope('The phosphate leaving oxygen is not mapped safely.')
            editable.RemoveAtom(acidic[-1]);offset=editable.GetNumAtoms()
            second=offset+_role_index(piece,f"{i+1}:O5'")
            editable.InsertMol(piece);host=editable.GetAtomWithIdx(second)
            if host.GetAtomicNum()!=8 or host.GetDegree()!=1 or host.GetTotalNumHs()!=1:
                raise _NucleosideInput('Each next nucleoside must retain an available 5-prime OH.')
            _consume_hydrogen(host);editable.AddBond(phosphorus,second,Chem.BondType.SINGLE)
            editable.UpdatePropertyCache(strict=False)
            links.append((i,first,phosphorus,second,acidic[0],thio))
        graph=editable.GetMol();_validate_graph(graph,Chem)
        data=_serialized_graph(graph,Chem);groups=dict(data['component_atom_maps']);residues=[];bindings=[]
        for i,parent in enumerate(parents,1):
            atom_maps=tuple(number for key,numbers in data['component_atom_maps'] if key.startswith(f'nucleoside:{i}:') for number in numbers)
            atom_roles=tuple((key.split(':',1)[1],number) for key,number in data['parent_atom_maps'] if key.startswith(str(i)+':'))
            residues.append(NucleosideResidueBinding(i,parent,tuple(sorted(atom_maps)),atom_roles))
        for i,first,p,second,acid,thio in links:
            if graph.GetAtomWithIdx(acid).GetTotalNumHs()!=1:
                raise _NucleosideScope('A phosphodiester has not retained its one acid hydrogen.')
            bindings.append(PhosphodiesterBinding(i,first+1,p+1,second+1,acid+1,groups['link:'+str(i)],thio))
        return OligonucleotideNamingResult(name="-(3'→5')-".join(words),generation_status='available',
            preferred_status='not-assessed',route='explicit-oligonucleotide',residue_bindings=tuple(residues),
            phosphodiester_bindings=tuple(bindings),**data)
    except _NucleosideInput as error:return _failure('invalid-input',str(error))
    except (_NucleosideScope,ValueError,RuntimeError,OverflowError) as error:return _failure('unavailable',str(error))


def oligonucleotide_to_iupac(sequence:OligonucleotideSequence) -> str:return oligonucleotide_to_iupac_result(sequence).name
