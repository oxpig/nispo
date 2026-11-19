"""Explicit whole-molecule P-93.1.2 relative and P-93.1.3 racemic input.

A relative-configuration declaration describes one enantiomer OR the other.
A racemate declares equal amounts of both. Neither interpretation is inferred
from an unspecified atom or from a single absolute-configurational SMILES.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Literal

from .naming_result import NamingResult, SelectionLimitation


@dataclass(frozen=True, slots=True)
class StereoEnsemble:
    representative_smiles: str
    interpretation: Literal["relative", "racemic"]


@dataclass(frozen=True, slots=True)
class StereoEnsembleNamingResult(NamingResult):
    interpretation: str = ""
    member_smiles: tuple[str, ...] = ()
    mole_fractions: tuple[Fraction, ...] = ()
    membership: str = ""


def _failure(status, reason, rule="P-93.1.2/P-93.1.3"):
    return StereoEnsembleNamingResult("",status,"not-assessed" if status=="invalid-input" else "unresolved",
        (SelectionLimitation("stereo-ensemble",rule,reason),),"explicit-stereo-ensemble")


def _leading_configuration(name, count):
    match=re.fullmatch(r"\(([^()]*)\)-(.+)",name)
    if match is None:return None
    pieces=[]
    for part in match[1].split(','):
        descriptor=re.fullmatch(r"(\d+[a-z]?'*|)([RSrsEZ])",part)
        if descriptor is None:return None
        pieces.append((descriptor[1],descriptor[2]))
    atom_parts=tuple((loc,code) for loc,code in pieces if code in 'RSrs')
    if len(atom_parts)!=count or any(not loc for loc,_ in atom_parts) and count!=1:
        return None
    # Embedded components need their own whole-ensemble numbering operation;
    # counting an internal descriptor as an outer one is not a mapping proof.
    if re.search(r"\((?:\d+[a-z]?'*)?[RSrs](?:,|\))",match[2]):return None
    from .parent_selection import locant_key
    chiral=sorted(((loc,code) for loc,code in atom_parts if code in 'RS'),
                 key=lambda pair:locant_key(pair[0]) if pair[0] else ())
    return (match[2],tuple(pieces),chiral[0][1] if chiral else None)


def stereo_ensemble_to_iupac_result(ensemble: StereoEnsemble) -> StereoEnsembleNamingResult:
    """Name an explicitly declared whole-molecule relative or racemic pair.

    The initial operation supports complete ordinary tetrahedral assignment
    and assigned ordinary E/Z. Mixed absolute/relative subsets, embedded
    descriptor groups, other stereogenic units and ambo need separate input
    and writing operations and return an explicit limitation.
    """
    from .api import Chem, mol_to_iupac_result
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    if (not isinstance(ensemble,StereoEnsemble)
            or ensemble.interpretation not in {'relative','racemic'}
            or not isinstance(ensemble.representative_smiles,str)):
        return _failure('invalid-input','A StereoEnsemble with an explicit relative or racemic interpretation is required.')
    try:mol=Chem.MolFromSmiles(ensemble.representative_smiles)
    except (TypeError,ValueError):mol=None
    if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol))!=1:
        return _failure('invalid-input','A connected representative molecular structure is required.')
    if mol.GetStereoGroups():
        return _failure('unavailable','Enhanced or mixed stereo groups require a separate subgroup operation.')
    tetra={Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    if any(a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED and a.GetChiralTag() not in tetra
           for a in mol.GetAtoms()):
        return _failure('unavailable','A non-tetrahedral stereogenic unit is outside this operation.')
    for feature in Chem.FindPotentialStereo(mol):
        if str(feature.type) not in {'Atom_Tetrahedral','Bond_Double'}:
            return _failure('unavailable','A nonordinary stereogenic unit requires another operation.')
        if str(feature.specified)!='Specified':
            return _failure('invalid-input','Every potential stereogenic unit in the representative must be assigned.')
    for atom in mol.GetAtoms():atom.SetAtomMapNum(0)
    from .stereochemistry import assigned_atom_stereochemistry
    assigned=assigned_atom_stereochemistry(mol,Chem,include_pseudo=True)
    if not assigned or not any(code in {'R','S'} for _,code in assigned):
        return _failure('invalid-input','The representative has no assigned ordinary chiral configuration.')
    mirror=Chem.Mol(mol)
    for atom in mirror.GetAtoms():
        if atom.GetChiralTag() in tetra:atom.InvertChirality()
    Chem.AssignStereochemistry(mirror,cleanIt=True,force=True)
    mirrored=dict(assigned_atom_stereochemistry(mirror,Chem,include_pseudo=True))
    reflection={'R':'S','S':'R','r':'r','s':'s'}
    if (set(mirrored)!={idx for idx,_ in assigned}
            or any(mirrored.get(idx)!=reflection[code] for idx,code in assigned)):
        return _failure('unavailable','Complete original-graph CIP reflection is not established.')
    identities=tuple(Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
                     for graph in (mol,mirror))
    if identities[0]==identities[1]:
        return _failure('invalid-input','An achiral or meso structure does not define a distinct enantiomeric pair.')
    # This operation has an explicit whole-ensemble interpretation and has
    # just proved both fully specified member graphs.  Name these private
    # representatives individually; the input-level ``r`` marker describes
    # the ensemble and must not be mistaken for an absolute member label.
    from .input_stereochemistry import _source_has_relative_flag
    for graph in (mol, mirror):
        if _source_has_relative_flag(graph):
            graph.ClearProp('_CXSMILES_Data')
    results=tuple(mol_to_iupac_result(graph) for graph in (mol,mirror))
    if any(not result.name for result in results):
        return _failure('unavailable','One enantiomer has no complete molecular naming operation.')
    configurations=tuple(_leading_configuration(result.name,len(assigned)) for result in results)
    if any(c is None for c in configurations):
        return _failure('unavailable','The molecular writer does not supply one complete leading configuration prefix.')
    if configurations[0][0]!=configurations[1][0]:
        return _failure('unavailable','The enantiomers do not share a completed constitutional name and numbering.')
    preferred=[i for i,configuration in enumerate(configurations) if configuration[2]=='R']
    if not preferred:
        return _failure('unavailable','The R descriptor at the lowest chirality-centre locant is unproved.')
    issues=[]
    for result in results:
        for issue in result.unresolved:
            if issue not in issues:issues.append(issue)
    if len(preferred)>1:
        issues.append(SelectionLimitation('stereo-ensemble','P-93.1.2/P-93.1.3',
                      'More than one numbering retains R at the lowest chirality-centre locant.'))
    chosen=min(preferred,key=lambda index:results[index].name)
    other=1-chosen
    prefix='rel' if ensemble.interpretation=='relative' else 'rac'
    name=prefix+'-'+results[chosen].name
    status=('unresolved' if issues else 'resolved' if all(r.preferred_status=='resolved' for r in results)
            else 'not-assessed')
    return StereoEnsembleNamingResult(
        name=name, generation_status='available', preferred_status=status,
        unresolved=tuple(issues), route='explicit-stereo-ensemble',
        interpretation=ensemble.interpretation,
        member_smiles=(identities[chosen],identities[other]),
        mole_fractions=(Fraction(1,2),Fraction(1,2)) if ensemble.interpretation=='racemic' else (),
        membership='or-enantiomer' if ensemble.interpretation=='relative' else 'and-enantiomer')


def stereo_ensemble_to_iupac(ensemble: StereoEnsemble) -> str:
    """String counterpart of :func:`stereo_ensemble_to_iupac_result`."""
    return stereo_ensemble_to_iupac_result(ensemble).name
