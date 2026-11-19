"""Productive, explicitly declared P-105 beta-D nucleoside modifications."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import re

from .naming_result import NamingResult,SelectionLimitation


_PARENTS={
    'adenosine':('A',2,'adenylic acid','adenylyl'),
    'guanosine':('G',2,'guanylic acid','guanylyl'),
    'cytidine':('C',2,'cytidylic acid','cytidylyl'),
    'uridine':('U',2,'uridylic acid','uridylyl'),
    'thymidine':('T',6,'thymidylic acid','thymidylyl'),
}
_PARENT_ALIASES={key:key for key in _PARENTS}
_PARENT_ALIASES.update({value[0].lower():key for key,value in _PARENTS.items()})
_ROLE='_nispo_nucleoside_parent_role'
_OWNER='_nispo_nucleoside_owner'
_HALOGENS={'fluoro':9,'chloro':17,'bromo':35,'iodo':53}


@dataclass(frozen=True,slots=True)
class SugarModification:
    position: int
    kind: str


@dataclass(frozen=True,slots=True)
class BaseModification:
    position: int|str
    kind: str


@dataclass(frozen=True,slots=True)
class NucleosideSpec:
    parent: str
    sugar_modifications: tuple[SugarModification,...] = ()
    base_modifications: tuple[BaseModification,...] = ()


@dataclass(frozen=True,slots=True)
class NucleosideNamingResult(NamingResult):
    parent: str = ''
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    parent_atom_maps: tuple[tuple[str,int],...] = ()
    component_atom_maps: tuple[tuple[str,tuple[int,...]],...] = ()
    sugar_configuration: str = 'beta-D'
    substance_form: str = 'conventional-neutral'
    nomenclature: str = 'specialized-P-105/P-106; PIN-not-assessed-under-P-100'


class _NucleosideScope(ValueError):pass
class _NucleosideInput(ValueError):pass


def _failure(status,reason):
    return NucleosideNamingResult('',status,'not-assessed',
        (SelectionLimitation('nucleoside-construction','P-105.2',reason),),'explicit-nucleoside')


def _locant_key(text):
    match=re.fullmatch(r'(N)?(\d+)(\'?)',str(text))
    if match is None:return (2,0,str(text))
    return (0 if match[1] else 1,int(match[2]),1 if match[3] else 0)


def _prefixes(terms,parent):
    """Cite source-typed sugar O and base/functional-replacement prefixes."""
    grouped=defaultdict(list);word_markers=defaultdict(set)
    for locant,word,marker in terms:
        grouped[(word,marker)].append(locant);word_markers[word].add(marker)
    if any(len(markers)>1 for markers in word_markers.values()):
        raise _NucleosideScope('Mixed attachment classes of one substituent need a proved joint prefix citation.')
    from .constants import MULT
    pieces=[]
    for (word,marker),locants in sorted(grouped.items(),key=lambda item:(item[0][0],item[0][1])):
        values=sorted(locants,key=_locant_key);count=len(values)
        if count>1 and count not in MULT:raise _NucleosideScope('The prefix multiplicity is outside the supported vocabulary.')
        multiplier=MULT[count] if count>1 else ''
        if marker=='O':unit=multiplier+'-O-'+word if multiplier else 'O-'+word
        elif marker=='P':unit=multiplier+'-P-'+word if multiplier else 'P-'+word
        else:unit=multiplier+word
        pieces.append(','.join(values)+'-'+unit if marker!='P' else unit)
    prefix='-'.join(pieces)
    return prefix+('-' if prefix and parent[:1].isdigit() else '')+parent


def _role_index(mol,role):
    indices=[a.GetIdx() for a in mol.GetAtoms() if a.HasProp(_ROLE) and a.GetProp(_ROLE)==role]
    if len(indices)!=1:raise _NucleosideInput('The requested parent atom/site is absent or ambiguous: '+role)
    return indices[0]


def _role_atom(mol,role):return mol.GetAtomWithIdx(_role_index(mol,role))


def _consume_hydrogen(atom):
    if atom.GetTotalNumHs(includeNeighbors=True)<1:
        raise _NucleosideInput('The requested substitution site has no replaceable hydrogen.')
    if atom.GetNumExplicitHs():atom.SetNumExplicitHs(atom.GetNumExplicitHs()-1)


def _new_atom(editable,Chem,number,owner):
    atom=Chem.Atom(number);atom.SetProp(_OWNER,owner)
    return editable.AddAtom(atom)


def _append_alkyl(editable,Chem,host,count,owner):
    _consume_hydrogen(editable.GetAtomWithIdx(host));previous=host
    for _ in range(count):
        carbon=_new_atom(editable,Chem,6,owner);editable.AddBond(previous,carbon,Chem.BondType.SINGLE);previous=carbon
    editable.UpdatePropertyCache(strict=False)


@lru_cache(maxsize=5)
def _parent_template(parent):
    from rdkit import Chem
    code,flavor,_acid,_yl=_PARENTS[parent]
    graph=Chem.MolFromSequence(code,flavor=flavor)
    if graph is None:raise _NucleosideScope('The standard beta-D nucleoside scaffold is unavailable.')
    names=[]
    for atom in graph.GetAtoms():
        info=atom.GetPDBResidueInfo()
        if info is None:raise _NucleosideScope('The scaffold has no source atom-role map.')
        role=info.GetName().strip().replace('*',"'")
        atom.SetProp(_ROLE,role);atom.SetProp(_OWNER,'parent');names.append(role)
    if len(names)!=len(set(names)):raise _NucleosideScope('Parent atom roles are not unique.')
    for a,b in (("C1'","C2'"),("C2'","C3'"),("C3'","C4'"),("C4'","O4'"),
                ("O4'","C1'"),("C4'","C5'"),("C5'","O5'"),("C3'","O3'"),
                ("C1'",'N9' if parent in {'adenosine','guanosine'} else 'N1')):
        bond=graph.GetBondBetweenAtoms(_role_index(graph,a),_role_index(graph,b))
        if bond is None or bond.GetBondType()!=Chem.BondType.SINGLE:
            raise _NucleosideScope('The scaffold does not prove the retained sugar/glycosidic structure.')
    if parent!='thymidine':
        bond=graph.GetBondBetweenAtoms(_role_index(graph,"C2'"),_role_index(graph,"O2'"))
        if bond is None or bond.GetBondType()!=Chem.BondType.SINGLE:raise _NucleosideScope('The ribose 2-prime OH is unavailable.')
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in graph.GetAtoms()):
        raise _NucleosideScope('The scaffold is not the conventional neutral retained parent.')
    if any(_role_atom(graph,key).GetChiralTag()==Chem.ChiralType.CHI_UNSPECIFIED
           for key in (("C1'","C3'","C4'") if parent=='thymidine' else ("C1'","C2'","C3'","C4'"))):
        raise _NucleosideScope('The scaffold does not retain its assigned sugar configuration.')
    return graph


def _normalized_spec(spec):
    if (not isinstance(spec,NucleosideSpec) or not isinstance(spec.parent,str)
            or spec.parent.lower() not in _PARENT_ALIASES
            or not isinstance(spec.sugar_modifications,tuple) or not isinstance(spec.base_modifications,tuple)):
        raise _NucleosideInput('A common retained beta-D parent and immutable explicit modifications are required.')
    parent=_PARENT_ALIASES[spec.parent.lower()]
    positions=[]
    for mod in spec.sugar_modifications:
        if (not isinstance(mod,SugarModification) or type(mod.position) is not int or mod.position not in {2,3,5}
                or not isinstance(mod.kind,str) or mod.kind not in {'deoxy','fluoro','O-methyl','O-acetyl'}):
            raise _NucleosideInput('Sugar operations require an available 2/3/5-prime site and a supported operation.')
        positions.append(mod.position)
    if len(set(positions))!=len(positions):raise _NucleosideInput('Two operations cannot consume the same sugar OH.')
    for mod in spec.base_modifications:
        if (not isinstance(mod,BaseModification) or not isinstance(mod.kind,str)
                or mod.kind not in {'methyl','ethyl','thio',*_HALOGENS}
                or not (type(mod.position) is int and 1<=mod.position<=9
                        or isinstance(mod.position,str) and re.fullmatch(r'N[246]',mod.position))):
            raise _NucleosideInput('Base operations need a numeric ring or N2/N4/N6 exocyclic locant and a supported substitution.')
    return parent


def _build_nucleoside(spec,Chem):
    parent=_normalized_spec(spec);editable=Chem.RWMol(_parent_template(parent));terms=[]
    for order,mod in enumerate(sorted(spec.sugar_modifications,key=lambda item:(item.position,item.kind))):
        role=f"O{mod.position}'";idx=_role_index(editable,role);atom=editable.GetAtomWithIdx(idx)
        if atom.GetAtomicNum()!=8 or atom.GetDegree()!=1 or atom.GetTotalNumHs()!=1:
            raise _NucleosideInput('The sugar site is not one available OH.')
        owner=f'sugar:{mod.position}:{mod.kind}:{order}'
        if mod.kind=='deoxy':
            host=atom.GetNeighbors()[0];host.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            if host.GetNoImplicit():host.SetNumExplicitHs(host.GetNumExplicitHs()+1)
            editable.RemoveAtom(idx);terms.append((str(mod.position)+"'",'deoxy',''))
        elif mod.kind=='fluoro':
            atom.SetAtomicNum(9);atom.SetNumExplicitHs(0);atom.SetNoImplicit(True)
            terms.extend(((str(mod.position)+"'",'deoxy',''),(str(mod.position)+"'",'fluoro','')))
        elif mod.kind=='O-methyl':
            _append_alkyl(editable,Chem,idx,1,owner);terms.append((str(mod.position)+"'",'methyl','O'))
        elif mod.kind=='O-acetyl':
            _consume_hydrogen(atom);carbon=_new_atom(editable,Chem,6,owner)
            oxygen=_new_atom(editable,Chem,8,owner);methyl=_new_atom(editable,Chem,6,owner)
            editable.AddBond(idx,carbon,Chem.BondType.SINGLE);editable.AddBond(carbon,oxygen,Chem.BondType.DOUBLE)
            editable.AddBond(carbon,methyl,Chem.BondType.SINGLE);terms.append((str(mod.position)+"'",'acetyl','O'))
        editable.UpdatePropertyCache(strict=False)
    for order,mod in enumerate(sorted(spec.base_modifications,key=lambda item:(_locant_key(item.position),item.kind))):
        loc=str(mod.position);owner=f'base:{loc}:{mod.kind}:{order}'
        if mod.kind=='thio':
            if type(mod.position) is not int:raise _NucleosideInput('Functional thio replacement needs a carbonyl locant.')
            atom=_role_atom(editable,'O'+loc)
            if atom.GetAtomicNum()!=8 or atom.GetDegree()!=1 or atom.GetBonds()[0].GetBondType()!=Chem.BondType.DOUBLE:
                raise _NucleosideInput('There is no source carbonyl oxygen at that base site.')
            atom.SetAtomicNum(16);terms.append((loc,'thio',''))
        else:
            if isinstance(mod.position,str):
                idx=_role_index(editable,loc);host=editable.GetAtomWithIdx(idx)
                if host.GetAtomicNum()!=7 or host.IsInRing():raise _NucleosideInput('An exocyclic N locant is required here.')
            else:
                matches=[a.GetIdx() for a in editable.GetAtoms() if a.IsInRing() and a.HasProp(_ROLE)
                    and a.GetProp(_ROLE) in {'C'+loc,'N'+loc}]
                if len(matches)!=1:raise _NucleosideInput('The numeric base ring atom is unavailable.')
                idx=matches[0];host=editable.GetAtomWithIdx(idx)
            if mod.kind in {'methyl','ethyl'}:
                _append_alkyl(editable,Chem,idx,1 if mod.kind=='methyl' else 2,owner)
            else:
                if host.GetAtomicNum()!=6:raise _NucleosideScope('N-halogenation requires a separate characteristic-class comparison.')
                _consume_hydrogen(host);halogen=_new_atom(editable,Chem,_HALOGENS[mod.kind],owner)
                editable.AddBond(idx,halogen,Chem.BondType.SINGLE)
            terms.append((loc,mod.kind,''))
        editable.UpdatePropertyCache(strict=False)
    graph=editable.GetMol();_validate_graph(graph,Chem)
    # Fail before delivery when a requested prefix composition is unproved.
    _prefixes(terms,parent)
    return graph,parent,tuple(terms)


def _validate_graph(graph,Chem):
    try:
        Chem.SanitizeMol(graph);Chem.AssignStereochemistry(graph,cleanIt=True,force=True)
        from rdkit.Chem import rdCIPLabeler
        rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
    except (ValueError,RuntimeError,OverflowError) as error:
        raise _NucleosideScope('The declared modifications do not complete valence/configuration validation.') from error
    if len(Chem.GetMolFrags(graph))!=1 or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in graph.GetAtoms()):
        raise _NucleosideScope('A connected conventional-neutral construction was not retained.')


def _serialized_graph(graph,Chem):
    identity=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True);groups=defaultdict(list);roles=[]
    for atom in graph.GetAtoms():
        number=atom.GetIdx()+1;atom.SetAtomMapNum(number)
        if not atom.HasProp(_OWNER):raise _NucleosideScope('An original atom has no mapped ownership.')
        groups[atom.GetProp(_OWNER)].append(number)
        if atom.HasProp(_ROLE):roles.append((atom.GetProp(_ROLE),number))
    mapped=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
    return dict(canonical_smiles=identity,mapped_smiles=mapped,
        parent_atom_maps=tuple(sorted(roles)),component_atom_maps=tuple((key,tuple(value)) for key,value in sorted(groups.items())))


def nucleoside_to_iupac_result(spec:NucleosideSpec) -> NucleosideNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        graph,parent,terms=_build_nucleoside(spec,Chem)
        return NucleosideNamingResult(name=_prefixes(terms,parent),generation_status='available',
            preferred_status='not-assessed',route='explicit-nucleoside',parent=parent,**_serialized_graph(graph,Chem))
    except _NucleosideInput as error:return _failure('invalid-input',str(error))
    except (_NucleosideScope,ValueError,RuntimeError,OverflowError) as error:return _failure('unavailable',str(error))


def nucleoside_to_iupac(spec:NucleosideSpec) -> str:return nucleoside_to_iupac_result(spec).name
