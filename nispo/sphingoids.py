"""Productive C18 sphinganine unsaturation and N/O-acyl operations, P-107.4.3."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .naming_result import NamingResult,SelectionLimitation
from .lipids import _OWNER,_LipidInput,_LipidScope,_acid_graph,_original_acyl_parts


@dataclass(frozen=True,slots=True)
class SphingoidDoubleBond:
    position: int
    configuration: str = 'E'


@dataclass(frozen=True,slots=True)
class SphingoidOAcyl:
    position: int
    acid_smiles: str


@dataclass(frozen=True,slots=True)
class SphingoidSpec:
    double_bonds: tuple[SphingoidDoubleBond,...] = ()
    n_acyl_acid: str|None = None
    o_acyl_groups: tuple[SphingoidOAcyl,...] = ()


@dataclass(frozen=True,slots=True)
class SphingoidAcylBinding:
    site: str
    carbonyl_carbon_map: int
    sphingoid_heteroatom_map: int
    atom_maps: tuple[int,...]
    acyl_name: str


@dataclass(frozen=True,slots=True)
class SphingoidNamingResult(NamingResult):
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    parent_atom_maps: tuple[tuple[str,int],...] = ()
    acyl_bindings: tuple[SphingoidAcylBinding,...] = ()
    double_bond_configurations: tuple[tuple[int,str],...] = ()
    parent_configuration: str = 'C18-sphinganine-(2S,3R)'
    substance_form: str = 'conventional-neutral'
    nomenclature: str = 'specialized-P-107; PIN-not-assessed-under-P-100'


def _failure(status,reason):
    return SphingoidNamingResult('',status,'not-assessed',
        (SelectionLimitation('sphingoid-construction','P-107.4.3.1/P-107.4.3.2',reason),),'explicit-sphingoid')


def _validate(spec):
    if (not isinstance(spec,SphingoidSpec) or not isinstance(spec.double_bonds,tuple)
            or not isinstance(spec.o_acyl_groups,tuple)
            or spec.n_acyl_acid is not None and (not isinstance(spec.n_acyl_acid,str) or not spec.n_acyl_acid)):
        raise _LipidInput('Declare immutable C18 double-bond/O-acyl operations and an optional N-acyl acid graph.')
    positions=[]
    for item in spec.double_bonds:
        if (not isinstance(item,SphingoidDoubleBond) or type(item.position) is not int or not 4<=item.position<=17
                or not isinstance(item.configuration,str) or item.configuration not in {'E','Z','unspecified'}
                or item.position==17 and item.configuration!='unspecified'):
            raise _LipidInput('Unsaturation requires a 4..17 bond; internal E/Z or unspecified, terminal 17 unspecified only.')
        positions.append(item.position)
    positions.sort()
    if any(b-a<2 for a,b in zip(positions,positions[1:])):
        raise _LipidInput('Duplicate or cumulated unsaturation is outside the sphinganine operation.')
    sites=[]
    for item in spec.o_acyl_groups:
        if (not isinstance(item,SphingoidOAcyl) or type(item.position) is not int or item.position not in {1,3}
                or not isinstance(item.acid_smiles,str) or not item.acid_smiles):
            raise _LipidInput('Each O-acyl operation requires one available 1/3 OH and an acid graph.')
        sites.append(item.position)
    if len(sites)!=len(set(sites)):raise _LipidInput('Two O-acyl operations cannot consume the same OH.')


def _parent(Chem,bonds):
    graph=Chem.RWMol()
    for _ in range(18):graph.AddAtom(Chem.Atom(6))
    for i in range(17):graph.AddBond(i,i+1,Chem.BondType.SINGLE)
    o1=graph.AddAtom(Chem.Atom(8));n2=graph.AddAtom(Chem.Atom(7));o3=graph.AddAtom(Chem.Atom(8))
    for c,x in ((0,o1),(1,n2),(2,o3)):graph.AddBond(c,x,Chem.BondType.SINGLE)
    for item in bonds:
        graph.GetBondBetweenAtoms(item.position-1,item.position).SetBondType(Chem.BondType.DOUBLE)
    graph=graph.GetMol();Chem.SanitizeMol(graph)
    for item in bonds:
        if item.configuration=='unspecified':continue
        bond=graph.GetBondBetweenAtoms(item.position-1,item.position)
        bond.SetStereoAtoms(item.position-2,item.position+1)
        bond.SetStereo(Chem.BondStereo.STEREOE if item.configuration=='E' else Chem.BondStereo.STEREOZ)
    Chem.SetDoubleBondNeighborDirections(graph)
    from rdkit.Chem import rdCIPLabeler
    for first in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
        for second in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            graph.GetAtomWithIdx(1).SetChiralTag(first);graph.GetAtomWithIdx(2).SetChiralTag(second)
            Chem.AssignStereochemistry(graph,cleanIt=True,force=True)
            rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
            if tuple(graph.GetAtomWithIdx(i).GetProp('_CIPCode') for i in (1,2))==('S','R'):
                for atom in graph.GetAtoms():atom.SetProp(_OWNER,'sphingoid')
                return graph,{'N':n2,'O1':o1,'O3':o3}
    raise _LipidScope('The source sphinganine parent configuration could not be established.')


def _word(parts,bonds,*,extra_prefixes=()):
    from .constants import MULT
    from .formatting import prefix_alphabetization_key,enclose_name_unit
    grouped=defaultdict(list);markers=defaultdict(set)
    for site,_center,_host,word,_owned in parts:
        marker='N' if site=='N' else 'O';grouped[(word,marker)].append(site);markers[word].add(marker)
    if any(len(values)>1 for values in markers.values()):
        raise _LipidScope('Identical-word N/O acyl composition needs a proved joint citation.')
    prefixes=[]
    for (word,marker),sites in sorted(grouped.items(),key=lambda item:prefix_alphabetization_key(item[0][0])):
        rendered=enclose_name_unit(word) if '-' in word or word[:1]=='(' else word
        if marker=='N':prefixes.append((prefix_alphabetization_key(word),'N-'+rendered))
        else:
            locants=','.join(str(int(site[1:])) for site in sorted(sites))
            prefixes.append((prefix_alphabetization_key(word),locants+'-'+('di-O-' if len(sites)==2 else 'O-')+rendered))
    ordered=tuple(sorted(bonds,key=lambda b:b.position));count=len(ordered)
    noun=('sphinganine' if not count else 'sphing-'+str(ordered[0].position)+'-enine' if count==1
          else 'sphinga-'+','.join(str(b.position) for b in ordered)+'-'+MULT[count]+'enine')
    prefixes.extend(extra_prefixes)
    prefix='-'.join(text for _key,text in sorted(prefixes,key=lambda item:item[0]))
    descriptors=','.join(str(b.position)+b.configuration for b in ordered if b.configuration!='unspecified')
    return ('('+descriptors+')-' if descriptors else '')+prefix+noun


def sphingoid_to_iupac_result(spec:SphingoidSpec) -> SphingoidNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        _validate(spec);graph,hosts=_parent(Chem,spec.double_bonds);editable=Chem.RWMol(graph);bindings=[]
        operations=([('N',spec.n_acyl_acid)] if spec.n_acyl_acid is not None else [])+[
            ('O'+str(item.position),item.acid_smiles) for item in sorted(spec.o_acyl_groups,key=lambda item:item.position)]
        for site,smiles in operations:
            acid,center,leaving=_acid_graph(smiles,Chem);fragment=Chem.RWMol(acid)
            fragment.RemoveAtom(leaving);center-=leaving<center;offset=editable.GetNumAtoms()
            for atom in fragment.GetAtoms():atom.SetProp(_OWNER,'acyl:'+site)
            editable.InsertMol(fragment.GetMol());editable.AddBond(hosts[site],offset+center,Chem.BondType.SINGLE)
            editable.UpdatePropertyCache(strict=False);bindings.append((site,offset+center,hosts[site]))
        graph=editable.GetMol();Chem.SanitizeMol(graph);Chem.AssignStereochemistry(graph,cleanIt=True,force=True)
        from rdkit.Chem import rdCIPLabeler
        rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
        parts=_original_acyl_parts(graph,Chem,bindings);name=_word(parts,spec.double_bonds)
        identity=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
        for atom in graph.GetAtoms():atom.SetAtomMapNum(atom.GetIdx()+1)
        roles=tuple([(str(i+1),i+1) for i in range(18)]+[(key,idx+1) for key,idx in sorted(hosts.items())])
        acyls=tuple(SphingoidAcylBinding(site,c+1,host+1,tuple(i+1 for i in sorted(owned)),word)
                    for site,c,host,word,owned in parts)
        return SphingoidNamingResult(name=name,generation_status='available',preferred_status='not-assessed',
            route='explicit-sphingoid',canonical_smiles=identity,mapped_smiles=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True),
            parent_atom_maps=roles,acyl_bindings=acyls,double_bond_configurations=tuple(sorted((b.position,b.configuration) for b in spec.double_bonds)))
    except _LipidInput as error:return _failure('invalid-input',str(error))
    except (_LipidScope,ValueError,RuntimeError,OverflowError) as error:return _failure('unavailable',str(error))


def sphingoid_to_iupac(spec:SphingoidSpec) -> str:return sphingoid_to_iupac_result(spec).name
