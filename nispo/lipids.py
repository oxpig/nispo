"""Productive P-107 O-acyl/O-alkyl glycerol construction in an explicit sn frame."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass,field
from functools import lru_cache

from .naming_result import NamingResult,SelectionLimitation


@dataclass(frozen=True,slots=True)
class GlycerolAcyl:
    position: int
    acid_smiles: str


@dataclass(frozen=True,slots=True)
class GlycerolEther:
    position: int
    alcohol_smiles: str


@dataclass(frozen=True,slots=True)
class GlycerolSpec:
    acyl_groups: tuple[GlycerolAcyl,...] = ()
    configuration: str = 'sn'
    ether_groups: tuple[GlycerolEther,...] = ()


@dataclass(frozen=True,slots=True)
class LipidAcylBinding:
    position: int
    carbonyl_carbon_map: int
    original_glycerol_oxygen_map: int
    atom_maps: tuple[int,...]
    acyl_name: str


@dataclass(frozen=True,slots=True)
class LipidEtherBinding:
    position: int
    attachment_carbon_map: int
    original_glycerol_oxygen_map: int
    atom_maps: tuple[int,...]
    organyl_name: str


@dataclass(frozen=True,slots=True)
class GlycerolNamingResult(NamingResult):
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    glycerol_atom_maps: tuple[tuple[str,int],...] = ()
    acyl_bindings: tuple[LipidAcylBinding,...] = ()
    ether_bindings: tuple[LipidEtherBinding,...] = field(default=(),kw_only=True)
    member_smiles: tuple[str,...] = ()
    proportions: tuple[tuple[int,int],...] = ()
    configuration: str = ''
    substance_form: str = 'conventional-neutral'
    nomenclature: str = 'specialized-P-107; PIN-not-assessed-under-P-100'


class _LipidInput(ValueError):pass
class _LipidScope(ValueError):pass
_OWNER='_nispo_lipid_owner'


def _failure(status,reason):
    return GlycerolNamingResult('',status,'not-assessed',
        (SelectionLimitation('glycerol-construction','P-107.2/P-107.3.2',reason),),'explicit-glycerol')


@lru_cache(maxsize=1)
def _sn_tag():
    """Native parity for neighbours C1, O2, C3 in the source Fischer frame."""
    from rdkit import Chem
    proof=Chem.MolFromSmiles('CC(O)C');conf=Chem.Conformer(4);conf.Set3D(True)
    for idx,point in enumerate(((0,1,-1),(0,0,0),(-1,0,1),(0,-1,-1))):conf.SetAtomPosition(idx,point)
    proof.AddConformer(conf);Chem.AssignAtomChiralTagsFromStructure(proof,replaceExistingTags=True)
    tag=proof.GetAtomWithIdx(1).GetChiralTag()
    if tag==Chem.ChiralType.CHI_UNSPECIFIED:raise _LipidScope('The sn Fischer geometry has no native parity.')
    return tag


def _validate_request(request,*,allow_unacylated=False):
    if (not isinstance(request,GlycerolSpec) or not isinstance(request.acyl_groups,tuple)
            or not isinstance(request.ether_groups,tuple)
            or not (0 if allow_unacylated else 1)<=len(request.acyl_groups)+len(request.ether_groups)<=3 or not isinstance(request.configuration,str)
            or request.configuration not in {'sn','R','S','unspecified','rac'}):
        raise _LipidInput('Declare one to three immutable O-acyl/O-alkyl groups and sn, R/S, unspecified or rac configuration.')
    positions=[]
    for group in request.acyl_groups:
        if (not isinstance(group,GlycerolAcyl) or type(group.position) is not int or group.position not in {1,2,3}
                or not isinstance(group.acid_smiles,str) or not group.acid_smiles):
            raise _LipidInput('Each O-acyl group requires a unique glycerol position and a monocarboxylic acid graph.')
        positions.append(group.position)
    for group in request.ether_groups:
        if (not isinstance(group,GlycerolEther) or type(group.position) is not int or group.position not in {1,2,3}
                or not isinstance(group.alcohol_smiles,str) or not group.alcohol_smiles):
            raise _LipidInput('Each O-alkyl group requires a unique glycerol position and a primary alcohol graph.')
        positions.append(group.position)
    if len(set(positions))!=len(positions):raise _LipidInput('Two operations cannot consume the same glycerol OH.')


def _ether_graph(smiles,Chem):
    graph=Chem.MolFromSmiles(smiles)
    if graph is None or len(Chem.GetMolFrags(graph))!=1:
        raise _LipidInput('Each ether source must be one valid connected primary alcohol graph.')
    if (graph.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
            or a.IsInRing() or int(a.GetChiralTag()) or a.GetAtomicNum() not in {6,8}
            or a.GetIsotope() and a.GetAtomicNum()!=6 for a in graph.GetAtoms())
            or any(b.GetBondType() not in {Chem.BondType.SINGLE,Chem.BondType.DOUBLE}
                   or b.GetStereo() not in {Chem.BondStereo.STEREONONE,Chem.BondStereo.STEREOE,Chem.BondStereo.STEREOZ,Chem.BondStereo.STEREOCIS,Chem.BondStereo.STEREOTRANS}
                   or b.GetBondDir() not in {Chem.BondDir.NONE,Chem.BondDir.ENDUPRIGHT,Chem.BondDir.ENDDOWNRIGHT} for b in graph.GetBonds())):
        raise _LipidScope('The ether port requires one unbranched carbon path with ordinary isolated alkene geometry; branched, cyclic, axial and alkyne ports need separate proofs.')
    oxygens=[a for a in graph.GetAtoms() if a.GetAtomicNum()==8]
    if (len(oxygens)!=1 or oxygens[0].GetDegree()!=1 or oxygens[0].GetTotalNumHs()!=1
            or oxygens[0].GetBonds()[0].GetBondType()!=Chem.BondType.SINGLE):
        raise _LipidInput('The ether source requires exactly one alcoholic OH oxygen.')
    oxygen=oxygens[0];root=oxygen.GetNeighbors()[0]
    carbons=[a for a in graph.GetAtoms() if a.GetAtomicNum()==6]
    if (not 1<=len(carbons)<=36 or root.GetAtomicNum()!=6
            or sum(n.GetAtomicNum()==6 for n in root.GetNeighbors())>1
            or any(sum(n.GetAtomicNum()==6 for n in a.GetNeighbors())>2 for a in carbons)):
        raise _LipidScope('The declared ether residue is outside the C1-C36 terminal hydroxy carbon-path scope.')
    if any(sum(b.GetBondType()==Chem.BondType.DOUBLE for b in a.GetBonds())>1 for a in carbons):
        raise _LipidScope('Cumulative double bonds require a separate axial/port configuration proof.')
    directed=set()
    for bond in graph.GetBonds():
        if bond.GetStereo()!=Chem.BondStereo.STEREONONE:
            ends=(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx());ligands=tuple(bond.GetStereoAtoms())
            if bond.GetBondType()!=Chem.BondType.DOUBLE or len(ligands)!=2:
                raise _LipidScope('The original alkene has no complete ordered ligand pair.')
            for i,j in zip(ends,ligands,strict=True):
                adjacent=graph.GetBondBetweenAtoms(i,j)
                if adjacent is None:raise _LipidScope('An original alkene ligand is not adjacent to its endpoint.')
                directed.add(adjacent.GetIdx())
    if any(int(b.GetBondDir()) and b.GetIdx() not in directed for b in graph.GetBonds()):
        raise _LipidScope('A directed bond lies outside a proved ordinary alkene unit.')
    if any(a.GetIsotope() and Chem.GetPeriodicTable().GetMassForIsotope(6,a.GetIsotope())<=0 for a in carbons):
        raise _LipidScope('An ether carbon isotope has no supported nuclide identity.')
    for atom in graph.GetAtoms():atom.SetAtomMapNum(0)
    return graph,root.GetIdx(),oxygen.GetIdx()


def _join_ether(graph,alcohol,root,leaving,oxygen,position,Chem):
    # Removing the original O can clear a C=C's stereo atoms at a vinyl
    # port. Transfer the actual ordered ligand references, not just E/Z text.
    stereo=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx(),tuple(b.GetStereoAtoms()),b.GetStereo())
            for b in alcohol.GetBonds() if b.GetStereo()!=Chem.BondStereo.STEREONONE]
    piece=Chem.RWMol(alcohol);piece.RemoveAtom(leaving);offset=graph.GetNumAtoms()
    for atom in piece.GetAtoms():atom.SetProp(_OWNER,'ether:'+str(position))
    graph.InsertMol(piece.GetMol());port=offset+root-int(root>leaving)
    graph.AddBond(oxygen,port,Chem.BondType.SINGLE);graph.UpdatePropertyCache(strict=False)
    def mapped(i):return oxygen if i==leaving else offset+i-int(i>leaving)
    for a,b,ligands,kind in stereo:
        target=graph.GetBondBetweenAtoms(mapped(a),mapped(b));first,second=map(mapped,ligands)
        if target is None or target.GetBondType()!=Chem.BondType.DOUBLE:
            raise _LipidScope('The original ether alkene endpoints did not survive condensation.')
        if target.GetBeginAtomIdx()!=mapped(a):first,second=second,first
        if (graph.GetBondBetweenAtoms(target.GetBeginAtomIdx(),first) is None
                or graph.GetBondBetweenAtoms(target.GetEndAtomIdx(),second) is None):
            raise _LipidScope('The final ether alkene ligand references are incomplete.')
        target.SetStereoAtoms(first,second);target.SetStereo(kind)


def _acid_graph(smiles,Chem):
    graph=Chem.MolFromSmiles(smiles)
    if graph is None or len(Chem.GetMolFrags(graph))!=1:
        raise _LipidInput('Each acyl source must be one valid connected acid graph.')
    if graph.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in graph.GetAtoms()):
        raise _LipidScope('Charged/radical or enhanced-stereo acyl sources need separate substance operations.')
    if any(a.GetIsotope() and a.GetAtomicNum()!=6 for a in graph.GetAtoms()):
        raise _LipidScope('Non-carbon isotope roles at acid condensation are not yet validated.')
    if any(b.GetStereo() in (Chem.BondStereo.STEREOATROPCW,Chem.BondStereo.STEREOATROPCCW) for b in graph.GetBonds()):
        raise _LipidScope('Assigned atropisomeric acid components need a mapped axial acyl writer.')
    sites=graph.GetSubstructMatches(Chem.MolFromSmarts('[C;X3](=[O;X1])-[O;X2;H1]'))
    if len(sites)!=1:raise _LipidInput('Exactly one carboxylic acid OH is required for each acyl source.')
    from .extended_inventory import selection_inventory
    inventory=selection_inventory(graph)
    if len(inventory.principal)!=1 or inventory.principal[0].kind!='acid':
        raise _LipidScope('The acid must retain its own principal class under component selection.')
    for atom in graph.GetAtoms():atom.SetAtomMapNum(0)
    return graph,sites[0][0],sites[0][2]


def _build(request,Chem,*,allow_unacylated=False):
    _validate_request(request,allow_unacylated=allow_unacylated);graph=Chem.RWMol(Chem.MolFromSmiles('OCC(O)CO'))
    carbon={1:1,2:2,3:4};oxygen={1:0,2:3,3:5};bindings=[]
    if tuple(n.GetIdx() for n in graph.GetAtomWithIdx(2).GetNeighbors())!=(1,3,4):
        raise _LipidScope('The glycerol atom frame is unavailable.')
    for atom in graph.GetAtoms():atom.SetProp(_OWNER,'glycerol')
    for group in sorted(request.acyl_groups,key=lambda g:g.position):
        acid,center,leaving=_acid_graph(group.acid_smiles,Chem);piece=Chem.RWMol(acid)
        piece.RemoveAtom(leaving);center-=leaving<center;offset=graph.GetNumAtoms()
        for atom in piece.GetAtoms():atom.SetProp(_OWNER,'acyl:'+str(group.position))
        graph.InsertMol(piece.GetMol());graph.AddBond(oxygen[group.position],offset+center,Chem.BondType.SINGLE)
        graph.UpdatePropertyCache(strict=False);bindings.append((group.position,offset+center,oxygen[group.position]))
    for group in sorted(request.ether_groups,key=lambda g:g.position):
        alcohol,root,leaving=_ether_graph(group.alcohol_smiles,Chem)
        _join_ether(graph,alcohol,root,leaving,oxygen[group.position],group.position,Chem)
    graph=graph.GetMol()
    if request.ether_groups:Chem.SetDoubleBondNeighborDirections(graph)
    if request.configuration!='unspecified':graph.GetAtomWithIdx(2).SetChiralTag(_sn_tag())
    try:
        Chem.SanitizeMol(graph);Chem.AssignStereochemistry(graph,cleanIt=True,force=True)
        from rdkit.Chem import rdCIPLabeler
        rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
    except (ValueError,RuntimeError,OverflowError) as error:
        raise _LipidScope('The assembled glycerol graph did not complete valence/configuration validation.') from error
    if request.configuration in {'R','S'}:
        atom=graph.GetAtomWithIdx(2)
        if not atom.HasProp('_CIPCode') or atom.GetProp('_CIPCode') not in {'R','S'}:
            raise _LipidInput('The constructed glycerol centre has no independent R/S configuration.')
        if atom.GetProp('_CIPCode')!=request.configuration:
            atom.InvertChirality();atom.ClearProp('_CIPCode')
            rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
        if atom.GetProp('_CIPCode')!=request.configuration:
            raise _LipidScope('The requested absolute glycerol configuration could not be established.')
    parts=_original_acyl_parts(graph,Chem,bindings)
    if request.configuration=='rac':
        # A racemic centre must be the only source of global handedness.
        if any(a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED and a.GetIdx()!=2 for a in graph.GetAtoms()):
            raise _LipidScope('A racemic glycerol centre with independently chiral acyl groups is not one enantiomer pair.')
        if any(info.type==Chem.StereoType.Atom_Tetrahedral and info.centeredOn!=2
               or info.type==Chem.StereoType.Bond_Double and info.specified==Chem.StereoSpecified.Unspecified
               for info in Chem.FindPotentialStereo(graph)):
            raise _LipidScope('An additional unassigned atom or alkene configuration prevents proof of a single glycerol enantiomer pair.')
        if graph.GetAtomWithIdx(2).GetChiralTag()==Chem.ChiralType.CHI_UNSPECIFIED:
            raise _LipidInput('An achiral glycerol construction has no distinct racemic pair.')
    return graph,carbon,oxygen,tuple(parts)


def _original_acyl_parts(graph,Chem,bindings):
    from .acid_derivative_methods import _acid_component,_acyl
    parts=[];citation_keys={};compound_words=set()
    for position,center,bridge in bindings:
        part=_acid_component(graph,Chem,center,bridge)
        word=_acyl(part.name) if part is not None else ''
        owned=frozenset(a.GetIdx() for a in graph.GetAtoms() if a.GetProp(_OWNER)=='acyl:'+str(position))
        if not word or part.atoms!=owned:
            raise _LipidScope('An original-graph acyl component has no complete mapped source construction.')
        from .formatting import prefix_alphabetization_key
        key=prefix_alphabetization_key(word)[:-1]
        if key in citation_keys and citation_keys[key]!=word:
            raise _LipidScope('Equal chemical acyl prefixes with distinct annotations need a proved citation comparison.')
        citation_keys[key]=word
        if part.construction.prefixes:
            if word in compound_words:
                raise _LipidScope('Multiplication of a chemically substituted O-acyl component needs a proved compound construction.')
            compound_words.add(word)
        parts.append((position,center,bridge,word,owned))
    return tuple(parts)


def _acyl_prefixes(parts,frame):
    from .formatting import prefix_alphabetization_key,enclose_name_unit
    from .constants import MULT
    grouped=defaultdict(list)
    for position,_center,_bridge,word,_atoms in parts:grouped[word].append(frame[position])
    rows=[]
    for word,positions in sorted(grouped.items(),key=lambda item:prefix_alphabetization_key(item[0])):
        locants=tuple(sorted(positions));count=len(locants)
        # Simple O-bound groups take di-/tri-O-. An annotated or
        # compound component keeps an explicit enclosing scope.
        complex_word=word[:1]=='(' or any(char in word for char in '-,[]()')
        if complex_word:
            rendered=enclose_name_unit(word)
            unit=({1:'O-',2:'di-O-',3:'tri-O-'}[count])+rendered
        else:unit=(MULT[count]+'-O-' if count>1 else 'O-')+word
        citation='' if len(grouped)==1 and locants==(1,2,3) else ','.join(str(i) for i in locants)+'-'
        rows.append((citation+unit,locants))
    all_locants=tuple(sorted(frame[p] for p,*_ in parts))
    key=(all_locants,tuple(i for _text,locants in rows for i in locants))
    return '-'.join(text for text,_locants in rows),key


def _original_ether_parts(graph,Chem):
    from .component_mapping import mapped_branch,MappedComponent,decorate_component
    from .formatting import prefix_alphabetization_key
    groups=defaultdict(set)
    for atom in graph.GetAtoms():
        if atom.HasProp(_OWNER) and atom.GetProp(_OWNER).startswith('ether:'):
            groups[int(atom.GetProp(_OWNER).partition(':')[2])].add(atom.GetIdx())
    parts=[];citation_keys={}
    for position,values in sorted(groups.items()):
        atoms=frozenset(values);ports=[(i,n.GetIdx()) for i in atoms for n in graph.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() not in atoms]
        if len(ports)!=1:raise _LipidScope('The original ether component has no unique glycerol oxygen port.')
        root,bridge=ports[0];oxygen=graph.GetAtomWithIdx(bridge)
        if (oxygen.GetAtomicNum()!=8 or not oxygen.HasProp(_OWNER) or oxygen.GetProp(_OWNER)!='glycerol'
                or graph.GetBondBetweenAtoms(root,bridge).GetBondType()!=Chem.BondType.SINGLE):
            raise _LipidScope('An ether part is not attached through its original glycerol oxygen.')
        if any(b.GetBondType()==Chem.BondType.DOUBLE for i in atoms for b in graph.GetAtomWithIdx(i).GetBonds()):
            path=[root];previous=bridge
            while True:
                choices=[n.GetIdx() for n in graph.GetAtomWithIdx(path[-1]).GetNeighbors() if n.GetIdx() in atoms and n.GetIdx()!=previous]
                if not choices:break
                if len(choices)!=1 or choices[0] in path:raise _LipidScope('The complete ether carbon path is not unique.')
                previous=path[-1];path.append(choices[0])
            if len(path)!=len(atoms):raise _LipidScope('The ether carbon path is incomplete.')
            doubles=tuple(i+1 for i in range(len(path)-1) if graph.GetBondBetweenAtoms(path[i],path[i+1]).GetBondType()==Chem.BondType.DOUBLE)
            from .unsaturated_substituent_methods import unsaturated_carbon_stem
            stem=unsaturated_carbon_stem(len(path),1,doubles,())
            word=decorate_component(graph,Chem,MappedComponent(stem,tuple((idx,str(i+1)) for i,idx in enumerate(path)),atoms,systematic_parent=True)) if stem else ''
            owned=atoms
        else:word,owned=mapped_branch(graph,Chem,root,bridge)
        if not word or owned!=atoms:raise _LipidScope('The complete original ether carbon component has no mapped organyl name.')
        key=prefix_alphabetization_key(word)[:-1]
        if key in citation_keys and citation_keys[key]!=word:
            raise _LipidScope('Distinct annotations on chemically identical ether groups need a proved citation comparison.')
        citation_keys[key]=word
        parts.append((position,root,bridge,word,atoms))
    return tuple(parts)


def glycerol_to_iupac_result(request:GlycerolSpec) -> GlycerolNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        graph,carbon,oxygen,parts=_build(request,Chem)
        ethers=_original_ether_parts(graph,Chem) if request.ether_groups else ()
        frames=({1:1,2:2,3:3},) if request.configuration=='sn' else ({1:1,2:2,3:3},{1:3,2:2,3:1})
        choices=[(*_acyl_prefixes(parts+ethers,frame),frame) for frame in frames]
        prefix,_key,frame=min(choices,key=lambda item:item[1])
        noun={'sn':'sn-glycerol','unspecified':'glycerol','rac':'rac-glycerol','R':'glycerol','S':'glycerol'}[request.configuration]
        name=prefix+('-' if request.configuration in {'sn','rac'} else '')+noun
        if request.configuration in {'R','S'}:name='(2'+request.configuration+')-'+name
        identity=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True);members=();proportions=()
        if request.configuration=='rac':
            other=Chem.Mol(graph);other.GetAtomWithIdx(2).InvertChirality()
            members=tuple(sorted((identity,Chem.MolToSmiles(other,canonical=True,isomericSmiles=True))))
            if members[0]==members[1]:raise _LipidInput('This construction has no distinct racemic enantiomers.')
            identity='';proportions=((1,2),(1,2))
        for atom in graph.GetAtoms():atom.SetAtomMapNum(atom.GetIdx()+1)
        mapped=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
        role_maps=tuple(sorted([(f'C{frame[p]}',idx+1) for p,idx in carbon.items()]+[(f'O{frame[p]}',idx+1) for p,idx in oxygen.items()]))
        bindings=tuple(sorted((LipidAcylBinding(frame[p],c+1,o+1,tuple(i+1 for i in sorted(atoms)),word)
                              for p,c,o,word,atoms in parts),key=lambda b:b.position))
        ether_bindings=tuple(sorted((LipidEtherBinding(frame[p],c+1,o+1,tuple(i+1 for i in sorted(atoms)),word)
                                    for p,c,o,word,atoms in ethers),key=lambda b:b.position))
        return GlycerolNamingResult(name=name,generation_status='available',preferred_status='not-assessed',
            route='explicit-glycerol',canonical_smiles=identity,mapped_smiles=mapped,glycerol_atom_maps=role_maps,
            acyl_bindings=bindings,ether_bindings=ether_bindings,member_smiles=members,proportions=proportions,configuration=request.configuration,
            substance_form='equimolar-enantiomer-mixture' if members else 'conventional-neutral')
    except _LipidInput as error:return _failure('invalid-input',str(error))
    except (_LipidScope,ValueError,RuntimeError,OverflowError) as error:return _failure('unavailable',str(error))


def glycerol_to_iupac(request:GlycerolSpec) -> str:return glycerol_to_iupac_result(request).name
