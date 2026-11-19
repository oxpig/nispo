"""P-93.3 original-ligand polyhedral configuration from explicit directions."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key,lru_cache
from itertools import permutations
from math import sqrt,isfinite

from .naming_result import NamingResult,SelectionLimitation


@dataclass(frozen=True,slots=True)
class PolyhedralGeometry:
    symbol: str
    donor_atoms: tuple[int,...]
    directions: tuple[tuple[float,float,float],...]
    angular_tolerance_degrees: float = 10.


@dataclass(frozen=True,slots=True)
class PolyhedralDescription:
    status: str
    descriptor: str = ''
    polyhedral_symbol: str = ''
    configuration_index: str = ''
    chirality_symbol: str = ''
    center_atom: int|None = None
    donor_atoms: tuple[int,...] = ()
    priority_numbers: tuple[int,...] = ()
    directions: tuple[tuple[float,float,float],...] = ()
    maximum_angular_error_degrees: float|None = None
    molecular_smiles: str = ''
    native_stereo_smiles: str = ''
    limitations: tuple[SelectionLimitation,...] = ()


@dataclass(frozen=True,slots=True)
class PolyhedralNamingResult(NamingResult):
    configuration: PolyhedralDescription|None = None
    molecular_smiles: str = ''
    native_stereo_smiles: str = ''


_IDEAL={
    'TPY-3':((1,1,1),(1,-1,-1),(-1,1,-1)),
    'TS-3':((0,0,1),(0,0,-1),(1,0,0)),
    'T-4':((1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)),
    'SP-4':((1,0,0),(0,1,0),(-1,0,0),(0,-1,0)),
    'SS-4':((0,0,1),(0,0,-1),(1,0,0),(-.5,sqrt(3)/2,0)),
    'TBPY-5':((0,0,1),(0,0,-1),(1,0,0),(-.5,sqrt(3)/2,0),(-.5,-sqrt(3)/2,0)),
    'SPY-5':((0,0,1),(1,0,0),(0,1,0),(-1,0,0),(0,-1,0)),
    'OC-6':((0,0,1),(0,0,-1),(1,0,0),(0,1,0),(-1,0,0),(0,-1,0)),
}


def _failure(reason,status='unavailable'):
    return PolyhedralDescription(status,limitations=(SelectionLimitation('polyhedral-configuration','P-93.3',reason),))


def _ligand_priorities(mol,center,donors,Chem):
    """Complete disjoint original arms; canonical ranks are never CIP ranks."""
    from .conformations import _precedence
    graph=Chem.Mol(mol)
    for atom in graph.GetAtoms():atom.SetAtomMapNum(0)
    periodic=Chem.GetPeriodicTable();normalized=Chem.Mol(graph)
    for atom in normalized.GetAtoms():
        if atom.GetIsotope()==periodic.GetMostCommonIsotope(atom.GetAtomicNum()):atom.SetIsotope(0)
    arms=[];seen=set()
    for root in donors:
        found=set();stack=[root]
        while stack:
            idx=stack.pop()
            if idx==center or idx in found:continue
            found.add(idx);stack.extend(a.GetIdx() for a in graph.GetAtomWithIdx(idx).GetNeighbors())
        if found&seen:return None,'Chelate/metal-cycle ligand priority needs the priming and joint-digraph rules.'
        seen.update(found)
        identity=Chem.MolFragmentToSmiles(normalized,atomsToUse=sorted(found),rootedAtAtom=root,
            canonical=True,isomericSmiles=True,allHsExplicit=True)
        arms.append((root,frozenset(found),identity))
        bond=graph.GetBondBetweenAtoms(root,center)
        if bond.GetBondType()==Chem.BondType.DATIVE:
            # A single supplied donor contact supplies the same single root
            # link in a whole-arm comparison, without changing free charges.
            bond.SetBondType(Chem.BondType.SINGLE)
    graph.UpdatePropertyCache(strict=False)
    size=len(arms);matrix=[[0]*size for _ in arms]
    for i in range(size):
        for j in range(i):
            value=_precedence(graph,Chem,arms[i],arms[j])
            reverse=_precedence(graph,Chem,arms[j],arms[i])
            if value is None or reverse is None or value!=-reverse:
                return None,'A same-element whole-ligand CIP comparison is outside the proved neutral uncoupled scope.'
            matrix[i][j]=value;matrix[j][i]=-value
    order=sorted(range(size),key=cmp_to_key(lambda i,j:-matrix[i][j]))
    for a,i in enumerate(order):
        if any(matrix[i][j]<0 for j in order[a+1:]):return None,'The whole-ligand priority relation is not a complete consistent order.'
    priorities=[0]*size;rank=0;previous=None
    for i in order:
        if previous is None or matrix[previous][i]:rank+=1
        priorities[i]=rank;previous=i
    if any((priorities[i]==priorities[j])!=(matrix[i][j]==0) for i in range(size) for j in range(size)):
        return None,'Equivalent-priority groups do not form a consistent partition.'
    return tuple(priorities),''


@lru_cache(maxsize=8)
def _ideal_data(symbol):
    import numpy as np
    points=np.asarray(_IDEAL[symbol],dtype=float);points/=np.linalg.norm(points,axis=1)[:,None]
    angles=np.degrees(np.arccos(np.clip(points@points.T,-1,1)))
    return angles,np.asarray(tuple(permutations(range(len(points)))),dtype=int)


def _fit(symbol,vectors):
    import numpy as np
    ideal,orders=_ideal_data(symbol)
    actual=np.degrees(np.arccos(np.clip(vectors@vectors.T,-1,1)))
    differences=np.max(np.abs(actual[orders[:,:,None],orders[:,None,:]]-ideal),axis=(1,2))
    best=float(np.min(differences))
    # All minimum frames represent the same ideal geometry. Required roles
    # (apex, opposite pairs, plane) are invariant under its automorphisms.
    return tuple(int(i) for i in orders[int(np.argmin(differences))]),best


def _cyclic_chirality(vectors,priorities,axis,plane):
    import numpy as np
    from math import atan2
    direction=vectors[axis];base=vectors[plane[0]]
    def angle(i):return atan2(float(np.dot(direction,np.cross(base,vectors[i]))),float(np.dot(base,vectors[i])))
    sequence=tuple(priorities[i] for i in sorted(plane,key=angle))
    def minimum_rotation(values):return min(values[i:]+values[:i] for i in range(len(values)))
    ccw=minimum_rotation(sequence);cw=minimum_rotation(tuple(reversed(sequence)))
    return '' if cw==ccw else 'C' if cw<ccw else 'A'


def _configuration(symbol,vectors,priorities,frame):
    import numpy as np
    p=priorities;f=frame;n=len(p)
    if symbol in {'TPY-3','T-4'}:
        if len(set(p))!=n:return '',''
        order=sorted(range(n),key=lambda i:p[i]);points=[vectors[i] for i in order]
        if symbol=='TPY-3':
            normal=np.cross(points[1]-points[0],points[2]-points[0])
            if np.dot(normal,points[0])>0:normal=-normal
            points.append(normal/np.linalg.norm(normal))
        volume=float(np.linalg.det(np.asarray([points[i]-points[3] for i in range(3)])))
        return '',('R' if volume<0 else 'S')
    if symbol=='TS-3':return str(p[f[2]]),''
    if symbol in {'SS-4','TBPY-5'}:
        first,second=f[:2];index=''.join(map(str,sorted((p[first],p[second]))))
        if p[first]==p[second]:return index,''
        axis=first if p[first]<p[second] else second;plane=f[2:]
        if symbol=='SS-4':
            a,b=sorted(plane,key=lambda i:p[i])
            if p[a]==p[b]:return index,''
            turn=float(np.dot(vectors[axis],np.cross(vectors[a],vectors[b])))
            return index,'A' if turn>0 else 'C'
        return index,_cyclic_chirality(vectors,p,axis,plane)
    if symbol in {'SP-4','SPY-5'}:
        plane=f if symbol=='SP-4' else f[1:]
        trans={plane[i]:plane[(i+2)%4] for i in range(4)}
        highest=min(p[i] for i in plane)
        opposite=max(p[trans[i]] for i in plane if p[i]==highest)
        index=str(opposite) if symbol=='SP-4' else str(p[f[0]])+str(opposite)
        return index,'' if symbol=='SP-4' else _cyclic_chirality(vectors,p,f[0],plane)
    trans={}
    for a,b in ((f[0],f[1]),(f[2],f[4]),(f[3],f[5])):trans[a]=b;trans[b]=a
    # IR-9.3.3.4 trans-maximum difference: retain the numerically greatest
    # trans priority, first on the reference axis, then within its plane.
    first=max(p[trans[i]] for i in range(n) if p[i]==1);candidates=[]
    for axis in range(n):
        if p[axis]!=1 or p[trans[axis]]!=first:continue
        plane=tuple(i for i in range(n) if i not in {axis,trans[axis]})
        highest=min(p[i] for i in plane)
        second=max(p[trans[i]] for i in plane if p[i]==highest)
        candidates.append((second,_cyclic_chirality(vectors,p,axis,plane)))
    second=max(row[0] for row in candidates);symbols={chirality for value,chirality in candidates if value==second}
    return str(first)+str(second),next(iter(symbols)) if len(symbols)==1 else ''


def _native(mol,center,donors,directions,symbol,chirality,Chem):
    """Encode only the central supplied geometry, retaining all other stereo."""
    probe=Chem.Mol(mol)
    for atom in probe.GetAtoms():atom.SetAtomMapNum(0)
    probe.RemoveAllConformers();conf=Chem.Conformer(probe.GetNumAtoms());conf.Set3D(True)
    for donor,direction in zip(donors,directions,strict=True):conf.SetAtomPosition(donor,direction)
    probe.AddConformer(conf);Chem.AssignAtomChiralTagsFromStructure(probe,replaceExistingTags=True)
    source=probe.GetAtomWithIdx(center);tagged=Chem.Mol(mol)
    for atom in tagged.GetAtoms():atom.SetAtomMapNum(0)
    target=tagged.GetAtomWithIdx(center)
    if symbol in {'TPY-3','T-4'}:
        if not chirality:return ''
        from rdkit.Chem import rdCIPLabeler
        for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            target.SetChiralTag(tag)
            if target.HasProp('_CIPCode'):target.ClearProp('_CIPCode')
            rdCIPLabeler.AssignCIPLabels(tagged,maxRecursiveIterations=1000000)
            if target.HasProp('_CIPCode') and target.GetProp('_CIPCode')==chirality:
                return Chem.MolToSmiles(tagged,canonical=True,isomericSmiles=True)
        return ''
    expected={'TS-3':Chem.ChiralType.CHI_SQUAREPLANAR,'SP-4':Chem.ChiralType.CHI_SQUAREPLANAR,
        'SS-4':Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,'TBPY-5':Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
        'SPY-5':Chem.ChiralType.CHI_OCTAHEDRAL,'OC-6':Chem.ChiralType.CHI_OCTAHEDRAL}[symbol]
    if source.GetChiralTag()!=expected or not source.HasProp('_chiralPermutation'):return ''
    target.SetChiralTag(source.GetChiralTag());target.SetIntProp('_chiralPermutation',source.GetIntProp('_chiralPermutation'))
    text=Chem.MolToSmiles(tagged,canonical=True,isomericSmiles=True)
    return text if Chem.MolFromSmiles(text) is not None else ''


def _independent_other_configuration(mol,center,Chem):
    """Existing ligand descriptors must already follow constitution/isotopes."""
    from rdkit.Chem import rdCIPLabeler
    original=Chem.Mol(mol);original.GetAtomWithIdx(center).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    if not any(int(a.GetChiralTag()) for a in original.GetAtoms()) and not any(int(b.GetStereo()) for b in original.GetBonds()):return True
    from .cumulene_configuration import _independent_ordinary_configuration
    if not _independent_ordinary_configuration(original,Chem):return False
    rdCIPLabeler.AssignCIPLabels(original,maxRecursiveIterations=1000000)
    for source in original.GetBonds():
        if not int(source.GetStereo()):continue
        if source.GetStereo() not in {Chem.BondStereo.STEREOE,Chem.BondStereo.STEREOZ,
                Chem.BondStereo.STEREOCIS,Chem.BondStereo.STEREOTRANS} or len(source.GetStereoAtoms())!=2:return False
        probe=Chem.Mol(original);Chem.RemoveStereochemistry(probe)
        for obj in list(probe.GetAtoms())+list(probe.GetBonds()):
            if obj.HasProp('_CIPCode'):obj.ClearProp('_CIPCode')
        target=probe.GetBondWithIdx(source.GetIdx());target.SetStereoAtoms(*source.GetStereoAtoms());target.SetStereo(source.GetStereo())
        rdCIPLabeler.AssignCIPLabels(probe,maxRecursiveIterations=1000000)
        if not source.HasProp('_CIPCode') or not target.HasProp('_CIPCode') or source.GetProp('_CIPCode')!=target.GetProp('_CIPCode'):return False
    return True


def describe_polyhedral_configuration(mol,center_atom:int,geometry:PolyhedralGeometry) -> PolyhedralDescription:
    from .api import Chem
    if Chem is None:return _failure('RDKit is unavailable.')
    try:
        if (not isinstance(mol,Chem.Mol) or isinstance(center_atom,bool) or not isinstance(center_atom,int)
                or not 0<=center_atom<mol.GetNumAtoms() or not isinstance(geometry,PolyhedralGeometry)
                or not isinstance(geometry.symbol,str) or geometry.symbol not in _IDEAL
                or not isinstance(geometry.donor_atoms,tuple) or not isinstance(geometry.directions,tuple)):
            return _failure('Declare a molecular graph, original centre and covered typed polyhedral geometry.','invalid-input')
        donors=geometry.donor_atoms;size=len(_IDEAL[geometry.symbol])
        if (len(donors)!=size or len(geometry.directions)!=size
                or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<mol.GetNumAtoms() for i in donors)
                or len(set(donors))!=size or set(donors)!={a.GetIdx() for a in mol.GetAtomWithIdx(center_atom).GetNeighbors()}):
            return _failure('Original distinct donor indices must cover exactly every attached atom.','invalid-input')
        if (isinstance(geometry.angular_tolerance_degrees,bool) or not isinstance(geometry.angular_tolerance_degrees,(int,float))
                or not isfinite(geometry.angular_tolerance_degrees) or not 0<=geometry.angular_tolerance_degrees<=20):
            return _failure('Declare a finite angular tolerance from zero through twenty degrees.','invalid-input')
        if any(not isinstance(v,tuple) or len(v)!=3 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not isfinite(x) for x in v) for v in geometry.directions):
            return _failure('Each donor needs a finite three-dimensional direction.','invalid-input')
        if mol.GetStereoGroups():return _failure('Enhanced substances require joint polyhedral ensemble semantics.')
        atom=mol.GetAtomWithIdx(center_atom)
        if atom.GetTotalNumHs(includeNeighbors=False) or atom.GetNumRadicalElectrons():return _failure('Every real central ligand must be explicit; central radicals need a separate model.')
        if any(b.GetBondType() not in {Chem.BondType.SINGLE,Chem.BondType.DATIVE} for b in atom.GetBonds()):
            return _failure('Multiple-bond/duplicate-atom central ligand geometry needs a separate joint writer.')
        if len(Chem.GetMolFrags(mol))!=1:return _failure('Describe one connected coordination entity.')
        import numpy as np
        vectors=np.asarray(geometry.directions,dtype=float);lengths=np.linalg.norm(vectors,axis=1)
        if np.min(lengths)<=1e-8:return _failure('A donor direction coincides with the centre.','invalid-input')
        vectors/=lengths[:,None];frame,error=_fit(geometry.symbol,vectors)
        if error>geometry.angular_tolerance_degrees+1e-7:return _failure('The supplied directions do not establish the declared idealized geometry within its angular tolerance.')
        priorities,reason=_ligand_priorities(mol,center_atom,donors,Chem)
        if priorities is None:return _failure(reason)
        if not _independent_other_configuration(mol,center_atom,Chem):
            return _failure('A ligand stereodescriptor needs a coupled polyhedral/ordinary priority digraph.')
        index,chirality=_configuration(geometry.symbol,vectors,priorities,frame)
        descriptor=chirality if geometry.symbol in {'TPY-3','T-4'} else '-'.join(x for x in (geometry.symbol,index,chirality) if x)
        native=_native(mol,center_atom,donors,geometry.directions,geometry.symbol,chirality,Chem)
        # An existing supplied central stereo assignment must agree with the
        # declared directions. Never silently replace a conflicting isomer.
        if atom.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED:
            original=Chem.Mol(mol)
            for a in original.GetAtoms():a.SetAtomMapNum(0)
            if not native or Chem.MolToSmiles(original,canonical=True,isomericSmiles=True)!=native:
                return _failure('Existing central native stereo conflicts with, or cannot be validated against, the supplied directions.')
        plain=Chem.Mol(mol)
        for a in plain.GetAtoms():a.SetAtomMapNum(0)
        return PolyhedralDescription('complete',descriptor,geometry.symbol,index,chirality,center_atom,donors,
            priorities,geometry.directions,error,Chem.MolToSmiles(plain,canonical=True,isomericSmiles=True),native)
    except (ValueError,RuntimeError,OverflowError,TypeError) as error:return _failure(str(error))


def polyhedral_to_iupac_result(mol,center_atom:int,geometry:PolyhedralGeometry) -> PolyhedralNamingResult:
    from .api import Chem,mol_to_iupac_result
    description=describe_polyhedral_configuration(mol,center_atom,geometry)
    if description.status!='complete':
        return PolyhedralNamingResult('',description.status,'unresolved',description.limitations,'explicit-polyhedral',configuration=description)
    # The mononuclear main-group root has no missing central locant; other
    # noncentral stereocentres need a joint mapped global descriptor writer.
    if (mol.GetAtomWithIdx(center_atom).GetAtomicNum() not in {14,15,16,32,33,34,50,51,52,82,83}
            or any(a.GetIdx()!=center_atom and a.GetAtomicNum() not in {1,6,9,17,35,53} for a in mol.GetAtoms())
            or any(a.GetIdx()!=center_atom and int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) for b in mol.GetBonds())):
        issue=SelectionLimitation('polyhedral-composition','P-93.3.4.2.2','A whole-name central locant and joint descriptor ordering remain unproved in this molecular context.')
        return PolyhedralNamingResult('','unavailable','unresolved',(issue,),'explicit-polyhedral',configuration=description)
    plain=Chem.Mol(mol);plain.GetAtomWithIdx(center_atom).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    base=mol_to_iupac_result(plain)
    name=('('+description.descriptor+')-' if description.descriptor else '')+base.name if base.name else ''
    return PolyhedralNamingResult(name,base.generation_status,base.preferred_status,base.unresolved,
        'explicit-polyhedral',configuration=description,molecular_smiles=description.molecular_smiles,native_stereo_smiles=description.native_stereo_smiles)


def polyhedral_to_iupac(mol,center_atom:int,geometry:PolyhedralGeometry) -> str:
    return polyhedral_to_iupac_result(mol,center_atom,geometry).name
