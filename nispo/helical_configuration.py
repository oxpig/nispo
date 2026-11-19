"""Explicit P-92 helical configuration on verified helicene parent graphs."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite,sin,radians

from .naming_result import NamingResult,SelectionLimitation


@dataclass(frozen=True,slots=True)
class HelicalUnit:
    ring_sequence: tuple[tuple[int,...],...]
    descriptor: str
    normalized_triple_products: tuple[float,...]


@dataclass(frozen=True,slots=True)
class HelicalNamingResult(NamingResult):
    canonical_smiles: str = ''
    conformer_id: int = -1
    coordinates: tuple[tuple[float,float,float],...] = ()
    helical_units: tuple[HelicalUnit,...] = ()
    atom_locants: tuple[tuple[int,str],...] = ()
    parent: str = ''
    helicity_encoded_in_native_smiles: bool = False


def _failure(status,reason,*,identity='',coordinates=(),conformer_id=-1,units=()):
    return HelicalNamingResult(name='', generation_status=status,
        preferred_status='unresolved' if status!='invalid-input' else 'not-assessed',
        unresolved=(SelectionLimitation('helical-configuration','P-92.1.2.2/P-25',reason),),
        route='explicit-3D-helicene', canonical_smiles=identity, conformer_id=conformer_id,
        coordinates=coordinates, helical_units=units)


@lru_cache(maxsize=1)
def _parents():
    from .api import Chem
    from .resource_templates import OPSIN_PARENT_TEMPLATES,_read_json_records
    rows=[]
    for name,text,locants in OPSIN_PARENT_TEMPLATES:
        if name not in {'pentahelicene','hexahelicene','heptahelicene'}:continue
        rows.append((name,Chem.MolFromSmiles(text),tuple(enumerate(locants))))
    for row in _read_json_records('opsin_polyacene_helicene_templates.json'):
        if row['name'] not in {'octahelicene','nonahelicene','decahelicene','undecahelicene','dodecahelicene'}:continue
        rows.append((row['name'],Chem.MolFromSmiles(row['smiles']),tuple((int(i),str(loc)) for i,loc in row['locants'].items())))
    if len(rows)!=8 or any(query is None or len(locants)!=query.GetNumAtoms() for _name,query,locants in rows):
        raise ValueError('The complete declared helicene parent resource is unavailable.')
    return tuple(rows)


def _parent_matches(mol,Chem):
    core=frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==6 and a.GetIsAromatic())
    if any(a.GetIdx() not in core and (a.GetAtomicNum() not in {1,9,17,35,53} or a.GetDegree()!=1
            or a.GetNeighbors()[0].GetIdx() not in core) for a in mol.GetAtoms()):return None
    neutral=Chem.Mol(mol)
    for atom in neutral.GetAtoms():atom.SetIsotope(0);atom.SetAtomMapNum(0)
    matches=[]
    for name,query,locants in _parents():
        if query.GetNumAtoms()!=len(core):continue
        images=neutral.GetSubstructMatches(query,uniquify=False,maxMatches=1024)
        images=tuple(image for image in images if frozenset(image)==core)
        if images:matches.append((name,locants,images))
    return (core,*matches[0]) if len(matches)==1 else None


def _face_path(mol,core):
    faces=tuple(tuple(ring) for ring in mol.GetRingInfo().AtomRings() if set(ring)<=core)
    if not 5<=len(faces)<=12 or any(len(face)!=6 for face in faces):return None
    adjacency={i:set() for i in range(len(faces))}
    for i,left in enumerate(faces):
        for j in range(i+1,len(faces)):
            shared=set(left)&set(faces[j])
            if not shared:continue
            if len(shared)!=2 or mol.GetBondBetweenAtoms(*shared) is None:return None
            adjacency[i].add(j);adjacency[j].add(i)
    ends=[i for i,neighbors in adjacency.items() if len(neighbors)==1]
    if len(ends)!=2 or any(len(neighbors)>2 for neighbors in adjacency.values()):return None
    sequence=[];previous=None;current=ends[0]
    while current is not None:
        if current in sequence:return None
        sequence.append(current);next_faces=adjacency[current]-{previous}
        previous,current=current,next(iter(next_faces)) if next_faces else None
    return tuple(faces[i] for i in sequence) if len(sequence)==len(faces) else None


def _geometry(mol,coordinates,faces,distance_tolerance,angular_tolerance):
    import numpy as np
    cloud=np.asarray(coordinates,dtype=float);edges={tuple(sorted((a,b))) for face in faces for a,b in zip(face,face[1:]+face[:1])}
    lengths=np.asarray([np.linalg.norm(cloud[a]-cloud[b]) for a,b in edges]);scale=float(np.median(lengths))
    if not np.isfinite(lengths).all() or not isfinite(scale) or scale<=distance_tolerance or np.min(lengths)<.75*scale or np.max(lengths)>1.25*scale:return None
    centers=[]
    for face in faces:
        points=cloud[list(face)];center=points.mean(axis=0);_u,values,vectors=np.linalg.svd(points-center,full_matrices=False)
        if values[1]<=distance_tolerance or np.max(np.abs((points-center)@vectors[-1]))>.15*scale:return None
        turns=[float(np.dot(np.cross(points[(j+1)%6]-points[j],points[(j+2)%6]-points[(j+1)%6]),vectors[-1])) for j in range(6)]
        if not (min(turns)>distance_tolerance*scale or max(turns)<-distance_tolerance*scale):return None
        centers.append(center)
    products=[]
    for start in range(len(centers)-3):
        first,second,third=[centers[start+j+1]-centers[start+j] for j in range(3)]
        normal=np.cross(first,second);denominator=float(np.linalg.norm(normal)*np.linalg.norm(third))
        if denominator<=distance_tolerance**3:return None
        products.append(float(np.dot(normal,third))/denominator)
    threshold=sin(radians(angular_tolerance))
    if min(products)>threshold:return 'P',tuple(products)
    if max(products)<-threshold:return 'M',tuple(products)
    return None


def _numbered_name(mol,Chem,parent,locants,images):
    from .component_mapping import MappedComponent,decorate_component,hydrogen_bindings,fully_covered
    from .parent_annotations import isotope_numbering_key
    from .parent_selection import locant_key
    from .formatting import prefix_alphabetization_key
    words={9:'fluoro',17:'chloro',35:'bromo',53:'iodo'};candidates=[]
    for image in images:
        mapping=hydrogen_bindings(mol,{image[i]:loc for i,loc in locants});prefixes=[];owned=set()
        for atom in mol.GetAtoms():
            if atom.GetIdx() in mapping:continue
            if atom.GetAtomicNum()==1 and not atom.GetIsotope():continue
            if atom.GetAtomicNum() not in words:return None
            host=atom.GetNeighbors()[0].GetIdx()
            if host not in mapping:return None
            prefixes.append((mapping[host],words[atom.GetAtomicNum()]));owned.add(atom.GetIdx())
        if not fully_covered(mol,mapping,owned):continue
        isotope=isotope_numbering_key(mol,mapping,include_substituent_groups=True)
        if isotope is None:continue
        name=decorate_component(mol,Chem,MappedComponent(parent,tuple(mapping.items()),frozenset(mapping),tuple(prefixes),frozenset(owned),systematic_parent=True))
        if not name:continue
        key=(tuple(sorted(locant_key(loc) for loc,_word in prefixes)),
            tuple(locant_key(loc) for loc,word in sorted(prefixes,key=lambda row:(prefix_alphabetization_key(row[1]),locant_key(row[0])))),isotope)
        candidates.append((key,name,tuple(mapping.items())))
    if not candidates:return None
    key=min(row[0] for row in candidates);best=[row for row in candidates if row[0]==key]
    if len({row[1] for row in best})!=1:return None
    return best[0][1:]


def helicene_to_iupac_result(mol,*,conformer_id=-1,ring_sequence=None,distance_tolerance=1e-3,angular_tolerance=1.0) -> HelicalNamingResult:
    """Name one explicitly supplied uniformly helical helicene geometry.

    The finite source parent family has five through twelve benzene faces;
    terminal halo groups and carbon isotopes retain mapped atom identity.
    No rotational-barrier or native-SMILES helicity is inferred.
    """
    from .api import Chem
    if (Chem is None or type(conformer_id) is not int or type(distance_tolerance) not in {int,float}
            or not isfinite(distance_tolerance) or distance_tolerance<=0
            or type(angular_tolerance) not in {int,float} or not isfinite(angular_tolerance) or not 0<angular_tolerance<30):
        return _failure('invalid-input','A 3D molecular conformer and finite positive geometric tolerances are required.')
    try:
        graph=Chem.Mol(mol)
        # Sanitization may discard an invalid native stereo flag on an
        # aromatic atom. Decline unsupported declarations before that step,
        # so a supplied additional unit cannot disappear into a helix name.
        if graph.GetStereoGroups() or any(int(a.GetChiralTag()) for a in graph.GetAtoms()) or any(int(b.GetStereo()) for b in graph.GetBonds()):
            return _failure('unavailable','Additional assigned stereogenic units require a joint configuration operation.')
        Chem.SanitizeMol(graph);conformer=graph.GetConformer(conformer_id)
        if not conformer.Is3D() or not graph.GetNumAtoms():return _failure('invalid-input','The selected conformer must be explicitly three-dimensional.')
        coordinates=tuple(tuple(float(v) for v in conformer.GetAtomPosition(i)) for i in range(graph.GetNumAtoms()))
        if any(not isfinite(value) for row in coordinates for value in row):return _failure('invalid-input','All coordinates must be finite.')
    except (ValueError,RuntimeError,TypeError,AttributeError):return _failure('invalid-input','A valid molecular graph and selected 3D conformer are required.')
    if (graph.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
            or a.GetIsotope() and (a.GetAtomicNum()!=6 or Chem.GetPeriodicTable().GetMassForIsotope(6,a.GetIsotope())<=0) for a in graph.GetAtoms())
            or any(int(b.GetStereo()) for b in graph.GetBonds())):
        return _failure('unavailable','Charged, radical, non-carbon isotope or additional assigned stereogenic units require a joint configuration operation.')
    for atom in graph.GetAtoms():atom.SetAtomMapNum(0)
    identity=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
    retained=dict(identity=identity,coordinates=coordinates,conformer_id=conformer.GetId())
    try:parent_data=_parent_matches(graph,Chem)
    except (ValueError,RuntimeError,KeyError):return _failure('unavailable','The source parent resource is unavailable.',**retained)
    if parent_data is None:return _failure('unavailable','The complete graph has no proved single helicene parent in this input scope.',**retained)
    core,parent,locants,images=parent_data;faces=_face_path(graph,core)
    if faces is None:return _failure('unavailable','The parent does not have one complete catacondensed helicene face path.',**retained)
    if ring_sequence is not None:
        if (not isinstance(ring_sequence,tuple) or any(not isinstance(face,tuple) or len(face)!=6
                or any(type(i) is not int or i not in core for i in face) or len(set(face))!=6 for face in ring_sequence)):
            return _failure('invalid-input','An explicit face sequence must contain complete immutable original-atom ring cycles.',**retained)
        supplied=tuple(frozenset(face) for face in ring_sequence);expected=tuple(frozenset(face) for face in faces)
        if supplied not in {expected,tuple(reversed(expected))} or any(graph.GetBondBetweenAtoms(a,b) is None
                for face in ring_sequence for a,b in zip(face,face[1:]+face[:1])):
            return _failure('invalid-input','The explicit sequence must be the complete verified helicene face path in either direction.',**retained)
        faces=ring_sequence
    geometry=_geometry(graph,coordinates,faces,distance_tolerance,angular_tolerance)
    if geometry is None:return _failure('unavailable','The supplied faces do not prove a nondegenerate uniform molecular screw.',**retained)
    descriptor,products=geometry;unit=HelicalUnit(faces,descriptor,products)
    construction=_numbered_name(graph,Chem,parent,locants,images)
    if construction is None:return _failure('unavailable','The mapped helicene component or numbering comparison is incomplete.',units=(unit,),**retained)
    name,mapping=construction
    source_anchor=(parent=='hexahelicene' and graph.GetNumAtoms()==len(core) and not any(a.GetIsotope() for a in graph.GetAtoms()))
    issues=() if source_anchor else (SelectionLimitation('helical-parent-comparison','P-25/P-52',
        'The helical descriptor and retained numbered parent are constructed; complete parent/method competition remains unproved.'),)
    return HelicalNamingResult(name='('+descriptor+')-'+name, generation_status='available',
        preferred_status='resolved' if source_anchor else 'unresolved', unresolved=issues,
        route='explicit-3D-helicene', canonical_smiles=identity, conformer_id=conformer.GetId(),
        coordinates=coordinates, helical_units=(unit,), atom_locants=mapping, parent=parent)


def helicene_to_iupac(mol,**kwargs) -> str:return helicene_to_iupac_result(mol,**kwargs).name
