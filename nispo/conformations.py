"""Explicit 3D P-94 descriptions, separate from configurational PINs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import isfinite

from .naming_result import NamingResult,SelectionLimitation


@dataclass(frozen=True, slots=True)
class TorsionReference:
    atoms: tuple[int,int,int,int]
    axis_path: tuple[int,...] = ()


@dataclass(frozen=True, slots=True)
class TorsionDescription:
    axis_atoms: tuple[int,...]
    reference_atoms: tuple[int,int,int,int] | tuple = ()
    angle_degrees: float | None = None
    descriptor: str = ""
    sign: str = ""
    selection_criteria: tuple = ()
    special_descriptor: str = ""
    boundary_options: tuple[str,...] = ()
    status: str = "unavailable"
    reason: str = ""
    special_reference_atoms: tuple = ()
    special_angle_degrees: float | None = None
    equivalent_angles: tuple[float,...] = ()


@dataclass(frozen=True, slots=True)
class RingConformation:
    atom_sequence: tuple[int,...]
    conformation: str
    displaced_atoms: tuple[int,...] = ()
    # Q, theta, phi, q2, q3; meaningful only for the supplied six-ring order.
    puckering_coordinates: tuple | None = None
    status: str = "complete"
    displacement_options: tuple[tuple[int,...],...] = ()


@dataclass(frozen=True, slots=True)
class ConformerDescription:
    status: str
    molecular_result: NamingResult
    canonical_smiles: str = ""
    conformer_id: int = -1
    coordinates: tuple[tuple[float,float,float],...] = ()
    torsions: tuple[TorsionDescription,...] = ()
    rings: tuple[RingConformation,...] = ()
    limitations: tuple[SelectionLimitation,...] = ()

    @property
    def molecular_name(self):
        return self.molecular_result.name


def _failure(reason,*,status='invalid-input'):
    return ConformerDescription(status,NamingResult('',status,'not-assessed'),
        limitations=(SelectionLimitation('conformation','P-94',reason),))


def _arm(mol,root,center,other):
    found=set();stack=[root]
    while stack:
        idx=stack.pop()
        if idx==center or idx in found:continue
        if idx==other:return None
        found.add(idx)
        stack.extend(a.GetIdx() for a in mol.GetAtomWithIdx(idx).GetNeighbors())
    return frozenset(found)


def _ligands(mol,Chem,center,other,*,allow_axis_double=False):
    atom=mol.GetAtomWithIdx(center)
    if atom.GetAtomicNum()!=6 or atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
        return None
    if any(bond.GetBondType()!=Chem.BondType.SINGLE
           and not (allow_axis_double and bond.GetOtherAtomIdx(center)==other
                    and bond.GetBondType()==Chem.BondType.DOUBLE)
           for bond in atom.GetBonds()):
        # P-94.2.2 counts duplicate/phantom atoms in the ligand set. Explicit
        # references and the separate R3C-C(Y)=X geometry remain available.
        return None
    ligands=[];seen=set();periodic=Chem.GetPeriodicTable()
    normalized=Chem.Mol(mol)
    for source in normalized.GetAtoms():
        source.SetAtomMapNum(0)
        if source.GetIsotope()==periodic.GetMostCommonIsotope(source.GetAtomicNum()):
            source.SetIsotope(0)
    for neighbor in atom.GetNeighbors():
        root=neighbor.GetIdx()
        if root==other:continue
        arm=_arm(mol,root,center,other)
        if arm is None or seen&arm:return None
        seen.update(arm)
        if neighbor.GetAtomicNum()==1:
            identity='H:'+str(neighbor.GetIsotope() or 1)
        else:
            identity=Chem.MolFragmentToSmiles(normalized,atomsToUse=sorted(arm),
                rootedAtAtom=root,canonical=True,isomericSmiles=True,allHsExplicit=True)
        ligands.append((root,arm,identity))
    ligands += [(None,frozenset(),'H:1')]*atom.GetTotalNumHs(includeNeighbors=False)
    return tuple(ligands) if 2<=len(ligands)<=3 else None


def _insert_arm(probe,mol,atoms,root,Chem):
    order=sorted(atoms);fragment=Chem.RWMol(mol)
    for idx in reversed(range(mol.GetNumAtoms())):
        if idx not in atoms:fragment.RemoveAtom(idx)
    offset=probe.GetNumAtoms();probe.InsertMol(fragment.GetMol())
    return offset+order.index(root)


def _probe_label(mol,Chem,left,right,number):
    from rdkit.Chem import rdCIPLabeler
    probe=Chem.RWMol();center=Chem.Atom(6);center.SetNoImplicit(True)
    probe.AddAtom(center)
    roots=[_insert_arm(probe,mol,arm,root,Chem) for root,arm,_identity in (left,right)]
    for root in roots:probe.AddBond(0,root,Chem.BondType.SINGLE)
    for element in (1,number+1):
        atom=Chem.Atom(element);atom.SetNoImplicit(True)
        idx=probe.AddAtom(atom);probe.AddBond(0,idx,Chem.BondType.SINGLE)
    graph=probe.GetMol();graph.RemoveAllConformers()
    for atom in graph.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp('_CIPCode'):atom.ClearProp('_CIPCode')
    graph.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    graph.UpdatePropertyCache(strict=False)
    rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=100000)
    atom=graph.GetAtomWithIdx(0)
    return atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else None


@lru_cache(maxsize=118)
def _known_probe_label(number):
    """Calibrate only RDKit tag parity against known heavier-isotope priority."""
    from rdkit import Chem
    periodic=Chem.GetPeriodicTable();graph=Chem.RWMol()
    for mass in (periodic.GetMostCommonIsotope(number)+1,periodic.GetMostCommonIsotope(number)):
        atom=Chem.Atom(number);atom.SetIsotope(mass);atom.SetNoImplicit(True)
        graph.AddAtom(atom)
    mol=graph.GetMol();mol.UpdatePropertyCache(strict=False)
    return _probe_label(mol,Chem,(0,frozenset({0}),'heavy'),(1,frozenset({1}),'light'),number)


def _precedence(mol,Chem,left,right):
    if left[0] is None or right[0] is None:
        if left[2]==right[2]:return 0
        # Virtual groups here are only ordinary H. A real heavy atom or a
        # heavier explicit H isotope has unambiguous sequence precedence.
        return -1 if left[0] is None else 1
    a=mol.GetAtomWithIdx(left[0]);b=mol.GetAtomWithIdx(right[0])
    if a.GetAtomicNum()!=b.GetAtomicNum():return 1 if a.GetAtomicNum()>b.GetAtomicNum() else -1
    periodic=Chem.GetPeriodicTable()
    am=a.GetIsotope() or periodic.GetMostCommonIsotope(a.GetAtomicNum())
    bm=b.GetIsotope() or periodic.GetMostCommonIsotope(b.GetAtomicNum())
    if am!=bm:return 1 if am>bm else -1
    if left[2]==right[2]:return 0
    number=a.GetAtomicNum()
    if not 1<number<118 or len(left[1]|right[1])>256:return None
    # The artificial centre supplies one ordinary bond to each whole arm.
    # A multiply bonded root also has a duplicate-atom ligand at the axis
    # centre; replacing that relationship by a single bond is not a proof.
    for root,arm,_identity in (left,right):
        if any(bond.GetBondType()!=Chem.BondType.SINGLE
               for bond in mol.GetAtomWithIdx(root).GetBonds()
               if bond.GetOtherAtomIdx(root) not in arm):
            return None
    # The probe preserves complete, disjoint ligands. Existing stereo or
    # ionic/radical ligand comparisons require a broader sequence proof.
    for idx in left[1]|right[1]:
        atom=mol.GetAtomWithIdx(idx)
        if int(atom.GetChiralTag()) or atom.GetFormalCharge() or atom.GetNumRadicalElectrons():return None
        if any(int(bond.GetStereo()) for bond in atom.GetBonds()):return None
    try:
        actual=_probe_label(mol,Chem,left,right,number)
        known=_known_probe_label(number)
    except (ValueError,RuntimeError,OverflowError):
        return None
    if actual not in {'R','S'} or known not in {'R','S'}:return None
    return 1 if actual==known else -1


def _select_ligands(mol,Chem,center,other):
    groups=_ligands(mol,Chem,center,other)
    if groups is None:return None
    counts=Counter(row[2] for row in groups)
    if len(counts)==1:return groups,3
    unique=[row for row in groups if counts[row[2]]==1]
    if len(unique)==1:return tuple(unique),2
    if len(counts)!=len(groups):return None
    selected=[]
    for candidate in groups:
        comparisons=[_precedence(mol,Chem,candidate,other_group) for other_group in groups if other_group is not candidate]
        if any(value is None for value in comparisons):return None
        if all(value==1 for value in comparisons):selected.append(candidate)
    return (tuple(selected),1) if len(selected)==1 else None


def _description(mol,Chem,coordinates,refs,axis,criteria,distance_tolerance,angular_tolerance):
    from .conformer_geometry import signed_torsion,torsion_region
    points=[coordinates[i] for i in refs]
    angle=signed_torsion(points,[coordinates[i] for i in axis],distance_tolerance)
    if angle is None:
        return TorsionDescription(tuple(axis),tuple(refs),selection_criteria=criteria,
            reason='Coincident axis, collinear reference, or noncollinear extended axis.')
    descriptor,options=torsion_region(angle,angular_tolerance)
    sign='' if abs(angle)<=angular_tolerance or abs(abs(angle)-180)<=angular_tolerance else '+' if angle>0 else '-'
    x,a,b,y=refs;special=''
    if (len(axis)==2 and mol.GetBondBetweenAtoms(a,b).GetBondType()==Chem.BondType.SINGLE
            and mol.GetBondBetweenAtoms(x,a).GetBondType()==Chem.BondType.DOUBLE
            and mol.GetBondBetweenAtoms(b,y).GetBondType()==Chem.BondType.DOUBLE):
        special='s-cis' if descriptor=='sp' else 's-trans' if descriptor=='ap' else ''
    return TorsionDescription(tuple(axis),tuple(refs),angle,descriptor,sign,criteria,special,
        options,'boundary' if options else 'complete')


def _auto_torsion(mol,Chem,coordinates,axis,distance_tolerance,angular_tolerance):
    a,b=axis
    first=_select_ligands(mol,Chem,a,b);second=_select_ligands(mol,Chem,b,a)
    if first is None or second is None:
        return TorsionDescription(tuple(axis),reason='Automatic complete ligand selection is outside the proved carbon-axis scope.')
    if any(row[0] is None for side,_criterion in (first,second) for row in side):
        return TorsionDescription(tuple(axis),selection_criteria=(first[1],second[1]),
            reason='A selected/all-identical hydrogen reference has no supplied coordinate.')
    candidates=[_description(mol,Chem,coordinates,(left[0],a,b,right[0]),axis,
        (first[1],second[1]),distance_tolerance,angular_tolerance)
        for left,right in product(first[0],second[0])]
    candidates=[row for row in candidates if row.angle_degrees is not None]
    if not candidates:return TorsionDescription(tuple(axis),reason='Every selected reference has a degenerate torsion.')
    smallest=min(abs(row.angle_degrees) for row in candidates)
    winners=[row for row in candidates if abs(abs(row.angle_degrees)-smallest)<=angular_tolerance]
    selected=min(winners,key=lambda row:row.reference_atoms)
    if len({row.descriptor for row in winners})!=1:
        return TorsionDescription(tuple(axis),reason='Equal minimum reference torsions have different descriptor regions.')
    if first[1]==second[1]==3:
        from dataclasses import replace
        from .conformer_geometry import regular_projected_ligands
        regular=all(regular_projected_ligands([coordinates[row[0]] for row in side],
            coordinates[center],coordinates[other],angular_tolerance,distance_tolerance)
            for side,center,other in ((first[0],a,b),(second[0],b,a)))
        special=('eclipsed' if smallest<=angular_tolerance else 'staggered' if abs(smallest-60)<=angular_tolerance else '') if regular else ''
        selected=replace(selected,special_descriptor=special)
    from dataclasses import replace
    angles=tuple(sorted({round(row.angle_degrees,10) for row in winners}))
    signs={row.sign for row in winners}
    selected=replace(selected,equivalent_angles=angles,sign=selected.sign if len(signs)==1 else '')
    return selected


def _carbonyl_special(mol,Chem,coordinates,axis,distance_tolerance,angular_tolerance):
    """P-94.3.1.2 R3C-C(Y)=X with a proved projected angle bisection."""
    from .conformer_geometry import regular_projected_ligands,signed_torsion
    for a,b in (axis,tuple(reversed(axis))):
        front=mol.GetAtomWithIdx(a);rear=mol.GetAtomWithIdx(b)
        if front.GetAtomicNum()!=6 or rear.GetAtomicNum()!=6:continue
        groups=[n for n in front.GetNeighbors() if n.GetIdx()!=b]
        if (len(groups)!=3 or any(n.GetAtomicNum()!=6 for n in groups)
                or any(mol.GetBondBetweenAtoms(a,n.GetIdx()).GetBondType()!=Chem.BondType.SINGLE for n in groups)):
            continue
        doubles=[n.GetIdx() for n in rear.GetNeighbors() if n.GetIdx()!=a
                 and mol.GetBondBetweenAtoms(b,n.GetIdx()).GetBondType()==Chem.BondType.DOUBLE]
        if len(doubles)!=1 or rear.GetDegree()!=3:continue
        if not regular_projected_ligands([coordinates[n.GetIdx()] for n in groups],coordinates[a],
                coordinates[b],angular_tolerance,distance_tolerance):continue
        for atom in groups:
            refs=(atom.GetIdx(),a,b,doubles[0])
            angle=signed_torsion([coordinates[i] for i in refs],distance_tolerance=distance_tolerance)
            if angle is None:continue
            kind=('eclipsing' if abs(angle)<=angular_tolerance else
                  'bisecting' if abs(abs(angle)-180)<=angular_tolerance else '')
            if kind:return kind,refs,angle
    return None


def describe_conformer(mol,*,conformer_id=-1,references=(),axes=None,rings=None,
                       distance_tolerance=1e-3,angular_tolerance=1e-6):
    """Describe supplied 3D geometry using P-94, without inventing a PIN string.

    Atom indices/coordinates refer to the original input graph. Explicit
    references report their specified torsions; automatic axes additionally
    require a proved ligand choice. Unclassified/nonideal ring geometry and
    unsupported priority comparisons remain visible. Default auto axes are
    nonring single bonds with supplied reference atoms on both ends.
    """
    from .api import Chem,mol_to_iupac_result
    if (Chem is None or mol is None or type(conformer_id) is not int
            or type(distance_tolerance) not in {int,float} or not isfinite(distance_tolerance) or distance_tolerance<=0
            or type(angular_tolerance) not in {int,float} or not isfinite(angular_tolerance) or not 0<=angular_tolerance<30):
        return _failure('A molecular 3D conformer and finite positive distance/nonnegative angular tolerances are required.')
    try:
        graph=Chem.Mol(mol);conformer=graph.GetConformer(conformer_id)
        if not graph.GetNumAtoms() or not conformer.Is3D():return _failure('The selected conformer must be explicitly three-dimensional.')
        coordinates=tuple(tuple(float(v) for v in conformer.GetAtomPosition(i)) for i in range(graph.GetNumAtoms()))
        if any(not isfinite(v) for row in coordinates for v in row):return _failure('All supplied coordinates must be finite.')
        Chem.SanitizeMol(graph)
    except (ValueError,RuntimeError,TypeError,AttributeError):
        return _failure('The molecular graph or selected conformer is unavailable.')
    from .stereochemistry import assigned_atom_stereochemistry,_indexed_bond_stereochemistry
    if (graph.GetStereoGroups()
            or any(a.GetChiralTag() not in {Chem.ChiralType.CHI_UNSPECIFIED,
                   Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
                   for a in graph.GetAtoms())
            or any(b.GetStereo() in {Chem.BondStereo.STEREOATROPCW,Chem.BondStereo.STEREOATROPCCW}
                   for b in graph.GetBonds())):
        return _failure('The conformer operation does not yet validate enhanced or advanced assigned configuration against coordinates.',status='unavailable')
    original=Chem.Mol(graph)
    original_atoms=dict(assigned_atom_stereochemistry(original,Chem,include_pseudo=True))
    original_bonds={(min(a,b),max(a,b)):d for a,b,d in _indexed_bond_stereochemistry(original,Chem)}
    for atom in graph.GetAtoms():atom.SetAtomMapNum(0)
    Chem.AssignStereochemistryFrom3D(graph,confId=conformer.GetId(),replaceExistingTags=True)
    Chem.AssignStereochemistry(graph,cleanIt=True,force=True)
    derived=dict(assigned_atom_stereochemistry(graph,Chem,include_pseudo=True))
    derived_bonds={(min(a,b),max(a,b)):d for a,b,d in _indexed_bond_stereochemistry(graph,Chem)}
    if (any(derived.get(idx)!=code for idx,code in original_atoms.items())
            or any(derived_bonds.get(key)!=code for key,code in original_bonds.items())):
        return _failure('The supplied coordinates conflict with explicitly assigned molecular configuration.')
    count=graph.GetNumAtoms()
    def ids(values):
        return all(type(i) is int and 0<=i<count for i in values)
    try:references=tuple(references)
    except TypeError:return _failure('References must be an iterable of TorsionReference values.')
    torsions=[];used=set()
    for reference in references:
        if (not isinstance(reference,TorsionReference) or not isinstance(reference.atoms,tuple)
                or len(reference.atoms)!=4 or not ids(reference.atoms) or len(set(reference.atoms))!=4
                or not isinstance(reference.axis_path,tuple)):
            return _failure('Each explicit reference needs four distinct valid atom indices.')
        x,a,b,y=reference.atoms;path=reference.axis_path or (a,b)
        if (len(path)<2 or not ids(path) or len(set(path))!=len(path) or path[0]!=a or path[-1]!=b
                or x in path or y in path or graph.GetBondBetweenAtoms(x,a) is None
                or graph.GetBondBetweenAtoms(b,y) is None
                or any(graph.GetBondBetweenAtoms(left,right) is None for left,right in zip(path,path[1:]))):
            return _failure('An explicit reference must follow its bonded axis path and terminal reference bonds.')
        torsions.append(_description(graph,Chem,coordinates,reference.atoms,path,
            ('explicit-references',),distance_tolerance,angular_tolerance))
        used.add(frozenset((a,b)))
    if axes is None:
        selected_axes=tuple((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()) for bond in graph.GetBonds()
            if bond.GetBondType()==Chem.BondType.SINGLE and not bond.IsInRing()
            and bond.GetBeginAtom().GetDegree()>1 and bond.GetEndAtom().GetDegree()>1)
    else:
        try:selected_axes=tuple(tuple(axis) for axis in axes)
        except TypeError:return _failure('Axes must contain valid bonded atom pairs.')
    for axis in selected_axes:
        if (len(axis)!=2 or not ids(axis) or axis[0]==axis[1]
                or graph.GetBondBetweenAtoms(*axis) is None):return _failure('An automatic axis must be a valid bonded atom pair.')
        if frozenset(axis) in used:continue
        used.add(frozenset(axis))
        bond=graph.GetBondBetweenAtoms(*axis)
        if bond.IsInRing() or bond.GetBondType()!=Chem.BondType.SINGLE:
            torsions.append(TorsionDescription(axis,reason='Automatic torsion selection requires a nonring single-bond axis.'))
        else:
            row=_auto_torsion(graph,Chem,coordinates,axis,distance_tolerance,angular_tolerance)
            special=_carbonyl_special(graph,Chem,coordinates,axis,distance_tolerance,angular_tolerance)
            if special:
                from dataclasses import replace
                kind,refs,angle=special
                if row.status=='unavailable':
                    row=_description(graph,Chem,coordinates,refs,(refs[1],refs[2]),
                        ('P-94.3.1.2',),distance_tolerance,angular_tolerance)
                row=replace(row,special_descriptor=kind,special_reference_atoms=refs,special_angle_degrees=angle)
            torsions.append(row)
    try:sequences=tuple(tuple(ring) for ring in (graph.GetRingInfo().AtomRings() if rings is None else rings))
    except TypeError:return _failure('Ring references must be ordered atom cycles.')
    ring_records=[]
    from .conformer_geometry import ring_shape
    for seq in sequences:
        if (len(seq)<3 or not ids(seq) or len(set(seq))!=len(seq)
                or any(graph.GetBondBetweenAtoms(a,b) is None for a,b in zip(seq,seq[1:]+seq[:1]))):
            return _failure('Each supplied ring sequence must be one simple bonded cycle.')
        orders=tuple(graph.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble() for a,b in zip(seq,seq[1:]+seq[:1]))
        shape,displaced,puckering,options=ring_shape([coordinates[i] for i in seq],orders,distance_tolerance,angular_tolerance)
        ring_records.append(RingConformation(seq,shape,tuple(seq[i] for i in displaced),puckering,
            'complete' if shape not in {'unclassified','ambiguous'} else shape,
            tuple(tuple(seq[i] for i in choice) for choice in options)))
    limitations=tuple(SelectionLimitation('conformation','P-94.2',row.reason) for row in torsions if row.status=='unavailable')
    limitations += tuple(SelectionLimitation('conformation','P-94.3.2',
        'Ring geometry does not prove a unique supported form at the supplied tolerances.')
        for row in ring_records if row.status!='complete')
    named=Chem.RemoveHs(graph)
    result=mol_to_iupac_result(named)
    return ConformerDescription('partial' if limitations else 'available',result,
        Chem.MolToSmiles(named,canonical=True,isomericSmiles=True),conformer.GetId(),coordinates,
        tuple(torsions),tuple(ring_records),limitations)
