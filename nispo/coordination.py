"""Explicit P-69 mononuclear ligand identity, kappa/eta binding and ocenes."""
from __future__ import annotations

from collections import Counter,defaultdict
from dataclasses import dataclass

from .naming_result import NamingResult,SelectionLimitation
from .polyhedral_configuration import PolyhedralGeometry,PolyhedralDescription,describe_polyhedral_configuration


@dataclass(frozen=True,slots=True)
class CoordinationLigand:
    smiles: str
    donor_atoms: tuple[int,...]
    mode: str = 'kappa'


@dataclass(frozen=True,slots=True)
class CoordinationGeometry:
    symbol: str
    # One direction for each original input ligand, all monodentate kappa.
    ligand_directions: tuple[tuple[float,float,float],...]
    angular_tolerance_degrees: float = 10.


@dataclass(frozen=True,slots=True)
class CoordinationSpec:
    metal: str
    ligands: tuple[CoordinationLigand,...]
    metal_formal_charge: int = 0
    oxidation_state: int|None = None
    geometry: CoordinationGeometry|None = None


@dataclass(frozen=True,slots=True)
class CoordinationBinding:
    mode: str
    metal_map: int
    donor_atom_maps: tuple[int,...]
    ligand_atom_maps: tuple[int,...]
    ligand_name: str
    free_ligand_smiles: str
    free_ligand_charge: int
    original_donor_indices: tuple[int,...] = ()


@dataclass(frozen=True,slots=True)
class CoordinationNamingResult(NamingResult):
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    metal_atom_map: int|None = None
    bindings: tuple[CoordinationBinding,...] = ()
    total_formal_charge: int = 0
    metal_formal_charge: int = 0
    oxidation_state: int|None = None
    configuration: PolyhedralDescription|None = None
    native_stereo_smiles: str = ''
    bonding_model: str = 'expanded-native-dative; eta-binding-is-one-declared-multicentre-unit'
    preferred_name_scope: str = 'P-69.0: transition-metal PINs not assigned except source retained ocenes'


class _Input(ValueError):pass
class _Scope(ValueError):pass


_METALS={'Sc':'scandium','Ti':'titanium','V':'vanadium','Cr':'chromium','Mn':'manganese','Fe':'iron','Co':'cobalt','Ni':'nickel',
         'Cu':'copper','Zn':'zinc','Y':'yttrium','Zr':'zirconium','Nb':'niobium','Mo':'molybdenum','Tc':'technetium',
         'Ru':'ruthenium','Rh':'rhodium','Pd':'palladium','Ag':'silver','Cd':'cadmium','Hf':'hafnium','Ta':'tantalum',
         'W':'tungsten','Re':'rhenium','Os':'osmium','Ir':'iridium','Pt':'platinum','Au':'gold','Hg':'mercury'}
_OCENES={'V':'vanadocene','Cr':'chromocene','Fe':'ferrocene','Co':'cobaltocene','Ni':'nickelocene','Ru':'ruthenocene','Os':'osmocene'}
_ANION_NAMES={'V':'vanadate','Cr':'chromate','Mn':'manganate','Fe':'ferrate','Co':'cobaltate','Ni':'nickelate','Cu':'cuprate','Zn':'zincate',
              'Mo':'molybdate','Ru':'ruthenate','Rh':'rhodate','Pd':'palladate','Ag':'argentate','Cd':'cadmate','W':'tungstate',
              'Os':'osmate','Ir':'iridate','Pt':'platinate','Au':'aurate','Hg':'mercurate'}


def _failure(status,reason):
    return CoordinationNamingResult('',status,'not-assessed',
        (SelectionLimitation('coordination-construction','P-69.2',reason),),'explicit-coordination')


def _complete_unannotated_ligand_witness(graph,named,Chem):
    """Check bounded, whole-graph witnesses for unresolved free names.

    A molecular name's availability alone is not evidence that a fallback
    owns the entire ligand. The admitted families are an unbranched saturated
    dialkyl sulfide and a carbon-substituted mononuclear phosphane. Require
    an independently reconstructed complete method decision, and the very
    same delivered name. This says nothing about preferred-parent selection.
    """
    if (named.generation_status!='available' or not named.name
            or any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or a.GetAtomicNum()==1 or a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED
                   for a in graph.GetAtoms())
            or any(int(b.GetStereo()) or b.GetBondDir()!=Chem.BondDir.NONE for b in graph.GetBonds())):
        return False
    from .naming_methods import NamingMethodStatus,select_naming_method_outcome
    from .prefix_presentation import preferred_prefix_presentation
    outcome=select_naming_method_outcome(Chem.Mol(graph),Chem)
    decision=outcome.decision
    all_atoms=frozenset(range(graph.GetNumAtoms()))
    if (outcome.status!=NamingMethodStatus.COMPLETE or decision is None
            or decision.atoms!=all_atoms
            or preferred_prefix_presentation(decision.name)!=named.name
            or graph.GetRingInfo().NumRings()
            or any(b.GetBondType()!=Chem.BondType.SINGLE for b in graph.GetBonds())):
        return False
    atoms=tuple(graph.GetAtoms())
    sulfurs=[a for a in atoms if a.GetAtomicNum()==16]
    if (len(sulfurs)==1 and all(a.GetAtomicNum() in {6,16} for a in atoms)
            and decision.rule=='P-51.1.1/P-51.4.1.1'):
        sulfur=sulfurs[0]
        ends=[a for a in atoms if a.GetDegree()==1]
        return (len(ends)==2 and all(a.GetAtomicNum()==6 for a in ends)
            and all(a.GetDegree()<=2 for a in atoms)
            and sulfur.GetDegree()==2 and all(a.GetAtomicNum()==6 for a in sulfur.GetNeighbors()))
    phosphorus=[a for a in atoms if a.GetAtomicNum()==15]
    if (len(phosphorus)!=1 or any(a.GetAtomicNum() not in {6,15} for a in atoms)
            or not 1<=phosphorus[0].GetDegree()<=3):
        return False
    # The mononuclear Group 15 method partitions the central parent and
    # every organyl branch; reject other producers and overlapping scopes.
    roles=decision.component_roles
    if not roles or roles[0][0]!='group15-parent-suffix' or frozenset(roles[0][1])!={phosphorus[0].GetIdx()}:
        return False
    if any(not role[0].startswith('organyl-') for role in roles[1:]):return False
    covered=set()
    for _kind,scope,_locants in roles:
        if covered.intersection(scope):return False
        covered.update(scope)
    return covered==set(all_atoms)


def _ligand_name(graph,donors,mode,Chem):
    from .api import mol_to_iupac_result
    charge=sum(a.GetFormalCharge() for a in graph.GetAtoms());atoms=tuple(graph.GetAtoms());numbers=Counter(a.GetAtomicNum() for a in atoms)
    hydrogen=sum(a.GetTotalNumHs(includeNeighbors=False) for a in atoms)+sum(a.GetAtomicNum()==1 for a in atoms)
    identity=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
    special_isotope=any(a.GetIsotope() for a in atoms) and (len(atoms)<=2 or numbers in ({6:5},{6:6},{6:3}) and charge==-1)
    if special_isotope:raise _Scope('Isotope decoration of a special coordination parent needs its own mapped ligand writer.')
    # Vocabulary is graph/charge bound. The entire metal complex is never a
    # template match, and these free parent ligands remain productive units.
    if len(atoms)==1:
        atom=atoms[0];z=atom.GetAtomicNum()
        if z==8 and charge==0 and hydrogen==2:return 'aqua',False,False,()
        if z==7 and charge==0 and hydrogen==3:return 'ammine',False,False,()
        if z==1 and charge==-1:return 'hydrido',False,False,()
        if z==8 and charge==-1 and hydrogen==1:return 'hydroxido',False,False,()
        if z==8 and charge==-2 and hydrogen==0:return 'oxido',False,False,()
        if z in {9,17,35,53} and charge==-1:return {9:'fluorido',17:'chlorido',35:'bromido',53:'iodido'}[z],False,False,()
    if len(atoms)==2 and numbers=={6:1,8:1} and hydrogen==0 and charge==0:
        carbon=next(a for a in atoms if a.GetAtomicNum()==6);oxygen=next(a for a in atoms if a.GetAtomicNum()==8)
        if donors==(carbon.GetIdx(),) and carbon.GetFormalCharge()==-1 and oxygen.GetFormalCharge()==1 and graph.GetBondBetweenAtoms(0,1).GetBondType()==Chem.BondType.TRIPLE:
            return 'carbonyl',False,False,()
    if len(atoms)==2 and numbers=={7:1,8:1} and hydrogen==0 and charge in {0,1}:
        nitrogen=next(a for a in atoms if a.GetAtomicNum()==7)
        if donors==(nitrogen.GetIdx(),):return 'nitrosyl',False,False,()
    if charge==-1 and numbers=={6:5} and hydrogen==5 and all(a.GetIsAromatic() for a in atoms) and graph.GetNumBonds()==5:
        return 'cyclopenta-2,4-dien-1-ido',True,True,()
    if charge==-1 and numbers=={6:6} and hydrogen==5 and all(a.GetIsAromatic() for a in atoms) and graph.GetNumBonds()==6:
        return 'benzenido',False,False,()
    if (charge==-1 and numbers=={6:3} and hydrogen==5 and graph.GetNumBonds()==2
            and sorted(b.GetBondTypeAsDouble() for b in graph.GetBonds())==[1.,2.]):
        return 'prop-2-en-1-ido',True,False,()
    named=mol_to_iupac_result(graph)
    if not named.name:raise _Scope('A complete free ligand has no supported molecular name.')
    name=named.name
    if charge<0:
        if not name.endswith(('ate','ite','ide')):raise _Scope('The organic anion ligand ending has no source ato/ito/ido conversion.')
        name=name[:-1]+'o'
    # A selected general name may be structurally complete while its parent
    # preference remains unresolved. Admit only a separately reconstructed
    # whole-graph method witness and retain, never discharge, the obligations.
    if named.unresolved and not _complete_unannotated_ligand_witness(graph,named,Chem):
        raise _Scope('A free ligand with unresolved preference lacks a complete matching unannotated method witness.')
    return name,any(c.isdigit() or c in '()[]' for c in name),False,named.unresolved


def _prepare(ligand,Chem):
    if (not isinstance(ligand,CoordinationLigand) or not isinstance(ligand.smiles,str)
            or not isinstance(ligand.donor_atoms,tuple) or not isinstance(ligand.mode,str) or ligand.mode not in {'kappa','eta'}):
        raise _Input('Declare a complete free-ligand SMILES, original donor indices and kappa/eta binding.')
    graph=Chem.MolFromSmiles(ligand.smiles)
    if graph is None or not graph.GetNumAtoms() or len(Chem.GetMolFrags(graph))!=1:raise _Input('A free ligand must be one connected explicit graph.')
    if graph.GetStereoGroups():raise _Scope('Enhanced ligand substances need a joint coordination-substance writer.')
    if any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<graph.GetNumAtoms() for i in ligand.donor_atoms):
        raise _Input('Donor indices must be distinct original atoms of the free ligand.')
    donors=tuple(sorted(ligand.donor_atoms))
    if not donors or len(set(donors))!=len(donors):
        raise _Input('Donor indices must be distinct original atoms of the free ligand.')
    if any(a.GetAtomicNum() not in {1,5,6,7,8,9,14,15,16,17,32,33,34,35,52,53} for a in graph.GetAtoms()):
        raise _Scope('The free ligand contains an unsupported metal or element.')
    for atom in graph.GetAtoms():atom.SetAtomMapNum(0)
    if any(a.GetChiralTag() not in {Chem.ChiralType.CHI_UNSPECIFIED,Chem.ChiralType.CHI_TETRAHEDRAL_CCW,Chem.ChiralType.CHI_TETRAHEDRAL_CW} for a in graph.GetAtoms()
            ) or any(b.GetStereo() in {Chem.BondStereo.STEREOATROPCW,Chem.BondStereo.STEREOATROPCCW} for b in graph.GetBonds()):
        raise _Scope('Advanced ligand geometry needs a joint coordinate descriptor writer.')
    name,compound,cp,issues=_ligand_name(graph,donors,ligand.mode,Chem)
    if any(a.GetNumRadicalElectrons() for a in graph.GetAtoms()) and name!='nitrosyl':raise _Scope('Radical ligands need an explicit joint electron/spin writer.')
    suffix=''
    if ligand.mode=='eta':
        # Full heavy-atom hapticity avoids guessing ligand locants. Partial
        # eta sets require their original committed parent numbering.
        heavy={a.GetIdx() for a in graph.GetAtoms() if a.GetAtomicNum()>1}
        if not 2<=len(donors)<=8 or set(donors)!=heavy or any(graph.GetAtomWithIdx(i).GetAtomicNum()!=6 for i in donors):
            raise _Scope('This eta writer covers complete two-to-eight-carbon ligands; partial hapticity remains unproved.')
        if not any(b.GetBondTypeAsDouble()>1 for b in graph.GetBonds()):raise _Scope('The supplied eta ligand has no unsaturated carbon framework.')
        if any(graph.GetAtomWithIdx(i).GetHybridization() not in {Chem.HybridizationType.SP,Chem.HybridizationType.SP2}
                and graph.GetAtomWithIdx(i).GetFormalCharge()!=-1 for i in donors):
            raise _Scope('Every atom in the full eta set needs an established unsaturated or carbanionic site.')
        rendered='η^'+str(len(donors))+'-'+name;compound=True
    else:
        if len(donors)!=1:raise _Scope('Multiple donor locants and chelate priority need a mapped kappa writer.')
        donor=graph.GetAtomWithIdx(donors[0]);z=donor.GetAtomicNum()
        if z==6:
            if name!='carbonyl' and donor.GetFormalCharge()>=0:raise _Scope('A neutral carbon sigma donor is not established by this model.')
        elif z not in {1,7,8,9,15,16,17,33,34,35,52,53}:raise _Scope('The source kappa donor element is not covered.')
        if z in {7,8,15,16,33,34,52} and name not in {'aqua','ammine','nitrosyl','hydroxido','oxido'}:
            if sum(a.GetAtomicNum()==z for a in graph.GetAtoms())!=1:
                raise _Scope('Multiple identical donor elements require source original ligand locants.')
            if donor.GetFormalCharge()>0:raise _Scope('The requested positive donor has no established lone-pair binding.')
            suffix='-κ'+donor.GetSymbol();compound=True
        rendered=name+suffix
    identity=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
    return graph,donors,name,rendered,compound,cp,identity,issues


def _retained_ocene(metal,metal_charge,rows):
    return _OCENES.get(metal) if metal_charge==2 and len(rows)==2 and all(row[5] and len(row[1])==5 and row[3].startswith('η^5-') for row in rows) else None


def coordination_to_iupac_result(spec:CoordinationSpec) -> CoordinationNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        if (not isinstance(spec,CoordinationSpec) or not isinstance(spec.metal,str) or spec.metal not in _METALS
                or not isinstance(spec.ligands,tuple) or not 1<=len(spec.ligands)<=12
                or isinstance(spec.metal_formal_charge,bool) or not isinstance(spec.metal_formal_charge,int) or not -8<=spec.metal_formal_charge<=8
                or spec.oxidation_state is not None and (isinstance(spec.oxidation_state,bool) or not isinstance(spec.oxidation_state,int))):
            raise _Input('Declare one covered Group 3–12 metal, one-to-twelve explicit ligands and an integer metal formal charge.')
        if spec.oxidation_state is not None and spec.oxidation_state!=spec.metal_formal_charge:
            raise _Scope('The ionic/dative graph model requires the stated metal oxidation state to agree with its formal charge; noninnocent partitions remain explicit.')
        # Preserve the original index at position 7 for geometry mapping;
        # free-ligand preference limitations follow it as a separate field.
        rows=[]
        for index,ligand in enumerate(spec.ligands):
            prepared=_prepare(ligand,Chem)
            rows.append((*prepared[:7],index,prepared[7]))
        from .formatting import prefix_alphabetization_key,enclose_name_unit
        from .constants import MULT
        rows.sort(key=lambda row:(prefix_alphabetization_key(row[2]),row[6],row[3]))
        if not any(a.GetAtomicNum()==6 and a.GetIdx() in donors for graph,donors,*_rest in rows for a in graph.GetAtoms()):
            raise _Scope('A P-69 organometallic declaration requires an actual carbon/metal binding.')
        graph=Chem.RWMol();metal=Chem.Atom(Chem.GetPeriodicTable().GetAtomicNumber(spec.metal));metal.SetFormalCharge(spec.metal_formal_charge)
        metal.SetNoImplicit(True);metal_index=graph.AddAtom(metal);bindings=[]
        for ligand,donors,name,rendered,compound,cp,identity,original_index,_issues in rows:
            offset=graph.GetNumAtoms();graph.InsertMol(ligand)
            for idx in donors:graph.AddBond(offset+idx,metal_index,Chem.BondType.DATIVE)
            bindings.append(CoordinationBinding('eta' if rendered.startswith('η') else 'kappa',metal_index+1,
                tuple(offset+i+1 for i in donors),tuple(range(offset+1,offset+ligand.GetNumAtoms()+1)),rendered,identity,
                sum(a.GetFormalCharge() for a in ligand.GetAtoms()),donors))
        graph=graph.GetMol();Chem.SanitizeMol(graph)
        if any(a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED for a in graph.GetAtoms()) or any(int(b.GetStereo()) for b in graph.GetBonds()):
            from rdkit.Chem import rdCIPLabeler
            rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
            for row,binding in zip(rows,bindings,strict=True):
                free=Chem.Mol(row[0]);rdCIPLabeler.AssignCIPLabels(free,maxRecursiveIterations=1000000)
                for atom in free.GetAtoms():
                    if atom.GetChiralTag()==Chem.ChiralType.CHI_UNSPECIFIED:continue
                    bound=graph.GetAtomWithIdx(binding.ligand_atom_maps[atom.GetIdx()]-1)
                    if not atom.HasProp('_CIPCode') or not bound.HasProp('_CIPCode') or atom.GetProp('_CIPCode')!=bound.GetProp('_CIPCode'):
                        raise _Scope('Coordination changes a ligand CIP descriptor; a contextual original-graph writer is required.')
                for bond in free.GetBonds():
                    if not int(bond.GetStereo()):continue
                    bound=graph.GetBondBetweenAtoms(binding.ligand_atom_maps[bond.GetBeginAtomIdx()]-1,binding.ligand_atom_maps[bond.GetEndAtomIdx()]-1)
                    if not bond.HasProp('_CIPCode') or not bound.HasProp('_CIPCode') or bond.GetProp('_CIPCode')!=bound.GetProp('_CIPCode'):
                        raise _Scope('Coordination changes a ligand double-bond descriptor; contextual priority remains unproved.')
        total=sum(a.GetFormalCharge() for a in graph.GetAtoms())
        retained=_retained_ocene(spec.metal,spec.metal_formal_charge,rows)
        if retained:name=retained
        else:
            grouped=defaultdict(list)
            for row in rows:grouped[row[2],row[3],row[4]].append(row)
            words=[]
            for (base,rendered,compound),values in grouped.items():
                if len({row[6] for row in values})!=1:raise _Scope('Distinct ligand identities have the same name; multiplication would obscure their states.')
                n=len(values)
                if compound:
                    multiplier='' if n==1 else {2:'bis',3:'tris',4:'tetrakis'}.get(n,MULT.get(n,'')+'kis')
                    words.append(multiplier+enclose_name_unit(rendered))
                else:words.append(('' if n==1 else MULT[n])+rendered)
            if total<0 and spec.metal not in _ANION_NAMES:raise _Scope('The anionic central-metal ate form is outside the validated vocabulary.')
            name=''.join(words)+(_ANION_NAMES[spec.metal] if total<0 else _METALS[spec.metal])
            if total:name+='('+str(abs(total))+('+' if total>0 else '−')+')'
        identity=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
        configuration=None
        if spec.geometry is not None:
            if (not isinstance(spec.geometry,CoordinationGeometry) or not isinstance(spec.geometry.ligand_directions,tuple)
                    or len(spec.geometry.ligand_directions)!=len(rows)):
                raise _Input('Declare one original direction for each input ligand in the typed coordination geometry.')
            if any(binding.mode!='kappa' or len(binding.donor_atom_maps)!=1 for binding in bindings):
                raise _Scope('Polyhedral sites require individual monodentate kappa ligands; eta/chelate geometry needs another model.')
            geometry=PolyhedralGeometry(spec.geometry.symbol,tuple(binding.donor_atom_maps[0]-1 for binding in bindings),
                tuple(spec.geometry.ligand_directions[row[7]] for row in rows),spec.geometry.angular_tolerance_degrees)
            configuration=describe_polyhedral_configuration(graph,metal_index,geometry)
            if configuration.status!='complete':raise _Scope('; '.join(issue.reason for issue in configuration.limitations))
            if configuration.descriptor:name='('+configuration.descriptor+')-'+name
        for atom in graph.GetAtoms():atom.SetAtomMapNum(atom.GetIdx()+1)
        mapped=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True);restored=Chem.MolFromSmiles(mapped)
        if restored is None:raise _Scope('The original expanded dative graph cannot be recovered.')
        for atom in restored.GetAtoms():atom.SetAtomMapNum(0)
        if Chem.MolToSmiles(restored,canonical=True,isomericSmiles=True)!=identity:
            raise _Scope('The mapped dative serialization does not conserve ligand state.')
        ligand_issues=tuple(sorted({SelectionLimitation('coordination-ligand/'+issue.stage,issue.rule,
            'Free ligand '+row[6]+': '+issue.reason)
            for row in rows for issue in row[8]},key=lambda issue:(issue.stage,issue.rule,issue.reason)))
        return CoordinationNamingResult(name=name,generation_status='available',preferred_status='resolved' if retained and not ligand_issues else 'not-assessed',
            unresolved=ligand_issues,route='explicit-coordination',canonical_smiles=identity,mapped_smiles=mapped,metal_atom_map=metal_index+1,bindings=tuple(bindings),
            total_formal_charge=total,metal_formal_charge=spec.metal_formal_charge,oxidation_state=spec.oxidation_state,
            configuration=configuration,native_stereo_smiles=configuration.native_stereo_smiles if configuration is not None else '')
    except _Input as error:return _failure('invalid-input',str(error))
    except (_Scope,ValueError,RuntimeError,OverflowError) as error:return _failure('unavailable',str(error))


def coordination_to_iupac(spec:CoordinationSpec) -> str:return coordination_to_iupac_result(spec).name


def native_ocene_decision(mol,Chem):
    """Read the full supplied native η5 contact set, never disconnected ions."""
    if mol.GetNumAtoms()!=11 or mol.GetNumBonds()!=20:return None
    metals=[a for a in mol.GetAtoms() if a.GetSymbol() in _OCENES]
    if len(metals)!=1:return None
    metal=metals[0]
    if metal.GetFormalCharge()!=2 or metal.GetDegree()!=10:return None
    if mol.GetStereoGroups() or any(a.GetIsotope() or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) for a in mol.GetAtoms()):return None
    if any(b.GetBondType()!=Chem.BondType.DATIVE or b.GetEndAtomIdx()!=metal.GetIdx() for b in metal.GetBonds()):return None
    copy=Chem.RWMol(mol);copy.RemoveAtom(metal.GetIdx());free=copy.GetMol();Chem.SanitizeMol(free)
    pieces=Chem.GetMolFrags(free,asMols=True)
    if len(pieces)!=2:return None
    if any(p.GetNumAtoms()!=5 or p.GetNumBonds()!=5 or sum(a.GetFormalCharge() for a in p.GetAtoms())!=-1
            or sum(a.GetTotalNumHs() for a in p.GetAtoms())!=5 or any(a.GetAtomicNum()!=6 or not a.GetIsAromatic() for a in p.GetAtoms()) for p in pieces):return None
    from .naming_methods import MethodDecision
    return MethodDecision(_OCENES[metal.GetSymbol()],'coordination',frozenset(),frozenset(range(11)),
        'P-69.2.7',atom_locants=((metal.GetIdx(),'metal'),),component_roles=(('retained-ocene-native-contact-graph',tuple(range(11)),()),))
