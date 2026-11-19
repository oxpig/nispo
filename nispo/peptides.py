"""Explicit P-103 linear eupeptide sequence and terminal construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .naming_result import NamingResult,SelectionLimitation


# Table 10.4 and P-103.2.5 constituent vocabulary; no molecule/name lookup.
_RESIDUES={
    'Ala':('A','alanine','alanyl','alaninamide','alaninate'),
    'Arg':('R','arginine','arginyl','argininamide','argininate'),
    'Asn':('N','asparagine','asparaginyl','asparaginamide','asparaginate'),
    'Asp':('D','aspartic acid','aspartyl','aspartic 1-amide','aspartate'),
    'Cys':('C','cysteine','cysteinyl','cysteinamide','cysteinate'),
    'Gln':('Q','glutamine','glutaminyl','glutaminamide','glutaminate'),
    'Glu':('E','glutamic acid','glutamyl','glutamic 1-amide','glutamate'),
    'Gly':('G','glycine','glycyl','glycinamide','glycinate'),
    'His':('H','histidine','histidyl','histidinamide','histidinate'),
    'Ile':('I','isoleucine','isoleucyl','isoleucinamide','isoleucinate'),
    'Leu':('L','leucine','leucyl','leucinamide','leucinate'),
    'Lys':('K','lysine','lysyl','lysinamide','lysinate'),
    'Met':('M','methionine','methionyl','methioninamide','methioninate'),
    'Phe':('F','phenylalanine','phenylalanyl','phenylalaninamide','phenylalaninate'),
    'Pro':('P','proline','prolyl','prolinamide','prolinate'),
    'Ser':('S','serine','seryl','serinamide','serinate'),
    'Thr':('T','threonine','threonyl','threoninamide','threoninate'),
    'Trp':('W','tryptophan','tryptophyl','tryptophanamide','tryptophanate'),
    'Tyr':('Y','tyrosine','tyrosyl','tyrosinamide','tyrosinate'),
    'Val':('V','valine','valyl','valinamide','valinate'),
}
_SYMBOLS={symbol.lower():symbol for symbol in _RESIDUES}
_SYMBOLS.update({values[0].lower():symbol for symbol,values in _RESIDUES.items()})


@dataclass(frozen=True,slots=True)
class PeptideResidue:
    symbol: str
    configuration: Literal['L','D'] = 'L'


@dataclass(frozen=True,slots=True)
class PeptideSequence:
    residues: tuple[PeptideResidue|str,...]
    n_terminus: Literal['free','acetyl'] = 'free'
    c_terminus: Literal['acid','amide','methyl-ester'] = 'acid'


@dataclass(frozen=True,slots=True)
class PeptideResidueBinding:
    number: int
    symbol: str
    configuration: str
    atom_maps: tuple[int,...]
    nitrogen_map: int
    alpha_carbon_map: int
    carbonyl_carbon_map: int
    carbonyl_oxygen_map: int


@dataclass(frozen=True,slots=True)
class PeptideNamingResult(NamingResult):
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    sequence_symbols: str = ''
    residue_bindings: tuple[PeptideResidueBinding,...] = ()
    peptide_bond_maps: tuple[tuple[int,int],...] = ()
    cap_atom_maps: tuple[tuple[str,tuple[int,...]],...] = ()
    n_terminus: str = ''
    c_terminus: str = ''
    substance_form: str = 'conventional-neutral'
    nomenclature: str = 'specialized-P-103; PIN-not-assessed-under-P-100'


def _failure(status,reason):
    return PeptideNamingResult(name='', generation_status=status, preferred_status='not-assessed',
        unresolved=(SelectionLimitation('peptide-construction','P-103.2/P-103.3',reason),),
        route='explicit-peptide-sequence')


def _source_residue_roles(mol,sequence,Chem):
    rows=[];seen=set()
    for number,(symbol,configuration) in enumerate(sequence,1):
        atoms=tuple(a.GetIdx() for a in mol.GetAtoms() if a.GetPDBResidueInfo() is not None
                    and a.GetPDBResidueInfo().GetResidueNumber()==number)
        if not atoms or seen.intersection(atoms):return None
        seen.update(atoms)
        named={mol.GetAtomWithIdx(idx).GetPDBResidueInfo().GetName().strip():idx for idx in atoms}
        if not {'N','CA','C','O'}<=named.keys() or len(named)!=len(atoms):return None
        n,alpha,carbon,oxygen=(named[key] for key in ('N','CA','C','O'))
        if ([mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in (n,alpha,carbon,oxygen)]!=[7,6,6,8]
                or any(mol.GetBondBetweenAtoms(a,b) is None or mol.GetBondBetweenAtoms(a,b).GetBondType()!=order
                       for a,b,order in ((n,alpha,Chem.BondType.SINGLE),(alpha,carbon,Chem.BondType.SINGLE),(carbon,oxygen,Chem.BondType.DOUBLE)))):
            return None
        rows.append((atoms,named))
    if seen!=set(range(mol.GetNumAtoms())):return None
    for (_a,left),(_b,right) in zip(rows,rows[1:]):
        bond=mol.GetBondBetweenAtoms(left['C'],right['N'])
        if bond is None or bond.GetBondType()!=Chem.BondType.SINGLE:return None
    if any('OXT' in named for _atoms,named in rows[:-1]) or 'OXT' not in rows[-1][1]:return None
    terminal=rows[-1][1];oxygen=mol.GetAtomWithIdx(terminal['OXT'])
    if (oxygen.GetAtomicNum()!=8 or oxygen.GetDegree()!=1
            or oxygen.GetNeighbors()[0].GetIdx()!=terminal['C']):return None
    return tuple(rows)


def _conventional_histidine(mol,rows,sequence,Chem):
    # Table 10.4 specifies the 1H-imidazol-4-yl constitutional tautomer.
    # Its distal N (PDB NE2) bears H; the adjacent ND1 is the pyridine N.
    for (symbol,_configuration),(_atoms,named) in zip(sequence,rows,strict=True):
        if symbol!='His':continue
        if not {'ND1','NE2'}<=named.keys():return False
        for key,hydrogens in (('ND1',0),('NE2',1)):
            atom=mol.GetAtomWithIdx(named[key])
            if atom.GetAtomicNum()!=7 or not atom.GetIsAromatic():return False
            atom.SetNumExplicitHs(hydrogens);atom.SetNoImplicit(True)
    try:Chem.SanitizeMol(mol)
    except (ValueError,RuntimeError):return False
    return True


def _sequence_name(sequence,n_terminus,c_terminus):
    words=[]
    for index,(symbol,configuration) in enumerate(sequence):
        row=_RESIDUES[symbol]
        if index<len(sequence)-1:word=row[2]
        else:word=row[1 if c_terminus=='acid' else 3 if c_terminus=='amide' else 4]
        if configuration=='D':word='D-'+word
        if words and word.startswith('D-'):words.append('-')
        words.append(word)
    name=''.join(words)
    if n_terminus=='acetyl':name='N-acetyl'+('-' if name.startswith('D-') else '')+name
    if c_terminus=='methyl-ester':name=('1-methyl ' if sequence[-1][0] in {'Asp','Glu'} else 'methyl ')+name
    return name


def peptide_to_iupac_result(request:PeptideSequence) -> PeptideNamingResult:
    """Construct and name a declared conventional-neutral linear eupeptide.

    Standard L is implicit in specialized P-103 peptide names; D is explicit
    on each affected constituent. Neither a pH state nor a distribution is
    inferred. Result map numbers bind every residue to the serialized graph.
    Chapter P-100 does not certify these specialized names as PINs.
    """
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    if (not isinstance(request,PeptideSequence) or not isinstance(request.residues,tuple)
            or not 2<=len(request.residues)<=1000
            or not isinstance(request.n_terminus,str) or not isinstance(request.c_terminus,str)
            or request.n_terminus not in {'free','acetyl'}
            or request.c_terminus not in {'acid','amide','methyl-ester'}):
        return _failure('invalid-input','A two-to-1000-residue explicit sequence and supported terminal modes are required.')
    sequence=[]
    for item in request.residues:
        residue=PeptideResidue(item) if isinstance(item,str) else item
        if (not isinstance(residue,PeptideResidue) or not isinstance(residue.symbol,str)
                or not isinstance(residue.configuration,str)
                or residue.configuration not in {'L','D'} or residue.symbol.lower() not in _SYMBOLS):
            return _failure('invalid-input','Each constituent needs one common-residue symbol and an assigned L or D declaration.')
        symbol=_SYMBOLS[residue.symbol.lower()]
        if symbol=='Gly' and residue.configuration=='D':return _failure('invalid-input','Achiral glycine does not have a D configuration.')
        sequence.append((symbol,residue.configuration))
    sequence=tuple(sequence)
    try:graph=Chem.MolFromSequence(''.join(_RESIDUES[symbol][0] for symbol,_configuration in sequence),flavor=0)
    except (ValueError,RuntimeError,TypeError):graph=None
    if graph is None:return _failure('unavailable','The standard residue scaffold constructor is unavailable.')
    rows=_source_residue_roles(graph,sequence,Chem)
    if rows is None or not _conventional_histidine(graph,rows,sequence,Chem):
        return _failure('unavailable','The constructed scaffold does not prove the source residue and backbone roles.')
    for (_symbol,configuration),(atoms,_named) in zip(sequence,rows,strict=True):
        if configuration=='D':
            for idx in atoms:
                atom=graph.GetAtomWithIdx(idx)
                if atom.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED:atom.InvertChirality()
    editable=Chem.RWMol(graph);caps=[]
    terminal=rows[-1][1]
    if request.c_terminus=='amide':
        oxygen=editable.GetAtomWithIdx(terminal['OXT']);oxygen.SetAtomicNum(7)
        oxygen.SetNumExplicitHs(0);oxygen.SetNoImplicit(False)
    elif request.c_terminus=='methyl-ester':
        carbon=editable.AddAtom(Chem.Atom(6));editable.AddBond(terminal['OXT'],carbon,Chem.BondType.SINGLE)
        caps.append(('C-methyl-ester',(carbon+1,)))
    if request.n_terminus=='acetyl':
        nitrogen=rows[0][1]['N']
        if graph.GetAtomWithIdx(nitrogen).GetTotalNumHs()<1:return _failure('unavailable','The N terminus has no replaceable hydrogen.')
        carbon=editable.AddAtom(Chem.Atom(6));oxygen=editable.AddAtom(Chem.Atom(8));methyl=editable.AddAtom(Chem.Atom(6))
        editable.AddBond(nitrogen,carbon,Chem.BondType.SINGLE)
        editable.AddBond(carbon,oxygen,Chem.BondType.DOUBLE);editable.AddBond(carbon,methyl,Chem.BondType.SINGLE)
        caps.append(('N-acetyl',(carbon+1,oxygen+1,methyl+1)))
    graph=editable.GetMol()
    try:
        Chem.SanitizeMol(graph);Chem.AssignStereochemistry(graph,cleanIt=True,force=True)
        from rdkit.Chem import rdCIPLabeler
        rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
    except (ValueError,RuntimeError,OverflowError):return _failure('unavailable','The declared terminal derivative cannot complete valence/configuration validation.')
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in graph.GetAtoms()):
        return _failure('unavailable','The explicit conventional-neutral construction was not preserved.')
    for (symbol,configuration),(_atoms,named) in zip(sequence,rows,strict=True):
        expected=({'CA':'R'} if symbol=='Cys' else {'CA':'S'}) if symbol!='Gly' else {}
        if symbol in {'Ile','Thr'}:expected['CB']='S' if symbol=='Ile' else 'R'
        for key,code in expected.items():
            atom=graph.GetAtomWithIdx(named[key]);wanted=code if configuration=='L' else {'R':'S','S':'R'}[code]
            if not atom.HasProp('_CIPCode') or atom.GetProp('_CIPCode')!=wanted:
                return _failure('unavailable','The constructed residue does not retain the source L/D stereoparent configuration.')
    bindings=tuple(PeptideResidueBinding(number,symbol,'achiral' if symbol=='Gly' else configuration,
        tuple(idx+1 for idx in atoms),named['N']+1,named['CA']+1,named['C']+1,named['O']+1)
        for number,((symbol,configuration),(atoms,named)) in enumerate(zip(sequence,rows,strict=True),1))
    peptide_bonds=tuple((left.carbonyl_carbon_map,right.nitrogen_map) for left,right in zip(bindings,bindings[1:]))
    identity=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
    for atom in graph.GetAtoms():atom.SetAtomMapNum(atom.GetIdx()+1)
    mapped=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
    symbols='-'.join(('D-' if configuration=='D' else '')+symbol for symbol,configuration in sequence)
    return PeptideNamingResult(name=_sequence_name(sequence,request.n_terminus,request.c_terminus),
        generation_status='available', preferred_status='not-assessed', route='explicit-peptide-sequence',
        canonical_smiles=identity, mapped_smiles=mapped, sequence_symbols=symbols,
        residue_bindings=bindings, peptide_bond_maps=peptide_bonds, cap_atom_maps=tuple(caps),
        n_terminus=request.n_terminus, c_terminus=request.c_terminus)


def peptide_to_iupac(request:PeptideSequence) -> str:
    return peptide_to_iupac_result(request).name
