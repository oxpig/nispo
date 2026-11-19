"""P-75 global electron-transfer states and mapped carbon radical ions."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .naming_result import NamingResult, SelectionLimitation


@dataclass(frozen=True, slots=True)
class ElectronTransfer:
    """A neutral molecular formula with globally added (positive) electrons.

    ``radical_electrons`` declares the unpaired-electron count. Localization
    and a spin coupling are not inferred from the electron-transfer count.
    """
    neutral_smiles: str
    electrons_added: int
    radical_electrons: int = 1


@dataclass(frozen=True, slots=True)
class ElectronTransferNamingResult(NamingResult):
    neutral_smiles: str = ""
    electrons_added: int = 0
    charge: int = 0
    radical_electrons: int = 0
    localization: str = "global-unspecified"
    molecular_formula: str = ""


def _transfer_failure(status, reason):
    return ElectronTransferNamingResult(name='', generation_status=status,
        preferred_status='unresolved',
        unresolved=(SelectionLimitation('electron-transfer','P-75.1',reason),),
        route='electron-transfer')


def electron_transfer_to_iupac_result(state: ElectronTransfer) -> ElectronTransferNamingResult:
    """Name a declared global radical ion of a supported neutral hydride.

    This API retains a global electron state, rather than manufacturing a
    localized charge/radical assignment in a molecular SMILES.
    """
    from .api import Chem, mol_to_iupac_result
    if Chem is None:
        return _transfer_failure('unavailable','RDKit is unavailable.')
    if (not isinstance(state,ElectronTransfer) or not isinstance(state.neutral_smiles,str)
            or type(state.electrons_added) is not int or not state.electrons_added
            or type(state.radical_electrons) is not int or state.radical_electrons < 1
            or (state.radical_electrons-abs(state.electrons_added)) % 2):
        return _transfer_failure('invalid-input','Nonzero integer electron transfer and a compatible positive unpaired-electron count are required.')
    try:
        mol=Chem.MolFromSmiles(state.neutral_smiles)
    except (TypeError,ValueError):
        mol=None
    if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol))!=1:
        return _transfer_failure('invalid-input','A connected neutral parent structure is required.')
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return _transfer_failure('invalid-input','The precursor must be neutral and closed-shell.')
    electrons=sum(a.GetAtomicNum()+a.GetTotalNumHs(includeNeighbors=False)
                  for a in mol.GetAtoms())+state.electrons_added
    if electrons<state.radical_electrons:
        return _transfer_failure('invalid-input','The declared ion has fewer electrons than the declared unpaired-electron count.')
    if mol.GetStereoGroups():
        return _transfer_failure('unavailable','Enhanced precursor configuration needs a separate ensemble operation.')
    # This bounded elium/elide route cannot append another suffix to an
    # existing characteristic-group suffix (P-75.1 note). A functional-class
    # radical-ion name is a separate operation, not a guessed elium name.
    from .extended_inventory import selection_inventory
    inventory=selection_inventory(mol)
    if not inventory.supported or inventory.groups:
        return _transfer_failure('unavailable','An independently suffixed characteristic group needs the P-75.3 operation.')
    from .constants import MULT
    count=abs(state.electrons_added)
    multiplier=MULT.get(count) if count>1 else ''
    if multiplier is None:
        return _transfer_failure('unavailable','The electron-transfer multiplier is outside the supported vocabulary.')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    parent=mol_to_iupac_result(mol)
    if not parent.name:
        return _transfer_failure('unavailable','The neutral precursor has no molecular name.')
    term='elide' if state.electrons_added>0 else 'elium'
    # Elide e before the vowel of el-; a di-/tri- multiplier begins with a
    # consonant. Parent assemblies ending in yl retain their complete unit
    # boundary, as in the P-75.1 biphenyl example.
    stem=parent.name[:-1] if parent.name.endswith('e') and not multiplier else parent.name
    if parent.name.endswith('yl'):
        stem='['+stem+']'
    name=stem+multiplier+term
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    return ElectronTransferNamingResult(name=name, generation_status='available',
        preferred_status=parent.preferred_status, unresolved=parent.unresolved,
        route='electron-transfer', neutral_smiles=Chem.MolToSmiles(mol,isomericSmiles=True),
        electrons_added=state.electrons_added, charge=-state.electrons_added,
        radical_electrons=state.radical_electrons, localization='global-unspecified',
        molecular_formula=CalcMolFormula(mol))


def electron_transfer_to_iupac(state: ElectronTransfer) -> str:
    return electron_transfer_to_iupac_result(state).name


def carbon_radical_ion_decision(mol, Chem):
    """P-75.2 suffix composition on an ordinary saturated carbon tree.

    A single parent path must own every radical and ionic centre. All other
    components are neutral ordinary alkyl prefixes. Unknown competing parent
    names decline rather than using traversal order to choose a spelling.
    """
    if (mol is None or not 1<=mol.GetNumHeavyAtoms()<=64
            or mol.GetNumAtoms()!=mol.GetNumHeavyAtoms()
            or mol.GetRingInfo().NumRings() or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol))!=1):
        return None
    atoms=tuple(mol.GetAtoms())
    if any(a.GetAtomicNum()!=6 or a.GetIsotope() or int(a.GetChiralTag())
           or a.GetFormalCharge() not in {-1,0,1}
           or a.GetNumRadicalElectrons() not in {0,1} for a in atoms):
        return None
    if any(b.GetBondType()!=Chem.BondType.SINGLE or int(b.GetStereo()) or int(b.GetBondDir())
           for b in mol.GetBonds()):
        return None
    radicals=frozenset(a.GetIdx() for a in atoms if a.GetNumRadicalElectrons())
    positive=frozenset(a.GetIdx() for a in atoms if a.GetFormalCharge()>0)
    negative=frozenset(a.GetIdx() for a in atoms if a.GetFormalCharge()<0)
    if not radicals or not positive|negative or mol.GetNumBonds()!=len(atoms)-1:
        return None
    # At ordinary C, a hydride/hydron loss creates each +/- centre and
    # each H-atom loss creates one radical electron. Check original H/valence.
    for atom in atoms:
        expected=4-abs(atom.GetFormalCharge())-atom.GetNumRadicalElectrons()
        if (atom.GetTotalValence()!=expected
                or atom.GetTotalNumHs(includeNeighbors=True)!=expected-atom.GetDegree()):
            return None
    from .constants import ROOTS, MULT
    from .formatting import attach_suffix, format_prefixes, join_prefix_parent, prefix_alphabetization_key
    from .component_mapping import mapped_branch, fully_covered
    from .naming_methods import MethodDecision
    graph={a.GetIdx():frozenset(n.GetIdx() for n in a.GetNeighbors()) for a in atoms}
    leaves=tuple(i for i,neighbors in graph.items() if len(neighbors)<=1)
    paths=[]
    endpoints=combinations(leaves,2) if len(atoms)>1 else ((0,0),)
    centres=radicals|positive|negative
    for left,right in endpoints:
        stack=[(left,(left,))]
        while stack:
            current,path=stack.pop()
            if current==right:
                if centres<=set(path):
                    paths.extend((path,tuple(reversed(path))))
                break
            stack.extend((neighbor,path+(neighbor,)) for neighbor in graph[current]-set(path))
    if not paths:
        return None
    longest=max(map(len,paths))
    paths=tuple(set(path for path in paths if len(path)==longest))
    candidates=[]
    for path in paths:
        if len(path) not in ROOTS:
            return None
        mapping={idx:str(pos) for pos,idx in enumerate(path,1)}
        rloc=tuple(sorted(int(mapping[i]) for i in radicals))
        nloc=tuple(sorted(int(mapping[i]) for i in negative))
        ploc=tuple(sorted(int(mapping[i]) for i in positive))
        prefixes=[];roles=[('parent-hydride',frozenset(path),tuple(mapping.items()))]
        owned=set(path)
        for host in path:
            for root in graph[host]-set(path):
                text,branch=mapped_branch(mol,Chem,root,host)
                if not text or not branch or branch&owned or branch&centres:
                    return None
                prefixes.append((mapping[host],text))
                roles.append(('alkyl-prefix',frozenset(branch),()))
                owned.update(branch)
        if not fully_covered(mol,owned,frozenset()):
            return None
        if len(path)==1:
            parent='methyliumyl' if positive else 'methanidyl'
        else:
            parent=ROOTS[len(path)]+'ane'
            for locants,term in ((ploc,'ylium'),(nloc,'ide')):
                if locants:
                    multiplier=MULT.get(len(locants)) if len(locants)>1 else ''
                    if multiplier is None:
                        return None
                    parent=attach_suffix(parent,multiplier+term,[str(i) for i in locants])
            multiplier=MULT.get(len(rloc)) if len(rloc)>1 else ''
            if multiplier is None:
                return None
            if parent.endswith('e'):
                parent=parent[:-1]
            parent=attach_suffix(parent,multiplier+'yl',[str(i) for i in rloc])
        name=join_prefix_parent(format_prefixes(prefixes),parent) if prefixes else parent
        citation=tuple(int(loc) for loc,text in sorted(prefixes,key=lambda row:(prefix_alphabetization_key(row[1]),int(row[0]))))
        rank=(rloc,nloc,ploc,-len(prefixes),tuple(sorted(int(loc) for loc,_ in prefixes)),citation)
        candidates.append((rank,name,path,tuple(roles)))
    best=min(row[0] for row in candidates)
    winners=[row for row in candidates if row[0]==best]
    if len({row[1] for row in winners})!=1:
        return None
    _,name,path,roles=winners[0]
    return MethodDecision(name,'radical-ion',radicals,frozenset(graph),
        'P-75.2/P-71.2/P-72.2/P-73.2/P-46.1',
        tuple((idx,str(pos)) for pos,idx in enumerate(path,1)),component_roles=roles)


def characteristic_radical_ion_decision(mol, Chem):
    """P-75.3 one H-loss radical on the unique N/O/S ionic centre."""
    atoms=tuple(mol.GetAtoms())
    centres=[a for a in atoms if a.GetNumRadicalElectrons() or a.GetFormalCharge()]
    if (len(centres)!=1 or len(Chem.GetMolFrags(mol))!=1 or mol.GetStereoGroups()
            or any(a.GetAtomicNum()==1 or a.GetIsotope() or int(a.GetChiralTag()) for a in atoms)
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    centre=centres[0]
    number=centre.GetAtomicNum();charge=centre.GetFormalCharge()
    target={(7,1):4,(7,-1):2,(8,1):3,(16,1):3}.get((number,charge))
    if (target is None or centre.GetNumRadicalElectrons()!=1
            or centre.GetTotalValence()!=target-1
            or not centre.GetDegree()
            or any(n.GetAtomicNum()!=6 for n in centre.GetNeighbors())):
        return None
    ionic=Chem.Mol(mol)
    completed=ionic.GetAtomWithIdx(centre.GetIdx())
    completed.SetNumRadicalElectrons(0)
    completed.SetNumExplicitHs(centre.GetTotalNumHs(includeNeighbors=True)+1)
    completed.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(ionic)
    except Exception:
        return None
    if (any(a.GetNumRadicalElectrons() for a in ionic.GetAtoms())
            or completed.GetTotalValence()!=target
            or completed.GetTotalNumHs(includeNeighbors=True)!=centre.GetTotalNumHs(includeNeighbors=True)+1):
        return None
    from .api import mol_to_iupac_result
    parent=mol_to_iupac_result(ionic)
    # P-75.3.1 explicitly selects the characteristic aminium parent for the
    # single phenyl-N radical cation. Its ordinary closed-ion legacy route
    # may still return an azaniumyl prefix on benzene; that prefix is not the
    # characteristic suffix to which yl can be appended.
    phenyl_n=(number==7 and charge==1 and centre.GetDegree()==1
              and len(atoms)==7 and mol.GetRingInfo().NumRings()==1
              and all(a.GetAtomicNum()==6 and a.GetIsAromatic() and a.IsInRing()
                      for a in atoms if a.GetIdx()!=centre.GetIdx()))
    if phenyl_n:
        from .naming_methods import MethodDecision
        return MethodDecision('benzen'+'aminium'+'yl','characteristic-radical-ion',
            frozenset({centre.GetIdx()}),frozenset(a.GetIdx() for a in atoms),
            'P-75.3.1/P-62.6.1',((centre.GetIdx(),'N'),),
            component_roles=(('phenyl-aminium-radical',tuple(a.GetIdx() for a in atoms),
                              ((centre.GetIdx(),'N'),)),))
    endings={(7,1):('aminium',),(7,-1):('aminide','azanide'),
             (8,1):('oxidanium',),(16,1):('sulfanium',)}[(number,charge)]
    if not parent.name.endswith(endings):
        return None
    name=(parent.name[:-1] if parent.name.endswith('e') else parent.name)+'yl'
    from .naming_methods import MethodDecision
    locant={7:'N',8:'O',16:'S'}[number]
    issues=parent.unresolved
    if parent.preferred_status=='not-assessed':
        issues += (SelectionLimitation('characteristic-radical-ion','P-75.3',
            'The ionic precursor preferred-parent comparison is not assessed.'),)
    return MethodDecision(name,'characteristic-radical-ion',frozenset({centre.GetIdx()}),
        frozenset(a.GetIdx() for a in atoms),'P-75.3.1/P-75.3.2',
        ((centre.GetIdx(),locant),),
        component_roles=(('H-completed-ionic-parent',tuple(a.GetIdx() for a in atoms),
                          ((centre.GetIdx(),locant),)),),preference_issues=issues)
