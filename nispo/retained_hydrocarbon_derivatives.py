"""Complete Type 1 retained carbon hydrides with proved ordinary pendant roles.

The whole original ring graph fixes the hydride and its indicated-H state.
Every exterior component has an explicit chemical role. Ordinary single-bond
substitution and a ketone replacing the two hydrogens at a source indicated-H
methylene preserve every internal bond. Other carbonyl changes, annotations
and competing pendant parents retain the other naming routes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True,slots=True)
class _Pendant:
    root: int
    host: int
    atoms: frozenset[int]
    kind: str
    prefix: str


_RANK={'acid':(7,0),'amide':(11,0),'nitrile':(14,0),'aldehyde':(15,0),'ketone':(16,0),
       'alcohol':(17,0),'thiol':(17,1),'amine':(19,0)}
_SUFFIX={'acid':'carboxylic acid','amide':'carboxamide','nitrile':'carbonitrile',
         'aldehyde':'carbaldehyde','ketone':'one','alcohol':'ol','thiol':'thiol','amine':'amine'}


@lru_cache(maxsize=19)
def _frame(text):
    from rdkit import Chem
    return Chem.MolFromSmiles(text)


def _numbering_key(mapping,suffix_parts,prefix_parts):
    from .formatting import prefix_alphabetization_key
    from .parent_selection import locant_set
    locs=sorted((mapping[p.host] for p in suffix_parts),key=lambda value:locant_set((value,)))
    prefixes=[(mapping[p.host],p.prefix) for p in prefix_parts]
    grouped={name:[loc for loc,text in prefixes if text==name] for _loc,name in prefixes}
    citation=tuple(locant_set(grouped[name]) for name in sorted(grouped,key=prefix_alphabetization_key))
    return (locant_set(locs),locant_set([loc for loc,_text in prefixes]),citation),locs,prefixes


def _topology_numbering_frontier(Chem,induced,frame,core,source_locs,suffix_parts,prefix_parts):
    """Detect a lower unproved numbering without erasing localized pi bonds.

    An unlabelled skeleton is a comparison witness only. It never licenses
    a name or a bond rearrangement, including at a partially aromatic fusion.
    The capped hydride H populations must still match at every vertex.
    """
    left,right=Chem.Mol(induced),Chem.Mol(frame)
    for graph in (left,right):
        for atom in graph.GetAtoms():atom.SetIsAromatic(False)
        for bond in graph.GetBonds():bond.SetIsAromatic(False);bond.SetBondType(Chem.BondType.SINGLE)
    matches=left.GetSubstructMatches(right,uniquify=False,maxMatches=4096)
    if len(matches)==4096:return None
    keys=[]
    for match in matches:
        if any(induced.GetAtomWithIdx(target).GetTotalNumHs()!=frame.GetAtomWithIdx(source).GetTotalNumHs()
               for source,target in enumerate(match)):continue
        mapping={core[target]:source_locs[source] for source,target in enumerate(match)}
        keys.append(_numbering_key(mapping,suffix_parts,prefix_parts)[0])
    return min(keys) if keys else None


def _pendant(mol,Chem,root,host,atoms):
    atom=mol.GetAtomWithIdx(root)
    if len(atoms)==1:
        if (atom.GetAtomicNum()==8 and atom.GetDegree()==1 and atom.GetTotalNumHs()==0
                and mol.GetBondBetweenAtoms(root,host).GetBondType()==Chem.BondType.DOUBLE):
            return _Pendant(root,host,atoms,'ketone','oxo')
        simple={(8,1):('alcohol','hydroxy'),(16,1):('thiol','sulfanyl'),(7,2):('amine','amino'),
                (9,0):('','fluoro'),(17,0):('','chloro'),(35,0):('','bromo'),(53,0):('','iodo')}
        role=simple.get((atom.GetAtomicNum(),atom.GetTotalNumHs()))
        if atom.GetDegree()==1 and role:return _Pendant(root,host,atoms,*role)
        if atom.GetAtomicNum()!=6:return None
    if atom.GetAtomicNum()==6:
        neighbours=[a for a in atom.GetNeighbors() if a.GetIdx()!=host]
        singles=[a for a in neighbours if mol.GetBondBetweenAtoms(root,a.GetIdx()).GetBondType()==Chem.BondType.SINGLE]
        double_o=[a for a in neighbours if a.GetAtomicNum()==8 and a.GetDegree()==1
                  and a.GetTotalNumHs()==0 and mol.GetBondBetweenAtoms(root,a.GetIdx()).GetBondType()==Chem.BondType.DOUBLE]
        if len(atoms)==2 and len(neighbours)==1:
            other=neighbours[0];bond=mol.GetBondBetweenAtoms(root,other.GetIdx())
            if other.GetAtomicNum()==7 and other.GetDegree()==1 and bond.GetBondType()==Chem.BondType.TRIPLE and atom.GetTotalNumHs()==other.GetTotalNumHs()==0:
                return _Pendant(root,host,atoms,'nitrile','cyano')
            if len(double_o)==1 and atom.GetTotalNumHs()==1:
                return _Pendant(root,host,atoms,'aldehyde','formyl')
        if len(atoms)==3 and len(neighbours)==2 and len(double_o)==len(singles)==1 and atom.GetTotalNumHs()==0:
            other=singles[0]
            if other.GetDegree()==1:
                if other.GetAtomicNum()==8 and other.GetTotalNumHs()==1:
                    return _Pendant(root,host,atoms,'acid','carboxy')
                if other.GetAtomicNum()==7 and other.GetTotalNumHs()==2:
                    return _Pendant(root,host,atoms,'amide','carbamoyl')
    # Existing mapped P-29 construction proves the complete saturated carbon
    # tree or O-linked alkyl tree. No characteristic group can hide inside it.
    alkoxy=(atom.GetAtomicNum()==8 and atom.GetDegree()==2 and atom.GetTotalNumHs()==0)
    if (atom.GetAtomicNum()==6 or alkoxy) and all(
            mol.GetAtomWithIdx(i).GetAtomicNum()==6 or alkoxy and i==root for i in atoms):
        if any(b.GetBondType()!=Chem.BondType.SINGLE for b in mol.GetBonds()
               if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms):return None
        from .component_mapping import mapped_branch
        text,owned=mapped_branch(mol,Chem,root,host)
        if text and owned==atoms:return _Pendant(root,host,atoms,'',text)
    return None


def _aryl_substituted_retained_frame(mol, Chem, core):
    """Render a retained three-face carbon parent with mapped aryl prefixes."""
    from types import SimpleNamespace
    from .component_mapping import fully_covered
    from .extended_inventory import selection_inventory
    from .mapped_ring_parents import skeletons
    from .naming_context import MoleculeContext
    from .naming_methods import MethodDecision
    from .naming_result import SelectionLimitation
    from .ring_state_parent import _render, _skeleton_references

    source = selection_inventory(mol)
    if not source.supported or source.principal:
        return None
    scope = frozenset(range(mol.GetNumAtoms()))
    inventory = SimpleNamespace(groups=(), principal=(), general_branches=True)
    ctx = MoleculeContext(mol, Chem)
    rows = []
    for skeleton in skeletons(mol, Chem, core):
        if skeleton[0] not in {'phenanthrene', 'fluorene'}:
            continue
        for reference in _skeleton_references(skeleton):
            rows.extend(_render(ctx, skeleton, reference, inventory, scope=scope))
    if not rows:
        return None
    key = min(row.numbering_key for row in rows)
    best = [row for row in rows if row.numbering_key == key]
    if len({row.name for row in best}) != 1:
        return None
    row = best[0]
    if not fully_covered(mol, row.construction.atoms, row.construction.embedded_atoms):
        return None
    rule = 'P-25.1.1/P-25.3.3/P-31.2.3/P-44'
    return MethodDecision(row.name, 'retained-hydrocarbon-derivative', frozenset(), scope,
        rule, row.atom_locants, preference_issues=(SelectionLimitation(
            'retained-hydrocarbon-parent-comparison', rule,
            'The retained framework and hydrogenation are mapped. Full parent comparison remains open.'),))


def retained_hydrocarbon_derivative_decision(mol,Chem):
    if (mol is None or not 10<=mol.GetNumAtoms()<=96 or mol.GetNumAtoms()!=mol.GetNumHeavyAtoms()
            or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol))!=1):return None
    core=tuple(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing());core_set=frozenset(core)
    if not 9<=len(core)<=32 or len(core)==mol.GetNumAtoms():return None
    if any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6 for i in core):return None
    if any(a.GetAtomicNum() not in {6,7,8,9,16,17,35,53} or a.GetFormalCharge()
           or a.GetNumRadicalElectrons() or a.GetIsotope() or int(a.GetChiralTag()) for a in mol.GetAtoms()):return None
    if any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds()):return None
    if len(core) >= 19 and mol.GetRingInfo().NumRings() >= 4:
        from .mapped_ring_parents import ring_systems

        systems = sorted(ring_systems(mol), key=len, reverse=True)
        if len(systems) > 1 and len(systems[0]) in {13, 14} and all(
                len(other) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in other)
                for other in systems[1:]):
            return _aryl_substituted_retained_frame(mol, Chem, systems[0])
    # Reconstruct only the induced hydride. Bond orders/arenes survive; each
    # original single substitutive port is later proved separately.
    induced=Chem.RWMol(mol)
    for i in reversed(range(mol.GetNumAtoms())):
        if i not in core_set:induced.RemoveAtom(i)
    induced=induced.GetMol()
    for atom in induced.GetAtoms():
        atom.SetAtomMapNum(0);atom.SetNumExplicitHs(0);atom.SetNoImplicit(False)
    try:Chem.SanitizeMol(induced)
    except (ValueError,RuntimeError):return None
    from .retained_hydrocarbon_parents import _vocabulary
    record=_vocabulary().get(Chem.MolToSmiles(induced,canonical=True,isomericSmiles=True))
    if record is None:return None
    unseen=set(range(mol.GetNumAtoms()))-core_set;parts=[]
    while unseen:
        pending=[min(unseen)];atoms=set()
        while pending:
            i=pending.pop()
            if i in atoms:continue
            atoms.add(i)
            pending.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                           if n.GetIdx() not in core_set and n.GetIdx() not in atoms)
        unseen.difference_update(atoms);ports=[(i,n.GetIdx()) for i in atoms
            for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in core_set]
        if len(ports)!=1:return None
        root,host=ports[0]
        bond=mol.GetBondBetweenAtoms(root,host)
        if bond.GetBondType()!=Chem.BondType.SINGLE:
            if (bond.GetBondType()!=Chem.BondType.DOUBLE or 'H-' not in record['name']
                    or len(atoms)!=1 or mol.GetAtomWithIdx(root).GetAtomicNum()!=8):return None
        part=_pendant(mol,Chem,root,host,frozenset(atoms))
        if part is None:return None
        parts.append(part)
    principal=min((_RANK[p.kind] for p in parts if p.kind),default=None)
    suffix_parts=[p for p in parts if p.kind and _RANK[p.kind]==principal]
    prefix_parts=[p for p in parts if p not in suffix_parts]
    from .formatting import format_prefixes,multiplied_suffix,attach_suffix
    frame=_frame(record['original_frame_smiles']);source_locs={int(i):loc for i,loc in record['original_locants'].items()}
    matches=induced.GetSubstructMatches(frame,uniquify=False,maxMatches=4096)
    if not matches or len(matches)==4096:return None
    candidates=[]
    for match in matches:
        mapping={core[target]:source_locs[source] for source,target in enumerate(match)}
        if any(p.kind=='ketone' and (mapping[p.host]!=record['name'].partition('H-')[0]
                or frame.GetAtomWithIdx(next(i for i,target in enumerate(match) if core[target]==p.host)).GetTotalNumHs()!=2)
               for p in parts):continue
        key,locs,prefixes=_numbering_key(mapping,suffix_parts,prefix_parts)
        parent=record['name']
        if suffix_parts:
            suffix=multiplied_suffix(len(suffix_parts),_SUFFIX[suffix_parts[0].kind])
            if not suffix:return None
            parent=attach_suffix(parent,suffix,locs)
        prefix=format_prefixes(prefixes,hierarchical_enclosures=True)
        # Numerical indicated-H and italic s/as descriptors retain a hyphen
        # at the actual detachable-prefix boundary (P-16).
        name=prefix+('-' if 'H-' in record['name'] or record['name'].startswith(('s-','as-')) else '')+parent if prefix else parent
        candidates.append((key,name,mapping))
    if not candidates:return None
    best=min(row[0] for row in candidates);frontier=[row for row in candidates if row[0]==best]
    if len({row[1] for row in frontier})!=1:return None
    _key,name,mapping=min(frontier,key=lambda row:tuple(sorted(row[2].items())))
    from .naming_methods import MethodDecision
    owned=frozenset(range(mol.GetNumAtoms()));roles=(('retained-carbon-hydride',core_set,tuple(sorted(mapping.items()))),)
    roles+=tuple((('suffix-' if p in suffix_parts else 'prefix-')+(p.kind or 'saturated-tree'),p.atoms,((p.root,p.host),)) for p in parts)
    issues=()
    if any(not bond.GetIsAromatic() for bond in frame.GetBonds()):
        witness=_topology_numbering_frontier(Chem,induced,frame,core,source_locs,suffix_parts,prefix_parts)
        if witness is None or witness<best:
            from .naming_result import SelectionLimitation
            issues=(SelectionLimitation('retained-hydrocarbon-numbering','P-25.7.1.2/P-44/P-58.2',
                'Localized pi bonds exclude a lower topology/H-compatible numbering; the complete comparison and any required Delta descriptor remain unproved.'),)
    return MethodDecision(name,'retained-hydrocarbon-derivative',frozenset(p.host if p.kind=='ketone' else p.root for p in suffix_parts),owned,
        'P-15.1.8.1/P-25.1.1/P-41/P-44/P-58.2',tuple(sorted(mapping.items())),component_roles=roles,preference_issues=issues)
