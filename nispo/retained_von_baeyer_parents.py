"""Complete original-frame mappings for the two P-23.7 retained carbon cages.

The source-numbered parent graphs supply vocabulary and every automorphism.
They never encode a complete derivative or choose its suffix/prefix locants.
"""
from functools import lru_cache


# P-23.7/Table 2.6, with the corresponding von Baeyer numbering.
_FRAMES=(
    ('adamantane',10,((1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,1),
                      (1,9),(9,5),(3,10),(10,7))),
    ('cubane',8,((1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,1),
                (1,6),(2,5),(3,8),(4,7))),
)
_MATCH_BOUND=4097


@lru_cache(maxsize=2)
def _frame_binary(Chem,name):
    _name,size,edges=next(row for row in _FRAMES if row[0]==name)
    graph=Chem.RWMol()
    for _ in range(size):graph.AddAtom(Chem.Atom(6))
    for a,b in edges:graph.AddBond(a-1,b-1,Chem.BondType.SINGLE)
    mol=graph.GetMol();Chem.SanitizeMol(mol)
    return mol.ToBinary(Chem.PropertyPickleOptions.AllProps)


def skeletons(graph,Chem):
    size,bonds=graph.GetNumAtoms(),graph.GetNumBonds()
    eligible=tuple(row for row in _FRAMES if row[1]==size and len(row[2])==bonds)
    if not eligible:return
    numbers = tuple(a.GetAtomicNum() for a in graph.GetAtoms())
    replaced = [(i, n) for i, n in enumerate(numbers) if n != 6]
    # P-23.3 permits skeletal replacement of the retained cage. The bounded
    # route admits one ordinary divalent O/S or trivalent N replacement.
    if (len(replaced) > 1 or any(n not in {7, 8, 16} for _, n in replaced)
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
            for a in graph.GetAtoms())
            or any(b.GetBondType()!=Chem.BondType.SINGLE for b in graph.GetBonds())):
        return
    for name,_size,edges in eligible:
        if replaced and name != 'adamantane':
            continue
        frame=Chem.Mol(_frame_binary(Chem,name))
        target = graph
        if replaced:
            target = Chem.Mol(graph)
            target.GetAtomWithIdx(replaced[0][0]).SetAtomicNum(6)
        matches=target.GetSubstructMatches(frame,uniquify=False,maxMatches=_MATCH_BOUND)
        if len(matches)==_MATCH_BOUND:return
        if replaced and matches:
            idx, element = replaced[0]
            loc = min(match.index(idx) for match in matches)
            matches = tuple(match for match in matches if match[loc] == idx)
            name = f"{loc + 1}-" + {7: 'aza', 8: 'oxa', 16: 'thia'}[element] + name
        for sequence in matches:
            if len(set(sequence))!=size:continue
            yield (name,tuple(sequence),tuple(map(str,range(1,size+1))),
                   tuple(numbers[i] for i in sequence),tuple((a-1,b-1) for a,b in edges),True,(),
                   'retained-von-Baeyer')


def parent_decision(mol,Chem):
    """P-23.7 permission for a complete unannotated retained hydride."""
    if (mol is None or mol.GetNumAtoms() not in {8,10}
            or mol.GetNumAtoms()!=mol.GetNumHeavyAtoms() or mol.GetStereoGroups()
            or any(a.GetAtomicNum() != 6 or a.GetIsotope() or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    rows=tuple(skeletons(mol,Chem))
    parents={row[0] for row in rows}
    if len(parents)!=1:return None
    from .naming_methods import MethodDecision
    atoms=frozenset(range(mol.GetNumAtoms()))
    return MethodDecision(next(iter(parents)),'retained-hydrocarbon-parent',frozenset(),atoms,
        'P-23.7/Table-2.6',component_roles=(('complete-source-retained-hydride',atoms,()),))
