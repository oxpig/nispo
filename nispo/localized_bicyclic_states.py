"""Explicit P-31 bond locants when an H state cannot identify a fused graph."""
from dataclasses import dataclass
from itertools import permutations

from .unsaturated_ring_parents import UnsaturationState
from .constants import MULT,ROOTS


def requires_explicit_aza_five_five_bonds(mol, core=None):
    """Reproduced boundary where fused shorthand loses localized pi state.

    Two edge-fused five-rings, one or two imine nitrogens and four distinct
    nonaromatic double bonds need the complete P-31 bond map. A topology/H
    match or a Delta token alone did not preserve these independent SM4
    controls, especially after a characteristic group distinguishes sites.
    """
    rings=tuple(frozenset(r) for r in mol.GetRingInfo().AtomRings()
                if core is None or set(r)<=core)
    if (len(rings)!=2 or any(len(r)!=5 for r in rings)
            or len(rings[0]&rings[1])!=2):return False
    atoms=rings[0]|rings[1]
    if core is not None and atoms!=core:return False
    nitrogen=[mol.GetAtomWithIdx(i) for i in atoms
              if mol.GetAtomWithIdx(i).GetAtomicNum()==7]
    if (len(nitrogen) not in {1,2}
            or any(a.GetTotalNumHs() or a.GetTotalValence()!=3 for a in nitrogen)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6,7}
                   or mol.GetAtomWithIdx(i).GetIsAromatic()
                   or mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in atoms)):
        return False
    bonds=[b for b in mol.GetBonds() if {b.GetBeginAtomIdx(),b.GetEndAtomIdx()}<=atoms]
    pi=[b for b in bonds if b.GetBondTypeAsDouble()==2]
    return (len(pi)==4 and all(b.GetBondTypeAsDouble() in {1,2} for b in bonds)
            and len({i for b in pi for i in (b.GetBeginAtomIdx(),b.GetEndAtomIdx())})==8)


def _numbered_saturated_frames(graph):
    """P-23 theta orientations including the proved zero-length fusion edge."""
    adjacency={a.GetIdx():{n.GetIdx() for n in a.GetNeighbors()} for a in graph.GetAtoms()}
    heads=[i for i,neighbors in adjacency.items() if len(neighbors)==3]
    if len(heads)!=2 or any(len(v) not in {2,3} for v in adjacency.values()):return
    paths=[];start,end=heads
    for neighbor in adjacency[start]:
        path=[start,neighbor]
        while path[-1]!=end:
            choices=adjacency[path[-1]]-{path[-2]}
            if len(choices)!=1:return
            following=next(iter(choices))
            if following in path:return
            path.append(following)
        paths.append(tuple(path))
    if len(paths)!=3 or set().union(*(set(path) for path in paths))!=set(adjacency):return
    rows=[];seen=set();elements=((8,'oxa'),(16,'thia'),(7,'aza'))
    for reverse in (False,True):
        oriented=[tuple(reversed(path)) if reverse else path for path in paths]
        for ordered in permutations(oriented):
            lengths=tuple(len(path)-2 for path in ordered)
            if lengths!=tuple(sorted(lengths,reverse=True)):continue
            a,b,c=ordered
            sequence=(*a,*tuple(reversed(b[1:-1])),*c[1:-1])
            if sequence in seen:continue
            seen.add(sequence)
            groups=tuple(tuple(j+1 for j,i in enumerate(sequence)
                if graph.GetAtomWithIdx(i).GetAtomicNum()==number) for number,_ in elements)
            key=(tuple(sorted(i for group in groups for i in group)),groups)
            parts=[','.join(map(str,locations))+'-'+(MULT[len(locations)] if len(locations)>1 else '')+term
                   for (_,term),locations in zip(elements,groups) if locations]
            parent='-'.join(parts)+'bicyclo['+'.'.join(map(str,lengths))+']'+ROOTS[len(sequence)]+'ane'
            inverse={i:j for j,i in enumerate(sequence)}
            edges=tuple((inverse[b.GetBeginAtomIdx()],inverse[b.GetEndAtomIdx()]) for b in graph.GetBonds())
            rows.append((parent,sequence,tuple(map(str,range(1,len(sequence)+1))),
                tuple(graph.GetAtomWithIdx(i).GetAtomicNum() for i in sequence),edges,True,key))
    if rows:
        best=min(row[6] for row in rows)
        yield from (row for row in rows if row[6]==best)


def _numbered_tricyclic_frames(graph):
    """Number the saturated carbon topology, then fix replacement locants."""
    from .tricyclic_von_baeyer_frontier import tricyclic_von_baeyer_frontier

    adjacency = {atom.GetIdx(): {n.GetIdx() for n in atom.GetNeighbors()}
                 for atom in graph.GetAtoms()}
    # The saturated C topology is ordinary at every vertex (degree <= 4).
    # Original neutral heteroatom valence was proved by the capped-core gate.
    if any(len(neighbors) > 4 for neighbors in adjacency.values()):
        return
    frontier = tricyclic_von_baeyer_frontier(adjacency, max_atoms=16)
    if frontier.status != 'complete':
        return
    elements = ((8, 'oxa'), (16, 'thia'), (7, 'aza'))
    rows = []
    for numbering in frontier.candidates:
        sequence = numbering.sequence
        groups = tuple(tuple(j + 1 for j, i in enumerate(sequence)
                             if graph.GetAtomWithIdx(i).GetAtomicNum() == number)
                       for number, _ in elements)
        key = (tuple(sorted(i for group in groups for i in group)), groups)
        parts = [','.join(map(str, locations)) + '-'
                 + (MULT[len(locations)] if len(locations) > 1 else '') + term
                 for (_, term), locations in zip(elements, groups) if locations]
        a, b, main, secondary = numbering.descriptor_lengths
        left, right = numbering.secondary_attachment_locants
        descriptor = f'{a}.{b}.{main}.{secondary}^{left},{right}'
        parent = '-'.join(parts) + f'tricyclo[{descriptor}]' + ROOTS[len(sequence)] + 'ane'
        inverse = {i: j for j, i in enumerate(sequence)}
        edges = tuple((inverse[edge.GetBeginAtomIdx()], inverse[edge.GetEndAtomIdx()])
                      for edge in graph.GetBonds())
        rows.append((parent, sequence, tuple(map(str, range(1, len(sequence) + 1))),
                     tuple(graph.GetAtomWithIdx(i).GetAtomicNum() for i in sequence),
                     edges, True, key))
    if rows:
        best = min(row[6] for row in rows)
        yield from (row for row in rows if row[6] == best)


def _equivalent_pi_matchings(graph, Chem):
    """Enumerate every matching with the same atom-mapped aromatic graph.

    Explicit ene locants cannot depend on RDKit's first Kekule arrangement.
    Keeping atom maps during the identity comparison permits resonance of an
    aromatic face while rejecting relocation of a localized multiple bond.
    """
    valence = {6: 4, 7: 3, 8: 2, 16: 2}
    hydrogen = tuple(atom.GetTotalNumHs() for atom in graph.GetAtoms())
    required = frozenset(atom.GetIdx() for atom in graph.GetAtoms()
                         if valence[atom.GetAtomicNum()] - atom.GetDegree()
                         - hydrogen[atom.GetIdx()] == 1)
    if any(valence[atom.GetAtomicNum()] - atom.GetDegree() - hydrogen[atom.GetIdx()]
           not in {0, 1} for atom in graph.GetAtoms()):
        return ()
    adjacency = {i: {atom.GetIdx() for atom in graph.GetAtomWithIdx(i).GetNeighbors()}
                 for i in required}
    source = Chem.Mol(graph)
    for atom in source.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    identity = Chem.MolToSmiles(source, canonical=True)

    def matchings(remaining):
        if not remaining:
            yield ()
            return
        a = min(remaining)
        for b in sorted(adjacency[a] & remaining):
            for pairs in matchings(remaining - {a, b}):
                yield ((min(a, b), max(a, b)), *pairs)

    equivalent = []
    for pairs in matchings(required):
        work = Chem.RWMol(source)
        for atom in work.GetAtoms():
            atom.SetIsAromatic(False)
            atom.SetNumExplicitHs(hydrogen[atom.GetIdx()])
            atom.SetNoImplicit(True)
        for bond in work.GetBonds():
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)
        for a, b in pairs:
            work.GetBondBetweenAtoms(a, b).SetBondType(Chem.BondType.DOUBLE)
        candidate = work.GetMol()
        try:
            Chem.SanitizeMol(candidate)
        except (ValueError, RuntimeError):
            continue
        if Chem.MolToSmiles(candidate, canonical=True) == identity:
            equivalent.append(tuple(sorted(pairs)))
    return tuple(equivalent)


@dataclass(frozen=True, slots=True)
class LocalizedBicyclicState:
    hetero_locants: tuple
    reference_hydrogens: tuple
    unsaturation: UnsaturationState
    von_baeyer: bool = True


def skeletons(mol, Chem):
    """Retain the localized two-face admission used before the fallback."""
    yield from _skeletons(mol, Chem, fallback=False)


def fallback_skeletons(mol, Chem):
    """Use explicit bond locants when no retained/fusion writer is available.

    This includes small edge-fused rings such as benzocyclobutane. Their
    aromatic face is represented by one source-equivalent Kekule matching;
    the complete P-31 locants retain every annular double bond. This is an
    alternative construction, with preferred fusion nomenclature pending.
    """
    yield from _skeletons(mol, Chem, fallback=True)


def _skeletons(mol, Chem, *, fallback):
    from .mapped_ring_parents import _core_graph
    from .hydrogen_accommodation import unique_capped_pi_graph,annular_pi_bonds

    rings=tuple(map(frozenset,mol.GetRingInfo().AtomRings()))
    if (len(rings) not in ({2, 3} if fallback else {2})
            or not fallback and any(len(ring) not in {5,6} for ring in rings)):return
    core=frozenset().union(*rings)
    if len(core)>16 or len(core) not in ROOTS:return
    prepared=_core_graph(mol,Chem,core)
    if prepared is None:return
    graph,original=prepared
    if not fallback and any(b.GetIsAromatic() for b in graph.GetBonds()):return
    numbers=tuple(a.GetAtomicNum() for a in graph.GetAtoms())
    edges=tuple((b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in graph.GetBonds())
    hydrogen=tuple(a.GetTotalNumHs() for a in graph.GetAtoms())
    actual_pi=annular_pi_bonds(graph,Chem,tuple(range(len(numbers))))
    if (actual_pi is None or not actual_pi
            or not fallback and unique_capped_pi_graph(numbers,edges,hydrogen)):
        return
    # Give every original multiple bond its explicit P-31.1.4 locants on a
    # numbered replacement hydride. These alternative constructions retain
    # the pending preferred fusion-nomenclature obligation.
    saturated=Chem.RWMol()
    for number in numbers:saturated.AddAtom(Chem.Atom(number))
    for a,b in edges:saturated.AddBond(a,b,Chem.BondType.SINGLE)
    parent=saturated.GetMol();Chem.SanitizeMol(parent)
    rank = graph.GetNumBonds() - graph.GetNumAtoms() + 1
    frames = (_numbered_saturated_frames(parent) if rank == 2 else
              _numbered_tricyclic_frames(parent) if rank == 3 else ())
    alternatives = _equivalent_pi_matchings(graph, Chem) if fallback else (actual_pi,)
    for row in frames:
        sequence=row[1];inverse={idx:j+1 for j,idx in enumerate(sequence)}
        for pairs in alternatives:
            multiple=tuple(sorted((min(inverse[a],inverse[b]),max(inverse[a],inverse[b]),2)
                for a,b in pairs))
            if not multiple:continue
            locations=tuple(a for a,b,degree in multiple)
            state=LocalizedBicyclicState(tuple(j+1 for j,i in enumerate(sequence) if numbers[i]!=6),
                tuple(hydrogen[i] for i in sequence),UnsaturationState(multiple,locations,locations,()))
            method = ('skeletal-replacement-explicit-bicyclic' if fallback and rank == 2 else
                      'skeletal-replacement-localized-bicyclic' if rank == 2 else
                      'skeletal-replacement-localized-tricyclic')
            yield (row[0],tuple(original[i] for i in sequence),*row[2:5],False,row[6],method,state)
