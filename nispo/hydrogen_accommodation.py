"""P-58.2: joint suffix accommodation on noncumulative ring pi systems.

A matching represents pi bonds, not a tautomerization of the input. Removing
an odd alternating path accommodates either two suffix deficits (2.2.3) or a
suffix deficit and an added-H site (2.2.1). Remaining H additions are hydro
operations. Every solution retains the actual input's annular pi endpoints.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

from .molecule_cache import mol_cache_dict
from .molecule_iteration import iter_bonds


@dataclass(frozen=True, slots=True)
class HydrogenAccommodation:
    hydro: tuple
    added: tuple
    pairs: tuple
    paths: tuple
    hydro_paths: tuple
    reference_pi: tuple
    accommodated_pi: tuple


def peripheral_ring_atoms(edges):
    """Prove the boundary of a monocycle or an edge-fused bicyclic skeleton.

    Fusion atoms on this boundary are peripheral too. P-58.2.2.2 does not say
    *nonfusion* peripheral, unlike the indicated-H rule in P-58.2.1.2.
    """
    degree = Counter(i for edge in edges for i in edge)
    boundary = [edge for edge in edges if not all(degree[i] == 3 for i in edge)]
    counts = Counter(i for edge in boundary for i in edge)
    if set(counts) != set(degree) or any(n != 2 for n in counts.values()):
        return None
    seen, pending = set(), [min(degree)]
    while pending:
        i = pending.pop()
        if i in seen:
            continue
        seen.add(i)
        pending.extend(b if a == i else a for a, b in boundary if i in (a, b))
    return frozenset(seen) if seen == set(degree) else None


def _alternating_paths(first, second, endpoints):
    difference = first ^ second
    neighbors = {i: set() for pair in difference for i in pair}
    for a, b in difference:
        neighbors[a].add(b); neighbors[b].add(a)
    if {i for i, ns in neighbors.items() if len(ns) == 1} != set(endpoints):
        return None
    pending, paths = set(endpoints), []
    while pending:
        start = min(pending)
        previous, current, path = None, start, [start]
        while True:
            following = neighbors[current]-{previous}
            if not following:
                break
            if len(following) != 1:
                return None
            previous, current = current, next(iter(following))
            if current in path:
                return None
            path.append(current)
        if current == start or current not in pending or len(path) % 2:
            return None
        pending -= {start, current}
        paths.append(tuple(path))
    return tuple(sorted(paths))


def annular_pi_bonds(mol, Chem, sequence):
    """Read an actual Kekule matching without changing the original molecule."""
    cache = mol_cache_dict(mol, 'annular_kekule_bonds')
    key = id(Chem)
    if key not in cache:
        graph = Chem.Mol(mol)
        try:
            Chem.Kekulize(graph, clearAromaticFlags=True)
        except (ValueError, RuntimeError):
            cache[key] = None
        else:
            cache[key] = tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                                bond.GetBondTypeAsDouble())
                               for bond in iter_bonds(graph))
    if cache[key] is None:
        return None
    # Numbering alternatives project the same physical core repeatedly. Prove
    # its induced matching once from the existing private Kekule view, before
    # translating its endpoints to each caller's original numbering. In
    # particular, long external arms need not be rescanned per orientation.
    sequence = tuple(sequence)
    core = frozenset(sequence)
    cores = mol_cache_dict(mol, 'annular_kekule_core_matchings')
    core_key = (key, core)
    if core_key not in cores:
        pairs = []
        invalid = False
        for a, b, order in cache[key]:
            if a not in core or b not in core:
                continue
            if order not in {1, 2}:
                invalid = True
                break
            if order == 2:
                pairs.append((a, b))
        if invalid or any(count > 1 for count in
                          Counter(i for pair in pairs for i in pair).values()):
            cores[core_key] = None
        else:
            cores[core_key] = tuple(pairs)
    pairs = cores[core_key]
    if pairs is None:
        return None
    return _project_annular_pairs(sequence, pairs)


@lru_cache(maxsize=2048)
def _project_annular_pairs(sequence, pairs):
    """Translate an already-proved matching using immutable integer tuples."""
    inverse = {idx: j for j, idx in enumerate(sequence)}
    return tuple(sorted(tuple(sorted((inverse[a], inverse[b])))
                        for a, b in pairs))


@lru_cache(maxsize=1024)
def unique_capped_pi_graph(numbers, edges, hydrogens, *, allow_automorphisms=False):
    """Prove that one H population fixes one sanitized annular pi graph.

    This allows source-equivalent Kekule choices in productive fused naming.
    Distinct localized pi isomers with the same H vector remain outside this
    permission and need explicit localized-bond nomenclature.

    A caller may allow skeleton automorphisms only after proving the whole
    graph has no substituents, suffix sites, isotopes or stereochemistry.
    """
    from rdkit import Chem
    valence={6:4,7:3,8:2,16:2}
    degree=Counter(i for edge in edges for i in edge)
    occupancy=tuple(valence[n]-degree[i]-h for i,(n,h) in enumerate(zip(numbers,hydrogens)))
    if any(value not in {0,1} for value in occupancy):return False
    required=frozenset(i for i,value in enumerate(occupancy) if value)
    adjacency={i:{b if a==i else a for a,b in edges if i in (a,b)} for i in required}
    def matchings(remaining):
        if not remaining:
            yield frozenset();return
        a=min(remaining)
        for b in adjacency[a] & remaining:
            for pairs in matchings(remaining-{a,b}):
                yield pairs | {frozenset((a,b))}
    identities=set()
    for pairs in matchings(required):
        graph=Chem.RWMol()
        for number,count in zip(numbers,hydrogens):
            atom=Chem.Atom(number);atom.SetNumExplicitHs(count);atom.SetNoImplicit(True)
            # Numbering and suffix attachment refer to these particular
            # vertices. An automorphism moving them cannot prove equality of
            # two localized pi allocations on one mapped parent.
            if not allow_automorphisms:
                atom.SetAtomMapNum(graph.GetNumAtoms()+1)
            graph.AddAtom(atom)
        for a,b in edges:
            graph.AddBond(a,b,Chem.BondType.DOUBLE if frozenset((a,b)) in pairs else Chem.BondType.SINGLE)
        mol=graph.GetMol()
        try:Chem.SanitizeMol(mol)
        except (ValueError,RuntimeError):return False
        identities.add(Chem.MolToSmiles(mol,canonical=True))
        if len(identities)>1:return False
    return len(identities)==1


@lru_cache(maxsize=8192)
def joint_hydrogen_states(numbers, edges, reference_h, actual_h, demands, actual_pi):
    """Prove all maximally paired suffix operations for one reference H state.

    Atom indices refer to the caller's numbered skeleton. Demand is summed at
    each suffix site, so geminal single-valence suffixes consume two H there.
    The bounded callers supply at most ten annular atoms. No distance or name
    spelling enters this chemical proof or the subsequent numbering decision.
    """
    valence = {6: 4, 7: 3, 8: 2, 16: 2}
    degree = Counter(i for edge in edges for i in edge)
    delta = tuple(a-b for a, b in zip(actual_h, reference_h))
    if actual_pi is None or any(d not in {0, 1} for d in delta) or sum(delta) % 2:
        return ()
    matched = frozenset(i for i, n in enumerate(numbers)
                        if valence[n]-degree[i]-reference_h[i] == 1)
    if any(valence[n]-degree[i]-reference_h[i] not in {0, 1} for i, n in enumerate(numbers)):
        return ()
    changed = frozenset(i for i, d in enumerate(delta) if d)
    actual = frozenset(tuple(sorted(pair)) for pair in actual_pi)
    if frozenset(i for pair in actual for i in pair) != matched-changed:
        return ()
    required = frozenset(i for i, demand in enumerate(demands) if demand > reference_h[i])
    if any(demand > actual_h[i] or demand-reference_h[i] > 1 for i, demand in enumerate(demands)) or not required <= changed:
        return ()
    adjacency = {i: frozenset(b if a == i else a for a, b in edges if i in (a, b))
                 for i in matched}

    @lru_cache(None)
    def perfect(remaining):
        if not remaining:
            return (frozenset(),)
        a = min(remaining)
        return tuple(pairs | {tuple(sorted((a, b)))}
                     for b in sorted(adjacency[a] & remaining)
                     for pairs in perfect(remaining-{a, b}))

    donor_pool = tuple(sorted(changed-required))
    initial_states = perfect(matched)
    # Suffix accommodation precedes the additional hydro operations. Requiring
    # hydro first would incorrectly exclude the published 4a(2H)-carboxylic
    # acid example in P-58.2.2.2. Both stages may rearrange the pi bonds.
    # P-58.2.2.3 is an operation requirement, not a short-name preference.
    for pair_count in range(len(required)//2, -1, -1):
        solutions = set()
        for donors in combinations(donor_pool, len(required)-2*pair_count):
            donor_set = frozenset(donors)
            hydro = changed-required-donor_set
            for intermediate in perfect(matched-required-donor_set):
                hydro_paths = _alternating_paths(intermediate, actual, hydro)
                if hydro_paths is None:
                    continue
                for initial in initial_states:
                    paths = _alternating_paths(initial, intermediate, required | donor_set)
                    if paths is None or any(not ({p[0], p[-1]} & required) for p in paths):
                        continue
                    pairs = tuple(sorted(tuple(sorted((p[0], p[-1]))) for p in paths
                                         if {p[0], p[-1]} <= required))
                    solutions.add(HydrogenAccommodation(tuple(sorted(hydro)), donors,
                        pairs, paths, hydro_paths, tuple(sorted(initial)), tuple(sorted(intermediate))))
        if solutions:
            return tuple(sorted(solutions, key=lambda s: (s.hydro, s.added, s.pairs, s.paths,
                s.hydro_paths, s.reference_pi, s.accommodated_pi)))
    return ()
