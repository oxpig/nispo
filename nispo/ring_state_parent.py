"""Mapped heteromonocyclic and retained fused hydrides (P-22/P-54/P-58).

The input is never tautomerized. Maximum noncumulative pi-bond matchings of a
reference skeleton establish its H population. External bonds are capped only
for that comparison; every actual substituent retains original atom ownership.
The numbering key follows explicit rules, with no length or scaffold bonus.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .molecule_iteration import iter_atoms, iter_bonds
from .molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set, naming_cache_state
from .component_mapping import MappedComponent, decorate_component, fully_covered, hydrogen_bindings, mapped_branch
from .constants import MULT
from .formatting import attach_suffix, multiplied_suffix, prefix_alphabetization_key
from .hydrogen_accommodation import annular_pi_bonds, joint_hydrogen_states, peripheral_ring_atoms
from .parent_selection import locant_key

_ELEMENTS = (8, 16, 34, 52, 7)  # P-22.2.2 Table 2.4 citation order.
_VALENCE = {6: 4, 7: 3, 8: 2, 16: 2}
# Hantzsch–Widman-only data. Do not widen _VALENCE: retained fused and other
# ring routes have separate evidence and must not inherit Se/Te admission.
_HW_VALENCE = {**_VALENCE, 34: 2, 52: 2}
_HW_DESCRIPTORS = {8: 'oxa', 16: 'thia', 34: 'selena', 52: 'tellura', 7: 'aza', 15: 'phospha'}
from .preferred_fused import RETAINED_FUSED_HYDRIDES as _FUSED_PARENTS
_CARBONYLS = {'ketone', 'lactam', 'lactone', 'pseudoketone'}
_EXTERNAL_HETEROATOM_ENDINGS = {
    'sulfonic-acid': 'sulfonic acid', 'sulfinic-acid': 'sulfinic acid',
    'sulfonamide': 'sulfonamide', 'phosphonic-acid': 'phosphonic acid',
}
_EXTERNAL_HETEROATOM_GROUPS = _EXTERNAL_HETEROATOM_ENDINGS.keys() | {'thiol'}
_SUPPORTED_GROUPS = _CARBONYLS | {'acid', 'carboxylate', 'carbamate-anion', 'amide', 'nitrile', 'aldehyde', 'alcohol', 'amine', 'imine', 'peroxol',
                                'oxime', 'oxime-ether', 'hydrazide'} | _EXTERNAL_HETEROATOM_GROUPS
_CRITERIA = ('P-22.2.2.1/P-25-fixed-numbering', 'P-58.2.3-suffix-accommodation',
    'P-58.2.1-peripheral-indicated-H', 'P-14.4(b)-indicated-H', 'P-14.4(c)-suffix',
    'P-58.2.2.3-paired-suffixes', 'P-58.2.2-peripheral-added-H', 'P-14.4(d)-added-H', 'P-14.4(e)/P-31.1.1.1-hydro-or-multiple-bond-locants', 'P-14.4(f)-prefix-set',
    'P-14.4(g)-prefix-citation', 'P-14.4(i)/P-82.5-isotopes', 'P-14.4(j)-stereo')


@dataclass(frozen=True, slots=True)
class RingStateRecord:
    name: str
    parent: str
    method: str
    parent_atoms: tuple
    atom_locants: tuple
    principal_centers: frozenset
    indicated_h_locants: tuple
    added_h_locants: tuple
    hydro_locants: tuple
    numbering_key: tuple
    reference_hydrogens: tuple
    capped_hydrogens: tuple
    paired_suffix_locants: tuple = ()
    hydrogen_paths: tuple = ()
    hydro_paths: tuple = ()
    reference_pi_bonds: tuple = ()
    accommodated_pi_bonds: tuple = ()
    prefixes: tuple = ()
    suffix_locants: tuple = ()
    hydrogen_budget_proved: bool = True
    annotation_coverage: bool = True
    construction: MappedComponent | None = None
    unsaturation_state: object | None = None
    fusion_construction: object | None = None
    preference_issues: tuple = ()
    hydride: str = ''


@dataclass(frozen=True, slots=True)
class RingStateSelection:
    selected: RingStateRecord
    candidates: tuple
    criteria: tuple = _CRITERIA


class _RingNumberingFrontier:
    """Call-local bounds from successfully constructed fixed-parent records.

    A failed rendering establishes no bound. Vocabulary is part of the key:
    success for one parent must never hide a second parent vocabulary. Strict
    prefix comparison keeps every chemical tie, including isotope/stereo ties.
    """

    def __init__(self, *, defer=True):
        self.keys = {}
        self.plans = []
        self.defer = defer

    @classmethod
    def for_context(cls, ctx):
        if (ctx.telemetry is not None
                or mol_cache_get(ctx.mol, 'candidate_telemetry') is not None
                or naming_cache_state(ctx.mol)):
            return None
        annotated = (any(int(atom.GetChiralTag()) for atom in iter_atoms(ctx.mol))
                     or any(int(bond.GetStereo()) or int(bond.GetBondDir())
                            for bond in iter_bonds(ctx.mol)))
        # Stereo-sensitive construction keeps the original observation order.
        # Each plan realizes its branches before a numbering bound may omit
        # decoration. The ordinary path can still schedule whole plans by key.
        return cls(defer=not annotated)

    def render_or_defer(self, plan):
        if self.defer:
            self.plans.append(plan)
            return ()
        return plan.render(self)

    def dominated(self, parent, prefix):
        previous = self.keys.get(parent)
        return previous is not None and prefix > previous[:len(prefix)]

    def accept(self, record):
        previous = self.keys.get(record.parent)
        if previous is None or record.numbering_key < previous:
            self.keys[record.parent] = record.numbering_key


    def finish(self):
        # Compare within each vocabulary, retaining its existence even when
        # another vocabulary has a better key. Stable original ordinals keep
        # the historical first witness when complete keys and names tie.
        vocabularies = {}
        for ordinal, plan in enumerate(self.plans):
            vocabularies.setdefault(plan.rendering.skeleton[0], []).append((ordinal, plan))
        from .ring_symmetry import RingRenderingSymmetry
        symmetry = RingRenderingSymmetry()
        rendered = []
        for parent, plans in vocabularies.items():
            # Graph proof and witness projection have a fixed cost. Single
            # plans and pairs of orientations are cheaper to render directly.
            # Counting ties affects scheduling only; no plan is discarded.
            ties = Counter(plan.chemical_key for _, plan in plans) if len(plans) >= 3 else {}
            for ordinal, plan in sorted(plans, key=lambda item: item[1].chemical_key):
                if not self.dominated(parent, plan.chemical_key):
                    share = ties.get(plan.chemical_key, 0) >= 3
                    rows = symmetry.reuse(plan) if share else None
                    if rows is None:
                        rows = plan.render(self)
                        if share:
                            symmetry.remember(plan, rows)
                    else:
                        # A later plan can share an earlier construction yet
                        # lose to a prefix bound established in between. This
                        # is the same second-stage check as _finish_ring_state.
                        rows = tuple(record for record in rows
                                     if not self.dominated(parent, record.numbering_key[:-2]))
                        for record in rows:
                            self.accept(record)
                    rendered.extend((ordinal, record) for record in rows)
        self.plans.clear()  # Suspended constructions must never escape this call.
        return [record for _, record in sorted(rendered, key=lambda item: item[0])]


_NO_RING_BRANCHES = object()


@dataclass(slots=True)
class _RingRendering:
    ctx: object
    skeleton: tuple
    core: frozenset
    mapping: dict
    groups: tuple
    kind: str
    prepared: object
    scope: object
    nmap: dict
    needed: frozenset
    h: tuple
    actual: tuple
    branches: object = _NO_RING_BRANCHES
    prefix_terms: object = None


@dataclass(frozen=True, slots=True)
class _RingStatePlan:
    rendering: _RingRendering
    state: object
    added: tuple
    hydro: tuple
    leading: tuple
    suffix_locs: tuple
    chemical_key: tuple

    def render(self, frontier=None):
        return _finish_ring_state(self, frontier)


@lru_cache(maxsize=1024)
def _ring_numbering_terms(locs, edges, perimeter, fixed_key, indicated,
                          site_locs, states, absolute_edges, von_baeyer):
    """Reuse only the exact primitive locant form of proved H states.

    States and their witnesses have already been proved on the original
    numbered graph. Every returned row still gets its own rendering and
    record; no parent vocabulary, prefix, annotation or name is cached here.
    """
    leading = tuple(sorted((locs[j] for j in indicated), key=locant_key))
    suffix_locs = tuple(sorted(site_locs, key=locant_key))
    degrees = _ring_degrees(len(locs), edges)
    peripheral = sum(degrees[j] != 2 for j in indicated)
    accommodation = (-sum(loc in leading for loc in site_locs)
                     if site_locs and len(leading) >= len(site_locs) else 0)
    leading_key = tuple(map(locant_key, leading))
    suffix_key = tuple(map(locant_key, suffix_locs))
    if absolute_edges is not None:
        def bond_locant(a, b):
            return (a,) if b - a == 1 or von_baeyer else (a, b)
        if von_baeyer and all(order == 2 for _, _, order in absolute_edges):
            # P-31.1.4.2 compares the count of compound bond locants before
            # the low endpoints, then includes compound high endpoints only
            # when the earlier criteria tie. The edges are projected from
            # the enumerated source-equivalent pi matchings, not name text.
            compound_count = sum(b - a != 1 for a, b, _ in absolute_edges)
            low = tuple(sorted((a,) for a, _, _ in absolute_edges))
            all_endpoints = tuple(sorted((loc,) for a, b, _ in absolute_edges
                                         for loc in ((a,) if b - a == 1 else (a, b))))
            absolute_key = (((compound_count,),), low, all_endpoints)
        else:
            absolute_key = (
                tuple(sorted(bond_locant(a, b) for a, b, order in absolute_edges)),
                tuple(sorted(bond_locant(a, b) for a, b, order in absolute_edges if order == 2)),
            )
    rows = []
    for state in states:
        added = tuple(sorted((locs[j] for j in state.added), key=locant_key))
        hydro = tuple(sorted((locs[j] for j in state.hydro), key=locant_key))
        if hydro and (len(hydro) % 2 or len(hydro) not in MULT):
            continue
        added_peripheral = sum(j not in perimeter for j in state.added)
        degree_key = absolute_key if absolute_edges is not None else tuple(map(locant_key, hydro))
        chemical_key = (fixed_key, accommodation, peripheral, leading_key,
            suffix_key, -len(state.pairs), added_peripheral,
            tuple(map(locant_key, added)), degree_key)
        rows.append((state, added, hydro, leading, suffix_locs, chemical_key))
    return tuple(rows)


def _hw_name(numbers, saturated):
    """P-22.2 Tables 2.2–2.5: vocabulary and productive HW morphology."""
    size, counts = len(numbers), Counter(numbers)
    # The nonstandard-valence producer also uses this vocabulary for P.
    # Do not silently drop an element that the ordinary ring-state search
    # does not admit; vocabulary support does not widen _HW_VALENCE.
    if any(n != 6 and n not in _HW_DESCRIPTORS for n in numbers):
        return ''
    locations = {e: tuple(i+1 for i, n in enumerate(numbers) if n == e) for e in _HW_DESCRIPTORS}
    signature = tuple((e, locations[e]) for e in _HW_DESCRIPTORS if locations[e])
    retained = {
        (5, ((7, (1,)),)): ('pyrrole', 'pyrrolidine'),
        (5, ((8, (1,)),)): ('furan', 'oxolane'),
        (5, ((16, (1,)),)): ('thiophene', 'thiolane'),
        (5, ((7, (1, 2)),)): ('pyrazole', 'pyrazolidine'),
        (5, ((7, (1, 3)),)): ('imidazole', 'imidazolidine'),
        (6, ((7, (1,)),)): ('pyridine', 'piperidine'),
        (6, ((8, (1,)),)): ('pyran', 'oxane'),
        (6, ((16, (1,)),)): ('thiopyran', 'thiane'),
        (6, ((7, (1, 2)),)): ('pyridazine', ''),
        (6, ((7, (1, 3)),)): ('pyrimidine', ''),
        (6, ((7, (1, 4)),)): ('pyrazine', 'piperazine'),
        (6, ((8, (1,)), (7, (4,)))): ('', 'morpholine'),
        (6, ((16, (1,)), (7, (4,)))): ('', 'thiomorpholine'),
        # Exact P-22.2.1 Tables 2.2–2.3 retained PIN spellings only.
        (5, ((34, (1,)),)): ('selenophene', ''),
        (5, ((52, (1,)),)): ('tellurophene', ''),
        (6, ((34, (1,)),)): ('selenopyran', ''),
        (6, ((52, (1,)),)): ('telluropyran', ''),
        (6, ((34, (1,)), (7, (4,)))): ('', 'selenomorpholine'),
        (6, ((52, (1,)), (7, (4,)))): ('', 'telluromorpholine'),
    }
    name = retained.get((size, signature), ('', ''))[int(saturated)]
    if name:
        return name
    if size == 3:
        non_n_hetero = sum(counts[e] for e in _ELEMENTS if e != 7)
        stem = ('iridine' if counts[7] else 'irane') if saturated else ('irine' if counts[7] and not non_n_hetero else 'irene')
    elif size == 4:
        stem = ('etidine' if counts[7] else 'etane') if saturated else 'ete'
    elif size == 5:
        stem = ('olidine' if counts[7] else 'olane') if saturated else 'ole'
    elif size == 6:
        # P-22.2.2.1.6 / Table 2.5: the least senior heteroatom controls
        # the stem; phosphorus belongs to group 6C (inine / inane).
        stem = ('inane' if counts[7] or counts[15] else 'ane') if saturated else ('inine' if counts[15] else 'ine')
    else:
        stem = {7: ('epine', 'epane'), 8: ('ocine', 'ocane'),
                9: ('onine', 'onane'), 10: ('ecine', 'ecane')}[size][int(saturated)]
    words, locants = [], []
    for e, prefix in _HW_DESCRIPTORS.items():
        count = counts[e]
        if not count:
            continue
        multiplier = MULT.get(count, '') if count > 1 else ''
        if multiplier.endswith('a') and prefix.startswith('a'):
            multiplier = multiplier[:-1]
        words.append(multiplier+prefix)
        locants.extend(locations[e])
    text = ''.join(w[:-1] if w.endswith('a') and nxt[0] in 'aeiou' else w
                   for w, nxt in zip(words, words[1:]+[stem]))+stem
    locant_prefix = ''
    if len(locants) > 1:
        from .hantzsch_widman_locants import heteroatom_locant_decision
        omission = heteroatom_locant_decision(numbers)
        if omission.status != 'omit':
            locant_prefix = ','.join(map(str, locants))+'-'
    return locant_prefix+text


def is_supported_hw_chalcogen_core(mol, core):
    """Whether ``core`` is one ordinary neutral Se/Te Hantzsch–Widman cycle.

    This narrow predicate is shared with mapped-ring parent construction. It
    does not widen the general/fused valence table: only a connected simple
    3–10-membered cycle with ordinary C/N/O/S/Se/Te atoms is admitted, and a
    Se/Te atom elsewhere in the molecule blocks this operation.
    """
    core = frozenset(core)
    if not 3 <= len(core) <= 10 or not core:
        return False
    if any(a.GetAtomicNum() in {34, 52} and a.GetIdx() not in core for a in iter_atoms(mol)):
        return False
    if not any(mol.GetAtomWithIdx(i).GetAtomicNum() in {34, 52} for i in core):
        return False
    neighbors = {i: set() for i in core}
    for bond in iter_bonds(mol):
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in core and b in core:
            neighbors[a].add(b)
            neighbors[b].add(a)
    if any(len(row) != 2 for row in neighbors.values()):
        return False
    reached, stack = set(), [next(iter(core))]
    while stack:
        atom = stack.pop()
        if atom in reached:
            continue
        reached.add(atom)
        stack.extend(neighbors[atom] - reached)
    if reached != core:
        return False
    allowed = {6, 7, 8, 16, 34, 52}
    for idx in core:
        atom = mol.GetAtomWithIdx(idx)
        number = atom.GetAtomicNum()
        if number not in allowed or atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
            return False
        try:
            if atom.GetTotalValence() != _HW_VALENCE[number]:
                return False
        except (RuntimeError, ValueError):
            return False
    return True


def _simple_hw_cycle(numbers, edges):
    if not 3 <= len(numbers) <= 10 or len(edges) != len(numbers):
        return False
    degree = Counter(i for edge in edges for i in edge)
    if set(degree) != set(range(len(numbers))) or any(degree[i] != 2 for i in range(len(numbers))):
        return False
    adjacency = {i: set() for i in range(len(numbers))}
    for a, b in edges:
        if a not in adjacency or b not in adjacency or a == b:
            return False
        adjacency[a].add(b)
        adjacency[b].add(a)
    reached, pending = set(), [0]
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        pending.extend(adjacency[current] - reached)
    return reached == set(adjacency)


@lru_cache(maxsize=512)
def _reference_states(numbers, edges, allow_tetravalent_carbon=False, saturated_atoms=()):
    """All distinct H populations of maximum noncumulative pi-bond matchings."""
    from rdkit import Chem
    degree = Counter(i for edge in edges for i in edge)
    if any(n in {34, 52} for n in numbers) and not _simple_hw_cycle(numbers, edges):
        return ()
    if any((degree[i] not in {2, 3}
            and not (allow_tetravalent_carbon and n == 6 and degree[i] == 4))
           or degree[i] > _HW_VALENCE[n] for i, n in enumerate(numbers)):
        return ()
    fixed = frozenset(saturated_atoms)
    if any(type(i) is not int or not 0 <= i < len(numbers) for i in fixed):
        return ()
    needed = frozenset(i for i, n in enumerate(numbers) if i not in fixed
                       and (n == 6 and degree[i] < 4 or n == 7 and degree[i] == 2))
    neighbors = {i: {b if a == i else a for a, b in edges if i in (a, b)} for i in needed}

    @lru_cache(None)
    def matchings(remaining):
        if not remaining:
            return ((),)
        a = min(remaining)
        choices = list(matchings(remaining-{a}))
        for b in sorted(neighbors[a] & remaining):
            choices.extend(((a, b),)+pairs for pairs in matchings(remaining-{a, b}))
        maximum = max(map(len, choices))
        return tuple(pairs for pairs in choices if len(pairs) == maximum)

    results, seen = [], set()
    for pairs in matchings(needed):
        matched = {i for edge in pairs for i in edge}
        h = tuple(_HW_VALENCE[n]-degree[i]-int(i in matched) for i, n in enumerate(numbers))
        if h in seen:
            continue
        graph = Chem.RWMol()
        for number, count in zip(numbers, h):
            atom = Chem.Atom(number)
            atom.SetNumExplicitHs(count)
            atom.SetNoImplicit(True)
            graph.AddAtom(atom)
        for a, b in edges:
            graph.AddBond(a, b, Chem.BondType.DOUBLE if (a, b) in pairs or (b, a) in pairs else Chem.BondType.SINGLE)
        ref = graph.GetMol()
        try:
            Chem.SanitizeMol(ref)
        except (ValueError, RuntimeError):
            continue
        indicated = tuple(i for i, n in enumerate(numbers) if n in {6, 7} and i not in matched and h[i])
        results.append((h, indicated, tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
            b.GetIsAromatic(), b.GetBondTypeAsDouble()) for b in iter_bonds(ref))))
        seen.add(h)
    return tuple(results)


def _monocyclic_skeletons(mol, ring):
    ordered = tuple(ring)
    anchor = next(
        (element for element in _ELEMENTS
         if any(mol.GetAtomWithIdx(i).GetAtomicNum() == element for i in ordered)),
        None,
    )
    if anchor is None:
        return
    choices = []
    for direction in (ordered, tuple(reversed(ordered))):
        for start in range(len(ring)):
            seq = direction[start:]+direction[:start]
            numbers = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in seq)
            if numbers[0] != anchor:
                continue
            locs = tuple(tuple(j+1 for j, n in enumerate(numbers) if n == e) for e in _ELEMENTS)
            key = (tuple(j+1 for j, n in enumerate(numbers) if n != 6), locs)
            choices.append((key, seq, numbers))
    best = min(row[0] for row in choices)
    for key, seq, numbers in choices:
        if key != best:
            continue
        edges = tuple((i, (i+1) % len(seq)) for i in range(len(seq)))
        saturated = all(mol.GetBondBetweenAtoms(seq[a], seq[b]).GetBondTypeAsDouble() == 1 for a, b in edges)
        yield _hw_name(numbers, saturated), seq, tuple(map(str, range(1, len(seq)+1))), numbers, edges, saturated, key


@lru_cache(maxsize=8)
def _fused_templates(Chem):
    from .parent_rules import _curated_fused_templates
    seen, result = set(), []
    for template in _curated_fused_templates(Chem):
        parent, query = template['name'], template['mol']
        if parent not in _FUSED_PARENTS or parent in seen:
            continue
        if all(a.IsInRing() and a.GetAtomicNum() in _VALENCE and not a.GetFormalCharge()
               and template['locants'].get(a.GetIdx()) for a in iter_atoms(query)):
            from .preferred_fused import (has_special_retained_fused_frame,
                                           special_retained_fused_locants)
            # These fixed P-25 retained frames have source numbering that
            # cannot be reconstructed by a generic sequential perimeter.
            # Validate every labelled boundary/fusion edge before admission.
            locants = (special_retained_fused_locants(parent, query, template['locants'])
                       if has_special_retained_fused_frame(parent)
                       else _peripheral_junction_locants(query, template['locants']))
            if locants is None:
                continue
            seen.add(parent)
            result.append(dict(template, locants=locants))
    from .preferred_fused import special_retained_fused_locants
    from .retained_hydrocarbon_parents import _vocabulary

    for record in _vocabulary().values():
        if record['name'] not in {'phenanthrene', '9H-fluorene'}:
            continue
        parent = record['name'].removeprefix('9H-')
        query = Chem.MolFromSmiles(record['original_frame_smiles'])
        locants = special_retained_fused_locants(parent, query, {
            int(idx): locant for idx, locant in record['original_locants'].items()})
        if locants is not None:
            result.append({'name': parent, 'mol': query, 'locants': locants})
    return tuple(result)


def _peripheral_junction_locants(query, original):
    """P-25.3.3.1.1: a junction follows the preceding peripheral numeric atom.

    Imported numeric positions anchor the orientation. Some resource records
    assign lettered junctions in the opposite traversal. Reconstruct these
    positions from the annular graph, only when one proved perimeter exists.
    Special fixed numeric systems (e.g. purine) keep their complete numbering.
    """
    if not any(str(loc)[-1:].isalpha() for loc in original.values()):
        return dict(original)
    vertices = {a.GetIdx() for a in iter_atoms(query) if a.GetDegree() == 3}
    junctions = {i for i in vertices if query.GetAtomWithIdx(i).GetAtomicNum()==6}
    peripheral = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()
        if not {a.GetIdx(), n.GetIdx()} <= vertices} for a in iter_atoms(query)}
    if any(len(neighbors) != 2 for neighbors in peripheral.values()):
        return None
    ordinary = {i: str(loc) for i, loc in original.items() if i not in junctions}
    if not ordinary or any(not loc.isdigit() for loc in ordinary.values()):
        return None
    if sorted(map(int, ordinary.values())) != list(range(1, len(ordinary)+1)):
        return None
    start = next(i for i, loc in ordinary.items() if loc == '1')
    for neighbor in sorted(peripheral[start]):
        traversal, previous, current = [start], start, neighbor
        while current != start and current not in traversal:
            traversal.append(current)
            previous, current = current, next(i for i in peripheral[current] if i != previous)
        if current != start or len(traversal) != query.GetNumAtoms():
            continue
        numeric = [ordinary[i] for i in traversal if i in ordinary]
        if numeric != list(map(str, range(1, len(ordinary)+1))):
            continue
        result, preceding, letter = {}, '', 0
        for idx in traversal:
            if idx in ordinary:
                preceding, letter = ordinary[idx], 0
                result[idx] = preceding
            else:
                result[idx] = preceding+chr(ord('a')+letter)
                letter += 1
        return result
    return None


def _skeletons(ctx):
    mol, Chem = ctx.mol, ctx.Chem
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) == 1:
        ring = rings[0]
        if 3 <= len(ring) <= 10 and any(mol.GetAtomWithIdx(i).GetAtomicNum() in _ELEMENTS for i in ring):
            yield from _monocyclic_skeletons(mol, ring)
    elif 5 <= len(rings) <= 6:
        from .fusion_state_skeletons import peri_benzo_ketone_skeletons

        yield from peri_benzo_ketone_skeletons(mol, Chem)
    elif 1 < len(rings) <= 4:
        # P-52.2.4 gives a constructible fusion parent precedence over a
        # bond-locanted von Baeyer fallback.  The fallback is tried by
        # select_ring_state only after the fused/retained H-state renderers
        # have declined the original graph.
        from .parent_rules import _cached_iter_fused_template_matches
        from .cycle_utils import ring_system_extends_core
        retained = False
        if len(rings) == 3 and sorted(map(len, rings)) in ([6, 6, 6], [6, 6, 7]):
            from .fusion_state_skeletons import dibenzo_heterocycle_templates

            for template, match in _cached_iter_fused_template_matches(ctx,
                    ctx.candidate_templates(dibenzo_heterocycle_templates(Chem)), Chem,
                    topology_query=True, require_one_ring_system=True,
                    reject_extra_core_bonds=True):
                if ring_system_extends_core(mol, set(match)):
                    continue
                query = template['mol']
                retained = True
                yield (template['name'], match,
                    tuple(template['locants'][i] for i in range(len(match))),
                    tuple(a.GetAtomicNum() for a in iter_atoms(query)),
                    tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in iter_bonds(query)),
                    False, (), template['method'])
        for template, match in _cached_iter_fused_template_matches(ctx,
                ctx.candidate_templates(_fused_templates(Chem)), Chem, topology_query=True,
                require_one_ring_system=True, reject_extra_core_bonds=True):
            if ring_system_extends_core(mol, set(match)):
                continue
            query = template['mol']
            from .numbered_bicyclic_parents import has_proved_bicyclic_numbering
            if has_proved_bicyclic_numbering(template):
                # Saturated numbered hydrides are not mancude references.
                # Their parent already specifies every annular single bond.
                if any(mol.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()],
                        match[b.GetEndAtomIdx()]).GetBondTypeAsDouble() != 1
                        for b in iter_bonds(query)):
                    continue
                method = ('skeletal-replacement-von-Baeyer'
                          if any(a.GetAtomicNum()!=6 for a in iter_atoms(query))
                          else 'von-Baeyer')
                retained = True
                yield (template['name'], match,
                    tuple(template['locants'][i] for i in range(len(match))),
                    tuple(a.GetAtomicNum() for a in iter_atoms(query)),
                    tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in iter_bonds(query)),
                    True, (), method)
                continue
            retained = True
            yield (template['name'], match, tuple(template['locants'][i] for i in range(len(match))),
                tuple(a.GetAtomicNum() for a in iter_atoms(query)),
                tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in iter_bonds(query)), False, ())
        if not retained and len(rings) == 2:
            from .fusion_state_skeletons import two_face_fusion_templates, benzo_heterocycle_skeletons
            benzo = tuple(benzo_heterocycle_skeletons(mol))
            if benzo:
                yield from benzo
                return
            for template, match in _cached_iter_fused_template_matches(ctx,
                    ctx.candidate_templates(two_face_fusion_templates(mol, Chem)), Chem,
                    topology_query=True, require_one_ring_system=True,
                    reject_extra_core_bonds=True):
                if ring_system_extends_core(mol, set(match)):
                    continue
                query = template['mol']
                yield (template['name'], match,
                    tuple(template['locants'][i] for i in range(len(match))),
                    tuple(a.GetAtomicNum() for a in iter_atoms(query)),
                    tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in iter_bonds(query)),
                    False, (), 'component-fused')
        if not retained and len(rings) == 3:
            specialized = False
            from .benzo_fused_pyridine import skeletons as benzo_fused_skeletons, source_supported
            benzo_fused = benzo_fused_skeletons(mol, Chem)
            if benzo_fused and source_supported(mol):
                specialized = True
                yield from benzo_fused
            from .three_face_aza_fusion import (skeletons as aza_fused_skeletons,
                                                source_supported as aza_fused_source_supported)
            aza_fused = aza_fused_skeletons(mol, Chem)
            if aza_fused and aza_fused_source_supported(mol):
                specialized = True
                yield from aza_fused
            if (sorted(map(len, rings)) == [5, 5, 6]
                    and sorted(a.GetAtomicNum() for a in iter_atoms(mol)) == [6] * 9 + [7, 7, 16]):
                from .fusion_state_skeletons import benzo_imidazothiazole_templates

                for template, match in _cached_iter_fused_template_matches(ctx,
                        ctx.candidate_templates(benzo_imidazothiazole_templates(Chem)), Chem,
                        topology_query=True, require_one_ring_system=True, reject_extra_core_bonds=True):
                    if ring_system_extends_core(mol, set(match)):
                        continue
                    specialized = True
                    query = template['mol']
                    yield (template['name'], match,
                           tuple(template['locants'][i] for i in range(len(match))),
                           tuple(a.GetAtomicNum() for a in iter_atoms(query)),
                           tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in iter_bonds(query)),
                           False, (), 'component-fused-benzo-imidazothiazole')
            from .fusion_state_skeletons import (dibenzopyran_fusion_templates,
                                                 dibenzo_five_member_fusion_templates)
            templates = (dibenzopyran_fusion_templates(mol, Chem)
                         + dibenzo_five_member_fusion_templates(mol, Chem))
            for template, match in _cached_iter_fused_template_matches(ctx,
                    ctx.candidate_templates(templates), Chem,
                    topology_query=True, require_one_ring_system=True,
                    reject_extra_core_bonds=True):
                if ring_system_extends_core(mol, set(match)):
                    continue
                specialized = True
                query = template['mol']
                yield (template['name'], match,
                    tuple(template['locants'][i] for i in range(len(match))),
                    tuple(a.GetAtomicNum() for a in iter_atoms(query)),
                    tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in iter_bonds(query)),
                    False, (), 'component-fused')
            if not specialized:
                from .general_three_face_fusion import skeletons as general_fused_skeletons
                yield from general_fused_skeletons(mol, Chem)
        if (not retained and len(rings) == 4 and sorted(map(len, rings)) == [5, 6, 6, 6]
                and sum(a.GetAtomicNum() == 7 and a.IsInRing() for a in iter_atoms(mol)) == 1):
            from .fusion_state_skeletons import benzo_carbazole_templates

            for template, match in _cached_iter_fused_template_matches(ctx,
                    ctx.candidate_templates(benzo_carbazole_templates(Chem)), Chem,
                    topology_query=True, require_one_ring_system=True,
                    reject_extra_core_bonds=True):
                if ring_system_extends_core(mol, set(match)):
                    continue
                query = template['mol']
                yield (template['name'], match,
                    tuple(template['locants'][i] for i in range(len(match))),
                    tuple(a.GetAtomicNum() for a in iter_atoms(query)),
                    tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in iter_bonds(query)),
                    False, (), template['ring_state_method'])


def _skeleton_references(skeleton):
    if skeleton[5] or len(skeleton) > 8:
        return (None,)
    parent, _seq, locants, numbers, edges = skeleton[:5]
    fixed_h = ()
    if parent == 'fluorene':
        fixed_h = (locants.index('9'),)
    elif parent in {'phenoxazine', 'phenothiazine'}:
        fixed_h = (locants.index('10'),)
    elif len(skeleton) > 7 and skeleton[7] == 'component-fused-benzocarbazole':
        fixed_h = (numbers.index(7),)
    return _reference_states(numbers, edges, saturated_atoms=fixed_h)


def _fully_reduced_naphthyridine_attachment(mol, parent, seq, numbers, edges,
                                           actual, groups):
    """P-54/P-58: the reduced N can supply its own substitutable H."""
    if (not parent.endswith('-naphthyridine') or len(groups) != 1
            or groups[0].kind != 'yl' or groups[0].center not in seq):
        return False
    idx = seq.index(groups[0].center)
    if numbers[idx] != 7 or actual[idx] != 1:
        return False
    rings = [ring for ring in mol.GetRingInfo().AtomRings() if set(ring) <= set(seq)]
    if len(rings) != 2:
        return False
    shared = set(rings[0]).intersection(rings[1])
    if len(shared) != 2:
        return False
    own = next((ring for ring in rings if seq[idx] in set(ring) - shared), None)
    return own is not None and all(
        mol.GetBondBetweenAtoms(seq[a], seq[b]).GetBondTypeAsDouble() == 1
        for a, b in edges if seq[a] in own and seq[b] in own
        and not {seq[a], seq[b]} <= shared)


def _inventory(mol, Chem):
    """Reuse the isotope-free recognition inventory for this original graph."""
    cached = mol_cache_get(mol, "ring_state_inventory")
    if cached is None:
        cached = mol_cache_set(mol, "ring_state_inventory", _build_inventory(mol, Chem))
    return cached


def _build_inventory(mol, Chem):
    from .extended_inventory import selection_inventory
    from types import SimpleNamespace
    # Recognition copy only: annular divalent S/Se/Te have the same
    # carbonyl-class relationship as annular O for detecting possible group
    # patterns. Such a converted atom must not itself become a functional
    # group atom; original element and isotope indices survive in all maps.
    copy = Chem.Mol(mol)
    converted = set()
    recognition_changed = False
    for atom in iter_atoms(copy):
        recognition_changed |= bool(atom.GetIsotope())
        atom.SetIsotope(0)
        if atom.GetAtomicNum() in {16, 34, 52} and atom.IsInRing() and atom.GetTotalValence()==2:
            if atom.GetAtomicNum() in {34, 52}:
                converted.add(atom.GetIdx())
            atom.SetAtomicNum(8)
            recognition_changed = True
    # The peroxide-carbonyl exception is proved on the actual input graph.
    # A recognition copy can otherwise fabricate the proof by erasing isotope
    # labels or replacing ring chalcogens with oxygen. In that case the old
    # inventory rejects the proxy at this carbonyl, so retain that outcome.
    from .functional_inventory import _simple_annular_peroxide_carbonyl
    if recognition_changed:
        for carbon in iter_atoms(copy):
            if carbon.GetAtomicNum() != 6:
                continue
            for oxygen in carbon.GetNeighbors():
                bond = copy.GetBondBetweenAtoms(carbon.GetIdx(), oxygen.GetIdx())
                if (oxygen.GetAtomicNum() == 8 and bond.GetBondTypeAsDouble() in {1, 1.5}
                        and _simple_annular_peroxide_carbonyl(copy, carbon, oxygen)
                        and not _simple_annular_peroxide_carbonyl(
                            mol, mol.GetAtomWithIdx(carbon.GetIdx()),
                            mol.GetAtomWithIdx(oxygen.GetIdx()))):
                    return SimpleNamespace(groups=(), supported=False, principal=())
    inventory = selection_inventory(copy)
    if not inventory.supported:
        return inventory
    if converted and any(converted.intersection(group.atoms) for group in inventory.groups):
        return SimpleNamespace(groups=inventory.groups, supported=False, principal=())
    peroxol_sites = tuple(g for g in inventory.groups if g.kind == 'peroxol')
    if not peroxol_sites:
        return inventory
    from .peroxol_components import principal_peroxol_sites
    selected_peroxols = principal_peroxol_sites(mol, peroxol_sites)
    if not selected_peroxols:
        return SimpleNamespace(groups=inventory.groups, supported=False, principal=())
    # P-63.7 compares the distinct class-18 suffix types after P-41 selects
    # class 18 over amine. Do not multiply unlike OOH/SO/OS/SS operations.
    seniority = min(g.seniority for g in inventory.groups)
    principal = (selected_peroxols if seniority[0] == 18 else
                 tuple(g for g in inventory.groups if g.seniority == seniority))
    return SimpleNamespace(groups=inventory.groups, supported=True, principal=principal)


def _render(ctx, skeleton, reference, inventory, *, principal=None, scope=None,
            numbering_frontier=None):
    groups=inventory.principal if principal is None else principal
    if groups and groups[0].kind=='amine':
        from .amine_components import suffix_site_maps
        seq,locs=skeleton[1:3]
        return [row for assignment in suffix_site_maps(ctx.mol,groups,frozenset(seq),dict(zip(seq,locs)))
                for row in _render_once(ctx,skeleton,reference,inventory,principal=principal,scope=scope,
                                        amine_suffix=assignment,numbering_frontier=numbering_frontier)]
    if groups and groups[0].kind=='imine':
        from .imine_components import suffix_site_maps
        seq,locs=skeleton[1:3]
        return [row for assignment in suffix_site_maps(ctx.mol,groups,frozenset(seq),dict(zip(seq,locs)))
                for row in _render_once(ctx,skeleton,reference,inventory,principal=principal,scope=scope,
                                        imine_suffix=assignment,numbering_frontier=numbering_frontier)]
    return _render_once(ctx,skeleton,reference,inventory,principal=principal,scope=scope,
                        numbering_frontier=numbering_frontier)


def monocyclic_oxo_suffix_name(mol, Chem, sequence, prefixes):
    """Join caller-mapped prefixes to a proved H-W carbonyl/H-state body.

    Legacy external-prefix writers may cover a larger branch vocabulary than
    the mapped whole-parent writer. That difference cannot authorize a bare
    mancude word plus ``one``: the ring still needs its P-58 H operation. A
    scratch ring caps only external substituents, retains actual oxo bonds,
    and supplies the same original numbering to the joint H-state renderer.
    ``handled`` distinguishes this bounded operation from other ring methods;
    a handled empty name is an exclusion, never permission for a raw fallback.
    """
    from .mapped_ring_parents import _core_graph, skeletons
    from .naming_context import MoleculeContext
    from .formatting import format_prefixes, join_prefix_parent

    sequence = tuple(sequence)
    core = frozenset(sequence)
    if (not 3 <= len(core) <= 10 or len(core) != len(sequence)
            or not any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in core)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in _HW_VALENCE
                   or mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetIsotope()
                   or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in core)):
        return False, ''
    edges = tuple(bond for bond in iter_bonds(mol)
                  if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} <= core)
    if (len(edges) != len(core)
            or any(sum(n.GetIdx() in core for n in mol.GetAtomWithIdx(i).GetNeighbors()) != 2
                   for i in core)
            or all(bond.GetBondTypeAsDouble() == 1 for bond in edges)):
        return False, ''
    oxo = tuple(str(loc) for loc, word in prefixes if word == 'oxo')
    if not oxo:
        return False, ''
    targets = []
    for loc in oxo:
        if not loc.isdigit() or not 1 <= int(loc) <= len(sequence):
            return False, ''
        center = sequence[int(loc) - 1]
        if mol.GetAtomWithIdx(center).GetAtomicNum() != 6:
            return False, ''  # Annular sulfur-oxide suffixes have another writer.
        oxygen = tuple(bond.GetOtherAtom(mol.GetAtomWithIdx(center))
                       for bond in mol.GetAtomWithIdx(center).GetBonds()
                       if bond.GetOtherAtomIdx(center) not in core
                       and bond.GetBondTypeAsDouble() == 2
                       and bond.GetOtherAtom(mol.GetAtomWithIdx(center)).GetAtomicNum() == 8)
        if (len(oxygen) != 1 or oxygen[0].GetDegree() != 1
                or oxygen[0].GetFormalCharge() or oxygen[0].GetIsotope()):
            return False, ''
        targets.append(center)
    if len(set(targets)) != len(targets):
        return True, ''
    # P-22 fixed heteroatom numbering precedes carbonyl and prefix locants.
    # A reverse legacy orientation must not survive merely because it gives
    # a substituent a lower locant.
    frames = tuple(row for row in skeletons(mol, Chem, core)
                   if row[7] == 'Hantzsch-Widman' and tuple(row[1]) == sequence)
    if not frames:
        return True, ''
    prepared = _core_graph(mol, Chem, core)
    if prepared is None:
        return True, ''
    graph, original = prepared
    inverse = {idx: local for local, idx in enumerate(original)}
    scratch = Chem.RWMol(graph)
    for center in targets:
        local = inverse[center]
        atom = scratch.GetAtomWithIdx(local)
        count = atom.GetTotalNumHs()
        if count < 2:
            return True, ''
        atom.SetNumExplicitHs(count - 2)
        atom.SetNoImplicit(True)
        oxygen = scratch.AddAtom(Chem.Atom(8))
        scratch.AddBond(local, oxygen, Chem.BondType.DOUBLE)
    work = scratch.GetMol()
    try:
        Chem.SanitizeMol(work)
    except (ValueError, RuntimeError):
        return True, ''
    inventory = _inventory(work, Chem)
    if (not inventory.supported or not inventory.principal
            or any(group.kind not in _CARBONYLS for group in inventory.principal)
            or {group.center for group in inventory.principal}
               != {inverse[idx] for idx in targets}):
        return True, ''
    ctx = MoleculeContext(work, Chem)
    rows = []
    for frame in frames:
        local = (frame[0], tuple(inverse[i] for i in frame[1]), *frame[2:])
        for reference in _reference_states(frame[3], frame[4]):
            rows.extend(_render(ctx, local, reference, inventory))
    if not rows:
        return True, ''
    best = min(row.numbering_key for row in rows)
    names = {row.name for row in rows if row.numbering_key == best}
    if len(names) != 1:
        return True, ''
    body = names.pop()
    remaining = tuple((loc, word) for loc, word in prefixes if word != 'oxo')
    return True, join_prefix_parent(format_prefixes(remaining), body)


def _render_atom_sets(mol):
    """Share original graph coverage across alternative ring numberings."""
    cached = mol_cache_get(mol, 'ring_render_atom_sets')
    if cached is None:
        ring, needed = set(), set()
        for atom in iter_atoms(mol):
            idx = atom.GetIdx()
            if atom.IsInRing():
                ring.add(idx)
            if atom.GetAtomicNum() > 1 or atom.GetIsotope() or int(atom.GetChiralTag()):
                needed.add(idx)
        cached = (frozenset(ring), frozenset(needed))
        mol_cache_set(mol, 'ring_render_atom_sets', cached)
    return cached


@lru_cache(maxsize=512)
def _ring_perimeter(edges, size, saturated, parent='', numbers=()):
    if saturated:
        return frozenset(range(size))
    peripheral = peripheral_ring_atoms(edges)
    if peripheral is not None:
        return peripheral
    if size >= 18 and parent.startswith('dibenzo[') and numbers == (6,) * size:
        from .fusion_state_skeletons import peri_benzo_perimeter

        return peri_benzo_perimeter(parent, edges, size)
    if size == 12 and parent.startswith('benzo[') and ']imidazo[' in parent:
        from .fusion_state_skeletons import benzo_imidazothiazole_perimeter

        return benzo_imidazothiazole_perimeter(parent, edges, tuple(numbers))
    if size == 17 and parent.startswith('benzo[') and parent.endswith(']carbazole'):
        from .fusion_state_skeletons import benzo_carbazole_perimeter

        return benzo_carbazole_perimeter(parent, edges, numbers)
    if parent == 'dibenzo[b,f][1,4]oxazepine':
        from .fusion_state_skeletons import dibenzo_heterocycle_perimeter

        return dibenzo_heterocycle_perimeter(parent, edges, tuple(numbers))
    # The ordinary helper is intentionally limited to two faces. The bounded
    # retained and constructed three-face families use independent topology
    # proofs; actual H/stereo still come from the input graph.
    from .preferred_fused import special_retained_fused_perimeter
    peripheral = special_retained_fused_perimeter(parent, edges, size)
    if peripheral is not None:
        return peripheral
    from .fusion_state_skeletons import dibenzopyran_fusion_perimeter
    peripheral = dibenzopyran_fusion_perimeter(parent, edges, size)
    if peripheral is not None:
        return peripheral
    from .fusion_state_skeletons import dibenzo_five_member_fusion_perimeter
    peripheral = dibenzo_five_member_fusion_perimeter(parent, edges, size, tuple(numbers))
    if peripheral is not None:
        return peripheral
    from .benzo_fused_pyridine import perimeter as benzo_fused_perimeter
    peripheral = benzo_fused_perimeter(parent, edges, size)
    if peripheral is not None:
        return peripheral
    from .three_face_aza_fusion import perimeter as aza_fused_perimeter
    peripheral = aza_fused_perimeter(parent, edges, size)
    if peripheral is not None:
        return peripheral
    from .three_face_heterolactam_fusion import perimeter as heterolactam_perimeter
    peripheral = heterolactam_perimeter(parent, edges, size)
    if peripheral is not None:
        return peripheral
    from .fused_imidazo_quinoline import perimeter as imidazo_quinoline_perimeter
    peripheral = imidazo_quinoline_perimeter(parent, edges, size)
    if peripheral is not None:
        return peripheral
    from .general_three_face_fusion import perimeter as general_fused_perimeter
    peripheral = general_fused_perimeter(parent, edges, size, tuple(numbers))
    if peripheral is not None:
        return peripheral
    from .three_face_benzoxepine_acid import perimeter as fused_acid_perimeter
    return fused_acid_perimeter(parent, edges, size)


@lru_cache(maxsize=512)
def _ring_degrees(size, edges):
    return tuple(sum(i in edge for edge in edges) for i in range(size))


@lru_cache(maxsize=512)
def _saturated_ring_hydrogens(numbers, edges):
    degree = _ring_degrees(len(numbers), edges)
    return tuple(_HW_VALENCE[n] - degree[i] for i, n in enumerate(numbers))


def _annular_sulfur_dioxide(mol, sulfur, core):
    atom = mol.GetAtomWithIdx(sulfur)
    if (atom.GetAtomicNum() != 16 or atom.GetDegree() != 4 or atom.GetTotalValence() != 6
            or atom.GetIsotope() or atom.GetFormalCharge() or atom.GetChiralTag()):
        return False
    internal, external = [], []
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetIdx() in core and bond.GetBondTypeAsDouble() == 1:
            internal.append(other.GetAtomicNum())
        elif (other.GetIdx() not in core and bond.GetBondTypeAsDouble() == 2
              and other.GetAtomicNum() == 8 and other.GetDegree() == 1
              and not other.GetIsotope() and not other.GetFormalCharge()):
            external.append(other.GetIdx())
    return len(internal) == len(external) == 2


@lru_cache(maxsize=512)
def _hydrogen_witness_locants(locs, pairs, paths, hydro_paths, reference_pi, accommodated_pi):
    """Translate exact immutable hydrogen witnesses, retaining their order."""
    return (
        tuple(tuple(locs[j] for j in pair) for pair in pairs),
        tuple(tuple(locs[j] for j in path) for path in paths),
        tuple(tuple(locs[j] for j in path) for path in hydro_paths),
        tuple(tuple(locs[j] for j in pair) for pair in reference_pi),
        tuple(tuple(locs[j] for j in pair) for pair in accommodated_pi),
    )


def _ring_bond_states(mol):
    cached = mol_cache_get(mol, 'ring_bond_states')
    if cached is None:
        cached = {}
        for bond in iter_bonds(mol):
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            cached[(a, b) if a < b else (b, a)] = (
                bond.GetBondTypeAsDouble(), bond.GetIsAromatic())
        mol_cache_set(mol, 'ring_bond_states', cached)
    return cached


def _ring_core_annotation_flags(mol, core, *, include_annotations=False):
    cache = mol_cache_dict(mol, 'ring_core_isotope_flags')
    if core not in cache:
        masses = {mol.GetAtomWithIdx(i).GetIsotope() for i in core
                  if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
        cache[core] = (bool(len(masses) > 1 and any(masses)), None)
    if not include_annotations:
        return cache[core]
    if cache[core][1] is None:
        atoms = tuple(mol.GetAtomWithIdx(i) for i in core)
        masses = {a.GetIsotope() for a in atoms}
        isotope_locants = bool(len(masses) > 1 and any(masses)
            or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                   for a in atoms for n in a.GetNeighbors()))
        cache[core] = (cache[core][0], isotope_locants)
    # Legacy stereo assignment can clean tags on the API-owned graph. Read
    # those live facts at the same rendering stage as before; isotope and
    # connectivity preparation remains reusable.
    numbered = (cache[core][1]
        or any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in core)
        or any(int(b.GetStereo()) for b in iter_bonds(mol)
               if b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core))
    return cache[core][0], numbered


_NO_RING_PREPARATION = object()


@dataclass(frozen=True, slots=True)
class _RingPreparation:
    carbonyls: tuple
    oxygen_suffix_groups: tuple
    modified_oxygen_suffix: bool
    sites: tuple
    nitrogen_sites: tuple
    aliases: tuple
    suffix_atoms: frozenset
    branch_edges: dict
    ester_owner_tokens: tuple = ()
    ester_arms: tuple = ()
    general_branches: bool = False


def _ordinary_ring_ester_owners(mol, Chem, groups, core):
    """Prove P-65.6 ring-C(=O)O-alkyl units on the original input graph."""
    names = {1:'methyl',2:'ethyl',3:'propyl',4:'butyl',5:'pentyl',6:'hexyl'}
    tokens, owned = [], set()
    for group in groups:
        if group.kind != 'ester' or group.owner not in core or group.center in core:
            return None
        carbon = mol.GetAtomWithIdx(group.center)
        oxygens = [atom for atom in carbon.GetNeighbors() if atom.GetAtomicNum()==8]
        double = [atom for atom in oxygens if mol.GetBondBetweenAtoms(group.center,atom.GetIdx()).GetBondTypeAsDouble()==2]
        single = [atom for atom in oxygens if mol.GetBondBetweenAtoms(group.center,atom.GetIdx()).GetBondTypeAsDouble()==1]
        if (carbon.GetAtomicNum()!=6 or carbon.GetTotalValence()!=4
                or carbon.GetDegree()!=3 or len(double)!=1 or len(single)!=1
                or single[0].GetIdx()!=group.ligand
                or double[0].GetDegree()!=1 or single[0].GetDegree()!=2
                or single[0].GetTotalValence()!=2
                or set(group.atoms)!={group.center,double[0].GetIdx(),single[0].GetIdx()}
                or mol.GetBondBetweenAtoms(group.center,group.owner) is None
                or mol.GetBondBetweenAtoms(group.center,group.owner).GetBondTypeAsDouble()!=1):
            return None
        roots=[atom.GetIdx() for atom in single[0].GetNeighbors() if atom.GetIdx()!=group.center]
        if len(roots)!=1 or mol.GetBondBetweenAtoms(roots[0],group.ligand).GetBondTypeAsDouble()!=1:
            return None
        text, atoms = mapped_branch(mol, Chem, roots[0], group.ligand)
        if (text!=names.get(len(atoms)) or not atoms or atoms & (core | owned | set(group.atoms))):
            return None
        for idx in atoms:
            atom=mol.GetAtomWithIdx(idx)
            carbon_neighbors=sum(n.GetIdx() in atoms for n in atom.GetNeighbors())
            if (atom.GetAtomicNum()!=6 or atom.IsInRing() or atom.GetTotalValence()!=4
                    or atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumRadicalElectrons()
                    or int(atom.GetChiralTag()) or carbon_neighbors>(1 if idx==roots[0] else 2)
                    or any(b.GetBondTypeAsDouble()!=1 or int(b.GetStereo()) for b in atom.GetBonds())
                    or any(n.GetAtomicNum()==1 and n.GetIsotope() for n in atom.GetNeighbors())):
                return None
        tokens.append(text);owned.update(atoms)
    if not tokens or len(set(tokens))!=1 or len(tokens)>1 and len(tokens) not in MULT:
        return None
    return tuple(tokens),frozenset(owned)


def _ring_preparation(mol, Chem, core, inventory_groups, groups, scope,
                      suffix_sites, nitrogen_atoms, ester_arms=(), general_branches=False):
    """Share physical suffix/branch facts, leaving all locants to the caller."""
    groups, inventory_groups = tuple(groups), tuple(inventory_groups)
    scope = None if scope is None else frozenset(scope)
    key = (core, inventory_groups, groups, scope, tuple(suffix_sites),
           tuple(nitrogen_atoms), ester_arms, general_branches)
    cache = mol_cache_dict(mol, 'ring_preparation')
    try:
        prepared = cache.get(key, _NO_RING_PREPARATION)
    except TypeError:
        return _build_ring_preparation(mol, Chem, *key)
    if prepared is _NO_RING_PREPARATION:
        prepared = _build_ring_preparation(mol, Chem, *key)
        cache[key] = prepared
    return prepared


def _build_ring_preparation(mol, Chem, core, inventory_groups, groups, scope,
                            suffix_sites, nitrogen_atoms, ester_arms=(), general_branches=False):
    kind = groups[0].kind if groups else ''
    if kind in {'oxime', 'oxime-ether', 'hydrazide'}:
        return None
    def external_subordinate_carbonic_derivative(group):
        # A selected acid can carry a separately mapped carbamate as a
        # prefix, including on an attached ring. A ring N may be the
        # carbamate N, but its carbonyl must lie outside the parent and any
        # shared atom must be that one N. Annular carbamates require a
        # different H-state and suffix construction.
        if group.kind not in {'carbamate', 'urea'} or group.center in core:
            return False
        if kind in {'yl', 'ylidene'} and scope is not None:
            from .extended_inventory import selection_inventory
            source = selection_inventory(mol)
            frontier = source.source_frontier
            if (not source.supported or frontier.status != 'selected'
                    or frontier.unknown_pairs or frontier.cycle or not frontier.groups
                    or tuple(frontier.groups) != tuple(source.principal)
                    or any(item.kind != 'acid' or item.center in scope for item in frontier.groups)):
                return False
        elif kind != 'acid':
            return False
        overlap = set(group.atoms) & core
        return not overlap or (len(overlap) == 1 and all(
            mol.GetAtomWithIdx(idx).GetAtomicNum() == 7
            and mol.GetBondBetweenAtoms(group.center, idx) is not None
            and mol.GetBondBetweenAtoms(group.center, idx).GetBondTypeAsDouble() == 1
            for idx in overlap))

    if any(g.kind not in _SUPPORTED_GROUPS
           and g.kind != 'ester'
           and not external_subordinate_carbonic_derivative(g)
           and not (general_branches and g.center not in core
                    and len(core.intersection(g.atoms)) <= 1)
           for g in inventory_groups):
        return None
    # Subordinate esters remain mapped alkoxycarbonyl branches. A selected
    # ester additionally requires the original ring/acyl/alkyl ownership proof
    # below before entering its functional-class carboxylate grammar.
    carbonyls = tuple(g.center for g in groups if g.kind in _CARBONYLS)
    ester_tokens, ester_atoms = (), frozenset()
    if ester_arms:
        if (kind != 'ester' or len(ester_arms) != len(groups)
                or any(arm.group.center != group.center or arm.group.ligand != group.ligand
                       for arm, group in zip(ester_arms, groups, strict=True))):
            return None
        ester_atoms = frozenset().union(*(arm.atoms for arm in ester_arms))
        if (len(ester_atoms) != sum(len(arm.atoms) for arm in ester_arms)
                or ester_atoms & (core | set().union(*(set(g.atoms) for g in groups)))):
            return None
    elif kind == 'ester':
        ester = _ordinary_ring_ester_owners(mol, Chem, groups, core)
        if ester is None:
            return None
        ester_tokens, ester_atoms = ester
    oxygen_suffix_groups = ()
    modified_oxygen_suffix = False
    if kind == 'alcohol' or kind in _CARBONYLS:
        from dataclasses import replace
        from .oxygen_characteristic_suffixes import oxygen_group
        oxygen_suffix_groups = tuple(replace(g, kind='ketone') if g.kind in _CARBONYLS
                                     else g for g in groups)
        units = tuple(oxygen_group(mol, g) for g in oxygen_suffix_groups)
        modified_oxygen_suffix = any(unit is not None and any(mol.GetAtomWithIdx(i).GetIsotope()
            for i in unit.oxygen_atoms + unit.attached_hydrogens) for unit in units)
    sites, suffix_atoms, nitrogen_sites, aliases = [], set(ester_atoms), [], []
    if kind == 'amide':
        # P-66.1.1.3.1.1 keeps exocyclic N substituents on the amide
        # suffix, rather than converting a heterocyclic carboxamide into a
        # chain replacement name. Prove ordinary ring-C(=O)-N roles first.
        from .complete_functional_parent import _ordinary_external_amide_core
        for g in groups:
            if g.owner not in core or not _ordinary_external_amide_core(mol, g):
                return None
            nitrogen_sites.append((g.ligand, g.owner))
            oxygen = next(i for i in g.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 8)
            aliases.append((oxygen, g.owner, 'O'))
    if kind == 'sulfonamide':
        from .sulfur_selection import sulfonamide_n_sites
        if len({g.owner for g in groups}) != len(groups):
            return None
        for g in groups:
            if sulfonamide_n_sites(mol, g) is None or g.owner not in core:
                return None
            nitrogen_sites.append((g.ligand, g.owner))
    for g in groups:
        if g.kind in _EXTERNAL_HETEROATOM_GROUPS and any(
            mol.GetAtomWithIdx(i).GetIsotope() or any(n.GetIsotope()
                for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() == 1)
            for i in g.atoms):
            return None
        if g.kind in {'alcohol', 'yl', 'ylidene'} | _CARBONYLS:
            site = g.center
        elif g.kind in {'amine', 'imine'}:
            site = suffix_sites[len(sites)]
        else:
            site = g.owner
            if g.kind == 'amide':
                # The carbonyl centre remains part of the suffix, with an
                # original-site C locant; N carries its own N/N^site locant.
                aliases.append((g.center, site, 'C'))
        if site not in core or mol.GetAtomWithIdx(site).GetAtomicNum() not in (
                {6, 7} if (kind in {'yl', 'nitrile'} or g.kind == 'carbamate-anion'
                           or ester_arms) else {6}):
            return None
        if g.kind in _EXTERNAL_HETEROATOM_GROUPS:
            aliases.append((g.center, site, 'P' if g.kind == 'phosphonic-acid' else 'S'))
        sites.append(site)
        outside = set(g.atoms) - core
        suffix_atoms.update(outside)
        if modified_oxygen_suffix:
            for idx in outside:
                if mol.GetAtomWithIdx(idx).GetAtomicNum() == 8:
                    aliases.append((idx, site, ''))
        if g.kind == 'peroxol':
            from .peroxol_components import suffix_operation
            operation = suffix_operation(mol, g)
            if operation is None:
                return None
            aliases.extend((oxygen, site, '') for oxygen in operation['atoms'])
    mapped = core | {target for target, _host, _prefix in aliases}
    mapped |= set(nitrogen_atoms) | {nitrogen for nitrogen, _owner in nitrogen_sites}
    suffix_atoms.difference_update(mapped)
    if any(mol.GetAtomWithIdx(i).GetIsotope() or any(n.GetIsotope()
            for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() == 1)
            for i in suffix_atoms):
        return None
    branch_edges = {}
    for idx in core | set(nitrogen_atoms) | {n for n, _owner in nitrogen_sites}:
        rows = []
        for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
            neighbor = atom.GetIdx()
            if (scope is not None and neighbor not in scope
                    or neighbor in mapped or neighbor in suffix_atoms
                    or atom.GetAtomicNum() == 1):
                continue
            order = mol.GetBondBetweenAtoms(idx, neighbor).GetBondTypeAsDouble()
            carbon_ylidene = (general_branches and atom.GetAtomicNum() == 6
                              and not atom.IsInRing() and not atom.GetFormalCharge()
                              and (atom.GetDegree() == 1 and atom.GetTotalNumHs() == 2
                                   or atom.GetDegree() == 2 and atom.GetTotalNumHs() == 1
                                   and any(n.GetIsAromatic() and n.GetAtomicNum() == 6
                                           for n in atom.GetNeighbors())))
            allowed = order == 1 or order == 2 and (
                atom.GetAtomicNum() in {7, 8} or carbon_ylidene)
            rows.append((neighbor, allowed))
        branch_edges[idx] = tuple(rows)
    return _RingPreparation(carbonyls, oxygen_suffix_groups,
        modified_oxygen_suffix, tuple(sites), tuple(nitrogen_sites), tuple(aliases),
        frozenset(suffix_atoms), branch_edges, ester_tokens, ester_arms, general_branches)


def _ring_render_branches(ctx, seq, nmap, mapping, core, needed, scope, prepared):
    """Map branches only after the physical ring state can be constructed."""
    mol, Chem = ctx.mol, ctx.Chem
    branch = mapped_branch
    if prepared.general_branches:
        from .component_mapping import mapped_general_branch

        branch = mapped_general_branch
    prefixes, embedded = [], set(prepared.suffix_atoms)
    prefix_branch_ownership = []
    for idx in (*seq, *nmap):
        for branch_root, allowed in prepared.branch_edges[idx]:
            if branch_root in embedded:
                continue
            if not allowed:
                return None
            text, atoms = branch(mol, Chem, branch_root, idx)
            if not text or any(c.isspace() for c in text) or atoms & (core | embedded):
                return None
            prefixes.append((mapping[idx], text))
            prefix_branch_ownership.append((int(branch_root), int(idx), frozenset(atoms), str(text)))
            embedded.update(atoms)
    bindings = hydrogen_bindings(mol, mapping)
    if set(bindings) | embedded != (needed if scope is None else needed & set(scope)):
        return None
    from .cross_route_selection import _annotation_ownership
    if not _annotation_ownership(mol, bindings):
        return None
    return prefixes, embedded, prefix_branch_ownership, bindings


def _render_once(ctx, skeleton, reference, inventory, *, principal=None, scope=None,
                 amine_suffix=None, imine_suffix=None, numbering_frontier=None):
    mol, Chem = ctx.mol, ctx.Chem
    parent, seq, locs, numbers, edges, saturated, fixed_key = skeleton[:7]
    absolute = skeleton[8] if len(skeleton) > 8 else None
    unsaturation = getattr(absolute, 'unsaturation', None)
    # An absolute P-31 state has no peripheral indicated-/added-H operation.
    # Its explicit bond locants remain sufficient for a fused graph with an
    # internal junction, where a single all-atom outer perimeter need not exist.
    perimeter = (frozenset(range(len(seq))) if absolute is not None else
                 _ring_perimeter(tuple(edges), len(seq), saturated, parent, tuple(numbers)))
    if perimeter is None:
        return ()
    core, mapping = frozenset(seq), dict(zip(seq, locs))
    ring_atoms, needed = _render_atom_sets(mol)
    if scope is None and core != ring_atoms:
        return ()
    groups = inventory.principal if principal is None else principal
    kind = groups[0].kind if groups else ''
    if len(skeleton) > 7 and skeleton[7] == 'component-fused-peri-ketone':
        if (kind != 'ketone' or len(groups) != 1 or len(needed - core) != 1
                or set(groups[0].atoms) != {groups[0].center} | (needed - core)):
            return ()
    suffix_sites, nmap = (), {}
    if kind in {'amine', 'imine'}:
        assignment = amine_suffix if kind == 'amine' else imine_suffix
        if assignment is None:
            return ()
        suffix_sites, nmap = assignment
    prepared = _ring_preparation(mol, Chem, core, inventory.groups, groups,
                                 scope, suffix_sites, tuple(nmap),
                                 getattr(inventory, 'ester_arms', ()),
                                 getattr(inventory, 'general_branches', False))
    if prepared is None:
        return ()
    carbonyls = prepared.carbonyls
    oxygen_suffix_groups = prepared.oxygen_suffix_groups
    modified_oxygen_suffix = prepared.modified_oxygen_suffix
    sites = prepared.sites
    if kind == 'amide':
        from .amide_site_locants import external_amide_nitrogen_locants
        assignment = external_amide_nitrogen_locants(mol, Chem, groups, mapping)
        if assignment is None:
            return ()
        nmap.update(assignment.nitrogen_locants)
    else:
        for nitrogen, owner in prepared.nitrogen_sites:
            nmap[nitrogen] = 'N' + (mapping[owner] if len(groups) > 1 else '')
    for target, host, prefix in prepared.aliases:
        mapping[target] = (mapping[host] if kind == 'amide'
            and mol.GetAtomWithIdx(target).GetAtomicNum() == 6
            and mol.GetAtomWithIdx(target).GetIsotope() else prefix + mapping[host])
    mapping.update(nmap)
    branches = None
    if ctx.telemetry is not None or mol_cache_get(mol, 'candidate_telemetry') is not None:
        # Keep the instrumented path's historical work counters and ordering.
        branches = _ring_render_branches(ctx, seq, nmap, mapping, core, needed, scope, prepared)
        if branches is None:
            return ()
    capped_cache = mol_cache_dict(mol, 'ring_capped_hydrogens')
    if core not in capped_cache:
        capped_cache[core] = {
            i: mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True) + sum(
                b.GetBondTypeAsDouble() for b in mol.GetAtomWithIdx(i).GetBonds()
                if b.GetOtherAtomIdx(i) not in core
                and mol.GetAtomWithIdx(b.GetOtherAtomIdx(i)).GetAtomicNum() > 1)
            for i in core
        }
    actual = tuple(capped_cache[core][i] for i in seq)
    if absolute is not None:
        # P-31 changes the multiple-bond ending of this particular numbered
        # skeleton. Its exact original annular bonds fix the H population;
        # this is not an alternative mancude state or a hydro operation.
        expected = absolute.reference_hydrogens
        if actual != expected:
            return ()
        h, indicated, deltas = expected, (), (0,) * len(seq)
    elif saturated:
        if any(n in {34, 52} for n in numbers) and not _simple_hw_cycle(numbers, edges):
            return ()
        expected = _saturated_ring_hydrogens(tuple(numbers), tuple(edges))
        if kind == 'acid' and 16 in numbers and _simple_hw_cycle(numbers, edges):
            # Oxidation by two pendant oxo groups changes sulfur's valence
            # without adding four hydrogens to the divalent ring hydride.
            actual = tuple(count - 4 if number == 16 and count == reference + 4
                           and _annular_sulfur_dioxide(mol, idx, core) else count
                           for idx, number, count, reference in zip(seq, numbers, actual, expected))
        if actual != expected:
            return ()
        h = expected
        indicated, deltas = (), (0,)*len(seq)
    else:
        h, indicated, bonds = reference
        deltas = tuple(a-b for a, b in zip(actual, h))
        if any(v not in {0, 1} for v in deltas) or sum(deltas) % 2:
            return ()
        deltas = tuple(map(int, deltas))
        original_bonds = _ring_bond_states(mol)
        from .localized_bicyclic_states import requires_explicit_aza_five_five_bonds
        unique_pi = False
        if requires_explicit_aza_five_five_bonds(mol, core):
            from .hydrogen_accommodation import unique_capped_pi_graph

            unique_pi = unique_capped_pi_graph(tuple(numbers), tuple(edges), tuple(map(int, actual)))
            if not unique_pi:
                return ()
        if not unique_pi and (len(skeleton)>7 and skeleton[7] in {
                'component-fused', 'component-fused-benzofuran-pyridine', 'component-fused-benzocarbazole',
                'component-fused-three-face-azine', 'component-fused-general',
                'component-fused-benzo-imidazothiazole', 'component-fused-peri-ketone',
                'benzo-heterocycle'}
                or len(skeleton)>7 and skeleton[7] == 'retained-fused'
                and (kind in {'acid', 'carboxylate'} and len(groups) == 1
                     and any(g.kind in _CARBONYLS and g.center in core for g in inventory.groups)
                     or not groups and any(deltas))
                or (len(numbers) in {3, 4, 7}
                    and set(numbers) <= {6, 7, 8, 16}
                    and _simple_hw_cycle(numbers, edges))):
            from .hydrogen_accommodation import unique_capped_pi_graph
            # P-54/P-58 hydrogenation or a subordinate annular oxo prefix
            # can relocate a double bond. Admit it only when capped valences
            # fix a unique pi graph; joint_hydrogen_states below must still
            # prove the original-to-input path and hydrogen balance. The
            # A retained fused acid with a subordinate annular carbonyl uses
            # the same source-graph uniqueness proof before H accommodation.
            unique_pi = unique_capped_pi_graph(tuple(numbers),tuple(edges),tuple(map(int,actual)))
            if (not unique_pi and not inventory.groups and len(core) == mol.GetNumAtoms()
                    and all(not a.GetIsotope() and not a.GetFormalCharge() and not a.GetNumRadicalElectrons()
                            and not int(a.GetChiralTag()) for a in mol.GetAtoms())
                    and all(not int(b.GetStereo()) for b in mol.GetBonds())):
                # With no substituents, annotations or suffix sites, a
                # skeleton automorphism cannot move a distinguished atom.
                # Compare every capped pi matching as an unlabeled molecule,
                # while preserving the stricter mapped proof for all other
                # cases. Distinct localized isomers still fail this check.
                unique_pi = unique_capped_pi_graph(tuple(numbers), tuple(edges), tuple(map(int,actual)),
                                                    allow_automorphisms=True)
        if (len(skeleton) > 7 and skeleton[7] in {
                'component-fused-general', 'component-fused-benzo-imidazothiazole',
                'component-fused-peri-ketone'} and not unique_pi):
            # The indexed witness proves numbered topology, not a localized
            # pi assignment. Leave distinct allocations to the complete
            # Delta or bond-locanted writer instead of emitting a bare name.
            return ()
        if (parent.startswith(('cyclopenta[', 'cyclohepta[')) and not unique_pi
                and not any(aromatic for _order, aromatic in original_bonds.values())):
            # A nonaromatic carbon-fused system can have multiple maximum
            # matchings with the same hydrogen population. A bare fusion
            # descriptor alone does not prove which pi graph it denotes;
            # retain the bond-locanted construction until that is known.
            return ()
        for a, b, aromatic, reference_order in bonds:
            left, right = seq[a], seq[b]
            order, aromatic_input = original_bonds[
                (left, right) if left < right else (right, left)]
            # A suffix can accommodate indicated H and rearrange the parent's
            # pi bonds (P-58.2.3). For a single monocycle its capped valence
            # vector fixes the remaining noncumulative pattern; testing the
            # reference's aromatic flags would exclude carbonyl-accommodating
            # C-indicated hydrides. For simple seven-membered H-W cycles,
            # a unique capped pi graph also proves P-31 hydro operations
            # whose remaining double bonds shifted along an alternating path.
            if aromatic_input and not aromatic and not carbonyls and not unique_pi and not any(deltas):
                return ()
            if not aromatic_input and order > 1 and not (aromatic or order == reference_order or carbonyls or unique_pi or any(deltas)):
                return ()
    # Demand belongs to suffix operations, not merely to distinct atom sites.
    # Two geminal carboxy groups each consume one substitutable H.
    demand = Counter()
    for site in sites:
        demand[seq.index(site)] += 2 if site in carbonyls or kind in {'imine', 'ylidene'} else 1
    if _fully_reduced_naphthyridine_attachment(mol, parent, seq, numbers, edges, actual, groups):
        # Saturation supplies the N–H before the substituent loses it. The
        # original capped H vector still has to prove all four hydro locants.
        demand.clear()
    if absolute is not None:
        if any(demand[i] > h[i] for i in range(len(seq))):
            return ()
        from .hydrogen_accommodation import HydrogenAccommodation
        pi = (tuple((a - 1, b - 1) for a, b, order in unsaturation.edge_locants if order == 2)
              if unsaturation is not None else annular_pi_bonds(mol, Chem, seq))
        states = (HydrogenAccommodation((), (), (), (), (), pi, pi),)
    elif any(n in {34, 52} for n in numbers):
        # P-54.4.1 applies hydro prefixes to partially hydrogenated H-W
        # monocycle parents. Reuse the existing P-58 alternating-path state
        # operation with a local valence-equivalent token for Se/Te. Those
        # atoms have bonding number 2 (P-22 Table 2.4) and are not vertices in
        # the helper's C/N pi matching, so mapping only their comparison token
        # to S preserves its graph operation without widening the shared
        # C/N/O/S chemistry domain. The original numbers remain in all naming,
        # mapping, suffix, and ownership paths.
        actual_pi = annular_pi_bonds(mol, Chem, seq)
        state_numbers = tuple(16 if number in {34, 52} else number for number in numbers)
        states = joint_hydrogen_states(state_numbers, tuple(edges), tuple(h),
            tuple(map(int, actual)), tuple(demand[i] for i in range(len(seq))), actual_pi)
        if not states:
            return ()
    else:
        states = joint_hydrogen_states(tuple(numbers), tuple(edges), tuple(h), tuple(map(int, actual)),
            tuple(demand[i] for i in range(len(seq))), annular_pi_bonds(mol, Chem, seq))
    if not states:
        return ()
    rendering = _RingRendering(ctx, skeleton, core, mapping, tuple(groups), kind,
        prepared, scope, nmap, needed, tuple(h), tuple(map(int, actual)),
        _NO_RING_BRANCHES if branches is None else branches)
    rows = []
    # P-31.1.1 compound bond locants and explicitly marked von Baeyer
    # endpoint omission remain separate inputs to the pure locant conversion.
    terms = _ring_numbering_terms(tuple(locs), tuple(edges), perimeter, fixed_key,
        tuple(indicated), tuple(mapping[i] for i in sites), tuple(states),
        unsaturation.edge_locants if unsaturation is not None else None,
        bool(getattr(absolute, 'von_baeyer', False)))
    for state, added, hydro, leading, suffix_locs, chemical_key in terms:
        plan = _RingStatePlan(rendering, state, added, hydro, leading, suffix_locs, chemical_key)
        if numbering_frontier is None:
            rows.extend(plan.render())
        else:
            rows.extend(numbering_frontier.render_or_defer(plan))
    return rows


def _finish_ring_state(plan, numbering_frontier=None):
    rendering = plan.rendering
    ctx, skeleton = rendering.ctx, rendering.skeleton
    mol, Chem = ctx.mol, ctx.Chem
    parent, seq, locs, numbers, edges, saturated, _fixed_key = skeleton[:7]
    absolute = skeleton[8] if len(skeleton) > 8 else None
    core, mapping, groups, kind = rendering.core, rendering.mapping, rendering.groups, rendering.kind
    prepared, scope, nmap, needed = rendering.prepared, rendering.scope, rendering.nmap, rendering.needed
    h, actual, sites = rendering.h, rendering.actual, prepared.sites
    if numbering_frontier is not None and numbering_frontier.defer and not nmap:
        previous = numbering_frontier.keys.get(parent)
        chemical = plan.chemical_key
        if previous is not None and previous[:len(chemical)] == chemical:
            # Any successful rendering has a prefix at every ring atom with
            # an outgoing branch edge. An earlier branch from another ring
            # atom cannot absorb it: that branch would contain a core atom
            # and fail the ownership check. Edges into suffix atoms were
            # already excluded by preparation. Thus the smallest such locant
            # is the first prefix locant of every successful construction.
            # This says nothing about branch names, annotations, or ties.
            branch_locs = (locant_key(mapping[i]) for i in seq
                           if prepared.branch_edges[i])
            first = min(branch_locs, default=None)
            previous_prefixes = previous[len(chemical)]
            if first is not None and (not previous_prefixes or first > previous_prefixes[0]):
                return ()
    oxygen_suffix_groups = prepared.oxygen_suffix_groups
    modified_oxygen_suffix = prepared.modified_oxygen_suffix
    state, added, hydro, leading = plan.state, plan.added, plan.hydro, plan.leading
    suffix_locs, chemical_key = plan.suffix_locs, plan.chemical_key
    rows = []
    if rendering.branches is _NO_RING_BRANCHES:
        rendering.branches = _ring_render_branches(ctx, seq, nmap, mapping, core, needed, scope, prepared)
    branches = rendering.branches
    if branches is None:
        return ()
    prefixes, embedded, prefix_branch_ownership, bindings = branches
    if kind == 'amide':
        from .amide_isotopes import amide_isotope_partition, ordinary_amide_numbering_proof
        partition = amide_isotope_partition(mol)
        if partition is not None and partition.active:
            from types import SimpleNamespace
            numbering = ordinary_amide_numbering_proof(mol, Chem, partition,
                SimpleNamespace(parent_atoms=seq, atom_locants=tuple(bindings.items())))
            if numbering.status != 'proved':
                return ()
    # These terms depend only on this rendering's immutable prefix strings and
    # locants, not on the hydrogen state. Multiple plans share the rendering.
    if rendering.prefix_terms is None:
        rendering.prefix_terms = (
            tuple(sorted(locant_key(loc) for loc, _ in prefixes)),
            tuple(locant_key(loc) for loc, text in sorted(prefixes,
                key=lambda p: (prefix_alphabetization_key(p[1]), locant_key(p[0])))),
        )
    prefix_key = (*chemical_key, *rendering.prefix_terms)
    # Ordered stereo plans reach this bound only after their branch writers
    # have run, exactly where the exhaustive path would observe native tags.
    # A bound comes from a complete successful record of this same parent;
    # later isotope/stereo criteria cannot reverse a strict prefix loss.
    if numbering_frontier is not None and numbering_frontier.dominated(parent, prefix_key):
        return ()
    hbase = ','.join(loc+'H' for loc in leading)+'-' if leading else ''
    # P-25.2.2.4: benzoheterocycle position locants are compulsory in
    # preferred names, including hydrogenated derivatives.
    vocabulary = {'benzofuran': '1-benzofuran', 'isobenzofuran': '2-benzofuran',
        'benzothiophene': '1-benzothiophene', 'benzimidazole': '1,3-benzimidazole'}.get(parent, parent)
    hydride = hbase+vocabulary
    locant_presentation = None
    if getattr(absolute, 'unsaturation', None) is not None:
        from .unsaturated_ring_parents import render_unsaturated_parent
        _numbered_carbon, numbered_annotations = _ring_core_annotation_flags(
            mol, core, include_annotations=True)
        omit_bond = (not groups and not prefixes and not numbered_annotations
                     and not absolute.hetero_locants
                     and absolute.unsaturation.multiple_bond_locants == (1,)
                     and absolute.unsaturation.edge_locants[0][1] == 2)
        if not (groups or prefixes or leading or added or hydro or absolute.hetero_locants):
            from .locant_presentation import monounsaturated_carbocycle_presentation
            locant_presentation = monounsaturated_carbocycle_presentation(
                mol, tuple(bindings.items()))
            omit_bond = omit_bond or locant_presentation is not None
        hydride = render_unsaturated_parent(
            vocabulary, absolute.unsaturation, omit_single_locant=omit_bond)
    hydro_prefix = ''
    if hydro:
        original_bonds = _ring_bond_states(mol)
        fully = not state.added and not state.pairs and all(
            original_bonds[(seq[a], seq[b]) if seq[a] < seq[b] else (seq[b], seq[a])][0] == 1
            for a, b in edges)
        # P-82.2.3: isotope-labelled H belongs to the modified skeleton.
        # Keep operation locants explicit when that H occupies an added
        # hydro position; these positions are no longer identically
        # modified for the unlocanted P-14.3.4.5 contraction.
        labelled_hydro = any(mol.GetAtomWithIdx(i).GetAtomicNum() == 1
            and mol.GetAtomWithIdx(i).GetIsotope() and loc in hydro for i, loc in bindings.items())
        hydro_prefix = ('' if fully and not labelled_hydro else ','.join(hydro)+'-')+MULT[len(hydro)]+'hydro'
    numbering_prefixes=[p for p in prefixes if not(kind in {'amine','imine','sulfonamide'} and p[0].startswith('N'))]
    # P-82.6.1.1: a required skeletal isotope locant also requires the
    # systematic suffix attachment locant, for every characteristic class.
    numbered_isotopes = _ring_core_annotation_flags(mol, core)[0]
    suffix_owned=frozenset()
    omit = (kind!='peroxol' and len(sites)==1 and not numbering_prefixes and not numbered_isotopes
        and parent.startswith('cyclo') and saturated and all(n==6 for n in numbers)
        and all(degree == 2 for degree in _ring_degrees(len(seq), tuple(edges))))
    suffix_presentation = None
    if not omit and kind in {'acid','amide','nitrile','alcohol','thiol'}:
        from .locant_presentation import sole_suffix_locant_presentation
        omission = sole_suffix_locant_presentation(mol, Chem, core, tuple(mapping.items()), groups,
            structural_locants_required=bool(leading or added or hydro
                or any(char.isdigit() for char in hydride)), prefixes=prefixes)
        omit = omission is not None
        if omit and prefixes:
            suffix_presentation = omission
    if kind in _EXTERNAL_HETEROATOM_ENDINGS:
        suffix = _EXTERNAL_HETEROATOM_ENDINGS[kind]
        # P-65/P-66: append the complete suffix to the hydride.
        # P-14.3.4.4 only omits its position when no isomer is possible.
        unlocanted=omit or parent=='benzene' and len(sites)==1 and not numbering_prefixes and not numbered_isotopes
        ending=hydride+('' if unlocanted else '-'+','.join(suffix_locs)+'-')+multiplied_suffix(len(sites),suffix)
    elif kind in {'acid', 'carboxylate', 'carbamate-anion', 'amide', 'nitrile', 'ester', 'aldehyde'}:
        # P-66.6.1.1.3: a CHO group attached to a ring is an external
        # carbaldehyde suffix, with its locant on the original ring atom.
        # Its carbon and oxygen stay in the mapped suffix ownership above.
        suffix = {'acid': 'carboxylic acid', 'amide': 'carboxamide', 'nitrile': 'carbonitrile',
                  'ester': 'carboxylate', 'carboxylate': 'carboxylate',
                  'carbamate-anion': 'carboxylate', 'aldehyde': 'carbaldehyde'}[kind]
        ending = hydride+('' if omit else '-'+','.join(suffix_locs)+('('+','.join(loc+'H' for loc in added)+')' if added else '')+'-')+multiplied_suffix(len(sites), suffix)
    elif kind == 'imine':
        # P-58.2.3: imine suffixes consume the same annular hydrogen
        # accommodation as oxo suffixes. Omitting its added-H descriptor
        # lets the reader put the surviving double bond on another edge.
        locant_text = ','.join(suffix_locs)+('('+','.join(loc+'H' for loc in added)+')' if added else '')
        ending = attach_suffix(hydride, multiplied_suffix(len(sites), 'imine'),
                               () if omit else (locant_text,))
    elif kind == 'peroxol':
        from .peroxol_components import peroxol_ending
        # P-14 omission is proved by moving the mapped suffix on the
        # original graph; only plain carbon monocycles opt into it here.
        # Fixed heterocycle/fused numbering keeps explicit locants.
        plain_monocycle = (all(n == 6 for n in numbers)
            and len(edges) == len(seq)
            and all(sum(j in edge for edge in edges) == 2 for j in range(len(seq)))
            and ((parent.startswith('cyclo') and saturated) or parent == 'benzene'))
        ending,suffix_owned=peroxol_ending(
            mol,Chem,hydride,groups,suffix_locs,parent_atoms=seq,
            allow_ring_omission=plain_monocycle,
            force_locants=bool(numbering_prefixes or numbered_isotopes))
        if not ending:return ()
    elif kind:
        suffix = 'one' if kind in _CARBONYLS else 'ol' if kind == 'alcohol' else 'thiol' if kind=='thiol' else kind if kind in {'yl', 'ylidene'} else 'amine'
        locant_text = ','.join(suffix_locs)+('('+','.join(loc+'H' for loc in added)+')' if added else '')
        ending = attach_suffix(hydride, multiplied_suffix(len(sites), suffix), () if omit else (locant_text,))
        if kind=='yl' and parent=='benzene':ending='phenyl'
        elif kind=='yl' and omit:ending=parent.removesuffix('ane')+'yl'
        elif (kind == 'yl' and len(sites) == 1 and tuple(suffix_locs) == ('1',)
              and parent.startswith('cyclo') and parent.endswith('ane')
              and saturated and all(n == 6 for n in numbers)
              and all(degree == 2 for degree in _ring_degrees(len(seq), tuple(edges)))
              and not (leading or added or hydro or numbered_isotopes)
              and all(not mol.GetAtomWithIdx(i).GetFormalCharge()
                      and not mol.GetAtomWithIdx(i).GetIsotope()
                      and not any(n.GetAtomicNum() == 1 and n.GetIsotope()
                                  for n in mol.GetAtomWithIdx(i).GetNeighbors()) for i in seq)):
            # P-29.3.3/P-29.4.1: a monovalent saturated monocyclic
            # hydrocarbon substituent is cycloalkyl; its free valence is
            # implicitly C1 even when the component bears other prefixes.
            # Keep explicit suffix locants for skeletal isotope numbering.
            ending = parent.removesuffix('ane') + 'yl'
    else:
        ending = hydride
    if parent=='benzene' and len(sites)==1:
        ending={'acid':'benzoic acid','amide':'benzamide','nitrile':'benzonitrile','ester':'benzoate','carboxylate':'benzoate','aldehyde':'benzaldehyde',
                'alcohol':'phenol','amine':'aniline','yl':'phenyl'}.get(kind,ending)
        if kind=='thiol' and not prefixes and not numbered_isotopes:ending='benzenethiol'
    if kind=='amine' and ending!='aniline':
        from .amine_components import amine_ending
        ending,suffix_owned=amine_ending(
            mol,Chem,hydride,groups,() if omit else (locant_text,),
            one_word_isotope=True)
        if not ending:return ()
    amide_evidence = None
    if kind == 'amide':
        from .amide_isotopes import amide_suffix_isotopes
        location = ('' if omit else '-' + ','.join(suffix_locs)
                    + ('(' + ','.join(loc + 'H' for loc in added) + ')' if added else '') + '-')
        amide = amide_suffix_isotopes(mol, groups, ending, external=True,
            hydride=hydride, location_text=location)
        if amide is None:
            return ()
        ending, suffix_owned, amide_evidence = amide
    if modified_oxygen_suffix:
        from .oxygen_characteristic_suffixes import oxygen_ending
        locant_text = ','.join(suffix_locs)+('('+','.join(loc+'H' for loc in added)+')' if added else '')
        ending, suffix_owned = oxygen_ending(
            mol, Chem, hydride, oxygen_suffix_groups,
            () if omit else (locant_text,),
            suffix='one' if kind in _CARBONYLS else 'ol',
        )
        if not ending:
            return ()
    from .amine_components import unlocanted_nitrogen
    from .imine_components import unlocanted_imine_nitrogen
    one_word_amine_isotope = (kind == 'amine' and any(
        mol.GetAtomWithIdx(g.center).GetIsotope()
        or any(n.GetAtomicNum() == 1 and n.GetIsotope()
               for n in mol.GetAtomWithIdx(g.center).GetNeighbors())
        for g in groups))
    hydrogen_context = frozenset(
        idx for group in groups for idx in group.atoms
        if idx not in bindings and mol.GetAtomWithIdx(idx).GetAtomicNum() > 1)
    if one_word_amine_isotope:
        # P-82.2.5 applies to the complete one-word name. The current
        # insertion precedence between an exocyclic amine descriptor and
        # a detachable ring hydro prefix is not source-closed here, so
        # keep that combination pending. The same applies to an isotope
        # inside a separately rendered prefix until its ordering against
        # the principal-N descriptor is proved.
        if hydro_prefix or any(mol.GetAtomWithIdx(i).GetIsotope() for i in embedded):
            return ()
        from .formatting import format_prefixes, join_prefix_parent
        prefix_text = format_prefixes(prefixes, hierarchical_enclosures=True)
        complete_one_word = join_prefix_parent(prefix_text, ending)
        construction = MappedComponent(
            complete_one_word, tuple(bindings.items()), frozenset(bindings),
            embedded_atoms=frozenset(embedded), systematic_parent=True,
            nitrogen_site_locants=True,
            prefix_branch_ownership=tuple(prefix_branch_ownership),
            isotope_unlocanted_elements=unlocanted_nitrogen(mol,scope),
            isotope_hydrogen_context_atoms=hydrogen_context,
        )
    else:
        # The component's original branch ownership supplies the common
        # locant proof during decoration; avoid a second legacy traversal.
        construction=MappedComponent(ending, tuple(bindings.items()),
            frozenset(bindings), tuple(prefixes), frozenset(embedded), hydro_prefix=hydro_prefix,
            systematic_parent=True,nitrogen_site_locants=kind in {'amine','imine','amide','sulfonamide'},
            prefix_branch_ownership=tuple(prefix_branch_ownership),
            isotope_unlocanted_elements=(unlocanted_nitrogen(mol,scope) if kind=='amine'
                else unlocanted_imine_nitrogen(mol,scope) if kind=='imine' else ()),
            local_isotope_atoms=suffix_owned,
            isotope_hydrogen_context_atoms=hydrogen_context)
        if locant_presentation is not None:
            from dataclasses import replace
            construction = replace(construction, locant_presentation=locant_presentation)
        if suffix_presentation is not None:
            from dataclasses import replace
            construction = replace(construction, substitution_locant_presentation=suffix_presentation)
        if amide_evidence is not None:
            from dataclasses import replace
            construction = replace(construction, amide_isotope_evidence=amide_evidence)
    name = decorate_component(mol, Chem, construction)
    if not name:
        return ()
    if prepared.ester_arms:
        from .mapped_ester_general import ester_organyl_words

        organyls = ester_organyl_words(
            mol, Chem, prepared.ester_arms, tuple(mapping[site] for site in sites))
        if not organyls:
            return ()
        name = organyls + ' ' + name
    elif prepared.ester_owner_tokens:
        owner = prepared.ester_owner_tokens[0]
        multiplier = MULT[len(prepared.ester_owner_tokens)] if len(prepared.ester_owner_tokens)>1 else ''
        name = multiplier+owner+' '+name
    from .parent_annotations import isotope_numbering_key, stereo_numbering
    isotope_mapping = bindings
    if kind == 'amide':
        from .amide_isotopes import amide_parent_annotation_mapping
        physical = amide_parent_annotation_mapping(mol, core, bindings)
        if physical is not None:
            isotope_mapping = physical
    isotope = isotope_numbering_key(mol, isotope_mapping, include_substituent_groups=True)
    stereo = stereo_numbering(mol, bindings)
    if isotope is None or stereo is None:
        return ()
    key = (*prefix_key, isotope, stereo)
    method = skeleton[7] if len(skeleton)>7 else 'Hantzsch-Widman' if len(mol.GetRingInfo().AtomRings()) == 1 else 'retained-fused'
    fusion = None
    if method in {'component-fused', 'component-fused-benzofuran-pyridine', 'component-fused-benzocarbazole',
                  'component-fused-three-face-azine', 'component-fused-general',
                  'component-fused-benzo-imidazothiazole', 'component-fused-peri-ketone'}:
        from .fusion_components import fusion_construction
        fusion = fusion_construction(parent)
    issues=()
    if method == 'component-fused-benzofuran-pyridine':
        from .naming_result import SelectionLimitation
        issues=(SelectionLimitation('ring-construction','P-25.3.5/P-44/P-45',
            'The bounded benzoheterocycle component frontier is graph-proved; '
            'complete competing parent and method selection remains unresolved.'),)
    if method == 'component-fused-three-face-azine':
        from .naming_result import SelectionLimitation
        issues=(SelectionLimitation('ring-construction','P-25.3.4/P-44/P-45',
            'The bounded three-face component frontier is graph-proved; '
            'complete competing parent and method selection remains unresolved.'),)
    if method in {'component-fused-general', 'component-fused-benzocarbazole'}:
        from .naming_result import SelectionLimitation
        issues=(SelectionLimitation('ring-construction','P-25/P-44/P-45/P-51',
            'The indexed two-component construction and numbered atom graph are mapped; '
            'P-25 orientation and complete competing parent/method selection remain unresolved.'),)
    if method in {'skeletal-replacement-localized-bicyclic',
                  'skeletal-replacement-explicit-bicyclic',
                  'skeletal-replacement-localized-tricyclic'}:
        from .naming_result import SelectionLimitation
        issues=(SelectionLimitation('ring-construction','P-25.7.1.2/P-51',
            'Explicit bond-locanted replacement name preserves the annular pi graph; preferred fusion spelling remains unresolved.'),)
    witnesses = _hydrogen_witness_locants(tuple(locs), state.pairs, state.paths,
        state.hydro_paths, state.reference_pi, state.accommodated_pi)
    rows.append(RingStateRecord(name, parent, method,
        tuple(seq), tuple(bindings.items())+tuple((g.center,mol.GetAtomWithIdx(g.center).GetSymbol()+mapping[site]) for g,site in zip(groups,sites) if g.center not in bindings), frozenset(g.center for g in groups), leading, added, hydro, key,
        tuple(h), tuple(map(int, actual)),
        *witnesses,tuple(prefixes),suffix_locs,
        construction=construction,
        unsaturation_state=getattr(absolute, 'unsaturation', None),
        fusion_construction=fusion,preference_issues=issues,hydride=hydride))
    if numbering_frontier is not None:
        numbering_frontier.accept(rows[-1])
    return rows


def select_ring_state(ctx, *, skeleton_source=None):
    mol, Chem = ctx.mol, ctx.Chem
    if not mol.GetRingInfo().NumRings() or mol.GetNumHeavyAtoms() > 64:
        return None
    from .peroxol_components import recognize_peroxols
    peroxol_chalcogens = {i for group in recognize_peroxols(mol) for i in group.atoms
        if mol.GetAtomWithIdx(i).GetAtomicNum() in {16, 34, 52}}
    rings = mol.GetRingInfo().AtomRings()
    chalcogen_core = (frozenset(rings[0]) if len(rings) == 1
                      and is_supported_hw_chalcogen_core(mol, rings[0]) else frozenset())
    if any((a.GetAtomicNum() not in {1, 6, 7, 8, 9, 16, 17, 35, 53}
            and not (a.GetAtomicNum() in {34, 52} and a.GetIdx() in chalcogen_core)
            and a.GetIdx() not in peroxol_chalcogens) or a.GetFormalCharge()
           or a.GetNumRadicalElectrons() or (a.GetAtomicNum() == 16 and not a.IsInRing()
                                           and a.GetIdx() not in peroxol_chalcogens)
           or a.GetAtomicNum() == 7 and int(a.GetChiralTag()) for a in iter_atoms(mol)):
        return None
    if mol.GetStereoGroups() or any(b.GetBondTypeAsDouble() > 2 and b.IsInRing() for b in iter_bonds(mol)):
        return None
    from .stereochemistry import assigned_atom_stereochemistry
    assigned = dict(assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True))
    if any(int(a.GetChiralTag()) and a.GetIdx() not in assigned for a in iter_atoms(mol)):
        return None
    inventory = _inventory(mol, Chem)
    if not inventory.supported:
        # An imine-like N=C bond entirely inside a fused parent hydride is
        # skeletal unsaturation, not an exocyclic imine suffix.  Admit an
        # empty characteristic-group inventory only when the entire heavy
        # graph is annular; all valence, pi matching and H-state checks still
        # run on the original graph below.
        if (len(rings) < 2 or any(a.GetAtomicNum() > 1 and not a.IsInRing()
                                  for a in iter_atoms(mol))):
            return None
        from types import SimpleNamespace
        inventory = SimpleNamespace(groups=(), principal=(), supported=True)
    rows = []
    numbering_frontier = _RingNumberingFrontier.for_context(ctx)
    for skeleton in (_skeletons(ctx) if skeleton_source is None else skeleton_source(mol)):
        parent, seq, locs, numbers, edges, saturated, fixed_key = skeleton[:7]
        if any(n not in _HW_VALENCE for n in numbers):
            continue
        for reference in _skeleton_references(skeleton):
            rows.extend(_render(ctx, skeleton, reference, inventory,
                                numbering_frontier=numbering_frontier))
    if numbering_frontier is not None:
        rows.extend(numbering_frontier.finish())
    if not rows and len(rings) == 2 and skeleton_source is None:
        from .localized_bicyclic_states import skeletons as localized_skeletons
        fallback_frontier = _RingNumberingFrontier.for_context(ctx)
        for skeleton in localized_skeletons(mol, Chem):
            rows.extend(_render(ctx, skeleton, None, inventory,
                                numbering_frontier=fallback_frontier))
        if fallback_frontier is not None:
            rows.extend(fallback_frontier.finish())
    if not rows or len({r.parent for r in rows}) != 1:
        return None  # Different parent vocabularies need their own P-52 evidence.
    best = min(r.numbering_key for r in rows)
    finalists = [r for r in rows if r.numbering_key == best]
    if len({r.name for r in finalists}) != 1:
        return None  # An unproved chemical tie must not become a spelling vote.
    return RingStateSelection(finalists[0], tuple(rows))
