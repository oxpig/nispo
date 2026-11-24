"""P-31.2.3.2 and P-54.4.2 saturated heteromonocyclic parent construction."""

from collections import Counter
from functools import lru_cache

from .naming_methods import MethodDecision


# P-22.2.2 Table 2.4 order and ordinary bonding numbers.
_ELEMENTS = ((8, 'oxa', 2), (16, 'thia', 2), (34, 'selena', 2), (52, 'tellura', 2),
             (7, 'aza', 3), (15, 'phospha', 3), (33, 'arsa', 3), (51, 'stiba', 3),
             (83, 'bisma', 3), (14, 'sila', 4), (32, 'germa', 4), (50, 'stanna', 4),
             (82, 'plumba', 4), (5, 'bora', 3), (13, 'aluma', 3), (31, 'galla', 3),
             (49, 'indiga', 3), (81, 'thalla', 3))
_EXTENDED = frozenset(n for n, _name, _val in _ELEMENTS[5:])
_VALENCES = {6: 4, **{n: val for n, _name, val in _ELEMENTS}}


def _element_pattern_is_unique(numbers):
    """Prove that element positions have only one dihedral arrangement."""
    from .hantzsch_widman_locants import _dihedral_key, _distinct_multiset_sequences

    size = len(numbers)
    # All equal and one heteroatom are the common large-ring cases. For the
    # remaining cases stop as soon as two distinct element arrangements exist.
    if len(set(numbers)) == 1 or sum(n != 6 for n in numbers) == 1:
        return True
    seen = set()
    for sequence in _distinct_multiset_sequences(Counter(numbers), size):
        seen.add(_dihedral_key(sequence))
        if len(seen) > 1:
            return False
    return len(seen) == 1


@lru_cache(maxsize=512)
def _parent(numbers, saturated=True):
    from .constants import MULT
    from .ring_state_parent import _hw_name

    count = Counter(numbers)
    size = len(numbers)
    if not any(n in _EXTENDED for n in numbers):
        return _hw_name(numbers, saturated)
    if size == 3:
        stem = ('iridine' if count[7] else 'irane') if saturated else 'irene'
    elif size == 4:
        stem = ('etidine' if count[7] else 'etane') if saturated else 'ete'
    elif size == 5:
        stem = ('olidine' if count[7] else 'olane') if saturated else 'ole'
    elif size == 6:
        # Table 2.5 assigns the stem by the last cited heteroatom.
        least_senior = next(n for n, _, _ in reversed(_ELEMENTS) if n in count)
        if saturated:
            stem = 'ane' if least_senior in {8, 16, 34, 52, 83} else 'inane'
        else:
            stem = 'inine' if least_senior in {15, 33, 51, 5, 13, 31, 49, 81} else 'ine'
    else:
        stems = {7: ('epine', 'epane'), 8: ('ocine', 'ocane'),
                 9: ('onine', 'onane'), 10: ('ecine', 'ecane')}
        stem = stems[size][int(saturated)]
    if not stem:
        return ''
    words, locants = [], []
    for element, term, _valence in _ELEMENTS:
        positions = [i for i, number in enumerate(numbers, 1) if number == element]
        if not positions:
            continue
        multiplier = MULT.get(len(positions), '') if len(positions) > 1 else ''
        if len(positions) > 1 and not multiplier:
            return ''
        if multiplier.endswith('a') and term.startswith('a'):
            multiplier = multiplier[:-1]
        words.append(multiplier + term)
        locants.extend(positions)
    text = ''.join(word[:-1] if word.endswith('a') and nxt[0] in 'aeiou' else word
                   for word, nxt in zip(words, words[1:] + [stem])) + stem
    if len(locants) > 1 and not _element_pattern_is_unique(numbers):
        text = ','.join(map(str, locants)) + '-' + text
    return text


def _hydrogenated_hw_ring(graph, ring, element):
    """Number a homogeneous Hantzsch-Widman cycle against its mancude parent."""
    from .constants import MULT

    size = len(ring)
    if _VALENCES[element] < 3:
        return ''
    parent = _parent((element,) * size, saturated=False)
    if not parent:
        return ''
    # With an odd number of sites, locant 1 bears the indicated hydrogen.
    # Deleting pairs from this reference matching must recover every actual
    # double bond under one and the same numbering of the ring.
    reference = set(range(size % 2, size - 1, 2))
    choices = []
    for order in (ring, ring[::-1]):
        for start in range(size):
            seq = order[start:] + order[:start]
            actual = {j for j, (a, b) in enumerate(zip(seq, seq[1:] + seq[:1]))
                      if graph.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() == 2}
            if actual <= reference:
                choices.append(tuple(loc for edge in sorted(reference - actual)
                                     for loc in (edge + 1, edge + 2)))
    if not choices:
        return ''
    hydro = min(choices)
    if size % 2:
        parent = '1H-' + parent
    if hydro:
        modifier = ','.join(map(str, hydro)) + '-' + MULT[len(hydro)] + 'hydro'
        parent = modifier + ('-' if size % 2 else '') + parent
    return parent


def homogeneous_ring_parent(mol, Chem, ring, element):
    """P-22.2.5, P-22.2.3 and P-31.1.3.1 on a bare homogeneous cycle."""
    size = len(ring)
    if (element not in _VALENCES or element in {5, 6} or size != mol.GetNumAtoms()
            or size != mol.GetNumBonds() or mol.GetStereoGroups()):
        return None
    if any(a.GetAtomicNum() != element or a.GetDegree() != 2 or a.GetTotalValence() != _VALENCES[element]
           or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
           or a.GetChiralTag() or a.HasQuery() for a in mol.GetAtoms()):
        return None
    graph = Chem.Mol(mol)
    try:
        Chem.Kekulize(graph, clearAromaticFlags=True)
    except (ValueError, RuntimeError):
        return None
    orders = [b.GetBondTypeAsDouble() for b in graph.GetBonds()]
    if any(b.HasQuery() for b in mol.GetBonds()) or any(order not in (1, 2, 3) for order in orders):
        return None
    multiple = [order for order in orders if order > 1]
    if size <= 10:
        if multiple:
            if 3 in multiple or any(b.GetStereo() for b in mol.GetBonds()):
                return None
            name = _hydrogenated_hw_ring(graph, ring, element)
            rule = 'P-31.2.3.1/P-22.2.2.1.4'
        else:
            name = _parent((element,) * size)
            rule = 'P-31.2.3.2/P-22.2.5'
    else:
        from .number_terms import basic_number_term
        from .constants import ROOTS

        term, root = basic_number_term(size), ROOTS.get(size)
        if not term or not root or len(multiple) > 1:
            return None
        prefix = next(word for n, word, _ in _ELEMENTS if n == element)
        ending = {(): 'ane', (2,): 'ene', (3,): 'yne'}[tuple(multiple)]
        name = term + prefix + 'cyclo' + root + ending
        rule = 'P-22.2.3/P-22.2.5/P-31.1.3.1/P-14.3.4.2(d)'
    if any(b.GetStereo() for b in mol.GetBonds()):
        from .stereochemistry import assigned_bond_stereochemistry

        descriptors = assigned_bond_stereochemistry(mol, Chem)
        if multiple != [2] or len(descriptors) != 1:
            return None
        name = '(' + descriptors[0][2] + ')-' + name
    return MethodDecision(name, 'homogeneous-ring', frozenset(), frozenset(ring), rule) if name else None


def saturated_heteromonocycle_decision(mol, Chem):
    """Number the ring first, then its suffixes, prefixes and configurations."""
    rings = mol.GetRingInfo().AtomRings()
    # A separate carbocyclic substituent can itself be cyclic. A fused or
    # spiro ring shares atoms with the candidate and is a different parent.
    heterorings = [r for r in rings if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in r)]
    if len(heterorings) != 1 or not 3 <= len(heterorings[0]) <= 10 or mol.GetStereoGroups():
        return None
    from .retained_hydrocarbon_derivatives import ring_pendants, ring_substitution

    ring = heterorings[0]
    core = frozenset(ring)
    if any(core.intersection(other) for other in rings if other != ring):
        return None
    numbers = {i: mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring}
    bare = len(ring) == mol.GetNumAtoms()
    branch_labels = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsotope() and a.GetIdx() not in core}
    if (not set(numbers.values()) <= _VALENCES.keys() or set(numbers.values()) == {5}
            or not (bare or branch_labels or _EXTENDED.intersection(numbers.values())
                    or set(numbers.values()) == {7})):
        return None
    if (any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.HasQuery()
            or a.GetChiralTag() and a.GetIdx() not in core for a in mol.GetAtoms())
            or any(b.GetStereo() or b.HasQuery() for b in mol.GetBonds())):
        return None
    if any(mol.GetAtomWithIdx(i).GetTotalValence() != _VALENCES[n] for i, n in numbers.items()):
        return None
    # RDKit aromaticity can mark even the fully hydrogenated N5 cycle as
    # aromatic. Kekulization recovers its actual single-bond pattern.
    graph = mol
    if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in core):
        graph = Chem.Mol(mol)
        try:
            Chem.Kekulize(graph, clearAromaticFlags=True)
        except (ValueError, RuntimeError):
            return None
    if any(graph.GetBondBetweenAtoms(a, b).GetBondType() != Chem.BondType.SINGLE
           for a, b in zip(ring, ring[1:] + ring[:1])):
        return None
    parts = ring_pendants(mol, Chem, core, branch_isotopes=branch_labels)
    if parts is None or any(numbers[p.host] != 6 and
                            (p.kind or mol.GetAtomWithIdx(p.root).GetAtomicNum() not in {6, 9, 17, 35, 53})
                            for p in parts):
        return None  # Heteroatom-bound functions may require other senior classes.
    senior = next(n for n, _, _ in _ELEMENTS if n in numbers.values())
    annotated = bool(branch_labels) or any(
        mol.GetAtomWithIdx(i).GetChiralTag() or mol.GetAtomWithIdx(i).GetIsotope() for i in core)
    scope = frozenset(range(mol.GetNumAtoms()))
    numberings = []
    for order in (ring, ring[::-1]):
        for j, start in enumerate(order):
            if numbers[start] != senior:
                continue
            seq = order[j:] + order[:j]
            elements = tuple(numbers[i] for i in seq)
            by_element = tuple(tuple(i for i, n in enumerate(elements, 1) if n == element)
                               for element, _term, _val in _ELEMENTS if element in elements)
            element_key = tuple(i for i, n in enumerate(elements, 1) if n != 6), by_element
            numberings.append((element_key, elements, seq))
    best_elements = min(key for key, _, _ in numberings)
    rows = []
    for key, elements, seq in numberings:
        if key != best_elements:
            continue
        parent = _parent(elements)
        if not parent:
            return None
        mapping = {idx: str(i) for i, idx in enumerate(seq, 1)}
        if bare and not annotated:
            rows.append(((), parent, mapping, frozenset()))
            break
        row = ring_substitution(parent, mapping, parts, omit_single=len(set(elements)) == 1 and not annotated,
                                annotated_graph=mol if annotated else None, branch_isotopes=bool(branch_labels))
        if row is not None:
            substitution_key, name, centers = row
            rows.append((substitution_key, name, mapping, centers))
    if not rows:
        return None
    best_score = min(row[0] for row in rows)
    winners = [row for row in rows if row[0] == best_score]
    if len({row[1] for row in winners}) != 1:
        return None
    _score, name, mapping, centers = winners[0]
    roles = [('ring-parent', core, tuple(mapping.items()))]
    roles.extend(('pendant', p.atoms, ((p.root, p.host),)) for p in parts)
    return MethodDecision(name, 'Hantzsch-Widman', centers, scope,
                          'P-31.2.3.2/P-54.4.2/P-22.2.2.1/P-14.4', tuple(mapping.items()),
                          component_roles=tuple(roles))
