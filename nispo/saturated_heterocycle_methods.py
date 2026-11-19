"""P-54.4.2 saturated heteromonocyclic parent construction."""

from collections import Counter

from .naming_methods import MethodDecision


# P-22.2.2 Table 2.4 order and ordinary bonding numbers.
_ELEMENTS = ((8, 'oxa', 2), (16, 'thia', 2), (34, 'selena', 2), (52, 'tellura', 2),
             (7, 'aza', 3), (15, 'phospha', 3), (33, 'arsa', 3), (51, 'stiba', 3),
             (83, 'bisma', 3), (14, 'sila', 4), (32, 'germa', 4), (50, 'stanna', 4),
             (82, 'plumba', 4), (5, 'bora', 3), (13, 'alumina', 3), (31, 'galla', 3),
             (49, 'inda', 3), (81, 'thalla', 3))
_EXTENDED = frozenset(n for n, _name, _val in _ELEMENTS[5:])
_HALOGENS = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}


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


def _parent(numbers):
    from .constants import MULT
    from .ring_state_parent import _hw_name

    count = Counter(numbers)
    size = len(numbers)
    if not any(n in _EXTENDED for n in numbers):
        return _hw_name(numbers, saturated=True)
    if size == 3:
        stem = 'iridine' if count[7] else 'irane'
    elif size == 4:
        stem = 'etidine' if count[7] else 'etane'
    elif size == 5:
        stem = 'olidine' if count[7] else 'olane'
    elif size == 6:
        stem = 'inane' if any(n in count for n in (7, 14, 32, 50, 82, 15, 33, 51, 5, 13, 31, 49, 81)) else 'ane'
    else:
        stem = {7: 'epane', 8: 'ocane', 9: 'onane', 10: 'ecane'}.get(size)
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


def saturated_heteromonocycle_decision(mol, Chem):
    """Name a neutral simple saturated ring, with methyl/halogen prefixes.

    Existing C/N/O/S/Se/Te mixed-ring producers remain authoritative; this
    adds standard main-group elements and homogeneous N rings. Whole-graph
    coverage excludes senior characteristic groups and large parent choices.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetRingInfo().NumRings() != 1
            or any(a.GetAtomicNum() not in {6, *_HALOGENS, *(x[0] for x in _ELEMENTS)}
                   or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    from .mapped_ring_parents import ring_systems
    from .formatting import format_prefixes, join_prefix_parent, prefix_alphabetization_key

    rings = ring_systems(mol)
    if len(rings) != 1 or not 3 <= len(rings[0]) <= 10:
        return None
    core = rings[0]
    numbers = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in core]
    if not (any(n in _EXTENDED for n in numbers) or all(n == 7 for n in numbers)):
        return None
    valences = {6: 4, **{n: v for n, _name, v in _ELEMENTS}}
    if any(mol.GetAtomWithIdx(i).GetTotalValence() != valences.get(mol.GetAtomWithIdx(i).GetAtomicNum())
           for i in core):
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
    adjacency = {}
    for idx in core:
        adjacency[idx] = [a.GetIdx() for a in mol.GetAtomWithIdx(idx).GetNeighbors() if a.GetIdx() in core]
        if len(adjacency[idx]) != 2 or any(
                graph.GetBondBetweenAtoms(idx, j).GetBondTypeAsDouble() != 1 for j in adjacency[idx]):
            return None
    branches = []
    for idx in core:
        for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
            root = atom.GetIdx()
            if root in core:
                continue
            if (atom.GetDegree() != 1 or mol.GetBondBetweenAtoms(idx, root).GetBondTypeAsDouble() != 1):
                return None
            if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs(includeNeighbors=True) == 3:
                name = 'methyl'
            elif atom.GetAtomicNum() in _HALOGENS:
                # A halogen directly on a heteroatom can form a senior acid
                # halide or acid amide; this ring-hydride proof cannot rank it.
                if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6:
                    return None
                name = _HALOGENS[atom.GetAtomicNum()]
            else:
                return None
            branches.append((idx, root, name))
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    if core | {root for _host, root, _name in branches} != heavy:
        return None
    order = {element: i for i, (element, _term, _valence) in enumerate(_ELEMENTS)}
    senior = min((n for n in numbers if n != 6), key=order.__getitem__)
    rows = []
    for start in core:
        if mol.GetAtomWithIdx(start).GetAtomicNum() != senior:
            continue
        for first in adjacency[start]:
            seq, prev, current = [start], start, first
            while current != start and len(seq) <= len(core):
                seq.append(current)
                following = [i for i in adjacency[current] if i != prev]
                if len(following) != 1:
                    break
                prev, current = current, following[0]
            if current != start or len(seq) != len(core) or frozenset(seq) != core:
                continue
            elements = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in seq)
            parent = _parent(elements)
            if not parent:
                continue
            mapping = {idx: str(i) for i, idx in enumerate(seq, 1)}
            prefixes = [(mapping[host], text) for host, _root, text in branches]
            omit_single = len(prefixes) == 1 and len(set(elements)) == 1
            rendered = [('', prefixes[0][1])] if omit_single else prefixes
            name = join_prefix_parent(format_prefixes(rendered, omit_locants=omit_single), parent) if rendered else parent
            by_element = tuple(tuple(i for i, n in enumerate(elements, 1) if n == element)
                               for element, _term, _val in _ELEMENTS if element in elements)
            texts = sorted({text for _loc, text in prefixes}, key=prefix_alphabetization_key)
            score = (tuple(i for i, n in enumerate(elements, 1) if n != 6), by_element,
                     tuple(sorted(int(loc) for loc, _text in prefixes)),
                     tuple(tuple(sorted(int(loc) for loc, value in prefixes if value == text))
                           for text in texts))
            rows.append((score, name, mapping))
    if not rows:
        return None
    best_score = min(row[0] for row in rows)
    winners = [row for row in rows if row[0] == best_score]
    if len({row[1] for row in winners}) != 1:
        return None
    _score, name, mapping = winners[0]
    roles = [('ring-parent', core, tuple(mapping.items()))]
    roles.extend(('substituent', frozenset((root,)), ()) for _host, root, _text in branches)
    return MethodDecision(name, 'Hantzsch-Widman', frozenset(), heavy,
                          'P-54.4.2/P-22.2.2.1/P-14.4', tuple(mapping.items()),
                          component_roles=tuple(roles))
