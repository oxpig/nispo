"""Full-mode, graph-closed constructions for additional Blue Book rules.

Producers decline competing functions or annotations outside their proofs.
They operate on source graphs, not reference-name or evaluation records.
"""
from collections import Counter

from .molecule_iteration import iter_atoms
from .full_bluebook_frontier import _bond, _clean, _decision, _sanitize


def _imino_proxy_isotopes(neutral):
    """Give distinct N nuclides proxy markers in descending seniority order."""
    masses = sorted({atom.GetIsotope() for atom in neutral if atom.GetIsotope()}, reverse=True)
    if len(masses) <= 1:
        return {mass: mass for mass in masses}
    # The proxy writer alphabetizes its isotope-bearing prefixes by their
    # spelling. Equal-width markers keep that order aligned with the higher
    # mass priority of the original N nuclides (P-82.5.2).
    return {mass: 200 + index for index, mass in enumerate(masses)}


def _imino_from_fluoro_proxy(name, isotope_markers):
    """Transfer each fluorine proxy's own isotope and locant to its imino."""
    import re
    original = {marker: mass for mass, marker in isotope_markers.items()}
    def replacement(match):
        mass = original.get(int(match[1]))
        return f'[({mass}N)imino]' if mass is not None else ''
    name = re.sub(r'\[\((\d+)F\)fluoro\]', replacement, name)
    name = re.sub(r'\((\d+)F\)fluoro', replacement, name)
    return name.replace('fluoro', 'imino')


def _imino_isotope_counts_match(mol, name):
    """Count multiplied, independently labelled imino prefix atoms."""
    import re
    from .constants import MULT
    multipliers = {value: key for key, value in MULT.items()}
    terms = '|'.join(sorted(map(re.escape, multipliers), key=len, reverse=True))
    adjustments = Counter()
    for match in re.finditer(rf'(?P<mult>{terms})?\[\((?P<mass>\d+)N\)imino\]', name):
        adjustments[int(match['mass'])] += multipliers.get(match['mult'], 1)-1
    isotopes = Counter((a.GetIsotope(), a.GetSymbol()) for a in mol.GetAtoms() if a.GetIsotope())
    for (mass, symbol), count in isotopes.items():
        occurrences = re.findall(rf'(?<!\d){mass}{re.escape(symbol)}(\d*)', name)
        actual = sum(int(value or '1') for value in occurrences)
        if symbol == 'N':
            actual += adjustments[mass]
        if actual != count:
            return False
    return True


def _rank_isotopic_imino_numbering(mol, name, charged, neutral, Chem):
    """Resolve isotope ties on a chemically symmetric acyclic carbon parent."""
    if not any(atom.GetIsotope() for atom in mol.GetAtoms()):
        return name
    hosts = {next(n.GetIdx() for n in atom.GetNeighbors()
                  if _bond(mol, atom.GetIdx(), n.GetIdx()) == 2) for atom in (*charged, *neutral)}
    chain = {next(iter(hosts))}
    while True:
        expanded = chain | {n.GetIdx() for idx in chain for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                            if n.GetAtomicNum() == 6}
        if expanded == chain:
            break
        chain = expanded
    if not hosts <= chain:
        return None
    adjacency = {idx: {n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() in chain}
                 for idx in chain}
    if any(len(row) > 2 for row in adjacency.values()):
        return None
    ends = [idx for idx, row in adjacency.items() if len(row) == 1]
    if len(chain) < 2:
        return name
    if len(ends) != 2:
        return None
    sequence, previous, current = [], None, ends[0]
    while current is not None:
        sequence.append(current)
        choices = adjacency[current] - ({previous} if previous is not None else set())
        previous, current = current, next(iter(choices)) if choices else None
    if set(sequence) != chain:
        return None
    isotope_free = Chem.Mol(mol)
    for atom in isotope_free.GetAtoms():
        atom.SetIsotope(0)
    matches = isotope_free.GetSubstructMatches(isotope_free, uniquify=False, maxMatches=4096)
    if len(matches) == 4096:
        return None
    if not any(match[sequence[0]] == sequence[-1] for match in matches):
        return name
    from .parent_annotations import isotope_numbering_key
    maps = [{idx: str(i+1) for i, idx in enumerate(order)}
            for order in (sequence, tuple(reversed(sequence)))]
    keys = [isotope_numbering_key(mol, mapping, include_substituent_groups=True) for mapping in maps]
    if any(key is None for key in keys):
        return None
    if keys[0] == keys[1]:
        return name
    neutral_ids = {atom.GetIdx() for atom in neutral}
    if any(atom.GetIsotope() and atom.GetIdx() not in chain | neutral_ids for atom in mol.GetAtoms()):
        return None  # Other isotope insertion boundaries need their own mapping.
    import re
    from .constants import MULT
    terms = '|'.join(sorted(map(re.escape, MULT.values()), key=len, reverse=True))
    prefix = re.compile(rf'(?<!\d)(?P<loc>\d+(?:,\d+)*)-(?P<mult>{terms})?'
                        r'(?:(?:\[\((?P<mass>\d+)N\)imino\])|(?P<plain>imino))')
    rows = list(prefix.finditer(name))
    if not rows:
        return None
    written_n = Counter((int(row['mass']), locant)
                        for row in rows if row['mass']
                        for locant in row['loc'].split(','))
    carbon = re.compile(r'\((?P<loc>\d+(?:,\d+)*)-(?P<mass>\d+)C(?P<count>\d*)\)')
    carbon_rows = list(carbon.finditer(name))
    written_c = Counter((int(row['mass']), locant)
                        for row in carbon_rows for locant in row['loc'].split(','))
    def source_labels(mapping):
        n = Counter((atom.GetIsotope(), mapping[next(x.GetIdx() for x in atom.GetNeighbors()
                                                     if _bond(mol, atom.GetIdx(), x.GetIdx()) == 2)])
                    for atom in neutral if atom.GetIsotope())
        c = Counter((mol.GetAtomWithIdx(idx).GetIsotope(), mapping[idx]) for idx in chain
                    if mol.GetAtomWithIdx(idx).GetIsotope())
        return n, c
    orientations = [index for index, mapping in enumerate(maps)
                    if source_labels(mapping) == (written_n, written_c)]
    if not orientations:
        return None
    if keys[orientations[0]] == min(keys):
        return name
    if any(name[left.end():right.start()] != '-' for left, right in zip(rows, rows[1:])):
        return None
    size = len(sequence)
    def reverse_locants(text):
        return ','.join(map(str, sorted(size+1-int(value) for value in text.split(','))))
    replacement = '-'.join(reverse_locants(row['loc']) + '-' + row.group(0).split('-', 1)[1]
                           for row in rows)
    name = name[:rows[0].start()] + replacement + name[rows[-1].end():]
    return carbon.sub(lambda row: '(' + reverse_locants(row['loc']) + '-' + row['mass']
                      + 'C' + row['count'] + ')', name)


def _stereo_unchanged_after_removal(mol, probe, removed, Chem):
    """Verify whole-graph CIP descriptors survive a mapped probe operation."""
    from .stereochemistry import assigned_atom_stereochemistry, assigned_bond_stereochemistry

    retained = [i for i in range(mol.GetNumAtoms()) if i not in removed]
    if len(retained) != probe.GetNumAtoms():
        return False
    old_to_new = {old: new for new, old in enumerate(retained)}
    original_atoms = dict(assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True))
    probe_atoms = dict(assigned_atom_stereochemistry(probe, Chem, atom_order='idx', include_pseudo=True))
    if any(int(atom.GetChiralTag()) and atom.GetIdx() not in original_atoms for atom in mol.GetAtoms()):
        return False
    if {old_to_new[i]: value for i, value in original_atoms.items() if i in old_to_new} != probe_atoms:
        return False
    assigned_bonds = assigned_bond_stereochemistry(mol, Chem)
    specified = {frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) for bond in mol.GetBonds()
                 if bond.GetStereo() in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                                         Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}}
    if specified != {frozenset((a, b)) for a, b, _desc in assigned_bonds}:
        return False
    original_bonds = {(frozenset((old_to_new[a], old_to_new[b])), desc)
                      for a, b, desc in assigned_bonds
                      if a in old_to_new and b in old_to_new}
    probe_bonds = {(frozenset((a, b)), desc) for a, b, desc in assigned_bond_stereochemistry(probe, Chem)}
    return original_bonds == probe_bonds


def _stereo_descriptors_preserved(mol, name, Chem):
    """Require the proxy-rendered name to own every specified stereo unit."""
    import re
    from .stereochemistry import assigned_atom_stereochemistry, assigned_bond_stereochemistry
    expected = Counter(label for _index, label in assigned_atom_stereochemistry(
        mol, Chem, atom_order='idx', include_pseudo=True))
    expected.update(label for _a, _b, label in assigned_bond_stereochemistry(mol, Chem))
    supplied = Counter(re.findall(r'(?<![A-Za-z])(?:\d+[a-z]?)?([RSEZrs])(?=[,)])', name))
    return bool(expected) and expected == supplied


def _remap_imino_parent_stereo(mol, name, charged, neutral, Chem):
    """Reverse proved parent-bond locants while retaining branch descriptors."""
    hosts = {next(neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                  if _bond(mol, atom.GetIdx(), neighbor.GetIdx()) == 2)
             for atom in (*charged, *neutral)}
    chain = set(hosts)
    while True:
        expanded = chain | {neighbor.GetIdx() for idx in chain
                            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                            if neighbor.GetAtomicNum() == 6}
        if expanded == chain:
            break
        chain = expanded
    if any(int(atom.GetChiralTag()) and atom.GetIdx() in chain for atom in mol.GetAtoms()):
        return None  # Parent-atom stereodescriptors need a separate mapping.
    if any(int(bond.GetStereo())
           and bool({bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & chain)
           and not {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} <= chain
           for bond in mol.GetBonds()):
        return None  # Exocyclic stereogenic bonds have a different locant rule.
    from .stereochemistry import assigned_bond_stereochemistry
    parent_bonds = [(a, b, desc) for a, b, desc in assigned_bond_stereochemistry(mol, Chem)
                    if {a, b} <= chain]
    if not parent_bonds:
        return name  # All specified stereo is local to attached groups.
    import re
    match = re.match(r'^\(((?:\d+[EZ],)*\d+[EZ])\)-', name)
    if not match:
        return None
    terms = [re.fullmatch(r'(\d+)([EZ])', term) for term in match[1].split(',')]
    if (len(terms) != len(parent_bonds)
            or Counter(term[2] for term in terms) != Counter(desc for _a, _b, desc in parent_bonds)):
        return None
    remapped = sorted((len(chain)-int(term[1]), term[2]) for term in terms)
    if any(not 1 <= position < len(chain) for position, _desc in remapped):
        return None
    return '('+','.join(str(position)+desc for position, desc in remapped)+')-'+name[match.end():]


def added_hydrogen_ring_radical(mol, Chem):
    """Recover an aromatic parent by the P-71.2.6 added-H operation."""
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 16, 17, 35, 53} or a.GetFormalCharge()
                   or a.GetIsotope() or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) for b in mol.GetBonds())):
        return None
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    radical = [a for a in mol.GetAtoms() if a.GetNumRadicalElectrons()]
    if (not 1 <= len(radical) <= 2 or any(a.GetIdx() not in ring
            or a.GetNumRadicalElectrons() not in {1, 2} for a in radical)):
        return None
    if len(radical) == 2 and sorted(a.GetNumRadicalElectrons() for a in radical) != [1, 2]:
        return None
    main = max(radical, key=lambda a: a.GetNumRadicalElectrons())
    if main.GetAtomicNum() not in {6, 7}:
        return None
    possible = [a for a in main.GetNeighbors() if a.IsInRing() and a.GetAtomicNum() == 6
                and a.GetTotalNumHs(includeNeighbors=True) > 0
                and _bond(mol, main.GetIdx(), a.GetIdx()) == 1]
    if not possible:
        return None
    from .component_mapping import mapped_branch
    from .phosphorus_ester_methods import _carbon_component
    halogens = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    branches, owned = [], set(ring)
    for host in ring:
        for atom in mol.GetAtomWithIdx(host).GetNeighbors():
            root = atom.GetIdx()
            if root in ring:
                continue
            if root in owned or _bond(mol, host, root) != 1:
                return None
            if atom.GetAtomicNum() in halogens and atom.GetDegree() == 1:
                word, group = halogens[atom.GetAtomicNum()], {root}
            elif atom.GetAtomicNum() == 6:
                group = _carbon_component(mol, root, host)
                word, mapped = mapped_branch(mol, Chem, root, host)
                if (not group or not word or group != frozenset(mapped)
                        or any(mol.GetAtomWithIdx(i).IsInRing() for i in group)):
                    return None
                group = set(group)
            else:
                return None
            if group & owned:
                return None
            owned.update(group)
            branches.append((host, word))
    if owned != set(range(mol.GetNumAtoms())):
        return None
    from .api import mol_to_iupac
    from .formatting import _locant_sort_key, format_prefixes, join_prefix_parent, prefix_alphabetization_key
    from .naming_context import MoleculeContext
    from .ring_state_parent import _skeletons
    rows = []
    for hydrogen in possible:
        rw = Chem.RWMol(mol)
        for atom in radical:
            target = rw.GetAtomWithIdx(atom.GetIdx())
            count = atom.GetTotalNumHs(includeNeighbors=True) + atom.GetNumRadicalElectrons() - (atom.GetIdx() == main.GetIdx())
            target.SetNumRadicalElectrons(0)
            target.SetNumExplicitHs(count)
            target.SetNoImplicit(True)
        target = rw.GetAtomWithIdx(hydrogen.GetIdx())
        target.SetNumExplicitHs(hydrogen.GetTotalNumHs(includeNeighbors=True) - 1)
        target.SetNoImplicit(True)
        rw.GetBondBetweenAtoms(main.GetIdx(), hydrogen.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
        parent = _sanitize(rw, Chem)
        if parent is None or not all(parent.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        full_parent = mol_to_iupac(parent)
        if not full_parent.endswith('e') or ' ' in full_parent:
            continue
        for skeleton in _skeletons(MoleculeContext(parent, Chem)):
            base = skeleton[0]
            if base not in full_parent or set(skeleton[1]) != ring:
                continue
            locants = dict(zip(skeleton[1], skeleton[2]))
            if any(idx not in locants for idx in (main.GetIdx(), hydrogen.GetIdx(), *(a.GetIdx() for a in radical))):
                continue
            radical_locs = tuple(sorted((_locant_sort_key(locants[a.GetIdx()]) for a in radical)))
            main_loc, h_loc = locants[main.GetIdx()], locants[hydrogen.GetIdx()]
            prefixes = [(locants[host], word) for host, word in branches]
            by_word = {word: tuple(sorted(_locant_sort_key(loc) for loc, value in prefixes if value == word))
                       for _loc, word in prefixes}
            key = (radical_locs, _locant_sort_key(main_loc), _locant_sort_key(h_loc),
                   tuple(sorted(_locant_sort_key(loc) for loc, _word in prefixes)),
                   tuple((prefix_alphabetization_key(word), by_word[word])
                         for word in sorted(by_word, key=prefix_alphabetization_key)))
            if len(radical) == 2:
                other = next(a for a in radical if a.GetIdx() != main.GetIdx())
                name = base[:-1] + '-' + locants[other.GetIdx()] + '-yl-' + main_loc + '(' + h_loc + 'H)-ylidene'
            else:
                ending = 'yl' if main.GetNumRadicalElectrons() == 1 else 'ylidene'
                name = base[:-1] + '-' + main_loc + '(' + h_loc + 'H)-' + ending
            name = join_prefix_parent(format_prefixes(prefixes, hierarchical_enclosures=True), name)
            rows.append((key, name))
    if not rows:
        return None
    best = min(key for key, _name in rows)
    names = {name for key, name in rows if key == best}
    if len(names) != 1:
        return None
    return _decision(mol, names.pop(), (a.GetIdx() for a in radical), 'P-71.2.6/P-58.2')


def _heterochalcogen_prefix(mol, Chem, roots, groups, chain, remote_side):
    from .component_mapping import mapped_branch
    from .formatting import _organyl_to_oxy, enclose_name_unit
    remote, owned = mapped_branch(mol, Chem, roots[remote_side], chain[0] if remote_side == 0 else chain[-1])
    if not remote or frozenset(owned) != groups[remote_side]:
        return ''
    order = chain if remote_side == 0 else tuple(reversed(chain))
    ordered_elements = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in order]
    runs = []
    for element in ordered_elements:
        if runs and runs[-1][0] == element:
            runs[-1] = (element, runs[-1][1] + 1)
        else:
            runs.append((element, 1))
    terms = {8: {1: 'oxy', 2: 'peroxy', 3: 'trioxidanyl'},
             16: {1: 'sulfanyl', 2: 'disulfanyl', 3: 'trisulfanyl'},
             34: {1: 'selanyl', 2: 'diselanyl', 3: 'triselanyl'},
             52: {1: 'tellanyl', 2: 'ditellanyl', 3: 'tritellanyl'}}
    word = remote
    for index, (element, count) in enumerate(runs):
        term = terms[element].get(count)
        if term is None:
            from .large_heteromonocycle_methods import _numerical_term
            multiplier = _numerical_term(count)
            stem = {8: 'oxidanyl', 16: 'sulfanyl', 34: 'selanyl', 52: 'tellanyl'}[element]
            if not multiplier:
                return ''
            term = multiplier + stem
        if index == 0 and element == 8 and count == 1:
            word = _organyl_to_oxy(word)
        elif index == 0:
            word = (enclose_name_unit(word) if any(ch.isdigit() or ch in '()[]{}' for ch in word)
                    else word) + term
        else:
            simple_oxy = word in {'methoxy', 'ethoxy', 'propoxy', 'butoxy', 'phenoxy', 'tert-butoxy'}
            word = (word if simple_oxy else enclose_name_unit(word)) + term
    return word


def mixed_chalcogen_organic_chain(mol, Chem):
    """Two organic ends joined by fewer than four chalcogen heterounits."""
    chalcogens = {8, 16, 34, 52}
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, *chalcogens} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
                   or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    from .heterochalcogen_chain_methods import _ordered_path, _organyl_component
    hetero = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in chalcogens}
    carbon = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
    if len(hetero) < 2 or len({mol.GetAtomWithIdx(i).GetAtomicNum() for i in hetero}) < 2:
        return None
    chain = _ordered_path(mol, hetero)
    if len(chain) != len(hetero) or any(mol.GetAtomWithIdx(i).IsInRing()
            or mol.GetAtomWithIdx(i).GetTotalValence() != 2 for i in chain):
        return None
    if any(_bond(mol, a, b) != 1 for a, b in zip(chain, chain[1:])):
        return None
    numbers = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in chain)
    runs = []
    for z in numbers:
        if runs and runs[-1][0] == z:
            runs[-1] = (z, runs[-1][1] + 1)
        else:
            runs.append((z, 1))
    if len(runs) >= 4:
        return None  # Four heterounits require the skeletal-replacement comparison.
    seniority = {8: 0, 16: 1, 34: 2, 52: 3}
    if (len(chain) % 2 and all(z == numbers[i % 2] for i, z in enumerate(numbers))
            and seniority[numbers[0]] > seniority[numbers[1]]):
        return None  # P-21.2.3.1 preselected a[ba] parent.
    roots = []
    for end in (chain[0], chain[-1]):
        attached = [a.GetIdx() for a in mol.GetAtomWithIdx(end).GetNeighbors() if a.GetAtomicNum() == 6]
        if len(attached) != 1:
            return None
        roots.append(attached[0])
    groups = [_organyl_component(mol, root, end, carbon) for root, end in zip(roots, (chain[0], chain[-1]))]
    if any(group is None for group in groups) or groups[0] & groups[1] or set(groups[0] | groups[1]) != carbon:
        return None
    aromatic = [all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in group) for group in groups]
    if all(aromatic):
        if (len(runs) == 3 and runs[0] == runs[-1] and runs[0][0] == 8
                and runs[0][1] in {1, 2} and runs[1][0] in {16, 34, 52}):
            linker = {16: 'sulfanediyl', 34: 'selanediyl', 52: 'tellanediyl'}[runs[1][0]]
            if runs[1][1] > 1:
                from .large_heteromonocycle_methods import _numerical_term
                multiplier = _numerical_term(runs[1][1])
                if not multiplier:
                    return None
                linker = multiplier + linker
            outer = 'oxy' if runs[0][1] == 1 else 'peroxy'
            return _decision(mol, "1,1'-[" + linker + 'bis(' + outer + ')]dibenzene', (),
                             'P-68.4.2.4/P-15.3.2.1', 'multiplicative')
    scores = [(aromatic[i], len(groups[i])) for i in (0, 1)]
    parent_side = 0 if scores[0] >= scores[1] else 1
    if scores[0] == scores[1]:
        rooted = [Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(groups[i]), rootedAtAtom=roots[i],
                                          canonical=True, isomericSmiles=True) for i in (0, 1)]
        if rooted[0] != rooted[1]:
            return None
        # Equally senior identical organic parents leave the substituent
        # comparison to alphanumerical citation (P-68.4.2.4/P-14.5).
        from .formatting import prefix_alphabetization_key
        choices = [(_heterochalcogen_prefix(mol, Chem, roots, groups, chain, 1-i), i) for i in (0, 1)]
        if not all(word for word, _side in choices):
            return None
        parent_side = min(choices, key=lambda row: prefix_alphabetization_key(row[0]))[1]
    parent_group, parent_root = groups[parent_side], roots[parent_side]
    if not aromatic[parent_side]:
        adjacency = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in parent_group]
                     for i in parent_group}
        if len(adjacency[parent_root]) > 1 or any(len(row) > 2 for row in adjacency.values()):
            return None
    word = _heterochalcogen_prefix(mol, Chem, roots, groups, chain, 1-parent_side)
    if not word:
        return None
    from .formatting import enclose_name_unit
    rw = Chem.RWMol(mol)
    keep = set(groups[parent_side])
    for idx in sorted(set(range(mol.GetNumAtoms())) - keep, reverse=True):
        rw.RemoveAtom(idx)
    remaining = sorted(keep)
    root = remaining.index(roots[parent_side])
    bromine = rw.AddAtom(Chem.Atom(35))
    rw.AddBond(root, bromine, Chem.BondType.SINGLE)
    probe = _sanitize(rw, Chem)
    if probe is None:
        return None
    from .api import mol_to_iupac
    base = mol_to_iupac(probe)
    if base.count('bromo') != 1 or ' ' in base:
        return None
    return _decision(mol, base.replace('bromo', enclose_name_unit(word), 1), (), 'P-68.4.2.4/P-51.4.1.1')


def _imidohydrazide_from_hydrazide(name, count):
    """Apply functional replacement to an identified hydrazide suffix."""
    from .constants import MULT
    if count == 1:
        endings = (('formohydrazide', 'methanimidohydrazide'),
                   ('acetohydrazide', 'ethanimidohydrazide'),
                   ('benzohydrazide', 'benzenecarboximidohydrazide'),
                   ('carbohydrazide', 'carboximidohydrazide'),
                   ('anehydrazide', 'animidohydrazide'),
                   ('enehydrazide', 'enimidohydrazide'),
                   ('ynehydrazide', 'ynimidohydrazide'))
        if any(name.endswith(mult + 'carbohydrazide') for mult in MULT.values() if mult):
            return ''
    else:
        multiplier = MULT.get(count)
        if not multiplier:
            return ''
        if count == 2 and name.endswith('oxalohydrazide'):
            return name[:-len('oxalohydrazide')] + 'ethanediimidohydrazide'
        endings = ((multiplier + 'carbohydrazide', multiplier + 'carboximidohydrazide'),
                   (multiplier + 'hydrazide', multiplier + 'imidohydrazide'))
    matches = [(old, new) for old, new in endings if name.endswith(old)]
    if len(matches) != 1:
        return ''
    old, new = matches[0]
    return name[:-len(old)] + new


def imidic_hydrazide(mol, Chem):
    """One imidic hydrazide with carbon substituents and junior oxygen groups."""
    if not _clean(mol, Chem, {6, 7, 8, 9, 17, 35, 53}):
        return None
    sites = []
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6 or carbon.IsInRing():
            continue
        imines = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() == 7 and n.GetDegree() == 1
                  and n.GetTotalNumHs(includeNeighbors=True) == 1 and _bond(mol, carbon.GetIdx(), n.GetIdx()) == 2]
        amino = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() == 7 and not n.IsInRing()
                 and n.GetDegree() in {2, 3} and n.GetTotalValence() == 3
                 and _bond(mol, carbon.GetIdx(), n.GetIdx()) == 1]
        if len(imines) != 1 or len(amino) != 1:
            continue
        ends = [n for n in amino[0].GetNeighbors() if n.GetAtomicNum() == 7 and n.GetIdx() != imines[0].GetIdx()
                and not n.IsInRing() and n.GetDegree() in {1, 2, 3} and n.GetTotalValence() == 3
                and _bond(mol, amino[0].GetIdx(), n.GetIdx()) == 1]
        if len(ends) == 1:
            sites.append((carbon.GetIdx(), imines[0].GetIdx(), amino[0].GetIdx(), ends[0].GetIdx()))
    if not sites or len(set(i for site in sites for i in site)) != 4 * len(sites):
        return None
    # Replacing =NH by =O may create a second ordinary hydrazide. In that
    # case the proxy's selected parent need not be the converted acid residue.
    from .extended_inventory import selection_inventory
    from .functional_class_queries import acylhydrazide_query
    if (mol.HasSubstructMatch(acylhydrazide_query(Chem)) or any(
            g.center not in {s[0] for s in sites} and g.seniority <= (12, 0)
            for g in selection_inventory(mol).groups)):
        return None
    from .api import mol_to_iupac
    # The corresponding ordinary hydrazide provides the interleaved prefix
    # ordering and N/N-prime locants for one or several identical sites.
    proxy_rw = Chem.RWMol(mol)
    for _center, imine, _amino, _end in sites:
        atom = proxy_rw.GetAtomWithIdx(imine)
        atom.SetAtomicNum(8)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    proxy = _sanitize(proxy_rw, Chem)
    if proxy is not None:
        name = _imidohydrazide_from_hydrazide(mol_to_iupac(proxy), len(sites))
        if name:
            return _decision(mol, name, (site[0] for site in sites), 'P-66.3.1.2.1/P-66.3.3/P-41')
    if len(sites) != 1:
        return None
    # In the acid-probe fallback, permit other carbonyls only when they are
    # ketones/aldehydes, which are junior to a principal hydrazide. Acids,
    # esters, amides, and acid halides need their own seniority comparison.
    for oxygen in (a for a in mol.GetAtoms() if a.GetAtomicNum() == 8):
        multiples = [b for b in oxygen.GetBonds() if b.GetBondTypeAsDouble() != 1]
        for bond in multiples:
            host = bond.GetOtherAtom(oxygen)
            if (bond.GetBondTypeAsDouble() != 2 or host.GetAtomicNum() != 6
                    or any(a.GetIdx() != oxygen.GetIdx() and a.GetAtomicNum() != 6 for a in host.GetNeighbors())):
                return None
    center, imine, amino, end = sites[0]
    from .component_mapping import mapped_branch
    from .phosphorus_ester_methods import _carbon_component
    prefixes, owned = [], set()
    for nitrogen, excluded, locant in ((amino, {center, end}, 'N'), (end, {amino}, "N'")):
        for atom in mol.GetAtomWithIdx(nitrogen).GetNeighbors():
            root = atom.GetIdx()
            if root in excluded:
                continue
            if atom.GetAtomicNum() != 6 or _bond(mol, nitrogen, root) != 1:
                return None
            group = _carbon_component(mol, root, nitrogen)
            word, mapped = mapped_branch(mol, Chem, root, nitrogen)
            if not group or not word or group != frozenset(mapped) or group & owned or center in group:
                return None
            owned.update(group)
            prefixes.append((locant, word))
    rw = Chem.RWMol(mol)
    for idx in (imine, amino):
        atom = rw.GetAtomWithIdx(idx)
        atom.SetAtomicNum(8)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(idx == imine)
    for idx in sorted(owned | {end}, reverse=True):
        rw.RemoveAtom(idx)
    acid = _sanitize(rw, Chem)
    if acid is None:
        return None
    base = mol_to_iupac(acid)
    retained = {'formic acid': 'methanimidohydrazide',
                'acetic acid': 'ethanimidohydrazide',
                'benzoic acid': 'benzenecarboximidohydrazide'}
    if base in retained:
        name = retained[base]
    elif base.endswith('benzoic acid'):
        name = base[:-len('benzoic acid')] + 'benzenecarboximidohydrazide'
    elif base.endswith('acetic acid'):
        # A substituted retained acid may omit a locant required by its
        # systematic modified-acid derivative; obtain it from the hydrazide.
        name = None
    elif base.endswith('carboxylic acid'):
        name = base[:-len('carboxylic acid')] + 'carboximidohydrazide'
    elif base.endswith('oic acid'):
        name = base[:-len('oic acid')] + 'imidohydrazide'
    else:
        return None
    if prefixes or name is None:
        # Without a structured map of the acid's own prefixes, combining
        # them with N substituents could violate alphabetical citation. A
        # corresponding hydrazide can instead supply the complete ordering.
        import re
        if name is None or not (base in retained or re.fullmatch(r'(?:meth|eth|prop|but|pent|hex|hept|oct|non|dec)anoic acid', base)
                or re.fullmatch(r'cyclo(?:prop|but|pent|hex|hept|oct)anecarboxylic acid', base)):
            probe_rw = Chem.RWMol(mol)
            oxygen = probe_rw.GetAtomWithIdx(imine)
            oxygen.SetAtomicNum(8)
            oxygen.SetNumExplicitHs(0)
            oxygen.SetNoImplicit(True)
            probe = _sanitize(probe_rw, Chem)
            if probe is None:
                return None
            hydrazide = mol_to_iupac(probe)
            name = _imidohydrazide_from_hydrazide(hydrazide, 1)
            if not name:
                return None
        else:
            from .formatting import format_prefixes, join_prefix_parent
            name = join_prefix_parent(format_prefixes(prefixes, hierarchical_enclosures=True), name)
    return _decision(mol, name, (center,), 'P-66.3.1.2.1/P-41')


def iminium_with_substituted_imine(mol, Chem):
    """A single N-substituted imino prefix on an iminium carbon parent."""
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or mol.GetStereoGroups()
            or mol.GetRingInfo().NumRings()
            or any(a.GetAtomicNum() not in {6, 7} or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(ns) != 2 or any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetAtomicNum() != 7):
        return None
    ions = [a for a in ns if a.GetFormalCharge() == 1]
    neutrals = [a for a in ns if not a.GetFormalCharge()]
    if len(ions) != 1 or len(neutrals) != 1:
        return None
    ion, imine = ions[0], neutrals[0]
    if (ion.GetDegree() not in {1, 2, 3} or ion.GetTotalNumHs(includeNeighbors=True) != 3-ion.GetDegree()
            or imine.GetDegree() != 2 or imine.GetTotalNumHs(includeNeighbors=True)
            or any(a.GetAtomicNum() != 6 for n in ns for a in n.GetNeighbors())
            or sorted(_bond(mol, ion.GetIdx(), a.GetIdx()) for a in ion.GetNeighbors())
            != [1] * (ion.GetDegree()-1) + [2]
            or sorted(_bond(mol, imine.GetIdx(), a.GetIdx()) for a in imine.GetNeighbors()) != [1, 2]
            or any(sum(n.GetAtomicNum() == 6 for n in a.GetNeighbors()) > 2
                   for a in mol.GetAtoms() if a.GetAtomicNum() == 6)):
        return None
    from .component_mapping import mapped_branch
    groups, owned = [], set()
    for n in (ion, imine):
        for root in n.GetNeighbors():
            if _bond(mol, n.GetIdx(), root.GetIdx()) != 1:
                continue
            word, mapped = mapped_branch(mol, Chem, root.GetIdx(), n.GetIdx())
            group = set(mapped)
            if (word not in {'methyl', 'ethyl', 'propyl', 'butyl', 'pentyl', 'hexyl'}
                    or not group or group & owned or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in group)
                    or any(a.GetIdx() not in group and a.GetIdx() != n.GetIdx()
                           for i in group for a in mol.GetAtomWithIdx(i).GetNeighbors())):
                return None
            owned.update(group)
            groups.append((n.GetIdx(), word))
    imino_words = [word for owner, word in groups if owner == imine.GetIdx()]
    if len(imino_words) != 1:
        return None
    rw = Chem.RWMol(mol)
    cation = rw.GetAtomWithIdx(ion.GetIdx())
    cation.SetNumExplicitHs(2)
    cation.SetNoImplicit(True)
    host = next(a for a in imine.GetNeighbors() if _bond(mol, imine.GetIdx(), a.GetIdx()) == 2)
    host_copy = rw.GetAtomWithIdx(host.GetIdx())
    if host_copy.GetNoImplicit():
        host_copy.SetNumExplicitHs(host_copy.GetNumExplicitHs() + 1)
    marker = rw.GetAtomWithIdx(imine.GetIdx())
    marker.SetAtomicNum(9)
    marker.SetNumExplicitHs(0)
    marker.SetNoImplicit(True)
    rw.GetBondBetweenAtoms(imine.GetIdx(), host.GetIdx()).SetBondType(Chem.BondType.SINGLE)
    for index in sorted(owned, reverse=True):
        rw.RemoveAtom(index)
    proxy = _sanitize(rw, Chem)
    if proxy is None:
        return None
    from .api import mol_to_iupac
    import re
    match = re.fullmatch(r'(\d+)-fluoro(.+iminium)', mol_to_iupac(proxy))
    if match is None:
        return None
    from .formatting import format_prefixes, join_prefix_parent
    imino = imino_words[0] + 'imino'
    prefixes = [('N', word) for owner, word in groups if owner == ion.GetIdx()]
    prefixes.append((match.group(1), imino))
    rendered = format_prefixes(prefixes, hierarchical_enclosures=True)
    token = match.group(1) + '-' + imino
    if rendered.count(token) != 1:
        return None
    rendered = rendered.replace(token, match.group(1) + '-(' + imino + ')', 1)
    return _decision(mol, join_prefix_parent(rendered, match.group(2)), (ion.GetIdx(),),
                     'P-62.3.1.2/P-62.6.1/P-73.1', 'functional-class')


def monoiminium_with_imines(mol, Chem):
    """One iminium suffix and terminal imino prefixes on a carbon chain."""
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or mol.GetStereoGroups()
            or mol.GetRingInfo().NumRings()
            or any(a.GetAtomicNum() not in {6, 7} or a.GetNumRadicalElectrons()
                   or a.HasQuery() for a in mol.GetAtoms())
            or any(b.HasQuery() for b in mol.GetBonds())):
        return None
    has_stereo = (any(int(a.GetChiralTag()) for a in mol.GetAtoms())
                  or any(int(b.GetStereo()) for b in mol.GetBonds()))
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    charged = [a for a in nitrogens if a.GetFormalCharge() == 1]
    neutral = [a for a in nitrogens if not a.GetFormalCharge()]
    if (len(charged) != 1 or not neutral or len(charged) + len(neutral) != len(nitrogens)
            or any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetAtomicNum() != 7)):
        return None
    if any(a.GetDegree() != 1 or a.GetNeighbors()[0].GetAtomicNum() != 6
           or _bond(mol, a.GetIdx(), a.GetNeighbors()[0].GetIdx()) != 2
           or a.GetTotalNumHs(includeNeighbors=True) != 1 for a in neutral):
        return None
    ion = charged[0]
    if (ion.GetDegree() not in {1, 2, 3}
            or ion.GetTotalNumHs(includeNeighbors=True) != 3 - ion.GetDegree()
            or any(a.GetAtomicNum() != 6 for a in ion.GetNeighbors())
            or sorted(_bond(mol, ion.GetIdx(), a.GetIdx()) for a in ion.GetNeighbors())
            != [1] * (ion.GetDegree() - 1) + [2]):
        return None
    if any(sum(n.GetAtomicNum() == 6 for n in a.GetNeighbors()) > 2
           for a in mol.GetAtoms() if a.GetAtomicNum() == 6):
        return None
    # The fluoride proxy preserves prefix ordering only if every N-bound
    # group sorts on the same side of both "fluoro" and "imino".
    from .component_mapping import mapped_branch
    from .formatting import prefix_alphabetization_key
    fluor_key = prefix_alphabetization_key('fluoro')
    imino_key = prefix_alphabetization_key('imino')
    for neighbor in ion.GetNeighbors():
        if _bond(mol, ion.GetIdx(), neighbor.GetIdx()) != 1:
            continue
        word, mapped = mapped_branch(mol, Chem, neighbor.GetIdx(), ion.GetIdx())
        if not word or not mapped:
            return None
        key = prefix_alphabetization_key(word)
        if fluor_key < key < imino_key:
            return None
    rw = Chem.RWMol(mol)
    isotope_markers = _imino_proxy_isotopes(neutral)
    for n in neutral:
        host = rw.GetAtomWithIdx(n.GetNeighbors()[0].GetIdx())
        if host.GetNoImplicit():
            # Bracketed isotope atoms often carry explicit H counts. The
            # single-bonded proxy needs the H released by replacing C=N.
            host.SetNumExplicitHs(host.GetNumExplicitHs() + 1)
        atom = rw.GetAtomWithIdx(n.GetIdx())
        atom.SetAtomicNum(9)
        if n.GetIsotope():
            atom.SetIsotope(isotope_markers[n.GetIsotope()])
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        rw.GetBondBetweenAtoms(n.GetIdx(), n.GetNeighbors()[0].GetIdx()).SetBondType(Chem.BondType.SINGLE)
    proxy = _sanitize(rw, Chem)
    if proxy is None or has_stereo and not _stereo_unchanged_after_removal(mol, proxy, set(), Chem):
        return None
    from .api import mol_to_iupac
    base = mol_to_iupac(proxy)
    if not base.endswith('iminium') or 'fluoro' not in base or ' ' in base:
        return None
    if has_stereo and not _stereo_descriptors_preserved(mol, base, Chem):
        return None
    name = _imino_from_fluoro_proxy(base, isotope_markers)
    if sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms()) == 1 and name.startswith('1-imino'):
        name = name[2:]
    before_isotope_ranking = name
    name = _rank_isotopic_imino_numbering(mol, name, charged, neutral, Chem)
    if not name:
        return None
    if has_stereo and name != before_isotope_ranking:
        name = _remap_imino_parent_stereo(mol, name, charged, neutral, Chem)
        if not name or not _stereo_descriptors_preserved(mol, name, Chem):
            return None
    if not _imino_isotope_counts_match(mol, name):
        return None
    return _decision(mol, name, (charged[0].GetIdx(),),
                     'P-62.3.1.2/P-62.6.1/P-73.1', 'functional-class')


def polyiminium_with_imines(mol, Chem):
    """Multiple protonated imines with additional terminal imino groups."""
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or mol.GetStereoGroups()
            or mol.GetRingInfo().NumRings()
            or any(a.GetAtomicNum() not in {6, 7} or a.GetNumRadicalElectrons()
                   or a.HasQuery() for a in mol.GetAtoms())
            or any(b.HasQuery() for b in mol.GetBonds())):
        return None
    has_stereo = (any(int(a.GetChiralTag()) for a in mol.GetAtoms())
                  or any(int(b.GetStereo()) for b in mol.GetBonds()))
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    charged = [a for a in ns if a.GetFormalCharge() == 1]
    neutral = [a for a in ns if not a.GetFormalCharge()]
    from .constants import MULT
    if (len(charged) < 2 or len(charged) not in MULT or not neutral
            or len(charged) + len(neutral) != len(ns)
            or any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetAtomicNum() != 7)):
        return None
    if any(a.GetDegree() != 1 or a.GetNeighbors()[0].GetAtomicNum() != 6
           or _bond(mol, a.GetIdx(), a.GetNeighbors()[0].GetIdx()) != 2
           or a.GetTotalNumHs(includeNeighbors=True) != 1 for a in neutral):
        return None
    if any(a.GetDegree() not in {1, 2} or a.GetTotalNumHs(includeNeighbors=True) != 3-a.GetDegree()
           or any(n.GetAtomicNum() != 6 for n in a.GetNeighbors())
           or sorted(_bond(mol, a.GetIdx(), n.GetIdx()) for n in a.GetNeighbors())
           != [1] * (a.GetDegree()-1) + [2] for a in charged):
        return None
    if any(sum(n.GetAtomicNum() == 6 for n in a.GetNeighbors()) > 2
           for a in mol.GetAtoms() if a.GetAtomicNum() == 6):
        return None
    from .component_mapping import mapped_branch
    from .formatting import prefix_alphabetization_key
    fluor, imino_key = prefix_alphabetization_key('fluoro'), prefix_alphabetization_key('imino')
    for n in charged:
        for root in n.GetNeighbors():
            if _bond(mol, n.GetIdx(), root.GetIdx()) != 1:
                continue
            word, mapped = mapped_branch(mol, Chem, root.GetIdx(), n.GetIdx())
            if not word or not mapped or fluor < prefix_alphabetization_key(word) < imino_key:
                return None
    rw = Chem.RWMol(mol)
    isotope_markers = _imino_proxy_isotopes(neutral)
    for n in neutral:
        host = rw.GetAtomWithIdx(n.GetNeighbors()[0].GetIdx())
        if host.GetNoImplicit():
            host.SetNumExplicitHs(host.GetNumExplicitHs()+1)
        atom = rw.GetAtomWithIdx(n.GetIdx())
        atom.SetAtomicNum(9)
        if n.GetIsotope():
            atom.SetIsotope(isotope_markers[n.GetIsotope()])
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        rw.GetBondBetweenAtoms(n.GetIdx(), n.GetNeighbors()[0].GetIdx()).SetBondType(Chem.BondType.SINGLE)
    proxy = _sanitize(rw, Chem)
    if proxy is None or has_stereo and not _stereo_unchanged_after_removal(mol, proxy, set(), Chem):
        return None
    from .api import mol_to_iupac
    base = mol_to_iupac(proxy)
    if not base.endswith(MULT[len(charged)] + 'iminium') or 'fluoro' not in base or ' ' in base:
        return None
    if has_stereo and not _stereo_descriptors_preserved(mol, base, Chem):
        return None
    name = _imino_from_fluoro_proxy(base, isotope_markers)
    before_isotope_ranking = name
    name = _rank_isotopic_imino_numbering(mol, name, charged, neutral, Chem)
    if not name:
        return None
    if has_stereo and name != before_isotope_ranking:
        name = _remap_imino_parent_stereo(mol, name, charged, neutral, Chem)
        if not name or not _stereo_descriptors_preserved(mol, name, Chem):
            return None
    if not _imino_isotope_counts_match(mol, name):
        return None
    return _decision(mol, name, (a.GetIdx() for a in charged),
                     'P-62.3.1.2/P-62.6.1/P-73.1', 'functional-class')


def polyiminium(mol, Chem):
    """Name a polyimine when each imine nitrogen is protonated once."""
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, 9, 17, 35, 53}
                   or a.GetNumRadicalElectrons() or a.HasQuery() for a in mol.GetAtoms())
            or any(b.HasQuery() for b in mol.GetBonds())):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    from .constants import MULT
    if len(ns) not in MULT or any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetAtomicNum() != 7):
        return None
    for n in ns:
        if (n.GetFormalCharge() != 1 or n.IsInRing() or n.GetDegree() not in {1, 2}
                or n.GetTotalNumHs(includeNeighbors=True) != 3 - n.GetDegree()
                or any(a.GetAtomicNum() != 6 for a in n.GetNeighbors())
                or sorted(_bond(mol, n.GetIdx(), a.GetIdx()) for a in n.GetNeighbors())
                != [1] * (n.GetDegree() - 1) + [2]):
            return None
    rw = Chem.RWMol(mol)
    for n in ns:
        atom = rw.GetAtomWithIdx(n.GetIdx())
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(2 - n.GetDegree())
        atom.SetNoImplicit(True)
    neutral = _sanitize(rw, Chem)
    if neutral is None:
        return None
    if (any(int(a.GetChiralTag()) for a in mol.GetAtoms()) or any(int(b.GetStereo()) for b in mol.GetBonds())):
        if not _stereo_unchanged_after_removal(mol, neutral, set(), Chem):
            return None
    from .api import mol_to_iupac
    base = mol_to_iupac(neutral)
    if not base.endswith(MULT[len(ns)] + 'imine') or ' ' in base:
        return None
    return _decision(mol, base[:-1] + 'ium', (n.GetIdx() for n in ns),
                     'P-62.6.1/P-73.1', 'functional-class')


def iminium(mol, Chem):
    """An imine-derived cation with zero, one, or two N substituents."""
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 16, 17, 34, 35, 52, 53}
                   or a.GetNumRadicalElectrons() or a.HasQuery() for a in mol.GetAtoms())
            or any(b.HasQuery() for b in mol.GetBonds())):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(ns) != 1:
        return None
    n = ns[0]
    if (n.GetFormalCharge() != 1 or n.IsInRing() or n.GetDegree() not in {1, 2, 3}
            or n.GetTotalNumHs(includeNeighbors=True) != 3 - n.GetDegree()
            or any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetIdx() != n.GetIdx())):
        return None
    bonds = sorted(_bond(mol, n.GetIdx(), x.GetIdx()) for x in n.GetNeighbors())
    if bonds != [1] * (n.GetDegree() - 1) + [2]:
        return None
    if n.GetDegree() != 2 or any(a.GetAtomicNum() in {16, 34, 52} and _bond(mol, n.GetIdx(), a.GetIdx()) == 1
                                 for a in n.GetNeighbors()):
        # With two N substituents, remove whole independently mapped branches
        # before naming the parent imine; this keeps the N locants explicit.
        from .component_mapping import mapped_branch
        from .formatting import format_prefixes, join_prefix_parent
        import re
        prefixes, owned = [], set()
        for root in n.GetNeighbors():
            if _bond(mol, n.GetIdx(), root.GetIdx()) != 1:
                continue
            word, mapped = mapped_branch(mol, Chem, root.GetIdx(), n.GetIdx())
            group = set(mapped)
            z = root.GetAtomicNum()
            # An N-bound hydroxy/alkoxy or chalcogen analogue is permitted
            # when it owns a single ordinary-valence chalcogen and a carbon
            # substituent. A connected, mapped group must be wholly detached
            # by removing the N--root bond.
            if (not group or not word or root.GetIdx() not in group or group & owned
                    or z not in {6, 8, 16, 34, 52}
                    or (z != 6 and (root.IsInRing() or root.GetTotalValence() != 2
                                     or root.GetDegree() not in {1, 2}))
                    or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 9, 17, 35, 53}
                           for i in group if i != root.GetIdx())
                    or any(a.GetIdx() not in group and a.GetIdx() != n.GetIdx()
                           for i in group for a in mol.GetAtomWithIdx(i).GetNeighbors())):
                return None
            owned.update(group)
            prefixes.append(('N', word))
        # Branch-local annotations need their own mapped prefix renderer.
        # Retained annotations are allowed only when the CIP comparison below
        # proves that detaching the branches leaves their descriptors intact.
        if (any(mol.GetAtomWithIdx(i).GetIsotope() or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in owned)
                or any(int(b.GetStereo()) and (b.GetBeginAtomIdx() in owned or b.GetEndAtomIdx() in owned)
                       for b in mol.GetBonds())):
            return None
        rw = Chem.RWMol(mol)
        atom = rw.GetAtomWithIdx(n.GetIdx())
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(1)
        atom.SetNoImplicit(True)
        for idx in sorted(owned, reverse=True):
            rw.RemoveAtom(idx)
        neutral = _sanitize(rw, Chem)
        if neutral is None:
            return None
        if (any(int(a.GetChiralTag()) for a in mol.GetAtoms()) or any(int(b.GetStereo()) for b in mol.GetBonds())):
            if not _stereo_unchanged_after_removal(mol, neutral, owned, Chem):
                return None
        from .api import mol_to_iupac
        base = mol_to_iupac(neutral)
        if not prefixes:
            if not base.endswith('imine') or ' ' in base:
                return None
            return _decision(mol, base[:-1] + 'ium', (n.GetIdx(),),
                             'P-62.6.1/P-73.1', 'functional-class')
        # Interleave the N substituents with the independently delimited
        # prefixes of the neutral imine, including identical substituents at
        # nitrogen and carbon. Decline an unparsed prefix or parent rather
        # than changing its citation order by textual concatenation.
        stereo = ''
        stereo_match = re.match(r'\((?:\d+[a-z]?[RSEZrs](?:,\d+[a-z]?[RSEZrs])*)\)-', base)
        if stereo_match:
            stereo, base = stereo_match.group(), base[stereo_match.end():]
        pattern = re.compile(r'(\d+(?:,\d+)*)-(di|tri|tetra|penta|hexa)?'
                             r'(fluoro|chloro|bromo|iodo|methyl|ethyl|propyl)')
        mult = {'di': 2, 'tri': 3, 'tetra': 4, 'penta': 5, 'hexa': 6}
        position = 0
        while True:
            match = pattern.match(base, position)
            if match is None:
                break
            locants = match.group(1).split(',')
            if len(locants) != mult.get(match.group(2), 1):
                return None
            prefixes.extend((locant, match.group(3)) for locant in locants)
            position = match.end()
            if position < len(base) and base[position] == '-' and pattern.match(base, position + 1):
                position += 1
                continue
            break
        parent = base[position:]
        isotope = ''
        isotope_match = re.match(r'\((?=[^()]*\d[A-Z])[^()]+\)', parent)
        if isotope_match:
            isotope, parent = isotope_match.group(), parent[isotope_match.end():]
        root = r'(?:meth|eth|prop|but|pent|hex|hept|oct|non|dec|undec|dodec|tridec|tetradec|pentadec|hexadec|heptadec|octadec|nonadec|icos)'
        if not re.fullmatch(r'(?:cyclo)?' + root + r'(?:an|(?:-\d+(?:,\d+)*-)?(?:en|yn))'
                            r'(?:-\d+(?:,\d+)*-(?:en|yn))*(?:-\d+-)?imine', parent):
            return None
        prefix = format_prefixes(prefixes, hierarchical_enclosures=True)
        name = (prefix + isotope + parent[:-1] + 'ium' if isotope
                else join_prefix_parent(prefix, parent[:-1] + 'ium'))
        return _decision(mol, stereo + name,
                         (n.GetIdx(),), 'P-62.6.1/P-73.1', 'functional-class')
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(n.GetIdx())
    atom.SetFormalCharge(0)
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(True)
    neutral = _sanitize(rw, Chem)
    if neutral is None:
        return None
    if (any(int(a.GetChiralTag()) for a in mol.GetAtoms()) or any(int(b.GetStereo()) for b in mol.GetBonds())):
        if not _stereo_unchanged_after_removal(mol, neutral, set(), Chem):
            return None
    from .api import mol_to_iupac
    base = mol_to_iupac(neutral)
    if not base.endswith('imine') or ' ' in base:
        return None
    return _decision(mol, base[:-1] + 'ium', (n.GetIdx(),), 'P-62.6.1/P-73.1', 'functional-class')


def sulfonoperoxoic_acid(mol, Chem):
    """Sulfonoperoxoic acid groups selected against other acid functions."""
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 16, 17, 35, 53}
                   or a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.HasQuery()
                   for a in mol.GetAtoms())
            or any(b.HasQuery() for b in mol.GetBonds())):
        return None
    sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    if not sulfurs:
        return None
    ends, owned = [], set()
    for s in sulfurs:
        oxo = [a for a in s.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
               and _bond(mol, s.GetIdx(), a.GetIdx()) == 2]
        bridges = [a for a in s.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 2
                   and _bond(mol, s.GetIdx(), a.GetIdx()) == 1]
        roots = [a for a in s.GetNeighbors() if a.GetAtomicNum() == 6 and _bond(mol, s.GetIdx(), a.GetIdx()) == 1]
        if s.IsInRing() or s.GetDegree() != 4 or len(oxo) != 2 or len(bridges) != 1 or len(roots) != 1:
            return None
        terminal = [a for a in bridges[0].GetNeighbors() if a.GetIdx() != s.GetIdx() and a.GetAtomicNum() == 8
                    and a.GetDegree() == 1 and a.GetTotalNumHs(includeNeighbors=True) == 1
                    and _bond(mol, bridges[0].GetIdx(), a.GetIdx()) == 1]
        if len(terminal) != 1:
            return None
        group = {a.GetIdx() for a in (*oxo, bridges[0], terminal[0])}
        if group & owned or any(mol.GetAtomWithIdx(i).GetIsotope() for i in group):
            return None
        owned.update(group)
        ends.append(terminal[0].GetIdx())
    rw = Chem.RWMol(mol)
    for idx in sorted(ends, reverse=True):
        rw.RemoveAtom(idx)
    acid = _sanitize(rw, Chem)
    if acid is None:
        return None
    if (any(int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) for b in mol.GetBonds())):
        if not _stereo_unchanged_after_removal(mol, acid, set(ends), Chem):
            return None
    from .api import mol_to_iupac
    base = mol_to_iupac(acid)
    if not base.endswith('sulfonic acid'):
        return None
    return _decision(mol, base[:-len('sulfonic acid')] + 'sulfonoperoxoic acid',
                     (s.GetIdx() for s in sulfurs), 'P-65.3.1.2/P-41')


def identical_amines_with_oxy_link(mol, Chem):
    """Two identical carbon amines joined through their nitrogen atoms."""
    chalcogens = {8, 16, 34, 52}
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, *chalcogens} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
                   or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    chain_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() in chalcogens]
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if not chain_atoms or len(nitrogens) != 2:
        return None
    indices = {a.GetIdx() for a in chain_atoms}
    if len(indices) == 1:
        chain = (next(iter(indices)),)
    else:
        from .heterochalcogen_chain_methods import _ordered_path
        chain = _ordered_path(mol, indices)
    if len(chain) != len(indices):
        return None
    sequence = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in chain)
    simple = {(8,): 'oxy', (16,): 'sulfanediyl', (34,): 'selanediyl', (52,): 'tellanediyl',
              (8, 8): 'peroxy', (16, 16): 'disulfanediyl',
              (34, 34): 'diselanediyl', (52, 52): 'ditellanediyl',
              (8, 8, 8): 'trioxidanediyl', (16, 16, 16): 'trisulfanediyl',
              (34, 34, 34): 'triselanediyl', (52, 52, 52): 'tritellanediyl'}
    if sequence in simple:
        linker = simple[sequence]
    elif len(set(sequence)) == 1:
        # P-68.4.1.2 derives a divalent linker from the complete homogeneous
        # chalcogen hydride, including chains longer than three atoms.
        from .large_heteromonocycle_methods import _numerical_term
        stem = {8: 'oxidanediyl', 16: 'sulfanediyl', 34: 'selanediyl', 52: 'tellanediyl'}[sequence[0]]
        multiplier = _numerical_term(len(sequence))
        if not multiplier:
            return None
        linker = multiplier + stem
    else:
        # Retain each homogeneous chalcogen run when it joins the same two
        # amine parents, and express identical terminal runs multiplicatively.
        linker = None
        for edge, end in ((1, 'oxy'), (2, 'peroxy')):
            middle = sequence[edge:-edge]
            if (sequence[:edge] == sequence[-edge:] == (8,) * edge
                    and 1 <= len(middle) <= 3 and len(set(middle)) == 1
                    and middle[0] in {16, 34, 52}):
                central = {16: 'sulfanediyl', 34: 'selanediyl', 52: 'tellanediyl'}[middle[0]]
                linker = ('', 'di', 'tri')[len(middle) - 1] + central + f'bis({end})'
                break
        if linker is None:
            return None
    if any(a.IsInRing() or a.GetDegree() != 2 or a.GetTotalValence() != 2
           or any(_bond(mol, a.GetIdx(), n.GetIdx()) != 1 for n in a.GetNeighbors())
           for a in chain_atoms):
        return None
    ends = [mol.GetAtomWithIdx(i) for i in (chain[0], chain[-1])]
    if len(chain) == 1:
        if {a.GetIdx() for a in ends[0].GetNeighbors()} != {a.GetIdx() for a in nitrogens}:
            return None
    elif ({n.GetIdx() for end in ends for n in end.GetNeighbors() if n.GetAtomicNum() == 7}
          != {n.GetIdx() for n in nitrogens}):
        return None
    for n in nitrogens:
        if (n.IsInRing() or n.GetDegree() not in {2, 3}
                or n.GetTotalNumHs(includeNeighbors=True) != 3 - n.GetDegree()
                or sum(a.GetIdx() in indices for a in n.GetNeighbors()) != 1
                or any(a.GetAtomicNum() != 6 for a in n.GetNeighbors() if a.GetIdx() not in indices)
                or any(_bond(mol, n.GetIdx(), a.GetIdx()) != 1 for a in n.GetNeighbors())):
            return None
    rw = Chem.RWMol(mol)
    for idx in sorted(indices, reverse=True):
        rw.RemoveAtom(idx)
    separated = _sanitize(rw, Chem)
    if separated is None:
        return None
    fragments = Chem.GetMolFrags(separated, asMols=True, sanitizeFrags=True)
    if len(fragments) != 2 or Chem.MolToSmiles(fragments[0]) != Chem.MolToSmiles(fragments[1]):
        return None
    from .api import mol_to_iupac
    from .formatting import enclose_name_unit
    parent = mol_to_iupac(fragments[0])
    if not parent.endswith(('amine', 'aniline')) or ' ' in parent:
        return None
    if all(n.GetDegree() == 2 for n in nitrogens):
        import re
        if re.fullmatch(r'[a-z]+', parent):
            repeated = 'di' + parent
        elif re.fullmatch(r'[a-z]+(?:-\d+(?:,\d+)*-(?:en|yn))?-\d+-amine', parent):
            repeated = 'di' + enclose_name_unit(parent)
        else:
            repeated = 'bis' + enclose_name_unit(parent)
    else:
        repeated = 'bis' + enclose_name_unit(parent)
    if '(' in linker:
        linker = enclose_name_unit(linker)
    return _decision(mol, "N,N'-" + linker + repeated,
                     (a.GetIdx() for a in nitrogens), 'P-15.3.2.4.1/P-62.2', 'multiplicative')


def differently_substituted_linked_amines(mol, Chem):
    """Select the more substituted of two otherwise identical amine parents."""
    chalcogens = {8, 16, 34, 52}
    if not _clean(mol, Chem, {6, 7, *chalcogens}):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    indices = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in chalcogens}
    if len(nitrogens) != 2 or not indices or sorted(a.GetDegree() for a in nitrogens) != [2, 3]:
        return None
    from .heterochalcogen_chain_methods import _ordered_path
    chain = (next(iter(indices)),) if len(indices) == 1 else _ordered_path(mol, indices)
    if len(chain) != len(indices):
        return None
    for idx in chain:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.IsInRing() or atom.GetDegree() != 2 or atom.GetTotalValence() != 2
                or any(_bond(mol, idx, a.GetIdx()) != 1 for a in atom.GetNeighbors())):
            return None
    for n in nitrogens:
        if (n.IsInRing() or n.GetTotalNumHs(includeNeighbors=True) != 3-n.GetDegree()
                or sum(a.GetIdx() in indices for a in n.GetNeighbors()) != 1
                or any(a.GetAtomicNum() != 6 for a in n.GetNeighbors() if a.GetIdx() not in indices)
                or any(_bond(mol, n.GetIdx(), a.GetIdx()) != 1 for a in n.GetNeighbors())):
            return None
    endpoints = {a.GetIdx() for idx in (chain[0], chain[-1])
                 for a in mol.GetAtomWithIdx(idx).GetNeighbors() if a.GetAtomicNum() == 7}
    if endpoints != {n.GetIdx() for n in nitrogens}:
        return None
    selected = next(n for n in nitrogens if n.GetDegree() == 3)
    other = next(n for n in nitrogens if n.GetDegree() == 2)
    rw = Chem.RWMol(mol)
    for idx in sorted(indices, reverse=True):
        rw.RemoveAtom(idx)
    separated = _sanitize(rw, Chem)
    if separated is None:
        return None
    fragments = Chem.GetMolFrags(separated, asMols=True, sanitizeFrags=True)
    fragment_ids = Chem.GetMolFrags(separated)
    # Translate original N indices after the chalcogen atoms have been removed.
    retained = [i for i in range(mol.GetNumAtoms()) if i not in indices]
    positions = {old: new for new, old in enumerate(retained)}
    by_n = {original.GetIdx(): next((fragment for ids, fragment in zip(fragment_ids, fragments, strict=True)
                                     if positions[original.GetIdx()] in ids), None)
            for original in nitrogens}
    if len(fragments) != 2 or any(fragment is None for fragment in by_n.values()):
        return None
    from .api import mol_to_iupac
    import re
    parent_name = mol_to_iupac(by_n[selected.GetIdx()])
    other_name = mol_to_iupac(by_n[other.GetIdx()])
    match = re.fullmatch(r'N-(methyl|ethyl|propyl|butyl|pentyl|hexyl)([a-z0-9,-]+amine)', parent_name)
    if match is None or match.group(2) != other_name:
        return None
    root = next(a for a in selected.GetNeighbors() if a.GetIdx() in indices)
    from .component_mapping import mapped_branch
    word, mapped = mapped_branch(mol, Chem, root.GetIdx(), selected.GetIdx())
    branch = set(mapped)
    if not word and len(chain) == 2 and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in chain):
        carbon_root = next(a for a in other.GetNeighbors() if a.GetAtomicNum() == 6)
        alkyl, other_atoms = mapped_branch(mol, Chem, carbon_root.GetIdx(), other.GetIdx())
        if alkyl in {'methyl', 'ethyl', 'propyl', 'butyl', 'pentyl', 'hexyl'}:
            word = '(' + alkyl + 'amino)peroxy'
            branch = set(other_atoms) | indices | {other.GetIdx()}
    if (not word or not branch or other.GetIdx() not in branch or indices - branch
            or any(a.GetIdx() not in branch and a.GetIdx() != selected.GetIdx()
                   for i in branch for a in mol.GetAtomWithIdx(i).GetNeighbors())):
        return None
    # Preserve an identical di- or trichalcogen run as one substituent group.
    from itertools import groupby
    runs = [(z, len(tuple(run))) for z, run in groupby(mol.GetAtomWithIdx(i).GetAtomicNum() for i in chain)]
    if any(count > 3 for _z, count in runs):
        return None
    for element, term in ((16, 'sulfanyl'), (34, 'selanyl'), (52, 'tellanyl')):
        for count, multiplier in ((3, 'tri'), (2, 'di')):
            if (element, count) in runs:
                word = word.replace(term * count, multiplier + term)
    carbon_root = next(a for a in other.GetNeighbors() if a.GetAtomicNum() == 6)
    alkyl, _mapped = mapped_branch(mol, Chem, carbon_root.GetIdx(), other.GetIdx())
    if alkyl in {'methyl', 'ethyl', 'propyl', 'butyl', 'pentyl', 'hexyl'}:
        amino = alkyl + 'amino'
        if word.startswith(amino) and len(word) > len(amino):
            word = '(' + amino + ')' + word[len(amino):]
        elif word.startswith('(' + amino) and len(word) > len(amino) + 1 and word[len(amino) + 1] != ')':
            end = word.find(')')
            if end > len(amino) + 1:
                word = '[(' + amino + ')' + word[len(amino) + 1:end] + ']' + word[end + 1:]
    from .formatting import format_prefixes, join_prefix_parent
    prefixes = [('N', match.group(1)), ('N', word)]
    name = join_prefix_parent(format_prefixes(prefixes, hierarchical_enclosures=True), match.group(2))
    return _decision(mol, name, (selected.GetIdx(),), 'P-15.3.2.4.2/P-44.3/P-45.2.1')


def different_linear_amines_with_chalcogen_link(mol, Chem):
    """Compare different straight-chain amine parents across a chalcogen path."""
    chalcogens = {8, 16, 34, 52}
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, *chalcogens} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
                   or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    hetero = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in chalcogens}
    carbons = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
    if len(nitrogens) != 2 or not hetero:
        return None
    from .heterochalcogen_chain_methods import _ordered_path, _organyl_component
    chain = (next(iter(hetero)),) if len(hetero) == 1 else _ordered_path(mol, hetero)
    if len(chain) != len(hetero):
        return None
    if any(mol.GetAtomWithIdx(i).IsInRing() or mol.GetAtomWithIdx(i).GetDegree() != 2
           or mol.GetAtomWithIdx(i).GetTotalValence() != 2
           or any(_bond(mol, i, n.GetIdx()) != 1 for n in mol.GetAtomWithIdx(i).GetNeighbors())
           for i in chain):
        return None
    groups, roots, endpoints = [], [], []
    for nitrogen in nitrogens:
        if (nitrogen.IsInRing() or nitrogen.GetDegree() != 2
                or nitrogen.GetTotalNumHs(includeNeighbors=True) != 1
                or any(_bond(mol, nitrogen.GetIdx(), n.GetIdx()) != 1 for n in nitrogen.GetNeighbors())):
            return None
        attached = [n.GetIdx() for n in nitrogen.GetNeighbors() if n.GetIdx() in hetero]
        carbon = [n.GetIdx() for n in nitrogen.GetNeighbors() if n.GetIdx() in carbons]
        if len(attached) != 1 or len(carbon) != 1:
            return None
        group = _organyl_component(mol, carbon[0], nitrogen.GetIdx(), carbons)
        if group is None or any(mol.GetAtomWithIdx(i).IsInRing() for i in group):
            return None
        groups.append(group); roots.append(carbon[0]); endpoints.append(attached[0])
    if groups[0] & groups[1] or groups[0] | groups[1] != carbons:
        return None
    if {endpoints[0], endpoints[1]} != {chain[0], chain[-1]} or len(groups[0]) == len(groups[1]):
        return None
    senior = 0 if len(groups[0]) > len(groups[1]) else 1
    other = 1 - senior
    # A straight senior chain has more skeletal atoms than the entire other
    # carbon tree, so branching of the latter cannot change the parent choice.
    if (any(sum(n.GetIdx() in groups[senior] for n in mol.GetAtomWithIdx(i).GetNeighbors()) > 2
            for i in groups[senior])
            or sum(n.GetIdx() in groups[senior] for n in mol.GetAtomWithIdx(roots[senior]).GetNeighbors()) > 1):
        return None
    from .component_mapping import mapped_branch
    remote, owned = mapped_branch(mol, Chem, roots[other], nitrogens[other].GetIdx())
    if not remote or frozenset(owned) != groups[other]:
        return None
    from .formatting import enclose_name_unit, format_prefixes, join_prefix_parent
    import re
    word = (remote if re.fullmatch(r'[a-z]+', remote) else enclose_name_unit(remote)) + 'amino'
    ordered = chain if endpoints[other] == chain[0] else tuple(reversed(chain))
    from itertools import groupby
    for element, run in groupby(mol.GetAtomWithIdx(i).GetAtomicNum() for i in ordered):
        count = len(tuple(run))
        terms = {8: ('oxy', 'peroxy', 'trioxidanyl'),
                 16: ('sulfanyl', 'disulfanyl', 'trisulfanyl'),
                 34: ('selanyl', 'diselanyl', 'triselanyl'),
                 52: ('tellanyl', 'ditellanyl', 'tritellanyl')}
        if count <= 3:
            term = terms[element][count-1]
        else:
            from .large_heteromonocycle_methods import _numerical_term
            multiplier = _numerical_term(count)
            if not multiplier:
                return None
            term = multiplier + {8: 'oxidanyl', 16: 'sulfanyl', 34: 'selanyl', 52: 'tellanyl'}[element]
        word = enclose_name_unit(word) + term
    keep = set(groups[senior]) | {nitrogens[senior].GetIdx()}
    rw = Chem.RWMol(mol)
    for idx in sorted(set(range(mol.GetNumAtoms()))-keep, reverse=True):
        rw.RemoveAtom(idx)
    fragment = _sanitize(rw, Chem)
    if fragment is None:
        return None
    from .api import mol_to_iupac
    parent = mol_to_iupac(fragment)
    if not parent.endswith('amine') or ' ' in parent:
        return None
    name = join_prefix_parent(format_prefixes([('N', word)], hierarchical_enclosures=True), parent)
    return _decision(mol, name, (nitrogens[senior].GetIdx(),), 'P-15.3.2.4.2/P-44.3/P-68.4.2.4')


def hydride_polyamines(mol, Chem):
    """Keep the higher-numbered amine parent on borane or silane."""
    if not _clean(mol, Chem, {5, 6, 7, 9, 14, 17, 35, 53}) or mol.GetRingInfo().NumRings():
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {5, 14}]
    if len(centers) != 1:
        return None
    center = centers[0]
    valence, parent = {5: (3, 'borane'), 14: (4, 'silane')}[center.GetAtomicNum()]
    nitrogens = list(center.GetNeighbors())
    count = len(nitrogens)
    if (not 2 <= count <= valence or center.GetTotalValence() != valence
            or center.GetTotalNumHs(includeNeighbors=True) != valence - count
            or any(n.GetAtomicNum() != 7 or n.GetDegree() not in {2, 3}
                   or n.GetTotalNumHs(includeNeighbors=True) != 3 - n.GetDegree()
                   or _bond(mol, center.GetIdx(), n.GetIdx()) != 1
                   or any(a.GetAtomicNum() != 6 for a in n.GetNeighbors() if a.GetIdx() != center.GetIdx())
                   for n in nitrogens)
            or sum(a.GetAtomicNum() == 7 for a in mol.GetAtoms()) != count):
        return None
    from .component_mapping import mapped_branch
    from .formatting import format_prefixes, multiplied_suffix
    groups, occupied = [], {center.GetIdx(), *(n.GetIdx() for n in nitrogens)}
    for n in nitrogens:
        words = []
        for atom in n.GetNeighbors():
            if atom.GetIdx() == center.GetIdx():
                continue
            word, mapped = mapped_branch(mol, Chem, atom.GetIdx(), n.GetIdx())
            group = set(mapped)
            if (not word or not group or group & occupied
                    or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 9, 17, 35, 53} for i in group)
                    or any(neighbor.GetIdx() not in group and neighbor.GetIdx() != n.GetIdx()
                           for i in group for neighbor in mol.GetAtomWithIdx(i).GetNeighbors())):
                return None
            occupied.update(group)
            words.append(word)
        groups.append(tuple(sorted(words)))
    if occupied != set(range(mol.GetNumAtoms())) or len(set(groups)) != 1:
        return None
    prefixes = [('N' + "'" * i, word) for i, words in enumerate(groups) for word in words]
    suffix = multiplied_suffix(count, 'amine')
    if not suffix:
        return None
    return _decision(mol, format_prefixes(prefixes, hierarchical_enclosures=True) + parent + suffix,
                     (n.GetIdx() for n in nitrogens), 'P-67.1.2.6.2/P-44.1.1')


def simple_large_heterocycle(mol, Chem):
    """A large heterocycle with mapped branches and ring characteristic groups."""
    valences = {6: 4, 7: 3, 8: 2, 16: 2, 34: 2, 52: 2, 15: 3, 33: 3,
                51: 3, 83: 3, 14: 4, 32: 4, 50: 4, 82: 4, 5: 3}
    halogens = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in set(valences) | set(halogens)
                   or a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.HasQuery()
                   for a in mol.GetAtoms())
            or any(b.HasQuery() for b in mol.GetBonds())):
        return None
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) != 1 or not 11 <= len(rings[0]) <= 9999:
        return None
    ring = set(rings[0])
    ring_bonds = [b for b in mol.GetBonds()
                  if b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring]
    if (any(b.GetBondTypeAsDouble() not in {1, 2, 3} for b in ring_bonds)
            or any(b.GetBondTypeAsDouble() != 1
                   and not (b.GetBondTypeAsDouble() == 2
                            and ((b.GetBeginAtomIdx() in ring and b.GetBeginAtom().GetAtomicNum() == 6
                                  and b.GetEndAtom().GetAtomicNum() in {8, 16} and b.GetEndAtom().GetDegree() == 1)
                                 or (b.GetEndAtomIdx() in ring and b.GetEndAtom().GetAtomicNum() == 6
                                     and b.GetBeginAtom().GetAtomicNum() in {8, 16} and b.GetBeginAtom().GetDegree() == 1)))
                   for b in mol.GetBonds()
                   if b.GetBeginAtomIdx() not in ring or b.GetEndAtomIdx() not in ring)):
        return None
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in valences
           or mol.GetAtomWithIdx(i).GetTotalValence() != valences[mol.GetAtomWithIdx(i).GetAtomicNum()]
           for i in ring):
        return None
    from .component_mapping import mapped_branch
    branches, characteristic, owned = [], [], set(ring)
    substituted_amino = False
    for host in ring:
        for atom in mol.GetAtomWithIdx(host).GetNeighbors():
            root = atom.GetIdx()
            if root in ring:
                continue
            if root in owned:
                return None
            if mol.GetAtomWithIdx(host).GetAtomicNum() == 6 and atom.GetDegree() == 1:
                z, order, h = atom.GetAtomicNum(), _bond(mol, host, root), atom.GetTotalNumHs(includeNeighbors=True)
                kind = {(8, 2, 0): 'one', (16, 2, 0): 'thione', (8, 1, 1): 'ol',
                        (16, 1, 1): 'thiol', (7, 1, 2): 'amine'}.get((z, order, h))
                if kind:
                    characteristic.append((kind, host, root))
                    owned.add(root)
                    continue
            if atom.GetAtomicNum() in halogens and atom.GetDegree() == 1:
                word, group = halogens[atom.GetAtomicNum()], {root}
            elif atom.GetAtomicNum() == 6:
                word, mapped = mapped_branch(mol, Chem, root, host)
                group = set(mapped)
                if (not group or not word or root not in group
                        or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, *halogens} for i in group)
                        or any(n.GetIdx() not in group and n.GetIdx() != host
                               for i in group for n in mol.GetAtomWithIdx(i).GetNeighbors())):
                    return None
            elif atom.GetAtomicNum() in {7, 8, 16, 34, 52}:
                word, mapped = mapped_branch(mol, Chem, root, host)
                group = set(mapped)
                if (not group or not word or root not in group or atom.IsInRing()
                        or _bond(mol, host, root) != 1
                        or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, *halogens}
                               for i in group if i != root)
                        or any(n.GetIdx() not in group and n.GetIdx() != host
                               for i in group for n in mol.GetAtomWithIdx(i).GetNeighbors())):
                    return None
                if atom.GetAtomicNum() == 7:
                    substituted_amino = True
            else:
                return None
            if group & owned:
                return None
            owned.update(group)
            branches.append((host, word))
    if owned != set(range(mol.GetNumAtoms())):
        return None
    from .formatting import attach_suffix, format_prefixes, join_prefix_parent, multiplied_suffix, prefix_alphabetization_key
    adjacency = {i: {n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in ring}
                 for i in ring}
    elements = {i: mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring}
    unsaturated = any(b.GetBondTypeAsDouble() > 1 for b in ring_bonds)
    from .large_heteromonocycle_methods import large_heteromonocycle_frontier
    n = len(ring)
    frontier = large_heteromonocycle_frontier(adjacency, elements, max_atoms=n,
                                               max_states=max(10000, 2*n*n + 2*n),
                                               max_candidates=max(128, 2*n))
    if frontier.status != 'complete' or not frontier.candidates:
        return None
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings
    from .parent_annotations import isotope_numbering_key, stereo_numbering
    annotated = (any(a.GetIsotope() or int(a.GetChiralTag()) for a in mol.GetAtoms())
                 or any(int(b.GetStereo()) for b in mol.GetBonds()))
    priority = ('one', 'thione', 'ol', 'thiol', 'amine')
    prefix_of = {'one': 'oxo', 'thione': 'sulfanylidene', 'ol': 'hydroxy',
                 'thiol': 'sulfanyl', 'amine': 'amino'}
    principal = next((kind for kind in priority if any(k == kind for k, _, _ in characteristic)), None)
    if substituted_amino and principal not in {'one', 'thione', 'ol', 'thiol'}:
        return None
    labelled_groups = [(kind, host, root) for kind, host, root in characteristic
                       if mol.GetAtomWithIdx(root).GetIsotope()]
    principal_sites = [(host, root) for kind, host, root in characteristic if kind == principal]
    labelled_principal = [(host, root) for kind, host, root in labelled_groups if kind == principal]
    if labelled_principal:
        # Among otherwise identical one-atom characteristic groups, modified
        # groups are senior; the higher mass breaks ties (P-82.2.2.2).
        mass = max(mol.GetAtomWithIdx(root).GetIsotope() for _host, root in labelled_principal)
        principal_sites = [(host, root) for host, root in labelled_principal
                           if mol.GetAtomWithIdx(root).GetIsotope() == mass]
    principal_roots = {root for _host, root in principal_sites}
    choices = []
    for row in frontier.candidates:
        locants = dict(row.atom_locants)
        prefixes = [(str(locants[host]), word) for host, word in branches]
        for kind, host, root in characteristic:
            if root in principal_roots:
                continue
            atom = mol.GetAtomWithIdx(root)
            isotope = f'({atom.GetIsotope()}{atom.GetSymbol()})' if atom.GetIsotope() else ''
            prefixes.append((str(locants[host]), isotope + prefix_of[kind]))
        by_word = {word: tuple(sorted(int(loc) for loc, value in prefixes if value == word))
                   for _loc, word in prefixes}
        parts = []
        for term in row.replacement_terms:
            # A single replacement locant may be omitted for the bare parent.
            # Once other positions are specified, retain 1 so that their
            # relation to the heteroatom is explicit (P-14.3.4.4/P-22.2.3).
            needs_locant = unsaturated or bool(branches or characteristic) or annotated
            locations = row.hetero_locants if term.omit_locant and needs_locant else term.locants
            parts.append((','.join(map(str, locations)) + '-' if locations else '') + term.term)
        parent = '-'.join(parts) + row.base_hydride
        if unsaturated:
            from .unsaturated_ring_parents import UnsaturationState, render_unsaturated_parent
            edges = []
            for bond in ring_bonds:
                order = int(bond.GetBondTypeAsDouble())
                if order > 1:
                    a, b = sorted((int(locants[bond.GetBeginAtomIdx()]),
                                   int(locants[bond.GetEndAtomIdx()])))
                    edges.append((a, b, order))
            edges = tuple(sorted(edges))
            state = UnsaturationState(edges, tuple(sorted(a for a, _b, _o in edges)),
                                      tuple(sorted(a for a, _b, o in edges if o == 2)),
                                      tuple(sorted(a for a, _b, o in edges if o == 3)))
            try:
                parent = render_unsaturated_parent(parent, state)
            except ValueError:
                return None
            unsaturation_key = (state.multiple_bond_locants, state.double_bond_locants,
                                state.edge_locants)
        else:
            unsaturation_key = ((), (), ())
        suffix_locants = tuple(sorted(int(locants[host]) for host, _root in principal_sites))
        if suffix_locants:
            suffix = multiplied_suffix(len(suffix_locants), principal)
            if not suffix:
                return None
            parent = attach_suffix(parent, suffix, suffix_locants)
            if labelled_principal:
                suffix_atom = mol.GetAtomWithIdx(principal_sites[0][1])
                isotope = f'({suffix_atom.GetIsotope()}{suffix_atom.GetSymbol()})'
                if len(principal_sites) == 1:
                    decorated_suffix = isotope + suffix
                else:
                    from .constants import MULT
                    from .formatting import enclose_name_unit
                    decorated_suffix = MULT[len(principal_sites)] + enclose_name_unit(isotope + principal)
                parent = parent[:-len(suffix)] + decorated_suffix
        isotope_key = isotope_numbering_key(mol, hydrogen_bindings(mol, locants),
                                            include_substituent_groups=True) if annotated else ()
        stereo_key = stereo_numbering(mol, locants) if annotated else ()
        if isotope_key is None or stereo_key is None:
            return None
        key = (suffix_locants, unsaturation_key, tuple(sorted(int(loc) for loc, _word in prefixes)),
               tuple((prefix_alphabetization_key(word), by_word[word])
                     for word in sorted(by_word, key=prefix_alphabetization_key)), isotope_key, stereo_key)
        if annotated:
            local = frozenset(root for _kind, _host, root in labelled_groups)
            bindings = tuple(row.atom_locants) + tuple((i, '') for i in local)
            component = MappedComponent(parent, bindings, frozenset(ring) | local,
                                        tuple(prefixes), frozenset(owned - ring - local),
                                        systematic_parent=True, local_isotope_atoms=local)
            name = decorate_component(mol, Chem, component)
        else:
            prefix_text = format_prefixes(prefixes, hierarchical_enclosures=True)
            # A single replacement term may omit locant 1. Its initial
            # letters attach directly to an ordinary locanted prefix; a
            # replacement term starting with a locant requires a hyphen.
            name = prefix_text + parent if prefix_text and parent[0].isalpha() else join_prefix_parent(prefix_text, parent)
        if not name:
            return None
        choices.append((key, name))
    best = min(key for key, _name in choices)
    names = {name for key, name in choices if key == best}
    if len(names) != 1:
        return None
    return _decision(mol, names.pop(), (), 'P-22.2.3.1/P-22.2.3.2')


def nitrosilane(mol, Chem):
    """Nitro and carbon-only ligands on one silicon hydride parent."""
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, 8, 14} or a.GetIsotope()
                   or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) for b in mol.GetBonds())):
        return None
    silicon = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 14]
    nitrogen = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    oxygen = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(silicon) != 1 or not 1 <= len(nitrogen) <= 4 or len(oxygen) != 2 * len(nitrogen):
        return None
    si = silicon[0]
    if si.IsInRing() or si.GetTotalValence() != 4:
        return None
    from .component_mapping import mapped_branch
    from .phosphorus_ester_methods import _carbon_component
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    owned = {si.GetIdx()}
    for n in nitrogen:
        os = [a for a in n.GetNeighbors() if a.GetAtomicNum() == 8]
        if (n.GetFormalCharge() != 1 or n.GetDegree() != 3
                or _bond(mol, si.GetIdx(), n.GetIdx()) != 1 or len(os) != 2
                or sorted(_bond(mol, n.GetIdx(), o.GetIdx()) for o in os) != [1, 2]
                or any(o.GetFormalCharge() != (-1 if _bond(mol, n.GetIdx(), o.GetIdx()) == 1 else 0)
                       or o.GetDegree() != 1 for o in os)):
            return None
        group = {n.GetIdx(), *(o.GetIdx() for o in os)}
        if group & owned:
            return None
        owned.update(group)
    if any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetIdx() not in owned):
        return None
    units = [PrefixUnit('nitro', 'simple') for _ in nitrogen]
    for atom in si.GetNeighbors():
        if atom.GetIdx() in owned:
            continue
        if atom.GetAtomicNum() != 6 or _bond(mol, si.GetIdx(), atom.GetIdx()) != 1:
            return None
        group = _carbon_component(mol, atom.GetIdx(), si.GetIdx())
        text, mapped = mapped_branch(mol, Chem, atom.GetIdx(), si.GetIdx())
        if not group or frozenset(mapped) != group or group & owned:
            return None
        path = all(sum(x.GetIdx() in group for x in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2 for i in group)
        units.append(PrefixUnit(text, 'simple' if text in {'phenyl', 'tert-butyl'} or path else 'compound'))
        owned.update(group)
    if owned != set(range(mol.GetNumAtoms())):
        return None
    seq = mononuclear_prefix_sequence(units)
    return _decision(mol, seq.text + 'silane', (si.GetIdx(),), 'P-61.5.1/P-16.5.1.3.1') if seq else None


def ring_thialdehyde(mol, Chem):
    """A ring-attached -CH=S on an otherwise carbon/heteroring parent."""
    if not _clean(mol, Chem, {6, 7, 8, 9, 16, 17, 35, 53}):
        return None
    sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    if len(sulfurs) != 1:
        return None
    s = sulfurs[0]
    if s.GetDegree() != 1:
        return None
    c = s.GetNeighbors()[0]
    if (c.GetAtomicNum() != 6 or c.IsInRing() or c.GetDegree() != 2
            or c.GetTotalNumHs(includeNeighbors=True) != 1 or _bond(mol, c.GetIdx(), s.GetIdx()) != 2):
        return None
    roots = [a for a in c.GetNeighbors() if a.GetIdx() != s.GetIdx()]
    if len(roots) != 1 or not roots[0].IsInRing() or _bond(mol, c.GetIdx(), roots[0].GetIdx()) != 1:
        return None
    if any(a.GetAtomicNum() == 8 and any(b.GetBondTypeAsDouble() != 1 for b in a.GetBonds())
           for a in mol.GetAtoms()):
        return None
    if any(a.GetAtomicNum() == 7 and not a.IsInRing()
           and any(b.GetBondTypeAsDouble() != 1 for b in a.GetBonds()) for a in mol.GetAtoms()):
        return None
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(s.GetIdx()).SetAtomicNum(8)
    probe = _sanitize(rw, Chem)
    if probe is None:
        return None
    from .api import mol_to_iupac
    base = mol_to_iupac(probe)
    if base.endswith('benzaldehyde'):
        name = base[:-len('benzaldehyde')] + 'benzenecarbothialdehyde'
    elif base.endswith('carbaldehyde'):
        name = base[:-len('carbaldehyde')] + 'carbothialdehyde'
    else:
        return None
    return _decision(mol, name, (c.GetIdx(),), 'P-66.6.3/P-41')


def _amine_sulfide_via_oxide(mol, Chem):
    """Transfer an oxide's proved amine parent to its sulfur analogue."""
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()
            or any(a.GetNumRadicalElectrons() or a.HasQuery()
                   or int(a.GetChiralTag()) and a.GetAtomicNum() != 6 for a in mol.GetAtoms())
            or any(b.HasQuery() for b in mol.GetBonds())):
        return None
    sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    if len(sulfurs) != 1:
        return None
    sulfur = sulfurs[0]
    if sulfur.GetFormalCharge() != -1 or sulfur.GetDegree() != 1 or sulfur.GetIsotope():
        return None
    n = sulfur.GetNeighbors()[0]
    if (n.GetAtomicNum() != 7 or n.IsInRing() or n.GetFormalCharge() != 1 or n.GetTotalValence() != 4
            or _bond(mol, n.GetIdx(), sulfur.GetIdx()) != 1
            or any(a.GetAtomicNum() != 6 or _bond(mol, n.GetIdx(), a.GetIdx()) != 1
                   for a in n.GetNeighbors() if a.GetIdx() != sulfur.GetIdx())):
        return None
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(sulfur.GetIdx()).SetAtomicNum(8)
    oxide = _sanitize(rw, Chem)
    if oxide is None:
        return None
    if (any(int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) for b in mol.GetBonds())):
        if not _stereo_unchanged_after_removal(mol, oxide, set(), Chem):
            return None
    from .api import mol_to_iupac
    base = mol_to_iupac(oxide)
    if base.endswith(' N-oxide'):
        return _decision(mol, base[:-len('oxide')] + 'sulfide', (n.GetIdx(),),
                         'P-62.5/P-41', 'functional-class')
    # Some isotope-bearing oxides use a functional-parent spelling even
    # though the single amine itself has a mapped substitutive name.
    if (sum(a.GetAtomicNum() == 7 for a in mol.GetAtoms()) != 1 or n.GetIsotope()
            or any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetIdx() not in {n.GetIdx(), sulfur.GetIdx()})):
        return None
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(n.GetIdx()).SetFormalCharge(0)
    rw.RemoveAtom(sulfur.GetIdx())
    neutral = _sanitize(rw, Chem)
    if neutral is None:
        return None
    if (any(int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) for b in mol.GetBonds())):
        if not _stereo_unchanged_after_removal(mol, neutral, {sulfur.GetIdx()}, Chem):
            return None
    parent = mol_to_iupac(neutral)
    if not parent.endswith(('amine', 'aniline')) or ' ' in parent:
        return None
    return _decision(mol, parent + ' N-sulfide', (n.GetIdx(),), 'P-62.5/P-41', 'functional-class')


def amine_sulfide(mol, Chem):
    """A single nonring amine sulfide with carbon-only substituents."""
    general = _amine_sulfide_via_oxide(mol, Chem)
    if general:
        return general
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, 16} or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) for b in mol.GetBonds())):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    ss = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    if len(ns) != 1 or len(ss) != 1:
        return None
    n, s = ns[0], ss[0]
    if (n.IsInRing() or n.GetFormalCharge() != 1 or s.GetFormalCharge() != -1
            or n.GetDegree() not in {2, 3, 4} or n.GetTotalValence() != 4
            or s.GetDegree() != 1 or _bond(mol, n.GetIdx(), s.GetIdx()) != 1
            or any(a.GetAtomicNum() != 6 for a in n.GetNeighbors() if a.GetIdx() != s.GetIdx())
            or any(_bond(mol, n.GetIdx(), a.GetIdx()) != 1 for a in n.GetNeighbors())
            or any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetIdx() not in {n.GetIdx(), s.GetIdx()})):
        return None
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(n.GetIdx()).SetFormalCharge(0)
    rw.RemoveAtom(s.GetIdx())
    neutral = _sanitize(rw, Chem)
    if neutral is None:
        return None
    from .api import mol_to_iupac
    base = mol_to_iupac(neutral)
    if not base.endswith(('amine', 'aniline')) or ' ' in base:
        return None
    return _decision(mol, base + ' N-sulfide', (n.GetIdx(),), 'P-62.5/P-41', 'functional-class')


def silyl_nitrogen_ring(mol, Chem):
    """A silyl group on a selected nitrogen heterocyclic ring system."""
    if not _clean(mol, Chem, {6, 7, 9, 14, 17, 35, 53}):
        return None
    silicons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 14]
    rings = mol.GetRingInfo().AtomRings()
    if len(silicons) != 1 or not rings:
        return None
    si = silicons[0]
    ring = set().union(*map(set, rings))
    roots = [a for a in si.GetNeighbors() if a.GetIdx() in ring]
    if (si.GetTotalValence() != 4 or len(roots) != 1
            or not any(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in ring)
            or _bond(mol, si.GetIdx(), roots[0].GetIdx()) != 1):
        return None
    from .component_mapping import mapped_branch
    from .formatting import enclose_name_unit, prefix_alphabetization_key
    word, owned = mapped_branch(mol, Chem, si.GetIdx(), roots[0].GetIdx())
    owned = set(owned)
    if not word or ring & owned:
        return None
    if word == 'tms':
        word = 'trimethylsilyl'
    if not word.endswith('silyl'):
        return None
    # A fused or spiro ring system must be one connected ring assembly, not
    # separate rings joined by a linker with an unresolved parent comparison.
    connected = {roots[0].GetIdx()}
    while True:
        expanded = connected | set().union(*(set(r) for r in rings if connected & set(r)))
        if expanded == connected:
            break
        connected = expanded
    if connected != ring:
        return None
    from .phosphorus_ester_methods import _carbon_component
    halogens = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    seen, branches = set(owned) | ring, []
    for host in ring:
        for atom in mol.GetAtomWithIdx(host).GetNeighbors():
            root = atom.GetIdx()
            if root in ring or root == si.GetIdx():
                continue
            if root in seen or _bond(mol, host, root) != 1:
                return None
            if atom.GetAtomicNum() in halogens and atom.GetDegree() == 1:
                text, group = halogens[atom.GetAtomicNum()], {root}
            elif atom.GetAtomicNum() == 6:
                group = _carbon_component(mol, root, host)
                text, mapped = mapped_branch(mol, Chem, root, host)
                if not group or not text or group != frozenset(mapped):
                    return None
                group = set(group)
            else:
                return None
            if group & seen:
                return None
            seen.update(group)
            branches.append(text)
    if seen != set(range(mol.GetNumAtoms())):
        return None
    base_rw = Chem.RWMol(mol)
    host = roots[0].GetIdx()
    for idx in sorted(owned, reverse=True):
        base_rw.RemoveAtom(idx)
    host -= sum(idx < host for idx in owned)
    from .api import mol_to_iupac
    options = (('trifluoromethyl', 6), ('methyl', 6), ('ethyl', 6),
               ('fluoro', 9), ('chloro', 17), ('bromo', 35), ('iodo', 53)) if branches else (('bromo', 35),)
    for marker_name, element in options:
        marker_key, word_key = prefix_alphabetization_key(marker_name), prefix_alphabetization_key(word)
        # Every other prefix must lie on the same side of the real and the
        # proxy group in citation order; otherwise a symmetric ring can be
        # numbered in the opposite direction by the probe.
        if any(marker_name in text or prefix_alphabetization_key(text) in {marker_key, word_key}
               or (prefix_alphabetization_key(text) < marker_key) != (prefix_alphabetization_key(text) < word_key)
               for text in branches):
            continue
        rw = Chem.RWMol(base_rw)
        marker = rw.AddAtom(Chem.Atom(element))
        rw.AddBond(host, marker, Chem.BondType.SINGLE)
        if marker_name == 'ethyl':
            tail = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(marker, tail, Chem.BondType.SINGLE)
        elif marker_name == 'trifluoromethyl':
            for _ in range(3):
                fluorine = rw.AddAtom(Chem.Atom(9))
                rw.AddBond(marker, fluorine, Chem.BondType.SINGLE)
        probe = _sanitize(rw, Chem)
        if probe is None:
            continue
        base = mol_to_iupac(probe)
        if base.count(marker_name) != 1 or ' ' in base or 'benzene' in base:
            continue
        prefix = word if word == 'silyl' else enclose_name_unit(word)
        enclosed_marker = '(' + marker_name + ')'
        name = (base.replace(enclosed_marker, prefix, 1) if enclosed_marker in base
                else base.replace(marker_name, prefix, 1))
        return _decision(mol, name, (), 'P-44.1.2.1/P-58.2.1')
    return None


def _isotopic_peroxyanhydride_acid(fragment, center_index, bridge_index, Chem):
    """Render one labeled carbon, phosphorus, or chalcogen acid component."""
    center = fragment.GetAtomWithIdx(center_index)
    if center.GetAtomicNum() not in {6, 15, 16, 34, 52}:
        return ''
    bridge = fragment.GetAtomWithIdx(bridge_index)
    oxos = [a for a in center.GetNeighbors() if a.GetAtomicNum() == 8
            and _bond(fragment, center_index, a.GetIdx()) == 2]
    allowed = {bridge_index, *(a.GetIdx() for a in oxos)}
    if center.GetAtomicNum() in {15, 16, 34, 52}:
        allowed.add(center_index)
    labelled = [a for a in fragment.GetAtoms() if a.GetIsotope()]
    oxygen_labels = [a.GetIsotope() for a in labelled if a.GetAtomicNum() == 8]
    all_oxygens = [a for a in fragment.GetAtoms() if a.GetAtomicNum() == 8]
    fully_modified_oxygen = (len(oxygen_labels) > 1 and len(set(oxygen_labels)) == 1
                             and len(oxygen_labels) == len(all_oxygens))
    if (not labelled or any(a.GetIdx() not in allowed for a in labelled)
            or (not fully_modified_oxygen and
                (len(oxygen_labels) > 2 or len(set(oxygen_labels)) != len(oxygen_labels)))
            or any(int(a.GetChiralTag()) for a in fragment.GetAtoms())
            or any(int(b.GetStereo()) for b in fragment.GetBonds())):
        return ''
    plain = Chem.Mol(fragment)
    for atom in plain.GetAtoms():
        atom.SetIsotope(0)
    from .api import mol_to_iupac
    acid = mol_to_iupac(plain)
    expected = ({6: ('oic acid', 'carboxylic acid', 'acetic acid', 'formic acid'),
                 15: ('phosphinic acid',), 16: ('sulfonic acid', 'sulfinic acid'),
                 34: ('selenonic acid', 'seleninic acid'),
                 52: ('telluronic acid', 'tellurinic acid')})[center.GetAtomicNum()]
    if not acid.endswith(expected):
        return ''
    # P-82.3.2 orders different nuclides of one element by increasing mass.
    # P-82.6.1.3 omits locants when every oxygen position in this component
    # bears the same nuclide. Other repeated-mass patterns require further
    # positional/multiplication evidence and remain outside this producer.
    parts = []
    if fully_modified_oxygen:
        parts.append((oxygen_labels[0], str(oxygen_labels[0]) + 'O' + str(len(oxygen_labels))))
    else:
        if bridge.GetIsotope():
            parts.append((bridge.GetIsotope(), 'O-' + str(bridge.GetIsotope()) + 'O'))
        for oxygen in oxos:
            if oxygen.GetIsotope():
                parts.append((oxygen.GetIsotope(), str(oxygen.GetIsotope()) + 'O' + ('1' if len(oxos) > 1 else '')))
    parts = [part for _mass, part in sorted(parts)]
    if center.GetIsotope():
        parts.append(str(center.GetIsotope()) + center.GetSymbol())
    if not parts:
        return ''
    descriptor = '(' + ','.join(parts) + ')'
    if center.GetAtomicNum() == 15:
        return acid[:-len('phosphinic acid')] + descriptor + 'phosphinic acid'
    return descriptor + acid


def noncarbon_peroxyanhydride(mol, Chem):
    """Name a neutral peroxide linking two independently identified acids."""
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()
            or any((a.GetIsotope() and a.GetAtomicNum() not in {6, 8, 15, 16, 34, 52}) or a.GetNumRadicalElectrons()
                   or a.HasQuery() for a in mol.GetAtoms())
            or any(b.HasQuery() for b in mol.GetBonds())):
        return None
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    pairs = [(bond.GetBeginAtom(), bond.GetEndAtom()) for bond in mol.GetBonds()
             if bond.GetBondTypeAsDouble() == 1 and bond.GetBeginAtom().GetAtomicNum() == 8
             and bond.GetEndAtom().GetAtomicNum() == 8]
    if len(pairs) != 1:
        return None
    a, b = pairs[0]
    if any(o.GetFormalCharge() or o.GetDegree() != 2 for o in (a, b)):
        return None

    def acid_center(bridge, other):
        neighbors = [n for n in bridge.GetNeighbors() if n.GetIdx() != other.GetIdx()]
        if len(neighbors) != 1:
            return None
        center = neighbors[0]
        ci = center.GetIdx()
        if center.IsInRing() or _bond(mol, ci, bridge.GetIdx()) != 1:
            return None
        rest = [n for n in center.GetNeighbors() if n.GetIdx() != bridge.GetIdx()]
        z = center.GetAtomicNum()
        oxo = [n for n in rest if n.GetAtomicNum() == 8 and n.GetDegree() == 1
               and _bond(mol, ci, n.GetIdx()) == 2 and not n.GetFormalCharge()]
        carbon = [n for n in rest if n.GetAtomicNum() == 6 and _bond(mol, ci, n.GetIdx()) == 1]
        hydrogen = center.GetTotalNumHs(includeNeighbors=True)
        if z == 7:
            anions = [n for n in rest if n.GetAtomicNum() == 8 and n.GetDegree() == 1
                      and _bond(mol, ci, n.GetIdx()) == 1 and n.GetFormalCharge() == -1]
            if center.GetFormalCharge() == 1 and len(oxo) == len(anions) == 1 and len(rest) == 2:
                return center, 'nitric', {ci, anions[0].GetIdx()}
            if not center.GetFormalCharge() and len(oxo) == 1 and len(rest) == 1 and not hydrogen:
                return center, 'nitrous', set()
            return None
        halogen_acids = {
            9: ('hypofluorous', 'fluorous', 'fluoric', 'perfluoric'),
            17: ('hypochlorous', 'chlorous', 'chloric', 'perchloric'),
            35: ('hypobromous', 'bromous', 'bromic', 'perbromic'),
            53: ('hypoiodous', 'iodous', 'iodic', 'periodic'),
        }
        if z in halogen_acids:
            anions = [n for n in rest if n.GetAtomicNum() == 8 and n.GetDegree() == 1
                      and _bond(mol, ci, n.GetIdx()) == 1 and n.GetFormalCharge() == -1]
            count = len(oxo) + len(anions)
            if (len(rest) == count and count <= 3 and not hydrogen
                    and center.GetFormalCharge() == len(anions)):
                return center, 'acid:' + halogen_acids[z][count], {ci, *(n.GetIdx() for n in anions)}
            return None
        if center.GetFormalCharge():
            return None
        if z == 6 and len(oxo) == 1 and len(carbon) + hydrogen == 1 and len(rest) == len(carbon) + 1:
            return center, 'carbon', set()
        if z == 5 and len(carbon) + hydrogen == 2 and len(rest) == len(carbon):
            return center, 'boron', set()
        if (z in {15, 33, 51} and len(oxo) == 1 and len(carbon) + hydrogen == 2
                and len(rest) == len(carbon) + 1):
            return center, 'pnictogen', set()
        if (z in {16, 34, 52} and len(oxo) in {1, 2} and len(carbon) == 1
                and len(rest) == len(oxo) + 1 and not hydrogen):
            return center, 'chalcogen', set()
        return None

    sites = (acid_center(a, b), acid_center(b, a))
    if any(site is None for site in sites) or all(site[1] == 'carbon' for site in sites):
        return None
    centers = [site[0] for site in sites]
    charged_atoms = set().union(*(site[2] for site in sites))
    if any(atom.GetFormalCharge() and atom.GetIdx() not in charged_atoms for atom in mol.GetAtoms()):
        return None
    # A remaining acidic OH makes this an acidic anhydride, for which the
    # preferred construction is different. Ordinary alcohols can remain.
    for oxygen in oxygens:
        if oxygen.GetDegree() != 1 or oxygen.GetTotalNumHs(includeNeighbors=True) != 1:
            continue
        host = oxygen.GetNeighbors()[0]
        if host.GetAtomicNum() in {5, 7, 14, 15, 16, 33, 34, 51, 52}:
            return None
        if host.GetAtomicNum() == 6 and any(n.GetAtomicNum() == 8 and _bond(mol, host.GetIdx(), n.GetIdx()) == 2
                                             for n in host.GetNeighbors()):
            return None
    rw = Chem.RWMol(mol)
    rw.RemoveBond(a.GetIdx(), b.GetIdx())
    for oxygen in (a, b):
        capped = rw.GetAtomWithIdx(oxygen.GetIdx())
        capped.SetNumExplicitHs(1)
        capped.SetNoImplicit(True)
    broken = _sanitize(rw, Chem)
    if broken is None:
        return None
    if (any(int(a.GetChiralTag()) for a in mol.GetAtoms()) or any(int(b.GetStereo()) for b in mol.GetBonds())):
        if not _stereo_unchanged_after_removal(mol, broken, set(), Chem):
            return None
    index_fragments = Chem.GetMolFrags(broken)
    frags = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
    if len(frags) != 2:
        return None
    from .api import mol_to_iupac
    acids = []
    for indices, fragment in zip(index_fragments, frags, strict=True):
        matches = [site for site in sites if site[0].GetIdx() in indices]
        if len(matches) != 1:
            return None
        kind = matches[0][1]
        noncarbon_isotopes = [atom for atom in fragment.GetAtoms() if atom.GetIsotope()
                              and atom.GetAtomicNum() != 6]
        if noncarbon_isotopes:
            source_center = matches[0][0].GetIdx()
            bridge_index = next((o.GetIdx() for o in (a, b) if o.GetIdx() in indices), None)
            if bridge_index is None:
                return None
            acid = _isotopic_peroxyanhydride_acid(fragment, indices.index(source_center),
                                                  indices.index(bridge_index), Chem)
        elif kind.startswith('acid:'):
            acid = kind[len('acid:'):] + ' acid'
        else:
            acid = kind + ' acid' if kind in {'nitric', 'nitrous'} else mol_to_iupac(fragment)
        if not acid.endswith(' acid'):
            return None
        import re
        isotope_counts = Counter((atom.GetIsotope(), atom.GetSymbol())
                                 for atom in fragment.GetAtoms() if atom.GetIsotope())
        for (mass, symbol), count in isotope_counts.items():
            matches = re.findall(rf'(?<!\d){mass}{re.escape(symbol)}(\d*)', acid)
            if sum(int(value or '1') for value in matches) != count:
                return None
        if any(int(atom.GetChiralTag()) for atom in fragment.GetAtoms()) or any(int(bond.GetStereo()) for bond in fragment.GetBonds()):
            from .stereochemistry import assigned_atom_stereochemistry, assigned_bond_stereochemistry
            expected = Counter(label for _index, label in assigned_atom_stereochemistry(fragment, Chem, atom_order='idx', include_pseudo=True))
            expected.update(label for _a, _b, label in assigned_bond_stereochemistry(fragment, Chem))
            reported = Counter(re.findall(r'\d+[a-z]?([RSEZrs])(?=[,)])', acid))
            if any(reported[label] < count for label, count in expected.items()):
                return None
        acids.append(acid)
    if (acids[0] != acids[1] and all(any(a.GetIsotope() for a in fragment.GetAtoms()) for fragment in frags)
            and any(a.GetIsotope() and a.GetAtomicNum() != 6 for a in mol.GetAtoms())):
        identities = []
        for fragment in frags:
            plain = Chem.Mol(fragment)
            for atom in plain.GetAtoms():
                atom.SetIsotope(0)
            identities.append(Chem.MolToSmiles(plain, canonical=True, isomericSmiles=True))
        if identities[0] == identities[1]:
            return None  # Different isotope patterns on the same acid need a full ordering proof.
    if acids[0] == acids[1] and Chem.MolToSmiles(frags[0]) != Chem.MolToSmiles(frags[1]):
        return None
    from .formatting import prefix_alphabetization_key
    words = sorted({acid[:-len(' acid')] for acid in acids}, key=prefix_alphabetization_key)
    return _decision(mol, ' '.join(words) + ' peroxyanhydride',
                     (atom.GetIdx() for atom in centers), 'P-67.1.3.3/P-65.7.4', 'functional-class')


def nitramide(mol, Chem):
    """Nitramides with carbon, silyl, or nitro nitrogen substituents."""
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 14, 17, 35, 53} or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) for b in mol.GetBonds())):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetFormalCharge() == 0
               and any(n.GetAtomicNum() == 7 and n.GetFormalCharge() == 1 for n in a.GetNeighbors())]
    if len(centers) != 1:
        return None
    center = centers[0]
    nitros = [a for a in center.GetNeighbors() if a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1]
    if center.IsInRing() or center.GetTotalValence() != 3 or not 1 <= len(nitros) <= 2:
        return None
    core = {center.GetIdx()}
    for n in nitros:
        os = [a for a in n.GetNeighbors() if a.GetAtomicNum() == 8]
        if (n.GetDegree() != 3 or _bond(mol, center.GetIdx(), n.GetIdx()) != 1 or len(os) != 2
                or sorted(_bond(mol, n.GetIdx(), o.GetIdx()) for o in os) != [1, 2]
                or sorted(o.GetFormalCharge() for o in os) != [-1, 0]
                or any(o.GetFormalCharge() != (-1 if _bond(mol, n.GetIdx(), o.GetIdx()) == 1 else 0)
                       for o in os)
                or any(o.GetDegree() != 1 for o in os)):
            return None
        core.update((n.GetIdx(), *(o.GetIdx() for o in os)))
    if any(a.GetFormalCharge() and a.GetIdx() not in core for a in mol.GetAtoms()):
        return None
    from .component_mapping import mapped_branch
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    units = [PrefixUnit('nitro', 'simple')] if len(nitros) == 2 else []
    for atom in center.GetNeighbors():
        if atom.GetIdx() in core:
            continue
        if atom.GetAtomicNum() not in {6, 14} or _bond(mol, center.GetIdx(), atom.GetIdx()) != 1:
            return None
        text, mapped = mapped_branch(mol, Chem, atom.GetIdx(), center.GetIdx())
        group = set(mapped)
        if (not text or not group or core & group
                or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 9, 14, 17, 35, 53} for i in group)
                or any(n.GetIdx() not in group and n.GetIdx() != center.GetIdx()
                       for i in group for n in mol.GetAtomWithIdx(i).GetNeighbors())):
            return None
        silicon = [i for i in group if mol.GetAtomWithIdx(i).GetAtomicNum() == 14]
        if silicon:
            if atom.GetAtomicNum() != 14 or len(silicon) != 1 or atom.GetTotalValence() != 4:
                return None
            if text == 'tms':
                text = 'trimethylsilyl'
            if not text.endswith('silyl'):
                return None
            units.append(PrefixUnit(text, 'simple' if text == 'silyl' else 'compound'))
            core.update(group)
            continue
        path = all(sum(x.GetIdx() in group for x in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2 for i in group)
        has_halo = any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in group)
        units.append(PrefixUnit(text, 'simple' if not has_halo and (text in {'phenyl', 'tert-butyl'} or path)
                                else 'compound'))
        core.update(group)
    if core != set(range(mol.GetNumAtoms())):
        return None
    seq = mononuclear_prefix_sequence(units) if units else None
    if units and seq is None:
        return None
    return _decision(mol, (seq.text if seq else '') + 'nitramide', (center.GetIdx(),), 'P-62.4/P-41')


def unsaturated_carbon_spiro(mol, Chem):
    """A two-ring carbon spiro parent, numbered before its branches."""
    halogens = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    if not _clean(mol, Chem, {6, *halogens}) or any(b.GetBondTypeAsDouble() not in {1, 2} for b in mol.GetBonds()):
        return None
    rings = tuple(map(set, mol.GetRingInfo().AtomRings()))
    if (len(rings) != 2 or len(rings[0] & rings[1]) != 1
            or not any(b.GetBondTypeAsDouble() == 2 for b in mol.GetBonds())):
        return None
    atoms = rings[0] | rings[1]
    if (any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms)
            or not any(b.GetBondTypeAsDouble() == 2 and b.GetBeginAtomIdx() in atoms
                       and b.GetEndAtomIdx() in atoms for b in mol.GetBonds())):
        return None
    from .component_mapping import mapped_branch
    from .phosphorus_ester_methods import _carbon_component
    branches, owned = [], set(atoms)
    for host in atoms:
        for atom in mol.GetAtomWithIdx(host).GetNeighbors():
            root = atom.GetIdx()
            if root in atoms:
                continue
            if root in owned or _bond(mol, host, root) != 1:
                return None
            if atom.GetAtomicNum() in halogens and atom.GetDegree() == 1:
                word, group = halogens[atom.GetAtomicNum()], {root}
            elif atom.GetAtomicNum() == 6:
                group = _carbon_component(mol, root, host)
                word, mapped = mapped_branch(mol, Chem, root, host)
                if not group or not word or group != frozenset(mapped):
                    return None
                group = set(group)
            else:
                return None
            if group & owned:
                return None
            owned.update(group)
            branches.append((host, word))
    if owned != set(range(mol.GetNumAtoms())):
        return None
    from .polycyclic_rules import _multiple_bond_locant_edges, _spiro_parent_sequence_candidates
    from .formatting import format_prefixes, join_prefix_parent, prefix_alphabetization_key
    graph = {i: {n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in atoms}
             for i in atoms}
    rows = _spiro_parent_sequence_candidates(mol, atoms, graph)
    if not rows:
        return None
    keys = []
    for parent, seq in rows:
        locants = {atom: str(index) for index, atom in enumerate(seq, 1)}
        prefixes = [(locants[host], word) for host, word in branches]
        by_word = {word: tuple(sorted(int(loc) for loc, value in prefixes if value == word))
                   for _loc, word in prefixes}
        key = (tuple(sorted(_multiple_bond_locant_edges(mol, seq))),
               tuple(sorted(int(loc) for loc, _word in prefixes)),
               tuple((prefix_alphabetization_key(word), by_word[word])
                     for word in sorted(by_word, key=prefix_alphabetization_key)))
        name = join_prefix_parent(format_prefixes(prefixes, hierarchical_enclosures=True), parent)
        keys.append((key, name))
    best = min(key for key, _name in keys)
    names = {name for key, name in keys if key == best}
    if len(names) != 1:
        return None
    return _decision(mol, names.pop(), (), 'P-24.2.1/P-31.1.5.1.1')


def partially_hydrogenated_biphenyl(mol, Chem):
    """A partially reduced biphenyl, including hydroxyls on both rings."""
    halogens = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    if not _clean(mol, Chem, {6, 8, *halogens}):
        return None
    rings = tuple(map(tuple, mol.GetRingInfo().AtomRings()))
    if len(rings) != 2 or any(len(r) != 6 for r in rings) or set(rings[0]) & set(rings[1]):
        return None
    aromatic = [ring for ring in rings if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)]
    if len(aromatic) != 1:
        return None
    reduced = next(r for r in rings if r != aromatic[0])
    if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in reduced):
        return None
    aromatic_ring = set(aromatic[0])
    reduced_ring = set(reduced)
    ring = aromatic_ring | reduced_ring
    # Hydrogenation changes bond orders and hydrogen population, never the
    # elements of the biphenyl skeleton. Oxygen is permitted only outside it.
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring):
        return None
    junctions = [b for b in mol.GetBonds()
                 if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} & aromatic_ring
                 and {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} & reduced_ring]
    if len(junctions) != 1 or junctions[0].GetBondTypeAsDouble() != 1:
        return None
    bond = junctions[0]
    roots = [next(i for i in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) if i in group)
             for group in (reduced_ring, aromatic_ring)]
    ring_bonds = [b for b in mol.GetBonds()
                  if b.GetBeginAtomIdx() in reduced_ring and b.GetEndAtomIdx() in reduced_ring]
    if (any(b.GetBondTypeAsDouble() not in {1, 2} for b in ring_bonds)
            or sum(b.GetBondTypeAsDouble() == 2 for b in ring_bonds) not in {1, 2}):
        return None
    saturated = {i for i in reduced_ring if all(b.GetBondTypeAsDouble() == 1
                 for b in mol.GetAtomWithIdx(i).GetBonds() if b.GetOtherAtomIdx(i) in reduced_ring)}
    if len(saturated) not in {2, 4}:
        return None
    from .component_mapping import mapped_branch
    from .phosphorus_ester_methods import _carbon_component
    branches, hydroxyls, owned = [], [], set(ring)
    for host in ring:
        for atom in mol.GetAtomWithIdx(host).GetNeighbors():
            root = atom.GetIdx()
            if root in ring:
                continue
            if root in owned or _bond(mol, host, root) != 1:
                return None
            if (atom.GetAtomicNum() == 8 and atom.GetDegree() == 1
                    and atom.GetTotalNumHs(includeNeighbors=True) == 1):
                hydroxyls.append(host)
                owned.add(root)
                continue
            if atom.GetAtomicNum() in halogens and atom.GetDegree() == 1:
                word, group = halogens[atom.GetAtomicNum()], {root}
            elif atom.GetAtomicNum() == 6:
                group = _carbon_component(mol, root, host)
                word, mapped = mapped_branch(mol, Chem, root, host)
                if not group or not word or group != frozenset(mapped):
                    return None
                group = set(group)
            else:
                return None
            if group & owned:
                return None
            owned.update(group)
            branches.append((host, word))
    if owned != set(range(mol.GetNumAtoms())):
        return None
    if hydroxyls and (not any(i in reduced_ring for i in hydroxyls)
                      or not any(i in aromatic_ring for i in hydroxyls)):
        return None
    from .constants import MULT
    from .formatting import _locant_sort_key, format_prefixes, join_prefix_parent, prefix_alphabetization_key
    orders = []
    for root, group in zip(roots, (reduced_ring, aromatic_ring)):
        rows = []
        for first in (n.GetIdx() for n in mol.GetAtomWithIdx(root).GetNeighbors() if n.GetIdx() in group):
            order = [root, first]
            while len(order) < 6:
                options = [n.GetIdx() for n in mol.GetAtomWithIdx(order[-1]).GetNeighbors()
                           if n.GetIdx() in group and n.GetIdx() not in order]
                if len(options) != 1:
                    break
                order.append(options[0])
            if len(order) == 6:
                rows.append(order)
        if not rows:
            return None
        orders.append(rows)
    choices = []
    for first in orders[0]:
        for second in orders[1]:
            for swap in ((False, True) if hydroxyls else (False,)):
                locants = {atom: str(i) + ("'" if swap else '') for i, atom in enumerate(first, 1)}
                locants.update({atom: str(i) + ('' if swap else "'") for i, atom in enumerate(second, 1)})
                hydro = tuple(sorted((locants[i] for i in saturated), key=_locant_sort_key))
                suffix = tuple(sorted((locants[i] for i in hydroxyls), key=_locant_sort_key))
                prefixes = [(locants[host], word) for host, word in branches]
                by_word = {word: tuple(sorted((_locant_sort_key(loc) for loc, value in prefixes if value == word)))
                           for _loc, word in prefixes}
                key = (tuple(_locant_sort_key(loc) for loc in suffix),
                       tuple(_locant_sort_key(loc) for loc in hydro),
                       tuple(sorted(_locant_sort_key(loc) for loc, _word in prefixes)),
                       tuple((prefix_alphabetization_key(word), by_word[word])
                             for word in sorted(by_word, key=prefix_alphabetization_key)))
                hydro_text = ','.join(hydro) + '-' + MULT[len(saturated)] + 'hydro'
                if hydroxyls:
                    multiplier = MULT.get(len(hydroxyls))
                    if not multiplier:
                        return None
                    ending = multiplier.removesuffix('a') + 'ol'
                    parent = hydro_text + "[1,1'-biphenyl]-" + ','.join(suffix) + '-' + ending
                else:
                    parent = hydro_text + "-1,1'-biphenyl"
                choices.append((key, join_prefix_parent(format_prefixes(prefixes, hierarchical_enclosures=True), parent)))
    best = min(key for key, _name in choices)
    names = {name for key, name in choices if key == best}
    if len(names) != 1:
        return None
    return _decision(mol, names.pop(), (), 'P-31.2.3.3.5.1/P-54.3', 'ring-assembly')


def acyclic_anilic_acid(mol, Chem):
    """A single terminal N-phenyl amide on an acyclic carboxylic acid."""
    if not _clean(mol, Chem, {6, 7, 8, 9, 17, 35, 53}):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(ns) != 1 or sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()) < 3:
        return None
    if sum(a.GetAtomicNum() == 8 and any(b.GetBondTypeAsDouble() == 2 for b in a.GetBonds())
           for a in mol.GetAtoms()) != 2:
        return None
    n = ns[0]
    if n.IsInRing() or n.GetDegree() != 2 or n.GetTotalNumHs(includeNeighbors=True) != 1:
        return None
    aryl = [a for a in n.GetNeighbors() if a.GetAtomicNum() == 6 and a.GetIsAromatic()]
    carbonyl = [a for a in n.GetNeighbors() if a.GetAtomicNum() == 6 and not a.IsInRing()
                and any(o.GetAtomicNum() == 8 and _bond(mol, a.GetIdx(), o.GetIdx()) == 2 for o in a.GetNeighbors())]
    if len(aryl) != 1 or len(carbonyl) != 1:
        return None
    acid_centers = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.IsInRing():
            continue
        doubles = [o for o in atom.GetNeighbors() if o.GetAtomicNum() == 8 and o.GetDegree() == 1
                   and _bond(mol, atom.GetIdx(), o.GetIdx()) == 2]
        hydroxy = [o for o in atom.GetNeighbors() if o.GetAtomicNum() == 8 and o.GetDegree() == 1
                   and o.GetTotalNumHs(includeNeighbors=True) == 1 and _bond(mol, atom.GetIdx(), o.GetIdx()) == 1]
        if len(doubles) == len(hydroxy) == 1:
            acid_centers.append(atom.GetIdx())
    if len(acid_centers) != 1:
        return None
    from .phosphorus_ester_methods import _carbon_component
    group = _carbon_component(mol, aryl[0].GetIdx(), n.GetIdx())
    if not group or len(group) != 6 or any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in group):
        return None
    if any(a.IsInRing() and a.GetIdx() not in group for a in mol.GetAtoms()):
        return None
    rw = Chem.RWMol(mol)
    for idx in sorted(group, reverse=True):
        rw.RemoveAtom(idx)
    probe = _sanitize(rw, Chem)
    if probe is None:
        return None
    from .api import mol_to_iupac
    base = mol_to_iupac(probe)
    if base == 'oxamic acid':
        name = 'anilino(oxo)acetic acid'
    elif base.endswith('oic acid') and base.count('amino') == 1 and 'oxo' in base:
        name = base.replace('amino', 'anilino', 1)
    else:
        return None
    return _decision(mol, name, acid_centers, 'P-65.1.6.2/P-41')


def _parse_acyclic_acid_prefixes(name):
    """Recover complete locanted prefix units from an acyclic acid probe."""
    import re
    from .constants import MULT
    from .prefix_presentation import _balanced_end, _LOCANT_FIELD, _SIMPLE_MULTIPLIERS, _SIMPLE_UNITS
    inverse = {word: count for count, word in MULT.items()}
    complex_mult = {'bis': 2, 'tris': 3, 'tetrakis': 4, 'pentakis': 5, 'hexakis': 6}
    rows, position = [], 0
    while True:
        field = _LOCANT_FIELD.match(name, position)
        if field is None:
            break
        locants = field.group()[:-1].split(',')
        if any(not loc.isdigit() for loc in locants):
            return None
        start = field.end()
        mult = next((p for p in complex_mult if name.startswith(p, start)
                     and name[start + len(p):start + len(p) + 1] in '([{'), '')
        if mult:
            start += len(mult)
        if name[start:start+1] in ('(', '[', '{'):
            end = _balanced_end(name, start)
            if end is None:
                return None
            unit, count = name[start+1:end-1], complex_mult.get(mult, 1)
        else:
            if mult:
                return None
            simple = next((p for p in _SIMPLE_MULTIPLIERS if name.startswith(p, start)
                           and any(name.startswith(u, start + len(p)) for u in _SIMPLE_UNITS)), '')
            unit_start = start + len(simple)
            unit = next((u for u in _SIMPLE_UNITS if name.startswith(u, unit_start)), '')
            if not unit:
                return None
            count = inverse.get(simple, 1)
            end = unit_start + len(unit)
        if len(locants) != count:
            return None
        rows.extend((locant, unit) for locant in locants)
        position = end
        if position < len(name) and name[position] == '-' and _LOCANT_FIELD.match(name, position + 1):
            position += 1
            continue
        break
    parent = name[position:]
    root = r'(?:meth|eth|prop|but|pent|hex|hept|oct|non|dec|undec|dodec|tridec|tetradec|pentadec|hexadec|heptadec|octadec|nonadec|icos)'
    if not re.fullmatch(root + r'(?:an|(?:-\d+(?:,\d+)*-)?(?:di|tri|tetra)?(?:en|yn))'
                        r'(?:-\d+(?:,\d+)*-(?:di|tri|tetra)?(?:en|yn))*oic acid', parent):
        return None
    return rows, parent


def _preferred_acyl_peroxy(mol, Chem, root, remote, group):
    """Name a wholly mapped acyl-peroxy branch using its acid component."""
    from .api import mol_to_iupac
    from .bluebook_lambda_amine_methods import _acid_to_acyl
    from .formatting import enclose_name_unit
    import re
    acid_rw = Chem.RWMol(mol)
    capped = acid_rw.GetAtomWithIdx(remote)
    capped.SetNumExplicitHs(0)
    capped.SetNoImplicit(False)
    for index in sorted(set(range(mol.GetNumAtoms())) - (group - {root}), reverse=True):
        acid_rw.RemoveAtom(index)
    acid = _sanitize(acid_rw, Chem)
    if acid is None:
        return ''
    acid_name = mol_to_iupac(acid)
    acyl = (acid_name[:-len('acetic acid')] + 'acetyl'
            if acid_name.endswith('acetic acid') else _acid_to_acyl(acid_name))
    if not acyl:
        return ''
    simple = re.fullmatch(r'(?:formyl|acetyl|benzoyl|[a-z]+anoyl)', acyl)
    return (acyl if simple else enclose_name_unit(acyl)) + 'peroxy'


def acidic_carbon_peroxyanhydride(mol, Chem):
    """Select a free carboxylic acid over a carbon peroxyanhydride."""
    if not _clean(mol, Chem, {6, 7, 8, 9, 17, 35, 53}) or mol.GetRingInfo().NumRings():
        return None
    acid_centers = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        oxo = [o for o in atom.GetNeighbors() if o.GetAtomicNum() == 8 and o.GetDegree() == 1
               and _bond(mol, atom.GetIdx(), o.GetIdx()) == 2]
        hydroxy = [o for o in atom.GetNeighbors() if o.GetAtomicNum() == 8 and o.GetDegree() == 1
                   and o.GetTotalNumHs(includeNeighbors=True) == 1
                   and _bond(mol, atom.GetIdx(), o.GetIdx()) == 1]
        if len(oxo) == len(hydroxy) == 1:
            acid_centers.append(atom.GetIdx())
    bridges = [b for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 1
               and b.GetBeginAtom().GetAtomicNum() == b.GetEndAtom().GetAtomicNum() == 8]
    if len(acid_centers) != 1 or len(bridges) != 1:
        return None
    from .component_mapping import mapped_branch
    from .formatting import format_prefixes, join_prefix_parent
    from .api import mol_to_iupac
    bridge = bridges[0]
    choices = []
    for root in (bridge.GetBeginAtom(), bridge.GetEndAtom()):
        hosts = [a for a in root.GetNeighbors() if a.GetAtomicNum() == 6
                 and any(o.GetAtomicNum() == 8 and _bond(mol, a.GetIdx(), o.GetIdx()) == 2
                         for o in a.GetNeighbors())]
        if root.GetDegree() != 2 or len(hosts) != 1:
            continue
        host = hosts[0]
        word, mapped = mapped_branch(mol, Chem, root.GetIdx(), host.GetIdx())
        group = set(mapped)
        if (not word or not word.endswith('peroxy') or root.GetIdx() not in group
                or acid_centers[0] in group or any(a.GetAtomicNum() == 9 and a.GetIdx() not in group
                                                   for a in mol.GetAtoms())):
            continue
        # The remote peroxide oxygen must itself be attached to a carbonyl,
        # so this is an acyl peroxyanhydride rather than an ordinary peroxide.
        remote = bridge.GetOtherAtom(root)
        if (remote.GetDegree() != 2 or not any(a.GetAtomicNum() == 6
                and any(o.GetAtomicNum() == 8 and _bond(mol, a.GetIdx(), o.GetIdx()) == 2
                        for o in a.GetNeighbors()) for a in remote.GetNeighbors())):
            continue
        # Derive the acyl group from its actual acid, rather than accepting
        # the more general carbonyl-substituent spelling returned by a branch
        # writer for e.g. chloroacetyl.
        word = _preferred_acyl_peroxy(mol, Chem, root.GetIdx(), remote.GetIdx(), group)
        if not word:
            continue
        rw = Chem.RWMol(mol)
        marker = rw.GetAtomWithIdx(root.GetIdx())
        marker.SetAtomicNum(9)
        marker.SetNoImplicit(True)
        for index in sorted(group - {root.GetIdx()}, reverse=True):
            rw.RemoveAtom(index)
        proxy = _sanitize(rw, Chem)
        if proxy is None:
            continue
        parsed = _parse_acyclic_acid_prefixes(mol_to_iupac(proxy))
        if parsed is None:
            continue
        prefixes, parent = parsed
        if sum(value == 'fluoro' for _locant, value in prefixes) != 1:
            continue
        prefixes = [(locant, word if value == 'fluoro' else value) for locant, value in prefixes]
        choices.append(join_prefix_parent(format_prefixes(prefixes, hierarchical_enclosures=True), parent))
    if len(set(choices)) != 1:
        return None
    return _decision(mol, choices[0], acid_centers, 'P-65.7.8.1/P-63.3.1/P-41')


def acidic_noncarbon_peroxyanhydride(mol, Chem):
    """An acyl-peroxy branch on a boron or phosphorus oxoacid parent."""
    if not _clean(mol, Chem, {5, 6, 8, 9, 15, 17, 35, 53}):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {5, 15}]
    if len(centers) != 1:
        return None
    center = centers[0]
    oh, oxo, links, carbon = [], [], [], []
    for a in center.GetNeighbors():
        if a.GetAtomicNum() == 6 and _bond(mol, center.GetIdx(), a.GetIdx()) == 1:
            carbon.append(a)
            continue
        if a.GetAtomicNum() != 8:
            return None
        bond = _bond(mol, center.GetIdx(), a.GetIdx())
        if a.GetDegree() == 1 and a.GetTotalNumHs(includeNeighbors=True) == 1 and bond == 1:
            oh.append(a.GetIdx())
        elif a.GetDegree() == 1 and bond == 2:
            oxo.append(a.GetIdx())
        elif a.GetDegree() == 2 and bond == 1:
            links.append(a)
        else:
            return None
    if (len(oh) not in {1, 2} or len(oh) + len(carbon) != 2 or len(links) != 1
            or len(oxo) != (1 if center.GetAtomicNum() == 15 else 0)):
        return None
    root = links[0]
    remote = [a for a in root.GetNeighbors() if a.GetIdx() != center.GetIdx()]
    if len(remote) != 1 or remote[0].GetAtomicNum() != 8 or remote[0].GetDegree() != 2:
        return None
    acyl = [a for a in remote[0].GetNeighbors() if a.GetIdx() != root.GetIdx()]
    if (len(acyl) != 1 or acyl[0].GetAtomicNum() != 6
            or not any(a.GetAtomicNum() == 8 and _bond(mol, acyl[0].GetIdx(), a.GetIdx()) == 2
                       for a in acyl[0].GetNeighbors())):
        return None
    group, pending = set(), [root.GetIdx()]
    while pending:
        current = pending.pop()
        if current in group:
            continue
        group.add(current)
        pending.extend(a.GetIdx() for a in mol.GetAtomWithIdx(current).GetNeighbors()
                       if a.GetIdx() != center.GetIdx() and a.GetIdx() not in group)
    core = {center.GetIdx(), *oh, *oxo}
    organyl = ''
    if carbon:
        from .component_mapping import mapped_branch
        organyl, mapped = mapped_branch(mol, Chem, carbon[0].GetIdx(), center.GetIdx())
        side = set(mapped)
        if (not organyl or not side or side & (core | group)
                or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 9, 17, 35, 53} for i in side)
                or any(a.GetIdx() not in side and a.GetIdx() != center.GetIdx()
                       for i in side for a in mol.GetAtomWithIdx(i).GetNeighbors())):
            return None
        core.update(side)
    if root.GetIdx() not in group or group & core or group | core != set(range(mol.GetNumAtoms())):
        return None
    word = _preferred_acyl_peroxy(mol, Chem, root.GetIdx(), remote[0].GetIdx(), group)
    if not word:
        return None
    from .formatting import enclose_name_unit, prefix_alphabetization_key
    if organyl:
        import re
        units = [(word, True), (organyl, re.fullmatch(r'[a-z]+', organyl) is None)]
        units.sort(key=lambda row: prefix_alphabetization_key(row[0]))
        text = ''.join(enclose_name_unit(value) if i or compound else value
                       for i, (value, compound) in enumerate(units))
        parent = 'phosphinic acid' if center.GetAtomicNum() == 15 else 'borinic acid'
    else:
        text = enclose_name_unit(word)
        parent = 'phosphonic acid' if center.GetAtomicNum() == 15 else 'boronic acid'
    return _decision(mol, text + parent, (center.GetIdx(),),
                     'P-67.3.1/P-65.7.8.1/P-41')


def carbon_acid_sulfur_peroxyanhydride(mol, Chem):
    """Use a carboxylic acid parent with an acylperoxy chalcogen prefix."""
    if not _clean(mol, Chem, {6, 8, 9, 16, 17, 34, 35, 52, 53}) or mol.GetRingInfo().NumRings():
        return None
    sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {16, 34, 52}]
    if len(sulfurs) != 1:
        return None
    sulfur = sulfurs[0]
    oxo = [a for a in sulfur.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
           and _bond(mol, sulfur.GetIdx(), a.GetIdx()) == 2]
    carbons = [a for a in sulfur.GetNeighbors() if a.GetAtomicNum() == 6
               and _bond(mol, sulfur.GetIdx(), a.GetIdx()) == 1]
    links = [a for a in sulfur.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 2
             and _bond(mol, sulfur.GetIdx(), a.GetIdx()) == 1]
    if (len(oxo) not in {1, 2} or len(carbons) != 1 or len(links) != 1
            or sulfur.GetDegree() != len(oxo) + 2):
        return None
    host, root = carbons[0], links[0]
    remote = [a for a in root.GetNeighbors() if a.GetIdx() != sulfur.GetIdx()]
    if len(remote) != 1 or remote[0].GetAtomicNum() != 8 or remote[0].GetDegree() != 2:
        return None
    acyl = [a for a in remote[0].GetNeighbors() if a.GetIdx() != root.GetIdx()]
    if (len(acyl) != 1 or acyl[0].GetAtomicNum() != 6
            or not any(a.GetAtomicNum() == 8 and _bond(mol, acyl[0].GetIdx(), a.GetIdx()) == 2
                       for a in acyl[0].GetNeighbors())):
        return None
    group, pending = set(), [root.GetIdx()]
    while pending:
        index = pending.pop()
        if index in group:
            continue
        group.add(index)
        pending.extend(a.GetIdx() for a in mol.GetAtomWithIdx(index).GetNeighbors()
                       if a.GetIdx() != sulfur.GetIdx() and a.GetIdx() not in group)
    branch = group | {sulfur.GetIdx(), *(a.GetIdx() for a in oxo)}
    if host.GetIdx() in branch:
        return None
    acids = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetIdx() not in branch
             and any(o.GetAtomicNum() == 8 and o.GetDegree() == 1 and _bond(mol, a.GetIdx(), o.GetIdx()) == 2
                     for o in a.GetNeighbors())
             and any(o.GetAtomicNum() == 8 and o.GetDegree() == 1 and o.GetTotalNumHs(includeNeighbors=True) == 1
                     and _bond(mol, a.GetIdx(), o.GetIdx()) == 1 for o in a.GetNeighbors())]
    if (len(acids) != 1 or acids[0].GetIdx() == host.GetIdx()
            or any(a.GetAtomicNum() == 9 and a.GetIdx() not in branch for a in mol.GetAtoms())):
        return None
    word = _preferred_acyl_peroxy(mol, Chem, root.GetIdx(), remote[0].GetIdx(), group)
    if not word:
        return None
    from .formatting import enclose_name_unit, format_prefixes, join_prefix_parent
    chalcogen_term = {16: ('sulfinyl', 'sulfonyl'), 34: ('seleninyl', 'selenonyl'),
                      52: ('tellurinyl', 'telluronyl')}[sulfur.GetAtomicNum()][len(oxo)-1]
    prefix = enclose_name_unit(word) + chalcogen_term
    rw = Chem.RWMol(mol)
    marker = rw.GetAtomWithIdx(sulfur.GetIdx())
    marker.SetAtomicNum(9)
    marker.SetNumExplicitHs(0)
    marker.SetNoImplicit(True)
    for index in sorted(branch - {sulfur.GetIdx()}, reverse=True):
        rw.RemoveAtom(index)
    proxy = _sanitize(rw, Chem)
    if proxy is None:
        return None
    from .api import mol_to_iupac
    base = mol_to_iupac(proxy)
    if base == 'fluoroacetic acid':
        name = enclose_name_unit(prefix) + 'acetic acid'
    else:
        parsed = _parse_acyclic_acid_prefixes(base)
        if parsed is None:
            return None
        rows, parent = parsed
        if sum(word == 'fluoro' for _locant, word in rows) != 1:
            return None
        rows = [(locant, prefix if word == 'fluoro' else word) for locant, word in rows]
        name = join_prefix_parent(format_prefixes(rows, hierarchical_enclosures=True), parent)
    return _decision(mol, name, (acids[0].GetIdx(),), 'P-65.7.8.1/P-67.1.4.4/P-41')


def substituted_anilic_acid(mol, Chem):
    """Use the substitutable anilino prefix on a terminal acyclic amide."""
    if not _clean(mol, Chem, {6, 7, 8, 9, 17, 35, 53}):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and not a.IsInRing()
          and any(n.GetAtomicNum() == 6 and n.GetIsAromatic() for n in a.GetNeighbors())
          and any(n.GetAtomicNum() == 6 and not n.IsInRing()
                  and any(o.GetAtomicNum() == 8 and _bond(mol, n.GetIdx(), o.GetIdx()) == 2
                          for o in n.GetNeighbors()) for n in a.GetNeighbors())]
    if len(ns) != 1:
        return None
    nitrogen = ns[0]
    if nitrogen.IsInRing() or nitrogen.GetDegree() not in {2, 3}:
        return None
    aryl = [a for a in nitrogen.GetNeighbors() if a.GetAtomicNum() == 6 and a.GetIsAromatic()]
    carbonyl = [a for a in nitrogen.GetNeighbors() if a.GetAtomicNum() == 6 and not a.IsInRing()
                and any(o.GetAtomicNum() == 8 and _bond(mol, a.GetIdx(), o.GetIdx()) == 2
                        for o in a.GetNeighbors())]
    if len(aryl) != 1 or len(carbonyl) != 1:
        return None
    acid_centers = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.IsInRing():
            continue
        oxo = [o for o in atom.GetNeighbors() if o.GetAtomicNum() == 8 and o.GetDegree() == 1
               and _bond(mol, atom.GetIdx(), o.GetIdx()) == 2]
        oh = [o for o in atom.GetNeighbors() if o.GetAtomicNum() == 8 and o.GetDegree() == 1
              and o.GetTotalNumHs(includeNeighbors=True) == 1 and _bond(mol, atom.GetIdx(), o.GetIdx()) == 1]
        if len(oxo) == len(oh) == 1:
            acid_centers.append(atom.GetIdx())
    if len(acid_centers) != 1:
        return None
    rings = mol.GetRingInfo().AtomRings()
    if (len(rings) != 1 or len(rings[0]) != 6 or aryl[0].GetIdx() not in rings[0]
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 or not mol.GetAtomWithIdx(i).GetIsAromatic()
                   for i in rings[0])):
        return None
    ni, ci = nitrogen.GetIdx(), carbonyl[0].GetIdx()
    cut = Chem.RWMol(mol)
    cut.RemoveBond(ni, ci)
    capped_n = cut.GetAtomWithIdx(ni)
    capped_n.SetNumExplicitHs(0)
    capped_n.SetNoImplicit(False)
    separated = _sanitize(cut, Chem)
    if separated is None:
        return None
    index_fragments = Chem.GetMolFrags(separated)
    fragments = Chem.GetMolFrags(separated, asMols=True, sanitizeFrags=True)
    if len(fragments) != 2:
        return None
    choices = [(set(ids), fragment) for ids, fragment in zip(index_fragments, fragments, strict=True) if ni in ids]
    if len(choices) != 1 or ci in choices[0][0]:
        return None
    side, amine = choices[0]
    from .api import mol_to_iupac
    aniline = mol_to_iupac(amine)
    if aniline.endswith('aniline') and ' ' not in aniline:
        prefix = aniline[:-1] + 'o'
    else:
        # Hydroxy on the phenyl ring can make the free amine a phenol, but
        # the corresponding substituted anilino prefix remains available.
        import re
        from .component_mapping import mapped_branch
        text, mapped = mapped_branch(mol, Chem, ni, ci)
        match = re.fullmatch(r'\((.+)phenyl\)amino', text)
        if nitrogen.GetDegree() != 2 or set(mapped) != side or match is None:
            return None
        prefix = match.group(1) + 'anilino'
    probe_rw = Chem.RWMol(mol)
    atom = probe_rw.GetAtomWithIdx(ni)
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(False)
    for idx in sorted(side - {ni}, reverse=True):
        probe_rw.RemoveAtom(idx)
    probe = _sanitize(probe_rw, Chem)
    if probe is None:
        return None
    base = mol_to_iupac(probe)
    from .formatting import enclose_name_unit, format_prefixes, join_prefix_parent
    if base == 'oxamic acid':
        term = prefix if prefix == 'anilino' else enclose_name_unit(prefix)
        name = term + '(oxo)acetic acid'
    else:
        # Recover only independently delimited simple prefixes. The acid
        # writer chooses the carbon parent; this reorders the complete prefix
        # units after replacing amino by the substituted anilino group.
        parsed = _parse_acyclic_acid_prefixes(base)
        if parsed is None:
            return None
        rows, parent = parsed
        rows = [(locant, prefix if word == 'amino' else word) for locant, word in rows]
        if (not rows
                or sum(word == prefix for _loc, word in rows) != 1
                or not any(word == 'oxo' for _loc, word in rows)):
            return None
        name = join_prefix_parent(format_prefixes(rows, hierarchical_enclosures=True), parent)
    return _decision(mol, name, acid_centers, 'P-65.1.6.2/P-62.2.1.1.1/P-41')


def silicon_chain_phosphane(mol, Chem):
    """A contiguous silicon hydride group on a substituted phosphane."""
    halogens = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    if not _clean(mol, Chem, {6, 14, 15, *halogens}) or mol.GetRingInfo().NumRings():
        return None
    ps = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 15]
    sis = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 14]
    if len(ps) != 1 or len(sis) < 2:
        return None
    p = ps[0]
    p_si = [a for a in p.GetNeighbors() if a.GetAtomicNum() == 14]
    if (p.GetTotalValence() != 3 or len(p_si) != 1
            or any(a.GetAtomicNum() not in {6, 14} for a in p.GetNeighbors())):
        return None
    indices = {a.GetIdx() for a in sis}
    root = p_si[0].GetIdx()
    path = [root]
    while len(path) < len(indices):
        choices = [n.GetIdx() for n in mol.GetAtomWithIdx(path[-1]).GetNeighbors()
                   if n.GetIdx() in indices and n.GetIdx() not in path]
        if len(choices) != 1:
            return None
        path.append(choices[0])
    if any(_bond(mol, a.GetIdx(), n.GetIdx()) != 1 for a in mol.GetAtoms() for n in a.GetNeighbors()):
        return None
    if any(a.GetTotalValence() != 4 or sum(n.GetIdx() in indices for n in a.GetNeighbors()) > 2 for a in sis):
        return None
    from .constants import MULT
    if len(path) not in MULT:
        return None
    hydride = MULT[len(path)] + 'silane'
    from .component_mapping import mapped_branch
    from .formatting import format_prefixes
    from .phosphorus_ester_methods import _carbon_component
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    owned = {p.GetIdx(), *indices}
    si_prefixes = []
    for position, host in enumerate(path, 1):
        for atom in mol.GetAtomWithIdx(host).GetNeighbors():
            if atom.GetIdx() in indices or atom.GetIdx() == p.GetIdx():
                continue
            if atom.GetAtomicNum() in halogens and atom.GetDegree() == 1:
                word, group = halogens[atom.GetAtomicNum()], {atom.GetIdx()}
            elif atom.GetAtomicNum() == 6:
                group = _carbon_component(mol, atom.GetIdx(), host)
                word, mapped = mapped_branch(mol, Chem, atom.GetIdx(), host)
                if not group or not word or group != frozenset(mapped):
                    return None
                group = set(group)
            else:
                return None
            if group & owned:
                return None
            owned.update(group)
            si_prefixes.append((str(position), word))
    if si_prefixes:
        component = format_prefixes(si_prefixes, hierarchical_enclosures=True) + hydride[:-1] + '-1-yl'
    else:
        component = hydride[:-1] + 'yl'
    units = [PrefixUnit(component, 'compound' if si_prefixes else 'simple')]
    for atom in p.GetNeighbors():
        if atom.GetIdx() == root:
            continue
        group = _carbon_component(mol, atom.GetIdx(), p.GetIdx())
        word, mapped = mapped_branch(mol, Chem, atom.GetIdx(), p.GetIdx())
        if not group or not word or group != frozenset(mapped) or group & owned:
            return None
        owned.update(group)
        path = all(sum(n.GetIdx() in group for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2 for i in group)
        units.append(PrefixUnit(word, 'simple' if path else 'compound'))
    if owned != set(range(mol.GetNumAtoms())):
        return None
    sequence = mononuclear_prefix_sequence(units)
    return _decision(mol, sequence.text + 'phosphane', (), 'P-44.1.2.1/P-51.4.1.1') if sequence else None


def select(mol, Chem):
    numbers = Counter(a.GetAtomicNum() for a in iter_atoms(mol))
    charged = any(a.GetFormalCharge() for a in iter_atoms(mol))
    if any(a.GetNumRadicalElectrons() for a in iter_atoms(mol)):
        return added_hydrogen_ring_radical(mol, Chem)
    if numbers[1] and numbers[8] and numbers[6]:
        from .oxygen_characteristic_suffixes import hydroxy_hydrogen_parent

        decision = hydroxy_hydrogen_parent(mol, Chem)
        if decision:
            return decision
    if (numbers[6] and 7 not in numbers and 8 not in numbers
            and (any(numbers[z] == 1 for z in (5, 13, 31, 49, 81)) or mol.GetRingInfo().NumRings()
                 and any(numbers[z] == 1 for z in (14, 32, 50, 82)))):
        from .homogeneous_hydrides import organyl_hydride_parent

        name = organyl_hydride_parent(mol, Chem)
        if name:
            return _decision(mol, name, (), 'P-14.3.4.2(a)/P-68.1.5.2.1/P-68.2.6.1')
    if (2 <= mol.GetNumAtoms() <= 8 and not mol.GetRingInfo().NumRings()
            and any(numbers[z] for z in (9, 17, 35, 53))):
        from .homogeneous_hydrides import halogenated_short_parent

        name = halogenated_short_parent(mol)
        if name:
            return _decision(mol, name, (), 'P-14.3.4.2(a,b,d)/P-31.1.2')
    if numbers[6] and mol.GetRingInfo().NumRings() == 1:
        ring = mol.GetRingInfo().AtomRings()[0]
        if len(ring) == numbers[6] and mol.GetNumAtoms() <= len(ring) + 1:
            from .homogeneous_hydrides import simple_carbocycle

            name = simple_carbocycle(mol, Chem, ring)
            if name:
                centers = [a.GetNeighbors()[0].GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() in (8, 16)]
                return _decision(mol, name, centers, 'P-14.3.4.2(c,d)/P-31.1.3')
        if len(ring) == 6 and numbers[8] >= 2:
            from .retained_hydrocarbon_derivatives import benzene_polyol

            result = benzene_polyol(mol, Chem, ring)
            if result:
                return result
    if (numbers[6] and numbers[8] >= 2 and (mol.GetRingInfo().NumRings() == 2
            and (len(numbers) == 2 or any(numbers[z] for z in (9, 17, 35, 53)))
            or len(numbers) == 3 and (any(numbers[z] for z in (16, 34, 52))
                                    or any(numbers[z] == 1 for z in (14, 15, 32, 33, 50, 51, 82, 83)))
            or mol.GetRingInfo().NumRings() == 2 and any(numbers[z] for z in (16, 34, 52)))):
        from .multiplicative_parent_extensions import linked_alcohol_decision

        result = linked_alcohol_decision(mol, Chem)
        if result is not None:
            return result
    if numbers[6] and numbers[8] >= 2 and len(numbers) == 2 and not mol.GetRingInfo().NumRings():
        from .multiplicative_parent_extensions import straight_polyol

        result = straight_polyol(mol, Chem, allow_branches=True)
        if result is not None:
            return result
    if charged:
        if numbers[7] and numbers[16]:
            result = amine_sulfide(mol, Chem)
            if result:
                return result
        if numbers[14] and numbers[7] and numbers[8] >= 2:
            result = nitrosilane(mol, Chem)
            if result:
                return result
        if numbers[7] and numbers[8] >= 2:
            result = nitramide(mol, Chem)
            if result:
                return result
        if numbers[8] >= 2:
            result = noncarbon_peroxyanhydride(mol, Chem)
            if result:
                return result
        if numbers[7]:
            return (polyiminium(mol, Chem) or polyiminium_with_imines(mol, Chem)
                    or iminium_with_substituted_imine(mol, Chem)
                    or monoiminium_with_imines(mol, Chem) or iminium(mol, Chem))
        return None
    if len(numbers) == 1 and (not numbers[6] or 2 <= numbers[6] <= 3):
        if not numbers[6] and mol.GetRingInfo().NumRings() == 1:
            from .saturated_heterocycle_methods import homogeneous_ring_parent

            ring = mol.GetRingInfo().AtomRings()[0]
            result = homogeneous_ring_parent(mol, Chem, ring, next(iter(numbers)))
            if result is not None:
                return result
        from .homogeneous_hydrides import monounsaturated_chain

        name = monounsaturated_chain(mol)
        if name:
            return _decision(mol, name, (), 'P-14.3.4.2(d)/P-31.1.2')
    if numbers[6] >= 20 and mol.GetRingInfo().NumRings() >= 5:
        from .polynaphthylenes import polynaphthylene_decision

        result = polynaphthylene_decision(mol, Chem, mol.GetRingInfo().AtomRings())
        if result is not None:
            return result
    if numbers[7] >= 2:
        from .graph_model import method_bond_counts
        bonds = method_bond_counts(mol, Chem)
        if bonds[6, 7, 2] or bonds[7, 16, 2] or bonds[7, 34, 2] or bonds[7, 52, 2]:
            from .functional_class_queries import amidine_query
            if mol.HasSubstructMatch(amidine_query(Chem)):
                from .amidine_parents import imidic_parent_decision
                result = imidic_parent_decision(mol, Chem)
                if result:
                    return result
    if numbers[7] >= 3 and numbers[6]:
        result = imidic_hydrazide(mol, Chem)
        if result:
            return result
    if numbers[16] and numbers[8] >= 4:
        result = sulfonoperoxoic_acid(mol, Chem)
        if result:
            return result
    if numbers[6] and numbers[8] >= 4:
        result = ((carbon_acid_sulfur_peroxyanhydride(mol, Chem) if numbers[16] or numbers[34] or numbers[52] else None)
                  or (acidic_noncarbon_peroxyanhydride(mol, Chem) if numbers[5] or numbers[15] else None)
                  or (acidic_carbon_peroxyanhydride(mol, Chem) if numbers[8] >= 6 else None))
        if result:
            return result
    if numbers[7] == 2 and any(numbers[z] for z in (8, 16, 34, 52)):
        result = (identical_amines_with_oxy_link(mol, Chem)
                  or differently_substituted_linked_amines(mol, Chem)
                  or different_linear_amines_with_chalcogen_link(mol, Chem))
        if result:
            return result
    if numbers[7] >= 2 and numbers[5] + numbers[14] == 1:
        result = hydride_polyamines(mol, Chem)
        if result:
            return result
    if numbers[16] == 1 and numbers[6] and mol.GetRingInfo().NumRings():
        result = ring_thialdehyde(mol, Chem)
        if result:
            return result
    if numbers[14] == 1 and numbers[7] and mol.GetRingInfo().NumRings():
        result = silyl_nitrogen_ring(mol, Chem)
        if result:
            return result
    if numbers[8] >= 2 and any(numbers[z] for z in (5, 7, 9, 15, 16, 17, 33, 34, 35, 51, 52, 53)):
        result = noncarbon_peroxyanhydride(mol, Chem)
        if result:
            return result
    if numbers[7] >= 1 and numbers[8] >= 3 and numbers[6]:
        result = acyclic_anilic_acid(mol, Chem) or substituted_anilic_acid(mol, Chem)
        if result:
            return result
    if numbers[14] >= 2 and numbers[15] == 1:
        result = silicon_chain_phosphane(mol, Chem)
        if result:
            return result
    if (numbers[6] and sum(numbers[z] for z in (8, 16, 34, 52)) >= 2
            and sum(bool(numbers[z]) for z in (8, 16, 34, 52)) >= 2):
        result = mixed_chalcogen_organic_chain(mol, Chem)
        if result:
            return result
    if mol.GetRingInfo().NumRings() == 2 and numbers[6]:
        result = unsaturated_carbon_spiro(mol, Chem) or partially_hydrogenated_biphenyl(mol, Chem)
        if result:
            return result
        from .primitive_three_face_fusion import polyalene_fusion
        result = polyalene_fusion(mol, Chem)
        if result:
            return result
        from .localized_fused_delta import localized_fused_delta
        result = localized_fused_delta(mol, Chem)
        if result:
            return result
    if mol.GetRingInfo().NumRings() >= 3 and numbers[6]:
        from .primitive_three_face_fusion import retained_peri_fusion
        result = retained_peri_fusion(mol, Chem)
        if result:
            return result
    if mol.GetRingInfo().NumRings() >= 3 and numbers[6]:
        from .primitive_three_face_fusion import (general_component_fusion,
                                                 retained_conflict_replacement,
                                                 three_face_fusion)
        result = general_component_fusion(mol, Chem)
        if not result and numbers[7]:
            result = retained_conflict_replacement(mol, Chem)
        if not result and mol.GetRingInfo().NumRings() == 3:
            result = three_face_fusion(mol, Chem)
        if result:
            return result
    if mol.GetRingInfo().NumRings() == 1 and mol.GetNumAtoms() >= 11 and numbers[6] and len(numbers) > 1:
        return simple_large_heterocycle(mol, Chem)
    return None
