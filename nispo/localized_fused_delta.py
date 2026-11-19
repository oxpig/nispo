"""P-25.7.1.2 localized bonds on completed two-face fused hydrides.

The construction compares all noncumulative pi matchings of the same H state.
A delta witness is emitted only if its bond locants distinguish the input
unlabelled graph from every other sanitized localized isomer. Atom numbering
and the fused component vocabulary come from the existing P-25 frontier.
"""
from collections import Counter
from itertools import combinations
import re


def localized_fused_delta(mol, Chem):
    from .localized_bicyclic_states import requires_explicit_aza_five_five_bonds
    if requires_explicit_aza_five_five_bonds(mol):
        return None
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups() or len(mol.GetRingInfo().AtomRings()) != 2
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 16, 17, 34, 35, 52, 53}
                   or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or b.HasQuery() for b in mol.GetBonds())
            or not any(b.GetBondTypeAsDouble() == 2 and not b.GetIsAromatic() for b in mol.GetBonds())):
        return None
    from .naming_context import MoleculeContext
    from .parent_selection import locant_key
    from .ring_state_parent import _HW_VALENCE, _reference_states, _skeletons, select_ring_state
    from .fused_branch_annotations import mapped_fused_branch
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7, 8, 16} for i in ring):
        return None
    owned, branches, branch_isotopes = set(ring), [], {}
    for host in ring:
        for root in mol.GetAtomWithIdx(host).GetNeighbors():
            if root.GetIdx() in ring:
                continue
            if mol.GetBondBetweenAtoms(host, root.GetIdx()).GetBondTypeAsDouble() != 1:
                return None
            word, mapped = mapped_fused_branch(mol, Chem, root.GetIdx(), host)
            group = set(mapped)
            if (not word or not group or group & owned or root.GetIdx() not in group
                    or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 8, 9, 16, 17, 34, 35, 52, 53}
                           or mol.GetAtomWithIdx(i).IsInRing()
                           for i in group)
                    or any(n.GetIdx() not in group and n.GetIdx() != host
                           for i in group for n in mol.GetAtomWithIdx(i).GetNeighbors())):
                return None
            chalcogens = {i for i in group if mol.GetAtomWithIdx(i).GetAtomicNum() in {8, 16, 34, 52}}
            if chalcogens:
                other = [a for a in root.GetNeighbors() if a.GetIdx() != host]
                if (chalcogens != {root.GetIdx()} or root.GetDegree() != 2 or root.GetTotalValence() != 2
                        or len(other) != 1 or other[0].GetAtomicNum() != 6
                        or mol.GetBondBetweenAtoms(root.GetIdx(), other[0].GetIdx()).GetBondTypeAsDouble() != 1):
                    return None
            labels = Counter((mol.GetAtomWithIdx(i).GetIsotope(), mol.GetAtomWithIdx(i).GetSymbol())
                             for i in group if mol.GetAtomWithIdx(i).GetIsotope())
            for (mass, symbol), count in labels.items():
                occurrences = re.findall(rf'(?<!\d){mass}{re.escape(symbol)}(\d*)', word)
                if sum(int(value or '1') for value in occurrences) != count:
                    return None
            if word in branch_isotopes and branch_isotopes[word] != labels:
                return None
            branch_isotopes[word] = labels
            branches.append((host, word))
            owned.update(group)
    if owned != set(range(mol.GetNumAtoms())):
        return None
    labelled_branch = any(mol.GetAtomWithIdx(i).GetIsotope() for i in owned-ring)
    ctx = MoleculeContext(mol, Chem)
    ordinary = select_ring_state(ctx)
    if ordinary is not None and not ordinary.selected.method.startswith('skeletal-replacement-'):
        # The existing fused-state proof already identifies this hydride
        # without a localized-bond witness; preserve its preferred form.
        return None
    target = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    candidates = []
    for skeleton in _skeletons(ctx):
        if len(skeleton) < 8 or skeleton[7] != 'component-fused':
            continue
        parent, sequence, locants, numbers, edges = skeleton[:5]
        if set(sequence) != ring or any(locant_key(loc) is None for loc in locants):
            continue
        hydrogens = tuple(mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True) + sum(
            int(bond.GetBondTypeAsDouble()) for bond in mol.GetAtomWithIdx(i).GetBonds()
            if bond.GetOtherAtomIdx(i) not in ring) for i in sequence)
        references = [(h, indicated) for h, indicated, _bonds in _reference_states(tuple(numbers), tuple(edges))
                      if h == hydrogens]
        if len(references) != 1:
            continue  # Only a proved maximum-unsaturation hydrogen state is eligible.
        indicated = references[0][1]
        indicated_locants = tuple(sorted((locants[i] for i in indicated), key=locant_key))
        indicated_text = ','.join(locant + 'H' for locant in indicated_locants)
        degree = Counter(i for edge in edges for i in edge)
        occupancy = tuple(_HW_VALENCE[number] - degree[i] - hydrogens[i]
                          for i, number in enumerate(numbers))
        if any(value not in {0, 1} for value in occupancy):
            continue
        required = frozenset(i for i, value in enumerate(occupancy) if value)
        adjacency = {i: {b if a == i else a for a, b in edges if i in (a, b)} & required
                     for i in required}
        count = [0]
        def matchings(remaining):
            count[0] += 1
            if count[0] > 8192:
                return
            if not remaining:
                yield frozenset()
                return
            first = min(remaining)
            for neighbor in sorted(adjacency[first] & remaining):
                for rest in matchings(remaining - {first, neighbor}):
                    yield rest | {tuple(sorted((first, neighbor)))}
        states = []
        for pairs in matchings(required):
            rw = Chem.RWMol()
            for source in mol.GetAtoms():
                atom = Chem.Atom(source.GetAtomicNum())
                atom.SetIsotope(source.GetIsotope())
                atom.SetNumExplicitHs(source.GetTotalNumHs(includeNeighbors=True))
                atom.SetNoImplicit(True)
                rw.AddAtom(atom)
            positions = {source: i for i, source in enumerate(sequence)}
            for original in mol.GetBonds():
                left, right = original.GetBeginAtomIdx(), original.GetEndAtomIdx()
                if left in ring and right in ring:
                    edge = tuple(sorted((positions[left], positions[right])))
                    bond = Chem.BondType.DOUBLE if edge in pairs else Chem.BondType.SINGLE
                else:
                    bond = original.GetBondType()
                rw.AddBond(left, right, bond)
            candidate = rw.GetMol()
            try:
                Chem.SanitizeMol(candidate)
            except (ValueError, RuntimeError):
                continue
            states.append((pairs, Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)))
        if count[0] > 8192:
            continue
        matching = [pairs for pairs, identity in states if identity == target]
        alternatives = [pairs for pairs, identity in states if identity != target]
        if not matching or not alternatives:
            continue
        ordered = sorted(locants, key=locant_key)
        positions = {locant: index for index, locant in enumerate(ordered)}
        if len(positions) != len(locants):
            continue
        def bond_term(edge):
            left, right = sorted((locants[edge[0]], locants[edge[1]]), key=locant_key)
            simple = positions[right] == positions[left] + 1
            text = left if simple else left + '(' + right + ')'
            return (locant_key(left), 0 if simple else 1, locant_key(right)), text
        for pairs in matching:
            for size in range(1, len(pairs) + 1):
                witnesses = []
                for subset in combinations(sorted(pairs), size):
                    if all(not set(subset) <= other for other in alternatives):
                        terms = sorted(map(bond_term, subset))
                        witnesses.append((tuple(term[0] for term in terms), ','.join(term[1] for term in terms)))
                if witnesses:
                    score, descriptor = min(witnesses)
                    hydrogen_key = tuple(locant_key(locant) for locant in indicated_locants)
                    name = ((indicated_text + '-' if indicated_text else '')
                            + 'Δ^(' + descriptor + ')-' + parent)
                    isotope_key = ()
                    if branches or any(mol.GetAtomWithIdx(i).GetIsotope() for i in sequence):
                        from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings
                        from .parent_annotations import isotope_numbering_key
                        mapping = dict(zip(sequence, locants, strict=True))
                        isotope_key = isotope_numbering_key(mol, hydrogen_bindings(mol, mapping),
                                                           include_substituent_groups=True)
                        prefix_rows = tuple((mapping[host], word) for host, word in branches)
                        citation_order = None
                        if labelled_branch:
                            from .formatting import prefix_alphabetization_key
                            distinct = {word for _locant, word in prefix_rows}
                            def isotope_order(word):
                                labels = branch_isotopes[word]
                                return (-sum(labels.values()),
                                        tuple((-mass, symbol, -count)
                                              for (mass, symbol), count in sorted(labels.items(), reverse=True)))
                            citation_order = tuple(sorted(distinct, key=lambda word: (
                                prefix_alphabetization_key(word)[:-1], isotope_order(word),
                                tuple(sorted(locant_key(locant) for locant, value in prefix_rows if value == word)))))
                        component = MappedComponent(name, tuple(mapping.items()), frozenset(sequence),
                                                    prefix_rows, frozenset(owned-ring), systematic_parent=True,
                                                    prefix_citation_order=citation_order)
                        name = decorate_component(mol, Chem, component)
                        if not name or isotope_key is None:
                            continue
                        if branches and not name[:name.index('Δ^')].endswith('-'):
                            name = name.replace('Δ^', '-Δ^', 1)
                    from .formatting import prefix_alphabetization_key
                    prefix_key = tuple(sorted(locant_key(locants[sequence.index(host)]) for host, _word in branches))
                    citation_key = tuple(sorted(((prefix_alphabetization_key(word)[:-1] if labelled_branch
                                                  else prefix_alphabetization_key(word)),
                                                 locant_key(locants[sequence.index(host)]))
                                                for host, word in branches))
                    candidates.append(((hydrogen_key, prefix_key, citation_key, isotope_key, size, score), parent, name))
                    break
    if not candidates or len({row[1] for row in candidates}) != 1:
        return None
    best = min(row[0] for row in candidates)
    names = {name for score, _parent, name in candidates if score == best}
    if len(names) != 1:
        return None
    from .full_bluebook_frontier import _decision
    return _decision(mol, names.pop(), (), 'P-25.7.1.1/P-25.7.1.2/P-52.2.4.1')
