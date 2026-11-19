"""Compare graph-derived component trees for catacondensed fused systems.

Components and their native locants come from the P-25 retained-name tables
or the independently numbered monocyclic Hantzsch--Widman vocabulary.  Face
partitions, fusion bonds and all native-label mappings are derived from the
input graph.  Unknown component identities or incomplete comparisons defer.
"""
from collections import Counter, deque
from dataclasses import replace
from functools import lru_cache
from itertools import combinations, permutations, product
import re


# P-25 Tables 2.7 and 2.8; availability and numbering still require a unique
# structural component from the installed fusion-component catalogue.
_RETAINED = (
    'ovalene', 'pyranthrene', 'coronene', 'rubicene', 'perylene', 'picene',
    'pleiadene', 'chrysene', 'pyrene', 'fluoranthene', 'anthracene',
    'phenanthrene', 'phenalene', 'fluorene', 's-indacene', 'as-indacene',
    'azulene', 'naphthalene', 'indene', 'phenazine', 'perimidine', 'acridine',
    'phenanthridine', 'carbazole', 'pteridine', 'cinnoline', 'quinazoline',
    'quinoxaline', 'phthalazine', 'quinoline', 'isoquinoline', 'quinolizine',
    'purine', 'indazole', 'indole', 'isoindole', 'indolizine', 'pyrrolizine',
    'xanthene', 'thioxanthene', 'selenoxanthene', 'telluroxanthene',
    # P-25.2.2.2/3 names are also eligible polycyclic components.
    'oxanthrene', 'thianthrene', 'selenoanthrene', 'telluranthrene',
    'phenoxazine', 'phenothiazine', 'phenoselenazine', 'phenotellurazine',
    'phenoxathiine', 'phenoxaselenine', 'phenoxatellurine',
)
_SYSTEMATIC_COMPONENTS = ('pentalene', 'heptalene', 'octalene',
                          'acenaphthylene', 'aceanthrylene', 'acephenanthrylene')


@lru_cache(maxsize=1)
def _component_index():
    from .fusion_components import _catalog, _systematic_fused_records, resolve_component
    from .fusion_networks import skeleton_key

    def key(component):
        labels = tuple(label for label, _number in component.label_elements)
        elements = dict(component.label_elements)
        positions = {label: i for i, label in enumerate(labels)}
        return skeleton_key([elements[label] for label in labels],
                            [tuple(positions[label] for label in edge) for edge in component.bonds])

    supported = {}
    systematic = tuple(_systematic_fused_records())
    for name in _RETAINED + _SYSTEMATIC_COMPONENTS + systematic:
        component = resolve_component(name)
        if component is not None and component.ring_count > 1:
            supported.setdefault((component.ring_count, key(component)), []).append((name, component))
    # Also detect catalogue components whose preferred spelling or treatment
    # has not yet been established. They cannot silently be split into rings.
    known = set()
    seen = set()
    for rows in _catalog().values():
        for component in rows:
            if component.ring_count > 1 and component.identity not in seen:
                seen.add(component.identity)
                known.add((component.ring_count, key(component)))
    return {k: tuple(v) for k, v in supported.items()}, frozenset(known)


def _skeleton(mol, Chem):
    graph = Chem.RWMol()
    for atom in mol.GetAtoms():
        new = Chem.Atom(atom.GetAtomicNum())
        new.SetNoImplicit(True)
        graph.AddAtom(new)
    for bond in mol.GetBonds():
        graph.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    return graph.GetMol()


def _retained_mappings(target, atoms, component, Chem):
    labels = tuple(label for label, _number in component.label_elements)
    numbers = dict(component.label_elements)
    positions = {label: i for i, label in enumerate(labels)}
    rw = Chem.RWMol()
    for label in labels:
        atom = Chem.Atom(numbers[label])
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for edge in component.bonds:
        left, right = tuple(edge)
        rw.AddBond(positions[left], positions[right], Chem.BondType.SINGLE)
    query = rw.GetMol()
    matches = target.GetSubstructMatches(query, uniquify=False, maxMatches=100000)
    if len(matches) == 100000:
        return ()
    return tuple(dict(zip(match, labels, strict=True)) for match in matches if frozenset(match) == atoms)


def _same_mapped_connectivity(choices):
    """Check that every choice places the same labelled graph on the source atoms."""
    if len(choices) < 2:
        return True

    def image(choice):
        _name, component, mapping = choice
        atoms = {label: source for source, label in mapping.items()}
        if (len(atoms) != len(mapping) or len(atoms) != component.atom_count
                or atoms.keys() != {label for label, _ in component.label_elements}):
            return None
        elements = frozenset((atoms[label], number) for label, number in component.label_elements)
        bonds = frozenset(frozenset(atoms[label] for label in edge) for edge in component.bonds)
        return component.ring_count, elements, bonds

    first = image(choices[0])
    return first is not None and all(image(choice) == first for choice in choices[1:])


def _first_order_key(letters, numbers):
    from .parent_selection import locant_key, locant_set

    # Within one physical placement, P-25.3.4.2.4(a-d) compares
    # parent letters before the attached-component locants.
    return tuple(sorted(letters)), letters, locant_set(numbers), tuple(map(locant_key, numbers))


def _prune_first_order_numberings(choices, graph, paths, compare_components, resolve):
    """Find the best first attachment for a fixed component tree."""
    from .fusion_networks import (_first_order_binding, _lettered_path, _multiplied,
                                  _numbered_path, _omit_peri_fusion_carbons, _prefix, _tokens)

    fixed = tuple({(name, component) for name, component, _ in row} for row in choices)
    if len(fixed) < 3 or any(len(row) != 1 for row in fixed):
        return None
    names, components = zip(*(next(iter(row)) for row in fixed))
    roots = [i for i, parent in enumerate(components) if len(graph[i]) == 1
             and all(i == j or compare_components(parent, other).preference == -1
                     for j, other in enumerate(components))]
    if len(roots) != 1:
        return None
    root = roots[0]
    branch = next(iter(graph[root]))

    # Along a single stem, every numbered descriptor has one lower-order
    # recipient. Only the final layer may have siblings. Their multiplication
    # depends on the fixed terminal component identities.
    parent, node = root, branch
    while len(children := graph[node] - {parent}) == 1:
        parent, node = node, next(iter(children))
    if any(len(graph[child]) != 1 for child in children):
        return None
    copies = Counter(components[child].identity for child in children)
    for i, (name, component) in enumerate(zip(names, components, strict=True)):
        if i == root:
            continue
        count = copies[component.identity] if i in children else 1
        prefix = _prefix(name)
        spelling = _multiplied(prefix, count) if prefix else None
        if not spelling or _tokens(spelling + '[1,2-a]' + names[root], resolve) != (
                [((component, count), '1,2-a', spelling)], (components[root], 1), names[root]):
            return None

    # Component seniority, tree depths, and multiplication now tie in
    # every admissible spelling. Compute the first-order key using the
    # same binding and locant expansion that rendering and parsing use.
    path = paths[(min(root, branch), max(root, branch))]
    best, allowed = None, set()
    for root_choice, branch_choice in product(choices[root], choices[branch]):
        attachment = tuple(branch_choice[2][atom] for atom in path)
        target = tuple(root_choice[2][atom] for atom in path)
        binding = _first_order_binding(attachment, target, components[root])
        if binding is None:
            return None
        numbers, letters = binding
        parent_path = _lettered_path(components[root], letters)
        printed = _omit_peri_fusion_carbons(components[branch], numbers)
        if parent_path is None or _numbered_path(components[branch], printed) != numbers:
            return None
        key = _first_order_key(parent_path[2], numbers)
        if best is None or key < best:
            best = key
            allowed.clear()
        if key == best:
            allowed.add((id(root_choice), id(branch_choice)))
    if len(allowed) == len(choices[root]) * len(choices[branch]):
        return None
    return root, branch, allowed, best


def _prune_unprinted_terminal_numberings(choices, graph, paths, component_layout):
    """Share terminal numberings that cannot affect the rendered citation."""
    from .fusion_networks import _implicit_terminal_hydrocarbon, _prefix
    from .parent_selection import locant_key

    fixed = tuple({(name, component) for name, component, _ in rows} for rows in choices)
    if any(len(unit) != 1 for unit in fixed):
        return choices
    names, components = zip(*(next(iter(unit)) for unit in fixed), strict=True)
    if len({c.identity for c in components}) != len(components):
        return choices
    layout = component_layout(components)
    if layout is None:
        return choices
    admitted_roots, _, _ = layout
    if len(admitted_roots) != 1:
        return choices
    roots, _ = admitted_roots[0]
    if len(roots) != 1:
        return choices
    root = roots[0]

    # With distinct identities no siblings multiply. Their alphabetical
    # citation keys must decide the order before any locant tie-break, at
    # every depth, for the invisible terminal locants to be immaterial.
    def alphabet(node, parent):
        children = [alphabet(child, node) for child in graph[node] if child != parent]
        prefix = _prefix(names[node])
        if prefix is None or None in children or len(children) != len(set(children)):
            return None
        return ''.join(sorted(children)) + re.sub(r'[^a-z]', '', prefix)

    if alphabet(root, None) is None:
        return choices
    reduced = list(choices)
    for leaf, rows in enumerate(choices):
        if leaf == root or len(graph[leaf]) != 1 or len(rows) == 1:
            continue
        target = next(iter(graph[leaf]))
        path = paths[min(leaf, target), max(leaf, target)]
        if not _implicit_terminal_hydrocarbon(components[leaf], tuple(rows[0][2][i] for i in path)):
            continue
        # A first-order descriptor depends on the parent's letters. At
        # higher orders the target endpoints alone must decide the direction.
        if target != root and any(locant_key(mapping[path[0]]) == locant_key(mapping[path[-1]])
                                  for _, _, mapping in choices[target]):
            continue
        reduced[leaf] = rows[:1]
    return reduced


def replacement_required_by_retained_conflict(core, faces, Chem):
    """Prove the P-25.5 overlap obstruction for every hetero parent.

    The ordinary fusion procedure cannot use either of two attached retained
    components when they share a face but neither contains the other.  Only
    return true when this obstruction is present for every known possible
    heterocyclic parent; unclassified polycyclic components make the proof
    inconclusive.
    """
    faces = tuple(frozenset(face) for face in faces)
    adjacency = {i: {j for j in range(len(faces)) if i != j and len(faces[i] & faces[j]) == 2}
                 for i in range(len(faces))}
    if not any(all(j in adjacency[i] for i, j in combinations(triple, 2))
               for triple in combinations(range(len(faces)), 3)):
        return False
    retained, known = _component_index()
    from .primitive_three_face_fusion import _topology
    components = []
    hetero_parents = []
    for i, face in enumerate(faces):
        if any(core.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in face):
            hetero_parents.append(frozenset({i}))
    for size in range(2, len(faces)):
        for selection in combinations(range(len(faces)), size):
            subset = frozenset(selection)
            todo, reached = [selection[0]], {selection[0]}
            for i in todo:
                for j in adjacency[i] & subset - reached:
                    reached.add(j)
                    todo.append(j)
            if reached != subset:
                continue
            atoms = frozenset().union(*(faces[i] for i in subset))
            key = (size, _topology(core, atoms))
            if key in known and key not in retained:
                return False
            if key not in retained:
                continue
            if any(name in _RETAINED for name, _component in retained[key]):
                components.append(subset)
            if any(core.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in atoms):
                hetero_parents.append(subset)
    if not hetero_parents:
        return False
    for parent in hetero_parents:
        attached = [part for part in components if not part & parent]
        if not any(a & b and not (a <= b or b <= a)
                   for a, b in combinations(attached, 2)):
            return False
    return True


def candidate_networks(core, faces, Chem, *, allow_peri_fused=False, _prune=True):
    """Enumerate source-numbered component decompositions and rerootings."""
    from .fusion_components import (_universal_winners, compare_components, compare_constructions,
                                    construction_criteria, resolve_component)
    from .fusion_networks import FusionLink, _finish, _render_rerooted, parse_network
    from .primitive_three_face_fusion import _topology
    from .three_face_aza_fusion import _component_rows

    faces = tuple(frozenset(face) for face in faces)
    n = len(faces)
    if n < 3:
        return ()
    adjacency = {i: {j for j in range(n) if j != i and len(faces[i] & faces[j]) == 2}
                 for i in range(n)}
    if not allow_peri_fused and sum(map(len, adjacency.values())) != 2*(n-1):
        return ()
    # P-25.3.5.3: a symmetric multiparent assembly takes precedence over
    # replacing a benzene/heterocycle pair by a benzoheterocycle. Recognize
    # the full face-star, not a preselected whole molecule.
    symmetric_benzo = None
    for center in range(n):
        face = faces[center]
        leaves = tuple(i for i in range(n) if i != center)
        if (len(face) != 6 or not all(core.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in face)
                or adjacency[center] != set(leaves) or any(adjacency[i] != {center} for i in leaves)
                or any(len(faces[i]) != 5 for i in leaves)
                or any(faces[i] & faces[j] for i, j in combinations(leaves, 2))):
            continue
        compositions = [tuple(sorted(core.GetAtomWithIdx(i).GetAtomicNum() for i in faces[j])) for j in leaves]
        if compositions and len(set(compositions)) == 1 and compositions[0] in {
                (6, 6, 6, 6, 8), (6, 6, 6, 6, 16)}:
            symmetric_benzo = (center, leaves)
            break
    retained, known = _component_index()
    target = _skeleton(core, Chem)
    # The core graph is invariant across every decomposition. Compute its
    # topology and tetravalent-junction proof once, rather than rebuilding a
    # canonical graph for each rendered component assignment.
    elements = tuple(sorted(Counter(atom.GetAtomicNum() for atom in core.GetAtoms()).items()))
    target_shape = (core.GetNumAtoms(), core.GetNumBonds(),
                    Counter((atom.GetAtomicNum(), atom.GetDegree()) for atom in core.GetAtoms()))
    target_facts = (core.GetNumAtoms(), core.GetNumBonds()-core.GetNumAtoms()+1,
                    elements, _topology(core, range(core.GetNumAtoms())))
    if any(atom.GetDegree() == 4 and (atom.GetAtomicNum() != 6
           or any(bond.GetIsAromatic() or bond.GetBondTypeAsDouble() != 1
                  for bond in atom.GetBonds())) for atom in core.GetAtoms()):
        return ()

    def same_skeleton(network):
        return (network.atom_count, network.ring_count, network.elements, network.skeleton) == target_facts

    groups = {}
    for i, face in enumerate(faces):
        rows = _component_rows(core, tuple(_cycle(core, face)))
        values = []
        for name, sequence, component in rows:
            if resolve_component(name) == component:
                values.append((name, component, dict(zip(sequence, map(str, range(1, len(sequence)+1)), strict=True))))
        if not values:
            return ()
        groups[frozenset({i})] = values
    unknown = set()
    for size in range(2, n+1):
        for selection in combinations(range(n), size):
            subset = frozenset(selection)
            todo, reached = [selection[0]], {selection[0]}
            for i in todo:
                for j in adjacency[i] & subset - reached:
                    reached.add(j); todo.append(j)
            if reached != subset:
                continue
            atoms = frozenset().union(*(faces[i] for i in subset))
            key = (size, _topology(core, atoms))
            if key not in known and key not in retained:
                continue
            if subset == frozenset(range(n)):
                return ()  # Existing whole-parent routes own retained names.
            options = []
            for name, component in retained.get(key, ()):
                options.extend((name, component, mapping)
                               for mapping in _retained_mappings(target, atoms, component, Chem))
            if options:
                groups[subset] = options
            else:
                unknown.add(subset)

    all_faces = frozenset(range(n))
    partitions = []
    def partition(remaining, selected):
        if not remaining:
            # Do not split a retained component when it can be formed by
            # merging complete components of this same partition.
            for subset in groups:
                if len(subset) < 2:
                    continue
                within = [part for part in selected if part <= subset]
                if len(within) > 1 and frozenset().union(*within) == subset:
                    return
            # Unknown polycyclic units may supersede a decomposition.
            if any(not (symmetric_benzo and len(subset) == 2 and symmetric_benzo[0] in subset
                        and bool(subset & set(symmetric_benzo[1])))
                   and len([part for part in selected if part <= subset]) > 1
                   and frozenset().union(*(part for part in selected if part <= subset)) == subset
                   for subset in unknown):
                return
            partitions.append(tuple(selected)); return
        first = min(remaining)
        for subset in groups:
            if first in subset and subset <= remaining:
                partition(remaining-subset, selected+[subset])
    partition(all_faces, [])
    candidates = {}
    pruning = None
    # A spelling completely determines the parsed network. The same text
    # can be rendered by several equivalent mapping/order combinations or by
    # different physical partitions, so parse it at most once per search.
    @lru_cache(maxsize=8192)
    def parse_text(text):
        return parse_network(text, resolve_component, expected_shape=target_shape)
    for parts in partitions:
        if len(parts) < 2:
            continue
        # An isolated benzene/heterocycle pair requires the separate P-25.3.5
        # benzoheterocycle and multiparent comparison.
        if (not symmetric_benzo and any(a.GetAtomicNum() != 6 for a in core.GetAtoms())
                and any(len(part) == 1 and len(faces[next(iter(part))]) == 6
               and all(core.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in faces[next(iter(part))])
               for part in parts)):
            continue
        atoms = [frozenset().union(*(faces[i] for i in part)) for part in parts]
        paths = {}
        valid_paths = True
        for left, right in combinations(range(len(parts)), 2):
            common = atoms[left] & atoms[right]
            if len(common) < 2:
                continue
            path = _shared_path(core, common)
            if path is None or len(path) > 2 and not allow_peri_fused:
                valid_paths = False
                break
            paths[(left, right)] = path
        if not valid_paths:
            continue
        graph = {i: {j for j in range(len(parts)) if j != i
                     and (min(i, j), max(i, j)) in paths} for i in range(len(parts))}
        if sum(map(len, graph.values())) != 2*(len(parts)-1):
            continue
        parent = {0: None}
        queue = deque([0])
        while queue:
            i = queue.popleft()
            for j in graph[i]:
                if j not in parent:
                    parent[j] = i; queue.append(j)
        if len(parent) != len(parts):
            continue
        # Atom mappings only differ materially at component fusion ports.
        choices = []
        for i, part in enumerate(parts):
            ports = sorted(set().union(*(atoms[i] & atoms[j] for j in graph[i])))
            unique = {}
            for name, component, mapping in groups[part]:
                key = (name, component.identity, tuple(mapping[a] for a in ports))
                unique[key] = (name, component, mapping)
            choices.append(tuple(unique.values()))
        # In a carbon-only star with one strictly senior central parent,
        # terminal benzenes cannot be roots. Each benzene has no descendant
        # whose identity or citation could depend on its numbering. Its
        # numbering cannot change the parent's fusion letters either. A
        # larger set of benzene fusion locants is therefore dominated by an
        # available mapping with the smallest set for the same physical
        # placement. Keep both orientations when the sets tie.
        centers = [i for i in graph if len(graph[i]) == len(parts) - 1]
        if (len(parts) >= 3 and len(centers) == 1
                and all(a.GetAtomicNum() == 6 for a in core.GetAtoms())):
            center = centers[0]
            center_components = {row[1] for row in choices[center]}
            if (len(center_components) == 1
                    and all(compare_components(other[1], next(iter(center_components))).preference == 1
                            for i, rows in enumerate(choices) if i != center for other in rows)):
                from .parent_selection import locant_set
                for i in graph[center]:
                    if not all(name == 'benzene' and component.ring_count == 1
                               for name, component, _mapping in choices[i]):
                        continue
                    path = paths[(min(i, center), max(i, center))]
                    keys = [locant_set(tuple(mapping[a] for a in path))
                            for _name, _component, mapping in choices[i]]
                    if keys and all(key is not None for key in keys):
                        minimum = min(keys)
                        choices[i] = tuple(row for row, key in zip(choices[i], keys) if key == minimum)
        # Each link joins the same source atoms across all numberings of a
        # layout. If the mapped component graphs agree, one assembly proves
        # the graph for every numbering of that layout.
        reuse_connectivity = all(_same_mapped_connectivity(options) for options in choices)
        if _prune and reuse_connectivity and len(partitions) == 1 and not symmetric_benzo:
            pruning = _prune_first_order_numberings(choices, graph, paths, compare_components,
                                                   resolve_component)
        physical_partition = frozenset(parts)

        @lru_cache(maxsize=128)
        def component_layout(components):
            # The partition and component identities fix root eligibility and
            # sibling permutations across all of their native numberings.
            identities = {}
            for i, component in enumerate(components):
                identities.setdefault(component.identity, []).append(i)
            repeated = tuple(tuple(indices) for indices in identities.values() if len(indices) > 1)
            # P-25.3.7.1 also applies when the interparent component is
            # itself identical to the parent components. Enumerate the
            # physical parent occurrences, rather than inferring them from
            # the multiplicity of the whole component vocabulary.
            multiparents = []
            if len(parts) >= 3:
                for center in range(len(parts)):
                    by_identity = {}
                    for i in graph[center]:
                        by_identity.setdefault(components[i].identity, []).append(i)
                    for eligible in by_identity.values():
                        for count in range(2, len(eligible)+1):
                            for roots in combinations(eligible, count):
                                if (any(len(atoms[i] & atoms[j]) > 1 for i, j in combinations(roots, 2))
                                        or any(compare_components(other, components[roots[0]]).preference
                                               not in (0, 1) for other in components)):
                                    continue
                                multiparents.append(roots)
            # If a different component is strictly senior, repeated less
            # senior components cannot form the parent of a multiparent
            # alternative. Their complete branches, rather than just their
            # component identities, determine whether they are multiplied.
            if not symmetric_benzo and any(
                    not any(compare_components(other, components[group[0]]).preference == -1
                            for other in components if other.identity != components[group[0]].identity)
                    and not any(components[roots[0]].identity == components[group[0]].identity
                                for roots in multiparents)
                    for group in repeated):
                return None
            root_sets = [(root,) for root in range(len(parts))]
            for multiparent in multiparents:
                # Prime assignments belong to occurrences, not atom IDs.
                # Compare every ordering of equivalent parent components.
                root_sets.extend(permutations(multiparent))
            # Only identical terminal siblings can be multiplied by the
            # renderer. Occurrences at different depths do not participate
            # in the same prime assignment, so permuting them is redundant.
            sibling_groups = []
            for target_i in graph:
                by_identity = {}
                for child in graph[target_i]:
                    if graph[child] == {target_i}:
                        by_identity.setdefault(components[child].identity, []).append(child)
                sibling_groups.extend(tuple(group) for group in by_identity.values() if len(group) > 1)
            admitted_roots = []
            for roots in root_sets:
                # A lower-ranked component cannot become the parent just
                # because the senior component's decomposition failed;
                # P-25.5 applies its alternative procedures first.
                if any(compare_components(other, components[roots[0]]).preference
                       not in (0, 1) for other in components):
                    continue
                # A retained unit wholly outside the chosen parent must
                # remain available as an attached component.  Checking
                # only unions of complete parts misses two overlapping
                # retained units: each may be split by a *different*
                # part, even though neither is a union of whole parts.
                # Units crossing the parent are not attached components
                # and must not invalidate it (P-25.3.5, P-25.5).
                parent_parts = frozenset(parts[root] for root in roots)
                parent_faces = frozenset().union(*parent_parts)
                attached_retained = (subset for subset, options in groups.items()
                                     if len(subset) > 1 and not subset & parent_faces
                                     and any(name in _RETAINED for name, _component, _mapping in options))
                if any(any(part & subset and not subset <= part for part in parts)
                       for subset in attached_retained):
                    continue
                admitted_roots.append((roots, (physical_partition, parent_parts)))
            sibling_sets = tuple(frozenset(group) for group in sibling_groups)
            sibling_permutations = tuple(tuple(permutations(group)) for group in sibling_groups)
            return tuple(admitted_roots), sibling_sets, sibling_permutations

        if _prune and reuse_connectivity and pruning is None and not symmetric_benzo:
            choices = _prune_unprinted_terminal_numberings(choices, graph, paths, component_layout)

        proved_network = None
        for selected in product(*choices):
            if pruning is not None:
                root, branch, allowed, _key = pruning
                if (id(selected[root]), id(selected[branch])) not in allowed:
                    continue
            names = tuple(row[0] for row in selected)
            components = tuple(row[1] for row in selected)
            layout = component_layout(components)
            if layout is None:
                continue
            admitted_roots, sibling_sets, sibling_permutations = layout
            links = []
            for child, target_i in parent.items():
                if target_i is None:
                    continue
                common = paths[(min(child, target_i), max(child, target_i))]
                links.append(FusionLink(child, target_i,
                                        tuple(selected[child][2][a] for a in common),
                                        tuple(selected[target_i][2][a] for a in common)))
            links = tuple(links)
            network = proved_network
            if network is None:
                network = _finish(components, (0,), links, names[0], expected_shape=target_shape)
                if network is None or not same_skeleton(network):
                    if reuse_connectivity:
                        break
                    continue
                if reuse_connectivity:
                    proved_network = network
            network = replace(network, components=components, links=links, names=names,
                              benzo_parent='benzo' in names[0])
            for orders in product(*sibling_permutations):
                sibling_orders = dict(zip(sibling_sets, orders))
                for roots, physical in admitted_roots:
                    text = _render_rerooted(network, roots, sibling_orders=sibling_orders)
                    if not text:
                        continue
                    if (text, physical) in candidates:
                        continue
                    parsed = parse_text(text)
                    if (parsed is not None and parsed.citation is True
                            and same_skeleton(parsed)):
                        if pruning is not None:
                            first = [link for link in parsed.links if link.target in parsed.parents]
                            if (len(parsed.parents) != 1 or len(first) != 1
                                    or _first_order_key(first[0].letters, first[0].child_locants) != pruning[3]):
                                return candidate_networks(core, faces, Chem,
                                                          allow_peri_fused=allow_peri_fused, _prune=False)
                        # Retain original face ownership: P-25.3.4.2.2(b,c)
                        # choose between physical locations of components, while
                        # P-25.3.4.2.4 orders alternative locant assignments for
                        # the same physical component locations.
                        candidates[(text, physical)] = parsed
    if not candidates:
        if pruning is not None:
            return candidate_networks(core, faces, Chem, allow_peri_fused=allow_peri_fused, _prune=False)
        return ()

    def preferred(left, right):
        (left_name, left_location), left_network = left
        (right_name, right_location), right_network = right
        if left_location != right_location:
            return compare_constructions(left_network, right_network).preference
        deferred = []
        layers = {}
        for rule, a, b in construction_criteria(left_network, right_network):
            if rule in {'P-25.3.4.2.2(b)', 'P-25.3.4.2.2(c)'}:
                deferred.append((a, b))
                continue
            if rule in {f'P-25.3.4.2.4({letter})' for letter in 'efgh'}:
                layers[rule[-2]] = (a, b)
                continue
            if a is None or b is None:
                return None
            if a != b:
                return -1 if a < b else 1
            if rule == 'P-25.3.4.2.4(b)':
                # The same numbers become the P-25.3.4.2.4(c,d)
                # locant tie-breaks after the parent-component letters.
                for x, y in deferred:
                    if x is None or y is None:
                        return None
                    if x != y:
                        return -1 if x < y else 1
        # Higher-order components are compared successively outwards. The
        # entire lower/upper locant comparison at one order precedes the
        # next order; flattening every order by criterion changes P-25's
        # published third-order example.
        if layers:
            count = max(len(value[0]) for value in layers.values())
            for depth in range(count):
                for letter in 'efgh':
                    a, b = layers[letter]
                    if depth >= len(a) or depth >= len(b):
                        return None
                    if a[depth] != b[depth]:
                        return -1 if a[depth] < b[depth] else 1
        return 0

    winners = {}
    for (name, _location), network in _universal_winners(candidates.items(), preferred):
        winners[name] = network
    return tuple(winners.items())


def _cycle(mol, face):
    from .primitive_three_face_fusion import _ordered_face
    return _ordered_face(mol, face)


def _shared_path(mol, atoms):
    """Return the shared fusion path; its direction is compared later."""
    atoms = frozenset(atoms)
    adjacency = {i: {neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(i).GetNeighbors()
                     if neighbor.GetIdx() in atoms} for i in atoms}
    ends = [i for i, neighbors in adjacency.items() if len(neighbors) == 1]
    if len(ends) != 2 or any(len(neighbors) not in (1, 2) for neighbors in adjacency.values()):
        return None
    path, previous, current = [], None, min(ends)
    while current is not None:
        if current in path:
            return None
        path.append(current)
        following = adjacency[current] - ({previous} if previous is not None else set())
        if len(following) > 1:
            return None
        previous, current = current, next(iter(following)) if following else None
    return tuple(path) if len(path) == len(atoms) else None
