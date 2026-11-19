"""Construct primitive-component three-face fused hydrides with proved numbering.

The topology is generated from component vocabularies and the input graph.
Cases with a retained polycyclic component, an unproved multiparent choice,
unknown orientation or unresolved localized-bond state are left to the
existing selector.
"""
from collections import Counter
from functools import lru_cache
from itertools import combinations
import re


@lru_cache(maxsize=1)
def _retained_topologies():
    """Topology index of independently catalogued two- and three-ring units."""
    from .fusion_components import _catalog
    from .fusion_networks import skeleton_key
    result = {2: set(), 3: set()}
    seen = set()
    for rows in _catalog().values():
        for component in rows:
            if component.ring_count not in result or component.identity in seen:
                continue
            seen.add(component.identity)
            labels = tuple(dict(component.label_elements))
            elements = dict(component.label_elements)
            indices = {label: i for i, label in enumerate(labels)}
            result[component.ring_count].add(skeleton_key(
                [elements[label] for label in labels],
                [tuple(indices[label] for label in bond) for bond in component.bonds]))
    return {rank: frozenset(values) for rank, values in result.items()}


def _topology(mol, atoms):
    from .fusion_networks import skeleton_key
    sequence = tuple(sorted(atoms))
    positions = {atom: i for i, atom in enumerate(sequence)}
    edges = [(positions[bond.GetBeginAtomIdx()], positions[bond.GetEndAtomIdx()])
             for bond in mol.GetBonds() if bond.GetBeginAtomIdx() in positions
             and bond.GetEndAtomIdx() in positions]
    return skeleton_key([mol.GetAtomWithIdx(i).GetAtomicNum() for i in sequence], edges)


def _ordered_face(mol, face):
    """Recover an actual cycle from an unordered, graph-proved face."""
    face = frozenset(face)
    adjacency = {i: sorted(a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetIdx() in face)
                 for i in face}
    if not face or any(len(row) != 2 for row in adjacency.values()):
        return ()
    start = min(face)
    path, previous, current = [start], None, start
    while True:
        choices = [i for i in adjacency[current] if i != previous]
        if not choices:
            return ()
        nxt = choices[0]
        if nxt == start:
            return tuple(path) if len(path) == len(face) else ()
        if nxt in path:
            return ()
        path.append(nxt)
        previous, current = current, nxt


def three_face_fusion(mol, Chem):
    return _component_fusion(mol, Chem, general=False)


def general_component_fusion(mol, Chem, *, include_locants=False):
    return _component_fusion(mol, Chem, general=True, include_locants=include_locants)


def retained_conflict_replacement(mol, Chem):
    """Use P-25.5 skeletal replacement after a proved component conflict.

    This route currently establishes the corresponding hydrocarbon for
    ordinary peripheral, pyridine-like nitrogen atoms.  Other bonding states
    must be proved separately before conversion to a carbon skeleton.
    """
    if (any(not a.IsInRing() or a.GetIsotope() or int(a.GetChiralTag())
            or a.GetFormalCharge() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) for b in mol.GetBonds())):
        return None
    replacements = tuple(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    if not replacements or any(a.GetAtomicNum() not in {6, 7} for a in mol.GetAtoms()):
        return None
    for idx in replacements:
        nitrogen = mol.GetAtomWithIdx(idx)
        if nitrogen.GetDegree() != 2 or (
            nitrogen.GetTotalNumHs(includeNeighbors=True) != (0 if nitrogen.GetIsAromatic() else 1)
            or not nitrogen.GetIsAromatic() and any(b.GetBondTypeAsDouble() != 1 for b in nitrogen.GetBonds())
        ):
            return None
    from .catacondensed_numbering import peripheral_numbering_frontier
    graph = peripheral_numbering_frontier(mol, allow_three_face_n_junction=True,
                                           allow_multi_face_n_junction=True,
                                           allow_larger_system=True,
                                           allow_shared_carbon_junctions=True,
                                           allow_peri_fused=True)
    if not graph.admitted:
        return None
    from .general_fusion_components import replacement_required_by_retained_conflict
    # A peri junction prevents ordinary pairwise fusion even when there is
    # no retained-component conflict at a single edge.
    junctions = sum(sum(atom.GetIdx() in face for face in graph.ring_faces) >= 3
                    for atom in mol.GetAtoms())
    conflict = replacement_required_by_retained_conflict(mol, graph.ring_faces, Chem)
    if not conflict and (not junctions or len(graph.ring_faces) < 4):
        return None
    rw = Chem.RWMol(mol)
    for idx in replacements:
        atom = rw.GetAtomWithIdx(idx)
        atom.SetAtomicNum(6)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
    carbon = rw.GetMol()
    try:
        Chem.SanitizeMol(carbon)
    except (ValueError, RuntimeError):
        return None
    # Restore both bonding state and hydrogen population on a copy of the
    # carbon framework before assigning any of its numbering to the source.
    restored = Chem.RWMol(carbon)
    for idx in replacements:
        original = mol.GetAtomWithIdx(idx)
        atom = restored.GetAtomWithIdx(idx)
        atom.SetAtomicNum(7)
        atom.SetNumExplicitHs(original.GetTotalNumHs())
        atom.SetNoImplicit(True)
    restored = restored.GetMol()
    try:
        Chem.SanitizeMol(restored)
    except (ValueError, RuntimeError):
        return None
    if Chem.MolToSmiles(restored) != Chem.MolToSmiles(mol):
        return None
    return _component_fusion(carbon, Chem, general=True, replacements=(mol, replacements))


def polyalene_fusion(mol, Chem):
    """P-25.1.2.3: two identical carbocyclic mancude components."""
    rings = tuple(mol.GetRingInfo().AtomRings())
    if (len(rings) != 2 or len(rings[0]) != len(rings[1])
            or len(rings[0]) < 5 or len(rings[0]) == 6
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for ring in rings for i in ring)):
        return None
    return _component_fusion(mol, Chem, general=True, polyalene=True)


@lru_cache(maxsize=1)
def _retained_peri_parents():
    """Index source-numbered retained and systematic hydrocarbon parents."""
    from rdkit import Chem
    from .fusion_components import _systematic_fused_records
    from .general_fusion_components import _RETAINED
    from .parent_rules import _curated_fused_templates

    result = {}
    for template in _curated_fused_templates(Chem):
        name, parent = template['name'], template['mol']
        if name not in _RETAINED or any(atom.GetAtomicNum() != 6 for atom in parent.GetAtoms()):
            continue
        key = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
        result.setdefault(key, set()).add(name)
    for name, (smiles, _locants) in _systematic_fused_records().items():
        parent = Chem.MolFromSmiles(smiles)
        if parent is None or any(atom.GetAtomicNum() != 6 for atom in parent.GetAtoms()):
            continue
        key = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
        result.setdefault(key, set()).add(name)
    return {key: tuple(sorted(names)) for key, names in result.items()}


def retained_peri_fusion(mol, Chem):
    """Number annotated retained peri-fused and systematic hydrocarbon parents."""
    if (mol.GetRingInfo().NumRings() < 3
            or not any(atom.GetIsotope() or not atom.IsInRing() for atom in mol.GetAtoms())
            or any(atom.IsInRing() and atom.GetAtomicNum() != 6 for atom in mol.GetAtoms())):
        return None
    return _component_fusion(mol, Chem, general=True, retained_peri=True)


def _carbon_reference_hydrogens_impossible(core):
    """Prove that the later maximum-pi reference cannot match this core.

    For a carbon-only core of degree two or three, a perfect matching is an
    upper-bound witness: every maximum matching covers every atom and hence
    gives the same hydrogen population. If that population differs from the
    input, `_render_component_fusion` must return None regardless of the
    component name. Incomplete matching searches make no such claim.
    """
    atoms = tuple(core.GetAtoms())
    if (not atoms or len(atoms) % 2 or any(a.GetAtomicNum() != 6 or a.GetDegree() not in (2, 3)
                                             for a in atoms)):
        return False
    if all(a.GetTotalNumHs(includeNeighbors=True) == 3 - a.GetDegree() for a in atoms):
        return False
    neighbors = {a.GetIdx(): frozenset(n.GetIdx() for n in a.GetNeighbors()) for a in atoms}
    visits = [0]

    class SearchLimit(Exception):
        pass

    @lru_cache(maxsize=None)
    def perfect(remaining):
        visits[0] += 1
        if visits[0] > 8192:
            raise SearchLimit
        if not remaining:
            return True
        first = min(remaining)
        return any(perfect(remaining - {first, other})
                   for other in sorted(neighbors[first] & remaining))

    try:
        return perfect(frozenset(neighbors))
    except SearchLimit:
        return False


def _component_fusion(mol, Chem, *, general, polyalene=False, retained_peri=False,
                      replacements=None, include_locants=False):
    ring_count = len(mol.GetRingInfo().AtomRings())
    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups() or (ring_count != 2 if polyalene else
                                        ring_count < 3 if general else ring_count != 3)
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 16, 17, 34, 35, 52, 53}
                   or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or (int(a.GetChiralTag()) and not general)
                   or a.HasQuery() for a in mol.GetAtoms())
            or any((int(b.GetStereo()) and (not general or b.IsInRing()))
                   or b.HasQuery() for b in mol.GetBonds())):
        return None
    from .mapped_ring_parents import ring_systems

    # Capping exocyclic branches keeps every ring atom and cannot join
    # separate ring systems into the connected face graph required below.
    if len(ring_systems(mol)) != 1:
        return None
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7, 8, 16} for i in ring):
        return None
    ring_stereo = any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in ring)
    if ring_stereo:
        from .stereochemistry import assigned_atom_stereochemistry
        assigned = dict(assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True))
        if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) and i not in assigned for i in ring):
            return None
    from .fused_branch_annotations import mapped_fused_branch
    owned, branches, branch_isotopes = set(ring), [], {}
    for host in ring:
        for root in mol.GetAtomWithIdx(host).GetNeighbors():
            if root.GetIdx() in ring:
                continue
            bond = mol.GetBondBetweenAtoms(host, root.GetIdx())
            if bond.GetBondTypeAsDouble() != 1:
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
            if general and (any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in group)
                            or any(int(b.GetStereo()) and
                                   {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} & group
                                   for b in mol.GetBonds())):
                from .stereochemistry import assigned_atom_stereochemistry, assigned_bond_stereochemistry
                atom_stereo = dict(assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True))
                if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) and i not in atom_stereo for i in group):
                    return None
                bond_stereo = assigned_bond_stereochemistry(mol, Chem)
                specified = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                             for b in mol.GetBonds() if int(b.GetStereo())
                             and {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} & group}
                if specified != {frozenset((a, b)) for a, b, _desc in bond_stereo
                                 if {a, b} & group}:
                    return None
                if any(not {a, b} <= group for a, b, _desc in bond_stereo if {a, b} & group):
                    return None
                expected = Counter(desc for i, desc in atom_stereo.items() if i in group)
                expected.update(desc for a, b, desc in bond_stereo if {a, b} <= group)
                supplied = Counter(re.findall(r'\d+[a-z]?([RSEZrs])(?=[,)])', word))
                if supplied != expected:
                    return None
            if word in branch_isotopes and branch_isotopes[word] != labels:
                return None
            branch_isotopes[word] = labels
            branches.append((host, word))
            owned.update(group)
    if owned != set(range(mol.GetNumAtoms())):
        return None
    labelled_branch = any(mol.GetAtomWithIdx(i).GetIsotope() for i in owned-ring)
    # Construct the unsubstituted, isotope-free parent only as a graph probe.
    # Original atom IDs and all substituents stay on the source molecule.
    source_atoms = tuple(sorted(ring))
    source_to_core = {old: new for new, old in enumerate(source_atoms)}
    if branches or ring_stereo or any(mol.GetAtomWithIdx(i).GetIsotope() for i in ring):
        rw = Chem.RWMol(mol)
        for idx in source_atoms:
            original = mol.GetAtomWithIdx(idx)
            atom = rw.GetAtomWithIdx(idx)
            atom.SetIsotope(0)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            external = sum(int(b.GetBondTypeAsDouble()) for b in original.GetBonds()
                           if b.GetOtherAtomIdx(idx) not in ring)
            if external:
                atom.SetNumExplicitHs(original.GetTotalNumHs(includeNeighbors=True) + external)
                atom.SetNoImplicit(True)
        for idx in sorted(owned - ring, reverse=True):
            rw.RemoveAtom(idx)
        core = rw.GetMol()
        try:
            Chem.SanitizeMol(core)
        except (ValueError, RuntimeError):
            return None
    else:
        core = mol
    if _carbon_reference_hydrogens_impossible(core):
        return None
    from .catacondensed_numbering import peripheral_numbering_frontier
    graph = peripheral_numbering_frontier(core, allow_three_face_n_junction=True,
                                           allow_multi_face_n_junction=general,
                                           allow_larger_system=general,
                                           allow_shared_carbon_junctions=general,
                                           allow_peri_fused=general)
    if not graph.admitted or len(graph.ring_faces) != ring_count:
        return None
    faces = tuple(_ordered_face(core, face) for face in graph.ring_faces)
    if any(len(face) not in (range(3, 9) if general else {5, 6, 7}) for face in faces):
        return None
    if retained_peri:
        from types import SimpleNamespace
        matches = _retained_peri_parents().get(Chem.MolToSmiles(core, canonical=True, isomericSmiles=True), ())
        if len(matches) != 1:
            return None
        parent = matches[0]
        if not (set(range(core.GetNumAtoms())) - set(graph.perimeter_cycle)):
            from .fusion_components import _systematic_fused_records
            if parent not in _systematic_fused_records():
                return None  # Traditionally numbered catacondensed parents use their own route.
        winners = ((parent, SimpleNamespace(parents=(0,), names=(parent,))),)
    elif polyalene:
        from types import SimpleNamespace
        from .constants import MULT
        size = len(faces[0])
        if (len(faces) != 2 or len(faces[1]) != size or size not in MULT
                or size < 5 or size == 6):
            return None
        prefix = MULT[size]
        parent = (prefix[:-1] if prefix.endswith('a') else prefix) + 'alene'
        winners = ((parent, SimpleNamespace(parents=(0,), names=(parent,))),)
    elif general:
        from .general_fusion_components import candidate_networks
        winners = candidate_networks(core, faces, Chem, allow_peri_fused=True)
        if len(winners) != 1:
            return None
    else:
        winners = _three_face_networks(core, faces)
        if len(winners) != 1:
            return None
    parent = winners[0][0]
    from .fused_orientation import fused_orientation_frontier
    junctions = {atom.GetIdx() for atom in core.GetAtoms() if atom.GetDegree() == 4 and atom.IsInRing()}
    proved_junction_orientation = True
    if junctions:
        # The source-supported shared-carbon numbering has a carbocyclic
        # five-membered interparent between two identical five-membered
        # components. Other geometries require a separate orientation proof;
        # a wrong whole-system locant can change the indicated-H structure.
        neighbors = {i: {j for j in range(len(faces)) if i != j and len(set(faces[i]) & set(faces[j])) == 2}
                     for i in range(len(faces))}
        centers = [i for i in neighbors if len(neighbors[i]) == 2]
        leaves = [i for i in neighbors if len(neighbors[i]) == 1]
        proved_junction_orientation = (len(junctions) == 1 and len(faces) == 3
            and all(len(face) == 5 for face in faces) and len(centers) == 1 and len(leaves) == 2
            and all(core.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in faces[centers[0]])
            and set(faces[leaves[0]]) & set(faces[leaves[1]]) == junctions
            and _topology(core, faces[leaves[0]]) == _topology(core, faces[leaves[1]]))
    drawing = (fused_orientation_frontier(core, allow_three_face_n_junction=True,
                                            allow_multi_face_n_junction=general,
                                            allow_larger_system=general,
                                            allow_shared_carbon_junctions=general,
                                            allow_peri_fused=general)
               if proved_junction_orientation else None)
    numbering = (peripheral_numbering_frontier(
        core, orientation_seeds=drawing.orientation_seeds, allow_three_face_n_junction=True,
        allow_multi_face_n_junction=general, allow_larger_system=general,
        allow_shared_carbon_junctions=general, allow_peri_fused=general)
        if drawing is not None and drawing.admitted else None)
    return _render_component_fusion(mol, Chem, core, ring, source_atoms, source_to_core,
                                    owned, branches, branch_isotopes, labelled_branch, ring_stereo,
                                    winners, parent, numbering, general=general,
                                    replacements=replacements, include_locants=include_locants)


def _three_face_networks(core, faces):
    """Existing three-face comparison, retained as a separate route."""
    benzene = [face for face in faces if len(face) == 6
               and all(core.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in face)]
    # P-25.3.5.3: a symmetric multiparent construction takes precedence
    # over treating a benzene/heterocycle pair as a benzoheterocycle.  Prove
    # the two distinct five-membered chalcogen faces and their independent
    # fusion bonds. N-containing retained components require a separate
    # retained-name comparison and continue to delegate.
    if benzene:
        if len(benzene) != 1:
            return ()
        leaves = [face for face in faces if face != benzene[0]]
        if (len(leaves) != 2 or set(leaves[0]) & set(leaves[1])
                or any(len(face) != 5 or len(set(face) & set(benzene[0])) != 2
                       or not any(core.GetAtomWithIdx(i).GetAtomicNum() in {8, 16} for i in face)
                       or any(core.GetAtomWithIdx(i).GetAtomicNum() not in {6, 8, 16} for i in face)
                       for face in leaves)):
            return ()
    retained = _retained_topologies()
    if _topology(core, range(core.GetNumAtoms())) in retained[3]:
        return ()
    for left, right in combinations(faces, 2):
        if len(set(left) & set(right)) == 2 and _topology(core, set(left) | set(right)) in retained[2]:
            if not benzene or benzene[0] not in (left, right):
                return ()
    from .three_face_aza_fusion import _candidate_networks, _component_rows
    rows = tuple(_component_rows(core, face) for face in faces)
    identities = tuple({row[2].identity for row in group} for group in rows)
    if any(len(group) != 1 for group in identities):
        return ()
    repeated = len({next(iter(group)) for group in identities}) != 3
    primitive = _candidate_networks(core, faces)
    from .fusion_components import _universal_winners, compare_constructions, resolve_component
    from .fusion_networks import _alternative_names, matches_skeleton, parse_network
    networks = dict(primitive)
    seen = set(networks)
    for name, _network in primitive:
        for alternative in _alternative_names(name):
            if alternative in seen:
                continue
            seen.add(alternative)
            network = parse_network(alternative, resolve_component)
            if (network is not None and network.citation is True
                    and matches_skeleton(network, core, range(core.GetNumAtoms()))):
                networks[alternative] = network
    if repeated and not any(len(network.parents) > 1 for network in networks.values()):
        return ()  # Do not guess when a multiparent construction is unresolved.
    def preference(left, right):
        return compare_constructions(left[1], right[1]).preference

    winners = _universal_winners(networks.items(), preference)
    if len(winners) != 1 or benzene and len(winners[0][1].parents) < 2:
        return ()
    return winners


def _render_component_fusion(mol, Chem, core, ring, source_atoms, source_to_core,
                             owned, branches, branch_isotopes, labelled_branch, ring_stereo,
                             winners, parent, numbering, *, general=False, replacements=None,
                             include_locants=False):
    from .ring_state_parent import _HW_VALENCE, _reference_states
    from .parent_selection import locant_key
    size = core.GetNumAtoms()
    numbers = tuple(a.GetAtomicNum() for a in core.GetAtoms())
    edges = tuple(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in core.GetBonds())
    hydrogens = tuple(a.GetTotalNumHs(includeNeighbors=True) for a in core.GetAtoms())
    reference = [(h, indicated) for h, indicated, _bonds in
                 _reference_states(numbers, edges, allow_tetravalent_carbon=general)
                 if h == hydrogens]
    if len(reference) != 1:
        return None
    indicated = reference[0][1]
    network = winners[0][1]
    # The completed fused hydride determines the maximum pi/H state. A
    # pyrrole component's isolated N-H is not an extra H in that completed
    # graph. Distinct pi allocations are qualified below by a Delta witness.
    degree = Counter(i for edge in edges for i in edge)
    occupancy = tuple(_HW_VALENCE[number] - degree[i] - hydrogens[i]
                      for i, number in enumerate(numbers))
    if any(value not in {0, 1} for value in occupancy):
        return None
    needed = frozenset(i for i, value in enumerate(occupancy) if value)
    adjacency = {i: {b if a == i else a for a, b in edges if i in (a, b)} & needed for i in needed}
    calls = [0]
    def matchings(remaining):
        calls[0] += 1
        if calls[0] > 8192:
            return
        if not remaining:
            yield frozenset()
            return
        first = min(remaining)
        for other in sorted(adjacency[first] & remaining):
            for rest in matchings(remaining - {first, other}):
                yield rest | {tuple(sorted((first, other)))}
    states = []
    for pairs in matchings(needed):
        rw = Chem.RWMol()
        for original in mol.GetAtoms():
            atom = Chem.Atom(original.GetAtomicNum())
            atom.SetIsotope(original.GetIsotope())
            atom.SetChiralTag(original.GetChiralTag())
            atom.SetNumExplicitHs(original.GetTotalNumHs(includeNeighbors=True))
            atom.SetNoImplicit(True)
            rw.AddAtom(atom)
        for original in mol.GetBonds():
            left, right = original.GetBeginAtomIdx(), original.GetEndAtomIdx()
            if left in ring and right in ring:
                edge = tuple(sorted((source_to_core[left], source_to_core[right])))
                order = Chem.BondType.DOUBLE if edge in pairs else Chem.BondType.SINGLE
            else:
                order = original.GetBondType()
            rw.AddBond(left, right, order)
            if left not in ring or right not in ring:
                rw.GetBondBetweenAtoms(left, right).SetBondDir(original.GetBondDir())
        for original in mol.GetBonds():
            if not int(original.GetStereo()):
                continue
            copied = rw.GetBondBetweenAtoms(original.GetBeginAtomIdx(), original.GetEndAtomIdx())
            stereo_atoms = tuple(original.GetStereoAtoms())
            if len(stereo_atoms) != 2:
                return None
            copied.SetStereoAtoms(*stereo_atoms)
            copied.SetStereo(original.GetStereo())
        candidate = rw.GetMol()
        try:
            Chem.SanitizeMol(candidate)
        except (ValueError, RuntimeError):
            continue
        states.append((pairs, Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)))
    if calls[0] > 8192:
        return None
    target = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    matching = [pairs for pairs, identity in states if identity == target]
    alternatives = [pairs for pairs, identity in states if identity != target]
    if not matching:
        return None
    if alternatives:
        from .fused_pi_source_evidence import published_unqualified_hydride
        parent_identity = Chem.MolToSmiles(core, canonical=True, isomericSmiles=True)
        if published_unqualified_hydride(parent_identity):
            # The cited source names this parent bond state without Delta.
            # The graph probe restores the parent's hydrogens before testing
            # that evidence, so isotope and substituent annotations retain
            # their original ownership while selecting the same parent state.
            # Different parent assignments still require a Delta witness.
            alternatives = []
    if general and alternatives:
        from .localized_bicyclic_states import requires_explicit_aza_five_five_bonds

        if requires_explicit_aza_five_five_bonds(mol, ring):
            # A unique ordinary fused state survives. The current Delta
            # spelling for this nonunique state lacks interoperable decoding.
            return None
    if general and numbering is not None:
        # This finite drawing frontier does not enumerate the full set of
        # distortions permitted by P-25.3.2.3.2 at mixed-size peri-fusions.
        # An interior carbon plus a nonhexagonal face can change the best
        # drawing, and hence every printed peripheral locant. A single
        # compatible outline is not a complete numbering proof. Keep bare
        # graph-proved names, but delegate operations requiring that map to
        # the existing explicit bond-locanted parent writer.
        interior = set(range(size)) - set(numbering.perimeter_cycle)
        unproved_peri_numbering = (any(core.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in interior)
                                  and any(len(face) != 6 for face in numbering.ring_faces))
        if unproved_peri_numbering and (include_locants or replacements or branches or ring_stereo
                or indicated or alternatives or any(mol.GetAtomWithIdx(i).GetIsotope() for i in ring)):
            return None
    if numbering is None or not numbering.admitted or not numbering.candidates:
        # A bare parent without indicated H or localized-bond ambiguity has
        # no whole-system locants to print. Its component construction and
        # complete original graph have already been proved, so a distorted
        # drawing is not needed merely to render this particular name.
        if (replacements or branches or ring_stereo or any(mol.GetAtomWithIdx(i).GetIsotope() for i in ring)
                or indicated or alternatives):
            return None
        from .full_bluebook_frontier import _decision
        return _decision(mol, parent, (), 'P-25.3.4/P-25.7/P-52.2.4.1')
    outputs = []
    for row in numbering.candidates:
        locants = dict(row.atom_locants)
        if set(locants) != set(range(size)) or any(locant_key(locants[i]) is None for i in range(size)):
            continue
        h_locants = tuple(sorted((locants[i] for i in indicated), key=locant_key))
        h_text = ','.join(value+'H' for value in h_locants)
        positions = {locants[atom]: i for i, atom in enumerate(row.seed.clockwise_order)}
        def bond_term(edge):
            left, right = sorted((locants[edge[0]], locants[edge[1]]), key=locant_key)
            adjacent = (left in positions and right in positions
                        and positions[right] == positions[left] + 1)
            return (locant_key(left), 0 if adjacent else 1, locant_key(right)), left if adjacent else left+'('+right+')'
        witnesses = []
        if alternatives:
            for pairs in matching:
                for length in range(1, len(pairs)+1):
                    this_length = []
                    for subset in combinations(sorted(pairs), length):
                        if all(not set(subset) <= other for other in alternatives):
                            terms = sorted(map(bond_term, subset))
                            this_length.append((length, tuple(t[0] for t in terms), ','.join(t[1] for t in terms)))
                    if this_length:
                        witnesses.extend(this_length)
                        break
            if not witnesses:
                continue
            length, score, descriptor = min(witnesses)
            delta = 'Δ^('+descriptor+')-'
        else:
            length, score, delta = 0, (), ''
        mapping = {source_atoms[index]: value for index, value in locants.items()}
        replacement_key = ()
        replacement_text = ''
        if replacements:
            from .constants import MULT
            original, atoms = replacements
            replacement_locants = tuple(sorted((mapping[i] for i in atoms), key=locant_key))
            replacement_key = tuple(map(locant_key, replacement_locants))
            multiplier = '' if len(atoms) == 1 else MULT.get(len(atoms))
            if multiplier is None:
                continue
            replacement_text = ','.join(replacement_locants) + '-' + multiplier + 'aza'
        name = (h_text+'-' if h_text else '') + delta + replacement_text + parent
        isotope_key = ()
        if branches or ring_stereo or any(mol.GetAtomWithIdx(i).GetIsotope() for i in ring):
            from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings
            from .parent_annotations import isotope_numbering_key
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
                    prefix_alphabetization_key(word)[:-1],
                    isotope_order(word),
                    tuple(sorted(locant_key(locant) for locant, value in prefix_rows if value == word)))))
            component = MappedComponent(name, tuple(mapping.items()), frozenset(ring),
                                        prefix_rows, frozenset(owned-ring), systematic_parent=True,
                                        prefix_citation_order=citation_order)
            name = decorate_component(mol, Chem, component)
            if not name or isotope_key is None:
                continue
            if delta and branches and not name[:name.index('Δ^')].endswith('-'):
                name = name.replace('Δ^', '-Δ^', 1)
        from .formatting import prefix_alphabetization_key
        prefix_key = tuple(sorted(locant_key(mapping[host]) for host, _word in branches))
        citation_key = tuple(sorted(((prefix_alphabetization_key(word)[:-1] if labelled_branch
                                      else prefix_alphabetization_key(word)), locant_key(mapping[host]))
                                    for host, word in branches))
        if ring_stereo:
            from .parent_annotations import stereo_numbering
            stereo_key = stereo_numbering(mol, mapping)
            if stereo_key is None:
                continue
        else:
            stereo_key = ()
        # Number the parent and its substituents before selecting the shortest
        # localized-bond witness. P-25.7.1.2's dimethylheptalene pair keeps
        # the same lowest substituent locants for both Delta alternatives.
        key = (tuple(locant_key(value) for value in h_locants),
               replacement_key, prefix_key, citation_key, isotope_key, stereo_key, length, score)
        outputs.append((key, name, tuple(sorted(mapping.items())) if include_locants else ()))
    if not outputs:
        return None
    best = min(key for key, _name, _map in outputs)
    names = {name for key, name, _map in outputs if key == best}
    if len(names) != 1:
        return None
    from .full_bluebook_frontier import _decision
    name, = names
    decision = _decision(replacements[0] if replacements else mol, name, (),
                         'P-25.5.1/P-25.5.2' if replacements else 'P-25.3.4/P-25.7/P-52.2.4.1')
    if include_locants:
        maps = {mapping for key, _name, mapping in outputs if key == best}
        if len(maps) == 1:
            from dataclasses import replace

            decision = replace(decision, atom_locants=maps.pop())
    return decision
