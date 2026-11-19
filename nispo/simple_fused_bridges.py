"""Discover mapped homogeneous divalent bridges on retained fused parents.

The bridge graph, parent mapping, and locants are obtained from the input.
This path covers a single unbranched saturated bridge whose two ends attach
to nonadjacent positions of one constituent ring. Other decompositions are
reported as a preference limitation rather than silently assumed absent.
"""
from __future__ import annotations

from functools import lru_cache

from .bridged_fused_methods import (
    FusedCoreProof, _core_name_on_original_graph, _locant_key,
    _next_bridge_locant, _proof_is_consistent, _same_core_graph,
    _without_indicated_hydrogen,
)


@lru_cache(maxsize=1)
def _bridge_templates(Chem):
    """Add P-25's traditionally numbered parents after a labelled-graph check."""
    from .fusion_components import resolve_component
    from .parent_rules import _curated_fused_templates
    from .ring_state_parent import _fused_templates

    traditional = {'anthracene', 'phenanthrene', 'acridine', 'carbazole',
                   'xanthene', 'thioxanthene', 'selenoxanthene', 'telluroxanthene',
                   'oxanthrene', 'thianthrene', 'selenoanthrene', 'telluranthrene',
                   'purine', 'cyclopenta[a]phenanthrene'}
    result = list(_fused_templates(Chem))
    seen = {template['name'] for template in result}
    for template in _curated_fused_templates(Chem):
        name, query, locants = template['name'], template['mol'], template['locants']
        if name not in traditional or name in seen:
            continue
        component = resolve_component(name)
        if component is None or set(locants) != set(range(query.GetNumAtoms())):
            continue
        elements = {str(locants[a.GetIdx()]): a.GetAtomicNum() for a in query.GetAtoms()}
        bonds = {frozenset((str(locants[b.GetBeginAtomIdx()]), str(locants[b.GetEndAtomIdx()])))
                 for b in query.GetBonds()}
        if elements != dict(component.label_elements) or bonds != set(component.bonds):
            continue
        seen.add(name)
        result.append(template)
    return tuple(result)


def _bridge_prefix(atoms):
    """P-25.4.2.1.1/.4 preselected names for a homogeneous path."""
    from .constants import MULT, ROOTS

    elements = {atom.GetAtomicNum() for atom in atoms}
    if len(elements) != 1:
        return None
    element, count = next(iter(elements)), len(atoms)
    if element == 6:
        stem = ROOTS.get(count)
        return stem + 'ano' if stem else None
    if element == 8:
        return {1: 'epoxy', 2: 'epidioxy', 3: 'epitrioxy'}.get(count)
    if element in {16, 34, 52}:
        base = {16: 'sulfano', 34: 'selano', 52: 'tellano'}[element]
        return base if count == 1 else MULT.get(count, '') + base if count in MULT else None
    if element == 7:
        return {1: 'azano', 2: 'diazano', 3: 'epitriazano'}.get(count)
    return None


def _eligible(atom):
    if (atom.GetIsAromatic() or atom.GetDegree() != 2 or atom.GetFormalCharge()
            or atom.GetIsotope() or atom.GetNumRadicalElectrons()
            or int(atom.GetChiralTag()) or not atom.IsInRing()
            or any(b.GetBondTypeAsDouble() != 1.0 for b in atom.GetBonds())):
        return False
    number = atom.GetAtomicNum()
    return ((number == 6 and atom.GetTotalNumHs(includeNeighbors=True) == 2)
            or (number in {8, 16, 34, 52} and atom.GetTotalNumHs(includeNeighbors=True) == 0
                and atom.GetTotalValence() == 2)
            or (number == 7 and atom.GetTotalNumHs(includeNeighbors=True) == 1
                and atom.GetTotalValence() == 3))


def _bridge_paths(mol):
    """Enumerate distinct homogeneous degree-two path subsets and boundaries."""
    eligible = {atom.GetIdx() for atom in mol.GetAtoms() if _eligible(atom)}
    seen, emitted = set(), set()
    for first in eligible:
        number = mol.GetAtomWithIdx(first).GetAtomicNum()
        stack = [(first, (first,))]
        while stack:
            current, path = stack.pop()
            group = frozenset(path)
            state = (current, group)
            if state in seen:
                continue
            seen.add(state)
            boundary = [(idx, neighbor.GetIdx()) for idx in group
                        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                        if neighbor.GetIdx() not in group]
            if (group not in emitted and len(boundary) == 2 and boundary[0][1] != boundary[1][1]
                    and _bridge_prefix([mol.GetAtomWithIdx(i) for i in path])):
                emitted.add(group)
                yield tuple(path), tuple(boundary)
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                nxt = neighbor.GetIdx()
                if (nxt in eligible and nxt not in group
                        and neighbor.GetAtomicNum() == number):
                    stack.append((nxt, path + (nxt,)))


def _different_ring_bridge(query, rings, endpoints):
    """Prove a P-25.4.1.2(b) bridge does not create a peri-fused face."""
    if not all(any(head in ring for ring in rings) for head in endpoints):
        return False
    ring_sets = tuple(frozenset(ring) for ring in rings)
    graph = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()} for a in query.GetAtoms()}
    stack = [(endpoints[0], (endpoints[0],))]
    calls = 0
    while stack:
        current, path = stack.pop()
        calls += 1
        if calls > 8192:
            return False
        used = frozenset(path)
        # A face fused to an existing face can share at most one bond with it.
        if any(len(used & ring) > 2 for ring in ring_sets):
            continue
        if current == endpoints[1]:
            for i, left in enumerate(ring_sets):
                a = used & left
                if len(a) != 2 or query.GetBondBetweenAtoms(*tuple(a)) is None:
                    continue
                for right in ring_sets[i+1:]:
                    b = used & right
                    shared = left & right
                    if (len(shared) == 2 and query.GetBondBetweenAtoms(*tuple(shared)) is not None
                            and len(b) == 2 and query.GetBondBetweenAtoms(*tuple(b)) is not None
                            and len(a | b) == 3):
                        return False
            continue
        stack.extend((nxt, path + (nxt,)) for nxt in graph[current] if nxt not in used)
    return True


def _generated_core_can_support_bridges(core, original, boundaries):
    """Apply necessary bridge/core checks before naming a generated core.

    Every generated template uses this very core in identity atom order.
    The same fused-core and endpoint conditions are checked again by the
    relevant bridge decision once a name and locants exist. A failure here
    therefore cannot discard a generated template that decision could use.
    Retained templates with their own atom mappings are not filtered here.
    """
    rings = tuple(tuple(ring) for ring in core.GetRingInfo().AtomRings())
    if not _proof_is_consistent(core, '', FusedCoreProof('', rings)):
        return False
    positions = {atom: index for index, atom in enumerate(original)}
    for boundary in boundaries:
        if len(boundary) != 2 or any(head not in positions for _end, head in boundary):
            return False
        endpoints = tuple(positions[head] for _end, head in boundary)
        if endpoints[0] == endpoints[1]:
            return False
        common = [ring for ring in rings if set(endpoints) <= set(ring)]
        adjacent = core.GetBondBetweenAtoms(*endpoints) is not None
        if ((not common and not _different_ring_bridge(core, rings, endpoints))
                or (common and adjacent and len(common) < 2)):
            return False
    return True


def _simple_bridge_decision(mol, Chem, template, match, path, boundary, *,
                            named_bridge=None, bridge_orders=None, require_pi_proof=False):
    from .bridged_fused_methods import RULE
    from .naming_methods import MethodDecision

    query, name, locants = template['mol'], template['name'], template['locants']
    rings = tuple(tuple(ring) for ring in query.GetRingInfo().AtomRings())
    if not _proof_is_consistent(query, name, FusedCoreProof(name, rings)):
        return None
    if (set(locants) != set(range(query.GetNumAtoms()))
            or any(a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or not a.IsInRing() for a in query.GetAtoms())):
        return None
    core, bridge = frozenset(match), frozenset(path)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if core & bridge or core | bridge != heavy:
        return None
    exact_core = _same_core_graph(mol, query, match)
    reconstructed = (not exact_core
                     and _reconstructed_unchanged_graph(mol, Chem, query, match, bridge,
                                                        bridge_orders=bridge_orders))
    by_original = {original: index for index, original in enumerate(match)}
    if any(head not in by_original for _end, head in boundary):
        return None
    endpoints = [by_original[head] for _end, head in boundary]
    if endpoints[0] == endpoints[1]:
        return None
    endpoint_set = set(endpoints)
    common_rings = [ring for ring in rings if endpoint_set <= set(ring)]
    adjacent = query.GetBondBetweenAtoms(*endpoints) is not None
    # P-25.4.1.2(a), (b) and (d): same-ring nonadjacent ends,
    # different-ring ends without a new peri fusion, or a fused bond.
    if (not common_rings and not _different_ring_bridge(query, rings, endpoints)
            or common_rings and adjacent and len(common_rings) < 2):
        return None
    head_labels = [str(locants[index]) for index in endpoints]
    keys = [_locant_key(label) for label in head_labels]
    if any(key is None for key in keys) or keys[0] == keys[1]:
        return None
    start = int(_next_bridge_locant(locants) or 0)
    if not start:
        return None
    bridge_name = named_bridge or _bridge_prefix([mol.GetAtomWithIdx(i) for i in path])
    core_name = (_core_name_on_original_graph(mol, template, match)
                 if exact_core and not require_pi_proof else None)
    if core_name is None and (exact_core or reconstructed):
        # Aromaticity perception can include a heteroatom bridge in another
        # ring. The reconstructed parent plus single bridge bonds is checked
        # against the whole input before accepting the unchanged parent.
        same_hydrogens = all(
            mol.GetAtomWithIdx(atom).GetTotalNumHs(includeNeighbors=True)
            + sum(n.GetIdx() not in core for n in mol.GetAtomWithIdx(atom).GetNeighbors())
            == query.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True)
            for i, atom in enumerate(match))
        implicit_indicated_h = any(
            a.GetAtomicNum() == 6 and not a.GetIsAromatic() and a.GetTotalNumHs() == 2
            and all(b.GetBondTypeAsDouble() == 1.0 for b in a.GetBonds())
            for a in query.GetAtoms())
        if (same_hydrogens and not implicit_indicated_h
                and not require_pi_proof
                and _reconstructed_unchanged_graph(mol, Chem, query, match, bridge,
                                                   bridge_orders=bridge_orders)):
            core_name = name
    if core_name is None:
        core_name = _bridge_aware_aromatic_core_name(mol, Chem, template, match, bridge,
                                                     bridge_orders=bridge_orders)
    if not bridge_name or not core_name:
        return None
    indicated, unindicated_name = _without_indicated_hydrogen(core_name)
    _, unindicated_template = _without_indicated_hydrogen(name)
    if not unindicated_name.endswith(unindicated_template):
        return None
    pre_core = unindicated_name[:-len(unindicated_template)]
    if named_bridge:
        by_end = {end: str(locants[by_original[head]]) for end, head in boundary}
        if path[0] not in by_end or path[-1] not in by_end:
            return None
        attachment = (by_end[path[0]], by_end[path[-1]])
    else:
        attachment = tuple(label for _key, label in sorted(zip(keys, head_labels)))
    bridge_term = ','.join(attachment) + '-' + bridge_name
    separator = '-' if pre_core and not pre_core.endswith('-') else ''
    rendered = (indicated + '-' if indicated else '') + pre_core + separator + bridge_term + unindicated_template
    # P-25.4.4 numbers an acyclic bridge from the higher-locanted bridgehead.
    high = 0 if keys[0] > keys[1] else 1
    first = boundary[high][0]
    ordered = tuple(path if path[0] == first else reversed(path))
    if ordered[0] != first:
        return None
    parent_labels = tuple((original, str(locants[i])) for i, original in enumerate(match))
    bridge_labels = tuple((atom, str(start+i)) for i, atom in enumerate(ordered))
    return MethodDecision(
        name=rendered, method='bridged-fused', principal_centers=frozenset(), atoms=heavy,
        rule=RULE + '/P-25.4.2.1.4/P-25.4.3.3/P-25.4.3.4.2',
        atom_locants=parent_labels + bridge_labels,
        component_roles=(('fused_core', core, parent_labels),
                         ('homogeneous_bridge', bridge, bridge_labels)))


def _reconstructed_unchanged_graph(mol, Chem, query, match, bridge, *, bridge_orders=None):
    """Check the named reference core plus single bridge against the input."""
    rebuilt = Chem.RWMol()
    for original in mol.GetAtoms():
        atom = Chem.Atom(original.GetAtomicNum())
        atom.SetFormalCharge(original.GetFormalCharge())
        atom.SetNumExplicitHs(original.GetTotalNumHs(includeNeighbors=True))
        atom.SetNoImplicit(True)
        rebuilt.AddAtom(atom)
    for bond in query.GetBonds():
        rebuilt.AddBond(match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()], bond.GetBondType())
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in bridge or b in bridge:
            order = (bridge_orders or {}).get(frozenset((a, b)), Chem.BondType.SINGLE)
            rebuilt.AddBond(a, b, order)
    candidate = rebuilt.GetMol()
    try:
        Chem.SanitizeMol(candidate)
    except (ValueError, RuntimeError):
        return False
    return (Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)
            == Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def _bridge_aware_aromatic_core_name(mol, Chem, template, match, bridge, *, bridge_orders=None):
    """Assign maximum noncumulative bonds after bridge attachment.

    This handles retained parents with carbon and divalent chalcogens, when
    their template already has the maximum noncumulative bond count. A source
    graph is accepted only if all assignments compatible with its hydrogen
    state sanitize to that graph; otherwise localization remains unresolved.
    """
    from .constants import MULT

    query, locants = template['mol'], template['locants']
    valence = {6: 4, 8: 2, 16: 2, 34: 2, 52: 2}
    if any(a.GetAtomicNum() not in valence for a in query.GetAtoms()):
        return None
    core = frozenset(match)
    edges = frozenset(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                      for b in query.GetBonds())
    neighbors = {i: {b if a == i else a for a, b in edges if i in (a, b)}
                 for i in range(query.GetNumAtoms())}
    capacity = tuple(valence[query.GetAtomWithIdx(i).GetAtomicNum()] - len(neighbors[i]) - sum(n.GetIdx() not in core
                     for n in mol.GetAtomWithIdx(original).GetNeighbors())
                     for i, original in enumerate(match))
    hydrogens = tuple(mol.GetAtomWithIdx(original).GetTotalNumHs(includeNeighbors=True)
                      for original in match)
    needed = tuple(capacity[i] - hydrogens[i] for i in range(len(match)))
    if any(capacity[i] < 0 or needed[i] not in (0, 1) for i in range(len(match))):
        return None
    calls = [0]

    def charge():
        calls[0] += 1
        if calls[0] > 8192:
            raise OverflowError

    @lru_cache(maxsize=None)
    def max_size(remaining):
        charge()
        if not remaining:
            return 0
        first = min(remaining)
        rest = remaining - {first}
        return max([max_size(rest)] + [1 + max_size(rest - {other})
                   for other in neighbors[first] & rest])

    def maxima(remaining):
        charge()
        if not remaining:
            yield frozenset()
            return
        first = min(remaining)
        rest = remaining - {first}
        maximum = max_size(remaining)
        if max_size(rest) == maximum:
            yield from maxima(rest)
        for other in neighbors[first] & rest:
            if 1 + max_size(rest - {other}) == maximum:
                for following in maxima(rest - {other}):
                    yield following | {tuple(sorted((first, other)))}

    def perfect(remaining):
        charge()
        if not remaining:
            yield frozenset()
            return
        first = min(remaining)
        for other in neighbors[first] & remaining:
            for following in perfect(remaining - {first, other}):
                yield following | {tuple(sorted((first, other)))}

    def identity(pairs):
        graph = Chem.RWMol()
        for original in mol.GetAtoms():
            atom = Chem.Atom(original.GetAtomicNum())
            atom.SetNumExplicitHs(original.GetTotalNumHs(includeNeighbors=True))
            atom.SetNoImplicit(True)
            graph.AddAtom(atom)
        for a, b in edges:
            graph.AddBond(match[a], match[b],
                          Chem.BondType.DOUBLE if (a, b) in pairs else Chem.BondType.SINGLE)
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in bridge or b in bridge:
                order = (bridge_orders or {}).get(frozenset((a, b)), Chem.BondType.SINGLE)
                graph.AddBond(a, b, order)
        candidate = graph.GetMol()
        try:
            Chem.SanitizeMol(candidate)
        except (ValueError, RuntimeError):
            return None
        return Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)

    try:
        # A saturated or partially hydrogenated resource template must not be
        # promoted as a mancude parent by this bridge-aware reassignment.
        reference_query = Chem.Mol(query)
        Chem.Kekulize(reference_query, clearAromaticFlags=True)
        reference_double_count = sum(b.GetBondTypeAsDouble() == 2.0
                                     for b in reference_query.GetBonds())
        free = frozenset(i for i, atom in enumerate(query.GetAtoms())
                         if valence[atom.GetAtomicNum()] > len(neighbors[i]))
        if max_size(free) != reference_double_count:
            return None
        target = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        forms = {identity(pairs) for pairs in perfect(frozenset(i for i, n in enumerate(needed) if n))}
        forms.discard(None)
        if forms != {target}:
            from .fused_pi_source_evidence import published_unqualified_hydride
            if target not in forms or not published_unqualified_hydride(target):
                return None
        eligible = frozenset(i for i, value in enumerate(capacity) if value)
        records = []
        for pairs in maxima(eligible):
            matched = {i for edge in pairs for i in edge}
            reference_h = tuple(capacity[i] - (i in matched) for i in range(len(match)))
            added = tuple(hydrogens[i] - reference_h[i] for i in range(len(match)))
            if any(value not in (0, 1) for value in added):
                continue
            indicated = tuple(sorted((str(locants[i]) for i in eligible
                                      if reference_h[i] and i not in matched), key=_locant_key))
            count = sum(added)
            if count and (count not in MULT or count % 2):
                continue
            if count and indicated:
                continue  # Combined hydro and indicated-H placement remains separate.
            labels = tuple(sorted((str(locants[i]) for i, value in enumerate(added) if value),
                                  key=_locant_key))
            records.append(((tuple(_locant_key(label) for label in indicated),
                             tuple(_locant_key(label) for label in labels)), indicated, labels))
    except (OverflowError, ValueError, RuntimeError):
        return None
    if not records:
        return None
    _key, indicated, labels = min(records)
    _old_h, parent = _without_indicated_hydrogen(template['name'])
    return ((','.join(label+'H' for label in indicated) + '-') if indicated else '') + (
        (','.join(labels) + '-' + MULT[len(labels)] + 'hydro') if labels else '') + parent


def _constructed_core_templates(core, Chem):
    """Generate a numbered fused hydrocarbon core from its component graph."""
    if (core.GetRingInfo().NumRings() < 3 or any(not atom.IsInRing()
            or atom.GetAtomicNum() != 6 for atom in core.GetAtoms())):
        return ()
    from .primitive_three_face_fusion import general_component_fusion
    parent = general_component_fusion(core, Chem)
    if parent is None or 'Δ^' in parent.name or 'hydro' in parent.name:
        return ()
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fused_orientation import fused_orientation_frontier
    options = dict(allow_three_face_n_junction=True, allow_multi_face_n_junction=True,
                   allow_larger_system=True, allow_shared_carbon_junctions=True,
                   allow_peri_fused=True)
    drawing = fused_orientation_frontier(core, **options)
    if not drawing.admitted:
        return ()
    numbered = peripheral_numbering_frontier(core, orientation_seeds=drawing.orientation_seeds,
                                             **options)
    if not numbered.admitted:
        return ()
    indicated, _name = _without_indicated_hydrogen(parent.name)
    indicated_atoms = {atom.GetIdx() for atom in core.GetAtoms()
                       if not atom.GetIsAromatic() and atom.GetTotalNumHs() == 2
                       and all(bond.GetBondTypeAsDouble() == 1 for bond in atom.GetBonds())}
    result = []
    for row in numbered.candidates:
        locants = dict(row.atom_locants)
        if any(_locant_key(value) is None for value in locants.values()):
            continue
        h = ','.join(value+'H' for value in sorted((locants[i] for i in indicated_atoms),
                                                    key=_locant_key))
        if h != indicated:
            continue
        result.append({'mol': core, 'name': parent.name, 'locants': locants})
    return tuple(result)


def select_simple_fused_bridge(mol, Chem):
    """Compare all admitted homogeneous bridge removals and retained cores."""
    from dataclasses import replace
    from collections import Counter
    from .molecule_iteration import iter_atoms
    from .naming_context import MoleculeContext
    from .naming_result import SelectionLimitation
    from .parent_rules import _cached_iter_fused_template_matches
    from .parent_selection import ring_seniority_key

    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or not 7 <= mol.GetNumAtoms() <= 96
            or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetNumBonds()-mol.GetNumAtoms()+1 < 3
            or any(a.GetAtomicNum() not in {6, 7, 8, 16, 34, 52} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
                   or not a.IsInRing() for a in iter_atoms(mol))
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    # RDKit may mark a bridge as aromatic when it contributes to an
    # overlapping ring. Kekulization recovers the bridge's single bonds;
    # every emitted name is subsequently checked against the original graph.
    working = Chem.Mol(mol)
    try:
        Chem.Kekulize(working, clearAromaticFlags=True)
    except (ValueError, RuntimeError):
        return None
    paths = tuple(_bridge_paths(working))
    if not paths:
        return None
    templates = _bridge_templates(Chem)

    def matches(graph):
        ctx = MoleculeContext(graph, Chem)
        return _cached_iter_fused_template_matches(
            ctx, ctx.candidate_templates(templates), Chem, topology_query=True,
            require_one_ring_system=True, reject_extra_core_bonds=True)

    if any(len(match) == mol.GetNumAtoms() for _template, match in matches(mol)):
        return None
    rows = []
    for path, boundary in paths:
        original = tuple(i for i in range(mol.GetNumAtoms()) if i not in path)
        rw = Chem.RWMol(working)
        for index in sorted(path, reverse=True):
            rw.RemoveAtom(index)
        core_graph = rw.GetMol()
        try:
            Chem.SanitizeMol(core_graph)
        except (ValueError, RuntimeError):
            continue
        if len(Chem.GetMolFrags(core_graph)) != 1:
            continue
        retained_matches = tuple((template, match) for template, match in matches(core_graph)
                                 if len(match) == len(original))
        generated = () if (retained_matches or not _generated_core_can_support_bridges(
            core_graph, original, (boundary,))) else tuple(
            (template, tuple(range(core_graph.GetNumAtoms())))
            for template in _constructed_core_templates(core_graph, Chem))
        for template, core_match in retained_matches + generated:
            if len(core_match) != len(original):
                continue
            query = template['mol']
            match = tuple(original[i] for i in core_match)
            decision = _simple_bridge_decision(mol, Chem, template, match, path, boundary)
            if decision is None:
                continue
            mapping = dict(decision.atom_locants)
            head_key = tuple(sorted(_locant_key(mapping[head]) for _end, head in boundary))
            rings = tuple(tuple(ring) for ring in query.GetRingInfo().AtomRings())
            elements = tuple(sorted(Counter(a.GetAtomicNum() for a in query.GetAtoms()).items()))
            core_key = (-len(rings), -query.GetNumAtoms(),
                        sum(a.GetAtomicNum() != 6 for a in query.GetAtoms()),
                        ring_seniority_key(elements, len(rings)),
                        tuple(-size for size in sorted(map(len, rings), reverse=True)))
            rows.append((core_key, template['name'], head_key, decision))
    if not rows:
        return None
    best = min(row[0] for row in rows)
    rows = [row for row in rows if row[0] == best]
    if len({row[1] for row in rows}) != 1:
        return None
    best_heads = min(row[2] for row in rows)
    choices = [row[3] for row in rows if row[2] == best_heads]
    if len({choice.name for choice in choices}) != 1:
        return None
    ranks = tuple(Chem.CanonicalRankAtoms(mol, breakTies=True))
    chosen = min(choices, key=lambda row: tuple(sorted((locant, ranks[idx])
                                                       for idx, locant in row.atom_locants)))
    return replace(chosen, preference_issues=(SelectionLimitation(
        'ring-construction', 'P-25.4.3.4/P-25.4.3.4.2',
        'Mapped homogeneous bridge and discovered fused-core candidates compared; '
        'other fused vocabularies and bridge decompositions remain unresolved.'),))
