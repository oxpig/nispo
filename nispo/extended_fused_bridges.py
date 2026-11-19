"""Graph-discovered composite and unsaturated divalent fused-ring bridges.

The paths and their attachment orientations are derived from the source graph.
Core names are independently constructed/numbered components. Unknown
localized states, incomplete core comparisons, and unsupported branch or
stereo cases defer to the existing selector.
"""
from collections import Counter
from dataclasses import replace

from .simple_fused_bridges import (
    _bridge_prefix, _bridge_templates, _constructed_core_templates, _eligible,
    _generated_core_can_support_bridges, _simple_bridge_decision,
)


def _bridge_atom(atom):
    if _eligible(atom):
        return True
    if (atom.GetAtomicNum() != 6 or atom.GetIsAromatic() or not atom.IsInRing()
            or atom.GetDegree() != 2 or atom.GetFormalCharge() or atom.GetIsotope()
            or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag())
            or atom.GetTotalNumHs(includeNeighbors=True) != 1):
        return False
    return sorted(b.GetBondTypeAsDouble() for b in atom.GetBonds()) == [1.0, 2.0]


def _bridge_specs(mol, path):
    """Return source-ranked name and orientations of a divalent path."""
    from .constants import MULT, ROOTS
    records = []
    for ordered in (tuple(path), tuple(reversed(path))):
        atoms = [mol.GetAtomWithIdx(i) for i in ordered]
        elements = [a.GetAtomicNum() for a in atoms]
        bonds = [mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
                 for a, b in zip(ordered, ordered[1:])]
        if any(order not in (1.0, 2.0) for order in bonds):
            continue
        if len(set(elements)) == 1 and elements[0] == 6 and 2.0 in bonds:
            positions = tuple(i + 1 for i, order in enumerate(bonds) if order == 2.0)
            if any(b == a + 1 for a, b in zip(positions, positions[1:])):
                continue
            root = ROOTS.get(len(atoms))
            if not root or len(positions) not in MULT and len(positions) != 1:
                continue
            if len(atoms) == 2 and positions == (1,):
                word = 'etheno'
            elif len(positions) == 1:
                word = root + '[' + str(positions[0]) + ']eno'
            else:
                word = root + 'a[' + ','.join(map(str, positions)) + ']' + MULT[len(positions)] + 'eno'
            records.append(((0, positions), word, ordered))
            continue
        if len(set(elements)) < 2 or any(order != 1.0 for order in bonds):
            continue
        segments = []
        for atom in atoms:
            if not segments or atom.GetAtomicNum() != segments[-1][0]:
                segments.append((atom.GetAtomicNum(), [atom]))
            else:
                segments[-1][1].append(atom)
        words = [_bridge_prefix(group) for _, group in segments]
        if any(not word for word in words):
            continue
        # P-25.4.2.3.1 ranks terminal simple segments by heteroatom,
        # chain length, then alphabetically. Internal segments follow in
        # their physical order and omit a noninitial epi prefix.
        priority = {8: 0, 16: 1, 34: 2, 52: 3, 7: 4, 6: 5}
        first, group = segments[0]
        if first not in priority:
            continue
        # Before an initial i or o the bridge prefix is written "ep", so
        # both epidioxy -> dioxy and epoxy -> oxy need elision here.
        tail = [word[3:] if word.startswith('epi') else
                word[2:] if word.startswith('epo') else word for word in words[1:]]
        word = '(' + words[0] + ''.join(tail) + ')'
        records.append(((1, priority[first], -len(group), tuple(words)), word, ordered))
    if not records:
        return ()
    best = min(row[0] for row in records)
    return tuple((word, ordered) for score, word, ordered in records if score == best)


def _extended_paths(mol):
    eligible = {a.GetIdx() for a in mol.GetAtoms() if _bridge_atom(a)}
    seen, emitted, paths = set(), set(), []
    work = 0
    for first in eligible:
        stack = [(first, (first,))]
        while stack:
            current, path = stack.pop()
            group = frozenset(path)
            state = (current, group)
            if state in seen:
                continue
            seen.add(state)
            work += 1
            if work > 20000:
                return ()
            boundary = [(idx, neighbor.GetIdx()) for idx in group
                        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                        if neighbor.GetIdx() not in group]
            if (len(path) >= 2 and group not in emitted and len(boundary) == 2
                    and boundary[0][1] != boundary[1][1]
                    and all(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() == 1.0
                            for a, b in boundary)):
                specs = _bridge_specs(mol, path)
                if specs:
                    emitted.add(group)
                    paths.append((tuple(boundary), specs))
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                idx = neighbor.GetIdx()
                if idx in eligible and idx not in group:
                    stack.append((idx, path + (idx,)))
    return tuple(paths)


def select_extended_fused_bridge(mol, Chem):
    """Compare a single eligible composite or carbon-unsaturated bridge."""
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .fusion_components import resolve_component
    from .fusion_networks import parse_network
    from .fusion_state_skeletons import _two_component_preference, two_face_fusion_templates
    from .molecule_iteration import iter_atoms
    from .naming_context import MoleculeContext
    from .naming_result import SelectionLimitation
    from .parent_rules import _cached_iter_fused_template_matches
    from .parent_selection import ring_seniority_key
    from .simple_fused_bridges import _locant_key

    if (mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or not 7 <= mol.GetNumAtoms() <= 96
            or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetNumBonds() - mol.GetNumAtoms() + 1 < 3
            or any(a.GetAtomicNum() not in {6, 7, 8, 16, 34, 52} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
                   or not a.IsInRing() for a in iter_atoms(mol))
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    working = Chem.Mol(mol)
    try:
        Chem.Kekulize(working, clearAromaticFlags=True)
    except (ValueError, RuntimeError):
        return None
    # An established full fused drawing must be considered before bridging.
    graph = peripheral_numbering_frontier(
        mol, allow_three_face_n_junction=True, allow_multi_face_n_junction=True,
        allow_larger_system=True, allow_shared_carbon_junctions=True, allow_peri_fused=True)
    if graph.admitted:
        return None
    paths = _extended_paths(working)
    if not paths:
        return None
    templates = _bridge_templates(Chem)

    def matches(graph, candidates):
        ctx = MoleculeContext(graph, Chem)
        return tuple(_cached_iter_fused_template_matches(
            ctx, ctx.candidate_templates(candidates), Chem, topology_query=True,
            require_one_ring_system=True, reject_extra_core_bonds=True))

    if any(len(match) == mol.GetNumAtoms() for _template, match in matches(mol, templates)):
        return None
    rows = []
    for boundary, specs in paths:
        path = specs[0][1]
        original = tuple(i for i in range(mol.GetNumAtoms()) if i not in path)
        rw = Chem.RWMol(working)
        for index in sorted(path, reverse=True):
            rw.RemoveAtom(index)
        core = rw.GetMol()
        try:
            Chem.SanitizeMol(core)
        except (ValueError, RuntimeError):
            continue
        if len(Chem.GetMolFrags(core)) != 1 or any(not a.IsInRing() for a in core.GetAtoms()):
            continue
        eligible = tuple((template, match) for template, match in matches(core, templates)
                         if len(match) == len(original))
        if not eligible and core.GetRingInfo().NumRings() == 2:
            generated = two_face_fusion_templates(core, Chem)
            # These generated templates are all-single topology queries.
            # Rebind their numbered atoms to the actual capped core before
            # proving its maximum-bond state after bridge insertion.
            eligible = tuple((dict(template, mol=Chem.RenumberAtoms(core, tuple(match))), match)
                             for template, match in matches(core, generated)
                             if len(match) == len(original))
        if (not eligible and core.GetRingInfo().NumRings() >= 3
                and _generated_core_can_support_bridges(core, original, (boundary,))):
            eligible = tuple((template, tuple(range(core.GetNumAtoms())))
                             for template in _constructed_core_templates(core, Chem))
        orders = {frozenset((a, b)): working.GetBondBetweenAtoms(a, b).GetBondType()
                  for a, b in zip(path, path[1:])
                  if working.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1.0}
        for template, core_match in eligible:
            query = template['mol']
            match = tuple(original[i] for i in core_match)
            for prefix, oriented in specs:
                decision = _simple_bridge_decision(
                    mol, Chem, template, match, oriented, boundary,
                    named_bridge=prefix, bridge_orders=orders, require_pi_proof=True)
                if decision is None:
                    continue
                labels = dict(decision.atom_locants)
                endpoint_key = tuple(sorted(_locant_key(labels[head]) for _, head in boundary))
                by_end = {end: _locant_key(labels[head]) for end, head in boundary}
                directed_key = (by_end[oriented[0]], by_end[oriented[-1]])
                rings = tuple(tuple(r) for r in query.GetRingInfo().AtomRings())
                elements = tuple(sorted(Counter(a.GetAtomicNum() for a in query.GetAtoms()).items()))
                core_key = (-len(rings), -query.GetNumAtoms(),
                            sum(a.GetAtomicNum() != 6 for a in query.GetAtoms()),
                            ring_seniority_key(elements, len(rings)),
                            tuple(-n for n in sorted(map(len, rings), reverse=True)))
                rows.append((core_key, template['name'], endpoint_key, directed_key, decision))
    if not rows:
        return None
    key = min(row[0] for row in rows)
    rows = [row for row in rows if row[0] == key]
    names = {row[1] for row in rows}
    if len(names) > 1:
        networks = {name: parse_network(name, resolve_component) for name in names}
        if any(network is None for network in networks.values()):
            return None
        winners = {name for name, network in networks.items()
                   if all(_two_component_preference(network, other) in (-1, 0)
                          for other in networks.values())}
        if len(winners) != 1:
            return None
        rows = [row for row in rows if row[1] in winners]
    best = min((row[2], row[3]) for row in rows)
    choices = [row[4] for row in rows if (row[2], row[3]) == best]
    if len({choice.name for choice in choices}) != 1:
        return None
    ranks = tuple(Chem.CanonicalRankAtoms(mol, breakTies=True))
    chosen = min(choices, key=lambda row: tuple(sorted((locant, ranks[idx])
                                                       for idx, locant in row.atom_locants)))
    return replace(chosen, preference_issues=(SelectionLimitation(
        'ring-construction', 'P-25.4.3.4/P-25.4.3.4.2',
        'Composite and unsaturated path candidates were compared on available fused cores.'),))
