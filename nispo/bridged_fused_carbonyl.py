"""Mapped hydro and added-hydrogen names for a bounded bridged fused ketone.

P-25.4.3.4.1 assigns the maximum number of noncumulative double bonds
*after* insertion of a bridge. P-58.2.2 then accommodates a ketone suffix by
added indicated hydrogen; P-58.2.4 introduces exocyclic prefixes after that
operation.  This producer works from the original atom/bond graph, retaining
all maximum matchings of its numbered fused core.  It currently admits one
unsubstituted methano bridge, one core ketone, and terminal methylidenes.
It deliberately declines annotation, unproved indicated-hydrogen parent
states, other decorations, and incomplete decomposition comparisons.
"""
from __future__ import annotations

from .bridged_fused_methods import FusedCoreProof, _locant_key, _next_bridge_locant, _proof_is_consistent
from .constants import MULT
from .formatting import format_prefixes
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def _perfect_matchings(vertices, edges, *, limit=4096):
    """Enumerate the full perfect-matching frontier, or decline on overflow."""
    adjacency = {i: set() for i in vertices}
    for edge in edges:
        left, right = tuple(edge)
        adjacency[left].add(right)
        adjacency[right].add(left)
    answers = []
    overflow = False

    def visit(remaining, chosen):
        nonlocal overflow
        if overflow:
            return
        if not remaining:
            answers.append(frozenset(chosen))
            overflow = len(answers) > limit
            return
        left = min(remaining, key=lambda i: (len(adjacency[i] & remaining), i))
        for right in sorted(adjacency[left] & remaining):
            visit(remaining - {left, right}, chosen + [frozenset((left, right))])

    if len(vertices) % 2:
        return None
    visit(set(vertices), [])
    return None if overflow else tuple(answers)


def _full_fused_ring_covered(mol, Chem):
    """A complete ordinary fused construction precedes bridged fusion."""
    from .naming_context import MoleculeContext
    from .parent_rules import _cached_iter_fused_template_matches
    from .ring_state_parent import _fused_templates

    ring = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
    rw = Chem.RWMol(mol)
    for i in sorted(set(range(mol.GetNumAtoms())) - ring, reverse=True):
        rw.RemoveAtom(i)
    graph = rw.GetMol()
    try:
        Chem.SanitizeMol(graph)
    except (ValueError, RuntimeError):
        # A failed proof cannot establish that a full fused parent is absent.
        return True
    ctx = MoleculeContext(graph, Chem)
    templates = _fused_templates(Chem)
    return any(len(match) == len(ring) for _template, match in
               _cached_iter_fused_template_matches(
                   ctx, ctx.candidate_templates(templates), Chem, topology_query=True,
                   require_one_ring_system=True, reject_extra_core_bonds=True))


def _candidate_rows(mol, Chem, bridge_id, oxygen_id, carbonyl_id, methylenes, kekulized):
    from .naming_context import MoleculeContext
    from .parent_rules import _cached_iter_fused_template_matches
    from .ring_state_parent import _fused_templates

    core = frozenset(a.GetIdx() for a in mol.GetAtoms()
                     if a.IsInRing() and a.GetIdx() != bridge_id)
    if not 8 <= len(core) <= 16:
        return ()
    original = tuple(sorted(core))
    rw = Chem.RWMol(mol)
    for atom_id in sorted(set(range(mol.GetNumAtoms())) - core, reverse=True):
        rw.RemoveAtom(atom_id)
    core_graph = rw.GetMol()
    try:
        Chem.SanitizeMol(core_graph)
    except (ValueError, RuntimeError):
        return ()
    ctx = MoleculeContext(core_graph, Chem)
    templates = _fused_templates(Chem)
    matches = _cached_iter_fused_template_matches(
        ctx, ctx.candidate_templates(templates), Chem, topology_query=True,
        require_one_ring_system=True, reject_extra_core_bonds=True)
    rows = []
    for template, core_match in matches:
        if len(core_match) != len(core):
            continue
        query = template['mol']
        name = template['name']
        locants = template['locants']
        if (not name.endswith('e') or set(locants) != set(range(query.GetNumAtoms()))
                or any(a.GetAtomicNum() != 6 or not a.GetIsAromatic()
                       or a.GetFormalCharge() or a.GetIsotope() for a in query.GetAtoms())):
            # The all-aromatic template proves an ordinary mancude fused
            # hydride; templates with pre-existing indicated H are not covered.
            continue
        rings = tuple(tuple(ring) for ring in query.GetRingInfo().AtomRings())
        if not _proof_is_consistent(query, name, FusedCoreProof(name, rings)):
            continue
        match = tuple(original[i] for i in core_match)
        if set(match) != set(core):
            continue
        mapping = {atom_id: str(locants[j]) for j, atom_id in enumerate(match)}
        keys = {atom_id: _locant_key(loc) for atom_id, loc in mapping.items()}
        if any(key is None for key in keys.values()):
            continue
        bridge_neighbors = tuple(n.GetIdx() for n in mol.GetAtomWithIdx(bridge_id).GetNeighbors())
        if len(bridge_neighbors) != 2 or not set(bridge_neighbors) <= core:
            continue
        reverse = {atom_id: j for j, atom_id in enumerate(match)}
        endpoint_ids = {reverse[atom_id] for atom_id in bridge_neighbors}
        if (query.GetBondBetweenAtoms(*endpoint_ids) is not None
                or not any(endpoint_ids <= set(ring) for ring in rings)):
            continue
        template_edges = {frozenset((match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()]))
                          for b in query.GetBonds()}
        source_edges = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                        for b in mol.GetBonds()
                        if b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core}
        if template_edges != source_edges:
            continue
        if any(mol.GetAtomWithIdx(i).GetDegree() > 3 for i in core):
            continue
        actual_double = {edge for edge in source_edges
                         if kekulized.GetBondBetweenAtoms(*edge).GetBondTypeAsDouble() == 2}
        # Fully saturated cores have separate locant-omission and hydro
        # parent choices which this producer has not proved.
        if not actual_double:
            continue
        if any(kekulized.GetBondBetweenAtoms(*edge).GetBondTypeAsDouble() not in (1, 2)
               for edge in source_edges):
            continue
        frontier = _perfect_matchings(core, source_edges)
        if not frontier:
            continue
        bridge_locant = _next_bridge_locant(locants)
        if bridge_locant is None:
            continue
        endpoints = tuple(sorted((keys[i] for i in bridge_neighbors)))
        core_key = (-len(rings), -len(core),
                    tuple(-size for size in sorted(map(len, rings), reverse=True)))
        for parent_double in frontier:
            if not actual_double <= parent_double:
                continue
            missing = parent_double - actual_double
            suffix_bonds = [edge for edge in missing if carbonyl_id in edge]
            if len(suffix_bonds) != 1:
                continue
            suffix_bond = suffix_bonds[0]
            added = next(iter(suffix_bond - {carbonyl_id}))
            if mol.GetAtomWithIdx(added).GetTotalNumHs() < 1:
                continue
            hydro_atoms = [i for edge in missing - {suffix_bond} for i in edge]
            if len(hydro_atoms) not in MULT and hydro_atoms:
                continue
            hydro_locs = tuple(sorted((keys[i], mapping[i]) for i in hydro_atoms))
            methyl_locs = tuple(sorted((keys[i], mapping[i], carbon)
                                       for i, carbon in methylenes))
            methyl_prefix = format_prefixes([(loc, 'methylidene') for _, loc, _ in methyl_locs])
            hydro = (','.join(loc for _, loc in hydro_locs) + '-'
                     + MULT[len(hydro_atoms)] + 'hydro-') if hydro_atoms else ''
            endpoint_text = ','.join(mapping[i] for i in sorted(bridge_neighbors, key=keys.get))
            parent = (f'{hydro}{endpoint_text}-methano{name[:-1]}'
                      f'-{mapping[carbonyl_id]}({mapping[added]}H)-one')
            rendered = methyl_prefix + '-' + parent if methyl_prefix else parent
            # P-25.4.3.3 fixes bridge locants before ordinary substitution
            # numbering. Added H is preferred at a peripheral core position.
            q_added = query.GetAtomWithIdx(reverse[added])
            chemical_key = (core_key, endpoints, keys[carbonyl_id],
                            q_added.GetDegree() != 2, keys[added],
                            tuple(key for key, _ in hydro_locs),
                            tuple(key for key, _, _ in methyl_locs))
            atom_locants = tuple((i, mapping[i]) for i in match) + ((bridge_id, bridge_locant),)
            roles = (("fused_core", core, atom_locants[:-1]),
                     ("methano_bridge", frozenset({bridge_id}), ((bridge_id, bridge_locant),)),
                     ("carbonyl_oxygen", frozenset({oxygen_id}), ()),
                     ("methylidene", frozenset(carbon for _, carbon in methylenes), ()))
            rows.append((chemical_key, name, rendered, atom_locants, roles))
    return tuple(rows)


def bridged_fused_ketone_decision(mol, Chem):
    """Return a mapped bridged-fused ketone, or ``None`` outside proved scope.

    This is a partial naming-method proposal; it does not prove the complete
    P-44 frontier and records that limitation on the returned decision.
    """
    if (mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in (6, 8) or a.GetIsotope()
                   or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    if not 10 <= mol.GetNumAtoms() <= 20 or mol.GetRingInfo().NumRings() < 3:
        return None
    oxygen = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(oxygen) != 1 or oxygen[0].IsInRing() or oxygen[0].GetDegree() != 1:
        return None
    oxygen_id = oxygen[0].GetIdx()
    carbonyl = oxygen[0].GetNeighbors()[0]
    carbonyl_id = carbonyl.GetIdx()
    if (not carbonyl.IsInRing() or carbonyl.GetAtomicNum() != 6
            or mol.GetBondBetweenAtoms(oxygen_id, carbonyl_id).GetBondTypeAsDouble() != 2):
        return None
    methylenes = []
    for atom in mol.GetAtoms():
        if atom.IsInRing() or atom.GetIdx() == oxygen_id:
            continue
        if (atom.GetAtomicNum() != 6 or atom.GetDegree() != 1
                or atom.GetTotalNumHs() != 2):
            return None
        neighbor = atom.GetNeighbors()[0]
        if (not neighbor.IsInRing()
                or mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2):
            return None
        methylenes.append((neighbor.GetIdx(), atom.GetIdx()))
    if len(methylenes) > 3:
        return None
    bridges = [a.GetIdx() for a in mol.GetAtoms()
               if a.IsInRing() and a.GetAtomicNum() == 6 and not a.GetIsAromatic()
               and a.GetDegree() == 2 and a.GetTotalNumHs() == 2
               and all(b.GetBondTypeAsDouble() == 1 for b in a.GetBonds())]
    if not bridges or _full_fused_ring_covered(mol, Chem):
        return None
    kekulized = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekulized, clearAromaticFlags=True)
    except (ValueError, RuntimeError):
        return None
    rows = [row for bridge in bridges for row in
            _candidate_rows(mol, Chem, bridge, oxygen_id, carbonyl_id, tuple(methylenes), kekulized)]
    if not rows:
        return None
    best_core = min(row[0][0] for row in rows)
    rows = [row for row in rows if row[0][0] == best_core]
    if len({row[1] for row in rows}) != 1:
        # Later P-44 fused-system criteria, not bridge/suffix locants or
        # spelling, must resolve distinct surviving parent vocabularies.
        return None
    best = min(row[0] for row in rows)
    winners = [row for row in rows if row[0] == best]
    if len({(row[1], row[2]) for row in winners}) != 1:
        return None
    ranks = tuple(Chem.CanonicalRankAtoms(mol, breakTies=True))
    chosen = min(winners, key=lambda row: tuple(sorted((loc, ranks[i]) for i, loc in row[3])))
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    roles = chosen[4]
    if (set().union(*(set(role[1]) for role in roles)) != set(heavy)
            or sum(len(role[1]) for role in roles) != len(heavy)):
        return None
    return MethodDecision(
        name=chosen[2], method='bridged-fused',
        principal_centers=frozenset((carbonyl_id, oxygen_id)), atoms=heavy,
        rule='P-25.4.3.4.1/P-25.4.3.3/P-44.2.2/P-58.2.2/P-58.2.4',
        atom_locants=chosen[3], component_roles=roles,
        preference_issues=(SelectionLimitation(
            'ring-construction', 'P-25.4.3.4/P-44.2.2',
            'The retained all-carbon fused core, methano bridge, hydro state and '
            'ketone are mapped; competing fused vocabularies and bridge '
            'decompositions outside this bounded search remain unproved.'),))
