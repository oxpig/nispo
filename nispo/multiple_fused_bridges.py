"""Independent bridges on a discovered fused parent."""

from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
from math import comb
from .bridged_fused_methods import (
    RULE,
    FusedCoreProof,
    _locant_key,
    _next_bridge_locant,
    _proof_is_consistent,
    _without_indicated_hydrogen,
)
from .simple_fused_bridges import (
    _bridge_paths,
    _bridge_prefix,
    _bridge_templates,
    _bridge_aware_aromatic_core_name,
    _constructed_core_templates,
    _different_ring_bridge,
    _generated_core_can_support_bridges,
)


def _multi_decision(mol, Chem, template, mapping, paths, working):
    from .constants import MULT
    from .formatting import COMPLEX_MULT, prefix_alphabetization_key
    from .naming_methods import MethodDecision

    query, name, locants = template["mol"], template["name"], template["locants"]
    rings = tuple(tuple(r) for r in query.GetRingInfo().AtomRings())
    if not _proof_is_consistent(query, name, FusedCoreProof(name, rings)):
        return None
    if set(locants) != set(range(query.GetNumAtoms())):
        return None
    core = frozenset(mapping)
    bridge = frozenset(i for p, _, _, _ in paths for i in p)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if core & bridge or core | bridge != heavy:
        return None
    reverse = {a: i for i, a in enumerate(mapping)}
    groups = defaultdict(list)
    bridge_orders = {}
    for path, boundary, named_prefix, directed in paths:
        if any(head not in reverse for _, head in boundary):
            return None
        heads = [reverse[head] for _, head in boundary]
        if len(set(heads)) != 2:
            return None
        common = [r for r in rings if set(heads) <= set(r)]
        adjacent = query.GetBondBetweenAtoms(*heads) is not None
        if (
            not common
            and not _different_ring_bridge(query, rings, heads)
            or common
            and adjacent
            and len(common) < 2
        ):
            return None
        labels = tuple(str(locants[i]) for i in heads)
        keys = tuple(_locant_key(label) for label in labels)
        if any(k is None for k in keys) or keys[0] == keys[1]:
            return None
        prefix = named_prefix or _bridge_prefix([mol.GetAtomWithIdx(i) for i in path])
        if not prefix:
            return None
        if directed:
            labels_by_end = {end: str(locants[reverse[head]]) for end, head in boundary}
            heads_sorted = (labels_by_end[path[0]], labels_by_end[path[-1]])
        else:
            heads_sorted = tuple(sorted(labels, key=_locant_key))
        high = 0 if keys[0] > keys[1] else 1
        start = boundary[high][0]
        sequence = tuple(path if path[0] == start else reversed(path))
        if sequence[0] != start:
            return None
        groups[prefix].append((heads_sorted, sequence))
        for a, b in zip(path, path[1:]):
            bond = working.GetBondBetweenAtoms(a, b)
            if bond.GetBondTypeAsDouble() != 1.0:
                bridge_orders[frozenset((a, b))] = bond.GetBondType()
    # Compare every admissible post-bridge pi assignment, even when the
    # current Kekule drawing happens to match the template. Otherwise the
    # result can depend on atom order and an unqualified name can describe a
    # different localized-bond graph. Published unqualified states are handled
    # explicitly by the shared source-evidence check.
    core_name = _bridge_aware_aromatic_core_name(
        mol, Chem, template, mapping, bridge, bridge_orders=bridge_orders)
    if not core_name:
        return None
    indicated, unindicated = _without_indicated_hydrogen(core_name)
    _, bare_template = _without_indicated_hydrogen(name)
    if not unindicated.endswith(bare_template):
        return None
    pre = unindicated[: -len(bare_template)]
    words = []
    ordered_groups = sorted(groups.items(), key=lambda item: prefix_alphabetization_key(item[0]))
    for prefix, rows in ordered_groups:
        rows.sort(key=lambda item: tuple(map(_locant_key, item[0])))
        count = len(rows)
        mult = "" if count == 1 else (COMPLEX_MULT if prefix.startswith('(') else MULT).get(count)
        if mult is None:
            return None
        labels = ":".join(",".join(pair) for pair, _ in rows)
        words.append(labels + "-" + mult + prefix)
    locants_text = "-".join(words)
    separator = "-" if pre and not pre.endswith("-") else ""
    rendered = (
        (indicated + "-" if indicated else "")
        + pre
        + separator
        + locants_text
        + bare_template
    )
    # P-25.4.5.2: independent bridges at higher bridgeheads are numbered first.
    bridge_order = sorted(
        (row for rows in groups.values() for row in rows),
        key=lambda row: tuple(sorted(map(_locant_key, row[0]), reverse=True)),
        reverse=True,
    )
    start = int(_next_bridge_locant(locants) or 0)
    if not start:
        return None
    atom_labels = []
    for pair, seq in bridge_order:
        atom_labels.extend((atom, str(start + i)) for i, atom in enumerate(seq))
        start += len(seq)
    parent_labels = tuple(
        (original, str(locants[i])) for i, original in enumerate(mapping)
    )
    all_heads = tuple(
        sorted((_locant_key(label) for pair, _ in bridge_order for label in pair))
    )
    per_citation = tuple(
        (prefix, tuple(tuple(map(_locant_key, pair)) for pair, _ in rows))
        for prefix, rows in ordered_groups
    )
    decision = MethodDecision(
        name=rendered,
        method="bridged-fused",
        principal_centers=frozenset(),
        atoms=heavy,
        rule=RULE + "/P-25.4.3.1/P-25.4.5.2",
        atom_locants=parent_labels + tuple(atom_labels),
        component_roles=(
            ("fused_core", core, parent_labels),
            ("independent_bridges", bridge, tuple(atom_labels)),
        ),
    )
    return all_heads, per_citation, decision


def select_multiple_fused_bridges(mol, Chem):
    from .catacondensed_numbering import peripheral_numbering_frontier
    from .extended_fused_bridges import _extended_paths
    from .molecule_iteration import iter_atoms
    from .naming_context import MoleculeContext
    from .naming_result import SelectionLimitation
    from .parent_rules import _cached_iter_fused_template_matches
    from .parent_selection import ring_seniority_key

    if (
        mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
        or not 7 <= mol.GetNumAtoms() <= 96
        or mol.GetStereoGroups()
        or len(Chem.GetMolFrags(mol)) != 1
        or mol.GetNumBonds() - mol.GetNumAtoms() + 1 < 4
        or any(
            a.GetAtomicNum() not in {6, 7, 8, 16, 34, 52}
            or a.GetFormalCharge()
            or a.GetIsotope()
            or a.GetNumRadicalElectrons()
            or int(a.GetChiralTag())
            or not a.IsInRing()
            for a in iter_atoms(mol)
        )
        or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())
    ):
        return None
    working = Chem.Mol(mol)
    try:
        Chem.Kekulize(working, clearAromaticFlags=True)
    except (ValueError, RuntimeError):
        return None
    paths = tuple((path, boundary, None, False) for path, boundary in _bridge_paths(working))
    paths += tuple((path, boundary, prefix, True)
                   for boundary, specs in _extended_paths(working)
                   for prefix, path in specs)
    if len(paths) < 2 or len(paths) > 80:
        return None
    # A graph admitted as a complete fused system must reach the fused-parent
    # comparison before a decomposition into several bridges is selected.
    # The perimeter frontier may itself defer; it is not a proof that every
    # non-admitted graph requires a bridged construction.
    fused = peripheral_numbering_frontier(
        mol, allow_three_face_n_junction=True, allow_multi_face_n_junction=True,
        allow_larger_system=True, allow_shared_carbon_junctions=True,
        allow_peri_fused=True)
    if fused.admitted:
        return None
    # The exhaustive comparison below must defer if its search limit is
    # reached. Prove that outcome before constructing any candidate cores.
    if sum(comb(len(paths), count) for count in range(2, min(5, len(paths) + 1))) > 5000:
        return None
    templates = _bridge_templates(Chem)

    def matches(graph):
        ctx = MoleculeContext(graph, Chem)
        return tuple(
            _cached_iter_fused_template_matches(
                ctx,
                ctx.candidate_templates(templates),
                Chem,
                topology_query=True,
                require_one_ring_system=True,
                reject_extra_core_bonds=True,
            )
        )

    if any(len(mapping) == mol.GetNumAtoms() for _, mapping in matches(mol)):
        return None
    rows = []
    checked = 0
    for count in range(2, min(5, len(paths) + 1)):
        for chosen in combinations(paths, count):
            checked += 1
            if checked > 5000:
                return None
            sets = [set(p) for p, _, _, _ in chosen]
            if sum(map(len, sets)) != len(set().union(*sets)):
                continue
            union = set().union(*sets)
            if any(head in union for _, boundary, _, _ in chosen for _, head in boundary):
                continue
            original = tuple(i for i in range(mol.GetNumAtoms()) if i not in union)
            rw = Chem.RWMol(working)
            for i in sorted(union, reverse=True):
                rw.RemoveAtom(i)
            core = rw.GetMol()
            try:
                Chem.SanitizeMol(core)
            except (ValueError, RuntimeError):
                continue
            if len(Chem.GetMolFrags(core)) != 1 or any(
                not a.IsInRing() for a in core.GetAtoms()
            ):
                continue
            retained = tuple((template, mapping) for template, mapping in matches(core)
                             if len(mapping) == len(original))
            generated = () if (retained or not _generated_core_can_support_bridges(
                core, original, tuple(boundary for _path, boundary, _prefix, _directed in chosen))) else tuple(
                (template, tuple(range(core.GetNumAtoms())))
                for template in _constructed_core_templates(core, Chem))
            for template, core_match in retained + generated:
                query = template["mol"]
                mapping = tuple(original[i] for i in core_match)
                result = _multi_decision(mol, Chem, template, mapping, chosen, working)
                if result is None:
                    continue
                h, c, decision = result
                rings = tuple(tuple(r) for r in query.GetRingInfo().AtomRings())
                elements = tuple(
                    sorted(Counter(a.GetAtomicNum() for a in query.GetAtoms()).items())
                )
                key = (
                    -len(rings),
                    -query.GetNumAtoms(),
                    sum(a.GetAtomicNum() != 6 for a in query.GetAtoms()),
                    ring_seniority_key(elements, len(rings)),
                    tuple(-s for s in sorted(map(len, rings), reverse=True)),
                )
                rows.append((key, template["name"], h, c, decision))
    if not rows:
        return None
    best = min(r[0] for r in rows)
    rows = [r for r in rows if r[0] == best]
    if len({r[1] for r in rows}) != 1:
        return None
    best_heads = min(r[2] for r in rows)
    rows = [r for r in rows if r[2] == best_heads]
    best_citation = min(r[3] for r in rows)
    rows = [r for r in rows if r[3] == best_citation]
    if len({r[4].name for r in rows}) != 1:
        return None
    ranks = tuple(Chem.CanonicalRankAtoms(mol, breakTies=True))
    chosen = min(
        (r[4] for r in rows),
        key=lambda x: tuple(sorted((loc, ranks[idx]) for idx, loc in x.atom_locants)),
    )
    return replace(
        chosen,
        preference_issues=(
            SelectionLimitation(
                "ring-construction",
                "P-25.4.3.4/P-25.4.3.4.2",
                "Independent divalent bridges compared on discovered fused cores; other bridge forms remain unresolved.",
            ),
        ),
    )
