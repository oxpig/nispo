"""Source-owned principal suffixes when the preferred method remains open."""

from .component_mapping import fully_covered
from .extended_inventory import selection_inventory
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation
from .parent_selection import parent_frontier


_ACYL = {'acid', 'amide', 'nitrile', 'carboxylate'}
_OXYGEN = {'alcohol', 'aldehyde', 'ketone', 'lactam', 'lactone', 'pseudoketone'}
_ACID_ANIONS = {'carboxylate', 'carbamate-anion'}
_SUPPORTED = _ACYL | _OXYGEN | _ACID_ANIONS | {
    'amine', 'sulfonamide', 'sulfonic-acid', 'sulfinic-acid', 'phosphonic-acid', 'thiol'}
_RULE = 'P-41/P-44.1.1/P-45'


def simple_acid_anion_scope(mol, groups, scope=None):
    """Prove charge ownership for terminal acid anions on one parent."""
    if not 1 <= len(groups) <= 2 or any(g.kind not in _ACID_ANIONS for g in groups):
        return False
    if scope is not None and any(not set(g.atoms) <= scope for g in groups):
        return False
    if len(groups) == 2:
        from .mapped_ring_parents import ring_systems

        owners = {g.owner for g in groups}
        if (len(owners) != 2 or not any(owners <= ring for ring in ring_systems(mol))
                or set(groups[0].atoms).intersection(groups[1].atoms)):
            return False
    charges = {(a.GetIdx(), a.GetFormalCharge()) for a in mol.GetAtoms() if a.GetFormalCharge()}
    return (charges == {(g.ligand, -1) for g in groups}
            and all(mol.GetAtomWithIdx(g.ligand).GetDegree() == 1 for g in groups))


def selected_acid_anions(mol, inventory):
    """Gather the two external sites when they share a ring carboxylate suffix."""
    groups = inventory.principal
    if not groups or groups[0].kind not in _ACID_ANIONS:
        return ()
    groups = tuple(g for g in inventory.groups if g.kind in _ACID_ANIONS)
    return groups if simple_acid_anion_scope(mol, groups) else ()


def senior_sites(mol):
    """The original graph's established senior groups, before parent choice."""
    inventory = selection_inventory(mol)
    if (not inventory.supported or not inventory.principal
            or inventory.source_frontier.status == 'barrier'
            or any(atom.GetIsotope() for atom in mol.GetAtoms())):
        return ()
    return inventory.principal


def _carbon_parent_plans(mol, groups, inventory):
    """Enumerate the carbon chains carrying terminal or external sites."""
    from .multifunctional_parent import _paths

    kind = groups[0].kind
    cyano_prefixes = {g.center for g in inventory.groups
                     if g.kind == 'nitrile' and g not in groups}
    carbon = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6
              and not atom.IsInRing() and not atom.GetFormalCharge()} - cyano_prefixes
    centers = {group.center for group in groups}
    if kind in {'sulfonamide', 'sulfonic-acid', 'sulfinic-acid', 'thiol'}:
        for path in _paths(mol, carbon):
            covered = tuple(group for group in groups if group.owner in path)
            if covered:
                for oriented in dict.fromkeys((path, path[::-1])):
                    yield oriented, covered, True
        return
    for external in (False, True) if len(groups) > 2 else (False,):
        for path in _paths(mol, carbon - centers if external else carbon):
            if external:
                covered = tuple(group for group in groups if group.owner in path)
                if len(covered) < 3:
                    continue
            else:
                covered = tuple(group for group in groups if group.center in {path[0], path[-1]})
                if (not covered or centers.intersection(path) != {g.center for g in covered}
                        or kind == 'carboxylate' and any(
                            g.owner >= 0 and mol.GetAtomWithIdx(g.owner).IsInRing() for g in covered)):
                    continue
            if kind == 'nitrile' and external and any(
                    mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
                    for a, b in zip(path, path[1:])):
                continue
            for oriented in dict.fromkeys((path, path[::-1])):
                if not external and len(covered) == 1 and oriented[0] != covered[0].center:
                    continue
                yield oriented, covered, external


def _carbon_proposals(mol, Chem, groups, inventory, minimum, *, branch_resolver=None):
    from .complete_functional_parent import _render_functional_candidate

    plans = [plan for plan in _carbon_parent_plans(mol, groups, inventory)
             if len(plan[1]) > minimum]
    if not plans:
        return
    count = max(len(covered) for _path, covered, _external in plans)
    # Each mode can have its own P-66/P-65 suffix grammar. Retain every
    # maximum length and unsaturation witness within that mode for ranking.
    for external in {plan[2] for plan in plans}:
        options = [(path, covered) for path, covered, mode in plans
                   if mode == external and len(covered) == count]
        if not options:
            continue
        def key(option):
            path, _covered = option
            orders = tuple(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
                           for a, b in zip(path, path[1:]))
            return len(path), sum(order in {2, 3} for order in orders), orders.count(2)

        maximum = max(map(key, options))
        for path, covered in options:
            if key((path, covered)) != maximum:
                continue
            skipped = {i for group in covered for i in group.atoms if i != group.center}
            row = _render_functional_candidate(
                mol, Chem, covered, groups[0].kind, path, external, False,
                skipped, (), mapped_ring_prefixes=True,
                charge_role_proofs=inventory.charge_role_proofs, general_branches=True,
                locanted_acid_fallback=True, branch_resolver=branch_resolver)
            if row is not None:
                yield row, row.parent_selection


def _ring_proposals(mol, Chem, groups, scope, minimum):
    from .mapped_ring_parents import _site, parent_facts, records, ring_systems

    for core in ring_systems(mol):
        if sum(_site(mol, group, core) in core for group in groups) <= minimum:
            continue
        for row in records(mol, Chem, scope=scope, core_filter=core, principal=groups,
                           general_branches=True):
            yield row, parent_facts(mol, row, groups[0].seniority)


def mapped_principal_decision(mol, Chem, *, minimum=0, carbon_branch_resolver=None):
    """Compare available whole-graph candidates using the senior site count."""
    groups = senior_sites(mol)
    if (len(groups) <= minimum or groups[0].kind not in _SUPPORTED
            or mol.GetStereoGroups()
            or any(atom.GetNumRadicalElectrons() or atom.HasQuery() for atom in mol.GetAtoms())):
        return None
    from .neutral_charge_components import validate_neutral_charge_role_proofs

    inventory = selection_inventory(mol)
    kind = groups[0].kind
    if kind in _ACID_ANIONS:
        groups = selected_acid_anions(mol, inventory)
        if (not groups and kind == 'carboxylate' and minimum < 2
                and len(inventory.principal) == 2 and len(Chem.GetMolFrags(mol)) == 1
                and not any(g.kind in {'acid', 'carbonic-acid'} for g in inventory.groups)):
            from .multiplicative_carboxylates import linked_carboxylate_decision

            units = tuple(inventory.principal)
            scope = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
            return linked_carboxylate_decision(mol, Chem, units, scope)
        if (len(groups) <= minimum or len(Chem.GetMolFrags(mol)) != 1
                or any(group.kind in {'acid', 'carbonic-acid'} for group in inventory.groups)):
            return None
        kind = groups[0].kind
    elif not validate_neutral_charge_role_proofs(mol, inventory.charge_role_proofs):
        return None
    scope = frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    rows = list(_ring_proposals(mol, Chem, groups, scope, minimum))
    if kind in _ACYL and not (kind in _ACID_ANIONS and len(groups) > 1):
        rows.extend(_carbon_proposals(mol, Chem, groups, inventory, minimum,
                                     branch_resolver=carbon_branch_resolver))
    elif kind in {'sulfonamide', 'sulfonic-acid', 'sulfinic-acid', 'thiol'}:
        rows.extend(_carbon_proposals(mol, Chem, groups, inventory, minimum))
    elif kind in _OXYGEN:
        from .cross_route_selection import oxygen_proposals

        rows.extend((row, row.parent_selection) for row in oxygen_proposals(
            mol, Chem, inventory, branch_resolver=carbon_branch_resolver))
    elif kind == 'amine':
        from .amine_components import chain_proposals

        rows.extend((row, row.parent_selection) for row in chain_proposals(mol, Chem, inventory))
    if (not rows and not minimum and kind == 'sulfonamide' and len(groups) == 1
            and mol.GetAtomWithIdx(groups[0].ligand).IsInRing()):
        from .spiro_component_rules import sulfonyl_monospiro_ketone

        general = (_ring_ketone_with_sulfonyl_prefix(mol, Chem, inventory, scope)
                   or sulfonyl_monospiro_ketone(mol, Chem, inventory))
        if general is not None:
            return general
    centers = {group.center for group in groups}
    if (kind == 'acid' and len(groups) == 1 and minimum == 0
            and groups[0].owner >= 0 and mol.GetRingInfo().NumRings() >= 3
            and mol.GetAtomWithIdx(groups[0].owner).IsInRing()
            and not any(facts is not None and facts.kind == 'ring' for _, facts in rows)):
        from .branch_rules import collect_subtree
        from .spiro_component_rules import _mapped_component_spiro

        group = groups[0]
        acid_side = frozenset(collect_subtree(mol, group.owner, group.center))
        if acid_side.isdisjoint(group.atoms) and acid_side | set(group.atoms) == scope:
            spiro = _mapped_component_spiro(mol, Chem, group.owner, group.center, acid_side,
                                            'carboxylic acid', general_branches=True)
            if spiro is not None:
                name, locants = spiro
                rule = _RULE + '/P-24/P-65.1.2.2.2'
                return MethodDecision(name, 'substitutive', frozenset(centers), scope, rule, locants,
                    preference_issues=(SelectionLimitation('principal-parent-and-method', rule,
                        'Complete preferred comparison for the mapped spiro acid remains open.'),))
    if (kind == 'carboxylate' and len(groups) == 1 and minimum == 0
            and not rows and mol.GetRingInfo().NumRings() >= 4):
        from .mapped_ring_acid_general import mapped_bridged_acid_decision

        bridged = mapped_bridged_acid_decision(mol, Chem, groups[0], scope)
        if bridged is not None:
            return bridged
    if (kind == 'amide' and len(groups) == 1 and mol.GetRingInfo().NumRings() >= 3
            and not any(facts is not None and facts.kind == 'ring' for _row, facts in rows)):
        from .spiro_component_rules import mapped_spiro_carboxamide

        spiro = mapped_spiro_carboxamide(mol, Chem, groups[0], scope)
        if spiro:
            name, locants = spiro
            rule = _RULE + '/P-66.1'
            return MethodDecision(
                name, 'substitutive', frozenset(centers), scope, rule, locants,
                preference_issues=(SelectionLimitation(
                    'principal-parent-and-method', rule,
                    'The spiro amide is mapped. Complete preferred comparison remains open.'),))
    usable = [(row, facts) for row, facts in rows
              if (minimum < len(row.principal_centers) and row.principal_centers <= centers
                  and row.construction is not None
                  and fully_covered(mol, row.construction.atoms, row.construction.embedded_atoms))]
    if not usable:
        return None
    maximum = max(len(row.principal_centers) for row, _facts in usable)
    best = [(row, facts) for row, facts in usable if len(row.principal_centers) == maximum]
    if all(facts is not None for _row, facts in best):
        best = parent_frontier(best, lambda row: row[1])
    row, _facts = min(best, key=lambda candidate: candidate[0].name)
    rule = _RULE + ('/P-72.2.2.2.1' if kind in _ACID_ANIONS else '')
    return MethodDecision(
        row.name, 'substitutive', row.principal_centers, scope, rule, row.atom_locants,
        preference_issues=(SelectionLimitation(
            'principal-parent-and-method', rule,
            'The cited senior sites and parent are mapped. Complete preferred comparison remains open.'),))


def mapped_amide_ring_arm_decision(mol, Chem):
    """Keep an acyclic amide suffix with a fully numbered cyclic N substituent."""
    groups = senior_sites(mol)
    if not groups or groups[0].kind != 'amide' or not any(
            any(neighbor.GetIdx() != group.center and neighbor.IsInRing()
                for neighbor in mol.GetAtomWithIdx(group.ligand).GetNeighbors())
            for group in groups):
        return None
    return mapped_principal_decision(mol, Chem, carbon_branch_resolver=_amide_ring_arm)


def _amide_ring_arm(mol, Chem, root, host):
    atom = mol.GetAtomWithIdx(root)
    if (atom.GetAtomicNum() != 6 or not atom.IsInRing()
            or mol.GetAtomWithIdx(host).GetAtomicNum() != 7
            or mol.GetBondBetweenAtoms(root, host).GetBondTypeAsDouble() != 1):
        return None
    from .branch_rules import collect_subtree
    from .component_mapping import (
        _complete_rooted_ring, _general_branch_stereo_is_cited, mapped_general_branch,
    )

    name, owned = mapped_general_branch(mol, Chem, root, host)
    if name:
        return name, owned
    atoms = frozenset(collect_subtree(mol, root, host))
    if {(i, n.GetIdx()) for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms} != {(root, host)}:
        return '', frozenset()
    name = _complete_rooted_ring(mol, Chem, root, atoms)
    return ((name, atoms) if name and _general_branch_stereo_is_cited(mol, Chem, atoms, name)
            else ('', frozenset()))


def _ring_ketone_with_sulfonyl_prefix(mol, Chem, inventory, scope):
    """Offer a mapped ketone when the sulfonamide N is part of its ring."""
    from .mapped_ring_parents import records

    sulfonamide = inventory.principal[0]
    for ring in mol.GetRingInfo().AtomRings():
        if sulfonamide.ligand not in ring:
            continue
        ketones = tuple(g for g in inventory.groups if g.kind == 'ketone' and g.center in ring)
        if len(ketones) != 1:
            continue
        rows = records(mol, Chem, scope=scope, core_filter=frozenset(ring),
                       principal=ketones, general_branches=True)
        if len(rows) == 1 and rows[0].annotation_coverage and rows[0].hydrogen_budget_proved:
            row = rows[0]
            owned = [atoms for root, host, atoms, _text in row.construction.prefix_branch_ownership
                     if root == sulfonamide.center and host == sulfonamide.ligand]
            if len(owned) != 1 or not set(sulfonamide.atoms) - {sulfonamide.ligand} <= owned[0]:
                continue
            rule = 'P-64.2/P-65.3.2.2'
            return MethodDecision(
                row.name, 'substitutive', row.principal_centers, scope, rule, row.atom_locants,
                preference_issues=(SelectionLimitation(
                    'principal-parent-and-method', rule,
                    'The annular ketone and sulfonyl branch are mapped. Sulfonamide parent comparison remains open.'),))
    return None
