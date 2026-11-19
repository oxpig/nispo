"""Deliver source-owned ring acids when the shared parent pool is incomplete."""

from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def mapped_bridged_acid_decision(mol, Chem, acid, scope):
    """Carry an exocyclic acid and native stereo into an existing bridge map."""
    expected_charge = {acid.ligand: -1} if acid.kind == 'carboxylate' else {}
    if (acid.owner < 0 or mol.GetRingInfo().NumRings() < 4 or mol.GetStereoGroups()
            or acid.kind not in {'acid', 'carboxylate'}
            or not mol.GetAtomWithIdx(acid.owner).IsInRing()
            or any(b.HasQuery() or b.GetStereo() or b.GetBondDir() == Chem.BondDir.UNKNOWN
                   for b in mol.GetBonds())
            or any(a.GetFormalCharge() != expected_charge.get(a.GetIdx(), 0)
                   or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or a.HasQuery() or a.GetAtomMapNum() for a in mol.GetAtoms())):
        return None
    from .component_mapping import MappedComponent, decorate_component, fully_covered, mapped_general_branch
    from .formatting import attach_suffix
    from .mapped_ring_parents import ring_systems
    from .polycyclic_rules import (
        _kekulized_polycyclic_view, _secondary_bridge_polycyclic_candidate_records,
    )

    core = next((r for r in ring_systems(mol) if acid.owner in r), frozenset())
    # This continuation uses the existing bounded macrocycle search. Small
    # bridged acids already have dedicated mapped ring routes.
    if not 20 <= len(core) <= 64:
        return None
    embedded = set(acid.atoms)
    if embedded & core:
        return None
    branches = []
    for host in core:
        for atom in mol.GetAtomWithIdx(host).GetNeighbors():
            root = atom.GetIdx()
            if root in core or (host, root) == (acid.owner, acid.center):
                continue
            text, owned = mapped_general_branch(mol, Chem, root, host)
            if not text or owned & (core | embedded) or not owned <= scope:
                return None
            embedded.update(owned)
            branches.append((host, text))
    if not fully_covered(mol, core, embedded):
        return None
    # A canonical traversal fixes both atom and bond order before the Kekule
    # view and bounded bridge search. Rebase all positions before rendering
    # the input molecule's branches and stereodescriptors.
    import json

    source = Chem.Mol(mol)
    graph = Chem.MolFromSmiles(Chem.MolToSmiles(source, canonical=True, isomericSmiles=True))
    order = json.loads(source.GetProp('_smilesAtomOutputOrder'))
    if graph is None or sorted(order) != list(range(mol.GetNumAtoms())):
        return None
    view = _kekulized_polycyclic_view(graph, Chem)
    if view is graph and any(graph.GetAtomWithIdx(i).GetIsAromatic()
                            for i, original in enumerate(order) if original in core):
        return None
    candidates = _secondary_bridge_polycyclic_candidate_records(
        view, Chem, frozenset(i for i, original in enumerate(order) if original in core),
        max_records=8, decorate_parent=False, suffix_owner=order.index(acid.owner))
    for parent, sequence, unsaturation, topology in candidates:
        if unsaturation or topology:
            continue
        locants = {order[i]: str(j) for j, i in enumerate(sequence, 1)}
        suffix = 'carboxylate' if acid.kind == 'carboxylate' else 'carboxylic acid'
        construction = MappedComponent(
            attach_suffix(parent, suffix, (locants[acid.owner],)),
            tuple(locants.items()), core, tuple((locants[i], text) for i, text in branches),
            frozenset(embedded), systematic_parent=True)
        name = decorate_component(mol, Chem, construction)
        if name:
            rule = 'P-23.2.6/P-65.1.2.2.2'
            return MethodDecision(
                name, 'mapped-ring-acid', frozenset({acid.center}), scope, rule,
                construction.atom_locants, preference_issues=(SelectionLimitation(
                    'ring-acid-parent', rule,
                    'The numbered bridge and acid suffix are mapped. Complete parent comparison remains open.'),))
    return None


def _has_ring_carboxy_motif(mol):
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.IsInRing():
            continue
        neighbors = tuple(atom.GetNeighbors())
        if (any(n.GetAtomicNum() == 6 and n.IsInRing()
                and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1
                for n in neighbors)
                and any(n.GetAtomicNum() == 8 and n.GetDegree() == 1
                        and n.GetTotalNumHs(includeNeighbors=True) == 1
                        and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1
                        for n in neighbors)
                and any(n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(
                    atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2 for n in neighbors)):
            return True
    return False


def mapped_ring_acid_decision(mol, Chem):
    """Return a locally proved ring-acid name with global preference open."""
    if (mol is None or not _has_ring_carboxy_motif(mol)
            or mol.GetNumHeavyAtoms() > 80 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetIsotope() or a.GetNumRadicalElectrons() or a.HasQuery()
                   or a.GetAtomMapNum() for a in mol.GetAtoms())
            or any(b.HasQuery() or int(b.GetStereo()) or int(b.GetBondDir())
                   for b in mol.GetBonds())):
        return None
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status not in {'selected', 'tie'}
            or frontier.unknown_pairs or frontier.cycle or not frontier.groups
            or any(group.kind != 'acid' for group in frontier.groups)
            or tuple(frontier.groups) != tuple(inventory.principal)):
        return None
    acids = tuple(frontier.groups)
    if (len({acid.center for acid in acids}) != len(acids)
            or any(acid.owner < 0 or not mol.GetAtomWithIdx(acid.owner).IsInRing()
                   or mol.GetAtomWithIdx(acid.owner).GetAtomicNum() != 6 for acid in acids)):
        return None
    from .mapped_ring_parents import records
    rows = records(mol, Chem, principal=tuple(frontier.groups))
    scope = frozenset(atom.GetIdx() for atom in mol.GetAtoms())
    general_branches = False
    if not rows and (any(group.kind == 'sulfamide' for group in inventory.groups)
                     or len(acids) == 1 and any(
                         a.GetAtomicNum() == 8 and not a.IsInRing() and a.GetDegree() == 2
                         and all(n.IsInRing() for n in a.GetNeighbors()) for a in mol.GetAtoms())):
        rows = records(mol, Chem, scope=scope, principal=acids, general_branches=True)
        general_branches = True
    if not rows and len(acids) == 1:
        return mapped_bridged_acid_decision(mol, Chem, acids[0], scope)
    if len(rows) != 1:
        return None
    row = rows[0]
    if (row.principal_centers != {acid.center for acid in acids} or not row.name
            or not row.hydrogen_budget_proved or not row.annotation_coverage
            or row.method not in {'retained-fused', 'Hantzsch-Widman', 'monocyclic',
                                  'component-fused', 'benzo-heterocycle',
                                  'component-fused-benzofuran-pyridine',
                                  'component-fused-three-face-azine', 'component-fused-general',
                                  'von-Baeyer', 'spiro', 'component-spiro',
                                  'skeletal-replacement-von-Baeyer', 'skeletal-replacement-spiro'}):
        return None
    component = row.construction
    if (component is None or component.atoms & component.embedded_atoms
            or component.atoms | component.embedded_atoms != scope
            or any(acid.owner not in component.atoms
                   or not set(acid.atoms) - {acid.owner} <= component.embedded_atoms
                   for acid in acids)):
        return None
    from .component_mapping import (decorate_component, mapped_branch, mapped_general_branch,
                                    _general_branch_stereo_is_cited, _mapped_stereochemistry_plan)
    from .stereochemistry import assigned_atom_stereochemistry
    tagged = {a.GetIdx() for a in mol.GetAtoms() if int(a.GetChiralTag())}
    assigned = dict(assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True))
    if set(assigned) != tagged or any(value not in {'R', 'S'} for value in assigned.values()):
        return None
    plan = _mapped_stereochemistry_plan(mol, Chem, component, dict(component.atom_locants))
    if plan is None or plan[2] or {item[0] for item in plan[1]} != tagged & component.atoms:
        return None
    if decorate_component(mol, Chem, component) != row.name:
        return None
    from .branch_rules import collect_subtree
    claimed = set().union(*(set(acid.atoms) for acid in acids)) - component.atoms
    embedded_stereo = set()
    branch = mapped_general_branch if general_branches else mapped_branch
    for root, host, owned, text in component.prefix_branch_ownership:
        owned = frozenset(owned)
        if (host not in component.atoms or root not in owned or claimed & owned
                or frozenset(collect_subtree(mol, root, host)) != owned
                or branch(mol, Chem, root, host) != (text, owned)):
            return None
        cuts = {(idx, neighbor.GetIdx()) for idx in owned
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in owned}
        if cuts != {(root, host)}:
            return None
        if tagged & owned:
            from .retained_acid_rules import mapped_ring_stereo_prefix_proved
            from .carboxamide_prefixes import proved_carbamoyl_construction
            if (not mapped_ring_stereo_prefix_proved(mol, Chem, root, host, owned, text)
                    and not proved_carbamoyl_construction(mol, Chem, root, host, text, owned)
                    and not (general_branches and _general_branch_stereo_is_cited(mol, Chem, owned, text))):
                return None
            embedded_stereo.update(tagged & owned)
        claimed.update(owned)
    if claimed != set(component.embedded_atoms) or tagged - component.atoms != embedded_stereo:
        return None
    issues = [*row.preference_issues, SelectionLimitation(
        'ring-acid-parent', 'P-41/P-44/P-45/P-51/P-65',
        'The acid suffixes, numbered ring and source branches are mapped; '
        'complete competing parent and naming-method selection remains unresolved.')]
    try:
        potential = tuple(Chem.FindPotentialStereo(mol))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if any(item.specified != Chem.StereoSpecified.Specified for item in potential):
        issues.append(SelectionLimitation('stereochemistry', 'P-91',
                                         'Potential stereochemistry is unassigned.'))
    return MethodDecision(row.name, 'mapped-ring-acid', frozenset(acid.center for acid in acids), scope,
                          'P-41/P-44/P-65', row.atom_locants,
                          preference_issues=tuple(issues))
