"""Source-owned acid parent with a subordinate mapped ring-carboxamido unit.

This bounded construction keeps both an acid-chain stereocentre and a
stereocentre in the attached ring component. It is a general name; it does
not claim to have exhausted competing whole-name methods or parent classes.
"""
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def has_scope_motifs(mol):
    """Cheap necessary conditions before invoking whole-graph inventory."""
    acid = amide = False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.IsInRing():
            continue
        neighbors = tuple(atom.GetNeighbors())
        if not any(n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(
                atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2 for n in neighbors):
            continue
        acid |= any(n.GetAtomicNum() == 8 and n.GetDegree() == 1
                    and n.GetTotalNumHs(includeNeighbors=True) == 1
                    and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1
                    for n in neighbors)
        amide |= any(n.GetAtomicNum() == 7 and not n.IsInRing()
                     and n.GetTotalNumHs(includeNeighbors=True) == 1
                     and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1
                     for n in neighbors)
        if acid and amide:
            return True
    return False


def mapped_amido_acid_decision(mol, Chem):
    """Construct a unique carbon-chain monoacid with a mapped ring amido branch."""
    if (mol is None or not has_scope_motifs(mol)
            or mol.GetNumHeavyAtoms() > 80 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 17, 35, 53}
                   or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or a.HasQuery() or a.GetAtomMapNum() for a in mol.GetAtoms())
            or any(b.HasQuery() or int(b.GetStereo()) or int(b.GetBondDir())
                   for b in mol.GetBonds())):
        return None
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status != 'selected'
            or frontier.unknown_pairs or frontier.cycle or len(frontier.groups) != 1
            or frontier.groups[0].kind != 'acid'
            or tuple(frontier.groups) != tuple(inventory.principal)
            or any(g.kind not in {'acid', 'amide', 'lactone', 'lactam', 'alcohol'}
                   for g in inventory.groups)):
        return None
    acid = frontier.groups[0]
    amides = tuple(g for g in inventory.groups if g.kind == 'amide')
    if len(amides) != 1 or mol.GetAtomWithIdx(acid.owner).IsInRing():
        return None
    amide = amides[0]
    nitrogen = mol.GetAtomWithIdx(amide.ligand)
    if (not mol.GetAtomWithIdx(amide.owner).IsInRing()
            or nitrogen.IsInRing() or nitrogen.GetTotalNumHs(includeNeighbors=True) != 1):
        return None
    from .parent_rules import _ester_acyl_parent_paths
    paths = _ester_acyl_parent_paths(mol, Chem, acid.center, acid.owner)
    # All structurally longest source paths must be represented. Do not use
    # a renderable path to break a tied parent frontier.
    if len(paths) != 1:
        return None
    path = (acid.center, *paths[0])
    if (len(path) < 3 or any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
                             for a, b in zip(path, path[1:]))
            or not any(mol.GetBondBetweenAtoms(amide.ligand, idx) is not None
                       for idx in path)):
        return None
    from .complete_functional_parent import _render_functional_candidate
    proposal = _render_functional_candidate(
        mol, Chem, (acid,), 'acid', path, False, False,
        set(acid.atoms) - {acid.center}, [], mapped_ring_prefixes=True,
        charge_role_proofs=(),
    )
    if proposal is None or not proposal.name:
        return None
    component = proposal.construction
    scope = frozenset(a.GetIdx() for a in mol.GetAtoms())
    if (component is None or component.atoms & component.embedded_atoms
            or component.atoms | component.embedded_atoms != scope
            or proposal.principal_centers != {acid.center}
            or set(acid.atoms) - {acid.center} - set(component.atoms)):
        return None
    # Rejoin each emitted prefix to its one-cut original-graph branch. Only a
    # ring-amido branch may own a stereogenic atom off the acid chain.
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch, mapped_branch_ring_atoms
    from .mapped_ring_parents import ring_systems
    roles, claimed, owned_amide_rings = [], set(), set()
    for root, host, owned, text in component.prefix_branch_ownership:
        owned = frozenset(owned)
        if (host not in path or root not in owned or claimed & owned
                or frozenset(collect_subtree(mol, root, host)) != owned
                or mapped_branch(mol, Chem, root, host) != (text, owned)):
            return None
        cuts = {(idx, neighbor.GetIdx()) for idx in owned
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in owned}
        if cuts != {(root, host)}:
            return None
        if root == amide.ligand:
            if not set(amide.atoms) <= owned:
                return None
            ring_atoms = mapped_branch_ring_atoms(mol, Chem, root, host, text, owned)
            rings = [core for core in ring_systems(mol) if amide.owner in core]
            if len(rings) != 1 or not rings[0] <= ring_atoms:
                return None
            owned_amide_rings.update(rings[0])
        claimed.update(owned)
        roles.append(('acid-prefix', owned, ()))
    if claimed != set(component.embedded_atoms) or not owned_amide_rings:
        return None
    from .stereochemistry import assigned_atom_stereochemistry
    tagged = {atom.GetIdx() for atom in mol.GetAtoms() if int(atom.GetChiralTag())}
    assigned = dict(assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True))
    if set(assigned) != tagged or not tagged <= set(path) | owned_amide_rings:
        return None
    # Unspecified stereochemistry is not invented by either component writer.
    issues = [SelectionLimitation(
        'acid-parent', 'P-41/P-44/P-45/P-51',
        'The acid suffix, carbon path and ring-amido atom/stereo ownership are mapped; '
        'complete competing parent and naming-method selection remains unresolved.')]
    try:
        potential = tuple(Chem.FindPotentialStereo(mol))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if any(item.type != Chem.StereoType.Atom_Tetrahedral for item in potential):
        return None
    if any(item.specified != Chem.StereoSpecified.Specified for item in potential):
        issues.append(SelectionLimitation(
            'stereochemistry', 'P-91', 'A potentially stereogenic atom is unassigned.'))
    return MethodDecision(
        proposal.name, 'mapped-amido-acid', frozenset({acid.center}), scope,
        'P-41/P-44.3/P-65.1/P-66.1.1.4.3/P-91', proposal.atom_locants,
        component_roles=(('acid-parent', component.atoms, component.atom_locants), *roles),
        preference_issues=tuple(issues),
    )


def requires_full(mol, Chem):
    """Hand off only a fully mapped acid construction to the full writer."""
    if not has_scope_motifs(mol):
        return False
    proof = mapped_amido_acid_decision(mol, Chem)
    if proof is None or not proof.preference_issues:
        return False
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision
    actual = _actual_principal_decision(mol, Chem)
    return bool(actual is not None and actual.output_name == proof.name
                and actual.comparison == proof and actual.prepared is None
                and set(proof.preference_issues) <= set(actual.preference_issues))
