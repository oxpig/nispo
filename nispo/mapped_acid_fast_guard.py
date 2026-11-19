"""Route any source-owned principal-acid proposal to the full writer."""


def requires_full(mol, Chem):
    """Join the full selector's actual acid proposal to the source graph.

    This is a structural handoff, not a text-based choice between names.
    Unsupported characteristic-group frontiers or incomplete atom/stereo
    ownership stay on the existing fast path.
    """
    if mol is None or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1:
        return False
    if not any(atom.GetAtomicNum() == 6 and not atom.IsInRing()
               and any(n.GetAtomicNum() == 8 and n.GetDegree() == 1
                       and n.GetTotalNumHs(includeNeighbors=True) == 1
                       and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1
                       for n in atom.GetNeighbors())
               and any(n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(
                       atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
                       for n in atom.GetNeighbors()) for atom in mol.GetAtoms()):
        return False
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status != 'selected' or frontier.unknown_pairs
            or frontier.cycle or not frontier.groups
            or any(group.kind != 'acid' for group in frontier.groups)
            or tuple(frontier.groups) != tuple(inventory.principal)):
        return False
    expected = frozenset(group.center for group in frontier.groups)
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision
    from .cross_route_selection import CrossRouteOutcome, CrossRouteSelection, ParentProposal

    actual = _actual_principal_decision(mol, Chem)
    if actual is None:
        return False
    if actual.prepared is not None:
        # The existing monocycle producer returns a prepared candidate rather
        # than a mapped ParentProposal. Its source-proved acid selection is
        # joined to the same ring owner and the functional-suffix operation.
        candidate = actual.prepared.candidate
        if (len(frontier.groups) != 1 or actual.requirement.kind != 'carboxylic-acid'
                or tuple(actual.requirement.atoms) != (frontier.groups[0].owner,)
                or candidate.family != 'single-ring-parent'
                or frontier.groups[0].owner not in candidate.parent_atoms
                or not any(suffix.kind == 'ring_functional_suffix' for suffix in candidate.suffixes)):
            return False
        from .rule_engine import candidate_safety_reasons
        return not candidate_safety_reasons(candidate)
    if not actual.output_name:
        return False
    comparison = actual.comparison
    if isinstance(comparison, CrossRouteOutcome):
        proposal = comparison.selected
    elif isinstance(comparison, CrossRouteSelection):
        proposal = comparison.selected
    else:
        return False
    if (not isinstance(proposal, ParentProposal) or proposal.principal_kind != 'acid'
            or proposal.principal_centers != expected or not proposal.annotation_coverage
            or proposal.name != actual.output_name):
        return False
    component = proposal.construction
    scope = frozenset(atom.GetIdx() for atom in mol.GetAtoms())
    if (component is None or component.atoms & component.embedded_atoms
            or component.atoms | component.embedded_atoms != scope):
        return False
    state = proposal.ring_state
    return bool(state is None or (state.hydrogen_budget_proved and state.annotation_coverage))
