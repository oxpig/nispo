"""Handoff for a source-mapped benzopyran acid with subordinate annular oxo."""


def requires_full(mol, Chem):
    """Use the full writer only when its actual acid/oxo proposal is proved."""
    if (len(mol.GetRingInfo().AtomRings()) < 2
            or sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms()) < 3
            or mol.GetStereoGroups() or any(
                a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                or int(a.GetChiralTag()) or a.HasQuery() or a.GetAtomMapNum()
                for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return False
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status != 'selected' or frontier.unknown_pairs
            or frontier.cycle or len(frontier.groups) != 1
            or frontier.groups[0].kind != 'acid'):
        return False
    acid = frontier.groups[0]
    ketones = tuple(g for g in inventory.groups if g.kind == 'ketone')
    if not ketones or not mol.GetAtomWithIdx(acid.owner).IsInRing():
        return False
    from .bounded_mixed_acid_ester_fast_guard import _actual_principal_decision
    actual = _actual_principal_decision(mol, Chem)
    outcome = None if actual is None else actual.comparison
    proposal = None if outcome is None else getattr(outcome, 'general_proposal', None)
    state = None if proposal is None else proposal.ring_state
    construction = None if state is None else state.construction
    if (proposal is None or state is None or construction is None
            or actual.prepared is not None or actual.output_name != proposal.name
            or proposal.principal_kind != 'acid' or proposal.principal_centers != {acid.center}
            or state.parent not in {'1-benzopyran', '2-benzopyran'}
            or not state.hydrogen_budget_proved or not state.annotation_coverage):
        return False
    core = frozenset(state.parent_atoms)
    required = frozenset(atom.GetIdx() for atom in mol.GetAtoms())
    return bool(len(core) == 10 and acid.owner in core
                and any(group.center in core for group in ketones)
                and set(acid.atoms) <= construction.embedded_atoms
                and construction.atoms == core
                and not construction.atoms & construction.embedded_atoms
                and construction.atoms | construction.embedded_atoms == required)
