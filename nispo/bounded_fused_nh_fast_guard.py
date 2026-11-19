"""Conservative fast-path boundary for substituted aromatic N/H ring states."""


def bounded_fused_nh_requires_full(mol, Chem):
    """Require the full writer for this graph-proved aromatic ring risk.

    A neutral aromatic NH is part of an edge-fused ring, and a neutral
    aromatic ring nitrogen is substituted by a carbon outside its own ring
    system. Independent atom-owned ring numbering and N attachment are both
    necessary to name this composition without changing the input graph.
    Handoff additionally requires a complete typed ring construction on the
    original graph; otherwise existing fast-path behavior is preserved.
    This is a shortcut-admission boundary, not a complete preferred-name
    claim; the full writer retains its own unresolved comparison evidence.
    Annotated or charged structures remain outside this bounded handoff.
    """
    if (mol is None or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or any(atom.GetFormalCharge() or atom.GetIsotope()
                   or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag())
                   for atom in mol.GetAtoms())
            or any(int(bond.GetStereo()) or int(bond.GetBondDir())
                   for bond in mol.GetBonds())):
        return False
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if len(rings) < 2:
        return False
    fused = set()
    for index, first in enumerate(rings):
        for second in rings[index + 1:]:
            shared = first & second
            if len(shared) == 2 and mol.GetBondBetweenAtoms(*tuple(shared)) is not None:
                fused.update(first)
                fused.update(second)
    if not any(atom.GetIdx() in fused and atom.GetAtomicNum() == 7
               and atom.GetIsAromatic() and atom.GetTotalNumHs(includeNeighbors=True) == 1
               for atom in mol.GetAtoms()):
        return False
    for nitrogen in mol.GetAtoms():
        if (nitrogen.GetAtomicNum() != 7 or not nitrogen.IsInRing()
                or not nitrogen.GetIsAromatic() or nitrogen.GetDegree() != 3
                or nitrogen.GetTotalValence() != 3
                or nitrogen.GetTotalNumHs(includeNeighbors=True)):
            continue
        system = set().union(*(ring for ring in rings if nitrogen.GetIdx() in ring))
        changed = True
        while changed:
            changed = False
            for ring in rings:
                if ring & system and not ring <= system:
                    system.update(ring)
                    changed = True
        internal = [neighbor for neighbor in nitrogen.GetNeighbors()
                    if neighbor.GetIdx() in system]
        external = [neighbor for neighbor in nitrogen.GetNeighbors()
                    if neighbor.GetIdx() not in system and neighbor.GetAtomicNum() > 1]
        if (len(internal) == 2 and len(external) == 1
                and all(mol.GetBondBetweenAtoms(nitrogen.GetIdx(), atom.GetIdx()).GetIsAromatic()
                        for atom in internal)
                and external[0].GetAtomicNum() == 6
                and mol.GetBondBetweenAtoms(nitrogen.GetIdx(),
                                            external[0].GetIdx()).GetBondTypeAsDouble() == 1):
            return _complete_mapped_ring_construction(mol, Chem)
    return False


def _complete_mapped_ring_construction(mol, Chem):
    """Bind ownership to the actual full pipeline's selected ring writer."""
    from .component_mapping import fully_covered
    from .bounded_mixed_acid_ester_fast_guard import _actual_pending_proposal, _one_cut_branch

    selected = _actual_pending_proposal(mol, Chem)
    if selected is not None:
        decision, proposal = selected
        record, component = proposal.ring_state, proposal.construction
        from .extended_inventory import selection_inventory
        inventory = selection_inventory(mol)
        groups = [group for group in inventory.principal
                  if group.center in proposal.principal_centers]
        if (inventory.supported and proposal.principal_kind == 'lactam'
                and len(groups) == len(inventory.principal) == 1
                and groups[0].kind == 'lactam'
                and proposal.principal_centers == {groups[0].center}
                and decision.requirement.atoms == (groups[0].center,)
                and record is not None and component is not None
                and record.construction == component and record.hydrogen_budget_proved
                and record.annotation_coverage and record.principal_centers == proposal.principal_centers
                and not component.atoms & component.embedded_atoms
                and fully_covered(mol, component.atoms, component.embedded_atoms)):
            branches = component.prefix_branch_ownership
            owned = set()
            valid = True
            for branch in branches:
                root, host, atoms, text = branch
                if (host not in component.atoms or owned & set(atoms)
                        or not _one_cut_branch(mol, Chem, branch)):
                    valid = False
                    break
                owned.update(atoms)
            suffix = set(groups[0].atoms) - set(component.atoms)
            if valid and owned | suffix == set(component.embedded_atoms) and not owned & suffix:
                return True

    # Function-free ring compounds use the separate P-29 ring-hydride
    # producer; join its source-owned partition to the actual full decision.
    from .ring_naming_methods import ring_naming_decision
    decision = ring_naming_decision(mol, Chem)
    if decision is None or not decision.name or not decision.component_roles:
        return False
    from . import api
    from .naming_context import MoleculeContext
    from .rule_engine import _PrincipalSelectionBackend

    backend = _PrincipalSelectionBackend(mol, Chem, MoleculeContext(mol, Chem))
    actual = api._engine()._principal_selection.select(backend)
    if (actual is None or not actual.direct_route.startswith('ring_method:')
            or actual.prepared is not None or actual.output_name != decision.name
            or tuple(actual.atom_locants) != tuple(decision.atom_locants)):
        return False
    heavy = frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    parts = tuple(frozenset(atoms) for _, atoms, _ in decision.component_roles)
    return (decision.atoms == heavy and sum(map(len, parts)) == len(heavy)
            and frozenset().union(*parts) == heavy)
