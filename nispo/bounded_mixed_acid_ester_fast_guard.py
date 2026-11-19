"""Handoff to an actually selected and completely mapped acid/ester writer."""


def _actual_principal_decision(mol, Chem):
    from . import api
    from .naming_context import MoleculeContext
    from .rule_engine import _PrincipalSelectionBackend

    engine = api._engine()
    backend = _PrincipalSelectionBackend(mol, Chem, MoleculeContext(mol, Chem))
    return engine._principal_selection.select(backend)


def _pending_proposal_from_decision(decision):
    from .cross_route_selection import CrossRouteOutcome

    if decision is None or not isinstance(decision.comparison, CrossRouteOutcome):
        return None
    outcome = decision.comparison
    proposal = outcome.general_proposal if outcome.status == 'pending' else None
    # This equality joins the selected typed proposal to the exact output of
    # the principal pipeline; it does not classify chemistry from spelling.
    if (proposal is None or decision.prepared is not None or not decision.output_name
            or decision.output_name != proposal.name or not proposal.annotation_coverage):
        return None
    return decision, proposal


def _actual_pending_proposal(mol, Chem):
    """Inspect the full principal pipeline's own pending typed delivery."""
    return _pending_proposal_from_decision(_actual_principal_decision(mol, Chem))


def _one_cut_branch(mol, Chem, row):
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch

    root, host, atoms, text = row
    if (root not in atoms or host in atoms or not text
            or frozenset(collect_subtree(mol, root, host)) != frozenset(atoms)):
        return False
    cuts = {(i, neighbor.GetIdx()) for i in atoms
            for neighbor in mol.GetAtomWithIdx(i).GetNeighbors()
            if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in atoms}
    if cuts != {(root, host)}:
        return False
    return mapped_branch(mol, Chem, root, host) == (text, atoms)


def bounded_mixed_acid_ester_requires_full(mol, Chem):
    """Handoff a single mapped ring-acid parent with one subordinate ester."""
    if (not mol.GetRingInfo().NumRings() or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.GetAtomicNum() == 1 for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return False
    # A terminal acid oxygen and a separate O-alkyl ester must both be present
    # before running the complete inventory or the full selection pipeline.
    acid_like = ester_like = False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.IsInRing():
            continue
        if not any(n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(
                atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
                   for n in atom.GetNeighbors()):
            continue
        for n in atom.GetNeighbors():
            if (n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(
                    atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1):
                acid_like |= n.GetDegree() == 1
                ester_like |= n.GetDegree() == 2
    if not acid_like or not ester_like:
        return False
    from .extended_inventory import selection_inventory
    from .component_mapping import fully_covered

    inventory = selection_inventory(mol)
    acids = [g for g in inventory.groups if g.kind == 'acid']
    esters = [g for g in inventory.groups if g.kind == 'ester']
    if (not inventory.supported or len(inventory.groups) != 2 or len(acids) != 1
            or len(esters) != 1 or tuple(inventory.principal) != tuple(acids)
            or inventory.source_frontier.status != 'selected'
            or not mol.GetAtomWithIdx(acids[0].owner).IsInRing()):
        return False
    acid, ester = acids[0], esters[0]
    selected = _actual_pending_proposal(mol, Chem)
    if selected is None:
        return False
    decision, proposal = selected
    construction, ring = proposal.construction, proposal.ring_state
    if (proposal.principal_kind != 'acid' or proposal.principal_centers != {acid.center}
            or decision.requirement.atoms != (acid.center,) or construction is None
            or ring is None or ring.construction != construction
            or not ring.hydrogen_budget_proved or not ring.annotation_coverage
            or ring.principal_centers != {acid.center} or acid.owner not in construction.atoms
            or acid.center not in dict(proposal.atom_locants)
            or not set(acid.atoms) <= construction.embedded_atoms
            or construction.atoms & construction.embedded_atoms
            or not fully_covered(mol, construction.atoms, construction.embedded_atoms)
            or len(construction.prefix_branch_ownership) != 1):
        return False
    branch = construction.prefix_branch_ownership[0]
    return (branch[1] in construction.atoms
            and set(ester.atoms) <= branch[2] and set(acid.atoms).isdisjoint(branch[2])
            and set(construction.embedded_atoms) == set(acid.atoms) | set(branch[2])
            and _one_cut_branch(mol, Chem, branch))


def bounded_annotated_bicyclic_ester_requires_full(mol, Chem):
    """Handoff a typed ester with a separately witnessed bicyclic ring arm."""
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if (len(rings) != 2 or not rings[0] & rings[1] or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or a.GetAtomicNum() == 1 for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return False
    # The input must have a nonring C(=O)(O) center attached directly to a
    # ring carbon. This cheap test precedes graph inventory/selection.
    if not any(a.GetAtomicNum() == 6 and not a.IsInRing()
               and any(n.GetAtomicNum() == 6 and n.IsInRing() for n in a.GetNeighbors())
               and sum(n.GetAtomicNum() == 8 for n in a.GetNeighbors()) == 2
               for a in mol.GetAtoms()):
        return False
    from .extended_inventory import selection_inventory
    from .component_mapping import fully_covered, mapped_branch_ring_atoms
    from .branch_rules import collect_subtree

    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.groups) != 1
            or len(inventory.principal) != 1 or inventory.principal[0].kind != 'ester'):
        return False
    ester = inventory.principal[0]
    if not mol.GetAtomWithIdx(ester.owner).IsInRing():
        return False
    actual = _actual_principal_decision(mol, Chem)
    selected = _pending_proposal_from_decision(actual)
    if selected is None:
        # A newer full producer can prove the actual bicyclic ring-carboxylate
        # directly instead of returning the earlier pending formate proposal.
        # Rejoin the actual decision to that producer and its disjoint roles.
        from .bicyclic_carboxylate_methods import bicyclic_carboxylate_decision

        proved = bicyclic_carboxylate_decision(mol, Chem)
        if (actual is None or proved is None or actual.prepared is not None
                or actual.comparison != proved or actual.output_name != proved.name
                or proved.principal_centers != {ester.center}
                or actual.requirement.atoms != (ester.center,)):
            return False
        roles = {name: frozenset(atoms) for name, atoms, _ in proved.component_roles}
        heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        if (set(roles) != {'parent-hydride', 'ring-substituents',
                           'carboxylate', 'O-organyl'}
                or len(proved.component_roles) != len(roles)
                or proved.atoms != heavy or sum(map(len, roles.values())) != len(heavy)
                or frozenset().union(*roles.values()) != heavy):
            return False
        carbonyl_oxygen = set(ester.atoms) - {ester.center, ester.ligand}
        return (ester.owner in roles['parent-hydride']
                and roles['carboxylate'] == {ester.center} | carbonyl_oxygen
                and ester.ligand in roles['O-organyl'])
    # The old formate continuation was independently checked for assigned
    # source stereo only. An unassigned bicyclic graph is admitted solely by
    # the direct ring-carboxylate producer and its own uncertainty evidence.
    if not any(int(a.GetChiralTag()) for a in mol.GetAtoms()):
        return False
    decision, proposal = selected
    component = proposal.construction
    if (proposal.principal_kind != 'ester' or proposal.principal_centers != {ester.center}
            or decision.requirement.atoms != (ester.center,) or component is None
            or not set(ester.atoms) <= component.atoms
            or not set(ester.atoms) <= dict(proposal.atom_locants).keys()
            or component.atoms & component.embedded_atoms
            or not fully_covered(mol, component.atoms, component.embedded_atoms)
            or len(component.prefix_branch_ownership) != 1):
        return False
    branch = component.prefix_branch_ownership[0]
    root, host, branch_atoms, text = branch
    if root != ester.owner or host != ester.center or not _one_cut_branch(mol, Chem, branch):
        return False
    ring_atoms = {i for i in branch_atoms if mol.GetAtomWithIdx(i).IsInRing()}
    if (not ring_atoms or not ring_atoms <= mapped_branch_ring_atoms(
            mol, Chem, root, host, text, branch_atoms)):
        return False
    bridge = mol.GetAtomWithIdx(ester.ligand)
    roots = [n.GetIdx() for n in bridge.GetNeighbors() if n.GetIdx() != ester.center]
    if len(roots) != 1:
        return False
    organyl = frozenset(collect_subtree(mol, roots[0], ester.ligand))
    return (not branch_atoms & organyl and organyl | branch_atoms == component.embedded_atoms
            and all(not int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in organyl))
