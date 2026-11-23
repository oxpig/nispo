"""Source-mapped general continuations for one ordinary carboxylic ester.

This producer does not resolve the complete parent/method frontier. It can
emit a functional-class construction only after the ester, organyl component,
parent skeleton and all branches cover the original graph exactly.
"""

from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def _one_cut(mol, root, host, atoms):
    from .branch_rules import collect_subtree

    atoms = frozenset(atoms)
    if not atoms or root not in atoms or host in atoms:
        return False
    if frozenset(collect_subtree(mol, root, host)) != atoms:
        return False
    return {(idx, neighbor.GetIdx()) for idx in atoms
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
            if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in atoms} == {(root, host)}


def _source_ester(inventory):
    """Admit one carboxylic ester for a mapped functional-class name."""
    if (not inventory.supported or len(inventory.principal) != 1
            or inventory.principal[0].kind != 'ester'
            or len(inventory.source_group_facts) != len(inventory.groups)):
        return None
    facts = inventory.source_group_facts
    if any(fact is None or fact.class_rank is None for fact in facts):
        return None
    group = inventory.principal[0]
    minima = [candidate for candidate, fact in zip(inventory.groups, facts, strict=True)
              if fact.class_rank == 9]
    # P-65.6.3.3.4 permits separate ester classes to be represented by a
    # functional-class name with other ester components cited as prefixes.
    # Leave broader same-class comparisons open. Here only subordinate
    # carbamate arms can share class 9 with the sole carboxylic ester.
    if (any(item != group and item.kind != 'carbamate' for item in minima)
            or any(fact.class_rank < 9 for fact in facts)):
        return None
    return group


def _ester_organyl(mol, Chem, group):
    from .component_mapping import mapped_branch

    carbonyl = mol.GetAtomWithIdx(group.center)
    bridge = mol.GetAtomWithIdx(group.ligand)
    if (carbonyl.GetAtomicNum() != 6 or carbonyl.IsInRing() or carbonyl.GetDegree() != 3
            or bridge.GetAtomicNum() != 8 or bridge.GetDegree() != 2
            or bridge.GetFormalCharge() or group.owner < 0
            or mol.GetAtomWithIdx(group.owner).GetAtomicNum() != 6):
        return None
    bond = mol.GetBondBetweenAtoms(group.center, group.ligand)
    owner_bond = mol.GetBondBetweenAtoms(group.center, group.owner)
    oxo = [neighbor.GetIdx() for neighbor in carbonyl.GetNeighbors()
           if neighbor.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(group.center, neighbor.GetIdx()).GetBondTypeAsDouble() == 2]
    if (bond is None or bond.GetBondTypeAsDouble() != 1 or owner_bond is None
            or owner_bond.GetBondTypeAsDouble() != 1 or len(oxo) != 1
            or mol.GetAtomWithIdx(oxo[0]).GetDegree() != 1
            or set(group.atoms) != {group.center, group.ligand, oxo[0]}):
        return None
    roots = [neighbor.GetIdx() for neighbor in bridge.GetNeighbors()
             if neighbor.GetIdx() != group.center and neighbor.GetAtomicNum() > 1]
    if len(roots) != 1:
        return None
    root = roots[0]
    if (mol.GetAtomWithIdx(root).GetAtomicNum() != 6
            or mol.GetBondBetweenAtoms(root, group.ligand).GetBondTypeAsDouble() != 1):
        return None
    text, atoms = mapped_branch(mol, Chem, root, group.ligand)
    atoms = frozenset(atoms)
    if not text or any(ch.isspace() for ch in text) or not _one_cut(mol, root, group.ligand, atoms):
        return None
    if any(int(mol.GetAtomWithIdx(idx).GetChiralTag()) for idx in atoms):
        return None  # Stereo on the alcohol-derived component needs a separate witness.
    return text, atoms


def _complete_component(mol, Chem, group, proposal, organyl, organyl_atoms, *, ring):
    from .component_mapping import _mapped_stereochemistry_plan, decorate_component, mapped_branch
    from .stereochemistry import assigned_atom_stereochemistry

    component = proposal.construction
    scope = frozenset(atom.GetIdx() for atom in mol.GetAtoms())
    if (component is None or component.atoms & component.embedded_atoms
            or component.atoms | component.embedded_atoms != scope
            or proposal.principal_centers != {group.center}
            or not organyl_atoms <= component.embedded_atoms):
        return False
    if ring:
        if (group.owner not in component.atoms
                or not set(group.atoms) <= component.embedded_atoms):
            return False
        claimed = set(group.atoms) | set(organyl_atoms)
    else:
        if not set(group.atoms) <= component.atoms:
            return False
        claimed = set(organyl_atoms)
    if len(claimed) != len(organyl_atoms) + (len(group.atoms) if ring else 0):
        return False
    tagged = {atom.GetIdx() for atom in mol.GetAtoms() if int(atom.GetChiralTag())}
    assigned = dict(assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True))
    if set(assigned) != tagged or any(value not in {'R', 'S'} for value in assigned.values()):
        return False
    plan = _mapped_stereochemistry_plan(mol, Chem, component, dict(component.atom_locants))
    if plan is None or plan[2] or {item[0] for item in plan[1]} != tagged & component.atoms:
        return False
    embedded_stereo = set()
    for root, host, atoms, text in component.prefix_branch_ownership:
        atoms = frozenset(atoms)
        if (host not in component.atoms or claimed & atoms or not _one_cut(mol, root, host, atoms)
                or mapped_branch(mol, Chem, root, host) != (text, atoms)):
            return False
        if tagged & atoms:
            from .retained_acid_rules import mapped_ring_stereo_prefix_proved
            if not mapped_ring_stereo_prefix_proved(mol, Chem, root, host, atoms, text):
                return False
            embedded_stereo.update(tagged & atoms)
        claimed.update(atoms)
    if claimed != set(component.embedded_atoms) or tagged - component.atoms != embedded_stereo:
        return False
    acid_part = decorate_component(mol, Chem, component)
    return bool(acid_part and proposal.name == organyl + ' ' + acid_part)


def mapped_monoester_decision(mol, Chem):
    """Name an exact mapped monoester, retaining unresolved preference."""
    if (mol is None or mol.GetNumHeavyAtoms() > 80 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons() or a.HasQuery() or a.GetAtomMapNum()
                   for a in mol.GetAtoms())
            or any(b.HasQuery() or int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    # Avoid paying for source inventory on the common non-ester path.
    if not any(atom.GetAtomicNum() == 6 and not atom.IsInRing()
               and any(n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(
                   atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2 for n in atom.GetNeighbors())
               and any(n.GetAtomicNum() == 8 and n.GetDegree() == 2 and mol.GetBondBetweenAtoms(
                   atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1 for n in atom.GetNeighbors())
               for atom in mol.GetAtoms()):
        return None
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    group = _source_ester(inventory)
    if group is None:
        return None
    owner = _ester_organyl(mol, Chem, group)
    if owner is None:
        return None
    organyl, organyl_atoms = owner
    from .amino_acid_esters import retained_amino_acid_ester

    retained = retained_amino_acid_ester(mol, Chem, group, organyl, organyl_atoms)
    if retained:
        rule = 'P-41/P-65.6/P-103.2.3/P-103.2.6'
        return MethodDecision(retained, 'mapped-monoester', frozenset({group.center}),
                              frozenset(a.GetIdx() for a in mol.GetAtoms()), rule,
                              preference_issues=(SelectionLimitation(
                                  'monoester-parent', rule,
                                  'The retained amino-acid residue and complete ester arms are mapped.'),))
    if mol.GetAtomWithIdx(group.owner).IsInRing():
        from .mapped_ring_parents import records
        rows = records(mol, Chem, principal=(group,))
        if len(rows) != 1:
            return None
        proposal = rows[0]
        if (not proposal.name or not proposal.hydrogen_budget_proved or not proposal.annotation_coverage
                or proposal.method not in {'retained-fused', 'Hantzsch-Widman', 'monocyclic',
                                           'component-fused', 'benzo-heterocycle', 'von-Baeyer', 'spiro',
                                           'skeletal-replacement-von-Baeyer', 'skeletal-replacement-spiro'}
                or not _complete_component(mol, Chem, group, proposal, organyl, organyl_atoms, ring=True)):
            return None
        inherited_issues = tuple(proposal.preference_issues)
    else:
        from .parent_rules import _ester_acyl_parent_paths
        paths = [(group.center, *path) for path in
                 _ester_acyl_parent_paths(mol, Chem, group.center, group.owner)]
        if not paths or len(paths) > 64:
            return None
        def unsaturation(path):
            orders = tuple(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
                           for a, b in zip(path, path[1:]))
            return sum(order in {2, 3} for order in orders), orders.count(2)
        maximum = max(map(unsaturation, paths))
        paths = [path for path in paths if unsaturation(path) == maximum]
        from .complete_functional_parent import _render_functional_candidate
        proposals = []
        for path in paths:
            candidate = _render_functional_candidate(
                mol, Chem, (group,), 'ester', path, False, False,
                (set(group.atoms) - {group.center}) | set(organyl_atoms),
                [(organyl, organyl_atoms)], mapped_ring_prefixes=True,
                charge_role_proofs=(), locanted_acid_fallback=True)
            if (candidate is None or not candidate.name
                    or not _complete_component(mol, Chem, group, candidate, organyl, organyl_atoms, ring=False)):
                return None
            proposals.append(candidate)
        if len(proposals) > 1:
            if any(row.parent_selection is None for row in proposals):
                return None
            from .parent_selection import parent_frontier
            proposals = parent_frontier(proposals, lambda row: row.parent_selection)
        if not proposals or len({row.name for row in proposals}) != 1:
            return None
        proposal = proposals[0]
        inherited_issues = ()
    issues = [*inherited_issues, SelectionLimitation(
        'monoester-parent', 'P-41/P-44/P-45/P-51/P-65',
        'The ester, numbered parent and source branches are mapped; complete competing parent and method selection remain unresolved.')]
    try:
        if any(site.specified != Chem.StereoSpecified.Specified for site in Chem.FindPotentialStereo(mol)):
            issues.append(SelectionLimitation('stereochemistry', 'P-91',
                                             'Potential stereochemistry is unassigned.'))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return MethodDecision(proposal.name, 'mapped-monoester', frozenset({group.center}),
                          frozenset(atom.GetIdx() for atom in mol.GetAtoms()),
                          'P-41/P-44/P-65.6', proposal.atom_locants,
                          preference_issues=tuple(issues))
