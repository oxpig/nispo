"""Source-owned retained fused lactam parent with mapped substituents."""


def _stereo_branch_proved(mol, Chem, root, host, atoms, text):
    """Prove annotated branch atoms on a fully mapped carbon path or ring."""
    tagged = {i for i in atoms if int(mol.GetAtomWithIdx(i).GetChiralTag())}
    if not tagged:
        return True
    if mol.GetAtomWithIdx(root).IsInRing():
        from .retained_acid_rules import mapped_ring_stereo_prefix_proved
        return mapped_ring_stereo_prefix_proved(mol, Chem, root, host, atoms, text)
    if mol.GetAtomWithIdx(root).GetAtomicNum() != 6:
        return False
    carbon, pending = set(), [root]
    while pending:
        idx = pending.pop()
        if idx in carbon:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if idx not in atoms or atom.IsInRing() or atom.GetAtomicNum() != 6:
            return False
        carbon.add(idx)
        pending.extend(n.GetIdx() for n in atom.GetNeighbors()
                       if n.GetIdx() in atoms and n.GetAtomicNum() == 6
                       and not n.IsInRing() and n.GetIdx() not in carbon)
    if not tagged <= carbon:
        return False
    from .component_mapping import _internally_rooted_carbon_prefix
    return _internally_rooted_carbon_prefix(
        mol, Chem, root, host, atoms, carbon) == (text, atoms)


def retained_fused_lactam_general(mol, Chem):
    """Deliver a completely mapped fused lactam, leaving method choice open.

    Exactly one source-senior lactam must lie within the retained fused core.
    Parent, suffix and each original one-cut branch cover the input, including
    a mapped witness for every nested ring and any assigned branch stereo.
    """
    if (mol is None or mol.GetNumHeavyAtoms() > 120 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons() or a.HasQuery() or a.GetAtomMapNum()
                   for a in mol.GetAtoms())
            or any(b.HasQuery() or int(b.GetStereo()) or int(b.GetBondDir())
                   for b in mol.GetBonds())):
        return None
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.principal) != 1
            or inventory.principal[0].kind != 'lactam'
            or len(inventory.groups) != len(inventory.source_group_facts)):
        return None
    principal = inventory.principal[0]
    pairs = tuple(zip(inventory.groups, inventory.source_group_facts, strict=True))
    if (any(fact is None or fact.class_rank is None for _, fact in pairs)
            or any(fact.class_rank != 16 if group is principal else fact.class_rank <= 16
                   for group, fact in pairs)):
        return None
    from .mapped_ring_parents import parent_facts, records, ring_systems
    cores = [core for core in ring_systems(mol) if principal.center in core]
    if len(cores) != 1 or principal.ligand not in cores[0] or len(cores[0]) < 8:
        return None
    rows = records(mol, Chem, core_filter=cores[0], principal=(principal,))
    if len(rows) != 1:
        return None
    record = rows[0]
    component = record.construction
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if (record.method != 'retained-fused' or not record.name
            or not record.annotation_coverage or not record.hydrogen_budget_proved
            or record.preference_issues or component is None
            or frozenset(record.parent_atoms) != cores[0]
            or record.principal_centers != {principal.center}
            or component.atoms & component.embedded_atoms
            or component.atoms | component.embedded_atoms != heavy
            or len(component.prefix_branch_ownership) != len(component.prefixes)
            or parent_facts(mol, record, principal.seniority) is None):
        return None
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch, mapped_branch_ring_atoms
    suffix = set(principal.atoms) - set(component.atoms)
    owned = set(suffix)
    for root, host, atoms, text in component.prefix_branch_ownership:
        atoms = frozenset(atoms)
        boundary = {(i, n.GetIdx()) for i in atoms
                    for n in mol.GetAtomWithIdx(i).GetNeighbors()
                    if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms}
        if (host not in component.atoms or root not in atoms or atoms & owned
                or boundary != {(root, host)}
                or frozenset(collect_subtree(mol, root, host)) != atoms
                or mapped_branch(mol, Chem, root, host) != (text, atoms)
                or not _stereo_branch_proved(mol, Chem, root, host, atoms, text)):
            return None
        rings = frozenset(i for i in atoms if mol.GetAtomWithIdx(i).IsInRing())
        if not rings <= mapped_branch_ring_atoms(mol, Chem, root, host, text, atoms):
            return None
        owned.update(atoms)
    if owned != set(component.embedded_atoms):
        return None
    from .naming_methods import MethodDecision
    from .naming_result import SelectionLimitation
    return MethodDecision(record.name, 'substitutive', record.principal_centers, heavy,
        'P-25.2.1/P-41/P-44/P-66', record.atom_locants,
        preference_issues=(SelectionLimitation(
            'parent-method', 'P-44/P-51',
            'The retained ring and original branches are mapped; the complete parent and method comparison remains unresolved.'),))
