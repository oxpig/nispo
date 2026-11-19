"""Original-graph aryloxy joins using a proved mapped ring prefix."""


def mapped_aryloxy_prefix(mol, Chem, root, host, atoms):
    """Return a ring-derived O prefix only when the complete join is proved."""
    oxygen = mol.GetAtomWithIdx(root)
    host_atom = mol.GetAtomWithIdx(host)
    bond = mol.GetBondBetweenAtoms(root, host)
    if (oxygen.GetAtomicNum() != 8 or oxygen.IsInRing() or oxygen.GetFormalCharge()
            or oxygen.GetDegree() != 2 or oxygen.GetTotalNumHs(includeNeighbors=True)
            or host_atom.GetAtomicNum() != 6 or bond is None or bond.GetBondTypeAsDouble() != 1
            or any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(host_atom).GetAtomicNum() == 8
                   for b in host_atom.GetBonds())):
        return None
    children = [n.GetIdx() for n in oxygen.GetNeighbors() if n.GetIdx() != host and n.GetAtomicNum() > 1]
    if len(children) != 1:
        return None
    child = children[0]
    child_atom = mol.GetAtomWithIdx(child)
    if (child_atom.GetAtomicNum() != 6 or not child_atom.IsInRing()
            or not child_atom.GetIsAromatic()
            or mol.GetBondBetweenAtoms(root, child).GetBondTypeAsDouble() != 1):
        return None
    # A benzene ring has the retained preferred prefix phenoxy, with full
    # substitution (P-63.2.2.2).  Do not replace that established writer with
    # a longer '(substituted phenyl)oxy' form.  Fused and heteroaryl rings do
    # not have the same retained-prefix rule.
    from .mapped_ring_parents import ring_systems
    child_system = next((system for system in ring_systems(mol) if child in system), frozenset())
    if (len(child_system) == 6 and all(
            mol.GetAtomWithIdx(i).GetAtomicNum() == 6
            and mol.GetAtomWithIdx(i).GetIsAromatic() for i in child_system)):
        return None
    atoms = frozenset(atoms)
    if (root not in atoms or host in atoms or any(
            mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetFormalCharge()
            or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
            or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in atoms)
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds()
                   if b.GetBeginAtomIdx() in atoms or b.GetEndAtomIdx() in atoms)):
        return None
    from .branch_rules import collect_subtree
    owned = frozenset(collect_subtree(mol, child, root))
    boundary = {(i, n.GetIdx()) for i in owned for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in owned}
    if not owned or owned | {root} != atoms or boundary != {(child, root)}:
        return None
    from .mapped_ring_parents import records, rooted_prefix
    rows = records(mol, Chem, scope=owned, attachment=child)
    if len(rows) != 1:
        return None
    row = rows[0]
    component = row.construction
    if (not row.name or row.method not in {'retained-fused', 'Hantzsch-Widman',
                                           'monocyclic', 'component-fused', 'benzo-heterocycle'}
            or row.preference_issues or not row.hydrogen_budget_proved or not row.annotation_coverage
            or component is None or component.atoms & component.embedded_atoms
            or component.atoms | component.embedded_atoms != owned
            or rooted_prefix(mol, Chem, child, root, owned) != (row.name, owned)):
        return None
    from .formatting import alkyl_to_alkoxy
    text = alkyl_to_alkoxy(row.name)
    ring_atoms = frozenset(i for i in owned if mol.GetAtomWithIdx(i).IsInRing())
    return text, atoms, ring_atoms
