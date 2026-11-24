"""Original-graph ylidene components for C=N substitutive nomenclature."""


def carbon_ylidene(mol, Chem, carbon, nitrogen, scope, *, terminal_contraction=False):
    """Name the carbon side of one C=N cut and return its skeletal locants.

    External geometry belongs to the consumer. The local component keeps
    internal bond orders, ring numbering and atom configurations.
    """
    from dataclasses import replace
    from .component_mapping import decorate_component
    from .formatting import attach_suffix

    if carbon not in scope or {(i, a.GetIdx()) for i in scope for a in mol.GetAtomWithIdx(i).GetNeighbors()
        if a.GetAtomicNum() > 1 and a.GetIdx() not in scope} != {(carbon, nitrogen)}:
        return None
    graph = Chem.Mol(mol)
    graph.GetBondBetweenAtoms(carbon, nitrogen).SetStereo(Chem.BondStereo.STEREONONE)
    if mol.GetAtomWithIdx(carbon).IsInRing():
        from .constants import ROOTS
        from .mapped_ring_parents import records

        if (len(scope) in ROOTS and any(scope == frozenset(r) for r in mol.GetRingInfo().AtomRings())
                and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                        and not mol.GetAtomWithIdx(i).GetIsotope()
                        and mol.GetAtomWithIdx(i).GetTotalNumHs() == 2 for i in scope - {carbon})
                and not mol.GetAtomWithIdx(carbon).GetIsotope()):
            return 'cyclo' + ROOTS[len(scope)] + 'ylidene', ((carbon, '1'),)
        rows = records(graph, Chem, scope=scope, attachment=carbon, principal=(),
                       general_branches=True, local_inventory=True)
        if len(rows) != 1 or rows[0].construction is None:
            return None
        component = rows[0].construction
        if component.atoms | component.embedded_atoms != scope:
            return None
        if any(mol.GetAtomWithIdx(i).GetChiralTag() for i in scope):
            from .component_mapping import _general_branch_stereo_is_cited
            from .stereochemistry import assigned_atom_stereochemistry
            original = dict(assigned_atom_stereochemistry(mol, Chem, include_pseudo=True))
            rendered = dict(assigned_atom_stereochemistry(graph, Chem, include_pseudo=True))
            if (any(original.get(i) != rendered.get(i) for i in scope)
                    or not _general_branch_stereo_is_cited(mol, Chem, scope, rows[0].name)):
                return None
        row = rows[0]
        if (terminal_contraction and row.parent.startswith('cyclo') and row.parent.endswith('ane')
                and any(set(ring) == set(row.parent_atoms) for ring in mol.GetRingInfo().AtomRings())
                and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in row.parent_atoms)
                and dict(row.atom_locants)[carbon] == '1' and not row.hydro_locants):
            body = decorate_component(graph, Chem, replace(component, parent=row.parent[:-3] + 'ylidene'))
            if body:
                return body, row.atom_locants
        return row.name, row.atom_locants
    if any(mol.GetAtomWithIdx(i).GetChiralTag() for i in scope):
        return None
    from .rooted_carbon_parents import chain_orientations

    best = []
    for key, component in chain_orientations(mol, Chem, carbon, nitrogen, scope):
        locant = dict(component.atom_locants)[carbon]
        parent = component.parent
        ylidene = attach_suffix(parent, 'ylidene', (locant,))
        if terminal_contraction and locant == '1' and parent.endswith('an'):
            ylidene = parent[:-2] + 'ylidene'
        if terminal_contraction and len(component.atoms) == 1 and component.prefixes:
            # P-29.6.1 retains unsubstituted benzylidene. Other aryl groups
            # substitute methylidene at its only carbon (P-14.3.4).
            if (len(component.prefixes) == 1 and component.prefixes[0][1] == 'phenyl' and len(scope) == 7
                    and not any(mol.GetAtomWithIdx(i).GetIsotope() for i in scope)):
                return 'benzylidene', component.atom_locants
            component = replace(component, prefix_mode='unlocanted')
        body = decorate_component(graph, Chem, replace(component, parent=ylidene))
        if body:
            best.append((key, body, component.atom_locants))
    if not best:
        return None
    lowest = min(key for key, _, _ in best)
    winners = [(body, seq) for key, body, seq in best if key == lowest]
    if len({body for body, _ in winners}) != 1:
        return None
    return winners[0]

