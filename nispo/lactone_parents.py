"""Heterocyclic pseudoketone names for lactones and their thiones (P-65.6.3.5.1)."""
from dataclasses import replace
from .functional_class_queries import lactone_query


def lactone_decision(mol, Chem):
    from .component_mapping import decorate_component
    from .extended_inventory import selection_inventory
    from .formatting import attach_suffix, multiplied_suffix
    from .mapped_ring_parents import records, ring_systems
    from .naming_methods import MethodDecision

    sites = mol.GetSubstructMatches(lactone_query(Chem))
    if not sites or mol.GetStereoGroups():
        return None
    chalcogens = {x for _, _, x, _ in sites}
    elements = {mol.GetAtomWithIdx(x).GetAtomicNum() for x in chalcogens}
    if len(elements) > 2:
        return None
    # P-64.3 cites the senior chalcogen as suffix (O, S, Se, Te).
    element = min(elements)
    subordinate = {x: c for _, c, x, _ in sites if mol.GetAtomWithIdx(x).GetAtomicNum() != element}
    graph = mol
    if elements != {8}:
        # The isovalent oxygen analogue supplies the ester inventory and ring
        # hydrogen accounting. The final suffix and descriptors use the source.
        if any(mol.GetAtomWithIdx(x).GetIsotope() for x in chalcogens):
            return None
        graph = Chem.Mol(mol)
        for x in chalcogens:
            graph.GetAtomWithIdx(x).SetAtomicNum(8)
        try:
            Chem.SanitizeMol(graph)
        except (ValueError, RuntimeError):
            return None
        if any(a.GetChiralTag() for a in mol.GetAtoms()) or any(b.GetStereo() for b in mol.GetBonds()):
            from .stereochemistry import assigned_atom_stereochemistry, assigned_bond_stereochemistry
            if (assigned_atom_stereochemistry(graph, Chem, atom_order='idx', include_pseudo=True)
                    != assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True)
                    or assigned_bond_stereochemistry(graph, Chem) != assigned_bond_stereochemistry(mol, Chem)):
                return None

    all_centers = frozenset(c for _, c, _, _ in sites)
    centers = all_centers - set(subordinate.values())
    inventory = selection_inventory(graph)
    groups = inventory.principal
    if (not inventory.supported or not groups or any(g.kind != 'lactone' for g in groups)
            or {g.center for g in groups} != all_centers):
        return None
    groups = tuple(g for g in groups if g.center in centers)
    core = next((ring for ring in ring_systems(graph) if all_centers <= ring), None)
    if core is None:
        return None
    scope = frozenset(range(mol.GetNumAtoms()))
    names = {}
    for row in records(graph, Chem, principal=groups, core_filter=core, general_branches=True):
        component = row.construction
        if (not row.hydrogen_budget_proved or not row.annotation_coverage or row.principal_centers != centers
                or component is None or component.atoms | component.embedded_atoms != scope):
            continue
        if elements == {8}:
            names[row.name] = row
            continue
        if subordinate:
            # Exchanging identical oxo prefixes for another single chalcogen
            # prefix preserves the numbering comparison only if all prefix
            # positions take part. Match each one to its actual exocyclic atom.
            ownership = component.prefix_branch_ownership
            if ({root: host for root, host, atoms, text in ownership
                 if atoms == {root} and text == 'oxo'} != subordinate
                    or len(ownership) != len(component.prefixes) or len(ownership) != len(subordinate)):
                continue
            prefix = {16: 'sulfanylidene', 34: 'selanylidene', 52: 'tellanylidene'}[max(elements)]
            component = replace(component,
                prefixes=tuple((loc, prefix) for loc, _ in component.prefixes),
                prefix_branch_ownership=tuple((a, b, atoms, prefix) for a, b, atoms, _ in ownership))
        if element == 8:
            name = decorate_component(mol, Chem, component)
            if name:
                names[name] = row
            continue
        locants = ','.join(row.suffix_locants)
        if row.added_h_locants:
            locants += '(' + ','.join(loc + 'H' for loc in row.added_h_locants) + ')'
        suffix = multiplied_suffix(len(groups), {16: 'thione', 34: 'selone', 52: 'tellone'}[element])
        parent = attach_suffix(row.hydride, suffix, (locants,))
        name = decorate_component(mol, Chem, replace(component, parent=parent))
        if name:
            names[name] = row
    if len(names) != 1:
        return None
    name, row = names.popitem()
    return MethodDecision(name, 'substitutive', centers, scope, 'P-65.6.3.5.1',
                          row.atom_locants, preference_issues=row.preference_issues)
