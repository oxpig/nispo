"""Source-owned N-substituted carbamoyl branches on cyclic parents."""


def mapped_subordinate_carbamoyl(mol, Chem, root, host):
    """Return one fully mapped ring-bound secondary amide prefix, or decline.

    A unique, strictly senior source class is required. The carbonyl, its
    nitrogen arm and every nested ring are bound to the original one-cut
    graph. Competing amide sites, annotations and unproved priorities remain
    outside this construction.
    """
    from .functional_groups import collect_subtree
    from ..component_mapping import mapped_branch, mapped_branch_ring_atoms
    from ..extended_inventory import selection_inventory
    from ..formatting import enclose_name_unit
    from ..group_priority import compare_functional_group_priority

    carbon = mol.GetAtomWithIdx(root)
    cut = mol.GetBondBetweenAtoms(root, host)
    if (carbon.GetAtomicNum() != 6 or carbon.IsInRing() or carbon.GetDegree() != 3
            or not mol.GetAtomWithIdx(host).IsInRing()
            or cut is None or cut.GetBondTypeAsDouble() != 1 or mol.GetStereoGroups()):
        return None
    scope = frozenset(collect_subtree(mol, root, host))
    if not scope or root not in scope or host in scope:
        return None
    boundary = {(i, n.GetIdx()) for i in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in scope}
    if (boundary != {(root, host)} or any(
            mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetFormalCharge()
            or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
            or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in scope)
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds()
                   if b.GetBeginAtomIdx() in scope or b.GetEndAtomIdx() in scope)):
        return None
    neighbors = [n for n in carbon.GetNeighbors() if n.GetIdx() != host]
    oxygens = [n for n in neighbors if n.GetAtomicNum() == 8 and n.GetDegree() == 1
               and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 2]
    nitrogens = [n for n in neighbors if n.GetAtomicNum() == 7 and not n.IsInRing()
                 and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 1]
    if len(neighbors) != 2 or len(oxygens) != 1 or len(nitrogens) != 1:
        return None
    nitrogen = nitrogens[0]
    if nitrogen.GetTotalNumHs(includeNeighbors=True) != 1:
        return None
    arms = [n.GetIdx() for n in nitrogen.GetNeighbors()
            if n.GetIdx() != root and n.GetAtomicNum() > 1]
    if len(arms) != 1 or mol.GetAtomWithIdx(arms[0]).GetAtomicNum() != 6:
        return None
    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.principal) != 1
            or len(inventory.source_group_facts) != len(inventory.groups)):
        return None
    principal = inventory.principal[0]
    target = [g for g in inventory.groups if g.kind == 'amide'
              and g.center == root and g.owner == host and g.ligand == nitrogen.GetIdx()]
    if len(target) != 1 or set(principal.atoms) & scope:
        return None
    facts = {id(g): f for g, f in zip(inventory.groups, inventory.source_group_facts, strict=True)}
    if facts.get(id(principal)) is None or any(
            f is None for f in inventory.source_group_facts):
        return None
    if any(compare_functional_group_priority(facts[id(principal)], facts[id(group)]).status != 'senior'
           for group in inventory.groups if group is not principal):
        return None
    organyl, owned = mapped_branch(mol, Chem, arms[0], nitrogen.GetIdx())
    owned = frozenset(owned)
    core = {root, oxygens[0].GetIdx(), nitrogen.GetIdx()}
    if (not organyl or owned & core or set(scope) != core | set(owned)
            or frozenset(collect_subtree(mol, arms[0], nitrogen.GetIdx())) != owned):
        return None
    ring_atoms = frozenset(i for i in owned if mol.GetAtomWithIdx(i).IsInRing())
    witnessed = mapped_branch_ring_atoms(mol, Chem, arms[0], nitrogen.GetIdx(), organyl, owned)
    if not ring_atoms <= witnessed:
        return None
    # Simple organyl names can be joined directly; locanted or nested
    # components require the enclosing marks prescribed by P-16.5.
    from ..organyl_prefix_components import organyl_prefix_nature
    nature = organyl_prefix_nature(mol, arms[0], owned, organyl)
    unit = (organyl if nature == 'simple' or not any(ch in organyl for ch in '-()[]{}')
            else enclose_name_unit(organyl))
    return unit + 'carbamoyl', scope


def mapped_subordinate_ring_nitrogen_carbonyl(mol, Chem, root, host):
    """Construct a source-senior parent's one-cut N-ring pseudoketone prefix."""
    from .functional_groups import collect_subtree
    from ..component_mapping import mapped_branch, mapped_branch_ring_atoms
    from ..extended_inventory import selection_inventory
    from ..formatting import enclose_name_unit
    from ..group_priority import compare_functional_group_priority

    carbon = mol.GetAtomWithIdx(root)
    cut = mol.GetBondBetweenAtoms(root, host)
    if (carbon.GetAtomicNum() != 6 or carbon.IsInRing() or carbon.GetDegree() != 3
            or not mol.GetAtomWithIdx(host).IsInRing()
            or cut is None or cut.GetBondTypeAsDouble() != 1 or mol.GetStereoGroups()):
        return None
    scope = frozenset(collect_subtree(mol, root, host))
    boundary = {(i, n.GetIdx()) for i in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in scope}
    if (root not in scope or host in scope or boundary != {(root, host)}
            or any(mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                   or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in scope)
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds()
                   if b.GetBeginAtomIdx() in scope or b.GetEndAtomIdx() in scope)):
        return None
    neighbors = [n for n in carbon.GetNeighbors() if n.GetIdx() != host]
    oxygen = [n for n in neighbors if n.GetAtomicNum() == 8 and n.GetDegree() == 1
              and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 2]
    nitrogen = [n for n in neighbors if n.GetAtomicNum() == 7 and n.IsInRing()
                and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 1]
    if len(neighbors) != 2 or len(oxygen) != 1 or len(nitrogen) != 1:
        return None
    n = nitrogen[0]
    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.principal) != 1
            or len(inventory.source_group_facts) != len(inventory.groups)):
        return None
    principal = inventory.principal[0]
    target = [g for g in inventory.groups if g.kind == 'pseudoketone'
              and g.center == root and g.owner == host and g.ligand == n.GetIdx()]
    facts = {id(g): f for g, f in zip(inventory.groups, inventory.source_group_facts, strict=True)}
    if (len(target) != 1 or set(principal.atoms) & scope
            or any(f is None for f in inventory.source_group_facts)
            or any(compare_functional_group_priority(facts[id(principal)], facts[id(group)]).status != 'senior'
                   for group in inventory.groups if group is not principal)):
        return None
    text, owned = mapped_branch(mol, Chem, n.GetIdx(), root)
    owned = frozenset(owned)
    if (not text or owned | {root, oxygen[0].GetIdx()} != scope
            or frozenset(collect_subtree(mol, n.GetIdx(), root)) != owned):
        return None
    rings = frozenset(i for i in owned if mol.GetAtomWithIdx(i).IsInRing())
    if not rings <= mapped_branch_ring_atoms(mol, Chem, n.GetIdx(), root, text, owned):
        return None
    return enclose_name_unit(text) + 'carbonyl', scope
