"""Mapped acid-derived sulfonyl components on original molecular atoms."""

from dataclasses import replace


def mapped_sulfonyl_prefix(mol, Chem, root, blocked, atoms):
    """Build a bounded ring-SO2- prefix and prove component ownership.

    P-65.3.2.2.2 derives sulfonyl prefixes from sulfonic acids. The current
    scope covers unannotated simple carbon-ring acid parents, with any
    independently mapped, unannotated substituents on the ring.

    Return ``None`` outside this scope and an empty pair for a recognized
    but unproved construction; a successful pair owns exactly ``atoms``.
    """
    acyclic = _mapped_linear_alkanesulfonyl(mol, root, blocked, frozenset(atoms))
    if acyclic is not None:
        return acyclic
    return _mapped_ring_sulfonyl_component(mol, Chem, root, blocked, atoms, 'sulfonyl')


def _mapped_linear_alkanesulfonyl(mol, root, blocked, scope):
    """Own a simple linear alkane-SO2 unit as its acid-derived prefix."""
    from .functional_groups import collect_subtree
    from .preferred_acyl import _carbon_paths
    from ..component_mapping import has_annotations
    from ..constants import ROOTS

    sulfur = mol.GetAtomWithIdx(root)
    cut = mol.GetBondBetweenAtoms(root, blocked)
    if (sulfur.GetAtomicNum() != 16 or sulfur.IsInRing()
            or sulfur.GetFormalCharge() or sulfur.GetNumRadicalElectrons()
            or cut is None or cut.GetBondTypeAsDouble() != 1
            or scope != frozenset(collect_subtree(mol, root, blocked))
            or has_annotations(mol, scope)):
        return None
    neighbors = [a for a in sulfur.GetNeighbors()
                 if a.GetAtomicNum() > 1 and a.GetIdx() != blocked]
    oxy = [a for a in neighbors if a.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(root, a.GetIdx()).GetBondTypeAsDouble() == 2]
    carbons = [a for a in neighbors if a.GetAtomicNum() == 6 and not a.IsInRing()
               and mol.GetBondBetweenAtoms(root, a.GetIdx()).GetBondTypeAsDouble() == 1]
    if (len(neighbors) != 3 or len(oxy) != 2 or len(carbons) != 1
            or any(a.GetDegree() != 1 or a.GetFormalCharge() for a in oxy)):
        return None
    paths = _carbon_paths(mol, carbons[0].GetIdx(), {root})
    if len(paths) != 1 or len(paths[0]) not in ROOTS:
        return None
    path = paths[0]
    if (set(scope) != {root, *path, *(a.GetIdx() for a in oxy)}
            or any(mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in path)):
        return None
    parent = ROOTS[len(path)] + 'ane'
    return parent + ('' if len(path) <= 2 else '-1-') + 'sulfonyl', scope


def mapped_sulfonamido_prefix(mol, Chem, root, blocked, atoms):
    """Build R-SO2-NH- using the preferred amide prefix (P-66.1.1.4.3)."""
    nitrogen = mol.GetAtomWithIdx(root)
    cut = mol.GetBondBetweenAtoms(root, blocked)
    if (nitrogen.GetAtomicNum() != 7 or nitrogen.IsInRing()
            or nitrogen.GetFormalCharge() or nitrogen.GetNumRadicalElectrons()
            or nitrogen.GetIsotope() or nitrogen.GetTotalNumHs(includeNeighbors=True) != 1
            or cut is None or cut.GetBondTypeAsDouble() != 1):
        return None
    children = [a for a in nitrogen.GetNeighbors()
                if a.GetAtomicNum() > 1 and a.GetIdx() != blocked]
    if len(children) != 1 or children[0].GetAtomicNum() != 16:
        return None
    sulfur = children[0].GetIdx()
    from .functional_groups import collect_subtree
    sulfur_atoms = frozenset(collect_subtree(mol, sulfur, root))
    if frozenset(collect_subtree(mol, root, blocked)) != frozenset(atoms):
        return '', frozenset()
    if sulfur_atoms | {root} != frozenset(atoms):
        return '', frozenset()
    acyclic = _mapped_methanesulfonamido(mol, sulfur, root, frozenset(atoms))
    if acyclic is not None:
        return acyclic
    built = _mapped_ring_sulfonyl_component(
        mol, Chem, sulfur, root, sulfur_atoms, 'sulfonamido')
    if built is None:
        return None
    name, owned = built
    return (name, frozenset(atoms)) if name and owned == sulfur_atoms else ('', frozenset())


def _mapped_methanesulfonamido(mol, sulfur, nitrogen, scope):
    """Build a fully owned (optionally halo-substituted) methane amide.

    The prefix is preferred only when a separate, strictly senior
    characteristic group is proved in the full molecular inventory.
    """
    from ..component_mapping import has_annotations
    from ..constants import HALO
    from ..extended_inventory import selection_inventory
    from ..formatting import format_prefixes

    center = mol.GetAtomWithIdx(sulfur)
    if (center.IsInRing() or center.GetFormalCharge() or center.GetNumRadicalElectrons()
            or has_annotations(mol, scope)):
        return None
    neighbors = [a for a in center.GetNeighbors()
                 if a.GetAtomicNum() > 1 and a.GetIdx() != nitrogen]
    oxy = [a for a in neighbors if a.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(sulfur, a.GetIdx()).GetBondTypeAsDouble() == 2]
    carbons = [a for a in neighbors if a.GetAtomicNum() == 6
               and mol.GetBondBetweenAtoms(sulfur, a.GetIdx()).GetBondTypeAsDouble() == 1]
    if (len(neighbors) != 3 or len(oxy) != 2 or len(carbons) != 1
            or any(a.GetDegree() != 1 or a.GetFormalCharge() for a in oxy)):
        return None
    carbon = carbons[0]
    substituents = [a for a in carbon.GetNeighbors()
                    if a.GetIdx() != sulfur and a.GetAtomicNum() > 1]
    if (carbon.IsInRing() or carbon.GetFormalCharge() or carbon.GetNumRadicalElectrons()
            or any(a.GetSymbol() not in HALO or a.GetDegree() != 1
                   or a.GetFormalCharge()
                   or mol.GetBondBetweenAtoms(carbon.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() != 1
                   for a in substituents)
            or set(scope) != {nitrogen, sulfur, carbon.GetIdx(),
                              *(a.GetIdx() for a in oxy),
                              *(a.GetIdx() for a in substituents)}):
        return None
    inventory = selection_inventory(mol)
    targets = [g for g in inventory.groups
               if g.kind == 'sulfonamide' and g.center == sulfur]
    if not inventory.supported or len(targets) != 1:
        return None
    senior = [g for g in inventory.groups if g.seniority < targets[0].seniority]
    if not senior or any(set(g.atoms) & scope for g in senior):
        return None
    prefixes = [('1', HALO[a.GetSymbol()]) for a in substituents]
    return format_prefixes(prefixes, omit_locants=True) + 'methanesulfonamido', scope


def _mapped_ring_sulfonyl_component(mol, Chem, root, blocked, atoms, ending):
    sulfur = mol.GetAtomWithIdx(root)
    parent = mol.GetAtomWithIdx(blocked)
    cut = mol.GetBondBetweenAtoms(root, blocked)
    if (sulfur.GetAtomicNum() != 16 or sulfur.IsInRing()
            or parent.GetAtomicNum() not in {6, 7}
            or cut is None or cut.GetBondTypeAsDouble() != 1):
        return None
    neighbors = [a for a in sulfur.GetNeighbors()
                 if a.GetAtomicNum() > 1 and a.GetIdx() != blocked]
    oxo = [a for a in neighbors if a.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(root, a.GetIdx()).GetBondTypeAsDouble() == 2]
    carbon = [a for a in neighbors if a.GetAtomicNum() == 6 and a.IsInRing()
              and mol.GetBondBetweenAtoms(root, a.GetIdx()).GetBondTypeAsDouble() == 1]
    if len(neighbors) != 3 or len(oxo) != 2 or len(carbon) != 1:
        return None
    atoms = frozenset(atoms)
    if sulfur.GetFormalCharge() or sulfur.GetNumRadicalElectrons() or sulfur.GetTotalNumHs(includeNeighbors=True):
        return '', frozenset()
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
           or a.GetTotalNumHs(includeNeighbors=True)
           or len([n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]) != 1
           for a in oxo):
        return '', frozenset()
    from .functional_groups import collect_subtree
    from ..component_mapping import decorate_component, has_annotations
    from ..mapped_ring_parents import records, ring_systems
    if (frozenset(collect_subtree(mol, root, blocked)) != atoms
            or blocked in atoms or has_annotations(mol, atoms)
            or any(mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in atoms)):
        return '', frozenset()
    ligand = carbon[0].GetIdx()
    ligand_atoms = frozenset(collect_subtree(mol, ligand, root))
    if ligand_atoms | {root, *(a.GetIdx() for a in oxo)} != atoms:
        return '', frozenset()
    core = next((system for system in ring_systems(mol) if ligand in system), None)
    if core is None or not core <= ligand_atoms or len(core) < 3 or len(core) > 8:
        return '', frozenset()
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in core):
        return '', frozenset()
    core_bonds = [b for b in mol.GetBonds()
                  if b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core]
    if (len(core_bonds) != len(core)
            or any(sum(n.GetIdx() in core for n in mol.GetAtomWithIdx(i).GetNeighbors()) != 2 for i in core)):
        return '', frozenset()
    if len(core) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in core) and all(b.GetIsAromatic() for b in core_bonds):
        acid_parent = 'benzene'
    elif all(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in core) and all(b.GetBondTypeAsDouble() == 1 for b in core_bonds):
        from ..constants import ROOTS
        stem = ROOTS.get(len(core))
        if not stem:
            return '', frozenset()
        acid_parent = 'cyclo' + stem + 'ane'
    else:
        return '', frozenset()
    candidates = records(mol, Chem, scope=ligand_atoms, attachment=ligand, core_filter=core)
    if len(candidates) != 1:
        return '', frozenset()
    record = candidates[0]
    component = record.construction
    if (record.parent != acid_parent or component is None
            or frozenset(record.parent_atoms) != core
            or component.atoms | component.embedded_atoms != ligand_atoms
            or dict(component.atom_locants).get(ligand) != '1'):
        return '', frozenset()
    name = decorate_component(mol, Chem, replace(component, parent=acid_parent + ending))
    return (name, atoms) if name else ('', frozenset())
