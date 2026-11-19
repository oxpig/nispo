"""Bounded, original-graph general continuations for principal amides.

The complete parent and method frontier may be blocked by a nested ring
prefix. A unique acyclic chain and an exact suffix/prefix atom partition
still supply a useful general name. This operation never marks that name as
preferred; unresolved comparisons remain attached to the delivery.
"""
from collections import Counter

from .branch_rules import collect_subtree
from .component_mapping import (
    _record_branch_ring_atoms, fully_covered, mapped_branch,
    mapped_branch_ring_atoms,
)
from .extended_inventory import selection_inventory
from .formatting import enclose_name_unit
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def _ring_methyl_sulfanyl(mol, Chem, root, blocked):
    """Join S-CH2-ring only when the original ring writer owns its scope."""
    atom = mol.GetAtomWithIdx(root)
    if (atom.GetAtomicNum() != 16 or atom.IsInRing() or atom.GetFormalCharge()
            or atom.GetNumRadicalElectrons() or atom.GetIsotope()
            or mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble() != 1):
        return None
    children = [a for a in atom.GetNeighbors()
                if a.GetIdx() != blocked and a.GetAtomicNum() > 1]
    if (len(children) != 1 or children[0].GetAtomicNum() != 6
            or children[0].IsInRing()
            or children[0].GetTotalNumHs(includeNeighbors=True) != 2
            or mol.GetBondBetweenAtoms(root, children[0].GetIdx()).GetBondTypeAsDouble() != 1):
        return None
    carbon = children[0].GetIdx()
    rings = [a for a in children[0].GetNeighbors()
             if a.GetIdx() != root and a.GetAtomicNum() > 1]
    if (len(rings) != 1 or not rings[0].IsInRing()
            or mol.GetBondBetweenAtoms(carbon, rings[0].GetIdx()).GetBondTypeAsDouble() != 1):
        return None
    from .annular_carbamate_carboxylate_methods import _proved_ring_branch
    ring_name, ring_atoms = _proved_ring_branch(mol, Chem, rings[0].GetIdx(), carbon)
    scope = frozenset(collect_subtree(mol, root, blocked))
    if not ring_name or ring_atoms | {root, carbon} != scope:
        return None
    text = enclose_name_unit(enclose_name_unit(ring_name) + 'methyl') + 'sulfanyl'
    _record_branch_ring_atoms(mol, Chem, root, blocked, text, scope,
        (i for i in ring_atoms if mol.GetAtomWithIdx(i).IsInRing()))
    return text, scope


def _branch(mol, Chem, root, blocked):
    text, atoms = mapped_branch(mol, Chem, root, blocked)
    if text and atoms:
        nested_heterorings = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()
                              and mol.GetAtomWithIdx(i).GetAtomicNum() != 6}
        proved = mapped_branch_ring_atoms(mol, Chem, root, blocked, text, atoms)
        if nested_heterorings <= proved:
            return text, atoms
    return _ring_methyl_sulfanyl(mol, Chem, root, blocked)


def acyclic_amide_general(mol, Chem):
    """Return a fully owned, unresolved general amide name or decline."""
    if (mol.GetNumHeavyAtoms() > 100 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()
            or any(a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
                   for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    inventory = selection_inventory(mol)
    source_frontier = getattr(inventory, 'source_frontier', None)
    if (not inventory.supported or len(inventory.principal) != 1
            or inventory.principal[0].kind != 'amide' or source_frontier is None
            or source_frontier.status != 'selected' or source_frontier.unknown_pairs
            or source_frontier.cycle):
        return None
    group = inventory.principal[0]
    carbonyl = mol.GetAtomWithIdx(group.center)
    nitrogen = mol.GetAtomWithIdx(group.ligand)
    if (carbonyl.IsInRing() or nitrogen.IsInRing() or nitrogen.GetAtomicNum() != 7
            or mol.GetBondBetweenAtoms(group.center, group.ligand).GetBondTypeAsDouble() != 1):
        return None
    oxo = [a for a in carbonyl.GetNeighbors() if a.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(group.center, a.GetIdx()).GetBondTypeAsDouble() == 2]
    owner = [a for a in carbonyl.GetNeighbors() if a.GetAtomicNum() == 6
             and mol.GetBondBetweenAtoms(group.center, a.GetIdx()).GetBondTypeAsDouble() == 1]
    if (len(oxo) != 1 or oxo[0].GetDegree() != 1 or len(owner) != 1
            or owner[0].IsInRing() or carbonyl.GetDegree() != 3
            or set(group.atoms) != {group.center, group.ligand, oxo[0].GetIdx()}):
        return None
    n_roots = [a.GetIdx() for a in nitrogen.GetNeighbors()
               if a.GetIdx() != group.center and a.GetAtomicNum() > 1]
    if (len(n_roots) > 2 or nitrogen.GetTotalValence() != 3
            or nitrogen.GetTotalNumHs(includeNeighbors=True) != 2 - len(n_roots)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                                or mol.GetBondBetweenAtoms(i, group.ligand).GetBondTypeAsDouble() != 1
                                for i in n_roots)):
        return None
    # A heteroaromatic nested component is the bounded unsupported family
    # motivating this continuation. Ordinary cases keep the established path.
    if not any(a.IsInRing() and a.GetAtomicNum() not in {1, 6} for a in mol.GetAtoms()):
        return None
    from .branch.preferred_acyl import _carbon_paths
    from .constants import ROOTS
    # Keep a subordinate nitrile's C#N unit outside an alkane parent, as at
    # the shared functional-parent producer boundary. This continuation calls
    # the renderer directly, so it must preserve the same skeletal exclusion.
    nitrile_carbons = {item.center for item in inventory.groups
                       if item.kind == 'nitrile' and item != group}
    paths = _carbon_paths(mol, group.center, {group.ligand, *nitrile_carbons})
    if len(paths) != 1 or len(paths[0]) < 2 or len(paths[0]) not in ROOTS:
        return None
    path = tuple(paths[0])
    skipped = set(group.atoms) - {group.center}
    for root in n_roots:
        scope = frozenset(collect_subtree(mol, root, group.ligand))
        if not scope or skipped & scope or set(path) & scope:
            return None
        skipped.update(scope)

    from .complete_functional_parent import _render
    result = _render(mol, Chem, (group,), 'amide', path, False, False,
                     skipped, (), mapped_ring_prefixes=True, return_component=True,
                     branch_resolver=_branch)
    if not result:
        return None
    rendered, component = result
    name, _locants, _prefix_locs, mapping, prefixes, _component_name, _owners = rendered
    if (not name or component.atoms & component.embedded_atoms
            or not fully_covered(mol, component.atoms, component.embedded_atoms)):
        return None
    recorded, branch_atoms = Counter(), set()
    for root, host, atoms, text in component.prefix_branch_ownership:
        if (host not in component.atoms or root not in atoms or branch_atoms & atoms
                or frozenset(collect_subtree(mol, root, host)) != atoms):
            return None
        rebuilt = _branch(mol, Chem, root, host)
        if rebuilt is None or rebuilt != (text, atoms):
            return None
        recorded[(mapping[host], text)] += 1
        branch_atoms.update(atoms)
    if recorded != Counter(prefixes) or branch_atoms != set(component.embedded_atoms):
        return None
    issues = [SelectionLimitation(
        'principal-amide-parent', 'P-44/P-45/P-66.1.1',
        'The complete mapped amide construction is available, but competing parent and method preference remain unresolved.')]
    if any(item.specified != Chem.StereoSpecified.Specified
           for item in Chem.FindPotentialStereo(mol)):
        issues.append(SelectionLimitation(
            'stereochemistry', 'P-91.2.1.2.1/P-91.2.2',
            'One or more potential stereogenic units have no source configuration.'))
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return MethodDecision(name, 'substitutive', frozenset({group.center}), heavy,
        'P-41/P-44/P-45/P-66.1.1.1.1/P-66.1.1.3.1.1',
        tuple(mapping.items()), preference_issues=tuple(issues))
