"""Mapped general proposal for a bounded equal-rank nested-amide family.

The selected ring-carboxamide proposal is not asserted to win the complete
P-44/P-45 frontier. Its two nested amide prefixes are constructed from
original graph atoms, then joined to existing mapped ring constructions.
"""
from collections import Counter
from dataclasses import dataclass, replace

from .branch_rules import collect_subtree
from .component_mapping import MappedComponent, decorate_component, mapped_branch
from .constants import ROOTS
from .extended_inventory import selection_inventory
from .mapped_ring_parents import records, ring_systems
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


@dataclass(frozen=True, slots=True)
class _Branch:
    root: int
    blocked: int
    atoms: frozenset[int]
    text: str


def _scope(mol, root, blocked):
    atoms = frozenset(collect_subtree(mol, root, blocked))
    cuts = {(i, n.GetIdx()) for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms}
    return atoms if atoms and cuts == {(root, blocked)} else frozenset()


def _amide_group(mol, groups, nitrogen):
    choices = [g for g in groups if g.kind == 'amide' and g.ligand == nitrogen]
    if len(choices) != 1:
        return None
    group = choices[0]
    carbon = mol.GetAtomWithIdx(group.center)
    oxy = [a.GetIdx() for a in carbon.GetNeighbors() if a.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(group.center, a.GetIdx()).GetBondTypeAsDouble() == 2]
    if (carbon.IsInRing() or carbon.GetDegree() != 3 or len(oxy) != 1
            or mol.GetAtomWithIdx(oxy[0]).GetDegree() != 1
            or group.owner < 0 or mol.GetAtomWithIdx(group.owner).GetAtomicNum() != 6
            or set(group.atoms) != {group.center, oxy[0], nitrogen}):
        return None
    for neighbor in (group.owner, nitrogen):
        bond = mol.GetBondBetweenAtoms(group.center, neighbor)
        if bond is None or bond.GetBondTypeAsDouble() != 1:
            return None
    return group, oxy[0]


def _amide_parent(mol, Chem, path):
    if len(path) not in ROOTS or len(path) < 2:
        return ''
    from .parent_rules import _generic_acyl_functional_parent_text
    text = _generic_acyl_functional_parent_text(mol, Chem, ROOTS[len(path)], path[1:], 'amide')
    return text[:-1] + 'o' if text.endswith('amide') else ''


def _inner_amido(mol, Chem, root, blocked, groups):
    nitrogen = mol.GetAtomWithIdx(root)
    if (nitrogen.GetAtomicNum() != 7 or nitrogen.IsInRing()
            or mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble() != 1):
        return None
    site = _amide_group(mol, groups, root)
    if site is None:
        return None
    group, oxo = site
    children = [a.GetIdx() for a in nitrogen.GetNeighbors()
                if a.GetIdx() not in {blocked, group.center} and a.GetAtomicNum() > 1]
    if (len(children) > 1 or nitrogen.GetTotalValence() != 3
            or nitrogen.GetTotalNumHs(includeNeighbors=True) != 1 - len(children)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   or mol.GetBondBetweenAtoms(root, i).GetBondTypeAsDouble() != 1
                   for i in children)):
        return None
    # The acyl side is an entire unbranched carbon chain; potential E/Z
    # configurations are excluded by the whole-graph annotation gate.
    path = [group.center]
    previous, current = group.center, group.owner
    while True:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetIsAromatic():
            return None
        path.append(current)
        if len(path) > 12:
            return None
        following = [a.GetIdx() for a in atom.GetNeighbors()
                     if a.GetIdx() != previous and a.GetAtomicNum() > 1]
        if not following:
            break
        if len(following) != 1 or mol.GetAtomWithIdx(following[0]).GetAtomicNum() != 6:
            return None
        bond = mol.GetBondBetweenAtoms(current, following[0]).GetBondTypeAsDouble()
        if bond not in {1, 2, 3} or following[0] in path:
            return None
        previous, current = current, following[0]
    parent = _amide_parent(mol, Chem, tuple(path))
    if not parent:
        return None
    prefixes, embedded, ownership = [], set(), []
    if children:
        from .carbamate_methods import _carbon_organyl
        child = children[0]
        text, atoms = _carbon_organyl(mol, Chem, child, root)
        if (not text or atoms != _scope(mol, child, root)
                or any(mol.GetAtomWithIdx(i).IsInRing() for i in atoms)
                or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds()
                       if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= atoms)):
            return None
        prefixes.append(('N', text))
        embedded.update(atoms)
        ownership.append((child, root, atoms, text))
    mapping = tuple((i, str(j)) for j, i in enumerate(path, 1)) + ((oxo, ''), (root, 'N'))
    atoms = frozenset(dict(mapping))
    scope = _scope(mol, root, blocked)
    if atoms & embedded or atoms | embedded != scope:
        return None
    component = MappedComponent(parent, mapping, atoms, tuple(prefixes), frozenset(embedded),
        systematic_parent=True, nitrogen_site_locants=bool(prefixes),
        prefix_branch_ownership=tuple(ownership))
    text = decorate_component(mol, Chem, component)
    return (_Branch(root, blocked, scope, text), group) if text else None


def _outer_amido(mol, Chem, root, blocked, groups):
    nitrogen = mol.GetAtomWithIdx(root)
    if (nitrogen.GetAtomicNum() != 7 or nitrogen.IsInRing()
            or nitrogen.GetTotalNumHs(includeNeighbors=True) != 1
            or nitrogen.GetTotalValence() != 3):
        return None
    site = _amide_group(mol, groups, root)
    if site is None:
        return None
    group, oxo = site
    heavy = [a.GetIdx() for a in nitrogen.GetNeighbors() if a.GetAtomicNum() > 1]
    if set(heavy) != {blocked, group.center}:
        return None
    path = [group.center]
    previous, current = group.center, group.owner
    while True:
        atom = mol.GetAtomWithIdx(current)
        if (atom.GetAtomicNum() != 6 or atom.IsInRing()
                or atom.GetTotalNumHs(includeNeighbors=True) != 2
                or mol.GetBondBetweenAtoms(previous, current).GetBondTypeAsDouble() != 1):
            return None
        path.append(current)
        if len(path) > 8:
            return None
        children = [a.GetIdx() for a in atom.GetNeighbors()
                    if a.GetIdx() != previous and a.GetAtomicNum() > 1]
        if len(children) != 1:
            return None
        child = children[0]
        if mol.GetAtomWithIdx(child).GetAtomicNum() == 7:
            break
        previous, current = current, child
    inner = _inner_amido(mol, Chem, child, current, groups)
    if inner is None:
        return None
    branch, inner_group = inner
    if inner_group == group:
        return None
    parent = _amide_parent(mol, Chem, tuple(path))
    if not parent:
        return None
    mapping = tuple((i, str(j)) for j, i in enumerate(path, 1)) + ((oxo, ''), (root, 'N'))
    atoms = frozenset(dict(mapping))
    scope = _scope(mol, root, blocked)
    if atoms & branch.atoms or atoms | branch.atoms != scope:
        return None
    component = MappedComponent(parent, mapping, atoms, ((str(len(path)), branch.text),),
        branch.atoms, systematic_parent=True,
        prefix_branch_ownership=((branch.root, branch.blocked, branch.atoms, branch.text),))
    text = decorate_component(mol, Chem, component)
    return (_Branch(root, blocked, scope, text), frozenset({group.center, inner_group.center})) if text else None


def _carbon_linker(mol, Chem, root, blocked, groups):
    path, previous, current = [], blocked, root
    while True:
        atom = mol.GetAtomWithIdx(current)
        if (atom.GetAtomicNum() != 6 or atom.IsInRing()
                or atom.GetTotalNumHs(includeNeighbors=True) != 2
                or mol.GetBondBetweenAtoms(previous, current).GetBondTypeAsDouble() != 1):
            return None
        path.append(current)
        if len(path) > 8:
            return None
        children = [a.GetIdx() for a in atom.GetNeighbors()
                    if a.GetIdx() != previous and a.GetAtomicNum() > 1]
        if len(children) != 1:
            return None
        child = children[0]
        if mol.GetAtomWithIdx(child).GetAtomicNum() == 7:
            break
        previous, current = current, child
    outer = _outer_amido(mol, Chem, child, current, groups)
    if outer is None or len(path) not in ROOTS:
        return None
    branch, centers = outer
    atoms = frozenset(path)
    scope = _scope(mol, root, blocked)
    if atoms & branch.atoms or atoms | branch.atoms != scope:
        return None
    mapping = tuple((i, str(j)) for j, i in enumerate(path, 1))
    component = MappedComponent(ROOTS[len(path)] + 'yl', mapping, atoms,
        ((str(len(path)), branch.text),), branch.atoms, systematic_parent=True,
        omit_methyl_prefix_locant=len(path) == 1,
        prefix_branch_ownership=((branch.root, branch.blocked, branch.atoms, branch.text),))
    text = decorate_component(mol, Chem, component)
    return (_Branch(root, blocked, scope, text), centers) if text else None


def _replace_ring_branch(mol, Chem, record, expected_scope, replacement):
    """Rebuild a mapped ring component with one graph-proved branch."""
    component = record.construction
    if (not record.annotation_coverage or not record.hydrogen_budget_proved
            or record.preference_issues or component is None
            or component.atoms & component.embedded_atoms
            or component.atoms | component.embedded_atoms != expected_scope):
        return ''
    mapping = dict(component.atom_locants)
    parts, observed, match_count = set(), Counter(), 0
    ownership, prefixes = [], list(component.prefixes)
    for root, anchor, atoms, text in component.prefix_branch_ownership:
        atoms = frozenset(atoms)
        if (anchor not in component.atoms or root not in atoms or parts & atoms
                or _scope(mol, root, anchor) != atoms):
            return ''
        rebuilt, owned = mapped_branch(mol, Chem, root, anchor)
        if rebuilt != text or owned != atoms:
            return ''
        observed[(mapping[anchor], text)] += 1
        parts.update(atoms)
        if (root, anchor, atoms) == (replacement.root, replacement.blocked, replacement.atoms):
            match_count += 1
            old_pair = (mapping[anchor], text)
            if old_pair not in prefixes:
                return ''
            prefixes[prefixes.index(old_pair)] = (mapping[anchor], replacement.text)
            ownership.append((root, anchor, atoms, replacement.text))
        else:
            ownership.append((root, anchor, atoms, text))
    if (match_count != 1 or parts != set(component.embedded_atoms)
            or observed != Counter(component.prefixes)):
        return ''
    return decorate_component(mol, Chem, replace(component, prefixes=tuple(prefixes),
        prefix_branch_ownership=tuple(ownership)))


def _ring_bridge(mol, Chem, root, blocked, groups, selected_center):
    from .amine_components import ordinary_amine
    atom = mol.GetAtomWithIdx(root)
    if (not ordinary_amine(mol, root) or atom.GetTotalNumHs(includeNeighbors=True) != 1
            or mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble() != 1):
        return None
    children = [a.GetIdx() for a in atom.GetNeighbors()
                if a.GetIdx() != blocked and a.GetAtomicNum() > 1]
    if len(children) != 1 or not mol.GetAtomWithIdx(children[0]).IsInRing():
        return None
    ring_root = children[0]
    scope, ring_scope = _scope(mol, root, blocked), _scope(mol, ring_root, root)
    if not ring_scope or ring_scope | {root} != scope:
        return None
    candidates = records(mol, Chem, scope=ring_scope, attachment=ring_root)
    if len(candidates) != 1 or candidates[0].construction is None:
        return None
    record = candidates[0]
    branches = []
    for child, anchor, atoms, _text in record.construction.prefix_branch_ownership:
        if mol.GetAtomWithIdx(child).GetAtomicNum() != 6:
            continue
        result = _carbon_linker(mol, Chem, child, anchor, groups)
        if result is not None:
            branch, centers = result
            if branch.atoms == frozenset(atoms) and selected_center not in centers:
                branches.append((branch, centers))
    if len(branches) != 1:
        return None
    branch, centers = branches[0]
    ring_text = _replace_ring_branch(mol, Chem, record, ring_scope, branch)
    if not ring_text:
        return None
    component = MappedComponent('amino', ((root, 'N'),), frozenset({root}),
        (('', ring_text),), ring_scope, prefix_mode='amino',
        prefix_branch_ownership=((ring_root, root, ring_scope, ring_text),))
    text = decorate_component(mol, Chem, component)
    return (_Branch(root, blocked, scope, text), centers) if text else None


def nested_amide_general_decision(mol, Chem):
    """Deliver a completely mapped candidate without claiming a PIN winner."""
    if (mol.GetNumHeavyAtoms() > 100 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups() or Chem.FindPotentialStereo(mol)
            or any(a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
                   for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    inventory = selection_inventory(mol)
    frontier = getattr(inventory, 'source_frontier', None)
    groups = tuple(inventory.principal)
    if (not inventory.supported or frontier is None or frontier.status != 'tie'
            or frontier.unknown_pairs or frontier.cycle or len(groups) != 3
            or any(g.kind != 'amide' for g in groups)):
        return None
    systems = ring_systems(mol)
    attached = [(g, ring) for g in groups for ring in systems if g.owner in ring]
    if len(attached) != 1:
        return None
    selected, ring = attached[0]
    n = mol.GetAtomWithIdx(selected.ligand)
    if n.GetDegree() != 1 or n.GetTotalNumHs(includeNeighbors=True) != 2:
        return None
    candidates = records(mol, Chem, principal=(selected,), core_filter=ring)
    if len(candidates) != 1 or candidates[0].construction is None:
        return None
    record = candidates[0]
    whole = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if record.principal_centers != frozenset({selected.center}):
        return None
    branches = []
    for root, anchor, atoms, _text in record.construction.prefix_branch_ownership:
        if mol.GetAtomWithIdx(root).GetAtomicNum() != 7:
            continue
        result = _ring_bridge(mol, Chem, root, anchor, groups, selected.center)
        if result is not None:
            branch, centers = result
            if branch.atoms == frozenset(atoms) and centers == {g.center for g in groups if g != selected}:
                branches.append(branch)
    if len(branches) != 1:
        return None
    name = _replace_ring_branch(mol, Chem, record, whole, branches[0])
    if not name:
        return None
    component = record.construction
    roles = (('ring-carboxamide-parent', component.atoms, component.atom_locants),
             ('mapped-parent-substituents', component.embedded_atoms, ()))
    issues = (SelectionLimitation(
        'equal-rank-amide-parent', 'P-41/P-44/P-45/P-66.1.1.4.3',
        'The ring carboxamide and nested amide prefixes are completely mapped; equal-rank parent and method preference remains unresolved.'),)
    return MethodDecision(name, 'substitutive', frozenset({selected.center}), whole,
        'P-41/P-44/P-45/P-66.1.1.1.1.3/P-66.1.1.4.3', component.atom_locants,
        component_roles=roles, preference_issues=issues)
