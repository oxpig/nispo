"""A bounded, original-graph branched-chain sulfinamide continuation.

Every acyclic carbon path through the sulfur-bearing carbon is enumerated.
Only maximum-length, lowest-suffix-locant paths that all render the same name
and are related by original-graph automorphisms are admitted. N-bound arms
retain their own complete atom and stereochemistry witnesses.
"""
from collections import Counter
from dataclasses import replace

from .branch_rules import collect_subtree, parent_chain_name
from .component_mapping import (
    MappedComponent, decorate_component, fully_covered, mapped_branch, mapped_general_branch,
)
from .constants import ROOTS
from .extended_inventory import selection_inventory
from .formatting import complete_parent_name
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation
from .sulfur_selection import sulfur_suffix_locants


def _carbon_tree_paths(mol, carbon, owner):
    """Enumerate all paths through owner in one bounded saturated carbon tree."""
    if not carbon or owner not in carbon or len(carbon) > 24:
        return ()
    adjacency = {i: set() for i in carbon}
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in carbon and right in carbon:
            if bond.GetBondTypeAsDouble() != 1:
                return ()
            adjacency[left].add(right)
            adjacency[right].add(left)
    if any(mol.GetAtomWithIdx(i).IsInRing() or mol.GetAtomWithIdx(i).GetIsAromatic()
           or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in carbon):
        return ()
    if sum(map(len, adjacency.values())) != 2 * (len(carbon) - 1):
        return ()
    seen, stack = set(), [owner]
    while stack:
        i = stack.pop()
        if i not in seen:
            seen.add(i)
            stack.extend(adjacency[i] - seen)
    if seen != carbon:
        return ()
    paths = set()
    for start in carbon:
        stack = [(start, (start,))]
        while stack:
            current, path = stack.pop()
            if owner in path:
                paths.add(path)
            stack.extend((n, path + (n,)) for n in adjacency[current] if n not in path)
    if not paths:
        return ()
    longest = max(map(len, paths))
    maxima = [path for path in paths if len(path) == longest]
    lowest_suffix = min(path.index(owner) + 1 for path in maxima)
    maxima = tuple(path for path in maxima if path.index(owner) + 1 == lowest_suffix)
    return maxima if len(maxima) <= 128 else ()


def _candidate(mol, Chem, site, path, arms, framework, sulfur_descriptor):
    parent = complete_parent_name(parent_chain_name(mol, path))
    if not parent or len(path) not in ROOTS:
        return None
    mapping = {idx: str(pos) for pos, idx in enumerate(path, 1)}
    mapping[site.ligand] = 'N'
    core = frozenset(site.atoms)
    prefixes, side_atoms, ownership, natures = [], set(), [], {}
    from .organyl_prefix_components import organyl_prefix_nature
    for arm in arms:
        nature = organyl_prefix_nature(mol, arm.root, arm.atoms, arm.text)
        if nature is None and arm.text and not arm.text.isalpha():
            # The branch resolver already closed the rooted scope. The
            # modified organyl word takes its own enclosing marks at N.
            nature = 'compound'
        if nature not in {'simple', 'compound'}:
            return None
        if arm.text in natures and natures[arm.text] != nature:
            return None
        natures[arm.text] = nature
        prefixes.append(('N', arm.text))
        ownership.append((arm.root, site.ligand, arm.atoms, arm.text))
    for anchor in path:
        for neighbor in mol.GetAtomWithIdx(anchor).GetNeighbors():
            root = neighbor.GetIdx()
            if root not in framework or root in path:
                continue
            atoms = frozenset(collect_subtree(mol, root, anchor))
            cuts = {(i, n.GetIdx()) for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()
                    if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms}
            if (not atoms or cuts != {(root, anchor)} or not atoms <= framework
                    or atoms & side_atoms or atoms & set(path)):
                return None
            text, owned = mapped_branch(mol, Chem, root, anchor)
            if not text or owned != atoms:
                return None
            nature = organyl_prefix_nature(mol, root, atoms, text)
            if nature is None and mol.GetAtomWithIdx(root).IsInRing():
                from .component_mapping import mapped_branch_ring_atoms

                rings = {i for i in owned if mol.GetAtomWithIdx(i).IsInRing()}
                if rings and rings <= mapped_branch_ring_atoms(mol, Chem, root, anchor, text, owned):
                    nature = 'compound'
            if nature not in {'simple', 'compound'}:
                return None
            if text in natures and natures[text] != nature:
                return None
            natures[text] = nature
            side_atoms.update(atoms)
            prefixes.append((mapping[anchor], text))
            ownership.append((root, anchor, atoms, text))
    if set(path) | side_atoms != framework:
        return None
    n_atoms = frozenset().union(*(arm.atoms for arm in arms)) if arms else frozenset()
    embedded = frozenset(side_atoms) | n_atoms | {site.center, site.oxo_oxygen}
    if embedded & set(mapping) or not fully_covered(mol, mapping, embedded):
        return None
    locants = sulfur_suffix_locants(
        count=1, locants=(int(mapping[site.owner]),), cyclic=False,
        path_length=len(path), prefix_count=len(prefixes),
        parent_prefix_count=len(prefixes) - len(arms))
    component = MappedComponent(
        parent + locants + 'sulfinamide', tuple(mapping.items()), frozenset(mapping),
        tuple(prefixes), embedded, systematic_parent=True,
        nitrogen_site_locants=bool(arms), prefix_natures=tuple(natures.items()),
        prefix_branch_ownership=tuple(ownership))
    name = decorate_component(mol, Chem, component)
    if not name:
        # Closed general branch words can fall outside the typed prefix
        # grammar, for instance when an unconfigured complex arm is long.
        name = decorate_component(mol, Chem, replace(component, prefix_natures=None))
    if not name:
        return None
    if sulfur_descriptor:
        name = f'({sulfur_descriptor})-' + name
    observed = Counter((mapping[anchor], text) for _root, anchor, _atoms, text in ownership)
    if observed != Counter(prefixes):
        return None
    roles = (('sulfinamide-parent', frozenset(path), tuple((i, mapping[i]) for i in path)),
             ('sulfinamide-suffix-unit', core, ((site.ligand, 'N'),)),
             ('parent-substituents', frozenset(side_atoms), ()),
             ('N-substituents', n_atoms, ()))
    return name, component, roles


def _equivalent_paths(mol, Chem, paths, site, arms):
    """Require atom- and stereo-preserving equivalence of every tied path."""
    if len(paths) <= 1:
        return True
    limit = 4096
    matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=True, maxMatches=limit)
    if not matches or len(matches) >= limit:
        return False
    fixed = {site.center, site.oxo_oxygen, site.ligand, site.owner,
             *(arm.root for arm in arms)}

    def preserves(permutation):
        if any(permutation[i] != i for i in fixed):
            return False
        if any(frozenset(permutation[i] for i in arm.atoms) != arm.atoms for arm in arms):
            return False
        for atom in mol.GetAtoms():
            other = mol.GetAtomWithIdx(permutation[atom.GetIdx()])
            if (atom.GetAtomicNum(), atom.GetIsAromatic(), atom.GetFormalCharge(),
                    atom.GetIsotope(), atom.GetTotalNumHs(includeNeighbors=True)) != (
                    other.GetAtomicNum(), other.GetIsAromatic(), other.GetFormalCharge(),
                    other.GetIsotope(), other.GetTotalNumHs(includeNeighbors=True)):
                return False
        for bond in mol.GetBonds():
            other = mol.GetBondBetweenAtoms(permutation[bond.GetBeginAtomIdx()],
                                             permutation[bond.GetEndAtomIdx()])
            if other is None or other.GetBondType() != bond.GetBondType():
                return False
        return True

    permutations = tuple(match for match in matches if preserves(match))
    return bool(permutations) and all(any(tuple(p[i] for i in paths[0]) == path
                                          for p in permutations) for path in paths)


def branched_sulfinamide_decision(mol, Chem):
    """Return a mapped branched carbon-tree sulfinamide, or decline."""
    from .sulfinamide_components import recognize_sulfinamides
    from .trisubstituted_urea import _Arm, _arm

    if (mol.GetNumHeavyAtoms() > 100 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()
            or any(a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    recognition = recognize_sulfinamides(mol, Chem)
    if recognition.status != 'supported' or len(recognition.sites) != 1:
        return None
    site = recognition.sites[0]
    if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in site.atoms if i != site.center):
        return None
    sulfur_descriptor = ''
    if mol.GetAtomWithIdx(site.center).GetChiralTag():
        from .stereochemistry import assigned_atom_stereochemistry

        sulfur_descriptor = dict(assigned_atom_stereochemistry(mol, Chem)).get(site.center)
        if sulfur_descriptor not in {'R', 'S'}:
            return None
    inventory = selection_inventory(mol)
    frontier = getattr(inventory, 'source_frontier', None)
    if inventory.supported and (frontier is None or frontier.status != 'selected'
            or frontier.unknown_pairs or frontier.cycle or len(frontier.groups) != 1
            or frontier.groups[0].kind != 'sulfinamide'
            or frontier.groups[0].center != site.center
            or frontier.groups[0].owner != site.owner
            or frontier.groups[0].ligand != site.ligand
            or frozenset(frontier.groups[0].atoms) != frozenset(site.atoms)):
        return None
    arms, n_atoms, stereo_atoms = [], set(), set()
    for root, expected in zip(site.n_substituents, site.n_substituent_atoms):
        # This existing reusable carbon-arm producer proves complete ring and
        # tetrahedral atom ownership; its result is independent of urea core.
        arm = _arm(mol, Chem, root, site.ligand)
        if arm is None:
            text, atoms = mapped_general_branch(mol, Chem, root, site.ligand)
            if text and atoms == expected:
                tagged = frozenset(i for i in atoms if mol.GetAtomWithIdx(i).GetChiralTag())
                from .component_mapping import _general_branch_stereo_is_cited

                if not tagged or _general_branch_stereo_is_cited(mol, Chem, atoms, text):
                    arm = _Arm(root, site.ligand, atoms, text, tagged)
        if arm is None or arm.atoms != expected or n_atoms & arm.atoms:
            return None
        arms.append(arm)
        n_atoms.update(arm.atoms)
        stereo_atoms.update(arm.stereo_atoms)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    framework = heavy - n_atoms - set(site.atoms)
    carbon, pending = set(), [site.owner]
    while pending:
        i = pending.pop()
        atom = mol.GetAtomWithIdx(i)
        if (i in carbon or i not in framework or atom.GetAtomicNum() != 6 or atom.IsInRing()):
            continue
        carbon.add(i)
        pending.extend(n.GetIdx() for n in atom.GetNeighbors())
    if not carbon:
        return None
    potential = tuple(Chem.FindPotentialStereo(mol))
    if sulfur_descriptor:
        stereo_atoms.add(site.center)
    if (any(item.type != Chem.StereoType.Atom_Tetrahedral for item in potential)
            or any(item.specified == Chem.StereoSpecified.Specified
                   and item.centeredOn not in stereo_atoms for item in potential)
            or {a.GetIdx() for a in mol.GetAtoms() if int(a.GetChiralTag())} != stereo_atoms):
        return None
    paths = _carbon_tree_paths(mol, carbon, site.owner)
    if not paths or (not arms and not any(set(path) != framework for path in paths)):
        return None
    if framework != carbon:
        # Equal-length paths through a cyclic arm's carbon linker can expose
        # more substituents than an alternative ending at two methyl groups.
        def numbering(path):
            positions = tuple(j for j, idx in enumerate(path, 1)
                              for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                              if n.GetIdx() in framework and n.GetIdx() not in path)
            return (-len(positions), positions)

        lowest = min(map(numbering, paths))
        paths = tuple(path for path in paths if numbering(path) == lowest)
    candidates = [_candidate(mol, Chem, site, path, arms, framework, sulfur_descriptor)
                  for path in paths]
    if any(candidate is None for candidate in candidates):
        return None
    if len({candidate[0] for candidate in candidates}) != 1:
        return None
    if not _equivalent_paths(mol, Chem, paths, site, arms):
        return None
    name, component, roles = candidates[0]
    role_atoms = [row[1] for row in roles]
    if (any(role_atoms[i] & role_atoms[j] for i in range(len(role_atoms))
            for j in range(i + 1, len(role_atoms)))
            or frozenset().union(*role_atoms) != heavy):
        return None
    issues = [SelectionLimitation(
        'sulfinamide-parent', 'P-44/P-45/P-66.1.1.2',
        'All bounded branched-chain paths are mapped; broader parent and method preference remains unresolved.')]
    if any(item.specified != Chem.StereoSpecified.Specified for item in potential):
        issues.append(SelectionLimitation(
            'stereochemistry', 'P-91.2.1.2.1/P-91.2.2',
            'One or more potential stereogenic units have no source configuration.'))
    return MethodDecision(name, 'substitutive', frozenset({site.center}), heavy,
        'P-41/P-42.1/P-44.3/P-45/P-66.1.1.2/P-66.1.1.3.1.1',
        component.atom_locants, component_roles=roles, preference_issues=tuple(issues))
