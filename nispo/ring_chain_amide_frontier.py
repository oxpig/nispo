"""Two mapped suffix-site candidates for a bounded ring/chain amide family.

Both source-tied amide sites must have complete, disjoint original-graph
constructions. P-44.1.2.2 favors the same-class ring parent over the chain;
the larger method/parent universe is deliberately left unresolved.
"""
from collections import Counter
from dataclasses import dataclass, replace

from .component_mapping import (MappedComponent, _record_branch_ring_atoms,
    decorate_component, has_annotations, mapped_branch, mapped_branch_ring_atoms)
from .constants import ROOTS
from .extended_inventory import selection_inventory
from .mapped_ring_parents import records, ring_systems
from .n_sulfonyl_ring_amides import _proved_record, _scope
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation
from .nested_amide_general import _Branch, _replace_ring_branch


@dataclass(frozen=True, slots=True)
class RingChainAmideFrontier:
    ring: MethodDecision
    chain: MethodDecision
    rule: str = 'P-44.1.2.2'


def _secondary_amide(mol, group):
    carbon, nitrogen = mol.GetAtomWithIdx(group.center), mol.GetAtomWithIdx(group.ligand)
    if (carbon.GetAtomicNum() != 6 or carbon.IsInRing() or carbon.GetDegree() != 3
            or nitrogen.GetAtomicNum() != 7 or nitrogen.IsInRing()
            or nitrogen.GetTotalValence() != 3 or nitrogen.GetTotalNumHs(includeNeighbors=True) != 1):
        return None
    oxy = [a.GetIdx() for a in carbon.GetNeighbors() if a.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(group.center, a.GetIdx()).GetBondTypeAsDouble() == 2]
    other = [a.GetIdx() for a in nitrogen.GetNeighbors()
             if a.GetIdx() != group.center and a.GetAtomicNum() > 1]
    if (len(oxy) != 1 or mol.GetAtomWithIdx(oxy[0]).GetDegree() != 1
            or set(group.atoms) != {group.center, oxy[0], group.ligand}
            or len(other) != 1 or mol.GetAtomWithIdx(other[0]).GetAtomicNum() != 6
            or mol.GetBondBetweenAtoms(group.center, group.ligand).GetBondTypeAsDouble() != 1
            or mol.GetBondBetweenAtoms(group.center, group.owner) is None
            or mol.GetBondBetweenAtoms(group.center, group.owner).GetBondTypeAsDouble() != 1
            or mol.GetBondBetweenAtoms(group.ligand, other[0]).GetBondTypeAsDouble() != 1):
        return None
    return oxy[0], other[0]


def _chain(mol, Chem, group, oxo, blocked):
    path, previous, current = [group.center], group.center, group.owner
    while True:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetIsAromatic() or current in path:
            return None
        path.append(current)
        if len(path) > 12:
            return None
        carbon_children = [n.GetIdx() for n in atom.GetNeighbors()
                           if n.GetIdx() != previous and n.GetAtomicNum() == 6 and not n.IsInRing()]
        if len(carbon_children) > 1:
            return None
        if not carbon_children:
            break
        previous, current = current, carbon_children[0]
    if len(path) not in ROOTS or len(path) < 2:
        return None
    potential = Chem.FindPotentialStereo(mol)
    stereo_ends = set()
    for item in potential:
        if item.type != Chem.StereoType.Bond_Double or item.specified != Chem.StereoSpecified.Specified:
            return None
        bond = mol.GetBondWithIdx(item.centeredOn)
        ends = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
        if not ends <= set(path) or bond.GetStereo() not in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}:
            return None
        stereo_ends.update(ends)
    if any(int(bond.GetBondDir()) and not ({bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & stereo_ends)
           for bond in mol.GetBonds()):
        return None
    from .parent_rules import _generic_acyl_functional_parent_text
    parent = _generic_acyl_functional_parent_text(mol, Chem, ROOTS[len(path)], path[1:], 'amide')
    if not parent.endswith('amide'):
        return None
    prefixes, embedded, ownership = [], set(), []
    for position, idx in enumerate(path, 1):
        allowed = set(path) | {group.ligand, oxo}
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            root = neighbor.GetIdx()
            if root in allowed or neighbor.GetAtomicNum() == 1:
                continue
            atoms = _scope(mol, root, idx)
            text, owned = mapped_branch(mol, Chem, root, idx)
            if (not text or not atoms or owned != atoms or embedded & atoms
                    or has_annotations(mol, atoms)):
                return None
            rings = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()}
            if not rings <= mapped_branch_ring_atoms(mol, Chem, root, idx, text, atoms):
                return None
            prefixes.append((str(position), text))
            embedded.update(atoms)
            ownership.append((root, idx, atoms, text))
    mapping = tuple((idx, str(position)) for position, idx in enumerate(path, 1))
    mapping += ((oxo, 'O'), (group.ligand, 'N'))
    atoms = frozenset(dict(mapping))
    scope = _scope(mol, group.ligand, blocked)
    if atoms & embedded or atoms | embedded != scope:
        return None
    component = MappedComponent(parent[:-1] + 'o', mapping, atoms, tuple(prefixes),
        frozenset(embedded), systematic_parent=True, prefix_branch_ownership=tuple(ownership))
    name = decorate_component(mol, Chem, component)
    return (tuple(path), component, name, scope) if name else None


def ring_chain_amide_frontier(mol, Chem):
    """Map both suffix sites; do not assert complete method preference."""
    if (mol.GetNumHeavyAtoms() > 100 or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) for a in mol.GetAtoms())):
        return None
    inventory = selection_inventory(mol)
    source = inventory.source_frontier
    groups = tuple(source.groups)
    if (not inventory.supported or source.status != 'tie' or source.unknown_pairs or source.cycle
            or len(groups) != 2 or any(g.kind != 'amide' for g in groups)):
        return None
    ring_groups = [g for g in groups if mol.GetAtomWithIdx(g.owner).IsInRing()]
    if len(ring_groups) != 1:
        return None
    ring_group = ring_groups[0]
    chain_group = next(g for g in groups if g != ring_group)
    ring_site = _secondary_amide(mol, ring_group)
    chain_site = _secondary_amide(mol, chain_group)
    if ring_site is None or chain_site is None:
        return None
    _ring_oxo, ring_attachment = ring_site
    chain_oxo, chain_attachment = chain_site
    cores = [core for core in ring_systems(mol) if ring_attachment in core and chain_attachment in core]
    if len(cores) != 1:
        return None
    core = cores[0]
    core_bonds = [b for b in mol.GetBonds()
                  if b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core]
    if (len(core) != 6 or len(core_bonds) != 6
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in core)
            or any(not b.GetIsAromatic() for b in core_bonds)):
        return None
    # The branch added below must have the same locant in every orientation
    # of the previously numbered ring. Opposite attachment sites in a
    # six-membered monocycle are both 1,4; other arrangements need a joint
    # numbering frontier that includes the as-yet-unrendered amido branch.
    distances = {ring_attachment: 0}
    queue = [ring_attachment]
    for idx in queue:
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            child = neighbor.GetIdx()
            if child in core and child not in distances:
                distances[child] = distances[idx] + 1
                queue.append(child)
    if distances.get(chain_attachment) != 3:
        return None
    built = _chain(mol, Chem, chain_group, chain_oxo, chain_attachment)
    if built is None:
        return None
    path, amido_component, amido_name, amido_scope = built
    central_scope = _scope(mol, ring_attachment, ring_group.ligand)
    parent_scope = _scope(mol, ring_group.center, ring_group.ligand) | {ring_group.ligand}
    whole = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if (not central_scope or not parent_scope or parent_scope & central_scope
            or parent_scope | central_scope != whole or not amido_scope <= central_scope):
        return None
    parents = records(mol, Chem, scope=parent_scope, principal=(ring_group,))
    central_base = central_scope - amido_scope
    central = records(mol, Chem, scope=central_base, attachment=ring_attachment, core_filter=core)
    if (len(parents) != 1 or len(central) != 1
            or not _proved_record(mol, Chem, parents[0], parent_scope)
            or not _proved_record(mol, Chem, central[0], central_base)):
        return None
    ring_component = central[0].construction
    locant = dict(ring_component.atom_locants).get(chain_attachment)
    if not locant or ring_component.atoms & amido_scope or ring_component.embedded_atoms & amido_scope:
        return None
    decorated_ring = replace(ring_component,
        prefixes=ring_component.prefixes + ((locant, amido_name),),
        embedded_atoms=ring_component.embedded_atoms | amido_scope,
        prefix_branch_ownership=ring_component.prefix_branch_ownership +
            ((chain_group.ligand, chain_attachment, amido_scope, amido_name),))
    phenyl = decorate_component(mol, Chem, decorated_ring)
    if not phenyl or decorated_ring.atoms | decorated_ring.embedded_atoms != central_scope:
        return None
    parent = parents[0].construction
    if parents[0].principal_centers != {ring_group.center} or dict(parent.atom_locants).get(ring_group.ligand) != 'N':
        return None
    ring_name = decorate_component(mol, Chem, replace(parent,
        prefixes=parent.prefixes + (('N', phenyl),), embedded_atoms=parent.embedded_atoms | central_scope,
        prefix_branch_ownership=parent.prefix_branch_ownership +
            ((ring_attachment, ring_group.ligand, central_scope, phenyl),)))
    if not ring_name:
        return None

    # Build the second suffix-site candidate independently, changing the
    # proved ring-amide component into its P-66.1.1.4.3 amido prefix.
    if not parent.parent.endswith('amide'):
        return None
    ring_amido = decorate_component(mol, Chem, replace(parent, parent=parent.parent[:-1] + 'o'))
    reverse_scope = _scope(mol, chain_attachment, chain_group.ligand)
    reverse_records = records(mol, Chem, scope=reverse_scope, attachment=chain_attachment, core_filter=core)
    if len(reverse_records) != 1:
        return None
    replacement = _Branch(ring_group.ligand, ring_attachment, parent_scope, ring_amido)
    reverse_phenyl = _replace_ring_branch(mol, Chem, reverse_records[0], reverse_scope, replacement)
    if not reverse_phenyl:
        return None
    ring_owned = frozenset(i for i in reverse_scope if mol.GetAtomWithIdx(i).IsInRing())
    _record_branch_ring_atoms(mol, Chem, chain_attachment, chain_group.ligand,
                              reverse_phenyl, reverse_scope, ring_owned)
    def resolve(m, C, root, blocked):
        return (reverse_phenyl, reverse_scope) if (root, blocked) == (
            chain_attachment, chain_group.ligand) else None
    from .complete_functional_parent import _render
    skipped = (set(chain_group.atoms) - {chain_group.center}) | set(reverse_scope)
    result = _render(mol, Chem, (chain_group,), 'amide', path, False, False,
                     skipped, (), mapped_ring_prefixes=True, return_component=True,
                     branch_resolver=resolve)
    if not result:
        return None
    rendered, component = result
    if component.atoms & component.embedded_atoms or component.atoms | component.embedded_atoms != whole:
        return None
    mapping = dict(component.atom_locants)
    observed, branch_atoms = Counter(), set()
    for root, host, atoms, text in component.prefix_branch_ownership:
        if host not in mapping or branch_atoms & atoms or _scope(mol, root, host) != atoms:
            return None
        expected = ((reverse_phenyl, reverse_scope) if (root, host) ==
                    (chain_attachment, chain_group.ligand) else mapped_branch(mol, Chem, root, host))
        if expected != (text, atoms):
            return None
        observed[(mapping[host], text)] += 1
        branch_atoms.update(atoms)
    if observed != Counter(component.prefixes) or branch_atoms != set(component.embedded_atoms):
        return None
    issue = (SelectionLimitation('equal-rank-amide-method', 'P-44.1.2.2/P-45/P-51.3',
        'Ring and chain suffix-site candidates are mapped; the complete method and parent universe remains unresolved.'),)
    ring_decision = MethodDecision(ring_name, 'substitutive', frozenset({ring_group.center}), whole,
        'P-44.1.2.2/P-66.1.1.4.3', parent.atom_locants,
        component_roles=(('ring-amide-parent', parent_scope, parent.atom_locants),
                         ('N-aryl-amido', central_scope, ())), preference_issues=issue)
    chain_decision = MethodDecision(rendered[0], 'substitutive', frozenset({chain_group.center}), whole,
        'P-44.1.2.2/P-66.1.1.4.3', component.atom_locants,
        component_roles=(('chain-amide-parent', component.atoms, component.atom_locants),
                         ('chain-substituents', component.embedded_atoms, ())), preference_issues=issue)
    return RingChainAmideFrontier(ring_decision, chain_decision)
