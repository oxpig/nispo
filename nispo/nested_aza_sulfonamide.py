"""Mapped P-29/P-66 join for a fused aza group nested in an N-aryl branch.

The fused group is named as an original-graph subtree by the existing bounded
P-25 producer. Its mapped prefix is composed with a numbered phenyl unit and
the existing mapped sulfonamide construction, carrying the nested selection
limitations to the final method decision. No source graph is modified.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class NestedPrefix:
    text: str
    atoms: frozenset[int]
    root: int
    blocked: int
    core: frozenset[int]
    atom_locants: tuple
    branches: frozenset[int]
    preference_issues: tuple


def _cut(mol, root, blocked):
    """One single, directed cut and an exclusively owned original subtree."""
    from .branch_rules import collect_subtree

    bond = mol.GetBondBetweenAtoms(root, blocked)
    if bond is None or bond.GetBondTypeAsDouble() != 1:
        return None
    atoms = frozenset(collect_subtree(mol, root, blocked))
    if root not in atoms or blocked in atoms:
        return None
    boundary = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
                if (b.GetBeginAtomIdx() in atoms) != (b.GetEndAtomIdx() in atoms)}
    if len(boundary) != 1 or frozenset(next(iter(boundary))) != frozenset((root, blocked)):
        return None
    return atoms


def _fusion_prefix(mol, Chem, root, blocked):
    from .fusion_networks import matches_skeleton
    from .mapped_ring_parents import capped_subtree_fragment, records, ring_systems
    from .three_face_aza_fusion import METHOD

    atom = mol.GetAtomWithIdx(root)
    if atom.GetAtomicNum() != 6 or not atom.GetIsAromatic():
        return None
    core = next((system for system in ring_systems(mol) if root in system), None)
    if (core is None or len(core) != 13 or sorted(mol.GetAtomWithIdx(i).GetAtomicNum()
                                                  for i in core) != [6] * 9 + [7] * 4):
        return None
    atoms = _cut(mol, root, blocked)
    if atoms is None or not core <= atoms:
        return None
    copied = capped_subtree_fragment(mol, Chem, atoms)
    if copied is None:
        return None
    fragment, local = copied
    inverse = {value: key for key, value in local.items()}
    if len(local) != len(atoms) or set(inverse) != set(range(fragment.GetNumAtoms())):
        return None
    # The copied fragment may gain exactly one capping H at the cut carbon.
    # Its remaining atom properties and all induced bonds must be unchanged.
    for original, index in local.items():
        source, copy = mol.GetAtomWithIdx(original), fragment.GetAtomWithIdx(index)
        if (source.GetAtomicNum() != copy.GetAtomicNum()
                or source.GetIsAromatic() != copy.GetIsAromatic()
                or copy.GetTotalNumHs(includeNeighbors=True)
                != source.GetTotalNumHs(includeNeighbors=True) + (original == root)):
            return None
    actual_bonds = {(frozenset((local[b.GetBeginAtomIdx()], local[b.GetEndAtomIdx()])),
                     b.GetBondTypeAsDouble(), b.GetIsAromatic()) for b in mol.GetBonds()
                    if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms}
    copied_bonds = {(frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())),
                     b.GetBondTypeAsDouble(), b.GetIsAromatic()) for b in fragment.GetBonds()}
    if actual_bonds != copied_bonds:
        return None
    candidates = records(fragment, Chem, scope=range(fragment.GetNumAtoms()),
                         attachment=local[root])
    if len(candidates) != 1:
        return None
    record = candidates[0]
    construction = record.construction
    if (record.method != METHOD or construction is None or not record.name
            or not record.preference_issues or not record.hydrogen_budget_proved
            or not record.annotation_coverage or record.fusion_construction is None
            or record.principal_centers != frozenset({local[root]})):
        return None
    mapped = frozenset(dict(record.atom_locants))
    from .fused_numbering_witness import numbering_matches
    original_sequence = tuple(inverse[i] for i in record.parent_atoms)
    local_locants = dict(record.atom_locants)
    if (mapped != construction.atoms or mapped & construction.embedded_atoms
            or mapped | construction.embedded_atoms != frozenset(range(fragment.GetNumAtoms()))
            or frozenset(original_sequence) != core
            or not matches_skeleton(record.fusion_construction, mol, core)
            or not numbering_matches(mol, original_sequence,
                                     tuple(local_locants[i] for i in record.parent_atoms), record.parent)):
        return None
    # A second cyclic branch must carry the mapped ring witness supplied by
    # its own rooted producer, rather than compatibility spelling alone.
    from .component_mapping import mapped_branch_ring_atoms
    for child, host, owned, text in construction.prefix_branch_ownership:
        physical_rings = frozenset(i for i in owned if fragment.GetAtomWithIdx(i).IsInRing())
        proof = mapped_branch_ring_atoms(fragment, Chem, child, host, text, owned)
        if physical_rings != proof:
            return None
    return NestedPrefix(record.name, atoms, root, blocked, core,
        tuple((inverse[i], locant) for i, locant in record.atom_locants),
        frozenset(inverse[i] for i in construction.embedded_atoms), record.preference_issues)


def _simple_prefix(mol, Chem, root, blocked):
    """Prove an ordinary halogen or unbranched saturated carbon branch."""
    from .component_mapping import mapped_branch

    atoms = _cut(mol, root, blocked)
    if atoms is None:
        return None
    root_atom = mol.GetAtomWithIdx(root)
    if root_atom.GetAtomicNum() in {9, 17, 35, 53}:
        if len(atoms) != 1 or root_atom.GetTotalValence() != 1:
            return None
    elif all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
             and not mol.GetAtomWithIdx(i).IsInRing()
             and not mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
        if len(atoms) > 8:
            return None
        degrees = {i: sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors())
                   for i in atoms}
        if (degrees[root] > 1 or any(value > 2 for value in degrees.values())
                or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds()
                       if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms)):
            return None
    else:
        return None
    text, owned = mapped_branch(mol, Chem, root, blocked)
    return (text, atoms) if text and owned == atoms else None


def _phenyl_prefix(mol, Chem, root, blocked):
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings
    from .formatting import prefix_alphabetization_key
    from .mapped_ring_parents import ring_systems
    from .parent_selection import locant_key

    atoms = _cut(mol, root, blocked)
    core = next((system for system in ring_systems(mol) if root in system), None)
    if (atoms is None or core is None or len(core) != 6 or not core <= atoms
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in core)):
        return None
    neighbors = {i: tuple(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                          if n.GetIdx() in core) for i in core}
    if any(len(values) != 2 for values in neighbors.values()):
        return None
    branches, fusion = [], []
    seen = set(core)
    for host in core:
        for neighbor in mol.GetAtomWithIdx(host).GetNeighbors():
            child = neighbor.GetIdx()
            if child in core or child == blocked:
                continue
            special = _fusion_prefix(mol, Chem, child, host)
            if special is not None:
                text, owned = special.text, special.atoms
                fusion.append(special)
            else:
                simple = _simple_prefix(mol, Chem, child, host)
                if simple is None:
                    return None
                text, owned = simple
            if owned & seen or not owned <= atoms:
                return None
            seen.update(owned)
            branches.append((host, child, text, owned))
    if seen != set(atoms) or len(fusion) != 1:
        return None

    rows = []
    for first in neighbors[root]:
        sequence = [root, first]
        while len(sequence) < 6:
            remaining = set(neighbors[sequence[-1]]) - {sequence[-2]}
            if len(remaining) != 1:
                return None
            nxt = remaining.pop()
            if nxt in sequence:
                return None
            sequence.append(nxt)
        if root not in neighbors[sequence[-1]]:
            return None
        mapping = {idx: str(i + 1) for i, idx in enumerate(sequence)}
        prefixes = tuple((mapping[host], text) for host, _child, text, _owned in branches)
        embedded = frozenset().union(*(owned for _, _, _, owned in branches))
        bindings = hydrogen_bindings(mol, mapping)
        construction = MappedComponent('phenyl', tuple(bindings.items()), frozenset(bindings),
            prefixes, embedded_atoms=embedded, systematic_parent=True,
            prefix_branch_ownership=tuple((child, host, owned, text)
                                          for host, child, text, owned in branches))
        text = decorate_component(mol, Chem, construction)
        if not text or frozenset(bindings) & embedded or frozenset(bindings) | embedded != atoms:
            return None
        key = (tuple(sorted(locant_key(loc) for loc, _ in prefixes)),
               tuple(locant_key(loc) for loc, _ in sorted(prefixes,
                     key=lambda pair: (prefix_alphabetization_key(pair[1]), locant_key(pair[0])))))
        rows.append((key, text, tuple(bindings.items())))
    best = min(key for key, _text, _map in rows)
    finalists = [(text, mapping) for key, text, mapping in rows if key == best]
    if len({text for text, _mapping in finalists}) != 1:
        return None
    nested = fusion[0]
    return NestedPrefix(finalists[0][0], atoms, root, blocked, core, finalists[0][1],
                        atoms - core, nested.preference_issues), nested


def nested_aza_sulfonamide_decision(mol, Chem):
    """A fully mapped general name; full parent preference remains pending."""
    rings = mol.GetRingInfo().AtomRings()
    if (len(rings) < 5 or not any(len(ring) == 5 for ring in rings)
            or sum(len(ring) == 6 for ring in rings) < 3
            or sum(atom.GetAtomicNum() == 7 and atom.GetIsAromatic()
                   for atom in mol.GetAtoms()) < 4):
        return None
    if (mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() == 1 or a.GetFormalCharge() or a.GetIsotope()
                   or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
                   or a.HasQuery() or a.GetAtomMapNum() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.groups) != 1
            or len(inventory.principal) != 1 or inventory.groups[0] != inventory.principal[0]
            or inventory.principal[0].kind != 'sulfonamide'):
        return None
    group = inventory.principal[0]
    nitrogen = mol.GetAtomWithIdx(group.ligand)
    other = [n.GetIdx() for n in nitrogen.GetNeighbors() if n.GetIdx() != group.center]
    if (nitrogen.GetAtomicNum() != 7 or len(other) != 1 or nitrogen.GetTotalNumHs() != 1
            or any(b.GetBondTypeAsDouble() != 1 for b in nitrogen.GetBonds())):
        return None
    phenyl = _phenyl_prefix(mol, Chem, other[0], group.ligand)
    if phenyl is None:
        return None
    bridge, fused = phenyl
    from .mapped_ring_parents import records, ring_systems
    core = next((s for s in ring_systems(mol) if group.owner in s), None)
    if (core is None or len(core) != 6 or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                 or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in core)):
        return None
    rows = records(mol, Chem, core_filter=core, principal=inventory.principal)
    if len(rows) != 1 or rows[0].parent != 'benzene' or rows[0].construction is None:
        return None
    construction = rows[0].construction
    ownership = construction.prefix_branch_ownership
    target = [entry for entry in ownership if entry[:2] == (bridge.root, bridge.blocked)
              and entry[2] == bridge.atoms]
    if len(target) != 1:
        return None
    # Only simple source-proved substituents are admitted on the outer
    # benzenesulfonamide ring; no unrelated legacy prefix may hide a missing
    # mapped child or pending component comparison.
    for child, host, atoms, text in ownership:
        if (child, host, atoms, text) == target[0]:
            continue
        simple = _simple_prefix(mol, Chem, child, host)
        if simple != (text, atoms):
            return None
    old = target[0][3]
    if sum(loc == 'N' and text == old for loc, text in construction.prefixes) != 1:
        return None
    prefixes = tuple((loc, bridge.text if loc == 'N' and text == old else text)
                     for loc, text in construction.prefixes)
    ownership = tuple((root, host, atoms, bridge.text if (root, host, atoms, text) == target[0] else text)
                      for root, host, atoms, text in ownership)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    mapped = frozenset(dict(construction.atom_locants))
    embedded = construction.embedded_atoms
    if (mapped != construction.atoms or mapped & embedded or mapped | embedded != heavy
            or not bridge.atoms <= embedded):
        return None

    # Re-evaluate both benzene directions with the actual replacement prefix;
    # the old nested spelling must not decide tied numbering alternatives.
    from .component_mapping import decorate_component
    from .formatting import prefix_alphabetization_key
    from .parent_selection import locant_key
    def flipped(locant):
        if locant == 'N':
            return locant
        if not locant.isdigit() or not 1 <= int(locant) <= 6:
            return None
        return locant if locant == '1' else str(8 - int(locant))
    alternatives = []
    for reverse in (False, True):
        numbered_prefixes = tuple(((flipped(loc) if reverse else loc), text) for loc, text in prefixes)
        if any(loc is None for loc, _ in numbered_prefixes):
            return None
        bindings = tuple((atom, flipped(loc) if reverse and atom in core else loc)
                         for atom, loc in construction.atom_locants)
        if any(loc is None for _atom, loc in bindings):
            return None
        candidate = replace(construction, prefixes=numbered_prefixes, atom_locants=bindings,
                            prefix_branch_ownership=ownership)
        name = decorate_component(mol, Chem, candidate)
        if not name:
            return None
        key = (tuple(sorted(locant_key(loc) for loc, _ in numbered_prefixes)),
               tuple(locant_key(loc) for loc, _ in sorted(numbered_prefixes,
                     key=lambda pair: (prefix_alphabetization_key(pair[1]), locant_key(pair[0])))))
        alternatives.append((key, name, candidate))
    best = min(key for key, _name, _component in alternatives)
    finalists = [(name, component) for key, name, component in alternatives if key == best]
    if len({name for name, _component in finalists}) != 1:
        return None
    name, selected = finalists[0]
    fused_branches = fused.branches
    aryl_branches = bridge.atoms - bridge.core - fused.atoms
    remainder = heavy - mapped - bridge.atoms
    roles = (('sulfonamide-parent', mapped, selected.atom_locants),
             ('N-aryl-bridge', bridge.core, bridge.atom_locants),
             ('N-aryl-branches', aryl_branches, ()),
             ('fused-parent', fused.core, fused.atom_locants),
             ('fused-parent-branches', fused_branches, ()),
             ('other-parent-branches-and-oxygen', remainder, ()))
    sets = [atoms for _label, atoms, _locants in roles]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)) or frozenset().union(*sets) != heavy:
        return None
    from .naming_methods import MethodDecision
    from .naming_result import SelectionLimitation
    issues = (*fused.preference_issues, SelectionLimitation('nested-aryl-sulfonamide',
        'P-44/P-45/P-51', 'The original graph has a complete nested mapped construction; '
        'global preferred parent and method comparison remains unresolved.'))
    return MethodDecision(name, 'substitutive', frozenset({group.center}), heavy,
        'P-25.3.4/P-29/P-44/P-45/P-51/P-66.1', selected.atom_locants,
        component_roles=roles, preference_issues=issues)
