"""Original-graph functional-class esters of monocyclic pyridine acids."""

from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def _cycle_from_nitrogen(mol, ring, nitrogen, first):
    path = [nitrogen, first]
    while len(path) < len(ring):
        neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(path[-1]).GetNeighbors()
                     if n.GetIdx() in ring and n.GetIdx() != path[-2]]
        if len(neighbors) != 1 or neighbors[0] in path:
            return ()
        path.append(neighbors[0])
    return tuple(path) if mol.GetBondBetweenAtoms(path[-1], nitrogen) else ()


def _one_cut(mol, root, host, owned):
    from .branch_rules import collect_subtree

    owned = frozenset(owned)
    if (not owned or root not in owned or host in owned
            or owned != frozenset(collect_subtree(mol, root, host))):
        return False
    cuts = {(i, n.GetIdx()) for i in owned for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() not in owned}
    return cuts == {(root, host)}


def _linear_ester_organyl(mol, root, oxygen, owned):
    """Prove a C1-C4 saturated, unbranched organyl on the ester oxygen."""
    if not 1 <= len(owned) <= 4:
        return False
    for idx in owned:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.GetIsAromatic() or atom.IsInRing()
                or any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds())):
            return False
        carbon_neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in owned]
        if len(carbon_neighbors) > (1 if idx == root else 2):
            return False
        external = [n.GetIdx() for n in atom.GetNeighbors()
                    if n.GetAtomicNum() > 1 and n.GetIdx() not in owned]
        if external != ([oxygen] if idx == root else []):
            return False
    return True


def pyridine_carboxylate_decision(mol, Chem):
    """Build a bounded pyridine-carboxylate ester with source-atom ownership.

    P-65.1.2.2.2 and P-65.6.3.3 select a systematic ring-carboxylate
    instead of picolinate/nicotinate aliases. P-14.4 gives the suffix its
    lowest locant before lower-priority ring prefixes. Only ordinary neutral,
    unannotated monocyclic pyridines and source-proved one-cut branches are
    admitted; the complete method and branch-preference frontier stays open.
    """
    if (mol is None or mol.GetNumHeavyAtoms() > 80 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 17, 35, 53}
                   or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() or a.GetAtomMapNum()
                   or a.HasProp('_UnknownStereo') for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    rings = tuple(frozenset(r) for r in mol.GetRingInfo().AtomRings())
    if len(rings) != 1 or len(rings[0]) != 6:
        return None
    ring = rings[0]
    nitrogens = [i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 7]
    if (len(nitrogens) != 1 or any(not mol.GetAtomWithIdx(i).GetIsAromatic()
            or mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7} for i in ring)):
        return None
    if any(sum(n.GetIdx() in ring for n in mol.GetAtomWithIdx(i).GetNeighbors()) != 2
           or any(not b.GetIsAromatic() for b in mol.GetAtomWithIdx(i).GetBonds()
                  if b.GetOtherAtomIdx(i) in ring) for i in ring):
        return None
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.principal) != 1
            or inventory.principal[0].kind != 'ester'
            or any(g.kind not in {'ester', 'carbamate', 'alcohol', 'amine'}
                   for g in inventory.groups)):
        return None
    group = inventory.principal[0]
    if (group.owner not in ring or mol.GetAtomWithIdx(group.owner).GetAtomicNum() != 6
            or group.center in ring or group.ligand in ring
            or len(group.atoms) != 3):
        return None
    carbonyl, bridge = mol.GetAtomWithIdx(group.center), mol.GetAtomWithIdx(group.ligand)
    carbonyl_oxygens = [n.GetIdx() for n in carbonyl.GetNeighbors() if n.GetAtomicNum() == 8
                        and mol.GetBondBetweenAtoms(group.center, n.GetIdx()).GetBondTypeAsDouble() == 2]
    if (len(carbonyl_oxygens) != 1 or bridge.GetAtomicNum() != 8 or bridge.GetDegree() != 2
            or set(group.atoms) != {group.center, group.ligand, carbonyl_oxygens[0]}
            or mol.GetBondBetweenAtoms(group.owner, group.center).GetBondTypeAsDouble() != 1
            or mol.GetBondBetweenAtoms(group.center, group.ligand).GetBondTypeAsDouble() != 1):
        return None
    roots = [n.GetIdx() for n in bridge.GetNeighbors() if n.GetIdx() != group.center]
    if len(roots) != 1:
        return None
    from .component_mapping import mapped_branch

    organyl, organyl_atoms = mapped_branch(mol, Chem, roots[0], group.ligand)
    if (not organyl or not _one_cut(mol, roots[0], group.ligand, organyl_atoms)
            or not _linear_ester_organyl(mol, roots[0], group.ligand, organyl_atoms)):
        return None
    occupied = set(ring) | set(group.atoms) | set(organyl_atoms)
    if len(occupied) != len(ring) + len(group.atoms) + len(organyl_atoms):
        return None
    branches = []
    for host in ring:
        atom = mol.GetAtomWithIdx(host)
        for neighbor in atom.GetNeighbors():
            root = neighbor.GetIdx()
            if root in ring or root == group.center:
                continue
            if atom.GetAtomicNum() != 6 or mol.GetBondBetweenAtoms(host, root).GetBondTypeAsDouble() != 1:
                return None
            text, owned = mapped_branch(mol, Chem, root, host)
            if (not text or not _one_cut(mol, root, host, owned)
                    or set(owned) & occupied):
                return None
            occupied.update(owned)
            branches.append((host, root, frozenset(owned), text))
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if occupied != heavy:
        return None
    # Every subordinate source class must be owned by one complete branch.
    if any(g != group and not any(set(g.atoms) <= owned for _, _, owned, _ in branches)
           for g in inventory.groups):
        return None
    nitrogen = nitrogens[0]
    adjacent = [n.GetIdx() for n in mol.GetAtomWithIdx(nitrogen).GetNeighbors()
                if n.GetIdx() in ring]
    from .formatting import format_prefixes, join_prefix_parent, prefix_alphabetization_key

    options = []
    for first in adjacent:
        seq = _cycle_from_nitrogen(mol, ring, nitrogen, first)
        if not seq:
            return None
        locants = {idx: str(i) for i, idx in enumerate(seq, 1)}
        suffix_loc = int(locants[group.owner])
        prefixes = [(locants[host], text) for host, _root, _owned, text in branches]
        prefix_locants = tuple(sorted(int(loc) for loc, _ in prefixes))
        citation_locants = tuple(int(loc) for loc, _ in sorted(
            prefixes, key=lambda row: (prefix_alphabetization_key(row[1]), int(row[0]))))
        prefix_text = format_prefixes(prefixes)
        acid = join_prefix_parent(prefix_text, f'pyridine-{suffix_loc}-carboxylate')
        name = f'{organyl} {acid}'
        options.append(((suffix_loc, prefix_locants, citation_locants), name, locants))
    rank = min(row[0] for row in options)
    selected = [row for row in options if row[0] == rank]
    if len({row[1] for row in selected}) != 1:
        return None
    _, name, locants = selected[0]
    issues = [SelectionLimitation('method', 'P-44/P-45/P-51',
        'The mapped pyridine ester is a general construction; complete competing-method and branch preference remain unresolved.')]
    if (inventory.source_frontier.status not in {'selected', 'tie'}
            and any(g.kind == 'carbamate' for g in inventory.groups)):
        issues.append(SelectionLimitation('principal-group', 'P-41',
            'The source-qualified class-9 comparison for the ester and carbamate is not complete.'))
    try:
        unknown = any(site.specified != Chem.StereoSpecified.Specified
                      for site in Chem.FindPotentialStereo(mol))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        unknown = True
    if unknown:
        issues.append(SelectionLimitation('stereochemistry', 'P-91.2.1.2.1/P-91.2.2',
            'Potentially stereogenic source features are unassigned; this name does not assert their configuration.'))
    mapping = tuple((idx, locants[idx]) for idx in sorted(ring, key=lambda i: int(locants[i])))
    mapping += ((group.center, 'C' + locants[group.owner]),)
    roles = [('pyridine-ring', frozenset(ring), mapping[:-1]),
             ('ester', frozenset(group.atoms), (mapping[-1],)),
             ('ester-organyl', frozenset(organyl_atoms), ())]
    roles.extend(('ring-prefix', owned, ()) for _, _, owned, _ in branches)
    return MethodDecision(name, 'functional-class', frozenset({group.center}), frozenset(heavy),
        'P-65.1.2.2.2/P-65.6.3.3.1/P-14.4', mapping,
        component_roles=tuple(roles), preference_issues=tuple(issues))
