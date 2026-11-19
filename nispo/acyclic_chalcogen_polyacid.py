"""Source-owned scope for partial carbon parents in O/S-linked polyacids.

The ordinary carbon chain may cover two acid suffixes while another acid is
part of a singly attached prefix.  This evidence admits a mapped *general*
construction; P-51 method and full preferred-parent competition remain open.
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AcyclicChalcogenPolyacidScope:
    linker: int
    acid_groups: tuple
    carbon_components: tuple[frozenset[int], ...]


def linear_ether_polyacid_decision(mol, Chem):
    """One linear C/O replacement chain carrying all its external acids."""
    if mol.GetRingInfo().NumRings() or not 8 <= mol.GetNumAtoms() <= 64 or mol.GetStereoGroups():
        return None
    from .constants import MULT, ROOTS
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    acids = inventory.source_frontier.groups
    if (not inventory.supported or inventory.source_frontier.status != 'tie'
            or len(acids) < 2 or len(acids) not in MULT
            or any(g.kind != 'acid' for g in acids)
            or any(g.kind not in {'acid', 'alcohol'} for g in inventory.groups)):
        return None
    atoms = tuple(mol.GetAtoms())
    if any(a.GetAtomicNum() not in {6, 8} or a.HasQuery() or a.GetIsotope() or a.GetFormalCharge()
           or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) for a in atoms):
        return None
    acid_atoms = frozenset(i for group in acids for i in group.atoms)
    hydroxy_atoms = frozenset(i for group in inventory.groups if group.kind == 'alcohol'
                              for i in group.atoms if atoms[i].GetAtomicNum() == 8)
    core = set(range(len(atoms))) - acid_atoms - hydroxy_atoms
    if not core or any(group.owner not in core for group in acids):
        return None
    neighbors = {i: tuple(a.GetIdx() for a in atoms[i].GetNeighbors() if a.GetIdx() in core)
                 for i in core}
    ends = [i for i, row in neighbors.items() if len(row) == 1]
    if (len(ends) != 2 or any(atoms[i].GetAtomicNum() != 6 for i in ends)
            or any(not 1 <= len(row) <= 2 for row in neighbors.values())
            or any(b.HasQuery() or b.GetBondTypeAsDouble() != 1 or int(b.GetStereo())
                   for i in core for b in atoms[i].GetBonds())):
        return None
    path = [ends[0]]
    previous = -1
    while len(path) < len(core):
        next_atoms = [i for i in neighbors[path[-1]] if i != previous]
        if len(next_atoms) != 1:
            return None
        previous = path[-1]
        path.append(next_atoms[0])
    if set(path) != core or len(path) not in ROOTS:
        return None
    oxygen = {i for i in core if atoms[i].GetAtomicNum() == 8}
    if (not oxygen or len(oxygen) > 1 and len(oxygen) not in MULT
            or any(len(neighbors[i]) != 2 or atoms[i].GetTotalValence() != 2
                   or any(atoms[j].GetAtomicNum() != 6 for j in neighbors[i]) for i in oxygen)):
        return None

    from .complete_functional_parent import _render
    from .naming_methods import MethodDecision
    from .naming_result import SelectionLimitation

    options = []
    for seq in (tuple(path), tuple(reversed(path))):
        positions = {i: locant for locant, i in enumerate(seq, 1)}
        replacements = tuple(sorted(positions[i] for i in oxygen))
        suffixes = tuple(sorted(positions[group.owner] for group in acids))
        prefix = ','.join(map(str, replacements)) + '-'
        if len(replacements) > 1:
            prefix += MULT[len(replacements)]
        parent = prefix + 'oxa' + ROOTS[len(seq)] + 'ane'
        row = _render(mol, Chem, acids, 'acid', seq, True, False, acid_atoms, (),
                      parent_override=parent, locanted_acid_fallback=True)
        if not row:
            return None
        options.append(((replacements, suffixes, row[2]), row))
    _, best = min(options, key=lambda item: (item[0], item[1][0]))
    issues = (SelectionLimitation(
        'replacement-chain-polyacid', 'P-15.4/P-44/P-65',
        'The single ether chain owns all original acid sites. Other naming methods remain unresolved.'),
        *polyacid_stereochemistry_issues(mol, Chem))
    return MethodDecision(best[0], 'mapped-replacement-chain-polyacid',
                          frozenset(group.center for group in acids), frozenset(range(len(atoms))),
                          'P-15.4/P-44/P-65', tuple(best[3].items()), preference_issues=issues)


def acyclic_chalcogen_polyacid_scope(mol, Chem, inventory):
    """Prove a neutral one-ether/thioether tree with a 2+1 acid partition.

    All oxygen atoms belong to acids except for the optional ether linker;
    optional amino nitrogen is a terminal, neutral primary amine.  Isotopic,
    ionic, unsaturated, enhanced-stereo, and additional functional groups
    require their own characteristic and annotation comparisons.
    """
    if (mol is None or mol.GetNumHeavyAtoms() > 48 or mol.GetRingInfo().NumRings()
            or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or not inventory.supported):
        return None
    atoms = tuple(mol.GetAtoms())
    if any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
           or a.GetAtomicNum() not in {1, 6, 7, 8, 16} for a in atoms):
        return None
    if any(int(b.GetStereo()) or b.GetIsAromatic() for b in mol.GetBonds()):
        return None
    acids = tuple(inventory.principal)
    all_acids = tuple(group for group in inventory.groups if group.kind == 'acid')
    frontier = inventory.source_frontier
    if (len(acids) != 3 or any(group.kind != 'acid' for group in acids)
            or set(acids) != set(all_acids)
            or frontier.status != 'tie' or frontier.unknown_pairs or frontier.cycle
            or set(frontier.groups) != set(acids)
            or any(group.kind not in {'acid', 'amine'} for group in inventory.groups)):
        return None
    acid_atoms = set()
    for group in acids:
        if (len(group.atoms) != 3 or group.center not in group.atoms
                or acid_atoms.intersection(group.atoms)):
            return None
        carbon = mol.GetAtomWithIdx(group.center)
        oxygens = [mol.GetAtomWithIdx(idx) for idx in group.atoms if idx != group.center]
        if (carbon.GetAtomicNum() != 6 or len(oxygens) != 2
                or any(atom.GetAtomicNum() != 8 or atom.GetDegree() != 1 for atom in oxygens)
                or group.owner < 0 or mol.GetAtomWithIdx(group.owner).GetAtomicNum() != 6
                or set(n.GetIdx() for n in carbon.GetNeighbors() if n.GetAtomicNum() > 1)
                    != {group.owner, *(atom.GetIdx() for atom in oxygens)}):
            return None
        oxygen_bonds = [(mol.GetBondBetweenAtoms(group.center, atom.GetIdx()).GetBondTypeAsDouble(),
                         atom.GetTotalNumHs(includeNeighbors=True)) for atom in oxygens]
        if sorted(oxygen_bonds) != [(1.0, 1), (2.0, 0)]:
            return None
        acid_atoms.update(group.atoms)
    linkers = [a for a in atoms if a.GetAtomicNum() in {8, 16}
               and a.GetIdx() not in acid_atoms]
    if len(linkers) != 1:
        return None
    linker = linkers[0]
    if (linker.GetDegree() != 2 or linker.GetTotalValence() != 2
            or linker.GetTotalNumHs(includeNeighbors=True) != 0
            or any(n.GetAtomicNum() != 6 for n in linker.GetNeighbors())
            or any(b.GetBondTypeAsDouble() != 1 for b in linker.GetBonds())):
        return None
    nitrogens = [a for a in atoms if a.GetAtomicNum() == 7]
    if len(nitrogens) > 1 or any(
            atom.GetDegree() != 1 or atom.GetTotalNumHs(includeNeighbors=True) != 2
            or atom.GetNeighbors()[0].GetAtomicNum() != 6
            or atom.GetBonds()[0].GetBondTypeAsDouble() != 1 for atom in nitrogens):
        return None
    carbon = {a.GetIdx() for a in atoms if a.GetAtomicNum() == 6}
    if not carbon or len(carbon) > 40:
        return None
    for bond in mol.GetBonds():
        first, second = bond.GetBeginAtom(), bond.GetEndAtom()
        order = bond.GetBondTypeAsDouble()
        if order == 1:
            continue
        if not (order == 2 and {first.GetAtomicNum(), second.GetAtomicNum()} == {6, 8}
                and first.GetIdx() in acid_atoms and second.GetIdx() in acid_atoms):
            return None
    # Connected and ring-free is a tree. Removing its one O/S linker produces
    # exactly two carbon components, with two acid centers on one and one on
    # the other. No all-carbon parent can own all three as suffixes.
    unseen = set(carbon)
    components = []
    while unseen:
        component, pending = set(), [next(iter(unseen))]
        while pending:
            idx = pending.pop()
            if idx not in unseen:
                continue
            unseen.remove(idx)
            component.add(idx)
            pending.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                           if n.GetIdx() in unseen and n.GetAtomicNum() == 6)
        components.append(frozenset(component))
    if (len(components) != 2
            or sorted(sum(group.center in component for group in acids)
                      for component in components) != [1, 2]):
        return None
    return AcyclicChalcogenPolyacidScope(linker.GetIdx(), acids, tuple(components))


def mapped_polyacid_proposal_complete(mol, Chem, scope, proposal):
    """Revalidate the selected suffix sites and all mapped prefix ownership."""
    from .component_mapping import fully_covered
    from .bounded_mixed_acid_ester_fast_guard import _one_cut_branch

    component = proposal.construction
    if (component is None or proposal.route != 'mapped-chain-partial-functional'
            or proposal.principal_kind != 'acid' or len(proposal.principal_centers) != 2
            or not proposal.annotation_coverage or not proposal.parent_atoms
            or any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in proposal.parent_atoms)
            or not proposal.principal_centers <= frozenset((proposal.parent_atoms[0],
                                                            proposal.parent_atoms[-1]))
            or component.atoms & component.embedded_atoms
            or not fully_covered(mol, component.atoms, component.embedded_atoms)):
        return False
    selected = [group for group in scope.acid_groups if group.center in proposal.principal_centers]
    omitted = [group for group in scope.acid_groups if group.center not in proposal.principal_centers]
    if (len(selected) != 2 or len(omitted) != 1
            or not all(set(group.atoms) <= component.atoms for group in selected)):
        return False
    branches, seen = component.prefix_branch_ownership, set()
    for branch in branches:
        root, host, owned, _text = branch
        if (host not in component.atoms or set(owned) & seen
                or not _one_cut_branch(mol, Chem, branch)):
            return False
        seen.update(owned)
    acid_branches = [branch for branch in branches
                     if set(omitted[0].atoms) <= set(branch[2])]
    # A directly attached O/S prefix keeps the acid-bearing carbon fragment
    # inside its own mapped branch. Other attachment shapes can invoke an
    # unproved hetero-chain prefix ending in oxygen, so they stay outside.
    return (seen == set(component.embedded_atoms) and len(acid_branches) == 1
            and acid_branches[0][0] == scope.linker)


def polyacid_stereochemistry_issues(mol, Chem):
    """Preserve unassigned source stereo in this general construction."""
    from .naming_result import SelectionLimitation

    try:
        unknown = any(site.specified != Chem.StereoSpecified.Specified
                      for site in Chem.FindPotentialStereo(mol))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        reason = 'The original-graph stereochemistry census did not complete.'
    else:
        if not unknown:
            return ()
        reason = ('At least one potentially stereogenic source atom is unassigned; '
                  'the general name does not assert its configuration.')
    return (SelectionLimitation('stereochemistry', 'P-91.2.1.2.1/P-91.2.2', reason),)
