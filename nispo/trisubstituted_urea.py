"""Original-graph construction of bounded N,N,N′-substituted ureas.

P-66.1.6.1.1 retains urea and its N/N′ sites. For 2+1 substitution the
two-branch nitrogen receives N by the lowest-set rule P-14.3.5/P-14.4(f).
Branch, stereo and characteristic-group proofs are local; the complete
preferred-parent comparison remains explicitly unresolved.
"""
from dataclasses import dataclass

from .branch_rules import collect_subtree
from .component_mapping import (
    MappedComponent, _mapped_stereochemistry_plan, decorate_component,
    mapped_branch, mapped_branch_ring_atoms,
)
from .extended_inventory import selection_inventory
from .formatting import format_prefixes
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


@dataclass(frozen=True, slots=True)
class _Arm:
    root: int
    nitrogen: int
    atoms: frozenset[int]
    text: str
    stereo_atoms: frozenset[int]


def _one_cut(mol, atoms, root, blocked):
    return {(i, n.GetIdx()) for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms} == {(root, blocked)}


def _mapped_ring(mol, Chem, root, blocked, atoms):
    """Prove ring text, original atom ownership and all assigned stereo."""
    from .mapped_ring_parents import records

    if not mol.GetAtomWithIdx(root).IsInRing() or not _one_cut(mol, atoms, root, blocked):
        return None
    candidates = records(mol, Chem, scope=atoms, attachment=root)
    if len(candidates) != 1:
        return None
    record = candidates[0]
    component = record.construction
    if (not record.name or not record.annotation_coverage or not record.hydrogen_budget_proved
            or record.preference_issues or component is None
            or component.atoms & component.embedded_atoms
            or component.atoms | component.embedded_atoms != atoms):
        return None
    tagged = frozenset(i for i in atoms if int(mol.GetAtomWithIdx(i).GetChiralTag()))
    if not tagged <= component.atoms:
        return None
    plan = _mapped_stereochemistry_plan(
        mol, Chem, component, dict(component.atom_locants), component.parent_annotation_proof)
    if plan is None or frozenset(row[0] for row in plan[1]) != tagged:
        return None
    text, owned = mapped_branch(mol, Chem, root, blocked)
    rings = frozenset(i for i in atoms if mol.GetAtomWithIdx(i).IsInRing())
    if (text != record.name or owned != atoms or not rings
            or mapped_branch_ring_atoms(mol, Chem, root, blocked, text, owned) != rings):
        return None
    return text, tagged


def _arm(mol, Chem, root, nitrogen):
    from .urea_methods import _ordinary_carbon_branch

    atoms = frozenset(collect_subtree(mol, root, nitrogen))
    if not atoms or not _one_cut(mol, atoms, root, nitrogen):
        return None
    root_atom = mol.GetAtomWithIdx(root)
    if root_atom.GetAtomicNum() != 6:
        return None
    if root_atom.IsInRing():
        mapped = _mapped_ring(mol, Chem, root, nitrogen, atoms)
        return _Arm(root, nitrogen, atoms, *mapped) if mapped else None

    carbon_atoms = _ordinary_carbon_branch(mol, root, nitrogen, Chem)
    if carbon_atoms == atoms:
        # This route does not yet construct stereo on a carbon-only arm.
        if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in atoms):
            return None
        text, owned = mapped_branch(mol, Chem, root, nitrogen)
        return _Arm(root, nitrogen, atoms, text, frozenset()) if text and owned == atoms else None

    # P-29.3.2: a ring may be linked through CH2, or through a CH bearing
    # one terminal methyl. The ring writer and carbon component retain
    # separate original atom/stereo ownership before their names are joined.
    children = [a.GetIdx() for a in root_atom.GetNeighbors()
                if a.GetIdx() != nitrogen and a.GetAtomicNum() > 1]
    rings = [i for i in children if mol.GetAtomWithIdx(i).IsInRing()]
    others = [i for i in children if not mol.GetAtomWithIdx(i).IsInRing()]
    if (len(rings) != 1 or len(others) > 1
            or any(mol.GetBondBetweenAtoms(root, i).GetBondTypeAsDouble() != 1
                   for i in children)):
        return None
    if not others:
        if root_atom.GetTotalNumHs(includeNeighbors=True) != 2 or int(root_atom.GetChiralTag()):
            return None
        parent, mapping, carbon_scope = 'methyl', ((root, '1'),), frozenset({root})
    else:
        methyl = mol.GetAtomWithIdx(others[0])
        if (root_atom.GetTotalNumHs(includeNeighbors=True) != 1
                or methyl.GetAtomicNum() != 6 or methyl.GetDegree() != 1
                or methyl.GetTotalNumHs(includeNeighbors=True) != 3
                or int(methyl.GetChiralTag())):
            return None
        parent = 'ethyl'
        mapping = ((root, '1'), (methyl.GetIdx(), '2'))
        carbon_scope = frozenset({root, methyl.GetIdx()})
    ring_scope = frozenset(collect_subtree(mol, rings[0], root))
    if ring_scope & carbon_scope or ring_scope | carbon_scope != atoms:
        return None
    ring = _mapped_ring(mol, Chem, rings[0], root, ring_scope)
    if ring is None:
        return None
    ring_text, ring_stereo = ring
    component = MappedComponent(
        parent, mapping, carbon_scope, (('1', ring_text),), ring_scope,
        systematic_parent=True, omit_methyl_prefix_locant=not others,
        prefix_branch_ownership=((rings[0], root, ring_scope, ring_text),))
    text = decorate_component(mol, Chem, component)
    plan = _mapped_stereochemistry_plan(mol, Chem, component, dict(mapping), None)
    carbon_tagged = frozenset(i for i in carbon_scope if int(mol.GetAtomWithIdx(i).GetChiralTag()))
    if not text or plan is None or frozenset(row[0] for row in plan[1]) != carbon_tagged:
        return None
    return _Arm(root, nitrogen, atoms, text, carbon_tagged | ring_stereo)


def trisubstituted_urea_decision(mol, Chem, carbonyl, oxygen, nitrogens):
    """Return a completely owned 2+1 urea general construction, or decline."""
    if (mol.GetAtomWithIdx(oxygen).GetAtomicNum() != 8 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons()
                   or (int(a.GetChiralTag()) and a.GetChiralTag() not in
                       (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW))
                   for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())
            or any(mol.GetAtomWithIdx(i).IsInRing() for i in (carbonyl, *nitrogens))):
        return None
    core = frozenset({carbonyl, oxygen, *nitrogens})
    inventory = selection_inventory(mol)
    frontier = getattr(inventory, 'source_frontier', None)
    if (not inventory.supported or frontier is None or frontier.status != 'selected'
            or frontier.unknown_pairs or frontier.cycle or len(frontier.groups) != 1
            or frontier.groups[0].kind != 'urea' or frontier.groups[0].center != carbonyl
            or frozenset(frontier.groups[0].atoms) != core):
        return None
    roots = {}
    for nitrogen in nitrogens:
        atom = mol.GetAtomWithIdx(nitrogen)
        children = [a.GetIdx() for a in atom.GetNeighbors()
                    if a.GetIdx() != carbonyl and a.GetAtomicNum() > 1]
        if (atom.GetAtomicNum() != 7 or atom.GetTotalValence() != 3
                or len(children) not in {1, 2}
                or atom.GetTotalNumHs(includeNeighbors=True) != 2 - len(children)
                or any(mol.GetBondBetweenAtoms(nitrogen, i).GetBondTypeAsDouble() != 1
                       for i in children)):
            return None
        roots[nitrogen] = children
    if sorted(map(len, roots.values())) != [1, 2]:
        return None
    double_n = next(n for n in nitrogens if len(roots[n]) == 2)
    single_n = next(n for n in nitrogens if len(roots[n]) == 1)
    arms, owned, stereo_atoms = [], set(core), set()
    for nitrogen in (double_n, single_n):
        for root in roots[nitrogen]:
            arm = _arm(mol, Chem, root, nitrogen)
            if arm is None or owned & arm.atoms:
                return None
            owned.update(arm.atoms)
            stereo_atoms.update(arm.stereo_atoms)
            arms.append(arm)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if frozenset(owned) != heavy:
        return None
    tagged = {a.GetIdx() for a in mol.GetAtoms() if int(a.GetChiralTag())}
    if tagged != stereo_atoms:
        return None
    potential = tuple(Chem.FindPotentialStereo(mol))
    if any(item.specified == Chem.StereoSpecified.Specified
           and (item.type != Chem.StereoType.Atom_Tetrahedral
                or item.centeredOn not in stereo_atoms) for item in potential):
        return None
    # P-14.5 excludes locants and stereo descriptors from alphabetization;
    # leave distinct, indistinguishable sort keys unresolved rather than
    # breaking the tie by original atom order.
    from .urea_methods import _name_order
    for i, left in enumerate(arms):
        for right in arms[i + 1:]:
            if _name_order(left.text, right.text) is None:
                return None
    locants = {double_n: 'N', single_n: 'N′'}
    prefix = format_prefixes(
        [(locants[arm.nitrogen], arm.text) for arm in arms], hierarchical_enclosures=True)
    if not prefix:
        return None
    issues = [SelectionLimitation(
        'urea-ring-substitution', 'P-66.1.6.1/P-44/P-45',
        'The complete mapped 2+1 N-substituted urea is available; preferred parent and method comparisons remain unresolved.')]
    if any(item.specified != Chem.StereoSpecified.Specified for item in potential):
        issues.append(SelectionLimitation(
            'stereochemistry', 'P-91.2.1.2.1/P-91.2.2',
            'One or more potential stereogenic units have no source configuration.'))
    roles = (('urea-core', core, ((double_n, 'N'), (single_n, 'N′'))),) + tuple(
        ('N-substituent', arm.atoms, ((arm.nitrogen, locants[arm.nitrogen]),))
        for arm in arms)
    return MethodDecision(prefix + 'urea', 'substitutive', frozenset({carbonyl}), heavy,
        'P-14.3.5/P-14.4(f)/P-14.5/P-66.1.6.1.1.1/P-66.1.6.1.1.2',
        ((double_n, 'N'), (single_n, 'N′')), component_roles=roles,
        preference_issues=tuple(issues))
