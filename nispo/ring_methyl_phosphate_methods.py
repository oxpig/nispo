"""Source-bounded phosphate monoesters with a mapped oxolane organyl arm.

P-41 places a noncyclic phosphate ester (class 9) above cyclic amides,
alcohols and amines. P-67.1.3.2 names an ordinary phosphate monoester using
the independently mapped organyl component, followed by dihydrogen phosphate.
This producer proves those specific source roles; global PIN preference stays
unresolved.
"""

from collections import Counter

from .branch_rules import collect_subtree
from .component_mapping import (
    _mapped_stereochemistry_plan, decorate_component, mapped_branch,
    mapped_branch_ring_atoms,
)
from .extended_inventory import selection_inventory
from .formatting import enclose_name_unit
from .input_stereochemistry import unsupported_relative_stereochemistry
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation
from .stereochemistry import assigned_atom_stereochemistry


def ring_methyl_phosphate_decision(mol, Chem):
    """Construct an oxolanyl-methyl dihydrogen phosphate, or decline.

    Exactly one neutral P(=O)(OH)2-O-CH2-ring unit is admitted. The five-ring
    must be a single C4/O oxolane with a unique mapped record; every other
    source group must be a fully owned cyclic amide, alcohol or amine. Labels,
    charges, non-tetrahedral/bond stereo and a second senior class decline.
    Unspecified ring stereo is allowed only with an explicit limitation.
    """
    atoms = tuple(mol.GetAtoms())
    if (mol.GetNumHeavyAtoms() > 100 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()
            or unsupported_relative_stereochemistry(mol, Chem)
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 15, 17, 35, 53}
                   or a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
                   or a.HasQuery() or a.GetAtomMapNum()
                   or (int(a.GetChiralTag()) and a.GetAtomicNum() != 6) for a in atoms)
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    phosphorus = [a for a in atoms if a.GetAtomicNum() == 15]
    if len(phosphorus) != 1:
        return None
    p = phosphorus[0]
    if p.IsInRing() or p.GetDegree() != 4:
        return None
    double, hydroxyl, bridges = [], [], []
    for oxygen in p.GetNeighbors():
        if oxygen.GetAtomicNum() != 8 or oxygen.IsInRing():
            return None
        order = mol.GetBondBetweenAtoms(p.GetIdx(), oxygen.GetIdx()).GetBondTypeAsDouble()
        if order == 2 and oxygen.GetDegree() == 1:
            double.append(oxygen.GetIdx())
        elif order == 1 and oxygen.GetDegree() == 1 and oxygen.GetTotalNumHs() == 1:
            hydroxyl.append(oxygen.GetIdx())
        elif order == 1 and oxygen.GetDegree() == 2 and oxygen.GetTotalNumHs() == 0:
            bridges.append(oxygen.GetIdx())
        else:
            return None
    if len(double) != 1 or len(hydroxyl) != 2 or len(bridges) != 1:
        return None
    bridge = bridges[0]
    roots = [a.GetIdx() for a in mol.GetAtomWithIdx(bridge).GetNeighbors()
             if a.GetIdx() != p.GetIdx()]
    if len(roots) != 1:
        return None
    root = roots[0]
    carbon = mol.GetAtomWithIdx(root)
    children = [a.GetIdx() for a in carbon.GetNeighbors() if a.GetIdx() != bridge]
    if (carbon.GetAtomicNum() != 6 or carbon.IsInRing() or carbon.GetDegree() != 2
            or carbon.GetTotalNumHs() != 2 or len(children) != 1
            or mol.GetBondBetweenAtoms(bridge, root).GetBondTypeAsDouble() != 1):
        return None
    attachment = children[0]
    ring_atom = mol.GetAtomWithIdx(attachment)
    if (ring_atom.GetAtomicNum() != 6 or not ring_atom.IsInRing()
            or mol.GetBondBetweenAtoms(root, attachment).GetBondTypeAsDouble() != 1):
        return None
    rings = [frozenset(ring) for ring in mol.GetRingInfo().AtomRings()
             if attachment in ring]
    if (len(rings) != 1 or len(rings[0]) != 5
            or sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in rings[0]) != [6, 6, 6, 6, 8]):
        return None
    ring_core = rings[0]
    if (any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_core)
            or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds()
                   if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= ring_core)):
        return None

    organic = frozenset(collect_subtree(mol, root, bridge))
    ring_scope = frozenset(collect_subtree(mol, attachment, root))
    cuts = {(i, n.GetIdx()) for i in organic for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() not in organic}
    ring_cuts = {(i, n.GetIdx()) for i in ring_scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
                 if n.GetAtomicNum() > 1 and n.GetIdx() not in ring_scope}
    if (organic != ring_scope | {root} or cuts != {(root, bridge)}
            or ring_cuts != {(attachment, root)}):
        return None
    core = frozenset({p.GetIdx(), *double, *hydroxyl})
    organyl = organic | {bridge}
    heavy = frozenset(a.GetIdx() for a in atoms)
    if core & organyl or core | organyl != heavy:
        return None

    # P-41's class-9 ester priority over these specific subordinate classes is
    # source-proved here. The shared source frontier currently does not cover
    # class 9, so its legacy numeric ordering is not accepted as the proof.
    inventory = selection_inventory(mol)
    phosphate = [g for g in inventory.groups
                 if g.kind == 'phosphate-acid-ester' and g.center == p.GetIdx()]
    if (not inventory.supported or len(phosphate) != 1
            or frozenset(phosphate[0].atoms) != core | {bridge}
            or len(inventory.principal) != 1 or inventory.principal[0] != phosphate[0]):
        return None
    for group in inventory.groups:
        if group == phosphate[0]:
            continue
        if (group.kind not in {'lactam', 'alcohol', 'amine'}
                or not set(group.atoms) <= organic
                or group.kind == 'lactam' and not mol.GetAtomWithIdx(group.center).IsInRing()):
            return None

    from .mapped_ring_parents import records

    candidates = records(mol, Chem, scope=ring_scope, attachment=attachment,
                         core_filter=ring_core)
    if len(candidates) != 1:
        return None
    record = candidates[0]
    component = record.construction
    if (component is None or not record.name or not record.hydrogen_budget_proved
            or not record.annotation_coverage or record.preference_issues
            or component.atoms != ring_core or component.atoms & component.embedded_atoms
            or component.atoms | component.embedded_atoms != ring_scope
            or decorate_component(mol, Chem, component) != record.name):
        return None
    branch_atoms, recorded = set(), Counter()
    ring_mapping = dict(component.atom_locants)
    for branch_root, host, owned, text in component.prefix_branch_ownership:
        if (host not in ring_core or branch_root not in owned or branch_atoms & set(owned)
                or owned & ring_core
                or frozenset(collect_subtree(mol, branch_root, host)) != owned):
            return None
        branch_cuts = {(idx, n.GetIdx()) for idx in owned
                       for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                       if n.GetAtomicNum() > 1 and n.GetIdx() not in owned}
        rebuilt, rebuilt_owned = mapped_branch(mol, Chem, branch_root, host)
        nested_rings = frozenset(i for i in owned if mol.GetAtomWithIdx(i).IsInRing())
        if (branch_cuts != {(branch_root, host)} or rebuilt != text or rebuilt_owned != owned
                or not nested_rings <= mapped_branch_ring_atoms(
                    mol, Chem, branch_root, host, rebuilt, rebuilt_owned)):
            return None
        branch_atoms.update(owned)
        recorded[(ring_mapping[host], text)] += 1
    if branch_atoms != set(component.embedded_atoms) or recorded != Counter(component.prefixes):
        return None
    # Rejoin the named record to the actual mapped ring writer and all ring
    # atoms it witnessed, including a nested heterocyclic substituent.
    ring_text, mapped = mapped_branch(mol, Chem, attachment, root)
    ring_atoms = frozenset(i for i in ring_scope if mol.GetAtomWithIdx(i).IsInRing())
    if (ring_text != record.name or mapped != ring_scope or not ring_atoms
            or mapped_branch_ring_atoms(mol, Chem, attachment, root, ring_text, mapped) != ring_atoms):
        return None
    plan = _mapped_stereochemistry_plan(
        mol, Chem, component, ring_mapping, component.parent_annotation_proof)
    tagged = {a.GetIdx() for a in atoms if int(a.GetChiralTag())}
    source_descriptors = dict(assigned_atom_stereochemistry(mol, Chem, include_pseudo=True))
    if (plan is None or tagged != {row[0] for row in plan[1]}
            or tagged != set(source_descriptors)
            or any(source_descriptors[idx] not in {'R', 'S'} for idx in tagged)
            or any(source_descriptors[idx] != value for idx, _locant, value in plan[1])):
        return None
    potential = tuple(Chem.FindPotentialStereo(mol))
    if any(item.type != Chem.StereoType.Atom_Tetrahedral
           or item.centeredOn not in ring_core for item in potential):
        return None
    issues = [SelectionLimitation(
        'phosphate-ester-parent', 'P-41/P-67.1.3.2/P-44/P-45',
        'The phosphate class and mapped organic component are available; complete preferred-name comparison remains unresolved.')]
    if any(item.specified != Chem.StereoSpecified.Specified for item in potential):
        issues.append(SelectionLimitation(
            'stereochemistry', 'P-91.2.1.2.1/P-91.2.2',
            'One or more potential stereogenic units have no source configuration.'))
    name = enclose_name_unit(record.name) + 'methyl dihydrogen phosphate'
    return MethodDecision(
        name, 'functional-class', frozenset({p.GetIdx()}), heavy,
        'P-41/P-29.3.2/P-67.1.3.2/P-16.5.4', tuple(component.atom_locants),
        component_roles=(('phosphorus-anion', core, ()),
                         ('O-organyl', organyl, tuple(component.atom_locants))),
        preference_issues=tuple(issues),
    )
