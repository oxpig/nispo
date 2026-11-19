"""Mapped saturated bicyclic carboxylates on the original ester graph.

The bounded theta topology supplies the complete ring-numbering frontier;
the selected ring attachment, suffix atoms and organyl arm retain separate
original-atom and stereochemistry ownership throughout construction.
"""
from collections import Counter
from dataclasses import replace

from .branch_rules import collect_subtree
from .component_mapping import _mapped_stereochemistry_plan, decorate_component, mapped_branch
from .extended_inventory import selection_inventory
from .mapped_ring_parents import records, ring_systems
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation
from .von_baeyer_frontier import bicyclic_theta_frontier


def bicyclic_carboxylate_decision(mol, Chem):
    """Construct a bounded single ester on a proved theta bicyclic parent."""
    if (mol.GetNumHeavyAtoms() > 100 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()
            or any(a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons()
                   or (int(a.GetChiralTag()) and a.GetChiralTag() not in
                       (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW))
                   for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.groups) != 1
            or len(inventory.principal) != 1 or inventory.principal[0].kind != 'ester'):
        return None
    group = inventory.principal[0]
    carbonyl, host, bridge = group.center, group.owner, group.ligand
    if any(not 0 <= i < mol.GetNumAtoms() for i in (carbonyl, host, bridge)):
        return None
    carbon = mol.GetAtomWithIdx(carbonyl)
    oxygen = mol.GetAtomWithIdx(bridge)
    bridge_bond = mol.GetBondBetweenAtoms(carbonyl, bridge)
    host_bond = mol.GetBondBetweenAtoms(carbonyl, host)
    if bridge_bond is None or host_bond is None:
        return None
    oxo = [a.GetIdx() for a in carbon.GetNeighbors() if a.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(carbonyl, a.GetIdx()).GetBondTypeAsDouble() == 2]
    if (carbon.IsInRing() or carbon.GetDegree() != 3 or len(oxo) != 1
            or mol.GetAtomWithIdx(oxo[0]).GetDegree() != 1
            or oxygen.GetAtomicNum() != 8 or oxygen.IsInRing() or oxygen.GetDegree() != 2
            or bridge_bond.GetBondTypeAsDouble() != 1
            or host_bond.GetBondTypeAsDouble() != 1
            or mol.GetAtomWithIdx(host).GetAtomicNum() != 6
            or not mol.GetAtomWithIdx(host).IsInRing()
            or set(group.atoms) != {carbonyl, oxo[0], bridge}):
        return None
    rings = ring_systems(mol)
    if len(rings) != 1 or host not in rings[0] or not 4 <= len(rings[0]) <= 12:
        return None
    ring = rings[0]
    if (sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in ring) > 1
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7}
                   or mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
            or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds()
                   if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= ring)):
        return None
    adjacency = {i: {a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors()
                     if a.GetIdx() in ring} for i in ring}
    elements = {i: mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring}
    topology = bicyclic_theta_frontier(adjacency, elements=elements)
    if topology.status != 'complete' or not topology.candidates:
        return None
    for i in ring:
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() == 7 and (atom.GetTotalValence() != 3
                or len(adjacency[i]) != 2 or atom.GetTotalNumHs(includeNeighbors=True) != 1):
            return None
        for neighbor in atom.GetNeighbors():
            n = neighbor.GetIdx()
            if n in ring or n == carbonyl:
                continue
            if (atom.GetAtomicNum() != 6 or neighbor.GetAtomicNum() not in {9, 17, 35, 53}
                    or neighbor.GetDegree() != 1
                    or mol.GetBondBetweenAtoms(i, n).GetBondTypeAsDouble() != 1):
                return None
    potential = tuple(Chem.FindPotentialStereo(mol))
    if (any(item.type != Chem.StereoType.Atom_Tetrahedral or item.centeredOn not in ring
            for item in potential)
            or any(int(a.GetChiralTag()) and a.GetIdx() not in ring for a in mol.GetAtoms())):
        return None
    roots = [a.GetIdx() for a in oxygen.GetNeighbors() if a.GetIdx() != carbonyl]
    if len(roots) != 1 or mol.GetAtomWithIdx(roots[0]).GetAtomicNum() != 6:
        return None
    organyl_scope = frozenset(collect_subtree(mol, roots[0], bridge))
    if (not organyl_scope or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
            or mol.GetAtomWithIdx(i).IsInRing() for i in organyl_scope)
            or any(b.GetBondTypeAsDouble() != 1 for b in mol.GetBonds()
                   if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= organyl_scope)):
        return None
    cuts = {(i, n.GetIdx()) for i in organyl_scope
            for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() not in organyl_scope}
    if cuts != {(roots[0], bridge)}:
        return None
    organyl, owned = mapped_branch(mol, Chem, roots[0], bridge)
    if not organyl or owned != organyl_scope:
        return None
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    suffix = frozenset({carbonyl, oxo[0]})
    organyl_atoms = organyl_scope | {bridge}
    ring_scope = heavy - suffix - organyl_atoms
    if (not ring <= ring_scope or ring_scope & (suffix | organyl_atoms)
            or ring_scope | suffix | organyl_atoms != heavy):
        return None
    candidates = records(mol, Chem, scope=ring_scope, attachment=host, core_filter=ring)
    if len(candidates) != 1:
        return None
    record = candidates[0]
    construction = record.construction
    if (not record.annotation_coverage or not record.hydrogen_budget_proved
            or record.preference_issues or construction is None
            or construction.atoms != ring or construction.atoms & construction.embedded_atoms
            or construction.atoms | construction.embedded_atoms != ring_scope
            or frozenset(record.parent_atoms) != ring or 'bicyclo[' not in record.parent):
        return None
    mapping = dict(construction.atom_locants)
    locant = mapping.get(host)
    source_numberings = {tuple(item.sequence) for item in topology.candidates}
    if (not locant or not locant.isdigit() or tuple(record.parent_atoms) not in source_numberings
            or int(locant) != min(dict(item.locants)[host] for item in topology.candidates)):
        return None
    # Every mapped outside ring atom must be a single proved halo prefix.
    branch_atoms, prefixes = set(), Counter()
    for root, anchor, atoms, text in construction.prefix_branch_ownership:
        if (anchor not in ring or root not in atoms or branch_atoms & atoms
                or atoms & ring or frozenset(collect_subtree(mol, root, anchor)) != atoms):
            return None
        rebuilt, rebuilt_atoms = mapped_branch(mol, Chem, root, anchor)
        if rebuilt != text or rebuilt_atoms != atoms:
            return None
        prefixes[(mapping[anchor], text)] += 1
        branch_atoms.update(atoms)
    if (branch_atoms != set(construction.embedded_atoms)
            or prefixes != Counter(construction.prefixes)):
        return None
    original_plan = _mapped_stereochemistry_plan(
        mol, Chem, construction, mapping, construction.parent_annotation_proof)
    component = replace(construction,
        parent=f'{record.parent}-{locant}-carboxylate',
        atoms=construction.atoms | suffix,
        atom_locants=construction.atom_locants + ((carbonyl, ''), (oxo[0], '')))
    new_plan = _mapped_stereochemistry_plan(
        mol, Chem, component, dict(component.atom_locants), component.parent_annotation_proof)
    tagged = {i for i in ring if int(mol.GetAtomWithIdx(i).GetChiralTag())}
    if (original_plan is None or new_plan != original_plan
            or {row[0] for row in original_plan[1]} != tagged):
        return None
    acid_name = decorate_component(mol, Chem, component)
    if not acid_name:
        return None
    issues = [SelectionLimitation(
        'bicyclic-carboxylate-parent', 'P-44/P-45/P-65.1.2.2.2',
        'The ring carboxylate and original stereochemistry are mapped; complete preferred-parent comparison remains unresolved.')]
    if any(item.specified != Chem.StereoSpecified.Specified for item in potential):
        issues.append(SelectionLimitation(
            'stereochemistry', 'P-91.2.1.2.1/P-91.2.2',
            'One or more potential stereogenic units have no source configuration.'))
    roles = (('parent-hydride', ring, construction.atom_locants),
             ('ring-substituents', frozenset(branch_atoms), ()),
             ('carboxylate', suffix, ((carbonyl, ''), (oxo[0], ''))),
             ('O-organyl', organyl_atoms, ()))
    return MethodDecision(organyl + ' ' + acid_name, 'functional-class',
        frozenset({carbonyl}), heavy,
        'P-23.2/P-23.3/P-44.1.2.2/P-65.1.2.2.2/P-65.6.3.2.1/P-65.6.3.3.1',
        component.atom_locants, component_roles=roles, preference_issues=tuple(issues))
