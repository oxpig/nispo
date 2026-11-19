"""A mapped general continuation for N-sulfonyl ring carboxamides.

The source graph must select a single senior carboxamide. The parent ring and
the N-sulfonyl arm are mapped separately on the original atoms; no capped
atom, finished-name rewrite, or preferred-parent claim is involved.
"""
from collections import Counter
from dataclasses import replace

from .branch_rules import collect_subtree
from .component_mapping import decorate_component, mapped_branch, mapped_branch_ring_atoms
from .extended_inventory import selection_inventory
from .mapped_ring_parents import records, ring_systems
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation
from .neutral_charge_components import validate_neutral_charge_role_proofs


def _scope(mol, root, blocked):
    atoms = frozenset(collect_subtree(mol, root, blocked))
    cuts = {(i, neighbor.GetIdx()) for i in atoms
            for neighbor in mol.GetAtomWithIdx(i).GetNeighbors()
            if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in atoms}
    return atoms if atoms and cuts == {(root, blocked)} else frozenset()


def _proved_record(mol, Chem, record, scope):
    component = record.construction
    if (component is None or not record.annotation_coverage
            or not record.hydrogen_budget_proved or record.preference_issues
            or component.atoms & component.embedded_atoms
            or component.atoms | component.embedded_atoms != scope):
        return False
    mapping = dict(component.atom_locants)
    found, observed = set(), Counter()
    for root, host, atoms, text in component.prefix_branch_ownership:
        atoms = frozenset(atoms)
        if (host not in component.atoms or host not in mapping or root not in atoms
                or found & atoms or _scope(mol, root, host) != atoms
                or mapped_branch(mol, Chem, root, host) != (text, atoms)):
            return False
        ring_atoms = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()}
        if not ring_atoms <= mapped_branch_ring_atoms(mol, Chem, root, host, text, atoms):
            return False
        observed[(mapping[host], text)] += 1
        found.update(atoms)
    return found == set(component.embedded_atoms) and observed == Counter(component.prefixes)


def _sulfonyl(mol, Chem, sulfur, nitrogen):
    atom = mol.GetAtomWithIdx(sulfur)
    cut = mol.GetBondBetweenAtoms(sulfur, nitrogen)
    if (atom.GetAtomicNum() != 16 or atom.IsInRing() or atom.GetFormalCharge()
            or atom.GetTotalNumHs(includeNeighbors=True) or cut is None
            or cut.GetBondTypeAsDouble() != 1):
        return None
    scope = _scope(mol, sulfur, nitrogen)
    neighbors = [a for a in atom.GetNeighbors() if a.GetIdx() != nitrogen and a.GetAtomicNum() > 1]
    oxygens = [a.GetIdx() for a in neighbors if a.GetAtomicNum() == 8
               and mol.GetBondBetweenAtoms(sulfur, a.GetIdx()).GetBondTypeAsDouble() == 2
               and a.GetDegree() == 1 and not a.GetFormalCharge()]
    carbons = [a.GetIdx() for a in neighbors if a.GetAtomicNum() == 6
               and mol.GetBondBetweenAtoms(sulfur, a.GetIdx()).GetBondTypeAsDouble() == 1]
    if len(neighbors) != 3 or len(oxygens) != 2 or len(carbons) != 1 or not scope:
        return None
    carbon = carbons[0]
    ligand_scope = _scope(mol, carbon, sulfur)
    if not ligand_scope or ligand_scope | {sulfur, *oxygens} != scope:
        return None
    if not mol.GetAtomWithIdx(carbon).IsInRing():
        from .branch.sulfonyl_components import mapped_sulfonyl_prefix
        result = mapped_sulfonyl_prefix(mol, Chem, sulfur, nitrogen, scope)
        return result if result and result[0] and result[1] == scope else None
    systems = [system for system in ring_systems(mol) if carbon in system]
    if len(systems) != 1:
        return None
    core = systems[0]
    bonds = [bond for bond in mol.GetBonds()
             if bond.GetBeginAtomIdx() in core and bond.GetEndAtomIdx() in core]
    if (len(core) < 3 or len(core) > 8 or len(bonds) != len(core)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   or sum(n.GetIdx() in core for n in mol.GetAtomWithIdx(i).GetNeighbors()) != 2
                   for i in core)):
        return None
    candidates = records(mol, Chem, scope=ligand_scope, attachment=carbon, core_filter=core)
    if len(candidates) != 1 or not _proved_record(mol, Chem, candidates[0], ligand_scope):
        return None
    record = candidates[0]
    if dict(record.construction.atom_locants).get(carbon) != '1':
        return None
    if (record.parent == 'benzene' and len(core) == 6
            and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in core)
            and all(bond.GetIsAromatic() for bond in bonds)):
        parent = 'benzenesulfonyl'
    elif all(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in core) and all(
            bond.GetBondTypeAsDouble() == 1 for bond in bonds):
        from .constants import ROOTS
        if len(core) not in ROOTS or record.parent != 'cyclo' + ROOTS[len(core)] + 'ane':
            return None
        parent = record.parent + 'sulfonyl'
    else:
        return None
    text = decorate_component(mol, Chem, replace(record.construction, parent=parent))
    return (text, scope) if text else None


def n_sulfonyl_ring_carboxamide_decision(mol, Chem):
    """Build one complete mapped candidate, keeping method preference open."""
    if (mol.GetNumHeavyAtoms() > 100 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups() or Chem.FindPotentialStereo(mol)
            or any(a.GetAtomicNum() == 1 or a.GetIsotope()
                   or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
                   for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    if (not inventory.supported or frontier.status != 'selected' or frontier.unknown_pairs
            or frontier.cycle or len(frontier.groups) != 1
            or frontier.groups[0].kind != 'amide'
            or not validate_neutral_charge_role_proofs(mol, inventory.charge_role_proofs)):
        return None
    group = frontier.groups[0]
    carbon, nitrogen = mol.GetAtomWithIdx(group.center), mol.GetAtomWithIdx(group.ligand)
    if (carbon.IsInRing() or carbon.GetDegree() != 3 or nitrogen.GetAtomicNum() != 7
            or nitrogen.IsInRing() or nitrogen.GetFormalCharge() or nitrogen.GetTotalValence() != 3
            or nitrogen.GetTotalNumHs(includeNeighbors=True) != 1
            or not mol.GetAtomWithIdx(group.owner).IsInRing()):
        return None
    heavy = [a.GetIdx() for a in nitrogen.GetNeighbors() if a.GetAtomicNum() > 1]
    sulfur = [i for i in heavy if i != group.center and mol.GetAtomWithIdx(i).GetAtomicNum() == 16]
    if len(heavy) != 2 or len(sulfur) != 1:
        return None
    sulfur = sulfur[0]
    sulfonamides = [g for g in inventory.groups if g.kind == 'sulfonamide'
                    and g.center == sulfur and g.ligand == group.ligand]
    if len(sulfonamides) != 1 or group.seniority >= sulfonamides[0].seniority:
        return None
    acid_scope = _scope(mol, group.center, group.ligand)
    sulfur_scope = _scope(mol, sulfur, group.ligand)
    parent_scope = acid_scope | {group.ligand}
    if not acid_scope or not sulfur_scope or parent_scope & sulfur_scope:
        return None
    whole = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if parent_scope | sulfur_scope != whole:
        return None
    candidates = records(mol, Chem, scope=parent_scope, principal=(group,))
    if len(candidates) != 1 or not _proved_record(mol, Chem, candidates[0], parent_scope):
        return None
    record = candidates[0]
    component = record.construction
    if (record.principal_centers != {group.center}
            or dict(component.atom_locants).get(group.ligand) != 'N'):
        return None
    prefix = _sulfonyl(mol, Chem, sulfur, group.ligand)
    if prefix is None or prefix[1] != sulfur_scope:
        return None
    name = decorate_component(mol, Chem, replace(component,
        prefixes=component.prefixes + (('N', prefix[0]),),
        embedded_atoms=component.embedded_atoms | sulfur_scope,
        prefix_branch_ownership=component.prefix_branch_ownership +
            ((sulfur, group.ligand, sulfur_scope, prefix[0]),)))
    if not name:
        return None
    roles = (('ring-carboxamide', parent_scope, component.atom_locants),
             ('N-sulfonyl', sulfur_scope, ()))
    issues = (SelectionLimitation('ring-carboxamide-parent', 'P-44/P-45/P-66.1.2',
        'The carboxamide and N-sulfonyl components are completely mapped; whole-parent and method preference remain unresolved.'),)
    return MethodDecision(name, 'substitutive', frozenset({group.center}), whole,
        'P-41/P-44/P-45/P-65.3.2.2.2/P-66.1.1.1.1.3/P-66.1.2',
        component.atom_locants, component_roles=roles, preference_issues=issues)
