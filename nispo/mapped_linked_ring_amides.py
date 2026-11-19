"""Mapped ring-carboxamide candidates with a linked ring-amido arm.

This is a general construction only: the bounded source comparison favors a
nitrogen-containing ring over an oxygen/sulfur ring, but neither global method
eligibility nor the complete parent frontier is asserted. All child prefixes
are proved on a private copy of the original graph before ring numbering.
"""
from collections import Counter
from dataclasses import replace

from .component_mapping import (_record_branch_ring_atoms, decorate_component,
    mapped_branch, mapped_branch_ring_atoms)
from .extended_inventory import selection_inventory
from .mapped_ring_parents import records, ring_systems
from .molecule_cache import mol_cache_dict, mol_cache_get, naming_cache_state
from .n_sulfonyl_ring_amides import _scope
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def _valid_record(mol, Chem, record, scope):
    component = record.construction
    if (component is None or not record.annotation_coverage or not record.hydrogen_budget_proved
            or component.atoms & component.embedded_atoms
            or component.atoms | component.embedded_atoms != scope):
        return False
    mapping, seen, observed = dict(component.atom_locants), set(), Counter()
    for root, host, atoms, text in component.prefix_branch_ownership:
        atoms = frozenset(atoms)
        if (host not in component.atoms or host not in mapping or seen & atoms
                or _scope(mol, root, host) != atoms
                or mapped_branch(mol, Chem, root, host) != (text, atoms)):
            return False
        rings = {idx for idx in atoms if mol.GetAtomWithIdx(idx).IsInRing()}
        if not rings <= mapped_branch_ring_atoms(mol, Chem, root, host, text, atoms):
            return False
        observed[(mapping[host], text)] += 1
        seen.update(atoms)
    return observed == Counter(component.prefixes) and seen == set(component.embedded_atoms)


def _install_proved_branch(mol, Chem, root, host, text, atoms):
    """Expose a graph-owned child to the existing complete ring numberer.

    The caller's source molecule is never changed: this runs only on a private
    copy after the child record's own ownership and annotation proof succeeds.
    """
    key = (id(Chem), root, host, naming_cache_state(mol),
           frozenset(mol_cache_get(mol, 'branch_prefix_in_progress', ())))
    mol_cache_dict(mol, 'mapped_branch')[key] = (text, atoms)
    _record_branch_ring_atoms(mol, Chem, root, host, text, atoms,
                              {idx for idx in atoms if mol.GetAtomWithIdx(idx).IsInRing()})


def _secondary_site(mol, group):
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
    return other[0]


def linked_ring_amide_decision(source_mol, Chem):
    """Return one completely owned, explicitly unresolved amide candidate."""
    mol = Chem.Mol(source_mol)
    if (mol.GetNumHeavyAtoms() > 100 or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) and a.GetAtomicNum() != 6 for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    potential = Chem.FindPotentialStereo(mol)
    if any(info.type != Chem.StereoType.Atom_Tetrahedral for info in potential):
        return None
    # OPSIN's support for pseudoasymmetric labels in these fused/bridged
    # substituents is not established by the bounded graph checks. Preserve
    # them through the existing fallback rather than silently dropping one.
    from .stereochemistry import assigned_atom_stereochemistry
    assigned = dict(assigned_atom_stereochemistry(mol, Chem, include_pseudo=True))
    tagged = {atom.GetIdx() for atom in mol.GetAtoms() if int(atom.GetChiralTag())}
    specified = {info.centeredOn for info in potential
                 if info.specified == Chem.StereoSpecified.Specified}
    if not tagged <= specified or any(assigned.get(idx) not in {'R', 'S'} for idx in tagged):
        return None
    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    groups = tuple(frontier.groups)
    if (not inventory.supported or frontier.status != 'tie' or frontier.unknown_pairs or frontier.cycle
            or len(groups) != 2 or any(group.kind != 'amide' for group in groups)):
        return None
    systems = ring_systems(mol)
    ring_for = {}
    for group in groups:
        cores = [core for core in systems if group.owner in core]
        if len(cores) != 1 or _secondary_site(mol, group) is None:
            return None
        ring_for[group.center] = cores[0]
    if ring_for[groups[0].center] & ring_for[groups[1].center]:
        return None
    n_groups = [group for group in groups if any(
        mol.GetAtomWithIdx(idx).GetAtomicNum() == 7 for idx in ring_for[group.center])]
    if len(n_groups) != 1:
        return None
    selected = n_groups[0]
    other = next(group for group in groups if group != selected)
    if not any(mol.GetAtomWithIdx(idx).GetAtomicNum() in {8, 16}
               for idx in ring_for[other.center]):
        return None
    central_root, linked_host = _secondary_site(mol, selected), _secondary_site(mol, other)
    central_cores = [core for core in systems if central_root in core]
    linked_cores = [core for core in systems if linked_host in core]
    if len(central_cores) != 1 or len(linked_cores) != 1:
        return None
    central_core, linked_core = central_cores[0], linked_cores[0]
    if (central_core & linked_core or central_core & ring_for[selected.center]
            or linked_core & ring_for[other.center]):
        return None
    joins = [(a, b) for a in linked_core for b in central_core
             if mol.GetBondBetweenAtoms(a, b) is not None
             and mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() == 1]
    if len(joins) != 1:
        return None
    linked_root, central_host = joins[0]
    selected_scope = _scope(mol, selected.center, selected.ligand) | {selected.ligand}
    other_scope = _scope(mol, other.center, other.ligand) | {other.ligand}
    central_scope = _scope(mol, central_root, selected.ligand)
    linked_scope = _scope(mol, linked_root, central_host)
    other_cut = _scope(mol, other.ligand, linked_host)
    whole = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if (not central_scope or not linked_scope or other_scope != other_cut
            or selected_scope & central_scope or selected_scope | central_scope != whole
            or not other_scope <= linked_scope or not linked_scope <= central_scope):
        return None
    selected_records = records(mol, Chem, scope=selected_scope, principal=(selected,))
    other_records = records(mol, Chem, scope=other_scope, principal=(other,))
    if (len(selected_records) != 1 or len(other_records) != 1
            or not _valid_record(mol, Chem, selected_records[0], selected_scope)
            or not _valid_record(mol, Chem, other_records[0], other_scope)):
        return None
    primary, subordinate = selected_records[0], other_records[0]
    if (primary.principal_centers != {selected.center} or subordinate.principal_centers != {other.center}
            or dict(primary.construction.atom_locants).get(selected.ligand) != 'N'
            or dict(subordinate.construction.atom_locants).get(other.ligand) != 'N'
            or not subordinate.construction.parent.endswith('amide')):
        return None
    amido = decorate_component(mol, Chem, replace(subordinate.construction,
        parent=subordinate.construction.parent[:-1] + 'o'))
    if not amido:
        return None
    _install_proved_branch(mol, Chem, other.ligand, linked_host, amido, other_scope)
    linked_records = records(mol, Chem, scope=linked_scope, attachment=linked_root, core_filter=linked_core)
    if len(linked_records) != 1 or not _valid_record(mol, Chem, linked_records[0], linked_scope):
        return None
    linked = linked_records[0]
    _install_proved_branch(mol, Chem, linked_root, central_host, linked.name, linked_scope)
    central_records = records(mol, Chem, scope=central_scope, attachment=central_root, core_filter=central_core)
    if len(central_records) != 1 or not _valid_record(mol, Chem, central_records[0], central_scope):
        return None
    central = central_records[0]
    final = replace(primary.construction,
        prefixes=primary.construction.prefixes + (('N', central.name),),
        embedded_atoms=primary.construction.embedded_atoms | central_scope,
        prefix_branch_ownership=primary.construction.prefix_branch_ownership +
            ((central_root, selected.ligand, central_scope, central.name),))
    name = decorate_component(mol, Chem, final)
    if not name:
        return None
    direct = (primary.construction.atoms, central.construction.atoms,
              linked.construction.atoms, subordinate.construction.atoms)
    if sum(map(len, direct)) != len(frozenset().union(*direct)):
        return None
    if any(info.centeredOn not in frozenset().union(*direct) for info in potential):
        return None
    rest = whole - frozenset().union(*direct)
    roles = (('selected-ring-amide', direct[0], primary.construction.atom_locants),
             ('central-ring', direct[1], central.construction.atom_locants),
             ('linked-ring', direct[2], linked.construction.atom_locants),
             ('subordinate-ring-amide', direct[3], subordinate.construction.atom_locants),
             ('mapped-substituents', rest, ()))
    issues = [SelectionLimitation('equal-rank-ring-amide-method', 'P-44.2.1/P-45/P-51.3',
        'The linked ring-carboxamide construction is mapped; the complete equal-rank parent and method universe remains unresolved.')]
    for record in (primary, subordinate, linked, central):
        for issue in record.preference_issues:
            if issue not in issues:
                issues.append(issue)
    if any(info.specified != Chem.StereoSpecified.Specified for info in potential):
        issues.append(SelectionLimitation('stereochemistry', 'P-91.2.1.2.1/P-91.2.2',
            'One or more source stereogenic units have no specified configuration.'))
    return MethodDecision(name, 'substitutive', frozenset({selected.center}), whole,
        'P-41/P-44.2.1/P-45/P-66.1.1.1.1.3/P-66.1.1.4.3',
        primary.construction.atom_locants, component_roles=roles, preference_issues=tuple(issues))
