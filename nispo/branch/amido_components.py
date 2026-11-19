"""Mapped amide-derived prefixes subordinate to a senior characteristic group.

P-66.1.1.4.3 prefers the complete amide-derived ``amido`` construction to
``acylamino`` when a more senior principal characteristic group is present.
Only graph-owned neutral carboxamide units with a separately recognized
senior group are admitted.
"""
from __future__ import annotations

from dataclasses import replace


def mapped_amido_prefix(mol, Chem, root, blocked, atoms, *, chain_only=False, ring_only=False):
    """Return a complete, singly attached amido unit, or ``None``.

    The acyl parent is either a mapped carbon chain or
    an original-graph mapped ring carboxamide. The chain owns its assigned
    tetrahedral stereochemistry and each singly attached mapped prefix.
    """
    from .functional_groups import collect_subtree
    nitrogen = mol.GetAtomWithIdx(root)
    cut = mol.GetBondBetweenAtoms(root, blocked)
    if (nitrogen.GetAtomicNum() != 7 or nitrogen.IsInRing()
            or nitrogen.GetFormalCharge() or nitrogen.GetNumRadicalElectrons()
            or nitrogen.GetIsotope() or int(nitrogen.GetChiralTag())
            or mol.GetAtomWithIdx(blocked).GetAtomicNum() != 6
            or cut is None or cut.GetBondTypeAsDouble() != 1):
        return None
    children = [a for a in nitrogen.GetNeighbors()
                if a.GetIdx() != blocked and a.GetAtomicNum() > 1]
    if (len(children) not in {1, 2}
            or nitrogen.GetTotalNumHs(includeNeighbors=True) != 2 - len(children)
            or any(a.GetAtomicNum() != 6 or mol.GetBondBetweenAtoms(
                root, a.GetIdx()).GetBondTypeAsDouble() != 1 for a in children)):
        return None
    acyls = [a for a in children if any(
        b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(a).GetAtomicNum() == 8
        for b in a.GetBonds())]
    if len(acyls) != 1:
        return None
    carbonyl = acyls[0]
    cidx = carbonyl.GetIdx()
    if (carbonyl.IsInRing() or carbonyl.GetFormalCharge()
            or mol.GetBondBetweenAtoms(root, cidx).GetBondTypeAsDouble() != 1):
        return None
    other = [a for a in carbonyl.GetNeighbors()
             if a.GetIdx() != root and a.GetAtomicNum() > 1]
    oxy = [a for a in other if a.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(cidx, a.GetIdx()).GetBondTypeAsDouble() == 2]
    ligands = [a for a in other if a.GetAtomicNum() == 6
               and mol.GetBondBetweenAtoms(cidx, a.GetIdx()).GetBondTypeAsDouble() == 1]
    formyl = (len(oxy) == 1 and not ligands and len(other) == 1
              and carbonyl.GetTotalNumHs(includeNeighbors=True) == 1)
    if not formyl and (len(oxy) != 1 or len(ligands) != 1 or len(other) != 2):
        return None
    ring_ligand = bool(ligands and ligands[0].IsInRing())
    n_organyl = next((a.GetIdx() for a in children if a.GetIdx() != cidx), None)
    if (ring_ligand and (chain_only or n_organyl is not None)) or (ring_only and not ring_ligand):
        return None
    scope = frozenset(atoms)
    if scope != frozenset(collect_subtree(mol, root, blocked)):
        return None
    acyl_atoms = frozenset(collect_subtree(mol, cidx, root))
    ligand_atoms = (frozenset(collect_subtree(mol, ligands[0].GetIdx(), cidx))
                    if ligands else frozenset())
    n_atoms = (frozenset(collect_subtree(mol, n_organyl, root))
               if n_organyl is not None else frozenset())
    if (scope != acyl_atoms | {root} | n_atoms or n_atoms & acyl_atoms
            or acyl_atoms != ligand_atoms | {cidx, oxy[0].GetIdx()}):
        return None
    from ..component_mapping import has_annotations
    if (any(mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in scope)
            or any(a.GetAtomicNum() == 1 and a.GetIsotope()
                   for i in scope for a in mol.GetAtomWithIdx(i).GetNeighbors())):
        return None
    if has_annotations(mol, scope):
        if (mol.GetStereoGroups()
                or any(mol.GetAtomWithIdx(i).GetIsotope() for i in scope)
                or (ring_ligand and any(int(bond.GetStereo()) or int(bond.GetBondDir())
                                       for bond in mol.GetBonds()
                                       if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & scope))):
            return None
        if ring_ligand:
            from ..mapped_ring_parents import ring_systems
            core = next((ring for ring in ring_systems(mol)
                         if ligands[0].GetIdx() in ring), frozenset())
            for idx in scope:
                atom = mol.GetAtomWithIdx(idx)
                if int(atom.GetChiralTag()) and (
                        idx not in core or atom.GetAtomicNum() != 6
                        or atom.GetChiralTag() not in {
                            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                            Chem.ChiralType.CHI_TETRAHEDRAL_CCW}):
                    return None
    # Only a separately owned, strictly more senior P-41 class justifies this
    # local choice. Do not infer that a second equal-rank amide is principal.
    from ..functional_inventory import functional_inventory
    inventory = functional_inventory(mol)
    if ring_ligand and not inventory.supported:
        from ..extended_inventory import selection_inventory
        inventory = selection_inventory(mol)
    if not inventory.supported:
        return None
    target = [g for g in inventory.groups if g.kind == "amide" and g.center == cidx]
    if len(target) != 1:
        return None
    senior = inventory.principal
    if not senior or any(g.seniority >= target[0].seniority or set(g.atoms) & scope
                         for g in senior):
        return None
    if ring_ligand:
        text = _ring_amido(mol, Chem, cidx, ligands[0].GetIdx(), ligand_atoms)
    else:
        text = _chain_amido(mol, Chem, cidx, root, blocked, oxy[0].GetIdx(), scope,
                            ligands[0].GetIdx() if ligands else None, n_organyl)
    return (text, scope) if text else None


def _chain_amido(mol, Chem, carbonyl, nitrogen, blocked, oxo, scope, acyl_owner, n_organyl):
    from ..component_mapping import _record_branch_ring_atoms
    from ..parent_rules import _ester_acyl_parent_paths

    paths = ([(carbonyl,)] if acyl_owner is None else
             [(carbonyl, *path) for path in _ester_acyl_parent_paths(mol, Chem, carbonyl, acyl_owner)])
    if not paths:
        return ""
    candidates = [_chain_amido_on_path(mol, Chem, path, nitrogen, oxo, scope, n_organyl)
                  for path in paths]
    if any(row is None for row in candidates) or len({row[0] for row in candidates}) != 1:
        return ""
    text = candidates[0][0]
    witnessed = frozenset.intersection(*(row[1] for row in candidates))
    _record_branch_ring_atoms(mol, Chem, nitrogen, blocked, text, scope, witnessed)
    return text


def _chain_amido_on_path(mol, Chem, path, nitrogen, oxo, scope, n_organyl):
    from ..branch_rules import parent_chain_name
    from ..component_mapping import (
        MappedComponent, decorate_component, mapped_branch, mapped_branch_ring_atoms,
    )
    from ..constants import ROOTS
    from ..formatting import attach_suffix
    from ..retained_acid_rules import mapped_ring_stereo_prefix_proved
    from ..stereochemistry import assigned_atom_stereochemistry, assigned_bond_stereochemistry

    if len(path) not in ROOTS:
        return None
    core = frozenset({nitrogen, oxo, *path})
    tagged = {i for i in scope if int(mol.GetAtomWithIdx(i).GetChiralTag())}
    if tagged:
        assigned = dict(assigned_atom_stereochemistry(mol, Chem, include_pseudo=True))
        if any(assigned.get(i) not in {"R", "S"} for i in tagged):
            return None
    geometry = {frozenset((a, b)) for a, b, _ in assigned_bond_stereochemistry(mol, Chem)
                if {a, b} & scope}
    if any(not ends <= scope for ends in geometry):
        return None
    embedded = set()
    prefixes = []
    ownership = []
    rings = set()
    branch_stereo = set()
    branch_geometry = set()
    if n_organyl is not None:
        text, atoms = mapped_branch(mol, Chem, n_organyl, nitrogen)
        if (not text or not atoms <= scope or atoms & (core | tagged)
                or any(ends & atoms for ends in geometry)):
            return None
        prefixes.append(("N", text))
        ownership.append((n_organyl, nitrogen, atoms, text))
        embedded.update(atoms)
        rings.update(mapped_branch_ring_atoms(mol, Chem, n_organyl, nitrogen, text, atoms))
    for locant, idx in enumerate(path, 1):
        atom = mol.GetAtomWithIdx(idx)
        if atom.IsInRing() or atom.GetIsAromatic():
            return None
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if neighbor.GetAtomicNum() == 1 or nidx in core:
                continue
            bond = mol.GetBondBetweenAtoms(idx, nidx)
            if bond.GetBondTypeAsDouble() != 1:
                return None
            prefix, atoms = mapped_branch(mol, Chem, nidx, idx)
            if not prefix or not atoms <= scope or atoms & (core | embedded):
                return None
            ring_atoms = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()}
            witnessed = mapped_branch_ring_atoms(mol, Chem, nidx, idx, prefix, atoms)
            if not ring_atoms <= witnessed:
                return None
            rings.update(witnessed)
            child_geometry = {ends for ends in geometry if ends & atoms}
            child_stereo = tagged & atoms
            if child_stereo or child_geometry:
                if neighbor.GetAtomicNum() == 7:
                    if mapped_amido_prefix(mol, Chem, nidx, idx, atoms) != (prefix, atoms):
                        return None
                elif child_geometry or not mapped_ring_stereo_prefix_proved(
                        mol, Chem, nidx, idx, atoms, prefix):
                    return None
                branch_stereo.update(child_stereo)
                branch_geometry.update(child_geometry)
            prefixes.append((str(locant), prefix))
            ownership.append((nidx, idx, atoms, prefix))
            embedded.update(atoms)
    if (core | embedded != scope or tagged - set(path) != branch_stereo
            or any(not ends <= core and ends not in branch_geometry for ends in geometry)):
        return None
    if len(path) == 1:
        return ("formamido", frozenset()) if not prefixes else None
    if len(path) == 2:
        # Substitution on the methyl group of retained acetamide needs no
        # locant; no substitution on its carbonyl carbon is admitted here.
        if any(loc not in {"2", "N"} for loc, _ in prefixes):
            return None
    parent = "acetamido" if len(path) == 2 else attach_suffix(parent_chain_name(mol, path), "amido")
    locants = tuple((idx, str(i)) for i, idx in enumerate(path, 1))
    name = decorate_component(mol, Chem, MappedComponent(
        parent, (*locants, (nitrogen, "N"), (oxo, "O")), core,
        tuple(prefixes), frozenset(embedded), systematic_parent=True,
        omit_methyl_prefix_locant=len(path) == 2 and n_organyl is None,
        prefix_branch_ownership=tuple(ownership),
    ))
    return (name, frozenset(rings)) if name else None


def _ring_amido(mol, Chem, carbonyl, ligand, ligand_atoms):
    from ..component_mapping import decorate_component
    from ..mapped_ring_parents import records, ring_systems
    from .preferred_acyl import ring_acid_derivative_parent

    core = next((x for x in ring_systems(mol) if ligand in x), None)
    if core is None or not core <= ligand_atoms:
        return ""
    candidates = records(mol, Chem, scope=ligand_atoms, attachment=ligand, core_filter=core)
    if len(candidates) != 1:
        return ""
    record = candidates[0]
    component = record.construction
    if (component is None or not record.annotation_coverage or not record.hydrogen_budget_proved
            or frozenset(record.parent_atoms) != core
            or component.atoms | component.embedded_atoms != ligand_atoms):
        return ""
    parent = ring_acid_derivative_parent(record, ligand, 'carboxamido')
    return decorate_component(mol, Chem, replace(component, parent=parent)) if parent else ''


def mapped_amido_ring_atoms(mol, Chem, root, blocked, atoms, text):
    """Rejoin every witnessed ring of an already constructed ring amido unit.

    The amido writer owns a ring-derived carbonyl and may itself contain
    additional cyclic substituents.  Only propagate those nested witnesses
    when the selected ring construction and each original one-cut branch
    reproduce the exact emitted component and collectively cover all rings.
    """
    from ..component_mapping import mapped_branch, mapped_branch_ring_atoms
    from ..mapped_ring_parents import records, ring_systems
    from .functional_groups import collect_subtree

    scope = frozenset(atoms)
    if not text or root not in scope or blocked in scope:
        return None
    nitrogen = mol.GetAtomWithIdx(root)
    acyl = [a for a in nitrogen.GetNeighbors() if a.GetIdx() != blocked
            and a.GetAtomicNum() == 6 and any(
                b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(a).GetAtomicNum() == 8
                for b in a.GetBonds())]
    if len(acyl) != 1:
        return None
    carbonyl = acyl[0].GetIdx()
    ligands = [a.GetIdx() for a in acyl[0].GetNeighbors()
               if a.GetIdx() != root and a.GetAtomicNum() == 6 and a.IsInRing()
               and mol.GetBondBetweenAtoms(carbonyl, a.GetIdx()).GetBondTypeAsDouble() == 1]
    if len(ligands) != 1:
        return None
    ligand = ligands[0]
    ligand_scope = frozenset(collect_subtree(mol, ligand, carbonyl))
    cores = [ring for ring in ring_systems(mol) if ligand in ring]
    if len(cores) != 1 or not cores[0] <= ligand_scope:
        return None
    rows = records(mol, Chem, scope=ligand_scope, attachment=ligand, core_filter=cores[0])
    if len(rows) != 1:
        return None
    row = rows[0]
    component = row.construction
    if (component is None or row.preference_issues or not row.annotation_coverage
            or not row.hydrogen_budget_proved or frozenset(row.parent_atoms) != cores[0]
            or component.atoms != cores[0]
            or component.atoms & component.embedded_atoms
            or component.atoms | component.embedded_atoms != ligand_scope
            or _ring_amido(mol, Chem, carbonyl, ligand, ligand_scope) != text):
        return None
    witnessed = set(cores[0])
    claimed = set()
    for child, host, owned, child_text in component.prefix_branch_ownership:
        owned = frozenset(owned)
        if (host not in component.atoms or claimed & owned
                or mapped_branch(mol, Chem, child, host) != (child_text, owned)):
            return None
        claimed.update(owned)
        witnessed.update(mapped_branch_ring_atoms(mol, Chem, child, host, child_text, owned))
    if claimed != set(component.embedded_atoms):
        return None
    rings = {i for i in scope if mol.GetAtomWithIdx(i).IsInRing()}
    return frozenset(rings) if rings <= witnessed else None
