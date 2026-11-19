"""Bounded mapped P-46 chain proof for a carbamate N substituent."""
from __future__ import annotations


def internally_rooted_carbamate_branch(mol, Chem, root, blocked):
    """Return an original-graph owned internally rooted carbon prefix.

    The entire acyclic carbon skeleton must be one saturated, unbranched path.
    Off-path units are fully mapped ring substituents, hydroxy groups and,
    optionally, a terminal subordinate aldehyde oxygen.
    Carbon-path R/S annotations are preserved by the mapped P-46 writer;
    isotopes, enhanced stereo, other annotated branches and unproved
    characteristic priorities decline.
    """
    from .branch_rules import collect_subtree
    from .component_mapping import (
        _internally_rooted_carbon_prefix, mapped_branch, mapped_branch_ring_atoms,
    )
    from .extended_inventory import selection_inventory
    from .group_priority import compare_functional_group_priority
    from .stereochemistry import assigned_atom_stereochemistry

    if (mol.GetStereoGroups() or mol.GetAtomWithIdx(root).GetAtomicNum() != 6
            or mol.GetAtomWithIdx(root).IsInRing()
            or mol.GetAtomWithIdx(blocked).GetAtomicNum() != 7):
        return None
    cut = mol.GetBondBetweenAtoms(root, blocked)
    if cut is None or cut.GetBondTypeAsDouble() != 1:
        return None
    scope = frozenset(collect_subtree(mol, root, blocked))
    if not scope or root not in scope or blocked in scope:
        return None

    def one_cut(atoms, child, host):
        return {(i, n.GetIdx()) for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms} == {(child, host)}

    if not one_cut(scope, root, blocked):
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetFormalCharge()
           or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
           or any(n.GetAtomicNum() == 1 for n in mol.GetAtomWithIdx(i).GetNeighbors())
           for i in scope):
        return None
    if any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds()
           if b.GetBeginAtomIdx() in scope and b.GetEndAtomIdx() in scope):
        return None

    carbon, pending = set(), [root]
    while pending:
        idx = pending.pop()
        if idx in carbon:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if (idx not in scope or atom.GetAtomicNum() != 6 or atom.IsInRing()
                or atom.GetIsAromatic() or atom.GetTotalValence() != 4):
            return None
        carbon.add(idx)
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx in scope and neighbor.GetAtomicNum() == 6 and not neighbor.IsInRing():
                if mol.GetBondBetweenAtoms(idx, nidx).GetBondTypeAsDouble() != 1:
                    return None
                pending.append(nidx)
    adjacency = {i: {n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                     if n.GetIdx() in carbon} for i in carbon}
    if (len(carbon) < 2 or len(adjacency[root]) not in {1, 2}
            or any(len(neighbors) > 2 for neighbors in adjacency.values())
            or sum(map(len, adjacency.values())) != 2 * (len(carbon) - 1)):
        return None
    if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in scope - carbon):
        return None
    descriptors = dict(assigned_atom_stereochemistry(mol, Chem, include_pseudo=True))
    if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) and descriptors.get(i) not in {"R", "S"}
           for i in carbon):
        return None

    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.groups) != len(inventory.source_group_facts)
            or len(inventory.principal) != 1 or inventory.principal[0].kind != "carbamate"):
        return None
    principal = inventory.principal[0]
    if principal.center in scope or set(principal.atoms) & scope:
        return None
    facts = {id(g): f for g, f in zip(inventory.groups, inventory.source_group_facts)}
    for group in inventory.groups:
        if set(group.atoms) & scope:
            if (not set(group.atoms) <= scope or group.center not in scope
                    or compare_functional_group_priority(
                        facts[id(principal)], facts[id(group)]).status != "senior"):
                return None

    embedded = set()
    for host in carbon:
        for neighbor in mol.GetAtomWithIdx(host).GetNeighbors():
            child = neighbor.GetIdx()
            if child == blocked or child in carbon or neighbor.GetAtomicNum() == 1:
                continue
            atoms = frozenset(collect_subtree(mol, child, host))
            if (not atoms or atoms & (carbon | embedded) or not atoms <= scope
                    or not one_cut(atoms, child, host)):
                return None
            bond = mol.GetBondBetweenAtoms(host, child).GetBondTypeAsDouble()
            if neighbor.IsInRing() and bond == 1:
                text, owned = mapped_branch(mol, Chem, child, host)
                ring_atoms = frozenset(i for i in atoms if mol.GetAtomWithIdx(i).IsInRing())
                if (not text or frozenset(owned) != atoms or not ring_atoms
                        or mapped_branch_ring_atoms(mol, Chem, child, host, text, owned) != ring_atoms):
                    return None
            elif (neighbor.GetAtomicNum() == 8 and bond == 1 and atoms == {child}
                  and neighbor.GetDegree() == 1 and not neighbor.GetFormalCharge()
                  and neighbor.GetTotalNumHs(includeNeighbors=True) == 1):
                # The original, one-atom OH component is a subordinate
                # hydroxy prefix; class seniority was proved above.
                text, owned = mapped_branch(mol, Chem, child, host)
                if text != 'hydroxy' or frozenset(owned) != atoms:
                    return None
            elif (neighbor.GetAtomicNum() == 8 and bond == 2
                  and len(adjacency[host]) == 1 and atoms == {child}
                  and neighbor.GetDegree() == 1
                  and mol.GetAtomWithIdx(host).GetTotalNumHs(includeNeighbors=True) == 1
                  and any(g.kind == "aldehyde" and g.center == host and child in g.atoms
                          for g in inventory.groups)):
                # P-66.6.1.3 expresses this terminal aldehyde as oxo on the
                # selected carbon path; its priority was proved above.
                pass
            else:
                return None
            embedded.update(atoms)
    if not embedded or frozenset(carbon | embedded) != scope:
        return None
    name, owned = _internally_rooted_carbon_prefix(mol, Chem, root, blocked, scope, carbon)
    if not name or owned != scope:
        return None
    ring_atoms = frozenset(i for i in scope if mol.GetAtomWithIdx(i).IsInRing())
    if (ring_atoms and mapped_branch_ring_atoms(mol, Chem, root, blocked, name, owned)
            != ring_atoms):
        return None
    return name, scope
