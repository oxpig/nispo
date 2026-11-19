"""Compose full-mode names around proved neutral azido substituents."""

from .molecule_cache import mol_cache_set
from .molecule_iteration import iter_atoms


def nested_azide_decision(mol, Chem):
    """Retry mapped principal-group construction after ordinary naming fails."""
    if mol.GetStereoGroups() or not any(
            atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1 and atom.GetDegree() == 2
            for atom in iter_atoms(mol)):
        return None
    from .extended_inventory import selection_inventory
    from .neutral_charge_components import neutral_charge_role_proofs

    proofs = neutral_charge_role_proofs(mol)
    if not proofs or any(proof.kind != 'azido' for proof in proofs):
        return None
    inventory = selection_inventory(mol)
    if not inventory.supported or not inventory.principal:
        return None
    # Earlier writers have memoized failed branches on the original graph.
    # Restrict the new branch resolver to this private copy and its readers.
    graph = Chem.Mol(mol)
    mol_cache_set(graph, 'nested_azido_sites', {proof.atoms[0]: proof for proof in proofs})
    if inventory.principal[0].kind in {'ester', 'carbamate'}:
        from .mapped_ester_general import mapped_ester_decision

        decision = mapped_ester_decision(graph, Chem)
        if decision is not None:
            return decision
        if any(group.kind == 'phosphate-acid-ester' for group in inventory.groups):
            from .oxoacid_ester_methods import oxoacid_ester_decision

            return oxoacid_ester_decision(graph, Chem)
        return None
    if inventory.principal[0].kind == 'pseudoketone':
        from .hidden_amide_pseudoketones import hidden_amide_pseudoketone_decision

        return hidden_amide_pseudoketone_decision(graph, Chem)
    if len(inventory.principal) == 1 and inventory.principal[0].kind == 'sulfamide':
        from .sulfuric_diamide import mapped_annular_sulfamide_decision

        return mapped_annular_sulfamide_decision(graph, Chem, inventory.principal[0])
    from .mapped_principal_general import mapped_principal_decision

    return mapped_principal_decision(graph, Chem)


def compose_azido_branch(mol, Chem, root, host, scope, sites):
    """Use complete rooted components on the original side of a single cut."""
    from .component_mapping import (
        _complete_rooted_ring, _general_branch_stereo_is_cited,
        _mapped_branch_recursive, _record_branch_ring_atoms,
        mapped_branch_ring_atoms, mapped_general_branch,
    )

    atom = mol.GetAtomWithIdx(root)
    direct = sites.get(root)
    if direct is not None:
        if direct.attachment_atom == host and scope == frozenset(direct.atoms):
            return 'azido', scope
        return '', frozenset()
    if (mol.GetBondBetweenAtoms(root, host).GetBondTypeAsDouble() != 1
            or atom.GetFormalCharge() or any(mol.GetAtomWithIdx(i).GetIsotope() for i in scope)):
        return '', frozenset()
    if atom.IsInRing():
        name = _complete_rooted_ring(mol, Chem, root, scope)
        if name and _general_branch_stereo_is_cited(mol, Chem, scope, name):
            _record_branch_ring_atoms(mol, Chem, root, host, name, scope,
                (i for i in scope if mol.GetAtomWithIdx(i).IsInRing()))
            return name, scope
        return '', frozenset()
    if atom.GetAtomicNum() in {6, 8}:
        name, owned = _mapped_branch_recursive(
            mol, Chem, root, host, scope, general_branches=True)
        if name and owned == scope and _general_branch_stereo_is_cited(mol, Chem, scope, name):
            return name, owned
        return '', frozenset()
    if (atom.GetAtomicNum() != 7 or atom.GetTotalValence() != 3 or atom.GetChiralTag()
            or any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds())):
        return '', frozenset()
    children = [n for n in atom.GetNeighbors() if n.GetIdx() != host]
    if not children or any(n.GetAtomicNum() != 6 for n in children):
        return '', frozenset()
    parts, owned, ring_atoms = [], {root}, set()
    for child in children:
        idx = child.GetIdx()
        text, atoms = mapped_general_branch(mol, Chem, idx, root)
        if not text or owned & atoms or not atoms <= scope:
            return '', frozenset()
        parts.append(text)
        owned.update(atoms)
        ring_atoms.update(mapped_branch_ring_atoms(mol, Chem, idx, root, text, atoms))
    if owned != scope:
        return '', frozenset()
    from .branch.hetero_atoms import _amino_prefix_from_ligands

    name = _amino_prefix_from_ligands(mol, root, host, parts)
    if name and _general_branch_stereo_is_cited(mol, Chem, scope, name):
        _record_branch_ring_atoms(mol, Chem, root, host, name, scope, ring_atoms)
        return name, scope
    return '', frozenset()
