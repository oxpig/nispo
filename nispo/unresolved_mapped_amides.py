"""A bounded general-name continuation for equivalent mapped amide parents.

This does not resolve P-45.3 or prove the preferred method. It only exposes a
complete original-graph construction when every source-maximal proposal is
mapped, renders identically, and differs by an automorphism preserving all
recorded parent, suffix and branch roles. Missing evidence retains the
existing fallback.
"""
from collections import Counter


def equivalent_mapped_amide_general(mol, Chem, inventory, frontier):
    """Return one graph-proved general candidate, or decline conservatively."""
    from .branch_rules import collect_subtree
    from .component_mapping import MappedComponent, fully_covered, mapped_branch
    from .parent_selection import compare_parents
    from .selection_evidence import proposal_identity

    if (frontier.status != 'conflict'
            or frontier.reason != 'surviving-parent-comparison-is-unknown'
            or not inventory.supported or not inventory.principal
            or any(group.kind != 'amide' for group in inventory.principal)
            or mol.GetNumHeavyAtoms() > 100 or len(Chem.GetMolFrags(mol)) != 1):
        return None
    rows = tuple(frontier.compared_candidates)
    complete = frontier.candidate_completeness
    if (not 2 <= len(rows) <= 8 or complete is None or not complete.complete
            or set(complete.expected) != set(complete.accounted)
            or set(complete.expected) != set(frontier.maximal_skeletons)
            or set(complete.proposal_keys) != {proposal_identity(row) for row in rows}
            or not set(frontier.candidates) <= set(rows)
            or len({row.name for row in rows}) != 1 or not rows[0].name):
        return None
    # Reject annotations and potential stereo: this continuation has no
    # source-mapped stereo or isotope ownership proof.
    if (mol.GetStereoGroups() or Chem.FindPotentialStereo(mol)
            or any(atom.GetIsotope() or atom.GetFormalCharge()
                   or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag())
                   for atom in mol.GetAtoms())
            or any(int(bond.GetStereo()) or int(bond.GetBondDir()) for bond in mol.GetBonds())):
        return None
    centers = frozenset(group.center for group in inventory.principal)
    for row in rows:
        component = row.construction
        if (not isinstance(component, MappedComponent)
                or row.suffix_mode not in {'embedded', 'chain-included'}
                or row.principal_centers != centers or row.omitted_group_centers
                or not centers <= frozenset(row.parent_atoms)
                or any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6
                       or mol.GetAtomWithIdx(idx).IsInRing() for idx in row.parent_atoms)
                or component.atoms & component.embedded_atoms
                or not fully_covered(mol, component.atoms, component.embedded_atoms)
                or dict(component.atom_locants) != dict(row.atom_locants)
                or set(dict(component.atom_locants)) != set(component.atoms)
                or tuple(component.prefixes) != tuple(row.prefixes)
                or compare_parents(row.parent_selection, row.parent_selection).rule != 'P-45.3'):
            return None
        locants = dict(row.atom_locants)
        recorded = Counter()
        for root, host, atoms, text in component.prefix_branch_ownership:
            if (host not in component.atoms or root not in atoms
                    or mol.GetBondBetweenAtoms(root, host) is None
                    or frozenset(collect_subtree(mol, root, host)) != frozenset(atoms)):
                return None
            rebuilt, scope = mapped_branch(mol, Chem, root, host)
            if rebuilt != text or frozenset(scope) != frozenset(atoms):
                return None
            recorded[(locants[host], text)] += 1
        if recorded != Counter(row.prefixes):
            return None

    def roles(component, permutation):
        return Counter((permutation[root], permutation[host],
                        frozenset(permutation[i] for i in atoms))
                       for root, host, atoms, _text in component.prefix_branch_ownership)

    def preserves(left, right, permutation):
        for atom in mol.GetAtoms():
            target = mol.GetAtomWithIdx(permutation[atom.GetIdx()])
            if (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope(),
                    atom.GetNumRadicalElectrons(), atom.GetIsAromatic(),
                    atom.GetTotalNumHs(includeNeighbors=True)) != (
                    target.GetAtomicNum(), target.GetFormalCharge(), target.GetIsotope(),
                    target.GetNumRadicalElectrons(), target.GetIsAromatic(),
                    target.GetTotalNumHs(includeNeighbors=True)):
                return False
        for bond in mol.GetBonds():
            target = mol.GetBondBetweenAtoms(permutation[bond.GetBeginAtomIdx()],
                                             permutation[bond.GetEndAtomIdx()])
            if target is None or target.GetBondType() != bond.GetBondType() or target.GetIsAromatic() != bond.GetIsAromatic():
                return False
        a, b = left.construction, right.construction
        return (tuple(permutation[i] for i in left.parent_atoms) == right.parent_atoms
                and frozenset(permutation[i] for i in left.principal_centers) == right.principal_centers
                and {permutation[i]: loc for i, loc in a.atom_locants} == dict(b.atom_locants)
                and frozenset(permutation[i] for i in a.atoms) == b.atoms
                and frozenset(permutation[i] for i in a.embedded_atoms) == b.embedded_atoms
                and roles(a, permutation) == roles(b, tuple(range(mol.GetNumAtoms()))))

    limit = 4096
    matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=True, maxMatches=limit)
    if not matches or len(matches) >= limit:
        return None
    if any(not any(preserves(rows[0], row, match) for match in matches) for row in rows):
        return None
    return rows[0]
