"""Source-mapped construction of substituted short alkoxy components.

P-63.2.2.2 retains methoxy, ethoxy, propoxy and butoxy with substitution
permitted.  This writer establishes the carbon parent on the original graph,
and contracts only a fully owned, terminally attached saturated C1--C4
organyl.  It declines incomplete, branched, cyclic or multiply bonded carbon
parents rather than altering a general name after assembly.
"""
from __future__ import annotations

from ..constants import ROOTS


def mapped_contracted_alkoxy(mol, Chem, oxygen_idx, parent_idx):
    """Return a preferred local alkoxy prefix and owned atoms, or ``None``."""
    if (not isinstance(oxygen_idx, int) or not isinstance(parent_idx, int)
            or oxygen_idx == parent_idx or mol.GetStereoGroups()):
        return None
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    boundary = mol.GetBondBetweenAtoms(oxygen_idx, parent_idx)
    if (oxygen.GetAtomicNum() != 8 or oxygen.IsInRing() or oxygen.GetIsotope()
            or oxygen.GetFormalCharge() or oxygen.GetNumRadicalElectrons()
            or oxygen.GetTotalNumHs(includeNeighbors=True) or boundary is None
            or boundary.GetBondTypeAsDouble() != 1):
        return None
    children = [a for a in oxygen.GetNeighbors()
                if a.GetIdx() != parent_idx and a.GetAtomicNum() > 1]
    if (len(children) != 1 or children[0].GetAtomicNum() != 6
            or children[0].IsInRing()
            or mol.GetBondBetweenAtoms(oxygen_idx, children[0].GetIdx()).GetBondTypeAsDouble() != 1):
        return None
    carbon_root = children[0]
    root = carbon_root.GetIdx()

    # A terminal methyl can carry explicit isotope H atoms. The oxygen and
    # labelled carbon share one unlocanted alkoxy unit under P-82.2.1.
    if (carbon_root.GetDegree() > 1 and not carbon_root.GetFormalCharge()
            and not carbon_root.GetNumRadicalElectrons()
            and carbon_root.GetTotalNumHs(includeNeighbors=True) == 3):
        hydrogens = tuple(a for a in carbon_root.GetNeighbors() if a.GetAtomicNum() == 1)
        if (len(hydrogens) == carbon_root.GetDegree() - 1
                and any(a.GetIsotope() for a in hydrogens)
                and all(not a.GetFormalCharge() and not a.GetNumRadicalElectrons()
                        and a.GetDegree() == 1 for a in hydrogens)):
            from ..component_mapping import MappedComponent, decorate_component

            atoms = frozenset((oxygen_idx, root, *(a.GetIdx() for a in hydrogens)))
            locants = tuple((i, '') for i in sorted(atoms))
            name = decorate_component(mol, Chem, MappedComponent('methoxy', locants, atoms))
            return (name, atoms) if name else None

    # Construct the exact one-cut scope, never an arbitrary subgraph that
    # could overlap the caller's parent or another substituent.
    scope, stack = set(), [oxygen_idx]
    while stack:
        idx = stack.pop()
        if idx == parent_idx or idx in scope:
            continue
        scope.add(idx)
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                     if n.GetAtomicNum() > 1 and n.GetIdx() != parent_idx)
    cuts = [(i, n.GetIdx()) for i in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() not in scope]
    if cuts != [(oxygen_idx, parent_idx)]:
        return None

    carbon, pending = set(), [root]
    while pending:
        idx = pending.pop()
        if idx in carbon:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetFormalCharge()
                or atom.GetIsotope() or atom.GetNumRadicalElectrons()
                or any(n.GetAtomicNum() == 1 and n.GetIsotope() for n in atom.GetNeighbors())
                or any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds())):
            return None
        carbon.add(idx)
        pending.extend(n.GetIdx() for n in atom.GetNeighbors()
                       if n.GetIdx() in scope and n.GetAtomicNum() == 6
                       and not n.IsInRing() and n.GetIdx() not in carbon)
    if len(carbon) not in {1, 2, 3, 4}:
        return None
    adjacency = {i: {n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                     if n.GetIdx() in carbon} for i in carbon}
    if (len(adjacency[root]) > 1 or any(len(n) > 2 for n in adjacency.values())
            or sum(len(n) for n in adjacency.values()) != 2 * (len(carbon) - 1)):
        return None

    # Delegate the child spelling and all stereochemical/branch ownership to
    # the existing mapped writer.  Only a spelling with the exact graph-proved
    # terminal chain stem is eligible for the local retained contraction.
    from ..component_mapping import mapped_branch, mapped_branch_ring_atoms, _record_branch_ring_atoms
    name, owned = mapped_branch(mol, Chem, root, oxygen_idx)
    owned = frozenset(owned)
    if not name or owned | {oxygen_idx} != scope or oxygen_idx in owned:
        return None
    stem = ROOTS[len(carbon)] + "yl"
    if not name.endswith(stem):
        return None
    from ..formatting import _organyl_to_oxy
    result = _organyl_to_oxy(name, parent_chain_length=len(carbon))
    if not result:
        return None
    atoms = frozenset(scope)
    _record_branch_ring_atoms(
        mol, Chem, oxygen_idx, parent_idx, result, atoms,
        mapped_branch_ring_atoms(mol, Chem, root, oxygen_idx, name, owned),
    )
    return result, atoms
