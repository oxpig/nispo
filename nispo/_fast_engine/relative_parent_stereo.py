"""Graph-owned relative configuration for a simple pseudoasymmetric parent."""

from ..local_cip import assigned_chiral_centers
from .molecule_cache import mol_cache_get

def relative_monocycle_configuration(mol, Chem, sequence) -> str:
    """Read the face relation of exactly two ligands on a saturated cycle.

    Native tetrahedral tags refer to each atom's stored neighbor order, not
    to a common ring plane. Normalize their permutation to the same oriented
    cycle before comparing faces. No conformer or guessed CIP label is used.
    """
    sequence = tuple(sequence)
    core = frozenset(sequence)
    if len(core) != len(sequence) or len(core) < 3:
        return ""
    for previous, current in zip(sequence, sequence[1:] + sequence[:1]):
        bond = mol.GetBondBetweenAtoms(previous, current)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
    substituted = []
    for position, idx in enumerate(sequence):
        atom = mol.GetAtomWithIdx(idx)
        inside = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in core]
        outside = [n.GetIdx() for n in atom.GetNeighbors()
                   if n.GetIdx() not in core and n.GetAtomicNum() > 1]
        if len(inside) != 2:
            return ""
        if outside:
            # Identical geminal terminal groups carry no face assignment.
            # They do not prevent the other two singly substituted positions
            # from defining the cycle's cis/trans relation.
            if (atom.GetAtomicNum() == 6
                    and atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
                    and len(outside) == 2
                    and all(mol.GetAtomWithIdx(i).GetDegree() == 1 for i in outside)):
                ligands = [mol.GetAtomWithIdx(i) for i in outside]
                signatures = [(a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(),
                               a.GetNumRadicalElectrons(), a.GetTotalNumHs()) for a in ligands]
                if signatures[0] == signatures[1]:
                    continue
            if (len(outside) != 1 or atom.GetAtomicNum() != 6
                    or atom.GetTotalNumHs(includeNeighbors=True) != 1
                    or atom.GetChiralTag() not in {
                        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                        Chem.ChiralType.CHI_TETRAHEDRAL_CCW}):
                return ""
            stored = [n.GetIdx() for n in atom.GetNeighbors()]
            hydrogens = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(stored) == 3 and not hydrogens:
                stored.append(-1)  # The implicit H follows native neighbors.
                hydrogens = [-1]
            if len(stored) != 4 or len(hydrogens) != 1:
                return ""
            ordered = [sequence[position - 1], sequence[(position + 1) % len(sequence)],
                       outside[0], hydrogens[0]]
            permutation = [stored.index(i) for i in ordered]
            inversions = sum(a > b for j, a in enumerate(permutation)
                             for b in permutation[j + 1:])
            orientation = int(atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            substituted.append(orientation ^ (inversions % 2))
    if len(substituted) != 2:
        return ""
    return "cis" if substituted[0] == substituted[1] else "trans"



def pseudo_relative_parent(mol, Chem, parent_atoms):
    """Prove that the complete parent ring owns exactly two pseudo sites."""
    parent = frozenset(parent_atoms)
    rings = tuple(mol.GetRingInfo().AtomRings())
    ring = next((r for r in rings if frozenset(r) == parent), None)
    if ring is None or any(parent & frozenset(other) for other in rings if other != ring):
        return ''
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in parent):
        return ''
    tagged = {atom.GetIdx() for atom in mol.GetAtoms() if int(atom.GetChiralTag())}
    if len(tagged) != 2 or not tagged <= parent:
        return ''
    if any(int(bond.GetStereo()) for bond in mol.GetBonds()):
        return ''
    centers = mol_cache_get(mol, "fast_pseudo_stereo_centers")
    if centers is None:
        centers = assigned_chiral_centers(mol, Chem)
    assigned = dict(centers)
    if set(assigned) != tagged or any(value not in {'r', 's'} for value in assigned.values()):
        return ''
    return relative_monocycle_configuration(mol, Chem, ring)
