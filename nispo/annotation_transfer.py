"""Compare stereochemical descriptors across mapped private graph copies.

Some name constructors search on a neutralized or capped copy, then write the
result for the original graph.  Stored bond ``STEREOE``/``STEREOTRANS`` enums
encode a drawing orientation, not a portable E/Z descriptor: CIP priorities can
change when a remote atom is removed.  This helper reassigns modern CIP labels
on clones of both graphs and compares mapped carbon tetrahedral and assigned
double-bond descriptors.  A missing or unassignable descriptor is unknown.
"""
from __future__ import annotations


def _modern_cip_clone(mol, Chem):
    """Return a clone freshly labelled by RDKit's modern CIP implementation."""
    from rdkit.Chem import rdCIPLabeler

    clone = Chem.Mol(mol)
    for atom in clone.GetAtoms():
        for prop in ("_CIPCode", "CIPCode", "_CIPRank"):
            if atom.HasProp(prop):
                atom.ClearProp(prop)
    for bond in clone.GetBonds():
        if bond.HasProp("_CIPCode"):
            bond.ClearProp("_CIPCode")
    try:
        rdCIPLabeler.AssignCIPLabels(clone)
    except Exception:
        return None
    return clone


def _private_to_original(private, original):
    """Read the map-number bijection used by oxide private-copy builders."""
    result = {}
    seen = set()
    for atom in private.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num <= 0:
            return None
        original_idx = map_num - 1
        if (original_idx < 0 or original_idx >= original.GetNumAtoms()
                or original_idx in seen):
            return None
        counterpart = original.GetAtomWithIdx(original_idx)
        if (counterpart.GetAtomicNum() != atom.GetAtomicNum()
                or counterpart.GetIsotope() != atom.GetIsotope()):
            return None
        result[atom.GetIdx()] = original_idx
        seen.add(original_idx)
    for bond in private.GetBonds():
        left = result[bond.GetBeginAtomIdx()]
        right = result[bond.GetEndAtomIdx()]
        original_bond = original.GetBondBetweenAtoms(left, right)
        if original_bond is None or original_bond.GetBondType() != bond.GetBondType():
            return None
    for bond in original.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in seen and right in seen:
            private_left = next((idx for idx, old in result.items() if old == left), None)
            private_right = next((idx for idx, old in result.items() if old == right), None)
            if (private_left is None or private_right is None
                    or private.GetBondBetweenAtoms(private_left, private_right) is None):
                return None
    return result


def _stereo_signature(labelled, atom_to_original, Chem):
    atoms = {}
    for atom in labelled.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        original_idx = atom_to_original.get(atom.GetIdx())
        if original_idx is None or not atom.HasProp("_CIPCode"):
            return None
        atoms[original_idx] = atom.GetProp("_CIPCode")

    bonds = {}
    for bond in labelled.GetBonds():
        if (bond.GetBondType() != Chem.BondType.DOUBLE
                or bond.GetStereo() == Chem.BondStereo.STEREONONE):
            continue
        left = atom_to_original.get(bond.GetBeginAtomIdx())
        right = atom_to_original.get(bond.GetEndAtomIdx())
        if left is None or right is None or not bond.HasProp("_CIPCode"):
            return None
        bonds[frozenset((left, right))] = bond.GetProp("_CIPCode")
    return atoms, bonds


def mapped_stereochemistry_equivalent(original, private, Chem):
    """Compare assigned carbon and alkene descriptors through atom maps.

    ``private`` must carry original atom indices plus one as atom-map numbers,
    as do the amine-oxide private-copy builders.  Return ``True`` when all
    relevant modern CIP descriptors transfer unchanged, ``False`` when both
    graphs have assignable but different descriptors, or ``None`` when the
    mapping/descriptor cannot be proved.  Callers must accept only ``True``.
    Neither input molecule is modified.
    """
    mapping = _private_to_original(private, original)
    if mapping is None:
        return None
    original_map = {idx: idx for idx in range(original.GetNumAtoms())}
    labelled_original = _modern_cip_clone(original, Chem)
    labelled_private = _modern_cip_clone(private, Chem)
    if labelled_original is None or labelled_private is None:
        return None
    original_signature = _stereo_signature(labelled_original, original_map, Chem)
    private_signature = _stereo_signature(labelled_private, mapping, Chem)
    if original_signature is None or private_signature is None:
        return None
    return original_signature == private_signature
