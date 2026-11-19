"""Fast fallback for source-proved ring-bound carboxylic acids."""


def bounded_ring_acid_requires_full(mol, Chem):
    """Preserve acid suffix priority in a bounded two-ring source graph.

    The older shortcut can attach the whole acid as a carboxy prefix to the
    pendant ring. Admit only one benzene bearing the unique neutral acid,
    directly bonded to one benzene or pyridine, with no other heavy atoms.
    The full writer retains any unresolved ring-parent comparison in its own
    status; this predicate does not certify the complete PIN.
    """
    atoms = tuple(mol.GetAtoms())
    if (mol.GetStereoGroups() or any(
            a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
            or int(a.GetChiralTag()) or a.GetAtomicNum() == 1 for a in atoms)
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return False
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if len(rings) != 2 or any(len(ring) != 6 for ring in rings) or rings[0] & rings[1]:
        return False
    if any(not mol.GetAtomWithIdx(i).GetIsAromatic() for ring in rings for i in ring):
        return False
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.groups) != 1
            or len(inventory.principal) != 1 or inventory.principal[0].kind != 'acid'):
        return False
    acid = inventory.principal[0]
    acid_rings = [ring for ring in rings if acid.owner in ring]
    if len(acid_rings) != 1:
        return False
    acid_ring = acid_rings[0]
    other_ring = next(ring for ring in rings if ring != acid_ring)
    if (any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in acid_ring)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7} for i in other_ring)
            or sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in other_ring) > 1
            or set(acid.atoms) | set(acid_ring) | set(other_ring)
                != {atom.GetIdx() for atom in atoms}):
        return False
    links = [bond for bond in mol.GetBonds()
             if ({bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & acid_ring
                 and {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & other_ring)]
    return (len(links) == 1 and links[0].GetBondTypeAsDouble() == 1
            and links[0].GetBeginAtom().GetAtomicNum() == 6
            and links[0].GetEndAtom().GetAtomicNum() == 6)
