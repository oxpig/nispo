"""Fast dispatch to the full ring-aldehyde suffix selection when proved.

P-41 gives aldehydes priority over ketones, alcohols and amines, while acids,
amides, nitriles and other senior classes take precedence. The cheap graph
precheck avoids building a functional inventory for inputs without an
aldehyde directly attached to a ring carbon. The complete writer, not this
guard, remains responsible for parent selection and suffix construction.
"""


def _ring_carbon_attached_aldehyde(mol, atom):
    if (atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetFormalCharge()
            or atom.GetTotalNumHs(includeNeighbors=True) != 1):
        return False
    heavy = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() > 1]
    if len(heavy) != 2:
        return False
    oxygens = [n for n in heavy if n.GetAtomicNum() == 8
               and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2]
    owners = [n for n in heavy if n.GetAtomicNum() == 6 and n.IsInRing()
              and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
    return len(oxygens) == len(owners) == 1


def bounded_ring_aldehyde_requires_full(mol, Chem):
    """Return true only for supported, principal ring-carbon aldehydes."""
    if not mol.GetRingInfo().NumRings() or not any(
        _ring_carbon_attached_aldehyde(mol, atom) for atom in mol.GetAtoms()
    ):
        return False
    from .functional_inventory import functional_inventory

    inventory = functional_inventory(mol)
    return bool(inventory.supported and inventory.principal
                and all(group.kind == "aldehyde"
                        and _ring_carbon_attached_aldehyde(mol, mol.GetAtomWithIdx(group.center))
                        for group in inventory.principal))
