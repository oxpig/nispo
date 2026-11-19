"""Source-specific retained ether parents (P-63.2.3)."""

from .naming_methods import MethodDecision


def retained_ether_decision(mol, Chem):
    """Anisole is the sole retained PIN in this class, without substitution."""
    atoms = tuple(mol.GetAtoms())
    heavy = tuple(a for a in atoms if a.GetAtomicNum() > 1)
    if (len(heavy) != 8 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {1, 6, 8} or a.GetIsotope()
                   or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) for a in atoms)
            or any(int(b.GetStereo()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    oxygens = [a for a in heavy if a.GetAtomicNum() == 8]
    ring = [a for a in heavy if a.GetIsAromatic() and a.GetAtomicNum() == 6]
    if len(oxygens) != 1 or len(ring) != 6:
        return None
    oxygen = oxygens[0]
    neighbors = [a for a in oxygen.GetNeighbors() if a.GetAtomicNum() > 1]
    methyl = [a for a in neighbors if a.GetAtomicNum() == 6 and not a.GetIsAromatic()
              and a.GetTotalNumHs(includeNeighbors=True) == 3]
    if (len(neighbors) != 2 or len(methyl) != 1
            or sum(a.GetIsAromatic() for a in neighbors) != 1
            or any(b.GetBondTypeAsDouble() != 1 for b in oxygen.GetBonds())
            or len(Chem.GetMolFrags(mol)) != 1):
        return None
    ring_ids = {a.GetIdx() for a in ring}
    if (any(sum(n.GetIdx() in ring_ids for n in a.GetNeighbors()) != 2 for a in ring)
            or any(a.GetTotalNumHs(includeNeighbors=True) !=
                   (0 if any(n.GetIdx() == oxygen.GetIdx() for n in a.GetNeighbors()) else 1)
                   for a in ring)):
        return None
    scope = frozenset(a.GetIdx() for a in heavy)
    return MethodDecision("anisole", "retained-parent", frozenset(), scope,
                          "P-63.2.3", (), component_roles=(("anisole", scope, ()),))
