"""Graph-derived bonding numbers and explicit anionic oxygen spelling."""


def neutral_lambda_locant(atom, locant: int) -> str:
    """Describe nonstandard neutral P/S bonding, including attached hydrogens."""
    if atom.GetFormalCharge() or atom.GetIsAromatic() or atom.GetNumRadicalElectrons():
        return ""
    standard = {"P": 3, "S": 2}.get(atom.GetSymbol())
    if standard is None:
        return ""
    skeletal_valence = sum(
        bond.GetBondTypeAsDouble()
        for bond in atom.GetBonds()
        if bond.GetOtherAtom(atom).GetAtomicNum() > 1
    )
    number = int(skeletal_valence + atom.GetTotalNumHs(includeNeighbors=True))
    if number != standard and number in {4, 5, 6}:
        return f"{locant}lambda{number}"
    return ""


def anionic_oxygen_prefix(parent_atom) -> str:
    # Oxido may be interpreted as an oxide/double bond on these atoms. The
    # substitutive oxidanidyl group fixes both the single bond and O(-1).
    return "oxidanidyl" if parent_atom.GetSymbol() in {"B", "P", "S"} else "oxido"
