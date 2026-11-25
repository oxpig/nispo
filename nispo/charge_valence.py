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


def ring_substituent_charge_site(mol, ring):
    """Identify one skeletal ionic centre before enumerating ring numberings."""
    site = None
    suffix = "ium"
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        charge = atom.GetFormalCharge()
        if not charge:
            continue
        if site is not None:
            return None, "ium"
        number = atom.GetAtomicNum()
        if (charge == -1 and number == 7 and atom.GetDegree() == 2 and atom.GetTotalValence() == 2
                and not atom.GetTotalNumHs() and not atom.GetNumRadicalElectrons()):
            suffix = "id"
        elif charge == 1 and number in {6, 7, 8, 16}:
            suffix = "ylium" if number == 6 else "ium"
        else:
            return None, "ium"
        site = idx
    return site, suffix


def fused_azolide_substituent(mol, template, match, template_idx, parent_name, locants, attachment):
    """Form an -id-yl prefix only when the mapped neutral hydride lost H at this N."""
    from .template_match_utils import match_preserves_aromatic_nitrogen_hydrogens

    site, suffix = ring_substituent_charge_site(mol, match)
    locant = locants.get(template_idx, "")
    if (site != match[template_idx] or suffix != "id" or not locant.isdigit()
            or template.GetAtomWithIdx(template_idx).GetTotalNumHs() != 1
            or not all(a.GetIsAromatic() and not a.GetFormalCharge() for a in template.GetAtoms())
            or not match_preserves_aromatic_nitrogen_hydrogens(mol, template, match)):
        return None
    return f"{parent_name.removesuffix('e')}-{locant}-id-{attachment}-yl"
