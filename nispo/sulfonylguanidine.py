"""Original atom roles for an N-sulfonylguanidine subordinate to an acid."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Sulfonylguanidine:
    carbon: int
    imino: int
    amino: tuple[int, int]
    sulfur: int
    oxygens: tuple[int, int]
    organyl: int


def sulfonylguanidine(mol, sulfur):
    atom = mol.GetAtomWithIdx(sulfur)
    if not _plain(atom, 16, 6) or atom.GetDegree() != 4:
        return None
    oxygen, nitrogen, carbon = [], [], []
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        number, order = other.GetAtomicNum(), bond.GetBondTypeAsDouble()
        if number == 8 and order == 2 and _plain(other, 8, 2) and other.GetDegree() == 1:
            oxygen.append(other.GetIdx())
        elif number == 7 and order == 1 and _plain(other, 7, 3) and other.GetDegree() == 2:
            nitrogen.append(other.GetIdx())
        elif number == 6 and order == 1 and not other.GetFormalCharge():
            carbon.append(other.GetIdx())
    if len(oxygen) != 2 or len(nitrogen) != 1 or len(carbon) != 1:
        return None
    imino = mol.GetAtomWithIdx(nitrogen[0])
    center = next(a for a in imino.GetNeighbors() if a.GetIdx() != sulfur)
    c = center.GetIdx()
    if (not _plain(center, 6, 4) or center.GetDegree() != 3
            or mol.GetBondBetweenAtoms(c, imino.GetIdx()).GetBondTypeAsDouble() != 2):
        return None
    amino = tuple(a for a in center.GetNeighbors() if a.GetIdx() != imino.GetIdx())
    if any(not _plain(a, 7, 3) or any(b.GetBondTypeAsDouble() != 1
                                      for b in a.GetBonds()) for a in amino):
        return None
    return Sulfonylguanidine(c, imino.GetIdx(), tuple(a.GetIdx() for a in amino),
                             sulfur, tuple(oxygen), carbon[0])


def _plain(atom, number, valence):
    return (atom.GetAtomicNum() == number and atom.GetTotalValence() == valence
            and not (atom.IsInRing() or atom.GetIsotope() or atom.GetFormalCharge()
                     or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag())))


def mapped_sulfonylguanidino_prefix(mol, Chem, root, host, scope):
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_general_branch
    from .formatting import enclose_name_unit, format_prefixes
    from .molecule_cache import mol_cache_dict
    from .stereochemistry import assigned_bond_stereochemistry

    near = mol.GetAtomWithIdx(root)
    if (not _plain(near, 7, 3) or near.GetDegree() != 2 or near.GetTotalNumHs() != 1
            or mol.GetAtomWithIdx(host).GetAtomicNum() != 6):
        return None
    center = next(a for a in near.GetNeighbors() if a.GetIdx() != host)
    if center.GetAtomicNum() != 6 or center.GetDegree() != 3:
        return None
    for bond in center.GetBonds():
        imino = bond.GetOtherAtom(center)
        if (bond.GetBondTypeAsDouble() == 2 and imino.GetAtomicNum() == 7
                and imino.GetDegree() == 2):
            sulfur = next(a for a in imino.GetNeighbors() if a.GetIdx() != center.GetIdx())
            site = sulfonylguanidine(mol, sulfur.GetIdx())
            if site is not None and root in site.amino:
                break
    else:
        return None

    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if (not inventory.supported or not inventory.principal
            or any(g.kind != "acid" or g.center in scope for g in inventory.principal)):
        return None
    other = next(i for i in site.amino if i != root)
    owned = {site.carbon, site.imino, *site.amino, site.sulfur, *site.oxygens}
    ligands = [("N''", site.organyl, site.sulfur)]
    ligands.extend(("N'", a.GetIdx(), other)
                   for a in mol.GetAtomWithIdx(other).GetNeighbors()
                   if a.GetIdx() != site.carbon and a.GetAtomicNum() > 1)
    prefixes = []
    for locant, child, parent in ligands:
        if mol.GetAtomWithIdx(child).GetAtomicNum() != 6:
            return None
        text, atoms = mapped_general_branch(mol, Chem, child, parent)
        if (not text or any(c.isspace() for c in text) or atoms & (owned | {host})
                or atoms != frozenset(collect_subtree(mol, child, parent))):
            return None
        owned.update(atoms)
        if parent == site.sulfur:
            text = (enclose_name_unit(text) if not text.isalpha() else text) + "sulfonyl"
        prefixes.append((locant, text))
    if owned != scope:
        return None
    text = format_prefixes(prefixes, hierarchical_enclosures=True) + "carbamimidamido"
    descriptor = next((d for a, b, d in assigned_bond_stereochemistry(mol, Chem)
                       if {a, b} == {site.carbon, site.imino}), None)
    if descriptor:
        text = f"({descriptor})-{text}"
        mol_cache_dict(mol, "rendered_mapped_branch_bond_stereo")[
            (site.carbon, site.imino, text, scope)] = descriptor
    elif int(mol.GetBondBetweenAtoms(site.carbon, site.imino).GetStereo()):
        return None
    return text, scope
