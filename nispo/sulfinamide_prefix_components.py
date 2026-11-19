"""Mapped P-66.1.1.4.2 amino…sulfinyl prefix components."""
from __future__ import annotations


def sulfinamide_subordinate_prefix(mol, Chem, root, blocked, atoms):
    """Build one source-shaped carbon-attached sulfinamide prefix.

    ``None`` means the sulfur graph is outside the one-oxo amide family. An
    empty text and empty ownership set means the family is recognized but a
    local annotation, branch, or ownership detail is not covered here.
    """
    from .component_mapping import MappedComponent, decorate_component, mapped_branch
    from .formatting import enclose_name_unit
    from .organyl_prefix_components import organyl_prefix_nature
    from .sulfinamide_components import recognize_sulfinamide_sites
    from .component_mapping import has_annotations

    sulfur = mol.GetAtomWithIdx(root)
    potential = (
        sulfur.GetAtomicNum() == 16 and blocked >= 0
        and mol.GetAtomWithIdx(blocked).GetAtomicNum() == 6
        and mol.GetBondBetweenAtoms(root, blocked) is not None
        and mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble() == 1
        and sum(n.GetAtomicNum() == 8
                and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 2
                for n in sulfur.GetNeighbors()) == 1
        and any(n.GetAtomicNum() == 7
                and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 1
                for n in sulfur.GetNeighbors())
    )
    if not potential:
        return None
    sites = tuple(site for site in recognize_sulfinamide_sites(mol, Chem)
                  if site.center == root and site.owner == blocked)
    if not sites:
        return "", frozenset()
    if len(sites) != 1:
        return "", frozenset()
    site = sites[0]
    sulfur = mol.GetAtomWithIdx(site.center)
    nitrogen = mol.GetAtomWithIdx(site.ligand)
    core = frozenset(site.atoms)
    expected_atoms = set(core)
    branches = []
    branch_atoms = set()
    if (sulfur.IsInRing() or nitrogen.IsInRing()
            or mol.GetAtomWithIdx(blocked).GetAtomicNum() != 6
            or mol.GetBondBetweenAtoms(root, blocked) is None
            or mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble() != 1
            or any(mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                   or mol.GetAtomWithIdx(i).GetIsotope()
                   or int(mol.GetAtomWithIdx(i).GetChiralTag())
                   for i in core)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() == 1 for i in atoms)):
        return "", frozenset()

    for branch_root, owned in zip(site.n_substituents, site.n_substituent_atoms):
        text, mapped_atoms = mapped_branch(mol, Chem, branch_root, site.ligand)
        owned = frozenset(owned)
        if (not text or frozenset(mapped_atoms) != owned
                or branch_atoms & set(owned) or core & owned):
            return "", frozenset()
        nature = organyl_prefix_nature(mol, branch_root, owned, text)
        if nature != "simple" or has_annotations(mol, owned):
            return "", frozenset()
        branches.append(text)
        branch_atoms.update(owned)
    if len(branches) == 2 and branches[0] != branches[1]:
        return "", frozenset()
    expected_atoms.update(branch_atoms)
    if expected_atoms != set(atoms):
        return "", frozenset()

    # The N remains the mapped amino host; organyl roots are attached as
    # locant-free amino prefixes. No ordinary-amine recognizer is used because
    # the N–S bond is part of the characteristic group, not an alkyl amine.
    component = MappedComponent(
        "amino", ((site.ligand, ""),), frozenset((site.ligand,)),
        tuple(("", branch) for branch in branches),
        frozenset(branch_atoms), prefix_mode="amino",
    )
    amino = decorate_component(mol, Chem, component)
    if not amino:
        return "", frozenset()
    prefix = (enclose_name_unit(amino) + "sulfinyl") if branches else "aminosulfinyl"
    return prefix, frozenset(expected_atoms)
