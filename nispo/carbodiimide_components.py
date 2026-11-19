"""Mapped neutral carbodiimide components under Blue Book P-62.3.1.4.

This local producer recognizes only a closed, acyclic N=C=N unit whose
nitrogens carry at most one carbon substituent each. The substituent names
are mapped from the original graph; unsupported annotations and competing
principal groups make the operation decline.
"""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds


RULES = (
    "P-62.3.1.4",
    "P-16.9.2/P-16.9.3",
    "P-16.5.1.3.1",
    "P-14.3.4.4",
    "P-14.5",
    "P-82.2/P-82.6",
)


@dataclass(frozen=True, slots=True)
class CarbodiimideComponent:
    """One mapped P-62.3.1.4 component, with original atom ownership."""

    name: str
    center: int
    nitrogens: tuple[int, int]
    atoms: frozenset[int]
    embedded_atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    branches: tuple[tuple[int, str, frozenset[int]], ...] = ()
    rule: str = "P-62.3.1.4"


def carbodiimide_core(mol, carbon_idx):
    """Return (center, N-left, N-right), or None for an unproved N=C=N core."""
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if (carbon.GetAtomicNum() != 6 or carbon.GetFormalCharge()
            or carbon.GetNumRadicalElectrons() or carbon.IsInRing()):
        return None
    neighbors = tuple(carbon.GetNeighbors())
    if len(neighbors) != 2 or any(a.GetAtomicNum() != 7 for a in neighbors):
        return None
    nitrogens = tuple(sorted(a.GetIdx() for a in neighbors))
    if any(mol.GetBondBetweenAtoms(carbon_idx, n).GetBondTypeAsDouble() != 2
           for n in nitrogens):
        return None
    for idx in nitrogens:
        nitrogen = mol.GetAtomWithIdx(idx)
        if (nitrogen.GetFormalCharge() or nitrogen.GetNumRadicalElectrons()
                or nitrogen.IsInRing() or nitrogen.GetTotalValence() != 3):
            return None
        heavy = [a for a in nitrogen.GetNeighbors()
                 if a.GetIdx() != carbon_idx and a.GetAtomicNum() > 1]
        if len(heavy) > 1 or any(a.GetAtomicNum() != 6 for a in heavy):
            return None
        if any(mol.GetBondBetweenAtoms(idx, a.GetIdx()).GetBondTypeAsDouble() != 1
               for a in heavy):
            return None
    return (carbon_idx, *nitrogens)


def _principal_conflict(mol, Chem, branches):
    """Decline a suffix proposal if a branch contains a recognized senior group."""
    from .component_mapping import chemical_inventory
    from .naming_methods import _fragment

    for _, _, owned in branches:
        if not owned:
            continue
        # A detached branch is checked independently. This avoids treating the
        # N=C=N site's own unsupported inventory status as evidence of safety.
        # Bracketed isotope atoms need explicit H capping after the cut;
        # otherwise an ordinary labelled alkyl can become an artificial
        # radical on this eligibility-only copy.
        branch, _original = _fragment(mol, Chem, owned)
        if branch is None:
            return True
        inventory = chemical_inventory(branch)
        if not inventory.supported or inventory.principal:
            return True
    return False


def carbodiimide_component(mol, Chem):
    """Build one fully mapped neutral carbodiimide name component, else None.

    P-14.3.4.4 justifies omitting terminal N locants when moving or
    interchanging prefixes cannot create another isomer. The two N sites in
    this isolated N=C=N parent are equivalent, including for unlike groups.
    P-16.9 gives N-site labels when they are needed; P-62.3.1.4 anchors the
    substitutive parent and the locant-free symmetric form.
    """
    from .component_mapping import (
        MappedComponent, decorate_component, mapped_branch,
    )

    if (mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in iter_atoms(mol))):
        return None
    candidates = []
    for atom in iter_atoms(mol):
        core = carbodiimide_core(mol, atom.GetIdx())
        if core is not None:
            candidates.append(core)
    if len(candidates) != 1:
        return None
    center, n1, n2 = candidates[0]
    core_atoms = frozenset((center, n1, n2))
    if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in core_atoms):
        return None
    if any(int(bond.GetStereo()) and
           {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} <= core_atoms
           for bond in iter_bonds(mol)):
        return None

    # Isotopic N locants have no established grammar for this parent. H/D
    # labels at the core likewise require a location grammar that is not
    # established by the cited carbodiimide example.
    for idx in (n1, n2):
        nitrogen = mol.GetAtomWithIdx(idx)
        if nitrogen.GetIsotope() or any(
            a.GetAtomicNum() == 1 and a.GetIsotope() for a in nitrogen.GetNeighbors()
        ):
            return None
    if any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in iter_atoms(mol)):
        return None

    branches = []
    embedded = set()
    prefixes = []
    for nitrogen in (n1, n2):
        roots = [a.GetIdx() for a in mol.GetAtomWithIdx(nitrogen).GetNeighbors()
                 if a.GetIdx() != center and a.GetAtomicNum() > 1]
        if not roots:
            continue
        if len(roots) != 1:
            return None
        root = roots[0]
        text, owned = mapped_branch(mol, Chem, root, nitrogen)
        if (not text or not owned or root not in owned or owned & core_atoms
                or owned & embedded or any(ch.isspace() for ch in text)):
            return None
        branches.append((nitrogen, text, frozenset(owned)))
        embedded.update(owned)
        prefixes.append(("", text))

    if _principal_conflict(mol, Chem, branches):
        return None
    all_heavy = frozenset(a.GetIdx() for a in iter_atoms(mol)
                          if a.GetAtomicNum() > 1)
    if core_atoms | embedded != all_heavy:
        return None

    # P-14.3.4.4 omits locants when exchanging prefixes cannot generate an
    # isomer. The terminal nitrogens in this isolated N=C=N parent are
    # equivalent, so the operation applies to unlike groups too. P-16.9
    # supplies N locant forms when distinction is necessary; it does not
    # require labels for this component. The existing `amino` prefix mode
    # implements locant-free P-16.5 enclosing and alphabetization behavior.
    mapping = ((center, "1"), (n1, "N"), (n2, "N'"))
    name = decorate_component(mol, Chem, MappedComponent(
        "methanediimine", mapping, core_atoms,
        prefixes=tuple(prefixes),
        embedded_atoms=frozenset(embedded),
        prefix_mode="amino",
    ))
    if not name:
        return None
    # decorate_component handles the core's C isotope descriptor; branch-local
    # stereo is already carried by each mapped branch name.
    return CarbodiimideComponent(
        name=name,
        center=center,
        nitrogens=(n1, n2),
        atoms=core_atoms,
        embedded_atoms=frozenset(embedded),
        atom_locants=mapping,
        branches=tuple(branches),
    )


def carbodiimide_decision(mol, Chem):
    """Expose the component proposal as a route-neutral decision record."""
    from .naming_methods import MethodDecision

    component = carbodiimide_component(mol, Chem)
    if component is None:
        return None
    owned = component.atoms | component.embedded_atoms
    return MethodDecision(
        component.name,
        "substitutive",
        frozenset((component.center,)),
        owned,
        rule=component.rule,
        atom_locants=component.atom_locants,
        component_roles=(
            ("carbodiimide-core", component.atoms, component.atom_locants),
            ("N-substituent-branches", component.embedded_atoms, ()),
        ),
    )
