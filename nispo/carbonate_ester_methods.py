"""Mapped functional-class names for simple, fully esterified carbonates.

The recognized core is a neutral, acyclic O=C(OR)(OR') carbonic-acid
derivative. The carbonate component owns its central carbon and all three
oxygen atoms; the two disjoint carbon organyl branches are mapped separately.
P-82.6.4 oxygen qualifiers remain typed to the ether oxygen of each arm.
"""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds


RULES = (
    "P-65.2/P-65.2.1",
    "P-65.6.3.2.1/P-65.6.3.3.2.1",
    "P-14.3.4.4/P-14.5",
    "P-82.2.1/P-82.2.2.1/P-82.3/P-82.6.1.2/P-82.6.4",
)


@dataclass(frozen=True, slots=True)
class CarbonateArm:
    """One mapped carbon organyl and its linking oxygen role."""

    oxygen: int
    root: int
    name: str
    atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    bridge_isotope: int = 0


@dataclass(frozen=True, slots=True)
class CarbonateComponent:
    """A closed proposal with original atom identities and typed components."""

    name: str
    center: int
    carbonyl_oxygen: int
    bridge_oxygens: tuple[int, int]
    atoms: frozenset[int]
    core_atoms: frozenset[int]
    core_locants: tuple[tuple[int, str], ...]
    arms: tuple[CarbonateArm, CarbonateArm]
    rule: str = (
        "P-65.2/P-65.6.3.2.1/P-65.6.3.3.2.1/P-14.5/"
        "P-82.2.1/P-82.2.2.1/P-82.3/P-82.6.1.2/P-82.6.4"
    )


def _carbonate_core(mol, center):
    atom = mol.GetAtomWithIdx(center)
    if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge()
            or atom.GetNumRadicalElectrons() or atom.IsInRing()):
        return None
    doubles, singles = [], []
    for neighbor in atom.GetNeighbors():
        order = mol.GetBondBetweenAtoms(center, neighbor.GetIdx()).GetBondTypeAsDouble()
        if neighbor.GetAtomicNum() != 8:
            return None
        if order == 2:
            doubles.append(neighbor.GetIdx())
        elif order == 1:
            singles.append(neighbor.GetIdx())
        else:
            return None
    if len(doubles) != 1 or len(singles) != 2:
        return None
    carbonyl = doubles[0]
    bridges = tuple(sorted(singles))
    for oxygen in (carbonyl, *bridges):
        o = mol.GetAtomWithIdx(oxygen)
        if o.GetFormalCharge() or o.GetNumRadicalElectrons() or o.IsInRing():
            return None
        if int(o.GetChiralTag()):
            return None
    if (mol.GetBondBetweenAtoms(center, carbonyl).GetBondTypeAsDouble() != 2
            or mol.GetAtomWithIdx(carbonyl).GetDegree() != 1):
        return None
    roots = []
    for oxygen in bridges:
        o = mol.GetAtomWithIdx(oxygen)
        if o.GetDegree() != 2:
            return None
        others = [n.GetIdx() for n in o.GetNeighbors() if n.GetIdx() != center]
        if len(others) != 1 or mol.GetAtomWithIdx(others[0]).GetAtomicNum() != 6:
            return None
        if mol.GetBondBetweenAtoms(oxygen, others[0]).GetBondTypeAsDouble() != 1:
            return None
        roots.append(others[0])
    return center, carbonyl, bridges, tuple(roots)


def carbonate_component(mol, Chem):
    """Recognize and map one narrow carbonate-ester construction, else None.

    P-82.6.4 supplies direct carbonate role examples; P-82.2.1, P-82.3 and
    P-82.6 supply the general descriptor/count/order rules applied here to any
    17O/18O assignment among the three carbonate oxygens. A single 17O or 18O
    bridge label on one of two otherwise identical arms is separately cited
    from the unmodified arm under P-82.2.2.1.
    """
    from .component_mapping import (
        MappedComponent, decorate_component, has_annotations, mapped_branch,
    )
    from .extended_inventory import selection_inventory
    from .formatting import prefix_alphabetization_key, multiply_component
    from .oxygen_isotope_suffixes import apply_organyl_oxygen_qualifier
    from .naming_methods import MethodDecision

    if (mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in iter_atoms(mol))):
        return None
    candidates = [core for a in iter_atoms(mol)
                  if (core := _carbonate_core(mol, a.GetIdx())) is not None]
    if len(candidates) != 1:
        return None
    center, carbonyl, bridges, roots = candidates[0]

    inventory = selection_inventory(mol)
    if (not inventory.principal or len(inventory.principal) != 1
            or inventory.principal[0].kind != "carbonate"):
        return None

    core_atoms = frozenset((center, carbonyl, *bridges))
    if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in core_atoms):
        return None
    if any(int(b.GetStereo()) and
           {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= core_atoms
           for b in iter_bonds(mol)):
        return None

    oxygens = (carbonyl, *bridges)
    o_labels = [mol.GetAtomWithIdx(i).GetIsotope() for i in oxygens]
    if any(mass not in (0, 17, 18) for mass in o_labels):
        return None
    if len(set(roots)) != 2:
        return None

    # Branch eligibility is checked as a detached rooted component, so a
    # competing principal group cannot be hidden beneath the carbonate.
    from .naming_methods import _fragment
    arms = []
    for oxygen, root in zip(bridges, roots):
        name, owned = mapped_branch(mol, Chem, root, oxygen)
        if not name or not owned or root not in owned:
            return None
        branch, _ = _fragment(mol, Chem, owned)
        if branch is None:
            return None
        branch_inventory = selection_inventory(branch)
        if not branch_inventory.supported or branch_inventory.principal:
            return None
        # mapped_branch returns a complete, internally decorated organyl
        # component and original atom set, but no local locant map. Keep the
        # role ownership opaque rather than inventing empty locants.
        arms.append(CarbonateArm(oxygen, root, name, owned, (),
                                 mol.GetAtomWithIdx(oxygen).GetIsotope()))
    first, second = arms
    if first.atoms & second.atoms or core_atoms & first.atoms or core_atoms & second.atoms:
        return None
    if core_atoms | first.atoms | second.atoms != frozenset(
            a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1):
        return None

    # Nested organyl oxygen labels and carbonate O-role qualifiers would
    # compete for one unlocanted O descriptor; that combined grammar is not
    # established by P-82.6.4's examples.
    if any(o_labels) and any(has_annotations(mol, arm.atoms) for arm in arms):
        return None

    for arm in arms:
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 7
               and mol.GetAtomWithIdx(i).GetIsotope() for i in arm.atoms):
            return None
    # Central carbon is the only carbon in the carbonate parent. A carbon-13
    # descriptor therefore follows P-82.6.1.2 without a locant. The typed
    # core descriptor composes it with any oxygen descriptor under P-82.2.1
    # and P-82.3 (element symbols alphabetically; O nuclides by mass).
    carbon_label = mol.GetAtomWithIdx(center).GetIsotope()
    if carbon_label not in (0, 13):
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() and
           mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 8}
           for i in range(mol.GetNumAtoms())):
        return None

    # The core is a typed component owning all carbonate O atoms. Its oxygen
    # isotope labels are emitted as a count; bridge qualifiers remain attached
    # to the organyl components below, preserving the P-82.6.4 role split.
    core_isotope_atoms = frozenset(i for i in core_atoms
                                   if mol.GetAtomWithIdx(i).GetIsotope())
    core_map = MappedComponent(
        "carbonate", ((center, "1"), (carbonyl, ""),
                      (bridges[0], ""), (bridges[1], "")),
        core_atoms, isotope_unlocanted_elements=(8,),
        local_isotope_atoms=core_isotope_atoms,
    )
    core_name = decorate_component(mol, Chem, core_map)
    if not core_name:
        return None
    if core_isotope_atoms:
        # Use one typed descriptor for all core labels. The carbonate parent
        # has one C, while its three O sites permit polysubstitution; the O
        # count therefore includes O1. P-82.3 orders C before O, then O
        # nuclides by increasing mass number.
        counts = {}
        for idx in core_isotope_atoms:
            atom = mol.GetAtomWithIdx(idx)
            key = (atom.GetSymbol(), atom.GetIsotope())
            counts[key] = counts.get(key, 0) + 1
        parts = []
        for (symbol, mass), count in sorted(counts.items()):
            suffix_count = str(count) if symbol == "O" or count > 1 else ""
            parts.append(f"{mass}{symbol}{suffix_count}")
        core_name = "(" + ",".join(parts) + ")" + core_name

    rendered = []
    for arm in arms:
        qualifier = ""
        if any(o_labels):
            qualifier = f"{arm.bridge_isotope}O-" if arm.bridge_isotope else "O-"
        text = apply_organyl_oxygen_qualifier(arm.name, qualifier)
        if not text:
            return None
        rendered.append((arm, text))
    # P-14.5 compares the unqualified organyl identity. If its chemical key
    # ties, graph identity resolves identical units; for otherwise-identical
    # isotopologues P-82.2.2.1 places the modified component first. Other ties
    # have no proven order in this local operation and are declined.
    def signatures(arm):
        indices = sorted(arm.atoms)
        branch_identity = Chem.MolFragmentToSmiles(
            mol, atomsToUse=indices, canonical=True, isomericSmiles=True,
        )
        unmodified = Chem.Mol(mol)
        for idx in indices:
            atom = unmodified.GetAtomWithIdx(idx)
            atom.SetIsotope(0)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for bond in unmodified.GetBonds():
            if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} <= arm.atoms:
                bond.SetStereo(Chem.BondStereo.STEREONONE)
                bond.SetBondDir(Chem.BondDir.NONE)
        base = Chem.MolFragmentToSmiles(
            unmodified, atomsToUse=indices, canonical=True, isomericSmiles=False,
        )
        # P-82.6.4 assigns a bridge-oxygen label to this organyl component.
        # Keep that role in its identity even though the linking O is owned by
        # the carbonate core, so isotopically different arms do not collapse.
        full = (branch_identity, arm.bridge_isotope)
        modified = (any(mol.GetAtomWithIdx(i).GetIsotope() for i in arm.atoms)
                    or bool(arm.bridge_isotope))
        return full, base, modified

    rendered = [(arm, text, signatures(arm)) for arm, text in rendered]
    base_keys = [prefix_alphabetization_key(arm.name)[:3]
                 for arm, _, _ in rendered]
    if base_keys[0] != base_keys[1]:
        ordered = sorted(rendered, key=lambda pair: prefix_alphabetization_key(pair[0].name)[:3])
    else:
        full_a, base_a, modified_a = rendered[0][2]
        full_b, base_b, modified_b = rendered[1][2]
        if any(o_labels):
            # Source-backed expansion is limited to one bridge 17O or 18O
            # and one unmodified bridge, with no carbonyl-O isotope. Other
            # identical-arm patterns lack proved grouping or ordering rules.
            bridge_masses = sorted(arm.bridge_isotope for arm in arms)
            bridge_label = next((mass for mass in bridge_masses if mass), 0)
            if not (bridge_label in {17, 18}
                    and o_labels.count(bridge_label) == 1
                    and sum(mass != 0 for mass in o_labels) == 1
                    and mol.GetAtomWithIdx(carbonyl).GetIsotope() == 0
                    and bridge_masses == [0, bridge_label]):
                return None
        if full_a == full_b:
            ordered = rendered
        elif base_a == base_b and modified_a != modified_b:
            ordered = sorted(rendered, key=lambda pair: not pair[2][2])
        else:
            return None
    if (ordered[0][0].name == ordered[1][0].name
            and ordered[0][1] == ordered[1][1]):
        if any(o_labels):
            return None
        if ordered[0][1] != ordered[1][1] or ordered[0][2][0] != ordered[1][2][0]:
            return None
        from .substituent_multiplication import terminally_substituted_carbon_component
        nature = ('compound' if terminally_substituted_carbon_component(
            mol, ordered[0][0].atoms) else None)
        arms_text = multiply_component(ordered[0][1], 2, nature=nature)
        if not arms_text:
            return None
    else:
        arms_text = " ".join(text for _, text, _ in ordered)
    name = arms_text + " " + core_name

    proposal = CarbonateComponent(
        name, center, carbonyl, bridges,
        core_atoms | first.atoms | second.atoms, core_atoms,
        tuple((idx, loc) for idx, loc in core_map.atom_locants),
        tuple(arms),
    )
    return MethodDecision(
        name, "functional-class", frozenset({center}), proposal.atoms,
        proposal.rule,
        proposal.core_locants + first.atom_locants + second.atom_locants,
        component_roles=(("carbonate", core_atoms, proposal.core_locants),
                        ("organyl:1", first.atoms, first.atom_locants),
                        ("organyl:2", second.atoms, second.atom_locants)),
    )


def carbonate_ester_decision(mol, Chem):
    """Public-compatible decision wrapper for the carbonate method."""
    return carbonate_component(mol, Chem)
