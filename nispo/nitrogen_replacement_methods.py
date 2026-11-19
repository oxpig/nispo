"""Mapped P-51 skeletal-replacement names with nonprincipal chain nitrogens.

This bounded producer handles neutral C/N/O heterochains whose principal
characteristic groups are carbon-bound alcohols. N atoms in the replacement
chain are admitted only after principal-group inventory proves that they are
not themselves part of a principal characteristic group. One separate
primary-amino prefix is supported, following the construction printed in
P-51.4.1.3.
"""

from __future__ import annotations

from dataclasses import dataclass


RULE = ("P-51.4.1.1/P-51.4.1.2/P-51.4.1.3/"
        "P-15.4.1.2/P-15.4.3.1/P-15.4.3.2.1/"
        "P-15.4.3.2.3/P-15.4.3.2.4/P-31.1.1/P-31.1.2.2.2/P-62.2.3")
_SUPPORTED = frozenset({6, 7, 8})
_HETERO_ORDER = (8, 7)  # P-15.4.1.2: O precedes N in this supported subset
_TERMS = {8: "oxa", 7: "aza"}


@dataclass(frozen=True, slots=True)
class NitrogenReplacementCandidate:
    """One surviving source-equivalent orientation with original atom maps."""

    decision: object
    path: tuple[int, ...]
    hetero_locants: tuple[int, ...]
    alcohol_locants: tuple[int, ...]
    amino_locants: tuple[int, ...]
    unsaturated_bonds: tuple[tuple[int, int, str, int], ...]


def _unsupported_annotation(mol, Chem):
    heavy = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if any(atom.GetIsotope() or int(atom.GetChiralTag()) for atom in heavy):
        return True
    if any(atom.GetAtomicNum() == 1 and atom.GetIsotope() for atom in mol.GetAtoms()):
        return True
    return any(int(bond.GetStereo()) != int(Chem.BondStereo.STEREONONE)
               or int(bond.GetBondDir()) != int(Chem.BondDir.NONE)
               for bond in mol.GetBonds())


def _groups_and_path(mol, Chem):
    """Prove the suffix inventory and return one complete mapped parent path."""
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetRingInfo().NumRings()
            or mol.GetStereoGroups() or _unsupported_annotation(mol, Chem)
            # This bounded unannotated grammar uses implicit hydrogen
            # inventories. Explicit H vertices would otherwise be omitted
            # from the role partition and can perturb neighbor assumptions.
            or any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())):
        return None
    heavy = frozenset(atom.GetIdx() for atom in mol.GetAtoms()
                      if atom.GetAtomicNum() > 1)
    if not heavy or any(atom.GetAtomicNum() not in _SUPPORTED
                        or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                        for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1):
        return None

    from .functional_inventory import functional_inventory
    inventory = functional_inventory(mol)
    if not inventory.supported or not inventory.groups:
        return None
    principal = inventory.principal
    if not principal or any(group.kind != "alcohol" for group in principal):
        return None
    groups = inventory.groups
    if any(group.kind not in {"alcohol", "amine"} for group in groups):
        return None
    alcohols = tuple(group for group in principal if group.kind == "alcohol")
    if not alcohols:
        return None

    skipped = set()
    principal_heteroatoms = set()
    alcohol_oxygens = []
    for group in alcohols:
        center = mol.GetAtomWithIdx(group.center)
        if center.GetAtomicNum() != 6:
            return None
        oxygens = [idx for idx in group.atoms
                   if mol.GetAtomWithIdx(idx).GetAtomicNum() == 8 and idx != group.center]
        if len(oxygens) != 1:
            return None
        oxygen = mol.GetAtomWithIdx(oxygens[0])
        if (oxygen.GetDegree() != 1 or oxygen.GetFormalCharge()
                or oxygen.GetTotalNumHs(includeNeighbors=True) != 1
                or mol.GetBondBetweenAtoms(group.center, oxygen.GetIdx()).GetBondTypeAsDouble() != 1):
            return None
        skipped.add(oxygen.GetIdx())
        principal_heteroatoms.add(oxygen.GetIdx())
        alcohol_oxygens.append(oxygen.GetIdx())

    amino_branches = []
    for group in groups:
        if group.kind != "amine":
            continue
        nitrogen = mol.GetAtomWithIdx(group.center)
        heavy_neighbors = [neighbor.GetIdx() for neighbor in nitrogen.GetNeighbors()
                           if neighbor.GetAtomicNum() > 1]
        if (nitrogen.GetAtomicNum() != 7 or nitrogen.GetFormalCharge()
                or nitrogen.GetNumRadicalElectrons()):
            return None
        if len(heavy_neighbors) == 1:
            # P-51's printed amino-plus-aza example supports a separately
            # cited terminal primary amino prefix. It is not a parent-chain
            # heterounit, and only one such branch is admitted here.
            if (nitrogen.GetTotalNumHs(includeNeighbors=True) != 2
                    or mol.GetAtomWithIdx(heavy_neighbors[0]).GetAtomicNum() != 6
                    or group.center in skipped):
                return None
            amino_branches.append((group.center, heavy_neighbors[0]))
            skipped.add(group.center)
        elif len(heavy_neighbors) == 2:
            # Internal secondary amine N remains in the proposed parent; it is
            # a replacement unit only because it is outside the principal OH.
            continue
        else:
            return None
    if len(amino_branches) > 1:
        return None

    parent_atoms = heavy - skipped
    if not parent_atoms:
        return None
    from .replacement_eligibility import acyclic_replacement_eligibility

    proof = acyclic_replacement_eligibility(
        mol, parent_atoms, principal_group_atoms=principal_heteroatoms,
        has_principal_suffix=True,
    )
    if not proof.eligible or not proof.path:
        return None
    path = tuple(proof.path)
    from .constants import ROOTS
    if (len(path) not in ROOTS or set(path) != set(parent_atoms)
            or any(len(unit.atoms) != 1 for unit in proof.heterounits)
            or any(mol.GetAtomWithIdx(unit.atoms[0]).GetAtomicNum() not in {7, 8}
                   for unit in proof.heterounits)):
        return None
    if any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in (path[0], path[-1])):
        return None

    # Every original heavy atom must be a path atom, an alcohol O, or the
    # explicitly typed primary-amino N; no unowned branch is tolerated.
    if set(path) | set(alcohol_oxygens) | {idx for idx, _ in amino_branches} != set(heavy):
        return None
    path_set = set(path)
    for position, atom_id in enumerate(path):
        atom = mol.GetAtomWithIdx(atom_id)
        number = atom.GetAtomicNum()
        heavy_neighbors = [neighbor for neighbor in atom.GetNeighbors()
                           if neighbor.GetAtomicNum() > 1]
        path_degree = sum(neighbor.GetIdx() in path_set for neighbor in heavy_neighbors)
        if number == 6:
            if atom.GetTotalValence() != 4:
                return None
        elif number == 8:
            if path_degree != 2 or atom.GetTotalValence() != 2:
                return None
        elif number == 7:
            if path_degree != 2 or atom.GetTotalValence() != 3:
                return None
        else:
            return None
        if position in {0, len(path) - 1} and number != 6:
            return None
    if any(mol.GetAtomWithIdx(amino).GetTotalValence() != 3 for amino, _ in amino_branches):
        return None
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
    except Exception:
        return None

    multiple = []
    multiple_locants = []
    for locant, (left, right) in enumerate(zip(path, path[1:]), 1):
        bond = mol.GetBondBetweenAtoms(left, right)
        order = bond.GetBondTypeAsDouble()
        if order != 1 and not (order in {2, 3}
                               and mol.GetAtomWithIdx(left).GetAtomicNum() == 6
                               and mol.GetAtomWithIdx(right).GetAtomicNum() == 6):
            return None
        if order > 1:
            kind = "double" if order == 2 else "triple"
            multiple.append((left, right, kind, locant))
            multiple_locants.append(locant)
    if any(right - left == 1 for left, right in zip(multiple_locants, multiple_locants[1:])):
        return None

    return (path, tuple(unit.atoms for unit in proof.heterounits),
            tuple(alcohol_oxygens), tuple(amino_branches), tuple(multiple))


def _facts(mol, path, alcohol_oxygens, amino_branches):
    hetero = tuple(position for position, idx in enumerate(path, 1)
                   if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6)
    by_element = tuple(tuple(position for position, idx in enumerate(path, 1)
                             if mol.GetAtomWithIdx(idx).GetAtomicNum() == number)
                       for number in _HETERO_ORDER)
    alcohol_locs = tuple(sorted(path.index(_alcohol_host(mol, idx)) + 1
                                for idx in alcohol_oxygens))
    amino_locs = tuple(sorted(path.index(host) + 1 for _, host in amino_branches))
    ene, yne, bond_map = [], [], []
    for locant, (left, right) in enumerate(zip(path, path[1:]), 1):
        order = mol.GetBondBetweenAtoms(left, right).GetBondTypeAsDouble()
        if order == 2:
            ene.append(locant)
            bond_map.append((left, right, "double", locant))
        elif order == 3:
            yne.append(locant)
            bond_map.append((left, right, "triple", locant))
    return (hetero, by_element), alcohol_locs, amino_locs, tuple(ene), tuple(yne), tuple(bond_map)


def _alcohol_host(mol, oxygen_idx):
    """Return the unique carbon host of a validated terminal OH oxygen."""
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    hosts = [neighbor.GetIdx() for neighbor in oxygen.GetNeighbors()
             if neighbor.GetAtomicNum() == 6]
    if len(hosts) != 1:
        raise ValueError("alcohol oxygen must have one carbon host")
    return hosts[0]


def _render_name(mol, path, alcohol_locs, amino_locs, ene_locs, yne_locs):
    from .constants import MULT, ROOTS
    from .formatting import complete_parent_name, format_prefixes, unsaturated_parent
    from .naming_methods import _alcohol_suffix

    elements = [mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in path]
    terms = []
    for element in _HETERO_ORDER:
        locants = [str(position) for position, number in enumerate(elements, 1)
                   if number == element]
        if locants:
            multiplier = MULT.get(len(locants)) if len(locants) > 1 else ""
            if len(locants) > 1 and multiplier is None:
                return None
            terms.append(",".join(locants) + "-" + multiplier + _TERMS[element])
    if (len(ene_locs) > 1 and MULT.get(len(ene_locs)) is None
            or len(yne_locs) > 1 and MULT.get(len(yne_locs)) is None):
        return None
    root = ROOTS.get(len(path))
    if root is None:
        return None
    base = complete_parent_name(unsaturated_parent(root, list(ene_locs), list(yne_locs)))
    if not base:
        return None
    parent = "-".join(terms) + base
    name = _alcohol_suffix(parent, [str(locant) for locant in alcohol_locs])
    if not name:
        return None
    if amino_locs:
        prefix = format_prefixes([(str(locant), "amino") for locant in amino_locs])
        if not prefix:
            return None
        name = prefix + "-" + name
    return name


def nitrogen_replacement_frontier(mol, Chem):
    """Return the complete mapped P-15/P-31 frontier, or None if unsupported."""
    proof = _groups_and_path(mol, Chem)
    if proof is None:
        return None
    source_path, units, alcohol_oxygens, amino_branches, multiple = proof
    paths = (source_path, tuple(reversed(source_path)))
    rows = [(path, *_facts(mol, path, alcohol_oxygens, amino_branches)) for path in paths]
    fixed_key = min(row[1] for row in rows)
    fixed = [row for row in rows if row[1] == fixed_key]

    def later_key(row):
        _, _, alcohol_locs, amino_locs, ene, yne, _ = row
        # After P-15 has fixed heteroatom numbering, suffix locants and then
        # multiple-bond locants follow the fixed parent numbering. Prefix
        # locants break only a remaining exact tie.
        return (alcohol_locs, tuple(sorted(ene + yne)), tuple(ene), tuple(yne), amino_locs)

    best = min(later_key(row) for row in fixed)
    frontier = [row for row in fixed if later_key(row) == best]
    from .naming_methods import MethodDecision

    all_heavy = frozenset(atom.GetIdx() for atom in mol.GetAtoms()
                          if atom.GetAtomicNum() > 1)
    output = []
    for path, (hetero, _), alcohol_locs, amino_locs, ene, yne, bond_map in frontier:
        name = _render_name(mol, path, alcohol_locs, amino_locs, ene, yne)
        if name is None:
            return None
        atom_locants = {atom_id: str(position) for position, atom_id in enumerate(path, 1)}
        for oxygen in alcohol_oxygens:
            host = _alcohol_host(mol, oxygen)
            atom_locants[oxygen] = str(path.index(host) + 1)
        for nitrogen, host in amino_branches:
            atom_locants[nitrogen] = str(path.index(host) + 1)
        if set(atom_locants) != set(all_heavy):
            return None
        roles = [
            ("parent-path", frozenset(path),
             tuple((atom, atom_locants[atom]) for atom in path)),
            ("alcohol-suffix-oxygen", frozenset(alcohol_oxygens),
             tuple((oxygen, str(path.index(_alcohol_host(mol, oxygen)) + 1))
                   for oxygen in alcohol_oxygens)),
        ]
        amino_atoms = frozenset(nitrogen for nitrogen, _ in amino_branches)
        if amino_atoms:
            roles.append(("amino-prefix-nitrogen", amino_atoms,
                          tuple((nitrogen, str(path.index(host) + 1))
                                for nitrogen, host in amino_branches)))
        roles = tuple(roles)
        role_atoms = [atom for _, owned, _ in roles for atom in owned]
        if len(role_atoms) != len(set(role_atoms)) or set(role_atoms) != set(all_heavy):
            return None
        if any(set(owned) != {atom for atom, _ in locants}
               for _, owned, locants in roles):
            return None
        decision = MethodDecision(
            name, "skeletal-replacement",
            frozenset(atom for atom in atom_locants if mol.GetAtomWithIdx(atom).GetAtomicNum() == 6
                      and any(oxygen in {n.GetIdx() for n in mol.GetAtomWithIdx(atom).GetNeighbors()}
                              for oxygen in alcohol_oxygens)),
            all_heavy, RULE, tuple(atom_locants.items()), units, roles,
        )
        output.append(NitrogenReplacementCandidate(
            decision, tuple(path), hetero, alcohol_locs, amino_locs, tuple(bond_map)))
    if not output or len({item.decision.name for item in output}) != 1:
        return None
    return tuple(output)


def nitrogen_replacement_decision(mol, Chem):
    """Convenience API returning one source-equivalent mapped decision."""
    frontier = nitrogen_replacement_frontier(mol, Chem)
    return frontier[0].decision if frontier else None
