"""Bounded P-51 skeletal-replacement names for unsaturated C/O/S chains.

The source-ordered orientation frontier is retained as mapped candidates;
numbering first follows P-15.4.3.2.1, then unsaturation follows P-15.4.3.2.4
and P-31.1.1/P-31.1.2.2.2.
"""

from __future__ import annotations

from dataclasses import dataclass


RULE = ("P-51.4.1.1/P-51.4.1.2/P-15.4.3.2.1/"
        "P-15.4.3.2.4/P-31.1.1/P-31.1.2.2.2")
_SUPPORTED = frozenset({6, 8, 16})
_HETERO_ORDER = (8, 16)  # P-15.4.1.2: O > S in this bounded scope
_TERMS = {8: "oxa", 16: "thia"}


@dataclass(frozen=True, slots=True)
class ReplacementCandidate:
    """One source-equivalent mapped numbering candidate."""

    decision: object
    path: tuple[int, ...]
    hetero_locants: tuple[int, ...]
    unsaturated_bonds: tuple[tuple[int, int, str, int], ...]


def _orientation_facts(mol, path):
    hetero = tuple(position for position, atom_id in enumerate(path, 1)
                   if mol.GetAtomWithIdx(atom_id).GetAtomicNum() != 6)
    by_element = tuple(tuple(position for position, atom_id in enumerate(path, 1)
                             if mol.GetAtomWithIdx(atom_id).GetAtomicNum() == element)
                       for element in _HETERO_ORDER)
    ene, yne, bonds = [], [], []
    for locant, (left, right) in enumerate(zip(path, path[1:]), 1):
        bond = mol.GetBondBetweenAtoms(left, right)
        if bond.GetBondTypeAsDouble() == 2:
            ene.append(locant)
            bonds.append((left, right, "double", locant))
        elif bond.GetBondTypeAsDouble() == 3:
            yne.append(locant)
            bonds.append((left, right, "triple", locant))
    return (hetero, by_element), tuple(ene), tuple(yne), tuple(bonds)


def _parent_name(mol, path, ene_locs, yne_locs):
    from .constants import MULT, ROOTS
    from .formatting import complete_parent_name, unsaturated_parent

    elements = [mol.GetAtomWithIdx(atom_id).GetAtomicNum() for atom_id in path]
    terms = []
    for element in _HETERO_ORDER:
        locants = [str(position) for position, number in enumerate(elements, 1)
                   if number == element]
        if locants:
            multiplier = MULT.get(len(locants)) if len(locants) > 1 else ""
            if len(locants) > 1 and multiplier is None:
                return None
            terms.append(",".join(locants) + "-" + multiplier + _TERMS[element])
    if len(ene_locs) > 1 and MULT.get(len(ene_locs)) is None:
        return None
    if len(yne_locs) > 1 and MULT.get(len(yne_locs)) is None:
        return None
    root = ROOTS.get(len(path))
    if root is None:
        return None
    base = complete_parent_name(unsaturated_parent(root, list(ene_locs), list(yne_locs)))
    if not base:
        return None
    return "-".join(terms) + base


def _eligible_path(mol, Chem):
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetRingInfo().NumRings()
            or mol.GetStereoGroups()):
        return None
    heavy = frozenset(atom.GetIdx() for atom in mol.GetAtoms()
                      if atom.GetAtomicNum() > 1)
    if not heavy or any(atom.GetAtomicNum() not in _SUPPORTED
                        or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                        or atom.GetIsotope() or int(atom.GetChiralTag())
                        for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1):
        return None
    if any(atom.GetAtomicNum() == 1 and atom.GetIsotope()
           for atom in mol.GetAtoms()):
        return None
    # This producer has no stereodescriptor writer. Reject both stereo stored
    # on a multiple bond and the directional single-bond markers used to
    # encode alkene geometry in SMILES, on the original graph.
    if any(int(bond.GetStereo()) != int(Chem.BondStereo.STEREONONE)
           or int(bond.GetBondDir()) != int(Chem.BondDir.NONE)
           for bond in mol.GetBonds()):
        return None
    from .replacement_eligibility import acyclic_replacement_eligibility

    proof = acyclic_replacement_eligibility(mol, heavy)
    if not proof.eligible or not proof.path:
        return None
    path = tuple(proof.path)
    from .constants import ROOTS
    if (len(path) not in ROOTS
            or any(len(unit.atoms) != 1 for unit in proof.heterounits)
            or any(mol.GetAtomWithIdx(unit.atoms[0]).GetAtomicNum() not in {8, 16}
                   for unit in proof.heterounits)):
        return None
    if set(path) != set(heavy):
        return None
    for position, atom_id in enumerate(path):
        atom = mol.GetAtomWithIdx(atom_id)
        number = atom.GetAtomicNum()
        neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if number == 6:
            if atom.GetTotalValence() != 4:
                return None
        elif number in {8, 16}:
            if len(neighbors) != 2 or atom.GetTotalValence() != 2:
                return None
        else:
            return None
        if position in {0, len(path) - 1} and number != 6:
            return None
    unsaturated = []
    for left, right in zip(path, path[1:]):
        bond = mol.GetBondBetweenAtoms(left, right)
        order = bond.GetBondTypeAsDouble()
        left_element = mol.GetAtomWithIdx(left).GetAtomicNum()
        right_element = mol.GetAtomWithIdx(right).GetAtomicNum()
        if order != 1 and not (order in {2, 3} and left_element == right_element == 6):
            return None
        if order > 1:
            unsaturated.append((left, right))
    # This is specifically an unsaturated-chain route. Saturated replacement
    # chains belong to the existing parent renderer. Keep adjacent multiple
    # bonds (cumulenes and enyne sites sharing a carbon) outside this first
    # bounded grammar until their locant/spelling interactions are covered.
    if not unsaturated:
        return None
    multiple_locants = [locant for locant, (left, right) in enumerate(zip(path, path[1:]), 1)
                        if mol.GetBondBetweenAtoms(left, right).GetBondTypeAsDouble() > 1]
    if any(right - left == 1 for left, right in zip(multiple_locants, multiple_locants[1:])):
        return None
    # Every multiple-bond site must be separated by enough single-bonded
    # valence to keep all atoms at their ordinary valence.
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
    except Exception:
        return None
    return path, tuple(unit.atoms for unit in proof.heterounits), tuple(unsaturated)


def unsaturated_replacement_frontier(mol, Chem):
    """Return every mapped numbering candidate surviving P-15/P-31, or None.

    The return is a tuple because mirror-equivalent parent numberings can tie
    on the fixed hetero and unsaturation locants while assigning the same PIN
    text to different original atoms. Each item contains a `MethodDecision`
    with its own original atom-to-locant map plus the multiple-bond map.
    """
    eligible = _eligible_path(mol, Chem)
    if eligible is None:
        return None
    source_path, units, _ = eligible
    orientations = (source_path, tuple(reversed(source_path)))
    facts = [(path, *_orientation_facts(mol, path)) for path in orientations]
    best_fixed = min(row[1] for row in facts)
    fixed_frontier = [row for row in facts if row[1] == best_fixed]
    def unsaturation_key(row):
        _, _, ene, yne, _ = row
        return (tuple(sorted(ene + yne)), tuple(sorted(ene)), tuple(sorted(yne)))
    best_unsaturation = min(unsaturation_key(row) for row in fixed_frontier)
    frontier = [row for row in fixed_frontier
                if unsaturation_key(row) == best_unsaturation]
    from .naming_methods import MethodDecision

    heavy = frozenset(path for path in source_path)
    output = []
    for path, (hetero_locants, _), ene, yne, bond_map in frontier:
        name = _parent_name(mol, path, ene, yne)
        if name is None:
            return None
        decision = MethodDecision(
            name, "skeletal-replacement", frozenset(), heavy, RULE,
            tuple((atom_id, str(position))
                  for position, atom_id in enumerate(path, 1)), units,
        )
        output.append(ReplacementCandidate(decision, tuple(path), hetero_locants,
                                           tuple(bond_map)))
    if not output or len({candidate.decision.name for candidate in output}) != 1:
        return None
    return tuple(output)


def unsaturated_replacement_decision(mol, Chem):
    """Convenience API: return one MethodDecision when the source frontier ties."""
    frontier = unsaturated_replacement_frontier(mol, Chem)
    return frontier[0].decision if frontier else None
