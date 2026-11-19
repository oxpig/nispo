"""Mapped acyclic skeletal-replacement parents with terminal acyl suffixes.

This bounded renderer accepts a caller-proposed path but independently checks
the P-51 replacement threshold, fixed numbering, typed suffix atoms, and full
heavy-atom ownership. It handles simple neutral acids, primary amides, and
nitriles, including two same-class suffix groups at opposite chain ends.
Terminal halogens and subordinate oxo groups on the chain are mapped as prefixes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .constants import HALO, MULT, ROOTS
from .formatting import format_prefixes, prefix_alphabetization_key
from .replacement_eligibility import acyclic_replacement_eligibility


RULE = "P-51.1.3/P-51.4.1.1/P-51.4.1.2/P-51.4.1.3/P-15.4.1.2/P-15.4.3.1/P-15.4.3.2.1/P-15.4.3.2.3/P-65.1.2.1/P-66.1.1.1.1.1/P-66.5.1.1.1"
PinExpectationStatus = Literal["resolved", "pending"]


@dataclass(frozen=True, slots=True)
class AcylReplacementProof:
    """Role, numbering, and whole-name confidence evidence for one proposal."""

    decision: object
    parent_path: tuple[int, ...]
    suffix_kind: str
    suffix_locant: str
    heterounit_atoms: tuple[tuple[int, ...], ...]
    equivalent_paths: tuple[tuple[int, ...], ...] = ()
    suffix_locants: tuple[str, ...] = ()
    pin_expectation_status: PinExpectationStatus = "resolved"
    pin_expectation_reason: str = ""
    prefixes: tuple[tuple[int, int, str], ...] = ()


def _key(mol, path, groups, prefixes):
    hetero = tuple(j for j, idx in enumerate(path, 1)
                   if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6)
    by_element = tuple(
        tuple(j for j, idx in enumerate(path, 1)
              if mol.GetAtomWithIdx(idx).GetAtomicNum() == number)
        for number in (8, 16, 7)
    )
    suffix = tuple(sorted(path.index(group.center) + 1 for group in groups))
    indexed = [(path.index(owner) + 1, text) for _, owner, text in prefixes]
    return (hetero, by_element, suffix, tuple(sorted(locant for locant, _ in indexed)),
            tuple(locant for locant, _ in sorted(indexed, key=lambda row: (
                prefix_alphabetization_key(row[1]), row[0]))))


def replacement_leaf_prefix(mol, idx):
    """Recognize a single, unannotated oxo or halo atom on a carbon parent."""
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetDegree() != 1 or atom.GetIsotope() or atom.GetFormalCharge():
        return None
    bond = atom.GetBonds()[0]
    owner = bond.GetOtherAtom(atom)
    if owner.GetAtomicNum() != 6 or atom.GetTotalNumHs(includeNeighbors=True):
        return None
    order = bond.GetBondTypeAsDouble()
    text = HALO.get(atom.GetSymbol()) if order == 1 else (
        'oxo' if order == 2 and atom.GetAtomicNum() == 8 else None)
    return (idx, owner.GetIdx(), text) if text else None


def decorate_replacement_prefixes(name, locants, prefixes):
    if not prefixes:
        return name
    text = format_prefixes([(locants[owner], word) for _, owner, word in prefixes])
    return text + ('-' if name[0].isdigit() else '') + name


def _parent_text(mol, path):
    if len(path) not in ROOTS:
        return ""
    pieces = []
    for number, prefix in ((8, "oxa"), (16, "thia"), (7, "aza")):
        positions = tuple(j for j, idx in enumerate(path, 1)
                          if mol.GetAtomWithIdx(idx).GetAtomicNum() == number)
        if positions:
            multiplier = MULT.get(len(positions), "") if len(positions) > 1 else ""
            if len(positions) > 1 and not multiplier:
                return ""
            pieces.append(",".join(map(str, positions)) + "-" + multiplier + prefix)
    return "-".join(pieces) + ROOTS[len(path)] + "ane"


def _typed_group(mol, group):
    """Return (suffix kind, external group atoms) for a primary terminal site."""
    if group.kind not in {"acid", "amide", "nitrile"}:
        return None
    center = group.center
    atom = mol.GetAtomWithIdx(center)
    if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() or atom.GetIsotope()
            or atom.GetTotalValence() != 4):
        return None
    neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
    double_oxygens = [i for i in neighbors
                      if mol.GetAtomWithIdx(i).GetAtomicNum() == 8
                      and mol.GetBondBetweenAtoms(center, i).GetBondTypeAsDouble() == 2]
    if len(double_oxygens) != 1:
        if group.kind == "nitrile":
            triple_nitrogens = [i for i in neighbors
                if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
                and mol.GetBondBetweenAtoms(center, i).GetBondTypeAsDouble() == 3]
            if len(triple_nitrogens) == 1 and set(group.atoms) == {center, triple_nitrogens[0]}:
                nitrogen = mol.GetAtomWithIdx(triple_nitrogens[0])
                if (not nitrogen.GetFormalCharge() and not nitrogen.GetIsotope()
                        and sum(n.GetAtomicNum() > 1 for n in nitrogen.GetNeighbors()) == 1
                        and nitrogen.GetTotalValence() == 3
                        and nitrogen.GetTotalNumHs(includeNeighbors=True) == 0):
                    return "nitrile", frozenset({triple_nitrogens[0]})
        return None
    if group.kind == "acid":
        hydroxyl = [i for i in neighbors
                    if i != double_oxygens[0]
                    and mol.GetAtomWithIdx(i).GetAtomicNum() == 8
                    and mol.GetBondBetweenAtoms(center, i).GetBondTypeAsDouble() == 1]
        if len(hydroxyl) != 1 or set(group.atoms) != {center, double_oxygens[0], hydroxyl[0]}:
            return None
        carbonyl_o = mol.GetAtomWithIdx(double_oxygens[0])
        hydroxyl_o = mol.GetAtomWithIdx(hydroxyl[0])
        if (carbonyl_o.GetTotalValence() != 2 or carbonyl_o.GetTotalNumHs(includeNeighbors=True) != 0
                or sum(n.GetAtomicNum() > 1 for n in hydroxyl_o.GetNeighbors()) != 1
                or hydroxyl_o.GetTotalValence() != 2
                or hydroxyl_o.GetTotalNumHs(includeNeighbors=True) != 1):
            return None
        return "acid", frozenset({double_oxygens[0], hydroxyl[0]})
    if group.kind == "amide":
        nitrogens = [i for i in neighbors
                     if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
                     and mol.GetBondBetweenAtoms(center, i).GetBondTypeAsDouble() == 1]
        if len(nitrogens) != 1 or set(group.atoms) != {center, double_oxygens[0], nitrogens[0]}:
            return None
        nitrogen = mol.GetAtomWithIdx(nitrogens[0])
        oxygen = mol.GetAtomWithIdx(double_oxygens[0])
        if (nitrogen.GetFormalCharge() or nitrogen.GetIsotope()
                or sum(n.GetAtomicNum() > 1 for n in nitrogen.GetNeighbors()) != 1
                or nitrogen.GetTotalValence() != 3
                or nitrogen.GetTotalNumHs(includeNeighbors=True) != 2
                or oxygen.GetTotalValence() != 2
                or oxygen.GetTotalNumHs(includeNeighbors=True) != 0):
            return None
        return "amide", frozenset({double_oxygens[0], nitrogens[0]})
    return None


def acyl_replacement_decision(mol, Chem, groups, path, *, return_proof=False):
    """Render a fixed-numbered C/N/O/S heterochain with terminal suffixes.

    ``groups`` is the complete one- or two-element typed principal-group
    sequence and ``path`` is the caller's ordered original-atom parent path.
    Two-site support is limited to one suffix class at opposite chain ends.
    Set ``return_proof`` to obtain a frozen evidence sidecar around the same
    complete ``MethodDecision``.
    """
    if (mol.GetNumHeavyAtoms() > 128 or mol.GetRingInfo().NumRings()
            or not isinstance(path, (tuple, list)) or not path
            or len(groups) not in {1, 2} or mol.GetStereoGroups()):
        return None
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    if not inventory.supported or tuple(groups) != tuple(inventory.principal):
        return None
    group = groups[0]
    if any(candidate.kind != group.kind for candidate in groups):
        return None
    typed_groups = tuple(_typed_group(mol, candidate) for candidate in groups)
    if any(typed is None for typed in typed_groups):
        return None
    kinds = {typed[0] for typed in typed_groups}
    if len(kinds) != 1:
        return None
    kind = next(iter(kinds))
    external_by_center = {candidate.center: typed[1]
                          for candidate, typed in zip(groups, typed_groups)}
    if len(external_by_center) != len(groups):
        return None
    external_group_atoms = frozenset().union(*external_by_center.values())
    path = tuple(path)
    if (len(set(path)) != len(path)
            or any(isinstance(i, bool) or not isinstance(i, int)
                   or i < 0 or i >= mol.GetNumAtoms() for i in path)
            or any(candidate.center not in path for candidate in groups)):
        return None
    suffix_positions = tuple(path.index(candidate.center) for candidate in groups)
    if ((len(groups) == 1 and suffix_positions[0] not in {0, len(path) - 1})
            or (len(groups) == 2 and set(suffix_positions) != {0, len(path) - 1})):
        return None
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    parent_atoms = frozenset(path)
    if parent_atoms & external_group_atoms or not (parent_atoms | external_group_atoms) <= heavy:
        return None
    leaf_atoms = heavy - parent_atoms - external_group_atoms
    prefixes = tuple(replacement_leaf_prefix(mol, idx) for idx in sorted(leaf_atoms))
    if any(row is None or row[1] not in parent_atoms for row in prefixes):
        return None
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7, 8, 16} for i in path):
        return None
    if mol.GetAtomWithIdx(path[0]).GetAtomicNum() != 6 or mol.GetAtomWithIdx(path[-1]).GetAtomicNum() != 6:
        return None
    for atom in mol.GetAtoms():
        if (atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                or atom.GetIsotope() or int(atom.GetChiralTag())):
            return None
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope():
                return None
    if any(mol.GetAtomWithIdx(idx).GetTotalValence() != {
            6: 4, 7: 3, 8: 2, 16: 2}[mol.GetAtomWithIdx(idx).GetAtomicNum()]
            for idx in path):
        return None
    for idx in path:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 7 and (
                sum(n.GetAtomicNum() > 1 for n in atom.GetNeighbors()) != 2
                or atom.GetTotalNumHs(includeNeighbors=True) != 1):
            return None
    if any(b.GetStereo() != Chem.BondStereo.STEREONONE
           or b.GetBondDir() != Chem.BondDir.NONE for b in mol.GetBonds()):
        return None
    if any(mol.GetBondBetweenAtoms(a, b) is None
           or mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
           for a, b in zip(path, path[1:])):
        return None
    leaf_by_owner = {}
    for idx, owner, _ in prefixes:
        leaf_by_owner.setdefault(owner, set()).add(idx)
    for pos, atom_id in enumerate(path):
        permitted = set(path[max(0, pos - 1):pos]) | set(path[pos + 1:pos + 2])
        permitted.update(external_by_center.get(atom_id, ()))
        permitted.update(leaf_by_owner.get(atom_id, ()))
        actual = {n.GetIdx() for n in mol.GetAtomWithIdx(atom_id).GetNeighbors()
                  if n.GetAtomicNum() > 1}
        if actual != permitted:
            return None
    external_owner = {idx: center for center, owned in external_by_center.items()
                      for idx in owned}
    for idx in external_group_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() not in {7, 8}:
            return None
        if any(n.GetIdx() not in {external_owner[idx]} and n.GetAtomicNum() > 1
               for n in atom.GetNeighbors()):
            return None
    proof = acyclic_replacement_eligibility(
        mol, parent_atoms,
        principal_group_atoms=frozenset().union(*(candidate.atoms for candidate in groups)),
        has_principal_suffix=True)
    if not proof.eligible or proof.path not in {path, tuple(reversed(path))}:
        return None
    preferred = min((path, tuple(reversed(path))),
                    key=lambda candidate: _key(mol, candidate, groups, prefixes))
    if path != preferred:
        return None
    parent = _parent_text(mol, path)
    if not parent:
        return None
    suffix_locants = tuple(sorted(str(path.index(candidate.center) + 1)
                                  for candidate in groups))
    suffix_locant = suffix_locants[0]
    stem = parent[:-1]  # P-65/P-66 suffixes are added to the new parent hydride.
    if len(groups) == 1:
        suffix = {"acid": "oic acid", "amide": "amide", "nitrile": "nitrile"}[kind]
        rendered = f"{stem}-{suffix_locant}-{suffix}"
    else:
        # P-14.3.4.1 permits omission for acyclic diacid/diamide/dinitrile
        # terminal sites; P-51.4.1.3 also prints a locantless replacement
        # diacid. The interaction with P-15.4.3.2.3 is recorded in the audit.
        ending = {"acid": "edioic acid", "amide": "diamide",
                  "nitrile": "dinitrile"}[kind]
        rendered = (stem if kind == "acid" else parent) + ending
    from .naming_methods import MethodDecision
    locants = tuple((idx, str(pos)) for pos, idx in enumerate(path, 1))
    positions = dict(locants)
    rendered = decorate_replacement_prefixes(rendered, positions, prefixes)
    suffix_role_locants = ()
    all_locants = locants
    roles = (
        ("replacement_parent", parent_atoms, locants),
        ("principal_suffix", external_group_atoms, suffix_role_locants),
    )
    if prefixes:
        roles += (("substituent_prefixes", leaf_atoms,
                   tuple((idx, positions[owner]) for idx, owner, _ in prefixes)),)
    if (set().union(*(set(row[1]) for row in roles)) != set(heavy)
            or sum(len(row[1]) for row in roles) != len(heavy)
            or set(dict(all_locants)) != set(parent_atoms)):
        return None
    decision = MethodDecision(
        name=rendered, method="skeletal-replacement",
        principal_centers=frozenset(candidate.center for candidate in groups),
        atoms=heavy, rule=RULE,
        atom_locants=all_locants, heterounits=tuple(unit.atoms for unit in proof.heterounits),
        component_roles=roles)
    if return_proof:
        equal_paths = tuple(candidate for candidate in (path, tuple(reversed(path)))
                            if _key(mol, candidate, groups, prefixes)
                            == _key(mol, preferred, groups, prefixes))
        if len(groups) == 1:
            pin_status = "resolved"
            pin_reason = (
                "single terminal suffix locant is fixed by P-15.4.3.2.3; "
                "terminal replacement-acid locants are expressly required by P-65.1.2.1"
            ) if kind == "acid" else (
                "single terminal suffix follows the fixed replacement numbering; "
                "bounded terminal amide/nitrile grammar is source-closed"
            )
        else:
            pin_status = "pending"
            pin_reason = (
                "terminal disuffix locant omission is unresolved: P-14.3.4.1 and "
                "the locantless P-51.4.1.3 replacement diacid example support omission, "
                "while P-15.4.3.2.3 and P-65.1.2.1 require/illustrate locants for "
                "heteroacyclic replacement chains"
            )
        return AcylReplacementProof(decision, path, kind, suffix_locant,
                                    tuple(unit.atoms for unit in proof.heterounits), equal_paths,
                                    suffix_locants, pin_status, pin_reason, prefixes)
    return decision
