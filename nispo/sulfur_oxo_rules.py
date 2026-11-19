from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from .molecule_iteration import iter_atoms
from .branch_rules import branch_prefix
from .molecule_cache import mol_cache_get, mol_cache_set
from .rule_engine import NameCandidate
from .rule_priority import parent_priority
from .simple_prefixes import simple_external_prefix


class SulfurOxoRetainedParentRule:
    family = "sulfur-oxo-retained-parent"
    priority = parent_priority("senior-functional-exception")

    def __init__(self, Chem):
        self._Chem = Chem

    def candidates(self, ctx):
        mol = ctx.mol
        seen: set[str] = set()
        for name in (
            *_sulfinamide_names(mol, self._Chem),
            *_sulfoximine_names(mol, self._Chem),
            *_sulfonylium_names(mol, self._Chem),
            *_sulfonyl_hydride_names(mol, self._Chem),
            *_thiosulfur_oxo_names(mol, self._Chem),
        ):
            if name in seen:
                continue
            seen.add(name)
            yield NameCandidate(
                name,
                self.family,
                self.priority,
                sort_key=(len(name), name),
                parent_atoms=_heavy_atom_indices(mol),
                source_family="sulfur-oxo-retained",
                estimated_length=len(name),
            )


def _sulfinamide_names(mol, Chem) -> list[str]:
    names: list[str] = []
    atom_set = _heavy_atom_indices(mol)
    for sulfur in iter_atoms(mol):
        if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0 or sulfur.IsInRing():
            continue
        record = _sulfinamide_record(mol, Chem, sulfur.GetIdx())
        if not record or record["covered_atoms"] != atom_set:
            continue
        substituents = _external_substituent_names(mol, record["nitrogen"], sulfur.GetIdx())
        if substituents is None or not substituents:
            continue
        prefix = _n_substituent_prefix(substituents)
        if prefix is None:
            continue
        names.append(prefix + "sulfinamide")
    return names


def _sulfinamide_record(mol, Chem, sulfur_idx: int) -> dict | None:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    if sulfur.GetTotalNumHs() != 1:
        return None
    double_o: list[int] = []
    single_n: list[int] = []
    covered = {sulfur_idx}
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_o.append(nidx)
            covered.add(nidx)
        elif nbr.GetSymbol() == "N" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.SINGLE:
            single_n.append(nidx)
            covered.add(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_o) != 1 or len(single_n) != 1:
        return None
    nitrogen_idx = single_n[0]
    for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == sulfur_idx or nbr.GetAtomicNum() <= 1:
            continue
        covered.update(_collect_component_atoms(mol, nidx, nitrogen_idx))
    return {"nitrogen": nitrogen_idx, "covered_atoms": covered}


def _sulfoximine_names(mol, Chem) -> list[str]:
    names: list[str] = []
    atom_set = _heavy_atom_indices(mol)
    for sulfur in iter_atoms(mol):
        if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0 or sulfur.IsInRing():
            continue
        record = _sulfoximine_record(mol, Chem, sulfur.GetIdx())
        if not record or record["covered_atoms"] != atom_set:
            continue
        substituents = _external_substituent_names(mol, record["imino_nitrogen"], sulfur.GetIdx())
        if substituents is None or len(substituents) != 1:
            continue
        prefix = _n_substituent_prefix(substituents)
        if prefix is None:
            continue
        names.append(prefix + "sulfoximine")
    return names


def _sulfoximine_record(mol, Chem, sulfur_idx: int) -> dict | None:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    if sulfur.GetTotalNumHs() != 2:
        return None
    double_o: list[int] = []
    double_n: list[int] = []
    covered = {sulfur_idx}
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_o.append(nidx)
            covered.add(nidx)
        elif nbr.GetSymbol() == "N" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_n.append(nidx)
            covered.add(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_o) != 1 or len(double_n) != 1:
        return None
    nitrogen_idx = double_n[0]
    for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == sulfur_idx or nbr.GetAtomicNum() <= 1:
            continue
        covered.update(_collect_component_atoms(mol, nidx, nitrogen_idx))
    return {"imino_nitrogen": nitrogen_idx, "covered_atoms": covered}


def _sulfonylium_names(mol, Chem) -> list[str]:
    names: list[str] = []
    atom_set = _heavy_atom_indices(mol)
    for sulfur in iter_atoms(mol):
        if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 1 or sulfur.IsInRing():
            continue
        sidx = sulfur.GetIdx()
        double_o: list[int] = []
        carbon_neighbors: list[int] = []
        covered = {sidx}
        ok = True
        for nbr in sulfur.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(sidx, nidx)
            if bond is None:
                ok = False
                break
            if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_o.append(nidx)
                covered.add(nidx)
            elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                carbon_neighbors.append(nidx)
            elif nbr.GetAtomicNum() > 1:
                ok = False
                break
        if not ok or len(double_o) != 2 or len(carbon_neighbors) != 1:
            continue
        owner_idx = carbon_neighbors[0]
        owner = _safe_owner_prefix(mol, owner_idx, sidx)
        if not owner:
            continue
        covered.update(_collect_component_atoms(mol, owner_idx, sidx))
        if covered == atom_set:
            names.append(owner + "sulfonylium")
    return names


def _sulfonyl_hydride_names(mol, Chem) -> list[str]:
    names: list[str] = []
    atom_set = _heavy_atom_indices(mol)
    for sulfur in iter_atoms(mol):
        if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0 or sulfur.IsInRing():
            continue
        sidx = sulfur.GetIdx()
        if sulfur.GetTotalNumHs() != 1:
            continue
        double_o: list[int] = []
        carbon_neighbors: list[int] = []
        covered = {sidx}
        ok = True
        for nbr in sulfur.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(sidx, nidx)
            if bond is None:
                ok = False
                break
            if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_o.append(nidx)
                covered.add(nidx)
            elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                carbon_neighbors.append(nidx)
            elif nbr.GetAtomicNum() > 1:
                ok = False
                break
        if not ok or len(double_o) != 2 or len(carbon_neighbors) != 1:
            continue
        owner_idx = carbon_neighbors[0]
        owner = _safe_owner_prefix(mol, owner_idx, sidx)
        if not owner:
            continue
        covered.update(_collect_component_atoms(mol, owner_idx, sidx))
        if covered == atom_set:
            names.append(owner + "sulfonyl hydride")
    return names


def _thiosulfur_oxo_names(mol, Chem) -> list[str]:
    names: list[str] = []
    atom_set = _heavy_atom_indices(mol)
    for center in iter_atoms(mol):
        if center.GetSymbol() != "S" or center.GetFormalCharge() != 0 or center.IsInRing():
            continue
        record = _thiosulfur_oxo_record(mol, Chem, center.GetIdx())
        if not record or record["covered_atoms"] != atom_set:
            continue
        owner = record["owner"]
        kind = record["kind"]
        if kind == "thiosulfonate":
            names.append(f"S-{owner} thiosulfonate")
        elif kind == "thiosulfinate":
            names.append(f"S-{owner} thiosulfinate")
        elif kind == "sulfinyl-sulfinate":
            names.append(f"{owner}sulfinyl sulfinate")
        elif kind == "sulfonyl-sulfinate":
            names.append(f"{owner}sulfonyl sulfinate")
    return names


def _thiosulfur_oxo_record(mol, Chem, center_idx: int) -> dict | None:
    center = mol.GetAtomWithIdx(center_idx)
    double_o: list[int] = []
    terminal_oxido: list[int] = []
    single_sulfur: list[int] = []
    covered = {center_idx}
    for nbr in center.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(center_idx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_o.append(nidx)
            covered.add(nidx)
        elif (
            nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() == -1
            and len(nbr.GetNeighbors()) == 1
            and bond.GetBondType() == Chem.BondType.SINGLE
        ):
            terminal_oxido.append(nidx)
            covered.add(nidx)
        elif nbr.GetSymbol() == "S" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.SINGLE:
            single_sulfur.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(single_sulfur) != 1:
        return None
    ligand_record = _sulfur_oxo_ligand_record(mol, Chem, single_sulfur[0], center_idx)
    if ligand_record is None:
        return None
    ligand_double_o = ligand_record["double_o"]
    if center.GetTotalNumHs() == 1 and not terminal_oxido and ligand_double_o == 0:
        if len(double_o) == 2:
            kind = "thiosulfonate"
        elif len(double_o) == 1:
            kind = "thiosulfinate"
        else:
            return None
    elif center.GetTotalNumHs() == 0 and len(double_o) == 1 and len(terminal_oxido) == 1:
        if ligand_double_o == 1:
            kind = "sulfinyl-sulfinate"
        elif ligand_double_o == 2:
            kind = "sulfonyl-sulfinate"
        else:
            return None
    else:
        return None
    covered.add(single_sulfur[0])
    covered.update(ligand_record["covered_atoms"])
    return {
        "owner": ligand_record["owner"],
        "kind": kind,
        "covered_atoms": covered,
    }


def _sulfur_oxo_ligand_record(mol, Chem, sulfur_idx: int, center_idx: int) -> dict | None:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    double_o = 0
    carbon_neighbors: list[int] = []
    covered = {sulfur_idx}
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == center_idx:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_o += 1
            covered.add(nidx)
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_neighbors.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(carbon_neighbors) != 1 or double_o > 2:
        return None
    owner_idx = carbon_neighbors[0]
    owner = _safe_owner_prefix(mol, owner_idx, sulfur_idx)
    if not owner:
        return None
    covered.update(_collect_component_atoms(mol, owner_idx, sulfur_idx))
    return {"owner": owner, "double_o": double_o, "covered_atoms": covered}


def _external_substituent_names(mol, atom_idx: int, blocked_idx: int) -> list[str] | None:
    names = []
    for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == blocked_idx or nbr.GetAtomicNum() <= 1:
            continue
        name = _safe_owner_prefix(mol, nidx, atom_idx)
        if not name:
            return None
        names.append(name)
    return names


def _safe_owner_prefix(mol, atom_idx: int, blocked_idx: int) -> str:
    name = simple_external_prefix(mol, atom_idx, blocked_idx) or branch_prefix(
        mol,
        atom_idx,
        blocked_idx,
        assume_simple_external_prefix_miss=True,
    )
    name = _substituent_alias(name)
    if not _safe_substituent_name(name):
        return ""
    return name


def _n_substituent_prefix(substituents: Sequence[str]) -> str | None:
    if len(substituents) > 2:
        return None
    counts = Counter(_substituent_alias(name) for name in substituents)
    if len(counts) == 1:
        name = next(iter(counts))
        if not _safe_substituent_name(name):
            return None
        if counts[name] == 1:
            return _n_substituent_piece("N", name)
        if counts[name] == 2:
            return _duplicate_n_substituent_piece("N,N", name)
        return None
    first, second = sorted(counts)
    if counts[first] != 1 or counts[second] != 1:
        return None
    if not _safe_substituent_name(first) or not _safe_substituent_name(second):
        return None
    return f"{_n_substituent_piece('N', first)}-{_n_substituent_piece('N', second)}"


def _n_substituent_piece(locant: str, name: str) -> str:
    if _simple_unparenthesized_substituent(name):
        return f"{locant}-{name}"
    return f"{locant}-({_strip_outer_parentheses(name)})"


def _duplicate_n_substituent_piece(locants: str, name: str) -> str:
    if _simple_unparenthesized_substituent(name):
        return f"{locants}-di{name}"
    return f"{locants}-bis({_strip_outer_parentheses(name)})"


def _simple_unparenthesized_substituent(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9']+", name))


def _safe_substituent_name(name: str) -> bool:
    if not name or len(name) > 120:
        return False
    if " " in name or "[" in name or "]" in name:
        return False
    if not re.fullmatch(r"[A-Za-z0-9,().'\-]+", name):
        return False
    return not any(fragment in name for fragment in ("phosph", "sulfon", "sulfin"))


def _substituent_alias(name: str) -> str:
    aliases = {
        "phenylmethyl": "benzyl",
        "prop-2-ynyl": "propargyl",
        "prop-2-yn-1-yl": "propargyl",
    }
    return aliases.get(name, name)


def _strip_outer_parentheses(name: str) -> str:
    if name.startswith("(") and name.endswith(")"):
        return name[1:-1]
    return name


def _collect_component_atoms(mol, start_idx: int, blocked_idx: int) -> set[int]:
    seen = {blocked_idx}
    stack = [start_idx]
    component: set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        component.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in seen and nbr.GetAtomicNum() > 1:
                stack.append(nidx)
    return component


def _heavy_atom_indices(mol) -> frozenset[int]:
    cached = mol_cache_get(mol, "heavy_atom_indices")
    if cached is not None:
        return cached
    return mol_cache_set(
        mol,
        "heavy_atom_indices",
        frozenset(atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1),
    )
