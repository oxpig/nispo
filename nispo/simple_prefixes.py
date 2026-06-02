from __future__ import annotations

from .constants import HALO


def simple_external_prefix(mol, idx: int, parent_idx: int) -> str:
    """Render safe terminal substituents without entering branch dispatch."""

    atom = mol.GetAtomWithIdx(idx)
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None:
        return ""
    order = bond.GetBondTypeAsDouble()
    symbol = atom.GetSymbol()
    if symbol in HALO:
        return _terminal_halogen_prefix(atom, parent_idx, order)
    if symbol == "C":
        return _simple_fluoromethyl_prefix(mol, idx, parent_idx, order) or _simple_linear_alkyl_prefix(
            mol,
            idx,
            parent_idx,
            order,
        )
    if symbol == "O":
        return _simple_oxygen_prefix(mol, atom, idx, parent_idx, order)
    if symbol == "N":
        return _simple_nitrogen_prefix(atom, parent_idx, order)
    return ""


def _terminal_halogen_prefix(atom, parent_idx: int, order: float) -> str:
    if order != 1.0 or atom.GetFormalCharge() != 0:
        return ""
    if any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
        return ""
    isotope = atom.GetIsotope()
    symbol = atom.GetSymbol()
    return f"[{isotope}{symbol}]{HALO[symbol]}" if isotope else HALO[symbol]


def _simple_fluoromethyl_prefix(mol, idx: int, parent_idx: int, order: float) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if order != 1.0 or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return ""
    if _atom_has_isotopic_hydrogen(atom):
        return ""
    external = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if not external:
        return ""
    if any(mol.GetAtomWithIdx(nidx).GetSymbol() != "F" for nidx in external):
        return ""
    count = len(external)
    if count == 1:
        return "fluoromethyl"
    if count == 2:
        return "difluoromethyl"
    if count == 3:
        return "trifluoromethyl"
    return ""


def _simple_linear_alkyl_prefix(mol, idx: int, parent_idx: int, order: float) -> str:
    names = {1: "methyl", 2: "ethyl", 3: "propyl", 4: "butyl"}
    if order != 1.0:
        return ""
    chain = []
    previous = parent_idx
    current = idx
    while len(chain) < 4:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            return ""
        if atom.GetIsotope() or _atom_has_isotopic_hydrogen(atom):
            return ""
        bond = mol.GetBondBetweenAtoms(previous, current)
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return ""
        chain.append(current)
        heavy = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetIdx() != previous and nbr.GetAtomicNum() > 1
        ]
        if not heavy:
            return names[len(chain)]
        if len(heavy) != 1:
            return ""
        next_idx = heavy[0]
        if mol.GetAtomWithIdx(next_idx).GetSymbol() != "C":
            return ""
        previous, current = current, next_idx
    return ""


def _simple_oxygen_prefix(mol, atom, idx: int, parent_idx: int, order: float) -> str:
    if atom.IsInRing() or atom.GetIsotope() or _atom_has_isotopic_hydrogen(atom):
        return ""
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if order == 2.0 and atom.GetFormalCharge() == 0 and not heavy:
        return "oxo"
    if order != 1.0:
        return ""
    if atom.GetFormalCharge() < 0 and not heavy:
        return "oxido"
    if atom.GetFormalCharge() != 0:
        return ""
    if not heavy:
        return "hydroxy"
    if len(heavy) == 1:
        return _simple_linear_alkoxy_prefix(mol, heavy[0], idx)
    return ""


def _simple_linear_alkoxy_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    names = {1: "methoxy", 2: "ethoxy", 3: "propoxy", 4: "butoxy"}
    chain = []
    previous = oxygen_idx
    current = carbon_idx
    while len(chain) < 4:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            return ""
        if atom.GetIsotope() or _atom_has_isotopic_hydrogen(atom):
            return ""
        bond = mol.GetBondBetweenAtoms(previous, current)
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return ""
        chain.append(current)
        heavy = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetIdx() != previous and nbr.GetAtomicNum() > 1
        ]
        if not heavy:
            return names[len(chain)]
        if len(heavy) != 1:
            return ""
        next_idx = heavy[0]
        if mol.GetAtomWithIdx(next_idx).GetSymbol() != "C":
            return ""
        previous, current = current, next_idx
    return ""


def _simple_nitrogen_prefix(atom, parent_idx: int, order: float) -> str:
    if atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope() or _atom_has_isotopic_hydrogen(atom):
        return ""
    if any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
        return ""
    if order == 1.0:
        return "amino"
    if order == 2.0:
        return "imino"
    return ""


def _atom_has_isotopic_hydrogen(atom) -> bool:
    return any(nbr.GetAtomicNum() == 1 and nbr.GetIsotope() for nbr in atom.GetNeighbors())
