from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AtomReference:
    kind: str
    value: str


def apply_token_operations(Chem, mol, labels: tuple[str, ...], token):
    for attr, applier in (
        ("addGroup", _apply_add_group),
        ("addHeteroAtom", _apply_add_hetero_atom),
        ("addBond", _apply_add_bond),
    ):
        spec = token.attrib.get(attr)
        if not spec:
            continue
        mol = applier(Chem, mol, labels, spec)
        if mol is None:
            return None
    return mol


def _apply_add_group(Chem, mol, labels: tuple[str, ...], spec: str):
    editable = Chem.RWMol(mol)
    group_specs = _parse_group_specs(spec, allow_label=True)
    if group_specs is None:
        return None
    for group, ref, _label in group_specs:
        atom_idx = _resolve_atom_reference(labels, ref)
        if atom_idx is None:
            return None
        if group == "=O":
            new_idx = editable.AddAtom(Chem.Atom("O"))
            editable.AddBond(atom_idx, new_idx, Chem.BondType.DOUBLE)
            continue
        group_mol = _mol_from_opsin_smiles(Chem, group)
        if group_mol is None or group_mol.GetNumAtoms() == 0:
            return None
        offset = editable.GetNumAtoms()
        for atom in group_mol.GetAtoms():
            editable.AddAtom(Chem.Atom(atom))
        for bond in group_mol.GetBonds():
            editable.AddBond(
                offset + bond.GetBeginAtomIdx(),
                offset + bond.GetEndAtomIdx(),
                bond.GetBondType(),
            )
        editable.AddBond(atom_idx, offset, Chem.BondType.SINGLE)
    return _sanitized(Chem, editable)


def _apply_add_hetero_atom(Chem, mol, labels: tuple[str, ...], spec: str):
    editable = Chem.RWMol(mol)
    group_specs = _parse_group_specs(spec, allow_label=False)
    if group_specs is None:
        return None
    for atom_smiles, ref, _label in group_specs:
        atom_idx = _resolve_atom_reference(labels, ref)
        if atom_idx is None:
            return None
        replacement = _replacement_atom(Chem, atom_smiles, mol.GetAtomWithIdx(atom_idx))
        if replacement is None:
            return None
        editable.ReplaceAtom(atom_idx, replacement, preserveProps=False)
    return _sanitized(Chem, editable)


def _apply_add_bond(Chem, mol, labels: tuple[str, ...], spec: str):
    editable = Chem.RWMol(mol)
    group_specs = _parse_group_specs(spec, allow_label=False)
    if group_specs is None:
        return None
    for order_text, ref, _label in group_specs:
        try:
            order = int(order_text)
        except ValueError:
            return None
        if order not in {2, 3}:
            return None
        start_idx = _resolve_atom_reference(labels, ref)
        if start_idx is None:
            return None
        end_idx = _next_locant_neighbor(mol, labels, start_idx)
        if end_idx is None:
            return None
        bond = editable.GetBondBetweenAtoms(start_idx, end_idx)
        if bond is None:
            return None
        bond.SetBondType(Chem.BondType.DOUBLE if order == 2 else Chem.BondType.TRIPLE)
    return _sanitized(Chem, editable)


def _parse_group_specs(spec: str, allow_label: bool) -> list[tuple[str, AtomReference, str]] | None:
    parsed = []
    for raw_part in spec.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        pieces = part.split()
        if len(pieces) < 3:
            return None
        if len(pieces) > (4 if allow_label else 3):
            return None
        ref_kind = pieces[1].lower()
        if ref_kind not in {"locant", "defaultlocant", "id", "defaultid"}:
            return None
        parsed.append((pieces[0], AtomReference(ref_kind, pieces[2]), pieces[3] if len(pieces) == 4 else ""))
    return parsed


def _resolve_atom_reference(labels: tuple[str, ...], ref: AtomReference) -> int | None:
    if ref.value == "required":
        return None
    if ref.kind in {"id", "defaultid"}:
        try:
            idx = int(ref.value) - 1
        except ValueError:
            return None
        return idx if 0 <= idx < len(labels) else None
    for idx, label in enumerate(labels):
        if label == ref.value:
            return idx
    return None


def _next_locant_neighbor(mol, labels: tuple[str, ...], start_idx: int) -> int | None:
    label = labels[start_idx]
    if label.isdigit():
        next_label = str(int(label) + 1)
        for idx, candidate in enumerate(labels):
            if candidate == next_label and mol.GetBondBetweenAtoms(start_idx, idx) is not None:
                return idx
    for nbr in mol.GetAtomWithIdx(start_idx).GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if labels[nbr_idx].isdigit():
            return nbr_idx
    return None


def _mol_from_opsin_smiles(Chem, smiles: str):
    normalized = _strip_opsin_valence_annotations(smiles)
    if normalized.startswith("-"):
        normalized = normalized[1:]
    if normalized.startswith("="):
        return None
    try:
        return Chem.MolFromSmiles(normalized)
    except Exception:
        return None


def _replacement_atom(Chem, atom_smiles: str, original_atom):
    atom_mol = _mol_from_opsin_smiles(Chem, atom_smiles)
    if atom_mol is None or atom_mol.GetNumAtoms() != 1:
        return None
    source = atom_mol.GetAtomWithIdx(0)
    replacement = Chem.Atom(source)
    if original_atom.GetIsAromatic():
        replacement.SetIsAromatic(True)
    return replacement


def _strip_opsin_valence_annotations(smiles: str) -> str:
    return re.sub(r"\[([A-Za-z][a-z]?)(?:H\d*)?\|[^]\s]+]", r"[\1]", smiles)


def _sanitized(Chem, editable):
    try:
        new_mol = editable.GetMol()
        Chem.SanitizeMol(new_mol)
    except Exception:
        return None
    return new_mol
