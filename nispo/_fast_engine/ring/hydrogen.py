from __future__ import annotations

from collections.abc import Sequence


def _indicated_hydrogen_prefix(mol, ordered: Sequence[int], seq: Sequence[int], indicated: int | None) -> str:
    if indicated is not None and indicated in seq:
        return f"{seq.index(indicated) + 1}H-"
    has_ring_n_h = any(
        mol.GetAtomWithIdx(idx).GetSymbol() == "N" and mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0
        for idx in ordered
    )
    return "1H-" if has_ring_n_h else ""


def _drop_indicated_hydrogen_prefix(parent_name: str) -> str:
    if len(parent_name) > 3 and parent_name[0].isdigit() and parent_name[1:3] == "H-":
        return parent_name[3:]
    return parent_name
