from __future__ import annotations

from collections.abc import Sequence

from .molecule_cache import mol_cache_dict


def match_has_extra_core_bonds(mol, query, match: Sequence[int]) -> bool:
    """Return true when a matched atom set has topology beyond the query."""
    return _core_bond_count(mol, match) > query.GetNumBonds()


def _core_bond_count(mol, match: Sequence[int]) -> int:
    atoms = tuple(sorted(match))
    cache = mol_cache_dict(mol, "core_bond_count_cache")
    cached = cache.get(atoms)
    if cached is not None:
        return cached
    atom_set = set(atoms)
    count = 0
    for atom_idx in atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in atom_set and atom_idx < nbr_idx:
                count += 1
    cache[atoms] = count
    return count
