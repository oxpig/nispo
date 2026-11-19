from __future__ import annotations

from collections.abc import Sequence

from .molecule_iteration import iter_atoms
from .molecule_cache import mol_cache_dict


def match_has_extra_core_bonds(mol, query, match: Sequence[int]) -> bool:
    """Return true when a matched atom set has topology beyond the query."""
    return _core_bond_count(mol, match) > query.GetNumBonds()


def match_preserves_formal_charges(mol, query, match: Sequence[int]) -> bool:
    """Retained aliases without charge rendering must match each atomic charge."""
    return all(
        atom.GetFormalCharge() == mol.GetAtomWithIdx(match[atom.GetIdx()]).GetFormalCharge()
        for atom in iter_atoms(query)
    )


def match_preserves_aromatic_nitrogen_hydrogens(mol, query, match: Sequence[int]) -> bool:
    """Reject aromatic N-H tautomer mismatches in unmodified ring cores.

    RDKit molecule queries match aromatic ``n`` and ``[nH]`` without enforcing
    their hydrogen counts. Moving an indicated hydrogen changes the named
    structure when an asymmetric decoration fixes the ring orientation.
    """
    core = set(match)
    # Exocyclic multiple bonds can shift indicated H via suffix rules. Their
    # renderers carry that information separately. Ordinary N substitution
    # consumes hydrogen at that same nitrogen; it does not authorize moving
    # the template's indicated H to another atom of the ring.
    for atom_idx in match:
        atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            if bond.GetOtherAtomIdx(atom_idx) in core:
                continue
            if bond.GetBondTypeAsDouble() > 1:
                return True
    for query_atom in iter_atoms(query):
        if query_atom.GetAtomicNum() != 7 or not query_atom.GetIsAromatic():
            continue
        atom = mol.GetAtomWithIdx(match[query_atom.GetIdx()])
        if not atom.GetIsAromatic():
            continue
        if atom.GetFormalCharge() != query_atom.GetFormalCharge():
            continue  # Charge renderers handle these separately.
        external_valence = sum(
            bond.GetBondTypeAsDouble()
            for bond in atom.GetBonds()
            if bond.GetOtherAtomIdx(atom.GetIdx()) not in core
        )
        if query_atom.GetTotalNumHs() != atom.GetTotalNumHs() + external_valence:
            return False
    return True


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
