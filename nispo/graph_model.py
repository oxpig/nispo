from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .molecule_cache import mol_cache_get, mol_cache_set


@dataclass(frozen=True, slots=True)
class MoleculeGraphView:
    """Cached immutable graph facts shared by naming rule families."""

    atom_count: int
    ring_count: int
    symbols: tuple[str, ...]
    atomic_nums: tuple[int, ...]
    formal_charges: tuple[int, ...]
    aromatic_atoms: tuple[bool, ...]
    ring_atoms: tuple[bool, ...]
    neighbors: tuple[tuple[int, ...], ...]
    bond_orders: dict[tuple[int, int], float]
    elements: Counter
    atom_rings: tuple[tuple[int, ...], ...]
    ring_systems: tuple[frozenset[int], ...]


def graph_view(mol, Chem) -> MoleculeGraphView:
    cached = mol_cache_get(mol, "graph_view")
    if cached is not None:
        return cached

    ring_info = mol.GetRingInfo()
    try:
        ring_count = ring_info.NumRings()
    except RuntimeError:
        if Chem is None:
            ring_count = 0
        else:
            Chem.GetSymmSSSR(mol)
            ring_info = mol.GetRingInfo()
            ring_count = ring_info.NumRings()
    atom_rings = tuple(tuple(ring) for ring in ring_info.AtomRings()) if ring_count else ()
    ring_systems = _ring_systems_from_atom_rings(atom_rings)
    atoms = tuple(mol.GetAtoms())
    symbols = tuple(atom.GetSymbol() for atom in atoms)
    view = MoleculeGraphView(
        atom_count=mol.GetNumAtoms(),
        ring_count=ring_count,
        symbols=symbols,
        atomic_nums=tuple(atom.GetAtomicNum() for atom in atoms),
        formal_charges=tuple(atom.GetFormalCharge() for atom in atoms),
        aromatic_atoms=tuple(atom.GetIsAromatic() for atom in atoms),
        ring_atoms=tuple(atom.IsInRing() for atom in atoms),
        neighbors=tuple(tuple(nbr.GetIdx() for nbr in atom.GetNeighbors()) for atom in atoms),
        bond_orders={
            _bond_key(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()): bond.GetBondTypeAsDouble()
            for bond in mol.GetBonds()
        },
        elements=Counter(symbols),
        atom_rings=atom_rings,
        ring_systems=ring_systems,
    )
    return mol_cache_set(mol, "graph_view", view)


def _ring_systems_from_atom_rings(atom_rings: tuple[tuple[int, ...], ...]) -> tuple[frozenset[int], ...]:
    if not atom_rings:
        return ()
    systems: list[set[int]] = []
    for ring in atom_rings:
        current = set(ring)
        rest = []
        for system in systems:
            if current & system:
                current |= system
            else:
                rest.append(system)
        rest.append(current)
        systems = rest

    changed = True
    while changed:
        changed = False
        merged = []
        while systems:
            current = systems.pop()
            overlapping = [system for system in systems if current & system]
            if overlapping:
                changed = True
                for system in overlapping:
                    current |= system
                    systems.remove(system)
            merged.append(current)
        systems = merged
    return tuple(frozenset(system) for system in systems)


def _bond_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)
