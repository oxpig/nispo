from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set


@dataclass(frozen=True, slots=True)
class MoleculeGraphView:
    """Cached immutable graph facts shared by naming rule families."""

    atom_count: int
    ring_count: int
    symbols: tuple[str, ...]
    atomic_nums: tuple[int, ...]
    isotopes: tuple[int, ...]
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
    atoms = tuple(iter_atoms(mol))
    symbols = tuple(atom.GetSymbol() for atom in atoms)
    native_edges = type(mol).__module__ == "rdkit.Chem.rdchem" and type(mol).__name__ in {"Mol", "RWMol"}
    if native_edges:
        neighbors = [[] for _ in atoms]
        bond_orders = {}
        for bond in iter_bonds(mol):
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            neighbors[left].append(right)
            neighbors[right].append(left)
            bond_orders[_bond_key(left, right)] = bond.GetBondTypeAsDouble()
        neighbors = tuple(tuple(row) for row in neighbors)
    view = MoleculeGraphView(
        atom_count=mol.GetNumAtoms(),
        ring_count=ring_count,
        symbols=symbols,
        atomic_nums=tuple(atom.GetAtomicNum() for atom in atoms),
        isotopes=tuple(atom.GetIsotope() for atom in atoms),
        formal_charges=tuple(atom.GetFormalCharge() for atom in atoms),
        aromatic_atoms=tuple(atom.GetIsAromatic() for atom in atoms),
        ring_atoms=tuple(atom.IsInRing() for atom in atoms),
        neighbors=(neighbors if native_edges else
                   tuple(tuple(nbr.GetIdx() for nbr in atom.GetNeighbors()) for atom in atoms)),
        bond_orders=(bond_orders if native_edges else {
            _bond_key(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()): bond.GetBondTypeAsDouble()
            for bond in iter_bonds(mol)
        }),
        elements=Counter(symbols),
        atom_rings=atom_rings,
        ring_systems=ring_systems,
    )
    return mol_cache_set(mol, "graph_view", view)


def mapped_graph_components(mol, mapped_atoms, *, include_isotopic_hydrogens=False):
    """Partition atoms outside one original-atom scope using immutable facts.

    Components retain every mapped attachment edge, including repeated hosts.
    Callers separately prove ownership, annotation coverage and valid attachments.
    Ordinary explicit hydrogen atoms are excluded, matching graph ownership;
    annotation consumers may additionally include isotope-bearing hydrogens.
    """
    view = graph_view(mol, None)
    mapped_atoms = frozenset(mapped_atoms)
    include_isotopic_hydrogens = bool(include_isotopic_hydrogens and any(
        number == 1 and mass for number, mass in zip(view.atomic_nums, view.isotopes)))
    key = mapped_atoms, include_isotopic_hydrogens
    cache = mol_cache_dict(mol, "mapped_graph_components")
    cached = cache.get(key)
    if cached is not None:
        return cached
    remaining = {i for i, number in enumerate(view.atomic_nums)
                 if number > 1 or include_isotopic_hydrogens and view.isotopes[i]} - mapped_atoms
    components = []
    while remaining:
        members, attachments, stack = set(), [], [next(iter(remaining))]
        while stack:
            index = stack.pop()
            if index not in remaining:
                continue
            remaining.remove(index)
            members.add(index)
            for other in view.neighbors[index]:
                if other in mapped_atoms:
                    attachments.append(other)
                elif other in remaining:
                    stack.append(other)
        components.append((frozenset(members), tuple(attachments)))
    result = tuple(components)
    cache[key] = result
    return result


def method_bond_counts(mol, Chem):
    """Share original element-pair/bond-order signals between method gates."""
    cached = mol_cache_get(mol, "method_bond_counts")
    if cached is None:
        view = graph_view(mol, Chem)
        numbers = view.atomic_nums
        cached = Counter((min(numbers[a], numbers[b]), max(numbers[a], numbers[b]), order)
                         for (a, b), order in view.bond_orders.items())
        mol_cache_set(mol, "method_bond_counts", cached)
    return cached


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
