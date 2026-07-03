from __future__ import annotations

from dataclasses import dataclass

from .graph_model import graph_view
from .molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set


@dataclass(frozen=True, slots=True)
class RingSystemTopology:
    """Cached graph-theoretic facts for one connected ring system."""

    atoms: frozenset[int]
    graph: dict[int, frozenset[int]]
    edge_count: int
    cycle_rank: int
    branch_atoms: tuple[int, ...]


def ring_system_topologies(mol, Chem) -> tuple[RingSystemTopology, ...]:
    cached = mol_cache_get(mol, "ring_system_topologies")
    if cached is not None:
        return cached

    view = graph_view(mol, Chem)
    topologies = tuple(_build_topology(view, atoms) for atoms in view.ring_systems)
    return mol_cache_set(mol, "ring_system_topologies", topologies)


def ring_system_graph(mol, Chem, atoms) -> dict[int, frozenset[int]]:
    atom_set = frozenset(atoms)
    cache = mol_cache_dict(mol, "ring_system_graph_by_atoms")
    cached = cache.get(atom_set)
    if cached is not None:
        return cached
    for topology in ring_system_topologies(mol, Chem):
        if topology.atoms == atom_set:
            cache[atom_set] = topology.graph
            return topology.graph
    graph = _graph_for_atom_set(graph_view(mol, Chem), atom_set)
    cache[atom_set] = graph
    return graph


def _build_topology(view, atoms: frozenset[int]) -> RingSystemTopology:
    graph = _graph_for_atom_set(view, atoms)
    edge_count = sum(len(nbrs) for nbrs in graph.values()) // 2
    return RingSystemTopology(
        atoms=frozenset(atoms),
        graph=graph,
        edge_count=edge_count,
        cycle_rank=edge_count - len(atoms) + 1,
        branch_atoms=tuple(sorted(idx for idx, nbrs in graph.items() if len(nbrs) >= 3)),
    )


def _graph_for_atom_set(view, atom_set: frozenset[int]) -> dict[int, frozenset[int]]:
    graph: dict[int, set[int]] = {idx: set() for idx in atom_set}
    for (left, right), _order in view.bond_orders.items():
        if left in atom_set and right in atom_set:
            graph[left].add(right)
            graph[right].add(left)
    return {idx: frozenset(nbrs) for idx, nbrs in graph.items()}
