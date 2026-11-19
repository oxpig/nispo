from __future__ import annotations

from collections.abc import Sequence

from .molecule_cache import mol_cache_dict


def ordered_ring(mol, ring: Sequence[int]) -> tuple[int, ...]:
    ring_set = set(ring)
    start = min(ring)
    ordered = [start]
    prev = None
    current = start
    while len(ordered) < len(ring):
        nbrs = sorted(
            n.GetIdx()
            for n in mol.GetAtomWithIdx(current).GetNeighbors()
            if n.GetIdx() in ring_set and n.GetIdx() != prev
        )
        nxt = None
        for candidate in nbrs:
            if candidate not in ordered:
                nxt = candidate
                break
        if nxt is None:
            return ()
        prev, current = current, nxt
        ordered.append(current)
    if not mol.GetBondBetweenAtoms(ordered[-1], ordered[0]):
        return ()
    return tuple(ordered)


def ring_system_extends_core(mol, core_set: set[int]) -> bool:
    cache = mol_cache_dict(mol, "ring_system_extends_core_cache")
    key = frozenset(core_set)
    cached = cache.get(key)
    if cached is not None:
        return cached
    ring_sets_cache = mol_cache_dict(mol, "atom_ring_set_cache")
    ring_sets = ring_sets_cache.get("rings")
    if ring_sets is None:
        ring_sets = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
        ring_sets_cache["rings"] = ring_sets
    for ring_set in ring_sets:
        if ring_set & core_set and not ring_set <= core_set:
            cache[key] = True
            return True
    cache[key] = False
    return False
