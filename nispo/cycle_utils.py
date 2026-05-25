from __future__ import annotations

from collections.abc import Sequence


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
    for ring in mol.GetRingInfo().AtomRings():
        ring_set = set(ring)
        if ring_set & core_set and not ring_set <= core_set:
            return True
    return False
