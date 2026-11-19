"""Intrinsic heteroatom-locant omission for ordinary Hantzsch–Widman rings.

P-22.2.2.1.7 permits all heteroatom locants to be omitted when the parent
element placement is unambiguous. For the bounded C/N/O/S/Se/Te monocycle route,
uniqueness is proved by enumerating the element sequences with the same
multiset and counting their distinct classes under the dihedral symmetries of
the ring. Enumeration stops as soon as a second class is found.

This operation decides only whether parent heteroatom locants may be omitted.
It does not change numbering, indicated-hydrogen locants, suffix locants, or
the original atom-to-locant maps used by the caller.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal


Status = Literal["omit", "required", "unknown"]
_ELEMENTS = frozenset({6, 7, 8, 16, 34, 52})


@dataclass(frozen=True, slots=True)
class HeteroLocantDecision:
    status: Status
    ring_size: int | None
    element_counts: tuple[tuple[int, int], ...]
    arrangement_classes: int | None
    source_rule: str = "P-22.2.2.1.7"
    reason: str = ""


def _dihedral_key(sequence: tuple[int, ...]) -> tuple[int, ...]:
    """Canonicalize a cyclic element pattern under rotation and reflection."""
    n = len(sequence)
    reverse = tuple(reversed(sequence))
    return min(
        candidate
        for orientation in (sequence, reverse)
        for start in range(n)
        for candidate in (orientation[start:] + orientation[:start],)
    )


def _distinct_multiset_sequences(counts: Counter[int], size: int):
    """Yield distinct multiset arrangements without materializing n! rows."""
    elements = tuple(sorted(counts))
    remaining = dict(counts)
    sequence: list[int] = []

    def visit():
        if len(sequence) == size:
            yield tuple(sequence)
            return
        for element in elements:
            if remaining[element] == 0:
                continue
            remaining[element] -= 1
            sequence.append(element)
            yield from visit()
            sequence.pop()
            remaining[element] += 1

    yield from visit()


def heteroatom_locant_decision(numbers: Iterable[int]) -> HeteroLocantDecision:
    """Return whether P-22.2.2.1.7 permits omission from intrinsic pattern.

    ``numbers`` is the ordered elemental composition around one simple ring,
    expressed as atomic numbers. The order is not used to rank numbering; it
    describes the cycle whose element-placement classes are being assessed.
    More than one dihedral class means that locants remain necessary. Invalid
    or out-of-scope input yields ``unknown`` and never permits omission.
    """
    try:
        sequence = tuple(numbers)
    except TypeError:
        return HeteroLocantDecision("unknown", None, (), None,
                                    reason="element sequence is not iterable")
    size = len(sequence)
    if not 3 <= size <= 10:
        return HeteroLocantDecision("unknown", size, (), None,
                                    reason="ring size is outside the 3–10-member scope")
    if any(type(element) is not int or element not in _ELEMENTS for element in sequence):
        return HeteroLocantDecision("unknown", size, (), None,
                                    reason="only ordinary C/N/O/S/Se/Te elemental patterns are admitted")
    counts: Counter[int] = Counter(sequence)
    element_counts = tuple(sorted(counts.items()))
    if not any(element != 6 for element in sequence):
        return HeteroLocantDecision("unknown", size, element_counts, None,
                                    reason="the parent must contain a heteroatom")

    seen: set[tuple[int, ...]] = set()
    for arrangement in _distinct_multiset_sequences(counts, size):
        seen.add(_dihedral_key(arrangement))
        if len(seen) > 1:
            return HeteroLocantDecision(
                "required", size, element_counts, 2,
                reason="at least two distinct cyclic element-placement classes exist",
            )
    if len(seen) == 1:
        return HeteroLocantDecision(
            "omit", size, element_counts, 1,
            reason="all placements with this element multiset are dihedrally equivalent",
        )
    return HeteroLocantDecision("unknown", size, element_counts, 0,
                                reason="no cyclic element arrangement was enumerated")
