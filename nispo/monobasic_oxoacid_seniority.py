"""Exact P-42.5 keys for other monobasic oxoacids used as functional parents.

The table is deliberately finite: it does not extrapolate oxidation states
from arbitrary molecular graphs or bond counts.
"""
from __future__ import annotations


# P-42.5 gives central-element seniority N > F > Cl > Br > I. Within one
# element, its displayed acid series is ordered by descending oxidation state.
# Values are (source element order, explicit oxidation number).
_SOURCE_ROWS = (
    (7, 2, 0, 5), (7, 1, 0, 3),
    (9, 3, 1, 7), (9, 2, 1, 5), (9, 1, 1, 3), (9, 0, 1, 1),
    (17, 3, 2, 7), (17, 2, 2, 5), (17, 1, 2, 3), (17, 0, 2, 1),
    (35, 3, 3, 7), (35, 2, 3, 5), (35, 1, 3, 3), (35, 0, 3, 1),
    (53, 3, 4, 7), (53, 2, 4, 5), (53, 1, 4, 3), (53, 0, 4, 1),
)
_KEYS_IN_SOURCE_ORDER = tuple((element_order, -oxidation) for _, _, element_order, oxidation in _SOURCE_ROWS)
_KEY_TO_FLAT = {key: 3 + index for index, key in enumerate(_KEYS_IN_SOURCE_ORDER)}
_ROW_LOOKUP = {(z, terminal_oxo): (element_order, oxidation)
               for z, terminal_oxo, element_order, oxidation in _SOURCE_ROWS}


def monobasic_oxoacid_key(
    central_atomic_number: int, terminal_oxo_count: int
) -> tuple[int, int] | None:
    """Return a P-42.5 sorting key, or ``None`` outside its literal table.

    Lower keys are senior. The first field is the source's central-element
    order; the second is negative oxidation number so higher oxidation ranks
    first. ``terminal_oxo_count`` counts the displayed central ``(O)`` oxos,
    excluding the single acidic ``OH`` group.
    """
    if (type(central_atomic_number) is not int
            or type(terminal_oxo_count) is not int):
        return None
    row = _ROW_LOOKUP.get((central_atomic_number, terminal_oxo_count))
    if row is None:
        return None
    element_order, oxidation = row
    return (element_order, -oxidation)


def class11_monobasic_oxoamide_subrank(
    central_atomic_number: int, terminal_oxo_count: int
) -> int | None:
    """Flatten a P-42.5 acid key after established class-11 ranks 0, 1, 2.

    P-41 orders amides by their corresponding acids. This reserves 0/1/2 for
    the existing amide/sulfonamide/urea entries and assigns the 18 exact P-42.5
    profiles the consecutive ranks 3 through 20, in source seniority order.
    A caller can preserve the existing two-field shape as ``(11, subrank)``.
    """
    key = monobasic_oxoacid_key(central_atomic_number, terminal_oxo_count)
    return None if key is None else _KEY_TO_FLAT[key]
