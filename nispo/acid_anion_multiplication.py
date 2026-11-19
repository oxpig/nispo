"""Typed multiplicative construction for repeated acid-anion components.

P-65.6.3.3.3.1 selects di/tri for unsubstituted anions and bis/tris for
chemically substituted anions. P-16.3.4 independently determines when the
multiplied unit needs enclosing marks. This helper deliberately does not infer
chemical substitution from rendered component text or isotope/stereo markers.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .constants import MULT
from .formatting import enclose_name_unit

# Parent-hydride roots whose initial text can be mistaken for an already
# multiplied component. These derive from the Greek multiplicative prefix
# sequence and are checked against the typed parent field, never the decorated
# acid-anion citation. The `dec` family has its separate P-16.3.4(d) rule.
_MULTIPLICATIVE_STARTS = (
    "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "nona",
    "deca", "undeca", "dodeca", "trideca", "tetradeca", "pentadeca",
    "hexadeca", "heptadeca", "octadeca", "nonadeca", "icosa",
)


@dataclass(frozen=True, slots=True)
class AnionMultiplication:
    """A source-rule result for a repeated identical acid-anion unit."""

    text: str
    multiplier: str
    chemical_class: str
    enclosure_reason: tuple[str, ...]
    rules: tuple[str, ...] = (
        'P-65.6.3.3.3.1', 'P-16.3.2', 'P-16.3.4',
        'P-16.3.5(a)', 'P-16.5.4.1.3',
    )


def _parent_component_facts(
    construction: Any, *, enclosure_proved: bool
) -> tuple[bool, tuple[str, ...]] | None:
    """Return locant/start enclosures from a typed mapped acid construction.

    `None` denotes an unresolved parent-start ambiguity without another proven
    reason to enclose. A missing construction is not silently treated as an
    ordinary simple component.
    """
    parent = getattr(construction, "parent", None)
    prefixes = getattr(construction, "prefixes", None)
    if not isinstance(parent, str) or not parent or prefixes is None:
        return None

    # P-16.3.4(e): functionalized parent hydrides with locants. A locant in a
    # typed systematic parent is sufficient; this does not inspect the final
    # anion citation, where isotope/stereo decorations may also contain digits.
    has_parent_locant = bool(re.search(r"(?:^|[^A-Za-z])\d+(?:,\d+)*-", parent))
    compact = parent.lower().replace("-", "").replace(" ", "")
    enclosure_reasons: list[str] = []
    if has_parent_locant:
        enclosure_reasons.append("functionalized-parent-with-locants")

    # P-16.3.4(c): a component beginning with a multiplicative prefix is
    # enclosed. The P-16.3.4(d) dec- family is handled explicitly because the
    # rule resolves the possible 12–19 atom acyclic reading.
    start = re.search(r"[a-z]+", compact)
    if not start:
        return (bool(has_parent_locant), tuple(enclosure_reasons)) if enclosure_proved else None
    root_start = start.group(0)
    if root_start.startswith("dec"):
        enclosure_reasons.append("dec-root")
    elif any(root_start.startswith(prefix) for prefix in _MULTIPLICATIVE_STARTS):
        enclosure_reasons.append("multiplicative-prefix-start")
    elif not re.match(r"(?:meth|eth|prop|but|pent|hex|hept|oct|non|icos|cyclo|benz|form|acet|glycol|lactic)", root_start):
        # A bounded vocabulary avoids guessing when the parent token does not
        # establish whether leading text is an ordinary root or multiplicative
        # morphology. New source-proved vocabulary can be added explicitly.
        if enclosure_proved:
            return bool(has_parent_locant), tuple(enclosure_reasons)
        return None
    return bool(has_parent_locant), tuple(enclosure_reasons)


def multiply_acid_anion(
    citation: str,
    count: int,
    *,
    construction: Any,
    isotope_modified: bool = False,
    stereochemically_modified: bool = False,
) -> AnionMultiplication | None:
    """Render a repeated identical acid anion, or decline unresolved inputs.

    ``construction`` is the typed mapped acid component (normally
    ``AcidComponent.construction``). Its ``prefixes`` describe chemical
    substitution; isotope and stereochemistry are separate explicit facts.
    The function accepts multiplicities present in ``constants.MULT`` and a
    complete typed parent construction.
    """
    if not isinstance(citation, str) or not citation:
        return None
    if (not isinstance(count, int) or isinstance(count, bool)
            or count not in MULT or count < 2):
        return None
    if not isinstance(isotope_modified, bool) or not isinstance(stereochemically_modified, bool):
        return None

    prefixes = getattr(construction, "prefixes", None)
    if prefixes is None:
        return None
    try:
        chemically_substituted = len(prefixes) > 0
    except TypeError:
        return None
    enclosure_proved = (chemically_substituted or isotope_modified
                        or stereochemically_modified
                        or bool(re.search(r"(?:^|[^A-Za-z])\d+(?:,\d+)*-",
                                          getattr(construction, "parent", ""))))
    parent_facts = _parent_component_facts(
        construction, enclosure_proved=enclosure_proved)
    if parent_facts is None:
        return None
    _parent_has_locants, reasons = parent_facts
    simple_or_compound = "compound" if chemically_substituted else "simple"
    simple_multiplier = MULT[count]
    compound_multiplier = simple_multiplier + "kis"
    if count == 2:
        compound_multiplier = "bis"
    elif count == 3:
        compound_multiplier = "tris"
    multiplier = compound_multiplier if chemically_substituted else simple_multiplier

    enclosure_reasons = list(reasons)
    if chemically_substituted:
        enclosure_reasons.append("chemically-substituted-component")
    if isotope_modified:
        enclosure_reasons.append("specific-isotope-descriptor")
    if stereochemically_modified:
        enclosure_reasons.append("stereochemical-descriptor")
    component = enclose_name_unit(citation) if enclosure_reasons else citation
    return AnionMultiplication(
        text=f"{multiplier}{component}",
        multiplier=multiplier,
        chemical_class=simple_or_compound,
        enclosure_reason=tuple(dict.fromkeys(enclosure_reasons)),
    )
