"""Typed ordering and enclosure for prefixes on mononuclear parent hydrides.

This module formats caller-constructed component names. It does not infer
chemical structure or choose a parent.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .constants import MULT
from .formatting import enclose_name_unit, prefix_alphabetization_key
from .substituent_multiplication import multiply_substituent


RULES = ("P-14.5.1", "P-14.5.2", "P-16.3.2", "P-16.5.1.3.1")


@dataclass(frozen=True, slots=True)
class PrefixUnit:
    """A complete component name and mapped-caller classification."""

    component: str
    nature: str
    count: int = 1


@dataclass(frozen=True, slots=True)
class OrderedPrefixUnit:
    component: str
    nature: str
    count: int
    text: str
    alphabetical_key: tuple
    first: bool


@dataclass(frozen=True, slots=True)
class PrefixSequence:
    text: str
    units: tuple[OrderedPrefixUnit, ...]
    rules: tuple[str, ...] = RULES


_ANNOTATION_OR_NESTING = re.compile(r"[\[\]{}]|\((?:\d|[RSZE])")
_LOCANTED = re.compile(r"(?:^|[^A-Za-z])\d+(?:,\d+)*(?:-[A-Za-z])")
_SAFE_SIMPLE_TOKEN = re.compile(r"[A-Za-z0-9,.'’′^+\-]+")


def _coerce_unit(value) -> PrefixUnit | None:
    if isinstance(value, PrefixUnit):
        unit = value
    elif isinstance(value, dict):
        try:
            unit = PrefixUnit(value["component"], value["nature"], value.get("count", 1))
        except (KeyError, TypeError):
            return None
    elif isinstance(value, (tuple, list)) and len(value) in (2, 3):
        try:
            unit = PrefixUnit(value[0], value[1], value[2] if len(value) == 3 else 1)
        except TypeError:
            return None
    else:
        return None
    if (not isinstance(unit.component, str) or not unit.component
            or unit.nature not in {"simple", "compound"}
            or isinstance(unit.count, bool) or not isinstance(unit.count, int)
            or unit.count < 1 or _ANNOTATION_OR_NESTING.search(unit.component)):
        return None
    return unit


def _is_locanted(component: str) -> bool:
    # Bounded component grammar: the callers currently pass ordinary alkyl
    # locants such as propan-2-yl. More elaborate locant grammars decline.
    return bool(_LOCANTED.search(component))


def _render_unit(unit: PrefixUnit, *, first: bool) -> str | None:
    multiplied = multiply_substituent(unit.component, unit.count, nature=unit.nature)
    # Graph-classified terminal simple groups such as fluoro, hydroxy, and
    # nitro are valid P-16 simple components even though the separate
    # hydrocarbon multiplier grammar does not parse their chemistry. This
    # formatter receives the caller's explicit source/graph type proof.
    characteristic = unit.component in {
        "fluoro", "chloro", "bromo", "iodo", "hydroxy", "nitro",
        "carboxy", "carboxylato", "cyano", "silyl",
    }
    if multiplied is None:
        if unit.nature != "simple" or not characteristic:
            if (unit.count != 1 or unit.nature != "simple"
                    or not _SAFE_SIMPLE_TOKEN.fullmatch(unit.component)):
                return None
            # A single caller-proved simple component can be cited without
            # reclassifying its spelling. Later-unit enclosure follows below.
            text = unit.component
        else:
            text = unit.component if unit.count == 1 else ""
    else:
        text = multiplied.text
    if unit.count > 1 and unit.component in {"tert-butyl", "tert-butoxy"} and not first:
        # P-16.2.4.1(d)'s hyphenated multiplier plus the special later-unit
        # enclosure needs direct combined proof beyond the present examples.
        return None
    if first:
        if characteristic and unit.count > 1:
            multiplier = MULT.get(unit.count)
            return multiplier + unit.component if multiplier else None
        if unit.count == 1 and (unit.nature == "compound" or _is_locanted(unit.component)):
            return enclose_name_unit(text)
        return text

    if unit.nature == "simple" and unit.count > 1:
        if characteristic:
            multiplier = MULT.get(unit.count)
            return multiplier + enclose_name_unit(unit.component) if multiplier else None
        if unit.component in {"tert-butyl", "tert-butoxy"}:
            return None
        # Parentheses enclose the later component only; the simple multiplier
        # remains outside them (P-16.5.1.3.1).
        return multiplied.multiplier + enclose_name_unit(unit.component)
    # A compound group's bis/tris form already encloses the component and is
    # the required later-unit enclosure; do not add a second outer pair.
    if unit.nature == "compound" and unit.count > 1:
        return multiplied.text
    # A later single compound unit still needs its own enclosure.
    return enclose_name_unit(text)


def mononuclear_prefix_sequence(units: Iterable[PrefixUnit | dict | tuple]) -> PrefixSequence | None:
    """Order and render typed substituent units, or decline unsupported input.

    Identical component/nature entries are coalesced before the P-14.5 sort,
    so their outer multiplicative prefix cannot alter alphabetical ordering.
    Annotation-bearing components are declined until an explicit source-aware
    ordering policy is provided.
    """
    try:
        values = [_coerce_unit(value) for value in units]
    except TypeError:
        return None
    if not values or any(value is None for value in values):
        return None

    combined: dict[tuple[str, str], int] = {}
    for unit in values:
        key = (unit.component, unit.nature)
        combined[key] = combined.get(key, 0) + unit.count
    if any(count > 6 for count in combined.values()):
        return None
    ordered = sorted(
        (PrefixUnit(name, nature, count) for (name, nature), count in combined.items()),
        key=lambda unit: prefix_alphabetization_key(unit.component),
    )
    # Equal Roman alphabetic names with materially different typed identities
    # need the complete P-14.5.4 comparison, which this bounded API does not
    # claim to implement.
    keys = [prefix_alphabetization_key(unit.component) for unit in ordered]
    for index in range(1, len(keys)):
        if keys[index - 1] == keys[index]:
            return None

    rendered: list[OrderedPrefixUnit] = []
    for index, unit in enumerate(ordered):
        text = _render_unit(unit, first=index == 0)
        if text is None:
            return None
        rendered.append(OrderedPrefixUnit(
            unit.component, unit.nature, unit.count, text,
            prefix_alphabetization_key(unit.component), index == 0,
        ))
    return PrefixSequence("".join(unit.text for unit in rendered), tuple(rendered))


build_prefix_sequence = mononuclear_prefix_sequence
