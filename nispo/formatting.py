from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .constants import MULT

COMPLEX_MULT = {
    2: "bis",
    3: "tris",
    4: "tetrakis",
    5: "pentakis",
    6: "hexakis",
}


def unsaturated_parent(root: str, ene_locs: list[int], yne_locs: list[int]) -> str:
    if not ene_locs and not yne_locs:
        return root + "ane"
    if ene_locs and not yne_locs:
        return loc_unsat(root, ene_locs, "ene")
    if yne_locs and not ene_locs:
        return loc_unsat(root, yne_locs, "yne")
    ene = loc_unsat(root, ene_locs, "en", terminal_e=False)
    yne = loc_unsat("", yne_locs, "yne", terminal_e=True)
    return ene + "-" + yne



def loc_unsat(root: str, locs: list[int], suffix: str, terminal_e: bool = True) -> str:
    if len(locs) == 1:
        if not root:
            return f"{locs[0]}-{suffix}"
        return f"{root}-{locs[0]}-{suffix}"
    mult = MULT.get(len(locs), "")
    joiner = "a-" if root and suffix in {"ene", "yne"} else "-"
    ending = f"{mult}{suffix}"
    if not terminal_e and ending.endswith("e"):
        ending = ending[:-1]
    return f"{root}{joiner}{join_locants(locs)}-{ending}"



def substituent_parent_suffix(parent: str) -> str:
    if parent == "methane":
        return "methyl"
    if parent == "ethane":
        return "ethyl"
    if parent == "propane":
        return "propyl"
    if parent == "butane":
        return "butyl"
    if parent.endswith("ane"):
        return parent[:-3] + "yl"
    if parent.endswith("ene") or parent.endswith("yne"):
        return parent + "-1-yl"
    return parent + "yl"



def alkyl_to_alkoxy(name: str) -> str:
    if not name:
        return ""
    simple = {
        "methyl": "methoxy",
        "ethyl": "ethoxy",
        "propyl": "propoxy",
        "butyl": "butoxy",
        "pentyl": "pentoxy",
        "hexyl": "hexoxy",
        "heptyl": "heptoxy",
        "octyl": "octoxy",
        "nonyl": "nonoxy",
        "decyl": "decoxy",
    }
    if name in simple:
        return simple[name]
    return f"({name})oxy"



def format_prefixes(prefixes: list[tuple[str, str]]) -> str:
    if not prefixes:
        return ""
    grouped: dict[str, list[str]] = defaultdict(list)
    complex_items: list[str] = []
    for loc, name in prefixes:
        if _prefix_needs_parentheses(name):
            complex_items.append(f"{loc}-({name})")
        else:
            grouped[name].append(loc)

    parts = []
    for name in sorted(grouped):
        locs = sorted(grouped[name], key=_locant_sort_key)
        if len(locs) == 1:
            parts.append(f"{locs[0]}-{name}")
        elif _multiplied_prefix_needs_parentheses(name):
            multiplier = COMPLEX_MULT.get(len(locs), MULT.get(len(locs), ""))
            parts.append(f"{','.join(locs)}-{multiplier}({name})")
        else:
            parts.append(f"{','.join(locs)}-{MULT.get(len(locs), '')}{name}")
    parts.extend(sorted(complex_items))
    return "-".join(parts)


def _prefix_needs_parentheses(name: str) -> bool:
    if name.startswith("(") or ")" in name or "-" in name:
        return True
    return _is_composite_substituent_name(name)


def _is_composite_substituent_name(name: str) -> bool:
    if name.endswith("diazenyl") and name != "diazenyl":
        return True
    if name.endswith("phosphoryl") and _has_leading_multiplier(name):
        return True
    return name.endswith("amino") and _has_leading_multiplier(name)


def _multiplied_prefix_needs_parentheses(name: str) -> bool:
    return name.endswith("aniumyl") or name.endswith("oniumyl")


def _has_leading_multiplier(name: str) -> bool:
    return any(name.startswith(multiplier) for count, multiplier in MULT.items() if count > 1)



def join_locants(locs: Sequence[int]) -> str:
    return ",".join(str(loc) for loc in sorted(locs))


def _locant_sort_key(locant: str) -> list[tuple[int, str]]:
    key = []
    for part in locant.split(","):
        digits = "".join(ch for ch in part if ch.isdigit())
        suffix = part[len(digits):]
        key.append((int(digits) if digits else 0, suffix))
    return key
