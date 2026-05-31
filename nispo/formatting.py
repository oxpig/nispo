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
        return root + "an"
    if ene_locs and not yne_locs:
        return loc_unsat(root, ene_locs, "ene")
    if yne_locs and not ene_locs:
        return loc_unsat(root, yne_locs, "yne")
    ene = loc_unsat(root, ene_locs, "en", terminal_e=False)
    yne = loc_unsat("", yne_locs, "yne", terminal_e=True)
    return ene + "-" + yne



def loc_unsat(root: str, locs: list[int], suffix: str, terminal_e: bool = False) -> str:
    suffix_text = suffix[:-1] if not terminal_e and suffix.endswith("e") else suffix
    if len(locs) == 1:
        if not root:
            return f"{locs[0]}-{suffix_text}"
        return f"{root}-{locs[0]}-{suffix_text}"
    mult = MULT.get(len(locs), "")
    if not root:
        ending = f"{mult}{suffix}"
        if not terminal_e and ending.endswith("e"):
            ending = ending[:-1]
        return f"{join_locants(locs)}-{ending}"
    joiner = "a-" if root and suffix in {"ene", "yne"} else "-"
    ending = f"{mult}{suffix}"
    if not terminal_e and ending.endswith("e"):
        ending = ending[:-1]
    return f"{root}{joiner}{join_locants(locs)}-{ending}"



def substituent_parent_suffix(parent: str) -> str:
    if parent in {"methane", "methan"}:
        return "methyl"
    if parent in {"ethane", "ethan"}:
        return "ethyl"
    if parent in {"propane", "propan"}:
        return "propyl"
    if parent in {"butane", "butan"}:
        return "butyl"
    if parent.endswith("an"):
        return parent[:-2] + "yl"
    if parent.endswith("ane"):
        return parent[:-3] + "yl"
    if parent.endswith(("ene", "yne", "en", "yn")):
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
        "cyclopropyl": "cyclopropoxy",
        "cyclobutyl": "cyclobutoxy",
        "cyclopentyl": "cyclopentoxy",
        "cyclohexyl": "cyclohexoxy",
    }
    if name in simple:
        return simple[name]
    retained = {
        "benzyl": "benzyloxy",
        "carboxymethyl": "carboxymethoxy",
        "carboxylatomethyl": "carboxylatomethoxy",
        "cyclopropylmethyl": "cyclopropylmethoxy",
        "cyanomethyl": "cyanomethoxy",
        "difluoromethyl": "difluoromethoxy",
        "isobutyl": "isobutoxy",
        "isopropyl": "isopropoxy",
        "tert-butyl": "tert-butoxy",
        "trifluoromethyl": "trifluoromethoxy",
        "trideuteriomethyl": "trideuteriomethoxy",
        "2,2-difluoroethyl": "2,2-difluoroethoxy",
        "2-bromoethyl": "2-bromoethoxy",
        "2-chloroethyl": "2-chloroethoxy",
        "2-fluoroethyl": "2-fluoroethoxy",
        "2-iodoethyl": "2-iodoethoxy",
        "2-morpholinoethyl": "2-morpholinoethoxy",
        "2-oxapropyl": "methoxymethoxy",
        "2,2,2-trifluoroethyl": "2,2,2-trifluoroethoxy",
        "3-oxabutyl": "2-methoxyethoxy",
        "1,1-difluoro-trifluoroethyl": "pentafluoroethoxy",
    }
    if name in retained:
        return retained[name]
    for suffix, oxy_suffix in (("methyl", "methoxy"), ("ethyl", "ethoxy")):
        if name.endswith(suffix) and _safe_substituted_alkoxy_stem(name[: -len(suffix)]):
            return name[: -len(suffix)] + oxy_suffix
    return f"({name})oxy"


def _safe_substituted_alkoxy_stem(stem: str) -> bool:
    if not stem or stem.endswith("-") or "(" in stem or ")" in stem:
        return False
    terms = (
        "fluoro",
        "chloro",
        "bromo",
        "iodo",
        "cyano",
        "carboxy",
        "carboxylato",
        "morpholino",
        "piperidino",
        "piperazino",
        "pyrrolidino",
        "phenoxy",
        "methoxy",
        "ethoxy",
        "trifluoromethyl",
        "difluoromethyl",
    )
    parts = [part for part in stem.split("-") if part]
    return bool(parts) and all(part.isdigit() or part.replace(",", "").isdigit() or part in terms for part in parts)



def format_prefixes(prefixes: list[tuple[str, str]]) -> str:
    if not prefixes:
        return ""
    grouped: dict[str, list[str]] = defaultdict(list)
    for loc, name in prefixes:
        grouped[name].append(loc)

    parts = []
    complex_parts = []
    for name in sorted(grouped):
        locs = sorted(grouped[name], key=_locant_sort_key)
        if _prefix_needs_parentheses(name):
            if len(locs) == 1:
                complex_parts.append(f"{locs[0]}-({name})")
            else:
                multiplier = COMPLEX_MULT.get(len(locs), MULT.get(len(locs), ""))
                complex_parts.append(f"{','.join(locs)}-{multiplier}({name})")
        elif len(locs) == 1:
            parts.append(f"{locs[0]}-{name}")
        elif _multiplied_prefix_needs_parentheses(name):
            multiplier = COMPLEX_MULT.get(len(locs), MULT.get(len(locs), ""))
            parts.append(f"{','.join(locs)}-{multiplier}({name})")
        else:
            parts.append(f"{','.join(locs)}-{MULT.get(len(locs), '')}{name}")
    parts.extend(sorted(complex_parts))
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
    if name != "carbamyl" and ("carbamyl" in name or "carbamoyl" in name):
        return True
    if _has_leading_multiplier(name):
        return True
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
