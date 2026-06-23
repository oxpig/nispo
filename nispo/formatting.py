from __future__ import annotations

import re
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
        if len(ene_locs) == 1:
            return root + "en" if ene_locs[0] == 1 else f"{ene_locs[0]}-{root}en"
        if len(ene_locs) > 1:
            return _compact_polyunsaturated_parent(root, ene_locs, "dien", "ene")
        return loc_unsat(root, ene_locs, "ene")
    if yne_locs and not ene_locs:
        if len(yne_locs) == 1:
            return root + "yn" if yne_locs[0] == 1 else f"{yne_locs[0]}-{root}yn"
        if len(yne_locs) > 1:
            return _compact_polyunsaturated_parent(root, yne_locs, "diyn", "yne")
        return loc_unsat(root, yne_locs, "yne")
    ene = loc_unsat(root, ene_locs, "en", terminal_e=False)
    yne = loc_unsat("", yne_locs, "yne", terminal_e=True)
    return ene + "-" + yne



def _compact_polyunsaturated_parent(root: str, locs: list[int], di_suffix: str, base_suffix: str) -> str:
    mult = MULT.get(len(locs), "")
    suffix = di_suffix if len(locs) == 2 else f"{mult}{base_suffix}"
    return f"{join_locants(locs)}-{root}a{suffix}"


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


_TERMINAL_ELIDED_UNSATURATION_RE = re.compile(
    r"(?<![A-Za-z])"
    r"([A-Za-z0-9,\.\[\]\(\)\^']+"
    r"-(?:[0-9]+(?:\([0-9]+\))?,)*[0-9]+(?:\([0-9]+\))?"
    r"-(?:en|dien|trien|tetraen|pentaen|hexaen))"
    r"(?!e)(?=$|[^A-Za-z])"
)


def restore_terminal_unsaturation_e(name: str) -> str:
    """Retain terminal ``e`` when renderer context needs an unsaturation parent."""

    if not name:
        return ""
    return _TERMINAL_ELIDED_UNSATURATION_RE.sub(lambda match: match.group(1) + "e", name)



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
        "hydroxy",
        "dihydroxy",
        "trihydroxy",
        "tetrahydroxy",
        "pentahydroxy",
        "hexahydroxy",
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


def _multiplied_prefix_unit_name(name: str) -> str:
    match = re.fullmatch(r"\((?:\d+[RSZE](?:,\d+[RSZE])*)\)-(.+)", name)
    if match is None:
        return name
    return match.group(1)


def join_prefix_parent(prefix_text: str, parent: str) -> str:
    parent = preferred_parent_stem(parent)
    if not prefix_text:
        return parent
    prefix_text = _drop_redundant_parent_leading_locant(prefix_text, parent)
    separator = "" if _can_elide_prefix_parent_hyphen(parent) else "-"
    return f"{prefix_text}{separator}{parent}"


_PREFERRED_PARENT_STEMS = (
    ("benzimidazole", "benzimidazol"),
    ("benzothiazole", "benzothiazol"),
    ("benzoxazole", "benzoxazol"),
    ("isoquinoline", "isoquinolin"),
    ("naphthyridine", "naphthyridin"),
    ("piperazine", "piperazin"),
    ("piperidine", "piperidin"),
    ("pyridazine", "pyridazin"),
    ("pyrimidine", "pyrimidin"),
    ("pyrrolidine", "pyrrolidin"),
    ("benzodioxole", "benzodioxol"),
    ("imidazole", "imidazol"),
    ("morpholine", "morpholin"),
    ("pyrazine", "pyrazin"),
    ("pyridine", "pyridin"),
    ("quinoline", "quinolin"),
    ("tetrazole", "tetrazol"),
    ("triazole", "triazol"),
    ("thiazole", "thiazol"),
    ("oxazole", "oxazol"),
    ("1,3-dioxolane", "1,3-dioxolan"),
    ("1,3-dioxane", "1,3-dioxan"),
    ("1,4-dioxane", "1,4-dioxan"),
    ("azepane", "azepan"),
    ("azetidine", "azetidin"),
    ("oxolane", "oxolan"),
    ("oxane", "oxan"),
    ("oxepane", "oxepan"),
    ("thiane", "thian"),
    ("indole", "indol"),
)


def preferred_parent_stem(parent: str) -> str:
    if not parent:
        return parent
    if "lambda" in parent:
        return parent
    stemmed = parent
    for full, stem in _PREFERRED_PARENT_STEMS:
        stemmed = stemmed.replace(full, stem)
    return stemmed


def _drop_redundant_parent_leading_locant(prefix_text: str, parent: str) -> str:
    if parent not in {"benzene", "benzen"}:
        if _can_drop_redundant_leading_one_locant(prefix_text, parent):
            return prefix_text[2:]
        return prefix_text
    if prefix_text.startswith("1-"):
        return prefix_text[2:]
    return prefix_text


def _can_drop_redundant_leading_one_locant(prefix_text: str, parent: str) -> bool:
    if not prefix_text.startswith("1-") or "," in prefix_text:
        return False
    prefix = prefix_text[2:]
    if "-" in prefix or "(" in prefix or ")" in prefix:
        return False
    if prefix not in {"amino", "azido", "bromo", "chloro", "fluoro", "hydroxy", "iodo", "nitro", "sulfanyl"}:
        return False
    lowered = _strip_leading_indicated_hydrogen(parent.lower())
    if any(char in lowered for char in " []',"):
        return False
    return lowered.endswith(("an", "en", "yn"))


_HYPHENLESS_RETAINED_PARENT_STARTS = (
    "benzene",
    "benzen",
    "benzimidazol",
    "benzothiazol",
    "benzoxazol",
    "benzopentacen",
    "benzopentacene",
    "biphenyl",
    "carbazol",
    "chroman",
    "chromen",
    "furan",
    "indazol",
    "indol",
    "isoindol",
    "isooxazol",
    "isoquinolin",
    "isonicotinamide",
    "isonicotinate",
    "isonicotinic acid",
    "isothiazol",
    "morpholin",
    "nicotinamide",
    "nicotinate",
    "nicotinic acid",
    "oxazol",
    "piperazin",
    "piperidin",
    "picolinamide",
    "picolinate",
    "picolinic acid",
    "purin",
    "pyrazin",
    "pyridazin",
    "pyridin",
    "pyrimidin",
    "pyrrol",
    "quinazolin",
    "quinolin",
    "tetrazol",
    "thiazol",
    "tetracen",
    "tetracene",
    "triazol",
    "hexacen",
    "hexacene",
    "heptahelicen",
    "heptahelicene",
    "hexahelicen",
    "hexahelicene",
    "indolocarbazol",
    "indolocarbazole",
    "pentacen",
    "pentacene",
    "pentahelicen",
    "pentahelicene",
    "phenanthroindol",
    "phenanthroindole",
    "spirobifluoren",
    "spirobifluorene",
    "xanthen",
)


def _can_elide_prefix_parent_hyphen(parent: str) -> bool:
    if not parent or "[" in parent or "]" in parent:
        return False
    lowered = parent.lower()
    if lowered.startswith(("s-", "as-")):
        return False
    lowered = _strip_leading_indicated_hydrogen(lowered)
    return lowered.startswith(_HYPHENLESS_RETAINED_PARENT_STARTS) or _is_simple_hyphenless_parent_token(lowered)


def _is_simple_hyphenless_parent_token(parent: str) -> bool:
    if not parent or "-" in parent or " " in parent or "'" in parent:
        return False
    if not parent[0].isalpha():
        return False
    if parent in {"methan", "methane", "ethan", "ethane"}:
        return False
    if parent.endswith(("yl", "oxy")):
        return False
    return all(char.isalpha() or char.isdigit() for char in parent)


def _strip_leading_indicated_hydrogen(name: str) -> str:
    while True:
        idx = 0
        while idx < len(name) and name[idx].isdigit():
            idx += 1
        if idx == 0 or not name.startswith("H-", idx):
            return name
        name = name[idx + 2 :]


def _prefix_needs_parentheses(name: str) -> bool:
    if name in _UNPARENTHESIZED_HYPHENATED_PREFIXES:
        return False
    if name.startswith("(") or ")" in name or "-" in name or " " in name:
        return True
    return _is_composite_substituent_name(name)


_UNPARENTHESIZED_HYPHENATED_PREFIXES = {
    "sec-butyl",
    "t-butyl",
    "tert-butyl",
}


def _is_composite_substituent_name(name: str) -> bool:
    if name.endswith("diazenyl") and name != "diazenyl":
        return True
    if name.endswith("phosphoryl") and _has_leading_multiplier(name):
        return True
    if name.endswith(("acetyl", "benzoyl", "formyl")):
        return True
    return name.endswith("amino") and _has_leading_multiplier(name)


def _multiplied_prefix_needs_parentheses(name: str) -> bool:
    if name != "carbamyl" and ("carbamyl" in name or "carbamoyl" in name):
        return True
    if name.endswith("phosphoryloxy"):
        return True
    if name.endswith("azanidyl"):
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
