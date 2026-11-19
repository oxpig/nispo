from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache

from .constants import MULT, ROOTS
from ..name_spelling import systematic_component_spelling
from ..parent_join_presentation import omit_monosubstituted_benzene_locant, starts_with_fusion_component

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
            return "ethen" if root == "eth" else loc_unsat(root, ene_locs, "en")
        if len(ene_locs) > 1:
            return _compact_polyunsaturated_parent(root, ene_locs, "dien", "ene")
        return loc_unsat(root, ene_locs, "ene")
    if yne_locs and not ene_locs:
        if len(yne_locs) == 1:
            return "ethyn" if root == "eth" else loc_unsat(root, yne_locs, "yn")
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
    if parent in {"ethen", "ethene", "ethyn", "ethyne"}:
        return parent.removesuffix("e") + "yl"
    if parent.endswith(("ene", "yne", "en", "yn")):
        return parent.removesuffix("e") + "-1-yl"
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
        "propan-2-yl": "(propan-2-yl)oxy",
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
    if isinstance(prefixes, (list, tuple)) and len(prefixes) == 1:
        loc, name = prefixes[0]
        if _prefix_needs_parentheses(name):
            return f"{loc}-({name})"
        return f"{loc}-{name}"
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
                multiplier = (MULT.get(len(locs), "") if name == "propan-2-yl"
                              else COMPLEX_MULT.get(len(locs), MULT.get(len(locs), "")))
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
    parent = complete_ring_parent(preferred_parent_stem(parent))
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


@lru_cache(maxsize=16384)
def preferred_parent_stem(parent: str) -> str:
    if not parent:
        return parent
    if "lambda" in parent:
        return parent
    stemmed = parent
    for full, stem in _PREFERRED_PARENT_STEMS:
        stemmed = stemmed.replace(full, stem)
    return stemmed


_COMPLETE_RING_STEMS = frozenset(stem for _name, stem in _PREFERRED_PARENT_STEMS) | {
    "azepin", "azocin", "carbazol", "diazepin", "imidazolidin", "indazol",
    "isothiazol", "isooxazol", "isoxazol", "morpholin", "oxadiazol", "oxazin",
    "oxazepin", "phenoxazin", "phenothiazin", "pteridin", "purin", "pyrazol",
    "pyrrol", "quinazolin", "quinoxalin", "thiadiazol", "thiazin", "thiazol", "thiophen",
} | {f"cyclo{root}an" for size, root in ROOTS.items() if size >= 3}
_RING_WORD_END = re.compile(r"[a-z]+$")
_FUSED_RING_ENDINGS = ("benz", "dibenzo", "benzo", "iso", "phen")
_RING_HYDRO = re.compile(r"^(?:di|tetra|hexa|octa|deca|dodeca)?hydro")


@lru_cache(maxsize=16384)
def complete_ring_parent(parent: str) -> str:
    """Supply the whole ring word at a parent boundary."""
    if not parent or parent.endswith("e"):
        return parent
    match = _RING_WORD_END.search(parent)
    if match is None:
        return parent
    word = _RING_HYDRO.sub("", match.group(), count=1)
    if word in _COMPLETE_RING_STEMS or any(
            word.startswith(prefix) and word[len(prefix):] in _COMPLETE_RING_STEMS
            for prefix in _FUSED_RING_ENDINGS):
        return parent + "e"
    return parent


_RING_CARBONYL_MULTIPLIERS = frozenset(value.removesuffix("a") for value in MULT.values())
_CARBONYL_CONSONANT_ENDINGS = ("thion", "selon", "tellon")
_CYCLIC_ROOT = "(?:" + "|".join(re.escape(root) for n, root in ROOTS.items() if n >= 3) + ")"
_WHOLE_SATURATED_RING_STEM = re.compile(rf"cyclo{_CYCLIC_ROOT}an$|(?:oxindol|thiatriazol|9-oxafluoren)$")
_SPIRO_HYDRIDE_STEM = re.compile(
    rf"(?<=spiro\[)cyclo{_CYCLIC_ROOT}an(?=-\d+'*(?:,|\]))"
)
_WHOLE_UNSATURATED_RING_STEM = re.compile(
    rf"(?:cyclo{_CYCLIC_ROOT}|cyclo\[[^\[\] ]+\]{_CYCLIC_ROOT})a?"
    r"-\d+(?:\(\d+\))?(?:,\d+(?:\(\d+\))?)*-"
    r"(?:(?:di|tri|tetra|penta|hexa)?(?:en|yn))$"
)
_MULTIPLIED_OXO_HYDRIDE = re.compile(
    r"(?P<stem>pyrimidin|piperidin|pyrrolidin|quinolin|triazin|azacyclo[a-z]+an)"
    r"(?=-\d[\d,()a-zH']*-(?:di|tri|tetr|pent)one$)"
)
_COMPOUND_RING_JOIN = re.compile(r"(?<=[)\]}])-(?=(?:bi|tri)?cyclo(?:\[|[a-z]))")
_ALKANE_WORDS = frozenset(root + "ane" for root in ROOTS.values())


def complete_selected_name(name: str) -> str:
    """Finish whole parents and joins after fast parent selection."""
    name = systematic_component_spelling(name)
    if name.endswith(("an", "ol", "en", "yn")):
        if _WHOLE_SATURATED_RING_STEM.search(name) or (
                "cyclo" in name and _WHOLE_UNSATURATED_RING_STEM.search(name)):
            name += "e"
    elif name.endswith(("dione", "trione", "tetrone", "pentone")):
        name = _MULTIPLIED_OXO_HYDRIDE.sub(r"\g<stem>e", name)
    elif name.endswith("ane"):
        prefix, _, parent = name.rpartition("-")
        if prefix.endswith((")", "]", "}")) and parent in _ALKANE_WORDS:
            from ..prefix_presentation import _stereo_before

            if not _stereo_before(prefix, len(prefix)):
                name = prefix + parent
    if "spiro[cyclo" in name and "an-" in name:
        name = _SPIRO_HYDRIDE_STEM.sub(r"\g<0>e", name)
    if ")-cyclo" in name or ")-bicyclo" in name or ")-tricyclo" in name:
        from ..prefix_presentation import _stereo_before

        name = _COMPOUND_RING_JOIN.sub(
            lambda match: match[0] if _stereo_before(name, match.start()) else "", name)
    if any(word in name for word in ("-methyl", "-ethyl", "-propyl", "-butyl", "-pentyl", "-hexyl")):
        from ..prefix_presentation import join_alkyl_prefixes

        name = join_alkyl_prefixes(name)
    return name


def complete_ring_carbonyl_suffix(name: str) -> str:
    """Render a selected ring carbonyl suffix without changing its ranking text."""
    left, separator, suffix = name.rpartition("-")
    if not separator:
        return name
    suffix_stem = suffix.removesuffix("e")
    for stem in (*_CARBONYL_CONSONANT_ENDINGS, "on"):
        if not suffix_stem.endswith(stem):
            continue
        multiplier = suffix_stem[:-len(stem)]
        if multiplier and multiplier not in _RING_CARBONYL_MULTIPLIERS:
            continue
        if multiplier or stem in _CARBONYL_CONSONANT_ENDINGS:
            hydride, sep, locants = left.rpartition("-")
            if sep and locants[:1].isdigit():
                hydride = complete_ring_parent(hydride)
                if hydride.endswith(("an", "en", "yn")):
                    hydride += "e"
                left = f"{hydride}-{locants}"
        return f"{left}-{suffix_stem}e"
    return name


def _drop_redundant_parent_leading_locant(prefix_text: str, parent: str) -> str:
    if parent not in {"benzene", "benzen"}:
        if _can_drop_redundant_leading_one_locant(prefix_text, parent):
            return prefix_text[2:]
        return prefix_text
    return omit_monosubstituted_benzene_locant(prefix_text)


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
    "adamantan",
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


_LOCANTED_MONOUNSATURATED_CHAIN = re.compile(
    r"(?:%s)-\d+-(?:en|yn)" % "|".join(ROOTS.values()))


def _can_elide_prefix_parent_hyphen(parent: str) -> bool:
    if not parent:
        return False
    if "[" in parent or "]" in parent:
        return parent.startswith(("spiro[", "dispiro[")) or starts_with_fusion_component(parent)
    lowered = parent.lower()
    if lowered.startswith(("s-", "as-")):
        return False
    lowered = _strip_leading_indicated_hydrogen(lowered)
    if _LOCANTED_MONOUNSATURATED_CHAIN.match(lowered):
        return True
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


@lru_cache(maxsize=32768)
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


@lru_cache(maxsize=32768)
def _is_composite_substituent_name(name: str) -> bool:
    if name.endswith("diazenyl") and name != "diazenyl":
        return True
    if name.endswith("phosphoryl") and _has_leading_multiplier(name):
        return True
    if name.endswith(("acetyl", "benzoyl", "formyl")):
        return True
    return name.endswith("amino") and _has_leading_multiplier(name)


@lru_cache(maxsize=32768)
def _multiplied_prefix_needs_parentheses(name: str) -> bool:
    if name not in {"carbamyl", "carbamoyl"} and ("carbamyl" in name or "carbamoyl" in name):
        return True
    if name.endswith("phosphoryloxy"):
        return True
    if name.endswith("azanidyl"):
        return True
    if _has_leading_multiplier(name):
        return True
    return name.endswith("aniumyl") or name.endswith("oniumyl")


@lru_cache(maxsize=32768)
def _has_leading_multiplier(name: str) -> bool:
    return any(name.startswith(multiplier) for count, multiplier in MULT.items() if count > 1)



def join_locants(locs: Sequence[int]) -> str:
    return ",".join(str(loc) for loc in sorted(locs))


@lru_cache(maxsize=8192)
def _locant_sort_key(locant: str) -> tuple[tuple[int, str], ...]:
    key = []
    for part in locant.split(","):
        digits = "".join(ch for ch in part if ch.isdigit())
        suffix = part[len(digits):]
        key.append((int(digits) if digits else 0, suffix))
    return tuple(key)
