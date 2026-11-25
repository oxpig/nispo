from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache

from .constants import MULT, ROOTS
from .parent_join_presentation import omit_monosubstituted_benzene_locant, starts_with_fusion_component

# Nuclide descriptors have their own enclosing parentheses. They do not by
# themselves make an otherwise simple prefix a compound unit (P-82.2). A
# complete descriptor consists of nuclides, optionally preceded by locants;
# a permissive letters-and-digits test also consumed ordinary substituents
# containing indicated H, such as (1-methyl-1H-1,2,3-triazol-4-yl).
_ISOTOPE_LOCANT = r"(?:[1-9]\d*[a-z]?'*(?:\^\d+)?|(?:Se|Te|As|Sb|Bi|[A-Z])'*(?:\^\d+|\d+)?)"
_ISOTOPE_NUCLIDE = r"\^?[1-9]\d*(?:Si|Cl|Br|Se|Te|H|C|N|O|F|P|S|I|B)\d*"
_ISOTOPE_ENTRY = rf"(?:{_ISOTOPE_LOCANT}(?:,{_ISOTOPE_LOCANT})*-)?{_ISOTOPE_NUCLIDE}"
_ISOTOPE_DESCRIPTOR = re.compile(rf"\({_ISOTOPE_ENTRY}(?:,{_ISOTOPE_ENTRY})*\)")


def _simple_isotope_prefix(name):
    match = _ISOTOPE_DESCRIPTOR.match(name)
    if match is None:
        return False
    base = name[match.end():]
    return bool(re.fullmatch(
        r"(?:[a-z]+yl|[a-z]+ylidene|[a-z]+oxy|amino|hydroxy|oxo|cyano|carboxy|"
        r"fluoro|chloro|bromo|iodo|nitro|sulfanyl)", base
    )) and not _is_composite_substituent_name(base)

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
    # P-31.1.1.2 keeps connecting a if either unsaturation suffix is
    # multiplied, including an en-diyne with only one double bond.
    ene_root = root + 'a' if len(ene_locs) == 1 and len(yne_locs) > 1 else root
    ene = loc_unsat(ene_root, ene_locs, "ene", terminal_e=False)
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
    if parent in {"ethen", "ethene"}:
        return "ethenyl"
    if parent in {"ethyn", "ethyne"}:
        return "ethynyl"
    if parent.endswith(("ene", "yne", "en", "yn")):
        return attach_suffix(parent,'yl',('1',))
    return parent + "yl"



def alkyl_to_alkoxy(name: str) -> str:
    from .component_vocabulary import preferred_component
    name=preferred_component(name)
    if not name:
        return ""
    isotope = _ISOTOPE_DESCRIPTOR.match(name)
    if isotope and re.fullmatch(r"[a-z]+yl", name[isotope.end():]):
        # Retained methoxy/ethoxy contractions survive isotope substitution;
        # the C/H atoms retain the same local attachment numbering.
        return name[:isotope.end()] + alkyl_to_alkoxy(name[isotope.end():])
    constructed = _organyl_to_oxy(name)
    if constructed:
        return constructed
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


def _organyl_to_oxy(
    organyl: str,
    *,
    attachment_locanted: bool = False,
    parent_chain_length: int | None = None,
) -> str:
    """Construct a bounded organyl–oxy prefix from a selected component.

    P-63.2.2.1.1 concatenates systematic organyl + oxy. The short retained
    C1–C4 parent stems contract ``yl`` to ``oxy`` even when substituted;
    cycloalkyl and longer roots retain the full ``yl`` + ``oxy`` sequence.
    """
    if not organyl:
        return ""
    if attachment_locanted or re.fullmatch(r"[a-z]+an-[1-9][0-9]*-yl", organyl):
        return f"({organyl})oxy"
    retained = {
        "methyl": "methoxy",
        "ethyl": "ethoxy",
        "propyl": "propoxy",
        "butyl": "butoxy",
        "tert-butyl": "tert-butoxy",
        "phenyl": "phenoxy",
    }
    if organyl in retained:
        return retained[organyl]
    roots_by_name = {root + "yl": count for count, root in ROOTS.items()}
    if organyl.startswith("cyclo") and organyl.endswith("yl"):
        root_name = organyl[len("cyclo"):-2]
        if root_name in ROOTS.values():
            return organyl + "oxy"
    direct = re.fullmatch(
        r"(?P<locants>[1-9][0-9]*(?:,[1-9][0-9]*)*)-"
        r"(?:(?P<mult>di|tri|tetra|penta|hexa|hepta|octa|nona|deca))?"
        r"methyl(?P<root>[a-z]+)yl",
        organyl,
    )
    root = None
    chain_length = parent_chain_length
    if organyl in roots_by_name:
        root = organyl[:-2]
        chain_length = roots_by_name[organyl]
    elif (parent_chain_length in ROOTS
          and organyl.endswith(ROOTS[parent_chain_length] + "yl")):
        # Prefer the mapped terminal chain before checking text patterns.
        # A prefix such as dimethylsulfamoyl can also start with "dimethyl".
        root = ROOTS[parent_chain_length]
    elif direct:
        root = direct.group("root")
        chain_length = roots_by_name.get(root + "yl")
    if root is None or chain_length is None:
        return ""
    if chain_length <= 4:
        return organyl[:-2] + "oxy"
    return organyl + "oxy"


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



def format_prefixes(prefixes: list[tuple[str, str]], *, legacy_complex_order: bool = False,
                    omit_locants: bool = False, hierarchical_enclosures: bool = False,
                    citation_order: Sequence[str] | None = None,
                    separate_unlocanted: bool = False) -> str:
    """Render prefixes; proved omission can retain each group's P-16 boundary.

    ``separate_unlocanted`` is for independently cited substituents whose
    attachment locants were omitted, rather than other unlocanted grammars.
    """
    if not prefixes:
        if citation_order is None:
            return ""
        if (not isinstance(citation_order, (tuple, list))
                or any(not isinstance(name, str) or not name for name in citation_order)
                or len(citation_order) != len(set(citation_order))
                or citation_order):
            return ""
        return ""
    if citation_order is None and type(prefixes) in (list, tuple) and len(prefixes) == 1:
        row = prefixes[0]
        if type(row) in (list, tuple) and len(row) == 2 and type(row[1]) is str:
            loc, name = row
            locator = "" if omit_locants else f"{loc}-"
            if _prefix_needs_parentheses(name):
                return locator+_enclose_prefix(name, hierarchical_enclosures)
            return f"{locator}{name}"
    grouped: dict[str, list[str]] = defaultdict(list)
    for loc, name in prefixes:
        grouped[name].append(loc)
    ordered_override = None
    if citation_order is not None:
        if (legacy_complex_order or not isinstance(citation_order, (tuple, list))
                or any(not isinstance(name, str) or not name for name in citation_order)
                or len(citation_order) != len(set(citation_order))
                or set(citation_order) != set(grouped)):
            return ""
        ordered_override = tuple(citation_order)
    if isinstance(prefixes, (list, tuple)) and len(prefixes) == 1:
        loc, name = prefixes[0]
        locator = "" if omit_locants else f"{loc}-"
        if _prefix_needs_parentheses(name):
            return locator+_enclose_prefix(name, hierarchical_enclosures)
        return f"{locator}{name}"
    parts = []
    complex_parts = []
    ordered_names = (list(ordered_override) if ordered_override is not None else
                     sorted(grouped) if legacy_complex_order else
                     sorted(grouped, key=prefix_alphabetization_key))
    for name in ordered_names:
        locs = sorted(grouped[name], key=_locant_sort_key)
        locator = "" if omit_locants else f"{','.join(locs)}-"
        if len(locs) > 1 and (_simple_isotope_prefix(name)
                              or _simple_hydride_oxosulfur_prefix(name)):
            parts.append(locator + MULT.get(len(locs), '') + enclose_name_unit(name))
            continue
        if _prefix_needs_parentheses(name):
            output_parts = complex_parts if legacy_complex_order else parts
            if len(locs) == 1:
                output_parts.append(locator+_enclose_prefix(name, hierarchical_enclosures))
            else:
                multiplier = COMPLEX_MULT.get(len(locs), MULT.get(len(locs), ""))
                output_parts.append(locator+multiplier+_enclose_prefix(name, hierarchical_enclosures))
        elif len(locs) == 1:
            if (omit_locants and parts and (separate_unlocanted
                    or name.endswith(('ylidene','ylidyne')))):
                # P-16.5: independently cited groups remain separate when
                # P-14 permits attachment-locant omission. Otherwise e.g.
                # trichloro + methyl becomes one trichloromethyl group.
                parts.append(enclose_name_unit(name))
            elif (omit_locants and len(ordered_names) > 1
                    and name.endswith(("amino", "sulfanyl", "selanyl", "tellanyl"))
                    and name not in {"amino", "sulfanyl", "selanyl", "tellanyl"}):
                # Concatenating compound heteroatom prefixes can describe a
                # single chain (methylaminomethylsulfanyl), not two groups.
                parts.append(f"({name})")
            else:
                parts.append(f"{locator}{name}")
        elif (omit_locants and name.endswith(("amino", "sulfanyl", "selanyl", "tellanyl"))
              and name not in {"amino", "sulfanyl", "selanyl", "tellanyl"}):
            # Without explicit locants, dimethylamino and dimethylsulfanyl
            # describe substitution at N/S rather than two complete groups.
            parts.append(f"{MULT.get(len(locs), '')}({name})")
        elif _multiplied_prefix_needs_parentheses(name):
            multiplier = COMPLEX_MULT.get(len(locs), MULT.get(len(locs), ""))
            parts.append(f"{locator}{multiplier}({name})")
        else:
            infix="-" if name.startswith(("tert-","sec-")) else ""
            unit = (enclose_name_unit(name)
                    if omit_locants and separate_unlocanted and parts else infix + name)
            parts.append(f"{locator}{MULT.get(len(locs), '')}{unit}")
    if legacy_complex_order:
        parts.extend(sorted(complex_parts))
    return ("" if omit_locants else "-").join(parts)


def multiply_component(name: str, count: int, *, nature: str | None = None) -> str:
    """Share the prefix grammar for complete unlocanted ligand components.

    Simple units take di/tri; compound units take bis/tris/tetrakis and their
    enclosures. Alphabetical ordering is performed on each complete component
    before this operation, exactly as in locanted substituent rendering.
    """
    if nature is None:
        # P-16.3.4(d): e.g. tri + decyl must not become one tridecyl arm.
        # Prefer the proved simple grammar where it applies; retain the
        # existing grammar for other complete component constructions.
        from .substituent_multiplication import multiply_substituent
        simple = multiply_substituent(name, count, nature='simple')
        if simple is not None:
            return simple.text
    if nature is not None:
        from .substituent_multiplication import multiply_substituent
        result = multiply_substituent(name, count, nature=nature)
        return result.text if result is not None else ""
    if count < 1 or count > 1 and count not in MULT:
        return ""
    return format_prefixes([("", name)] * count, omit_locants=True)


@lru_cache(maxsize=32768)
def prefix_alphabetization_key(name: str) -> tuple:
    """Order prefix units by P-14.5, before inserting outer multipliers.

    Internal multipliers belong to the complete compound prefix. Locants,
    enclosing marks and stereodescriptors do not lead its alphabetical key;
    italic structural terms only break a tie between the Roman letters.
    """
    text = _ISOTOPE_DESCRIPTOR.sub('', name)
    text = re.sub(r"\((?:\d*[RSZErs](?:\*)?)(?:,\d*[RSZErs](?:\*)?)*\)-?", "", text)
    text = re.sub(r"\((?:\d+-)?\d+[A-Z][a-z]?\d*\)-?", "", text)
    text = re.sub(r"lambda\d+", "", text)
    # Indicated hydrogen in e.g. 1H-1,2,3-triazol is italic (P-16.6.3).
    # It is not part of the Roman-letter comparison in P-14.5.3, even when
    # another numeric ring locant follows the hyphen. Keep the H and its
    # locant available for the later tie-breaking stages.
    italic = tuple(value.lower() for value in re.findall(
        r"(?<=\d)H(?=-)|(?<![a-z])(?:tert|sec|t|as|s)(?=-)", text))
    text = re.sub(r"(?<=\d)H(?=-)", "", text)
    text = re.sub(r"(?<![a-z])[HNOSP](?:[0-9']*)-(?=[([{a-z])", "", text)
    text = re.sub(r"(?<![a-z])(?:tert|sec|t|as|s)-", "", text)
    letters = re.sub(r"[^a-zA-Z]", "", text).lower()
    locants = tuple(int(loc) for loc in re.findall(r"\d+", text))
    return letters, italic, locants, name


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
    ("piperazine", "piperazin"),
    ("piperidine", "piperidin"),
    ("pyridazine", "pyridazin"),
    ("pyrimidine", "pyrimidin"),
    ("pyrrolidine", "pyrrolidin"),
    ("benzodioxole", "benzodioxol"),
    ("imidazole", "imidazol"),
    ("imidazoline", "imidazolin"),
    ("morpholine", "morpholin"),
    ("pyrazine", "pyrazin"),
    ("pyrazoline", "pyrazolin"),
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
    """Keep complete parents intact; suffix renderers perform vowel elision.

    The historical helper name remains for the internal rendering interface.
    Global replacement of pyridine by pyridin also shortened standalone names
    and parents before consonant-starting suffixes, contrary to P-16.7.
    """
    return complete_parent_name(parent)


_COMPLETE_PARENT_NAMES = {
    **{stem: full for full, stem in _PREFERRED_PARENT_STEMS},
    **{root + "an": root + "ane" for root in ROOTS.values()},
    **{"cyclo" + root + "an": "cyclo" + root + "ane" for root in ROOTS.values()},
}
_CHAIN_ROOT_PATTERN = "(?:" + "|".join(sorted(ROOTS.values(), key=len, reverse=True)) + ")"
_UNSATURATION_MULTIPLIER_PATTERN = "(?:" + "|".join(sorted(MULT.values(), key=len, reverse=True)) + ")"
_LEGACY_LOCANTED_CHAIN_RE = re.compile(
    rf"^(?P<locants>\d+(?:,\d+)*)-(?P<root>{_CHAIN_ROOT_PATTERN})"
    rf"(?P<ending>en|yn|a{_UNSATURATION_MULTIPLIER_PATTERN}(?:en|yn)e?)$"
)
_LEGACY_UNLOCANTED_CHAIN_RE = re.compile(rf"^(?P<root>{_CHAIN_ROOT_PATTERN})(?P<ending>en|yn)$")
_MODERN_ELIDED_CHAIN_RE = re.compile(
    rf"^{_CHAIN_ROOT_PATTERN}a?-\d+(?:,\d+)*-"
    rf"(?:en|yn|{_UNSATURATION_MULTIPLIER_PATTERN}(?:en|yn))$"
)


@lru_cache(maxsize=1)
def _complete_retained_hydride_vocabulary():
    """Whole parent words already registered from P-25, without graph lookup.

    Parser resources store suffix-ready stems as well as complete words.
    Recover the lexical terminal e only when the complete hydride is in the
    existing source vocabulary. This does not infer an ending from an
    arbitrary word, a derivative, a fusion prefix or a completed suffix.
    """
    from .preferred_fused import RETAINED_FUSED_HYDRIDES
    from .resource_io import read_json_resource

    data = read_json_resource('resources', 'bluebook_retained_hydrocarbon_parents.json')
    names = set(RETAINED_FUSED_HYDRIDES)
    names.update(row['name'] for row in data['parents'])
    # Indicated H remains the producer's responsibility. The bare lexical
    # word and the explicitly indicated parent have the same terminal vowel.
    names.update(re.sub(r'^(?:\d+[a-z]?H(?:,\d+[a-z]?H)*-)+', '', name)
                 for name in tuple(names))
    return frozenset(names)


@lru_cache(maxsize=16384)
def complete_parent_name(parent: str) -> str:
    """Render a complete parent hydride separately from a suffix-bearing stem."""
    full = _COMPLETE_PARENT_NAMES.get(parent)
    if full is not None:
        return full
    if parent + 'e' in _complete_retained_hydride_vocabulary():
        return parent + 'e'
    if _MODERN_ELIDED_CHAIN_RE.fullmatch(parent):
        return parent + 'e'
    match = _LEGACY_LOCANTED_CHAIN_RE.fullmatch(parent)
    if match is not None:
        root, ending = match.group("root", "ending")
        joiner = "a" if ending.startswith("a") else ""
        suffix = ending.removeprefix("a").removesuffix("e") + "e"
        return f"{root}{joiner}-{match.group('locants')}-{suffix}"
    match = _LEGACY_UNLOCANTED_CHAIN_RE.fullmatch(parent)
    if match is not None:
        root, ending = match.group("root", "ending")
        if root in {"eth", "prop"}:
            return root + ending + "e"
        return f"{root}-1-{ending}e"
    return parent


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
    "acetic acid",
    "adamantan",
    "benzoic acid",
    "benzamide",
    "benzoate",
    "benzonitrile",
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
    # User-provided string-like objects can observe their methods dynamically.
    # Memoize only an ordinary immutable text value; retain their old path.
    if (type(parent) is str
            and _prefix_parent_hyphen_grammar is _SOURCE_HYPHEN_GRAMMAR
            and _strip_leading_indicated_hydrogen is _SOURCE_HYDROGEN_STRIP
            and _is_simple_hyphenless_parent_token is _SOURCE_HYPHENLESS_TOKEN):
        return _cached_prefix_parent_hyphen_grammar(parent)
    return _prefix_parent_hyphen_grammar(parent)


@lru_cache(maxsize=16384)
def _cached_prefix_parent_hyphen_grammar(parent: str) -> bool:
    return _prefix_parent_hyphen_grammar(parent)


def _prefix_parent_hyphen_grammar(parent: str) -> bool:
    if parent == 'flavan-4-one':
        return True
    # The internal von Baeyer descriptor does not separate a detachable
    # prefix from the initial letters of a polycyclic parent hydride.
    if re.match(r"^(?:(?:bi|tri|tetra|penta|hexa|hepta|octa|nona|deca)cyclo|(?:di|tri|tetra|penta)?spiro)\[", parent):
        return True
    if not parent:
        return False
    if "[" in parent or "]" in parent:
        return starts_with_fusion_component(parent)
    lowered = parent.lower()
    if lowered.startswith(("s-", "as-")):
        return False
    lowered = _strip_leading_indicated_hydrogen(lowered)
    # A suffix or unsaturation locant inside a parent does not require a
    # hyphen between a substituent prefix and that parent's initial letters.
    if re.match(rf"^(?:cyclo)?{_CHAIN_ROOT_PATTERN}(?:anoic acid|anoyl|ane?|a)?(?:-|$)", lowered):
        return True
    if re.fullmatch(rf"(?:cyclo)?{_CHAIN_ROOT_PATTERN}ane(?:di|tri|tetra|penta|hexa)oic acid", lowered):
        return True
    if any(lowered.startswith(stem) for _full, stem in _PREFERRED_PARENT_STEMS):
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


_SOURCE_HYPHEN_GRAMMAR = _prefix_parent_hyphen_grammar
_SOURCE_HYDROGEN_STRIP = _strip_leading_indicated_hydrogen
_SOURCE_HYPHENLESS_TOKEN = _is_simple_hyphenless_parent_token


@lru_cache(maxsize=32768)
def _prefix_needs_parentheses(name: str) -> bool:
    if name.endswith(('hydrazonamido', 'carbamoylamino', 'oxyimino')):
        return True
    # P-65.2.2 cites the complete thiocyanato substituent as a unit.
    if name == 'thiocyanato':
        return True
    if _simple_isotope_prefix(name):
        return False
    if _simple_hydride_oxosulfur_prefix(name) or _hydride_sulfonamido_prefix(name):
        return True
    if name in _UNPARENTHESIZED_HYPHENATED_PREFIXES:
        return False
    if _is_systematic_organyloxy_compound(name):
        return True
    if name.startswith("(") or ")" in name or "-" in name or " " in name:
        return True
    return _is_composite_substituent_name(name)


@lru_cache(maxsize=1024)
def _simple_hydride_oxosulfur_prefix(name: str) -> bool:
    """P-16.5.1.4: enclose simple sulfonyl/sulfinyl hydride components.

    Enclosure alone does not make these unsubstituted acid-derived components
    compound: repeated units still use a basic multiplier. Substituted and
    more complex hydride names keep their own source-aware writer.
    """
    suffix = next((ending for ending in ('sulfonyl', 'sulfinyl', 'selenonyl', 'seleninyl',
                                         'telluronyl', 'tellurinyl')
                   if name.endswith(ending)), None)
    if suffix is None:
        return False
    parent = name[:-len(suffix)]
    return _simple_sulfur_hydride_parent(parent)


@lru_cache(maxsize=1024)
def _hydride_sulfonamido_prefix(name: str) -> bool:
    """P-66.1.1.4.3: enclose a complete hydride-derived amido prefix."""
    suffix = next((ending for ending in ('sulfonamido', 'sulfinamido')
                   if name.endswith(ending)), None)
    return suffix is not None and _simple_sulfur_hydride_parent(name[:-len(suffix)])


def _simple_sulfur_hydride_parent(parent: str) -> bool:
    return (parent == 'benzene' or parent in {root + 'ane' for root in ROOTS.values()}
            or parent in {'cyclo' + root + 'ane'
                          for count, root in ROOTS.items() if count >= 3})


_UNPARENTHESIZED_HYPHENATED_PREFIXES = {
    "sec-butyl",
    "t-butyl",
    "tert-butyl",
    # P-63.2.2.2 lists tert-butoxy among the retained simple oxy prefixes.
    "tert-butoxy",
}


@lru_cache(maxsize=32768)
def _is_composite_substituent_name(name: str) -> bool:
    # These unmodified retained acyl prefixes are simple units. Substituted
    # derivatives still take the compound-prefix enclosure path below.
    if name in {"acetyl", "benzoyl", "formyl"}:
        return False
    if name.endswith("hydrazinylidene") and name != "hydrazinylidene":
        # A prefix on hydrazinylidene is part of one compound substituent,
        # even when its complete spelling contains no internal locant.
        return True
    if name.endswith('carbamoyl') and name != 'carbamoyl':
        return True
    # A substituted amide-derived prefix is one complete group. Without its
    # boundary, a multiplier/halogen can be read as modifying the host parent.
    if name.endswith('acetamido') and name != 'acetamido':
        return True
    if _is_systematic_organyloxy_compound(name):
        return True
    # These complete bounded component grammars describe a substituted
    # methyl or amino unit, even when their names contain no locants. P-29.1.2
    # and P-16.5.1.1 require their enclosure at the next citation boundary.
    # The methyl grammar counts terminal halogens so it cannot silently
    # classify a different, impossible methyl graph as a compound unit.
    if _halogenated_methyl_component(name):
        return True
    # P-29.1.2/P-16.5.1.1: these are compound substituent groups,
    # even though their contracted names contain no locants or hyphens.
    # The mapped prefix producers establish the corresponding graph before
    # reaching this citation boundary.  In contrast, the unmodified retained
    # methoxy/ethoxy/phenoxy units themselves are simple prefixes.
    if name.endswith("methoxy") and _halogenated_methyl_component(name[:-3] + "yl"):
        return True
    if name == "benzyloxy":
        return True
    if re.fullmatch(r"(?:methoxy|ethoxy|propoxy|butoxy|phenoxy)(?:methyl|methoxy)", name):
        return True
    if re.fullmatch(rf"cyclo{_CHAIN_ROOT_PATTERN}yl(?:methyl|amino)", name):
        return True
    if re.fullmatch(rf"{_CHAIN_ROOT_PATTERN}ylamino", name):
        return True
    if re.fullmatch(rf"(?:amino|hydroxy|cyano|carboxy|formyl|nitro|sulfanyl|sulfo|sulfino|sulfamoyl){_CHAIN_ROOT_PATTERN}yl",name):
        return True
    # P-65.6.3.2.3: these are complete compound ester prefixes. Keep the
    # acyl/oxy or alkoxy/carbonyl construction enclosed at its citation site.
    if name in {'acetyloxy','formyloxy'} or name.endswith(('oyloxy','carbonyloxy','oxycarbonyl')):
        return True
    # A substituted carbonyl is one complete acyl component. Without this
    # boundary a subsequent presentation pass can move its leading ligand
    # independently and attach the residual carbonyl to another prefix.
    if name.endswith('carbonyl') and name != 'carbonyl':
        return True
    if name.endswith("methylidene") and name != "methylidene":
        return True
    # P-63.3.1: the complete additive peroxide/disulfide unit is enclosed as
    # one substituent, e.g. methylperoxy or methyldisulfanyl.
    if name.endswith("peroxy") and name not in {"peroxy", "hydroperoxy"}:
        return True
    if name.endswith("disulfanyl") and name != "disulfanyl":
        return True
    if name.endswith("diazenyl") and name != "diazenyl":
        return True
    if name.endswith("phosphoryl") and _has_leading_multiplier(name):
        return True
    if name.endswith(("acetyl", "benzoyl", "formyl")):
        return True
    return name.endswith("amino") and _has_leading_multiplier(name)


@lru_cache(maxsize=1024)
def _halogenated_methyl_component(name: str) -> bool:
    """Recognize a complete mono- to trihalogenated methyl component."""
    if not name.endswith("methyl"):
        return False
    head = name[:-len("methyl")]
    matches = list(re.finditer(r"(di|tri)?(bromo|chloro|fluoro|iodo)", head))
    if not matches or "".join(match.group() for match in matches) != head:
        return False
    elements = [match.group(2) for match in matches]
    if elements != sorted(set(elements)):
        return False
    count = sum({None: 1, "di": 2, "tri": 3}[match.group(1)] for match in matches)
    return count <= 3


@lru_cache(maxsize=32768)
def _multiplied_prefix_needs_parentheses(name: str) -> bool:
    if name in {"methoxy", "ethoxy", "propoxy", "butoxy", "phenoxy", "tert-butoxy"}:
        return False
    if _is_systematic_organyloxy_compound(name):
        return True
    # P-63.1.5: repeated –SH groups are bis(sulfanyl), whereas disulfanyl
    # denotes an S–S group. Preserve that distinction for the chalcogen series.
    if name in {'sulfanyl','selanyl','tellanyl'}:
        return True
    # The internal di- belongs to a complete disulfanyl substituent. An
    # additional di- can otherwise be read as substitution on the S-S unit.
    if name.endswith('disulfanyl') and name!='disulfanyl':
        return True
    if name.startswith("hydroxy") and name.endswith("yl"):
        return True
    if name not in {"carbamyl", "carbamoyl"} and ("carbamyl" in name or "carbamoyl" in name):
        return True
    if name.endswith("phosphoryloxy"):
        return True
    if name.endswith(("azanidyl", "oxidanidyl")):
        return True
    if _has_leading_multiplier(name):
        return True
    return name.endswith("aniumyl") or name.endswith("oniumyl")


@lru_cache(maxsize=1)
def _systematic_organyloxy_grammar():
    roots = frozenset(ROOTS.values())
    short_roots = frozenset(root for count, root in ROOTS.items() if count <= 4)
    simple = frozenset(root + 'yloxy' for count, root in ROOTS.items() if count >= 5)
    simple |= frozenset('cyclo' + root + 'yloxy' for count, root in ROOTS.items() if count >= 3)
    branch_multipliers = '|'.join(re.escape(multiplier) for count, multiplier in MULT.items() if count > 1)
    grammar = rf'[1-9][0-9]*(?:,[1-9][0-9]*)*-(?:{branch_multipliers})?methyl'
    return roots, short_roots, simple, grammar


@lru_cache(maxsize=32768)
def _is_systematic_organyloxy_compound(name: str) -> bool:
    """Recognize the bounded P-63 organyloxy forms constructed from alkyls.

    The retained simple terms are handled separately. This grammar is limited
    to unbranched C5+ and cycloalkyl roots, and methyl-substituted primary
    alkyl roots constructed by the mapped branch writer. It does not classify
    every word ending in ``oxy`` as compound.
    """
    # Every admitted spelling below has the same literal terminal group.
    # Decline other native strings before constructing the root vocabulary or
    # trying its alternatives; no complete prefix needs to be reparsed here.
    if type(name) is str and not name.endswith("oxy"):
        return False
    roots, short_roots, simple, grammar = _systematic_organyloxy_grammar()
    if name in simple:
        return True
    # Parse the stem once, then check its vocabulary membership. Compiling a
    # different expression for every root exceeds Python's regex cache with
    # the longer vocabulary, even on an ordinary short-prefix miss.
    full = re.fullmatch(grammar + r"(?P<root>[a-z]+)yloxy", name)
    if full is not None and full['root'] in roots:
        return True
    contracted = re.fullmatch(grammar + r"(?P<root>[a-z]+)oxy", name)
    return contracted is not None and contracted['root'] in short_roots


@lru_cache(maxsize=32768)
def _has_leading_multiplier(name: str) -> bool:
    return any(name.startswith(multiplier) for count, multiplier in MULT.items() if count > 1)



def join_locants(locs: Sequence[int]) -> str:
    return ",".join(str(loc) for loc in sorted(locs))


@lru_cache(maxsize=8192)
def _locant_sort_key(locant: str) -> tuple:
    # Element-site superscripts compare parent position before N-site prime,
    # as in P-66.4.1.4.1's N′^1,N^3,N^3 sequence.
    from .locant_order import writing_locant_key

    source_key = writing_locant_key(locant)
    if source_key is not None:
        return (source_key,)
    # Keep the existing numeric fallback for legacy compound text inside
    # its own rank, without mixing its key shape with typed source locants.
    key = []
    for part in locant.split(","):
        digits = "".join(ch for ch in part if ch.isdigit())
        suffix = part[len(digits):]
        key.append((int(digits) if digits else 0, suffix))
    return ((99,tuple(key)),)


def multiplied_suffix(count: int, suffix: str) -> str:
    """Elide multiplier ``a`` only before suffixes beginning with a or o.

    P-16.7.1(c) retains it before i, e and u, including ``tetraimine``;
    P-31.1.1.2 also retains it before the endings ``ene`` and ``yne``.
    """
    if count == 1:
        return suffix
    multiplier = MULT.get(count)
    if not multiplier:
        return ""
    if suffix and suffix[:1] in "ao":
        multiplier = multiplier.removesuffix("a")
    return multiplier + suffix


def attach_suffix(parent: str, suffix: str, locants=()) -> str:
    """P-16.7 terminal-e elision at an explicit parent/suffix boundary."""
    if not parent or not suffix:return ''
    parent = complete_parent_name(parent)
    stem=parent[:-1] if parent.endswith('e') and suffix[0] in 'aeiouy' else parent
    return stem+('-'+','.join(map(str,locants))+'-' if locants else '')+suffix


def enclose_name_unit(text: str) -> str:
    """P-16.5 enclosing marks around a complete, already rendered unit."""
    if '(' in text and '[' in text:return '{'+text+'}'
    if '(' in text:return '['+text+']'
    return '('+text+')'


def _enclose_prefix(name: str, hierarchical: bool) -> str:
    # A branch may already delimit a complete compound word for embedding.
    # Descriptors, locants and internally nested groups still need a wrapper.
    if name.startswith("(") and name.endswith(")") and name[1:-1].isalpha():
        return name
    return enclose_name_unit(name) if hierarchical else f"({name})"
