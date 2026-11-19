"""Conventional spellings at boundaries in an already selected name."""

import re


_TRIAZINE_LOCANTS = {"s": "1,3,5", "as": "1,2,4", "v": "1,2,3"}
_TRIAZINE = re.compile(r"(?<![a-z])(?P<alias>as|s|v)-(?=triazin(?:e\b|[-)\]} ]|$))")
_FURYL_THIENYL = re.compile(
    r"(?<![\w^'])\b([23])-(furyl|thienyl)(?=$|[^a-z]|carbonyl|methyl|oxy|sulfanyl)"
)
_CBZ = re.compile(
    r"(?<![a-z])cbz(?=$|[^a-z]|amino|amido|piperidin|piperazin|pyrrolidin|azetidin)"
)
_FMOC = re.compile(r"(?<![a-z])fmoc(?=$|[^a-z]|amino|amido)")
_ALKYL_CARBONYL = re.compile(r"(?<![a-z])((?:hept|oct|non|dec)an)e(?=-\d+-ylcarbonyl)")
_ISOINDOLINYL_DIOXO = re.compile(r"(?<![\w,'])3-oxo-1-oxo(?=isoindolin-2-yl)")
_REPEATED_BONDING_NUMBER = re.compile(r"(?<![\w,^'])(\d+[a-z]?'*lambda\d+)-\1(?=-)")
_HYDRO_FUSION_JOIN = re.compile(r"(?<=hydro)-(?=[a-z]+\[\d)")
_SIMPLE_MULTIPLICATION = re.compile(
    r"\b(bis|tris|tetrakis)\((methoxy|ethoxy|methyl|ethyl|chloro|fluoro|bromo|hydroxy|amino)\)"
)
_SIMPLE_MULTIPLIERS = {"bis": "di", "tris": "tri", "tetrakis": "tetra"}
_WHOLE_RING_STEM = re.compile(
    r"(?:dioxolan|dioxan|isoindolin|pyrrolizidin|azaquinolin|oxacycloheptan|phosphacyclohexan)$"
)
_REPLACEMENT_SPIRO_JOIN = re.compile(r"(?<=aza)-(?=spiro\[)")
# The 1,2-b fusion and the retained indole have the same numbering. Leave
# fusion into a larger parent to the corresponding numbered component writer.
_INDOLE_ALIAS = re.compile(r"(?<![a-z\]])(?<!\dH-)(?:1H-)?benzo\[1,2-b\]pyrrol(e?)\b(?!\[)")


def _systematic_triazine(match):
    locants = _TRIAZINE_LOCANTS[match["alias"]]
    preceding = match.string[match.start() - 1:match.start()]
    if preceding == "]":
        before = match.string[:match.start()]
        if re.search(r"[a-z]+\[[0-9,]+-[a-z]\]$", before):
            return f"[{locants}]"
    return ("-" if preceding in ")]}" and preceding else "") + locants + "-"


def retained_indole_spelling(name: str) -> str:
    if "benzo[1,2-b]pyrrol" not in name:
        return name

    def substitute(match):
        # P-16.2.4.1: the newly cited indicated-H locant follows a hyphen
        # when the preceding substituent ends with an enclosing mark.
        preceding = name[match.start() - 1:match.start()]
        join = "-" if preceding and preceding in ")]}" else ""
        return f"{join}1H-indol{match[1]}"

    return _INDOLE_ALIAS.sub(substitute, name)


def systematic_component_spelling(name: str) -> str:
    if "triazin" in name:
        name = _TRIAZINE.sub(_systematic_triazine, name)
    if "furyl" in name or "thienyl" in name:
        name = _FURYL_THIENYL.sub(
            lambda match: f'{"furan" if match[2] == "furyl" else "thiophen"}-{match[1]}-yl', name)
    if "chinolin" in name:
        name = name.replace("hydro-chinolin]", "hydroquinoline]").replace("chinolin", "quinolin")
    if "cbz" in name:
        name = _CBZ.sub("(benzyloxycarbonyl)", name)
    if "fmoc" in name:
        name = _FMOC.sub("[(9H-fluoren-9-ylmethoxy)carbonyl]", name)
    if "ylcarbonyl" in name:
        name = _ALKYL_CARBONYL.sub(r"\1", name)
    if "oxoisoindolin-2-yl" in name:
        name = _ISOINDOLINYL_DIOXO.sub("1,3-dioxo", name)
    if name.count("lambda") > 1:
        name = _REPEATED_BONDING_NUMBER.sub(r"\1", name)
    if "hydro-" in name:
        name = _HYDRO_FUSION_JOIN.sub("", name)
    if "bis(" in name or "tris(" in name or "tetrakis(" in name:
        name = _SIMPLE_MULTIPLICATION.sub(lambda m: _SIMPLE_MULTIPLIERS[m[1]] + m[2], name)
    if name.endswith(("an", "in")) and _WHOLE_RING_STEM.search(name):
        name += "e"
    if "aza-spiro[" in name:
        name = _REPLACEMENT_SPIRO_JOIN.sub("", name)
    return retained_indole_spelling(name)
