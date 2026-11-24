"""Preferred presentation of already selected substituent names.

P-16.5 enclosing marks and P-29.3.5 assembly prefixes are applied after
selection, so changes in presentation cannot perturb compact-name ranking.
The grammar recognizers deliberately retain unsupported prefix spellings.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .constants import MULT, REPLACEMENT, ROOTS
from .formatting import COMPLEX_MULT, prefix_alphabetization_key
from .name_spelling import retained_indole_spelling, systematic_component_spelling

_ALKYL = "|".join(re.escape(root + "yl") for root in ROOTS.values())
_ALKYL_JOIN = re.compile(rf"(?<=[a-z)\]}}])-(?P<parent>{_ALKYL})(?![a-zA-Z0-9])")
_HYDRO_JOIN = re.compile(
    r"(?<=[a-z)\]}])-(?=(?:di|tetra|hexa|octa|deca|dodeca)hydro"
    r"(?:naphthalen|anthracen|phenanthren|quinolin|isoquinolin|indol|chromen))"
)
_HYDROCARBYL = rf"(?:{_ALKYL}|phenyl|benzyl|cyclo(?:{_ALKYL}))"
_COMPOUND = re.compile(
    rf"(?<![A-Za-z0-9])(?P<loc>(?:[0-9]+[a-z']*|[NOSP])(?:,(?:[0-9]+[a-z']*|[NOSP]))*-)?(?P<mult>di|tri|tetra|penta|hexa)?"
    rf"(?P<unit>{_HYDROCARBYL}(?:sulfonyl|sulfinyl|sulfanyl|carbonyl)|mesyl)(?=[a-zA-Z]|-|$)"
)
_ACYL_PREFIXES = {"mesyl": "methanesulfonyl"}
for _suffix in ("sulfonyl", "sulfinyl"):
    _ACYL_PREFIXES["phenyl" + _suffix] = "benzene" + _suffix
    _ACYL_PREFIXES["benzyl" + _suffix] = "phenylmethane" + _suffix
    for _count, _root in ROOTS.items():
        _ACYL_PREFIXES[_root + "yl" + _suffix] = (
            _root + "ane" + ("" if _count <= 2 else "-1-") + _suffix
        )
        _ACYL_PREFIXES["cyclo" + _root + "yl" + _suffix] = (
            "cyclo" + _root + "ane" + _suffix
        )
for _root in ROOTS.values():
    _ACYL_PREFIXES["cyclo" + _root + "ylcarbonyl"] = "cyclo" + _root + "anecarbonyl"
_SIMPLE_ACYL_PREFIXES = frozenset(
    value for key, value in _ACYL_PREFIXES.items() if not key.startswith("benzyl")
)


_BIPHENYL = re.compile(
    r"(?P<attachment>[2-6])-(?:phenyl|\((?P<prefix>(?:[2-6](?:,[2-6])*-"
    r"(?:(?:di|tri|tetra|penta)?(?:fluoro|chloro|bromo|iodo|methyl|hydroxy))-?)*)phenyl\))phenyl"
)
_PLAIN_BIPHENYL_LIGAND = re.compile(r"\(([2-6]-phenylphenyl)\)(?=amino\b)")
_PHENYLALKYL = re.compile(r"(?<![\w,'])(?:2-\(phenyl\)ethyl|3-\(phenyl\)propyl)\b")
_FIXED_HYDROGEN = re.compile(r"[0-9]+[a-z]?(?:,[0-9]+[a-z]?)*H")
_FUSION_SIDE = r"(?:[a-z]{1,4}|[a-z][1-9][0-9]*)"
_FUSION_SIDES = re.compile(rf"{_FUSION_SIDE}(?:,{_FUSION_SIDE})*")
_STEREODESCRIPTOR = re.compile(
    r"(?:(?:[0-9]+[a-z']*)?[RSZErs](?:\*)?)"
    r"(?:,(?:[0-9]+[a-z']*)?[RSZErs](?:\*)?)*|cis|trans"
)
_MARKS = (("(", ")"), ("[", "]"), ("{", "}"))
_ASSEMBLY_JUNCTION_LOCANT = r"[1-4]\^\([1-6]\)"
_ASSEMBLY_JUNCTION_PAIR = rf"{_ASSEMBLY_JUNCTION_LOCANT},{_ASSEMBLY_JUNCTION_LOCANT}"
_PRIMABLE_ASSEMBLY = re.compile(
    rf"(?P<open>\[)?(?P<junctions>{_ASSEMBLY_JUNCTION_PAIR}"
    rf"(?::{_ASSEMBLY_JUNCTION_PAIR}){{1,2}})-"
    r"(?P<count>ter|quater)(?:phenyl|pyridine?|thiophene?)(?![a-z])"
    r"(?(open)\])"
)
_ASSEMBLY_LOCANT = r"(?<![\w^'])(([1-4])\^(?:\(([1-6])\)|([1-6])))(?![\w^'])"
_ASSEMBLY_SCOPE_TOKEN = re.compile(rf"{_ASSEMBLY_LOCANT}|[()\[\]{{}}]")
_ASSEMBLY_SUFFIX_LOCANTS = re.compile(rf"-(?:{_ASSEMBLY_LOCANT},)*{_ASSEMBLY_LOCANT}-")


def primed_ring_assembly_presentation(name: str) -> str:
    """Use P-28.3.1 general locants on a recognized linear ring assembly."""
    if "^(" not in name:
        return name
    assemblies = tuple(_PRIMABLE_ASSEMBLY.finditer(name))
    if not assemblies:
        return name

    # Replace the ring occurrences in the junctions, the direct prefixes,
    # and the suffix of this assembly. A nested substituent owns its locants.
    locants = []
    owners = {}
    stack = []
    closing = dict((right, left) for left, right in _MARKS)
    starts = {assembly.start(): assembly for assembly in assemblies}
    for token in _ASSEMBLY_SCOPE_TOKEN.finditer(name):
        start = token.start()
        if start in starts:
            owners[start] = tuple(stack)
        word = token[0]
        if token[1]:
            locants.append((token, tuple(stack)))
        elif word in closing:
            if not stack or name[stack[-1]] != closing[word]:
                return name
            stack.pop()
        else:
            stack.append(start)
    if stack:
        return name

    edits = {}
    for assembly in assemblies:
        owner = owners[assembly.start()]
        count = 3 if assembly["count"] == "ter" else 4
        start, end = assembly.span("junctions")
        junctions = [token for token, _ in locants if start <= token.start() < end]
        occurrences = [int(token[2]) for token in junctions]
        if occurrences != [i for pair in zip(range(1, count), range(2, count + 1)) for i in pair]:
            continue
        prefix_start = max(owner[-1] + 1 if owner else 0,
                           name.rfind(" ", 0, assembly.start()) + 1)
        if any(other is not assembly and owners[other.start()] == owner
               and prefix_start <= other.start() < assembly.start() for other in assemblies):
            continue
        prefix = [token for token, scope in locants
                  if scope == owner and prefix_start <= token.start() < assembly.start()]
        suffix = _ASSEMBLY_SUFFIX_LOCANTS.match(name, assembly.end())
        owned = prefix + junctions
        if suffix:
            owned += [token for token, scope in locants if scope == owner
                      and suffix.start() <= token.start() < suffix.end()]
        if all(int(token[2]) <= count for token in owned):
            edits.update((token.span(), (token[3] or token[4]) + "'" * (int(token[2]) - 1))
                         for token in owned)
    for (start, end), replacement in sorted(edits.items(), reverse=True):
        name = name[:start] + replacement + name[end:]
    return name


@dataclass
class _Group:
    children: list
    opening: str
    fixed: bool = False
    ignored: bool = False


def is_balanced_name_unit(text: str) -> bool:
    """Check enclosure syntax in a complete text supplied by a graph producer."""
    return bool(text) and _parse_groups(text) is not None


def enclose_complete_name_unit(text: str) -> str:
    """Supply one enclosure at the boundary of a complete rendered ligand."""
    if text and text[0] in "([{" and _balanced_end(text, 0) == len(text):
        return text
    return normalize_enclosing_marks(f"({text})")


def normalize_enclosing_marks(name: str) -> str:
    """Render P-16.5 nesting without changing prefix spelling or citation."""
    if name.count("(") < 2 and "[" not in name and "{" not in name:
        return name
    tree = _parse_groups(name)
    return _render(tree)[0] if tree is not None else name


def numbered_biphenyl_ligand(name: str) -> str:
    """Render the complete unsubstituted phenylphenyl ligand of an amino unit."""
    if "phenylphenyl)amino" not in name:
        return name
    return _PLAIN_BIPHENYL_LIGAND.sub(lambda match: "(" + _biphenyl_prefix(match[1]) + ")", name)


def simple_phenylalkyl_prefixes(name: str) -> str:
    """Join the simple phenyl prefix on a locanted ethyl or propyl branch."""
    if "(phenyl)" not in name:
        return name
    return _PHENYLALKYL.sub(lambda match: match[0].replace("(phenyl)", "phenyl"), name)


def preferred_prefix_presentation(name: str) -> str:
    """Render prefix joins, recognized assemblies and enclosing marks.

    This consumes only the generated name, rather than selecting a new
    molecular parent or making a whole-molecule synonym substitution.
    """
    if not name:
        return name
    name = simple_phenylalkyl_prefixes(systematic_component_spelling(name))
    original = name
    name = join_alkyl_prefixes(name)
    name = _HYDRO_JOIN.sub(_join_hydro, name)
    if not any(c in name for c in "([{") and _COMPOUND.search(name) is None:
        return name
    tree = _parse_groups(name)
    if tree is None:
        return original
    tree, changed = _preferred_units(tree)
    rendered = retained_indole_spelling(_render(tree)[0])
    if changed or rendered != name:
        return normalize_enclosing_marks(rendered)
    return rendered


def join_alkyl_prefixes(name: str) -> str:
    return _ALKYL_JOIN.sub(_join_alkyl, name)


def _join_alkyl(match):
    if _stereo_before(match.string, match.start()):
        return match[0]
    if re.search(r"(?<![A-Za-z])(?:Se|Te)$", match.string[:match.start()]):
        # These two-letter element locants are not a detachable prefix to
        # join with the following organyl word.
        return match[0]
    if re.search(r"(?:N'*\^\(\d+'*(?:\^\d+)?\)|\d+\^(?:\{\d+\}|\(\d+\)))$",
                 match.string[:match.start()]):
        # P-14/P-16: this closing mark belongs to a structural locant, not
        # an enclosed substituent that joins directly to an alkyl parent.
        return match[0]
    if re.search(r"(?:^|[-,([{])\d+[a-z]+'*$", match.string[:match.start()]):
        # A fusion locant such as 4a ends in a letter, but its hyphen is
        # still the locant/prefix boundary, not an organyl-parent join.
        return match[0]
    preceding = re.search(r"[a-z]+$", match.string[: match.start()])
    if preceding and preceding[0].endswith(("t", "tert", "sec")):
        # t-/tert-/sec- are structural descriptors, including multiplied
        # forms such as dit-butyl; their hyphens are compulsory.
        return match[0]
    return match["parent"]


def _join_hydro(match):
    # A lettered fusion locant is part of the hydro operation, not an
    # alphabetic substituent prefix. Its locant/operation hyphen is mandatory.
    before = match.string[:match.start()]
    locants = re.search(r"(?:^|[-([{])(?:\d+[a-z]*'*)(?:,\d+[a-z]*'*)*$", before)
    return match[0] if locants or _stereo_before(match.string, match.start()) else ""


def _stereo_before(text, end):
    if not end or text[end - 1] != ")":
        return False
    start = text.rfind("(", 0, end)
    return start >= 0 and _STEREODESCRIPTOR.fullmatch(text[start + 1 : end - 1]) is not None


def _parse_groups(name):
    stack = [([], "", "")]
    closing = {")": "(", "]": "[", "}": "{"}
    text = ""
    for char in name:
        if char in "([{":
            if text:
                stack[-1][0].append(text)
            stack.append(([], char, text))
            text = ""
        elif char in closing:
            if len(stack) == 1 or stack[-1][1] != closing[char]:
                return None
            if text:
                stack[-1][0].append(text)
            children, opening, before = stack.pop()
            raw = _raw(children)
            # P-14.7.2 places added indicated H immediately after the locant
            # of its structural operation. P-16.5.4.1.1 ignores only those
            # parentheses. A nuclide such as an unlocanted isotope-H token
            # before its modified unit participates in nesting (.1.3), even
            # though its characters can also match the digit-plus-H grammar.
            ignored = (opening == "(" and _FIXED_HYDROGEN.fullmatch(raw) is not None
                       and re.search(r"\d+[a-z']*$", before) is not None)
            numeric_superscript = (opening in {'{', '('} and re.fullmatch(r'\d+',raw) is not None
                                   and re.search(r'\d+\^$',before) is not None)
            n_site_qualifier = (opening == '(' and re.fullmatch(r"\d+'*(?:\^\d+)?",raw) is not None
                                and re.search(r"(?:^|[^A-Za-z])N'*\^$",before) is not None)
            # A fixed nuclide enclosure contributes to nesting, unlike a
            # skeletal bracket or a superscript's presentation marks.
            nuclide = opening == '[' and re.fullmatch(r'\d+[A-Z][a-z]?\d*', raw) is not None
            structural = (opening == "[" and _structural_bracket(raw, before)
                          or numeric_superscript or n_site_qualifier)
            stereo = opening == "(" and _STEREODESCRIPTOR.fullmatch(raw) is not None
            # Indicated hydrogen and skeletal descriptors retain their marks;
            # skeletal brackets do not contribute to prefix nesting levels.
            stack[-1][0].append(
                _Group(children, opening, structural or ignored or stereo,
                       (structural and not nuclide) or ignored)
            )
            text = ""
        else:
            text += char
    if len(stack) != 1:
        return None
    if text:
        stack[0][0].append(text)
    return stack[0][0]


def _structural_bracket(raw, before):
    if re.search(r"(?:cyclo|spiro(?:bi)?|fulleren)$", before):
        return True
    if re.fullmatch(r"[0-9,.'^:()\-]+", raw):
        return True
    # Fusion of hydrocarbon components can omit all numerical locants;
    # [a], [cd], and [a,e] remain structural descriptors in that case.
    if _FUSION_SIDES.fullmatch(raw) and re.search(r"[a-z]$", before):
        return True
    if (re.search(r"[a-z]$", before)
            and re.fullmatch(r"[0-9,.'^:;\-]+(?:[a-z]{1,4}'*(?:[0-9,.'^:;\-]+[a-z]{1,4}'*)*)?",
                             raw)):
        return True
    if re.fullmatch(r"[0-9',:;]+-(?:biphenyl|bipyridin|binaphthalen|terphenyl)", raw):
        return True
    # P-16.5.2.1: the brackets belong to the assembly parent before a
    # principal/cumulative suffix. Composite superscripts and integral
    # bi(cyclo...) parentheses do not change that structural boundary.
    assembly_locant = r"\d+(?:\^(?:\{\d+\}|\(\d+\)|\d+))?'*"
    assembly = (r"(?:bi(?!s)|ter|quater|quinque|sexi)[a-z]+"
                r"|bi\(cyclo(?:\[[\d.'^,()]+\])?[a-z]+\)")
    if re.fullmatch(rf"{assembly_locant}(?:[,:]{assembly_locant})*-(?:{assembly})", raw):
        return True
    return re.fullmatch(r"\d+[A-Z][a-z]?\d*", raw) is not None


def _raw(nodes):
    return "".join(
        n
        if isinstance(n, str)
        else n.opening + _raw(n.children) + dict(_MARKS)[n.opening]
        for n in nodes
    )


def _preferred_units(nodes):
    result, changed = [], False
    for index, node in enumerate(nodes):
        if isinstance(node, _Group):
            if node.fixed:
                result.append(node)
                continue
            raw = _raw(node.children)
            stereo_acyl = re.fullmatch(
                r"(\((?:[0-9]*[RSrs])(?:,[0-9]*[RSrs])*\))-([a-z]+)", raw
            )
            if stereo_acyl and stereo_acyl[2] in _ACYL_PREFIXES:
                node.children = _parse_groups(
                    stereo_acyl[1] + "-" + _ACYL_PREFIXES[stereo_acyl[2]]
                )
                changed = True
                result.append(node)
                continue
            assembly = _biphenyl_prefix(raw)
            if assembly:
                node.children = _parse_groups(assembly)
                changed = True
            elif _COMPOUND.fullmatch(raw) and not _COMPOUND.fullmatch(raw)["loc"]:
                # An already enclosed complete unit needs a spelling change,
                # not another redundant enclosing group.
                preferred = _ACYL_PREFIXES.get(raw, raw)
                node.children = [preferred]
                changed |= preferred != raw
                if preferred in _SIMPLE_ACYL_PREFIXES and result and isinstance(result[-1], str):
                    for count, multiplier in COMPLEX_MULT.items():
                        if result[-1].endswith(multiplier):
                            result[-1] = result[-1][:-len(multiplier)] + MULT[count]
                            changed = True
                            break
                result.append(node)
                continue
            node.children, child_changed = _preferred_units(node.children)
            changed |= child_changed
            result.append(node)
            continue
        offset = 0
        for match in _COMPOUND.finditer(node):
            result.append(node[offset : match.start()])
            loc = match["loc"] or ""
            locants = loc[:-1].split(",") if loc else []
            multiplier = match["mult"]
            before = node[: match.start()]
            if not loc and (match.start() != 0 or any(nodes[:index])):
                # A text node after an enclosed substituent can be its parent
                # fragment: (indol-3-yl)methylcarbonyl, not two substituents.
                result.append(match[0])
                offset = match.end()
                continue
            descriptor = re.search(r"(?:^|[^a-z])(?:t|tert|sec)-$", before)
            implicit_acyl_count = (
                next((count for count, text in MULT.items() if text == multiplier), 0)
                if not loc and _ACYL_PREFIXES.get(match["unit"]) in _SIMPLE_ACYL_PREFIXES
                else 0
            )
            count = len(locants) or implicit_acyl_count
            if descriptor or multiplier and MULT.get(count) != multiplier:
                result.append(match[0])
            else:
                unit = _ACYL_PREFIXES.get(match["unit"], match["unit"])
                changed |= unit != match["unit"]
                multiplicative = (
                    MULT[count] if unit in _SIMPLE_ACYL_PREFIXES
                    else COMPLEX_MULT[count]
                ) if multiplier else ""
                result.append(loc + multiplicative)
                result.append(_Group([unit], "("))
            offset = match.end()
        result.append(node[offset:])
    if changed:
        result = _reorder_mononuclear_prefixes(result)
        result = _reorder_prefix_run(result)
    return result, changed


_LOCANT_FIELD = re.compile(r"(?:[0-9]+[a-z']*|[NOSP])(?:,(?:[0-9]+[a-z']*|[NOSP]))*-")
_SIMPLE_UNITS = tuple(
    sorted(
        {
            "amino",
            "azido",
            "bromo",
            "chloro",
            "fluoro",
            "iodo",
            "hydroxy",
            "oxo",
            "nitro",
            "cyano",
            "carboxy",
            "carboxylato",
            "sulfanyl",
            "phenyl",
            "benzyl",
            "formyl",
            "acetyl",
            "benzoyl",
            "carbamoyl",
            "methylidene",
            "hydroperoxy",
            *(_root + "yl" for _root in ROOTS.values()),
            *(_root + "oxy" for _root in ROOTS.values()),
        },
        key=len,
        reverse=True,
    )
)
_SIMPLE_MULTIPLIERS = tuple(
    sorted((m for n, m in MULT.items() if n > 1), key=len, reverse=True)
)
_REPLACEMENT_PARENT = re.compile(
    r"(?:" + "|".join(re.escape(m) for m in _SIMPLE_MULTIPLIERS) + r")?"
    r"(?:" + "|".join(sorted(set(REPLACEMENT.values()), key=len, reverse=True)) + r")(?=[a-z-])"
)


def _reorder_mononuclear_prefixes(nodes):
    """Recite complete unlocanted substituents on a mononuclear nitrogen.

    Enclosures delimit each substituent; a terminal fragment of a substituted
    acyl group is never treated as another nitrogen substituent here.
    """
    items = [n for n in nodes if n != ""]
    if len(items) < 3 or not isinstance(items[-1], str):
        return nodes
    if items[-1] not in ("amino", "amine", "azane", "azanide", "azaniumyl"):
        return nodes
    records = []
    for item in items[:-1]:
        if isinstance(item, _Group):
            if item.fixed:
                return nodes
            name = _raw(item.children)
        elif item in _SIMPLE_UNITS:
            name = item
        else:
            return nodes
        records.append((prefix_alphabetization_key(name), item))
    ordered = sorted(records, key=lambda r: r[0])
    if ordered == records:
        return nodes
    result = []
    for index, (_key, item) in enumerate(ordered):
        if index and isinstance(item, str):
            item = _Group([item], "(")
        result.append(item)
    return result + [items[-1]]


def _balanced_end(text, start):
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for pos in range(start, len(text)):
        char = text[pos]
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return None
            if not stack:
                return pos + 1
    return None


def _reorder_prefix_run(nodes):
    """Recite supported prefix fields after an acyl/assembly spelling change.

    Field locants and parent spelling are retained. An incomplete parse keeps
    the original citation, rather than guessing an implicit attachment locant.
    """
    raw = _raw(nodes)
    heading = ""
    if raw[:1] in "([{":
        end = _balanced_end(raw, 0)
        if (
            end
            and raw[end : end + 1] == "-"
            and re.fullmatch(
                r"[0-9a-z,'()*]*[RSZErs](?:,[0-9a-z,'()*]*[RSZErs])*", raw[1 : end - 1]
            )
        ):
            heading, raw = raw[: end + 1], raw[end + 1 :]
    pos, records = 0, []
    while True:
        field = _LOCANT_FIELD.match(raw, pos)
        if not field:
            break
        if _REPLACEMENT_PARENT.match(raw, field.end()):
            # Skeletal replacement descriptors belong to the parent, rather
            # than to the substituent citation run preceding it.
            break
        loc = field[0][:-1]
        start = field.end()
        multiplier = ""
        complex_multiplier = re.match(
            r"(bis|tris|tetrakis|pentakis|hexakis)(?=[(\[{])", raw[start:]
        )
        if complex_multiplier:
            multiplier = complex_multiplier[0]
            start += len(multiplier)
        if start < len(raw) and raw[start] in "([{":
            end = _balanced_end(raw, start)
            if end is None:
                return nodes
            unit = raw[start:end]
            key_name = unit[1:-1]
            body = multiplier + unit
        else:
            if multiplier:
                return nodes
            simple_multiplier = next(
                (m for m in _SIMPLE_MULTIPLIERS if raw.startswith(m, start)), ""
            )
            unit_start = start + len(simple_multiplier)
            unit = next((u for u in _SIMPLE_UNITS if raw.startswith(u, unit_start)), "")
            if not unit:
                return nodes
            end = unit_start + len(unit)
            if end < len(raw) and raw[end] != "-" and not raw[end].isalpha():
                return nodes
            if end < len(raw) and raw[end].isalpha():
                # A simple-token prefix is not a proved complete field.
                # For example benzyl at the start of benzyloxymethyl cannot
                # move independently of the residual oxymethyl construction.
                # Preserve the producer's complete citation on any such tie.
                return nodes
            key_name = unit
            body = simple_multiplier + unit
        records.append((loc, body, prefix_alphabetization_key(key_name)))
        pos = end
        if pos < len(raw) and raw[pos] == "-" and _LOCANT_FIELD.match(raw, pos + 1):
            pos += 1
            continue
        break
    if len(records) < 2 or pos >= len(raw):
        return nodes
    # Reject a residual prefix field we could not parse, including descriptors
    # such as sec-/tert-, to avoid detaching it from its parent or suffix.
    if (
        _LOCANT_FIELD.match(raw, pos)
        and not _REPLACEMENT_PARENT.match(raw, _LOCANT_FIELD.match(raw, pos).end())
        or raw[pos : pos + 1] == "-"
        and _LOCANT_FIELD.match(raw, pos + 1)
    ):
        return nodes
    parent = raw[pos:]
    fields = "-".join(
        loc + "-" + body for loc, body, _key in sorted(records, key=lambda r: r[2])
    )
    separator = "-" if _LOCANT_FIELD.match(parent) else ""
    return _parse_groups(heading + fields + separator + parent) or nodes


def _biphenyl_prefix(raw):
    match = _BIPHENYL.fullmatch(raw)
    if not match:
        return ""
    # The attached phenyl has no other substituents. Re-root it at the
    # inter-ring junction; the free-valence locant has the same ortho/meta/
    # para distance. The distal ring keeps its numbering, now primed.
    attachment = int(match["attachment"])
    attachment = min(attachment, 8 - attachment)
    prefix = re.sub(r"[2-6]", lambda m: m[0] + "'", match["prefix"] or "").rstrip("-")
    return prefix + f"[1,1'-biphenyl]-{attachment}-yl"


def _render(nodes):
    parts, depth = [], 0
    for node in nodes:
        if isinstance(node, str):
            parts.append(node)
            continue
        if node.fixed:
            parts.append(
                node.opening + _raw(node.children) + dict(_MARKS)[node.opening]
            )
            level = 0 if node.ignored else 1
        else:
            content, child_depth = _render(node.children)
            level = child_depth + 1
            opening, closing = _MARKS[(level - 1) % len(_MARKS)]
            children = [n for n in node.children if n != ""]
            if children and (
                isinstance(children[0], _Group)
                and not children[0].ignored
                and content.startswith(opening)
                or isinstance(children[-1], _Group)
                and not children[-1].ignored
                and content.endswith(closing)
            ):
                # P-16.5.4.1.5: avoid adjacent marks of the same level,
                # including stereodescriptors after a full nesting cycle.
                level += 1
                opening, closing = _MARKS[(level - 1) % len(_MARKS)]
            parts.append(opening + content + closing)
        depth = max(depth, level)
    return "".join(parts), depth
