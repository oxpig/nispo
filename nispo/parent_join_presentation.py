"""Punctuation at a rendered prefix/parent boundary."""

import re


_FUSION_START = re.compile(
    r"^[a-z]+\[(?:(?:[0-9]+[a-z]?'*,)*[0-9]+[a-z]?'*-)?"
    r"[a-z]'*(?:,[a-z]'*)*\](?=[a-z\[])"
)
_TRIAZOLE_FUSION_ALIAS = re.compile(r"(?<![a-z])(?:s|sym)-triazolo(?=\[)")

# A functional suffix is already complete when this parent becomes a branch.
CARBONYL_SUFFIX_ENDINGS = ("-one", "-dione", "-trione", "-tetrone", "-pentone",
                            "-thione", "-dithione", "-trithione", "-selone", "-tellone")


def starts_with_fusion_component(parent):
    """The fusion descriptor is internal to an alphabetic parent name."""
    return _FUSION_START.match(parent) is not None


def normalize_fusion_component_spelling(parent):
    """The legacy triazole fusion aliases share the [1,2,4] locant map."""
    if "triazolo[" in parent:
        return _TRIAZOLE_FUSION_ALIAS.sub("[1,2,4]triazolo", parent)
    return parent


def omit_monosubstituted_benzene_locant(prefix_text):
    """P-14.3.3–4: distinguish the benzene's locants from nested prefix locants."""
    if not prefix_text.startswith("1-"):
        return prefix_text
    group = prefix_text[2:]
    if "-" not in group:
        return group
    depth = 0
    for index, char in enumerate(group):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "-" and not depth and group[index + 1:index + 2].isdigit():
            return prefix_text
    return group if depth == 0 else prefix_text
