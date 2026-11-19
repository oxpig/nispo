"""P-57 preferred vocabulary at complete substituent-component boundaries.

General-nomenclature names and parser aliases remain recognisable, but a
component producer emits its preferred form. This never rewrites substrings in
an assembled molecular name. Unlisted components retain their existing policy.
Retained tert-butyl and benzyl apply only to their unsubstituted exact units.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComponentTerm:
    preferred: str
    source_status: str


SUBSTITUENT_VOCABULARY = {
    # P-66.4.1.3.1: this complete acyl-derived group replaces the former
    # OPSIN resource aliases; do not rewrite arbitrary assembled names.
    "amidino": ComponentTerm("carbamimidoyl", "not-recommended"),
    "guanyl": ComponentTerm("carbamimidoyl", "not-recommended"),
    # P-65.2.1.5, for complete carbonyl-attached amide units.
    "carbamyl": ComponentTerm("carbamoyl", "parser-alias"),
    "methylcarbamyl": ComponentTerm("methylcarbamoyl", "parser-alias"),
    "dimethylcarbamyl": ComponentTerm("dimethylcarbamoyl", "parser-alias"),
    "isopropyl": ComponentTerm("propan-2-yl", "general-nomenclature"),
    "1-methylethyl": ComponentTerm("propan-2-yl", "systematic-alternative"),
    "1-methyl-ethyl": ComponentTerm("propan-2-yl", "parser-alias"),
    "isobutyl": ComponentTerm("2-methylpropyl", "not-recommended"),
    "2-methyl-propyl": ComponentTerm("2-methylpropyl", "parser-alias"),
    "sec-butyl": ComponentTerm("butan-2-yl", "not-recommended"),
    "1-methylpropyl": ComponentTerm("butan-2-yl", "systematic-alternative"),
    "1-methyl-propyl": ComponentTerm("butan-2-yl", "parser-alias"),
    "isopentyl": ComponentTerm("3-methylbutyl", "not-recommended"),
    "tert-pentyl": ComponentTerm("2-methylbutan-2-yl", "not-recommended"),
    "neopentyl": ComponentTerm("2,2-dimethylpropyl", "not-recommended"),
    # P-57.1.6: name this complete graph-matched alkenyl unit systematically.
    "methallyl": ComponentTerm("2-methylprop-2-en-1-yl", "general-nomenclature"),
    "phenethyl": ComponentTerm("2-phenylethyl", "not-recommended"),
    "benzhydryl": ComponentTerm("diphenylmethyl", "not-recommended"),
    "trityl": ComponentTerm("triphenylmethyl", "general-nomenclature"),
    "isopropylidene": ComponentTerm("propan-2-ylidene", "general-nomenclature"),
    "t-butyl": ComponentTerm("tert-butyl", "parser-alias"),
    "1,1-dimethylethyl": ComponentTerm("tert-butyl", "systematic-alternative"),
    "1,1-dimethyl-ethyl": ComponentTerm("tert-butyl", "parser-alias"),
    "isopropoxy": ComponentTerm("(propan-2-yl)oxy", "general-nomenclature"),
    "isobutoxy": ComponentTerm("2-methylpropoxy", "not-recommended"),
    "isopentyloxy": ComponentTerm("3-methylbutoxy", "not-recommended"),
    "neopentyloxy": ComponentTerm("2,2-dimethylpropoxy", "not-recommended"),
    "t-butoxy": ComponentTerm("tert-butoxy", "parser-alias"),
    "tert-butyl": ComponentTerm("tert-butyl", "preferred-retained"),
    "benzyl": ComponentTerm("benzyl", "preferred-retained"),
}

# P-29.3.3/.4 and P-29.6.2: these are complete simple components, not
# substring replacements in substituted or assembled names.
for _old, _stem, _positions in (
    ("furyl", "furan", (2, 3)),
    ("thienyl", "thiophen", (2, 3)),
    ("pyridyl", "pyridin", (2, 3, 4)),
    ("piperidyl", "piperidin", (2, 3, 4)),
):
    for _position in _positions:
        SUBSTITUENT_VOCABULARY[f"{_position}-{_old}"] = ComponentTerm(
            f"{_stem}-{_position}-yl", "general-nomenclature")
for _old, _preferred in (
    ("morpholino", "morpholin-4-yl"),
    ("piperidino", "piperidin-1-yl"),
    ("pyrrolidino", "pyrrolidin-1-yl"),
    ("piperazino", "piperazin-1-yl"),
):
    SUBSTITUENT_VOCABULARY[_old] = ComponentTerm(_preferred, "general-nomenclature")


def preferred_component(name):
    term = SUBSTITUENT_VOCABULARY.get(name)
    return term.preferred if term is not None else name


def preferred_acyl_component(name):
    """P-65.1.7 preferred complete acyl units at an ester-prefix boundary."""
    return {'propionyl':'propanoyl','butyryl':'butanoyl','valeryl':'pentanoyl',
            'isobutyryl':'2-methylpropanoyl','pivaloyl':'2,2-dimethylpropanoyl',
            'acryloyl':'prop-2-enoyl'}.get(name,name)


def with_preferred_keys(mapping):
    """Recognition tables accept preferred forms without changing semantics."""
    result = dict(mapping)
    for alias, term in SUBSTITUENT_VOCABULARY.items():
        if alias in mapping:
            result.setdefault(term.preferred, mapping[alias])
    # Recognition of retained functional parents can still use their old
    # substituted-benzyl keys. This adds equivalent complete-component keys;
    # it neither emits those aliases nor changes a retained-parent preference.
    for alias, value in mapping.items():
        if alias.endswith('benzyl') and alias!='benzyl':
            result.setdefault(f"({alias[:-len('benzyl')]}phenyl)methyl", value)
    return result


_ALIASES_BY_PREFERRED = {}
for _alias, _term in SUBSTITUENT_VOCABULARY.items():
    _ALIASES_BY_PREFERRED.setdefault(_term.preferred, []).append(_alias)


def recognised_component(name, allowed):
    return name in allowed or any(
        alias in allowed for alias in _ALIASES_BY_PREFERRED.get(name, ())
    )


def component_recognition_length(name):
    """Keep legacy scope bounds invariant under approved unit expansion.

    This is an eligibility budget, never emitted name text. Complete preferred
    units (and their own enclosure) have the cost of a recognised source unit;
    unrelated long components do not acquire support from a spelling change.
    """
    if len(name)<=48:
        return len(name)
    import re

    length = len(name)
    for preferred, aliases in _ALIASES_BY_PREFERRED.items():
        cost = min(len(alias) for alias in aliases)
        if cost >= len(preferred):
            continue
        pattern = (
            r"\("
            + re.escape(preferred)
            + r"\)|(?<![A-Za-z])"
            + re.escape(preferred)
            + r"(?![A-Za-z])"
        )
        length -= sum(len(match.group()) - cost for match in re.finditer(pattern, name))
    return length
