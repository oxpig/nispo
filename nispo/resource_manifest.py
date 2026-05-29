from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpsinResourceFamily:
    name: str
    files: tuple[str, ...]
    role: str
    note: str = ""


RETAINED_PARENT_FAMILY = OpsinResourceFamily(
    name="retained-parent-tokens",
    role="parent",
    note="Standalone OPSIN group tokens that can act as retained parent skeletons.",
    files=(
        "alkanes.xml",
        "aminoAcids.xml",
        "arylGroups.xml",
        "atomHydrides.xml",
        "carbohydrates.xml",
        "cyclicUnsaturableHydrocarbon.xml",
        "groupStemsAllowingAllSuffixes.xml",
        "groupStemsAllowingInlineSuffixes.xml",
        "naturalProducts.xml",
        "simpleCyclicGroups.xml",
        "simpleGroups.xml",
    ),
)


FUSION_COMPONENT_FAMILY = OpsinResourceFamily(
    name="fusion-component-tokens",
    role="fusion-component",
    note="OPSIN FR component vocabulary. NISPO keeps this separate from standalone parents.",
    files=("fusionComponents.xml",),
)


SUBSTITUENT_FAMILY = OpsinResourceFamily(
    name="substituent-tokens",
    role="substituent",
    note="OPSIN monovalent and multivalent substituent fragments usable as prefixes.",
    files=(
        "arylSubstituents.xml",
        "carbohydrates.xml",
        "multiRadicalSubstituents.xml",
        "simpleSubstituents.xml",
        "substituents.xml",
    ),
)


WORD_AND_SUFFIX_FAMILY = OpsinResourceFamily(
    name="word-and-suffix-tokens",
    role="grammar",
    note="Packaged for rule development; not interpreted directly by the current name generator.",
    files=(
        "functionalTerms.xml",
        "hwSuffixes.xml",
        "infixes.xml",
        "inlineChargeSuffixes.xml",
        "inlineSuffixes.xml",
        "miscTokens.xml",
        "carboxylicAcids.xml",
        "nonCarboxylicAcids.xml",
        "suffixPrefix.xml",
        "suffixes.xml",
        "unsaturators.xml",
        "wordRules.xml",
    ),
)


OPSIN_RESOURCE_FAMILIES = (
    RETAINED_PARENT_FAMILY,
    FUSION_COMPONENT_FAMILY,
    SUBSTITUENT_FAMILY,
    WORD_AND_SUFFIX_FAMILY,
)


def files_for_role(role: str) -> tuple[str, ...]:
    files: list[str] = []
    for family in OPSIN_RESOURCE_FAMILIES:
        if family.role == role:
            files.extend(family.files)
    return tuple(dict.fromkeys(files))
