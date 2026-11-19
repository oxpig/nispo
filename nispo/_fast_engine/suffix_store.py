from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from xml.etree import ElementTree


HIGH_IMPACT_SUFFIXES = {
    "al",
    "amide",
    "amidine",
    "ide",
    "ium",
    "nitrile",
    "oate",
    "oic acid",
    "ol",
    "one",
    "sulfone",
    "sulfonamide",
    "sulfoxide",
    "thioamide",
    "thione",
    "ylium",
}


@dataclass(frozen=True, slots=True)
class SuffixSpec:
    value: str
    rule_key: str
    source_file: str
    operation_tags: tuple[str, ...] = ()


class InverseSuffixStore:
    """Compiled OPSIN suffix resources for structure-to-name suffix work."""

    def __init__(self, specs: tuple[SuffixSpec, ...]):
        self.specs = specs
        self.by_value = _group_by_value(specs)

    def high_impact_specs(self) -> tuple[SuffixSpec, ...]:
        return tuple(spec for spec in self.specs if spec.value in HIGH_IMPACT_SUFFIXES)


@lru_cache(maxsize=1)
def inverse_suffix_store() -> InverseSuffixStore:
    applicability = _suffix_applicability()
    rules = _suffix_rules()
    specs = []
    for value, rule_key in sorted(applicability.items()):
        operation_tags = rules.get(rule_key, ())
        if value in HIGH_IMPACT_SUFFIXES or operation_tags:
            specs.append(
                SuffixSpec(
                    value=value,
                    rule_key=rule_key,
                    source_file="suffixApplicability.xml/suffixRules.xml",
                    operation_tags=operation_tags,
                )
            )
    return InverseSuffixStore(tuple(specs))


def _suffix_applicability() -> dict[str, str]:
    root = _read_opsin_xml("suffixApplicability.xml")
    if root is None:
        return {}
    mapping: dict[str, str] = {}
    for suffix in root.iter():
        value = suffix.attrib.get("value")
        if value and suffix.text:
            mapping[value.strip()] = suffix.text.strip()
    return mapping


def _suffix_rules() -> dict[str, tuple[str, ...]]:
    root = _read_opsin_xml("suffixRules.xml")
    if root is None:
        return {}
    rules: dict[str, tuple[str, ...]] = {}
    for rule in root:
        rule_key = rule.attrib.get("value") or rule.attrib.get("type") or rule.tag
        tags = tuple(child.tag for child in rule)
        if rule_key:
            rules[rule_key] = tags
    return rules


def _read_opsin_xml(filename: str):
    try:
        path = resources.files(__package__).joinpath("resources", "opsin_tokens", filename)
        return ElementTree.fromstring(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _group_by_value(specs: tuple[SuffixSpec, ...]) -> dict[str, tuple[SuffixSpec, ...]]:
    grouped: dict[str, list[SuffixSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.value, []).append(spec)
    return {value: tuple(items) for value, items in grouped.items()}
