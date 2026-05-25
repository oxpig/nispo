from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .resource_templates import (
    build_fusion_component_parent_templates,
    build_generated_fused_parent_templates,
    build_opsin_resource_templates,
    build_retained_parent_templates,
    build_static_parent_templates,
)


@dataclass(frozen=True)
class ParentScaffoldCatalog:
    """Data-driven parent scaffold families used by the naming rules."""

    static_retained: tuple[dict, ...]
    opsin_retained_fused: tuple[dict, ...]
    fusion_component_parents: tuple[dict, ...]
    fused: tuple[dict, ...]
    exact: tuple[dict, ...]
    single_ring: tuple[dict, ...]
    by_source_family: dict[str, tuple[dict, ...]]


@lru_cache(maxsize=8)
def parent_scaffold_catalog(Chem) -> ParentScaffoldCatalog:
    static_retained = build_static_parent_templates(Chem)
    opsin_retained_fused = build_opsin_resource_templates(Chem)
    fusion_component_parents = build_fusion_component_parent_templates(Chem)
    exact = _dedupe_templates(
        build_retained_parent_templates(Chem, min_rings=0, max_atoms=48)
        + static_retained
        + fusion_component_parents,
        Chem,
    )
    single_ring = build_retained_parent_templates(Chem, min_rings=1, max_rings=1)
    fused = static_retained + opsin_retained_fused + fusion_component_parents
    return ParentScaffoldCatalog(
        static_retained=static_retained,
        opsin_retained_fused=opsin_retained_fused,
        fusion_component_parents=fusion_component_parents,
        fused=fused,
        exact=exact,
        single_ring=single_ring,
        by_source_family=_group_by_source_family(fused + exact + single_ring),
    )


@lru_cache(maxsize=8)
def generated_fused_parent_templates(Chem) -> tuple[dict, ...]:
    return build_generated_fused_parent_templates(Chem)


def _group_by_source_family(templates: tuple[dict, ...]) -> dict[str, tuple[dict, ...]]:
    grouped: dict[str, list[dict]] = {}
    for template in templates:
        grouped.setdefault(template.get("source_family", "unknown"), []).append(template)
    return {family: tuple(items) for family, items in sorted(grouped.items())}


def _dedupe_templates(templates: tuple[dict, ...], Chem) -> tuple[dict, ...]:
    unique = []
    seen: set[tuple[str, str]] = set()
    for template in templates:
        key = (template["name"], Chem.MolToSmiles(template["mol"], canonical=True))
        if key in seen:
            continue
        seen.add(key)
        unique.append(template)
    return tuple(unique)
