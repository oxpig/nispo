from __future__ import annotations

from functools import lru_cache

from .template_store import ParentScaffoldCatalog, template_store


@lru_cache(maxsize=8)
def parent_scaffold_catalog(Chem) -> ParentScaffoldCatalog:
    return template_store(Chem).catalog


@lru_cache(maxsize=8)
def generated_fused_parent_templates(Chem) -> tuple[dict, ...]:
    return template_store(Chem).generated_fused_parent_templates()
