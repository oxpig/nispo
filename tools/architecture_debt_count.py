#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "nispo"
RESOURCE_ROOT = PACKAGE_ROOT / "resources"

PRIORITY_RULE_MODULES = {
    "acyclic_rules.py",
    "branch_rules.py",
    "parent_rules.py",
    "polycyclic_rules.py",
    "ring_rules.py",
}

TEMPLATE_LAYER_MODULES = {
    "resource_templates.py",
    "scaffold_catalog.py",
    "template_access.py",
    "template_index.py",
    "template_store.py",
}

CENTRAL_RULE_MODULES = {
    "api.py",
    "molecule_cache.py",
    "naming_context.py",
    "resource_manifest.py",
    "rule_dispatch.py",
    "rule_engine.py",
    "scaffold_catalog.py",
    "template_access.py",
    "template_index.py",
    "template_store.py",
}

BYPASS_CALL_NAMES = {
    "build_generated_fused_parent_templates",
    "build_generated_fused_parent_templates_from_records",
    "build_opsin_resource_templates",
    "build_retained_parent_templates",
    "build_static_parent_templates",
    "candidate_generated_fused_templates",
    "candidate_templates",
    "curated_fused_templates",
    "generated_fused_parent_templates",
    "name_acyclic",
    "resource_ring_parent_candidates",
    "ring_parent_candidates",
}


def main() -> None:
    report = architectural_debt_report()
    print(json.dumps(report, indent=2, sort_keys=True))


def architectural_debt_report() -> dict[str, Any]:
    categories = {
        "unregistered_name_producing_rule_surfaces": _count_unregistered_name_surfaces(),
        "direct_template_dict_access_outside_template_layer": _count_direct_template_access(),
        "duplicated_generated_fused_resource_records": _count_generated_resource_duplication(),
        "rdkit_mol_nispo_hidden_cache_write_sites": _count_hidden_mol_cache_writes(),
        "rule_or_naming_paths_bypassing_central_registries": _count_registry_bypass_calls(),
    }
    return {
        "architectural_debt_count": sum(categories.values()),
        "categories": categories,
        "notes": {
            "scope": str(PACKAGE_ROOT),
            "generated_resource_duplication": (
                "Embedded generated fused records in derived JSON indexes beyond the unique "
                "(name, smiles, locants) records they reference."
            ),
            "rule_surfaces": (
                "Name-producing functions in priority rule modules that are not represented "
                "as first-class registry/rule objects."
            ),
        },
    }


def _python_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.glob("*.py"))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _count_unregistered_name_surfaces() -> int:
    total = 0
    for path in _python_files():
        if path.name not in PRIORITY_RULE_MODULES:
            continue
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.FunctionDef) and _looks_name_producing(node.name):
                total += 1
    return total


def _looks_name_producing(name: str) -> bool:
    if name in {"candidates"}:
        return False
    return (
        name.startswith("name_")
        or name.startswith("_name_")
        or name.startswith("_named_")
        or name.endswith("_name")
        or name.endswith("_names")
        or name.endswith("_prefix")
        or name.endswith("_prefixes")
    )


def _count_direct_template_access() -> int:
    total = 0
    for path in _python_files():
        if path.name in TEMPLATE_LAYER_MODULES:
            continue
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Subscript) and _is_template_name(node.value):
                total += 1
    return total


def _is_template_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id in {"template", "templates"}


def _count_generated_resource_duplication() -> int:
    unique_records = _record_keys(_load_json("generated_fused_templates.json"))
    embedded_records = []
    for filename in (
        "generated_fused_template_index.json",
        "generated_fused_template_graph_index.json",
        "generated_fused_template_element_graph_index.json",
    ):
        embedded_records.extend(_iter_records(_load_json(filename)))
    return max(0, len(embedded_records) - len(set(unique_records)))


def _load_json(filename: str):
    path = RESOURCE_ROOT / filename
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []


def _iter_records(value):
    if isinstance(value, dict):
        if "name" in value and "smiles" in value:
            yield value
            return
        for item in value.values():
            yield from _iter_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_records(item)


def _record_keys(records) -> set[tuple[str, str, tuple[str, ...]]]:
    return {_record_key(record) for record in _iter_records(records)}


def _record_key(record: dict) -> tuple[str, str, tuple[str, ...]]:
    locants = record.get("locants") or ()
    return (
        str(record.get("name", "")),
        str(record.get("smiles", "")),
        tuple(str(item) for item in locants),
    )


def _count_hidden_mol_cache_writes() -> int:
    total = 0
    for path in _python_files():
        if path.name == "molecule_cache.py":
            continue
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Attribute) and node.attr.startswith("_nispo_"):
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    total += 1
            elif isinstance(node, ast.Call) and _is_setattr_hidden_cache_write(node):
                total += 1
    return total


def _is_setattr_hidden_cache_write(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Name) or node.func.id != "setattr":
        return False
    if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
        return False
    return isinstance(node.args[1].value, str) and node.args[1].value.startswith("_nispo_")


def _count_registry_bypass_calls() -> int:
    total = 0
    for path in _python_files():
        if path.name in CENTRAL_RULE_MODULES:
            continue
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in BYPASS_CALL_NAMES:
                    total += 1
    return total


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


if __name__ == "__main__":
    main()
