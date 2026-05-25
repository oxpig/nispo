#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "nispo"
TOOLS_ROOT = REPO_ROOT / "tools"
TESTS_ROOT = REPO_ROOT / "tests"
RESOURCE_ROOT = PACKAGE_ROOT / "resources"


def main() -> None:
    print(json.dumps(audit_closure_score(), indent=2, sort_keys=True))


def audit_closure_score() -> dict:
    items = [
        _score_template_store(),
        _score_generated_resources(),
        _score_context_cache(),
        _score_rule_routing_and_modularity(),
        _score_structured_rendering(),
        _score_suffix_layer(),
        _score_fragment_layer(),
        _score_diagnostics(),
        _score_tests(),
    ]
    return {
        "audit_closure_score": sum(item["score"] for item in items),
        "max_score": 100,
        "items": items,
    }


def _score_template_store() -> dict:
    path = PACKAGE_ROOT / "template_store.py"
    if not path.exists():
        return _item("typed_template_store_resource_layer", 0, 15, ["missing template_store.py"])
    tree = _parse(path)
    classes = _class_names(tree)
    source = path.read_text(encoding="utf-8")
    score = 0
    evidence = []
    for name in ("TemplateRecord", "CompiledTemplate", "TemplateFeatures", "TemplateStore"):
        if name in classes:
            score += 2
            evidence.append(f"{name} exists")
    method_names = _method_names(tree, "TemplateStore")
    for name in (
        "static_retained",
        "opsin_retained_fused",
        "fusion_component_parents",
        "generated_fused_parent_templates",
        "candidate_templates",
        "compiled_template",
        "template_features",
        "scaffold_key",
    ):
        if name in method_names:
            score += 1
            evidence.append(f"TemplateStore.{name} exists")
    template_mutations = _template_dict_nispo_mutations()
    evidence.append(f"template dict _nispo_* mutation sites: {template_mutations}")
    if template_mutations:
        score = min(score, 11)
    if "legacy" in source or "compat" in source:
        score = min(score, 12)
        evidence.append("TemplateStore still exposes legacy/compatibility naming")
    return _item("typed_template_store_resource_layer", max(0, min(score, 15)), 15, evidence)


def _score_generated_resources() -> dict:
    evidence = []
    score = 0
    record_table = RESOURCE_ROOT / "generated_fused_template_index_records.json"
    if record_table.exists():
        score += 3
        evidence.append("canonical generated fused index record table exists")
        evidence.append(f"record table size bytes: {record_table.stat().st_size}")
    for filename in (
        "generated_fused_templates.json",
        "generated_fused_template_index.json",
        "generated_fused_template_graph_index.json",
        "generated_fused_template_element_graph_index.json",
    ):
        path = RESOURCE_ROOT / filename
        if path.exists() and _json_uses_integer_leaves(path):
            score += 1 if filename == "generated_fused_templates.json" else 2
            evidence.append(f"{filename} uses compact integer leaves")
        elif path.exists():
            evidence.append(f"{filename} still embeds full records")
    if (RESOURCE_ROOT / "generated_fused_templates.json").exists() and not _json_uses_integer_leaves(
        RESOURCE_ROOT / "generated_fused_templates.json"
    ):
        score = min(score, 6)
    return _item("generated_fused_resource_consolidation", min(score, 10), 10, evidence)


def _score_context_cache() -> dict:
    debt = _architecture_debt()
    hidden_writes = debt.get("categories", {}).get("rdkit_mol_nispo_hidden_cache_write_sites")
    evidence = [f"hidden RDKit _nispo_* write sites: {hidden_writes}"]
    score = 0
    context = PACKAGE_ROOT / "naming_context.py"
    if hidden_writes == 0:
        score += 4
    if context.exists() and "TemplateStore" in context.read_text(encoding="utf-8"):
        score += 2
        evidence.append("MoleculeContext owns TemplateStore access")
    if (PACKAGE_ROOT / "molecule_cache.py").exists():
        score += 1
        evidence.append("central molecule_cache exists, but not full ContextCache ownership")
    if "class ContextCache" in _package_text():
        score += 3
        evidence.append("ContextCache exists")
    return _item("molecule_context_cache_ownership", min(score, 10), 10, evidence)


def _score_rule_routing_and_modularity() -> dict:
    branch_path = PACKAGE_ROOT / "branch_rules.py"
    parent_path = PACKAGE_ROOT / "parent_rules.py"
    branch_lines = branch_path.read_text(encoding="utf-8").splitlines()
    parent_lines = parent_path.read_text(encoding="utf-8").splitlines()
    branch_text = "\n".join(branch_lines)
    parent_text = "\n".join(parent_lines)
    debt = _architecture_debt().get("categories", {})
    unregistered = int(debt.get("unregistered_name_producing_rule_surfaces", 9999))
    bypass = int(debt.get("rule_or_naming_paths_bypassing_central_registries", 9999))
    parent_class_count = parent_text.count("class ")
    score = 0
    evidence = [
        f"branch_rules.py lines: {len(branch_lines)}",
        f"parent_rules.py lines: {len(parent_lines)}",
        f"parent_rules.py class count: {parent_class_count}",
        f"unregistered name-producing rule surfaces: {unregistered}",
        f"registry bypass calls: {bypass}",
    ]
    if "RuleSpec" in parent_text and "RulePlan" not in (PACKAGE_ROOT / "rule_dispatch.py").read_text(encoding="utf-8"):
        score += 5
        evidence.append("parent naming uses explicit RuleSpec without RulePlan introspection")
    if "BranchRuleSpec" in branch_text and (PACKAGE_ROOT / "branch").exists():
        score += 4
        evidence.append("BranchRuleSpec present")
    if len(branch_lines) <= 1200:
        score += 4
    elif len(branch_lines) <= 1800:
        score += 2
    if len(parent_lines) <= 900 or parent_class_count <= 8:
        score += 4
    elif len(parent_lines) <= 1300 or parent_class_count <= 14:
        score += 2
    if unregistered <= 30:
        score += 4
    elif unregistered <= 60:
        score += 2
    if bypass <= 5:
        score += 3
    elif bypass <= 10:
        score += 1
    return _item("explicit_rule_routing_and_module_decomposition", min(score, 20), 20, evidence)


def _score_structured_rendering() -> dict:
    text = _package_text()
    evidence = []
    score = 0
    if "class NameParts" in text:
        score += 6
        evidence.append("NameParts exists")
    if "class Prefix" in text and "class Suffix" in text:
        score += 2
        evidence.append("structured Prefix/Suffix records exist")
    usage_count = text.count("NameParts(")
    evidence.append(f"NameParts constructor uses: {usage_count}")
    if usage_count >= 4:
        score += 2
    elif usage_count:
        score += 1
    return _item("structured_rendering_layer", min(score, 10), 10, evidence or ["missing structured rendering layer"])


def _score_suffix_layer() -> dict:
    text = _package_text()
    evidence = []
    score = 0
    if "class SuffixSpec" in text and "class InverseSuffixStore" in text:
        score += 4
        evidence.append("suffix spec/store exists")
    if "suffixRules.xml" in text and "suffixApplicability.xml" in text:
        score += 3
        evidence.append("bundled OPSIN suffix resources are referenced by code")
    runtime_uses = [
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if path.name != "suffix_store.py"
        and path.name != "__init__.py"
        and "suffix_store" in path.read_text(encoding="utf-8")
    ]
    evidence.append(f"runtime suffix_store users: {[path.relative_to(PACKAGE_ROOT).as_posix() for path in runtime_uses]}")
    if runtime_uses:
        score += 3
    return _item("inverse_suffix_resource_layer_drives_naming", min(score, 10), 10, evidence or ["missing inverse suffix layer"])


def _score_fragment_layer() -> dict:
    text = _package_text()
    evidence = []
    score = 0
    if "FragmentRule" in text or "FragmentAssemblyRule" in text:
        score += 3
        evidence.append("fragment rule class/spec exists")
    if (PACKAGE_ROOT / "fragment_rules.py").exists():
        score += 2
        evidence.append("fragment_rules.py exists")
    fragment_text = (PACKAGE_ROOT / "fragment_rules.py").read_text(encoding="utf-8") if (PACKAGE_ROOT / "fragment_rules.py").exists() else ""
    named_fragment_functions = sum(
        1
        for node in ast.walk(ast.parse(fragment_text or "\n"))
        if isinstance(node, ast.FunctionDef) and ("fragment" in node.name or "anion" in node.name or "cation" in node.name)
    )
    evidence.append(f"fragment/cation/anion functions: {named_fragment_functions}")
    if named_fragment_functions < 4:
        score = min(score, 3)
    return _item("first_class_fragment_assembly", min(score, 5), 5, evidence or ["fragment assembly remains in main engine"])


def _score_diagnostics() -> dict:
    candidates = [
        REPO_ROOT.parent / "benchmark" / "explain_rows.py",
        TOOLS_ROOT / "explain_rows.py",
    ]
    evidence = [str(path) for path in candidates if path.exists()]
    return _item("selected_row_diagnostics", 5 if evidence else 0, 5, evidence or ["missing selected-row diagnostic tool"])


def _score_tests() -> dict:
    if not TESTS_ROOT.exists():
        return _item("focused_tests", 0, 5, ["missing tests directory"])
    tests = sorted(path.name for path in TESTS_ROOT.glob("test_*.py"))
    score = min(5, len(tests))
    return _item("focused_tests", score, 5, tests)


def _architecture_debt() -> dict:
    import importlib.util

    path = TOOLS_ROOT / "architecture_debt_count.py"
    spec = importlib.util.spec_from_file_location("architecture_debt_count", path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.architectural_debt_report()


def _item(name: str, score: int, max_score: int, evidence: list[str]) -> dict:
    return {
        "name": name,
        "score": score,
        "max_score": max_score,
        "evidence": evidence,
    }


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_names(tree: ast.Module) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def _method_names(tree: ast.Module, class_name: str) -> set[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
    return set()


def _json_uses_integer_leaves(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    leaves = []
    _collect_list_leaves(value, leaves)
    return bool(leaves) and all(all(isinstance(item, int) for item in leaf) for leaf in leaves)


def _collect_list_leaves(value, leaves: list[list]) -> None:
    if isinstance(value, list):
        leaves.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_list_leaves(item, leaves)


def _package_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _template_dict_nispo_mutations() -> int:
    total = 0
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                if _subscript_key_starts_nispo(node.slice):
                    total += 1
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"setdefault", "update"} and _call_writes_nispo_key(node):
                    total += 1
    return total


def _subscript_key_starts_nispo(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.startswith("_nispo_")
    return False


def _call_writes_nispo_key(node: ast.Call) -> bool:
    for arg in node.args:
        if isinstance(arg, ast.Dict):
            for key in arg.keys:
                if key is not None and _subscript_key_starts_nispo(key):
                    return True
    for keyword in node.keywords:
        if keyword.arg and keyword.arg.startswith("_nispo_"):
            return True
    return False


if __name__ == "__main__":
    main()
