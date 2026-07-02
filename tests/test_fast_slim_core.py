from __future__ import annotations

import importlib.resources
import json

from rdkit import Chem

from nispo.api import _smiles_to_iupac_batch_with_telemetry
from nispo.naming_context import MoleculeContext
from nispo.parent_rules import parent_rules
from nispo.rule_dispatch import compile_rules_for_context


def test_context_exposes_lazy_molecule_facts_and_route_plan():
    mol = Chem.MolFromSmiles("c1ccccc1")
    ctx = MoleculeContext(mol, Chem)

    facts = ctx.facts
    route = ctx.route_plan

    assert facts.ring_count == 1
    assert facts.element_counts == (("C", 6),)
    assert route.primary_route == "simple_monocycle"
    assert route.routes == ("simple_monocycle",)


def test_benchmark_telemetry_reports_routes_rules_and_fallbacks():
    names, telemetry = _smiles_to_iupac_batch_with_telemetry(
        ["CC", "c1ccccc1", "C1CC2CCC1C2"],
        branch_miss_shape_detail=True,
    )

    assert len(names) == 3
    assert "primary_route_count" in telemetry
    assert "route_plan_count" in telemetry
    assert "rule_evaluations" in telemetry
    assert "rule_gate_skips" in telemetry
    assert "compiled_rule_skip_count" in telemetry
    assert "legacy_fallback_attempt_count" in telemetry


def test_compiled_rules_keep_every_rule_that_can_apply():
    mol = Chem.MolFromSmiles("CC(=O)O")
    ctx = MoleculeContext(mol, Chem)
    rules = parent_rules(Chem)

    compiled = set(compile_rules_for_context(ctx, rules))
    applicable = {rule for rule in rules if rule.applies(ctx)}

    assert applicable <= compiled
    assert len(compiled) < len(rules)


def test_generated_fused_indexes_use_compact_record_ids():
    resource_root = importlib.resources.files("nispo").joinpath("resources")
    for filename in (
        "generated_fused_template_index.json",
        "generated_fused_template_graph_index.json",
        "generated_fused_template_element_graph_index.json",
    ):
        data = json.loads(resource_root.joinpath(filename).read_text(encoding="utf-8"))
        assert not _contains_embedded_generated_record(data)


def _contains_embedded_generated_record(value) -> bool:
    if isinstance(value, dict):
        if "name" in value and "smiles" in value:
            return True
        return any(_contains_embedded_generated_record(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_embedded_generated_record(item) for item in value)
    return False
