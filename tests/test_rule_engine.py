from __future__ import annotations

from rdkit import Chem

from nispo.branch_rendering import BranchRenderContext, render_branch_candidates
from nispo.telemetry import CandidateTelemetry, merge_telemetry_dicts, telemetry_to_dict


def test_candidate_telemetry_serializes_and_merges_deterministically():
    left = CandidateTelemetry()
    left.rule_seconds["slow-rule"] = 0.1234567
    left.candidate_count["slow-rule"] = 2
    left.rendered_candidate_count["slow-rule"] = 1
    left.selected_with_parent_atoms = 1
    left.substructure_match_calls = 3

    right = CandidateTelemetry()
    right.rule_seconds["slow-rule"] = 0.2
    right.rule_seconds["other-rule"] = 0.1
    right.candidate_count["slow-rule"] = 4
    right.rejected_candidate_count["other-rule"] = 5
    right.selected_with_template_id = 2
    right.substructure_match_calls = 7

    serialized = telemetry_to_dict(left)
    merged = merge_telemetry_dicts((serialized, telemetry_to_dict(right)))

    assert serialized["rule_seconds"] == {"slow-rule": 0.123457}
    assert merged["rule_seconds"] == {"other-rule": 0.1, "slow-rule": 0.323457}
    assert merged["candidate_count"] == {"slow-rule": 6}
    assert merged["rendered_candidate_count"] == {"slow-rule": 1}
    assert merged["rejected_candidate_count"] == {"other-rule": 5}
    assert merged["selected_with_parent_atoms"] == 1
    assert merged["selected_with_template_id"] == 2
    assert merged["substructure_match_calls"] == 10


def test_branch_prefix_miss_telemetry_records_graph_shape():
    mol = Chem.MolFromSmiles("CC(=O)C(=O)C")
    default_telemetry = CandidateTelemetry()
    default_ctx = BranchRenderContext(mol=mol, telemetry=default_telemetry)
    assert render_branch_candidates(default_ctx, 1, 0) == ()
    assert default_telemetry.branch_structured_prefix_misses == {"simple_external_prefix": 1}

    telemetry = CandidateTelemetry(branch_miss_shape_detail=True)
    ctx = BranchRenderContext(mol=mol, telemetry=telemetry)

    assert render_branch_candidates(ctx, 1, 0) == ()

    keys = list(telemetry.branch_structured_prefix_misses)
    assert len(keys) == 1
    assert keys[0].startswith("simple_external_prefix:root=C;bond=1;")
    assert "hetero=O2" in keys[0]
    assert "unsat=2" in keys[0]
