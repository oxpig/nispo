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


def test_n_acyl_branch_families_preserve_public_graphs_and_inputs():
    from nispo import (mol_to_iupac, mol_to_iupac_result, smiles_to_iupac_batch,
                       smiles_to_iupac_batch_results)
    from test_roundtrip_failure_families import _assert_roundtrips

    inputs = (
        "c1cc(NC(=O)OC)ccc1",
        "c1cc(NC(=O)OCC)ccc1",
        "c1cc(NC(=O)C(F)(F)F)ccc1",
        "CC(=O)N1CC2CCC1[C@@H]2N",
        "CC(=O)N1C[C@H]2CC[C@@H]1[C@@H]2N",
        "CC(=O)NCC1=C(F)C=C(F)C=C1",
    )
    _assert_roundtrips(inputs)
    names = smiles_to_iupac_batch(inputs)
    assert [r.name for r in smiles_to_iupac_batch_results(inputs)] == names
    assert smiles_to_iupac_batch(reversed(inputs)) == list(reversed(names))
    for smiles, name in zip(inputs, names, strict=True):
        mol = Chem.MolFromSmiles(smiles)
        before = Chem.MolToMolBlock(mol)
        assert mol_to_iupac(mol) == mol_to_iupac_result(mol).name == name
        assert Chem.MolToMolBlock(mol) == before


def test_common_ring_graphs_survive_public_naming_and_atom_reordering():
    """Exercise these ring families without freezing branch spelling."""
    from nispo import smiles_to_iupac_batch
    from test_roundtrip_failure_families import _assert_roundtrips

    inputs = ("CC1CCC(F)(F)CC1", "CC1CC(F)(F)C1", "Nc1nc(C)cs1", "CC1=CCOCC1")
    reordered = []
    for smiles in inputs:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        reverse = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        reordered.append(Chem.MolToSmiles(reverse, canonical=False, isomericSmiles=True))
    _assert_roundtrips([*inputs, *reordered])
    assert smiles_to_iupac_batch(inputs) == smiles_to_iupac_batch(reordered)
