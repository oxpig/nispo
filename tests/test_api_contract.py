from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from shutil import which

import pytest
from rdkit import Chem

from nispo import smiles_to_iupac, smiles_to_iupac_batch
from nispo.api import _engine, _smiles_to_iupac_batch_with_telemetry
from nispo.naming_context import MoleculeContext
from nispo.rule_engine import _prepare_candidates


def test_public_api_single_and_batch_contract():
    assert smiles_to_iupac("c1ccccc1")

    inputs = ["CC", "not-a-smiles", ""]
    outputs = smiles_to_iupac_batch(inputs)

    assert len(outputs) == len(inputs)
    assert outputs[0]
    assert outputs[1:] == ["", ""]


def test_batch_does_not_abort_on_unsupported_molecule():
    outputs = smiles_to_iupac_batch(["CC", "[Fe+2]", "c1ccccc1"])

    assert len(outputs) == 3
    assert outputs[0]
    assert outputs[2]


def test_batch_matches_scalar_for_mixed_inputs():
    inputs = [
        "c1ccccc1",
        "not-a-smiles",
        "C1CCCCC1",
        "",
        "O=C1NC(=O)CC1",
        "c1ccc2ccccc2c1",
    ]

    scalar_outputs = [smiles_to_iupac(smiles) for smiles in inputs]
    batch_outputs = smiles_to_iupac_batch(inputs)

    assert len(batch_outputs) == len(inputs)
    assert batch_outputs == scalar_outputs


def test_private_telemetry_batch_matches_public_batch():
    inputs = ["CC", "not-a-smiles", "", "c1ccccc1", "c1ccc2ccccc2c1"]

    public_outputs = smiles_to_iupac_batch(inputs)
    telemetry_outputs, telemetry = _smiles_to_iupac_batch_with_telemetry(inputs)

    assert telemetry_outputs == public_outputs
    assert len(telemetry_outputs) == len(inputs)
    assert telemetry_outputs[1:3] == ["", ""]
    assert "rule_applies" in telemetry
    assert "candidate_count" in telemetry
    assert "branch_prefix_calls" in telemetry
    assert "branch_prefix_produced" in telemetry
    assert "branch_legacy_prefix_detail" in telemetry
    assert "substructure_match_calls" in telemetry


def test_positive_replacement_heterocycle_parent_roundtrips_with_opsin():
    input_smiles = "C1CC[PH2+]CC1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize("input_smiles", ["C1=C[N]N=C1", "C1=CSC[N]1"])
def test_lambda2_unsaturated_heterocycles_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C1C[SH4]C1",
        "C1CC[SH4]C1",
        "C1=CC=[SH3]C=C1",
        "C1CC[PH3]C1",
        "C1=CC=[PH2]C=C1",
    ],
)
def test_hypervalent_ring_lambda_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC(C)(C)[C@H]1C=CN(c2cccc(N3C=C[C@H](C(C)(C)C)C3)n2)C1",
        "CC(C)[C@H]1COC(c2cccc(C3=N[C@@H](C(C)C)CO3)n2)=N1",
        "O=c1[nH]c(C2C=NON2NCCN2CC=NS2(=O)=O)no1",
    ],
)
def test_unsaturated_replacement_heterocycle_substituents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Cc1cc2[nH]nc(I)c2s1",
        "NC(=O)c1cc2c(-c3ccncc3)n[nH]c2s1",
    ],
)
def test_thienopyrazole_generated_fused_parent_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OS(=O)(=O)O)[C@@H](O)[C@H]1O",
        "CC(C)N(C)Cc1cc2c(-c3ccc4[nH]nc(N)c4c3)ccnc2[nH]1",
        "[N-]=[N+]=NC(COCCOCCOCCOCCOCCOCCC(N)=O)Cc1ccc(COc2nc(N)nc3[nH]cnc23)cc1",
    ],
)
def test_generated_fused_graph_supplement_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CON(C)C(=O)c1cc(F)c(F)c2[nH]ncc12",
        "[O-][n+]1cc(Br)c2cn[nH]c2c1",
        "Nc1n[nH]c2nc3ccccc3nc12",
        "Nc1[nH]nc2nnccc12",
        "COc1nnc(C)c2c(C)[nH]nc12",
        "O=S(=O)(Oc1ccc2cnc3[nH]ncc3c2c1)C(F)(F)F",
        "Cn1cc(-c2ccc3cnc4[nH]ncc4c3c2)cn1",
        "Nc1nc2cc(-c3cc[nH]n3)ccc2c2nc(C3CCNC3)[nH]c12",
        "CCOCOc1cc(C(F)(F)F)cc(C)c1-c1cnc2c[nH]nc2n1",
        "CC1(NC(=O)OC(C)(C)C)CCN(c2nc3[nH]nc(-c4cccc(Cl)c4Cl)c3nc2C(N)=O)CC1",
    ],
)
def test_generated_pyrazole_fused_orientation_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_retained_hydro_fused_template_pair_roundtrips_with_opsin():
    input_smiles = "Cc1cc(C#C[C@]2(O)CCc3cccnc32)cc(-c2ncc3ccnc(N)c3n2)c1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=C1/C(=C(\\Nc2ccccc2)c2cccc(Cl)c2)c2ccccc2C1c1ccccc1",
        "O=S(=Nc1ccc(Cl)c(Cl)c1)(Oc1ccccc1)Oc1ccc(-c2ccccc2)cc1",
    ],
)
def test_earlier_external_hetero_single_ring_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C1=NCCc2nc[nH]c21",
        "Cc1nn(C)c2c1CC(N1CCc3nc(-c4ccccc4Cl)[nH]c3C1)CC2",
        "CCc1cc(O)c(F)cc1-c1ccc2c(-c3nc4c([nH]3)CN(C(C)C)C(C(=O)N3CC(N(C)C)C3)C4)n[nH]c2c1",
    ],
)
def test_generated_hydro_pyridino_imidazole_parent_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCNC(=O)CC[C@H](N)C(=O)N1CCc2nc[nH]c2C1",
        "Fc1ccccc1-c1nc2c([nH]1)CN(C1CC3CC3C1)CC2",
    ],
)
def test_generated_hydro_pyrido45d_imidazole_parent_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "NC(=O)c1ccc(C2Cc3cccc(-c4cncnc4)c32)c(C(N)=O)n1",
        "COc1ccc2c(c1)C(CN1CCN(C3CCC34CCNCC4)C(c3ccccc3C(C)C)C1)C2",
        "O=C1NC[C@@H](C(=O)N[C@H](c2ccc(Cl)cc2)c2ccc3c(c2)CC3(F)F)N1",
        "CC(C)(O)c1coc(S(=N)(=O)NC(=O)Nc2c(CC3CCC3)ccc3c2CC3)c1",
        "CNCc1cc2c(cc1-c1ccc([C@@H](C)NC(=O)c3ccc(=O)n(-c4ccccc4F)c3)s1)CC2",
        "CO[C@@H]1O[C@H]([C@H](OC(=O)c2ccc(-c3ccccc3)cc2)c2ccc3c(c2)CC3)[C@H]2OC(C)(C)O[C@@H]12",
        "CN1CC[C@@H]2CCN(c3nnc(-c4ccc5c(c4O)CC5)n(C)c3=O)[C@@H]2C1",
        "CC(C)(NC(=O)OCc1ccccc1)c1cc(OC2C3CN(C(=O)[O-])CC32)nc(-c2ccc3c(c2)CC3(F)F)c1",
    ],
)
def test_retained_benzocyclobutene_parent_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C1=C[NH2+]N=C1",
        "C1C[NH2+]N1",
        "C1=NCC[NH2+]1",
        "C1C[NH+]=NN1",
        "C1=C[NH2+]C=NC1",
        "[NH2+]1CCSCC1",
    ],
)
def test_positive_single_ring_cn_heterocycles_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Cc1cc[o+]cc1",
        "CCc1c[nH+]c[nH]1",
        "O=C(O)n1c[nH+]nn1",
        "N#CC(C#N)c1cccc[o+]1",
    ],
)
def test_charged_aromatic_ring_skeleton_parent_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=c1c[n+](-c2ccccc2)[n-]o1",
        "C[n+]1[n-]oc(=O)c1",
    ],
)
def test_sydnone_retained_parent_roundtrips(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=c1[nH]cno1",
        "O=c1[nH]cn[nH]1",
        "O=c1[nH]nn[nH]1",
        "O=c1nncc[nH]1",
        "O=c1ncnn[nH]1",
        "S=c1[nH]cn[nH]1",
        "S=c1nc[nH]cn1",
    ],
)
def test_single_ring_lactam_and_thione_indicated_h_suffixes_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "COc1nccc(-c2ccc3c(c2)cc(C(=O)O)n3[C@@]2(c3noc(=O)[nH]3)C[C@@H]2C)c1C",
        "C[C@H]1C[C@]1(c1noc(=O)[nH]1)n1c(C(=O)O)c(N2CCOC(C)(C)C2)c2ccccc21",
        "C[C@H]1C[C@]1(c1noc(=O)[nH]1)n1c(C(=O)O)cc2cc(C3CCC(F)(F)CC3)ccc21",
        "CC1(C)C[C@H](c2ccn3c(C4(c5noc(=O)[nH]5)CC4)c(C(=O)O)cc3c2)CCO1",
        "CC1(C)C[C@@H](c2ccn3c(C4(c5noc(=O)[nH]5)CC4)c(C(=O)O)cc3c2)CCO1",
        "CN1C(=O)/C(=C/c2ccc3[nH]ncc3c2)N=C1N[C@H]1CCCOC1",
        "CCN1CC(=O)N([C@@H]2CCN(C(=O)c3cc(/C=C4\\OC(=O)c5ccccc54)ccc3F)C2)C1=O",
    ],
)
def test_single_ring_oxo_suffix_parent_selection_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=c1oc(C2CC2)ns1",
        "O=c1oc(-c2ccccn2)ns1",
    ],
)
def test_oxathiazolone_substituent_ring_parent_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=C(c1noc(=O)[nH]1)N1C[C@@H](c2ccccc2)C2(CCC2)C1",
        "O=C(c1noc(=O)[nH]1)N1CC(c2cccc(F)c2)C2(CCC2)C1",
        "O=C(c1noc(=O)[nH]1)N1CC(c2ccc(F)cc2)C2(CCC2)C1",
        "O=C(c1noc(=O)[nH]1)N1C[C@H](c2ccc(F)cc2)C2(CCC2)C1",
        "O=C(c1noc(=O)[nH]1)N1C[C@H](c2ccc(F)c(F)c2)C2(CCC2)C1",
        "O=C(OCc1ccccc1)N1CCC2(C[C@H]1c1ccc(-c3n[nH]c(=O)[nH]3)cc1)CC(F)(F)C2",
    ],
)
def test_senior_azolone_parent_deferred_from_spiro_core_roundtrips(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[C@@H]1OCC2(CCN(c3nc4[nH]nc(C5(c6ncco6)CC5)c4c(=O)[nH]3)CC2)[C@@H]1N",
        "C[C@@H]1OCC2(CCN(c3nc4[nH]nc(C5(c6ccccn6)CC5)c4c(=O)[nH]3)CC2)[C@@H]1N",
        "C[C@@H]1OCC2(CCN(c3nc4[nH]nc(C5(c6ccncc6)CC5)c4c(=O)[nH]3)CC2)[C@@H]1N",
        "C[C@@H]1OCC2(CCN(c3nc4[nH]nc(C5(c6cccc(Br)c6)CC5)c4c(=O)[nH]3)CC2)[C@@H]1N",
        "C[C@@H]1OCC2(CCN(c3nc4[nH]nc(C5(c6nc7ccccc7o6)CC5)c4c(=O)[nH]3)CC2)[C@@H]1N",
    ],
)
def test_senior_fused_oxo_parent_deferred_from_spiro_core_roundtrips(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[C@@H]1CN(c2nc(=O)n3c4c(c(-c5ccc(F)cc5F)c(Cl)cc24)OCC2(COC2)C3)[C@@H](C)CN1",
        "C=CC(=O)N1CCN(c2nc(=O)n3c4c(c(-c5ccc(F)cc5F)c(C(F)(F)F)cc24)SCC2(COC2)C3)CC1",
        "C[C@@H]1CN(c2nc(=O)n3c4c(c(-c5cc(Cl)c(F)cc5F)c(C(F)(F)F)cc24)SCC2(CCC2)C3)C[C@H](C)N1",
    ],
)
def test_component_spiro_oxo_orientation_roundtrips(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=c1[nH]c(=O)n2c3c(c(Cl)c(C(F)(F)F)cc13)SCC1(COC1)C2",
        "O=c1[nH]c(=O)n2c3c(c(Cl)c(C(F)(F)F)cc13)SCC1(CCC1)C2",
    ],
)
def test_component_spiro_dioxo_orientation_control_roundtrips(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC(C)(C)[C@H](NC(=O)Cc1ccc(Cl)c(Cl)c1)C(=O)N1CC2(CC2)C[C@H]1C(=O)NC(C#N)c1cncc2ccccc12",
        "CCN(C)S(=O)(=O)Nc1ccc(F)c(Oc2ccc3ncn([C@H]4CCC5(CCN(C(=O)OC(C)(C)C)CC5)C4)c(=O)c3c2)c1C#N",
        "CCN(C)S(=O)(=O)Nc1ccc(F)c(Oc2ccc3ncn([C@H]4CCC5(CCN(C(=O)OC(C)(C)C)CC5)C4)c(=O)c3c2Cl)c1C#N",
        "CCN(C)S(=O)(=O)Nc1ccc(F)c(Oc2ccc3ncn([C@H]4CCC5(CCN(C(=O)OC(C)(C)C)CC5)C4)c(=O)c3c2C)c1C#N",
    ],
)
def test_spiro_functional_chain_alternatives_beat_small_spiro_parents(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_spiro_functional_chain_defer_skips_noncarbon_group14_parent():
    input_smiles = "CC(C)(C)OC(=O)N1CCC2(CC1)CC(O[Si](C)(C)C(C)(C)C)C[C@H]2N[S@](=O)C(C)(C)C"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "NC(=O)[C@H](C[C@@H]1CCCNC1=O)NC(=O)C1CC2(CCCCC2)CN1C(=O)c1cc2c(Cl)cccc2[nH]1",
        "COc1cccc2[nH]c(C(=O)N3CC4(CCCCC4)CC3C(=O)N[C@@H](C[C@@H]3CCCNC3=O)C(N)=O)cc12",
        "COc1cccc2[nH]c(C(=O)N3CC4(CCCCC4)CC3C(=O)N[C@@H](C[C@@H]3CCNC3=O)C(N)=O)cc12",
        "COc1cc(Cl)c2[nH]c(C(=O)N3CC4(CCCCC4)CC3C(=O)N[C@@H](C[C@@H]3CCCNC3=O)C(N)=O)cc2c1",
    ],
)
def test_spiro_chain_deferred_to_single_ring_lactam_parent(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_spiro_chain_deferral_skips_charged_exocyclic_imino_ring_suffix():
    input_smiles = (
        "CCC1(C)CC(=O)N([C@H](CCOC)c2cccc(C(=O)NC3c4ccccc4C4(CC4)C3O)c2)"
        "C(=[NH2+])N1"
    )
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=C1Nc2ccc(F)cc2/C1=C1/Nc2ccccc2/C1=N\\OCC(=O)N1CC2(CNC2)C1",
        "CN(C(=O)C1CC2(COC(=O)N2)C1)C1CC(c2ccc3c(c2)CC3)C1",
    ],
)
def test_generic_spiro_deferred_to_compact_fused_retained_parent(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_generic_spiro_deferral_preserves_charged_parent():
    input_smiles = "Cn1ccc2c(OC3CC4(C[NH2+]C4)C3)ccc(Cl)c2c1=O"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_unsaturated_n_acyl_amino_acid_roundtrips_with_opsin():
    input_smiles = "CCCCCCCC/C=C\\CCCCCCCC(=O)N[C@@H](C)C(=O)O"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_parent_locanted_stereo_is_not_suppressed_by_substituent_locants():
    input_smiles = "O=C(CO[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O)[C@@H](O)[C@H](O)[C@H](O)CO"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_repeated_locanted_substituent_stereo_roundtrips_with_opsin():
    input_smiles = "C[C@@H]1O[C@@H](OCCC(CCO[C@@H]2O[C@@H](C)[C@@H](O)[C@@H](O)[C@@H]2O)(CC(N)=O)CC(N)=O)[C@@H](O)[C@H](O)[C@@H]1O"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC(N)C(=O)O",
        "CC[C@@H](C)[C@@H](N)C(=O)O",
        "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@@H](O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@H]1O",
        "CC(C)=CCC[C@H](C)CCO",
        "[NH3+][O-]",
    ],
)
def test_retained_shortcuts_defer_to_roundtripping_systematic_alternatives(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "[O-][I+][O-]",
        "[O-][NH+]([O-])[O-]",
        "O=C[O-]",
        "O=CO",
        "O=C([O-])C(F)(F)F",
        "N#C[O-]",
        "[C-]#[N+][O-]",
        "[O-]P([O-])([O-])=S",
        "[O-]P([O-])(=S)[S-]",
        "NO",
        "[N-]=O",
        "[N-]=N",
        "[O-][ClH+]",
        "[O-][BrH+]",
        "[NH-]I",
        "I[N-]I",
        "[Se]=[SeH-]",
        "[O-]P([O-])[O-]",
        "[O-]P([O-])OP([O-])[O-]",
        "[PH2](=O)[O-]",
        "N=P([O-])([O-])[O-]",
        "[O-]P([O-])([O-])=[Se]",
        "[O-][PH]([O-])=S",
        "[O-][PH](=S)[S-]",
        "[O-][SH]=S",
        "O=[SH]([O-])=S",
        "c1c[se]c[nH+]1",
        "[CH2-]C(C)=O",
        "C=N[O-]",
        "FI",
        "BrI",
        "O=c1nccco1",
        "O=c1cccccn1",
        "O=c1[nH]c1=O",
        "O=c1[nH]o[nH]1",
        "O=c1nn1",
        "O=c1ccccs1",
        "O=C1C=NO[NH+]1[O-]",
        "O=[C+]c1ccccc1",
        "b1ccco1",
        "c1coo1",
        "c1cso1",
        "c1nnno1",
        "O=c1nn[nH]o1",
        "c1cccoocc1",
        "O=S([O-])([O-])=S",
        "NS(=[NH2+])O",
        "N#CI(=O)=O",
        "O=[I+]1C=CC=C1",
    ],
)
def test_common_polyatomic_fragments_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=[PH2+]",
        "O=[PH-][O-]",
        "[O-]P([O-])(=[Se])[Se-]",
        "N#C[Se-]",
        "[N-]=[NH2+]",
        "[O-]F",
        "N#C[N+](=O)[O-]",
        "N#CS[O-]",
        "CS[O-]",
        "CCS[O-]",
        "[O-]P1CCCC1",
        "[O-]P1CCCCC1",
    ],
)
def test_charged_polyatomic_empty_coverage_fragments_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC(=[NH+][O-])C",
        "CCC(=[NH+][O-])C",
        "CC(C)(C)C(=[NH+][O-])c1ccccc1",
        "CC(C)(C)C(=[NH+][O-])C1C=CC(=S(=O)=O)CC1=S(=O)=O",
    ],
)
def test_two_substituent_nitrone_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[N+](C)([O-])C1CCCCC1",
        "C[N+](C)([O-])c1cccc2ccccc12",
        "CC[N+]([O-])(CC)C1=CCCCCCC1",
        "CCCCCCCCCCCCC(CCCCCCCCC)[N+](C)(C)[O-]",
        "C=C(C)[C@@H]1CC[C@](C)(O)[C@@H]([N+](C)(C)[O-])C1",
    ],
)
def test_tertiary_amine_oxide_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCCCCCCCCCC(=O)[NH2+][O-]",
        "C=CC(=O)[NH2+][O-]",
    ],
)
def test_primary_amide_oxide_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC[N+]([O-])(O)CC",
        "CCCCCCCCCCCC[N+]([O-])(O)CCCCCCCCCCCC",
        "C[N+]([O-])(O)CC",
        "C[N+]([O-])(O)O",
        "CC[N+]([O-])(O)O",
        "CCCCCCCCCCCC(CC)[N+]([O-])(O)O",
        "CCCCCCCCCCCCCCCCCCCC[N+]([O-])(O)O",
        "CC[NH+]([O-])O",
    ],
)
def test_hydroxylamine_oxide_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCCCCCCCC(C)/C=[N+](\\[O-])c1ccccc1",
        "CCCCCCCCCC(C)/C=[N+](/[O-])C1CCCCC1",
        "CCCCCCCCCC(C)/C=[N+](/[O-])C1CCCC1",
        "Cc1cc(CC(C)C)ccc1CC/C=[N+](/[O-])C1CCCCC1",
        "Cc1cc(CC(C)C)ccc1CC/C=[N+](\\[O-])C1CCC1",
        "CC(=CCC/C=[N+](/[O-])C1CCCCC1)CCCC(C)(C)O",
    ],
)
def test_n_substituted_imine_oxide_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O[N-]C",
        "O[N-]c1ccccc1",
        "S[N-]c1ccccc1",
        "CO[N-]c1ccccc1",
        "CO[N-]Cc1ccccc1",
        "CCS[N-]c1ccccc1",
        "CCCCC(CC)CS[N-]c1cc(Br)ccc1Br",
    ],
)
def test_hetero_substituted_anionic_amide_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[CH+]C",
        "C[CH-]C",
        "C=CO[N-]OC=C",
        "C[O+](C)C",
        "CC[O+](CC)CC",
        "[CH2+]Cc1ccccc1",
        "c1ccc([C+](c2ccccc2)c2ccccc2)cc1",
        "N#C[C-](C#N)C#N",
        "N#C[C+](C#N)C#N",
        "[O-][NH+]=Cc1ccccc1",
    ],
)
def test_high_impact_charged_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=C([O-])CCC(O)CC(=O)C(=O)[O-]",
        "O=C([O-])C(=O)CC(O)C(=O)[O-]",
        "O=C([O-])C(=O)C[C@H](O)C(=O)[O-]",
    ],
)
def test_hydroxy_oxo_dicarboxylate_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CNN(C)Cc1cc2ccccc2n1CCC(=O)NCC(=O)NCC(=O)NCC(=O)N(C)C(C)C(=O)[O-]",
        "CNN(C)Cc1cc2cccnc2n1CCC(=O)N(C)C(C)C(=O)[O-]",
        "CCC(=O)NCCNCC(=O)NCCOCCOCCC(=O)N(C)C(C)Cn1c(CN(C)NC)cc2ccccc21",
    ],
)
def test_complex_hydrazino_methyl_branches_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCOC(=O)CCC#N",
        "N#CCC(=O)N",
    ],
)
def test_acyl_parent_paths_do_not_absorb_nitrile_carbons(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_skeletal_replacement_fused_parent_prefers_numeric_replacement_locants():
    input_smiles = "c1ccc2c(c1)Nc1cc3ccccc3cc1O2"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC(C)n1ncc2c1C(=O)CC1(CCNCC1)C2",
        "O=c1c2c(cnn1-c1ncccn1)CC1(CC2)OCCO1",
    ],
)
def test_skeletal_replacement_fused_parents_align_template_bonds(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=C1Nc2cnc3c(ccn3S(=O)(=O)c3ccccc3)c2C12CCCC2",
        "CC1OC2(CCNCC2F)c2c1n(CC(=O)Nc1ccc(C(F)(F)F)cc1Cl)c1nc(Br)nn1c2=O",
    ],
)
def test_skeletal_replacement_spiro_fused_locants_stay_opsin_stable(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)OCc1ccccc1",
        "Cc1cccc(NC(=O)Nc2cccc(C)c2)c1",
        "Cc1ccccc1NC(=O)Nc1ccccc1C",
    ],
)
def test_retained_alias_shortcuts_defer_to_systematic_candidates(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Cc1ccccc1C(=O)OC1CCCCC1",
        "O=C(OC1CCCCC1)c1ccccc1C(=O)OC1CCCCC1",
        "C[C@@H](Oc1ccc(N)cc1)C1CC1",
        "CC(c1nnco1)C1CC1",
        "c1ccc(C2c3ccccc3-c3cccc(-c4ccc5ccccc5c4)c32)cc1",
    ],
)
def test_resource_templates_do_not_collapse_extra_core_bonds(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C=C(CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C)C(=O)O",
        "C=C(CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C)C(=O)[O-]",
    ],
)
def test_steroid_terminal_side_chain_natural_topology_roundtrips(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCC1(O)C(=O)Nc2nccc(Cl)c21",
        "CC1(C)C(=O)Nc2cc(-c3ccccc3)cnc21",
        "CC1(C)NC(=O)c2ccc(-c3c[nH]c4c(Cl)cccc34)nc21",
    ],
)
def test_locanted_diazaindane_lactams_defer_to_roundtripping_parent(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "[CH2-]CCC",
        "[CH2-]CCCC(C)C",
        "[CH2-]CCCCC(=O)[O-]",
        "[CH2-]C(=O)[O-]",
        "CCCCCC[C-]=O",
        "CC[C+]=O",
        "N#C[CH-]N=O",
        "C=[N+]([O-])[O-]",
        "CC(C)=[N+]([O-])[O-]",
        "OCC=[N+]([O-])[O-]",
        "CO[NH-]",
        "C=CCO[NH-]",
        "CC(C)(C)O[NH-]",
        "[NH-]OC=O",
        "CC(=O)O[NH-]",
        "[NH-]C(=O)C([NH-])=O",
        "[NH-]C(=O)CCCCC([NH-])=O",
        "C=[N-]",
        "CC(C)=[N-]",
        "CC(O)(P(=O)(O)O)P(=O)(O)O",
        "NCCCC(O)(P(=O)(O)O)P(=O)(O)O",
        "[CH-]=CCC",
        "[CH2+]CCC",
        "[CH2+]CCCC(C)C",
        "O=C1C=CC(=O)N1",
        "C1CC[IH]C1",
        "C1=CC[IH]C=C1",
        "C1=C[IH][IH]C1",
        "C1=C[IH]N=C1",
        "CC1(C)CC(=O)NC(=[NH2+])N1",
        "CCC1(CC)CC(=O)NC(=[NH2+])N1",
        "CCCCCCP(=O)=O",
        "O=C([O-])P(=O)=O",
        "O=P(=O)C=Cc1ccccc1",
        "N#CC(P(=O)=O)P(=O)=O",
        "O=P(=O)c1ccc(P(=O)=O)cc1",
        "O=C(P(=O)=O)P(=O)=O",
        "c1ccc([I+]c2ccccc2)cc1",
        "CC(C)(C)c1ccc([I+]c2ccc(C(C)(C)C)cc2)cc1",
        "Cc1ccc([I+]c2ccc(CC(C)C)cc2)cc1",
        "CC[I+]CC",
        "c1ccc(B(c2ccccc2)c2ccccc2)cc1",
        "CCB(CC)CC",
        "FC(F)(F)c1cc(B(c2cc(C(F)(F)F)cc(C(F)(F)F)c2)c2cc(C(F)(F)F)cc(C(F)(F)F)c2)cc(C(F)(F)F)c1",
        "[BH](c1ccccc1)c1ccccc1",
        "CC[BH2]",
        "O=[N+]([O-])Cl",
        "O=[N+]([O-])F",
        "CS(=O)([O-])=S",
        "Cc1ccc(S(=O)([O-])=S)cc1",
        "c1ccc(S(=O)([O-])=S)cc1",
        "O=S(=O)(F)[N-]S(=O)(=O)F",
        "O=S(=O)(F)[N-]S(=O)(=O)C(F)(F)F",
        "[NH-]S(=O)(=O)C(F)(F)F",
        "CS(=O)(=O)[N-]S(=O)(=O)F",
        "O=C1OI(C(F)(F)F)c2ccccc21",
        "CC1(C)OI(C(F)(F)F)c2ccccc21",
        "O=C1OI(=O)(O)c2ccccc21",
        "c1ccc2c(c1)CO[IH]2",
        "C1CCN(P(N2CCCC2)N2CCCC2)C1",
        "N1(CCCCC1)P(N1CCCCC1)N1CCCCC1",
        "CN(C)P(N(C)C)N(C)C",
        "O=c1nc(-c2ccccc2)ss1",
        "Nc1nc(=S)ss1",
        "CN(C)C=Nc1nc(=S)ss1",
        "[N-]=[N+]=Nc1ccc(C=C2CCCC(=Cc3ccc(N=[N+]=[N-])cc3)C2=O)cc1",
        "CC1CC(=Cc2ccc(N=[N+]=[N-])cc2)C(=O)C(=Cc2ccc(N=[N+]=[N-])cc2)C1",
        "Cc1ccccc1OOc1ccccc1C",
        "C[N+](C)(C#P=O)CCO",
        "C=CC(=O)OCC[P+]([O-])=C(O)C[N+](C)(C)C",
        "C=C(C)C(=O)OCC[P+]([O-])=C(O)C[N+](C)(C)C",
        "C" * 50,
        ("C" * 22) + "O" + ("C" * 22),
    ],
)
def test_empty_output_family_regressions_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)C=O",
        "O=C(CO)[C@H](O)[C@H](O)[C@H](O)CO",
        "O=C[C@H](O)CO",
    ],
)
def test_standalone_carbohydrate_retained_stems_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=c1oc1=O",
        "O=C1O[IH]c2ccccc21",
        "O=C1OI(Cl)c2ccccc21",
        "O=C1OI(Br)c2ccccc21",
    ],
)
def test_small_retained_oxirane_and_benziodoxolone_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "ClI(c1ccccc1)c1ccccc1",
        "Cc1ccc(I(Cl)c2ccccc2)cc1",
        "CC(C)(C)c1ccc(I(Cl)c2ccccc2)cc1",
    ],
)
def test_neutral_iodane_parent_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C1=C[IH]C=C1",
        "CC1=CC=C[IH]1",
        "C1=C[IH]C[IH]1",
        "C1=C[IH]OC1",
    ],
)
def test_neutral_iodine_hydride_ring_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C1=C[C+]=CCC1",
        "C1=C[C-]=CCC1",
        "c1ccc2[cH-]ccc2c1",
        "[C+]1=CC=CC2=CC=CC=C12",
    ],
)
def test_charged_carbon_ring_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O[B-]1(OCCO1)O",
        "O=C(O)c1cccc2c1O[B-](O)(O)CC2",
        "OB1OB(O)O[B-]2(O1)OB(O)OB(O)O2",
    ],
)
def test_boron_anion_ring_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_bridge_lactam_parent_roundtrips_with_opsin():
    input_smiles = "O=C1CC2CC2N1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "N=C1C=C2C(=O)NCC1C1=C2CNC1=O",
        "O=c1oc(=O)c2ccc1c1ccccc21",
    ],
)
def test_zero_bridge_tricyclic_parent_records_prepare_candidates(input_smiles):
    mol = Chem.MolFromSmiles(input_smiles)
    assert mol is not None

    prepared = _prepare_candidates(mol, Chem, MoleculeContext(mol, Chem), _engine()._rules)

    assert any(item.candidate.family == "kekulized-generic-polycyclic-parent" for item in prepared)


def test_fact_switch_gate_preserves_roundtripping_fused_azole_parent():
    input_smiles = "c1cc(-c2n[nH]c3cccnc23)c2nn[nH]c2c1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_fact_switch_gate_preserves_large_generated_fused_parent():
    input_smiles = "Brc1ccc2c(ccc3nc(-c4ccccc4)[nH]c32)c1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_fact_switch_gate_preserves_generated_fused_parent_against_retained_shortcut():
    input_smiles = "CSc1ncc2cc(-c3ccc4ncn(C)c4c3)c(=O)[nH]c2n1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "COc1ccc(O)cc1",
        "O=C(O)c1ccc(O)cc1",
        "CC1(C)OBOC1(C)C",
        "O=C1COc2ccccc2N1",
    ],
)
def test_retained_parent_aliases_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "NCc1ccc(O)c(OC)c1",
        "COc1ccc(O)c(O)c1",
    ],
)
def test_prefixed_methoxyphenol_aliases_stay_explicit_for_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "c1ccc2c(c1)C[C@H]1NCC[C@@]23CCCC[C@@H]13",
        "c1ccc2c(c1)C[C@H]1[N-]CC[C@@]23CCCC[C@@H]13",
        "c1ccc2c(c1)C[C@H]1[NH2+]CC[C@@]23CCCC[C@@H]13",
        "Oc1ccc2c(c1)[C@@]13CCCC[C@H]1[C@@H](C2)NCC3",
        "O=CN1CC[C@]23CCCC[C@H]2[C@H]1Cc1ccccc13",
        "CN1CC[C@]23CCCC[C@H]2[C@H]1Cc1ccccc13",
        "COc1ccc2c(c1)[C@@]13CCCC[C@H]1[C@@H](C2)N(C)CC3",
    ],
)
def test_morphin_retained_parent_terminalizes_to_morphinan(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "c1ccc2cc3cc4cc5cc6cc7cc8ccccc8cc7cc6cc5cc4cc3cc2c1",
        "c1ccc2cc3cc4cc5cc6cc7cc8cc9ccccc9cc8cc7cc6cc5cc4cc3cc2c1",
        "c1ccc2cc3cc4cc5cc6cc7cc8cc9cc%10ccccc%10cc9cc8cc7cc6cc5cc4cc3cc2c1",
    ],
)
def test_long_linear_acene_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C1=Cc2occc2OC1",
        "O=C1C[C@H]2SC(Cl)CCN12",
    ],
)
def test_fragile_fused_alias_fallbacks_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCCn1c(=O)oc(=O)c2ccccc21",
        "O=c1[nH]c2c(Cl)cccc2c(=O)o1",
        "O=c1[nH]c2c(Cl)cc(Cl)cc2c(=O)o1",
        "N#CCCc1cc2c(=O)oc(=O)[nH]c2c(F)c1Br",
    ],
)
def test_cyclic_anhydride_topology_replacements_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCc1cc2c(cc3ccccc3[n+]2S(=O)(=O)O)c(C)c1C",
        "CC=CCc1c(O)cc(O)c2c1OC(c1ccc(O)cc1)CC2=O",
        "O=c1[nH]c(-c2ccc3ccccc3c2)ns1",
    ],
)
def test_audit_ranked_parent_alternatives_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Cc1nc2[nH]nc(I)c2c(=O)[nH]1",
        "O=c1[nH]c(-c2cnccn2)nc2[nH]ncc12",
        "O=c1ccc2cn[nH]c2[nH]1",
        "O=c1[nH]cnc2[nH]ncc12",
        "O=c1[nH]ccc2[nH]cnc12",
    ],
)
def test_generated_fused_lactam_multiple_indicated_h_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC(C)c1[nH]nc2c(=O)[nH]c(=O)[nH]c12",
        "CCc1nn(CC)c2[nH]c(=O)[nH]c(=O)c12",
    ],
)
def test_generated_fused_dione_multiple_indicated_h_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Nc1nc2nc[nH]c2c(=S)[nH]1",
    ],
)
def test_generated_fused_thione_indicated_h_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCOC(=O)C=C1CCC2(CC1)OCCO2",
        "CCOC(=O)C(F)=C1CCC2(CC1)OCCO2",
        "C(CCOCc1ccccc1)=C1CCC2(CC1)OCCO2",
        "COC(=O)[C@@H](CCC(=O)C=C1CCC2(CC1)OCCO2)NC(=O)OC(C)(C)C",
    ],
)
def test_functionalized_spiro_alkylidene_substituents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=C(O)c1ccc2c(c1)N=CC1(C=COC1)O2",
        "O=C([O-])c1ccc2c(c1)N=CC1(C=COC1)O2",
        "CC(C)(C)OC(=O)N1CCC2(CC1)OCc1cc(Br)ccc1O2",
    ],
)
def test_explicit_hetero_component_spiro_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC1(C)Oc2ncccc2NC1=O",
        "O=C1COc2nc(Br)ccc2N1",
        "O=C1COc2cc(Br)cnc2N1",
        "C#Cc1cnc2c(c1)OCC(=O)N2",
    ],
)
def test_high_impact_pyrido_oxazinone_templates_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "FC(F)(F)[SiH](C(F)(F)F)C(F)(F)F",
        "CCCOC[Si](CCC)(CCC)COCCC",
        "CC(C)[Si](CCl)(CCl)C(C)C",
        "CCCCC(=C[SiH3])CCCC",
        "C=CC[SiH](F)Cl",
    ],
)
def test_group14_hetero_ligand_candidates_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCn1c(=NN=c2sc3cc(S(=O)(=O)O)ccc3n2CC)sc2cc(S(=O)(=O)O)ccc21",
        "CCn1c(=NN=c2sc3cc(S(=O)(=O)[O-])ccc3n2CC)sc2cc(S(=O)(=O)[O-])ccc21",
        "CCn1c(=NN=c2sc3ccccc3n2CC)sc2ccccc21",
    ],
)
def test_azinobis_benzothiazoline_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "[O-]P([O-])c1ccc(-c2ccc(P([O-])[O-])cc2)cc1",
        "CC(C)(C)c1ccc(OP([O-])c2ccc(-c3ccc(P([O-])[O-])cc3)cc2)c(C(C)(C)C)c1",
        "CC(C)(C)c1ccc(OP([O-])c2ccc(-c3cccc(P([O-])[O-])c3)cc2)c(C(C)(C)C)c1",
        "CC(C)(C)c1ccc(OP([O-])c2cccc(-c3cccc(P([O-])[O-])c3)c2)c(C(C)(C)C)c1",
    ],
)
def test_biphenyl_diphosphonite_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "COP([O-])[O-]",
        "COP(O)[O-]",
        "COP(O)O",
        "c1ccccc1OP([O-])[O-]",
        "CC(C)(C)c1ccc(OP([O-])[O-])c(C(C)(C)C)c1",
    ],
)
def test_monoester_phosphite_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCCCCP([O-])OCCCC",
        "CCCCOP([O-])SCCCC",
        "C=COP(N)[O-]",
        "N#CP([O-])[O-]",
        "CC[PH]([O-])(OC)OC",
        "CCSP([O-])SCC",
        "CCSP([S-])SCC",
        "[O-]P(Cl)Oc1ccccc1",
        "CC(C)N(C(C)C)P([O-])OCCC#N",
        "CCOP[O-]",
        "[O-]P(c1ccccc1)c1ccccc1",
        "Cc1cc(C)c(C(=O)c2ccccc2P([O-])c2ccccc2)c(C)c1",
    ],
)
def test_mixed_p3_anionic_phosphorus_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CP(O)O",
        "CCP(O)O",
        "c1ccccc1P(O)O",
        "C1CC1P(O)O",
    ],
)
def test_phosphonous_acid_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CP(=O)([O-])[O-]",
        "c1ccccc1P(=O)([O-])[O-]",
        "C#CP(N)(=O)[O-]",
        "CN(C)[PH](=O)[O-]",
        "CCN[PH](=O)[O-]",
        "COP(=O)[O-]",
        "COO[PH](=O)[O-]",
        "C=P([O-])([O-])[O-]",
        "CCCCC(CC)C(c1ccccc1)(c1ccccc1)P(=O)([O-])[O-]",
        "CP([O-])([O-])=S",
    ],
)
def test_charged_organophosphorus_oxo_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CN(C)P(=O)([O-])Cl",
        "CCN(CC)P(=O)([O-])F",
        "CC(C)N(C(C)C)P([O-])Cl",
        "CN(C)P([O-])Br",
    ],
)
def test_halo_phosphoramidite_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCCOP(=O)(O)Oc1ccccc1",
        "COP(=O)(O)O",
        "COP(=O)(O)OC",
        "CO[PH](=O)O",
        "COP(C)(=O)O",
        "COP(C)(=O)[O-]",
    ],
)
def test_phosphate_ester_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCC(=O)OP(=O)([O-])OP(=O)([O-])[O-]",
        "C=CC(=O)OP(=O)([O-])OP(=O)([O-])[O-]",
        "CC(C)=CCC/C(C)=C/C(=O)OP(=O)([O-])OP(=O)([O-])[O-]",
    ],
)
def test_acyl_diphosphate_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_sydnone_urea_amidate_retained_parent_roundtrips_with_opsin():
    input_smiles = "O=C([N-]c1c[n+](Cc2ccccc2)no1)Nc1cccc(C(F)(F)F)c1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC(C)(C)OC(=O)NC1CCCN(C(=O)Cc2ccccc2)C1",
        "CN1CCC(NC(=O)OC(C)(C)C)CC1",
    ],
)
def test_acyl_substituted_saturated_ring_carbamates_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_cyclic_n_carbamate_carboxymethylidene_parent_roundtrips_with_opsin():
    input_smiles = "CC(C)(C)OC(=O)N1CCC(=CC(=O)O)CC1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_acylated_piperidine_substituent_uses_opsin_stable_yl_suffix():
    input_smiles = "C[C@H]1CCCCN1C(=O)Cc1ccccc1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "c1ccc2c(c1)nnn2O[P+](N1CCCC1)(N1CCCC1)N1CCCC1",
        "CN(C)[P+](On1nnc2ccccc21)(N(C)C)N(C)C",
        "CC(C)(C)N=P(N1CCCC1)(N1CCCC1)N1CCCC1",
        "CC(C)(C)N=P(N(C)C)(N(C)C)N(C)C",
    ],
)
def test_amino_phosphonium_and_phosphanimine_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "[N]=O",
        "O=[PH3]",
        "O=[Si]",
    ],
)
def test_small_element_oxide_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCC(=O)C(=O)[O-]",
        "CC(C)C(=O)C(=O)[O-]",
        "CC(CC(=O)C(=O)[O-])C",
        "O=C([O-])CCC(=O)C(=O)[O-]",
    ],
)
def test_alpha_keto_carboxylate_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CN(C)CCCN=C=N",
        "C(=NCCN1CCOCC1)=NC1CCCCC1",
        "C1CCCCC1N=C=NC1CCCCC1",
    ],
)
def test_carbodiimide_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "c1ccccc1N=CN(C)CC",
        "CCCOCC(C)Oc1nc(C)c(N=CN(C)CC)cc1Br",
    ],
)
def test_formamidine_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CN(C)C=[NH2+]",
        "[NH2+]=CNCCc1ccccc1",
        "Cc1cc(Oc2ccccc2)ccc1NC=[NH2+]",
        "NC(=[NH2+])Nc1ccccc1",
    ],
)
def test_charged_iminium_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "COC(=O)c1ccccc1C(=O)N=C=S",
        "Cc1nc(C)c(C(=O)N=C=S)o1",
        "O=C(N=C=S)c1nn(C)c(C)c1",
    ],
)
def test_acyl_isothiocyanate_imino_thione_candidates_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "P(=S)([O-])([O-])[O-]",
        "OP(O)(O)=S",
        "NP([O-])([O-])=S",
        "COP([O-])([O-])=S",
        "COP(=S)(OC)OC",
        "CCO[PH]([O-])=S",
        "CCCOS(=O)([O-])=S",
        "CO[SH]=S",
        "CO[SH](=O)=S",
        "Cc1cccc(OP(=S)(Oc2cccc(C)c2C)Oc2cccc(C)c2C)c1C",
    ],
)
def test_thiooxo_ester_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "NP(N)(N)=S",
        "[NH-]P([NH-])([NH-])=S",
        "C[N-]P([NH-])([NH-])=S",
        "CCCC[N-]P([NH-])([NH-])=S",
        "CCCCNP(N)(N)=S",
        "CCN(CC)P(N)(N)=S",
        "S=P(N1CC1)(N1CC1)N1CC1",
    ],
)
def test_phosphorothioic_triamide_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "[O-]P(=S)([S-])[S-]",
        "S=[PH]([S-])[S-]",
        "CCCCCCCCSP([O-])([O-])=S",
        "CC(C)SP(O)(O)=S",
        "CCCCCCCCCCCCCSP(=S)([S-])[S-]",
    ],
)
def test_dithiophosphate_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCOP([O-])(=S)[S-]",
        "CCOP(=S)([S-])OCC",
        "CCOP(N)([O-])=S",
        "COP(=S)(Cl)Cl",
        "Cc1ccc(OP(=S)([S-])Oc2cccc(C)c2)cc1",
        "O=c1ccn([C@H]2C[C@H](OP(O)(=S)S)[C@@H](CO)O2)c(=O)[nH]1",
    ],
)
def test_o_substituted_phosphorothioate_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "N=S(=O)=O",
        "N=S=O",
        "N=[SH2]=O",
        "N=S(F)F",
        "N=S(=O)(F)F",
        "N=S(=O)(F)Cl",
        "N=S(=O)([O-])[O-]",
        "N#CN=S(=O)=O",
        "C[SH](=N)=O",
        "CC[SH](=N)=O",
        "N=[SH](=O)c1ccccc1",
    ],
)
def test_small_sulfur_imide_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC(C)(C)OC(=O)N[SH](=O)=O",
        "CCCCCCCCN[SH](=O)=O",
        "NCCN[SH](=O)=O",
        "O=[SH](=O)NC(F)(F)F",
        "CN(CN)[SH](=O)=O",
        "Cc1cc(N[SH](=O)=O)no1",
    ],
)
def test_n_substituted_sulfonamide_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[SH]=O",
        "C[SH3]=O",
        "C[SH2](C)=O",
        "CC[SH2](=O)CC",
        "C[SH2](C)=N",
        "C[SH3]=N",
        "C[SH2](=O)c1ccccc1",
    ],
)
def test_small_lambda_sulfur_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "COc1cc2nncc(N3CCC4(CC3)CCN([SH2](=O)C=N)C4)c2cc1OC",
        "COc1cc2ncnc(N3CCC4(CCN([SH2](=O)C=N)CC4)C3)c2cc1OC",
        "COc1cc2ncnc(N3CCC4(CC3)CCN([SH2](=O)C=N)CC4)c2cc1OC",
    ],
)
def test_spiro_lambda_sulfanone_substituent_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C1[N]CC12COC2",
        "O=C1[N]CCC2(CC2)N1",
        "NC(=O)C1CC2(CC[SH4]2)C1",
    ],
)
def test_polycyclic_lambda_spiro_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CN1COCN(Cc2cnc(Cl)s2)C1=N[N+](=O)[O-]",
        "O=[N+]([O-])N=C1NCCN1Cc1ccc(Cl)nc1",
        "O=[N+]([O-])N=C1NCCN1Cc1cnc(Cl)s1",
    ],
)
def test_cyclic_nitroimine_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "c1ccc(-c2ccc[o+]c2-c2ccccc2)cc1",
        "c1ccc(-c2cccc(-c3ccccc3)[o+]2)cc1",
    ],
)
def test_intrinsic_pyrylium_ring_charge_not_duplicated(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CN(C(=N[N+](=O)[O-])N)C",
        "CNC(=N[N+](=O)[O-])NC[C@H]1CCOC1",
        "CNC(=N[N+](=O)[O-])NCc1cnc(Cl)s1",
    ],
)
def test_nitroguanidine_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=c1ccnccn1",
        "O=c1cncccn1",
    ],
)
def test_diazepinone_retained_parent_templates_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_bis_sulfonium_sulfide_retained_parent_roundtrips_with_opsin():
    input_smiles = "c1ccc([S+](c2ccccc2)c2ccc(Sc3ccc([S+](c4ccccc4)c4ccccc4)cc3)cc2)cc1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "c1ccc(O[S+](Oc2ccccc2)Oc2ccccc2)cc1",
        "CN(C)[S+](N(C)C)N(C)C",
        "CCN(CC)[S+](F)F",
    ],
)
def test_hetero_ligated_sulfanium_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CN(C)C(=S)[SH-]Cc1ccccc1",
        "C[SH-]C(=S)NCCc1c[nH]c2ccccc12",
        "CC[SH-]C(=S)NC(=N)c1ccc(Cl)cc1",
    ],
)
def test_dithiocarbamate_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[SH-]C(=S)OC(C)(C)C",
        "CCOC(=S)[SH-]C(C)C(=O)OC",
        "C[SH-]C(=S)Oc1cnc(Br)cc1C",
        "CCCCCC[SH-]C(=S)[S-]",
    ],
)
def test_dithiocarbonate_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_diacylperoxy_dimethylhexane_retained_parent_roundtrips_with_opsin():
    input_smiles = "Cc1cccc(C(=O)OOC(C)(C)CCC(C)(C)OOC(=O)c2cccc(C)c2)c1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_bis_diethylaminobenzylidene_cyclopentanone_roundtrips_with_opsin():
    input_smiles = "CCN(CC)c1ccc(C=C2CCC(=Cc3ccc(N(CC)CC)cc3)C2=O)cc1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_sulfonylamide_retained_parent_roundtrips_with_opsin():
    input_smiles = "[N-]=S(=O)=O"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "NC(=NS(=O)(=O)c1ccccc1)c1ccccc1",
        "CC(=NS(=O)(=O)c1ccc(C)cc1)c1ccc(C#N)cc1",
        "CC(=NNS(=O)(=O)c1ccc(C)cc1)C1CN(C(=O)OC(C)(C)C)C1",
        "CC(C)=C/C(=C\\C=NS(=O)(=O)c1ccc([N+](=O)[O-])cc1)N1CCCC1",
    ],
)
def test_imino_sulfonamide_and_sulfonohydrazide_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCCCCN[SH](=N)=O",
        "CS(N)(=O)=NC(=O)Nc1c2c(cc3c1CCC3)CCC2",
        "Cc1nn(C)cc1S(N)(=O)=NC(=O)Nc1c(C(C)C)cc(F)cc1C(C)C",
    ],
)
def test_sulfonimidamide_parent_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "NC(=O)C1CC[SH2]CC1",
        "O=C(Nc1ccc(Br)cc1)c1cnc[se]1",
        "CN(C)C(=O)c1cccnc1",
    ],
)
def test_ring_carboxamide_parent_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_oxazinone_retained_parent_roundtrips_with_opsin():
    input_smiles = "CN(C)c1nc(C(F)(F)F)cc(=O)o1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCC[SH-]C(=S)N(CCCC)CN(CCCC)C(=S)[SH-]CCCC",
        "CC(C)(C)c1cc(CCC(=O)[N-]CCCCCC[N-]C(=O)CCc2cc(C(C)(C)C)c(O)c(C(C)(C)C)c2)cc(C(C)(C)C)c1O",
    ],
)
def test_retained_bis_functional_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "COc1ccc(C(C#N)=NOS(=O)(=O)c2ccc(C)cc2)cc1",
        "Cc1ccc(S(=O)(=O)ON=C2SC=CC2=C(C#N)c2ccccc2C)cc1",
        "CCCS(=O)(=O)ON=C1SC=CC1=C(C#N)c1ccccc1C",
        "Cc1ccc(S(=O)(=O)ON=C2C=CC(=C(C#N)c3ccccc3C)S2)cc1",
    ],
)
def test_oxime_sulfonate_nitrile_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C(=C1\\Cc2ccccc2O1)/c1ccccc1",
        "O=C1Nc2ccc(F)cc2/C1=C/c1ccccc1",
        "C1=Nc2ccccc2/C1=C\\c1ccccc1",
        "OC/C=C1\\CCc2ccccc21",
    ],
)
def test_exocyclic_ring_alkene_substituents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "N#CC(=C1C(=C(C#N)c2c(F)c(F)c(C#N)c(F)c2F)C1=C(C#N)c1c(F)c(F)c(C#N)c(F)c1F)c1c(F)c(F)c(C#N)c(F)c1F",
        "N#CC(=C1C(=C(C#N)c2c(F)c(F)c(F)c(F)c2F)C1=C(C#N)c1c(F)c(F)c(F)c(F)c1F)c1c(F)c(F)c(F)c(F)c1F",
    ],
)
def test_cyclopropane_triylidene_nitrile_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=C1CC[C@H]2[C@H]3Cc4ccccc4[C@@]2(CCN3)C1",
        "O=C1CCCc2ccccc2N1",
    ],
)
def test_saturated_fused_lactam_suffixes_roundtrip_without_indicated_h(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C1=C2CCCNCc3cccc(c3)CNC(=N2)N=[SH]1",
        "O=C1NCCCCCCC2=NN=C(C2)c2cccc1n2",
        "O=C1NCCCc2ccnc(n2)NCS2=CC1=CC=C2",
    ],
)
def test_unsaturated_secondary_bridge_tricyclic_parents_roundtrip(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "c1cc2ccc3nnnc4[nH]c(c1)c2c34",
        "c1cc2nccn3c4c(cnc(c1)c24)=NC3",
        "C1=CCC23CCNC(=Cc4cc5cccnc5cc42)C3=C1",
        "C1=c2ccc([nH]2)=C2COc3nccc4ncnc(c34)N12",
    ],
)
def test_unsaturated_secondary_bridge_larger_polycycles_roundtrip(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CN1C[C@H]2C[C@@H]1CN2c1cccc2[nH]cnc12",
        "Cc1nc2c(N3C[C@H]4C[C@@H]3CN4C(=O)OC(C)(C)C)c(N)ccc2n1C",
    ],
)
def test_generated_fused_parent_can_beat_locanted_bicyclic_stereo_fallback(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCCCCCCCCCCCC=[NH+][O-]",
        "c1ccccc1B(Cl)Cl",
        "c1ccccc1P([O-])[O-]",
        "O=[P+]([O-])c1ccccc1",
        "CCO[P+]([O-])=NC",
        "CCO[P+]([O-])=N[C@@H](C)C(=O)[O-]",
        "CCCC[P+]([O-])=C(O)C[N+](C)(C)C",
        "O=[SH2]1CCC1",
        "O=[SH2]1CCNCC1",
        "O=S1(=O)CCNCC1",
    ],
)
def test_charged_pnictogen_and_boryl_functional_families_roundtrip(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCCCCCC/C=C\\CCCCCCCC[NH-]",
        "C[C@@H](O)[C@H](N)C([NH-])=O",
    ],
)
def test_direct_fragment_names_preserve_stereochemistry(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCCCCCC/C=C\\CCCCCCCC[N-]CCCCCCCC/C=C\\CCCCCCCC",
        "C[C@@H]([N-][C@H](C)c1ccccc1)c1ccccc1",
    ],
)
def test_repeated_onium_substituents_use_global_stereo_roundtrip(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "c1c[nH][nH+]c1",
        "c1c[nH+]c[nH]1",
        "C1COCC[N]1",
        "O=C1CCCC[N]1",
        "[NH2+]1CC[NH2+]CC1",
        "C[NH+]1CC[NH+](CC1)C",
        "[NH2+]1C[NH2+]CC1",
        "[NH+]1=CC=[NH+]C=C1",
    ],
)
def test_ring_parent_charge_and_lambda2_forms_roundtrip(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Cc1cc(C(=O)C2=CS[C-](S)N2)cc(=O)n1C",
        "O=C1/C(=C(\\Nc2ccccc2)c2cccs2)c2ccccc2C1c1ccccc1",
        "COc1cccc(-c2ccc(-c3cc4c(cc3Cl)NC(=O)C4=C(O)c3cc(C4CC4)no3)cc2)c1O",
        "CCOC(=O)C1CC2(NC(=O)OCc3ccccc3)C=CC1C1C=CC12",
        "Cc1cc(NC(=O)C(N)Cc2ccccc2)ccc1-c1ccnc2[nH]ccc12",
        "CCN(C)S(=O)(=O)Nc1cccc(C(=O)c2c[nH]c3ncc(-c4cocn4)cc23)c1F",
        "CCN(C)S(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(C=O)cc4)cc23)c1F",
        "C=CC(=O)Nc1cnc2c(c1)c(C#Cc1sc3ccccc3c1C)cn2C",
        "C=CC(=O)Nc1cnc2c(c1)c(C#Cc1cc3cc(Cl)ccc3s1)cn2C",
        "C=CC(=O)Nc1cnc2c(c1)c(C#Cc1cc3cc(C(F)(F)F)ccc3s1)cn2C",
    ],
)
def test_structured_contextual_parent_selection_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[C@H](Nc1ccc2c(c1)OCCn1cc(N3C(=O)OC[C@H]3C(F)F)nc1-2)C(N)=O",
        "C[C@@H](Nc1ccc2c(c1)OCCn1cc(N3C(=O)OC[C@H]3CF)nc1-2)C(N)=O",
        "C[C@H](Nc1ccc2c(c1)OCCn1cc(N3C(=C=O)CNC[C@H]3C(F)F)nc1-2)C(N)=O",
    ],
)
def test_large_structured_kekulized_parents_beat_chain_decompositions(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCOC(=O)[C@H]1[C@@H](C2COC2)[N@]1C(c1ccccc1)c1ccccc1",
        "CCOC(=O)[C@H]1[C@@H](C2COC2)[N@@]1C(c1ccccc1)c1ccccc1",
    ],
)
def test_small_late_single_ring_parent_defers_to_early_six_member_parent(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[C@@H](Oc1ccccc(=O)c1O)C1CC1",
        "COc1ccc([C@@H]2CN([C@H]3CC[C@H](OC)CC3)C[C@@H]2F)cc1",
        "NC(=O)c1ccc(OC[C@H]2CC[C@H](C(=O)N3OCC[C@H]3c3cnccn3)CC2)nc1",
        "COC(=O)N1CCC[C@H](c2cccc(O)n2)[C@@H]1CO[C@H]1CC[C@@H](c2ccccc2)CC1",
    ],
)
def test_late_nonaromatic_single_ring_parent_defers_to_earlier_aromatic_parent(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=C(/C=C/c1ccccc1NC(=O)C(c1ccccc1)C1CCCC1)NO",
        "Cc1ncc(C)c(N[C@@H](C)c2ccc(C(=O)O/N=C(\\N)C3CC3)cc2)n1",
        "CC(C)[C@H](NC(=O)OC(C)(C)C)C(=O)N[C@@H](C)C(=O)Nc1ccc(COC(=O)N(C)[C@@H]2C[C@@H]2COC(=O)Oc2ccc([N+](=O)[O-])cc2)cc1",
    ],
)
def test_small_functional_single_ring_parent_ranker_switches_remain_available(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC(C)C[C@H](NC(=O)C(F)(F)F)C(=O)N1C2CCC(C1C(=O)NC(C#N)CC1CCNC1=O)C(F)(F)C2",
        "CC(C)C[C@@H](NC(=O)C(F)(F)F)C(=O)N1[C@@H]2CC[C@H]([C@@H]1C(=O)N[C@H](C#N)C[C@H]1CCNC1=O)C(F)(F)C2",
        "Cc1cc2cccc(C(=O)N3[C@@H]4CC[C@H]([C@@H]3C(=O)N[C@H](C#N)C[C@H]3CCNC3=O)C(F)(F)C4)c2[nH]1",
        "CCOCOc1cc(C(F)(F)F)cc(C)c1-c1ccc2cn(CC34CNC(=O)C3C4)nc2n1",
        "C[C@H](c1cnc(N2C[C@H]3C[C@H]3C2=O)nc1)n1cc(C(=O)N[C@H]2C[C@@H](c3cc(Cl)ccc3C#N)C2)c(C#N)n1",
    ],
)
def test_compact_bicyclic_parent_ranker_switches_beat_long_acyclic_decompositions(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C=C[C@H]1CN2CCC1CC2",
        "C[N+]12CC[C@@H](COCCOCCOCC3C4CC(C3)C4)C(C)(C)[C@H]1CC2",
        "COC[C@@]1(COP(N)(=O)OC[C@]2(COC)C(=O)C3CCN2CC3)C(=O)C2CCN1CC2",
    ],
)
def test_retained_quinuclidin_parent_beats_generic_bicyclic_aliases(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC1(C)CN2C(=O)CC2S1",
        "CC(=O)/C=C1/C(=O)N2[C@@H]1SC(C)(C)[C@@H]2C(=O)OC(c1ccccc1)c1ccccc1",
        "CC1(COC(=O)NNC(=O)c2ccc(O)c(O)c2)C(C(=O)OC(c2ccccc2)c2ccccc2)N2C(=O)CC2S1(=O)=O",
    ],
)
def test_retained_azabicyclo320_heptane_parent_beats_decompositions(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCN(C)S(=O)(=O)Nc1ccc(F)c(C(=O)n2ccc3cc(-c4ccc(N5CCNCC5)cc4)cnc32)c1F",
        "CCN(C)S(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(OC5CCNCC5)cc4)cc23)c1F",
    ],
)
def test_large_acyclic_parent_ranker_switches_beat_generated_fused_aliases(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=c1occoc2cccc(c2)oc(=O)c2cccc1c2",
        "CC(C)(C)c1ccc2c(C(C)(C)C)c1Cc1c(C(C)(C)C)ccc(c1C(C)(C)C)OOP(=O)([O-])OO2",
    ],
)
def test_early_kekulized_polycycle_ranker_switches_beat_late_orientations(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C(#Cn1cnc2ccccc21)c1ccccc1",
        "CN(C)N(C)C(=O)c1ccc2nc[nH]c2c1",
        "CN(C(=O)c1ccc2nc[nH]c2c1)c1ccccc1",
        "NC(=O)c1cccc2[nH]c(-c3ccc(C(=O)N4CCN(c5nc6ccccc6[nH]5)CC4)cc3)nc12",
        "C[C@@H](C(=O)N(C)c1ccc2[nH]c(-c3n[nH]c4c3C[C@@H]3C[C@]3(C)C4)nc2c1)C1CCOCC1",
        "CCn1cc2[nH]c(=O)c(-c3nc4ccc(N5CCOCC5)cc4[nH]3)c(NC(C)c3cocn3)c2n1",
        "O=C(O)c1ccc2[nH]c(-c3cccc(O)c3)nc2c1",
        "Cn1nccc1C(=O)NC(c1nc2c(F)cccc2[nH]1)C1CCCCCCC1",
        "CCC(Nc1c(-c2nc3ccc(N4CCOCC4)cc3[nH]2)c(=O)[nH]c2cn(C)nc12)c1ncccn1",
        "Cc1cc2nc(C(=O)NCCCn3cnc4ccccc43)c(=O)n(C[C@H](O)[C@H](O)[C@H](O)CO)c2cc1C",
    ],
)
def test_benzimidazole_template_orientation_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "N#Cc1c(C2CC2)[nH]c2c(-c3nc4cc(C5CCNCC5)ccc4[nH]3)c(=O)[nH]cc12",
        "N#Cc1c[nH]c2c(-c3nc4c(OCC(F)F)c(C5CCNCC5)ccc4[nH]3)c(=O)[nH]cc12",
        "CN1CCC(c2ccc3[nH]c(-c4c(=O)[nH]cc5c(C#N)c[nH]c45)nc3c2OCC(F)F)CC1",
        "N#Cc1c(C2CC2)[nH]c2c(-c3nc4c(OCC(F)F)c(C5CCNCC5)ccc4[nH]3)c(=O)[nH]cc12",
    ],
)
def test_retained_benzimidazole_fragments_defer_to_generated_heterofused_parent(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_same_parent_generated_benzimidazole_alias_does_not_displace_retained_parent():
    input_smiles = (
        "COc1cc2nc(-c3cccc(Cl)c3-c3ccc(C(=O)N[C@H](c4ccccc4)C4CC4)cc3C(=O)O)"
        "[nH]c2cc1F"
    )
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "OC(=CC1=Nc2cc(N3CCOCC3)ccc2C2=NCCN12)c1ccc2[nH]cnc2c1",
        "COCC(=O)c1ccc2[nH]c([C@H](COC(C)(C)C(F)(F)F)NC(=O)OC(C)(C)C)nc2c1",
        "CC(C)CC(N)c1ccc2[nH]c([C@@H](NC(=O)c3ccnn3C)C3CCC(F)(F)CC3)nc2c1",
    ],
)
def test_benzimidazole_low_complexity_alternatives_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "[B]c1cccc2cc3ccccc3cc12",
        "[N]c1ccc2ccc3cccc4ccc1c2c34",
        "[Si]c1cccc2nc3ccccc3cc12",
        "[B]c1nc2ccccc2o1",
    ],
)
def test_fused_parent_element_radical_substituents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "[C]c1ccccc1",
        "[B]c1ccccn1",
        "[S]c1nccs1",
        "[Si]c1ccc[nH]1",
    ],
)
def test_single_ring_element_radical_substituents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "[B]c1ccccc1C",
        "[B]c1ccc(Br)cc1",
        "Cc1ccccc1[P]",
        "[S]c1ccc(Br)cc1",
    ],
)
def test_substituted_single_ring_element_radicals_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=c1[nH]c(=O)c2nc[nH]c2[nH]1",
    ],
)
def test_purine_retained_parent_tautomer_aliases_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Nc1nc2[nH]cnc2c(=O)[nH]1",
    ],
)
def test_chalcogen_purine_alias_alternatives_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCn1ccc(S(N)(=O)=NC(=O)Nc2c3c(nc4c2CCC42CC2)CCC3)n1",
        "O=C(Cn1c2c(c(=O)n3nc(Br)nc13)C1(CCNCC1)CC2)Nc1ccc(C(F)(F)F)cc1Cl",
        "O=C(Cn1c2c(c(=O)n3nc(-c4ccccc4)c(F)c13)C1(CC2)CCN(C(=O)c2ncccc2O)CC1)Nc1ccc(C(F)(F)F)cc1Cl",
    ],
)
def test_generated_retained_spiro_scaffold_templates_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[C@H](C(=O)O)[C@H](c1ccc2c(c1)O[C@@H](C1CCNCC1)CC2)C1CC1",
        "C[C@H](C(=O)O)[C@H](c1ccc2c(c1)N[C@@H](C1CCN(Cc3cc(C(F)(F)F)ccc3C(F)(F)F)CC1)CC2)C1CC1",
    ],
)
def test_retained_hydride_fused_parent_stereo_order_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_spiro_tetraline_component_stereo_order_stays_roundtripping():
    input_smiles = (
        "C=CC[C@@H](O)[C@H](C)S(=O)(=O)NC(=O)c1ccc2c(c1)"
        "N(C[C@@H]1CC[C@H]1[C@H](/C=C/CCC)OC)C[C@@]1(CCCc3cc(Cl)ccc31)CO2"
    )
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "c1ccc2c(c1)CC1(CCNCC1)C21CC1",
        "CC(C)(C)[S@@](=O)N[C@@H]1c2ccccc2C2(CC2)C12CCNCC2",
    ],
)
def test_component_dispiro_fused_middle_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Fc1ccc2c(c1)COC21CCC2(CC1)OCCO2",
        "CC1=Nc2cccc(Br)c2C12CCC1(CC2)OCCO1",
    ],
)
def test_component_dispiro_complete_graph_candidates_beat_nested_spiro_aliases(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC(C)(C)N1CCC2(CC1)Cc1ccc(C#N)cc1C2N[S@](=O)C(C)(C)C",
        "CC(C)(C)N1CCC2(CC1)Cc1ccc(C#N)cc1C2N[S@@](=O)C(C)(C)C",
    ],
)
def test_sulfinyl_branch_stereo_stays_on_sulfur(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "COc1ccc(C(=O)C[C@H](c2ccc(OC)cc2)[C@@H]2COC3(CCCCC3)O2)cc1",
        "N#Cc1ccc2ncn(CC3(O)CCCC4(C3)OCCO4)c2c1",
        "CC1(C)CC(C)(Cn2cnc3ccc(C#N)cc32)CC2(C1)OCCO2",
        "Cn1c(=O)n(C2CCC(=O)NC2=O)c2ccc(C3CCC4(CC3)OCCO4)cc21",
    ],
)
def test_ketal_spiro_substituents_use_systematic_dioxaspiro_names(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C12=NC=C(N1)C=C1C=CC(=N1)C=C1C=CC(N1)=CC=1C=CC(N1)=C2",
        "C12=NN=C(N1)C=C1C=CC(=N1)C=C1C=CC(N1)=CC=1C=CC(N1)=C2",
        "S=C1C2=CC3=CC=C(N3)C=C3C=CC(C=C4C=CC(=CC(=C1)N2)N4)=N3",
        "[C-]1=[C-]C=C2C=CC=C3C4=CC=CC5=CC=CC(C1=C23)=C45",
        "C1=CC2=CC=CC3=CC=4C5(C=CC=C1C5=C32)CC=CC4",
        "C1=C2C=CC34C(C=CC5=CC=C(C=C1)C2=C53)=CC=CC4",
        "C1=CC2=C3C(CC4=CC=CC5=CC=C1C3=C45)=CC=C2",
        "C1=CC=C2C3=C(C=4C=C5C(=C6C=CC1=C2C64)C=CC=C5)C=CC=C3",
        "C1=CC=CC2=[NH+]C3=CC=CC=C3C(=C12)C=1C2=CC=CC=C2[NH+]=C2C=CC=CC12",
        "C1=CC(=CC=2C3=CC=CC=C3NC12)B([O-])[O-]",
    ],
)
def test_high_impact_large_fused_retained_templates_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC(=O)N1CCC[C@H](NS(=O)(=O)C2CCCCC2)[C@@H]1CO[C@H]1CC[C@@H](C(C)C)CC1",
        "NC[C@H]1CC[C@H](C(F)(F)c2cc(Cl)nc(N3CCN(S(=O)(=O)c4ccc(N5C[C@H](N)CC5=O)cc4)CC3)c2)CC1",
        "CS(=O)(=O)[C@H]1CCC[C@H](CN=C=S)C1",
    ],
)
def test_single_ring_stereo_prefix_alternatives_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCCS(=O)[O-]",
        "CC(C)(C)c1ccc(S(=O)[O-])cc1",
        "O=C([O-])C(O)S(=O)[O-]",
    ],
)
def test_sulfinate_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCC[IH+]",
        "Cc1ccc([IH+])cc1",
        "CCCC[PH3+]",
        "Cc1cccc([PH3+])c1",
        "O=C(O)[PH3+]",
    ],
)
def test_hydride_onium_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[S+](C)(=O)CCl",
        "COc1ccc([S+](C)(C)=O)cc1",
        "O=[S+](c1ccccc1)(c1ccccc1)c1ccco1",
    ],
)
def test_multiligand_oxosulfanium_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "N#CC#[N+][O-]",
        "Cc1cc(O)cc(C)c1C#[N+][O-]",
        "[O-][N+]#Cc1cccc2ccccc12",
        "[O-][N+]#Cc1c(OCC2CO2)ccc2ccccc12",
        "[O-][N+]#Cc1ccc(N2c3ccccc3Sc3ccccc32)cc1",
        "[O-][N+]#CC=CC=Cc1ccccc1",
    ],
)
def test_nitrile_oxide_retained_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "N=C(N)[NH3+]",
    ],
)
def test_charged_guanidine_parent_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "NCCCC[C@H](NC1(CO)O[C@H](CO)[C@@H](O)[C@@H]1O)C(=O)O",
    ],
)
def test_nonring_linked_carbohydrate_branch_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CC1=C(CCC(=O)O)c2cc3[nH]c(cc4[nH]c(cc5nc(cc1n2)C(C)=C5CCC(=O)O)c(CCC(=O)O)c4C)c(C)c3CCC(=O)O",
        "C=Cc1c(C)c2cc3c(C=C)c(C)c(cc4nc(cc5nc(cc1[nH]2)C(C)=C5CCC(=O)O)C(CCC(=O)O)=C4C)n3C",
    ],
)
def test_large_kekulized_bridge_polycycle_orientation_roundtrips(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=C1Oc2ccccc2-c2ccc(cc2)-c2ccccc2OC(=O)c2ccc1cc2C(=O)[O-]",
        "O=C(O)c1ccc2c3ccc4c5c(ccc(c6ccc(C(=O)O)c1c26)c53)C(=O)OC4=O",
    ],
)
def test_large_hetero_kekulized_bridge_polycycle_orientation_roundtrips(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "O=C(O)Cc1ccc(-n2c(=O)c3ccc4oc5ccc(C(F)(F)F)cc5c5ccc(c2=O)c3c45)cc1",
        "CC(C)(CN1CCCCCCc2ccccc2-c2nc(sc2I)NS(=O)(=O)c2cccc1n2)C(=O)O",
    ],
)
def test_large_hetero_kekulized_bridge_order_preserves_lower_unsaturation_controls(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "c1cc2cc(c1)Oc1ccc(cc1)C[C@@H]1NCCc3ccc(cc31)Oc1cccc3c1[C@@H](C2)NCC3",
        "C[C@@H]1COc2c(Br)c(F)cc3ncc4c(c23)n1c(=O)n4C",
        "Cn1c(=O)n2c3c4c(c(Br)c(F)cc4ncc31)OC[C@H]2Cc1ccccc1",
    ],
)
def test_stereo_kekulized_bridge_polycycles_use_opsin_stable_bridge_selection(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[C@@H]1COc2ccc3[nH]nc(c3c2)-c2cnn(c2)CCOCCO1",
        "C[C@@H]1CCOC[C@@H](C)Oc2cncc(c2)-c2n[nH]c3ccc(cc23)O1",
        "C[C@H]1COc2nccc(n2)-c2nn(C3CCCCO3)c3ccc(cc23)O[C@@H](C)CCO1",
    ],
)
def test_stereo_hetero_kekulized_macrocycles_use_stable_bridge_numbering(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_component_spiro_preserves_polycyclic_component_graph_order():
    input_smiles = "O=C1Oc2ccc(cc2)C2(CCCCC2)c2ccc(cc2)O1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_component_spiro_prefers_stable_polycyclic_component_numbering():
    input_smiles = "c1cc2c3c(c1)ncc1ncn(c13)C1(CCC1)CO2"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def test_pseudoasymmetric_bicyclo_bridgeheads_render_trans_relative_stereo():
    input_smiles = "C1CC[C@H]2CCCC[C@@H]2C1"
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C=P(c1ccccc1)(c1ccccc1)c1ccccc1",
        "N#CC=P(c1ccccc1)(c1ccccc1)c1ccccc1",
        "COC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1",
        "CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1",
        "CC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1",
        "C(c1ccccc1)=P(c1ccccc1)(c1ccccc1)c1ccccc1",
        "CC(C)(C)OC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1",
        "CON(C)C(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1",
    ],
)
def test_phosphorane_ylides_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Cc1ccc2c(n1)CCN(c1nn3c(=O)ccnc3cc1C)C2",
        "Cc1ccc(-c2cnc3c(c2)CN(c2nn4c(=O)cc(C)nc4cc2C)CC3)cc1",
        "O=c1ccc2ccc(OCCCn3c(=O)cnc4ccccc43)cc2o1",
        "Cc1ccc2oc(=O)cc(CSc3nc4ccccc4o3)c2c1",
    ],
)
def test_high_signal_generated_topology_fused_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCOc1ccc(N2C[C@H]3CC(C)(C)[C@@H]2CNC3=O)cc1",
        "COC(=O)N1CC2C=CC1C(c1ccccc1)C2",
        "O=C([O-])C(=O)c1ccc([N+](=O)[O-])cc1C1CCN=C2NCCCN21",
    ],
)
def test_compact_functional_bicyclic_parent_alternatives_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[C@](O)(c1nc2c(Cl)nc(Cl)nc2[nH]1)C1CC1",
        "C[C@@H](Oc1ccc(-c2nc3c(Cl)ncnc3[nH]2)cc1)c1ccccc1",
    ],
)
def test_generated_purine_parent_alternatives_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Cc1cccc(-c2nc(-c3n[nH]c4ccncc34)c3nc[nH]c3n2)n1",
        "CO[C@H](C)CN(CCCCc1ccc2c(n1)NCCC2)CC[C@H](Nc1ncnc2[nH]cnc12)C(=O)O",
    ],
)
def test_generated_purine_orientation_parent_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "c1ccc2oc(C3=NCCc4[nH]cnc43)nc2c1",
        "c1ccc2sc(C3=NCCc4[nH]cnc43)nc2c1",
        "c1ccc2[nH]c(C3=NCCc4[nH]cnc43)nc2c1",
    ],
)
def test_generated_benzazole_parent_alternatives_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "c1ccc(-c2cnnc3[nH]cnc23)cc1",
        "NC(=O)c1cc2nc[nH]c2s1",
        "NC(=O)c1ccc2ncc3cn[nH]c3c2c1",
        "NC(=O)c1ccc2ncc3[nH]ncc3c2c1",
        "CSc1nc2cc(-c3ccc(-c4ccccc4)cc3)c3ccccc3c2[nH]1",
    ],
)
def test_preferred_generated_fused_template_alternatives_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "NN=c1nc[nH]c2c1cnn2[C@@H]1O[C@H]([C@H](O)C#CC(F)(F)F)[C@@H](O)[C@H]1O",
        "CO/N=c1/nc[nH]c2c1cnn2[C@@H]1C[C@H]([C@H](O)c2ccc(Cl)cc2)[C@@H](O)[C@H]1O",
    ],
)
def test_exocyclic_imino_pyrazolopyrimidine_orientation_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "C[C@@](O)(c1ccc(Cl)cc1)[C@H]1C[C@@H](n2cnc3/c(=N\\N)nc[nH]c32)[C@H](O)[C@@H]1O",
        "CC(=O)N/N=c1/nc[nH]c2c1ncn2[C@@H]1C[C@H]([C@H](O)c2ccc(Cl)cc2)[C@@H](O)[C@H]1O",
    ],
)
def test_exocyclic_imino_imidazopyrimidine_orientation_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "N=c1[nH]c2cncnc2[nH]1",
        "CN(Cc1cnc2nc(=N)[nH]c(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",
    ],
)
def test_fused_imine_suffix_parents_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Oc1ccc2c(c1)oc1cc(O)ccc12",
    ],
)
def test_oxafluoren_skeletal_replacement_alternatives_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCOP([O-])[O-]",
        "CCCCCCCCCCCCCOP(OCCCCCCCCCCCCC)OP([O-])[O-]",
        "C#C[PH]([O-])(OCc1ccc(-c2cccc([N+](=O)[O-])c2[N+](=O)[O-])cc1)OCc1ccc(-c2cccc([N+](=O)[O-])c2[N+](=O)[O-])cc1",
    ],
)
def test_phosphite_ester_parent_survives_fact_ranker_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "CCCCOP(=O)(O)OC[C@H]1O[C@@H](n2cnc3cnc(N)nc32)C[C@@H]1O",
        "CCCCOP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(=O)[nH]c(N)nc32)C[C@@H]1O",
    ],
)
def test_purine_nucleotide_phosphate_acyclic_parent_roundtrips_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H](C[C@@H](O1)N1C=NC=2C(=O)NC(N)=NC12)O",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C=NC=2C(=O)NC(N)=NC12)O)O",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C=NC=2C(O)=NC=NC12)O)O",
    ],
)
def test_standard_nucleoside_tetraphosphate_templates_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    "input_smiles",
    [
        "Cc1cc(=O)oc2cc(OCC(O)CO)ccc12",
        "[NH-]OCC(O)CO",
    ],
)
def test_polyhydroxy_alkyl_branches_roundtrip_with_opsin(input_smiles):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


def _opsin_parse_name(name: str) -> str:
    opsin_jar = Path(__file__).resolve().parents[2] / "benchmark" / "opsin-cli-2.9.0-jar-with-dependencies.jar"
    if not opsin_jar.exists():
        pytest.skip("OPSIN jar is not available")
    java_executable = _java_executable()
    if not java_executable:
        pytest.skip("Java runtime is not available")

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        names_file = temp_dir / "names.txt"
        output_file = temp_dir / "opsin.smi"
        names_file.write_text(name + "\n")
        completed = subprocess.run(
            [java_executable, "-jar", str(opsin_jar), "-o", "smi", str(names_file), str(output_file)],
            check=False,
            capture_output=True,
            text=True,
        )
        if "Unable to locate a Java Runtime" in completed.stderr:
            pytest.skip("Java runtime is not available")
        assert completed.returncode == 0, completed.stderr or completed.stdout
        return output_file.read_text().strip()


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _java_executable() -> str:
    homebrew_java = Path("/opt/homebrew/opt/openjdk/bin/java")
    if homebrew_java.exists():
        return str(homebrew_java)
    return which("java") or ""
