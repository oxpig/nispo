from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from shutil import which

import pytest
from rdkit import Chem

from nispo import smiles_to_iupac, smiles_to_iupac_batch
from nispo.api import _smiles_to_iupac_batch_with_telemetry


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
        "[CH-]=CCC",
        "[CH2+]CCC",
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


def test_bridge_lactam_parent_roundtrips_with_opsin():
    input_smiles = "O=C1CC2CC2N1"
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
    assert "guaiacol" not in generated_name
    assert "mequinol" not in generated_name

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
    assert "morphinan" in generated_name
    assert "morphin-" not in generated_name
    assert not generated_name.endswith("morphin")

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
    ("input_smiles", "retained_fragment"),
    [
        ("COP([O-])[O-]", "methyl phosphite"),
        ("COP(O)[O-]", "methyl hydrogen phosphite"),
        ("COP(O)O", "methyl dihydrogen phosphite"),
        ("c1ccccc1OP([O-])[O-]", "phenyl phosphite"),
        ("CC(C)(C)c1ccc(OP([O-])[O-])c(C(C)(C)C)c1", "phosphite"),
    ],
)
def test_monoester_phosphite_parents_roundtrip_with_opsin(input_smiles, retained_fragment):
    generated_name = smiles_to_iupac(input_smiles)
    assert retained_fragment in generated_name
    assert "phosphamethyl" not in generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    ("input_smiles", "retained_fragment"),
    [
        ("c1ccc2c(c1)nnn2O[P+](N1CCCC1)(N1CCCC1)N1CCCC1", "benzotriazol-1-yloxytris(pyrrolidino)phosphonium"),
        ("CN(C)[P+](On1nnc2ccccc21)(N(C)C)N(C)C", "benzotriazol-1-yloxytris(dimethylamino)phosphonium"),
        ("CC(C)(C)N=P(N1CCCC1)(N1CCCC1)N1CCCC1", "t-butylimino-tris(pyrrolidino)phosphine"),
        ("CC(C)(C)N=P(N(C)C)(N(C)C)N(C)C", "t-butylimino-tris(dimethylamino)phosphine"),
    ],
)
def test_amino_phosphonium_and_phosphanimine_parents_roundtrip_with_opsin(input_smiles, retained_fragment):
    generated_name = smiles_to_iupac(input_smiles)
    assert retained_fragment in generated_name
    assert "phosphamethyl" not in generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    ("input_smiles", "retained_fragment"),
    [
        ("[N]=O", "oxonitrogen"),
        ("O=[PH3]", "phosphane oxide"),
        ("O=[Si]", "oxosilicon"),
    ],
)
def test_small_element_oxide_retained_parents_roundtrip_with_opsin(input_smiles, retained_fragment):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name == retained_fragment

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
    assert "oxo" in generated_name
    assert "oate" in generated_name
    assert "oxalo" not in generated_name

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
    assert "carbodiimide" in generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    ("input_smiles", "retained_fragment"),
    [
        ("P(=S)([O-])([O-])[O-]", "thiophosphate"),
        ("OP(O)(O)=S", "thiophosphoric acid"),
        ("NP([O-])([O-])=S", "thiophosphoramidate"),
        ("COP([O-])([O-])=S", "thiophosphate"),
        ("COP(=S)(OC)OC", "thiophosphate"),
        ("CCO[PH]([O-])=S", "thiophosphonate"),
        ("CCCOS(=O)([O-])=S", "thiosulfate"),
        ("CO[SH]=S", "thiosulfinate"),
        ("CO[SH](=O)=S", "thiosulfonate"),
        ("Cc1cccc(OP(=S)(Oc2cccc(C)c2C)Oc2cccc(C)c2C)c1C", "thiophosphate"),
    ],
)
def test_thiooxo_ester_retained_parents_roundtrip_with_opsin(input_smiles, retained_fragment):
    generated_name = smiles_to_iupac(input_smiles)
    assert retained_fragment in generated_name
    assert "thiaethen" not in generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    ("input_smiles", "retained_fragment"),
    [
        ("NP(N)(N)=S", "thiophosphoric triamide"),
        ("[NH-]P([NH-])([NH-])=S", "thiophosphoryltriamide"),
        ("CCCCNP(N)(N)=S", "thiophosphoric triamide"),
        ("CCN(CC)P(N)(N)=S", "thiophosphoric triamide"),
        ("S=P(N1CC1)(N1CC1)N1CC1", "phosphine sulfide"),
    ],
)
def test_phosphorothioic_triamide_retained_parents_roundtrip_with_opsin(input_smiles, retained_fragment):
    generated_name = smiles_to_iupac(input_smiles)
    assert retained_fragment in generated_name
    assert "thiaethen" not in generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    ("input_smiles", "retained_fragment"),
    [
        ("[O-]P(=S)([S-])[S-]", "trithiophosphate"),
        ("S=[PH]([S-])[S-]", "trithiophosphonate"),
        ("CCCCCCCCSP([O-])([O-])=S", "dithiophosphate"),
        ("CC(C)SP(O)(O)=S", "dihydrogen dithiophosphate"),
        ("CCCCCCCCCCCCCSP(=S)([S-])[S-]", "tetrathiophosphate"),
    ],
)
def test_dithiophosphate_retained_parents_roundtrip_with_opsin(input_smiles, retained_fragment):
    generated_name = smiles_to_iupac(input_smiles)
    assert retained_fragment in generated_name
    assert "thiaethen" not in generated_name

    parsed_smiles = _opsin_parse_name(generated_name)
    assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)


@pytest.mark.parametrize(
    ("input_smiles", "retained_fragment"),
    [
        ("N=S(=O)=O", "iminosulfone"),
        ("N=S=O", "iminosulfanone"),
        ("N=[SH2]=O", "imino-lambda6-sulfanone"),
        ("N=S(F)F", "difluorosulfimide"),
        ("N=S(=O)(F)F", "sulfonimidoyl fluoride"),
        ("N=S(=O)(F)Cl", "chloro(fluoro)sulfoximine"),
        ("N=S(=O)([O-])[O-]", "sulfurimidate"),
        ("N#CN=S(=O)=O", "N-sulfonylcyanamide"),
        ("C[SH](=N)=O", "methylsulfoximine"),
        ("CC[SH](=N)=O", "ethylsulfoximine"),
        ("N=[SH](=O)c1ccccc1", "phenylsulfoximine"),
    ],
)
def test_small_sulfur_imide_retained_parents_roundtrip_with_opsin(input_smiles, retained_fragment):
    generated_name = smiles_to_iupac(input_smiles)
    assert retained_fragment in generated_name
    assert "thiaethen" not in generated_name

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
    ("input_smiles", "parent_fragment", "forbidden_fragment"),
    [
        ("O=C1CC[C@H]2[C@H]3Cc4ccccc4[C@@]2(CCN3)C1", "morphinan-6-one", "morphinan-6(17H)-one"),
        ("O=C1CCCc2ccccc2N1", "benzazepan-2-one", "benzazepan-2(1H)-one"),
    ],
)
def test_saturated_fused_lactam_suffixes_roundtrip_without_indicated_h(
    input_smiles,
    parent_fragment,
    forbidden_fragment,
):
    generated_name = smiles_to_iupac(input_smiles)
    assert generated_name
    assert parent_fragment in generated_name
    assert forbidden_fragment not in generated_name

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
    assert "tricyclo[" in generated_name

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
    assert "tetracyclo[" in generated_name or "pentacyclo[" in generated_name

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
    assert "imidazo[4,5-a]benzene" in generated_name

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
