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
