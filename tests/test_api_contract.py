from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from shutil import which

import pytest
from rdkit import Chem

from nispo import smiles_to_iupac, smiles_to_iupac_batch
from nispo.api import _smiles_to_iupac_batch_with_telemetry
import nispo.parent_rules as parent_rules


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


def test_amino_acid_parent_rejects_unrendered_sidechain():
    mol = Chem.MolFromSmiles(
        "CNc1ccc2ccc(N(CC[C@H](N)C(=O)O)C(=O)[C@H]3O"
        "[C@@H](n4cc(-c5ccn(C)n5)c5c(N)ncnc54)[C@H](O)[C@@H]3O)cc2n1"
    )

    assert parent_rules._amino_acid_parent_names(mol, Chem) == []


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
