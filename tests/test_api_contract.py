from __future__ import annotations

from nispo import smiles_to_iupac, smiles_to_iupac_batch


def test_public_api_single_and_batch_contract():
    assert smiles_to_iupac("c1ccccc1") == "benzene"

    inputs = ["CC", "not-a-smiles", ""]
    outputs = smiles_to_iupac_batch(inputs)

    assert len(outputs) == len(inputs)
    assert outputs[0] == "ethane"
    assert outputs[1:] == ["", ""]


def test_batch_does_not_abort_on_unsupported_molecule():
    outputs = smiles_to_iupac_batch(["CC", "[Fe+2]", "c1ccccc1"])

    assert outputs[0] == "ethane"
    assert outputs[2] == "benzene"


def test_fluorine_18_label_marks_fluoro_substituent():
    assert smiles_to_iupac("[18F]C1CCC1") == "1-[18F]fluoro-cyclobutane"
