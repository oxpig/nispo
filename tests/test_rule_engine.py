from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest
from rdkit import Chem

from nispo import smiles_to_iupac, smiles_to_iupac_batch
from nispo.rule_engine import NameCandidate, _candidate_key


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPSIN_JAR = PROJECT_ROOT / "benchmark" / "opsin-cli-2.9.0-jar-with-dependencies.jar"
JAVA = Path("/opt/homebrew/opt/openjdk/bin/java")


def test_candidate_keys_are_comparable_across_descriptor_shapes():
    candidates = [
        NameCandidate("flat-parent", "flat", 29, sort_key=(1, "flat-parent")),
        NameCandidate(
            "nested-parent",
            "nested",
            29,
            sort_key=((750, 999, (), 999, "nested-parent"), len("nested-parent"), "nested-parent"),
        ),
    ]

    assert min(candidates, key=_candidate_key) is candidates[0]


def test_nested_sort_key_parent_selection_does_not_empty_api_output():
    smiles = "O=C1CC2(CCCCC2)c2cn[nH]c2N1"

    scalar = smiles_to_iupac(smiles)
    batch = smiles_to_iupac_batch([smiles])

    assert scalar
    assert batch == [scalar]


def test_single_ring_functional_suffix_parents_roundtrip():
    smiles_values = [
        "O=C(N)c1ccncc1",
        "N#Cc1ccncc1",
        "O=C([O-])c1ccncc1",
        "O=C(O)c1ccsc1",
        "O=C(O)c1cc(C#N)ncc1",
        "c1ccc(-c2ccccc2)cc1",
        "N#Cc1cc(-c2ccccc2)c(C#N)cc1",
        "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",
        "Clc1ccc(-c2ccc(-c3ccc(Cl)cc3)cc2)cc1",
        "Fc1c(F)c(F)c(OC(F)(F)F)c(F)c1F",
        "FC(F)(F)ON1CCCC1",
        "FC(F)OC1CNC1",
        "O=S(=O)(c1ccccc1)N1CCCCC1",
        "O=C(c1cccnc1)N1CCCCC1",
        "Nc1ccc(-c2nc3cc(F)ccc3s2)cc1",
        "c1ccc2cc(-c3cn4ccccc4n3)ccc2c1",
        "O=C(Nc1ccccc1)c1ccccc1",
        "O=C(Nc1ccccc1)c1ccncc1",
        "O=C(Nc1ccccc1)c1ccco1",
        "Nc1ccc(C(=O)Nc2ccccc2)cc1",
        "O=C(Nc1ccccc1)NC1CCCCC1",
        "B1(OC(C(O1)(C)C)(C)C)c2ccccc2",
        "CCCCCCCCCCCCCCCCCCCCC(=O)O",
        "CCCCCC(=O)[O-]",
        "Nc1ccc(C(=O)Nc2ccccc2N)cc1",
        "C[Si](C)(C)CCOCn1ccc2ccc(N)cc21",
        "C[Si](C)(C)CCOCOc1ccc2cccnc2c1Br",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    assert any("biphenyl" in name for name in names)
    assert any("diphenyl" in name for name in names)
    assert all("fluoromethyloxy" not in name for name in names)
    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert _canonical_smiles(parsed_smiles) == _canonical_smiles(smiles)


def test_generated_fused_indicated_h_parents_roundtrip():
    smiles_values = [
        "Ic1n[nH]c2cccnc12",
        "NC(=O)c1cncc2c[nH]nc12",
        "Nc1ncnc2[nH]ncc12",
        "Cn1ccc2cc[nH]c(=O)c21",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    assert any("H-pyrazolo" in name or "H-pyrrolo" in name for name in names)
    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert _canonical_smiles(parsed_smiles) == _canonical_smiles(smiles)


def test_fused_indicated_h_ketone_suffixes_roundtrip():
    smiles_values = [
        "O=c1nc[nH]c2cc([N+](=O)[O-])ccc12",
        "Cc1ccc2[nH]cnc(=O)c2c1",
        "O=c1cc[nH]c2nccnc12",
        "O=c1[nH][nH]c2ncncc12",
        "O=c1c2ccccc2[nH]cn1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    assert any("(1H)-one" in name or "(5H)-one" in name for name in names)
    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert _canonical_smiles(parsed_smiles) == _canonical_smiles(smiles)


def test_fused_functional_suffix_parents_roundtrip():
    smiles_values = [
        "N#Cc1ccc2n[nH]c(O)c2c1",
        "N#Cc1ccc2[nH][nH]c(=O)c2c1",
        "O=C(NCCC(=O)N1CCOCC1)c1ccc2nc[nH]c2c1",
        "O=C(Nc1ccc([N+](=O)[O-])cc1Cl)c1cc(Cl)cc2[nH]c(C(F)(F)F)nc12",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    assert any("carbonitril" in name for name in names)
    assert any("carboxamid" in name for name in names)
    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert _canonical_smiles(parsed_smiles) == _canonical_smiles(smiles)


def test_generated_oxido_fused_parents_roundtrip():
    smiles_values = [
        "[O-][n+]1cccc2[nH]ccc21",
        "[O-][n+]1cccc2ncccc12",
        "Brc1csc2[n+]([O-])cccc12",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    assert all("oxido" in name and "ium" in name for name in names)
    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert _canonical_smiles(parsed_smiles) == _canonical_smiles(smiles)


def test_charged_saturated_heterocycle_parents_roundtrip():
    smiles_values = [
        "CC1CCCC(C)[NH2+]1",
        "C1CC[NH2+]CC1",
        "C1COCC[NH2+]1",
        "C1CC[NH2+]C1",
    ]

    names = smiles_to_iupac_batch(smiles_values)
    parsed = _opsin_parse(names)

    assert all("ium" in name for name in names)
    for smiles, name, parsed_smiles in zip(smiles_values, names, parsed, strict=True):
        assert name
        assert _canonical_smiles(parsed_smiles) == _canonical_smiles(smiles)


def _opsin_parse(names: list[str]) -> list[str]:
    if not OPSIN_JAR.exists():
        pytest.skip(f"OPSIN jar not found: {OPSIN_JAR}")
    with tempfile.TemporaryDirectory(prefix="nispo_test_opsin_") as temp_name:
        temp_dir = Path(temp_name)
        names_file = temp_dir / "names.txt"
        output_file = temp_dir / "opsin.smi"
        names_file.write_text("\n".join(names) + "\n")
        java = str(JAVA) if JAVA.exists() else "java"
        completed = subprocess.run(
            [java, "-jar", str(OPSIN_JAR), "-o", "smi", str(names_file), str(output_file)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        parsed = output_file.read_text().splitlines()
    return (parsed + [""] * len(names))[: len(names)]


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
