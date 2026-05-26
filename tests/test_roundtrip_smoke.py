from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from nispo import smiles_to_iupac_batch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPSIN_JAR = PROJECT_ROOT / "benchmark" / "opsin-cli-2.9.0-jar-with-dependencies.jar"
JAVA = Path("/opt/homebrew/opt/openjdk/bin/java")


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def test_representative_names_roundtrip_through_opsin(tmp_path):
    if not OPSIN_JAR.exists():
        pytest.skip("OPSIN jar is not available")
    java = str(JAVA) if JAVA.exists() else "java"
    java_check = subprocess.run([java, "-version"], capture_output=True, text=True, check=False)
    if java_check.returncode != 0:
        pytest.skip("Java runtime is not available for OPSIN")

    smiles = [
        "CC",
        "c1ccccc1",
        "CCO",
        "CC(=O)N",
        "OC[C@@H](O)C(O)[C@@H](O)CO",
        "C/C=C/C",
        "CCC(=O)N[C@H](CO)c1cccc(N2CCN(C)[C@H](C)C2)n1",
        "ClC(=NNc1ccccc1)c1ccccc1",
        "COC(=O)C(O)c1ccc(NC(=O)[C@H](C)NC(=O)[C@@H](NC(=O)OC(C)(C)C)C(C)C)cc1",
        "C[C@H]1C[C@@H](NC(=O)OC(C)(C)C)CC[C@H]1O",
        "CC(=CC#N)CCC1CC1(C)C",
        "O=C(C=C(c1cc(Cl)cc(Cl)c1)C(F)(F)F)OC(=O)c1ccccc1",
        "CN1CC[NH+](Cc2ccccc2)C1",
        "CO/N=C/C=N/Nc1cccc(F)c1C(=O)O",
        "O=C(/C=C/C=C/c1ccc2c(c1)OCO2)N1CCN(c2ccc(F)cc2F)CC1",
        "Cc1ccc(S(=O)(=O)N/N=C/C(=O)O)cc1",
        "C/C(=N/NS(=O)(=O)c1ccc(C)cc1)c1cccc(Br)c1",
        "C=CC(=O)OC12CC3CC(C1)CC(OC(=O)C=C)(C3)C2",
        "Cc1ccc(S(=O)(=O)OC[C@H](O)[C@@H](O)COS(=O)(=O)c2ccc(C)cc2)cc1",
        "CCOC(=O)[C@@H](OS(=O)(=O)c1ccc([N+](=O)[O-])cc1)[C@H](O)c1ccccc1",
        "[2H]Nc1ccc(C)cc1",
        "[2H]OCCOCCOCC",
        "CN1CCc2cccc3c2[C@H]1Cc1ccccc1-3",
        "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
        "COc1cccc2[nH]c(C(=O)N3C[C@@H]4CCCC[C@@H]4[C@H]3C(=O)O)cc12",
        "[13CH3][13C](=O)O",
        "OCc1ccc([18F])cc1",
        "Nc1ccc(-n2[se]c3cc(N)ccc3c2=O)cc1",
        "Brc1cccc2c1[te]c1ccccc12",
        "O=C(O)CCC(P(=O)(O)O)P(=O)(O)O",
        "CCCOOP(=O)([O-])[O-]",
    ]
    names = smiles_to_iupac_batch(smiles)
    names_file = tmp_path / "names.txt"
    out_file = tmp_path / "opsin.smi"
    names_file.write_text("\n".join(names) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [java, "-jar", str(OPSIN_JAR), "-o", "smi", str(names_file), str(out_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    parsed = out_file.read_text(encoding="utf-8").splitlines()
    assert [_canonical(item) for item in parsed] == [_canonical(item) for item in smiles]
