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
