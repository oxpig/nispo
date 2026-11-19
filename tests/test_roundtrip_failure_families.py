"""Graph-equivalence checks for shared numbering and attachment fixes."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest
from rdkit import Chem

from nispo import smiles_to_iupac_batch


def _structures():
    return json.loads(
        (Path(__file__).parent / "data" / "roundtrip_failure_family_structures.json").read_text()
    )


def _assert_roundtrips(inputs, *, name_batch=smiles_to_iupac_batch):
    jar = Path(
        os.environ.get(
            "NISPO_OPSIN_JAR",
            Path(__file__).resolve().parents[2]
            / "benchmark/opsin-cli-2.9.0-jar-with-dependencies.jar",
        )
    )
    java = os.environ.get("JAVA") or shutil.which("java")
    if not java or not jar.is_file():
        pytest.skip("OPSIN JAR and Java are required for structure equivalence")
    names = name_batch(iter(inputs))
    assert len(names) == len(inputs)
    assert all(names)
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "names.txt"
        output = Path(directory) / "structures.tsv"
        source.write_text("\n".join(names) + "\n", encoding="utf-8")
        subprocess.run(
            [java, "-jar", str(jar), "-o", "smi", "-n", str(source), str(output)],
            capture_output=True, text=True, check=True, timeout=120,
        )
        lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(inputs)
    for smiles, name, line in zip(inputs, names, lines, strict=True):
        parsed, separator, echo = line.partition("\t")
        assert separator and echo == name
        if not parsed and name.startswith("Δ^"):
            # OPSIN 2.9 cannot parse the Blue Book Delta convention. Its
            # numbered parent CML can still be checked independently by
            # enumerating the valence-consistent localized bond graphs.
            from tools.numbered_stereo_verifier import verify_delta_graph as verify
            assert verify(smiles, name, java=java, jar=jar), (smiles, name)
            continue
        actual = Chem.MolFromSmiles(parsed) if parsed else None
        expected = Chem.MolFromSmiles(smiles)
        assert actual is not None, (smiles, name)
        assert Chem.MolToSmiles(actual) == Chem.MolToSmiles(expected), (smiles, name, parsed)


def test_fast_surechembl_ring_hydrogen_and_stereo_roundtrip():
    from nispo.fast_selection import smiles_to_iupac_fast

    def fast_batch(inputs):
        return [smiles_to_iupac_fast(smiles) for smiles in inputs]

    # SureChEMBL development graphs, Zenodo 16755947, selection seed 20261026.
    # Frozen development baseline 766bad466743; source rows below.
    _assert_roundtrips([
        "O=C1Cc2ccccc2N1",  # 805
        "S=C1Cc2ccccc2N1",  # 19060
        r"CC(C)(C)OC(=O)N1CCC2(CC/C(=C\c3ccccc3)C2)C1",  # 1794051
        "NC(=O)/C=C/c1cc(O)c(O)cc1C1CCC(c2cc(O)c(O)cc2/C=C/C(N)=O)C1O",  # 723918
        "CCCCC(=O)N(Cc1ccc(-c2cc(-c3ccccc3)ccc2/C(N)=N/O)cc1)C1(C(=O)OC)CCCCC1",  # 2981004
        (r"C=C1/C(=C\C=C2/CCC[C@]3(C)[C@@H]([C@H](C)/C=C/[C@@H](O)"
         r"C4(/C=C/C(=O)OCC)CC4)CC[C@@H]23)C[C@@H](O)C[C@@H]1O"),  # 4001412
        "COC(=O)C1(O)C[C@@H](OC/C=C/c2ccc(O)c(O)c2)C(O)[C@H](OC/C=C/c2ccc(O)c(O)c2)C1",  # 2944009
        (r"C(=C\c1ccc(N(c2ccc(/C=C/c3ccncc3)cc2)c2ccc(/C=C/c3ccncc3)cc2)cc1)"
         r"/c1ccncc1"),  # 982508
    ], name_batch=fast_batch)


def test_full_surechembl_carboxyalkanoyl_alternative_roundtrips():
    # SureChEMBL development, Zenodo 16755947, seed 20261026,
    # frozen baseline 766bad466743, source row 17987.
    def full_batch(inputs):
        return smiles_to_iupac_batch(inputs, timeout_seconds=None, fallback_to_fast=False)

    _assert_roundtrips(["CC(C)C[C@H](NC(=O)C(=O)O)[C@@H](O)CC(=O)O"], name_batch=full_batch)


def test_full_surechembl_ring_without_supported_element_returns_cleanly():
    from nispo import smiles_to_iupac_result

    # SureChEMBL development, Zenodo 16755947, seed 20261026,
    # frozen baseline 766bad466743, source row 240709.
    result = smiles_to_iupac_result("CC(=O)OC1=c2ccccc2=C[IH]1",
                                    timeout_seconds=None, fallback_to_fast=False)
    assert result.generation_status in {"available", "unavailable"}


@pytest.mark.parametrize(
    "group", ["retained-locants-and-tautomers", "attachment-bonds-and-bridges"]
)
def test_numbering_and_attachment_families_preserve_structure(group):
    _assert_roundtrips([row["smiles"] for row in _structures() if row["group"] == group])


def test_numbering_and_attachment_families_preserve_structure_after_atom_reordering():
    inputs = []
    representatives = [
        "NC(=O)c1ccc(-c2ccc3c(c2F)CC3)cc1",
        "NC(=O)c1ccc(-c2ccc3[nH]cnc3c2F)cc1",
        "O=C1NC(=C=S)N(CC)c2cc[nH]c21",
        "CC1(C)OC2(C)C(F)CC1CC2O",
    ]
    for smiles in representatives:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        for shift in (0, 2):
            order = list(range(mol.GetNumAtoms()))
            order = order[shift:] + order[:shift]
            for indices in (order, order[::-1]):
                reordered = Chem.RenumberAtoms(mol, indices)
                inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    _assert_roundtrips(inputs)


def test_bridged_ketones_alcohols_and_sulfur_ligands_preserve_structure():
    _assert_roundtrips([
        "CC12CCC(CC1=O)C2(C)C",
        "CC1(C)C2CCC1(C)C(O)C2",
        "O=C1CCC2CC1CC2",
        "OC1CCC2CC1CC2",
        "O=C1C(O)CC2CC1CC2",
        "OC1C(O)CC2CC1CC2",
        "O=C1CCC2CC1CC2=O",
        "O=C1CCC2C(C)C1CC2",
        "OC1CCC2C(F)C1CC2",
        "Cc1ccc(-c2c(C(C)C)nc(N=S(C)(C)=O)c3cc4[nH]ncc4cc23)cc1",
        "CC(C)c1nc(N=S(C)(=O)c2ccccc2)c2cc3[nH]ncc3cc2c1-c1ccc(F)cc1",
        "CS(N)(=O)=NC(=O)Nc1ccccc1",
        # Controls: heterocyclic, fused, spiro and side-chain alcohol parents
        # exercise the existing naming routes outside suffix promotion.
        "O=C1CCN2CCC1CC2",
        "OC1CCC2CCCCC2C1",
        "OC1CCC2(CC1)CCCC2",
        "OCCC1CC2CCC1C2",
    ])


def test_bridged_functional_stereochemistry_survives_atom_reordering():
    inputs = []
    for smiles in [
        "CC1(C)[C@H]2CC[C@]1(C)[C@@H](O)C2",
        "CC1(C)[C@@H]2CC[C@@]1(C)[C@H](O)C2",
        "O=C1C[C@H]2CC[C@@H]1C2",
        "OC1C(O)CC2CC1CC2",
    ]:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        for shift in (0, 2):
            order = list(range(mol.GetNumAtoms()))
            order = order[shift:] + order[:shift]
            for indices in (order, order[::-1]):
                reordered = Chem.RenumberAtoms(mol, indices)
                inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    _assert_roundtrips(inputs)


@pytest.mark.parametrize("group", [
    "boron", "phosphorus", "sulfur", "amidate", "nitrile-oxide",
    "charged-arene", "sulfamide-and-imine",
])
def test_charge_valence_and_attachment_families_preserve_structure(group):
    rows = json.loads(
        (Path(__file__).parent / "data" / "charge_valence_structures.json").read_text()
    )
    inputs = [row["smiles"] for row in rows if row["group"] == group]
    assert smiles_to_iupac_batch(inputs) == smiles_to_iupac_batch(iter(inputs))
    _assert_roundtrips(inputs)


def test_ionic_bonding_states_survive_atom_reordering():
    inputs = []
    for smiles in [
        "Cc1ccc(B([O-])[O-])cc1",
        "C#C[PH]([O-])(OC)OCC",
        "CCN([SH2](=O)C=N)CC",
        "O=C([N-][n+]1ccc(C)cc1)c1ccccc1",
        "[O-][N+]#Cc1ccc(F)cc1",
        "[C-]1=[C-]C=CC=C1",
        "CCN(C)S(=O)(=O)Nc1ccccc1",
        "C/C(=N\\NCC(=O)O)c1ccccn1",
    ]:
        mol = Chem.MolFromSmiles(smiles)
        for shift in (0, 2):
            order = list(range(mol.GetNumAtoms()))
            order = order[shift:] + order[:shift]
            for indices in (order, order[::-1]):
                reordered = Chem.RenumberAtoms(mol, indices)
                inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    _assert_roundtrips(inputs)


@pytest.mark.parametrize("group", [
    "complete-parents", "suffix-vowels", "prefix-citation",
    "cyclic-alcohols", "stereo-compatibility",
])
def test_nomenclature_rendering_families_preserve_structure(group):
    rows = json.loads(
        (Path(__file__).parent / "data" / "bluebook_roundtrip_structures.json").read_text()
    )
    inputs = [row["smiles"] for row in rows if row["group"] == group]
    assert smiles_to_iupac_batch(inputs) == smiles_to_iupac_batch(iter(inputs))
    _assert_roundtrips(inputs)


def test_parent_and_prefix_rendering_survives_atom_reordering():
    inputs = []
    for smiles in [
        "CC/C=C/CC", "C=CC=CC=C", "C#CC#CC#C",
        "O=C1CCNCC1", "O[C@H]1CC[C@@H](C)CC1", "OC1C(O)CC(O)CC1",
        "FC1CCC(c2ccc(Cl)cc2)CC1", "CCOP(=O)(OCC)[N@]1CC1C",
    ]:
        mol = Chem.MolFromSmiles(smiles)
        for shift in (0, 2):
            order = list(range(mol.GetNumAtoms()))
            order = order[shift:] + order[:shift]
            for indices in (order, order[::-1]):
                reordered = Chem.RenumberAtoms(mol, indices)
                inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    _assert_roundtrips(inputs)


@pytest.mark.parametrize("group", [
    "ketone-suffix", "polyol-suffix", "acid-parent", "ring-suffix", "seniority-controls",
])
def test_principal_oxygen_families_preserve_structure(group):
    rows = json.loads(
        (Path(__file__).parent / "data" / "parent_suffix_roundtrip_structures.json").read_text()
    )
    inputs = [row["smiles"] for row in rows if row["group"] == group]
    assert smiles_to_iupac_batch(inputs) == smiles_to_iupac_batch(iter(inputs))
    _assert_roundtrips(inputs)


def test_principal_oxygen_families_survive_atom_reordering():
    inputs = []
    for smiles in [
        "CC(=O)C[C@H](O)C", "C/C=C/C(=O)CC", "C/C=C\\C(=O)CC",
        "OCC(CO)(CO)CCO", "C[C@H](O)[C@@H](O)CO", "O=C(O)CCC(=O)O",
        "ON1CCCC1", "Oc1ccc2cc(O)ccc2c1", "O=C1CCCC2(CCCC2=O)C1",
    ]:
        mol = Chem.MolFromSmiles(smiles)
        for shift in (0, 2):
            order = list(range(mol.GetNumAtoms()))
            order = order[shift:] + order[:shift]
            for indices in (order, order[::-1]):
                reordered = Chem.RenumberAtoms(mol, indices)
                inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    _assert_roundtrips(inputs)


@pytest.mark.parametrize("group", [
    "aldehyde-seniority", "suffix-multipliers-and-methanol",
    "fused-suffix-numbering", "sulfonamide-coverage", "lactone-seniority", "flagged",
])
def test_principal_group_followup_families_preserve_structure(group):
    rows = json.loads(
        (Path(__file__).parent / "data" / "principal_group_followup_structures.json").read_text()
    )
    inputs = [row["smiles"] for row in rows if row["group"] == group]
    assert smiles_to_iupac_batch(inputs) == smiles_to_iupac_batch(iter(inputs))
    _assert_roundtrips(inputs)


def test_principal_group_followup_survives_atom_reordering():
    inputs = []
    representatives = [
        "O=C[C@H](O)CO", "O=C/C=C/CO", "O=C/C=C\\CO", "O=C(O)C(O)C=O",
        "OCC(O)C(O)C(O)C(O)CO", "OCc1ccc(F)cc1",
        "Oc1cc(F)c2c(c1)CCCC2", "O=C([O-])c1ccc[nH+]c1",
        "O=S(=O)(Nc1ccc(S(=O)(=O)c2ccccc2)cc1)C(F)(F)F",
        "O=c1oc2ccccc2c(O)c1C(C[C@H](O)c2ccccc2)c1ccccc1",
        "CC[C@@H](C)CC[C@H]1[C@@H](C)CC[C@H]2C(C)(C)CC(O)C(O)(O)[C@]12C",
    ]
    for smiles in representatives:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        for shift in (0, 2):
            order = list(range(mol.GetNumAtoms()))
            order = order[shift:] + order[:shift]
            for indices in (order, order[::-1]):
                reordered = Chem.RenumberAtoms(mol, indices)
                inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    _assert_roundtrips(inputs)


@pytest.mark.parametrize("group", [
    "branched-prefix-joins", "compound-prefixes", "biphenyl-prefixes",
    "enclosing-marks-controls", "targets",
])
def test_prefix_presentation_families_preserve_structure(group):
    rows = json.loads(
        (Path(__file__).parent / "data" / "prefix_presentation_structures.json").read_text()
    )
    inputs = [row["smiles"] for row in rows if row["group"] == group]
    assert smiles_to_iupac_batch(inputs) == smiles_to_iupac_batch(iter(inputs))
    _assert_roundtrips(inputs)


def test_prefix_presentation_preserves_structure_after_atom_reordering():
    rows = json.loads(
        (Path(__file__).parent / "data" / "prefix_presentation_structures.json").read_text()
    )
    representatives = [r["smiles"] for r in rows if r["group"] == "targets"] + [
        "CC[C@H](C)CCNC(=O)C", "C[S@](=O)c1ccc(C(=O)O)cc1",
        "NC(=O)CC(c1c(-c2cc(Cl)ccc2)cccc1)O",
        "NC(=O)CC(c1ccc(-c2cc(C)cc(C)c2)cc1)O",
        "CC(C)(C)OC(=O)Nc1ccccc1", "[13cH]1ccccc1",
    ]
    inputs = []
    for smiles in representatives:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        for shift in (0, 2):
            order = list(range(mol.GetNumAtoms()))
            order = order[shift:] + order[:shift]
            for indices in (order, order[::-1]):
                reordered = Chem.RenumberAtoms(mol, indices)
                inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    _assert_roundtrips(inputs)


@pytest.mark.parametrize("group", ["fusion", "acy-citation", "scope", "stereo-nesting"])
def test_presentation_followup_families_preserve_structure(group):
    rows = json.loads(
        (Path(__file__).parent / "data" / "presentation_followup_structures.json").read_text()
    )
    inputs = [row["smiles"] for row in rows if row["group"] == group]
    assert smiles_to_iupac_batch(inputs) == smiles_to_iupac_batch(iter(inputs))
    _assert_roundtrips(inputs)


def test_presentation_followup_survives_atom_reordering():
    rows = json.loads(
        (Path(__file__).parent / "data" / "presentation_followup_structures.json").read_text()
    )
    inputs = []
    for group in ("fusion", "acy-citation", "scope", "stereo-nesting"):
        representatives = [row for row in rows if row["group"] == group][:2]
        for row in representatives:
            mol = Chem.MolFromSmiles(row["smiles"])
            assert mol is not None
            for shift in (0, 2):
                order = list(range(mol.GetNumAtoms()))
                order = order[shift:] + order[:shift]
                for indices in (order, order[::-1]):
                    reordered = Chem.RenumberAtoms(mol, indices)
                    inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    _assert_roundtrips(inputs)
