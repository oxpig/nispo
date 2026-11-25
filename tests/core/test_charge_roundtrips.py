"""Round trips for P-72, P-73 and P-74 on drug-like scaffolds.

The small analogues are independently drawn. The four numbered development
records are original source SMILES from SureChEMBL, Zenodo 16755947 (2025-08-06),
selected with seed 20261124 and compared with f906cba89cf41518dd8792b7e21922bc707476da.
"""

import pytest
from rdkit import Chem


@pytest.mark.current_only
@pytest.mark.parametrize("structures", [
    pytest.param((
        "c1ccccc1C(=O)[N-]OC", "O=C([O-])c1ccc(-c2nnn[n-]2)cc1",
        "O=C([O-])c1ccc(-c2nc3cc(F)ccc3[n-]2)cc1",
        # SureChEMBL data lines 3679128 and 3667615. Ring and pendant stereocentres.
        "O[B-]1([C@@H]2C[C@@H]2C2=CC=C(C(=C2C1)C(=O)O)OC1CN(C1)C(C[C@@H]1CN(CCO1)C)=O)O",
        "C(N)(=O)[C@H]1C[C@H](CN1)N1CC(C1)OC1=C(C=2O[B-]([C@H]3C[C@H]3C2C=C1)(O)O)C(=O)[O-]",
    ), id="anions"),
    pytest.param((
        "NC(=[NH2+])c1ccc(F)cc1", "NC(=N)c1ccc(F)cc1",
        "CN=C(N)c1cccnc1", "C[NH+]=C(N)c1cccnc1",
        "C[S@](=O)(=[NH+]C#N)c1ccccc1", "C[S@@](=O)(=[NH+]C#N)c1ccccc1",
        "CS(=O)(=NC#N)c1ccccc1", "CS(c1ccccc1)=[NH+]CC",
        # SureChEMBL data line 2569048, supplied protonation on the imino N.
        "C(#N)[NH+]=S(=O)(C(F)F)C1=C2C(CCC2=C(C=C1)OC1=CC(=CC(=C1)F)C#N)=O",
    ), id="cations-and-neutral-controls"),
    pytest.param((
        "C[n+]1cccc(-c2nnn[n-]2)c1", "C[n+]1cccc(-c2nn[n-]n2)c1",
        "C[n+]1cccc(-c2[n-]c3cc(F)ccc3n2)c1", "C[n+]1cccc(-c2cc3ccccc3[n-]2)c1",
        "C[n+]1cccc(-c2ccc3[n-]nnc3c2)c1", "C[n+]1cccc(-c2ccc3n[n-]nc3c2)c1",
        "c1ccccc1S(=O)(C)=[NH+]CC(=O)[O-]", "CC=[N+](C)[CH-]c1ccccc1",
        "[CH2-][P+](C)(C)c1ccc(F)cc1", "CC[CH-][S+](C)c1ccccc1",
        "CC[CH-][O+](C)c1ccccc1", "c1ccccc1[CH-][Se+](C)C",
        "CC[CH-][Te+](C)c1ccccc1", "C[CH-][As+](C)(C)c1ccccc1",
        "C[CH-][Sb+](C)(C)c1ccccc1",
        # SureChEMBL data line 4095500. The nitro group lies inside an N substituent.
        "O1[C@H](COCC1)[C@H]1NC2=C(OC1)C=C(C=C2[N+](=O)[O-])S(=O)(=O)NC(C2=C(C=C(C=C2)N2CCC1(CC(C1)N1[C@@H](CCC1)C1=C(C=CC=C1)C(F)(F)F)CC2)N2[C@@H]1[C@@H](OC3=C2C=C2C(=N3)NC=C2)COCC1)=O",
    ), id="zwitterions-and-dipoles"),
])
def test_charged_graph_survives_naming(naming, assert_graphs, structures):
    inputs = list(structures)
    for smiles in structures:
        mol = Chem.MolFromSmiles(smiles)
        reordered = Chem.RenumberAtoms(mol, tuple(reversed(range(mol.GetNumAtoms()))))
        inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    names = naming.names(inputs)
    assert all(names)
    assert_graphs(inputs, names, raw_only=True)
