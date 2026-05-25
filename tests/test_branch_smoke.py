from __future__ import annotations

from nispo import smiles_to_iupac_batch


def test_branch_smoke_names_remain_stable():
    names = smiles_to_iupac_batch(
        [
            "CCc1ccccc1",
            "CC(=O)N",
            "CC#N",
            "CCO",
        ]
    )

    assert names == [
        "1-ethyl-benzene",
        "1-amino-1-oxo-ethane",
        "1-cyano-methane",
        "1-hydroxy-ethane",
    ]
