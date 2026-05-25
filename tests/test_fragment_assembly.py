from __future__ import annotations

from nispo import smiles_to_iupac_batch


def test_existing_fragment_assembly_smoke():
    names = smiles_to_iupac_batch(["Cl", "[Cl-]", "Cl.CC[NH3+]", "[Na+].[Cl-]"])

    assert names == [
        "hydrogen chloride",
        "chloride",
        "1-azaniumyl-ethane hydrochloride",
        "sodium chloride",
    ]
