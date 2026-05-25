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


def test_common_counterions_and_hydrides_are_named():
    names = smiles_to_iupac_batch(
        [
            "F[B-](F)(F)F",
            "F[P-](F)(F)(F)(F)F",
            "O=[N+]([O-])[O-]",
            "[Na+].O=[N+]([O-])[O-]",
            "O",
            "[NH4+]",
            "[N-]=C=O",
            "[N-]=[N+]=[N-]",
            "O=S(=O)([O-])C(F)(F)F",
        ]
    )

    assert names == [
        "tetrafluoroborate",
        "hexafluorophosphate",
        "nitrate",
        "sodium nitrate",
        "water",
        "ammonium",
        "isocyanate",
        "azide",
        "trifluoromethanesulfonate",
    ]


def test_phosphonium_salts_use_substituted_phosphanium_parent():
    names = smiles_to_iupac_batch(["CCCCCCC[P+](CCCCCCC)(CCCCCCC)CCCCCCC.[Br-]"])

    assert names == ["tetraheptylphosphanium bromide"]
