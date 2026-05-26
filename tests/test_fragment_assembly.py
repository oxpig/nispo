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
            "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
            "[BH4-]",
            "[Na+].[BH4-]",
            "[2H-].[2H-].[Ti+2]",
            "[BH3-]C#N",
            "CC(=O)O[BH-](OC(C)=O)OC(C)=O",
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
        "bis(trifluoromethylsulfonyl)azanide",
        "borohydride",
        "sodium borohydride",
        "titanium(2+) dideuteride",
        "cyanoborohydride",
        "triacetoxyborohydride",
    ]


def test_phosphonium_salts_use_substituted_phosphanium_parent():
    names = smiles_to_iupac_batch(["CCCCCCC[P+](CCCCCCC)(CCCCCCC)CCCCCCC.[Br-]"])

    assert names == ["tetraheptylphosphanium bromide"]


def test_repeated_complex_onium_substituents_use_complex_multipliers():
    names = smiles_to_iupac_batch(
        [
            "C[S+](c1ccc(C)cc1)c1ccc(C)cc1",
            "[P+](c1ccc(C)cc1)(c1ccc(C)cc1)(c1ccccc1)c1ccccc1",
            "CN(C)[P+](N(C)C)(N(C)C)ON1N=NC=C1.F[P-](F)(F)(F)(F)F",
        ]
    )

    assert names == [
        "methylbis(4-methylphenyl)sulfanium",
        "diphenylbis(4-methylphenyl)phosphanium",
        "((1,2,3-triazol-1-yl)oxy)tris(dimethylamino)phosphanium hexafluorophosphate",
    ]


def test_complex_counterions_use_complex_multipliers():
    names = smiles_to_iupac_batch(
        ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.[Cu+2]"]
    )

    assert names == ["copper(2+) bis(bis(trifluoromethylsulfonyl)azanide)"]
