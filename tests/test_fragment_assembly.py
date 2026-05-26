from __future__ import annotations

from nispo import smiles_to_iupac_batch


def test_existing_fragment_assembly_smoke():
    names = smiles_to_iupac_batch(["Cl", "[Cl-]", "Cl.CC[NH3+]", "[Na+].[Cl-]"])

    assert names == [
        "hydrogen chloride",
        "chloride",
        "ethylazanium hydrochloride",
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


def test_organotrifluoroborate_salts_use_functional_fragment_names():
    names = smiles_to_iupac_batch(
        [
            "C[B-](F)(F)F.[K+]",
            "CC[B-](F)(F)F.[K+]",
            "CC(C)(C)OC(=O)NC[B-](F)(F)F.[K+]",
        ]
    )

    assert names == [
        "potassium methyltrifluoroborate",
        "potassium ethyltrifluoroborate",
        "potassium (5,5-dimethyl-3-oxo-2-aza-4-oxahexyl)trifluoroborate",
    ]


def test_tetraorganoborates_use_substituted_borate_names():
    names = smiles_to_iupac_batch(
        [
            "c1ccc([B-](c2ccccc2)(c2ccccc2)c2ccccc2)cc1",
            "Fc1cc([B-](c2ccccc2)(c2ccccc2)c2ccccc2)cc(F)c1F",
            "CC[B-](CC)(CC)CC.[Na+]",
            "N#C[B-](C#N)(C#N)C#N",
        ]
    )

    assert names == [
        "tetraphenylborate",
        "(3,4,5-trifluorophenyl)triphenylborate",
        "sodium tetraethylborate",
        "tetracyanoborate",
    ]


def test_azanide_fragments_use_substituted_parent_names():
    names = smiles_to_iupac_batch(
        [
            "C=C[N-]C(C)C",
            "C[Si](C)(C)[N-][Si](C)(C)C",
            "C[Si](C)(C)[N-][Si](C)(C)C.[Li+]",
            "CC[N-]C.CC[N-]C.CC[N-]C.CC[N-]C.[Zr+4]",
        ]
    )

    assert names == [
        "(1-methyl-ethyl)(eth-1-ene-1-yl)azanide",
        "bis(trimethylsilyl)azanide",
        "lithium bis(trimethylsilyl)azanide",
        "zirconium(4+) tetrakis(ethylmethylazanide)",
    ]


def test_phosphonium_salts_use_substituted_phosphanium_parent():
    names = smiles_to_iupac_batch(["CCCCCCC[P+](CCCCCCC)(CCCCCCC)CCCCCCC.[Br-]"])

    assert names == ["tetraheptylphosphanium bromide"]


def test_ammonium_salts_use_substituted_azanium_parent():
    names = smiles_to_iupac_batch(
        [
            "CCCC[NH3+].[Cl-]",
            "C[N+](C)(C)C",
            "C[N+](C)(C)Cc1ccccc1.[Cl-]",
            "CCCCCCCCCCCC[NH+](C)C.[Br-]",
            "OCC[N+](C)(C)CCO.[Cl-]",
            "[O-]C(=O)CC[N+](C)(C)C",
        ]
    )

    assert names == [
        "butylazanium chloride",
        "tetramethylazanium",
        "trimethylphenylmethylazanium chloride",
        "dodecyldimethylazanium bromide",
        "dimethylbis(3-oxapropyl)azanium chloride",
        "trimethyl(1-(1-carboxylato-methyl)-methyl)azanium",
    ]


def test_neutral_phosphines_use_substituted_phosphane_parent():
    names = smiles_to_iupac_batch(
        [
            "P(c1ccccc1)(c1ccccc1)c1ccccc1",
            "P(C1CCCCC1)(C1CCCCC1)C1CCCCC1",
            "C[As](C)C",
            "[Sb](c1ccccc1)(c1ccccc1)c1ccccc1",
            "CC[Bi](CC)CC",
        ]
    )

    assert names == [
        "triphenylphosphane",
        "tricyclohexylphosphane",
        "trimethylarsane",
        "triphenylstibane",
        "triethylbismuthane",
    ]


def test_group14_hydrides_use_substituted_parent_names():
    names = smiles_to_iupac_batch(
        [
            "C[SiH](C)C",
            "C[GeH](C)C",
            "CC[SnH](CC)CC",
            "BrC(Br)[SiH2]c1cccc(C=C)c1",
            "C[SiH](O)O",
            "C[SiH2]Cl",
            "Cl[SiH](Cl)c1ccccc1",
        ]
    )

    assert names == [
        "trimethylsilane",
        "trimethylgermane",
        "triethylstannane",
        "(1,1-dibromo-methyl)(3-(eth-1-ene-1-yl)phenyl)silane",
        "methylsilanediol",
        "methylchlorosilane",
        "phenyldichlorosilane",
    ]


def test_neutral_organophosphorus_acids_use_functional_parent_names():
    names = smiles_to_iupac_batch(
        [
            "CCP(=O)(O)O",
            "CP(=O)O",
            "O=P(O)(O)CC(=O)O",
        ]
    )

    assert names == [
        "ethylphosphonic acid",
        "methylphosphinic acid",
        "(2-oxo-3-oxapropyl)phosphonic acid",
    ]


def test_organophosphorus_oxoanion_salts_use_functional_fragment_names():
    names = smiles_to_iupac_batch(
        [
            "CCCCCC[PH](=O)[O-].CCCCCC[PH](=O)[O-].[Fe+2]",
            "CP(=O)([O-])[O-].[Ca+2]",
            "CP(=O)(O)[O-].[Na+]",
        ]
    )

    assert names == [
        "iron(2+) bis(hexylphosphinate)",
        "calcium methylphosphonate",
        "sodium hydrogen methylphosphonate",
    ]


def test_organomagnesium_halides_use_grignard_names():
    names = smiles_to_iupac_batch(
        [
            "CCC[CH2][Mg][Br]",
            "CCCCOCOCCC[CH](C)[Mg][I]",
        ]
    )

    assert names == [
        "butylmagnesium bromide",
        "(1-methyl-5,7-dioxaundecyl)magnesium iodide",
    ]


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
        [
            "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.[Cu+2]",
            "O=P([O-])(O)O.O=P([O-])(O)O.[Ca+2]",
            "O=S(=O)(O)[O-].O=S(=O)(O)[O-].[Zn+2]",
        ]
    )

    assert names == [
        "copper(2+) bis(bis(trifluoromethylsulfonyl)azanide)",
        "calcium bis(dihydrogen phosphate)",
        "zinc bis(hydrogen sulfate)",
    ]
