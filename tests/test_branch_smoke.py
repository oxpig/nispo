from __future__ import annotations

from rdkit import Chem

from nispo import smiles_to_iupac_batch
from nispo.branch.functional_suffixes import suffix_prefix_term, suffix_supported_term
from nispo.branch.ring_substituents import _generic_polycyclic_ring_substituent_prefix
from nispo.branch_rules import branch_prefix
from nispo.ring_rules import ring_parent_candidates


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


def test_branch_root_isotopes_are_rendered_on_substituent_prefixes():
    names = smiles_to_iupac_batch(
        [
            "[13CH3]c1ccccc1",
            "[13CH3]N1CCCCC1=O",
            "C[13C](=O)N1CCCC1",
            "c1ccc([125I])cc1",
            "c1ccc([18F])cc1",
        ]
    )

    assert names == [
        "1-(methyl-13C)-benzene",
        "1-(methyl-13C)-piperidin-2-one",
        "1-(methylcarbonyl-13C)-pyrrolidine",
        "1-[125I]iodo-benzene",
        "1-[18F]fluoro-benzene",
    ]


def test_high_multiplicity_prefixes_use_extended_multipliers():
    name = smiles_to_iupac_batch(
        [
            "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)"
            "C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)I"
        ]
    )[0]

    assert (
        name
        == "1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12-pentacosafluoro-12-iodo-dodecane"
    )


def test_composite_substituent_prefixes_are_parenthesized():
    names = smiles_to_iupac_batch(
        [
            "CN(C)c1ccccc1N=Nc1ccccc1",
            "Oc1cc(N=Nc2ccccc2)cc(O)c1N=Nc1ccccc1",
        ]
    )

    assert names == [
        "1-(2-(dimethylamino)phenyldiazenyl)-benzene",
        "1-(2,6-dihydroxy-4-(phenyldiazenyl)phenyldiazenyl)-benzene",
    ]


def test_extracted_hetero_branch_family_keeps_nitrooxy_wiring():
    mol = Chem.MolFromSmiles("CO[N+](=O)[O-]")

    assert branch_prefix(mol, 1, 0) == "nitrooxy"


def test_sulfonyl_branch_uses_complex_carbon_ligand_renderer():
    mol = Chem.MolFromSmiles(
        "COC(=O)c1sccc1S(=O)(=O)CC1(NC(=O)OC(C)(C)C)CC1"
    )

    assert "cyclopropylmethylsulfonyl" in branch_prefix(mol, 9, 8)


def test_oxymethyl_branch_accepts_complex_oxy_prefixes():
    mol = Chem.MolFromSmiles(
        "Cc1ccc(S(=O)(=O)OC[C@H]2CC[C@@H](COS(=O)(=O)c3ccc(C)cc3)O2)cc1"
    )

    assert branch_prefix(mol, 14, 13) == "(4-methylphenylsulfonyloxy)methyl"


def test_methyl_branch_accepts_two_ring_ligands():
    mol = Chem.MolFromSmiles(
        "CC(=O)Oc1nnn(C(c2ccccc2)C2CCN(C(=O)OC(C)(C)C)CC2)n1"
    )

    assert (
        branch_prefix(mol, 8, 7)
        == "(1-((1,1-dimethyl-ethyl)oxycarbonyl)piperidin-4-yl)(phenyl)methyl"
    )


def test_extracted_ring_branch_family_keeps_polycyclic_fallback_importable():
    mol = Chem.MolFromSmiles("Cc1ccccc1")

    assert _generic_polycyclic_ring_substituent_prefix(mol, 1, 0) == ""


def test_kekulized_aromatic_ring_substituent_keeps_retained_prefix():
    mol = Chem.MolFromSmiles("Cc1ccccc1")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    mol.SetBoolProp("_nispo_kekulized_aromatic_view", True)
    ring = next(ring for ring in mol.GetRingInfo().AtomRings() if 1 in ring)

    assert branch_prefix(mol, 1, 0) == "phenyl"
    assert ring_parent_candidates(mol, tuple(ring))[0][0] == "benzene"


def test_functional_branch_terms_are_backed_by_inverse_suffix_resources():
    assert suffix_prefix_term("amide", "") == "carbamoyl"
    assert suffix_prefix_term("nitrile", "") == "cyano"
    assert suffix_prefix_term("one", "") == "oxo"
    assert suffix_prefix_term("thione", "") == "thioxo"
    assert suffix_supported_term("ium", "azaniumyl") == "azaniumyl"
