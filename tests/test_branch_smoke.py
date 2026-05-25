from __future__ import annotations

from rdkit import Chem

from nispo import smiles_to_iupac_batch
from nispo.branch.functional_suffixes import suffix_prefix_term, suffix_supported_term
from nispo.branch.ring_substituents import _generic_polycyclic_ring_substituent_prefix
from nispo.branch_rules import branch_prefix


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


def test_extracted_hetero_branch_family_keeps_nitrooxy_wiring():
    mol = Chem.MolFromSmiles("CO[N+](=O)[O-]")

    assert branch_prefix(mol, 1, 0) == "nitrooxy"


def test_extracted_ring_branch_family_keeps_polycyclic_fallback_importable():
    mol = Chem.MolFromSmiles("Cc1ccccc1")

    assert _generic_polycyclic_ring_substituent_prefix(mol, 1, 0) == ""


def test_functional_branch_terms_are_backed_by_inverse_suffix_resources():
    assert suffix_prefix_term("amide", "") == "carbamoyl"
    assert suffix_prefix_term("nitrile", "") == "cyano"
    assert suffix_prefix_term("one", "") == "oxo"
    assert suffix_prefix_term("thione", "") == "thioxo"
    assert suffix_supported_term("ium", "azaniumyl") == "azaniumyl"
