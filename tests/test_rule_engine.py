from __future__ import annotations

from nispo import smiles_to_iupac, smiles_to_iupac_batch
from nispo.rule_engine import NameCandidate, _candidate_key, _opsin_stability_penalty


def test_candidate_keys_are_comparable_across_descriptor_shapes():
    candidates = [
        NameCandidate("flat-parent", "flat", 29, sort_key=(1, "flat-parent")),
        NameCandidate(
            "nested-parent",
            "nested",
            29,
            sort_key=((750, 999, (), 999, "nested-parent"), len("nested-parent"), "nested-parent"),
        ),
    ]

    assert min(candidates, key=_candidate_key).name == "flat-parent"


def test_nested_sort_key_parent_selection_does_not_empty_api_output():
    smiles = "O=C1CC2(CCCCC2)c2cn[nH]c2N1"

    scalar = smiles_to_iupac(smiles)
    batch = smiles_to_iupac_batch([smiles])

    assert scalar
    assert batch == [scalar]


def test_cyclohexatrienyl_substituent_does_not_penalize_polycyclic_parent():
    assert (
        _opsin_stability_penalty(
            "1-(cyclohexa-1,3,5-trien-1-yl)-2-azabicyclo[4.3.0]nona-1(2),3(4),5(6),7(8)-tetraene"
        )
        == 0
    )
    assert _opsin_stability_penalty("cyclohexa-1,3,5-trien-1-ylmethane") > 0


def test_oxabutyl_amino_spiro_substituent_is_opsin_fragile_shape():
    assert (
        _opsin_stability_penalty(
            "3-((2-hydroxy-4-(3-methylaminosulfonylphenyl)-4-oxabutyl)amino)-8-aza-1-oxaspiro[4.5]decane"
        )
        > 0
    )


def test_mixed_ring_alkyl_silane_parent_is_opsin_fragile_shape():
    assert _opsin_stability_penalty("phenylpropylsilane") > 0
    assert _opsin_stability_penalty("triphenylsilane") == 0


def test_spiro_methylidene_methane_artifact_is_opsin_fragile_shape():
    assert (
        _opsin_stability_penalty(
            "1-(7-methylidene-1,3-dioxo-2,6-diazaspiro[4.4]nonan-8-yl)-methane"
        )
        > 0
    )
