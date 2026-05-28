from __future__ import annotations

from nispo import smiles_to_iupac, smiles_to_iupac_batch
from nispo.rule_engine import NameCandidate, _candidate_key


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

    assert min(candidates, key=_candidate_key) is candidates[0]


def test_nested_sort_key_parent_selection_does_not_empty_api_output():
    smiles = "O=C1CC2(CCCCC2)c2cn[nH]c2N1"

    scalar = smiles_to_iupac(smiles)
    batch = smiles_to_iupac_batch([smiles])

    assert scalar
    assert batch == [scalar]
