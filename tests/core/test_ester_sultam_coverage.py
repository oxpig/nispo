"""Graph recovery for ester attachment and fused sulfur heterones.

The controls were constructed from the P-29, P-65 and P-66 graph patterns
with independent substituents and attachment positions.
"""

import pytest


@pytest.mark.full_only
@pytest.mark.parametrize(
    "inputs",
    (
        pytest.param(
            (
                "COC(=O)[C@H]1CCCN1S(=O)(=O)c1ccc(O[C@@H]2CCC[C@H](CC=O)C2)nc1",
                "COC(=O)[C@H]1CCCN1S(=O)(=O)c1ccc(O[C@@H]2CCC[C@H](CCNC(=O)OC(C)(C)C)C2)cc1",
            ),
            id="sulfonyl-cyclic-amino-acid-esters",
        ),
        pytest.param(
            ("O=S1(=O)NCCc2cc(Cl)ccc21", "O=C1COc2ccc(Cl)cc2S(=O)(=O)N1"),
            id="fused-sultam-dione-and-trione",
        ),
        pytest.param(
            ("COC(=O)C=C1CC(C(=O)O)CC1", "COC(=O)CC1CC(C(=O)O)CC1"),
            id="double-and-single-pendant-ester-attachment",
        ),
    ),
)
def test_functional_groups_recover_the_original_graph(naming, assert_graphs, inputs):
    names = naming.names(inputs)
    assert all(names)
    assert_graphs(inputs, names, raw_only=True)
