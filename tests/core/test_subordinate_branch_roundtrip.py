"""Graph regressions found in the authorized SureChEMBL development source.

Canonical isomeric SMILES from the 2025-08-06 release (Zenodo 16755947),
with one-based release data rows beside each case. The archive draw used
seed 20261103 and the motif replay used seed 2026112311. The baseline is
the frozen round-four candidate on 6b74e6c8b6d07eee0f57daef51c8cbc89128ff72,
recorded in the round-five audit freeze manifest.
"""

import pytest


@pytest.mark.full_only
def test_subordinate_branch_graph_recovery(naming, assert_graphs):
    inputs = (
        # 3441002: two hydrogens are part of the sulfur bonding-number state.
        "Cc1cc2cnc(Nc3ccc(N4CCC(N5CCN(C)CC5)CC4)cc3)nc2n1-c1cccc(N=[SH2]=O)n1",
        # 2999838: the oxanyl ligand belongs to the lactam nitrogen.
        "Cc1ccc(S(=O)(=O)n2c(Cc3nc4ccn(C5CCOCC5)c(=O)c4[nH]3)cc3cccc(S(N)(=O)=O)c32)cc1",
        # 1792028: internal alkene geometry belongs to the ylidene branch.
        "CC(C)=C/C(=C\\C=NS(=O)(=O)c1ccccc1)N1CCCCCC1",
    )
    names = naming.names(inputs)
    assert all(names)
    assert_graphs(inputs, names, raw_only=True)
