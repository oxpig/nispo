"""Protect graph fidelity for nested acyl, fused-ring and anionic groups.

The dianion and benzimidazolylidene controls were authored from Blue Book
P-72.6.1 and P-58.2, respectively. The other graphs are canonical isomeric
SMILES from the authorized SureChEMBL release of 2025-08-06, Zenodo 16755947.
Their one-based release data rows are given alongside the input. The
100,000-record source draw used seed 20261103, and the motif replay used
seed 2026112309 against frozen NISPO 6b74e6c8b6d07eee0f57daef51c8cbc89128ff72.
"""

import pytest


@pytest.mark.full_only
def test_nested_groups_recover_the_source_graph(naming, assert_graphs):
    inputs = (
        # Independently authored controls.
        "O=C([O-])C1CCC(N2CCC(C(=O)[O-])CC2)CC1",
        "O=C(O)c1ccc(N=c2[nH]c3ccccc3[nH]2)cc1",
        # SureChEMBL data rows 3141819 and 3136943.
        "NC(=O)CCN(NC(=O)[C@@H](OC(N)=O)C(Cc1ccccc1)C1CCCCC1)C(=O)[C@@H](F)Cl",
        "CC(C)[C@H](NC(=O)c1cc2ccccc2[nH]1)C(=O)NN(CCC(N)=O)C(=O)C(F)Cl",
        # SureChEMBL data rows 521990 and 1347484.
        "CCC/C=C/C(=O)N[C@H](C(=O)OCC)C(C)C",
        "NCC(=O)NCC(=O)N[C@@H](Cc1ccccc1)C(N)=O",
        # SureChEMBL data rows 1709288 and 2293754.
        "C[C@H](NS(=O)(=O)c1ccc2[nH]c(=O)c(=O)[nH]c2c1)C(=O)[O-]",
        "CCOP(=O)(/N=c1/[nH]c2cccc(C(=O)OC(C)(C)C)c2[nH]1)OCC",
    )
    names = naming.names(inputs)
    assert all(names)
    assert_graphs(inputs, names, raw_only=True)
