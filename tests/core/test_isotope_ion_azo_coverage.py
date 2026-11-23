"""Public graph recovery for independently constructed rule controls."""

import pytest


@pytest.mark.parametrize(
    'inputs',
    (
        pytest.param(
            (
                '[2H]C([2H])([2H])Oc1nccc(F)c1',
                '[2H]c1c([2H])c([2H])c(-c2cnn(COC([2H])([2H])[2H])c2)c([2H])c1[2H]',
                'CC(=O)OC([2H])([2H])C([2H])(O)C([2H])([2H])OCc1ccccc1',
            ),
            id='nested-isotope-labels',
        ),
        pytest.param(
            (
                '[Mg+2].O=C([O-])O.O=C([O-])O',
                '[Al+3].[Al+3].O=P([O-])([O-])[O-].O=P([O-])([O-])[O-]',
            ),
            id='repeated-hydrogen-carbonate-and-phosphate',
        ),
        pytest.param(
            (
                'N#CCCN=NCCC#N',
                'N#CCC/N=N/CCC#N',
                r'N#CCC/N=N\CCC#N',
                'O=C(O)c1ccc(N=Nc2ccc(C(=O)O)cc2)cc1',
            ),
            id='azo-bridge-between-matching-functional-arms',
        ),
    ),
)
def test_original_isotopes_ions_and_azo_bonds_round_trip(naming, assert_graphs, inputs):
    names = naming.names(inputs)
    assert all(names)
    assert_graphs(inputs, names, raw_only=True)
