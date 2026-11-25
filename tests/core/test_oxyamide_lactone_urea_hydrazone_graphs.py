"""Independent structural variations on the four published functional-class definitions."""
import pytest


pytestmark = pytest.mark.current_only


@pytest.mark.parametrize('inputs', [
    pytest.param(('O=S(NCC)c1cccnc1', 'N[Se](=O)c1cccnc1', 'CC(C)C[Te](=O)NCC',
                  'CC[Se](=[18O])Nc1ccccc1', 'C=CCC[Se](=O)(=O)N',
                  'CC[S@](=O)NCC', 'CC[S@@](=O)NCC', 'CCN[Se](=O)CCC[Se](=O)NC',
                  'N[Te](=O)(=O)c1ccc([Te](=O)(=O)NCC)cc1'), id='oxyamide'),
    pytest.param(('O=C1OC2CCCCC2C1', 'O=C1OC2C=CCCC2C1', 'S=C1OC2CCCCC2C1',
                  'O=C1O[C@H]2CCCC[C@@H]2C1', 'O=C1O[C@H]2CCCC[C@H]2C1',
                  'S=C1OCCC(=S)OCC1', 'S=C1OCC2(CC1)CCCCC2', 'S=C1OCCC(=[Se])OCC1'), id='lactone'),
    pytest.param(('CCNC(=O)N=C1CCOCC1', 'NC(=O)/N=C(/C)CCC', 'NC(=O)/N=C(\\C)CCC',
                  '[15NH2]C(=O)N=C(C)CCC', 'NC(=O)N=[13C](C)CC', 'CC=NC(=O)N=C1CCOCC1',
                  'COC(=O)CNC(=O)N=C1CCCC1'), id='urea'),
    pytest.param(('NN=C1CCOCC1', 'CC(C)CC=NN', 'CC/C=N/N', 'CC/C=N\\N',
                  'CCNN=C1CCC(C(=O)O)CC1', 'NN=C1CCC(O)CC1', 'CCC=NNc1ccc(F)cc1',
                  'CCNN=CCCC=NNCC', 'c1ccc(NN=CC(C)C=NNc2ccccc2)cc1'), id='hydrazone'),
    pytest.param(('N[Se](=O)CCC[Te](=O)(=O)N', 'N[Se](=O)(=O)c1cccc(S(=O)N)c1',
                  'NS(=O)CCCC(CC[Se](=O)N)CCCCCCC', 'CN[Se](=O)CC(S(=O)N)CCS(=O)NCC',
                  'NS(=O)C[C@H](F)C[Se](=O)N', 'N[S@](=O)C[C@H](F)C[Se](=O)N',
                  'N[S@@](=O)C[C@H](F)C[Se](=O)N', 'N[S@](=O)CC[Se@@](=O)N',
                  'N[S@](=O)CC[Se@](=O)N', 'NS(=O)(=O)CC[Se@@](=O)N'), id='mixed-oxyamides'),
    pytest.param(('O=C1OC2CC3CCC(C1)C23', 'O=C1OC2CCC3CC(OC3=S)C2C1',
                  'O=C1O[C@H]2CC3CCC(C1)C23', 'O=C1OC2=CC3CC=C(C1)C23',
                  '[18O]=C1OC2CC3CCC(C1)C23'), id='bridged-lactones'),
    pytest.param(('C[13CH2]NC(=O)N=C1CCCC1', '[2H]N(CC)C(=O)/N=C(/C)CC',
                  'CC[15NH][13C](=[18O])N(C)CC', '[2H]NC(=[Se])N=C1CCCC1',
                  'CC=[15N]N', 'CC=N[15NH2]', 'CC/C=N/[15NH2]',
                  '[15NH2]N=C1CCC(O)CC1'), id='labelled-ureas-and-hydrazones'),
    pytest.param(('CCN/N=C/CC/C=N/NCC', 'CCN/N=C/CC/C=N\\NCC', 'CCN/N=C\\CC/C=N\\NCC',
                  'CCN/N=C/CCC=NNCC', 'CCN/N=C/C=C/C=N/NCC', 'CCN/N=C/[C@H](F)C/C=N/NCC',
                  'CCN/N=C(C)/CC/C(C)=N/NCC', 'CC[15NH]/N=C/CC/C=N/[15NH]CC'),
                 id='bis-hydrazone-geometry'),
])
def test_graph_and_reversed_atom_order(naming, assert_graphs, inputs):
    assert_graphs(inputs, naming.order_invariant_names(inputs), raw_only=True)
