"""Independent graph controls drawn from the definitions in P-68, P-24 and P-65.

The structures vary ligand, ring, bridge and substitution independently.
No benchmark panel records or molecule-level panel results supplied the inputs.
"""
import re

import pytest
from rdkit import Chem


pytestmark = pytest.mark.current_only


@pytest.mark.parametrize('inputs', [
    pytest.param(('CC/C(C)=N/Oc1ccc(F)cc1', 'CC/C(C)=N\\Oc1ccc(F)cc1',
                  'C/C(=N/O[C@H](C)CC)CC', 'CO/N=C/C=C',
                  'CCC(=N[18OH])CC(=O)O', 'CC/C(C)=N/[18O]c1ccccc1'), id='oxime-stereo-isotope'),
    pytest.param(('ON=C1CC=CCC1', 'CCON=C1CC=CC(C(=O)O)C1', 'ON=C1CCC(CC1)=NOC',
                  'CCC(=NOc1ccccc1)CC(=O)N', 'CC/C=N/OCCC(=O)O'), id='oxime-parent'),
    pytest.param(('O1CCC2(CC1)CC[SiH2]CC2', '[SiH2]1CCC12CCC3(CC2)CCC3',
                  'C1CC[Si]2(CC1)CCCC2', 'O1CC(O)C2(CC1)CC[SiH2]CC2',
                  'O1CC(C)C2(CC1)CC[Se]CC2'), id='spiro-replacement'),
    pytest.param(('O=C(c1cccnc1)OS(=O)(=O)C', 'CCCC(O)CC(=O)OS(=O)(=O)CC',
                  'CCCC(=O)OB(O)O', 'CCC(=O)OP(=O)(O)O', 'CCCC(=O)OC(=S)C',
                  'C[C@H](F)CC(=O)OS(=O)(=O)C', 'CCC(CC)C(=O)OS(=O)(=O)C',
                  'CC(C)CC(C)C(=O)OS(=O)(=O)CCC'), id='mixed-anhydride'),
    pytest.param(('CC(=NO)N', 'CC(=NOC(C)=O)C', 'CCC(=O)OCC', 'CCOS(=O)(=O)C',
                  'CCCC(=O)OC(=O)CCC', 'O=C1OC(=O)CCC1'), id='adjacent-classes'),
    pytest.param(('COC(=O)CCON=C1CCCC1', 'COC(=O)CCON=C1CCOCC1',
                  'CCOC(=O)CCO/N=C1/CCCC(C)C1', 'CCOC(=O)CCO/N=C1\\CCCC(C)C1',
                  'CCOC(=O)CCO/N=C1/CC[C@H](C)C1', 'CCOC(=O)CCO/N=C1/CC[C@@H](C)C1'),
                 id='oxime-ether-senior-ester'),
    pytest.param(('O1CC2(CC2)CC3(CCC3)CC4(CCCC4)C1', 'O1CC2(NC2)CC3(CCC3)CC4(CCCC4)C1',
                  'O1CC2(CC2)CC3(CCC3)CC4(CCCC4)C(O)1',
                  'O1CC2(CC2)CC3(CC4(CCC4)CC5(CCCC5)C3)CC6(CCCC6)C1',
                  'O1CC12CC23CC34CC45CC56CCCC6'), id='branched-and-extended-spiro'),
    pytest.param(('CCCC(=O)O[As](=O)(O)O', 'CC(C)C(=O)O[AsH]O', 'CCCC(=O)O[Si](O)(O)O',
                  'O=C(c1ccccc1)O[Se](=O)O', 'CCCC(=O)O[Cl+2]([O-])[O-]',
                  'CCCC(=O)O[Sb](=O)(O)O'), id='inorganic-acid-residues'),
])
def test_original_graph_and_atom_order(naming, assert_graphs, inputs):
    reversed_smiles = []
    for smiles in inputs:
        graph = Chem.MolFromSmiles(smiles)
        other = Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms()))))
        reversed_smiles.append(Chem.MolToSmiles(other, canonical=False))
    names = naming.names(inputs)
    assert names == naming.names(reversed_smiles)
    assert_graphs(inputs, names, raw_only=True)


def test_oxime_hydroxyl_isotopes(naming, assert_graphs):
    inputs = ('CCC(C)=NO[2H]', 'CC/C(C)=N/O[3H]', 'CC/C(C)=N\\O[3H]',
              'CCC(C)=N[18O][2H]', '[2H]ON=CC=NO', '[2H]ON=CC=NO[2H]',
              '[3H]ON=C1CCC(O)CC1', 'CCOC(=O)CC(=NO[2H])CC')
    graphs = [Chem.MolFromSmiles(s) for s in inputs]
    reverse = [Chem.MolToSmiles(Chem.RenumberAtoms(m, list(reversed(range(m.GetNumAtoms())))),
                                canonical=False) for m in graphs]
    names = naming.names(inputs)
    assert names == naming.names(reverse)
    # OPSIN 2.9 needs the single hydroxyl hydrogen count written explicitly.
    # Expand only that default count in the name; the parser still assigns
    # the isotope to its site, reconstructs the graph and resolves its stereo.
    names = [re.sub(r'\(([23])H(,\d+O)?\)(?=hydroxy)', r'(\1H1\2)', n) for n in names]
    assert_graphs(inputs, names, raw_only=True)
