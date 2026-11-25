"""Public graph checks for P-66.4.1.1, P-66.4.2.1 and P-66.3.3.3.

The final two acylhydrazides are synthetic combinations prompted by SureChEMBL
development rows 1591484 and 2994460 (Zenodo 16755947, seed 20261124, baseline
da85ecc). The other graphs were drawn independently from the published rules.
"""

import pytest
from rdkit import Chem


pytestmark = pytest.mark.current_only


@pytest.mark.parametrize('inputs', [
    pytest.param(('N=C([15NH2])CCC(=[15NH])N', 'CCNC(=N)c1ncc(C(=NC)N)cc1',
                  'NC(=N)C1(C(=N)NC)CCCCC1', 'N=[13C](N)c1ccccc1',
                  'N=C(N)[SiH2][SiH2]C(=N)N', 'NC(=N)C1CC=CCC1',
                  'CS(=NC)(=NCC)N', 'CS(=[Se])(=N)N', 'C[34S](=O)(=N)N',
                  'C/N=C(/N)c1ccccc1', 'C/N=C(\\N)c1ccccc1'), id='amidines'),
    pytest.param(('CC(=NCC)N(C)NC', 'CC(=NNC)NCC', 'NC(=NN)CCC(=N)NN',
                  'NC(=NN)C1CCC(C(=NN)N)CC1', 'CC/C(N)=N/NC', 'CC/C(N)=N\\NC',
                  'CCC(=[15NH])NN', 'CCC(=N[15NH2])N', 'N[13C](=NN)c1ncccc1',
                  'CC[Se](=NN)N', 'CN/N=C(/N)c1ccccc1'), id='amidrazones'),
    pytest.param(('CC(=O)N(N)C(=O)CCC', 'CC(=O)NNC(=O)c1ccc(F)cc1',
                  'CC(=O)NN(C(=O)CC)C(=O)CCCC', 'CC(=O)N(C(=O)CC)N(C(=O)CCC)C(=O)CCCC',
                  'CCC(=O)[15NH]NC(=O)C', 'CCC(=O)N[15NH]C(=O)C',
                  '[2H]N(C(=O)C)NC(=O)CCC', 'CCC(=[18O])NNC(=O)C',
                  'C[C@@H](F)C(=O)NNC(=O)C', 'C[C@H](F)C(=O)NNC(=O)C',
                  'CCS(=O)NNS(=O)c1ccccc1', 'CC(=N)NNC(=O)CCC',
                  'CS(=O)(=O)NNC(=O)CCC(=O)NC'), id='acylhydrazides'),
    pytest.param(('NC(=N)c1ccc(CCc2ccc(C(=N)N)cc2)cc1', 'NC(=N)c1ccc(Cc2ccccc2C(=N)N)cc1',
                  'NC(=N)c1ccc(-c2ccc(C(=N)NC)cc2)cc1', 'NC(=N)c1ccnc(-c2cc(C(=N)N)ccn2)c1',
                  'NC(=N)C1CCC(C2CCC(C(=N)N)CC2)CC1', 'NC(=N)C1CC(C2CCNC(C(=N)N)C2)CCN1',
                  'NC(=N)c1ccc(/C=C/c2ccc(C(=N)N)cc2)cc1', 'NC(=N)c1ccc(C(C)(C)c2ccc(C(=N)N)cc2)cc1',
                  'NC(=N)c1cc(C(=N)N)cc(Oc2cc(C(=N)N)cc(C(=N)N)c2)c1'), id='linked-amidine-rings'),
    pytest.param(('CN/N=C(/N)c1ccc(/C(N)=N\\NC)cc1', 'CCN/N=C(\\N)c1ccc(/C(N)=N\\NC)cc1',
                  'CCNN=C(N)c1ccc(/C(N)=N\\NC)cc1', 'CCN/N=C(\\N)c1ccc(C(N)=NNC)cc1',
                  'CCNN=C(N)C1(/C(N)=N/NC)CCCCC1', 'N/N=C(/NC)C1(/C(NCC)=N/N)CCCCC1',
                  'CN/N=C(/N)c1ccc(-c2ccc(/C(N)=N/NC)cc2)cc1'), id='bis-hydrazonamide-configurations'),
    pytest.param(('CC(=O)NNC(=O)CC(=O)NNC(=O)C', 'CC(=O)N(N)C(=O)CC(=O)N(N)C(=O)C',
                  'CC(=O)NNC(=O)/C=C/C(=O)NNC(=O)C', 'CC(=O)[15NH]NC(=O)CCC(=O)NNC(=O)C',
                  'CC(=O)NNC(=[18O])c1ccc(C(=O)NNC(=O)C)cc1', 'CC(=O)NN[13C](=O)c1ccc(C(=O)NNC(=O)C)cc1',
                  'CC(=O)NNC(=O)C1(C(=O)N(N)C(=O)CCC)CCCCC1',
                  'CC(=O)NNC(=O)C1(C(=[18O])NNC(=O)CC)CCCCC1', 'CC(=O)NNC(=[18O])C1(C(=O)NNC(=O)CC)CCCCC1',
                  'CC(=O)NNC(=O)c1cc(C(=O)NNC(=O)C)cc(C(=O)NNC(=O)C)c1'), id='polyacylhydrazides'),
])
def test_original_graph_and_atom_order(naming, assert_graphs, inputs):
    reversed_smiles = []
    for smiles in inputs:
        mol = Chem.MolFromSmiles(smiles)
        permuted = Chem.RenumberAtoms(mol, tuple(reversed(range(mol.GetNumAtoms()))))
        reversed_smiles.append(Chem.MolToSmiles(permuted, canonical=False))
    names = naming.names(inputs)
    assert names == naming.names(reversed_smiles)
    assert_graphs(inputs, names, raw_only=True)
