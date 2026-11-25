"""Public graph controls derived independently from P-68.5.3, P-61.3.2.2 and P-51.3.1."""
import pytest
from rdkit import Chem

pytestmark = pytest.mark.current_only


@pytest.mark.parametrize('inputs', [
    ('CCNI(=O)=O', 'CCNI=O', 'CC[15NH]Br', 'C[C@H](F)CNCl', 'CCN(Cl)Br',
     'CCN(Cl)Cl', 'CCCC(=O)N(Cl)Br', '[13CH3]CCNI(=O)=O', 'CCOCCNCl', 'ClNc1ccc(F)cc1'),
    ('COCCPCl', 'FC1=CC=C(SCl)C=C1', 'CC[Se]Br', 'CCOBr',
     'COCCP(Cl)Br', 'CCOI(=O)=O', 'C[13CH2]SCl'),
    ('OCCCOC(Cl)COCC(Cl)OCCCO', 'O=C(O)CP(CC(=O)O)CC(=O)O',
     'O=C(O)CC[Se]CCC(=O)O', 'O=C(O)C1CCC(OC2CCC(C(=O)O)CC2)CC1',
     'NC(=O)c1ccnc(Oc2cc(C(=O)N)ccn2)c1', 'CNS(=O)(=O)c1ccc(Oc2ccc(S(=O)(=O)NC)cc2)cc1',
     'N#Cc1ccc(P(c2ccc(C#N)cc2)c2ccc(C#N)cc2)cc1', 'OCCOC(F)COCC(Cl)OCCO', 'NC(=O)c1ccc(Sc2cc(F)c(C(=O)N)cc2)cc1'),
])
def test_derivative_and_multiplicand_graphs(naming, assert_graphs, inputs):
    reversed_inputs = []
    for smiles in inputs:
        mol = Chem.MolFromSmiles(smiles)
        other = Chem.RenumberAtoms(mol, tuple(reversed(range(mol.GetNumAtoms()))))
        reversed_inputs.append(Chem.MolToSmiles(other, canonical=False))
    names = naming.names(inputs)
    assert names == naming.names(reversed_inputs)
    assert_graphs(inputs, names, raw_only=True)


def test_bromine_oxygen_counts_and_equivalent_drawings(naming):
    # RDKit rejects OPSIN's neutral Br(V) output on this N. The public name
    # must retain the oxidation-state distinction across equivalent drawings.
    series = naming.names(('CNBr', 'CN[Br+][O-]', 'CN[Br+2]([O-])[O-]', 'CN[Br+3]([O-])([O-])[O-]'))
    assert all(series) and len(set(series)) == 4
    assert naming.names(('CNBr(=O)=O', 'CN[Br+](=O)[O-]')) == [series[2], series[2]]
