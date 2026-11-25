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
    # Independently paired functional ends. Source positions distinguish an
    # external carbonyl isotope, ether-bound stereocentres and alkene geometry.
    ('O=[13C](O)c1ccc(Oc2ccc([13C](=O)O)cc2)cc1',
     'O=[13C](O)c1ccc(N2CCN(c3ccc([13C](=O)O)cc3)CC2)cc1',
     'O=C(O)/C=C/CCOCC/C=C/C(=O)O', 'O=C(O)[C@@H](O)COC[C@H](O)C(=O)O',
     'C[C@@H](O[C@H](C)C(=O)O)C(=O)O', 'C[C@H](S[C@H](C)C(=O)O)C(=O)O',
     'CCNC(=O)COCC(=O)NC'),
    # The central ring, its outward arms, and the two terminal parents each
    # own different atoms. Include both faces and an unequal pair of arms.
    ('N#C[C@@H](F)CCO[C@@H]1CC[C@@H](OCC[C@H](F)C#N)CC1',
     'N#C[C@@H](F)CCO[C@@H]1CC[C@H](OCC[C@H](F)C#N)CC1',
     'O=C(O)c1ccc(OC(F)COc2ccc(OCC(F)Oc3ccc(C(=O)O)cc3)cc2)cc1',
     'CCNC(=O)c1cncc(OC(F)COc2ccc(OCC(Cl)Oc3cncc(C(=O)NCC)c3)cc2)c1',
     'CCNC(=O)COc1cc(OCC(=O)NCC)cc(OCC(=O)NCC)c1',
     'C[C@H](O)c1cc(N2CCN(c3ccc(C#N)c([C@H](C)O)c3)CC2)ccc1C#N'),
    # Independently drawn repeated ketones, aldehydes and amines. Labels
    # select source atoms even when a ring or linker contains the same element.
    ('O=C1CCC(OC2CCC(=O)CC2)CC1', 'CC(=[18O])COCC(C)=[18O]',
     'O=CC(F)COCC(F)C=O', '[18O]=[13CH]c1cncc(Sc2cncc([13CH]=[18O])c2)c1',
     'O=C/C=C/CCOCC/C=C/C=O', 'O=CCOCC(=O)O'),
    ('NCC(F)COCC(F)CN', 'CNCC(F)COCC(F)CNC',
     '[15NH2]CC(F)COCC(F)C[15NH2]', 'c1ccccc1NCCOCCNc1ccccc1',
     'NC[C@H](F)COC[C@H](F)CN', 'NCC(F)COC(F)CCN'),
    ('O=C1CC(OC2CC(=O)CC(=O)C2)CC(=O)C1',
     'O=Cc1cc(C=O)cc(Oc2cc(C=O)cc(C=O)c2)c1', 'Nc1cc(N)cc(Oc2cc(N)cc(N)c2)c1',
     'O=C1CCC(OC2CCC(=O)CC2=O)C(=O)C1', 'O=Cc1ccc(Oc2ccc(C=O)cc2C=O)c(C=O)c1',
     'O=C1CCC(OC2CC(=O)CC(=O)C2)CC1'),
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


def test_unequally_labelled_ends_preserve_the_two_isotope_positions(naming, assert_graphs):
    original = Chem.MolFromSmiles('NC(=O)C[13CH2]OC[13CH2]C(N)=O')
    reversed_mol = Chem.RenumberAtoms(original, tuple(reversed(range(original.GetNumAtoms()))))
    inputs = [Chem.MolToSmiles(m, canonical=False) for m in (original, reversed_mol)]
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


def test_bromine_oxygen_counts_and_equivalent_drawings(naming):
    # RDKit rejects OPSIN's neutral Br(V) output on this N. The public name
    # must retain the oxidation-state distinction across equivalent drawings.
    series = naming.names(('CNBr', 'CN[Br+][O-]', 'CN[Br+2]([O-])[O-]', 'CN[Br+3]([O-])([O-])[O-]'))
    assert all(series) and len(set(series)) == 4
    assert naming.names(('CNBr(=O)=O', 'CN[Br+](=O)[O-]')) == [series[2], series[2]]
