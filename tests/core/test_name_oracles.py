"""Related release contracts consolidated without dropping cases or assertions.

Original module provenance is retained in each section below.
"""


# Originally test_acid_name_oracle.py
"""Synthetic name-part tests for the independent acid-H decoder.

These are parser-infrastructure cases, not generated NISPO names or PIN
expectations. The callback supplies independently chosen small graph parts.
"""
from collections import Counter

import pytest
from rdkit import Chem

from _name_oracles.acid_hydrogen import recover_acid_hydrogen_isotope


@pytest.mark.parametrize("mass", [2, 3])
@pytest.mark.parametrize("stem,suffix", [("component-", "carboxylic acid"), ("ethan", "oic acid")])
def test_one_acid_hydrogen_isotope_is_installed_without_changing_constitution(mass, stem, suffix):
    seen = []

    def parser(name):
        seen.append(name)
        return "CC(=O)O"

    output = recover_acid_hydrogen_isotope(f"{stem}({mass}H){suffix}", parser)
    graph = Chem.MolFromSmiles(output)
    assert seen == [stem + suffix]
    assert graph is not None
    assert Counter((a.GetAtomicNum(), a.GetIsotope()) for a in graph.GetAtoms() if a.GetIsotope()) == {
        (1, mass): 1
    }
    isotope = next(a for a in graph.GetAtoms() if a.GetIsotope())
    assert isotope.GetAtomicNum() == 1 and isotope.GetNeighbors()[0].GetAtomicNum() == 8
    isotope.SetIsotope(0)
    assert Chem.MolToSmiles(Chem.RemoveHs(graph)) == Chem.MolToSmiles(Chem.MolFromSmiles("CC(=O)O"))


def test_acid_hydrogen_is_distinguished_from_an_unrelated_alcohol_hydrogen():
    output = recover_acid_hydrogen_isotope("component-(2H)carboxylic acid", lambda _: "OCC(=O)O")
    graph = Chem.MolFromSmiles(output)
    assert graph is not None
    hydrogen = next(a for a in graph.GetAtoms() if a.GetIsotope())
    oxygen = hydrogen.GetNeighbors()[0]
    carbon = next(a for a in oxygen.GetNeighbors() if a.GetAtomicNum() == 6)
    assert any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(carbon).GetAtomicNum() == 8
               for b in carbon.GetBonds())


def test_parser_owned_molecule_and_its_properties_are_not_modified():
    base = Chem.MolFromSmiles("CC(=O)O")
    base.SetProp("parser_owned", "retain")
    before = base.ToBinary(Chem.PropertyPickleOptions.AllProps)
    output = recover_acid_hydrogen_isotope("[1,1'-bi(component)]-2-(2H)carboxylic acid", lambda _: base)
    assert output is not None
    assert base.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


@pytest.mark.parametrize("name", [
    "component-(4H)carboxylic acid", "component-(2H2)carboxylic acid",
    "component-(2H)(2H)carboxylic acid", "component-(2H,18O)carboxylic acid",
    "component-(2H)ol", "component-(2H)carboxylate", "component-(2H)dicarboxylic acid",
    "(13C)component-(2H)carboxylic acid", None,
])
def test_unsupported_descriptor_grammar_does_not_invoke_the_base_parser(name):
    def forbidden(_):
        pytest.fail("Rejected grammar must not reach the independent parser")

    assert recover_acid_hydrogen_isotope(name, forbidden) is None


@pytest.mark.parametrize("base", [
    "O=C(O)CC(=O)O", "CC(=O)OC", "CCO", "O", "CC(=O)[O-]",
    "[13CH3]C(=O)O", "C[C@H](O)C(=O)O", "[CH2]C(=O)O", "CC(=O)O.O",
])
def test_ambiguous_or_outside_scope_base_graphs_are_not_guessed(base):
    assert recover_acid_hydrogen_isotope("component-(2H)carboxylic acid", lambda _: base) is None


def test_unparsed_or_query_bases_are_not_reconstructed():
    name = "component-(2H)carboxylic acid"
    assert recover_acid_hydrogen_isotope(name, lambda _: None) is None
    assert recover_acid_hydrogen_isotope(name, lambda _: Chem.MolFromSmarts("CC(=O)O")) is None


# Originally test_ester_name_oracle.py
"""Synthetic contract checks for the independent mixed-ester supplement.

Names are hand-authored parser inputs from the general functional-class
grammar, not expected strings from NISPO. No naming engine is imported.
"""
import pytest
from rdkit import Chem

from _name_oracles.mixed_esters import recover_mixed_diyl_ester


COMPONENTS = {
    'ethane-1,2-diol': 'OCCO',
    'propane-1,3-diol': 'OCCCO',
    'propane-1,2-diol': 'OCC(O)C',
    'butane-1,3-diol': 'OCCC(O)C',
    'acetate': 'CC(=O)[O-]',
    'propanoate': 'CCC(=O)[O-]',
    'butanoate': 'CCCC(=O)[O-]',
}


@pytest.mark.parametrize(
    'name,expected',
    (
        ('ethane-1,2-diyl acetate propanoate', 'CC(=O)OCCOC(=O)CC'),
        ('propane-1,3-diyl acetate butanoate', 'CC(=O)OCCCOC(=O)CCC'),
    ),
)
def test_symmetric_diol_assignments_have_one_graph_identity(name, expected):
    result = recover_mixed_diyl_ester(name, COMPONENTS.get)
    assert result is not None
    assert result == Chem.MolToSmiles(Chem.MolFromSmiles(expected), isomericSmiles=True)


@pytest.mark.parametrize('diol', ('propane-1,2-diyl', 'butane-1,3-diyl'))
def test_unspecified_asymmetric_sites_are_declined(diol):
    assert recover_mixed_diyl_ester(diol+' acetate propanoate', COMPONENTS.get) is None


def test_component_parser_molecules_are_not_mutated():
    molecules = {name: Chem.MolFromSmiles(text) for name, text in COMPONENTS.items()}
    before = {name: mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for name, mol in molecules.items()}
    calls = []

    def parser(name):
        calls.append(name)
        return molecules.get(name)

    assert recover_mixed_diyl_ester('ethane-1,2-diyl acetate propanoate', parser)
    assert calls == ['ethane-1,2-diol', 'acetate', 'propanoate']
    assert before == {name: mol.ToBinary(Chem.PropertyPickleOptions.AllProps) for name, mol in molecules.items()}


@pytest.mark.parametrize('name', (
    'ethane-1,2-diyl diacetate',
    'ethane-1,2-diyl diacetate propanoate',
    'ethane-1,2-diyl acetate acetate',
    '(1R)-propane-1,2-diyl acetate propanoate',
    'ethane-1,2-diyl 1-acetate 2-propanoate',
    'ethane-1,1-diyl acetate propanoate',
    'ethane-1,3-diyl acetate propanoate',
    'benzene-1,4-diyl acetate propanoate',
    'ethane-1,2-diyl acetate',
))
def test_out_of_scope_names_are_not_sent_to_component_parser(name):
    def forbidden(_name):
        raise AssertionError('Out-of-scope grammar must not be rewritten')

    assert recover_mixed_diyl_ester(name, forbidden) is None


@pytest.mark.parametrize('replacement', (
    None, 'CC(=O)O', 'O=C([O-])CC(=O)[O-]', '[13CH3]C(=O)[O-]',
    'C[C@H](O)C(=O)[O-]', 'CC(=O)[O-].[Na+]',
))
def test_missing_neutral_polyacid_labelled_stereo_and_salt_components_are_declined(replacement):
    def parser(name):
        return replacement if name == 'acetate' else COMPONENTS.get(name)

    assert recover_mixed_diyl_ester('ethane-1,2-diyl acetate propanoate', parser) is None


@pytest.mark.parametrize('diol', ('CCO', 'COCO', 'OC(O)C', 'OCC(O)O'))
def test_missing_extra_or_coincident_hydroxy_sites_are_declined(diol):
    def parser(name):
        return diol if name == 'ethane-1,2-diol' else COMPONENTS.get(name)

    assert recover_mixed_diyl_ester('ethane-1,2-diyl acetate propanoate', parser) is None


# Originally test_peroxolate_name_oracle.py
"""Small independent salt-name-part proofs; no NISPO output snapshots."""
from collections import Counter

import pytest
from rdkit import Chem

from _name_oracles.peroxolate_salts import recover_alkali_peroxolate_salt


@pytest.mark.parametrize("metal,ion", [("lithium", "[Li+]"), ("sodium", "[Na+]"), ("potassium", "[K+]")])
def test_binary_light_alkali_peroxolate_has_exact_hydrogen_and_charge_balance(metal, ion):
    calls = []

    def parser(name):
        calls.append(name)
        return {"ethaneperoxol": "CCOO", metal + "(1+)": ion}.get(name)

    output = recover_alkali_peroxolate_salt(metal + " ethaneperoxolate", parser)
    graph = Chem.MolFromSmiles(output)
    assert graph is not None
    assert calls == ["ethaneperoxol", metal + "(1+)"]
    assert Counter(a.GetFormalCharge() for a in graph.GetAtoms()) == {-1: 1, 0: 3, 1: 1}
    assert sum(a.GetTotalNumHs(includeNeighbors=True) for a in graph.GetAtoms()) == 5
    assert len(Chem.GetMolFrags(graph)) == 2
    charged_oxygen = next(a for a in graph.GetAtoms() if a.GetFormalCharge() == -1)
    assert charged_oxygen.GetAtomicNum() == 8 and charged_oxygen.GetDegree() == 1
    assert charged_oxygen.GetNeighbors()[0].GetAtomicNum() == 8


def test_parser_owned_graphs_and_properties_are_preserved():
    base, cation = Chem.MolFromSmiles("CCOO"), Chem.MolFromSmiles("[Na+]")
    base.SetProp("owner", "parser")
    cation.SetProp("owner", "parser")
    before = [g.ToBinary(Chem.PropertyPickleOptions.AllProps) for g in (base, cation)]
    output = recover_alkali_peroxolate_salt(
        "sodium ethaneperoxolate", lambda name: base if name == "ethaneperoxol" else cation
    )
    assert output is not None
    assert before == [g.ToBinary(Chem.PropertyPickleOptions.AllProps) for g in (base, cation)]


@pytest.mark.parametrize("name", [
    None, "calcium ethaneperoxolate", "disodium ethaneperoxolate",
    "sodium (2H)ethaneperoxolate", "sodium ethaneperoxolate hydrate",
    "sodium ethaneperoxide", "sodium ethaneperoxol", "sodium ethaneperoxolateperoxolate",
    "sodium ethane-1,2-bis(peroxolate)", "sodium [ethane]peroxolate",
])
def test_unsafe_or_nonbinary_name_grammar_never_invokes_the_parser(name):
    def forbidden(_):
        pytest.fail("Unsafe grammar must not reach the independent parser")

    assert recover_alkali_peroxolate_salt(name, forbidden) is None


@pytest.mark.parametrize("base", [
    "CCO", "COOC", "COOCCOO", "C1CC1OO", "CCO[O-]", "NCCOO", "[13CH3]COO",
    "[CH3:1]COO", "C[C@H](CC)COO", "[CH2]COO", "CCOO.O", "C/C=C/COO", "C->COO",
])
def test_ambiguous_or_annotated_peroxol_base_graphs_are_declined(base):
    def parser(name):
        return base if name == "ethaneperoxol" else "[Na+]"

    assert recover_alkali_peroxolate_salt("sodium ethaneperoxolate", parser) is None


@pytest.mark.parametrize("cation", ["[Na]", "[Na-]", "[K+]", "[Na+].[Na+]", "[23Na+]", "[Na+:1]"])
def test_cation_charge_identity_and_component_count_are_verified(cation):
    def parser(name):
        return "CCOO" if name == "ethaneperoxol" else cation

    assert recover_alkali_peroxolate_salt("sodium ethaneperoxolate", parser) is None


def test_unparsed_and_query_peroxols_are_not_reconstructed():
    name = "sodium ethaneperoxolate"
    assert recover_alkali_peroxolate_salt(name, lambda _: None) is None
    assert recover_alkali_peroxolate_salt(name, lambda _: Chem.MolFromSmarts("CCOO")) is None


# Originally test_functional_name_oracles.py
"""Parser-operation contracts, independent of NISPO naming and panel data.

Synthetic component labels isolate grammar and graph conservation. The
existing independently authored functional boundaries exercise real OPSIN.
"""
import pytest
from rdkit import Chem

from _name_oracles.functional_operations import (
    recover_amine_dianion, recover_symmetric_dianhydride)


def test_primary_amine_loses_two_hydrons_without_mutating_parser_graph():
    base = Chem.MolFromSmiles('CCN')
    before = base.ToBinary(Chem.PropertyPickleOptions.AllProps)
    requested = []

    def parse(part):
        requested.append(part)
        return base

    graph = recover_amine_dianion('exampleaminediide', parse)
    assert requested == ['exampleamine']
    assert Chem.MolToSmiles(Chem.MolFromSmiles(graph)) == 'CC[N-2]'
    assert base.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


@pytest.mark.parametrize('base', ('CNC', 'CN(C)C', 'NCCN', '[NH3+]CC', '[13CH3]CN', 'C[C@H](F)N', None))
def test_dianion_decoder_declines_ambiguous_or_annotated_bases(base):
    assert recover_amine_dianion('exampleaminediide', lambda _: base) is None


def test_two_anhydride_links_conserve_graph_and_parser_ownership():
    parts = {'end acid': Chem.MolFromSmiles('CCC(=O)O'),
             'middle acid': Chem.MolFromSmiles('O=C(O)CCC(=O)O')}
    before = {key: graph.ToBinary(Chem.PropertyPickleOptions.AllProps) for key, graph in parts.items()}
    graph = recover_symmetric_dianhydride('diend middle dianhydride', parts.get)
    expected = Chem.MolFromSmiles('CCC(=O)OC(=O)CCC(=O)OC(=O)CC')
    assert Chem.MolToSmiles(Chem.MolFromSmiles(graph)) == Chem.MolToSmiles(expected)
    assert {key: value.ToBinary(Chem.PropertyPickleOptions.AllProps) for key, value in parts.items()} == before


@pytest.mark.parametrize('end,middle', (
    ('CC(=O)[O-]', 'O=C(O)CC(=O)O'),
    ('CC(=O)O', 'CC(=O)O'),
    ('CC(=O)O', 'O=C(O)C(O)C(=O)O'),
    ('OC(=O)CC(=O)O', 'O=C(O)CC(=O)O'),
    ('[13CH3]C(=O)O', 'O=C(O)CC(=O)O'),
    ('CC(=O)O', 'O=C(O)C=CC(=O)O'),
    ('CC(=O)O', 'O=C(O)C1CCC1C(=O)O'),
    (None, 'O=C(O)CC(=O)O'),
))
def test_dianhydride_decoder_declines_unsupported_components(end, middle):
    assert recover_symmetric_dianhydride(
        'diend middle dianhydride', {'end acid': end, 'middle acid': middle}.get) is None


@pytest.mark.parametrize('decoder,name', (
    (recover_amine_dianion, '(2R)-exampleaminediide'),
    (recover_amine_dianion, 'exampleaminium'),
    (recover_symmetric_dianhydride, 'end middle dianhydride'),
    (recover_symmetric_dianhydride, 'diend middle trianhydride'),
    (recover_symmetric_dianhydride, 'di(1R)-end middle dianhydride'),
))
def test_unsupported_grammar_never_calls_the_component_parser(decoder, name):
    def forbidden(_part):
        raise AssertionError('Out-of-scope grammar reached the parser')

    assert decoder(name, forbidden) is None


# Originally test_numbered_cip_name_oracle.py
"""Independent parser infrastructure: hand-written names, no naming snapshots."""
import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler
from _name_oracles.numbered_stereo import verify_names


def test_multiplied_component_stereo_is_not_assigned_to_only_one_copy(assert_graphs):
    # This decoder deliberately declines multiple occurrences. Raw OPSIN
    # owns them until an all-occurrences name-only proof is implemented.
    name='bis[(2R)-2-fluoropropan-1-yl] ether'
    proof=verify_names([name],include_uppercase=True).get(name)
    assert proof is not None and not proof['unique']


def test_numbered_monocycle_geometries_are_unique_and_conserve_constitution(assert_graphs):
    names=[f'(1{d})-1-ethylcyclooct-1-ene' for d in ('E','Z')]
    proofs=verify_names(names,include_uppercase=True)
    graphs=[]
    for name,descriptor in zip(names,('E','Z')):
        assert proofs[name]['unique']
        graph=Chem.MolFromSmiles(proofs[name]['graph'])
        rdCIPLabeler.AssignCIPLabels(graph)
        assert [b.GetProp('_CIPCode') for b in graph.GetBonds() if b.HasProp('_CIPCode')]==[descriptor]
        graphs.append(graph)
    assert Chem.MolToSmiles(graphs[0])!=Chem.MolToSmiles(graphs[1])
    for graph in graphs:Chem.RemoveStereochemistry(graph)
    assert Chem.MolToSmiles(graphs[0])==Chem.MolToSmiles(graphs[1])


@pytest.mark.parametrize('name', [
    '(1E)-cycloocta-1,3-diene', '(1E)-1-ethylcyclooct-1-en-3-ol',
    '(2E)-1-ethylcyclooct-1-ene', '(2Z)-1-ethylcyclooct-1-ene',
])
def test_other_geometric_scopes_decline_without_a_partial_reconstruction(name,assert_graphs):
    assert name not in verify_names([name],include_uppercase=True)


def test_numbered_ol_isotope_precedes_whole_graph_cip_assignment(assert_graphs):
    name='(2R)-2-hydroxybutan-1-(2H)ol'
    proof=verify_names([name],include_uppercase=True)[name]
    assert proof['unique']
    mol=Chem.MolFromSmiles(proof['graph']);rdCIPLabeler.AssignCIPLabels(mol)
    assert [a.GetProp('_CIPCode') for a in mol.GetAtoms() if a.HasProp('_CIPCode')]==['R']
    isotope,=[a for a in mol.GetAtoms() if a.GetIsotope()]
    assert isotope.GetAtomicNum()==1 and isotope.GetIsotope()==2
    oxygen=isotope.GetNeighbors()[0]
    assert oxygen.GetAtomicNum()==8
    carbon,=[a for a in oxygen.GetNeighbors() if a.GetAtomicNum()==6]
    assert carbon.GetTotalNumHs()==2


# Originally test_name_oracle_contracts.py
"""Small independent checks for the bounded supplementary parser itself.

The names here are hand-authored oracle inputs following general P-82 rules,
not snapshots of generated NISPO wording.  No NISPO naming code is called.
"""

import pytest
from rdkit import Chem

from _name_oracles import recover_ol_hydrogen_isotope


@pytest.mark.parametrize("mass", (2, 3))
def test_single_ol_hydrogen_isotope_is_placed_on_oxygen_without_mutating_base(mass):
    base = Chem.MolFromSmiles("CO")
    before = base.ToBinary(Chem.PropertyPickleOptions.AllProps)
    requested = []

    def parse_base(name):
        requested.append(name)
        return base

    recovered = recover_ol_hydrogen_isotope(f"methan({mass}H)ol", parse_base)
    assert requested == ["methanol"]
    graph = Chem.MolFromSmiles(recovered)
    assert graph is not None
    labelled = [atom for atom in graph.GetAtoms() if atom.GetIsotope()]
    assert len(labelled) == 1
    assert labelled[0].GetAtomicNum() == 1 and labelled[0].GetIsotope() == mass
    assert len(list(labelled[0].GetNeighbors())) == 1
    assert next(iter(labelled[0].GetNeighbors())).GetAtomicNum() == 8
    assert Chem.MolToSmiles(graph) == Chem.MolToSmiles(Chem.MolFromSmiles(f"[{mass}H]OC"))
    assert base.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


@pytest.mark.parametrize("parsed", ("OCCO", "COC", None, "[2H]OC"))
def test_ol_hydrogen_reconstruction_declines_ambiguous_missing_or_already_labelled_base(parsed):
    assert recover_ol_hydrogen_isotope("methan(2H)ol", lambda _name: parsed) is None


@pytest.mark.parametrize("name", ("methan(1H)ol", "methan(4H)ol", "methan(2H2)ol", "(1R)-ethan(2H)ol"))
def test_out_of_scope_ol_descriptor_is_not_rewritten_or_sent_to_the_parser(name):
    def forbidden(_name):
        raise AssertionError("Out-of-scope grammar must be declined before base parsing")

    assert recover_ol_hydrogen_isotope(name, forbidden) is None


def test_unique_hydroxy_prefix_hydrogen_uses_the_same_oxygen_proof():
    requested=[]
    def parse(name):
        requested.append(name)
        return 'CCC(O)C(N)=O'
    result=recover_ol_hydrogen_isotope('2-(2H)hydroxybutanamide',parse)
    assert requested==['2-hydroxybutanamide']
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result))==Chem.MolToSmiles(Chem.MolFromSmiles('[2H]OC(CC)C(N)=O'))


# Originally test_phospho_name_oracle.py
"""Independent synthetic prefix semantics; no production naming calls."""
import pytest
from rdkit import Chem
from _name_oracles.phospho_hydrogen import recover_neutral_diphosphooxy


def test_neutral_prefix_retains_three_acid_hydrogens_and_the_named_cation():
    raw=Chem.MolFromSmiles('C[N+](C)(C)CCOP(=O)([O-])OP(=O)(O)O')
    before=raw.ToBinary(Chem.PropertyPickleOptions.AllProps)
    graph=recover_neutral_diphosphooxy('2-diphosphooxyethyltrimethylazanium',lambda _:raw)
    expected=Chem.MolFromSmiles('C[N+](C)(C)CCOP(=O)(O)OP(=O)(O)O')
    assert Chem.MolToSmiles(Chem.MolFromSmiles(graph))==Chem.MolToSmiles(expected)
    assert raw.ToBinary(Chem.PropertyPickleOptions.AllProps)==before


@pytest.mark.parametrize('raw',[
    'C[N+](C)(C)CCOP(=O)(O)OP(=O)(O)O',
    'C[N+](C)(C)CCOP(=O)([O-])OP(=O)([O-])O',
    'C[N+](C)(C)CCOP(=O)([O-])O',
    'CCOP(=O)([O-])OP(=O)(O)O',
])
def test_ambiguous_or_out_of_scope_charge_inventory_declines(raw):
    assert recover_neutral_diphosphooxy('2-diphosphooxyethyltrimethylazanium',lambda _:raw) is None
