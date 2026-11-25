"""Independent graph controls drawn from valence and the published ring construction."""
import pytest
from rdkit import Chem


pytestmark = pytest.mark.current_only


@pytest.mark.full_only
def test_complete_unit_numbering_preserves_structure(naming, assert_graphs):
    inputs = [
        'Cl[SiH]=[SiH2]', 'N/N=N/N', '[SiH]1=[SiH]' + '[SiH2]' * 9 + '1',
        'OC1CCCP1C1CC1', '[PH]1CC[13CH2]CC1', 'F[C@@H]1CCC[SiH2]C1',
        'OCCS(=O)CCO', 'OCCCCP(CCCCO)CCCCO',
        'Oc1ccc(S(=O)(=O)c2cccc(O)c2)cc1',
    ]
    results = naming.results(inputs)
    assert all(r.generation_status == 'available' for r in results)
    assert_graphs(inputs, [r.name for r in results], raw_only=True)


@pytest.mark.full_only
def test_sulfoxide_resonance_forms_recover_the_same_neutral_compound(naming, assert_graphs):
    neutral = 'Oc1ccc(S(=O)c2ccc(O)cc2)cc1'
    separated = 'Oc1ccc([S+]([O-])c2ccc(O)cc2)cc1'
    graphs = [Chem.MolFromSmiles(s) for s in (neutral, separated)]
    assert all(Chem.GetFormalCharge(g) == 0 for g in graphs)
    assert Chem.MolToInchiKey(graphs[0]) == Chem.MolToInchiKey(graphs[1])
    # OPSIN renders the sulfoxide with S=O. Check both names against that
    # explicit bond-order drawing after establishing input equivalence.
    assert_graphs([neutral, neutral], naming.names([neutral, separated]), raw_only=True)


def test_indicated_hydrogen_and_isotope_positions_roundtrip(naming, assert_graphs):
    inputs = ['[SiH]1=[SiH][SiH2]1', '[SiH]1=[SiH]' + '[SiH2]' * 3 + '1',
              'C[13CH2]P1CCCC1', '[13CH3]CP1CCCC1', '[13CH3]C1CCC(C)P1',
              'C[13CH2]OC1CCCP1', '[2H]C([2H])CP1CCCC1', '[3H]CCP1CCCC1', '[2H]CCOC1CCCP1']
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


def test_multiplicative_alcohols_preserve_each_units_configuration(naming, assert_graphs):
    inputs = ['C[C@H](O)CCS(=O)(=O)CC[C@@H](O)C', 'C[C@H](O)CCS(=O)(=O)CC[C@H](O)C',
              'CC(O)CCS(=O)(=O)CC[C@@H](O)C', 'OC/C=C/CS(=O)(=O)C/C=C/CO',
              'OC/C=C/CS(=O)(=O)C/C=C\\CO']
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


def test_short_hydrides_preserve_configuration_and_isotope_position(naming, assert_graphs):
    inputs = ['Cl/[SiH]=[SiH]/Cl', 'Cl/[SiH]=[SiH]\\Cl', 'F[Si@H](Cl)Br',
              'F[Si@@H](Cl)Br', 'F[Si](F)(Cl)Br', 'Cl[29SiH2][SiH3]', 'Cl[SiH2][29SiH3]',
              '[37Cl][SiH2][SiH2]Cl', 'Cl[SiH]([2H])[SiH3]', 'Cl[SiH2][SiH2][2H]',
              'F[C@H]([2H])Cl', 'F[C@@H]([2H])Cl']
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


def test_sole_hydrogen_isotope_is_distinct_and_order_invariant(naming):
    names = naming.names(['F[Si](Cl)(Br)[2H]', '[2H][Si](Br)(F)Cl', 'F[SiH](Cl)Br'])
    assert all(names) and names[0] == names[1] and names[0] != names[2]


def test_trivalent_group13_hydrides_preserve_ligand_attachment(naming, assert_graphs):
    inputs = ['CCC[Al](CCC)CCC', 'CC[Ga](C)CC', 'CC[InH]CC', 'CC(C)C[AlH]CC(C)C',
              'CC(C)[Ga](C)C']
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


def test_aryl_ligands_and_chalcogen_linker_lengths_roundtrip(naming, assert_graphs):
    inputs = ['[AlH](c1ccc(Cl)cc1)c1ccc(Cl)cc1', '[GaH2]c1cccc2ccccc12',
              'C[AlH]c1ccc(F)cc1', 'OCCSSCCO', 'OCCSSSCCO', 'OCC[Se][Se]CCO',
              'CC(C)(O)CCSSCCC(C)(C)O', 'C[C@H](O)CCSSCC[C@@H](O)C']
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


SINGLE_HYDROXYL_ISOTOPES = ('[2H]OC1CCCP1', '[2H]OC1CCPC1', '[3H]OC1CCCP1', 'CO[2H]')


def test_hydroxyl_isotope_names_are_distinct_and_order_invariant(naming):
    inputs = SINGLE_HYDROXYL_ISOTOPES + (
        '[2H][18O]C1CCCP1', '[2H]OC1C(O[2H])CCP1',
        '[2H]OC1CCC(O)C(O)C1', 'Oc1ccc(O[2H])cc1',
    )
    assert len(set(naming.order_invariant_names(inputs))) == len(inputs)


def test_single_hydroxyl_isotope_positions_with_parser_supplement(naming, assert_graphs):
    assert_graphs(SINGLE_HYDROXYL_ISOTOPES, naming.names(SINGLE_HYDROXYL_ISOTOPES))


def test_mixed_ligands_on_group14_chains_recover_the_parent_and_sites(naming, assert_graphs):
    inputs = ['C[SiH](Cl)[SiH3]', 'CC[GeH](Br)[GeH3]', 'C[SiH](Cl)[SiH2]C',
              'CCC[SiH](Cl)[SiH2][SiH3]']
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


def test_characteristic_isotopes_stay_on_their_original_ring_sites(naming, assert_graphs):
    inputs = ['[18OH]C1CCCP1', '[18OH]C1CCPC1', '[18OH]C1C(O)CCP1',
              '[17OH]C1C([18OH])CCP1', '[18OH]C1CCCO1', '[18O]=C1CCCP1']
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


def test_branched_alcohol_units_preserve_bond_geometry_and_connectivity(naming, assert_graphs):
    inputs = ['CC(C)(O)CCS(=O)(=O)CCC(O)(C)C', 'CC(C)(O)CCS(=O)(=O)CC(O)(C)C',
              'OC(C)(C)/C=C/CS(=O)(=O)C/C=C/C(C)(C)O', 'OCC(CCC)C(O)CO']
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


def test_linked_phenols_keep_each_rings_substituents(naming, assert_graphs):
    inputs = ['Oc1c(Br)cc(S(=O)(=O)c2cc(Br)c(O)cc2)cc1',
              'Oc1c(Br)cc(Cc2cc(Cl)c(O)cc2)cc1', 'Oc1c(CC)cc(Oc2cc(CC)c(O)cc2)cc1']
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


@pytest.mark.parametrize(('arms', 'extra_edge', 'substituted'), (
    (4, None, False), (4, 0, False), (4, 1, True), (4, 0, True), (5, 0, True)))
def test_fused_parent_is_invariant_to_order_and_kekule_form(naming, assert_graphs, arms, extra_edge, substituted):
    # Grow two linearly fused hexagons on alternate sides of the central ring.
    graph = Chem.RWMol()
    def vertices(n):
        return [graph.AddAtom(Chem.Atom(6)) for _ in range(n)]
    def edges(path):
        for left, right in zip(path, path[1:]):
            graph.AddBond(left, right, Chem.BondType.AROMATIC)
    center = vertices(2 * arms)
    edges(center + center[:1])
    for arm, (left, right) in enumerate(zip(center[::2], center[1::2])):
        inner, outer = vertices(4), vertices(4)
        edges([left, *inner, right])
        edges([inner[1], *outer, inner[2]])
        if extra_edge is not None and arm < 2:
            edges([outer[extra_edge], *vertices(4), outer[extra_edge + 1]])
        if substituted and arm == 3:
            graph.AddBond(outer[1], graph.AddAtom(Chem.Atom(9)), Chem.BondType.SINGLE)
    graph = graph.GetMol()
    Chem.SanitizeMol(graph)
    source = Chem.MolToSmiles(graph)
    other = Chem.RenumberAtoms(graph, list(reversed(range(graph.GetNumAtoms()))))
    Chem.Kekulize(other, clearAromaticFlags=True)
    alternate = Chem.MolToSmiles(other, canonical=False, kekuleSmiles=True)
    names = naming.names([source, alternate])
    assert names[0] and names[0] == names[1]
    # OPSIN 2.9 disagrees between reflected angular descriptors and declines
    # substituted decagonal centres. Both encodings still have to agree.
    if arms == 4 and (not substituted or extra_edge == 1):
        assert_graphs([source], names[:1], raw_only=True)
