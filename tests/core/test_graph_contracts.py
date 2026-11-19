"""Related release contracts consolidated without dropping cases or assertions.

Original module provenance is retained in each section below.
"""


# Originally test_ring_contracts.py
"""Small, independent ring graphs protect graph recovery, not name spelling.

Provenance: these controls were authored for the test revamp from elementary
ring topology and general nomenclature categories.  They were not selected
from the Blue Book panel, its result files, or reference-name disagreements.
Each group is deliberately bounded; complete development corpora remain in
the separately selected integration evaluation.
"""

import pytest
from rdkit import Chem


RING_FAMILIES = (
    pytest.param(
        ("C1CC1", "C1CCC1", "C1CCCC1", "C1CCCCC1", "C1CCCCCC1", "C1CCCCCCC1"),
        id="saturated-monocycles",
    ),
    pytest.param(
        ("C1=CCC1", "C1=CCCC1", "C1=CCCCC1", "C1=CCCCCC1", "C1=CC=CCC1", "C1=CC=CCCC1"),
        id="unsaturated-monocycles",
    ),
    pytest.param(
        ("c1ccccc1", "Cc1ccccc1", "Fc1ccccc1", "Clc1ccccc1", "COc1ccccc1", "Oc1ccccc1"),
        id="aromatic-substituents",
    ),
    pytest.param(
        ("C1CCOC1", "C1COCCO1", "C1CCNCC1", "C1CCSCC1", "C1CNCCN1", "C1COCCN1"),
        id="saturated-heterocycles",
    ),
    pytest.param(
        ("c1ccncc1", "c1ccnnc1", "n1ccncc1", "c1ncncn1", "c1ccoc1", "c1ccsc1", "c1cc[nH]c1", "c1cn[nH]c1"),
        id="aromatic-heterocycles",
    ),
    pytest.param(
        ("c1ccc2ccccc2c1", "C1CCC2CCCCC2C1", "C1CCC2(CC1)CCCC2", "C1CC2CCC1C2", "C1CC2CC1C2", "C1CCC2CC1CCC2"),
        id="fused-spiro-and-bridged-topologies",
    ),
    pytest.param(
        ("O=C1CCCCC1", "OC1CCCCC1", "NC1CCCCC1", "CC1CCCCC1", "CC1CCC(C)CC1", "O=C(O)C1CCCCC1"),
        id="ring-characteristic-groups",
    ),
)


RING_BOUNDARIES = (
    pytest.param(
        ("O1CCCCCCCCCC1", "N1CCCCCCCCCC1", "S1CCCCCCCCCC1", "O1CCCCOCCCCC1"),
        id="large-replacement-monocycles",
    ),
    pytest.param(
        ("O=C1CCC2CC1CC2", "OC1CCC2CC1CC2", "O=C(O)C1CC2CCC1C2", "O=C1CCC2CC1CC2=O"),
        id="bridged-characteristic-groups",
    ),
    pytest.param(
        ("C1CCC2(CC1)CCC1(CC2)CCCC1", "C1CC2(CC1)CCC1(CC2)CCC1", "C1CCC2(C1)CC1CCC2C1", "C1COC2(C1)CC1CCC2C1"),
        id="polyspiro-and-compound-spiro",
    ),
    pytest.param(
        ("O=C(Nc1ccccc1)c1ccccc1", "CC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1", "O=C(NS(=O)(=O)c1ccccc1)c1ccccc1", "O=C(NS(=O)(=O)C)c1ccccc1"),
        id="linked-ring-amide-priority",
    ),
    pytest.param(
        ("O=P(O)(O)OCC1CCCO1", "O=P(O)(O)OCC1COCC1", "O=P(O)(O)OCC1OC(O)CC1", "O=P(O)(O)OCC1OC(N)CC1"),
        id="oxolanyl-methyl-phosphates",
    ),
    pytest.param(
        ("O=S1(=O)NCc2ccccc21", "CN1Cc2ccccc2S1(=O)=O", "O=C1NCCc2ccccc12", "O=C1Nc2ccccc2O1"),
        id="fused-heterones-and-lactams",
    ),
    pytest.param(
        ("O=C1OC(=O)C2C3C(=O)OC(=O)C3C12", "O=C1OC(=O)C2CC3C(=O)OC(=O)C3C12", "O=C1OC(=O)C2CCCCC12", "O=C1OC(=O)CC1"),
        id="anhydride-ring-closures",
    ),
    pytest.param(
        ("CC1Cc2ccccc21", "O=C(O)C1Cc2ccccc21", "c1ccc2[nH]ncc2c1", "O=C1CCc2ccccc2N1"),
        id="small-annulated-boundaries",
    ),
)


@pytest.mark.parametrize("inputs", RING_FAMILIES)
def test_small_ring_families_recover_the_input_graph(naming, assert_graphs, inputs):
    """A single small batch checks each topology without API cross-products."""
    names = naming.names(inputs)
    assert len(names) == len(inputs)
    assert all(names)
    assert_graphs(inputs, names)


@pytest.mark.parametrize("inputs", RING_BOUNDARIES)
def test_ring_construction_boundaries_recover_the_input_graph(naming, assert_graphs, inputs):
    """Four independently built controls reach each previously absent scope.

    The v1 branch report showed no execution in large replacement-ring,
    compound-spiro, ring/chain amide, N-sulfonyl amide, ring-phosphate and
    dianhydride producers.  These are small examples of those general
    structural categories, not samples from their development corpora.
    The positive and adjacent-boundary graphs protect complete recovery;
    no particular producer, preferred parent or rendered wording is required.
    """
    names = naming.names(inputs)
    assert len(names) == len(inputs)
    assert all(names)
    assert_graphs(inputs, names)


def test_ring_atom_traversal_does_not_change_recovered_connectivity(naming, assert_graphs):
    # One monocycle, one heteroaromatic and one bridge exercise serialization
    # boundaries without permuting every graph in every ring family.
    inputs = []
    for text in ("CC1CCCCC1", "c1ccncc1", "C1CC2CCC1C2"):
        mol = Chem.MolFromSmiles(text)
        assert mol is not None
        reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        inputs.extend((text, Chem.MolToSmiles(reversed_mol, canonical=False)))
    names = naming.names(inputs)
    assert len(names) == len(inputs)
    assert all(names)
    # Different valid nomenclature choices are acceptable; connectivity is
    # the public invariant being checked here.
    assert_graphs(inputs, names)


@pytest.mark.full_only
def test_full_cyclic_oxime_ether_preserves_configuration(naming, assert_graphs):
    # An independently constructed substituted indanone oxime ether, with
    # both C=N configurations on the same graph.
    inputs = (
        r"COCCO/N=C1\CCc2cc(OC3CCC(N)CC3)ccc21",
        "COCCO/N=C1/CCc2cc(OC3CCC(N)CC3)ccc21",
    )
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


@pytest.mark.full_only
def test_full_sulfur_substituents_are_preserved_on_esters(naming, assert_graphs):
    inputs = (
        "COC(=O)c1ccc(S(F)(F)(F)(F)F)cc1",
        "CCN(CC)S(=O)(=O)NCC(=O)OC",
        "COC(=O)CCNS(=O)(=O)NCC",
    )
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


@pytest.mark.full_only
def test_full_imidoyl_carbamate_preserves_nested_azide_and_stereo(naming, assert_graphs):
    inputs = (
        "COC(=O)NC(=N)c1ccc(CNC(=O)[C@H]2CC(CN=[N+]=[N-])N2)cc1",
        "CC(C)OC(=O)NC(=N)CC",
    )
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


@pytest.mark.full_only
def test_full_nested_azido_sulfonamide_preserves_original_graph(naming, assert_graphs):
    # A pendant azidomethyl group beyond the sulfonamide N-alkyl arm.
    cases = (
        "Cc1ccc(S(=O)(=O)NCCN2CCC(CN=[N+]=[N-])CC2)cc1",
        "Cc1ccc(S(=O)(=O)NCCN2CC[C@H](CN=[N+]=[N-])C[C@H]2C)cc1",
    )
    assert_graphs(cases, naming.names(cases), raw_only=True)


@pytest.mark.full_only
def test_full_nested_azide_and_sulfonate_graphs(naming, assert_graphs):
    # Independent amine, annular sulfamide, N-acyl ring, and sulfonate graphs.
    cases = (
        "CC(N)c1ccc(N2CC(CN=[N+]=[N-])CC2)cc1",
        "NS(=O)(=O)N1CC(OC)CC(CCN2CCC(CN=[N+]=[N-])CC2)C1",
        "O=C([C@H]1CC(OCN=[N+]=[N-])CCC1)N1CCC(COc2ccc(N3CCOCC3)cc2)CC1",
        "FC(F)(F)S(=O)(=O)Oc1ccc([C@H](O)C)cc1",
    )
    assert_graphs(cases, naming.names(cases), raw_only=True)


@pytest.mark.full_only
@pytest.mark.current_only
def test_full_linked_polycycles_preserve_attachment_atoms(naming, assert_graphs):
    # Independent bromoaryl assemblies join fused O/S rings and bridged aza rings.
    cases = (
        "Brc1cc(-c2ccc3c(ccc4cc5ccsc5cc43)c2)ccc1-c1ccc2c(ccc3cc4ccoc4cc32)c1",
        "Brc1cc(-c2ccc3c(c2)Nc2ccccc2-c2ccccc2-3)ccc1-c1ccc2c(c1)c1cccc3c4ccccc4n2c31",
    )
    assert_graphs(cases, naming.names(cases), raw_only=True)


@pytest.mark.full_only
def test_full_mapped_imino_phosphorus_preserves_charges(naming, assert_graphs):
    cases = ("COC(=O)[C@H](C)N=[P+]([O-])Oc1cccc(OC[C@H]2OC(C)(C)C(O)C2O)c1",)
    assert_graphs(cases, naming.names(cases), raw_only=True)


# Originally test_principal_group_numbering_invariance.py
"""Graph and public molecule API regressions on authorized SureChEMBL controls.

Source release: 10.5281/zenodo.16755947, selection seed 20261112. The source
graphs are canonical isomeric SMILES from the 528-record development audit
(SHA-256 0198ca6f140d3720479515b61c10d9727fc180e9f7b32d9a8aa61aefbc3c6553).
The pre-change naming package digest was
5f46d61c89992142cb03e01194555f1f2ef87a0a1b5cb6ad575da6fac7cc6797.
"""

import random

import pytest
from rdkit import Chem


@pytest.mark.full_only
@pytest.mark.parametrize(('smiles', 'seed'), (
    pytest.param('Cn1cnc(CC(=O)N2CC[C@]34CCN(CC5CC5)[C@H](Cc5ccc(O)cc53)[C@]4(O)CC2)c1',
                 2026111302, id='surechembl-row-3818655'),
    pytest.param('COc1ccc2c(c1)[C@]1(C[C@H]1c1ccc3c(N)nn(C(=O)OC(C)(C)C)c3c1)C(=O)N2C(=O)OC(C)(C)C',
                 2026111301, id='surechembl-row-3420715'),
))
def test_principal_names_preserve_graph_and_atom_order(smiles, seed, naming, assert_graphs):
    source = Chem.MolFromSmiles(smiles)
    order = list(range(source.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    reordered = Chem.RenumberAtoms(source, order)

    names = [naming.molecule_result(mol).name for mol in (source, reordered)]

    assert_graphs([source, reordered], names, raw_only=True)
    assert names[0] == names[1]


# Originally test_principal_group_roundtrip.py
"""Graph conservation for principal-group repairs and fast branch writers."""

import pytest
from rdkit import Chem


def _structures(controls):
    inputs = list(controls)
    for smiles in tuple(inputs):
        mol = Chem.MolFromSmiles(smiles)
        reverse = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        inputs.append(Chem.MolToSmiles(reverse, canonical=False, isomericSmiles=True))
    return inputs


@pytest.mark.full_only
def test_full_principal_groups_conserve_bonds_and_configuration(naming, assert_graphs):
    # Independent controls cover N substitution, other ring unsaturation
    # states, a second acyl chain length, and an absolute stereo pair.
    inputs = _structures((
        'CC(=NC)NCCC(=O)O',
        'CN/C(N)=N\\c1ccc(C(=O)N[C@H](F)C)cc1',
        'CC1(S(=O)(=O)OCC)C=CCCC1',
        'CC1(S(=O)(=O)OC)C=CCC=C1',
        'CCCCCCCCCCCCCCC(=O)N(CC(=O)O)CC(=O)O',
        'NC(=O)[C@H]1CCC[C@@H](F)C1',
    ))
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


@pytest.mark.fast_only
def test_fast_branch_writers_conserve_bonds_and_configuration(naming, assert_graphs):
    # Independent controls exercise multiple annotated components and a
    # stereogenic branch with its own unsaturated ligand.
    inputs = _structures((
        'C[C@H](/C=C/S(C)(=O)=O)NC(=O)c1cccc(C2CC2)c1',
        'c1ccc(/C=C/c2ccc(/C=C\\Cl)cc2)cc1',
        'C[C@@H](O)CO[C@H](F)c1ccc(/C=C/c2ccccc2)cc1',
    ))
    assert_graphs(inputs, naming.names(inputs), raw_only=True)


# Independent atom-conservation control retained from the former cohort module.
@pytest.mark.current_only
def test_chalcogen_rendering_preserves_each_original_atom(naming, assert_graphs):
    # Independent controls for two and three linked sulfur atoms and a
    # terminal carbonyl oxide. Each added chalcogen changes the graph.
    smiles = ['CSSc1ccc(C(=O)O)cc1', 'CSSSc1ccc(C(=O)O)cc1', 'CC(C)=[O+][O-]']
    results = naming.results(smiles)
    assert_graphs(smiles, [result.name for result in results], raw_only=True)

# Self-contained graph regressions retained from test_roundtrip_failure_families.py.
# Original controls, SureChEMBL provenance and atom permutations are unchanged;
# the deleted corpus loaders are not replayed here.
from nispo import smiles_to_iupac_batch


@pytest.mark.fast_only
def test_fast_surechembl_ring_hydrogen_and_stereo_roundtrip(assert_roundtrips):
    from nispo.fast_selection import smiles_to_iupac_fast

    def fast_batch(inputs):
        return [smiles_to_iupac_fast(smiles) for smiles in inputs]

    # SureChEMBL development graphs, Zenodo 16755947, selection seed 20261026.
    # Frozen development baseline 766bad466743; source rows below.
    assert_roundtrips([
        "O=C1Cc2ccccc2N1",  # 805
        "S=C1Cc2ccccc2N1",  # 19060
        r"CC(C)(C)OC(=O)N1CCC2(CC/C(=C\c3ccccc3)C2)C1",  # 1794051
        "NC(=O)/C=C/c1cc(O)c(O)cc1C1CCC(c2cc(O)c(O)cc2/C=C/C(N)=O)C1O",  # 723918
        "CCCCC(=O)N(Cc1ccc(-c2cc(-c3ccccc3)ccc2/C(N)=N/O)cc1)C1(C(=O)OC)CCCCC1",  # 2981004
        (r"C=C1/C(=C\C=C2/CCC[C@]3(C)[C@@H]([C@H](C)/C=C/[C@@H](O)"
         r"C4(/C=C/C(=O)OCC)CC4)CC[C@@H]23)C[C@@H](O)C[C@@H]1O"),  # 4001412
        "COC(=O)C1(O)C[C@@H](OC/C=C/c2ccc(O)c(O)c2)C(O)[C@H](OC/C=C/c2ccc(O)c(O)c2)C1",  # 2944009
        (r"C(=C\c1ccc(N(c2ccc(/C=C/c3ccncc3)cc2)c2ccc(/C=C/c3ccncc3)cc2)cc1)"
         r"/c1ccncc1"),  # 982508
    ], name_batch=fast_batch)


@pytest.mark.full_only
def test_full_surechembl_carboxyalkanoyl_alternative_roundtrips(assert_roundtrips):
    # SureChEMBL development, Zenodo 16755947, seed 20261026,
    # frozen baseline 766bad466743, source row 17987.
    def full_batch(inputs):
        return smiles_to_iupac_batch(inputs, timeout_seconds=None, fallback_to_fast=False)

    assert_roundtrips(["CC(C)C[C@H](NC(=O)C(=O)O)[C@@H](O)CC(=O)O"], name_batch=full_batch)


@pytest.mark.full_only
def test_full_surechembl_ring_without_supported_element_returns_cleanly():
    from nispo import smiles_to_iupac_result

    # SureChEMBL development, Zenodo 16755947, seed 20261026,
    # frozen baseline 766bad466743, source row 240709.
    result = smiles_to_iupac_result("CC(=O)OC1=c2ccccc2=C[IH]1",
                                    timeout_seconds=None, fallback_to_fast=False)
    assert result.generation_status in {"available", "unavailable"}


@pytest.mark.full_only
def test_numbering_and_attachment_families_preserve_structure_after_atom_reordering(assert_roundtrips):
    inputs = []
    representatives = [
        "NC(=O)c1ccc(-c2ccc3c(c2F)CC3)cc1",
        "NC(=O)c1ccc(-c2ccc3[nH]cnc3c2F)cc1",
        "O=C1NC(=C=S)N(CC)c2cc[nH]c21",
        "CC1(C)OC2(C)C(F)CC1CC2O",
    ]
    for smiles in representatives:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        for shift in (0, 2):
            order = list(range(mol.GetNumAtoms()))
            order = order[shift:] + order[:shift]
            for indices in (order, order[::-1]):
                reordered = Chem.RenumberAtoms(mol, indices)
                inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    assert_roundtrips(inputs)


@pytest.mark.full_only
def test_bridged_ketones_alcohols_and_sulfur_ligands_preserve_structure(assert_roundtrips):
    assert_roundtrips([
        "CC12CCC(CC1=O)C2(C)C",
        "CC1(C)C2CCC1(C)C(O)C2",
        "O=C1CCC2CC1CC2",
        "OC1CCC2CC1CC2",
        "O=C1C(O)CC2CC1CC2",
        "OC1C(O)CC2CC1CC2",
        "O=C1CCC2CC1CC2=O",
        "O=C1CCC2C(C)C1CC2",
        "OC1CCC2C(F)C1CC2",
        "Cc1ccc(-c2c(C(C)C)nc(N=S(C)(C)=O)c3cc4[nH]ncc4cc23)cc1",
        "CC(C)c1nc(N=S(C)(=O)c2ccccc2)c2cc3[nH]ncc3cc2c1-c1ccc(F)cc1",
        "CS(N)(=O)=NC(=O)Nc1ccccc1",
        # Controls: heterocyclic, fused, spiro and side-chain alcohol parents
        # exercise the existing naming routes outside suffix promotion.
        "O=C1CCN2CCC1CC2",
        "OC1CCC2CCCCC2C1",
        "OC1CCC2(CC1)CCCC2",
        "OCCC1CC2CCC1C2",
    ])


@pytest.mark.full_only
def test_bridged_functional_stereochemistry_survives_atom_reordering(assert_roundtrips):
    inputs = []
    for smiles in [
        "CC1(C)[C@H]2CC[C@]1(C)[C@@H](O)C2",
        "CC1(C)[C@@H]2CC[C@@]1(C)[C@H](O)C2",
        "O=C1C[C@H]2CC[C@@H]1C2",
        "OC1C(O)CC2CC1CC2",
    ]:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        for shift in (0, 2):
            order = list(range(mol.GetNumAtoms()))
            order = order[shift:] + order[:shift]
            for indices in (order, order[::-1]):
                reordered = Chem.RenumberAtoms(mol, indices)
                inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    assert_roundtrips(inputs)


@pytest.mark.full_only
def test_ionic_bonding_states_survive_atom_reordering(assert_roundtrips):
    inputs = []
    for smiles in [
        "Cc1ccc(B([O-])[O-])cc1",
        "C#C[PH]([O-])(OC)OCC",
        "CCN([SH2](=O)C=N)CC",
        "O=C([N-][n+]1ccc(C)cc1)c1ccccc1",
        "[O-][N+]#Cc1ccc(F)cc1",
        "[C-]1=[C-]C=CC=C1",
        "CCN(C)S(=O)(=O)Nc1ccccc1",
        "C/C(=N\\NCC(=O)O)c1ccccn1",
    ]:
        mol = Chem.MolFromSmiles(smiles)
        for shift in (0, 2):
            order = list(range(mol.GetNumAtoms()))
            order = order[shift:] + order[:shift]
            for indices in (order, order[::-1]):
                reordered = Chem.RenumberAtoms(mol, indices)
                inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    assert_roundtrips(inputs)


@pytest.mark.full_only
def test_parent_and_prefix_rendering_survives_atom_reordering(assert_roundtrips):
    inputs = []
    for smiles in [
        "CC/C=C/CC", "C=CC=CC=C", "C#CC#CC#C",
        "O=C1CCNCC1", "O[C@H]1CC[C@@H](C)CC1", "OC1C(O)CC(O)CC1",
        "FC1CCC(c2ccc(Cl)cc2)CC1", "CCOP(=O)(OCC)[N@]1CC1C",
    ]:
        mol = Chem.MolFromSmiles(smiles)
        for shift in (0, 2):
            order = list(range(mol.GetNumAtoms()))
            order = order[shift:] + order[:shift]
            for indices in (order, order[::-1]):
                reordered = Chem.RenumberAtoms(mol, indices)
                inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    assert_roundtrips(inputs)


@pytest.mark.full_only
def test_principal_oxygen_families_survive_atom_reordering(assert_roundtrips):
    inputs = []
    for smiles in [
        "CC(=O)C[C@H](O)C", "C/C=C/C(=O)CC", "C/C=C\\C(=O)CC",
        "OCC(CO)(CO)CCO", "C[C@H](O)[C@@H](O)CO", "O=C(O)CCC(=O)O",
        "ON1CCCC1", "Oc1ccc2cc(O)ccc2c1", "O=C1CCCC2(CCCC2=O)C1",
    ]:
        mol = Chem.MolFromSmiles(smiles)
        for shift in (0, 2):
            order = list(range(mol.GetNumAtoms()))
            order = order[shift:] + order[:shift]
            for indices in (order, order[::-1]):
                reordered = Chem.RenumberAtoms(mol, indices)
                inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    assert_roundtrips(inputs)


@pytest.mark.full_only
def test_principal_group_followup_survives_atom_reordering(assert_roundtrips):
    inputs = []
    representatives = [
        "O=C[C@H](O)CO", "O=C/C=C/CO", "O=C/C=C\\CO", "O=C(O)C(O)C=O",
        "OCC(O)C(O)C(O)C(O)CO", "OCc1ccc(F)cc1",
        "Oc1cc(F)c2c(c1)CCCC2", "O=C([O-])c1ccc[nH+]c1",
        "O=S(=O)(Nc1ccc(S(=O)(=O)c2ccccc2)cc1)C(F)(F)F",
        "O=c1oc2ccccc2c(O)c1C(C[C@H](O)c2ccccc2)c1ccccc1",
        "CC[C@@H](C)CC[C@H]1[C@@H](C)CC[C@H]2C(C)(C)CC(O)C(O)(O)[C@]12C",
    ]
    for smiles in representatives:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        for shift in (0, 2):
            order = list(range(mol.GetNumAtoms()))
            order = order[shift:] + order[:shift]
            for indices in (order, order[::-1]):
                reordered = Chem.RenumberAtoms(mol, indices)
                inputs.append(Chem.MolToSmiles(reordered, canonical=False, isomericSmiles=True))
    assert_roundtrips(inputs)


# Public graph regressions retained from test_rule_engine.py.


@pytest.mark.full_only
def test_n_acyl_branch_families_preserve_public_graphs_and_inputs(assert_roundtrips):
    from nispo import (mol_to_iupac, mol_to_iupac_result, smiles_to_iupac_batch,
                       smiles_to_iupac_batch_results)

    inputs = (
        "c1cc(NC(=O)OC)ccc1",
        "c1cc(NC(=O)OCC)ccc1",
        "c1cc(NC(=O)C(F)(F)F)ccc1",
        "CC(=O)N1CC2CCC1[C@@H]2N",
        "CC(=O)N1C[C@H]2CC[C@@H]1[C@@H]2N",
        "CC(=O)NCC1=C(F)C=C(F)C=C1",
    )
    assert_roundtrips(inputs)
    names = smiles_to_iupac_batch(inputs)
    assert [r.name for r in smiles_to_iupac_batch_results(inputs)] == names
    assert smiles_to_iupac_batch(reversed(inputs)) == list(reversed(names))
    for smiles, name in zip(inputs, names, strict=True):
        mol = Chem.MolFromSmiles(smiles)
        before = Chem.MolToMolBlock(mol)
        assert mol_to_iupac(mol) == mol_to_iupac_result(mol).name == name
        assert Chem.MolToMolBlock(mol) == before


@pytest.mark.full_only
def test_common_ring_graphs_survive_public_naming_and_atom_reordering(assert_roundtrips):
    """Exercise these ring families without freezing branch spelling."""
    from nispo import smiles_to_iupac_batch

    inputs = ("CC1CCC(F)(F)CC1", "CC1CC(F)(F)C1", "Nc1nc(C)cs1", "CC1=CCOCC1")
    reordered = []
    for smiles in inputs:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        reverse = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        reordered.append(Chem.MolToSmiles(reverse, canonical=False, isomericSmiles=True))
    assert_roundtrips([*inputs, *reordered])
    assert smiles_to_iupac_batch(inputs) == smiles_to_iupac_batch(reordered)
