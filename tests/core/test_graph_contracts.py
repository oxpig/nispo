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

import json
from pathlib import Path

import pytest
from rdkit import Chem


def _structures(mode, controls):
    fixture = Path(__file__).parents[1] / 'data' / 'development_graphs.json'
    source = json.loads(fixture.read_text())["cohorts"]['surechembl_principal_group_roundtrip_20261113.json'][mode]
    inputs = [row['smiles'] for row in source]
    inputs.extend(controls)
    for smiles in tuple(inputs):
        mol = Chem.MolFromSmiles(smiles)
        reverse = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        inputs.append(Chem.MolToSmiles(reverse, canonical=False, isomericSmiles=True))
    return inputs


@pytest.mark.full_only
def test_full_principal_groups_conserve_bonds_and_configuration(naming, assert_graphs):
    # Independent controls cover N substitution, other ring unsaturation
    # states, a second acyl chain length, and an absolute stereo pair.
    inputs = _structures('full', (
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
    inputs = _structures('fast', (
        'C[C@H](/C=C/S(C)(=O)=O)NC(=O)c1cccc(C2CC2)c1',
        'c1ccc(/C=C/c2ccc(/C=C\\Cl)cc2)cc1',
        'C[C@@H](O)CO[C@H](F)c1ccc(/C=C/c2ccccc2)cc1',
    ))
    assert_graphs(inputs, naming.names(inputs), raw_only=True)
