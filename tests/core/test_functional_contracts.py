"""Related release contracts consolidated without dropping cases or assertions.

Original module provenance is retained in each section below.
"""
from __future__ import annotations


# Originally test_chain_contracts.py
"""Small graph contracts for independent chain and bond-order controls.

Provenance: these structures were authored directly from elementary carbon
valence, branching, bond-order and heteroatom patterns for this test revamp.
They were not selected from a benchmark panel, source-name list or naming
outcomes. Each family changes one observable chemical feature; there is no
Cartesian expansion of substituents or atom permutations.
"""


import pytest
from rdkit import Chem


CHAIN_FAMILIES = (
    pytest.param(("C", "CC", "CCCC", "CCCCCCCCCCC"), id="chain-length"),
    pytest.param(
        ("CC(C)C", "CCC(C)C", "CC(C)(C)C", "CC(C)CC(C)C"),
        id="branching",
    ),
    pytest.param(
        ("C=C", "CC=C", "C=CC=C", "CC=CC=C", "CC(C)=C"),
        id="double-bonds",
    ),
    pytest.param(
        ("C#C", "CC#C", "C#CCC", "C#CC#C", "CC#CCC"),
        id="triple-bonds",
    ),
    pytest.param(("C=CC#C", "CC=CC#C", "CC#CC=C"), id="mixed-unsaturation"),
    pytest.param(
        ("CO", "CCCO", "CC(O)CC", "CC(O)CO", "CC(CO)CO"),
        id="hydroxy-groups",
    ),
    pytest.param(
        ("CF", "CCCl", "CCCBr", "CI", "CC(F)(F)F"),
        id="halogen-substitution",
    ),
    pytest.param(
        ("COC", "CCOCC", "CSC", "CCSCC", "CSSCC"),
        id="heteroatom-links",
    ),
)


@pytest.mark.parametrize("inputs", CHAIN_FAMILIES)
def test_chain_feature_survives_naming(naming, assert_graphs, inputs):
    """A short feature family is named once and validated in one parser batch."""
    names = naming.names(inputs)
    assert len(names) == len(inputs)
    assert all(isinstance(name, str) and name for name in names)
    assert_graphs(inputs, names)


def test_four_chain_controls_are_stable_under_atom_order(naming, assert_graphs):
    """Exercise traversal invariance without repeating every chemical family."""
    inputs = ("CCC(C)C", "CC=CC#C", "CC(O)CO", "CCSCC")
    reordered = []
    for text in inputs:
        mol = Chem.MolFromSmiles(text)
        assert mol is not None
        other = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        reordered.append(Chem.MolToSmiles(other, canonical=False, isomericSmiles=True))

    original_names = naming.names(inputs)
    reordered_names = naming.names(reordered)
    assert original_names == reordered_names
    assert_graphs(reordered, reordered_names)


# Originally test_functional_boundaries.py
"""Small controls for functional classes missed by the initial core matrix.

Provenance: these graphs were authored from elementary functional-group
valence patterns and the general P-65/P-66/P-67/P-68/P-72 class definitions.
Coverage reports identified classes to exercise, not molecule identities or
preferred names. No panel records or molecule-level disagreement files were
consulted. Each group has only a few controls and one graph-oracle batch.

Full-only groups concern the current mapped acid/amide and charged-parent
operations. The assertions deliberately do not inspect producer selection,
ranking, generated-name spelling, or private component objects.
"""


import pytest


FUNCTIONAL_BOUNDARIES = (
    pytest.param(
        ("NN", "CNN", "CN(C)N", "CON", "CNOC"),
        id="nitrogen-nitrogen-and-nitrogen-oxygen-links",
    ),
    pytest.param(
        ("COOC", "CCOOC", "CCOOC(C)C"),
        id="organic-peroxides",
    ),
    pytest.param(
        ("COC(=O)OC", "CCOC(=O)OCC", "COC(=S)OC", "CSC(=O)OC", "CSC(=S)OC"),
        id="carbonates-and-chalcogen-replacement",
    ),
    pytest.param(
        ("C=NN", "CC=NN", "CC=NN(C)C", "CC=NN=CC"),
        id="hydrazones-and-azine",
        marks=pytest.mark.full_only,
    ),
    pytest.param(
        ("CC(=O)NN", "CCC(=O)NNC", "CC(=O)N(C)N", "NNC(=O)CC(=O)NN"),
        id="carbonyl-hydrazides",
        marks=pytest.mark.full_only,
    ),
    pytest.param(
        ("CS(=O)N", "CCS(=O)NC", "CS(=O)(=O)NN", "CS(=O)NN"),
        id="sulfur-amides-and-hydrazides",
        marks=pytest.mark.full_only,
    ),
    pytest.param(
        ("NP(=O)(N)N", "COP(=O)(N)OC", "CP(=O)(N)N", "CCOP(=O)(N)OCC"),
        id="phosphorus-amides-and-amidic-esters",
        marks=pytest.mark.full_only,
    ),
    pytest.param(
        ("CB(O)O", "CB(C)O", "c1ccccc1B(O)O", "CCB(O)O"),
        id="boron-oxoacids",
        marks=pytest.mark.full_only,
    ),
    pytest.param(
        ("CC(=O)O[Si](C)(C)C", "CC(=O)O[Ge](C)(C)C", "CCC(=O)O[Si](C)(C)CC"),
        id="group-fourteen-pseudoesters",
        marks=pytest.mark.full_only,
    ),
    pytest.param(
        ("CC(=O)OO", "CCC(=O)OO", "CC(C)C(=O)OO", "O=C(OO)c1ccccc1"),
        id="peroxyacids",
        marks=pytest.mark.full_only,
    ),
    pytest.param(
        ("COC(=O)CC(=O)O", "CCOC(=O)CCC(=O)O", "COC(=O)C(C)C(=O)O", "COC(=O)CC(O)C(=O)O"),
        id="partial-polyacid-esters",
        marks=pytest.mark.full_only,
    ),
    pytest.param(
        ("CC(=O)OC(=O)C", "CC(=O)OC(=O)CC", "CC(=O)OC(=O)CC(=O)OC(=O)C"),
        id="one-and-two-anhydride-bridges",
        marks=pytest.mark.full_only,
    ),
    pytest.param(
        ("OCC(=O)[O-]", "O=C([O-])CC(=O)[O-]", "NCC(=O)[O-]", "C[NH2+]CC(=O)[O-]"),
        id="charged-multifunctional-graphs",
        marks=pytest.mark.full_only,
    ),
    pytest.param(
        ("CC[NH-]", "CC[N-]C", "CC[N-2]"),
        id="amine-hydrogen-loss-and-charge",
        marks=pytest.mark.full_only,
    ),
)


@pytest.mark.parametrize("inputs", FUNCTIONAL_BOUNDARIES)
def test_functional_boundary_preserves_the_declared_graph(naming, assert_graphs, inputs):
    names = naming.names(inputs)
    assert len(names) == len(inputs)
    assert all(isinstance(name, str) and name for name in names)
    assert_graphs(inputs, names)


@pytest.mark.full_only
@pytest.mark.parametrize("text", ("C[NH-2]", "C[N-2]C"), ids=("excess-hydrogen", "excess-carbon-valence"))
def test_ineligible_amine_dianion_valence_is_not_named(naming, text):
    """Charge/H inconsistencies are invalid input, not a neutral fallback."""
    result = naming.result(text)
    assert not result.name
    assert result.generation_status == "invalid-input"


# Originally test_functional_contracts.py
"""Focused graph recovery across independently authored functional groups.

Provenance: the controls were constructed from elementary valence patterns
for acid derivatives, nitrogen/sulfur functions and ordinary ions. No panel
structures, source names or molecule-level benchmark disagreements were used.
Assertions concern graph recovery, never the spelling or preference of names.
"""


import pytest


FUNCTIONAL_FAMILIES = (
    pytest.param(
        ("CC(=O)O", "CCC(=O)O", "CC(C)C(=O)O", "OC(=O)CC(=O)O"),
        id="carboxylic-acids",
    ),
    pytest.param(
        ("COC(=O)C", "CCOC(=O)CC", "COC(=O)C(C)C", "COC(=O)CC(=O)OC"),
        id="esters",
    ),
    pytest.param(
        ("CC=O", "CCC=O", "CC(=O)C", "CCC(=O)CC", "CC(=O)CC(=O)C"),
        id="carbonyl-groups",
    ),
    pytest.param(
        ("CN", "CCN", "CNC", "CN(C)C", "NCCN", "CCNC"),
        id="amine-substitution",
    ),
    pytest.param(
        ("CC(N)=O", "CCC(=O)NC", "CC(=O)N(C)C", "NC(=O)N", "NC(=O)CC(=O)N"),
        id="amides",
    ),
    pytest.param(
        ("CC#N", "CCC#N", "N#CCC#N", "CC=N", "CC=NO"),
        id="nitrogen-bond-order",
    ),
    pytest.param(
        ("CC(=S)N", "CS(=O)C", "CS(=O)(=O)C", "CS(=O)(=O)N", "OP(=O)(O)O"),
        id="heteroatom-oxidation",
    ),
    pytest.param(
        ("CC(=O)[O-]", "C[NH3+]", "C[N+](C)(C)C", "CS(=O)(=O)[O-]"),
        id="formal-charge",
    ),
)


@pytest.mark.parametrize("inputs", FUNCTIONAL_FAMILIES)
def test_functional_feature_survives_naming(naming, assert_graphs, inputs):
    """Check a bounded family rather than replaying a full development corpus."""
    names = naming.names(inputs)
    assert len(names) == len(inputs)
    assert all(isinstance(name, str) and name for name in names)
    assert_graphs(inputs, names)
