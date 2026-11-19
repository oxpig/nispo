"""Small original-graph isotope and stereo boundaries through the public API.

Independent origin (11 November 2026): two linked cyclopropane rings with a
single conventional acid, amide or nitrile substituent; one cyclic radical;
and short acetal-ether chains with terminal characteristic groups. Isotopes
were placed at declared parent, junction, suffix N/O and attached-H sites.
No panel structures, reference names or disagreement files supplied inputs.
The fixed definitions are frozen before their first evaluation in
evaluation/coverage_expansion_20261111/annotations/definitions.json.
"""
from collections import Counter

import pytest
from rdkit import Chem


pytestmark = pytest.mark.full_only


CASES = (
    ("assembly-acid-carbon", "O=C(O)C1CC1C2CC2", "O=[13C](O)C1CC1C2CC2"),
    ("assembly-acid-oxygen", "O=C(O)C1CC1C2CC2", "O=C([18OH])C1CC1C2CC2"),
    ("assembly-acid-hydrogen", "O=C(O)C1CC1C2CC2", "[2H]OC(=O)C1CC1C2CC2"),
    ("assembly-amide-nitrogen", "NC(=O)C1CC1C2CC2", "[15NH2]C(=O)C1CC1C2CC2"),
    ("assembly-amide-hydrogen", "NC(=O)C1CC1C2CC2", "[2H]NC(=O)C1CC1C2CC2"),
    ("assembly-nitrile-nitrogen", "N#CC1CC1C2CC2", "[15N]#CC1CC1C2CC2"),
    ("assembly-junction-carbon", "C1CC1C2CC2", "[13CH]1(CC1)C2CC2"),
    ("assembly-amide-oxygen", "NC(=O)C1CC1C2CC2", "NC(=[18O])C1CC1C2CC2"),
    ("assembly-absolute-stereo", "O=C(O)C1CC1C2CC2", "O=C(O)[C@H]1C[C@@H]1C2CC2"),
    ("assembly-stereo-and-isotope", "O=C(O)C1CC1C2CC2", "O=[13C](O)[C@H]1C[C@@H]1C2CC2"),
    ("assembly-cyclic-radical", "[CH]1CC1C2CC2", "[13CH]1CC1C2CC2"),
    ("assembly-anionic-oxygen", "O=C([O-])C1CC1C2CC2", "O=C([18O-])C1CC1C2CC2"),
    ("ether-acid-hydrogen", "COCOCC(=O)O", "[2H]OC(=O)COCOC"),
    ("ether-amide-nitrogen", "COCOCC(=O)N", "COCOCC(=O)[15NH2]"),
    ("ether-amide-hydrogen", "COCOCC(=O)N", "[2H]NC(=O)COCOC"),
    ("ether-nitrile-nitrogen", "COCOCC#N", "COCOCC#[15N]"),
)

CALLER_CASE_IDS = (
    "assembly-acid-hydrogen", "assembly-amide-nitrogen",
    "assembly-stereo-and-isotope", "assembly-cyclic-radical",
)


def _census(graph):
    return Counter((atom.GetAtomicNum(), atom.GetIsotope())
                   for atom in graph.GetAtoms() if atom.GetIsotope())


def _assert_declined(result):
    assert result.name == ""
    assert result.generation_status == "unavailable"
    assert result.preferred_status in {"unresolved", "not-assessed"}
    assert result.unresolved
    assert all(issue.stage and issue.rule and issue.reason for issue in result.unresolved)


@pytest.fixture(scope="module")
def assembly_boundary_results(naming, assert_graphs):
    # Name each fixed input once, then amortize the external parser startup
    # across this bounded feature set. Individual tests retain case-level IDs.
    inputs = tuple(dict.fromkeys(text for _, ordinary, labelled in CASES
                                for text in (ordinary, labelled)))
    results = dict(zip(inputs, naming.results(inputs), strict=True))
    delivered = tuple(dict.fromkeys(result.name for result in results.values() if result.name))
    recovered = dict(zip(delivered, assert_graphs.parse_names(delivered), strict=True))
    return results, recovered


@pytest.mark.parametrize("case_id,ordinary,labelled", CASES, ids=[row[0] for row in CASES])
def test_labelled_assembly_preserves_whole_graph_or_explicitly_declines(
    assembly_boundary_results, assert_graphs, case_id, ordinary, labelled
):
    results, recovered = assembly_boundary_results
    baseline, result = results[ordinary], results[labelled]
    expected = Chem.MolFromSmiles(labelled)
    assert expected is not None
    assert baseline.generation_status == "available" and baseline.name
    assert_graphs([ordinary], [baseline.name])

    if not result.name:
        _assert_declined(result)
        return

    assert result.generation_status == "available"
    assert_graphs([labelled], [result.name])
    actual = Chem.MolFromSmiles(recovered[result.name])
    assert actual is not None
    assert _census(actual) == _census(expected)
    assert sum(atom.GetFormalCharge() for atom in actual.GetAtoms()) == sum(
        atom.GetFormalCharge() for atom in expected.GetAtoms()
    )
    assert sum(atom.GetNumRadicalElectrons() for atom in actual.GetAtoms()) == sum(
        atom.GetNumRadicalElectrons() for atom in expected.GetAtoms()
    )


@pytest.mark.parametrize("case_id", CALLER_CASE_IDS)
def test_molecule_entrypoint_preserves_assembly_isotopes_stereo_and_caller_data(
    naming, assembly_boundary_results, assert_graphs, case_id
):
    _, _, labelled = next(row for row in CASES if row[0] == case_id)
    results, _ = assembly_boundary_results
    graph = Chem.MolFromSmiles(labelled)
    assert graph is not None
    graph.SetProp("independent_origin", case_id)
    for atom in graph.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 101)
    before = graph.ToBinary(Chem.PropertyPickleOptions.AllProps)
    result = naming.molecule_result(graph)

    assert graph.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    assert result.generation_status == results[labelled].generation_status
    if result.name:
        assert_graphs([labelled], [result.name])
    else:
        _assert_declined(result)
