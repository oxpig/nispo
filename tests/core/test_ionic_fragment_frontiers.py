"""Independent, bounded functional-class and ionic fragment boundaries.

Origin: authored from ordinary C/S/O/N valences, formal-charge balance and
functional-class definitions, not from panel records, reference names or
production-comment structures. The fixed set samples individual thiooxo,
azanide, peroxide, pseudohalide and chalcogen-oxo sites, without expansions.
Definitions are frozen before evaluation in
evaluation/coverage_frontiers_20261111/ionic/definitions.json. All 24 inputs
were available in the frozen full/fast baseline; the v2 contract therefore
requires continued availability as well as complete graph conservation.
"""
from collections import Counter

import pytest
from rdkit import Chem


pytestmark = pytest.mark.current_only


CASES = (
    ("thiooxo-ester", "COC(=S)C"),
    ("mixed-dithiocarbonate", "COC(=S)SC"),
    ("dithiol-carbonate", "CSC(=O)SC"),
    ("monoalkyl-dithiocarbonate-anion", "[Na+].COC(=S)[S-]"),
    ("primary-acyl-azanide", "[Na+].CC(=O)[NH-]"),
    ("alkyl-acyl-azanide", "[Na+].CC(=O)[N-]C"),
    ("bis-acyl-azanide", "[Na+].CC(=O)[N-]C(=O)C"),
    ("primary-carbamate-azanide", "[Na+].COC(=O)[NH-]"),
    ("alkyl-carbamate-azanide", "[Na+].COC(=O)[N-]C"),
    ("peroxyoxalate", "[Na+].[Na+].O=C([O-])C(=O)O[O-]"),
    ("cyanonitrate", "N#C[N+](=O)[O-]"),
    ("thiosulfate", "[Na+].[Na+].[O-]S(=O)(=S)[O-]"),
    ("hydrogen-thiosulfate", "[Na+].OS(=O)(=S)[O-]"),
    ("thioselenate", "[Na+].[Na+].[O-][Se](=O)(=S)[O-]"),
    ("selenosulfate", "[Na+].[Na+].[O-]S(=O)(=[Se])[O-]"),
    ("sulfur-cyano-pseudohalide", "[Na+].[S-]C#N"),
    ("sulfur-isocyano-pseudohalide", "[Na+].[N-]=C=S"),
    ("sulfinyl-cyanide", "CCS(=O)C#N"),
    ("sulfonyl-cyanide", "CCS(=O)(=O)C#N"),
    ("sulfinyl-cation", "[Cl-].CC[S+]=O"),
    ("sulfonyl-cation", "[Cl-].CC[S+](=O)=O"),
    ("carbon-bound-thiolate", "[Na+].CC[S-]"),
    ("carbon-bound-peroxolate", "[Na+].CCO[O-]"),
    ("thioacyl-cation", "[Cl-].CC[C+]=S"),
)


def _element_census(graph):
    return Counter(atom.GetAtomicNum() for atom in graph.GetAtoms())


@pytest.fixture(scope="module")
def ionic_frontier_results(naming, assert_graphs):
    inputs = tuple(text for _, text in CASES)
    # Validate the independently declared valence graphs before any name is
    # treated as a result. Counterions are part of the graph contract.
    assert all(Chem.MolFromSmiles(text) is not None for text in inputs)
    results = dict(zip(inputs, naming.results(inputs), strict=True))
    names = tuple(dict.fromkeys(result.name for result in results.values() if result.name))
    parsed = dict(zip(names, assert_graphs.parse_names(names), strict=True))
    return results, parsed


@pytest.mark.parametrize("case_id,smiles", CASES, ids=[row[0] for row in CASES])
def test_ionic_functional_class_retains_availability_all_atoms_and_charge(
    ionic_frontier_results, assert_graphs, case_id, smiles
):
    results, parsed = ionic_frontier_results
    result = results[smiles]
    assert result.generation_status == "available"
    assert result.name
    assert_graphs([smiles], [result.name])
    expected = Chem.MolFromSmiles(smiles)
    recovered = Chem.MolFromSmiles(parsed[result.name])
    assert expected is not None and recovered is not None
    assert _element_census(recovered) == _element_census(expected)
    assert sum(a.GetFormalCharge() for a in recovered.GetAtoms()) == sum(
        a.GetFormalCharge() for a in expected.GetAtoms()
    )
    assert sum(a.GetNumRadicalElectrons() for a in recovered.GetAtoms()) == sum(
        a.GetNumRadicalElectrons() for a in expected.GetAtoms()
    )
