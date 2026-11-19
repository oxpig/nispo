"""Molecule traversal and caller ownership for small parent/branch frontiers.

Eight authorized SureChEMBL development graphs were selected from previously
verified fixtures by structural category and smallest size, independently of
names or private ranking results. Each uses its supplied atom order and one
reversal; there is no atom-permutation or API cross product.
"""
import json
from pathlib import Path

import pytest
from rdkit import Chem


DATA = json.loads((Path(__file__).resolve().parents[1] /
                   'data/development_graphs.json').read_text())["cohorts"]['molecule_traversal_frontiers_20261111.json']
CASES = DATA['cases']
pytestmark = pytest.mark.full_only


@pytest.fixture(scope='module')
def traversed_results(naming, assert_graphs):
    results = {}
    for case in CASES:
        source = Chem.MolFromSmiles(case['smiles'])
        assert source is not None
        for view in ('source', 'reverse'):
            graph = (Chem.Mol(source) if view == 'source' else
                     Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms())))))
            assert Chem.MolToSmiles(graph, isomericSmiles=True) == case['smiles']
            graph.SetProp('caller_origin', case['case_id'])
            for atom in graph.GetAtoms():
                atom.SetIntProp('caller_atom_index', atom.GetIdx())
            before = graph.ToBinary(Chem.PropertyPickleOptions.AllProps)
            result = naming.molecule_result(graph)
            unchanged = graph.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
            results[case['case_id'], view] = result, unchanged
    assert_graphs.parse_names([row[0].name for row in results.values()])
    return results


@pytest.mark.parametrize('case', CASES, ids=lambda row: row['category'])
@pytest.mark.parametrize('view', ('source', 'reverse'))
def test_molecule_traversal_keeps_complete_graph_and_caller_data(
    case, view, traversed_results, assert_graphs
):
    result, unchanged = traversed_results[case['case_id'], view]
    assert unchanged, 'Naming changed caller-owned molecule or atom properties'
    assert result.generation_status == 'available', result.unresolved
    assert_graphs([case['smiles']], [result.name])
