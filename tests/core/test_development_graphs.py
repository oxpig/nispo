"""Related release contracts consolidated without dropping cases or assertions.

Original module provenance is retained in each section below.
"""


# Originally test_surechembl_coverage.py
"""Frozen, coverage-selected SureChEMBL development graph regressions.

Candidates come only from the authorized Zenodo 16755947 development export.
Selection uses measured coverage gain and naming cost after raw OPSIN graph
verification; it is not an accuracy sample or a preferred-name oracle. Source
row identities, original graphs, selection seed and frozen code are recorded
in the fixture. No source or generated names are expected strings.
"""
import hashlib
import json
from pathlib import Path

import pytest


SURECHEMBL_COVERAGE_FIXTURE = json.loads((Path(__file__).resolve().parents[1] /
                      'data/development_graphs.json').read_text())["cohorts"]['surechembl_coverage_20261110.json']
SURECHEMBL_COVERAGE_CASES = SURECHEMBL_COVERAGE_FIXTURE['cases']
pytestmark = pytest.mark.current_only


@pytest.fixture(scope='module')
def _surechembl_coverage_measured_results(naming, assert_graphs):
    """Name each mode's small cohort once and share one parser batch."""
    from rdkit import Chem

    selected = [case for case in SURECHEMBL_COVERAGE_CASES if naming.mode in case['verified_modes']]
    assert SURECHEMBL_COVERAGE_FIXTURE['source'] == '10.5281/zenodo.16755947'
    for case in selected:
        graph = Chem.MolFromSmiles(case['smiles'])
        original = Chem.MolFromSmiles(case['source_smiles'])
        assert graph is not None and original is not None
        assert Chem.MolToSmiles(graph) == Chem.MolToSmiles(original)
        assert hashlib.sha256(case['smiles'].encode()).hexdigest() == case['graph_sha256']
    results = naming.results([case['smiles'] for case in selected])
    assert len(results) == len(selected)
    # Cache parser work; each test below still reports its own graph failure.
    assert_graphs.parse_names([result.name for result in results], raw_only=True)
    return dict(zip((case['case_id'] for case in selected), results, strict=True))


@pytest.mark.parametrize(
    'case',
    [pytest.param(case, marks=pytest.mark.modes(*case['verified_modes']))
     for case in SURECHEMBL_COVERAGE_CASES],
    ids=lambda case: case['case_id'],
)
def test_surechembl_raw_opsin_graph(case, naming, _surechembl_coverage_measured_results, assert_graphs):
    result = _surechembl_coverage_measured_results[case['case_id']]
    assert result.generation_status == 'available', result.unresolved
    assert_graphs([case['smiles']], [result.name], raw_only=True)


# Originally test_surechembl_frontier.py
"""Additional coverage-selected graphs from authorized SureChEMBL development.

The previous 10 November cohort remains intact. This separate frozen cohort
adds measured executable statements/reportable branch outcomes outside each
mode's existing core coverage. Selection uses source-row graph provenance and
raw OPSIN verification, never historical names or panel disagreements.
"""
import hashlib
import json
from pathlib import Path

import pytest


SURECHEMBL_FRONTIER_DATA = json.loads((Path(__file__).resolve().parents[1] /
                   'data/development_graphs.json').read_text())["cohorts"]['surechembl_coverage_20261111.json']
SURECHEMBL_FRONTIER_CASES = SURECHEMBL_FRONTIER_DATA['cases']
pytestmark = pytest.mark.current_only


@pytest.fixture(scope='module')
def _surechembl_frontier_delivered(naming, assert_graphs):
    from rdkit import Chem

    selected = [row for row in SURECHEMBL_FRONTIER_CASES if naming.mode in row['verified_modes']]
    assert SURECHEMBL_FRONTIER_DATA['source'] == '10.5281/zenodo.16755947'
    for row in selected:
        original = Chem.MolFromSmiles(row['source_smiles'])
        graph = Chem.MolFromSmiles(row['smiles'])
        assert original is not None and graph is not None
        assert Chem.MolToSmiles(original) == Chem.MolToSmiles(graph)
        assert hashlib.sha256(row['smiles'].encode()).hexdigest() == row['graph_sha256']
    results = naming.results([row['smiles'] for row in selected])
    assert len(results) == len(selected)
    assert_graphs.parse_names([result.name for result in results], raw_only=True)
    return dict(zip((row['case_id'] for row in selected), results, strict=True))


@pytest.mark.parametrize(
    'case',
    [pytest.param(case, marks=pytest.mark.modes(*case['verified_modes']))
     for case in SURECHEMBL_FRONTIER_CASES],
    ids=lambda row: row['case_id'],
)
def test_additional_development_graph_roundtrips(case, naming, _surechembl_frontier_delivered, assert_graphs):
    result = _surechembl_frontier_delivered[case['case_id']]
    assert result.generation_status == 'available', result.unresolved
    assert_graphs([case['smiles']], [result.name], raw_only=True)


# Originally test_surechembl_branch_frontiers.py
"""Coverage-selected regressions focused on the largest naming-module gaps.

The authorized SureChEMBL development cohort is separate from both earlier
freezes. Selection uses only new reportable outcomes in parent/rule/fragment
and acyclic modules, with raw OPSIN verification and recorded source lineage.
No generated names or private parent/ranking decisions are expectations.
"""
import hashlib
import json
from pathlib import Path

import pytest


SURECHEMBL_BRANCH_FRONTIERS_DATA = json.loads((Path(__file__).resolve().parents[1] /
                   'data/development_graphs.json').read_text())["cohorts"]['surechembl_branch_frontiers_20261111.json']
SURECHEMBL_BRANCH_FRONTIERS_CASES = SURECHEMBL_BRANCH_FRONTIERS_DATA['cases']
pytestmark = pytest.mark.current_only


@pytest.fixture(scope='module')
def _surechembl_branch_frontiers_delivered_frontier(naming, assert_graphs):
    from rdkit import Chem

    selected = [row for row in SURECHEMBL_BRANCH_FRONTIERS_CASES if naming.mode in row['verified_modes']]
    assert SURECHEMBL_BRANCH_FRONTIERS_DATA['source'] == '10.5281/zenodo.16755947'
    for row in selected:
        original = Chem.MolFromSmiles(row['source_smiles'])
        graph = Chem.MolFromSmiles(row['smiles'])
        assert original is not None and graph is not None
        assert Chem.MolToSmiles(original) == Chem.MolToSmiles(graph)
        assert hashlib.sha256(row['smiles'].encode()).hexdigest() == row['graph_sha256']
    results = naming.results([row['smiles'] for row in selected])
    assert len(results) == len(selected)
    assert_graphs.parse_names([result.name for result in results], raw_only=True)
    return dict(zip((row['case_id'] for row in selected), results, strict=True))


@pytest.mark.parametrize(
    'case',
    [pytest.param(case, marks=pytest.mark.modes(*case['verified_modes']))
     for case in SURECHEMBL_BRANCH_FRONTIERS_CASES],
    ids=lambda row: row['case_id'],
)
def test_largest_gap_development_graph_roundtrip(case, naming, _surechembl_branch_frontiers_delivered_frontier, assert_graphs):
    result = _surechembl_branch_frontiers_delivered_frontier[case['case_id']]
    assert result.generation_status == 'available', result.unresolved
    assert_graphs([case['smiles']], [result.name], raw_only=True)


# Originally test_surechembl_name_repairs.py
"""Raw OPSIN regressions on the authorized 2,000-graph development cohort."""

import hashlib
import json
from pathlib import Path

import pytest
from rdkit import Chem


SURECHEMBL_NAME_REPAIRS_DATA = json.loads((Path(__file__).resolve().parents[1] /
                   'data/development_graphs.json').read_text())["cohorts"]['surechembl_name_repairs_20261114.json']
pytestmark = pytest.mark.current_only


@pytest.fixture(scope='module')
def _surechembl_name_repairs_delivered_repairs(naming, assert_graphs):
    values = []
    for case in SURECHEMBL_NAME_REPAIRS_DATA['cases']:
        original = Chem.MolFromSmiles(case['source_smiles'])
        canonical = Chem.MolFromSmiles(case['smiles'])
        assert original is not None and canonical is not None
        assert Chem.MolToSmiles(original) == Chem.MolToSmiles(canonical)
        assert hashlib.sha256(case['smiles'].encode()).hexdigest() == case['graph_sha256']
        alternate = Chem.RenumberAtoms(canonical, list(reversed(range(canonical.GetNumAtoms()))))
        values.extend((case['smiles'], Chem.MolToSmiles(alternate, canonical=False)))
    results = naming.results(values)
    assert_graphs.parse_names([result.name for result in results], raw_only=True)
    pairs = zip(results[::2], results[1::2], strict=True)
    return dict(zip((case['case_id'] for case in SURECHEMBL_NAME_REPAIRS_DATA['cases']), pairs, strict=True))


@pytest.mark.parametrize('case', SURECHEMBL_NAME_REPAIRS_DATA['cases'], ids=lambda case: case['case_id'])
def test_name_repair_preserves_original_graph(case, _surechembl_name_repairs_delivered_repairs, assert_graphs):
    results = _surechembl_name_repairs_delivered_repairs[case['case_id']]
    assert all(result.generation_status == 'available' for result in results)
    assert_graphs([case['smiles']] * len(results), [result.name for result in results], raw_only=True)


def test_chalcogen_rendering_preserves_each_original_atom(naming, assert_graphs):
    # Independent controls for two and three linked sulfur atoms and a
    # terminal carbonyl oxide. Each added chalcogen changes the graph.
    smiles = ['CSSc1ccc(C(=O)O)cc1', 'CSSSc1ccc(C(=O)O)cc1', 'CC(C)=[O+][O-]']
    results = naming.results(smiles)
    assert_graphs(smiles, [result.name for result in results], raw_only=True)


# Originally test_surechembl_followup_name_repairs.py
"""Raw OPSIN regressions from the follow-up SureChEMBL development draw."""

import hashlib
import json
from pathlib import Path

import pytest
from rdkit import Chem


SURECHEMBL_FOLLOWUP_NAME_REPAIRS_CASES = json.loads((Path(__file__).resolve().parents[1] /
                    'data/development_graphs.json').read_text())["cohorts"]['surechembl_name_repairs_20261114_followup.json']['cases']
pytestmark = pytest.mark.current_only


@pytest.fixture(scope='module')
def _surechembl_followup_name_repairs_names_by_case(naming, assert_graphs):
    inputs = []
    for case in SURECHEMBL_FOLLOWUP_NAME_REPAIRS_CASES:
        canonical = case['smiles']
        original = Chem.MolFromSmiles(case['source_smiles'])
        mol = Chem.MolFromSmiles(canonical)
        assert original is not None and mol is not None
        assert Chem.MolToSmiles(original) == Chem.MolToSmiles(mol)
        assert hashlib.sha256(canonical.encode()).hexdigest() == case['graph_sha256']
        reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        inputs.extend((canonical, Chem.MolToSmiles(reordered, canonical=False)))

    if naming.mode == 'full':
        from nispo import smiles_to_iupac_batch_results

        results = smiles_to_iupac_batch_results(inputs, timeout_seconds=None, fallback_to_fast=False)
    else:
        results = naming.results(inputs)
    assert_graphs.parse_names([result.name for result in results], raw_only=True)
    return dict(zip((case['case_id'] for case in SURECHEMBL_FOLLOWUP_NAME_REPAIRS_CASES),
                    zip(results[::2], results[1::2], strict=True), strict=True))


@pytest.mark.parametrize('case', SURECHEMBL_FOLLOWUP_NAME_REPAIRS_CASES, ids=lambda case: case['case_id'])
def test_surechembl_followup_name_repairs_repaired_name_recovers_the_original_graph(case, _surechembl_followup_name_repairs_names_by_case, assert_graphs):
    results = _surechembl_followup_name_repairs_names_by_case[case['case_id']]
    assert all(result.generation_status == 'available' for result in results)
    assert_graphs([case['smiles']] * len(results), [result.name for result in results], raw_only=True)


# Originally test_surechembl_refreshed_name_repairs.py
"""Raw OPSIN regressions from the refreshed SureChEMBL development draw."""

import hashlib
import json
from pathlib import Path

import pytest
from rdkit import Chem


SURECHEMBL_REFRESHED_NAME_REPAIRS_CASES = json.loads((Path(__file__).resolve().parents[1] /
                    'data/development_graphs.json').read_text())["cohorts"]['surechembl_name_repairs_20261114_refresh.json']['cases']
pytestmark = pytest.mark.current_only


@pytest.fixture(scope='module')
def _surechembl_refreshed_name_repairs_names_by_case(naming, assert_graphs):
    inputs = []
    for case in SURECHEMBL_REFRESHED_NAME_REPAIRS_CASES:
        canonical = case['smiles']
        original = Chem.MolFromSmiles(case['source_smiles'])
        mol = Chem.MolFromSmiles(canonical)
        assert original is not None and mol is not None
        assert Chem.MolToSmiles(original) == Chem.MolToSmiles(mol)
        assert hashlib.sha256(canonical.encode()).hexdigest() == case['graph_sha256']
        reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        inputs.extend((canonical, Chem.MolToSmiles(reordered, canonical=False)))

    if naming.mode == 'full':
        from nispo import smiles_to_iupac_batch_results

        results = smiles_to_iupac_batch_results(inputs, timeout_seconds=None, fallback_to_fast=False)
    else:
        results = naming.results(inputs)
    assert_graphs.parse_names([result.name for result in results], raw_only=True)
    return dict(zip((case['case_id'] for case in SURECHEMBL_REFRESHED_NAME_REPAIRS_CASES),
                    zip(results[::2], results[1::2], strict=True), strict=True))


@pytest.mark.parametrize('case', SURECHEMBL_REFRESHED_NAME_REPAIRS_CASES, ids=lambda case: case['case_id'])
def test_surechembl_refreshed_name_repairs_repaired_name_recovers_the_original_graph(case, _surechembl_refreshed_name_repairs_names_by_case, assert_graphs):
    results = _surechembl_refreshed_name_repairs_names_by_case[case['case_id']]
    assert all(result.generation_status == 'available' for result in results)
    assert_graphs([case['smiles']] * len(results), [result.name for result in results], raw_only=True)


# Originally test_release_development_regressions.py
"""Every matched-cohort raw graph loss found while preparing local rc1.

Authorized SureChEMBL development only; preserved baseline, candidate, source
release and graph hashes are in the fixture. No expected names are stored.
"""
import hashlib
import json
from pathlib import Path
import random

import pytest
from rdkit import Chem

RELEASE_DEVELOPMENT_REGRESSIONS_DATA = json.loads((Path(__file__).resolve().parents[1] /
                   'data/development_graphs.json').read_text())["cohorts"]['release_rc1_development_regressions.json']
RELEASE_DEVELOPMENT_REGRESSIONS_FOLLOWUP = json.loads((Path(__file__).resolve().parents[1] /
                       'data/development_graphs.json').read_text())["cohorts"]['release_rc1_preference_regressions.json']
RELEASE_DEVELOPMENT_REGRESSIONS_DATA['cases'] += RELEASE_DEVELOPMENT_REGRESSIONS_FOLLOWUP['cases']
pytestmark = pytest.mark.current_only


@pytest.fixture(scope='module')
def _release_development_regressions_delivered(naming, assert_graphs):
    selected = [row for row in RELEASE_DEVELOPMENT_REGRESSIONS_DATA['cases'] if naming.mode in row['verified_modes']]
    assert RELEASE_DEVELOPMENT_REGRESSIONS_DATA['source'] == '10.5281/zenodo.16755947'
    for row in selected:
        assert hashlib.sha256(row['smiles'].encode()).hexdigest() == row['graph_sha256']
    results = naming.results([row['smiles'] for row in selected])
    assert len(results) == len(selected)
    assert_graphs.parse_names([result.name for result in results], raw_only=True)
    return dict(zip((row['case_id'] for row in selected), results, strict=True))


@pytest.mark.parametrize(
    'case',
    [pytest.param(case, marks=pytest.mark.modes(*case['verified_modes']))
     for case in RELEASE_DEVELOPMENT_REGRESSIONS_DATA['cases']],
    ids=lambda row: row['case_id'],
)
def test_preserve_previous_raw_graph_recovery(case, naming, _release_development_regressions_delivered, assert_graphs):
    result = _release_development_regressions_delivered[case['case_id']]
    assert result.generation_status == 'available', result.unresolved
    assert_graphs([case['smiles']], [result.name], raw_only=True)


@pytest.mark.parametrize('case', [row for row in RELEASE_DEVELOPMENT_REGRESSIONS_FOLLOWUP['cases']
                                if row['case_id'].startswith('rc1-parent-proof-')],
                         ids=lambda row: row['case_id'])
def test_repaired_parent_frames_preserve_atom_order_and_graph(case, naming, assert_graphs):
    mol = Chem.MolFromSmiles(case['smiles'])
    order = list(range(mol.GetNumAtoms()))
    random.Random(20261117).shuffle(order)
    variant = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False,
                              isomericSmiles=True)
    results = naming.results([case['smiles'], variant])
    assert all(result.generation_status == 'available' for result in results)
    assert results[0].name == results[1].name
    assert_graphs([case['smiles'], variant], [result.name for result in results], raw_only=True)
