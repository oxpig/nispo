"""Related release contracts consolidated without dropping cases or assertions.

Original module provenance is retained in each section below.
"""
from __future__ import annotations


# Originally test_api_contracts.py
"""Small public delivery contracts, independent of nomenclature spelling.

Provenance: the two ordinary graphs below were selected as elementary API
controls (a three-atom alcohol and a three-carbon chain), independently of any
benchmark panel or naming disagreement. Invalid inputs are authored parser and
Python-type boundary probes. Synthetic result payloads test the public value
objects, and are not expected generated names.
"""


from dataclasses import FrozenInstanceError
import importlib
import inspect
import math
import pickle

import pytest
from rdkit import Chem


INVALID_SMILES = [
    pytest.param('', id='empty'),
    pytest.param(' ', id='whitespace'),
    pytest.param('not-a-smiles', id='malformed'),
    pytest.param('C1CC', id='unclosed-ring'),
    pytest.param(None, id='none'),
    pytest.param([], id='unhashable'),
]


@pytest.mark.parametrize('value', INVALID_SMILES)
def test_invalid_single_input_returns_no_name(naming, value):
    assert naming.name(value) == ''


@pytest.mark.parametrize('factory', [list, tuple, iter], ids=['list', 'tuple', 'iterator'])
def test_empty_batch_has_no_results(naming, factory):
    assert naming.names(factory(())) == []


def test_generator_batch_preserves_order_invalid_slots_and_duplicates(naming, assert_graphs):
    inputs = ('CCO', '', 'CCC', 'CCO', 'not-a-smiles')
    consumed = []

    def once():
        for index, value in enumerate(inputs):
            consumed.append(index)
            yield value

    names = naming.names(once())
    assert consumed == list(range(len(inputs)))
    assert len(names) == len(inputs)
    assert names[0] and names[2]
    assert names[1] == names[4] == ''
    assert names[0] == names[3]
    assert_graphs([inputs[0], inputs[2]], [names[0], names[2]])


def test_common_exports_remain_callable():
    import nispo

    for symbol in ('smiles_to_iupac', 'mol_to_iupac', 'smiles_to_iupac_batch'):
        assert symbol in nispo.__all__
        assert callable(getattr(nispo, symbol))
    assert isinstance(nispo.__version__, str) and nispo.__version__


def test_unknown_export_raises_attribute_error():
    import nispo

    with pytest.raises(AttributeError):
        getattr(nispo, '__unregistered_api_contract_probe__')


@pytest.mark.modes("full", "public")
def test_molecule_call_preserves_graph_and_user_properties(naming, assert_graphs):
    mol = Chem.MolFromSmiles('CCO')
    mol.SetProp('caller_label', 'independent-api-control')
    mol.GetAtomWithIdx(0).SetIntProp('caller_index', 7)
    mol.GetBondWithIdx(0).SetBoolProp('caller_flag', True)
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)

    name = naming.molecule_name(mol)

    assert name
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    assert_graphs(['CCO'], [name])


@pytest.mark.modes("full", "public")
@pytest.mark.parametrize('kind', ['none', 'empty', 'wrong-python-type'])
def test_invalid_molecule_returns_no_name(naming, kind):
    value = None if kind == 'none' else Chem.Mol() if kind == 'empty' else object()
    assert naming.molecule_name(value) == ''


@pytest.mark.current_only
@pytest.mark.parametrize('value', INVALID_SMILES)
def test_invalid_structured_input_has_explicit_status(naming, value):
    if not naming.supports_results:
        pytest.skip('The public release does not expose structured results.')
    result = naming.result(value)
    assert not result.name
    assert result.generation_status == 'invalid-input'
    assert result.preferred_status == 'not-assessed'


@pytest.mark.current_only
def test_structured_delivery_matches_the_string_api(naming):
    if not naming.supports_results:
        pytest.skip('The public release does not expose structured results.')
    result = naming.result('CCO')
    assert result.generation_status == 'available'
    assert result.name == naming.name('CCO')
    assert result.preferred_status in {'resolved', 'unresolved', 'not-assessed'}
    if result.preferred_status == 'resolved':
        assert result.justification is not None and result.justification.complete
        assert result.justification.name == result.name
    if naming.mode == 'fast':
        assert result.preferred_status == 'unresolved'
        assert result.unresolved


@pytest.mark.current_only
def test_structured_batch_preserves_generator_slots_and_duplicate_values(naming):
    if not naming.supports_results:
        pytest.skip('The public release does not expose structured results.')
    results = naming.results(iter(('CCC', '', 'CCC', None)))
    assert len(results) == 4
    assert results[0].name and results[0].generation_status == 'available'
    assert results[0] == results[2]
    for result in (results[1], results[3]):
        assert not result.name
        assert result.generation_status == 'invalid-input'
        assert result.preferred_status == 'not-assessed'


@pytest.mark.full_only
def test_molecule_result_preserves_caller_data_and_matches_string_delivery(naming):
    mol = Chem.MolFromSmiles('CCO')
    mol.SetProp('caller_label', 'structured-api-control')
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    result = naming.molecule_result(mol)
    assert result.name == naming.name('CCO')
    assert result.generation_status == 'available'
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


@pytest.mark.current_only
def test_result_value_objects_are_immutable_and_pickleable():
    from nispo import NamingResult, SelectionLimitation

    issue = SelectionLimitation('runtime', 'synthetic-rule', 'synthetic limitation')
    result = NamingResult('', 'timeout', 'not-assessed', (issue,), 'synthetic-route')
    assert pickle.loads(pickle.dumps(result)) == result
    with pytest.raises(FrozenInstanceError):
        result.generation_status = 'available'
    with pytest.raises(FrozenInstanceError):
        issue.reason = 'replacement'


@pytest.mark.full_only
@pytest.mark.parametrize('function_name', [
    'smiles_to_iupac', 'smiles_to_iupac_result', 'mol_to_iupac',
    'mol_to_iupac_result', 'smiles_to_iupac_batch', 'smiles_to_iupac_batch_results',
])
def test_every_timed_entrypoint_validates_options_before_naming(function_name):
    api = importlib.import_module('nispo.api')
    function = getattr(api, function_name)
    value = [] if 'batch' in function_name else None
    with pytest.raises(ValueError):
        function(value, timeout_seconds=0)
    with pytest.raises(ValueError):
        function(value, timeout_seconds=None, fallback_to_fast='yes')


@pytest.mark.full_only
@pytest.mark.parametrize('seconds', [
    pytest.param(-1, id='negative'),
    pytest.param(True, id='boolean'),
    pytest.param(math.inf, id='infinite'),
    pytest.param(math.nan, id='nan'),
    pytest.param('2', id='text'),
    pytest.param(10**1000, id='overflow'),
])
def test_timeout_rejects_nonfinite_and_non_numeric_limits(seconds):
    from nispo import smiles_to_iupac_result

    with pytest.raises(ValueError):
        smiles_to_iupac_result(None, timeout_seconds=seconds)


@pytest.mark.full_only
def test_timeout_and_fallback_defaults_are_part_of_the_public_signature():
    from nispo import smiles_to_iupac_result

    parameters = inspect.signature(smiles_to_iupac_result).parameters
    assert parameters['timeout_seconds'].default == 10.0
    assert parameters['fallback_to_fast'].default is True


# Originally test_telemetry_contracts.py
"""Small benchmark telemetry contracts, independent of any molecule panel.

Counter labels below are invented reporting buckets, not naming rules. The
only chemistry controls are independently selected textbook ethanol and
ethanoic acid, used to check that optional instrumentation preserves output.
No reference name spelling is asserted.
"""


from copy import deepcopy
import importlib
import json

import pytest


@pytest.fixture
def telemetry_module(naming):
    module = "nispo._fast_engine.telemetry" if naming.mode == "fast" else "nispo.telemetry"
    return importlib.import_module(module)


def test_empty_telemetry_is_a_fresh_merge_identity(telemetry_module):
    empty = telemetry_module.empty_telemetry_dict()
    other = telemetry_module.empty_telemetry_dict()
    assert telemetry_module.merge_telemetry_dicts(iter((None, {}, empty))) == other
    assert all(value == {} if isinstance(value, dict) else value == 0 for value in empty.values())

    empty["candidate_count"]["independent-bucket"] = 1
    assert other["candidate_count"] == {}
    assert telemetry_module.empty_telemetry_dict()["candidate_count"] == {}


def test_merge_adds_shared_buckets_and_preserves_disjoint_buckets(telemetry_module):
    left = {
        "candidate_count": {"shared": 2, "left": 1},
        "rule_seconds": {"shared": 0.125},
        "branch_prefix_calls": 3,
    }
    right = {
        "candidate_count": {"shared": 3, "right": 4},
        "rule_seconds": {"shared": 0.25, "right": 0.5},
        "branch_prefix_calls": 4,
        "substructure_match_calls": 2,
    }
    merged = telemetry_module.merge_telemetry_dicts(iter((None, left, {}, right)))
    assert merged["candidate_count"] == {"left": 1, "right": 4, "shared": 5}
    assert merged["rule_seconds"] == {"right": 0.5, "shared": 0.375}
    assert merged["branch_prefix_calls"] == 7
    assert merged["substructure_match_calls"] == 2


def test_worker_chunk_grouping_and_order_do_not_change_totals(telemetry_module):
    parts = [
        {"candidate_count": {"a": 1}, "substructure_match_calls": 1},
        {"candidate_count": {"a": 2, "b": 1}, "substructure_match_calls": 2},
        {"candidate_count": {"c": 3}, "substructure_match_calls": 4},
    ]
    merge = telemetry_module.merge_telemetry_dicts
    expected = merge(parts)
    assert merge(reversed(parts)) == expected
    assert merge((merge(parts[:2]), parts[2])) == expected
    assert merge((parts[0], merge(parts[1:]))) == expected


def test_merging_does_not_mutate_or_alias_worker_payloads(telemetry_module):
    payload = {"candidate_count": {"independent-bucket": 2}, "branch_prefix_calls": 1}
    before = deepcopy(payload)
    merged = telemetry_module.merge_telemetry_dicts([payload, payload])
    assert payload == before
    merged["candidate_count"]["independent-bucket"] = 100
    assert payload == before


def test_every_reported_counter_survives_chunk_merging(telemetry_module):
    # Discover the serialized benchmark schema, rather than naming private
    # implementation fields or assuming a particular set of engine rules.
    schema = telemetry_module.empty_telemetry_dict()
    payload = {
        key: {"independent-bucket": 1} if isinstance(value, dict) else 1
        for key, value in schema.items()
    }
    merged = telemetry_module.merge_telemetry_dicts([payload, payload])
    for key, value in payload.items():
        assert merged[key] == ({"independent-bucket": 2} if isinstance(value, dict) else 2), key


def test_serialization_is_deterministic_json_safe_and_independent(telemetry_module):
    telemetry = telemetry_module.CandidateTelemetry()
    telemetry.candidate_count.update({"zeta": 2, "alpha": 1, "unused": 0})
    telemetry.rule_seconds.update({"zeta": 0.123456789, "alpha": 0.25})
    telemetry.branch_prefix_calls = 3

    serialized = telemetry_module.telemetry_to_dict(telemetry)
    assert json.loads(json.dumps(serialized)) == serialized
    assert list(serialized["candidate_count"]) == ["alpha", "zeta"]
    assert serialized["candidate_count"] == {"alpha": 1, "zeta": 2}
    assert serialized["rule_seconds"]["zeta"] == pytest.approx(0.123456789, abs=1e-6)
    assert serialized["branch_prefix_calls"] == 3

    serialized["candidate_count"]["alpha"] = 99
    assert telemetry.candidate_count["alpha"] == 1
    assert telemetry_module.telemetry_to_dict(telemetry)["candidate_count"]["alpha"] == 1


def test_compact_summary_preserves_path_totals_and_stable_top_order(telemetry_module, tmp_path):
    payload = {
        "candidate_count": {"beta": 3, "alpha": 3, "gamma": 1},
        "rule_seconds": {"alpha": 0.25, "beta": 0.5},
        "rendered_candidate_count": {"alpha": 2, "beta": 3},
        "rejected_candidate_count": {"alpha": 1},
        "candidate_facts_total": {"alpha": 7},
        "branch_prefix_calls": 7,
        "branch_prefix_cache_hits": 2,
        "substructure_match_calls": 9,
    }
    before = deepcopy(payload)
    output = tmp_path / "telemetry.json"
    summary = telemetry_module.compact_telemetry_summary(payload, output)
    assert summary["path"] == str(output)
    assert [row["key"] for row in summary["top_candidate_count"]] == ["alpha", "beta", "gamma"]
    assert summary["top_rule_seconds"][0] == {"key": "beta", "value": 0.5}
    assert summary["total_rendered_candidates"] == 5
    assert summary["total_rejected_candidates"] == 1
    assert summary["total_candidate_facts"] == 7
    assert summary["total_branch_prefix_calls"] == 7
    assert summary["total_branch_prefix_cache_hits"] == 2
    assert summary["total_substructure_match_calls"] == 9
    assert payload == before


def test_compact_empty_summary_has_no_stale_work(telemetry_module, tmp_path):
    summary = telemetry_module.compact_telemetry_summary({}, tmp_path / "empty.json")
    for key, value in summary.items():
        if key.startswith("top_"):
            assert value == [], key
        elif key.startswith("total_"):
            assert value == 0, key


@pytest.mark.modes("full", "public")
@pytest.mark.parametrize("detail", [False, True], ids=["basic", "shape-detail"])
def test_benchmark_instrumentation_preserves_small_public_batch(naming, assert_graphs, monkeypatch, detail):
    from nispo.api import _smiles_to_iupac_batch_with_telemetry

    # This documented benchmark switch keeps both paths instrumented without
    # routing a chemistry comparison through machine-dependent API deadlines.
    monkeypatch.setenv("NISPO_BENCH_FULL_TELEMETRY", "1")
    inputs = ("CCO", "CC(=O)O")
    expected = naming.names(inputs)
    names, stats = _smiles_to_iupac_batch_with_telemetry(iter(inputs), branch_miss_shape_detail=detail)

    assert names == expected
    assert all(names)
    assert_graphs(inputs, names)
    assert json.loads(json.dumps(stats)) == stats
    assert any(
        count > 0
        for field in ("candidate_count", "rule_evaluations", "substructure_match_calls_by_rule")
        for count in stats[field].values()
    )


@pytest.mark.modes("full", "public")
def test_rejected_benchmark_inputs_preserve_positions_without_work(naming, monkeypatch):
    from nispo.api import _smiles_to_iupac_batch_with_telemetry
    from nispo.telemetry import empty_telemetry_dict

    monkeypatch.setenv("NISPO_BENCH_FULL_TELEMETRY", "1")
    names, stats = _smiles_to_iupac_batch_with_telemetry(iter(("not-a-smiles", "")))
    assert names == ["", ""]
    assert stats == empty_telemetry_dict()
