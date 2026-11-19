"""Benchmark telemetry separates completed naming from bounded execution.

One independently chosen two-carbon alcohol exercises the audit protocol;
synthetic runtime labels describe transport outcomes, not nomenclature.
No generated-name spelling or parent selection is prescribed.
"""
import importlib.util
import json
from pathlib import Path

import pytest
from rdkit import Chem

import nispo
from nispo.naming_result import NamingResult, SelectionLimitation


@pytest.fixture
def composition_audit(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / 'tools'
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        'composition_audit_contract', tools / 'check_release_acceptance.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'isolate_nispo_imports', lambda root: None)
    monkeypatch.setattr(module, 'check_nispo_scope', lambda root: None)
    monkeypatch.setattr(module, 'controls', lambda: [
        {'id': 'independent-alcohol', 'smiles': 'CCO', 'expected': None}])
    monkeypatch.setattr(module, 'opsin', lambda names: [Chem.MolFromSmiles('CCO')])
    return module


def install_controlled_apis(monkeypatch, default_result, *, bad_unbounded_batch=False):
    full = nispo.smiles_to_iupac_result('CCO', timeout_seconds=None)
    assert full.name
    monkeypatch.setattr(nispo, 'mol_to_iupac_result', lambda mol, **kwargs: full)
    monkeypatch.setattr(nispo, 'mol_to_iupac', lambda mol, **kwargs: full.name)

    def batch(values, **kwargs):
        if kwargs.get('timeout_seconds', 'default') is None:
            return ['unbounded-transport-error' if bad_unbounded_batch else full.name
                    for _ in values]
        return [default_result.name for _ in values]

    def structured(values, **kwargs):
        result = full if kwargs.get('timeout_seconds', 'default') is None else default_result
        return [result for _ in values]

    monkeypatch.setattr(nispo, 'smiles_to_iupac_batch', batch)
    monkeypatch.setattr(nispo, 'smiles_to_iupac_batch_results', structured)
    return full


@pytest.mark.parametrize('available', [True, False], ids=['fallback', 'exhausted'])
def test_audit_keeps_deadline_outcomes_separate_from_completed_parity(
        composition_audit, monkeypatch, tmp_path, available):
    limitation = SelectionLimitation('runtime', 'full-timeout', 'Controlled deadline.')
    result = NamingResult('runtime-alternative' if available else '',
                          'available' if available else 'timeout',
                          'unresolved' if available else 'not-assessed',
                          (limitation,), 'full-timeout:controlled')
    install_controlled_apis(monkeypatch, result)
    output = tmp_path / 'report.json'
    assert composition_audit.run(tmp_path, output) == 0
    report = json.loads(output.read_text())
    assert report['batch_parity'] is True
    assert report['default_batch_contract'] is True
    assert report['default_batch_runtime_limited'] == 1
    event = report['default_runtime_events'][0]
    assert event['id'] == 'independent-alcohol'
    assert event['default_result']['generation_status'] == result.generation_status
    assert event['default_result']['name'] == result.name


@pytest.mark.parametrize('result', [
    NamingResult('', 'error', 'not-assessed'),
    NamingResult('silent-transport-change', 'available', 'unresolved'),
    NamingResult('', 'timeout', 'not-assessed'),
], ids=['error', 'silent-change', 'unexplained-timeout'])
def test_audit_still_fails_unexplained_default_batch_discrepancies(
        composition_audit, monkeypatch, tmp_path, result):
    install_controlled_apis(monkeypatch, result)
    output = tmp_path / 'report.json'
    assert composition_audit.run(tmp_path, output) == 1
    report = json.loads(output.read_text())
    assert report['batch_parity'] is True
    assert report['default_batch_contract'] is False
    assert report['failures'] == ['independent-alcohol']
    assert report['rows'][0]['default_result']['generation_status'] == result.generation_status


def test_audit_records_the_input_and_both_outputs_for_unbounded_parity_failure(
        composition_audit, monkeypatch, tmp_path):
    limitation = SelectionLimitation('runtime', 'full-timeout', 'Controlled deadline.')
    result = NamingResult('', 'timeout', 'not-assessed', (limitation,),
                          'full-timeout:controlled')
    full = install_controlled_apis(monkeypatch, result, bad_unbounded_batch=True)
    output = tmp_path / 'report.json'
    assert composition_audit.run(tmp_path, output) == 1
    report = json.loads(output.read_text())
    assert report['batch_parity'] is False
    mismatch = report['batch_differences'][0]
    assert mismatch['index'] == 0 and mismatch['smiles'] == 'CCO'
    assert mismatch['scalar_name'] == full.name
    assert mismatch['batch_name'] != full.name
