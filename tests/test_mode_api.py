"""Mode dispatch contracts, independent of nomenclature wording.

Graphs are elementary alcohol/acid/halogenated stereocentre,
alkene and carbon-isotope API controls selected independently
of evaluation panels. Synthetic payloads below test routing
and transport, not generated chemical names.
"""
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import math
import subprocess
import sys
from threading import Barrier

import pytest

import nispo
from nispo import (smiles_to_iupac, smiles_to_iupac_result,
                   smiles_to_iupac_batch, smiles_to_iupac_batch_results)
from nispo.fast_selection import smiles_to_iupac_fast, smiles_to_iupac_fast_result


SCALAR = (smiles_to_iupac, smiles_to_iupac_result)
BATCH = (smiles_to_iupac_batch, smiles_to_iupac_batch_results)


def test_default_mode_and_module_aliases_are_compatible():
    from nispo import api

    for call in (*SCALAR, *BATCH):
        assert inspect.signature(call).parameters['mode'].default == 'full'
        assert getattr(api, call.__name__) is call
    values = ['CCO', 'CC(=O)O', '', 'CCO']
    for call in SCALAR:
        assert call(values[0], timeout_seconds=None) == call(
            values[0], mode='full', timeout_seconds=None)
    for call in BATCH:
        assert call(values, timeout_seconds=None) == call(values, mode='full', timeout_seconds=None)


def test_fast_mode_matches_existing_fast_apis_and_recovers_graphs(assert_graphs):
    values = ['CCO', 'C[C@H](F)Cl', 'F/C=C/F']
    names = []
    for value in values:
        name = smiles_to_iupac(value, mode='fast')
        result = smiles_to_iupac_result(value, mode='fast')
        assert name == smiles_to_iupac_fast(value)
        assert result == smiles_to_iupac_fast_result(value)
        names.append(name)
    assert smiles_to_iupac_batch(values, mode='fast') == names
    assert [r.name for r in smiles_to_iupac_batch_results(values, mode='fast')] == names
    assert_graphs(values, names, raw_only=True)


def test_common_fast_entry_does_not_import_full_engine():
    script = '''
import json, sys
from nispo import (smiles_to_iupac, smiles_to_iupac_result,
                   smiles_to_iupac_batch, smiles_to_iupac_batch_results)
assert 'nispo.api' not in sys.modules
assert smiles_to_iupac('CCO', mode='fast')
assert smiles_to_iupac_result('CCO', mode='fast').name
assert smiles_to_iupac_batch(['CCO'], mode='fast')[0]
assert smiles_to_iupac_batch_results(['CCO'], mode='fast')[0].name
print(json.dumps([m for m in ('nispo.api', 'nispo.rule_engine', 'nispo.parent_rules') if m in sys.modules]))
'''
    output = subprocess.run([sys.executable, '-c', script], capture_output=True,
                            text=True, check=True, timeout=90)
    assert json.loads(output.stdout) == []


@pytest.mark.parametrize('call', (*SCALAR, *BATCH))
@pytest.mark.parametrize('mode', ['slow', 'FAST', '', None, True, 1, []],
                         ids=['unknown', 'uppercase', 'empty', 'none', 'boolean', 'integer', 'list'])
def test_invalid_mode_is_rejected(call, mode):
    with pytest.raises(ValueError, match='mode'):
        call(['CCO'] if call in BATCH else 'CCO', mode=mode)


@pytest.mark.parametrize('seconds', [0, -1, True, math.inf, math.nan, '10'])
def test_fast_mode_validates_handoff_timeout(seconds):
    with pytest.raises(ValueError, match='timeout_seconds'):
        smiles_to_iupac('CCO', mode='fast', timeout_seconds=seconds)


def test_invalid_mode_does_not_consume_batch():
    consumed = []

    def values():
        consumed.append(True)
        yield 'CCO'

    with pytest.raises(ValueError, match='mode'):
        smiles_to_iupac_batch(values(), mode='invalid')
    assert not consumed


@pytest.mark.parametrize('structured', [False, True], ids=['string', 'structured'])
def test_fast_batch_preserves_generator_order_and_shares_duplicates(monkeypatch, structured):
    from nispo import fast_selection
    calls = []
    consumed = []

    def fake(value):
        calls.append(value)
        return (nispo.NamingResult('', 'unavailable', 'not-assessed', route=value)
                if structured else 'delivery-' + str(len(calls)))

    monkeypatch.setattr(fast_selection, 'smiles_to_iupac_fast_result' if structured else
                        'smiles_to_iupac_fast', fake)

    def values():
        for index, value in enumerate(('a', 'b', 'a')):
            consumed.append(index)
            yield value

    call = smiles_to_iupac_batch_results if structured else smiles_to_iupac_batch
    results = call(values(), mode='fast')
    assert consumed == [0, 1, 2] and calls == ['a', 'b']
    assert len(results) == 3 and results[0] == results[2]
    assert results[0] != results[1]


@pytest.mark.parametrize('structured', [False, True], ids=['string', 'structured'])
def test_fast_handoff_honors_timeout_and_restores_direct_fast_defaults(monkeypatch, structured):
    from nispo import api
    observed = []

    def full(value, *, timeout_seconds=10.0, fallback_to_fast):
        observed.append((timeout_seconds, fallback_to_fast))
        return nispo.NamingResult('', 'timeout', 'not-assessed') if structured else ''

    monkeypatch.setattr(api, 'smiles_to_iupac_result' if structured else 'smiles_to_iupac', full)
    call = smiles_to_iupac_result if structured else smiles_to_iupac
    direct = smiles_to_iupac_fast_result if structured else smiles_to_iupac_fast
    call('[13CH3]CO', mode='fast', timeout_seconds=None)
    call('[13CH3]CO', mode='fast', timeout_seconds=0.25, fallback_to_fast=False)
    direct('[13CH3]CO')
    assert observed == [(None, False), (0.25, False), (10.0, False)]


def test_concurrent_fast_handoffs_keep_their_own_deadlines(monkeypatch):
    from nispo import api
    barrier = Barrier(4, timeout=10)
    timeouts = (None, 0.25, 30.0, 10.0)

    def full(value, *, timeout_seconds=10.0, fallback_to_fast):
        assert fallback_to_fast is False
        barrier.wait()
        return nispo.NamingResult('', 'timeout', 'not-assessed', route=repr(timeout_seconds))

    monkeypatch.setattr(api, 'smiles_to_iupac_result', full)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda seconds: smiles_to_iupac_result(
            '[13CH3]CO', mode='fast', timeout_seconds=seconds), timeouts))
    assert [r.route for r in results] == [repr(value) for value in timeouts]
