"""Narrow public process contracts with independently selected simple controls.

The chemistry controls are ethanol, propane and a textbook stereocentre,
selected for transport/annotation behavior rather than benchmark outcomes.
Only synthetic peers use short deadlines. Real naming gets a generous limit;
correctness tests elsewhere exercise unrestricted chemistry directly.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.current_only, pytest.mark.workers]


@pytest.fixture
def controlled(monkeypatch, tmp_path):
    from nispo import _timed_parallel
    from worker_helpers import controlled_worker

    log = tmp_path/'worker.jsonl'
    monkeypatch.setenv('NISPO_CORE_WORKER_LOG', str(log))
    monkeypatch.setattr(_timed_parallel, '_deadline_worker', controlled_worker)
    return log


def _requests(log, value):
    return [row for row in (json.loads(line) for line in log.read_text().splitlines())
            if row['event']=='started' and row['value']==value]


def _assert_gone(pid):
    if os.name=='posix':
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def _assert_completed_or_explicit_timeout(actual, expected):
    if actual.generation_status=='timeout':
        assert not actual.name and actual.preferred_status=='not-assessed'
        assert actual.unresolved and all(issue.stage=='runtime' for issue in actual.unresolved)
    else:
        assert actual==expected


def test_timeout_and_crash_keep_order_and_recover(controlled):
    from nispo import IUPACBatchPool

    with IUPACBatchPool(workers=1, timeout_seconds=0.4, fallback_to_fast=False) as pool:
        results = pool.smiles_to_iupac_batch_results(['first', 'hang', 'first', 'crash', 'last'])
        assert [r.generation_status for r in results]==['available', 'timeout', 'available', 'error', 'available']
        assert results[0]==results[2]
        assert results[0].name.endswith('::first') and results[-1].name.endswith('::last')
        assert not results[1].name and not results[3].name
        assert all(r.unresolved and r.unresolved[0].stage=='runtime' for r in (results[1], results[3]))
        assert pool.smiles_to_iupac_batch(['next'])[0].endswith('::next')
    assert len(_requests(controlled, 'first'))==1
    _assert_gone(_requests(controlled, 'hang')[0]['pid'])


def test_recycling_counts_unique_requests_across_batches(controlled):
    from nispo import IUPACBatchPool

    with IUPACBatchPool(workers=1, timeout_seconds=1, max_requests_per_worker=2) as pool:
        first = pool.smiles_to_iupac_batch(['a', 'a'])
        second = pool.smiles_to_iupac_batch(['b', 'c'])
    assert first[0]==first[1]
    assert first[0].split('::')[0]==second[0].split('::')[0]
    assert second[0].split('::')[0]!=second[1].split('::')[0]


def test_disabled_recycling_still_replaces_a_timed_out_worker(controlled):
    from nispo import IUPACBatchPool

    with IUPACBatchPool(workers=1, timeout_seconds=0.4, fallback_to_fast=False,
                       max_requests_per_worker=None) as pool:
        before = pool.smiles_to_iupac_batch(['a', 'b'])
        results = pool.smiles_to_iupac_batch_results(['hang', 'after'])
    assert before[0].split('::')[0]==before[1].split('::')[0]
    assert results[0].generation_status=='timeout'
    assert results[1].generation_status=='available'
    assert before[0].split('::')[0]!=results[1].name.split('::')[0]


def test_other_worker_can_release_a_waiting_request(controlled, monkeypatch, tmp_path):
    from nispo import IUPACBatchPool

    monkeypatch.setenv('NISPO_CORE_RELEASE', str(tmp_path/'release'))
    with IUPACBatchPool(workers=2, timeout_seconds=3, fallback_to_fast=False) as pool:
        results = pool.smiles_to_iupac_batch_results(['hold', 'release'])
    assert all(r.generation_status=='available' for r in results)
    assert results[0].name.split('::')[0]!=results[1].name.split('::')[0]


def test_initialization_does_not_consume_request_budget(controlled, monkeypatch):
    from nispo import IUPACBatchPool

    monkeypatch.setenv('NISPO_CORE_STARTUP_DELAY', '0.6')
    with IUPACBatchPool(workers=1, timeout_seconds=0.4, fallback_to_fast=False) as pool:
        assert pool.smiles_to_iupac_batch(['ready'])[0].endswith('::ready')


def test_startup_failure_can_be_retried(controlled, monkeypatch):
    from nispo import IUPACBatchPool

    monkeypatch.setenv('NISPO_CORE_FAIL_START', '1')
    with IUPACBatchPool(workers=1, timeout_seconds=1) as pool:
        with pytest.raises(RuntimeError, match='initialization'):
            pool.smiles_to_iupac_batch(['a'])
        monkeypatch.delenv('NISPO_CORE_FAIL_START')
        assert pool.smiles_to_iupac_batch(['a'])[0].endswith('::a')


@pytest.mark.skipif(os.name!='posix', reason='partial pipe frames use POSIX descriptors')
def test_partial_message_does_not_prevent_timeout(controlled):
    from nispo import IUPACBatchPool

    with IUPACBatchPool(workers=1, timeout_seconds=0.4, fallback_to_fast=False) as pool:
        results = pool.smiles_to_iupac_batch_results(['partial', 'after'])
    assert [r.generation_status for r in results]==['timeout', 'available']
    _assert_gone(_requests(controlled, 'partial')[0]['pid'])


def test_fallback_success_decline_and_second_timeout_are_explicit(controlled):
    from nispo import IUPACBatchPool

    with IUPACBatchPool(workers=1, timeout_seconds=0.4) as pool:
        results = pool.smiles_to_iupac_batch_results(['fallback', 'decline', 'both-hang'])
    assert [r.generation_status for r in results]==['available', 'timeout', 'timeout']
    assert results[0].preferred_status=='unresolved'
    assert results[0].route.startswith('full-timeout:')
    assert all(r.unresolved and r.unresolved[0].stage=='runtime' for r in results)
    assert not results[1].name and not results[2].name
    assert len(_requests(controlled, 'both-hang'))==2


def test_completed_provisional_result_survives_fallback_timeout(controlled):
    from nispo import IUPACBatchPool

    with IUPACBatchPool(workers=1, timeout_seconds=0.4) as pool:
        result = pool.smiles_to_iupac_batch_results(['provisional'])[0]
    assert result.name.endswith('::provisional')
    assert result.generation_status=='available' and result.preferred_status=='unresolved'
    assert any(issue.rule=='full-timeout' for issue in result.unresolved)


def test_fast_worker_exit_reports_failure_or_preserves_completed_provisional(controlled):
    from nispo import IUPACBatchPool

    with IUPACBatchPool(workers=1, timeout_seconds=0.4) as pool:
        failed, provisional = pool.smiles_to_iupac_batch_results(['fast-crash', 'provisional-crash'])
    assert failed.generation_status=='timeout' and not failed.name
    assert failed.unresolved and all(issue.stage=='runtime' for issue in failed.unresolved)
    assert provisional.generation_status=='available' and provisional.name.endswith('::provisional')
    assert provisional.preferred_status=='unresolved'
    assert any(issue.rule=='full-timeout' for issue in provisional.unresolved)


def test_context_exception_closes_workers_and_pool(controlled):
    from nispo import IUPACBatchPool

    pool = IUPACBatchPool(workers=1, timeout_seconds=1)
    with pytest.raises(LookupError):
        with pool:
            pid = int(pool.smiles_to_iupac_batch(['a'])[0].split('::')[0])
            raise LookupError('caller stopped')
    _assert_gone(pid)
    pool.close()
    with pytest.raises(RuntimeError, match='closed'):
        pool.smiles_to_iupac_batch([])
    with pytest.raises(RuntimeError, match='closed'):
        pool.__enter__()


@pytest.fixture(scope='module')
def real_baseline():
    from nispo import smiles_to_iupac_batch_results

    values = ['CCO', 'CCC', '']
    results = smiles_to_iupac_batch_results(values, timeout_seconds=None)
    assert all(r.generation_status=='available' and r.name for r in results[:2])
    assert results[2].generation_status=='invalid-input' and not results[2].name
    return dict(zip(values, results))


@pytest.mark.parametrize('timeout', [None, 15.0], ids=['untimed-spawn', 'timed-spawn'])
def test_real_spawned_pool_preserves_results_and_duplicates(real_baseline, timeout):
    from nispo import IUPACBatchPool

    inputs = ['CCO', 'CCC', 'CCO', '']
    with IUPACBatchPool(workers=2, chunk_size=1, timeout_seconds=timeout,
                       fallback_to_fast=False) as pool:
        actual = pool.smiles_to_iupac_batch_results(iter(inputs))
        assert len(actual)==len(inputs) and actual[0]==actual[2]
        for value, result in zip(inputs, actual, strict=True):
            if timeout is None:
                assert result==real_baseline[value]
            else:
                _assert_completed_or_explicit_timeout(result, real_baseline[value])
        if timeout is None:
            assert pool.smiles_to_iupac_batch(reversed(inputs))==[real_baseline[s].name for s in reversed(inputs)]
        else:
            assert pool.smiles_to_iupac_batch(['CCO'])[0] in {'', real_baseline['CCO'].name}
        assert pool.smiles_to_iupac_batch([])==[]


def test_untimed_single_worker_pool_keeps_the_serial_contract(real_baseline):
    from nispo import IUPACBatchPool

    values = ['CCC', 'CCO', 'CCC']
    with IUPACBatchPool(workers=1, timeout_seconds=None) as pool:
        assert pool.smiles_to_iupac_batch(values)==[real_baseline[s].name for s in values]


def test_real_full_timeout_handoff_is_safe_and_bounded(monkeypatch, assert_graphs):
    from nispo import IUPACBatchPool, _timed_parallel
    from worker_helpers import delayed_full_worker

    monkeypatch.setattr(_timed_parallel, '_deadline_worker', delayed_full_worker)
    with IUPACBatchPool(workers=1, timeout_seconds=2) as pool:
        result = pool.smiles_to_iupac_batch_results(['CCO'])[0]
    assert result.generation_status in {'available', 'timeout'}
    assert result.unresolved and any(issue.stage=='runtime' for issue in result.unresolved)
    if result.name:
        assert result.generation_status=='available' and result.preferred_status=='unresolved'
        assert_graphs(['CCO'], [result.name])
    else:
        assert result.generation_status=='timeout' and result.preferred_status=='not-assessed'


def test_real_single_subprocess_preserves_molecule_metadata():
    from rdkit import Chem
    from nispo import mol_to_iupac_result

    molecule = Chem.MolFromSmiles('CCO')
    molecule.SetProp('caller-note', 'preserve')
    molecule.GetAtomWithIdx(0).SetAtomMapNum(19)
    before = molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)
    expected = mol_to_iupac_result(molecule, timeout_seconds=None)
    actual = mol_to_iupac_result(molecule, timeout_seconds=15, fallback_to_fast=False)
    _assert_completed_or_explicit_timeout(actual, expected)
    from nispo import mol_to_iupac
    assert mol_to_iupac(molecule, timeout_seconds=15, fallback_to_fast=False) in {'', expected.name}
    assert molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)==before


def test_relative_stereo_metadata_survives_subprocess_transport():
    from rdkit import Chem
    from nispo import mol_to_iupac_result

    molecule = Chem.MolFromSmiles('C[C@H](F)Cl |&1:1|')
    before = molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)
    actual = mol_to_iupac_result(molecule, timeout_seconds=15, fallback_to_fast=False)
    expected = mol_to_iupac_result(molecule, timeout_seconds=None)
    _assert_completed_or_explicit_timeout(actual, expected)
    assert not actual.name and actual.generation_status in {'unavailable', 'timeout'}
    assert actual.unresolved
    assert molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)==before


def test_concurrent_single_callers_get_their_own_results(real_baseline):
    from nispo import smiles_to_iupac_result

    values = ['CCO', 'CCC', 'CCO']
    call = partial(smiles_to_iupac_result, timeout_seconds=15, fallback_to_fast=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(call, values))
    for value, result in zip(values, actual, strict=True):
        _assert_completed_or_explicit_timeout(result, real_baseline[value])


def test_default_timeout_is_declared_without_changing_correctness_defaults():
    from nispo import IUPACBatchPool, smiles_to_iupac_result

    signature = inspect.signature(smiles_to_iupac_result)
    assert signature.parameters['timeout_seconds'].default==10.0
    assert signature.parameters['fallback_to_fast'].default is True
    with IUPACBatchPool() as pool:
        assert pool.timeout_seconds==10.0 and pool.fallback_to_fast is True
        assert pool.max_requests_per_worker==100000


def _script_environment():
    import nispo

    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(nispo.__file__).resolve().parent.parent)+os.pathsep+env.get('PYTHONPATH', '')
    return env


_COVERAGE_FLUSH_SCRIPT = (
    "def save_trace():\n"
    "    if os.environ.get('COVERAGE_PROCESS_CONFIG') or os.environ.get('COVERAGE_PROCESS_START'):\n"
    "        from coverage import Coverage\n"
    "        active = Coverage.current()\n"
    "        if active is not None:\n"
    "            active.save()\n"
)


@pytest.mark.parametrize('abrupt_exit', [False, True], ids=['normal-exit', 'abrupt-owner-exit'])
def test_unguarded_script_releases_owned_worker(tmp_path, abrupt_exit):
    if abrupt_exit and os.name!='posix':
        pytest.skip('abrupt-owner monitoring uses POSIX parent identity')
    script = tmp_path/'ordinary_script.py'
    script.write_text(
        "import os\nfrom nispo import smiles_to_iupac_result\n"
        +_COVERAGE_FLUSH_SCRIPT+
        "result = smiles_to_iupac_result('CCO', timeout_seconds=15, fallback_to_fast=False)\n"
        "assert result.generation_status in {'available', 'timeout'}\n"
        "assert result.name if result.generation_status == 'available' else result.unresolved\n"
        "print('finished', flush=True)\n"+('save_trace(); os._exit(0)\n' if abrupt_exit else ''))
    # Capture waits for EOF on the inherited stderr pipe. In the abrupt case
    # it cannot finish while an orphaned naming worker keeps that pipe open.
    completed = subprocess.run([sys.executable, str(script)], env=_script_environment(),
                               capture_output=True, text=True, check=True, timeout=90)
    assert completed.stdout.strip()=='finished'


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='fork is unavailable')
def test_forked_caller_does_not_share_a_live_worker(tmp_path):
    script = tmp_path/'forked_script.py'
    script.write_text(
        "import os\nfrom nispo import smiles_to_iupac_result\n"
        +_COVERAGE_FLUSH_SCRIPT+
        "def named():\n"
        "    result = smiles_to_iupac_result('CCO', timeout_seconds=15, fallback_to_fast=False)\n"
        "    assert result.generation_status in {'available', 'timeout'}\n"
        "    return bool(result.name if result.generation_status == 'available' else result.unresolved)\n"
        "assert named()\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    completed = named()\n"
        "    save_trace()\n"
        "    os._exit(0 if completed else 2)\n"
        "_, status = os.waitpid(pid, 0)\n"
        "assert os.waitstatus_to_exitcode(status) == 0\n"
        "assert named()\n")
    subprocess.run([sys.executable, str(script)], env=_script_environment(),
                   capture_output=True, text=True, check=True, timeout=120)
