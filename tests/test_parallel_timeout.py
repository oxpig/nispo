"""Public timeout behavior and worker ownership, using independent inputs."""
import math
import os
import time

import pytest

from nispo import IUPACBatchPool, smiles_to_iupac_batch_results
from nispo import _timed_parallel
from timeout_process_helper import controlled_worker


@pytest.mark.parametrize('timeout', [0, -1, True, math.inf, math.nan, '2', 10**1000])
def test_pool_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        IUPACBatchPool(timeout_seconds=timeout)


def test_timeout_stops_worker_preserves_order_and_continues(monkeypatch, tmp_path):
    pid_file = tmp_path / 'slow.pid'
    monkeypatch.setenv('NISPO_TEST_DEADLINE_PID', str(pid_file))
    monkeypatch.setattr(_timed_parallel, '_deadline_worker', controlled_worker)
    values = ['a', 'slow', 'b', 'slow', 'crash', 'c']
    with IUPACBatchPool(workers=1, timeout_seconds=0.2, fallback_to_fast=False) as pool:
        results = pool.smiles_to_iupac_batch_results(values)
        assert [r.generation_status for r in results] == [
            'available', 'timeout', 'available', 'timeout', 'error', 'available']
        assert [r.name for r in results] == ['named:a', '', 'named:b', '', '', 'named:c']
        assert results[1] == results[3]
        assert results[1].preferred_status == 'not-assessed'
        assert results[1].unresolved[0].stage == 'runtime'
        assert pool.smiles_to_iupac_batch(['d', 'slow', 'd']) == ['named:d', '', 'named:d']
    if os.name == 'posix':
        pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_real_timed_pool_matches_serial_for_completed_names():
    values = ['CCO', 'CC(=O)O', 'c1ccncc1', '', 'CCO']
    expected = smiles_to_iupac_batch_results(values)
    with IUPACBatchPool(workers=2, timeout_seconds=5.0) as pool:
        assert pool.smiles_to_iupac_batch_results(values) == expected


def test_real_worker_recycling_preserves_graphs_and_cross_batch_order(assert_graphs):
    # Independent elementary graphs and the first algorithmic SM4 fused-N
    # topology. No names or reference-panel structures are prescribed.
    values = ['CCO', 'C1CC2CCNC2C1', 'CC(=O)O', 'C[C@H](F)Cl', 'c1ccncc1', 'CCO']
    expected = smiles_to_iupac_batch_results(values, timeout_seconds=None)
    with IUPACBatchPool(workers=1, timeout_seconds=15, fallback_to_fast=False,
                       max_requests_per_worker=2) as pool:
        actual = pool.smiles_to_iupac_batch_results(values)
        assert actual == expected
        assert pool.smiles_to_iupac_batch_results(reversed(values)) == list(reversed(expected))
    assert all(result.generation_status == 'available' for result in actual)
    assert_graphs(values, [result.name for result in actual], raw_only=True)


def test_timed_workers_recycle_without_losing_results(monkeypatch):
    monkeypatch.setattr(_timed_parallel, '_deadline_worker', controlled_worker)
    with IUPACBatchPool(workers=1, timeout_seconds=0.2, max_requests_per_worker=2) as pool:
        names = pool.smiles_to_iupac_batch(['pid:a', 'pid:a'])
        assert names[0] == names[1]
        # The cap counts unique dispatched requests and persists across batches.
        names += pool.smiles_to_iupac_batch(['pid:b', 'pid:c'])
    first, duplicate, second, third = (name.split(':', 1)[0] for name in names)
    assert first == duplicate
    assert first == second and third != second


@pytest.mark.parametrize('limit', [0, -1, True, False, 1.5, math.inf, math.nan, '1000'])
def test_pool_rejects_invalid_worker_request_limit(limit):
    with pytest.raises(ValueError, match='max_requests_per_worker'):
        IUPACBatchPool(max_requests_per_worker=limit)


def test_disabled_recycling_retains_worker_across_batches(monkeypatch):
    monkeypatch.setattr(_timed_parallel, '_deadline_worker', controlled_worker)
    with IUPACBatchPool(workers=1, timeout_seconds=0.2, max_requests_per_worker=None) as pool:
        first = pool.smiles_to_iupac_batch(f'pid:{i}' for i in range(6000))
        second = pool.smiles_to_iupac_batch(f'pid:{i}' for i in range(6000, 10002))
    assert len(first) == 6000 and len(second) == 4002
    assert len({name.split(':', 1)[0] for name in first + second}) == 1


def test_disabled_recycling_still_replaces_timed_out_worker(monkeypatch, tmp_path):
    monkeypatch.setenv('NISPO_TEST_DEADLINE_PID', str(tmp_path / 'slow.pid'))
    monkeypatch.setattr(_timed_parallel, '_deadline_worker', controlled_worker)
    with IUPACBatchPool(workers=1, timeout_seconds=0.2, fallback_to_fast=False,
                       max_requests_per_worker=None) as pool:
        results = pool.smiles_to_iupac_batch_results(['pid:a', 'slow', 'pid:b'])
    assert [result.generation_status for result in results] == ['available', 'timeout', 'available']
    assert results[0].name.split(':', 1)[0] != results[2].name.split(':', 1)[0]


def test_timeout_does_not_block_other_workers(monkeypatch, tmp_path):
    monkeypatch.setenv('NISPO_TEST_DEADLINE_PID', str(tmp_path / 'slow.pid'))
    monkeypatch.setattr(_timed_parallel, '_deadline_worker', controlled_worker)
    with IUPACBatchPool(workers=2, timeout_seconds=0.2, fallback_to_fast=False) as pool:
        results = pool.smiles_to_iupac_batch_results(['slow', 'a', 'b', 'c'])
    assert [r.generation_status for r in results] == ['timeout', 'available', 'available', 'available']
    assert [r.name for r in results] == ['', 'named:a', 'named:b', 'named:c']


def test_worker_startup_failure_is_reported(monkeypatch):
    monkeypatch.setenv('NISPO_TEST_DEADLINE_FAIL_START', '1')
    monkeypatch.setattr(_timed_parallel, '_deadline_worker', controlled_worker)
    with IUPACBatchPool(workers=1, timeout_seconds=0.2, fallback_to_fast=False) as pool:
        with pytest.raises(RuntimeError, match='initialization'):
            pool.smiles_to_iupac_batch(['a'])


@pytest.mark.skipif(os.name != 'posix', reason='the deliberately partial pipe frame uses POSIX descriptors')
def test_partial_worker_message_cannot_block_the_deadline(monkeypatch):
    monkeypatch.setattr(_timed_parallel, '_deadline_worker', controlled_worker)
    began = time.monotonic()
    with IUPACBatchPool(workers=1, timeout_seconds=0.2, fallback_to_fast=False) as pool:
        results = pool.smiles_to_iupac_batch_results(['partial', 'a'])
    assert [r.generation_status for r in results] == ['timeout', 'available']
    assert time.monotonic() - began < 8


def test_default_limit_and_fast_fallback_are_enabled():
    pool = IUPACBatchPool(workers=1)
    assert pool.timeout_seconds == 10.0 and pool.fallback_to_fast is True
    assert pool.max_requests_per_worker == 100000
    pool.close()


def test_one_way_fast_fallback_is_explicit_and_bounded(monkeypatch, tmp_path):
    monkeypatch.setenv('NISPO_TEST_DEADLINE_PID', str(tmp_path / 'slow.pid'))
    monkeypatch.setattr(_timed_parallel, '_deadline_worker', controlled_worker)
    with IUPACBatchPool(workers=2, timeout_seconds=0.2) as pool:
        results = pool.smiles_to_iupac_batch_results(['slow', 'no-fast', 'both-slow', 'a'])
    assert [r.generation_status for r in results] == ['available', 'timeout', 'timeout', 'available']
    assert results[0].name == 'fast:slow' and results[0].preferred_status == 'unresolved'
    assert results[0].route.startswith('full-timeout:')
    assert results[0].unresolved[0].rule == 'full-timeout'
    assert results[1].name == results[2].name == ''


def test_completed_provisional_name_survives_fast_worker_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv('NISPO_TEST_DEADLINE_PID', str(tmp_path / 'slow.pid'))
    monkeypatch.setattr(_timed_parallel, '_deadline_worker', controlled_worker)
    with IUPACBatchPool(workers=1, timeout_seconds=0.2) as pool:
        results = pool.smiles_to_iupac_batch_results(['provisional-slow', 'a'])
    assert [result.generation_status for result in results] == ['available', 'available']
    assert results[0].name == 'provisional'
    assert results[0].preferred_status == 'unresolved'
    assert results[0].route == 'full-timeout:fast:structural-von-baeyer'
    assert results[1].name == 'named:a'
