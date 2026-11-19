"""Default bounded behavior on independent simple API controls."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest
from rdkit import Chem

import nispo
from nispo import (IUPACBatchPool, mol_to_iupac, mol_to_iupac_result,
                   smiles_to_iupac, smiles_to_iupac_result, smiles_to_iupac_batch,
                   smiles_to_iupac_batch_results)
from nispo import api, fast_selection


@pytest.mark.parametrize('function', [smiles_to_iupac, smiles_to_iupac_result,
                                      mol_to_iupac, mol_to_iupac_result,
                                      smiles_to_iupac_batch,
                                      smiles_to_iupac_batch_results])
def test_full_apis_enable_ten_second_default(function):
    params = inspect.signature(function).parameters
    assert params['timeout_seconds'].default == 10.0
    assert params['fallback_to_fast'].default is True


@pytest.mark.parametrize('smiles', ['CCO', 'CC(=O)O', 'C[C@H](F)Cl'],
                         ids=['alcohol', 'acid', 'stereocentre'])
def test_default_calls_match_completed_full_results_or_report_the_deadline(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol.SetProp('independent_control', 'preserve')
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    expected = smiles_to_iupac_result(smiles, timeout_seconds=None)
    assert mol_to_iupac_result(mol, timeout_seconds=None) == expected
    for result in (smiles_to_iupac_result(smiles), mol_to_iupac_result(mol)):
        deadline = any(issue.stage == 'runtime' and issue.rule in {'timeout', 'full-timeout'}
                       for issue in result.unresolved)
        if deadline:
            assert result.generation_status in {'available', 'timeout'}
            assert bool(result.name) == (result.generation_status == 'available')
            assert result.preferred_status == ('unresolved' if result.name else 'not-assessed')
        else:
            assert result == expected
    # Alias parity is a correctness contract, so its comparison is untimed.
    assert smiles_to_iupac(smiles, timeout_seconds=None) == expected.name
    assert mol_to_iupac(mol, timeout_seconds=None) == expected.name
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_relative_stereogroup_metadata_survives_worker_serialization():
    for smiles in ('C[C@H](F)Cl |&1:1|', 'C[C@H](F)Cl |r|'):
        mol = Chem.MolFromSmiles(smiles)
        result = mol_to_iupac_result(mol, timeout_seconds=15, fallback_to_fast=False)
        assert result == mol_to_iupac_result(mol, timeout_seconds=None)
        assert not result.name


@pytest.mark.parametrize('limit', [0, -1, float('inf'), float('nan'), True, '5'])
def test_invalid_timeouts_are_rejected_for_single_calls_and_pool(limit):
    with pytest.raises(ValueError):
        smiles_to_iupac_result('CCO', timeout_seconds=limit)
    with pytest.raises(ValueError):
        IUPACBatchPool(timeout_seconds=limit)


def test_fallback_setting_must_be_boolean():
    with pytest.raises(ValueError):
        smiles_to_iupac_result('CCO', fallback_to_fast='yes')


@pytest.mark.parametrize('smiles', ['[13CH3]CO', '[Na+].CCO[O-]',
                                  'C[S-]'],
                         ids=['isotope', 'peroxolate-charge', 'sulfur-charge'])
def test_fast_only_rejects_unsupported_input_without_full_handoff(monkeypatch, smiles):
    # The peroxolate is the independently authored ionic-fragment core
    # control, reused to protect the one-way timeout fallback contract.
    # The other plain graphs are independent topology/charge/stereo controls.
    def fail(*args, **kwargs):
        raise AssertionError('fast-only fallback re-entered full mode')
    monkeypatch.setattr(fast_selection, '_full_after_failure', fail)
    result = fast_selection._fast_dispatch(smiles, structured=True, allow_full_fallback=False)
    assert result.generation_status == 'unavailable' and not result.name
    mol_result = fast_selection._fast_mol_only_result(Chem.MolFromSmiles(smiles))
    assert mol_result.generation_status == 'unavailable' and not mol_result.name


def test_fast_fallback_declines_query_and_substance_group_molecules():
    query = Chem.MolFromSmarts('[#6]-[#8]')
    annotated = Chem.MolFromSmiles('CCO')
    group = Chem.CreateMolSubstanceGroup(annotated, 'SUP')
    group.AddAtomWithIdx(0)
    for mol in (query, annotated):
        result = fast_selection._fast_mol_only_result(mol)
        assert result.generation_status == 'unavailable' and not result.name


def test_fast_first_handoff_disables_a_return_trip_to_fast(monkeypatch):
    calls = []
    def full(smiles, *, fallback_to_fast):
        calls.append((smiles, fallback_to_fast))
        return nispo.NamingResult('', 'timeout', 'not-assessed')
    monkeypatch.setattr(api, 'smiles_to_iupac_result', full)
    result = fast_selection._fast_dispatch('[13CH3]CO', structured=True)
    assert result.generation_status == 'timeout'
    assert calls == [('[13CH3]CO', False)]


def test_default_single_calls_work_without_a_main_guard(tmp_path):
    script = tmp_path / 'unguarded.py'
    script.write_text("from nispo import smiles_to_iupac_result\n"
                      "result = smiles_to_iupac_result('CCO')\n"
                      "assert result.generation_status in {'available', 'timeout'}\n"
                      "assert bool(result.name) == (result.generation_status == 'available')\n"
                      "assert result.name or result.unresolved\n")
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(nispo.__file__).resolve().parent.parent)
    subprocess.run([sys.executable, str(script)], env=env, check=True, timeout=30)


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='fork is unavailable')
def test_fork_after_single_call_does_not_share_a_worker(tmp_path):
    script = tmp_path / 'fork_check.py'
    script.write_text("import os\nfrom nispo import smiles_to_iupac_result\n"
                      "def named():\n"
                      "    return bool(smiles_to_iupac_result('CCO', timeout_seconds=15, fallback_to_fast=False).name)\n"
                      "assert named()\n"
                      "pid = os.fork()\n"
                      "if pid == 0:\n"
                      "    os._exit(0 if named() else 2)\n"
                      "_, status = os.waitpid(pid, 0)\n"
                      "assert os.waitstatus_to_exitcode(status) == 0\n"
                      "assert named()\n")
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(nispo.__file__).resolve().parent.parent)
    subprocess.run([sys.executable, str(script)], env=env, check=True, timeout=120)


def test_concurrent_single_callers_have_correct_results():
    inputs = ('CCO', 'CC(=O)O', 'C1CC1', 'CCO')
    expected = [smiles_to_iupac_result(s, timeout_seconds=None) for s in inputs]
    call = partial(smiles_to_iupac_result, timeout_seconds=15, fallback_to_fast=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(call, inputs)) == expected


@pytest.mark.skipif(os.name != 'posix', reason='parent PID monitoring uses POSIX reparenting')
def test_orphaned_worker_does_not_keep_running(tmp_path):
    script = tmp_path / 'abrupt_exit.py'
    script.write_text(
        "import os, subprocess, sys\n"
        "code = ('import os,sys,time; from nispo._timed_parallel import _start_parent_monitor; '"
        "        '_start_parent_monitor(int(sys.argv[1])); print(\"ready\", flush=True); time.sleep(8)')\n"
        "p = subprocess.Popen([sys.executable, '-u', '-c', code, str(os.getpid())], "
        "stdout=subprocess.PIPE, text=True)\n"
        "print(p.stdout.readline().strip(), flush=True)\n"
        "os._exit(0)\n")
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(nispo.__file__).resolve().parent.parent)
    # The worker inherits stderr; capture_output cannot reach EOF while it is
    # alive. No platform-specific process-list command or signal is needed.
    completed = subprocess.run([sys.executable, str(script)], env=env, check=True,
                               capture_output=True, text=True, timeout=5)
    assert completed.stdout.strip() == 'ready'
