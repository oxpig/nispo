"""Process-isolated, per-input deadlines and one-way fast fallback."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import pickle
from queue import Empty, Queue
import struct
import subprocess
import sys
from threading import Thread
import time

from ._timeout_policy import DEFAULT_MAX_REQUESTS_PER_WORKER, normalize_worker_request_limit


def _start_parent_monitor(parent_pid):
    """Stop an orphaned worker if its owning process exits unexpectedly."""
    if os.name != 'posix':
        return
    if os.getppid() != parent_pid:
        os._exit(70)

    def watch():
        while os.getppid() == parent_pid:
            time.sleep(0.5)
        os._exit(70)

    Thread(target=watch, daemon=True).start()


def _deadline_worker(connection, expected_package, parent_pid):
    """Run naming requests in a worker that the parent can terminate."""
    try:
        _start_parent_monitor(parent_pid)
        from .parallel import _initialize_worker
        _initialize_worker(expected_package)
        from . import api

        connection.send(('ready',))
        while True:
            request = connection.recv()
            if request is None:
                return
            index, value, structured, kind, stage = request
            connection.send(('started', index, time.monotonic()))
            if kind == 'mol':
                from rdkit import Chem
                value = Chem.Mol(value)
            if stage == 'fast':
                from rdkit import Chem
                from .nitrogen_anion_components import requires_nitrogen_anion_proof
                from .naming_result import NamingResult, SelectionLimitation
                graph = value if kind == 'mol' else Chem.MolFromSmiles(value)
                if graph is not None and requires_nitrogen_anion_proof(graph):
                    # The preserved fast writer has no aminediide proof. A
                    # full-mode deadline must not reintroduce its known
                    # monoanion/H-budget error, even as a provisional result.
                    result = NamingResult('', 'unavailable', 'unresolved',
                        (SelectionLimitation('nitrogen-anion-component', 'P-72.2.2.2.3',
                            'The fast timeout fallback cannot establish this nitrogen charge and H population.'),),
                        'unsupported:amine-derived-anion')
                    connection.send(('result', index, result, time.monotonic()))
                    continue
                from .fast_selection import _fast_dispatch, _fast_mol_only_result
                def provisional(result):
                    connection.send(('provisional', index, result, time.monotonic()))
                result = (_fast_mol_only_result(value, _provisional_callback=provisional) if kind == 'mol'
                          else _fast_dispatch(value, structured=True, allow_full_fallback=False,
                                              _provisional_callback=provisional))
            else:
                if kind == 'mol':
                    function = api._mol_to_iupac_result_direct if structured else api._mol_to_iupac_direct
                else:
                    function = api._smiles_to_iupac_result_direct if structured else api._smiles_to_iupac_direct
                result = api._run_direct(function, value)
            connection.send(('result', index, result, time.monotonic()))
    except EOFError:
        pass
    finally:
        connection.close()


class _StreamConnection:
    """Private, framed pipes for a subprocess that does not import __main__."""

    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer

    def send(self, value):
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        self.writer.write(struct.pack('!Q', len(data)))
        self.writer.write(data)
        self.writer.flush()

    def _read_exactly(self, length):
        pieces = []
        while length:
            piece = self.reader.read(length)
            if not piece:
                raise EOFError
            pieces.append(piece)
            length -= len(piece)
        return b''.join(pieces)

    def recv(self):
        size = struct.unpack('!Q', self._read_exactly(8))[0]
        return pickle.loads(self._read_exactly(size))

    def close(self):
        self.reader.close()
        self.writer.close()


class _Subprocess:
    def __init__(self, process):
        self.process = process

    def is_alive(self):
        return self.process.poll() is None

    def terminate(self):
        self.process.terminate()

    def kill(self):
        self.process.kill()

    def join(self, timeout=None):
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def close(self):
        pass


def _stdio_main(expected_package, parent_pid):
    reader, writer = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    _deadline_worker(_StreamConnection(reader, writer), expected_package, int(parent_pid))


@dataclass(eq=False)
class _Worker:
    process: object
    connection: object
    phase: str
    deadline: float
    index: int | None = None
    stage: str | None = None
    completed: int = 0
    reader: object | None = None
    provisional: object | None = None


class _DeadlinePool:
    """Persistent spawned workers; a timed-out request kills its worker."""

    _STARTUP_TIMEOUT_SECONDS = 60.0
    def __init__(self, workers, timeout_seconds, *, backend='spawn',
                 max_requests_per_worker=DEFAULT_MAX_REQUESTS_PER_WORKER):
        self.workers = workers
        self.timeout_seconds = timeout_seconds
        self.max_requests_per_worker = normalize_worker_request_limit(max_requests_per_worker)
        if backend not in {'spawn', 'subprocess'}:
            raise ValueError('Unsupported worker backend')
        self._backend = backend
        self._context = multiprocessing.get_context('spawn') if backend == 'spawn' else None
        self._expected_package = str(Path(__file__).resolve().parent)
        self._workers = []
        self._events = Queue()
        self._closed = False

    def _read_events(self, worker):
        try:
            while True:
                self._events.put((worker, worker.connection.recv()))
        except Exception:
            self._events.put((worker, None))

    def _start(self):
        if self._backend == 'subprocess':
            package_root = str(Path(self._expected_package).parent)
            command = [sys.executable, '-u', '-c',
                       'import sys; sys.path.insert(0, sys.argv[1]); '
                       'from nispo._timed_parallel import _stdio_main; _stdio_main(sys.argv[2], sys.argv[3])',
                       package_root, self._expected_package, str(os.getpid())]
            popen = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            parent = _StreamConnection(popen.stdout, popen.stdin)
            process = _Subprocess(popen)
        else:
            parent, child = self._context.Pipe()
            process = self._context.Process(target=_deadline_worker,
                                            args=(child, self._expected_package, os.getpid()), daemon=True)
            try:
                process.start()
            except BaseException:
                parent.close()
                child.close()
                raise
            child.close()
        worker = _Worker(process, parent, 'initializing', time.monotonic() + self._STARTUP_TIMEOUT_SECONDS)
        self._workers.append(worker)
        worker.reader = Thread(target=self._read_events, args=(worker,), daemon=True)
        worker.reader.start()
        return worker

    def _stop(self, worker):
        if worker in self._workers:
            self._workers.remove(worker)
        if worker.process.is_alive():
            worker.process.terminate()
            worker.process.join(timeout=0.5)
        if worker.process.is_alive():
            worker.process.kill()
            worker.process.join(timeout=1.0)
        if worker.process.is_alive():
            raise RuntimeError('The naming worker could not be terminated')
        worker.process.join()
        worker.connection.close()
        if worker.reader is not None:
            worker.reader.join(timeout=1.0)
        worker.process.close()

    def close(self):
        self._closed = True
        for worker in tuple(self._workers):
            self._stop(worker)

    @staticmethod
    def _failure(structured, status, reason):
        if not structured:
            return ''
        from .naming_result import NamingResult, SelectionLimitation
        return NamingResult('', status, 'not-assessed',
                            (SelectionLimitation('runtime', status, reason),), 'runtime:' + status)

    def run(self, values, *, structured, kind='smiles', fallback_to_fast=True, timeout_seconds=None):
        if self._closed:
            raise RuntimeError('The naming pool has been closed')
        if kind not in {'smiles', 'mol'}:
            raise ValueError('Unsupported naming input')
        limit = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        unique = list(dict.fromkeys(values))
        if not unique:
            return []
        pending = deque((index, 'full') for index in range(len(unique)))
        results = [None] * len(unique)
        remaining = len(unique)
        desired = min(self.workers, len(unique))

        def finish(index, result):
            nonlocal remaining
            if results[index] is not None:
                raise RuntimeError('A naming worker returned the same request twice')
            results[index] = result
            remaining -= 1

        def full_timeout(index):
            if fallback_to_fast:
                pending.appendleft((index, 'fast'))
            else:
                finish(index, self._failure(structured, 'timeout',
                       f'Full naming exceeded the {limit:g}-second limit.'))

        def fast_result(index, result):
            from .naming_result import NamingResult, SelectionLimitation
            if not isinstance(result, NamingResult):
                raise RuntimeError('The fast fallback returned an invalid result')
            limitation = SelectionLimitation(
                'runtime', 'full-timeout',
                f'Full naming exceeded the {limit:g}-second limit; fast fallback is not a verified PIN or graph.',
            )
            if result.name:
                value = NamingResult(result.name, 'available', 'unresolved',
                                     (limitation, *result.unresolved), 'full-timeout:' + result.route)
            else:
                value = NamingResult('', 'timeout', 'not-assessed',
                                     (limitation, *result.unresolved), 'full-timeout:fast-unavailable')
            finish(index, value if structured else value.name)

        def fast_failed(index, reason):
            finish(index, self._failure(structured, 'timeout',
                   f'Full naming exceeded the {limit:g}-second limit; fast fallback {reason}.'))

        def failed_worker(worker):
            if worker.phase == 'initializing':
                raise RuntimeError('A naming worker exited during initialization')
            if worker.index is not None:
                if worker.stage == 'fast':
                    if worker.provisional is not None:
                        fast_result(worker.index, worker.provisional)
                    else:
                        fast_failed(worker.index, 'worker exited unexpectedly')
                else:
                    finish(worker.index, self._failure(structured, 'error', 'The naming worker exited unexpectedly.'))
            self._stop(worker)

        while remaining:
            while pending and len(self._workers) < desired:
                self._start()
            for worker in tuple(self._workers):
                if not pending or worker.phase != 'ready':
                    continue
                index, stage = pending.popleft()
                worker.index = index
                worker.stage = stage
                worker.provisional = None
                worker.phase = 'assigned'
                worker.deadline = time.monotonic() + self._STARTUP_TIMEOUT_SECONDS
                try:
                    worker.connection.send((index, unique[index], structured if stage == 'full' else True, kind, stage))
                except (BrokenPipeError, EOFError, OSError):
                    failed_worker(worker)
            if not remaining:
                break
            if not self._workers:
                continue
            deadlines = [w.deadline for w in self._workers if w.phase != 'ready']
            timeout = max(0.0, min(deadlines) - time.monotonic()) if deadlines else None
            try:
                events = [self._events.get(timeout=timeout)]
            except Empty:
                events = []
            while True:
                try:
                    events.append(self._events.get_nowait())
                except Empty:
                    break
            for worker, message in events:
                if worker not in self._workers:
                    continue
                if message is None:
                    failed_worker(worker)
                    continue
                event = message[0]
                if event == 'ready' and worker.phase == 'initializing':
                    worker.phase = 'ready'
                elif event == 'started' and worker.phase == 'assigned' and message[1] == worker.index:
                    worker.phase = 'running'
                    worker.deadline = message[2] + limit
                elif (event == 'provisional' and worker.phase == 'running' and worker.stage == 'fast'
                      and message[1] == worker.index):
                    from .naming_result import NamingResult
                    result = message[2]
                    if not isinstance(result, NamingResult) or not result.name or result.preferred_status != 'unresolved':
                        raise RuntimeError('The fast worker returned an invalid provisional result')
                    if message[3] <= worker.deadline:
                        worker.provisional = result
                elif event == 'result' and worker.phase == 'running' and message[1] == worker.index:
                    if message[2] is None:
                        raise RuntimeError('A naming worker returned no result')
                    late = message[3] > worker.deadline
                    if late:
                        if worker.stage == 'fast':
                            if worker.provisional is not None:
                                fast_result(worker.index, worker.provisional)
                            else:
                                fast_failed(worker.index, f'exceeded its {limit:g}-second limit')
                        else:
                            full_timeout(worker.index)
                        self._stop(worker)
                    else:
                        if worker.stage == 'fast':
                            fast_result(worker.index, message[2])
                        else:
                            finish(worker.index, message[2])
                        worker.index = None
                        worker.stage = None
                        worker.provisional = None
                        worker.phase = 'ready'
                        worker.completed += 1
                        if (self.max_requests_per_worker is not None
                                and worker.completed >= self.max_requests_per_worker):
                            self._stop(worker)
                else:
                    raise RuntimeError('Unexpected naming worker response')
            now = time.monotonic()
            for worker in tuple(self._workers):
                if worker.phase == 'ready' or now < worker.deadline:
                    continue
                if worker.phase == 'running':
                    if worker.stage == 'fast':
                        if worker.provisional is not None:
                            fast_result(worker.index, worker.provisional)
                        else:
                            fast_failed(worker.index, f'exceeded its {limit:g}-second limit')
                    else:
                        full_timeout(worker.index)
                    self._stop(worker)
                elif worker.phase == 'initializing':
                    raise RuntimeError('A naming worker did not initialize within 60 seconds')
                else:
                    raise RuntimeError('A naming worker did not start its request within 60 seconds')
        by_value = dict(zip(unique, results, strict=True))
        return [by_value[value] for value in values]
