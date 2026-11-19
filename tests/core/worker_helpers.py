"""Small deterministic protocol peers for public pool lifecycle tests.

The request strings are synthetic transport tokens, not molecule fixtures or
generated nomenclature. Hung peers block without using a CPU; the real parent
supervisor must terminate them. Actual chemistry is exercised separately.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
from threading import Event
import time


def _record(event, value=None, stage=None):
    path = os.environ.get('NISPO_CORE_WORKER_LOG')
    if path:
        with open(path, 'a') as stream:
            stream.write(json.dumps({'event': event, 'value': value,
                                     'stage': stage, 'pid': os.getpid()})+'\n')


def _block():
    # Finite backstop if a failing test loses its owner; successful tests kill
    # this peer at the short per-request deadline.
    Event().wait(60)
    raise RuntimeError('A synthetic hung request was not terminated')


def controlled_worker(connection, expected_package, parent_pid):
    from nispo.naming_result import NamingResult
    from nispo._timed_parallel import _start_parent_monitor

    _start_parent_monitor(parent_pid)
    if os.environ.get('NISPO_CORE_FAIL_START'):
        os._exit(17)
    startup_delay = float(os.environ.get('NISPO_CORE_STARTUP_DELAY', '0'))
    if startup_delay:
        time.sleep(startup_delay)
    connection.send(('ready',))
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            index, value, structured, kind, stage = request
            connection.send(('started', index, time.monotonic()))
            _record('started', value, stage)
            if stage == 'full':
                if value == 'crash':
                    os._exit(17)
                if value == 'partial':
                    # A valid multiprocessing frame length with no body must
                    # not trap the supervisor in its reader thread.
                    os.write(connection.fileno(), struct.pack('!i', 1024*1024))
                    _block()
                if value in {'hang', 'fallback', 'decline', 'both-hang', 'provisional',
                             'fast-crash', 'provisional-crash'}:
                    _block()
                if value == 'hold':
                    signal = Path(os.environ['NISPO_CORE_RELEASE'])
                    while not signal.exists():
                        time.sleep(0.01)
                elif value == 'release':
                    Path(os.environ['NISPO_CORE_RELEASE']).touch()
                name = f'{os.getpid()}::{value}'
                result = NamingResult(name, 'available', 'not-assessed') if structured else name
            else:
                if value == 'fast-crash':
                    os._exit(17)
                if value == 'both-hang':
                    _block()
                if value in {'provisional', 'provisional-crash'}:
                    result = NamingResult(f'{os.getpid()}::provisional', 'available',
                                          'unresolved', (), 'fast:synthetic')
                    connection.send(('provisional', index, result, time.monotonic()))
                    if value == 'provisional-crash':
                        os._exit(17)
                    _block()
                result = NamingResult('' if value=='decline' else f'{os.getpid()}::{value}',
                                      'unavailable' if value=='decline' else 'available',
                                      'not-assessed' if value=='decline' else 'unresolved',
                                      (), 'fast:synthetic')
            connection.send(('result', index, result, time.monotonic()))
            _record('result', value, stage)
    except EOFError:
        pass
    finally:
        connection.close()


def delayed_full_worker(connection, expected_package, parent_pid):
    """Hold full naming, then use the actual production fast fallback."""
    from nispo import api
    from nispo._timed_parallel import _deadline_worker

    def held_full(value):
        _block()

    api._smiles_to_iupac_result_direct = held_full
    _deadline_worker(connection, expected_package, parent_pid)
