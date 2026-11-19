"""Controllable spawned worker for the public deadline contract tests."""
import os
from pathlib import Path
import struct
import time


def nitrogen_timeout_worker(connection, expected_package, parent_pid):
    """Delay only full naming, then exercise the actual timeout fallback."""
    from nispo import api
    from nispo._timed_parallel import _deadline_worker
    original = api._smiles_to_iupac_result_direct
    def delayed(value):
        time.sleep(4)
        return original(value)
    api._smiles_to_iupac_result_direct = delayed
    _deadline_worker(connection, expected_package, parent_pid)


def controlled_worker(connection, expected_package, parent_pid):
    from nispo.naming_result import NamingResult

    if os.environ.get('NISPO_TEST_DEADLINE_FAIL_START'):
        os._exit(17)
    time.sleep(0.3)  # Startup must not consume a molecule's budget.
    connection.send(('ready',))
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            index, value, structured, kind, stage = request
            connection.send(('started', index, time.monotonic()))
            if stage == 'fast':
                if value == 'no-fast':
                    result = NamingResult('', 'unavailable', 'not-assessed', (), 'fast:no-full-handoff')
                elif value == 'provisional-slow':
                    result = NamingResult('provisional', 'available', 'unresolved', (), 'fast:structural-von-baeyer')
                    connection.send(('provisional', index, result, time.monotonic()))
                    time.sleep(1.0)
                    result = NamingResult('late-fast', 'available', 'unresolved', (), 'fast:test')
                elif value == 'both-slow':
                    time.sleep(1.0)
                    result = NamingResult('late-fast', 'available', 'unresolved', (), 'fast:test')
                else:
                    result = NamingResult('fast:' + value, 'available', 'unresolved', (), 'fast:test')
                connection.send(('result', index, result, time.monotonic()))
                continue
            if value in {'slow', 'no-fast', 'both-slow', 'provisional-slow'}:
                Path(os.environ['NISPO_TEST_DEADLINE_PID']).write_text(str(os.getpid()))
                time.sleep(1.0)
            if value == 'partial':
                os.write(connection.fileno(), struct.pack('!i', 1024 * 1024))
                time.sleep(10.0)
            if value == 'crash':
                os._exit(17)
            name = f'{os.getpid()}:{value}' if value.startswith('pid:') else 'named:' + value
            result = NamingResult(name, 'available', 'not-assessed') if structured else name
            connection.send(('result', index, result, time.monotonic()))
    except EOFError:
        pass
    finally:
        connection.close()
