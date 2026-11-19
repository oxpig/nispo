"""Lazily own one reusable subprocess for ordinary naming APIs."""
from __future__ import annotations

import atexit
import os
from threading import RLock

_lock = RLock()
_pool = None


def run(values, *, timeout_seconds, structured, kind, fallback_to_fast):
    global _pool
    from ._timed_parallel import _DeadlinePool

    with _lock:
        if _pool is None:
            _pool = _DeadlinePool(1, timeout_seconds, backend='subprocess')
        try:
            return _pool.run(values, structured=structured, kind=kind,
                             fallback_to_fast=fallback_to_fast, timeout_seconds=timeout_seconds)
        except BaseException:
            _pool.close()
            _pool = None
            raise


def _cleanup():
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def _before_fork():
    global _pool
    _lock.acquire()
    try:
        if _pool is not None:
            _pool.close()
            _pool = None
    except BaseException:
        _lock.release()
        raise


def _after_fork_parent():
    _lock.release()


def _after_fork_child():
    global _lock, _pool
    _pool = None
    _lock = RLock()


atexit.register(_cleanup)
if hasattr(os, 'register_at_fork'):
    os.register_at_fork(before=_before_fork, after_in_parent=_after_fork_parent, after_in_child=_after_fork_child)
