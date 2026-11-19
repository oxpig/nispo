"""Isolate optional fast-mode ring work from ordinary and concurrent naming."""

from contextvars import ContextVar


_FAST_CARBON_RING_NUMBERING = ContextVar('nispo_fast_carbon_ring_numbering', default=False)


def fast_carbon_ring_numbering_enabled():
    return _FAST_CARBON_RING_NUMBERING.get()


def with_fast_carbon_ring_numbering(function, *args, **kwargs):
    token = _FAST_CARBON_RING_NUMBERING.set(True)
    try:
        return function(*args, **kwargs)
    finally:
        _FAST_CARBON_RING_NUMBERING.reset(token)


def without_fast_carbon_ring_numbering(function, *args, **kwargs):
    token = _FAST_CARBON_RING_NUMBERING.set(False)
    try:
        return function(*args, **kwargs)
    finally:
        _FAST_CARBON_RING_NUMBERING.reset(token)
