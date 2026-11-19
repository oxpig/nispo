"""Lightweight public SMILES entry points for full and fast naming."""
from __future__ import annotations

from collections.abc import Iterable
from contextvars import ContextVar

from ._timeout_policy import DEFAULT_TIMEOUT_SECONDS, normalize_timeout, validate_fallback
from .naming_result import NamingResult


_UNSET = object()
_FAST_HANDOFF_TIMEOUT = ContextVar('nispo_fast_handoff_timeout', default=_UNSET)


def _check_mode(mode):
    if not isinstance(mode, str) or mode not in ('full', 'fast'):
        raise ValueError("mode must be 'full' or 'fast'")


def _fast_handoff_keywords():
    """Pass a caller-selected deadline to a one-way full handoff."""
    timeout = _FAST_HANDOFF_TIMEOUT.get()
    if timeout is _UNSET:
        return {}
    return {'timeout_seconds': timeout}


def _fast_request(values, *, structured, batch, timeout_seconds, fallback_to_fast):
    timeout = normalize_timeout(timeout_seconds)
    validate_fallback(fallback_to_fast)
    from .fast_selection import smiles_to_iupac_fast, smiles_to_iupac_fast_result

    call = smiles_to_iupac_fast_result if structured else smiles_to_iupac_fast
    token = _FAST_HANDOFF_TIMEOUT.set(timeout)
    try:
        if not batch:
            return call(values)
        cache = {}
        results = []
        for value in values:
            if value not in cache:
                cache[value] = call(value)
            results.append(cache[value])
        return results
    finally:
        _FAST_HANDOFF_TIMEOUT.reset(token)


def smiles_to_iupac(smiles: str, *, mode='full', timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                    fallback_to_fast=True) -> str:
    """Name a SMILES string using full mode (default) or fast mode."""
    _check_mode(mode)
    if mode == 'full':
        from .api import _smiles_to_iupac_full
        return _smiles_to_iupac_full(smiles, timeout_seconds=timeout_seconds,
                                    fallback_to_fast=fallback_to_fast)
    return _fast_request(smiles, structured=False, batch=False,
                         timeout_seconds=timeout_seconds, fallback_to_fast=fallback_to_fast)


def smiles_to_iupac_result(smiles: str, *, mode='full', timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                           fallback_to_fast=True) -> NamingResult:
    """Name a SMILES string in either mode and expose delivery status."""
    _check_mode(mode)
    if mode == 'full':
        from .api import _smiles_to_iupac_result_full
        return _smiles_to_iupac_result_full(smiles, timeout_seconds=timeout_seconds,
                                           fallback_to_fast=fallback_to_fast)
    return _fast_request(smiles, structured=True, batch=False,
                         timeout_seconds=timeout_seconds, fallback_to_fast=fallback_to_fast)


def smiles_to_iupac_batch(smiles_list: Iterable[str], *, mode='full',
                          timeout_seconds=DEFAULT_TIMEOUT_SECONDS, fallback_to_fast=True) -> list[str]:
    """Name a batch in either mode, preserving input order and duplicates."""
    _check_mode(mode)
    if mode == 'full':
        from .api import _smiles_to_iupac_batch_full
        return _smiles_to_iupac_batch_full(smiles_list, timeout_seconds=timeout_seconds,
                                          fallback_to_fast=fallback_to_fast)
    return _fast_request(smiles_list, structured=False, batch=True,
                         timeout_seconds=timeout_seconds, fallback_to_fast=fallback_to_fast)


def smiles_to_iupac_batch_results(smiles_list: Iterable[str], *, mode='full',
                                  timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                                  fallback_to_fast=True) -> list[NamingResult]:
    """Structured batch naming in full or fast mode."""
    _check_mode(mode)
    if mode == 'full':
        from .api import _smiles_to_iupac_batch_results_full
        return _smiles_to_iupac_batch_results_full(smiles_list, timeout_seconds=timeout_seconds,
                                                  fallback_to_fast=fallback_to_fast)
    return _fast_request(smiles_list, structured=True, batch=True,
                         timeout_seconds=timeout_seconds, fallback_to_fast=fallback_to_fast)
