from __future__ import annotations

import os
from collections.abc import Iterable
from contextvars import ContextVar
from functools import lru_cache

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from .parent_rules import parent_rules
from .rule_engine import NamingEngine
from .telemetry import CandidateTelemetry, telemetry_to_dict
from .naming_result import NamingResult
from ._timeout_policy import DEFAULT_TIMEOUT_SECONDS
from .input_stereochemistry import (
    unsupported_relative_stereochemistry,
    unsupported_relative_stereochemistry_result,
)

# Retained for compatibility with older probes/tests that clear the symbol.
# Global name caching stays disabled. The batch API only memoizes duplicate
# inputs inside one call so output order/count stay unchanged while repeated
# work in a submitted batch is avoided.
_BATCH_NAME_CACHE_MAX_SIZE = 0
_BATCH_NAME_CACHE: dict[str, str] = {}
_IN_DIRECT_NAMING = ContextVar('nispo_in_direct_naming', default=False)


def mol_to_iupac(mol) -> str:
    """Name one RDKit molecule within the supported IUPAC rule scopes.

    The input molecule is copied so naming does not attach NISPO's internal
    caches to the caller's object.
    """
    if Chem is None or mol is None:
        return ""
    try:
        mol_copy = Chem.Mol(mol)
        if mol_copy.GetNumAtoms() == 0:
            return ""
        return _name_molecule(mol_copy)
    except Exception:
        return ""


def smiles_to_iupac(smiles: str) -> str:
    """Name one SMILES string within the supported IUPAC rule scopes."""
    if Chem is None:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return ""
    if mol is None or mol.GetNumAtoms() == 0:
        return ""
    try:
        return _name_molecule(mol)
    except Exception:
        return ""


def mol_to_iupac_result(mol) -> NamingResult:
    """Return a name and its generation/preferred-selection disposition.

    An available name with unresolved preference is usable general
    nomenclature; it must not be treated as a verified PIN. ``resolved``
    describes the implemented rule scope, not independent PIN certification.
    The caller's molecule is copied as in :func:`mol_to_iupac`.
    """
    if Chem is None:
        return NamingResult('', 'unavailable', 'not-assessed')
    try:
        if mol is None:
            return NamingResult('', 'invalid-input', 'not-assessed')
        copied = Chem.Mol(mol)
        if not copied.GetNumAtoms():
            return NamingResult('', 'invalid-input', 'not-assessed')
    except Exception:
        return NamingResult('', 'invalid-input', 'not-assessed')
    if unsupported_relative_stereochemistry(copied, Chem):
        return unsupported_relative_stereochemistry_result()
    try:
        return _engine().name_result(copied)
    except Exception:
        return NamingResult('', 'error', 'not-assessed')


def smiles_to_iupac_result(smiles: str) -> NamingResult:
    """Name one SMILES with explicit unresolved preferred-name obligations."""
    if Chem is None:
        return NamingResult('', 'unavailable', 'not-assessed')
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None or not mol.GetNumAtoms():
        return NamingResult('', 'invalid-input', 'not-assessed')
    if unsupported_relative_stereochemistry(mol, Chem):
        return unsupported_relative_stereochemistry_result()
    try:
        # Parsing already gives this API its own molecule. The molecule API
        # still copies externally owned graphs before naming them.
        return _engine().name_result(mol)
    except Exception:
        return NamingResult('', 'error', 'not-assessed')


def smiles_to_iupac_batch_results(smiles_list: Iterable[str]) -> list[NamingResult]:
    """Structured counterpart of the string batch API, preserving order."""
    cache: dict[str, NamingResult] = {}
    results = []
    for smiles in smiles_list:
        if smiles not in cache:
            cache[smiles] = smiles_to_iupac_result(smiles)
        results.append(cache[smiles])
    return results


def smiles_to_iupac_batch(smiles_list: Iterable[str]) -> list[str]:
    """Convert a batch of SMILES strings to IUPAC-style names.

    One output is returned for each input. Molecules outside the current rule
    coverage return an empty string so the benchmark can continue.
    """
    smiles_values = list(smiles_list)
    if Chem is None:
        return ["" for _ in smiles_values]
    engine = _engine()
    names: list[str] = []
    batch_cache: dict[str, str] = {}
    mol_from_smiles = Chem.MolFromSmiles
    name_molecule = engine.name
    cache_get = batch_cache.get
    cache_set = batch_cache.__setitem__
    append_name = names.append
    for smiles in smiles_values:
        cached = cache_get(smiles)
        if cached is not None:
            append_name(cached)
            continue
        try:
            mol = mol_from_smiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                name = ""
                cache_set(smiles, name)
                append_name(name)
                continue
            if unsupported_relative_stereochemistry(mol, Chem):
                name = ""
                cache_set(smiles, name)
                append_name(name)
                continue
            name = name_molecule(mol)
        except Exception:
            name = ""
        cache_set(smiles, name)
        append_name(name)
    return names


# Keep the original writers as the worker's internal implementation. Nested
# nomenclature calls remain inside the same supervised job rather than spawning
# another worker or resetting its deadline.
_mol_to_iupac_direct = mol_to_iupac
_smiles_to_iupac_direct = smiles_to_iupac
_mol_to_iupac_result_direct = mol_to_iupac_result
_smiles_to_iupac_result_direct = smiles_to_iupac_result
_smiles_to_iupac_batch_direct = smiles_to_iupac_batch
_smiles_to_iupac_batch_results_direct = smiles_to_iupac_batch_results
_MOL_SERIALIZATION_FAILED = object()


def _run_direct(function, *args):
    token = _IN_DIRECT_NAMING.set(True)
    try:
        return function(*args)
    finally:
        _IN_DIRECT_NAMING.reset(token)


def _settings(timeout_seconds, fallback_to_fast):
    from ._timeout_policy import normalize_timeout, validate_fallback
    return normalize_timeout(timeout_seconds), validate_fallback(fallback_to_fast)


def _molecule_payload(mol):
    if Chem is None or mol is None:
        return None
    try:
        copied = Chem.Mol(mol)
        if not copied.GetNumAtoms():
            return None
    except Exception:
        return None
    try:
        return copied.ToBinary(Chem.PropertyPickleOptions.AllProps)
    except Exception:
        return _MOL_SERIALIZATION_FAILED


def _supervised(values, *, timeout_seconds, fallback_to_fast, structured, kind):
    from ._single_timeout import run
    return run(values, timeout_seconds=timeout_seconds, fallback_to_fast=fallback_to_fast,
               structured=structured, kind=kind)


def mol_to_iupac(mol, *, timeout_seconds=DEFAULT_TIMEOUT_SECONDS, fallback_to_fast=True) -> str:
    """Name one molecule, with a ten-second full-mode deadline by default."""
    if _IN_DIRECT_NAMING.get():
        return _mol_to_iupac_direct(mol)
    timeout_seconds, fallback_to_fast = _settings(timeout_seconds, fallback_to_fast)
    if timeout_seconds is None:
        return _run_direct(_mol_to_iupac_direct, mol)
    payload = _molecule_payload(mol)
    if payload is _MOL_SERIALIZATION_FAILED:
        return ''
    if payload is None:
        return _run_direct(_mol_to_iupac_direct, mol)
    return _supervised([payload], timeout_seconds=timeout_seconds, fallback_to_fast=fallback_to_fast,
                       structured=False, kind='mol')[0]


def smiles_to_iupac(smiles: str, *, timeout_seconds=DEFAULT_TIMEOUT_SECONDS, fallback_to_fast=True) -> str:
    """Name one SMILES, with a ten-second full-mode deadline by default."""
    if _IN_DIRECT_NAMING.get():
        return _smiles_to_iupac_direct(smiles)
    timeout_seconds, fallback_to_fast = _settings(timeout_seconds, fallback_to_fast)
    try:
        hash(smiles)
    except TypeError:
        return _run_direct(_smiles_to_iupac_direct, smiles)
    if timeout_seconds is None or Chem is None:
        return _run_direct(_smiles_to_iupac_direct, smiles)
    return _supervised([smiles], timeout_seconds=timeout_seconds, fallback_to_fast=fallback_to_fast,
                       structured=False, kind='smiles')[0]


def mol_to_iupac_result(mol, *, timeout_seconds=DEFAULT_TIMEOUT_SECONDS, fallback_to_fast=True) -> NamingResult:
    """Name one molecule and expose timeout or fallback limitations."""
    if _IN_DIRECT_NAMING.get():
        return _mol_to_iupac_result_direct(mol)
    timeout_seconds, fallback_to_fast = _settings(timeout_seconds, fallback_to_fast)
    if timeout_seconds is None:
        return _run_direct(_mol_to_iupac_result_direct, mol)
    payload = _molecule_payload(mol)
    if payload is _MOL_SERIALIZATION_FAILED:
        from .naming_result import SelectionLimitation
        return NamingResult('', 'error', 'not-assessed',
                            (SelectionLimitation('runtime', 'worker-serialization',
                                                 'The molecule could not be sent to the naming worker.'),),
                            'runtime:serialization-error')
    if payload is None:
        return _run_direct(_mol_to_iupac_result_direct, mol)
    return _supervised([payload], timeout_seconds=timeout_seconds, fallback_to_fast=fallback_to_fast,
                       structured=True, kind='mol')[0]


def smiles_to_iupac_result(smiles: str, *, timeout_seconds=DEFAULT_TIMEOUT_SECONDS, fallback_to_fast=True) -> NamingResult:
    """Name one SMILES and expose timeout or fallback limitations."""
    if _IN_DIRECT_NAMING.get():
        return _smiles_to_iupac_result_direct(smiles)
    timeout_seconds, fallback_to_fast = _settings(timeout_seconds, fallback_to_fast)
    try:
        hash(smiles)
    except TypeError:
        return _run_direct(_smiles_to_iupac_result_direct, smiles)
    if timeout_seconds is None or Chem is None:
        return _run_direct(_smiles_to_iupac_result_direct, smiles)
    return _supervised([smiles], timeout_seconds=timeout_seconds, fallback_to_fast=fallback_to_fast,
                       structured=True, kind='smiles')[0]


def smiles_to_iupac_batch(smiles_list: Iterable[str], *, timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                          fallback_to_fast=True) -> list[str]:
    """Name a batch with a per-input deadline, preserving input order."""
    if _IN_DIRECT_NAMING.get():
        return _smiles_to_iupac_batch_direct(smiles_list)
    timeout_seconds, fallback_to_fast = _settings(timeout_seconds, fallback_to_fast)
    values = list(smiles_list)
    if not values:
        return []
    if timeout_seconds is None or Chem is None:
        return _run_direct(_smiles_to_iupac_batch_direct, values)
    return _supervised(values, timeout_seconds=timeout_seconds, fallback_to_fast=fallback_to_fast,
                       structured=False, kind='smiles')


def smiles_to_iupac_batch_results(smiles_list: Iterable[str], *, timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                                  fallback_to_fast=True) -> list[NamingResult]:
    """Name a batch and expose timeout or fallback limitations for each input."""
    if _IN_DIRECT_NAMING.get():
        return _smiles_to_iupac_batch_results_direct(smiles_list)
    timeout_seconds, fallback_to_fast = _settings(timeout_seconds, fallback_to_fast)
    values = list(smiles_list)
    if not values:
        return []
    if timeout_seconds is None or Chem is None:
        return _run_direct(_smiles_to_iupac_batch_results_direct, values)
    return _supervised(values, timeout_seconds=timeout_seconds, fallback_to_fast=fallback_to_fast,
                       structured=True, kind='smiles')


# Retain the full-only delivery implementation for the public mode
# dispatcher. The original direct writers captured above remain the
# supervised worker implementation.
_smiles_to_iupac_full = smiles_to_iupac
_smiles_to_iupac_result_full = smiles_to_iupac_result
_smiles_to_iupac_batch_full = smiles_to_iupac_batch
_smiles_to_iupac_batch_results_full = smiles_to_iupac_batch_results

from ._smiles_dispatch import (  # noqa: E402,F811
    smiles_to_iupac, smiles_to_iupac_result,
    smiles_to_iupac_batch, smiles_to_iupac_batch_results,
)


def _smiles_to_iupac_batch_with_telemetry(
    smiles_list: Iterable[str],
    *,
    branch_miss_shape_detail: bool = False,
) -> tuple[list[str], dict]:
    """Convert a batch and return serialized internal naming telemetry.

    This helper is intentionally private: benchmark tooling can use it for
    measurement while the public batch API remains unchanged.
    """

    smiles_values = list(smiles_list)
    telemetry = CandidateTelemetry(branch_miss_shape_detail=branch_miss_shape_detail)
    if not branch_miss_shape_detail and not _full_benchmark_telemetry_enabled():
        return smiles_to_iupac_batch(smiles_values), telemetry_to_dict(telemetry)
    if Chem is None:
        return ["" for _ in smiles_values], telemetry_to_dict(telemetry)
    engine = _engine()
    names: list[str] = []
    for smiles in smiles_values:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                names.append("")
                continue
            if unsupported_relative_stereochemistry(mol, Chem):
                names.append("")
                continue
            names.append(engine.name(mol, telemetry=telemetry))
        except Exception:
            names.append("")
    return names, telemetry_to_dict(telemetry)


def _full_benchmark_telemetry_enabled() -> bool:
    value = os.environ.get("NISPO_BENCH_FULL_TELEMETRY", "")
    return value.strip().lower() in {"1", "true", "yes", "on", "full"}


def _remember_batch_name(smiles: str, name: str) -> None:
    return None


def _cached_batch_name(smiles: str) -> str | None:
    return None


def _name_molecule(mol) -> str:
    if unsupported_relative_stereochemistry(mol, Chem):
        return ""
    return _engine().name(mol)


@lru_cache(maxsize=1)
def _engine() -> NamingEngine:
    return NamingEngine(Chem, parent_rules(Chem))
