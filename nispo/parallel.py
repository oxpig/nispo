"""Explicit process ownership for parallel, order-preserving batch naming."""
from __future__ import annotations

from ._timeout_policy import DEFAULT_MAX_REQUESTS_PER_WORKER, DEFAULT_TIMEOUT_SECONDS

def _initialize_worker(expected_package):
    from pathlib import Path

    if str(Path(__file__).resolve().parent) != expected_package:
        raise RuntimeError("The naming worker imported a different NISPO package")
    from .api import _engine

    _engine()


def _name_chunk(request):
    values, structured = request
    from .api import smiles_to_iupac_batch, smiles_to_iupac_batch_results

    return (smiles_to_iupac_batch_results(values, timeout_seconds=None) if structured
            else smiles_to_iupac_batch(values, timeout_seconds=None))


class IUPACBatchPool:
    """Name independent batches with a persistent pool of isolated processes.

    Use as a context manager to release workers explicitly. A pool retains
    imported rules and templates between batches; generated names are only
    memoized within each submitted batch. Results preserve input order and
    duplicate/unsupported-input behavior of the serial APIs.

    Workers use Python's spawn context. Executable Python scripts should put
    pool creation inside their ``if __name__ == '__main__':`` guard.

    ``timeout_seconds`` defaults to ten seconds per unique full naming call.
    A timed-out worker is terminated and the independent fast writer is tried
    once, with its own limit. Set ``fallback_to_fast=False`` to report the
    timeout directly, or ``timeout_seconds=None`` for unrestricted full mode.
    Startup and queue time are excluded. ``max_requests_per_worker`` controls
    recycling of timed workers after completed requests (default 100,000).
    Set it higher, or to ``None``, to retain warm caches longer at the cost of
    retaining worker memory. Timeouts still stop and replace workers. With
    ``timeout_seconds=None``, workers already persist until the pool closes;
    the request limit applies only to timed execution.
    """

    def __init__(self, workers=4, *, chunk_size=16, timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                 fallback_to_fast=True, max_requests_per_worker=DEFAULT_MAX_REQUESTS_PER_WORKER):
        for label, value in (("workers", workers), ("chunk_size", chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        from ._timeout_policy import normalize_timeout, normalize_worker_request_limit, validate_fallback
        self.workers = workers
        self.chunk_size = chunk_size
        self.timeout_seconds = normalize_timeout(timeout_seconds)
        self.fallback_to_fast = validate_fallback(fallback_to_fast)
        self.max_requests_per_worker = normalize_worker_request_limit(max_requests_per_worker)
        self._executor = None
        self._deadline_pool = None
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise RuntimeError("The naming pool has been closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        """Wait for submitted work and release all worker processes."""
        self._closed = True
        if self._deadline_pool is not None:
            self._deadline_pool.close()
            self._deadline_pool = None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def _run(self, smiles_list, *, structured):
        if self._closed:
            raise RuntimeError("The naming pool has been closed")
        values = list(smiles_list)
        if not values:
            return []
        if self.timeout_seconds is not None:
            if self._deadline_pool is None:
                from ._timed_parallel import _DeadlinePool
                self._deadline_pool = _DeadlinePool(
                    self.workers, self.timeout_seconds,
                    max_requests_per_worker=self.max_requests_per_worker)
            try:
                return self._deadline_pool.run(values, structured=structured, fallback_to_fast=self.fallback_to_fast)
            except BaseException:
                self._deadline_pool.close()
                self._deadline_pool = None
                raise
        if self.workers == 1:
            return _name_chunk((values, structured))
        unique = list(dict.fromkeys(values))
        if self._executor is None:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            from pathlib import Path

            self._executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(str(Path(__file__).resolve().parent),),
            )
        chunks = (unique[start:start + self.chunk_size]
                  for start in range(0, len(unique), self.chunk_size))
        records = []
        for result in self._executor.map(
                _name_chunk, ((chunk, structured) for chunk in chunks), chunksize=1):
            records.extend(result)
        by_smiles = dict(zip(unique, records, strict=True))
        return [by_smiles[value] for value in values]

    def smiles_to_iupac_batch(self, smiles_list):
        """Return one name per input, preserving the serial batch contract."""
        return self._run(smiles_list, structured=False)

    def smiles_to_iupac_batch_results(self, smiles_list):
        """Return complete NamingResults in the original input order."""
        return self._run(smiles_list, structured=True)
