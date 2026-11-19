# Execution settings

The SMILES APIs accept `mode="full"` (the default) or `mode="fast"`, including
batch and structured-result variants. Molecule APIs and `IUPACBatchPool` use
full mode.

## Timeouts and fallback

Full mode defaults to `timeout_seconds=10.0` and `fallback_to_fast=True`.
The deadline starts when the worker begins naming; startup and queue time are
additional. On a full-mode timeout, its worker is terminated and fast mode is
tried with another ten-second limit. A fallback cannot re-enter full mode.
Total naming time can therefore reach twenty seconds, plus overhead.

- `timeout_seconds=30.0` changes each attempt's deadline.
- `fallback_to_fast=False` reports the full-mode timeout without a fast attempt.
- `timeout_seconds=None` runs full mode in the caller process without a deadline.

Direct fast mode runs in the caller process without a deadline. Its
`timeout_seconds` argument controls any handoff to full mode; that handoff
cannot fall back to fast again.

String APIs return an empty string when no name is available. Structured
results expose `generation_status` and route information. Independently timed
calls may take different fallback paths.

Ordinary timed calls share one persistent helper process and serialize calls
from multiple threads. Use an explicit pool for parallel full-mode naming.

## Parallel pools

`IUPACBatchPool` defaults to four workers. Set `workers` for the available CPU
and memory budget; each worker loads its own rules and caches. Keep the pool
open across batches to retain those caches.

```python
from nispo import IUPACBatchPool

if __name__ == "__main__":
    with IUPACBatchPool(workers=4) as pool:
        results = pool.smiles_to_iupac_batch_results(inputs)
```

The pool preserves input order, count and duplicates. Duplicate input strings
share work within a submitted batch; later batches process them again.
`smiles_to_iupac_batch` itself has no `workers` argument.

`max_requests_per_worker=100_000` retires a worker after that many completed
attempts, counting full and fast attempts across batches. Set a positive integer
to change the interval or `None` to disable count-based retirement. A timeout
or worker failure can still cause replacement. With `timeout_seconds=None`,
pool workers persist until the pool closes and this count-based setting has
no effect.

`chunk_size=16` applies only to an unbounded pool (`timeout_seconds=None`). It
sends up to sixteen unique inputs to a worker per message, reducing transport
overhead. Each worker still names them sequentially. Timed execution dispatches
one input at a time, so chunk size has no effect on the default timed path.
