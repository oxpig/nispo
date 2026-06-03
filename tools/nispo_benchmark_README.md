# NISPO Cluster Benchmark CLI

`nispo_benchmark.py` is repo-safe cluster benchmark tooling. It does not contain
usernames, scratch paths, tokens, or Slurm account details. Cluster-specific
paths are supplied with CLI flags or environment variables.

Typical cluster defaults:

```bash
export NISPO_BENCH_ROOT=/scratch/fast/$USER/nispo-bench
export NISPO_REPO=$HOME/nispo-bench/nispo
export NISPO_OPSIN_JAR=$HOME/nispo-bench/opsin/opsin-cli-2.9.0-jar-with-dependencies.jar
export NISPO_CONDA_ENV_PREFIX=/scratch/fast/$USER/conda-envs/rdkit-pubchem
export NISPO_BENCH_EXCLUDE=swan11.cpu.stats.ox.ac.uk,swan21.cpu.stats.ox.ac.uk
```

The benchmark root layout is:

```text
sets/
  0001_surechembl_ha50_organic_full/
    manifest.json
    molecules.tsv.gz
    shards.tsv
    shards/
runs/
  0001_surechembl_full_934fa50/
    manifest.json
    summary.json
    row_status.tsv.gz
    generated_names.tsv.gz
    failures.tsv.gz
    long_names.tsv.gz
    telemetry.json
    shards/
      shard_00000.summary.json
      shard_00000.telemetry.json
      shard_00000.row_status.tsv.gz
      shard_00000.generated_names.tsv.gz
      shard_00000.failures.tsv.gz
      shard_00000.long_names.tsv.gz
registry/
  sets.tsv
  runs.tsv
  latest_run.txt
logs/
```

Run directories use one short sortable name:

```text
<run_number>_<dataset>_<mode>_<nispo_commit7>
```

For example:

```text
0007_surechembl_full_934fa50
```

CLI lookups accept either the full directory name, `latest`, or a unique
numeric prefix such as `0007`. Timestamps, full set ids, commands, and input
provenance live in `manifest.json` rather than the directory name.

Row IDs are consistently zero-based in this cluster benchmark system:

```text
row_id
source_row_id
```

`row_id` is the row within the benchmark set. `source_row_id` preserves the
source dataset row when available.

Derived sets also preserve parent mapping:

```text
parent_set_id
parent_row_id
source_run_id
selection_metric
selection_value
```

Merged row output files are keyed by `row_id`; consumers should join on
`row_id` rather than relying on file order.

Create a canonical set from a selected molecule TSV:

```bash
python tools/nispo_benchmark.py \
  --root "$NISPO_BENCH_ROOT" \
  create-set \
  --set 0001_surechembl_ha50_organic_full \
  --input /scratch/fast/$USER/surechembl/selections/organic_single_fragment_ha50_no_isotope_v1/surechembl_subset.tsv.gz \
  --workers 200 \
  --min-rows-per-shard 10000 \
  --selection-reason "SureChEMBL organic single-fragment HA50 full set"
```

Shard count is auto-resolved as:

```text
min(workers, ceil(row_count / min_rows_per_shard))
```

Use `--shards N` only when an explicit override is needed. This avoids creating
hundreds of tiny shards for small derived sets.

`create-set` reads the full input and writes all shard files. For multi-million
row datasets, create sets through Slurm or reuse a prepared set; do not run
large set creation on the login node.

Submit a full Slurm run:

```bash
python tools/nispo_benchmark.py \
  --root "$NISPO_BENCH_ROOT" \
  run \
  --set 0001_surechembl_ha50_organic_full \
  --mode full \
  --submit-slurm \
  --workers 200 \
  --exclude "$NISPO_BENCH_EXCLUDE"
```

Slurm submission has safety caps:

```text
--max-workers 200
--max-array-tasks 200
```

These defaults can be overridden explicitly, but ordinary benchmark runs should
stay at or below 200 workers/tasks. Slurm submission also refuses benchmark
roots and conda environments outside `/scratch/fast`.

Run manifests record command arguments, CLI script path/hash, resolved set and
shard manifests, small manifest hashes, row/shard counts, resource requests,
Slurm job ids after submission, NISPO commit, OPSIN checksum, conda env path,
and lightweight Python/RDKit/Java version data when available. Large molecule
gzip inputs are not hashed during run submission.

Create derived sets:

```bash
python tools/nispo_benchmark.py --root "$NISPO_BENCH_ROOT" subset \
  --from-run latest \
  --mode failures \
  --out-set 0002_failures_from_0001 \
  --workers 200 \
  --min-rows-per-shard 10000

python tools/nispo_benchmark.py --root "$NISPO_BENCH_ROOT" subset \
  --from-run latest \
  --mode long-names \
  --min-length 120 \
  --out-set 0003_long_names_from_0001 \
  --workers 200 \
  --min-rows-per-shard 10000
```

Current derived subset modes are `failures`, `failure-type`, and `long-names`.
Planned modes, not implemented yet, include `telemetry-hotspots`,
`regressions`, `new-successes`, `changed-output`, `random`, `stratified`,
`rows-file`, and `smiles-file`.

Inspect and compare:

```bash
python tools/nispo_benchmark.py --root "$NISPO_BENCH_ROOT" inspect --run latest
python tools/nispo_benchmark.py --root "$NISPO_BENCH_ROOT" compare --run 0002 --baseline 0001
```

Lifecycle vocabulary is `created`, `submitted`, `running`, `merging`,
`complete`, `failed`, `partial`, and `cancelled`. Current `inspect` status
inference is file-based, does not mutate Slurm jobs, and may not distinguish
live scheduler states such as `running` or `cancelled` without future Slurm
query integration.

Telemetry is aggregate by default. NISPO owns telemetry; the benchmark CLI is a
collector and generic merger. Numeric totals and timings are summed, counter
and histogram dictionaries are merged additively, lists are concatenated, and
unknown incompatible JSON-safe values are reported under `_merge_warnings` and
`_unmerged_telemetry`. Ordinary new NISPO telemetry counters should not require
benchmark CLI changes.

The merge job uses `afterok:<array_job>`, so if any shard fails, the automatic
merge does not run. Per-shard artifacts and Slurm logs remain the first recovery
evidence. `retry-missing` is read-only in this version and reports missing shard
ids; future recovery work should add `merge --allow-partial`, `repair --run`,
or real failed-shard resubmission.
