#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import re
import shlex
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUSES = [
    "match",
    "mismatch",
    "empty_name",
    "nispo_error",
    "input_invalid_smiles",
    "opsin_parse_failed",
    "opsin_invalid_smiles",
]

DEFAULT_MAX_SLURM_ARRAY_TASKS = 200
DEFAULT_MAX_SLURM_WORKERS = 200

MOLECULE_COLUMNS = [
    "row_id",
    "source_row_id",
    "smiles",
    "canonical_smiles",
    "iupac_names",
    "name_variant_count",
    "heavy_atom_count",
    "molecular_formula",
    "ring_count",
    "fused_ring_system_count",
    "largest_fused_ring_system_ring_count",
    "explicit_chiral_carbon_count",
    "explicit_alkene_stereo_count",
    "murcko_scaffold_smiles",
    "generic_topological_scaffold_smiles",
    "parent_set_id",
    "parent_row_id",
    "source_run_id",
    "selection_metric",
    "selection_value",
    "selection_reason",
    "selection_source",
]

SHARD_COLUMNS = [
    "row_id",
    "source_row_id",
    "cid",
    "smiles",
    "rdkit_parse_ok",
    *[column for column in MOLECULE_COLUMNS if column not in {"row_id", "source_row_id", "smiles"}],
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_root() -> Path:
    if os.environ.get("NISPO_BENCH_ROOT"):
        return Path(os.environ["NISPO_BENCH_ROOT"])
    user = os.environ.get("USER")
    if user and Path("/scratch/fast").exists():
        return Path("/scratch/fast") / user / "nispo-bench"
    return Path.cwd() / "nispo-bench"


def reject_bitbucket(*paths: Path | str) -> None:
    for path in paths:
        text = str(path)
        if "/bitbucket" in text:
            raise SystemExit(f"refusing /bitbucket path: {text}")


def require_scratch_fast(path: Path | str, label: str) -> None:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to("/scratch/fast")
    except ValueError:
        raise SystemExit(f"{label} must be under /scratch/fast for Slurm benchmark work: {resolved}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_args() -> dict[str, Any]:
    return {
        "argv": sys.argv[:],
        "command_line": " ".join(shlex.quote(part) for part in sys.argv),
    }


def optional_command_output(command: list[str], *, timeout: int = 10) -> str:
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
    except Exception:
        return ""
    output = (completed.stdout or completed.stderr).strip()
    return output if completed.returncode == 0 else ""


def environment_versions(conda_env: str) -> dict[str, str]:
    java_version = optional_command_output(["java", "-version"])
    versions = {
        "python": sys.version.split()[0],
        "java": java_version.splitlines()[0] if java_version else "",
        "rdkit": "",
    }
    python_bin = Path(conda_env) / "bin" / "python" if conda_env else None
    if python_bin and python_bin.exists():
        versions["python_conda"] = optional_command_output([str(python_bin), "--version"])
        versions["rdkit"] = optional_command_output([str(python_bin), "-c", "import rdkit; print(rdkit.__version__)"])
    return versions


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def rows_from_tsv(path: Path):
    with open_text(path, "rt") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def mkdir_layout(root: Path) -> None:
    for name in ["sets", "runs", "registry", "logs"]:
        (root / name).mkdir(parents=True, exist_ok=True)


def resolve_set_dir(root: Path, set_id: str) -> Path:
    set_dir = root / "sets" / set_id
    if not set_dir.exists():
        raise SystemExit(f"missing benchmark set: {set_dir}")
    return set_dir


def resolve_run_dir(root: Path, run: str) -> Path:
    if run == "latest":
        latest = root / "registry" / "latest_run.txt"
        if not latest.exists():
            raise SystemExit("latest run is not defined")
        run = latest.read_text(encoding="utf-8").strip()
    run_dir = root / "runs" / run
    if run_dir.exists():
        return run_dir
    if run.isdigit():
        matches = sorted(path for path in (root / "runs").glob(f"{run}_*") if path.is_dir())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise SystemExit(f"ambiguous benchmark run prefix {run}: {', '.join(path.name for path in matches)}")
    raise SystemExit(f"missing benchmark run: {run_dir}")


def safe_slug(value: str, *, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return (slug or "run")[:max_length]


def run_number_from_name(name: str) -> int | None:
    match = re.match(r"^(\d{4})(?:_|$)", name)
    return int(match.group(1)) if match else None


def dataset_label_from_set_id(set_id: str) -> str:
    lowered = set_id.lower()
    for label in ["surechembl", "pubchem", "smoke"]:
        if label in lowered:
            return label
    stripped = re.sub(r"^\d+[_-]+", "", set_id)
    stripped = re.sub(r"\d{8}T\d{6}Z", "", stripped)
    stripped = stripped.strip("_-")
    return safe_slug(stripped, max_length=32)


def allocate_run_dir(root: Path, *, set_id: str, mode: str, commit: str) -> tuple[str, str, Path]:
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in runs.iterdir():
        if path.is_dir():
            number = run_number_from_name(path.name)
            if number is not None:
                existing.append(number)
    next_id = max(existing, default=0) + 1
    while True:
        run_number = f"{next_id:04d}"
        run_id = "_".join([
            run_number,
            dataset_label_from_set_id(set_id),
            safe_slug(mode, max_length=30),
            safe_slug(commit[:7] if commit and commit != "unknown" else "unknown", max_length=20),
        ])
        run_dir = runs / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            next_id += 1
            continue
        (run_dir / "shards").mkdir()
        return run_number, run_id, run_dir


def append_registry_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def hash_shard(row_id: int, num_shards: int, seed: str) -> int:
    digest = hashlib.blake2b(f"{seed}\t{row_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_shards


def count_tsv_rows(path: Path) -> int:
    with open_text(path, "rt") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def resolve_shard_count(*, row_count: int, explicit_shards: int, workers: int, min_rows_per_shard: int) -> int:
    if explicit_shards:
        return explicit_shards
    if workers <= 0:
        raise SystemExit("--workers must be positive")
    if min_rows_per_shard <= 0:
        raise SystemExit("--min-rows-per-shard must be positive")
    if row_count <= 0:
        return 1
    shards_needed = (row_count + min_rows_per_shard - 1) // min_rows_per_shard
    return max(1, min(workers, shards_needed))


def create_set(args: argparse.Namespace) -> int:
    root = args.root
    mkdir_layout(root)
    set_dir = root / "sets" / args.set_id
    if set_dir.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing set: {set_dir}")
    set_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = set_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    molecules_path = set_dir / "molecules.tsv.gz"
    manifest_path = set_dir / "manifest.json"
    shards_manifest = set_dir / "shards.tsv"
    reject_bitbucket(args.input, set_dir)

    source_row_count = count_tsv_rows(args.input)
    shard_count = resolve_shard_count(
        row_count=source_row_count if not args.limit else min(source_row_count, args.limit),
        explicit_shards=args.shards,
        workers=args.workers,
        min_rows_per_shard=args.min_rows_per_shard,
    )

    shard_handles = []
    shard_writers = []
    shard_paths = []
    shard_counts = [0] * shard_count
    tmp_suffix = f".{os.getpid()}.tmp"
    try:
        for shard_number in range(shard_count):
            path = shard_dir / f"shard_{shard_number:05d}.tsv.gz"
            shard_paths.append(path)
            handle = gzip.open(path.with_suffix(path.suffix + tmp_suffix), "wt", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=SHARD_COLUMNS, delimiter="\t")
            writer.writeheader()
            shard_handles.append(handle)
            shard_writers.append(writer)

        total = 0
        with gzip.open(molecules_path.with_suffix(molecules_path.suffix + tmp_suffix), "wt", newline="", encoding="utf-8") as out_handle:
            molecule_writer = csv.DictWriter(out_handle, fieldnames=MOLECULE_COLUMNS, delimiter="\t")
            molecule_writer.writeheader()
            for total, row in enumerate(rows_from_tsv(args.input), start=1):
                row_id = total - 1
                smiles = row.get("canonical_smiles") or row.get("smiles") or row.get("SMILES") or ""
                if str(row.get("source_row_index_1based", "")).isdigit():
                    source_row_id = str(int(row["source_row_index_1based"]) - 1)
                elif str(row.get("row_index_1based", "")).isdigit():
                    source_row_id = str(int(row["row_index_1based"]) - 1)
                elif str(row.get("source_row_id", "")).isdigit():
                    source_row_id = row["source_row_id"]
                else:
                    source_row_id = str(row_id)
                molecule = {
                    "row_id": row_id,
                    "source_row_id": source_row_id,
                    "smiles": smiles,
                    "canonical_smiles": row.get("canonical_smiles", smiles),
                    "iupac_names": row.get("iupac_names") or row.get("IUPAC Name") or "",
                    "name_variant_count": row.get("name_variant_count", ""),
                    "heavy_atom_count": row.get("heavy_atom_count", ""),
                    "molecular_formula": row.get("molecular_formula", ""),
                    "ring_count": row.get("ring_count", ""),
                    "fused_ring_system_count": row.get("fused_ring_system_count", ""),
                    "largest_fused_ring_system_ring_count": row.get("largest_fused_ring_system_ring_count", ""),
                    "explicit_chiral_carbon_count": row.get("explicit_chiral_carbon_count", ""),
                    "explicit_alkene_stereo_count": row.get("explicit_alkene_stereo_count", ""),
                    "murcko_scaffold_smiles": row.get("murcko_scaffold_smiles", ""),
                    "generic_topological_scaffold_smiles": row.get("generic_topological_scaffold_smiles", ""),
                    "parent_set_id": row.get("parent_set_id", ""),
                    "parent_row_id": row.get("parent_row_id", ""),
                    "source_run_id": row.get("source_run_id", ""),
                    "selection_metric": row.get("selection_metric", ""),
                    "selection_value": row.get("selection_value", ""),
                    "selection_reason": args.selection_reason,
                    "selection_source": row.get("selection_source") or getattr(args, "source_input_override", "") or str(args.input),
                }
                molecule_writer.writerow(molecule)
                shard_idx = hash_shard(row_id, shard_count, args.seed)
                shard_row = {
                    "row_id": row_id,
                    "source_row_id": source_row_id,
                    "cid": row_id,
                    "smiles": smiles,
                    "rdkit_parse_ok": "1",
                    **{column: molecule.get(column, "") for column in SHARD_COLUMNS if column not in {"row_id", "source_row_id", "cid", "smiles", "rdkit_parse_ok"}},
                }
                shard_writers[shard_idx].writerow(shard_row)
                shard_counts[shard_idx] += 1
                if args.limit and total >= args.limit:
                    break
                if total % 1_000_000 == 0:
                    print(f"processed_rows={total}", flush=True)
    finally:
        for handle in shard_handles:
            handle.close()

    molecules_path.with_suffix(molecules_path.suffix + tmp_suffix).replace(molecules_path)
    for path in shard_paths:
        path.with_suffix(path.suffix + tmp_suffix).replace(path)

    with shards_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["shard_index", "path", "rows"], delimiter="\t")
        writer.writeheader()
        for idx, (path, count) in enumerate(zip(shard_paths, shard_counts, strict=True)):
            writer.writerow({"shard_index": f"{idx:05d}", "path": path, "rows": count})

    manifest = {
        "set_id": args.set_id,
        "created_at_utc": utc_now(),
        "source_input": getattr(args, "source_input_override", "") or str(args.input),
        "molecules": str(molecules_path),
        "shards_manifest": str(shards_manifest),
        "shards": shard_count,
        "requested_workers": args.workers,
        "explicit_shards": args.shards,
        "min_rows_per_shard": args.min_rows_per_shard,
        "row_id_base": 0,
        "row_count": total,
        "source_row_count": source_row_count,
        "randomization": "deterministic blake2b row_id shard assignment",
        "seed": args.seed,
        "selection_reason": args.selection_reason,
        "min_shard_rows": min(shard_counts) if shard_counts else 0,
        "max_shard_rows": max(shard_counts) if shard_counts else 0,
    }
    write_json(manifest_path, manifest)
    (set_dir / "README.md").write_text(
        f"# {args.set_id}\n\nRows: {total}\n\nManifest: `manifest.json`\n",
        encoding="utf-8",
    )
    append_registry_row(root / "registry" / "sets.tsv", {
        "set_id": args.set_id,
        "created_at_utc": manifest["created_at_utc"],
        "row_count": total,
        "path": set_dir,
    })
    print(json.dumps(manifest, sort_keys=True))
    return 0


def canonicalize(smiles: str, chem) -> tuple[str, str]:
    if not smiles:
        return "", "blank"
    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        return "", "invalid_smiles"
    return chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), "ok"


def normalize_names(raw_names: Any, expected_count: int) -> tuple[list[str], str]:
    try:
        raw_names = list(raw_names)
    except TypeError:
        return [""] * expected_count, "batch returned non-iterable names"
    if len(raw_names) != expected_count:
        return [""] * expected_count, f"batch returned {len(raw_names)} names for {expected_count} SMILES"
    return [" ".join(name.split()) if isinstance(name, str) else "" for name in raw_names], ""


def parse_opsin(names: list[str], opsin_jar: Path, work_dir: Path) -> list[str]:
    parsed = [""] * len(names)
    indexed = [(idx, name) for idx, name in enumerate(names) if name]
    if not indexed:
        return parsed
    with tempfile.TemporaryDirectory(prefix="opsin_", dir=work_dir) as tmp:
        tmp_dir = Path(tmp)
        names_file = tmp_dir / "names.txt"
        out_file = tmp_dir / "opsin.smi"
        names_file.write_text("\n".join(name for _, name in indexed) + "\n", encoding="utf-8")
        completed = subprocess.run(
            ["java", "-jar", str(opsin_jar), "-o", "smi", str(names_file), str(out_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        lines = out_file.read_text(encoding="utf-8").splitlines() if completed.returncode == 0 and out_file.exists() else []
    lines = (lines + [""] * len(indexed))[: len(indexed)]
    for (idx, _), line in zip(indexed, lines, strict=True):
        parsed[idx] = line.strip()
    return parsed


def generic_merge_telemetry(items: list[dict[str, Any]]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    unmerged: dict[str, dict[str, Any]] = {}

    def merge_values(values: list[Any], path: str) -> Any:
        values = [value for value in values if value is not None]
        if not values:
            return None
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            return sum(values)
        if all(isinstance(value, dict) for value in values):
            keys = sorted({key for value in values for key in value})
            merged = {}
            for key in keys:
                child_path = f"{path}.{key}" if path else str(key)
                child_value = merge_values([value.get(key) for value in values], child_path)
                if child_value is not None:
                    merged[key] = child_value
            return merged
        if all(isinstance(value, list) for value in values):
            return [item for value in values for item in value]
        warnings.append({"path": path or "<root>", "reason": "incompatible telemetry value types", "unmerged_count": len(values)})
        unmerged[path or "<root>"] = {"values": values[:20], "unmerged_count": len(values)}
        return None

    merged = merge_values(items, "") or {}
    if warnings:
        merged["_merge_warnings"] = warnings
        merged["_unmerged_telemetry"] = unmerged
    return merged


def iter_batches(shard_path: Path, batch_size: int):
    batch = []
    with gzip.open(shard_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("rdkit_parse_ok", "1") != "1":
                continue
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def load_manifest_row(manifest: Path, task_index: int) -> dict[str, str]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for offset, row in enumerate(reader, start=1):
            if offset == task_index:
                return row
    raise SystemExit(f"task index {task_index} not found in {manifest}")


def run_worker(args: argparse.Namespace) -> int:
    reject_bitbucket(args.manifest, args.run_dir, args.opsin_jar)
    row = load_manifest_row(args.manifest, args.task_index)
    shard_index = row["shard_index"]
    shard_path = Path(row["path"])
    reject_bitbucket(shard_path)
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    shard_out = args.run_dir / "shards"
    shard_out.mkdir(parents=True, exist_ok=True)

    if args.nispo_repo and str(args.nispo_repo) not in sys.path:
        sys.path.insert(0, str(args.nispo_repo))
    from rdkit import Chem, RDLogger
    from nispo.api import _smiles_to_iupac_batch_with_telemetry
    try:
        from nispo.telemetry import compact_telemetry_summary
    except Exception:
        compact_telemetry_summary = None

    RDLogger.DisableLog("rdApp.*")
    records = []
    telemetry_items = []
    nispo_seconds = 0.0
    for batch in iter_batches(shard_path, args.batch_size):
        smiles = [item.get("smiles", "") for item in batch]
        input_canon = [canonicalize(value, Chem) for value in smiles]
        started = time.perf_counter()
        try:
            raw_names, telemetry = _smiles_to_iupac_batch_with_telemetry(smiles)
            batch_error = ""
        except Exception as exc:
            raw_names = [""] * len(smiles)
            telemetry = {}
            batch_error = repr(exc)
        nispo_seconds += time.perf_counter() - started
        names, normalize_error = normalize_names(raw_names, len(smiles))
        if normalize_error:
            batch_error = normalize_error
        telemetry_items.append(telemetry)
        for input_row, (input_smiles, input_status), name in zip(batch, input_canon, names, strict=True):
            records.append((input_row, input_smiles, input_status, name, batch_error))

    opsin_started = time.perf_counter()
    opsin_smiles = parse_opsin([record[3] for record in records], args.opsin_jar, work_dir)
    opsin_seconds = time.perf_counter() - opsin_started
    telemetry = generic_merge_telemetry(telemetry_items)

    status_counts = Counter({status: 0 for status in STATUSES})
    length_histogram = Counter()
    row_status_path = shard_out / f"shard_{shard_index}.row_status.tsv.gz"
    names_path = shard_out / f"shard_{shard_index}.generated_names.tsv.gz"
    failures_path = shard_out / f"shard_{shard_index}.failures.tsv.gz"
    long_path = shard_out / f"shard_{shard_index}.long_names.tsv.gz"
    with gzip.open(row_status_path, "wt", newline="", encoding="utf-8") as status_handle, \
        gzip.open(names_path, "wt", newline="", encoding="utf-8") as names_handle, \
        gzip.open(failures_path, "wt", newline="", encoding="utf-8") as failures_handle, \
        gzip.open(long_path, "wt", newline="", encoding="utf-8") as long_handle:
        status_writer = csv.DictWriter(status_handle, fieldnames=["row_id", "source_row_id", "status"], delimiter="\t")
        names_writer = csv.DictWriter(names_handle, fieldnames=["row_id", "source_row_id", "generated_name", "generated_length"], delimiter="\t")
        failure_fields = [
            "row_id", "source_row_id", "smiles", "input_canonical_smiles", "generated_name",
            "generated_length", "opsin_smiles", "opsin_canonical_smiles", "status",
        ]
        failure_writer = csv.DictWriter(failures_handle, fieldnames=failure_fields, delimiter="\t")
        long_writer = csv.DictWriter(long_handle, fieldnames=failure_fields, delimiter="\t")
        status_writer.writeheader()
        names_writer.writeheader()
        failure_writer.writeheader()
        long_writer.writeheader()
        for (input_row, input_canonical, input_status, name, batch_error), parsed in zip(records, opsin_smiles, strict=True):
            opsin_canonical, opsin_status = canonicalize(parsed, Chem)
            if batch_error:
                status = "nispo_error"
            elif not name:
                status = "empty_name"
            elif input_status != "ok":
                status = "input_invalid_smiles"
            elif opsin_status == "blank":
                status = "opsin_parse_failed"
            elif opsin_status != "ok":
                status = "opsin_invalid_smiles"
            elif input_canonical == opsin_canonical:
                status = "match"
            else:
                status = "mismatch"
            status_counts[status] += 1
            row_id = input_row.get("row_id", input_row.get("cid", ""))
            source_row_id = input_row.get("source_row_id", "")
            generated_length = len(name)
            status_writer.writerow({"row_id": row_id, "source_row_id": source_row_id, "status": status})
            names_writer.writerow({
                "row_id": row_id,
                "source_row_id": source_row_id,
                "generated_name": name,
                "generated_length": generated_length,
            })
            if name:
                length_histogram[generated_length] += 1
            detail = {
                "row_id": row_id,
                "source_row_id": source_row_id,
                "smiles": input_row.get("smiles", ""),
                "input_canonical_smiles": input_canonical,
                "generated_name": name,
                "generated_length": generated_length,
                "opsin_smiles": parsed,
                "opsin_canonical_smiles": opsin_canonical,
                "status": status,
            }
            if status != "match":
                failure_writer.writerow(detail)
            if generated_length >= args.long_name_threshold:
                long_writer.writerow(detail)

    telemetry_path = shard_out / f"shard_{shard_index}.telemetry.json"
    write_json(telemetry_path, telemetry)
    telemetry_summary = {"path": str(telemetry_path)}
    if compact_telemetry_summary is not None:
        try:
            telemetry_summary = compact_telemetry_summary(telemetry, telemetry_path)
        except Exception as exc:
            telemetry_summary = {"path": str(telemetry_path), "warning": repr(exc)}

    processed = len(records)
    summary = {
        "run_id": args.run_id,
        "shard_index": shard_index,
        "input_path": str(shard_path),
        "processed_rows": processed,
        "status_counts": {status: int(status_counts[status]) for status in STATUSES},
        "name_length": {
            "nonempty_count": sum(length_histogram.values()),
            "nonempty_sum": sum(length * count for length, count in length_histogram.items()),
            "histogram": {str(k): int(v) for k, v in sorted(length_histogram.items())},
        },
        "timing": {
            "nispo_seconds": round(nispo_seconds, 6),
            "nispo_milliseconds_per_molecule": round(nispo_seconds * 1000 / processed, 6) if processed else 0.0,
            "opsin_seconds": round(opsin_seconds, 6),
            "opsin_milliseconds_per_molecule": round(opsin_seconds * 1000 / processed, 6) if processed else 0.0,
        },
        "telemetry": telemetry_summary,
        "tooling": {"nispo_commit": args.nispo_commit, "opsin_sha256": args.opsin_sha256},
    }
    write_json(shard_out / f"shard_{shard_index}.summary.json", summary)
    return 0


def concatenate_tsv_gz(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    header_written = False
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as out_handle:
        writer = None
        for path in inputs:
            with gzip.open(path, "rt", newline="", encoding="utf-8") as in_handle:
                reader = csv.DictReader(in_handle, delimiter="\t")
                if writer is None:
                    writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames, delimiter="\t")
                if not header_written:
                    writer.writeheader()
                    header_written = True
                for row in reader:
                    writer.writerow(row)
    tmp.replace(output)


def percentile(hist: Counter[int], q: float) -> int | None:
    total = sum(hist.values())
    if not total:
        return None
    target = max(1, int(total * q + (0 if total * q == int(total * q) else 1)))
    seen = 0
    for key in sorted(hist):
        seen += hist[key]
        if seen >= target:
            return key
    return max(hist)


def shard_indexes_from_manifest(shards_manifest: Path) -> list[str]:
    with shards_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row["shard_index"] for row in reader]


def run_status(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {
            "lifecycle_status": "failed",
            "reason": "manifest.json missing",
            "expected_shards": 0,
            "completed_shards": 0,
            "missing_shards": [],
            "merged_outputs_complete": False,
        }
    manifest = read_json(manifest_path)
    expected: list[str] = []
    manifest_error = ""
    shards_manifest_value = manifest.get("input_manifest", "")
    if shards_manifest_value:
        try:
            expected = shard_indexes_from_manifest(Path(shards_manifest_value))
        except Exception as exc:
            manifest_error = repr(exc)
    shard_dir = run_dir / "shards"
    summaries = {}
    if shard_dir.exists():
        for path in shard_dir.glob("shard_*.summary.json"):
            try:
                summaries[read_json(path)["shard_index"]] = str(path)
            except Exception:
                continue
    completed = sorted(idx for idx in expected if idx in summaries)
    missing = sorted(idx for idx in expected if idx not in summaries)
    merged_files = ["summary.json", "row_status.tsv.gz", "generated_names.tsv.gz", "failures.tsv.gz", "long_names.tsv.gz", "telemetry.json"]
    merged_outputs = {name: (run_dir / name).exists() for name in merged_files}
    merged_complete = all(merged_outputs.values())
    slurm = manifest.get("slurm", {})
    if merged_complete and not missing:
        lifecycle = "complete"
    elif expected and completed and missing:
        lifecycle = "partial"
    elif slurm.get("merge_job") and not merged_complete and not missing and completed:
        lifecycle = "merging"
    elif slurm.get("array_job") and not completed:
        lifecycle = "submitted"
    elif expected and not completed:
        lifecycle = "created"
    else:
        lifecycle = "failed"
    return {
        "lifecycle_status": lifecycle,
        "expected_shards": len(expected),
        "completed_shards": len(completed),
        "missing_shard_count": len(missing),
        "missing_shards": missing[:100],
        "missing_shards_truncated": len(missing) > 100,
        "merged_outputs": merged_outputs,
        "merged_outputs_complete": merged_complete,
        "shards_manifest_error": manifest_error,
    }


def merge_run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir
    manifest = read_json(run_dir / "manifest.json")
    shards_manifest = Path(manifest["input_manifest"])
    expected = shard_indexes_from_manifest(shards_manifest)
    shard_dir = run_dir / "shards"
    summaries = {json.loads(path.read_text())["shard_index"]: path for path in shard_dir.glob("shard_*.summary.json")}
    missing = [idx for idx in expected if idx not in summaries]
    if missing:
        raise SystemExit(f"missing {len(missing)} shard summaries: {', '.join(missing[:20])}")

    status_counts = Counter({status: 0 for status in STATUSES})
    length_hist = Counter()
    telemetry_items = []
    nispo_seconds = opsin_seconds = 0.0
    total = 0
    for idx in expected:
        summary = read_json(summaries[idx])
        total += int(summary["processed_rows"])
        for status, count in summary["status_counts"].items():
            status_counts[status] += int(count)
        for length, count in summary["name_length"]["histogram"].items():
            length_hist[int(length)] += int(count)
        timing = summary.get("timing", {})
        nispo_seconds += float(timing.get("nispo_seconds", 0.0))
        opsin_seconds += float(timing.get("opsin_seconds", 0.0))
        telemetry_path = shard_dir / f"shard_{idx}.telemetry.json"
        if telemetry_path.exists():
            telemetry_items.append(read_json(telemetry_path))

    concatenate_tsv_gz([shard_dir / f"shard_{idx}.row_status.tsv.gz" for idx in expected], run_dir / "row_status.tsv.gz")
    concatenate_tsv_gz([shard_dir / f"shard_{idx}.generated_names.tsv.gz" for idx in expected], run_dir / "generated_names.tsv.gz")
    concatenate_tsv_gz([shard_dir / f"shard_{idx}.failures.tsv.gz" for idx in expected], run_dir / "failures.tsv.gz")
    concatenate_tsv_gz([shard_dir / f"shard_{idx}.long_names.tsv.gz" for idx in expected], run_dir / "long_names.tsv.gz")
    telemetry = generic_merge_telemetry(telemetry_items)
    write_json(run_dir / "telemetry.json", telemetry)

    match = status_counts["match"]
    nonempty = sum(length_hist.values())
    summary = {
        "run_id": manifest["run_id"],
        "run_number": manifest.get("run_number", manifest["run_id"][:4]),
        "set_id": manifest["set_id"],
        "mode": manifest["mode"],
        "total": total,
        "match": match,
        "roundtrip_success_rate": match / total if total else 0.0,
        "status_counts": {status: int(status_counts[status]) for status in STATUSES},
        "name_length": {
            "nonempty_count": nonempty,
            "mean_nonempty_name_length": (sum(k * v for k, v in length_hist.items()) / nonempty) if nonempty else None,
            "median_nonempty_name_length": percentile(length_hist, 0.5),
            "p90_nonempty_name_length": percentile(length_hist, 0.9),
        },
        "timing": {
            "nispo_seconds": round(nispo_seconds, 6),
            "nispo_milliseconds_per_molecule": round(nispo_seconds * 1000 / total, 6) if total else 0.0,
            "opsin_seconds": round(opsin_seconds, 6),
            "opsin_milliseconds_per_molecule": round(opsin_seconds * 1000 / total, 6) if total else 0.0,
        },
        "outputs": {
            "row_status": str(run_dir / "row_status.tsv.gz"),
            "generated_names": str(run_dir / "generated_names.tsv.gz"),
            "failures": str(run_dir / "failures.tsv.gz"),
            "long_names": str(run_dir / "long_names.tsv.gz"),
            "telemetry": str(run_dir / "telemetry.json"),
        },
        "completed_at_utc": utc_now(),
        "tooling": manifest.get("tooling", {}),
    }
    write_json(run_dir / "summary.json", summary)
    summary["run_status"] = run_status(run_dir)
    write_json(run_dir / "summary.json", summary)
    root = run_dir.parents[1]
    (root / "registry" / "latest_run.txt").write_text(manifest["run_id"] + "\n", encoding="utf-8")
    append_registry_row(root / "registry" / "runs.tsv", {
        "run_id": manifest["run_id"],
        "run_number": manifest.get("run_number", manifest["run_id"][:4]),
        "set_id": manifest["set_id"],
        "mode": manifest["mode"],
        "completed_at_utc": summary["completed_at_utc"],
        "total": total,
        "match": match,
        "roundtrip_success_rate": summary["roundtrip_success_rate"],
        "path": run_dir,
    })
    print(json.dumps(summary, sort_keys=True))
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    root = args.root
    mkdir_layout(root)
    set_dir = resolve_set_dir(root, args.set)
    set_manifest = read_json(set_dir / "manifest.json")
    nispo_repo = Path(args.nispo_repo or os.environ.get("NISPO_REPO", Path.cwd()))
    opsin_jar = Path(args.opsin_jar or os.environ.get("NISPO_OPSIN_JAR", Path.home() / "nispo-bench/opsin/opsin-cli-2.9.0-jar-with-dependencies.jar"))
    conda_env = args.conda_env or os.environ.get("NISPO_CONDA_ENV_PREFIX", "")
    commit = subprocess.check_output(["git", "-C", str(nispo_repo), "rev-parse", "HEAD"], text=True).strip() if (nispo_repo / ".git").exists() else "unknown"
    opsin_sha = hashlib.sha256(opsin_jar.read_bytes()).hexdigest() if opsin_jar.exists() else ""
    run_number, run_id, run_dir = allocate_run_dir(root, set_id=args.set, mode=args.mode, commit=commit)
    script = Path(__file__).resolve()
    set_manifest_path = set_dir / "manifest.json"
    shards_manifest_path = Path(set_manifest["shards_manifest"])
    shard_count = sum(1 for _ in shards_manifest_path.open(encoding="utf-8")) - 1
    manifest = {
        "run_id": run_id,
        "run_number": run_number,
        "run_dir": str(run_dir),
        "set_id": args.set,
        "set_dir": str(set_dir),
        "set_manifest": str(set_manifest_path),
        "input_manifest": str(shards_manifest_path),
        "mode": args.mode,
        "from_run": args.from_run,
        "created_at_utc": utc_now(),
        "command": command_args(),
        "cli": {
            "script_path": str(script),
            "script_sha256": sha256_file(script) if script.exists() else "",
        },
        "input": {
            "set_id": args.set,
            "set_dir": str(set_dir),
            "set_manifest": str(set_manifest_path),
            "set_manifest_sha256": sha256_file(set_manifest_path),
            "shards_manifest": str(shards_manifest_path),
            "shards_manifest_sha256": sha256_file(shards_manifest_path),
            "row_count": set_manifest.get("row_count"),
            "source_row_count": set_manifest.get("source_row_count"),
            "shard_count": shard_count,
            "source_input": set_manifest.get("source_input", ""),
        },
        "row_id_base": 0,
        "batch_size": args.batch_size,
        "long_name_threshold": args.long_name_threshold,
        "telemetry": args.telemetry,
        "tooling": {
            "nispo_repo": str(nispo_repo),
            "nispo_commit": commit,
            "opsin_jar": str(opsin_jar),
            "opsin_sha256": opsin_sha,
            "conda_env": conda_env,
            "environment_versions": environment_versions(conda_env),
        },
        "slurm": {
            "partition": args.partition,
            "time": args.time,
            "mem": args.mem,
            "exclude": args.exclude,
            "requested_workers": args.workers,
            "max_workers": args.max_workers,
            "max_array_tasks": args.max_array_tasks,
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    if not args.submit_slurm:
        print(f"created run manifest: {run_dir / 'manifest.json'}")
        return 0

    if not conda_env:
        raise SystemExit("--conda-env or NISPO_CONDA_ENV_PREFIX is required for --submit-slurm")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    if args.max_array_tasks <= 0:
        raise SystemExit("--max-array-tasks must be positive")
    if args.workers > args.max_workers:
        raise SystemExit(f"refusing --workers {args.workers}; cap is --max-workers {args.max_workers}")
    require_scratch_fast(root, "benchmark root")
    require_scratch_fast(conda_env, "conda env")
    if shard_count <= 0:
        raise SystemExit(f"no shards found in {set_manifest['shards_manifest']}")
    if shard_count > args.max_array_tasks:
        raise SystemExit(f"refusing to submit {shard_count} array tasks; cap is --max-array-tasks {args.max_array_tasks}")
    array_spec = f"1-{shard_count}" if shard_count <= args.workers else f"1-{shard_count}%{args.workers}"
    export = {
        "NISPO_BENCH_ROOT": str(root),
        "RUN_DIR": str(run_dir),
        "RUN_ID": run_id,
        "MANIFEST": set_manifest["shards_manifest"],
        "OPSIN_JAR": str(opsin_jar),
        "NISPO_REPO": str(nispo_repo),
        "NISPO_COMMIT": commit,
        "OPSIN_SHA256": opsin_sha,
        "CONDA_ENV_PREFIX": conda_env,
        "BATCH_SIZE": str(args.batch_size),
        "LONG_NAME_THRESHOLD": str(args.long_name_threshold),
    }
    exclude = args.exclude or ""
    array_cmd = [
        "sbatch", "--parsable",
        f"--array={array_spec}",
        "--partition", args.partition,
        "--cpus-per-task", "1",
        "--mem", args.mem,
        "--time", args.time,
        "--job-name", "nispo-bench",
        "--output", str(root / "logs" / f"run_{run_id}_%A_%a.log"),
        "--export", "ALL," + ",".join(f"{key}={value}" for key, value in export.items()),
    ]
    if exclude:
        array_cmd.extend(["--exclude", exclude])
    array_cmd.extend([
        "--wrap",
        f': "${{SCRATCH_FAST:?SCRATCH_FAST is required}}"; "$CONDA_ENV_PREFIX/bin/python" {script} _worker --run-dir "$RUN_DIR" --run-id "$RUN_ID" --manifest "$MANIFEST" --task-index "$SLURM_ARRAY_TASK_ID" --opsin-jar "$OPSIN_JAR" --nispo-repo "$NISPO_REPO" --nispo-commit "$NISPO_COMMIT" --opsin-sha256 "$OPSIN_SHA256" --batch-size "$BATCH_SIZE" --long-name-threshold "$LONG_NAME_THRESHOLD" --work-dir "$SCRATCH_FAST/nispo-benchmark/run_${{RUN_ID}}/task_${{SLURM_ARRAY_TASK_ID}}_${{SLURM_JOB_ID}}"',
    ])
    array_job = subprocess.check_output(array_cmd, text=True).strip()
    merge_cmd = [
        "sbatch", "--parsable",
        "--dependency", f"afterok:{array_job}",
        "--partition", args.partition,
        "--cpus-per-task", "1",
        "--mem", "4G",
        "--time", "00:30:00",
        "--job-name", "nispo-bench-merge",
        "--output", str(root / "logs" / f"run_{run_id}_merge_%j.log"),
    ]
    if exclude:
        merge_cmd.extend(["--exclude", exclude])
    merge_cmd.extend(["--wrap", f'python3 {script} --root {root} merge --run {run_id}'])
    merge_job = subprocess.check_output(merge_cmd, text=True).strip()
    manifest["slurm"].update({
        "array_job": array_job,
        "merge_job": merge_job,
        "array_spec": array_spec,
        "submitted_at_utc": utc_now(),
        "requested_workers": args.workers,
        "max_workers": args.max_workers,
        "max_array_tasks": args.max_array_tasks,
    })
    write_json(run_dir / "manifest.json", manifest)
    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"array_job={array_job}")
    print(f"merge_job={merge_job}")
    return 0


def inspect_run(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.root, args.run)
    print(f"run_dir={run_dir}")
    print(json.dumps({"run_status": run_status(run_dir)}, indent=2, sort_keys=True))
    for name in ["manifest.json", "summary.json"]:
        path = run_dir / name
        if path.exists():
            print(json.dumps(read_json(path), indent=2, sort_keys=True))
        else:
            print(f"{name}=missing")
    return 0


def subset(args: argparse.Namespace) -> int:
    root = args.root
    source_run = resolve_run_dir(root, args.from_run)
    run_manifest = read_json(source_run / "manifest.json")
    set_dir = resolve_set_dir(root, run_manifest["set_id"])
    statuses = {row["row_id"]: row["status"] for row in rows_from_tsv(source_run / "row_status.tsv.gz")}
    lengths = {row["row_id"]: int(row.get("generated_length") or 0) for row in rows_from_tsv(source_run / "generated_names.tsv.gz")}
    selected_rows = []
    for row in rows_from_tsv(set_dir / "molecules.tsv.gz"):
        status = statuses.get(row["row_id"], "")
        keep = False
        reason = ""
        if args.mode == "failures" and status != "match":
            keep, reason = True, f"failure_from_run:{run_manifest['run_id']}:{status}"
        elif args.mode == "failure-type" and status == args.status:
            keep, reason = True, f"failure_type_from_run:{run_manifest['run_id']}:{status}"
        elif args.mode == "long-names" and lengths.get(row["row_id"], 0) >= args.min_length:
            keep, reason = True, f"long_name_from_run:{run_manifest['run_id']}:gte_{args.min_length}"
        if keep:
            row = dict(row)
            row["parent_set_id"] = run_manifest["set_id"]
            row["parent_row_id"] = row["row_id"]
            row["source_run_id"] = run_manifest["run_id"]
            row["selection_metric"] = args.mode
            row["selection_value"] = status if args.mode != "long-names" else str(lengths.get(row["row_id"], 0))
            row["selection_reason"] = reason
            row["selection_source"] = str(source_run)
            selected_rows.append(row)
    if args.limit:
        selected_rows = selected_rows[: args.limit]
    tmp = args.root / "sets" / f".{args.out_set}.input.tsv.gz"
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MOLECULE_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected_rows)
    create_args = argparse.Namespace(
        root=args.root,
        set_id=args.out_set,
        input=tmp,
        shards=args.shards,
        workers=args.workers,
        min_rows_per_shard=args.min_rows_per_shard,
        seed=args.seed,
        selection_reason=f"{args.mode} from run {run_manifest['run_id']}",
        limit=0,
        force=args.force,
        source_input_override=str(source_run),
    )
    try:
        return create_set(create_args)
    finally:
        tmp.unlink(missing_ok=True)


def compare(args: argparse.Namespace) -> int:
    run = read_json(resolve_run_dir(args.root, args.run) / "summary.json")
    baseline = read_json(resolve_run_dir(args.root, args.baseline) / "summary.json")
    output = {
        "run": run["run_id"],
        "baseline": baseline["run_id"],
        "match_delta": run["match"] - baseline["match"],
        "success_rate_delta": run["roundtrip_success_rate"] - baseline["roundtrip_success_rate"],
        "status_delta": {
            status: run["status_counts"].get(status, 0) - baseline["status_counts"].get(status, 0)
            for status in STATUSES
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def retry_missing(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.root, args.run)
    status = run_status(run_dir)
    print(json.dumps({"run_dir": str(run_dir), "run_status": status}, indent=2, sort_keys=True))
    if not status["missing_shards"]:
        return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nispo-benchmark")
    parser.add_argument("--root", type=Path, default=default_root())
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-set")
    p.add_argument("--set", dest="set_id", required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--shards", type=int, default=0, help="Explicit shard count override; default auto-resolves from --workers and --min-rows-per-shard.")
    p.add_argument("--workers", type=int, default=200, help="Maximum useful benchmark worker count when resolving shard count.")
    p.add_argument("--min-rows-per-shard", type=int, default=10000)
    p.add_argument("--seed", default="nispo-benchmark-v1")
    p.add_argument("--selection-reason", default="canonical full set")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=create_set)

    p = sub.add_parser("run")
    p.add_argument("--set", required=True)
    p.add_argument("--mode", choices=["full"], default="full")
    p.add_argument("--from-run", default="")
    p.add_argument("--submit-slurm", action="store_true")
    p.add_argument("--workers", type=int, default=200, help="Maximum concurrent Slurm array tasks.")
    p.add_argument("--max-workers", type=int, default=int(os.environ.get("NISPO_BENCH_MAX_WORKERS", DEFAULT_MAX_SLURM_WORKERS)), help="Safety cap for requested concurrent Slurm array tasks.")
    p.add_argument("--max-array-tasks", type=int, default=int(os.environ.get("NISPO_BENCH_MAX_ARRAY_TASKS", DEFAULT_MAX_SLURM_ARRAY_TASKS)), help="Safety cap for total submitted Slurm array tasks.")
    p.add_argument("--partition", default="standard-cpu")
    p.add_argument("--time", default="03:00:00")
    p.add_argument("--mem", default="8G")
    p.add_argument("--exclude", default=os.environ.get("NISPO_BENCH_EXCLUDE", ""))
    p.add_argument("--conda-env", default=os.environ.get("NISPO_CONDA_ENV_PREFIX", ""))
    p.add_argument("--opsin-jar", default=os.environ.get("NISPO_OPSIN_JAR", ""))
    p.add_argument("--nispo-repo", default=os.environ.get("NISPO_REPO", ""))
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--long-name-threshold", type=int, default=120)
    p.add_argument("--telemetry", choices=["aggregate"], default="aggregate")
    p.set_defaults(func=run_benchmark)

    p = sub.add_parser("_worker")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--task-index", type=int, required=True)
    p.add_argument("--opsin-jar", type=Path, required=True)
    p.add_argument("--nispo-repo", type=Path, default=None)
    p.add_argument("--nispo-commit", required=True)
    p.add_argument("--opsin-sha256", required=True)
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--long-name-threshold", type=int, default=120)
    p.add_argument("--work-dir", type=Path, required=True)
    p.set_defaults(func=run_worker)

    p = sub.add_parser("merge")
    p.add_argument("--run", required=True)
    p.set_defaults(func=lambda args: merge_run(argparse.Namespace(run_dir=resolve_run_dir(args.root, args.run))))

    p = sub.add_parser("inspect")
    p.add_argument("--run", default="latest")
    p.set_defaults(func=inspect_run)

    p = sub.add_parser("compare")
    p.add_argument("--run", required=True)
    p.add_argument("--baseline", required=True)
    p.set_defaults(func=compare)

    p = sub.add_parser("subset")
    p.add_argument("--from-run", required=True)
    p.add_argument("--mode", choices=["failures", "failure-type", "long-names"], required=True)
    p.add_argument("--status", default="")
    p.add_argument("--min-length", type=int, default=120)
    p.add_argument("--out-set", required=True)
    p.add_argument("--shards", type=int, default=0, help="Explicit shard count override; default auto-resolves from --workers and --min-rows-per-shard.")
    p.add_argument("--workers", type=int, default=200)
    p.add_argument("--min-rows-per-shard", type=int, default=10000)
    p.add_argument("--seed", default="nispo-benchmark-subset-v1")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=subset)

    p = sub.add_parser("retry-missing")
    p.add_argument("--run", required=True)
    p.set_defaults(func=retry_missing)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.root = args.root.resolve()
    reject_bitbucket(args.root)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
