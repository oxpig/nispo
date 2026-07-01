#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import csv
import json
import os
import pstats
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NISPO_SOURCE = PROJECT_ROOT / "nispo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile NISPO-only naming on a benchmark-set slice.")
    parser.add_argument("set_dir", type=Path, help="Benchmark set directory containing smiles.txt")
    parser.add_argument("--offset", type=int, default=0, help="0-based first row to profile after warmup")
    parser.add_argument("--limit", type=int, default=600, help="Profiled molecule count")
    parser.add_argument("--warmup-offset", type=int, default=0, help="0-based first warmup row")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup molecule count")
    parser.add_argument("--sort", default="cumtime", help="pstats sort key")
    parser.add_argument("--top", type=int, default=40, help="Number of pstats rows to print")
    parser.add_argument("--no-profile", action="store_true", help="Time the slice without cProfile overhead.")
    parser.add_argument("--profile-out", type=Path, default=None, help="Optional cProfile output path")
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Collect full internal NISPO telemetry for the profiled slice.",
    )
    parser.add_argument(
        "--compare-run",
        type=Path,
        default=None,
        help="Optional benchmark run directory with generated_names.tsv for exact-name comparison.",
    )
    return parser.parse_args()


def load_smiles(set_dir: Path) -> list[str]:
    smiles_file = set_dir / "smiles.txt"
    if not smiles_file.exists():
        raise SystemExit(f"missing benchmark input: {smiles_file}")
    return [line.strip() for line in smiles_file.read_text().splitlines() if line.strip()]


def baseline_names(run_dir: Path, start_row_1based: int, end_row_1based: int) -> dict[int, str]:
    names_file = run_dir / "generated_names.tsv"
    if not names_file.exists():
        raise SystemExit(f"missing generated names: {names_file}")
    names: dict[int, str] = {}
    with names_file.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            row_index = int(row["row_index"])
            if start_row_1based <= row_index <= end_row_1based:
                names[row_index] = row.get("generated_name", "")
    return names


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(NISPO_SOURCE))

    if args.telemetry:
        os.environ["NISPO_BENCH_FULL_TELEMETRY"] = "1"
        from nispo.api import _smiles_to_iupac_batch_with_telemetry
        from nispo.telemetry import compact_telemetry_summary

        name_batch = _smiles_to_iupac_batch_with_telemetry
    else:
        from nispo import smiles_to_iupac_batch

        name_batch = smiles_to_iupac_batch
        compact_telemetry_summary = None

    smiles = load_smiles(args.set_dir)
    warmup_smiles = smiles[args.warmup_offset : args.warmup_offset + args.warmup]
    profile_smiles = smiles[args.offset : args.offset + args.limit]

    if warmup_smiles:
        name_batch(warmup_smiles)

    started = time.perf_counter()
    profiler = None if args.no_profile else cProfile.Profile()
    if profiler is not None:
        profiler.enable()
    try:
        if args.telemetry:
            outputs, telemetry = name_batch(profile_smiles)
        else:
            outputs = name_batch(profile_smiles)
            telemetry = None
    finally:
        if profiler is not None:
            profiler.disable()
    seconds = time.perf_counter() - started

    start_row = args.offset + 1
    end_row = args.offset + len(profile_smiles)
    comparison = None
    if args.compare_run is not None:
        expected = baseline_names(args.compare_run, start_row, end_row)
        diffs = [
            {
                "row_index": row_index,
                "expected": expected.get(row_index, ""),
                "actual": actual,
            }
            for row_index, actual in zip(range(start_row, end_row + 1), outputs, strict=True)
            if expected.get(row_index, "") != actual
        ]
        comparison = {
            "run": str(args.compare_run),
            "diff_count": len(diffs),
            "first_diffs": diffs[:10],
        }

    summary = {
        "set_dir": str(args.set_dir),
        "offset": args.offset,
        "start_row_1based": start_row,
        "limit": len(profile_smiles),
        "warmup_offset": args.warmup_offset,
        "warmup": len(warmup_smiles),
        "seconds": round(seconds, 6),
        "milliseconds_per_molecule": round(seconds * 1000 / len(profile_smiles), 6)
        if profile_smiles
        else 0.0,
        "nonempty": sum(bool(output) for output in outputs),
        "comparison": comparison,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if telemetry is not None and compact_telemetry_summary is not None:
        print(json.dumps({"telemetry": compact_telemetry_summary(telemetry, "<memory>")}, indent=2, sort_keys=True))

    if profiler is None:
        return 0
    if args.profile_out is not None:
        args.profile_out.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.profile_out)
    pstats.Stats(profiler, stream=sys.stdout).strip_dirs().sort_stats(args.sort).print_stats(args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
