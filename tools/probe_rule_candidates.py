#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NISPO_SOURCE = PROJECT_ROOT
BENCHMARK_DIR = Path(os.environ.get("NISPO_BENCHMARK_DIR", PROJECT_ROOT / "benchmark"))
OPSIN_JAR = BENCHMARK_DIR / "opsin-cli-2.9.0-jar-with-dependencies.jar"
LATEST_RESULTS = BENCHMARK_DIR / "latest" / "results.tsv"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-score all NISPO rule-family candidates for benchmark rows."
    )
    parser.add_argument("--results", type=Path, default=LATEST_RESULTS)
    parser.add_argument("--statuses", default="mismatch,opsin_parse_failed")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if str(NISPO_SOURCE) not in sys.path:
        sys.path.insert(0, str(NISPO_SOURCE))

    from rdkit import Chem, RDLogger
    from nispo.naming_context import MoleculeContext
    from nispo.parent_rules import parent_rules

    RDLogger.DisableLog("rdApp.*")

    statuses = {item.strip() for item in args.statuses.split(",") if item.strip()}
    rows = _load_rows(args.results, statuses, args.limit)
    rules = parent_rules(Chem)

    probes = []
    candidate_counts = Counter()
    for row in rows:
        mol = Chem.MolFromSmiles(row["input_smiles"])
        if mol is None:
            continue
        ctx = MoleculeContext(mol, Chem)
        row["input_canonical"] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        seen_names = set()
        for rule in rules:
            try:
                candidates = list(rule.candidates(ctx) or ())
            except Exception:
                continue
            for candidate in candidates:
                if not candidate.name or candidate.name in seen_names:
                    continue
                seen_names.add(candidate.name)
                candidate_counts[candidate.family] += 1
                probes.append((row, candidate))

    parsed = _run_opsin([candidate.name for _, candidate in probes])
    summaries = []
    wins_by_family = Counter()
    candidate_total_by_status = Counter()
    for (row, candidate), parsed_smiles in zip(probes, parsed, strict=True):
        candidate_total_by_status[row["status"]] += 1
        parsed_mol = Chem.MolFromSmiles(parsed_smiles) if parsed_smiles else None
        if parsed_mol is None:
            continue
        parsed_canonical = Chem.MolToSmiles(parsed_mol, canonical=True, isomericSmiles=True)
        if parsed_canonical != row["input_canonical"]:
            continue
        wins_by_family[candidate.family] += 1
        summaries.append(
            {
                "row_index": int(row["row_index"]),
                "status": row["status"],
                "family": candidate.family,
                "priority": candidate.priority,
                "sort_key": repr(candidate.sort_key),
                "name": candidate.name,
                "current_name": row["nispo_iupac"],
                "input_smiles": row["input_smiles"],
            }
        )

    summaries.sort(key=lambda item: (item["row_index"], item["priority"], item["family"], item["name"]))
    report = {
        "rows": len(rows),
        "probes": len(probes),
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "candidate_total_by_status": dict(sorted(candidate_total_by_status.items())),
        "wins": len({item["row_index"] for item in summaries}),
        "wins_by_family": dict(sorted(wins_by_family.items())),
        "matches": summaries,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _load_rows(path: Path, statuses: set[str], limit: int) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if statuses and row.get("status") not in statuses:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def _run_opsin(names: list[str]) -> list[str]:
    if not names:
        return []
    with tempfile.TemporaryDirectory(prefix="nispo_candidates_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        names_file = temp_dir / "names.txt"
        out_file = temp_dir / "smiles.txt"
        names_file.write_text("\n".join(names) + "\n", encoding="utf-8")
        command = ["java", "-jar", str(OPSIN_JAR), "-o", "smi", str(names_file), str(out_file)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise SystemExit(completed.stderr or completed.stdout)
        return out_file.read_text(encoding="utf-8").splitlines()


if __name__ == "__main__":
    raise SystemExit(main())
