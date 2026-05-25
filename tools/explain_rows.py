#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PACKAGE_REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_REPO.parent
OPSIN_JAR = PROJECT_ROOT / "benchmark" / "opsin-cli-2.9.0-jar-with-dependencies.jar"
JAVA = Path("/opt/homebrew/opt/openjdk/bin/java")


def main() -> int:
    args = parse_args()
    set_dir = args.set_dir.resolve()
    smiles = _load_smiles(set_dir)
    rows = [int(row) for row in args.rows.split(",") if row.strip()]
    if str(PACKAGE_REPO) not in sys.path:
        sys.path.insert(0, str(PACKAGE_REPO))

    from nispo import smiles_to_iupac

    details = []
    for row in rows:
        if row < 1 or row > len(smiles):
            details.append({"row_index": row, "error": "row out of range"})
            continue
        input_smiles = smiles[row - 1]
        name = smiles_to_iupac(input_smiles)
        parsed = _opsin_parse(name) if name else ""
        details.append(
            {
                "row_index": row,
                "input_smiles": input_smiles,
                "nispo_name": name,
                "opsin_smiles": parsed,
            }
        )
    print(json.dumps(details, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate selected-row NISPO/OPSIN diagnostics.")
    parser.add_argument("set_dir", type=Path)
    parser.add_argument("--rows", required=True, help="Comma-separated 1-based row indices")
    return parser.parse_args()


def _load_smiles(set_dir: Path) -> list[str]:
    smiles_file = set_dir / "smiles.txt"
    return [line.strip() for line in smiles_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _opsin_parse(name: str) -> str:
    if not name or not OPSIN_JAR.exists():
        return ""
    java = str(JAVA) if JAVA.exists() else "java"
    with tempfile.TemporaryDirectory(prefix="nispo_explain_") as tmp:
        temp_dir = Path(tmp)
        names_file = temp_dir / "name.txt"
        out_file = temp_dir / "opsin.smi"
        names_file.write_text(name + "\n", encoding="utf-8")
        completed = subprocess.run(
            [java, "-jar", str(OPSIN_JAR), "-o", "smi", str(names_file), str(out_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not out_file.exists():
            return ""
        lines = out_file.read_text(encoding="utf-8").splitlines()
        return lines[0] if lines else ""


if __name__ == "__main__":
    raise SystemExit(main())
