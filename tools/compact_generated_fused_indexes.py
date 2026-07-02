#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = REPO_ROOT / "nispo" / "resources"
DEFAULT_RECORD_TABLE = RESOURCE_ROOT / "generated_fused_template_index_records.json.gz"
DEFAULT_INDEXES = (
    RESOURCE_ROOT / "generated_fused_template_index.json",
    RESOURCE_ROOT / "generated_fused_template_graph_index.json",
    RESOURCE_ROOT / "generated_fused_template_element_graph_index.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replace embedded generated fused records with record IDs.")
    parser.add_argument("--record-table", type=Path, default=DEFAULT_RECORD_TABLE)
    parser.add_argument("--indexes", type=Path, nargs="*", default=DEFAULT_INDEXES)
    parser.add_argument("--check", action="store_true", help="Fail if an index still embeds records.")
    args = parser.parse_args()

    table = _read_json(args.record_table)
    id_by_key = {
        _record_key(record): index
        for index, record in enumerate(table)
        if isinstance(record, dict)
    }
    report = {}
    for path in args.indexes:
        raw = json.loads(path.read_text(encoding="utf-8"))
        compacted, stats = _compact(raw, id_by_key)
        report[str(path)] = stats
        if stats["unmapped_records"]:
            raise SystemExit(f"{path} contains {stats['unmapped_records']} records not present in table")
        if args.check:
            if stats["embedded_records_after"]:
                raise SystemExit(f"{path} still embeds {stats['embedded_records_after']} records")
            continue
        path.write_text(json.dumps(compacted, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _read_json(path: Path):
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _compact(value: Any, id_by_key: dict[tuple, int]) -> tuple[Any, dict[str, int]]:
    stats = {
        "embedded_records_before": 0,
        "embedded_records_after": 0,
        "integer_refs_before": 0,
        "integer_refs_after": 0,
        "unmapped_records": 0,
    }

    def visit(item):
        if isinstance(item, int):
            stats["integer_refs_before"] += 1
            stats["integer_refs_after"] += 1
            return item
        if isinstance(item, dict):
            if "name" in item and "smiles" in item:
                stats["embedded_records_before"] += 1
                record_id = id_by_key.get(_record_key(item))
                if record_id is None:
                    stats["embedded_records_after"] += 1
                    stats["unmapped_records"] += 1
                    return item
                stats["integer_refs_after"] += 1
                return record_id
            return {str(key): visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        return item

    return visit(value), stats


def _record_key(record: dict) -> tuple:
    return (
        str(record.get("name", "")),
        str(record.get("smiles", "")),
        tuple(sorted((record.get("locants") or {}).items())),
        tuple(str(flag) for flag in record.get("flags", ()) if str(flag)),
        str(record.get("source_file") or ""),
        str(record.get("source_subtype", "")),
        record.get("general_parent_safe", None),
    )


if __name__ == "__main__":
    raise SystemExit(main())
