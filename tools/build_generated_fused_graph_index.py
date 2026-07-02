#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NISPO_SOURCE = PROJECT_ROOT
DEFAULT_INPUT = NISPO_SOURCE / "nispo" / "resources" / "generated_fused_template_index.json"
DEFAULT_OUTPUT = NISPO_SOURCE / "nispo" / "resources" / "generated_fused_template_graph_index.json"
DEFAULT_ELEMENT_OUTPUT = (
    NISPO_SOURCE / "nispo" / "resources" / "generated_fused_template_element_graph_index.json"
)
DEFAULT_RECORD_TABLE = NISPO_SOURCE / "nispo" / "resources" / "generated_fused_template_index_records.json.gz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build graph-shape buckets for generated fused templates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--element-output", type=Path, default=DEFAULT_ELEMENT_OUTPUT)
    parser.add_argument("--record-table", type=Path, default=DEFAULT_RECORD_TABLE)
    args = parser.parse_args()

    if str(NISPO_SOURCE) not in sys.path:
        sys.path.insert(0, str(NISPO_SOURCE))

    from rdkit import Chem, RDLogger
    from nispo.template_index import element_graph_signature_for_atom_set, graph_signature_for_atom_set

    RDLogger.DisableLog("rdApp.*")

    feature_buckets = _read_json(args.input)
    record_table = _read_json(args.record_table)
    record_ids = _record_id_map(record_table)
    graph_buckets: dict[str, dict[str, list[dict]]] = {}
    element_graph_buckets: dict[str, dict[str, dict[str, list[dict]]]] = {}
    indexed_records = 0
    skipped_records = 0
    compacted_refs = 0
    embedded_records = 0
    for feature_key, records in feature_buckets.items():
        grouped: dict[str, list[dict]] = {}
        for item in records:
            record = _resolve_record(item, record_table)
            if not isinstance(record, dict):
                skipped_records += 1
                continue
            mol = Chem.MolFromSmiles(record.get("smiles", ""))
            if mol is None:
                skipped_records += 1
                continue
            signature = graph_signature_for_atom_set(mol, range(mol.GetNumAtoms()), Chem)
            element_signature = element_graph_signature_for_atom_set(mol, range(mol.GetNumAtoms()), Chem)
            record_ref = item if isinstance(item, int) else record_ids.get(_record_key(record), record)
            if isinstance(record_ref, int):
                compacted_refs += 1
            else:
                embedded_records += 1
            grouped.setdefault(signature, []).append(record_ref)
            element_graph_buckets.setdefault(feature_key, {}).setdefault(signature, {}).setdefault(
                element_signature, []
            ).append(record_ref)
            indexed_records += 1
        if grouped:
            graph_buckets[feature_key] = grouped

    args.output.write_text(json.dumps(graph_buckets, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    args.element_output.write_text(
        json.dumps(element_graph_buckets, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "element_output": str(args.element_output),
                "feature_buckets": len(graph_buckets),
                "graph_buckets": sum(len(items) for items in graph_buckets.values()),
                "indexed_records": indexed_records,
                "compacted_refs": compacted_refs,
                "embedded_records": embedded_records,
                "output": str(args.output),
                "skipped_records": skipped_records,
            },
            sort_keys=True,
        )
    )
    return 0


def _read_json(path: Path):
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_record(item, record_table: list[dict]):
    if isinstance(item, int) and 0 <= item < len(record_table):
        return record_table[item]
    return item


def _record_id_map(records: list[dict]) -> dict[tuple, int]:
    return {
        _record_key(record): index
        for index, record in enumerate(records)
        if isinstance(record, dict)
    }


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
