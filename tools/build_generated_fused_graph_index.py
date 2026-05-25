#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build graph-shape buckets for generated fused templates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--element-output", type=Path, default=DEFAULT_ELEMENT_OUTPUT)
    args = parser.parse_args()

    if str(NISPO_SOURCE) not in sys.path:
        sys.path.insert(0, str(NISPO_SOURCE))

    from rdkit import Chem, RDLogger
    from nispo.template_index import element_graph_signature_for_atom_set, graph_signature_for_atom_set

    RDLogger.DisableLog("rdApp.*")

    feature_buckets = json.loads(args.input.read_text(encoding="utf-8"))
    graph_buckets: dict[str, dict[str, list[dict]]] = {}
    element_graph_buckets: dict[str, dict[str, dict[str, list[dict]]]] = {}
    indexed_records = 0
    skipped_records = 0
    for feature_key, records in feature_buckets.items():
        grouped: dict[str, list[dict]] = {}
        for record in records:
            mol = Chem.MolFromSmiles(record.get("smiles", ""))
            if mol is None:
                skipped_records += 1
                continue
            signature = graph_signature_for_atom_set(mol, range(mol.GetNumAtoms()), Chem)
            element_signature = element_graph_signature_for_atom_set(mol, range(mol.GetNumAtoms()), Chem)
            grouped.setdefault(signature, []).append(record)
            element_graph_buckets.setdefault(feature_key, {}).setdefault(signature, {}).setdefault(
                element_signature, []
            ).append(record)
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
                "output": str(args.output),
                "skipped_records": skipped_records,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
