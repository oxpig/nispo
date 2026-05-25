#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NISPO_SOURCE = PROJECT_ROOT
BENCHMARK_DIR = Path(os.environ.get("NISPO_BENCHMARK_DIR", PROJECT_ROOT / "benchmark"))
LATEST_RESULTS = BENCHMARK_DIR / "latest" / "results.tsv"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify benchmark failures by broad OPSIN/IUPAC scaffold rule family."
    )
    parser.add_argument("--results", type=Path, default=LATEST_RESULTS)
    parser.add_argument("--statuses", default="empty_name,mismatch,opsin_parse_failed")
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if str(NISPO_SOURCE) not in sys.path:
        sys.path.insert(0, str(NISPO_SOURCE))

    from rdkit import Chem, RDLogger
    from nispo.parent_rules import parent_rules

    RDLogger.DisableLog("rdApp.*")
    statuses = {item.strip() for item in args.statuses.split(",") if item.strip()}
    rules = parent_rules(Chem)

    groups: Counter[tuple[str, str, str]] = Counter()
    candidate_groups: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    with args.results.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            status = row.get("status", "")
            if statuses and status not in statuses:
                continue
            mol = Chem.MolFromSmiles(row.get("input_smiles", ""))
            if mol is None:
                continue
            profile = _profile_molecule(mol)
            family = _family_hint(profile)
            rule_state, by_rule = _candidate_state(mol, rules)
            key = (status, family, rule_state)
            groups[key] += 1
            for rule_family, count in by_rule.items():
                candidate_groups[(family, rule_family)] += count
            if len(examples[key]) < args.examples:
                examples[key].append(
                    {
                        "row_index": int(row["row_index"]),
                        "smiles": row["input_smiles"],
                        "nispo_iupac": row.get("nispo_iupac", ""),
                        "ring_count": profile["ring_count"],
                        "ring_system_count": profile["ring_system_count"],
                        "ring_atom_count": profile["ring_atom_count"],
                        "ring_hetero_count": profile["ring_hetero_count"],
                        "aromatic_ring_atom_count": profile["aromatic_ring_atom_count"],
                    }
                )

    report = {
        "results": str(args.results),
        "groups": [
            {
                "count": count,
                "status": status,
                "family_hint": family,
                "candidate_state": rule_state,
                "examples": examples[(status, family, rule_state)],
            }
            for (status, family, rule_state), count in groups.most_common()
        ],
        "candidate_counts_by_gap_family": [
            {"family_hint": family, "rule_family": rule_family, "candidate_count": count}
            for (family, rule_family), count in candidate_groups.most_common()
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _candidate_state(mol, rules) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    for rule in rules:
        try:
            candidates = tuple(rule.candidates(mol) or ())
        except Exception:
            continue
        for candidate in candidates:
            if candidate.name:
                counts[candidate.family] += 1
    if not counts:
        return "no-nispo-candidate", counts
    return "has-nispo-candidate", counts


def _profile_molecule(mol) -> dict:
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    ring_atoms = set().union(*rings) if rings else set()
    fused_edges = 0
    spiro_atoms = set()
    for i, ring_i in enumerate(rings):
        for ring_j in rings[i + 1 :]:
            shared = ring_i & ring_j
            if len(shared) >= 2:
                fused_edges += 1
            elif len(shared) == 1:
                spiro_atoms |= shared
    ring_systems = _ring_systems(rings)
    ring_bond_degree = Counter()
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in ring_atoms and end in ring_atoms and bond.IsInRing():
            ring_bond_degree[begin] += 1
            ring_bond_degree[end] += 1
    bridgehead_like = sum(1 for atom_idx, degree in ring_bond_degree.items() if degree >= 3)
    return {
        "ring_count": len(rings),
        "ring_system_count": len(ring_systems),
        "ring_atom_count": len(ring_atoms),
        "ring_hetero_count": sum(
            1
            for atom_idx in ring_atoms
            if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() not in (1, 6)
        ),
        "aromatic_ring_atom_count": sum(
            1 for atom_idx in ring_atoms if mol.GetAtomWithIdx(atom_idx).GetIsAromatic()
        ),
        "fused_edge_count": fused_edges,
        "spiro_atom_count": len(spiro_atoms),
        "bridgehead_like_count": bridgehead_like,
    }


def _ring_systems(rings: list[set[int]]) -> list[set[int]]:
    systems: list[set[int]] = []
    for ring in rings:
        merged = set(ring)
        rest = []
        for system in systems:
            if merged & system:
                merged |= system
            else:
                rest.append(system)
        rest.append(merged)
        systems = rest
    changed = True
    while changed:
        changed = False
        merged_systems: list[set[int]] = []
        for system in systems:
            for other in merged_systems:
                if system & other:
                    other |= system
                    changed = True
                    break
            else:
                merged_systems.append(set(system))
        systems = merged_systems
    return systems


def _family_hint(profile: dict) -> str:
    if profile["ring_count"] == 0:
        return "acyclic replacement/suffix"
    if profile["ring_system_count"] > 1:
        return "ring assembly or cyclic substituent"
    if profile["spiro_atom_count"] > 0:
        return "spiro/polyspiro"
    if profile["fused_edge_count"] > 0:
        if profile["bridgehead_like_count"] > profile["fused_edge_count"] + 1:
            return "bridged fused/von Baeyer"
        if profile["aromatic_ring_atom_count"] == profile["ring_atom_count"]:
            return "fully unsaturated fused retained/HW"
        return "hydro or partially saturated fused retained"
    if profile["ring_count"] > 1:
        return "bridged/von Baeyer"
    if profile["ring_hetero_count"]:
        return "single-ring Hantzsch-Widman/retained"
    return "single-ring carbocycle"


if __name__ == "__main__":
    raise SystemExit(main())
