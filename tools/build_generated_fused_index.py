#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NISPO_SOURCE = PROJECT_ROOT
OPSIN_JAR = Path(os.environ.get("OPSIN_JAR", PROJECT_ROOT / "opsin-cli-2.9.0-jar-with-dependencies.jar"))
DEFAULT_INPUT = NISPO_SOURCE / "nispo" / "resources" / "generated_fused_template_index_records.json"
DEFAULT_OUTPUT = NISPO_SOURCE / "nispo" / "resources" / "generated_fused_template_index.json"
NS = {"c": "http://www.xml-cml.org/schema"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NISPO's compact generated fused-template index.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-skeleton", type=int, default=3)
    args = parser.parse_args()

    if str(NISPO_SOURCE) not in sys.path:
        sys.path.insert(0, str(NISPO_SOURCE))

    from rdkit import Chem, RDLogger
    from nispo.fused_template_families import generated_template_key
    from nispo.template_index import _ring_system_feature_key, molecule_features

    RDLogger.DisableLog("rdApp.*")

    records = json.loads(args.input.read_text(encoding="utf-8"))
    selected = _select_records(Chem, records, args.per_skeleton, molecule_features, _ring_system_feature_key, generated_template_key)
    _fill_cml_locants(Chem, selected, generated_template_key)
    buckets = _bucket_records(selected)
    for items in buckets.values():
        items.sort(key=lambda item: generated_template_key(item))
    args.output.write_text(json.dumps(buckets, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    full_locants = sum(
        1
        for items in buckets.values()
        for item in items
        if len(item.get("locants", {})) >= Chem.MolFromSmiles(item["smiles"]).GetNumAtoms()
    )
    print(
        json.dumps(
            {
                "buckets": len(buckets),
                "full_locants": full_locants,
                "indexed_records": sum(len(items) for items in buckets.values()),
                "output": str(args.output),
                "source_records": len(records),
            },
            sort_keys=True,
        )
    )
    return 0


def _select_records(Chem, records, per_skeleton, molecule_features, ring_system_feature_key, generated_template_key):
    grouped: dict[str, list[dict]] = {}
    seen = set()
    for record in records:
        name = record.get("name", "")
        mol = Chem.MolFromSmiles(record.get("smiles", ""))
        if not name or mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        key = (name, canonical)
        if key in seen:
            continue
        seen.add(key)
        features = molecule_features(mol, Chem)
        if not features["all_atoms_in_one_ring_system"]:
            continue
        grouped.setdefault(canonical, []).append(
            {
                "name": name,
                "smiles": record["smiles"],
                "locants": record.get("locants", {}),
                "feature_key": ring_system_feature_key(features),
            }
        )

    selected = []
    for items in grouped.values():
        selected.extend(sorted(items, key=generated_template_key)[:per_skeleton])
    return selected


def _fill_cml_locants(Chem, selected: list[dict], generated_template_key) -> None:
    selected.sort(key=generated_template_key)
    with tempfile.TemporaryDirectory(prefix="nispo_cml_locants_") as tmp:
        tmp_path = Path(tmp)
        names_file = tmp_path / "names.txt"
        cml_file = tmp_path / "out.cml"
        names_file.write_text("\n".join(item["name"] for item in selected) + "\n", encoding="utf-8")
        subprocess.run(
            ["java", "-jar", str(OPSIN_JAR), "-o", "cml", str(names_file), str(cml_file)],
            check=True,
            capture_output=True,
            text=True,
        )
        molecules = ET.parse(cml_file).getroot().findall(".//c:molecule", NS)
        for item, mol_el in zip(selected, molecules, strict=False):
            locants = _locants_from_cml_match(Chem, item["smiles"], mol_el)
            if len(locants) >= len(item.get("locants", {})):
                item["locants"] = locants


def _locants_from_cml_match(Chem, smiles: str, mol_el) -> dict[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    query = Chem.RWMol()
    id_to_idx = {}
    query_locs = {}
    for atom_el in mol_el.findall(".//c:atom", NS):
        elem = atom_el.attrib.get("elementType")
        if elem == "H":
            continue
        idx = query.AddAtom(Chem.Atom(elem))
        id_to_idx[atom_el.attrib["id"]] = idx
        labels = [label.attrib.get("value", "") for label in atom_el.findall("./c:label", NS)]
        locant = next((label for label in labels if label[:1].isdigit()), "")
        if locant:
            query_locs[idx] = locant
    for bond_el in mol_el.findall(".//c:bond", NS):
        refs = bond_el.attrib.get("atomRefs2", "").split()
        if len(refs) != 2:
            continue
        a, b = refs
        if a in id_to_idx and b in id_to_idx:
            query.AddBond(id_to_idx[a], id_to_idx[b], Chem.BondType.UNSPECIFIED)
    matches = mol.GetSubstructMatches(query.GetMol(), uniquify=False)
    if not matches:
        return {}
    match = matches[0]
    return {str(match[qidx]): locant for qidx, locant in query_locs.items() if qidx < len(match)}


def _bucket_records(selected: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for item in selected:
        feature_key = item.pop("feature_key")
        key = json.dumps(feature_key, separators=(",", ":"))
        buckets.setdefault(key, []).append(item)
    return buckets


if __name__ == "__main__":
    raise SystemExit(main())
