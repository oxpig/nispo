#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NISPO_SOURCE = PROJECT_ROOT
OPSIN_JAR = Path(os.environ.get("OPSIN_JAR", PROJECT_ROOT / "opsin-cli-2.9.0-jar-with-dependencies.jar"))
GENERATED_TEMPLATES = NISPO_SOURCE / "nispo" / "resources" / "generated_fused_templates.json"


@dataclass(frozen=True)
class SpiroRing:
    name: str
    locant: str


SPIRO_RINGS = (
    SpiroRing("cyclopropane", "1"),
    SpiroRing("cyclobutane", "1"),
    SpiroRing("cyclopentane", "1"),
    SpiroRing("cyclohexane", "1"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OPSIN-validated spiro parent templates for NISPO.")
    parser.add_argument("--output", type=Path, default=GENERATED_TEMPLATES)
    parser.add_argument("--max-parents", type=int, default=900)
    parser.add_argument("--max-candidates", type=int, default=220000)
    parser.add_argument("--include-existing", action="store_true")
    args = parser.parse_args()

    if str(NISPO_SOURCE) not in sys.path:
        sys.path.insert(0, str(NISPO_SOURCE))

    from rdkit import Chem, RDLogger
    from nispo.resource_templates import build_opsin_parent_templates, build_retained_parent_templates
    from nispo.template_index import max_ring_system_size

    RDLogger.DisableLog("rdApp.*")

    existing = _load_existing(args.output) if args.include_existing else []
    existing = [record for record in existing if _record_allowed(Chem, record)]
    seen_keys = {
        (record["name"], _canonical_smiles(Chem, record.get("smiles", "")))
        for record in existing
        if record.get("name") and record.get("smiles")
    }

    parent_templates = (
        build_opsin_parent_templates(Chem)
        + build_retained_parent_templates(Chem, min_rings=1, max_atoms=18)
    )
    parents = _candidate_parents(Chem, parent_templates, args.max_parents)
    names = _candidate_names(parents, args.max_candidates)

    parsed = _run_opsin(names)
    accepted_candidates = []
    seen = set(seen_keys)
    for name, smiles in zip(names, parsed, strict=True):
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() > 36:
            continue
        if mol.GetRingInfo().NumRings() < 2 or max_ring_system_size(mol) < 6:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        key = (name, canonical)
        if key in seen:
            continue
        accepted_candidates.append((name, Chem.MolToSmiles(mol, canonical=False), mol))
        seen.add(key)

    locants_by_name = _derive_locants_batch(Chem, [(name, mol) for name, _, mol in accepted_candidates])
    accepted = []
    for name, ordered_smiles, mol in accepted_candidates:
        locants = locants_by_name.get(name, {})
        if len(locants) < max(2, mol.GetNumAtoms() // 3):
            continue
        accepted.append({"name": name, "smiles": ordered_smiles, "locants": locants})

    records = sorted(existing + accepted, key=lambda item: (item["name"], item["smiles"]))
    args.output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "parents": len(parents),
                "candidates": len(names),
                "accepted_new": len(accepted),
                "records": len(records),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def _candidate_parents(Chem, templates: tuple[dict, ...], limit: int) -> list[tuple[str, tuple[str, ...], int]]:
    scored = []
    seen = set()
    for template in templates:
        name = template.get("name", "")
        mol = template.get("mol")
        if not name or mol is None:
            continue
        if not name.isascii() or any(ch in name for ch in "() "):
            continue
        if any(term in name for term in ("spiro", "bicyclo")):
            continue
        atom_count = mol.GetNumAtoms()
        ring_count = mol.GetRingInfo().NumRings()
        if ring_count < 1 or atom_count > 18:
            continue
        if any(atom.GetSymbol() not in {"C", "N", "O", "S"} for atom in mol.GetAtoms()):
            continue
        locants = tuple(
            loc
            for _idx, loc in sorted(template.get("locants", {}).items())
            if loc and (loc[0].isdigit() or loc[0].isalpha())
        )
        if not locants:
            continue
        key = (name, Chem.MolToSmiles(mol, canonical=True))
        if key in seen:
            continue
        seen.add(key)
        hetero = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() != "C")
        fused_penalty = name.count("[")
        scored.append(((fused_penalty, ring_count, atom_count, -hetero, name), name, locants, atom_count))
    scored.sort(key=lambda item: item[0])
    return [(name, locants, atom_count) for _score, name, locants, atom_count in scored[:limit]]


def _candidate_names(parents: list[tuple[str, tuple[str, ...], int]], max_candidates: int) -> list[str]:
    names = []
    seen = set()
    for parent_name, locants, _atom_count in parents:
        for ring in SPIRO_RINGS:
            for locant in locants:
                name = f"spiro[{ring.name}-{ring.locant},{locant}'-{parent_name}]"
                if name in seen:
                    continue
                seen.add(name)
                names.append(name)
                if len(names) >= max_candidates:
                    return names
    return names


def _run_opsin(names: list[str]) -> list[str]:
    if not names:
        return []
    with tempfile.TemporaryDirectory(prefix="nispo_spiro_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        names_file = temp_dir / "names.txt"
        out_file = temp_dir / "smiles.txt"
        names_file.write_text("\n".join(names) + "\n", encoding="utf-8")
        completed = subprocess.run(
            ["java", "-jar", str(OPSIN_JAR), "-o", "smi", str(names_file), str(out_file)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise SystemExit(completed.stderr or completed.stdout)
        return out_file.read_text(encoding="utf-8").splitlines()


def _derive_locants_batch(Chem, parents: list[tuple[str, object]]) -> dict[str, dict[str, str]]:
    probes = []
    for parent_name, parent_mol in parents:
        for locant in _spiro_locant_pool(parent_mol.GetNumAtoms()):
            probes.append((parent_name, parent_mol, locant, f"{locant}-chloro-{parent_name}"))

    parsed = _run_opsin([name for *_, name in probes])
    by_name: dict[str, dict[str, str]] = {}
    for (parent_name, parent_mol, locant, _), smiles in zip(probes, parsed, strict=True):
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        chloro_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "Cl"]
        if len(chloro_atoms) != 1:
            continue
        query, old_to_query = _mol_with_no_chlorine(Chem, mol)
        matches = parent_mol.GetSubstructMatches(query, uniquify=False)
        query_to_parent = True
        if not matches:
            matches = mol.GetSubstructMatches(parent_mol, uniquify=False)
            query_to_parent = False
        for match in matches:
            if len(match) != parent_mol.GetNumAtoms():
                continue
            chlorine = mol.GetAtomWithIdx(chloro_atoms[0])
            attached = [nbr.GetIdx() for nbr in chlorine.GetNeighbors()]
            if len(attached) != 1:
                continue
            attached_idx = attached[0]
            if query_to_parent and attached_idx in old_to_query:
                query_attached_idx = old_to_query[attached_idx]
                if query_attached_idx < len(match):
                    by_name.setdefault(parent_name, {})[str(match[query_attached_idx])] = locant
                    break
            elif not query_to_parent and attached_idx in match:
                template_idx = match.index(attached_idx)
                by_name.setdefault(parent_name, {})[str(template_idx)] = locant
                break
    return by_name


def _mol_with_no_chlorine(Chem, mol):
    editable = Chem.RWMol(mol)
    chlorine_indices = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "Cl"}
    old_to_new = {}
    offset = 0
    for idx in range(mol.GetNumAtoms()):
        if idx in chlorine_indices:
            offset += 1
        else:
            old_to_new[idx] = idx - offset
    for idx in sorted(chlorine_indices, reverse=True):
        editable.RemoveAtom(idx)
    return editable.GetMol(), old_to_new


def _spiro_locant_pool(atom_count: int) -> list[str]:
    values = [str(i) for i in range(1, atom_count + 1)]
    values.extend(f"{i}'" for i in range(1, atom_count + 1))
    values.extend(f"{i}a'" for i in range(1, atom_count + 1))
    values.extend(f"{i}b'" for i in range(1, atom_count + 1))
    return values


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _record_allowed(Chem, record: dict) -> bool:
    mol = Chem.MolFromSmiles(record.get("smiles", ""))
    if mol is None:
        return False
    return all(atom.GetSymbol() in {"C", "N", "O", "S"} for atom in mol.GetAtoms())


def _canonical_smiles(Chem, smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else ""


if __name__ == "__main__":
    raise SystemExit(main())
