#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NISPO_SOURCE = PROJECT_ROOT
OPSIN_JAR = Path(os.environ.get("OPSIN_JAR", PROJECT_ROOT / "opsin-cli-2.9.0-jar-with-dependencies.jar"))
OPSIN_TOKENS = NISPO_SOURCE / "nispo" / "resources" / "opsin_tokens"
GENERATED_TEMPLATES = NISPO_SOURCE / "nispo" / "resources" / "generated_fused_templates.json"
EXTRA_GENERATOR_BASES = ("oxazole", "thiazole")


@dataclass(frozen=True)
class FusionComponent:
    alias: str
    atom_count: int
    locant_pairs: tuple[tuple[str, str], ...]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate OPSIN-validated fused parent templates for NISPO."
    )
    parser.add_argument("--output", type=Path, default=GENERATED_TEMPLATES)
    parser.add_argument("--max-bases", type=int, default=220)
    parser.add_argument("--max-base-rings", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=180000)
    parser.add_argument("--max-component-atoms", type=int, default=8)
    parser.add_argument("--letters", default="abcdefghijklmno")
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
    existing_by_name = {record["name"]: record for record in existing}
    existing_keys = {
        (record["name"], _canonical_smiles(Chem, record.get("smiles", "")))
        for record in existing
        if record.get("name") and record.get("smiles")
    }

    base_templates = (
        build_opsin_parent_templates(Chem)
        + build_retained_parent_templates(Chem, min_rings=1, max_rings=1, max_atoms=14)
    )
    bases = _candidate_bases(Chem, base_templates, args.max_bases, args.max_base_rings)
    components = _fusion_components(Chem, args.max_component_atoms)
    names = _candidate_names(components, bases, args.letters, args.max_candidates)

    parsed = _run_opsin(names)
    accepted_candidates = []
    seen_keys = set(existing_keys)
    for name, smiles in zip(names, parsed, strict=True):
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() > 36:
            continue
        if mol.GetRingInfo().NumRings() < 2 or max_ring_system_size(mol) < 8:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        key = (name, canonical)
        if key in seen_keys or name in existing_by_name:
            continue
        ordered_smiles = Chem.MolToSmiles(mol, canonical=False)
        accepted_candidates.append((name, canonical, ordered_smiles, mol))
        seen_keys.add(key)

    locants_by_name = _derive_locants_batch(Chem, [(name, mol) for name, _, _, mol in accepted_candidates])
    accepted = []
    for name, _canonical, ordered_smiles, mol in accepted_candidates:
        locants = locants_by_name.get(name, {})
        if len(locants) < mol.GetNumAtoms() // 2:
            continue
        accepted.append({"name": name, "smiles": ordered_smiles, "locants": locants})

    records = sorted(existing + accepted, key=lambda item: (item["name"], item["smiles"]))
    args.output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "bases": len(bases),
                "components": len(components),
                "candidates": len(names),
                "accepted_new": len(accepted),
                "records": len(records),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def _candidate_bases(Chem, templates: tuple[dict, ...], limit: int, max_base_rings: int) -> list[str]:
    scored = []
    seen = set()
    for template in templates:
        name = template.get("name", "")
        mol = template.get("mol")
        if not name or mol is None:
            continue
        if not name.isascii() or any(ch in name for ch in "() "):
            continue
        if "," in name and not ("[" in name and "]" in name):
            continue
        if any(term in name for term in ("hydro", "spiro", "bicyclo", "quinone", "one")):
            continue
        ring_count = mol.GetRingInfo().NumRings()
        atom_count = mol.GetNumAtoms()
        if ring_count < 1 or ring_count > max_base_rings or atom_count > 14:
            continue
        if any(atom.GetSymbol() not in {"C", "N", "O", "S"} for atom in mol.GetAtoms()):
            continue
        if any(not atom.IsInRing() for atom in mol.GetAtoms()):
            continue
        key = (name, Chem.MolToSmiles(mol, canonical=True))
        if key in seen:
            continue
        seen.add(key)
        penalty = _preferred_base_penalty(name, ring_count)
        if penalty < 0:
            continue
        hetero = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() != "C")
        score = (penalty, ring_count, atom_count, -hetero, name)
        scored.append((score, name))
    scored.sort(key=lambda item: item[0])
    names = [name for _, name in scored[:limit]]
    for extra in EXTRA_GENERATOR_BASES:
        if extra not in names:
            _insert_preferred_base(names, extra)
    return names[:limit]


def _insert_preferred_base(names: list[str], name: str) -> None:
    penalty = _preferred_base_penalty(name, 1)
    insert_at = len(names)
    for idx, existing in enumerate(names):
        existing_penalty = _preferred_base_penalty(existing, 2 if "[" in existing else 1)
        if existing_penalty >= 0 and existing_penalty > penalty:
            insert_at = idx
            break
    names.insert(insert_at, name)


def _preferred_base_penalty(name: str, ring_count: int) -> int:
    retained_names = (
        "benzene",
        "pyridine",
        "pyrimidine",
        "pyrazine",
        "pyridazine",
        "s-triazine",
        "as-triazine",
        "v-triazine",
        "furan",
        "thiophene",
        "pyrrole",
        "pyrazole",
        "imidazole",
        "triazole",
        "tetrazole",
        "oxazole",
        "isoxazole",
        "thiazole",
        "isothiazole",
        "indole",
        "indazole",
        "benzofuran",
        "benzothiophene",
        "quinoline",
        "isoquinoline",
        "naphthyridine",
        "quinazoline",
        "quinoxaline",
        "cinnoline",
        "phthalazine",
        "purine",
        "pteridine",
    )
    try:
        return retained_names.index(name)
    except ValueError:
        if "[" in name and "]" in name and ring_count >= 2:
            return len(retained_names) + _fusion_base_penalty(name)
        return -1


def _fusion_base_penalty(name: str) -> int:
    # Prefer compact, lower-order fusion descriptors as recursive bases. This
    # mirrors the runtime generated-template preference while keeping the
    # generator independent of benchmark outcomes.
    descriptor_count = name.count("[")
    expanded_penalty = sum(name.count(suffix) for suffix in ("ano[", "eno[", "ino[", "olo["))
    return descriptor_count * 100 + expanded_penalty * 10 + len(name)


def _fusion_components(Chem, max_component_atoms: int = 8) -> list[FusionComponent]:
    from nispo.resource_operations import apply_token_operations

    path = OPSIN_TOKENS / "fusionComponents.xml"
    root = ET.parse(path).getroot()
    components = []
    seen = set()
    for token in root.iter("token"):
        value = token.attrib.get("value", "")
        labels = token.attrib.get("labels", "")
        if not value or not labels or "." in value:
            continue
        mol = Chem.MolFromSmiles(value)
        label_list = [_primary_label(item) for item in labels.split("/")]
        if mol is None or len(label_list) != mol.GetNumAtoms():
            continue
        mol = apply_token_operations(Chem, mol, tuple(label_list), token)
        if mol is None or mol.GetNumAtoms() > max_component_atoms:
            continue
        if any(atom.GetSymbol() not in {"C", "N", "O", "S"} for atom in mol.GetAtoms()):
            continue
        if len(label_list) != mol.GetNumAtoms():
            continue
        locant_pairs = _fusion_locant_pairs(mol, label_list)
        if not locant_pairs:
            continue
        for alias in (part.strip() for part in (token.text or "").split("|")):
            if (
                not alias
                or not alias.isascii()
                or not alias.endswith("o")
                or alias.startswith("(")
                or any(ch in alias for ch in "[](), ")
            ):
                continue
            key = (alias, locant_pairs)
            if key in seen:
                continue
            seen.add(key)
            components.append(FusionComponent(alias, mol.GetNumAtoms(), locant_pairs))
    components.sort(key=lambda item: (item.atom_count, item.alias))
    return components


def _primary_label(label: str) -> str:
    return label.split(",", 1)[0].strip()


def _fusion_locant_pairs(mol, labels: list[str]) -> tuple[tuple[str, str], ...]:
    pairs = set()
    for bond in mol.GetBonds():
        a = labels[bond.GetBeginAtomIdx()]
        b = labels[bond.GetEndAtomIdx()]
        if not a.isdigit() or not b.isdigit():
            continue
        pairs.add((a, b))
        pairs.add((b, a))
    return tuple(sorted(pairs, key=lambda item: (int(item[0]), int(item[1]))))


def _candidate_names(
    components: list[FusionComponent], bases: list[str], letters: str, max_candidates: int
) -> list[str]:
    names = []
    seen = set()
    for base in bases:
        for component in components:
            for loc1, loc2 in component.locant_pairs:
                for letter in letters:
                    name = f"{component.alias}[{loc1},{loc2}-{letter}]{base}"
                    if name in seen:
                        continue
                    seen.add(name)
                    names.append(name)
                    if len(names) >= max_candidates:
                        return names
    return names


def _run_opsin(names: list[str]) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="nispo_fused_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        names_file = temp_dir / "names.txt"
        out_file = temp_dir / "smiles.txt"
        names_file.write_text("\n".join(names) + "\n", encoding="utf-8")
        command = ["java", "-jar", str(OPSIN_JAR), "-o", "smi", str(names_file), str(out_file)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise SystemExit(completed.stderr or completed.stdout)
        return out_file.read_text(encoding="utf-8").splitlines()


def _derive_locants_batch(Chem, parents: list[tuple[str, object]]) -> dict[str, dict[str, str]]:
    probes = []
    for parent_name, parent_mol in parents:
        for locant in _locant_pool(parent_mol.GetNumAtoms()):
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
        query, old_to_query = mol_with_no_chlorine(Chem, mol)
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
            elif query_to_parent:
                continue
            if not query_to_parent and attached_idx in match:
                template_idx = match.index(attached_idx)
                by_name.setdefault(parent_name, {})[str(template_idx)] = locant
                break
    return by_name


def mol_with_no_chlorine(Chem, mol):
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


def _locant_pool(atom_count: int) -> list[str]:
    values = [str(i) for i in range(1, atom_count + 1)]
    values.extend(f"{i}a" for i in range(1, atom_count + 1))
    values.extend(f"{i}b" for i in range(1, atom_count + 1))
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
