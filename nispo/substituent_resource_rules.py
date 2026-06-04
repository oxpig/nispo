from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache

from .opsin_resource_utils import (
    OpsinTokenRecord,
    iter_opsin_token_records,
    preferred_token_name,
    primary_label,
    unsupported_token_attributes,
)
from .resource_manifest import SUBSTITUENT_FAMILY
from .resource_operations import apply_token_operations
from .template_access import template_record_mol, template_record_name


SUBSTITUENT_RESOURCE_FILES = (
    *SUBSTITUENT_FAMILY.files,
)

SUPPORTED_SUBSTITUENT_ATTRIBUTES = {
    "value",
    "labels",
    "outIDs",
    "usableAsAJoiner",
    "defaultInLocant",
    "defaultInID",
    "functionalIDs",
    "addGroup",
    "addBond",
    "addHeteroAtom",
    "frontLocantsExpected",
    "suffixAppliesTo",
    "suffixAppliesToByDefault",
    "acceptsAdditiveBonds",
    "iminoLike",
}


def resource_substituent_prefix(mol, idx: int, parent_idx: int, Chem) -> str:
    template = resource_substituent_match(mol, idx, parent_idx, Chem)
    return _preferred_resource_substituent_name(template_record_name(template)) if template else ""


def _preferred_resource_substituent_name(name: str) -> str:
    return {
        "benzyloxycarbonyl": "cbz",
        "carbbenzoxy": "cbz",
        "carbmethoxy": "methoxycarbonyl",
        "carbethoxy": "ethoxycarbonyl",
        "carboethoxy": "ethoxycarbonyl",
        "carbopropoxy": "propoxycarbonyl",
        "carbpropoxy": "propoxycarbonyl",
        "carbobutoxy": "butoxycarbonyl",
        "carbbutoxy": "butoxycarbonyl",
        "carbodecoxy": "decoxycarbonyl",
        "carbdecoxy": "decoxycarbonyl",
        "carbphenoxy": "phenoxycarbonyl",
        "carbophenoxy": "phenoxycarbonyl",
        "carbphenyloxy": "phenoxycarbonyl",
        "carbophenyloxy": "phenoxycarbonyl",
        "fluorenylmethoxycarbonyl": "fmoc",
        "tert-butoxycarbonyl": "boc",
        "tert-butyloxycarbonyl": "boc",
    }.get(name, name)


def resource_substituent_match(mol, idx: int, parent_idx: int, Chem) -> dict | None:
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    atoms = _collect_subtree(mol, idx, parent_idx)
    key = _subtree_key(mol, idx, atoms)
    for template in _substituent_templates(Chem).get(key, ()):
        query = template_record_mol(template)
        for match in mol.GetSubstructMatches(query, uniquify=False, useChirality=True):
            if set(match) != atoms:
                continue
            if template.get("unassigned_root_only") and mol.GetAtomWithIdx(idx).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                continue
            if _acyclic_carbalkoxy_token_on_ring(template, mol, match):
                continue
            if match[0] == idx:
                return template
    return None


def _acyclic_carbalkoxy_token_on_ring(template: dict, mol, match: tuple[int, ...]) -> bool:
    name = template_record_name(template)
    if name not in {"carbmethoxy", "carbethoxy", "carbopropoxy", "carbobutoxy", "carbphenoxy"}:
        return False
    return any(mol.GetAtomWithIdx(atom_idx).IsInRing() for atom_idx in match)


@lru_cache(maxsize=1)
def _substituent_templates(Chem) -> dict[tuple, tuple[dict, ...]]:
    grouped = defaultdict(list)
    for template in _manual_substituent_templates(Chem):
        grouped[_template_key(template_record_mol(template))].append(template)
    for record in iter_opsin_token_records(SUBSTITUENT_RESOURCE_FILES):
        template = _template_from_token(Chem, record)
        if template:
            grouped[_template_key(template_record_mol(template))].append(template)
    return {key: tuple(value) for key, value in grouped.items()}


_MANUAL_CARBOHYDRATE_SUBSTITUENTS = (
    (
        "glucopyranosyl",
        "C1([C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)",
        0,
        True,
    ),
    (
        "alpha-D-glucopyranosyl",
        "[C@H]1([C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)",
        0,
        False,
    ),
    (
        "beta-D-glucopyranosyl",
        "[C@@H]1([C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)",
        0,
        False,
    ),
    (
        "galactopyranosyl",
        "C1([C@H](O)[C@@H](O)[C@@H](O)[C@H](O1)CO)",
        0,
        True,
    ),
    (
        "alpha-D-galactopyranosyl",
        "[C@H]1([C@H](O)[C@@H](O)[C@@H](O)[C@H](O1)CO)",
        0,
        False,
    ),
    (
        "beta-D-galactopyranosyl",
        "[C@@H]1([C@H](O)[C@@H](O)[C@@H](O)[C@H](O1)CO)",
        0,
        False,
    ),
    (
        "mannopyranosyl",
        "C1([C@@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)",
        0,
        True,
    ),
    (
        "alpha-D-mannopyranosyl",
        "[C@H]1([C@@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)",
        0,
        False,
    ),
    (
        "beta-D-mannopyranosyl",
        "[C@@H]1([C@@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)",
        0,
        False,
    ),
    (
        "fucopyranosyl",
        "C1([C@@H](O)[C@H](O)[C@H](O)[C@@H](O1)C)",
        0,
        True,
    ),
    (
        "alpha-L-fucopyranosyl",
        "[C@@H]1([C@@H](O)[C@H](O)[C@H](O)[C@@H](O1)C)",
        0,
        False,
    ),
    (
        "beta-L-fucopyranosyl",
        "[C@H]1([C@@H](O)[C@H](O)[C@H](O)[C@@H](O1)C)",
        0,
        False,
    ),
    (
        "ribofuranosyl",
        "C1([C@H](O)[C@H](O)[C@H](O1)CO)",
        0,
        True,
    ),
    (
        "alpha-D-ribofuranosyl",
        "[C@H]1([C@H](O)[C@H](O)[C@H](O1)CO)",
        0,
        False,
    ),
    (
        "beta-D-ribofuranosyl",
        "[C@@H]1([C@H](O)[C@H](O)[C@H](O1)CO)",
        0,
        False,
    ),
    (
        "fructofuranosyl",
        "OCC1([C@@H](O)[C@H](O)[C@H](O1)CO)",
        2,
        True,
    ),
    (
        "alpha-D-fructofuranosyl",
        "OC[C@]1([C@@H](O)[C@H](O)[C@H](O1)CO)",
        2,
        False,
    ),
    (
        "beta-D-fructofuranosyl",
        "OC[C@@]1([C@@H](O)[C@H](O)[C@H](O1)CO)",
        2,
        False,
    ),
    (
        "apiofuranosyl",
        "C1([C@H](O)C(CO1)(O)CO)",
        0,
        True,
    ),
    (
        "alpha-D-apiofuranosyl",
        "[C@H]1([C@H](O)C(CO1)(O)CO)",
        0,
        False,
    ),
    (
        "beta-D-apiofuranosyl",
        "[C@@H]1([C@H](O)C(CO1)(O)CO)",
        0,
        False,
    ),
)


def _manual_substituent_templates(Chem) -> tuple[dict, ...]:
    templates = []
    for name, smiles, root_idx, unassigned_root_only in _MANUAL_CARBOHYDRATE_SUBSTITUENTS:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        if root_idx:
            order = (root_idx,) + tuple(idx for idx in range(mol.GetNumAtoms()) if idx != root_idx)
            mol = Chem.RenumberAtoms(mol, list(order))
        templates.append(
            {
                "name": name,
                "mol": mol,
                "source_family": "manual-carbohydrate-substituent",
                "source_file": "carbohydrates.xml",
                "source_subtype": "glycosyl-substituent",
                "unassigned_root_only": unassigned_root_only,
            }
        )
    return tuple(templates)


def _template_from_token(Chem, record_or_token) -> dict | None:
    record = record_or_token if isinstance(record_or_token, OpsinTokenRecord) else None
    token = record.token if record is not None else record_or_token
    if unsupported_token_attributes(token, SUPPORTED_SUBSTITUENT_ATTRIBUTES):
        return None
    value = token.attrib.get("value", "")
    if "." in value:
        return None
    out_ids = tuple(part.strip() for part in token.attrib.get("outIDs", "").split(",") if part.strip())
    if not value.startswith("-") and len(out_ids) != 1:
        return None
    smiles = value[1:] if value.startswith("-") else value
    if not smiles:
        return None
    name = preferred_token_name(token.text or "")
    if not name:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    labels_text = token.attrib.get("labels", "")
    labels = tuple(primary_label(label) for label in labels_text.split("/")) if labels_text else tuple(
        str(idx + 1) for idx in range(mol.GetNumAtoms())
    )
    if len(labels) != mol.GetNumAtoms():
        return None
    root_idx = _template_root_idx(value, labels, out_ids)
    if root_idx is None:
        return None
    mol = apply_token_operations(Chem, mol, labels, token)
    if mol is None or mol.GetNumAtoms() > 24:
        return None
    if root_idx:
        order = (root_idx,) + tuple(idx for idx in range(mol.GetNumAtoms()) if idx != root_idx)
        mol = Chem.RenumberAtoms(mol, list(order))
    return {
        "name": name,
        "mol": mol,
        "source_family": _record_family(record),
        "source_file": record.file_name if record is not None else "",
        "source_subtype": record.context.sub_type if record is not None else "",
    }


def _template_root_idx(value: str, labels: tuple[str, ...], out_ids: tuple[str, ...]) -> int | None:
    if value.startswith("-"):
        return 0
    out_id = out_ids[0] if out_ids else ""
    if out_id in labels:
        return labels.index(out_id)
    if out_id.isdigit():
        idx = int(out_id) - 1
        if 0 <= idx < len(labels):
            return idx
    return None


def _record_family(record: OpsinTokenRecord | None) -> str:
    if record is None:
        return "opsin-token"
    context = record.context
    parts = [record.file_name.removesuffix(".xml")]
    if context.type:
        parts.append(context.type)
    if context.sub_type:
        parts.append(context.sub_type)
    return ":".join(parts)


def _collect_subtree(mol, start_idx: int, blocked_idx: int) -> set[int]:
    seen = {blocked_idx}
    stack = [start_idx]
    atoms = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        atoms.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in seen:
                stack.append(nidx)
    return atoms


def _subtree_key(mol, root_idx: int, atoms: set[int]) -> tuple:
    return (
        mol.GetAtomWithIdx(root_idx).GetSymbol(),
        len(atoms),
        tuple(sorted(Counter(mol.GetAtomWithIdx(idx).GetSymbol() for idx in atoms).items())),
    )


def _template_key(mol) -> tuple:
    return (
        mol.GetAtomWithIdx(0).GetSymbol(),
        mol.GetNumAtoms(),
        tuple(sorted(Counter(atom.GetSymbol() for atom in mol.GetAtoms()).items())),
    )
