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
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    atoms = _collect_subtree(mol, idx, parent_idx)
    key = _subtree_key(mol, idx, atoms)
    for template in _substituent_templates(Chem).get(key, ()):
        query = template["mol"]
        for match in mol.GetSubstructMatches(query, uniquify=False):
            if set(match) != atoms:
                continue
            if match[0] == idx:
                return template["name"]
    return ""


@lru_cache(maxsize=1)
def _substituent_templates(Chem) -> dict[tuple, tuple[dict, ...]]:
    grouped = defaultdict(list)
    for record in iter_opsin_token_records(SUBSTITUENT_RESOURCE_FILES):
        template = _template_from_token(Chem, record)
        if template:
            grouped[_template_key(template["mol"])].append(template)
    return {key: tuple(value) for key, value in grouped.items()}


def _template_from_token(Chem, record_or_token) -> dict | None:
    record = record_or_token if isinstance(record_or_token, OpsinTokenRecord) else None
    token = record.token if record is not None else record_or_token
    if unsupported_token_attributes(token, SUPPORTED_SUBSTITUENT_ATTRIBUTES):
        return None
    value = token.attrib.get("value", "")
    if "." in value:
        return None
    if not value.startswith("-"):
        return None
    if "." in value:
        return None
    smiles = value[1:]
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
    mol = apply_token_operations(Chem, mol, labels, token)
    if mol is None or mol.GetNumAtoms() > 24:
        return None
    return {
        "name": name,
        "mol": mol,
        "source_family": _record_family(record),
        "source_file": record.file_name if record is not None else "",
        "source_subtype": record.context.sub_type if record is not None else "",
    }


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
