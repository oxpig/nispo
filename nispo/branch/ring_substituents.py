from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from ..constants import ROOTS
from ..cycle_utils import ordered_ring, ring_system_extends_core
from ..formatting import format_prefixes as _format_prefixes
from ..fused_rules import FUSED_SUBSTITUENT_STEMS
from ..fused_template_families import (
    candidate_generated_fused_templates as _candidate_generated_fused_templates,
    curated_fused_templates as _curated_fused_templates,
    generated_fusion_preference as _generated_fusion_preference,
)
from ..molecule_cache import mol_cache_dict
from ..ring.hydrogen import _drop_indicated_hydrogen_prefix
from ..ring_resource_rules import resource_ring_parent_candidates as _resource_ring_parent_candidates
from ..ring_rules import (
    ring_parent_candidates as _ring_parent_candidates,
    ring_substituent_suffix as _ring_substituent_suffix,
)
from ..stereochemistry import _has_assigned_bond_stereo, assigned_atom_stereochemistry
from ..substituent_resource_rules import (
    resource_substituent_match as _resource_substituent_match,
    resource_substituent_prefix as _resource_substituent_prefix,
)
from ..template_access import template_record_locants, template_record_mol, template_record_name
from ..template_store import template_store
from .functional_groups import (
    _is_nitrile_carbon,
    _name_subtree_path,
    _sulfur_prefix,
    collect_subtree,
    substituent_from_root,
    substituent_locants,
)


def branch_prefix(mol, idx: int, parent_idx: int) -> str:
    from .. import branch_rules

    return branch_rules.branch_prefix(mol, idx, parent_idx)

def _ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    return _carbohydrate_ring_substituent_prefix(mol, attachment_idx, parent_idx) or _simple_ring_substituent_prefix(
        mol, attachment_idx, parent_idx
    ) or _fused_ring_substituent_prefix(
        mol, attachment_idx, parent_idx
    ) or _resource_substituent_prefix(mol, attachment_idx, parent_idx, Chem) or _generic_polycyclic_ring_substituent_prefix(
        mol, attachment_idx, parent_idx
    )


def _carbohydrate_ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    template = _resource_substituent_match(mol, attachment_idx, parent_idx, Chem)
    if not template or template.get("source_file") != "carbohydrates.xml":
        return ""
    name = template_record_name(template)
    if name.endswith(("pyranosyl", "furanosyl", "osyl", "aminyl", "muramyl", "neuraminyl")):
        return name
    return ""


def _generic_polycyclic_ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    from ..polycyclic_rules import polycyclic_substituent_prefix

    return polycyclic_substituent_prefix(mol, Chem, attachment_idx, parent_idx)


def _ring_alkyl_prefix(mol, idx: int, parent_idx: int) -> str:
    path = []
    previous = parent_idx
    current = idx
    ring_neighbor = None
    while len(path) < 8:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
            return ""
        bond = mol.GetBondBetweenAtoms(previous, current)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        path.append(current)
        ring_neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != previous and n.IsInRing()]
        carbon_neighbors = [
            n.GetIdx()
            for n in atom.GetNeighbors()
            if n.GetIdx() != previous and not n.IsInRing() and n.GetSymbol() == "C"
        ]
        other_neighbors = [
            n.GetIdx()
            for n in atom.GetNeighbors()
            if n.GetIdx() != previous and not n.IsInRing() and n.GetSymbol() != "C"
        ]
        if other_neighbors:
            return ""
        if ring_neighbors:
            if len(ring_neighbors) != 1 or carbon_neighbors:
                return ""
            ring_neighbor = ring_neighbors[0]
            break
        if len(carbon_neighbors) != 1:
            return ""
        previous, current = current, carbon_neighbors[0]
    if ring_neighbor is None:
        return ""
    ring = _ring_substituent_prefix(mol, ring_neighbor, path[-1])
    if not ring:
        return ""
    if len(path) != 1:
        root = ROOTS.get(len(path))
        if not root:
            return ""
        return f"{len(path)}-({ring}){root}yl"
    return ring + "methyl"


def _di_ring_methyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    ring_neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx and n.IsInRing()]
    other_heavy = [
        n.GetIdx()
        for n in atom.GetNeighbors()
        if n.GetIdx() != parent_idx and not n.IsInRing() and n.GetAtomicNum() > 1
    ]
    if len(ring_neighbors) != 2 or other_heavy:
        return ""
    names = []
    for ring_idx in ring_neighbors:
        name = _ring_substituent_prefix(mol, ring_idx, idx)
        if not name:
            return ""
        names.append(name)
    counts = {name: names.count(name) for name in set(names)}
    if len(counts) == 1:
        name = names[0]
        if name == "phenyl":
            return "diphenylmethyl"
        return f"bis({_methyl_ligand_unit(name)})methyl"
    ordered = sorted(names)
    first, second = ordered
    return _methyl_ligand_unit(first, first=True) + _methyl_ligand_unit(second) + "methyl"


def _methyl_ligand_unit(name: str, *, first: bool = False) -> str:
    if first and "-" not in name and not name[:1].isdigit() and not name.startswith("("):
        return name
    return f"({name})"


def _aryloxy_methyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    oxy = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            oxy.append(nidx)
        else:
            return ""
    if len(oxy) != 1:
        return ""
    oidx = oxy[0]
    ring_neighbors = [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(oidx).GetNeighbors()
        if n.GetIdx() != idx and n.IsInRing()
    ]
    if len(ring_neighbors) != 1:
        return _complex_oxymethyl_prefix(mol, oidx, idx)
    ring = _ring_substituent_prefix(mol, ring_neighbors[0], oidx)
    if not ring:
        return _complex_oxymethyl_prefix(mol, oidx, idx)
    if ring == "phenyl":
        return "phenoxymethyl"
    return f"({ring}oxy)methyl"


def _complex_oxymethyl_prefix(mol, oidx: int, carbon_idx: int) -> str:
    oxy_prefix = branch_prefix(mol, oidx, carbon_idx)
    if not oxy_prefix or not oxy_prefix.endswith("oxy"):
        return ""
    if "sulfonyloxy" not in oxy_prefix:
        return ""
    if "-" in oxy_prefix or oxy_prefix[:1].isdigit():
        return f"({oxy_prefix})methyl"
    return oxy_prefix + "methyl"


def _iminomethyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    double_n = []
    other = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_n.append(nidx)
        else:
            other.append(nidx)
    if len(double_n) != 1 or other:
        return ""
    nidx = double_n[0]
    distal = [n.GetIdx() for n in mol.GetAtomWithIdx(nidx).GetNeighbors() if n.GetIdx() != idx]
    if len(distal) != 1:
        return ""
    didx = distal[0]
    if mol.GetAtomWithIdx(didx).IsInRing():
        name = _ring_substituent_prefix(mol, didx, nidx)
    else:
        name = substituent_from_root(mol, didx, nidx)
    if not name:
        return ""
    return name + "iminomethyl"


def _difluoro_substituted_methyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    fluorines = []
    heavy = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if nbr.GetSymbol() == "F":
            fluorines.append(nidx)
        else:
            heavy.append(nidx)
    if len(fluorines) != 2 or len(heavy) != 1:
        return ""
    attached = heavy[0]
    if mol.GetAtomWithIdx(attached).IsInRing():
        name = _ring_substituent_prefix(mol, attached, idx)
    else:
        name = substituent_from_root(mol, attached, idx)
    if not name:
        return ""
    return f"({name})difluoromethyl"


def _sulfonamidomethyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    nitrogens = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
            nitrogens.append(nidx)
        else:
            return ""
    if len(nitrogens) != 1:
        return ""
    nidx = nitrogens[0]
    n_atom = mol.GetAtomWithIdx(nidx)
    sulfur = []
    for nbr in n_atom.GetNeighbors():
        sidx = nbr.GetIdx()
        if sidx == idx:
            continue
        bond = mol.GetBondBetweenAtoms(nidx, sidx)
        if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.SINGLE:
            sulfur.append(sidx)
        else:
            return ""
    if len(sulfur) != 1:
        return ""
    sulfonyl = _sulfur_prefix(mol, sulfur[0], nidx)
    if not sulfonyl or not sulfonyl.endswith("sulfonyl"):
        return ""
    return f"({sulfonyl}amino)methyl"


def _ring_ethynyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    triple_c = []
    for nbr in atom.GetNeighbors():
        if nbr.GetIdx() == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.TRIPLE:
            triple_c.append(nbr.GetIdx())
        else:
            return ""
    if len(triple_c) != 1:
        return ""
    distal = triple_c[0]
    ring_neighbors = [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(distal).GetNeighbors()
        if n.GetIdx() != idx and n.IsInRing()
    ]
    if len(ring_neighbors) != 1:
        return ""
    ring = _ring_substituent_prefix(mol, ring_neighbors[0], distal)
    if not ring:
        return ""
    return ring + "ethynyl"


def _dicyanoethenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    alkene = []
    other = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "C" and nbr.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
            alkene.append(nidx)
        else:
            other.append(nidx)
    if len(alkene) != 1 or other:
        return ""
    distal = alkene[0]
    cyano = []
    for nbr in mol.GetAtomWithIdx(distal).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx:
            continue
        if _is_nitrile_carbon(mol, nidx):
            cyano.append(nidx)
        else:
            return ""
    if len(cyano) != 2:
        return ""
    return "2,2-dicyanoethenyl"


def _alkynyl_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    triple_neighbors = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
    ]
    if len(triple_neighbors) != 1:
        return ""
    paths = _linear_alkynyl_paths(mol, idx, parent_idx, triple_neighbors[0], max_len=8)
    if not paths:
        return ""
    best = ""
    best_key = None
    for path in paths:
        if not _simple_alkynyl_side_branches(mol, path, parent_idx):
            continue
        name = _name_subtree_path(mol, path, set(collect_subtree(mol, idx, parent_idx)), parent_idx)
        if not name:
            continue
        key = (-len(path), substituent_locants(mol, path, set(path), parent_idx), name)
        if best_key is None or key < best_key:
            best_key = key
            best = name
    return best


def _simple_alkynyl_side_branches(mol, path: Sequence[int], parent_idx: int) -> bool:
    path_set = set(path)
    for idx in path:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == parent_idx or nidx in path_set:
                continue
            if nbr.IsInRing():
                return False
            if nbr.GetSymbol() == "C" and not _is_nitrile_carbon(mol, nidx):
                return False
    return True


def _linear_alkynyl_paths(
    mol, root_idx: int, parent_idx: int, first_idx: int, max_len: int
) -> list[tuple[int, ...]]:
    paths: list[tuple[int, ...]] = []
    stack = [(first_idx, root_idx, (root_idx, first_idx))]
    while stack:
        current, previous, path = stack.pop()
        carbon_next = []
        ok = True
        for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == previous:
                continue
            bond = mol.GetBondBetweenAtoms(current, nidx)
            if nbr.GetSymbol() == "C" and not nbr.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE:
                carbon_next.append(nidx)
            elif nidx != parent_idx:
                continue
        if len(carbon_next) > 1:
            ok = False
        if ok:
            paths.append(path)
        if ok and len(path) < max_len:
            for nidx in carbon_next:
                if nidx not in path:
                    stack.append((nidx, current, path + (nidx,)))
    return paths


def _simple_ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    rings = [tuple(r) for r in mol.GetRingInfo().AtomRings() if attachment_idx in r and parent_idx not in r]
    if len(rings) != 1:
        return ""
    ring = rings[0]
    ordered = ordered_ring(mol, ring)
    if not ordered:
        return ""
    candidates = _ring_parent_candidates(mol, ordered)
    candidates = _merge_ring_parent_candidates(
        candidates,
        _resource_ring_parent_candidates(mol, ordered, Chem),
    )
    if not candidates:
        return ""
    ring_set = set(ordered)
    raw_subs = []
    for ring_idx in ordered:
        atom = mol.GetAtomWithIdx(ring_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == parent_idx or nidx in ring_set:
                continue
            prefix = branch_prefix(mol, nidx, ring_idx)
            if not prefix:
                return ""
            raw_subs.append((ring_idx, prefix))

    best = ""
    best_key = None
    for parent_name, seq in candidates:
        if attachment_idx not in seq:
            continue
        loc = seq.index(attachment_idx) + 1
        charge_loc = _charged_ring_nitrogen_locant(mol, seq)
        suffix = _ring_substituent_suffix(parent_name, loc, charge_loc)
        if not suffix:
            continue
        suffix = _ring_substituent_unsaturated_suffix(mol, attachment_idx, parent_idx, suffix)
        if not suffix:
            continue
        pos = {idx: i + 1 for i, idx in enumerate(seq)}
        prefixes = [
            (str(pos[idx]), prefix)
            for idx, prefix in raw_subs
            if not (
                parent_name in {"1,3,4-dioxazol-2-one", "azet-2-one"} and pos[idx] == 2 and prefix == "oxo"
            )
            and not (
                parent_name in {"maleimide", "succinimide"} and pos[idx] in {2, 5} and prefix == "oxo"
            )
        ]
        prefix_text = _format_prefixes(prefixes)
        if prefix_text:
            separator = "-" if suffix[:1].isdigit() else ""
            name = prefix_text + separator + suffix
        else:
            name = suffix
        stereo_prefix = _simple_ring_substituent_stereo_prefix(mol, seq)
        if stereo_prefix:
            name = f"({stereo_prefix})-{name}"
        locants = sorted(pos[idx] for idx, _ in raw_subs)
        key = (0 if loc == 1 else 1, locants, name)
        if best_key is None or key < best_key:
            best_key = key
            best = name
    return best


def _merge_ring_parent_candidates(
    base_candidates: Sequence[tuple[str, tuple[int, ...]]],
    resource_candidates: Sequence[tuple[str, tuple[int, ...]]],
) -> list[tuple[str, tuple[int, ...]]]:
    merged: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    base_by_seq: dict[tuple[int, ...], set[str]] = {}
    for name, seq in base_candidates:
        seq_key = tuple(seq)
        base_by_seq.setdefault(seq_key, set()).add(name)
        base_by_seq[seq_key].add(_drop_indicated_hydrogen_prefix(name))
    for name, seq in tuple(resource_candidates) + tuple(base_candidates):
        key = (name, tuple(seq))
        if key in seen:
            continue
        if _resource_ring_alias_shadowed_by_base(name, tuple(seq), base_by_seq):
            continue
        seen.add(key)
        merged.append(key)
    return merged


def _resource_ring_alias_shadowed_by_base(
    name: str,
    seq: tuple[int, ...],
    base_by_seq: dict[tuple[int, ...], set[str]],
) -> bool:
    base_names = base_by_seq.get(seq, set())
    if not base_names:
        return False
    return name + "e" in base_names or _RESOURCE_RING_ALIAS_EQUIVALENTS.get(name, "") in base_names


_RESOURCE_RING_ALIAS_EQUIVALENTS = {
    "benzol": "benzene",
}


def _simple_ring_substituent_stereo_prefix(mol, seq: Sequence[int]) -> str:
    if not seq:
        return ""
    seq_set = set(seq)
    if _has_assigned_bond_stereo(mol, Chem):
        return ""
    centers = [
        (idx, descriptor)
        for idx, descriptor in assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
        if descriptor in {"R", "S"} and idx in seq_set
    ]
    if not centers:
        return _simple_ring_relative_cis_trans_prefix(mol, seq)
    locants = {idx: loc for loc, idx in enumerate(seq, start=1)}
    return ",".join(
        f"{locants[idx]}{descriptor}"
        for idx, descriptor in sorted(centers, key=lambda item: locants[item[0]])
    )


def _simple_ring_relative_cis_trans_prefix(mol, seq: Sequence[int]) -> str:
    """Return a relative cis/trans descriptor for tagged disubstituted rings.

    RDKit reports some symmetric cycloalkane stereocentres as lowercase
    relative ``r/s`` rather than absolute ``R/S``.  OPSIN does not reliably
    apply unlocanted ``(r,s)`` descriptors to substituent groups, but it does
    roundtrip conventional ``(cis)-``/``(trans)-`` prefixes for simple
    disubstituted cycloalkyl substituents.
    """

    seq_set = set(seq)
    tagged = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetIdx() in seq_set and atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    if len(tagged) != 2:
        return ""
    if any(atom.GetSymbol() != "C" or atom.GetIsAromatic() for atom in tagged):
        return ""
    if any(_ring_atom_external_degree(mol, atom.GetIdx(), seq_set) != 1 for atom in tagged):
        return ""
    ring_bonds = [
        bond
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in seq_set and bond.GetEndAtomIdx() in seq_set
    ]
    if any(bond.GetBondType() != Chem.BondType.SINGLE for bond in ring_bonds):
        return ""
    tags = [atom.GetChiralTag() for atom in tagged]
    return "cis" if tags[0] == tags[1] else "trans"


def _ring_atom_external_degree(mol, atom_idx: int, ring_atoms: set[int]) -> int:
    return sum(1 for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors() if nbr.GetIdx() not in ring_atoms)


def _ring_substituent_unsaturated_suffix(
    mol, attachment_idx: int, parent_idx: int, suffix: str
) -> str:
    bond = mol.GetBondBetweenAtoms(attachment_idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return suffix
    atom = mol.GetAtomWithIdx(attachment_idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    if suffix.endswith("yl"):
        return suffix + "idene"
    return ""


def _charged_ring_nitrogen_locant(mol, seq: Sequence[int]) -> int | None:
    charged = [
        loc
        for loc, idx in enumerate(seq, start=1)
        if mol.GetAtomWithIdx(idx).GetSymbol() in {"N", "S"} and mol.GetAtomWithIdx(idx).GetFormalCharge() > 0
    ]
    return charged[0] if len(charged) == 1 else None


def _fused_ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    curated = _fused_ring_substituent_prefix_from_templates(
        mol, attachment_idx, parent_idx, _curated_fused_templates(Chem), generated=False
    )
    if curated:
        return curated
    generated_templates = _candidate_generated_fused_templates(mol, Chem)
    generated = _fused_ring_substituent_prefix_from_templates(
        mol, attachment_idx, parent_idx, generated_templates, generated=True
    ) or _hydro_fused_ring_substituent_prefix_from_templates(mol, attachment_idx, parent_idx, generated_templates)
    if generated:
        return generated
    broad_generated_templates = _candidate_generated_fused_templates(mol, Chem, graph_filter=False)
    generated = _fused_ring_substituent_prefix_from_templates(
        mol, attachment_idx, parent_idx, broad_generated_templates, generated=True
    ) or _hydro_fused_ring_substituent_prefix_from_templates(mol, attachment_idx, parent_idx, broad_generated_templates)
    return generated


def _fused_ring_substituent_prefix_from_templates(
    mol, attachment_idx: int, parent_idx: int, templates: tuple[dict, ...], generated: bool
) -> str:
    best = ""
    best_key = None
    for template in template_store(Chem).candidate_templates(mol, templates):
        core = template_record_mol(template)
        for match in _substruct_matches(mol, core):
            if attachment_idx not in match or parent_idx in match:
                continue
            core_set = set(match)
            if ring_system_extends_core(mol, core_set):
                continue
            reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
            template_idx = reverse[attachment_idx]
            locant = template_record_locants(template).get(template_idx)
            if not locant or locant.endswith("a"):
                continue
            suffix = _fused_substituent_suffix(template_record_name(template), locant)
            if not suffix:
                continue
            prefix_text = _fused_substituent_prefix_text(mol, template, match, reverse, parent_idx, core_set)
            if prefix_text is None:
                continue
            name = _combine_fused_substituent_prefix(prefix_text, suffix)
            key = _fused_substituent_candidate_key(template, match, locant, name, generated)
            if best_key is None or (key < best_key if generated else key > best_key):
                best_key = key
                best = name
    return best


def _fused_substituent_candidate_key(
    template: dict, match: Sequence[int], locant: str, name: str, generated: bool
) -> tuple:
    if generated:
        return (_generated_fusion_preference(template_record_name(template)), -len(match), locant, name)
    return (len(match), locant, name)


def _hydro_fused_ring_substituent_prefix_from_templates(
    mol, attachment_idx: int, parent_idx: int, templates: tuple[dict, ...]
) -> str:
    best = ""
    best_key = None
    for template in template_store(Chem).candidate_templates(mol, templates):
        if not _template_is_fully_aromatic(template_record_mol(template)):
            continue
        query = _topology_query(template_record_mol(template))
        for match in _substruct_matches(mol, query):
            if attachment_idx not in match or parent_idx in match:
                continue
            core_set = set(match)
            if ring_system_extends_core(mol, core_set):
                continue
            parent_name = _hydro_fused_parent_name(mol, template, match)
            if not parent_name:
                continue
            reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
            template_idx = reverse[attachment_idx]
            locant = template_record_locants(template).get(template_idx)
            if not locant or locant.endswith("a"):
                continue
            suffix = _fused_substituent_suffix(parent_name, locant)
            prefix_text = _fused_substituent_prefix_text(mol, template, match, reverse, parent_idx, core_set)
            if prefix_text is None:
                continue
            name = _combine_fused_substituent_prefix(prefix_text, suffix)
            key = (_generated_fusion_preference(template_record_name(template)), -len(match), locant, name)
            if best_key is None or key < best_key:
                best_key = key
                best = name
    return best


def _substruct_matches(mol, query) -> tuple[tuple[int, ...], ...]:
    cache = mol_cache_dict(mol, "branch_ring_template_matches")
    key = id(query)
    cached = cache.get(key)
    if cached is not None:
        return cached
    matches = tuple(tuple(match) for match in mol.GetSubstructMatches(query, uniquify=False))
    cache[key] = matches
    return matches


def _fused_substituent_prefix_text(
    mol,
    template: dict,
    match: Sequence[int],
    reverse: dict[int, int],
    parent_idx: int,
    core_set: set[int],
) -> str | None:
    prefixes = []
    for mol_idx in match:
        core_loc = template_record_locants(template).get(reverse[mol_idx])
        for nbr in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == parent_idx or nidx in core_set:
                continue
            if not core_loc or core_loc.endswith("a"):
                return None
            prefix = branch_prefix(mol, nidx, mol_idx)
            if not prefix:
                return None
            prefixes.append((core_loc, prefix))
    return _format_prefixes(prefixes)


@lru_cache(maxsize=None)
def _topology_query(template_mol):
    query = Chem.RWMol()
    for atom in template_mol.GetAtoms():
        query.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    for bond in template_mol.GetBonds():
        query.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.UNSPECIFIED)
    return query.GetMol()


@lru_cache(maxsize=None)
def _template_is_fully_aromatic(template_mol) -> bool:
    return all(atom.GetIsAromatic() for atom in template_mol.GetAtoms())


def _hydro_fused_parent_name(mol, template: dict, match: Sequence[int]) -> str:
    hydro_locs = []
    charged_locs = []
    charged_has_hydrogen = False
    for template_idx, mol_idx in enumerate(match):
        template_atom = template_record_mol(template).GetAtomWithIdx(template_idx)
        atom = mol.GetAtomWithIdx(mol_idx)
        charge_matches = template_atom.GetFormalCharge() == atom.GetFormalCharge()
        charge_is_parent_ium = (
            template_atom.GetFormalCharge() == 0
            and atom.GetFormalCharge() > 0
            and atom.GetSymbol() in {"N", "S"}
        )
        if not charge_matches and not charge_is_parent_ium:
            return ""
        locant = template_record_locants(template).get(template_idx, "")
        if charge_is_parent_ium:
            if not locant.isdigit():
                return ""
            charged_locs.append(int(locant))
            if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0:
                charged_has_hydrogen = True
        if not template_atom.GetIsAromatic() or atom.GetIsAromatic():
            continue
        if _has_unsaturated_core_bond(mol, mol_idx, match):
            continue
        if not locant.isdigit():
            return ""
        hydro_locs.append(int(locant))
    if not hydro_locs:
        return ""
    hydro_locs = sorted(set(hydro_locs))
    hydro = {
        2: "dihydro",
        4: "tetrahydro",
        6: "hexahydro",
        8: "octahydro",
        10: "decahydro",
        12: "dodecahydro",
    }.get(len(hydro_locs))
    if not hydro:
        return ""
    locants = ",".join(str(loc) for loc in hydro_locs)
    name = f"{locants}-{hydro}-{template_record_name(template)}"
    if len(charged_locs) == 1:
        stem = name[:-1] if name.endswith("e") else name
        return f"{stem}-{charged_locs[0]}-ium"
    return name


def _has_unsaturated_core_bond(mol, mol_idx: int, match: Sequence[int]) -> bool:
    core = set(match)
    atom = mol.GetAtomWithIdx(mol_idx)
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx not in core:
            continue
        bond = mol.GetBondBetweenAtoms(mol_idx, nidx)
        if bond is not None and bond.GetBondTypeAsDouble() > 1.0:
            return True
    return False


def _fused_substituent_suffix(parent_name: str, locant: str) -> str:
    retained = _retained_fused_substituent_suffix(parent_name, locant)
    if retained:
        return retained
    stem = FUSED_SUBSTITUENT_STEMS.get(parent_name)
    if not stem:
        stem = parent_name
    return f"{stem}-{locant}-yl"


def _combine_fused_substituent_prefix(prefix_text: str, suffix: str) -> str:
    if not prefix_text:
        return suffix
    if suffix and suffix[0].isdigit():
        return f"{prefix_text}-{suffix}"
    return prefix_text + suffix


def _retained_fused_substituent_suffix(parent_name: str, locant: str) -> str:
    if parent_name == "naphthalene":
        return f"{locant}-naphthyl"
    if parent_name == "quinoline":
        return f"{locant}-quinolyl"
    if parent_name == "quinoxaline":
        return f"{locant}-quinoxalyl"
    if parent_name == "benzothiazole":
        return f"{locant}-benzothiazolyl"
    if parent_name == "benzoxazole":
        return f"{locant}-benzoxazolyl"
    return ""
