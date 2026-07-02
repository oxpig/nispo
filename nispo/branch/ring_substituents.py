from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from ..constants import MULT, ROOTS
from ..cycle_utils import ordered_ring, ring_system_extends_core
from ..formatting import format_prefixes as _format_prefixes
from ..benzimidazole_orientation import preferred_benzimidazole_n1_benzenoid_locant_map
from ..fused_rules import FUSED_SUBSTITUENT_STEMS
from ..fused_template_families import (
    candidate_generated_fused_templates as _candidate_generated_fused_templates,
    curated_fused_templates as _curated_fused_templates,
    generated_fusion_preference as _generated_fusion_preference,
)
from ..graph_model import graph_view
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
from ..simple_prefixes import simple_external_prefix
from ..template_access import template_record_locants, template_record_mol, template_record_name
from ..template_match_utils import match_has_extra_core_bonds
from ..template_store import template_store
from .functional_groups import (
    _is_nitrile_carbon,
    _name_subtree_path,
    _sulfur_prefix,
    collect_subtree,
    rooted_paths,
    substituent_from_root,
    substituent_locants,
)


def branch_prefix(
    mol,
    idx: int,
    parent_idx: int,
    *,
    assume_simple_external_prefix_miss: bool = False,
) -> str:
    from .. import branch_rules

    return branch_rules.branch_prefix(
        mol,
        idx,
        parent_idx,
        assume_simple_external_prefix_miss=assume_simple_external_prefix_miss,
    )


def _simple_or_branch_prefix(mol, idx: int, parent_idx: int) -> str:
    return simple_external_prefix(mol, idx, parent_idx) or branch_prefix(
        mol,
        idx,
        parent_idx,
        assume_simple_external_prefix_miss=True,
    )


def _ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    cache = mol_cache_dict(mol, "ring_substituent_prefix_cache")
    cache_key = (
        attachment_idx,
        parent_idx,
        mol.HasProp("_nispo_allow_fragment_substituent_fallback"),
        mol.GetProp("_nispo_fragment_substituent_depth") if mol.HasProp("_nispo_fragment_substituent_depth") else "",
    )
    if cache_key in cache:
        return cache[cache_key]
    if _same_ring_system_attachment(mol, attachment_idx, parent_idx):
        cache[cache_key] = ""
        return ""
    prefix = _carbohydrate_ring_substituent_prefix(
        mol,
        attachment_idx,
        parent_idx,
    ) or _cyclobutenedione_substituent_prefix(
        mol,
        attachment_idx,
        parent_idx,
    ) or _unsubstituted_carbocycle_prefix(
        mol,
        attachment_idx,
        parent_idx,
    ) or _unsubstituted_simple_ring_prefix(
        mol,
        attachment_idx,
        parent_idx,
    ) or _simple_ring_substituent_prefix(
        mol,
        attachment_idx,
        parent_idx,
    ) or _fused_ring_substituent_prefix(
        mol, attachment_idx, parent_idx
    ) or _resource_substituent_prefix(mol, attachment_idx, parent_idx, Chem) or _generic_polycyclic_ring_substituent_prefix(
        mol, attachment_idx, parent_idx
    ) or _fragment_named_ring_substituent_prefix(
        mol, attachment_idx, parent_idx
    )
    if prefix:
        prefix = _ring_substituent_unsaturated_suffix(mol, attachment_idx, parent_idx, prefix)
    cache[cache_key] = prefix
    return prefix


def _unsubstituted_carbocycle_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    rings = [
        tuple(ring)
        for ring in mol.GetRingInfo().AtomRings()
        if attachment_idx in ring and parent_idx not in ring
    ]
    if len(rings) != 1:
        return ""
    ring = rings[0]
    ring_set = set(ring)
    if any(
        mol.GetAtomWithIdx(idx).GetSymbol() != "C"
        or mol.GetAtomWithIdx(idx).GetFormalCharge() != 0
        or mol.GetAtomWithIdx(idx).GetIsotope()
        for idx in ring
    ):
        return ""
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set:
                continue
            if idx == attachment_idx and nidx == parent_idx:
                continue
            if nbr.GetAtomicNum() > 1:
                return ""
    if len(ring) == 6 and all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
        return "phenyl"
    if any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
        return ""
    for left, right in zip(ring, ring[1:] + ring[:1]):
        bond = mol.GetBondBetweenAtoms(left, right)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
    root = ROOTS.get(len(ring), "")
    return f"cyclo{root}yl" if root else ""


_SIMPLE_RING_FAST_SUFFIX_ALIASES = {
    "isoxazol": "isooxazol",
    "thiomorpholin": "thiamorpholin",
}


def _unsubstituted_simple_ring_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    parent_bond = mol.GetBondBetweenAtoms(attachment_idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    rings = [
        tuple(ring)
        for ring in mol.GetRingInfo().AtomRings()
        if attachment_idx in ring and parent_idx not in ring
    ]
    if len(rings) != 1:
        return ""
    ring = rings[0]
    ring_set = set(ring)
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() != 0 or atom.GetIsotope():
            return ""
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return ""
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set:
                continue
            if idx == attachment_idx and nidx == parent_idx:
                continue
            if nbr.GetAtomicNum() > 1:
                return ""
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in ring_set and bond.GetEndAtomIdx() in ring_set:
            if bond.GetStereo() != Chem.BondStereo.STEREONONE:
                return ""
    ordered = ordered_ring(mol, ring)
    if not ordered:
        return ""
    best = ""
    best_key = None
    for parent_name, seq in _ring_parent_candidates(mol, ordered):
        if attachment_idx not in seq:
            continue
        loc = seq.index(attachment_idx) + 1
        suffix = _ring_substituent_suffix(parent_name, loc, None, "")
        if not suffix:
            continue
        suffix = _fast_simple_ring_suffix_alias(suffix)
        key = (0 if loc == 1 else 1, suffix)
        if best_key is None or key < best_key:
            best_key = key
            best = suffix
    return best


def _fast_simple_ring_suffix_alias(suffix: str) -> str:
    for old, new in _SIMPLE_RING_FAST_SUFFIX_ALIASES.items():
        if suffix.startswith(old):
            return new + suffix[len(old) :]
    return suffix


def _same_ring_system_attachment(mol, attachment_idx: int, parent_idx: int) -> bool:
    attachment = mol.GetAtomWithIdx(attachment_idx)
    parent = mol.GetAtomWithIdx(parent_idx)
    if not attachment.IsInRing() or not parent.IsInRing():
        return False
    for system in graph_view(mol, Chem).ring_systems:
        if attachment_idx in system and parent_idx in system:
            return True
    return False


def _carbohydrate_ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    if not mol.GetAtomWithIdx(parent_idx).IsInRing():
        return ""
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


def _cyclobutenedione_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    rings = [tuple(r) for r in mol.GetRingInfo().AtomRings() if attachment_idx in r and parent_idx not in r]
    if len(rings) != 1 or len(rings[0]) != 4:
        return ""
    ring = rings[0]
    ring_set = set(ring)
    if any(
        mol.GetAtomWithIdx(idx).GetSymbol() != "C" or mol.GetAtomWithIdx(idx).GetFormalCharge() != 0
        for idx in ring
    ):
        return ""
    exocyclic_oxo: dict[int, str] = {}
    for idx in ring:
        symbol = _exocyclic_carbonyl_symbol(mol, idx, ring_set)
        if symbol:
            exocyclic_oxo[idx] = symbol
    if len(exocyclic_oxo) != 2:
        return ""
    carbonyl_atoms = tuple(exocyclic_oxo)
    if not _ring_atoms_adjacent(mol, carbonyl_atoms[0], carbonyl_atoms[1]):
        return ""
    alkene_atoms = [idx for idx in ring if idx not in exocyclic_oxo]
    if attachment_idx not in alkene_atoms:
        return ""
    other_alkene = next(idx for idx in alkene_atoms if idx != attachment_idx)
    alkene_bond = mol.GetBondBetweenAtoms(attachment_idx, other_alkene)
    if alkene_bond is None or alkene_bond.GetBondTypeAsDouble() not in {1.5, 2.0}:
        return ""
    seq = _cyclobutenedione_locant_sequence(mol, attachment_idx, other_alkene, carbonyl_atoms)
    if not seq:
        return ""
    pos = {idx: str(loc) for loc, idx in enumerate(seq, start=1)}
    prefixes = [(pos[idx], "oxo" if symbol == "O" else "thioxo") for idx, symbol in exocyclic_oxo.items()]
    for ring_idx in seq:
        for nbr in mol.GetAtomWithIdx(ring_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == parent_idx or nidx in ring_set:
                continue
            bond = mol.GetBondBetweenAtoms(ring_idx, nidx)
            if (
                ring_idx in exocyclic_oxo
                and nbr.GetSymbol() == exocyclic_oxo[ring_idx]
                and bond is not None
                and bond.GetBondTypeAsDouble() == 2.0
            ):
                continue
            prefix = _simple_or_branch_prefix(mol, nidx, ring_idx)
            if not prefix:
                return ""
            prefixes.append((pos[ring_idx], prefix))
    prefix_text = _format_prefixes(prefixes)
    return prefix_text + "cyclobut-1-en-1-yl" if prefix_text else ""


def _exocyclic_carbonyl_symbol(mol, atom_idx: int, ring_set: set[int]) -> str:
    found = []
    for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx in ring_set or nbr.GetSymbol() not in {"O", "S"}:
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
        if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
            found.append(nbr.GetSymbol())
    return found[0] if len(found) == 1 else ""


def _cyclobutenedione_locant_sequence(
    mol,
    attachment_idx: int,
    other_alkene_idx: int,
    carbonyl_atoms: Sequence[int],
) -> tuple[int, ...]:
    for left, right in (tuple(carbonyl_atoms), tuple(reversed(carbonyl_atoms))):
        seq = (attachment_idx, other_alkene_idx, left, right)
        if all(
            mol.GetBondBetweenAtoms(a, b) is not None
            for a, b in zip(seq, seq[1:] + seq[:1])
        ):
            return seq
    return ()


def _ring_atoms_adjacent(mol, left: int, right: int) -> bool:
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.IsInRing()


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
    oxy_prefix = _simple_or_branch_prefix(mol, oidx, carbon_idx)
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


def _ring_alkenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    double_c = []
    other = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if (
            nbr.GetSymbol() == "C"
            and nbr.GetFormalCharge() == 0
            and bond is not None
            and bond.GetBondType() == Chem.BondType.DOUBLE
        ):
            double_c.append(nidx)
        else:
            other.append(nidx)
    if len(double_c) != 1 or other:
        return ""
    distal = double_c[0]
    if not mol.GetAtomWithIdx(distal).IsInRing():
        return ""
    ring = _ring_substituent_prefix(mol, distal, idx)
    if not ring:
        return ""
    if not ring.endswith("ylidene"):
        if not ring.endswith("yl"):
            return ""
        ring += "idene"
    if "-" in ring or ring[:1].isdigit() or ring.startswith("("):
        return f"({ring})methyl"
    return ring + "methyl"


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
            prefix = _simple_or_branch_prefix(mol, nidx, ring_idx)
            if not prefix:
                return ""
            prefix = _with_ring_branch_atom_stereo_prefix(mol, nidx, ring_idx, prefix)
            raw_subs.append((ring_idx, prefix))

    best = ""
    best_key = None
    for parent_name, seq in candidates:
        if attachment_idx not in seq:
            continue
        loc = seq.index(attachment_idx) + 1
        charge_loc, charge_suffix = _charged_ring_substituent_charge(mol, seq, parent_name)
        suffix = _ring_substituent_suffix(parent_name, loc, charge_loc, charge_suffix)
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
        stereo_prefix = _simple_ring_substituent_stereo_prefix(
            mol,
            seq,
            parent_name,
            attachment_idx=attachment_idx,
            parent_idx=parent_idx,
        )
        if stereo_prefix:
            name = f"({stereo_prefix})-{name}"
        locants = sorted(pos[idx] for idx, _ in raw_subs)
        key = (0 if loc == 1 else 1, locants, name)
        if best_key is None or key < best_key:
            best_key = key
            best = name
    return best


def _with_ring_branch_atom_stereo_prefix(mol, root_idx: int, blocked_idx: int, prefix: str) -> str:
    if not prefix or _has_locanted_atom_stereo_prefix(prefix):
        return prefix
    stereo_prefix = _ring_branch_atom_stereo_prefix(mol, root_idx, blocked_idx)
    if not stereo_prefix:
        return prefix
    localized = _with_internal_n_ligand_stereo_prefix(prefix, stereo_prefix)
    if localized:
        return localized
    if prefix.startswith("(") and prefix.endswith(")"):
        inner = prefix[1:-1]
        if inner:
            return f"(({stereo_prefix})-{inner})"
    return f"({stereo_prefix})-{prefix}"


def _has_locanted_atom_stereo_prefix(prefix: str) -> bool:
    return bool(re.search(r"\((?:\d+'?)?[RS](?:,(?:\d+'?)?[RS])*\)-", prefix))


def _with_internal_n_ligand_stereo_prefix(prefix: str, stereo_prefix: str) -> str:
    match = re.fullmatch(r"\d+([RS])", stereo_prefix)
    if not match:
        return ""
    if prefix.startswith("N-(") and prefix.endswith(")carbamoyl"):
        ligand = prefix[len("N-(") : -len(")carbamoyl")]
        if not ligand or _has_locanted_atom_stereo_prefix(ligand):
            return ""
        return f"N-((1{match.group(1)})-{ligand})carbamoyl"
    if prefix.startswith("((") and prefix.endswith(")amino)carbonyl"):
        ligand = prefix[2 : -len(")amino)carbonyl")]
        if not ligand or _has_locanted_atom_stereo_prefix(ligand):
            return ""
        return f"(((1{match.group(1)})-{ligand})amino)carbonyl"
    return ""


def _ring_branch_atom_stereo_prefix(mol, root_idx: int, blocked_idx: int) -> str:
    if mol.GetAtomWithIdx(root_idx).IsInRing():
        return ""
    assigned = assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
    if not assigned:
        return ""
    atoms = collect_subtree(mol, root_idx, blocked_idx)
    if not atoms:
        return ""
    relevant = [(idx, descriptor) for idx, descriptor in assigned if idx in atoms]
    if not relevant:
        return ""
    paths = rooted_paths(mol, root_idx, atoms)
    if not paths:
        return ""
    best_path = ()
    best_key = None
    for path in paths:
        name = _name_subtree_path(mol, path, atoms, blocked_idx)
        if not name:
            continue
        key = (-len(path), substituent_locants(mol, path, atoms, blocked_idx), name)
        if best_key is None or key < best_key:
            best_key = key
            best_path = path
    if not best_path:
        return ""
    locant_by_atom = {idx: loc for loc, idx in enumerate(best_path, start=1)}
    connector_offset = _ring_branch_stereo_locant_connector_offset(mol, root_idx, blocked_idx)
    pieces = []
    for atom_idx, descriptor in relevant:
        locant = locant_by_atom.get(atom_idx)
        if locant is None:
            continue
        if connector_offset:
            if atom_idx == root_idx:
                continue
            locant -= connector_offset
            if locant < 1:
                continue
        pieces.append(f"{locant}{descriptor}")
    return ",".join(pieces)


def _ring_branch_stereo_locant_connector_offset(mol, root_idx: int, blocked_idx: int) -> int:
    root = mol.GetAtomWithIdx(root_idx)
    if root.GetSymbol() == "C":
        return 0
    if root.GetFormalCharge() != 0 or root.IsInRing():
        return 0
    parent_bond = mol.GetBondBetweenAtoms(root_idx, blocked_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return 0
    heavy = [
        nbr.GetIdx()
        for nbr in root.GetNeighbors()
        if nbr.GetIdx() != blocked_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return 0
    if root.GetSymbol() in {"N", "O", "S"}:
        return 1
    return 0


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


def _simple_ring_substituent_stereo_prefix(
    mol,
    seq: Sequence[int],
    parent_name: str = "",
    *,
    attachment_idx: int | None = None,
    parent_idx: int | None = None,
) -> str:
    if not seq:
        return ""
    seq_set = set(seq)
    centers = [
        (idx, descriptor)
        for idx, descriptor in assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
        if descriptor in {"R", "S"} and idx in seq_set
    ]
    relative = _simple_ring_relative_cis_trans_prefix(
        mol,
        seq,
        allow_hetero=parent_name in _RELATIVE_STEREO_SUBSTITUENT_PARENTS,
    )
    relative_parent_scope = parent_name in _RELATIVE_STEREO_SUBSTITUENT_PARENTS and _relative_dioxane_scope(
        mol,
        seq,
        parent_idx,
    )
    if relative:
        if parent_name in _RELATIVE_STEREO_SUBSTITUENT_PARENTS:
            if relative_parent_scope:
                return relative
        elif len(centers) < 2:
            return relative
    if len(centers) < 2:
        relative = _simple_ring_relative_cis_trans_prefix(mol, seq)
        if relative:
            return relative
    if not centers:
        return ""
    locants = {idx: loc for loc, idx in enumerate(seq, start=1)}
    return ",".join(
        f"{locants[idx]}{descriptor}"
        for idx, descriptor in sorted(centers, key=lambda item: locants[item[0]])
    )


def _simple_ring_relative_cis_trans_prefix(
    mol,
    seq: Sequence[int],
    *,
    allow_hetero: bool = False,
) -> str:
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
    if allow_hetero:
        if any(mol.GetAtomWithIdx(atom_idx).GetSymbol() not in {"C", "O"} for atom_idx in seq_set):
            return ""
    elif any(mol.GetAtomWithIdx(atom_idx).GetSymbol() != "C" for atom_idx in seq_set):
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


_RELATIVE_STEREO_SUBSTITUENT_PARENTS = {
    "1,3-dioxan",
    "1,3-dioxane",
}


def _relative_dioxane_scope(mol, seq: Sequence[int], parent_idx: int | None) -> bool:
    seq_set = set(seq)
    if sum(_ring_atom_external_degree(mol, atom_idx, seq_set) for atom_idx in seq_set) != 2:
        return False
    if parent_idx is None:
        return False
    for ring in mol.GetRingInfo().AtomRings():
        if parent_idx in ring and len(ring) <= 3:
            return False
    return True


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


def _charged_ring_substituent_charge(mol, seq: Sequence[int], parent_name: str = "") -> tuple[int | None, str]:
    if "ium" in parent_name:
        return None, "ium"
    charged = [
        (loc, "ylium" if atom.GetSymbol() == "C" else "ium")
        for loc, idx in enumerate(seq, start=1)
        for atom in (mol.GetAtomWithIdx(idx),)
        if atom.GetSymbol() in {"C", "N", "O", "S"} and atom.GetFormalCharge() > 0
    ]
    return charged[0] if len(charged) == 1 else (None, "ium")


def _charged_ring_nitrogen_locant(mol, seq: Sequence[int]) -> int | None:
    locant, suffix = _charged_ring_substituent_charge(mol, seq)
    return locant if suffix == "ium" else None


def _fused_ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    aromatic_view = _kekulized_aromatic_substituent_view(mol)
    if aromatic_view is not None:
        prefix = _fused_ring_substituent_prefix_for_view(
            aromatic_view,
            attachment_idx,
            parent_idx,
            carbocyclic_only=True,
        )
        if prefix:
            return prefix
    return _fused_ring_substituent_prefix_for_view(mol, attachment_idx, parent_idx)


def _fused_ring_substituent_prefix_for_view(
    mol,
    attachment_idx: int,
    parent_idx: int,
    *,
    carbocyclic_only: bool = False,
) -> str:
    if not _attachment_in_multi_ring_system(mol, attachment_idx, parent_idx):
        return ""
    curated_templates = _curated_fused_templates(Chem)
    if carbocyclic_only:
        curated_templates = _carbocyclic_fused_templates(curated_templates)
    curated = _fused_ring_substituent_prefix_from_templates(
        mol, attachment_idx, parent_idx, curated_templates, generated=False
    )
    if curated:
        return curated
    generated_templates = _candidate_generated_fused_templates(mol, Chem)
    if carbocyclic_only:
        generated_templates = _carbocyclic_fused_templates(generated_templates)
    generated = _fused_ring_substituent_prefix_from_templates(
        mol, attachment_idx, parent_idx, generated_templates, generated=True
    ) or _hydro_fused_ring_substituent_prefix_from_templates(mol, attachment_idx, parent_idx, generated_templates)
    if generated:
        return generated
    broad_generated_templates = _candidate_generated_fused_templates(mol, Chem, graph_filter=False)
    if carbocyclic_only:
        broad_generated_templates = _carbocyclic_fused_templates(broad_generated_templates)
    generated = _fused_ring_substituent_prefix_from_templates(
        mol, attachment_idx, parent_idx, broad_generated_templates, generated=True
    ) or _hydro_fused_ring_substituent_prefix_from_templates(mol, attachment_idx, parent_idx, broad_generated_templates)
    return generated


_CARBOCYCLIC_FUSED_TEMPLATE_CACHE: dict[int, tuple[dict, ...]] = {}


def _carbocyclic_fused_templates(templates: tuple[dict, ...]) -> tuple[dict, ...]:
    cached = _CARBOCYCLIC_FUSED_TEMPLATE_CACHE.get(id(templates))
    if cached is not None:
        return cached
    filtered = tuple(template for template in templates if _template_core_is_carbocyclic(template))
    _CARBOCYCLIC_FUSED_TEMPLATE_CACHE[id(templates)] = filtered
    return filtered


def _template_core_is_carbocyclic(template: dict) -> bool:
    core = template_record_mol(template)
    return all(atom.GetSymbol() == "C" for atom in core.GetAtoms())


def _kekulized_aromatic_substituent_view(mol):
    if not getattr(mol, "HasProp", lambda _name: False)("_nispo_kekulized_aromatic_view"):
        return None
    cache = mol_cache_dict(mol, "kekulized_aromatic_substituent_view")
    if "view" in cache:
        return cache["view"]
    view = Chem.Mol(mol)
    try:
        Chem.SetAromaticity(view)
        if view.HasProp("_nispo_kekulized_aromatic_view"):
            view.ClearProp("_nispo_kekulized_aromatic_view")
    except Exception:
        view = None
    cache["view"] = view
    return view


def _attachment_in_multi_ring_system(mol, attachment_idx: int, parent_idx: int) -> bool:
    rings = [frozenset(ring) for ring in mol.GetRingInfo().AtomRings() if parent_idx not in ring]
    containing = [ring for ring in rings if attachment_idx in ring]
    if not containing:
        return False
    system = set(containing[0])
    changed = True
    while changed:
        changed = False
        for ring in rings:
            if system & ring and not ring <= system:
                system.update(ring)
                changed = True
    return sum(1 for ring in rings if ring <= system) >= 2


def _fused_ring_substituent_prefix_from_templates(
    mol, attachment_idx: int, parent_idx: int, templates: tuple[dict, ...], generated: bool
) -> str:
    best = ""
    best_key = None
    for template in _candidate_templates(mol, templates):
        if _template_requires_systematic_substituent_name(template):
            continue
        core = template_record_mol(template)
        for match in _substruct_matches(mol, core):
            if attachment_idx not in match or parent_idx in match:
                continue
            if match_has_extra_core_bonds(mol, core, match):
                continue
            core_set = set(match)
            if ring_system_extends_core(mol, core_set):
                continue
            reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
            template_idx = reverse[attachment_idx]
            locant = template_record_locants(template).get(template_idx)
            if not locant or locant.endswith("a"):
                continue
            suffix = _charged_fused_substituent_suffix(mol, template, match, reverse, locant)
            if not suffix:
                suffix = _fused_substituent_suffix(template_record_name(template), locant)
            if not suffix:
                continue
            prefix_records = _fused_substituent_prefix_records(
                mol,
                template,
                match,
                reverse,
                parent_idx,
                core_set,
                attachment_locant=locant,
            )
            if prefix_records is None:
                continue
            prefix_text = _fused_substituent_prefix_text_from_records(prefix_records)
            hydro_prefix_text = _fused_substituent_oxo_hydro_prefix_text(
                mol,
                template,
                reverse,
                prefix_records,
            )
            if hydro_prefix_text:
                prefix_text = hydro_prefix_text
            name = _combine_fused_substituent_prefix(prefix_text, suffix)
            stereo_prefix = _fused_substituent_stereo_prefix(mol, template, match)
            if stereo_prefix:
                name = f"({stereo_prefix})-{name}"
            prefix_locants = _fused_substituent_prefix_locants_from_records(
                template,
                prefix_records,
                attachment_locant=locant,
            )
            key = _fused_substituent_candidate_key(
                mol,
                template,
                match,
                locant,
                name,
                generated,
                prefix_locants,
            )
            if best_key is None or (key < best_key if generated else key > best_key):
                best_key = key
                best = name
    return best


def _template_requires_systematic_substituent_name(template: dict) -> bool:
    return template_record_name(template) in _SYSTEMATIC_ONLY_SUBSTITUENT_TEMPLATE_NAMES


_SYSTEMATIC_ONLY_SUBSTITUENT_TEMPLATE_NAMES = frozenset(
    {
        # OPSIN accepts the bare retained acetal scaffold, but not its
        # locanted substituent forms. Let the graph renderer emit
        # 1,4-dioxaspiro[4.5]decane substituents instead.
        "cyclohexanone ethylene ketal",
        # OPSIN accepts dibenzofuran parents, but some locanted fused-substituent
        # aliases are orientation-fragile in larger fused systems. Keep the
        # retained parent available and use the graph renderer for substituents.
        "dibenzofuran",
    }
)


def _fused_substituent_candidate_key(
    mol,
    template: dict,
    match: Sequence[int],
    locant: str,
    name: str,
    generated: bool,
    prefix_locants: Sequence[str] = (),
) -> tuple:
    template_name = template_record_name(template)
    if generated:
        return (_generated_fusion_preference(template_name), -len(match), locant, name)
    h_state_key = _retained_fused_substituent_h_state_key(mol, template, match)
    preferred_locant = _PREFERRED_RETAINED_FUSED_SUBSTITUENT_LOCANTS.get(template_name)
    if preferred_locant:
        return (
            len(match),
            h_state_key,
            2,
            _preferred_retained_fused_substituent_locant_key(locant, preferred_locant),
            (0,),
            (0,),
            name,
        )
    if template_name in _LOW_LOCANT_RETAINED_FUSED_SUBSTITUENT_PARENTS:
        return (
            len(match),
            h_state_key,
            1,
            _low_locant_retained_fused_substituent_key(locant),
            (_low_locant_retained_fused_substituent_template_key(template_name),),
            _low_locant_retained_fused_substituent_prefix_key(prefix_locants),
            name,
        )
    return (
        len(match),
        h_state_key,
        0,
        _retained_fused_substituent_default_locant_key(locant),
        (0,),
        (0,),
        name,
    )


def _retained_fused_substituent_h_state_key(mol, template: dict, match: Sequence[int]) -> tuple[int, int, int]:
    template_name = template_record_name(template)
    if not _retained_fused_lactam_template_needs_h_state_preference(template_name):
        return (0, 0, 0)
    matches = 0
    mismatches = 0
    template_mol = template_record_mol(template)
    for template_idx, mol_idx in enumerate(match):
        template_atom = template_mol.GetAtomWithIdx(template_idx)
        if template_atom.GetSymbol() != "N":
            continue
        atom = mol.GetAtomWithIdx(mol_idx)
        if atom.GetSymbol() != "N":
            continue
        if template_atom.GetTotalNumHs() == atom.GetTotalNumHs():
            matches += 1
        else:
            mismatches += 1
    if not matches and not mismatches:
        return (0, 0, 0)
    return (1, -mismatches, matches)


def _retained_fused_lactam_template_needs_h_state_preference(template_name: str) -> bool:
    if not re.search(r"(?:^|[,( -])\d+[a-z]?H", template_name):
        return False
    return bool(re.search(r"(?:one|thione|dione)\b", template_name))


_PREFERRED_RETAINED_FUSED_SUBSTITUENT_LOCANTS = {
    # OPSIN's retained benzodioxane numbering for fused aryl substituents maps
    # the common para-fused external attachment to 6, not the highest symmetric
    # locant selected by the generic retained-template tie break.
    "1,3-benzodioxane": 6,
    "1,4-benzodioxane": 6,
}


def _preferred_retained_fused_substituent_locant_key(locant: str, preferred: int) -> tuple:
    if locant.isdigit():
        value = int(locant)
        return (1 if value == preferred else 0, -abs(value - preferred), -value)
    return (-1, locant)


_LOW_LOCANT_TAUTOMER_FUSED_SUBSTITUENT_PARENTS = frozenset(
    {
        "1H-benzimidazole",
        "1H-benzotriazole",
        "2H-benzotriazole",
    }
)

_LOW_LOCANT_RETAINED_FUSED_SUBSTITUENT_PARENTS = _LOW_LOCANT_TAUTOMER_FUSED_SUBSTITUENT_PARENTS | {
    "dibenzofuran",
}


def _low_locant_retained_fused_substituent_key(locant: str) -> tuple:
    if locant.isdigit():
        return (0, 100 - int(locant))
    return (-1, locant)


def _low_locant_retained_fused_substituent_prefix_key(locants: Sequence[str]) -> tuple:
    numeric = sorted(int(locant) for locant in locants if locant.isdigit())
    if numeric:
        return tuple(100 - locant for locant in numeric)
    return (0,)


def _low_locant_retained_fused_substituent_template_key(template_name: str) -> int:
    if template_name == "1H-benzotriazole":
        return 2
    if template_name == "2H-benzotriazole":
        return 1
    return 0


def _retained_fused_substituent_default_locant_key(locant: str) -> tuple:
    if locant.isdigit():
        return (0, int(locant))
    return (-1, locant)


def _locant_sort_key(locant: str) -> tuple[int, str]:
    if locant.isdigit():
        return (int(locant), "")
    match = re.fullmatch(r"(\d+)([a-z]?)", locant)
    if match:
        return (int(match.group(1)), match.group(2))
    return (999, locant)


def _hydro_fused_ring_substituent_prefix_from_templates(
    mol, attachment_idx: int, parent_idx: int, templates: tuple[dict, ...]
) -> str:
    best = ""
    best_key = None
    for template in _candidate_templates(mol, templates):
        if not _template_is_fully_aromatic(template_record_mol(template)):
            continue
        query = _topology_query(template_record_mol(template))
        for match in _substruct_matches(mol, query):
            if attachment_idx not in match or parent_idx in match:
                continue
            if match_has_extra_core_bonds(mol, query, match):
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
            prefix_text = _fused_substituent_prefix_text(
                mol,
                template,
                match,
                reverse,
                parent_idx,
                core_set,
                attachment_locant=locant,
            )
            if prefix_text is None:
                continue
            name = _combine_fused_substituent_prefix(prefix_text, suffix)
            stereo_prefix = _fused_substituent_stereo_prefix(mol, template, match)
            if stereo_prefix:
                name = f"({stereo_prefix})-{name}"
            prefix_locants = _fused_substituent_prefix_locants(
                mol,
                template,
                match,
                reverse,
                parent_idx,
                core_set,
                attachment_locant=locant,
            )
            key = (
                _generated_fusion_preference(template_record_name(template)),
                -len(match),
                locant,
                prefix_locants,
                name,
            )
            if best_key is None or key < best_key:
                best_key = key
                best = name
    return best


def _candidate_templates(mol, templates: tuple[dict, ...]) -> tuple[dict, ...]:
    cache = mol_cache_dict(mol, "branch_ring_candidate_templates")
    key = id(templates)
    cached = cache.get(key)
    if cached is not None:
        return cached
    candidates = tuple(template_store(Chem).candidate_templates(mol, templates))
    cache[key] = candidates
    return candidates


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
    *,
    attachment_locant: str = "",
) -> str | None:
    records = _fused_substituent_prefix_records(
        mol,
        template,
        match,
        reverse,
        parent_idx,
        core_set,
        attachment_locant=attachment_locant,
    )
    if records is None:
        return None
    return _fused_substituent_prefix_text_from_records(records)


def _fused_substituent_prefix_records(
    mol,
    template: dict,
    match: Sequence[int],
    reverse: dict[int, int],
    parent_idx: int,
    core_set: set[int],
    *,
    attachment_locant: str = "",
) -> tuple[tuple[str, str, str], ...] | None:
    prefix_records = []
    template_mol = template_record_mol(template)
    for mol_idx in match:
        core_loc = template_record_locants(template).get(reverse[mol_idx])
        for nbr in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == parent_idx or nidx in core_set:
                continue
            if not core_loc or core_loc.endswith("a"):
                return None
            prefix = _simple_or_branch_prefix(mol, nidx, mol_idx)
            if not prefix:
                return None
            prefix_records.append((core_loc, prefix, template_mol.GetAtomWithIdx(reverse[mol_idx]).GetSymbol()))
    if prefix_records:
        locant_map = _benzimidazole_n1_substituent_benzenoid_prefix_locant_map(
            template,
            attachment_locant,
            prefix_records,
        )
        return tuple((locant_map.get(locant, locant), prefix, symbol) for locant, prefix, symbol in prefix_records)
    return ()


def _fused_substituent_prefix_text_from_records(
    records: Sequence[tuple[str, str, str]],
) -> str:
    return _format_prefixes((locant, prefix) for locant, prefix, _symbol in records)


def _fused_substituent_oxo_hydro_prefix_text(
    mol,
    template: dict,
    reverse: dict[int, int],
    records: Sequence[tuple[str, str, str]],
) -> str:
    parent_name = template_record_name(template)
    if not _template_is_fully_aromatic(template_record_mol(template)):
        return ""
    if _fused_substituent_parent_has_hydro_segment(parent_name):
        return ""
    chalcogen_records = [
        (locant, prefix)
        for locant, prefix, _symbol in records
        if prefix in {"oxo", "thioxo"} and locant.isdigit()
    ]
    if len(chalcogen_records) != 1:
        return ""
    h_locs = _fused_substituent_indicated_hydrogen_locants(mol, template, reverse)
    if not h_locs:
        return ""
    chalcogen_loc, chalcogen_prefix = chalcogen_records[0]
    hydro_locs = tuple(sorted({chalcogen_loc, *h_locs}, key=_locant_sort_key))
    if len(hydro_locs) < 2:
        return ""
    remaining = [
        (locant, prefix)
        for locant, prefix, _symbol in records
        if not (locant == chalcogen_loc and prefix == chalcogen_prefix)
    ]
    prefixes = _format_prefixes([*remaining, (chalcogen_loc, chalcogen_prefix)])
    hydro = f"{','.join(hydro_locs)}-{MULT.get(len(hydro_locs), '')}hydro"
    return f"{prefixes}-{hydro}" if prefixes else hydro


def _fused_substituent_indicated_hydrogen_locants(
    mol,
    template: dict,
    reverse: dict[int, int],
) -> tuple[str, ...]:
    h_locs = []
    for mol_idx, template_idx in reverse.items():
        atom = mol.GetAtomWithIdx(mol_idx)
        locant = template_record_locants(template).get(template_idx)
        if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0 and _is_fused_indicated_h_locant(locant):
            h_locs.append(locant)
    return tuple(h_locs)


def _is_fused_indicated_h_locant(locant: str | None) -> bool:
    return bool(locant and re.match(r"\d+[a-z]?$", locant))


def _fused_substituent_parent_has_hydro_segment(parent_name: str) -> bool:
    return bool(re.search(r"(?:^|-)(?:di|tri|tetra|penta|hexa|hepta|octa)?hydro", parent_name))


def _fused_substituent_prefix_locants(
    mol,
    template: dict,
    match: Sequence[int],
    reverse: dict[int, int],
    parent_idx: int,
    core_set: set[int],
    *,
    attachment_locant: str = "",
) -> tuple[str, ...]:
    locants = []
    records = []
    template_mol = template_record_mol(template)
    for mol_idx in match:
        core_loc = template_record_locants(template).get(reverse[mol_idx])
        for nbr in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == parent_idx or nidx in core_set:
                continue
            if core_loc and not core_loc.endswith("a"):
                records.append((core_loc, template_mol.GetAtomWithIdx(reverse[mol_idx]).GetSymbol()))
    locant_map = _benzimidazole_n1_substituent_benzenoid_locant_map(
        template,
        attachment_locant,
        records,
    )
    for core_loc, _symbol in records:
        locants.append(locant_map.get(core_loc, core_loc))
    return tuple(sorted(locants, key=_locant_sort_key))


def _fused_substituent_prefix_locants_from_records(
    template: dict,
    records: Sequence[tuple[str, str, str]],
    *,
    attachment_locant: str = "",
) -> tuple[str, ...]:
    locant_map = _benzimidazole_n1_substituent_benzenoid_prefix_locant_map(
        template,
        attachment_locant,
        records,
    )
    return tuple(sorted((locant_map.get(locant, locant) for locant, _prefix, _symbol in records), key=_locant_sort_key))


def _benzimidazole_n1_substituent_benzenoid_prefix_locant_map(
    template: dict,
    attachment_locant: str,
    prefix_records: Sequence[tuple[str, str, str]],
) -> dict[str, str]:
    records = [(locant, symbol) for locant, _prefix, symbol in prefix_records]
    return _benzimidazole_n1_substituent_benzenoid_locant_map(template, attachment_locant, records)


def _benzimidazole_n1_substituent_benzenoid_locant_map(
    template: dict,
    attachment_locant: str,
    records: Sequence[tuple[str, str]],
) -> dict[str, str]:
    if template_record_name(template) != "1H-benzimidazole":
        return {}
    has_fixed_n1 = attachment_locant == "1" or any(
        locant == "1" and symbol == "N" for locant, symbol in records
    )
    if not has_fixed_n1:
        return {}
    carbon_locants = [locant for locant, symbol in records if symbol != "N"]
    return preferred_benzimidazole_n1_benzenoid_locant_map(carbon_locants)


def _fused_substituent_stereo_prefix(mol, template: dict, match: Sequence[int]) -> str:
    descriptors = dict(assigned_atom_stereochemistry(mol, Chem, atom_order="idx"))
    if not descriptors:
        return ""
    locants = template_record_locants(template)
    pieces = []
    seen = set()
    for template_idx, mol_idx in enumerate(match):
        locant = str(locants.get(template_idx, ""))
        if not locant.isdigit():
            continue
        atom = mol.GetAtomWithIdx(mol_idx)
        if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        descriptor = descriptors.get(mol_idx)
        if descriptor is None:
            continue
        text = f"{locant}{descriptor}"
        if text in seen:
            continue
        seen.add(text)
        pieces.append((int(locant), text))
    return ",".join(text for _locant, text in sorted(pieces))


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


def _charged_fused_substituent_suffix(
    mol,
    template: dict,
    match: Sequence[int],
    reverse: dict[int, int],
    attachment_locant: str,
) -> str:
    parent_name = template_record_name(template)
    if _charged_fused_substituent_template_is_unstable(parent_name):
        return ""
    template_mol = template_record_mol(template)
    locants = template_record_locants(template)
    cation_locs: list[str] = []
    cation_suffix = ""
    charged_has_hydrogen = False
    for mol_idx in match:
        template_idx = reverse[mol_idx]
        atom = mol.GetAtomWithIdx(mol_idx)
        template_atom = template_mol.GetAtomWithIdx(template_idx)
        locant = locants.get(template_idx, "")
        if (
            not locant
            or not locant.isdigit()
            or atom.GetFormalCharge() <= 0
            or template_atom.GetFormalCharge() != 0
        ):
            continue
        if atom.GetSymbol() == "C":
            suffix = "ylium"
        elif atom.GetSymbol() in {"N", "O", "S"}:
            suffix = "ium"
        else:
            continue
        cation_locs.append(locant)
        if cation_suffix and cation_suffix != suffix:
            return ""
        cation_suffix = suffix
        if atom.GetSymbol() in {"N", "O"} and atom.GetTotalNumHs() > 0:
            charged_has_hydrogen = True
    if len(cation_locs) != 1 or not cation_suffix:
        return ""
    if not charged_has_hydrogen:
        parent_name = _drop_indicated_hydrogen_prefix(parent_name)
    stem = parent_name[:-1] if parent_name.endswith("e") else parent_name
    charged_parent = f"{stem}-{cation_locs[0]}-{cation_suffix}"
    return _fused_substituent_suffix(charged_parent, attachment_locant)


def _charged_fused_substituent_template_is_unstable(parent_name: str) -> bool:
    if "ium" in parent_name:
        return True
    if "mercaptopurin" in parent_name or "hypoxanthin" in parent_name:
        return True
    return any(
        fragment in parent_name
        for fragment in (
            "imidazo[4,5-b]pyridine",
            "imidazo[4,5-c]pyridine",
            "pyrido[4,5-d]imidazole",
            "pyrido[5,6-d]imidazole",
        )
    )


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


def _fragment_named_ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    if not mol.HasProp("_nispo_allow_fragment_substituent_fallback"):
        return ""
    depth = int(mol.GetProp("_nispo_fragment_substituent_depth")) if mol.HasProp("_nispo_fragment_substituent_depth") else 0
    if mol.HasProp("_nispo_fragment_substituent_view") or depth >= 1:
        return ""
    atoms = collect_subtree(mol, attachment_idx, parent_idx)
    if len(atoms) < 5 or len(atoms) > 70:
        return ""
    if not any(mol.GetAtomWithIdx(atom_idx).IsInRing() for atom_idx in atoms):
        return ""
    fragment = _fragment_from_subtree_atoms(mol, atoms)
    if fragment is None:
        return ""
    frag_mol, original_to_fragment = fragment
    if attachment_idx not in original_to_fragment:
        return ""
    attachment = original_to_fragment[attachment_idx]
    try:
        from ..api import _engine
        from ..naming_context import MoleculeContext
        from ..rule_engine import (
            _first_valid_prepared_candidate,
            _prepare_candidates,
        )

        frag_mol.SetProp("_nispo_fragment_substituent_view", "1")
        frag_mol.SetProp("_nispo_fragment_substituent_depth", str(depth + 1))
        ctx = MoleculeContext(frag_mol, Chem)
        selected = _first_valid_prepared_candidate(
            frag_mol,
            Chem,
            ctx,
            _engine()._rules,
            max_length=360,
            output_max_length=360,
        )
        prepared = ()
        if selected is None or _fragment_substituent_candidate_needs_retained_selection(selected):
            prepared = _prepare_candidates(
                frag_mol,
                Chem,
                ctx,
                _engine()._rules,
                max_length=360,
                output_max_length=360,
            )
    except Exception:
        return ""
    if selected is None and not prepared:
        return ""
    if selected is None:
        selected = _fragment_substituent_best_prepared_candidate(prepared, attachment)
    elif prepared:
        selected = _fragment_substituent_best_prepared_candidate(prepared, attachment, selected) or selected
    locant = _candidate_atom_locant(selected.candidate, attachment)
    if not locant:
        return ""
    prefix = _parent_name_to_substituent_prefix(selected.output_name or selected.name, locant)
    if len(prefix) > 280:
        return ""
    return prefix


_RETAINED_FRAGMENT_SUBSTITUENT_FAMILIES = frozenset(
    {
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "hydro-fused-retained-parent",
        "partial-hydro-fused-retained-parent",
        "retained-ring-scaffold-parent",
        "retained-topology-replacement-fused-parent",
        "skeletal-replacement-fused-parent",
    }
)


def _fragment_substituent_candidate_needs_retained_selection(selected) -> bool:
    return selected.candidate.family == "kekulized-generic-polycyclic-parent"


def _fragment_substituent_best_prepared_candidate(
    prepared: Sequence,
    attachment: int,
    selected=None,
):
    selected_prefix = _fragment_substituent_prefix_for_item(selected, attachment)
    selected_key = _fragment_substituent_preference_key(selected, attachment) if selected is not None else None
    best = selected
    best_key = selected_key
    for item in prepared:
        if selected is not None and item.candidate.family not in _RETAINED_FRAGMENT_SUBSTITUENT_FAMILIES:
            continue
        prefix = _fragment_substituent_prefix_for_item(item, attachment)
        if not prefix:
            continue
        if selected_prefix and len(prefix) >= len(selected_prefix):
            continue
        key = _fragment_substituent_preference_key(item, attachment, prefix=prefix)
        if key is None:
            continue
        if best_key is None or key < best_key:
            best = item
            best_key = key
    return best


def _fragment_substituent_prefix_for_item(item, attachment: int) -> str:
    if item is None:
        return ""
    locant = _candidate_atom_locant(item.candidate, attachment)
    if not locant:
        return ""
    return _parent_name_to_substituent_prefix(item.output_name or item.name, locant)


def _fragment_substituent_preference_key(item, attachment: int, *, prefix: str = "") -> tuple | None:
    if item is None:
        return None
    if not prefix:
        prefix = _fragment_substituent_prefix_for_item(item, attachment)
        if not prefix:
            return None
    if item.candidate.parent_atoms and attachment not in item.candidate.parent_atoms:
        return None
    retained_rank = 0 if item.candidate.family in _RETAINED_FRAGMENT_SUBSTITUENT_FAMILIES else 1
    return (
        retained_rank,
        len(prefix),
        item.candidate.render_rank if item.candidate.render_rank is not None else 999,
        item.candidate.priority,
        prefix,
    )


def _fragment_from_subtree_atoms(mol, atoms: set[int]):
    ordered = tuple(sorted(atoms))
    original_to_fragment = {old_idx: new_idx for new_idx, old_idx in enumerate(ordered)}
    rw = Chem.RWMol()
    for old_idx in ordered:
        atom = Chem.Atom(mol.GetAtomWithIdx(old_idx))
        atom.SetAtomMapNum(0)
        rw.AddAtom(atom)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in original_to_fragment or end not in original_to_fragment:
            continue
        rw.AddBond(
            original_to_fragment[begin],
            original_to_fragment[end],
            bond.GetBondType(),
        )
        new_bond = rw.GetBondBetweenAtoms(original_to_fragment[begin], original_to_fragment[end])
        if new_bond is not None:
            new_bond.SetIsAromatic(bond.GetIsAromatic())
    frag = rw.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except Exception:
        try:
            Chem.SanitizeMol(
                frag,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
        except Exception:
            return None
    return frag, original_to_fragment


def _candidate_atom_locant(candidate, atom_idx: int) -> str:
    for idx, locant in (*candidate.locants, *candidate.stereo_locants):
        if int(idx) == atom_idx and str(locant):
            return str(locant)
    if candidate.parent_atoms and atom_idx in candidate.parent_atoms and len(candidate.parent_atoms) == 1:
        return "1"
    return ""


def _parent_name_to_substituent_prefix(name: str, locant: str) -> str:
    if not name or not locant:
        return ""
    retained = _retained_fused_substituent_suffix(name, locant)
    if retained:
        return retained
    if name == "benzene":
        return "phenyl"
    if name.endswith("e"):
        return f"{name[:-1]}-{locant}-yl"
    return f"{name}-{locant}-yl"
