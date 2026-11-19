from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from functools import lru_cache

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from ..molecule_iteration import iter_atoms, iter_bonds
from ..constants import MULT, ROOTS
from ..cycle_utils import ordered_ring, ring_system_extends_core
from ..fast_ring_scope import fast_carbon_ring_numbering_enabled
from ..formatting import format_prefixes as _format_prefixes
from ..benzimidazole_orientation import preferred_benzimidazole_n1_benzenoid_locant_map
from ..fused_rules import FUSED_SUBSTITUENT_STEMS
from ..fused_template_families import (
    candidate_generated_fused_templates as _candidate_generated_fused_templates,
    curated_fused_templates as _curated_fused_templates,
    generated_fusion_preference as _generated_fusion_preference,
)
from ..molecule_cache import mol_cache_dict, mol_cache_get
from ..parent_join_presentation import CARBONYL_SUFFIX_ENDINGS
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
from ..template_index import intern_template_tuple
from ..template_match_utils import match_has_extra_core_bonds, match_preserves_aromatic_nitrogen_hydrogens
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


_RING_SUBSTITUENT_PREFIX_CACHE_MISS = object()
_RING_FRAGMENT_PREFIX_CACHE_LIMIT = 128


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
    cached = cache.get(cache_key, _RING_SUBSTITUENT_PREFIX_CACHE_MISS)
    if cached is not _RING_SUBSTITUENT_PREFIX_CACHE_MISS:
        return cached
    if _same_ring_system_attachment(mol, attachment_idx, parent_idx):
        cache[cache_key] = ""
        return ""
    prefix = _mapped_saturated_spiro_prefix(
        mol, attachment_idx, parent_idx
    ) or _mapped_hydrogenated_heterocycle_prefix(
        mol, attachment_idx, parent_idx
    ) or _carbohydrate_ring_substituent_prefix(
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
    ) or _proved_fused_ring_substituent_prefix(
        mol, attachment_idx, parent_idx
    ) or _fused_ring_substituent_prefix(
        mol, attachment_idx, parent_idx
    ) or _resource_substituent_prefix(mol, attachment_idx, parent_idx, Chem) or _generic_polycyclic_ring_substituent_prefix(
        mol, attachment_idx, parent_idx
    ) or _fragment_named_ring_substituent_prefix(
        mol, attachment_idx, parent_idx
    )
    if prefix:
        if prefix.endswith('phenyl') and prefix.count('phenyl') == 2:
            from ..ring_assembly_methods import rooted_biphenyl_from_prefix

            prefix = rooted_biphenyl_from_prefix(mol, Chem, attachment_idx, parent_idx, prefix)
        prefix = _ring_substituent_unsaturated_suffix(mol, attachment_idx, parent_idx, prefix)
        # Capping a ring fragment can make its attachment carbon achiral.
        # Restore that site's original whole-graph configuration here too:
        # amino/acyl writers may call this ring writer directly.
        from ..branch_rules import _localize_attachment_stereo
        prefix = _localize_attachment_stereo(mol, attachment_idx, parent_idx, prefix)
    cache[cache_key] = prefix
    return prefix


def _proved_fused_ring_substituent_prefix(mol, attachment_idx, parent_idx):
    from ..mapped_ring_parents import proved_fused_rooted_prefix, ring_systems, rooted_benzo_imidazothiazole
    atoms = frozenset(collect_subtree(mol, attachment_idx, parent_idx))
    if len(atoms) >= 12 and mol.GetRingInfo().NumRings() >= 3:
        core = next((ring for ring in ring_systems(mol) if attachment_idx in ring), frozenset())
        text = rooted_benzo_imidazothiazole(mol, Chem, attachment_idx, atoms, core)
        if text:
            return text
    text, owned = proved_fused_rooted_prefix(mol, Chem, attachment_idx, parent_idx, atoms)
    return text if owned == atoms else ''


def _mapped_hydrogenated_heterocycle_prefix(mol, attachment_idx, parent_idx):
    """Keep a subordinate oxo/imino monocycle's actual H accommodation.

    A retained ring word plus an oxo prefix alone can describe a different
    tautomer. The rooted P-58 construction supplies the required hydro/added-H
    operation before this component is embedded in an alkyl or amino branch.
    A single cycle has one annular path for this rooted operation. Fused
    systems need a complete fused-parent construction: inserting this local
    word into a legacy fused writer can change its independent H assignment.
    Ordinary unsubstituted and saturated ring vocabulary is unchanged.
    """
    from ..mapped_ring_parents import ring_systems, records
    core = next((system for system in ring_systems(mol) if attachment_idx in system), None)
    if (core is None or parent_idx in core
            or any(sum(n.GetIdx() in core for n in mol.GetAtomWithIdx(i).GetNeighbors()) != 2
                   for i in core)
            or not any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in core)
            or not any(b.GetBondTypeAsDouble() == 2
                       and b.GetOtherAtom(mol.GetAtomWithIdx(i)).GetIdx() not in core
                       and b.GetOtherAtom(mol.GetAtomWithIdx(i)).GetAtomicNum() in {7, 8, 16}
                       for i in core for b in mol.GetAtomWithIdx(i).GetBonds())):
        return ''
    atoms = frozenset(collect_subtree(mol, attachment_idx, parent_idx))
    rows = records(mol, Chem, scope=atoms, attachment=attachment_idx)
    if len(rows) != 1:
        return ''
    row = rows[0]
    if (not (row.hydro_locants or row.added_h_locants)
            or not row.hydrogen_budget_proved or not row.annotation_coverage
            or row.preference_issues or row.construction is None
            or row.construction.atoms | row.construction.embedded_atoms != atoms):
        return ''
    return row.name


def _mapped_saturated_spiro_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    from ..mapped_ring_parents import ring_systems, rooted_prefix

    core = next((system for system in ring_systems(mol) if attachment_idx in system), None)
    if core is None or parent_idx in core:
        return ''
    bonds = [b for b in iter_bonds(mol)
             if b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core]
    if (len(bonds)-len(core)+1 not in {2, 3}
            or any(b.GetBondTypeAsDouble() != 1 for b in bonds)
            or not any(sum(n.GetIdx() in core for n in mol.GetAtomWithIdx(i).GetNeighbors()) == 4
                       for i in core)):
        return ''
    atoms = frozenset(collect_subtree(mol, attachment_idx, parent_idx))
    text, owned = rooted_prefix(mol, Chem, attachment_idx, parent_idx, atoms)
    return text if owned == atoms else ''


def _unsubstituted_carbocycle_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    rings = _rings_for_attachment_excluding_parent(mol, attachment_idx, parent_idx)
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
    "thiomorpholin": "thiamorpholin",
}


def _unsubstituted_simple_ring_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    parent_bond = mol.GetBondBetweenAtoms(attachment_idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    rings = _rings_for_attachment_excluding_parent(mol, attachment_idx, parent_idx)
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
    for bond in iter_bonds(mol):
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
    cache = mol_cache_dict(mol, "same_ring_system_attachment_cache")
    key = (attachment_idx, parent_idx)
    if key in cache:
        return cache[key]
    attachment = mol.GetAtomWithIdx(attachment_idx)
    parent = mol.GetAtomWithIdx(parent_idx)
    if not attachment.IsInRing() or not parent.IsInRing():
        cache[key] = False
        return False
    system_ids = _ring_system_id_by_atom(mol)
    result = system_ids[attachment_idx] >= 0 and system_ids[attachment_idx] == system_ids[parent_idx]
    cache[key] = result
    return result


def _ring_system_id_by_atom(mol) -> tuple[int, ...]:
    cached = mol_cache_dict(mol, "ring_system_id_by_atom_cache").get("ids")
    if cached is not None:
        return cached
    atom_rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    systems = _merged_ring_systems(atom_rings)
    ids = [-1] * mol.GetNumAtoms()
    for system_id, system in enumerate(systems):
        for atom_idx in system:
            ids[atom_idx] = system_id
    result = tuple(ids)
    mol_cache_dict(mol, "ring_system_id_by_atom_cache")["ids"] = result
    return result


def _merged_ring_systems(atom_rings: tuple[frozenset[int], ...]) -> tuple[frozenset[int], ...]:
    systems: list[set[int]] = []
    for ring in atom_rings:
        current = set(ring)
        rest = []
        for system in systems:
            if current & system:
                current |= system
            else:
                rest.append(system)
        rest.append(current)
        systems = rest
    changed = True
    while changed:
        changed = False
        merged = []
        while systems:
            current = systems.pop()
            overlapping = [system for system in systems if current & system]
            if overlapping:
                changed = True
                for system in overlapping:
                    current |= system
                    systems.remove(system)
            merged.append(current)
        systems = merged
    return tuple(frozenset(system) for system in systems)


def _rings_for_attachment_excluding_parent(
    mol,
    attachment_idx: int,
    parent_idx: int,
) -> tuple[tuple[int, ...], ...]:
    cache = mol_cache_dict(mol, "attachment_rings_excluding_parent_cache")
    key = (attachment_idx, parent_idx)
    cached = cache.get(key)
    if cached is not None:
        return cached
    rings = tuple(
        tuple(ring)
        for ring in mol.GetRingInfo().AtomRings()
        if attachment_idx in ring and parent_idx not in ring
    )
    cache[key] = rings
    return rings


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

    existing = polycyclic_substituent_prefix(mol, Chem, attachment_idx, parent_idx)
    if existing:
        return existing
    from ..principal_carbonyl import higher_carbopolycycle_prefix

    return higher_carbopolycycle_prefix(mol, Chem, attachment_idx, parent_idx)


def _cyclobutenedione_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    rings = _rings_for_attachment_excluding_parent(mol, attachment_idx, parent_idx)
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
    # P-29.1.2/P-29.3.2: the distal ring substitutes position 2 of
    # ethynyl. Keep that compound unit explicit before outer multiplication.
    # Otherwise diphenylethynyl can denote two phenyls on one ethynyl group.
    from ..formatting import format_prefixes
    return format_prefixes([('2', ring)]) + "ethynyl"


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
    rings = _rings_for_attachment_excluding_parent(mol, attachment_idx, parent_idx)
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

    pyridine_numbering_proved = _proved_pyridine_substituent_numbering(
        mol, ordered, attachment_idx, parent_idx, candidates, raw_subs
    )
    # This gate is deliberately limited to the substituted diazine parents.
    # The current branch writer is also used by fast naming, so other rings
    # must not incur the additional graph and resource-completeness proof.
    diazine_numbering_proved = bool(
        candidates
        and candidates[0][0] in _DIAZINE_NUMBERING_ORBITS
        and _proved_diazine_substituent_numbering(
            mol, ordered, attachment_idx, parent_idx, candidates, raw_subs
        )
    )
    best = ""
    best_key = None
    for parent_name, seq in candidates:
        parent_name = _pyrazole_hydride_parent(mol, parent_name, seq)
        if not parent_name:
            continue
        if attachment_idx not in seq:
            continue
        loc = seq.index(attachment_idx) + 1
        locants = sorted(seq.index(idx) + 1 for idx, _ in raw_subs)
        pos = {idx: i + 1 for i, idx in enumerate(seq)}
        carbocycle = parent_name.startswith("cyclo") and all(
            mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 for idx in seq
        )
        multiple, double = [], []
        for bond in iter_bonds(mol) if carbocycle else ():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            order = bond.GetBondTypeAsDouble()
            if left in pos and right in pos and order in {2, 3}:
                a, b = sorted((pos[left], pos[right]))
                bond_loc = a if b == a + 1 else b
                multiple.append(bond_loc)
                if order == 2:
                    double.append(bond_loc)
        # Heterocyclic template candidates have their own fixed numbering and
        # carbonyl/H-state conventions. Retain their established ordering.
        numbering_key = ((loc if carbocycle or pyridine_numbering_proved or diazine_numbering_proved
                          else 0 if loc == 1 else 1),
                         tuple(sorted(multiple)), tuple(sorted(double)), tuple(locants))
        # The rendered text is only the final tie breaker. Keep every tied
        # numbering, including its stereo descriptors, in the original order.
        if best_key is not None and numbering_key > best_key[:len(numbering_key)]:
            continue
        charge_loc, charge_suffix = _charged_ring_substituent_charge(mol, seq, parent_name)
        suffix = _ring_substituent_suffix(parent_name, loc, charge_loc, charge_suffix)
        if not suffix:
            continue
        suffix = _ring_substituent_unsaturated_suffix(
            mol, attachment_idx, parent_idx, suffix, decorate_stereo=False)
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
        # An oxo group on an aromatic heterocycle changes its hydrogen state.
        # Include the hydro locants so a suffix-parent change preserves the
        # lactam tautomer rather than producing a different ring valence.
        oxo_locs = [locant for locant, prefix in prefixes if prefix in {"oxo", "thioxo"}]
        nh_locs = [str(pos[idx]) for idx in seq
                   if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7
                   and mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0]
        if (len(oxo_locs) == 1 and nh_locs
                and all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in seq)
                and not _fused_substituent_parent_has_hydro_segment(parent_name)
                and not parent_name.endswith(("one", "thione"))):
            hydro_locs = sorted(set(oxo_locs + nh_locs), key=_locant_sort_key)
            if len(hydro_locs) == 2:
                suffix = _drop_indicated_hydrogen_prefix(suffix)
                prefix_text += f"-{','.join(hydro_locs)}-dihydro"
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
        from ..stereochemistry import apply_mapped_branch_bond_stereochemistry
        name = apply_mapped_branch_bond_stereochemistry(
            mol, Chem, name, tuple((idx, str(pos[idx])) for idx in seq),
            external_parent=parent_idx, attachment_idx=attachment_idx)
        key = (*numbering_key, name)
        if best_key is None or key < best_key:
            best_key = key
            best = name
    return best


def _proved_pyridine_substituent_numbering(
    mol, ordered, attachment_idx, parent_idx, candidates, raw_subs
):
    """Prove the two fixed pyridine orientations and their owned branches.

    P-14.4(c) assigns the free valence a lower locant before the ordinary
    substituent-prefix locants. This only changes the old heterocycle key
    where both orientations and the complete, unannotated original branch
    are available; other ring parents retain their existing ordering.
    """
    core = set(ordered)
    if (len(core) != 6 or ring_system_extends_core(mol, core)
            or attachment_idx not in core or parent_idx in core
            or mol.GetBondBetweenAtoms(attachment_idx, parent_idx) is None
            or mol.GetBondBetweenAtoms(attachment_idx, parent_idx).GetBondTypeAsDouble() != 1):
        return False
    atoms = [mol.GetAtomWithIdx(i) for i in core]
    if (sum(atom.GetAtomicNum() == 7 for atom in atoms) != 1
            or sum(atom.GetAtomicNum() == 6 for atom in atoms) != 5
            or any(not atom.GetIsAromatic() or atom.GetFormalCharge() for atom in atoms)
            or mol.GetAtomWithIdx(attachment_idx).GetAtomicNum() != 6):
        return False
    if any(name != 'pyridine' for name, _ in candidates) or len(candidates) != 2:
        return False
    first, second = (tuple(seq) for _, seq in candidates)
    if (set(first) != core or set(second) != core or first[0] != second[0]
            or first[0] not in core or mol.GetAtomWithIdx(first[0]).GetAtomicNum() != 7
            or first[1:] != tuple(reversed(second[1:]))):
        return False
    scope = frozenset(collect_subtree(mol, attachment_idx, parent_idx))
    from ..component_mapping import has_annotations, mapped_branch
    if has_annotations(mol, scope):
        return False
    if any(any(atom.GetIdx() in scope for atom in group.GetAtoms())
           for group in mol.GetStereoGroups()):
        return False
    cuts = {(i, nbr.GetIdx()) for i in scope for nbr in mol.GetAtomWithIdx(i).GetNeighbors()
            if nbr.GetIdx() not in scope and (nbr.GetAtomicNum() > 1 or nbr.GetIsotope())}
    if cuts != {(attachment_idx, parent_idx)}:
        return False
    from collections import Counter
    expected = Counter(raw_subs)
    actual = Counter()
    owned = set(core)
    for host in core:
        for nbr in mol.GetAtomWithIdx(host).GetNeighbors():
            root = nbr.GetIdx()
            if root in core or root == parent_idx or nbr.GetAtomicNum() == 1:
                continue
            child = collect_subtree(mol, root, host)
            if child & owned or not child <= scope:
                return False
            child_cuts = {(i, n.GetIdx()) for i in child for n in mol.GetAtomWithIdx(i).GetNeighbors()
                          if n.GetIdx() not in child and (n.GetAtomicNum() > 1 or n.GetIsotope())}
            if child_cuts != {(root, host)}:
                return False
            text, mapped = mapped_branch(mol, Chem, root, host)
            if not text or set(mapped) != child:
                return False
            actual[(host, text)] += 1
            owned.update(child)
    return owned == set(scope) and actual == expected


_DIAZINE_NUMBERING_ORBITS = {
    # Fixed heteroatom positions and the size of their complete dihedral orbit.
    "pyridazine": ((1, 2), 2),
    "pyrimidine": ((1, 3), 2),
    "pyrazine": ((1, 4), 4),
}


def _proved_diazine_substituent_numbering(
    mol, ordered, attachment_idx, parent_idx, candidates, raw_subs
):
    """Prove one P-14.4(c) numbering of a fixed, substituted diazine.

    This is local numbering of the ring substituent, not a whole-name PIN
    decision. Tied free-valence locants, annotations, other ring parents,
    incomplete resource matches and non-owned branches remain outside scope.
    """
    if len(ordered) != 6 or not candidates or not raw_subs:
        return False
    name = candidates[0][0]
    spec = _DIAZINE_NUMBERING_ORBITS.get(name)
    if spec is None:
        return False
    hetero_locants, orbit_size = spec
    if len(candidates) != orbit_size or any(candidate_name != name
                                             for candidate_name, _ in candidates):
        return False
    core = set(ordered)
    if (len(core) != 6 or ring_system_extends_core(mol, core)
            or attachment_idx not in core or parent_idx in core):
        return False
    attachment_bond = mol.GetBondBetweenAtoms(attachment_idx, parent_idx)
    if attachment_bond is None or attachment_bond.GetBondTypeAsDouble() != 1:
        return False
    atoms = [mol.GetAtomWithIdx(i) for i in ordered]
    if (sum(atom.GetAtomicNum() == 6 for atom in atoms) != 4
            or sum(atom.GetAtomicNum() == 7 for atom in atoms) != 2
            or mol.GetAtomWithIdx(attachment_idx).GetAtomicNum() != 6
            or any(not atom.GetIsAromatic() or atom.GetFormalCharge()
                   or atom.GetNumRadicalElectrons() or atom.GetIsotope()
                   or (atom.GetAtomicNum() == 7 and atom.GetTotalNumHs(includeNeighbors=True))
                   for atom in atoms)):
        return False
    annular = [bond for idx in core for bond in mol.GetAtomWithIdx(idx).GetBonds()
               if bond.GetBeginAtomIdx() == idx and bond.GetEndAtomIdx() in core]
    if (len(annular) != 6 or any(not bond.GetIsAromatic()
            or bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in annular)):
        return False

    # Derive the entire fixed-numbering orbit from the original cycle, rather
    # than assuming that an emitted or resource-matched candidate is complete.
    orientations = set()
    sequence = tuple(ordered)
    for direction in (sequence, tuple(reversed(sequence))):
        for offset in range(6):
            candidate = direction[offset:] + direction[:offset]
            if any(mol.GetBondBetweenAtoms(a, b) is None for a, b in
                   zip(candidate, candidate[1:] + candidate[:1])):
                return False
            positions = tuple(i + 1 for i, idx in enumerate(candidate)
                              if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7)
            if positions == hetero_locants:
                orientations.add(candidate)
    if len(orientations) != orbit_size:
        return False
    if {tuple(seq) for _, seq in candidates} != orientations:
        return False
    attachment_locs = [seq.index(attachment_idx) + 1 for seq in orientations]
    if attachment_locs.count(min(attachment_locs)) != 1:
        return False

    # Resource matches are globally capped by RDKit. The caller has already
    # populated these exact query caches; reject a saturated or missing cache
    # instead of concluding that no competing resource alias exists.
    from ..ring_resource_rules import _single_ring_templates_by_symbols
    matches_by_query = mol_cache_get(mol, "resource_ring_matches_by_atom_set", {})
    templates = _single_ring_templates_by_symbols(Chem).get(
        (6, ("C", "C", "C", "C", "N", "N")), ()
    )
    for template, _ in templates:
        groups = matches_by_query.get(template_record_mol(template))
        if groups is None or sum(len(rows) for rows in groups.values()) >= 1000:
            return False

    scope = frozenset(collect_subtree(mol, attachment_idx, parent_idx))
    from ..component_mapping import has_annotations, mapped_branch
    if has_annotations(mol, scope):
        return False
    if any(any(atom.GetIdx() in scope for atom in group.GetAtoms())
           for group in mol.GetStereoGroups()):
        return False
    cuts = {(i, nbr.GetIdx()) for i in scope for nbr in mol.GetAtomWithIdx(i).GetNeighbors()
            if nbr.GetIdx() not in scope and (nbr.GetAtomicNum() > 1 or nbr.GetIsotope())}
    if cuts != {(attachment_idx, parent_idx)}:
        return False

    # Every already-rendered child must own precisely its disjoint original
    # graph component, attached by one ordinary bond to this same ring.
    from collections import Counter
    expected = Counter(raw_subs)
    actual = Counter()
    owned = set(core)
    for host in core:
        for nbr in mol.GetAtomWithIdx(host).GetNeighbors():
            root = nbr.GetIdx()
            if root in core or root == parent_idx or nbr.GetAtomicNum() == 1:
                continue
            if mol.GetBondBetweenAtoms(host, root).GetBondTypeAsDouble() != 1:
                return False
            child = collect_subtree(mol, root, host)
            if child & owned or not child <= scope:
                return False
            child_cuts = {(i, n.GetIdx()) for i in child for n in mol.GetAtomWithIdx(i).GetNeighbors()
                          if n.GetIdx() not in child and (n.GetAtomicNum() > 1 or n.GetIsotope())}
            if child_cuts != {(root, host)}:
                return False
            text, mapped = mapped_branch(mol, Chem, root, host)
            if not text or set(mapped) != child:
                return False
            actual[(host, text)] += 1
            owned.update(child)
    return owned == set(scope) and actual == expected


def _pyrazole_hydride_parent(mol, parent_name: str, seq: Sequence[int]) -> str:
    """Identify the neutral aromatic pyrazole hydride before substitution.

    P-14.7.1 and P-15.1.5.1.3 make indicated hydrogen part of the parent
    name.  Its atom can instead bear a substituent in the actual molecule.
    A valid 1H-pyrazole numbering places this pyrrole-like nitrogen at 1;
    discard the reversed sequence rather than attaching 1H to another atom.
    Leave charged, isotopic and non-mancude cases to their existing paths.
    """
    if parent_name != "pyrazole" or len(seq) != 5:
        return parent_name
    atoms = [mol.GetAtomWithIdx(idx) for idx in seq]
    if (any(not atom.GetIsAromatic() or atom.GetFormalCharge() or atom.GetIsotope()
            for atom in atoms)
            or any(nbr.GetAtomicNum() == 1 and nbr.GetIsotope()
                   for atom in atoms for nbr in atom.GetNeighbors())
            or [atom.GetAtomicNum() for atom in atoms] != [7, 7, 6, 6, 6]):
        return parent_name
    core = set(seq)
    pyrrolic = []
    for atom in atoms[:2]:
        outside = [nbr for nbr in atom.GetNeighbors() if nbr.GetIdx() not in core]
        hydrogen = atom.GetTotalNumHs()
        if ((hydrogen == 1 and not outside)
                or (hydrogen == 0 and len(outside) == 1
                    and mol.GetBondBetweenAtoms(atom.GetIdx(), outside[0].GetIdx()).GetBondTypeAsDouble() == 1)):
            pyrrolic.append(atom.GetIdx())
    if len(pyrrolic) != 1:
        return parent_name
    return "1H-pyrazole" if pyrrolic[0] == seq[0] else ""


def _with_ring_branch_atom_stereo_prefix(mol, root_idx: int, blocked_idx: int, prefix: str) -> str:
    if not prefix or _has_locanted_atom_stereo_prefix(prefix):
        return prefix
    from ..stereochemistry import retained_branch_stereo_is_complete
    if retained_branch_stereo_is_complete(
            mol, Chem, prefix, collect_subtree(mol, root_idx, blocked_idx)):
        return prefix
    if _aminoacyloxymethyl_has_complete_stereo(mol, root_idx, blocked_idx, prefix):
        # The generic parent decorator tracks numeric R/S operations. Use a
        # fully numbered component here, rather than mixing its locants with
        # a D/L acyl word or leaving that word unaccounted for upstream.
        from .functional_groups import substituent_from_root
        return substituent_from_root(mol, root_idx, blocked_idx) or prefix
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


def _aminoacyloxymethyl_has_complete_stereo(mol, root_idx, blocked_idx, prefix):
    """Prove alpha configuration already owned by a retained acyl component.

    An acyl-oxy-methyl component has no single chain numbering through its
    carbonyl. Its retained aminoacyl word already supplies alpha configuration;
    a locant from an alternative skeletal-replacement chain cannot be added.
    The methylene is achiral and no other assigned center is admitted here.
    """
    root = mol.GetAtomWithIdx(root_idx)
    if (root.GetAtomicNum() != 6 or root.IsInRing() or root.GetFormalCharge()
            or root.GetIsotope() or root.GetTotalNumHs() != 2):
        return False
    distal = [n for n in root.GetNeighbors() if n.GetIdx() != blocked_idx and n.GetAtomicNum() > 1]
    if len(distal) != 1 or distal[0].GetAtomicNum() != 8:
        return False
    oxygen = distal[0]
    if oxygen.GetFormalCharge() or oxygen.GetDegree() != 2:
        return False
    acyl = [n for n in oxygen.GetNeighbors() if n.GetIdx() != root_idx]
    if len(acyl) != 1 or acyl[0].GetAtomicNum() != 6:
        return False
    carbonyl = acyl[0]
    alpha = [n for n in carbonyl.GetNeighbors() if n.GetAtomicNum() == 6]
    oxo = [b for b in carbonyl.GetBonds()
           if b.GetOtherAtom(carbonyl).GetAtomicNum() == 8 and b.GetBondTypeAsDouble() == 2]
    if len(alpha) != 1 or len(oxo) != 1:
        return False
    atoms = collect_subtree(mol, root_idx, blocked_idx)
    assigned = {i for i, _ in assigned_atom_stereochemistry(mol, Chem, atom_order='idx') if i in atoms}
    if assigned != {alpha[0].GetIdx()}:
        return False
    from .functional_groups import _retained_aminoacyl_prefix
    name = _retained_aminoacyl_prefix(mol, carbonyl.GetIdx(), alpha[0].GetIdx(),
                                    parent_idx=oxygen.GetIdx())
    return bool(name) and prefix == name + 'oxymethyl'


def _has_locanted_atom_stereo_prefix(prefix: str) -> bool:
    # A group can contain both tetrahedral and double-bond descriptors.
    # Its R/S operation is still present when E/Z appears in the same group.
    return bool(re.search(
        r"\((?=[^()]*[RS])(?:\d+[a-z]?'?)?[RSZE]"
        r"(?:,(?:\d+[a-z]?'?)?[RSZE])*\)-", prefix))


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
    if parent_name in {"cyclohexane", "cyclohexan"}:
        from ..stereochemistry import opposite_cyclohexyl_configuration

        relation = opposite_cyclohexyl_configuration(mol, Chem, seq)
        if relation:
            return relation
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
        for atom in (mol.GetAtomWithIdx(idx) for idx in sorted(seq_set))
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
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
        for bond in iter_bonds(mol)
        if bond.GetBeginAtomIdx() in seq_set and bond.GetEndAtomIdx() in seq_set
    ]
    if any(bond.GetBondType() != Chem.BondType.SINGLE for bond in ring_bonds):
        return ""
    from ..stereochemistry import relative_monocycle_configuration
    return relative_monocycle_configuration(mol, Chem, seq)


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
    mol, attachment_idx: int, parent_idx: int, suffix: str, *, decorate_stereo=True
) -> str:
    bond = mol.GetBondBetweenAtoms(attachment_idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return suffix
    host = mol.GetAtomWithIdx(parent_idx)
    acyclic_methine = (not host.IsInRing() and host.GetAtomicNum() == 6
                      and host.GetDegree() == 2 and host.GetTotalNumHs() == 1
                      and all(n.GetAtomicNum() == 6 for n in host.GetNeighbors()))
    if (bond.GetStereo() != Chem.BondStereo.STEREONONE
            and (host.GetAtomicNum() != 6
                 or not (host.IsInRing() or acyclic_methine))):
        # Both cyclic components have an explicit free-valence location.
        # N-hosted and acyclic-hosted legacy branches need their separate
        # local parent/hydrogen-state proof and retain the previous guard.
        return ""
    if bond.GetStereo() != Chem.BondStereo.STEREONONE and decorate_stereo:
        from ..stereochemistry import (
            _indexed_bond_stereochemistry, _rendered_ylidene_junctions,
            _rendered_mapped_branch_bonds,
        )
        branch_atoms = collect_subtree(mol, attachment_idx, parent_idx)
        owned = (set(_rendered_ylidene_junctions(mol, suffix))
                 | set(_rendered_mapped_branch_bonds(mol, suffix)))
        if any({a,b} != {attachment_idx,parent_idx} and {a,b} & branch_atoms
               and (frozenset((a,b)) not in owned
                    or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in (a, b)))
               for a,b,_descriptor in _indexed_bond_stereochemistry(mol, Chem)):
            # This bounded branch owns one junction only. A second E/Z
            # carbon-carbon unit needs a joined local bond map. Configured
            # imines retain their separate parent/junction construction;
            # their parent can already own this same junction descriptor.
            return ""
    atom = mol.GetAtomWithIdx(attachment_idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    source = suffix
    if not suffix.endswith("ylidene"):
        if not suffix.endswith("yl"):
            return ""
        suffix += "idene"
    if bond.GetStereo() == Chem.BondStereo.STEREONONE or not decorate_stereo:
        return suffix
    # P-93.5.1.4.2.1(1)(b): a cyclic ylidene may own its junction's
    # configuration locally. This remains valid when the enclosing parent
    # is several branches away and therefore cannot number either end.
    match = re.search(r"-([1-9][0-9]*[a-z]?'*)-ylidene$", suffix)
    if match is None:
        return ""  # No proved local free-valence locant in this grammar.
    from ..stereochemistry import _indexed_bond_stereochemistry, assigned_bond_stereochemistry
    raw = next((d for a,b,d in _indexed_bond_stereochemistry(mol, Chem)
                       if {a,b} == {attachment_idx,parent_idx}), '')
    descriptor = next((d for a,b,d in assigned_bond_stereochemistry(mol, Chem)
                       if {a,b} == {attachment_idx,parent_idx}), '')
    if descriptor not in {'E', 'Z'}:
        return ""
    locant = match.group(1)
    from ..stereochemistry import apply_ring_branch_junction_descriptor
    prefix = apply_ring_branch_junction_descriptor(
        mol, attachment_idx, parent_idx, source, suffix, locant, descriptor)
    # Retain the actual emitted prefix and original bond identity. The
    # enclosing descriptor writer must not describe that same bond twice.
    ownership = mol_cache_dict(mol, 'rendered_ylidene_junction_stereo')
    ownership[(attachment_idx, parent_idx, prefix)] = (raw, descriptor, locant)
    return prefix


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
    aromatic_view = None
    if _attachment_multi_ring_system_is_carbocyclic(mol, attachment_idx, parent_idx):
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
    prefix = _fused_ring_substituent_prefix_for_view(mol, attachment_idx, parent_idx)
    if ('-one-' in prefix or 'tetrahydropterin-' in prefix) and prefix.endswith('-yl'):
        from ..component_mapping import _complete_rooted_ring

        scope = frozenset(collect_subtree(mol, attachment_idx, parent_idx))
        if parent_idx not in scope and len(scope) <= 40:
            rooted = _complete_rooted_ring(mol, Chem, attachment_idx, scope)
            if 'oxo' in rooted and '-one-' not in rooted and 'pterin-' not in rooted:
                return rooted
    return prefix


def _fused_ring_substituent_prefix_for_view(
    mol,
    attachment_idx: int,
    parent_idx: int,
    *,
    carbocyclic_only: bool = False,
) -> str:
    if not _attachment_in_multi_ring_system(mol, attachment_idx, parent_idx):
        return ""
    from ..mapped_ring_parents import ring_systems

    # A template match must contain every ring it touches. Following the
    # overlapping rings from the attachment would also include the blocked
    # parent in this case, so neither template search can supply a prefix.
    if any(attachment_idx in core and parent_idx in core for core in ring_systems(mol)):
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
    if not _should_try_broad_generated_fused_substituent_templates(mol):
        return ""
    broad_generated_templates = _candidate_generated_fused_templates(mol, Chem, graph_filter=False)
    if carbocyclic_only:
        broad_generated_templates = _carbocyclic_fused_templates(broad_generated_templates)
    generated = _fused_ring_substituent_prefix_from_templates(
        mol, attachment_idx, parent_idx, broad_generated_templates, generated=True
    ) or _hydro_fused_ring_substituent_prefix_from_templates(mol, attachment_idx, parent_idx, broad_generated_templates)
    return generated


def _should_try_broad_generated_fused_substituent_templates(mol) -> bool:
    return bool(getattr(mol, "HasProp", lambda _name: False)("_nispo_allow_fragment_substituent_fallback"))


_CARBOCYCLIC_FUSED_TEMPLATE_CACHE: dict[int, tuple[dict, ...]] = {}


def _carbocyclic_fused_templates(templates: tuple[dict, ...]) -> tuple[dict, ...]:
    cached = _CARBOCYCLIC_FUSED_TEMPLATE_CACHE.get(id(templates))
    if cached is not None:
        return cached
    filtered = intern_template_tuple(template for template in templates if _template_core_is_carbocyclic(template))
    _CARBOCYCLIC_FUSED_TEMPLATE_CACHE[id(templates)] = filtered
    return filtered


def _template_core_is_carbocyclic(template: dict) -> bool:
    core = template_record_mol(template)
    return all(atom.GetSymbol() == "C" for atom in iter_atoms(core))


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
    return bool(_attachment_multi_ring_system(mol, attachment_idx, parent_idx))


def _attachment_multi_ring_system_is_carbocyclic(mol, attachment_idx: int, parent_idx: int) -> bool:
    system = _attachment_multi_ring_system(mol, attachment_idx, parent_idx)
    if not system:
        return False
    return all(mol.GetAtomWithIdx(idx).GetSymbol() == "C" for idx in system)


def _attachment_multi_ring_system(mol, attachment_idx: int, parent_idx: int) -> frozenset[int]:
    cache = mol_cache_dict(mol, "attachment_multi_ring_system_cache")
    key = (attachment_idx, parent_idx)
    cached = cache.get(key)
    if cached is not None:
        return cached
    containing = [frozenset(ring) for ring in _rings_for_attachment_excluding_parent(mol, attachment_idx, parent_idx)]
    if not containing:
        cache[key] = frozenset()
        return frozenset()
    rings = [frozenset(ring) for ring in mol.GetRingInfo().AtomRings() if parent_idx not in ring]
    if not rings:
        cache[key] = frozenset()
        return frozenset()
    system = set(containing[0])
    changed = True
    while changed:
        changed = False
        for ring in rings:
            if system & ring and not ring <= system:
                system.update(ring)
                changed = True
    if sum(1 for ring in rings if ring <= system) < 2:
        cache[key] = frozenset()
        return frozenset()
    result = frozenset(system)
    cache[key] = result
    return result


def _fused_ring_substituent_prefix_from_templates(
    mol, attachment_idx: int, parent_idx: int, templates: tuple[dict, ...], generated: bool
) -> str:
    from ..numbered_bicyclic_parents import has_proved_bicyclic_numbering
    from ..preferred_fused import (
        preferred_fused_name_locants,
        preferred_fused_substituent_prefix_records,
    )

    best = ""
    best_key = None
    for source_template in _candidate_templates(mol, templates):
        if _template_requires_systematic_substituent_name(source_template):
            continue
        core = template_record_mol(source_template)
        aromatic_nitrogen = None
        numbered = has_proved_bicyclic_numbering(source_template)
        numbered_options = []
        for match in _substruct_matches(mol, core):
            if attachment_idx not in match or parent_idx in match:
                continue
            if match_has_extra_core_bonds(mol, core, match):
                continue
            if aromatic_nitrogen is None:
                aromatic_nitrogen = any(atom.GetAtomicNum() == 7 and atom.GetIsAromatic()
                                        for atom in iter_atoms(core))
            # Without an aromatic query N the hydrogen validator always
            # returns true; its full matched-core boundary scan is unnecessary.
            if aromatic_nitrogen and not match_preserves_aromatic_nitrogen_hydrogens(mol, core, match):
                continue
            core_set = set(match)
            if ring_system_extends_core(mol, core_set):
                continue
            preferred_name, preferred_locants = preferred_fused_name_locants(
                mol, source_template, match, substituent=True,
                attachment_idx=attachment_idx, parent_idx=parent_idx,
            )
            template = source_template
            if preferred_name != template_record_name(source_template):
                template = {**source_template, "name": preferred_name, "locants": preferred_locants,
                            "intrinsic_prefixes": preferred_fused_substituent_prefix_records(source_template)}
            reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
            template_idx = reverse[attachment_idx]
            locant = template_record_locants(template).get(template_idx)
            if not locant or locant.endswith("a"):
                continue
            # A cation suffix requires a positive original atom in this core.
            # Observe current native charges at every match, retaining the
            # complete charge writer for every potentially eligible core.
            suffix = (_charged_fused_substituent_suffix(mol, template, match, reverse, locant)
                      if any(mol.GetAtomWithIdx(i).GetFormalCharge() > 0 for i in match) else "")
            if not suffix:
                suffix = _fused_substituent_suffix(template_record_name(template), locant)
            if not suffix:
                continue
            if suffix == "purin-9-yl":
                from ..ring_template_presentation import purine_attachment_hydride

                hydride = purine_attachment_hydride(mol, match, template_record_locants(template),
                                                    locant)
                if hydride:
                    suffix = f"{hydride}-{locant}-yl"
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
            if hydro_prefix_text is None:
                # A partial template map cannot locate a required hydrogen
                # operation. Try a fully mapped template or graph route.
                continue
            if hydro_prefix_text and not _fused_substituent_parent_has_hydro_segment(suffix):
                prefix_text = hydro_prefix_text
            name = _combine_fused_substituent_prefix(prefix_text, suffix)
            stereo_prefix = _fused_substituent_stereo_prefix(mol, template, match)
            if stereo_prefix:
                name = f"({stereo_prefix})-{name}"
            if "-one-" in suffix:
                from ..ring_template_presentation import fused_lactam_substituent_presentation

                name = fused_lactam_substituent_presentation(
                    mol, template_record_name(template), match, template_record_locants(template),
                    locant, prefix_records, stereo_prefix,
                ) or name
            from ..stereochemistry import apply_mapped_branch_bond_stereochemistry
            name = apply_mapped_branch_bond_stereochemistry(
                mol, Chem, name,
                tuple((idx, str(template_record_locants(template).get(t_idx, "")))
                      for t_idx, idx in enumerate(match)),
                external_parent=parent_idx, attachment_idx=attachment_idx)
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
                parent_idx=parent_idx,
                prefix_records=prefix_records,
            )
            if numbered:
                numbering_key = (int(locant), tuple(_locant_sort_key(loc) for loc in prefix_locants))
                numbered_options.append((numbering_key, key, name))
                continue
            if best_key is None or (key < best_key if generated else key > best_key):
                best_key = key
                best = name
        if numbered_options:
            # Close the numbering of this one parent before comparing it to
            # other parent vocabularies. All decorations use the same map.
            lowest = min(row[0] for row in numbered_options)
            options = [row for row in numbered_options if row[0] == lowest]
            _, key, name = (min(options, key=lambda row: row[1]) if generated
                            else max(options, key=lambda row: row[1]))
            if best_key is None or (key < best_key if generated else key > best_key):
                best_key, best = key, name
    return best


def _template_requires_systematic_substituent_name(template: dict) -> bool:
    return template_record_name(template) in _SYSTEMATIC_ONLY_SUBSTITUENT_TEMPLATE_NAMES


_SYSTEMATIC_ONLY_SUBSTITUENT_TEMPLATE_NAMES = frozenset(
    {
        # This suffix-bearing scaffold is a parent template. Radical names
        # need oxo-prefix/H-state conversion, so retain the graph branch route.
        "1,5,6,7-tetrahydro-4H-pyrrolo[3,2-c]pyridin-4-one",
        # OPSIN accepts the bare retained acetal scaffold, but not its
        # locanted substituent forms. Let the graph renderer emit
        # 1,4-dioxaspiro[4.5]decane substituents instead.
        "cyclohexanone ethylene ketal",
        # OPSIN accepts dibenzofuran parents, but some locanted fused-substituent
        # aliases are orientation-fragile in larger fused systems. Keep the
        # retained parent available and use the graph renderer for substituents.
        "dibenzofuran",
        # The bare token and the fusion-derived radical use different locants
        # and saturation in OPSIN. Use the graph renderer for attachments.
        "benzocyclobutene",
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
    *,
    parent_idx: int | None = None,
    prefix_records=(),
) -> tuple:
    template_name = template_record_name(template)
    if generated:
        return (_generated_fusion_preference(template_name), -len(match), locant, name)
    h_state_key = _retained_fused_substituent_h_state_key(mol, template, match)
    if (template_name in {'naphthalene', 'naphthalen'} and parent_idx is not None
            and _proved_naphthalene_substituent_numbering(
                mol, template, match, locant, parent_idx, prefix_records)):
        # Keep the existing inter-template rank and compare the complete
        # graph-matched orientations by the free-valence locant (P-14.4(c)).
        retained_rank = int(bool(template.get('preferred_iupac_parent', False)
                                 or template_name in _LOW_LOCANT_RETAINED_FUSED_SUBSTITUENT_PARENTS))
        return (len(match), h_state_key, retained_rank,
                _low_locant_retained_fused_substituent_key(locant), (0,), (0,), name)
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
    if template.get("preferred_iupac_parent", False) or template_name in _LOW_LOCANT_RETAINED_FUSED_SUBSTITUENT_PARENTS:
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


def _proved_naphthalene_substituent_numbering(mol, template, match, locant, parent, records):
    """Prove all ring symmetries and ownership of its original branches."""
    core = set(match)
    if len(core) != 10 or not str(locant).isdigit() or ring_system_extends_core(mol, core):
        return False
    mapping = template_record_locants(template)
    roots = [match[i] for i in range(len(match)) if mapping.get(i) == locant]
    if len(roots) != 1:
        return False
    for idx in core:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or not atom.GetIsAromatic()
                or atom.GetFormalCharge() or atom.GetIsotope()
                or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                or any(n.GetAtomicNum() == 1 and n.GetIsotope() for n in atom.GetNeighbors())):
            return False
    annular = [bond for bond in iter_bonds(mol)
               if bond.GetBeginAtomIdx() in core and bond.GetEndAtomIdx() in core]
    if (len(annular) != 11 or any(not bond.GetIsAromatic()
            or bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in annular)):
        return False
    if any(any(atom.GetIdx() in core for atom in group.GetAtoms())
           for group in mol.GetStereoGroups()):
        return False
    root = roots[0]
    bond = mol.GetBondBetweenAtoms(root, parent)
    if bond is None or bond.GetBondTypeAsDouble() != 1:
        return False
    scope = frozenset(collect_subtree(mol, root, parent))
    from ..component_mapping import has_annotations, mapped_branch
    if has_annotations(mol, scope):
        return False
    if any(any(atom.GetIdx() in scope for atom in group.GetAtoms())
           for group in mol.GetStereoGroups()):
        return False
    cuts = {(i, n.GetIdx()) for i in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetIdx() not in scope and (n.GetAtomicNum() > 1 or n.GetIsotope())}
    if cuts != {(root, parent)}:
        return False
    # The full four-element naphthalene symmetry orbit must survive template
    # matching; a globally truncated match list cannot prove the minimum.
    junctions = {i for i in core if sum(n.GetIdx() in core
                  for n in mol.GetAtomWithIdx(i).GetNeighbors()) == 3}
    if len(junctions) != 2 or root in junctions:
        return False
    alpha = any(n.GetIdx() in junctions for n in mol.GetAtomWithIdx(root).GetNeighbors())
    expected = {'1', '4', '5', '8'} if alpha else {'2', '3', '6', '7'}
    matches = {tuple(candidate) for candidate in _substruct_matches(mol, template_record_mol(template))
               if set(candidate) == core}
    if len(matches) != 4:
        return False
    observed = {mapping.get(candidate.index(root)) for candidate in matches}
    if observed != expected:
        return False
    # Prefix names and locants must be produced from the same original-atom
    # orientation as the selected ring, with disjoint single-cut ownership.
    from collections import Counter
    reverse = {mol_idx: i for i, mol_idx in enumerate(match)}
    actual = Counter()
    owned = set(core)
    for host in core:
        for nbr in mol.GetAtomWithIdx(host).GetNeighbors():
            child_root = nbr.GetIdx()
            if child_root in core or child_root == parent or nbr.GetAtomicNum() == 1:
                continue
            child = collect_subtree(mol, child_root, host)
            if child & owned or not child <= scope:
                return False
            child_cuts = {(i, n.GetIdx()) for i in child for n in mol.GetAtomWithIdx(i).GetNeighbors()
                          if n.GetIdx() not in child and (n.GetAtomicNum() > 1 or n.GetIsotope())}
            if child_cuts != {(child_root, host)}:
                return False
            text, mapped = mapped_branch(mol, Chem, child_root, host)
            if not text or set(mapped) != child:
                return False
            actual[(mapping.get(reverse[host]), text, 'C')] += 1
            owned.update(child)
    return owned == set(scope) and actual == Counter(records)


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
        "1H-1,2,3-benzotriazole",
        "2H-1,2,3-benzotriazole",
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
    if template_name in {"1H-benzotriazole", "1H-1,2,3-benzotriazole"}:
        return 2
    if template_name in {"2H-benzotriazole", "2H-1,2,3-benzotriazole"}:
        return 1
    return 0


def _retained_fused_substituent_default_locant_key(locant: str) -> tuple:
    if locant.isdigit():
        return (0, int(locant))
    return (-1, locant)


@lru_cache(maxsize=8192)
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
    from ..stereochemistry import retained_stereo_is_covered
    matches = tuple(tuple(match) for match in mol.GetSubstructMatches(query, uniquify=False)
                    if retained_stereo_is_covered(mol, query, match))
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
    prefix_records = list(template.get("intrinsic_prefixes", ()))
    intrinsic = {(locant, prefix) for locant, prefix, _symbol in prefix_records}
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
            if (core_loc, prefix) in intrinsic and nbr.GetAtomicNum() == 8 and nbr.GetDegree() == 1:
                continue
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
) -> str | None:
    parent_name = template_record_name(template)
    if not _template_is_fully_aromatic(template_record_mol(template)):
        return ""
    if _fused_substituent_parent_has_hydro_segment(parent_name):
        return ""
    # P-58.2.3: an exocyclic imino operation consumes two annular
    # hydrogens just as an oxo operation does. The original double bond,
    # rather than the spelling of a compound prefix, establishes its role.
    imino_locants = {
        template_record_locants(template).get(template_idx)
        for idx, template_idx in reverse.items()
        for bond in mol.GetAtomWithIdx(idx).GetBonds()
        if bond.GetOtherAtomIdx(idx) not in reverse
        and bond.GetBondTypeAsDouble() == 2
        and bond.GetOtherAtom(mol.GetAtomWithIdx(idx)).GetAtomicNum() == 7
        and not bond.GetOtherAtom(mol.GetAtomWithIdx(idx)).GetFormalCharge()
    }
    chalcogen_records = [
        (locant, prefix)
        for locant, prefix, _symbol in records
        if (prefix in {"oxo", "thioxo"} or locant in imino_locants) and locant.isdigit()
    ]
    if len(chalcogen_records) > 1 and imino_locants:
        return _fused_multi_prefix_hydro(mol, template, reverse, records)
    if len(chalcogen_records) != 1:
        return ""
    h_locs = _fused_substituent_indicated_hydrogen_locants(mol, template, reverse)
    if h_locs is None:
        return None
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


def _fused_multi_prefix_hydro(mol, template, reverse, prefix_records) -> str | None:
    """Cite proved annular reduction, or decline an unsupported template."""
    from ..hydrogen_accommodation import (
        annular_pi_bonds, joint_hydrogen_states, unique_capped_pi_graph,
    )

    query = template_record_mol(template)
    locants=template_record_locants(template)
    if len(reverse)!=query.GetNumAtoms() or len(reverse)>10:
        return None
    match=tuple(i for i,_ in sorted(reverse.items(),key=lambda x:x[1]))
    numbers=tuple(a.GetAtomicNum() for a in query.GetAtoms())
    if any(n not in {6,7,8,16} for n in numbers):
        return None
    if any(mol.GetAtomWithIdx(i).GetFormalCharge() or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in match):
        return None
    edges=tuple((b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in query.GetBonds())
    reference=tuple(a.GetTotalNumHs() for a in query.GetAtoms())
    capped=tuple(mol.GetAtomWithIdx(i).GetTotalNumHs()+sum(
        b.GetBondTypeAsDouble() for b in mol.GetAtomWithIdx(i).GetBonds()
        if b.GetOtherAtomIdx(i) not in reverse) for i in match)
    actual_pi=annular_pi_bonds(mol,Chem,match)
    states=joint_hydrogen_states(numbers,edges,reference,capped,(0,)*len(capped),actual_pi)
    if not states or not unique_capped_pi_graph(numbers,edges,tuple(map(int,capped))):
        return None
    if not states[0].hydro:
        return ""
    sites=tuple(locants.get(i) for i in states[0].hydro)
    if any(not _is_fused_indicated_h_locant(site) for site in sites) or len(sites) not in MULT or len(sites)<2:
        return None
    hydro=f"{','.join(sorted(sites,key=_locant_sort_key))}-{MULT[len(sites)]}hydro"
    return f"{_fused_substituent_prefix_text_from_records(prefix_records)}-{hydro}"


def _fused_substituent_indicated_hydrogen_locants(
    mol,
    template: dict,
    reverse: dict[int, int],
) -> tuple[str, ...] | None:
    h_locs = []
    for mol_idx, template_idx in reverse.items():
        atom = mol.GetAtomWithIdx(mol_idx)
        locant = template_record_locants(template).get(template_idx)
        if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0:
            if not _is_fused_indicated_h_locant(locant):
                return None
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
        if not re.fullmatch(r"\d+[a-z]?'*", locant):
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
        from ..parent_selection import locant_key
        pieces.append((locant_key(locant), text))
    return ",".join(text for _locant, text in sorted(pieces))


@lru_cache(maxsize=None)
def _topology_query(template_mol):
    query = Chem.RWMol()
    for atom in iter_atoms(template_mol):
        query.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    for bond in iter_bonds(template_mol):
        query.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.UNSPECIFIED)
    return query.GetMol()


@lru_cache(maxsize=None)
def _template_is_fully_aromatic(template_mol) -> bool:
    return all(atom.GetIsAromatic() for atom in iter_atoms(template_mol))


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
    # P-16.2.4.1 and P-31.2.3.3.2: a hydro prefix joins directly to an
    # alphabetic parent component (e.g. tetrahydroimidazo...), whereas a
    # following indicated-H, numerical or bracketed locant needs a hyphen.
    parent_name = template_record_name(template)
    # Hydrogenation does not erase the indicated H of the remaining
    # mancude portion. Bind it to the source N before forming the -yl group.
    indicated = sorted({template_record_locants(template).get(i, '')
                        for i, idx in enumerate(match)
                        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7
                        and mol.GetAtomWithIdx(idx).GetIsAromatic()
                        and mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0},
                       key=_locant_sort_key)
    if indicated and all(indicated):
        parent_name = re.sub(r'^(?:\d+[a-z]?H,)*\d+[a-z]?H-', '', parent_name)
        parent_name = ','.join(loc + 'H' for loc in indicated) + '-' + parent_name
    separate = (not parent_name[:1].isalpha()
                or bool(re.match(r"(?:s|as|v|sym|unsym)-", parent_name, re.IGNORECASE)))
    name = f"{locants}-{hydro}{'-' if separate else ''}{parent_name}"
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
    if stem is None:
        stem = parent_name if parent_name.endswith(CARBONYL_SUFFIX_ENDINGS) else parent_name.removesuffix("e")
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
    # P-57.1.5.1/.3 and P-29.3.4: the traditional -yl words such as
    # naphthyl and quinolyl are general-nomenclature forms. For these
    # mapped retained hydrides use the complete parent and the suffix
    # attachment locant; this operation retains the template's atom map.
    stems = {
        "naphthalene": "naphthalen",
        "quinoline": "quinolin",
        "isoquinoline": "isoquinolin",
        "quinoxaline": "quinoxalin",
        "benzothiazole": "1,3-benzothiazol",
        "benzoxazole": "1,3-benzoxazol",
    }
    stem = stems.get(parent_name)
    return f"{stem}-{locant}-yl" if stem else ""


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
        from ..api import _IN_DIRECT_NAMING, _engine
        from ..naming_context import MoleculeContext
        from ..rule_engine import (
            _RING_FRAGMENT_PREFIXES,
            _first_valid_prepared_candidate,
            _prepare_candidates,
        )

        frag_mol.SetProp("_nispo_fragment_substituent_view", "1")
        frag_mol.SetProp("_nispo_fragment_substituent_depth", str(depth + 1))
        prefixes = _RING_FRAGMENT_PREFIXES.get() if _IN_DIRECT_NAMING.get() else None
        cache_key = None
        if prefixes is not None:
            # Share fragment-local text across copies with the same native
            # state and attachment. Callers still render source-graph stereo.
            try:
                cache_key = (frag_mol.ToBinary(Chem.PropertyPickleOptions.AllProps), attachment,
                             fast_carbon_ring_numbering_enabled())
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass
            if cache_key is not None and cache_key in prefixes:
                return prefixes[cache_key]
        ctx = MoleculeContext(frag_mol, Chem)
        selected = _first_valid_prepared_candidate(
            frag_mol,
            Chem,
            ctx,
            _engine()._rules,
            max_length=sys.maxsize,
            output_max_length=sys.maxsize,
        )
        prepared = ()
        if selected is None or _fragment_substituent_candidate_needs_retained_selection(selected):
            prepared = _prepare_candidates(
                frag_mol,
                Chem,
                ctx,
                _engine()._rules,
                max_length=sys.maxsize,
                output_max_length=sys.maxsize,
            )
    except Exception:
        return ""
    if selected is None and not prepared:
        prefix = ""
    else:
        if selected is None:
            selected = _fragment_substituent_best_prepared_candidate(prepared, attachment)
        elif prepared:
            selected = _fragment_substituent_best_prepared_candidate(prepared, attachment, selected) or selected
        locant = _candidate_atom_locant(selected.candidate, attachment)
        prefix = _parent_name_to_substituent_prefix(selected.output_name or selected.name, locant) if locant else ""
    if cache_key is not None and len(prefixes) < _RING_FRAGMENT_PREFIX_CACHE_LIMIT:
        prefixes[cache_key] = prefix
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
    from ..mapped_ring_parents import capped_subtree_fragment
    return capped_subtree_fragment(mol, Chem, atoms)


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
    if name.endswith('amine') and re.fullmatch(r"\d+[a-z]?'*", locant):
        # A skeletal free valence cannot be appended to the amine suffix.
        # It needs a separately constructed amino-prefix parent and map.
        return ''
    if name == "benzene":
        return "phenyl"
    if name.endswith("e"):
        return f"{name[:-1]}-{locant}-yl"
    return f"{name}-{locant}-yl"
