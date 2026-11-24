from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from ..molecule_iteration import iter_atoms
from ..constants import HALO, MULT, ROOTS
from ..charge_valence import anionic_oxygen_prefix
from ..formatting import _organyl_to_oxy, alkyl_to_alkoxy as _alkyl_to_alkoxy, format_prefixes
from ..molecule_cache import mol_cache_dict
from ..substituent_resource_rules import resource_substituent_match as _resource_substituent_match
from ..template_access import template_record_name
from .functional_groups import (
    _carbonyl_substituent_prefix,
    _is_azido_attachment,
    _is_diazo_attachment,
    _is_diazonio_attachment,
    _is_isocyano_nitrogen,
    _is_nitrile_carbon,
    _is_nitro_nitrogen,
    _is_nitroso_attachment,
    _is_tbdms_group,
    _is_trimethylsilyl_group,
    _phosphorus_prefix,
    _sulfur_prefix,
    substituent_from_root,
)
from .functional_suffixes import suffix_prefix_term, suffix_supported_term
from .ring_substituents import _ring_substituent_prefix


def _branch_rules():
    from .. import branch_rules

    return branch_rules


def branch_prefix(
    mol,
    idx: int,
    parent_idx: int,
    *,
    assume_simple_external_prefix_miss: bool = False,
) -> str:
    return _branch_rules().branch_prefix(
        mol,
        idx,
        parent_idx,
        assume_simple_external_prefix_miss=assume_simple_external_prefix_miss,
    )


def _alkylidene_prefix(mol, carbon_idx: int, parent_idx: int) -> str:
    return _branch_rules()._alkylidene_prefix(mol, carbon_idx, parent_idx)


def _alkynyl_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._alkynyl_substituent_prefix(mol, idx, parent_idx)


def _enal_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._enal_substituent_prefix(mol, idx, parent_idx)


def _enone_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._enone_substituent_prefix(mol, idx, parent_idx)


def _haloethenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._haloethenyl_prefix(mol, idx, parent_idx)


def _ketenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._ketenyl_prefix(mol, idx, parent_idx)


def _linear_carbon_chain(mol, start_idx: int, blocked_idx: int, max_len: int) -> tuple[int, ...]:
    return _branch_rules()._linear_carbon_chain(mol, start_idx, blocked_idx, max_len)


def _nitro_branch_predicate(mol, idx: int, parent_idx: int) -> bool:
    return _branch_rules()._nitro_branch_predicate(mol, idx, parent_idx)

def _oxygen_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetIsotope():
        return ""
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    parent = mol.GetAtomWithIdx(parent_idx)
    heavy = _nonplain_hydrogen_neighbor_indices(atom, parent_idx)
    if atom.GetFormalCharge() < 0 and parent.GetSymbol() == "N" and parent.GetFormalCharge() > 0:
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            return "oxido"
    if atom.GetFormalCharge() < 0 and not heavy:
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            return anionic_oxygen_prefix(parent)
    if atom.GetFormalCharge() > 0 and not heavy:
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            return "oxidaniumyl"
    if atom.GetFormalCharge() > 0 and len(heavy) == 1:
        ylidene_oxidanium = _ylidene_oxidaniumyl_prefix(mol, heavy[0], idx)
        if ylidene_oxidanium:
            return ylidene_oxidanium
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        return suffix_prefix_term("one", "oxo")
    if atom.GetFormalCharge() != 0:
        return ""
    if not heavy:
        return "hydroxy"
    if len(heavy) == 1:
        alcohol = mol.GetAtomWithIdx(heavy[0])
        if alcohol.GetAtomicNum() == 6 and not alcohol.IsInRing() and alcohol.GetChiralTag():
            from ..component_mapping import mapped_branch

            organyl, owned = mapped_branch(mol, Chem, heavy[0], idx)
            if organyl and heavy[0] in owned and not owned & {idx, parent_idx}:
                return _alkyl_to_alkoxy(organyl)
        if _is_nitro_nitrogen(mol, heavy[0]):
            return "nitrooxy"
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "H":
            return _isotopic_hydrogen_oxy_prefix(mol.GetAtomWithIdx(heavy[0]))
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() in HALO:
            from .functional_groups import _is_terminal_halogen_ligand
            if _is_terminal_halogen_ligand(mol, heavy[0], idx):
                return HALO[mol.GetAtomWithIdx(heavy[0]).GetSymbol()] + "oxy"
        carbonyl = _carbonyl_substituent_prefix(mol, heavy[0], idx)
        if carbonyl:
            from ..carboxamide_prefixes import mapped_carbamoyl_linker
            carbamoyloxy = mapped_carbamoyl_linker(mol, Chem, idx, parent_idx, (carbonyl,))
            if carbamoyloxy is not None:
                return carbamoyloxy[0]
            # The complete acyl unit belongs to this oxygen, including when
            # its preferred spelling later introduces an enclosure.  Cite
            # that boundary now so another parent prefix cannot enter it.
            return "(" + carbonyl + ")oxy"
        glycosyl = _glycosyloxy_prefix(mol, heavy[0], idx)
        if glycosyl:
            return glycosyl
        protected = _protected_alkoxy_prefix(mol, heavy[0], idx)
        if protected:
            return protected
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "S":
            if parent.GetSymbol() not in {"Si", "Ge", "Sn"}:
                sulfur_oxy = _sulfur_oxy_alias(mol, heavy[0], idx)
                if sulfur_oxy:
                    return sulfur_oxy
            sulfur = _sulfur_prefix(mol, heavy[0], idx)
            if (
                sulfur.endswith(("sulfonyl", "sulfonothioyl"))
                and _sulfur_has_ring_carbon_ligand(mol, heavy[0], idx)
                and not _sulfur_has_nitroaryl_ligand(mol, heavy[0], idx)
            ):
                return "(" + sulfur + ")oxy"
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "P":
            phosphorus = _phosphorus_oxy_prefix(mol, heavy[0], idx)
            if phosphorus:
                return phosphorus
            phosphorus = _phosphorus_prefix(mol, heavy[0], idx)
            if phosphorus:
                return phosphorus + "oxy"
        if mol.GetAtomWithIdx(heavy[0]).IsInRing():
            ring = _ring_substituent_prefix(mol, heavy[0], idx)
            if not ring:
                return ""
            if _component_has_unsupported_annotation(mol, heavy[0], idx):
                return ""
            phenoxy = _phenoxy_substituent_prefix(ring)
            if phenoxy:
                return phenoxy
            return _organyloxy_component(
                ring,
                attachment_locanted=bool(re.search(r"-[0-9]+-yl$", ring)),
            )
        retained = _retained_oxygen_resource_prefix(mol, idx, parent_idx)
        if retained:
            return retained
        ring_methoxy = _ring_methoxy_prefix(mol, heavy[0], idx)
        if ring_methoxy:
            return ring_methoxy
        organyl = _acyclic_organyl_name(mol, heavy[0], idx)
        if organyl is not None:
            alk, attachment_locanted, parent_chain_length = organyl
            return _organyloxy_component(
                alk,
                attachment_locanted=attachment_locanted,
                parent_chain_length=parent_chain_length,
            )
        # If the short carbon parent contains mapped heteroatom or ring
        # substituents, the pure-carbon shortcut above cannot prove its
        # length.  The graph-bound writer contracts only a complete terminal
        # C1--C4 component and preserves the existing child's annotations.
        from .contracted_alkoxy import mapped_contracted_alkoxy
        contracted = mapped_contracted_alkoxy(mol, Chem, idx, parent_idx)
        if contracted is not None:
            return contracted[0]
        # Keep the prior mapped-component route for annotated isotope-bearing
        # carbon organyls whose established descriptor conversion is bounded.
        if _supports_legacy_isotope_organyl(mol, heavy[0], idx):
            alk = _branch_rules().branch_prefix(mol, heavy[0], idx)
            return _alkyl_to_alkoxy(alk)
        # Heteroatom-rich and unsaturated components retain their previous
        # mapped writer route; this graph writer makes no new conformance claim
        # for those classes.
        if _contains_nonalkyl_component(mol, heavy[0], idx):
            alk = _branch_rules().branch_prefix(mol, heavy[0], idx)
            if not alk:
                alk = substituent_from_root(mol, heavy[0], idx)
            return _alkyl_to_alkoxy(alk)
        return ""
    return ""


def _ring_methoxy_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    """Build a ring-substituted methoxy at its mapped methyl/oxygen boundary.

    P-57.4 gives (4-chlorophenyl)methoxy as the preferred construction;
    unsubstituted benzyl is retained and continues through its own route.
    """
    carbon = mol.GetAtomWithIdx(carbon_idx)
    bond = mol.GetBondBetweenAtoms(carbon_idx, oxygen_idx)
    if (
        carbon.GetAtomicNum() != 6 or carbon.IsInRing() or carbon.GetFormalCharge()
        or carbon.GetIsotope() or carbon.GetNumRadicalElectrons()
        or carbon.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        or carbon.GetTotalNumHs(includeNeighbors=True) != 2
        or bond is None or bond.GetBondType() != Chem.BondType.SINGLE
    ):
        return ""
    others = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() > 1 and a.GetIdx() != oxygen_idx]
    if len(others) != 1 or not others[0].IsInRing():
        return ""
    ring_idx = others[0].GetIdx()
    if mol.GetBondBetweenAtoms(carbon_idx, ring_idx).GetBondType() != Chem.BondType.SINGLE:
        return ""
    if _component_has_unsupported_annotation(mol, carbon_idx, oxygen_idx):
        return ""
    ring = _ring_substituent_prefix(mol, ring_idx, carbon_idx)
    if not ring or ring == "phenyl":
        return ""
    from ..formatting import enclose_name_unit
    return (enclose_name_unit(ring) if "-" in ring or "(" in ring or ")" in ring else ring) + "methoxy"


def _organyloxy_component(
    organyl: str,
    *,
    attachment_locanted: bool = False,
    parent_chain_length: int | None = None,
) -> str:
    """Form an R–O substituent prefix from its mapped organyl component.

    P-63.2.2.1.1 adds ``oxy`` to the organyl prefix. A few short or common
    components have retained contracted forms; locanted attachment names
    keep their own enclosure before ``oxy``. This consumes the graph-derived
    organyl name, so chain length and branching are determined upstream from
    the mapped component rather than inferred from a fixed name list.
    """
    if not organyl:
        return ""
    return _organyl_to_oxy(
        organyl,
        attachment_locanted=attachment_locanted,
        parent_chain_length=parent_chain_length,
    )


def _supports_legacy_isotope_organyl(mol, start_idx: int, blocked_idx: int) -> bool:
    atoms = _mapped_component_atoms(mol, start_idx, blocked_idx)
    return bool(atoms) and all(
        mol.GetAtomWithIdx(index).GetSymbol() == "C"
        and mol.GetAtomWithIdx(index).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
        and not any(
            neighbor.GetSymbol() == "H" and neighbor.GetIsotope()
            for neighbor in mol.GetAtomWithIdx(index).GetNeighbors()
        )
        for index in atoms
    ) and any(mol.GetAtomWithIdx(index).GetIsotope() for index in atoms)


def _contains_nonalkyl_component(mol, start_idx: int, blocked_idx: int) -> bool:
    atoms = _mapped_component_atoms(mol, start_idx, blocked_idx)
    return any(
        mol.GetAtomWithIdx(index).GetSymbol() != "C"
        or any(
            bond.GetBondType() != Chem.BondType.SINGLE
            for bond in mol.GetAtomWithIdx(index).GetBonds()
            if bond.GetOtherAtomIdx(index) in atoms
        )
        for index in atoms
    )


def _mapped_component_atoms(mol, start_idx: int, blocked_idx: int) -> set[int]:
    atoms = set()
    pending = [start_idx]
    while pending:
        current = pending.pop()
        if current in atoms:
            continue
        atoms.add(current)
        pending.extend(
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors()
            if neighbor.GetIdx() != blocked_idx and neighbor.GetAtomicNum() > 1
        )
    return atoms


def _acyclic_organyl_name(
    mol, start_idx: int, blocked_idx: int
) -> tuple[str, bool, int] | None:
    """Name an unannotated saturated carbon tree from its cut graph.

    P-46 compares chains containing the free valence by length, free-valence
    locant, number of substituents, and their locants/alphanumerical order.
    Off-chain trees are named by the same construction. The graph boundary
    excludes annotations whose transfer this writer cannot yet prove.
    """
    atoms: set[int] = set()
    pending = [start_idx]
    while pending:
        current = pending.pop()
        if current in atoms:
            continue
        atoms.add(current)
        atom = mol.GetAtomWithIdx(current)
        if (atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0
                or atom.GetNumRadicalElectrons() or atom.GetIsotope()
                or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                or atom.IsInRing()):
            return None
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() == blocked_idx:
                continue
            if neighbor.GetAtomicNum() == 1:
                if neighbor.GetIsotope():
                    return None
                continue
            if neighbor.GetSymbol() != "C":
                return None
            bond = mol.GetBondBetweenAtoms(current, neighbor.GetIdx())
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            pending.append(neighbor.GetIdx())
    adjacency = {
        atom_idx: {
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            if neighbor.GetIdx() in atoms
        }
        for atom_idx in atoms
    }
    if sum(map(len, adjacency.values())) // 2 != len(atoms) - 1:
        return None
    root_degree = len(adjacency[start_idx])
    stem = ROOTS.get(len(atoms))
    if stem is None:
        return None
    if root_degree == 0:
        return stem + "yl", False, len(atoms)
    if root_degree == 3 and len(atoms) == 4 and all(
        len(adjacency[atom_idx]) == 1 for atom_idx in atoms if atom_idx != start_idx
    ):
        return "tert-butyl", False, 3

    from ..formatting import prefix_alphabetization_key

    leaves = [i for i, neighbors in adjacency.items() if len(neighbors) == 1]
    paths = []
    if root_degree == 1:
        # P-29.3.2.1: terminal free valence, numbered from that end.
        paths = [tuple(Chem.rdmolops.GetShortestPath(mol, start_idx, leaf))
                 for leaf in leaves if leaf != start_idx]
    else:
        for pos, left in enumerate(leaves):
            for right in leaves[pos + 1:]:
                path = tuple(Chem.rdmolops.GetShortestPath(mol, left, right))
                if start_idx in path:
                    paths.extend((path, path[::-1]))
    if not paths:
        return None
    longest = max(map(len, paths))
    candidates = []
    for path in paths:
        if len(path) != longest or not set(path) <= atoms:
            continue
        prefixes = []
        for locant, atom_idx in enumerate(path, start=1):
            for neighbor_idx in adjacency[atom_idx] - set(path):
                branch = _acyclic_organyl_name(mol, neighbor_idx, atom_idx)
                if branch is None:
                    return None  # Do not compare an incomplete set of chains.
                prefixes.append((str(locant), branch[0]))
        attachment = path.index(start_idx) + 1
        by_name = sorted(prefixes, key=lambda row: (
            prefix_alphabetization_key(row[1]), int(row[0])))
        key = (attachment, -len(prefixes),
               tuple(sorted(int(loc) for loc, _ in prefixes)),
               tuple((prefix_alphabetization_key(name), int(loc))
                     for loc, name in by_name))
        root = ROOTS.get(len(path))
        if not root:
            return None
        parent = root + "yl" if root_degree == 1 else f"{root}an-{attachment}-yl"
        name = format_prefixes(prefixes) + parent
        candidates.append((key, name))
    if not candidates:
        return None
    best = min(key for key, _ in candidates)
    names = {name for key, name in candidates if key == best}
    if len(names) != 1:
        return None
    return names.pop(), root_degree != 1, longest


def _component_has_unsupported_annotation(mol, start_idx: int, blocked_idx: int) -> bool:
    """Keep annotation transfer out of this bounded organyloxy writer."""
    component = set()
    pending = [start_idx]
    while pending:
        current = pending.pop()
        if current in component:
            continue
        component.add(current)
        atom = mol.GetAtomWithIdx(current)
        if atom.GetIsotope() or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return True
        if any(
            neighbor.GetSymbol() == "H" and neighbor.GetIsotope()
            for neighbor in atom.GetNeighbors()
        ):
            return True
        pending.extend(
            neighbor.GetIdx()
            for neighbor in atom.GetNeighbors()
            if neighbor.GetIdx() != blocked_idx and neighbor.GetAtomicNum() > 1
        )
    component_bonds = (
        mol.GetBondBetweenAtoms(left, right)
        for left in component
        for right in component
        if left < right
    )
    return any(
        bond is not None and bond.GetStereo() != Chem.BondStereo.STEREONONE
        for bond in component_bonds
    )


def _phenoxy_substituent_prefix(ring: str) -> str:
    if ring == "phenyl":
        return "phenoxy"
    if ring.endswith("phenyl") and not ring.endswith("diphenyl"):
        return ring[:-2] + "oxy"
    return ""


def _protected_alkoxy_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    ethoxy = _silylethoxy_prefix(mol, carbon_idx, oxygen_idx)
    if ethoxy:
        return ethoxy
    tbdmsoxyethoxy = _tbdmsoxyethoxy_prefix(mol, carbon_idx, oxygen_idx)
    if tbdmsoxyethoxy:
        return tbdmsoxyethoxy
    return ""


def _phosphorus_oxy_prefix(mol, phosphorus_idx: int, oxygen_idx: int) -> str:
    from .functional_groups import _mapped_phosphorus_prefix
    mapped = _mapped_phosphorus_prefix(mol, phosphorus_idx, oxygen_idx)
    if mapped:
        return mapped + 'oxy'
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)
    if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(phosphorus_idx, oxygen_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    chain_length = _neutral_phosphate_chain_length(mol, phosphorus_idx, oxygen_idx)
    if chain_length in {2, 3}:
        return {2: "diphosphooxy", 3: "triphosphooxy"}[chain_length]
    double_oxygen_count = 0
    hydroxy_count = 0
    oxido_count = 0
    ligand_records = []
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == oxygen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(phosphorus_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
            continue
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and len(nbr.GetNeighbors()) == 1:
            if nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                hydroxy_count += 1
                continue
            if nbr.GetFormalCharge() < 0:
                oxido_count += 1
                continue
            return ""
        if bond.GetBondType() != Chem.BondType.SINGLE or nbr.GetAtomicNum() <= 1:
            return ""
        if nbr.GetSymbol() not in {"C", "O"}:
            return ""
        ligand = branch_prefix(mol, nidx, phosphorus_idx)
        if not ligand:
            return ""
        ligand_records.append((_phosphorus_oxy_ligand_class(nbr), ligand))
    if double_oxygen_count != 1:
        return ""
    if not ligand_records:
        if hydroxy_count == 2 and oxido_count == 0:
            return "phosphonooxy"
        if hydroxy_count == 0 and oxido_count == 2:
            return "phosphonatooxy"
        return ""
    if hydroxy_count == 1 and oxido_count == 0 and len(ligand_records) == 1:
        ligand_class, ligand = ligand_records[0]
        if ligand_class == 1:
            # This ligand already owns the original P--O--X oxygen.
            # Substitution on a phosphono OH would introduce a second O
            # between the centers.  Assemble the two real phosphoryl
            # ligands, OH and O--X, before adding the attachment oxygen.
            return "(hydroxy(" + ligand + ")phosphoryl)oxy"
        if _phosphorus_oxy_ligand_needs_parentheses(ligand):
            return ""
        return _phosphorus_oxy_ligand_unit(ligand) + "phosphonooxy"
    if hydroxy_count == 0 and oxido_count == 0 and len(ligand_records) == 2:
        if any(_phosphorus_oxy_ligand_needs_parentheses(ligand) for _ligand_class, ligand in ligand_records):
            return ""
        if {ligand_class for ligand_class, _ligand in ligand_records} == {0, 1}:
            return _phosphorus_oxy_distinct_ligand_text(ligand_records) + "phosphoryloxy"
        counts = Counter(ligand_records)
        parts = []
        for (_ligand_class, ligand) in sorted(counts):
            count = counts[(_ligand_class, ligand)]
            unit = _phosphorus_oxy_ligand_unit(ligand)
            parts.append(unit if count == 1 else MULT.get(count, "") + unit)
        return "".join(parts) + "phosphoryloxy"
    return ""


def _neutral_phosphate_chain_length(mol, phosphorus_idx: int, blocked_oxygen_idx: int) -> int:
    length = 0
    current_p = phosphorus_idx
    blocked_oxygen = blocked_oxygen_idx
    seen_phosphorus: set[int] = set()
    while current_p not in seen_phosphorus:
        seen_phosphorus.add(current_p)
        atom = mol.GetAtomWithIdx(current_p)
        if atom.GetSymbol() != "P" or atom.GetFormalCharge() != 0:
            return 0
        length += 1
        double_oxygen_count = 0
        hydroxy_count = 0
        bridge: tuple[int, int] | None = None
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == blocked_oxygen:
                continue
            bond = mol.GetBondBetweenAtoms(current_p, nidx)
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygen_count += 1
                continue
            if nbr.GetSymbol() != "O" or bond.GetBondType() != Chem.BondType.SINGLE:
                return 0
            if len(nbr.GetNeighbors()) == 1:
                if nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                    hydroxy_count += 1
                    continue
                return 0
            next_phosphorus = [
                other.GetIdx()
                for other in nbr.GetNeighbors()
                if other.GetIdx() != current_p and other.GetSymbol() == "P"
            ]
            if len(next_phosphorus) != 1 or bridge is not None:
                return 0
            bridge = (nidx, next_phosphorus[0])
        if double_oxygen_count != 1:
            return 0
        if bridge is None:
            return length if hydroxy_count == 2 else 0
        if hydroxy_count != 1:
            return 0
        blocked_oxygen, current_p = bridge
    return 0


def _phosphorus_oxy_ligand_unit(ligand: str) -> str:
    if _phosphorus_oxy_ligand_needs_parentheses(ligand):
        return f"({ligand})"
    return ligand


def _phosphorus_oxy_distinct_ligand_text(ligand_records: Sequence[tuple[int, str]]) -> str:
    first, second = sorted(ligand_records)
    return _phosphorus_oxy_ligand_unit(first[1]) + f"({_phosphorus_oxy_ligand_unit(second[1])})"


def _phosphorus_oxy_ligand_class(atom) -> int:
    if atom.GetSymbol() == "C" and not atom.GetIsAromatic():
        return 0
    if atom.GetSymbol() == "O":
        return 1
    return 2


def _phosphorus_oxy_ligand_needs_parentheses(ligand: str) -> bool:
    return "-" in ligand or ligand[:1].isdigit() or ligand.startswith("(")


def _silylethoxy_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    first = mol.GetAtomWithIdx(carbon_idx)
    if first.GetSymbol() != "C" or first.GetFormalCharge() != 0 or first.IsInRing():
        return ""
    second = _single_heavy_neighbor(mol, carbon_idx, oxygen_idx, "C")
    if second is None:
        return ""
    silicon = _single_heavy_neighbor(mol, second, carbon_idx, "Si")
    if silicon is None:
        return ""
    return "2-tms-ethoxy" if _is_trimethylsilyl_group(mol, silicon, second) else ""


def _tbdmsoxyethoxy_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    first = mol.GetAtomWithIdx(carbon_idx)
    if first.GetSymbol() != "C" or first.GetFormalCharge() != 0 or first.IsInRing():
        return ""
    second = _single_heavy_neighbor(mol, carbon_idx, oxygen_idx, "C")
    if second is None:
        return ""
    distal_oxygen = _single_heavy_neighbor(mol, second, carbon_idx, "O")
    if distal_oxygen is None:
        return ""
    silicon = _single_heavy_neighbor(mol, distal_oxygen, second, "Si")
    if silicon is None:
        return ""
    return "2-tbdmsoxyethoxy" if _is_tbdms_group(mol, silicon, distal_oxygen) else ""


def _single_heavy_neighbor(mol, atom_idx: int, blocked_idx: int, symbol: str) -> int | None:
    atom = mol.GetAtomWithIdx(atom_idx)
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != blocked_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    neighbor_idx = heavy[0]
    neighbor = mol.GetAtomWithIdx(neighbor_idx)
    bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
    if neighbor.GetSymbol() != symbol or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    return neighbor_idx


def _sulfur_oxy_alias(mol, sulfur_idx: int, oxygen_idx: int) -> str:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0:
        return ""
    double_oxygen_count = 0
    single_ligands = []
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == oxygen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            single_ligands.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return ""
    if double_oxygen_count != 2 or len(single_ligands) != 1:
        return ""
    ligand_idx = single_ligands[0]
    ligand = mol.GetAtomWithIdx(ligand_idx)
    if ligand.GetSymbol() == "F":
        return "fluorosulfonyloxy"
    if ligand.GetSymbol() == "O" and _is_terminal_hydroxy_oxygen(mol, ligand_idx, sulfur_idx):
        return "sulfooxy"
    if ligand.GetSymbol() == "C" and _is_methyl_sulfonyl_ligand(mol, ligand_idx, sulfur_idx):
        return "(methanesulfonyl)oxy"
    if ligand.GetSymbol() == "C" and _is_trifluoromethyl(mol, ligand_idx, sulfur_idx):
        return "(trifluoromethanesulfonyl)oxy"
    if ligand.IsInRing():
        ring = _ring_substituent_prefix(mol, ligand_idx, sulfur_idx)
        if ring in {"p-tolyl", "4-methylphenyl"}:
            return "(4-methylbenzenesulfonyl)oxy"
    return ""


def _is_terminal_hydroxy_oxygen(mol, oxygen_idx: int, sulfur_idx: int) -> bool:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0:
        return False
    return not [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != sulfur_idx and nbr.GetAtomicNum() > 1
    ]


def _is_methyl_sulfonyl_ligand(mol, carbon_idx: int, sulfur_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing() or carbon.GetIsotope():
        return False
    bond = mol.GetBondBetweenAtoms(carbon_idx, sulfur_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    return not [
        nbr.GetIdx()
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != sulfur_idx and nbr.GetAtomicNum() > 1
    ]


def _is_trifluoromethyl(mol, carbon_idx: int, sulfur_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return False
    fluorines = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == sulfur_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if nbr.GetSymbol() == "F" and bond.GetBondType() == Chem.BondType.SINGLE:
            fluorines.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return False
    return len(fluorines) == 3


def _retained_oxygen_resource_prefix(mol, oxygen_idx: int, parent_idx: int) -> str:
    template = _resource_substituent_match(mol, oxygen_idx, parent_idx, Chem)
    if not template:
        return ""
    name = template_record_name(template)
    if "siloxy" in name or "silanoxy" in name:
        return name
    return ""


def _glycosyloxy_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    if mol.GetAtomWithIdx(carbon_idx).GetSymbol() != "C":
        return ""
    template = _resource_substituent_match(mol, carbon_idx, oxygen_idx, Chem)
    if not template or template.get("source_file") != "carbohydrates.xml":
        return ""
    name = template_record_name(template)
    if not name.endswith(("osyl", "aminyl", "muramyl", "neuraminyl")):
        return ""
    return name + "oxy"


def _sulfur_has_ring_carbon_ligand(mol, sulfur_idx: int, oxygen_idx: int) -> bool:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    return any(
        nbr.GetIdx() != oxygen_idx and nbr.GetSymbol() == "C" and nbr.IsInRing()
        for nbr in sulfur.GetNeighbors()
    )


def _sulfur_has_nitroaryl_ligand(mol, sulfur_idx: int, oxygen_idx: int) -> bool:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    ring_starts = [
        nbr.GetIdx()
        for nbr in sulfur.GetNeighbors()
        if nbr.GetIdx() != oxygen_idx and nbr.GetSymbol() == "C" and nbr.IsInRing()
    ]
    seen: set[int] = set()
    stack = list(ring_starts)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        atom = mol.GetAtomWithIdx(current)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx == sulfur_idx:
                continue
            if _is_nitro_nitrogen(mol, nbr_idx):
                return True
            if nbr.IsInRing() and nbr_idx not in seen:
                stack.append(nbr_idx)
    return False


def _nonplain_hydrogen_neighbor_indices(atom, parent_idx: int) -> list[int]:
    return [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and not (nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0)
    ]


def _isotopic_hydrogen_prefix(atom) -> str:
    if atom.GetSymbol() != "H":
        return ""
    if atom.GetIsotope() == 2:
        return "deuterio"
    if atom.GetIsotope() == 3:
        return "tritio"
    return ""


def _isotopic_hydrogen_oxy_prefix(atom) -> str:
    isotope = _isotopic_hydrogen_prefix(atom)
    return isotope + "oxy" if isotope else ""


def _ylidene_oxidaniumyl_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(carbon_idx, oxygen_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in carbon.GetNeighbors() if nbr.GetIdx() != oxygen_idx]
    if not heavy:
        return "methylideneoxidaniumyl"
    if len(heavy) != 1:
        return ""
    chain = _linear_carbon_chain(mol, heavy[0], carbon_idx, max_len=9)
    if not chain:
        return ""
    root = ROOTS.get(len(chain) + 1)
    return root + "ylideneoxidaniumyl" if root else ""


def _nitrogen_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if _is_diazo_attachment(mol, idx, parent_idx):
        return "diazo"
    if parent_bond is not None and parent_bond.GetBondType() == Chem.BondType.DOUBLE:
        return _imino_prefix(mol, idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        # A terminal nitrile N is not an amino group. Triple-bonded units
        # need their own complete C#N construction, including direct callers.
        return ""
    from ..hydrazonamide_prefixes import hydrazonamido_prefix
    from ..branch_rules import collect_subtree
    hydrazonamide = hydrazonamido_prefix(mol, Chem, idx, parent_idx,
                                        collect_subtree(mol, idx, parent_idx))
    if hydrazonamide is not None:
        return hydrazonamide[0]
    cumulene = _isocyanato_prefix(mol, idx, parent_idx)
    if cumulene:
        return cumulene
    if _is_isocyano_nitrogen(mol, idx):
        return "isocyano"
    if _is_diazonio_attachment(mol, idx, parent_idx):
        return "diazonio"
    azoxy = _azoxy_prefix(mol, idx, parent_idx)
    if azoxy:
        return azoxy
    diazenyl = _diazenyl_prefix(mol, idx, parent_idx)
    if diazenyl:
        return diazenyl
    hydrazone = _hydrazone_prefix(mol, idx, parent_idx)
    if hydrazone:
        return hydrazone
    imino_amino = _imino_amino_prefix(mol, idx, parent_idx)
    if imino_amino:
        return imino_amino
    amidino_amino = _amidino_amino_prefix(mol, idx, parent_idx)
    if amidino_amino:
        return amidino_amino
    sulfur_ylidene_amino = _sulfur_ylidene_amino_prefix(mol, idx, parent_idx)
    if sulfur_ylidene_amino:
        return sulfur_ylidene_amino
    if _is_azido_attachment(mol, idx, parent_idx):
        return "azido"
    if _nitro_branch_predicate(mol, idx, parent_idx):
        return "nitro"
    if _is_nitroso_attachment(mol, idx, parent_idx):
        return "nitroso"
    # P-68.3.1.2.1: the complete terminal –NH–NH2 substituent has the
    # preselected hydrazinyl name, independent of the atom it is attached to.
    if (atom.GetFormalCharge() == 0 and not atom.GetIsotope() and not atom.GetNumRadicalElectrons()
            and not atom.IsInRing()
            and not atom.GetIsAromatic() and atom.GetDegree() == 2
            and atom.GetTotalNumHs(includeNeighbors=True) == 1
            and parent_bond is not None and parent_bond.GetBondTypeAsDouble() == 1):
        terminal = [a for a in atom.GetNeighbors() if a.GetIdx() != parent_idx]
        if (len(terminal) == 1 and terminal[0].GetAtomicNum() == 7
                and not terminal[0].IsInRing() and not terminal[0].GetIsAromatic()
                and not terminal[0].GetFormalCharge() and not terminal[0].GetIsotope()
                and not terminal[0].GetNumRadicalElectrons() and terminal[0].GetDegree() == 1
                and terminal[0].GetTotalNumHs(includeNeighbors=True) == 2
                and mol.GetBondBetweenAtoms(idx, terminal[0].GetIdx()).GetBondTypeAsDouble() == 1):
            return "hydrazinyl"
    hydrazinyl = _terminal_chain_hydrazinyl(mol, idx, parent_idx)
    if hydrazinyl:
        return hydrazinyl
    if atom.GetFormalCharge() > 0:
        heavy = _nonplain_hydrogen_neighbor_indices(atom, parent_idx)
        if not heavy:
            return suffix_supported_term("ium", "azaniumyl")
        names = []
        for nidx in heavy:
            if mol.GetAtomWithIdx(nidx).IsInRing():
                name = _ring_substituent_prefix(mol, nidx, idx)
            else:
                name = branch_prefix(mol, nidx, idx) or substituent_from_root(mol, nidx, idx)
            if not name:
                return ""
            names.append(name)
        from ..formatting import format_prefixes
        ligands = format_prefixes([('', name) for name in names],
                                 omit_locants=True, hierarchical_enclosures=True,
                                 separate_unlocanted=True)
        # All N ligands belong to this one charged substituent. Without the
        # complete boundary a preceding alkyl can attach to the outer parent.
        return '(' + ligands + "azaniumyl)"
    if atom.GetFormalCharge() < 0:
        return _azanidyl_prefix(mol, idx, parent_idx)
    if atom.GetFormalCharge() != 0:
        return ""
    heavy = _nonplain_hydrogen_neighbor_indices(atom, parent_idx)
    if heavy and all(mol.GetAtomWithIdx(nidx).GetSymbol() in HALO for nidx in heavy):
        names = [HALO[mol.GetAtomWithIdx(nidx).GetSymbol()] for nidx in heavy]
        counts = Counter(names)
        return "".join((MULT.get(count, "") if count > 1 else "") + name for name, count in sorted(counts.items())) + "amino"
    if not heavy:
        return "amino"
    if len(heavy) == 1:
        child = mol.GetAtomWithIdx(heavy[0])
        if (child.GetAtomicNum() == 6 and not child.IsInRing()
                and any(n.GetAtomicNum() == 6 and n.IsInRing() for n in child.GetNeighbors())
                and any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(child).GetAtomicNum() == 8
                        for b in child.GetBonds())):
            from .amido_components import mapped_amido_prefix

            amido = mapped_amido_prefix(
                mol, Chem, idx, parent_idx, collect_subtree(mol, idx, parent_idx), ring_only=True)
            if amido is not None:
                return amido[0]
    names = []
    for nidx in heavy:
        name = _nitrogen_ligand_prefix(mol, nidx, idx)
        if not name:
            return ""
        names.append(name)
    return _amino_prefix_from_ligands(mol, idx, parent_idx, names)


def _terminal_chain_hydrazinyl(mol, idx: int, parent_idx: int) -> str:
    """P-66.3.2.2 hydrazinyl on an ordinary acyclic carbonyl chain end.

    Prove the original C(=O)-N-N topology and every methyl branch before
    replacing recursive amino-amino wording with a hydrazine-derived prefix.
    This local branch construction does not choose the whole parent chain.
    """
    parent = mol.GetAtomWithIdx(parent_idx)
    n1 = mol.GetAtomWithIdx(idx)
    if (parent.GetAtomicNum() != 6 or parent.IsInRing() or parent.GetIsAromatic()
            or parent.GetFormalCharge() or parent.GetIsotope()
            or parent.GetNumRadicalElectrons() or parent.GetDegree() != 3
            or n1.GetAtomicNum() != 7 or n1.IsInRing() or n1.GetIsAromatic()
            or n1.GetFormalCharge() or n1.GetIsotope()
            or n1.GetNumRadicalElectrons() or int(n1.GetChiralTag())):
        return ""
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1:
        return ""
    other = [a for a in parent.GetNeighbors() if a.GetIdx() != idx]
    oxygens = [a for a in other if a.GetAtomicNum() == 8 and a.GetDegree() == 1
               and not a.GetIsotope() and not a.GetFormalCharge()
               and mol.GetBondBetweenAtoms(parent_idx, a.GetIdx()).GetBondTypeAsDouble() == 2]
    chains = [a for a in other if a.GetAtomicNum() == 6 and not a.IsInRing()
              and not a.GetIsAromatic()
              and mol.GetBondBetweenAtoms(parent_idx, a.GetIdx()).GetBondTypeAsDouble() == 1]
    if len(oxygens) != 1 or len(chains) != 1:
        return ""
    partners = [a for a in n1.GetNeighbors() if a.GetIdx() != parent_idx
                and a.GetAtomicNum() == 7]
    if len(partners) != 1:
        return ""
    n2 = partners[0]
    if (n2.IsInRing() or n2.GetIsAromatic() or n2.GetFormalCharge()
            or n2.GetIsotope() or n2.GetNumRadicalElectrons()
            or int(n2.GetChiralTag())
            or mol.GetBondBetweenAtoms(idx, n2.GetIdx()).GetBondTypeAsDouble() != 1):
        return ""
    locants = []
    for position, atom, blocked, maximum in ((1, n1, {parent_idx, n2.GetIdx()}, 1),
                                               (2, n2, {idx}, 2)):
        if any(neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope()
               for neighbor in atom.GetNeighbors()):
            return ""
        ligands = [neighbor for neighbor in atom.GetNeighbors()
                   if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in blocked]
        if len(ligands) > maximum or atom.GetTotalValence() != 3:
            return ""
        for ligand in ligands:
            if (ligand.GetAtomicNum() != 6 or ligand.GetDegree() != 1
                    or ligand.GetIsotope() or ligand.GetFormalCharge()
                    or ligand.GetNumRadicalElectrons() or ligand.GetTotalNumHs() != 3
                    or mol.GetBondBetweenAtoms(atom.GetIdx(), ligand.GetIdx()).GetBondTypeAsDouble() != 1):
                return ""
            locants.append(str(position))
    if not locants:
        return "hydrazinyl"
    return f"{','.join(locants)}-{MULT.get(len(locants), '')}methylhydrazin-1-yl"


def _nitrogen_ligand_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    symbol = atom.GetSymbol()
    if symbol == "H":
        return _isotopic_hydrogen_prefix(atom)
    if symbol in HALO:
        isotope = atom.GetIsotope()
        return f"[{isotope}{symbol}]{HALO[symbol]}" if isotope else HALO[symbol]
    if atom.IsInRing():
        return _ring_substituent_prefix(mol, idx, parent_idx)
    if symbol == "N":
        return _nitrogen_prefix(mol, idx, parent_idx)
    if symbol == "S":
        return _sulfur_prefix(mol, idx, parent_idx)
    if symbol == "P":
        # Keep the complete original P acyl unit intact before legacy
        # functional-carbon/resource writers can concatenate its ligands.
        from ..phosphorus_acyl_prefixes import phosphorus_acyl_prefix
        acyl = phosphorus_acyl_prefix(mol, Chem, idx, parent_idx)
        if acyl is not None:
            return acyl.name
    from .preferred_acyl import alkanoate_amino_ligand
    preferred_ligand = alkanoate_amino_ligand(mol, idx, parent_idx)
    if preferred_ligand:
        return preferred_ligand
    name = (
        suffix_prefix_term("nitrile", "cyano")
        if _is_nitrile_carbon(mol, idx)
        else _functional_carbon_substituent_name(mol, idx, parent_idx)
        or _carbonyl_substituent_prefix(mol, idx, parent_idx)
        or branch_prefix(mol, idx, parent_idx)
        or substituent_from_root(mol, idx, parent_idx)
    )
    # A descriptor inside a child branch does not describe this ligand's root.
    # Explicitly numbered alkyl ligands start at the nitrogen attachment atom.
    if symbol == "C" and name.startswith("1-") and atom.HasProp("_CIPCode"):
        descriptor = atom.GetProp("_CIPCode")
        if descriptor in {"R", "S"}:
            name = f"(1{descriptor})-{name}"
    return name


def _sulfur_ylidene_amino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if (
        atom.GetSymbol() != "N" or atom.GetFormalCharge() or atom.IsInRing()
        or atom.GetIsotope() or bond is None or bond.GetBondType() != Chem.BondType.SINGLE
    ):
        return ""
    neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]
    if len(neighbors) != 1:
        return ""
    from .functional_groups import _hydrogen_rich_sulfur_prefix, _lambda6_sulfanylidene_prefix

    sulfur = mol.GetAtomWithIdx(neighbors[0])
    if (sulfur.GetAtomicNum() == 16 and sulfur.GetTotalNumHs(includeNeighbors=True)
            and mol.GetBondBetweenAtoms(idx, sulfur.GetIdx()).GetBondTypeAsDouble() == 2):
        from ..formatting import enclose_name_unit

        prefix = _hydrogen_rich_sulfur_prefix(mol, sulfur.GetIdx(), idx)
        return enclose_name_unit(prefix) + "amino" if prefix else ""

    prefix = _lambda6_sulfanylidene_prefix(mol, neighbors[0], idx)
    return prefix + "amino" if prefix else ""


def _amino_prefix_from_ligands(mol, amino_idx: int, parent_idx: int, names: Sequence[str]) -> str:
    from ..carboxamide_prefixes import mapped_carbamoyl_linker
    from ..formatting import prefix_alphabetization_key
    from ..prefix_presentation import enclose_complete_name_unit
    def has_ligand_marks(name: str) -> bool:
        return any(mark in name for mark in "-([{")

    carbamoylamino = mapped_carbamoyl_linker(mol, Chem, amino_idx, parent_idx, names)
    if carbamoylamino is not None:
        return carbamoylamino[0]
    if len(names) == 1:
        # P-67.1.4.1.1.5: the complete acid-derived P acyl component is
        # one ligand of amino. Its internal parentheses do not introduce
        # an N locant or change the outer attachment operation.
        ligands = [a for a in mol.GetAtomWithIdx(amino_idx).GetNeighbors()
                   if a.GetAtomicNum() > 1 and a.GetIdx() != parent_idx]
        if len(ligands) == 1 and ligands[0].GetAtomicNum() == 15:
            from ..phosphorus_acyl_prefixes import phosphorus_acyl_prefix
            acyl = phosphorus_acyl_prefix(mol, Chem, ligands[0].GetIdx(), amino_idx)
            if acyl is not None and acyl.name == names[0]:
                return enclose_complete_name_unit(acyl.name) + 'amino'
        retained = _retained_single_ligand_amino_prefix(mol, amino_idx, parent_idx, names[0])
        if retained:
            return retained
        if names[0].endswith("phosphorothioyl"):
            return enclose_complete_name_unit(names[0]) + "amino"
        if mol.GetAtomWithIdx(parent_idx).GetSymbol() != "S" and _single_ligand_amino_needs_explicit_n(names[0]):
            return f"N-{enclose_complete_name_unit(names[0])}amino"
        if has_ligand_marks(names[0]):
            return enclose_complete_name_unit(names[0]) + "amino"
        return names[0] + "amino"
    names = tuple(_multiligand_amino_safe_name(name) for name in names)
    counts = Counter(names)
    if len(counts) > 1 and all(count == 1 for count in counts.values()) and any(
        _needs_explicit_n_ligand_name(name) for name in counts
    ):
        parts = []
        for name in sorted(counts, key=prefix_alphabetization_key):
            parts.append(f"N-{enclose_complete_name_unit(name)}" if has_ligand_marks(name) else f"N-{name}")
        return "-".join(parts) + "amino"
    if all(count == 1 for count in counts.values()) and all(
        not has_ligand_marks(name) for name in counts
    ):
        ordered = sorted(counts, key=prefix_alphabetization_key)
        return ordered[0] + "".join(enclose_complete_name_unit(name) for name in ordered[1:]) + "amino"
    if all(count == 1 for count in counts.values()):
        parts = []
        for name in sorted(counts, key=prefix_alphabetization_key):
            if has_ligand_marks(name):
                parts.append(f"N-{enclose_complete_name_unit(name)}")
            else:
                parts.append(f"N-{name}")
        return "-".join(parts) + "amino"
    parts = []
    # P-16.3.2(c)/P-16.5: substitution on an amino ligand makes the
    # complete ligand compound. Multiplying methylamino as dimethylamino
    # would put two methyls on one N and discard the second ligand's N.
    amino_ligands = [mol.GetAtomWithIdx(i) for i in
                     _nonplain_hydrogen_neighbor_indices(mol.GetAtomWithIdx(amino_idx), parent_idx)]
    compound_amino_names = {
        name for ligand, name in zip(amino_ligands, names)
        if ligand.GetAtomicNum() == 7
        and any(a.GetAtomicNum() > 1 and a.GetIdx() != amino_idx
                for a in ligand.GetNeighbors())
    }
    for name in sorted(counts, key=prefix_alphabetization_key):
        count = counts[name]
        # A simple multiplier on a chalcogenyl ligand can denote a chain
        # (disulfanyl is S-S), not separate ligands attached to nitrogen.
        if count > 1 and (name in compound_amino_names or name.endswith(("sulfonyl", "sulfinyl", "sulfanyl", "selanyl", "tellanyl"))):
            multi = {2: "bis", 3: "tris", 4: "tetrakis"}.get(count, MULT.get(count, "") + "kis")
            parts.append(multi + enclose_complete_name_unit(name))
        elif count > 1 and (
            has_ligand_marks(name) or _repeated_amino_ligand_needs_complex_multiplier(name, count)
        ):
            multi = {2: "bis", 3: "tris", 4: "tetrakis"}.get(count, MULT.get(count, "") + "kis")
            parts.append(multi + enclose_complete_name_unit(name))
        elif count == 1 and (name.endswith("carbonyl") or name.endswith("sulfonyl")) and len(names) > 1:
            parts.append(enclose_complete_name_unit(name))
        elif has_ligand_marks(name):
            parts.append(enclose_complete_name_unit(name))
        else:
            parts.append((MULT.get(count, "") if count > 1 else "") + name)
    return "".join(parts) + "amino"


def _repeated_amino_ligand_needs_complex_multiplier(name: str, count: int) -> bool:
    if name in {"methyl", "ethyl", "propyl", "butyl", "phenyl", "benzyl"}:
        return False
    from ..formatting import _is_composite_substituent_name
    return ((count == 3 and name == "decyl")
            or _is_composite_substituent_name(name)
            or name.endswith(("methyl", "ethyl", "propyl", "butyl")))


def _single_ligand_amino_needs_explicit_n(name: str) -> bool:
    if "(" not in name or ")" not in name:
        return False
    if name.startswith("("):
        return False
    if name.endswith(
        ("carbonyl", "sulfonyl", "sulfinyl", "sulfonimidoyl", "sulfinimidoyl", "sulfimidoyl")
    ):
        return False
    return True


def _retained_single_ligand_amino_prefix(mol, amino_idx: int, parent_idx: int, name: str) -> str:
    # P-62.2.1.1.1 allows full substitution on retained anilino.
    if name.endswith("phenyl"):
        nitrogen = mol.GetAtomWithIdx(amino_idx)
        if (not nitrogen.IsInRing() and not nitrogen.GetIsotope()
                and not nitrogen.GetFormalCharge() and not nitrogen.GetNumRadicalElectrons()
                and nitrogen.GetDegree() == 2 and nitrogen.GetTotalNumHs() == 1
                and nitrogen.GetTotalValence() == 3
                and mol.GetAtomWithIdx(parent_idx).GetAtomicNum() == 6):
            aryl = next(a for a in nitrogen.GetNeighbors() if a.GetIdx() != parent_idx)
            if aryl.GetAtomicNum() == 6 and aryl.GetIsAromatic():
                rings = mol.GetRingInfo()
                if any(len(ring) == 6 and aryl.GetIdx() in ring
                       and all(rings.NumAtomRings(i) == 1
                               and mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                               and mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                       for ring in rings.AtomRings()):
                    return name[:-len("phenyl")] + "anilino"
    if name.endswith("sulfonyl") and len(name) > len("sulfonyl"):
        return name[: -len("sulfonyl")] + "sulfonamido"
    acyl_amides = {
        "acroyl": "acrylamido",
        "acryloyl": "acrylamido",
        "acetyl": "acetamido",
        "benzoyl": "benzamido",
        "butyryl": "butyramido",
        "cyanoacetyl": "cyanoacetamido",
        "cyanomethylcarbonyl": "cyanoacetamido",
        "cyclopropylcarbonyl": "cyclopropanecarboxamido",
        "cyclobutylcarbonyl": "cyclobutanecarboxamido",
        "cyclopentylcarbonyl": "cyclopentanecarboxamido",
        "cyclohexylcarbonyl": "cyclohexanecarboxamido",
        "difluoroacetyl": "difluoroacetamido",
        "formyl": "formamido",
        "isobutyryl": "isobutyramido",
        "methacryloyl": "methacrylamido",
        "phenylacetyl": "(phenylacetyl)amino",
        "pivaloyl": "pivalamido",
        "propionyl": "propionamido",
        "trifluoroacetyl": "trifluoroacetamido",
        "valeryl": "valeramido",
    }
    if name in acyl_amides:
        return acyl_amides[name]
    if name.endswith("acetyl"):
        return name[:-len("acetyl")] + "acetamido"
    if name in {"boc", "tert-butoxycarbonyl"} and _bocamino_prefix_allowed(mol, amino_idx, parent_idx):
        return "(tert-butoxycarbonyl)amino"
    if name in {"cbz", "benzyloxycarbonyl"}:
        return "(benzyloxycarbonyl)amino"
    if name == "fmoc":
        return "fmocamino"
    if name in {"carbamyl", "carbamoyl"}:
        return "carbamoylamino"
    return ""


def _multiligand_amino_safe_name(name: str) -> str:
    return {
        "carbamyl": "carbamoyl",
        "boc": "tert-butoxycarbonyl",
        "cbz": "carbbenzoxy",
        "fmoc": "((fluorene-9-yl)methyl)oxycarbonyl",
    }.get(name, name)


def _bocamino_prefix_allowed(mol, amino_idx: int, parent_idx: int) -> bool:
    parent = mol.GetAtomWithIdx(parent_idx)
    if not parent.IsInRing() or mol.GetRingInfo().NumRings() < 2:
        return True
    for atom in iter_atoms(mol):
        if atom.GetSymbol() != "C" or not atom.IsInRing():
            continue
        cidx = atom.GetIdx()
        has_exocyclic_oxo = False
        has_ring_nitrogen = False
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if (
                nbr.GetSymbol() == "O"
                and not nbr.IsInRing()
                and bond is not None
                and bond.GetBondType() == Chem.BondType.DOUBLE
            ):
                has_exocyclic_oxo = True
            elif nbr.GetSymbol() == "N" and nbr.IsInRing() and nidx != amino_idx:
                has_ring_nitrogen = True
        if has_exocyclic_oxo and has_ring_nitrogen:
            return False
    return True


def _needs_explicit_n_ligand_name(name: str) -> bool:
    return name.endswith(("carbonyl", "formyl")) or name in {"chloroformyl", "fluoroformyl", "bromoformyl", "iodoformyl"}


def _azanidyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    # Azanidyl is a monoanion derived from NH2(-), not a generic negative
    # nitrogen token. Its attachment consumes one of that parent's H atoms.
    # Check the H budget as well as charge, including any other N ligands.
    if (atom.GetFormalCharge() != -1 or atom.GetNumRadicalElectrons()
            or atom.GetTotalValence() != 2
            or any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds())):
        return ""
    parent = mol.GetAtomWithIdx(parent_idx)
    if parent.GetFormalCharge() < 0:
        return ""
    heavy = _nonplain_hydrogen_neighbor_indices(mol.GetAtomWithIdx(idx), parent_idx)
    if not heavy:
        return "azanidyl"
    if len(heavy) != 1:
        return ""
    substituent_idx = heavy[0]
    substituent = mol.GetAtomWithIdx(substituent_idx)
    if substituent.IsInRing():
        name = _ring_substituent_prefix(mol, substituent_idx, idx)
    elif substituent.GetSymbol() in HALO:
        isotope = substituent.GetIsotope()
        symbol = substituent.GetSymbol()
        name = f"[{isotope}{symbol}]{HALO[symbol]}" if isotope else HALO[symbol]
    elif substituent.GetSymbol() == "S":
        name = _sulfur_prefix(mol, substituent_idx, idx)
    elif substituent.GetSymbol() == "C":
        name = (
            suffix_prefix_term("nitrile", "cyano")
            if _is_nitrile_carbon(mol, substituent_idx)
            else _functional_carbon_substituent_name(mol, substituent_idx, idx)
            or _carbonyl_substituent_prefix(mol, substituent_idx, idx)
            or substituent_from_root(mol, substituent_idx, idx)
        )
    elif substituent.GetSymbol() == "N":
        name = _nitrogen_prefix(mol, substituent_idx, idx)
    else:
        return ""
    if not name:
        return ""
    if "-" in name or name.startswith("("):
        return f"({name})azanidyl"
    return name + "azanidyl"


def _functional_carbon_substituent_name(mol, idx: int, parent_idx: int) -> str:
    if mol.GetAtomWithIdx(idx).GetSymbol() != "C":
        return ""
    if mol.GetAtomWithIdx(parent_idx).GetAtomicNum() == 7:
        from .preferred_acyl import n_attached_chain_acyl_prefix

        acyl = n_attached_chain_acyl_prefix(mol, Chem, idx, parent_idx)
        if acyl:
            return acyl
    return (
        _ketenyl_prefix(mol, idx, parent_idx)
        or _enone_substituent_prefix(mol, idx, parent_idx)
        or _enal_substituent_prefix(mol, idx, parent_idx)
        or _haloethenyl_prefix(mol, idx, parent_idx)
        or _alkynyl_substituent_prefix(mol, idx, parent_idx)
    )


def _imino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N":
        return ""
    if atom.GetFormalCharge() == 1 and atom.GetTotalNumHs() == 2:
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx and n.GetAtomicNum() > 1]
        if not heavy and not atom.IsInRing() and not atom.GetIsotope():
            return "azaniumylidene"
        return ""
    if atom.GetFormalCharge() == 1:
        return _substituted_azaniumylidene_prefix(mol, idx, parent_idx)
    if atom.GetFormalCharge() != 0:
        return ""
    heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
    if not heavy:
        return "imino"
    if len(heavy) != 1:
        return ""
    attached = heavy[0]
    other = mol.GetAtomWithIdx(attached)
    host = mol.GetAtomWithIdx(parent_idx)
    if other.GetAtomicNum() == 8 and host.GetAtomicNum() == 6:
        from ..oxime_components import oxime_imino_prefix

        atoms = frozenset(_mapped_component_atoms(mol, idx, parent_idx))
        text, _ = oxime_imino_prefix(mol, Chem, idx, parent_idx, atoms)
        if text:
            return text
    if (other.GetAtomicNum() == 7 and not other.IsInRing()
            and not (other.GetFormalCharge() or other.GetIsotope())
            and not atom.IsInRing() and not atom.GetIsotope()
            and mol.GetBondBetweenAtoms(idx, attached).GetBondTypeAsDouble() == 1):
        ligands = [a.GetIdx() for a in other.GetNeighbors()
                   if a.GetIdx() != idx and a.GetAtomicNum() > 1]
        if (not ligands and other.GetDegree() == 1 and other.GetTotalNumHs(includeNeighbors=True) == 2
                and not other.GetNumRadicalElectrons() and not atom.GetNumRadicalElectrons()):
            return "hydrazinylidene"
        if len(ligands) == 1 and other.GetTotalNumHs() == 1:
            # P-68.3.1.2.5: when a more senior characteristic group owns
            # the parent, an N-acyl semicarbazone is expressed as a
            # carbamoylhydrazinylidene prefix. Validate the complete acyl
            # branch before dropping the otherwise explicit hydrazine locants.
            if mol.GetAtomWithIdx(ligands[0]).GetAtomicNum() == 6:
                from ..carboxamide_prefixes import preferred_carbamoyl_prefix
                from ..component_mapping import mapped_branch
                acyl, owned = mapped_branch(mol, Chem, ligands[0], attached)
                preferred = preferred_carbamoyl_prefix(mol, ligands[0], attached, acyl, owned)
                if preferred:
                    unit = preferred if preferred == 'carbamoyl' else f'({preferred})'
                    return unit + 'hydrazinylidene'
            ligand = branch_prefix(mol, ligands[0], attached)
            if ligand:
                return format_prefixes([("2", ligand)]) + "hydrazin-1-ylidene"
    if _nitro_branch_predicate(mol, attached, idx):
        return "nitroimino" if _nitroimino_parent_site_supported(mol, parent_idx) else ""
    if mol.GetAtomWithIdx(attached).IsInRing():
        name = _ring_substituent_prefix(mol, attached, idx)
    else:
        name = (
            suffix_prefix_term("nitrile", "cyano")
            if _is_nitrile_carbon(mol, attached)
            else substituent_from_root(mol, attached, idx)
        )
    if not name:
        return ""
    if (host.GetAtomicNum() == 7 and not host.IsInRing() and not atom.IsInRing()
            and atom.GetDegree() == 2 and host.GetDegree() == 2
            and mol.GetBondBetweenAtoms(idx, parent_idx).GetBondType() == Chem.BondType.DOUBLE):
        # In an N-acyl diazene this imino prefix owns the original N=N
        # descriptor, independently of any alkene on the acyl component.
        from ..stereochemistry import assigned_bond_stereochemistry

        bond = mol.GetBondBetweenAtoms(idx, parent_idx)
        if bond.GetStereo():
            label = next((value for a, b, value in assigned_bond_stereochemistry(mol, Chem)
                          if {a, b} == {idx, parent_idx}), '')
            if label and not host.GetFormalCharge() and any(n.GetAtomicNum() == 6 and any(
                    edge.GetBondType() == Chem.BondType.DOUBLE and edge.GetOtherAtom(n).GetAtomicNum() == 8
                    for edge in n.GetBonds()) for n in host.GetNeighbors() if n.GetIdx() != idx):
                text = f'({label})-({name})imino'
                ownership = mol_cache_dict(mol, 'rendered_mapped_branch_bond_stereo')
                ownership[(idx, parent_idx, text, frozenset({idx, parent_idx}))] = label
                return text
    if "-" in name or name.startswith("("):
        return f"({name})imino"
    return name + "imino"


def _substituted_azaniumylidene_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.IsInRing() or atom.GetIsotope():
        return ""
    for ring in mol.GetRingInfo().AtomRings():
        if parent_idx in ring and len(ring) <= 3:
            return ""
    names = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if (
            nbr.GetSymbol() != "C"
            or nbr.GetFormalCharge() != 0
            or bond is None
            or bond.GetBondType() != Chem.BondType.SINGLE
        ):
            return ""
        if nbr.IsInRing():
            name = _ring_substituent_prefix(mol, nidx, idx)
        else:
            name = branch_prefix(mol, nidx, idx) or substituent_from_root(mol, nidx, idx)
        if not name or " " in name or "[" in name or "]" in name:
            return ""
        names.append(name)
    if not names:
        return ""
    counts = Counter(names)
    parts = []
    for name in sorted(counts):
        count = counts[name]
        part = _azaniumylidene_ligand_unit(name, count)
        if not part:
            return ""
        parts.append(part)
    return "".join(parts) + "azaniumylidene"


_COMPLEX_AZANIUMYLIDENE_MULTIPLIERS = {
    2: "bis",
    3: "tris",
    4: "tetrakis",
    5: "pentakis",
    6: "hexakis",
}


def _azaniumylidene_ligand_unit(name: str, count: int) -> str:
    if count < 1:
        return ""
    needs_parentheses = _azaniumylidene_ligand_needs_parentheses(name)
    if count == 1:
        return f"({name})" if needs_parentheses else name
    if needs_parentheses:
        multiplier = _COMPLEX_AZANIUMYLIDENE_MULTIPLIERS.get(count, "")
        return f"{multiplier}({name})" if multiplier else ""
    multiplier = MULT.get(count, "")
    return multiplier + name if multiplier else ""


def _azaniumylidene_ligand_needs_parentheses(name: str) -> bool:
    return name[:1].isdigit() or "-" in name or "," in name or name.startswith("(")


def _nitroimino_parent_site_supported(mol, parent_idx: int) -> bool:
    parent = mol.GetAtomWithIdx(parent_idx)
    if not parent.IsInRing():
        return False
    for ring in mol.GetRingInfo().AtomRings():
        if parent_idx not in ring or len(ring) != 5:
            continue
        symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring]
        if Counter(symbols) == Counter({"C": 2, "N": 3}) and all(
            mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring
        ):
            return False
    return True


def _isocyanato_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    for nbr in atom.GetNeighbors():
        cidx = nbr.GetIdx()
        if cidx == parent_idx or nbr.GetSymbol() != "C":
            continue
        n_c_bond = mol.GetBondBetweenAtoms(idx, cidx)
        if n_c_bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        terminal = [
            other
            for other in nbr.GetNeighbors()
            if other.GetIdx() != idx
            and mol.GetBondBetweenAtoms(cidx, other.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        ]
        if len(terminal) != 1:
            continue
        symbol = terminal[0].GetSymbol()
        if symbol == "O":
            return "isocyanato"
        if symbol == "S":
            return "isothiocyanato"
    return ""


def _diazenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx or nbr.GetSymbol() != "N" or nbr.GetFormalCharge() != 0:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        distal = [n.GetIdx() for n in nbr.GetNeighbors() if n.GetIdx() != idx]
        if not distal:
            return "diazenyl"
        if len(distal) != 1:
            continue
        didx = distal[0]
        if mol.GetAtomWithIdx(didx).IsInRing():
            name = _ring_substituent_prefix(mol, didx, nidx)
        else:
            name = substituent_from_root(mol, didx, nidx)
        if name:
            return name + "diazenyl"
    return ""


def _azoxy_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx or nbr.GetSymbol() != "N" or nbr.GetFormalCharge() <= 0:
            continue
        n_bond = mol.GetBondBetweenAtoms(idx, nidx)
        if n_bond is None or n_bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        oxido = []
        organic = []
        for distal in nbr.GetNeighbors():
            didx = distal.GetIdx()
            if didx == idx:
                continue
            bond = mol.GetBondBetweenAtoms(nidx, didx)
            if (
                distal.GetSymbol() == "O"
                and distal.GetFormalCharge() < 0
                and bond is not None
                and bond.GetBondType() == Chem.BondType.SINGLE
            ):
                oxido.append(didx)
            elif bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
                organic.append(didx)
        if len(oxido) != 1 or len(organic) != 1:
            continue
        didx = organic[0]
        organic_atom = mol.GetAtomWithIdx(didx)
        if organic_atom.IsInRing():
            name = _ring_substituent_prefix(mol, didx, nidx)
            if name and not (name.startswith("(") or "-" in name):
                return f"({name}azoxy)"
        else:
            name = substituent_from_root(mol, didx, nidx)
        if not name:
            continue
        if "-" in name or name.startswith("("):
            return f"({name})azoxy"
        return name + "azoxy"
    return ""


def _hydrazone_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [nbr for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1]
    nitrogens = [nbr for nbr in heavy if nbr.GetAtomicNum() == 7]
    if len(nitrogens) != 1 or len(heavy) > 2:
        return ""
    distal_n = nitrogens[0]
    distal_n_idx = distal_n.GetIdx()
    if distal_n.GetFormalCharge() != 0:
        return ""
    n_bond = mol.GetBondBetweenAtoms(idx, distal_n_idx)
    if n_bond is None or n_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    imine_carbons = [
        nbr.GetIdx()
        for nbr in distal_n.GetNeighbors()
        if nbr.GetIdx() != idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(distal_n_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]
    if len(imine_carbons) != 1:
        return ""
    if atom.GetIsotope() or distal_n.GetIsotope():
        return ""  # These nitrogen isotope roles need their own descriptor map.
    from ..hydrazone_methods import mapped_ylidene_prefix
    from ..formatting import enclose_name_unit
    ylidene = mapped_ylidene_prefix(mol, Chem, imine_carbons[0], distal_n_idx, idx)
    if not ylidene and len(heavy) == 2 and mol.GetBondBetweenAtoms(imine_carbons[0], distal_n_idx).GetStereo():
        return ""
    if not ylidene:
        ylidene = _imine_carbon_ylidene(mol, imine_carbons[0], distal_n_idx)
    if not ylidene:
        ylidene = _alkylidene_prefix(mol, imine_carbons[0], distal_n_idx)
    if not ylidene:
        return ""
    if len(heavy) == 1:
        return enclose_name_unit(ylidene) + "hydrazinyl"

    substituent = next(nbr for nbr in heavy if nbr.GetIdx() != distal_n_idx)
    root = substituent.GetIdx()
    if (atom.IsInRing() or substituent.GetAtomicNum() != 6
            or mol.GetBondBetweenAtoms(root, idx).GetBondType() != Chem.BondType.SINGLE):
        return ""
    from ..component_mapping import mapped_general_branch
    from ..formatting import format_prefixes

    name, owned = mapped_general_branch(mol, Chem, root, idx)
    if not name or owned & {parent_idx, distal_n_idx, imine_carbons[0]}:
        return ""
    prefixes = format_prefixes([("1", name), ("2", ylidene)], hierarchical_enclosures=True)
    return prefixes.rstrip("-") + "hydrazin-1-yl"


def _imino_amino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    imine_carbons = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]
    if len(imine_carbons) != 1:
        return ""
    from ..hydrazone_methods import mapped_ylidene_prefix
    from ..formatting import enclose_name_unit
    carbon = imine_carbons[0]
    ylidene = mapped_ylidene_prefix(mol, Chem, carbon, idx)
    if not ylidene:
        ylidene = _imine_carbon_ylidene(mol, carbon, idx)
    if not ylidene:
        return ""
    if (not ylidene.endswith("methylidene") or ylidene.startswith(("(E)-", "(Z)-"))
            or not mol.GetBondBetweenAtoms(carbon, idx).GetStereo()):
        return enclose_name_unit(ylidene) + "amino"
    # The mononuclear geometry belongs to the imine branch even when its
    # host or a separate branch has another configuration descriptor.
    from ..branch_rules import collect_subtree
    from ..stereochemistry import assigned_bond_stereochemistry

    descriptor = next((value for a, b, value in assigned_bond_stereochemistry(mol, Chem)
                       if {a, b} == {carbon, idx}), None)
    if descriptor is None:
        return ""
    text = enclose_name_unit(f"({descriptor})-{ylidene}") + "amino"
    scope = frozenset(collect_subtree(mol, idx, parent_idx))
    mol_cache_dict(mol, "rendered_mapped_branch_bond_stereo")[carbon, idx, text, scope] = descriptor
    return text


def _amidino_amino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]
    if len(heavy) != 1:
        return ""
    carbon_idx = heavy[0]
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return ""
    n_c_bond = mol.GetBondBetweenAtoms(idx, carbon_idx)
    if n_c_bond is None or n_c_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    terminal_imino = []
    other_heavy = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if (
            nbr.GetSymbol() == "N"
            and nbr.GetFormalCharge() == 0
            and bond is not None
            and bond.GetBondType() == Chem.BondType.DOUBLE
            and not [nnbr for nnbr in nbr.GetNeighbors() if nnbr.GetIdx() != carbon_idx and nnbr.GetAtomicNum() > 1]
        ):
            terminal_imino.append(nidx)
        else:
            other_heavy.append(nidx)
    if len(terminal_imino) != 1 or other_heavy:
        return ""
    return "iminomethylamino"


def _imine_carbon_ylidene(mol, carbon_idx: int, nitrogen_idx: int) -> str:
    atom = mol.GetAtomWithIdx(carbon_idx)
    if atom.IsInRing():
        rings = [ring for ring in mol.GetRingInfo().AtomRings() if carbon_idx in ring]
        if len(rings) != 1:
            return ""
        ring = rings[0]
        if not all(mol.GetAtomWithIdx(idx).GetSymbol() == "C" for idx in ring):
            return ""
        ring_set = set(ring)
        for a, b in zip(ring, ring[1:] + ring[:1]):
            bond = mol.GetBondBetweenAtoms(a, b)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return ""
        for ring_idx in ring:
            for nbr in mol.GetAtomWithIdx(ring_idx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx not in ring_set and nidx != nitrogen_idx:
                    return ""
        root = ROOTS.get(len(ring))
        return f"cyclo{root}ylidene" if root else ""
    substituents = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != nitrogen_idx]
    if len(substituents) == 1:
        sub_idx = substituents[0]
        sub_atom = mol.GetAtomWithIdx(sub_idx)
        sub_bond = mol.GetBondBetweenAtoms(carbon_idx, sub_idx)
        if (
            sub_atom.GetSymbol() == "N"
            and sub_atom.GetFormalCharge() == 0
            and sub_bond is not None
            and sub_bond.GetBondType() == Chem.BondType.SINGLE
            and not [nbr for nbr in sub_atom.GetNeighbors() if nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1]
        ):
            return "aminomethylidene"
        if (
            sub_atom.GetSymbol() == "N"
            and sub_atom.GetFormalCharge() == 0
            and sub_bond is not None
            and sub_bond.GetBondType() == Chem.BondType.SINGLE
        ):
            amino = _nitrogen_prefix(mol, sub_idx, carbon_idx)
            if amino.endswith(("amino", "amido", "anilino")):
                from ..formatting import enclose_name_unit
                # The substituted amino unit is a branch of methylidene.
                # Its own N ligands must not become substituents of that
                # carbon, e.g. ethyl(methyl)amino + methylidene.
                return (amino if amino == "amino" else enclose_name_unit(amino)) + "methylidene"
        if (
            sub_atom.GetSymbol() == "N"
            and sub_atom.GetFormalCharge() == 0
            and sub_bond is not None
            and sub_bond.GetBondType() == Chem.BondType.DOUBLE
            and not [nbr for nbr in sub_atom.GetNeighbors() if nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1]
        ):
            return "iminomethylidene"
        if mol.GetAtomWithIdx(sub_idx).IsInRing():
            name = _ring_substituent_prefix(mol, sub_idx, carbon_idx)
            if name and name.endswith("yl"):
                return name + "methylidene"
            return ""
        name = substituent_from_root(mol, sub_idx, carbon_idx)
        if name and "-" not in name and not name.startswith("("):
            return name + "methylidene"
        return ""
    if len(substituents) == 2:
        # In a guanidine-derived ylidene, keep each amino ligand attached
        # independently to the imine carbon. Their N ligands stay local.
        if all(mol.GetAtomWithIdx(i).GetAtomicNum() == 7
               and mol.GetBondBetweenAtoms(carbon_idx, i).GetBondTypeAsDouble() == 1
               for i in substituents):
            names = [_nitrogen_prefix(mol, i, carbon_idx) for i in substituents]
            if all(name.endswith(("amino", "amido", "anilino")) for name in names):
                return format_prefixes([("", name) for name in names], omit_locants=True,
                                       hierarchical_enclosures=True,
                                       separate_unlocanted=True) + "methylidene"
        functional_ethylidene = _branch_rules()._functionalized_ethylidene_prefix(
            mol,
            substituents,
            carbon_idx,
        )
        if functional_ethylidene:
            return functional_ethylidene
        names = []
        for sub_idx in substituents:
            if mol.GetAtomWithIdx(sub_idx).IsInRing():
                name = _ring_substituent_prefix(mol, sub_idx, carbon_idx)
            else:
                name = substituent_from_root(mol, sub_idx, carbon_idx)
            if not name or "-" in name or name.startswith("("):
                return ""
            names.append(name)
        counts = Counter(names)
        parts = []
        for name in sorted(counts):
            count = counts[name]
            parts.append((MULT.get(count, "") if count > 1 else "") + name)
        return "".join(parts) + "methylidene"
    return _alkylidene_prefix(mol, carbon_idx, nitrogen_idx)
