from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from .branch_rules import (
    branch_prefix,
    can_be_in_parent_skeleton,
    is_carbonyl_carbon,
    parent_chain_name,
    replacement_prefixes,
    rooted_paths,
    substituent_from_root,
    substituent_locants,
    collect_subtree,
)
from .constants import MULT, ROOTS
from .formatting import format_prefixes as _format_prefixes
from .graph_model import graph_view


MAX_ACYCLIC_PARENT_PATHS = 96


@dataclass(frozen=True, slots=True)
class _AcyclicGraphView:
    symbols: tuple[str, ...]
    neighbors: tuple[tuple[int, ...], ...]
    bond_orders: dict[tuple[int, int], float]


def name_acyclic(mol, allow_ring_atoms: bool = True) -> str:
    profile = _acyclic_functional_profile(mol)
    for signal, namer in (
        ("oxidanium", _name_oxidanium),
        ("ketene", _name_ketene),
        ("thioamide", _name_thioamide),
        ("ylidene_dioxosulfane", _name_ylidene_dioxosulfane),
        ("thione", _name_thione),
    ):
        if profile[signal]:
            name = namer(mol)
            if name:
                return name

    paths = _candidate_parent_paths(mol, allow_ring_atoms=allow_ring_atoms)
    if not paths:
        return ""

    return _best_acyclic_name_from_paths(mol, paths)


def _acyclic_functional_profile(mol) -> dict[str, bool]:
    cached = getattr(mol, "_nispo_acyclic_functional_profile", None)
    if cached is not None:
        return cached
    has_c_c_o_cumulene = False
    has_c_s_double = False
    has_c_s_n_thioamide = False
    has_c_sulfone_cumulene = False
    has_protonated_carbonyl = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        cidx = atom.GetIdx()
        double_sulfurs = []
        single_n = False
        double_c = []
        for nbr in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(cidx, nbr.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                if nbr.GetSymbol() == "S":
                    double_sulfurs.append(nbr.GetIdx())
                elif nbr.GetSymbol() == "O" and nbr.GetFormalCharge() > 0:
                    has_protonated_carbonyl = True
                elif nbr.GetSymbol() == "C":
                    double_c.append(nbr.GetIdx())
            elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetSymbol() == "N":
                single_n = True
        if double_sulfurs:
            has_c_s_double = True
            for sulfur_idx in double_sulfurs:
                if _is_dioxosulfane_sulfur(mol, sulfur_idx):
                    has_c_sulfone_cumulene = True
            if single_n:
                has_c_s_n_thioamide = True
        for carbon_idx in double_c:
            if any(
                onbr.GetSymbol() == "O"
                and mol.GetBondBetweenAtoms(carbon_idx, onbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                for onbr in mol.GetAtomWithIdx(carbon_idx).GetNeighbors()
                if onbr.GetIdx() != cidx
            ):
                has_c_c_o_cumulene = True
    profile = {
        "oxidanium": has_protonated_carbonyl,
        "ketene": has_c_c_o_cumulene,
        "thioamide": has_c_s_n_thioamide,
        "ylidene_dioxosulfane": has_c_sulfone_cumulene,
        "thione": has_c_s_double,
    }
    try:
        mol._nispo_acyclic_functional_profile = profile
    except Exception:
        pass
    return profile


def _name_oxidanium(mol) -> str:
    motifs = _oxidanium_motifs(mol)
    if len(motifs) != 1:
        return ""
    carbon_idx, oxygen_idx = motifs[0]
    paths = _oxidanium_parent_paths(mol, carbon_idx)
    orientations = _oriented_paths_by_quality(mol, paths)
    index = 0
    while index < len(orientations):
        descriptor = orientations[index][0]
        names = []
        while index < len(orientations) and orientations[index][0] == descriptor:
            _descriptor, oriented = orientations[index]
            name = _name_oxidanium_path(mol, oriented, carbon_idx, oxygen_idx)
            if name:
                names.append(name)
            index += 1
        if names:
            return min(names)
    return ""


def _oxidanium_motifs(mol) -> list[tuple[int, int]]:
    motifs = []
    for atom in mol.GetAtoms():
        cidx = atom.GetIdx()
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() != "O" or nbr.GetFormalCharge() <= 0:
                continue
            bond = mol.GetBondBetweenAtoms(cidx, nbr.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                motifs.append((cidx, nbr.GetIdx()))
    return motifs


def _oxidanium_parent_paths(mol, carbon_idx: int) -> list[tuple[int, ...]]:
    allowed = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if not atom.IsInRing() and can_be_in_parent_skeleton(mol, atom.GetIdx())
    }
    if carbon_idx not in allowed:
        return []
    view = _acyclic_graph_view(mol)
    endpoints = [
        idx
        for idx in allowed
        if idx == carbon_idx
        or (
            view.symbols[idx] == "C"
            and not is_carbonyl_carbon(mol, idx)
            and sum(1 for nidx in view.neighbors[idx] if nidx in allowed) <= 1
        )
    ]
    if len(endpoints) < 2:
        endpoints = [idx for idx in allowed if idx == carbon_idx or view.symbols[idx] == "C"]
    paths: set[tuple[int, ...]] = set()
    best_len = 0
    for endpoint in endpoints:
        if endpoint == carbon_idx:
            continue
        path = Chem.rdmolops.GetShortestPath(mol, carbon_idx, endpoint)
        if any(idx not in allowed for idx in path):
            continue
        path_len = len(path)
        if path_len > best_len:
            best_len = path_len
            paths.clear()
        if path_len == best_len:
            paths.add(tuple(path))
    if not paths:
        paths.add((carbon_idx,))
    return sorted(paths)


def _name_oxidanium_path(mol, path: Sequence[int], carbon_idx: int, oxygen_idx: int) -> str:
    if carbon_idx not in path:
        return ""
    prefixes = _path_substituent_prefixes(
        mol,
        path,
        skip_bonds={_bond_key(carbon_idx, oxygen_idx)},
    )
    if prefixes is None:
        return ""

    replacement_text = replacement_prefixes(mol, path)
    parent = parent_chain_name(mol, path, as_substituent=False)
    if not parent or not parent.endswith("e"):
        return ""
    locant = path.index(carbon_idx) + 1
    parent = replacement_text + parent if replacement_text else parent
    name = f"{parent[:-1]}-{locant}-ylideneoxidanium"
    prefix_text = _format_prefixes(prefixes)
    return prefix_text + "-" + name if prefix_text else name


def _best_acyclic_name_from_paths(mol, paths: Sequence[Sequence[int]]) -> str:
    orientations = _oriented_paths_by_quality(mol, paths)
    index = 0
    while index < len(orientations):
        descriptor = orientations[index][0]
        names = []
        while index < len(orientations) and orientations[index][0] == descriptor:
            _descriptor, oriented = orientations[index]
            name = _name_from_parent_path(mol, oriented, as_substituent=False)
            if name:
                names.append(name)
            index += 1
        if names:
            return min(names)
    return ""


def _name_ketene(mol) -> str:
    motifs = _ketene_motifs(mol)
    if len(motifs) != 1:
        return ""
    alpha_idx, carbonyl_idx, oxygen_idx = motifs[0]
    paths = [path for path in _candidate_parent_paths(mol) if alpha_idx in path]
    orientations = _oriented_paths_by_quality(mol, paths)
    index = 0
    while index < len(orientations):
        descriptor = orientations[index][0]
        names = []
        while index < len(orientations) and orientations[index][0] == descriptor:
            _descriptor, oriented = orientations[index]
            if alpha_idx in oriented:
                name = _name_ketene_path(mol, oriented, alpha_idx, carbonyl_idx, oxygen_idx)
                if name:
                    names.append(name)
            index += 1
        if names:
            return min(names)
    return ""


def _ketene_motifs(mol) -> list[tuple[int, int, int]]:
    motifs = []
    for atom in mol.GetAtoms():
        alpha_idx = atom.GetIdx()
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
            continue
        for nbr in atom.GetNeighbors():
            carbonyl_idx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(alpha_idx, carbonyl_idx)
            if nbr.GetSymbol() != "C" or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            oxygens = [
                onbr.GetIdx()
                for onbr in nbr.GetNeighbors()
                if onbr.GetIdx() != alpha_idx
                and onbr.GetSymbol() == "O"
                and mol.GetBondBetweenAtoms(carbonyl_idx, onbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            ]
            if len(oxygens) == 1:
                motifs.append((alpha_idx, carbonyl_idx, oxygens[0]))
    return motifs


def _name_ketene_path(
    mol, path: Sequence[int], alpha_idx: int, carbonyl_idx: int, oxygen_idx: int
) -> str:
    prefixes = _path_substituent_prefixes(
        mol,
        path,
        skip_bonds={_bond_key(alpha_idx, carbonyl_idx)},
    )
    if prefixes is None:
        return ""

    replacement_text = replacement_prefixes(mol, path)
    parent = parent_chain_name(mol, path, as_substituent=False)
    if not parent:
        return ""
    if replacement_text:
        parent = replacement_text + parent
    if not parent.endswith("e"):
        return ""
    locant = path.index(alpha_idx) + 1
    ketene_parent = f"{parent[:-1]}-{locant}-ylidenemethanone"
    prefix_text = _format_prefixes(prefixes)
    return prefix_text + "-" + ketene_parent if prefix_text else ketene_parent


def _candidate_parent_paths(mol, allow_ring_atoms: bool = True) -> list[tuple[int, ...]]:
    view = _acyclic_graph_view(mol)
    allowed = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if (allow_ring_atoms or not atom.IsInRing())
        and can_be_in_parent_skeleton(mol, atom.GetIdx())
    ]
    allowed_set = set(allowed)
    if not allowed:
        return []

    endpoints = [
        idx
        for idx in allowed
        if view.symbols[idx] == "C"
        and not is_carbonyl_carbon(mol, idx)
    ]
    if len(endpoints) < 2:
        endpoints = [idx for idx in allowed if view.symbols[idx] == "C"]
    if not endpoints:
        endpoints = allowed

    endpoint_groups = _allowed_endpoint_groups(mol, allowed_set, endpoints, view)
    paths: set[tuple[int, ...]] = set()
    best_len = 0
    for endpoint_group in endpoint_groups:
        for a, b in combinations(endpoint_group, 2):
            path = Chem.rdmolops.GetShortestPath(mol, a, b)
            if any(idx not in allowed_set for idx in path):
                continue
            path_len = len(path)
            if path_len > best_len:
                best_len = path_len
            paths.add(tuple(path))
    if not paths:
        for idx in endpoints:
            paths.add((idx,))
    min_len = max(1, best_len - 4)
    return sorted(
        (path for path in paths if len(path) >= min_len),
        key=lambda path: (-len(path), path),
    )[:MAX_ACYCLIC_PARENT_PATHS]


def _allowed_endpoint_groups(
    mol,
    allowed_set: set[int],
    endpoints: Sequence[int],
    view: _AcyclicGraphView | None = None,
) -> list[tuple[int, ...]]:
    """Group endpoints by the allowed skeleton graph before shortest-path probes."""

    view = view or _acyclic_graph_view(mol)
    endpoint_set = set(endpoints)
    remaining = set(allowed_set)
    groups: list[tuple[int, ...]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            idx = stack.pop()
            for nidx in view.neighbors[idx]:
                if nidx not in remaining:
                    continue
                remaining.remove(nidx)
                component.add(nidx)
                stack.append(nidx)
        group = tuple(sorted(component & endpoint_set))
        if len(group) >= 2:
            groups.append(group)
    return groups


def _oriented_paths_by_quality(mol, paths: Sequence[Sequence[int]]) -> list[tuple[tuple, tuple[int, ...]]]:
    orientations = []
    seen = set()
    for path in paths:
        forward = tuple(path)
        reverse = tuple(reversed(path))
        for oriented in (forward, reverse):
            if oriented in seen:
                continue
            seen.add(oriented)
            orientations.append((_path_quality_descriptor(mol, oriented), oriented))
    orientations.sort(key=lambda item: item[0])
    return orientations


def _path_quality_descriptor(mol, path: Sequence[int]) -> tuple:
    path = tuple(path)
    cache = getattr(mol, "_nispo_path_quality_cache", None)
    if cache is None:
        cache = {}
        try:
            mol._nispo_path_quality_cache = cache
        except Exception:
            cache = None
    if cache is not None:
        cached = cache.get(path)
        if cached is not None:
            return cached
    view = _acyclic_graph_view(mol)
    path_set = set(path)
    locants = []
    carbon_count = 0
    suffix_bonus = 0
    for pos, idx in enumerate(path, start=1):
        if view.symbols[idx] == "C":
            carbon_count += 1
        if view.symbols[idx] != "C":
            locants.append(pos)
        for nidx in view.neighbors[idx]:
            if nidx not in path_set:
                locants.append(pos)
    for bond_locant, (a, b) in enumerate(zip(path, path[1:]), start=1):
        if view.bond_orders.get(_bond_key(a, b), 1.0) != 1.0:
            locants.append(bond_locant)
            suffix_bonus += 1
    descriptor = (-len(path), -suffix_bonus, tuple(locants), -carbon_count)
    if cache is not None:
        cache[path] = descriptor
    return descriptor


def _acyclic_graph_view(mol) -> _AcyclicGraphView:
    cached = getattr(mol, "_nispo_acyclic_graph_view", None)
    if cached is not None:
        return cached
    graph = graph_view(mol, Chem)
    view = _AcyclicGraphView(symbols=graph.symbols, neighbors=graph.neighbors, bond_orders=graph.bond_orders)
    try:
        mol._nispo_acyclic_graph_view = view
    except Exception:
        pass
    return view


def _bond_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _path_substituent_prefixes(
    mol,
    path: Sequence[int],
    *,
    skip_bonds: set[tuple[int, int]] | None = None,
    skip_atoms: set[int] | None = None,
    allowed_atoms: set[int] | None = None,
    skip_plain_hydrogen: bool = False,
) -> list[tuple[str, str]] | None:
    path_set = set(path)
    skip_bonds = skip_bonds or set()
    skip_atoms = skip_atoms or set()
    prefixes: list[tuple[str, str]] = []
    for pos, idx in enumerate(path, start=1):
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in path_set or nidx in skip_atoms:
                continue
            if allowed_atoms is not None and nidx not in allowed_atoms:
                continue
            if _bond_key(idx, nidx) in skip_bonds:
                continue
            if skip_plain_hydrogen and nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0:
                continue
            prefix = branch_prefix(mol, nidx, idx)
            if not prefix:
                return None
            prefixes.append((str(pos), prefix))
    return prefixes


def _name_from_parent_path(mol, path: Sequence[int], as_substituent: bool) -> str:
    if not path:
        return ""
    prefixes = _path_substituent_prefixes(mol, path, skip_plain_hydrogen=True)
    if prefixes is None:
        return ""

    replacement_text = replacement_prefixes(mol, path)
    parent = parent_chain_name(mol, path, as_substituent=as_substituent)
    if not parent:
        return ""
    prefix_text = _format_prefixes(prefixes)
    pieces = []
    if prefix_text:
        pieces.append(prefix_text)
    if replacement_text:
        pieces.append(replacement_text + parent)
    else:
        pieces.append(parent)
    return "-".join(pieces)


def _name_thioamide(mol) -> str:
    candidates = []
    for atom in mol.GetAtoms():
        cidx = atom.GetIdx()
        if atom.GetSymbol() != "C":
            continue
        double_s = []
        single_n = []
        carbon_neighbors = []
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_s.append(nidx)
            elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
                single_n.append(nidx)
            elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                carbon_neighbors.append(nidx)
            else:
                carbon_neighbors = []
                break
        if len(double_s) != 1 or len(single_n) != 1 or len(carbon_neighbors) > 1:
            continue
        parent = _thioamide_parent_name(mol, cidx, double_s[0], single_n[0])
        if not parent:
            continue
        n_prefix = _thioamide_n_prefix(mol, single_n[0], cidx)
        if n_prefix is None:
            continue
        name = n_prefix + parent if n_prefix else parent
        candidates.append((len(name), name))
    return min(candidates)[1] if candidates else ""


def _name_thione(mol) -> str:
    motif = _single_exocyclic_sulfur_motif(mol)
    if motif is None:
        return ""
    thione_idx, sulfur_idx = motif
    paths = _thione_parent_paths(mol, thione_idx)
    orientations = _oriented_paths_by_quality(mol, paths)
    index = 0
    while index < len(orientations):
        descriptor = orientations[index][0]
        names = []
        while index < len(orientations) and orientations[index][0] == descriptor:
            _descriptor, oriented = orientations[index]
            if thione_idx in oriented:
                name = _name_thione_path(mol, oriented, thione_idx, sulfur_idx)
                if name:
                    names.append(name)
            index += 1
        if names:
            return min(names)
    return ""


def _name_ylidene_dioxosulfane(mol) -> str:
    motif = _single_exocyclic_sulfur_motif(mol, sulfur_predicate=_is_dioxosulfane_sulfur)
    if motif is None:
        return ""
    ylidene_idx, sulfur_idx = motif
    paths = _thione_parent_paths(mol, ylidene_idx)
    orientations = _oriented_paths_by_quality(mol, paths)
    index = 0
    while index < len(orientations):
        descriptor = orientations[index][0]
        names = []
        while index < len(orientations) and orientations[index][0] == descriptor:
            _descriptor, oriented = orientations[index]
            if ylidene_idx in oriented:
                name = _name_ylidene_dioxosulfane_path(mol, oriented, ylidene_idx, sulfur_idx)
                if name:
                    names.append(name)
            index += 1
        if names:
            return min(names)
    return ""


def _single_exocyclic_sulfur_motif(mol, sulfur_predicate=None) -> tuple[int, int] | None:
    motifs = []
    for atom in mol.GetAtoms():
        cidx = atom.GetIdx()
        if atom.GetSymbol() != "C":
            continue
        sulfur = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetSymbol() == "S"
            and mol.GetBondBetweenAtoms(cidx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            and (sulfur_predicate is None or sulfur_predicate(mol, nbr.GetIdx()))
        ]
        if sulfur:
            if len(sulfur) != 1:
                return None
            motifs.append((cidx, sulfur[0]))
    if len(motifs) != 1:
        return None
    return motifs[0]


def _is_dioxosulfane_sulfur(mol, sulfur_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(sulfur_idx)
    if atom.GetSymbol() != "S":
        return False
    double_oxygen_count = 0
    for nbr in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nbr.GetIdx())
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
    return double_oxygen_count == 2


def _thione_parent_paths(mol, thione_idx: int) -> list[tuple[int, ...]]:
    carbons = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "C" and not atom.IsInRing()}
    if thione_idx not in carbons:
        return []
    endpoints = [
        idx
        for idx in carbons
        if sum(1 for nbr in mol.GetAtomWithIdx(idx).GetNeighbors() if nbr.GetIdx() in carbons) <= 1
    ]
    if not endpoints:
        endpoints = list(carbons)
    paths = set()
    best_len = 0
    for a, b in combinations(endpoints, 2):
        path = Chem.rdmolops.GetShortestPath(mol, a, b)
        if thione_idx not in path or any(idx not in carbons for idx in path):
            continue
        if len(path) > best_len:
            best_len = len(path)
            paths.clear()
        if len(path) == best_len:
            paths.add(tuple(path))
    if not paths:
        paths.add((thione_idx,))
    return sorted(paths)


def _name_ylidene_dioxosulfane_path(
    mol,
    path: Sequence[int],
    ylidene_idx: int,
    sulfur_idx: int,
) -> str:
    return _name_exocyclic_sulfur_suffix_path(
        mol,
        path,
        center_idx=ylidene_idx,
        sulfur_idx=sulfur_idx,
        suffix="ylidene-dioxosulfane",
    )


def _name_thione_path(mol, path: Sequence[int], thione_idx: int, sulfur_idx: int) -> str:
    return _name_exocyclic_sulfur_suffix_path(
        mol,
        path,
        center_idx=thione_idx,
        sulfur_idx=sulfur_idx,
        suffix="thione",
    )


def _name_exocyclic_sulfur_suffix_path(
    mol,
    path: Sequence[int],
    *,
    center_idx: int,
    sulfur_idx: int,
    suffix: str,
) -> str:
    if any(mol.GetAtomWithIdx(idx).GetSymbol() != "C" for idx in path):
        return ""
    prefixes = _path_substituent_prefixes(
        mol,
        path,
        skip_bonds={_bond_key(center_idx, sulfur_idx)},
    )
    if prefixes is None:
        return ""
    parent = parent_chain_name(mol, path, as_substituent=False)
    if not parent or not parent.endswith("ane"):
        return ""
    locant = path.index(center_idx) + 1
    parent = parent[:-1] + f"-{locant}-{suffix}"
    prefix_text = _format_prefixes(prefixes)
    return prefix_text + "-" + parent if prefix_text else parent


def _thioamide_parent_name(mol, carbonyl_idx: int, sulfur_idx: int, nitrogen_idx: int) -> str:
    atoms = collect_subtree(mol, carbonyl_idx, nitrogen_idx)
    paths = rooted_paths(mol, carbonyl_idx, atoms)
    best = ""
    best_key = None
    for path in paths:
        if not path or path[0] != carbonyl_idx:
            continue
        if any(mol.GetAtomWithIdx(atom_idx).GetSymbol() != "C" for atom_idx in path):
            continue
        root = ROOTS.get(len(path))
        if not root:
            continue
        prefixes = _path_substituent_prefixes(
            mol,
            path,
            skip_bonds={_bond_key(carbonyl_idx, sulfur_idx)},
            skip_atoms={nitrogen_idx},
            allowed_atoms=atoms,
        )
        if prefixes is None:
            continue
        prefix_text = _format_prefixes(prefixes)
        name = root + "anethioamide"
        if prefix_text:
            name = prefix_text + name
        key = (-len(path), substituent_locants(mol, path, atoms, nitrogen_idx), name)
        if best_key is None or key < best_key:
            best_key = key
            best = name
    return best


def _thioamide_n_prefix(mol, nitrogen_idx: int, carbonyl_idx: int) -> str | None:
    groups = []
    for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbonyl_idx:
            continue
        name = substituent_from_root(mol, nidx, nitrogen_idx)
        if not name:
            return None
        groups.append(name)
    if not groups:
        return ""
    counts = {}
    for name in groups:
        counts[name] = counts.get(name, 0) + 1
    parts = []
    for name in sorted(counts):
        count = counts[name]
        locants = ",".join("N" for _ in range(count))
        if count == 1:
            parts.append(f"{locants}-{name}")
        else:
            parts.append(f"{locants}-{MULT.get(count, '')}{name}")
    return "-".join(parts) + "-"
