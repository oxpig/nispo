from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .constants import MULT, REPLACEMENT, ROOTS, SKELETON_ATOMS
from .formatting import join_locants as _join_locants, loc_unsat, restore_terminal_unsaturation_e
from .molecule_cache import mol_cache_dict
from .ring.hydrogen import _drop_indicated_hydrogen_prefix, _indicated_hydrogen_prefix


@dataclass(frozen=True, slots=True)
class _SaturatedHeterocycleSpec:
    size: int
    counts: tuple[tuple[str, int], ...]
    anchor_symbol: str
    parent: str
    locant_symbol: str = ""
    locants: tuple[int, ...] = ()


_SATURATED_HETEROCYCLE_SPECS: tuple[_SaturatedHeterocycleSpec, ...] = (
    _SaturatedHeterocycleSpec(3, (("C", 2), ("N", 1)), "N", "aziridine"),
    _SaturatedHeterocycleSpec(3, (("C", 2), ("O", 1)), "O", "oxirane"),
    _SaturatedHeterocycleSpec(3, (("C", 2), ("S", 1)), "S", "thiirane"),
    _SaturatedHeterocycleSpec(4, (("C", 3), ("N", 1)), "N", "azetidine"),
    _SaturatedHeterocycleSpec(4, (("C", 3), ("O", 1)), "O", "oxetane"),
    _SaturatedHeterocycleSpec(4, (("C", 3), ("S", 1)), "S", "thietane"),
    _SaturatedHeterocycleSpec(5, (("B", 1), ("C", 2), ("O", 2)), "O", "1,3,2-dioxaborolane", "O", (1, 3)),
    _SaturatedHeterocycleSpec(5, (("C", 3), ("O", 2)), "O", "1,3-dioxolane", "O", (1, 3)),
    _SaturatedHeterocycleSpec(5, (("C", 3), ("N", 2)), "N", "imidazolidine", "N", (1, 3)),
    _SaturatedHeterocycleSpec(5, (("C", 4), ("N", 1)), "N", "pyrrolidine"),
    _SaturatedHeterocycleSpec(5, (("C", 4), ("O", 1)), "O", "oxolane"),
    _SaturatedHeterocycleSpec(6, (("C", 4), ("O", 2)), "O", "1,3-dioxane", "O", (1, 3)),
    _SaturatedHeterocycleSpec(6, (("C", 4), ("O", 2)), "O", "1,4-dioxane", "O", (1, 4)),
    _SaturatedHeterocycleSpec(6, (("C", 5), ("N", 1)), "N", "piperidine"),
    _SaturatedHeterocycleSpec(6, (("C", 5), ("O", 1)), "O", "oxane"),
    _SaturatedHeterocycleSpec(6, (("C", 5), ("S", 1)), "S", "thiane"),
    _SaturatedHeterocycleSpec(6, (("C", 4), ("N", 2)), "N", "piperazine", "N", (1, 4)),
    _SaturatedHeterocycleSpec(6, (("C", 4), ("N", 1), ("O", 1)), "O", "morpholine", "N", (4,)),
    _SaturatedHeterocycleSpec(6, (("C", 4), ("N", 1), ("S", 1)), "S", "thiomorpholine", "N", (4,)),
    _SaturatedHeterocycleSpec(6, (("C", 4), ("O", 1), ("S", 1)), "O", "1,4-oxathiane", "S", (4,)),
    _SaturatedHeterocycleSpec(7, (("C", 6), ("N", 1)), "N", "azepane"),
    _SaturatedHeterocycleSpec(7, (("C", 6), ("O", 1)), "O", "oxepane"),
    _SaturatedHeterocycleSpec(7, (("C", 6), ("S", 1)), "S", "thiepane"),
)


_SATURATED_SUBSTITUENT_STEMS = {
    "aziridine": "aziridin",
    "azetidine": "azetidin",
    "pyrrolidine": "pyrrolidin",
    "azepane": "azepan",
    "oxirane": "oxiran",
    "oxetane": "oxetan",
    "oxolane": "oxolan",
    "oxane": "oxan",
    "oxepane": "oxepan",
    "thiirane": "thiiran",
    "thietane": "thietan",
    "thiane": "thian",
    "thiepane": "thiepan",
    "thiomorpholine": "thiomorpholin",
    "1,4-oxathiane": "1,4-oxathian",
    "1,3-dioxolane": "1,3-dioxolan",
    "1,3-dioxane": "1,3-dioxan",
    "1,4-dioxane": "1,4-dioxan",
}


_CHARGEABLE_SUBSTITUENT_STEMS = {
    "pyridine": "pyridin",
    "pyrimidine": "pyrimidin",
    "pyrazine": "pyrazin",
    "pyridazine": "pyridazin",
    "thiophene": "thiophen",
    "thiazole": "thiazol",
    "isothiazole": "isothiazol",
    "imidazolidine": "imidazolidin",
    "piperidine": "piperidin",
    "piperazine": "piperazin",
    "morpholine": "morpholin",
}

_AROMATIC_N_H_CATION_PARENTS = {
    "imidazole",
    "pyrazole",
    "triazole",
    "tetrazole",
    "oxazole",
    "isoxazole",
    "thiazole",
    "isothiazole",
    "oxadiazole",
    "thiadiazole",
}


_NEUTRAL_SUBSTITUENT_STEMS = {
    "furan": "furan",
    "oxazole": "oxazol",
    "isoxazole": "isoxazol",
}


_INDICATED_H_SUBSTITUENT_STEMS = {
    "1H-pyrrole": "1H-pyrrol",
    "1H-1,2,3-triazole": "1H-1,2,3-triazol",
    "1H-1,2,4-triazole": "1H-1,2,4-triazol",
    "1H-tetrazole": "1H-tetrazol",
    "3H-diazirine": "3H-diazirin",
}


_AROMATIC_6_DIAZINE_NAMES = {2: "pyridazine", 3: "pyrimidine", 4: "pyrazine"}
_AROMATIC_6_TRIAZINE_NAMES = {
    (1, 3, 5): "s-triazine",
    (1, 2, 4): "1,2,4-triazine",
    (1, 2, 3): "v-triazine",
}
_AROMATIC_5_SINGLE_HETERO_NAMES = {
    "B": "borole",
    "N": "1H-pyrrole",
    "O": "furan",
    "P": "phosphole",
    "S": "thiophene",
    "Se": "selenole",
    "Si": "silole",
}
_AROMATIC_5_DIAZOLE_NAMES = {2: "pyrazole", 3: "imidazole"}
_AROMATIC_5_TRIAZOLE_NAMES = {
    (1, 2, 3): "1,2,3-triazole",
    (1, 2, 4): "1,2,4-triazole",
}
_AROMATIC_5_AZOLE_NAMES = {
    "O": {2: "isoxazole", 3: "oxazole"},
    "S": {2: "isothiazole", 3: "thiazole"},
}
_AROMATIC_5_DIAZOLE_HETERO_FAMILIES = {"O": "oxadiazole", "S": "thiadiazole"}
_AROMATIC_5_DIAZOLE_HETERO_LOCANTS = {
    (1, 2, 3): "1,2,3",
    (1, 2, 4): "1,2,4",
    (1, 2, 5): "1,2,5",
    (1, 3, 4): "1,3,4",
}
_MEDIUM_AROMATIC_N_RING_PARENT = {
    (8, 1): "azocine",
    (8, 2): "diazocine",
    (8, 3): "triazocine",
    (8, 4): "tetrazocine",
    (9, 1): "azonine",
    (9, 2): "diazonine",
    (9, 3): "triazonine",
    (9, 4): "tetrazonine",
}


def _single_indicated_hydrogen_n(mol, ordered: Sequence[int]) -> int | None:
    atoms = [
        idx
        for idx in ordered
        if mol.GetAtomWithIdx(idx).GetSymbol() == "N" and mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0
    ]
    return atoms[0] if len(atoms) == 1 else None


def ring_parent_candidates(mol, ordered: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
    cache = mol_cache_dict(mol, "ring_parent_candidates")
    cache_key = tuple(ordered)
    cached = cache.get(cache_key)
    if cached is not None:
        return list(cached)
    candidates = _ring_parent_candidates_uncached(mol, ordered)
    cache[cache_key] = tuple(candidates)
    return candidates


def _ring_parent_candidates_uncached(mol, ordered: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
    n = len(ordered)
    symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in ordered]
    aromatic = [mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ordered]
    all_aromatic = all(aromatic)
    all_aliphatic = not any(aromatic)
    all_single_bonds = all(
        mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() == 1.0
        for a, b in zip(ordered, ordered[1:] + ordered[:1])
    )
    kekulized_aromatic = _kekulized_aromatic_ring(mol, ordered, symbols)
    candidates: list[tuple[str, tuple[int, ...]]] = []

    candidates = _simple_iodine_ring_candidates(mol, ordered, symbols)
    if candidates:
        return candidates

    if n == 3 and (all_aromatic or kekulized_aromatic) and Counter(symbols) == Counter({"O": 3}):
        candidates.extend(_ring_pattern_candidates(mol, ordered, "1,2,3-trioxirene", ("O", "O", "O")))
        if candidates:
            return candidates

    if n == 3 and symbols.count("C") == 1 and symbols.count("N") == 2:
        candidates.extend(
            _ring_pattern_candidates(
                mol,
                ordered,
                "3H-diazirine",
                ("N", "N", "C"),
                checks=(
                    lambda mol, seq, _ring_set: (bond := mol.GetBondBetweenAtoms(seq[0], seq[1]))
                    is not None
                    and bond.GetBondTypeAsDouble() == 2.0,
                ),
            )
        )
        if candidates:
            return candidates

    if n == 4 and symbols.count("C") == 3 and symbols.count("N") == 1:
        candidates.extend(
            _ring_pattern_candidates(
                mol,
                ordered,
                "azet-2-one",
                ("N", "C", "C", "C"),
                checks=(
                    lambda mol, seq, ring_set: _has_exocyclic_oxo(mol, seq[1], ring_set),
                    lambda mol, seq, _ring_set: (bond := mol.GetBondBetweenAtoms(seq[2], seq[3]))
                    is not None
                    and bond.GetBondTypeAsDouble() == 2.0,
                ),
            )
        )
        if candidates:
            return candidates

    if n == 4 and (all_aromatic or kekulized_aromatic):
        candidates = _four_membered_aromatic_heterocycle_candidates(mol, ordered, Counter(symbols))
        if candidates:
            return candidates

    if n == 4 and (all_aromatic or kekulized_aromatic) and all(sym == "C" for sym in symbols):
        candidates = _cyclobutene_chalcogenone_candidates(mol, ordered)
        if candidates:
            return candidates

    if n == 6 and (all_aromatic or kekulized_aromatic):
        candidates = _six_membered_aromatic_candidates(mol, ordered, symbols)
        if candidates:
            return candidates

    if (
        n in {6, 7}
        and (all_aromatic or kekulized_aromatic or not all_single_bonds)
        and _ring_has_exocyclic_suffix_site(mol, ordered)
    ):
        candidates = _unsaturated_heterocycle_suffix_parent_candidates(mol, ordered, Counter(symbols))
        if candidates:
            return candidates

    if (
        all_aromatic
        and all(sym == "C" for sym in symbols)
        and 3 <= n <= max(ROOTS)
        and _aromatic_carbocycle_has_supported_valence(mol, ordered)
    ):
        for seq in _ring_numberings(ordered):
            parent = _aromatic_carbocycle_parent(seq)
            if parent:
                candidates.append((parent, seq))
        if candidates:
            return candidates

    if n == 5 and (all_aromatic or kekulized_aromatic):
        counts = Counter(symbols)
        if counts == Counter({"C": 2, "N": 1, "O": 2}):
            candidates.extend(
                _ring_pattern_candidates(
                    mol,
                    ordered,
                    "1,3,4-dioxazol-2-one",
                    ("O", "C", "O", "N", "C"),
                    checks=(lambda mol, seq, ring_set: _has_exocyclic_oxo(mol, seq[1], ring_set),),
                )
            )
            if candidates:
                return candidates
        candidates = _five_membered_aromatic_candidates(mol, ordered, counts)
        if candidates:
            return candidates

    if n in {8, 9} and (all_aromatic or kekulized_aromatic):
        candidates = _medium_aromatic_heterocycle_candidates(mol, ordered, Counter(symbols))
        if candidates:
            return candidates

    if (
        all_aromatic
        and 3 <= n <= max(ROOTS)
        and all(sym in SKELETON_ATOMS for sym in symbols)
        and any(sym != "C" for sym in symbols)
        and _aromatic_replacement_ring_supported(mol, ordered)
    ):
        for seq in _ring_numberings(ordered):
            if mol.GetAtomWithIdx(seq[0]).GetSymbol() == "C":
                continue
            parent = _aromatic_heterocycle_parent(mol, seq)
            if parent:
                candidates.append((parent, seq))
        if candidates:
            return candidates

    if n == 6 and symbols.count("N") == 2 and symbols.count("C") == 4 and not all_aromatic:
        candidates.extend(_unsaturated_diazine_candidates(mol, ordered))
        if candidates:
            return candidates

    if n == 6 and symbols.count("N") == 1 and symbols.count("C") == 5 and not all_aromatic:
        candidates.extend(_partially_hydrogenated_pyridine_candidates(mol, ordered))
        if candidates:
            return candidates

    if n == 6 and symbols.count("O") == 1 and symbols.count("C") == 5 and not all_aromatic:
        candidates.extend(_unsaturated_pyran_candidates(mol, ordered))
        if candidates:
            return candidates

    if n == 5 and symbols.count("N") == 1 and symbols.count("C") == 4:
        candidates.extend(_cyclic_imide_candidates(mol, ordered))
        if candidates:
            return candidates

    if all_aliphatic and all(sym == "C" for sym in symbols) and 3 <= n <= max(ROOTS):
        for seq in _ring_numberings(ordered):
            parent = _carbocycle_parent(mol, seq)
            if parent:
                candidates.append((parent, seq))
        if candidates:
            return candidates

    candidates = _positive_cn_heterocycle_candidates(mol, ordered, symbols, all_single_bonds)
    if candidates:
        return candidates

    if (
        all_aliphatic
        and 3 <= n <= max(ROOTS)
        and all(sym in SKELETON_ATOMS for sym in symbols)
        and any(sym != "C" for sym in symbols)
        and not all_single_bonds
    ):
        for seq in _ring_numberings(ordered):
            if mol.GetAtomWithIdx(seq[0]).GetSymbol() == "C":
                continue
            parent = _heterocycle_parent(mol, seq)
            if parent:
                candidates.append((parent, seq))
        if candidates:
            return candidates

    if n == 5 and symbols.count("N") == 2 and symbols.count("C") == 3 and not all_single_bonds:
        charged_n = [
            idx
            for idx in ordered
            if mol.GetAtomWithIdx(idx).GetSymbol() == "N" and mol.GetAtomWithIdx(idx).GetFormalCharge() > 0
        ]
        if len(charged_n) == 1:
            for seq in _numberings_with_atom_at(ordered, mol, "N"):
                if seq[0] == charged_n[0]:
                    candidates.append(("1H-imidazol-1-ium", seq))
            if candidates:
                return candidates

    if all_aliphatic and all_single_bonds:
        candidates = _fixed_saturated_heterocycle_candidates(mol, ordered, symbols)
        if candidates:
            return candidates
        if 3 <= n <= max(ROOTS) and all(sym in SKELETON_ATOMS for sym in symbols) and any(sym != "C" for sym in symbols):
            for seq in _ring_numberings(ordered):
                if mol.GetAtomWithIdx(seq[0]).GetSymbol() == "C":
                    continue
                parent = _heterocycloalkane_parent(mol, seq)
                if parent:
                    candidates.append((parent, seq))
            return candidates

    return candidates


def _positive_cn_heterocycle_candidates(
    mol,
    ordered: Sequence[int],
    symbols: Sequence[str],
    all_single_bonds: bool,
) -> list[tuple[str, tuple[int, ...]]]:
    if any(sym not in {"C", "N"} for sym in symbols):
        return []
    if any(mol.GetAtomWithIdx(idx).GetFormalCharge() < 0 for idx in ordered):
        return []
    charged_n = [
        idx
        for idx in ordered
        if mol.GetAtomWithIdx(idx).GetSymbol() == "N" and mol.GetAtomWithIdx(idx).GetFormalCharge() > 0
    ]
    if len(charged_n) != 1:
        return []
    charged_idx = charged_n[0]
    counts = Counter(symbols)
    double_count = sum(
        1
        for left, right in zip(ordered, ordered[1:] + ordered[:1])
        if (bond := mol.GetBondBetweenAtoms(left, right)) is not None and bond.GetBondTypeAsDouble() == 2.0
    )
    if all_single_bonds:
        if len(ordered) == 4 and counts == Counter({"C": 2, "N": 2}):
            return _positive_cn_ring_pattern_candidates(
                mol,
                ordered,
                charged_idx,
                (
                    ("1,2-diazetidine", ("N", "N", "C", "C")),
                    ("1,3-diazetidine", ("N", "C", "N", "C")),
                ),
            )
        return []
    if len(ordered) == 5 and counts == Counter({"C": 3, "N": 2}):
        return _positive_diazole_diazoline_candidates(mol, ordered, charged_idx, double_count)
    if len(ordered) == 5 and counts == Counter({"C": 2, "N": 3}) and double_count == 1:
        return _positive_triazoline_candidates(mol, ordered, charged_idx)
    return []


def _positive_diazole_diazoline_candidates(
    mol,
    ordered: Sequence[int],
    charged_idx: int,
    double_count: int,
) -> list[tuple[str, tuple[int, ...]]]:
    candidates: list[tuple[str, tuple[int, ...]]] = []
    for seq in _numberings_with_atom_at(ordered, mol, "N"):
        if seq[0] != charged_idx:
            continue
        n_locs = [i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "N"]
        second = sorted(n_locs)[1]
        if double_count >= 2:
            parent = _AROMATIC_5_DIAZOLE_NAMES.get(second, "")
            if parent:
                indicated = charged_idx if mol.GetAtomWithIdx(charged_idx).GetTotalNumHs() > 0 else None
                candidates.append((_indicated_hydrogen_prefix(mol, ordered, seq, indicated) + parent, seq))
        elif double_count == 1:
            parent = {2: "pyrazoline", 3: "imidazoline"}.get(second, "")
            if parent:
                candidates.append((parent, seq))
    return candidates


def _positive_triazoline_candidates(
    mol,
    ordered: Sequence[int],
    charged_idx: int,
) -> list[tuple[str, tuple[int, ...]]]:
    candidates: list[tuple[str, tuple[int, ...]]] = []
    parent_by_locs = {
        (1, 2, 3): "1,2,3-triazoline",
        (1, 2, 4): "1,2,4-triazoline",
    }
    for seq in _numberings_with_atom_at(ordered, mol, "N"):
        if seq[0] != charged_idx:
            continue
        n_locs = tuple(sorted(i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "N"))
        parent = parent_by_locs.get(n_locs, "")
        if parent:
            candidates.append((parent, seq))
    return candidates


def _positive_cn_ring_pattern_candidates(
    mol,
    ordered: Sequence[int],
    charged_idx: int,
    patterns: Sequence[tuple[str, Sequence[str]]],
) -> list[tuple[str, tuple[int, ...]]]:
    candidates: list[tuple[str, tuple[int, ...]]] = []
    for seq in _ring_numberings(ordered):
        if seq[0] != charged_idx:
            continue
        seq_symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in seq]
        for parent, pattern in patterns:
            if seq_symbols == list(pattern):
                candidates.append((parent, seq))
    return candidates


def _kekulized_aromatic_ring(mol, ordered: Sequence[int], symbols: Sequence[str]) -> bool:
    if not mol.HasProp("_nispo_kekulized_aromatic_view"):
        return False
    if any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ordered):
        return False
    if not all(sym in SKELETON_ATOMS for sym in symbols):
        return False
    bonds = [mol.GetBondBetweenAtoms(a, b) for a, b in zip(ordered, ordered[1:] + ordered[:1])]
    if any(bond is None for bond in bonds):
        return False
    orders = [bond.GetBondTypeAsDouble() for bond in bonds]
    if any(order not in {1.0, 2.0} for order in orders):
        return False
    double_positions = [idx for idx, order in enumerate(orders) if order == 2.0]
    if len(ordered) == 6:
        return len(double_positions) == 3 and not _has_adjacent_cyclic_positions(double_positions, len(ordered))
    if len(ordered) == 5 and any(sym != "C" for sym in symbols):
        return len(double_positions) == 2 and not _has_adjacent_cyclic_positions(double_positions, len(ordered))
    return False


def _has_adjacent_cyclic_positions(positions: Sequence[int], size: int) -> bool:
    values = set(positions)
    return any(((pos + 1) % size) in values for pos in values)


def _six_membered_aromatic_candidates(
    mol, ordered: Sequence[int], symbols: Sequence[str]
) -> list[tuple[str, tuple[int, ...]]]:
    counts = Counter(symbols)
    if counts == Counter({"C": 6}):
        return [("benzene", seq) for seq in _ring_numberings(ordered)]
    if counts == Counter({"C": 5, "N": 1}):
        indicated = _single_indicated_hydrogen_n(mol, ordered)
        return [
            (_indicated_hydrogen_prefix(mol, ordered, seq, indicated) + "pyridine", seq)
            for seq in _numberings_with_atom_at(ordered, mol, "N")
        ]
    if counts == Counter({"C": 4, "N": 2}):
        return _aromatic_nitrogen_ring_candidates(
            mol,
            ordered,
            _AROMATIC_6_DIAZINE_NAMES,
            indicated_hydrogen=True,
            use_second_n_locant=True,
        )
    if counts == Counter({"C": 3, "N": 3}):
        return _aromatic_nitrogen_ring_candidates(mol, ordered, _AROMATIC_6_TRIAZINE_NAMES)
    if counts == Counter({"C": 2, "N": 4}):
        return _aromatic_polynitrogen_ring_candidates(mol, ordered, "tetrazine")
    if counts == Counter({"C": 1, "N": 5}):
        return _aromatic_polynitrogen_ring_candidates(mol, ordered, "pentazine")
    if counts == Counter({"N": 6}):
        return [("hexazine", seq) for seq in _ring_numberings(ordered)]
    if counts == Counter({"C": 5, "O": 1}):
        parent = "pyran" if _ring_has_exocyclic_oxo(mol, ordered) else "2H-pyran"
        return [(parent, seq) for seq in _pyran_numberings(ordered, mol)]
    return []


def _five_membered_aromatic_candidates(
    mol, ordered: Sequence[int], counts: Counter[str]
) -> list[tuple[str, tuple[int, ...]]]:
    candidates: list[tuple[str, tuple[int, ...]]] = []
    oxoheterole = _five_membered_oxoheterole_candidates(mol, ordered, counts)
    if oxoheterole:
        return oxoheterole

    for hetero, parent in _AROMATIC_5_SINGLE_HETERO_NAMES.items():
        if counts == Counter({"C": 4, hetero: 1}):
            return [(parent, seq) for seq in _numberings_with_atom_at(ordered, mol, hetero)]

    if counts == Counter({"C": 3, "O": 2}) and _ring_has_exocyclic_oxo(mol, ordered):
        locant_names = {
            (1, 2): "1,2-dioxole",
            (1, 3): "1,3-dioxole",
        }
        for seq in _numberings_with_atom_at(ordered, mol, "O"):
            name = locant_names.get(tuple(_symbol_locants(mol, seq, "O")))
            if name:
                candidates.append((name, seq))
        return candidates

    if counts == Counter({"C": 3, "N": 2}):
        return _aromatic_nitrogen_ring_candidates(
            mol,
            ordered,
            _AROMATIC_5_DIAZOLE_NAMES,
            indicated_hydrogen=True,
            use_second_n_locant=True,
        )
    if counts == Counter({"C": 2, "N": 3}):
        return _aromatic_nitrogen_ring_candidates(
            mol,
            ordered,
            _AROMATIC_5_TRIAZOLE_NAMES,
            indicated_hydrogen=True,
        )
    if counts == Counter({"C": 1, "N": 4}):
        indicated = _single_indicated_hydrogen_n(mol, ordered)
        for seq in _numberings_with_atom_at(ordered, mol, "N"):
            if mol.GetAtomWithIdx(seq[4]).GetSymbol() != "C":
                continue
            hprefix = _indicated_hydrogen_prefix(mol, ordered, seq, indicated)
            candidates.append((f"{hprefix}tetrazole", seq))
        return candidates
    if counts == Counter({"N": 5}):
        return [("pentazole", seq) for seq in _ring_numberings(ordered)]

    candidates = _five_membered_main_group_heterole_candidates(mol, ordered, counts)
    if candidates:
        return candidates

    if counts == Counter({"C": 1, "N": 3, "O": 1}):
        candidates.extend(
            _ring_pattern_candidates(
                mol,
                ordered,
                "1,2,3-triazoxole",
                ("N", "N", "N", "O", "C"),
            )
        )
        if candidates:
            return candidates

    for hetero, names in _AROMATIC_5_AZOLE_NAMES.items():
        if counts != Counter({"C": 3, "N": 1, hetero: 1}):
            continue
        for seq in _numberings_with_atom_at(ordered, mol, hetero):
            n_locs = _symbol_locants(mol, seq, "N")
            if n_locs and n_locs[0] in names:
                candidates.append((names[n_locs[0]], seq))
        return candidates

    if counts == Counter({"C": 3, "B": 1, "N": 1}):
        for seq in _numberings_with_atom_at(ordered, mol, "N"):
            locs = tuple(
                i + 1
                for i, idx in enumerate(seq)
                if mol.GetAtomWithIdx(idx).GetSymbol() in {"B", "N"}
            )
            candidates.append((f"{_join_locants(locs)}-azaborole", seq))
        return candidates

    if counts == Counter({"C": 3, "S": 2}):
        candidates.extend(
            _ring_pattern_candidates(
                mol,
                ordered,
                "3H-1,2-dithiole",
                ("S", "S", "C", "C", "C"),
                checks=(lambda mol, seq, ring_set: _has_exocyclic_thioxo(mol, seq[2], ring_set),),
            )
        )
        if candidates:
            return candidates

    for hetero, family in _AROMATIC_5_DIAZOLE_HETERO_FAMILIES.items():
        if counts != Counter({"C": 2, "N": 2, hetero: 1}):
            continue
        locant_names = {
            locants: f"{prefix}-{family}"
            for locants, prefix in _AROMATIC_5_DIAZOLE_HETERO_LOCANTS.items()
        }
        for seq in _numberings_with_atom_at(ordered, mol, hetero):
            locs = tuple(
                i + 1
                for i, idx in enumerate(seq)
                if mol.GetAtomWithIdx(idx).GetSymbol() in {hetero, "N"}
            )
            name = locant_names.get(locs)
            if name:
                candidates.append((name, seq))
        return candidates
    return candidates


def _five_membered_main_group_heterole_candidates(
    mol, ordered: Sequence[int], counts: Counter[str]
) -> list[tuple[str, tuple[int, ...]]]:
    pattern_specs = (
        (Counter({"C": 3, "N": 1, "Si": 1}), "azasilole", ("N", "Si", "C", "C", "C")),
        (Counter({"C": 2, "N": 2, "Si": 1}), "diazasilole", ("N", "N", "Si", "C", "C")),
        (Counter({"C": 1, "N": 3, "Si": 1}), "triazasilole", ("N", "N", "N", "Si", "C")),
        (Counter({"B": 1, "C": 2, "N": 2}), "diazaborole", ("N", "N", "B", "C", "C")),
        (Counter({"B": 1, "C": 3, "S": 1}), "thiaborole", ("S", "B", "C", "C", "C")),
        (Counter({"B": 1, "C": 3, "O": 1}), "oxaborole", ("O", "B", "C", "C", "C")),
        (Counter({"B": 1, "C": 2, "O": 2}), "dioxaborole", ("O", "O", "B", "C", "C")),
    )
    for target_counts, name, pattern in pattern_specs:
        if counts != target_counts:
            continue
        return _ring_pattern_candidates(mol, ordered, name, pattern)
    return []


def _four_membered_aromatic_heterocycle_candidates(
    mol, ordered: Sequence[int], counts: Counter[str]
) -> list[tuple[str, tuple[int, ...]]]:
    if counts == Counter({"C": 2, "S": 2}):
        return [
            ("1,2-dithiete", seq)
            for seq in _numberings_with_atom_at(ordered, mol, "S")
            if _symbol_locants(mol, seq, "S") == [1, 2]
        ]
    if counts == Counter({"C": 2, "O": 2}):
        return [
            ("1,2-dioxete", seq)
            for seq in _numberings_with_atom_at(ordered, mol, "O")
            if _symbol_locants(mol, seq, "O") == [1, 2]
        ]
    if counts == Counter({"C": 2, "O": 1, "S": 1}):
        return [
            ("1,2-oxathiete", seq)
            for seq in _numberings_with_atom_at(ordered, mol, "O")
            if _symbol_locants(mol, seq, "S") == [2]
        ]
    if counts != Counter({"C": 2, "N": 1, "O": 1}):
        return []
    names = {2: "1,2-oxazete", 3: "1,3-oxazete"}
    candidates = []
    for seq in _numberings_with_atom_at(ordered, mol, "O"):
        n_locs = _symbol_locants(mol, seq, "N")
        name = names.get(n_locs[0]) if n_locs else None
        if name:
            candidates.append((name, seq))
    return candidates


def _cyclobutene_chalcogenone_candidates(mol, ordered: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
    ring_set = set(ordered)
    chalcogenone_atoms = [idx for idx in ordered if _exocyclic_chalcogenone_symbol(mol, idx, ring_set)]
    if len(chalcogenone_atoms) != 2:
        return []
    if mol.GetBondBetweenAtoms(chalcogenone_atoms[0], chalcogenone_atoms[1]) is None:
        return []
    alkene_atoms = [idx for idx in ordered if idx not in chalcogenone_atoms]
    if len(alkene_atoms) != 2:
        return []
    alkene_bond = mol.GetBondBetweenAtoms(alkene_atoms[0], alkene_atoms[1])
    if alkene_bond is None or alkene_bond.GetBondTypeAsDouble() not in {1.5, 2.0}:
        return []
    candidates = []
    for left, right in (tuple(chalcogenone_atoms), tuple(reversed(chalcogenone_atoms))):
        for first, second in (tuple(alkene_atoms), tuple(reversed(alkene_atoms))):
            seq = (left, right, first, second)
            if all(
                mol.GetBondBetweenAtoms(a, b) is not None
                for a, b in zip(seq, seq[1:] + seq[:1])
            ):
                candidates.append(("cyclobut-3-ene", seq))
    return candidates


def _exocyclic_chalcogenone_symbol(mol, atom_idx: int, ring_set: set[int]) -> str:
    found = []
    for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx in ring_set or nbr.GetSymbol() not in {"O", "S"}:
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
        if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
            found.append(nbr.GetSymbol())
    return found[0] if len(found) == 1 else ""


def _medium_aromatic_heterocycle_candidates(
    mol, ordered: Sequence[int], counts: Counter[str]
) -> list[tuple[str, tuple[int, ...]]]:
    nitrogen_count = counts.get("N", 0)
    if counts == Counter({"C": len(ordered) - nitrogen_count, "N": nitrogen_count}):
        parent = _MEDIUM_AROMATIC_N_RING_PARENT.get((len(ordered), nitrogen_count))
        if parent:
            if nitrogen_count == 1:
                return [(parent, seq) for seq in _numberings_with_atom_at(ordered, mol, "N")]
            return _aromatic_polynitrogen_ring_candidates(mol, ordered, parent)
    if len(ordered) == 8 and counts == Counter({"C": 6, "O": 2}):
        return [
            ("1,2-dioxocine", seq)
            for seq in _numberings_with_atom_at(ordered, mol, "O")
            if _symbol_locants(mol, seq, "O") == [1, 2]
        ]
    if len(ordered) == 8 and counts == Counter({"C": 6, "N": 1, "O": 1}):
        return _medium_oxaaza_candidates(mol, ordered, "1,2-oxazocine")
    if len(ordered) == 8 and counts == Counter({"C": 6, "N": 1, "S": 1}):
        return _ring_pattern_candidates(mol, ordered, "thiazocine", ("S", "N", "C", "C", "C", "C", "C", "C"))
    if len(ordered) == 9 and counts == Counter({"C": 8, "P": 1}):
        return [("phosphonine", seq) for seq in _numberings_with_atom_at(ordered, mol, "P")]
    if len(ordered) == 9 and counts == Counter({"C": 8, "O": 1}):
        return [("oxonine", seq) for seq in _numberings_with_atom_at(ordered, mol, "O")]
    if len(ordered) == 9 and counts == Counter({"C": 7, "N": 1, "O": 1}):
        return _medium_oxaaza_candidates(mol, ordered, "1,2-oxazonine")
    return []


def _medium_oxaaza_candidates(mol, ordered: Sequence[int], parent: str) -> list[tuple[str, tuple[int, ...]]]:
    return [
        (parent, seq)
        for seq in _numberings_with_atom_at(ordered, mol, "O")
        if _symbol_locants(mol, seq, "N") == [2]
    ]


def _unsaturated_heterocycle_suffix_parent_candidates(
    mol,
    ordered: Sequence[int],
    counts: Counter[str],
) -> list[tuple[str, tuple[int, ...]]]:
    if len(ordered) == 7:
        if counts == Counter({"C": 6, "N": 1}):
            return [("azepine", seq) for seq in _numberings_with_atom_at(ordered, mol, "N")]
        if counts == Counter({"C": 5, "N": 2}):
            return _locanted_unsaturated_heteroring_candidates(mol, ordered, "N", {"N"}, "diazepine")
        return []

    families = {
        (("C", 4), ("N", 1), ("O", 1)): ("O", {"N", "O"}, "oxazine"),
        (("C", 4), ("N", 1), ("S", 1)): ("S", {"N", "S"}, "thiazine"),
        (("C", 3), ("N", 2), ("O", 1)): ("O", {"N", "O"}, "oxadiazine"),
        (("C", 3), ("N", 2), ("S", 1)): ("S", {"N", "S"}, "thiadiazine"),
    }
    spec = families.get(tuple(sorted(counts.items())))
    if spec is None:
        return []
    anchor, locant_symbols, family = spec
    return _locanted_unsaturated_heteroring_candidates(mol, ordered, anchor, locant_symbols, family)


def _locanted_unsaturated_heteroring_candidates(
    mol,
    ordered: Sequence[int],
    anchor_symbol: str,
    locant_symbols: set[str],
    family: str,
) -> list[tuple[str, tuple[int, ...]]]:
    candidates: list[tuple[str, tuple[int, ...]]] = []
    for seq in _numberings_with_atom_at(ordered, mol, anchor_symbol):
        locants = tuple(
            i + 1
            for i, idx in enumerate(seq)
            if mol.GetAtomWithIdx(idx).GetSymbol() in locant_symbols
        )
        if not locants:
            continue
        candidates.append((f"{_join_locants(locants)}-{family}", seq))
    return candidates


def _five_membered_oxoheterole_candidates(
    mol, ordered: Sequence[int], counts: Counter[str]
) -> list[tuple[str, tuple[int, ...]]]:
    if counts == Counter({"B": 2, "C": 2, "O": 1}):
        return [
            (f"{_join_locants(_symbol_locants(mol, seq, 'B'))}-diboroxole", seq)
            for seq in _numberings_with_atom_at(ordered, mol, "O")
        ]
    families = {
        "B": "oxaborole",
        "P": "oxaphosphole",
        "Si": "oxasilole",
    }
    candidates = []
    for symbol, family in families.items():
        if counts != Counter({"C": 3, "O": 1, symbol: 1}):
            continue
        for seq in _numberings_with_atom_at(ordered, mol, "O"):
            locs = _symbol_locants(mol, seq, symbol)
            if not locs:
                continue
            if locs[0] in {2, 3}:
                candidates.append((f"1,{locs[0]}-{family}", seq))
        return candidates
    return candidates


def _aromatic_nitrogen_ring_candidates(
    mol,
    ordered: Sequence[int],
    name_by_locant,
    *,
    indicated_hydrogen: bool = False,
    use_second_n_locant: bool = False,
) -> list[tuple[str, tuple[int, ...]]]:
    candidates: list[tuple[str, tuple[int, ...]]] = []
    indicated = _single_indicated_hydrogen_n(mol, ordered) if indicated_hydrogen else None
    for seq in _numberings_with_atom_at(ordered, mol, "N"):
        n_locs = tuple(_symbol_locants(mol, seq, "N"))
        key = sorted(n_locs)[1] if use_second_n_locant and len(n_locs) > 1 else n_locs
        name = name_by_locant.get(key)
        if not name:
            continue
        hprefix = _indicated_hydrogen_prefix(mol, ordered, seq, indicated) if indicated_hydrogen else ""
        candidates.append((hprefix + name, seq))
    return candidates


def _aromatic_polynitrogen_ring_candidates(
    mol, ordered: Sequence[int], parent: str
) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    for seq in _numberings_with_atom_at(ordered, mol, "N"):
        n_locs = tuple(_symbol_locants(mol, seq, "N"))
        candidates.append((f"{_join_locants(n_locs)}-{parent}", seq))
    return candidates


def _symbol_locants(mol, seq: Sequence[int], symbol: str) -> list[int]:
    return [i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == symbol]


def _unsaturated_pyran_candidates(mol, ordered: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
    candidates: list[tuple[str, tuple[int, ...]]] = []
    if _ring_has_exocyclic_oxo(mol, ordered):
        return candidates
    for seq in _numberings_with_atom_at(ordered, mol, "O"):
        unsat_locs = set()
        valid = True
        for loc, begin in enumerate(seq, start=1):
            end = seq[loc % len(seq)]
            bond = mol.GetBondBetweenAtoms(begin, end)
            if bond is None:
                valid = False
                break
            order = bond.GetBondTypeAsDouble()
            if order == 2.0:
                unsat_locs.add(loc)
            elif order != 1.0:
                valid = False
                break
        if not valid:
            continue
        if unsat_locs == {3, 5}:
            candidates.append(("2H-pyran", seq))
        elif unsat_locs == {2, 5}:
            candidates.append(("4H-pyran", seq))
    return candidates


def _partially_hydrogenated_pyridine_candidates(mol, ordered: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
    candidates: list[tuple[str, tuple[int, ...]]] = []
    ring_set = set(ordered)
    for seq in _numberings_with_atom_at(ordered, mol, "N"):
        saturated_locs = _h_pyridine_saturated_locs(mol, seq)
        oxo_methylidene_locs = [
            loc
            for loc, idx in enumerate(seq, start=1)
            if _has_exocyclic_oxo_methylidene(mol, idx, ring_set)
        ]
        oxo_h_locs = [loc for loc in oxo_methylidene_locs if loc in saturated_locs]
        if len(oxo_h_locs) == 1:
            candidates.append((f"{oxo_h_locs[0]}H-pyridine", seq))
            continue
        if len(saturated_locs) == 1:
            candidates.append((f"{saturated_locs[0]}H-pyridine", seq))
            continue
        if len(saturated_locs) != 2:
            continue
        if (
            1 in saturated_locs
            and _has_external_n_substituent(mol, seq[0], ring_set)
            and not _has_external_ring_carbon_substituent(mol, seq, ring_set)
        ):
            candidates.append(("pyridine", seq))
        else:
            locants = ",".join(str(loc) for loc in saturated_locs)
            candidates.append((f"{locants}-dihydropyridine", seq))
    return candidates


def _has_exocyclic_oxo_methylidene(mol, atom_idx: int, ring_set: set[int]) -> bool:
    for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        carbon_idx = nbr.GetIdx()
        if carbon_idx in ring_set or nbr.GetSymbol() != "C" or nbr.GetFormalCharge() != 0:
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, carbon_idx)
        if bond is None or bond.GetBondTypeAsDouble() != 2.0:
            continue
        oxygen_count = 0
        extra = False
        for distal in nbr.GetNeighbors():
            distal_idx = distal.GetIdx()
            if distal_idx == atom_idx:
                continue
            distal_bond = mol.GetBondBetweenAtoms(carbon_idx, distal_idx)
            if distal.GetSymbol() == "O" and distal_bond is not None and distal_bond.GetBondTypeAsDouble() == 2.0:
                oxygen_count += 1
            elif distal.GetAtomicNum() > 1:
                extra = True
                break
        if oxygen_count == 1 and not extra:
            return True
    return False


def _h_pyridine_saturated_locs(mol, seq: Sequence[int]) -> list[int]:
    orders = [
        mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
        for a, b in zip(seq, seq[1:] + seq[:1])
    ]
    if any(order not in {1.0, 2.0} for order in orders):
        return []
    if orders.count(2.0) != 2:
        return []
    locs = []
    for loc in range(1, len(seq) + 1):
        left = (loc - 2) % len(seq)
        right = (loc - 1) % len(seq)
        if orders[left] == 1.0 and orders[right] == 1.0:
            locs.append(loc)
    return locs


def _has_external_n_substituent(mol, nitrogen_idx: int, ring_set: set[int]) -> bool:
    return any(
        nbr.GetIdx() not in ring_set and nbr.GetAtomicNum() > 1
        for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors()
    )


def _has_external_ring_carbon_substituent(mol, seq: Sequence[int], ring_set: set[int]) -> bool:
    for idx in seq[1:]:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "C":
            continue
        if any(nbr.GetIdx() not in ring_set and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
            return True
    return False


def _fixed_saturated_heterocycle_candidates(
    mol, ordered: Sequence[int], symbols: Sequence[str]
) -> list[tuple[str, tuple[int, ...]]]:
    candidates: list[tuple[str, tuple[int, ...]]] = []
    counts = Counter(symbols)
    size = len(ordered)
    for spec in _SATURATED_HETEROCYCLE_SPECS:
        if size != spec.size or counts != Counter(dict(spec.counts)):
            continue
        for seq in _numberings_with_atom_at(ordered, mol, spec.anchor_symbol):
            if spec.locant_symbol:
                locants = tuple(
                    i + 1
                    for i, idx in enumerate(seq)
                    if mol.GetAtomWithIdx(idx).GetSymbol() == spec.locant_symbol
                )
                if locants != spec.locants:
                    continue
            candidates.append((spec.parent, seq))
    return candidates


def _cyclic_imide_candidates(mol, ordered: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
    candidates: list[tuple[str, tuple[int, ...]]] = []
    ring_set = set(ordered)
    for seq in _numberings_with_atom_at(ordered, mol, "N"):
        if mol.GetAtomWithIdx(seq[0]).GetFormalCharge() != 0:
            continue
        if not _cyclic_imide_n_substitution_supported(mol, seq[0], ring_set):
            continue
        if not _has_exocyclic_oxo(mol, seq[1], ring_set) or not _has_exocyclic_oxo(mol, seq[4], ring_set):
            continue
        if _cyclic_imide_has_alkene_disubstitution(mol, seq, ring_set):
            continue
        if any(_has_exocyclic_oxo(mol, idx, ring_set) for idx in seq[2:4]):
            continue
        orders = [
            mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
            for a, b in zip(seq, seq[1:] + seq[:1])
        ]
        if any(order not in {1.0, 2.0} for order in orders):
            continue
        if orders == [1.0, 1.0, 2.0, 1.0, 1.0]:
            candidates.append(("maleimide", seq))
        elif orders == [1.0, 1.0, 1.0, 1.0, 1.0]:
            candidates.append(("succinimide", seq))
    return candidates


def _cyclic_imide_n_substitution_supported(mol, nitrogen_idx: int, ring_set: set[int]) -> bool:
    external = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors()
        if nbr.GetIdx() not in ring_set and nbr.GetAtomicNum() > 1
    ]
    if not external:
        return True
    if len(external) != 1:
        return False
    root = external[0]
    if mol.GetAtomWithIdx(root).GetSymbol() != "C":
        return False
    seen = {nitrogen_idx}
    stack = [root]
    atoms = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        atoms.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        if atom.IsInRing() or atom.GetSymbol() != "C":
            return False
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in seen or nidx in ring_set or nbr.GetAtomicNum() <= 1:
                continue
            stack.append(nidx)
    return len(atoms) <= 10


def _cyclic_imide_has_alkene_disubstitution(mol, seq: Sequence[int], ring_set: set[int]) -> bool:
    return all(
        any(nbr.GetIdx() not in ring_set and nbr.GetAtomicNum() > 1 for nbr in mol.GetAtomWithIdx(idx).GetNeighbors())
        for idx in seq[2:4]
    )


def _carbocycle_parent(mol, seq: Sequence[int]) -> str:
    root = ROOTS.get(len(seq))
    if not root:
        return ""
    ene_locs = []
    yne_locs = []
    for loc, (a, b) in enumerate(zip(seq, seq[1:] + seq[:1]), start=1):
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            return ""
        if bond.GetBondTypeAsDouble() == 1.0:
            continue
        if bond.GetBondTypeAsDouble() == 2.0:
            ene_locs.append(loc)
            continue
        if bond.GetBondTypeAsDouble() == 3.0:
            yne_locs.append(loc)
            continue
        return ""
    root = "cyclo" + root
    if not ene_locs and not yne_locs:
        if any(str(mol.GetAtomWithIdx(idx).GetChiralTag()) != "CHI_UNSPECIFIED" for idx in seq):
            return root + "ane"
        return root + "an"
    if ene_locs and not yne_locs:
        return loc_unsat(root, ene_locs, "ene")
    if yne_locs and not ene_locs:
        return loc_unsat(root, yne_locs, "yne")
    ene = loc_unsat(root, ene_locs, "en", terminal_e=False)
    yne = loc_unsat("", yne_locs, "yne", terminal_e=True)
    return ene + "-" + yne


def _aromatic_carbocycle_parent(seq: Sequence[int]) -> str:
    root = ROOTS.get(len(seq))
    if not root:
        return ""
    ene_count = len(seq) // 2
    if ene_count <= 0:
        return ""
    locants = tuple(range(1, ene_count * 2, 2))
    return loc_unsat("cyclo" + root, locants, "ene")


def _aromatic_carbocycle_has_supported_valence(mol, ordered: Sequence[int]) -> bool:
    ring_set = set(ordered)
    for idx in ordered:
        exocyclic = [nbr for nbr in mol.GetAtomWithIdx(idx).GetNeighbors() if nbr.GetIdx() not in ring_set]
        if len(exocyclic) > 1:
            return False
    return True


def _aromatic_replacement_ring_supported(mol, ordered: Sequence[int]) -> bool:
    ring_set = set(ordered)
    for idx in ordered:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nbr.GetIdx() in ring_set:
                continue
            bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
            if bond is not None and bond.GetBondTypeAsDouble() > 1.0:
                return False
    return True


def _unsaturated_diazine_candidates(mol, ordered: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    for seq in _numberings_with_atom_at(ordered, mol, "N"):
        n_locs = [i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "N"]
        second = sorted(n_locs)[1]
        parent = {2: "pyridazine", 3: "pyrimidine", 4: "pyrazine"}.get(second)
        if not parent:
            continue
        double_bonds = 0
        saturated_locs = []
        for loc, idx in enumerate(seq, start=1):
            prev_idx = seq[loc - 2]
            next_idx = seq[loc % len(seq)]
            prev_bond = mol.GetBondBetweenAtoms(idx, prev_idx)
            next_bond = mol.GetBondBetweenAtoms(idx, next_idx)
            has_double = (
                prev_bond is not None
                and prev_bond.GetBondTypeAsDouble() == 2.0
                or next_bond is not None
                and next_bond.GetBondTypeAsDouble() == 2.0
            )
            if not has_double:
                saturated_locs.append(loc)
            bond = mol.GetBondBetweenAtoms(idx, next_idx)
            if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
                double_bonds += 1
        if double_bonds == 2 and len(saturated_locs) == 2:
            locants = ",".join(str(loc) for loc in sorted(saturated_locs))
            candidates.append((f"{locants}-dihydro{parent}", seq))
    return candidates



def _heterocycloalkane_parent(mol, seq: Sequence[int]) -> str:
    parent = _heterocycle_parent(mol, seq)
    return parent if parent.endswith(("ane", "an")) else ""


def _aromatic_heterocycle_parent(mol, seq: Sequence[int]) -> str:
    parent = _aromatic_carbocycle_parent(seq)
    if not parent:
        return ""
    replacement_prefix = _replacement_prefix(mol, seq)
    return f"{replacement_prefix}{parent}" if replacement_prefix else ""


def _heterocycle_parent(mol, seq: Sequence[int]) -> str:
    root = ROOTS.get(len(seq))
    if not root:
        return ""
    replacement_prefix = _replacement_prefix(mol, seq)
    if not replacement_prefix:
        return ""
    parent = _carbocycle_parent(mol, seq)
    return f"{replacement_prefix}{parent}" if parent else ""


def _replacement_prefix(mol, seq: Sequence[int]) -> str:
    grouped: dict[str, list[int]] = defaultdict(list)
    for loc, idx in enumerate(seq, start=1):
        symbol = mol.GetAtomWithIdx(idx).GetSymbol()
        if symbol == "C":
            continue
        term = REPLACEMENT.get(symbol)
        if not term:
            return ""
        grouped[term].append(loc)
    if not grouped:
        return ""
    parts = []
    for term in sorted(grouped):
        locants = grouped[term]
        if len(locants) == 1:
            parts.append(f"{locants[0]}-{term}")
        else:
            parts.append(f"{_join_locants(locants)}-{MULT.get(len(locants), '')}{term}")
    return "-".join(parts)



def _numberings_with_atom_at(ordered: Sequence[int], mol, symbol: str) -> list[tuple[int, ...]]:
    seqs = []
    for seq in _ring_numberings(ordered):
        if mol.GetAtomWithIdx(seq[0]).GetSymbol() == symbol:
            seqs.append(seq)
    return seqs


def _pyran_numberings(ordered: Sequence[int], mol) -> list[tuple[int, ...]]:
    seqs = _numberings_with_atom_at(ordered, mol, "O")
    oxo_at_2 = []
    for seq in seqs:
        second = seq[1]
        if _has_exocyclic_oxo(mol, second, set(ordered)):
            oxo_at_2.append(seq)
    return oxo_at_2 or seqs


def _ring_has_exocyclic_oxo(mol, ordered: Sequence[int]) -> bool:
    ring_set = set(ordered)
    return any(_has_exocyclic_oxo(mol, idx, ring_set) for idx in ordered)


def _ring_has_exocyclic_suffix_site(mol, ordered: Sequence[int]) -> bool:
    ring_set = set(ordered)
    return any(
        _has_exocyclic_oxo(mol, idx, ring_set)
        or _has_exocyclic_thioxo(mol, idx, ring_set)
        or _has_exocyclic_imino(mol, idx, ring_set)
        for idx in ordered
    )


def _has_exocyclic_oxo(mol, idx: int, ring_set: set[int]) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return False
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx in ring_set or nbr.GetSymbol() != "O":
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
            return True
    return False


def _has_exocyclic_thioxo(mol, idx: int, ring_set: set[int]) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return False
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx in ring_set or nbr.GetSymbol() != "S":
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
            return True
    return False


def _has_exocyclic_imino(mol, idx: int, ring_set: set[int]) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return False
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx in ring_set or nbr.GetSymbol() != "N":
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
            return True
    return False



def adjust_ring_parent_for_charge(mol, parent_name: str, seq: Sequence[int], raw_subs) -> str:
    lambda_name = _lambda2_ring_parent_name(mol, parent_name, seq)
    if lambda_name:
        parent_name = lambda_name
    multi_positive_name = _multi_positive_n_ring_parent_name(mol, parent_name, seq)
    if multi_positive_name:
        return multi_positive_name
    for loc, idx in enumerate(seq, start=1):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() < 0:
            stem = parent_name[:-1] if parent_name.endswith("e") else parent_name
            return f"{stem}-{loc}-ide"
        if atom.GetFormalCharge() <= 0:
            continue
        if atom.GetSymbol() in {"O", "S"}:
            stem = parent_name[:-1] if parent_name.endswith("e") else parent_name
            return f"{stem}-{loc}-ium"
        if atom.GetSymbol() != "C" and _is_replacement_heterocycle_parent(parent_name, atom.GetSymbol()):
            stem = parent_name[:-1] if parent_name.endswith("e") else parent_name
            return f"{stem}-{loc}-ium"
        if atom.GetSymbol() != "N":
            continue
        aromatic_cation = _aromatic_n_h_cation_parent_name(parent_name, atom)
        if aromatic_cation:
            return aromatic_cation
        charge_base = _drop_indicated_hydrogen_prefix(parent_name)
        if charge_base == "pyridine" and loc == 1:
            return "pyridin-1-ium"
        if charge_base in {"pyrimidine", "pyrazine", "pyridazine"}:
            stem = charge_base[:-1]
            return f"{stem}-{loc}-ium"
        if charge_base in {
            "aziridine",
            "azetidine",
            "pyrrolidine",
            "piperidine",
            "azepane",
            "1,2-diazetidine",
            "1,3-diazetidine",
            "imidazolidine",
            "imidazoline",
            "piperazine",
            "morpholine",
            "pyrazoline",
            "thiomorpholine",
        }:
            stem = charge_base[:-1] if charge_base.endswith("e") else charge_base
            return f"{stem}-{loc}-ium"
        if re.fullmatch(r"1,2,[34]-triazoline", charge_base):
            stem = charge_base[:-1] if charge_base.endswith("e") else charge_base
            return f"{stem}-{loc}-ium"
        if re.fullmatch(r"\d+(?:,\d+)*-dihydropyrimidine", charge_base):
            stem = charge_base[:-1] if charge_base.endswith("e") else charge_base
            return f"{stem}-{loc}-ium"
        if charge_base.endswith(("triazine", "tetrazine", "pentazine", "hexazine")):
            stem = charge_base[:-1] if charge_base.endswith("e") else charge_base
            return f"{stem}-{loc}-ium"
        if parent_name == "1H-imidazole":
            base = parent_name if atom.GetTotalNumHs() > 0 else _drop_indicated_hydrogen_prefix(parent_name)
            stem = base[:-1] if base.endswith("e") else base
            return f"{stem}-{loc}-ium"
        if parent_name == "1H-imidazol-1-ium" and loc == 1:
            return parent_name
        if parent_name.endswith(
            (
                "triazole",
                "tetrazole",
                "pyrazole",
                "oxazole",
                "isoxazole",
                "thiazole",
                "isothiazole",
                "oxadiazole",
                "thiadiazole",
            )
        ):
            base = parent_name if atom.GetTotalNumHs() > 0 else _drop_indicated_hydrogen_prefix(parent_name)
            stem = base[:-1] if base.endswith("e") else base
            return f"{stem}-{loc}-ium"
        stem = _CHARGEABLE_SUBSTITUENT_STEMS.get(parent_name)
        if stem:
            return f"{stem}-{loc}-ium"
        return ""
    return parent_name


def _multi_positive_n_ring_parent_name(mol, parent_name: str, seq: Sequence[int]) -> str:
    if "lambda" in parent_name:
        return ""
    positive_locs = []
    for loc, idx in enumerate(seq, start=1):
        atom = mol.GetAtomWithIdx(idx)
        charge = atom.GetFormalCharge()
        if charge < 0:
            return ""
        if charge <= 0:
            continue
        if atom.GetSymbol() != "N":
            return ""
        positive_locs.append(loc)
    if len(positive_locs) < 2:
        return ""
    charge_base = _drop_indicated_hydrogen_prefix(parent_name)
    if not _multi_positive_n_parent_supported(charge_base):
        return ""
    multiplier = MULT.get(len(positive_locs))
    if not multiplier:
        return ""
    stem = charge_base[:-1] if charge_base.endswith("e") else charge_base
    return f"{stem}-{_join_locants(positive_locs)}-{multiplier}ium"


def _multi_positive_n_parent_supported(charge_base: str) -> bool:
    if charge_base in {
        "pyrimidine",
        "pyrazine",
        "pyridazine",
        "1,2-diazetidine",
        "1,3-diazetidine",
        "imidazolidine",
        "imidazoline",
        "piperazine",
        "pyrazoline",
        "dihydropyrimidine",
    }:
        return True
    if re.fullmatch(r"1,2,[34]-triazoline", charge_base):
        return True
    if re.fullmatch(r"\d+(?:,\d+)*-dihydropyrimidine", charge_base):
        return True
    if charge_base.endswith(("triazine", "tetrazine", "pentazine", "hexazine")):
        return True
    if charge_base.endswith(
        (
            "imidazole",
            "triazole",
            "tetrazole",
            "pyrazole",
            "oxadiazole",
            "thiadiazole",
        )
    ):
        return True
    return False


def _aromatic_n_h_cation_parent_name(parent_name: str, atom) -> str:
    if not atom.GetIsAromatic() or atom.GetTotalNumHs() <= 0:
        return ""
    base = _drop_indicated_hydrogen_prefix(parent_name)
    if "lambda" in base:
        return ""
    if not any(base == parent or base.endswith(f"-{parent}") for parent in _AROMATIC_N_H_CATION_PARENTS):
        return ""
    stem = base[:-1] if base.endswith("e") else base
    return f"{stem}ium"


def _lambda2_ring_parent_name(mol, parent_name: str, seq: Sequence[int]) -> str:
    if "lambda" in parent_name:
        return ""
    locants = []
    for loc, idx in enumerate(seq, start=1):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
            continue
        if atom.GetTotalNumHs() > 0:
            continue
        if atom.GetNumRadicalElectrons() or atom.GetExplicitValence() <= 2:
            locants.append(str(loc))
    if not locants:
        return ""
    prefix = ",".join(f"{loc}lambda2" for loc in locants)
    return f"{prefix}-{restore_terminal_unsaturation_e(parent_name)}"


def _is_replacement_heterocycle_parent(parent_name: str, charged_symbol: str) -> bool:
    replacement = REPLACEMENT.get(charged_symbol)
    if not replacement:
        return False
    return bool(re.search(rf"(?:^|-)\d+-{re.escape(replacement)}", parent_name))


def _ring_pattern_candidates(
    mol, ordered: Sequence[int], parent: str, symbols: Sequence[str], *, checks=()
) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    ring_set = set(ordered)
    for seq in _ring_numberings(ordered):
        if [mol.GetAtomWithIdx(idx).GetSymbol() for idx in seq] != list(symbols):
            continue
        if not all(check(mol, seq, ring_set) for check in checks):
            continue
        candidates.append((parent, seq))
    return candidates


def _simple_iodine_ring_candidates(
    mol,
    ordered: Sequence[int],
    symbols: Sequence[str],
) -> list[tuple[str, tuple[int, ...]]]:
    if "I" not in symbols:
        return []
    if _has_external_heavy_substituent(mol, ordered):
        return []
    if any(mol.GetAtomWithIdx(idx).GetFormalCharge() != 0 for idx in ordered):
        return []
    counts = Counter(symbols)
    double_count = sum(
        1
        for left, right in zip(ordered, ordered[1:] + ordered[:1])
        if mol.GetBondBetweenAtoms(left, right).GetBondTypeAsDouble() == 2.0
    )
    parent = ""
    if counts == Counter({"C": 3, "I": 1}) and len(ordered) == 4 and double_count == 1:
        parent = "iodetine"
    elif counts == Counter({"C": 4, "I": 1}) and len(ordered) == 5:
        if double_count == 0:
            parent = "iodolane"
        elif double_count == 1:
            parent = "iodoline"
    elif counts == Counter({"C": 5, "I": 1}) and len(ordered) == 6:
        if double_count == 0:
            parent = "iodinane"
        elif double_count == 2:
            parent = "iodinine"
    elif counts == Counter({"C": 3, "I": 1, "N": 1}) and len(ordered) == 5 and double_count == 2:
        parent = "iodazole"
    elif (
        counts == Counter({"C": 3, "I": 2})
        and len(ordered) == 5
        and double_count == 1
        and _adjacent_symbol_bond_count(mol, ordered, "I", "I") == 1
    ):
        parent = "diiodol"
    elif (
        counts == Counter({"C": 2, "I": 3})
        and len(ordered) == 5
        and double_count == 1
        and _adjacent_symbol_bond_count(mol, ordered, "I", "I") == 2
    ):
        parent = "triiodol"
    if not parent:
        return []
    return [(parent, _ring_sequence_starting_with(mol, ordered, "I"))]


def _has_external_heavy_substituent(mol, ordered: Sequence[int]) -> bool:
    ring_set = set(ordered)
    return any(
        nbr.GetIdx() not in ring_set and nbr.GetAtomicNum() > 1
        for idx in ordered
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
    )


def _adjacent_symbol_bond_count(mol, ordered: Sequence[int], left_symbol: str, right_symbol: str) -> int:
    count = 0
    for left, right in zip(ordered, ordered[1:] + ordered[:1]):
        symbols = {mol.GetAtomWithIdx(left).GetSymbol(), mol.GetAtomWithIdx(right).GetSymbol()}
        if symbols == {left_symbol, right_symbol}:
            count += 1
    return count


def _ring_sequence_starting_with(mol, ordered: Sequence[int], symbol: str) -> tuple[int, ...]:
    for seq in _ring_numberings(ordered):
        if mol.GetAtomWithIdx(seq[0]).GetSymbol() == symbol:
            return tuple(seq)
    return tuple(ordered)


def _ring_numberings(ordered: Sequence[int]) -> list[tuple[int, ...]]:
    seqs = []
    n = len(ordered)
    base = tuple(ordered)
    for direction in (base, tuple(reversed(base))):
        for i in range(n):
            seqs.append(direction[i:] + direction[:i])
    return seqs



def ring_substituent_suffix(parent_name: str, loc: int, charge_loc: int | None = None) -> str:
    if parent_name == "maleimide":
        if loc in {3, 4}:
            return f"1H-pyrrole-2,5-dione-{loc}-yl"
        return ""
    if parent_name == "succinimide":
        return ""
    if parent_name == "benzene":
        return "phenyl" if loc == 1 else ""
    if parent_name == "furan":
        return f"{loc}-furyl"
    if parent_name == "thiophene" and charge_loc is None:
        return f"{loc}-thienyl"
    if parent_name == "pyridine" and charge_loc is None:
        return f"{loc}-pyridyl"
    if parent_name == "piperidine" and charge_loc is None and loc != 1:
        return f"{loc}-piperidyl"
    retained_n_substituent = _retained_saturated_n_substituent_suffix(parent_name, loc, charge_loc)
    if retained_n_substituent:
        return retained_n_substituent
    h_azine = _azine_indicated_hydrogen_substituent_suffix(parent_name, loc, charge_loc)
    if h_azine:
        return h_azine
    charge_base = _drop_indicated_hydrogen_prefix(parent_name) if charge_loc is not None else parent_name
    stem = _CHARGEABLE_SUBSTITUENT_STEMS.get(charge_base)
    if stem:
        return _substituent_suffix_from_stem(stem, loc, charge_loc)
    stem = _NEUTRAL_SUBSTITUENT_STEMS.get(parent_name)
    if stem:
        return _substituent_suffix_from_stem(stem, loc)
    if parent_name.endswith("dihydropyrimidine"):
        return f"{parent_name[:-1]}-{loc}-yl"
    if parent_name.endswith("H-imidazole"):
        if charge_loc is not None:
            return f"{_drop_indicated_hydrogen_prefix(parent_name)[:-1]}-{charge_loc}-ium-{loc}-yl"
        return f"{parent_name[:-1]}-{loc}-yl"
    if parent_name.endswith("H-pyrazole"):
        if charge_loc is not None:
            return f"{_drop_indicated_hydrogen_prefix(parent_name)[:-1]}-{charge_loc}-ium-{loc}-yl"
        return f"{parent_name[:-1]}-{loc}-yl"
    stem = _INDICATED_H_SUBSTITUENT_STEMS.get(parent_name)
    if stem:
        return _substituent_suffix_from_stem(stem, loc)
    if parent_name == "azet-2-one":
        return f"2-oxoazet-{loc}-yl"
    if parent_name == "1,3,4-dioxazol-2-one":
        return f"2-oxo-1,3,4-dioxazol-{loc}-yl"
    if parent_name.endswith(("triazole", "tetrazole")):
        if charge_loc is not None:
            base = _drop_indicated_hydrogen_prefix(parent_name)
            stem = base[:-1] if base.endswith("e") else base
            return f"{stem}-{charge_loc}-ium-{loc}-yl"
        return f"{parent_name[:-1]}-{loc}-yl"
    if parent_name.startswith("cyclo") and parent_name.endswith("ane"):
        return parent_name[:-3] + "yl" if loc == 1 else ""
    if parent_name.startswith("cyclo") and parent_name.endswith("an"):
        return parent_name[:-2] + "yl" if loc == 1 else ""
    stem = _SATURATED_SUBSTITUENT_STEMS.get(parent_name)
    if stem:
        return _substituent_suffix_from_stem(stem, loc, charge_loc)
    if parent_name.isascii() and " " not in parent_name and loc >= 1:
        restored_unsaturation = restore_terminal_unsaturation_e(parent_name)
        if restored_unsaturation != parent_name and _is_replacement_heterocycle_parent_name(parent_name):
            return _substituent_suffix_from_stem(restored_unsaturation, loc, charge_loc)
        stem = parent_name[:-1] if parent_name.endswith("e") else parent_name
        return _substituent_suffix_from_stem(stem, loc, charge_loc)
    return ""


def _is_replacement_heterocycle_parent_name(parent_name: str) -> bool:
    return "cyclo" in parent_name and any(f"-{prefix}" in parent_name for prefix in REPLACEMENT.values())


def _retained_saturated_n_substituent_suffix(parent_name: str, loc: int, charge_loc: int | None) -> str:
    if charge_loc is not None:
        return ""
    return {
        ("piperidine", 1): "piperidino",
        ("piperazine", 1): "piperazino",
        ("piperazine", 4): "piperazino",
        ("pyrrolidine", 1): "pyrrolidino",
        ("morpholine", 4): "morpholino",
    }.get((parent_name, loc), "")


def _substituent_suffix_from_stem(stem: str, loc: int, charge_loc: int | None = None) -> str:
    if charge_loc is not None:
        return f"{stem}-{charge_loc}-ium-{loc}-yl"
    return f"{stem}-{loc}-yl"


def _azine_indicated_hydrogen_substituent_suffix(
    parent_name: str, loc: int, charge_loc: int | None = None
) -> str:
    if charge_loc is not None or not parent_name[:1].isdigit() or "H-" not in parent_name:
        return ""
    h_prefix, base = parent_name.split("-", 1)
    stem = {
        "pyridine": "pyridin",
        "pyrimidine": "pyrimidin",
        "pyrazine": "pyrazin",
        "pyridazine": "pyridazin",
    }.get(base)
    if not stem:
        return ""
    return f"{h_prefix}-{stem}-{loc}-yl"
