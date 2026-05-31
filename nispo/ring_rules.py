from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .constants import MULT, REPLACEMENT, ROOTS, SKELETON_ATOMS
from .formatting import join_locants as _join_locants, loc_unsat
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
_AROMATIC_5_SINGLE_HETERO_NAMES = {"O": "furan", "S": "thiophene", "N": "1H-pyrrole"}
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


def _single_indicated_hydrogen_n(mol, ordered: Sequence[int]) -> int | None:
    atoms = [
        idx
        for idx in ordered
        if mol.GetAtomWithIdx(idx).GetSymbol() == "N" and mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0
    ]
    return atoms[0] if len(atoms) == 1 else None


def ring_parent_candidates(mol, ordered: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
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

    if n == 6 and (all_aromatic or kekulized_aromatic):
        candidates = _six_membered_aromatic_candidates(mol, ordered, symbols)
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

    if n == 6 and symbols.count("O") == 1 and symbols.count("C") == 5 and not all_aromatic:
        candidates.extend(_unsaturated_pyran_candidates(mol, ordered))
        if candidates:
            return candidates

    if all_aliphatic and all(sym == "C" for sym in symbols) and 3 <= n <= max(ROOTS):
        for seq in _ring_numberings(ordered):
            parent = _carbocycle_parent(mol, seq)
            if parent:
                candidates.append((parent, seq))
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

    for hetero, names in _AROMATIC_5_AZOLE_NAMES.items():
        if counts != Counter({"C": 3, "N": 1, hetero: 1}):
            continue
        for seq in _numberings_with_atom_at(ordered, mol, hetero):
            n_locs = _symbol_locants(mol, seq, "N")
            if n_locs and n_locs[0] in names:
                candidates.append((names[n_locs[0]], seq))
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
        return root + "ane"
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
    return parent if parent.endswith("ane") else ""


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



def adjust_ring_parent_for_charge(mol, parent_name: str, seq: Sequence[int], raw_subs) -> str:
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
        if atom.GetSymbol() != "N":
            continue
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
            "imidazolidine",
            "piperazine",
            "morpholine",
        }:
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


def _ring_numberings(ordered: Sequence[int]) -> list[tuple[int, ...]]:
    seqs = []
    n = len(ordered)
    base = tuple(ordered)
    for direction in (base, tuple(reversed(base))):
        for i in range(n):
            seqs.append(direction[i:] + direction[:i])
    return seqs



def ring_substituent_suffix(parent_name: str, loc: int, charge_loc: int | None = None) -> str:
    if parent_name == "benzene":
        return "phenyl" if loc == 1 else ""
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
    stem = _SATURATED_SUBSTITUENT_STEMS.get(parent_name)
    if stem:
        return _substituent_suffix_from_stem(stem, loc, charge_loc)
    if parent_name.isascii() and " " not in parent_name and loc >= 1:
        stem = parent_name[:-1] if parent_name.endswith("e") else parent_name
        return _substituent_suffix_from_stem(stem, loc, charge_loc)
    return ""


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
