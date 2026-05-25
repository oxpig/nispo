from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from .constants import MULT, REPLACEMENT, ROOTS, SKELETON_ATOMS
from .formatting import loc_unsat


def _join_locants(locs: Sequence[int]) -> str:
    return ",".join(str(loc) for loc in sorted(locs))


def _single_indicated_hydrogen_n(mol, ordered: Sequence[int]) -> int | None:
    atoms = [
        idx
        for idx in ordered
        if mol.GetAtomWithIdx(idx).GetSymbol() == "N" and mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0
    ]
    return atoms[0] if len(atoms) == 1 else None


def _indicated_hydrogen_prefix(mol, ordered: Sequence[int], seq: Sequence[int], indicated: int | None) -> str:
    if indicated is not None and indicated in seq:
        return f"{seq.index(indicated) + 1}H-"
    has_ring_n_h = any(
        mol.GetAtomWithIdx(idx).GetSymbol() == "N" and mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0
        for idx in ordered
    )
    return "1H-" if has_ring_n_h else ""


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
    candidates: list[tuple[str, tuple[int, ...]]] = []

    if n == 3 and symbols.count("C") == 1 and symbols.count("N") == 2:
        candidates.extend(_diazirine_candidates(mol, ordered))
        if candidates:
            return candidates

    if n == 4 and symbols.count("C") == 3 and symbols.count("N") == 1:
        candidates.extend(_azetone_candidates(mol, ordered))
        if candidates:
            return candidates

    if n == 6 and all_aromatic:
        if symbols.count("C") == 6:
            return [("benzene", seq) for seq in _ring_numberings(ordered)]
        if symbols.count("N") == 1 and symbols.count("C") == 5:
            indicated = _single_indicated_hydrogen_n(mol, ordered)
            return [
                (_indicated_hydrogen_prefix(mol, ordered, seq, indicated) + "pyridine", seq)
                for seq in _numberings_with_atom_at(ordered, mol, "N")
            ]
        if symbols.count("N") == 2 and symbols.count("C") == 4:
            indicated = _single_indicated_hydrogen_n(mol, ordered)
            for seq in _numberings_with_atom_at(ordered, mol, "N"):
                n_locs = [i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "N"]
                second = sorted(n_locs)[1]
                name = {2: "pyridazine", 3: "pyrimidine", 4: "pyrazine"}.get(second)
                if name:
                    candidates.append((_indicated_hydrogen_prefix(mol, ordered, seq, indicated) + name, seq))
            return candidates
        if symbols.count("N") == 3 and symbols.count("C") == 3:
            for seq in _numberings_with_atom_at(ordered, mol, "N"):
                n_locs = tuple(i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "N")
                name = {
                    (1, 3, 5): "s-triazine",
                    (1, 2, 4): "1,2,4-triazine",
                    (1, 2, 3): "v-triazine",
                }.get(n_locs)
                if name:
                    candidates.append((name, seq))
            return candidates
        if symbols.count("N") == 4 and symbols.count("C") == 2:
            for seq in _numberings_with_atom_at(ordered, mol, "N"):
                n_locs = tuple(i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "N")
                candidates.append((f"{_join_locants(n_locs)}-tetrazine", seq))
            return candidates
        if symbols.count("N") == 5 and symbols.count("C") == 1:
            for seq in _numberings_with_atom_at(ordered, mol, "N"):
                n_locs = tuple(i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "N")
                candidates.append((f"{_join_locants(n_locs)}-pentazine", seq))
            return candidates
        if symbols.count("N") == 6:
            return [("hexazine", seq) for seq in _ring_numberings(ordered)]
        if symbols.count("O") == 1 and symbols.count("C") == 5:
            return [("2H-pyran", seq) for seq in _pyran_numberings(ordered, mol)]

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

    if n == 5 and all_aromatic:
        counts = Counter(symbols)
        if counts == Counter({"C": 2, "N": 1, "O": 2}):
            candidates.extend(_dioxazolone_candidates(mol, ordered))
            if candidates:
                return candidates
        if counts == Counter({"C": 4, "O": 1}):
            return [("furan", seq) for seq in _numberings_with_atom_at(ordered, mol, "O")]
        if counts == Counter({"C": 4, "S": 1}):
            return [("thiophene", seq) for seq in _numberings_with_atom_at(ordered, mol, "S")]
        if counts == Counter({"C": 4, "N": 1}):
            return [("1H-pyrrole", seq) for seq in _numberings_with_atom_at(ordered, mol, "N")]
        if counts == Counter({"C": 3, "N": 2}):
            indicated = _single_indicated_hydrogen_n(mol, ordered)
            for seq in _numberings_with_atom_at(ordered, mol, "N"):
                n_locs = [i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "N"]
                second = sorted(n_locs)[1]
                stem = {2: "pyrazole", 3: "imidazole"}.get(second)
                hprefix = _indicated_hydrogen_prefix(mol, ordered, seq, indicated)
                name = f"{hprefix}{stem}" if stem else ""
                if name:
                    candidates.append((name, seq))
            return candidates
        if counts == Counter({"C": 2, "N": 3}):
            indicated = _single_indicated_hydrogen_n(mol, ordered)
            for seq in _numberings_with_atom_at(ordered, mol, "N"):
                n_locs = tuple(i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "N")
                stem = {(1, 2, 3): "1,2,3-triazole", (1, 2, 4): "1,2,4-triazole"}.get(n_locs)
                hprefix = _indicated_hydrogen_prefix(mol, ordered, seq, indicated)
                name = f"{hprefix}{stem}" if stem else ""
                if name:
                    candidates.append((name, seq))
            return candidates
        if counts == Counter({"C": 1, "N": 4}):
            indicated = _single_indicated_hydrogen_n(mol, ordered)
            for seq in _numberings_with_atom_at(ordered, mol, "N"):
                if mol.GetAtomWithIdx(seq[4]).GetSymbol() != "C":
                    continue
                hprefix = _indicated_hydrogen_prefix(mol, ordered, seq, indicated)
                candidates.append((f"{hprefix}tetrazole", seq))
            return candidates
        for hetero, names in (("O", {2: "isoxazole", 3: "oxazole"}), ("S", {2: "isothiazole", 3: "thiazole"})):
            if counts == Counter({"C": 3, "N": 1, hetero: 1}):
                for seq in _numberings_with_atom_at(ordered, mol, hetero):
                    n_locs = [i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "N"]
                    if n_locs and n_locs[0] in names:
                        candidates.append((names[n_locs[0]], seq))
                return candidates
        for hetero, family in (("O", "oxadiazole"), ("S", "thiadiazole")):
            if counts == Counter({"C": 2, "N": 2, hetero: 1}):
                valid = {
                    (1, 2, 3): f"1,2,3-{family}",
                    (1, 2, 4): f"1,2,4-{family}",
                    (1, 2, 5): f"1,2,5-{family}",
                    (1, 3, 4): f"1,3,4-{family}",
                }
                for seq in _numberings_with_atom_at(ordered, mol, hetero):
                    locs = tuple(
                        i + 1
                        for i, idx in enumerate(seq)
                        if mol.GetAtomWithIdx(idx).GetSymbol() in {hetero, "N"}
                    )
                    name = valid.get(locs)
                    if name:
                        candidates.append((name, seq))
                return candidates

    if n == 6 and symbols.count("N") == 2 and symbols.count("C") == 4 and not all_aromatic:
        candidates.extend(_unsaturated_diazine_candidates(mol, ordered))
        if candidates:
            return candidates

    if all_aliphatic and all(sym == "C" for sym in symbols) and 3 <= n <= 20:
        for seq in _ring_numberings(ordered):
            parent = _carbocycle_parent(mol, seq)
            if parent:
                candidates.append((parent, seq))
        if candidates:
            return candidates

    if (
        all_aliphatic
        and 5 <= n <= 20
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

    if n == 5 and symbols.count("N") == 2 and symbols.count("C") == 3:
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
        if n == 3 and symbols.count("N") == 1 and symbols.count("C") == 2:
            return [("aziridine", seq) for seq in _numberings_with_atom_at(ordered, mol, "N")]
        if n == 3 and symbols.count("O") == 1 and symbols.count("C") == 2:
            return [("oxirane", seq) for seq in _numberings_with_atom_at(ordered, mol, "O")]
        if n == 3 and symbols.count("S") == 1 and symbols.count("C") == 2:
            return [("thiirane", seq) for seq in _numberings_with_atom_at(ordered, mol, "S")]
        if n == 4 and symbols.count("N") == 1 and symbols.count("C") == 3:
            return [("azetidine", seq) for seq in _numberings_with_atom_at(ordered, mol, "N")]
        if n == 4 and symbols.count("O") == 1 and symbols.count("C") == 3:
            return [("oxetane", seq) for seq in _numberings_with_atom_at(ordered, mol, "O")]
        if n == 4 and symbols.count("S") == 1 and symbols.count("C") == 3:
            return [("thietane", seq) for seq in _numberings_with_atom_at(ordered, mol, "S")]
        if n == 5 and symbols.count("O") == 2 and symbols.count("C") == 3:
            for seq in _numberings_with_atom_at(ordered, mol, "O"):
                o_locs = sorted(i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "O")
                if o_locs == [1, 3]:
                    candidates.append(("1,3-dioxolane", seq))
            if candidates:
                return candidates
        if n == 6 and symbols.count("O") == 2 and symbols.count("C") == 4:
            for seq in _numberings_with_atom_at(ordered, mol, "O"):
                o_locs = sorted(i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "O")
                if o_locs == [1, 3]:
                    candidates.append(("1,3-dioxane", seq))
                if o_locs == [1, 4]:
                    candidates.append(("1,4-dioxane", seq))
            if candidates:
                return candidates
        if n == 6 and symbols.count("N") == 1 and symbols.count("C") == 5:
            return [("piperidine", seq) for seq in _numberings_with_atom_at(ordered, mol, "N")]
        if n == 6 and symbols.count("O") == 1 and symbols.count("C") == 5:
            return [("oxane", seq) for seq in _numberings_with_atom_at(ordered, mol, "O")]
        if n == 6 and symbols.count("S") == 1 and symbols.count("C") == 5:
            return [("thiane", seq) for seq in _numberings_with_atom_at(ordered, mol, "S")]
        if n == 6 and symbols.count("N") == 2 and symbols.count("C") == 4:
            for seq in _numberings_with_atom_at(ordered, mol, "N"):
                n_locs = sorted(i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "N")
                if n_locs == [1, 4]:
                    candidates.append(("piperazine", seq))
            if candidates:
                return candidates
        if n == 6 and symbols.count("N") == 1 and symbols.count("O") == 1 and symbols.count("C") == 4:
            for seq in _numberings_with_atom_at(ordered, mol, "O"):
                n_locs = sorted(i + 1 for i, idx in enumerate(seq) if mol.GetAtomWithIdx(idx).GetSymbol() == "N")
                if n_locs == [4]:
                    candidates.append(("morpholine", seq))
            if candidates:
                return candidates
        if n == 7 and symbols.count("N") == 1 and symbols.count("C") == 6:
            return [("azepane", seq) for seq in _numberings_with_atom_at(ordered, mol, "N")]
        if n == 7 and symbols.count("O") == 1 and symbols.count("C") == 6:
            return [("oxepane", seq) for seq in _numberings_with_atom_at(ordered, mol, "O")]
        if n == 7 and symbols.count("S") == 1 and symbols.count("C") == 6:
            return [("thiepane", seq) for seq in _numberings_with_atom_at(ordered, mol, "S")]
        if n == 5 and symbols.count("N") == 1 and symbols.count("C") == 4:
            return [("pyrrolidine", seq) for seq in _numberings_with_atom_at(ordered, mol, "N")]
        if n == 5 and symbols.count("O") == 1 and symbols.count("C") == 4:
            return [("oxolane", seq) for seq in _numberings_with_atom_at(ordered, mol, "O")]
        if 3 <= n <= max(ROOTS) and all(sym in SKELETON_ATOMS for sym in symbols) and any(sym != "C" for sym in symbols):
            for seq in _ring_numberings(ordered):
                if mol.GetAtomWithIdx(seq[0]).GetSymbol() == "C":
                    continue
                parent = _heterocycloalkane_parent(mol, seq)
                if parent:
                    candidates.append((parent, seq))
            return candidates

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


def _heterocycle_parent(mol, seq: Sequence[int]) -> str:
    root = ROOTS.get(len(seq))
    if not root:
        return ""
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
    parent = _carbocycle_parent(mol, seq)
    if not parent:
        return ""
    return "-".join(parts) + parent



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
        if atom.GetFormalCharge() <= 0:
            continue
        if atom.GetSymbol() in {"O", "S"}:
            stem = parent_name[:-1] if parent_name.endswith("e") else parent_name
            return f"{stem}-{loc}-ium"
        if atom.GetSymbol() != "N":
            continue
        if parent_name == "pyridine" and loc == 1:
            return "pyridin-1-ium"
        if parent_name in {"pyrimidine", "pyrazine", "pyridazine"}:
            stem = parent_name[:-1]
            return f"{stem}-{loc}-ium"
        if parent_name.endswith(("triazine", "tetrazine", "pentazine", "hexazine")):
            stem = parent_name[:-1] if parent_name.endswith("e") else parent_name
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
        return ""
    return parent_name


def _drop_indicated_hydrogen_prefix(parent_name: str) -> str:
    if len(parent_name) > 3 and parent_name[0].isdigit() and parent_name[1:3] == "H-":
        return parent_name[3:]
    return parent_name


def _dioxazolone_candidates(mol, ordered: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    ring_set = set(ordered)
    for seq in _ring_numberings(ordered):
        if [mol.GetAtomWithIdx(idx).GetSymbol() for idx in seq] != ["O", "C", "O", "N", "C"]:
            continue
        if not _ring_atom_has_exocyclic_oxo(mol, seq[1], ring_set):
            continue
        candidates.append(("1,3,4-dioxazol-2-one", seq))
    return candidates


def _diazirine_candidates(mol, ordered: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    for seq in _ring_numberings(ordered):
        if [mol.GetAtomWithIdx(idx).GetSymbol() for idx in seq] != ["N", "N", "C"]:
            continue
        bond = mol.GetBondBetweenAtoms(seq[0], seq[1])
        if bond is None or bond.GetBondTypeAsDouble() != 2.0:
            continue
        candidates.append(("3H-diazirine", seq))
    return candidates


def _azetone_candidates(mol, ordered: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    ring_set = set(ordered)
    for seq in _ring_numberings(ordered):
        if [mol.GetAtomWithIdx(idx).GetSymbol() for idx in seq] != ["N", "C", "C", "C"]:
            continue
        if not _ring_atom_has_exocyclic_oxo(mol, seq[1], ring_set):
            continue
        unsat = mol.GetBondBetweenAtoms(seq[2], seq[3])
        if unsat is None or unsat.GetBondTypeAsDouble() != 2.0:
            continue
        candidates.append(("azet-2-one", seq))
    return candidates


def _ring_atom_has_exocyclic_oxo(mol, atom_idx: int, ring_set: set[int]) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() != "C":
        return False
    return any(
        nbr.GetIdx() not in ring_set
        and nbr.GetSymbol() == "O"
        and mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx()).GetBondTypeAsDouble() == 2.0
        for nbr in atom.GetNeighbors()
    )



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
    h_azine = _azine_indicated_hydrogen_substituent_suffix(parent_name, loc, charge_loc)
    if h_azine:
        return h_azine
    if parent_name == "pyridine":
        if charge_loc is not None:
            return f"pyridin-{charge_loc}-ium-{loc}-yl"
        return f"pyridin-{loc}-yl"
    if parent_name == "pyrimidine":
        if charge_loc is not None:
            return f"pyrimidin-{charge_loc}-ium-{loc}-yl"
        return f"pyrimidin-{loc}-yl"
    if parent_name.endswith("dihydropyrimidine"):
        return f"{parent_name[:-1]}-{loc}-yl"
    if parent_name == "pyrazine":
        if charge_loc is not None:
            return f"pyrazin-{charge_loc}-ium-{loc}-yl"
        return f"pyrazin-{loc}-yl"
    if parent_name == "pyridazine":
        if charge_loc is not None:
            return f"pyridazin-{charge_loc}-ium-{loc}-yl"
        return f"pyridazin-{loc}-yl"
    if parent_name == "furan":
        return f"furan-{loc}-yl"
    if parent_name == "thiophene":
        if charge_loc is not None:
            return f"thiophen-{charge_loc}-ium-{loc}-yl"
        return f"thiophen-{loc}-yl"
    if parent_name == "1H-pyrrole":
        return f"1H-pyrrol-{loc}-yl"
    if parent_name == "1H-imidazole":
        if charge_loc is not None:
            return f"imidazol-{charge_loc}-ium-{loc}-yl"
        return f"1H-imidazol-{loc}-yl"
    if parent_name == "1H-pyrazole":
        if charge_loc is not None:
            return f"pyrazol-{charge_loc}-ium-{loc}-yl"
        return f"1H-pyrazol-{loc}-yl"
    if parent_name.endswith("H-imidazole"):
        if charge_loc is not None:
            return f"{_drop_indicated_hydrogen_prefix(parent_name)[:-1]}-{charge_loc}-ium-{loc}-yl"
        return f"{parent_name[:-1]}-{loc}-yl"
    if parent_name.endswith("H-pyrazole"):
        if charge_loc is not None:
            return f"{_drop_indicated_hydrogen_prefix(parent_name)[:-1]}-{charge_loc}-ium-{loc}-yl"
        return f"{parent_name[:-1]}-{loc}-yl"
    if parent_name == "oxazole":
        return f"oxazol-{loc}-yl"
    if parent_name == "isoxazole":
        return f"isoxazol-{loc}-yl"
    if parent_name == "3H-diazirine":
        return f"3H-diazirin-{loc}-yl"
    if parent_name == "azet-2-one":
        return f"2-oxoazet-{loc}-yl"
    if parent_name == "1,3,4-dioxazol-2-one":
        return f"2-oxo-1,3,4-dioxazol-{loc}-yl"
    if parent_name == "thiazole":
        if charge_loc is not None:
            return f"thiazol-{charge_loc}-ium-{loc}-yl"
        return f"thiazol-{loc}-yl"
    if parent_name == "isothiazole":
        if charge_loc is not None:
            return f"isothiazol-{charge_loc}-ium-{loc}-yl"
        return f"isothiazol-{loc}-yl"
    if parent_name == "1H-1,2,3-triazole":
        return f"1H-1,2,3-triazol-{loc}-yl"
    if parent_name == "1H-1,2,4-triazole":
        return f"1H-1,2,4-triazol-{loc}-yl"
    if parent_name == "1H-tetrazole":
        return f"1H-tetrazol-{loc}-yl"
    if parent_name.endswith("triazole"):
        if charge_loc is not None:
            base = _drop_indicated_hydrogen_prefix(parent_name)
            stem = base[:-1] if base.endswith("e") else base
            return f"{stem}-{charge_loc}-ium-{loc}-yl"
        return f"{parent_name[:-1]}-{loc}-yl"
    if parent_name.endswith("tetrazole"):
        if charge_loc is not None:
            base = _drop_indicated_hydrogen_prefix(parent_name)
            stem = base[:-1] if base.endswith("e") else base
            return f"{stem}-{charge_loc}-ium-{loc}-yl"
        return f"{parent_name[:-1]}-{loc}-yl"
    if parent_name.startswith("cyclo") and parent_name.endswith("ane"):
        return parent_name[:-3] + "yl" if loc == 1 else ""
    if parent_name == "piperidine":
        if charge_loc is not None:
            return f"piperidin-{charge_loc}-ium-{loc}-yl"
        return f"piperidin-{loc}-yl"
    if parent_name == "piperazine":
        if charge_loc is not None:
            return f"piperazin-{charge_loc}-ium-{loc}-yl"
        return f"piperazin-{loc}-yl"
    if parent_name == "morpholine":
        if charge_loc is not None:
            return f"morpholin-{charge_loc}-ium-{loc}-yl"
        return f"morpholin-{loc}-yl"
    saturated_stems = {
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
    stem = saturated_stems.get(parent_name)
    if stem:
        if charge_loc is not None:
            return f"{stem}-{charge_loc}-ium-{loc}-yl"
        return f"{stem}-{loc}-yl"
    if parent_name.isascii() and " " not in parent_name and loc >= 1:
        stem = parent_name[:-1] if parent_name.endswith("e") else parent_name
        if charge_loc is not None:
            return f"{stem}-{charge_loc}-ium-{loc}-yl"
        return f"{stem}-{loc}-yl"
    return ""


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
