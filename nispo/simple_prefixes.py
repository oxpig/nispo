from __future__ import annotations

from .candidate_facts import PrefixCandidate
from .constants import HALO, MULT, ROOTS
from .molecule_cache import mol_cache_dict, mol_cache_get


_SIMPLE_PREFIX_SOURCE = "simple_external_prefix"


def simple_external_prefix(mol, idx: int, parent_idx: int) -> str:
    """Render safe terminal substituents without entering branch dispatch."""

    candidates = simple_external_prefix_candidates(mol, idx, parent_idx)
    if not candidates:
        return ""
    candidate = candidates[0]
    _record_structured_prefix_selected(mol, candidate)
    return candidate.text


def simple_external_prefix_candidates(mol, idx: int, parent_idx: int) -> tuple[PrefixCandidate, ...]:
    """Return graph-backed facts for safe terminal substituents."""

    cache = mol_cache_dict(mol, "simple_external_prefix_candidate_cache")
    key = (idx, parent_idx)
    cached = cache.get(key)
    if cached is not None:
        return cached
    atom = mol.GetAtomWithIdx(idx)
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None:
        cache[key] = ()
        return ()
    order = bond.GetBondTypeAsDouble()
    symbol = atom.GetSymbol()
    if symbol in HALO:
        candidate = _terminal_halogen_prefix(atom, idx, parent_idx, order)
    elif symbol == "C":
        candidate = (
            _simple_methylidene_prefix(mol, idx, parent_idx, order)
            or _simple_isotopic_methyl_prefix(mol, idx, parent_idx, order)
            or _simple_fluoromethyl_prefix(mol, idx, parent_idx, order)
            or _simple_halomethyl_prefix(mol, idx, parent_idx, order)
            or _simple_haloethyl_prefix(mol, idx, parent_idx, order)
            or _simple_fluoroalkyl_prefix(mol, idx, parent_idx, order)
            or _simple_branched_alkyl_prefix(mol, idx, parent_idx, order)
            or _simple_benzyl_prefix(mol, idx, parent_idx, order)
            or _simple_hydroxymethyl_prefix(mol, idx, parent_idx, order)
            or _simple_alkoxymethyl_prefix(mol, idx, parent_idx, order)
            or _simple_aminomethyl_prefix(mol, idx, parent_idx, order)
            or _simple_dimethylaminomethyl_prefix(mol, idx, parent_idx, order)
            or _simple_bocaminomethyl_prefix(mol, idx, parent_idx, order)
            or _simple_cyanomethyl_prefix(mol, idx, parent_idx, order)
            or _simple_carboxymethyl_prefix(mol, idx, parent_idx, order)
            or _simple_acetenyl_prefix(mol, idx, parent_idx, order)
            or _simple_alkenyl_prefix(mol, idx, parent_idx, order)
            or _simple_oxaalkyl_prefix(mol, idx, parent_idx, order)
            or _simple_cycloalkylmethyl_prefix(mol, idx, parent_idx, order)
            or _simple_linear_alkyl_prefix(
                mol,
                idx,
                parent_idx,
                order,
            )
        )
    elif symbol == "O":
        candidate = _simple_oxygen_prefix(mol, atom, idx, parent_idx, order)
    elif symbol == "N":
        candidate = _simple_nitrogen_prefix(mol, atom, idx, parent_idx, order)
    else:
        candidate = None
    candidates = (candidate,) if candidate is not None else ()
    cache[key] = candidates
    return candidates


def _terminal_halogen_prefix(atom, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0 or atom.GetFormalCharge() != 0:
        return None
    if any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
        return None
    isotope = atom.GetIsotope()
    symbol = atom.GetSymbol()
    text = f"[{isotope}{symbol}]{HALO[symbol]}" if isotope else HALO[symbol]
    return _prefix_candidate(text, {idx}, idx, parent_idx, "terminal-halogen")


def _simple_methylidene_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 2.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    if any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
        return None
    return _prefix_candidate("methylidene", {idx}, idx, parent_idx, "methylidene")


def _simple_isotopic_methyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    deuterium_count = 0
    tritium_count = 0
    hydrogens = set()
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        if nbr.GetSymbol() != "H":
            return None
        hydrogens.add(nidx)
        if nbr.GetIsotope() == 2:
            deuterium_count += 1
        elif nbr.GetIsotope() == 3:
            tritium_count += 1
        else:
            return None
    if deuterium_count and tritium_count:
        return None
    if deuterium_count:
        text = _isotopic_methyl_text(deuterium_count, "deuterio")
    elif tritium_count:
        text = _isotopic_methyl_text(tritium_count, "tritio")
    else:
        return None
    return _prefix_candidate(text, {idx, *hydrogens}, idx, parent_idx, "isotopic-methyl")


def _isotopic_methyl_text(count: int, isotope_prefix: str) -> str:
    if count == 1:
        return isotope_prefix + "methyl"
    multiplier = MULT.get(count)
    return f"{multiplier}{isotope_prefix}methyl" if multiplier else ""


def _simple_fluoromethyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if order != 1.0 or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(atom):
        return None
    external = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if not external:
        return None
    if any(mol.GetAtomWithIdx(nidx).GetSymbol() != "F" for nidx in external):
        return None
    count = len(external)
    if count == 1:
        text = "fluoromethyl"
    elif count == 2:
        text = "difluoromethyl"
    elif count == 3:
        text = "trifluoromethyl"
    else:
        return None
    return _prefix_candidate(text, {idx, *external}, idx, parent_idx, "fluoromethyl")


def _simple_halomethyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if order != 1.0 or atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(atom):
        return None
    external = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    halo_text = _same_terminal_halo_prefix(mol, idx, external, skip_symbol="F")
    if halo_text is None:
        return None
    return _prefix_candidate(f"{halo_text}methyl", {idx, *external}, idx, parent_idx, "halomethyl")


def _simple_haloethyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if order != 1.0 or atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(atom):
        return None
    external = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(external) != 1:
        return None
    distal_idx = external[0]
    distal = mol.GetAtomWithIdx(distal_idx)
    if distal.GetSymbol() != "C" or distal.GetFormalCharge() != 0 or distal.IsInRing() or distal.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(distal):
        return None
    bond = mol.GetBondBetweenAtoms(idx, distal_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    halogens = [
        nbr.GetIdx()
        for nbr in distal.GetNeighbors()
        if nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1
    ]
    halo_text = _same_terminal_halo_prefix(mol, distal_idx, halogens)
    if halo_text is None:
        return None
    locants = {1: "2", 2: "2,2", 3: "2,2,2"}.get(len(halogens))
    if locants is None:
        return None
    return _prefix_candidate(
        f"{locants}-{halo_text}ethyl",
        {idx, distal_idx, *halogens},
        idx,
        parent_idx,
        "haloethyl",
    )


def _simple_fluoroalkyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    previous = parent_idx
    current = idx
    path: list[int] = []
    fluoro_locs: list[int] = []
    fluorines: set[int] = set()
    while len(path) < max(ROOTS):
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
            return None
        if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
            return None
        bond = mol.GetBondBetweenAtoms(previous, current)
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None
        path.append(current)
        locant = len(path)
        next_carbons: list[int] = []
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == previous or nbr.GetAtomicNum() <= 1:
                continue
            nbr_bond = mol.GetBondBetweenAtoms(current, nidx)
            if nbr_bond is None or nbr_bond.GetBondTypeAsDouble() != 1.0:
                return None
            if nbr.GetSymbol() == "F" and nbr.GetFormalCharge() == 0 and not nbr.GetIsotope():
                if any(extra.GetIdx() != current and extra.GetAtomicNum() > 1 for extra in nbr.GetNeighbors()):
                    return None
                fluoro_locs.append(locant)
                fluorines.add(nidx)
                continue
            if nbr.GetSymbol() != "C":
                return None
            next_carbons.append(nidx)
        if len(next_carbons) > 1:
            return None
        if not next_carbons:
            if not fluoro_locs or len(path) < 4:
                return None
            root = ROOTS.get(len(path))
            fluoro_prefix = _locanted_multiplier_prefix(fluoro_locs, "fluoro")
            if not root or not fluoro_prefix:
                return None
            text = f"{fluoro_prefix}{root}yl"
            return _prefix_candidate(text, {*path, *fluorines}, idx, parent_idx, "fluoroalkyl")
        previous, current = current, next_carbons[0]
    return None


def _simple_alkenyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    external = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(external) != 1:
        return None
    first_idx = external[0]
    first = mol.GetAtomWithIdx(first_idx)
    if first.GetSymbol() != "C" or first.GetFormalCharge() != 0 or first.IsInRing() or first.GetIsotope():
        return None
    if _atom_has_assigned_stereo(first) or _atom_has_isotopic_hydrogen(first):
        return None
    first_bond = mol.GetBondBetweenAtoms(idx, first_idx)
    if first_bond is None:
        return None
    first_external = [
        nbr.GetIdx()
        for nbr in first.GetNeighbors()
        if nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1
    ]
    if first_bond.GetBondTypeAsDouble() == 2.0:
        if first_external:
            return None
        return _prefix_candidate("ethen-1-yl", {idx, first_idx}, idx, parent_idx, "alkenyl")
    if first_bond.GetBondTypeAsDouble() != 1.0 or len(first_external) != 1:
        return None
    second_idx = first_external[0]
    second = mol.GetAtomWithIdx(second_idx)
    if second.GetSymbol() != "C" or second.GetFormalCharge() != 0 or second.IsInRing() or second.GetIsotope():
        return None
    if _atom_has_assigned_stereo(second) or _atom_has_isotopic_hydrogen(second):
        return None
    second_bond = mol.GetBondBetweenAtoms(first_idx, second_idx)
    if second_bond is None or second_bond.GetBondTypeAsDouble() != 2.0:
        return None
    if any(nbr.GetIdx() != first_idx and nbr.GetAtomicNum() > 1 for nbr in second.GetNeighbors()):
        return None
    return _prefix_candidate("2-propen-1-yl", {idx, first_idx, second_idx}, idx, parent_idx, "alkenyl")


def _simple_acetenyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(atom):
        return None
    external = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(external) != 1:
        return None
    terminal_idx = external[0]
    terminal = mol.GetAtomWithIdx(terminal_idx)
    if terminal.GetSymbol() != "C" or terminal.GetFormalCharge() != 0 or terminal.IsInRing() or terminal.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(terminal):
        return None
    triple_bond = mol.GetBondBetweenAtoms(idx, terminal_idx)
    if triple_bond is None or triple_bond.GetBondTypeAsDouble() != 3.0:
        return None
    if any(nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1 for nbr in terminal.GetNeighbors()):
        return None
    return _prefix_candidate("acetenyl", {idx, terminal_idx}, idx, parent_idx, "acetenyl")


def _simple_oxaalkyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    external = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(external) != 1:
        return None
    carbon_idx = external[0]
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing() or carbon.GetIsotope():
        return None
    if _atom_has_assigned_stereo(carbon) or _atom_has_isotopic_hydrogen(carbon):
        return None
    carbon_bond = mol.GetBondBetweenAtoms(idx, carbon_idx)
    if carbon_bond is None or carbon_bond.GetBondTypeAsDouble() != 1.0:
        return None
    carbon_external = [
        nbr.GetIdx()
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1
    ]
    if len(carbon_external) != 1:
        return None
    oxygen_idx = carbon_external[0]
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0 or oxygen.IsInRing() or oxygen.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(oxygen):
        return None
    oxygen_bond = mol.GetBondBetweenAtoms(carbon_idx, oxygen_idx)
    if oxygen_bond is None or oxygen_bond.GetBondTypeAsDouble() != 1.0:
        return None
    oxygen_external = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1
    ]
    if not oxygen_external:
        return _prefix_candidate("3-oxapropyl", {idx, carbon_idx, oxygen_idx}, idx, parent_idx, "oxaalkyl")
    if len(oxygen_external) != 1:
        return None
    methyl_idx = oxygen_external[0]
    if not _is_terminal_methyl(mol, methyl_idx, oxygen_idx):
        return None
    return _prefix_candidate(
        "3-oxabutyl",
        {idx, carbon_idx, oxygen_idx, methyl_idx},
        idx,
        parent_idx,
        "oxaalkyl",
    )


def _same_terminal_halo_prefix(
    mol,
    carbon_idx: int,
    halogen_indices: list[int],
    *,
    skip_symbol: str = "",
) -> str | None:
    if not halogen_indices or len(halogen_indices) > 3:
        return None
    symbols = []
    for halogen_idx in halogen_indices:
        atom = mol.GetAtomWithIdx(halogen_idx)
        bond = mol.GetBondBetweenAtoms(carbon_idx, halogen_idx)
        symbol = atom.GetSymbol()
        if (
            symbol not in HALO
            or symbol == skip_symbol
            or atom.GetFormalCharge() != 0
            or atom.GetIsotope()
            or bond is None
            or bond.GetBondTypeAsDouble() != 1.0
        ):
            return None
        if any(nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
            return None
        symbols.append(symbol)
    if len(set(symbols)) != 1:
        return None
    return _halo_count_prefix(symbols[0], len(symbols))


def _halo_count_prefix(symbol: str, count: int) -> str:
    text = HALO.get(symbol, "")
    if not text:
        return ""
    if count == 1:
        return text
    return {2: "di", 3: "tri"}.get(count, "") + text


def _locanted_multiplier_prefix(locs: list[int], term: str) -> str:
    if not locs:
        return ""
    ordered = sorted(locs)
    locant_text = ",".join(str(value) for value in ordered)
    if len(ordered) == 1:
        return f"{locant_text}-{term}"
    return f"{locant_text}-{MULT.get(len(ordered), '')}{term}"


def _simple_linear_alkyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    names = {1: "methyl", 2: "ethyl", 3: "propyl", 4: "butyl"}
    if order != 1.0:
        return None
    chain = []
    previous = parent_idx
    current = idx
    while len(chain) < 4:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            return None
        if atom.GetIsotope() or _atom_has_isotopic_hydrogen(atom):
            return None
        bond = mol.GetBondBetweenAtoms(previous, current)
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None
        chain.append(current)
        heavy = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetIdx() != previous and nbr.GetAtomicNum() > 1
        ]
        if not heavy:
            text = names[len(chain)]
            return _prefix_candidate(text, set(chain), idx, parent_idx, "linear-alkyl")
        if len(heavy) != 1:
            return None
        next_idx = heavy[0]
        if mol.GetAtomWithIdx(next_idx).GetSymbol() != "C":
            return None
        previous, current = current, next_idx
    return None


def _simple_branched_alkyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(atom):
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) in {2, 3}:
        if not all(_is_terminal_methyl(mol, nbr_idx, idx) for nbr_idx in heavy):
            return None
        text = "isopropyl" if len(heavy) == 2 else "t-butyl"
        return _prefix_candidate(text, {idx, *heavy}, idx, parent_idx, "branched-alkyl")
    if len(heavy) != 1:
        return None
    attached_idx = heavy[0]
    attached = mol.GetAtomWithIdx(attached_idx)
    if attached.GetSymbol() != "C" or attached.GetFormalCharge() != 0 or attached.IsInRing() or attached.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(attached):
        return None
    bond = mol.GetBondBetweenAtoms(idx, attached_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    attached_heavy = [
        nbr.GetIdx()
        for nbr in attached.GetNeighbors()
        if nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1
    ]
    if len(attached_heavy) in {2, 3} and all(
        _is_terminal_methyl(mol, nbr_idx, attached_idx) for nbr_idx in attached_heavy
    ):
        text = "isobutyl" if len(attached_heavy) == 2 else "neopentyl"
        return _prefix_candidate(text, {idx, attached_idx, *attached_heavy}, idx, parent_idx, "branched-alkyl")
    return None


def _is_terminal_methyl(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return False
    if _atom_has_isotopic_hydrogen(atom):
        return False
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return False
    return not any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors())


def _simple_benzyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(atom):
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    ring_idx = heavy[0]
    ring_atom = mol.GetAtomWithIdx(ring_idx)
    if not ring_atom.IsInRing():
        return None
    bond = mol.GetBondBetweenAtoms(idx, ring_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    ring = _phenyl_ring(mol, ring_idx, idx)
    if ring is None:
        return None
    substituent = _single_substituted_benzyl_substituent(mol, ring, ring_idx, idx)
    if substituent is not None:
        locant, substituent_text, substituent_atoms = substituent
        return _prefix_candidate(
            f"{locant}-{substituent_text}benzyl",
            {idx, *ring, *substituent_atoms},
            idx,
            parent_idx,
            "substituted-benzyl",
        )
    if not _phenyl_ring_is_unsubstituted(mol, ring, ring_idx, idx):
        return None
    return _prefix_candidate("benzyl", {idx, *ring}, idx, parent_idx, "benzyl")


def _simple_cycloalkylmethyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    ring_idx = heavy[0]
    ring = _simple_cycloalkyl_ring(mol, ring_idx, idx)
    if ring is None:
        return None
    names = {
        3: "cyclopropylmethyl",
        4: "cyclobutylmethyl",
        5: "cyclopentylmethyl",
        6: "cyclohexylmethyl",
    }
    text = names.get(len(ring))
    if text is None:
        return None
    return _prefix_candidate(text, {idx, *ring}, idx, parent_idx, "cycloalkylmethyl")


def _simple_hydroxymethyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    oxygen_idx = heavy[0]
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0 or oxygen.IsInRing() or oxygen.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(oxygen):
        return None
    bond = mol.GetBondBetweenAtoms(idx, oxygen_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    if any(nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1 for nbr in oxygen.GetNeighbors()):
        return None
    return _prefix_candidate("hydroxymethyl", {idx, oxygen_idx}, idx, parent_idx, "hydroxymethyl")


def _simple_alkoxymethyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    oxygen_idx = heavy[0]
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0 or oxygen.IsInRing() or oxygen.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(oxygen):
        return None
    bond = mol.GetBondBetweenAtoms(idx, oxygen_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    oxygen_heavy = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1
    ]
    if len(oxygen_heavy) != 1:
        return None
    alkoxy = _simple_linear_alkoxy_prefix(mol, oxygen_heavy[0], oxygen_idx, idx)
    if alkoxy is None:
        return None
    return _prefix_candidate(
        f"{alkoxy.text}methyl",
        {idx, *alkoxy.atoms},
        idx,
        parent_idx,
        "alkoxymethyl",
    )


def _simple_aminomethyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    nitrogen_idx = heavy[0]
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing() or nitrogen.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(nitrogen):
        return None
    bond = mol.GetBondBetweenAtoms(idx, nitrogen_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    if any(nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1 for nbr in nitrogen.GetNeighbors()):
        return None
    return _prefix_candidate("aminomethyl", {idx, nitrogen_idx}, idx, parent_idx, "aminomethyl")


def _simple_dimethylaminomethyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    nitrogen_idx = heavy[0]
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing() or nitrogen.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(nitrogen):
        return None
    bond = mol.GetBondBetweenAtoms(idx, nitrogen_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    methyls = [
        nbr.GetIdx()
        for nbr in nitrogen.GetNeighbors()
        if nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1
    ]
    if len(methyls) != 2 or not all(_is_terminal_methyl(mol, methyl_idx, nitrogen_idx) for methyl_idx in methyls):
        return None
    return _prefix_candidate(
        "dimethylaminomethyl",
        {idx, nitrogen_idx, *methyls},
        idx,
        parent_idx,
        "dimethylaminomethyl",
    )


def _simple_bocaminomethyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    nitrogen_idx = heavy[0]
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing() or nitrogen.GetIsotope():
        return None
    if _atom_has_isotopic_hydrogen(nitrogen):
        return None
    bond = mol.GetBondBetweenAtoms(idx, nitrogen_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    carbonyls = []
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx or nbr.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if nbr.GetSymbol() == "C" and bond is not None and bond.GetBondTypeAsDouble() == 1.0:
            boc_atoms = _terminal_boc_carbonyl_atoms(mol, nidx, nitrogen_idx)
            if boc_atoms is None:
                return None
            carbonyls.append(boc_atoms)
        else:
            return None
    if len(carbonyls) != 1:
        return None
    return _prefix_candidate(
        "bocaminomethyl",
        {idx, nitrogen_idx, *carbonyls[0]},
        idx,
        parent_idx,
        "bocaminomethyl",
    )


def _terminal_boc_carbonyl_atoms(mol, carbonyl_idx: int, nitrogen_idx: int) -> set[int] | None:
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0 or carbonyl.IsInRing() or carbonyl.GetIsotope():
        return None
    double_oxygens = []
    single_oxygens = []
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond is not None and bond.GetBondTypeAsDouble() == 2.0:
            if nbr.GetFormalCharge() != 0 or nbr.GetIsotope():
                return None
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "O" and bond is not None and bond.GetBondTypeAsDouble() == 1.0:
            if nbr.GetFormalCharge() != 0 or nbr.IsInRing() or nbr.GetIsotope():
                return None
            single_oxygens.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 1 or len(single_oxygens) != 1:
        return None
    tert_butyl = _tert_butyl_oxy_atoms(mol, single_oxygens[0], carbonyl_idx)
    if tert_butyl is None:
        return None
    return {carbonyl_idx, double_oxygens[0], single_oxygens[0], *tert_butyl}


def _tert_butyl_oxy_atoms(mol, oxygen_idx: int, carbonyl_idx: int) -> set[int] | None:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    heavy = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != carbonyl_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    central_idx = heavy[0]
    central = mol.GetAtomWithIdx(central_idx)
    if central.GetSymbol() != "C" or central.GetFormalCharge() != 0 or central.IsInRing() or central.GetIsotope():
        return None
    bond = mol.GetBondBetweenAtoms(oxygen_idx, central_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    methyls = [
        nbr.GetIdx()
        for nbr in central.GetNeighbors()
        if nbr.GetIdx() != oxygen_idx and nbr.GetAtomicNum() > 1
    ]
    if len(methyls) != 3 or not all(_is_terminal_methyl(mol, methyl_idx, central_idx) for methyl_idx in methyls):
        return None
    return {central_idx, *methyls}


def _simple_cyanomethyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    nitrile = _terminal_nitrile_atoms(mol, heavy[0], idx)
    if nitrile is None:
        return None
    return _prefix_candidate("cyanomethyl", {idx, *nitrile}, idx, parent_idx, "cyanomethyl")


def _terminal_nitrile_atoms(mol, carbon_idx: int, parent_idx: int) -> set[int] | None:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing() or carbon.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(carbon_idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    nitrogens = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if (
            nbr.GetSymbol() == "N"
            and nbr.GetFormalCharge() == 0
            and not nbr.IsInRing()
            and not nbr.GetIsotope()
            and bond is not None
            and bond.GetBondTypeAsDouble() == 3.0
        ):
            nitrogens.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(nitrogens) != 1:
        return None
    return {carbon_idx, nitrogens[0]}


def _simple_carboxymethyl_prefix(mol, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if order != 1.0:
        return None
    parent = mol.GetAtomWithIdx(parent_idx)
    if parent.GetSymbol() == "N":
        return None
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    carboxy = _terminal_carboxyl_atoms(mol, heavy[0], idx)
    if carboxy is None:
        return None
    text, atoms = carboxy
    return _prefix_candidate(f"{text}methyl", {idx, *atoms}, idx, parent_idx, f"{text}methyl")


def _terminal_carboxyl_atoms(mol, carbon_idx: int, parent_idx: int) -> tuple[str, set[int]] | None:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing() or carbon.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(carbon_idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    double_oxygens = []
    single_oxygens = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 2.0:
            if nbr.GetFormalCharge() != 0 or nbr.GetIsotope():
                return None
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 1.0:
            if nbr.GetIsotope() or any(extra.GetIdx() != carbon_idx and extra.GetAtomicNum() > 1 for extra in nbr.GetNeighbors()):
                return None
            single_oxygens.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 1 or len(single_oxygens) != 1:
        return None
    single_oxygen = mol.GetAtomWithIdx(single_oxygens[0])
    if single_oxygen.GetFormalCharge() == 0:
        text = "carboxy"
    elif single_oxygen.GetFormalCharge() < 0:
        text = "carboxylato"
    else:
        return None
    return text, {carbon_idx, double_oxygens[0], single_oxygens[0]}


def _phenyl_ring(mol, attachment_idx: int, parent_idx: int) -> tuple[int, ...] | None:
    rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings() if attachment_idx in ring and parent_idx not in ring]
    if len(rings) != 1:
        return None
    ring = rings[0]
    if len(ring) != 6:
        return None
    ring_set = set(ring)
    atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
    if any(
        atom.GetSymbol() != "C"
        or atom.GetFormalCharge() != 0
        or atom.GetIsotope()
        or not atom.GetIsAromatic()
        or _atom_has_assigned_stereo(atom)
        or _atom_has_isotopic_hydrogen(atom)
        for atom in atoms
    ):
        return None
    return ring


def _simple_cycloalkyl_ring(mol, attachment_idx: int, parent_idx: int) -> tuple[int, ...] | None:
    rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings() if attachment_idx in ring and parent_idx not in ring]
    if len(rings) != 1:
        return None
    ring = rings[0]
    if len(ring) not in {3, 4, 5, 6}:
        return None
    ring_set = set(ring)
    atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
    if any(
        atom.GetSymbol() != "C"
        or atom.GetFormalCharge() != 0
        or atom.GetIsotope()
        or atom.GetIsAromatic()
        or _atom_has_assigned_stereo(atom)
        or _atom_has_isotopic_hydrogen(atom)
        for atom in atoms
    ):
        return None
    for atom in atoms:
        ring_neighbor_count = 0
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nidx)
            if nidx in ring_set:
                if bond is None or bond.GetBondTypeAsDouble() != 1.0:
                    return None
                ring_neighbor_count += 1
                continue
            if atom.GetIdx() == attachment_idx and nidx == parent_idx:
                if bond is None or bond.GetBondTypeAsDouble() != 1.0:
                    return None
                continue
            if nbr.GetAtomicNum() > 1:
                return None
        if ring_neighbor_count != 2:
            return None
    return ring


def _unsubstituted_phenyl_ring(mol, attachment_idx: int, parent_idx: int) -> tuple[int, ...] | None:
    ring = _phenyl_ring(mol, attachment_idx, parent_idx)
    if ring is None or not _phenyl_ring_is_unsubstituted(mol, ring, attachment_idx, parent_idx):
        return None
    return ring


def _phenyl_ring_is_unsubstituted(mol, ring: tuple[int, ...], attachment_idx: int, parent_idx: int) -> bool:
    ring_set = set(ring)
    atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
    for atom in atoms:
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set or (atom.GetIdx() == attachment_idx and nidx == parent_idx):
                continue
            if nbr.GetAtomicNum() > 1:
                return False
    return True


_BENZYL_SIMPLE_SUBSTITUENTS = {
    "amino",
    "bromo",
    "chloro",
    "fluoro",
    "hydroxy",
    "iodo",
    "methoxy",
    "methyl",
    "trifluoromethyl",
}


def _single_substituted_benzyl_substituent(
    mol,
    ring: tuple[int, ...],
    attachment_idx: int,
    parent_idx: int,
) -> tuple[int, str, frozenset[int]] | None:
    ring_set = set(ring)
    substituents = []
    for atom_idx in ring:
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set or (atom_idx == attachment_idx and nidx == parent_idx):
                continue
            if nbr.GetAtomicNum() <= 1:
                continue
            candidates = simple_external_prefix_candidates(mol, nidx, atom_idx)
            if len(candidates) != 1:
                return None
            candidate = candidates[0]
            if candidate.text not in _BENZYL_SIMPLE_SUBSTITUENTS:
                return None
            substituents.append((atom_idx, candidate))
    if len(substituents) != 1:
        return None
    atom_idx, candidate = substituents[0]
    locant = _ring_locant(mol, ring_set, attachment_idx, atom_idx)
    if locant is None or locant <= 1:
        return None
    return locant, candidate.text, candidate.atoms


def _ring_locant(mol, ring_set: set[int], start_idx: int, target_idx: int) -> int | None:
    seen = {start_idx}
    frontier = [(start_idx, 0)]
    while frontier:
        atom_idx, distance = frontier.pop(0)
        if atom_idx == target_idx:
            return distance + 1
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in ring_set or nidx in seen:
                continue
            seen.add(nidx)
            frontier.append((nidx, distance + 1))
    return None


def _atom_has_assigned_stereo(atom) -> bool:
    return str(atom.GetChiralTag()) != "CHI_UNSPECIFIED"


def _simple_oxygen_prefix(mol, atom, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if atom.IsInRing() or atom.GetIsotope() or _atom_has_isotopic_hydrogen(atom):
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if order == 2.0 and atom.GetFormalCharge() == 0 and not heavy:
        return _prefix_candidate("oxo", {idx}, idx, parent_idx, "terminal-oxygen")
    if order != 1.0:
        return None
    if atom.GetFormalCharge() < 0 and not heavy:
        return _prefix_candidate("oxido", {idx}, idx, parent_idx, "terminal-oxygen")
    if atom.GetFormalCharge() != 0:
        return None
    if not heavy:
        return _prefix_candidate("hydroxy", {idx}, idx, parent_idx, "terminal-oxygen")
    if len(heavy) == 1:
        return (
            _simple_acetyloxy_prefix(mol, heavy[0], idx, parent_idx)
            or _simple_linear_alkoxy_prefix(mol, heavy[0], idx, parent_idx)
            or _simple_retained_alkoxy_prefix(mol, heavy[0], idx, parent_idx)
            or _simple_phenoxy_prefix(mol, heavy[0], idx, parent_idx)
        )
    return None


def _simple_acetyloxy_prefix(
    mol,
    carbonyl_idx: int,
    oxygen_idx: int,
    parent_idx: int,
) -> PrefixCandidate | None:
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0 or carbonyl.IsInRing() or carbonyl.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(carbonyl_idx, oxygen_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    double_oxygens = []
    methyl_ligands = []
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == oxygen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 2.0:
            if nbr.GetFormalCharge() != 0 or nbr.GetIsotope():
                return None
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "C" and bond.GetBondTypeAsDouble() == 1.0:
            if not _is_terminal_methyl(mol, nidx, carbonyl_idx):
                return None
            methyl_ligands.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 1 or len(methyl_ligands) != 1:
        return None
    return _prefix_candidate(
        "acetyloxy",
        {oxygen_idx, carbonyl_idx, double_oxygens[0], methyl_ligands[0]},
        oxygen_idx,
        parent_idx,
        "acetyloxy",
    )


def _simple_linear_alkoxy_prefix(
    mol,
    carbon_idx: int,
    oxygen_idx: int,
    parent_idx: int,
) -> PrefixCandidate | None:
    names = {1: "methoxy", 2: "ethoxy", 3: "propoxy", 4: "butoxy"}
    chain = []
    previous = oxygen_idx
    current = carbon_idx
    while len(chain) < 4:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            return None
        if atom.GetIsotope() or _atom_has_isotopic_hydrogen(atom):
            return None
        bond = mol.GetBondBetweenAtoms(previous, current)
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None
        chain.append(current)
        heavy = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetIdx() != previous and nbr.GetAtomicNum() > 1
        ]
        if not heavy:
            text = names[len(chain)]
            return _prefix_candidate(text, {oxygen_idx, *chain}, oxygen_idx, parent_idx, "linear-alkoxy")
        if len(heavy) != 1:
            return None
        next_idx = heavy[0]
        if mol.GetAtomWithIdx(next_idx).GetSymbol() != "C":
            return None
        previous, current = current, next_idx
    return None


def _simple_retained_alkoxy_prefix(
    mol,
    carbon_idx: int,
    oxygen_idx: int,
    parent_idx: int,
) -> PrefixCandidate | None:
    candidates = simple_external_prefix_candidates(mol, carbon_idx, oxygen_idx)
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    retained = {
        "benzyl": "benzyloxy",
        "difluoromethyl": "difluoromethoxy",
        "isobutyl": "isobutoxy",
        "isopropyl": "isopropoxy",
        "trifluoromethyl": "trifluoromethoxy",
    }.get(candidate.text)
    if retained is None:
        return None
    return _prefix_candidate(retained, {oxygen_idx, *candidate.atoms}, oxygen_idx, parent_idx, "retained-alkoxy")


def _simple_phenoxy_prefix(
    mol,
    ring_idx: int,
    oxygen_idx: int,
    parent_idx: int,
) -> PrefixCandidate | None:
    ring = _unsubstituted_phenyl_ring(mol, ring_idx, oxygen_idx)
    if ring is None:
        return None
    return _prefix_candidate("phenoxy", {oxygen_idx, *ring}, oxygen_idx, parent_idx, "phenoxy")


def _simple_nitrogen_prefix(mol, atom, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope() or _atom_has_isotopic_hydrogen(atom):
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if order == 1.0:
        if not heavy:
            return _prefix_candidate("amino", {idx}, idx, parent_idx, "terminal-nitrogen")
        if len(heavy) in {1, 2} and all(_is_terminal_methyl(mol, nbr_idx, idx) for nbr_idx in heavy):
            text = "methylamino" if len(heavy) == 1 else "dimethylamino"
            return _prefix_candidate(text, {idx, *heavy}, idx, parent_idx, "methylamino")
    if order == 2.0 and not heavy:
        return _prefix_candidate("imino", {idx}, idx, parent_idx, "terminal-nitrogen")
    return None


def _prefix_candidate(
    text: str,
    atoms: set[int],
    attachment_atom: int,
    parent_atom: int,
    kind: str,
) -> PrefixCandidate:
    return PrefixCandidate(
        text=text,
        atoms=frozenset(atoms),
        attachment_atom=attachment_atom,
        parent_atom=parent_atom,
        kind=kind,
        source=_SIMPLE_PREFIX_SOURCE,
        estimated_length=len(text),
    )


def _record_structured_prefix_selected(mol, candidate: PrefixCandidate) -> None:
    telemetry = mol_cache_get(mol, "candidate_telemetry")
    if telemetry is None:
        return
    telemetry.branch_structured_prefix_selected[_structured_prefix_key(candidate)] += 1


def _structured_prefix_key(candidate: PrefixCandidate) -> str:
    return f"{candidate.source}:{candidate.kind}"


def _atom_has_isotopic_hydrogen(atom) -> bool:
    return any(nbr.GetAtomicNum() == 1 and nbr.GetIsotope() for nbr in atom.GetNeighbors())
