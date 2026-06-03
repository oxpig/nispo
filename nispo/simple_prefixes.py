from __future__ import annotations

from .candidate_facts import PrefixCandidate
from .constants import HALO
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
            _simple_fluoromethyl_prefix(mol, idx, parent_idx, order)
            or _simple_branched_alkyl_prefix(mol, idx, parent_idx, order)
            or _simple_benzyl_prefix(mol, idx, parent_idx, order)
            or _simple_hydroxymethyl_prefix(mol, idx, parent_idx, order)
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
        candidate = _simple_nitrogen_prefix(atom, idx, parent_idx, order)
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
    if len(heavy) not in {2, 3}:
        return None
    if not all(_is_terminal_methyl(mol, nbr_idx, idx) for nbr_idx in heavy):
        return None
    text = "isopropyl" if len(heavy) == 2 else "t-butyl"
    return _prefix_candidate(text, {idx, *heavy}, idx, parent_idx, "branched-alkyl")


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
    ring = _unsubstituted_phenyl_ring(mol, ring_idx, idx)
    if ring is None:
        return None
    return _prefix_candidate("benzyl", {idx, *ring}, idx, parent_idx, "benzyl")


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


def _unsubstituted_phenyl_ring(mol, attachment_idx: int, parent_idx: int) -> tuple[int, ...] | None:
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
    for atom in atoms:
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set or (atom.GetIdx() == attachment_idx and nidx == parent_idx):
                continue
            if nbr.GetAtomicNum() > 1:
                return None
    return ring


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
        return _simple_linear_alkoxy_prefix(mol, heavy[0], idx, parent_idx)
    return None


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


def _simple_nitrogen_prefix(atom, idx: int, parent_idx: int, order: float) -> PrefixCandidate | None:
    if atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope() or _atom_has_isotopic_hydrogen(atom):
        return None
    if any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
        return None
    if order == 1.0:
        return _prefix_candidate("amino", {idx}, idx, parent_idx, "terminal-nitrogen")
    if order == 2.0:
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
