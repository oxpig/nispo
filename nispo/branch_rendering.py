from __future__ import annotations

from dataclasses import dataclass

from .candidate_facts import PrefixCandidate
from .branch.functional_groups import _formyl_prefix_allowed
from .molecule_cache import mol_cache_dict, mol_cache_get
from .simple_prefixes import simple_external_prefix_candidates


@dataclass(frozen=True, slots=True)
class BranchRenderContext:
    """Shared context for graph-backed branch prefix renderers."""

    mol: object
    Chem: object | None = None
    telemetry: object | None = None
    active_rule_family: str = ""


def branch_render_context(mol, Chem=None) -> BranchRenderContext:
    telemetry = mol_cache_get(mol, "candidate_telemetry")
    active_rule_family = telemetry.active_rule_family if telemetry is not None else ""
    return BranchRenderContext(
        mol=mol,
        Chem=Chem,
        telemetry=telemetry,
        active_rule_family=active_rule_family,
    )


def render_branch_candidates(
    ctx: BranchRenderContext,
    idx: int,
    parent_idx: int,
) -> tuple[PrefixCandidate, ...]:
    """Return structured branch-prefix facts before legacy string rendering."""

    cache = mol_cache_dict(ctx.mol, "branch_render_candidate_cache")
    key = (idx, parent_idx)
    candidates = cache.get(key)
    if candidates is None:
        candidates = simple_external_prefix_candidates(ctx.mol, idx, parent_idx)
        if not candidates:
            candidate = _local_functional_prefix_candidate(ctx.mol, idx, parent_idx)
            candidates = (candidate,) if candidate is not None else ()
        cache[key] = candidates
    if ctx.telemetry is not None:
        if candidates:
            for candidate in candidates:
                ctx.telemetry.branch_structured_prefix_candidates[_structured_prefix_key(candidate)] += 1
        else:
            ctx.telemetry.branch_structured_prefix_misses["simple_external_prefix"] += 1
    return candidates


def select_branch_prefix_text(
    ctx: BranchRenderContext,
    candidates: tuple[PrefixCandidate, ...],
) -> str:
    if not candidates:
        return ""
    candidate = candidates[0]
    if ctx.telemetry is not None:
        ctx.telemetry.branch_structured_prefix_selected[_structured_prefix_key(candidate)] += 1
    return candidate.text


def record_branch_legacy_prefix_call(ctx: BranchRenderContext) -> None:
    if ctx.telemetry is None:
        return
    ctx.telemetry.branch_legacy_prefix_calls += 1
    rule_family = ctx.active_rule_family or "unknown"
    ctx.telemetry.branch_legacy_prefix_calls_by_rule[rule_family] += 1


def _structured_prefix_key(candidate: PrefixCandidate) -> str:
    return f"{candidate.source}:{candidate.kind}"


def _local_functional_prefix_candidate(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    symbol = atom.GetSymbol()
    if symbol == "H":
        return _isotopic_hydrogen_prefix(atom, idx, parent_idx)
    if atom.IsInRing():
        return None
    if symbol == "C":
        return (
            _cyano_prefix(mol, idx, parent_idx)
            or _formyl_prefix(mol, idx, parent_idx)
            or _carboxy_prefix(mol, idx, parent_idx)
        )
    if symbol == "N":
        return _nitro_prefix(mol, idx, parent_idx)
    return None


def _isotopic_hydrogen_prefix(atom, idx: int, parent_idx: int) -> PrefixCandidate | None:
    if atom.GetIsotope() == 2:
        return _prefix_candidate("deuterio", {idx}, idx, parent_idx, "isotopic-hydrogen")
    if atom.GetIsotope() == 3:
        return _prefix_candidate("tritio", {idx}, idx, parent_idx, "isotopic-hydrogen")
    return None


def _cyano_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetFormalCharge() != 0 or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    nitrile_nitrogens = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "N"
        and nbr.GetFormalCharge() == 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondTypeAsDouble() == 3.0
    ]
    if len(nitrile_nitrogens) != 1:
        return None
    if any(
        nbr.GetIdx() not in {parent_idx, nitrile_nitrogens[0]} and nbr.GetAtomicNum() > 1
        for nbr in atom.GetNeighbors()
    ):
        return None
    return _prefix_candidate("cyano", {idx, nitrile_nitrogens[0]}, idx, parent_idx, "cyano")


def _formyl_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetFormalCharge() != 0 or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    double_oxygens = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 2.0:
            double_oxygens.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 1 or not _formyl_prefix_allowed(mol, idx, parent_idx):
        return None
    return _prefix_candidate("formyl", {idx, double_oxygens[0]}, idx, parent_idx, "formyl")


def _carboxy_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetFormalCharge() != 0 or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    double_oxygens = []
    single_oxygens = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 2.0:
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 1.0:
            single_oxygens.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 1 or len(single_oxygens) != 1:
        return None
    oxygen = mol.GetAtomWithIdx(single_oxygens[0])
    if oxygen.GetFormalCharge() != 0 or _atom_has_isotopic_hydrogen(oxygen):
        return None
    if any(nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1 for nbr in oxygen.GetNeighbors()):
        return None
    return _prefix_candidate("carboxy", {idx, double_oxygens[0], single_oxygens[0]}, idx, parent_idx, "carboxy")


def _nitro_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetFormalCharge() <= 0 or atom.GetIsotope():
        return None
    oxygens = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "O"]
    if len(oxygens) < 2:
        return None
    non_oxygen = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() != "O"]
    if len(non_oxygen) == 1:
        if non_oxygen[0] != parent_idx:
            return None
    elif non_oxygen:
        return None
    elif mol.GetAtomWithIdx(parent_idx).GetSymbol() != "O":
        return None
    return _prefix_candidate("nitro", {idx, *oxygens}, idx, parent_idx, "nitro")


def _atom_has_isotopic_hydrogen(atom) -> bool:
    return any(nbr.GetAtomicNum() == 1 and nbr.GetIsotope() for nbr in atom.GetNeighbors())


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
        source="branch_rendering",
        estimated_length=len(text),
    )
