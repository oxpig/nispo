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
        return _simple_ring_prefix(mol, idx, parent_idx)
    if symbol == "C":
        return (
            _cyano_prefix(mol, idx, parent_idx)
            or _formyl_prefix(mol, idx, parent_idx)
            or _carboxylato_prefix(mol, idx, parent_idx)
            or _retained_acyl_prefix(mol, idx, parent_idx)
            or _carbamyl_prefix(mol, idx, parent_idx)
            or _retained_alkoxycarbonyl_prefix(mol, idx, parent_idx)
            or _carboxy_prefix(mol, idx, parent_idx)
        )
    if symbol == "N":
        return (
            _azido_prefix(mol, idx, parent_idx)
            or _nitro_prefix(mol, idx, parent_idx)
            or _bocamino_prefix(mol, idx, parent_idx)
        )
    if symbol == "S":
        return (
            _simple_sulfur_prefix(mol, idx, parent_idx)
            or _thioxo_prefix(mol, idx, parent_idx)
            or _mesyl_prefix(mol, idx, parent_idx)
        )
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


def _retained_acyl_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    double_oxygens = []
    carbon_ligands = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 2.0:
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "C" and bond.GetBondTypeAsDouble() == 1.0:
            carbon_ligands.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 1 or len(carbon_ligands) != 1:
        return None
    oxygen = mol.GetAtomWithIdx(double_oxygens[0])
    if oxygen.GetFormalCharge() != 0 or oxygen.GetIsotope():
        return None
    ligand_candidates = simple_external_prefix_candidates(mol, carbon_ligands[0], idx)
    if len(ligand_candidates) != 1:
        return None
    ligand = ligand_candidates[0]
    retained = {"methyl": "acetyl"}.get(ligand.text)
    if retained is None:
        return None
    return _prefix_candidate(
        retained,
        {idx, double_oxygens[0], *ligand.atoms},
        idx,
        parent_idx,
        "retained-acyl",
    )


def _carboxylato_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    double_oxygens = []
    oxido_oxygens = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 2.0:
            if nbr.GetFormalCharge() != 0 or nbr.GetIsotope():
                return None
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 1.0:
            if nbr.GetFormalCharge() >= 0 or nbr.GetIsotope():
                return None
            if any(extra.GetIdx() != idx and extra.GetAtomicNum() > 1 for extra in nbr.GetNeighbors()):
                return None
            oxido_oxygens.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 1 or len(oxido_oxygens) != 1:
        return None
    return _prefix_candidate(
        "carboxylato",
        {idx, double_oxygens[0], oxido_oxygens[0]},
        idx,
        parent_idx,
        "carboxylato",
    )


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


def _carbamyl_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    double_oxygens = []
    nitrogens = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 2.0:
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "N" and bond.GetBondTypeAsDouble() == 1.0:
            nitrogens.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 1 or len(nitrogens) != 1:
        return None
    oxygen = mol.GetAtomWithIdx(double_oxygens[0])
    nitrogen = mol.GetAtomWithIdx(nitrogens[0])
    if oxygen.GetFormalCharge() != 0 or oxygen.GetIsotope():
        return None
    if nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing() or nitrogen.GetIsotope():
        return None
    if any(nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1 for nbr in nitrogen.GetNeighbors()):
        return None
    return _prefix_candidate("carbamyl", {idx, double_oxygens[0], nitrogens[0]}, idx, parent_idx, "carbamyl")


def _retained_alkoxycarbonyl_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
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
    if oxygen.GetFormalCharge() != 0 or oxygen.IsInRing() or oxygen.GetIsotope():
        return None
    alkyl = _retained_alkoxy_group(mol, single_oxygens[0], idx)
    if alkyl is None:
        return None
    text, alkyl_atoms = alkyl
    names = {
        "methyl": "carbmethoxy",
        "ethyl": "carbethoxy",
        "benzyl": "cbz",
        "t-butyl": "boc",
    }
    name = names.get(text)
    if name is None:
        return None
    return _prefix_candidate(name, {idx, double_oxygens[0], single_oxygens[0], *alkyl_atoms}, idx, parent_idx, name)


def _retained_alkoxy_group(mol, oxygen_idx: int, carbonyl_idx: int) -> tuple[str, set[int]] | None:
    heavy = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(oxygen_idx).GetNeighbors()
        if nbr.GetIdx() != carbonyl_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    carbon_idx = heavy[0]
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing() or carbon.GetIsotope():
        return None
    if _atom_has_assigned_stereo(carbon) or _atom_has_isotopic_hydrogen(carbon):
        return None
    bond = mol.GetBondBetweenAtoms(oxygen_idx, carbon_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    external = [
        nbr.GetIdx()
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != oxygen_idx and nbr.GetAtomicNum() > 1
    ]
    if not external:
        return ("methyl", {carbon_idx})
    if len(external) == 1 and _is_plain_terminal_methyl(mol, external[0], carbon_idx):
        return ("ethyl", {carbon_idx, external[0]})
    if len(external) == 1:
        ring = _simple_ring_prefix(mol, external[0], carbon_idx)
        if ring is not None and ring.text == "phenyl":
            return ("benzyl", {carbon_idx, *ring.atoms})
    if len(external) == 3 and all(_is_plain_terminal_methyl(mol, nbr_idx, carbon_idx) for nbr_idx in external):
        return ("t-butyl", {carbon_idx, *external})
    return None


def _is_plain_terminal_methyl(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return False
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return False
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return False
    return not any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors())


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


def _azido_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetIsotope() or atom.IsInRing():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    middle_candidates = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        if nbr.GetSymbol() == "N" and nbr.GetFormalCharge() > 0 and not nbr.GetIsotope():
            middle_candidates.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(middle_candidates) != 1:
        return None
    middle_idx = middle_candidates[0]
    terminal_candidates = []
    middle = mol.GetAtomWithIdx(middle_idx)
    for nbr in middle.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx:
            continue
        if nbr.GetSymbol() == "N" and nbr.GetFormalCharge() < 0 and not nbr.GetIsotope():
            terminal_candidates.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(terminal_candidates) != 1:
        return None
    terminal_idx = terminal_candidates[0]
    terminal = mol.GetAtomWithIdx(terminal_idx)
    if any(nbr.GetIdx() != middle_idx and nbr.GetAtomicNum() > 1 for nbr in terminal.GetNeighbors()):
        return None
    return _prefix_candidate("azido", {idx, middle_idx, terminal_idx}, idx, parent_idx, "azido")


def _bocamino_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    parent = mol.GetAtomWithIdx(parent_idx)
    if parent.GetSymbol() == "C" and parent.IsInRing() and not parent.GetIsAromatic():
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and not (nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0)
    ]
    if len(heavy) != 1:
        return None
    ligand = _retained_alkoxycarbonyl_prefix(mol, heavy[0], idx)
    if ligand is None or ligand.text != "boc":
        return None
    if not _bocamino_prefix_allowed(mol, idx, parent_idx):
        return None
    return _prefix_candidate("bocamino", {idx, *ligand.atoms}, idx, parent_idx, "bocamino")


def _bocamino_prefix_allowed(mol, amino_idx: int, parent_idx: int) -> bool:
    parent = mol.GetAtomWithIdx(parent_idx)
    if not parent.IsInRing() or mol.GetRingInfo().NumRings() < 2:
        return True
    for atom in mol.GetAtoms():
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
                and bond.GetBondTypeAsDouble() == 2.0
            ):
                has_exocyclic_oxo = True
            elif nbr.GetSymbol() == "N" and nbr.IsInRing() and nidx != amino_idx:
                has_ring_nitrogen = True
        if has_exocyclic_oxo and has_ring_nitrogen:
            return False
    return True


def _mesyl_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "S" or atom.GetFormalCharge() != 0 or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    double_oxygens = []
    methyl_ligands = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 2.0:
            if nbr.GetFormalCharge() != 0 or nbr.GetIsotope():
                return None
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "C" and bond.GetBondTypeAsDouble() == 1.0:
            if not _is_plain_terminal_methyl(mol, nidx, idx):
                return None
            methyl_ligands.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 2 or len(methyl_ligands) != 1:
        return None
    return _prefix_candidate("mesyl", {idx, *double_oxygens, methyl_ligands[0]}, idx, parent_idx, "mesyl")


def _simple_sulfur_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "S" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if not heavy:
        return _prefix_candidate("sulfanyl", {idx}, idx, parent_idx, "simple-sulfur")
    if len(heavy) != 1:
        return None
    methyl_idx = heavy[0]
    if not _is_plain_terminal_methyl(mol, methyl_idx, idx):
        return None
    return _prefix_candidate("methylsulfanyl", {idx, methyl_idx}, idx, parent_idx, "simple-sulfur")


def _thioxo_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "S" or atom.GetFormalCharge() != 0 or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 2.0:
        return None
    if any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors()):
        return None
    return _prefix_candidate("thioxo", {idx}, idx, parent_idx, "thioxo")


def _simple_ring_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings() if idx in ring and parent_idx not in ring]
    if len(rings) != 1:
        return None
    ring = rings[0]
    ring_set = set(ring)
    atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
    if any(
        atom.GetFormalCharge() != 0
        or atom.GetIsotope()
        or _atom_has_assigned_stereo(atom)
        or _atom_has_isotopic_hydrogen(atom)
        for atom in atoms
    ):
        return None
    if len(ring) == 6 and all(atom.GetIsAromatic() for atom in atoms):
        if all(atom.GetSymbol() == "C" for atom in atoms):
            return _simple_phenyl_prefix(mol, ring, idx, parent_idx)
        pyridyl = _simple_pyridyl_prefix(mol, ring, idx, parent_idx)
        if pyridyl is not None:
            return pyridyl
    if any(atom.GetSymbol() != "C" for atom in atoms):
        return _simple_saturated_heterocycle_prefix(mol, ring, idx, parent_idx)
    if not _ring_has_only_parent_external(mol, ring_set, idx, parent_idx):
        return None
    if any(atom.GetIsAromatic() for atom in atoms):
        return None
    if not _ring_bonds_are_single(mol, ring):
        return None
    names = {
        3: "cyclopropyl",
        4: "cyclobutyl",
        5: "cyclopentyl",
        6: "cyclohexyl",
    }
    text = names.get(len(ring))
    if text is None:
        return None
    return _prefix_candidate(text, ring_set, idx, parent_idx, "simple-cycloalkyl-ring")


def _simple_saturated_heterocycle_prefix(
    mol,
    ring: tuple[int, ...],
    idx: int,
    parent_idx: int,
) -> PrefixCandidate | None:
    ring_set = set(ring)
    atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
    if any(atom.GetIsAromatic() for atom in atoms):
        return None
    if not _ring_bonds_are_single(mol, ring):
        return None
    if not _ring_has_only_parent_external(mol, ring_set, idx, parent_idx):
        return None
    counts = _ring_symbol_counts(mol, ring)
    text = None
    if len(ring) == 6 and counts == {"C": 4, "N": 1, "O": 1}:
        text = _morpholine_prefix(mol, ring_set, idx)
    elif len(ring) == 6 and counts == {"C": 4, "N": 2}:
        text = _piperazine_prefix(mol, ring_set, idx)
    elif len(ring) == 6 and counts == {"C": 5, "N": 1}:
        text = _piperidine_prefix(mol, ring_set, idx)
    elif len(ring) == 5 and counts == {"C": 4, "N": 1}:
        text = "pyrrolidino" if mol.GetAtomWithIdx(idx).GetSymbol() == "N" else None
    elif len(ring) in {4, 5, 6} and counts == {"C": len(ring) - 1, "O": 1}:
        text = _simple_oxa_ring_prefix(mol, ring_set, idx, len(ring))
    if text is None:
        return None
    return _prefix_candidate(text, ring_set, idx, parent_idx, "simple-saturated-heterocycle")


def _ring_symbol_counts(mol, ring: tuple[int, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for atom_idx in ring:
        symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _ring_has_only_parent_external(mol, ring_set: set[int], idx: int, parent_idx: int) -> bool:
    for atom_idx in ring_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set or (atom_idx == idx and nidx == parent_idx):
                continue
            if nbr.GetAtomicNum() > 1:
                return False
    return True


def _morpholine_prefix(mol, ring_set: set[int], idx: int) -> str | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N":
        return None
    oxygens = _ring_atoms_by_symbol(mol, ring_set, "O")
    if len(oxygens) != 1:
        return None
    distance = _ring_distance(mol, ring_set, oxygens[0], idx)
    return "morpholino" if distance == 3 else None


def _piperazine_prefix(mol, ring_set: set[int], idx: int) -> str | None:
    if mol.GetAtomWithIdx(idx).GetSymbol() != "N":
        return None
    nitrogens = _ring_atoms_by_symbol(mol, ring_set, "N")
    if len(nitrogens) != 2:
        return None
    other = nitrogens[0] if idx == nitrogens[1] else nitrogens[1]
    distance = _ring_distance(mol, ring_set, idx, other)
    return "piperazino" if distance == 3 else None


def _piperidine_prefix(mol, ring_set: set[int], idx: int) -> str | None:
    nitrogens = _ring_atoms_by_symbol(mol, ring_set, "N")
    if len(nitrogens) != 1:
        return None
    if idx == nitrogens[0]:
        return "piperidino"
    distance = _ring_distance(mol, ring_set, nitrogens[0], idx)
    if distance is None:
        return None
    locant = distance + 1
    return f"{locant}-piperidyl" if locant in {2, 3, 4} else None


def _simple_oxa_ring_prefix(mol, ring_set: set[int], idx: int, size: int) -> str | None:
    oxygens = _ring_atoms_by_symbol(mol, ring_set, "O")
    if len(oxygens) != 1 or idx == oxygens[0]:
        return None
    distance = _ring_distance(mol, ring_set, oxygens[0], idx)
    if distance is None:
        return None
    locant = distance + 1
    stems = {4: "oxetan", 5: "oxolan", 6: "oxan"}
    stem = stems.get(size)
    return f"{stem}-{locant}-yl" if stem is not None and locant >= 2 else None


def _ring_atoms_by_symbol(mol, ring_set: set[int], symbol: str) -> list[int]:
    return [atom_idx for atom_idx in ring_set if mol.GetAtomWithIdx(atom_idx).GetSymbol() == symbol]


def _simple_pyridyl_prefix(mol, ring: tuple[int, ...], idx: int, parent_idx: int) -> PrefixCandidate | None:
    ring_set = set(ring)
    nitrogen_indices = []
    for atom_idx in ring:
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol == "N":
            nitrogen_indices.append(atom_idx)
        elif symbol != "C":
            return None
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set or (atom_idx == idx and nidx == parent_idx):
                continue
            if nbr.GetAtomicNum() > 1:
                return None
    if len(nitrogen_indices) != 1 or idx == nitrogen_indices[0]:
        return None
    distance = _ring_distance(mol, ring_set, nitrogen_indices[0], idx)
    if distance is None:
        return None
    locant = distance + 1
    if locant not in {2, 3, 4}:
        return None
    return _prefix_candidate(f"{locant}-pyridyl", ring_set, idx, parent_idx, "simple-pyridyl-ring")


def _simple_phenyl_prefix(mol, ring: tuple[int, ...], idx: int, parent_idx: int) -> PrefixCandidate | None:
    ring_set = set(ring)
    substituents = []
    for atom_idx in ring:
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_set or (atom_idx == idx and nidx == parent_idx):
                continue
            if nbr.GetAtomicNum() <= 1:
                continue
            candidates = _simple_phenyl_substituent_candidates(mol, nidx, atom_idx)
            if len(candidates) != 1:
                return None
            candidate = candidates[0]
            if candidate.text not in _PHENYL_SIMPLE_SUBSTITUENTS:
                return None
            substituents.append((atom_idx, candidate))
    if not substituents:
        return _prefix_candidate("phenyl", ring_set, idx, parent_idx, "simple-aryl-ring")
    if len(substituents) != 1:
        return None
    atom_idx, candidate = substituents[0]
    locant = _phenyl_substituent_locant(mol, ring_set, idx, atom_idx)
    if locant is None or locant <= 1:
        return None
    return _prefix_candidate(
        f"{locant}-{candidate.text}phenyl",
        ring_set | set(candidate.atoms),
        idx,
        parent_idx,
        "substituted-aryl-ring",
    )


def _simple_phenyl_substituent_candidates(mol, idx: int, parent_idx: int) -> tuple[PrefixCandidate, ...]:
    candidates = simple_external_prefix_candidates(mol, idx, parent_idx)
    if candidates:
        return candidates
    if mol.GetAtomWithIdx(idx).IsInRing():
        return ()
    candidate = _local_functional_prefix_candidate(mol, idx, parent_idx)
    return (candidate,) if candidate is not None else ()


def _ring_distance(mol, ring_set: set[int], start_idx: int, target_idx: int) -> int | None:
    seen = {start_idx}
    frontier = [(start_idx, 0)]
    while frontier:
        atom_idx, distance = frontier.pop(0)
        if atom_idx == target_idx:
            return distance
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in ring_set or nidx in seen:
                continue
            seen.add(nidx)
            frontier.append((nidx, distance + 1))
    return None


_PHENYL_SIMPLE_SUBSTITUENTS = {
    "amino",
    "bromo",
    "carboxy",
    "chloro",
    "cyano",
    "fluoro",
    "hydroxy",
    "iodo",
    "methoxy",
    "methyl",
    "nitro",
    "trifluoromethyl",
}


def _phenyl_substituent_locant(mol, ring_set: set[int], attachment_idx: int, substituent_atom_idx: int) -> int | None:
    distance = _ring_distance(mol, ring_set, attachment_idx, substituent_atom_idx)
    return None if distance is None else distance + 1


def _ring_bonds_are_single(mol, ring: tuple[int, ...]) -> bool:
    ring_set = set(ring)
    for atom_idx in ring:
        atom = mol.GetAtomWithIdx(atom_idx)
        ring_neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() in ring_set]
        if len(ring_neighbors) != 2:
            return False
        for nbr_idx in ring_neighbors:
            bond = mol.GetBondBetweenAtoms(atom_idx, nbr_idx)
            if bond is None or bond.GetBondTypeAsDouble() != 1.0:
                return False
    return True


def _atom_has_assigned_stereo(atom) -> bool:
    return str(atom.GetChiralTag()) != "CHI_UNSPECIFIED"


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
